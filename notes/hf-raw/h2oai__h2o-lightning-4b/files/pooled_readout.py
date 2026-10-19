"""The pooled readout: label log-probabilities from the pooling runner's final hidden state (two-way attention).

Two-way (bidirectional) reading runs vLLM's POOLING runner, whose embeddings endpoint returns the final hidden
state at the last prompt position (after the final norm, before lm_head; `--pooler-config '{"task": "embed",
"seq_pooling_type": "LAST", "use_activation": false}'`) and nothing else: no logits, no log-probabilities. The shim does the lm_head's last
step itself, for the label tokens only:

    logit[id]    = h . lm_head[id]                     (softcapped when the model has a final logit softcap)
    log P(label) = logsumexp(logit over the label's ids) - logsumexp(logit over every asked label's ids)

That is the softmax restricted to the asked labels' tokens, which is what the one-way path reads from vLLM's
`logprob_token_ids` too (the full-vocabulary normaliser cancels). The lm_head rows come from the LABEL-ROW FILE
shipped beside serve_config.json (`label_rows.lightning`: the safetensors FORMAT under a name vLLM's weight loader
does not glob, since `*.safetensors` beside model.safetensors would be read as a weight shard): the fp32 tensor `W`
[n_tokens, hidden] and `token_ids` int64 [n_tokens] (the flattened label groups in the contract's label order), and
in the metadata the hidden size, the softcap, the label order, the groups, the rows' sha256 and the weights' sha256.
This module reads and writes that format itself (8-byte little-endian header length, JSON header, raw bytes) so the
shim needs neither torch nor safetensors; numpy is used when present.
"""
import base64
import json
import math
import struct

try:
    import numpy as np
except ImportError:  # optional: the pure-Python path is used without it
    np = None

FORMAT = "lightning-label-rows-1"
FLOOR = 1e-12


# --------------------------------------------------------------------------------------------------- label rows
class LabelRows:
    """The exported lm_head rows: `ids` (token ids, the row order), `rows` (n x hidden, fp32), `groups`
    [(label, [ids]), ...] in the trained label order, `softcap` (0 = none), `hidden`, and the file's metadata."""

    def __init__(self, ids, rows, groups, softcap, metadata):
        self.ids = [int(i) for i in ids]
        self.rows = rows
        self.groups = [(str(label), [int(i) for i in g]) for label, g in groups]
        self.softcap = float(softcap or 0.0)
        self.metadata = dict(metadata)
        self.hidden = int(metadata["hidden"])
        self.index = {i: k for k, i in enumerate(self.ids)}
        self.labels = [label for label, _ in self.groups]
        if len(self.index) != len(self.ids):
            raise ValueError("label rows: a token id appears twice")
        missing = [i for _, g in self.groups for i in g if i not in self.index]
        if missing:
            raise ValueError(f"label rows: no row for token ids {missing[:5]} of the label groups")
        seen = [i for _, g in self.groups for i in g]
        if len(set(seen)) != len(seen):
            raise ValueError("label rows: a token id belongs to two labels")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("label rows: the labels are not distinct")

    def group_for(self, label):
        return next(g for l, g in self.groups if l == label)


def _header(ids, hidden, groups, softcap, meta):
    m = {
        "format": FORMAT,
        "hidden": str(int(hidden)),
        "softcap": "none" if not softcap else repr(float(softcap)),
        "label_order": json.dumps([str(l) for l, _ in groups], ensure_ascii=False),
        "groups": json.dumps([[str(l), [int(i) for i in g]] for l, g in groups], ensure_ascii=False),
    }
    for k, v in (meta or {}).items():
        if v is not None:
            m[str(k)] = v if isinstance(v, str) else json.dumps(v)
    return m


def rows_sha256(rows):
    """sha256 of the fp32 rows as written (little-endian, row by row): the file's `lm_head_sha256`."""
    import hashlib
    h = hashlib.sha256()
    for r in rows:
        h.update(struct.pack(f"<{len(r)}f", *[float(x) for x in r]))
    return h.hexdigest()


def write_label_rows(path, rows, ids, groups, softcap, hidden=None, meta=None):
    """Write the label-row file: tensors `W` fp32 [n, hidden] (rows in the order of `ids`) and `token_ids` int64
    [n]; metadata as _header plus `lm_head_sha256` of the rows (and whatever `meta` adds)."""
    rows = [[float(x) for x in r] for r in rows]
    if len(rows) != len(ids):
        raise ValueError(f"label rows: {len(rows)} rows for {len(ids)} token ids")
    hidden = int(hidden if hidden is not None else (len(rows[0]) if rows else 0))
    if any(len(r) != hidden for r in rows):
        raise ValueError("label rows: the rows do not all have the hidden size")
    LabelRows(ids, rows, groups, softcap, {"hidden": hidden})  # the same checks the reader makes
    n_w = len(rows) * hidden * 4
    n_i = len(ids) * 8
    header = {
        "__metadata__": {**_header(ids, hidden, groups, softcap, meta), "lm_head_sha256": rows_sha256(rows)},
        "W": {"dtype": "F32", "shape": [len(rows), hidden], "data_offsets": [0, n_w]},
        "token_ids": {"dtype": "I64", "shape": [len(ids)], "data_offsets": [n_w, n_w + n_i]},
    }
    hb = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    hb += b" " * (-len(hb) % 8)  # safetensors pads the header to 8 bytes
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for r in rows:
            f.write(struct.pack(f"<{hidden}f", *r))
        f.write(struct.pack(f"<{len(ids)}q", *[int(i) for i in ids]))


def read_label_rows(path):
    """Read the label-row file written by write_label_rows (or by safetensors with the same tensors and metadata)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n).decode("utf-8"))
        meta = header.get("__metadata__") or {}
        if meta.get("format") != FORMAT:
            raise ValueError(f"label rows: {path} is not a {FORMAT} file (format {meta.get('format')!r})")
        w, t = header.get("W"), header.get("token_ids")
        if not isinstance(w, dict) or w.get("dtype") != "F32" or len(w.get("shape") or []) != 2:
            raise ValueError(f"label rows: no fp32 2-d tensor 'W' in {path}")
        if not isinstance(t, dict) or t.get("dtype") != "I64" or len(t.get("shape") or []) != 1:
            raise ValueError(f"label rows: no int64 1-d tensor 'token_ids' in {path}")
        rows_n, hidden = (int(x) for x in w["shape"])
        a, b = (int(x) for x in w["data_offsets"])
        c, d = (int(x) for x in t["data_offsets"])
        if b - a != rows_n * hidden * 4 or d - c != int(t["shape"][0]) * 8 or int(t["shape"][0]) != rows_n:
            raise ValueError("label rows: the tensors' byte ranges do not match their shapes")
        f.seek(8 + n + a)
        raw = f.read(b - a)
        f.seek(8 + n + c)
        raw_ids = f.read(d - c)
    if len(raw) != b - a or len(raw_ids) != d - c:
        raise ValueError("label rows: the file is truncated")
    ids = list(struct.unpack(f"<{rows_n}q", raw_ids))
    if np is not None:
        rows = np.frombuffer(raw, dtype="<f4").reshape(rows_n, hidden).astype(np.float64)
    else:
        flat = struct.unpack(f"<{rows_n * hidden}f", raw)
        rows = [list(flat[k * hidden : (k + 1) * hidden]) for k in range(rows_n)]
    if int(meta["hidden"]) != hidden:
        raise ValueError("label rows: the hidden metadata does not match the tensor")
    sc = meta.get("softcap", "none")
    softcap = 0.0 if sc in ("none", "", None) else float(sc)
    if "groups" in meta:
        groups = json.loads(meta["groups"])
    else:  # one token per label, in label order
        labels = json.loads(meta["label_order"])
        if len(labels) != rows_n:
            raise ValueError("label rows: no groups metadata and the label order does not match the ids one to one")
        groups = [[l, [i]] for l, i in zip(labels, ids)]
    return LabelRows(ids, rows, groups, softcap, meta)


# ------------------------------------------------------------------------------------------------------ readout
def softcap(x, cap):
    """A final logit softcap (some models have one); identity for cap 0 / None."""
    return cap * math.tanh(x / cap) if cap else x


def decode_pooled(data, hidden=None):
    """The pooled vector from an embeddings / pooling response item: a float list, or base64 fp32 little-endian
    (encoding_format base64, embed_dtype float32). Checks the length against `hidden` when given."""
    if isinstance(data, str):
        raw = base64.b64decode(data, validate=True)
        if len(raw) % 4:
            raise ValueError("pooled vector: the base64 payload is not whole fp32 values")
        h = (
            np.frombuffer(raw, dtype="<f4").astype(np.float64)
            if np is not None
            else list(struct.unpack(f"<{len(raw) // 4}f", raw))
        )
    elif isinstance(data, list) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in data):
        h = np.asarray(data, dtype=np.float64) if np is not None else [float(x) for x in data]
    else:
        raise ValueError("pooled vector: expected a float list or a base64 string")
    if hidden is not None and len(h) != hidden:
        raise ValueError(f"pooled vector has {len(h)} values; the label rows have hidden size {hidden}")
    if not all(math.isfinite(float(x)) for x in h):
        raise ValueError("pooled vector holds a non-finite value")
    return h


def l2_norm(h):
    return math.sqrt(sum(float(x) * float(x) for x in h))


def check_not_normalised(h, tol=1e-3):
    """An engine started with the activation on returns unit vectors and the logits are garbage: refuse."""
    if abs(l2_norm(h) - 1.0) < tol:
        raise ValueError("the pooled vector is L2-normalised (norm 1.0): the engine's pooler runs with "
                         "use_activation on; serve it with use_activation false")


def label_logits(h, rows, ids=None):
    """{token id: softcapped logit} for the label rows' ids (or the subset `ids`) at the pooled state h."""
    want = rows.ids if ids is None else [int(i) for i in ids]
    k = [rows.index[i] for i in want]
    if np is not None:
        hv = np.asarray(h, dtype=np.float64)
        raw = (np.asarray(rows.rows)[k] @ hv).tolist()
    else:
        raw = [sum(a * b for a, b in zip(rows.rows[j], h)) for j in k]
    return {i: softcap(float(x), rows.softcap) for i, x in zip(want, raw)}


def _logsumexp(v):
    m = max(v)
    return m + math.log(sum(math.exp(x - m) for x in v))


def group_logprobs(logits, groups):
    """log P(label) for the labels in `groups` [[label, [ids]], ...] from {id: logit}: per-label logsumexp over the
    label's ids, renormalised over these labels (the softmax restricted to their tokens), floored at FLOOR."""
    acc = []
    for _, ids in groups:
        v = [logits[i] for i in ids]
        if not v or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v):
            raise ValueError(f"an invalid logit among token ids {list(ids)[:5]}")
        acc.append(_logsumexp(v))
    z = _logsumexp(acc)
    return [max(a - z, math.log(FLOOR)) for a in acc]


def read_pooled(data, rows, groups):
    """One question's readout: (log P(label) in the order of `groups`, the pooled vector) from a response item's
    vector (`embedding` of /v1/embeddings, `data` of /pooling)."""
    h = decode_pooled(data, rows.hidden)
    ids = [i for _, g in groups for i in g]
    return group_logprobs(label_logits(h, rows, ids), groups), h


def probabilities(logprobs, temperature=1.0):
    """softmax(log-probabilities / T): the distribution the mode probe compares (T = 1)."""
    v = [x / float(temperature) for x in logprobs]
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]
