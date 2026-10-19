"""Vela 2.0 unified schema encoder: package-free inference (drop this file next to the checkpoint).

Dependencies: numpy, tokenizers; plus torch + transformers + safetensors for backend="torch" (modeling_vela2.py), or
onnxruntime for backend="onnx" (no torch needed).

Loading:
    torch   m = transformers.AutoModel.from_pretrained(repo_id_or_dir, trust_remote_code=True)  (modeling_vela2.Vela2Model:
            same methods as below, delegated to this engine on the model's device)
            or Vela2.from_pretrained(repo_id_or_dir, backend="torch", device="cuda")
    onnx    Vela2.from_pretrained(repo_id_or_dir, backend="onnx")  (no torch / transformers needed)
    A Hub repo id is fetched with huggingface_hub.snapshot_download (only the files the backend needs).

Three interfaces over one engine:

  SystemOne (TypeSafe POST /v1/systemone request / response; the primary interface; vela2_serve.py serves it over HTTP)
    m.system_one("I was charged twice.", {"billing": {"type": "noul", "instructions": "Is this about billing?"}})
    -> {"model", "answers": {"billing": {"type": "noul", "noul": 0.97}}, "usage": {...}}
    Choice / Noul / Score as in SystemOne; extension types "span" and "set" add "spans" / "sets" / "thresholds" to the
    response plus a Noul view of each in "answers"; a JSON-object state is mapped onto typed parts (SYSTEM_ONE_NOTES).
  GLiNER2-style
    m = Vela2.from_pretrained(path, backend="onnx")
    m.classify_text(text, ["positive", "negative"])                               -> {"label": "positive"}
    m.classify_text(text, {"sentiment": ["pos", "neg"], "topics": {"labels": [...], "multi_label": True}})
    m.extract_entities(text, ["person", "company"], threshold=None)              -> {"entities": {"person": [...]}}
    m.detect_hallucination(context, answer, question=None)                        -> unsupported spans of the answer
    m.score_relevance(query, document)                                            -> relevance in [0, 1] + distribution
  Kai/Jev-style JSON
    m.predict({"parts": [{"type": "user"|"context"|"answer", "text": ...}],
               "questions": [{"id", "type": "choice"|"set"|"score"|"span", "text",
                              "options" | "labels": [{"name", "description"}], "target_part": "user" | [...]}]})
    -> {"answers": [...], "meta": {...}}  (probabilities per option; spans with code-point char offsets)

Everything below reproduces the research scorer (score_schema_v2 + train_schema_generic eval assembly + schema_lib_r4
word units + span_decode_r4 decoder) without importing it: same sequence layout, same truncation order, same 8,192
window with 512 overlap for a labelled part that does not fit (choice logits averaged over windows, set logits max,
span word logits averaged), same calibrated probabilities (one temperature per question type) and dev thresholds.
Probabilities of a choice / score question are over the real options (the eval convention the temperatures were
fitted on); the [ABS] (abstain) probability is reported separately (softmax including [ABS]).
"""
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# =====================================================================================================================
# word units (schema_lib_r4._WORD, GLiNER2 WhitespaceTokenSplitter + CJK char level, r4 URL rule)
# =====================================================================================================================
_CJK = ("⺀-⿿　-〿぀-ヿ㄀-ㇿ㐀-䶿一-鿿ꀀ-꓏"
        "가-힯豈-﫿︰-﹏＀-￯")
_TRAIL = ".,;:!?)\\]}>\"'”’»"
_WORD = re.compile(rf"""(?:https?://|www\.)[^\s{_CJK}]+?(?=[{_TRAIL}]*(?![^\s{_CJK}]))
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{{2,}}
    |@[a-z0-9_]+
    |[{_CJK}]
    |[^\W{_CJK}]+(?:[-_][^\W{_CJK}]+)*
    |\S""", re.VERBOSE | re.IGNORECASE)


def split_words(text):
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def word_first_tokens(text, offs):
    """(word char offsets (W, 2), index of each word's first overlapping sub-word token (W,)) (schema_lib_r4)."""
    w = split_words(text)
    if not w or offs is None or len(offs) == 0:
        return np.zeros((0, 2), np.int32), np.zeros(0, np.int32)
    offs = np.asarray(offs)
    st, en = offs[:, 0], offs[:, 1]
    valid = en > st
    ends = np.where(valid, en, -1)
    wo, ft = [], []
    cummax_end = np.maximum.accumulate(ends)
    for a, b in w:
        t = int(np.searchsorted(cummax_end, a, side="right"))
        while t < len(offs) and not valid[t]:
            t += 1
        if t < len(offs) and st[t] < b:
            wo.append((a, b)); ft.append(t)
    return np.asarray(wo, np.int32).reshape(-1, 2), np.asarray(ft, np.int32)


# =====================================================================================================================
# span decoder (span_decode_r4.decode_v2; word readout -> extend=False)
# =====================================================================================================================
_UBODY = rf"[^\s\]\)\}}>「」『』\"'“”{_CJK}]"
_UNIT = re.compile(rf"(?:https?://|www\.){_UBODY}+?(?=[.,;:!?]*(?!{_UBODY}))"
                   rf"|(?:(?![{_CJK}])[^\W_]|[@._\-+])+"
                   rf"|\S")
_ALNUM = re.compile(r"[^\W_]")
EDGE = set("[](){}<>「」『』【】《》〈〉\"'“”‘’«»‹›")
_URL_START = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_CJK_RE = re.compile(rf"[{_CJK}]")
URL_TRAIL = set(".,;:!?)")


def _units(text):
    u = np.full(len(text) + 1, -1, dtype=np.int64)
    for k, m in enumerate(_UNIT.finditer(text)):
        u[m.start():m.end()] = k
    return u


def _first_nonspace(text, a, b):
    while a < b and text[a].isspace():
        a += 1
    return a


def _trim_edges(text, s, e):
    while s < e and (text[s].isspace() or text[s] in EDGE):
        s += 1
    while e > s and (text[e - 1].isspace() or text[e - 1] in EDGE):
        e -= 1
    return s, e


def _trim_url(text, s, e):
    if not _URL_START.match(text, s):
        return s, e
    m = _CJK_RE.search(text, s, e)
    if m:
        e = m.start()
    while e > s and (text[e - 1] in URL_TRAIL or text[e - 1] in EDGE or text[e - 1].isspace()):
        e -= 1
    return s, e


def _trim(text, s, e):
    s, e = _trim_edges(text, s, e)
    return _trim_url(text, s, e)


def decode_spans(probs, offs, names, text, thr, fill=True, extend=False, unit_vote=True):
    """span_decode_r4.decode_v2 with the scorer's word-readout arguments (extend=False, trim_edges=True, no bridge)."""
    probs = np.asarray(probs, dtype=np.float32); offs = np.asarray(offs)
    T = len(offs)
    if T == 0:
        return []
    valid = offs[:, 1] > offs[:, 0]
    k = probs.argmax(1); pk = probs[np.arange(T), k]
    lab = np.where(valid & (pk > thr), k, -1)
    u = _units(text)
    tu = np.full(T, -1, dtype=np.int64)
    alnum = np.zeros(T, dtype=bool)
    for t in range(T):
        if not valid[t]:
            continue
        a, b = int(offs[t, 0]), int(offs[t, 1])
        f = _first_nonspace(text, a, b)
        tu[t] = u[f] if f < b and f < len(text) else -1
        alnum[t] = bool(_ALNUM.search(text[a:b]))
    if unit_vote or fill or extend:
        t = 0
        while t < T:
            if tu[t] < 0:
                t += 1; continue
            t2 = t
            while t2 + 1 < T and (tu[t2 + 1] == tu[t] or not valid[t2 + 1]):
                t2 += 1
            idx = [i for i in range(t, t2 + 1) if valid[i]]
            L = [i for i in idx if lab[i] >= 0]
            if L:
                if unit_vote and len({int(lab[i]) for i in L}) > 1:
                    sc = {}
                    for i in L:
                        sc[int(lab[i])] = sc.get(int(lab[i]), 0.0) + float(pk[i])
                    w = max(sc, key=sc.get)
                    for i in L:
                        lab[i] = w
                if fill:
                    for c in set(int(lab[i]) for i in L):
                        pos = [i for i in L if lab[i] == c]
                        for i in idx:
                            if pos[0] < i < pos[-1] and lab[i] < 0:
                                lab[i] = c
                if extend:
                    for c in set(int(lab[i]) for i in idx if lab[i] >= 0):
                        pos = [idx.index(i) for i in idx if lab[i] == c]
                        if extend == "unit":
                            an = [j for j, i in enumerate(idx) if alnum[i]]
                            lo, hi = min([pos[0]] + an[:1]), max([pos[-1]] + an[-1:])
                            if all(lab[idx[j]] < 0 or lab[idx[j]] == c for j in range(lo, hi + 1)):
                                for j in range(lo, hi + 1):
                                    lab[idx[j]] = c
                        else:
                            j = pos[0] - 1
                            while j >= 0 and lab[idx[j]] < 0 and alnum[idx[j]]:
                                lab[idx[j]] = c; j -= 1
                            j = pos[-1] + 1
                            while j < len(idx) and lab[idx[j]] < 0 and alnum[idx[j]]:
                                lab[idx[j]] = c; j += 1
            t = t2 + 1
    spans, cur = [], None
    for t in range(T):
        if not valid[t]:
            continue
        c = int(lab[t])
        if cur is not None and c == cur[0]:
            cur[2] = int(offs[t, 1]); cur[3].append(t); continue
        if cur is not None:
            spans.append(cur)
        cur = [c, int(offs[t, 0]), int(offs[t, 1]), [t]] if c >= 0 else None
    if cur is not None:
        spans.append(cur)
    res = []
    for c, s, e, ts in spans:
        s, e = _trim(text, s, e)
        if e <= s:
            continue
        res.append(dict(start=s, end=e, label=names[c], probability=float(np.mean(probs[ts, c]))))
    return res


# =====================================================================================================================
# sequence assembly (schema_lib.assemble, eval path: cap_schema=False)
# =====================================================================================================================
def assemble(mk, bos, eos, questions, elabels, segments, max_len, protected, shrink, prompt_floor=64):
    qs = [dict(q, opts=list(q["opts"])) for q in questions]
    seg = [(r, list(t)) for r, t in segments]
    fixed = 2 + 1 + len(seg)
    elabels = list(elabels)
    tr = {"shrink_cut": 0, "protected_cut": 0}

    def over():
        return fixed + sum(q["len"] for q in qs) + sum(e["len"] for e in elabels) + sum(len(t) for _, t in seg) - max_len
    si = next((i for i, (r, _) in enumerate(seg) if r == shrink), None)
    if over() > 0 and si is not None:
        keep = max(min(len(seg[si][1]), prompt_floor), len(seg[si][1]) - over())
        tr["shrink_cut"] += len(seg[si][1]) - keep
        seg[si] = (seg[si][0], seg[si][1][:keep])
    if over() > 0 and si is not None:
        keep = max(1, len(seg[si][1]) - over())
        tr["shrink_cut"] += len(seg[si][1]) - keep
        seg[si] = (seg[si][0], seg[si][1][:keep])
    if over() > 0:
        for i, (r, t) in enumerate(seg):
            if r in protected and over() > 0:
                keep = max(1, len(t) - over())
                tr["protected_cut"] += len(t) - keep
                seg[i] = (r, t[:keep])
    if over() > 0:
        raise ValueError(f"schema alone does not fit max_length ({over()} tokens over)")
    ids = [bos]
    qrec, erec = [], []
    for q in qs:
        rec = dict(qid=q["qid"], type=q["type"], over=q["over"], q_pos=len(ids), opt_pos=[], opt_names=[], abs_pos=None)
        ids.append(mk["[Q]"]); ids.extend(q["qtoks"])
        for n, t in q["opts"]:
            rec["opt_pos"].append(len(ids)); rec["opt_names"].append(n)
            ids.append(mk["[O]"]); ids.extend(t)
        if q["abstain"]:
            rec["abs_pos"] = len(ids); ids.append(mk["[ABS]"])
        qrec.append(rec)
    for e in elabels:
        erec.append(dict(name=e["name"], pos=len(ids)))
        ids.append(mk["[E]"]); ids.extend(e["toks"])
    ids.append(mk["[SEP_SCHEMA]"])
    ranges = {}
    for r, t in seg:
        start = len(ids)
        ids.append(mk[f"[SEG_{r}]"]); ids.extend(t)
        ranges[r] = (start, len(ids), len(t))
    ids.append(eos)
    return dict(ids=ids, questions=qrec, elabels=erec, ranges=ranges, trunc=tr, seg_lens={r: len(t) for r, t in seg})


def _opt_text(name, desc):
    return name if not desc else f"{name}: {desc}"


def _softmax(x):
    x = np.asarray(x, dtype=np.float64); e = np.exp(x - x.max()); return e / e.sum()


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=np.float64)))


# =====================================================================================================================
# backends
# =====================================================================================================================
class _TorchBackend:
    def __init__(self, path, device="cpu", amp_bf16=False, attn_implementation="sdpa", model=None):
        import torch
        self.torch = torch
        if model is None:
            import sys
            sys.path.insert(0, str(path))
            try:  # try-wrapped so transformers' remote-code import check skips it (standalone path only)
                from modeling_vela2 import Vela2SchemaEncoder
            except ImportError:
                raise
            model = Vela2SchemaEncoder.from_pretrained(path, device=device, attn_implementation=attn_implementation)
        self.model = model  # a Vela2SchemaEncoder or a modeling_vela2.Vela2Model (same hidden / heads)
        self.amp_bf16 = amp_bf16

    def __call__(self, x, m, qi, oi, ui, ei):
        t = self.torch
        dev = next(self.model.parameters()).device
        with t.no_grad():
            X, M = t.from_numpy(x).to(dev), t.from_numpy(m).to(dev)
            if self.amp_bf16:
                with t.autocast("cuda", dtype=t.bfloat16):
                    h = self.model.hidden(X, M)
            else:
                h = self.model.hidden(X, M)
            lg, sp = self.model.heads(h, *(t.from_numpy(a).to(dev) for a in (qi, oi, ui, ei)))
        return lg.float().cpu().numpy(), sp.float().cpu().numpy()


class _OnnxBackend:
    def __init__(self, path, onnx_file="onnx/model.onnx", providers=None, threads=0):
        try:  # optional dependency; try-wrapped so transformers' remote-code import check skips it
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("the ONNX backend needs onnxruntime: pip install onnxruntime") from e
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(Path(path) / onnx_file), so, providers=providers or ["CPUExecutionProvider"])
        self.fp16 = "fp16" in onnx_file

    def __call__(self, x, m, qi, oi, ui, ei):
        lg, sp = self.sess.run(["opt_logits", "span_logits"], dict(input_ids=x, attention_mask=m, q_index=qi, opt_index=oi,
                                                                    unit_index=ui, ent_index=ei))
        return lg.astype(np.float32), sp.astype(np.float32)


# =====================================================================================================================
# engine
# =====================================================================================================================
class Vela2:
    PII_QUESTION = "Which spans are personal information?"

    def __init__(self, path, backend="torch", device="cpu", amp_bf16=False, onnx_file="onnx/model.onnx", providers=None,
                 threads=0, token_budget=None, batch_size=None, model=None, revision=None):
        """path: a release dir or a Hub repo id. model: an already loaded Vela2SchemaEncoder / Vela2Model to run
        (torch backend; `device` should then be the model's device)."""
        from tokenizers import Tokenizer
        if not Path(path).is_dir():
            from huggingface_hub import snapshot_download
            need = ["config.json", "calibration.json", "tokenizer.json"]
            if backend == "onnx":
                need.append(onnx_file)
            elif model is None:
                need += ["model.safetensors", "modeling_vela2.py", "configuration_vela2.py", "vela2_inference.py"]
            path = snapshot_download(str(path), revision=revision, allow_patterns=need)
        path = Path(path)
        self.path = path
        self.config = json.loads((path / "config.json").read_text())
        self.cal = json.loads((path / "calibration.json").read_text())
        self.mk = self.config["marker_ids"]
        self.bos, self.eos, self.pad = self.config["bos_token_id"], self.config["eos_token_id"], self.config["pad_token_id"]
        # text is encoded with the UNMODIFIED Vela tokenizer (markers removed), so "[Q]" typed by a user stays text
        tj = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
        tj["added_tokens"] = [t for t in tj["added_tokens"] if t["content"] not in self.mk]
        self.tok = Tokenizer.from_str(json.dumps(tj))
        self.tok.no_padding(); self.tok.no_truncation()
        self._tc = {}
        self.max_len = self.config["max_length"]
        self.overlap = self.config["window_overlap"]
        self.budget = token_budget or (32768 if (backend == "torch" and device != "cpu") else 16384)
        self.bs = batch_size or 32
        if backend == "torch":
            self.run = _TorchBackend(path, device=device, amp_bf16=amp_bf16, model=model)
        elif backend == "onnx":
            self.run = _OnnxBackend(path, onnx_file=onnx_file, providers=providers, threads=threads)
        else:
            raise ValueError(backend)
        self.stats = defaultdict(int)

    @classmethod
    def from_pretrained(cls, path, **kw):
        return cls(path, **kw)

    # ------------------------------------------------------------------ tokenisation
    def _ids(self, s):
        if s not in self._tc:
            self._tc[s] = self.tok.encode(s, add_special_tokens=False).ids
        return self._tc[s]

    def _ids_offs(self, s):
        e = self.tok.encode(s, add_special_tokens=False)
        return np.asarray(e.ids, np.int32), np.asarray(e.offsets, np.int32).reshape(-1, 2)

    # ------------------------------------------------------------------ generic contract -> internal row
    def _prep(self, row):
        """Generic-contract row {parts: [{role, text}], questions: [{id, type, over, text, options, levels, labels,
        abstain}]} -> tokenised internal row (same fields as the research scorer's pool rows)."""
        roles = [p["role"] for p in row["parts"]]
        if len(set(roles)) != len(roles):
            raise ValueError(f"one part per role, got {roles}")
        span_roles = {q["over"] for q in row["questions"] if q["type"] == "span"}
        if len([q for q in row["questions"] if q["type"] == "span"]) > 1:
            raise ValueError("one span question per row (predict() splits requests with several)")
        parts = []
        for p in row["parts"]:
            ids, offs = self._ids_offs(p["text"])
            words = word_first_tokens(p["text"], offs) if p["role"] in span_roles else None
            parts.append((p["role"], ids, offs if p["role"] in span_roles else None, words))
        for q in row["questions"]:
            for o in (q["over"] if isinstance(q["over"], list) else [q["over"]]):
                if o not in roles:
                    raise ValueError(f"question {q.get('id')} is over '{o}' but the row has parts {roles}")
        return dict(parts=parts, qs=row["questions"])

    def _build_question(self, q):
        typ = q["type"]; qtoks = self._ids(q["text"])
        if typ == "span":
            labels = q["labels"]
            return "span", [dict(name=n, toks=self._ids(_opt_text(n, labels[n])), len=1 + len(self._ids(_opt_text(n, labels[n]))))
                            for n in labels]
        opts = q.get("options") or {}
        if typ == "score":
            names = list(q["levels"]); abstain = False
        elif typ == "set":
            names = list(opts); abstain = False
        elif typ == "choice":
            names = list(opts); abstain = bool(q.get("abstain"))
        else:
            raise ValueError(typ)
        ot = [(n, self._ids(_opt_text(n, opts.get(n, "")))) for n in names]
        length = 1 + len(qtoks) + sum(1 + len(t) for _, t in ot) + (1 if abstain else 0)
        return "q", dict(qid=q["id"], type="choice" if typ == "score" else typ, qtype=typ, over=q["over"], qtoks=qtoks,
                         opts=ot, abstain=abstain, len=length)

    def _assemble(self, r):
        qs, el, span_over = [], [], None
        for q in r["qs"]:
            kind, x = self._build_question(q)
            if kind == "span":
                el, span_over = x, q["over"]
            else:
                qs.append(x)
        overs = set()
        for q in qs:
            overs |= set(q["over"] if isinstance(q["over"], list) else [q["over"]])
        if span_over:
            overs.add(span_over)
        roles = [p[0] for p in r["parts"]]
        shrink = next((x for x in roles if x not in overs), None)
        rec = assemble(self.mk, self.bos, self.eos, qs, el, [(p[0], p[1]) for p in r["parts"]], self.max_len, overs, shrink)
        R = rec["ranges"]
        for q, orig in zip(rec["questions"], qs):
            o = q["over"]
            q["pool"] = (R[o[0]][0], R[o[-1]][1]) if isinstance(o, list) else (R[o][0], R[o][1])
            q["qtype"] = orig["qtype"]
        if span_over:
            s, e, n = R[span_over]
            part = next(p for p in r["parts"] if p[0] == span_over)
            wo, ft = part[3]
            keep = ft < n
            rec.update(names=[x["name"] for x in rec["elabels"]], e_pos=[x["pos"] for x in rec["elabels"]],
                       w_offs=wo[keep], w_pos=(s + 1 + ft[keep]).astype(np.int64))
        return rec

    def _windows(self, r, rec):
        span_over = next((q["over"] for q in r["qs"] if q["type"] == "span"), None)
        overs = [q["over"] for q in r["qs"]]
        role = span_over or (overs[0] if all(o == overs[0] for o in overs) and isinstance(overs[0], str) else None)
        if role is None:
            self.stats["cut_not_windowed"] += 1
            return None
        part = next(p for p in r["parts"] if p[0] == role)
        N = len(part[1]); L = rec["seg_lens"][role]
        if L >= N:
            return None
        ov = min(self.overlap, L // 4); stride = max(1, L - ov)
        out = []
        for a in range(0, max(1, N - ov), stride):
            b = min(N, a + L)
            words = gidx = None
            if part[3] is not None:
                wo, ft = part[3]; sel = np.nonzero((ft >= a) & (ft < b))[0]
                words = (wo[sel], ft[sel] - a); gidx = sel
            np_ = (role, part[1][a:b], None if part[2] is None else part[2][a:b], words)
            out.append((dict(r, parts=[np_ if p[0] == role else p for p in r["parts"]]), (a, b), gidx))
            if b >= N:
                break
        self.stats["windowed_rows"] += 1; self.stats["windows"] += len(out)
        return out

    def _forward(self, recs):
        B = len(recs); W = max(len(r["ids"]) for r in recs)
        x = np.full((B, W), self.pad, np.int64); m = np.zeros((B, W), np.int64)
        qi, oi, ui, ei, qsz, spn = [], [], [], [], [], []
        for b, r in enumerate(recs):
            x[b, :len(r["ids"])] = r["ids"]; m[b, :len(r["ids"])] = 1
            for q in r["questions"]:
                pos = list(q["opt_pos"]) + ([q["abs_pos"]] if q["abs_pos"] is not None else [])
                own = len(qi); qi.append((b, q["q_pos"], q["pool"][0], q["pool"][1]))
                oi.extend((b, p, own) for p in pos); qsz.append(len(pos))
            if "w_pos" in r:
                k0, e0 = len(ui), len(ei)
                ui.extend((b, int(p)) for p in r["w_pos"]); ei.extend((b, int(p)) for p in r["e_pos"])
                spn.append((k0, len(ui), e0, len(ei)))
            else:
                spn.append(None)
        nq, no, nu, ne = len(qi), len(oi), len(ui), len(ei)
        # never feed an empty index tensor (a dummy row is added and dropped)
        qa = np.asarray(qi or [(0, 0, 0, 1)], np.int64).reshape(-1, 4)
        oa = np.asarray(oi or [(0, 0, 0)], np.int64).reshape(-1, 3)
        ua = np.asarray(ui or [(0, 0)], np.int64).reshape(-1, 2)
        ea = np.asarray(ei or [(0, 0)], np.int64).reshape(-1, 2)
        lg, sp = self.run(x, m, qa, oa, ua, ea)
        lg = lg[:no]; out = []; c = 0; qk = 0
        for b, r in enumerate(recs):
            qs = []
            for q in r["questions"]:
                n = qsz[qk]; qk += 1
                qs.append(lg[c:c + n]); c += n
            s = spn[b]
            out.append(dict(qs=qs, span=None if s is None else sp[s[0]:s[1], s[2]:s[3]]))
        return out

    def predict_raw(self, rows):
        """Generic-contract rows -> per row dict(qs=[dict(qid, qtype, names, logits (incl. [ABS] last if any),
        has_abs)], span=dict(names, offs (word char offsets), logits (W, L)) | None, n_tokens, windows)."""
        prepped = [self._prep(r) for r in rows]
        jobs = []
        for k, r in enumerate(prepped):
            rec = self._assemble(r)
            Wn = self._windows(r, rec) if rec["trunc"]["protected_cut"] else None
            if not Wn:
                if rec["trunc"]["protected_cut"]:
                    self.stats["labelled_cut"] += 1
                jobs.append((k, None, None, rec))
            else:
                for r2, ab, gidx in Wn:
                    jobs.append((k, ab, gidx, self._assemble(r2)))
        outs = [None] * len(jobs)
        order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][3]["ids"]))
        cur = []

        def flush(js):
            for j, o in zip(js, self._forward([jobs[j][3] for j in js])):
                outs[j] = o
        for j in order:
            w = max([len(jobs[i][3]["ids"]) for i in cur] + [len(jobs[j][3]["ids"])])
            if cur and (len(cur) + 1 > self.bs or w * (len(cur) + 1) > self.budget):
                flush(cur); cur = []
            cur.append(j)
        if cur:
            flush(cur)
        per = defaultdict(list)
        for (k, ab, gidx, rec), o in zip(jobs, outs):
            per[k].append((ab, gidx, rec, o))
        final = []
        for k, r in enumerate(prepped):
            lst = per[k]; rec0 = lst[0][2]
            qs = []
            for qi, q in enumerate(rec0["questions"]):
                L = np.stack([o["qs"][qi] for _, _, _, o in lst])
                agg = L.max(0) if q["qtype"] == "set" else L.mean(0)
                qs.append(dict(qid=q["qid"], qtype=q["qtype"], names=q["opt_names"], logits=agg, has_abs=q["abs_pos"] is not None))
            span = None
            if "w_pos" in rec0:
                names = rec0["names"]
                if len(lst) == 1 and lst[0][0] is None:
                    span = dict(names=names, offs=np.asarray(rec0["w_offs"]), logits=lst[0][3]["span"].astype(np.float64))
                else:
                    role = next(q["over"] for q in r["qs"] if q["type"] == "span")
                    part = next(p for p in r["parts"] if p[0] == role)
                    U = len(part[3][0])
                    acc = np.zeros((U, len(names)), np.float64); cnt = np.zeros(U, np.float64)
                    for (a, b), gidx, _, o in lst:
                        lg = o["span"]; n = len(lg)
                        acc[gidx[:n]] += lg; cnt[gidx[:n]] += 1
                    cov = cnt > 0
                    span = dict(names=names, offs=np.asarray(part[3][0])[:U],
                                logits=np.where(cov[:, None], acc / np.maximum(cnt, 1)[:, None], -30.0))
                    self.stats["units_uncovered"] += int((~cov).sum())
            n_tok = {p[0]: len(p[1]) for p in r["parts"]}
            final.append(dict(qs=qs, span=span, n_tokens=n_tok, windows=len(lst) if lst[0][0] is not None else 0,
                              n_input=sum(len(x[2]["ids"]) for x in lst)))
        return final

    # ------------------------------------------------------------------ calibration
    def temperature(self, qtype):
        return float(self.cal["temperature"].get(qtype, 1.0))

    def is_pii(self, qid, names):
        return qid == "pii" or bool(names and set(names) <= set(self.cal.get("pii_types", [])))

    def pii_length_threshold(self, n_tokens):
        """calibration.json pii_length_rule: log(thr) piecewise-linear in log(n_tokens) between anchors, constant outside."""
        a = self.cal["pii_length_rule"]["anchors"]
        return float(np.exp(np.interp(np.log(max(n_tokens, 1)), [np.log(x["n_tokens"]) for x in a], [np.log(x["threshold"]) for x in a])))

    def pii_sparse_gate(self, probs, offs, names, text, thr, rule):
        """calibration.json pii_sparse_gate: a doc with <= K spans decoded at probe_threshold is decoded at
        max(thr, t_sparse). Second decode pass over the same probabilities (no second forward)."""
        g = self.cal.get("pii_sparse_gate")
        if not g or g.get("K", -1) < 0 or thr >= g["t_sparse"] or not len(offs):
            return thr, rule
        if len(decode_spans(probs, offs, names, text, float(g.get("probe_threshold", 0.5)))) <= g["K"]:
            return float(g["t_sparse"]), rule + "+sparse"
        return thr, rule

    def span_threshold(self, qid, names, n_tokens):
        """Threshold before the (probability-dependent) PII sparse gate; calibrated() applies the gate."""
        th = self.cal["thresholds"]
        if self.is_pii(qid, names):
            if "pii_length_rule" in self.cal:
                return self.pii_length_threshold(n_tokens), "pii:length"
            for b in self.cal["pii_buckets"]:  # fallback: v1 calibration.json (no pii_length_rule)
                if n_tokens <= b["max_tokens"]:
                    return float(b["threshold"]), f"pii:{b['name']}"
        for fam in ("halu", "toxic"):
            if qid == fam:
                return float(th[f"span:{fam}"]), fam
        return float(th.get(f"span:{qid}", th["span:*"])), ("span:" + qid if f"span:{qid}" in th else "span:*")

    def set_threshold(self, qid):
        th = self.cal["thresholds"]
        return float(th.get(f"set:{qid}", th["set:*"]))

    def calibrated(self, raw, row, thresholds=None):
        """Raw logits of one row -> calibrated answers (list aligned with row['questions'])."""
        thresholds = thresholds or {}
        texts = {p["role"]: p["text"] for p in row["parts"]}
        qmap = {q["id"]: q for q in row["questions"]}
        ans = []
        for q in raw["qs"]:
            sch = qmap[q["qid"]]; lg = np.asarray(q["logits"], np.float64)
            n = len(q["names"]); T = self.temperature(q["qtype"])
            if q["qtype"] == "set":
                p = _sigmoid(lg[:n] / T); thr = float(thresholds.get(q["qid"], self.set_threshold(q["qid"])))
                probs = dict(zip(q["names"], p.tolist()))
                ans.append(dict(id=q["qid"], type="set", answer=[k for k, v in probs.items() if v > thr], probabilities=probs, threshold=thr))
                continue
            p = _softmax(lg[:n] / T); probs = dict(zip(q["names"], p.tolist()))
            a = dict(id=q["qid"], type=q["qtype"], answer=q["names"][int(np.argmax(lg[:n]))], probabilities=probs)
            if q["has_abs"]:
                a["abstain_probability"] = float(_softmax(lg / T)[-1])
            if q["qtype"] == "score":
                vals = sch.get("values") or {nm: (i / (n - 1) if n > 1 else 0.0) for i, nm in enumerate(q["names"])}
                a["expected_index"] = float(sum(i * pp for i, pp in enumerate(p)))
                a["score"] = float(sum(vals[nm] * pp for nm, pp in zip(q["names"], p)))
            ans.append(a)
        if raw["span"] is not None:
            sq = next(q for q in row["questions"] if q["type"] == "span")
            role = sq["over"]; text = texts[role]
            thr, rule = (float(thresholds[sq["id"]]), "given") if sq["id"] in thresholds else \
                self.span_threshold(sq["id"], raw["span"]["names"], raw["n_tokens"][role])
            probs = _sigmoid(raw["span"]["logits"] / self.temperature("span"))
            if rule.startswith("pii:"):
                thr, rule = self.pii_sparse_gate(probs, raw["span"]["offs"], raw["span"]["names"], text, thr, rule)
            sp = decode_spans(probs, raw["span"]["offs"], raw["span"]["names"], text, thr) if len(raw["span"]["offs"]) else []
            for s in sp:
                s["text"] = text[s["start"]:s["end"]]
            ans.append(dict(id=sq["id"], type="span", target_part=role, spans=sp, threshold=thr, threshold_rule=rule))
        return ans

    def predict_generic(self, rows, thresholds=None):
        raws = self.predict_raw(rows)
        return [dict(answers=self.calibrated(r, row, thresholds), meta=dict(tokens=r["n_tokens"], windows=r["windows"]))
                for r, row in zip(raws, rows)]

    # ------------------------------------------------------------------ Kai / Jev JSON
    @staticmethod
    def _names(lst):
        if isinstance(lst, dict):
            return dict(lst)
        out = {}
        for o in lst:
            if isinstance(o, str):
                out[o] = ""
            else:
                out[o["name"]] = o.get("description") or ""
        return out

    def _to_generic(self, req):
        parts = [dict(role=p["type"], text=p["text"]) for p in req["parts"]]
        roles = [p["role"] for p in parts]
        for r in roles:
            if f"[SEG_{r}]" not in self.mk:
                raise ValueError(f"part type must be one of user / context / answer, got {r}")
        qs, thr = [], {}
        for i, q in enumerate(req["questions"]):
            typ = q["type"]; qid = str(q.get("id", f"q{i}"))
            tp = q.get("target_part") or ("user" if "user" in roles else roles[0])
            g = dict(id=qid, type=typ, over=tp, text=q.get("text") or "")
            if typ == "span":
                g["labels"] = self._names(q.get("labels") or q.get("options"))
            else:
                opts = self._names(q.get("options") or q.get("labels") or q.get("levels"))
                g["options"] = opts
                if typ == "score":
                    g["levels"] = list(opts)
                    if q.get("values"):
                        g["values"] = dict(zip(opts, q["values"])) if isinstance(q["values"], list) else dict(q["values"])
                if typ == "choice":
                    g["abstain"] = bool(q.get("abstain", True))
            if q.get("threshold") is not None:
                thr[qid] = float(q["threshold"])
            qs.append(g)
        return parts, qs, thr

    def predict(self, req):
        """Kai/Jev-style request (dict or list of dicts) -> {"answers": [...], "meta": {...}} in question order.
        Several span questions are run as separate sequences (the model takes one span question per sequence)."""
        if isinstance(req, list):
            return [self.predict(r) for r in req]
        return self._predict(req)[0]

    def _predict(self, req, span_limits=None):
        """predict() plus what the SystemOne view needs from the raw pass: (result, {span qid: max word probability},
        input tokens over all sequences). span_limits {qid: (start, end)} restricts the max to words inside a range."""
        parts, qs, thr = self._to_generic(req)
        spans = [q for q in qs if q["type"] == "span"]; rest = [q for q in qs if q["type"] != "span"]
        rows = [dict(parts=parts, questions=rest + spans[:1])] + [dict(parts=parts, questions=[s]) for s in spans[1:]]
        rows = [r for r in rows if r["questions"]]
        raws = self.predict_raw(rows)
        res = [dict(answers=self.calibrated(r, row, thr), meta=dict(tokens=r["n_tokens"], windows=r["windows"]))
               for r, row in zip(raws, rows)]  # = predict_generic(rows, thr)
        span_max = {}
        for r, row in zip(raws, rows):
            if r["span"] is None:
                continue
            sq = next(q for q in row["questions"] if q["type"] == "span")
            p = _sigmoid(r["span"]["logits"] / self.temperature("span")).max(1) if len(r["span"]["offs"]) else np.zeros(0)
            lim = (span_limits or {}).get(sq["id"])
            if lim is not None and len(p):
                o = r["span"]["offs"]; p = p[(o[:, 0] >= lim[0]) & (o[:, 1] <= lim[1])]
            span_max[sq["id"]] = float(p.max()) if len(p) else 0.0
        by = {a["id"]: a for r in res for a in r["answers"]}
        out = dict(answers=[by[q["id"]] for q in qs], meta=dict(tokens=res[0]["meta"]["tokens"], windows=max(r["meta"]["windows"] for r in res),
                                                                sequences=len(rows)))
        return out, span_max, sum(r["n_input"] for r in raws)

    # ------------------------------------------------------------------ SystemOne
    def system_one(self, state, questions, model=None):
        """TypeSafe SystemOne request -> SystemOne response (the primary interface).

        state      str | JSON object | JSON array (a str holding a JSON object is read as that object)
        questions  {qid: question}: {"type": "noul", "instructions", "criteria"?: {"true", "false"}} |
                   {"type": "choice", "instructions", "criteria": {key: description | None}} |
                   {"type": "score", "instructions", "criteria": [level description, ...]} (dicts, or the typesafe-sdk
                   Noul / Choice / Score objects); extensions: {"type": "span" | "set", "instructions",
                   "criteria": {label: description | None}, "over"?, "threshold"?}. Every question may name the
                   state field(s) it is asked "over".
        returns    {"model", "answers": {qid: {"type": "noul", "noul"} | {"type": "choice", "choice", "confidence",
                   "probabilities"} | {"type": "score", "score", "confidence", "legend", "probabilities"}},
                   "usage": {"input_tokens", "output_tokens": 0}}, plus for span / set questions "spans" /
                   "sets" / "thresholds" and their Noul views in "answers" (see SYSTEM_ONE_NOTES)."""
        return system_one_response(self, state, questions, model)

    def system_one_request(self, body):
        """The body of POST /v1/systemone ({"state", "model", "questions"}) -> SystemOne response."""
        return system_one_request(self, body)

    # ------------------------------------------------------------------ GLiNER2-style
    def classify_text(self, text, labels, multi_label=False, threshold=None, include_confidence=False, question=None,
                      abstain=True, context=None):
        """labels: list | {name: description} (one task) or {task: list | {name: desc} | {"labels", "multi_label",
        "cls_threshold", "question"}} (several tasks, one forward pass). Returns {task: label | [labels]} (GLiNER2)."""
        single = isinstance(labels, list) or (isinstance(labels, dict) and labels and all(isinstance(v, str) for v in labels.values()))
        tasks = {"label": labels} if single else labels
        qs, thr = [], {}
        for task, spec in tasks.items():
            ml, th, qt = multi_label, threshold, question
            if isinstance(spec, dict) and "labels" in spec:
                ml = spec.get("multi_label", ml); th = spec.get("cls_threshold", th); qt = spec.get("question", qt); spec = spec["labels"]
            opts = self._names(spec)
            if qt is None:
                qt = ("Which labels apply to this text?" if ml else "Which label applies to this text?") if single else \
                     (f"Which {task} labels apply to this text?" if ml else f"Which {task} applies to this text?")
            q = dict(id=task, type="set" if ml else "choice", text=qt, options=opts, target_part="user")
            if not ml:
                q["abstain"] = abstain
            if th is not None:
                q["threshold"] = th
            qs.append(q)
        parts = [dict(type="user", text=text)] + ([dict(type="context", text=context)] if context else [])
        out = self.predict(dict(parts=parts, questions=qs))
        res = {}
        for a in out["answers"]:
            if a["type"] == "set":
                v = [dict(label=l, confidence=a["probabilities"][l]) for l in a["answer"]] if include_confidence else a["answer"]
            else:
                v = dict(label=a["answer"], confidence=a["probabilities"][a["answer"]]) if include_confidence else a["answer"]
            res[a["id"]] = v
        return res

    def extract_entities(self, text, labels, threshold=None, include_confidence=False, include_spans=False, question=None):
        """labels: list | {name: description}. Returns {"entities": {label: [text | {text, start, end, confidence}]}}.
        The 17 Vela PII types (no descriptions given) use the trained PII schema and the per-length PII thresholds."""
        opts = self._names(labels)
        pii = self.cal.get("pii_schema", {})
        if all(not d for d in opts.values()) and set(opts) <= set(pii.get("labels", {})):
            opts = {n: pii["labels"][n] for n in opts}; question = question or pii["text"]; qid = "pii"
        else:
            qid = "entities"
        q = dict(id=qid, type="span", text=question or "Which spans are named entities of these types?", labels=opts, target_part="user")
        if threshold is not None:
            q["threshold"] = threshold
        a = self.predict(dict(parts=[dict(type="user", text=text)], questions=[q]))["answers"][0]
        ents = {n: [] for n in opts}
        for s in a["spans"]:
            ents[s["label"]].append(dict(text=s["text"], start=s["start"], end=s["end"], confidence=s["probability"])
                                    if (include_confidence or include_spans) else s["text"])
        return dict(entities=ents)

    def detect_hallucination(self, context, answer, question=None, threshold=None):
        """Spans of `answer` not supported by `context` (the trained halu schema; context = 'User request: <question>
        \\n\\n<context>' when a question is given, as in training)."""
        hs = self.cal["halu_schema"]
        ctx = f"User request: {question}\n\n{context}" if question else context
        q = dict(id="halu", type="span", text=hs["text"], labels=hs["labels"], target_part="answer")
        if threshold is not None:
            q["threshold"] = threshold
        a = self.predict(dict(parts=[dict(type="context", text=ctx), dict(type="answer", text=answer)], questions=[q]))["answers"][0]
        return dict(hallucinated=bool(a["spans"]), spans=a["spans"], threshold=a["threshold"])

    def score_relevance(self, query, document):
        """Relevance of `document` to `query` as a score question over the user + context parts (0 / 0.5 / 1 levels)."""
        rs = self.cal["relevance_schema"]
        q = dict(id="relevance", type="score", text=rs["text"], options=[dict(name=k, description=v) for k, v in rs["levels"].items()],
                 values=rs["values"], target_part=["user", "context"])
        a = self.predict(dict(parts=[dict(type="user", text=query), dict(type="context", text=document)], questions=[q]))["answers"][0]
        return dict(score=a["score"], label=a["answer"], probabilities=a["probabilities"])

    def score(self, text, question, levels, context=None, values=None, target_part="user"):
        """Generic score question: levels = ordered list | {level: description}."""
        q = dict(id="score", type="score", text=question, options=levels, values=values, target_part=target_part)
        parts = [dict(type="user", text=text)] + ([dict(type="context", text=context)] if context else [])
        return self.predict(dict(parts=parts, questions=[q]))["answers"][0]


# =====================================================================================================================
# SystemOne (TypeSafe) interface: request/response shape of POST /v1/systemone, mapped onto predict()
# =====================================================================================================================
DEFAULT_MODEL_NAME = "vllm-sr/Vela-2.0-0.3B"
NOUL_DEFAULT_NO = "No. The statement or question is not satisfied."    # the Vela-family Noul defaults of the
NOUL_DEFAULT_YES = "Yes. The statement or question is satisfied."      # Decision runtime (Kai)
STATE_KEY_ROLES = {"request": "user", "user": "user", "prompt": "user",
                   "source": "context", "context": "context", "document": "context", "tools": "context",
                   "previous_answer": "context",
                   "answer": "answer", "response": "answer"}
ROLE_ORDER = ("user", "context", "answer")
SYSTEM_ONE_TYPES = ("noul", "choice", "score", "span", "set")
SYSTEM_ONE_NOTES = """
Mapping onto the typed-parts engine (predict()):
  choice -> choice question (options key: description; a None description renders the key alone; [ABS] present as in
            predict()); probabilities over the criteria keys; confidence = top-two probability margin.
  score  -> score question whose ordered levels are the criteria; probabilities / legend keyed "0".."n-1";
            score = expected zero-based level; confidence = 1 - variance / variance of the uniform distribution.
  noul   -> choice over ("no": criteria.false, "yes": criteria.true), defaults NOUL_DEFAULT_NO / NOUL_DEFAULT_YES;
            noul = P(yes).
  span   -> span question; "spans": {qid: [{label, start, end, text, probability}]} (code-point offsets into the text
            of the `over` field; when that field shares a part with other fields, spans are clipped to it); "thresholds": {qid: t}; answers[qid] = {"type": "noul", "noul": max word probability
            over that field} (above t exactly when a word is labelled, so spans are non-empty up to edge trimming).
  set    -> set question; "sets": {qid: {"selected": [...], "probabilities": {label: p}}}; "thresholds": {qid: t};
            answers["<qid>.<label>"] = {"type": "noul", "noul": p} for every label.
  state  -> str: the user part. JSON object: request / user / prompt -> user; source / context / document / tools /
            previous_answer -> context; answer / response -> answer; other keys -> context. A part fed by one known
            key is that value's text; a part fed by several keys (or by an unknown key) is "key:\n<value>" blocks
            joined by a blank line. Non-string values are canonical JSON. A JSON array is canonical JSON in user.
  over   -> a state key, a part name (user / context / answer) or a list of them. Default: every part of the state for
            noul / choice / score / set; for span the answer part if present, else user (else the first part).
  thresholds: span / set thresholds follow calibration.json by question id exactly as in predict() ("pii" or the 17 PII
            labels -> PII length rule + sparse gate; "halu" 0.55; "toxic" 0.1; set 0.3), unless "threshold" is given.
"""


class SystemOneError(ValueError):
    """An invalid SystemOne request; `loc` is the path of the offending value (as in a 422 `detail[].loc`)."""

    def __init__(self, loc, msg, type_="value_error"):
        super().__init__(f"{'.'.join(str(x) for x in loc)}: {msg}")
        self.loc, self.msg, self.type = list(loc), msg, type_

    def detail(self):
        return {"loc": ["body"] + self.loc, "msg": self.msg, "type": self.type}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_text(value):
    """JSON content -> model text: strings as they are, anything else as canonical JSON (as the Decision runtime)."""
    return value if isinstance(value, str) else canonical_json(value)


def _plain(q):
    """A question dict, or a typesafe-sdk Noul / Choice / Score (pydantic) object -> plain dict."""
    if isinstance(q, dict):
        return dict(q)
    if hasattr(q, "model_dump"):
        return {k: v for k, v in q.model_dump(mode="json").items() if v is not None}
    if hasattr(q, "type"):
        return {k: getattr(q, k) for k in ("type", "instructions", "criteria", "over", "threshold") if getattr(q, k, None) is not None}
    raise SystemOneError(["questions"], "a question must be an object with a type")


def _nonblank(v):
    return not (isinstance(v, str) and not v.strip())


def system_one_state(state):
    """SystemOne state -> (parts [{"type", "text"}] in user / context / answer order, {key: (role, start, end)})."""
    if isinstance(state, str):
        st = state.strip()
        if st.startswith("{"):
            try:
                obj = json.loads(st)
            except ValueError:
                obj = None
            if isinstance(obj, dict):
                state = obj
    if isinstance(state, str):
        if not state.strip():
            raise SystemOneError(["state"], "state must not be empty or whitespace")
        return [dict(type="user", text=state)], {"request": ("user", 0, len(state))}
    if isinstance(state, list):
        t = canonical_json(state)
        return [dict(type="user", text=t)], {"request": ("user", 0, len(t))}
    if not isinstance(state, dict) or not state:
        raise SystemOneError(["state"], "state must be a non-empty string, JSON object or JSON array")
    by_role = defaultdict(list)
    for k, v in state.items():
        role = STATE_KEY_ROLES.get(str(k).lower(), "context")
        by_role[role].append((str(k), content_text(v), str(k).lower() in STATE_KEY_ROLES))
    parts, fields = [], {}
    for role in ROLE_ORDER:
        items = by_role.get(role)
        if not items:
            continue
        if len(items) == 1 and items[0][2]:
            k, t, _ = items[0]; fields[k] = (role, 0, len(t)); text = t
        else:
            text = ""
            for k, t, _ in items:
                if text:
                    text += "\n\n"
                head = f"{k}:\n"; fields[k] = (role, len(text) + len(head), len(text) + len(head) + len(t))
                text += head + t
        if not text.strip():
            raise SystemOneError(["state"], f"the {role} part of the state is empty")
        parts.append(dict(type=role, text=text))
    return parts, fields


def _resolve_over(over, fields, roles, qloc):
    """`over` (state key | part name | list) -> (target_part for predict(), field range for span offsets or None)."""
    items = over if isinstance(over, list) else [over]
    out, rng = [], None
    for o in items:
        if not isinstance(o, str):
            raise SystemOneError(qloc + ["over"], "over must be a state key, a part name or a list of them")
        if o in fields:
            role, a, b = fields[o]
            if not isinstance(over, list):
                rng = (a, b)
        elif o in roles:
            role = o
        elif o in STATE_KEY_ROLES and STATE_KEY_ROLES[o] in roles:
            role = STATE_KEY_ROLES[o]
        else:
            raise SystemOneError(qloc + ["over"], f"'{o}' is not a field of the state (fields: {sorted(fields)}, parts: {roles})")
        if role not in out:
            out.append(role)
    if not out:
        raise SystemOneError(qloc + ["over"], "over must not be empty")
    out = [r for r in roles if r in out]  # sequence order, so the pooled range covers exactly these parts
    return (out[0] if len(out) == 1 else out), rng


def system_one_to_predict(state, questions):
    """SystemOne (state, questions) -> (predict() request, per-question plan used to build the response)."""
    parts, fields = system_one_state(state)
    roles = [p["type"] for p in parts]
    if hasattr(questions, "items") and not isinstance(questions, dict):
        questions = dict(questions.items())
    if not isinstance(questions, dict) or not questions:
        raise SystemOneError(["questions"], "questions must be a non-empty object of named questions")
    pq, plan = [], []
    for qid, q in questions.items():
        loc = ["questions", qid]
        if not isinstance(qid, str) or not qid.strip():
            raise SystemOneError(["questions"], "question IDs must not be empty or whitespace")
        q = _plain(q); typ = q.get("type")
        if typ not in SYSTEM_ONE_TYPES:
            raise SystemOneError(loc + ["type"], f"type must be one of {list(SYSTEM_ONE_TYPES)}", "literal_error")
        allowed = {"type", "instructions", "criteria", "over"} | ({"threshold"} if typ in ("span", "set") else set())
        extra = sorted(set(q) - allowed)
        if extra:
            raise SystemOneError(loc + [extra[0]], "Extra inputs are not permitted", "extra_forbidden")
        ins = q.get("instructions")
        if not _nonblank(ins):
            raise SystemOneError(loc + ["instructions"], "instructions must not be empty or whitespace")
        text = "" if ins is None else content_text(ins)
        crit = q.get("criteria")
        if "over" in q and q["over"] is not None:
            tp, rng = _resolve_over(q["over"], fields, roles, loc)
        elif typ == "span":
            role = "answer" if "answer" in roles else ("user" if "user" in roles else roles[0])
            tp, rng = role, None
        else:
            tp, rng = (roles[0] if len(roles) == 1 else list(roles)), None
        g = dict(id=qid, type=typ, text=text, target_part=tp)
        if typ == "noul":
            if crit is not None and (not isinstance(crit, dict) or set(crit) - {"true", "false"}):
                raise SystemOneError(loc + ["criteria"], "noul criteria is an object with optional true / false")
            crit = crit or {}
            for k in ("true", "false"):
                if not _nonblank(crit.get(k)):
                    raise SystemOneError(loc + ["criteria", k], "Noul criterion must not be empty or whitespace")
            no = NOUL_DEFAULT_NO if crit.get("false") is None else content_text(crit["false"])
            yes = NOUL_DEFAULT_YES if crit.get("true") is None else content_text(crit["true"])
            g.update(type="choice", options=[dict(name="no", description=no), dict(name="yes", description=yes)])
        elif typ in ("choice", "set", "span"):
            lo, hi = (2, 255) if typ == "choice" else (1, 255)
            if not isinstance(crit, dict) or not lo <= len(crit) <= hi:
                raise SystemOneError(loc + ["criteria"], f"{typ} criteria must be an object with {lo} to {hi} entries")
            for k, v in crit.items():
                if not isinstance(k, str) or not k.strip() or not _nonblank(v):
                    raise SystemOneError(loc + ["criteria", k], "option names and descriptions must not be empty or whitespace")
            opts = [dict(name=k, description="" if v is None else content_text(v)) for k, v in crit.items()]
            g["labels" if typ == "span" else "options"] = opts
        else:  # score
            if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
                raise SystemOneError(loc + ["criteria"], "score criteria must be a list of 2 to 10 levels")
            names = []
            for i, v in enumerate(crit):
                if v is None or not _nonblank(v):
                    raise SystemOneError(loc + ["criteria", i], "Score criterion must not be empty or whitespace")
                names.append(content_text(v))
            if len(set(names)) != len(names):
                raise SystemOneError(loc + ["criteria"], "score levels must be distinct")
            g["options"] = names
        if q.get("threshold") is not None:
            t = q["threshold"]
            if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0.0 <= t <= 1.0:
                raise SystemOneError(loc + ["threshold"], "threshold must be a number in [0, 1]")
            g["threshold"] = float(t)
        pq.append(g)
        plan.append(dict(qid=qid, type=typ, criteria=crit, range=rng, part=tp))
    for p in plan:  # the Noul views of set labels must not overwrite a question
        if p["type"] == "set":
            for lab in p["criteria"]:
                if f"{p['qid']}.{lab}" in questions:
                    raise SystemOneError(["questions", f"{p['qid']}.{lab}"], "collides with the Noul view of a set label")
    return dict(parts=parts, questions=pq), plan


def _choice_confidence(p):
    v = sorted(p, reverse=True)
    return 1.0 if len(v) == 1 else min(1.0, max(0.0, v[0] - v[1]))


def _score_confidence(p):
    n = len(p)
    if n == 1:
        return 1.0
    mean = math.fsum(i * x for i, x in enumerate(p))
    var = math.fsum(x * (i - mean) ** 2 for i, x in enumerate(p))
    return min(1.0, max(0.0, 1.0 - var / ((n * n - 1) / 12)))


def system_one_response(engine, state, questions, model=None):
    req, plan = system_one_to_predict(state, questions)
    limits = {p["qid"]: p["range"] for p in plan if p["type"] == "span" and p["range"] is not None}
    try:
        res, span_max, n_input = engine._predict(req, span_limits=limits)
    except ValueError as e:
        if "does not fit" in str(e):
            raise SystemOneError(["questions"], f"the questions alone do not fit the {engine.max_len}-token window",
                                 "too_long") from e
        raise
    by = {a["id"]: a for a in res["answers"]}
    answers, spans, sets, thresholds = {}, {}, {}, {}
    for p in plan:
        a, qid = by[p["qid"]], p["qid"]
        if p["type"] == "noul":
            answers[qid] = dict(type="noul", noul=float(a["probabilities"]["yes"]))
        elif p["type"] == "choice":
            pr = {k: float(a["probabilities"][k]) for k in p["criteria"]}
            answers[qid] = dict(type="choice", choice=a["answer"], confidence=_choice_confidence(list(pr.values())),
                                probabilities=pr)
        elif p["type"] == "score":
            names = [content_text(v) for v in p["criteria"]]
            pv = [float(a["probabilities"][n]) for n in names]
            answers[qid] = dict(type="score", score=math.fsum(i * x for i, x in enumerate(pv)),
                                confidence=_score_confidence(pv), legend={str(i): v for i, v in enumerate(p["criteria"])},
                                probabilities={str(i): x for i, x in enumerate(pv)})
        elif p["type"] == "set":
            pr = {k: float(a["probabilities"][k]) for k in p["criteria"]}
            sets[qid] = dict(selected=list(a["answer"]), probabilities=pr)
            thresholds[qid] = float(a["threshold"])
            for k, v in pr.items():
                answers[f"{qid}.{k}"] = dict(type="noul", noul=v)
        else:  # span; over a field of a shared part: spans clipped to the field, offsets into the field's text
            out = []
            if p["range"] is None:
                out = [dict(label=z["label"], start=z["start"], end=z["end"], text=z["text"], probability=float(z["probability"]))
                       for z in a["spans"]]
            else:
                lo, hi = p["range"]; ptext = next(x["text"] for x in req["parts"] if x["type"] == p["part"])
                for z in a["spans"]:
                    st_, en_ = _trim(ptext, max(z["start"], lo), min(z["end"], hi))
                    if en_ > st_:
                        out.append(dict(label=z["label"], start=st_ - lo, end=en_ - lo, text=ptext[st_:en_],
                                        probability=float(z["probability"])))
            spans[qid] = out
            thresholds[qid] = float(a["threshold"])
            answers[qid] = dict(type="noul", noul=span_max.get(qid, 0.0))
    resp = dict(model=model or DEFAULT_MODEL_NAME, answers=answers, usage=dict(input_tokens=int(n_input), output_tokens=0))
    if spans:
        resp["spans"] = spans
    if sets:
        resp["sets"] = sets
    if thresholds:
        resp["thresholds"] = thresholds
    return resp


def system_one_request(engine, body, served=None):
    """A POST /v1/systemone body -> response; `served` (a set of model names) rejects other models as the runtime does."""
    if not isinstance(body, dict):
        raise SystemOneError([], "the request body must be a JSON object")
    for k in ("state", "model", "questions"):
        if k not in body:
            raise SystemOneError([k], "Field required", "missing")
    extra = sorted(set(body) - {"state", "model", "questions"})
    if extra:
        raise SystemOneError([extra[0]], "Extra inputs are not permitted", "extra_forbidden")
    m = body["model"]
    if not isinstance(m, str) or not m.strip():
        raise SystemOneError(["model"], "model must not be empty or whitespace")
    if served is not None and m not in served:
        raise SystemOneError(["model"], "Model is not served by this runtime", "value_error.model")
    return system_one_response(engine, body["state"], body["questions"], m)
