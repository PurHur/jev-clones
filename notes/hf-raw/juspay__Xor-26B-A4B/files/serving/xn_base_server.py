#!/usr/bin/env python3
"""xn-base: frozen-base /v1/systemone baseline for XOR Next.

One scoring row per question (independent rows), all rows of a request sent as ONE SGLang /generate batch of token ids;
SGLang's radix cache shares the state prefix between rows. Each row, in the model's own chat template with thinking off:

    <user turn>  Context:\n<state>\n\nQuestion: <instructions>\nOptions:\n(A) <opt>\n(B) <opt>...  <turn end, model turn>
    Answer: (                                  <- read the logprobs of the label tokens here

Labels: A..Z, then single-token upper-case pairs AA, AB, ... up to 255 (all verified one token in context at startup).
choice options render "key: description" (or "key" alone); noul renders ["no[: false]", "yes[: true]"].
Probabilities = softmax(label logprobs / T[type]); T defaults to 1.0. Every answer also carries `x_lp`, the raw label
logprobs in option order, so a temperature can be fitted offline on held-out data without re-running.

No per-benchmark prompts, no truncation, no option pruning: requests that do not fit are refused with 422 and a
capacity marker. Env: XN_SGLANG_URL, XN_MODEL_PATH, XN_MODEL_ID, XN_MAX_CONTEXT, XN_TEMP_JSON, XN_PORT.
"""
import base64, binascii, hashlib, http.client, json, math, os, string, sys, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from functools import lru_cache

from transformers import AutoProcessor, AutoTokenizer

SGLANG = os.environ.get("XN_SGLANG_URL", "http://127.0.0.1:49040")
MODEL_PATH = os.environ.get("XN_MODEL_PATH", "/model")
MODEL_ID = os.environ.get("XN_MODEL_ID", "xn-base-gemma4-26b-a4b")
MAX_CONTEXT = int(os.environ.get("XN_MAX_CONTEXT", "131072"))
TEMP = {"choice": 1.0, "noul": 1.0, "score": 1.0, **{k: float(v) for k, v in json.loads(os.environ.get("XN_TEMP_JSON", "{}")).items()}}
MAX_OPTIONS = 255
NULL_LP = -100.0          # SGLang reports null for vanishingly unlikely tokens
MAX_BODY_BYTES = 8 * 1024 * 1024

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
E = lambda s: tok.encode(s, add_special_tokens=False)

_S = "@@XN_USER@@"
_head, _tail = tok.apply_chat_template([{"role": "user", "content": _S}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False).split(_S)
HEAD, TAIL = E(_head), E(_tail)
ANSWER = E("Answer: (")
OPEN = E("\n(")


def _labels():
    names = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    out, seen = [], set()
    for n in names:
        # model-agnostic: the readout label = the ACTUAL next-token of 'Answer: (L'
        # (for gemma this is the standalone letter id — byte-identical to before;
        # for merging tokenizers it is the merged token, which is what the model
        # actually emits after the answer prefix)
        t = E("Answer: (" + n)
        if len(t) <= len(ANSWER):
            continue          # this letter merges into the prefix — unusable
        if t[-1] in seen:
            continue          # duplicate token (e.g. a double-letter name)
        seen.add(t[-1])
        out.append((n, t[-1]))
        if len(out) == MAX_OPTIONS:
            break
    ids = [i for _, i in out]
    assert len(ids) == MAX_OPTIONS == len(set(ids)), 'label ids must be unique'
    return [n for n, _ in out], ids


LABEL_NAMES, LABEL_IDS = _labels()
CONFIG = {"model_id": MODEL_ID, "model_path": MODEL_PATH, "sglang": SGLANG, "temperatures": TEMP, "max_context": MAX_CONTEXT,
          "layout": "chat/state-first/independent-rows", "labels_sha256": hashlib.sha256(json.dumps(LABEL_IDS).encode()).hexdigest(),
          "template_head": _head, "template_tail": _tail, "source_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}


class Refused(ValueError):
    """Declared capacity limit: 422 with a marker the Decision Index http engine maps to `unsupported`."""


def _txt(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def render_state(state):
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def render_question(qid, q):
    """-> (instructions, [option texts], finisher(probs, lps) -> answer)."""
    t = q.get("type")
    crit = q.get("criteria")
    ins = q.get("instructions")
    if t == "choice":
        if isinstance(crit, list):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict):
            raise ValueError(f"{qid}: choice criteria must be a map")
        if len(crit) < 2:
            raise Refused(f"{qid}: a choice needs at least two options")
        if len(crit) > MAX_OPTIONS:
            raise Refused(f"{qid}: {len(crit)} options; the limit is {MAX_OPTIONS} options per choice")
        keys = list(crit)
        opts = [k if crit[k] in (None, "") else f"{k}: {_txt(crit[k])}" for k in keys]
        if ins in (None, ""):
            raise ValueError(f"{qid}: choice question without instructions")

        def finish(p, lp):
            j = max(range(len(p)), key=p.__getitem__)
            return {"type": "choice", "choice": keys[j], "probabilities": {k: x for k, x in zip(keys, p)},
                    "confidence": round(p[j], 6), "x_lp": [round(x, 4) for x in lp]}
        return _txt(ins), opts, finish
    if t == "noul":
        c = crit if isinstance(crit, dict) else {}
        f, tr = c.get("false"), c.get("true")
        opts = ["no" if f in (None, "") else f"no: {_txt(f)}", "yes" if tr in (None, "") else f"yes: {_txt(tr)}"]
        if ins in (None, ""):
            if f in (None, "") and tr in (None, ""):
                raise ValueError(f"{qid}: noul question without instructions or criteria")
            ins = "Which answer fits the context?"

        def finish(p, lp):
            return {"type": "noul", "noul": p[1], "x_lp": [round(x, 4) for x in lp]}
        return _txt(ins), opts, finish
    if t == "score":
        if isinstance(crit, dict):                                   # legend form {"0": "...", "1": "..."}
            crit = [crit[k] for k in sorted(crit, key=float)]
        if not isinstance(crit, list):
            raise ValueError(f"{qid}: score criteria must be an ordered list of level descriptions")
        if not 2 <= len(crit) <= 10:
            raise Refused(f"{qid}: a score takes 2 to 10 levels, got {len(crit)}")
        if ins in (None, ""):
            raise ValueError(f"{qid}: score question without instructions")
        opts = [f"{i}: {_txt(c)}" for i, c in enumerate(crit)]

        def finish(p, lp):
            return {"type": "score", "score": round(sum(i * x for i, x in enumerate(p)), 4),
                    "legend": {str(i): _txt(c) for i, c in enumerate(crit)},
                    "probabilities": {str(i): x for i, x in enumerate(p)},
                    "confidence": round(max(p), 6), "x_lp": [round(x, 4) for x in lp]}
        return _txt(ins), opts, finish
    raise ValueError(f"{qid}: unsupported question type {t!r}")


def build_row(state_ids, ins, opts):
    ids = HEAD + E("Context:\n") + state_ids + E(f"\n\nQuestion: {ins}\nOptions:")
    for j, o in enumerate(opts):
        # natural per-tokenizer encoding of the option line (identical ids for gemma)
        ids += E("\n(" + LABEL_NAMES[j] + ") " + o)
    return ids + TAIL + ANSWER


def softmax_T(lp, T):
    m = max(lp)
    e = [math.exp((x - m) / T) for x in lp]
    s = sum(e)
    return [x / s for x in e]


def validate_media(body):
    images = body.get("images")
    video = body.get("video")
    if images is not None:
        if not isinstance(images, list) or not 1 <= len(images) <= 8:
            raise ValueError("images must contain 1 to 8 image data URLs")
        for value in images:
            validate_data_url(value, "image")
    if video is not None:
        validate_data_url(video, "video")
    if images is not None and video is not None:
        raise ValueError("send images or video, not both in the same request")
    return images, video


def validate_data_url(value, kind):
    if not isinstance(value, str) or not value.startswith(f"data:{kind}/"):
        raise ValueError(f"{kind} must be a base64 data URL")
    header, separator, encoded = value.partition(",")
    if not separator or not header.endswith(";base64") or not encoded:
        raise ValueError(f"{kind} must be a non-empty base64 data URL")
    try:
        base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError(f"{kind} contains invalid base64") from None


@lru_cache(maxsize=9)
def media_template_head(image_count, has_video):
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    media = ([{"type": "image"} for _ in range(image_count)] if image_count else [{"type": "video"}])
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": media + [{"type": "text", "text": _S}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    head, tail = rendered.split(_S)
    if E(tail) != TAIL:
        raise ValueError("media template suffix differs from the scoring template")
    return E(head)


def generate(rows, label_sets, images=None, video=None):
    body = {"input_ids": rows, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob": True, "logprob_start_len": -1, "token_ids_logprob": label_sets}
    if images:
        body["image_data"] = [images] * len(rows)
    if video:
        body["video_data"] = [[video]] * len(rows)
    data = json.dumps(body).encode()
    for attempt in range(6):          # scoring is side-effect free, so a connection the backend reset under load is retried
        try:
            req = urllib.request.Request(SGLANG + "/generate", data, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                out = json.load(r)
            return out if isinstance(out, list) else [out]
        except (ConnectionResetError, ConnectionRefusedError, http.client.RemoteDisconnected) as e:
            if attempt == 5:
                raise
            time.sleep(0.2 * 2 ** attempt)
        except urllib.error.URLError as e:
            if not isinstance(e.reason, (ConnectionResetError, ConnectionRefusedError)) or attempt == 5:
                raise
            time.sleep(0.2 * 2 ** attempt)


def evaluate(state, questions, images=None, video=None):
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty map")
    state_ids = E("") if state is None else tok.encode(render_state(state), add_special_tokens=False)
    prefix = media_template_head(len(images) if images else 0, bool(video)) if images or video else None
    order, rows, sets, finishers = [], [], [], []
    for qid, q in questions.items():
        ins, opts, finish = render_question(qid, q)
        row = build_row(state_ids, ins, opts)
        if prefix is not None:
            row = prefix + row[len(HEAD):]
        if len(row) + 1 > MAX_CONTEXT:
            raise Refused(f"{qid}: prompt of {len(row)} tokens exceeds the maximum context length of {MAX_CONTEXT}")
        order.append((qid, q["type"]))
        rows.append(row)
        sets.append(LABEL_IDS[:len(opts)])
        finishers.append(finish)
    # two-order averaging (env-gated): each choice question is also rendered with
    # its options reversed; the per-option probabilities are averaged in the
    # original index space (kills the position bias; measured +1.0 top1 on the
    # 800-row gold probe). The merged probs re-enter the standard pipeline as
    # pseudo-logprobs, so softmax_T/TEMP and the tempfit stay untouched.
    TWO_ORDER = os.environ.get("XN_TWO_ORDER", "0") == "1"
    n_fwd = len(rows)
    rev_map = []   # rev-row j -> the forward index in `order`
    if TWO_ORDER:
        for qi, (qid, qtype) in enumerate(order):
            if qtype != "choice":
                continue
            ins, opts, _f = render_question(qid, questions[qid])
            if len(opts) < 2:
                continue
            rrow = build_row(state_ids, ins, opts[::-1])
            if prefix is not None:
                rrow = prefix + rrow[len(HEAD):]
            if len(rrow) + 1 > MAX_CONTEXT:
                continue
            rows.append(rrow)
            sets.append(LABEL_IDS[:len(opts)])
            rev_map.append(qi)
    t0 = time.time()
    outs = generate(rows, sets, images, video) if images or video else generate(rows, sets)
    engine_ms = (time.time() - t0) * 1000
    lps = []
    for out, ids in zip(outs, sets):
        got = {int(tid): (lp if lp is not None else NULL_LP) for lp, tid, _ in out["meta_info"]["output_token_ids_logprobs"][0]}
        lps.append([got.get(i, NULL_LP) for i in ids])
    if TWO_ORDER:
        for j, ridx in enumerate(range(n_fwd, len(rows))):
            if j >= len(rev_map):
                break
            qi = rev_map[j]
            qid, qtype = order[qi]
            if qtype != "choice":
                continue
            K = len(lps[qi])
            if K < 2 or len(lps[ridx]) != K:
                continue
            p_f = softmax_T(lps[qi], 1.0)
            p_r = softmax_T(lps[ridx], 1.0)
            pa = [(p_f[j] + p_r[K - 1 - j]) / 2 for j in range(K)]
            lps[qi] = [math.log(max(p, 1e-12)) for p in pa]
    answers, in_tokens = {}, 0
    for (qid, qtype), lp, finish in zip(order, lps, finishers):
        answers[qid] = finish(softmax_T(lp, TEMP[qtype]), lp)
    in_tokens = sum(o["meta_info"]["prompt_tokens"] for o in outs[:n_fwd])
    return answers, in_tokens, engine_ms


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj, extra=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send(200, {"data": [{"id": MODEL_ID}]})
        elif self.path == "/config":
            self._send(200, CONFIG)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        t0 = time.time()
        try:
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length", "0"))
            if n > MAX_BODY_BYTES:
                self.close_connection = True
                return self._send(422, {"error": "request body exceeds 8 MB"})
            if n <= 0:
                raise ValueError("request body must be non-empty")
            body = json.loads(self.rfile.read(n))
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
            images, video = validate_media(body)
            answers, in_tokens, engine_ms = evaluate(body.get("state"), body.get("questions"), images, video)
            total_ms = (time.time() - t0) * 1000
            self._send(200, {"model": MODEL_ID, "answers": answers,
                             "usage": {"input_tokens": in_tokens, "output_tokens": len(answers)}},
                       {"x-engine-ms": f"{engine_ms:.1f}", "x-total-ms": f"{total_ms:.1f}"})
        except Refused as e:
            self._send(422, {"error": str(e)})
        except (ValueError, KeyError, TypeError) as e:
            self._send(400, {"error": str(e)})
        except urllib.error.HTTPError as e:
            self._send(502, {"error": f"backend {e.code}: {e.read()[:300]!r}"})
        except Exception as e:
            print(json.dumps({"xn_error": f"{type(e).__name__}: {e}"[:500]}), file=sys.stderr, flush=True)
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    port = int(os.environ.get("XN_PORT", "30002"))
    print(json.dumps({"xn_base": "ready", "port": port, **{k: v for k, v in CONFIG.items() if not k.startswith("template")}}), flush=True)
    ThreadingHTTPServer.request_queue_size = 256
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer.allow_reuse_port = True     # several processes on one port: the kernel spreads connections (GIL)
    ThreadingHTTPServer((os.environ.get("XN_HOST", "127.0.0.1"), port), Handler).serve_forever()
