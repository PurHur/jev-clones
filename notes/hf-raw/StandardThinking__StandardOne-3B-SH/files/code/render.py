"""Template `schema-v1`: render a schema request into one user turn and locate every question/option span.

Design: request rendering for the joint schema head.

A *schema request* (the internal, normalised form used everywhere in this package):

    {"state": str | dict | list,
     "images": [data URL, ...],                       # optional
     "questions": {key: {"type": "choice" | "noul" | "score",
                         "instructions": str | dict | list,
                         "options": [{"name": str, "description": str | dict | list | None}, ...]}}}

Options are always stored in CANONICAL order: choice = caller order, noul = [true, false], score = levels 0..K-1.
Rendering may show them in another order (augmentation); spans are always returned per canonical index.

Encoding (``Encoder.encode``) returns token ids plus token spans. Spans are half-open [start, end) token indices.
"""
import hashlib
import os
import re
import json
import math
import random

SYSTEM_PROMPT = os.environ.get("SH_SYSTEM_PROMPT", "none")   # this model was trained without the template's default system prompt; "default" keeps it
TEMPLATE_ID = "schema-v1" if SYSTEM_PROMPT == "default" else "schema-v1-nosys"
MAX_LENGTH = 32768
MAX_OPTIONS = 255
MAX_QUESTIONS = 256
MAX_SCORE_LEVELS = 10
QTYPES = ("choice", "noul", "score")
# Token roles used by the head's role embedding.
ROLE_OTHER, ROLE_STATE, ROLE_PREVIEW, ROLE_QUESTION, ROLE_OPTION = 0, 1, 2, 3, 4
N_ROLES = 5

# Wording variants. W0 is canonical (eval always uses W0); W1..W3 are augmentation only.
WORDINGS = {
    "W0": {"preamble": "Read the state and answer every question. Each question lists its possible answers.",
           "preview": "Questions to answer:", "state": "State:", "questions": "Questions:",
           "choice": "(choose one)", "noul": "(true or false)", "score": "(score 0 to {top})", "bullet": "- "},
    "W1": {"preamble": "Answer each question below using only the information provided. Every question lists "
                       "the answers it allows.",
           "preview": "You will be asked:", "state": "Context:", "questions": "Answer these:",
           "choice": "[pick one]", "noul": "[true/false]", "score": "[rate 0-{top}]", "bullet": "* "},
    "W2": {"preamble": "Use the input to decide every question. Pick exactly one of the listed answers for each.",
           "preview": "Questions:", "state": "Input:", "questions": "Decide:",
           "choice": "(one of)", "noul": "(true/false)", "score": "(scale 0..{top})", "bullet": "- "},
    "W3": {"preamble": "Below is some material followed by questions. Choose one listed answer per question.",
           "preview": "Asked below:", "state": "Material:", "questions": "Questions and answers:",
           "choice": "[choose one]", "noul": "[true or false]", "score": "[score from 0 to {top}]", "bullet": "* "},
}


class EncodeError(ValueError):
    """A request that cannot be encoded; `reason` is a short machine-readable code (drop statistics)."""

    def __init__(self, reason, message=""):
        super().__init__(f"{reason}: {message}" if message else reason)
        self.reason = reason


def require(condition, reason, message=""):
    if not condition:
        raise EncodeError(reason, message)


def render_value(value):
    """Plain strings verbatim; structured values as sorted, indented JSON (= jev-adapter render_native_state_text)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)


# ----------------------------------------------------------------------------------------------- validation
def validate_request(req):
    """Structural checks of a schema request (raises EncodeError). Returns the list of question keys."""
    require(isinstance(req, dict) and isinstance(req.get("questions"), dict), "bad_request", "questions missing")
    keys = list(req["questions"])
    require(1 <= len(keys) <= MAX_QUESTIONS, "bad_question_count", str(len(keys)))
    for key in keys:
        require(isinstance(key, str) and key.strip() and "\n" not in key and "[" not in key and "]" not in key,
                "bad_question_key", repr(key))
        q = req["questions"][key]
        t = q.get("type")
        require(t in QTYPES, "bad_type", repr(t))
        opts = q.get("options")
        require(isinstance(opts, list), "bad_options", key)
        names = [o.get("name") for o in opts]
        require(all(isinstance(n, str) and n.strip() and "\n" not in n for n in names), "bad_option_name", key)
        require(len(set(names)) == len(names), "duplicate_option_name", key)
        if t == "choice":
            require(2 <= len(opts) <= MAX_OPTIONS, "bad_option_count", f"{key}: {len(opts)}")
        elif t == "noul":
            require(names == ["true", "false"], "bad_noul_options", f"{key}: {names}")
        else:
            require(2 <= len(opts) <= MAX_SCORE_LEVELS, "bad_score_levels", f"{key}: {len(opts)}")
            require(names == [str(i) for i in range(len(opts))], "bad_score_names", key)
    return keys


# ----------------------------------------------------------------------------------------------- rendering
def render(req, wording="W0", question_order=None, option_orders=None, preview=True):
    """Render the user text. Returns (content, segments).

    question_order: list of keys in display order (default: request order).
    option_orders: {key: [canonical index shown at display position 0, 1, ...]} (default identity; score must be
                   identity because levels are ordinal).
    segments: {"state": (c0, c1), "preview": {key: (c0, c1)}, "question": {key: (c0, c1)},
               "option": {key: [(c0, c1) per CANONICAL option index]}} as character offsets into `content`.
    The state segment includes its header line so it is never empty.
    """
    keys = validate_request(req)
    order = list(question_order) if question_order is not None else keys
    require(sorted(order) == sorted(keys), "bad_question_order")
    w = WORDINGS[wording]
    parts, pos = [], 0
    segs = {"state": None, "preview": {}, "question": {}, "option": {}}

    def emit(text):
        nonlocal pos
        start = pos
        parts.append(text)
        pos += len(text)
        return start, pos

    emit(w["preamble"] + "\n\n")
    if preview:
        emit(w["preview"] + "\n")
        for key in order:
            text = f"[{key}] {render_value(req['questions'][key]['instructions'])}"
            segs["preview"][key] = emit(text)
            emit("\n")
        emit("\n")
    s0, _ = emit(w["state"] + "\n")
    _, s1 = emit(render_value(req.get("state", "")))
    segs["state"] = (s0, s1)
    emit("\n\n" + w["questions"])
    for key in order:
        q = req["questions"][key]
        n = len(q["options"])
        tag = w[q["type"]].format(top=n - 1)
        emit("\n")
        segs["question"][key] = emit(f"[{key}] {tag} {render_value(q['instructions'])}")
        perm = list(range(n)) if not option_orders or key not in option_orders else list(option_orders[key])
        require(sorted(perm) == list(range(n)), "bad_option_order", key)
        require(q["type"] != "score" or perm == list(range(n)), "score_order_must_be_identity", key)
        spans = [None] * n
        for canonical in perm:
            o = q["options"][canonical]
            desc = render_value(o.get("description"))
            text = o["name"] if not desc else f"{o['name']}: {desc}"
            emit("\n" + w["bullet"])
            spans[canonical] = emit(text)
        segs["option"][key] = spans
    return "".join(parts), segs


# ----------------------------------------------------------------------------------------------- augmentation
def _draw(row_id, epoch, what):
    return int(hashlib.sha256(f"{row_id}|{epoch}|{what}|schema-aug-v1".encode()).hexdigest()[:16], 16)


def augmentation(req, row_id, epoch, p_choice=1.0, p_noul=0.5, p_field=1.0, p_wording=0.5):
    """Deterministic per (row id, epoch) augmentation plan: wording, question order, option orders."""
    keys = list(req["questions"])
    frac = lambda what: _draw(row_id, epoch, what) / float(1 << 64)  # noqa: E731
    wording = "W0"
    if frac("wording") < p_wording:
        wording = ("W1", "W2", "W3")[_draw(row_id, epoch, "wording-pick") % 3]
    order = list(keys)
    if len(keys) > 1 and frac("fields") < p_field:
        random.Random(_draw(row_id, epoch, "field-perm")).shuffle(order)
    option_orders = {}
    for key in keys:
        q = req["questions"][key]
        n = len(q["options"])
        p = p_choice if q["type"] == "choice" else p_noul if q["type"] == "noul" else 0.0
        if p > 0 and frac("opt-gate|" + key) < p:
            perm = list(range(n))
            random.Random(_draw(row_id, epoch, "opt-perm|" + key)).shuffle(perm)
            option_orders[key] = perm
    return {"wording": wording, "question_order": order, "option_orders": option_orders}


# ----------------------------------------------------------------------------------------------- encoding
class Encoder:
    """Request -> token ids + token spans, with the trainer's chat-template call shape.

    tokenizer: the canonical tokenizer (AutoTokenizer as the trainer loads it); used for apply_chat_template and
               the round-trip check.
    fast: a `tokenizers.Tokenizer` loaded from the same tokenizer.json (character offsets).
    processor: AutoProcessor, only needed for requests with images.
    """

    def __init__(self, tokenizer, fast, processor=None, max_length=MAX_LENGTH, image_token_id=None,
                 image_end_id=None, check_roundtrip=True):
        self.tokenizer, self.fast, self.processor = tokenizer, fast, processor
        self.max_length = max_length
        self.check_roundtrip = check_roundtrip
        self.image_token_id = image_token_id
        self.image_end_id = image_end_id
        self.special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        self.chat_template = None
        if SYSTEM_PROMPT == "none":
            tpl = tokenizer.chat_template
            new, n = re.subn(r"set default_system_message = '(?:[^'\\]|\\.)*'", "set default_system_message = ''", tpl)
            require(n == 1, "no_default_system_message_in_template")
            self.chat_template = new

    @classmethod
    def from_snapshot(cls, snapshot, with_processor=True, max_length=MAX_LENGTH, mistral_regex_fix=True):
        import transformers
        from tokenizers import Tokenizer
        kwargs = {"fix_mistral_regex": True} if mistral_regex_fix else {}
        tok = transformers.AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                                         trust_remote_code=False, token=False, **kwargs)
        fast = Tokenizer.from_file(str(snapshot) + "/tokenizer.json")
        proc = img = img_end = None
        if with_processor:
            proc = transformers.AutoProcessor.from_pretrained(str(snapshot), local_files_only=True,
                                                              trust_remote_code=False, token=False, **kwargs)
            img = tok.convert_tokens_to_ids(proc.image_token)
            img_end = tok.convert_tokens_to_ids(proc.image_end_token)
        enc = cls(tok, fast, proc, max_length, img, img_end)
        enc.snapshot = str(snapshot)
        return enc

    # -- helpers
    def _chat_text(self, content, n_images):
        if n_images:
            messages = [{"role": "user", "content": [{"type": "image"} for _ in range(n_images)]
                         + [{"type": "text", "text": content}]}]
            # no enable_thinking here: the Ministral template has no such variable, and the processor's
            # apply_chat_template treats unknown kwargs as processor kwargs (warning on every image encode). The text
            # is identical either way (tests/test_image_batch.py::test_processor_kwargs_accepted).
            kw = {"chat_template": self.chat_template} if self.chat_template else {}
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw), messages
        messages = [{"role": "user", "content": content}]
        kw = {"chat_template": self.chat_template} if self.chat_template else {}
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False, **kw), messages

    def encode(self, req, wording="W0", question_order=None, option_orders=None, preview=True, pil_images=None):
        """Returns a dict (all plain Python ints/lists):
        input_ids, n_tokens, keys (rendered question order), qtype, n_opt,
        q_span[i] = (s, e) question line, p_span[i] = (s, e) preview line or None,
        o_span[i] = [(s, e) per CANONICAL option], o_display[i] = canonical index per display position,
        state_spans = [(s, e), ...] (state text incl. header, plus the image block), roles = run-length
        [(role, start, end)], image info."""
        content, segs = render(req, wording, question_order, option_orders, preview)
        images = req.get("images") or []
        chat, _messages = self._chat_text(content, len(images))
        require(chat.count(content) == 1, "content_not_unique_in_chat")
        c0 = chat.index(content)
        enc = self.fast.encode(chat, add_special_tokens=False)
        ids, offsets = list(enc.ids), list(enc.offsets)
        if self.check_roundtrip and not images:
            ref = self.tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                                     add_generation_prompt=True, enable_thinking=False,
                                                     return_dict=False,
                                                     **({"chat_template": self.chat_template} if self.chat_template else {}))
            require(list(ref) == ids, "roundtrip_mismatch")
        image_span = None
        pixel = None
        if images:
            require(self.processor is not None, "no_processor")
            img, img_end = self.image_token_id, self.image_end_id
            require(ids.count(img) == len(images), "image_placeholder_count")
            if pil_images is None:
                pil_images = decode_images(images)
            # flat add_special_tokens: _merge_kwargs routes it to the tokenizer (no extra BOS; checked below and in
            # tests/test_image_batch.py)
            out = self.processor(text=chat, images=pil_images, return_tensors="pt", add_special_tokens=False)
            expanded = out["input_ids"][0].tolist()
            plain_ids, plain_off = ids, offsets
            first_plain = plain_ids.index(img)
            last_plain = len(plain_ids) - 1 - plain_ids[::-1].index(img)
            require(plain_off[last_plain][1] <= c0, "images_not_before_text")
            require(img in expanded and img_end in expanded, "no_image_tokens_after_processor")
            first_exp = expanded.index(img)
            last_exp = len(expanded) - 1 - expanded[::-1].index(img_end)
            require(expanded[:first_exp] == plain_ids[:first_plain], "image_prefix_mismatch")
            require(expanded[last_exp + 1:] == plain_ids[last_plain + 1:], "image_suffix_mismatch")
            shift = last_exp - last_plain
            image_span = (first_exp, last_exp + 1)
            ids = expanded
            offsets = [None] * len(expanded)
            for i in range(first_plain):
                offsets[i] = plain_off[i]
            for i in range(last_plain + 1, len(plain_ids)):
                offsets[i + shift] = plain_off[i]
            pixel = {"pixel_values": out["pixel_values"], "image_sizes": out.get("image_sizes")}
        n = len(ids)
        require(1 <= n <= self.max_length, "too_long", str(n))

        # char -> segment id map over the content region
        seg_names = []
        seg_ranges = []

        def add(name, rng):
            seg_names.append(name)
            seg_ranges.append((rng[0] + c0, rng[1] + c0))
            return len(seg_names) - 1

        keys = list(question_order) if question_order is not None else list(req["questions"])
        sid_state = add(("state",), segs["state"])
        sid_prev, sid_q, sid_o = {}, {}, {}
        for key in keys:
            if key in segs["preview"]:
                sid_prev[key] = add(("preview", key), segs["preview"][key])
            sid_q[key] = add(("question", key), segs["question"][key])
            sid_o[key] = [add(("option", key, j), r) for j, r in enumerate(segs["option"][key])]
        # sort segment ranges for binary search
        order = sorted(range(len(seg_ranges)), key=lambda i: seg_ranges[i][0])
        starts = [seg_ranges[i][0] for i in order]
        import bisect

        def seg_of(ch):
            k = bisect.bisect_right(starts, ch) - 1
            if k < 0:
                return -1
            sid = order[k]
            a, b = seg_ranges[sid]
            return sid if a <= ch < b else -1

        token_seg = [-1] * n
        content_end = c0 + len(content)
        for t, off in enumerate(offsets):
            if off is None:
                continue
            a, b = off
            if b <= c0 or a >= content_end:
                continue
            ch = a
            while ch < b and chat[ch].isspace():
                ch += 1
            if ch >= b:
                ch = a
            token_seg[t] = seg_of(ch)
            # special tokens must not appear inside the caller's content (e.g. a key spelled like a control token)
            require(ids[t] not in self.special_ids, "special_token_in_content", str(ids[t]))

        # spans per segment + coverage check (every non-space char of a segment lies in one of its tokens)
        spans = {}
        for t, sid in enumerate(token_seg):
            if sid < 0:
                continue
            s = spans.get(sid)
            spans[sid] = (t, t + 1) if s is None else (s[0], t + 1)
        for sid, (a, b) in spans.items():
            require(all(token_seg[t] == sid for t in range(a, b)), "span_not_contiguous", str(seg_names[sid]))
        for sid, (ca, cb) in enumerate(seg_ranges):
            require(sid in spans, "empty_span", str(seg_names[sid]))
            a, b = spans[sid]
            covered_lo = offsets[a][0]
            covered_hi = offsets[b - 1][1]
            first = ca
            while first < cb and chat[first].isspace():
                first += 1
            last = cb
            while last > first and chat[last - 1].isspace():
                last -= 1
            require(covered_lo <= first and covered_hi >= last, "span_straddle", str(seg_names[sid]))

        state_spans = [spans[sid_state]]
        if image_span is not None:
            state_spans.insert(0, image_span)
        out = {"template": TEMPLATE_ID, "wording": wording, "input_ids": ids, "n_tokens": n, "keys": keys,
               "qtype": [req["questions"][k]["type"] for k in keys],
               "n_opt": [len(req["questions"][k]["options"]) for k in keys],
               "q_span": [spans[sid_q[k]] for k in keys],
               "p_span": [spans[sid_prev[k]] if k in sid_prev else None for k in keys],
               "o_span": [[spans[s] for s in sid_o[k]] for k in keys],
               "o_display": [list(option_orders[k]) if option_orders and k in option_orders
                             else list(range(len(req["questions"][k]["options"]))) for k in keys],
               "state_spans": state_spans, "n_images": len(images), "image_tokens": (image_span[1] - image_span[0]
                                                                                     if image_span else 0)}
        if pixel is not None:
            out["_pixel"] = pixel
        return out


def decode_images(data_urls):
    import base64
    import io
    from PIL import Image
    out = []
    for url in data_urls:
        require(isinstance(url, str) and url.startswith("data:image/") and "," in url, "bad_image")
        out.append(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))
    return out


def token_roles(encoded):
    """Per-token role ids (list of length n_tokens)."""
    roles = [ROLE_OTHER] * encoded["n_tokens"]
    for a, b in encoded["state_spans"]:
        for t in range(a, b):
            roles[t] = ROLE_STATE
    for i in range(len(encoded["keys"])):
        if encoded["p_span"][i]:
            a, b = encoded["p_span"][i]
            for t in range(a, b):
                roles[t] = ROLE_PREVIEW
        a, b = encoded["q_span"][i]
        for t in range(a, b):
            roles[t] = ROLE_QUESTION
        for a, b in encoded["o_span"][i]:
            for t in range(a, b):
                roles[t] = ROLE_OPTION
    return roles


def request_from_systemone(body):
    """/v1/systemone request JSON (jev-adapter protocol) -> schema request (canonical option order)."""
    questions = {}
    for key, q in body["questions"].items():
        t = q["type"]
        if t == "choice":
            opts = [{"name": n, "description": d} for n, d in q["criteria"].items()]
        elif t == "noul":
            crit = q.get("criteria") or {}
            opts = [{"name": "true", "description": crit.get("true")}, {"name": "false", "description": crit.get("false")}]
        elif t == "score":
            opts = [{"name": str(i), "description": d} for i, d in enumerate(q["criteria"])]
        else:
            raise EncodeError("bad_type", repr(t))
        questions[key] = {"type": t, "instructions": q["instructions"], "options": opts}
    return {"state": body.get("state", ""), "images": list(body.get("images") or []), "questions": questions}


def entropy_confidence(p):
    h = -sum(x * math.log(x) for x in p if x > 0)
    return min(1.0, max(0.0, 1.0 - h / math.log(len(p))))
