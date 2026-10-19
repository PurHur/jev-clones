"""Decision Index engine for linne-fern-1.0 (transformers, in-process).

    python -m decision_index run --engine linne_engine:LinneEngine --option model=beyoru/linne-fern-1-0 --option device=cuda:0

Prompt = the generic /v1/systemone layout the model was trained on: one user message with the state (JSON compact when not a
string), a blank line, then
  choice : "Question: <instructions>", one line "<label>: <name> - <description>" per option, "Answer with the letter of one option only."
  noul   : "Is the following true? <instructions>", "yes: <true>", "no: <false>", "Answer with yes or no only."
Chat template with thinking off; the answer is the softmax (temperature 1) over the next-token logits of the labels.
Labels are A..Z, or for more than 26 options two-letter labels (AA, AB, ...) kept only when they are one distinct token.
Questions of one row are scored together in length-sorted batches under a token budget; right padding keeps every
sequence's own positions unchanged (causal attention, recurrent layers run left to right).
"""
import json
import string

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decision_index.engines.base import Engine, Unsupported

PAIRS = [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]


def render_text(v):
    if v is None:
        return ""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def blank(v):
    return not (v.strip() if isinstance(v, str) else v)


def render(state, q, labels):
    text = render_text(state)
    qt = "" if blank(q.get("instructions")) else render_text(q.get("instructions"))
    if q["type"] == "choice":
        lines = [f"Question: {qt}"] if qt else []
        for lab, (name, det) in zip(labels, q["criteria"].items()):
            d = render_text(det)
            lines.append(f"{lab}: {name} - {d}" if d else f"{lab}: {name}")
        lines.append("Answer with the letter of one option only.")
    else:
        c = q.get("criteria") or {}
        lines = [f"Is the following true? {qt}" if qt else "Is the following true?"]
        for lab, det in (("yes", c.get("true")), ("no", c.get("false"))):
            d = render_text(det)
            if d:
                lines.append(f"{lab}: {d}")
        lines.append("Answer with yes or no only.")
    return "\n".join([text, "", *lines])


class LinneEngine(Engine):
    name = "linne-fern-1-0"
    latency = "In-process request wall time including prompt construction and tokenization; excludes model loading."

    def __init__(self, model="beyoru/linne-fern-1-0", device="cuda", dtype="bfloat16", token_budget=32768, max_tokens=65536,
                 noncausal=False, chat_template=None, **options):
        super().__init__(**options)
        self.tok = AutoTokenizer.from_pretrained(model)
        if chat_template:
            self.tok.chat_template = open(chat_template).read()
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype), attn_implementation="sdpa").to(device).eval()
        cfg = self.model.config.get_text_config() if hasattr(self.model.config, "get_text_config") else self.model.config
        self.cap = float(getattr(cfg, "final_logit_softcapping", None) or 0)  # Gemma: logit qua cap*tanh(l/cap) nhu luc serve
        self.noncausal = str(noncausal).lower() in ("1", "true", "yes")
        if self.noncausal:  # lop full_attention hai chieu (kieu pplx-decider v1.1), lop khac giu nguyen
            from nc_hook import enable_noncausal
            enable_noncausal(self.model.model)
        self.device, self.budget, self.max_tokens = device, int(token_budget), int(max_tokens)
        self.added = set(self.tok.get_added_vocab().values())
        self.pairs = None
        self.provenance = {"model": model, "dtype": dtype, "prompt": "generic systemone layout, thinking off", "temperature": 1.0,
                           "noncausal_full_attention": self.noncausal, "softcap": self.cap}

    def prompt(self, content):
        return self.tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def label_ids(self, prompt, ids, labels):
        """Token of each label after the prompt. Like the server, check on the text after the prompt's last added token
        (it tokenizes on its own), so the cost does not grow with the prompt; fall back to the whole prompt otherwise."""
        last = max((i for i, t in enumerate(ids) if t in self.added), default=None)
        text, tids = prompt, ids
        if last is not None:
            suffix = self.tok.decode(ids[last + 1:])
            if self.tok(suffix, add_special_tokens=False)["input_ids"] == ids[last + 1:]:
                text, tids = suffix, ids[last + 1:]
        out = []
        for lab in labels:
            x = self.tok(text + lab, add_special_tokens=False)["input_ids"]
            if len(x) != len(tids) + 1 or x[:-1] != tids or x[-1] in out:
                raise Unsupported(f"label {lab!r} is not one distinct token after the prompt")
            out.append(x[-1])
        return out

    def pair_labels(self):
        """Two-letter labels that are distinct single tokens after the generation prompt (text after its last added token)."""
        if self.pairs is None:
            p = self.prompt("x")
            ids = self.tok(p, add_special_tokens=False)["input_ids"]
            last = max(i for i, t in enumerate(ids) if t in self.added)
            suffix = self.tok.decode(ids[last + 1:])
            sids = self.tok(suffix, add_special_tokens=False)["input_ids"]
            self.pairs, seen = [], set()
            for lab in PAIRS:
                x = self.tok(suffix + lab, add_special_tokens=False)["input_ids"]
                if len(x) == len(sids) + 1 and x[:-1] == sids and x[-1] not in seen:
                    self.pairs.append(lab); seen.add(x[-1])
        return self.pairs

    @torch.no_grad()
    def __call__(self, state, questions):
        items = []
        for key, q in questions.items():
            if q["type"] == "choice":
                n = len(q["criteria"])
                labels = list(string.ascii_uppercase[:n]) if n <= 26 else self.pair_labels()[:n]
                if len(labels) < n:
                    raise Unsupported(f"{n} options, at most {len(self.pair_labels())} labels")
            elif q["type"] == "noul":
                labels = ["yes", "no"]
            else:
                raise Unsupported("unsupported question type " + q["type"])
            p = self.prompt(render(state, q, labels))
            ids = self.tok(p, add_special_tokens=False)["input_ids"]
            if len(ids) + 1 > self.max_tokens:
                raise Unsupported(f"the prompt has {len(ids)} tokens, over the context length {self.max_tokens}")
            items.append((key, q, ids, self.label_ids(p, ids, labels)))
        probs, batch, mx = {}, [], 0
        order = sorted(items, key=lambda x: -len(x[2]))
        for it in order + [None]:
            if it is not None and (not batch or max(mx, len(it[2])) * (len(batch) + 1) <= self.budget):
                batch.append(it); mx = max(mx, len(it[2])); continue
            if batch:
                L = max(len(x[2]) for x in batch)
                ids = torch.full((len(batch), L), self.tok.pad_token_id or 0, dtype=torch.long)
                am = torch.zeros((len(batch), L), dtype=torch.long)
                for i, x in enumerate(batch):
                    ids[i, :len(x[2])] = torch.tensor(x[2]); am[i, :len(x[2])] = 1
                ids, am = ids.to(self.device), am.to(self.device)
                h = self.model.model(input_ids=ids, attention_mask=am).last_hidden_state
                last = h[torch.arange(len(batch), device=self.device), am.sum(1) - 1]
                W = self.model.lm_head.weight
                for i, x in enumerate(batch):
                    lg = last[i].float() @ W[torch.tensor(x[3], device=self.device)].float().T
                    if self.cap:
                        lg = self.cap * torch.tanh(lg / self.cap)
                    probs[x[0]] = torch.softmax(lg, 0).tolist()
            batch, mx = ([it], len(it[2])) if it is not None else ([], 0)
        answers = {}
        for key, q, _, _ in items:
            pr = probs[key]
            if q["type"] == "choice":
                names = list(q["criteria"])
                answers[key] = {"type": "choice", "choice": names[max(range(len(pr)), key=pr.__getitem__)],
                                "probabilities": dict(zip(names, pr))}
            else:
                answers[key] = {"type": "noul", "noul": pr[0]}
        return {"model": self.name, "answers": answers}, None

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
