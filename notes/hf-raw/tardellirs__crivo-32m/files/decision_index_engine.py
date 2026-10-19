"""Decision Index engine, optimized: batched tokenization (state once per request), optional test-time option
permutations (TTA), fp32 head, configurable attention. Nothing is truncated; a question above max_length raises
Unsupported.

    python -m decision_index run --engine engine2:DecisionEngine2 --option path=<ckpt> [--option tta=4] ...
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

from decision_index.engines import Engine, Unsupported

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import QUESTION_TYPES, collate, load, option_texts, render  # noqa: E402


def permutations(n: int, k: int) -> list[list[int]]:
    """K deterministic option orders for a question with n options (identity first; same for every benchmark)."""
    perms = [list(range(n))]
    for j in range(1, k):
        if n == 2:
            p = [1, 0] if j % 2 else [0, 1]
        else:
            p = list(range(n))
            random.Random(f"tta:{j}:{n}").shuffle(p)
        perms.append(p)
    return perms


class DecisionEngine2(Engine):
    name = "decision-engine2"
    latency = ("In-process request wall time: tokenization plus encoder passes for all questions of the request "
               "(batched, K option orders if tta>1); excludes model loading.")

    def __init__(self, path: str = str(Path(__file__).resolve().parent), device: str | None = None,
                 batch_size: int = 32, max_length: int = 32768, token_budget: int = 65536, tta: int = 1,
                 head_fp32: bool = True, attn: str = "sdpa", **options):
        super().__init__(**options)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model, self.tok, self.extra = load(path, device=self.device, attn=attn)
            self.attn = attn
        except Exception as e:  # e.g. flash-attn kernels unavailable on this machine
            self.model, self.tok, self.extra = load(path, device=self.device, attn="sdpa")
            self.attn = f"sdpa (fallback from {attn}: {type(e).__name__})"
        self.batch_size, self.token_budget, self.max_length = int(batch_size), int(token_budget), int(max_length)
        self.tta = int(tta)
        self.head_fp32 = head_fp32 if isinstance(head_fp32, bool) else str(head_fp32).lower() in ("1", "true", "yes")
        self.opt_id = self.tok.convert_tokens_to_ids("[OPT]")
        self.provenance = {"path": str(path), "device": self.device, "max_length": self.max_length,
                           "token_budget": self.token_budget, "batch_size": self.batch_size, "tta": self.tta,
                           "head_fp32": self.head_fp32, "attention": self.attn,
                           "truncation": "none: questions above max_length raise Unsupported"}

    def _ids_many(self, texts: list[str]) -> list[list[int]]:
        nonempty = [t for t in texts if t]
        enc = self.tok(nonempty, add_special_tokens=False)["input_ids"] if nonempty else []
        it = iter(enc)
        return [next(it) if t else [] for t in texts]

    @torch.no_grad()
    def __call__(self, state, questions):
        t = self.tok
        keys = list(questions)
        rendered, meta = [], []
        for k in keys:
            q = questions[k]
            if q["type"] not in ("choice", "noul"):
                raise Unsupported(f"question type {q['type']}")
            texts, labels = option_texts(q["type"], q.get("criteria"), q.get("instructions"))
            rendered.append((render(q.get("instructions")), texts))
            meta.append((k, q["type"], labels, len(texts)))
        # one tokenizer call for the state, one for all instructions and options
        state_ids = self._ids_many([render(state)])[0]
        flat = [ins for ins, _ in rendered] + [o for _, texts in rendered for o in texts]
        toks = self._ids_many(flat)
        ins_ids, opt_ids, c = toks[:len(rendered)], [], len(rendered)
        for _, texts in rendered:
            opt_ids.append(toks[c:c + len(texts)]); c += len(texts)
        items, owners = [], []
        for qi, (k, typ, labels, n) in enumerate(meta):
            prefix = [t.cls_token_id, *ins_ids[qi], t.sep_token_id, *state_ids, t.sep_token_id]
            for perm in permutations(n, self.tta):
                ids, pos = list(prefix), []
                for oi in perm:
                    pos.append(len(ids)); ids += [self.opt_id, *opt_ids[qi][oi]]
                if len(ids) > self.max_length:
                    raise Unsupported(f"question needs {len(ids)} tokens, above max_length={self.max_length}")
                items.append((ids, pos, QUESTION_TYPES[typ])); owners.append((qi, perm))
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        chunks, cur = [], []
        for i in order:
            if cur and (len(cur) >= self.batch_size or len(items[i][0]) * (len(cur) + 1) > self.token_budget):
                chunks.append(cur); cur = []
            cur.append(i)
        if cur:
            chunks.append(cur)
        probs = [[0.0] * m[3] for m in meta]
        for chunk in chunks:
            b = {kk: v.to(self.device) for kk, v in collate([items[i] for i in chunk], t.pad_token_id).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
                logits = self.model(b["input_ids"], b["attention_mask"], b["opt_positions"], b["opt_mask"], b["qtype"],
                                    head_fp32=self.head_fp32)
            p_all = logits.double().softmax(-1).cpu().tolist()   # one device->host copy per batch
            for j, i in enumerate(chunk):
                qi, perm = owners[i]
                for slot, oi in enumerate(perm):
                    probs[qi][oi] += p_all[j][slot] / self.tta
        answers = {}
        for (k, typ, labels, n), p in zip(meta, probs):
            s = sum(p); p = [x / s for x in p]
            if typ == "noul":
                answers[k] = {"type": "noul", "noul": float(p[1])}
            else:
                best = max(range(n), key=p.__getitem__)
                answers[k] = {"type": "choice", "choice": labels[best], "probabilities": dict(zip(labels, map(float, p)))}
        return {"model": self.name, "answers": answers}, None

    def synchronize(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()


DecisionEngine = DecisionEngine2  # name used in the kit instructions (decision_index_engine:DecisionEngine)
