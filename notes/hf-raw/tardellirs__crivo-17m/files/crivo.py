"""Crivo-400M: standalone inference (no Decision Index kit needed).

    from crivo import Crivo
    model = Crivo()                       # loads from this folder (or Crivo("tardellirs/crivo-400m"))
    out = model.answer(
        state={"ticket": "My card was charged twice for the same order."},
        questions={
            "intent": {"type": "choice", "instructions": "What does the customer want?",
                       "criteria": {"refund": "Money back for the duplicate charge",
                                    "cancel": "Cancel the order", "info": "Just asking for information"}},
            "urgent": {"type": "noul", "instructions": "The customer needs a fast response."},
        })
    # {"intent": {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.96, "cancel": 0.03, "info": 0.01}},
    #  "urgent": {"type": "noul", "noul": 0.12}}

Questions are typed decisions: `choice` (pick one key of `criteria`), `noul` (probability that the statement in
`instructions` is true; optional criteria {"true": ..., "false": ...}) and `score` (criteria = ordered levels, answered
with the distribution and its expected level). Nothing is truncated: a question longer than `max_length` tokens raises
ValueError.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import QUESTION_TYPES, Encoder, collate, load, option_texts  # noqa: E402


def _resolve(path_or_repo: str | Path, revision: str | None = None) -> Path:
    p = Path(path_or_repo)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(path_or_repo), revision=revision))


class Crivo:
    def __init__(self, path_or_repo: str | Path = Path(__file__).resolve().parent, device: str | None = None,
                 revision: str | None = None, max_length: int = 32768, batch_size: int = 32, token_budget: int = 65536):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tok, self.extra = load(_resolve(path_or_repo, revision), device=self.device)
        self.enc = Encoder(self.tok, max_length=max_length)
        self.batch_size, self.token_budget = batch_size, token_budget

    @torch.no_grad()
    def answer(self, state: Any, questions: dict[str, dict]) -> dict[str, dict]:
        items, meta = [], []
        for key, q in questions.items():
            texts, labels = option_texts(q["type"], q.get("criteria"), q.get("instructions"))
            ids, pos = self.enc.encode(q.get("instructions"), state, texts)
            if len(ids) > self.enc.max_length:
                raise ValueError(f"question {key!r} needs {len(ids)} tokens, above max_length={self.enc.max_length}")
            items.append((ids, pos, QUESTION_TYPES[q["type"]]))
            meta.append((key, q["type"], labels, q.get("criteria")))
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        batches, cur = [], []
        for i in order:
            if cur and (len(cur) >= self.batch_size or len(items[i][0]) * (len(cur) + 1) > self.token_budget):
                batches.append(cur); cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        out = {}
        for batch in batches:
            b = {k: v.to(self.device) for k, v in collate([items[i] for i in batch], self.tok.pad_token_id).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
                logits = self.model(b["input_ids"], b["attention_mask"], b["opt_positions"], b["opt_mask"], b["qtype"],
                                    head_fp32=True)
            for j, i in enumerate(batch):
                key, typ, labels, criteria = meta[i]
                p = logits[j, :len(items[i][1])].double().softmax(-1).cpu().tolist()
                if typ == "noul":
                    out[key] = {"type": "noul", "noul": p[1]}
                elif typ == "choice":
                    best = max(range(len(p)), key=p.__getitem__)
                    out[key] = {"type": "choice", "choice": labels[best], "probabilities": dict(zip(labels, p))}
                else:
                    out[key] = {"type": "score", "score": sum(k * x for k, x in enumerate(p)),
                                "probabilities": {str(k): x for k, x in enumerate(p)}}
        return out


if __name__ == "__main__":
    m = Crivo()
    print(m.answer({"ticket": "My card was charged twice for the same order."},
                   {"intent": {"type": "choice", "instructions": "What does the customer want?",
                               "criteria": {"refund": "Money back for the duplicate charge", "cancel": "Cancel the order",
                                            "info": "Just asking for information"}},
                    "urgent": {"type": "noul", "instructions": "The customer needs a fast response."}}))
