# -*- coding: utf-8 -*-
"""ZTC-Judge-27B — 남의 모델이 쓴 답을 읽고 믿어도 되는지 판정한다.

    글자를 한 개도 생성하지 않는다. 한 번 읽고, 곱셈 한 번 한다.

    from ztc_judge import ZTCJudge
    j = ZTCJudge.from_pretrained("FINAL-Bench/ZTC-Judge-27B")
    j.score(question, answer, domain="과학추론")
"""
import json
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class ZTCJudge:
    def __init__(self, path, device="cuda", dtype=torch.bfloat16):
        self.cfg = json.load(open(os.path.join(path, "ztc_config.json"), encoding="utf-8"))
        p = dict(np.load(os.path.join(path, self.cfg["probe_file"]), allow_pickle=True))
        self.w = p["w"].astype(np.float64)
        self.mu = p["mu"].astype(np.float64)
        self.sd = p["sd"].astype(np.float64)
        self.per = self.cfg["measured"]["per_domain_lodo"]
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModel.from_pretrained(path, dtype=dtype,
                                               low_cpu_mem_usage=True,
                                               attn_implementation="sdpa").eval().to(device)
        self.device = device

    @classmethod
    def from_pretrained(cls, repo, token=None, **kw):
        from huggingface_hub import snapshot_download
        return cls(snapshot_download(repo, token=token), **kw)

    @torch.no_grad()
    def _hidden(self, texts):
        b = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                     max_length=self.cfg["max_length"])
        ids = b["input_ids"].to(self.device)
        am = b["attention_mask"].to(self.device)
        h = self.model(input_ids=ids, attention_mask=am).last_hidden_state
        idx = am.sum(1) - 1
        return torch.stack([h[i, idx[i]] for i in range(len(texts))]).float().cpu().numpy()

    def _verdict(self, s):
        t = self.cfg["thresholds"]
        return "검토 요망" if s <= t["review"] else ("이상 없음" if s >= t["clear"] else "판단 보류")

    def score(self, question, answer, domain=None):
        """점수가 높을수록 '맞을 가능성이 높다'. 안 재본 분야에는 신뢰도를 주지 않는다."""
        v = self._hidden([self.cfg["template"] % (question, answer)])[0].astype(np.float64)
        s = float(((v - self.mu) / self.sd) @ self.w)
        return {"score": round(s, 2), "verdict": self._verdict(s),
                "domain_auc": self.per.get(domain),      # None = 이 분야는 측정 안 됨
                "overall_auc": self.cfg["measured"]["cv_auc"],
                "generated_tokens": 0}

    def score_batch(self, pairs, domain=None):
        """🔑 생성이 없고 읽기만이라 묶을수록 그대로 빨라진다 (VKIE)."""
        V = self._hidden([self.cfg["template"] % (q, a) for q, a in pairs]).astype(np.float64)
        S = ((V - self.mu) / self.sd) @ self.w
        return [{"score": round(float(s), 2), "verdict": self._verdict(s),
                 "domain_auc": self.per.get(domain)} for s in S]
