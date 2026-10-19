"""Mixture-of-products joint and exact, coherent query answering from it.

    p(x_1..x_n) = sum_m w_m prod_i p_m(x_i)

Every query (marginal, conjunction, negation, conditional) is answered in closed form from this ONE joint, so the
answers of an input are coherent by construction (no Dutch book is possible against them):

    P(E) for a box event E = prod_i S_i  = sum_m w_m prod_i sum_{j in S_i} p_m(j | i)
    P(not E) = 1 - P(E);   P(E | G) = P(E and G) / P(G)
"""
from __future__ import annotations

import torch


class MixtureJoint:
    """One input's joint. logw: [M] log mixture weights (normalized); logp: list over questions of [M, K_i]
    per-component log-probabilities (normalized over the K_i options, in the caller's option order)."""

    def __init__(self, logw: torch.Tensor, logp: list[torch.Tensor]):
        self.logw = logw
        self.logp = logp

    def double(self) -> "MixtureJoint":
        return MixtureJoint(self.logw.detach().double(), [lp.detach().double() for lp in self.logp])

    @property
    def sizes(self):
        return [lp.shape[-1] for lp in self.logp]

    def log_joint(self) -> torch.Tensor:
        """Materialized log joint, shape [K_1, ..., K_n] (row-major over question order)."""
        M, n = self.logw.shape[0], len(self.logp)
        acc = self.logw.view(M, *([1] * n))
        for i, lp in enumerate(self.logp):
            shape = [M] + [1] * n
            shape[i + 1] = lp.shape[-1]
            acc = acc + lp.view(shape)
        return torch.logsumexp(acc, 0)

    def joint(self) -> torch.Tensor:
        return self.log_joint().exp()

    def marginal(self, i: int) -> torch.Tensor:
        return (self.logw[:, None] + self.logp[i]).logsumexp(0).exp()

    def log_prob_assign(self, golds) -> torch.Tensor:
        """log p(x_i = golds[i] for all i with golds[i] >= 0); questions with -1/None are marginalized out."""
        acc = self.logw
        for i, g in enumerate(golds):
            if g is not None and g >= 0:
                acc = acc + self.logp[i][:, g]
        return acc.logsumexp(0)

    def prob_box(self, sets: dict[int, list[int]]) -> torch.Tensor:
        """P(x_i in S_i for all i in sets). Empty S_i -> 0."""
        acc = self.logw.clone()
        for i, S in sets.items():
            if len(S) == 0:
                return acc.new_zeros(())
            idx = torch.as_tensor(sorted(set(S)), device=acc.device, dtype=torch.long)
            acc = acc + self.logp[i][:, idx].logsumexp(-1)
        return acc.logsumexp(0).exp()

    def answer(self, query: dict, var_index: dict[str, int]) -> list[float]:
        """BookieBench query semantics. query = {"kind": "marginal", "var": name} -> [p_1..p_K];
        {"kind": "noul", "event": {name: [option idx, ...]}, "neg": bool} -> [P(event)] (or 1 - P);
        {"kind": "cond", "event": {...}, "given": {...}, "neg": bool} -> [P(event | given)] (or 1 - P)."""
        kind = query["kind"]
        if kind == "marginal":
            return self.marginal(var_index[query["var"]]).detach().tolist()
        ev = {var_index[k]: list(v) for k, v in query.get("event", {}).items()}
        if kind == "noul":
            p = self.prob_box(ev)
            if query.get("neg"):
                p = 1 - p
            return [float(p.detach())]
        if kind == "cond":
            gv = {var_index[k]: list(v) for k, v in query["given"].items()}
            both = dict(gv)
            for i, S in ev.items():
                both[i] = sorted(set(S) & set(both[i])) if i in both else S
            pg = self.prob_box(gv)
            p = self.prob_box(both) / pg.clamp_min(1e-30)
            if query.get("neg"):
                p = 1 - p
            return [float(p.detach().clamp(0, 1))]
        raise ValueError(f"unknown query kind {kind}")
