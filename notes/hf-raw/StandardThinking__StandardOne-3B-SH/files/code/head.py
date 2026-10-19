"""Joint schema head (DESIGN.md section 2.2). Own design; provenance in DESIGN.md section 0.

Reads tapped backbone hidden states for ONE request at a time (a micro-batch is a Python loop over requests; the
head is small) and returns one logit vector per question over that question's options in CANONICAL order.

Nodes: one per question and one per option. No positional encoding over nodes, attention pooling inside spans and
relation-type attention biases that depend only on structure make the head exactly equivariant to the order of
choice/noul options and of questions (tests/test_equivariance.py). Score levels carry an ordinal encoding and are
deliberately not permutation-equivariant.
"""
import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

QTYPE_ID = {"choice": 0, "noul": 1, "score": 2}
# relation types for node pairs (i attends to j)
REL_SELF, REL_SIBLING, REL_OPT_TO_OWN_Q, REL_Q_TO_OWN_OPT, REL_OPT_TO_OTHER_OPT, REL_OPT_TO_OTHER_Q, \
    REL_Q_TO_OTHER_Q, REL_Q_TO_OTHER_OPT = range(8)
N_REL = 8
ORD_FEATS = 33


@dataclass
class HeadConfig:
    hidden_size: int = 4096
    n_taps: int = 2
    d: int = 1024
    heads: int = 16
    ffn: int = 2816
    route_blocks: int = 2
    interact_blocks: int = 2
    dropout: float = 0.1
    n_roles: int = 5
    ord_hidden: int = 256
    max_nodes: int = 4096
    grad_checkpoint: bool = True
    taps: list = field(default_factory=lambda: ["final", 25])
    template: str = "schema-v1"

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d, hidden, dropout):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Attention(nn.Module):
    """Multi-head attention (queries from nodes, keys/values from `kv`), SDPA with optional additive bias."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.p = dropout

    def _chunk(self, q, part, p):
        k = self.k(part).view(-1, self.h, self.dh).transpose(0, 1).unsqueeze(0)
        v = self.v(part).view(-1, self.h, self.dh).transpose(0, 1).unsqueeze(0)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        lse = torch.logsumexp(scores, -1, keepdim=True)
        probs = F.dropout(torch.softmax(scores, -1), p, self.training)
        return probs @ v, lse

    def forward(self, x, kv, bias=None, kv_chunk=0):
        n, m = x.shape[0], kv.shape[0]
        q = self.q(x).view(n, self.h, self.dh).transpose(0, 1).unsqueeze(0)
        p = self.p if self.training else 0.0
        if kv_chunk and m > kv_chunk and bias is None:
            # exact chunked attention over keys (log-sum-exp merge) to bound K/V memory at long contexts.
            # Dropout is applied to each chunk's attention PROBABILITIES (as SDPA's dropout_p): the global weight of
            # key j is softmax_chunk(j) * exp(lse_chunk - lse_all), so dropping the chunk-local probability with
            # scale 1/(1-p) is the same random variable as dropping the global one. When training, every chunk is
            # recomputed in backward (checkpoint), so its scores/probabilities are not kept for the whole sequence.
            outs, lses = [], []
            use_ckpt = self.training and torch.is_grad_enabled()
            for s in range(0, m, kv_chunk):
                part = kv[s:s + kv_chunk]
                if use_ckpt:
                    o, lse = checkpoint(self._chunk, q, part, p, use_reentrant=False)
                else:
                    o, lse = self._chunk(q, part, p)
                outs.append(o)
                lses.append(lse)
            lse_all = torch.logsumexp(torch.cat(lses, -1), -1, keepdim=True)
            out = sum(o * torch.exp(l - lse_all) for o, l in zip(outs, lses))
        else:
            k = self.k(kv).view(m, self.h, self.dh).transpose(0, 1).unsqueeze(0)
            v = self.v(kv).view(m, self.h, self.dh).transpose(0, 1).unsqueeze(0)
            mask = None if bias is None else bias.unsqueeze(0)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        return self.o(out.squeeze(0).transpose(0, 1).reshape(n, -1))


class RouteBlock(nn.Module):
    """Stage 1 (evidence routing): nodes cross-attend over the state tokens, then FFN."""

    def __init__(self, cfg):
        super().__init__()
        self.n1, self.n2, self.nkv = RMSNorm(cfg.d), RMSNorm(cfg.d), RMSNorm(cfg.d)
        self.cross = Attention(cfg.d, cfg.heads, cfg.dropout)
        self.ffn = SwiGLU(cfg.d, cfg.ffn, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, state, kv_chunk=0):
        x = x + self.drop(self.cross(self.n1(x), self.nkv(state), kv_chunk=kv_chunk))
        return x + self.ffn(self.n2(x))


class InteractBlock(nn.Module):
    """Stage 2 (field interaction): node self-attention with relation-type biases, cross-attention back to the
    state tokens, FFN."""

    def __init__(self, cfg):
        super().__init__()
        self.n1, self.n2, self.n3, self.nkv = RMSNorm(cfg.d), RMSNorm(cfg.d), RMSNorm(cfg.d), RMSNorm(cfg.d)
        self.self_attn = Attention(cfg.d, cfg.heads, cfg.dropout)
        self.cross = Attention(cfg.d, cfg.heads, cfg.dropout)
        self.rel_bias = nn.Parameter(torch.zeros(N_REL, cfg.heads))
        self.ffn = SwiGLU(cfg.d, cfg.ffn, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, state, rel, kv_chunk=0):
        bias = self.rel_bias[rel].permute(2, 0, 1)  # [heads, N, N]
        y = self.n1(x)
        x = x + self.drop(self.self_attn(y, y, bias=bias))
        x = x + self.drop(self.cross(self.n2(x), self.nkv(state), kv_chunk=kv_chunk))
        return x + self.ffn(self.n3(x))


def ordinal_features(k, K):
    """33 features of score level k of K (2 <= K <= 10): position, scale size, one-hots, sin/cos."""
    pos = k / (K - 1)
    feats = [pos, K / 10.0]
    feats += [1.0 if i == k else 0.0 for i in range(10)]
    feats += [1.0 if K == i else 0.0 for i in range(2, 11)]
    for f in range(1, 7):
        feats += [math.sin(math.pi * f * pos), math.cos(math.pi * f * pos)]
    assert len(feats) == ORD_FEATS
    return feats


class JointSchemaHead(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.tap_norms = nn.ModuleList([RMSNorm(cfg.hidden_size) for _ in range(cfg.n_taps)])
        self.tap_mix = nn.Parameter(torch.zeros(cfg.n_taps))
        self.tap_gain = nn.Parameter(torch.ones(1))
        self.in_proj = nn.Linear(cfg.hidden_size, d, bias=False)
        self.role_emb = nn.Embedding(cfg.n_roles, d)
        self.qtype_emb = nn.Embedding(3, d)
        self.kind_emb = nn.Embedding(2, d)  # 0 question node, 1 option node
        self.pool_query = nn.Parameter(torch.randn(2, d) / math.sqrt(d))
        self.pool_key = nn.Linear(d, d, bias=False)
        self.pool_out = nn.Linear(2 * d, d, bias=False)
        self.ord_mlp = nn.Sequential(nn.Linear(ORD_FEATS, cfg.ord_hidden), nn.GELU(), nn.Linear(cfg.ord_hidden, d))
        self.route = nn.ModuleList([RouteBlock(cfg) for _ in range(cfg.route_blocks)])
        self.interact = nn.ModuleList([InteractBlock(cfg) for _ in range(cfg.interact_blocks)])
        self.final_norm = RMSNorm(d)
        self.scorer = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.log_temp = nn.Parameter(torch.zeros(3))
        self.kv_chunk = 0
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------ structure
    @staticmethod
    def plan(enc, device):
        """Index tensors for one encoded request (see render.Encoder.encode)."""
        node_kind, node_q, node_qtype, node_ord = [], [], [], []
        pool_node, pool_tok, last_tok = [], [], []
        q_node, opt_nodes = [], []
        nq = len(enc["keys"])
        for i in range(nq):
            qt = enc["qtype"][i]
            idx = len(node_kind)
            q_node.append(idx)
            node_kind.append(0); node_q.append(i); node_qtype.append(qt); node_ord.append(None)
            toks = list(range(*enc["q_span"][i]))
            if enc["p_span"][i]:
                toks = list(range(*enc["p_span"][i])) + toks
            pool_node += [idx] * len(toks); pool_tok += toks
            last_tok.append(enc["q_span"][i][1] - 1)
            ids = []
            K = enc["n_opt"][i]
            for j in range(K):
                idx = len(node_kind)
                ids.append(idx)
                node_kind.append(1); node_q.append(i); node_qtype.append(qt)
                node_ord.append((j, K) if qt == "score" else None)
                a, b = enc["o_span"][i][j]
                pool_node += [idx] * (b - a); pool_tok += list(range(a, b))
                last_tok.append(b - 1)
            opt_nodes.append(ids)
        n = len(node_kind)
        kind = torch.tensor(node_kind)
        qn = torch.tensor(node_q)
        same_q = qn[:, None] == qn[None, :]
        ki, kj = kind[:, None], kind[None, :]
        rel = torch.empty(n, n, dtype=torch.long)
        rel[(ki == 1) & (kj == 1) & same_q] = REL_SIBLING
        rel[(ki == 1) & (kj == 0) & same_q] = REL_OPT_TO_OWN_Q
        rel[(ki == 0) & (kj == 1) & same_q] = REL_Q_TO_OWN_OPT
        rel[(ki == 1) & (kj == 1) & ~same_q] = REL_OPT_TO_OTHER_OPT
        rel[(ki == 1) & (kj == 0) & ~same_q] = REL_OPT_TO_OTHER_Q
        rel[(ki == 0) & (kj == 0)] = REL_Q_TO_OTHER_Q
        rel[(ki == 0) & (kj == 1) & ~same_q] = REL_Q_TO_OTHER_OPT
        rel.fill_diagonal_(REL_SELF)
        ords = [ordinal_features(*o) if o else [0.0] * ORD_FEATS for o in node_ord]
        state_tok = [t for a, b in enc["state_spans"] for t in range(a, b)]
        p = {"n": n, "kind": kind.to(device), "qtype": torch.tensor([QTYPE_ID[t] for t in node_qtype]).to(device),
             "is_score": torch.tensor([o is not None for o in node_ord]).to(device),
             "ord": torch.tensor(ords, dtype=torch.float32).to(device), "rel": rel.to(device),
             "pool_node": torch.tensor(pool_node).to(device), "pool_tok": torch.tensor(pool_tok).to(device),
             "last_tok": torch.tensor(last_tok).to(device), "q_node": q_node, "opt_nodes": opt_nodes,
             "state_tok": torch.tensor(state_tok).to(device), "qtypes": list(enc["qtype"])}
        return p

    # ------------------------------------------------------------------ forward
    def mix_taps(self, taps):
        """taps: list of [T, hidden] tensors (any float dtype) -> [T, hidden] in the head's dtype (fp32)."""
        w = torch.softmax(self.tap_mix, 0)
        dtype = self.in_proj.weight.dtype
        out = 0
        for i, t in enumerate(taps):
            out = out + w[i] * self.tap_norms[i](t.to(dtype))
        return out * self.tap_gain

    def _token_proj(self, roles, *taps):
        return self.in_proj(self.mix_taps(list(taps))) + self.role_emb(roles)

    def project_tokens(self, taps, roles, use_ckpt, chunk=8192):
        """Per-token tap mix + in-proj + role embedding, in token chunks (each checkpointed when training), so the
        fp32 intermediates of only one chunk exist at a time (forward and recompute)."""
        outs = []
        for a in range(0, roles.shape[0], chunk):
            args = (roles[a:a + chunk], *[t[a:a + chunk] for t in taps])
            outs.append(checkpoint(self._token_proj, *args, use_reentrant=False) if use_ckpt else self._token_proj(*args))
        return outs[0] if len(outs) == 1 else torch.cat(outs, 0)

    def forward_one(self, taps, enc, roles):
        """taps: list of [T, hidden]; enc: encoded request; roles: LongTensor [T]. Returns list of logit vectors
        (one per question in enc['keys'] order), each over CANONICAL option order, already divided by the
        per-type temperature."""
        cfg = self.cfg
        dev = taps[0].device
        p = self.plan(enc, dev)
        require_nodes = p["n"] <= cfg.max_nodes
        if not require_nodes:
            raise ValueError(f"too many nodes: {p['n']} > {cfg.max_nodes}")
        use_ckpt = cfg.grad_checkpoint and self.training and torch.is_grad_enabled()
        # token-level projection: chunked + checkpointed so only the (already alive, bf16) tapped states are kept for
        # backward, not ~4 fp32 [T, hidden] intermediates per tap (about 2 GB per tap at 32k tokens)
        x_tok = self.project_tokens(taps, roles, use_ckpt)
        state = x_tok.index_select(0, p["state_tok"])
        # attention pooling inside spans (segment softmax via scatter)
        toks = x_tok.index_select(0, p["pool_tok"])
        node_of = p["pool_node"]
        kind_of = p["kind"].index_select(0, node_of)
        query = self.pool_query.index_select(0, kind_of)
        scores = (self.pool_key(toks) * query).sum(-1) / math.sqrt(cfg.d)
        mx = torch.full((p["n"],), -1e30, device=dev, dtype=scores.dtype)
        mx = mx.scatter_reduce(0, node_of, scores, reduce="amax", include_self=True)
        e = torch.exp(scores - mx.index_select(0, node_of))
        den = torch.zeros(p["n"], device=dev, dtype=e.dtype).index_add(0, node_of, e)
        w = e / den.index_select(0, node_of)
        pooled = torch.zeros(p["n"], cfg.d, device=dev, dtype=toks.dtype).index_add(0, node_of, toks * w[:, None])
        last = x_tok.index_select(0, p["last_tok"])
        x = self.pool_out(torch.cat([pooled, last], -1))
        x = x + self.qtype_emb(p["qtype"]) + self.kind_emb(p["kind"])
        x = x + self.ord_mlp(p["ord"].to(x.dtype)) * p["is_score"][:, None].to(x.dtype)
        for blk in self.route:
            x = checkpoint(blk, x, state, self.kv_chunk, use_reentrant=False) if use_ckpt else blk(x, state, self.kv_chunk)
        for blk in self.interact:
            x = (checkpoint(blk, x, state, p["rel"], self.kv_chunk, use_reentrant=False) if use_ckpt
                 else blk(x, state, p["rel"], self.kv_chunk))
        x = self.final_norm(x)
        out = []
        for i, qn in enumerate(p["q_node"]):
            opts = torch.tensor(p["opt_nodes"][i], device=dev)
            o = x.index_select(0, opts)
            q = x[qn].expand_as(o)
            z = self.scorer(torch.cat([o, q, o * q], -1)).squeeze(-1)
            t = torch.exp(self.log_temp[QTYPE_ID[p["qtypes"][i]]])
            out.append(z / t)
        return out


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())
