"""decider-2b-coherent: stock Mapika/decider-2b (frozen, unchanged) + a sims-calibrated marginal adapter behind a domain
gate + a learned coupling, answered from ONE joint (a mixture of products) that is I-projected onto the served
marginals. Every answer to one input comes from that joint, so the answers are coherent by construction.

    forward (frozen decider-2b, decider's own state_first prompt, token-identical)
      -> letter logits l_i per question; hidden states h
      -> domain gate p_sims = sigmoid(w . standardize([mean slot hidden, mean context hidden]) + b)
            p_sims <  0.5 ("real"): q_i = softmax(l_i / T), T = 1.145 (exactly what stock decider serves)
            p_sims >= 0.5 ("sims"): q_i = softmax(a * l_i + <U h_slot, V h_option> / sqrt(d_a))   (the adapter)
      -> coupling: mixture of M=8 products with component logits log q_i + learned offsets
      -> factored IPF (I-projection) onto q_1..q_n: the joint's marginals equal q_i (to 1e-12), the dependence is learned
    Single-question inputs skip the coupling: the joint is q itself.

The prompt helpers below are vendored from Mapika/decider (decider/prompt.py, Apache-2.0) so this package has no
dependency on the decider code base.
"""
from __future__ import annotations

import json
import math
import os
import string
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .joint import MixtureJoint

BASE = "Mapika/decider-2b"
BASE_REVISION = "533964dae8be954c5b5e19fa4948e48408094c1e"
REPO = "Mapika/decider-2b-coherent"

# one attention-backend setting for every forward (cuDNN SDPA excluded: NaN backward with masks on some stacks); the
# released numbers were computed with this list
SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, SDPBackend.FLASH_ATTENTION]

# ---------------------------------------------------------------------------------------------------------------------
# decider prompt (vendored from Mapika/decider decider/prompt.py)
# ---------------------------------------------------------------------------------------------------------------------
LETTERS = "ABCDEFGHIJ"
NARROW = len(LETTERS)          # <= NARROW options: "(A) .. (J)" rendering, tokenized as one string
MAX_OPTIONS = 255              # > NARROW options: "wide" rendering, one label token per option
_LABELS: dict = {}


def _enc_opt(tok, text):
    return tok.encode(f") {text}", add_special_tokens=False)


def label_table(tok):
    key = id(tok)
    if key not in _LABELS:
        U = string.ascii_uppercase
        names = list(U) + [a + b for a in U for b in U]
        out = []
        for n in names:
            t = tok.encode(n, add_special_tokens=False)
            if len(t) == 1:
                out.append((n, t[0]))
            if len(out) == MAX_OPTIONS:
                break
        assert len(out) == MAX_OPTIONS and len({i for _, i in out}) == MAX_OPTIONS
        _LABELS[key] = ([n for n, _ in out], [i for _, i in out], tok.encode("\n(", add_special_tokens=False))
    return _LABELS[key]


def letter_ids(tok):
    ids = label_table(tok)[1]
    for j, L in enumerate(LETTERS):
        assert tok.encode(L, add_special_tokens=False) == [ids[j]], L
    return ids


def build_prompt(tok, context, qs, perms, max_ctx=1536):
    """decider's state_first prompt with option spans. qs: list of (question text, options); perms[k]: the canonical
    option index shown at each display position of question k. Returns dict(ids, slots, nopts, perms, spans, ctx_len)."""
    ids = tok.encode("Context:\n" + context, add_special_tokens=False)[:max_ctx]
    ctx_len = len(ids)
    multi = len(qs) > 1
    slots, spans = [], []
    for k, ((text, options), order) in enumerate(zip(qs, perms)):
        head = f"\n\nQuestion{' ' + str(k + 1) if multi else ''}: {text}\nOptions:"
        tail = f"\nAnswer{' ' + str(k + 1) if multi else ''}: ("
        sp = []
        if len(order) <= NARROW:
            # decider encodes head + options + tail as ONE string (tokens may merge across boundaries): mirror it
            s, bounds = head, []
            for j, oi in enumerate(order):
                pre = f"\n({LETTERS[j]}) "
                bounds.append((len(s) + len(pre), len(s) + len(pre) + len(str(options[oi]))))
                s += pre + str(options[oi])
            s += tail
            ids_s = tok.encode(s, add_special_tokens=False)
            enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
            base = len(ids)
            if enc["input_ids"] == ids_s:
                for lo_c, hi_c in bounds:
                    toks = [t for t, (a_, b_) in enumerate(enc["offset_mapping"]) if b_ > lo_c and a_ < hi_c]
                    sp.append((base + toks[0], base + toks[-1] + 1) if toks else (base, base + 1))
            else:
                for lo_c, hi_c in bounds:
                    lo_t = len(tok.encode(s[:lo_c], add_special_tokens=False))
                    hi_t = max(lo_t + 1, len(tok.encode(s[:hi_c], add_special_tokens=False)))
                    sp.append((base + min(lo_t, len(ids_s) - 2), base + min(hi_t, len(ids_s) - 1)))
            ids += ids_s
        else:
            _, lab_ids, open_ids = label_table(tok)
            ids += tok.encode(head, add_special_tokens=False)
            for j, oi in enumerate(order):
                ids += open_ids + [lab_ids[j]]
                body = _enc_opt(tok, options[oi])
                sp.append((len(ids) + 1, len(ids) + len(body)) if len(body) > 1 else (len(ids), len(ids) + len(body)))
                ids += body
            ids += tok.encode(tail, add_special_tokens=False)
        slots.append(len(ids) - 1)
        spans.append(sp)
    return dict(ids=ids, slots=slots, nopts=[len(p) for p in perms], perms=[list(p) for p in perms], spans=spans,
                ctx_len=ctx_len)


# ---------------------------------------------------------------------------------------------------------------------
# BookieBench instance helpers (instance = {"prelude", "variables", "queries", "steps": [{"evidence"}...]})
# ---------------------------------------------------------------------------------------------------------------------

def var_question(inst, vi):
    name = inst["variables"][vi]["name"]
    for q in inst.get("queries", []):
        if q["kind"] == "marginal" and q["var"] == name and "step" not in q and q.get("text"):
            return q["text"]
    return f"What is the {name}?"


def state_text(inst, k=None):
    """prelude + evidence of steps 0..k (default: all steps)."""
    T = len(inst["steps"])
    k = T - 1 if k is None else k
    return inst.get("prelude", "") + "\n" + "\n".join(inst["steps"][i]["evidence"] for i in range(k + 1))


# ---------------------------------------------------------------------------------------------------------------------
# factored IPF
# ---------------------------------------------------------------------------------------------------------------------

def ipf_mixture(logw, logp, logq, n_iter=None, tol=1e-12, max_iter=5000, check_every=10):
    """I-project the mixture (logw [M], logp list of [M,K_i]) onto target log-marginals logq (list of [K_i]).
    Scaling a mixture of products by per-variable factors keeps it a mixture of products, so each step is O(M K_i) and
    the joint is never materialised. Sweeps until max |marginal - q| < tol (or exactly n_iter sweeps)."""
    logp = list(logp)
    sweeps, err = 0, float("inf")
    while True:
        for i in range(len(logp)):
            logm = torch.logsumexp(logw[:, None] + logp[i], 0)
            lp = logp[i] + (logq[i] - logm)[None]
            z = torch.logsumexp(lp, -1)
            logp[i] = lp - z[:, None]
            logw = logw + z
            logw = logw - torch.logsumexp(logw, 0)
        sweeps += 1
        if n_iter is not None:
            if sweeps >= n_iter:
                break
            continue
        if sweeps % check_every and sweeps < max_iter:
            continue
        with torch.no_grad():
            err = max(float((torch.logsumexp(logw[:, None] + lp_, 0).exp() - lq.exp()).abs().max()) for lp_, lq in zip(logp, logq))
        if err < tol or sweeps >= max_iter:
            break
    return logw, logp, sweeps, err


# ---------------------------------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Prediction:
    """marginals[k][j]: P(question k = option j) in the order the options were given. joint: [K_1, ..., K_n] array
    (None if joint=False). p_sims: domain-gate score (>= 0.5: the sims adapter was used). mj: the MixtureJoint."""
    marginals: list
    joint: np.ndarray | None
    p_sims: float
    adapter_used: bool
    mj: MixtureJoint = field(repr=False)
    names: list = field(repr=False, default_factory=list)

    def answer(self, query: dict) -> list[float]:
        """Coherent answer from the one joint. Question names: q1..qn by default (or the instance's variable names).
        See MixtureJoint.answer for the query format."""
        return self.mj.answer(query, {n: i for i, n in enumerate(self.names)})


class CoherentDecider(nn.Module):
    def __init__(self, lm, tok, cfg):
        super().__init__()
        self.lm, self.tok, self.cfg = lm, tok, cfg
        self.lm.requires_grad_(False)
        self.lm.eval()
        M, d, d_a, H = cfg["M"], cfg["d"], cfg["d_a"], lm.config.get_text_config().hidden_size
        self.M, self.d, self.d_a, self.T = M, d, d_a, cfg["temperature"]
        self.register_buffer("labels", torch.tensor(letter_ids(tok)), persistent=False)
        self.ln_s, self.ln_o, self.ln_j = nn.LayerNorm(H), nn.LayerNorm(H), nn.LayerNorm(H)
        self.Wo = nn.Linear(H, d)
        self.Wc = nn.Linear(H, M * d)
        self.Wcj = nn.Linear(H, M * d)
        self.Ww = nn.Linear(H, M)
        self.gain = nn.Parameter(torch.tensor(0.1))
        self.ad_loga = nn.Parameter(torch.tensor(0.0))
        self.ad_ln_s, self.ad_ln_o = nn.LayerNorm(H), nn.LayerNorm(H)
        self.ad_U = nn.Linear(H, d_a)
        self.ad_V = nn.Linear(H, d_a)
        for n_, shape in (("gate_mu", (2 * H,)), ("gate_sd", (2 * H,)), ("gate_w", (2 * H,)), ("gate_b", ())):
            self.register_buffer(n_, torch.zeros(shape))
        self.register_buffer("gate_on", torch.ones((), dtype=torch.bool))
        self.last_psims = []

    # ---- building inputs --------------------------------------------------------------------------------------------
    def build(self, context: str, questions, perms=None):
        """questions: list of (text, options). perms[k] (optional): canonical option index per display position."""
        perms = perms or [list(range(len(o))) for _, o in questions]
        it = build_prompt(self.tok, context, [(t, list(o)) for t, o in questions], perms, self.cfg.get("max_ctx", 1536))
        it["sizes"] = [len(o) for _, o in questions]
        return it

    def build_instance(self, inst, step=None, perms=None):
        V = inst["variables"]
        qs = [(var_question(inst, i), v["options"]) for i, v in enumerate(V)]
        return self.build(state_text(inst, step), qs, perms)

    # ---- forward ----------------------------------------------------------------------------------------------------
    def backbone(self, items):
        dev = self.labels.device
        T = max(len(b["ids"]) for b in items)
        ids = torch.full((len(items), T), self.tok.pad_token_id or 0, dtype=torch.long)
        am = torch.zeros_like(ids)
        rows, cols = [], []
        for i, b in enumerate(items):
            ids[i, :len(b["ids"])] = torch.tensor(b["ids"])
            am[i, :len(b["ids"])] = 1
            rows += [i] * len(b["slots"])
            cols += b["slots"]
        with torch.no_grad(), sdpa_kernel(SDPA), torch.autocast(dev.type, dtype=torch.bfloat16):
            h = self.lm.model(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state
        hs = h[torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)]
        W = self.lm.lm_head.weight if getattr(self.lm.lm_head, "weight", None) is not None else self.lm.model.embed_tokens.weight
        lg = F.linear(hs.float(), W[self.labels].float())
        out, r = [], 0
        for b in items:
            out.append([lg[r + k, :b["nopts"][k]] for k in range(len(b["slots"]))])
            r += len(b["slots"])
        return h, out

    @staticmethod
    def gate_features(h_item, it):
        hs = h_item[torch.tensor(it["slots"], device=h_item.device)].float().mean(0)
        hc = h_item[: max(1, it.get("ctx_len", 1))].float().mean(0)
        return torch.cat([hs, hc])

    def p_sims(self, feats):
        if not bool(self.gate_on):
            return torch.zeros(feats.shape[0], device=feats.device)
        return torch.sigmoid(((feats - self.gate_mu) / self.gate_sd) @ self.gate_w + self.gate_b)

    @torch.no_grad()
    def forward(self, items, temperature=None, need_joint=True, domains=None):
        """items: from build()/build_instance(). -> list[MixtureJoint] in canonical option order.
        temperature: served decider temperature on the real (non-adapter) path (default 1.145, stock decider's).
        need_joint=False: skip the coupling (marginals only; they are identical either way).
        domains: optional list of "sims"/"real" overriding the gate."""
        T = self.T if temperature is None else temperature
        dev = self.labels.device
        h, logits = self.backbone(items)
        feats = torch.stack([self.gate_features(h[b], it) for b, it in enumerate(items)])
        ps = self.p_sims(feats)
        self.last_psims = ps.tolist()
        use_ad = [(domains[b] == "sims") if domains is not None else bool(ps[b] >= 0.5) for b in range(len(items))]
        out = []
        for b, (it, lgs) in enumerate(zip(items, logits)):
            logq = []
            for k, lg in enumerate(lgs):
                inv = torch.empty(len(lg), dtype=torch.long, device=dev)
                inv[torch.tensor(it["perms"][k], device=dev)] = torch.arange(len(lg), device=dev)
                if use_ad[b]:
                    hb_ = h[b].float()
                    o = torch.stack([hb_[lo:hi].mean(0) for lo, hi in it["spans"][k]])
                    u = self.ad_U(self.ad_ln_s(hb_[it["slots"][k]]))
                    v = self.ad_V(self.ad_ln_o(o))
                    al = self.ad_loga.exp() * lg + (v @ u) / math.sqrt(self.d_a)
                    logq.append(torch.log_softmax(al.double(), -1)[inv])
                else:
                    logq.append(torch.log_softmax(lg.double() / T, -1)[inv])
            if len(logq) == 1 or not need_joint:
                out.append(MixtureJoint(torch.zeros(1, dtype=torch.float64, device=dev), [lq[None] for lq in logq]))
                continue
            hb = h[b].float()
            hs = hb[torch.tensor(it["slots"], device=dev)]
            g = self.ln_j(hs.mean(0))
            wl = self.Ww(g)
            qj = self.Wcj(g).view(self.M, self.d)
            comps = []
            for k in range(len(logq)):
                o = torch.stack([hb[lo:hi].mean(0) for lo, hi in it["spans"][k]])
                ov = self.Wo(self.ln_o(o))
                qk = self.Wc(self.ln_s(hs[k])).view(self.M, self.d) + qj
                off = (qk @ ov.T) / math.sqrt(self.d) * self.gain
                inv = torch.empty(ov.shape[0], dtype=torch.long, device=dev)
                inv[torch.tensor(it["perms"][k], device=dev)] = torch.arange(ov.shape[0], device=dev)
                comps.append(torch.log_softmax(logq[k][None] + off[:, inv].double(), -1))
            logw = torch.log_softmax(wl.double(), -1)
            logw, comps, _, _ = ipf_mixture(logw.cpu(), [c.cpu() for c in comps], [q.cpu() for q in logq])
            out.append(MixtureJoint(logw, comps))
        return out

    # ---- convenience ------------------------------------------------------------------------------------------------
    def _wrap(self, mjs, names_list, joint):
        res = []
        for mj, names, p in zip(mjs, names_list, self.last_psims):
            mj = mj.double()
            res.append(Prediction([mj.marginal(k).cpu().tolist() for k in range(len(mj.logp))],
                                  mj.joint().cpu().numpy() if joint else None, float(p), bool(p >= 0.5), mj, names))
        return res

    def predict(self, context: str, questions, joint=True, temperature=None, domain=None) -> Prediction:
        """One context, questions = [(text, [options...]), ...]. Questions are named q1..qn for answer()."""
        return self.predict_batch([(context, questions)], joint, temperature, domain)[0]

    def predict_batch(self, inputs, joint=True, temperature=None, domain=None, bs=32):
        out = []
        for i in range(0, len(inputs), bs):
            chunk = inputs[i:i + bs]
            items = [self.build(c, q) for c, q in chunk]
            mjs = self(items, temperature=temperature, need_joint=joint,
                       domains=None if domain is None else [domain] * len(items))
            out += self._wrap(mjs, [[f"q{k + 1}" for k in range(len(q))] for _, q in chunk], joint)
        return out

    def predict_instance(self, inst, step=None, joint=True, domain=None) -> Prediction:
        """A BookieBench-format instance at evidence step `step` (default: last). answer() takes the instance's
        variable names; e.g. {q["id"]: pred.answer(q) for q in inst["queries"]} (final-step queries)."""
        mjs = self([self.build_instance(inst, step)], need_joint=joint, domains=None if domain is None else [domain])
        return self._wrap(mjs, [[v["name"] for v in inst["variables"]]], joint)[0]


def load(base: str = BASE, revision: str | None = None, weights: str | None = None, repo: str = REPO,
         repo_revision: str | None = None, device: str | None = None, token=None) -> CoherentDecider:
    """Build the model: base decider-2b at the pinned revision (config.json "base_revision") + this repo's model.pt.
    weights: a local model.pt (default: downloaded from `repo`)."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(here, "config.json")
    if not os.path.exists(cfg_path):
        cfg_path = hf_hub_download(repo, "config.json", revision=repo_revision, token=token)
    cfg = json.load(open(cfg_path))
    revision = revision or cfg["base_revision"]
    if weights is None:
        local = os.path.join(here, "model.pt")
        weights = local if os.path.exists(local) else hf_hub_download(repo, "model.pt", revision=repo_revision, token=token)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base, revision=revision)
    lm = AutoModelForCausalLM.from_pretrained(base, revision=revision, dtype=torch.bfloat16)
    m = CoherentDecider(lm, tok, cfg)
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("lm.")]
    assert not bad and not unexpected, (bad, unexpected)
    return m.to(device).eval()
