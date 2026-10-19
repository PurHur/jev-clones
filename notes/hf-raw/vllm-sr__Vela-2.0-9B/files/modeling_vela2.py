"""Vela 2.0 decoder member: model definition (Decision 2.0 base + Vela 2.0 routing recipe + span head v2).

Needs torch, transformers (Qwen3.5 text model, >= 5.x as used in training: 5.17.0) and safetensors, plus the sibling files
configuration_vela2.py and vela2_inference.py (numpy, tokenizers). Optional: flash-linear-attention (`fla`) for the
Gated-DeltaNet kernel on GPU; without it the transformers torch reference kernel is used (slower; not parity-tested on GPU).
Parameter names are exactly those of the export (backbone.* = the Qwen3_5TextModel of the research TreeQwen, head.* = the
DEV2.0 CandidateHead, set_bias, span2.* = SpanHeadV2), so the safetensors shards load with a strict key check.

  backbone  Qwen3.5 text model (hybrid: Gated-DeltaNet linear-attention layers + one full-attention layer in four), used as
            an encoder with no generation, run as a TREE forward (exact prefix fork):
              * the parts (typed segments) run once per row, LEFT-padded (zero inputs keep the DeltaNet state exactly 0);
              * every question block continues from its row's state: DeltaNet conv over [conv tail ; block] and the chunk
                gated-delta rule with initial_state = the parts state; full attention: block queries attend [parts KV of
                the owner row ; own block KV, causal].
            => block i sees the parts and itself only, and equals the single sequence [parts + block i]: bundled == single.
  choice / score / set  DEV2.0 CandidateHead (FP32): each option is read at its last token (endpoint), the block's last token
            is the global query; bilinear + MLP score. set = sigmoid(score + set_bias) per option.
  span      SpanHeadV2 (FP32): label vector = mean hidden state over the label block + a label-slot embedding; word vector =
            the first sub-word state of each word (layout c: the repeated copy of the target after the labels); per-row
            mean-centring; score[w, l] = bilinear + MLP (the CandidateHead form, word x label); sigmoid per word x label.

Two entry points, same parameters and forward:
  Vela2DecoderModel.from_pretrained(dir_or_repo)        what vela2_inference.Vela2(path) loads (torch backend)
  AutoModel.from_pretrained(repo, trust_remote_code=True)  the same class; it adds the SystemOne interface (system_one,
            system_one_request) and the typed-parts / GLiNER2-style methods (predict, predict_generic, classify_text,
            extract_entities, detect_hallucination, score_relevance, score), delegated to vela2_inference.Vela2.
dtypes: parameters are held in FP32 (the stored backbone matmul / conv weights are bf16, exactly the values bf16 autocast
computes with; everything else is stored FP32). Default inference = bf16 autocast on GPU for the backbone, heads in FP32 (the
research evaluation setting); FP32 (CPU or GPU, amp_bf16=False) also works. fp16 is not supported.
"""
import contextlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel

try:  # transformers remote code: this file is loaded as transformers_modules.<repo>.modeling_vela2
    from .configuration_vela2 import Vela2DecoderConfig
    from .vela2_inference import Vela2 as _Vela2Engine
except ImportError:  # standalone: the release dir is on sys.path
    if __package__:
        raise
    from configuration_vela2 import Vela2DecoderConfig
    from vela2_inference import Vela2 as _Vela2Engine


# ===================================================================================================================== heads
class CandidateHead(nn.Module):
    """DEV2.0 decision head (decision2/_vendor/dev2model/decision_model.CandidateHead, verbatim)."""

    def __init__(self, hidden_size, head_dim=256):
        super().__init__()
        self.head_dim = head_dim
        self.candidate_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key = nn.Linear(hidden_size, head_dim, bias=False)
        self.query = nn.Linear(hidden_size, head_dim, bias=False)
        self.candidate_mlp = nn.Linear(hidden_size, head_dim)
        self.query_mlp = nn.Linear(hidden_size, head_dim, bias=False)
        self.scalar = nn.Linear(head_dim, 1, bias=False)

    def forward(self, candidates, query):
        with torch.autocast(device_type=candidates.device.type, enabled=False):
            candidate = self.candidate_norm(candidates.float())
            global_query = self.query_norm(query.float())
            bilinear = (self.key(candidate) * self.query(global_query)[:, None, :]).sum(-1) / math.sqrt(self.head_dim)
            nonlinear = self.scalar(F.gelu(self.candidate_mlp(candidate) + self.query_mlp(global_query)[:, None, :])).squeeze(-1)
            return bilinear + nonlinear


class SpanHeadV2(nn.Module):
    """schema_lib_d3_20261001.SpanHeadV2, verbatim (inference part)."""

    def __init__(self, hidden, d=256, slots=64):
        super().__init__()
        self.d = d
        self.wn, self.ln = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.slot = nn.Embedding(slots, hidden)
        self.K, self.Q = nn.Linear(hidden, d, bias=False), nn.Linear(hidden, d, bias=False)
        self.M, self.N = nn.Linear(hidden, d), nn.Linear(hidden, d, bias=False)
        self.v = nn.Linear(d, 1, bias=False)
        self.o = nn.Linear(hidden, 1)

    def forward(self, words, labels):
        """words (W, H), labels (L, H) raw states (label = block mean). Returns score (W, L)."""
        with torch.autocast(device_type=words.device.type, enabled=False):
            w = self.wn(words.float()); w = w - w.mean(0, keepdim=True)
            L = labels.shape[0]
            e = self.ln(labels.float()) + self.slot(torch.arange(L, device=labels.device).clamp(max=self.slot.num_embeddings - 1))
            if L >= 2:
                e = e - e.mean(0, keepdim=True)
            k, q = self.K(w), self.Q(e)
            bil = (k @ q.T) / math.sqrt(self.d)
            mlp = self.v(F.gelu(self.M(w)[:, None, :] + self.N(e)[None, :, :])).squeeze(-1)
            return bil + mlp


BROAD_FILE = "broad_head.safetensors"


def attach_broad_head(model, file):
    """2H: load broad_head.safetensors (keys span_broad.*) into a new SpanHeadV2 on `model.span_broad` (FP32, model's device).
    The release weights (span2.* and everything else) are not touched; vela2_inference.Vela2 dispatches span questions."""
    from safetensors.torch import load_file
    sd = load_file(str(file))
    sd = {k[len("span_broad."):]: v for k, v in sd.items() if k.startswith("span_broad.")}
    head = SpanHeadV2(sd["wn.weight"].shape[0], d=sd["K.weight"].shape[0], slots=sd["slot.weight"].shape[0])
    head.load_state_dict(sd, strict=True)
    model.span_broad = head.float().to(next(model.parameters()).device).eval()
    return model


# ===================================================================================================================== tree forward
def _autocast_on(device_type):
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:  # older torch: no device argument (CUDA autocast state)
        return torch.is_autocast_enabled()


class _TreeForward:
    """schema_lib_dec_20260930.TreeQwen (v1 path: no dense masks, pads sliced off), causal parts (span variants a / c)."""

    def _gdr(self, q, k, v, g, beta, **kw):
        if q.is_cuda:
            if not hasattr(self, "_fla"):
                try:
                    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
                    self._fla = chunk_gated_delta_rule
                except ImportError:
                    self._fla = None
            if self._fla is not None:
                return self._fla(q, k, v, g, beta, **kw)
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
        # CPU: the transformers torch reference itself (its hub-kernel wrapper would dispatch to fla's Triton kernels)
        f = getattr(torch_chunk_gated_delta_rule, "__wrapped__", torch_chunk_gated_delta_rule)
        return f(q, k, v, g=g, beta=beta, **kw)

    def _lin(self, la, hP, mP, hB, mB, owner):
        hP = hP * mP[..., None].to(hP.dtype); hB = hB * mB[..., None].to(hB.dtype)
        B, Lp, _ = hP.shape; N, Lb, _ = hB.shape
        K = la.conv_kernel_size; C = la.conv_dim
        w = la.conv1d.weight
        xP = la.in_proj_qkv(hP); xB = la.in_proj_qkv(hB)
        cP = F.conv1d(F.pad(xP.transpose(1, 2), (K - 1, 0)), w, la.conv1d.bias, groups=C)
        tail = xP[:, Lp - (K - 1):] if Lp >= K - 1 else F.pad(xP, (0, 0, K - 1 - Lp, 0))
        cB = F.conv1d(torch.cat([tail[owner], xB], 1).transpose(1, 2), w, la.conv1d.bias, groups=C)
        cP = F.silu(cP).transpose(1, 2); cB = F.silu(cB).transpose(1, 2)

        def qkv(x, L, n):
            q, k, v = torch.split(x, [la.key_dim, la.key_dim, la.value_dim], dim=-1)
            q = q.reshape(n, L, -1, la.head_k_dim); k = k.reshape(n, L, -1, la.head_k_dim); v = v.reshape(n, L, -1, la.head_v_dim)
            r = la.num_v_heads // la.num_k_heads
            if r > 1:
                q = q.repeat_interleave(r, dim=2); k = k.repeat_interleave(r, dim=2)
            return q, k, v

        def gates(h):
            beta = la.in_proj_b(h).sigmoid()
            g = -la.A_log.float().exp() * F.softplus(la.in_proj_a(h).float() + la.dt_bias)
            return g, beta

        dt = torch.bfloat16 if _autocast_on(hP.device.type) else hP.dtype
        qP, kP, vP = qkv(cP, Lp, B); qB, kB, vB = qkv(cB, Lb, N)
        gP, bP = gates(hP); gB, bB = gates(hB)
        c = lambda t: t.to(dt).contiguous()  # noqa: E731
        oP, sP = self._gdr(c(qP), c(kP), c(vP), gP.contiguous(), c(bP), initial_state=None, output_final_state=True,
                           use_qk_l2norm_in_kernel=True)
        oB, _ = self._gdr(c(qB), c(kB), c(vB), gB.contiguous(), c(bB), initial_state=sP[owner].contiguous(),
                          output_final_state=False, use_qk_l2norm_in_kernel=True)

        def out(o, h, n, L):
            z = la.in_proj_z(h).reshape(-1, la.head_v_dim)
            o = la.norm(o.reshape(-1, la.head_v_dim), z).reshape(n, L, -1)
            return la.out_proj(o)
        return out(oP, hP, B, Lp), out(oB, hB, N, Lb)

    def _attn(self, sa, hP, peP, hB, peB, owner, pad, lb):
        from torch.nn.attention.bias import causal_lower_right
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        def proj(h, pe):
            shp = h.shape[:-1]
            q, gate = torch.chunk(sa.q_proj(h).view(*shp, -1, sa.head_dim * 2), 2, dim=-1)
            q = sa.q_norm(q).transpose(1, 2)
            k = sa.k_norm(sa.k_proj(h).view(*shp, -1, sa.head_dim)).transpose(1, 2)
            v = sa.v_proj(h).view(*shp, -1, sa.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *pe)
            return q, k, v, gate.reshape(*shp, -1)
        qP, kP, vP, gP = proj(hP, peP); qB, kB, vB, gB = proj(hB, peB)
        rep = sa.num_key_value_groups
        kr = lambda t: t.repeat_interleave(rep, dim=1) if rep > 1 else t  # noqa: E731
        dt = vP.dtype  # rotary leaves q / k in FP32 under autocast; SDPA with an attn-bias object does not autocast
        qP, kP, qB, kB, vB = qP.to(dt), kP.to(dt), qB.to(dt), kB.to(dt), vB.to(dt)
        outs = []
        for b in range(qP.shape[0]):
            s = int(pad[b])
            o = F.scaled_dot_product_attention(qP[b:b + 1, :, s:], kr(kP[b:b + 1, :, s:]), kr(vP[b:b + 1, :, s:]),
                                               is_causal=True, scale=sa.scaling)
            outs.append(F.pad(o, (0, 0, s, 0)))
        oP = torch.cat(outs, 0)
        Lb = qB.shape[2]; outs = []
        for j in range(qB.shape[0]):
            o_, n = int(owner[j]), int(lb[j]); s = int(pad[o_])
            kk = torch.cat([kP[o_:o_ + 1, :, s:], kB[j:j + 1, :, :n]], 2); vv = torch.cat([vP[o_:o_ + 1, :, s:], vB[j:j + 1, :, :n]], 2)
            o = F.scaled_dot_product_attention(qB[j:j + 1, :, :n], kr(kk), kr(vv), scale=sa.scaling,
                                               attn_mask=causal_lower_right(n, kk.shape[2]))
            outs.append(F.pad(o, (0, 0, 0, Lb - n)))
        oB = torch.cat(outs, 0)

        def out(o, gate):
            o = o.transpose(1, 2).reshape(*gate.shape)
            return sa.o_proj(o * torch.sigmoid(gate))
        return out(oP, gP), out(oB, gB)

    def tree(self, p_ids, p_mask, b_ids, b_mask, owner):
        """p_ids / p_mask (B, Lp) LEFT-padded parts; b_ids / b_mask (N, Lb) right-padded blocks; owner (N,) row of each
        block. Returns final-norm hidden states (B, Lp, H), (N, Lb, H)."""
        bb = self.backbone; dev = p_ids.device
        B, Lp = p_ids.shape; N, Lb = b_ids.shape
        lp = p_mask.sum(1)
        pad = Lp - lp
        posP = (torch.arange(Lp, device=dev)[None] - pad[:, None]).clamp(min=0)
        posB = lp[owner][:, None] + torch.arange(Lb, device=dev)[None]
        xP = bb.embed_tokens(p_ids); xB = bb.embed_tokens(b_ids)
        peP = bb.rotary_emb(xP, posP[None].expand(3, B, Lp)); peB = bb.rotary_emb(xB, posB[None].expand(3, N, Lb))
        padl, lbl = pad.tolist(), b_mask.sum(1).tolist()
        for layer in bb.layers[: bb.config.num_hidden_layers]:
            hP = layer.input_layernorm(xP); hB = layer.input_layernorm(xB)
            if layer.block_type == "linear_attention":
                aP, aB = self._lin(layer.linear_attn, hP, p_mask, hB, b_mask, owner)
            else:
                aP, aB = self._attn(layer.self_attn, hP, peP, hB, peB, owner, padl, lbl)
            xP = xP + aP; xB = xB + aB
            xP = xP + layer.mlp(layer.post_attention_layernorm(xP)); xB = xB + layer.mlp(layer.post_attention_layernorm(xB))
        return bb.norm(xP), bb.norm(xB)

    # ------------------------------------------------------------------ one batch of rendered rows
    @torch.no_grad()
    def run_batch(self, recs, amp_bf16=False):
        """recs: rendered rows of vela2_inference (p_ids, blocks). Returns per rec dict(qs=[logits (n_ends,) float32 numpy,
        incl. the [ABS] endpoint last when shown; set: + set_bias], span=(W, L) float32 numpy | None) -- the research
        DecScorer / D3Scorer forward (train_vela2_decoder_d3_20261001.forward + collate_tree + question_tensors + heads)."""
        dev = next(self.parameters()).device
        pad_id = int(self.config.pad_token_id)
        B = len(recs)
        Lp = max(len(r["p_ids"]) for r in recs)
        blocks = [(b, blk) for b, r in enumerate(recs) for blk in r["blocks"]]
        Lb = max(len(blk["ids"]) for _, blk in blocks)
        p_ids = torch.full((B, Lp), pad_id, dtype=torch.long); p_mask = torch.zeros((B, Lp), dtype=torch.long)
        shift = []
        for b, r in enumerate(recs):
            n = len(r["p_ids"]); s = Lp - n; shift.append(s)
            p_ids[b, s:] = torch.tensor(r["p_ids"]); p_mask[b, s:] = 1
        N = len(blocks)
        b_ids = torch.full((N, Lb), pad_id, dtype=torch.long); b_mask = torch.zeros((N, Lb), dtype=torch.long)
        owner = torch.tensor([b for b, _ in blocks], dtype=torch.long)
        for j, (_, blk) in enumerate(blocks):
            b_ids[j, :len(blk["ids"])] = torch.tensor(blk["ids"]); b_mask[j, :len(blk["ids"])] = 1
        T = lambda x: x.to(dev)  # noqa: E731
        amp = torch.autocast(dev.type, dtype=torch.bfloat16) if amp_bf16 else contextlib.nullcontext()
        with amp:
            HP, HB = self.tree(T(p_ids), T(p_mask), T(b_ids), T(b_mask), T(owner))
        res = [dict(qs=[], span=None) for _ in recs]
        qs = [(j, b, blk) for j, (b, blk) in enumerate(blocks) if blk["kind"] == "q"]
        if qs:
            W = max(len(blk["ends"]) for _, _, blk in qs)
            pos = torch.zeros((len(qs), W), dtype=torch.long); cm = torch.zeros((len(qs), W), dtype=torch.bool)
            for i, (_, _, blk) in enumerate(qs):
                pos[i, :len(blk["ends"])] = torch.tensor(blk["ends"]); cm[i, :len(blk["ends"])] = True
            blk_ix = T(torch.tensor([j for j, _, _ in qs])); qpos = T(torch.tensor([blk["query"] for _, _, blk in qs]))
            pos, cm = T(pos), T(cm)
            S = self.head(HB[blk_ix[:, None], pos], HB[blk_ix, qpos]).masked_fill(~cm, -float("inf")).float()
            for i, (_, b, blk) in enumerate(qs):
                n = len(blk["ends"])
                lg = S[i, :n] + (self.set_bias.float() if blk["qtype"] == "set" else 0.0)
                res[b]["qs"].append(lg.cpu().numpy())
        for j, (b, blk) in enumerate(blocks):
            if blk["kind"] != "span" or blk.get("skip"):
                continue
            wp = torch.as_tensor(blk["w_pos"], device=dev)
            w = HB[j, wp] if blk["words_in_block"] else HP[b, wp + shift[b]]
            lab = torch.stack([HB[j, s:e + 1].mean(0) for s, e in blk["label_ranges"]])
            res[b]["span"] = self.span2(w, lab).float().cpu().numpy()
        return res


def _build(m, config):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    bc = dict(config.backbone_config)
    bc.pop("architectures", None); bc.pop("transformers_version", None); bc.pop("model_type", None)
    bcfg = Qwen3_5TextConfig(**bc)
    bcfg._attn_implementation = "sdpa"  # unused by the tree forward (it calls SDPA itself); set for the stock modules
    bcfg.use_cache = False
    m.backbone = Qwen3_5TextModel(bcfg)
    h = bcfg.hidden_size
    m.head = CandidateHead(h, config.head_dim)
    m.set_bias = nn.Parameter(torch.zeros(()))
    sh = config.span_head
    m.span2 = SpanHeadV2(h, d=sh.get("d", 256), slots=sh.get("slots", 64))


@contextlib.contextmanager
def _no_init():
    """Skip random init while building the module (every parameter is overwritten by the checkpoint)."""
    ctx = None
    for mod, name in (("transformers.initialization", "no_init_weights"), ("transformers.modeling_utils", "no_init_weights")):
        try:
            ctx = getattr(__import__(mod, fromlist=[name]), name)
            break
        except (ImportError, AttributeError):
            continue
    saved = {k: getattr(nn.init, k) for k in ("normal_", "uniform_", "kaiming_uniform_", "zeros_", "ones_", "trunc_normal_")}
    try:
        for k in saved:
            setattr(nn.init, k, lambda t, *a, **kw: t)
        with (ctx() if ctx is not None else contextlib.nullcontext()):
            yield
    finally:
        for k, f in saved.items():
            setattr(nn.init, k, f)


_HUB_KWARGS = ("cache_dir", "force_download", "proxies", "local_files_only", "token", "revision", "subfolder")
_ALLOW = ["config.json", "calibration.json", "tokenizer.json", "tokenizer_config.json", "*.safetensors", "model.safetensors.index.json",
          "modeling_vela2.py", "configuration_vela2.py", "vela2_inference.py"]


def resolve_dir(name_or_path, **hub):
    p = Path(str(name_or_path))
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    kw = {k: v for k, v in hub.items() if k in ("cache_dir", "force_download", "proxies", "local_files_only", "token", "revision")}
    return Path(snapshot_download(str(name_or_path), allow_patterns=_ALLOW, **kw))


def load_weights(model, d):
    """Strict load of the safetensors shards of release dir d into model (cast to the parameter dtype, FP32)."""
    from safetensors.torch import load_file
    d = Path(d)
    idx = d / "model.safetensors.index.json"
    files = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]
    params = dict(model.named_parameters())
    seen = set()
    for f in files:
        sd = load_file(str(d / f))
        for k, v in sd.items():
            if k not in params:
                raise KeyError(f"unexpected tensor {k} in {f}")
            if tuple(params[k].shape) != tuple(v.shape):
                raise ValueError(f"shape mismatch {k}: {tuple(v.shape)} vs {tuple(params[k].shape)}")
            with torch.no_grad():
                params[k].copy_(v.to(params[k].dtype))
            seen.add(k)
        del sd
    missing = sorted(set(params) - seen)
    if missing:
        raise KeyError(f"missing tensors: {missing[:5]} (+{max(0, len(missing) - 5)})")
    return model


class Vela2DecoderModel(_TreeForward, PreTrainedModel):
    config_class = Vela2DecoderConfig
    base_model_prefix = "vela2"  # no submodule has this name: checkpoint keys load as they are
    main_input_name = "input_ids"
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        _build(self, config)
        self._vela2_dir = None
        self._vela2_engine = None

    def _init_weights(self, module):  # every parameter comes from the checkpoint
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, config=None, **kwargs):
        """Load a release dir or Hub repo. Parameters are FP32 (dtype / torch_dtype other than float32 are refused: the stored
        backbone weights are the bf16 values autocast uses, and the heads and norms must stay FP32)."""
        dt = kwargs.pop("dtype", None) or kwargs.pop("torch_dtype", None)
        if dt not in (None, "auto", torch.float32, "float32"):
            raise ValueError("Vela 2.0 decoder parameters load in float32 (bf16 compute = amp_bf16=True in the engine); "
                             f"dtype={dt} is not supported")
        device_map = kwargs.pop("device_map", None)
        hub = {k: kwargs[k] for k in _HUB_KWARGS if kwargs.get(k) is not None}
        d = resolve_dir(pretrained_model_name_or_path, **hub)
        if config is None or not isinstance(config, Vela2DecoderConfig):
            config = Vela2DecoderConfig.from_pretrained(str(d))
        with _no_init():
            model = cls(config)
        model = load_weights(model.float(), d).eval()
        model._vela2_dir = str(d)
        if isinstance(device_map, str) and device_map not in ("auto",):
            model = model.to(device_map)
        elif device_map == "auto" and torch.cuda.is_available():
            model = model.to("cuda")
        return model

    @property
    def vela2_engine(self):
        """The vela2_inference.Vela2 engine (tokenisation, layout, windows, calibration, span decoding) over this model, built
        on first use on the device the model is on at that point (bf16 autocast on GPU, FP32 on CPU)."""
        if self._vela2_engine is None:
            if self._vela2_dir is None:
                self._vela2_dir = str(resolve_dir(self.config._name_or_path))
            self._vela2_engine = _Vela2Engine(self._vela2_dir, model=self, device=str(next(self.parameters()).device))
        return self._vela2_engine

    def system_one(self, *args, **kwargs):
        return self.vela2_engine.system_one(*args, **kwargs)

    def system_one_request(self, *args, **kwargs):
        return self.vela2_engine.system_one_request(*args, **kwargs)

    def classify_text(self, *args, **kwargs):
        return self.vela2_engine.classify_text(*args, **kwargs)

    def extract_entities(self, *args, **kwargs):
        return self.vela2_engine.extract_entities(*args, **kwargs)

    def detect_hallucination(self, *args, **kwargs):
        return self.vela2_engine.detect_hallucination(*args, **kwargs)

    def score_relevance(self, *args, **kwargs):
        return self.vela2_engine.score_relevance(*args, **kwargs)

    def score(self, *args, **kwargs):
        return self.vela2_engine.score(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.vela2_engine.predict(*args, **kwargs)

    def predict_generic(self, *args, **kwargs):
        return self.vela2_engine.predict_generic(*args, **kwargs)


for _n in ("system_one", "system_one_request", "classify_text", "extract_entities", "detect_hallucination", "score_relevance", "score",
           "predict", "predict_generic"):
    getattr(Vela2DecoderModel, _n).__doc__ = getattr(_Vela2Engine, _n).__doc__
