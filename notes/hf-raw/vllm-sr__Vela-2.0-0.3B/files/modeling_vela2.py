"""Vela 2.0 unified schema encoder: model definition.

Needs torch, transformers' ModernBERT and safetensors, plus the sibling files configuration_vela2.py and
vela2_inference.py (numpy, tokenizers). Parameter names are exactly those of the training checkpoint (SchemaEncoderV2 in the research code), so `model.safetensors` loads with strict=True.

  encoder      ModernBERT (Vela-1.0-Encoder-307M geometry, 22 layers, h=768, YaRN to 32k), vocab 256,008
               (256,000 base + 8 marker rows: [Q] [O] [ABS] [SEP_SCHEMA] [SEG_user] [SEG_answer] [E] [SEG_context])
  choice / set / score readout ("both" = cosine + MLP, logits summed):
      cosine  u = normalize(W_o LN(h[O]) + W_q LN(h[Q])),  v = normalize(W_p LN(mean(h[part range]))),  u.v / tau
      mlp     Linear(h, 2h) - ReLU - Dropout(0.1) - Linear(2h, 1)  on h[O]
  span readout (word units): normalize(W_t LN(h[first sub-word of word])) . normalize(W_e LN(h[E])) / tau_span

Index tensors (all int64) used by `heads` / the ONNX graph:
  q_index   (N, 4)  [batch row, position of [Q], pool start, pool end)   one per choice / set / score question
  opt_index (M, 3)  [batch row, position of [O] or [ABS], owning question 0..N-1]
  unit_index(K, 2)  [batch row, position of a word's first sub-word token]
  ent_index (E, 2)  [batch row, position of an [E] label marker]
Outputs: opt_logits (M,), span_logits (K, E) (all word x label pairs; callers keep the same-row blocks).

Two classes, same parameter names, same forward:
  Vela2SchemaEncoder  plain nn.Module, used by vela2_inference.Vela2(path, backend="torch") when this dir is on sys.path
  Vela2Model          transformers PreTrainedModel for AutoModel.from_pretrained(repo, trust_remote_code=True); it adds
                      the SystemOne interface (system_one, system_one_request) and the typed-parts / GLiNER2-style
                      methods (predict, predict_generic, classify_text, extract_entities, detect_hallucination,
                      score_relevance, score) by delegating to vela2_inference.Vela2 on the model's device
"""
import json
from pathlib import Path

import torch
from torch import nn
from transformers import PreTrainedModel

try:  # transformers remote code: this file is loaded as transformers_modules.<repo>.modeling_vela2
    from .configuration_vela2 import Vela2Config
    from .vela2_inference import Vela2 as _Vela2Engine
except ImportError:  # standalone: the release dir is on sys.path
    if __package__:
        raise
    from configuration_vela2 import Vela2Config
    from vela2_inference import Vela2 as _Vela2Engine


def _create_mlp(h, dropout=0.1):
    return nn.Sequential(nn.Linear(h, 2 * h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * h, 1))


def _build(m, encoder_config, proj_dim, readout, mlp_dropout, attn_implementation):
    """Create the encoder and the readout heads on module `m` (the parameter names of model.safetensors)."""
    from transformers import ModernBertConfig, ModernBertModel
    ecfg = ModernBertConfig(**encoder_config)
    ecfg._attn_implementation = attn_implementation
    m.encoder = ModernBertModel(ecfg)
    h, d = ecfg.hidden_size, proj_dim
    m.readout = readout
    m.norm = nn.LayerNorm(h)
    m.w_o = nn.Linear(h, d)
    m.w_q = nn.Linear(h, d)
    m.w_p = nn.Linear(h, d)
    m.log_tau = nn.Parameter(torch.zeros(()))
    m.w_t = nn.Linear(h, d)
    m.w_e = nn.Linear(h, d)
    m.log_tau_span = nn.Parameter(torch.zeros(()))
    if readout in ("mlp", "both"):
        m.cls_mlp = _create_mlp(h, mlp_dropout)


class _Vela2Forward:

    # ------------------------------------------------------------------ forward
    def hidden(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def heads(self, hid, q_index, opt_index, unit_index, ent_index):
        H = hid.float()
        ob, op, own = opt_index[:, 0], opt_index[:, 1], opt_index[:, 2]
        Ho = H[ob, op]
        lg = torch.zeros(Ho.shape[0], dtype=H.dtype, device=H.device)
        if self.readout in ("cosine", "both"):
            cs = nn.functional.pad(H.cumsum(1), (0, 0, 1, 0))
            b, qp, ps, pe = q_index[:, 0], q_index[:, 1], q_index[:, 2], q_index[:, 3]
            pooled = (cs[b, pe] - cs[b, ps]) / (pe - ps).clamp(min=1).unsqueeze(-1).float()
            v = nn.functional.normalize(self.w_p(self.norm(pooled)), dim=-1)
            uq = self.w_q(self.norm(H[b, qp]))
            u = nn.functional.normalize(self.w_o(self.norm(Ho)) + uq[own], dim=-1)
            lg = lg + (u * v[own]).sum(-1) / self.log_tau.exp()
        if self.readout in ("mlp", "both"):
            lg = lg + self.cls_mlp(Ho).squeeze(-1)
        t = nn.functional.normalize(self.w_t(self.norm(H[unit_index[:, 0], unit_index[:, 1]])), dim=-1)
        e = nn.functional.normalize(self.w_e(self.norm(H[ent_index[:, 0], ent_index[:, 1]])), dim=-1)
        span = (t @ e.T) / self.log_tau_span.exp()
        return lg, span

    def forward(self, input_ids, attention_mask, q_index, opt_index, unit_index, ent_index):
        return self.heads(self.hidden(input_ids, attention_mask), q_index, opt_index, unit_index, ent_index)


class Vela2SchemaEncoder(_Vela2Forward, nn.Module):
    def __init__(self, config, attn_implementation="sdpa"):
        super().__init__()
        self.config = config
        _build(self, config["encoder_config"], config["proj_dim"], config["readout"], config.get("mlp_dropout", 0.1),
               attn_implementation)

    @classmethod
    def from_pretrained(cls, path, device="cpu", attn_implementation="sdpa"):
        from safetensors.torch import load_file
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        m = cls(config, attn_implementation=attn_implementation)
        sd = load_file(str(path / "model.safetensors"))
        m.load_state_dict(sd, strict=True)
        return m.to(device).eval()


_HUB_KWARGS = ("cache_dir", "force_download", "proxies", "local_files_only", "token", "revision", "subfolder")
_ENGINE_FILES = ("config.json", "calibration.json", "tokenizer.json")


class Vela2Model(_Vela2Forward, PreTrainedModel):
    config_class = Vela2Config
    base_model_prefix = "vela2"  # no submodule has this name: checkpoint keys load as they are, no prefix added or cut
    main_input_name = "input_ids"
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        attn = getattr(config, "_attn_implementation", None)
        _build(self, config.encoder_config, config.proj_dim, config.readout, config.mlp_dropout,
               attn if attn in ("eager", "sdpa", "flash_attention_2") else "sdpa")
        self._vela2_dir = None
        self._vela2_engine = None
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        out = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model = out[0] if isinstance(out, tuple) else out
        # fetch the engine files (tokenizer, calibration) now, from the same repo / revision / cache as the weights
        hub = {k: kwargs[k] for k in _HUB_KWARGS if kwargs.get(k) is not None}
        if getattr(model.config, "_commit_hash", None):
            hub["revision"] = model.config._commit_hash
        model._vela2_dir = cls._resolve_dir(pretrained_model_name_or_path, **hub)
        return out

    @staticmethod
    def _resolve_dir(name_or_path, **hub):
        from transformers.utils import cached_file
        files = [cached_file(str(name_or_path), f, **hub) for f in _ENGINE_FILES]
        return str(Path(files[0]).parent)

    @property
    def vela2_engine(self):
        """The vela2_inference.Vela2 engine (tokenisation, schema assembly, windowing, calibration, span decoding) over
        this model, torch backend. Built on first use, on the device the model is on at that point."""
        if self._vela2_engine is None:
            if self._vela2_dir is None:
                self._vela2_dir = self._resolve_dir(self.config._name_or_path)
            self._vela2_engine = _Vela2Engine(self._vela2_dir, backend="torch", model=self,
                                              device=str(next(self.parameters()).device))
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


for _n in ("system_one", "system_one_request", "classify_text", "extract_entities", "detect_hallucination", "score_relevance", "score", "predict",
           "predict_generic"):
    getattr(Vela2Model, _n).__doc__ = getattr(_Vela2Engine, _n).__doc__
