"""LemonDecider full-weight decision model. No autoregressive generation head.

The checkpoint's backbone is copied byte-for-byte from the base LM, while readout
rows are selected from its original LM head. T is used in postprocessing, not CE loss.
"""
from dataclasses import dataclass
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5Attention
from .configuration_lemondecider import LemonDeciderConfig
from .option_loss import option_cross_entropy
from .contract import LemonDeciderReadout, enable_noncausal_full_attention, sdpa_backends, linearize_patch_embed


@dataclass
class LemonDeciderOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None


def final_norm_module(model):
    """Last language RMSNorm feeding the decision readout."""
    backbone = model.backbone if hasattr(model, "backbone") else model
    language = getattr(backbone, "language_model", None)
    if language is not None and hasattr(language, "norm"):
        return language.norm
    if hasattr(backbone, "norm"):
        return backbone.norm
    raise AttributeError("Backbone has no final RMSNorm")


class LemonDeciderForDecisionClassification(PreTrainedModel):
    config_class = LemonDeciderConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock", "LlamaDecoderLayer"]
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = False

    def __init__(self, config):
        super().__init__(config)
        hidden = config.text_config.hidden_size
        attn = getattr(config, "_attn_implementation", None) or getattr(config, "_attn_implementation_internal", None)
        attention_mode = getattr(config, "attention_mode", "causal")
        if attention_mode not in {"causal", "noncausal_full_attention"}:
            raise ValueError(f"Unknown attention mode: {attention_mode}")
        if getattr(config, "pooling", "last") != "last":
            raise ValueError(f"Unknown pooling mode: {getattr(config, 'pooling', 'last')}")
        if getattr(config, "backbone_type", "qwen3_5") == "llama":
            if attention_mode != "causal":
                raise ValueError("noncausal_full_attention is only implemented for the Qwen3.5 hybrid backbone")
            inner = config.llama_config
            if attn:
                inner._attn_implementation = attn
            self.backbone = LlamaModel(inner)
        else:
            if attention_mode == "noncausal_full_attention":
                config.text_config.is_causal = False
            self.backbone = Qwen3_5Model(config)
            self.backbone.language_model.config.is_causal = attention_mode == 'causal'
            # SDPA may omit the all-valid mask. Qwen's module flag otherwise
            # re-enables causality even when its config requested bidirectional.
            # Restrict to full attention; never modify the linear recurrence.
            for module in self.backbone.language_model.modules():
                if isinstance(module, Qwen3_5Attention):
                    module.is_causal = attention_mode == 'causal'
        decision = getattr(config, 'decision_protocol', 'autojev-v1') == 'lemondecider-v1'
        if decision:
            if linearize_patch_embed(self.backbone) != 1:
                raise ValueError('LemonDecider requires exactly one matrix-product vision patch stem')
            enable_noncausal_full_attention(self.backbone.language_model)
            self._keep_in_fp32_modules_strict = ['readout']
        self.readout = (LemonDeciderReadout if decision else torch.nn.Linear)(hidden, config.max_options, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, **kwargs):
        super().gradient_checkpointing_enable(**kwargs)
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)

    def forward(self, input_ids=None, attention_mask=None, option_counts=None,
                labels=None, return_dict=True, **kwargs):
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a nonempty [batch, length] tensor")
        if input_ids.shape[1] > self.config.max_input_length:
            raise ValueError("Input exceeds max_input_length; truncation is not allowed")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape or not bool((attention_mask[:, -1] == 1).all()):
            raise ValueError("Left padding required: last prompt position must be valid")
        if option_counts is None:
            raise ValueError("option_counts is required for each input")
        counts = torch.as_tensor(option_counts, device=input_ids.device)
        if counts.dtype == torch.bool or counts.is_floating_point() or counts.shape != (input_ids.shape[0],):
            raise ValueError("option_counts must contain one integer per row")
        if not bool(((counts >= 1) & (counts <= self.config.max_options)).all()):
            raise ValueError(f"Each question must contain 1..{self.config.max_options} options")
        if kwargs.pop("use_cache", False) or kwargs.get("past_key_values") is not None:
            raise ValueError("Decision forward does not accept KV-cache decoding")
        if getattr(self.config, "attention_mode", "causal") == "noncausal_full_attention":
            implementation = getattr(self.config, "_attn_implementation", None) or getattr(
                self.config, "_attn_implementation_internal", None
            )
            if implementation != "sdpa":
                raise ValueError("noncausal_full_attention requires SDPA")
            language = getattr(self.backbone, "language_model", None)
            if language is None or getattr(language.config, "is_causal", True):
                raise ValueError("noncausal_full_attention requires language_model.config.is_causal=False")
        vision_zero = None
        if getattr(self.config, 'vision_training_mode', 'frozen') == 'all-trainable' and kwargs.get('pixel_values') is None:
            # Execute the SAME vision FSDP units on image-free and dummy ranks.
            # A minimal synthetic patch grid has zero loss mass, not a dataset row.
            visual = self.backbone.visual
            vc = self.config.vision_config
            merge = vc.spatial_merge_size
            pixels = torch.zeros(merge * merge, vc.in_channels * vc.temporal_patch_size * vc.patch_size ** 2,
                                 device=input_ids.device, dtype=visual.dtype)
            grid = torch.tensor([[1, merge, merge]], device=input_ids.device, dtype=torch.long)
            vision_zero = visual(pixels, grid_thw=grid, return_dict=True).pooler_output.sum() * 0.0
        backbone_ids = input_ids
        if vision_zero is not None:
            # Link BEFORE the language stack, like real image embeddings, so
            # backward FSDP collective order also agrees across ranks.
            embeds = self.backbone.get_input_embeddings()(input_ids)
            kwargs['inputs_embeds'] = embeds + vision_zero.to(embeds.dtype)
            if kwargs.get('position_ids') is None:
                # Same text-only arange as QwenTextModel; avoid stale M-RoPE
                # deltas from a preceding multimodal batch when passing embeds.
                kwargs['position_ids'] = torch.arange(input_ids.shape[1], device=input_ids.device)[None, None].expand(3, input_ids.shape[0], -1)
            backbone_ids = None
        backend_context = (sdpa_backends(input_ids.device)
                           if getattr(self.config, 'decision_protocol', 'autojev-v1') == 'lemondecider-v1' else nullcontext())
        with backend_context:
            output = self.backbone(input_ids=backbone_ids, attention_mask=attention_mask,
                                   use_cache=False, return_dict=True, **kwargs)
        hidden = output.last_hidden_state[:, -1, :]
        # A separate FSDP precision island keeps the readout weights and GEMM FP32.
        # Passing FP32 also prevents any mixed-precision wrapper from downcasting logits.
        logits = self.readout(hidden.float() if getattr(self.config, 'decision_protocol', 'autojev-v1') == 'lemondecider-v1' else hidden).float()
        invalid = torch.arange(self.config.max_options, device=logits.device)[None] >= counts[:, None]
        logits = logits.masked_fill(invalid, -1e9)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if labels.shape == (len(counts),):
                if labels.is_floating_point() or labels.dtype == torch.bool or not bool(((labels >= 0) & (labels < counts)).all()):
                    raise ValueError("Hard targets must be valid option indices")
                target = F.one_hot(labels.long(), self.config.max_options).float()
                loss = option_cross_entropy(logits, target, counts)
            elif labels.shape == logits.shape:
                target = labels.float()
                if not bool(torch.isfinite(target).all()) or bool((target < 0).any()):
                    raise ValueError("Invalid soft targets")
                if not torch.allclose(target.sum(-1), torch.ones(len(counts), device=target.device), atol=1e-6, rtol=0):
                    raise ValueError("Soft targets must sum to one")
                if bool((target.masked_select(invalid) != 0).any()):
                    raise ValueError("Soft targets assign mass to invalid options")
                loss = option_cross_entropy(logits, target, counts)
            else:
                raise ValueError(f"labels must be [batch] or [batch,{self.config.max_options}]")
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return LemonDeciderOutput(loss=loss, logits=logits)


LemonDeciderConfig.register_for_auto_class()
LemonDeciderForDecisionClassification.register_for_auto_class("AutoModel")
