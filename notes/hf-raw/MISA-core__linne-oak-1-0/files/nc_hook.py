"""Noncausal cho lop full_attention (kieu pplx-decider v1.1): mask hai chieu chi che padding; lop sliding/linear giu mask goc.

Hook dat tren wrapper (Gemma4Model / Qwen3_5Model, cung object train_lm va linne_engine goi), truoc khi wrapper tu dung dict mask.
Can attn_implementation="sdpa": sdpa co mask thi khong ap causal.
"""
import torch
from transformers.masking_utils import create_masks_for_generate


def enable_noncausal(wrapper):
    cfg = wrapper.config.get_text_config() if hasattr(wrapper.config, "get_text_config") else wrapper.config
    assert getattr(cfg, "_attn_implementation", "sdpa") == "sdpa", "noncausal can sdpa"

    def pre(module, args, kwargs):
        if args or kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("noncausal: chi goi keyword, khong KV cache")
        ids, emb = kwargs.get("input_ids"), kwargs.get("inputs_embeds")
        B, L = (ids.shape if ids is not None else emb.shape[:2])
        dev = ids.device if ids is not None else emb.device
        pad = kwargs.get("attention_mask")
        if pad is None:
            pad = torch.ones((B, L), dtype=torch.long, device=dev)
        if isinstance(pad, dict):
            return args, kwargs
        like = torch.empty((B, L, 1), dtype=next(module.parameters()).dtype, device=dev)
        pos = kwargs.get("position_ids")
        if pos is None:
            pos = torch.arange(L, device=dev).unsqueeze(0).expand(B, -1)
        m = create_masks_for_generate(config=cfg, inputs_embeds=like, attention_mask=pad, past_key_values=None, position_ids=pos)
        if not isinstance(m, dict):
            m = {"full_attention": m}
        m = dict(m)
        m["full_attention"] = pad[:, None, None, :].bool()
        kwargs["attention_mask"] = m
        return args, kwargs

    wrapper.register_forward_pre_hook(pre, with_kwargs=True)
