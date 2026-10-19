"""Two-way attention over the request (contract key "prefix_attention": "bidirectional_v1").

A Nimble prompt is the request (system prompt, context and every field of the schema), then
"Requested field:" and one field's name ahead of the assistant turn. Read one way, the context is encoded
before any field has been seen. With this on, every request token, up to and including the last
"Requested field:", sees every other request token in the full-attention layers. The field name and the
chat tail after it stay one-way, as do the recurrent (Gated DeltaNet) layers. Request tokens never see a
field name, so the request part is the same for all of a request's fields and is still encoded once.

The mask is the library's block mask (`block_sequence_ids`: tokens of one block see each other both ways,
-1 marks one-way tokens), handed to the Qwen3.5 text model's mask builder by a small wrapper.
"""
import contextlib
import threading

import torch

MODE = "bidirectional_v1"
MARKER_TEXT = "Requested field:"
_local = threading.local()
_marker = None


def enable(tokenizer):
    """Switch two-way request attention on for this process's later forward passes."""
    global _marker
    _marker = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    install()


def disable():
    global _marker
    _marker = None


def enabled():
    return _marker is not None


def configure(mode, tokenizer):
    """Apply a plan's or contract's `prefix_attention` value; None means one-way, as before."""
    if mode is None:
        disable()
    elif mode == MODE:
        enable(tokenizer)
    else:
        raise ValueError(f"Unknown prefix_attention {mode!r}")


def install():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    if getattr(qwen.create_causal_mask, "nimble_blocks", False):
        return
    original = qwen.create_causal_mask

    def create_causal_mask(*args, **kwargs):
        if getattr(_local, "whole", False):
            mask = kwargs.get("attention_mask")
            if mask is not None and not bool(mask.all()):
                raise ValueError("A whole-request forward must be one unpadded row")
            return None  # with is_causal off, SDPA attends both ways without a mask
        blocks = getattr(_local, "blocks", None)
        if blocks is not None:
            kwargs["block_sequence_ids"] = blocks
        return original(*args, **kwargs)

    create_causal_mask.nimble_blocks = True
    qwen.create_causal_mask = create_causal_mask


def request_ends(input_ids):
    """Index just past the last "Requested field:" in each row."""
    marker = torch.tensor(_marker, device=input_ids.device)
    hits = (input_ids.unfold(1, len(_marker), 1) == marker).all(-1)
    starts = torch.arange(hits.shape[1], device=input_ids.device)
    last = torch.where(hits, starts, -1).max(1).values
    if bool((last < 0).any()):
        raise ValueError(f"A prompt has no {MARKER_TEXT!r} marker")
    return last + len(_marker)


def block_ids(input_ids, attention_mask=None):
    """0 for request tokens (one two-way block per row), -1 for padding and for the field's tail."""
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    inside = positions[None, :] < request_ends(input_ids)[:, None]
    if attention_mask is not None:
        inside &= attention_mask.bool()
    return torch.where(inside, 0, -1)


@contextlib.contextmanager
def request_blocks(input_ids, attention_mask=None):
    """Forward passes inside read each row's request both ways; does nothing when switched off."""
    if _marker is None:
        yield
        return
    _local.blocks = block_ids(input_ids, attention_mask)
    try:
        yield
    finally:
        _local.blocks = None


@contextlib.contextmanager
def whole_request(model):
    """A forward over a request part alone, one unpadded row, every token seeing every other.

    Full-attention layers run without a mask and with is_causal off, so SDPA keeps its fused kernels
    instead of reading a materialized square mask.
    """
    if _marker is None:
        yield
        return
    layers = [m for m in model.modules() if type(m).__name__ == "Qwen3_5Attention"]
    if not layers:
        raise ValueError("No Qwen3.5 full-attention layers found")
    _local.whole = True
    for layer in layers:
        layer.is_causal = False
    try:
        yield
    finally:
        _local.whole = False
        for layer in layers:
            layer.is_causal = True
