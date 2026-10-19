"""Inference batching with multiple answer slots per input.

Right-padded, bidirectional batches; padding never becomes an attention key.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def attention(module, x, rope, valid_keys):
    bsz, length, _ = x.shape
    q, k, v = module.qkv(x).reshape(bsz, length, 3, module.num_heads, module.head_dim).unbind(2)
    q, k = rope.apply_bshd(q, k)
    q, k, v = [a.permute(0, 2, 1, 3).contiguous() for a in (q, k, v)]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=valid_keys,
                                         dropout_p=0.0, is_causal=False)
    return module.proj_drop(module.proj(out.permute(0, 2, 1, 3).contiguous().reshape(bsz, length, module.attn_dim)))


@torch.inference_mode()
def batch_logits(model, sequences, positions, timestep=0.5, pad_id=1):
    if model.training or not sequences or len(sequences) != len(positions):
        raise ValueError("Batch requires an eval model and answer slots per input")
    if any(not ps or any(not 0 <= p < len(s) for p in ps) for s, ps in zip(sequences, positions)):
        raise ValueError("Answer slot outside input")
    device = next(model.parameters()).device
    width = max(map(len, sequences))
    x = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        x[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    lengths = torch.tensor([len(s) for s in sequences], device=device)
    keys = (torch.arange(width, device=device)[None, :] < lengths[:, None])[:, None, None, :]
    hidden = model.token_embed(x)
    condition = model.time_embed(torch.full((len(sequences),), timestep, device=device, dtype=torch.float32))
    for block in model.blocks:
        s1, c1, g1, s2, c2, g2 = block.adaLN(condition).chunk(6, -1)
        normalized = block.norm1(hidden) * (1 + c1[:, None]) + s1[:, None]
        hidden = hidden + g1[:, None] * attention(block.attn, normalized, model.rope, keys)
        normalized = block.norm2(hidden) * (1 + c2[:, None]) + s2[:, None]
        hidden = hidden + g2[:, None] * block.mlp(normalized)
    hidden = model.final_ada(hidden, condition)
    rows = [i for i, ps in enumerate(positions) for _ in ps]
    columns = [p for ps in positions for p in ps]
    slots = hidden[torch.tensor(rows, device=device), torch.tensor(columns, device=device)]
    logits = model.logits_from_hidden(slots).float()
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite batched answer logits")
    return logits
