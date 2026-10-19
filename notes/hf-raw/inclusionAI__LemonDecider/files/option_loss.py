"""Shared valid-option FP32 CE for trainer and standalone HF forward(labels=...)."""
import torch
import torch.nn.functional as F

MAX_OPTIONS = 255


def option_cross_entropy(logits, target, counts, reduction='mean'):
    """FP32 masked CE. Same AutoJev objective; strict target validation added.

    Normalize over valid options ONLY, then zero padded log-probabilities
    before multiplying targets (avoids 0 * -inf). This is the AutoJev CE
    objective without reliance on a finite mask dominating valid logits.
    Padded logits have zero gradient. Each decision contributes one loss,
    regardless of scale length; there is no per-option or per-type weighting.
    """
    if logits.ndim != 2 or not 1 <= logits.shape[1] <= MAX_OPTIONS or target.shape != logits.shape or not len(logits):
        raise ValueError('Expected nonempty logits and targets of shape [B,N], 1 <= N <= 255')
    width = logits.shape[1]
    counts = torch.as_tensor(counts, device=logits.device)
    if counts.shape != (len(logits),) or counts.dtype == torch.bool or counts.is_floating_point():
        raise ValueError('One integer option count required per row')
    if not bool(((counts >= 1) & (counts <= width)).all()):
        raise ValueError(f'Option counts must be 1..{width}')
    invalid = torch.arange(width, device=logits.device)[None] >= counts[:, None]
    values = target.to(device=logits.device, dtype=torch.float32)
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()) or bool((values > 1).any()):
        raise ValueError('Nonfinite/negative/out-of-range target')
    if not torch.allclose(values.sum(-1), torch.ones(len(values), device=values.device), atol=1e-6, rtol=0):
        raise ValueError('Target mass must sum to one')
    if bool((values.masked_select(invalid) != 0).any()):
        raise ValueError('Target mass on padded options')
    if not bool(torch.isfinite(logits.masked_select(~invalid)).all()):
        raise ValueError('Nonfinite valid logits')
    masked = logits.float().masked_fill(invalid, -torch.inf)
    logp = F.log_softmax(masked, dim=-1).masked_fill(invalid, 0.0)
    loss = -(values * logp).sum(-1)
    if reduction == 'none':
        return loss
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'mean':
        return loss.mean()
    raise ValueError(f'Unknown reduction: {reduction}')

