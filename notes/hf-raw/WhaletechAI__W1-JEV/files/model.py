"""
Language Diffusion Transformer (DiT for text).

Public open-source version:
- pure PyTorch only
- state_dict key layout kept compatible with the internal model
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo import disable as dynamo_disable


class FallbackRMSNorm(nn.Module):
    """Minimal RMSNorm implementation used by the public model."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x_float = x.float()
        norm = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x_float * norm
        return (out.to(dtype=x_dtype) * self.weight).to(dtype=x_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(
    seqlen: int,
    dim: int,
    theta: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    t = torch.arange(seqlen, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype)[None, :, None, :]
    sin = emb.sin().to(dtype=dtype)[None, :, None, :]
    return cos, sin


class TokenEmbedding(nn.Module):
    """Token embedding (untied from output)."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding -> conditioning vector."""

    def __init__(self, cond_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, cond_dim, bias=True),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(-1)
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].to(dtype=torch.float32) * freqs[None]
        embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        embed = embed.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(embed)


class RotaryEmbedding(nn.Module):
    """Pure PyTorch RoPE implementation."""

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = int(dim)
        self.theta = float(theta)
        self.max_seq_len = max_seq_len

    def apply_bshd(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = _rope_cos_sin(
            q.shape[1],
            q.shape[-1],
            self.theta,
            device=q.device,
            dtype=q.dtype,
        )
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


class Attention(nn.Module):
    """
    Multi-head attention with expanded attention dimension.
    hidden_size -> attn_dim for Q,K,V -> hidden_size
    """

    def __init__(
        self,
        hidden_size: int,
        attn_dim: int,
        num_heads: int,
        head_dim: int = 128,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = attn_dim
        self.attn_drop = float(attn_drop)

        self.qkv = nn.Linear(hidden_size, attn_dim * 3, bias=False)
        self.proj = nn.Linear(attn_dim, hidden_size, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    @dynamo_disable
    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[RotaryEmbedding] = None,
        pack: Optional[object] = None,
    ) -> torch.Tensor:
        if pack is not None:
            raise RuntimeError("Packed attention is not included in the public torch-only model.")

        bsz, seqlen, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seqlen, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        if rope is not None:
            q, k = rope.apply_bshd(q, k)

        qh = q.permute(0, 2, 1, 3).contiguous()
        kh = k.permute(0, 2, 1, 3).contiguous()
        vh = v.permute(0, 2, 1, 3).contiguous()
        out = F.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        x = out.permute(0, 2, 1, 3).contiguous().reshape(bsz, seqlen, self.attn_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLU(nn.Module):
    """SwiGLU FFN with custom intermediate dimension."""

    def __init__(self, hidden_size: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w12 = nn.Linear(hidden_size, 2 * ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, hidden_size, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.drop(F.silu(x1) * x2))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    DiT transformer block with AdaLN-Zero modulation.
    Pre-norm: RMSNorm -> modulate -> Attention/FFN -> gate
    """

    def __init__(
        self,
        hidden_size: int,
        attn_dim: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int = 128,
        cond_dim: int = 256,
        attn_drop: float = 0.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = FallbackRMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, attn_dim, num_heads, head_dim, attn_drop, drop)
        self.norm2 = FallbackRMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLU(hidden_size, ffn_dim, dropout=drop)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rope: Optional[RotaryEmbedding] = None,
        pack: Optional[object] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope,
            pack=pack,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalAdaLN(nn.Module):
    """Final AdaLN block that produces prelogits."""

    def __init__(self, hidden_size: int, cond_dim: int):
        super().__init__()
        self.norm = FallbackRMSNorm(hidden_size, eps=1e-6)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


class LangDiT(nn.Module):
    """Language Diffusion Transformer."""

    def __init__(
        self,
        vocab_size: int = 64512,
        hidden_size: int = 2048,
        attn_dim: int = 3072,
        ffn_dim: int = 7168,
        depth: int = 48,
        num_heads: int = 24,
        head_dim: int = 128,
        max_seq_len: int = 4096,
        timestep_freq_dim: int = 256,
        rope_theta: float = 10000.0,
        cond_dim: int = 256,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.token_embed = TokenEmbedding(vocab_size, hidden_size)
        self.time_embed = TimestepEmbedding(cond_dim, freq_dim=timestep_freq_dim)
        self.rope = RotaryEmbedding(head_dim, max_seq_len, theta=rope_theta)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    attn_dim=attn_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    cond_dim=cond_dim,
                    attn_drop=attn_dropout,
                    drop=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_ada = FinalAdaLN(hidden_size, cond_dim)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        def init_fn(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        self.apply(init_fn)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        nn.init.zeros_(self.final_ada.adaLN[-1].weight)
        nn.init.zeros_(self.final_ada.adaLN[-1].bias)

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        pack: Optional[object] = None,
    ) -> torch.Tensor:
        if pack is not None:
            raise RuntimeError("Packed attention is not included in the public torch-only model.")

        x = self.token_embed(input_ids)
        c = self.time_embed(timesteps)

        for block in self.blocks:
            x = block(x, c, self.rope, pack=None)

        return self.final_ada(x, c)

    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        pack: Optional[object] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        hidden = self.forward_hidden(input_ids, timesteps, pack=pack)
        if return_hidden:
            return hidden
        return self.logits_from_hidden(hidden)


def create_model(config: dict) -> LangDiT:
    """Create model from config dict."""
    model_cfg = config["model"]
    return LangDiT(
        vocab_size=model_cfg["vocab_size"],
        hidden_size=model_cfg["hidden_size"],
        attn_dim=model_cfg["attn_dim"],
        ffn_dim=model_cfg["ffn_dim"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        head_dim=model_cfg["head_dim"],
        max_seq_len=model_cfg["max_seq_len"],
        timestep_freq_dim=model_cfg.get("timestep_freq_dim", 256),
        rope_theta=model_cfg.get("rope_theta", 10000.0),
        cond_dim=model_cfg["cond_dim"],
        dropout=model_cfg.get("dropout", 0.0),
        attn_dropout=model_cfg.get("attn_dropout", 0.0),
    )
