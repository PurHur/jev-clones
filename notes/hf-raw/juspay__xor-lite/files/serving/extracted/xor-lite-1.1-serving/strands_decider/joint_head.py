"""The joint schema head, ported from Cloudflare/clef to the strands decider.

clef renders every field of a record in one prompt and decodes them jointly.
Our pipeline renders one question per forward, so the multi-field decoder
degenerates; what ports cleanly -- and what our measured gaps actually call
for -- is the rest of the architecture:

  * lexical anchoring: option queries carry the option tokens' own embedding
    vectors, so the scorer binds to option TEXT, not to a rendering convention;
  * evidence routing: the options cross-attend the full state memory instead
    of reading only each option's last token;
  * the gated scorer: prior (lexical-anchor cosine) + sigmoid(gate) x (joint
    cosine + residual MLP), whose learned gate is calibration-shaped.

This implementation is fully batched: span means come from a gather over
padded flat indices rather than per-row python loops (the loop form measured
~75 s/batch on the training corpus; the batched form is ~50-100x fewer
kernel launches).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


class EvidenceRoutingLayer(torch.nn.Module):
    """Options cross-attend the state memory (clef verbatim)."""

    def __init__(self, width: int, heads: int, feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.memory_norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.feedforward_norm = torch.nn.LayerNorm(width)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(width, feedforward),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(feedforward, width),
            torch.nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        normalized_memory = self.memory_norm(memory)
        routed, _ = self.attention(
            normalized_queries, normalized_memory, normalized_memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        queries = queries + self.attention_dropout(routed)
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointHead(torch.nn.Module):
    """clef's joint schema head, per-question form, batched.

    forward() takes:
      hidden       [B, L, d]  the torso's final hidden states (any dtype)
      input_ids    [B, L]     the prompt tokens (for lexical embeddings)
      attention_mask [B, L]
      opt_span     [B, K, 2]  (first, last+1) token index per option; -1 padded
      q_span       [B, 2]     the question text's (first, last+1) token span
      q_type       [B]        0 noul / 1 choice / 2 score
      embed_weight [V, d]     the torso's input embedding matrix
    and returns logits [B, K] (masked softmax happens downstream, as for the
    pointer head).
    """

    MAX_SPAN = 64  # options longer than this many tokens pool over a prefix

    def __init__(
        self,
        hidden_size: int,
        width: int = 512,
        routing_layers: int = 2,
        layers: int = 2,
        heads: int = 8,
        feedforward: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_norm = torch.nn.LayerNorm(hidden_size)
        self.memory_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.global_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = torch.nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = torch.nn.Embedding(3, width)
        self.evidence_layers = torch.nn.ModuleList(
            [
                EvidenceRoutingLayer(width, heads, feedforward, dropout)
                for _ in range(routing_layers)
            ]
        )
        self.option_summary_norm = torch.nn.LayerNorm(width)
        self.field_layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerDecoderLayer(
                    d_model=width,
                    nhead=heads,
                    dim_feedforward=feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.field_norm = torch.nn.LayerNorm(width)
        self.option_norm = torch.nn.LayerNorm(width)
        self.residual_scorer = torch.nn.Sequential(
            torch.nn.Linear(width * 4, width),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, 1),
        )
        self.prior_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.residual_gate = torch.nn.Parameter(torch.zeros(()))

    def _span_gather(
        self, seq: torch.Tensor, starts: torch.Tensor, lens: torch.Tensor
    ) -> torch.Tensor:
        """Mean-pool spans: seq [B, L, d], starts/lens [B, K] -> [B, K, d].

        Padded options (start < 0) return zero vectors; the caller masks them
        out of the loss via the -1e9 logit fill.
        """
        b, k = starts.shape
        device = seq.device
        span_len = self.MAX_SPAN
        offsets = torch.arange(span_len, device=device).view(1, 1, span_len)
        valid = (offsets < lens.clamp_max(span_len).unsqueeze(-1)) & (starts >= 0).unsqueeze(-1)
        # NOTE: no backward clamp on starts -- options live at the END of the
        # prompt, so clamping start to L-span_len would move their gather onto
        # wrong tokens whenever an option begins inside the last span_len tokens
        # (every short row at serve time). `flat` is already clamped to L-1.
        safe_starts = starts.clamp_min(0)
        flat = (safe_starts.unsqueeze(-1) + offsets).clamp_max(seq.size(1) - 1)  # [B,K,S]
        flat = flat.view(b, k * span_len)
        gathered = seq.gather(1, flat.unsqueeze(-1).expand(-1, -1, seq.size(-1)))
        gathered = gathered.view(b, k, span_len, seq.size(-1))
        counts = valid.sum(-1).clamp_min(1)
        return (gathered * valid.unsqueeze(-1)).sum(dim=2) / counts.unsqueeze(-1)

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        opt_span: torch.Tensor,
        q_span: torch.Tensor,
        q_type: torch.Tensor,
        embed_weight: torch.Tensor,
    ) -> torch.Tensor:
        device = hidden.device
        batch, length, _ = hidden.shape

        starts = opt_span[..., 0]
        ends = opt_span[..., 1]
        lens = (ends - starts).clamp_min(0)
        opt_valid = starts >= 0  # [B, K]

        hidden = hidden.to(torch.float32)
        embed_weight = embed_weight.to(torch.float32)
        seq = self.hidden_norm(hidden)
        memory = self.memory_projection(seq)  # [B, L, w]
        mem_pad = attention_mask == 0  # [B, L]

        # span means: contexts from hidden, lexicals from the embedding matrix
        contexts = self._span_gather(seq, starts, lens)  # [B, K, d]
        emb_rows = embed_weight[input_ids]  # [B, L, d]
        lexicals = self._span_gather(emb_rows, starts, lens)  # [B, K, d]

        q_starts = q_span[:, 0].unsqueeze(1)
        q_lens = (q_span[:, 1] - q_span[:, 0]).clamp_min(1).unsqueeze(1)
        question_vectors = self._span_gather(seq, q_starts, q_lens)[:, 0]  # [B, d]

        lengths = attention_mask.sum(dim=1, keepdim=True)  # [B, 1]
        last_idx = (lengths - 1).clamp_min(0)
        global_vector = seq.gather(
            1, last_idx.unsqueeze(-1).expand(-1, 1, seq.size(-1))
        ).squeeze(1)  # [B, d]

        option_queries = (
            self.option_context_projection(contexts)
            + self.option_lexical_projection(lexicals)
            + self.option_question_projection(question_vectors).unsqueeze(1)
        )  # [B, K, w]
        opt_pad = ~opt_valid
        for layer in self.evidence_layers:
            option_queries = layer(option_queries, memory, mem_pad)
        routed = option_queries  # [B, K, w]

        base_fields = self.question_projection(question_vectors)  # [B, w]
        routing_weights = functional.softmax(
            (routed * base_fields.unsqueeze(1)).sum(-1) / math.sqrt(routed.shape[-1]),
            dim=1,
        ).unsqueeze(-1)  # [B, K, 1]
        option_summary = (routing_weights * routed).sum(dim=1)  # [B, w]
        fields = (
            base_fields
            + self.option_summary_norm(option_summary)
            + self.global_projection(global_vector)
            + self.type_embedding(q_type)
        ).unsqueeze(1)  # [B, 1, w]
        for layer in self.field_layers:
            fields = layer(
                fields, memory,
                memory_key_padding_mask=mem_pad,
            )
        field = self.field_norm(fields[:, 0])  # [B, w]

        anchor = functional.normalize(question_vectors + global_vector, dim=-1)  # [B, d]
        lexical_anchor = functional.normalize(lexicals, dim=-1)  # [B, K, d]
        prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        prior = prior_scale * torch.einsum("bkd,bd->bk", lexical_anchor, anchor)  # [B, K]

        options = self.option_norm(routed)  # [B, K, w]
        repeated_field = field.unsqueeze(1).expand_as(options)  # [B, K, w]
        cosine = functional.cosine_similarity(repeated_field, options, dim=-1)  # [B, K]
        features = torch.cat(
            [
                repeated_field,
                options,
                repeated_field * options,
                torch.abs(repeated_field - options),
            ],
            dim=-1,
        )
        residual = self.residual_scorer(features).squeeze(-1)  # [B, K]
        joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
        joint = joint_scale * cosine + residual
        logits = prior + torch.sigmoid(self.residual_gate) * joint
        return torch.where(opt_valid, logits, torch.full_like(logits, -1e9))
