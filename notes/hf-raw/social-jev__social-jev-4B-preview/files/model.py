"""Self-contained HF export of the packed linear and MLP decision models."""

from functools import cached_property

import torch
from torch.nn.attention.flex_attention import create_mask
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .attention import AttentionMask, Positions
from .config import SocialJevConfig
from .data import Processor
from .head import DecisionHead, DecisionHeadConfig
from .types import Example, Query, Request, Result


class SocialJevModel(PreTrainedModel):
    """Returns typed outputs for typed requests. Supply a state and queries.

    Results are indexed by request and query, including for a single request.

    Args:
        config: Backbone, decision-head, and message-format settings.
            Restored automatically when loading a checkpoint.

    Examples:
        >>> model = AutoModel.from_pretrained(repo, trust_remote_code=True)
        >>> request = model.Request(
        ...     state="We will expand the bus network.",
        ...     queries=[
        ...         model.Query(
        ...             text="Which policy area does this statement concern?",
        ...             options=["Education", "Transport"],
        ...         )
        ...     ],
        ... )
        >>> result = model.query(request)[0][0]
        >>> result.choice in request.queries[0].options
        True
    """

    Request = Request
    """Construct a typed request without importing an external package."""
    Query = Query
    """Define a classification question and candidate answers."""
    Example = Example
    """Supply a labeled example for an exported model's query."""
    Result = Result
    """Type of each classification result returned by query()."""

    config_class = SocialJevConfig
    base_model_prefix = "backbone"
    _supports_sdpa = True
    _supports_flex_attn = True

    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_config(
            config.backbone_config,
            attn_implementation=config._attn_implementation or "sdpa",
        )
        self.decision_head = DecisionHead(
            DecisionHeadConfig(**config.decision_head), config.backbone_config.hidden_size
        )
        self.post_init()

    @cached_property
    def processor(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.name_or_path,
            config=self.config.backbone_config,
            revision=getattr(self.config, "_commit_hash", None),
        )
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        return Processor(tokenizer, self.config)

    def forward(self, batch):
        mask = AttentionMask(batch).value
        if self.backbone.config._attn_implementation != "flex_attention":
            size, length = batch.input_ids.shape
            mask = create_mask(
                mask.mask_mod, size, 1, length, length, device=str(batch.input_ids.device)
            )
        hidden = self.backbone(
            input_ids=batch.input_ids,
            position_ids=Positions(batch).value,
            attention_mask={"full_attention": mask},
            use_cache=False,
        ).last_hidden_state
        positions = batch.option_positions
        options = hidden[positions[:, 0], positions[:, 1]]
        dtype = next(self.decision_head.parameters()).dtype
        logits = self.decision_head(options.to(dtype))
        indices = batch.option_indices
        return logits[indices.clamp(min=0)].masked_fill(indices < 0, -torch.inf)

    @torch.inference_mode()
    def query(self, requests: Request | list[Request]) -> list[list[Result]]:
        encoded = self.processor.encode(requests)
        scores = self(encoded.batch.to(self.device))
        return self.processor.decode(scores, encoded)


SocialJevModel.register_for_auto_class("AutoModel")
