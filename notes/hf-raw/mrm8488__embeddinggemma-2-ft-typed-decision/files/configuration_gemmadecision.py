"""Configuration for GemmaDecision: a typed decision head on top of an EmbeddingGemma encoder."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.utils import logging

logger = logging.get_logger(__name__)

QUESTION_TYPES = ("choice", "score", "noul")


class GemmaDecisionConfig(PretrainedConfig):
    r"""
    Configuration for [`GemmaDecisionModel`].

    The model reproduces a sentence-transformers embedding pipeline (backbone → mean pooling → dense projections →
    Matryoshka truncation → L2 normalization) and puts a schema-conditioned decision head on top. Every option of a
    typed question (`choice`, `score` or `noul`) is embedded as text and scored against the state and the question.

    Args:
        encoder_config (`dict` or `PretrainedConfig`):
            Configuration of the backbone (e.g. the text-only EmbeddingGemma 2 model).
        encoder_class (`str`, *optional*, defaults to `"AutoModel"`):
            Name of the `transformers` class used to build the backbone.
        encoder_method (`str`, *optional*, defaults to `"forward"`):
            Backbone method that returns token embeddings.
        encoder_output (`list[str]`, *optional*, defaults to `["last_hidden_state"]`):
            Keys to follow in the backbone output to reach the token embeddings.
        embed_dim (`int`, *optional*, defaults to 768):
            Embedding size after Matryoshka truncation (768, 512, 256 or 128 for EmbeddingGemma 2).
        prompt (`str`, *optional*, defaults to `""`):
            Task prefix prepended to every text (the sentence-transformers `Classification` prompt).
        max_seq_length (`int`, *optional*):
            Truncation length used when tokenizing.
        pooling_mode (`str`, *optional*, defaults to `"mean"`):
            Token pooling. Only `"mean"` is supported.
        pooling_include_prompt (`bool`, *optional*, defaults to `True`):
            Whether prompt tokens take part in the mean pooling.
        projections (`list[dict]`, *optional*):
            Dense layers applied after pooling, each `{"in_features", "out_features", "bias", "activation"}`.
        head_width (`int`, *optional*, defaults to 256):
            Hidden width of the decision head.
        head_dropout (`float`, *optional*, defaults to 0.1):
            Dropout of the decision head.
        question_types (`list[str]`, *optional*, defaults to `["choice", "score", "noul"]`):
            Supported question types, in the order of the head's type embedding.
        label_smoothing, brier_weight, ordinal_weight (`float`, *optional*):
            Weights of the training loss (soft cross-entropy + Brier + ordinal distance on score questions).
    """

    model_type = "gemma_decision"
    sub_configs = {"encoder_config": AutoConfig}

    def __init__(
        self,
        encoder_config: dict[str, Any] | PretrainedConfig | None = None,
        encoder_class: str = "AutoModel",
        encoder_method: str = "forward",
        encoder_output: list[str] | None = None,
        embed_dim: int = 768,
        prompt: str = "",
        max_seq_length: int | None = None,
        pooling_mode: str = "mean",
        pooling_include_prompt: bool = True,
        projections: list[dict[str, Any]] | None = None,
        head_width: int = 256,
        head_dropout: float = 0.1,
        question_types: list[str] | None = None,
        label_smoothing: float = 0.0,
        brier_weight: float = 0.25,
        ordinal_weight: float = 0.25,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        if isinstance(encoder_config, dict):
            model_type = encoder_config.get("model_type")
            if model_type is None:
                raise ValueError("`encoder_config` needs a `model_type` key")
            encoder_config = CONFIG_MAPPING[model_type].from_dict(encoder_config)
        self.encoder_config = encoder_config
        self.encoder_class = encoder_class
        self.encoder_method = encoder_method
        self.encoder_output = list(encoder_output or ["last_hidden_state"])
        self.embed_dim = embed_dim
        self.prompt = prompt
        self.max_seq_length = max_seq_length
        self.pooling_mode = pooling_mode
        self.pooling_include_prompt = pooling_include_prompt
        self.projections = list(projections or [])
        self.head_width = head_width
        self.head_dropout = head_dropout
        self.question_types = list(question_types or QUESTION_TYPES)
        self.label_smoothing = label_smoothing
        self.brier_weight = brier_weight
        self.ordinal_weight = ordinal_weight
        self.initializer_range = initializer_range

        if self.pooling_mode != "mean":
            raise ValueError(f"Only mean pooling is supported, got {self.pooling_mode!r}")
        if self.question_types != list(QUESTION_TYPES):
            logger.warning("Custom `question_types` must match the order used to train the head.")

        super().__init__(**kwargs)


__all__ = ["GemmaDecisionConfig", "QUESTION_TYPES"]
