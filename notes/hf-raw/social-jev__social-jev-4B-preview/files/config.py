"""Nested backbone and decision-head configuration for HF exports."""

from dataclasses import asdict
from typing import ClassVar

from transformers import AutoConfig, PretrainedConfig

from .head import DecisionHeadConfig


class SocialJevConfig(PretrainedConfig):
    model_type = "social_jev"
    sub_configs: ClassVar = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config=None,
        decision_head=None,
        role_start="<|im_start|>",
        role_end="<|im_end|>",
        **kwargs,
    ):
        kwargs.pop("tie_word_embeddings", None)
        if backbone_config is None:
            backbone_config = {"model_type": "qwen3"}
        if isinstance(backbone_config, dict):
            values = dict(backbone_config)
            backbone_config = AutoConfig.for_model(values.pop("model_type"), **values)
        self.backbone_config = backbone_config
        super().__init__(**kwargs)
        self.tie_word_embeddings = False
        self.backbone_config.tie_word_embeddings = False
        self.decision_head = asdict(DecisionHeadConfig(**(decision_head or {})))
        self.role_start = role_start
        self.role_end = role_end

    def get_text_config(self, decoder=None, encoder=None):
        del decoder, encoder
        return self.backbone_config


SocialJevConfig.register_for_auto_class()
