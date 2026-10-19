"""Maincode MATILDA configuration aliases for the Transformers backbone."""
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig,
)


class MatildaJevTextConfig(Qwen3_5TextConfig):
    model_type = "matilda_jev_text"


class MatildaJevVisionConfig(Qwen3_5VisionConfig):
    model_type = "matilda_jev_vision"


class MatildaJevConfig(Qwen3_5Config):
    model_type = "matilda_jev"
    sub_configs = {"text_config": MatildaJevTextConfig, "vision_config": MatildaJevVisionConfig}


MatildaJevConfig.register_for_auto_class("AutoConfig")
