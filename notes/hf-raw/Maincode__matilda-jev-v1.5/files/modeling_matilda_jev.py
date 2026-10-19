"""MATILDA backbone alias. The separate JEV decision readout is loaded by the bundled runtime."""
from transformers import AutoModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5TextModel, Qwen3_5VisionModel
from .configuration_matilda_jev import MatildaJevConfig, MatildaJevTextConfig, MatildaJevVisionConfig


class MatildaJevTextModel(Qwen3_5TextModel):
    config_class = MatildaJevTextConfig


class MatildaJevVisionModel(Qwen3_5VisionModel):
    config_class = MatildaJevVisionConfig


# The multimodal backbone instantiates both submodels through AutoModel.
AutoModel.register(MatildaJevTextConfig, MatildaJevTextModel, exist_ok=True)
AutoModel.register(MatildaJevVisionConfig, MatildaJevVisionModel, exist_ok=True)


class MatildaJevModel(Qwen3_5Model):
    config_class = MatildaJevConfig

    def __init__(self, config):
        # Local dynamic modules may load configuration and model dependencies into
        # different content-hash namespaces. Register the actual config types too.
        AutoModel.register(type(config.text_config), MatildaJevTextModel, exist_ok=True)
        AutoModel.register(type(config.vision_config), MatildaJevVisionModel, exist_ok=True)
        super().__init__(config)


MatildaJevModel.register_for_auto_class("AutoModel")
