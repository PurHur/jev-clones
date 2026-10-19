"""Configuration for the Efficient-DLM encoder used by Drex DLM."""

from transformers import Qwen3Config


class EfficientDLMConfig(Qwen3Config):
    model_type = "edlm"

    def __init__(
        self,
        block_size=32,
        dlm_arch="encoder",
        dlm_paradigm="bidirectional",
        dlm_type="llada",
        mask_token_id=151662,
        **kwargs,
    ):
        rope_parameters = kwargs.get("rope_parameters")
        super().__init__(**kwargs)
        if isinstance(rope_parameters, dict) and rope_parameters.get("rope_theta") is not None:
            self.rope_theta = rope_parameters["rope_theta"]
        self.block_size = block_size
        self.dlm_arch = dlm_arch
        self.dlm_paradigm = dlm_paradigm
        self.dlm_type = dlm_type
        self.mask_token_id = mask_token_id
