"""Vela 2.0 decoder member: transformers config (AutoConfig via trust_remote_code).

Holds config.json as is: `backbone_config` (the Qwen3.5 text geometry of the Decision 2.0 base, a plain dict passed to
Qwen3_5TextConfig), the head sizes, the prompt layout strings, the 16,384-token input limit and the span window settings.
vela2_inference.Vela2 reads the same config.json directly, so nothing here changes how the engine runs.
"""
from transformers import PretrainedConfig


class Vela2DecoderConfig(PretrainedConfig):
    model_type = "vela2-decoder"

    def __init__(self, backbone_config=None, head_dim=256, span_head=None, max_length=16384, pad_token_id=None, **kwargs):
        self.backbone_config = backbone_config
        self.head_dim = head_dim
        self.span_head = span_head or {"d": 256, "slots": 64}
        # every other config.json key (layout, span_windows, weights_dtype, provenance) is kept as an attribute by
        # PretrainedConfig
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        # the 16,384-token input limit; set after super() because transformers treats max_length as a generation
        # parameter (4.x overwrites it with its default of 20 unless passed through, 5.x drops it from the config)
        self.max_length = max_length
