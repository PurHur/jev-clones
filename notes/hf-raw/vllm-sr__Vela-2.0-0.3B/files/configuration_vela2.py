"""Vela 2.0 unified schema encoder: transformers config (AutoConfig via trust_remote_code).

Holds config.json as is: `encoder_config` (the ModernBERT geometry, a plain dict passed to ModernBertConfig), the
readout sizes, the marker ids, the sequence layout, the 8,192 window with its overlap and the learned temperatures.
vela2_inference.Vela2 reads the same config.json directly, so nothing here changes how the engine runs.
"""
from transformers import PretrainedConfig


class Vela2Config(PretrainedConfig):
    model_type = "vela2-unified"

    def __init__(self, encoder_config=None, proj_dim=256, readout="both", span_readout="word", mlp_dropout=0.1,
                 marker_ids=None, max_length=8192, window_overlap=512, bos_token_id=2, eos_token_id=1, pad_token_id=0, **kwargs):
        self.encoder_config = encoder_config
        self.proj_dim = proj_dim
        self.readout = readout
        self.span_readout = span_readout
        self.mlp_dropout = mlp_dropout
        self.marker_ids = marker_ids
        self.window_overlap = window_overlap
        # every other config.json key (markers, window_agg, layout, tau, tau_span, provenance) is kept as an attribute
        # by PretrainedConfig
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)
        # the 8,192-token sequence window; set after super() because transformers treats max_length as a generation
        # parameter (4.x overwrites it with its default of 20 unless passed through, 5.x drops it from the config)
        self.max_length = max_length
