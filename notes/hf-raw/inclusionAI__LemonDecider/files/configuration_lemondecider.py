"""Configuration for a Qwen or Llama backbone with a dynamic-option decision head."""
import math
from transformers.configuration_utils import PreTrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config


_LLAMA_KEYS = {
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "hidden_act",
    "max_position_embeddings", "initializer_range", "rms_norm_eps", "use_cache",
    "pad_token_id", "bos_token_id", "eos_token_id", "pretraining_tp",
    "tie_word_embeddings", "rope_parameters", "rope_scaling", "rope_theta",
    "attention_bias", "attention_dropout", "mlp_bias", "head_dim",
}


def _apply_decision_fields(obj, *, max_options, code_scheme, temperature, codes, token_ids,
                           base_model, initialization_step, max_input_length, vocab_size,
                           attention_mode="causal", pooling="last"):
    if type(max_options) is not int or not 1 <= max_options <= 255:
        raise ValueError("max_options must be an integer in 1..255")
    if code_scheme not in ("letters-v1", "number30-v1"):
        raise ValueError(f"Unknown answer code scheme: {code_scheme}")
    if code_scheme == "number30-v1" and max_options != 30:
        raise ValueError("number30-v1 requires exactly 30 option positions")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if attention_mode not in ("causal", "noncausal_full_attention"):
        raise ValueError(f"Unknown attention mode: {attention_mode}")
    if pooling not in ("last",):
        raise ValueError(f"Unknown pooling mode: {pooling}")
    codes = list(codes or [])
    token_ids = list(token_ids or [])
    if codes or token_ids:
        if len(codes) != max_options or len(set(codes)) != max_options:
            raise ValueError(f"Expected {max_options} unique answer codes")
        if len(token_ids) != max_options or len(set(token_ids)) != max_options:
            raise ValueError(f"Expected {max_options} unique answer token IDs")
        if any(type(i) is not int or not 0 <= i < vocab_size for i in token_ids):
            raise ValueError("Answer token IDs outside base vocabulary")
    max_input_length = int(max_input_length)
    if max_input_length < 1:
        raise ValueError("max_input_length must be positive")
    obj.code_scheme = code_scheme
    obj.max_options = max_options
    obj.codes = codes
    obj.token_ids = token_ids
    obj.temperature = float(temperature)
    obj.base_model = base_model
    obj.initialization_step = initialization_step
    obj.max_input_length = max_input_length
    obj.attention_mode = attention_mode
    obj.pooling = pooling


def _llama_requested(backbone_type, llama_config, kwargs):
    if backbone_type == "llama" or llama_config is not None:
        return True
    if backbone_type in ("qwen3_5",):
        return False
    if kwargs.get("text_config") is not None or kwargs.get("vision_config") is not None:
        return False
    return False


def _llama_from_kwargs(llama_config, kwargs):
    llama_kwargs = dict(llama_config or {})
    llama_kwargs.update({k: kwargs.pop(k) for k in list(kwargs) if k in _LLAMA_KEYS})
    llama_kwargs.pop("model_type", None)
    llama_kwargs.pop("architectures", None)
    if llama_kwargs.get("rope_scaling") is None:
        llama_kwargs.pop("rope_scaling", None)
    if "rope_theta" in llama_kwargs and "rope_parameters" not in llama_kwargs:
        llama_kwargs["rope_parameters"] = {"rope_theta": llama_kwargs.pop("rope_theta"), "rope_type": "default"}
    return LlamaConfig(**llama_kwargs)


class LemonDeciderConfig(Qwen3_5Config):
    model_type = "lemondecider_decision"

    def __init__(self, codes=None, token_ids=None, max_options=255, temperature=1.0,
                 base_model="Qwen/Qwen3.8-27B", initialization_step=0,
                 max_input_length=8192, code_scheme="letters-v1",
                 backbone_type=None, llama_config=None,
                 attention_mode="causal", pooling="last", decision_protocol="autojev-v1", **kwargs):
        # auto_map is the serialized trust_remote_code contract of every published
        # checkpoint: the serving/eval stack reloads exported configs via AutoModel.
        # Stash instead of dropping it, then re-attach after the base constructors
        # (which do not accept it on this fork of Llama/Qwen config init paths).
        auto_map = kwargs.pop("auto_map", None)
        project_name = kwargs.pop("project_name", None)
        model_id = kwargs.pop("model_id", None)
        attention_mode = kwargs.pop("attention_mode", attention_mode)
        pooling = kwargs.pop("pooling", pooling)
        if "torch_dtype" in kwargs and "dtype" not in kwargs:
            kwargs["dtype"] = kwargs.pop("torch_dtype")
        llama_mode = _llama_requested(backbone_type, llama_config, kwargs)
        if llama_mode:
            if attention_mode != "causal":
                raise ValueError("noncausal_full_attention is only implemented for the Qwen3.5 hybrid backbone")
            inner = _llama_from_kwargs(llama_config, kwargs)
            kwargs.pop("text_config", None)
            kwargs.pop("vision_config", None)
            PreTrainedConfig.__init__(self, **kwargs)
            object.__setattr__(self, "backbone_type", "llama")
            object.__setattr__(self, "llama_config", inner)
            vocab_size = inner.vocab_size
        else:
            kwargs.pop("llama_config", None)
            super().__init__(**kwargs)
            object.__setattr__(self, "backbone_type", "qwen3_5")
            object.__setattr__(self, "llama_config", None)
            vocab_size = super().__getattribute__("text_config").vocab_size
        _apply_decision_fields(
            self, max_options=max_options, code_scheme=code_scheme, temperature=temperature,
            codes=codes, token_ids=token_ids, base_model=base_model,
            initialization_step=initialization_step, max_input_length=max_input_length,
            vocab_size=vocab_size, attention_mode=attention_mode, pooling=pooling,
        )
        if decision_protocol not in ('autojev-v1', 'lemondecider-v1'):
            raise ValueError('Unknown decision protocol')
        if decision_protocol == 'lemondecider-v1' and (self.backbone_type != 'qwen3_5' or max_options != 255
                or code_scheme != 'letters-v1' or attention_mode != 'noncausal_full_attention'):
            raise ValueError('LemonDecider requires Qwen3.5-family, 255 letter codes and noncausal full attention')
        self.decision_protocol = decision_protocol
        self._sync_text_causality()
        if auto_map is not None:
            self.auto_map = auto_map
        if project_name is not None:
            self.project_name = project_name
        if model_id is not None:
            self.model_id = model_id

    def __getattribute__(self, name):
        if name == "text_config":
            data = object.__getattribute__(self, "__dict__")
            if data.get("backbone_type") == "llama":
                return data["llama_config"]
        return super().__getattribute__(name)

    @property
    def inner_backbone_config(self):
        if getattr(self, "backbone_type", "qwen3_5") == "llama":
            return self.llama_config
        return self

    def _sync_text_causality(self):
        """Qwen3.5 full-attention reads is_causal from the nested text config."""
        if getattr(self, "backbone_type", "qwen3_5") != "qwen3_5":
            return
        text = super().__getattribute__("text_config")
        text.is_causal = self.attention_mode == "causal"

    def to_dict(self):
        if getattr(self, "backbone_type", "qwen3_5") == "llama":
            output = PreTrainedConfig.to_dict(self)
            output["backbone_type"] = "llama"
            output["llama_config"] = self.llama_config.to_dict()
            output.pop("text_config", None)
            output.pop("vision_config", None)
            return output
        output = super().to_dict()
        output["backbone_type"] = "qwen3_5"
        output.pop("llama_config", None)
        output["attention_mode"] = getattr(self, "attention_mode", "causal")
        output["pooling"] = getattr(self, "pooling", "last")
        text = output.get("text_config")
        if isinstance(text, dict):
            text["is_causal"] = output["attention_mode"] == "causal"
        return output
