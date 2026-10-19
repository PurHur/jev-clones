"""Efficient-DLM encoder.

The blocks use the Qwen3 tensor layout. The attention does not. With no mask,
every token can see every other token. A caller can still pass its own mask.
"""

import torch
from transformers import Qwen3PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from configuration_edlm import EfficientDLMConfig


class EfficientDLMModel(Qwen3Model):
    config_class = EfficientDLMConfig

    def forward(self, *args, attention_mask=None, **kwargs):
        if attention_mask is None and getattr(self.config, "dlm_paradigm", None) == "bidirectional":
            attention_mask = self._full_mask(*args, **kwargs)
        return super().forward(*args, attention_mask=attention_mask, **kwargs)

    def _full_mask(self, *args, **kwargs):
        # Qwen3Model turns a missing mask into a causal mask. A zero 4D mask is returned as-is.
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        inputs_embeds = kwargs.get("inputs_embeds")
        past = kwargs.get("past_key_values")
        if inputs_embeds is not None:
            batch, query, _ = inputs_embeds.shape
            device = inputs_embeds.device
            dtype = inputs_embeds.dtype
        else:
            batch, query = input_ids.shape
            device = input_ids.device
            dtype = next(self.parameters()).dtype
        seen = past.get_seq_length() if past is not None else 0
        return torch.zeros(batch, 1, query, seen + query, dtype=dtype, device=device)


class EfficientDLM(Qwen3PreTrainedModel):
    config_class = EfficientDLMConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = EfficientDLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        # Same position, not the next token. This head is not the decision pointer.
        logits = self.lm_head(out.last_hidden_state)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=out.past_key_values,
            hidden_states=out.last_hidden_state,
        )
