# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
from torch import nn
from transformers import Mistral3Config, Mistral3Model
from transformers.models.mistral3.modeling_mistral3 import Mistral3PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.conversion_mapping import register_checkpoint_conversion_mapping, WeightRenaming

class Mistral3ForSequenceClassification(Mistral3PreTrainedModel):
    config_class = Mistral3Config
    base_model_prefix = "model"
    def __init__(self, config):
        super().__init__(config)
        self.model = Mistral3Model(config)
        self.score = nn.Linear(config.text_config.hidden_size, 1, bias=False)
        self.post_init()
    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, image_sizes=None, position_ids=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_sizes=image_sizes, position_ids=position_ids, **kwargs)
        hidden = outputs.last_hidden_state
        pooled = hidden.masked_fill(~attention_mask[..., None].bool(), 0).sum(1) / attention_mask.sum(1)[:,None]
        logits = torch.nn.functional.linear(pooled.float(), self.score.weight.float())
        return SequenceClassifierOutputWithPast(logits=logits)

register_checkpoint_conversion_mapping(
    'Mistral3ForSequenceClassification',
    [WeightRenaming(source_patterns=r'^language_model\.score\.weight$', target_patterns='score.weight')],
    overwrite=True,
)
