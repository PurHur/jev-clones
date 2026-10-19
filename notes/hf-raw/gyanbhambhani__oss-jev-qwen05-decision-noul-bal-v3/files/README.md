---
library_name: transformers
license: apache-2.0
base_model: gyanbhambhani/oss-jev-qwen05-decision-noul-v2
tags:
- generated_from_trainer
model-index:
- name: oss-jev-qwen05-decision-noul-bal-v3
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# oss-jev-qwen05-decision-noul-bal-v3

This model is a fine-tuned version of [gyanbhambhani/oss-jev-qwen05-decision-noul-v2](https://huggingface.co/gyanbhambhani/oss-jev-qwen05-decision-noul-v2) on an unknown dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-06
- train_batch_size: 2
- eval_batch_size: 8
- seed: 43
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- training_steps: 1200

### Training results



### Framework versions

- Transformers 5.20.1
- Pytorch 2.14.1+cu130
- Tokenizers 0.23.3
