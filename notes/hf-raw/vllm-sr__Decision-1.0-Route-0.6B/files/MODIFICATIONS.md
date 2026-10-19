# Modifications

This repository is a fine-tune of `vllm-sr/Decision-1.0-Kai-0.6B`. The Choice encoder, the shared encoder (`native/encoder/model.safetensors`) and the decision heads were changed by fine-tuning on router signal data, and `config.json` names the model `Decision-1.0-Route-0.6B`. The Score encoder and the tokenizer are Kai's, unchanged. The licence files under this folder are Kai's, passed on unchanged.
