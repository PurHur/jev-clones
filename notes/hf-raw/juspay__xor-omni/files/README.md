---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
library_name: transformers
pipeline_tag: image-text-to-text
inference: false
tags:
  - sglang
  - gemma4
  - multimodal
  - mixture-of-experts
  - typed-classification
  - conversational
  - text-generation-inference
  - safetensors
---

# XOR Omni

XOR Omni (`xor-omni`) is a multimodal decision model post-trained from [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). It is designed for typed decision tasks and structured one-pass readout.

## Model details

| Field | Value |
|---|---|
| Base model | `google/gemma-4-26B-A4B-it` |
| Architecture | Gemma 4 multimodal mixture-of-experts |
| Parameters | Approximately 26B total, about 4B active per token |
| Released precision | BF16 |
| Context length | 32,768 tokens |
| Input modalities | Text and image |
| Output | Text |
| Runtime | SGLang |
| License | Apache License 2.0 |
| Packaging | Fully merged weights; no adapter loading is required |

## Intended use

XOR Omni is intended for typed decision workloads, including:

- Binary decisions
- Categorical choices
- Ordinal scoring
- Structured readout from combined text and image context

It is optimized for deterministic, one-pass decision output rather than open-ended assistant conversation.

## Quick start

Download the model:

```bash
hf download juspay/xor-omni --local-dir xor-omni
```

Example SGLang launch:

```bash
python -m sglang.launch_server \
  --model-path ./xor-omni \
  --trust-remote-code \
  --context-length 32768 \
  --host 0.0.0.0 \
  --port 30000
```

Use the included `chat_template.jinja` when serving the model. The exact deployment configuration should be validated for your target hardware and workload.

## Limitations

- This model is specialized for typed decision tasks and is not intended as a general-purpose assistant.
- Multimodal behavior should be validated on your own data before production use.
- Performance may vary with prompt format, context length, hardware, and serving configuration.
- This repository contains merged model weights only; training scripts, adapters, benchmark artifacts, and logs are intentionally not included.

## License

This model is released under the Apache License 2.0, inherited from the base Gemma 4 checkpoint.
