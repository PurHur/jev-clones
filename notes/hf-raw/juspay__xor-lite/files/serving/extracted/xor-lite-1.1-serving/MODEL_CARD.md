---
license: apache-2.0
base_model: openbmb/MiniCPM5-2B
library_name: transformers
inference: false
tags:
  - typed-classification
  - lora
  - peft
---

# XOR Lite

XOR Lite (`xor-lite`) is a post-trained version of [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) for typed decision tasks. It is served through a TypeSafe-compatible `/v1/systemone` API

Version 1.1 replaces the token readout of 1.0 with a pointer readout head and ships the LoRA adapter separately from the base weights. The adapter must be applied at runtime; the included serving bundle does this

## Release

| Field | Value |
|---|---|
| Version | 1.1, revision tag `v1.1` |
| Base checkpoint | `openbmb/MiniCPM5-2B` |
| Base revision | `12a3808a956f869c767195e9266b59c4d21d92e2` |
| Architecture | Dense `LlamaForCausalLM` torso, LoRA adapter, pointer readout head |
| Parameters | Approximately 2.52 billion in the base, 50.2 million in the adapter, 1.1 million in the head |
| Released precision | BF16 base; FP32 adapter and head |
| License | Apache License 2.0 |
| Packaging | Base weights plus a separate adapter and head. The adapter is applied at runtime and is not merged |

Pin the published revision for reproducible use

```bash
hf download juspay/xor-lite --revision v1.1 --local-dir xor-lite
```

## Lineage

```text
openbmb/MiniCPM5-2B @ 12a3808
 ├─ XOR Lite 1.0 weights                ×0.3 ┐
 └─ a second MiniCPM5-2B fine-tune      ×0.7 ┘─ linear weight average (bf16) → 1.1 base
                                                 + LoRA r32, alpha 64 (applied at runtime)
                                                 + pointer readout head
```

The 1.1 base is a linear average of two fine-tunes of `openbmb/MiniCPM5-2B`: the released XOR Lite 1.0 weights (0.3) and a second fine-tune from the same campaign (0.7). The adapter and head were then trained on that base

## Repository layout

| Path | Contents |
|---|---|
| `config.json`, `model.safetensors`, tokenizer files | The 1.1 base model only |
| `decider/` | The 1.1 release checkpoint: `lora/` adapter, `head.safetensors`, tokenizer, and `strands_decider_config.json` |
| `serving/` | The serving bundle |
| `eval/decision-index-0.3.json` | The Decision Index result reported below |

Loading the repository root with `AutoModelForCausalLM` gives the base model, not XOR Lite 1.1. Do not merge the adapter into the base: the bf16 base weights cannot represent part of the FP32 adapter delta, and a merged model gives different answers

## Interface

The server accepts a state and a map of typed questions

- `noul`: binary probability
- `choice`: categorical decision, full probability distribution, and confidence
- `score`: expected ordinal score, full probability distribution, legend, and confidence

The readout head scores each option from the model's hidden state at the end of that option. `choice` accepts 2 to 255 options and `score` accepts 2 to 10 levels; larger requests are rejected with HTTP 422. The state and questions are text or JSON; image and video inputs are not supported

## Quick start

```bash
hf download juspay/xor-lite --revision v1.1 --local-dir xor-lite

(cd xor-lite/serving && sha256sum -c xor-lite-1.1-serving.tar.gz.sha256)

mkdir -p xor-lite-runtime
tar -xzf xor-lite/serving/xor-lite-1.1-serving.tar.gz -C xor-lite-runtime --strip-components=1

cd xor-lite-runtime
cp .env.example .env
sed -i "s|^MODEL_DIR=.*|MODEL_DIR=$(cd ../xor-lite && pwd)|" .env
./run.sh
```

`run.sh` verifies every model file, builds the API image, and runs a smoke test. When it succeeds, the API is available at `http://127.0.0.1:30002/v1/systemone`

## Decision Index 0.3

| Score | Value |
|---|---:|
| **Decision Index 0.3** | **37.51** |
| Raw index | 53.66 |
| Breadth skill | 34.74 |

| Area | Skill | Raw |
|---|---:|---:|
| Knowledge & Reasoning | 0.1828 | 0.3879 |
| Language Understanding | 0.4156 | 0.5936 |
| Retrieval & Classification | 0.5006 | 0.6439 |
| Tools & Automation | 0.5664 | 0.6041 |
| Arts & Human Taste | 0.1670 | 0.4353 |

This is a self-run result on the public Decision Index 0.3 suite, with complete coverage. It is not an official board entry. The run scored the released weights in-process on one GPU. Its `noul` answers were then recomputed with the released serving bundle, because the run's own `noul` readout reported the larger of the two probabilities instead of the probability of `true`. That correction changed one benchmark and raised the index from 37.05 to 37.51. A replay of 1,000 sampled cases through the released bundle matched the run's `choice` answers in 2,005 of 2,007 questions, with the other two within 0.015. The full per-benchmark result is in `eval/decision-index-0.3.json`

The training data includes ChessBench-style rows, and ChessBench is one of the index benchmarks. Results on that benchmark should not be read as held-out performance

## Validated runtime

| Setting | Value |
|---|---|
| Runtime | PyTorch 2.13.0 (CUDA 13.0), Transformers 5.14.1, PEFT at commit `194cb8f` |
| Validated GPU | 1 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB |
| NVIDIA driver | `595.91.07` |
| Workers | 1 |
| Shared-prefix cache | Off |
| Probability temperature | `1.0` for `noul`, `choice`, and `score` |

The loaded service used about 21.9 GB of GPU memory after serving the replay above. Latency has not been measured for this release

The service binds to localhost by default and does not require an API key. Remote deployments must add authentication, TLS, rate limits, and request-size limits at the ingress layer

## Changes from 1.0

- The base is now the linear weight average described above; 1.0 contributes 0.3 of it
- Answers come from a pointer readout head instead of single-token log probabilities, and the temperature is `1.0` instead of `100.0`
- The adapter and head ship as separate files and are applied at runtime. Deployments that load the repository root directly in SGLang or Transformers get the base model only
- Responses include `confidence` for `choice` and `score`, and `legend` for `score`
- The serving bundle runs the model in PyTorch rather than SGLang

## Known provenance gaps

- The second fine-tune in the base average was identified from its weights: recomputing the 0.7/0.3 average reproduces every base tensor bit for bit. Its training configuration is not part of this release
- The adapter was saved with a development build of PEFT (`0.19.2.dev0`, commit `194cb8f`); the serving bundle installs that exact commit
