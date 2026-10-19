---
license: apache-2.0
base_model:
  - google/gemma-4-26B-A4B-it
library_name: transformers
pipeline_tag: text-generation
inference: false
tags:
  - decision-model
  - sglang
  - mixture-of-experts
  - typed-classification
  - gemma4
---

# Xor 26B-A4B v1.2

Post-trained Gemma 4 26B-A4B for typed decision tasks, distributed as merged BF16 weights. No adapter merge is required for inference. The bundled wrapper exposes `/v1/systemone` for text, image, and video input and `noul`, `choice`, and `score` questions.

Base: `google/gemma-4-26B-A4B-it`, revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. Architecture, tokenizer, and processor are unchanged from v1.1.

## What changed from v1.1

v1.2 is a weight blend of two post-trained checkpoints of the same base:

- Non-expert tensors: 0.60 × the v1.1 checkpoint (`xn-lora-xl-5050`) + 0.40 × a second post-trained checkpoint (`ndnew`, a LoRA merged onto the base and trained on a separate decision-task pool), computed in float and cast back to BF16.
- Routed-expert tensors (`experts.down_proj` and `experts.gate_up_proj` in all 30 layers, 60 tensors): taken unchanged from the v1.1 checkpoint.

The 0.60/0.40 split was chosen after comparing several blends on public Decision Index results, so any advantage over simpler choices is partly tuned on that public suite.

The serving bundle is the v1.1 bundle with one configuration change: the temperature map is now `choice=1.95`, `noul=1.0`, `score=1.0` (v1.1: `choice=2.0`, `noul=1.7`, `score=1.0`). The wrapper source and SGLang image are byte-identical to v1.1. The `choice` temperature was carried over from an earlier checkpoint's fit and was not re-fitted on held-out data for this blend.

## Revisions

`main` contains v1.2; the `v1.2` tag pins this release. The `v1` and `v1.1` tags and historical revisions remain available. Pin the tag for reproducible inference. The NVFP4 companion repository is a separate release and is not updated by this BF16 release.

## Public evaluation

Decision Index 0.3, self-run on 2026-10-16 with this repository's serving bundle, using upstream commit `f5a6e4fd85fed05e77878dfc4966d0568bfb45ec`:

| Metric | Result |
|---|---:|
| Public Decision Index | 59.46 |
| Requests attempted | 140,620 |
| Requests scored after upstream exclusions | 140,178 |
| Errors | 0 |
| Coverage | 100% |
| Mean HTTP request latency | 559.5 ms |
| Median HTTP request latency | 176.6 ms |
| p95 HTTP request latency | 2,268.4 ms |

Hardware: two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (96 GB each), one isolated TP1 replica per GPU started with this bundle's `run.sh`, six simultaneous requests per replica. Twelve unchanged upstream sequential runners processed disjoint shards, followed by unchanged upstream scoring. Latencies above include wrapper and inference under load; they are not single-GPU sequential eligibility measurements. The result is the self-run public index, not the private Full score or a maintainer-verified leaderboard result. Aggregate reports are in `evaluation/`.

## Reproducible serving

Requires a Linux NVIDIA GPU host, Docker Compose with GPU support, NVIDIA Container Toolkit, and approximately 96 GB GPU memory for the tested settings. CPU-only and smaller-GPU configurations are not validated. The public model download requires the Hugging Face CLI.

```bash
hf download juspay/Xor-26B-A4B --revision v1.2 --local-dir ./xor-v1.2
export MODEL_DIR="$(pwd)/xor-v1.2"
mkdir -p ./xor-v1.2-runtime
tar -xzf "$MODEL_DIR/serving/xor-v1.2-serving.tar.gz" -C ./xor-v1.2-runtime
cd ./xor-v1.2-runtime
export GPU_ID=0
export API_PORT=8000
bash ./run.sh
```

The API binds to localhost, has no API-key authentication, and should not be exposed directly to the Internet. Use an authenticated gateway when external access is needed.

```bash
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"xor-26b-a4b-v1.2","state":"The customer was charged twice.","questions":{"billing":{"type":"noul","instructions":"Is this a billing issue?"}}}'
```

Stop this Compose project with `bash ./stop.sh`. It does not delete model weights.

## Inference settings and limits

Pinned SGLang image: `prakhar1611/xor-sglang@sha256:94c48d2a6cc98dc456cf93f723707ea7dd81dddfe1061e823b348d68bbe8158f`, built from `lmsysorg/sglang@sha256:6bcaa47db52f78ce0d67863b8b2431221b79bc23204a80cad757fa819d00e921` with the tensor/list candidate-logprob handling fix. The exact frozen wrapper is bundled; two-order scoring is enabled, with temperatures `choice=1.95`, `noul=1.0`, and `score=1.0`.

The wrapper scores candidate label-token probabilities rather than generating explanatory prose. `choice` supports 2–255 options; `score` supports 2–10 ordered levels. Context limit is 131,072 tokens. Confidence is the largest candidate probability. The wrapper does not support PDF files, audio, arbitrary text extraction, or general chat generation.

Requests may include `images`, an array of 1–8 base64 image data URLs, or `video`, one base64 video data URL. Send only one kind of media per request. Media is processed with Gemma's own chat template and the matching processor embedded in `processor_config.json`; all questions and reverse-order scoring rows receive the same supplied media. Video is processed as sampled frames, without its audio track. Request bodies larger than 8 MB are rejected. Only data URLs are accepted; the wrapper does not fetch client-supplied remote URLs. Expanded media tokens consume context capacity and are subject to backend context checks. Image and video checks for this release are functional smoke tests, not a multimodal benchmark.

Public benchmark evaluation is not a guarantee of downstream quality or calibrated probabilities.

## Training data and known gaps

No claim of benchmark-clean training is made. Known issues:

- The v1.1 checkpoint's training mixes contain test rows from the public suite's POP909 benchmark. v1.2 inherits this through the v1.1 tensors.
- The second checkpoint's training pool was screened against the Decision Index 0.3 public suite. Whole-field screening found no matches; a stricter per-field screen found 26 exact matches and about 3,500 near-matching contexts, almost all shared tool-retrieval contexts.
- The second checkpoint's training rows do not carry per-row source or licence provenance.

## License

Apache License 2.0. See `LICENSE` and `THIRD_PARTY_NOTICES.md` for notices. Benchmark inputs are not redistributed in the model repository; their upstream dataset terms continue to apply.
