---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
library_name: transformers
pipeline_tag: text-generation
inference: false
tags:
  - sglang
  - mixture-of-experts
  - typed-classification
---

# Xor 1.2

Xor 1.2 (`xor-1.2`) is a post-trained version of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) for typed decision tasks. It is served through a TypeSafe-compatible `/v1/systemone` API.

## Changes from Xor 1.1

- New post-trained weights (LoRA rank 16, merged into BF16 base weights)
- Serving bundle, API, calibration, and media handling are unchanged from Xor 1.1
- Validated on one GPU (TP1) with the same pinned, patched SGLang runtime

## Revisions

| Revision | Release |
|---|---|
| `main` | Latest stable release (currently Xor 1.1) |
| `release/xor-1.2` | Xor 1.2 |
| `v1.1` | Xor 1.1, immutable |
| `xor-v1` | Xor 1.0, immutable |

Pin a revision for reproducible results. `release/xor-1.2` is a branch; pin its commit hash for byte-exact reproducibility.

```bash
hf download juspay/xor --revision release/xor-1.2 --local-dir xor-1.2
```

## Base model

| Field | Value |
|---|---|
| Base checkpoint | `Qwen/Qwen3.6-35B-A3B` |
| Base revision | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Architecture | Mixture-of-experts causal language model |
| Total parameters | 35 billion |
| Activated parameters | Approximately 3 billion per token |
| Released precision | BF16 |
| Base license | Apache License 2.0 |
| Packaging | Fully merged weights; no adapter loading or merging is required |

## Interface

The server accepts a state and a map of typed questions:

- `noul`: binary probability
- `choice`: categorical decision and full probability distribution
- `score`: expected ordinal score and full probability distribution

Requests may include an `images` array with up to eight image data URLs, or one `video` data URL. Typed questions can classify information from the supplied media and text. Send one kind of media per request: the API accepts images and a video together, but the model does not reliably tell them apart.

The complete request body must not exceed 8 MB, which leaves roughly 6 MB of media after base64 encoding. Larger requests are rejected with HTTP 422.

The serving layer performs deterministic single-token candidate readout, forward and reverse option-order evaluation, probability calibration, and schema conversion. The serving layer is part of the released inference configuration and must be used for reproducible results.

## Quick start

Download the release, verify and extract the serving bundle, and start Xor on one GPU:

```bash
hf download juspay/xor --revision release/xor-1.2 --local-dir xor-1.2

(cd xor-1.2/serving && sha256sum -c xor-1.2-serving.tar.gz.sha256)

mkdir -p xor-1.2-runtime
tar -xzf xor-1.2/serving/xor-1.2-serving.tar.gz -C xor-1.2-runtime --strip-components=1

cd xor-1.2-runtime
cp .env.example .env
sed -i "s|^MODEL_DIR=.*|MODEL_DIR=$(cd ../xor-1.2 && pwd)|" .env
./run.sh
```

`run.sh` verifies every model file against `checksums.sha256` before starting. When the smoke test succeeds, the API is available at `http://127.0.0.1:30002/v1/systemone`.

```bash
curl -sS -X POST http://127.0.0.1:30002/v1/systemone \
  -H 'Content-Type: application/json' \
  --data @examples/request.json

curl -sS -X POST http://127.0.0.1:30002/v1/systemone \
  -H 'Content-Type: application/json' \
  --data @examples/image-request.json
```

The setup requires Linux x86-64, the Hugging Face CLI, Docker Engine with a recent Docker Compose v2 (the bundle uses the service-level `gpus` key, which older Compose releases reject), the NVIDIA Container Toolkit, and approximately 120 GB of free disk space.

## Public JEVBench self-run

Xor 1.2 was evaluated locally on the public tiers of [JEVBench](https://github.com/fstandhartinger/jevbench) using harness commit `1bcc55eb6c8cffde2306b3db03ede39b61c6152a`, the existing `typesafe` adapter, and one request at a time.

The run used the released serving bundle unmodified on 1 x NVIDIA H200 NVL (143 GB), tensor parallelism 1, data parallelism 1, with request caching disabled.

| Tier | Attempted | Valid | Correct | Accuracy | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 48 | 48 | 48 | 1.0000 | 0.0679 s | 0.0709 s |
| Original | 72 | 72 | 69 | 0.9583 | 0.0670 s | 0.0701 s |
| Hard public | 111 | 111 | 91 | 0.8198 | 0.0832 s | 0.1845 s |
| **All public** | **231** | **231** | **208** | **0.9004** | 0.0691 s | 0.1616 s |

Across all 231 public decisions: macro accuracy 0.9070, Brier mean 0.1790, ECE 0.0733. Operational success, coverage, schema validity, and strict schema validity were 1.0000.

These are self-run public-tier results, not an official JEVBench rank. Latency is hardware-specific and was measured locally without network overhead.

## Validated runtime

| Setting | Value |
|---|---|
| SGLang image | `prakhar1611/xor-sglang@sha256:94c48d2a6cc98dc456cf93f723707ea7dd81dddfe1061e823b348d68bbe8158f` |
| Upstream SGLang base | `lmsysorg/sglang@sha256:6bcaa47db52f78ce0d67863b8b2431221b79bc23204a80cad757fa819d00e921` |
| Tensor parallelism | 1 |
| Data parallelism | 1 |
| Validated GPU | 1 x NVIDIA H200 NVL, 143 GB |
| Maximum prefill tokens | 250,000 |
| Static memory fraction | 0.85 |

The runtime image adds a small fix to SGLang's candidate-logprob result handling that prevents a scheduler crash when scoring and plain generation requests share a batch. `Dockerfile.sglang-logprob-fix` in the serving bundle reproduces it from the upstream image.

## Operational notes

The model files occupy approximately 66 GB. Each data-parallel worker loads a complete model replica. Alternative hardware and parallelism settings must be validated independently before publishing performance results.

Artifact hashes, the base revision, adapter and merge metadata, the runtime image, and known provenance gaps are recorded in `RELEASE_PROVENANCE.json` and `checksums.sha256`.

The service does not require an API key when bound to localhost. Remote deployments must add authentication, TLS, rate limits, and request-size limits at the ingress layer.
