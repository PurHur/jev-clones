# Xor 1.2 serving bundle

This bundle starts Xor 1.2 (`xor-1.2`) behind a TypeSafe-compatible `/v1/systemone` API. SGLang runs on an internal Docker network; only the compatibility API is bound to the host, on `127.0.0.1` by default.

## Requirements

- Linux on x86-64
- Docker Engine with Docker Compose v2
- NVIDIA Container Toolkit
- One NVIDIA H200 GPU with 143 GB of memory
- Approximately 120 GB of free disk space
- The released `xor-1.2` model directory

The release configuration was validated on one NVIDIA H200 GPU with 143 GB of memory. It loads one model replica with tensor parallelism 1 and data parallelism 1.

## Model files

Place the released model directory at `./model`, or copy `.env.example` to `.env` and set `MODEL_DIR` to an absolute path.

The directory must contain:

```text
model-00001-of-00016.safetensors
...
model-00016-of-00016.safetensors
model.safetensors.index.json
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
chat_template.jinja
configuration.json
merges.txt
vocab.json
preprocessor_config.json
video_preprocessor_config.json
```

The published hashes are in `checksums.sha256`; `run.sh` verifies every required model file before starting the containers. `RELEASE_PROVENANCE.json` records the pinned base revision, adapter hashes, merge hash, runtime image digest, and known provenance gaps.

## Start

```bash
cp .env.example .env
./run.sh
```

The first start pulls approximately 15 GB of container images and loads approximately 66 GB of model weights.

When the smoke test succeeds, the API is available at:

```text
http://127.0.0.1:30002/v1/systemone
```

## Test

```bash
curl -sS -X POST http://127.0.0.1:30002/v1/systemone \
  -H 'Content-Type: application/json' \
  --data @examples/request.json
```

Test image input with the included data-URL request:

```bash
curl -sS -X POST http://127.0.0.1:30002/v1/systemone \
  -H 'Content-Type: application/json' \
  --data @examples/image-request.json
```

Run the four labeled red/blue PNG and MP4 probes (a media-path check, not a
general vision benchmark):

```bash
python3 bench_multimodal.py --endpoint http://127.0.0.1:30002 \
  --out media-check
```

List the served model:

```bash
curl -sS http://127.0.0.1:30002/v1/models
```

## Operations

```bash
./status.sh
docker compose logs -f jev-api
docker compose logs -f sglang
./stop.sh
```

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `MODEL_DIR` | `./model` | Host path containing the model release |
| `API_PORT` | `30002` | Loopback port for the JEV-compatible API |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU assigned to SGLang |
| `TP_SIZE` | `1` | SGLang tensor-parallel size |
| `DP_SIZE` | `1` | SGLang data-parallel size |
| `MAX_PREFILL_TOKENS` | `250000` | Maximum batched prefill budget |
| `MEM_FRACTION_STATIC` | `0.85` | Fraction of GPU memory reserved by SGLang |
| `OPENJEV_TEMP_JSON` | `{"noul":1.4,"choice":1.1,"score":1.0}` | Per-type probability temperature; restart the API after changing it |

Reducing the prefill budget may allow the model to run on smaller GPUs, but any alternative configuration should be validated independently before publishing performance results.

The Compose file pins a derivative of SGLang nightly `408d2334`. The only runtime change makes candidate-logprob serialization accept the mixed tensor/list representation produced by the scheduler. `Dockerfile.sglang-logprob-fix` records the complete patch, and `RELEASE_PROVENANCE.json` pins both the upstream base and published derivative digests.

## Network exposure

The default port binding is loopback-only. To expose the service remotely, place an authenticated TLS reverse proxy in front of it. Do not change the binding to `0.0.0.0` without adding authentication and request limits.

The SGLang service has no published host port. Request caching and request-signature logging are disabled in `docker-compose.yml`.

## API behaviour

The compatibility server translates `noul`, `choice`, and `score` questions into constrained single-token classification prompts. It supports up to 255 choices using the bundled `markers_255.json` codebook, evaluates each option ordering in forward and reverse order, reads candidate-label log probabilities from SGLang, calibrates and combines them, and returns native probability distributions. If `unknown` adds a choice, it counts toward the 255-option limit. At startup, the wrapper requires every marker to resolve to the exact token ID recorded in the codebook.

Requests may include an `images` array of up to eight image data URLs and one `video` data URL. The media bytes are included directly in the JSON request, which has an 8 MB limit. Multimodal handling is enabled in the included Compose configuration.

The `usage` object reports logical API tokens and decisions. The backend performs two option-order evaluations per question.
