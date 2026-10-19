# XOR Lite 1.1 serving bundle

This bundle starts XOR Lite 1.1 (`xor-lite`) behind a TypeSafe-compatible `/v1/systemone` API, bound to `127.0.0.1` by default

XOR Lite 1.1 is a LoRA adapter and a pointer readout head on a MiniCPM5-2B-derived base. The server applies the adapter at runtime with PEFT. Do not merge the adapter into the base weights: the merged bf16 weights cannot represent part of the adapter delta, and a merged model gives different answers

## Requirements

- Linux on x86-64
- Docker Engine with Docker Compose v2
- NVIDIA Container Toolkit and an NVIDIA driver that supports CUDA 13.0
- One NVIDIA GPU
- The released `xor-lite` model directory, revision `v1.1`

## Start

Place the released model directory at `./model`, or copy `.env.example` to `.env` and set `MODEL_DIR` to its absolute path

```bash
cp .env.example .env
./run.sh
```

`run.sh` verifies every model file against `checksums.sha256`, builds the API image, starts it, and runs `smoke_test.py`. The API is then available at `http://127.0.0.1:30002/v1/systemone`

## Test

```bash
curl -sS -X POST http://127.0.0.1:30002/v1/systemone \
  -H 'Content-Type: application/json' \
  --data @examples/request.json
```

## Operations

```bash
./status.sh
docker compose logs -f xor-lite-api
./stop.sh
```

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `MODEL_DIR` | `./model` | Host path containing the model release |
| `API_PORT` | `30002` | Loopback port for the SystemOne-compatible API |
| `CUDA_VISIBLE_DEVICES` | `0` | The one GPU the server uses |

The release settings are fixed in `app.py` and `docker-compose.yml`: bf16 weights, adapter applied at runtime, temperature `1.0`, shared-prefix cache off, one worker. A `choice` question accepts up to 255 options; a `score` question accepts 2 to 10 levels

## Model directory layout

| Path | Contents |
|---|---|
| `config.json`, `model.safetensors`, tokenizer files | The base model only. Loading this directory with `AutoModelForCausalLM` does not give XOR Lite 1.1 |
| `decider/` | The release checkpoint: `lora/` adapter, `head.safetensors` readout, tokenizer, and `strands_decider_config.json` |

## Network exposure

The default port binding is loopback-only. To expose the service remotely, place an authenticated TLS reverse proxy with request limits in front of it
