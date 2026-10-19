---
base_model: XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
license: other
license_name: nace-ai-open-rail-m
license_link: LICENSE
tags:
- decision-model
- system-one
- kev
- drex
- typed-decisions
- decision-index
- classification
- qwen3_5
- calibrated-probabilities
- decisionbench
- zero-shot
- recommendation
---

# Drex v1.5

A decision model from [Nace.AI](https://www.nace.ai/): it reads a state (text or JSON) and typed questions (`choice`, `noul`, ordinal `score`) and returns a probability for every option in one forward pass per question. Nothing is generated. It serves the TypeSafe-compatible `/v1/systemone` API through the bundled Kev code, the same request and response format as the hosted [Drex API](https://drex.nace.ai/docs) and the open [Drex DLM](https://huggingface.co/nace-ai/drex-dlm).

**Drex v1.5 is the #1 model under 10B parameters on the public Decision Index 0.3.1 (58.08).**

[Drex repository](https://github.com/nace-ai/drex-decision-models) · [Drex DLM](https://huggingface.co/nace-ai/drex-dlm) · [Hosted API docs](https://drex.nace.ai/docs) · [Agent skill](https://github.com/nace-ai/drex-agent-skill)

## Model

| | |
|---|---|
| Backbone | `Qwen3_5ForCausalLM`, 32 layers, hybrid attention (3 linear-attention layers for every full-attention layer) |
| Parameters | about 9B (8.95B), bf16 weights, about 18 GB on disk |
| Pointer head | `head.pt`, scores each option from the backbone's hidden states |
| Context | 16,384 tokens by default, up to 131,072; long-document accuracy is reported up to 128k tokens (see [Performance](#long-context-benchmarks)) |
| Base model | [XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) |

## Performance

### Decision Index 0.3.1 (public only)

| Model | Public index | Raw | Knowledge & Reasoning | Language | Retrieval | Tools | Arts |
|---|---|---|---|---|---|---|---|
| **Drex v1.5** (9B) | **58.08** | 68.08 | 44.6 | 60.4 | 61.7 | 75.0 | 48.8 |
| Jev 1.13.0 | 57.96 | 68.55 | 53.9 | 59.2 | 55.4 | 75.1 | 39.1 |
| Bespoke Nimble 9B v3 | 57.19 | 68.38 | 41.0 | 60.7 | 55.5 | 84.2 | 44.0 |
| Cloudflare clef-flash | 56.15 | 66.53 | 53.0 | 47.2 | 52.2 | 81.9 | 48.4 |
| ezjev 4B s2 | 50.82 | 63.16 | 31.0 | 60.5 | 56.3 | 69.9 | 31.1 |

Public index only: the 37 public benchmarks in five areas, without the private tests. Area columns are chance-corrected skill × 100. The table shows Drex v1.5, Jev and the top three other models under 10B parameters on the [leaderboard](https://huggingface.co/spaces/multimodalart/jev-decision-index). The Drex v1.5 numbers are from our own run of the official kit on the full public suite (the text suite is the same in 0.3 and 0.3.1); the other rows are the leaderboard's. Drex v1.5 is the highest-scoring model under 10B parameters; Jev's parameter count is not published. Drex v1.5, Jev and Nimble are within the board's 0.9-point tie band. Drex v1.5 is ahead of Jev on 20 of the 37 benchmarks.

### JevBench (231 public items)

| Model | Accuracy | Easy | Standard | Hard |
|---|---|---|---|---|
| Jev 1.13.0 | 87.0% | 100.0% | 98.6% | 73.9% |
| **Drex v1.5** | **86.2%** | 100.0% | 95.8% | **73.9%** |
| decider-4b v2.1 | 83.1% | 100.0% | 98.6% | 65.8% |

### Long-context benchmarks

| Context length | Accuracy | Truncated to 8k | Median latency |
|---|---|---|---|
| 8k – 32k tokens | 89.5% | 76.5% | 0.65 s |
| 32k – 128k tokens | 93.4% | 78% | 2.0 s |

Long documents with typed questions; every request was answered.

### Game arena against Jev

Eight OpenSpiel games, 32 games each (16 openings × both colours): **122 wins, 47 draws, 87 losses (56.8%)**.

![Drex v1.5 vs Jev game arena](game_arena.png)

## Inference

### Requirements

- Python with the packages in `requirements.txt` (`torch`, `transformers`).
- A CUDA GPU for `inference.py` and `serve.py`. The bf16 weights are about 18 GB, and memory use grows with document length.
- Apple silicon (Metal) and CPU: use llama.cpp or Ollama below. The Q8_0 GGUF is about 9.5 GB.

### Score a request from the command line

```bash
hf download nace-ai/drex-v1.5 --local-dir drex-v1.5
cd drex-v1.5
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python inference.py                                    # built-in example
python inference.py --request examples/request.json    # your own System One request
```

`inference.py` loads the checkpoint in its own folder (`--model` selects another) and prints the answers as JSON.

### Serve over HTTP

```bash
python serve.py --port 8000
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' -d @examples/request.json
```

The endpoint is `POST /v1/systemone` and needs no `Authorization` header; `GET /health` returns `{"status": "ok", "model": "drex-v1.5"}` once the weights are loaded. A request can also be `{"requests": [...]}` to score several decisions in one call. The server listens on `127.0.0.1`; keep it there unless you put authentication in front of it.

### Run with llama.cpp

```bash
git clone -b drex-v1.5 https://github.com/nace-ai/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j --target llama-server llama-quantize

python convert_hf_to_gguf.py ../drex-v1.5 --no-mtp --outtype bf16 --outfile drex-v1.5.gguf
./build/bin/llama-quantize drex-v1.5.gguf drex-v1.5-Q8_0.gguf Q8_0     # optional, about 9.5 GB

./build/bin/llama-server -m drex-v1.5.gguf --embedding --pooling none \
  -np 2 -c 32768 -b 16384 -ub 2048 --port 8000
```

The server answers `POST /v1/systemone` with the same request and response as above. Use `-np 2` so the state is encoded once and shared by every question; `-c` is split across slots, so each gets half. On a GPU (CUDA or Metal), add `-ngl 99`; leave it off to run on the CPU.

This command accepts requests up to 16,384 tokens: `-c` is shared by the two slots, so each gets 16,384. For up to 131,072 tokens, set `SYSTEMONE_CONTEXT=131072` and use `-c 262144`.

Tested on an AWS g5.2xlarge (NVIDIA A10G 24 GB, 8 vCPU AMD EPYC 7R32, 32 GiB RAM, Ubuntu 24.04, CUDA 13.2), against the Python server, every answer is the same in bf16 and Q8_0.

On an Apple M5 Pro (18-core CPU, 20-core GPU, 48 GB unified memory, macOS 26.5.2), the Q8_0 GGUF gives the same answers on Metal and on the CPU.

### Run with Ollama

Use the Nace fork of Ollama with a `llama-server` built from the `drex-v1.5` branch above. Convert the GGUF first, as in the llama.cpp steps.

```bash
git clone --branch drex-v1.5 --single-branch https://github.com/nace-ai/ollama.git && cd ollama
export OLLAMA_LLAMA_CPP_SOURCE=/path/to/llama.cpp     # your drex-v1.5 llama.cpp checkout
cmake -S llama/server --preset darwin
cmake --build build/llama-server-darwin --target llama-server --parallel 8
GOTOOLCHAIN=auto go build -trimpath -o ollama .
```

For CPU use the `cpu` preset and `build/llama-server-cpu`; for NVIDIA on Linux use `llama_cuda_v13_linux` (or `llama_cuda_v12_linux`) and `build/llama-server-cuda_v13` (or `-cuda_v12`) in place of `darwin` and `build/llama-server-darwin` here and below.

Write a `Modelfile` next to the GGUF:

```
FROM ./drex-v1.5.gguf
CAPABILITY decision
PARAMETER num_ctx 16384
```

Start the daemon with the fork's binary, then import and call the model:

```bash
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_LLAMA_SERVER=$PWD/build/llama-server-darwin/bin/llama-server ./ollama serve
OLLAMA_HOST=127.0.0.1:11434 ./ollama create drex-v1.5 -f Modelfile
curl http://127.0.0.1:11434/v1/systemone -H 'Content-Type: application/json' -d @examples/request.json
```

Set `"model": "drex-v1.5"` in the request. Tested through Ollama on the same g5.2xlarge, every answer is the same as the Python server's. Ollama starts `llama-server` with two slots, so the state is encoded once per request. The default is 16,384 tokens. For up to 131,072, set `PARAMETER num_ctx 131072`, re-import, and start the daemon with `SYSTEMONE_CONTEXT=131072`.

### Request format

```json
{"state": {"ticket": "I was charged twice for order A-104."},
 "questions": {"team": {"type": "choice", "instructions": "Which team handles this?",
                        "criteria": {"billing": "Charges and refunds", "shipping": "Delivery", "other": null}},
               "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"}}}
```

| Field | What to put there |
|---|---|
| `state` | The document: a string, object, or list. |
| `questions` | A dictionary of named questions. The names come back as the keys of `answers`. |

| Type | You supply | You get back |
|---|---|---|
| `choice` | Named options with short descriptions. Key order is option order. | The chosen option and a probability per option |
| `noul` | A yes/no question, with optional `criteria.true` / `criteria.false`. | The probability of "yes" (0 to 1) |
| `score` | An ordered scale, lowest first. | A probability-weighted score with per-level probabilities |

### How scoring works

- Each question is scored in one forward pass over the state plus that question. The model reads the options and returns a probability for each one; no tokens are sampled, so the `temperature`, `top_p` and `top_k` values in `generation_config.json` do not apply.
- Requests with several questions run one pass per question.

### Long documents

Requests up to 16,384 tokens are accepted by default. Set `KEV_CONTEXT` (maximum 131,072) to accept longer ones:

```bash
KEV_CONTEXT=131072 python serve.py --port 8000
```

`KEV_SERVE_MAX_STATE`, `KEV_SERVE_MAX_BRANCH` and `KEV_SERVE_MAX_PACKED` set the individual caps, and `inference.py` reads the same variables. Latency and accuracy at 8k–128k tokens are in [Long-context benchmarks](#long-context-benchmarks).

## Integrations

Drex v1.5 speaks one wire format, System One (`POST /v1/systemone`), so anything written for that format works against it.

### Where to run it

| Option | What it is | Status |
|---|---|---|
| Python (Kev) | `inference.py` and `serve.py` from this repo; PyTorch on CUDA | Supported |
| Hosted Drex API | `https://drex.nace.ai`, bearer-key auth (`nace_sk_...`), see the [API docs](https://drex.nace.ai/docs) | Hosted model is `drex-latest`; check `GET /v1/models` for the current catalog |
| llama.cpp | [`drex-v1.5` branch](https://github.com/nace-ai/llama.cpp/tree/drex-v1.5): `llama-server` with `/v1/systemone`, GGUF (bf16 or Q8_0); CUDA, Metal and CPU | Supported |
| Ollama | [`drex-v1.5` branch](https://github.com/nace-ai/ollama/tree/drex-v1.5) of the Nace fork, same GGUF; CUDA, Metal and CPU | Supported |

### SDKs and clients

- **TypeSafe SDK** (`@typesafe-ai/sdk`) works against the hosted API without a fork: set `TYPESAFE_BASE_URL=https://drex.nace.ai`, `TYPESAFE_API_KEY=nace_sk_...` and `TYPESAFE_DEFAULT_MODEL=drex-latest`. See [Migrate from TypeSafe](https://drex.nace.ai/docs/guides/migrate-from-typesafe).
- **Kev-format servers and SDKs.** Drex follows the System One request format also used by [Kev](https://github.com/jaredpalmer/kev) and Jev, so clients written for those servers work unchanged against a self-hosted v1.5 server.
- **Plain HTTP.** Any HTTP client can call `POST /v1/systemone` with the JSON above.

### Coding agents

The [Drex agent skill](https://github.com/nace-ai/drex-agent-skill) makes Drex the System One decision provider for coding agents. It installs with each agent's native skill mechanism:

| Agent | Install path |
|---|---|
| Claude Code | Native plugin: `claude plugin install drex@nace-ai` |
| OpenCode, Cursor, Windsurf, Cline | Link `skills/drex` into the agent's skills directory |
| Codex | `$skill-installer` from the GitHub directory |
| Hermes Agent | `hermes skills install nace-ai/drex-agent-skill/skills/drex` |
| Gemini CLI | `gemini skills install ... --path skills/drex` |
| GitHub Copilot | `copilot skill add` or `.github/skills/drex/` |
| OpenClaw, Pi, DeepSeek Harness | Native installer or link, per the skill README |
| Any [Agent Skills](https://agentskills.io/specification) harness | Link `skills/drex` into its skills directory |

The skill uses the hosted API with `DREX_API_KEY`. To use your own server instead, add a line to the agent's instructions (for example `AGENTS.md`): *For System One decisions, use my self-hosted Drex at http://127.0.0.1:8000. Do not fall back to hosted Drex.* Self-hosted requests carry no `Authorization` header, and a hosted key must never be sent to your own server. See the [skill README](https://github.com/nace-ai/drex-agent-skill#self-hosted-drex) for per-agent steps.

## Files

| | |
|---|---|
| `model-*.safetensors`, `config.json` | backbone (bf16) |
| `head.pt` | pointer head |
| `code/kev/` | Kev package (Apache-2.0) that loads and scores the model |
| `inference.py` | scores one System One request |
| `serve.py` | `POST /v1/systemone` server |
| `requirements.txt` | Python dependencies |
| `examples/request.json` | sample request |

## License

Released under the Nace.AI Open RAIL-M license; see [LICENSE](https://huggingface.co/nace-ai/drex-v1.5-Q8_0/blob/main/LICENSE). The Kev code in `code/` is Apache-2.0.
