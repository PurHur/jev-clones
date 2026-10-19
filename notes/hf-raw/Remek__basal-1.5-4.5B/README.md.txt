---
license: apache-2.0
language:
- pl
- en
base_model: speakleash/Bielik-4.5B-v3.0-Instruct
library_name: transformers
pipeline_tag: text-generation
tags:
- typed-decisions
- decision-making
- classification
- calibration
- polish
---

# basal-1.5-4.5B

[![GitHub](https://img.shields.io/badge/GitHub-rkinas%2Fbasal%20%E2%80%94%20inference%20engine-black?logo=github)](https://github.com/rkinas/basal)
[![Website](https://img.shields.io/badge/web-basal.si5.pl-0a7d5a)](https://basal.si5.pl/)
[![Collection](https://img.shields.io/badge/%F0%9F%A4%97-basal--1.5%20collection-yellow)](https://huggingface.co/collections/Remek/basal-15-6ac0de0e9199d121c6adac20)

**basal-1.5 (4.5B)** is the main model of the family: the best balance of accuracy and speed. Part of the basal-1.5 family of typed-decision models for Polish and English.

**What it is.** Inspired by *System 1* (fast, intuitive) decision models such as Jev: instead of writing an answer,
the model reads a *state* (a message, a document, retrieved passages, an agent trace, a web page as JSON) and answers a
typed question about it (`choice`, yes/no `noul`, `score`, and in the basal engine `multi` and `act`) by returning a
**calibrated probability for each allowed answer**, in one forward pass, without generating text. The answer never
falls outside the options you give, and the probability says how sure the model is.

**What it is for: a dynamic classifier.** The classes are described *in the request*, in plain language, so one model
serves many tasks without retraining: routing with changing categories, rule and policy checks, questions about long
documents, RAG decisions (relevance, sufficiency, conflicts, claim support, "cannot be determined"), urgency and risk
scores, judging answers, agent and tool decisions, and triage with a confidence threshold.

| model | params | use |
|---|---|---|
| [basal-1.5](https://huggingface.co/Remek/basal-1.5-4.5B) | 4.5B | main model |
| [basal-1.5-max](https://huggingface.co/Remek/basal-1.5-max) | 11B | most accurate in the family |
| [basal-1.5-mini](https://huggingface.co/Remek/basal-1.5-mini) | 1.5B | fastest, distilled from max |

Each model also comes as `-MLX-8bit` and `-MLX-fp4` (Apple Silicon) and `-GGUF` (Ollama, llama.cpp).
NVIDIA ModelOpt checkpoints for vLLM: `-FP8` and `-NVFP4`.

## Results

**Werdykt** (our hidden Polish–English benchmark of typed decisions: 10 categories × 2 languages, one protocol for every
model, open models on one H100; macro accuracy over the 20 cells): **0.721** (Polish
0.724, English 0.718), median latency 33.8 ms per decision
with SGLang, $0.048 per 1,000 decisions. Leaderboard and public samples:
https://basal.si5.pl/.

| benchmark | basal-1.5 (4.5B) | basal-1.0-4.5B |
|---|---|---|
| Werdykt v1, macro | 0.721 | 0.660 |
| PL sealed test (3,000 new decisions, scored once) | 0.931 | 0.874 |
| PL decisions (held-out test of basal-1.0, 7,081 items) | 0.923 | 0.886 |
| PL general (knowledge, exams, reading; 3,079 items) | 0.741 | 0.737 |
| EN external panel (10 public English decision tasks, 3,842 items) | 0.688 | 0.671 |
| Public EN decision benchmark (231 items; in-domain for 1.5) | 0.762 | 0.740 |
| Coverage at 1% error (share decided automatically with the shipped threshold) | 51.0% | 55.2% |

All served with the basal engine on one H100 (`--mode fast`, bf16), both option orders averaged. Definitions:
[README](https://github.com/rkinas/basal#performance).

On the held-out test (fp32 readouts of the release checkpoints): basal-1.0-4.5B 0.858 → basal-1.5 0.875 (+1.7, 95% interval [+1.0, +2.4]).

## Quick start

**1. Install** into a fresh [uv](https://docs.astral.sh/uv/) environment (no git needed):

```bash
uv venv --python 3.12 ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[fp8] @ https://github.com/rkinas/basal/archive/refs/tags/v1.5.0.tar.gz"
```

Install torch first, from the CUDA 12.8 index: the newest torch on PyPI may need a newer GPU driver, and the
torchvision preinstalled on cloud GPU images breaks `transformers`. DGX Spark and B300: use the cu130 build
([hardware notes](https://github.com/rkinas/basal/blob/v1.5.0/docs/HARDWARE.md)).

**2. Start the server** and wait until it prints `basal: model ... ready`:

```bash
basal-serve --model Remek/basal-1.5-4.5B --mode fast --port 8000
```

The first start in mode `fast` compiles the model for a few minutes; `--mode fast-nocompile` starts in seconds and
`--mode fp8` is faster on workstation, consumer and desktop GPUs. vLLM and SGLang: `--mode vllm` / `--mode sglang`
in separate environments (vLLM: `basal[vllm]`; SGLang: install `sglang[srt]==0.5.21` first, then basal with
`--no-deps`, as in the [README](https://github.com/rkinas/basal#vllm-or-sglang)). Apple Silicon: [basal-1.5-4.5B-MLX-8bit](https://huggingface.co/Remek/basal-1.5-4.5B-MLX-8bit);
Ollama: [basal-1.5-4.5B-GGUF](https://huggingface.co/Remek/basal-1.5-4.5B-GGUF). FP8 / NVFP4 checkpoints for vLLM: [basal-1.5-4.5B-FP8](https://huggingface.co/Remek/basal-1.5-4.5B-FP8) · [basal-1.5-4.5B-NVFP4](https://huggingface.co/Remek/basal-1.5-4.5B-NVFP4).

**3. Ask questions** (in a second terminal). Several questions about one state go into one request and are answered
over one shared state:

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Klient: od wczoraj nie mogę zalogować się do bankowości internetowej, system pokazuje błąd hasła.",
  "questions": {
    "dept":   {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?",
               "criteria": {"cards": "Reklamacje kart", "online": "Wsparcie bankowości elektronicznej", "loans": "Kredyty"}},
    "urgent": {"type": "noul", "instructions": "Czy klient nie może korzystać z usługi?"}}}'
```

The response has a calibrated probability for every option, the chosen key and the confidence:

```json
{"model": "basal-1.5-4.5B",
 "answers": {"dept": {"type": "choice", "choice": "online", "probabilities": {"cards": …, "online": …, "loans": …}, "confidence": …},
             "urgent": {"type": "noul", "noul": …, "probabilities": {"true": …, "false": …}, "confidence": …}},
 "usage": {"input_tokens": …, "output_tokens": 0, "questions": 2, "branches": 2, "latency_ms": …}}
```

From Python (the `basal` package includes a client):

```python
from basal.client import Basal
b = Basal("http://127.0.0.1:8000")
state = "Klient: od wczoraj nie mogę zalogować się do bankowości internetowej."
a = b.choice(state, "Do którego działu skierować zgłoszenie?",
             {"cards": "Reklamacje kart", "online": "Wsparcie bankowości elektronicznej", "loans": "Kredyty"})
print(a["choice"], a["confidence"])                                      # chosen key and its probability
print(b.yes_no(state, "Czy klient zgłasza problem techniczny?")["noul"])  # P(yes)
print(b.score(state, "Jak pilne jest zgłoszenie?", ["niska", "średnia", "wysoka"])["score"])  # expected level 0-2
```

Many items at once: `basal-run --input items.jsonl --output answers.jsonl`. Full API (`multi`, `act`, `"evidence":
true`, `"facts": "auto"`, option keys): [API](https://github.com/rkinas/basal#api) and
[features](https://github.com/rkinas/basal/blob/v1.5.0/docs/FEATURES.md).

## Engines

| engine | `basal-serve` mode | hardware | what it is for | quick start | status (basal-1.5, 4.5B) |
|---|---|---|---|---|---|
| **basal engine** (PyTorch) | `fast` (also `fast-nocompile`, `fp8`, `nvfp4`); `eager --device mps\|cpu` | NVIDIA GPUs (sm80+); Apple Silicon (MPS) and CPU with `eager` | the primary server: packed requests (SOAM), evidence spans, `fp8` on Hopper and Blackwell | `basal-serve --model Remek/basal-1.5-4.5B --mode fast` | **verified**: served check on one H100, 13.7 ms p50 per question over HTTP, 64.4 decisions/s with 32 clients |
| **vLLM** | `vllm` | NVIDIA GPUs | an alternative CUDA server | `basal-serve --model Remek/basal-1.5-4.5B --mode vllm` (extra `basal[vllm]`) | **verified offline** (one H100, `basal-bench`, 1,000 development decisions): agreement 0.994 with the fp32 engine, accuracy 0.928 vs 0.930, 147 decisions/s |
| **SGLang** | `sglang` | NVIDIA GPUs | an alternative CUDA server (RadixAttention prefix sharing) | `basal-serve --model Remek/basal-1.5-4.5B --mode sglang` (own environment: `sglang[srt]==0.5.21`, then basal with `--no-deps`) | **verified offline** (one H100, `basal-bench`, 1,000 development decisions): agreement 0.995 with the fp32 engine, accuracy 0.925 vs 0.930, 104.6 decisions/s; on Werdykt the same answer as the basal engine on 98.3% of items (97.0% for states over 8k tokens) |
| **MLX** | `mlx` | Apple Silicon | Macs: `-MLX-8bit` (recommended) or `-MLX-fp4` (half the memory) | `basal-serve --model Remek/basal-1.5-4.5B-MLX-8bit --mode mlx` (extra `basal[mlx]`) | **verified** on an Apple M4 Pro (500 development decisions): agreement with bf16 0.992 (`-MLX-8bit`) / 0.970 (`-MLX-fp4`), 587 / 581 ms per question |
| **Ollama** | `ollama` | CPU, Apple Silicon, consumer GPUs | local and desktop use with Ollama (`-GGUF`) | 1. `hf download Remek/basal-1.5-4.5B-GGUF --local-dir basal-1.5-4.5B-GGUF` 2. in that folder: `ollama create basal-1.5-4.5b:q8_0 -f Modelfile.Q8_0` 3. `basal-serve --model ./basal-1.5-4.5B-GGUF --mode ollama --ollama-model basal-1.5-4.5b:q8_0` | **verified** on an Apple M4 Pro (500 development decisions): agreement with bf16 0.994 (`Q8_0`) / 0.980 (`Q4_K_M`), 518 / 542 ms per question |
| **llama.cpp** | `llamacpp` | CPU, Apple Silicon, consumer GPUs | `llama-server` with the `-GGUF` files; the engine's token ids are sent as is | 1. `llama-server -m basal-1.5-4.5B-GGUF/basal-1.5-4.5B-Q8_0.gguf -c 4096 --port 8080` 2. `basal-serve --model ./basal-1.5-4.5B-GGUF --mode llamacpp` | **verified** on an Apple M4 Pro (500 development decisions): agreement with bf16 0.994 (`Q8_0`) / 0.980 (`Q4_K_M`), 427 / 446 ms per question |

**`basal-serve` is the decision layer in every row.** vLLM, SGLang, MLX, Ollama and llama.cpp run the weights; the typed answers (a calibrated probability for each of your options, both option orders, the state shared by several questions, `multi`, `act` and `facts`) come from `basal-serve --mode <engine>` in front of them, with the same HTTP API in every mode. A plain `ollama run` or `llama-cli` gives free text only. Evidence spans need the PyTorch engine (`fast`, `eager`); vLLM and SGLang start with a 4,096-token context (longer states are refused); raise it with `--max-len` (e.g. 32768 for long documents). Install and first request for each engine: [quick start](https://github.com/rkinas/basal#quick-start).

## Speed

One decision = one question in both option orders (the default), median latency. About 9 GB of GPU memory for the bf16 weights.

**Release check** (one H100, `basal-serve --mode fast`, bf16, HTTP load test): one question 13.7 ms p50 /
23.8 ms p95, 64.4 decisions/s with 32 concurrent clients (basal-1.0 was not re-measured on that machine, so this is not
a per-question comparison). The speed-up of basal-1.5 is for requests with several questions (below).

**Per GPU, measured with basal-1.0-4.5B** (the same architecture; offline `basal-bench` with the basal-1.0 engine,
batch size 1, throughput with 32 option-order passes per forward). It shows what each GPU does with this model size; it
is not a measurement of basal-1.5 and not comparable with the HTTP numbers above.

| GPU | `fast` (bf16) | `fp8` |
|---|---|---|
| B300 SXM6 | **8.8 ms**, 109 dec/s | 9.7 ms, 102 dec/s |
| H100 80GB | 12.5 ms, 63 dec/s | 11.3 ms, 80 dec/s |
| RTX PRO 6000 Blackwell | 18.9 ms, 39 dec/s | 14.9 ms, 58 dec/s |
| RTX 5090 | 27.3 ms, 24 dec/s | 19.4 ms, 40 dec/s |
| DGX Spark (GB10) | 92.0 ms, 7 dec/s | **44.6 ms**, 8 dec/s |

Apple Silicon (MLX, Apple M4 Pro, 24 GB): 587 ms per decision with basal-1.5.

Several questions per request share the state: in an in-process engine comparison on one H100 (4.5B, same node and
requests) 5 questions take 37.8 ms instead of 111.7 ms as separate prompts, 12 questions 80.3 ms instead of 222.0 ms
([SOAM](https://github.com/rkinas/basal/blob/v1.5.0/docs/SOAM.md)). Every GPU we measured:
[HARDWARE.md](https://github.com/rkinas/basal/blob/v1.5.0/docs/HARDWARE.md).

## Files

- `model.safetensors`, tokenizer, `chat_template.jinja`: the model (Llama architecture, 60 layers, 32k Polish vocabulary)
- `CALIBRATION.json`: per-type temperatures and confidence thresholds for 1% / 5% error (applied by the basal server,
  also in the vLLM, SGLang, MLX and Ollama modes)
- `evidence_head.pt`: the pointer head behind `"evidence": true` (experimental)
- `basal.json`: prompt format and readout protocol

## How it works

The model receives a fixed chat prompt with the state, the question and lettered options; the assistant turn is
prefilled with `{"answer": "` and the decision is the softmax over the next-token logits of the option letters only
(one forward pass, no text generation). The server asks every question with the options in original and reversed order
and averages the two distributions (this reduces sensitivity to option order), then applies the calibrated temperature
of the question type from `CALIBRATION.json`, fitted on exactly this averaged prediction.

**Use the confidence.** `CALIBRATION.json` stores confidence thresholds chosen on calibration data, before testing, for
a target error of 1% or 5% among accepted decisions; on the test data they accept 51.0% / 76.6%
of decisions at 0.55% / 3.7% observed error. Accept decisions above the threshold
automatically and route the rest to a person. The thresholds are validated on descriptions-only prompts
(`"option_keys": "hide"`) and yes/no questions; for other formats, refit them on your own labelled requests.

**How to ask.** One condition per yes/no question (or a `multi` with one label per condition); `choice` with described
sides for named binary outcomes; ask in the language of the state; put the facts and rules in the state; offer "cannot
be determined" when the state may not contain the answer; compute exact numbers in code (or send `"facts": "auto"` for
Polish dates and amounts).

## Training

Fine-tuned from [speakleash/Bielik-4.5B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct) for typed decisions in Polish and English: rules and deadlines,
real documents, long documents, retrieval decisions, contracts, routing, ordinal scores, judging answers, action
selection and "cannot be determined" answers. The recipe includes a reinforcement-learning (RL) stage.

## Limitations

- Validate on your own documents before relying on it: hidden and held-out test sets do not cover every domain.
- Polish world knowledge of a small model is limited: provide the relevant facts in the state. Legal rules change; the
  model does not know rules introduced after its training.
- Exact arithmetic and thresholds are a weak spot of one-pass decisions: compute them in code or use `"facts": "auto"`.
- Averaging the original and reversed option order reduces, but does not remove, sensitivity to option order for three
  or more options. Up to 10 options per question.
- `act` and evidence spans are experimental ([features](https://github.com/rkinas/basal/blob/v1.5.0/docs/FEATURES.md)).
- Decisions with serious consequences for people should be reviewed by a person.

## Citation

```bibtex
@software{kinas2026basal15,
  title   = {basal-1.5: typed-decision models and inference engine for Polish and English},
  author  = {Kinas, Remigiusz},
  year    = {2026},
  version = {1.5.0},
  url     = {https://github.com/rkinas/basal}
}
```

The method of the previous release is described in the basal-1.0 technical report
([PDF](https://github.com/rkinas/basal/blob/main/docs/basal-1.pdf), doi:10.5281/zenodo.23022986).

## License and attribution

Apache-2.0. Fine-tuned from [speakleash/Bielik-4.5B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct) (Apache-2.0).
