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

# basal-1.0-4.5B

[![GitHub](https://img.shields.io/badge/GitHub-rkinas%2Fbasal%20%E2%80%94%20inference%20engine-black?logo=github)](https://github.com/rkinas/basal)
[![Technical report](https://img.shields.io/badge/technical%20report-PDF-b31b1b.svg)](https://github.com/rkinas/basal/blob/main/docs/basal-1.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23022986.svg)](https://doi.org/10.5281/zenodo.23022986)
[![Collection](https://img.shields.io/badge/%F0%9F%A4%97-basal--1.0%20collection-yellow)](https://huggingface.co/collections/Remek/basal-10-6ab8224bf7bd8732d7a6117d)

![basal-1.0 overview](basal.png)

**What it is.** Inspired by *System 1* (fast, intuitive) decision models such as Jev: instead of writing an answer,
the model reads a *state* (a message, a document, a case file, a web page as JSON) and answers a typed question about it
— `choice`, yes/no (`noul`) or `score` — by returning a **calibrated probability for each allowed answer**, in a single
forward pass, without generating text. The answer can never fall outside the options you give, and the probability says
how sure the model is. The name comes from the *basal ganglia*, which select one action among competing options.

**What it is for: a dynamic classifier.** The classes are described *in the request*, in plain language, so one model
serves many tasks without retraining: ticket and document routing with changing categories, rule and policy checks
("is the claim covered?", "was the appeal filed in time?"), urgency or risk scores, agent and tool decisions and guard
checks in LLM pipelines, and triage with a confidence threshold (accept confident decisions automatically, send the rest
to a person).

- **Best on Polish decisions**: 0.884, against 0.780 for Jev 1.13.0 and 0.779 for the best of eleven open decision
  systems (AutoJev-27B); on English decisions not distinguishable from Jev or the best open systems (differences of −1.2 to +0.5 points, all
  within their 95% intervals).
- **Calibrated**: per-type temperatures and confidence thresholds for 1% / 5% accepted error in `CALIBRATION.json`;
  with the shipped threshold (fixed before testing, target 1% error) it decides 58.6% of the held-out test decisions
  automatically at 1.2% observed error (Jev 1.13.0 under the same procedure: 18.1%).
- **Fast**: 8.8 ms per decision (both option orders) on a B300, 12.5 ms on an H100 (14.1 ms end to end over HTTP; 13.9 ms on the benchmark sets of the leaderboard),
  27 ms on an RTX 5090, 45 ms on a DGX Spark with FP8.
- **Early exits** (`exit_heads/`): optional per-request speed setting, 12.2 → 10.5 ms on H100 at unchanged agreement (see below).

> 📊 **More results — accuracy and speed of basal-1.0 against Jev and other open decision models:** [jev-pl-benchmark](https://huggingface.co/spaces/Remek/jev-pl-benchmark) (Polish score, English score, speed).

## Quick start

**1. Install** into a fresh [uv](https://docs.astral.sh/uv/) environment (no git needed):

```bash
uv venv --python 3.12 ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[fp8] @ https://github.com/rkinas/basal/archive/refs/tags/v1.0.1.tar.gz"
```

Install torch first, from the CUDA 12.8 index: the newest torch on PyPI may need a newer GPU driver, and the
torchvision preinstalled on cloud GPU images breaks `transformers`. DGX Spark and B300: see
[installation](https://github.com/rkinas/basal#quick-start).

**2. Start the server** and wait until it prints `basal: model ... ready`:

```bash
basal-serve --model Remek/basal-1.0-4.5B --mode fast --port 8000
```

The first start in mode `fast` compiles the model for a few minutes; `--mode fast-nocompile` starts in seconds. `--mode fast-exit` enables per-request early exits and `--mode fp8` is faster on workstation, consumer and desktop GPUs.

**3. Ask a question** (in a second terminal):

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Klient: od wczoraj nie mogę zalogować się do bankowości internetowej.",
  "questions": {"dept": {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?",
    "criteria": {"cards": "Reklamacje kart", "online": "Wsparcie bankowości elektronicznej", "loans": "Kredyty"}}}}'
```

The response has a calibrated probability for every option, the chosen key and the confidence:

```json
{"model": "basal-1.0-4.5B",
 "answers": {"dept": {"type": "choice", "choice": "online",
   "probabilities": {"cards": …, "online": …, "loans": …}, "confidence": …}},
 "usage": {"input_tokens": …, "output_tokens": 0, "questions": 1, "latency_ms": …}}
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

Many items at once: `basal-run --input items.jsonl --output answers.jsonl`
([JSONL format](https://github.com/rkinas/basal#inference-on-a-jsonl-file)). Several questions per request, option keys, early exit and the
full response format: [API](https://github.com/rkinas/basal#api).

## Quality

| system | params | PL decisions | PL general | EN decisions | Public bench. |
|---|---|---|---|---|---|
| **basal-1.0-4.5B** | 4.5B | **0.884** | 0.737 | 0.741 | 0.740 |
| basal-1.0-1.5B | 1.5B | 0.849 | 0.656 | 0.734 | 0.675 |
| Jev 1.13.0 (commercial API) | – | 0.780 | – | 0.736 | 0.861 |
| Cygnet | 12B | 0.688 | 0.793 | 0.703 | **0.879** |
| AutoJev-27B | 27B | 0.779 | **0.833** | **0.753** | 0.870 |
| Jev-Omni | 12B | 0.687 | 0.768 | 0.694 | 0.866 |
| JevK5 v0.2 | 4B | 0.630 | 0.744 | 0.670 | 0.857 |
| Winnow-12B | 12B | 0.688 | 0.772 | 0.703 | 0.853 |
| decider-4b v2 | 4B | 0.709 | 0.717 | 0.694 | 0.835 |
| decider-35B-A3B | 35B (3B active) | 0.694 | 0.781 | 0.751 | 0.831 |
| Hopper | 4B | 0.649 | 0.727 | 0.669 | 0.823 |
| reflex-4B | 4B | 0.586 | 0.729 | 0.645 | 0.814 |
| nimble-9B v2 | 9B | 0.685 | 0.758 | 0.669 | 0.805 |
| kev-4B | 4B | 0.694 | 0.690 | 0.666 | 0.758 |

*PL decisions*: 7,081 held-out Polish decisions from unseen templates, statutes and domains; *PL general*: Polish
knowledge, exams and reading comprehension; *EN decisions*: 1,479 held-out English decisions; *Public bench.*: the
231-item public English decision benchmark (official harness). All systems served on one H100 with their own servers;
both option orders averaged. Leaderboard: [jev-pl-benchmark](https://huggingface.co/spaces/Remek/jev-pl-benchmark). The basal public-benchmark scores are measured with engine v1.0.1, which shows option keys next to their
descriptions by default (the benchmark's options have meaningful keys); with v1.0 they were 0.706 (4.5B) and 0.662 (1.5B).

## Speed

One decision = both option orders (default; reduces sensitivity to option order), batch size 1, median latency; throughput with 32
option-order passes per forward. `fast` keeps decisions practically identical to fp32 (agreement 0.99–1.00); `fp8` changes 2–4%.

| GPU | class | `fast` (bf16) | `fp8` | HTTP (`fast`) |
|---|---|---|---|---|
| B300 SXM6 | server (Blackwell) | **8.8 ms**, 109 dec/s | 9.7 ms, 102 dec/s | **9.7 ms, 109 dec/s** |
| H100 80GB | server | **12.5 ms**, 63 dec/s | 11.3 ms, 80 dec/s | 14.1 ms, 63 dec/s |
| RTX PRO 6000 Blackwell | workstation | 18.9 ms, 39 dec/s | 14.9 ms, 58 dec/s | 22.5 ms, 39 dec/s |
| RTX 5090 | consumer | 27.3 ms, 24 dec/s | 19.4 ms, 40 dec/s | 32.3 ms, 23 dec/s |
| DGX Spark (GB10) | desktop | 92.0 ms, 7 dec/s | **44.6 ms**, 8 dec/s | – |

Where FP8 helps depends on the bottleneck: on the DGX Spark (memory-bandwidth-bound) it halves latency, on workstation
and consumer cards it gives 1.3–1.4×, and on the B300 (launch-overhead-bound at batch 1) bf16 is already fastest.
**NVFP4** (4-bit) lowers accuracy on this model by about 3 points (0.79 vs 0.82 on our speed sample; ~10% of decisions
change) and is not recommended for single requests; with vLLM it raises batch throughput on the DGX Spark from 8 to 27
decisions/s. Details for every GPU: https://github.com/rkinas/basal/blob/main/docs/HARDWARE.md.

**Early exit** (`--mode fast-exit`, request field `"early_exit"`). *What it is:* the model has 60 layers, and for
many questions the answer is already clear before the last one. Small trained **exit heads** in `exit_heads/` (a
normalisation layer and a low-rank adapter that reuse the model's output head) sit after layers 30, 35, 40, 45 and 50–55. During
the forward pass the server checks them in turn; when the probability of the top option exceeds a threshold calibrated
for that layer, the remaining layers are skipped and the exit head's answer is returned. The thresholds are calibrated
so that the early answer agrees with the full model on a chosen share of decisions (99.9%, 99.5%, 99% or 98% on
calibration data). Each request picks its level (`"early_exit": "0.99"`) or `"off"` (default: always the final layer),
so one server serves both. The gain is modest because the decision becomes readable only in the last ten of 60 layers, and a batch stops
only when all its requests are confident. Measured on H100 (B300: 8.8 → 7.7 ms at `0.99`, agreement 0.998):

| `early_exit` | latency | agreement with fp32 | decisions stopped early |
|---|---|---|---|
| `off` | 12.2 ms | 0.994 | 0% |
| `0.995` | 11.0 ms | 0.993 | 33% |
| `0.99` | **10.5 ms** | 0.992 | 47% |
| `0.98` | 10.9 ms | 0.982 | 79% |

## Files

- `model.safetensors`, tokenizer, `chat_template.jinja` — the model (Llama architecture, 60 layers, 32k Polish vocabulary)
- `CALIBRATION.json` — per-type temperatures and confidence thresholds (applied by the basal server)
- `exit_heads/` — trained early-exit heads (layers 30–55) and calibrated thresholds per agreement level
- `basal.json` — prompt format, readout protocol and test metrics

## How it works

The model receives a fixed chat prompt with the state, the question and lettered options; the assistant turn is
prefilled with `{"answer": "` and the decision is the softmax over the next-token logits of the option letters only
(one forward pass, no text generation). The server asks every question with the options in original and reversed order
and averages the two distributions (reduces sensitivity to option order), then applies the calibrated temperature of the
question type from `CALIBRATION.json`, fitted on exactly this averaged prediction (calibration v1.0.1).

**Use the confidence.** `CALIBRATION.json` also stores confidence thresholds chosen on the calibration split, before
testing, for a target error of 1% or 5% among accepted decisions; applied once to the test split they accept
58.6% / 75.0% of test decisions at 1.2% / 4.4% observed error. Accept decisions above the threshold automatically and route the rest to a person;
with your own data, refit the thresholds on a labelled sample. These numbers were measured on descriptions-only prompts (`"option_keys": "hide"`); in the default mode, which
also shows option keys, they are not validated — refit the thresholds on your own labelled requests.

## Training data

Polish and English decision data whose labels are computed by code (deadlines, amounts, rule families with twin
pairs that differ in one fact), grounded in statutes (verbatim evidence quotes checked by independent verifiers), or
agreed by independent verifier models; plus English decisions from a public dataset (about 21% of the training items). Generated
data are split by template, statute and domain, so test items come from templates, statutes and domains never seen in
training; the English items follow the source corpus's own train/test split. Part of the data
was generated or verified with commercial models.

## Limitations

- Evaluated on held-out items from the same generation pipelines as training plus public benchmarks; validate on your
  own documents before relying on it.
- Polish world knowledge of a small model is limited: provide the relevant facts in the state.
- Legal rules change; the model does not know rules introduced after its training.
- A generator error in the training data taught the model the wrong notice period (art. 36 § 1 KP) when three years of
  employment are completed during a one-month notice: it answers one month instead of three. Evaluation labels are
  corrected; the model will be retrained in the next release.
- Averaging the original and reversed option order reduces, but does not remove, sensitivity to option order for three
  or more options.
- About 21% of the training items (13,500 English items) come from an aggregated public corpus whose upstream sources
  could not be traced item by item; see the technical report.
- The test split was consulted during development; its results come from an adaptive process on held-out templates,
  not from a single untouched final evaluation.
- Decisions with serious consequences for people should be reviewed by a person.


## Citation

```bibtex
@techreport{kinas2026basal,
  title       = {basal-1.0: Reliable, Highly Optimized Typed Decisions for Polish},
  author      = {Kinas, Remigiusz},
  institution = {ai5},
  year        = {2026},
  type        = {Technical report},
  doi         = {10.5281/zenodo.23022986},
  url         = {https://doi.org/10.5281/zenodo.23022986}
}
```

## License and attribution

Apache-2.0. Fine-tuned from [speakleash/Bielik-4.5B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct) (Apache-2.0).

Training data: English decision items were converted from [avbiswas/bev-decision-150K](https://huggingface.co/datasets/avbiswas/bev-decision-150K), which aggregates questions derived from many upstream sources; its maintainers ask users to check and attribute those sources, and row-level source identifiers are not available (see the technical report).
