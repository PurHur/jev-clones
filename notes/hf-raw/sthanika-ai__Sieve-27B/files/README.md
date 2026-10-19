---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3.8-27B
base_model_relation: adapter
pipeline_tag: text-classification
tags:
  - sieve
  - decision-model
  - calibration
  - lora
  - multiple-choice
  - qwen3.8
metrics:
  - accuracy
  - expected_calibration_error
model-index:
  - name: Sieve-27B
    results:
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "Decision Index 0.3.2 leaderboard (public suite and the maintainers' private tests)" }
        metrics:
          - { type: decision_index, value: 60.17, name: "Decision Index 0.3.2, Full score (leaderboard)" }
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "Decision Index 0.3 public suite (140,178 scored requests; the index averages 37 benchmarks)" }
        metrics:
          - { type: decision_index, value: 60.04, name: "Decision Index 0.3, public index (chance-corrected)" }
---

# Sieve-27B

Sieve-27B is a decision model from the [Sieve family](https://huggingface.co/collections/sthanika-ai/sieve-6abb93a61dd495f69811c586). You give it a piece of text
or JSON (the *state*) and typed questions, and it returns a calibrated probability for every option of every question.
It reads the state once, scores the options directly, and **never generates text**.

It is a LoRA adapter (rank 32) and a pointer head on [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
(revision `1d4bf0f2`). The code is at [github.com/sthanika-ai/Sieve](https://github.com/sthanika-ai/Sieve).

## Summary

| | |
|---|---|
| Backbone | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (revision `1d4bf0f2`) |
| Design | pointer head |
| Trained weights | LoRA r=32, α=64 on 12 projection modules, and a pointer head |
| Question types | `choice`, `noul` and `score` |
| Decision Index | **Full score 60.17** and public score 60.04 on the leaderboard (edition 0.3.2) |
| Temperature | T = 1.050 |
| Token limits | 16,384 tokens for the state and 16,384 for each question with its options |
| GPU memory | about 55 GB in bf16 |

## How it works

- **Backbone.** Only the text part of Qwen3.8-27B is used; the model cannot produce text.
- **Readout.** A pointer head scores the end of each option against the decision point. A calibrated softmax gives the
  probabilities.
- **Isolation.** Each question runs as its own row from the state's cache, so questions never see each other.

| type | options | returns |
|---|---|---|
| `choice` | the caller's keys, optionally described (up to 255) | a probability per key |
| `noul` | yes / no | P(yes) |
| `score` | ordered levels | a probability per level |

## Use

```bash
pip install "sieve-decisions[cuda,serve] @ git+https://github.com/sthanika-ai/Sieve"
pip install --no-build-isolation causal-conv1d==1.7.0   # recommended: the fused kernel our runs used
```

```python
from sieve import load_sieve, decide

m = load_sieve("sthanika-ai/Sieve-27B")
decide(m, {"subject": "Duplicate charge on invoice 4411", "body": "Billed twice. Refund today or we cancel."},
       {"department": {"type": "choice", "instructions": "Which team handles this?",
                       "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs and outages"}},
        "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to cancel?"}})
```

- **Loading.** Load it with `sieve`; the adapter alone is not a text-generation model. The backbone is downloaded at
  its pinned revision on first use.
- **GPU memory.** About 55 GB in bf16 (52,683 MiB in use after loading), so one 80 GB GPU is enough.
- **Token limits.** States and questions up to 16,384 tokens each, as in training. `sieve` 0.2.1 or later uses these
  limits by default; earlier versions stop at 8,192 tokens. Longer requests are refused, never truncated. The Decision
  Index engine declares larger limits (a 65,536-token state and a 32,768-token question). In the Decision Index run
  below, 241 requests had a question longer than 16,384 tokens (at most 29,944, from ToolRet and BRIGHT) and were
  answered in full; no state was longer than 16,384.
- **Server.** `sieve-serve --model sthanika-ai/Sieve-27B --graphs`.
- **Decision Index.** `python -m decision_index run --engine sieve.decision_index_engine:SieveEngine --option
  model=sthanika-ai/Sieve-27B --out runs/Sieve-27B`.
- **causal-conv1d.** Without it, transformers falls back to a slower PyTorch convolution, whose probabilities can
  differ slightly. Install it to reproduce our numbers.
- **Temperature.** `head.pt` stores T = 1.050, fitted by NLL on the calibration split (43,309 questions). T never
  changes an answer, and `temperature=1.0` gives raw probabilities.

## Results

### Decision Index

On the [Decision Index leaderboard](https://huggingface.co/spaces/multimodalart/jev-decision-index) (edition 0.3.2,
board data of 2026-10-12), Sieve-27B has a **Full score of 60.17** and a public score of 60.04. Our own run of the
public 0.3 suite, with the official [kit](https://github.com/apolinario/decision-index) at `62d2f51` (0.3) and its
scorer, scores **60.04** (raw index 69.32, breadth 58.67).

- **Coverage.** All 140,178 scoreable requests over 43 benchmarks were answered, with no truncation, no unsupported
  request and no error. The index averages 37 of the benchmarks.
- **How the run was made.** One complete run of the 0.3 suite on two A100 80GB PCIe GPUs, in two shards.
- **Results.** [sthanika-ai/Sieve-27B-decision-index-results](https://huggingface.co/datasets/sthanika-ai/Sieve-27B-decision-index-results).
- **Full score.** The board ranks by the Full score: the public benchmarks count 20%, the maintainers' private tests
  of the same skills 50% and their private tasks from new domains 30%.

| Decision Index 0.3, public suite | Knowledge & Reasoning | Language Understanding | Retrieval & Classification | Tools & Automation | Arts & Human Taste |
|--:|--:|--:|--:|--:|--:|
| **60.04** | 43.5 | 65.7 | 67.8 | 77.5 | 40.8 |

The index and area scores are chance-corrected: 0 is random guessing and 100 is perfect. Below, the score is in each
benchmark's own metric and the skill is chance-corrected.

<details>
<summary>All 43 benchmarks</summary>

| benchmark | metric | score | skill (×100) | requests |
|---|---|--:|--:|--:|
| ACOS | per-review F1 | 0.258 | 23.4 | 1,565 |
| Amazon ESCI | macro-F1 | 0.599 | 49.7 | 5,000 |
| ANLI | macro-F1 | 0.760 | 64.0 | 3,200 |
| API-Bank | accuracy | 0.839 | 83.5 | 508 |
| BANKING77 | macro-F1 | 0.902 | 90.1 | 3,080 |
| BBH fixed-option tasks | accuracy | 0.803 | 71.5 | 5,507 |
| BFCL | case exact accuracy | 0.978 | 97.0 | 1,694 |
| BPoMP | accuracy | 0.944 | 88.8 | 5,000 |
| BRIGHT | nDCG@10 | 0.483 | 41.5 | 220 |
| cfcolor | accuracy | 0.643 | 29.3 | 5,000 |
| ChessBench | accuracy | 0.166 | 9.2 | 5,000 |
| CLadder | accuracy | 0.748 | 49.6 | 5,000 |
| CLINC150+OOS | macro-F1 | 0.943 | 94.3 | 5,500 |
| ContractNLI | macro-F1 | 0.803 | 71.5 | 123 |
| CRUXEval | accuracy | 0.761 | 62.1 | 570 |
| FinEntity | macro-F1 | 0.936 | 90.6 | 979 |
| GPQA Diamond | accuracy | 0.485 | 31.3 | 196 |
| GSM8K | accuracy | 0.599 | 52.2 | 2,638 |
| Habermas Machine | accuracy | 0.402 | 13.2 | 1,676 |
| HellaSwag | accuracy | 0.976 | 96.8 | 10,042 |
| HLE | accuracy | 0.102 | 0.0 | 501 |
| Home appliance simulator | case exact accuracy | 0.614 | 61.4 | 88 |
| HoVer claim verification | accuracy | 0.870 | 74.0 | 4,000 |
| Humicroedit | accuracy | 0.631 | 26.1 | 2,628 |
| iSarcasmEval | Sarcasm F1 · track A, English | 0.633 | 52.9 | 4,600 |
| MMLU-Pro | accuracy | 0.653 | 61.0 | 12,032 |
| MuSR | accuracy | 0.645 | 43.5 | 752 |
| New Yorker caption matching | accuracy | 0.731 | 66.4 | 528 |
| NLI4CT | macro-F1 | 0.855 | 71.8 | 5,500 |
| PhishNChips phishing decisions | accuracy | 0.765 | 53.0 | 2,000 |
| POP909-CL | accuracy | 0.219 | 21.1 | 2,000 |
| RAGTruth response-level hallucination | F1 on hallucinated class | 0.826 | 64.0 | 2,700 |
| SATA-Bench | case exact accuracy | 0.075 | 6.3 | 1,650 |
| ToolRet | nDCG@10 | 0.691 | 64.3 | 685 |
| VAST | macro-F1 | 0.669 | 50.3 | 3,006 |
| When2Call MCQ | accuracy | 0.821 | 76.2 | 3,652 |
| WinoGrande | accuracy | 0.926 | 85.2 | 1,267 |
| ARC-Challenge (not in the index) | accuracy | 0.974 | 96.6 | 1,172 |
| ARC-Easy (not in the index) | accuracy | 0.989 | 98.6 | 2,376 |
| MMLU (not in the index) | accuracy | 0.846 | 79.5 | 14,033 |
| RouterBench (not in the index) | selected quality (quality objective) | 0.799 | 53.0 | 10,000 |
| SGD/SGD-X (not in the index) | macro-F1 | 0.523 | 20.6 | 2,500 |
| SimpleBench (not in the index) | accuracy | 0.200 | 4.0 | 10 |

</details>

### Calibration

The calibration split chose the checkpoint and fitted the temperature.

| split | questions | accuracy | NLL | Brier | ECE |
|---|--:|--:|--:|--:|--:|
| calibration, T = 1.050 (shipped) | 43,309 | 0.9013 | 0.2499 | 0.1369 | 0.0054 |
| calibration, raw (T = 1) | 43,309 | 0.9013 | 0.2502 | 0.1368 | 0.0061 |

### Latency

- **Per Decision Index request.** In-process wall time, one request at a time, eager, over the 140,620 requests of
  the 0.3 run (A100 80GB PCIe): median 151 ms, mean 416 ms, 80th percentile 425 ms, 95th percentile 966 ms. This is
  not the maintainers' RTX PRO 6000 measurement.
- **Cached state.** Server-reported p50 on one A100 80GB PCIe (bf16, merged, CUDA graphs), for one choice question on a
  1,433-token state:

| options | 2 | 10 | 50 |
|---|--:|--:|--:|
| state cached (ms) | 137.5 | 141.6 | 236.5 |
| state new (ms) | 580.8 | 583.8 | 678.6 |

- **By option count.** For one choice question on a 41-token state, which is below the server's 64-token caching
  threshold, so every request ran the state again. Server p50 in ms, CUDA graphs / eager:

| options | 2 | 5 | 10 | 25 | 50 | 100 | 255 |
|---|--:|--:|--:|--:|--:|--:|--:|
| graphs | 177.3 | 182.7 | 186.7 | 252.6 | 369.4 | 535.8 | 1,182.2 |
| eager | 257.4 | 257.9 | 259.8 | 275.0 | 349.0 | 410.2 | 1,059.0 |

## Training

- **Recipe.**
  - LoRA r=32, α=64, dropout 0.05 on the attention, MLP and linear-attention projections (12 modules), with the
    pointer head trained from scratch.
  - AdamW, lr 5e-5 for the adapter and the head, OneCycle with 10% warm-up, 1 epoch, 32 records per step, bf16.
  - States and questions up to 16,384 tokens each.
  - 4,813 steps, about 27 hours on two A100 80GB PCIe GPUs.
  - Permuted choice options and "none of the above" augmentation, with a cross-entropy loss.
- **Selection.** The checkpoint with the highest accuracy on a fixed 2,001-question calibration sample, with NLL
  breaking ties. This rule was fixed before training, and it chose step 4,500. The benchmarks were never used to
  choose.
- **Data.** 153,988 training records, with 30,776 for calibration and 123,177 for test (50/10/40).
  - **Sources.** Public datasets and generated decision records.
  - **Not published.** The training files are not part of this release; the SHA-256 of each file is in
    [`training_config.json`](training_config.json).

## Files

| file | contents |
|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | the LoRA adapter (rank 32, PEFT format) |
| `head.pt` | the pointer head and the calibrated temperature |
| `tokenizer.json`, `tokenizer_config.json` | Qwen3.8-27B's tokenizer, unchanged |
| `training_config.json`, `training_metrics.json`, `train.log` | the recipe, the run and its log |
| `result.json` | the calibration, Decision Index and latency numbers above |
| `provenance.json` | SHA-256 of every file and of the training data, and library versions |

## License

Apache-2.0, see [`LICENSE`](LICENSE).
