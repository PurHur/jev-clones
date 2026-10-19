---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
tags:
- decision-engine
- typed-decision
- choice
- classification
- merge
- qwen3.5
---

# LemonDecider 4B

**Model ID:** `lemondecider-4b-20261015-v2`
**Base model:** `Qwen/Qwen3.5-4B`

LemonDecider 4B is a typed decision engine: it reads a state and a question with a
list of options, runs **one forward pass**, and returns a probability distribution
over the options (choice / noul questions) or an ordered rating (score questions).
No text is generated and no input is ever truncated.

## Model description

| Property | Value |
|---|---|
| Architecture | Qwen3.5-4B-class hybrid backbone (linear-attention + periodic full-attention) + a bias-free **255-way readout** (`LemonDeciderReadout`, FP32 precision island), loaded through `modeling_lemondecider.LemonDeciderForDecisionClassification` (`trust_remote_code`) |
| `model_type` / protocol | `lemondecider_decision` / `lemondecider-v1` |
| Attention | `noncausal_full_attention` — all-attend prompt for the periodic full-attention layers, causal recurrence for the Gated DeltaNet layers, SDPA only |
| Pooling / temperature | last-token pooling, temperature 1.0 (uncalibrated) |
| Max input length | **262,144 tokens (256K context)** — the backbone's full position window; requests over the limit are rejected, never cut |
| Precision | backbone BF16, readout FP32 (readout GEMM executed in FP32 under autocast) |
| Serialized tensors | 724 (all vision tensors included; vision stem executed as a matrix-product patch projection) |
| Runtime | in-package Python runtime (`modeling_lemondecider.py`, `configuration_lemondecider.py`, `contract.py`, `decision_utils.py`, `model.py`), validated with transformers 5.17, flash-linear-attention 0.5.2, PyTorch SDPA |

## Training

LemonDecider 4B was fine-tuned (SFT) from the common Qwen3.5-4B initialization on
**more than 700,000 public decision/QA rows** collected by the project (about 805k
train rows from ~818k collected rows incl. dev/test splits). Every source is
public and permissive; no benchmark or evaluation data was used for training. The
released weights combine two independent SFT runs over this corpus (weight-level
combination, no training after that).

**Question formats:** choice (up to 255 options), noul (no/yes) and score (ordinal
ratings); some multimodal rows are image questions over locally registered
project-generated charts, and roughly 600k train rows carry teacher-style soft
label distributions.

**Data categories:**

- service decision arcs — user-intent routing and case triage (refund, logistics,
  account actions), policy and adequacy judgment;
- language judgment — sentiment, sarcasm, NLI/factuality and paraphrase-style checks;
- knowledge and reasoning QA — multiple-choice knowledge, multi-hop reasoning,
  arithmetic and math word problems;
- retrieval and selection — document and tabular retrieval, product/route selection;
- multimodal reading — chart/graph and small-graphic question answering;
- calibration-style rows — probability and ordinal confidence judgments.

**Training parameters** (from the training receipts):

- distributed FSDP on 64 GPUs (4 nodes x 16 GPUs);
- effective batch 256 rows per step, learning rate 2e-06;
- two independent runs: 2,446 steps (sequence cap 8,192) and 700 steps (sequence
  cap 2,048);
- objective `valid_option_soft_cross_entropy` over valid options only
  (`noul_tie_policy = true_on_equal_probability`, `choice_permutation =
  global-reference`, vision tower trainable);
- precision policy fp32-master-bf16; export dtype BF16, readout FP32, temperature 1.0.

## Package artifacts

### Weight files

| File | Size | SHA-256 |
|---|---:|---|
| `model-00001-of-00002.safetensors` | 4,973,336,472 | `27b985a94ccf63aace88b2ff67c1ea0871ce745a49b22b9cda43ea4d32297d8a` |
| `model-00002-of-00002.safetensors` | 4,107,897,400 | `81f61ce9daead4be7e3d950cf027f63f16f757a176407f020406190dc2068bc0` |
| `tokenizer.json` | 19,989,492 | `6f32ce20dc35f57a7f9ad1eac03525bd7d30f9df8cea6507e958279cc3657706` |
| `model.safetensors.index.json` | 68,426 | `bda09c55c0152d7a6fb2db6898aea70e728dc528df6f9c026cefb0bd5d1d673b` |


## Evaluation results

Both evaluations ran on 2026-10-15 on self-hosted compute (64 GPUs) through the
model's own HTTP `/v1/systemone` wire contract. **No result here
is an official leaderboard or board submission**; the harnesses enforce no-truncation,
no option filtering, one fixed rendering, and coverage-adjusted scoring. Latency is
per-request server wall time on internal accelerators and is not the sub-1-second
latency-admission measurement of any external board.

Evidence directories keep the run names under which they were created during the v1
execution (`lemondeeder-eval-20261015/`); no shipped artifact in the model packages
carries that legacy spelling.

### 1) Decision Index, edition 0.3 — FULL public suite (complete run)

⚙️ 140,620 requests issued across 64 shards (frozen suite-0.3 staging, official
scorer, coverage-adjusted, chance-corrected) → **140,178 scored requests,
`complete: true`**.

| Headline | Value |
|---|---:|
| **Decision Index (public 0.3)** | **50.31** |
| Raw index | 62.47 |
| Median / mean / p95 request latency | 50.4 ms / 205.6 ms / 756.2 ms |

The public index averages the native, chance-corrected, coverage-adjusted score of
37 benchmarks in five areas with fixed weights (knowledge 25.8%, language 25.8%,
retrieval 20.0%, tools 18.3%, arts 10%); inside an area, ★ gold benchmarks weigh
1.2 and the rest 1.0. Unanswered counts as wrong.

#### Per-area scores

| Area | Raw | Skill (chance-corrected) | Coverage | Benchmarks |
|---|---:|---:|---:|---:|
| Knowledge & Reasoning | 0.4895 | 0.3395 | 0.9998 | 11 |
| Language Understanding | 0.6962 | 0.5751 | 1.0000 | 9 |
| Retrieval & Classification | 0.6661 | 0.5494 | 1.0000 | 6 |
| Tools & Automation | 0.7108 | 0.6741 | 0.9986 | 5 |
| Arts & Human Taste | 0.5487 | 0.3340 | 0.9978 | 6 |

#### Per-benchmark native scores (★ = counted in the public index)

| # | Benchmark | Metric | Requests | Score | Skill | Cov. |
|---:|---|---|---:|---:|---:|---:|
| 1 | BFCL ★ | case exact accuracy | 1,694 | 0.9416 | 0.9211 | 1.000 |
| 2 | ToolRet ★ | nDCG@10 | 685 | 0.6540 | 0.5982 | 0.997 |
| 3 | API-Bank ★ | accuracy | 508 | 0.8320 | 0.8254 | 0.996 |
| 4 | BANKING77 ★ | macro-F1 | 3,080 | 0.7418 | 0.7385 | 1.000 |
| 5 | CLINC150+OOS ★ | macro-F1 | 5,500 | 0.8634 | 0.8626 | 1.000 |
| 6 | RouterBench | selected quality | 10,000 | 0.7746 | 0.4715 | 1.000 |
| 9 | Home appliance simulator ★ | case exact accuracy | 88 | 0.4091 | 0.4091 | 1.000 |
| 10 | SGD/SGD-X | macro-F1 | 2,500 | 0.4360 | 0.0616 | 1.000 |
| 11 | ContractNLI ★ | macro-F1 | 123 | 0.7389 | 0.6224 | 1.000 |
| 12 | ANLI ★ | macro-F1 | 3,200 | 0.6729 | 0.5100 | 1.000 |
| 20 | BPoMP ★ | accuracy | 5,000 | 0.8718 | 0.7525 | 1.000 |
| 21 | Humicroedit ★ | accuracy | 2,628 | 0.6111 | 0.2222 | 1.000 |
| 22 | POP909-CL ★ | accuracy | 2,000 | 0.1890 | 0.1701 | 0.987 |
| 23 | cfcolor ★ | accuracy | 5,000 | 0.6088 | 0.1827 | 1.000 |
| 24 | MMLU | accuracy | 14,033 | 0.7914 | 0.7219 | 1.000 |
| 25 | GPQA Diamond ★ | accuracy | 196 | 0.4286 | 0.2381 | 1.000 |
| 26 | ARC-Easy | accuracy | 2,376 | 0.9836 | 0.9781 | 1.000 |
| 27 | ARC-Challenge | accuracy | 1,172 | 0.9548 | 0.9397 | 1.000 |
| 28 | WinoGrande ★ | accuracy | 1,267 | 0.8019 | 0.6038 | 1.000 |
| 29 | HellaSwag ★ | accuracy | 10,042 | 0.8520 | 0.8027 | 1.000 |
| 30 | GSM8K ★ | accuracy | 2,638 | 0.5716 | 0.4866 | 1.000 |
| 31 | ChessBench ★ | accuracy | 5,000 | 0.1384 | 0.0616 | 1.000 |
| 32 | MuSR ★ | accuracy | 752 | 0.5731 | 0.3213 | 1.000 |
| 33 | SATA-Bench ★ | case exact accuracy | 1,650 | 0.1539 | 0.1427 | 1.000 |
| 34 | SimpleBench | accuracy | 10 | 0.1000 | 0.0000 | 1.000 |
| 36 | BRIGHT ★ | nDCG@10 | 220 | 0.4444 | 0.3714 | 1.000 |
| 37 | Amazon ESCI ★ | macro-F1 | 5,000 | 0.4827 | 0.3511 | 1.000 |
| 38 | ACOS ★ | per-review F1 | 1,565 | 0.1970 | 0.1713 | 1.000 |
| 39 | FinEntity ★ | macro-F1 | 979 | 0.9381 | 0.9090 | 1.000 |
| 40 | iSarcasmEval ★ | Sarcasm F1 (track A, EN) | 4,600 | 0.5552 | 0.4278 | 1.000 |
| 41 | VAST ★ | macro-F1 | 3,006 | 0.7251 | 0.5876 | 1.000 |
| 42 | NLI4CT ★ | macro-F1 | 5,500 | 0.7997 | 0.6103 | 1.000 |
| 43 | CRUXEval ★ | accuracy | 570 | 0.6035 | 0.3709 | 1.000 |
| 44 | CLadder ★ | accuracy | 5,000 | 0.6846 | 0.3692 | 1.000 |
| 45 | HLE ★ | accuracy | 501 | 0.1140 | 0.0000 | 0.998 |
| 50 | Habermas Machine ★ | accuracy | 1,676 | 0.4063 | 0.1384 | 1.000 |
| 56 | PhishNChips ★ | phishing-decision accuracy | 2,000 | 0.7315 | 0.4630 | 1.000 |
| 57 | MMLU-Pro ★ | accuracy | 12,032 | 0.5780 | 0.5254 | 1.000 |
| 58 | BBH fixed-option tasks ★ | accuracy | 5,507 | 0.7018 | 0.5678 | 1.000 |
| 59 | RAGTruth ★ | F1 on hallucinated class | 2,700 | 0.7601 | 0.5026 | 1.000 |
| 61 | HoVer ★ | claim-verification accuracy | 4,000 | 0.7225 | 0.4450 | 1.000 |
| 62 | When2Call MCQ ★ | accuracy | 3,652 | 0.6528 | 0.5371 | 1.000 |
| 64 | New Yorker caption matching ★ | accuracy | 528 | 0.6307 | 0.5384 | 1.000 |

Evidence: `lemondeeder-eval-20261015/di/runs/20261015-lemondecider-4b-di-full-0.3/`
(64 shard dirs + `merged/results.jsonl` + official `scores.json`, `index.json`,
`benchmark-summary.json`).

### 2) JevBench public-231 (typed-decision public sanity suite)

231/231 items attempted and scorable, **complete**, native probability output,
dataset hash-pinned (`dc3995d8…de51`, jevbench-v1 harness, typesafe adapter).

| Metric | Value |
|---|---:|
| **Accuracy** | **83.12%** (192/231) |
| Macro accuracy | 84.55% |
| Brier (mean) | 0.2392 |
| Ordinal MAE (score items) | 0.1207 |
| Paraphrase consistency | 36/36 pairs agree (100%) |
| Coverage / operational success / schema validity | 1.0 / 1.0 / 1.0 (strict 1.0) |
| Latency p50 / p95 | 47.1 ms / 340.3 ms |

#### Per-family results

| Family | n | Accuracy | Brier |
|---|---:|---:|---:|
| adequacy | 12 | 1.000 | 0.051 |
| adversarial | 6 | 1.000 | 0.002 |
| ambiguous | 7 | 0.571 | 0.661 |
| extraction | 24 | 1.000 | 0.000 |
| fact | 12 | 1.000 | 0.000 |
| intent | 24 | 1.000 | 0.001 |
| judge_hard | 17 | 0.647 | 0.499 |
| long_policy | 19 | 0.579 | 0.585 |
| multi_hop | 18 | 0.722 | 0.481 |
| ordinal | 12 | 1.000 | 0.000 |
| policy | 12 | 1.000 | 0.007 |
| probability | 10 | 0.800 | 0.338 |
| routing | 12 | 1.000 | 0.013 |
| routing_hard | 5 | 1.000 | 0.000 |
| temporal_numeric | 15 | 0.067 | 1.006 |
| tool_selection | 12 | 1.000 | 0.000 |
| tradeoff | 6 | 0.833 | 0.450 |
| trap | 8 | 1.000 | 0.038 |

Known weak spots on this suite: long-policy judgement, ambiguous states, hard-judge
calls, and especially **temporal/numeric reasoning (6.7%)**, which the model inherits
from its parents' training data.

Evidence: `lemondeeder-eval-20261015/jevbench/20261015-lemondecider-4b-jevbench-public231/`
(`results.jsonl`, `ledger.jsonl`, `public-summary.json`, 5-item smoke gate passed).

## Usage

Dependencies: `transformers==5.17.0` (needs the Qwen3.5 hybrid classes),
`flash-linear-attention==0.5.2`, `pillow`, `torch` with SDPA.
This package honours a single runtime env var: `LEMONDECIDER_CUDNN_SDPA=1`
(keep the cuDNN SDPA backend on CUDA; defaults are kept on CPU/ROCm).

```python
import sys, torch
from transformers import AutoModel

ckpt = "lemondecider-4b-20261015-v2"      # local package directory
sys.path.insert(0, ckpt)                  # make the in-package runtime importable
from decision_utils import load_processor, predict_batch

model = AutoModel.from_pretrained(
    ckpt, trust_remote_code=True, local_files_only=True,
    dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
).eval()
processor = load_processor(ckpt)

rows = [{
    "state": "The user asks how to return a damaged order placed last week.",
    "question": {"type": "choice",
                 "instructions": "Pick the best next action",
                 "criteria": {"refund": "Process a refund",
                              "replace": "Send a replacement",
                              "escalate": "Escalate to a human"}},
}]
answers = predict_batch(model, processor, rows)
print(answers["answers"][0])   # choice, probabilities over the 3 options, confidence
```

CLI from inside the package: `python3 model.py --checkpoint . --device cuda:0 --input rows.json`.

Batching: each `predict_batch` call runs ONE forward pass over all the rows you pass
(inputs are left-padded to the longest prompt); the HTTP serving bridge chunks its
requests at 8 questions per forward pass. Requests over the 262,144-token
`max_input_length` limit are rejected with the limit named (HTTP 422 in server
wrappers) — nothing is truncated.

## Intended use & limitations

- Intended: typed decision endpoints — routing, intent classification, policy/
  adequacy checks, structured QA over explicit option lists, pipeline control.
- The model emits **option probabilities only**; it is not a generative assistant.
- Ratings (`score` items) are supported through the same readout; they inherit the
  parents' weak arithmetic/timing behavior (see the temporal_numeric row above).
- Trained on public data sources assembled by the project; no evaluation rows were
  used for training (corpus admission contracts enforced this).
- Uncalibrated: temperature 1.0, no post-hoc calibration fitted. The 6 low-confidence
  bins in the JevBench calibration table show mild over-confidence in the 0.6–0.8
  confidence range.

## License

- `LICENSE` — Apache-2.0 text as preserved from the parent checkpoint lineage
  (base: Qwen/Qwen3.5-4B, inherit the base model's terms).
- `LICENSE.code` — MIT for runtime/server code components adapted from upstream.
- No benchmark score is claimed by this package beyond the reproducible evidence
  files listed above; the Decision Index owner reserves 80% of its Full score for
  private tests, so the 50.31 public index is not a board submission.


## Acknowledgements

Built by the Lemon project. Runtime and server components build on Apache-2.0
and MIT upstream code (see LICENSE files). Base backbone by the Qwen team.