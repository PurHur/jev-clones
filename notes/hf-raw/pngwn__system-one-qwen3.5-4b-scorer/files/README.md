---
license: cc-by-nc-4.0
base_model: Qwen/Qwen3.5-4B-Base
library_name: peft
tags:
- sequence-classification
- calibration
- decision-making
- lora
- system-one
---

# System One scorer — Qwen/Qwen3.5-4B-Base + scalar scoring head

A single-pass "System One" decision model in the shape of TypeSafe's Jev: it takes
unstructured state plus a set of typed questions (yes/no, Choice, numeric Score) and
returns a **probability distribution over exactly the options the caller supplied** —
in one forward pass, with no autoregressive generation.

Each (state, question, option) triple is scored by a sequence-classification head and
the per-question logits are softmaxed. Because the output space *is* the option set,
the output is type-safe by construction rather than by post-hoc parsing: there is no
token stream that can drift outside the schema.

Training code: [`system_one.py`](./system_one.py) in this repository.

## Intended use

Route, classify, prioritise or score a decision whose options you already know.
This is not a chat model and not a generator — it cannot produce free text.

## Results

Held-out test split, option sets **uncapped at evaluation**, temperature 1.75 fitted on val.

| task | n | accuracy | ECE | Brier |
| --- | --- | --- | --- | --- |
| ag_news | 64 | 0.922 | 0.021 | 0.131 |
| banking77 | 64 | 0.891 | 0.049 | 0.176 |
| go_emotions | 64 | 0.859 | 0.062 | 0.202 |
| tickets_language | 64 | 0.891 | 0.069 | 0.189 |
| tickets_type | 64 | 0.750 | 0.052 | 0.329 |
| mmlu | 64 | 0.703 | 0.188 | 0.407 |
| yelp_score | 64 | 0.641 | 0.117 | 0.460 |
| tickets_priority | 64 | 0.469 | 0.058 | 0.612 |
| tickets_queue | 64 | 0.234 | 0.244 | 0.848 |
| **ALL** | **576** | **0.707** | **0.044** | **0.373** |

Validation split, all 9 families, per-task cap 64 (n=537): accuracy 0.752, ECE 0.032.

### Calibration is the point

| model | accuracy | ECE | Brier |
| --- | --- | --- | --- |
| this model, raw head (test, n=576) | 0.707 | 0.135 | 0.415 |
| this model, T=1.75 fitted on val (test, n=576) | 0.707 | **0.044** | 0.373 |
| prompted Qwen3.5-4B-Base, 26-letter answer (val, n=112) | 0.679 | 0.093 | 0.439 |

Temperature scaling cuts ECE from 0.135 to 0.044 — about 2× better calibrated than the
prompted baseline, at identical accuracy. The raw head is *over*confident; that gap is
the whole reason a calibration stage exists.

The baseline row is **not** measured on the same rows: it covers 7 of 9 families
(112 questions, 16 per family), because banking77 (77 options) and ticket routing (52)
do not fit its 26-letter answer alphabet. On those 7 families this scorer averages
0.748 test accuracy against the baseline's 0.679.

### Latency

112.3 ms per question at 4 options — one forward pass, option batch scored together.
A smoke run measured 559.7 ms at 77 options. Latency scales with option count and
sequence length, not with output length, because nothing is generated.

## Training

- data: `pngwn/system-one-decisions` — 12,913 train questions across 9 task families
- 2,200 optimizer steps (step-capped, ~1.4 epochs), batch 8 questions, max_len 384,
  option cap 16, lr 1e-4 cosine with 3% warmup
- LoRA r=16 over all linear projections plus a new scalar `score` head
  (30.5M trainable of 4.24B)
- bf16 with gradient checkpointing; a100-large; 2h58m wall clock including eval and push
- train loss 1.682 (step 25) → 0.302 (step 2200); loss plateaued near 0.5 by step ~500

## Limitations

- **Option-cardinality mismatch.** High-cardinality tasks are trained with a cap of 16
  options but evaluated over all of them (banking77 77, ticket routing 52), because the
  cap is what keeps batching tractable. `tickets_queue` is the visible casualty at 0.234
  accuracy, and it is also the worst-calibrated task (ECE 0.244).
- **384-token truncation.** Long states (MMLU questions, long reviews) are truncated, so
  MMLU and `yelp_score` are the weakest non-routing tasks.
- **Knowledge-heavy multiple choice is not the strength** of encoder-style single-pass
  scoring. It trades world knowledge for latency and schema safety.
- **The ticket component of the training data is CC-BY-NC-4.0**, so this model inherits a
  non-commercial restriction.
- **A 4-epoch run was attempted and cancelled** around step 500: the measured 0.18–0.22
  steps/s could not finish inside the timeout, and the script only pushes after training
  completes, so continuing would have produced nothing.
- Accuracy is far below frontier models. The claims here are type-safe output, calibrated
  distributions and single-pass latency — not intelligence.
