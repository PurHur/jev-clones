---
license: apache-2.0
tags:
- vega
- decision-model
- typed-decisions
- universal-adapter
- calibration
---

# vega-U: one engine, any reader

One 14.3M decision engine in a shared 1024-d feature space that any frozen LLM plugs into
through a small linear port. The engine bytes are identical behind every reader in this repo
(sha16 `78f96be7c6a06abb`, asserted at upload); only the port and the per-reader calibration differ. The
reader is never fine-tuned and the engine is never retrained to add one.

![vega-U benchmarks: four readers, one byte-identical engine](assets/vega_u_benchmarks.png)

## Readers in this repository

| reader | where | calibration |
|---|---|---|
| Qwen3.5-0.8B | repo root | fitted |
| Qwen3.5-4B | `Qwen3.5-4B/` | fitted |
| Qwen3.8-27B | `Qwen3.8-27B/` | fitted |
| gemma-3-1b-it | `gemma-3-1b-it/` | fitted |

The repo root is the Qwen3.5-0.8B reader, so loading with no subfolder just works. Each reader
dir is self-contained: engine, port, extraction statistics, config with calibration.
`universal/` carries the shared-space targets and ports that the training-side tooling
(`attach_backbone.py`) uses to attach the next reader: a closed-form port fit, a short
refine against the frozen engine, and a calibration refit.

## Use

vega-U is the default model from `vegaml` 0.11, and changing the base model is naming it;
the matching port and calibration are picked from this repo by the reader's name. The token
question belongs to the reader, not to vega-U: Qwen loads with no token, Gemma 3 is gated so
`HF_TOKEN` must be set and its licence accepted.

```python
# pip install "vegaml>=0.11.0"
import vegaml

v = vegaml.load()                                # this repo, the default reader
# v = vegaml.load("4b")                          # same engine, the Qwen3.5-4B reader
# v = vegaml.load(backbone="Qwen/Qwen3.8-27B")   # the 27B reader, attached cold
# v = vegaml.load(backbone="google/gemma-3-1b-it")  # gated reader: set HF_TOKEN

out = v.decide({"body": "I was billed twice, refund me or I cancel."},
               {"churn": {"type": "boolean",
                          "instructions": "Does the customer threaten to leave?"}})
```

## Measured, October 2026 runs

Trained jointly on Qwen3.5-0.8B and Gemma 3 1B (708,657 uid-matched items, records up to
73,728 tokens, cross-reader invariance loss). The 4B and the 27B were attached cold
afterwards, and every cold-attached reader scored above the readers the engine trained with:

| reader | how | held-out acc | zero-shot acc | ECE | AURC | acc @ 50% cov |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | attached cold | **0.811** | **0.629** | 0.021 | **0.053** | **0.976** |
| Qwen3.5-4B | attached cold | 0.791 | 0.568 | 0.017 | 0.063 | 0.965 |
| Qwen3.5-0.8B | trained jointly | 0.765 | 0.534 | 0.010 | 0.073 | 0.956 |
| gemma-3-1b-it | trained jointly | 0.738 | 0.504 | 0.007 | 0.093 | 0.927 |

Where the 27B's extra capacity lands, by benchmark (0.8B in parentheses): long context to
70k tokens 0.941 (0.831), hard short context 0.972 (0.875), ARC 0.922 (0.647), MNLI 0.919
(0.836), CommonsenseQA 0.792 (0.490), CLINC150 0.948, jailbreak 0.988. Saturated perception
tasks (phishing, spam, intents, around 0.99) stay flat across readers, which is what an
unchanged engine behind better eyes should look like. The ordinal `score` question type sits
near 0.61 for every reader: that ceiling belongs to the engine or the labels, not the reader.

![the full benchmark board: every source, four readers](assets/vega_u_full_board.png)

The 27B's held-out split is capped at 400 records per source (n=21,008); the smaller readers
are scored on the full split (n=50,676). Per-reader detail lives in each dir's
`eval_report.json`.

Calibration never transfers between readers; every dir here ships its own, fitted on a
held-out calibration split.
