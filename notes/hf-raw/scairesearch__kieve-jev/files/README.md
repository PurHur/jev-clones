---
library_name: kieve
tags:
  - decision-engine
  - jev
  - modernbert
  - lora
  - multiple-choice
  - calibration
license: apache-2.0
---

# Kieve-JeV

Kieve-JeV is an encoder-based **typed decision engine**. It takes a `state` (any
text) and a set of typed questions and returns, for every question, one answer
with a calibrated **probability distribution over the supplied options** (or a
probability for the yes/no `noul` primitive).

Two question primitives are supported:

- `choice` — pick one `criteria` option; the model emits a full distribution.
- `noul` — a single yes/no probability (a learned no-answer head).

The model never invents options: the distribution is defined over the options
you pass in, so it works with arbitrary, task-specific answer sets.

## Architecture

- Backbone: `answerdotai/ModernBERT-base` (frozen) with LoRA adapters (rank 16).
- Variant: `jev` — state/question encoders plus a paired cross-attention option
  scorer and a separate `noul` head.
- Operators: fused neural operators (`use_ops`) enabled.
- Trainable parameters: ~21.9M.

## Files

| File | Description |
| --- | --- |
| `checkpoint_latest.pt` | Training checkpoint. `model_state` holds the weights; `temperature` the calibrated softmax temperature. |
| `config.json` | Inference config consumed by Kieve (`params` + `fixed`). |

## Usage

### As a decision-index engine

```bash
python -m decision_index pipeline \
  --engine decision_index.engines.kieve:KieveEngine \
  --option model=scairesearch/kieve-jev \
  --out runs/kieve-jev
```

### Via the Kieve decision server (`/v1/systemone`)

```bash
python -m kieve.serve \
  --checkpoint checkpoint_latest.pt \
  --config config.json \
  --host 0.0.0.0 --port 8000
```

Then POST `{"state": ..., "questions": [...]}` to `/v1/systemone`.

## Evaluation

Single-GPU JeV run (`steps_completed` 451) on the mixed JevBench slices
(`aegis2, atbench500, jailbreak_classification, prompt_injections,
scienceqa_text, pubmedqa`):

| Metric | Value |
| --- | --- |
| Accuracy | 0.406 |
| NLL | 0.833 |

Decision Index (public 0.3) results are reported in the linked run dataset.

## License

Apache-2.0. ModernBERT backbone is subject to its own license.
