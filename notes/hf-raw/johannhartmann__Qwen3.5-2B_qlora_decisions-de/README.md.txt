---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.5-2B
base_model_relation: finetune
datasets:
- johannhartmann/decisions-de
language:
- de
tags:
- clef
- systemone
- decision-model
- structured-decisions
- calibration
- qwen3.5
- unsloth
- german
- custom-code
---

# Qwen3.5-2B_qlora_decisions-de

A small German decision model: [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) (via
[`unsloth/Qwen3.5-2B`](https://huggingface.co/unsloth/Qwen3.5-2B)) fine-tuned with QLoRA and a fresh
Clef joint-schema head on [`johannhartmann/decisions-de`](https://huggingface.co/datasets/johannhartmann/decisions-de)
in Unsloth Studio. Given a state and a set of typed questions (`choice`, `noul`, `score`), it returns a
probability for every allowed option of every question in one forward pass, in the System One /
Decision API answer format.

The repository uses the same layout as Cloudflare's Clef releases
([`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash)): a merged bf16 Qwen3.5
backbone (vision encoder included), `joint_head.safetensors`, and Cloudflare's unmodified
`joint_schema_model.py` with `load_release_model` and `systemone`. It loads with plain `transformers`;
Unsloth is not needed.

This is an independent fine-tune by johannhartmann. It is not affiliated with or endorsed by
Cloudflare, Qwen, or Unsloth.

## Results

Measured on the final model with Cloudflare's loader (bf16, plain `transformers`, no Unsloth):

| Eval set | Cases | Decisions | Accuracy | All answers of a case right | ECE |
|---|---:|---:|---:|---:|---:|
| `decisions-de` validation (in-domain) | 565 | 1,215 | **0.842** | 0.750 | 0.031 |
| German Decision Index 0.3 sample (local, see below) | 1,525 | 13,307 | 0.793 | 0.413 | 0.190 |
| [`typed-decisions-de`](https://huggingface.co/datasets/johannhartmann/typed-decisions-de) `test` (other domain) | 400 | 2,000 | 0.504 | 0.048 | 0.129 |

By question type on the validation split: `noul` 0.940, `choice` 0.769, `score` 0.588.

Unsloth Studio's own held-out check (400 decisions from the training data): accuracy 0.503 before →
0.848 after training, ECE 0.138 → 0.042.

- **Broad, not specialised.** `decisions-de` mixes 90 task sources (intent detection, NLI, reading
  comprehension, moderation, preference judgements, logic, rule-based agent decisions). The model
  does well on that mix, but it was not trained on the four business workflows of
  `typed-decisions-de` and their 5-level score scales, and it is weak there (0.50). For those
  workflows, train on `typed-decisions-de` as well, or use
  [`johannhartmann/gemma-4-E4B-it-unsloth-bnb-4bit_qlora_typed-decisions-de`](https://huggingface.co/johannhartmann/gemma-4-E4B-it-unsloth-bnb-4bit_qlora_typed-decisions-de)
  (0.776 on that test split).
- **Decision Index sample:** 2,000 representative questions rebuilt with the public Decision Index
  0.3 kit, translated to German, reduced to the `choice`/`noul` questions that survive translation
  (35 benchmarks). Most of its decisions are binary yes/no `choice` questions, so the accuracy is
  not comparable with the official leaderboard. The set is not redistributable and was never used
  for training.
- **Calibration:** in-domain the confidences match the accuracy closely (ECE 0.03). Out of domain the
  model is underconfident (mean confidence 0.62 at accuracy 0.79 on the Decision Index sample).
- The training data is mostly hard labels (95 %), so out-of-domain distributions are less graded than
  those of a model trained on soft teacher labels.

## Usage

```python
import sys

from huggingface_hub import snapshot_download

path = snapshot_download("johannhartmann/Qwen3.5-2B_qlora_decisions-de")
sys.path.insert(0, path)
from joint_schema_model import load_release_model, systemone

model, processor = load_release_model(path, device="cuda")
print(systemone(model, processor, {
    "model": "decider",
    "state": {"rechnung": {"betrag": 1250.0, "status": "seit 30 Tagen überfällig"}},
    "questions": {
        "status": {"type": "choice", "instructions": "Welchen Status hat die Rechnung?",
                   "criteria": {"paid": "bezahlt", "overdue": "überfällig", "draft": "Entwurf"}},
        "gross": {"type": "noul", "instructions": "Ist der Betrag höher als 1000 EUR?"},
    },
}))
# {'answers': {'status': {'type': 'choice', 'choice': 'overdue', 'confidence': 0.9998, ...},
#              'gross': {'type': 'noul', 'noul': 0.9905}}, ...}
```

Tested with `torch` and `transformers` 5.3. With Unsloth, `FastDecisionModel.from_pretrained` loads the
folder too and additionally applies the per-type temperatures in `unsloth_decision_config.json`
(noul 1.13, choice 0.88, score 0.67). Cloudflare's loader ignores that file. Unsloth's global
temperature fit (1.56) is already folded into the head weights, so both loaders return calibrated
probabilities. The per-type temperatures improve in-domain ECE only slightly (0.031 → 0.021).

**GGUF for llama.cpp:**
[`johannhartmann/Qwen3.5-2B_qlora_decisions-de-GGUF`](https://huggingface.co/johannhartmann/Qwen3.5-2B_qlora_decisions-de-GGUF)
(Q8_0, Q6_K, Q4_K_M, BF16; served by `llama-server` at `/v1/systemone`, llama.cpp b11443 or newer).
Q8_0 scores 0.845 on the `decisions-de` validation split, the same as PyTorch.

## Training

| | |
|---|---|
| Base | `unsloth/Qwen3.5-2B`, loaded in 4-bit (QLoRA), merged into a bf16 copy of the base after training |
| Head | new Clef joint-schema head (width 512, 2 routing + 4 decoder layers, 8 heads) |
| Data | `johannhartmann/decisions-de`, config `all`, split `train` (19,286 rows) |
| LoRA | r 64, alpha 64, dropout 0, q/k/v/o/gate/up/down, language and vision layers |
| Optimiser | adamw_torch, lr 1e-4, linear schedule, 5 warmup steps, weight decay 0.01 |
| Batch | 16 × 2 gradient accumulation |
| Length | max_seq_length 4096 |
| Schedule | 2 epochs, 1,194 steps |
| Objective | soft cross-entropy on the gold distributions |
| Calibration | temperatures fitted on held-out data |
| Hardware | 1 × RTX A6000, 72 min, 12.4 GB peak |

## License

The weights are released under Apache-2.0, like the base model (`LICENSE`).
`joint_schema_model.py` is Cloudflare's Apache-2.0 reference code, unmodified.

The training data keeps per-source upstream licenses (CC-BY, CC-BY-SA, MIT, Apache-2.0 and others;
see the [dataset card](https://huggingface.co/datasets/johannhartmann/decisions-de)). Machine-translated
moderation data means the model has seen offensive content.
