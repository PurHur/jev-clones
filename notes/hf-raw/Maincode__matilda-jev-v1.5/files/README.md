---
license: apache-2.0
library_name: transformers
tags:
- matilda
- jev
- decision-model
- classification
- maincode
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: zero-shot-classification
---

# MATILDA JEV v1.5 by Maincode

> All scores on this card are from the **Decision Index 0.3** public suite (the public dataset of the current
> leaderboard edition).

A BF16 JEV decision model with a 255-option readout. This is the
highest-scoring checkpoint in the completed MATILDA JEV experiments as of
9 October 2026: **Decision Index 63.47** (0.3 public suite).

The model accepts a state and one or more decision questions. It supports
`choice`, `noul` (probability of true), and ordinal `score`. Inference is one
forward pass per question; the model has no text-generation head and does not
generate a chain of thought. The LoRA updates are already merged into the
weights. No adapter merge is required.

## Loading

Use Python 3.12 or newer and install PyTorch for your accelerator. The validated
cluster environment uses PyTorch 2.14.0, Transformers 5.17.0, and an AMD Instinct
MI355X. Other dependencies are listed in `requirements-runtime.txt`.

```python
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

checkpoint = Path(snapshot_download('Maincode/matilda-jev-v1.5'))
sys.path.insert(0, str(checkpoint / 'runtime'))
from maincode_jev_serve.model import DecisionModel, answer

model = DecisionModel(checkpoint=checkpoint, device='cuda:0')
row = {
    'state': 'The parcel arrived on schedule.',
    'question': {
        'type': 'choice',
        'instructions': 'Classify delivery.',
        'criteria': {'on_time': 'On time', 'late': 'Late'},
    },
}
print(answer(row['question'], model.predict([row])[0]))
```

The loader uses the custom code bundled in this repository. `AutoModel` alone
loads the backbone; use `DecisionModel` to include the separate decision head.
The package retains the existing MATILDA configuration aliases and the underlying
Transformers implementation. Model weights, tokenizer vocabulary, input template,
answer codes, readout, and temperature match the evaluated checkpoint.

`predict.py` accepts JSON lines containing either `state` + `question` or
`state` + `questions`:

```json
{"state":"Two plus two equals four.","questions":{"truth":{"type":"noul","instructions":"Is this statement true?"}}}
```

```bash
python /path/to/downloaded/model/predict.py --device cuda:0 < requests.jsonl
```

## Evaluation

Decision Index 0.3 public suite: **140,178 of 140,178 scored requests answered across 37 benchmarks**
(kit [`62d2f51`](https://github.com/apolinario/decision-index/commit/62d2f51)).
Saved predictions were independently rescored with identical results. The full run, including every
prediction, is in [Maincode/matilda-jev-decision-index](https://huggingface.co/datasets/Maincode/matilda-jev-decision-index)
under `runs/matilda-jev-v1.5/`. [`scores.json`](scores.json) in this repository is the earlier 0.2.1 run
(62.76), kept for reference.

| Metric | Jev | MATILDA JEV v1 | **MATILDA JEV v1.5** |
|---|---:|---:|---:|
| Decision Index (public) | 57.96 | 60.16 | **63.47** |
| Raw | 68.55 | 70.04 | **72.62** |
| Breadth | 57.27 | 59.21 | **62.45** |
| Knowledge & Reasoning | **53.87** | 46.84 | 49.83 |
| Language Understanding | 59.23 | 64.80 | **69.91** |
| Retrieval & Classification | 55.42 | 61.07 | **65.96** |
| Tools & Automation | 75.09 | 79.07 | **80.84** |
| Arts & Human Taste | 39.08 | **46.22** | 45.41 |

Jev is the public-suite part of its Decision Index 0.3 board entry (per-benchmark public results, as published in the
kit's `tests/fixtures/board-0.3.json`), not its Full score. MATILDA JEV v1 is
[Maincode/matilda-jev-v1](https://huggingface.co/Maincode/matilda-jev-v1) (`runs/matilda-jev-v1.3/` in the same dataset).

Area scores are chance-corrected skill multiplied by 100. These results are
diagnostic: earlier training in this model's lineage used benchmark-related
material, and this release was chosen after comparing experimental scores.
The numbers are not an independent held-out generalization estimate. The full
suite evaluated text decisions; vision inference was not validated for this
release.