---
license: apache-2.0
base_model:
- Qwen/Qwen3.5-4B
- berkay123001/turnike-jev-4b
base_model_relation: merge
pipeline_tag: text-classification
language:
- tr
- en
tags:
- decision
- decision-model
- typed-decisions
- system-one
- jev
- noul
- choice
- score
- turkish
---

# Turnike-JEV-4B — Open-Weight Decision Model (merged bf16 weights)

The LoRA adapter [berkay123001/turnike-jev-4b](https://huggingface.co/berkay123001/turnike-jev-4b) merged into Qwen/Qwen3.5-4B (bf16, same merge that
`karar.py` performs at load time). Use it without PEFT:

```python
from karar import Karar   # karar.py from berkay123001/turnike-jev-4b
k = Karar(None, taban="berkay123001/turnike-jev-4b-merged")
k.sor("Kulaklık bozuk geldi, paramı geri istiyorum.", {"Yazar iade istiyor mu?": {"type": "noul"}})
```

Model card, evaluation, training data, limitations and declarations (training data not released; not distilled
from a commercial decision API; AI assistance): see [berkay123001/turnike-jev-4b](https://huggingface.co/berkay123001/turnike-jev-4b).
License: Apache-2.0 (base model Qwen/Qwen3.5-4B: Apache-2.0, Copyright 2026 Alibaba Cloud).
