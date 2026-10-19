---
license: apache-2.0
base_model: autotrust/GEV-26B-Decide
library_name: transformers
tags:
- decision
- systemone
---

# autotrust/GEV-26B-Decide: decision files

Chat template, settings and head of [autotrust/GEV-26B-Decide](https://huggingface.co/autotrust/GEV-26B-Decide) for `pipeline("decision")`
([transformers#49566](https://github.com/huggingface/transformers/pull/49566)). The backbone is google/gemma-4-26B-A4B-it with the System 1 LoRA of the original (`adapter_model.safetensors`), loaded by `from_pretrained`.

```python
from transformers import pipeline

# the LoRA at the root makes `pipeline` load its base model, so the decision files are passed explicitly
decide = pipeline("decision", model="ArthurZ/GEV-26B-Decide", decision_config="ArthurZ/GEV-26B-Decide")
decide({"state": "I was charged twice and want a refund.", "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}}})
```

- `decision_template.jinja`: the chat template; `<|mark|>` marks the positions the head reads.
- `decision_config.json`: temperatures and option order.
- `decision_head/`: `JevDecisionHead`, converted from `head.safetensors` of the original.
- `decision_test.json`: a request and the input ids that the original code builds for it, used by the tests.

Derived from [autotrust/GEV-26B-Decide](https://huggingface.co/autotrust/GEV-26B-Decide) (Apache-2.0).
