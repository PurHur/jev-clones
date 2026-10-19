---
license: apache-2.0
base_model: autotrust/JEV-9B
library_name: transformers
tags:
- decision
- systemone
---

# autotrust/JEV-9B: decision files

Chat template, settings and head of [autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B) for `pipeline("decision")`
([transformers#49566](https://github.com/huggingface/transformers/pull/49566)). The backbone is the base model at the root of the original with the System 1 LoRA of the original (`adapter_model.safetensors`), loaded by `from_pretrained`.

```python
from transformers import pipeline

# the LoRA at the root makes `pipeline` load its base model, so the decision files are passed explicitly
decide = pipeline("decision", model="ArthurZ/JEV-9B", decision_config="ArthurZ/JEV-9B")
decide({"state": "I was charged twice and want a refund.", "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}}})
```

- `decision_template.jinja`: the chat template; `<|mark|>` marks the positions the head reads.
- `decision_config.json`: temperatures and option order.
- `decision_head/`: `JevDecisionHead`, converted from `head.safetensors` of the original.
- `decision_test.json`: a request and the input ids that the original code builds for it, used by the tests.

Derived from [autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B) (Apache-2.0).
