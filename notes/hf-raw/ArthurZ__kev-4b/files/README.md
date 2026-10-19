---
license: apache-2.0
base_model: jaredpalmer/kev-4b
library_name: transformers
tags:
- decision
- systemone
---

# jaredpalmer/kev-4b: decision files

Chat template, settings and head of [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) for `pipeline("decision")`
([transformers#49566](https://github.com/huggingface/transformers/pull/49566)). The backbone stays the original checkpoint.

```python
from transformers import pipeline

decide = pipeline("decision", model="jaredpalmer/kev-4b", decision_config="ArthurZ/kev-4b")
decide({"state": "I was charged twice and want a refund.", "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}}})
```

- `decision_template.jinja`: the chat template; `<|mark|>` marks the positions the head reads.
- `decision_config.json`: temperatures and option order.
- `decision_head/`: `KevDecisionHead`, converted from `head.pt` of the original.
- `decision_test.json`: a request and the input ids that the original code builds for it, used by the tests.

Derived from [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) (Apache-2.0).
