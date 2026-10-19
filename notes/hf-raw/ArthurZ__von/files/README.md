---
license: apache-2.0
base_model: wfzyx/von
library_name: transformers
tags:
- decision
- systemone
---

# wfzyx/von: decision files

Chat template, settings and head of [wfzyx/von](https://huggingface.co/wfzyx/von) for `pipeline("decision")`
([transformers#49566](https://github.com/huggingface/transformers/pull/49566)). The backbone stays the original checkpoint.

```python
from transformers import pipeline

decide = pipeline("decision", model="wfzyx/von", decision_config="ArthurZ/von")
decide({"state": "I was charged twice and want a refund.", "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}}})
```

- `decision_template.jinja`: the chat template; `<|mark|>` marks the positions the head reads.
- `decision_config.json`: temperatures and option order.
- `decision_head/`: `VonDecisionHead`, converted from `option_marker.pt` of the original.
- `decision_test.json`: a request and the input ids that the original code builds for it, used by the tests.

Derived from [wfzyx/von](https://huggingface.co/wfzyx/von) (Apache-2.0).
