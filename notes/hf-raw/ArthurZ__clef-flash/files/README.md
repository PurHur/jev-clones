---
license: apache-2.0
base_model: Cloudflare/clef-flash
library_name: transformers
tags:
- decision
- systemone
---

# Cloudflare/clef-flash: decision files

Chat template, settings and head of [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) for `pipeline("decision")`
([transformers#49566](https://github.com/huggingface/transformers/pull/49566)). The backbone stays the original checkpoint.

```python
from transformers import pipeline

decide = pipeline("decision", model="Cloudflare/clef-flash", decision_config="ArthurZ/clef-flash")
decide({"state": "I was charged twice and want a refund.", "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}}})
```

- `decision_template.jinja`: the chat template; `<|mark|>` marks the positions the head reads.
- `decision_config.json`: temperatures and option order.
- `decision_head/`: `ClefDecisionHead`, converted from `joint_head.safetensors` of the original.
- `decision_test.json`: a request and the input ids that the original code builds for it, used by the tests.

Derived from [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) (Apache-2.0).
