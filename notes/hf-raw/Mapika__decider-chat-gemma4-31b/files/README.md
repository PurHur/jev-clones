---
license: apache-2.0
base_model: google/gemma-4-31B-it
language: [en]
pipeline_tag: text-classification
tags: [decision-model, calibrated, structured-output, system-one, one-pass]
---

# decider-chat-gemma4-31b: google/gemma-4-31B-it read through the decider readout

This repository holds [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) (revision `842da379`, weights unchanged,
62.5 GB in bf16) and the `decider_config.json` that reads it as a typed-decision model. There is no training. It is the
entry "Decider chat · Gemma-4-31B" on the [Decision Index](https://multimodalart-jev-decision-index.static.hf.space).

- **Input:** a state and one or more typed questions (Choice, Noul yes/no, Score), each with an explicit option list.
- **Output:** a probability distribution over the options for every question, from one forward pass. There is no decoding.
- **Readout:** the model's chat template with thinking off, the option letter read at the answer slot, softmax over the
  option letters.
- **Temperature:** T(n) = max(0.05, 10.124 − 1.633 ln n) for a question with n options (`temperature_by_options`, decider-ai ≥ 1.8.0). Gemma-4 instruct models are overconfident on questions with few options and underconfident on questions with many, so one global temperature cannot calibrate them. The rule was fitted by NLL on our own in-task rows, with the six tasks whose test splits are in the Decision Index suite excluded.

## Decision Index (edition v0.2.1, 2026-09-28, measured by the index on one RTX PRO 6000)

| entry | score (chance-corrected) | rank | ECE | latency, median |
|---|---|---|---|---|
| Decider chat · Gemma-4-31B | 57.33 | #2 of 70 | 0.047 | 108.5 ms |

Jev 1.13.0 scores 57.91 on the same edition.

We replayed 300 stored rows of the index run through `decider.serve` with this repository's config. The package picked the
same answer as the stored run on 1,412 of 1,412 answers, with a median
probability difference of 0.0001.

## Usage

Needs `decider-ai>=1.8.0`.

```python
from decider.infer import Decider
d = Decider("Mapika/decider-chat-gemma4-31b")
d.system_one(state, {"refund": {"type": "noul", "instructions": "Is the refund allowed under the policy?",
                                "criteria": {"true": "allowed", "false": "not allowed"}}})
```

Serving: `DECIDER_MODEL=Mapika/decider-chat-gemma4-31b uvicorn decider.serve:app` (POST `/v1/systemone`). For large checkpoints,
`decider.serve_vllm` serves the same readout on vLLM (see the repository's docs/SERVING.md).

## Limitations

The model is the stock base model. Its knowledge, reasoning and failure modes are the base model's. The readout adds a fixed
answer slot and a fitted temperature, nothing else.

## Licence

Weights: google/gemma-4-31B-it, Apache-2.0 as published. Readout code: Apache-2.0 (github.com/Mapika/decider).
