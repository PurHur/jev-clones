---
license: apache-2.0
base_model: Qwen/Qwen3.6-27B
language: [en]
pipeline_tag: text-classification
tags: [decision-model, calibrated, structured-output, system-one, one-pass]
---

# decider-chat-qwen3.6-27b: Qwen/Qwen3.6-27B read through the decider readout

This repository holds [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) (revision `6a9e13bd`, weights unchanged,
55.6 GB in bf16) and the `decider_config.json` that reads it as a typed-decision model. There is no training. It is the
entry "Decider chat · Qwen3.6-27B" on the [Decision Index](https://multimodalart-jev-decision-index.static.hf.space).

- **Input:** a state and one or more typed questions (Choice, Noul yes/no, Score), each with an explicit option list.
- **Output:** a probability distribution over the options for every question, from one forward pass. There is no decoding.
- **Readout:** the model's chat template with thinking off, the option letter read at the answer slot, softmax over the
  option letters.
- **Temperature:** one temperature, T 1.943. The temperature was fitted by NLL on our own in-task rows, with the six tasks whose test splits are in the Decision Index suite excluded. An option-count rule gives this model almost nothing, so it keeps the single temperature.

## Decision Index (edition v0.2.1, 2026-09-28, measured by the index on one RTX PRO 6000)

| entry | score (chance-corrected) | rank | ECE | latency, median |
|---|---|---|---|---|
| Decider chat · Qwen3.6-27B | 51.35 | #8 of 70 | 0.021 | 83.6 ms |

Jev 1.13.0 scores 57.91 on the same edition.

We replayed 300 stored rows of the index run through `decider.serve` with this repository's config. The package picked the
same answer as the stored run on 1,431 of 1,438 answers, with a median
probability difference of 0.0003.

## Usage

Needs `decider-ai>=1.5.0`.

```python
from decider.infer import Decider
d = Decider("Mapika/decider-chat-qwen3.6-27b")
d.system_one(state, {"refund": {"type": "noul", "instructions": "Is the refund allowed under the policy?",
                                "criteria": {"true": "allowed", "false": "not allowed"}}})
```

Serving: `DECIDER_MODEL=Mapika/decider-chat-qwen3.6-27b uvicorn decider.serve:app` (POST `/v1/systemone`). For large checkpoints,
`decider.serve_vllm` serves the same readout on vLLM (see the repository's docs/SERVING.md).

## Limitations

The model is the stock base model. Its knowledge, reasoning and failure modes are the base model's. The readout adds a fixed
answer slot and a fitted temperature, nothing else.

## Licence

Weights: Qwen/Qwen3.6-27B, Apache-2.0 as published. Readout code: Apache-2.0 (github.com/Mapika/decider).
