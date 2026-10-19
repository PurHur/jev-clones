---
pipeline_tag: zero-shot-classification
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
base_model_relation: finetune
tags:
- zero-shot-classification
- multilingual
- decision-model
- classification
- qwen3_5
- pytorch
---

![Open Decision Foundation Models — Lux-9B](assets/decision-lux-9b-header.png)


# Decision-1.0-Lux-9B

*Lux, Latin for light.*

Give Lux evidence, questions and possible answers. It returns decisions and probabilities for labels you define at runtime.


[Decision family](https://huggingface.co/collections/vllm-sr/decision-10-6ab12177bd0002394d8409f9)

| Type | Use it for | Output |
|---|---|---|
| **Choice** | Route a request or choose among 2–255 actions. | Selected ID + probability distribution |
| **Noul** | Judge a condition against supplied evidence. | P(true) |
| **Score** | Apply 2–10 ordered rubric descriptions. | Expected index + probability distribution |

## Measured capability

**77.40% weighted accuracy** across 3,766 decisions and 54 tasks: **+5.51 points over Kev-9B** and **+4.32 over Nox-4B** on the same benchmark.

| Model | Size | Decisions | Composition | Reading | Inference | Transfer | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lux-9B | 9B | **84.38** | **52.75** | 90.16 | **91.46** | 77.72 | **77.40** |
| Nox-4B | 4B | **83.00** | **51.79** | 79.06 | **86.25** | 69.60 | **73.09** |
| Kev-9B | 9B | 76.75 | 45.75 | 86.72 | 83.54 | 79.25 | 71.89 |
| Kev-4B | 4B | 71.90 | 48.54 | 81.88 | 84.58 | 76.10 | 70.09 |
| Qwen3.5-9B | 9B | 73.91 | 44.62 | 89.84 | 79.58 | 73.23 | 69.73 |
| Decider | 2B | 64.01 | 46.58 | 92.03 | 84.38 | 69.31 | 67.71 |
| Qwen3.5-4B | 4B | 69.89 | 43.33 | 87.97 | 79.79 | 68.83 | 67.29 |
| Sol-2B | 2B | 73.75 | 46.08 | 76.56 | 84.17 | 57.07 | 66.32 |
| Eos-0.8B | 0.8B | 65.94 | 46.04 | 70.31 | 81.67 | 52.01 | 61.89 |
| Kev-0.8B | 0.8B | 60.14 | 42.29 | 67.81 | 68.75 | 61.19 | 58.28 |
| Qwen3.5-2B | 2B | 57.12 | 39.00 | 73.75 | 72.29 | 56.31 | 57.24 |
| Kai-0.6B | 0.6B | 57.96 | 40.83 | 54.69 | 69.79 | 48.37 | 53.52 |
| Laya · English | 0.421B | 56.54 | 35.33 | 51.41 | 63.75 | 53.06 | 51.03 |
| Laya · Multilingual | 0.322B | 47.25 | 38.92 | 50.78 | 57.29 | 47.13 | 47.19 |
| Jev | — | 79.10 | 66.38 | 94.53 | 89.79 | 87.19 | 81.05 |

Accuracy (%), using the same five-panel decision benchmark. General decisions contribute 30%; composition contributes 25%; reading, inference and external transfer each contribute 15%. Bold marks a Decision model strictly above every external open reference in that column; Jev and other Decision models are excluded from this threshold. [Full tasks, uncertainty and comparator identities](evaluation/EVALUATION.md).

![Decision benchmark ranking](assets/decision-ranking.png)

![Capability matrix](assets/decision-matrix.png)

[All 54 tasks](evaluation/TASKS.md) · [Probability quality, order and missing evidence](evaluation/DIAGNOSTICS.md)

## Download the complete model repository

```bash
hf download vllm-sr/Decision-1.0-Lux-9B --local-dir Decision-1.0-Lux-9B
```

This downloads the complete model release. The root `config.json` lists the backbone, tokenizer, decision head, and calibration files.

## Use with 🤗 Transformers

The repository includes its inference code, so stock Transformers can download and run the complete model locally with `trust_remote_code=True`. `system_one` takes and returns the same System One request and response bodies as the Decision runtime; nothing is generated.

```bash
pip install "transformers>=5.17" torch safetensors huggingface_hub
```

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("vllm-sr/Decision-1.0-Lux-9B", trust_remote_code=True)
response = model.system_one(
    state="Customer requests a refund.",
    questions={
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Payments and refunds", "technical": "Product faults"},
        },
        "priority": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["Not urgent", "Soon", "Today", "Immediately"],
        },
    },
)
print(response["answers"]["route"]["choice"], response["answers"]["priority"]["score"])
```

`pipeline("decision", model="vllm-sr/Decision-1.0-Lux-9B", trust_remote_code=True)` accepts the same request body. A malformed question is answered with an `invalid_question` error. The model loads on the first GPU when one is visible, otherwise on the CPU (pass `device="cpu"` or `device="cuda:0"` to choose). On a GPU the backbone runs in BF16 with an FP32 decision head; on the CPU everything runs in FP32. A complete question, its candidates and the state are limited to 16,384 tokens; if a question is longer, every question of the request is answered with a `max_length_exceeded` error and nothing is truncated. [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) speeds up the linear-attention layers on a GPU.

## Serve with vLLM Semantic Router

This repository contains model data and its Transformers loading code. Use the vLLM Semantic Router Decision runtime to load `vllm-sr/Decision-1.0-Lux-9B` and serve Choice, Noul, and Score requests. The serving implementation and its dependencies live in vLLM Semantic Router; this release bundles no serving code. The weights do not depend on a particular accelerator; which hardware can serve them is decided by the runtime. For local inference without a server, see Use with 🤗 Transformers above.

After configuring a compatible Decision endpoint, send a [SystemOne request](https://docs.typesafe.ai/api) (replace the placeholder URL and key):

```bash
curl -X POST 'https://your-decision-endpoint.example/v1/systemone' \
  -H 'Authorization: Bearer YOUR_ENDPOINT_API_KEY' \
  -H 'Content-Type: application/json' \
  --data-raw '{"model":"Decision-1.0-Lux-9B","state":"Customer requests a refund.","questions":{"route":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and refunds","technical":"Product faults"}}}}'
```

The Hugging Face repository is a model download, not a hosted inference endpoint.

The published model's complete state, question, and candidates have a 16,384-token input limit. See the [evaluation scope](evaluation/EVALUATION.md) for measured conditions.

## More questions, measured

![Lux request latency](assets/decision-question-scaling.png)

Latency uses the same architecture and runtime, measured with earlier weights.

Distinct Choice questions, fixed at 499 input tokens per question. Thirty measured requests per point across six fresh processes on an otherwise idle AMD GPU. Python latency includes tokenization, inference and response construction; loading and network are excluded. [p95, memory and hardware](evaluation/QUESTION-SCALING.md).

## Architecture

![Lux decoder architecture](assets/architecture.png)

A causal Qwen3.5 text backbone combines Gated DeltaNet and full attention. A shared candidate head reads contextual candidate endpoints and the final query vector, producing one probability per supplied answer.

[Candidate head](assets/readout.png) · [Vector architecture](assets/architecture.svg) · [Model details](evaluation/METHODS.md)

Lux judges supplied evidence without live retrieval, so confidence does not guarantee factual correctness.

[License](LICENSE) · [Attributions](ATTRIBUTIONS.md)
