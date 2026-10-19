---
license: apache-2.0
library_name: transformers
base_model: cross-encoder/ettin-reranker-150m-v1
datasets:
- tasksource/tasksource-jev-typed-decisions
- tasksource/synthetic-typed-decisions
tags:
- typed-decisions
- decision-index
- decision-model
- cross-encoder
---

# tasksource-decider-nano

This repository was `tasksource/tasksource-jev-nano-v1`. Its `main` (tag `v4`) is the joint cross-encoder trained for
48,000 steps. The same recipe at 24,000 steps (public index 29.05) stays at tag
[`v3`](https://huggingface.co/tasksource/tasksource-decider-nano/tree/v3), at 12,000 steps (26.77) at tag
[`v2`](https://huggingface.co/tasksource/tasksource-decider-nano/tree/v2), and the previous multi-vector model at tag
[`v1`](https://huggingface.co/tasksource/tasksource-decider-nano/tree/v1).

A 150M-parameter typed-decision model for the [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index).
It takes a `state` and typed questions (`choice`, `noul`) and returns a probability for every option. It does not
generate text.

| | |
|---|---|
| Decision Index 0.3, public index | **30.90** |
| Requests | 140,171 `ok`, 7 `unsupported` (pairs above 8,192 tokens) |
| Latency, 1× NVIDIA A10 23 GB, one request at a time | median 52 ms · mean 65 ms · p80 63 ms |
| Complete results | [runs/tasksource-decider-nano](https://huggingface.co/datasets/tasksource/decision-index-results/tree/main/runs/tasksource-decider-nano) |

## Architecture

A joint cross-encoder on [cross-encoder/ettin-reranker-150m-v1](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1)
(ModernBERT-style encoder, 8,192-token context). Every option is scored against the full state, its question
and the option text, like the reranker pair `(state, "Question: <instructions>\nOption: <option>")`, by a linear
head on the mean of the option's hidden states. The scores are softmaxed over each question's options.

The state is encoded **once per request**. All questions and options of a request share it through a tree
attention mask: the state attends to itself, a question to the state and itself, and an option to the state, its
question and itself. Rotary positions are those of the single pair, so every option gets the score it would get on its
own (up to float rounding), while a request with 50 options costs one state encoding instead of 50.

Inference reads every input in full. There is no truncation: a pair above 8,192 tokens raises `Unsupported`.

## Usage

```python
from decider import Decider
model = Decider.from_pretrained("tasksource/tasksource-decider-nano")
model.answer(state={"review": "The battery died after a week."},
             questions={"q": {"type": "choice", "instructions": "What is the sentiment?",
                              "criteria": {"neg": "negative", "pos": "positive"}}})
```

Decision Index run (kit 0.3, `9eb2dbe`), from a local copy of this repository:

```bash
hf download tasksource/tasksource-decider-nano --local-dir tasksource-decider-nano
cd tasksource-decider-nano
python -m decision_index pipeline --engine decision_index_engine:DeciderEngine \
    --option repo=. --edition 0.3 --out runs/tasksource-decider-nano
```

`decider.py` and `decision_index_engine.py` are self-contained (PyTorch and transformers only).

## Training

48,000 steps, batches of 8 requests, learning rate 2e-5, inputs up to 2,048 tokens, seed 7. The loss is soft
cross-entropy over each question's options. The training mix has 480,000 requests:

| Share | Source |
|---:|---|
| 42% | [tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) (658 tasks, at most 600 questions per task) |
| 30% | *train* splits of public datasets behind Decision Index benchmarks, e.g. MMLU, ANLI, WinoGrande, HellaSwag, ARC, CLINC150, BANKING77, Amazon ESCI, HoVer, RAGTruth, New Yorker caption contest, When2Call, ACOS, iSarcasmEval, NLI4CT, ContractNLI, GSM8K (at most 10,000 requests each) |
| 12% | tool use: ToolRet train, Glaive function calling, Hermes function calling |
| 8% | LLM routing: xRouteBench and LLMRouterBench train data |
| 8% | [tasksource/synthetic-typed-decisions](https://huggingface.co/datasets/tasksource/synthetic-typed-decisions) |

No evaluation split is used, and benchmarks without a train split (for example RouterBench) are evaluation only.
Every training request goes through a firewall against the Decision Index 0.3 suite: an exact match of the
normalized state or of the (state, question) pair, and a template-insensitive word 6-gram content check, drop the
request.

## Evaluation

| Area | Skill | Raw |
|---|---:|---:|
| Knowledge & Reasoning | 7.8 | 30.8 |
| Language Understanding | 37.7 | 54.7 |
| Retrieval & Classification | 49.4 | 63.3 |
| Tools & Automation | 43.6 | 48.3 |
| Arts & Human Taste | 12.8 | 42.7 |

Raw index 47.86; kit `9eb2dbe`, edition 0.3, 140,171 requests scored.
