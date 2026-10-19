---
language:
- en
library_name: gliclass
pipeline_tag: zero-shot-classification
tags:
- gliclass
- deberta
- zero-shot-classification
- text-classification
- multi-label-classification
- instruction-following
- reranking
- guardrails
- content-moderation
- knowledge-graph-reasoning
license: apache-2.0
base_model: microsoft/deberta-v3-large
---
# GLiClass v4 Edge

**Instruction-following zero-shot classification in a 0.4B encoder, built for CPUs and edge devices.**

`knowledgator/gliclass-v4-edge` is the 0.4B-parameter edge release of
[GLiClass v4](https://github.com/Knowledgator/GLiClass). The labels you pass and an
optional instruction tell it what to decide at inference time. It returns a score for
every label in one forward pass and generates no text. It reaches a Decision Index public skill of 47.5 at a median
latency of 11 ms per decision, ahead of every model up to 1B parameters on the public board (the best of them scores 36.9).


## Usage

```bash
pip install gliclass -U
```

Or install from source in a Python 3.10+ environment:

```bash
git clone https://github.com/Knowledgator/GLiClass.git
cd GLiClass
pip install -e .
```

The following examples reuse `pipeline`:

```python
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline

model_id = "knowledgator/gliclass-v4-edge"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = GLiClassModel.from_pretrained(model_id)
pipeline = ZeroShotClassificationPipeline(
    model,
    model_id,  # loads the tokenizer
    classification_type="multi-label",
    max_length=2048,
    device=device,
)
```

For a local copy, replace the model ID with the checkpoint directory.

### Multi-label classification

```python
results = pipeline(
    "The battery lasts two days, but the screen scratches easily.",
    ["battery", "display", "durability", "price", "shipping"],
    threshold=0.5,
)[0]
for result in results:
    print(result["label"], round(result["score"], 3))
```

Each prediction contains `label` and `score`. Pass a list of texts and `batch_size=8` for batched inference.

Scores are calibrated on held-out data; predictions also carry `initial_score`, the uncalibrated score.

### Single-label decisions and scoring

Use `classification_type="single-label"` to pick one answer: scores are a softmax over the labels.
To score an input, use the levels of a scale as labels and read their probabilities:

```python
answer = pipeline(
    "Our refund window is 30 days. The customer bought the item 45 days ago.",
    ["refund approved", "refund denied", "escalate to a human"],
    prompt="Apply the refund policy to the customer's request.",
    classification_type="single-label",
)[0]
print(answer)

quality = pipeline(
    "Q: What causes tides? A: Mostly the Moon's gravity, with a smaller effect from the Sun.",
    ["1 - wrong", "2 - poor", "3 - partly correct", "4 - good", "5 - excellent"],
    prompt="Rate the factual quality of the answer.",
    classification_type="single-label",
    return_hierarchical=True,
)[0]
print(quality)  # every level with its probability
```

### Instruction following

Use `prompt` for the instruction, policy, or rubric. Labels are interpreted in light of it:

```python
policy = (
    "Flag a message only if it shares another person's contact details without consent. "
    "Public business contact pages are allowed."
)
results = pipeline(
    [
        "You can reach our support team at help@acme.com.",
        "Here is my neighbour's personal phone number: 555-0134, call her at night.",
    ],
    ["violates policy", "complies with policy"],
    prompt=policy,
    classification_type="single-label",
)
```

Few-shot examples (`examples=[{"text": ..., "labels": [...]}]`) and hierarchical labels
(`{"sentiment": ["positive", "negative"], "topic": ["product", "service"]}`) are also supported.

### Knowledge graph reasoning

Pass the graph as text next to the story, the question as the instruction, and the answers as labels:

```python
graph = (
    "Alice -member_of-> Finance team\n"
    "Finance team -can_read-> Reports folder\n"
    "Reports folder -contains-> Salaries.xlsx\n"
    "Alice -explicit_deny-> Salaries.xlsx\n"
    "Rule: explicit restrictions override team permissions."
)
answer = pipeline(
    graph,
    ["yes", "no"],
    prompt="Can Alice open Salaries.xlsx?",
    classification_type="single-label",
)[0]
print(answer)  # [{'label': 'no', 'score': ...}]
```

The same pattern covers multi-hop questions, agent and web-browsing steps (which link, tool, or action
comes next), and NLI (`entailment`, `neutral`, `contradiction`).

### Reranking

```python
query = "How do I reset a forgotten router password?"
passages = [
    "Hold the reset button for 10 seconds to restore factory settings, then log in with the default password.",
    "Our routers support Wi-Fi 7 and dual-band operation.",
    "To change your Wi-Fi name, open the admin panel and edit the SSID field.",
]
scores = pipeline(
    passages,
    [query],
    prompt="Does the passage answer the query?",
    threshold=0.0,
)
ranked = sorted(zip(passages, scores), key=lambda p: -p[1][0]["score"])
```

## Serving a Jev-style decision API

Decision engines such as Jev take a **state** (text or JSON) and a set of typed **questions**, and return a
probability for every option of every question, with no generated text. GLiClass answers the same requests
directly: each question is one GLiClass pass in which the state is the text, the question's `instructions`
are the prompt, and every option is a label. This is the setup used for the Decision Index results below.

| Question type | Request fields | GLiClass labels | Answer |
| --- | --- | --- | --- |
| `choice` | `instructions`, `criteria: {key: description}` | the option descriptions (the key when a description is missing) | `choice`, `confidence`, `probabilities` per key |
| `noul` (yes / no) | `instructions`, optional `criteria: {"true": ..., "false": ...}` | the true and false descriptions (`yes` / `no` when missing) | `noul` = P(true) |
| `score` | `instructions`, `criteria: [level 0, level 1, ...]` | the level descriptions, in order | `score` = expected level, `confidence`, `probabilities` per level |

Probabilities are a softmax over the option logits. A JSON state is rendered as `key: value` lines, with string lists as `- item` lines.

The server below exposes `POST /v1/systemone` and `GET /v1/models`, the request and response shape of
TypeSafe's System One API, so clients written for it (for example the
[WorkflowEvals](https://github.com/typesafe-ai/WorkflowEvals) runner) can point at a local GLiClass model.
It needs only `gliclass`, `fastapi`, and `uvicorn`:

```python
# jev_api.py
import json
import os

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gliclass import GLiClassModel, ZeroShotClassificationPipeline

MODEL_ID = os.environ.get("MODEL", "knowledgator/gliclass-v4-edge")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "16384"))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32

model = GLiClassModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE).eval()
pipe = ZeroShotClassificationPipeline(model, MODEL_ID, max_length=10**9, classification_type="single-label",
                                      device=DEVICE, progress_bar=False).pipe
app = FastAPI(title="GLiClass System One API")


class Request(BaseModel):
    state: str | dict | list | None = None
    questions: dict[str, dict]


def as_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def render_state(state):
    if state in (None, "", {}, []):
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, list):
        return "\n".join(f"- {as_text(item)}" for item in state)
    lines = []
    for key, value in state.items():
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            lines += [f"{key}:"] + [f"- {item}" for item in value]
        else:
            lines.append(f"{key}: {as_text(value)}")
    return "\n".join(lines)


def options(question):
    kind, criteria = question.get("type"), question.get("criteria")
    if kind == "choice":
        if not criteria or len(criteria) < 2:
            raise HTTPException(422, "a choice question needs at least two criteria")
        return list(criteria), [as_text(v) if v is not None else k for k, v in criteria.items()]
    if kind == "noul":
        c = criteria or {}
        return ["true", "false"], [as_text(c.get("true") or "yes"), as_text(c.get("false") or "no")]
    if kind == "score":
        if not criteria or len(criteria) < 2:
            raise HTTPException(422, "a score question needs at least two levels")
        return [str(i) for i in range(len(criteria))], [as_text(c) for c in criteria]
    raise HTTPException(422, f"unsupported question type {kind!r}")


@torch.no_grad()
def probabilities(text, prompt, labels):
    inputs = pipe.prepare_inputs([text], labels, same_labels=True, prompt=prompt)
    length = inputs["input_ids"].shape[-1]
    if length > MAX_LENGTH:  # refuse instead of truncating
        raise HTTPException(422, f"input of {length} tokens exceeds the {MAX_LENGTH}-token limit")
    logits = model(**inputs, max_num_classes=len(labels)).logits[0, : len(labels)].float()
    return torch.softmax(logits, dim=-1).tolist(), length


@app.get("/v1/models")
def models():
    return {"models": [{"name": MODEL_ID, "description": "GLiClass"}]}


@app.post("/v1/systemone")
def systemone(request: Request):
    text, answers, tokens = render_state(request.state), {}, 0
    for name, question in request.questions.items():
        keys, labels = options(question)
        prompt = as_text(question.get("instructions") or "") or None
        probs, length = probabilities(text, prompt, labels)
        tokens += length
        p = dict(zip(keys, probs))
        best = max(keys, key=p.get)
        if question["type"] == "choice":
            answers[name] = {"type": "choice", "choice": best, "confidence": p[best], "probabilities": p}
        elif question["type"] == "noul":
            answers[name] = {"type": "noul", "noul": p["true"]}
        else:
            answers[name] = {"type": "score", "score": sum(int(k) * v for k, v in p.items()),
                             "confidence": p[best], "probabilities": p}
    return {"model": MODEL_ID, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}
```

```bash
pip install gliclass fastapi uvicorn
uvicorn jev_api:app --host 0.0.0.0 --port 8765
```

```bash
curl -s localhost:8765/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "Checkout errors hit 12% after the 14:05 deploy.",
            "facts": ["Rolling back takes two minutes.", "No incident is open yet."]},
  "questions": {
    "action": {"type": "choice", "instructions": "What should the on-call engineer do right now?",
               "criteria": {"rollback": "Roll back to the last healthy build.",
                            "wait": "Watch error rates for another hour.",
                            "escalate": "Page the payments team lead."}},
    "urgent": {"type": "noul", "instructions": "Does this need action within the next hour?",
               "criteria": {"true": "Yes, customers are affected right now.", "false": "No, it can wait."}},
    "severity": {"type": "score", "instructions": "Rate the incident severity.",
                 "criteria": ["minor", "moderate", "major", "critical"]}
  }}'
```

The response has one answer per question. This one comes from gliclass-v4-edge (rounded):

```json
{
  "model": "knowledgator/gliclass-v4-edge",
  "answers": {
    "action": {"type": "choice", "choice": "rollback", "confidence": 0.938, "probabilities": {"rollback": 0.938, "wait": 0.059, "escalate": 0.003}},
    "urgent": {"type": "noul", "noul": 0.300},
    "severity": {"type": "score", "score": 0.24, "confidence": 0.832, "probabilities": {"0": 0.832, "1": 0.124, "2": 0.013, "3": 0.031}}
  },
  "usage": {"input_tokens": 172, "output_tokens": 0}
}
```

Here the edge model picks the rollback but rates the incident as minor and not urgent; check judgment-style
questions such as severity on your own data before relying on them.

- **Instructions matter.** Put the decision rule or rubric in `instructions` and give `noul` questions explicit `true` / `false` descriptions; bare yes / no labels are less reliable.
- **Throughput.** The server answers questions one at a time. For batch jobs, group questions that share a state with `pipe.prepare_inputs([text] * n, labels_per_question, same_labels=False, prompt=prompts)` and run one forward pass.
- **Long inputs** over `MAX_LENGTH` tokens are refused with HTTP 422, never truncated, as the Decision Index rules require.
- **Calibration.** For calibrated checkpoints `logits` are the calibrated logits; `uncalibrated_logits` holds the raw ones.
- The Decision Index results below come from a kit engine (`gliclass_engine.GLiClassEngine`) that applies exactly this mapping, run with `python -m decision_index run --engine gliclass_engine:GLiClassEngine --model knowledgator/gliclass-v4-edge`.

## Reported evaluation results

Columns **Edge 0.4B**, **9B**, and **27B** are [gliclass-v4-edge](https://huggingface.co/knowledgator/gliclass-v4-edge),
[gliclass-v4-9b](https://huggingface.co/knowledgator/gliclass-v4-9b), and
[gliclass-v4-27b](https://huggingface.co/knowledgator/gliclass-v4-27b); this card is **Edge 0.4B**.

### Decision Index 0.3

Public part of the [Decision Index](https://github.com/apolinario/decision-index) 0.3 (37 benchmarks in five
areas, 140,178 requests), scored with the kit's own scorers. Skill is chance-corrected (0 = random guessing,
100 = perfect) and coverage-adjusted (unanswered = wrong); the public index weighs the areas 25.8 / 25.8 / 20.0 /
18.3 / 10.0%.

GLiClass models: one pass per question through the Jev-style mapping above, on one RTX PRO 6000
(this model: bf16, `max_length=16384`, batch size 8). Edge 0.4B, 9B, and 27B take a median of 11.0, 34.0, and 73.7 ms per request. Other models: the per-benchmark public results in the kit's board snapshot, with area scores recomputed using the kit's aggregation, which reproduces each published index
exactly. The main board ranking (the Full score) also includes private tests; this table covers the public
part only.

| Model | Public index (skill) | Raw index | Breadth | Knowledge | Language | Retrieval | Tools | Arts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Edge 0.4B** | 47.49 | 59.70 | 44.54 | 21.4 | 54.6 | 66.7 | 58.1 | 38.6 |
| **9B** | 55.56 | 66.10 | 53.99 | 36.6 | 61.6 | 61.6 | 74.8 | 41.5 |
| **27B** | 61.74 | 70.76 | 60.66 | 45.7 | 67.6 | 63.0 | 81.0 | 50.3 |
| Torchcast 27B | 65.10 | 73.64 | 63.86 | 46.7 | 72.0 | 67.5 | 84.6 | 54.4 |
| pplx-decider 27B | 62.25 | 71.66 | 61.52 | 52.0 | 67.5 | 61.3 | 78.9 | 47.0 |
| clef | 61.71 | 71.07 | 61.04 | 52.8 | 61.3 | 62.8 | 80.7 | 48.9 |
| GLiDE 28B | 59.06 | 69.61 | 57.96 | 45.8 | 59.7 | 58.4 | 83.6 | 48.2 |
| Jev 1.13 | 57.96 | 68.55 | 57.27 | 53.9 | 59.2 | 55.4 | 75.1 | 39.1 |
| Kev 27B | 56.69 | 67.55 | 55.91 | 45.7 | 62.6 | 55.7 | 72.6 | 42.7 |

Every benchmark, as skill (★ = gold benchmark, weight 1.2 inside its area):

<details>
<summary>Per-benchmark Decision Index results</summary>

#### Knowledge & Reasoning

| Benchmark | **Edge 0.4B** | **9B** | **27B** | Torchcast 27B | pplx-decider 27B | clef | GLiDE 28B | Jev 1.13 | Kev 27B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPQA Diamond ★ | 10.2 | 25.2 | 26.5 | 35.4 | 41.5 | 31.3 | 36.0 | 71.4 | 29.9 |
| GSM8K | 2.1 | 42.3 | 60.1 | 58.1 | 72.0 | 52.1 | 61.2 | 66.2 | 65.5 |
| ChessBench | 10.7 | 10.4 | 13.7 | 10.8 | 10.3 | 17.8 | 9.9 | 9.8 | 11.0 |
| MuSR | 34.7 | 38.9 | 49.7 | 48.0 | 53.5 | 74.2 | 44.8 | 46.1 | 43.8 |
| SATA-Bench | 24.5 | 26.0 | 34.4 | 13.9 | 27.2 | 32.6 | 28.6 | 25.4 | 26.9 |
| CRUXEval | 0.0 | 28.7 | 57.4 | 66.0 | 77.2 | 78.6 | 57.4 | 57.1 | 63.8 |
| CLadder | 42.2 | 54.4 | 63.8 | 49.0 | 58.6 | 88.2 | 54.9 | 45.3 | 48.3 |
| HLE ★ | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.7 | 0.0 |
| MMLU-Pro ★ | 20.5 | 46.4 | 57.5 | 66.7 | 67.4 | 61.7 | 66.9 | 80.5 | 63.2 |
| BBH fixed-option tasks ★ | 28.1 | 55.8 | 66.9 | 72.0 | 76.8 | 61.8 | 68.6 | 89.7 | 70.8 |
| WinoGrande ★ | 60.4 | 71.4 | 73.6 | 87.7 | 84.8 | 87.1 | 71.9 | 83.9 | 76.6 |

#### Language Understanding

| Benchmark | **Edge 0.4B** | **9B** | **27B** | Torchcast 27B | pplx-decider 27B | clef | GLiDE 28B | Jev 1.13 | Kev 27B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ContractNLI | 69.6 | 77.0 | 76.8 | 80.4 | 68.7 | 73.0 | 69.5 | 59.1 | 68.0 |
| ANLI ★ | 42.6 | 52.8 | 62.5 | 68.2 | 61.5 | 55.0 | 60.0 | 62.2 | 56.5 |
| HellaSwag ★ | 77.1 | 89.8 | 93.6 | 96.8 | 91.0 | 97.6 | 93.4 | 92.7 | 92.5 |
| ACOS | 13.3 | 28.2 | 21.3 | 55.9 | 25.9 | 31.0 | 22.1 | 27.3 | 20.8 |
| FinEntity | 86.4 | 83.3 | 92.2 | 90.0 | 93.3 | 94.2 | 88.3 | 80.8 | 91.2 |
| iSarcasmEval | 37.1 | 38.9 | 54.5 | 35.8 | 62.7 | 31.0 | 22.2 | 36.3 | 50.2 |
| VAST | 59.6 | 61.7 | 66.5 | 72.9 | 65.1 | 39.0 | 54.4 | 46.9 | 52.4 |
| NLI4CT | 45.2 | 56.9 | 71.0 | 73.2 | 70.6 | 66.9 | 69.7 | 69.0 | 67.2 |
| RAGTruth response-level hallucination | 58.6 | 62.2 | 65.8 | 71.0 | 65.0 | 57.6 | 51.4 | 51.3 | 59.8 |

#### Retrieval & Classification

| Benchmark | **Edge 0.4B** | **9B** | **27B** | Torchcast 27B | pplx-decider 27B | clef | GLiDE 28B | Jev 1.13 | Kev 27B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BANKING77 ★ | 90.7 | 80.1 | 80.0 | 91.0 | 80.3 | 94.0 | 79.4 | 79.5 | 85.9 |
| CLINC150+OOS ★ | 92.2 | 85.0 | 88.9 | 93.4 | 91.5 | 97.4 | 87.3 | 89.2 | 86.5 |
| BRIGHT ★ | 22.4 | 31.5 | 33.3 | 44.3 | 44.4 | 40.6 | 43.6 | 40.6 | 41.7 |
| Amazon ESCI | 39.3 | 45.4 | 50.3 | 51.9 | 50.3 | 46.5 | 47.9 | 43.8 | 46.5 |
| PhishNChips phishing decisions | 86.8 | 55.8 | 47.9 | 40.7 | 46.3 | 58.7 | 37.7 | 25.1 | 24.8 |
| HoVer claim verification | 67.6 | 69.2 | 74.8 | 78.8 | 48.2 | 31.2 | 47.7 | 45.7 | 39.7 |

#### Tools & Automation

| Benchmark | **Edge 0.4B** | **9B** | **27B** | Torchcast 27B | pplx-decider 27B | clef | GLiDE 28B | Jev 1.13 | Kev 27B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BFCL ★ | 87.2 | 94.4 | 96.2 | 96.9 | 96.2 | 97.9 | 97.0 | 94.3 | 97.0 |
| ToolRet | 47.9 | 56.4 | 60.8 | 63.5 | 62.6 | 64.3 | 63.4 | 59.9 | 63.5 |
| API-Bank ★ | 82.5 | 87.8 | 86.8 | 85.0 | 85.2 | 91.8 | 84.8 | 88.0 | 82.7 |
| Home appliance simulator | 12.5 | 58.0 | 77.3 | 89.8 | 80.7 | 80.7 | 92.0 | 52.3 | 63.6 |
| When2Call MCQ | 49.7 | 71.2 | 79.7 | 85.3 | 65.1 | 63.3 | 77.9 | 74.6 | 49.4 |

#### Arts & Human Taste

| Benchmark | **Edge 0.4B** | **9B** | **27B** | Torchcast 27B | pplx-decider 27B | clef | GLiDE 28B | Jev 1.13 | Kev 27B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BPoMP | 82.4 | 83.2 | 90.9 | 91.1 | 90.5 | 94.1 | 88.0 | 81.8 | 90.6 |
| Humicroedit | 27.6 | 27.0 | 28.8 | 25.2 | 22.3 | 33.1 | 21.7 | 23.7 | 24.9 |
| POP909-CL | 23.1 | 11.6 | 46.0 | 50.0 | 55.5 | 14.8 | 70.0 | 15.9 | 35.2 |
| cfcolor | 12.1 | 25.5 | 30.2 | 43.8 | 28.1 | 34.0 | 30.3 | 28.8 | 28.8 |
| Habermas Machine | 33.8 | 38.6 | 36.6 | 42.3 | 15.8 | 54.5 | 14.5 | 21.5 | 13.7 |
| New Yorker caption matching | 52.6 | 63.1 | 69.7 | 73.7 | 69.5 | 62.8 | 64.5 | 62.6 | 62.8 |

</details>

#### Models up to 1B parameters

Edge 0.4B against the sub-1B entries of the live board (the jev-decision-index Space, data generated 2026-10-15): the
eight highest-scoring ones, plus LiquidAI d1-omni-600M and GLiNER2.5-Decide. Sizes are the board's served
parameters; models over 1B served parameters are excluded even when their name says 0.8B. Board rank is the
position on the public index among the 168 board entries plus Edge 0.4B. Area scores are recomputed from each model's
per-benchmark public results with the kit's aggregation and match every published index.

| Model | Parameters | Public index (skill) | Board rank | Raw index | Breadth | Knowledge | Language | Retrieval | Tools | Arts | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Edge 0.4B** | 435M | 47.49 | 71 / 169 | 59.70 | 44.54 | 21.4 | 54.6 | 66.7 | 58.1 | 38.6 | 100.0% |
| vLLM-SR d3-lite | 853M | 36.93 | 106 / 169 | 52.58 | 34.41 | 18.6 | 34.2 | 53.7 | 56.3 | 22.3 | 100.0% |
| Crivo-150M | 150M | 36.39 | 107 / 169 | 52.45 | 31.50 | 11.8 | 41.8 | 52.2 | 60.7 | 10.1 | 100.0% |
| Sieve 0.8B | 590M | 32.26 | 114 / 169 | 49.40 | 28.90 | 14.3 | 26.6 | 54.8 | 51.2 | 13.5 | 100.0% |
| tasksource nano v4 | 149M | 30.90 | 115 / 169 | 47.86 | 26.81 | 7.8 | 37.7 | 49.4 | 43.6 | 12.8 | 100.0% |
| Saina Helm 2.0 | 853M | 30.47 | 119 / 169 | 48.35 | 25.67 | 8.7 | 31.6 | 43.8 | 58.3 | 6.3 | 100.0% |
| jiwo 0.8B v2 | 752M | 29.58 | 123 / 169 | 47.20 | 26.75 | 10.9 | 36.1 | 44.2 | 40.3 | 12.2 | 100.0% |
| DecisionTune 1.0 | 395M | 28.85 | 125 / 169 | 46.84 | 25.43 | 11.7 | 29.9 | 45.0 | 46.5 | 5.7 | 100.0% |
| vLLM-SR d3-edge | 594M | 28.82 | 126 / 169 | 46.79 | 25.44 | 11.1 | 24.1 | 53.6 | 42.7 | 11.8 | 100.0% |
| d1-omni-600M | 587M | 17.90 | 146 / 169 | 37.58 | 16.11 | 6.5 | 19.3 | 35.0 | 19.5 | 6.5 | 100.0% |
| GLiNER2.5-Decide | 486M | 11.48 | 155 / 169 | 33.08 | 10.13 | 2.3 | 8.3 | 25.3 | 14.8 | 9.8 | 100.0% |

<details>
<summary>Per-benchmark Decision Index results, models up to 1B</summary>

#### Knowledge & Reasoning (models up to 1B)

| Benchmark | **Edge 0.4B** | vLLM-SR d3-lite | Crivo-150M | Sieve 0.8B | tasksource nano v4 | Saina Helm 2.0 | jiwo 0.8B v2 | DecisionTune 1.0 | vLLM-SR d3-edge | d1-omni-600M | GLiNER2.5-Decide |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPQA Diamond ★ | 10.2 | 10.9 | 0.0 | **15.0** | 0.0 | 3.4 | 1.4 | 0.0 | 4.1 | 0.0 | 0.0 |
| GSM8K | 2.1 | **29.9** | 2.0 | 24.0 | 7.5 | 13.5 | 7.1 | 2.0 | 21.6 | 1.1 | 0.2 |
| ChessBench | **10.7** | 0.2 | 0.0 | 1.1 | 1.6 | 9.6 | 0.0 | 0.0 | 0.0 | 5.2 | 0.0 |
| MuSR | **34.7** | 33.0 | 26.7 | 29.2 | 32.4 | 18.0 | 31.5 | 20.5 | 20.3 | 1.5 | 12.9 |
| SATA-Bench | **24.5** | 18.6 | 0.0 | 13.0 | 8.4 | 14.7 | 21.0 | 1.7 | 10.2 | 6.2 | 0.0 |
| CRUXEval | 0.0 | 0.0 | 7.3 | 0.0 | 8.1 | 0.0 | 0.0 | 5.9 | **9.8** | 6.2 | 5.1 |
| CLadder | 42.2 | 22.1 | **61.0** | 6.5 | 0.6 | 3.2 | 2.9 | 44.8 | 8.1 | 0.0 | 0.2 |
| HLE ★ | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.4** | 0.0 | **0.4** |
| MMLU-Pro ★ | 20.5 | **24.0** | 3.4 | 17.8 | 5.7 | 9.8 | 20.2 | 1.7 | 13.5 | 10.1 | 5.6 |
| BBH fixed-option tasks ★ | **28.1** | 22.5 | 11.6 | 20.4 | 6.8 | 16.2 | 24.4 | 9.4 | 10.6 | 17.7 | 0.2 |
| WinoGrande ★ | **60.4** | 41.8 | 21.9 | 28.5 | 16.8 | 8.3 | 10.7 | 43.8 | 23.6 | 20.9 | 1.3 |

#### Language Understanding (models up to 1B)

| Benchmark | **Edge 0.4B** | vLLM-SR d3-lite | Crivo-150M | Sieve 0.8B | tasksource nano v4 | Saina Helm 2.0 | jiwo 0.8B v2 | DecisionTune 1.0 | vLLM-SR d3-edge | d1-omni-600M | GLiNER2.5-Decide |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ContractNLI | 69.6 | 33.8 | 74.7 | 62.6 | 63.7 | **78.2** | 69.0 | 72.2 | 37.1 | 51.1 | 0.0 |
| ANLI ★ | **42.6** | 24.1 | 19.0 | 15.0 | 18.7 | 4.7 | 20.8 | 0.0 | 11.3 | 4.1 | 0.0 |
| HellaSwag ★ | 77.1 | 76.9 | 48.9 | 67.5 | 52.4 | 25.9 | 65.8 | **78.5** | 45.3 | 46.6 | 9.7 |
| ACOS | 13.3 | 3.5 | **38.9** | 7.2 | 30.9 | 5.0 | 6.9 | 3.3 | 4.8 | 0.0 | 0.0 |
| FinEntity | **86.4** | 82.5 | 76.7 | 48.9 | 62.5 | 63.5 | 72.9 | 77.7 | 71.3 | 45.5 | 50.2 |
| iSarcasmEval | **37.1** | 13.7 | 0.0 | 0.0 | 0.0 | 21.0 | 4.6 | 0.0 | 12.2 | 6.7 | 13.0 |
| VAST | 59.6 | 33.1 | **61.4** | 5.7 | 59.9 | 15.5 | 13.5 | 4.0 | 17.9 | 8.2 | 2.9 |
| NLI4CT | **45.2** | 34.0 | 15.1 | 26.6 | 17.0 | 21.8 | 27.6 | 0.8 | 15.8 | 9.0 | 0.0 |
| RAGTruth response-level hallucination | **58.6** | 0.0 | 44.3 | 0.0 | 35.5 | 55.7 | 41.1 | 28.9 | 0.0 | 0.0 | 0.0 |

#### Retrieval & Classification (models up to 1B)

| Benchmark | **Edge 0.4B** | vLLM-SR d3-lite | Crivo-150M | Sieve 0.8B | tasksource nano v4 | Saina Helm 2.0 | jiwo 0.8B v2 | DecisionTune 1.0 | vLLM-SR d3-edge | d1-omni-600M | GLiNER2.5-Decide |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BANKING77 ★ | 90.7 | 90.2 | 90.6 | 85.5 | 88.8 | 84.4 | 60.2 | **92.1** | 89.8 | 83.5 | 65.2 |
| CLINC150+OOS ★ | 92.2 | **93.9** | 91.8 | 90.5 | 86.6 | 75.9 | 81.5 | 91.8 | 91.9 | 85.2 | 60.2 |
| BRIGHT ★ | 22.4 | 28.4 | 18.6 | 18.0 | 18.3 | 5.1 | **30.7** | 7.8 | 22.1 | 0.0 | 9.6 |
| Amazon ESCI | **39.3** | 26.4 | 29.8 | 26.3 | 22.6 | 32.1 | 36.5 | 37.3 | 11.6 | 8.8 | 3.8 |
| PhishNChips phishing decisions | **86.8** | 9.7 | 4.0 | 85.3 | 2.9 | 0.0 | 31.2 | 0.2 | 35.7 | 0.0 | 0.0 |
| HoVer claim verification | 67.6 | 63.5 | **69.4** | 17.1 | 67.9 | 58.3 | 17.2 | 29.6 | 62.1 | 19.6 | 1.4 |

#### Tools & Automation (models up to 1B)

| Benchmark | **Edge 0.4B** | vLLM-SR d3-lite | Crivo-150M | Sieve 0.8B | tasksource nano v4 | Saina Helm 2.0 | jiwo 0.8B v2 | DecisionTune 1.0 | vLLM-SR d3-edge | d1-omni-600M | GLiNER2.5-Decide |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BFCL ★ | **87.2** | 81.1 | 79.0 | 78.8 | 65.0 | 83.9 | 78.2 | 73.6 | 62.9 | 29.3 | 27.4 |
| ToolRet | 47.9 | 55.5 | **55.7** | 55.6 | 55.1 | 53.2 | 55.0 | 49.7 | 50.4 | 2.6 | 28.6 |
| API-Bank ★ | **82.5** | 56.1 | 29.4 | 54.0 | 25.6 | 74.7 | 17.5 | 45.2 | 39.0 | 38.8 | 5.9 |
| Home appliance simulator | 12.5 | 11.4 | **79.5** | 0.0 | 0.0 | 0.0 | 8.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| When2Call MCQ | 49.7 | **72.8** | 62.5 | 61.6 | 71.4 | 71.2 | 39.5 | 58.7 | 57.8 | 21.0 | 11.2 |

#### Arts & Human Taste (models up to 1B)

| Benchmark | **Edge 0.4B** | vLLM-SR d3-lite | Crivo-150M | Sieve 0.8B | tasksource nano v4 | Saina Helm 2.0 | jiwo 0.8B v2 | DecisionTune 1.0 | vLLM-SR d3-edge | d1-omni-600M | GLiNER2.5-Decide |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BPoMP | **82.4** | 43.2 | 0.0 | 12.1 | 18.9 | 0.0 | 0.7 | 18.4 | 8.9 | 0.0 | 14.0 |
| Humicroedit | **27.6** | 14.7 | 4.5 | 16.6 | 0.0 | 5.6 | 7.1 | 0.0 | 6.7 | 2.3 | 1.3 |
| POP909-CL | **23.1** | 0.9 | 1.2 | 0.6 | 0.8 | 0.0 | 0.0 | 0.6 | 0.4 | 0.1 | 0.9 |
| cfcolor | 12.1 | 12.8 | 0.6 | 13.5 | 4.0 | 15.2 | **24.1** | 0.0 | 5.0 | 9.3 | 6.4 |
| Habermas Machine | **33.8** | 9.9 | 10.1 | 14.4 | 3.3 | 0.0 | 10.4 | 8.9 | 4.3 | 7.5 | 20.8 |
| New Yorker caption matching | **52.6** | 51.9 | 44.1 | 24.0 | 50.0 | 16.9 | 31.1 | 6.2 | 45.8 | 20.0 | 15.2 |

</details>

### KGJevBench

Decision tasks from the GLiClass benchmark suite (KGJevBench): policy decisions, instruction following,
knowledge-graph walking, tool calling and routing, judging, and typed Jev-style decisions. GLiClass models were run
with the suite's `run_sample_500.sh` (up to 500 examples per task, bf16); the 12-relations row uses
`eval_kg_reasoning.py --mode stepwise --pad-relations 12`. The other columns are the KGJevBench comparison results for
those models on the same tasks and metrics; they were not re-run here. All values ×100; the best value in each row is bold.
Averages are recomputed from the cells shown, so the reference columns can differ by up to 0.3 from averages computed
on unrounded scores.

| Benchmark (metric) | **Edge 0.4B** | **9B** | **27B** | pplx-decider-27b | glide | Jev-1.1B | clef | d1 | clef-flash | gpt-6-luna | Kev-4B | DeBERTa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fast-decisions (domain avg) | 54.8 | 53.7 | 54.3 | 59.5 | 61.7 | 61.2 | 59.4 | **61.8** | 57.3 | 60.2 | 58.5 | 56.7 |
| task routing (top-1 primary) | **93.3** | 88.3 | 88.3 | 82.0 | 73.0 | 74.0 | 85.0 | 75.0 | 77.3 | 53.7 | 65.7 | 68.7 |
| tool calling (exact set) | **89.3** | 88.3 | 89.0 | 88.0 | 79.0 | 65.3 | 83.0 | 72.3 | 78.3 | 46.7 | 59.0 | 88.3 |
| instruction following (group acc) | 86.5 | 85.6 | 87.4 | 92.8 | 94.6 | 93.7 | **95.5** | 93.7 | 91.9 | 92.8 | 82.9 | 42.3 |
| KG step-by-step (path acc) | 98.3 | 97.7 | 96.3 | **98.7** | 91.0 | 95.7 | 92.7 | 97.0 | 91.0 | 90.7 | 86.7 | 67.0 |
| KG step-by-step, 12 relations/step | 93.0 | 94.7 | 93.3 | **95.3** | 82.0 | 85.7 | 87.7 | 87.0 | 85.3 | 79.7 | 73.7 | 51.7 |
| KG path ranking (Hits@1) | 98.7 | 98.7 | **100.0** | 95.0 | 94.0 | 87.7 | 87.0 | 81.3 | 56.0 | 92.0 | 67.7 | 50.0 |
| triple relevance (F1, keyword) | **90.6** | 84.7 | 83.2 | 31.8 | 33.0 | 30.7 | 28.4 | 28.4 | 21.9 | 21.2 | 47.3 | 69.5 |
| reranking (R@1) | 92.7 | 95.7 | 94.3 | 96.7 | 96.0 | 96.7 | 94.7 | 95.7 | 96.7 | **97.0** | 93.3 | 81.7 |
| NLI (accuracy, yes/no) | 89.2 | 96.9 | **100.0** | 90.7 | 96.0 | 98.8 | 92.0 | 94.1 | 92.6 | 89.5 | 93.2 | 85.5 |
| fact checking (acc, checkable) | 87.0 | 94.3 | 98.0 | **99.3** | 93.5 | 98.0 | 92.7 | 97.3 | 94.7 | 97.7 | 83.3 | 86.3 |
| moderation rules (acc, yes/no) | 85.6 | 81.2 | 92.4 | 93.3 | **94.0** | 92.3 | 89.0 | 82.0 | 85.7 | 85.0 | 77.3 | 78.3 |
| policy decisions (acc) | 64.0 | 94.0 | 96.3 | **97.3** | 97.0 | 96.7 | 95.7 | 95.3 | 84.0 | 89.7 | 81.3 | 72.0 |
| logical reasoning MCQA (acc; chance 25) | 43.7 | 70.0 | 75.7 | **90.0** | 83.0 | 81.3 | 76.0 | 79.3 | 71.3 | 65.0 | 63.0 | 38.0 |
| web actions (step acc) | 51.0 | 82.0 | **85.0** | 79.0 | 73.0 | 72.0 | 78.0 | 71.0 | 60.0 | 55.0 | 59.0 | 47.0 |
| LogiQA (acc; chance 25) | 42.4 | 59.0 | 65.6 | 72.0 | 65.0 | **77.1** | 63.9 | 69.4 | 60.2 | 61.1 | 50.8 | 31.5 |
| MedMCQA (acc; chance 25) | 34.8 | 60.2 | 67.2 | **74.6** | 67.0 | 74.5 | 69.1 | 72.5 | 66.5 | 72.3 | 54.0 | 28.0 |
| OpenBookQA closed (acc; chance 25) | 75.6 | 88.2 | 89.6 | **97.0** | 93.0 | 95.2 | 95.6 | 95.0 | 93.6 | 84.8 | 82.0 | 47.0 |
| CommonsenseQA (acc; chance 20) | 78.6 | 81.6 | 86.2 | 88.1 | 88.0 | 89.0 | 90.3 | 86.3 | **93.6** | 81.5 | 76.9 | 58.9 |
| JevBench public (acc) | 77.9 | 82.3 | 87.4 | 88.7 | 91.0 | 85.7 | 88.7 | 86.6 | 82.3 | **91.8** | 72.3 | 65.8 |
| JevBench HF gold questions (acc) | 86.8 | 86.0 | 92.4 | **92.5** | 88.0 | 91.3 | 92.4 | 90.6 | 89.3 | 91.0 | 89.6 | 88.4 |
| JevJudge text (skill ×100) | -0.1 | 17.5 | 32.2 | 43.9 | **52.4** | 42.2 | 37.8 | 38.5 | 23.5 | 40.1 | 24.5 | 11.2 |
| Open-Jev (agreement) | 59.6 | 75.1 | 82.4 | **89.6** | 85.9 | 78.3 | 78.3 | 81.0 | 74.1 | 78.1 | 71.2 | 57.8 |
| tasksource-Jev (agreement) | 78.4 | 78.6 | 78.8 | **83.9** | 75.0 | 73.6 | 73.3 | 75.0 | 68.9 | 66.9 | 65.1 | 53.8 |
| Jev Distill (agreement w/ teacher) | 51.5 | 54.8 | 61.0 | 63.7 | 57.9 | **72.7** | 62.0 | 65.1 | 62.3 | 67.0 | 57.4 | 42.7 |
| Behavior (present-class F1) | 37.6 | 56.7 | 68.3 | 68.9 | **87.5** | 67.0 | 66.0 | 67.4 | 52.6 | 57.4 | 46.3 | 27.7 |
| typed decisions (acc) | 49.3 | 64.2 | 68.5 | 73.9 | 71.8 | 74.1 | 71.4 | **74.2** | 70.2 | 69.3 | — | 51.6 |
| agent tool decisions (macro acc) | 57.1 | 78.5 | **87.6** | 85.4 | 81.8 | 82.9 | 73.9 | 83.4 | 64.1 | 75.2 | — | 40.3 |
| *Average, 26 rows all models have* | 70.8 | 78.7 | 82.5 | 82.8 | 80.4 | 79.9 | 79.0 | 78.4 | 73.5 | 73.4 | 68.5 | 57.5 |
| *Average, all 28 rows* | 69.5 | 78.2 | 82.2 | 82.6 | 80.2 | 79.8 | 78.6 | 78.4 | 73.0 | 73.3 | — | 56.7 |

Several KGJevBench sets are written in the formats GLiClass is trained on (for example triple relevance, task
routing, and tool calling), and tasksource-Jev overlaps the training data; read those rows with that in mind.

## Training and intended use

Training had three stages:

1. **Large-scale supervised training** on a mixture that recasts classification, NLI, QA, reranking,
   extraction, tool calling, moderation, judging, and reasoning data as decisions over candidate labels,
   with instructions, label descriptions where available. Single-label, multi-label, and scoring examples are mixed
   in one run with a focal loss.
2. **Reinforcement learning** with a PPO-style objective whose actions are the label decisions. Rewards
   are task metrics (F1, accuracy, recall), with a value baseline, a clipped ratio, and optional KL and entropy terms.
3. **Calibration**: with the backbone frozen, a small calibrator reads the text and label representations and logit statistics and predicts a per-pair temperature and bias, fitted by negative log-likelihood on held-out decision data.

Use this model for zero-shot and instruction-following classification, scoring, reranking, natural language
inference, knowledge graph and multi-step reasoning, guardrails, policy moderation, routing, and LLM-output
judging. It returns label probabilities and does not generate text.

## Limitations

- Label wording, instructions, domain, input length, and thresholds affect predictions.
- Knowledge-heavy and multi-step reasoning tasks are the main gap to the larger models: multi-step arithmetic (GSM8K), hard science and exam questions (GPQA, HLE, MMLU-Pro, MedMCQA), logic puzzles, and long agent-trace judging remain weak spots.
- Results are strongest in English; quality on other languages is lower.
- Calibrated scores are fitted on held-out data and can drift on domains far from training.
- Treat outputs used for safety, moderation, legal, or medical decisions as signals for review, not final verdicts.

## Citation

```bibtex
@misc{stepanov2026scxrouter,
  title  = {SCX Router: Streaming Zero-Shot Model Selection with a Decoder-KV Classifier and a Real-World Task Ontology},
  author = {Stepanov, Ihor and Smechov, Aleksandr and Shtopko, Mykhailo and Vodianytskyi, Dmytro and Lukashov, Oleksandr},
  year   = {2026},
  eprint = {2609.02292},
  archivePrefix = {arXiv}
}
```
