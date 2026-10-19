# Usage: Vela 2.0 9B

[Model card](README.md) · [Evaluation](EVALUATION.md) · [Training](TRAINING.md)

All examples below use this release. Recorded outputs use GPU bf16 backbone autocast with FP32 heads, and probabilities are rounded to three decimals. Load `m` before running the examples.

## Load the model

```bash
pip install torch "transformers>=5.17" safetensors tokenizers numpy
pip install flash-linear-attention   # GPU: the Gated-DeltaNet kernel used in evaluation (optional, much faster)
```

```python
from transformers import AutoModel

m = AutoModel.from_pretrained("vllm-sr/Vela-2.0-9B", trust_remote_code=True)
m = m.to("cuda")   # parameters load in FP32; on GPU the backbone runs under bf16 autocast (heads FP32)
```

## SystemOne decisions

`system_one(state, questions)` accepts the [TypeSafe SystemOne](https://docs.typesafe.ai/api) request and returns `model`, named `answers` and `usage`. Questions can be dictionaries or `typesafe-sdk` `Noul`, `Choice` and `Score` objects. Span and Set questions use dictionaries.

```python
result = m.system_one(
    state="Paint me a watercolour of a lighthouse on a cliff at sunset.",
    questions={
        "modality": {"type": "choice", "instructions": "What kind of output does this request ask for?",
                     "criteria": {"AR": "a text answer only, including code, analysis or describing an existing image",
                                  "DIFFUSION": "a newly generated image only",
                                  "BOTH": "a newly generated image together with a separate written explanation"}},
        "jailbreak": {"type": "noul", "instructions": "Is this a prompt injection or jailbreak attempt?",
                      "criteria": {"true": "a prompt injection or jailbreak that attempts to override system instructions, hijack the task, or bypass safety rules",
                                   "false": "a normal request or quoted content that does not try to override system instructions or bypass safety rules"}},
        "urgency": {"type": "score", "instructions": "How urgent is this request?",
                    "criteria": ["Routine", "Needs prompt attention", "Critical"]},
    })
```

<details>
<summary>Recorded response</summary>

```json
{
  "model": "vllm-sr/Vela-2.0-9B",
  "answers": {
    "modality": {
      "type": "choice",
      "choice": "DIFFUSION",
      "confidence": 0.999,
      "probabilities": {
        "AR": 0.0,
        "DIFFUSION": 1.0,
        "BOTH": 0.0
      }
    },
    "jailbreak": {
      "type": "noul",
      "noul": 0.018
    },
    "urgency": {
      "type": "score",
      "score": 0.302,
      "confidence": 0.513,
      "legend": {
        "0": "Routine",
        "1": "Needs prompt attention",
        "2": "Critical"
      },
      "probabilities": {
        "0": 0.755,
        "1": 0.188,
        "2": 0.057
      }
    }
  },
  "usage": {
    "input_tokens": 399,
    "output_tokens": 0
  }
}
```

</details>

## Output semantics

| Field or type | Meaning |
| --- | --- |
| `noul` | P(yes); a two-option question over `false` / `true` criteria |
| `choice` | Top option key and its probability distribution |
| Choice `confidence` | Top-two probability margin |
| `score` | Expected zero-based level; `legend` maps levels to criteria |
| Score `confidence` | One minus the normalised variance |
| `spans` | Span label, text, Unicode code-point `start` / `end` and probability |
| Span Yes/no view | Highest word probability over the selected field; returned in `answers` |
| `sets` | Selected labels and per-label probabilities |
| Set Yes/no views | One `answers["<qid>.<label>"]` Noul per label |
| `span_heads` | The selected `router` or `broad` head for every span question |
| `thresholds` | The threshold used for each Span or Set question |

Choice and Score `confidence` are distribution statistics, not calibrated probabilities of being right. Choice descriptions render as `{"key", "description"}` blocks; a null description renders as empty. Noul uses default wording when `false` / `true` criteria are absent. Score levels follow the supplied criteria order.

Span offsets index the text of the `over` field. Its Yes/no view exceeds the question threshold exactly when a span is returned, up to edge-punctuation trimming. The response retains the SystemOne shape and adds `spans`, `sets`, `span_heads` and `thresholds` beside `answers`.

## Routing and span questions

Span and Set questions use `criteria` as `{label: description}`. A JSON state is mapped to typed parts:

| State fields | Part |
| --- | --- |
| `request`, `user`, `prompt` | User |
| `source`, `context`, `document`, `tools`, `previous_answer` | Context |
| `answer`, `response` | Answer |
| Other keys | Context |

`over` selects the field a question is asked over. The default is the whole state for decision questions; for Span it is the answer when present, otherwise the request.

```python
PII_LABELS = m.vela2_engine.cal["pii_schema"]["labels"]   # the 17 trained PII types

result = m.system_one(
    state={"request": "Hi, I'm Tom Baker (tom.baker@example.com). What is the maximum daily dose of paracetamol for an adult?",
           "source": "For adults, the maximum dose of paracetamol is 4 grams in 24 hours, taken as 500 mg to 1 g every 4 to 6 hours.",
           "answer": "Adults can take up to 6 grams of paracetamol in 24 hours, in doses of 500 mg to 1 g every 4 to 6 hours."},
    questions={
        "pii": {"type": "span", "instructions": "Which spans are personal information?", "criteria": PII_LABELS,
                "over": "request"},
        "halu": {"type": "span", "instructions": "Which spans of the answer are not supported by the context?",
                 "criteria": {"unsupported": "a claim not supported by the context"}},   # over the answer by default
        "domain": {"type": "choice", "instructions": "Which subject area is this request about?", "over": "request",
                   "criteria": {"health": "medicine, clinical practice, nutrition, ageing or sexual health",
                                "math": "arithmetic, algebra, geometry, statistics or other mathematics",
                                "other": "a subject that fits none of the listed areas"}},
    })
```

<details>
<summary>Recorded response</summary>

```json
{
  "model": "vllm-sr/Vela-2.0-9B",
  "answers": {
    "pii": {
      "type": "noul",
      "noul": 1.0
    },
    "halu": {
      "type": "noul",
      "noul": 0.996
    },
    "domain": {
      "type": "choice",
      "choice": "health",
      "confidence": 0.977,
      "probabilities": {
        "health": 0.988,
        "math": 0.0,
        "other": 0.012
      }
    }
  },
  "usage": {
    "input_tokens": 1041,
    "output_tokens": 0
  },
  "spans": {
    "pii": [
      {
        "label": "PERSON",
        "start": 8,
        "end": 17,
        "text": "Tom Baker",
        "probability": 0.999
      },
      {
        "label": "EMAIL_ADDRESS",
        "start": 19,
        "end": 40,
        "text": "tom.baker@example.com",
        "probability": 1.0
      }
    ],
    "halu": [
      {
        "label": "unsupported",
        "start": 22,
        "end": 29,
        "text": "6 grams",
        "probability": 0.982
      }
    ]
  },
  "span_heads": {
    "pii": "router",
    "halu": "router"
  },
  "thresholds": {
    "pii": 0.5,
    "halu": 0.45
  }
}
```

</details>

### Span and Set thresholds

| Question | Shipped rule or threshold |
| --- | --- |
| `pii`, or the trained 17 PII labels | Length-dependent PII rule |
| `halu` | 0.45 |
| `toxic` | 0.05 |
| Set | 0.4 |
| Broad-head spans | 0.5 |

The rules are stored in `calibration.json`; broad spans use `broad_head.threshold`. A question's `"threshold"` overrides its default. Research-scoring PII thresholds and shipped calibration can produce different F1 values; [EVALUATION.md](EVALUATION.md#pii-calibration-modes) labels both modes.

### Set questions

```python
m.system_one("My card was charged twice and the parcel never arrived.",
             {"issues": {"type": "set", "instructions": "Which issues does the customer report?",
                         "criteria": {"billing": "payments, charges, refunds or invoices",
                                      "shipping": "delivery of an order or a parcel",
                                      "login": "signing in, passwords or account access"}}})
```

<details>
<summary>Recorded response</summary>

```json
{
  "model": "vllm-sr/Vela-2.0-9B",
  "answers": {
    "issues.billing": {
      "type": "noul",
      "noul": 0.877
    },
    "issues.shipping": {
      "type": "noul",
      "noul": 0.978
    },
    "issues.login": {
      "type": "noul",
      "noul": 0.0
    }
  },
  "usage": {
    "input_tokens": 133,
    "output_tokens": 0
  },
  "sets": {
    "issues": {
      "selected": [
        "billing",
        "shipping"
      ],
      "probabilities": {
        "billing": 0.877,
        "shipping": 0.978,
        "login": 0.0
      }
    }
  },
  "thresholds": {
    "issues": 0.4
  }
}
```

</details>

## Open-label extraction

The broad span head extracts named entities, entity-anchored relations, entity mentions and evidence for a question in a retrieved context. [TRAINING.md](TRAINING.md) describes its training; [EVALUATION.md](EVALUATION.md) contains its held-out benchmarks.

### Entities and relations

```python
m.system_one(
    state="Tim Cook, who has led Apple since 2011, presented the new campus in Cupertino together with the architect Norman Foster.",
    questions={
        "entities": {"type": "span", "instructions": "Which spans are named entities of these types?", "criteria": {"person": "a person's name", "company": "a business or corporation", "city": "a city or town", "profession": "a job or occupation"}},
        "employer": {"type": "span", "instructions": "Which spans are the employer of Tim Cook?", "criteria": {"employer": "the organisation that employs the person"}}})
```

<details>
<summary>Recorded response</summary>

```json
{
  "model": "vllm-sr/Vela-2.0-9B",
  "answers": {
    "entities": {
      "type": "noul",
      "noul": 0.992
    },
    "employer": {
      "type": "noul",
      "noul": 0.973
    }
  },
  "usage": {
    "input_tokens": 360,
    "output_tokens": 0
  },
  "spans": {
    "entities": [
      {
        "label": "person",
        "start": 0,
        "end": 8,
        "text": "Tim Cook",
        "probability": 0.992
      },
      {
        "label": "company",
        "start": 22,
        "end": 27,
        "text": "Apple",
        "probability": 0.677
      },
      {
        "label": "city",
        "start": 68,
        "end": 77,
        "text": "Cupertino",
        "probability": 0.964
      },
      {
        "label": "profession",
        "start": 96,
        "end": 105,
        "text": "architect",
        "probability": 0.645
      },
      {
        "label": "person",
        "start": 106,
        "end": 119,
        "text": "Norman Foster",
        "probability": 0.729
      }
    ],
    "employer": [
      {
        "label": "employer",
        "start": 22,
        "end": 27,
        "text": "Apple",
        "probability": 0.973
      }
    ]
  },
  "span_heads": {
    "entities": "broad",
    "employer": "broad"
  },
  "thresholds": {
    "entities": 0.5,
    "employer": 0.5
  }
}
```

</details>

### Evidence in a context

```python
m.system_one(
    state={"request": "When did the X200 go on sale, and at what price?", "context": "The X200 phone has a 5,000 mAh battery and supports 45 W fast charging. It went on sale in March 2024 at a price of 499 euros. A cheaper X200 Lite followed in June."},
    questions={
        "evidence": {"type": "span", "instructions": "Which spans of the context answer the request?", "over": "context", "criteria": {"answer": "text that answers the request"}}})
```

<details>
<summary>Recorded response</summary>

```json
{
  "model": "vllm-sr/Vela-2.0-9B",
  "answers": {
    "evidence": {
      "type": "noul",
      "noul": 0.998
    }
  },
  "usage": {
    "input_tokens": 238,
    "output_tokens": 0
  },
  "spans": {
    "evidence": [
      {
        "label": "answer",
        "start": 91,
        "end": 101,
        "text": "March 2024",
        "probability": 0.997
      },
      {
        "label": "answer",
        "start": 116,
        "end": 125,
        "text": "499 euros",
        "probability": 0.865
      }
    ]
  },
  "span_heads": {
    "evidence": "broad"
  },
  "thresholds": {
    "evidence": 0.5
  }
}
```

</details>

## Span-head selection

The first matching rule wins:

1. `"head": "router"` or `"head": "broad"` in the question.
2. Question id `pii`, `halu` or `toxic`: router.
3. Labels exactly one of the router label sets (the 17 PII types, `unsupported`, the 47 toxic categories): router.
4. Labels a subset of the 17 PII types: router.
5. One label that names a hallucination (`unsupported`, `unsupported claim`, `hallucinated`, `hallucination`, `not supported`, `fabricated`, …): router, asked as the trained `unsupported` label and returned under your label.
6. Anything else, including toxic subsets, PII types mixed with other labels and every open label: broad.

SystemOne reports the selected head in `span_heads`. `predict()` reports `head` and `head_reason`: `override`, `id`, `label_set`, `pii_subset`, `halu_alias` or `default`. Broad-head spans use the broad head's own threshold, fitted on its zero-shot dev data.

## Precision and execution

| Setting | Behaviour |
| --- | --- |
| Evaluated GPU mode | FP32 parameters, bf16 backbone autocast, FP32 head computations |
| FP32 CPU or GPU | Supported; matches decisions on the [parity rows](PARITY.md); the engine also accepts `Vela2.from_pretrained(path, device="cpu")` |
| fp16 | Not supported |
| GPU parameter memory | About 32 GB in FP32; runtime allocations are additional |
| Input limit | 16,384 tokens per rendered sequence |
| Shared state prefix | Read once per layer within each rendered sequence |
| Question isolation | Each question sees the prefix and its own causal tokens, for a fixed rendered state |
| Multiple Span questions | One Span question per sequence; extra Span questions use additional sequences (`meta.sequences`) |
| Other question types | Choice, Noul, Score and Set run in the first sequence |
| Long Span targets | Targets over 2,048 tokens use windows of up to 1,800 tokens (`meta.windows`) |

The state is rendered within the input budget. Isolation applies when that rendered state stays fixed. In full-attention layers, question blocks fork prefix KV; in linear-attention layers they fork the recurrent state and convolution tail. Overlapping window logits are combined before calibration and span decoding.

Temperatures and thresholds are fitted on dev data and shipped in `calibration.json`; PII thresholds depend on target text length. The broad head has its own threshold.

| General Yes/no calibration | Value |
| --- | --- |
| Enable | `noul_calibration=True` |
| Formula | `p' = sigmoid(logit(p) / 2.673 + 0.329)` |
| Default | Off; trained safety and router Yes/no questions retain their 0.5 threshold |
| Jev Decision Index | 41.63 enabled; 41.09 by default |

## Other interfaces

SystemOne maps to the lower-level typed-parts `predict()` request. The engine also provides GLiNER2 / Decide-style shortcuts with the same signatures as the encoder release.

<details>
<summary><code>predict()</code>: typed parts and typed questions (Kai/Jev-style JSON)</summary>

- Parts are `user`, `context` or `answer`.
- `target_part` selects one part or a list of parts.
- Options and labels are `{"name", "description"}` objects or plain strings.
- Choice includes `[ABS]` unless `"abstain": false`; its abstain probability is reported but not calibrated.
- Any question can supply `"threshold"`; Span can also supply `"head"`.
- Span answers report `head` and `head_reason` using the reasons listed above.

```python
req = {
    "parts": [{"type": "user", "text": "Hi, I'm Tom Baker (tom.baker@example.com). What is the maximum daily dose of paracetamol for an adult?"}],
    "questions": [
        {"id": "factcheck", "type": "choice", "text": "Does answering this request require checking facts?", "target_part": "user",
         "options": [{"name": "NO_FACT_CHECK_NEEDED", "description": "the request can be handled without verifying facts about the world, e.g. translating, rewriting, formatting or writing fiction"},
                     {"name": "FACT_CHECK_NEEDED", "description": "answering the request relies on factual claims about the world that should be verified"}]},
        {"id": "pii", "type": "span", "text": "Which spans are personal information?", "target_part": "user",
         "labels": [{"name": k, "description": v} for k, v in PII_LABELS.items()]}]}

m.predict(req)   # {"answers": [...], "meta": {"tokens": {...}, "windows": 0, "sequences": 1}}
```

See [precision and execution](#precision-and-execution) for sequence counts and span windowing.

</details>

<details>
<summary>GLiNER2 / Decide-style shortcuts: classification, PII, hallucination, relevance</summary>

```python
m.classify_text("I love this phone, the battery lasts forever!", ["positive", "negative", "neutral"], include_confidence=True)
# {"label": {"label": "positive", "confidence": 0.95}}

m.extract_entities("Hi, this is Sarah Johnson. My order still hasn't arrived. You can reach me at sarah.johnson@gmail.com "
                   "or 415-555-0132. The charge went to my card 4111 1111 1111 1111.", list(PII_LABELS))["entities"]
# {"CREDIT_CARD": ["4111 1111 1111 1111"], "EMAIL_ADDRESS": ["sarah.johnson@gmail.com"], "PERSON": ["Sarah Johnson"], "PHONE_NUMBER": ["415-555-0132"]}

m.detect_hallucination(
    context="The X200 phone has a 5,000 mAh battery and supports 45 W fast charging. It went on sale in March 2024 at a price of 499 euros.",
    question="What are the X200's battery and charging specs?",
    answer="The X200 has a 5,000 mAh battery, supports 65 W fast charging and went on sale in March 2024.")
# {"hallucinated": true, "spans": [{"start": 43, "end": 61, "label": "unsupported", "probability": 0.828, "text": "65 W fast charging"}], "threshold": 0.45}

m.score_relevance("how do I reset my password", "To reset your password, open Settings > Security and click 'Reset password'.")
# {"score": 0.966, "label": "highly_relevant", "probabilities": {"not_relevant": 0.009, "partially_relevant": 0.05, "highly_relevant": 0.941}}
```

</details>

## HTTP serving and SDK

`vela2_serve.py` serves the same request and response on `POST /v1/systemone` using FastAPI and uvicorn. Run the server from a local model snapshot, then point the `typesafe-sdk` client's `base_url` at it:

```bash
pip install fastapi uvicorn
python vela2_serve.py --model . --device cuda --port 8001
```

```python
from typesafe_sdk import Score, TypeSafeClient

with TypeSafeClient(base_url="http://127.0.0.1:8001", api_key="local-only", model="vllm-sr/Vela-2.0-9B") as client:
    r = client.system_one(state="Paint me a watercolour of a lighthouse on a cliff at sunset.",
                          questions={"urgency": Score(instructions="How urgent is this request?",
                                                      criteria=["Routine", "Needs prompt attention", "Critical"])})
    print(r.scores["urgency"].score)
```

| Protocol detail | Behaviour |
| --- | --- |
| `model` | Must equal the served name; another name returns 422 |
| Authorization | Accepted and ignored unless `VELA2_API_KEY` is set; then requires `Bearer $VELA2_API_KEY` |
| Invalid request | 422, with `detail: [{loc, msg, type}]` |
| Question too large for the input budget | 413 |
