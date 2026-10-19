---
license: apache-2.0
pipeline_tag: zero-shot-classification
base_model:
- vllm-sr/Decision-1.0-Kai-0.6B
- vllm-sr/Vela-1.0-Encoder-307M
base_model_relation: finetune
datasets:
- nvidia/Aegis-AI-Content-Safety-Dataset-2.0
- ToxicityPrompts/PolyGuardMix
- nvidia/Nemotron-Safety-Guard-Dataset-v3
- microsoft/llmail-inject-challenge
- OpenSafetyLab/Salad-Data
- KRLabsOrg/lettucedetect-prose-hallucination
- KRLabsOrg/lettucedetect-code-hallucination
language:
- ar
- zh
- cs
- nl
- en
- fr
- de
- hi
- it
- ja
- ko
- pl
- pt
- ru
- es
- sv
- th
tags:
- decision-model
- system-one
- modernbert
- semantic-router
- vela
- zero-shot-classification
- token-classification
- pii-detection
- hallucination-detection
- content-safety
- onnx
- multilingual
- long-context
---

<div align="center">
  <img src="https://huggingface.co/vllm-sr/Vela-2.0-0.3B/resolve/main/assets/vela2-banner.jpg" alt="Vela 2.0: Towards Open Foundation Routing Models. vLLM Semantic Router x KR Labs" width="100%" />
  <p>
    <a href="https://vllm-sr.ai/"><strong>Docs</strong></a> |
    <a href="https://vllm-sr.ai/blog/vela-2-0-open-foundation-routing-models"><strong>Blog</strong></a> |
    <a href="https://github.com/vllm-project/semantic-router"><strong>GitHub</strong></a> |
    <a href="https://huggingface.co/collections/vllm-sr/vela-20"><strong>Vela 2.0 collection</strong></a>
  </p>
</div>

# Vela 2.0 0.3B

**Open Foundation Routing Models**

*Routing decisions. Safety checks. Precise text spans.*

Vela 2.0 brings routing questions and span-level decisions into one model interface. Supply your options, labels and rubrics at request time; ask about the request, its context and the answer together.

The 0.3B is the family's compact multilingual encoder, with an 8,192-token input budget and Torch or ONNX inference on CPU or GPU.

1. **Route with your own criteria.** Choose a route, check a policy condition, assign a score or select multiple labels.
2. **Locate the text behind a signal.** Return spans of personal information and unsupported claims with character offsets and probabilities.
3. **Ask across the whole interaction.** Use typed request, context and answer parts in one request, through Python or a SystemOne-compatible HTTP API.

| Model specification | Vela 2.0 0.3B |
| --- | --- |
| Backbone | 22-layer bidirectional ModernBERT encoder |
| Hidden width | 768 |
| Input budget | 8,192 tokens, including schema and state |
| Questions | Choice, Yes/no (Noul), Score, Span, Set |
| Inference | Torch fp32; ONNX fp32 or fp16 encoder with fp32 heads |
| Deployment | CPU or GPU; Python API and HTTP server |

## Quickstart

Ask for personal information, unsupported answer text and a route in one request:

For focused examples, see [Span questions](#span-questions-locate-text) and [Set questions](#set-questions-select-multiple-labels) below.

```bash
pip install torch "transformers>=4.57" safetensors tokenizers numpy
```

```python
from transformers import AutoModel

m = AutoModel.from_pretrained("vllm-sr/Vela-2.0-0.3B", trust_remote_code=True)
# Runs on CPU by default; optionally use m = m.to("cuda"). Keep Torch loading in fp32.
PII_LABELS = m.vela2_engine.cal["pii_schema"]["labels"]

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

Selected fields from the recorded Torch CPU response, with probabilities rounded to three decimals:

```json
{
  "answers": {
    "pii": {"type": "noul", "noul": 1.0},
    "halu": {"type": "noul", "noul": 0.984},
    "domain": {"type": "choice", "choice": "health", "confidence": 0.51, "probabilities": {"health": 0.727, "math": 0.217, "other": 0.056}}
  },
  "spans": {
    "pii": [{"label": "PERSON", "start": 8, "end": 17, "text": "Tom Baker", "probability": 1.0}, {"label": "EMAIL_ADDRESS", "start": 19, "end": 40, "text": "tom.baker@example.com", "probability": 0.983}],
    "halu": [{"label": "unsupported", "start": 16, "end": 29, "text": "up to 6 grams", "probability": 0.895}]
  }
}
```

Offsets are Unicode code-point offsets into the targeted field. [Full response, HTTP, SDK and ONNX examples](USAGE.md).

### Span questions: locate text

Use `"type": "span"` to extract text with labels and character offsets. Set `criteria` to a label-to-description mapping and `over` to the state field to inspect. This PII example reuses `m` and the trained `PII_LABELS` from above:

```python
span_result = m.system_one(
    state={"request": "Hi, I'm Tom Baker (tom.baker@example.com)."},
    questions={
        "pii": {
            "type": "span",
            "instructions": "Which spans are personal information?",
            "criteria": PII_LABELS,
            "over": "request",
        }
    },
)
print(span_result["spans"]["pii"])
```

Each returned span includes `label`, `text`, `start`, `end` and `probability`. Offsets use Unicode code points, with `start` inclusive and `end` exclusive, in the selected field. The first example also shows hallucination detection: use the `unsupported` label, target `over="answer"`, and supply grounding text in `state["source"]`.

### Set questions: select multiple labels

Use `"type": "set"` when zero, one or several labels can apply. Supply your labels and descriptions in `criteria`; each label is scored independently. This example reuses `m` from above:

```python
set_result = m.system_one(
    state="My card was charged twice and the parcel never arrived.",
    questions={
        "issues": {
            "type": "set",
            "instructions": "Which issues does the customer report?",
            "criteria": {
                "billing": "payments, charges, refunds or invoices",
                "shipping": "delivery of an order or a parcel",
                "login": "signing in, passwords or account access",
            },
        }
    },
)
print(set_result["sets"]["issues"]["selected"])
print(set_result["sets"]["issues"]["probabilities"])
```

Read the selected labels and all per-label probabilities from `sets["issues"]`. Probabilities do not need to sum to one. Labels are selected when their probability exceeds the question's threshold; omit `threshold` to use shipped calibration, or add it to the question to set your own cutoff. The applied value is returned in `set_result["thresholds"]["issues"]`.

## Questions and outputs

| Question | Use it for | Output |
| --- | --- | --- |
| **Choice** | Route a request or pick one of 2–255 supplied options. | Selected key and distribution |
| **Yes/no (Noul)** | Check a condition against the state. | P(yes) |
| **Score** | Rate against 2–10 ordered levels. | Expected level and distribution |
| **Span** | Locate personal information or claims unsupported by the source. | Labels, text, offsets and probabilities |
| **Set** | Select any number of supplied labels. | Selected labels and per-label probabilities |

All five types share one interface. Span and Set preserve SystemOne compatibility through additional response fields and Noul views. [Request and response reference](USAGE.md#span-and-set-questions).

## Results

<p align="center"><img src="https://huggingface.co/vllm-sr/Vela-2.0-0.3B/resolve/main/assets/vela2-family.jpg" alt="Four sizes, one family: 0.3B, 0.8B, 4B, 9B" width="80%" /></p>

| Task | Vela 2.0 0.3B | Reference |
| --- | ---: | --- |
| Router safety, macro AUC over 14 sets | **0.871** | GLiNER2.5-Decide 0.704 |
| Prompt attacks, unseen families, AUC | **0.882** | Vela 1.0 Guard 0.792 |
| Short PII, exact micro-F1 | **0.995** | Vela 1.0 PII 0.976 |

- **Selection:** checkpoint selection used dev splits; final release selection also considered test results.
- **Safety:** these task families were part of Vela's training, unlike Decide's. This comparison measures router safety performance, rather than zero-shot generalisation.

Selected results above use identical rows for each task comparison. [Full results and evaluation protocol](EVALUATION.md) include every benchmark, confidence intervals and the separate shipped, test-blind and research PII calibration paths.

## Architecture

<p align="center"><img src="https://huggingface.co/vllm-sr/Vela-2.0-0.3B/resolve/main/assets/architecture/01-vela-2.0-0.3b-architecture.png" alt="Vela 2.0 0.3B model architecture" width="720" /></p>

A 22-layer bidirectional ModernBERT encoder with schema-conditioned decision and word–label span readouts. The first attention pre-norm is Identity; this diagram shows the SDPA path. [Editable SVG](assets/architecture/01-vela-2.0-0.3b-architecture.svg).

<details>
<summary>Operator and readout diagrams</summary>

**Attention and GEGLU**

<p align="center"><img src="https://huggingface.co/vllm-sr/Vela-2.0-0.3B/resolve/main/assets/architecture/05-encoder-attention-geglu.png" alt="Vela 2.0 0.3B: QKV, YaRN attention and GEGLU operators" width="100%" /></p>

The SDPA attention path uses 12 heads of width 64 and applies ×4 YaRN RoPE to Q/K. GEGLU has two 1,152-wide branches. [Editable SVG](assets/architecture/05-encoder-attention-geglu.svg).

**Decision and span readouts**

<p align="center"><img src="https://huggingface.co/vllm-sr/Vela-2.0-0.3B/resolve/main/assets/architecture/06-encoder-readouts.png" alt="Vela 2.0 0.3B: option cosine plus MLP and word-label cosine readouts" width="100%" /></p>

The decision and span readouts use 256-dimensional projections and shared readout LayerNorm parameters. Learned τ and inference calibration temperatures T are distinct. [Editable SVG](assets/architecture/06-encoder-readouts.svg).

</details>

The backbone reads schema and typed state bidirectionally. Decision heads combine option cosine scores with an option MLP; the span head compares word and label states. [Model, calibration and training details](TRAINING.md).

## Reference

- [Usage](USAGE.md): Python, HTTP, SDK, Span/Set, typed parts, shortcuts and ONNX.
- [Evaluation](EVALUATION.md): full tables, scoring paths, references and selection disclosures.
- [Training and model details](TRAINING.md): data, training recipe, input schema, calibration and repository files.
- [Export parity](PARITY.md): Torch/research-scorer and ONNX comparisons on the measured rows.

## Credit and citation

Vela 2.0 is led by KR Labs and [vLLM Semantic Router](https://github.com/vllm-project/semantic-router).

Read the [Vela 2.0 technical overview](https://vllm-sr.ai/blog/vela-2-0-open-foundation-routing-models/).

```bibtex
@misc{vela2_unified_2026,
  title        = {Vela 2.0: Towards Open Foundation Routing Models},
  author       = {{KR Labs} and {vLLM Semantic Router}},
  year         = {2026},
  note         = {Blog post. Model: vllm-sr/Vela-2.0-0.3B},
  howpublished = {\url{https://vllm-sr.ai/blog/vela-2-0-open-foundation-routing-models/}}
}
```

## Licence

Apache-2.0 for this model's weights, code and documentation. It is derived from [Decision-1.0-Kai-0.6B](https://huggingface.co/vllm-sr/Decision-1.0-Kai-0.6B) (Apache-2.0) and [Vela-1.0-Encoder-307M](https://huggingface.co/vllm-sr/Vela-1.0-Encoder-307M) (MIT, from mmBERT); their licences and notices are passed on in `LICENSE`, `NOTICE`, `NOTICE_SOURCES.json` and `LICENSES/`. The tokenizer carries the Gemma Terms of Use: [DISTRIBUTION_TERMS.md](DISTRIBUTION_TERMS.md) · [License scope](LICENSING_STATUS.md). Changes against Kai and Vela, including the eight added tokenizer entries: [MODIFICATIONS.md](MODIFICATIONS.md). Training data keep their own licences, some of them CC-BY-SA share-alike (see `NOTICE_SOURCES.json`).

[Explore the Vela model collection](https://huggingface.co/collections/vllm-sr/vela-20)
