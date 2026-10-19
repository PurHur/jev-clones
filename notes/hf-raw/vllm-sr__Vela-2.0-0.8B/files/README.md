---
license: apache-2.0
pipeline_tag: zero-shot-classification
base_model:
- vllm-sr/Decision-2.0-Eos-0.8B
base_model_relation: finetune
datasets:
- nvidia/Aegis-AI-Content-Safety-Dataset-2.0
- ToxicityPrompts/PolyGuardMix
- nvidia/Nemotron-Safety-Guard-Dataset-v3
- microsoft/llmail-inject-challenge
- OpenSafetyLab/Salad-Data
- KRLabsOrg/lettucedetect-prose-hallucination
- KRLabsOrg/lettucedetect-code-hallucination
- numind/NuNER
- urchade/pile-mistral-v0.1
- knowledgator/GLINER-multi-task-synthetic-data
- nlpaueb/finer-139
- tonytan48/Re-DocRED
- MultiCoNER/multiconer_v2
- martinjosifoski/SynthIE
- rajpurkar/squad_v2
- hotpotqa/hotpot_qa
- google-research-datasets/natural_questions
- KRLabsOrg/verbatim-spans
- KRLabsOrg/tool-output-extraction-swebench
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
- qwen3.5
- semantic-router
- vela
- zero-shot-classification
- token-classification
- named-entity-recognition
- pii-detection
- hallucination-detection
- content-safety
- multilingual
- long-context
---

<div align="center">
  <img src="https://huggingface.co/vllm-sr/Vela-2.0-0.8B/resolve/main/assets/vela2-banner.jpg" alt="Vela 2.0: Towards Open Foundation Routing Models. vLLM Semantic Router x KR Labs" width="100%" />
  <p>
    <a href="https://vllm-sr.ai/"><strong>Docs</strong></a> |
    <a href="https://vllm-sr.ai/blog/vela-2-0-open-foundation-routing-models"><strong>Blog</strong></a> |
    <a href="https://github.com/vllm-project/semantic-router"><strong>GitHub</strong></a> |
    <a href="https://huggingface.co/collections/vllm-sr/vela-20"><strong>Vela 2.0 collection</strong></a>
  </p>
</div>

# Vela 2.0 0.8B

**Open Foundation Routing Models**

*Routing decisions. Safety checks. Precise text spans.*

The compact hybrid member of Vela 2.0: a 756M-parameter model for multilingual routing, safety checks and span-level decisions through one interface.

Define options, labels and rubrics at request time. Ask multiple named questions about a request, context and answer, and receive structured decisions with the text spans that support your workflow.

1. **Compact deployment.** About 3 GB of GPU memory for FP32 parameters, with a 16,384-token input limit.
2. **Routing and safety together.** Use one request for routing, prompt-attack checks, PII and unsupported-claim detection.
3. **Decisions at span resolution.** Return precise text locations for trained router labels and open-label extraction, with probabilities and character offsets.

| Specification | Value |
| --- | --- |
| Reported parameters | 756M |
| Backbone | 24-layer Qwen3.5 hybrid: Gated DeltaNet and gated GQA |
| Input limit | 16,384 tokens |
| Output | Choice, Yes/no (Noul), Score, Span and Set; no text generation |
| Evaluated precision | FP32 parameters; bf16 backbone autocast on GPU; FP32 heads |
| GPU parameter memory | About 3 GB in FP32 |
| Licence | Apache-2.0 |

## Quickstart

Install the dependencies, load the model and ask for routing and spans in one request:

For focused examples, see [Span questions](#span-questions-locate-text) and [Set questions](#set-questions-select-multiple-labels) below.

```bash
pip install torch "transformers>=5.17" safetensors tokenizers numpy
pip install flash-linear-attention   # GPU: the Gated-DeltaNet kernel used in evaluation (optional, much faster)
```

```python
from transformers import AutoModel

m = AutoModel.from_pretrained("vllm-sr/Vela-2.0-0.8B", trust_remote_code=True)
m = m.to("cuda")   # parameters load in FP32; on GPU the backbone runs under bf16 autocast (heads FP32)

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

Recorded response excerpt from this release; the [full response](USAGE.md#routing-and-span-questions) includes usage and thresholds:

```json
{
  "answers": {
    "domain": {
      "type": "choice",
      "choice": "health",
      "confidence": 0.929,
      "probabilities": {
        "health": 0.962,
        "math": 0.005,
        "other": 0.033
      }
    }
  },
  "spans": {
    "pii": [
      {
        "label": "PERSON",
        "start": 8,
        "end": 17,
        "text": "Tom Baker",
        "probability": 0.984
      },
      {
        "label": "EMAIL_ADDRESS",
        "start": 19,
        "end": 40,
        "text": "tom.baker@example.com",
        "probability": 0.985
      }
    ],
    "halu": [
      {
        "label": "unsupported",
        "start": 19,
        "end": 29,
        "text": "to 6 grams",
        "probability": 0.778
      }
    ]
  },
  "span_heads": {
    "pii": "router",
    "halu": "router"
  }
}
```

GPU with bf16 autocast is the evaluated setting. FP32 on CPU or GPU is also supported; fp16 is not. Parameter memory above excludes runtime allocations. See [precision and execution](USAGE.md#precision-and-execution).

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

**Two span heads, one interface.** The router span head handles trained PII, hallucination and toxic labels. The broad span head handles open-label extraction. The engine selects a head for each span question; you can also set `"head": "router"` or `"head": "broad"`. See [span-head selection](USAGE.md#span-head-selection). For entities, relations and evidence examples, see [open-label extraction](USAGE.md#open-label-extraction).

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
| **Choice** | Route a request or select one of 2–255 supplied options. | Selected key and distribution |
| **Yes/no (Noul)** | Check a condition against the state. | P(yes) |
| **Score** | Rate against 2–10 ordered levels. | Expected level and distribution |
| **Span** | Locate personal information, unsupported claims, entities, relations or evidence. | Text spans, character offsets and probabilities |
| **Set** | Select any number of supplied labels. | Selected labels and per-label probabilities |

Span offsets refer to Unicode code points in the selected text. Span and Set answers also provide Yes/no views in the SystemOne response. [Output semantics and thresholds](USAGE.md#output-semantics) explain these fields.

## Results

Selected results for this checkpoint. [Full evaluation](EVALUATION.md) includes every benchmark, family comparison and scoring protocol.

| Benchmark | Result | Metric and mode |
| --- | ---: | --- |
| Safety, 14 public sets | **0.875** | Mean AUC; trained task families |
| Prompt attacks, unseen families | **0.940** | AUC |
| PII, short texts | **0.984** | F1; shipped calibration |
| ACL-Verbatim evidence | **23.6** | Word-F1; broad span head, zero-shot |

- **Evaluation:** checkpoint selection, temperatures and thresholds used dev data. Safety results cover trained task families; they are not zero-shot comparisons.
- **Scoring:** PII uses shipped calibration; ACL-Verbatim was held out of training. [Benchmark protocols](EVALUATION.md#open-extraction-benchmarks).

## Architecture

<p align="center"><img src="https://huggingface.co/vllm-sr/Vela-2.0-0.8B/resolve/main/assets/architecture/02-vela-2.0-0.8b-architecture.png" alt="Vela 2.0 0.8B model architecture" width="720" /></p>

This model has its own 24-layer hybrid backbone, with Gated DeltaNet and gated GQA blocks, a candidate head, and separate router/broad span-head weights. [Editable SVG](assets/architecture/02-vela-2.0-0.8b-architecture.svg).

<details>
<summary>Operator and readout diagrams</summary>

**Gated DeltaNet**

<p align="center"><img src="https://huggingface.co/vllm-sr/Vela-2.0-0.8B/resolve/main/assets/architecture/07-gated-deltanet.png" alt="Vela 2.0 hybrid backbone: Gated DeltaNet operator" width="100%" /></p>

Causal QKV convolution feeds the gated-delta update, followed by per-head normalization and output gating. [Editable SVG](assets/architecture/07-gated-deltanet.svg).

**Gated GQA and SwiGLU**

<p align="center"><img src="https://huggingface.co/vllm-sr/Vela-2.0-0.8B/resolve/main/assets/architecture/08-gated-gqa-swiglu.png" alt="Vela 2.0 hybrid backbone: gated GQA and SwiGLU operators" width="100%" /></p>

Gated GQA uses partial RoPE on Q/K and a sigmoid output gate; SwiGLU uses separate up and gate projections. [Editable SVG](assets/architecture/08-gated-gqa-swiglu.svg).

**Candidate readout**

<p align="center"><img src="https://huggingface.co/vllm-sr/Vela-2.0-0.8B/resolve/main/assets/architecture/09-decoder-candidate-head.png" alt="Vela 2.0 hybrid models: CandidateHead bilinear and MLP scoring" width="100%" /></p>

CandidateHead combines scaled bilinear and additive MLP scores over runtime options, then applies task-specific calibration. [Editable SVG](assets/architecture/09-decoder-candidate-head.svg).

**Router and broad span heads**

<p align="center"><img src="https://huggingface.co/vllm-sr/Vela-2.0-0.8B/resolve/main/assets/architecture/10-decoder-span-heads.png" alt="Vela 2.0 hybrid models: word-label SpanHeadV2 and router/broad selection" width="100%" /></p>

Router and broad span heads have separate weights, with one selected for each span question. Overlapping word logits are combined before decoding. [Editable SVG](assets/architecture/10-decoder-span-heads.svg).

**State prefix and question isolation**

<p align="center"><img src="https://huggingface.co/vllm-sr/Vela-2.0-0.8B/resolve/main/assets/architecture/11-state-prefix-and-question-isolation.png" alt="Vela 2.0 hybrid models: state prefix reuse and isolated question blocks" width="100%" /></p>

For a fixed rendered state, each question block sees the same state prefix and its own causal tokens. Additional span questions use separate sequences. [Editable SVG](assets/architecture/11-state-prefix-and-question-isolation.svg).

</details>

Choice, Yes/no, Score and Set share a candidate readout. Router and broad span heads use the same word-by-label topology with separate weights. The hybrid backbone reuses the state prefix within each rendered sequence; execution details are in [USAGE.md](USAGE.md#precision-and-execution).

## Reference

| Document | Contents |
| --- | --- |
| [USAGE.md](USAGE.md) | Full examples and recorded responses, typed parts, head selection, calibration, SDK and HTTP serving |
| [EVALUATION.md](EVALUATION.md) | Complete family results, research versus shipped PII modes, benchmark protocols and evaluation disclosures |
| [TRAINING.md](TRAINING.md) | Training stages, data mixtures, provenance, licences and release files |
| [PARITY.md](PARITY.md) | Export parity against the research scorer |

## Credit and citation

Vela 2.0 is led by KR Labs and [vLLM Semantic Router](https://github.com/vllm-project/semantic-router).

Read the [Vela 2.0 technical overview](https://vllm-sr.ai/blog/vela-2-0-open-foundation-routing-models/).

```bibtex
@misc{vela2_08b_2026,
  title        = {Vela 2.0: Towards Open Foundation Routing Models},
  author       = {{KR Labs} and {vLLM Semantic Router}},
  year         = {2026},
  note         = {Blog post. Model: vllm-sr/Vela-2.0-0.8B},
  howpublished = {\url{https://vllm-sr.ai/blog/vela-2-0-open-foundation-routing-models/}}
}
```

## Licence

Apache-2.0 for this model's weights, code and documentation. It is derived from [Decision-2.0-Eos-0.8B](https://huggingface.co/vllm-sr/Decision-2.0-Eos-0.8B) (Apache-2.0), itself built on Qwen3.5-0.8B (Apache-2.0); their licences and notices are passed on in `LICENSE`, `NOTICE`, `ATTRIBUTIONS.md` and `LICENSES/`. Changes against Decision-2.0-Eos-0.8B: [MODIFICATIONS.md](MODIFICATIONS.md). Training data keep their own licences, some of them CC-BY-SA share-alike.
