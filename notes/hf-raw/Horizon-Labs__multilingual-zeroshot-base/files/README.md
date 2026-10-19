---
license: apache-2.0
language:
- multilingual
- en
- de
- fr
- es
- pt
- it
- nl
- pl
- ru
- uk
- cs
- tr
- ar
- fa
- he
- hi
- bn
- zh
- ja
- ko
- vi
- id
- th
- sw
- el
library_name: transformers
pipeline_tag: zero-shot-classification
base_model: jhu-clsp/mmBERT-base
tags:
- zero-shot-classification
- zero-shot
- nli
- natural-language-inference
- text-classification
- multilingual
- modernbert
- mmbert
- onnx
- transformers.js
widget:
- text: "Mi pedido llegó roto y quiero que me devuelvan el dinero."
  candidate_labels: "refund request, shipping question, product praise, account problem"
  multi_class: false
- text: "Die Bundesregierung hat ein neues Klimaschutzpaket beschlossen, das den Ausbau der Windenergie beschleunigen soll."
  candidate_labels: "politics, sports, business, science and technology, entertainment"
  multi_class: false
- text: "画面はとてもきれいだけど、バッテリーが半日しか持たないのが残念です。"
  candidate_labels: "screen, battery, price, camera"
  multi_class: true
datasets:
- nyu-mll/multi_nli
- stanfordnlp/snli
- alisawuffles/WANLI
- Horizon-Labs/multilingual-zeroshot-synthetic
---

# Multilingual Zero-Shot Classifier (base, 308M)

Classify text in 30+ languages into **any labels you choose**, with no training. Use it with the transformers
`zero-shot-classification` pipeline, like `facebook/bart-large-mnli`, but multilingual, smaller, and with an 8k-token
context window (fine-tuned at up to 1,024 tokens).

- **Multilingual**: the text can be in any of the languages below; labels and the hypothesis template stay in English.
- **Commercially clean**: Apache-2.0, trained only on data that allows commercial use (no XNLI, ANLI or other
  non-commercial sets). See Training.
- **Small and fast**: 308M parameters, ModernBERT architecture (mmBERT), ONNX included for CPU and the browser.
- **Honest numbers**: all models below were run by us with the same script and templates.

Try it in the browser: [Horizon-Labs/multilingual-zeroshot demo](https://huggingface.co/spaces/Horizon-Labs/multilingual-zeroshot).

Part of Horizon Labs' open models ([collection](https://huggingface.co/collections/Horizon-Labs/agent-i-o-guards-6ab403c49494bc2b71ca7669)).
Source code: [github.com/horizon-ai-labs/agent-io-guards](https://github.com/horizon-ai-labs/agent-io-guards).

## Quick start

```python
from transformers import pipeline

clf = pipeline("zero-shot-classification", model="Horizon-Labs/multilingual-zeroshot-base")
clf("Mi pedido llegó roto y quiero que me devuelvan el dinero.",
    candidate_labels=["refund request", "shipping question", "product praise", "account problem"])
# {'labels': ['refund request', ...], 'scores': [...]}

# several labels can apply at once
clf("The camera is great but the battery dies by noon.", ["camera", "battery", "screen", "price"], multi_label=True)

# a task-specific template often helps
clf("¿Me pones una alarma a las siete?", ["set an alarm", "play music", "weather"], hypothesis_template="The user wants to {}.")
```

Labels: `not_entailment` (0) and `entailment` (1). For each candidate label the model scores whether the text entails
"This example is {label}." (or your `hypothesis_template`). Any NLI-style use works too: pass `text` and `text_pair`
to a `text-classification` pipeline.

## Evaluation

Sortable comparison with more models: [multilingual zero-shot classification leaderboard](https://huggingface.co/spaces/Horizon-Labs/multilingual-zeroshot-leaderboard).

Accuracy, single-label (`multi_label=False`: the label with the highest entailment score wins). English templates and
labels for every language; the same template for every model (e.g. "This text is about {}." for SIB-200). No model saw
these datasets' training splits, except where marked. ‡ = trained partly on data with non-commercial licenses (their
`-c` variants are the commercially usable ones). Script: `zeroshot/evaluate_zs.py`.

§ = **label names seen in our synthetic training data**. Since v1.1 our training data includes generic label
taxonomies (topics, news sections, Q&A question topics, emotions, sentiment) whose label names overlap these benchmarks'
label sets; for Yahoo Answers and AG News almost exactly. No benchmark texts were used, but on these rows our models are
not zero-shot with respect to the label names, so compare with care. For CLINC150, 31 of the 150 intent names occur in
our training data. The MASSIVE, Banking77, MTOP and XNLI label sets were not used (1-2 names each occur by chance, such
as "play music").

### Multilingual

| | **this model** (308M) | small (141M) | large (568M) | bge-m3-zeroshot-v2.0-c (568M) | mDeBERTa-v3-base-xnli (278M) | xlm-roberta-large-xnli (560M) | bge-m3-zeroshot-v2.0 (568M) ‡ |
|---|---|---|---|---|---|---|---|
| MASSIVE intents (60 labels), 16 languages | 0.505 | 0.418 | 0.535 | 0.411 | 0.351 | 0.404 | **0.611** |
| MTOP intents (88 labels), 6 languages | 0.420 | 0.350 | 0.424 | 0.303 | 0.353 | 0.386 | **0.442** |
| SIB-200 topics (7 labels), 16 languages § | 0.826 | 0.797 | 0.834 | 0.782 | 0.654 | 0.526 | **0.837** |

Per language, mean of MASSIVE and SIB-200:

| | **this model** (308M) | small (141M) | large (568M) | bge-m3-zeroshot-v2.0-c (568M) | mDeBERTa-v3-base-xnli (278M) | xlm-roberta-large-xnli (560M) | bge-m3-zeroshot-v2.0 (568M) ‡ |
|---|---|---|---|---|---|---|---|
| English | 0.713 | 0.681 | 0.716 | 0.596 | 0.537 | 0.504 | **0.752** |
| German | 0.675 | 0.600 | 0.685 | 0.607 | 0.527 | 0.472 | **0.748** |
| French | 0.703 | 0.653 | 0.704 | 0.617 | 0.526 | 0.487 | **0.753** |
| Spanish | 0.645 | 0.596 | 0.658 | 0.561 | 0.488 | 0.456 | **0.742** |
| Portuguese | 0.692 | 0.640 | 0.679 | 0.585 | 0.491 | 0.446 | **0.713** |
| Russian | 0.671 | 0.633 | 0.663 | 0.597 | 0.497 | 0.453 | **0.724** |
| Polish | 0.679 | 0.639 | 0.712 | 0.640 | 0.533 | 0.486 | **0.756** |
| Turkish | 0.671 | 0.592 | 0.680 | 0.605 | 0.491 | 0.455 | **0.713** |
| Arabic | 0.620 | 0.554 | 0.666 | 0.559 | 0.476 | 0.433 | **0.681** |
| Hindi | 0.641 | 0.563 | 0.675 | 0.589 | 0.511 | 0.459 | **0.719** |
| Chinese | 0.687 | 0.643 | 0.709 | 0.627 | 0.514 | 0.481 | **0.760** |
| Japanese | 0.706 | 0.656 | 0.695 | 0.637 | 0.530 | 0.492 | **0.752** |
| Korean | 0.656 | 0.593 | 0.707 | 0.610 | 0.494 | 0.486 | **0.714** |
| Vietnamese | 0.650 | 0.606 | 0.671 | 0.605 | 0.478 | 0.490 | **0.735** |
| Indonesian | 0.687 | 0.628 | 0.733 | 0.625 | 0.520 | 0.483 | **0.749** |
| Swahili | 0.546 | 0.444 | **0.595** | 0.484 | 0.430 | 0.359 | 0.573 |

### English

| | **this model** (308M) | small (141M) | large (568M) | bge-m3-zeroshot-v2.0-c (568M) | mDeBERTa-v3-base-xnli (278M) | xlm-roberta-large-xnli (560M) | bge-m3-zeroshot-v2.0 (568M) ‡ | bart-large-mnli (407M) | deberta-v3-base-zeroshot-v2.0 (184M) ‡ |
|---|---|---|---|---|---|---|---|---|---|
| AG News (4) § | 0.839 | 0.832 | 0.814 | 0.726 | 0.670 | 0.591 | **0.886** | 0.684 | 0.884 |
| Yahoo Answers (10) § | 0.649 | 0.626 | 0.658 | 0.564 | 0.497 | 0.529 | 0.654 | 0.586 | **0.672** |
| Banking77 (77) | 0.568 | 0.543 | 0.623 | 0.430 | 0.287 | 0.166 | 0.695 | 0.480 | **0.714** |
| CLINC150 (150) § | 0.672 | 0.596 | **0.680** | 0.597 | 0.484 | 0.623 | 0.661 | 0.679 | 0.615 |
| Emotion (6) § | 0.488 | 0.413 | 0.511 | 0.476 | 0.483 | 0.345 | 0.677 | 0.463 | **0.737** |
| SST-2 (2) § | 0.891 | 0.841 | 0.880 | 0.865 | 0.844 | 0.820 | 0.905 | 0.922 | **0.947** |
| MASSIVE, English only | 0.573 | 0.503 | 0.583 | 0.413 | 0.397 | 0.443 | 0.680 | 0.530 | **0.710** |
| SIB-200, English only | 0.853 | **0.858** | 0.848 | 0.779 | 0.676 | 0.564 | 0.824 | 0.760 | 0.745 |

### Labels in the text's language

Users often write the candidate labels and the template in the language of the text. Below, the MASSIVE and SIB-200
label sets and templates were translated into each of the 15 non-English languages (by Qwen3.8-27B); accuracy is the mean
over those languages, next to the same languages with English labels.

| | MASSIVE, native labels | MASSIVE, English labels | SIB-200, native labels | SIB-200, English labels |
|---|---|---|---|---|
| small (141M) | 0.423 | 0.413 | 0.761 | 0.793 |
| **base (308M)** (this model) | 0.494 | 0.500 | 0.809 | 0.824 |
| large (568M) | 0.522 | 0.532 | 0.831 | 0.833 |
| bge-m3-zeroshot-v2.0-c | 0.423 | 0.411 | 0.741 | 0.782 |
| mDeBERTa-v3-base-xnli | 0.344 | 0.348 | 0.647 | 0.653 |
| bge-m3-zeroshot-v2.0 ‡ | 0.560 | 0.606 | 0.806 | 0.838 |

### Versions

v1.1 adds data with broad, reusable label taxonomies, so it is better on common categories (topics, emotions, sentiment,
aspects) — the § rows, where the label names are familiar to it. On label sets it has not seen it stays within about
±0.015 of v1.0 (slightly lower on some).

v1.2 is distilled from the large model: same data, with half of the loss on the large model's probabilities instead of
the hard labels. It gains most on the unseen MASSIVE label set; other rows move by about ±0.01.

v1.3 (this version) adds native-language labels: for half of the non-English training texts, the label set and template were translated
into the text's language (by Qwen3.8-27B). It is clearly better when labels are written in the text's language (see
above); with English labels it is about the same or better on the unseen label sets (MASSIVE, Banking77), and the
table shows the other rows, which move in both directions. Like v1.2, it is distilled from the large model (now on
the v1.3 data).

To pin an earlier model, load it with `revision="v1.0"` or `revision="v1.1"` or `revision="v1.2"`.

| | v1.0 | v1.1 | v1.2 | **v1.3 (this version)** |
|---|---|---|---|---|
| MASSIVE (unseen label set) | 0.492 | 0.482 | **0.507** | 0.505 |
| MASSIVE, native labels (15 languages) | – | – | 0.473 | **0.494** |
| SIB-200, native labels (15 languages) | – | – | 0.787 | **0.809** |
| Banking77 (unseen label set) | **0.568** | 0.554 | 0.555 | **0.568** |
| MTOP (unseen label set) | – | – | **0.424** | 0.420 |
| CLINC150 § | – | – | 0.655 | **0.672** |
| XNLI (balanced acc.) | **0.801** | 0.799 | 0.792 | 0.798 |
| SIB-200 § | 0.798 | 0.813 | 0.819 | **0.826** |
| AG News § | 0.789 | 0.826 | 0.824 | **0.839** |
| Yahoo Answers § | 0.503 | **0.649** | 0.640 | **0.649** |
| Emotion § | 0.466 | **0.491** | 0.483 | 0.488 |
| SST-2 § | 0.884 | 0.875 | 0.883 | **0.891** |

### NLI

| | **this model** (308M) | small (141M) | large (568M) | bge-m3-zeroshot-v2.0-c (568M) | mDeBERTa-v3-base-xnli (278M) | xlm-roberta-large-xnli (560M) | bge-m3-zeroshot-v2.0 (568M) ‡ |
|---|---|---|---|---|---|---|---|
| XNLI test, 12 languages (balanced acc.) † | 0.798 | 0.758 | 0.800 | 0.825 | 0.845 | **0.992** | 0.818 |

† XNLI is included for reference only: xlm-roberta-large-xnli and mDeBERTa-xnli were trained on XNLI data (the
first scores 0.99, which suggests it saw the test sentences). Our models never saw XNLI.

## Limitations

- English-only models trained with more (partly non-commercial) classification data are better on English topic
  and emotion benchmarks (e.g. deberta-v3-base-zeroshot-v2.0 on Emotion and Yahoo). If you only need English, compare
  them on your data.
- bge-m3-zeroshot-v2.0 (568M, trained partly on non-commercial data) scores higher on MASSIVE, SIB-200 and MTOP.
- Zero-shot accuracy depends a lot on label wording and the template. Use descriptive labels ("request a refund"
  rather than "refund_req") and try a template that fits your task. The model links explicit wording better than
  implied categories. Example (small v1.1, multi-label, a gym review not like our training domains): "The machines are
  always taken after 5pm and half the treadmills are broken, but the coaches really know their stuff. For 60 euros a
  month I expected cleaner showers." gives equipment 0.99, trainers 0.97, membership cost 0.82, but hygiene only 0.28
  and crowding 0.03 (v1.0: trainers 0.56, membership cost 0.58).
- Broad labels (e.g. "world news", "education") tend to win over specific ones. Emotions close in meaning (joy / love
  / surprise) are often confused.
- With `multi_label=True`, scores are independent; tune the threshold on a few examples of your own.
- Lower-resource languages (e.g. Swahili) score clearly lower than high-resource ones.
- Much of the training data is synthetic (Qwen3.8-27B) or machine-translated.

## Training

- Backbone: [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) (MIT), sequence-pair classification, bf16, max length 1024.
- Data (label = does the text entail the hypothesis):
  - English NLI: [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) (OANC and CC-BY-SA-3.0 parts),
    [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) (CC-BY-SA-4.0), [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) (CC-BY-4.0).
  - 120k MultiNLI/WANLI pairs machine-translated by Qwen3.8-27B into 24 languages, with native and English hypotheses.
  - Synthetic zero-shot tasks by Qwen3.8-27B: FineWeb-Edu / FineWeb-2 passages (ODC-BY) labelled by topic, genre,
    audience, tone and purpose with near-miss wrong labels, and ~90k short texts (requests, reviews, tickets, posts,
    headlines) over 26 task types, 32 domains and 33 languages, each with an invented label set and hypothesis template.
  - (v1.3) Native-language labels: label sets and templates of the non-English synthetic items translated by
    Qwen3.8-27B into the item's language; half of those items are trained with the native labels.
  - (v1.2) Distillation: half of the loss uses the probabilities of multilingual-zeroshot-large (568M) on the same
    training pairs instead of the hard labels.
  - (v1.1) Generic taxonomies by Qwen3.8-27B: 24k new FineWeb / FineWeb-2 passages labelled for topic, text type,
    sentiment, audience, purpose and news section; ~130k short texts written for fixed label sets (emotion, sentiment,
    Q&A question topic, news section, customer-message topic, urgency, formality, spam) without using the label words;
    ~25k reviews in 8 domains mentioning aspects (e.g. "internet", "food") without naming them.
- The Qwen-generated classification data (short texts, taxonomies, aspects, labelled passages) is published as
  [Horizon-Labs/multilingual-zeroshot-synthetic](https://huggingface.co/datasets/Horizon-Labs/multilingual-zeroshot-synthetic).
- Not used: XNLI, ANLI, FEVER-NLI, any benchmark above.
