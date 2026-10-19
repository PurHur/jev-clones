---
license: cc-by-sa-4.0
base_model: Qwen/Qwen3-0.6B-Base
language:
  - en
  - de
tags:
  - web-agents
  - browser-automation
  - element-grounding
  - decision-model
  - long-context
library_name: zeiger
---

# Zeiger 0.6B

> **Pre-alpha.** Training is still running; the weights here are a snapshot of the current best round and will be
> replaced as later rounds improve on it. Numbers below move with each release.

*Zeiger* — German for the hand that points. Give it a web page and an instruction, and it points at the element
the instruction means — or says **none of these** — in a single forward pass.

A browser agent has to answer one question per step: *which element do I act on?* A real page offers 40 to 1,500
candidates. Short-context models cannot read that, so agents fall back to tournaments of 5-way questions: slow,
blind to the rest of the page, and structurally unable to answer "the element is not here".

Zeiger reads the page whole. Every option is scored against every other option, `"none"` competes with the rest,
and the confidences are calibrated.

## Results

One thousand short questions, four hundred whole pages and fifteen hundred general decisions — all on websites
that appear nowhere in training, every model answering the identical questions, scored by the same code.

| | Zeiger 0.6B | Jev (`jev-latest`) |
|---|---:|---:|
| short questions, top-1 | **0.900** | 0.775 |
| whole pages, top-1 | **0.793** | 0.708 † |
| "none of these", F1 (short) | **0.914** | 0.449 |
| "none of these", F1 (whole pages) | **0.788** | 0.289 |
| general typed decisions, top-1 | 0.807 | **0.850** |
| calibration error, short (ECE, lower is better) | **0.029** | 0.072 |
| calibration error, whole pages | **0.047** | 0.070 |
| calibration error, general | **0.036** | 0.051 |
| ms per question (short / whole page) | **140 / 1502** | 570 / 888 |

† Jev accepts at most **255 options per call**, so whole pages are answered as a tournament of chunks with a
final round between the winners — 192 of the 400 pages needed one. Zeiger answers each page in a single pass.

[Jev](https://typesafe.ai) is a general-purpose decision model, not trained on this task; it leads on general
typed decisions, and Zeiger leads on the browser work it was built for. The confidences are the other
difference: Zeiger's are better calibrated on all three kinds of question, so a stated 0.9 is closer to being
right nine times in ten — and the gap in "none of these" means it far more reliably says *the element is not
here* instead of acting on the wrong one.

## Specifications

| | |
|---|---|
| parameters | 614M — 596M encoder (Qwen3-0.6B-Base, language-model head removed) + 18M decision head |
| context | **32,768 tokens trained**; longer pages are chunked, so the served window is not the limit |
| options per question | tested to **1,501**; the ceiling is memory, not architecture — the head is O(K²) |
| per option | up to 200 tokens (an element's role, name and a little context) |
| state | up to 1,536 tokens; instruction up to 512 |
| question types | `choice` (with `"none"`), `noul` (yes/no), `score` (ordinal) |
| precision | bf16 on GPU, fp32 on CPU |
| calibration | temperature per option-count bucket, stored in the config and applied before the softmax |

## Use it

The inference engine is a separate, small package: **https://github.com/PurHur/zeiger**

```bash
pip install -r requirements.txt          # torch + transformers
huggingface-cli download php-ai/zeiger-0.6b --local-dir models/zeiger-0.6b
```

```python
from zeiger import Engine

engine = Engine("models/zeiger-0.6b", warmup=True)   # picks the GPU when there is one

answers = engine.decide(
    {"instruction": 'Click the "Sign in" button', "page": {"title": "Shop", "url": "https://shop.test/"}},
    {"q": {"type": "choice",
           "instructions": "Which page element does the instruction refer to?",
           "criteria": {"e0": 'link "Home" href=/',
                        "e1": 'button "Sign in" in header',
                        "e2": 'input "Search"',
                        "none": "none of the listed elements fits this step"}}})

answers["q"]["choice"]         # 'e1'
answers["q"]["confidence"]     # 0.999
answers["q"]["probabilities"]  # every option, calibrated
```

Or over HTTP, including Docker for CPU and AMD ROCm:

```bash
python serve.py --model models/zeiger-0.6b        # POST /decide
docker compose up zeiger
```

## How it works

`Qwen3-0.6B-Base` with its language-model head removed: the transformer body is used as an **encoder** and
nothing is ever generated. The page becomes one sequence — state, question, then every option followed by a
reserved marker token — and the hidden state at each marker is that option's summary. A small head reads the
markers: a 2-layer bidirectional transformer across the options (so an option is judged against the ones that
come later on the page too), a question-type embedding, and a scorer giving one logit per option.

The head costs O(K²) in the number of options, never O(L²) in page length, and pages beyond the window are
encoded in 2k-token chunks that share a prefix — so attention cost stays linear in page size. Trained at a 32k
window; the chunked encoder is not limited by it.

Three question types are served: `choice` (pick one, `"none"` allowed), `noul` (yes/no, answered as the
probability that the statement holds) and `score` (ordinal levels).

## Limits

* It reads **text descriptions of elements**, not pixels: give it the accessible name, role and a little context.
  It is not a screenshot model.
* Instructions phrased as a *goal* ("book the cheapest flight") are harder for it than instructions that name the
  element ("click Search") — 0.699 against 0.975 on held-out websites. It is the weakest part of the model.
* It answers questions; it does not plan, click, or produce text.
* English and German.

## Licence and credits

Weights: **CC-BY-SA-4.0** — share adaptations of the weights under the same licence. The inference code is
Apache-2.0 and lives in the [engine repository](https://github.com/PurHur/zeiger).

Built on **[Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)** (Apache-2.0).

Trained on public datasets, credited to their authors under their licences: SNLI, STS-B, ARC (CC-BY-SA-4.0);
BoolQ, DBpedia, MultiNLI (CC-BY-SA-3.0); Mind2Web, MT-Mind2Web, MASSIVE, Banking77, SMS-Spam (CC-BY-4.0);
CLINC150 (CC-BY-3.0); GoEmotions (Apache-2.0); CommonsenseQA, HellaSwag (MIT); civil_comments (CC0).
