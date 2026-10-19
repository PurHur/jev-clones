---
license: apache-2.0
language:
- en
- tr
- de
base_model: internlm/Intern-Decision-4B
library_name: transformers
pipeline_tag: text-classification
datasets:
- mertkayacs/jevalt-data
tags:
- decision-model
- calibration
- conformal-prediction
- uncertainty
- reasoning
- routing
- triage
- jev
- typesafe
- qwen3.5
- english
- small-language-model
- local-llm
- on-device
- text-classification
- english-llm
widget:
- text: '{"state":"Hi, I was charged twice for my March subscription: two payments
    of €29 on 3 March. Please refund the duplicate today, otherwise I will cancel.\nThanks,
    Daniel","questions":{"decision":{"type":"choice","instructions":"Which team should
    handle this ticket?","criteria":{"Billing":"payments, invoices, refunds","Technical
    support":"bugs, errors, outages","Sales":"prices, upgrades, new contracts","Account":"login,
    password, profile changes"}}},"reasoning":"off","abstain":false}'
  example_title: 'Recorded full-precision Deem-4B: support ticket, 1 October 2026'
  output:
  - label: Billing
    score: 0.947664
  - label: Technical support
    score: 0.030092
  - label: Sales
    score: 0.013086
  - label: Account
    score: 0.009158
base_model_relation: finetune
---
# Deem-4B: small English language model for local decisions
This project is by Eschatia Labs, an independent non-profit open-source AI research lab.

Deem-4B is a 4B English language model for text classification, routing and decisions with probabilities. **Deem-4B: 94.7% accuracy; Kev-4B: 84.7%** on held-out English decisions. The test set comes from the same data pipeline as the training data.

[Try a decision in the browser](https://huggingface.co/spaces/mertkayacs/JevAlt), or run the GGUF build locally below.

## Quick facts

| Item | Value |
|---|---|
| Parameters | 4B |
| Base model | internlm/Intern-Decision-4B |
| Held-out English accuracy | 94.7% (Kev-4B: 84.7%) |
| Brier score | 0.091 (Kev-4B: 0.256) |
| Planted instruction flips | 14.0% (Kev-4B: 36.0%) |
| Live test (4 Oct 2026) | 122/130 |
| Q4_K_M file size | 2.71 GB |
| Peak RAM (Q4_K_M, 4k context) | 3.04 GB |
| Hugging Face downloads | 215 (as of 2026-10-15) |
| GitHub stars | 5 |

## Project links

- [Deem-4B project page](https://jevalt.mertkayacs.com/models/deem-4b/)
- [JevAlt GitHub repository](https://github.com/mertkayacs/jevalt)
- [Try the Space](https://huggingface.co/spaces/mertkayacs/JevAlt)
- [GGUF builds](https://huggingface.co/mertkayacs/Deem-4B-GGUF)
- [Training data](https://huggingface.co/datasets/mertkayacs/jevalt-data)
- [Benchmark results](https://huggingface.co/datasets/mertkayacs/jevalt-bench)
- [JevOss evaluator](https://github.com/mertkayacs/jevoss)
- [Benchmark page](https://eschatialabs.com/benchmarks/jevalt-bench/)

## Cite this

```bibtex
@software{kaya2026jevalt,
  author = {Kaya, Mert},
  title  = {JevAlt: Small Decision Models for English, Turkish and German},
  year   = {2026},
  license = {Apache-2.0},
  url    = {https://github.com/mertkayacs/jevalt}
}
```
## Run locally

The Q4_K_M build used **3.04 GB RAM** at a 4k context. The JevAlt server uses llama.cpp to read decision probabilities and load the matching calibration.

```sh
pip install "jevalt[serve,gguf] @ git+https://github.com/mertkayacs/jevalt" && jevalt serve --model mertkayacs/Deem-4B-GGUF --file Deem-4B-Q4_K_M.gguf
```

Ollama: `ollama run mertkayacs/deem-4b` ([library page](https://ollama.com/mertkayacs/deem-4b)). It answers with the option letter; the JevAlt server adds the Jev API and calibrated probabilities.

Send a situation, question and options to the local server:

```python
import requests

request = {
    "state": "I was charged twice for my subscription. Please refund the duplicate.",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "payments, refunds", "technical": "bugs, outages"},
        }
    },
    "reasoning": "off",
    "abstain": False,
}
response = requests.post("http://127.0.0.1:8000/v1/systemone", json=request)
response.raise_for_status()
print(response.json())
```

Write the situation, question and options in English. The API also supports yes/no questions, ordered scores and an `unknown` option with `abstain: true`. The card widget shows a recorded answer from the full-precision model.

<details>
<summary>Use the full-precision weights with Transformers</summary>

This repository holds the full-precision weights. The JevAlt Transformers backend preserves the decision-token format and calibration:

```sh
pip install "jevalt[hf] @ git+https://github.com/mertkayacs/jevalt" && jevalt serve --backend hf --model mertkayacs/Deem-4B
```

See the [local run guide](https://github.com/mertkayacs/jevalt/blob/main/docs/local.md) for memory and client settings.

</details>

## Recorded examples

The [JevAlt Space](https://huggingface.co/spaces/mertkayacs/JevAlt) examples below were recorded from Deem-4B on 1 October 2026. The source file stores every option probability and the reasoning trace where reasoning was enabled.

| Use case | Question | Recorded answer |
|---|---|---|
| Duplicate subscription charge | Which team should handle the ticket? | **Billing**, 94.8% |
| Checkout outage | How severe is the incident? | **Critical**, 87.6% |
| Qualified sales lead | How should sales treat the lead? | **Hot**, 92.3% |
| 14-day return window | Is the return within the window? | **No**, 97.9% |
| Phishing email with a planted instruction | Where should the email go? | **Quarantine**, 94.3% |
| Hotel message missing the booking detail | Which room type did the guest book? | **unknown**, 97.6% |
| Village barn fire | What should Mirka do next? | **Help with the fire**, 66.0% |

[Recorded requests and probabilities](https://huggingface.co/datasets/mertkayacs/jevalt-bench/blob/main/results/examples/space-examples.json).

## Live test

We sent Deem-4B 130 English requests with known answers on 4 October 2026. It answered 122 correctly. The released files retain every request, raw response and miss.

| Case | Requests | Result |
|---|---|---|
| Planted instructions | 30 phishing emails with a planted line, plus the same 10 without it | 26/30 quarantined; 10/10 controls |
| Long policies | 20 customers checked against one six-rule return policy | 17/20 |
| Negations | 15 facts asked plainly and with a negation | 29/30 |
| Missing facts | 10 cases without the deciding fact, plus 10 with it | 10/10 `unknown`; 10/10 controls |
| Casual messages | 20 English messages with typos and slang | 20/20 |

[Live requests and misses](https://huggingface.co/datasets/mertkayacs/jevalt-bench/tree/main/results/tested).

## Training

Deem-4B starts from `internlm/Intern-Decision-4B` and uses LoRA on bf16 weights, rank 32 and alpha 32, on one A100 80 GB. Training ran for one epoch over the shared English, Turkish and German mix (17,363 rows, 543 steps, 63 minutes), followed by one epoch on the English-weighted mix (7,783 rows, 303 steps, 38 minutes). The released models used about 2.7 A100 hours. The full project, including labeling, cost 32.2 USD across 11 training jobs.

| Training source | English | Turkish | German |
|---|---:|---:|---:|
| Teacher-written scenarios | 1,287 | 1,741 | 1,408 |
| Emberwick requests | 124 | 129 | 107 |
| Targeted fix sets | 1,256 | 1,197 | 1,257 |
| Converted public datasets | 7,188 | 4,115 | 4,074 |

The writers, labelers and trace writers were GLM-5.x, Mistral Large 3, DeepSeek V4 Pro, DeepSeek V4.1 Flash, Gemma 4 26B-A4B, Qwen3.5-122B-A10B, Qwen3.5-35B-A3B, Kimi K3 and MiniMax M3. The [training-data card](https://huggingface.co/datasets/mertkayacs/jevalt-data) documents provenance and licenses.

### Targeted data

- **Planted instructions:** 827 rows place a hostile line at the start, middle or end of the state. On the probe, planted lines flip 14.0% of Deem-4B answers and 36.0% of Kev-4B answers. On 203 held-out rows, JevAlt reaches 90.1% and Kev-4B reaches 81.3%.
- **Missing facts:** 310 rows ask a question with and without the deciding fact. JevAlt selects `unknown` in 9/11 held-out missing-fact cases. Intern-Decision-4B also selects it in 9/11, while training raises its mean probability from 0.55 to 0.74.
- **Policies and long text:** 390 policy rows and 1,188 padded rows cover rules with exceptions and facts buried in up to 3,000 tokens. JevAlt reaches 80.0% on 150 policy rows and 95.4% on 285 padded rows; Kev-4B reaches 59.3% and 87.4%. With 600 unrelated words first, Deem-4B loses 17.4 accuracy points, Kev-4B loses 5.4 and Laya loses 10.4.
- **Negation:** 368 paired rows mirror plain and negated questions. JevAlt reaches 96.7% on 30 held-out negated questions and Kev-4B reaches 76.7%.
- **Dates and numbers:** 390 date rows and 383 number rows use programmatic answers. On 80 held-out date rows, JevAlt reaches 71.3% and Kev-4B reaches 67.5%. Both reach 68.2% on 44 number rows.
- **Calibration:** Temperatures were fitted on 3,224 held-out decisions by question type and language. Deem-4B's fitted temperature is 1.10. The 80%, 90% and 95% answer sets use conformal thresholds fitted on the same rows.
- **Reasoning:** Short traces were kept when they reached the correct answer and trained at a lower weight. On English date, number and policy rows, accuracy is 0.761 with reasoning off and 0.769 with `reasoning: "auto"`; the interval for the gain includes zero.

## Results and limits

On the same held-out English decisions, Deem-4B's probability error (Brier score, lower is better) is **0.091**, against **0.256 for Kev-4B**. A separate live test on 4 October 2026 scored **122/130** for Deem-4B. [Comparison files](https://huggingface.co/datasets/mertkayacs/jevalt-bench/tree/main/results/comparison) and [live requests, including misses](https://huggingface.co/datasets/mertkayacs/jevalt-bench/tree/main/results/tested) record both tests.

The paired bootstrap over the held-out English split, using 2,000 resamples, gives a 95% interval of +8.8 to +11.4 accuracy points for Deem-4B's gap over Kev-4B. On external suites, Deem-4B scores 70.3% against Kev-4B's 54.1% on JevBench-hard, 55.5% against 51.3% on TurkishMMLU, 62.0% against 65.3% on GermEval 2017, and 59.5% against 65.3% on 10kGNAD. The typed-decisions train split was included in training. The other suites were outside the training data.

The models were fine-tuned with LoRA from [Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B), using [JevAlt training data](https://huggingface.co/datasets/mertkayacs/jevalt-data). English is this model's focus.

- Refit calibration on your own data before using confidence thresholds. See [JevOss](https://github.com/mertkayacs/jevoss).
- The model accepts text, with an 8k context. Long irrelevant text and date arithmetic remain weak spots.
- Planted instructions still change some answers. Keep authorization checks outside the model and review consequential decisions.

## Downloads and project

[GGUF files and measured agreement](https://huggingface.co/mertkayacs/Deem-4B-GGUF) | [Code and training](https://github.com/mertkayacs/jevalt) | [Deem-4B project page](https://jevalt.mertkayacs.com/models/deem-4b/) | [Training and evaluation details](https://github.com/mertkayacs/jevalt/blob/main/docs/how-it-was-built.md) | [Benchmark page](https://eschatialabs.com/benchmarks/jevalt-bench/).

Apache-2.0. Cite the [JevAlt repository](https://github.com/mertkayacs/jevalt) for this release.

An [Eschatia Labs](https://eschatialabs.com) project. [Mert Kaya](https://mertkayacs.com).
