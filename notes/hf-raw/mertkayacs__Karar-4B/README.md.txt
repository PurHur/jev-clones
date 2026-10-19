---
license: apache-2.0
language:
- tr
- en
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
- turkish
- small-language-model
- local-llm
- on-device
- text-classification
- turkish-llm
widget:
- text: '{"state":"Merhaba, mart aboneliğim için iki kez ücret alındı: 3 Mart''ta
    iki ayrı 29 € ödeme. Fazla çekilen tutarı bugün iade edin, yoksa aboneliğimi iptal
    edeceğim.\nTeşekkürler, Deniz","questions":{"karar":{"type":"choice","instructions":"Bu
    talebi hangi ekip ele almalı?","criteria":{"Fatura":"ödemeler, faturalar, iadeler","Teknik
    destek":"hatalar, arızalar, kesintiler","Satış":"fiyatlar, paket yükseltme, yeni
    sözleşmeler","Hesap":"giriş, şifre, profil değişiklikleri"}}},"reasoning":"off","abstain":false}'
  example_title: 'Recorded full-precision Karar-4B: support ticket, 1 October 2026'
  output:
  - label: Fatura
    score: 0.913536
  - label: Teknik destek
    score: 0.043269
  - label: Satış
    score: 0.027539
  - label: Hesap
    score: 0.015655
base_model_relation: finetune
---
# Karar-4B: small Turkish language model for local decisions
This project is by Eschatia Labs, an independent non-profit open-source AI research lab.

Karar-4B is a 4B Turkish language model for text classification, routing and decisions with probabilities. **Karar-4B: 96.8% accuracy; Kev-4B: 87.1%** on held-out Turkish decisions. The test set comes from the same data pipeline as the training data.

[Try a decision in the browser](https://huggingface.co/spaces/mertkayacs/JevAlt), or run the GGUF build locally below.

## Quick facts

| Item | Value |
|---|---|
| Parameters | 4B |
| Base model | internlm/Intern-Decision-4B |
| Held-out Turkish accuracy | 96.8% (Kev-4B: 87.1%) |
| Brier score | 0.058 (Kev-4B: 0.214) |
| Planted instruction flips | 19.0% (Kev-4B: 36.0%) |
| Live test (4 Oct 2026) | 113/130 |
| Q4_K_M file size | 2.71 GB |
| Peak RAM (Q4_K_M, 4k context) | 3.03 GB |
| Hugging Face downloads | 217 (as of 2026-10-15) |
| GitHub stars | 5 |

## Project links

- [Karar-4B project page](https://jevalt.mertkayacs.com/models/karar-4b/)
- [JevAlt GitHub repository](https://github.com/mertkayacs/jevalt)
- [Try the Space](https://huggingface.co/spaces/mertkayacs/JevAlt)
- [GGUF builds](https://huggingface.co/mertkayacs/Karar-4B-GGUF)
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

The Q4_K_M build used **3.03 GB RAM** at a 4k context. The JevAlt server uses llama.cpp to read decision probabilities and load the matching calibration.

```sh
pip install "jevalt[serve,gguf] @ git+https://github.com/mertkayacs/jevalt" && jevalt serve --model mertkayacs/Karar-4B-GGUF --file Karar-4B-Q4_K_M.gguf
```

Ollama: `ollama run mertkayacs/karar-4b` ([library page](https://ollama.com/mertkayacs/karar-4b)). It answers with the option letter; the JevAlt server adds the Jev API and calibrated probabilities.

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

Write the situation, question and options in Turkish. The API also supports yes/no questions, ordered scores and an `unknown` option with `abstain: true`. The card widget shows a recorded answer from the full-precision model.

<details>
<summary>Use the full-precision weights with Transformers</summary>

This repository holds the full-precision weights. The JevAlt Transformers backend preserves the decision-token format and calibration:

```sh
pip install "jevalt[hf] @ git+https://github.com/mertkayacs/jevalt" && jevalt serve --backend hf --model mertkayacs/Karar-4B
```

See the [local run guide](https://github.com/mertkayacs/jevalt/blob/main/docs/local.md) for memory and client settings.

</details>

<details lang="tr">
<summary>Türkçe özet</summary>

Karar-4B, metni sınıflandırmak ve seçenekler arasında karar vermek için geliştirilmiş küçük bir Türkçe dil modelidir. Ayrı tutulan Türkçe karar örneklerinde Karar-4B %96,8, Kev-4B %87,1 doğruluk elde etti. Eğitim ve test verisi aynı veri üretim sürecinden gelir. Q4_K_M sürümü yaklaşık 3 GB RAM ile kendi bilgisayarınızda çalışır.

</details>

## Recorded examples

The [JevAlt Space](https://huggingface.co/spaces/mertkayacs/JevAlt) examples below were recorded from Karar-4B on 1 October 2026.

| Use case | Recorded answer |
|---|---|
| Duplicate subscription charge | **Fatura**, 91.4% |
| Checkout outage | **Kritik**, 83.7% |
| Qualified sales lead | **Sıcak**, 91.0% |
| 14-day return window | **Hayır**, 97.0% |
| Phishing email with a planted instruction | **Karantina**, 84.2% |
| Hotel message missing the booking detail | **bilinmiyor**, 96.6% |
| Village barn fire | **Yangını söndürmeye yardım et**, 64.0% |

[Recorded requests, questions and probabilities](https://huggingface.co/datasets/mertkayacs/jevalt-bench/blob/main/results/examples/space-examples.json).

## Live test

We sent Karar-4B 130 Turkish requests with known answers on 4 October 2026. It answered 113 correctly.

| Case | Requests | Result |
|---|---|---|
| Planted instructions | 30 phishing emails with a planted line, plus the same 10 without it | 22/30 quarantined; 7/10 controls |
| Long policies | 20 customers checked against one six-rule return policy | 16/20 |
| Negations | 15 facts asked plainly and with a negation | 29/30 |
| Missing facts | 10 cases without the deciding fact, plus 10 with it | 10/10 `unknown`; 10/10 controls |
| Casual messages | 20 Turkish messages with typos and slang | 19/20 |

[Live requests and misses](https://huggingface.co/datasets/mertkayacs/jevalt-bench/tree/main/results/tested).

## Training

Karar-4B starts from `internlm/Intern-Decision-4B` and uses LoRA on bf16 weights, rank 32 and alpha 32, on one A100 80 GB. Training ran for one epoch over the shared mix (17,363 rows, 543 steps, 63 minutes), followed by one epoch on the Turkish-weighted mix (5,982 rows, 281 steps, 35 minutes). The released models used about 2.7 A100 hours. The full project, including labeling, cost 32.2 USD across 11 training jobs.

| Training source | English | Turkish | German |
|---|---:|---:|---:|
| Teacher-written scenarios | 1,287 | 1,741 | 1,408 |
| Emberwick requests | 124 | 129 | 107 |
| Targeted fix sets | 1,256 | 1,197 | 1,257 |
| Converted public datasets | 7,188 | 4,115 | 4,074 |

GLM-5.x, Mistral Large 3, DeepSeek V4 Pro, DeepSeek V4.1 Flash, Gemma 4 26B-A4B, Qwen3.5-122B-A10B, Qwen3.5-35B-A3B, Kimi K3 and MiniMax M3 wrote or labeled rows and traces. Turkish rows were written directly in Turkish. A native-edit pass reviewed 725 rows and rewrote 356. The Turkish run draws 70% of its rows from Turkish.

### Targeted data

- **Planted instructions:** 827 rows place a hostile line inside the state. The probe flips 19.0% of Karar-4B answers and 36.0% of Kev-4B answers. On 203 held-out rows, JevAlt reaches 90.1% and Kev-4B reaches 81.3%.
- **Missing facts:** 310 rows ask with and without the deciding fact. JevAlt selects `unknown` in 9/11 held-out missing-fact cases. Training raises its mean probability from 0.55 to 0.74.
- **Policies and long text:** 390 policy rows and 1,188 padded rows cover exceptions and facts buried in up to 3,000 tokens. JevAlt reaches 80.0% on 150 policy rows and 95.4% on 285 padded rows; Kev-4B reaches 59.3% and 87.4%. With 600 unrelated words first, Karar-4B loses 17.4 accuracy points, Kev-4B loses 5.4 and Laya loses 10.4.
- **Negation:** 368 paired rows mirror plain and negated questions. JevAlt reaches 96.7% on 30 held-out questions and Kev-4B reaches 76.7%.
- **Dates and numbers:** 390 date rows and 383 number rows use programmatic answers. JevAlt reaches 71.3% on 80 date rows and Kev-4B reaches 67.5%. Both reach 68.2% on 44 number rows.
- **Calibration:** Temperatures were fitted on 3,224 held-out decisions. Karar-4B's fitted temperature is 1.08. The 80%, 90% and 95% answer sets use conformal thresholds fitted on the same rows.
- **Reasoning:** The fitted auto thresholds almost never start a trace on Turkish fix-set rows, so `reasoning: "auto"` produces the same answers as `"off"` there.

## Results and limits

On the same held-out Turkish decisions, Karar-4B's probability error (Brier score, lower is better) is **0.058**, against **0.214 for Kev-4B**. A separate live test on 4 October 2026 scored **113/130** for Karar-4B. [Comparison files](https://huggingface.co/datasets/mertkayacs/jevalt-bench/tree/main/results/comparison) and [live requests, including misses](https://huggingface.co/datasets/mertkayacs/jevalt-bench/tree/main/results/tested) record both tests.

The paired bootstrap over the held-out Turkish split, using 2,000 resamples, gives a 95% interval of +8.5 to +10.7 accuracy points for Karar-4B's gap over Kev-4B. Karar-4B scores 58.5% against Kev-4B's 51.3% on TurkishMMLU, 65.5% against 65.3% on GermEval 2017, and 60.5% against 65.3% on 10kGNAD. These suites were outside the training data.

Two blind raters from other model families scored Turkish traces on a 1 to 5 native-feel scale. Karar-4B received 2.80 and 3.45, compared with 2.42 and 3.22 for Intern-Decision-4B. The target was 4.0, and the released model did not reach it.

The models were fine-tuned with LoRA from [Intern-Decision-4B](https://huggingface.co/internlm/Intern-Decision-4B), using [JevAlt training data](https://huggingface.co/datasets/mertkayacs/jevalt-data). Turkish is this model's focus.

- Refit calibration on your own data before using confidence thresholds. See [JevOss](https://github.com/mertkayacs/jevoss).
- The model accepts text, with an 8k context. Long irrelevant text and date arithmetic remain weak spots.
- Planted instructions still change some answers. Keep authorization checks outside the model and review consequential decisions.

## Downloads and project

[GGUF files and measured agreement](https://huggingface.co/mertkayacs/Karar-4B-GGUF) | [Code and training](https://github.com/mertkayacs/jevalt) | [Karar-4B project page](https://jevalt.mertkayacs.com/models/karar-4b/) | [Training and evaluation details](https://github.com/mertkayacs/jevalt/blob/main/docs/how-it-was-built.md) | [Benchmark page](https://eschatialabs.com/benchmarks/jevalt-bench/).

Apache-2.0. Cite the [JevAlt repository](https://github.com/mertkayacs/jevalt) for this release.

An [Eschatia Labs](https://eschatialabs.com) project. [Mert Kaya](https://mertkayacs.com).
