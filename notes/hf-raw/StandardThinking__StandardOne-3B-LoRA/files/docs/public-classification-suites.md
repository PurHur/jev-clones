# Public classification and decision suites

> **Note.** The Standard One 3B v2 figures for these suites are on its model card; the tables below are still those of v1.1.

Measured 25 September 2026 through the served endpoints (merged BF16, SGLang 0.5.20, jev-adapter, native wording, no system prompt, single option order). Each suite: 400 cases sampled with seed 13 from the public test split, one request per case; ECE (10-bin, top label) uses the served temperature (8B T=1.65, 3B T=1.55). Accuracy and macro-F1 are argmax metrics and do not depend on the temperature. banking77 (77 labels) is not runnable: the endpoint accepts at most 26 options per question.

## Public datasets

| Suite | n | Acc 8B | Acc 3B | Macro-F1 8B | Macro-F1 3B | ECE 8B | ECE 3B |
|---|---|---|---|---|---|---|---|
| AG News (4 topics) | 400 | 84.2 | 83.0 | 82.9 | 81.4 | 0.084 | 0.094 |
| DAIR Emotion (6 labels) | 400 | 56.5 | 58.8 | 46.6 | 48.6 | 0.191 | 0.064 |
| SST-5 (5-level sentiment, ordinal score) | 400 | — | — | — | — | — | — |
| Typed decisions (400 cases, 2,000 decisions) | 400 | 69.3 | 69.0 | 60.2 | 58.9 | 0.056 | 0.048 |

## Application workflows (400 cases each, public sources)

| Suite | n | Acc 8B | Acc 3B | Macro-F1 8B | Macro-F1 3B | ECE 8B | ECE 3B |
|---|---|---|---|---|---|---|---|
| Email spam | 400 | 92.2 | 93.0 | 92.2 | 93.0 | 0.047 | 0.045 |
| Phishing | 400 | 86.0 | 89.8 | 84.2 | 88.6 | 0.048 | 0.037 |
| LLM guardrails (jailbreak vs benign) | 400 | 50.7 | 57.5 | 50.5 | 56.7 | 0.314 | 0.181 |
| Moderation (toxic vs benign) | 400 | 85.0 | 81.5 | 85.0 | 81.4 | 0.079 | 0.046 |
| RAG passage relevance | 400 | 60.2 | 57.2 | 56.9 | 53.7 | 0.307 | 0.289 |
| Support triage (10 queues) | 400 | 35.2 | 39.5 | 36.1 | 37.5 | 0.435 | 0.359 |
| Model routing (domain) | 399 | 93.0 | 83.7 | 56.5 | 52.8 | 0.048 | 0.046 |

## MASSIVE intent, 20 options (chance = 5 %)

| Language | n | Acc 8B | Acc 3B | Macro-F1 8B | Macro-F1 3B | ECE 8B | ECE 3B |
|---|---|---|---|---|---|---|---|
| en | 400 | 88.0 | 85.5 | 61.7 | 54.3 | 0.024 | 0.055 |
| ja | 400 | 87.0 | 86.0 | 60.1 | 65.2 | 0.032 | 0.037 |
| zh | 400 | 86.0 | 84.0 | 63.2 | 60.7 | 0.042 | 0.054 |
| es | 400 | 82.0 | 82.0 | 55.5 | 53.5 | 0.025 | 0.064 |
| fr | 400 | 84.5 | 82.8 | 59.2 | 54.2 | 0.040 | 0.061 |
| de | 400 | 85.8 | 83.8 | 62.6 | 52.8 | 0.039 | 0.043 |
| pt | 400 | 83.5 | 80.5 | 53.4 | 48.5 | 0.044 | 0.056 |
| ru | 400 | 85.0 | 81.8 | 62.0 | 55.5 | 0.038 | 0.062 |
| ko | 400 | 88.2 | 83.8 | 67.2 | 58.0 | 0.034 | 0.029 |

## XNLI, 3 labels

| Language | n | Acc 8B | Acc 3B | Macro-F1 8B | Macro-F1 3B | ECE 8B | ECE 3B |
|---|---|---|---|---|---|---|---|
| en | 400 | 63.5 | 69.2 | 64.2 | 68.7 | 0.136 | 0.080 |
| es | 400 | 57.2 | 64.0 | 57.5 | 63.7 | 0.186 | 0.057 |
| fr | 400 | 61.0 | 64.2 | 61.3 | 63.7 | 0.152 | 0.067 |
| de | 400 | 60.5 | 60.5 | 60.3 | 60.2 | 0.160 | 0.024 |
| ru | 400 | 57.2 | 60.8 | 56.3 | 60.2 | 0.212 | 0.047 |
| zh | 400 | 61.0 | 53.2 | 61.0 | 52.6 | 0.169 | 0.061 |

## Option-order robustness

Share of cases whose answer changes when the options are presented in a permuted order (lower is better).

| Suite | 8B | 3B |
|---|---|---|
| AG News (4 topics) | 0.033 | 0.037 |
| DAIR Emotion (6 labels) | 0.072 | 0.185 |
| massive_intent.en | 0.133 | 0.135 |

SST-5 score MAE on the 0–4 scale: 8B 0.566, 3B 0.653.
