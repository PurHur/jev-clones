---
license: other
license_name: mixed-permissive
license_link: https://huggingface.co/datasets/mertkayacs/jevalt-data#licenses
language:
- en
- tr
- de
task_categories:
- text-classification
tags:
- text-classification
- decision-model
- small-language-model
- turkish
- german
- soft-labels
- calibration
- jev
size_categories:
- 10K<n<100K
configs:
- config_name: en
  data_files:
  - split: train
    path: data/en/train.jsonl
  - split: calibration
    path: data/en/calib.jsonl
  - split: validation
    path: data/en/valid.jsonl
- config_name: tr
  data_files:
  - split: train
    path: data/tr/train.jsonl
  - split: calibration
    path: data/tr/calib.jsonl
  - split: validation
    path: data/tr/valid.jsonl
- config_name: de
  data_files:
  - split: train
    path: data/de/train.jsonl
  - split: calibration
    path: data/de/calib.jsonl
  - split: validation
    path: data/de/valid.jsonl
pretty_name: JevAlt text classification training data
---
# JevAlt training data: text classification in English, Turkish and German
This project is by Eschatia Labs, an independent non-profit open-source AI research lab.

Decision-training data for small language models: situations, questions and probability labels in English, Turkish and German. Used to train Deem-4B, Karar-4B and Wähler-4B. **Karar-4B scores 96.8% against Kev-4B's 87.1%** on Turkish decisions from the same data pipeline. The 96.8% score is Karar-4B's result.

## Quick facts

| Item | Value |
|---|---|
| Languages | English, Turkish, German |
| English rows | 9,855 train / 495 calibration / 286 validation |
| Turkish rows | 7,182 train / 391 calibration / 251 validation |
| German rows | 6,846 train / 448 calibration / 289 validation |
| Total rows | 26,043 |
| Question types | choice, yes/no, ordered scores, abstain |
| Hugging Face downloads | 262 (as of 2026-10-15) |
| License | mixed permissive (see record-level `license` field) |

## Project links

- [JevAlt project site](https://jevalt.mertkayacs.com)
- [GitHub repository](https://github.com/mertkayacs/jevalt)
- [Deem-4B](https://huggingface.co/mertkayacs/Deem-4B)
- [Karar-4B](https://huggingface.co/mertkayacs/Karar-4B)
- [Wähler-4B](https://huggingface.co/mertkayacs/Wahler-4B)
- [Benchmark](https://huggingface.co/datasets/mertkayacs/jevalt-bench)
- [JevAlt Space](https://huggingface.co/spaces/mertkayacs/JevAlt)
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
## Load the data

```python
from datasets import load_dataset

data = load_dataset("mertkayacs/jevalt-data", "tr")
print(data["train"][0])
```

Use `en`, `tr` or `de` as the configuration. The splits are `train`, `calibration` and `validation`. Held-out test rows are in [jevalt-bench](https://huggingface.co/datasets/mertkayacs/jevalt-bench).

| Language | Train | Calibration | Validation |
|---|---|---|---|
| English | 9,855 | 495 | 286 |
| Turkish | 7,182 | 391 | 251 |
| German | 6,846 | 448 | 289 |

Training rows by source:

| Source | English | Turkish | German |
|---|---:|---:|---:|
| Teacher-written scenarios | 1,287 | 1,741 | 1,408 |
| Emberwick game requests | 124 | 129 | 107 |
| Targeted fix sets | 1,256 | 1,197 | 1,257 |
| Converted public datasets | 7,188 | 4,115 | 4,074 |
| Rows with a reasoning trace | 125 | 196 | 129 |
| Rows with `unknown`, or its control twin | 48 | 62 | 72 |

Splits come from a keyed hash of each row's group. Derived rows, including negation twins and padded copies, stay with their source row.

## Records and provenance

Each JSONL record contains `state`, typed `questions` and `targets`: the answer label and a probability for each option. Questions are choices, yes/no decisions or ordered scores. Rows also record language, source, teacher distributions, tags and license; some include reasoning or an `unknown` option.

Sources include directly written native-language scenarios, Emberwick game requests, targeted examples for failure cases, and converted MASSIVE, Open-Jev, PAWS-X and typed-decisions records. Two teachers from other labs label each written scenario, and it stays only when both agree with the writer's selected answer. Related rows stay in one split through a group hash. [Data pipeline and training recipe](https://github.com/mertkayacs/jevalt/blob/main/docs/how-it-was-built.md).

### Writers and labels

Teacher models wrote each scenario directly in its language. GLM-5.x wrote the English set. Mistral Large 3 and DeepSeek V4 Pro wrote Turkish after scoring 4.67 and 4.50 out of 5 in a blind native-feel test. Gemma 4 26B-A4B and DeepSeek V4.1 Flash wrote German. Kimi K3 added 54 rows (41 English, 7 Turkish and 6 German), and MiniMax M3 added two English reasoning traces.

A seeded 30% sample of Turkish scenarios, plus every row that failed the language check, received an edit pass from another model family. Editors returned 725 rows and changed 356. German failures received the same treatment.

Two teachers label each written scenario, with three teachers on the 85 pilot rows. Labelers come from a different lab than the writer and editor. Each sees a separate shuffled option order and returns a probability per option. The target is `0.98 * mean + 0.02 * uniform`. A question stays when both teachers and the writer select the same answer. Every teacher distribution remains in the record.

### Targeted fix sets

| Set | Weak spot | Rows across all splits |
|---|---|---:|
| F1 | Missing deciding fact and `unknown` | 310 |
| F2 | Negation consistency | 368 |
| F3 | Noul versus equivalent two-option Choice | 358 |
| F5 | Dates, deadlines and business days | 390 |
| F6 | Numbers, sums and unit conversions | 383 |
| F7 | Long irrelevant padding | 1,188 |
| F8 | Prompt injection hidden in the state | 827 |
| F10 | Criteria that invert the question wording | 390 |
| F11 | Ordinal Score targets | 5 |
| F12 | Long policies with exceptions | 390 |

The Emberwick generator produced 2,663 distinct requests per language across ten seeded runs of four scenarios. A sample of 499 per language was labeled by Gemma 4 26B-A4B and Qwen3.5-122B-A10B. Near-duplicate removal kept 152 English, 153 Turkish and 133 German rows across all splits.

GLM-5.x or DeepSeek V4 Pro wrote short reasoning traces for date, number, inverted-criteria and policy rows. Traces stay only when their conclusion matches the answer. Turkish traces received a native edit that had to preserve every number and the conclusion. Two blind raters scored samples on a 1 to 5 native-feel scale:

| Turkish trace writer | Qwen3.5-122B rater | Gemma 4 26B rater |
|---|---:|---:|
| DeepSeek V4 Pro, before / after edit | 3.30 / 3.50 | 3.30 / 3.95 |
| GLM-5.x, before / after edit | 3.60 / 3.90 | 4.10 / 4.15 |

German traces written by GLM-5.x, without an edit pass, scored 3.93 and 4.13.

Public rows retain their source labels. An exact normalized-state match found no overlap with the evaluation suites except repeated MASSIVE utterances: 57 Turkish, 45 German and 5 English test utterances also occur in the MASSIVE train split. Exact deduplication, MinHash on character 5-grams at Jaccard 0.8, schema checks and language checks remove duplicate or invalid rows.

### Record shape

```json
{
  "id": "tr-g-...",
  "lang": "tr",
  "source": "gen:mistral",
  "state": "...",
  "questions": {"q": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": "..."}}},
  "targets": {"q": {"label": "a", "dist": {"a": 0.83, "b": 0.17}}},
  "reasoning": "...",
  "abstain": false,
  "tags": ["domain:...", "group:..."],
  "license": "apache-2.0"
}
```

`reasoning` and `abstain` appear where applicable. Setting `abstain` adds the `unknown` option.

## Licenses

The dataset has mixed licenses. Read each record's `license` field before reuse.

| Source | Terms |
|---|---|
| Written scenarios, game rows, fix sets and traces | Apache-2.0 |
| [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) | CC BY 4.0; credit Amazon.com Inc. or its affiliates |
| [Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) | CC0 |
| [PAWS-X](https://huggingface.co/datasets/google-research-datasets/paws-x) | Free for any use; credit Google LLC |
| [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), train only | Apache-2.0 |

Teacher models use Apache-2.0, MIT or the GLM-5.3 license, as recorded in the source card.

## Limits and models

Written scenarios use invented or replaced names, emails and numbers. Dropping teacher disagreements removes many ambiguous cases. Probability labels reflect teacher agreement, and reasoning traces can read stiffly. MASSIVE repeats some states across its own train and test sets.

[Deem-4B](https://huggingface.co/mertkayacs/Deem-4B) | [Karar-4B](https://huggingface.co/mertkayacs/Karar-4B) | [Wähler-4B](https://huggingface.co/mertkayacs/Wahler-4B) | [Comparison results](https://huggingface.co/datasets/mertkayacs/jevalt-bench/tree/main/results/comparison) | [Project](https://jevalt.mertkayacs.com).

An [Eschatia Labs](https://eschatialabs.com) project. [Mert Kaya](https://mertkayacs.com).
