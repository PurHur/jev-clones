---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: peft
tags:
- omni-modal
- decision-making
- qwen3.5
- lora
- v1.1
language:
- en
- zh
---

# OmniJev-4B — v1.1

Release date: 2026-09-26. [Code](https://github.com/tinnel123666888/OmniJev) · [中文说明](https://github.com/tinnel123666888/OmniJev/blob/main/README_zh.md) · [All metrics](results_v11_zh.md) · [Machine-readable reports](results_v11.json)

This is the **4B OmniJev decision adapter and heads**, requiring `Qwen/Qwen3.5-4B` separately. It produces typed choice/noul/score probabilities without generating answer text. All three sizes use the Qwen3.5 prefix-branch inference path.

```bash
git clone https://github.com/tinnel123666888/OmniJev && cd OmniJev
pip install -r requirements.txt
hf download tinnel123/OmniJev-4B --revision v1.1 --local-dir ckpt
hf download Qwen/Qwen3.5-4B --local-dir base
```

```python
from mso.infer import MSO1
model = MSO1("ckpt", "base")
answers = model.system_one(
    {"images": ["screen.png"]},
    {"visible": {"type": "noul", "instructions": "A person is visible."}},
)
```

Use the v1.1 code: it reads all still images as a numbered panel and applies the checkpoint's optional noul bias before temperature scaling. A video state remains a frame mosaic. Missing multi-image files raise an error. `MSO_NO_PANELS=1` explicitly restores the historical first-image path for controlled comparisons.

The repository contains LoRA weights, decision and ordinal heads, added-token embeddings, tokenizer and processor files. The full Qwen backbone is not bundled. See `release_manifest.json` for SHA-256 checksums. Earlier weights remain available through the `pre-v1.1-20260926` revision; `v1.1` pins this release.

> **Score correction (2026-09-26):** use the [per-dataset and question-type audit](dataset_scores_v11.md), including question counts and missing coverage. The old A-OKVQA `genmcq` aggregate used incorrectly generated yes/no labels and is withdrawn. Existing valid multiple-choice predictions score 56.67% / 68.40% / 77.07% for 0.8B / 2B / 4B (750 questions each); these remain custom project tasks, not official benchmark scores. The old overall macro/micro averages also include the invalid labels and must not be treated as validated overall performance. Historical raw reports are retained as evidence.
>
> Mario's mixed 56.05% is not next-action accuracy (43.52% on 193 questions), and the released replays do not establish closed-loop gameplay. OK-VQA 93.47% is candidate selection with the gold answer supplied, not official open-ended VQA. The old POPE row covers only its random subset; safety covers HaGRID. Base/SFT and decision models were not evaluated on identical samples.
>
> **v1.2 remains experimental:** [experimental action interface](https://github.com/tinnel123666888/OmniJev/blob/main/docs/v12_development.md). This repository still contains v1.1 weights.

## Results

| Model | Families | Questions | Macro accuracy % | Micro accuracy % | Mean family ECE % ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base 0.8B | 30 | 21,456 | 40.07 | 40.04 | 20.83 |
| SFT 0.8B | 30 | 41,951 | 47.86 | 48.09 | — |
| v1.1 0.8B | 30 | 41,975 | 64.52 | 65.40 | 4.30 |
| Base 2B | 30 | 21,456 | 36.30 | 35.94 | 34.21 |
| v1.1 2B | 30 | 41,975 | 64.46 | 65.42 | 6.27 |
| Base 4B | 30 | 21,456 | 40.12 | 39.79 | 30.04 |
| v1.1 4B | 30 | 41,975 | 70.00 | 70.82 | 5.89 |

| Family | Base 0.8B | SFT 0.8B | v1.1 0.8B | Base 2B | v1.1 2B | Base 4B | v1.1 4B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| androidcontrol | 29.49 | 40.73 | 71.70 | 21.54 | 71.84 | 28.81 | 77.30 |
| atari | 45.68 | 45.07 | 71.67 | 7.82 | 63.40 | 34.29 | 63.87 |
| audio | 20.71 | 44.53 | 51.07 | 35.39 | 42.01 | 34.16 | 53.08 |
| chess2 | 24.01 | 38.93 | 61.80 | 20.99 | 61.67 | 16.74 | 62.40 |
| chess3 | 5.62 | 13.00 | 20.00 | 6.17 | 15.13 | 5.49 | 22.67 |
| events | 39.09 | 48.53 | 85.87 | 48.56 | 86.47 | 55.56 | 89.13 |
| game | 41.15 | 47.33 | 83.51 | 17.15 | 80.32 | 15.36 | 90.36 |
| genmcq (invalid mixed labels; withdrawn) | — | — | — | — | — | — | — |
| gomoku | 52.54 | 45.67 | 69.80 | 18.24 | 68.27 | 16.60 | 70.60 |
| jat | 36.90 | 50.33 | 63.73 | 27.71 | 63.00 | 25.79 | 65.87 |
| jog_bridge | 28.12 | 28.80 | 43.84 | 24.14 | 48.43 | 29.36 | 45.97 |
| jog_full | 24.69 | 27.60 | 49.24 | 32.24 | 49.04 | 28.26 | 51.56 |
| longtext | 21.26 | 62.53 | 76.42 | 36.21 | 77.68 | 59.40 | 80.28 |
| lvb | 41.80 | 52.60 | 47.20 | 57.38 | 50.00 | 59.02 | 56.80 |
| mario | 27.43 | 40.14 | 53.88 | 52.13 | 54.69 | 46.78 | 56.05 |
| music | 34.98 | — | — | 32.92 | — | 34.71 | — |
| mutex | 47.33 | 50.73 | 73.33 | 25.10 | 70.73 | 25.24 | 79.07 |
| old | 53.91 | 61.60 | 67.54 | 58.44 | 66.06 | 58.02 | 70.76 |
| pilot_web | 39.51 | 37.20 | 67.13 | 37.04 | 66.47 | 42.80 | 75.80 |
| point_phone | 40.27 | 37.68 | 60.91 | 39.09 | 60.69 | 46.61 | 72.53 |
| pope | 85.46 | 84.20 | 66.73 | 82.44 | 66.87 | 86.42 | 84.73 |
| roboarena_wrist | 38.55 | 55.87 | 65.93 | 32.92 | 66.27 | 33.20 | 67.80 |
| robot_long | 54.87 | 49.80 | 79.79 | 29.77 | 78.46 | 29.36 | 85.04 |
| safety | 52.81 | 68.93 | 97.73 | 58.44 | 99.07 | 67.90 | 99.53 |
| snake | 44.03 | 47.00 | 83.60 | 34.57 | 82.87 | 44.86 | 84.47 |
| video | 39.09 | 43.47 | 57.51 | 38.41 | 59.10 | 49.11 | 62.81 |
| vqa | 79.15 | 76.47 | 63.87 | 83.26 | 79.60 | 85.73 | 93.47 |
| web | 41.02 | 48.27 | 66.87 | 31.55 | 67.53 | 31.28 | 76.80 |
| webtest | 40.47 | 48.40 | 67.13 | 26.20 | 67.87 | 30.45 | 77.40 |
| wiki | 41.84 | 47.80 | 68.00 | 39.78 | 68.53 | 33.74 | 72.13 |
| xiangqi | 8.09 | 12.87 | 22.27 | 6.31 | 18.07 | 10.15 | 23.67 |

## Protocol and limitations

The table reports existing evaluations, **not a matched-sample ablation**. Base models use candidate-answer probabilities (`A1_raw`), with at most 1,000 questions per family before dev/test splitting. OmniJev and SFT typically use about 1,500 questions per family. Sampling, option truncation and image inputs differ. The 30-family summary excludes music, for which only base results exist.

The released SFT baseline is 0.8B only. Its 41,951 generations were judged and audited: 97 incorrect explicit answer labels were accepted by the raw LLM judge. Audited accuracy is **48.0894%** (equal to the parser), rather than the uncorrected judge's 48.3207%.

v1.1 OmniJev uses numbered multi-image panels. The historical SFT and base evaluations used the first still image only. Multi-image SFT retraining and inference over the newly frozen 272,561-question set are **not completed**. The 20-step distributed smoke checkpoint is not part of this release.

These are project evaluation families, not uniform official benchmark scores. POPE and LongVideoBench were excluded from training; Mind2Web uses official splits. Other families include custom row-level holdouts; the event-timing family uses videos from the Charades-STA training set. Row-level exclusion does not establish episode-level separation.

The main table contains raw evaluation reports, **not a new evaluation of the calibrated serving outputs**. Released temperature metadata comes from a separate calibration split. The 4B checkpoint also contains `biases.noul=0.0531085661`, which moves the yes/no boundary; therefore its raw POPE result must not be restated as a serving-path result.

4B has the highest 30-family macro accuracy, but POPE (84.73% vs 86.42% base) and LVB (56.80% vs 59.02% base) remain lower. The 0.8B and 2B models also have substantial POPE regressions. All rows are retained.

## License and attribution

Apache-2.0 for the adapter and project code; the Qwen backbone retains its own license. Developed by Beijing Zhongguancun Academy, the Institute of Automation, Chinese Academy of Sciences, and Zevo.
