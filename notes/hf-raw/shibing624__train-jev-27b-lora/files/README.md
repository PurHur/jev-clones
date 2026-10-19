---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: adapter
library_name: peft
datasets:
- SargeDev/jev-distill-corpus-v3
- openai/gsm8k
- google-research-datasets/paws
- deepmind/aqua_rat
- allenai/qasc
- nvidia/HelpSteer2
tags:
- decision-model
- typed-decisions
- system-one
- lora
- knowledge-distillation
---

# train-jev-27b-lora v0.3.0

A **Qwen3.8-27B + LoRA + 24-slot decision head** for calibrated yes/no (`noul`), multiple-choice (`choice`) and rating (`score`) decisions. It returns probabilities without generating text.

**78.72% sealed PK accuracy** versus 73.47% for autotrust/JEV-27B-VL, with **0.032 transfer-v4 ECE** versus 0.097. LoRA has 108.8M parameters; inference uses an 80 GB GPU.

For a smaller model, use **[shibing624/train-jev-4b-lora](https://huggingface.co/shibing624/train-jev-4b-lora)**. Training, evaluation and serving code: **[train-jev](https://github.com/shibing624/train-jev)**.

## Core evaluation results

| Evaluation | train-jev-4b-lora<br>(Ours) | train-jev-27b-lora<br>(Ours) | Kev-4B | autotrust/JEV-27B-VL | TypeSafe Jev v1.13<br>(official API) |
|---|---:|---:|---:|---:|---:|
| [train-jev-pk](https://github.com/shibing624/train-jev/tree/main/data/train-jev-pk) accuracy (2,494 questions, task-macro) ↑ | 74.18% | **78.72%** | 63.50% | 73.47% | – |
| train-jev-pk ECE ↓ | 0.112 | **0.098** | 0.159 | 0.107 | – |
| [train-jev-classification-pk](https://github.com/shibing624/train-jev/tree/main/data/train-jev-classification-pk) accuracy (1,300 questions) ↑ | 0.642 | 0.687 | 0.640 | 0.690 | **0.700** |
| Classification PK ECE ↓ | 0.050 | **0.048** | 0.083 | 0.094 | 0.130 |
| Kev transfer-v4 accuracy (656 questions) ↑ | 0.805 | **0.858** | 0.817 | 0.852 | 0.857 |
| Kev transfer-v4 ECE ↓ | 0.054 | **0.032** | 0.042 | 0.097 | – |
| Decision Index 0.3, sampled score (6,000 questions) ↑ | 41.5 | 56.1 | 38.4 | 54.1 | **59.7** |
| Decision Index 0.3, full-panel score (110,087 questions) ↑ | 39.01 | **55.58** | – | – | – |

Decision Index is a composite score, not accuracy: the sample covers 35 benchmarks, and the full run covers only the public 37-benchmark text panel, not the private Full leaderboard. Sealed PK includes fresh items from training sources and rule families; transfer-v4 tests unseen sources. Some official API numbers come from the official leaderboard and Kev reports. Full protocols, sources and limitations: [results](https://github.com/shibing624/train-jev/blob/main/docs/results.en.md).

## Use

```bash
git clone https://github.com/shibing624/train-jev.git && cd train-jev
pip install -r requirements.txt
python -m trainjev.serve --base Qwen/Qwen3.8-27B \
  --adapter shibing624/train-jev-27b-lora --port 8020
```

The first run downloads the backbone, adapter, custom head and calibration. The service exposes `POST /v1/systemone`. Use the train-jev loader: this is a decision adapter, not a generative-chat LoRA. API examples and deployment options: [serving guide](https://github.com/shibing624/train-jev/blob/main/docs/serving.en.md).

## Training data and reproduction

Both model sizes use the same two-stage method: **KL distillation**, then **87.5% distillation replay + 12.5% new supervision per step**. Only LoRA and the decision head are trained; temperatures are fitted on an independent calibration set.

| Training data | Purpose | Processed training rows |
|---|---|---:|
| [SargeDev/jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) | Initial distillation and replay | 655,806 |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Mathematics choices with two distractor constructions | 11,148 |
| [PAWS](https://huggingface.co/datasets/google-research-datasets/paws), [AQuA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat), [QASC](https://huggingface.co/datasets/allenai/qasc), [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | Paraphrases, mathematics/science choices and correctness ratings | 15,999 |
| Independently verified rules (CC0) | Routing, evidence, coreference, refunds, calendars and counting | 10,000 |

From the cloned repository, start initial distillation with:

```bash
BASE_REPO=Qwen/Qwen3.8-27B BASE_DIR=models/Qwen3.8-27B bash scripts/prepare_data.sh
GPUS=0,1,2,3 bash scripts/run_train_27b.sh
```

For the published model's continued training, initialize from the `v0.2.0` adapter, mix the listed gold and rule data at 0.125, use LoRA/head learning rates of 2e-5 / 4e-5 and select on an independent development set. Exact settings and source provenance are in [training_recipe.json](training_recipe.json); commands and calibration are in the [training guide](https://github.com/shibing624/train-jev/blob/main/docs/training.en.md#4-gold-and-rule-refinement).

Processed new data and their construction scripts are not yet distributed, so the training method can be reused but exact released weights cannot currently be reproduced in one command.

## License

Apache-2.0 for this adapter and the code; upstream datasets retain their own licenses. PAWS requires Google LLC acknowledgement under its permissive terms; QASC and HelpSteer2 require CC-BY-4.0 attribution. Independent reproduction, not affiliated with TypeSafe AI or autotrust.
