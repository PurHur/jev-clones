---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
library_name: pytorch
datasets:
- Valen-Team/Valen-Training-General-100k
- Valen-Team/Valen-Eval-Game
tags:
- valen
- sokoban
- multimodal
- decision-making
- reinforcement-learning
pipeline_tag: image-text-to-text
---

<p align="center">
  <img src="card-assets/valen-logo-v5-compact.svg" alt="Valen — a teal aperture with a forward arrow and uppercase wordmark" width="720"><br>
  <img src="card-assets/valen-slogan.svg" alt="System One Model, now with vision." width="700">
</p>

# Valen-Preview-0923

<p align="center">
  <a href="https://github.com/Liuziyu77/Valen">GitHub</a> · <a href="https://github.com/Liuziyu77/Valen/blob/main/README_zh.md">中文 README</a> · <a href="https://huggingface.co/Valen-Team">Hugging Face</a> · <a href="https://github.com/Liuziyu77/Valen/blob/main/docs/technical.md">Technical notes</a>
</p>

## ✨ Introduction

Valen brings visual perception to System One decision-making. Inspired by [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), it evaluates text, images and video against task instructions and returns probabilities over supplied candidates, without generating answer tokens.

This repository hosts the **2B Sokoban-trained preview**, with the same checkpoint as [Valen-Sokoban-RLCD-2B](https://huggingface.co/Valen-Team/Valen-Sokoban-RLCD-2B). It builds on Qwen3.5-2B and supports the project's Choice, Noul and Score decision interface. The Valen name comes from Leigh Van Valen's Red Queen hypothesis: continual evolution is necessary to keep pace.

Training starts from the General 100k SFT decision head, then runs RLCD on 30,000 Sokoban records for three epochs using `vision_top` (90,000 record exposures). The checkpoint includes the decision head, language-model LoRA, visual merger and final four visual blocks, plus optimizer and training state. It does not duplicate the frozen backbone; both downloads below are required.

## 🎬 Demos

<a id="demo-comparison"></a>

### Against a 27B generation model

On the same Sokoban level with a 10-step cap, **Valen-Preview-0923** solves the puzzle in 9 decisions with 1.13 seconds of cumulative decision latency. Qwen3.8-27B-FP8 takes 198.05 seconds in thinking mode and fails to solve it in no-thinking mode. The thinking replay runs at 20× speed; the other two run at 1×.

<p align="center">
  <img src="card-assets/sokoban-model-comparison.gif" alt="Side-by-side Sokoban demo comparing Valen-Preview-0923 with Qwen3.8-27B-FP8 in no-thinking and thinking modes." width="1000"><br>
</p>

### Image blur and action confidence

Nine Gaussian blur levels show how **Valen-Preview-0923** changes its action probabilities and decision confidence as visual information becomes less clear. On the clear image, the optimal action `DOWN` has 93.7% probability and decision confidence is 91.6%; at 72 px blur, these values are 39.4% and 19.2%.

<p align="center">
  <img src="card-assets/valen-preview-0923-blur-confidence.gif" alt="Valen-Preview-0923 action probabilities and decision confidence across nine measured levels of Gaussian image blur." width="1000"><br>
  <sub>Each plot dot is one model inference; values between samples are interpolated for playback.</sub>
</p>

### Four games in parallel

Four successful **Valen-Preview-0923** trajectories run side by side at 1× speed with no playback acceleration. Each game takes 7–10 decisions, averaging 122–128 ms per step, and all four finish within 1.24 seconds.

<p align="center">
  <img src="card-assets/sokoban-four-game-showcase.gif" alt="Four successful Valen-Preview-0923 games replayed in parallel at recorded speed." width="1000"><br>
</p>

## 📥 Model downloads

| Model | Role | Download |
| --- | --- | --- |
| Valen-Preview-0923 | Trained Valen checkpoint | [🤗 This repository](https://huggingface.co/Valen-Team/Valen-Preview-0923/tree/main) |
| Qwen3.5-2B | Frozen base model | [🤗 Hugging Face](https://huggingface.co/Qwen/Qwen3.5-2B) |

Required base revision: `15852e8c16360a2fea060d615a32b45270f8a8fc`. The setup script in Quick start downloads and verifies that revision.

Datasets: [General 100k training](https://huggingface.co/datasets/Valen-Team/Valen-Training-General-100k) · [General 5k evaluation](https://huggingface.co/datasets/Valen-Team/Valen-Eval-General-5k) · [Sokoban training and evaluation](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game).

## 📊 Evaluation results

The following results apply to **this Preview checkpoint**:

| Evaluation | Result |
| --- | ---: |
| 500 single-step Sokoban action questions | 87.60% accuracy |
| 100 complete Sokoban games, at most 200 moves | 38/100 solved |

These are the selected simple Sokoban levels: 35 easy and 65 medium. The set retained all successful 2B RLCD cases before filling to 100, so it is model-outcome-conditioned, not an unbiased full benchmark. Single-step accuracy and complete-game success are different metrics.

The figure below provides context across the **Valen model family**. General is on the left and Sokoban is on the right; each panel uses separately trained checkpoints. The General results are not measurements of this Sokoban preview. `RL` denotes RLCD, and the Preview corresponds to `Valen-2B(RL)` in the Sokoban panel.

<p align="center">
  <img src="card-assets/evaluation-overview.png" alt="General and Sokoban accuracy versus mean end-to-end latency across the Valen model family." width="1200">
</p>

Full training details, Model Cards, loss curves and timing conditions are in the [technical notes](https://github.com/Liuziyu77/Valen/blob/main/docs/technical.md#完整实验).

## 🚀 Quick start

Use Linux, Python 3.10+ and an NVIDIA GPU. Install the [Valen](https://github.com/Liuziyu77/Valen) environment, then download both the Preview checkpoint and the matching base model:

```bash
git clone https://github.com/Liuziyu77/Valen.git Valen
cd Valen
bash scripts/setup/bootstrap.sh
source .venv/bin/activate

hf download Valen-Team/Valen-Preview-0923 --local-dir models/Valen-Preview-0923
python scripts/setup/prepare_model.py

python -m valen.inference \
  --checkpoint models/Valen-Preview-0923 \
  --data data/smoke/train.jsonl \
  --output predictions.jsonl
```

`data/smoke` contains synthetic examples for checking the pipeline. Replace it with your own JSONL file using the [documented record format](https://github.com/Liuziyu77/Valen/blob/main/docs/data-format.md) for your task.

The published `config.json` uses `model_path=models/Qwen3.5-2B` and `stage=vision_top`. Run from the Valen repository root and keep that stage so all trained visual and LoRA parameters are loaded. If you store the base model elsewhere, change only `model_path` in your local copy. This is the project's `checkpoint.pt` + `config.json` format, not an `AutoModel.from_pretrained` repository.

## Checkpoint provenance

The published `checkpoint.pt` and `config.json` are copied byte-for-byte from the completed three-epoch run. Training details and parameter counts are in [training_metadata.json](training_metadata.json).

Checkpoint SHA-256:

```text
836622efe78fe757e2627aa6050c223d461c424ea430a1c110c15e6f42f5a012
```

## 🤝 Contributions

Contributions to Valen are welcome. Open an [issue](https://github.com/Liuziyu77/Valen/issues) to report a problem, share a use case or discuss experimental results. Submit a [pull request](https://github.com/Liuziyu77/Valen/pulls) to improve the code or documentation, contribute training data or add evaluation tasks.

Scan the QR code below to join the Valen WeChat group, discuss the project and share your experiments.

<p align="center">
  <img src="card-assets/wechat_0930.jpg" alt="QR code for the Valen WeChat discussion group" width="200">
</p>

## 📄 License and acknowledgments

This repository retains its [Apache 2.0](LICENSE) license. The Qwen3.5 base model and source datasets follow their respective licenses. Built on [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B), with the decision interface inspired by [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
