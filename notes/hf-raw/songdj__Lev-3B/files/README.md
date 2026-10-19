---
license: apache-2.0
library_name: peft
base_model: Nanbeige/Nanbeige4.2-3B-Base
base_model_relation: adapter
language:
- en
tags:
- decision-model
- system-1.5
- looped-transformer
- lora
---
# Lev-3B

### Thinking fast and deep with looped decision models

[![Lev — one checkpoint, multiple depths](https://openlair.github.io/lev/assets/lev-social.png)](https://openlair.github.io/lev/)

[![System 1.5](https://img.shields.io/badge/System_1.5-Decision_Model-2847E8?style=flat-square)](https://openlair.github.io/lev/)
![Native depths](https://img.shields.io/badge/Native_depths-1--2-2847E8?style=flat-square)
[![Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-596268?style=flat-square)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2610.20951-B31B1B?style=flat-square)](https://arxiv.org/abs/2610.20951)
[![Discuss on alphaXiv](https://img.shields.io/badge/alphaXiv-Discuss-A63D40?style=flat-square)](https://www.alphaxiv.org/abs/2610.20951)
[![Code on GitHub](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github)](https://github.com/OpenLAIR/lev)

**[Project](https://openlair.github.io/lev/) · [Paper on arXiv](https://arxiv.org/abs/2610.20951) · [PDF](https://arxiv.org/pdf/2610.20951) · [alphaXiv](https://www.alphaxiv.org/abs/2610.20951) · [GitHub](https://github.com/OpenLAIR/lev) · [Models & data](https://huggingface.co/collections/songdj/lev)**

**One checkpoint. Multiple depths. A probability for every option.**
Lev gives text decisions an adjustable compute budget: reuse the same recurrent weights, then read the shared decision head at the chosen depth. It returns typed option probabilities without decoding a chain of reasoning.

| Native depths | JevBench · depth 1 | JevBench · depth 2 | Package |
| :---: | :---: | :---: | :---: |
| **1–2** | **69.3%** | **72.7%** | LoRA adapter + decision head |

*Paper measurements on 231 public JevBench items. See the full table and measurement conditions below.*

[Results](#reported-results) · [Quick start](#quick-start) · [Model details](#model-and-training) · [Data](#data-availability) · [Limitations](#limitations) · [Citation](#paper)

## The Lev family

| Model | Backbone | Native depths |
| --- | --- | :---: |
| [Lev-1.4B](https://huggingface.co/songdj/Lev-1.4B) | Ouro-1.4B | 1–4 |
| [Lev-2.6B](https://huggingface.co/songdj/Lev-2.6B) | Ouro-2.6B | 1–4 |
| [Lev-3B](https://huggingface.co/songdj/Lev-3B) | Nanbeige4.2-3B-Base | 1–2 |

Each repository contains its seed-0 adapter, decision head, tokenizer assets, and per-depth temperature calibration. Base model weights are downloaded separately.

## Reported results

| Depth | Public JevBench accuracy | Hard-tier accuracy | H200 bf16 median latency (ms) |
| --- | --- | --- | --- |
| 1 | 69.3% | 46.8% | 23.0 |
| 2 | 72.7% | 49.5% | 44.7 |

These are the paper's existing measurements on 231 public JevBench items, not new evaluations of this package. Accuracy uses the study's evaluation path; timing is a separate, single-H200, bf16, one-request-at-a-time measurement without fused kernels or graph capture. Hardware and implementation changes may change latency. The checkpoint's `weights_dtype` is preserved from training; the actual evaluation kernel and backbone dtype must be matched for parity.

Lev-3B exceeded its fixed-depth frontier on locked test, but the result is workload-specific. Most controls and the 1.4B/3B headline models have one evaluated seed in that study.

## Quick start

Use the accompanying [Lev code](https://github.com/OpenLAIR/lev), whose import name remains `kev`. A standard Transformers text-generation pipeline or PEFT loader alone does not implement the decision head and loop readouts.

After installing the companion code, download this repository:

```python
from huggingface_hub import snapshot_download
checkpoint = snapshot_download("songdj/Lev-3B")
```

Then run the companion example with the downloaded local directory:

```bash
python examples/fixed_depth.py --checkpoint /path/to/downloaded-checkpoint \
  --depth 2 --temperatures /path/to/downloaded-checkpoint/temperatures.json
```

For a reproducible experiment, pin `snapshot_download` to the verified Hub commit hash shown in the repository history. `Checkpoint.load` uses the base identifier and immutable revision in `head.pt` to load the base tokenizer. The included tokenizer assets are preserved and cleaned for inspection and compatible local use.

### Calibration

`head.pt` retains its original scalar temperature of 1.0. **Per-depth calibration is a separate file, `temperatures.json`, and must be applied explicitly.** It was fitted on 1,148 questions from the decision-v7 calibration partition. The fixed-depth example applies `softmax(logits / T_depth)`; passing `--all-depths` computes every loop and returns all readouts.

## Model and training

- Base: [`Nanbeige/Nanbeige4.2-3B-Base`](https://huggingface.co/Nanbeige/Nanbeige4.2-3B-Base/tree/4a38e817c14b9f2c4b69b1e05cbda90395872f6b), pinned revision `4a38e817c14b9f2c4b69b1e05cbda90395872f6b`.
- Native loop depths: 1–2; shared weights and pointer head at each depth.
- LoRA rank 16, alpha 32, dropout 0.05; one epoch, learning rate 1e-4, effective batch 64, uniform loss across depths, and seed 0.
- This seed-0 model accepted 31,334 records after context filtering; the constructed source mixture contains 18,764 records plus 12,576 replay records before context admission. Max state length is 5,120, with none-pair augmentation 0.25.
- Source training arguments and local provenance strings are removed from `head.pt`; all inference metadata and every head tensor are preserved. Adapter tensors are copied byte for byte. `adapter_config.json` explicitly pins the same base revision as `head.pt`.

## Adaptive depth: offline analysis only

The paper's stopper predicts agreement with the deepest answer and is fitted on unlabelled requests from the target workload. Two-fold cross-fitting produces workload-specific classifiers and thresholds; there is no universal trained stopper in this model package. The companion `scripts/stopper_config.py` supports offline fitting and application. **Live per-question stopping and continuation are not implemented in the current release code.** Reporting fewer simulated loop steps is not a measured serving speedup.

## Data availability

The [Lev collection](https://huggingface.co/collections/songdj/lev) contains **three models and three dataset repositories**. Dataset packages are versioned separately from the original frozen inputs:

| Resource | Purpose | Notes |
| --- | --- | --- |
| [LevBench](https://huggingface.co/datasets/songdj/LevBench) | Development/test source configurations | Rights-filtered subset |
| [Lev-train-v2](https://huggingface.co/datasets/songdj/Lev-train-v2) | S31/S60/S120, replay and order recipes | Rights-filtered subset |
| [Lev-results](https://huggingface.co/datasets/songdj/Lev-results) | Existing numerical results, calibration and latency | Question text removed |

Each dataset card reports hosted counts, missing sources, pinned revisions, applicable licenses, and file hashes. These subsets do not reproduce the complete frozen corpus; unresolved sources require lawful local acquisition. No new evaluation was run for packaging.

See the companion [data documentation](https://github.com/OpenLAIR/lev/blob/main/docs/DATA.md) and [dataset cards](https://github.com/OpenLAIR/lev/tree/main/data-cards) for reconstruction and distribution requirements.

## Limitations

Extra depth helps mostly on harder items and can also overturn a correct answer. Agreement with the deepest answer does not guarantee correctness. Temperature calibration and stopping behavior depend on the workload. The low-latency advantage of small Lev models partly reflects gaps in the dense baselines; at matched FLOPs on development data, Lev is at or below those alternatives. The study covers 1.4B–3B backbones and the public JevBench items.

`validation.json` records the package checks: decision-head metadata and tensors, and byte-identical adapter and tokenizer files. `manifest.json` gives SHA-256 hashes of the packaged artifacts. Packaging did not read test examples or training-question text and did not rerun evaluations.

## License and upstream use statement

The pinned upstream model card declares Apache-2.0.

The base card at the pinned revision is the provenance for these statements; no separate LICENSE or NOTICE file was listed in that revision when this package was prepared. The Apache-2.0 text and attribution notices accompany this derived adapter/head package. Base weights and third-party data retain their own terms; no dataset text is included. See `provenance.json` for the pinned card digest and source artifact hashes.

## Paper

**Lev: Thinking Fast and Deep with Looped Decision Models** — Dingjie Song, Zhiling Yan, Yuxuan Zhang, and Lichao Sun.

Read the paper on [arXiv](https://arxiv.org/abs/2610.20951) ([PDF](https://arxiv.org/pdf/2610.20951)) or discuss it on [alphaXiv](https://www.alphaxiv.org/abs/2610.20951). Code is on [GitHub](https://github.com/OpenLAIR/lev) and the research overview is on the [project page](https://openlair.github.io/lev/).

```bibtex
@misc{song2026levthinkingfastdeep,
      title={LEV: Thinking Fast and Deep with Looped Decision Models},
      author={Dingjie Song and Zhiling Yan and Yuxuan Zhang and Lichao Sun},
      year={2026},
      eprint={2610.20951},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.20951},
}
```
