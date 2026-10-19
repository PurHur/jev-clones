---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
library_name: peft
pipeline_tag: image-text-to-text
language:
- en
tags:
- typed-decisions
- jev
- decision-model
- vision-language
- calibrated-probabilities
- lora
- mlx
---

<p align="center">
  <img src="assets/logo-light.png" class="block dark:hidden" width="300" alt="imajev">
  <img src="assets/logo-dark.png" class="hidden dark:block" width="300" alt="imajev">
</p>

<h3 align="center">Decisions for real-world cases.</h3>

<p align="center">Small open models that read the photos, records and text a business already has and answer in the options you set,
with a probability on each and an explicit <i>can't tell</i>. Your system acts when it is sure and hands the rest to a person.</p>

<p align="center">
  <a href="https://huggingface.co/spaces/mohit67890/imajev"><img alt="Try the live demo" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg"></a>
  <a href="https://github.com/mohit67890/imajev"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-mohit67890%2Fimajev-111111?style=flat-square&logo=github"></a>
  <a href="https://mohit67890.github.io/imajev/"><img alt="Website" src="https://img.shields.io/badge/website-checked%20examples-555555?style=flat-square"></a>
  <a href="https://huggingface.co/datasets/mohit67890/imajev-bench"><img alt="ImajevBench" src="https://img.shields.io/badge/benchmark-ImajevBench-555555?style=flat-square"></a>
  <img alt="Apache-2.0" src="https://img.shields.io/badge/licence-Apache--2.0-111111?style=flat-square">
</p>

<p align="center"><b>imajev-9b</b> is the largest size of the family · other sizes: <a href="https://huggingface.co/mohit67890/imajev-2b">imajev-2b</a> · <a href="https://huggingface.co/mohit67890/imajev-4b">imajev-4b</a><br>
<a href="https://huggingface.co/spaces/mohit67890/imajev"><b>Live demo</b></a> · <a href="https://mohit67890.github.io/imajev/">Website</a> ·
<a href="https://github.com/mohit67890/imajev">Code and results</a> · <a href="https://mohit67890.github.io/imajev/report/">Technical report</a></p>

<p align="center"><a href="https://benchmarkheaven.com/jev-models"><img src="assets/ranks-light.png" class="block dark:hidden" width="820" alt="#1 of 91 on JevBench v1.4.2.2 (scored 27 Sep 2026); #1 of 49 on Image JevBench v0.1.3 (released 28 Sep 2026); #3 of 56 on DecisionBench (eng, v1), 28 Sep 2026."><img src="assets/ranks-dark.png" class="hidden dark:block" width="820" alt="#1 of 91 on JevBench v1.4.2.2; #3 of 56 on DecisionBench"></a></p>
<p align="center"><sub>Ranks of the family's <a href="https://huggingface.co/mohit67890/imajev-4b">imajev-4b</a>. This 9B has no official text-board entry.</sub></p>

<p align="center"><b>Independent results</b> (screenshots of the official leaderboards, 28 Sep 2026)</p>
<p align="center"><a href="https://benchmarkheaven.com/jev-models"><img src="assets/proof-jevbench-score.png" width="760" alt="JevBench v1.4.2.2 Composite Score: 1 Imajev-4B 67.4, 2 Plumb-4B 65.8, 3 decider-4b v2 64.1, 4 Jev 1.13.0 63.3"></a></p>
<p align="center"><a href="https://benchmarkheaven.com/image-jev-bench"><img src="assets/proof-image-jevbench.png" width="760" alt="Image JevBench v0.1.3 composite: 1 Imajev-4B 76.39, 2 Jev-Omni 73.10, 3 NeoHorse Jev 4B 71.94"></a></p>
<p align="center"><a href="https://huggingface.co/spaces/Hanno-Labs/decision-bench-leaderboard"><img src="assets/proof-decisionbench.png" width="760" alt="DecisionBench (eng, v1): 1 bosun-v3.1-1.7b 87.29, 2 bosun-v3.1-0.6b 83.20, 3 imajev-4b 79.65, ahead of glm-5.3-flash, jev-1.13, deepseek-v4.1-flash, gpt-5.6-luna"></a></p>
<p align="center"><sub>The family's imajev-4b: JevBench v1.4.2.2, scored 27 Sep 2026 (<a href="https://benchmarkheaven.com/jev-models">board</a>). Image JevBench v0.1.3, released 28 Sep 2026: #1 of 49, 76.39, ahead of Jev-Omni (<a href="https://benchmarkheaven.com/image-jev-bench">board</a>). DecisionBench (eng, v1): #3 of 56 models, 79.65, ahead of GLM-5.3 Flash (320B), Jev 1.13, DeepSeek V4.1 Flash (552B) and GPT-5.6 Luna; the two above are the benchmark team's own models (<a href="https://huggingface.co/spaces/Hanno-Labs/decision-bench-leaderboard">leaderboard</a>).</sub></p>

<p align="center">Built by <b>Mohit Garg</b>. I build decision systems like this for operations, support and back-office teams:<br>
refunds, returns, ticket routing, document and photo checks. <a href="https://mohitgarg.ai"><b>Work with me → mohitgarg.ai</b></a></p>

<p align="center"><img src="assets/request-listing.png" class="block dark:hidden" alt="imajev-4b checks a listing against its photo: listing.color says red, the photo shows beige shoes; the model names listing.color at 0.999 and the app holds the listing"><img src="assets/request-listing-dark.png" class="hidden dark:block" alt="imajev-4b checks a listing against its photo: listing.color says red, the photo shows beige shoes; the model names listing.color at 0.999 and the app holds the listing"></p>

imajev-9b is the largest size: it automates the most decisions and is strongest on knowledge-heavy text (MMLU 79.2%, measured on the previous imajev-9b).


> **2026-09-26:** the 4B tier moved to a new adapter (phase 3: ImajevBench 83.9%, DecisionBench full suite 79.7%, JevBench hard 72.1% as shipped) and is now the family's best size on ImajevBench; this 9B adapter is unchanged and remains the previous generation. See https://huggingface.co/mohit67890/imajev-4b.

## What sets it apart

<p><img src="assets/highlights.png" class="block dark:hidden" alt="Five highlights: a photo read against your record; two photos, one decision; a trained can't tell; open, small and local; Jev's contract, now with images"><img src="assets/highlights-dark.png" class="hidden dark:block" alt="Five highlights: a photo read against your record; two photos, one decision; a trained can't tell; open, small and local; Jev's contract, now with images"></p>

- **A photo read against your record.** Checks a photo against your own fields and names the one that is wrong. Trained on 72k
  photo-vs-record and two-photo decisions.
- **Two photos, one decision.** A reference and a target in the same request: shipped against returned, a known-good part against
  the one on the line.
- **A trained *can't tell*.** Every answer carries a probability for `unknown`, so the app can stop instead of guessing.
- **Open, small and local.** Apache-2.0, MLX on a Mac or PyTorch on one GPU; photos and customer data never leave your network.
- **Jev's contract, now with images.** TypeSafe's Jev request and response (`POST /v1/systemone`), plus `images`,
  `unknown_probability` and `abstained`. Jev itself is text-only and hosted; its state limit (32k tokens) is larger than imajev's (32 KB).

## One request, every answer typed

The exact script we ran against imajev-4b and its output (rounded, `usage` shortened); 1.15 s on a Mac Studio (four option orders averaged, calibration file applied). Swap the adapter for
this size and the request is unchanged.

```python
import json, requests

URL = "http://127.0.0.1:8765/v1/systemone"

listing = {
    "title": "Men's suede boat shoes",
    "color": "red",
    "product_type": "shoe",
}

questions = {
    "contradicted_field": {
        "type": "choice",
        "instructions":
            "Which field of `listing` does this photo contradict?",
        "criteria": {
            "listing.color": None,
            "listing.product_type": None,
            "none of these": "the photo agrees with every field",
        },
    },
    "color_matches": {
        "type": "noul",
        "instructions":
            "The product in the photo matches `listing.color`.",
    },
    "type_matches": {
        "type": "noul",
        "instructions": "The photo shows the kind of product "
                        "given in `listing.product_type`.",
    },
}

request = {"state": {"listing": listing}, "questions": questions}
with open("listing.jpg", "rb") as photo:
    r = requests.post(URL, files={"image": photo},
                      data={"request": json.dumps(request)})
print(json.dumps(r.json(), indent=2))
```

<details><summary>Result</summary>

```json
{
  "model": "imajev-4b",
  "answers": {
    "contradicted_field": {
      "type": "choice",
      "choice": "listing.color",
      "probabilities": {
        "listing.color": 0.95,
        "listing.product_type": 0.006,
        "none of these": 0.043
      },
      "confidence": 0.919,
      "unknown_probability": 0.007,
      "abstained": false
    },
    "color_matches": {
      "type": "noul",
      "noul": 0.082,
      "unknown_probability": 0.022,
      "abstained": false
    },
    "type_matches": {
      "type": "noul",
      "noul": 0.989,
      "unknown_probability": 0.004,
      "abstained": false
    }
  },
  "usage": {
    "total_ms": 1152.9,
    "input_tokens": 224
  }
}
```
</details>

<p><img src="assets/request-ticket.png" class="block dark:hidden" alt="A support ticket answered in one text-only request: department, urgency and frustration"><img src="assets/request-ticket-dark.png" class="hidden dark:block" alt="A support ticket answered in one text-only request: department, urgency and frustration"></p>

## Automate what is clear, route the rest

imajev-9b on the 279 ImajevBench test questions (photos, records and text; 21 whose honest answer is *can't tell*), raw
probabilities, scored with the benchmark's own rule:

| Act automatically when at least… | Decisions automated | Automatic decisions right |
|---|---:|---:|
| 80% sure | 77% | 87.9% |
| 90% sure | 70% | 91.8% |
| 99% sure | 52% | 99.3% |

The rest go to a person. The benchmark is built to be hard; measure on a few hundred of your own cases before choosing a threshold.
Other sizes at 90%: 2B 38% automated at 95.3% right, 4B 58% at 94.5%, 9B 70% at 91.8%.

## How it was made

About a million training decisions across the family, in four stages, for about $676 of rented GPU time for the whole project.
The 9B was trained on the 504k human-labelled decisions, then on about 23k hard questions kept only when open-weight teachers agreed, then a soft-target continuation on 39,515 rows carrying Qwen3.6-35B-A3B's full probability distributions (with the strict slice of the Eikos decisions set (`caiovicentino1/eikos-decisions`, CC-BY-4.0; attribution and per-source licences in `docs/eikos-decisions-usage.md`) and 5k replayed image decisions). The shipped adapter is the weight-space average of two adapters: the hard-question adapter and that continuation. It skipped stage 2 because it produced those labels for the 2B and 4B. Every teacher is open-weight; no Jev outputs, paid-API outputs or JevBench items were used.

## This adapter

This repository holds the **9B adapter**, the quality tier. It is a LoRA (rank 16, alpha 32) on the language layers of
Qwen3.5-9B (revision `c2022362`) plus a 255-code decision readout, in PEFT format at the root and in MLX format under `mlx/`;
the weights are the element-wise average (0.5 / 0.5, LoRA matrices and readout) of the hard-question adapter and its soft-target continuation.
Code, server and evaluation harness: https://github.com/mohit67890/imajev.
Other tiers: https://huggingface.co/mohit67890/imajev-4b (recommended default), https://huggingface.co/mohit67890/imajev-2b (latency).

**Which size?** On our measurements the 4B is within noise of the 9B on ImajevBench (82.4% vs 82.1%; a paired test on the previous
versions gave p = 1.0) and one item ahead on JevBench hard (70.3% vs 69.4%), at roughly half the memory. Pick the 9B for knowledge-heavy text questions and when memory is
not a constraint; otherwise start with the 4B.

## Technical specification

| | |
|---|---|
| Base model | Qwen/Qwen3.5-9B, revision `c2022362` (Apache-2.0) |
| LoRA | rank 16, alpha 32, dropout 0, no bias, on every language-model projection: `q,k,v,o`, `gate,up,down` and the DeltaNet `in_proj_qkv`, `in_proj_z`, `out_proj`; vision encoder frozen, no LoRA |
| Decision readout | one bias-free linear layer, 255 × 4096, float32 |
| Trainable parameters | 40,108,032 LoRA + 1,044,480 readout = 41,152,512 |
| Files | `adapter_model.safetensors` 160.5 MB (F32); readout 4.2 MB |
| Precision | base weights bfloat16; LoRA and readout float32 (MLX copies under `mlx/` converted from the same files) |
| Request limits | 0–2 images (resized to at most 400,000 pixels), state up to 32 KB, 1–8 questions, 2–254 options per `choice`, 2–10 levels per `score`, at most 4,096 tokens (longer requests are refused, not truncated); English only |
| Calibration | one temperature, 1.748, fitted on 150 template-generated JevBench-style items (none from JevBench), also used to pick checkpoints |

**Training path.** One trainer for every stage (PyTorch + PEFT): cross-entropy on the readout logits (soft targets where a
record carries a distribution), AdamW with weight decay 0, linear warm-up then cosine decay to 10% of the peak rate, gradient
clipping 1.0, seed 0, 4 GPUs. Gradient checkpointing in stages 3 and 4 only. The soft-target stage adds a rationale loss (weight 0.3,
at most 192 tokens) and permutes the options of every question.

| Stage | Started from | Epochs | Peak LR | Steps (kept / total) | Hardware | Time |
|---|---|---|---|---|---|---|
| Stage 1 | Qwen3.5-9B | 1 | 2e-4 | 3,100 / 3,508 | 4×H200 | 2.6 h |
| Stage 3, round 1 | stage 1 | 2 | 3e-5 | 300 / 404 | 4×H200 | 22 min |
| Stage 3, round 2 | round 1 | 2 | 2e-5 | 260 / 365 | 4×H100 | 20 min (27 min wall) |
| Stage 4, soft-target continuation | round 2 | 2 | 2e-5 | best on dev / 1,018 | 4×H100 | 1.7 h |
| Weight-space average | ½ round 2 + ½ stage 4, element-wise (LoRA and readout) | – | – | – | – | – |

Stage 1 crashed eight times in its first 980 steps (a diagnostic timer in the trainer, since removed) and resumed from 20-step
checkpoints; no data was skipped.

**Data this size saw.**

- **Stage 1:** 504,000 decisions from 36 licence-admitted sources (15 text, 21 image), including 4,000 photo-vs-listing contradictions.
- **Stage 2:** skipped; the 9B (after stage 1) labelled the stage-2 data for the 2B and 4B.
- **Stage 3, round 1:** 14,112 training records: kept teacher questions (9,368 of 13,386 kept on two-answerer agreement) plus the
  training share of 8,532 human reasoning items from 10 licensed sets.
- **Stage 3, round 2:** 7,812 training records: 3,598 new (4,852 of 8,097 kept on three-answerer agreement) + 4,214 replayed from round 1.
- **Stage 4:** 39,515 records: the stage-3 teacher questions relabelled with Qwen3.6-35B-A3B's probability distributions (thinking
  mode), 9,880 new hard, judge and programmatic questions, the strict slice of the Eikos decisions set (10,570 rows, open-weight
  teachers only) and 5,000 replayed image decisions.

**Compute.** $499.07 of rented GPU time on RunPod through stage 3 plus about $177 for stage 4 (8×H100, 6 h 20 min, all three sizes):
about $676 for the whole project, every run included.

Full specification: https://github.com/mohit67890/imajev/blob/main/docs/technical-specification.md

## Results (2026-09-24; every number reproducible from the code repository's `results/`)

| Benchmark | imajev-9b | Notes |
|---|---|---|
| JevBench public hard (111) | **69.4%** served (4 option rotations + `calibration.json`), 69.4% raw | same protocol, our runs: JevK5 v0.2.0 73.9%, Eikos-4B 73.9%, Hopper 67.6%, imajev-4b 70.3%; the previous imajev-9b 68.5% raw, 69.4% with rotations |
| JevBench public original (72) / easy (48) | 100% / 100% | |
| JevBench hard ECE | 0.187 raw, 0.092 served with `calibration.json` and 4 rotations | the raw model is over-confident on hard items; the previous imajev-9b 0.106 |
| ImajevBench v2.0-lite test (279: text, photo, photo+state) | **82.1%** (229/279), 95% CI [0.76, 0.88]; ECE 0.108 | Qwen3.5-9B base 76.7%; imajev-4b 82.4%; the previous imajev-9b 82.8%; frontier APIs 91–99.6% by structured generation |
| · text / visual / joint tracks | 29/37 · 104/120 · 96/122 | |
| · correct Unknown / false abstention | 15/21 · 2/258 | |
| ImajevBench private-1 hidden split (202; aggregates only) | 84.7% (171/202); text 24/30 · visual 80/84 · joint 67/88; ECE 0.079 | the previous imajev-9b 84.2% |
| MLX (Mac) vs PyTorch on ImajevBench | 82.4% on MLX vs 82.1% on PyTorch; 277 of 279 answers agree (99.3%) | parity check of the `mlx/` weights against the pod run |
| MMLU-1000, text-only / with an unrelated photo | 79.2% / 78.8% | measured on the previous imajev-9b, not re-run; an earlier imajev-9b: 73.8% / 72.1% |
| Irrelevance panel (2,823: MMLU with and without an unrelated photo, ABO, VizWiz) | 84.0% | the previous imajev-9b 83.6% (false abstention 0.7%, correct abstention 94.8%) |
| Hard-question test (435): correct on Unknown-gold rows / false abstention | 14/14 · 0.24% | ship gates |
| typed-decisions test (2,000) | 67.0% | measured on the previous imajev-9b, not re-run; an earlier imajev-9b: 66.2% |
| State probe (200) / pairs probe (60) | 73.5% / 90.0% | authored, templated |
| Reasoning dev (6,240 items; also used for checkpoint selection) | not measured for the shipped average; 68.9% for the soft-target checkpoint it averages, 67.4% for the previous imajev-9b | 69.2% before the last part of the hard-question stage (its measured trade-off) |
| p50 latency, JevBench hard item, 1×H100, serial | 96 ms raw (316 ms served with 4 rotations and calibration) | shared pod, under load |

Our pre-registered test against the untuned base model on ImajevBench (paired cluster sign-flip over 89 evidence clusters): the
shipped imajev-9b vs the untuned Qwen3.5-9B, +5.4 points (229 vs 214) [−1.2, +12.4], **p = 0.131, not significant at 0.05** (36 discordant clusters). The previous
imajev-9b (previous adapter) gave +6.1 [+0.0, +12.7], p = 0.074, also not significant; the earlier version of imajev-9b the test was registered with gave +4.3 points, p = 0.031. We report all three. JevBench is text-only; the image capability shows only on ImajevBench and in
use. The official JevBench board adds 308 sealed items and a four-axis score run by its maintainer; this 9B has no official
text-board entry. The family's imajev-4b is #1 of 91 there (JevBench v1.4.2.2, scored 27 Sep 2026; https://benchmarkheaven.com/jev-models).

Older panels were measured only on an earlier version of imajev-9b, before the hard-question stage, and are not repeated here as current numbers: held-out photo sources 55.9%,
real two-image pairs 26.6% (see `results/imajev-9b/`).

## Calibration
`calibration.json` (schema 1.1) applies one temperature (1.748) to every question type × option-count bucket, fitted by negative
log-likelihood on 150 template-generated JevBench-style items (none from JevBench), also used to pick checkpoints. Temperature scaling never changes an
answer, only its probability. `unknown` offsets are 0: bounded offsets were tested and changed no panel by more than 0.1 points.

Checked through the released server (ECE, uncalibrated → with `calibration.json`); the off-distribution rows were measured on the
previous imajev-9b with its own temperature (2.19) and have not been re-run for this version:

| Panel | ECE |
|---|---|
| JevBench public hard (served with 4 rotations, 111) | 0.187 raw → 0.092 |
| MMLU-1000, text-only | 0.123 → 0.031 |
| typed-decisions test (2,000) | 0.159 → 0.026 |
| SST-5 (2,210) | 0.211 → 0.029 |
| Photo-only verification (ABO + VizWiz, 823) | 0.027 → 0.053 |

Pooled ECE over all 231 public JevBench items in the served configuration: 0.050. An ECE-fit temperature would lower the hard-tier
ECE but raise the pooled ECE, as for the 4B and 2B, so the NLL fit is shipped.

On photo-only verification the raw probabilities are already calibrated and the temperature over-softens them; if your traffic is
mostly photo-against-record checks, serve without `--calibration` or fit your own temperature on a held-out sample.

## How to use

```sh
git clone https://github.com/mohit67890/imajev && cd imajev
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e ".[serve,mlx]"            # Apple silicon;  elsewhere: pip install -e ".[serve,torch]"
python scripts/download_model.py --model 9b
hf download mohit67890/imajev-9b --local-dir adapters/imajev-9b
# Mac (MLX)
PYTHONPATH=src:scripts python scripts/playground/server.py --model-bundle artifacts/model-qwen9b.json \
  --adapter adapters/imajev-9b/mlx --calibration adapters/imajev-9b/calibration.json --model-name imajev-9b --port 8767
# Linux / CUDA (PyTorch + PEFT)
PYTHONPATH=src:scripts python scripts/playground/server.py --backend torch --model-bundle artifacts/model-qwen9b.json \
  --adapter adapters/imajev-9b --calibration adapters/imajev-9b/calibration.json --model-name imajev-9b --port 8767
```

Then `POST /v1/systemone` with a Jev-shaped request (`state`, `questions`, optional `images`). `--rotations 4` averages four option
orders; the numbers above use it (for this version it changed no JevBench hard item, at ~3× latency). The 9B needs ~19 GB resident; do not keep it and another model loaded on the same Mac.

## Training
- Base: Qwen3.5-9B (Apache-2.0). LoRA r16/α32 on every language-model projection including the DeltaNet projections; 255
  single-token option codes read at the decision position through a float32 readout head; vision tower frozen.
- Licence-checked decisions (the 9B's first stage): one epoch over 504,000 human-labelled image and text decisions from licence-verified sources, lr 2e-4, 4×H200.
- Hard-question stage: 2 epochs, lr 3e-5, on 17,898 hard typed questions (documents written by Qwen3.6-27B, kept only when two
  answerers of different families, Qwen3.6-27B thinking and gpt-oss-20b, agree) plus licence-verified human reasoning sets.
  JevBench hard went from 42.3% to 67.6%: the first-stage recipe had erased the base model's reasoning (base 64.9%), and this data restored it.
- Last part of the hard-question stage: 2 epochs, lr 2e-5, on 9,066 rows: 4,852 new questions kept only when three open-weight answerers agree
  unanimously (adding Qwen3.6-35B-A3B thinking) plus 30% replay of the earlier hard questions. Selected on held-out dev rows.
- Soft-target stage: 2 epochs, lr 2e-5, on 39,515 rows: the hard questions relabelled with Qwen3.6-35B-A3B's full probability
  distributions (thinking mode), 9,880 new hard, judge and programmatic questions, the strict slice of the Eikos decisions set
  (10,570 rows, open-weight teachers only) and 5,000 replayed image decisions; soft-target cross-entropy plus a rationale loss (0.3)
  with option permutation. The shipped adapter is the weight-space average of the previous adapter and this stage's best checkpoint
  (0.5 / 0.5, LoRA matrices and readout): the checkpoint alone passes every gate but falls to 66.7% on JevBench hard.

## Data and licence posture
- Adapter, readout and code: Apache-2.0. Base model Qwen3.5-9B: Apache-2.0.
- No JevBench items (8-gram contamination lint against the public files), no outputs from Jev, and no outputs from any paid API
  were used in training. All teacher models are open-weight.
- Image and text sources carry per-source licence receipts in the code repository; 16 of the 21 image sources from stage 1 are used
  under their annotation licences only, with photos under upstream terms (not redistributed); for abo, vizwiz, vizwiz_quality
  and defects the grant covers the images too.

## Limitations
- Single pass, no reasoning at inference: multi-step arithmetic and answer-quality judging trail reasoning models (a frozen
  Qwen3.6-35B-A3B *with thinking* scores 97.3% on JevBench hard, at seconds and thousands of tokens per decision). The gap to JevK5
  is concentrated in judge-style items.
- Over-confident without `calibration.json`; fit your own temperature for unfamiliar domains.
- Not better than the 4B on ImajevBench; the last part of the hard-question stage lowered the score on our reasoning dev set (also used for checkpoint selection) by 1.8 points, the soft-target checkpoint recovers it to 68.9%, and the shipped average was not measured on that set.
- At most two images, 32 KB state, 254 options, 8 questions per request. English only. No free text.

## Intended use
Typed decisions inside applications: verification of a photo against a record, routing, extraction into fixed option sets,
abstention when evidence is missing. Not a safety classifier, not a certificate of correctness, and not for decisions about
people without human review.

## Citation
```
@software{imajev2026,
  author = {Garg, Mohit},
  title = {imajev: an open Jev-style typed decision model family for images and text},
  year = {2026},
  url = {https://github.com/mohit67890/imajev}
}
```
ImajevBench, the photo-and-text benchmark released alongside: https://huggingface.co/datasets/mohit67890/imajev-bench.
