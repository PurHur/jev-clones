---
base_model: Qwen/Qwen3.5-2B-Base
datasets:
- ccdv/arxiv-classification
- clinc/clinc_oos
- community-datasets/yahoo_answers_topics
- dair-ai/emotion
- fancyzhx/ag_news
- fancyzhx/dbpedia_14
- google-research-datasets/paws
- google/boolq
- google/civil_comments
- legacy-datasets/banking77
- mteb/amazon_massive_intent
- nyu-mll/glue
- osyvokon/pavlick-formality-scores
- papluca/language-identification
- qiaojin/PubMedQA
- raquiba/Sarcasm_News_Headline
- sealuzh/app_reviews
- SetFit/20_newsgroups
- SetFit/sst5
- SetFit/subj
- SetFit/TREC-QC
- tals/vitaminc
- tasksource/ruletaker
- ucberkeley-dlab/measuring-hate-speech
- ucirvine/sms_spam
- Yelp/yelp_review_full
- nvidia/HelpSteer2
- tasksource/Boardgame-QA
- hotpotqa/hotpot_qa
library_name: peft
license: apache-2.0
pipeline_tag: text-classification
tags:
- strands-decider
- decision-model
- hobson
- lora
- peft
- qwen3.5
- classification
- calibration
- typed-decisions
base_model_relation: adapter
model-index:
- name: hobson-2b-v21
  results:
  - task:
      type: text-classification
    dataset:
      name: JevBench public, served at 4096
      type: jevbench
      config: w4096
    metrics:
    - type: accuracy
      value: 0.7619
      name: accuracy (176/231)
  - task:
      type: text-classification
    dataset:
      name: 'eval: held-out short tasks'
      type: hobson-internal
      config: 'eval: held-out short tasks'
    metrics:
    - type: accuracy
      value: 0.65
      name: accuracy (n=6000)
  - task:
      type: text-classification
    dataset:
      name: 'eval: boardgame'
      type: hobson-internal
      config: 'eval: boardgame'
    metrics:
    - type: accuracy
      value: 0.821
      name: accuracy (n=900)
  - task:
      type: text-classification
    dataset:
      name: 'eval: contractnli'
      type: hobson-internal
      config: 'eval: contractnli'
    metrics:
    - type: accuracy
      value: 0.865
      name: accuracy (n=1026)
  - task:
      type: text-classification
    dataset:
      name: 'eval: hotpotqa (held out)'
      type: hobson-internal
      config: 'eval: hotpotqa (held out)'
    metrics:
    - type: accuracy
      value: 0.746
      name: accuracy (n=959)
  - task:
      type: text-classification
    dataset:
      name: 'eval: musique'
      type: hobson-internal
      config: 'eval: musique'
    metrics:
    - type: accuracy
      value: 0.882
      name: accuracy (n=1199)
  - task:
      type: text-classification
    dataset:
      name: 'eval: musique, answerable'
      type: hobson-internal
      config: 'eval: musique, answerable'
    metrics:
    - type: accuracy
      value: 0.886
      name: accuracy (n=599)
  - task:
      type: text-classification
    dataset:
      name: 'eval: musique, unanswerable'
      type: hobson-internal
      config: 'eval: musique, unanswerable'
    metrics:
    - type: accuracy
      value: 0.877
      name: accuracy (n=600)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [4] adequacy_hs2'
      type: hobson-internal
      config: 'eval: [4] adequacy_hs2'
    metrics:
    - type: accuracy
      value: 0.739
      name: accuracy (n=234)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [5] gen:adequacy'
      type: hobson-internal
      config: 'eval: [5] gen:adequacy'
    metrics:
    - type: accuracy
      value: 0.788
      name: accuracy (n=302)
---

# strands-decider-2B-hobson-v21

Strands decider is a small, fast decision model for agentic AI. Unlike an LLM, which can
generate arbitrary text, a decision model picks between sets of options and rates things on
a scale, and every answer carries a calibrated confidence. It fits the decisions inside an
agentic workflow: model routing, tool selection, argument checking, triage, guardrails,
evaluations, and the rote decisions of a hybrid agent that leaves the hard ones to an LLM.

v21 is v19's recipe plus checked question paraphrases and distillation from
`Qwen/Qwen3.5-4B` where it agrees with the gold label. It is the released seed of six, picked by
a rule written before the seeds were ranked ("How this checkpoint was chosen", below).

This repository holds one trained checkpoint: a LoRA adapter on `Qwen/Qwen3.5-2B-Base` plus a
small readout head that scores the options of a typed question (`noul`, a yes/no question;
`choice`, one of N options; `score`, a level on an ordered scale). The code, the training
recipe, the data inventory and the evaluations are at https://github.com/strands-labs/strands-decider, under the same
Apache-2.0 license as this model (see `LICENSE.md`).

## Use

```bash
pip install strands-decider
```

Ask one or more questions about a state. Pass `--device cuda`, `mps` or `cpu`; without it,
the best available device is used. The base weights download from the Hub at first use.

```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v21 \
  --state "Help! My payouts have been failing for 3 days! " \
  --choice "Which team should handle this?=billing,sales,retail" \
  --noul "Does this convey urgency?" \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```

The output of this checkpoint (`--device cuda`; other devices differ in the last digits):

```
noul_0 noul = 0.875
choice_0 -> billing (confidence 0.837)
  billing                  0.891
  sales                    0.055
  retail                   0.054
score_0 score = 1.07 (confidence 0.603)
  0: calm                                     0.140
  1: frustrated                               0.649
  2: depressed                                0.211
```

Images: `pip install "strands-decider[vision]"` (transformers 5.18 or later; the image path
is on the code repository's `main` and in releases after 0.1.0), then `serve --vision` or
`ask --image FILE`. See "Images" below and `docs/vision.md` in the code repository.

Or serve it over HTTP. The server binds to `127.0.0.1` and has no authentication: use it
for local experiments.

```bash
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v21 --port 8000
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days!",
  "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}
}'
```

In Python, `strands_decider.modeling.StrandsDeciderModel.load("StrandsAgents/strands-decider-2B-hobson-v21")`. The adapter in
`lora/` is a standard PEFT adapter on the Qwen3.5 text decoder
(`transformers.Qwen3_5ForCausalLM(...).model`); the head is `head.safetensors`.

## Results

| evaluation | tasks right | Brier | ECE | served |
| --- | --- | --- | --- | --- |
| JevBench public, window 4096 | 176/231 | 0.323 | 0.064 | these files |

| internal set | accuracy | n |
| --- | --- | --- |
| eval: held-out short tasks | 0.650 | 6,000 |
| eval: boardgame | 0.821 | 900 |
| eval: contractnli | 0.865 | 1,026 |
| eval: hotpotqa (held out) | 0.746 | 959 |
| eval: musique | 0.882 | 1,199 |
| eval: musique, answerable | 0.886 | 599 |
| eval: musique, unanswerable | 0.877 | 600 |
| eval: [4] adequacy_hs2 | 0.739 | 234 |
| eval: [5] gen:adequacy | 0.788 | 302 |

JevBench is the external benchmark (231 public tasks); the internal sets are this recipe's
held-out short tasks, multi-step documents and answer-adequacy judgements.
`evaluation/README.md` in the code repository describes each set, and its results pages
give the figures by version. Per-skill results and the raw reports and logs are in
`eval/summary.json` and `eval/internal/`.

## How this checkpoint was chosen

Six seeds of the same config trained on one host (8x H100). Each passes the validity gates
(231 JevBench tasks attempted, strict schema 1.000, easy tier 48/48, one epoch, every stage
exit code 0). The rule, written before the seeds were ranked: for each of five measures
(JevBench tasks right, MuSiQue, ContractNLI, BoardgameQA, HotpotQA), take the six-seed mean
and SD; a seed's distance is the root of the sum of its squared z-scores; release the seed
with the smallest distance. Seed 5 has distance 1.48 (next: seed 3, 1.93). So this is a
representative seed, not the best one on the public tasks. Training-time results:

| measure | s0 | s1 | s2 | s3 | s4 | s5 (released) | six-seed mean (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| JevBench public, tasks right of 231 (window 4096) | 177 | 171 | 173 | 177 | 176 | 176 | 175.0 (2.4) |
| JevBench Brier | 0.339 | 0.335 | 0.328 | 0.340 | 0.322 | 0.323 | 0.331 (0.008) |
| MuSiQue | 0.898 | 0.881 | 0.897 | 0.891 | 0.886 | 0.882 | 0.889 (0.007) |
| ContractNLI | 0.872 | 0.858 | 0.861 | 0.851 | 0.870 | 0.865 | 0.863 (0.008) |
| BoardgameQA | 0.804 | 0.820 | 0.818 | 0.810 | 0.793 | 0.821 | 0.811 (0.011) |
| HotpotQA (never trained on) | 0.750 | 0.751 | 0.730 | 0.764 | 0.771 | 0.746 | 0.752 (0.014) |
| held-out short tasks (6,000) | 0.642 | 0.645 | 0.645 | 0.636 | 0.645 | 0.650 | 0.644 (0.004) |
| HelpSteer2 adequacy (234) | 0.744 | 0.756 | 0.735 | 0.718 | 0.718 | 0.739 | 0.735 (0.015) |
| generated adequacy (302) | 0.818 | 0.811 | 0.811 | 0.798 | 0.785 | 0.788 | 0.802 (0.014) |
| generated documents, v16's set | 0.857 | 0.857 | 0.866 | 0.854 | 0.854 | 0.849 | 0.856 (0.006) |
| generated documents, v18's set | 0.761 | 0.757 | 0.745 | 0.773 | 0.765 | 0.741 | 0.757 (0.012) |

Against v19: on one other host, six seeds of v19's recipe averaged 172.8 JevBench tasks
(Brier 0.341) and six of this config 172.3 (Brier 0.331). Same accuracy, lower Brier score.
The published v19 is a single run (167/231 at window 3072, 168 at 4096), so do not read the
difference between the two single runs as a gain.

## Release checks

These files, through the code repository's `main` (not the training harness), on an NVIDIA
L40S (text) and an L4 (images), against the training-time run on H100:

| check | training time (H100, training harness) | release (`main`) |
| --- | --- | --- |
| JevBench public, tasks right (window 4096) | 176/231 | 176/231, the same answer on every task, from the training-format copy and from these files |
| JevBench Brier / ECE | 0.323 / 0.074 | 0.323 / 0.064 |
| MuSiQue / ContractNLI / BoardgameQA | 0.882 / 0.865 / 0.821 | 0.882 / 0.864 / 0.822 |
| HotpotQA (never trained on) | 0.746 | 0.745 |
| held-out short tasks (6,000) | 0.650 | 0.650 |
| generated adequacy (302) | 0.788 | 0.788 |
| image evaluations (below) | not run | run on these files with `--vision` |
| code: test suite (CPU), ruff and mypy, as CI runs them | | pass |

Every accuracy is within one item of its training-time figure. JevBench ECE, binned over
231 tasks, moves with small probability changes between GPUs.

## Images

With `--vision` the vision tower of Qwen3.5-2B-Base is kept and the adapter and head are
used unchanged; nothing was trained on images. Same evaluations as `docs/vision.md` in the
code repository (NaturalBench, first 300 groups, 1,200 questions; POPE adversarial, first
600 items), both checkpoints on the same L4:

| | NaturalBench acc | G-Acc | ECE | POPE-adv acc | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| v21 (this checkpoint), `--vision` | 0.785 | 0.313 | 0.043 | 0.878 | 0.179 | 0.036 |
| v19, `--vision` | 0.784 | 0.327 | 0.012 | 0.877 | 0.202 | 0.072 |
| v21 minus v19, paired bootstrap 95% CI | +0.001 (-0.012 to +0.014) | -0.013 | +0.031 (-0.004 to +0.048) | +0.002 (-0.012 to +0.015) | -0.023 (-0.033 to -0.012) | -0.035 (-0.057 to -0.009) |

The same accuracy. On POPE v21 is better calibrated; on NaturalBench the ECE difference is
not resolved. v21 is more confident than v19 throughout (mean confidence +0.04 to +0.07).

Without the image both fall to chance, so the answers come from the image. Text training
does not teach the model to say "I cannot tell": with the image removed, it still answers
at a mean confidence of 0.722 on NaturalBench and 0.792 on POPE, ECE 0.222 and 0.292. This is
worse than v19 (0.652 and 0.725, ECE 0.152 and 0.225; paired 95% CI of the ECE difference
+0.054 to +0.087 and +0.065 to +0.070). Do not use its confidence to detect a missing,
blank or unreadable image: check for the image before you ask.

## Limitations

- **Questions are read less than documents.** With the state and options fixed, a changed
  question often gets the same answer. Phrase a question so the obvious reading is the
  intended one.
- **Long, multi-step documents are the weak spot**: JevBench's hard tier scores far below
  its easy tier.
- **`score` and `noul` transfer poorly** to rubrics and yes/no tasks unlike the training
  mix. Train on a rubric resembling yours.
- **Calibration is one temperature per primitive**, fitted on held-out short
  classification. The confidence bands are established there only: measure on your own
  traffic before you trust a threshold.
- **Trained on public datasets**, so it inherits their domains and their label noise.

`evaluation/README.md`, section "Limitations", has the measured figures behind each point.

## Training data

The `datasets` list in the metadata holds the Hub datasets that the recipe reads.
`hotpotqa/hotpot_qa` is for evaluation only. The recipe also uses these sources:

- ContractNLI and MuSiQue, from their authors' releases (not from the Hub).
- Synthetic rows that open-weight language models generated and checked.
- The output distributions of the frozen teacher model `Qwen/Qwen3.5-4B`. They are training
  targets, directly or through a parent model that the recipe trains first.

In the code repository, `data/sources.md` lists every source with its revision, role,
license and attribution, and `data/README.md` states what is committed, what is downloaded
and the reproduction contract.

## Training

Trained by `training/recipe.sh train calibrate eval` of the code repository with
`TRAIN_CONFIG=configs/experiments/v21b.yaml` and seed 5 (`training/configs/v21b-s5.yaml`), on
a `p5.48xlarge` host (8x NVIDIA H100 80GB, `NGPU=8`): training stage 1,686 s, one epoch of
3,738 steps. Stage timings: `training/stages.jsonl`; data hashes: `training/data_sha256.txt`;
config: `train_config.json`.

To retrain, run the same recipe on a Linux or WSL2 host with NVIDIA GPUs: about 11 hours
on one RTX 3090, or about 1 h 10 min on eight H100s with `NGPU=8 FAST=1` through the AWS
runner. `training/README.md` has the setup, the stages and the hardware notes.

## Provenance

`strands_decider_config.json` is the trained checkpoint's, less one key, `full_weight_targets: []`,
that the training code wrote and released versions of strands-decider do not read (empty
means LoRA only, so the model is the same). `provenance.json`: base model and revision
(inferred: the hosts did not pin one; the loader pins it from this file), and the
sha256 of the config, adapter, head and source pickle. `MANIFEST.sha256` lists every file;
`python -m strands_decider.hf_export verify <folder>` checks them. The run records keep
their timings and results; host paths, cloud identifiers and cost fields are removed.
