---
base_model: Qwen/Qwen3.5-2B-Base
datasets:
- ccdv/arxiv-classification
- clinc/clinc_oos
- dair-ai/emotion
- fancyzhx/dbpedia_14
- google-research-datasets/paws
- google/boolq
- google/civil_comments
- legacy-datasets/banking77
- mteb/amazon_massive_intent
- nyu-mll/glue
- osyvokon/pavlick-formality-scores
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
- nvidia/HelpSteer2
- tasksource/Boardgame-QA
- hotpotqa/hotpot_qa
- tasksource/tasksource-jev-typed-decisions
- tasksource/procedural-typed-decisions
- BEE-spoke-data/consumer-finance-complaints
- glaiveai/glaive-function-calling-v2
- NousResearch/hermes-function-calling-v1
- theatticusproject/cuad-qa
- theatticusproject/maud
- coastalcph/lex_glue
- Anthropic/hh-rlhf
- nvidia/HelpSteer3
- nvidia/HelpSteer
- OpenAssistant/oasst2
- osunlp/Mind2Web
- tasksource/doc-nli
- rajpurkar/squad_v2
- stanfordnlp/snli
- pietrolesci/nli_fever
- openlifescienceai/medmcqa
- deepmind/aqua_rat
- tasksource/sharc
- Zhengping/UNLI
- bigcode/commitpackft
- deepmind/code_contests
library_name: peft
license: apache-2.0
pipeline_tag: text-classification
tags:
- strands-decider
- decision-model
- lora
- peft
- qwen3.5
- classification
- calibration
- typed-decisions
- code
base_model_relation: adapter
model-index:
- name: strands-decider-2B-qwen3.5-v1-2610
  results:
  - task:
      type: text-classification
    dataset:
      name: JevBench public, served at 4096
      type: jevbench
      config: w4096
    metrics:
    - type: accuracy
      value: 0.7792
      name: accuracy (180/231)
  - task:
      type: text-classification
    dataset:
      name: 'eval: held-out short tasks'
      type: hobson-internal
      config: 'eval: held-out short tasks'
    metrics:
    - type: accuracy
      value: 0.653
      name: accuracy (n=6000)
  - task:
      type: text-classification
    dataset:
      name: 'eval: boardgame'
      type: hobson-internal
      config: 'eval: boardgame'
    metrics:
    - type: accuracy
      value: 0.81
      name: accuracy (n=900)
  - task:
      type: text-classification
    dataset:
      name: 'eval: contractnli'
      type: hobson-internal
      config: 'eval: contractnli'
    metrics:
    - type: accuracy
      value: 0.878
      name: accuracy (n=1026)
  - task:
      type: text-classification
    dataset:
      name: 'eval: hotpotqa (held out)'
      type: hobson-internal
      config: 'eval: hotpotqa (held out)'
    metrics:
    - type: accuracy
      value: 0.832
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
      value: 0.876
      name: accuracy (n=599)
  - task:
      type: text-classification
    dataset:
      name: 'eval: musique, unanswerable'
      type: hobson-internal
      config: 'eval: musique, unanswerable'
    metrics:
    - type: accuracy
      value: 0.887
      name: accuracy (n=600)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [4] adequacy_hs2'
      type: hobson-internal
      config: 'eval: [4] adequacy_hs2'
    metrics:
    - type: accuracy
      value: 0.709
      name: accuracy (n=234)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [5] gen:adequacy'
      type: hobson-internal
      config: 'eval: [5] gen:adequacy'
    metrics:
    - type: accuracy
      value: 0.828
      name: accuracy (n=302)
---

# strands-decider-2B-qwen3.5-v1-2610

> The main home of this model is [`StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610). The same files are also at [`amazon/strands-decider-2B-qwen3.5-v1-2610`](https://huggingface.co/amazon/strands-decider-2B-qwen3.5-v1-2610).

Strands decider is a small, fast decision model for agentic AI. Unlike an LLM, which can
generate arbitrary text, a decision model picks between sets of options and rates things on
a scale, and every answer carries a calibrated confidence. It fits the decisions inside an
agentic workflow: model routing, tool selection, argument checking, triage, guardrails,
evaluations, and the rote decisions of a hybrid agent that leaves the hard ones to an LLM.

**v1** is the first model of the new names: `strands-decider-<size>-<base>-v<N>-<YYMM>`, that is the
base model's size, the base model, the recipe generation and the release month. The earlier
`strands-decider-2B-hobson-v19` and `-v21` keep their names (`docs/naming.md` in the code repository).

What is new in v1, against hobson-v21:

- **A balanced data mix** of 246,678 rows, about twice v21's. It adds code tasks and new
  rule-application rows (apply a written policy to a case), and it adds rows from more public
  datasets so that the mix follows the kinds of decisions in the evaluations. It drops v21's
  20,000 rows from datasets whose terms restrict commercial use.
- **A larger teacher:** the output distributions of `google/gemma-4-31B-it` are training targets
  where its answer agrees with the gold label.
- **The v19 anchor:** training keeps the answers close to the released
  `StrandsAgents/strands-decider-2B-hobson-v19` (a KL term, weight 0.3), not to the untrained base.
- **A soup:** this checkpoint averages three training runs (seeds 0, 1 and 2) of the same recipe.

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
strands-decider ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --state "Help! My payouts have been failing for 3 days! " \
  --choice "Which team should handle this?=billing,sales,retail" \
  --noul "Does this convey urgency?" \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```

The output of this checkpoint (`--device cpu`; other devices differ in the last digits):

```
noul_0 noul = 0.858
choice_0 -> billing (confidence 0.659)
  billing                  0.773
  sales                    0.121
  retail                   0.106
score_0 score = 1.03 (confidence 0.509)
  0: calm                                     0.196
  1: frustrated                               0.574
  2: depressed                                0.230
```

Images: `pip install "strands-decider[vision]"` (transformers 5.18 or later; the image path
is on the code repository's `main` and in releases after 0.1.0), then `serve --vision` or
`ask --image FILE`. See "Images" below and `docs/vision.md` in the code repository.

Or serve it over HTTP. The server binds to `127.0.0.1` and has no authentication: use it
for local experiments.

```bash
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8000
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days!",
  "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}
}'
```

In Python, `strands_decider.modeling.StrandsDeciderModel.load("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610")`. The adapter in
`lora/` is a standard PEFT adapter on the Qwen3.5 text decoder
(`transformers.Qwen3_5ForCausalLM(...).model`); the head is `head.safetensors`.

## Results

| evaluation | tasks right | Brier | ECE | served |
| --- | --- | --- | --- | --- |
| JevBench public, window 4096 | 180/231 | 0.280 | 0.072 | these files |

| internal set | accuracy | n |
| --- | --- | --- |
| eval: held-out short tasks | 0.653 | 6,000 |
| eval: boardgame | 0.810 | 900 |
| eval: contractnli | 0.878 | 1,026 |
| eval: hotpotqa (held out) | 0.832 | 959 |
| eval: musique | 0.882 | 1,199 |
| eval: musique, answerable | 0.876 | 599 |
| eval: musique, unanswerable | 0.887 | 600 |
| eval: [4] adequacy_hs2 | 0.709 | 234 |
| eval: [5] gen:adequacy | 0.828 | 302 |

JevBench is the external benchmark (231 public tasks); the internal sets are this recipe's
held-out short tasks, multi-step documents and answer-adequacy judgements (the code tasks are in
the table under "How this checkpoint was made"). MuSiQue,
ContractNLI and BoardgameQA are dev splits of sets whose train splits are in the training
data; HotpotQA is never trained on.
`evaluation/README.md` in the code repository describes each set, and its results pages
give the figures by version. Per-skill results and the raw reports and logs are in
`eval/summary.json` and `eval/internal/`.

## How this checkpoint was made

This checkpoint is a **soup** of three training runs of one recipe (seeds 0, 1 and 2). Each layer's
LoRA update is the exact mean of the three runs' updates (stored as one rank-48 adapter), and the
three readout heads are stacked so that the soup's score for an option is the mean of theirs. The
soup was then calibrated like any trained checkpoint. A soup needs no choice between seeds. It
scores at or above its seeds' mean on most measures below.

Seeds: their training runs (8x A100). Soup: these files through the code repository's `main`
(one NVIDIA L4).

| measure | seed 0 | seed 1 | seed 2 | seeds, mean ± SD | **soup (this checkpoint)** |
| --- | --- | --- | --- | --- | --- |
| JevBench public, tasks right of 231 (window 4096) | 178 | 177 | 184 | 179.7 ± 3.8 | **180** |
| JevBench Brier | 0.305 | 0.288 | 0.289 | 0.294 ± 0.010 | **0.280** |
| JevBench ECE | 0.079 | 0.088 | 0.060 | 0.076 ± 0.014 | **0.072** |
| MuSiQue (dev split; train split trained on) | 0.882 | 0.894 | 0.887 | 0.888 ± 0.006 | **0.882** |
| ContractNLI (dev split; train split trained on) | 0.866 | 0.871 | 0.870 | 0.869 ± 0.003 | **0.878** |
| BoardgameQA (dev split; train split trained on) | 0.800 | 0.803 | 0.801 | 0.801 ± 0.002 | **0.810** |
| HotpotQA (never trained on) | 0.822 | 0.825 | 0.823 | 0.823 ± 0.002 | **0.832** |
| held-out short tasks (6,000) | 0.660 | 0.649 | 0.642 | 0.650 ± 0.009 | **0.653** |
| code tasks, held out (1,996) | 0.826 | 0.833 | 0.831 | 0.830 ± 0.004 | **0.836** |
| HelpSteer2 adequacy (234) | 0.718 | 0.714 | 0.705 | 0.712 ± 0.007 | **0.709** |
| generated adequacy (302) | 0.818 | 0.788 | 0.808 | 0.805 ± 0.015 | **0.828** |
| generated documents, v16's set | 0.849 | 0.849 | 0.863 | 0.853 ± 0.008 | **0.857** |
| generated documents, v18's set | 0.745 | 0.753 | 0.737 | 0.745 ± 0.008 | **0.781** |
| JevBench tiers, easy / standard / hard | 48 / 65 / 65 | 48 / 66 / 63 | 48 / 64 / 72 | | **48 / 67 / 65** |

On the Decision Index (balanced skill) the soup scores 30.82, measured with the training harness.

## Release checks

These files through the code repository's `main` (not the training harness):

| check | result |
| --- | --- |
| JevBench public (window 4096), on one NVIDIA L4 | 180/231, Brier 0.280, ECE 0.072, strict schema 1.000 |
| the same on CPU | the same answer on all 231 tasks |
| against the training harness (A100) | the same answer on 230 of 231 tasks (181/231 there); the one other task is a near tie (0.346 against 0.345) |
| internal sets (`training/recipe.sh eval`) and the code tasks, on the L4 | the soup column above, each within 2 items of the harness |
| example output above, `--device cpu` and `--device cuda` | the same answers; probabilities within 0.006 |
| code: test suite (CPU), ruff and mypy, as CI runs them | pass |

## Images

With `--vision` the vision tower of Qwen3.5-2B-Base is kept and the adapter and head are used
unchanged; nothing was trained on images. NaturalBench (first 300 groups, 1,200 questions) and POPE
adversarial (first 600 items), this checkpoint and hobson-v21 on the same L4:

| | NaturalBench acc | G-Acc | Brier | ECE | POPE-adv acc | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 (this checkpoint), `--vision` | 0.776 | 0.310 | 0.307 | 0.039 | 0.872 | 0.192 | 0.023 |
| hobson-v21, `--vision` | 0.785 | 0.313 | 0.311 | 0.043 | 0.877 | 0.179 | 0.037 |
| v1 minus v21, paired bootstrap 95% CI | -0.009 (-0.022 to +0.003) | -0.003 | -0.005 (-0.013 to +0.004) | -0.004 (-0.019 to +0.016) | -0.005 (-0.020 to +0.008) | +0.013 (+0.003 to +0.023) | -0.014 (-0.036 to +0.010) |

The same accuracy as v21 within the intervals; POPE Brier is a little worse.

**Image missing.** Without the image both fall to chance, so the answers come from the image. With
the image removed this checkpoint still answers: mean confidence 0.711 on NaturalBench and 0.834 on
POPE, ECE 0.211 and 0.334 (v21: 0.722 and 0.792, ECE 0.222 and 0.292). On POPE it is more
overconfident than v21 (ECE +0.042, 95% CI +0.039 to +0.044). Do not use its confidence to detect a
missing, blank or unreadable image: check for the image before you ask.

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

246,678 rows in nine files:

| part | rows | what it is |
| --- | ---: | --- |
| short tasks | 80,449 | v21's public classification datasets, without the four whose terms restrict commercial use (AG News, Yahoo Answers topics, Yelp reviews, language identification) |
| multi-step documents | 12,909 | v21's, unchanged |
| generated documents | 3,815 | v21's, written and checked by open-weight models |
| answer adequacy | 6,166 | v21's: HelpSteer2 (CC BY 4.0) and generated items |
| code | 24,533 | generated programs with answers checked by running them; functions from 286 GitHub repositories with permissive licences (MIT, Apache-2.0, BSD, ISC); CuBERT (Apache-2.0), CommitPackFT (MIT), CodeReviewer (CC BY 4.0), CodeContests (CC BY 4.0), Juliet C/C++ 1.3 (public domain) |
| gap fill | 96,638 | rows from public datasets for the decision types that v21 had few rows for: tool choice, agent steps, preference, legal and contract text, routing, NLI, multiple choice, and 119 sources of `tasksource/tasksource-jev-typed-decisions` |
| rule application | 22,168 | new synthetic rows: apply a written policy to a case (below) |

**Rule-application rows.** Our templates hold each rule. `google/gemma-4-31B-it` (Apache-2.0) wrote
the policy text and the case files. The gold answers were computed by code from the rule. Claude was
used only as an independent check to filter rows: a row was kept only where gemma's and Claude's
answers both agree with the code. Claude's text is not in the data.

**Teacher.** The output distributions of `google/gemma-4-31B-it` are training targets on the yes/no
and choice rows where its answer is the gold label; the multi-step rows keep an earlier model's own
distributions; the generated-document and adequacy rows train on the gold label only.

**Checks on every row.** Licence, a 13-gram overlap check against JevBench's public tasks, the
Decision Index suite, HotpotQA dev and our held-out sets (rows with any overlap were dropped), the
4,096-token window, and a hand read of samples. HotpotQA is not in the training data; the mix does
have 2WikiMultihopQA, a similar multi-hop task.

`training/sources.md` lists every source with its revision, licence and attribution.

## Training

Each of the three runs trained one epoch on the mix with `training/recipe.sh train calibrate eval`
of the code repository, on `p4de.24xlarge` hosts (8x NVIDIA A100 80GB; about 79 min of training each). Their configs are in
`training/configs/`; they differ only in `seed`. The soup was made with `strands-decider soup`, then
calibrated with `training/recipe.sh calibrate` like any trained checkpoint. The anchor setting
(`kl_frozen_reference`, strands-labs/strands-decider#56) and the soup command (#58) are in the code
repository's `main`. The builders of this data mix are not yet, so the training data cannot be
rebuilt from `main`. Config: `train_config.json` (seed 0's).

## Provenance

`strands_decider_config.json` is the soup's, less three keys that the training code wrote and
released versions of strands-decider do not read: `full_weight_targets: []`, `force_bos: false`
and `host_embeddings: false` (each the off value, so the model is the same). `provenance.json`:
base model and revision (inferred: the hosts did not pin one; the loader pins it from this file),
and the
sha256 of the config, adapter, head and source pickle. `MANIFEST.sha256` lists every file;
`python -m strands_decider.hf_export verify <folder>` checks them. The run records keep
their timings and results; host paths, cloud identifiers and cost fields are removed.

## Changelog

- 2026-10-09: first release.
