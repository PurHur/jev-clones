---
base_model: google/gemma-4-26B-A4B-it
datasets:
- fancyzhx/ag_news
- legacy-datasets/banking77
- clinc/clinc_oos
- fancyzhx/dbpedia_14
- papluca/language-identification
- community-datasets/yahoo_answers_topics
- ucirvine/sms_spam
- google/civil_comments
- nyu-mll/glue
- google/boolq
- google-research-datasets/paws
- tals/vitaminc
- qiaojin/PubMedQA
- Yelp/yelp_review_full
- SetFit/sst5
- sealuzh/app_reviews
- osyvokon/pavlick-formality-scores
- tasksource/ruletaker
- tasksource/Boardgame-QA
- nvidia/HelpSteer2
- claudios/cubert_ETHPy150Open
- bigcode/commitpackft
- fasterinnerlooper/codereviewer
- LorenzH/juliet_test_suite_c_1_3
- deepmind/code_contests
library_name: peft
license: apache-2.0
pipeline_tag: text-classification
tags:
- strands-decider
- decision-model
- lora
- peft
- gemma4
- classification
- calibration
- typed-decisions
base_model_relation: adapter
model-index:
- name: strands-decider-26B-A4B-gemma4-v1-2610
  results:
  - task:
      type: text-classification
    dataset:
      name: JevBench public, served at 4096
      type: jevbench
      config: w4096
    metrics:
    - type: accuracy
      value: 0.8961
      name: accuracy (207/231)
  - task:
      type: text-classification
    dataset:
      name: 'eval: held-out short tasks'
      type: hobson-internal
      config: 'eval: held-out short tasks'
    metrics:
    - type: accuracy
      value: 0.708
      name: accuracy (n=6000)
  - task:
      type: text-classification
    dataset:
      name: 'eval: boardgame'
      type: hobson-internal
      config: 'eval: boardgame'
    metrics:
    - type: accuracy
      value: 0.869
      name: accuracy (n=900)
  - task:
      type: text-classification
    dataset:
      name: 'eval: contractnli'
      type: hobson-internal
      config: 'eval: contractnli'
    metrics:
    - type: accuracy
      value: 0.87
      name: accuracy (n=1026)
  - task:
      type: text-classification
    dataset:
      name: 'eval: hotpotqa (held out)'
      type: hobson-internal
      config: 'eval: hotpotqa (held out)'
    metrics:
    - type: accuracy
      value: 0.897
      name: accuracy (n=959)
  - task:
      type: text-classification
    dataset:
      name: 'eval: musique'
      type: hobson-internal
      config: 'eval: musique'
    metrics:
    - type: accuracy
      value: 0.922
      name: accuracy (n=1199)
  - task:
      type: text-classification
    dataset:
      name: 'eval: musique, answerable'
      type: hobson-internal
      config: 'eval: musique, answerable'
    metrics:
    - type: accuracy
      value: 0.917
      name: accuracy (n=599)
  - task:
      type: text-classification
    dataset:
      name: 'eval: musique, unanswerable'
      type: hobson-internal
      config: 'eval: musique, unanswerable'
    metrics:
    - type: accuracy
      value: 0.928
      name: accuracy (n=600)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [4] adequacy_hs2'
      type: hobson-internal
      config: 'eval: [4] adequacy_hs2'
    metrics:
    - type: accuracy
      value: 0.808
      name: accuracy (n=234)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [5] gen:adequacy'
      type: hobson-internal
      config: 'eval: [5] gen:adequacy'
    metrics:
    - type: accuracy
      value: 0.911
      name: accuracy (n=302)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [6] code/bug_swapped_operands'
      type: hobson-internal
      config: 'eval: [6] code/bug_swapped_operands'
    metrics:
    - type: accuracy
      value: 0.921
      name: accuracy (n=89)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [6] code/bug_variable_misuse'
      type: hobson-internal
      config: 'eval: [6] code/bug_variable_misuse'
    metrics:
    - type: accuracy
      value: 0.854
      name: accuracy (n=89)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [6] code/bug_wrong_binary_operator'
      type: hobson-internal
      config: 'eval: [6] code/bug_wrong_binary_operator'
    metrics:
    - type: accuracy
      value: 0.775
      name: accuracy (n=89)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [6] code/commit_message'
      type: hobson-internal
      config: 'eval: [6] code/commit_message'
    metrics:
    - type: accuracy
      value: 0.989
      name: accuracy (n=89)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [6] code/correct_solution'
      type: hobson-internal
      config: 'eval: [6] code/correct_solution'
    metrics:
    - type: accuracy
      value: 0.562
      name: accuracy (n=89)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [6] code/cwe_choice'
      type: hobson-internal
      config: 'eval: [6] code/cwe_choice'
    metrics:
    - type: accuracy
      value: 0.933
      name: accuracy (n=89)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [6] code/docstring_match'
      type: hobson-internal
      config: 'eval: [6] code/docstring_match'
    metrics:
    - type: accuracy
      value: 0.955
      name: accuracy (n=89)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [6] code/exception_type'
      type: hobson-internal
      config: 'eval: [6] code/exception_type'
    metrics:
    - type: accuracy
      value: 0.899
      name: accuracy (n=89)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [6] code/exec_input'
      type: hobson-internal
      config: 'eval: [6] code/exec_input'
    metrics:
    - type: accuracy
      value: 0.796
      name: accuracy (n=142)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [6] code/exec_output'
      type: hobson-internal
      config: 'eval: [6] code/exec_output'
    metrics:
    - type: accuracy
      value: 0.834
      name: accuracy (n=662)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [6] code/needle_function'
      type: hobson-internal
      config: 'eval: [6] code/needle_function'
    metrics:
    - type: accuracy
      value: 0.98
      name: accuracy (n=391)
  - task:
      type: text-classification
    dataset:
      name: 'eval: [6] code/review_needed'
      type: hobson-internal
      config: 'eval: [6] code/review_needed'
    metrics:
    - type: accuracy
      value: 0.607
      name: accuracy (n=89)
---

# strands-decider-26B-A4B-gemma4-v1-2610

> The main home of this model is [`StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610). The same files are also at [`amazon/strands-decider-26B-A4B-gemma4-v1-2610`](https://huggingface.co/amazon/strands-decider-26B-A4B-gemma4-v1-2610).

Strands decider is a small, fast decision model for agentic AI. Unlike an LLM, which can
generate arbitrary text, a decision model picks between sets of options and rates things on
a scale, and every answer carries a calibrated confidence. It fits the decisions inside an
agentic workflow: model routing, tool selection, argument checking, triage, guardrails,
evaluations, and the rote decisions of a hybrid agent that leaves the hard ones to an LLM.

This is **v1, the first Strands decider on Gemma 4**, on `google/gemma-4-26B-A4B-it`. It trains on
hobson-v21's data (123,339 rows) plus 12,263 code-task rows. The teacher is `Qwen/Qwen3.5-4B`: its
output distributions are training targets where its answer agrees with the gold label. A frozen-KL
anchor keeps the answers close to those of `StrandsAgents/strands-decider-2B-hobson-v19`. The
checkpoint is a **soup** of three seeds of this recipe:
each layer's LoRA update is the exact mean of the seeds' updates (stored as one rank-48 adapter),
and the score of an option is the mean of the three heads' scores. The soup was then calibrated.

**Model names:** `strands-decider-<size>-<base>-v<N>-<YYMM>`, that is the base model's size, the
base model family, the release on that family (`v1` is the first) and the release month
(`docs/naming.md` in the code repository).

This repository holds a LoRA adapter on `google/gemma-4-26B-A4B-it` plus a small readout head that
scores the options of a typed question (`noul`, a yes/no question; `choice`, one of N options;
`score`, a level on an ordered scale). The code, the training recipe, the data inventory and the
evaluations are at https://github.com/strands-labs/strands-decider, under the same Apache-2.0
license as this model (see `LICENSE.md`).

## Use

```bash
pip install "strands-decider @ git+https://github.com/strands-labs/strands-decider"
```

This installs the code repository's `main`, which has Gemma 4 support; the package on PyPI does not have it yet.

Ask one or more questions about a state. Pass `--device cuda`, `mps` or `cpu`; without it, the
best available device is used. The base weights download from the Hub at first use.

```bash
strands-decider ask StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610 \
  --state "Help! My payouts have been failing for 3 days! " \
  --choice "Which team should handle this?=billing,sales,retail" \
  --noul "Does this convey urgency?" \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```

The output of this checkpoint (`--device cpu`; other devices differ in the last digits):

```
noul_0 noul = 0.981
choice_0 -> billing (confidence 0.712)
    billing                  0.808
    sales                    0.127
    retail                   0.065
score_0 score = 1.00 (confidence 0.828)
    0: calm                                     0.094
    1: frustrated                               0.812
    2: depressed                                0.094
```

Or serve it over HTTP. The server binds to `127.0.0.1` and has no authentication: use it for
local experiments.

```bash
strands-decider serve StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610 --port 8000
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days!",
  "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}
}'
```

In Python, `strands_decider.modeling.StrandsDeciderModel.load("StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610")`. The adapter in `lora/` is a
standard PEFT adapter on the Gemma 4 text decoder
(`transformers.Gemma4ForConditionalGeneration(...).model.language_model`); the head is `head.safetensors`.

## Results

| evaluation | tasks right | Brier | ECE | served |
| --- | --- | --- | --- | --- |
| JevBench public, window 4096 | 207/231 | 0.162 | 0.051 | these files |

| benchmark | score |
| --- | --- |
| Decision Index 0.2.1, balanced skill (window 32,768) | 48.69 |

| internal set | accuracy | n |
| --- | --- | --- |
| eval: held-out short tasks | 0.708 | 6,000 |
| eval: boardgame | 0.869 | 900 |
| eval: contractnli | 0.870 | 1,026 |
| eval: hotpotqa (held out) | 0.897 | 959 |
| eval: musique | 0.922 | 1,199 |
| eval: musique, answerable | 0.917 | 599 |
| eval: musique, unanswerable | 0.928 | 600 |
| eval: [4] adequacy_hs2 | 0.808 | 234 |
| eval: [5] gen:adequacy | 0.911 | 302 |
| eval: [6] code/bug_swapped_operands | 0.921 | 89 |
| eval: [6] code/bug_variable_misuse | 0.854 | 89 |
| eval: [6] code/bug_wrong_binary_operator | 0.775 | 89 |
| eval: [6] code/commit_message | 0.989 | 89 |
| eval: [6] code/correct_solution | 0.562 | 89 |
| eval: [6] code/cwe_choice | 0.933 | 89 |
| eval: [6] code/docstring_match | 0.955 | 89 |
| eval: [6] code/exception_type | 0.899 | 89 |
| eval: [6] code/exec_input | 0.796 | 142 |
| eval: [6] code/exec_output | 0.834 | 662 |
| eval: [6] code/needle_function | 0.980 | 391 |
| eval: [6] code/review_needed | 0.607 | 89 |

JevBench is the external benchmark (231 public tasks); the Decision Index is the decision-model
index of the `decision-index` kit. The internal sets are this recipe's held-out short tasks,
multi-step documents, answer-adequacy judgements and code tasks. `evaluation/README.md` in the
code repository describes each set. Per-skill results and the raw reports and logs are in
`eval/summary.json` and `eval/internal/`.

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

The `datasets` list in the metadata holds the 25 Hub datasets that the training rows come from:
18 short-task classification datasets, `tasksource/Boardgame-QA`, `nvidia/HelpSteer2`, and five code
datasets. The other sources are:

- ContractNLI and MuSiQue, from their authors' releases (not from the Hub).
- Synthetic rows that open-weight language models generated and checked.
- Code-task rows built here: generated programs with execution-checked answers, and functions from
  permissively licensed GitHub repositories. The five code datasets give the other code tasks
  (CuBERT ETH Py150 Open, CommitPackFT, CodeReviewer, Juliet C/C++ 1.3, CodeContests). Code rows
  carry gold labels only.
- The output distributions of the frozen teacher model `Qwen/Qwen3.5-4B`. They are training
  targets, next to an anchor toward the distributions of `StrandsAgents/strands-decider-2B-hobson-v19`.

These Hub datasets are for calibration and evaluation only, not training: `dair-ai/emotion`,
`mteb/amazon_massive_intent`, `raquiba/Sarcasm_News_Headline`, `ucberkeley-dlab/measuring-hate-speech`
and `hotpotqa/hotpot_qa`.

In the code repository, `data/sources.md` lists every source with its revision, role, license and
attribution. `data/README.md` states what is committed,
what is downloaded and the reproduction contract. Read a source's own terms before you redistribute
data built from it.

## Training

Three seeds (0, 1, 2) of one config (`train_config.json`; the seeds differ only in `seed`), each
trained by `training/recipe.sh train` on one host with 8 NVIDIA H100 80GB GPUs (`NGPU=8`), one epoch
of 4,109 steps. The soup was made by `strands-decider soup` and calibrated and evaluated by
`training/recipe.sh calibrate eval`. `training/README.md` has the setup, the stages and the hardware notes.

## Provenance

`provenance.json`: base model and revision (inferred: the hosts did not pin one), and the
sha256 of the config, adapter, head and source pickle. `MANIFEST.sha256` lists every file;
`python -m strands_decider.hf_export verify <folder>` checks them. The run records keep
their timings and results; host paths, cloud identifiers and cost fields are removed.

## Changelog

- 2026-10-09 (Hub tag `261009`): first release.
- 2026-10-09: card corrected (the recipe, the install line, the dataset list). The weights did not change.
