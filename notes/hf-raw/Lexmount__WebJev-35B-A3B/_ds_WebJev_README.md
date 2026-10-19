---
pretty_name: WebJev
license: apache-2.0
language:
- en
multilinguality: monolingual
size_categories:
- 10K<n<100K
source_datasets:
- original
annotations_creators:
- machine-generated
language_creators:
- found
- machine-generated
task_categories:
- multiple-choice
- text-classification
task_ids:
- multiple-choice-qa
tags:
- web-agent
- browser-agent
- gui-agent
- action-prediction
- element-grounding
- decision-making
- live-web
configs:
- config_name: all
  default: true
  data_files:
  - split: train
    path: data/*/train-*.parquet
- config_name: action_prediction
  data_files:
  - split: train
    path: data/action_prediction/train-*.parquet
- config_name: element_grounding
  data_files:
  - split: train
    path: data/element_grounding/train-*.parquet
dataset_info:
- config_name: all
  features: &id001
  - name: id
    dtype: string
  - name: task_type
    dtype: string
  - name: question_type
    dtype: string
  - name: action
    dtype: string
  - name: context
    dtype: string
  - name: question
    dtype: string
  - name: options
    list: string
  - name: gold
    dtype: int32
  - name: answer
    dtype: string
  - name: num_options
    dtype: int32
  - name: num_tokens
    dtype: int32
  - name: context_format
    dtype: string
  - name: website
    dtype: string
  - name: page_url
    dtype: string
  - name: task_source
    dtype: string
  - name: label_source
    dtype: string
  - name: task_id
    dtype: string
  - name: instance_id
    dtype: string
  splits:
  - name: train
    num_bytes: 1139312203
    num_examples: 64122
  download_size: 120376472
  dataset_size: 1139312203
- config_name: action_prediction
  features: *id001
  splits:
  - name: train
    num_bytes: 395573253
    num_examples: 31050
  download_size: 33971063
  dataset_size: 395573253
- config_name: element_grounding
  features: *id001
  splits:
  - name: train
    num_bytes: 743738950
    num_examples: 33072
  download_size: 86405409
  dataset_size: 743738950
---

# WebJev

**Training web agents to complete real tasks on the live web**

To complete a task on a real website, a web agent must get a long chain of decisions right, from the first page to
the final answer: *what to do next*, and *which element to act on*. WebJev captures these decisions along complete
task trajectories on the live web. It contains **64,122 decisions** from **3,858 tasks** on
**1,443 real-world websites**. They cover every stage of a task: searching, navigating, filtering, filling in
forms, reading results and deciding when the task is done.

Each instance pairs an agent's observation of the current page with one multiple-choice question. Together the
instances cover two complementary skills:

- **Action prediction** (31,050 instances): choose the next operation among those valid on the current page
  (3–13 options). The operations are click, type, select, scroll, wait, submit, dismiss, go back, finish and give up.
- **Element grounding** (33,072 instances): choose the element to act on among up to 137
  candidates from the same page (median 32).

Observations follow the format a deployed browser agent sends to its decision model. They include:

- the page's URL, title and visible text;
- the actionable elements in the viewport, with their roles, values and positions;
- the recent action history.

An LLM annotator produced the labels while operating the agent on live websites, and only labels that passed
multi-signal verification were kept. The grounding instances are further enriched with same-page hard negatives at
several candidate scales.

![One task, completed decision by decision](assets/trajectory.png)
*One WebJev task on spanishdict.com, completed decision by decision: type the query, pick the suggestion, dismiss a
pop-up, read the entry and finish. The blue boxes mark the elements the agent acts on.*

## Highlights

- **Whole-task coverage.** Decisions come from complete task trajectories on live websites. They span every stage of
  a task, from the first search to the final `DONE`.
- **Live and diverse.** 3,858 tasks on 1,443 real-world websites, from shopping, travel and real estate
  to education, research, software and sports.
- **Two decision skills.** 31,050 action-prediction and 33,072 element-grounding instances.
- **Deployment-faithful observations.** 89% of the instances use the structured observation of a
  production browser agent; the rest use a textual page rendering that adds format diversity.
- **Hard grounding.** Grounding questions have a median of 32 candidates and up to
  137, with same-page hard negatives and multi-scale candidate sets.
- **Long context.** The median instance has 4,262 tokens, and 10% of instances have more than
  9,730.
- **Verified labels.** Labels are kept only after multi-signal verification: execution on the live page,
  re-annotation consistency and adjudication. Independent audits of stratified samples found 98% of annotated labels
  correct or acceptable.

| At a glance | |
|---|---|
| Instances | 64,122 (31,050 action prediction, 33,072 element grounding) |
| Tasks · websites | 3,858 · 1,443 |
| Distinct decision points | 40,230 |
| Candidates per question | action: 3–13 (median 7); grounding: 2–137 (median 32) |
| Context length | median 4,262 tokens, 90th percentile 9,730, maximum 16,383 (327.6M in total) |
| Observation formats | structured agent observation 57,283; textual rendering 6,839 |
| Language | English tasks and instructions; page text mostly English |
| Collection period | August–September 2026 |

![Grounding examples from twelve websites](assets/gallery.png)
*Element-grounding examples from twelve of the 1,443 websites. The blue box marks the gold element; each caption
gives the intended step and the number of candidates.*

## Task formulation

A web task unfolds as a chain of decisions. At every step the agent observes the page, decides what to do and, for
clicks, typing and selects, which element to act on. Each decision yields one instance per question.

![Anatomy of one decision](assets/overview.png)
*One decision on spanishdict.com. After typing "spring" into the search box, the agent must decide what to do next
(action prediction) and which of the 28 visible elements to click (element grounding).*

Each instance is a tuple *(observation, instruction, candidates, answer)*:

- **Observation** (`context`). What the agent sees at this step. The *structured* form is a JSON object:
  - `page`: URL, title and visible text;
  - `elements`: actionable elements of the viewport, each with its index, role, label, current value, supported
    operations and position;
  - `recent_actions`: the recent action history and whether each action changed the page.

  The *textual* form renders the website, the task, the current subgoal, the actions taken so far and the page text
  as plain text.
- **Instruction** (`question`). What is asked. Structured instances carry the user's task, the current subgoal, the
  date and the agent's decision policy. Textual instances carry a single question.
- **Candidates** (`options`). For action prediction, the operations valid on the page. For element grounding, the
  elements of the page that support the requested operation.
- **Answer** (`gold`). The index of the correct candidate.

**Action space.**

| Action | Meaning |
|---|---|
| `CLICK` | click a link, button, menu item, suggestion or date |
| `TYPE_TEXT` | enter text into an editable field |
| `SELECT` | choose a value in a drop-down |
| `SCROLL_DOWN` / `SCROLL_UP` | scroll the page |
| `SCROLL_REGION` | scroll inside a scrollable page region (appears as `E<n>` in the options) |
| `WAIT` | wait for the page to update |
| `PRESS_ENTER` | submit the focused field |
| `PRESS_ESCAPE` | close a menu or pop-up |
| `GO_BACK` | return to the previous page |
| `DONE` | the goal is visibly satisfied |
| `BLOCKED` | no available action can make progress |

**Model interface.** The dataset suits decision models that score options rather than generate free text. We render
each instance as follows:

```text
Context:
{context}

Question: {question}
Options:
(A) {option}
(B) {option}
…
Answer: (
```

- The option order is shuffled.
- Options are labelled `A`, `B`, … and then `AA`, `AB`, … for large candidate sets.
- The model is trained with cross-entropy over the option labels at the answer position.
- `num_tokens` reports the length of this rendering under the Qwen3.5 tokenizer.

## Data distribution

![Distributions of WebJev](assets/distributions.png)
*(a) Action labels of action-prediction instances. (b) Composition by question type. (c) Candidate-set size of
element-grounding instances. (d) Context length.*

### Composition

| Task type | Question type | Structured observation | Textual observation | Total |
|---|---|---:|---:|---:|
| Action prediction | `operation` | 31,050 | 0 | 31,050 |
| Element grounding | `click_target` | 25,465 | 6,197 | 31,662 |
| Element grounding | `type_target` | 541 | 604 | 1,145 |
| Element grounding | `select_target` | 227 | 38 | 265 |
| **Total** | | **57,283** | **6,839** | **64,122** |

### Actions

The action labels follow the natural frequency of decisions during web navigation. Clicks and scrolling dominate, and
terminal decisions (`DONE`, `BLOCKED`) account for 8.6%.

| Action | Instances | Share |
|---|---:|---:|
| `CLICK` | 14,364 | 46.3% |
| `SCROLL_DOWN` | 6,135 | 19.8% |
| `TYPE_TEXT` | 3,527 | 11.4% |
| `DONE` | 2,397 | 7.7% |
| `GO_BACK` | 1,553 | 5.0% |
| `WAIT` | 1,137 | 3.7% |
| `SCROLL_UP` | 713 | 2.3% |
| `PRESS_ENTER` | 362 | 1.2% |
| `SCROLL_REGION` | 355 | 1.1% |
| `BLOCKED` | 259 | 0.8% |
| `SELECT` | 226 | 0.7% |
| `PRESS_ESCAPE` | 22 | 0.1% |

### Grounding difficulty

Grounding questions are hard: the correct element must be chosen among many look-alike candidates of the same page.

- **Click targets** are the largest group. Their candidate sets are multi-scale (32 / 64 / 128 / all elements), which
  exposes models to increasing distractor density.
- **Select targets** often list long option menus.
- **Type targets** usually choose between a few input fields.

| Question type | Instances | Min | Median | 90th percentile | Max |
|---|---:|---:|---:|---:|---:|
| `operation` (action prediction) | 31,050 | 3 | 7 | 9 | 13 |
| `click_target` | 31,662 | 2 | 32 | 86 | 137 |
| `type_target` | 1,145 | 2 | 2 | 4 | 12 |
| `select_target` | 265 | 2 | 26 | 81 | 131 |
| all grounding | 33,072 | 2 | 32 | 84 | 137 |

### Context length

| Subset | Instances | Median | 90th percentile | Max |
|---|---:|---:|---:|---:|
| Action prediction | 31,050 | 3,056 | 6,902 | 16,371 |
| Element grounding | 33,072 | 5,665 | 11,366 | 16,383 |
| Structured observations | 57,283 | 4,102 | 9,332 | 16,383 |
| Textual observations | 6,839 | 6,102 | 11,587 | 16,376 |
| **All** | 64,122 | 4,262 | 9,730 | 16,383 |

### Language

- The tasks, instructions and candidates are in English.
- Page text is predominantly English. In 0.9% of the instances a non-Latin script makes up at least 20%
  of the observation: CJK 434, Cyrillic 65, Devanagari 53, Arabic 36, Greek 5.
- Localized pages in other Latin-script languages also occur.

### Label sources

| Label source | Instances | Share | Description |
|---|---:|---:|---|
| `live_annotation` | 43,890 | 68.4% | action chosen by the LLM annotator on the live page, kept after verification |
| `trajectory` | 20,011 | 31.2% | target taken from a successful agent trajectory and verified |
| `trajectory_corrected` | 221 | 0.3% | trajectory target replaced by the reviewer's corrected target |

## Dataset structure

### Configs and splits

| Config | Instances | Content |
|---|---:|---|
| `all` (default) | 64,122 | all instances |
| `action_prediction` | 31,050 | next-action questions |
| `element_grounding` | 33,072 | click, type and select target questions |

The dataset is released as a single `train` split. Instances that share a `task_id` come from the same task, and
instances that share an `instance_id` come from the same decision step: its action and grounding questions, or the
same grounding question at different candidate scales. Use `task_id` to build leakage-free held-out splits.

### Data fields

| Field | Type | Description |
|---|---|---|
| `id` | string | unique instance id |
| `task_type` | string | `action_prediction` or `element_grounding` |
| `question_type` | string | `operation`, `click_target`, `type_target` or `select_target` |
| `action` | string | action prediction: the gold action (see the action space). Grounding: the operation being grounded |
| `context` | string | the observation (model input) |
| `question` | string | the instruction and question (model input) |
| `options` | list[string] | the candidates (model input) |
| `gold` | int32 | index of the correct candidate in `options` |
| `answer` | string | `options[gold]` |
| `num_options` | int32 | number of candidates |
| `num_tokens` | int32 | length of the rendered instance (Qwen3.5 tokenizer) |
| `context_format` | string | `structured` or `text` |
| `website` | string | the task's target website |
| `page_url` | string, nullable | URL of the observed page (structured observations) |
| `task_source` | string | public task collection of the task |
| `label_source` | string | `live_annotation`, `trajectory` or `trajectory_corrected` (see Label sources) |
| `task_id` | string | task identifier, for grouping |
| `instance_id` | string | decision-step identifier, for grouping |

Only `context`, `question` and `options` are model inputs, and `gold` is the target. The remaining fields describe
the instance and are meant for analysis and splitting.

### Example

```json
{
  "task_type": "element_grounding",
  "question_type": "click_target",
  "action": "CLICK",
  "context": "{\"page\": {\"url\": \"https://www.spanishdict.com/\", \"title\": \"SpanishDictionary.com | English to Spanish Translation, Dictionary, Translator\", \"text\": \"Learn Spanish\\nTranslation\\nConjugation\\n…\"}, \"elements\": [{\"role\": \"link\", \"index\": \"1\", \"label\": \"SpanishDictionary.com Homepage\", \"operations\": [\"CLICK\"], \"position\": {\"x\": 213, \"y\": 0}}, …], \"recent_actions\": [{\"action\": \"Translate Spanish or English\", \"kind\": \"fill\", \"text\": \"spring\", \"page_changed\": true}]}",
  "question": "{\"goal\": \"… The user's original task: On https://www.spanishdict.com: Search for the word 'spring' on SpanishDict and identify two different Spanish translations that represent distinct meanings. …\", \"operation\": \"CLICK\", \"rules\": [\"Advance the user's entire goal from the CURRENT page using one operation. …\"]}",
  "options": ["1: {\"element\": \"[1] SpanishDictionary.com Homepage\", …}", "…", "14: {\"element\": \"[14] spring\", \"role\": \"option\", …}", "15: {\"element\": \"[15] spring break\", \"role\": \"option\", …}", "…"],
  "gold": 12,
  "answer": "14: {\"element\": \"[14] spring\", \"current_value\": \"0\", \"role\": \"option\", \"selected\": \"false\"}",
  "num_options": 28,
  "num_tokens": 3949,
  "context_format": "structured",
  "website": "spanishdict.com",
  "task_source": "MolmoWeb",
  "label_source": "live_annotation"
}
```

### Usage

```python
from datasets import load_dataset

ds = load_dataset("Lexmount/WebJev", "all", split="train")
grounding = load_dataset("Lexmount/WebJev", "element_grounding", split="train")

def render(ex):
    labels = [chr(65 + i) for i in range(26)] + [a + b for a in map(chr, range(65, 91)) for b in map(chr, range(65, 91))]
    options = "".join(f"\n({labels[i]}) {o}" for i, o in enumerate(ex["options"]))
    return f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}\nOptions:{options}\nAnswer: ("

# leakage-free split by task
tasks = sorted(set(ds["task_id"]))
held_out = set(tasks[: len(tasks) // 20])
train = ds.filter(lambda ex: ex["task_id"] not in held_out)
valid = ds.filter(lambda ex: ex["task_id"] in held_out)
```

## Dataset construction

![Construction of WebJev](assets/pipeline.png)

WebJev was built in four stages.

1. **Task curation.** We build on the task collections of MolmoWeb and WebGym. We keep tasks on real, publicly
   reachable websites that need no account.
2. **Agent-in-the-loop collection.** An LLM annotator (DeepSeek-V4.1-Flash) operated a production-grade browser
   agent on the live websites, in isolated cloud browser sessions.
   - At every step, the agent's observation and questions were recorded exactly as served.
   - The annotator's choice was then executed, and its effect on the page was recorded with it.
   - A successful demonstration of the same task guided the annotator. It is never part of the model input, so each
     instance must be answerable from the observation alone.
3. **Multi-signal verification.** Each candidate label was checked against complementary signals:
   - execution evidence and a meaningful page-state change;
   - trajectory-level loop detection;
   - agreement under independent re-annotation with shuffled options and without guidance;
   - adjudication by an independent verifier for terminal and ambiguous decisions.

   A final review restored valid decisions that conservative checks had rejected and removed near-duplicates. About
   half of all recorded decisions survived verification.
4. **Grounding enrichment.** Successful steps of agent trajectories on the same websites were converted into
   additional element-grounding instances.
   - Candidate sets are built from the actionable elements of the same page, with mined hard negatives.
   - Each question is rendered at several candidate scales.
   - An LLM reviewer checks the target, and a blind challenge removes candidates that could also be correct.
   - A textual observation view adds format diversity.

**Quality control.**

- Independent audits of stratified samples, checked against page screenshots, found 98% of the annotated labels
  correct or acceptable.
- Every instance fits in 16K tokens without truncation.
- Exact duplicates were removed.

## Licensing and access

- **Licence.** WebJev is released under the [Apache License 2.0](LICENSE). You may download it, use it to train and
  evaluate models, including commercial ones, and redistribute it. Please keep the licence and attribution notice.
- **Third-party rights.** Page text, URLs and element labels were captured from publicly accessible websites and
  remain the property of their owners. The task descriptions derive from public task collections under their own
  licences, and the labels were produced with DeepSeek models under their terms of use. The Apache licence covers
  Lexmount's contribution (collection, labels, annotations and structure). It grants no rights in third-party content.

## Citation

```bibtex
@misc{lexmount2026webjev,
  title        = {WebJev: Training Web Agents to Complete Real Tasks on the Live Web},
  author       = {{Lexmount}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/Lexmount/WebJev}}
}
```

## Contact

Lexmount, via the Lexmount organization on Hugging Face.
