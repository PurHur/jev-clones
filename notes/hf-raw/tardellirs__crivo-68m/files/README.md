---
license: cc-by-nc-4.0
language:
- en
base_model: cross-encoder/ettin-reranker-68m-v1
pipeline_tag: text-classification
library_name: transformers
tags:
- decision-model
- jev-style
- typed-decisions
- system-one
- decision-index
- knowledge-distillation
- calibration
- modernbert
- ettin
- cross-encoder
- zero-shot-classification
- classification
- calibrated-probabilities
- routing
- multiple-choice
- tool-selection
- jev
datasets:
- hotchpotch/bekko-system-one-dataset-v0
- tasksource/procedural-typed-decisions
- tasksource/tasksource-jev-typed-decisions
- NousResearch/hermes-function-calling-v1
- Team-ACE/ToolACE
- glaiveai/glaive-function-calling-v2
- causalNLP/cladder
- galileo-ai/ragbench
- theatticusproject/cuad
- facebook/anli
- nvidia/When2Call
model-index:
- name: Crivo-68M
  results:
  - task:
      type: text-classification
      name: Typed decisions (choice / noul / score)
    dataset:
      name: Decision Index 0.3
      type: decision-index
    metrics:
    - type: public_index
      value: 32.82
      name: Public index (self-run, complete public suite)
    source:
      name: Decision Index 0.3 public suite, official kit
      url: https://huggingface.co/datasets/tardellirs/decision-index-results/tree/main/runs/crivo-68m
---

# Crivo-68M

**A 68M-parameter typed-decision encoder, distilled from Crivo-400M.**

[![Decision Index public](https://img.shields.io/badge/Decision%20Index%200.3-public%2032.82-2a78d6)](#decision-index-03--public-index-self-run-complete)
[![Full score](https://img.shields.io/badge/Full%20score-pending-999)](#decision-index-03--public-index-self-run-complete)
[![Parameters](https://img.shields.io/badge/parameters-68M-555)](#model-details)
[![Language](https://img.shields.io/badge/language-English-555)](#intended-use-and-limitations)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-555)](#license)

Crivo-68M reads a *state* and a set of typed *questions* (`choice`, `noul`, `score`) and returns a probability for
every option in a single encoder pass. It does not generate text. It is built for the classify, route, gate and verify
decisions that agents and pipelines make thousands of times, and sits in the middle of the Crivo family, between
[Crivo-32M](https://huggingface.co/tardellirs/crivo-32m) and [Crivo-150M](https://huggingface.co/tardellirs/crivo-150m).
It was trained on the data of [Crivo-400M](https://huggingface.co/tardellirs/crivo-400m), with Crivo-400M as its
teacher.

| | |
|---|---|
| Architecture | Ettin / ModernBERT encoder, 19 layers, hidden 512, `[OPT]` readout + MLP scorer |
| Parameters | 68.4M (68,410,881) |
| Base model | [`cross-encoder/ettin-reranker-68m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-68m-v1) |
| Teacher | [`tardellirs/crivo-400m`](https://huggingface.co/tardellirs/crivo-400m) (knowledge distillation) |
| Decision Index 0.3 | public index **32.82** (self-run, 140,178 / 140,178 requests answered); Full score pending (maintainer-run) |
| Latency, 1× A100 40 GB, bf16 (one request at a time) | median 16.4 ms · mean 21.2 ms · p95 28.6 ms per request |
| Context | trained up to 8,192 tokens; reads longer inputs whole, up to 32,768, without truncation |
| Language | English |
| License | CC BY-NC 4.0 (see [License](#license)) |

Crivo follows the typed-decision interface popularized by TypeSafe's Jev (a state plus typed questions in, a
probability for every option out) and runs unmodified in the open
[Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) kit. It is an independent open model,
not affiliated with or endorsed by TypeSafe, and not trained on Jev outputs. *Crivo* is Portuguese for "sieve";
*passar pelo crivo*, to pass through the sieve, means to be judged and sorted.

<details>
<summary><b>Resumo em português</b></summary>

Crivo-68M é um encoder de decisões tipadas com 68M de parâmetros (Ettin/ModernBERT), destilado do Crivo-400M e
treinado com os mesmos dados dele. Ele lê um *state* e perguntas `choice`, `noul` ou `score` e devolve uma
probabilidade para cada opção numa única passada, sem gerar texto. No índice público do Decision Index 0.3 marca
**32,82** (run completo publicado, sem truncamento), entre o Crivo-32M (25,06) e o Crivo-150M (36,39), com mediana de
16,4 ms por requisição numa A100. O Full score, com os testes privados dos mantenedores, ainda não saiu. Foi treinado só
com dados abertos, com verificação de contaminação contra a suíte. Funciona em inglês. Licença CC BY-NC 4.0 (uso não
comercial).

</details>

## Results

### Decision Index 0.3 — public index (self-run, complete)

Run with the official kit ([apolinario/decision-index](https://github.com/apolinario/decision-index), commit
`9eb2dbe`, edition 0.3) on the complete public suite, with the engine shipped in this repository: every one of the
140,178 scoreable requests answered, none unsupported, nothing truncated. Scores are chance-corrected skill (0 = random
guessing, 100 = perfect), weighted as the board does. Full results:
[`tardellirs/decision-index-results` → `runs/crivo-68m`](https://huggingface.co/datasets/tardellirs/decision-index-results/tree/main/runs/crivo-68m).

The board's main ranking (Full score) is 20% this public index, 50% private tests of the same skills and 30% private
tasks from new domains, run by the maintainers. **The Full score will be added here once it is published.**

| Model | Public index | Knowledge & Reasoning | Language | Retrieval & Classification | Tools & Automation | Arts & Human Taste |
|---|---:|---:|---:|---:|---:|---:|
| Crivo-32M | 25.06 | 2.0 | 25.6 | 46.7 | 36.7 | 18.5 |
| **Crivo-68M** | **32.82** | 5.0 | 30.8 | 48.8 | 64.0 | 20.9 |
| Crivo-150M | 36.39 | 11.8 | 41.8 | 52.2 | 60.7 | 10.1 |

A comparison with other models of similar size will be added together with the Full score, since the public index alone does not rank models by Full score.

<details>
<summary><b>Per-benchmark skill (0–100, chance-corrected) — click to expand</b></summary>

★ = gold benchmark (weight 1.2 within its area). The index uses the native metric listed, chance-corrected.

| Benchmark | Native metric | Crivo-68M | Crivo-32M | Crivo-150M |
|---|---|---:|---:|---:|
| **Knowledge & Reasoning** | | | | |
| GPQA Diamond ★ | accuracy | 0.0 | 0.0 | 0.0 |
| GSM8K | accuracy | 0.5 | 0.3 | 2.0 |
| ChessBench | accuracy | 0.0 | 0.0 | 0.0 |
| MuSR | accuracy | 20.1 | 0.0 | 26.7 |
| SATA-Bench | case exact accuracy | 0.0 | 0.0 | 0.0 |
| CRUXEval | accuracy | 12.3 | 8.4 | 7.3 |
| CLadder | accuracy | 8.1 | 8.4 | 61.0 |
| HLE ★ | accuracy | 0.0 | 0.0 | 0.0 |
| MMLU-Pro ★ | accuracy | 3.5 | 1.1 | 3.4 |
| BBH fixed-option tasks ★ | accuracy | 6.1 | 1.4 | 11.6 |
| WinoGrande ★ | accuracy | 5.9 | 3.4 | 21.9 |
| **Language Understanding** | | | | |
| ContractNLI | macro-F1 | 70.8 | 59.0 | 74.7 |
| ANLI ★ | macro-F1 | 4.1 | 8.4 | 19.0 |
| HellaSwag ★ | accuracy | 19.0 | 11.0 | 48.9 |
| ACOS | per-review F1 | 19.6 | 25.4 | 38.9 |
| FinEntity | macro-F1 | 78.4 | 52.2 | 76.7 |
| iSarcasmEval | sarcasm F1, track A English | 0.0 | 0.0 | 0.0 |
| VAST | macro-F1 | 54.3 | 51.0 | 61.4 |
| NLI4CT | macro-F1 | 12.9 | 0.0 | 15.1 |
| RAGTruth | F1 on hallucinated class | 26.1 | 30.0 | 44.3 |
| **Retrieval & Classification** | | | | |
| BANKING77 ★ | macro-F1 | 85.8 | 86.7 | 90.6 |
| CLINC150+OOS ★ | macro-F1 | 87.6 | 86.2 | 91.8 |
| BRIGHT ★ | nDCG@10 | 20.9 | 17.5 | 18.6 |
| Amazon ESCI | macro-F1 | 25.7 | 21.3 | 29.8 |
| PhishNChips | accuracy (verdict) | 0.4 | 0.0 | 4.0 |
| HoVer | accuracy | 62.9 | 58.5 | 69.4 |
| **Tools & Automation** | | | | |
| BFCL ★ | case exact accuracy | 76.4 | 54.2 | 79.0 |
| ToolRet | nDCG@10 | 53.8 | 49.4 | 55.7 |
| API-Bank ★ | accuracy | 59.7 | 7.9 | 29.4 |
| Home appliance simulator | case exact accuracy | 75.0 | 27.3 | 79.5 |
| When2Call | accuracy | 53.7 | 47.2 | 62.5 |
| **Arts & Human Taste** | | | | |
| BPoMP | accuracy | 72.7 | 67.3 | 0.0 |
| Humicroedit | accuracy | 0.0 | 0.0 | 4.5 |
| POP909-CL | accuracy | 0.6 | 0.0 | 1.2 |
| cfcolor | accuracy | 0.8 | 0.0 | 0.6 |
| Habermas Machine | accuracy | 12.4 | 11.6 | 10.1 |
| New Yorker caption matching | accuracy | 38.9 | 32.3 | 44.1 |

</details>

- **Strong:** tool selection and API calls (BFCL 76.4, API-Bank 59.7, ToolRet 53.8, smart-home control 75.0), intent
  classification (BANKING77 85.8, CLINC150 87.6), entity sentiment (FinEntity 78.4), long-document NLI (ContractNLI
  70.8, HoVer 62.9) and original-vs-perturbed text (BPoMP 72.7). Its Tools & Automation area (64.0) is above
  Crivo-150M's (60.7).
- **Weak:** expert knowledge and multi-step reasoning (GPQA, HLE, MMLU-Pro, GSM8K, chess, code execution), causal
  inference (CLadder 8.1, against 61.0 for Crivo-150M), adversarial NLI and commonsense completion (ANLI 4.1, HellaSwag
  19.0), sarcasm and subjective "taste" benchmarks.

## Quick start

### Standalone (no kit)

```python
from huggingface_hub import snapshot_download
import sys; path = snapshot_download("tardellirs/crivo-68m"); sys.path.insert(0, path)
from crivo import Crivo

model = Crivo(path)                                 # CUDA if available, else CPU
out = model.answer(
    state={"ticket": "My card was charged twice for the same order."},
    questions={
        "intent": {"type": "choice", "instructions": "What does the customer want?",
                   "criteria": {"refund": "Money back for the duplicate charge",
                                "cancel": "Cancel the order",
                                "info": "Just asking for information"}},
        "urgent": {"type": "noul", "instructions": "The customer needs a fast response."},
    })
print(out["intent"]["choice"], out["intent"]["probabilities"])   # refund {'refund': 0.80, 'cancel': 0.17, 'info': 0.03}
print(out["urgent"]["noul"])                                     # 0.06
```

| type | `criteria` | answer |
|---|---|---|
| `choice` | `{key: description or null}`, 2+ options (no upper limit) | `choice` + `probabilities` over every key |
| `noul` | optional `{"true": ..., "false": ...}` describing both poles | `noul` = P(statement in `instructions` is true) |
| `score` | ordered list of level descriptions, lowest first | `score` = expected level + `probabilities` |

`state` and `instructions` can be strings or JSON objects (rendered as compact JSON with sorted keys, exactly as in
training).

### Decision Index kit

```bash
pip install "decision-index[transformers] @ git+https://github.com/apolinario/decision-index@9eb2dbe"
hf download tardellirs/crivo-68m --local-dir crivo-68m && cd crivo-68m
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-68m
```

The engine defaults are the settings of the reported run: one option order (`tta=1`), fp32 scoring head, SDPA
attention, `batch_size=32`, `token_budget=65536`, `max_length=32768`, bf16 autocast on CUDA. The state is tokenized
once per request and all questions are batched. It raises `Unsupported` only for a question above `max_length` tokens,
and there is none in the 0.3 suite.

## Model details

Same design as the other Crivo models. Each question is one sequence

```
[CLS] instructions [SEP] state [SEP] [OPT] option_1 [OPT] option_2 … [OPT] option_n
```

(`option_i` = `key: description`). The hidden state at each `[OPT]` marker plus a question-type embedding goes through a
small MLP scorer (run in fp32) to one logit per option; a softmax over the question's options gives the distribution.
Options of a question see each other, and option order was randomized during training. Training sequences were up to
8,192 tokens; longer inputs are read whole with rotary positions up to 32,768.

Files: `model.safetensors` (fp32 weights), `config.json` (ModernBERT config + `decision` block), tokenizer files,
`model.py`, `crivo.py` (standalone API), `decision_index_engine.py` (kit engine).

## Training

Full fine-tune of `cross-encoder/ettin-reranker-68m-v1` in three stages, on exactly the data of Crivo-400M, with
Crivo-400M as the teacher in stages 1 and 2. Every example is one question with a target distribution over its options.

### Data

| Stage | Questions | Content |
|---|---:|---|
| 1 | 638,306 | [Bekko System One dataset v0](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0) (≤1,200 per subset) · code-generated decision families with exact labels (policies with exceptions and precedence, opaque label rules, schedules with constraints, ordinal severity, dates, propositional logic, executed code and tests, link-based phishing, multi-select) · format-invariance augmentations (opaque keys, key-only options, distractors, "none of the above") · 328 task families from [`tasksource/tasksource-jev-typed-decisions`](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) · sibling sources of the suite's skills (argument quality, SEntFiN, RAGBench, Hermes function calling, xLAM irrelevance, bitext banking and e-commerce intents, CUAD contracts) · train splits of the public benchmarks' sources, in the original layout and rewritten · [`tasksource/procedural-typed-decisions`](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) · original-vs-perturbed public-domain text · open teacher labels from [`Mapika/decider`](https://github.com/Mapika/decider) (open Qwen models) · CFPB complaint routing · commit-message matching (CommitPackFT) |
| 2 | 169,437 | tool selection ([ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE), [glaive](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)), CLadder variants not used by the suite, SEntFiN entity sentiment, synthetic smart-home cases, 6,500 synthetic "which API next" dialogues with catalogs of ~50 APIs (gold calls from [Hermes function calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)), + replay of stage 1 |
| 3 | 24,635 | long context up to 8,192 tokens: ContractNLI, RAGTruth, NLI4CT and other long inputs, plus 5,000 "which API next" dialogues with long catalogs (~6k tokens) |

No data labeled by Jev or any other proprietary decision API was used.

### Distillation

Crivo-400M's logits were computed once over the stage-1 and stage-2 training rows (it agrees with the gold label on
82.7% and 87.8% of them). The loss is

`α · (CE + 0.5 · Brier) + (1 − α) · τ² · KL(teacher ‖ student)`, with α = 0.5,

τ = 2 for questions with more than two options and τ = 1 otherwise. Sources where the teacher agrees with the gold label
on fewer than 60% of their training rows (mostly math, logic and spatial reasoning, human-preference ratings and many
tasksource families), plus smart-home control, CLadder and RAGTruth, train on the gold label only: 163 sources in
stage 1 and 46 in stage 2. Stage 3 (long context) trains on the gold label only.

### Recipe

| Stage | Init | Steps | LR (encoder / head) | Max length | Tokens per micro-batch × accumulation | Teacher |
|---|---|---:|---|---:|---|---|
| 1 | Ettin reranker 68M | 5,918 (1 epoch, weight EMA 0.999) | 6e-5 / 3e-4 | 2,048 | 16,384 × 2 | Crivo-400M |
| 2 | stage 1 (final weights) | 2,330 | 4e-5 / 2e-4 | 2,048 | 16,384 × 2 | Crivo-400M |
| 3 | stage 2 | 4,156 | 3e-5 / 1.5e-4 | 8,192 | 12,288 × 2 | none (gold) |
| Final | — | average of the stage-2 and stage-3 weights | | | | |

AdamW (weight decay 0.01), linear warmup (5% stage 1, 3% after) and linear decay, bf16 autocast, option order shuffled
for every `choice` example. Every checkpoint choice (stage-1 final vs EMA as the stage-2 init; stage 2, stage 3 or their
average; interpolations of 10% and 20% toward the base model, WiSE-FT) was made on our own panels: held-out new-domain
decision families, a development set of the board's skills, the Akita benchmark and a public-suite sample. None of them
used the board. The stage-2/3 average was best; both WiSE interpolations were worse.

**Compute.** One NVIDIA A100 40 GB (Lambda Cloud). Crivo-68M and Crivo-17M were trained and evaluated in the same run
of about 7 hours: teacher logits, the 68M, the three 17M arms, the panels and both complete public suites.

## Contamination firewall

The training files are the ones of Crivo-400M, checked at three levels:

1. **Suite test rows.** Every training example was checked against the 0.3 suite: same normalized state and question,
   same distinctive state, or a shared passage of ≥120 characters were dropped.
2. **Our own panels.** Strict leaf-level filter (any string of ≥25 normalized characters) against the held-out panels
   used for checkpoint selection, so the selection stayed honest.
3. **Final audit** of the exact training files: every question against all 563,686 questions of the 0.3 suite
   (normalized state + instructions): **0 matches**; the rule above re-applied to every row: 0 hits.

The API-Bank train split was **not** used: its dialogues and API catalogs repeat those of the test pool. The API-Bank
skill comes only from the synthetic dialogues above, built from Hermes function-calling data and filtered against the
suite. Benchmarks whose sources have no clean train split were used for evaluation only. The teacher only labeled these
same training rows; it never saw suite items.

## Intended use and limitations

**Intended use:** fast, local, probabilistic decisions over text or JSON state, as a first pass or a component inside an
agent or pipeline: routing, intent and topic classification, policy gates, tool and API selection, relevance checks,
NLI-style verification — where Crivo-150M or Crivo-400M is too heavy.

- **Private parts not yet scored.** The public index is 20% of the board's Full score; how the model does on the
  private tests, especially tasks from new domains (the weak spot of the small Crivo models: Crivo-32M scored 0.92
  there), is not known yet. Validate on your own data before relying on it for an unfamiliar task.
- **English only.**
- **Little world knowledge, no step-by-step reasoning**: near chance on expert QA, math word problems, chess, causal
  inference and code execution. Use Crivo-150M or Crivo-400M for those.
- **Distilled model**: it inherits Crivo-400M's biases and errors; where the teacher was wrong, the student often is
  too.
- **The smart-home score is optimistic**: the synthetic training cases reuse the suite's request templates (new houses
  only).
- **Long inputs beyond 8,192 tokens** are read without truncation but were never seen in training.
- **No temperature fitted**: treat probabilities as scores to threshold on your own data.
- Not for high-stakes decisions about people without human review.

## License

**CC BY-NC 4.0**, the same as its teacher, Crivo-400M: several training sources are licensed for non-commercial use
only (for example ANLI, CC BY-NC 4.0, and some Bekko dataset subsets). Upstream dataset licenses and terms continue to
apply to the data. The base model ([Ettin reranker 68M](https://huggingface.co/cross-encoder/ettin-reranker-68m-v1)) is
Apache-2.0.

## See also

- The Crivo family, by size: [Crivo-17M](https://huggingface.co/tardellirs/crivo-17m) ·
  [Crivo-32M](https://huggingface.co/tardellirs/crivo-32m) · **Crivo-68M** ·
  [Crivo-150M](https://huggingface.co/tardellirs/crivo-150m) ·
  [Crivo-400M](https://huggingface.co/tardellirs/crivo-400m) (public 39.79, its teacher) · [Crivo-1B](https://huggingface.co/tardellirs/crivo-1b) (public 47.77).
- [Crivo collection](https://huggingface.co/collections/tardellirs/crivo-6acaa1731d71f839970f04a6).

## Acknowledgements

- [Ettin](https://huggingface.co/jhu-clsp/ettin-encoder-68m) (JHU CLSP) and the
  [Ettin reranker 68M](https://huggingface.co/cross-encoder/ettin-reranker-68m-v1) used as the starting point.
- [Bekko System One](https://github.com/hotchpotch/bekko-system-one) (hotchpotch) for the open training dataset.
- [Dinah-0](https://huggingface.co/Lukitaduarte/dinah-0) (Lukitaduarte) for the `[OPT]` readout design this model follows.
- [tasksource](https://huggingface.co/tasksource) for procedural and multi-task typed-decision data.
- [Decider](https://github.com/Mapika/decider) (Mapika) for its open teacher data.
- NousResearch, MadeAgents, Team-ACE, Glaive, Galileo, The Atticus Project, bitext and BigCode for the open
  datasets behind the sibling sources.
- The [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) maintainers and its
  [open kit](https://github.com/apolinario/decision-index), which made a reproducible comparison possible.

## Citation

```bibtex
@misc{crivo68m2026,
  title        = {Crivo-68M: a typed-decision encoder distilled from Crivo-400M},
  author       = {Tardelli Stekel},
  year         = {2026},
  howpublished = {Hugging Face model card},
  url          = {https://huggingface.co/tardellirs/crivo-68m}
}
```
