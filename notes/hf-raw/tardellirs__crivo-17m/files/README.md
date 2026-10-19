---
license: cc-by-nc-4.0
language:
- en
base_model: cross-encoder/ettin-reranker-17m-v1
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
- name: Crivo-17M
  results:
  - task:
      type: text-classification
      name: Typed decisions (choice / noul / score)
    dataset:
      name: Decision Index 0.3
      type: decision-index
    metrics:
    - type: public_index
      value: 22.15
      name: Public index (self-run, complete public suite)
    source:
      name: Decision Index 0.3 public suite, official kit
      url: https://huggingface.co/datasets/tardellirs/decision-index-results/tree/main/runs/crivo-17m
---

# Crivo-17M

**The smallest Crivo: a 17M-parameter typed-decision encoder for CPUs, edge devices and very high request volumes.**

[![Decision Index public](https://img.shields.io/badge/Decision%20Index%200.3-public%2022.15-2a78d6)](#decision-index-03--public-index-self-run-complete)
[![Full score](https://img.shields.io/badge/Full%20score-pending-999)](#decision-index-03--public-index-self-run-complete)
[![Parameters](https://img.shields.io/badge/parameters-17M-555)](#model-details)
[![Language](https://img.shields.io/badge/language-English-555)](#intended-use-and-limitations)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-555)](#license)

Crivo-17M reads a *state* and a set of typed *questions* (`choice`, `noul`, `score`) and returns a probability for
every option in a single encoder pass. It does not generate text. It is the smallest member of the Crivo family, below
[Crivo-32M](https://huggingface.co/tardellirs/crivo-32m): 7 layers and a hidden size of 256, light enough to sit inside
every step of an agent loop on a CPU. It was trained on the data of
[Crivo-400M](https://huggingface.co/tardellirs/crivo-400m), distilled from
[Crivo-150M](https://huggingface.co/tardellirs/crivo-150m).

| | |
|---|---|
| Architecture | Ettin / ModernBERT encoder, 7 layers, hidden 256, `[OPT]` readout + MLP scorer |
| Parameters | 16.9M (16,865,025) |
| Base model | [`cross-encoder/ettin-reranker-17m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-17m-v1) |
| Teacher | [`tardellirs/crivo-150m`](https://huggingface.co/tardellirs/crivo-150m) (knowledge distillation; chosen over Crivo-400M and Crivo-68M) |
| Decision Index 0.3 | public index **22.15** (self-run, 140,178 / 140,178 requests answered); Full score pending (maintainer-run) |
| Latency, 1× A100 40 GB, bf16 (one request at a time) | median 7.7 ms · mean 9.7 ms · p95 12.4 ms per request |
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

Crivo-17M é o menor modelo da família Crivo: um encoder de decisões tipadas com 17M de parâmetros (Ettin/ModernBERT,
7 camadas), treinado com os dados do Crivo-400M e destilado do Crivo-150M. Ele lê um *state* e perguntas `choice`,
`noul` ou `score` e devolve uma probabilidade para cada opção numa única passada, sem gerar texto. No índice público do
Decision Index 0.3 marca **22,15** (run completo publicado, sem truncamento), com mediana de
7,7 ms por requisição numa A100. O Full score, com os testes privados dos mantenedores, ainda não saiu. Foi treinado só
com dados abertos, com verificação de contaminação contra a suíte. Funciona em inglês. Licença CC BY-NC 4.0 (uso não
comercial).

</details>

## Results

### Decision Index 0.3 — public index (self-run, complete)

Run with the official kit ([apolinario/decision-index](https://github.com/apolinario/decision-index), commit
`9eb2dbe`, edition 0.3) on the complete public suite, with the engine shipped in this repository: every one of the
140,178 scoreable requests answered, none unsupported, nothing truncated. Scores are chance-corrected skill (0 = random
guessing, 100 = perfect), weighted as the board does. Full results:
[`tardellirs/decision-index-results` → `runs/crivo-17m`](https://huggingface.co/datasets/tardellirs/decision-index-results/tree/main/runs/crivo-17m).

The board's main ranking (Full score) is 20% this public index, 50% private tests of the same skills and 30% private
tasks from new domains, run by the maintainers. **The Full score will be added here once it is published.**

| Model | Public index | Knowledge & Reasoning | Language | Retrieval & Classification | Tools & Automation | Arts & Human Taste |
|---|---:|---:|---:|---:|---:|---:|
| **Crivo-17M** | **22.15** | 1.2 | 21.0 | 40.1 | 36.5 | 17.0 |
| Crivo-32M | 25.06 | 2.0 | 25.6 | 46.7 | 36.7 | 18.5 |

A comparison with other models of similar size will be added together with the Full score, since the public index alone does not rank models by Full score.

<details>
<summary><b>Per-benchmark skill (0–100, chance-corrected) — click to expand</b></summary>

★ = gold benchmark (weight 1.2 within its area). The index uses the native metric listed, chance-corrected.

| Benchmark | Native metric | Crivo-17M | Crivo-32M |
|---|---|---:|---:|
| **Knowledge & Reasoning** | | | |
| GPQA Diamond ★ | accuracy | 0.0 | 0.0 |
| GSM8K | accuracy | 0.9 | 0.3 |
| ChessBench | accuracy | 0.0 | 0.0 |
| MuSR | accuracy | 1.3 | 0.0 |
| SATA-Bench | case exact accuracy | 0.0 | 0.0 |
| CRUXEval | accuracy | 5.1 | 8.4 |
| CLadder | accuracy | 0.0 | 8.4 |
| HLE ★ | accuracy | 0.0 | 0.0 |
| MMLU-Pro ★ | accuracy | 0.0 | 1.1 |
| BBH fixed-option tasks ★ | accuracy | 2.0 | 1.4 |
| WinoGrande ★ | accuracy | 4.3 | 3.4 |
| **Language Understanding** | | | |
| ContractNLI | macro-F1 | 59.0 | 59.0 |
| ANLI ★ | macro-F1 | 0.0 | 8.4 |
| HellaSwag ★ | accuracy | 7.4 | 11.0 |
| ACOS | per-review F1 | 8.0 | 25.4 |
| FinEntity | macro-F1 | 52.1 | 52.2 |
| iSarcasmEval | sarcasm F1, track A English | 0.0 | 0.0 |
| VAST | macro-F1 | 39.7 | 51.0 |
| NLI4CT | macro-F1 | 1.9 | 0.0 |
| RAGTruth | F1 on hallucinated class | 27.9 | 30.0 |
| **Retrieval & Classification** | | | |
| BANKING77 ★ | macro-F1 | 79.3 | 86.7 |
| CLINC150+OOS ★ | macro-F1 | 79.6 | 86.2 |
| BRIGHT ★ | nDCG@10 | 13.3 | 17.5 |
| Amazon ESCI | macro-F1 | 6.9 | 21.3 |
| PhishNChips | accuracy (verdict) | 16.1 | 0.0 |
| HoVer | accuracy | 34.9 | 58.5 |
| **Tools & Automation** | | | |
| BFCL ★ | case exact accuracy | 47.6 | 54.2 |
| ToolRet | nDCG@10 | 39.5 | 49.4 |
| API-Bank ★ | accuracy | 49.0 | 7.9 |
| Home appliance simulator | case exact accuracy | 3.4 | 27.3 |
| When2Call | accuracy | 38.1 | 47.2 |
| **Arts & Human Taste** | | | |
| BPoMP | accuracy | 61.6 | 67.3 |
| Humicroedit | accuracy | 2.4 | 0.0 |
| POP909-CL | accuracy | 1.0 | 0.0 |
| cfcolor | accuracy | 2.6 | 0.0 |
| Habermas Machine | accuracy | 7.7 | 11.6 |
| New Yorker caption matching | accuracy | 26.9 | 32.3 |

</details>

- **Strong:** intent classification (BANKING77 79.3, CLINC150 79.6), long-document NLI (ContractNLI 59.0, level with
  Crivo-32M), entity sentiment (FinEntity 52.1), tool and API selection (API-Bank 49.0, BFCL 47.6, ToolRet 39.5) and
  original-vs-perturbed text (BPoMP 61.6). Against Crivo-32M it is ahead on API-Bank (49.0 vs 7.9) and PhishNChips.
- **Weak:** anything that needs world knowledge or reasoning (Knowledge & Reasoning 1.2), adversarial NLI and
  commonsense completion (ANLI 0.0, HellaSwag 7.4), smart-home control (3.4), product relevance (ESCI 6.9), claim
  verification (HoVer 34.9), aspect sentiment (ACOS 8.0), sarcasm and subjective "taste" benchmarks.

## Quick start

### Standalone (no kit)

```python
from huggingface_hub import snapshot_download
import sys; path = snapshot_download("tardellirs/crivo-17m"); sys.path.insert(0, path)
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
print(out["intent"]["choice"], out["intent"]["probabilities"])   # refund {'refund': 0.85, 'cancel': 0.06, 'info': 0.09}
print(out["urgent"]["noul"])                                     # 0.10
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
hf download tardellirs/crivo-17m --local-dir crivo-17m && cd crivo-17m
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-17m
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

Full fine-tune of `cross-encoder/ettin-reranker-17m-v1` in three stages, on exactly the data of Crivo-400M, with a
larger Crivo as the teacher in stages 1 and 2. Every example is one question with a target distribution over its
options.

### Data

| Stage | Questions | Content |
|---|---:|---|
| 1 | 638,306 | [Bekko System One dataset v0](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0) (≤1,200 per subset) · code-generated decision families with exact labels (policies with exceptions and precedence, opaque label rules, schedules with constraints, ordinal severity, dates, propositional logic, executed code and tests, link-based phishing, multi-select) · format-invariance augmentations (opaque keys, key-only options, distractors, "none of the above") · 328 task families from [`tasksource/tasksource-jev-typed-decisions`](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) · sibling sources of the suite's skills (argument quality, SEntFiN, RAGBench, Hermes function calling, xLAM irrelevance, bitext banking and e-commerce intents, CUAD contracts) · train splits of the public benchmarks' sources, in the original layout and rewritten · [`tasksource/procedural-typed-decisions`](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) · original-vs-perturbed public-domain text · open teacher labels from [`Mapika/decider`](https://github.com/Mapika/decider) (open Qwen models) · CFPB complaint routing · commit-message matching (CommitPackFT) |
| 2 | 169,437 | tool selection ([ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE), [glaive](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)), CLadder variants not used by the suite, SEntFiN entity sentiment, synthetic smart-home cases, 6,500 synthetic "which API next" dialogues with catalogs of ~50 APIs (gold calls from [Hermes function calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)), + replay of stage 1 |
| 3 | 24,635 | long context up to 8,192 tokens: ContractNLI, RAGTruth, NLI4CT and other long inputs, plus 5,000 "which API next" dialogues with long catalogs (~6k tokens) |

No data labeled by Jev or any other proprietary decision API was used.

### Distillation and the choice of teacher

The loss is

`α · (CE + 0.5 · Brier) + (1 − α) · τ² · KL(teacher ‖ student)`, with α = 0.5,

τ = 2 for questions with more than two options and τ = 1 otherwise. Teacher logits were computed once over the stage-1
and stage-2 training rows. Sources where the teacher agrees with the gold label on fewer than 60% of their training
rows, plus smart-home control, CLadder and RAGTruth, train on the gold label only. Stage 3 (long context) trains on the
gold label only.

Crivo-400M is 23× the size of this model, so three students were trained in parallel, one per teacher, with the same
data, stages and hyperparameters:

| Arm | Teacher | Teacher agrees with gold (stage 1 / 2 rows) | Arm's best checkpoint | Same-skill dev panel | New-domain panel | Akita (accuracy) |
|---|---|---:|---|---:|---:|---:|
| A | [Crivo-400M](https://huggingface.co/tardellirs/crivo-400m) (23×) | 82.7% / 87.8% | stage-2/3 average | 0.355 | 0.128 | 32.3% |
| B | [Crivo-68M](https://huggingface.co/tardellirs/crivo-68m) (4×) | 80.6% / 84.5% | stage-2/3 average, 10% WiSE-FT | 0.349 | 0.117 | 37.7% |
| **C** | [**Crivo-150M**](https://huggingface.co/tardellirs/crivo-150m) (9×) | 70.7% / 79.9% | **stage 3** | **0.398** | 0.108 | 33.8% |

The panel columns are our own held-out panels (not board numbers): the core of a development set of the board's skills
and a proxy built from held-out new-domain decision families. Arm C was selected: it led on the same-skill panel and on
a public-suite sample, while arms A and B were slightly ahead on the new-domain panel, which is noisy at this size.
With Crivo-150M as the teacher, 302 sources in stage 1 and 60 in stage 2 trained on gold only, among them the
code-generated families added for Crivo-400M.

### Recipe (released arm, C)

| Stage | Init | Steps | LR (encoder / head) | Max length | Tokens per micro-batch | Teacher |
|---|---|---:|---|---:|---:|---|
| 1 | Ettin reranker 17M | 16,318 (2 epochs, weight EMA 0.999) | 1e-4 / 5e-4 | 2,048 | 24,576 | Crivo-150M |
| 2 | stage-1 EMA | 3,134 | 6e-5 / 3e-4 | 2,048 | 24,576 | Crivo-150M |
| 3 | stage 2 | 3,287 | 4e-5 / 2e-4 | 8,192 | 24,576 | none (gold) |

AdamW (weight decay 0.01), linear warmup (5% stage 1, 3% after) and linear decay, bf16 autocast, no gradient
accumulation, option order shuffled for every `choice` example. The released weights are the end of stage 3. Every
checkpoint choice (stage-1 final vs EMA as the stage-2 init; stage 2, stage 3 or their average; interpolations of 10%
and 20% toward the base model, WiSE-FT; and the arm) was made on our own panels: held-out new-domain decision families,
a development set of the board's skills, the Akita benchmark and a public-suite sample. None of them used the board.

**Compute.** One NVIDIA A100 40 GB (Lambda Cloud). Crivo-17M and Crivo-68M were trained and evaluated in the same run
of about 7 hours: teacher logits, the 68M, the three 17M arms (in parallel on the same GPU), the panels and both
complete public suites.

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
suite. Benchmarks whose sources have no clean train split were used for evaluation only. The teachers only labeled
these same training rows; they never saw suite items.

## Intended use and limitations

**Intended use:** fast, local decisions over text or JSON state — routing, intent and topic classification, policy
gates, tool selection, relevance checks — where even a 32M model is too heavy (CPU, edge, very high request volume).

- **Private parts not yet scored.** The public index is 20% of the board's Full score; how the model does on the
  private tests, especially tasks from new domains (the weak spot of the small Crivo models: Crivo-32M scored 0.92
  there), is not known yet. Validate on your own data before relying on it for an unfamiliar task.
- **English only.**
- **Almost no world knowledge and no step-by-step reasoning**: at or near chance on expert QA, math, commonsense and
  causal-inference benchmarks. Use Crivo-150M or Crivo-400M for those.
- **Smart-home control is weak** (3.4) even though synthetic smart-home cases were in training; do not use it for
  device-control decisions without testing.
- **Distilled model**: it inherits Crivo-150M's biases and errors; where the teacher was wrong, the student often is
  too.
- **Long inputs beyond 8,192 tokens** are read without truncation but were never seen in training.
- **No temperature fitted**: treat probabilities as scores to threshold on your own data.
- Not for high-stakes decisions about people without human review.

## License

**CC BY-NC 4.0**, the same as its teacher, Crivo-150M: several training sources are licensed for non-commercial use
only (for example ANLI, CC BY-NC 4.0, and some Bekko dataset subsets). Upstream dataset licenses and terms continue to
apply to the data. The base model ([Ettin reranker 17M](https://huggingface.co/cross-encoder/ettin-reranker-17m-v1)) is
Apache-2.0.

## See also

- The Crivo family, by size: **Crivo-17M** · [Crivo-32M](https://huggingface.co/tardellirs/crivo-32m) ·
  [Crivo-68M](https://huggingface.co/tardellirs/crivo-68m) · [Crivo-150M](https://huggingface.co/tardellirs/crivo-150m)
  (v1, rev. `7a6c745`, its teacher) · [Crivo-400M](https://huggingface.co/tardellirs/crivo-400m) (public 39.79) · [Crivo-1B](https://huggingface.co/tardellirs/crivo-1b) (public 47.77).
- [Crivo collection](https://huggingface.co/collections/tardellirs/crivo-6acaa1731d71f839970f04a6).

## Acknowledgements

- [Ettin](https://huggingface.co/jhu-clsp/ettin-encoder-17m) (JHU CLSP) and the
  [Ettin reranker 17M](https://huggingface.co/cross-encoder/ettin-reranker-17m-v1) used as the starting point.
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
@misc{crivo17m2026,
  title        = {Crivo-17M: a small typed-decision encoder distilled from Crivo-150M},
  author       = {Tardelli Stekel},
  year         = {2026},
  howpublished = {Hugging Face model card},
  url          = {https://huggingface.co/tardellirs/crivo-17m}
}
```
