---
license: cc-by-nc-4.0
language:
- en
base_model: cross-encoder/ettin-reranker-150m-v1
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
- name: Crivo-150M
  results:
  - task:
      type: text-classification
      name: Typed decisions (choice / noul / score)
    dataset:
      name: Decision Index 0.3
      type: decision-index
    metrics:
    - type: public_index
      value: 34.38
      name: Public index (self-run, complete public suite; version 2)
    source:
      name: Decision Index 0.3 public suite, official kit
      url: https://huggingface.co/datasets/tardellirs/decision-index-results/tree/main/runs/crivo-150m-v2
---

# Crivo-150M

**A compact typed-decision encoder for agents and pipelines. Version 2, distilled from Crivo-400M.**

[![Decision Index public](https://img.shields.io/badge/Decision%20Index%200.3-public%2034.38-2a78d6)](#decision-index-03--public-index-self-run-complete)
[![Full score](https://img.shields.io/badge/Full%20score-pending-999)](#decision-index-03--public-index-self-run-complete)
[![Version](https://img.shields.io/badge/version-2-555)](#versions)
[![Parameters](https://img.shields.io/badge/parameters-149M-555)](#model-details)
[![Language](https://img.shields.io/badge/language-English-555)](#intended-use-and-limitations)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-555)](#license)

Crivo-150M reads a *state* and a set of typed *questions* (`choice`, `noul`, `score`) and returns a probability for
every option in a single encoder pass. It does not generate text. It is built for the classify, route, gate and verify
decisions that agents and pipelines make thousands of times, where a generative model is slow and expensive.

**This is version 2.** It has the same architecture and size as version 1, but it was retrained with the recipe of the
newer Crivo models: the training data of [Crivo-400M](https://huggingface.co/tardellirs/crivo-400m), with Crivo-400M
as its teacher. Version 1 (board Full 22.21, public 36.39) stays available at revision `7a6c745`; see
[Versions](#versions) for what changed and why.

<p align="center">
  <img src="https://huggingface.co/tardellirs/crivo-150m/resolve/main/assets/pareto.png" width="960"
       alt="Decision Index 0.3, Full score versus served parameters for 147 submissions, coloured by architecture. The Pareto frontier is drawn as a staircase through the frontier models; Crivo-150M version 1 (Full 22.21) is the frontier point at 150M parameters.">
</p>
<p align="center"><sub>Board standing of <b>version 1</b> (Full 22.21, on the Pareto frontier at 150M parameters).
This chart will be updated when the Full score of version 2 is published. Generated from the board's
<code>data/index.json</code> by <a href="https://huggingface.co/tardellirs/crivo-150m/blob/main/assets/plot_pareto.py"><code>assets/plot_pareto.py</code></a>.</sub></p>

| | |
|---|---|
| Architecture | Ettin / ModernBERT encoder, 22 layers, hidden 768, `[OPT]` readout + MLP scorer |
| Parameters | 149.6M (149,610,241) |
| Base model | [`cross-encoder/ettin-reranker-150m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1) |
| Teacher | [`tardellirs/crivo-400m`](https://huggingface.co/tardellirs/crivo-400m) (knowledge distillation) |
| Decision Index 0.3 | public index **34.38** (self-run, 140,178 / 140,178 requests answered); Full score pending (maintainer-run) |
| Latency, 1× A100 40 GB, bf16 (one request at a time) | median 18.0 ms · mean 26.1 ms · p95 44.4 ms per request |
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

Esta é a versão 2 do Crivo-150M, um encoder de decisões tipadas com 149M de parâmetros (Ettin/ModernBERT). Ele lê um
*state* e perguntas `choice`, `noul` ou `score` e devolve uma probabilidade para cada opção numa única passada, sem
gerar texto. A v2 tem a mesma arquitetura da v1, mas foi retreinada com os dados do Crivo-400M e destilada dele. No
índice público do Decision Index 0.3 marca **34,38** (run completo publicado), contra 36,39 da v1: a v2 usa bem menos
dados no formato exato das fontes da suíte e mais famílias de tarefas novas, e nos nossos painéis internos generaliza
melhor (domínios novos e mesmas skills). O Full score da v2, com os testes privados dos mantenedores, ainda não saiu; a
v1 (Full 22,21) continua disponível na revisão `7a6c745`. Treinado só com dados abertos, com verificação de
contaminação contra a suíte. Funciona em inglês. Licença CC BY-NC 4.0 (uso não comercial).

</details>

## Contents

- [Results](#results)
- [Versions](#versions)
- [Quick start](#quick-start)
- [Model details](#model-details)
- [Training](#training)
- [Contamination firewall](#contamination-firewall)
- [Intended use and limitations](#intended-use-and-limitations)
- [License](#license)
- [See also](#see-also)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Results

### Decision Index 0.3 — public index (self-run, complete)

Run with the official kit ([apolinario/decision-index](https://github.com/apolinario/decision-index), commit
`9eb2dbe`, edition 0.3) on the complete public suite, with the engine shipped in this repository: every one of the
140,178 scoreable requests answered, none unsupported, nothing truncated. Scores are chance-corrected skill (0 = random
guessing, 100 = perfect), weighted as the board does. Full results:
[`tardellirs/decision-index-results` → `runs/crivo-150m-v2`](https://huggingface.co/datasets/tardellirs/decision-index-results/tree/main/runs/crivo-150m-v2).

The board's main ranking (Full score) is 20% this public index, 50% private tests of the same skills and 30% private
tasks from new domains, run by the maintainers. **The Full score of version 2 will be added here once it is
published.**

| Version | Public index | Raw | Knowledge & Reasoning | Language | Retrieval & Classification | Tools & Automation | Arts & Human Taste |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 (`7a6c745`) | 36.39 | 52.45 | 11.8 | 41.8 | 52.2 | 60.7 | 10.1 |
| **v2 (this revision)** | **34.38** | 51.32 | 8.3 | 34.5 | 48.2 | 62.8 | 21.8 |

<details>
<summary><b>Per-benchmark skill (0–100, chance-corrected), v1 and v2 — click to expand</b></summary>

★ = gold benchmark (weight 1.2 within its area). The index uses the native metric listed, chance-corrected.

| Benchmark | Native metric | v1 | **v2** |
|---|---|---:|---:|
| **Knowledge & Reasoning** | | | |
| GPQA Diamond ★ | accuracy | 0.0 | 1.4 |
| GSM8K | accuracy | 2.0 | 1.2 |
| ChessBench | accuracy | 0.0 | 0.0 |
| MuSR | accuracy | 26.7 | 21.1 |
| SATA-Bench | case exact accuracy | 0.0 | 2.6 |
| CRUXEval | accuracy | 7.3 | 14.0 |
| CLadder | accuracy | 61.0 | 32.1 |
| HLE ★ | accuracy | 0.0 | 0.4 |
| MMLU-Pro ★ | accuracy | 3.4 | 4.3 |
| BBH fixed-option tasks ★ | accuracy | 11.6 | 8.2 |
| WinoGrande ★ | accuracy | 21.9 | 9.4 |
| **Language Understanding** | | | |
| ContractNLI | macro-F1 | 74.7 | 71.4 |
| ANLI ★ | macro-F1 | 19.0 | 7.6 |
| HellaSwag ★ | accuracy | 48.9 | 39.0 |
| ACOS | per-review F1 | 38.9 | 19.6 |
| FinEntity | macro-F1 | 76.7 | 78.5 |
| iSarcasmEval | sarcasm F1, track A English | 0.0 | 0.0 |
| VAST | macro-F1 | 61.4 | 58.5 |
| NLI4CT | macro-F1 | 15.1 | 17.1 |
| RAGTruth | F1 on hallucinated class | 44.3 | 23.5 |
| **Retrieval & Classification** | | | |
| BANKING77 ★ | macro-F1 | 90.6 | 85.7 |
| CLINC150+OOS ★ | macro-F1 | 91.8 | 89.5 |
| BRIGHT ★ | nDCG@10 | 18.6 | 21.9 |
| Amazon ESCI | macro-F1 | 29.8 | 21.5 |
| PhishNChips | accuracy (verdict) | 4.0 | 0.1 |
| HoVer | accuracy | 69.4 | 60.3 |
| **Tools & Automation** | | | |
| BFCL ★ | case exact accuracy | 79.0 | 68.0 |
| ToolRet | nDCG@10 | 55.7 | 57.0 |
| API-Bank ★ | accuracy | 29.4 | 70.1 |
| Home appliance simulator | case exact accuracy | 79.5 | 63.6 |
| When2Call | accuracy | 62.5 | 52.6 |
| **Arts & Human Taste** | | | |
| BPoMP | accuracy | 0.0 | 73.0 |
| Humicroedit | accuracy | 4.5 | 0.9 |
| POP909-CL | accuracy | 1.2 | 1.1 |
| cfcolor | accuracy | 0.6 | 0.4 |
| Habermas Machine | accuracy | 10.1 | 11.7 |
| New Yorker caption matching | accuracy | 44.1 | 43.4 |

</details>

- **Strong:** tool and API selection (API-Bank 70.1, BFCL 68.0, ToolRet 57.0, smart-home control 63.6), intent
  classification (BANKING77 85.7, CLINC150 89.5), entity sentiment (FinEntity 78.5), long-document NLI (ContractNLI
  71.4, HoVer 60.3), stance (VAST 58.5) and original-vs-perturbed text (BPoMP 73.0).
- **Weak:** expert knowledge and multi-step reasoning (GPQA, HLE, MMLU-Pro, GSM8K, chess), adversarial NLI (ANLI 7.6),
  sarcasm and most subjective "taste" benchmarks.
- **Against v1:** large gains on API-Bank (+40.7) and BPoMP (+73.0); large drops on CLadder (−28.9), RAGTruth (−20.8),
  ACOS (−19.3), smart-home control (−15.9), WinoGrande (−12.5), ANLI (−11.4) and BFCL (−11.0). The drops are mostly on
  benchmarks where v1 trained on many rows of the benchmark's own source in the suite's exact layout (see
  [Versions](#versions)).

A comparison with other models of similar size will be added together with the Full score, since the public index alone does not rank models by Full score.

### Internal selection panels (our own evaluation, not board scores)

Checkpoints were chosen on our own panels, never on the board and never on the complete public suite (it was run only
on the final package). Two of the panels, run on v1 and v2 with the same engine, show what the new recipe changed. These
are **internal proxies**, not substitutes for the board's private tests, and no Full score is estimated here.

| Panel | What it measures | v1 | **v2** |
|---|---|---:|---:|
| ND-proxy | mean chance-corrected skill on 21 held-out decision families from outside the suite (written rules and policies with exceptions, schedules, dates, propositional logic, code outputs and tests, arithmetic in context, rewritten items, product-domain decisions, and others), chosen because they track the board's new-domain scores across 26 models | 0.150 | **0.224** |
| SS-core | mean chance-corrected skill on our own development sets for 8 of the board's skills (BFCL, BANKING77, CLINC150, When2Call, NLI4CT, ANLI, Amazon ESCI, smart-home control), not built from suite rows | 0.459 | **0.556** |
| Akita v1 (originals) | accuracy on the 130 original items of the Akita benchmark | 43.8% | 38.5% |

v2 is better on both proxies of the private parts of the board and worse on Akita. One caveat on the ND-proxy: the v2
training data includes code-generated decision families of related kinds (policies with exceptions, schedules, dates,
logic, executed code). They were filtered against the panel rows, but the panel partly measures transfer from them.

## Versions

The repository holds two versions in its git history. `main` is version 2.

| Version | Revision | Recipe | Public index (self-run) | Full score (board) |
|---|---|---|---:|---:|
| **v2 (current)** | `a5e21d8067c5c37ba5f88cd498306ae5abfd625d` | data of Crivo-400M, distilled from Crivo-400M, end of stage 3 | 34.38 | pending |
| v1 | [`7a6c745`](https://huggingface.co/tardellirs/crivo-150m/tree/7a6c7453c06be576e1b48aeb70429318402791d7) | own data mix, no teacher, average of stages 2 and 3 | 36.39 | **22.21** (private same skills 27.74, new domains 9.77) |

v1 is the board entry submitted in October 2026 (public 36.39, Full 22.21, on the Pareto frontier at 150M parameters
at the time). It is also the teacher of [Crivo-17M](https://huggingface.co/tardellirs/crivo-17m) and of Crivo-32M v1.
To load it:

```python
from huggingface_hub import snapshot_download
path = snapshot_download("tardellirs/crivo-150m", revision="7a6c7453c06be576e1b48aeb70429318402791d7")
```

**What changed in v2, and why.** v1 scored well on private tests of familiar skills (27.74) but poorly on tasks from
new domains (9.77): it had learned the suite's request layouts more than the skills behind them. v2 trades some of that
fit for breadth:

- **Less exact source.** v1 trained on 209,707 rows of the benchmarks' own train splits, all in the suite's layout, and
  on Bekko subsets capped at 6,000 decisions each. v2 caps every benchmark source at 5,000 rows, about half of them
  rewritten into other layouts (about 70,000 such rows in stage 1), and caps Bekko subsets at 1,200.
- **More new task families.** Code-generated decision families with exact labels, 328 tasksource task families,
  format-invariance augmentations, sibling sources of the suite's skills, open teacher labels and synthetic "which API
  next" dialogues (the API-Bank train split itself is not used; it repeats the test pool).
- **A teacher.** v2 is distilled from Crivo-400M (public 39.79) on the same rows.
- **Engine.** The scoring head now runs in fp32 and the engine is the one shared by all current Crivo models; the
  interface, the inputs and the outputs are unchanged.

The result is a lower public index (36.39 → 34.38), with losses concentrated on benchmarks v1 had seen in exact layout
(CLadder, RAGTruth, ACOS, smart-home control, BFCL), and higher scores on both internal proxies of the private parts
(ND-proxy 0.150 → 0.224, SS-core 0.459 → 0.556). Whether this turns into a higher Full score is what the
maintainers' run will show. If your task is close to one of the benchmarks where v1 is clearly ahead, v1 may serve
you better; pin its revision.

## Quick start

### Standalone (no kit)

```python
from huggingface_hub import snapshot_download
import sys; path = snapshot_download("tardellirs/crivo-150m"); sys.path.insert(0, path)
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
print(out["intent"]["choice"], out["intent"]["probabilities"])   # refund {'refund': 0.69, 'cancel': 0.08, 'info': 0.23}
print(out["urgent"]["noul"])                                     # 0.64
```

Question types:

| type | `criteria` | answer |
|---|---|---|
| `choice` | `{key: description or null}`, 2+ options (no upper limit) | `choice` + `probabilities` over every key |
| `noul` | optional `{"true": ..., "false": ...}` describing both poles | `noul` = P(statement in `instructions` is true) |
| `score` | ordered list of level descriptions, lowest first | `score` = expected level + `probabilities` |

`state` and `instructions` can be strings or JSON objects (rendered as compact JSON with sorted keys, exactly as in
training). Pin a revision (`snapshot_download(..., revision=...)`) if you need a fixed version.

### Decision Index kit

```bash
pip install "decision-index[transformers] @ git+https://github.com/apolinario/decision-index@9eb2dbe"
hf download tardellirs/crivo-150m --local-dir crivo-150m && cd crivo-150m
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-150m-v2
```

The engine defaults are the settings of the reported run: one option order (`tta=1`), fp32 scoring head, SDPA
attention, `batch_size=32`, `token_budget=65536`, `max_length=32768`, bf16 autocast on CUDA. The state is tokenized
once per request and all questions are batched. It raises `Unsupported` only for a question above `max_length` tokens,
and there is none in the 0.3 suite.

## Model details

**Input.** Each question is one sequence:

```
[CLS] instructions [SEP] state [SEP] [OPT] option_1 [OPT] option_2 … [OPT] option_n
```

where `option_i` is `key: description` (or just `key`). `noul` is two options (`false`, `true`); `score` levels are
options `0: …`, `1: …`, …. A special token `[OPT]` (initialized from `[SEP]`) marks each option.

**Readout.** The encoder's hidden state at each `[OPT]` marker, plus a learned question-type embedding, goes through a
small MLP scorer (LayerNorm → Linear → GELU → Linear, run in fp32) to one logit per option; a softmax over the options
of the question gives the distribution. All questions of a request are batched together; options of a question see each
other (full bidirectional attention), and option order was randomized during training so the model does not rely on
position.

**Context.** ModernBERT/Ettin uses rotary position embeddings with alternating local (128-token) and global attention.
Training used sequences up to 8,192 tokens. Longer inputs (in the 0.3 suite, mostly ToolRet requests carrying tool
documents of ~23k tokens) are read whole with rotary positions up to 32,768 rather than refused or truncated.

**Files.** `model.safetensors` (fp32 weights), `config.json` (ModernBERT config + `decision` block), tokenizer files,
`model.py` (architecture, rendering, tokenization), `crivo.py` (standalone API), `decision_index_engine.py` (kit engine).

## Training

Full fine-tune of `cross-encoder/ettin-reranker-150m-v1` in three stages, on exactly the training files of Crivo-400M,
with Crivo-400M as the teacher in stages 1 and 2. Every example is one question with a target distribution over its
options. The released weights are the end of stage 3.

### Data

| Stage | Questions | Content |
|---|---:|---|
| 1 | 638,306 | [Bekko System One dataset v0](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0) (≤1,200 per subset) · code-generated decision families with exact labels (policies with exceptions and precedence, opaque label rules, schedules with constraints, ordinal severity, dates, propositional logic, executed code and tests, link-based phishing, multi-select) · format-invariance augmentations (opaque keys, key-only options, distractors, "none of the above") · 328 task families from [`tasksource/tasksource-jev-typed-decisions`](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) · sibling sources of the suite's skills (argument quality, SEntFiN, RAGBench, Hermes function calling, xLAM irrelevance, bitext banking and e-commerce intents, CUAD contracts) · train splits of the public benchmarks' sources, in the original layout and rewritten (≤5,000 per source) · [`tasksource/procedural-typed-decisions`](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) · original-vs-perturbed public-domain text · open teacher labels from [`Mapika/decider`](https://github.com/Mapika/decider) (open Qwen models) · CFPB complaint routing · commit-message matching (CommitPackFT) |
| 2 | 169,437 | tool selection ([ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE), [glaive](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)), CLadder variants not used by the suite, SEntFiN entity sentiment, synthetic smart-home cases, 6,500 synthetic "which API next" dialogues with catalogs of ~50 APIs (gold calls from [Hermes function calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)), + replay of stage 1 |
| 3 | 24,635 | long context up to 8,192 tokens: ContractNLI, RAGTruth, NLI4CT and other long inputs, plus 5,000 "which API next" dialogues with long catalogs (~6k tokens) |

40 tasksource families were held out of training entirely and kept as a reserved check. No data labeled by Jev or any
other proprietary decision API was used.

### Distillation

Crivo-400M's logits were computed once over the stage-1 and stage-2 training rows (it agrees with the gold label on
82.7% and 87.8% of them); the same logits were used for Crivo-68M and Crivo-17M. The loss is

`α · (CE + 0.5 · Brier) + (1 − α) · τ² · KL(teacher ‖ student)`, with α = 0.5,

τ = 2 for questions with more than two options and τ = 1 otherwise. Sources where the teacher agrees with the gold label
on fewer than 60% of their training rows (mostly math, logic and spatial reasoning, human-preference ratings and many
tasksource families), plus smart-home control, CLadder and RAGTruth, train on the gold label only: 163 sources in stage
1 and 46 in stage 2. Stage 3 (long context) trains on the gold label only.

### Recipe

| Stage | Init | Steps | LR (encoder / head) | Max length | Tokens per micro-batch × accumulation | Teacher |
|---|---|---:|---|---:|---|---|
| 1 | Ettin reranker 150M | 5,941 (1 epoch) | 3e-5 / 3e-4 | 2,048 | 8,192 × 4 | Crivo-400M |
| 2 | stage 1 (final weights) | 2,406 | 2e-5 / 2e-4 | 2,048 | 8,192 × 4 | Crivo-400M |
| 3 | stage 2 | 2,589 | 1.5e-5 / 1e-4 | 8,192 | 8,192 × 4 | none (gold) |
| Final | — | end of stage 3 | | | | |

AdamW (weight decay 0.01), linear warmup (5% stage 1, 3% after) and linear decay, bf16 autocast, option order shuffled
for every `choice` example. A weight EMA (0.999) was kept during stage 1; the final stage-1 weights scored slightly
better than the EMA and initialized stage 2. Every checkpoint choice (stage-1 final vs EMA; stage 2, stage 3 or their
average; interpolations of 10% and 20% toward the base model, WiSE-FT) was made on our own panels: held-out new-domain
decision families, a development set of the board's skills, the Akita benchmark and a public-suite sample. None of them
used the board, and the complete public suite was run only on the final package. Stage 3 was best overall (SS-core 0.480 →
0.556 over stage 2, while the ND-proxy dipped from 0.242 to 0.224); the stage-2/3 average and both WiSE interpolations
were worse.

**Compute.** One NVIDIA A100 40 GB (Lambda Cloud), shared with the training of Crivo-32M v2. Training, panels,
packaging and the complete public suite of both models took about 5 h 15 min of wall time; the teacher logits were
reused from the Crivo-68M / Crivo-17M run. A reproducibility check on the package (500 questions, batch size 32 vs 4)
gave the same answer on every question (max probability difference 0).

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

### Intended use

Fast, local, probabilistic decisions over text or JSON state, as a cheap first pass or a component inside an agent or
pipeline:

- routing, intent and topic classification;
- policy and guardrail gates;
- tool and API selection;
- relevance and hallucination checks, NLI-style verification.

### Out of scope

- Open-ended generation, extraction or any task that needs free-text output.
- Expert knowledge questions, math and multi-step reasoning.
- High-stakes decisions about people without human review.

### Limitations

- **Private parts not yet scored.** The public index is 20% of the board's Full score. v1 was weak on the board's
  private tasks from new domains (9.77); v2 is better on our internal proxy for them, but the board's number for v2 is
  not known yet. Validate on your own data before relying on it for an unfamiliar task.
- **Some skills regressed from v1.** On the public suite v2 is clearly below v1 on causal inference (CLadder),
  hallucination detection (RAGTruth), aspect sentiment (ACOS), smart-home control, function calling (BFCL) and
  adversarial NLI (ANLI). If one of these is your use case, compare with v1 (revision `7a6c745`).
- **English only.** Training data is almost entirely English; other languages are untested and likely weak.
- **Little world knowledge, no step-by-step reasoning.** Expect near-chance results on expert QA (GPQA, MMLU-Pro, HLE),
  math word problems, chess and code execution.
- **Distilled model.** It inherits Crivo-400M's biases and errors; where the teacher was wrong, the student often is
  too.
- **Format sensitivity.** Very different phrasings of a task may lose accuracy. Descriptive option text
  (`key: description`) works better than bare keys.
- **The smart-home score is optimistic.** The synthetic training cases reuse the suite's request templates (new houses
  only).
- **Long inputs beyond 8,192 tokens** are read without truncation but were never seen in training; quality there is
  uncertain.
- **No temperature fitted.** Treat probabilities as scores to threshold on your own data.

## License

**CC BY-NC 4.0**, the same as its teacher, Crivo-400M, and as v1: several training sources are licensed for
non-commercial use only (for example ANLI, CC BY-NC 4.0, and some Bekko dataset subsets), so the weights are released
for non-commercial use. Upstream dataset licenses and terms continue to apply to the data. The base model
([Ettin reranker 150M](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1)) is Apache-2.0.

## See also

- The Crivo family, by size: [Crivo-17M](https://huggingface.co/tardellirs/crivo-17m) ·
  [Crivo-32M](https://huggingface.co/tardellirs/crivo-32m) · [Crivo-68M](https://huggingface.co/tardellirs/crivo-68m) ·
  **Crivo-150M** · [Crivo-400M](https://huggingface.co/tardellirs/crivo-400m) (public 39.79, its teacher) · [Crivo-1B](https://huggingface.co/tardellirs/crivo-1b) (public 47.77).
- [Crivo collection](https://huggingface.co/collections/tardellirs/crivo-6acaa1731d71f839970f04a6).

## Acknowledgements

- [Ettin](https://huggingface.co/jhu-clsp/ettin-encoder-150m) (JHU CLSP) and the
  [Ettin reranker](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1) used as the starting point.
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
@misc{crivo150m2026,
  title        = {Crivo-150M: a small typed-decision encoder},
  author       = {Tardelli Stekel},
  year         = {2026},
  note         = {Version 2, distilled from Crivo-400M},
  howpublished = {Hugging Face model card},
  url          = {https://huggingface.co/tardellirs/crivo-150m}
}
```
