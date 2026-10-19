---
license: cc-by-nc-4.0
language:
- en
base_model: cross-encoder/ettin-reranker-32m-v1
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
- name: Crivo-32M
  results:
  - task:
      type: text-classification
      name: Typed decisions (choice / noul / score)
    dataset:
      name: Decision Index 0.3
      type: decision-index
    metrics:
    - type: public_index
      value: 27.34
      name: Public index (self-run, complete public suite; version 2)
    source:
      name: Decision Index 0.3 public suite, official kit
      url: https://huggingface.co/datasets/tardellirs/decision-index-results/tree/main/runs/crivo-32m-v2
---

# Crivo-32M

**A 32M-parameter typed-decision encoder. Version 2, distilled from Crivo-400M.**

[![Decision Index public](https://img.shields.io/badge/Decision%20Index%200.3-public%2027.34-2a78d6)](#decision-index-03--public-index-self-run-complete)
[![Full score](https://img.shields.io/badge/Full%20score-pending-999)](#decision-index-03--public-index-self-run-complete)
[![Version](https://img.shields.io/badge/version-2-555)](#versions)
[![Parameters](https://img.shields.io/badge/parameters-32M-555)](#model-details)
[![Language](https://img.shields.io/badge/language-English-555)](#intended-use-and-limitations)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-555)](#license)

Crivo-32M reads a *state* and a set of typed *questions* (`choice`, `noul`, `score`) and returns a probability for
every option in a single encoder pass. It does not generate text. It is light enough for CPUs and edge devices and fast
enough to sit inside every step of an agent loop.

**This is version 2.** It has the same architecture and size as version 1, but it was retrained with the recipe of the
newer Crivo models: the training data of [Crivo-400M](https://huggingface.co/tardellirs/crivo-400m), with Crivo-400M
as its teacher (v1 was distilled from Crivo-150M v1). Version 1 (board Full 11.87, public 25.06) stays available at
revision `bbf3f19`; see [Versions](#versions) for what changed and why.

| | |
|---|---|
| Architecture | Ettin / ModernBERT encoder, 10 layers, hidden 384, `[OPT]` readout + MLP scorer |
| Parameters | 32.0M (32,033,665) |
| Base model | [`cross-encoder/ettin-reranker-32m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1) |
| Teacher | [`tardellirs/crivo-400m`](https://huggingface.co/tardellirs/crivo-400m) (knowledge distillation) |
| Decision Index 0.3 | public index **27.34** (self-run, 140,178 / 140,178 requests answered); Full score pending (maintainer-run) |
| Latency, 1× A100 40 GB, bf16 (one request at a time) | median 26.9 ms · mean 30.5 ms · p95 41.3 ms per request, **measured while the GPU was shared with another training job** (inflated; see below) |
| Latency of v1 (same architecture and size) | median 10.3 ms on an unshared A100 40 GB; 7.3 ms on the board's RTX PRO 6000 (measured by the maintainers) |
| Context | trained up to 2,048 tokens; reads longer inputs whole, up to 32,768, without truncation |
| Language | English |
| License | CC BY-NC 4.0 (see [License](#license)) |

The public-suite run of v2 shared its A100 with the training of Crivo-150M v2, so its request times are not
representative. The network, its size and the engine's batching are the same as in v1, so the per-request cost should
be close to v1's; the board measures latency on its own hardware.

Crivo follows the typed-decision interface popularized by TypeSafe's Jev (a state plus typed questions in, a
probability for every option out) and runs unmodified in the open
[Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) kit. It is an independent open model,
not affiliated with or endorsed by TypeSafe, and not trained on Jev outputs. *Crivo* is Portuguese for "sieve";
*passar pelo crivo*, to pass through the sieve, means to be judged and sorted.

<details>
<summary><b>Resumo em português</b></summary>

Esta é a versão 2 do Crivo-32M, um encoder de decisões tipadas com 32M de parâmetros (Ettin/ModernBERT). Ele lê um
*state* e perguntas `choice`, `noul` ou `score` e devolve uma probabilidade para cada opção numa única passada, sem
gerar texto. A v2 tem a mesma arquitetura da v1, mas foi retreinada com os dados do Crivo-400M e destilada dele (a v1
vinha do Crivo-150M). No índice público do Decision Index 0.3 marca **27,34** (run completo publicado), contra 25,06 da
v1, e nos nossos painéis internos generaliza bem melhor: a v1 estava perto do acaso em domínios novos (0,92 no placar).
O Full score da v2, com os testes privados dos mantenedores, ainda não saiu; a v1 (Full 11,87) continua disponível na
revisão `bbf3f19`. A latência medida no nosso run está inflada porque a GPU estava dividida com outro treino. Treinado
só com dados abertos, com verificação de contaminação contra a suíte. Funciona em inglês. Licença CC BY-NC 4.0 (uso não
comercial).

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
[`tardellirs/decision-index-results` → `runs/crivo-32m-v2`](https://huggingface.co/datasets/tardellirs/decision-index-results/tree/main/runs/crivo-32m-v2).

The board's main ranking (Full score) is 20% this public index, 50% private tests of the same skills and 30% private
tasks from new domains, run by the maintainers. **The Full score of version 2 will be added here once it is
published.**

| Version | Public index | Raw | Knowledge & Reasoning | Language | Retrieval & Classification | Tools & Automation | Arts & Human Taste |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 (`bbf3f19`) | 25.06 | 43.20 | 2.0 | 25.6 | 46.7 | 36.7 | 18.5 |
| **v2 (this revision)** | **27.34** | 45.75 | 2.7 | 25.0 | 45.6 | 50.0 | 19.0 |

<details>
<summary><b>Per-benchmark skill (0–100, chance-corrected), v1 and v2 — click to expand</b></summary>

★ = gold benchmark (weight 1.2 within its area). The index uses the native metric listed, chance-corrected.

| Benchmark | Native metric | v1 | **v2** |
|---|---|---:|---:|
| **Knowledge & Reasoning** | | | |
| GPQA Diamond ★ | accuracy | 0.0 | 3.4 |
| GSM8K | accuracy | 0.3 | 0.6 |
| ChessBench | accuracy | 0.0 | 0.4 |
| MuSR | accuracy | 0.0 | 3.8 |
| SATA-Bench | case exact accuracy | 0.0 | 0.0 |
| CRUXEval | accuracy | 8.4 | 9.0 |
| CLadder | accuracy | 8.4 | 3.0 |
| HLE ★ | accuracy | 0.0 | 0.0 |
| MMLU-Pro ★ | accuracy | 1.1 | 1.2 |
| BBH fixed-option tasks ★ | accuracy | 1.4 | 7.8 |
| WinoGrande ★ | accuracy | 3.4 | 1.0 |
| **Language Understanding** | | | |
| ContractNLI | macro-F1 | 59.0 | 58.4 |
| ANLI ★ | macro-F1 | 8.4 | 2.0 |
| HellaSwag ★ | accuracy | 11.0 | 10.6 |
| ACOS | per-review F1 | 25.4 | 14.6 |
| FinEntity | macro-F1 | 52.2 | 64.7 |
| iSarcasmEval | sarcasm F1, track A English | 0.0 | 0.0 |
| VAST | macro-F1 | 51.0 | 47.3 |
| NLI4CT | macro-F1 | 0.0 | 5.3 |
| RAGTruth | F1 on hallucinated class | 30.0 | 29.7 |
| **Retrieval & Classification** | | | |
| BANKING77 ★ | macro-F1 | 86.7 | 86.0 |
| CLINC150+OOS ★ | macro-F1 | 86.2 | 84.7 |
| BRIGHT ★ | nDCG@10 | 17.5 | 16.2 |
| Amazon ESCI | macro-F1 | 21.3 | 19.7 |
| PhishNChips | accuracy (verdict) | 0.0 | 0.0 |
| HoVer | accuracy | 58.5 | 56.7 |
| **Tools & Automation** | | | |
| BFCL ★ | case exact accuracy | 54.2 | 69.0 |
| ToolRet | nDCG@10 | 49.4 | 45.8 |
| API-Bank ★ | accuracy | 7.9 | 56.3 |
| Home appliance simulator | case exact accuracy | 27.3 | 23.9 |
| When2Call | accuracy | 47.2 | 49.8 |
| **Arts & Human Taste** | | | |
| BPoMP | accuracy | 67.3 | 68.3 |
| Humicroedit | accuracy | 0.0 | 0.0 |
| POP909-CL | accuracy | 0.0 | 0.0 |
| cfcolor | accuracy | 0.0 | 4.5 |
| Habermas Machine | accuracy | 11.6 | 8.7 |
| New Yorker caption matching | accuracy | 32.3 | 32.8 |

</details>

- **Strong:** intent classification (BANKING77 86.0, CLINC150 84.7), tool and API selection (BFCL 69.0, API-Bank
  56.3, ToolRet 45.8), original-vs-perturbed text (BPoMP 68.3), entity sentiment (FinEntity 64.7), long-document NLI
  (ContractNLI 58.4, HoVer 56.7).
- **Weak:** anything that needs world knowledge or multi-step reasoning (GPQA, MMLU-Pro, GSM8K, MuSR, CLadder,
  WinoGrande), adversarial NLI (ANLI 2.0), sarcasm and most "taste" benchmarks.
- **Against v1:** the Tools & Automation area rises from 36.7 to 50.0 (API-Bank +48.4, BFCL +14.8), FinEntity +12.5
  and BBH +6.4; the main drops are ACOS (−10.8), ANLI (−6.4) and CLadder (−5.4).

A comparison with other models of similar size will be added together with the Full score, since the public index alone does not rank models by Full score.

### Internal selection panels (our own evaluation, not board scores)

Checkpoints were chosen on our own panels, never on the board and never on the complete public suite (it was run only
on the final package). Two of the panels, run on v1 and v2 with the same engine, show what the new recipe changed. These
are **internal proxies**, not substitutes for the board's private tests, and no Full score is estimated here.

| Panel | What it measures | v1 | **v2** |
|---|---|---:|---:|
| ND-proxy | mean chance-corrected skill on 21 held-out decision families from outside the suite (written rules and policies with exceptions, schedules, dates, propositional logic, code outputs and tests, arithmetic in context, rewritten items, product-domain decisions, and others), chosen because they track the board's new-domain scores across 26 models | 0.019 | **0.126** |
| SS-core | mean chance-corrected skill on our own development sets for 8 of the board's skills (BFCL, BANKING77, CLINC150, When2Call, NLI4CT, ANLI, Amazon ESCI, smart-home control), not built from suite rows | 0.390 | **0.453** |
| Akita v1 (originals) | accuracy on the 130 original items of the Akita benchmark | not run | 40.8% |

v1's ND-proxy (0.019) is at the noise floor of the panel, consistent with its near-chance new-domain score on the board
(0.92); v2 is clearly above it. One caveat: the v2 training data includes code-generated decision families of related
kinds (policies with exceptions, schedules, dates, logic, executed code). They were filtered against the panel rows, but
the panel partly measures transfer from them.

## Versions

The repository holds two versions in its git history. `main` is version 2.

| Version | Revision | Recipe | Public index (self-run) | Full score (board) |
|---|---|---|---:|---:|
| **v2 (current)** | `93970f0562b85138e2c44b65ed0bc46a6d082f14` | data of Crivo-400M, distilled from Crivo-400M, end of stage 2 | 27.34 | pending |
| v1 | [`bbf3f19`](https://huggingface.co/tardellirs/crivo-32m/tree/bbf3f1924523f530b3a9e621f2f20f314ea462aa) | Crivo-150M v1 data + augmentations, distilled from Crivo-150M v1, stage average + WiSE-FT | 25.06 | **11.87** (private same skills 18.84, new domains 0.92) |

v1 is the board entry submitted in October 2026 (public 25.06, Full 11.87, on the Pareto frontier at 32M parameters at
the time). To load it:

```python
from huggingface_hub import snapshot_download
path = snapshot_download("tardellirs/crivo-32m", revision="bbf3f1924523f530b3a9e621f2f20f314ea462aa")
```

**What changed in v2, and why.** v1 kept much of its teacher's skill on private tests of familiar skills (18.84) but
was near chance on tasks from new domains (0.92): it had learned the suite's request layouts more than the skills
behind them. v2 changes the data and the teacher:

- **Less exact source.** v1 trained on about 209,000 rows of the benchmarks' own train splits, all in the suite's
  layout, and on Bekko subsets capped at 6,000 decisions each. v2 caps every benchmark source at 5,000 rows, about half
  of them rewritten into other layouts (about 70,000 such rows in stage 1), and caps Bekko subsets at 1,200.
- **More new task families.** Code-generated decision families with exact labels, sibling sources of the suite's
  skills, open teacher labels and synthetic "which API next" dialogues, on top of the tasksource families and
  format-invariance augmentations v1 already had (the API-Bank train split itself is not used; it repeats the test
  pool).
- **A stronger teacher.** v2 is distilled from Crivo-400M (public 39.79) instead of Crivo-150M v1 (36.39).
- **Longer stage 1, no long-context stage.** Two epochs of stage 1; the long-context stage 3 was trained but scored
  lower on our panels, so v2 is the end of stage 2 (see [Training](#training)).
- **Engine.** The engine is now the one shared by all current Crivo models (same defaults as v1: one option order,
  fp32 scoring head); the interface, the inputs and the outputs are unchanged.

The result is a higher public index (25.06 → 27.34), driven by tools and API selection, and higher scores on both
internal proxies of the private parts (ND-proxy 0.019 → 0.126, SS-core 0.390 → 0.453). Whether this turns into a higher
Full score is what the maintainers' run will show.

## Quick start

### Standalone (no kit)

```python
from huggingface_hub import snapshot_download
import sys; path = snapshot_download("tardellirs/crivo-32m"); sys.path.insert(0, path)
from crivo import Crivo

model = Crivo(path)                                   # CUDA if available, else CPU
out = model.answer(
    state={"ticket": "My card was charged twice for the same order."},
    questions={
        "intent": {"type": "choice", "instructions": "What does the customer want?",
                   "criteria": {"refund": "Money back for the duplicate charge",
                                "cancel": "Cancel the order",
                                "info": "Just asking for information"}},
        "urgent": {"type": "noul", "instructions": "The customer needs a fast response."},
    })
print(out["intent"]["choice"], out["intent"]["probabilities"])   # refund {'refund': 0.85, 'cancel': 0.01, 'info': 0.14}
print(out["urgent"]["noul"])                                     # 0.45
```

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
hf download tardellirs/crivo-32m --local-dir crivo-32m && cd crivo-32m
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-32m-v2
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
Options of a question see each other, and option order was randomized during training. The released weights were
trained on sequences up to 2,048 tokens; longer inputs are read whole with rotary positions up to 32,768.

Files: `model.safetensors` (fp32 weights), `config.json` (ModernBERT config + `decision` block), tokenizer files,
`model.py`, `crivo.py` (standalone API), `decision_index_engine.py` (kit engine).

## Training

Full fine-tune of `cross-encoder/ettin-reranker-32m-v1` on exactly the training files of Crivo-400M, with Crivo-400M as
the teacher. Every example is one question with a target distribution over its options. The released weights are the
end of stage 2.

### Data

| Stage | Questions | Content |
|---|---:|---|
| 1 | 638,306 | [Bekko System One dataset v0](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0) (≤1,200 per subset) · code-generated decision families with exact labels (policies with exceptions and precedence, opaque label rules, schedules with constraints, ordinal severity, dates, propositional logic, executed code and tests, link-based phishing, multi-select) · format-invariance augmentations (opaque keys, key-only options, distractors, "none of the above") · 328 task families from [`tasksource/tasksource-jev-typed-decisions`](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) · sibling sources of the suite's skills (argument quality, SEntFiN, RAGBench, Hermes function calling, xLAM irrelevance, bitext banking and e-commerce intents, CUAD contracts) · train splits of the public benchmarks' sources, in the original layout and rewritten (≤5,000 per source) · [`tasksource/procedural-typed-decisions`](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) · original-vs-perturbed public-domain text · open teacher labels from [`Mapika/decider`](https://github.com/Mapika/decider) (open Qwen models) · CFPB complaint routing · commit-message matching (CommitPackFT) |
| 2 | 169,437 | tool selection ([ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE), [glaive](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)), CLadder variants not used by the suite, SEntFiN entity sentiment, synthetic smart-home cases, 6,500 synthetic "which API next" dialogues with catalogs of ~50 APIs (gold calls from [Hermes function calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)), + replay of stage 1 |
| 3 (trained, not released) | 24,635 | long context up to 8,192 tokens: ContractNLI, RAGTruth, NLI4CT and other long inputs, plus 5,000 "which API next" dialogues with long catalogs (~6k tokens) |

40 tasksource families were held out of training entirely and kept as a reserved check. No data labeled by Jev or any
other proprietary decision API was used.

### Distillation

Crivo-400M's logits were computed once over the stage-1 and stage-2 training rows (it agrees with the gold label on
82.7% and 87.8% of them); the same logits were used for Crivo-68M, Crivo-17M and Crivo-150M v2. The loss is

`α · (CE + 0.5 · Brier) + (1 − α) · τ² · KL(teacher ‖ student)`, with α = 0.5,

τ = 2 for questions with more than two options and τ = 1 otherwise. Sources where the teacher agrees with the gold label
on fewer than 60% of their training rows (mostly math, logic and spatial reasoning, human-preference ratings and many
tasksource families), plus smart-home control, CLadder and RAGTruth, train on the gold label only: 163 sources in stage
1 and 46 in stage 2.

### Recipe

| Stage | Init | Steps | LR (encoder / head) | Max length | Tokens per micro-batch × accumulation | Teacher |
|---|---|---:|---|---:|---|---|
| 1 | Ettin reranker 32M | 16,318 (2 epochs) | 1e-4 / 5e-4 | 2,048 | 24,576 × 1 | Crivo-400M |
| 2 | stage 1 (final weights) | 3,134 | 5e-5 / 2.5e-4 | 2,048 | 24,576 × 1 | Crivo-400M |
| Final | — | end of stage 2 | | | | |
| 3 (not released) | stage 2 | 3,287 | 4e-5 / 1.5e-4 | 8,192 | 24,576 × 1 | none (gold) |

AdamW (weight decay 0.01), linear warmup (5% stage 1, 3% after) and linear decay, bf16 autocast, option order shuffled
for every `choice` example. A weight EMA (0.999) was kept during stage 1; the final stage-1 weights scored about the same
as the EMA and initialized stage 2. Every checkpoint choice (stage-1 final vs EMA; stage 2, stage 3 or their average;
interpolations of 10% and 20% toward the base model, WiSE-FT) was made on our own panels: held-out new-domain decision
families, a development set of the board's skills, the Akita benchmark and a public-suite sample. None of them used the
board, and the complete public suite was run only on the final package. Stage 2 was best: stage 3 lowered the
skills panel (SS-core 0.453 → 0.436) while raising the ND-proxy only slightly (0.126 → 0.134), and the stage-2/3
average and both WiSE interpolations were also worse.

**Compute.** One NVIDIA A100 40 GB (Lambda Cloud), shared with the training of Crivo-150M v2. Training, panels,
packaging and the public suite of the 32M took about 3 h 35 min of wall time; the teacher logits were reused from the
Crivo-68M / Crivo-17M run. A reproducibility check on the package (500 questions, batch size 32 vs 4) gave the same
answer on every question (max probability difference 0).

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

**Intended use:** fast, local decisions over text or JSON state (routing, intent and topic classification, policy
gates, tool and API selection, relevance checks) where a larger model is too heavy: CPU, edge, very high request
volume.

- **Private parts not yet scored.** The public index is 20% of the board's Full score. v1 was near chance on the
  board's private tasks from new domains (0.92); v2 is clearly better on our internal proxy for them, but the board's
  number for v2 is not known yet. Validate on your own data before relying on it for an unfamiliar task.
- **English only.**
- **Very little world knowledge and no step-by-step reasoning**: near-zero on expert QA, math, commonsense reasoning
  and causal inference benchmarks. Use a larger Crivo for those.
- **Long inputs.** The released weights saw sequences up to 2,048 tokens in training (v1 saw up to 8,192). Longer inputs
  are read whole, without truncation, but quality there is less certain than for the larger Crivo models.
- **Distilled model**: it inherits Crivo-400M's biases and errors; where the teacher was wrong, the student often is
  too.
- **The smart-home score is optimistic**: the synthetic training cases reuse the suite's request templates (new houses
  only).
- **No temperature fitted**: treat probabilities as scores to threshold on your own data.
- Not for high-stakes decisions about people without human review.

## License

**CC BY-NC 4.0**, the same as its teacher, Crivo-400M, and as v1: several training sources are non-commercial (for
example ANLI, CC BY-NC 4.0, and some Bekko dataset subsets). Upstream dataset licenses and terms continue to apply. The
base model ([Ettin reranker 32M](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1)) is Apache-2.0.

## See also

- The Crivo family, by size: [Crivo-17M](https://huggingface.co/tardellirs/crivo-17m) · **Crivo-32M** ·
  [Crivo-68M](https://huggingface.co/tardellirs/crivo-68m) · [Crivo-150M](https://huggingface.co/tardellirs/crivo-150m)
  · [Crivo-400M](https://huggingface.co/tardellirs/crivo-400m) (public 39.79, its teacher) · [Crivo-1B](https://huggingface.co/tardellirs/crivo-1b) (public 47.77).
- [Crivo collection](https://huggingface.co/collections/tardellirs/crivo-6acaa1731d71f839970f04a6).

## Acknowledgements

- [Ettin](https://huggingface.co/jhu-clsp/ettin-encoder-32m) (JHU CLSP) and the
  [Ettin reranker 32M](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1) used as the starting point.
- [Bekko System One](https://github.com/hotchpotch/bekko-system-one) (hotchpotch) for the open training dataset.
- [Dinah-0](https://huggingface.co/Lukitaduarte/dinah-0) (Lukitaduarte) for the `[OPT]` readout design.
- [tasksource](https://huggingface.co/tasksource) for procedural and multi-task typed-decision data.
- [Decider](https://github.com/Mapika/decider) (Mapika) for its open teacher data.
- NousResearch, MadeAgents, Team-ACE, Glaive, Galileo, The Atticus Project, bitext and BigCode for the open
  datasets behind the sibling sources.
- The [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) maintainers and its
  [open kit](https://github.com/apolinario/decision-index).

## Citation

```bibtex
@misc{crivo32m2026,
  title        = {Crivo-32M: a small typed-decision encoder},
  author       = {Tardelli Stekel},
  year         = {2026},
  note         = {Version 2, distilled from Crivo-400M},
  howpublished = {Hugging Face model card},
  url          = {https://huggingface.co/tardellirs/crivo-32m}
}
```
