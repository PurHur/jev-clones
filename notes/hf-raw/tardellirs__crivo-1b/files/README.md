---
license: cc-by-nc-4.0
language:
- en
base_model: cross-encoder/ettin-reranker-1b-v1
pipeline_tag: text-classification
library_name: transformers
tags:
- decision-model
- jev-style
- typed-decisions
- system-one
- decision-index
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
- name: Crivo-1B
  results:
  - task:
      type: text-classification
      name: Typed decisions (choice / noul / score)
    dataset:
      name: Decision Index 0.3
      type: decision-index
    metrics:
    - type: public_index
      value: 47.77
      name: Public index (self-run, complete public suite)
    source:
      name: Decision Index 0.3 public suite, official kit
      url: https://huggingface.co/datasets/tardellirs/decision-index-results/tree/main/runs/crivo-1b
---

# Crivo-1B

**A 1B-parameter typed-decision encoder, the largest of the Crivo models.**

[![Decision Index public](https://img.shields.io/badge/Decision%20Index%200.3-public%2047.77-2a78d6)](#decision-index-03--public-index-self-run-complete)
[![Full score](https://img.shields.io/badge/Full%20score-pending-999)](#decision-index-03--public-index-self-run-complete)
[![Parameters](https://img.shields.io/badge/parameters-1.03B-555)](#model-details)
[![Language](https://img.shields.io/badge/language-English-555)](#intended-use-and-limitations)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-555)](#license)

Crivo-1B reads a *state* and a set of typed *questions* (`choice`, `noul`, `score`) and returns a probability for
every option in a single encoder pass. It does not generate text. It is built for the classify, route, gate and verify
decisions that agents and pipelines make thousands of times, and is the top of the Crivo family, above
[Crivo-400M](https://huggingface.co/tardellirs/crivo-400m). It was trained with the data and the three-stage recipe of
Crivo-400M, without a teacher.

| | |
|---|---|
| Architecture | Ettin / ModernBERT encoder, 28 layers, hidden 1792, `[OPT]` readout + MLP scorer |
| Parameters | 1.03B (1,031,276,289) |
| Base model | [`cross-encoder/ettin-reranker-1b-v1`](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1) |
| Teacher | none (gold labels only) |
| Decision Index 0.3 | public index **47.77** (self-run, 140,178 / 140,178 requests answered); Full score pending (maintainer-run) |
| Latency, 1× H100 PCIe 80 GB, bf16 (one request at a time) | median 13.2 ms · mean 36.4 ms · p95 94.1 ms per request |
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

Crivo-1B é um encoder de decisões tipadas com 1,03 bilhão de parâmetros (Ettin/ModernBERT), o maior da família Crivo.
Ele lê um *state* e perguntas `choice`, `noul` ou `score` e devolve uma probabilidade para cada opção numa única
passada, sem gerar texto. Foi treinado com os dados e a receita em três estágios do Crivo-400M, sem professor. No
índice público do Decision Index 0.3 marca **47,77** (run completo publicado, sem truncamento), contra 39,79 do
Crivo-400M, com mediana de 13,2 ms por requisição numa H100. O Full score, com os testes privados dos mantenedores,
ainda não saiu. Foi treinado só com dados abertos, com verificação de contaminação contra a suíte. Funciona em inglês.
Licença CC BY-NC 4.0 (uso não comercial).

</details>

## Contents

- [Results](#results)
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
[`tardellirs/decision-index-results` → `runs/crivo-1b`](https://huggingface.co/datasets/tardellirs/decision-index-results/tree/main/runs/crivo-1b).

The board's main ranking (Full score) is 20% this public index, 50% private tests of the same skills and 30% private
tasks from new domains, run by the maintainers. **The Full score will be added here once it is published.**

| Model | Public index | Raw | Knowledge & Reasoning | Language | Retrieval & Classification | Tools & Automation | Arts & Human Taste |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Crivo-1B** | **47.77** | 61.29 | 18.5 | 56.3 | 55.8 | 78.7 | 28.9 |
| Crivo-400M | 39.79 | 55.55 | 11.5 | 44.3 | 51.0 | 69.0 | 25.4 |

<details>
<summary><b>Per-benchmark skill (0–100, chance-corrected) — click to expand</b></summary>

★ = gold benchmark (weight 1.2 within its area). The index uses the native metric listed, chance-corrected.

| Benchmark | Native metric | Crivo-1B | Crivo-400M |
|---|---|---:|---:|
| **Knowledge & Reasoning** | | | |
| GPQA Diamond ★ | accuracy | 1.4 | 0.0 |
| GSM8K | accuracy | 3.4 | 2.3 |
| ChessBench | accuracy | 0.0 | 0.1 |
| MuSR | accuracy | 40.8 | 28.1 |
| SATA-Bench | case exact accuracy | 15.5 | 1.4 |
| CRUXEval | accuracy | 0.0 | 2.9 |
| CLadder | accuracy | 64.9 | 47.3 |
| HLE ★ | accuracy | 0.0 | 0.0 |
| MMLU-Pro ★ | accuracy | 10.8 | 6.6 |
| BBH fixed-option tasks ★ | accuracy | 17.5 | 13.3 |
| WinoGrande ★ | accuracy | 51.1 | 26.4 |
| **Language Understanding** | | | |
| ContractNLI | macro-F1 | 76.7 | 74.4 |
| ANLI ★ | macro-F1 | 33.4 | 15.9 |
| HellaSwag ★ | accuracy | 79.3 | 63.6 |
| ACOS | per-review F1 | 32.9 | 29.9 |
| FinEntity | macro-F1 | 88.2 | 81.0 |
| iSarcasmEval | sarcasm F1, track A English | 42.3 | 17.2 |
| VAST | macro-F1 | 67.1 | 63.0 |
| NLI4CT | macro-F1 | 36.1 | 25.9 |
| RAGTruth | F1 on hallucinated class | 50.3 | 29.6 |
| **Retrieval & Classification** | | | |
| BANKING77 ★ | macro-F1 | 91.2 | 85.8 |
| CLINC150+OOS ★ | macro-F1 | 93.0 | 88.6 |
| BRIGHT ★ | nDCG@10 | 28.2 | 27.1 |
| Amazon ESCI | macro-F1 | 36.6 | 25.9 |
| PhishNChips | accuracy (verdict) | 3.7 | 5.2 |
| HoVer | accuracy | 73.3 | 63.9 |
| **Tools & Automation** | | | |
| BFCL ★ | case exact accuracy | 90.6 | 82.6 |
| ToolRet | nDCG@10 | 58.5 | 58.7 |
| API-Bank ★ | accuracy | 80.3 | 68.7 |
| Home appliance simulator | case exact accuracy | 87.5 | 71.6 |
| When2Call | accuracy | 74.1 | 60.9 |
| **Arts & Human Taste** | | | |
| BPoMP | accuracy | 89.3 | 82.6 |
| Humicroedit | accuracy | 8.6 | 3.3 |
| POP909-CL | accuracy | 0.6 | 1.7 |
| cfcolor | accuracy | 1.5 | 1.6 |
| Habermas Machine | accuracy | 13.8 | 10.1 |
| New Yorker caption matching | accuracy | 59.5 | 52.9 |

</details>

- **Strong:** tool selection and API calls (BFCL 90.6, API-Bank 80.3, smart-home control 87.5, When2Call 74.1),
  intent classification (BANKING77 91.2, CLINC150 93.0), entity sentiment (FinEntity 88.2), commonsense completion
  (HellaSwag 79.3), long-document NLI and claim verification (ContractNLI 76.7, HoVer 73.3), causal inference
  (CLadder 64.9) and original-vs-perturbed text (BPoMP 89.3).
- **Weak:** expert knowledge and multi-step reasoning (GPQA, HLE, MMLU-Pro, GSM8K, chess, code execution), phishing
  verdicts (PhishNChips 3.7), and music, color and humor "taste" benchmarks.
- **Against Crivo-400M:** higher in every area, with the largest gains on iSarcasmEval (+25.1), WinoGrande (+24.7),
  RAGTruth (+20.7), CLadder (+17.6), ANLI (+17.5), smart-home control (+15.9) and HellaSwag (+15.7). It is level or
  slightly behind on ToolRet, PhishNChips, CRUXEval, POP909-CL, cfcolor and ChessBench.

A comparison with other models of similar size will be added together with the Full score, since the public index alone does not rank models by Full score.

### Internal selection panels (our own evaluation, not board scores)

Checkpoints were chosen on our own panels, never on the board and never on the complete public suite (it was run only
on the final package). These are **internal proxies**, not substitutes for the board's private tests, and no Full score
is estimated here. All three models below were run on the same panels with the same engine.

| Panel | What it measures | Crivo-150M v1 | Crivo-400M | **Crivo-1B** |
|---|---|---:|---:|---:|
| ND-proxy | mean chance-corrected skill on 21 held-out decision families from outside the suite (written rules and policies with exceptions, schedules, dates, propositional logic, code outputs and tests, arithmetic in context, rewritten items, product-domain decisions, and others) | 0.150 | 0.311 | **0.405** |
| SS-core | mean chance-corrected skill on our own development sets for 8 of the board's skills (BFCL, BANKING77, CLINC150, When2Call, NLI4CT, ANLI, Amazon ESCI, smart-home control), not built from suite rows | 0.459 | 0.686 | **0.754** |
| Akita v1 (originals) | accuracy on the 130 original items of the Akita benchmark (evaluation only, never trained on) | 43.8% | 59.2% | **71.5%** |

One caveat on the ND-proxy: the training data includes code-generated decision families of related kinds (policies
with exceptions, schedules, dates, logic, executed code). They were filtered against the panel rows, but the panel
partly measures transfer from them.

## Quick start

### Standalone (no kit)

```python
from huggingface_hub import snapshot_download
import sys; path = snapshot_download("tardellirs/crivo-1b"); sys.path.insert(0, path)
from crivo import Crivo

model = Crivo(path)                                 # CUDA if available, else CPU (slow at this size)
out = model.answer(
    state={"ticket": "My card was charged twice for the same order."},
    questions={
        "intent": {"type": "choice", "instructions": "What does the customer want?",
                   "criteria": {"refund": "Money back for the duplicate charge",
                                "cancel": "Cancel the order",
                                "info": "Just asking for information"}},
        "urgent": {"type": "noul", "instructions": "The customer needs a fast response."},
    })
print(out["intent"]["choice"], out["intent"]["probabilities"])   # the chosen key + a probability for every key
print(out["urgent"]["noul"])                                     # P(statement is true)
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
hf download tardellirs/crivo-1b --local-dir crivo-1b && cd crivo-1b
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-1b
```

The engine defaults are the settings of the reported run: one option order (`tta=1`), fp32 scoring head, SDPA
attention, `batch_size=32`, `token_budget=65536`, `max_length=32768`, bf16 autocast on CUDA. The state is tokenized
once per request and all questions are batched. It raises `Unsupported` only for a question above `max_length` tokens,
and there is none in the 0.3 suite. The weights are 4.1 GB in fp32; the reported run used one 80 GB GPU.

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
`model.py`, `crivo.py` (standalone API), `decision_index_engine.py` (kit engine, the same file as Crivo-400M's).

## Training

Full fine-tune of `cross-encoder/ettin-reranker-1b-v1` in three stages, on exactly the data of Crivo-400M. Every
example is one question with a target distribution over its options; loss = cross-entropy + 0.5 × Brier score. There
is no teacher: no larger Crivo model exists to distill from, so every stage trains on the gold labels.

### Data

| Stage | Questions | Content |
|---|---:|---|
| 1 | 638,306 | [Bekko System One dataset v0](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0) (≤1,200 per subset) · code-generated decision families with exact labels (policies with exceptions and precedence, opaque label rules, schedules with constraints, ordinal severity, dates, propositional logic, executed code and tests, link-based phishing, multi-select) · format-invariance augmentations (opaque keys, key-only options, distractors, "none of the above") · 328 task families from [`tasksource/tasksource-jev-typed-decisions`](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) · sibling sources of the suite's skills (argument quality, SEntFiN, RAGBench, Hermes function calling, xLAM irrelevance, bitext banking and e-commerce intents, CUAD contracts) · train splits of the public benchmarks' sources, in the original layout and rewritten · [`tasksource/procedural-typed-decisions`](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) · original-vs-perturbed public-domain text · open teacher labels from [`Mapika/decider`](https://github.com/Mapika/decider) (open Qwen models) · CFPB complaint routing · commit-message matching (CommitPackFT) |
| 2 | 169,437 | tool selection ([ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE), [glaive](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)), CLadder variants not used by the suite, SEntFiN entity sentiment, synthetic smart-home cases, 6,500 synthetic "which API next" dialogues with catalogs of ~50 APIs (gold calls from [Hermes function calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)), + replay of stage 1 |
| 3 | 24,635 | long context up to 8,192 tokens: ContractNLI, RAGTruth, NLI4CT and other long inputs, plus 5,000 "which API next" dialogues with long catalogs (~6k tokens) |

No data labeled by Jev or any other proprietary decision API was used.

### Recipe

The recipe of Crivo-400M, adapted to the larger model: one epoch per stage, a shorter weight EMA in stage 1 (0.999
instead of 0.9995), lower learning rates and no WiSE-FT interpolation (it never helped the smaller Crivo models).

| Stage | Init | Steps | LR (encoder / head) | Max length | Tokens per micro-batch × accumulation |
|---|---|---:|---|---:|---|
| 1 | Ettin reranker 1B | 12,384 (1 epoch, weight EMA 0.999) | 1e-5 / 1e-4 | 2,048 | 4,096 × 4 |
| 2 | stage-1 EMA | 5,235 (1 epoch) | 7e-6 / 7e-5 | 2,048 | 4,096 × 4 |
| 3 | stage 2 | 2,605 (1 epoch) | 5e-6 / 4e-5 | 8,192 | 8,192 × 4 |

AdamW (weight decay 0.01), linear warmup (5% stage 1, 3% after) and linear decay, bf16 autocast, option order shuffled
for every `choice` example.

**Checkpoint selection.** Three candidates were evaluated: the end of stage 2, the end of stage 3 and their weight
average. The choice was made on our own panels (held-out new-domain decision families, a development set of the board's
skills, the Akita benchmark and a public-suite sample), never on the board, and a candidate also had to pass
protection gates relative to Crivo-150M v1: no protected development-set skill more than 0.02 below it, Akita above
43.8%, and the public-sample estimate no more than 2.5 points below its public index. The stage-2/3 average scored
marginally higher on the aggregate of our panels but failed the protection gate on RAGTruth (−0.027 against Crivo-150M
v1 on our development set; the end of stage 2 failed it too, at −0.093). The end of stage 3 passed every gate and is the
released model. The complete public suite was run only on it. Two runs of 500 suite questions with `batch_size` 32 and
4 gave the same answer on every question (max probability difference 0).

**Compute.** Training, the selection panels and the complete public suite ran in one session of about 7.7 hours on one
NVIDIA H100 PCIe 80 GB (Lambda Cloud), about US$26.

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
suite. Benchmarks whose sources have no clean train split were used for evaluation only.

## Intended use and limitations

### Intended use

Fast, local, probabilistic decisions over text or JSON state, as a first pass or a component inside an agent or
pipeline:

- routing, intent and topic classification;
- policy and guardrail gates;
- tool and API selection;
- relevance and hallucination checks, NLI-style verification.

It is the most accurate Crivo model on our panels; where latency or memory matter more, Crivo-400M or a smaller
sibling may be the better trade-off.

### Out of scope

- Open-ended generation, extraction or any task that needs free-text output.
- Expert knowledge questions, math and multi-step reasoning.
- High-stakes decisions about people without human review.

### Limitations

- **Private parts not yet scored.** The public index is 20% of the board's Full score. The earlier Crivo models scored
  much lower on the private tests than on the public index, especially on tasks from new domains (Crivo-150M v1: 9.77
  there). Crivo-1B is better on our internal proxy for them, but the board's number is not known yet. Validate on your
  own data before relying on it for an unfamiliar task.
- **English only.** Training data is almost entirely English; other languages are untested and likely weak.
- **Little world knowledge, no step-by-step reasoning.** Expect near-chance results on expert QA (GPQA, MMLU-Pro, HLE),
  math word problems, chess and code execution.
- **Size.** 4.1 GB of fp32 weights; it needs a GPU for interactive use, and its tail latency (p95 94 ms on an H100) is
  higher than the smaller Crivo models'.
- **Format sensitivity.** Very different phrasings of a task may lose accuracy. Descriptive option text
  (`key: description`) works better than bare keys.
- **The smart-home score is optimistic.** The synthetic training cases reuse the suite's request templates (new houses
  only).
- **Long inputs beyond 8,192 tokens** are read without truncation but were never seen in training; quality there is
  uncertain.
- **No temperature fitted.** Treat probabilities as scores to threshold on your own data.

## License

**CC BY-NC 4.0**, the same as Crivo-400M: several training sources are licensed for non-commercial use only (for
example ANLI, CC BY-NC 4.0, and some Bekko dataset subsets), so the weights are released for non-commercial use.
Upstream dataset licenses and terms continue to apply to the data. The base model
([Ettin reranker 1B](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1)) is Apache-2.0.

## See also

- The Crivo family, by size: [Crivo-17M](https://huggingface.co/tardellirs/crivo-17m) ·
  [Crivo-32M](https://huggingface.co/tardellirs/crivo-32m) · [Crivo-68M](https://huggingface.co/tardellirs/crivo-68m) ·
  [Crivo-150M](https://huggingface.co/tardellirs/crivo-150m) ·
  [Crivo-400M](https://huggingface.co/tardellirs/crivo-400m) (public 39.79) · **Crivo-1B**.
- [Crivo collection](https://huggingface.co/collections/tardellirs/crivo-6acaa1731d71f839970f04a6).

## Acknowledgements

- [Ettin](https://huggingface.co/jhu-clsp/ettin-encoder-1b) (JHU CLSP) and the
  [Ettin reranker 1B](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1) used as the starting point.
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
@misc{crivo1b2026,
  title        = {Crivo-1B: a typed-decision encoder},
  author       = {Tardelli Stekel},
  year         = {2026},
  howpublished = {Hugging Face model card},
  url          = {https://huggingface.co/tardellirs/crivo-1b}
}
```
