---
license: cc-by-nc-4.0
language:
- en
base_model: cross-encoder/ettin-reranker-400m-v1
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
---

# Crivo-400M

**A typed-decision encoder for agents and pipelines, the largest of the Crivo models.**

[![Decision Index public](https://img.shields.io/badge/Decision%20Index%200.3-public%2039.79-2a78d6)](#decision-index-03--public-index-self-run-complete)
[![Full score](https://img.shields.io/badge/Full%20score-pending-999)](#decision-index-03--public-index-self-run-complete)
[![Parameters](https://img.shields.io/badge/parameters-396M-555)](#model-details)
[![Language](https://img.shields.io/badge/language-English-555)](#intended-use-and-limitations)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-555)](#license)

Crivo-400M reads a *state* and a set of typed *questions* (`choice`, `noul`, `score`) and returns a probability for
every option in a single encoder pass. It does not generate text. It is built for the classify, route, gate and verify
decisions that agents and pipelines make thousands of times, and is the larger sibling of
[Crivo-150M](https://huggingface.co/tardellirs/crivo-150m) and [Crivo-32M](https://huggingface.co/tardellirs/crivo-32m).

| | |
|---|---|
| Architecture | Ettin / ModernBERT encoder, 28 layers, hidden 1024, `[OPT]` readout + MLP scorer |
| Parameters | 396M |
| Base model | [`cross-encoder/ettin-reranker-400m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1) |
| Decision Index 0.3 | public index **39.79** (self-run, 140,178 / 140,178 requests answered); Full score pending (maintainer-run) |
| Latency, 1× A100 40 GB, bf16 (one request at a time) | median 23 ms · mean 38 ms · p95 74 ms per request |
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

Crivo-400M é um encoder de decisões tipadas com 396M de parâmetros (Ettin/ModernBERT), o maior da família Crivo. Ele lê
um *state* e perguntas `choice`, `noul` ou `score` e devolve uma probabilidade para cada opção numa única passada, sem
gerar texto. No índice público do Decision Index 0.3 marca **39,79** (run completo publicado, sem truncamento), contra
36,39 do Crivo-150M; o Full score, com os testes privados dos mantenedores, ainda não saiu. Foi treinado só com dados
abertos, com verificação de contaminação contra a suíte. Funciona em inglês. Licença CC BY-NC 4.0 (uso não comercial).

</details>

## Results

### Decision Index 0.3 — public index (self-run, complete)

Run with the official kit ([apolinario/decision-index](https://github.com/apolinario/decision-index), commit
`9eb2dbe`, edition 0.3) on the complete public suite, with the engine shipped in this repository: every one of the
140,178 scoreable requests answered, none unsupported, nothing truncated. Scores are chance-corrected skill (0 = random
guessing, 100 = perfect), weighted as the board does. Full results:
[`tardellirs/decision-index-results` → `runs/crivo-400m`](https://huggingface.co/datasets/tardellirs/decision-index-results/tree/main/runs/crivo-400m).

The board's main ranking (Full score) is 20% this public index, 50% private tests of the same skills and 30% private
tasks from new domains, run by the maintainers. **The Full score will be added here once it is published.**

| Model | Public index | Knowledge & Reasoning | Language | Retrieval & Classification | Tools & Automation | Arts & Human Taste |
|---|---:|---:|---:|---:|---:|---:|
| **Crivo-400M** | **39.79** | 11.5 | 44.3 | 51.0 | 69.0 | 25.4 |
| Crivo-150M | 36.39 | 11.8 | 41.8 | 52.2 | 60.7 | 10.1 |

<details>
<summary><b>Per-benchmark skill (0–100, chance-corrected) — click to expand</b></summary>

★ = gold benchmark (weight 1.2 within its area). The index uses the native metric listed, chance-corrected.

| Benchmark | Native metric | Crivo-400M | Crivo-150M |
|---|---|---:|---:|
| **Knowledge & Reasoning** | | | |
| GPQA Diamond ★ | accuracy | 0.0 | 0.0 |
| GSM8K | accuracy | 2.3 | 2.0 |
| ChessBench | accuracy | 0.1 | 0.0 |
| MuSR | accuracy | 28.1 | 26.7 |
| SATA-Bench | case exact accuracy | 1.4 | 0.0 |
| CRUXEval | accuracy | 2.9 | 7.3 |
| CLadder | accuracy | 47.3 | 61.0 |
| HLE ★ | accuracy | 0.0 | 0.0 |
| MMLU-Pro ★ | accuracy | 6.6 | 3.4 |
| BBH fixed-option tasks ★ | accuracy | 13.3 | 11.6 |
| WinoGrande ★ | accuracy | 26.4 | 21.9 |
| **Language Understanding** | | | |
| ContractNLI | macro-F1 | 74.4 | 74.7 |
| ANLI ★ | macro-F1 | 15.9 | 19.0 |
| HellaSwag ★ | accuracy | 63.6 | 48.9 |
| ACOS | per-review F1 | 29.9 | 38.9 |
| FinEntity | macro-F1 | 81.0 | 76.7 |
| iSarcasmEval | sarcasm F1, track A English | 17.2 | 0.0 |
| VAST | macro-F1 | 63.0 | 61.4 |
| NLI4CT | macro-F1 | 25.9 | 15.1 |
| RAGTruth | F1 on hallucinated class | 29.6 | 44.3 |
| **Retrieval & Classification** | | | |
| BANKING77 ★ | macro-F1 | 85.8 | 90.6 |
| CLINC150+OOS ★ | macro-F1 | 88.6 | 91.8 |
| BRIGHT ★ | nDCG@10 | 27.1 | 18.6 |
| Amazon ESCI | macro-F1 | 25.9 | 29.8 |
| PhishNChips | accuracy (verdict) | 5.2 | 4.0 |
| HoVer | accuracy | 63.9 | 69.4 |
| **Tools & Automation** | | | |
| BFCL ★ | case exact accuracy | 82.6 | 79.0 |
| ToolRet | nDCG@10 | 58.7 | 55.7 |
| API-Bank ★ | accuracy | 68.7 | 29.4 |
| Home appliance simulator | case exact accuracy | 71.6 | 79.5 |
| When2Call | accuracy | 60.9 | 62.5 |
| **Arts & Human Taste** | | | |
| BPoMP | accuracy | 82.6 | 0.0 |
| Humicroedit | accuracy | 3.3 | 4.5 |
| POP909-CL | accuracy | 1.7 | 1.2 |
| cfcolor | accuracy | 1.6 | 0.6 |
| Habermas Machine | accuracy | 10.1 | 10.1 |
| New Yorker caption matching | accuracy | 52.9 | 44.1 |

</details>

- **Strong:** tool selection and API calls (BFCL 82.6, API-Bank 68.7, ToolRet 58.7), intent classification
  (BANKING77 85.8, CLINC150 88.6), entity sentiment and stance (FinEntity 81.0, VAST 63.0), long-document NLI
  (ContractNLI 74.4, HoVer 63.9), commonsense completion (HellaSwag 63.6) and original-vs-perturbed text (BPoMP 82.6).
- **Weak:** expert knowledge and multi-step reasoning (GPQA, HLE, MMLU-Pro, GSM8K, chess, code execution) and
  subjective "taste" benchmarks. Against Crivo-150M it is behind on RAGTruth, CLadder and ACOS.

## Quick start

### Standalone (no kit)

```python
from huggingface_hub import snapshot_download
import sys; path = snapshot_download("tardellirs/crivo-400m"); sys.path.insert(0, path)
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
print(out["intent"]["choice"], out["intent"]["probabilities"])   # refund {'refund': 0.96, 'cancel': 0.03, 'info': 0.01}
print(out["urgent"]["noul"])                                     # 0.12
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
hf download tardellirs/crivo-400m --local-dir crivo-400m && cd crivo-400m
python -m decision_index pipeline --engine decision_index_engine:DecisionEngine --edition 0.3 --out runs/crivo-400m
```

The engine defaults are the settings of the reported run: one option order (`tta=1`), fp32 scoring head, SDPA
attention, `batch_size=32`, `token_budget=65536`, `max_length=32768`, bf16 autocast on CUDA. The state is tokenized
once per request and all questions are batched. It raises `Unsupported` only for a question above `max_length` tokens,
and there is none in the 0.3 suite.

## Model details

Same design as Crivo-150M. Each question is one sequence

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

Full fine-tune of `cross-encoder/ettin-reranker-400m-v1` in three stages. Every example is one question with a target
distribution over its options; loss = cross-entropy + 0.5 × Brier score.

### Data

| Stage | Questions | Content |
|---|---:|---|
| 1 | 638,306 | [Bekko System One dataset v0](https://huggingface.co/datasets/hotchpotch/bekko-system-one-dataset-v0) (≤1,200 per subset) · code-generated decision families with exact labels (policies with exceptions and precedence, opaque label rules, schedules with constraints, ordinal severity, dates, propositional logic, executed code and tests, link-based phishing, multi-select) · format-invariance augmentations (opaque keys, key-only options, distractors, "none of the above") · 328 task families from [`tasksource/tasksource-jev-typed-decisions`](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) · sibling sources of the suite's skills (argument quality, SEntFiN, RAGBench, Hermes function calling, xLAM irrelevance, bitext banking and e-commerce intents, CUAD contracts) · train splits of the public benchmarks' sources, in the original layout and rewritten · [`tasksource/procedural-typed-decisions`](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) · original-vs-perturbed public-domain text · open teacher labels from [`Mapika/decider`](https://github.com/Mapika/decider) (open Qwen models) · CFPB complaint routing · commit-message matching (CommitPackFT) |
| 2 | 169,437 | tool selection ([ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE), [glaive](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)), CLadder variants not used by the suite, SEntFiN entity sentiment, synthetic smart-home cases, 6,500 synthetic "which API next" dialogues with catalogs of ~50 APIs (gold calls from [Hermes function calling](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)), + replay of stage 1 |
| 3 | 24,635 | long context up to 8,192 tokens: ContractNLI, RAGTruth, NLI4CT and other long inputs, plus 5,000 "which API next" dialogues with long catalogs (~6k tokens) |

No data labeled by Jev or any other proprietary decision API was used.

### Recipe

| Stage | Init | Steps | LR (encoder / head) | Max length | Tokens per step |
|---|---|---:|---|---:|---:|
| 1 | Ettin reranker 400M | 11,919 (1 epoch, weight EMA 0.9995) | 1.5e-5 / 1.5e-4 | 2,048 | 16,384 |
| 2 | stage-1 EMA | 4,818 | 1e-5 / 1e-4 | 2,048 | 16,384 |
| 3 | stage 2 | 5,210 | 8e-6 / 5e-5 | 8,192 | 16,384 |

AdamW, linear warmup (5% stage 1, 3% after) and linear decay, bf16 autocast, gradient accumulation 2, option order
shuffled for every `choice` example. The released weights are the end of stage 3. Stage 2, a stage-2/3 weight average
and interpolations toward the base model (WiSE-FT) were also evaluated; all were selected against our own panels
(held-out new-domain decision families, a development set of the board's skills and a public-suite sample), never against
the board, and stage 3 was best.

**Compute.** Training on one NVIDIA H100 SXM5 80 GB (Lambda Cloud) and the complete public suite on one A100 40 GB:
about US$28 in total.

## Contamination firewall

The index counts any test item found in an entrant's training data as wrong, so contamination was checked at several
levels:

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

**Intended use:** fast, local, probabilistic decisions over text or JSON state, as a first pass or a component inside an
agent or pipeline: routing, intent and topic classification, policy gates, tool and API selection, relevance and
hallucination checks, NLI-style verification.

- **Private parts not yet scored.** The public index is 20% of the board's Full score; how the model does on the
  private tests, especially tasks from new domains (the weak spot of Crivo-150M), is not known yet. Validate on your
  own data before relying on it for an unfamiliar task.
- **English only.**
- **Little world knowledge, no step-by-step reasoning**: near chance on expert QA, math word problems, chess and code
  execution.
- **The smart-home score is optimistic**: the synthetic training cases reuse the suite's request templates (new houses
  only).
- **Long inputs beyond 8,192 tokens** are read without truncation but were never seen in training.
- **No temperature fitted**: treat probabilities as scores to threshold on your own data.
- Not for high-stakes decisions about people without human review.

## License

**CC BY-NC 4.0.** Several training sources are licensed for non-commercial use only (for example ANLI, CC BY-NC 4.0, and
some Bekko dataset subsets), so the weights are released for non-commercial use. Upstream dataset licenses and terms
continue to apply to the data. The base model ([Ettin reranker 400M](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1))
is Apache-2.0.

## See also

- [Crivo-150M](https://huggingface.co/tardellirs/crivo-150m) (Full 22.21) and [Crivo-32M](https://huggingface.co/tardellirs/crivo-32m)
  (Full 11.87), both on the Pareto frontier of the Decision Index.
- [Crivo collection](https://huggingface.co/collections/tardellirs/crivo-6acaa1731d71f839970f04a6).

## Acknowledgements

- [Ettin](https://huggingface.co/jhu-clsp/ettin-encoder-400m) (JHU CLSP) and the
  [Ettin reranker 400M](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1) used as the starting point.
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
@misc{crivo400m2026,
  title        = {Crivo-400M: a typed-decision encoder},
  author       = {Tardelli Stekel},
  year         = {2026},
  howpublished = {Hugging Face model card},
  url          = {https://huggingface.co/tardellirs/crivo-400m}
}
```
