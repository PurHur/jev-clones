---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
language:
- en
pipeline_tag: text-classification
tags:
- decision
- calibration
- reasoning
- speculative-decoding
- qwen3.5
---

# Jeeves-9B

A Jev-like decision model that reasons before it decides when it is unsure, trained on public data only. Give it a state (text or JSON) and yes/no, multiple-choice or rating questions; it answers directly when it is confident, otherwise writes a short reasoning chain first, and returns a calibrated probability for every option.

This is Jeeves v2. v1 (SFT + CISPO) is the `v1` tag of this repository: `hf download PostHog/jeeves --revision v1`.

This repository holds the fused weights (Qwen3.5-9B with the LoRA merged), the pointer head, the two fitted temperatures, and two speculative-decoding drafters. It loads with the Jeeves code, not with `transformers`: the pointer head and the prompt format are part of the model.

## Files

| file | contents |
|---|---|
| `model-*.safetensors`, `config.json` | fused Qwen3.5-9B weights, bf16 |
| `head.pt` | pointer head (query and key projections, 256-dim) |
| `export.json` | temperatures (1.522 direct, 5.420 after reasoning), prompt format, training step and config |
| `tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`, `chat_template.jinja` | Qwen3.5 tokenizer |
| `drafter_k4.safetensors` | block-4 drafter, the serving default, bf16 (trained on v1; v2 drafters to follow) |
| `drafter_k8.safetensors` | block-8 drafter, faster for single-question requests, bf16 |
| `drafter_k4.config.json`, `drafter_k8.config.json` | drafter training settings |
| `LICENSE` | Apache-2.0, inherited from Qwen3.5-9B |

## Usage

```bash
hf download PostHog/jeeves --local-dir jeeves-weights
git clone https://github.com/PostHog/jeeves && cd jeeves
python -m inference.serve --model ../jeeves-weights --drafter ../jeeves-weights/drafter_k4.safetensors --port 8009
```

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice for one order.",
  "questions": {
    "billing": {"type": "noul", "instructions": "Is this about billing?"},
    "tone": {"type": "choice", "instructions": "What is the customer'"'"'s tone?", "criteria": {"calm": null, "frustrated": null, "angry": null}}
  }}'
```

The request and response follow Jev's `/v1/systemone` format. An optional `options` object sets `think`, `max_think` (truncate chains), `nothink_threshold` (skip thinking when the no-think answer is already confident) and `return_reasoning`. Serving runs on a CUDA GPU or on an Apple Silicon Mac with 48 GB (add `--max-rows 4 --max-len 4096` there). It serves bf16 by default; `--precision fp8` runs the linear layers in FP8 on GPUs with compute capability 8.9 or higher and on Macs. FP8 weights can be downloaded from [PostHog/jeeves-fp8](https://huggingface.co/PostHog/jeeves-fp8).

## Results

Jeeves answers directly when its calibrated confidence is at least 0.9 and otherwise reasons first (`nothink_threshold` 0.9, greedy, 2,560-token cap); the percentage is the share of questions that reasoned. Kev-9B and Jev numbers are the ones Kev publishes. JevK5 (v0.3) does not reason; we ran it ourselves on the same items. JevBench uses the same 231 public items for every model (the sealed judge tier is not included); the other rows use the same sources with different items.

| | Kev-9B | Jev | JevK5 | Jeeves v1 | Jeeves v2 |
|---|---|---|---|---|---|
| Test overall (out-of-domain and held-out) | 0.822 | 0.857 | 0.802 | 0.884 (41%) | **0.905** (39%) |
| MMLU-Pro and buried state | 0.579 | **0.800** | 0.561 | 0.728 (86%) | 0.780 (76%) |
| JevBench, public tiers | 0.715* | 0.866 | 0.861 | **0.913** (44%) | 0.896 (29%) |
| JevBench hard | 0.451* | 0.730 | 0.748 | **0.820** | 0.793 |

\* Kev-8B (Qwen3); no Kev-9B JevBench result is published.

Without thinking v2 scores 0.812 on the Jeeves test split, against 0.851 reasoning when unsure. On held-out questions from the 15 public datasets new in v2 it scores 0.853 and 0.900.

Serving on one H100 with `--precision fp8`, 325 dev questions:

| setting | accuracy | median / p90 latency |
|---|---|---|
| full thinking | 0.818 | 1.9 s / 8.1 s |
| `nothink_threshold` 0.9 | 0.822 | 1.2 s / 8.2 s |
| `max_think` 1024, `nothink_threshold` 0.9 | 0.818 | 1.2 s / 4.3 s |
| no thinking | 0.806 | 22 ms / 46 ms |

v2 was trained on chains of at most 1,024 tokens, so lower caps cut most of its chains.

## Training

- **SFT.** LoRA r=16 on every projection plus the pointer head, 1 epoch (16,867 steps) on 453,531 questions from about 175 public sources. About 15% of examples carry a reasoning chain sampled from earlier rounds of the model, kept only if it closes within 1,024 tokens, does not loop and reaches the right answer.
- **Calibration.** Two softmax temperatures fitted on the dev set: 1.522 for direct answers and 5.420 for answers after reasoning. Every training chain reaches the right answer, so answers after reasoning come out overconfident and need the larger temperature.
- **Drafters.** Diffusion views of the frozen model, inspired by Orthrus, distilled by KL divergence on chains sampled from v1. They work with v2 (every draft is verified, so outputs are unchanged) but accept fewer tokens; v2 drafters will follow. At one question with v1 the block-4 drafter decodes 1.6× faster than graphed greedy decoding and the block-8 drafter 1.76×.

v1 was trained with SFT followed by CISPO; CISPO did not improve v2. The training code, dataset build script (`python -m prep.prep`, which rebuilds v2's training data byte for byte from pinned revisions) and the id lists of the mix are in the GitHub repository. The reasoning chains are not published yet.

## Training data

Public sources only; no PostHog or customer data. Every new record was checked against the eval sets and dev for shared 8-word sequences. Each source stays under its own licence; the main groups are:

- **Public datasets.** AG News, Amazon reviews, ARC (CC BY-SA 4.0), Banking77, Bias in Bios, BoolQ (CC BY-SA 3.0), Civil Comments, CLINC150, CommonsenseQA, DBpedia (CC BY-SA 3.0), GoEmotions, IMDB, LIAR2, MASSIVE, MedMCQA, MedQA, Mind2Web, MNLI, Measuring Hate Speech, ProsocialDialog, QASC, SNLI (CC BY-SA 4.0), SST-5, TREC, WANLI, Yelp, HelpSteer2 and HelpSteer3, Anthropic HH-RLHF.
- **New in v2.** GSM8K, AQuA-RAT, RuleTaker, WinoGrande, HellaSwag, SWAG, Cosmos QA, OpenBookQA, the New Yorker caption contest, When2Call, ToolACE, Glaive function calling v2, RAGTruth, ESCI and Schema-Guided Dialogue (CC BY-SA 4.0).
- **Third-party decision sets.** Open-Jev v1 and v1.1 (CC0), the Kev suites, spark, typed-decisions, SemIf and decider's converted sets and teacher data. Their labels, and for several of them their texts, come from LLM teachers.
- **Prompt injection.** Gandalf, LLMail-Inject, the neuralchemy and S-Labs prompt-injection sets, InjecAgent and permissively licensed parts of ShieldLM's curation.
- **Generated.** Some sources are LLM-generated (for example Glaive, ToolACE, When2Call, Open-Jev and the teacher-labelled sets), and v1's synthetic policy data is generated from templates.

The full list of repositories, revisions and licences is in `data/manifest.json` in the GitHub repository.

## Limitations

- Knowledge questions trail Jev (MMLU 0.845 vs 0.900, MMLU-Pro 0.770 vs 0.840).
- v2 trails v1 on JevBench when reasoning only when unsure (0.896 vs 0.913): it is overconfident without reasoning on some long-policy and multi-hop items, so it skips reasoning on them. With reasoning on every item the two are level (0.922 vs 0.926).
- Thinking is slow at the tail (8 s at p90 with full chains); use `max_think` and `nothink_threshold` when latency matters.
- Evaluated on English inputs only.

## License

The weights are derived from Qwen3.5-9B and are released under its Apache-2.0 license (`LICENSE`). The Jeeves code is MIT licensed.

## Citation

```bibtex
@software{waltz2026jeeves,
  author = {Waltz, Nicholas P.},
  title  = {Jeeves: Reasoning Improves Jev-like Decisions},
  year   = {2026},
  url    = {https://github.com/PostHog/jeeves},
  note   = {Qwen3.5-9B decision model that reasons when unsure, with a block-4 diffusion drafter}
}
```
