# Jev clones: architecture comparison

As of 2026-10-19. This summarises the profiles in `clones/`; every number and claim there carries its source.
**Scale:** about 1,320 HF repos and 2,200 GitHub repos appeared after Jev's launch (2026-09-15), grouped into 197
model families (`notes/discovery.md`). `notes/arch/arch_table.tsv` holds the config fields of 479 HF repos from all
families, extracted by `scripts/hf_arch.py`.

## 1. What every clone has in common

Every clone follows the same contract from TypeSafe's System One: a state plus typed questions (`noul` / `choice` /
`score`) go in, a probability distribution per question comes out, and **nothing is decoded**. Clones that generate
reasoning first (PostHog jeeves when unsure, StartLux AutoThink) are the exception. The clones differ along six
axes:

1. **Backbone**, and whether attention is causal, partly noncausal, or a bidirectional encoder
2. **How several questions share one state**: question isolation, which is Jev's documented guarantee
3. **Readout**, meaning how a hidden state becomes option probabilities
4. **Option interaction**: whether options see each other, and how sensitive the answer is to option order
5. **Calibration**
6. **Training signal**: human or programmatic labels, an LLM teacher, Jev itself, or RL with proper scoring rules

What is known about Jev itself (`clones/jev-reference.md`):
- **Official:** the state is read once, questions are isolated, a request holds up to 64k tokens, and a choice has 2–255 options.
- **Official:** post-trained from a pretrained LM with RLCD on synthetic data.
- **Official:** confidence is a fixed formula applied to the probabilities.
- **Reconstructed, not official:** a causal decoder with a cached prefix, one branch per question and a pointer head. Its tokenizer is closest to Qwen's.

## 2. Backbones

| Backbone | Share of the 479 configs | Who |
|---|---|---|
| Qwen3.5 hybrid (Gated DeltaNet + full attention, 3:1) 0.8B–9B, 35B-A3B | ~140 | Kev 1.0, pplx flash, d3, Strands-2B, H2O, Tev1, Intern, Mapika, LemonDecider, lev, jeeves |
| Qwen3.8-27B (48 DeltaNet + 16 full attention) | most of the top 30 on the board | pplx-decider, d3, StartLux, Kev-27B, Clef, AutoTrust, Open-Jev (Zefan) |
| Gemma 4 (E2B…31B, 26B-A4B MoE; sliding + full attention) | ~27 | Strands, decider-12b/31b, GEV, Xor, many board entries |
| Qwen3 / Qwen2.5 dense | ~26 | kev gen 1, **Zeiger (Qwen3-0.6B)**, NanoJev, Decision 2.0 Kai, CLM |
| ModernBERT / DeBERTa / mmBERT encoders | ~20 | **Laya**, Von, Vela, GLiNER2.5, Julia, Crivo, open-jev-deberta |
| Others | – | Granite 4.1 (IBM), LFM2.5 conv hybrid (Liquid), Ministral 3 (Standard One), Ouro looped (OpenLAIR lev), Muse-Glimmer (Meta), diffusion LM (Drex), Bielik (Polish basal) |

- **Big labs bring their own backbones; the community uses Qwen.** None of Google, Microsoft, NVIDIA, Qwen, Mistral or DeepSeek has released a Jev-style model on HF. Microsoft-Decision-1 and OpenAI Decisions are API-only.
- **Hybrid backbones changed the design.** Qwen3.5's recurrent DeltaNet layers cannot follow a block-causal mask. Kev 1.0 therefore gave up packed multi-question sequences in favour of a cached state prefix with one row per question (§3).

## 3. Multi-question handling and isolation

```
A  packed + block-causal mask        [state | q1 | q2 | q3]      q_i sees state + itself only
   kev gen 1, OpenLAIR lev, basal-1.5 (tree mask), Decision 2.0 (opt-in tree), Drex-DLM

B  cached state prefix, one row per question
   [state]──KV/recurrent cache──┬─ q1
                                ├─ q2           exact isolation, state encoded once
                                └─ q3
   Kev 1.0, Strands, Mapika decider, WebJev, jeeves, Drex, Liquid d1-3B (tree, conv needs only 2 taps),
   IBM aLoRA, Xor (SGLang prefix cache), lev (llama.cpp fork), FRIDA

C  one full sequence per question      [state q1]  [state q2]  [state q3]
   exact isolation, state re-encoded every time
   pplx v1/v1.1, d3 (noncausal ⇒ no prefix cache), Granite LoRA, H2O, Tev1, AutoTrust, Laya, Von,
   d1-omni-600M, Standard One v2.2, StartLux, HF transformers `jev` type, **Zeiger**

D  all questions jointly, isolation deliberately broken     [state q1 q2 q3] + joint head
   pplx v1.2 (no mask between questions), Cloudflare Clef (joint head across questions), Intern-Decision
   (JSON skeleton, causal), Vela, GLiNER2.5-Decide, Standard One SH, Valen, open-jev-deberta
```

- **Pattern B is the closest public match to Jev's documented behaviour.** It is also the cheapest when one state has many questions.
- **Pattern D buys accuracy on related questions at a cost on independent ones.** Perplexity reports up to +11 points when questions are related and up to −20 when they are independent (flash). Adding, removing or reordering one question changes the other answers.
- **Pattern C plus noncausal layers rules out caching.** pplx v1.1 and d3 make the full-attention layers bidirectional, so the state's tokens depend on the question and vLLM has to disable prefix caching.

## 4. Readout

| Readout | How | Who |
|---|---|---|
| **LM-head label tokens** | logits of single-token labels (A…Z, AA…, yes/no, digits) at the answer slot, renormalised over the valid labels | Granite, Intern-Decision, H2O, Tev1 (generated letter), Liquid d1-3B, Mapika decider, WebJev, Nimble, StartLux (A–Z plus group rounds), Xor, Standard One v2.2 |
| **Separate code matrix** | bias-free 255×d matrix started from the LM-head rows of the codes; LM head dropped | pplx-decider, vllm-sr d3, LemonDecider, AutoTrust (24 slots, shipped as an LM-head LoRA), HF `jev` head type |
| **Pointer head** | dot product of the projected decision-token state and each option's end-token state, then softmax | kev, Strands (24 slots), jeeves, Drex, OpenLAIR lev, Decision 2.0 (bilinear + MLP) |
| **Option markers + set module** | one marker hidden state per option, then a small transformer over the options, one logit each | **Zeiger** (2-layer set transformer + type embedding), Laya (`[MASK]` markers), ejhshen/OpenJev (no positions → order-free), lev mode B, Clef (joint over all questions), Standard One SH |
| **Scalar per candidate** | cross-encoder score per (question, option), softmax over the scores | Zefan-Cai/Open-Jev (init from the Yes−No logit), Von, NanoJev, Decima, Blink |
| **Bi-encoder** | cosine(state, option)/τ, candidates can be cached | Meta MetaEncoder-30B, CLM |
| **Exotic** | particle dynamics toward one well per option (adds 0.6–2.3 pp over step 0) | vegaml |

**How each design handles question types:**
- **Shared head:** almost every clone serves all three types with one head. noul becomes the options `[false, true]`, and score becomes ordered levels read as an expected value.
- **Separate type signal:** only a few clones give the type its own signal: Zeiger (type embedding), Strands and AutoTrust (per-type temperature), and the `jev` head type (per-type row offsets).

## 5. Option interaction and order

| | Options see each other | Order effect |
|---|---|---|
| Causal row (most decoders, kev, Strands, Zeiger backbone) | only earlier options | measurable (Strands ≈0.016 shift); Jev itself shifts ≈0.02 |
| Noncausal layers (pplx v1.1+, d3, H2O two-way) | all options, in both directions | lower, but positions remain |
| Encoder in one sequence (Laya, Julia, Crivo) | all options | position bias measured in Laya |
| Set module without positions (ejhshen/OpenJev, Standard One SH) | all options in the head | none by construction |
| Independent per candidate (Von v1.2, Decima, Blink, Open-Jev Zefan, bi-encoders) | none | none, but no comparison between options |
| Two-order averaging (Xor: original + reversed order; lev: two orders) | – | cancelled at 2× cost |

Zeiger sits in between. The backbone sees only earlier options, while the set transformer compares every option with
every other but still carries position information.

## 6. Calibration

| Scheme | Who |
|---|---|
| none / T = 1 | LemonDecider, d3 (raw softmax), Clef (reference code), ejhshen/OpenJev, NanoJev, Tev1, Granite, WebJev |
| one global temperature | Kev 1.0 (2.2–2.4 small, 1.32 for 27B), pplx (1.0–2.2), jeeves (1.52 direct / 5.42 after reasoning), Intern (1.99), Open-Jev Zefan |
| per question type | Strands (fitted to ECE), AutoTrust, H2O, Xor, StartLux, Vela |
| per option count | **Zeiger** (`choice:11+`), Laya (bug: T = 0.10 applied unclamped until 2026-09-21), d1-omni |
| depends on the input | Von (entropy, state length, option count), HF `jev` type (4-feature temperature) |
| certified thresholds | basal-1.5 (confidence thresholds certified at 1 % and 5 % error) |

**Ways the published ECE numbers can mislead:**
- **AutoTrust:** its ECE 0.0007 measures agreement with Jev; against ground truth it is 0.07–0.085.
- **GLiClass:** its shipped calibrator worsens calibration error on its own held-out report.
- **H2O:** the default `noul_floor` (0.801) never returns a yes/no probability between 0.2 and 0.8.
- **Mapika:** decider-12b's noul T = 0.05 is tuned for a benchmark.

## 7. Training signal

| Signal | Who |
|---|---|
| **Jev outputs distilled** (TypeSafe's agreement §2.3(b) forbids this) | AutoTrust JEV/GEV (~0.4M Jev-labelled rows via OpenRouter, soft targets), NanoJev (partly) |
| Jev used only as a yardstick | pplx-decider (a checkpoint is kept only if its dev ECE/Brier is within 0.01 of Jev 1.13's) |
| Another LLM as teacher | Strands (gemma-4-31B / Qwen3.5-4B), Mapika (local Qwen 27B), WebJev (DeepSeek, verified), d3 flash distilled from the 27B models |
| Human / public / programmatic labels | kev, pplx (annotator distributions, soft CE), Intern (hard labels), Granite (undisclosed) |
| RL / proper-scoring rewards (RLCD-like) | ejhshen/OpenJev (SFT → Brier/log-score RL), Clef ("RLCD", whole-record reward), Laya (RLCD term adds nothing over soft CE), StartLux AutoThink, jeeves v1 (CISPO) |
| Benchmark train splits in the mix (scores may be inflated) | StartLux (14 DI benchmarks), GEV / JEV-35B (12), GLiClass calibrator (inferred) |
| Edit pairs that flip the answer | Nimble |

**Training method:**
- **LoRA rank 16 dominates** among the adapters (63 of 86).
- **Full fine-tunes** are the norm among the top board entries.
- **Seed soups or merges** are common: Strands, Decision 2.0, LemonDecider, Kev-27B (0.85/0.15) and Zeiger.

## 8. Context and option limits

| | Context | Options |
|---|---|---|
| Jev | 64k per request, 32k per question (262k with the danger-zone header) | 2–255 |
| pplx v1.2 / LemonDecider | 262k / 256k (trained to ~32k) | 255 |
| Kev 1.0 | 64k state served; validated 8k small / 65k for 27B | 255 |
| d3, Clef | 64k hosted (Clef-flash 24k) | 255 |
| Strands | 4k | 24 |
| AutoTrust, Tev1 | – | 24 |
| Intern-Decision | 8k, 1–16 questions | 62 |
| Laya | 512 (192 for options) | ~48 |
| **Zeiger** | 131k head limit via 2k chunks (trained to 32k) | **40–1,500 tested, refused rather than truncated** |

**Zeiger handles large option sets better than any other clone surveyed.** Every other clone stops at 255 options.
WebJev, the only other browser-specialised model, caps a page at 250 elements and 6,000 characters.

## 9. Scores (Decision Index 0.3.2, community board, snapshot 2026-10-16)

| # | Score | Model | Architecture |
|---:|---:|---|---|
| 1 | 69.12 | StartLux-Decision v1.1 27B | LM-head letters; bidirectional "Instant" mode; non-commercial weights, trained on benchmark train splits |
| 2 | 66.26 | vllm-sr d3 27B | pattern C, noncausal, 255-code matrix |
| 4 | 65.55 | pplx-decider v1.2 | pattern D, noncausal, 255-code matrix |
| 5 | 64.62 | Mapika decider-31b | pattern B, LM-head labels |
| 13 | 61.53 | **Jev 1.14** | – |
| 18 | 60.11 | Jev 1.13 | – |
| 24 | 58.78 | Kev-27B | pattern B, pointer head |
| 37–44 | 55.6–57.0 | AutoTrust JEV-35B / GEV / JEV-27B | Jev distillation |
| 47 | 54.81 | pplx-flash-v1.2 (4B) | best 4B |
| 99 | 38.90 | Liquid d1-3B | pattern B, LM-head |
| – | 22.21 | Crivo-150M | best small encoder |
| – | 9.09 / 13.66 | d1-omni-600M / Kev-0.8B | sub-1B models sit far below |

**How to read the board:**
- **Self-reported numbers aren't comparable with board scores.** The public split and the full panel differ: Clef scores 61.7 on public but 53.1 on full.
- **Size matters more than architecture.** Every top-12 entry is a 26–31B model, and no sub-1B decoder clone is near the top.
- **Zeiger isn't on the board.** On Jev's general sample it scored 0.824 against Jev's 0.850.

## 10. What this means for Zeiger

1. **Keep:** chunked long context and the set module over options. No other clone handles whole web pages with
   more than 255 elements.
2. **Isolation pattern:** switching Zeiger to pattern B (shared state cache, one row per question) would let an agent
   ask several questions about one page without re-encoding the page each time. Qwen3-0.6B is pure attention, so a
   KV cache fork or kev gen-1's block-causal mask both work with no DeltaNet complications.
3. **Order bias:** removing positions from the set module (ejhshen/OpenJev) or averaging over two option orders
   (Xor) are cheap tests against order bias.
4. **Calibration:** per-type temperatures (Strands) or input-dependent temperatures (Von, HF `jev`) next to our
   per-option-count buckets; basal's certified thresholds fit the action gate in jevpilot.
5. **Browser task, from WebJev:** borrow jev-ultrafast's wire format (which opens their 125-task live benchmark), a
   separate operation/DONE question, richer element fields, and their hard-negative dataset.
6. **Small-model recipes:** Crivo-150M's distillation recipe and Decima's order-free, linear-cost option scoring
   for a CPU tier.
7. **Export:** the HF transformers `jev` model type (draft PR #49566) would need a new head type for our set
   module, and cannot express chunked pages yet.
8. **Licence hygiene:** stay off Jev-distilled data (AutoTrust's corpus) and benchmark train splits.

## Profiles

Reference: [Jev](clones/jev-reference.md) · Ours: [Zeiger](clones/zeiger.md)
Big labs: [Perplexity](clones/pplx-decider.md) · [vLLM-SR](clones/vllm-sr.md) · [Cloudflare](clones/cloudflare-clef.md) ·
[Liquid](clones/liquid-d1.md) · [Amazon Strands](clones/amazon-strands-decider.md) · [IBM](clones/ibm-granite-decision.md) ·
[InternLM](clones/intern-decision.md) · [H2O](clones/h2o-lightning.md) · [Together](clones/together-tev1.md) ·
[Meta](clones/meta-encoder.md) · [Juspay / Nace / StartLux](clones/misc-company-clones.md) · [Bespoke](clones/bespoke-nimble.md) ·
[PostHog](clones/posthog-jeeves.md) · [Standard Thinking](clones/standardone.md)
Community: [kev](clones/kev.md) · [Laya](clones/laya.md) · [vegaml](clones/vegaml.md) · [AutoTrust](clones/autotrust-jev.md) ·
[decider](clones/mapika-decider.md) · [WebJev](clones/webjev.md) · [von](clones/von.md) · [OpenJev](clones/openjev.md) ·
[NanoJev](clones/nanojev.md) · [lev](clones/lev.md) · [CLM](clones/clm.md)
Groups: [label-set encoders](clones/label-set-encoders.md) · [multimodal](clones/group-multimodal.md) ·
[small encoders](clones/group-small-encoders.md) · [languages](clones/group-language.md) · [HF `jev` type](clones/hf-jev-model-type.md)
