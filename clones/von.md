# Von (wfzyx)

Open Jev clone built on a **bidirectional encoder** (ModernBERT-large, 395M). It packs the question, state and every option into one sequence, puts a `[MASK]` marker in front of each option, scores each marker with a small MLP, and takes a softmax over the K logits. Since 1.2, options attend only to the prompt and to themselves, so the scores are order-invariant by construction. It serves TypeSafe's `/v1/systemone` contract on CPU. The GitHub repo was archived on 2026-10-17. It is the only encoder clone with its own head type (`von`) in the HF transformers decision-model PR.

| ref | what | where |
|---|---|---|
| `G` = GitHub `a47a74e` (2026-10-17, "docs: add archived status banner") | shallow clone, never executed | `/home/ai/projects/jev-clones/sources/von` |
| `HF` = `wfzyx/von` @`498ceba3` (lastModified 2026-10-02) | api.json, tree, README, config, `calibration.json`, `marker_calibration.json`, `run.log`, tokenizer_config (no weights) | `notes/hf-raw/wfzyx__von/` |
| `PR` = huggingface/transformers #49566 (draft, ArthurZucker) | `models/von/*`, `docs/model_doc/von.md`; decision repo `ArthurZ/von` | `notes/hf-raw/_transformers_pr49566.diff.txt`, `notes/hf-raw/ArthurZ__von/` |

Citations take the form `G:src/von/models/option_marker.py:240`.

---

## Identity

| | |
|---|---|
| Author | Victor Hugo Panisa (`wfzyx`) (`HF:README.md` Citation). Independent, with no company behind it. The model is named after von Neumann and von Mises (`G:docs/benchmarks.md` "Theoretical Homage") |
| Repo | github.com/wfzyx/von, Apache-2.0, created 2026-09-18, **archived 2026-10-17**, ★917 (gh API). Weights: `wfzyx/von` (the old id `wfzyx/von-1.0` redirects there). 118k downloads, 62 likes |
| Packages | PyPI `von-sdk` (≥1.3.7), npm `von-sdk`, Docker `ghcr.io/wfzyx/von:cpu` (OpenVINO) (`G:README.md:22-28`) |
| Versions | 1.0 (2026-09-19, NLI cross-encoder, inferred) → 1.1 (option-marker, full cross-option attention) → **1.2** (weights; independent options, released 2026-09-24) → **1.3** (2026-09-28, engine only: chains, band rule; weights unchanged). Last SDK is 1.3.7 (2026-10-02) (gh releases, `HF:README.md` header) |
| Von 2 | Started 2026-10-02 (`G:docs/von-2.md`): Qwen3.5-0.8B + LoRA r64 + vendored Cloudflare Clef `JointSchemaHead`. Runs r1 and r2 trained; r3/r4 were staged (polarity data, a pair-contrastive loss in `9319ced`, 10-08). No von-2 weights were published |
| Why archived | `G:docs/von-2.md` §0 sets a go/no-go rule on 2026-10-07: "r3 must beat LiquidAI d1-omni-600M on the Decision Index 0.3 public index, 17.9, or the project stops", followed by "write the post-mortem into this file and archive". No r3 result or post-mortem was committed. After the 10-08 r4 code commit, the only remaining commit is the archive banner. **The bar was presumably not cleared (inferred)** |
| Decision Index 0.3.2 | "Von 1.3", full fine-tune, technique `encoder`, 0.40B, base `answerdotai/ModernBERT-large` (`notes/decision_index_0.3.2_entries.tsv`). DI 0.3 Full **8.2, #101 of 114** (public 13.9, same-skill 12.8, new-domain 4.9 → 0) (`G:docs/von-2.md` §1). DI v0.2.1: 13.74 (`HF:README.md`) |

## Architecture

### Backbone
- ModernBERT-large: 28 layers, d=1024, 16 heads, a global attention layer every 3rd layer, local window 128, RoPE θ 160k global / 10k local, **8,192 positions**. 394.8M F32 params (`HF:files/config.json`, `HF:api.json` safetensors).
- All weights are fine-tuned, with no LoRA. The fp32 checkpoint is 1.5 GB. `option_marker.pt` (1.58 GB) is a second full copy plus the head (`HF:tree.json`), which is why "~3 GB" is downloaded (`G:README.md:30`).
- The HF `config.json` still carries `ModernBertModel` with 134 `tasks` and NLI `id2label`. That is the fingerprint of `tasksource/ModernBERT-large-nli`, which is the default `model_id` of the 1.0 trainer (`G:training/train_rlcd.py:144`). The lineage is therefore tasksource NLI → von-1.0 cross-encoder → option-marker (inferred from the config and the trainer defaults).

### Input layout
- **Each question is its own sequence**: `[CLS] {instructions} {state} [SEP] [MASK] opt0 [MASK] opt1 … [SEP]` (`G:src/von/models/option_marker.py:224-243`). The question comes *before* the state.
- State JSON becomes `key: value` lines (`G:src/von/backends/option_marker_backend.py:51`).
- Optional `digit_split` puts a space between every digit, e.g. "2026" → "2 0 2 6" (`option_marker.py:22-39`). It is enabled in 1.2 per the card ("digit-split tokenisation").
- `[MASK]`/`[SEP]` literals in user text are broken with a zero-width joiner, so option boundaries cannot be forged (`option_marker.py:160-165,238`).
- Over-long states are middle-truncated (60% head / 40% tail) so that the question and options always fit. `--on-overflow refuse` returns a 422 instead (`G:README.md` flag table, `option_marker_backend.py:452-480`).

### Attention & question isolation
- **Between questions.** `evaluate()` loops over the questions and runs one encoder pass per question (`option_marker_backend.py:853-866`), so they are isolated trivially. The state is **re-encoded for every question**; no prefix is cached, and none could be, because the question precedes the state. The README claim that "several questions over one state cost one forward pass" (`G:README.md:59`) does not match the code. `usage.input_tokens` sums every pass, so the state is counted per question.
- **Between options, ≤1.1.** Full bidirectional attention: options see each other and their packing order. 1.1 flipped 49.5% of answers on JevBench hard over 4 orderings (`HF:README.md` Architecture).
- **Between options, 1.2+ (`independent_options: true` in `marker_calibration.json`).** Prefix tokens attend to the prefix. An option token attends to the prefix plus its own span. The trailing `[SEP]` is kept out of the last option's span so that the last slot is not special (`option_marker.py:63-121`).
  - Every option's `position_ids` restart at the prefix length (`:124-156`).
  - The sliding-window mask is recomputed from those ids rather than from raw indices (`:112-119`).
  - Result: logit_k = f(prefix, option_k). 0/111 flips (`HF:marker_calibration.json` `training.order_invariance`; test `G:tests/test_option_invariance.py`).
  - A side effect: the prefix (question + state) cannot see the options at all.
- **PR form.** `VonDecisionHead.backbone_inputs` builds the same mask with the masking API (`packed_sequence_mask_function` over block and option ids, a bidirectional mask, a sliding window on the reset positions). It works for sdpa, eager and flex attention (`PR:models/von/modeling_von.py:62-104`). The PR test checks that an option's states are identical when it is moved from slot 2 to slot 1 (`PR:tests/models/von/test_modeling_von.py`).

### Readout head
`OptionMarkerScorer`: LayerNorm → Linear 1024→512 → GELU → LayerNorm → Dropout → Linear 512→1, applied to the last hidden state of each option's `[MASK]` (`option_marker.py:41-60,214-222`). The answer is the softmax over the K marker logits. The head is identical in the PR (`VonDecisionHead`, `PR:modeling_von.py:47-55,106-121`).

### Question types
| type | options fed | output | confidence |
|---|---|---|---|
| choice | criteria descriptions in request order | argmax + probabilities | `(n·p_max−1)/(n−1)` = Jev's formula (`option_marker_backend.py:36-50,678`) |
| noul | `[true desc, false desc]`, defaulting to "Yes, condition holds true." / "No, condition is false." (`:702-708`) | `noul = p[0]` (`:760`) | none |
| score | level texts; dict levels may carry `what` + `examples` (`:790-803`) | `Σ i·p_i`, legend, probabilities (`:830`) | **the same choice-margin formula** (`:832`), not Jev's MAD-based score confidence |

Noul has two special paths:
- **Zero-shot debiasing.** Without criteria, a second "null" pass with an empty state is run, and the true-logit is lowered by `a·(null_true−null_false)+b`. The prior is a = −0.5, b = 0.3, fitted on 150 BoolQ + 25 handwritten items; without it a flat 0.7 is used (`:728-752`, `HF:marker_calibration.json`).
- **Band rule** (1.3.2+, on by default). Maps p to `0.8 + 0.1·(p−0.5)` above 0.5 and mirrors it below, so every answer leaves JevBench's 0.2–0.8 abstention band. The pre-band value is exposed as `noul_raw` (`:132-155,770-775`).

**Chain-of-options** (1.3): a regex proposer finds dates, durations and amounts in the state; a TOML operator library (8 chains) computes facts; Von itself chooses ambiguous slots; a computed value either answers directly or is appended to the state (`G:README.md:120`, `G:src/von/chains/`). Hard tier went 0.378 → 0.441 (McNemar p = 0.065, not significant).

### Limits
- 8,192-token window per question; chains stand down above 4,096 state tokens.
- No hard cap on option or question count; K is bounded only by the window (inferred from the code; none found in `types.py`).
- English only. Text only (`G:docs/multimodal.md` is a design note, not a shipped feature).

### Calibration
- **Input-conditioned temperature.** `T = 2.0151 − 3.2369·H_norm + 11.4916·log10(state_tokens)/4 − 3.5574·K/8`, clamped to [0.3, 12]. H_norm is the normalised entropy of the raw distribution (`option_marker_backend.py:484-537`, `HF:marker_calibration.json`).
  - It is monotonic, so the argmax never changes.
  - It was fitted on the **231 public JevBench items** (objective 0.5·JevBench-Calibration + 0.5·per-tier honesty), so it is in-sample: ECE easy/standard/hard 0.109 / 0.045 / 0.089.
  - The card says outright that it "does not carry to other distributions".
- `von calibrate labels.jsonl` refits the map on CPU, with weights frozen. It chooses between a scalar and the feature map by k-fold CV. On 38 coding-agent labels, a scalar T = 2.15 took ECE from 0.21 to 0.11 (`G:src/von/calibrate.py`, `HF:README.md` Calibration).
- The 1.0 cross-encoder stage used a scalar post-hoc T = 1.1692 (`HF:calibration.json`, `HF:run.log` tail).

## Training
- **1.0 (inferred).** `train_rlcd.py`: `AutoModelForSequenceClassification` NLI cross-encoder. For each option it takes the entailment logit of a (premise, option) pair, applies a softmax over the options, and trains with CE + 0.5·Brier, cosine schedule, 10% warm-up, then a post-hoc T (`G:training/train_rlcd.py:83-139,175-185,218-222`).
- **1.0 → 1.1 continuation.** `HF:run.log`: 65,991 "operational decision" samples, 1 epoch, 4-GPU DDP, effective batch 64. Val acc 96.43%, T 1.1692. The run is spot-launched on AWS (`launch_aws_spot.py`).
- **Option-marker trainer.** `train_option_marker.py`: listwise softmax CE + 0.5·Brier against a one-hot or **soft** target distribution (`:220-245,275-318`), lr 3e-5, token-budgeted batches with a long-sample ratio (`:113-216`).
- **1.2 recipe.** Full retrain from the 1.1 weights with `independent_options=True`: 1 epoch, lr 3e-5, batch 8 × accum 2 × 4 GPU. Corpus: 290k "universal" + 40k long-context + 40k synthetic two-hop/numeric (`HF:marker_calibration.json` `training`).
- **Corpus mix** (`HF:README.md` Training data):

  | Share | Cluster |
  |---|---|
  | 25% | ops/enterprise |
  | 20% | security/DevOps |
  | 15% | safety/policy |
  | 15% | linguistic |
  | 10% | triage |
  | 15% | ANLI R1–3 + WANLI anchor |

  All of it is synthetic or public. Builders are in `G:training/prepare_*`, `generate_*`. A `contamination_audit.py` exists. No JevBench item is used for gradients.
- **"RLCD" is a misnomer.** The name cites DeepMost's arXiv 2503.23303, not TypeSafe's method. The objective is supervised CE+Brier; no RL is involved (`G:docs/benchmarks.md` refs).
- **1.4 attempts, all null on the paired McNemar gate and not shipped:**
  - GLiClass trunk init;
  - judge/diversity mix;
  - stack (`G:results/v1.4/*.json`, commits `aa3dd07`, `80f73b6`, `99ffb2a`).

  `G:docs/von-2.md` §1 lists seven encoder-side interventions that "came back null".

## Evals
- **JevBench v1.4, Von 1.2:**

  | Axis | Score |
  |---|---|
  | Intelligence | 34.5 |
  | **Calibration** | **75.7** (Jev 76.3) |
  | Speed | 70.5 |
  | Cost | 77.8 |
  | Composite | 27.5 (Jev 63.3) |
  | Hard | 0.373 |
  | Sealed | 0.279 |
  | Sealed ECE | 0.107 (best in the encoder class) |

  Source: `G:README.md:126-133`. JevBench v1.5.1: Noul −92…−100 because of the abstention rule, which the 1.3 band rule fixes. The open→sealed Choice gap is 14.7 pp, against a field median of 5.2 (`HF:README.md`).
- **Out of sample:**
  - jabr v2: 72.0% macro (Choice 83.0%);
  - Decision Index v0.2.1: 13.74 (raw 34.11, coverage 99.96%, p50 32.8 ms on an A10G);
  - ViZDoom Defend-the-Center: 9.0 kills at ~18 ms per decision, vs Jev 1.13 at 5.6 (`HF:README.md`, `G:docs/benchmarks.md`).
- **DI 0.3 per benchmark.** Strong on intent: BANKING77 75, CLINC150 60. Weak on reading: BFCL 15 vs 69, ARC-Easy 44 vs 80, ContractNLI 0 vs 61, API-Bank 0 vs 29 (`G:docs/von-2.md` §1).
- **Von 2 nano** (Qwen3.5-0.8B + Clef head, unreleased): pooled held-out 70.7 / 71.1 for r1 / r2, vs 64.7 for Von 1.2, p < 0.05. The **Noul yes-bias** survives the trunk swap, so the author diagnoses it as a data defect (`G:docs/von-2.md` §4-5).
- **Method.** Every claim goes through a paired exact McNemar test plus a minimum detectable effect, with "UNRESOLVABLE" below the MDE (`G:benchmarks/stat_gate.py`).

## Serving
- `von serve`: FastAPI `/v1/systemone`, byte-compatible with TypeSafe. It accepts `VON_BASE_URL`, and bearer auth turns on when `VON_API_KEY` is set (client falls back to `TYPESAFE_API_KEY`).
- Extras: a `truncation` field and `X-Von-Truncated` header, plus `output_tokens = number of answers` (`option_marker_backend.py:887`).
- Devices: auto-picks cuda, mps, OpenVINO GPU/CPU (compiled graphs for both mask modes, `:245-332`), or cpu. ROCm is supported too.
- Latency (raw p50): 0.096 s on 4 vCPU (OpenVINO) and 0.023 s on an A10G (`G:results/speed_remeasure.md`). Chains push the hard-tier p50 to 4.2 s on CPU.
- Single global engine with a lock (`G:src/von/engine.py:49`). Patterns: `confidence_gate` at a measured 0.80 threshold, where Choice keeps 25% of items at 92.4% accuracy (`G:README.md:63`).
- Also packaged by `ollaya-dev/ollaya`, and via **transformers `pipeline("decision", model="wfzyx/von", decision_config="ArthurZ/von")`**. The Jinja template there reproduces the SDK packing, the null block and the state-token feature for the temperature map (`PR:docs/model_doc/von.md`, `ArthurZ__von/files/decision_template.jinja`, `decision_config.json`). It was contributed 2026-10-16, after von itself had stopped.

## Licence
Code and weights are Apache-2.0 (`G:LICENSE.md`, `HF:api.json` cardData). ModernBERT-large is Apache-2.0. Training sources such as ANLI, WANLI and Banking77 carry their own licences, which are not audited here.

## What is novel vs Jev and kev
1. **Encoder, not decoder.** A 395M bidirectional model on CPU at about 0.1 s. That is the opposite size bet to kev's 4B–27B Qwen decoders. It is the most-downloaded encoder clone, at 118k.
2. **Order invariance as a hard architectural guarantee** (masking plus per-option position resets, including the sliding window), shipped as default weights. kev has the same idea (`option_isolation`) but keeps it off because it cost 5.8 pp. Jev itself reads options jointly and leans toward the first option. The cost in Von is that options cannot see each other, and the prefix cannot see the options, so "none of the above" style reasoning is lost (inferred).
3. **Per-option marker MLP, not a pointer.** kev dot-products `</opt>` against a `<decide>` query that has read all the options. Von scores each `[MASK]` alone, with no question token.
4. **No shared-state economics.** Unlike Jev, kev and the DeltaNet row form, the state is re-encoded once per question, and the question is placed before the state.
5. **Feature-conditioned temperature** (entropy, state length, K) plus a CLI refit on user labels. This is more elaborate than kev's single T, but fitted in-sample on JevBench.
6. **Jev-wire engineering that kev lacks:**
   - a context-free null pass that debiases zero-shot Noul;
   - a benchmark-driven Noul band rule;
   - deterministic **chain-of-options** tools for dates and numbers, with zero generated tokens.
7. **A documented negative result.** The author's own conclusion (`G:docs/von-2.md` §1) is that a 400M encoder cannot reach DI ≥ 50 "on recipe", and that head + data + RL matter more than scale within a size class. That led to the planned pivot to a 0.8B decoder with a Clef head, and then to the archive.
