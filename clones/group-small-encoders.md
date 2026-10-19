# Group: small encoders (≤ ~450M, CPU/browser-class)

Compiled 2026-10-19. Sources: HF API (`?expand[]=safetensors`) and small text files (README, configs, custom code)
saved under `notes/hf-raw/<org>__<model>/` (`api.json`, `tree.json`, `files/`); GitHub via `gh api` and read-only
shallow clones `sources/blink` (sqliteai/blink @ 18d6ce8) and `sources/mayaz` (acl-research/MayaZ @ 295deb8). No
weights downloaded, no upstream code run. Decision Index (DI) = `multimodalart/jev-decision-index` 0.3.2 snapshot
(`notes/hf-raw/_spaces/dix/index.json`, 2026-10-16; Full = `balanced_skill`, chance-corrected; Jev 1.14.0 = 61.53,
Jev 1.13 = 60.11). DI latency is GPU median per request on 1× RTX PRO 6000. "(inferred)" marks our reading, not a
stated fact.

## Comparison table

| Family (main repo) | Params | Backbone | Enc/dec | Option marking | Layout per pass | Context (shipped) | CPU latency (as reported) | Calibration | DI 0.3.2 Full (ECE) |
|---|---:|---|---|---|---|---|---|---|---|
| Julia-1 (SupersonicLabs) | 144.3M | mmBERT-small | encoder | `[MASK]` before each option (Laya layout) | 1 question/row, options bidirectional | 8192 runtime / 1024 benchmarked; 2–20 opts | ~0.30 s/question, 4 threads fp32 (derived) | none shipped (`calibration: null`) | 2.56 (0.420) |
| Phocinae-Largha-150M-v1 | 144.3M | mmBERT-small | encoder | Laya layout (inferred: `rl_agent_config.json`) | 1 question/row | 512 / head 192 | 1.64 s per 5-question case, 1 thread | per-type T + optional power γ | not listed |
| Verdict / rlcd-modernbert-151m (heman10x) | 151.4M | GLiClass-modern-base-v2.0 (ModernBERT-base) | encoder | `<<LABEL>>` tokens prepended, `<<SEP>>` then text | 1 question/row, labels first | 512 (trained on <71-token states); 25 slots | p50 35.6 ms (hardware unstated) | per-k temperature | 0 (shown −1) (0.154) |
| open-jev-deberta-v3-large (com-kotobalabs) | 434.0M | DeBERTa-v3-large | encoder | `[STATE] [Q] [OPT]` added tokens, span mean pool | **all questions in one sequence, no isolation** | 512 (state 256) | 1.8 s for 4 questions, M1 Max fp32 | 1 global T=1.05 | not listed |
| Decima (amyrmahdy) small/base/agent/guard | 122M / 321M | multilingual-e5-small / mmBERT-base/-small | encoder, late interaction | none: each option encoded on its own | state+question once, options separately, cross-attn scorer | 512 state, 64/option (agent 8192, guard 2048) | ~16–20 ms, 1 x86 core int8, 4 opts | 1 global T, ordinal head for score | not listed (self-run DI 0.1: 12.8) |
| Bekko System One v0 (hotchpotch) 17M/68M/400M | 16.9M / 68M / 395M | Ettin rerankers (ModernBERT) | encoder, shared-prefix mask | candidate branches, mean-pooled | prefix (instr+state) bidir; branches see prefix+self only | 7,999 (train 4096/2048) | CPU fp32 not quantified; browser ONNX | none stated ("not guaranteed") | 2.93 / 8.42 / 15.24 (0.161 / 0.113 / 0.111) |
| MayaZ (acl-research) 17M/32M/68M | 17.7M / 33.9M / 71.7M | Ettin encoders | encoder, parallel positions | `[OPT] … [/OPT]` (`[unused0]/[unused1]`) | option k sees shared+self; shared positions restart | 512 positions, 1536 tokens/pass, 64 opts (256 tournament) | 0.60 / 1.50 / 6.35 s p50 at **0.1 vCPU**; WASM 1-thread 0.1 / 0.3 / 1.0 s | per (type × k-bucket) T | not listed |
| Crivo (tardellirs) 17M–1B, focus 150M | 149.6M | ettin-reranker-150m-v1 | encoder | `[OPT]` (init from `[SEP]`) after state | 1 question/row, options bidirectional | trained 8192, reads to 32,768 | no CPU figure (A100 18 ms) | **no temperature** | 150M v1 22.21 (0.144); 32M 11.87 (0.087) |
| Blink (sqliteai) | 407k (tiny) | none, from scratch, byte-level | conv/attn encoder, separate streams | none: options encoded separately, cosine attention | state, question, menu encoded separately; options independent | 256 B state / 128 B question / 48 B option; ≤64 opts | **406 µs** fresh, 54 µs state reused (1 core M5 Pro) | 1 global T (bounded fit) | not listed |
| *Ref: Laya* (`clones/laya.md`) | 421M | ModernBERT-large | encoder | `[MASK]` | 1 question/row | 512 / 192 | – | per-type + bucket T | 4.43 (0.140) |
| *Ref: Zeiger* (`clones/zeiger.md`) | 0.6B (+18M head) | Qwen3-0.6B-Base | causal decoder as encoder | `<\|object_ref_end\|>` after each option | 1 question/row, causal | 2048 chunks, trained to 32k | 340 ms at 10 options, 16 cores bf16 | T per option-count bucket | not listed |

Pattern in one line: everything here is a **bidirectional encoder** (no decoder; Blink is not even pretrained), and
the families split by **how options meet each other**: joint listwise (Julia, Phocinae, Verdict, Crivo, Kotoba),
masked-apart branches (Bekko, MayaZ), or fully independent options (Decima, Blink), the last two being permutation
invariant by construction.

---

## Julia-1 (Supersonic Labs)

**Identity.** <https://huggingface.co/SupersonicLabs/Julia-1>, created 2026-09-23, modified 2026-09-26, 5,818
downloads, 487 likes, Apache-2.0. ONNX/WebGPU export in `SupersonicLabs/Julia-1-ONNX` (2026-09-25). No GitHub repo.
"First released test of our training system".

**Backbone.** `jhu-clsp/mmBERT-small` fully fine-tuned: `model_type modernbert`, 22 layers, d=384, 6 heads, vocab
256k, max positions 8192, local window 128, global every 3rd layer (`files/encoder/config.json`). 144,292,870 params
total (API safetensors), fp32.

**Input layout.** Laya's serialization, re-implemented: `[CLS] <type> question … [SEP] [MASK] opt0 [MASK] opt1 … [SEP]
<state> [SEP]`; each option cut to 48 tokens, `[MASK]` in user text rejected under strict encoding
(`files/julia/data.py:75-106`). Head budget 512, total 8,192 (`inference-policy.json`).

**Isolation.** One sequence per question; questions batched as separate rows. Options attend to each other fully
(bidirectional). Same as Laya's "sequential" layout.

**Readout.** Exact copy of Laya's `DecisionModel`: `type_emb` (3 types) added to hidden states, 2×
`TransformerEncoderLayer(d, d/64, 4d)` over the whole sequence, `scorer = LN→Linear→GELU→Linear` at marker positions,
plus an unused-in-practice `act_head` (n_act 2) and a 3-slot `temperature` buffer (`files/julia/model.py:14-73`).
Eval-only shortcut `_selected_head` computes the last block only at marker positions (`model.py:76-94`).

**Question types.** `choice` 2–20, `score` 2–20 levels (expected zero-based index), `noul` two slots false/true; a
hierarchical `Router` narrows longer choice lists (README "larger choice sets").

**Calibration.** None shipped: `"calibration": null` (`inference-policy.json`); temperatures stay 1.0 (inferred from
buffer init). DI ECE 0.420 is the worst in this group.

**Training data.** Private pipeline; `provenance.json` lists validation on AG News, Banking77 (joint/rank), bilingual
EN/PT, DAIR Emotion and `jev:*` probes (ARC 0.285, MMLU 0.263), 500 post-training steps from a parent checkpoint.

**Evals.** typed-decisions test 73.15% (choice 71.3 / noul 80.7 / score 68.9); AG News pilot 94/100; MASSIVE 52
locales 71.5% (README). CPU re-run: 2,000 questions in 601.7 s on 4 threads at 1024 tokens (`metrics/typed-cpu-
20260926.json`), i.e. ~0.30 s per question (derived). Literal-criteria noul ablation drops noul to 65.2%. **DI 0.3.2
Full 2.56**, raw 27.52, acc 0.374, ECE 0.420, GPU median 5.8 ms.

**Novel.** Little architecturally: a Laya head on a 3× smaller multilingual backbone, plus strict-encoding refusal
instead of silent truncation. Shows the Laya head transfers to mmBERT-small, and that without a fitted temperature
it is badly calibrated off-distribution.

## Phocinae-Largha-150M-v1 (Phocinae)

**Identity.** <https://huggingface.co/Phocinae/Phocinae-Largha-150M-v1> and <https://github.com/Phocinae/Phocinae-Largha-150M-v1>
(Apache-2.0, 191 stars, created 2026-10-08, pushed 2026-10-15). HF: 713 downloads, 236 likes, Apache-2.0. EN/ZH.

**Backbone.** mmBERT-small again (`files/encoder/config.json` identical to Julia's: 22 L, d 384, 6 heads, 256k vocab,
8192 positions). 144,292,870 params, **bit-identical count to Julia-1**, which their own README notes
(`README.md:169`). fp16 safetensors 288.6 MB.

**Input layout.** Laya's checkpoint format: `rl_agent_config.json` has `encoder`, `head_layers 2`, `max_len 512`,
`head_max_len 192`, `act_costs {escalate: 0.5}`, three per-type temperatures. Served by "phocinae-server". No modelling
code in the repo; layout is Laya's `[MASK]`-marker sequence (inferred from the config keys, which are Laya's).

**Isolation.** One sequence per question (inferred, Laya runtime).

**Readout.** Laya head (2 transformer layers, head context 192); the tech report calls it "over mean-pooled encoder
output" (`docs/technical-report.md` §2), which contradicts the Laya marker head the config implies; treat the
report wording as loose (inferred).

**Question types.** noul (threshold 0.5), choice (0-based index), score 2–10; plus a τ=0.6 "escalate" gate on
confidence.

**Calibration.** Shipped temperatures 0.866 / 0.808 / 0.662 (`rl_agent_config.json`) give ECE 0.252 (en). Optional
`calib/` post-hoc power transform `q ∝ p^γ` (γ 4.33 en, 2.51 zh) gets ECE 0.017 but raises Brier 0.021→0.122
(`calib/README.md`). The reported Brier values look implausibly low for ECE 0.25 (flag, not verified).

**Training data.** `LocalLLaMA/typed-decisions` train split (1,200 cases) plus flip-augmented reorderings, ~2,400
machine-translated and ~1,400 native Chinese rows; CE+Brier, Adafactor, 300 updates, 0.044 GPU-hours.

**Evals.** typed-decisions en/zh 0.791/0.848 (maintainers' curated re-score), JevBench public-231 0.5455; option-flip
rate 0.028 (any of 3 reorders). Latency: RTX 5090 fp16 p50 21.0 ms; CPU 1 thread 1.64 s per 5-decision case; CPU
8 threads 8–20 decisions/s. Not on DI 0.3.2.

**Novel.** Nothing architectural; a 0.04 GPU-hour fine-tune of the Laya recipe on mmBERT-small with an explicit
cost-of-escalation write-up. Useful mainly as a second data point that the Laya head on mmBERT-small runs at
~0.3 s/decision on one CPU thread.

## Verdict / rlcd-modernbert-151m and openJev-verdict-2.0 (Heman10x-NGU)

**Identity.** HF <https://huggingface.co/heman10x/rlcd-modernbert-151m> (created 2026-09-17, modified 2026-09-20,
40,980 downloads, 38 likes, Apache-2.0). GitHub <https://github.com/Heman10x-NGU/Verdict-open-jev> (110 stars,
2026-09-17) and <https://github.com/Heman10x-NGU/openJev-verdict-2.0> (295 stars, created 2026-09-19, pushed
2026-09-20, licence NOASSERTION). Verdict 2.0 weights are only a Git LFS pointer (`artifacts/verdict2-base/model.pt`,
134 B in the tree); `heman10x/openJev-verdict-2.0` does not resolve on HF. Raw: `notes/hf-raw/heman10x__rlcd-modernbert-151m/`,
`notes/hf-raw/github__Heman10x-NGU__openJev-verdict-2.0/`.

**Backbone.** v1: `knowledgator/gliclass-modern-base-v2.0`, i.e. GLiClass `uni-encoder` over ModernBERT-base
(d 768, vocab 50,370 with added class tokens, `max_num_classes 25`, `scorer_encoder_num_layers 2`, `class_token_pooling
first`; `files/config.json`). 151,378,176 params. v2.0: plain `answerdotai/ModernBERT-base` with a Laya-style head
(`verdict2/model.py`).

**Input layout.** v1: `<<LABEL>>desc1<<LABEL>>desc2…<<SEP>>Question: …\n\nContext: …` with candidate text templated as
`It is {description}` (NLI hypothesis framing, added in v1.4) and an extra `__insufficient_evidence__` candidate
(`core/formatting.py:20-120`). v2.0: Laya's `[CLS] <type> <question> [SEP] [MASK]opt … [SEP] state [SEP]` (README
quickstart).

**Isolation.** One sequence per question; labels and text bidirectional inside it.

**Readout.** v1: GLiClass label-token pooling + scorer, up to 25 logits (24 options + abstain), truncation above that
(`core/engine_encoder.py:190-197`). v2.0: marker hidden + `type_emb` → MLP logit, softmax; a **separate 7-feature
correctness head** (p1, margin, entropy, 1/K, type one-hot) predicts P(argmax correct) (`verdict2/model.py:20-90`).

**Question types.** choice/score (levels as `desc (Value: v)`) and noul as `true: P` / `false: not P`, each plus the
abstain slot.

**Calibration.** v1: per-k temperatures 1.5–5.0 fitted on 1,753 held-out items (`calibrator.json`). v2.0: temperature
per (type × k bucket 2..9) plus the correctness head (ECE 0.0144 on its own channel). Training loss CE + 1.0·Brier;
v2.0 adds symmetric permutation-KL between option-shuffled twins (`verdict2/losses.py:32-53`).

**Training data.** v1: Banking77/CLINC-OOS plus 12 generated task files (`data/gen/T01..T12.jsonl`), cardinality
slices k3–k25, 3 epochs on MPS in 646 s (`train_manifest.json`). v2.0: typed-decisions, 8.8 h on a GTX 1660 Ti.

**Evals.** v1 card: Banking held-out 95.0%, ECE 3.4%, OOS abstention recall 97.5%, p50 35.6 ms. v2.0 README:
typed-decisions 77.1%, Brier 0.064, flip 4.76%. **DI 0.3.2 Full 0** (board lists −1), raw 12.91, acc 0.369, ECE 0.154.

**Novel.** The explicit abstention candidate, and (2.0) splitting "distribution" and "is the argmax right" into two
heads, which is the cleanest attempt here at Jev's distinction between probabilities and `confidence`. v1 is the most
downloaded model in this group but the weakest on DI: trained on short banking states, it does not generalize.

## open-jev-deberta-v3-large (com-kotobalabs / Mithril)

**Identity.** <https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large>, created 2026-09-18, modified
2026-09-24, 3,579 downloads, 74 likes, Apache-2.0. Code <https://github.com/kotoba-lang/typed-decisions> (10 stars,
created 2026-09-18, licence NOASSERTION). Publisher Mithril (formerly Kotoba Cloud).

**Backbone.** `microsoft/deberta-v3-large`: 24 layers, d 1024, 16 heads, 512 positions (relative), vocab 128,100
plus 3 added markers (`files/config.json`, `added_tokens.json`). 434,012,160 params; eager attention.

**Input layout.** `[CLS] [STATE] state [Q] instructions [OPT] opt1 [OPT] opt2 … [Q] … [SEP]`; state cut to 256
tokens, total 512, overflow raises (`typed_decisions/encoder.py:103-123`).

**Isolation.** **None.** State and all questions sit in one sequence under a plain padding mask
(`encoder.py:54`), so every question attends to every other question and its options. This is the only model in the
group that breaks Jev's "adding a question does not change other answers" property (inferred from the code; not
measured by the authors).

**Readout.** `pool: span`: mean of each option's tokens and of the question's tokens, features `[q; opt; q·opt]`
into a 3-layer head (`head.safetensors`), softmax within each question's option group (`encoder.py:57-67`). The card
reports that a fresh marker-token head "does not learn" here while the span head does.

**Question types.** choice up to 255, score 2–10 (expected index), noul as 2 options `("no","yes")` read as
p(options[1]) (`schema.py:24`).

**Calibration.** One post-hoc temperature 1.05 on a validation split (`open_jev_config.json`); loss CE + Brier.

**Training data.** Public gold only: banking77, SST-5, BoolQ; 18k states / 42k questions, 1 epoch, 229 s on one H100,
augmentation p=0.7 (shuffle, paraphrase, distractor drop, level relabel, noul negation).

**Evals.** In-domain acc 0.854 (ECE 0.022), OOD (new instructions/option sets) 0.690 (ECE 0.035), banking77-77 0.916.
CPU: 1.8 s for 4 questions on M1 Max fp32; H100 28 ms for 10 questions. Not on DI 0.3.2.

**Novel.** All questions in one pass at encoder scale (cheap, but non-isolated), and the honest in-domain vs OOD
split showing how much of the question such an encoder actually reads.

## Decima (amyrmahdy): small, base, base-clean, agent, guard, guard-small

**Identity.** <https://huggingface.co/amyrmahdy/decima-small> (2026-09-27, 253 dl) plus `decima-base` (2026-10-04,
120 dl), `decima-agent` (2026-10-04, 144 dl), `decima-guard`, `decima-base-clean`, `decima-guard-small` (all
2026-10-18), all modified up to 2026-10-19, Apache-2.0. Code <https://github.com/amyrmahdy/decima> (Apache-2.0,
created 2026-09-27, pushed 2026-10-19), PyPI `decima-ai`. Weights ship as int8/fp32 ONNX plus a PyTorch folder, so
the API reports no safetensors count.

**Backbone.** small: `intfloat/multilingual-e5-small` (BERT, 12 L, d 384, 512 positions, vocab 250k), 122,388,869
params (`files/config.json`). base/agent/guard/base-clean: `jhu-clsp/mmBERT-base` (22 L, d 768, 8192 positions),
321,125,381 params. guard-small: mmBERT-small (~140M).

**Input layout.** Late interaction. State encoded with the question prepended (`question_in_state: true`, prefix
`query: ` on e5); each option encoded on its own, with the question (`question_in_choices: true`, prefix `passage: `),
≤64 tokens per option (`pytorch/decima.json`).

**Isolation.** Questions isolated (each question is its own state encoding, since the question is part of it).
Options never share a context: each reads the state through 2 cross-attention layers, so option scores depend only on
(state, option) and the softmax is permutation-equivariant by construction (README "Under the hood" §1).

**Readout.** Scorer: 2 cross-attention layers (6 heads, FFN 1536) → logit per option, softmax/T. `score` uses a
cumulative-link ordinal head with monotone thresholds computed from each level's state-conditioned vector; `rank`
uses an independent sigmoid per option (README §3).

**Question types.** choose / verify (noul) / score / rank; `/v1/systemone` server mapping choice/noul/score
(`python -m decima.serve`). Options cached per option set; cost ≈ t(state) + n·t(scorer).

**Calibration.** One global temperature per checkpoint (0.94 small, 0.73 base, 0.61 agent) fitted on a 5% hold-out.

**Training data.** ~300k decisions soft-labelled by a Gemma-4-26B-A4B teacher across domain × language × kind ×
option-count; v1.1 adds 45,729 teacher-labelled Jev-format items; MASSIVE and XNLI/MNLI train splits (so those evals
are in-distribution). base-clean drops non-commercial data and all BTZSC overlap.

**Evals.** small: jabr/classifier-benchmark v2 0.616 (Laya 0.583, Jev 0.966); option-shuffle flip 0.0%; MASSIVE 13
non-EN langs 0.857; typed-decisions 0.427 (below majority 0.461); self-run DI 0.1 = 12.8, below chance, because
tools/retrieval requests exceed its 512/64 budget. base: jabr v2 0.673, ECE 0.051. CPU: **16 ms at 4 options, ~1 ms
per extra option, 1.06 s at 1,000 options** on one x86 core int8 (small); base 47 ms on 1 ARM thread, 21 ms on 4.
Not on DI 0.3.2.

**Novel.** The only text model here that is order-invariant by construction and scales to 1,000 options with option
caching, plus block-wise int8 (blocks of 32) that keeps int8 at 99.4% agreement with fp32. The trade-off is that
options cannot compare against each other, so tasks that need cross-option reasoning lose (inferred).

## Bekko System One v0 (hotchpotch): 17M / 68M / 400M

**Identity.** <https://huggingface.co/hotchpotch/bekko-system-one-v0-68m> (+ `-17m`, `-400m`), created 2026-09-29,
modified 2026-09-30, likes 14/3/15, **0 downloads shown** (custom sentence-transformers code; the Hub counter
probably does not trigger, inferred). Weights licence "remains to be finalized"; code
<https://github.com/hotchpotch/bekko-system-one> MIT (37 stars, 2026-09-26). Author also runs the S1MB benchmark.

**Backbone.** Full fine-tunes of `cross-encoder/ettin-reranker-{17m,68m,400m}-v1` (card; DI lists the plain Ettin
encoders): ModernBERT, 7 L/d 256, 19 L/d 512, 28 L/d 1024; 7,999 positions; local window 128, global every 3rd
(`0_BekkoInference/backbone_config.json`). 16.8M / 68M / 395M total; "active" without embeddings 3.9M / 42M / 343M.

**Input layout.** Prefix = instructions + state (two orders trained, `prefix_layout instruction_state`); each
candidate is its own branch. Native limits: context 7,999, query 7,997, candidate 3,800 tokens, adaptive budget
(`inference_config.json`).

**Isolation.** Custom mask: prefix bidirectional over itself only; each candidate branch attends to the prefix and
to its own tokens; branches do not see each other; prefix K/V reused across candidates within a microbatch (README
"Architecture and input limits"). Decisions on one state are separate prefix encodings.

**Readout.** Mean pool over each candidate branch, task-specific heads for choice/noul/score, softmax over candidates
within a decision. Score = probability-weighted expectation of explicit numeric `value`s.

**Question types.** choice, noul (authored true/false meanings required), score (numeric rubric), plus relative
document ranking.

**Calibration.** Not claimed: "outputs are not guaranteed to be calibrated". DI ECE ≈ 0.11–0.16.

**Training data.** `hotchpotch/bekko-system-one-dataset-v0`: 153 subsets, 6.59M cases, 8.42M judgments from NLP and
retrieval datasets plus synthetic sources; 16,517 updates, batch 512; 2.1 h / 5.6 h / 25.3 h training loop.

**Evals.** S1MB (author's benchmark, overlapping task families) Task Avg 27.6 / 40.5 / 50.6 vs Jev 1.13 59.6; on the
synthetic generalization view 19.0 / 32.5 / 54.5 vs Jev 96.3. **DI 0.3.2 Full 2.93 / 8.42 / 15.24** (raw 29.8 / 33.9 /
40.8). GPU: compiled SDPA on RTX 5090 1.38 / 2.69 / 5.03 ms per 4-option choice; DI GPU median 6.8 / 11.8 / 17.1 ms.
CPU uses fp32 SDPA, no number published. Browser ONNX 29 MB (17M) / 196 MB (68M).

**Novel.** The closest encoder analogue of Jev's reconstructed shared-prefix mask (one prefix, isolated branches),
here applied to options instead of questions. Largest training set in the group; its 30 MB 17M browser model is the
smallest pretrained-text model with all three types.

## MayaZ (acl-research): 17M / 32M / 68M

**Identity.** <https://huggingface.co/acl-research/MayaZ-68M> (+ `-17M`, `-32M`), all created and modified
2026-10-13, 12–15 downloads each, MIT. Code <https://github.com/acl-research/MayaZ> (MIT, 1 star, read-only clone
`sources/mayaz` @ 295deb8). ONNX only (fp32, int8, int8+int8-embeddings).

**Backbone.** `jhu-clsp/ettin-encoder-{17m,32m,68m}`: 7 L/d 256, 10 L/d 384, 19 L/d 512; global every 3, local 128,
θ 160k (`files/config.json` `enc_cfg`). 17,684,480 / 33,877,248 / 71,688,192 params incl. head. 17M is distilled from
68M.

**Input layout.** `[CLS] state [SEP] question [SEP] [OPT] opt1 [/OPT] … [OPT] optK [/OPT]`, markers are
`[unused0]/[unused1]` (`src/mayaz/model/render.py:5,24`). **Parallel position ids**: every option restarts at the
same position after the shared prefix, so no option is first (`render.py:7`). max_positions 512, 1,536 tokens per
pass, option cap 48 tokens, question cap 64.

**Isolation.** Segment-based mask: option k sees the shared tokens and itself; shared tokens see everything
(`render.py:8-9`), so the state can still absorb all options. One question per pass.

**Readout.** Head lives inside the ONNX graph; its inputs are `opt_start`, `opt_end`, `q_mask`
(`src/mayaz/runtime.py:15`), so it scores each option from its `[OPT]`/`[/OPT]` states with a pooled question
(inferred; head code not published). Lists >64 options go through a tournament (`model/tournament.py`), up to 256.

**Question types.** choice 2–256, noul (keys true/false), score (keys "0".."n-1"); `/v1/systemone` server with an
`abstain` flag.

**Calibration.** Per (type × option-count bucket 2, 3–5, 6–20, 21+) temperatures (`temperatures.json`); 68M ECE
0.029 overall, 0.076 in the 21+ bucket.

**Training data.** Soft labels from open-weight teachers over train splits of 13 permissive datasets
(typed-decisions, banking77, CLINC, MASSIVE-en, GoEmotions, DBpedia, CFPB, SNLI, Amazon reviews, BoolQ, Aegis 2.0,
prompt-injection, jailbreak).

**Evals.** typed-decisions 0.740 / 0.732 / 0.764; held-out 0.781 / 0.781 / 0.813; argmax flip under shuffle 0.01%.
CPU at a **0.1 vCPU quota**, 1 ORT thread: p50 0.60 / 1.50 / 6.35 s, peak RSS 367 MB (68M); browser WASM 1 thread
~0.1 / 0.3 / 1.0 s. A full core should be roughly 10× faster than the quota figures (inferred). Not on DI 0.3.2.

**Novel.** Laya's opt-in "parallel" layout actually shipped (Laya itself never released one), with near-zero order
flips at 17M params and a deployment measured under a tiny CPU quota.

## Crivo (tardellirs): 17M / 32M / 68M / 150M / 400M / 1B, focus Crivo-150M

**Identity.** <https://huggingface.co/tardellirs/crivo-150m>, created 2026-10-10, modified 2026-10-18 (v2;
v1 at revision `7a6c745`), 267 downloads, 13 likes, **CC BY-NC 4.0** (non-commercial training sources). Siblings:
`crivo-32m` (10-10), `crivo-400m` (10-13), `crivo-17m`/`-68m`/`-1b` (10-17/18). No GitHub; code in the repo
(`model.py`, `crivo.py`, `decision_index_engine.py`).

**Backbone.** `cross-encoder/ettin-reranker-150m-v1`: ModernBERT, 22 layers, d 768, FFN 1152, 12 heads, alternating
full/sliding layers, `max_position_embeddings 7999` (`files/config.json`). 149,610,241 params. Siblings: 16.9M, 32.0M,
68.4M, 395.8M, 1,031.3M (ettin-reranker-1b).

**Input layout.** `[CLS] instructions [SEP] state [SEP] [OPT] option_1 [OPT] option_2 …`, options `key: description`,
noul as `false`/`true`, score as `0: …`, `1: …`; `[OPT]` embedding initialized from `[SEP]` (`model.py:1-3,114-127`).

**Isolation.** One sequence per question; options see each other (full bidirectional); option order shuffled in
training.

**Readout.** `[OPT]` hidden + question-type embedding → LN → Linear → GELU → Linear (fp32) → softmax over the
question's options (`model.py:93-111`). No extra transformer layers (simpler than Laya).

**Question types.** choice, noul, score; runs unmodified in the DI kit.

**Calibration.** **No temperature fitted** ("treat probabilities as scores", README limitations). Loss
α·(CE + 0.5·Brier) + (1−α)·τ²·KL(teacher‖student), α 0.5.

**Training data.** Three stages: 638k questions (Bekko dataset ≤1,200/subset, code-generated exact-label families,
328 tasksource families, Mapika/decider teacher labels, …), 169k tool-selection / CLadder / smart-home, 24.6k
long-context up to 8,192 tokens. Distilled from Crivo-400M; contamination check against all 563,686 DI 0.3 questions
(0 matches). One A100 40 GB, ~5 h.

**Evals.** **DI 0.3.2 Full 22.21** for v1 (raw 52.45, acc 0.579, ECE 0.144, GPU median 13 ms), the best Full score
in this group and above vLLM-SR d3-edge 0.6B (20.24), Kev 0.8B r15 (13.66) and Intern-Decision-0.8B (11.78). Crivo-32M
v1 Full 11.87 (ECE 0.087, 7.3 ms). Self-run public index v2: 17M 22.15, 32M 27.34, 68M 32.82, 150M 34.38 (v1 36.39),
400M 39.79, 1B 47.77. A100 bf16 median 18 ms per request; no CPU figure. Reads up to 32,768 tokens without
truncation (beyond 8,192 untrained).

**Novel.** No architecture novelty (it is the simplest marker head here); the gain comes from data breadth (tools,
long context, code-generated exact-label families), distillation from a larger sibling, and a contamination firewall.
Strongest evidence so far that a 150M encoder can beat 0.6–0.8B decoder clones on the DI.

## Blink (sqliteai)

**Identity.** <https://github.com/sqliteai/blink>, Apache-2.0, 23 stars, created 2026-09-22, pushed 2026-09-23
(read-only clone `sources/blink` @ 18d6ce8). **No HF repo** (no `sqliteai` models on the Hub); the weights are
committed in the repo as `artifacts/blink-tiny-*.blink` (9 containers, 468–503 KB: synthetic s7/s17/s27, wanli
s7/s17/s27, snake, doom, pilot). blink-small (7.9M) is experimental and not shipped (README "Performance").

**Backbone.** None: trained from scratch, **byte-level, no tokenizer**. Preset tiny (`python/blink_train/config.py:92-105`): width 64,
blocks 2, ffn 128, rank 64, 2 heads, stride 4, 1 mixer block, 4,096 bigram buckets; 407k params, int8 weights
(452 KiB) mmapped read-only. Runtime C99 + libm, zero allocation while scoring; 66 KB wasm build.

**Input layout.** State, question and each option are separate byte streams: unigram + hashed bigram + position +
segment embedding → depthwise-conv stem → mean-pool by stride → conv/global-mean/FFN blocks + self-attention mixer
(README diagram). Limits (tiny): state 256 B, question 128 B, option 48 B; ≤64 options (`include/blink.h:55`).

**Isolation.** State and question encoded separately (state cache reuse is bit-exact, `memcmp`-tested); question
cross-attends to cached state K/V; options never attend to each other; each option logit is independent and only the
softmax couples them, so permuting options permutes probabilities exactly (`docs/ARCHITECTURE.md` §2.2).

**Readout.** Question summary FiLM-modulates each option vector; multi-head **cosine** option attention over the
context, Σ head cosines × scale / temperature → softmax; logits bounded by heads × |scale| (AGENTS.md invariants).

**Question types.** Choice-style option sets only (2–64); no native noul/score type (a noul would be a 2-option
menu, inferred).

**Calibration.** CE (proper scoring rule) + one temperature fitted on validation NLL, bounded [0.05, 20],
`temperature_pinned` treated as a failed run. Test ECE 0.003–0.03 on its own corpora.

**Training data.** Deterministic synthetic corpus (four families: routing, judgment, comparison, selection), WANLI,
Wikispeedia, and per-demo corpora (Snake 47.6k rows, Doom 75.8k, pilot 63.5k).

**Evals.** Synthetic: routing 0.99, judgment 0.95, comparison/selection at chance; WANLI (SemIf's 256 rows) 0.44 vs
0.64 for a frozen Qwen3.5-4B; external fixtures ~chance. Snake/Doom/driving demos near oracle (the planner precomputes
outcomes). **CPU: 406 µs fresh, 54 µs with cached state, one M5 Pro core** (172 µs Accelerate, 221 µs W8A8; 760 µs /
90 µs AVX2 on a GitHub runner); wasm ~0.8 ms. Not on DI.

**Novel.** The extreme end of the deployment axis: microsecond latency, 2.5 MiB peak RSS, no allocation, bit-exact
state reuse, embeddable C and wasm. It is not a language model and says so: it learns the form of a decision, not
reading. The `state_set` / `menu_set` / `score` split is the only exact, cache-by-construction version of Jev's
"state ingested once" in this group.

---

## Small-model deployment: versus Laya and Zeiger

**Encoder vs decoder.** All nine families are bidirectional encoders (or, for Blink, a from-scratch byte encoder).
Laya is the template most copy (Julia and Phocinae copy its head almost verbatim; Verdict 2.0 and Crivo simplify it).
Zeiger is the opposite choice: a causal 0.6B decoder used as an encoder, where option *i* cannot see options after it
and a separate 2-layer set transformer restores cross-option comparison.

**How options are marked.**
- *Marker token per option, joint listwise* (Laya `[MASK]`, Julia, Phocinae, Verdict 2.0, Crivo `[OPT]`, Kotoba
  `[OPT]` + span pool): cheapest, lets options compare, but position-sensitive (Laya's own five-identical-options test)
  and needs shuffling in training.
- *Masked-apart branches* (Bekko shared-prefix, MayaZ parallel positions): keep one pass per question but remove the
  first-slot advantage; MayaZ reports 0.01% argmax flips.
- *Independent options* (Decima late interaction, Blink): order-invariant by construction and cacheable per option
  set, scale to hundreds or thousands of options, but options cannot reference each other.
- Zeiger's marker-after-option with a causal backbone sits in the first group but with a learned set transformer on
  top; Kev/Jev reconstructions use the same pointer idea at decoder scale.

**Context length.** Shipped budgets are small: 512 (Laya, Phocinae, Verdict, Kotoba, Decima-small, MayaZ), 1024
(Julia benchmarks). The long-context exceptions are Crivo (trained to 8,192, reads 32k), Bekko (7,999 native),
Decima-agent (8,192), Julia (8,192 runtime, untested accuracy). Zeiger's chunked prefix (2k chunks, trained to 32k)
is still the only one built for whole web pages with 1,500 options. Decima's DI collapse (requests exceeding its
512/64 budget count as wrong) shows that budget, not capacity, is what kills small models on mixed suites.

**CPU latency (as reported; hardware differs, so order of magnitude only).** Blink 0.4 ms ≪ Decima-small ~16–20 ms
(1 core int8) < Decima-base 21–47 ms < Zeiger 340 ms (16 cores bf16, 10 options) ≈ Julia ~300 ms/question (4 threads,
1024 tok) ≈ Phocinae ~330 ms/decision (1 thread) < Kotoba ~450 ms/question (M1 Max fp32). MayaZ figures are at a
0.1 vCPU quota (0.6–6.4 s). On GPU the encoders converge at 5–18 ms (DI medians), Laya 5.8 ms. For us the practical
rule: int8 ONNX encoders at ≤150M give interactive CPU latency; the Laya head over the full sequence (two extra
transformer layers on every token) is what makes Julia/Phocinae slow on CPU (inferred).

**Accuracy on a shared board (DI 0.3.2 Full).** Crivo-150M 22.21 > Bekko-400M 15.24 > Crivo-32M 11.87 > Bekko-68M
8.42 > Laya 4.43 > Bekko-17M 2.93 > Julia 2.56 > Verdict 0. For scale: Kev 0.8B 13.66, d3-edge 0.6B 20.24, Jev 1.14
61.53. Zeiger is not on the board. Training data breadth (Crivo, Bekko's 8.4M judgments) matters more than head design;
the Laya-head clones trained on typed-decisions alone (Julia, Phocinae, Verdict) score near zero.

**Implication for Zeiger-style deployment.** For CPU or browser use of a browser-grounding model, the useful
transfers are (1) Decima's independent, cacheable options for long element lists, (2) Bekko/MayaZ-style masks to get
order robustness without losing a single pass, (3) Crivo's recipe (Ettin reranker 150M, broad data, teacher
distillation) as the accuracy baseline to beat, and (4) Blink's exact state cache if per-step latency must reach
sub-millisecond. None of them handles 1,500 options in a long page natively; MayaZ (tournament to 256) and Decima
(linear in option count) come closest (inferred).
