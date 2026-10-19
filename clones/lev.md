# lev: two unrelated projects with the same name

Two independent projects share the name "lev". They are unrelated, so each has its own profile below.

- **Part A**, **interfaze-ai/lev** (Interfaze, Abhinav). A Qwen3.5-4B LoRA that reads label-token logits, with a candidate-path head as fallback. This is the one we run locally on :8000.
- **Part B**, **OpenLAIR/lev** (Song, Yan, Zhang, Sun; arXiv 2610.20951). Looped "System 1.5" decision models, forked from kev, on the Ouro and Nanbeige looped backbones.

Two more models with similar names are also unrelated:
- `franckverrot/lev-350m` is a kev clone on LiquidAI/LFM2.5-350M, licence `lfm1.0` (`HF:franckverrot/lev-350m/README.md`).
- `theaviv/Lev1-3.7B` is a LoRA (r32/α64) on IFM/K2-Horizon-3.7B with 42.28 on Decision Index 0.2.1 (`HF:theaviv/Lev1-3.7B/README.md`). It is listed as "Lev1 3.7B" in DI 0.3.2.

Compiled 2026-10-19. Inferences are marked *(inferred)*.

---

# Part A: interfaze-ai/lev

| ref | what | where |
|---|---|---|
| `G` = github.com/**Abhinavexists/lev** @`1b8da5ce` (2026-10-15) | Canonical source: created 2026-09-20, 128 commits by Abhinavexists plus 10 by hiroki-abe-58, ★56. Shallow clone, never executed. | `sources/abhinavexists-lev` |
| `GI` = github.com/**InterfazeAI/lev** @`79e1661a` (2026-10-15) | Org copy, **not a GitHub fork**. Created 2026-09-25, ★24, synced by PR ("sync/adapter-scale-and-docstring-pass"). Differs from `G` only in README table formatting, badges and one comment line in `contamination.py` (`diff -rq`). | `sources/interfazeai-lev` |
| `HF` = `interfaze-ai/lev` @`8721908e` (lastModified 2026-10-15) | API, tree, README, `lev_release.json`, `calibration.json`, `adapter_config.json`, chat template. No weights. | `notes/hf-raw/interfaze-ai__lev/` |
| `LOC` = our deployment | `/home/ai/lev-server`: compose, Dockerfile, `app/`, `shim/levshim.cpp`, a pinned clone of `G` @`cf104b69` (2026-09-25), HF snapshot `7bdc748d` | read only |
| `B` = Interfaze blog | [interfaze.ai/blog/jev-now-open-source-lev](https://interfaze.ai/blog/jev-now-open-source-lev). Describes the first release (68.9%). | WebFetch 2026-10-19 |

Relation between the repos: the Decision Index `code_url` (`github.com/Abhinavexists/lev`) and the HF card badge (`HF:README.md:27`) both point to Abhinavexists. Abhinavexists is the author's personal repo, and InterfazeAI/lev is a mirror that Interfaze publishes. **Cite `G`.**

## Identity

| | |
|---|---|
| Maker | Interfaze (interfaze.ai), built by GitHub user Abhinavexists. Contributor hiroki-abe-58 also commits to both repos. "Not affiliated with or endorsed by TypeSafe AI" (`HF:README.md:244`). |
| Weights | Only `interfaze-ai/lev`: a 170 MB LoRA, the 15 MB `mode_b_head.pt`, the tokenizer and `calibration.json` (`HF tree.json`). Created 2026-09-24. 1,150 downloads, 127 likes. |
| Versions | **r1** (HF `7bdc748d`, ~2026-09-24): adapter at full strength, S1Bench macro 0.689 / ECE 0.115 (`B`; `G:docs/FINDINGS.md` §19 "control"). **r2** (HF `8721908e`, 2026-10-15): same adapter weights, plus `"adapter_scale": 0.25` and re-fitted temperatures. Macro 0.743 / ECE 0.097 (`HF:lev_release.json`, `G:docs/FINDINGS.md` §19). |
| Decision Index 0.3.2 | `lev`, Interfaze, "LoRA + head", technique autoregressive, 4.66B, base `Qwen/Qwen3.5-4B-Base`, code `Abhinavexists/lev` (`notes/decision_index_0.3.2_entries.tsv`). **The base listed there is wrong.** The release manifest names the post-trained `Qwen/Qwen3.5-4B` (`HF:lev_release.json` `base_model`, preset `4b-instruct`). |
| Lineage | A design survey of S1Bench entries (`G:docs/ARCHITECTURE.md` §1–4). It takes Family A, label-token readout (simple-jev, decider, reflex), and Family B, candidate-path scoring (NanoJev, jeff), and composes the two. kev appears only as a leaderboard row; no code is shared with kev. |

## Architecture

**Backbone.** `Qwen/Qwen3.5-4B`, the instruct model. It has 32 layers (24 Gated-DeltaNet + 8 full attention), d = 2560, 16 q-heads and 4 KV heads, 262k positions, and a 248,320 vocabulary (`G:docs/ARCHITECTURE.md:139-146`). LoRA settings:
- r = 32, α = 64, dropout 0.05 on q/k/v/o/gate/up/down;
- `task_type CAUSAL_LM`, peft 0.21 (`HF:adapter_config.json`);
- **the LM head is kept**, because Mode A reads vocabulary logits (`G:packages/lev/src/lev/model.py:413-418`).
- In r2 the adapter is applied at `adapter_scale` 0.25 via `LoraLayer.set_scale` (`model.py:553,581-583,619-627`). This is a WiSE-FT-style interpolation back toward the frozen backbone (`FINDINGS.md` §19).

**Input layout.** The prompt is plain text in the backbone's own chat template (`prompt_style: chat`, ADR-027).
- **System prompt**: "You are a System One decision model … You answer with the single option label only" (`prompt.py:31-35`).
- **Prefix** = system + `# Evidence\n<state>`. The state is a string, or JSON with `sort_keys` so the prefix stays byte-stable (`prompt.py:58-62,192-201`).
- **Suffix** per question = `# Criterion`, then lettered options `A. key: desc` or levels `A. (level i of n) desc`, then an ask line (for example "Respond with only the letter of the best option."), then `<|im_end|><|im_start|>assistant\n<think>\n\n</think>\n\n` (`prompt.py:37-43,113-161`).
- The **scored position is the last suffix token**, right after an empty think block.
- A schema-first layout (questions before the state) is trained 50/50 but never served (`ARCHITECTURE.md:218-221`).

**Attention and question isolation.** **Every question is its own row**, prefix + suffix, with a normal causal mask and no packed block mask. Isolation therefore holds by construction. There are two forward modes:
- **Default `single`**: one batched forward over all rows, recomputing the prefix per row. ADR-023 found this faster (169 → 69 ms) because at 4B the forward is launch-bound (`model.py:348-381`, `HF:README.md:196`).
- **`fork`**: prefill once, deepcopy the KV + DeltaNet state per row (`model.py:383-411,651-656`).

Choice and binary noul are also read in **two option orders (shown and reversed) and averaged**, so each question costs 2 rows. Score is not averaged by default (`ARCHITECTURE.md:204`, `model.py:51-53,288-304`). The card states that questions are independent (`HF:README.md:211`).

**Readout head.** A router chooses the readout per question (`router.py:175-204`):
- **Mode A, label-token** (0 added parameters). Options get codes A..Z, AA.., AAA..; **codes that tokenize to 2 tokens are skipped**, so several hundred options fit (`labels.py:30-56`). Logits at the scored position, restricted to the code ids, are passed through a softmax (`model.py:426-435`). Training caps Mode A at 26 options; serving routes up to the tokenizer limit (ADR-025/026; "68 → several hundred" after ADR-028).
- **Mode B, candidate path** (`readout/mode_b.py`, 3.67M params). Each candidate text (≤ 64 tokens) is encoded **separately** through the backbone, and its last-token state is cached per candidate set (`model.py:454-481`). The question representation is the last-token hidden state. q and candidates are projected 2560 → 512, candidates pass through **set attention across candidates** (MHA, 8 heads), then `+ q`, LayerNorm and a linear score (`mode_b.py:12-38`). Mode B is used when codes run out. MASSIVE-60 fell from 0.746 (Mode A) to 0.231 on Mode B, so Mode B is a last resort (ADR-025).

**Question types.**

| type | Mode A readout | answer | confidence |
|---|---|---|---|
| noul (`noul_readout: rating`) | 9 rating tokens "0".."8" ("Respond with only a digit from 0 to 8") | `noul = Σ (i/8)·p_i`; also returns `probabilities`, unlike Jev (`labels.py:57-63`, `model.py:503-511`) | Gini `(KΣp² − 1)/(K − 1)` |
| noul (binary option) | yes/no codes | `p[yes]`, no distribution | Gini |
| choice | lettered options | argmax + probabilities | Gini (`model.py:660-664`): **not** Jev's `(p_max − 1/n)/(1 − 1/n)` |
| score | lettered levels | `Σ i·p_i`, legend | Gini |

**Limits.**
- Training sequences are ≤ 2,048 tokens (`train/config.py:50`); the measured mean is 128 tokens (`ARCHITECTURE.md:260`).
- Serving uses a 16,384-token batch budget. An oversized request runs alone; past 64 pending requests the server returns 503, and after 30 s in the queue it returns 504 (`HF:README.md:128`, `batcher.py:49-55`).
- No validated long-context number is given. The card says "English only" (`HF:README.md:212`).

**Calibration.** Post-hoc temperatures per bucket (type × mode × option band for choice), applied at load (`calibrate.py`, `HF:calibration.json`). Each bucket keeps whichever of a row-weighted or family-weighted fit "transfers better to held-out task families" (ADR-028).

| bucket | r1 T | r2 T |
|---|---|---|
| noul:A | 2.33 | 1.28 |
| choice:A | 1.77 | 1.46 |
| score:A | 2.80 | 1.87 |
| choice:B | 0.63 | 0.58 |

- Both sets were fitted on the calibration split, about 10–14k rows per bucket (`HF:calibration.json`; r1 values from `LOC:hf/.../7bdc748d/calibration.json`).
- Training also uses a proper scoring loss and abstain augmentation (see Training).

## Training

- **Data**: 200,000 examples from 29 sources over 26 public HF datasets: intent (banking77, clinc_oos, snips), NLI (SNLI, ANLI, FEVER), paraphrase with word-swap negatives, MCQA (RACE, ARC, SciQ, OBQA, CSQA, StrategyQA), toxicity and safety (ToxiGen, ToxicChat, BeaverTails), and UltraFeedback.
- **Augmentation**: questions are paraphrased, negated and recast between types; option sets are shuffled and resized (`HF:README.md:203`). 10% of examples are abstain rows with soft targets (`train/config.py:53`).
- **Contamination guard**: the build refuses any source that resolves to one of the 13 S1Bench subsets (`HF:README.md:204`, `data/contamination.py`).
- **Loss**: `CE + 0.5·Brier + 0.25·ordinal`, where the ordinal term is the expected squared level distance and applies to score and noul rows in both modes (`train/loop.py:84-119`, `docs/TRAINING.md:185-193`). The Mode B head is trained jointly (`train_mode_b_head=True`).
- **Recipe**: preset `4b-instruct`, lr 5e-5, 3 epochs, 18,750 steps at batch 32, one H100, 7.8 h (`HF:README.md:205`, `HF:lev_release.json` step 18750). Development history: 3 checkpoints at 0.489 → 0.697 → 0.725 (six-subset definitions) (`docs/STATUS.md`).
- **RL** with a belief reward is planned, not done (`ARCHITECTURE.md:280`).

## Evals

- **S1Bench**, all 13 subsets, 3,880 pinned items, same harness for lev and Jev (Jev reproduced within 0.8 pp of its published numbers):
  - r2: **0.743 vs Jev 0.761**; r1: 0.689. Mean ECE r2 0.097 vs Jev 0.091.
  - lev is ahead on multinli, aegis2, helpsteer2 and summeval-relevance, all within noise. It is behind on vitaminc (−7.5) and paws (−5.6) (`HF:README.md:136-158`).
  - The scale 0.25 was **chosen on the same subsets**. The leave-one-family-out estimate is **0.728** (`FINDINGS.md` §19). summeval-consistency moved 0.271 → 0.806 with the scale change.
- **Held-out split** (r1, before ADR-028): weighted acc 0.807, ECE 0.061 (0.180 raw), FEVER 0.872. After ADR-028: banking77 0.980, clinc_oos 0.968 (`HF:README.md:166-176`).
- **Blog (r1)**: 0.689 vs Jev 0.761; on the six subsets every model completed, 0.719, level with reflex-4b (`B`).

## Serving

- `lev serve` is a FastAPI `/v1/systemone` server, TypeSafe-SDK compatible. It batches across requests and returns `output_tokens` 0 (`HF:README.md:95-128`, `server.py`, `batcher.py`).
- **Speed**: 69 ms engine compute on an H100 for a short state. End to end from a laptop: 414–654 ms on Modal, vs Jev 335–346 ms (`HF:README.md:180-184`).
- **Our :8000 deployment (`LOC`)**: compose service `lev`, image `lev-cpu:gguf`.
  - Upstream `lev` code at `G@cf104b69` does the routing, prompts, readout and calibration. Only the forward pass is replaced: `GGUFEngine(DecisionEngine)` calls llama.cpp `d8fbd25` (Vulkan, Radeon 8060S) through `levshim.cpp` (`LOC:app/levgguf.py`, `LOC:Dockerfile`).
  - The shim prefills the state into seq 0, `llama_memory_seq_cp` forks it into up to 16 sibling sequences, and decodes the question suffixes with `logits=1` **only at each suffix's last token**. It returns full-vocabulary logits there (`LOC:shim/levshim.cpp`, `lev_eval`). This is `prefix_mode="fork"`, not upstream's default `single`. A repeated state skips the prefill.
  - The Mode B head runs in torch on the shim's `lev_encode` hidden states.
  - Settings: `LEV_CTX` 16384, `LEV_SEQ` 17, `mem_limit` 8g, `cpuset` 0-3.
  - **Status 2026-10-19: container `lev` exited cleanly (code 0) on 2026-10-12T07:14Z** (docker logs end with a uvicorn "Shutting down"; probably GPU hand-off, *(inferred)*). Nothing listens on :8000; another container's internal 8000/tcp is not published. It was not restarted, so no live requests were made.

### Verdict: is our :8000 lev a decision model or a generative LM?

**It is a decision model built on a causal LM, run prefill-only.** The weights in the GGUF are a generative LM: Qwen3.5-4B with the LoRA merged in (`LOC:convert/merge_lora.py`), LM head included. llama.cpp, however, never samples or decodes. It is linked as a library (`-DLLAMA_BUILD_SERVER=OFF`, so there is no `llama-server` and no `/v1/models`, `/props` or `/completion` endpoint), and only the logits of the scored position are requested.
- Mode A softmaxes over the verified label-code token ids.
- Mode B uses the separate head.
- `output_tokens` is 0.

This matches the card's claim ("reads each answer from the logits it already computed", `HF:README.md:25`). Unlike kev, it has **no separate readout head for most questions**: the "head" is the LM head restricted to label tokens.

**Our deployment does not match current HF in three ways:**
1. It was built from **r1** (`7bdc748d`). `merge_lora.py` merges at `scale = α/r = 2.0`, i.e. full strength. It has no `adapter_scale`, while r2 runs at 0.25 × that. Merged weights cannot be rescaled at serve time.
2. It uses **r1's temperatures** (e.g. noul T 2.33 vs 1.28). These are consistent with full strength, so the deployment is internally coherent.
3. The code is `G@cf104b69`, which predates the skip-split-codes/ADR-028 and the cross-request batcher (*(inferred)* from dates).

Expected quality is therefore about S1Bench 0.689, not 0.743. Moving to r2 would need a re-merge at effective scale 0.5 (= 0.25 × α/r) and the r2 `calibration.json` *(inferred)*.

## Licence

- Code and adapter are Apache-2.0 (`G:LICENSE`, `HF cardData`). The base Qwen3.5-4B is Apache-2.0.
- The card warns that some training datasets carry non-commercial terms (`HF:README.md:242`).

## What is novel vs Jev and kev

1. **Dual readout with a router.** Free label-token readout from the LM head (no new parameters; works untrained) up to the tokenizer's single-token-code limit, then a set-attention candidate head above it. kev uses one pointer head for everything.
2. **Order averaging as a fix for position bias.** Choice is read in two orders and averaged at 2× rows. kev accepts order sensitivity; Jev leans toward the first option.
3. **Noul as a 9-level rating distribution** with `probabilities`, so it can be calibrated like the other types. Jev returns a bare float.
4. **Adapter scale as an out-of-distribution knob.** Shrinking the LoRA to 0.25 at serve time gave +5.5 macro (+3.9 held-out estimate) with no retraining. kev used WiSE-FT only for the 27B weight average.
5. **Per-bucket temperatures selected for cross-family transfer**, not in-distribution fit. This is a similar lesson to kev U ("temperature fitted in-domain does not transfer").
6. **Weakness**: confidence uses a Gini statistic rather than Jev's formula. It is a 4B instruct base with a ~2k training context and no long-state validation.

---

# Part B: OpenLAIR/lev (looped "System 1.5")

| ref | what | where |
|---|---|---|
| `O` = github.com/**OpenLAIR/lev** @`fd584e9b` (2026-10-19) | Created 2026-10-14, ★3, Apache-2.0. Shallow clone, never executed. Python package still imports as `kev`. | `sources/openlair-lev` |
| `HF1/HF2/HF3` = `songdj/Lev-1.4B` @`89cf8956`, `songdj/Lev-2.6B` @`9d85df95`, `songdj/Lev-3B` @`ae4cef0c` (all created 2026-10-14) | API, tree, README, manifest, provenance, temperatures, validation, adapter config | `notes/hf-raw/songdj__Lev-*/` |
| `P` = arXiv [2610.20951](https://arxiv.org/abs/2610.20951) v1 (2026-10-14) | "LEV: Thinking Fast and Deep with Looped Decision Models", Dingjie Song, Zhiling Yan, Yuxuan Zhang, Lichao Sun | abstract via WebFetch; PDF not read |

## Identity

| | |
|---|---|
| Makers | OpenLAIR (HF user songdj = Dingjie Song). The Lichao Sun co-authorship suggests an academic lab *(inferred)*. |
| Lineage | **A fork of jaredpalmer/kev**: "derived from Jared Palmer's Kev" (`O:README.md:162`, `O:NOTICE`). It adds native recurrent backbones, shared heads across depths, per-depth losses, depth readouts and an offline stopper. |
| Models | Lev-1.4B (Ouro-1.4B), Lev-2.6B (Ouro-2.6B), Lev-3B (Nanbeige4.2-3B-Base). Seed-0 only. 45–52 downloads each. Datasets: `songdj/LevBench`, `songdj/Lev-train-v2`, `songdj/Lev-results` (`O:README.md:47-63`). |
| Status | Research release. "Full-size GPU parity has not been re-run on the released packages"; `validation.json` shows `gpu_parity: pending`, `model_inference_run: false` (`HF2:validation.json`). |
| Decision Index 0.3.2 | **No entry.** |

## Architecture

**Backbone.** Natively **looped** (weight-shared, recurrent-depth) pretrained LMs:

| model | backbone @revision | stack × loops | d | context (base) |
|---|---|---|---|---|
| Lev-1.4B | ByteDance/Ouro-1.4B @`574fa66c` | 24 layers × 4 steps | 2048 | – |
| Lev-2.6B | ByteDance/Ouro-2.6B @`1ed04250` | 48 layers × 4 (`total_ut_steps` 4) | 2048 | 65,536 |
| Lev-3B | Nanbeige/Nanbeige4.2-3B-Base @`4a38e817` | 22 layers × 2 (`num_loops` 2) | 3072 | 131,072 |

(Revisions are from `HF*/adapter_config.json`. Base configs were fetched 2026-10-19. The 1.4B layer count comes from the `kev/ouro.py` docstring.)

- Ouro is **reimplemented** in `kev/ouro.py` (sandwich norm, no KV cache, `step_hidden_states`), because its remote code does not load under transformers 5. It is checked against the original by `tests/test_ouro.py`. The LM head and the exit gate are dropped (`O:kev/ouro.py:1-17`).
- Nanbeige is run as a plain pre-norm loop. Configs that enable mHC, depth-attention or loop-shared KV are refused (`O:kev/nanbeige.py:1-40`).
- LoRA: r16, α32, dropout 0.05 on all projections, `FEATURE_EXTRACTION` (kev defaults).
- `kev/loop.py` additionally offers a **retrofit loop** for non-looped bases: a repeated layer block with zero-initialised ReZero gates and input injection, plus an unshared control. It is used as an ablation; the released checkpoints use native loops (`O:kev/loop.py:1-40`).

**Input layout.** kev's packed layout: `<state>` S, then per question `<q> I (<opt> o </opt>)* <decide>`.
- Delimiters are **reused special tokens** per tokenizer. Ouro uses `<filename>`, `<issue_start>`, `<jupyter_code>`, `<jupyter_output>`, `<empty_output>`. Nanbeige uses `<s>`, `<|im_start|>`, `<tool_call>`, `</tool_call>`, `<|im_end|>`, whose trained embedding rows (ids 166100+) are re-added to the Base tokenizer.
- These tokens are sanitised in user text (`O:kev/ouro.py:31-33`, `O:kev/nanbeige.py:22-28`).

**Attention and question isolation.** The backbones are attention-only, so the release uses kev's **single packed pass under a block-causal mask** (state visible to all, branches mutually masked, branch positions restarting after the state). There is **no KV cache and no prefix reuse**: `prefix_min_tokens = 10**9` (`O:kev/ouro.py:14-17`, `O:kev/model.py:396-397,445`). Each loop step re-applies the whole stack to the whole packed sequence with the same mask.

**Readout head.** kev's `PointerHead`: `z_j = k(h[</opt>_j]) · q(h[<decide>]) / √256` (`O:kev/model.py:247-266`; `head.pt` q/k are 256×2048 for 2.6B per `HF2:validation.json`).
- **One shared head is read at every loop step**, so one forward gives a distribution per depth (`forward_steps_batch`).
- Choosing depth d at inference computes only d steps (`O:kev/model.py:383-397,509-525`).
- An optional parameter-free `LetterHead` (LM-head letter logits) exists as an alternative readout (`O:kev/model.py:268-290`).

**Question types.** kev's API mapping (`kev/api.py`): noul → `[no, yes]`, choice → keys, score → levels, with the expected level and closed-form confidences. **No HTTP server ships**: kev's `serve.py` was removed ("remove optional upstream entry points", `O:NOTICE`). Only `examples/fixed_depth.py` is provided.

**Limits.** Max training state 5,120 tokens (`O:model-cards/*.md:21`). Option and question limits are kev's (≤ 255 options) *(inferred from shared code)*.

**Calibration.** **Per-depth temperatures** in a separate `temperatures.json`, fitted on 1,148 questions of kev's decision-v7 calibration split. `head.pt` keeps `temperature=1.0`, and the file must be applied explicitly (`HF3:README.md:83`).
- Lev-2.6B T = [1.15, 1.18, 1.15, 1.15]; NLL 0.528 → 0.372 from depth 1 to 4.
- Lev-1.4B T ≈ 1.05–1.12.
- Lev-3B T = [1.00, 1.02] (`HF*/temperatures.json`).
- The values are near 1, so the models are already close to calibrated in distribution.

## Training

- **S31 recipe**: the original kev-0.8B mixture (18,764 records) plus a decision-v7 replay (12,576 records). For Ouro-2.6B, 31,329 records remain after the context filter.
- Settings: LoRA r16, lr 1e-4, 1 epoch, effective batch 64, bf16 autocast, length sorting, none-pair augmentation 0.25 (`O:model-cards/Lev-2.6B.md:21`, `O:docs/REPRODUCING.md:11`).
- **Loss**: CE **averaged uniformly over all loop steps** (`--step_weights`), so every depth is a usable decision model. Optional self-distillation pulls earlier steps toward the last step's distribution (`O:kev/train.py:356-392`).
- Lev-train-v2 also packages S60/S120 mixtures and replay and order recipes (`O:README.md:62`).

## Evals

**JevBench public** (231 tasks, fp32), accuracy per depth (`O:model-cards/*.md`):

| model | d1 | d2 | d3 | d4 | hard tier d1 → deepest | H200 bf16 p50 per depth (ms) |
|---|---|---|---|---|---|---|
| Lev-1.4B | 58.0 | 65.4 | 68.0 | 68.0 | 31.5 → 44.1 | 15 / 30 / 44 / 58 |
| Lev-2.6B | 63.2 | 71.0 | 75.3 | **76.2** | 39.6 → 56.8 | 51 / 101 / 150 / 200 |
| Lev-3B | 69.3 | 72.7 | – | – | 46.8 → 49.5 | 23 / 45 |

- Depth helps hard items most, but "can also overturn a correct earlier decision".
- **Locked LevBench**: 2.6B goes 47.4 → 64.4% from depth 1 to 4. **Lev-2.6B failed gate G3** (a positive lower confidence bound vs the fixed-depth frontier) in all 3 seeds. Lev-1.4B and Lev-3B passed G3 with one seed each (`O:README.md:45,121-126`).
- **Paper**: an agreement-based stopper (threshold fitted on unlabelled requests) saves 15–24% compute at under 1 point of accuracy loss. 1.4B/3B are +5.4 to +8.8 points over dense models at matched latency (`P` abstract). The stopper is **offline only**; online adaptive continuation is not implemented (`O:README.md:123`).
- No S1Bench, Decision Index or direct Jev/kev numbers are given in the cards.

## Serving

- There is no server. `examples/fixed_depth.py --depth d [--temperatures …]`, and `--all-depths` computes every step (no saving).
- Timing is batch-one bf16 on an H200 without fused kernels. Latency is roughly linear in depth because each step re-runs the full stack (table above).

## Licence

- Code is Apache-2.0, with kev's licence retained (`O:LICENSE`, `O:NOTICE`). HF cards declare apache-2.0.
- The repo's own cards leave the weight licence "deliberately unset" and quote Ouro's "intended for research purposes only" note (`O:model-cards/Lev-2.6B.md:47-49`).
- Dataset subsets are rights-filtered and the full corpus is not redistributable (`O:docs/LICENSING.md`).

## What is novel vs Jev and kev

1. **Depth as an inference-time compute knob ("System 1.5").** One checkpoint answers at 1–4 recurrent depths without decoding tokens, with one shared pointer head across steps. Jev and kev have fixed depth; the reasoning-style alternatives (Jeeves/CISPO, StartLux AutoThink) spend decoded tokens instead.
2. **Natively looped pretrained backbones** (Ouro, Nanbeige4.2) used as decision models, plus a retrofit-loop ablation with ReZero gates and input injection.
3. **Per-depth calibration and a label-free agreement stopper.** The stopper is a research result (offline, workload-cross-fitted), not a serving feature.
4. **Inherits kev's mechanics unchanged**: block-causal packed isolation, pointer head, decision-v7 data. Losing the KV cache means no state reuse across questions or requests. That is a regression versus kev's Kev 1.0 prefix cache and Jev's "pay for state once" economics *(inferred)*.
5. **Honest negative result**: the headline 2.6B model fails its own preregistered gate on the locked test.
