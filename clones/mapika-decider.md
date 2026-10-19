# decider (Mapika / Mark Marosi)

Open "System One" reproduction. A Qwen3.5 (later Gemma 4) causal LM, **fully fine-tuned or LoRA-merged, no new head**,
reads the **LM-head rows of single-token option letters** (`A..J`, then `K..Z` and two-letter codes up to 255) at an
`Answer: (` slot. One causal row per question, state prefix computed once and its cache forked. Serves TypeSafe's
`POST /v1/systemone` wire format. Labels are public datasets, programmatic generators and a local Qwen3.5-27B /
Qwen3.6-27B teacher; **nothing distilled from Jev**.

Source pins used for this profile:

| ref | what | where |
|---|---|---|
| `G` = GitHub `Mapika/decider` @ **`d6fc522a`** (2026-10-12 12:04 +0200, "README: decider-12b v2 Noul temperature trade-off…") | shallow clone (`--depth 1 --filter=blob:limit=2m`), read only, never executed; package `decider-ai` 1.9.0 (`G:pyproject.toml:6-7`) | `/home/ai/projects/jev-clones/sources/decider` |
| HF | API JSON (`?expand[]=safetensors,…`), recursive tree, every small text file (README, `config.json`, `decider_config.json`, `eval_results.json`, chat template, the vendored `decider/*.py` some repos carry), no weights, for the 14 `Mapika/decider*` repos | `notes/hf-raw/Mapika__*/`; author listing `notes/hf-raw/_authors/Mapika_models_2026-10-19.json` |
| index | Decision Index 0.3.2 entry list (names/params only, no scores) | `notes/decision_index_0.3.2_entries.tsv` |

Citation form: `G:decider/prompt.py:188` is file:line at `d6fc522a`. `HF:decider-2b/decider_config.json` is a fetched
file under `notes/hf-raw/Mapika__decider-2b/files/`.

> **Main findings.**
> 1. **Readout = the backbone's own LM head, restricted.** `DecisionModel.slot_logits` takes the last hidden state at each
>    `Answer k: (` position and multiplies it by `lm_head.weight[letters]` (255 rows), masks options ≥ n to −inf and
>    softmaxes at a config temperature (`G:decider/model.py:23-32`). No pointer, no new parameters. On the tied-embedding
>    Qwen3.5 dense bases (0.8B/2B/4B) the readout rows **are** the input embeddings of the letter tokens.
> 2. **Isolation is by rows, not masks.** `system_one(independent=True)` (the default) renders one row per question
>    (`G:decider/infer.py:251`), runs the shared state prefix once and forks its KV + Gated-DeltaNet conv/recurrent cache
>    to every row (`G:decider/shared_prefix.py:1-25`). This is the same mechanism Kev 1.0 adopted for hybrid bases. The
>    packed one-row-many-slots form still exists (training data mixes both) and leaks question order: reversing it flips
>    up to 12% of answers (`G:docs/RESULTS.md:427-430`).
> 3. **Score is (by default) judged per level**: each level becomes its own yes/no row ("Proposed answer: … Does the
>    proposed answer fit?") and P(yes) is renormalised across levels (`G:decider/systemone.py:83-103`), a Jev-claimed
>    behaviour few other clones copy.
> 4. **The family drifted from "train a small model" to "read a big stock model".** decider-chat-gemma4-31b and
>    decider-chat-qwen3.6-27b are **unmodified** instruct checkpoints plus a temperature; they outscore the trained
>    2B/4B/35B on the Decision Index. decider-31b (Gemma-4-31B-it, merged fine-tune, NVFP4) reports **64.27** on the DI 0.3
>    public suite (own run) vs Jev 57.96.
> 5. **Most downloaded independent clone** (decider-2b 318,711 downloads; family ≈ 386k) and the most forked base: ≥ 20
>    derivatives (FLock this-that-model, NPU/LiteRT/ONNX/MLX ports, language fine-tunes).

---

## Identity

| | |
|---|---|
| Author | Mark Marosi (`Mapika` on GitHub and HF) (`G:README.md:440-448`, `G:pyproject.toml:13`). Independent, "not affiliated with or endorsed by TypeSafe AI" (`G:README.md:46-49`) |
| Code | github.com/Mapika/decider, Apache-2.0, ★1163 (`notes/families.json`), created 2026-09-16, last push 2026-10-12. PyPI `decider-ai` (import name `decider`) 1.0.2 → 1.9.0 (`G:docs/CHANGELOG.md:6-690`) |
| Lineage | HISTORY's stated goal: "reproduce the shape of TypeSafe AI's *Jev* … on a single GH200" (`G:docs/HISTORY.md:10-12`). v1-v5 predate Jev's public docs (2026-09-15); v6 is the explicit port to Jev's input shapes (`G:docs/HISTORY.md:158-174`). No reference to Archer Hume's reconstruction |
| HF repos (2026-10-19) | 14 decider repos + 2 unrelated NVFP4 quants (MiniMax-M3, GLM-5.2) (`_authors/Mapika_models_2026-10-19.json`) |

| repo (sha) | created | dl / ♥ | what | params (safetensors) |
|---|---|---|---|---|
| `decider-2b` (`ff99089b`) | 09-16 | **318,711** / 108 | **v11**; tags `v10`, `v8` | BF16 1,881,825,088 |
| `decider-4b` (`99d7fd5b`) | 09-22 | 29,745 / 29 | **v2.1**; tags `v2`, `v1` | BF16 4,205,751,296 |
| `decider-0.8b` (`a0a01d6f`) | 09-19 | 16,150 / 7 | 0.8b-v1, single `train.sh full` run | BF16 752,393,024 |
| `decider-2b-vision` (`863e2908`) | 09-16 | 4,591 / 34 | v5 text weights in the Qwen3.5-2B VLM | BF16 2,213,241,664 (incl. vision tower) |
| `decider-35b-a3b` (`91470e65`) | 09-20 | 2,658 / 9 | 35b-a3b-v1 | BF16 34,660,610,688 |
| `decider-35b-a3b-nvfp4` (`798555c0`) | 09-20 | 1,036 / 8 | ModelOpt NVFP4 | U8 16.8G + BF16 1.04G (packed) |
| `decider-12b` (`64d8d3f0`) | 09-29 | 713 / 4 | **v2**: Gemma-4-12B-it + merged LoRA r32; tag `v1` = stock weights | BF16 11,959,730,224 |
| `decider-31b` (`e12ddf20`) | 10-07 | 371 / 5 | Gemma-4-31B-it, merged fine-tune, NVFP4 MLP / bf16 attn / FP8 KV | U8 10.40G + BF16 10.46G (packed) |
| `decider-chat-gemma4-31b` (`22975cbd`) | 09-29 | 643 / 4 | **stock** Gemma-4-31B-it + config | BF16 31,273,088,876 |
| `decider-chat-qwen3.6-27b` (`f18c9385`) | 09-29 | 104 / 0 | **stock** Qwen3.6-27B + config | BF16 27,781,427,952 |
| `decider-2b-GGUF` (`ff2e5e68`), `decider-4b-GGUF` (`b79f09d9`) | 09-27 | 4,118 / 7,680 | Q4_K_M, Q8_0, BF16 of 2b v11 / 4b v2.1 | – |
| `decider-2b-coherent` (`516f5cad`) | 09-27 | 76 / 1 | 9.5M-param add-on (joint coupling + gated adapter) over frozen decider-2b | – |

(sha, dates, counts: `notes/hf-raw/Mapika__*/api.json` and the author listing; params: `api.json:safetensors.parameters`.)

**Derivatives** (`notes/hf_candidates.tsv`): `flock-io/this-that-model-1.1/1.2` (built from decider-2b, MIT; in the DI list
as "this-that 1.2", 1.88B), `Louiskq/jev-class-9b-sft` (decider's SFT recipe on Qwen3.5-9B-Base), `hayriyigit/decider-4b-tr`
(Turkish LoRA), `nativ-community/decider-2b` (MLX), `rovangju/decider-4b-W8A16/W4A16`, `seimanu/decider-*-NPU2` (Ryzen NPU),
`tielmane/decider-12b-NPU-LPBQ-X-Elite` (Snapdragon NPU, `G:README.md:198-203`), `litert-community/decider-0.8b/2b-vision-LiteRT`,
`ollaya-dev/decider` (ONNX), `hipfire-models/decider-4b`, `verdict-models/decider-2b`, `daviestechlabs/decider-2b-homelab-router`,
`IqSantiagow/edceleste-decider-0.8b`, `leafiy/mia-decider-2b` (Chinese sentiment). **No `ArthurZ/*` port** of any decider exists
(no `notes/hf-raw/ArthurZ__decider*`); the format would map onto HF's `jev` head with 255 code rows and no code
(`clones/hf-jev-model-type.md`, "What the abstraction fixes") (inferred).

---

## Architecture

### Backbone per checkpoint

| checkpoint | base | layers / mix | d | what is trained | source |
|---|---|---|---|---|---|
| decider-0.8b | `Qwen/Qwen3.5-0.8B-Base` | 24 = 18 DeltaNet + 6 attn | 1024 | **all weights**, one epoch of the full mixture | `HF:decider-0.8b/decider_config.json`, `notes/arch/arch_table.tsv` |
| decider-2b v1-v10 | `Qwen/Qwen3.5-2B-Base` | 24 = 18 + 6 | 2048 | **all weights**, staged continuation epochs v1→v9, then RL (v10) | `G:docs/HISTORY.md:14-26,248-295` |
| decider-2b v11 | v10 | same | 2048 | LoRA r64 α128 on attention + MLP, merged into bf16 | `HF:decider-2b/decider_config.json:stage` |
| decider-4b v1 | `Qwen/Qwen3.5-4B-Base` | 32 = 24 + 8 | 2560 | **all weights**, AdamW without fp32 master copy | `G:docs/HISTORY.md:367-381` |
| decider-4b v2 / v2.1 | v1 | same | 2560 | LoRA r64 α128 attn + MLP, merged | `G:docs/HISTORY.md:396-442` |
| decider-35b-a3b | `Qwen/Qwen3.5-35B-A3B-Base` (34.7B, 256 routed experts, 8 active + shared) | 40 = 30 + 10 | 2048 | **2.45B non-expert params** (attention, DeltaNet, shared experts, routers, norms, embeddings, LM head); 32.2B routed experts frozen; Muon on block matrices | `G:docs/HISTORY.md:303-308,319-328` |
| decider-12b v2 | `google/gemma-4-12B-it` @`707f0a3b` | 48 = 40 sliding + 8 full | 3840 | LoRA r32 α64 on q/k/v/o/gate/up/down, 1 epoch, merged; v1 = stock | `HF:decider-12b/README.md:41-43`, `decider_config.json` |
| decider-31b | `google/gemma-4-31B-it` | 60 | 5376 | "a fine-tune merged into the attention and MLP projections" (rank not stated) | `HF:decider-31b/README.md:18-20` |
| decider-chat-gemma4-31b / -qwen3.6-27b | stock Gemma-4-31B-it / Qwen3.6-27B | 60 / 64 | 5376 / 5120 | **nothing**; only `decider_config.json` | `HF:decider-chat-*/decider_config.json:base` |
| decider-2b-vision | Qwen3.5-2B VLM, v5 text weights transplanted | 24 | 2048 | supervised frames + Cauldron MCQ, then pixel PPO | `G:docs/HISTORY.md:470-525` |

- The backbone is loaded as `AutoModelForCausalLM` and only `lm.model` (no logits over the full vocab) runs in the readout
  (`G:decider/model.py:13,26`). Gradient checkpointing on in training (`:14-15`).
- 0.8B/2B/4B/12B/Gemma-31B have `tie_word_embeddings: true`; 35B-A3B and Qwen3.6-27B do not (`HF:*/config.json`). So on the
  dense Qwen bases the letter readout rows are shared with the input embeddings of `A`, `B`, … and full fine-tuning moves both
  (inferred from the configs + `model.py:28`).
- **No architecture changes.** HISTORY reports that "block-bidirectional attention, readout heads and pause tokens were also
  tried … and gave nothing under this optimizer, so the architecture is unchanged" (`G:docs/HISTORY.md:378-381`).
- Gemma checkpoints: the readout applies the model's `final_logit_softcapping` (30) to the letter logits, `c·tanh(z/c)`
  (`G:decider/model.py:17,38-50`); before 1.7.0 Gemma was read "too sharply" (`G:README.md:87-88`).

### Input layout

Two text layouts, both rendered by `decider/prompt.py`. No special tokens; label letters are ordinary single tokens.

**State-first (default)** (`G:decider/prompt.py:188-226`):
- `"Context:\n" + state` truncated to `max_ctx_tokens` (training 1,536 by default, `G:decider/train.py:65`; serving 32,768,
  `G:decider/serve.py:74`).
- per question k: `"\n\nQuestion[ k]: <text>\nOptions:"`, then the options, then `"\nAnswer[ k]: ("`; the slot is the last
  token, `" ("` (`:210-222`). Numbers appear only when the row has several questions.
- ≤ 10 options ("narrow"): `\n(A) opt …` rendered as one string; > 10 ("wide"): built from ids, `"\n(" + <label token> + ") opt"`,
  so each label is exactly one token (`:212-220`).
- **Label table**: `A..Z` then the first two-letter uppercase strings that are single tokens, 255 total; assertion that all
  are distinct single tokens (`G:decider/prompt.py:107-124`). Same scheme as the 255-code readouts in HF's `jev` head.
- **Schema-first** (`G:decider/prompt.py:147-185`): all question/option blocks first, then `"\n\nContext:\n<state>"`, then the
  answer slots. The question prefix is state-independent and is cached once per schema (`decider/schema_engine.py`). Trained
  50/50 with state-first from v8; costs 1.5 pts on fixed label sets, 5 on per-example options, 5-24 on 50-219 options or long
  states, so it is opt-in (`G:docs/RESULTS.md:169-173`).
- **Chat layout** (1.2.0, used by 12b/31b/chat-*): same content inside the tokenizer's chat template, one user turn, thinking
  disabled (`enable_thinking=False`, or an open `<think>` / Gemma `<|channel>thought` block closed), answer pieces in the
  assistant turn; validated that `Answer: (X` tokenises as prefix + one label token (`G:decider/prompt.py:229-306,343-358`).
- JSON state is serialised with `json.dumps`; arrays of ≥ 8 elements get `"_index": i` written into each element so that path
  questions (`records[47].text`) become a lookup (json_k64 probe 0.49 → 0.57) (`G:decider/systemone.py:19-37`).
- Request ids never reach the model (`G:decider/systemone.py:8`). Option text for a choice is `name` or `name: <description>`
  where structured descriptions are JSON-dumped (`:58`).
- "none of the above"-like options are rewritten to `"not listed here"` for v4-era weights (`neutralize_none`, default true,
  off in every current config) (`G:decider/infer.py:30-44,122`).

```
State-first, independent rows (default system_one / serve, plain layout)            chat layout wraps the same text:

row 1:  Context:\n{"ticket":…}  \n\nQuestion: Which team?\nOptions:\n(A) returns: …\n(B) billing: …\n(C) other   \nAnswer: (
        |------ shared state prefix -------|  |----------------- question 1 suffix ------------------------------|       ^slot
row 2:  Context:\n{"ticket":…}  \n\nQuestion: Does `ticket.messages[0].text` request a refund?\nOptions:\n(A) no\n(B) yes\nAnswer: (
row 3a: Context:\n{"ticket":…}  \n\nQuestion: How frustrated…\nProposed answer: calm\nDoes the proposed answer fit?\nOptions:\n(A) no\n(B) yes\nAnswer: (
row 3b:                 …                                  Proposed answer: frustrated …                                   Answer: (
row 3c:                 …                                  Proposed answer: very frustrated …                              Answer: (
           score question -> one yes/no row per level (isolated levels), P(level) = P_yes(level) / Σ P_yes

serving:  prefix run once -> DynamicCache (attn KV + DeltaNet conv/recurrent state) forked in memory-bounded chunks
          -> each suffix continues it  (decider.shared_prefix, when state ≥ 768 tokens and > 1 row)

readout at each slot (the " (" token):   h_slot · lm_head.weight[A, B, C, …]  -> mask j ≥ n -> softmax(z / T)

Packed form (independent=False; part of the training mix), one causal row:
   plain:        Context:\n…  \n\nQuestion 1: … Options … \nAnswer 1: (  \n\nQuestion 2: … \nAnswer 2: (
   chat/schema:  all question blocks first, then  Answer 1: (  \nAnswer 2: (  …
   slot k sees the state and earlier question texts, never an answer letter (letters are read, not inserted)
```

### Isolation between questions

- `system_one(..., independent=True)` is the default: "scores every question in its own row (state + that question only), so
  adding, removing or reordering questions cannot change any other answer; the state is run once and its cache forked to every
  question (Engine.score_shared)" (`G:decider/infer.py:208-216,251`). Same in the HTTP server (`G:decider/serve.py:120-133`).
- Shared prefix: `shared_prefix.score_shared` forks the prefix cache in chunks of m rows, m = byte budget
  (`DECIDER_SHARED_FORK_GB`, 8 GB, capped at half the free memory) / prefix bytes; suffix tokens per chunk bounded by
  `DECIDER_SHARED_SUFFIX_TOKENS` (65,536) since 1.8.1 (`G:decider/shared_prefix.py:1-25`). Used when the state is ≥ 768 tokens
  (`DECIDER_SHARED_MIN_TOKENS`, `G:decider/serve.py:76`). 7 questions on an 11k-token state: 252 ms vs 1,464 ms, max |Δp| 0.002
  in fp32 (`G:docs/HISTORY.md:226-231`).
- `independent=False` packs all questions behind one copy of the state; "later questions can then see earlier question texts"
  (`G:decider/infer.py:216-217`). Answer letters are never inserted, so no slot sees another answer (`G:decider/prompt.py:70-76`).
  Reversing question order in packed form flips up to 12% (v6) / 17% (v5) of answers; per-row scoring is within 0.7-1.5 points of
  packed accuracy (`G:docs/HISTORY.md:226-229`, `G:docs/RESULTS.md:427-430`).
- Training data deliberately includes single-question and reordered-subset versions of multi-question examples so that "a question
  scored alone must be in-distribution" (`G:decider/data/mixture.py:111-113`).
- Within a question, options are read causally in a list; nothing enforces permutation invariance. decider-31b adds a **second
  reading** with options reversed for questions whose top probability < 0.7, averaging the two log-softmaxes (`G:decider/serve_vllm.py:40-41,109-121`,
  `HF:decider-31b/README.md:105-112`); the RL stage trains a rendering-consistency KL across orders and layouts (`G:docs/RL.md:40-43`).

### Readout math

```
h_k   = backbone(input_ids)[slot_k]                        last hidden state at the " (" token        model.py:26-27
z_k   = W_L · h_k          W_L = lm_head.weight[label_ids]  [255, d]  (fp32 cast)                    model.py:28-29
z_k   = c · tanh(z_k / c)  only for Gemma (c = 30)                                                  model.py:38-41
z_kj  = −inf for j ≥ n_k                                                                             model.py:30-31
p_k   = softmax(z_k / T(type_k, n_k))                                                              temperature.py
```
Probabilities are renormalised over the first n options and rounded to 4 decimals (`G:decider/systemone.py:176-189`).

### Per-type handling (`G:decider/systemone.py:40-74,176-189`)

| type | rendered options | answer | confidence |
|---|---|---|---|
| `noul` | `["no[: false-desc]", "yes[: true-desc]"]` (A = no, B = yes); missing instructions → fixed text "Which answer fits the context?" if a criterion is described (`:44-48,65-70,80`) | `noul = p[1]` | none (as Jev) |
| `choice` | criteria keys in request order, `name` or `name: desc`, 2..255 (`:53-58`) | argmax name + `probabilities` | TypeSafe's `(n·p_max − 1)/(n − 1)` since 1.3.0 (`:155-159`); extras `x_p_max`, `certainty = 1 − H/ln n` (`:139-142`) |
| `score` | list or `{"0":…}` legend, 2..10 levels (`:59-64`) | `score = Σ i·p_i` (2 decimals) + legend + probabilities | `max(0, 1 − Σ p_i·|i − k| / D)`, D = mean |i − (n−1)/2| (TypeSafe's adapter formula) (`:162-173`) |
| `score`, isolated (default when `isolated_levels: true`) | one row per level: `"{q}\nProposed answer: {level}\nDoes the proposed answer fit?"` with options no/yes, level number stripped (`:83-96`) | P(level) = P_yes / Σ P_yes; also `level_fit` and `fit_mass` (unnormalised sum: ~1 when exactly one level fits) (`:99-103,127-136`) | same formula on the normalised p |

`isolated_levels` is true for 0.8b, 2b, 4b, 35b-a3b and false for 12b, 31b and the chat repos (`HF:*/decider_config.json`).
v8 trained this form because naive isolated scoring on v7 gave fits summing to 1.4-3.5 and −20 points; v8 sums 0.99-1.12 and
matches listwise within ~1 point (`G:docs/HISTORY.md:262-266`).

### Calibration

All calibration is **post-hoc temperature on the letter logits**, stored in `decider_config.json`
(`G:decider/temperature.py:1-28`):
- `temperature` (one T), `temperature_by_type` (choice/noul/score, 1.4.0), `temperature_by_options` T(n) = max(min, a + b ln n)
  (1.8.0; replaces the others on state-first), `temperature_schema_first[_by_type]` for the schema cache.
- Fitting: grid search 0.05..20 (801 log-spaced points) minimising NLL on T=1 logits; `fit_by_type` per type with ≥ 50 rows
  (`G:decider/calibrate.py:1-27,145-165`).
- Shipped values: 0.8b T 1.03; 2b v11 1.145, by type choice 1.164 / noul 1.624 / score 1.124; 4b v2.1 1.099, by type 1.11 / 1.56 /
  1.287; 35b 1.08; 12b v2 choice 1.5 / **noul 0.05** / score 1.0; chat-gemma4-31b T(n) = max(0.05, 10.124 − 1.633 ln n);
  decider-31b T1(n) = 2.042 + 0.003 ln n, second reading T2(n) = 1.953 + 0.010 ln n below p 0.7; chat-qwen3.6-27b 1.943
  (`HF:*/decider_config.json`).
- Fit sets: in-task half of their own regression set (61 or 67 tasks; 102,804 rows for 2b v11) plus own validation rows; "no
  JevBench or Decision Index item was used for training, selection or temperature" (`HF:decider-2b/eval_results.json:temperature_fit`).
- **12b noul T 0.05 is tuned for JevBench's scoring** (p in 0.2-0.8 counts as wrong): 92% of answers become exactly 0 or 1, ECE
  0.151 vs 0.038 at T 1.0; the README tells users to set `noul: 1.0` for calibrated probabilities (`G:README.md:317-329`).
- **Training-side calibration**: CE (proper) with optional Brier (`G:decider/train.py:42-51`); v2.1/v11 train replay rows
  toward the parent's distribution (KL) to stop logit sharpening, which had pushed the fitted T to 1.935-2.06 (`G:docs/HISTORY.md:426-441`);
  v10's RL adds a log-score "belief" term against exact outcome laws (`G:docs/RL.md:36-39`).
- decider-2b-coherent adds a learned joint over several questions, IPF-projected onto the served marginals so the answers are
  Dutch-book-coherent (`HF:decider-2b-coherent/README.md:14-40`). It is the only cross-question coupling in the family.

### Limits

| | value | source |
|---|---|---|
| Choice options | 2..255 (label table width) | `G:decider/systemone.py:12,56`, `prompt.py:81` |
| Score levels | 2..10 | `G:decider/systemone.py:12,62` |
| State | 32,768 tokens served (`max_state_tokens`); trained to 16k (v6) / `max_ctx 16384` (35B), probed to 30k | `G:decider/serve.py:74`, `HISTORY.md:168`, `:315` |
| Row | state + 4,096 (`DECIDER_MAX_ROW_TOKENS`); 413 above | `G:decider/serve.py:85,170-185` |
| Rows per request | 1,024 (questions after level expansion) | `G:decider/serve.py:84` |
| Request tokens | 1,048,576 summed over rows | `:86` |
| Queue | 4,096 outstanding rows, then 503 | `:87,188-193` |
| `/decide` context | 1,536 tokens | `:88` |
| Backbone window | 262,144 positions in every config (not used beyond 32k) | `HF:*/config.json` |

Stated weaknesses (`G:README.md:380-415`): no multi-step arithmetic or multi-hop in one pass; calibration on hard items
(2b v10 top-label ECE 0.31 on JevBench hard; v11/4b v2.1 0.18; 35B 0.15); knowledge-heavy MCQ; English only; rules written into
the question not followed at 2B (form probe: one sentence 0.67, a paragraph of rules 0.24); position-addressed JSON records
(0.51 at 64 records); teacher bias (teacher agreed with 72% of its own generic-option labels); browser results only on 22
click-only MiniWoB++ tasks.

---

## Training

### Data (`G:decider/data/mixture.py:1-27,36-37`)

- **~95 public decision datasets** converted to `Example(context, [Q(text, options, gold)])` (`G:decider/data/core.py:1-40`):
  intent (CLINC, Banking77, MASSIVE, HWU64…), topic, sentiment/emotion, toxicity, NLI, MCQ/knowledge, agents (AgentTraj-L,
  Mind2Web), etc. Train cap 20k per task, eval cap 1.5k (`core.py:18-21`); label sets > 10 sub-sampled per example with gold
  kept and shuffled (`:3-6`). Held-out tasks are never trained.
- **Format families** built on top (counts in `MIX`): `wide` full label sets up to 255 (8k per big task), `padded` small sets
  padded with labels from unrelated families (25k), `described` options with descriptions / JSON rubrics under opaque names
  (40k), `json` multi-record JSON states with path questions (32k + 20k indexed), `single` (30k), `isolated` per-level / per-option
  yes/no rows (26k + 9k + 1.2k), `rules` programmatic rule-conditioned records with state and rule twins (90k), `contrastive`
  teacher pairs ×3, `general` replay (200k in delta mode) (`mixture.py:36-37,80-114,141-147,216-220`).
- **Abstention augmentation** at train time: 75% add an abstain option with the gold unchanged, 25% replace all options by an
  unrelated task's labels + abstain = gold (`G:decider/data/augment.py:236-256`; the v5 fix, `G:docs/HISTORY.md:144-156`).
- **Teacher data** (`teacher_data/`, all in the repo: custom_questions 3,043 recs, routing 7,449 + 6,399 terse, commands 4,771,
  situations 1,502, label descriptions for 669 labels):
  - `Qwen/Qwen3.5-27B` via HF `generate`, thinking off, writes label descriptions (`what` / `not_for`) from the label name and 5
    train examples, names only for held-out tasks (`G:decider/data/teacher_labels.py:1-18`);
  - writes states + 8 typed questions with answers in the Jev request shape over 55 domains, then **re-answers each question
    alone from letter logits; only agreeing questions are kept** (`G:decider/data/teacher_questions.py:1-40`);
  - situation→action items (`teacher_situations.py:1-22`); contrastive pairs (Bespoke Nimble's curation recipe: two states
    differing in one fact, both re-answered blind and kept only if both match and differ) (`teacher_contrastive.py:1-30`);
  - routing labels kept only under trust rules (teacher agreement, or the generic group where the checker chose the catch-all)
    (`mixture.py:53-77`). Six domains are held out as probes (`mixture.py:38`).
  - Later stages: document questions by **Qwen3.6-27B with thinking on**, kept when two further independent answers agree
    (`G:docs/HISTORY.md:381-385`); 12b v2 state-tracking logs with gold computed by replaying the log (`HF:decider-12b/README.md:44-52`).
- Mixture v2 (4b and later) = the public mixture (60% of tokens) + 26 further public datasets + 10 programmatic families with
  verifiable gold, 742M tokens; **its builders are not public** (`G:docs/HISTORY.md:370-373`, `G:README.md:179-181`).
- **No Jev outputs**: "Nothing was distilled from Jev" (`G:README.md:48-49`). Two external fixtures (102 TypeSafe workflow rows,
  5,252-row OpenJev) are reporting-only (`G:docs/RL.md:29-30`).

**Targets: hard labels**, CE on the gold index (`G:decider/train.py:42-51`), optional Brier and label smoothing (both default 0).
The only soft targets: KL toward the **parent's own distribution** on replay rows (v2.1/v11), and the RL belief term against
exact laws. Teacher labels enter as hard golds after agreement filtering.

### Recipes

`decider/train.py`: AdamW betas (0.9, 0.95), wd 0, peak lr 1e-5, 200 warm-up, cosine, 16,384 tokens per micro-batch × accum 2,
clip 1.0, max_ctx 1,536, max_options 10 by default (`G:decider/train.py:54-103,123`). Token-budget shape buckets (64 / 512 /
2,048) to keep ~24 compiled shapes (`:22-39`). Each example rendered state- or schema-first at random (`:13-19`).

| checkpoint | recipe | compute | source |
|---|---|---|---|
| `train.sh full` (reference, ≈ v9) | one epoch from Qwen3.5-2B-Base, 1.47M examples, 455M tokens; T fitted 1.03 | 5.3 h GH200 | `G:README.md:339-343` |
| decider-2b v1-v9 | staged continuation epochs: v4 situations/agents/games (45k + 2× replay, 35 min), v5 abstention, v6 Jev input shapes (391k ex, 173M tok, 2.4 h), v7 generic vs catch-all, v8 isolated levels + schema-first 50/50, v9 terse buckets + commands | GH200 | `G:docs/HISTORY.md:84-274` |
| decider-2b v10 | v8 + **RL**: 384 (of 576) steps, PPO actor 0.1 + log-score belief 0.2 + rendering-consistency KL 1.0 + retention KL(v8‖student) 0.7 with a hard 0.01/0.05-nat gate; AdamW fp32 masters, lr 1e-6; live MiniWoB++ click tasks, minesweeper, slippery grid, bag draws; no gold labels | 4 GPUs (separate research repo, not public) | `G:docs/RL.md:1-60`, `HISTORY.md:276-295` |
| decider-2b v11 | v10 + LoRA r64 α128, lr 1e-4, 2 ep, 1,676 steps × 65,536 tok, 42,749 rows, replay KL to v10 | 150 min shared B300 | `HF:decider-2b/decider_config.json:stage` |
| decider-0.8b | `train.sh full` single run | – | `HF:decider-0.8b/decider_config.json` |
| decider-35b-a3b | full mixture, 1,543,567 items / 463M tokens, max_ctx 16,384, none_prob 0.1, max_options 255, schema_first 0.5; **Muon** on 250 block matrices + AdamW elsewhere, fp32 masters, lr 1e-5, 16,287 steps × 32,768 tok; experts frozen | 394 min, 4 × B300 | `G:docs/HISTORY.md:315-328` |
| decider-4b v1 | mixture v2, 742M tok, AdamW bf16 **no fp32 master**, lr 1e-5, 26,729 steps × 32,768 | 577 min, 2 × B300 | `:367-381` |
| decider-4b v2 / v2.1 | LoRA r64 α128 attn+MLP, lr 1e-4, 2 ep, ~1,518 steps × 65,536; 29,325 rows (8,000 code-labelled families, 11,356 Qwen3.6-27B document Qs, 3,293 human-labelled, 6,676 replay); v2.1 replay toward v1 via KL | 94 min, 1 × B300 | `:396-441` |
| decider-12b v2 | LoRA r32 α64, 1 ep, 10,000 items (10.6M tok), lr 5e-5, CE on letters in chat layout | 45 min, 1 × B300 | `HF:decider-12b/README.md:39-55` |
| decider-31b | decision items only in serving layout; public train splits of DI-weak tasks (CLINC150, iSarcasmEval, VAST, ACOS, POP909-CL, GSM8K, NLI4CT, generated smart-home), decontaminated against every DI 0.3 suite row; plus earlier decider data | B300 | `HF:decider-31b/README.md:119-137` |
| decider-2b-vision v3 | v5 text weights transplanted into the VLM; teacher-labelled game frames, DAgger, The Cauldron MCQ; 12 pixel-PPO iterations | – | `G:docs/HISTORY.md:470-525` |

Finding the author draws: fp32 master copies moved small models further from the base and cost knowledge accuracy, so the 2B
and 4B were trained without them; the 35B was trained with them and a retrain is planned (`G:README.md:391-394`).

---

## Evals

Own suites: a **regression set** (eval half of the mixture; 94 tasks on GH200, rebuilt to 67 in-task / 28 held-out on B300),
probes (abstention, JSON paths, routing, contrastive, form filling, games, browser), ECE 15 bins (`HF:decider-2b/eval_results.json:protocol`).

| model | regression in-task acc / ECE | held-out acc / NLL / ECE | T |
|---|---|---|---|
| Qwen3.5-2B-Base zero-shot (94-task) | 0.620 / 0.121 | 0.642 / 0.853 / 0.105 | – |
| decider-2b v8 (94-task) | 0.811 / 0.037 | 0.741 / 0.655 / 0.088 | 1.30 |
| decider-2b v10 (rebuilt) | 0.805 / 0.037 | 0.755 / 0.622 / 0.084 | 1.30 |
| decider-2b v11 (rebuilt, by-type map) | 0.802 / 0.039 | 0.752 / 0.626 / 0.083 | map |
| decider-4b v1 / v2 (rebuilt) | 0.834 / 0.027 · 0.824 / 0.041 | 0.788 / 0.558 / 0.071 · 0.779 / 0.566 / 0.080 | 1.05 / 1.935 |
| decider-35b-a3b (rebuilt) | 0.855 / 0.026 | 0.810 / 0.497 / 0.069 | 1.08 |

(`G:docs/RESULTS.md:147-166`, `HF:decider-2b/eval_results.json:regression`). decider-0.8b: 0.78 / 0.71 on the 94 tasks
(`G:docs/RESULTS.md:175-176`). Full label sets at v6: CLINC 151-way 0.879, DBpedia-L3 219-way 0.871, HWU64 (held-out) 0.841
(`G:docs/HISTORY.md:188-199`).

**Decision Index** (third party, chance-corrected). The 0.3.2 entry list carries seven decider rows
(`notes/decision_index_0.3.2_entries.tsv`): "Decider chat · Gemma-4-31B" (inference technique, 32.68B), "Decider chat ·
Qwen3.6-27B" (27.78B), "Decider 35B-A3B" (full fine-tune, via the NVFP4 repo, 35.95B), "Decider 4B" (4.66B), "Decider 2B" (2.27B),
"Decider 31B (NVFP4)" (32.68B), plus FLock "this-that 1.2" built on decider-2b. The TSV has no scores; the scores below are as
quoted by the author.

| edition / source | entry | score | rank | ECE |
|---|---|---|---|---|
| v0.2.1 board, read 2026-09-29 (`G:README.md:122-139`) | Jev 1.13.0 | 57.91 | listed separately | |
| | decider-chat-gemma4-31b (stock weights) | 57.33 | 2 / 70 | 0.047 |
| | decider-chat-qwen3.6-27b (stock weights) | 51.35 | 8 | 0.021 |
| | decider-35b-a3b (NVFP4) | 47.11 | 12 | 0.023 |
| | decider-4b | 40.70 | 19 | 0.084 |
| | decider-2b | 28.97 | 37 | 0.077 |
| 0.3 public suite, **own run** 2026-10-07/08, 140,178 requests, not verified by maintainers (`HF:decider-31b/README.md:27-66`) | decider-31b | **64.27** (raw 73.24) | – | ECE_bw 0.039 |
| board 2026-10-06 (their runs) | Torchcast Decision 27B / pplx-decider v1.1 / Jev | 65.10 / 62.25 / 57.96 | | |

Per area, decider-35b-a3b vs Jev (v0.2.1): knowledge 0.51 vs 0.69, language 0.61 / 0.62, retrieval 0.34 / 0.37, tools 0.72 / 0.73,
arts 0.53 / 0.56 (`G:README.md:141-153`). decider-31b: Tools 84.23, Language 66.69, Retrieval 65.47, Arts 55.19, Knowledge 50.32;
GPQA 0.35, ChessBench 0.16, HLE 0.00 skill (`HF:decider-31b/README.md:37-43,141-142`). The author's own reading: "On the index
the base model sets most of the score: our trained 35B, 4B and 2B are below these two stock models read the same way"
(`G:README.md:137-139`).

**JevBench** v1.5.2 (Benchmark Heaven, read 2026-09-29): Jev 1.13.0 #3 72.1; decider-4b v2 #7 71.3 (intelligence 55.8,
calibration 85.6, speed 90.9, cost 64.5); decider-2b #28 45.1; decider-35b-a3b #41 27.5 (cost-gated) (`G:README.md:102-120`).
Public hard tier (111 items, own runner): Jev 0.730, decider-12b v1 0.730 / v2 0.712, 35B 0.676, 4b v2 0.676 / v2.1 0.649, 2b v11
0.577 (`G:README.md:153-155`).

**Other**: Bespoke public suite 4b v1 0.757 / v2 0.773 macro; Mind2Web next-element 81.1% → 82.7% (v8 → v10), 4b v1 88.3%; live
MiniWoB++ click tasks sampled 83.0% → 93.2% (v8 → v10), held-out six 72.9% → 91.7%; OpenJev fixture ~63-64%
(`G:docs/HISTORY.md:285-289,388-391`, `G:docs/RESULTS.md:447-454`). Text games and Super Mario from typed decisions, with PPO
(`G:docs/HISTORY.md:99-142,527-580`).

---

## Serving

- **`decider.serve`** (FastAPI): `POST /v1/systemone` (TypeSafe wire format; "their SDKs work unchanged with
  `TYPESAFE_BASE_URL=http://localhost:8000`", `G:README.md:273-274`), `POST /decide` (plain `{context, schema}` with
  bool/choice/scale), `GET /v1/models`, `/health`, `/stats` (`G:decider/serve.py` request handlers after `:454`).
  - Response `{model, answers, usage{input_tokens, output_tokens: 0}}`; `input_tokens` counts the shared prefix once
    (`unique_tokens`, `G:decider/systemone.py:192-199`), so billing semantics follow Jev's "state counted once". Schema-cache
    responses add `cached_tokens`.
  - Extra request fields: `independent` (default true), `layout`; extra answer fields `x_p_max`, `certainty`, `level_fit`,
    `fit_mass` (`G:decider/systemone.py:176-189`).
  - **No authentication**; binds 127.0.0.1 since 1.7.1 (`G:README.md:79-80,270`). 413 / 503 limits as in the table above.
  - CUDA: shape-bucketed CUDA graphs captured at start-up, `torch.compile`, optional FP8 e4m3 linears, continuous batching with
    an adaptive wait window (`G:decider/serve.py:68-87`, `G:docs/HISTORY.md:60-82`). MPS and CPU eager (CPU float32).
- **`decider.serve_vllm`** (vLLM 0.29.0, own venv) for large stock / chat-layout checkpoints: independent `/v1/systemone` only
  (422 for `independent: false`), label tokens as the only allowed ids, `processed_logits` logprobs; second reading and attention
  backend from the config (`G:decider/serve_vllm.py:1-47,366-370`, `G:docs/HISTORY.md:362-366`).
- **GGUF / llama.cpp** through `Decider(..., gguf_file=…)` (1.6.0): one row per decode; packing rows into one decode moves
  probabilities by up to 0.16 in Q4_K_M, so it is off. Not served over HTTP (`G:README.md:232-255`).
- **Library**: `Decider.system_one(state, questions)`, `decide()`, `schema()` (compiled question set; questions-first cache)
  (`G:decider/infer.py:75-246`).
- Latency claims: 2B on GH200 FP8 engine 4.1 ms per 3-question request in-process, HTTP 1 client p50 6.8 ms / 134 req/s, 64
  clients 2,152 decisions/s (`G:docs/HISTORY.md:62-79`); 35B eager on B300 47 ms per request (`:350-353`); decider-31b on B300
  median 39 ms, mean 116 ms, p95 531 ms (`HF:decider-31b/README.md:68-79`); 2B Q4_K_M on 8 CPU threads 0.12-0.31 s (`G:README.md:41`).

---

## Licence

- Code: Apache-2.0 (`G:LICENSE`, `G:README.md:451-453`). `pip install decider-ai`.
- Weights: every repo tagged `apache-2.0` (`HF:*/api.json:cardData.license`). Gemma-based repos state "derived from
  google/gemma-4-…-it, Apache-2.0 as published by Google" (`HF:decider-12b/README.md:106-109`, `HF:decider-chat-gemma4-31b/README.md:52-54`).
  Qwen3.5 / Qwen3.6 bases are Apache-2.0.
- Training-data terms are listed per source on decider-31b: CC BY 3.0 (CLINC150), MIT (GSM8K, iSarcasmEval, MMLU, CSQA),
  CC BY-SA (ARC, SNLI, LogiQA 2.0, BoolQ), **RACE non-commercial** (250 replayed rows), VAST/ACOS/NLI4CT with no stated licence
  (`HF:decider-31b/README.md:154-166`). The repos are still tagged Apache-2.0 as a whole (reported, not judged).
- No Jev outputs are used, so TypeSafe's terms do not arise for this family (`G:README.md:48-49`).

---

## Notable design choices vs Jev

1. **Same contract, same answer formulas.** Jev's three types, limits (255 choice, 2-10 score), `noul` without confidence,
   TypeSafe's choice and score confidence formulas, state referenced by backticked paths, ids hidden from the model, wire-compatible
   `/v1/systemone`. No `box` type and no image assets on the System One endpoint (vision is a separate `decider/vision` model).
2. **Readout through the LM vocabulary.** Option letters are scored by the base's own LM head. This keeps zero new parameters and
   lets stock instruct models be served as "decision models" with only a temperature (the chat-* repos), which Jev's unknown head
   cannot be compared to directly.
3. **Isolation by forked rows.** One causal row per question over a shared, cached state prefix gives Jev's "add or remove questions
   without changing others" exactly. Within a question, options interact and order matters, as Hume measured for Jev; 31b mitigates
   it with a reversed second reading on unsure questions.
4. **Score per level, isolated.** Each level judged alone without its number or neighbours, P(level) renormalised, with
   `fit_mass` as a "no level fits / several fit" signal.
5. **Calibration = temperature after CE (plus RL on the 2B).** No RLCD; post-hoc T per type / per option count, fitted on own
   rows. v10's RL is the only "calibrated decisions via RL" analogue: a log-score belief term on exact outcome laws.
6. **Context**: 32k state accepted, trained mostly ≤ 16k, no 262k mode. Gaps the author names: knowledge (−0.18 vs Jev on DI's
   knowledge area at 35B), multi-step arithmetic, hard-item calibration.

## Notable design choices vs kev

1. **No pointer head.** kev scores `</opt>` hidden states against a `<decide>` query through a trained 256-d bilinear head
   (`clones/kev.md` Readout head). decider reads letter-token logits at one slot after all options. The HF port vocabulary for this
   is `jev` (linear verbalizer) vs `kev` (pointer).
2. **Full fine-tuning on small bases.** decider-0.8b/2b/4b-v1 train all weights at lr 1e-5; LoRA (r64, lr 1e-4) is used only for
   later deltas, then merged. kev keeps the base frozen with r16 LoRA at 5e-5 (until Kev-27B v2's full FT). Both report that
   aggressive updates erode base knowledge; decider's fix is no fp32 master copy and KL-to-parent replay, kev's is a lower lr.
3. **Same isolation mechanics on the same bases.** Both run Qwen3.5 hybrid Gated-DeltaNet models as independent causal rows with
   a cached state prefix (kev `probs()` / `--shared_prefix`; decider `shared_prefix.score_shared`). decider got there first
   (v6, ~2026-09-16/19) and adds memory-bounded chunked forks; neither uses a block-causal mask on hybrid bases.
4. **Plain-text delimiters vs reused special tokens.** decider uses `Context:` / `Question:` / `(A)` / `Answer: (` text, so user
   text can contain lookalikes (the vision model takes the *last* slots because "a state may quote 'Answer 1: (' itself",
   `G:decider/vision/model.py:46`) (inferred: forging is not sanitised in the text path). kev reserves five Qwen special tokens and
   escapes `<|…|>` in user text.
5. **Data**: both use public datasets + programmatic labels; decider adds a large local-teacher layer (Qwen3.5-27B / Qwen3.6-27B
   with self-agreement filtering), RL on live browser/game environments, and a JSON `_index` preprocessor. kev's `KEV_DATE_FACTS`
   is the analogous crutch for dates.
6. **Calibration**: kev ships one T in `head.pt` fitted on held-out datasets (since 9B/27B v2); decider ships per-type or
   per-option-count T in `decider_config.json`, fitted on in-task rows, and one benchmark-tuned value (12b noul 0.05).
7. **Scale strategy**: kev scales by training bigger (27B full FT); decider ends up reading stock 27-31B instruct models and a
   light Gemma-4-31B fine-tune, which tops its own DI number (64.27 public, own run) against Kev-27B's 52.3 on kev's own breadth
   index (different benchmarks; not comparable directly).
