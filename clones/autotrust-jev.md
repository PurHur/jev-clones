# AutoTrust JEV / GEV (JEV-9B, JEV-27B, JEV-27B-VL, JEV-35B, GEV-26B-Decide, AutoTrust-JEV-27B v1.1)

The most downloaded Jev clone family (≈ 4.4M downloads across the repos). JEV-9B, JEV-27B and GEV are **students
distilled directly from TypeSafe Jev 1.13 output distributions**. Each is a frozen open backbone plus a LoRA and a
**24-slot linear verbalizer head** at the last token of a plain-text prompt. One prompt is built per question, and
answers come from one prefill pass. They are served by vLLM with AutoTrust's own `POST /v1/decide`, not TypeSafe's
`/v1/systemone`.

Source pins used for this profile (fetched 2026-10-19; no weights downloaded):

| ref | what | where |
|---|---|---|
| `HF:JEV-9B` | `autotrust/JEV-9B` @ `7ac669b8`: README, judge/calibration/adapter configs, `adapter_vllm/decision_head.json`, `vl/serve_decide.py`, `reports/*` | `notes/hf-raw/autotrust__JEV-9B/files/` |
| `HF:JEV-27B` | `autotrust/JEV-27B` @ `14b42526`: README, configs, `serve_decide.py`, `reports/*` | `notes/hf-raw/autotrust__JEV-27B/files/` |
| `HF:VL` | `autotrust/JEV-27B-VL` @ `9e53b8a8`: README, `blog/JEV-27B-VL.md`, `serve.sh`, configs | `notes/hf-raw/autotrust__JEV-27B-VL/files/` |
| `HF:35B` | `autotrust/JEV-35B` @ `a5d0d580` (+ `JEV-35B-NVFP4` @ `10c2c626`): README, `serve.sh`, `serve_decide.py`, `judge_config.json`, `decision_head.json` | `notes/hf-raw/autotrust__JEV-35B{,-NVFP4}/files/` |
| `HF:GEV` | `autotrust/GEV-26B-Decide` @ `eea84903` (= `JEV-Gemma4-26B-A4B` @ `068ec870`, renamed): README, configs, `calibration_gold.json`, vLLM patch, reports | `notes/hf-raw/autotrust__GEV-26B-Decide/files/`, `…__JEV-Gemma4-26B-A4B/files/` |
| `HF:v1.1` | `autotrust/AutoTrust-JEV-27B-v1.1` @ `ae156900`: README, `NOTICE.base`, `decision_config.json`, `jev_format.py`, `jev_vllm_engine.py`, `serve_vllm.sh` | `notes/hf-raw/autotrust__AutoTrust-JEV-27B-v1.1/files/` |
| `DS:corpus` | dataset `SargeDev/jev-distill-corpus-v3` @ `75111a13`: card, tree, 80 sampled rows via datasets-server (no full download) | `notes/hf-raw/_datasets/SargeDev__jev-distill-corpus-v3/` |
| `DS:DI` | dataset `autotrust/jev-decision-index-results`: card + tree | `notes/hf-raw/_datasets/autotrust__jev-decision-index-results/` |
| `BLOG` | HF community blog "One Decision Model Instead of a Ranking Team" (2026-10-13, Hai Yu / Josh Liu) | `notes/hf-raw/_blogs/autotrust/blog_ranking_team.{html,txt}` |
| `TS:MCA` | TypeSafe Master Customer Agreement (last updated 2026-09-23) and site Terms (2026-09-19) | `notes/web/typesafe/mca.{html,txt}`, `t_terms.html`, `mca_extract.txt` |
| `DIX` | Decision Index 0.3.2 snapshot (2026-10-16) | `notes/hf-raw/_spaces/dix/index.json`, `dix_leaderboard.md` |

Citation form: `HF:JEV-27B/README.md:850` is file:line in the fetched file. `HF:JEV-27B/serve_decide.py:252` is the
shared server; GEV, JEV-27B and JEV-27B-VL ship byte-identical copies (md5 `5cd2f192…`). JEV-35B's copy differs only in the
merged-head, no-MoE-LoRA and no-System-2 switches.

> **Main findings.**
> 1. **The recipe is direct distillation of Jev 1.13's soft distributions.** The labels are 498,010 rows (`yuri_v3`) of
>    synthetic operational scenarios over 53 domains, sent to Jev 1.13 through the **OpenRouter Decisions API**, with the
>    full returned distribution kept as the target (2-decimal precision). Two more streams are added: 94,801 Open-Jev
>    programmatic rows (CC0) and 148,154 rows with **placeholder uniform [0.5, 0.5] labels**, down-weighted ×0.05. The loss is
>    `KL(target ‖ model)` over the active slots plus 0.5·RPS for `score`, trained for about 0.93–0.98 epoch on one B200
>    (`HF:JEV-27B/README.md:850-860`, `DS:corpus/README.md:44-48,87-93`).
> 2. **The architecture is a verbalizer readout, not a pointer head.** It is a rank-16 LoRA on every projection, including
>    the Gated-DeltaNet ones, plus a `Linear(d, 24)` head **initialised from the `lm_head` rows** of `false,true,0..5,A..P`.
>    The head reads the final-norm hidden state of the last token of a plain-text `[kind]…[decision]:` prompt. In vLLM the
>    head is shipped as an `lm_head` LoRA, so a decision is a 1-token, `allowed_token_ids` log-prob read plus a client-side
>    bias and a per-kind T. The backbone and `lm_head` stay bit-identical, so the same weights also serve the base chat
>    model ("System 2") (`HF:JEV-9B/README.md:342-364,601-604`).
> 3. **The family is not one recipe.** JEV-9B/27B/27B-VL use the pure distillation recipe. GEV-26B-Decide (= JEV-Gemma4)
>    and JEV-35B add **ground-truth training splits of 12 Decision Index benchmarks** (author-declared). JEV-35B has a
>    trained 256-wide choice block and merged weights. **AutoTrust-JEV-27B v1.1 is a different model**: its `NOTICE.base`
>    credits **vllm-sr d3**, it uses d3's byte-identical prompt and 255-code readout with prefix-bidirectional attention,
>    it does not declare its training data, and it makes no Jev-distillation claim.
> 4. **The "ECE 0.0007" is calibration against the teacher, not against outcomes.** On the Decision Index ground truth,
>    the family is over-confident: JEV-27B shows acc 0.712 against mean confidence 0.788, ECE 0.076; Jev 1.13 has ECE
>    0.078 (`DIX`).
> 5. **Licence and terms.** Every card tags the weights Apache-2.0 and calls the corpus "public data". TypeSafe's MCA
>    §2.3(b) forbids using "the Services or any Output … to perform model distillation, train a model to imitate the output
>    of the Services". No card, the corpus card, or the HF port addresses that clause. Details are reported below without
>    judgement.

---

## Identity

| | |
|---|---|
| Org | **AutoTrust AI Lab** (`huggingface.co/autotrust`, 15 repos; ModelScope mirror `AutoTrust-Labs`). The blog authors are Hai Yu (`cloudyu`) and Josh Liu (`AutoTrustAILab`) (`BLOG` byline). The vLLM Gemma-4 patch is authored by GitHub user `yuhai-china` (`HF:GEV/patches/vllm-gemma4-lm-head-lora.patch:2`), the same user as the demo repo `github.com/yuhai-china/JEV-27B-DEMO` (`HF:VL/blog/JEV-27B-VL.md:142`). There is no GitHub training repo; the training code is **not published** |
| Corpus author | `SargeDev` (a separate HF account). `SargeDev/jev-distill-corpus` v1 (2026-09-21, memory-relevance pairs) and `jev-distill-corpus-v3` (2026-09-21, last modified 10-08; 4,037 downloads). SargeDev also publishes a "Jev-Gate Student B" LoRA (Qwen2.5-0.5B), `Jev_Qwen3.8-27B` and "Uncensored" JEV-27B/VL re-bases on an abliterated Qwen (`_authors/SargeDev_*_2026-10-19.json`). Any affiliation with AutoTrust is not stated (inferred: separate parties) |
| Lineage claimed | "System 1 distilled from TypeSafe Jev 1.13" (`HF:JEV-9B/README.md:38,196-200`). It also invokes a "Blocks of Experts" framing: a frozen backbone as the System 2 block plus a detachable System 1 block (`HF:JEV-9B/README.md:171-176`). TypeSafe is credited only for "the System One framing and the noul/choice/score primitives" (`HF:JEV-9B/README.md:785-787`) |
| HF transformers | The `jev` head type in PR #49566 is named after `autotrust/JEV-9B`, and `ArthurZ/*` hosts ports of all of them (`clones/hf-jev-model-type.md`) |

| repo (sha, created) | backbone | params (safetensors API) | downloads / likes | what is trained |
|---|---|---|---|---|
| `JEV-9B` (`7ac669b8`, 09-23) | `Qwen/Qwen3.5-9B` text tower | 8.95B (BF16 8,953,799,424 + F32 3,840) | 454,435 / 170 | LoRA 40.1M + head 98k |
| `JEV-27B` (`14b42526`, 09-25) | `Qwen/Qwen3.8-27B` @`1d4bf0f2` text tower | 26.90B BF16 | 239,683 / 93 | LoRA 108.8M + head 123k (122,904) |
| `JEV-Gemma4-26B-A4B` (`068ec870`, 09-29) | `google/gemma-4-26B-A4B-it` @`4d7ae498` | 25.81B BF16 | 33,900 / 22 | LoRA r32 + head 67,608 |
| `JEV-27B-VL` (`9e53b8a8`, 09-30) | full multimodal Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`) + JEV-27B's adapter | 27.78B BF16 | **1,929,553 / 7,923** | nothing new: the same adapter/head as JEV-27B |
| `GEV-26B-Decide` (`eea84903`, 10-02) | as JEV-Gemma4; "the weights are the same" (`HF:GEV/README.md:214`) | 25.81B | 1,119,803 / 2,453 | as JEV-Gemma4 (calibration source `s2_gemma4_step7000` in both) |
| `JEV-35B` (`a5d0d580`, 10-11) | `Qwen/Qwen3.5-35B-A3B` (MoE, 256 experts, 8 active, d=2048) | 35.95B (BF16 + F32 4,800) | 454,857 / 2,831 | LoRA (250 modules, no routed experts) + **264-slot** head, all **merged** |
| `JEV-35B-NVFP4`, `GEV-26B-Decide-NVFP4` | routed experts in NVFP4 | 19.85B / 14.39B stored | 136,542 / 69,156 | quantisations |
| `AutoTrust-JEV-27B-v1.1` (`ae156900`, 10-17) | d3 (vllm-sr) on Qwen3.8-27B @`1d4bf0f2` | 27.36B BF16 + `lm_head.safetensors` 2.54 GB | 5,615 / 72 | undisclosed (full weights shipped) |

Other autotrust repos are GLM-5.3 GGUF/DGX-Spark quantisations and a Qwen3.8-27B NVFP4, which are unrelated. Third-party
derivatives include about 14 JEV-27B-VL quantisations (GGUF/AWQ/FP8/NVFP4/MLX/EXL3/GPTQ), JEV-9B CoreML/MLX/GGUF,
`dakebakery/Bonsai-JEV` (ternary 27B), `inference-sh/JEV-27B-VL-merged` and `LEAP-LAB-KUS/jevkt-checkpoints` (HF
`base_model:` filter, 2026-10-19).

---

## Architecture

### Backbone and blocks

- **JEV-9B / JEV-27B.** The repo is the base text tower **including the original `lm_head`**, bit-identical to the base
  (vision tower and MTP head dropped), plus an unmerged peft LoRA in `adapter/` and `head.safetensors`
  (`proj.weight [24, d]`, `proj.bias [24]`, fp32) (`HF:JEV-9B/README.md:352-357,763-780`; `HF:JEV-27B/README.md:898-915`).
  `config.json` declares `Qwen3_5ForCausalLM` / `qwen3_5_text`.
- **The adapter is frozen-backbone LoRA.** It uses r=16, α=32, dropout 0.05 on `in_proj_qkv, in_proj_z, out_proj` (the
  Gated-DeltaNet projections) and `q/k/v/o_proj, gate/up/down_proj` (`HF:JEV-9B/adapter/adapter_config.json:23-48`,
  `HF:JEV-9B/judge_config.json:84-100`). No DoRA or rsLoRA. That is the same target set as Kev 1.0's hybrid-base LoRA.
- **`adapter_vllm/`** holds the same LoRA zero-padded to r=32 (α 64, so the scale is unchanged at 2) **plus `lm_head` as
  a LoRA target**. The 24-slot head is "re-expressed as an lm_head LoRA (only the 24 verbalizer rows change)"
  (`HF:JEV-9B/adapter_vllm/adapter_config.json:23-49`, `HF:JEV-9B/README.md:452-456`). The bias lives in
  `decision_head.json` and is added client-side; it is tiny (|b| ≤ 0.007, `HF:JEV-27B/adapter_vllm/decision_head.json`).
- **System 1 vs System 2.** Adapter off: the pristine base, with HumanEval 70.7% (9B) / 78.0% (27B), all 164 completions
  byte-identical to the base. A LoRA merged into the backbone with the original `lm_head` drops HumanEval to 61.6%, which
  is why the blocks are kept separate (`HF:JEV-9B/README.md:359-364,669-675`). In vLLM the two systems are two LoRA
  "models" in one engine (`--lora-modules jev-decision=…/adapter_vllm`) (`HF:JEV-9B/README.md:461-465`).
- **GEV / JEV-Gemma4.** A LoRA r=32, α=64, dropout 0.05 on `self_attn.(q|k|v|o)_proj` and `mlp.(gate|up|down)_proj` of the
  language model only (regex in `HF:GEV/judge_config.json`). The DeltaNet-style targets do not exist on Gemma. The head
  is `Linear(2816, 24)` with a **softcap of 30** (`z = 30·tanh(z/30)`, like Gemma's own final logits), and the prompt is
  prefixed by `<bos>` (`prefix_token 2`) (`HF:JEV-Gemma4/README.md:84-92,126-134`). vLLM needs the bundled patch that allows
  LoRA on Gemma-4's tied `lm_head` and raises vLLM's LoRA vocab limit from 258,048 to 262,144
  (`HF:GEV/patches/vllm-gemma4-lm-head-lora.patch:1-30`). `weights_mode: "dual"`.
- **JEV-35B.** `Qwen3_5MoeForConditionalGeneration` (40 layers = 30 DeltaNet + 10 attention, 256 experts / 8 active,
  vision tower kept) (`HF:35B/config.json`). `judge_config.json`: `weights_mode "merged"`, `"backbone_lora": "250 modules,
  scaling 2, merged in fp32 -> bf16"`, `"decision_head": "264 lm_head rows replaced (bias client-side)"`, `layout
  wide256`, `model_version 1.1.0`. The LoRA has **no routed-expert weights** ("attention, linear attention and shared expert
  only", `HF:35B/serve_decide.py:56-60`). Because the LoRA and head are merged, it is **System 1 only**: "it does not chat
  or generate text" (`HF:35B/README.md:366-368`). `serve.sh` sets `JEV_DECIDE_SYSTEM2=0`.
- **AutoTrust-JEV-27B v1.1.** Full fine-tuned weights (`Qwen3_5ForConditionalGeneration`, `jev_prompt_attention:
  "bidirectional"` in `config.json`) plus `lm_head.safetensors`, whose "rows at readout_token_ids = decision readout weight
  (bf16); other rows zero" (`HF:v1.1/vllm_readout.json:note`). `NOTICE.base` reads: "d3 (Decision 3.0) Copyright 2026 vLLM
  Semantic Router Team. Built on Qwen/Qwen3.8-27B". `jev_format.py` says it is "Adapted from the upstream package credited
  in NOTICE.base … every prompt string is unchanged" (`HF:v1.1/jev_format.py:13-14`). Its strings match `vllm-sr/d3`'s
  `d3_format.py` (e.g. `_CLOSING_LINE`, `HF:vllm-sr__d3/d3_format.py:36`). Serving needs the StartLux prefix-bidirectional
  vLLM plugin (`STARTLUX_PREFIX_BIDIR=1`, no prefix caching, no chunked prefill) (`HF:v1.1/serve_vllm.sh:5-14`). See
  `clones/vllm-sr.md` for the d3 architecture. The rest of this file is about the distilled models unless v1.1 is named.

### Input layout (`bare-v1`, protocol `jev27-bare-v1`)

One **plain-text string per question**, tokenised whole, with no chat template and no special tokens (except `<bos>` on
Gemma) (`HF:JEV-27B/serve_decide.py:59,252-253`; `HF:JEV-9B/README.md:589-604`):

```
[kind] {noul|choice|score}\n[state] {state}\n[question] {question}\n[options]\n{line_0}\n…\n{line_k-1}\n[decision]:
```

- `state`: a string verbatim; a dict becomes `json.dumps`; a list mixes text and `{"image": url|data:}` parts
  (`HF:JEV-27B/serve_decide.py:148-165`).
- Option lines: `choice` → `"{label}) {option}"`, with labels A–P first, then Q–Z, then two-letter single-token labels
  (`single_token_labels("x\n{}) y")`, asserted to start with the trained A–P ids, `:112-124`). `noul` → the literal lines
  `false`, `true`. `score` → the literal lines `0`…`5` (`:246-251`, `:365`).
- **There is no place for Jev's `criteria` descriptions on noul and score.** The level texts and the true/false
  descriptions are not rendered, so a score scale has to be described inside `question` (inferred from `:251,365`). The
  Decision Index engine rewrites every noul statement as `"Is this scenario one where: <statement>?"`, "the form in which
  all 150,515 TypeSafe Jev 1.13-labelled noul rows of the training corpus present a statement" (`DS:DI/README.md:46-48`).
- Delimiters are plain text (`[kind]`, `A) `, newlines). Nothing escapes them; the card only advises "keep newlines out of
  option text" (`HF:JEV-27B/README.md` "Writing System 1 prompts" item 6) (inferred: option lines and `[decision]:` are
  forgeable from state text, unlike kev's escaped special tokens).
- Training sequences: max 856 tokens, mean ≈ 129, p95 ≈ 351, with `max_seq_len 1024` (`HF:JEV-9B/reports/data_audit.md:3-16`).
  The "original reference server truncates states longer than 1,024 tokens (head 60% / tail 40%)"; vLLM paths use the
  whole prompt up to 262,144 (`HF:JEV-27B/README.md` Limitations, "Context length").

```
One row per question (causal; no packing, no mask tricks)

row q1:  [kind] choice\n[state] <state tokens …>\n[question] <q1>\n[options]\nA) opt0\nB) opt1\n…\nP) opt15\n[decision]:
                                                                                                                       ^ read h_last
row q2:  [kind] noul\n[state] <same state, re-encoded>\n[question] <q2>\n[options]\nfalse\ntrue\n[decision]:
                                                                                                    ^ read h_last
         head z = W·h_last + b  (W ⊂ lm_head rows, 24 or 264 slots)
         slots: 0-1 false,true | 2-7 "0".."5" | 8-23 A..P  (JEV-35B: 8-263 A..JU)

prefix sharing (vLLM --enable-prefix-caching, 528-token blocks): only rows of the SAME kind share
"[kind] X\n[state] …" -> state is reused only across questions of one kind, only if > 528 tokens

GEV/JEV-Gemma4: "<bos>" + same string; z = 30·tanh((W·h+b)/30)
v1.1 (d3 format): chat template, system "You are a decision engine…", "State:\n…\n\nQuestion:\n…\n\nOptions:\nA: …\n\n
                  Reply with only the code of the best option." -> 255 codes at the assistant position, bidirectional prompt
```

### Isolation between questions

- **Exact, through separate prompts.** `/v1/decide` takes **one** question per request (`kind, state, question,
  options`) (`HF:JEV-27B/serve_decide.py:76-89`). Batched engines put each question in its own row: the Decision Index
  `fast_run.py` uses length-sorted rows with right padding and no mask, "exact for this causal model because pad tokens
  follow the last real token" (`DS:DI/README.md:65-70`).
- **The state is re-encoded for every question.** Prefix caching recovers part of that cost: "the template puts `[kind]`
  before `[state]`, so only questions of the same kind share a prefix", with 528-token blocks on the hybrid architecture,
  giving +14–20% throughput on 7.7 questions per state (`HF:JEV-9B/README.md:573-576`). Jev's "state once, many
  questions" is therefore not reproduced structurally.
- Within a question, options are read jointly and in order (causal). Order sensitivity: 3.9% (9B) / 2.9% (27B) top-1
  flips under shuffle on the test set; on `decision-models-under-pressure` with 16 options, 11.5% / 7.4% against Jev's
  7.0% (`HF:JEV-9B/README.md:247,251`).

### Readout math

```
h      = backbone(prompt)[-1]                         final-norm hidden state of the last token ":"  (fp32 for the head)
z_s    = W_s · h + b_s ,  s ∈ slots(kind)             W ∈ R^{24×d} (JEV-35B 264×d), init W = lm_head[verbalizer_ids]
z_s    = 30·tanh(z_s/30)                              Gemma only
p      = softmax_s( z_s / T_kind )                    over the kind's active slots, first n for choice
vLLM:  z_s = logprob_LoRA(token_s | allowed_token_ids) + b_s     (log-softmax normaliser cancels)
```

- **Initialisation is the key trick.** At step 0 the head equals the backbone's zero-shot restricted next-token
  distribution (|Δp| 8.6e-07 measured). The untrained reference point is choice top-1 0.532, noul AUROC 0.824, KL 0.510
  (`HF:JEV-9B/README.md:368-372,705`).
- **vLLM path.** `max_tokens=1`, `temperature 1`, `top_k 0`, `top_p 1`, `--logprobs-mode processed_logprobs` and
  `allowed_token_ids = ids` "makes the returned log-probabilities respect allowed_token_ids". Above `--max-logprobs` the
  server falls back to `logprob_token_ids` in chunks of 128 (`HF:JEV-27B/serve_decide.py:58,71-72,199-238`). The generation
  config's `top_k 20 / top_p 0.95` would otherwise truncate the distribution (`HF:VL/README.md:637-640`). On GEV the bias
  is added after vLLM's softcap, which is "exact to ~1e-2" (`HF:GEV/adapter_vllm/decision_head.json:note`).

### Per-type handling

| kind | slots / lines | answer returned by `/v1/decide` | limits |
|---|---|---|---|
| `noul` | `false`,`true` (slots 0–1) | `probabilities [P(false), P(true)]`, `choice`, `choice_index` | options fixed |
| `score` | `0`…`5` (slots 2–7) | distribution over 6 levels (the README adds an expected score) | **always 6 levels**, 0–5; Jev allows 2–10 with level texts |
| `choice` | A–P (slots 8–23), labels continue past P | distribution aligned with `options` | 2–256 on vLLM; 2–16 on the transformers head path |

- **More than 16 options** (`HF:JEV-27B/serve_decide.py:258-293`):
  - `single`: one pass with untrained labels Q…, AA…, read from the base `lm_head` rows (biases 0). This is the Qwen
    default and gives `adaptation: native`/strategy name.
  - `tournament`: ⌈n/16⌉ contiguous groups, then a final of 16. Finalists get `final × P(answer among finalists)`; the rest
    get `group share × in-group p`. This is the Gemma default and the Decision Index method.
  - `permute`: 4 orders, averaged.
  - JEV-35B trains all 256 choice slots (`wide256`), so up to 256 options are native (`HF:35B/serve_decide.py:292-296`).
- **Confidence.** `/v1/decide` returns **no `confidence` field** and no `legend`/`score` keys, so it is not TypeSafe's answer
  schema. The kit engine maps it back to System One answers (inferred from `DS:DI/README.md:38-49`; the engine file itself
  was not fetched).
- **Training distribution of option counts.** It is dominated by 4-option questions: 142,002 of 163,657 choice rows
  have 4 options and only 3,100 have 16 (`HF:JEV-9B/reports/data_audit.md:70`).

### Adaptive thinking (System 1 → System 2), `thinking: "auto"|"on"`

If `max(p1) < threshold` (default 0.8), the base model thinks in its own chat template. When the thinking channel
closes, `"Answer: ("` is appended and the answer-letter distribution p2 is read in one step. The returned distribution is
`p = (1−w)·p1 + w·p2` with w = 0.5, "which keeps the calibration of System 1; p2 alone is close to one-hot"
(`HF:JEV-27B/serve_decide.py:26-29,308-339,393-395`). It is not available for `score` or on JEV-35B. On GEV it was tuned
on 1,754 questions outside the Decision Index: accuracy 73.3 → 83.4 while thinking on 47.8% of questions, ECE 0.035
unchanged (`HF:GEV/README.md:270-296`). It hurt BANKING77, "whose training split System 1 was trained on"
(`HF:GEV/README.md:340-342`).

### Calibration

- **Per-kind scalar temperature**, fitted with L-BFGS by **KL against the teacher targets** on the corpus `calibration`
  split (10,954 rows, `yuri_v1` placeholders excluded, `d1_excluded: true`).
  - JEV-9B: 1.002 / 0.984 / 1.012.
  - JEV-27B (and VL): 1.014 / 1.016 / 1.004.
  - GEV: 1.003 / 1.017 / 0.999.
  - JEV-35B: 0.984 / 1.012 / 1.008.
  - KL changes in the 4th decimal; on JEV-9B choice it goes from 0.04468 to 0.04462 (`HF:*/calibration.json`).
- "Calibration falls out of the objective": distilling full teacher distributions gives T ≈ 1 with no post-hoc
  correction (`HF:JEV-9B/README.md:380-381`).
- **GEV additionally ships `calibration_gold.json`**, "calibrated against ground-truth answers": noul 1.214, choice 1.098,
  score 1.000. It is recommended "when you gate automatic actions on confidence" (`HF:GEV/README.md:527-534`). That is an
  admission that teacher-matched T is softer than outcome-matched T needs to be (inferred: the student, like Jev, is
  over-confident against truth).
- **The reported ECE 0.0007 is against teacher soft targets** (15 bins) (`HF:JEV-9B/README.md:61-62,216`). Against Decision
  Index ground truth (`DIX`):

| | acc | mean conf | ECE | Brier |
|---|---|---|---|---|
| JEV-9B | 0.658 | 0.730 | 0.073 | 0.446 |
| JEV-27B | 0.712 | 0.788 | 0.076 | 0.383 |
| GEV-26B-Decide | 0.737 | 0.810 | 0.072 | 0.353 |
| JEV-35B | 0.741 | 0.826 | 0.085 | 0.359 |
| Jev 1.13 | 0.738 | 0.815 | 0.078 | 0.358 |

  The students reproduce the teacher's over-confidence closely (inferred from the table).

### VL path

- **No vision training at all.** "Its decision head never saw an image during training" (`HF:VL/blog/JEV-27B-VL.md:9`).
  JEV-27B-VL is the **unmodified multimodal Qwen3.8-27B** (vision encoder unchanged) carrying JEV-27B's text-trained
  `adapter_vllm`. JEV-9B's `vl/` does the same with multimodal Qwen3.5-9B: "the same weights with the layer names moved".
  Text decisions match the text model: max |Δp| 0.011 on 300 decisions for 9B, and the same top option on 24/24 for 27B
  (`HF:JEV-9B/README.md:157-160`; `HF:VL/blog/JEV-27B-VL.md:102`).
- **Images go in `state` as list parts.** On the image path, `/v1/decide` sends a chat-completion request with a raw
  template that just concatenates text and `<|vision_start|><|image_pad|><|vision_end|>` (Gemma: `<|image|>`), so the
  prompt stays `bare-v1` (`HF:JEV-27B/serve_decide.py:65-68,133-138,226-230`). This needs `--trust-request-chat-template`.
- **`--max-num-seqs 8` is required** on JEV-27B-VL: "With more than 8 sequences in one batch, vLLM's LoRA path for this
  multimodal model class returns wrong System 1 probabilities" (`HF:VL/serve.sh:7-8`, `HF:VL/README.md:629-632`). At most
  8 images per prompt.
- GEV keeps Gemma-4's vision encoder, and JEV-35B keeps Qwen3.5-MoE's. Both are zero-shot on images.

### Limits

| | JEV-9B / 27B / VL | GEV | JEV-35B |
|---|---|---|---|
| choice options | native 16; 2–256 (vLLM); 2–16 (transformers) | native 16, tournament to 256 | native 256 |
| score levels | 6 (0–5) fixed | same | same |
| questions per request | 1 (`/v1/decide`) | 1 | 1 |
| context | 262,144 native (`--max-model-len`, default 32,768 in VL `serve.sh`); trained ≤ 856 tok | tested to 128K | 65,536 default |
| types | noul/choice/score; **no box** | + adaptive thinking | System 1 only |

- Needle tests (one fact at a random depth in PubMedQA text): 10/10 yes/no and 10/10 16-option at 4K–250K, mean p ≈ 0.997,
  250K in 28.7 s (`HF:JEV-27B/README.md` "Context length"). This is single-fact retrieval only.
- Choice with all 150 CLINC intents: 89.5% (names) / 93.8% (one-line "use when" descriptions), ECE 0.079 / 0.034
  (`HF:JEV-27B/README.md:636-659`).

---

## Training

**Data: `SargeDev/jev-distill-corpus-v3`** (740,957 rows, Apache-2.0) (`DS:corpus/README.md:37-85`;
`HF:JEV-27B/README.md:850`)

| stream | rows | labels | content |
|---|---|---|---|
| `yuri_v3` | 498,010 | **TypeSafe Jev 1.13 full output distributions, "via the OpenRouter Decisions API"** | synthetic operational scenarios, 53 domains (business, technical, agent, knowledge, spatial, structured, medical, biology, genomics, physics, chemistry, theology, science), 3 primitives per domain |
| `openjev_v2` | 94,801 | Open-Jev programmatic ground truth (CC0, `ZefanCai/Open-Jev release-v2-redistributable`) | games, pixel geometry, workflow-control policies |
| `yuri_v1` | 148,154 | card: "distilled from a 32B teacher" over 18 QA datasets; **in fact 100% exact-uniform [0.5, 0.5]** (`HF:JEV-9B/reports/data_audit.md:24`) | memory-relevance noul pairs; down-weighted ×0.05 |

- **How the Jev labels were obtained.** One Jev 1.13 call per (state, question). The full distribution per question was
  kept; "degenerate labels (max-prob ≥ 0.999 with confidence ≥ 0.95)" were filtered (1.5%); the noul mean is 0.45, sd 0.28
  (`DS:corpus/README.md:87-93`). Score targets are "distributed over ordered levels 0..5 (labels distilled from Jev's
  expected-score distribution)" (`:80-83`).
- **The rows are short and templated.** Sampled states are ≈ 170–240 characters. Questions follow fixed templates such
  as `"Is this scenario one where: this warrants waking the on-call?"`, `"Next action for this scenario."` with 4 options,
  and `"Rate emotional intensity for this scenario on a 0-5 scale."`. Targets are at 2-decimal precision, e.g.
  `[0.32, 0.09, 0.57, 0.02]` (`DS:corpus/rows_*.json`).
- **Training-set size.** Train mix: noul 52.1%, choice 25.0%, score 22.9%; yuri_v3 67.7%. ≈ 84.6M train tokens per epoch
  (`HF:JEV-9B/reports/data_audit.md:16,56-68`). The models saw ≈ 608k (9B) / 640k (27B) rows, i.e. **≈ 0.4M Jev-labelled
  rows in practice** (inferred from the 67.7% share).
- **Splits:** train 655,806; validation 14,111; calibration 13,766; test 14,261; `ood` 13,058 (Open-Jev families not in
  train); `test_set_30k` 29,955 (25,376 Jev-labelled).
- **There is no Jev-labelled OOD set**, so "indistinguishable" holds only on the 53 training domains
  (`HF:JEV-9B/README.md:741-743`).

**Soft vs hard targets.** The targets are **soft**, full teacher distributions, for every Jev row. Open-Jev rows are
near one-hot. GEV/JEV-Gemma4 and JEV-35B additionally use hard ground-truth rows from the public **training splits of
12 Decision Index datasets**: BANKING77, CLINC150+OOS, WinoGrande, HellaSwag, GSM8K, ContractNLI, Amazon ESCI, HoVer,
RAGTruth, When2Call, New Yorker caption matching and Habermas Machine (`DS:DI/README.md:119`; `HF:GEV/README.md:536-542`;
DI board note for JEV-35B: "LoRA on Qwen3.5-35B-A3B plus a decision head, using public training splits of suite benchmark
datasets and distillation from Jev 1.13 answer distributions (author-declared)", `DIX`). The JEV-35B card itself has no
training section. The data for its 256-wide choice slots is undisclosed (inferred: the 77/150-intent training splits).

**Recipe (JEV-9B / JEV-27B; identical code and hyper-parameters, "only model_path and memory settings changed")**
(`HF:JEV-9B/README.md:711-726`, `HF:JEV-27B/README.md:846-861`):

| item | JEV-9B | JEV-27B |
|---|---|---|
| loss | KL(target ‖ model) over active slots + 0.5·RPS for score | same |
| augmentation | 30% random permutation of choice options (targets permuted with them) | same |
| batching | 128 rows/step, kind-stratified (≥ 1/6 each), length-bucketed, micro-batches ≤ 24k padded tokens, grad checkpointing | same |
| optimiser | AdamW fused, β (0.9, 0.98), lr head 2e-4 / LoRA 1e-4, cosine, warm-up 3%, clip 1.0. 2,500 steps, then a restart on the unseen rest of the epoch (fresh AdamW, warm-up 2%) for 2,250 more, ending at lr ≈ ×0.07 | one cosine over 5,124 steps (min ×0.02); released step 5,000 (0.98 epoch), selected by val KL |
| rows seen | 608k (4,750 steps, 0.93 epoch) | 640k |
| compute | 1× B200, ≈ 3 GPU-h, 7–9k tok/s | 1× B200, ≈ 9.2 GPU-h, 2.4k tok/s, peak 79 GB |
| software | torch 2.13 + cu130, transformers 5.16, peft 0.21, flash-linear-attention 0.5.2 | same |

- Trajectory for 9B: val KL 0.094 at 64k rows → 0.026 after a 500-step cool-down → 0.019 at release. "Annealing matters"
  (`HF:JEV-9B/README.md:694-709`).
- GEV/JEV-Gemma4 is checkpoint `s2_gemma4_step7000`; JEV-35B is `s2_q35moe_ts/last` (`HF:*/calibration.json:source`).
  Other hyper-parameters are not published.

---

## Evals

**Teacher fidelity (self-reported, `test_set_30k`, Jev-labelled rows)** (`HF:JEV-9B/README.md:209-256,322-329,677-692`)

| | JEV-9B | JEV-27B |
|---|---|---|
| mean KL to Jev (all Jev rows) | ≈ 0.019 | ≈ 0.017 |
| per kind KL noul / choice / score (9B) | 0.005 / 0.028 / 0.023 | – |
| choice top-1 agreement (all / teacher-decisive gap ≥ 0.1) | 90.2% / 95.4% | 90.5% / 95.8% |
| noul AUROC vs teacher | 0.994 | 0.995 |
| score E[level] MAE | 0.103 | 0.098 |
| Open-Jev OOD KL / top-1 (ground truth) | 0.234 / 0.918 | 0.104 / 0.942 |

On near-ties (14% of rows have a top-2 gap < 0.1) agreement is near chance for any mimic (0.46 at a gap < 0.05)
(`HF:JEV-9B/README.md:746-750`).

**Decision Index 0.3.2 board** (independent, 2026-10-16 snapshot, `DIX`):

| entry | DI Full (rank of 173) | public skill | median latency |
|---|---|---|---|
| Jev 1.14 (reference) | 61.53 (#13) | – | – |
| Jev 1.13 | 60.11 (#18) | 57.96 | 524 ms (hosted) |
| JEV-35B | 56.95 (#37) | 59.73 | 50.5 ms |
| GEV-26B-Decide | 56.17 (#39) | 58.03 | 128.5 ms |
| JEV-27B | 55.64 (#44) | 53.64 | 102.1 ms |
| JEV-9B | 45.21 (#78) | 43.28 | 49.5 ms |
| (vllm-sr d3 27B, v1.1's upstream) | 66.26 (#2) | 65.45 | – |

- The board labels JEV-9B/27B as "full fine-tune", although they are LoRA (`dix_leaderboard.md:48,82`). The JEV-35B entry
  was listed under a maintainer exception: 98.42% agreement against a 98.5% bar on private-test reruns (board note).
- **Self-reported scores, not board entries:**
  - DI 0.2.1: JEV-9B 43.14, JEV-27B 53.30, JEV-Gemma4 58.05 vs the Jev 1.13 board value of 57.91 (`DS:DI/README.md:13-17`).
  - GEV with adaptive thinking on Knowledge & Reasoning: **62.48** (`HF:GEV/README.md:134-158`).
  - DI 0.3 public: JEV-35B 59.73 (`HF:35B/README.md:150-165`); **v1.1 67.31** (`HF:v1.1/README.md:18-62`, vLLM 0.31, B300).
  - Vision board 0.3 (2026-10-07): **JEV-27B-VL #1 of 20, 69.82**, ECE 0.019 (`HF:VL/README.md:120-174`).
  - The train-split overlap above applies to GEV, JEV-35B and (undisclosed) v1.1 scores. GEV's card says "MMMU / MMMU-Pro
    are not valid evaluations for this model" (`HF:GEV/README.md:536-542`).

**Other self-reported**

- Six public groups (JevBench, Kev, OpenJev text, Nimble, VitaminC, MASSIVE-en): JEV-27B 84.07 vs Jev 1.13 (AutoTrust's
  run, 2026-09-26) 83.85. Jev is ahead on Kev (85.52 vs 83.75) and VitaminC (`HF:JEV-27B/README.md:120-173`).
- `decision-models-under-pressure` (human gold labels), 2/4/8/16 options: Jev 0.890/0.801/0.782/0.769; JEV-27B
  0.876/0.784/0.767/0.740; JEV-9B 0.868/0.774/0.735/0.694 (`HF:JEV-9B/README.md:397-403`).
- Judge benchmarks: RewardBench 89.9 (System 1, both orders); VL-RewardBench 78.3% (27B-VL) / 78.4% (GEV); MMRB2
  text-to-image 69.2% (`HF:VL/blog/JEV-27B-VL.md:14-24,47-62`; `HF:GEV/README.md:358-369`).
- ERQA: 27B-VL 53.8% one pass / 61.5% with thinking (medium); JEV-35B 51.3%; GEV 43.5% / 56.8% (`HF:VL/README.md:43-118`).
- Demos: computer use with numbered boxes + element text, 95% of 60 tasks (9B, 27B-VL, 35B); boxes only, 37% / 10% / 38%.
  The robot arm uses binary left/right and up/down servo questions; "choose one of 8 motor commands directly did not work"
  (`HF:JEV-9B/README.md:110-131`; `HF:35B/README.md:279-311`).
- Speed (B200, no network): 9B 87–90 ms single, 2.5 ms/decision at batch 128; 27B 137 ms / 4.2 ms; GEV ≈ 45 ms; 35B 184 ms
  text, 346 ms image median over HTTP (`HF:JEV-9B/README.md:296-308`; `HF:35B/README.md:269-277`).

**GuruSearch demo claims** (`BLOG`; banner on every card, "new, 11 October 2026", `HF:JEV-9B/README.md:85-94`)

- `news.guru.so`, recommendation: "60 decisions in 0.46 s", one importance noul per headline across 6 feeds. Search:
  Google/Bing/Yahoo News results re-ranked by the average of P(relevant) and a 0–5 rating, 22–24 decisions in 0.23–0.62 s,
  shown next to the engines' order, with the top 8 feeding a cited short answer. It runs on JEV-27B or JEV-35B.
- `shorts.guru.so`: 8.9M English TikTok videos (`datasocial/tiktok-5.6B-videos`, CC BY-NC 4.0). Candidates come from a
  hand-written scoring function over an ≈ 11k pool plus `google/embeddinggemma-2` search. The ranking is **one choice
  question** over the covers and captions of the last 5 videos and up to 8 candidates, run on JEV-27B-VL, median 0.73 s per
  batch.
- The demos call a hosted API, `https://api.scienceguru.ai/v1/decide` (model `jev-27b-vl`, "one API key (120 decisions
  per minute)") (`BLOG` lines 202-206, 288).
- Offline numbers quoted:
  - MicroLens next-watch AUC 0.727 vs item-CF 0.728 (95% CI of the difference −0.041 to +0.040); the demo's own question
    gives 0.760 on 100 users.
  - MIND AUC 0.642 vs LightGBM 0.616.
  - TREC-COVID nDCG@10 0.858 vs bge-reranker-v2-m3 0.793.
  - One batch choice vs one noul per candidate: AUC 0.931 vs 0.905, 0.42 s vs 1.54 s.
- The claim that AutoTrust's approach replaces "at least 100 engineers" with "about 10" is "our estimate, not a
  measurement". The offline proxies "still need A/B tests" (`BLOG` Limitations).

---

## Serving

- **Not System One-compatible.** `POST /v1/decide {kind, state, question, options, strategy, thinking, threshold,
  think_budget, reasoning_effort, chat_template_kwargs, return_reasoning, debug, system2_only}` returns `{kind,
  effective_kind, options, probabilities, choice_index, choice, adaptation, protocol, model, usage{prompt_tokens,
  completion_tokens, total_tokens}, elapsed_seconds, num_model_requests[, thinking]}`. There is also `GET /v1/decide/info`
  (`HF:JEV-27B/serve_decide.py:8-33,346-414`). It takes one question per call, has no `questions` map, no `/v1/systemone`
  route, no `confidence` and no auth. The docstring calls it "the same shape as the hosted JEV API", meaning AutoTrust's own
  hosted API (inferred: `api.scienceguru.ai`, see `BLOG`).
- **Runtime.** `serve_decide.py` monkey-patches vLLM's `build_app` to add the router onto the standard OpenAI server, so
  `/v1/chat/completions` (System 2) and `/v1/completions` with model `jev-decision` (System 1) remain available. It needs
  a September 2026 vLLM dev build with `logprob_token_ids`; start-up takes 3–8 min of CUDA-graph capture with LoRA
  (`HF:JEV-27B/serve_decide.py:417-430`; `HF:JEV-9B/README.md:577-579`; `HF:VL/README.md:633-636`).
- **Other paths.**
  - Plain transformers + peft (merge in memory, read `last_hidden_state[0,-1]`) (`HF:JEV-9B/README.md:608-653`).
  - The Decision Index engine `submissions/jev/jev_engine.py` in the kit (`HF:JEV-Gemma4/README.md:94-95`).
  - HF `pipeline("decision")` via the `ArthurZ/*` ports, model type `jev` (`clones/hf-jev-model-type.md`).
  - v1.1: a kit engine `jev_vllm_engine.JevVllmEngine`, which fans one completion per question out over a thread pool and
    returns System One-shaped `answers` with `usage` set to 0 (`HF:v1.1/jev_vllm_engine.py:112-138`).
- **Throughput.** 205 decisions/s over HTTP at 256 clients (9B); 24 short text decisions in ≈ 1.4 s with the VL cap of 8
  sequences; 4,000 six-image decisions in 316 s (`HF:JEV-9B/README.md:561-567`; `HF:VL/README.md:633-634`).

---

## Licence

**What the cards say**

- Every repo's card tag is `apache-2.0`.
  - JEV-9B / JEV-27B: "Weights: Apache-2.0 (base model Qwen… Apache-2.0; training corpus `SargeDev/jev-distill-corpus-v3`
    is Apache-2.0, its `openjev_v2` stream additionally CC0)". They describe the model as "an independent student model
    trained on public data and shares no weights, code or affiliation with TypeSafe AI" (`HF:JEV-9B/README.md:782-787`;
    `HF:JEV-27B/README.md:917-922`). The card text says System 1 is "trained on [TypeSafe Jev 1.13's] published output
    distributions" (`HF:JEV-9B/README.md:196-200`).
  - JEV-27B-VL: Apache-2.0, "contains the weights of Qwen/Qwen3.8-27B … unchanged" plus the adapter (`HF:VL/README.md:655-659`).
  - GEV / JEV-Gemma4: "Apache-2.0 for the adapter, head and calibration files; base model under the Gemma 4 terms", with
    `license_link` to the Gemma 4 licence (`HF:GEV/README.md:579-582`, front matter).
  - JEV-35B: "Apache-2.0. A fine-tune of Qwen/Qwen3.5-35B-A3B (Apache-2.0)". There is no mention of TypeSafe or of the
    training data (`HF:35B/README.md:414-416`).
  - v1.1: Apache-2.0, with `LICENSE.base-apache-2.0` and `NOTICE.base` crediting d3 / vLLM Semantic Router. "This license
    statement … does not change the licenses or usage restrictions of third-party datasets" (`HF:v1.1/README.md:190-194`).
- Corpus card: Apache-2.0, with an attribution request ("credit it in your final product"). It labels the yuri_v3 stream
  "distilled from Jev 1.13 (TypeSafe) via OpenRouter" and has no section on upstream terms (`DS:corpus/README.md:46,101-113`).
- Third-party data in demos: `datasocial/tiktok-5.6B-videos` is CC BY-NC 4.0, and MIND has Microsoft Research terms
  (`BLOG` Limitations). GEV/35B training uses benchmark train splits under their own licences (not listed).

**What TypeSafe's terms say** (quoted, `TS:MCA`, Master Customer Agreement, "Last updated Sep 23, 2026", §2.3 License
Restrictions): "Customer will not do (and will not attempt to do) … (b) use the Services or any Output (defined below) to
perform model distillation, train a model to imitate the output of the Services, or develop (or to facilitate the
development of) a similar or competing product or service; (c) reverse engineer … or attempt to … derive … the underlying
ideas, algorithms, structure, or organization …". §4.2 separately says "TypeSafe disclaims ownership of Output" and
assigns Output to the customer. The site Terms of Use (2026-09-19) cover the website only; they ban reverse-engineering and
scraping of the Site and say nothing about model outputs (`notes/web/typesafe/t_terms.html`).

**Facts relevant to which terms applied, without judgement**

- The corpus says the labels were obtained through **OpenRouter**, not `api.typesafe.ai`. OpenRouter lists TypeSafe as
  the provider of `typesafe/jev-1.13` (Decisions API `POST /api/alpha/decisions`). Its Jev guide page shows no terms and
  defers "data policy" to the model page ([guide](https://openrouter.ai/docs/guides/community/jev)); the model pages
  returned 404 to our fetch. Whether the MCA, OpenRouter's terms, or provider pass-through terms governed those calls is
  not established by anything we fetched.
- Dates: Jev launched 2026-09-15; corpus v3 was created 2026-09-21; the MCA text above is dated 2026-09-23; JEV-9B was
  created 2026-09-23. Earlier MCA versions were not retrieved.
- No AutoTrust card, the corpus card or the HF transformers PR discusses TypeSafe's terms (`clones/hf-jev-model-type.md`
  Licence: "How that squares with TypeSafe's terms is not addressed anywhere"). We found no public statement by TypeSafe
  about these models.

---

## Notable design choices vs Jev

1. **They copy the teacher, not the outcome.** It is the only family whose main target is Jev's distribution. KL ≈ 0.017
   on in-domain rows is the headline, and "a faithful copy of System 1 is also a faithful copy of its blind spots" (the
   poker shove, 0.70 vs Jev's 0.62) (`HF:JEV-9B/README.md:337-340`). The Decision Index ECE (0.072–0.085) mirrors Jev's
   own 0.078 (`DIX`).
2. **No "state once" structure.** Each question is a separate prompt with the state re-encoded, and `/v1/decide` takes
   one question. Jev bills and encodes the state once for many questions. Prefix caching only helps within one question
   kind.
3. **A narrower type surface than Jev.** Score is fixed at 6 levels with no level texts; noul has no true/false
   descriptions; choice has 16 trained slots (256 on JEV-35B) against Jev's 2–255 with option descriptions. There is no
   box, no `confidence` field, and no System One wire format.
4. **The verbalizer head at the last token is close to "LM logits on label tokens".** Initialising from `lm_head` rows
   means the student starts as the zero-shot LM. Jev's readout is unknown (`clones/jev-reference.md` §3). This is exactly
   HF's `jev` head type.
5. **Hybrid System 1 / System 2 from one engine** (frozen backbone, LoRA only on the decision path) and adaptive thinking
   that folds p2 into p1 at 50/50. Jev bans generation entirely.
6. **Vision for free.** The text-trained adapter is put on the original VLM, and image decisions are zero-shot. Jev 1.14
   added images as a "dreamer-ready" option. JEV-27B-VL tops the vision board.
7. **Long context comes from the backbone.** 262K tokens are accepted, while training rows are ≤ 856 tokens; Jev has
   64k/32k by default and 262k behind a header.

## Notable design choices vs kev

1. **Labels:** kev uses hard labels from public data and code and explicitly **no Jev outputs**. JEV uses soft Jev
   distributions with a KL loss (+ RPS). kev's later soft-target support never shipped (`clones/kev.md` Training).
2. **Readout:** kev scores a `<decide>` query against `</opt>` keys (pointer, any K ≤ 255, one head for all types). JEV
   uses a fixed-slot linear verbalizer per type (2 / 6 / 16 or 256). kev-style options can be arbitrarily many with no
   untrained labels; JEV past 16 relies on untrained labels or a tournament.
3. **Same LoRA, same base family.** Both use r16/α32 over the attention, MLP and Gated-DeltaNet projections of
   Qwen3.5/3.8 hybrids, frozen backbone. kev-27B later switched to full fine-tuning; JEV stays with LoRA and keeps the
   backbone pristine so System 2 survives.
4. **Isolation:** both run one causal row per question on hybrid bases. kev's server runs the state once and continues
   rows from a cached replica. JEV re-encodes the state per question and relies on vLLM block prefix caching, which its
   `[kind]`-first template limits.
5. **Delimiters:** kev reuses Qwen special tokens and escapes user `<|…|>`, so options are unforgeable. JEV uses
   plain-text `[kind]`/`A) `/newline markers with no escaping (inferred: forgeable).
6. **Calibration:** both use a scalar temperature. kev's (≈ 1.3–2.4) is fitted against ground truth on held-out
   datasets. JEV's (≈ 1.0) is fitted against the teacher, so it inherits Jev's over-confidence. GEV's
   `calibration_gold.json` is the only outcome-fitted variant.
7. **API:** kev serves TypeSafe's `/v1/systemone` (SDK works with a changed base URL). JEV serves its own `/v1/decide`.
8. **Evaluation:** kev uses frozen hashed suites with a locked test, and Jev is scored on the same items. JEV evaluates
   mainly on the teacher's own held-out rows (same 53 domains) plus re-runs of public sets. GEV and JEV-35B train on the
   train splits of benchmarks used on the board.
