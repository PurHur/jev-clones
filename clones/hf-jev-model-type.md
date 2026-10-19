# HF transformers `model_type: "jev"` and `pipeline("decision")` (Arthur Zucker)

Hugging Face is building a shared runtime for Jev-style decision models inside `transformers`. It adds a `decision`
pipeline, a `/v1/systemone` endpoint in `transformers serve`, a repo-level `decision_config.json` +
`decision_template.jinja` format, and four **decision-head model types** (`jev`, `kev`, `von`, `clef`). Each head is
named after the first model that used it, sits on top of an unmodified backbone loaded from the original repo, and
reads hidden states at positions the template marks with `<|mark|>`. `ArthurZ/*` on the Hub holds the converted
repos. None of this is in a transformers release: it is a **draft PR**.

Source pins used for this profile:

| ref | what | where |
|---|---|---|
| `T` = branch `add-jev-models` @ **`92b46a8f`** (2026-10-19 09:14 UTC, "document the main contribution better") | PR [huggingface/transformers#49566](https://github.com/huggingface/transformers/pull/49566), shallow sparse clone, read only, never executed | `/home/ai/projects/jev-clones/sources/transformers` |
| `main` @ `5be474e9` (2026-10-19) | for comparison: **no `models/jev/` on main** | same clone, `FETCH_HEAD` |
| PR metadata, full diff (5,454 lines), 74-commit log | `gh pr view/diff` | `notes/hf-raw/_transformers_pr49566.json`, `_transformers_pr49566.diff.txt`, `_transformers_pr49566_commits.txt` |
| HF | API JSON (`?expand[]=safetensors,…`), recursive tree, every small text file of the 21 `ArthurZ/*` decision repos (no weights) | `notes/hf-raw/ArthurZ__*/` |
| HF (sources of the ports) | `autotrust/GEV-26B-Decide`, `autotrust/JEV-9B`, `autotrust/JEV-27B` (from the earlier sweep) | `notes/hf-raw/autotrust__*/` |

Citation form: `T:src/transformers/models/jev/modeling_jev.py:54` is file:line at `92b46a8f`. `HF:ArthurZ/d3/config.json`
is a fetched file under `notes/hf-raw/ArthurZ__d3/files/`.

> **Main findings.**
> 1. **"JEV" and "GEV" are not TypeSafe's Jev.** They are **AutoTrust's** `autotrust/JEV-*` models (Qwen3.5/3.8 LoRA
>    students distilled from TypeSafe Jev 1.13) and `autotrust/GEV-26B-Decide` (the same recipe on Gemma-4-26B-A4B).
>    The head type `jev` is named after `autotrust/JEV-9B`, the first model whose readout it implements
>    (`T:…/jev/configuration_jev.py:22`, `T:docs/source/en/model_doc/jev.md:20`, commit `ffa7f5d7` "Decision heads as
>    models named after the first model that used them").
> 2. **`model_type: "jev"` is only one of four heads.** It is the *linear verbalizer* head: one `Linear(d, V)` read at the
>    token before each `<|mark|>`, with per-type row offsets. It serves AutoTrust JEV/GEV (24 or 264 slots) and every
>    **255-code readout** (pplx-decider, d3, Darwin-ZTC, PhysicalEye). **kev-4b is `model_type: "kev"`** (pointer head),
>    **clef-flash is `"clef"`** (joint schema head) and **von is `"von"`** (MLP over `[MASK]` markers).
> 3. **The shared abstraction is a template + marks + flat scores contract**, not a shared architecture. What is fixed:
>    the three question types, per-question softmax over options, the Jev answer formulas (`noul` = p(true), choice
>    and score confidence), batch size 1, no KV/prefix reuse. Everything else goes through config: the jinja template,
>    labels, per-type temperature, the head class and the backbone.
> 4. **For Zeiger export:** the request/answer side fits as is. The head does not fit: Zeiger's set-transformer readout is
>    neither `jev`, `kev` nor `von`. Exporting would mean a new `zeiger` head model type (~60 lines, kev-shaped), plus
>    explicit `labels` to get past a **default 26-option cap**. Our option-count temperature buckets and chunked long
>    pages cannot be expressed in the format.

---

## Identity

| | |
|---|---|
| PR | [#49566](https://github.com/huggingface/transformers/pull/49566) "add support for jev like models", **DRAFT, open**. Branch `add-jev-models`, head `92b46a8f`. Opened 2026-10-13, updated 2026-10-19. 74 commits, 71 files, +3,809 / −411 (`_transformers_pr49566.json`) |
| Author | Arthur Zucker (`ArthurZucker` on GitHub, `ArthurZ` on HF), transformers core maintainer |
| Depends on | [#47386](https://github.com/huggingface/transformers/pull/47386) "Add chat input sanitization" (Matt / Rocketknight1, merged 2026-10-13), which supplies the `sanitize_control_tokens` argument of `apply_chat_template` (PR body: "Needs #47386") |
| Version string | `transformers_version: "5.19.0.dev0"` in every converted config. The code is **not on main** at `5be474e9` and in no release |
| Reviews | Only bots so far (doc builder, run-slow CI on AMD + Nvidia for `models/{auto,clef,jev,kev,von}`, and an "mlinter" pass flagging `nn.Sequential` and similar). No human review recorded (`gh api …/reviews`, `…/comments`) |
| Commit history | 09-22 … 10-08: prototype ("alternate approach", control-token sanitization). 10-13: the pipeline, `DecisionConfig`, the four heads, Hub decision files. 10-15: heads own their backbone via `backbone_config`; sub-models load from the repo named in their config. 10-16: code-readout conversion (pplx, d3, Darwin, PhysicalEye), fp32 readout, LoRA JEV conversions. 10-19: cleanup, docs (`_transformers_pr49566_commits.txt`) |
| Public API added | `transformers.DecisionConfig`, `DecisionPipeline`, `pipeline("decision")`, `CalibratedTemperatureLogitsWarper`, `NoulBandLogitsWarper`, `JevConfig/JevDecisionHead`, `KevConfig/KevDecisionHead`, `VonConfig/VonDecisionHead`, `ClefConfig/ClefDecisionHead` (PR diff `src/transformers/__init__.py`; `models/auto/modeling_auto.py` registers all four heads under **`AutoModel`**) |

### What "JEV" and "GEV" refer to

| name in the PR | actually | evidence |
|---|---|---|
| **JEV** (model type and head) | `autotrust/JEV-9B`, `JEV-27B`, `JEV-27B-VL`, `JEV-35B`, `JEV-Gemma4-26B-A4B`: AutoTrust's "System 1" models, a LoRA on Qwen3.5-9B / Qwen3.8-27B / Qwen3.5-35B-A3B / Gemma-4. The card calls JEV-9B the "student of TypeSafe Jev 1.13", trained on `SargeDev/jev-distill-corpus-v3`; 25,376 of its 29,955 test targets are Jev 1.13 distributions | `HF:autotrust/JEV-9B/README.md` model-index; `T:…/jev/convert_jev_to_hf.py:15-20` |
| **GEV** | `autotrust/GEV-26B-Decide`: the same recipe on `google/gemma-4-26B-A4B-it` (LoRA r32, alpha 64), with "System 2" adaptive thinking on the unmodified Gemma as a fallback. Card: Decision Index 0.2.1 62.48 (own scoring) | `HF:autotrust/GEV-26B-Decide/README.md`, `judge_config.json` (`model_name "gev-26b-a4b"`) |
| "System 1" | AutoTrust's name for the one-pass decision path: "decision logits = lm_head-LoRA logprobs of verbalizer ids + bias, then per-kind temperature" | `HF:autotrust/GEV-26B-Decide/adapter_vllm/decision_head.json:note` |
| `bare-v1` | AutoTrust's prompt protocol (`serve_decide.py: PROTOCOL = "jev27-bare-v1"`), which the `jev` template reproduces | `T:…/jev/convert_jev_to_hf.py:38-59` |

The HF `jev` docs say the head "was introduced by JEV and is also used by GEV" (`T:docs/source/en/model_doc/jev.md:20`).
TypeSafe's Jev appears only as the API being imitated: "System One API", `/v1/systemone`, the
`api.typesafe.ai/openapi.json` link (`T:src/transformers/cli/serving/systemone.py:15,50`).

### Converted Hub repos (`ArthurZ/*`, fetched 2026-10-19)

| repo (sha) | created | head type | backbone (loaded from) | layout | head params | decision config |
|---|---|---|---|---|---|---|
| `JEV-9B` (`36f61554`) | 10-13 | jev, 24 slots | `autotrust/JEV-9B` + root LoRA r16 | **old**: `decision_head/` + root `adapter_model.safetensors` | `decision_head/model.safetensors` 393 kB | T noul 1.002 / choice 0.984 / score 1.012 |
| `GEV-26B-Decide` (`aa61647e`) | 10-13 | jev, 24 slots, **softcap 30** | `google/gemma-4-26B-A4B-it` + root LoRA r32 | old | 271 kB | T 1.003 / 1.017 / 0.999 |
| `JEV-27B` (`396862d0`), `JEV-27B-VL` (`a901c5a7`) | 10-16 | jev, 24 slots | `autotrust/JEV-27B(-VL)` + `backbone/` LoRA r16 (435 MB) | **new**: head at root + `backbone_config` | F32 122,904 | T 1.014 / 1.016 / 1.004 |
| `JEV-35B` (`605d5c65`) | 10-16 | jev, **264 slots** (2 + 6 + 256) | `autotrust/JEV-35B` (Qwen3.5-35B-A3B MoE, LoRA merged) | new | F32 540,936 | T 0.984 / 1.012 / 1.008 |
| `JEV-Gemma4-26B-A4B` (`5077fa08`) | 10-16 | jev, 24, softcap 30 | `autotrust/JEV-Gemma4-26B-A4B` + `backbone/` LoRA | new | F32 67,608 | T 1.003 / 1.017 / 0.999 |
| `pplx-decider-v1.1-27b` (`cd9d6371`) | 10-16 | jev, 255 codes | `perplexity-ai/pplx-decider-v1.1-27b`, **bidirectional full-attn** | new | BF16 1,305,855 | T 1.009 (all types) |
| `pplx-decider-v1-27b` (`fbbe3f28`) | 10-16 | jev, 255 | `perplexity-ai/pplx-decider-v1-27b`, **causal** | new | BF16 1.31M | T 2.208 |
| `d3` (`84cb803f`), `d3-flash`, `-mini`, `-nano`, `-lite`, `-edge` | 10-16 | jev, 255 | `vllm-sr/d3*`, bidirectional, fp32 readout | new | F32 1.31M (d=5120) … 261k (d=1024) | T 1.0 |
| `Darwin-27B-ZTC`, `-v2`, `-v5` | 10-16 | jev, 255 | `FINAL-Bench/Darwin-27B-ZTC*`, causal | new | BF16 1.31M | T 1.069 / 1.219 / 1.311 |
| `PhysicalEye-Decide-35B` (`ab8eb878`) | 10-16 | jev, 255 | `masterset-ai/PhysicalEye-Decide-35B` (Qwen3.5 MoE), causal | new | BF16 522k | T 1.344 |
| `kev-4b` (`76c6336d`) | 10-13 | **kev** (head_dim 256) | `jaredpalmer/kev-4b` (main = Qwen3.5-4B Kev 1.0) | old | 5.2 MB | T 2.406 (all types) |
| `von` (`bc0e623c`) | 10-13 | **von** | `wfzyx/von` (ModernBERT) | old | 2.1 MB | calibrated T + noul band (below) |
| `clef-flash` (`3f9f19b6`) | 10-13 | **clef** | `Cloudflare/clef-flash` (Qwen3.5-9B) | old | 243.5 MB bf16 | none; `sort_choices: true` |

`backbone_config` of the code readouts records `use_bidirectional_attention` (true for d3*, pplx-v1.1; false for
pplx-v1, Darwin, PhysicalEye: `HF:ArthurZ/*/config.json:backbone_config.text_config`). The four new-layout AutoTrust
ports (`JEV-27B`, `JEV-27B-VL`, `JEV-35B`, `JEV-Gemma4-26B-A4B`) have no licence tag on the Hub.

---

## Architecture

### Layering

```
request {state, questions{name: {type, instructions, criteria}}, assets?}
   |
   |  DecisionTemplate (decision/template_utils.py:320)
   |    parse_questions -> Question(name, type, options, labels, descriptions)   :138-191  (validated, cached LRU 256)
   |    jinja = decision_template.jinja  (or DECISION_PROMPT inside the model's own chat template)  :113-134
   |    one prompt per question            <- template uses `question`
   |    one prompt for all questions       <- template uses `questions`         :348, :408
   |    apply_chat_template(..., sanitize_control_tokens) ; drop <|mark|>/<|split|>   :426-452
   v
DecisionPrompt(input_ids, marks, questions, backbone_kwargs)                       :226-251
   marks[i] = number of kept tokens before the i-th <|mark|>   (no mark -> [len(input_ids)])
   |
   |  DecisionPipeline._forward (pipelines/decision.py:109-121), one prompt at a time, batch 1
   +-- model.config has backbone_config  ->  head(input_ids, marks, num_options, question_type_ids, backbone_kwargs)
   |                                           returns ONE flat vector: the option scores of every question, in order
   +-- otherwise (any chat LM)           ->  lm(...).logits at marks-1, gathered at the label token ids
   v
split by num_options -> per question: LogitsProcessorList[type] (temperature, calibrated T, noul band)   :123-130
   -> softmax over that question's options -> Question.answer(probabilities)        template_utils.py:193-213
   v
answers {name: {type, noul | choice+probabilities+confidence | score+legend+probabilities+confidence}}
```

The **only interface a head implements** is
`forward(input_ids [L], marks [M], num_options [Q], question_type_ids [Q], backbone_kwargs) -> scores [Σ num_options]`.
The template decides where marks sit and what they mean. The head runs its own `self.backbone` (`T:…/jev/modeling_jev.py:39,52`).

### Backbone

- Any `AutoModel` built from `config.backbone_config` (`T:…/jev/modeling_jev.py:39`; default sub-config type `qwen3_5_text`,
  `T:…/jev/configuration_jev.py:42`). In the converted repos the backbones are Qwen3.5 / Qwen3.8 dense hybrids
  (`Qwen3_5Model`, `Qwen3_5TextModel`, `Qwen3_5ForCausalLM`), Qwen3.5 MoE (`Qwen3_5MoeModel`,
  `…ForConditionalGeneration`), Gemma 4, and ModernBERT for von.
- **The repo holds only the head.** `from_pretrained` sees that the checkpoint has no `backbone.*` keys, so it resolves
  the backbone's weights from `backbone_config._name_or_path`, i.e. the original repo, and loads them under the
  `backbone.` prefix (PR diff `modeling_utils.py`, `from_pretrained` "+Sub-models without weights in the checkpoint are
  read from the repo saved in their config"). A PEFT adapter in a subfolder named after the sub-model
  (`backbone/adapter_config.json`) is then attached to it ("…find their adapter … in the subfolder named after them").
  - That lookup passes **`"revision": None`**, so the backbone always comes from the original repo's **main**. It is
    not pinned. Inferred consequence: `jaredpalmer/kev-4b` main changed from Qwen3-4B to Qwen3.5-4B in Kev 1.0 (see
    `clones/kev.md`), and a moving upstream would silently change what an `ArthurZ` head reads. The decision tests pin
    a tokenizer revision (`decision_test.json:revision`), not the weights.
- **Qwen3.5 gains `use_bidirectional_attention`** (default false) on `Qwen3_5TextConfig` and the MoE config: the
  full-attention layers OR a bidirectional mask into the causal one, and the Gated-DeltaNet layers stay causal
  (`T:src/transformers/models/qwen3_5/configuration_qwen3_5.py:41-43,112`, `modeling_qwen3_5.py:1269-1271`). This is
  the "noncausal full-attention" mode of pplx-decider v1.1 and d3. `convert_decider_to_hf.py:93` sets it from
  `decision_config.json:attention_mode == "noncausal_full_attention"`.
- `keep_in_fp32_modules` becomes a common `PreTrainedConfig` field and is merged into `_keep_in_fp32_modules_strict`
  (PR diff `configuration_utils.py`, `modeling_utils.py:post_init`). The jev ports use `["^proj"]`, so the 255-row
  readout runs in fp32 on a bf16 backbone "as d3 runs it" (commit `2fdb4c97`).
- Nothing is trained in this PR. These are **inference ports** of existing weights.

### The four heads

| model_type | named after | trained params (example) | reads | score per option |
|---|---|---|---|---|
| **`jev`** | autotrust/JEV-9B | `proj: Linear(d, V)` (+bias) | hidden state at `marks − 1`, **one mark per question** | row `offsets[type] + j` of `proj(h)`, optional softcap |
| **`kev`** | jaredpalmer/kev-4b | `key, query: Linear(d, 256)` | one mark after each option + one after the question | `key(h_opt) · query(h_q) / √256` |
| **`von`** | wfzyx/von | `LN → Linear(d, d/2) → GELU → LN → Linear(d/2, 1)` | hidden state of each option's `[MASK]` (mark − 1) | MLP scalar; zero-shot noul debiased by a state-free block |
| **`clef`** | Cloudflare/clef | 243 MB: 2 evidence cross-attn layers + 4 decoder layers (d 1024, 16 heads) + scorer | pooled **spans** between mark pairs, for all questions in one prompt | prior·lexical cosine + gate·(cosine + MLP) against the question's field |

#### `jev`: linear verbalizer head (`T:src/transformers/models/jev/modeling_jev.py:35-61`)

```
h_q   = backbone(input_ids)[marks_q − 1]                       one row per question (:52-54)
z     = W h_q + b              W: [V, d]  (V = num_verbalizers)  (:38, :54)  in fp32 if keep_in_fp32_modules
z     = c · tanh(z / c)        if softcap c is set (Gemma: 30)   (:55-56)
s_q,j = z[offsets[type_q] + j],   j = 0 .. num_options_q − 1     (:58-61, a boolean mask over rows)
```

- **Why a linear layer reproduces "lm_head log-probs of the slot tokens":** the log-softmax normaliser over the
  vocabulary is the same for every slot, so it cancels in the per-question softmax. The head therefore keeps only the
  `lm_head` rows of the slot tokens, with any `lm_head` LoRA folded in as `W += (α/r)·B[ids]·A`, plus AutoTrust's
  per-slot bias, or AutoTrust's trained `head.safetensors` when it exists (`T:…/jev/convert_jev_to_hf.py:15-20,70-87`).
  For code readouts, `W` is `readout.safetensors:weight` and `b = 0` (`convert_decider_to_hf.py:106-107`).
- **Verbalizer layout differs by family** (from `offsets` + `decision_config.labels`):

```
AutoTrust JEV-27B / JEV-9B / GEV  (num_verbalizers 24, offsets noul 0, score 2, choice 8)
 row:   0      1      2   3   4   5   6   7   8   9  ... 23
 tok:  false  true   "0" "1" "2" "3" "4" "5"  A   B  ...  P
       '-noul-'      '------ score ≤6 -----'  '-- choice ≤16 --'
AutoTrust JEV-35B: same, choice block is 256 wide (A..), num_verbalizers 264

Code readouts (pplx-decider, d3*, Darwin-ZTC*, PhysicalEye; num_verbalizers 255, all offsets 0)
 row:   0  1  2 ... 25  26  27 ... 254
 tok:   A  B  C ...  Z  AA  AB ... JT        one code table shared by all three types
        noul = rows 0,1 (A = false, B = true); score = rows 0..k-1; choice = rows 0..k-1
```

  The 255 codes are `A..Z` followed by two-letter codes, **skipping 25 pairs** such as BQ, BZ, CJ, CQ, EY and JF, up to JT
  (`HF:ArthurZ/d3/decision_config.json:labels`; inferred: the skipped pairs are the ones that are not single Qwen tokens,
  since `label_token_ids` requires single tokens, `T:decision/template_utils.py:216-222`).
- One question = one mark. The code-readout template writes **no** `<|mark|>` at all, so `marks = [len(input_ids)]` and
  the head reads the **last prompt token** (`T:decision/template_utils.py:451-452`; `HF:ArthurZ/d3/decision_test.json`
  marks `[119]` for a 119-token prompt). The AutoTrust template ends `"\n[decision]:<|mark|>"`, so it reads the `:`
  token (`T:…/jev/convert_jev_to_hf.py:58`).

#### `kev` (`T:src/transformers/models/kev/modeling_kev.py:38-59`)

`blocks = split(h[marks − 1], num_options + 1)`, then `scores = key(block[:-1]) @ query(block[-1]) / √head_dim`. This is
kev's `PointerHead` (`</opt>` keys against a `<decide>` query; see `clones/kev.md` Readout head). The template rebuilds
`kev.api.to_record` + `encode` as **one causal row per question**:
`<|fim_prefix|>state<|fim_middle|>instr (<|box_start|>opt<|box_end|><|mark|>)* <|fim_suffix|><|mark|>`, with kev's
`user_tokens` escaping (`<|x|>` → `<¦x¦>`) done in jinja (`HF:ArthurZ/kev-4b/decision_template.jinja`). That is kev's
row form, so isolation is exact. kev's packed block-causal mask is not used.

#### `von` (`T:src/transformers/models/von/modeling_von.py:47-122`)

One sequence per question: `[CLS]{question} {state} [SEP] [MASK]<|mark|> opt0 [MASK]<|mark|> opt1 …[SEP]<|mark|>`.
The head builds its own masks with the masking API: blocks are mutually invisible, each option attends to the prompt and
to itself, option positions restart where the first option starts, and the sliding window is measured in those
positions (`:62-106`). This is the only head here that builds per-option isolation. A zero-shot `noul` (no criteria)
gets a **second block without the state**, and the "true" score is lowered by `a·(null_true − null_false) + b` with
`a = −0.5, b = 0.3` (`:118-121`, `HF:ArthurZ/von/decision_head/config.json`). The raw state text is written once more
after the last block, purely so that the calibrator can count its tokens. The head drops it before the backbone (`:71-72`).

#### `clef` (`T:src/transformers/models/clef/modular_clef.py:137-308`)

Clef is the only **joint, cross-question** head. Its template renders **one prompt for the whole request** (it uses the
`questions` variable) and marks a start/end pair around each instruction and each option
(`HF:ArthurZ/clef-flash/decision_template.jinja`). The head works in four steps:
1. Span means of the normalised hidden states and of the backbone's **output embeddings** give, per option, a
   context vector, a lexical vector and its question's vector. A lexical prior is computed as
   `cos(lexical_opt, question + h_last)` (`:156-194`).
2. Two evidence layers let the options cross-attend to the projected prompt memory (`:220-221`). Each question
   "field" then routes over its own options by dot-product softmax, with other questions' options masked (`:223-227`).
3. Four non-causal decoder layers let the **fields attend to each other** and to the prompt (`:128-134, 228-230`).
4. `score = exp(prior_scale)·prior + σ(gate)·(exp(joint_scale)·cos(field, opt) + MLP[f, o, f·o, |f−o|])` (`:244-253`).

Original-checkpoint weight names are mapped by a `conversion_mapping["clef"]` entry, not by a script (PR diff
`conversion_mapping.py`).

### Input encoding and the config schema

Two files sit next to the model, both loaded by `DecisionConfig.from_pretrained` (`T:src/transformers/decision/configuration_utils.py:30-104`):

| `decision_config.json` field | default | meaning | used by |
|---|---|---|---|
| `labels{type: [str]}` | noul `Yes`/`No`, else `A..Z` | single-token label per option slot; **also the per-type option limit** | the template (shows them) and the LM-logit path (reads them) |
| `temperature{type: T}` | 1.0 | `TemperatureLogitsWarper(T)` before the softmax | all |
| `calibration{bias, entropy, log_tokens, n_options, lo, hi}` | none | input-conditioned T (below) | von |
| `noul_band[edge, slope]` | none | squashes p(true) into a band away from 0.5 | von |
| `noul_options` | `["true","false"]` | order of the two noul options handed to the template and head | jev/kev ports use `false,true`; von/clef `true,false` |
| `sort_choices` | false | sort choice keys before rendering | clef only (true) |
| `sanitize_control_tokens` | true | encode control tokens written in the request as text; templates that escape them themselves (kev, von) set false | all |
| `chat_template` | `DECISION_PROMPT` in the model's chat template | stored as `decision_template.jinja` | all |

Template variables: `state` (string or JSON, or a list of text/image parts when `assets` placeholders are found), and
`question` (`name, type, instructions, options, labels, descriptions, legend`) **or** `questions` (a list). The sentinels
`<|mark|>` (a head reads at `mark − 1`) and `<|split|>` (tokenise the text in pieces, as the original code did) are
added as special tokens and removed from the ids (`T:decision/template_utils.py:337-348, 438-452`). Requests cannot
write either one: JSON-string requests are parsed first so that `\u` escapes cannot hide a sentinel, and control
tokens are sanitised (`:375-387`; `tests/pipelines/test_pipelines_decision.py:79-105`).

Every converted repo ships `decision_test.json`, which holds a request plus the `input_ids` and `marks` that **the
original model's own code** builds for it. Tests assert that the jinja template reproduces them token for token
(`T:tests/test_decision_head_common.py:56-58`, `T:tests/models/jev/test_modeling_jev.py:117-127`). This is the
mechanism that keeps a port faithful to its original prompt format.

### Attention and isolation

- **Isolation comes from separate prompts, not masks.** A template that uses `question` gets one prompt per question,
  and the pipeline runs one forward per prompt, batch 1, without a cache (`use_cache=False`) (`T:pipelines/decision.py:109-121`;
  `# TODO: batch / continious batching` at `:110`). So questions are exactly independent, but **the state is re-encoded
  for every question**. Jev's "state ingested once" is not implemented. Neither is kev's cached-prefix serving.
- Within a prompt, attention is whatever the backbone does: causal (AutoTrust JEV/GEV, pplx-v1, Darwin, PhysicalEye,
  kev), causal-DeltaNet plus bidirectional full-attention layers (d3*, pplx-v1.1), von's custom option-isolation
  masks, or clef's joint prompt.
- **clef breaks Jev's isolation guarantee by design.** All questions share one prompt and the fields attend to each
  other (`modular_clef.py:131`, "Not causal: the questions attend to each other"), so adding a question can change
  another's answer.

### Readout per type (fixed in `Question.answer`, `T:decision/template_utils.py:193-213`)

| type | options passed to template / head | answer | confidence |
|---|---|---|---|
| `noul` | `noul_options` order, descriptions from `criteria.true/false` or `None` (the template supplies fallbacks such as "Yes / true") | `p[true]` after a **2-way softmax** (no sigmoid) | none |
| `choice` | criteria keys in request order (or sorted) | argmax key + `probabilities` | `(p_max − 1/n)/(1 − 1/n)`, Jev's formula (`:201`) |
| `score` | levels as `"0".."k-1"`, legend = level texts | `Σ i·p_i` (expected index, not normalised) | `max(0, 1 − Σ p_i·|i − argmax| / MAD_unif)`, MAD_unif = mean |i − (k−1)/2| (`:204-206`), Jev's formula |

Probabilities are not rounded. Score has no ordinal head; for code readouts the levels are just the first k codes.

### Calibration

- Per-type temperature for every port, taken from the source: AutoTrust `calibration.json:per_kind`
  (`convert_jev_to_hf.py:119`), or one `readout_config.temperature` for all types in code readouts
  (`convert_decider_to_hf.py:120`). The shipped values are close to 1 for JEV/GEV/d3/pplx-v1.1, 2.2 for pplx-v1 and
  2.41 for kev-4b (table above).
- `CalibratedTemperatureLogitsWarper` (`T:src/transformers/generation/logits_process.py:306-344`):
  `T = clamp(bias + entropy·H/log K + log_tokens·log10(n)/4 + n_options·K/8, lo, hi)`. H is the entropy of the raw
  softmax and n is the length of the `input_ids` handed to the processor. The pipeline passes only the tokens **after
  the last mark** (`T:pipelines/decision.py:126-129`), which is why von's template writes the state after its last
  block. For every other template n = 0, so log10(max(0, 1)) = 0 (inferred from the code). von ships
  `{bias 2.0151, entropy −3.2369, log_tokens 11.4916, n_options −3.5574, lo 0.3, hi 12}`.
- `NoulBandLogitsWarper` (`logits_process.py:347-371`): `p' = edge + slope·(p − 0.5)` above 0.5, mirrored below, then
  log. von ships `[0.8, 0.1]`, so every von noul lands in [0.15, 0.25] ∪ [0.75, 0.85] (arithmetic from the formula). It is
  a confidence floor and ceiling, not a calibration map.
- No per-option-count buckets, no histogram binning, no conformal sets.

### Limits

- **Options per question = `len(labels[type])`**, checked in `Question.from_request` (`T:decision/template_utils.py:183-188`):
  - AutoTrust JEV-27B/9B/GEV: **16 choice options, 6 score levels**. JEV-35B: 256 choice. Code readouts: 255 for all types.
  - **Heads that ignore labels (kev, von, clef) still inherit the default `A..Z`**, because their decision configs leave
    `labels: {}`. They are therefore capped at **26 options**, although kev itself allows 255 (inferred from
    `:183`; the code was not run).
- Context: whatever fits in one forward of the backbone. No truncation, chunking, or `max_tokens` check in the pipeline.
- `transformers serve /v1/systemone`: at most **64 questions** and **4 assets** per request (`T:src/transformers/cli/serving/systemone.py:57,59`),
  no remote image URLs (serving doc).

---

## Training

None. The PR contains only conversion code:
- `convert_jev_to_hf.py`, for AutoTrust checkpoints. It takes either `head.safetensors` + `adapter/`, or the `lm_head`
  slot rows + `adapter_vllm/` LoRA (with `lm_head` LoRA folded into `proj`), and saves the backbone LoRA in
  `backbone/` with `lm_head` dropped from the targets (`T:…/jev/convert_jev_to_hf.py:67-123`).
- `convert_decider_to_hf.py`, for 255-code readouts. It refuses exports whose `pooling` is not `last`, picks the
  `pplx` or `d3` system prompt and closing line, and sets the image budget (pplx 2048², d3 1,638,400 px) (`:65-123`).
- `kev`/`von`/`clef` were converted from `head.pt`, `option_marker.pt` and `joint_head.safetensors` (READMEs). Their
  scripts are not in the PR; clef's commit message says "conversion mapping instead of a conversion script".

What the source models were trained on is in their own profiles (`clones/kev.md`, the sweep's §3.1-3.3 and §3.16).

## Evals

There are no benchmarks. The tests are faithfulness and regression checks only:
- **Template fidelity**: for all 12 code-readout ports and each head family's representative repo, the jinja output
  equals the original code's `input_ids`/`marks` (`T:tests/models/jev/test_modeling_jev.py:117-127`,
  `T:tests/test_decision_head_common.py:56-58`).
- **Expected answers** (`@slow`, accelerator, atol 1e-2) on one shared request: refund noul, spam noul, a 3-way tone
  choice, a 4-level urgency score. Excerpt (`test_modeling_jev.py:41-114, 133-148`):

| port | refund p | spam p | tone (calm/angry/sad) | urgency argmax (p) |
|---|---|---|---|---|
| pplx-decider-v1.1-27b | 0.9993 | 0.0003 | .13 / **.80** / .07 | 2 (0.78) |
| d3 | 0.9908 | 0.0006 | .14 / **.81** / .05 | 2 (0.72) |
| d3-mini (4B) | 0.9793 | 0.0003 | .33 / **.51** / .16 | 2 (0.55) |
| d3-edge (0.6B) | 0.9871 | 0.0101 | **.51** / .23 / .26 | 3 (0.36) |
| Darwin-27B-ZTC-v5 | 0.9569 | 0.0171 | .451 / .451 / .098 (tie) | 2 (0.89) |
| PhysicalEye-Decide-35B | 0.9962 | 0.0066 | .03 / **.91** / .06 | 2 (0.85) |
| JEV-9B (own request) | intent refund 0.9999 | urgent 0.5419 | – | anger 2 (0.52) |

- The tiny-config tests check head save/load round trips and that a head saved without backbone weights reloads its
  backbone from the path in `backbone_config` (`test_decision_head_common.py:60-98`).

## Serving

- `pipeline("decision", model=…, decision_config=…, questions=…)`. Task type `multimodal`. Model classes tried in order:
  `AutoModelForImageTextToText`, `AutoModelForCausalLM`, `AutoModel`. The default model is `Qwen/Qwen2.5-0.5B-Instruct`
  @`7ae5576` (`T:src/transformers/pipelines/__init__.py:259-264`).
  - A **bare state** re-asks the last questions, or the ones given up front (`template_utils.py:356-402`).
  - Images come as System One `assets`: placeholders in state, instructions or criteria are replaced by image parts,
    in the order the template renders them (`:272-317, 416-425`).
- **Any chat LM works without a head**: `DECISION_PROMPT` becomes the user turn, thinking is disabled, and the
  next-token logits of the labels are read after the generation prompt (`template_utils.py:113-134`, `pipelines/decision.py:117-120`).
  This makes the pipeline a zero-shot System One adapter for plain instruct models too.
- **`transformers serve` → `POST /v1/systemone`** (`T:src/transformers/cli/serving/systemone.py`). It validates the
  whole request with pydantic and `parse_questions` before loading a model, returns 422 with one message listing every
  invalid question plus "Did you mean" hints, then runs the same pipeline in the inference thread (`:88-125`).
  - The response is `{model, answers, usage{input_tokens, output_tokens: 0}}`. `input_tokens` = Σ prompt lengths, so
    **the state is counted once per question**, unlike Jev's billing (`:121-125,153`).
  - The **decision config is server-wide**, set by `--decision-config file.json`, else defaults. The handler always
    passes a `DecisionConfig` instance, so a repo's own `decision_config.json` is **not** read in serve (inferred from
    `systemone.py:77-81,116` and `pipelines/decision.py:56-61`). The serving doc says decision models "do not yet
    share a standard for storing prompts, answer labels, and calibration settings".
  - The official TypeSafe TS SDK works with `baseURL` changed (serving doc example).
- No batching across prompts or requests, no prefix/KV cache, no CUDA graphs.

**Probable inconsistency in the 10-13 repos** (inferred from reading, not run): `ArthurZ/JEV-9B` and `ArthurZ/GEV-26B-Decide`
keep the head in `decision_head/` and a LoRA at the root, and their READMEs load them with
`pipeline("decision", model="ArthurZ/JEV-9B", decision_config="ArthurZ/JEV-9B")`. Since commit `b348601f` (10-15) a
head needs `backbone_config` at the root. Such a load resolves the LoRA's base causal LM, which has no `backbone_config`,
and so takes the **LM-logit path with default labels** (`Yes`/`No`, `A..Z`) instead of JEV's `false`/`true`/`0..5`
verbalizers and trained head. JEV-27B/35B/Gemma4 were re-converted on 10-16 into the new root layout. JEV-9B and GEV
were not, and the docs example `model="autotrust/GEV-26B-Decide", decision_config="ArthurZ/GEV-26B-Decide"`
(`docs/source/en/model_doc/jev.md:28`) has the same problem.

## Licence

- Code: Apache-2.0 (transformers headers on every new file).
- Ports: `apache-2.0` card tag on d3*, pplx*, Darwin*, PhysicalEye, kev-4b, von, clef-flash, JEV-9B and GEV. **No
  licence tag** on `ArthurZ/JEV-27B`, `-27B-VL`, `-35B` and `-Gemma4-26B-A4B` (`HF:ArthurZ__*/api.json:cardData`).
  d3* and pplx* ship upstream `LICENSE` + `NOTICE`. Backbones keep their own terms; GEV/JEV-Gemma4 are Gemma 4
  (the AutoTrust card links the Gemma licence).
- AutoTrust JEV is distilled from TypeSafe Jev 1.13 outputs (card). How that squares with TypeSafe's terms is not
  addressed anywhere.

---

## What the abstraction fixes, and what varies

| fixed by the framework | varies per model, through config or files |
|---|---|
| question types `noul`/`choice`/`score` (`Question.questions_schemes`; `box` is absent) | the full prompt: `decision_template.jinja` (or the generic prompt inside the model's chat template) |
| request/response shapes = TypeSafe System One; `noul` = p(true), choice/score confidence formulas, score = expected index | where the head reads: `<|mark|>` positions; `<|split|>` tokenisation pieces |
| per-question **softmax over options**, after a `LogitsProcessorList` | labels per type (= option limit), `noul_options` order, `sort_choices` |
| head signature: `(input_ids, marks, num_options, question_type_ids, backbone_kwargs) → flat scores` | head class via `model_type` (`jev`/`kev`/`von`/`clef`) and its config (`num_verbalizers`, `offsets`, `softcap`, `head_dim`, `noul_prior`, layer counts) |
| one forward per prompt, batch 1, `use_cache=False`, no prefix sharing | per-question vs one-prompt-per-request (template uses `question` or `questions`) |
| control-token sanitisation; sentinels never forgeable | backbone: any `AutoModel` in `backbone_config`, plus `use_bidirectional_attention` (Qwen3.5) and `keep_in_fp32_modules` |
| calibration vocabulary: per-type T, a 4-feature linear T, a noul band | values of T / calibration / band |
| backbone weights from the original repo's main (unpinned), adapter from `<sub-model>/` | whether the backbone carries a LoRA |

The real convergence point is the **mark + label/score contract**. "JEV" is just the simplest head in it: `Linear` at
the last token. Any decoder clone whose answer is "logits of option codes at one position" (pplx, d3, Darwin,
PhysicalEye, AutoTrust, Mapika-style deciders, Tev1) maps onto `jev` with no code. Anything with a learned head over
per-option positions needs its own small `model_type`.

## Zeiger as an export target (assessment)

Zeiger (see `clones/zeiger.md`): Qwen3-0.6B-Base fully fine-tuned, causal, one question per sequence,
`opt <M>` markers (`<|object_ref_end|>`), then a 2-layer bidirectional set transformer over the marker states with a type
embedding, an MLP scorer, temperature per option-count bucket, and chunked prefixes for pages larger than 2,048 tokens.

| Zeiger piece | fits the HF format? |
|---|---|
| request/answer, three types, noul as 2 options `false`/`true` | **yes**; set `noul_options: ["false","true"]` |
| prompt (`state \n kind question: instr \n opt0 <M> …`) | **yes**, via `decision_template.jinja` writing `<|object_ref_end|><|mark|>` after each option, so `mark − 1` is the `<M>` state. Ship a `decision_test.json` from `zeiger/encoding.py` to prove token identity |
| backbone | **yes**: `Qwen3ForCausalLM`/`Qwen3Model` in `backbone_config`; full FT weights live in our own repo (the backbone weights must sit in a repo named in the config, or in the checkpoint under `backbone.`) |
| set-transformer head + type embedding + MLP | **no existing head.** `jev` is linear at one position, `kev` is bilinear without cross-option attention, `von` is a per-option MLP without cross-option attention. It needs a new `zeiger` (or generic "marker set") head; its forward is about kev's size: gather `h[marks − 1]`, split by `num_options`, add `type_emb[question_type_ids]`, run a 2-layer `TransformerEncoder` per block, then the MLP |
| 1,500 options | only with explicit `labels.choice` of ≥ 1,500 strings in `decision_config.json`; the default cap is 26 (`template_utils.py:183-188`) |
| `temperature_by_options` buckets (`choice:11+`, …) | **no**. Per-type T only; `CalibratedTemperatureLogitsWarper`'s `n_options·K/8` term gives a linear-in-K approximation, clamped |
| chunked-prefix long pages (state + question repeated per chunk, options gathered across chunks) | **no**: one prompt is one forward. It would need a head-side chunking hook, or a template that emits one prompt per chunk plus a cross-prompt merge, which the pipeline does not have |
| speed | worse than our server: no batching, state re-encoded per question |

Verdict: exporting is feasible for short pages (state ≤ one 2k window) as a new head type plus repo files, and would get
`pipeline("decision")` + `transformers serve /v1/systemone` for free. The format is still a draft PR with a server-wide
config and no long-context story, so it is not yet a substitute for our own `serve.py`. Inferred, not tested.

---

## Notable design choices vs Jev

1. **An adapter layer, not a model.** It standardises how a Jev-like request becomes tokens and how scores become
   System One answers. The architecture is left to each model's head. TypeSafe's Jev internals are unknown
   (`clones/jev-reference.md` §3), and the PR does not claim to reproduce them.
2. **Isolation by separate forwards.** Answers are exactly independent (except clef), but the state is re-encoded per
   question and `usage.input_tokens` counts it per question. Jev ingests the state once and bills it once.
3. **Option limits come from label tables**: 16 for AutoTrust JEV, 255 for code readouts, 26 by default. Jev allows
   255 choice options (two-stage above that) and 2-10 score levels.
4. **The same answer formulas as Jev.** noul has no confidence; choice uses `(p−1/n)/(1−1/n)`; score uses the
   MAD-from-middle normalisation. Probabilities are a softmax at a per-type temperature, so there is no learned
   calibration in the runtime.
5. **No `box` type**, though images (`assets`) are supported for VLM backbones.
6. **Shared readout tables across types** in the code-readout family: noul reuses codes A/B, and score uses the first k
   codes. Jev's per-type heads are unknown.

## Notable design choices vs kev

1. **kev is ported in its row form** (one causal row per question), so a request's questions cost Q full forwards.
   kev's own server runs the state once and continues rows from a cached replica (`clones/kev.md` Serving), and its
   attention-only builds pack the questions under a block-causal mask. The HF port has neither the prefix cache nor
   the packed mask.
2. **kev's pointer head survives unchanged** as `KevDecisionHead` (`key·query/√256`), but under the default `A..Z`
   labels the pipeline caps it at 26 options, against kev's 255.
3. **kev's temperature (2.41) is applied as a per-type warper**, matching kev's eval-mode `head.pt` temperature. kev's
   `KEV_DATE_FACTS` preprocessing and 422-on-overlong states are not reproduced.
4. **The HF repo pins kev-4b's tokenizer revision (`6cfce5c2`) in `decision_test.json` but loads the backbone from main**,
   so the port depends on kev-4b main staying the Qwen3.5 Kev 1.0 weights.
5. **What `jev` adds beyond kev**: verbalizer-row readouts with per-type offsets and softcap, bidirectional
   full-attention backbones, fp32 readout pinning and image assets. These cover the pplx/d3 family, which kev's design
   does not.
