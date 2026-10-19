# WebJev-35B-A3B (Lexmount)

A browser-agent decision model. It is Qwen3.5-35B-A3B-Base with every **non-expert** weight fine-tuned (2.45B of
34.66B; the routed experts stay frozen). It is trained with the **Mapika/decider** recipe unchanged: a lettered
multiple-choice prompt ending in `Answer: (`, read out from the **LM-head rows of the option-label tokens**. The data
mix is 3.19M decision records, 64k of which are live-web operation/element decisions labelled by DeepSeek-V4.1-Flash.
It serves TypeSafe's `POST /v1/systemone` contract (and OpenRouter's `/api/alpha/decisions`) through vLLM plus a FastAPI
adapter. It is built to drop into **browser-use/jev-ultrafast** in place of hosted Jev, and our Zeiger targets the same job.

Source pins used for this profile:

| ref | what | where |
|---|---|---|
| `G` = `lexmount/WebJev` @ `e7692e77` (2026-09-29, "Show complete browser windows in the showcase GIFs (#2)") | shallow clone (`--depth 1 --filter=blob:limit=2m`), read only, never executed | `/home/ai/projects/jev-clones/sources/WebJev` |
| `U` = `browser-use/jev-ultrafast` @ `1231850a` (2026-09-18) | the harness WebJev plugs into; shallow clone, read only. WebJev vendors this exact commit (`G:apps/browser-agent/vendor/PROVENANCE.json`) | `/home/ai/projects/jev-clones/sources/jev-ultrafast` |
| `D` = `Mapika/decider` @ `d6fc522a` (2026-10-12) | upstream training framework. **WebJev pins `c4daaac`**, which this shallow clone does not contain; `D:` citations are HEAD and may differ in detail. The prompt code WebJev shipped is cited from `HF:` instead | `/home/ai/projects/jev-clones/sources/decider` (cloned for `clones/mapika-decider.md`) |
| `HF` = `Lexmount/WebJev-35B-A3B` @ `679020a7` (created 2026-09-29, modified 10-10) | API JSON (`?expand[]=safetensors,…`), tree, README, configs, shipped `decider/` package, card images (no weights; `model.safetensors.index.json` 3.3 MB skipped) | `notes/hf-raw/Lexmount__WebJev-35B-A3B/` (`api.json`, `tree.json`, `files/…`) |
| `DS` = dataset `Lexmount/WebJev` @ `fb2f378d` | card, tree, `statistics.json` (no parquet) | same dir, `_ds_WebJev_*` |

Citation form: `G:train/README.md:46` is file:line in the WebJev clone, `U:jev_ultrafast/model.py:81` in jev-ultrafast,
`HF:decider/systemone.py:59` a fetched HF file under `files/`, and `DS:README.md:130` the dataset card
(`_ds_WebJev_README.md`).

> **Main findings.**
> 1. **Not a new architecture.** WebJev is Mapika/decider's "letter-logit" readout scaled to a 35B-A3B MoE: causal
>    prefill, read the hidden state at `Answer: (`, dot it with the LM-head rows of `A`, `B`, … (single tokens up to
>    255 labels), softmax at T = 1.0. There is no added head, no LoRA and no RL. Lexmount's contributions are the
>    **live-web data** (64,122 decisions, 3,858 tasks, 1,443 sites), an 8×A100 frozen-expert trainer, a vLLM serving
>    adapter and a 125-task live-web eval.
> 2. **Input = jev-ultrafast's request, unchanged.** One request per step: `state = {page{url,title,text≤6000 chars},
>    elements[≤250 viewport elements], recent_actions[≤10]}`, with one `operation` choice question and one
>    `<op>_target` choice question per operation the page allows. Element options are keyed by a running index
>    (`"14"`, select options as `"9:2"`). The model sees them as `14: {"element": "[14] spring", …}`. Training
>    observations were recorded from that same runtime (`DS:README.md:124-128`).
> 3. **Claims:** 38.5% vs Jev 1.13's 16.7% task success on 125 live tasks with the same agent. On 8 single-step
>    benchmarks it is roughly at parity (83.72 vs 84.70 mean; MMLU-Pro 69.4 vs 83.4). There is no Decision Index entry.
> 4. **For Zeiger:** the operation/target split as two questions in one request, and the `elements` JSON with
>    `operations`, `position` and nearby/row context, are directly adoptable. WebJev has **no long-page story**: it
>    sends viewport elements only, caps them at 250 and silently truncates the state at 16k/32k tokens. It also has
>    no cross-option module beyond the backbone's causal reading.

---

## Identity

| | |
|---|---|
| Maker | Lexmount (lexmount.com), a Chinese cloud-browser vendor ("Lexmount Browser", browser.lexmount.com). The repo pushes it as the recommended browser backend (`G:README.md:116-125`). Time zone of the eval runs: `Asia/Shanghai` (`G:evaluation/liveweb/README.md:142`) |
| HF org | `Lexmount`: one model, `WebJev-35B-A3B`, and two datasets, `Lexmount/WebJev` (training data, 2026-09-29, 427 downloads) and `Lexmount/LexBench-Browser` (unrelated earlier browser benchmark, CC-BY-4.0, 2026-01) (`_author_*_Lexmount.json`). The lowercase `lexmount` HF author is empty |
| Model repo | `Lexmount/WebJev-35B-A3B`: 65 downloads (all-time), 2 likes; tags `qwen3_5_moe_text`, `decision-model`, `web-agent`, `pipeline_tag: text-classification`, languages en/zh (`api.json`) |
| GitHub | `lexmount/WebJev` (Apache-2.0, ★5, created and last pushed 2026-09-29, `notes/families.json`) |
| Derivatives | `emoubarak/WebJev-35B-A3B-GGUF` (2026-10-09, 277 downloads): Q4_K_M/Q6_K/Q8_0, plus a llama.cpp `adapter.py` exposing `/v1/systemone` and a copy of the `decider/` prompt package (`_deriv_emoubarak_gguf_api.json`; linked from `HF:README.md:195-203`). No other `base_model:` children |
| Lineage | Training framework, prompt format and data recipe from **Mapika/decider @ `c4daaac`, "used unchanged"** (`G:train/README.md:77-78`, `G:train/NOTICE.md:3-11`). Agent runtime adapted from **browser-use/jev-ultrafast @ `1231850a`** (MIT) (`G:README.md:194-197`). Not affiliated with TypeSafe (`G:README.md:210-211`) |
| Family | One checkpoint, no versions or tags. Closest relative: `Decider 35B-A3B` (Mapika, same base) in the Decision Index list (`notes/decision_index_0.3.2_entries.tsv`) |

---

## Architecture

### Backbone

| | | source |
|---|---|---|
| Base | `Qwen/Qwen3.5-35B-A3B-Base` @ revision `0f081307` (Apache-2.0) | `G:train/README.md:44`, `G:train/configs/webjev-35b-a3b.yaml:4` |
| Class | `Qwen3_5MoeForCausalLM`, `model_type: qwen3_5_moe_text` (text only; vision tower and MTP not loaded in training) | `HF:config.json:2-3,67`; `G:train/src/model_runtime.py:60` |
| Layers | 40 = 30 Gated-DeltaNet linear-attention + 10 full-attention (`full_attention_interval: 4`) | `HF:config.json:12,17-58` |
| Widths | hidden 2,048; full attn 16 q-heads / 2 KV, head_dim 256, gated output, partial RoPE 0.25, θ 1e7, mRoPE sections [11,11,10]; linear attn 16 key / 32 value heads × 128, conv kernel 4 | `HF:config.json:7,13,15,59-63,71-90` |
| MoE | 256 routed experts (8 per token) + 1 shared expert, expert width 512, router aux loss 0.001 | `HF:config.json:68,72-73,91-92` |
| Vocab / window | 248,320 (untied LM head), `max_position_embeddings` 262,144 | `HF:config.json:65,93,96` |
| Params | **34,660,610,688** BF16 (HF safetensors metadata), "about 3B active" | `api.json:safetensors`, `HF:README.md:76,228` |

### What is trained

- **Full fine-tuning of every non-routed-expert weight**: attention, DeltaNet, shared experts, routers, embeddings,
  norms and the LM head. That is **2,448,355,968** trainable parameters; the **32,212,254,720** routed-expert parameters
  stay frozen. "No LoRA, no RL" (`G:train/README.md:46`). Freezing rule: `".experts." in name and "shared" not in name`
  (`G:train/src/model_runtime.py:46-47,68-70`). Frozen experts still pass gradients to their inputs and router weights
  (`:71-74`).
- **No new head and no new tokens.** The readout is the existing `lm_head` restricted to label rows (below). Because
  the LM head is trained, the label rows move too (inferred from `:3-5` together with `README.md:46`).
- Optimiser: upstream Muon (momentum 0.95, Nesterov, 5 Newton-Schulz steps, update × 0.2·√max-dim) for block
  matrices and AdamW (0.9/0.95, eps 1e-8) for the rest. FP32 master weights, **no weight decay**, clip 1.0
  (`G:train/README.md:47`). The FP32 state is sharded across 8 ranks and parked on CPU between updates (`:50`).

### Input encoding

The request is rendered by the shipped `decider` package. That package is exactly what the adapter imports
(`G:evaluation/serving/adapter.py:12-16,61-65`); the HF repo ships a copy under `decider/` (`HF:README.md:262`).

- **State** (`HF:decider/systemone.py:33-36`): a string is passed through; JSON goes through `json.dumps(…,
  ensure_ascii=False)` after `annotate_indices`, which writes `{"_index": i, …}` into every array with ≥ 8 entries,
  "otherwise makes the model count" (json_k64 probe 0.49 → 0.57) (`:18-30`). The jev-ultrafast `elements` list is
  therefore re-indexed with `_index` on top of its own `index` field (inferred from the code; the training rows went
  through the same function, so this is train/serve consistent).
- **Question text** = `instructions`, JSON-dumped if it is not a string (`:14-15,41`). jev-ultrafast sends objects
  (`{"goal","rules"}` / `{"goal","operation","rules":[…]}`), so the question line is a compact JSON object
  (`U:jev_ultrafast/model.py:92,105`; the dataset example question shows the same shape, `DS:README.md:357`).
- **Options by type** (`HF:decider/systemone.py:39-63`):
  - `choice`: `"name"` or `"name: <desc as JSON>"` per criteria key in request order, 2..255;
  - `score`: `"i: <level>"`, 2..10 levels;
  - `noul`: `["no[: desc_false]", "yes[: desc_true]"]`, names `[False, True]`.
- **Prompt** (`HF:decider/prompt.py:111-145`, state-first layout, the only one used for web data and serving):

```
Context:\n{state}                                    <- tokenised together, then cut to max_ctx_tokens (head kept)  :116
\n\nQuestion: {instructions}\nOptions:
\n(A) opt0\n(B) opt1 ... \n(J) opt9                   <- <= 10 options: "narrow", tokenised as one string            :131-133
   or  "\n(" <label tok> ") opt" per option           <- > 10: "wide", built from ids so each label is ONE token     :134-139
\nAnswer: (                                          <- slot = position of the final " (" token                    :141
```

  - Labels: `A..Z`, then the first 229 two-letter uppercase strings that are single Qwen tokens, up to 255
    (`HF:decider/prompt.py:10-13,38-55`).
  - No chat template, no BOS (`add_special_tokens=False`).
  - **No delimiter sanitisation.** A page that contains `\n(C) …` or `Answer: (` lands in the state as plain text.
    Option boundaries are ordinary text tokens in the narrow form (inferred: nothing in `prompt.py`/`systemone.py`
    escapes them; contrast kev's `user_tokens` and HF's `sanitize_control_tokens`).

### Isolation between questions

- **Serving: one prompt per question** (`G:evaluation/serving/adapter.py:115-123`; `rows = [[r] for r in flat]`).
  Every prompt starts with the same `Context:\n{state}` token ids, so **vLLM automatic prefix caching** prefills the
  state once per request (`G:evaluation/serving/serve.sh:17-18`, `README.md:124-126`). Answers are therefore exactly
  independent. Because the base is hybrid DeltaNet, this needs vLLM's prefix caching for linear-attention state as
  well; the authors claim it works at vLLM 0.30.0 (inferred that this covers the recurrent state; not verified).
- `independent: false` (several slots in one prompt) returns **422**, because vLLM reads only the last position
  (`adapter.py:23-24,172-174`).
- **Training is not isolated for multi-question records** (inferred from code). `build()` appends question k's block
  and slot after question k−1's, in one causal sequence (`HF:decider/prompt.py:120-145`), so slot k sees earlier
  questions and their options. The **schema-first** layout (all questions, then context, then all slots), used with
  p = 0.5 on the general component (`G:train/data/README.md:85-87`, `HF:decider/prompt.py:78-91`), lets every slot see
  every question. Web rows are single-question, state-first (`G:train/data/web.py:75-76`), so the browser-specific
  data matches serving exactly.

### Readout

`DecisionModel.slot_logits` (`D:decider/model.py:22-32`; WebJev imports it unchanged, `G:train/src/model_runtime.py:24`):

```
h_k   = backbone(ids).last_hidden_state[slot_k]         hidden state at the " (" of "Answer: ("
W     = lm_head.weight[label_ids]                         [255, 2048]   rows of A, B, ..., AA, AB, ...
z_k   = W · h_k   (fp32; softcap only for Gemma, none here)
z_kj  = -inf  for j >= K_k                                mask unused labels
p_k   = softmax(z_k / T),   T = 1.0                       HF:decider_config.json:3, adapter.py:69,159-163
```

- This is an exact linear verbalizer: the full-vocabulary log-softmax normaliser cancels inside the per-question
  softmax. That makes it the same readout as HF's `jev` head type (see `clones/hf-jev-model-type.md`, "code readouts"),
  so a `jev`-type port with 255 code rows would need no code (inferred; no `ArthurZ/WebJev` port exists).
- Serving reproduces it through vLLM: `max_tokens 1`, `allowed_token_ids = LABEL_IDS[:K]`,
  `--logprobs-mode processed_logits --max-logprobs 256`, then the adapter softmaxes the returned logits
  (`adapter.py:133-163`). It checks that vLLM saw exactly the sent prompt length (`:151-152`).

### Per-type handling (`HF:decider/systemone.py`)

| type | rows scored | answer | extra fields |
|---|---|---|---|
| `choice` | 1 row, K = 2..255 options (K = 1 is short-circuited at p = 1 without a model call, `adapter.py:21-22,198-210`) | `choice` = argmax name, `probabilities` by name (`:119-126`) | **`confidence = p_max`**, not Jev's `(p_max − 1/n)/(1 − 1/n)`; `certainty = 1 − H/log K` (`:113-116,126`) |
| `noul` | 1 row, options `[no, yes]` | `noul = p[1]` (`:123-124`) | none |
| `score` | **one yes/no row per level** (`isolated_levels: true`, `HF:decider_config.json:10`): `"{q}\nProposed answer: {level without its number}\nDoes the proposed answer fit?"` (`:67,71-79`) | `p_i = fit_i / Σ fit`; `score = Σ i·p_i` (`:82-86,128`) | `confidence = p_max`, `certainty`, `level_fit`, `fit_mass` = Σ fit (low if no level fits) (`:101-107`) |

Probabilities are rounded to 4 decimals (`:119`). A score question with L levels costs L prompts, all sharing the state
prefix.

### Calibration

- None fitted: `temperature 1.0`, **`temperature_fitted: false`**, `neutralize_none: false` (`HF:decider_config.json:3-5`).
  The card says probabilities "rank options reliably. To act on confidence thresholds, check calibration on your own
  labels" (`HF:README.md:251-252`). No ECE, Brier or reliability numbers are published.
- Training loss is plain CE over the K labels (`D:decider/train.py:42-51`; the Brier and label-smoothing options
  default to 0 and the WebJev trainer passes neither, `G:train/src/train.py:221`).

### Limits

| | value | source |
|---|---|---|
| Options per question | 2–255 (choice), 2–10 (score) | `HF:decider/systemone.py:11,47-48,54` |
| Training context | 16,384-token context budget; every web row fits without truncation (max 16,383) | `G:train/data/README.md:31`, `DS:README.md:161,423` |
| Serving state budget | `max_state_tokens: 32768` (env `WEBJEV_MAX_STATE_TOKENS`); vLLM `--max-model-len 34816`. Over-budget states are **silently truncated** (tail cut, `prompt.py:116`) | `HF:decider_config.json:8`, `adapter.py:34,70`, `serve.sh:15,42` |
| Card's stated input | "trained on inputs of up to 16,384 tokens" | `HF:README.md:77,253` |
| Questions per request | no cap in the adapter; each is a separate vLLM request (`max_num_seqs 64`) | `adapter.py:182`, `serve.sh:42` |
| Hardware | one 80 GB GPU in BF16; on an A100, 1.9–4.2 GiB is left for KV cache | `G:evaluation/serving/README.md:18-19,44-46` |

```
                       WebJev input layout (serving, state-first, one prompt per question)

 prompt for "operation":
 ┌──────────────────────────────── shared prefix (vLLM prefix cache) ───────────────────────────────┐
 │ Context:\n{"page":{"url","title","text"≤6000ch},"elements":[{"_index":0,"index":"1","role":"link", │
 │   "label":"…","operations":["CLICK"],"position":{x,y},"context":…},…≤250],"recent_actions":[≤10]}  │
 └────────────────────────────────────────────────────────────────────────────────────────────────────┘
   \n\nQuestion: {"goal":"<task>","rules":"<NEXT_ACTION>"}\nOptions:
   \n(A) CLICK: "Click an element, button, …"   \n(B) TYPE_TEXT: "…"   \n(C) SCROLL_DOWN: "Scroll down"
   \n(D) WAIT: "…"  \n(E) PRESS_ENTER: "…"  \n(F) GO_BACK: "…"  \n(G) DONE: "…"  \n(H) BLOCKED: "…"
   \nAnswer: (            <- read h here; logits = LM-head rows of A..H; softmax

 prompt for "click_target" (same prefix, then):
   \n\nQuestion: {"goal":"…","operation":"CLICK","rules":["<NEXT_ACTION>","<TARGET>"]}\nOptions:
   \n(A) 1: {"element": "[1] SpanishDictionary.com Homepage", "current_value": "", "role": "link"}
   …                                                       (> 10 options: labels are single tokens K..Z, AA, AB, …)
   \n(AB) 14: {"element": "[14] spring", "current_value": "0", "role": "option", "selected": "false"}
   \nAnswer: (            <- logits = LM-head rows of the first K labels

 + "type_text_target" and "select_target" prompts when the page has fields / dropdowns (select keys "9:2")
 + score questions: one "Proposed answer: … Does the proposed answer fit?" [no/yes] prompt per level

 Attention inside each prompt: plain causal (30 DeltaNet + 10 full-attn layers); prompts never see each other.
 Option j sees state + question + options < j; the slot sees everything.
```

---

## Training

**Mixture** (one epoch, every record once; `G:train/data/README.md:11-19`; `G:train/README.md:3-7`):

| component | records | questions | tokens | labels from |
|---|---:|---:|---:|---|
| 1. General decisions (Mapika/decider recipe @ `c4daaac`, `full` mode, seed 6) | 1,587,715 | 2,022,356 | 470.6M | 70 of 99 public tasks (81 HF datasets at pinned revisions; 29 tasks held out), the teacher files shipped with decider, **teacher-written contrastive pairs** (DeepSeek-V4.1-Flash; kept only if two blind re-answers match, 5,595/6,000 pass) and programmatic augmentation (`:38-90`) |
| 2. **Live-web decisions (Lexmount)** | 64,122 | 64,122 | 327.6M | DeepSeek-V4.1-Flash operating the agent on live sites, verified (below) |
| 3. Open-Jev typed decisions (ZefanCai/Open-Jev + v1.1, train splits) | 294,573 | 294,573 | 118.6M | hard one-hot rows only; labels **re-derived** from visible rules / the dataset authors' pinned checker / original WANLI (`:105-127`) |
| 4. Knowledge MCQ: Nemotron-RL-knowledge-mcqa, OpenScienceReasoning-2 (traces never read), Nimble | 1,247,313 | 1,247,313 | 295.7M | dataset gold (`:129-141`) |
| **Total** | **3,193,723** | **3,628,364** | **1.21B** | |

- **Programmatic parts of component 1** (`:65-83`): `wide` 64k (full label sets up to 255, small sets padded with
  unrelated labels), `padded` 25k distractors, `described` 40k, `json` 32k, `json_indexed` 20k, `isolated` 237k (one
  yes/no per rating level, matching the serving score path), `rules` 90k twin pairs, `routing`, `commands`,
  `contrastive` 30k, and so on. Tokenisation adds "none of the above" with p = 0.1 and uses schema-first with p = 0.5
  (`:85-87`).
- **Over-255 option sets** are subsampled at build time, always keeping the gold and any abstain-style option
  (`HF:decider/prompt.py:14-20,58-66,121-127`). **Options are shuffled every time** (`:65,128`). Serving keeps the
  caller's order with a no-op RNG (`adapter.py:102-104`).
- **Leakage guards:** exact-match removal of eval items after normalisation for components 3–4 (`eval_guard.py`;
  `README.md:143-146`), and cross-component dedup of inputs (`:126-127,155-157`).
- **No Jev outputs are used.** No component calls TypeSafe; the only LLM teacher is DeepSeek-V4.1-Flash, plus
  decider's shipped teacher files (`G:train/data/README.md:160-163`). Open-Jev v1.1 contains "TypeSafe short
  questions" with "their documented source-license limitation" (`sources/zefan-open-jev/README.md:299`). Whether
  any of those pass WebJev's family filter is not stated (`G:train/data/README.md:109-114` lists the admitted
  families and none is named that way).

**Live-web data** (`DS:README.md`):
- 64,122 instances = 31,050 `operation` (3–13 options, median 7) + 33,072 grounding (`click_target` 31,662,
  `type_target` 1,145, `select_target` 265; 2–137 candidates, median 32). 3,858 tasks from MolmoWeb and WebGym on 1,443
  sites, collected August–September 2026 (`:155-164,239-245,276-282,396-397`).
- Observation formats: 57,283 **structured** (the jev-ultrafast `{page, elements, recent_actions}` JSON) and 6,839
  **textual** renderings (website, task, subgoal, actions so far, page text) "for format diversity" (`:146-147,181-188`).
  Context median 4,262 tokens, p90 9,730, max 16,383 (`:286-292`).
- Labelling (`:398-418`):
  - DeepSeek-V4.1-Flash drives "a production-grade browser agent", guided by a successful demonstration of the same
    task that is never part of the model input.
  - Each label must show execution evidence and a page-state change, pass trajectory loop detection, **agree under
    re-annotation with shuffled options and no guidance**, and survive adjudication for terminal and ambiguous steps.
    About half of the recorded decisions survived.
  - Grounding enrichment converts successful trajectory steps into extra grounding items with mined same-page hard
    negatives at **multiple candidate scales (32/64/128/all)** (`:271-272`). A blind challenge removes candidates
    that could also be correct.
  - Label sources: live_annotation 68.4%, trajectory 31.2%, trajectory_corrected 0.3% (`:303-307`). An audit against
    screenshots found 98% of labels correct or acceptable (`:421-422`).
  - The labels are **hard** (gold index only); no soft targets anywhere in the web data (`:193`).

**Recipe** (`G:train/README.md:40-53`, `G:train/configs/webjev-35b-a3b.yaml`):

| | |
|---|---|
| Objective | CE over the listed options at each answer slot (`train.py:221`) |
| lr | 1e-5 **constant** (warmup_cosine available, not used) (`yaml:16-18`) |
| Batching | length-bucketed batches ≤ 8,192 padded tokens, 16 per update, per-GPU micro-batch 2,048 tokens (`yaml:19-23`). That is ≈ 131k padded tokens per update, so ≈ 9k updates for 1.21B tokens (inferred arithmetic; the step count is not published) |
| Hardware | one node, 8 × A100 80 GB; routed experts run grouped-MM in checkpointed 1,024-row blocks (`yaml:12-13,24`, `model_runtime.py:8-10,30-43`) |
| Wall time, cost | not published |
| Checkpoint | every 2,000 updates; export writes the full model directory plus the `decider/` package (`G:train/NOTICE.md:11-13`) |

---

## Evals

All numbers are self-reported. WebJev was served by vLLM BF16 at T = 1.0. Jev 1.13 was measured on the same inputs
through **OpenRouter** (`typesafe/jev-1.13-20260917`), 2026-09-21 for the benchmarks (`HF:README.md:89-90`,
`G:evaluation/benchmarks/README.md:31-33`).

**Live web, 125 tasks** (Online-Mind2Web 75, WebGym 26, WebVoyager 24; 76 start sites; deterministic evidence-based
verifiers, no LLM judge; 900 s and 60 actions; same agent and same text helper DeepSeek-V4.1-Flash; no planner, no
recovery; `G:evaluation/liveweb/README.md:122-146`, `results/summary.json`):

| subset | WebJev-35B-A3B | Jev 1.13 |
|---|---:|---:|
| all | **38.52%** (47/122) | 16.67% (20/120) |
| Online-Mind2Web | **38.36%** (28/73) | 16.90% (12/71) |
| WebGym | **32.00%** (8/25) | 8.00% (2/25) |
| WebVoyager | **45.83%** (11/24) | 25.00% (6/24) |

- On the 120 tasks gradable for both: WebJev only 30, Jev only 3, both 17, neither 70 (`G:README.md:164-166`).
  Showcase: 6 hand-picked tasks re-run 3× each, WebJev 16/18 vs Jev 0/18 (`:24-27`).
- Caveats:
  - The 125 tasks were "hand-picked", with unstable or anti-bot sites dropped (`liveweb/README.md:52`).
  - **WebGym is both a training task source and an eval source.** No task- or site-disjointness check between
    `DS` and the 125 tasks is documented (inferred risk; `DS:README.md:396`).
  - The agent runtime and the training observations come from the same Lexmount build, so Jev is evaluated
    off its home format while WebJev is on it (inferred). Jev, however, is jev-ultrafast's original target.
- Decision latency in the live run: median 1.18 s, p90 2.89 s, p99 5.02 s per step, over 3 A100 replicas with 10
  episodes each; one step is 1–4 prompts (`G:evaluation/serving/README.md:146-149`).

**Single-step benchmarks** (accuracy; `G:README.md:168-182`, `HF:README.md:112-122`):

| benchmark (items) | WebJev | Jev 1.13 |
|---|---:|---:|
| JevBench public (231) | **87.88** | 85.71 |
| Nimble eval (324) | **92.90** | 92.59 |
| SemIf external (252) | 98.02 | **98.41** |
| scienthoon tickets (873) | **76.29** | 74.91 |
| kev transfer-v4 dev (764) | 85.08 | **85.21** |
| kev decision-v7 dev (1,468) | **86.31** | 83.31 |
| MMLU-Pro 10-way (1,000) | 69.40 | **83.40** |
| typed-decisions test (2,000) | 73.90 | **74.05** |
| **mean of 8** | 83.72 | **84.70** |

- Per-item results are committed (`G:evaluation/benchmarks/results/webjev-35b-a3b/*.jsonl.gz`, not read here).
- kev's own Jev numbers on transfer-v4 (0.857) and decision-v4 (0.845) are close to these (`clones/kev.md` Evals).
  Note that decision-v7 is kev's *training* distribution, not held out for kev; for WebJev it is external.
- **Decision Index 0.3.2:** WebJev is **not listed**. `Decider 35B-A3B` (same base, same framework) is
  (`notes/decision_index_0.3.2_entries.tsv`; the TSV holds metadata only, no scores).
- No calibration metrics, no option-permutation or isolation tests, and no grounding accuracy on a held-out split
  of the web data.

**Speed** (1 A100, vLLM BF16, one client): 72 ms short claim, ~145 ms policy/MMLU-Pro, 228 ms for a 3-question
ticket, **337 ms p50 / 755 ms p95 for one web-page decision** (`HF:README.md:126-134`,
`G:evaluation/serving/README.md:128-144`).

---

## Serving

- **Routes** (`G:evaluation/serving/adapter.py:3-8,213-239`): `POST /v1/systemone` (TypeSafe), `POST
  /api/alpha/decisions` (OpenRouter), `GET /v1/models`, `GET /health`, `GET /stats`. Both POST routes take the
  same body; `model` and `layout` are accepted and ignored (`:107-112`). Optional Bearer key via `WEBJEV_API_KEY`;
  `serve.sh` refuses non-localhost binds without one (`:79-89`, `serve.sh:33-35`).
- **Response:** `{model, answers, usage{input_tokens, output_tokens: 0}}`. `input_tokens` counts the shared prefix
  once (longest common prefix of the rows, `HF:decider/systemone.py:132-139`), so it bills like Jev.
- **Wire compatibility with Jev clients:**
  - It works with jev-ultrafast's validator: probabilities sum within 0.02, the choice is the argmax, and
    `confidence` exists and lies in [0,1] (`U:jev_ultrafast/model.py:30-45`).
  - Single-option choices are answered locally (`adapter.py:198-210`).
  - Differences from Jev:
    - `confidence` is `p_max`, not Jev's normalised formula;
    - extra `certainty`, `level_fit` and `fit_mass` fields;
    - score is computed by per-level isolation rather than a joint distribution;
    - no `box` type and no image `assets`;
    - no 64k/32k limit errors (the state is truncated instead);
    - `independent: false` returns 422.
- **Engine:** stock `vllm serve` (0.30.0) with `--max-model-len 34816 --max-num-seqs 64 --max-num-batched-tokens 16384
  --logprobs-mode processed_logits --max-logprobs 256 --seed 0`, prefix caching on (`serve.sh:15-21,41-43`). The first
  start runs torch.compile for several minutes (`README.md:44-46`).
- **Without the adapter:** the card shows plain `transformers` (`logits[0,-1]` gathered at the label ids) and raw vLLM
  `/v1/completions` with `allowed_token_ids` (`HF:README.md:143-193`). Community llama.cpp/GGUF exists (above).
- **Agent integration:** WebJev's vendored jev-ultrafast picks TypeSafe when `TYPESAFE_API_KEY` is set, otherwise
  OpenRouter (`G:apps/browser-agent/vendor/jev_ultrafast/model.py:66-75`). The web app points it at a self-hosted
  WebJev. Hedged duplicate requests are available via `JEV_HEDGE_MS` (`:15-21,24-44`).

---

## Licence

- Code and weights: **Apache-2.0** (`G:LICENSE`, `HF:README.md:256-262`, `api.json:cardData.license`). The base
  Qwen3.5-35B-A3B-Base is Apache-2.0. The `decider/` copy is Apache-2.0 (Mapika). jev-ultrafast is MIT (vendored,
  provenance hashes in `PROVENANCE.json`).
- Dataset `Lexmount/WebJev`: Apache-2.0 for Lexmount's contribution. Page text, URLs and labels "remain the property
  of their owners"; the labels "were produced with DeepSeek models under their terms of use" (`DS:README.md:428-433`).
- Public training sources "keep their own license and terms"; the repo does not redistribute them
  (`G:train/data/README.md:165-166`, `G:train/NOTICE.md:145-148`).
- No TypeSafe outputs are used for training, so the TypeSafe-ToS question raised for AutoTrust does not arise here.
  Jev outputs appear only as eval baselines.

---

## Notable design choices vs Jev

1. **Domain-specialised System 1.** The thesis is that a decision model trained on the agent's own observation
   format beats a general one inside that agent: 2.3× task success, at a cost of 14 pp MMLU-Pro. Jev is a general
   model that jev-ultrafast merely calls.
2. **Readout through the LM head, not a custom head.** Option labels are single vocabulary tokens read at one slot.
   Jev's readout is unknown; Hume's tests show that Jev's option boundaries cannot be forged. WebJev's narrow-form
   boundaries are plain text, `\n(B) `, with no sanitisation (inferred vulnerability, untested).
3. **Isolation by separate prompts plus prefix caching**, not one packed pass. Semantics match Jev ("every answer
   independent", state billed once). Compute is one prefill of the state plus one suffix per question; for a score
   question, one suffix **per level**.
4. **Score via isolated per-level yes/no rows** with `fit_mass` as a "no level fits" signal. Jev returns one joint
   distribution and documents weak numeric calibration between levels.
5. **Uncalibrated by design** (T = 1.0, `temperature_fitted: false`), and `confidence = p_max`. Jev advertises
   calibration (RLCD) and uses a normalised confidence formula.
6. **Smaller context in practice.** Trained to 16k, served up to 32k with silent truncation. Jev allows 64k per
   request and 32k per question, and returns `max_tokens_exceeded`.
7. **Text-only.** Screenshots are captured by the agent, but neither model sees them in this loop. Jev 1.14 accepts images.

## Notable design choices vs kev

1. **Readout:** an LM-head-row verbalizer at `Answer: (` against kev's trained pointer head (`</opt>` keys ·
   `<decide>` query). Both read all options from a slot placed after them. WebJev's needs no new parameters but
   ties K ≤ 255 to the label table.
2. **What is trained:** a full fine-tune of 2.45B non-expert weights of a 35B MoE with Muon at lr 1e-5, no LoRA. kev
   uses LoRA r16 at 5e-5 on dense/hybrid Qwen; only its 27B is full fine-tuned, at 2e-6 with WiSE-FT.
3. **Data scale:** 3.19M records / 1.21B tokens in one epoch, against kev's 25k–146k records. Both use hard labels
   only and no Jev distillation. WebJev uses an LLM teacher (DeepSeek) for the web and contrastive data; kev uses code
   and public labels (plus two-teacher agreement for CFPB).
4. **Isolation mechanics:** both run hybrid DeltaNet bases as one causal row per question over a cached state.
   kev implements the cache itself (`DynamicCache` replicas); WebJev delegates it to vLLM prefix caching.
5. **Calibration:** kev ships a held-out-fitted temperature in `head.pt` (e.g. 27B T = 1.32). WebJev ships none.
6. **Delimiters:** kev reuses Qwen special tokens and escapes user text. WebJev uses plain-text `(A)` labels with no
   escaping.
7. **Evaluation:** both score Jev on the same items. WebJev reuses kev's transfer-v4/decision-v7 suites and adds an
   end-to-end live-web benchmark, which kev lacks. It has no locked-test discipline or bootstrap CIs.

---

## Comparison with Zeiger

Zeiger (`clones/zeiger.md`): Qwen3-0.6B-Base full FT, one question per sequence, `opt <M>` markers, a 2-layer
bidirectional set transformer plus MLP over the marker states, per-option-count temperature, chunked prefixes for
long pages, a non-wire-compatible `POST /decide`.

| aspect | WebJev (via jev-ultrafast, Lexmount build) | Zeiger |
|---|---|---|
| State shape | JSON `{page:{url,title,text}, elements:[…], recent_actions:[≤10 {action,kind,text,page_changed}]}`, compact `json.dumps`; arrays ≥ 8 get `_index` (`U:model.py:107-116`, `HF:decider/systemone.py:18-36`). 11% of training rows use a plain-text rendering instead (`DS:README.md:147,187`) | `<state JSON ≤1536 tok>` then `kind question: instr`, then options (`zeiger/encoding.py:118-140`) |
| Page text | visible-in-viewport text nodes, **≤ 6,000 chars**, incl. open shadow roots and img alt in the Lexmount build (`U:snapshot.js:82-92`; `PROVENANCE.json`) | page content in the state (budget 1,536 tok per chunk) |
| Which elements | interactive elements **in the viewport only** (center inside `innerWidth×innerHeight`), visible, enabled, not password/file/hidden; roles button/link/checkbox/radio/switch/tab/menuitem*/option/gridcell/combobox/textbox/searchbox/spinbutton (`U:snapshot.js:24-27,57-60`). Lexmount adds scrollable panels as `E<n>` controls | whole page (Common Crawl element tables, live crawl, Mind2Web) |
| Element identity | running index `"1".."N"` per observation (dedup by DOM node); select options as `"<idx>:<k>"`; the agent maps back to stable `e<n>` action ids (`U:model.py:48-78,129`) | option position / marker order |
| Element fields | `index, role, label, value, operations[], checked/selected/expanded, options[]` for selects (`U:model.py:61-76`); Lexmount adds `position{x,y}`, `context` (nearby text ≤70), `state_classes`, `field_context`, `row_context`, `input_constraints`, `already_visited` (`G:apps/browser-agent/backend/browser_adapter.py:248-274`, vendored `snapshot.js:180-203`) | element text per option (≤ 200 tok) |
| Option text | target option = `"<idx>: {\"element\": \"[idx] label\", \"current_value\": …, role, checked…}"`; elements appear **twice**, once in `state.elements` and once as the option (`U:model.py:96-104`) | `optK text <M>`, element described once, in the option |
| Operation vs element | **two questions in one request**: `operation` (CLICK/TYPE_TEXT/SELECT plus available controls SCROLL_DOWN/UP, WAIT, PRESS_ENTER, GO_BACK, PRESS_ESCAPE, scroll-region `E<n>`, DONE, BLOCKED), plus one `<op>_target` per operation present. The agent uses the target answer of the chosen operation and ignores the others (`U:model.py:83-106,120-133`; `browser_adapter.py:356-371`) | target only (pick an element or "none"); the operation is the instruction |
| "none" option | no none in target questions; `BLOCKED`/`WAIT`/`SCROLL_*` at the operation level play that role. decider training adds "none of the above" with p = 0.1 for general data only (`G:train/data/README.md:86`) | `"none"` is a regular option |
| Max options | 250 elements per snapshot (`U:snapshot.js:99-101`; `omitted_actions` counted, not shown); model 2–255; training grounding up to 137 (median 32) | 40–1,500 tested; refuses rather than truncates |
| Long pages | **no chunking**: viewport-only elements, scroll actions to see more, state truncated at 16k (train) / 32k (serve) tokens | chunked prefixes, options gathered across chunks, trained to 32k, head limit 131k |
| Vision | none to the model. Screenshots are taken for recording and auditing only (`U:agent.py:21-23,44`) | none |
| Cross-option comparison | the backbone's causal reading only (option j sees options < j; the slot sees all). Shuffled at train time | dedicated 2-layer bidirectional set transformer over markers |
| Readout | LM-head rows of label tokens at `Answer: (`, softmax, T = 1 | MLP logit per marker, softmax, temperature per option-count bucket |
| Training data | 64k live-web decisions (DeepSeek-V4.1-Flash on MolmoWeb/WebGym tasks, execution- and re-annotation-verified, multi-scale hard negatives) + 3.1M general decisions | Common Crawl element tables (jevpilot analyser), polite live crawl, Mind2Web/MT-Mind2Web, judge-verified jevpilot steps, whole-page questions, general decision data |
| Model size / cost | 35B total / 3B active, one 80 GB GPU, 337 ms per page decision on A100 | 0.6B dense, ~140 ms per short question, 1.5 s per whole page on GPU; CPU-capable |
| API | `/v1/systemone` wire-compatible (drop-in for jev-ultrafast) | `POST /decide`, Jev-like but not wire-compatible |

**What Zeiger could adopt** (suggestions, not tested):
1. **The jev-ultrafast request as an input format.** Zeiger would become a drop-in decision model for this harness
   (and its Lexmount fork), and could be benchmarked end to end against Jev and WebJev on the 125-task suite, whose
   tasks, verifiers and harness are all public (`G:evaluation/liveweb/`).
2. **An operation question next to the target question**, sharing one state. WebJev's DONE/BLOCKED/WAIT/scroll
   operation layer is where Jev fails most in their showcase (loops, premature give-ups, `G:README.md:29-36`). Zeiger
   currently grounds only.
3. **Element features**: `operations[]`, `position`, nearby/row context, `state_classes`, `already_visited` and
   `input_constraints`. These are cheap DOM features that disambiguate look-alike candidates.
4. **The Lexmount dataset** (`Lexmount/WebJev`, Apache-2.0 for the labels; page content third-party) as extra
   grounding/action data, especially the **multi-scale candidate sets** (32/64/128/all) with same-page hard negatives,
   and as a held-out set split by `task_id`. Note it is viewport-scale (≤ 137 candidates), while Zeiger targets
   whole pages.
5. **Their label verification protocol**: execution evidence, re-annotation with shuffled options and no guidance,
   and adjudication, applied to our judge-verified jevpilot steps.
6. **Not worth adopting:** silent state truncation, `confidence = p_max` and uncalibrated T = 1. Zeiger's refusal on
   overflow and per-bucket temperatures are stricter.
