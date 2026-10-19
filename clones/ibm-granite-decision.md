# IBM Granite decision adapters (granitelib-core `decision`, Granite Switch 4.1)

A Jev-style decision engine built as a **PEFT adapter on a stock chat LM**, not as a new model. Granite 4.1 (dense, full-attention decoder) is kept frozen. A rank-64 LoRA or **aLoRA** on all attention and MLP projections is trained to answer with **one option-label token** (`A`, `B`, ...). The answer distribution is the **softmax of the LM-head logits restricted to the label ids** at the first assistant position. Every question is **its own forward pass** over the same state. The aLoRA variant runs the state on the frozen base, so its KV cache is the base model's and can be shared across questions, across adapters and with ordinary chat. The adapters ship in `ibm-granite/granitelib-core-r1.0/decision/` and are pre-composed (aLoRA only) into `ibm-granite/granite-switch-4.1-8b` and `-30b`.

Source pins used for this profile:

| ref | what | where |
|---|---|---|
| `CORE` | `ibm-granite/granitelib-core-r1.0` @ main (decision added in commit #48 on 2026-10-15 18:56 UTC and renamed `decisions/` → `decision/` in #49): `decision/README.md`, 4× `adapter_config.json`, 4× `io.yaml`, repo README, `commits.json`, `tree.json` | `notes/hf-raw/ibm-granite__granitelib-core-r1.0/` |
| `SW8` / `SW30` | `ibm-granite/granite-switch-4.1-8b` / `-30b` (created 2026-10-14 21:38 / 21:23 UTC): `config.json`, `BUILD.md`, `adapter_index.json`, `chat_template.jinja`, `io_configs/decision/io.yaml`, `compose_report.json`, `model.safetensors.index.json`, `api.json` | `notes/hf-raw/ibm-granite__granite-switch-4.1-{8b,30b}/` |
| `BASE` | `ibm-granite/granite-4.1-8b` and `-30b` `config.json`, `api.json`; 8B `tokenizer.json` and `chat_template.jinja` | `notes/hf-raw/ibm-granite__granite-4.1-{8b,30b}/` |
| `GS` | github.com/generative-computing/granite-switch, shallow clone @ `7e724560` (2026-10-15). Read only, never executed | `sources/granite-switch/` |
| `BLOG` | HF blog "Granite Switch 4.1: Chat, Decisions, and Speech from One Endpoint" (2026-10-15; Soule, Chaudhury, Strobelt, Lastras) | `notes/hf-raw/_blogs/ibm-granite-switch-4-1.{html,txt}` |
| `SPC` | Spaces `ibm-granite/speech-to-decisions` (`app.py`) and `ibm-granite/granitelib-evals` (`data.js`) | `notes/hf-raw/ibm-granite___spaces/` |
| `DIX` | Decision Index board snapshot (0.3.2, 2026-10-16) | `notes/hf-raw/_spaces/dix/index.json`, `dix_leaderboard.md` |

Citation form: `CORE:decision/README.md` means a fetched file, and `GS:docs/MULTISWITCH_EXPLAINED.md:343` means file:line in the clone.

> **Main finding.** This is the simplest possible "Jev on an LLM" design, and it was shipped by a big vendor. It has no new head, no special tokens, no mask tricks and no calibration step. One label token is read from the LM head per question, and each question is its own request. Its interesting part is the systems side, not the model. The **aLoRA** keeps the state's KV on the frozen base, so N questions over one state cost one state prefill plus N short question suffixes. Under Granite Switch the same cache is also shared with chat and with other adapters (guardian). The cost of that design is accuracy: aLoRA is 2.4–3.4 DI points below the always-on LoRA (`CORE:decision/README.md` Results). **No training data, loss, hyperparameters or calibration method is published.**

---

## Identity

| | |
|---|---|
| Vendor | IBM Research (Granite team; Granite Libraries / Granite Switch / Mellea projects under github.com/generative-computing) |
| Artefacts | `granitelib-core-r1.0/decision/granite-4.1-8b/{lora,alora}` and `.../granite-4.1-30b/{lora,alora}`: `adapter_config.json`, `adapter_model.safetensors`, `io.yaml`, `model.sig` (`CORE:tree.json`) |
| Pre-composed | `granite-switch-4.1-8b` (8.78B, 980 dl / 30 likes) and `granite-switch-4.1-30b` (29.99B, 768 dl / 22 likes). Each holds the decision **aLoRA** plus `guardian-core` aLoRA and a Granite Speech 5.0 TurboCTC ASR cascade (`SW8:README.md`, `SW*:api.json`) |
| Release | Switch models 2026-10-14 (card "Release Date: October 14, 2026"). Blog 2026-10-15. Adapters published to granitelib-core 2026-10-15 (`CORE:commits.json`). The Switch `BUILD.md` lists the decision source as a local `alora` directory, while guardian-core is pinned to a HF revision. So the decision adapter was composed **before** it was published to granitelib-core (inferred from `SW8:BUILD.md`) |
| Positioning | "A new Decision aLoRA for general-purpose decision-making (e.g. v1/systemone)". IBM calls its earlier specialised granitelib adapters "the OG decision models" (`BLOG`) |
| Signing | Sigstore `model_signing` detached signature per variant, certificate chain to DigiCert Trusted Root G4. Unlike the other granitelib adapters, it is not `Granite-sign@ibm.com` (`CORE:decision/README.md`, `CORE:README.md:60-68`) |
| Not covered | No decision adapter exists for granite-4.1-3b (`CORE:README.md:64`). The `-preview` Switch builds (3B/8B/30B, "complete Granite Libraries bundle") predate it and were not checked |

---

## Architecture

### Backbone

| | Granite-4.1-8B | Granite-4.1-30B | src |
|---|---|---|---|
| Class | `GraniteForCausalLM`, dense decoder, GQA, RoPE, SwiGLU, RMSNorm, all full attention | same | `BASE:*/config.json` |
| Layers / d / FFN | 40 / 4096 / 12,800 | 64 / 4096 / 32,768 | same |
| Heads / KV heads | 32 / 8 (head_dim 128) | 32 / 8 | same |
| Context | 131,072 (`rope_theta` 1e7) | 131,072 (`rope_theta` 5e7) | same |
| Vocab / embeddings | 100,352, **tied** | 100,352, tied | same |
| µP scalars | `embedding_multiplier` 12, `attention_multiplier` 1/128, `residual_multiplier` 0.22, **`logits_scaling` 16** | same except residual 0.175 | same |
| Params (HF API) | 8,791,592,960 BF16 | 28,865,728,512 BF16 (the "30B" is 28.9B) | `BASE:*/api.json` |
| Checkpoint | the **post-trained instruct** model (SFT + RL), not `-Base` | same | `BASE:granite-4.1-8b/README.md:15` |

- The backbone stays **frozen**. The adapter is "never merged" into the base (`CORE:decision/README.md`).
- The tokenizer is byte-level BPE (cl100k-style split regex), and `bos = eos = <|end_of_text|>`. The invocation marker `### DECISION` tokenises as `['###', 'ĠDEC', 'ISION']` = `[14711, 43917, 25189]` (checked against `BASE:granite-4.1-8b/files/tokenizer.json`).

### Trained parameters

All four `adapter_config.json` files are identical apart from `base_model_name_or_path`, the module list order and `alora_invocation_tokens` (`CORE:decision/*/*/adapter_config.json`):

| field | value |
|---|---|
| `peft_type` / `task_type` | `LORA` / **`CAUSAL_LM`** (kev uses FEATURE_EXTRACTION, because it drops the LM head; IBM keeps and uses it) |
| `r` / `lora_alpha` / dropout | **64 / 128** / 0.05, so scale α/r = 2 (same ratio as kev's 16/32) |
| targets | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (every linear in every layer) |
| `alora_invocation_tokens` | `[14711, 43917, 25189]` on `alora/`, `null` on `lora/` |
| extras | no DoRA, no rsLoRA, no `modules_to_save`, no `trainable_token_indices`: **no head, no new embeddings** |
| `peft_version` | 0.18.1 (aLoRA needs peft ≥ 0.18) |

Trained parameter count, computed from the shapes. It matches the safetensors sizes in `CORE:tree.json` to within the file header:

| | per layer | layers | total | bf16 bytes (file) | share of base |
|---|---|---|---|---|---|
| 8B | q,o 2×524,288 + k,v 2×327,680 + gate,up,down 3×1,081,344 = 4,947,968 | 40 | **197.9M** | 395.8 MB (file 395,913,520) | 2.25% |
| 30B | 1,703,936 + 3×2,359,296 = 8,781,824 | 64 | **562.0M** | 1,124.1 MB (file 1,124,195,808) | 1.95% |

The rank is unusually high for a classification adapter. IBM's aLoRA blog says early aLoRA prototypes trailed regular LoRA because the base KV has no task-aware state, and that "increasing the adapter's rank" closed the gap ([research.ibm.com/blog/inference-friendly-aloras-lora](https://research.ibm.com/blog/inference-friendly-aloras-lora)). Applying that reasoning to r = 64 for both variants here is inferred.

### Input encoding

There are **no reserved tokens and no custom encoder**. The request is a Granite chat-template prompt. The README code is the only published encoder; IBM's own `decide_alora.py`, cited in `SW*:io_configs/decision/io.yaml`, is not public.

```
LoRA variant  (one user turn; adapter active from token 0)

<|start_of_role|>system<|end_of_role|>{fixed decision-engine system prompt?}<|end_of_text|>\n      (io.yaml: "a fixed system prompt"; README code passes none)
<|start_of_role|>user<|end_of_role|>State:\n{state}\n\nQuestion:\n{question}\n\nOptions:\nA: {opt0}\nB: {opt1}\n...\n\nReply with exactly one option label.<|end_of_text|>\n
<|start_of_role|>assistant<|end_of_role|>   <- logits read HERE, at the last prompt position
                                            (LoRA active on every token)

aLoRA variant  (two user turns; adapter switches on at the marker)

<|start_of_role|>user<|end_of_role|>State:\n{state}<|end_of_text|>\n                  <- frozen base, KV shareable
<|start_of_role|>user<|end_of_role|>### DECISION\nQuestion:\n{q}\n\nOptions:\nA: ...\nB: ...\n\nReply with exactly one option label.<|end_of_text|>\n
<|start_of_role|>assistant<|end_of_role|>   <- logits read HERE
             ^^^^^^^^^^^^^ adapter active from '###' (id 14711) to the end of the sequence

N questions over one state = N sequences sharing a prefix (vLLM prefix cache / reused KV):

  [ state turn (base KV) ] ──┬── [ ### DECISION q1 + options ][asst] -> softmax over q1's label ids
                             ├── [ ### DECISION q2 + options ][asst] -> softmax over q2's label ids
                             └── [ ### DECISION q3 + options ][asst] -> ...
```

- **Labels.** Options map to single-token labels: `A`–`Z`, "then digits / letter-pairs for wide questions" (`CORE:decision/README.md` How it works). In the Granite vocab all 26 upper, 26 lower and 10 digit characters are single tokens, and **552 of the 676** `AA..ZZ` pairs are. So the scheme must skip the pairs that are not in the vocab (counted from `tokenizer.json`). The exact label sequence beyond `Z` and the maximum option count are **not published**.
- **noul.** The README says noul takes "none (optional false/true descriptions)" and returns `{false, true}` probabilities, but does not say how they are rendered. IBM's own demo renders noul as a 2-option choice `A: Yes`, `B: No` and takes `noul = p(A)` (`SPC:speech-to-decisions.app.py:119-123,489-500`). Whether training used that order, or `false`/`true` with descriptions, is unknown.
- **score.** 2–10 ordinal level descriptions are presented as lettered options. The returned score is `Σ i·p(i)` (`CORE:decision/README.md`). That level i maps to the i-th letter is inferred.
- **System prompt.** `io.yaml` (all four) says "a fixed system prompt", but the text is not published. The README code sends no system turn. The Switch card explicitly sends "no system prompt", so that "the state turn's tokens [are] identical to a plain chat request about the same state" for prefix caching (`SW8:README.md` decision section). It is unclear which one matches training. Note that the Granite template injects no default system prompt when none is given (`BASE:granite-4.1-8b/files/chat_template.jinja`).
- **No forgery guard.** Option text is plain text, and nothing stops a state from containing `B: ...` lines or `### DECISION`. The README's aLoRA snippet asserts the marker appears **exactly once** in the token sequence; otherwise PEFT would activate at the wrong place (`CORE:decision/README.md` Use). The Switch template injects the control token before the **last** occurrence of the marker in the **last** user message that contains it (`SW8:chat_template.jinja`, `GS:docs/MULTISWITCH_EXPLAINED.md:454-470`).

### Mask and isolation between questions

- **Isolation is per request, not per mask.** "A request may carry several questions; each is one independent pass over the same state" (`CORE:decision/README.md`). No packed or block-causal sequence exists. Isolation between questions is therefore exact by construction, as in kev's row form.
- Attention is the stock causal mask. Inside a question, options are read **jointly and in order**: each option sees the earlier ones, and the read position sees all of them plus the reply instruction. Jev-like order sensitivity, such as a first-option bias, is expected (inferred; no permutation study is published).
- **aLoRA semantics.** PEFT applies the LoRA delta only from the first invocation token onward. Every state token is therefore computed by the frozen base, and the question tokens attend to **base** keys/values of the state ([aLoRA, arXiv 2504.12397](https://arxiv.org/abs/2504.12397); `CORE:decision/README.md`). The state cannot be adapted to the question type. IBM's DI table puts the cost at −2.4 (30B) and −3.4 (8B) points.
- **LoRA variant.** The adapter is active over the state too, so the state KV is adapter-specific. It is still identical across questions over the same state, which means a prefix cache can share it between decision questions, but not with base chat or guardian (inferred from the layout; IBM only advertises sharing for aLoRA).

### Readout

```
z      = LM_head(h_last) / logits_scaling           (Granite divides logits by 16; tied embeddings)
z_lab  = z[label_id_0 .. label_id_{K-1}]            (gather only this question's label ids)
p      = softmax(z_lab)                              (T = 1: no temperature, no calibration map)
choice = argmax p ; noul = p(true label) ; score = Σ i·p_i
```

- One position and one LM-head row per label, which makes this a **verbaliser over the existing vocabulary**. It is the Together/Intern family of readout, not kev's pointer head. Because the embedding matrix is tied, the label "head" is the input embedding of `A`, `B`, ... (`BASE:config.json tie_word_embeddings true`).
- `io.yaml`: `max_completion_tokens: 1`, `temperature: 0.0`. The model does a 1-token "generate", but the answer is the label-restricted distribution and never the sampled token (`CORE:decision/*/io.yaml`).
- **Confidence** is not mentioned. No Jev-style closed-form `confidence` field is returned by anything IBM published.
- **Calibration.** The Switch card calls the output "its calibrated probability over the option labels" and "the softmax over option labels `decision` is trained to produce" (`SW8:README.md`). No temperature, ECE or reliability figure is published, and nothing in the configs stores a temperature. So whatever calibration there is comes only from the training loss (inferred: likely CE on the label token).

### Context and limits

| | value | src |
|---|---|---|
| Context | 131,072 tokens (base window). No decision-specific training or validated length is given | `BASE:config.json` |
| Options per question | ≥ 2 for choice, 2–10 for score. The maximum is not stated; the label alphabet allows 62 single characters + 552 pairs | `CORE:decision/README.md`; tokenizer count |
| vLLM label cap | `logprob_token_ids` takes at most **128 ids** per request (HTTP 400 above). IBM's workaround is to resend the identical prompt per 128-label chunk and softmax the concatenated logprobs ("mathematically identical", one extra request per 128 options; prefix cache makes the repeats cheap) | `SW8:README.md` "more than 128 options" |
| Questions per request | one per forward pass, any number of requests per state | `CORE:decision/README.md` |
| Switch control-token budget | under vLLM bf16 the MultiSwitch counting head is exact up to **188 retained control tokens** per request, so very long multi-adapter conversations re-prefill | `GS:docs/MULTISWITCH_EXPLAINED.md` §8 |

### Granite Switch packaging (how the aLoRA is served)

- `GraniteSwitchForCausalLM` (`model_type granite_switch`, transformers 5.17) embeds every adapter as a stacked LoRA bank: `lora_A/B` per adapter, padded to `max_lora_rank` 64 (`SW8:compose_report.json` `adapter_stacked`). It fuses `q,k,v` → `qkv_proj` and `gate,up` → `shared_mlp.input_linear` (`lora_target_modules [o_proj, qkv_proj, shared_input_linear, shared_output_linear]`) (`SW8:config.json`, `SW8:compose_report.json`).
- **Control tokens** `<|guardian-core|>` = 100352 and `<|decision|>` = 100353 are added to the vocab (100,355 with `<|audio|>`). The chat template, given `chat_template_kwargs.adapter_name = "decision"`, rewrites `### DECISION` to `<|decision|>` + ` DECISION` (`invocation_drop_chars 3`). The **switch** then rewrites the control token back to its substitute id 14711 (`###`) before embedding, so the decoder sees exactly the token ids the PEFT aLoRA was trained on (`SW8:chat_template.jinja:1-4`, `SW8:config.json adapter_substitute_token_ids [27, 14711]`, `GS:src/granite_switch/config.py:18-35`).
- The **MultiSwitch** router is parameter-free: a counting head and a Kerdock-coded memory head (`codebook [2048, 64]`, `ms_code_m 6`, `control_token_gain 15`, `ms_memory_gain 28`). Together they give a per-token adapter index that is piecewise constant and latest-wins. Index 0 is base. It occupies **2 cache-layer slots**, which is why `num_hidden_layers` is 42 / 66 while the weights hold 40 / 64 decoder layers (`GS:src/granite_switch/config.py:9-16`, `GS:docs/MULTISWITCH_EXPLAINED.md:1-90`; `SW8:model.safetensors.index.json` has layers 0–39 plus `model.switch.codebook` and `control_to_substitute_lut`).
- Composition adds +4.72% params at 8B (8,380,551,168 → 8,776,400,896) and +3.89% at 30B (`SW*:BUILD.md`). The decision adapter accounts for 197.9M / 562.0M of that.
- "KV cache normalization: each adapter sees only the base model's KV cache, never another adapter's internal state" (`SW8:README.md` How It Works).

---

## Training

**Essentially nothing is disclosed.** The decision README has no Training section. Other granitelib cards (e.g. requirement-check) describe their data, teacher models and splits (`CORE:requirement-check/README.md:217-`), but this one does not. Not published:

- training data (sources, synthetic vs public, size);
- teacher or distillation targets;
- loss (CE on the label token, soft targets, proper scoring rule);
- epochs, lr, batch, hardware;
- whether noul and score were trained as lettered choice;
- any calibration step;
- whether any Decision Index public rows were in training (the board counts such rows as wrong in its private parts; `DIX:suite.calibration_note`).

What can be inferred from the artefacts:
- The adapters were trained as **PEFT CAUSAL_LM adapters on the post-trained instruct checkpoint**, at r 64 / α 128 / dropout 0.05 with the same 7 targets for both variants (`CORE:*/adapter_config.json`). The aLoRA was trained with the `### DECISION` invocation, so the loss only reaches adapter weights through question-side tokens (aLoRA method, arXiv 2504.12397).
- The target is a **single label token**: "trained to answer by gathering the logits of the single-token option labels and taking their softmax" (`SW8:README.md`). Whether the loss was softmax-over-labels or full-vocab next-token CE is not stated. The phrasing suggests the former (inferred).
- IBM's lineage is a library of **task-specific** adapters (guardian, requirement-check, uncertainty, context-attribution, RAG). The blog pitches combining one generalist decision adapter with specialists ("broad coverage from the generalist and higher accuracy from specialists") rather than one model trained on everything (`BLOG`).
- An Adapter SDK in Mellea "for training your own custom and third-party adapters" is "planned" (`BLOG`, `SW8:README.md`). No training code was found in `GS` (a grep for decision/systemone/noul hit only routing code and tests).

---

## Evals

**IBM-reported, Decision Index v0.3 public set** (`CORE:decision/README.md` Results; per-area values are skill × 100):

| Model | Knowledge | Language | Retrieval | Tools | Arts | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Decision LoRA, Granite-4.1-30B | 34.8 | 61.3 | 64.4 | 69.3 | 29.2 | **53.30** |
| Decision aLoRA, Granite-4.1-30B | 34.3 | 54.0 | 60.8 | 70.3 | 30.7 | **50.91** |
| Decision LoRA, Granite-4.1-8B | 28.3 | 52.1 | 60.5 | 62.7 | 16.5 | **46.00** |
| Decision aLoRA, Granite-4.1-8B | 27.7 | 43.5 | 56.2 | 58.6 | 22.2 | **42.56** |

**Switch cards (aLoRA as composed, public "fast subset", 1,068 rows, edition-v0.3, hosted vLLM)** (`SW8:README.md`, `SW30:README.md`):

| | Overall | Knowledge | Language | Retrieval | Tools | Arts |
|---|---:|---:|---:|---:|---:|---:|
| granite-switch-4.1-8b | 42.9 | 29.1 | 48.9 | 49.8 | 59.8 | 18.0 |
| granite-switch-4.1-30b | 50.1 | 32.4 | 58.6 | 49.8 | 76.5 | 26.4 |

- Local and hosted serving "agree to within ~0.6 points". The two tables use different subsets, so their Retrieval/Tools values differ a lot even though the overall scores are close.
- **Not on the DI 0.3.2 board** (no Granite row in `DIX:dix_leaderboard.md` or `dix/index.json`). So no Full score (20% public + 50% private same-skill + 30% private new-domain) exists, and no ECE.
- For scale, **public_skill** on the board: Jev 1.14 60.70, Jev 1.13 57.96, Kev 27B 56.69, clef-flash 56.15, Kev 9B v2 41.54, Kev 4B 37.95, Intern-Decision-4B 38.00 (`DIX:index.json scores.public_skill`, `jev.scores`). If IBM's "public set" is the same component, the 30B LoRA's 53.30 sits about 4.7 below Jev 1.13 and about 3.4 below Kev-27B. 8B LoRA (46.0) is above Kev 9B. That this is a like-for-like comparison is **inferred**. IBM ran its own harness, and public scores overstate Full scores for models that may have seen public data (e.g. clef-flash 56.15 public vs 47.61 Full).
- Pattern: the aLoRA penalty falls on **Language** (−7.3 at 30B, −8.6 at 8B) and Retrieval (−3.6 / −4.3). On Tools and Arts the aLoRA is equal or better. This fits the idea that tasks needing a question-conditioned reading of the state are the ones a frozen-base state hurts most (inferred).
- Other IBM decision comparisons (granitelib-evals space) set *specialised* granitelib adapters against Jev 1.13, microsoft-decision-1, gpt-6-luna-decisions and clef-flash on their own tasks. Example, requirement-check IFEval BAcc: adapter 0.855 vs Jev 0.685 (`CORE:requirement-check/README.md:209-215`). The blog claims guardian-core "on average outperforms Jev 1.13, Microsoft-Decision-1, GPT-Luna-Decisions, and Clef-Flash on safety-related decision tasks" (`BLOG`). These numbers are for the specialists, not for the `decision` adapter.
- No calibration metrics, permutation tests, isolation tests or latency tables are published for the decision adapter.

---

## Serving

- **Transformers + PEFT.** `PeftModel.from_pretrained(base, "granite-4.1-*/{lora,alora}")`, one `model(input_ids)` forward, gather the label logits, softmax (`CORE:decision/README.md` Use). Requires `transformers ≥ 4.44`, `peft ≥ 0.18`.
- **vLLM + Granite Switch** (vLLM ≥ 0.26 < 0.28 per the card; the `GS` README says 0.26–0.30; torch ≥ 2.11, transformers 5.16–5.17). It uses the OpenAI-compatible `/v1/chat/completions` with:
  - `extra_body.chat_template_kwargs.adapter_name = "decision"`;
  - `logprob_token_ids = [label ids]` (a vLLM extension that returns logprobs for exactly those ids);
  - `max_completion_tokens 1`, `logprobs true`, `temperature 0`.

  The client applies the softmax over the returned label logprobs. One server also serves base chat, guardian-core and ASR (`SW8:README.md`).
- **API compatibility.** There is **no `/v1/systemone` endpoint** in any published IBM code. The blog says the model "runs in vllm with a plug-in ... and has native inference support in llama.cpp and downstream platforms which allows the model to seamlessly swap in for /v1/systemone API calls anywhere they're used" (`BLOG`). No such shim was found in `GS`, so a caller has to translate `{state, questions{type, instructions, criteria}}` into chat turns and labels themselves. Mellea has **no decision intrinsic yet** ("planned") (`SW8:README.md`).
- **Cost model.** Per question: one prefill of the question suffix, one decode step for the 1-token completion, and with aLoRA no state recompute when the state is cached. IBM's demo shares the state turn across 3 guardian calls, a decision call and a summary call (`SPC:speech-to-decisions.app.py:1-45,100-125`). IBM's "aLoRA vs LoRA race" shows a 74% vs 29% KV hit rate (`GS:README.md`). The aLoRA blog claims individual tasks run "20 to 30 times faster than a traditional LoRA" in multi-adapter flows. Decision-specific latency is not published. The Doom demo claims a custom 3B Switch model makes "60–100 decisions / second, compared to Jev's reported 10" (`BLOG`).
- **Determinism.** Greedy and deterministic by configuration (`io.yaml`). Batch-invariance is not discussed.

---

## Licence

- Adapters: **Apache-2.0** (`CORE:decision/README.md` front matter and License; HF `cardData.license apache-2.0`). The README adds: "Base-model use is additionally governed by the Granite model terms".
- Base Granite 4.1 8B/30B: Apache-2.0 (`BASE:granite-4.1-8b/README.md:23`, `BASE:granite-4.1-30b/api.json`).
- Granite Switch checkpoints and the `granite-switch` code: Apache-2.0 (`SW8:README.md`, `GS:LICENSE`).
- Weights are signed (sigstore, DigiCert chain); the README gives the expected fingerprints.

---

## Notable design choices vs Jev

1. **One request per question instead of many questions in one pass.** Jev answers any number of questions per request, with isolated answers. IBM gets the same isolation trivially, by running separate passes over a shared prefix cache. The cost is per-question decode overhead and HTTP round trips, and there is no batched typed API.
2. **Keeps the LM head and reads label tokens.** There is no new head and no learned verbaliser. Options are letters, so the option *text* only influences the answer through the context, and option-name keys from the request must be mapped to letters client-side. Jev's documented first-option bias is likely shared, because the options are causal and ordered (inferred).
3. **Calibration by training only.** Jev documents RLCD and closed-form confidence. IBM ships neither a temperature nor a confidence field, and publishes no ECE.
4. **Not a System One model.** It is an adapter on a chat LM and runs on the same endpoint as chat, guardian and ASR. That is the opposite of Jev's "no generation at all", and it is IBM's selling point.
5. **Interface gap.** There are no typed `criteria` objects, no `legend`, no ≥128-option handling without chunking, and no `/v1/systemone` server in the published code (only the blog claims compatibility).
6. **Scale.** A 30B dense backbone reaches 53.3 on the public DI split, vs Jev 1.13 at 57.96 and Jev 1.14 at 60.70 public_skill on the board, without the private parts.

## Notable design choices vs kev

1. **Readout.** kev drops the LM head and trains a pointer head over `</opt>` hidden states against a `<decide>` token. IBM keeps the LM head, uses existing letter tokens as labels, and trains no head at all (`task_type CAUSAL_LM` vs kev's FEATURE_EXTRACTION). Both read at a position after all options, so both can in principle use "none of the above" context.
2. **Isolation.** kev v0 uses a packed block-causal mask with restarted positions; Kev 1.0 uses rows over a cached state. IBM matches the row form, with the state cached by the serving engine (vLLM prefix cache) rather than by the model code.
3. **aLoRA vs LoRA on the state.** kev's LoRA adapts the state encoding, and its `--shared_prefix` training backpropagates into the state. IBM's aLoRA deliberately leaves the state on the frozen base so the cache can be shared with non-decision traffic, at a measured −2.4 to −3.4 DI cost. IBM's LoRA variant is the kev-like choice.
4. **Adapter size.** r 64 on 7 projections (198M / 562M params) vs kev r 16 (11–45M). IBM's aLoRA blog motivates the higher rank by aLoRA's capacity needs.
5. **Delimiters.** kev reuses 5 reserved Qwen tokens and sanitises user text so they cannot be forged. IBM uses plain text (`State:`, `Options:`, `A: `) and a multi-token text marker, with no sanitising, so option boundaries are forgeable in principle (inferred; not tested).
6. **Openness.** kev publishes code, data generators, frozen suites, calibration temperatures and Jev side-by-sides. IBM publishes only adapters, configs, a usage snippet and DI public scores. The training recipe, data and calibration are closed.
