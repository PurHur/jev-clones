# Intern-Decision (InternLM / Shanghai AI Lab)

Open Jev-style decision models from Shanghai AI Laboratory's InternLM team. Each is a **full fine-tune of a Qwen3.5 instruct VLM** (0.8B / 2B / 4B) that keeps its tied LM head. The request is rendered as an ordinary chat in which the assistant turn is a **pre-filled JSON skeleton with one `<decision>` token per question**. One causal forward pass reads the LM-head logits at the position **just before each `<decision>`**, restricted to that field's single-token option symbols (`A..Z a..z 0..9`), then a per-checkpoint temperature. All questions share one sequence and there is **no isolation between them**.

Source pins used for this profile:

| ref | what | where |
|---|---|---|
| `HF:<size>` | HF API JSON (`expand[]=safetensors`), tree, README, `config.json`, `inference.py`, tokenizer/added/special tokens, chat template, safetensors index, LICENSE. No weights. Repo heads: 0.8B and 2B unchanged since 2026-09-26 upload; 4B `1e65a988` (2026-10-16, tag-only commit) | `/home/ai/projects/jev-clones/notes/hf-raw/internlm__Intern-Decision-{0.8B,2B,4B}/` |
| `G` | GitHub `internlm/Intern-Decision` @ `3572c8a` (2026-09-29, "upload license"). Shallow clone, read only, never executed | `/home/ai/projects/jev-clones/sources/intern-decision` |
| `S` | HF Space `internlm/intern-decision` (Docker, CPU, serves 0.8B) | `/home/ai/projects/jev-clones/notes/hf-raw/internlm__intern-decision-space/` |
| web | Only secondary coverage (nxcode, orcarouter, 36kr). No InternLM blog post or paper found | `notes/hf-raw/internlm__Intern-Decision-4B/extra/web_sources.md` |

Citation form: `HF:4B/inference.py:232` = file:line in the fetched HF file. `G:src/inputs/schema.py:152` = file:line at `3572c8a`.

> **Main finding.** Of all the clones surveyed, Intern-Decision has the least machinery. It adds no head, no custom mask and no position tricks, and has a single new token. It is a "masked next-token" SFT: the label sits on the `<decision>` token and the causal shift moves it onto the preceding position (`G:src/inputs/masked.py:61-76,219-220`). This makes it vanilla-HF and vLLM-friendly, with multi-question answering in one pass. The cost is that questions are **not independent**: every field sees the whole schema, and later fields attend to the hidden states at earlier `<decision>` markers. This is the opposite of Jev's documented "add or remove questions without changing the others' results". On the community Decision Index the 4B scores **38.21** (rank 104), about level with Kev-4B (39.5) and far from Jev 1.14 (61.53). That contrasts with the card's self-chosen 7-suite average, on which the 4B is 90.02 vs Jev 88.74.

---

## Identity

| | |
|---|---|
| Org | `internlm` on HF ("Intern Large Models", Shanghai AI Laboratory). The weights collection is `internlm/intern-decision-6ab75776af19ca84df0bc004` |
| Models | `internlm/Intern-Decision-0.8B`, `-2B`, `-4B`, all created **2026-09-26** (05:35–05:36 UTC; files uploaded 07:57) (`HF:*/api.json`) |
| Params (safetensors API) | 0.8B **852,985,920** (2,592 F32); 2B **2,213,241,664** (2,592 F32); 4B **4,539,265,536** (3,840 F32). The rest is BF16, and the totals include the vision tower. The small F32 group is probably the GDN `A_log`/`dt_bias` (inferred) |
| Usage (2026-10-19) | downloads 3,012 / 2,226 / 4,458; likes 37 / 18 / 88 |
| Base | Card `base_model`: `Qwen/Qwen3.5-{0.8B,2B,4B}`, i.e. the **instruct/chat** checkpoints. The Space README says the 0.8B is "fine-tuned from the Qwen3.5-0.8B instruction-tuned/chat checkpoint" (`S:README.md`), and the training launcher asks for "a local Qwen3.5 instruction checkpoint" (`G:scripts/train.sh`). The DI board lists `Qwen3.5-*-Base`, which looks like board metadata error |
| Code | GitHub `internlm/Intern-Decision`: training (XTuner), HF and XTuner inference backends, eval and calibration tooling, a FastAPI demo, and the bundled test suites. Training data is **not** released (`G:README.md:20-27`) |
| Demos | Mario, mouse-moving, Doom (Defend, Health Gathering) and browser-use videos (`G:assets/demos/`). They show decisions used as an agent/game action policy over screenshots |

---

## Architecture

### Backbone

`Qwen3_5ForConditionalGeneration` (`model_type qwen3_5`), with the vision tower kept. `config.json` differs only in dimensions across sizes (diffed):

| | 0.8B | 2B | 4B |
|---|---|---|---|
| text layers | 24 = 18 GDN + 6 full | 24 = 18 + 6 | 32 = 24 + 8 |
| d / FFN | 1024 / 3584 | 2048 / 6144 | 2560 / 9216 |
| attn heads / KV | 8 / 2 | 8 / 2 | 16 / 4 |
| GDN value / key heads | 16 / 16 | 16 / 16 | 32 / 16 |
| vision depth / width | 12 / 768 | 24 / 1024 | 24 / 1024 |
| shared | head_dim 256, `attn_output_gate`, partial RoPE 0.25 (mRoPE 11/11/10, θ 1e7), `max_position_embeddings` 262,144, vocab 248,320, **`tie_word_embeddings: true`**, `transformers_version 4.57.0.dev0` | | |

- **No LM-head tensor and no MTP weights** are in the safetensors index; the head is the tied embedding (`HF:*/model.safetensors.index.json`: only `model.language_model.*` + `model.visual.*`). The config keeps `mtp_num_hidden_layers: 1`, but those weights were not saved.
- Shards are named `model-{language,projector,vision}-000N-save_rank0.safetensors`, the XTuner HF-save naming.
- **What is trained**: the whole language model, embeddings included (`freeze_language=False`). The **vision tower and projector are frozen**, and the code asserts this (`G:configs/training/qwen35.py:38-48`, `G:src/model/qwen.py` `DecisionQwen.__init__`). There is no LoRA and no extra parameters.

### Vocabulary

- One new special token: **`<decision>` = id 248077**, `special: true` (`HF:*/added_tokens.json`, `tokenizer_config.json`). It falls inside the padded 248,320-row embedding, so no resize is needed. Training adds it with `tokenizer.add_special_tokens` (`G:src/inputs/masked.py:82,105`).
- Answer symbols are **62 single-token characters** `ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789` (`G:src/inputs/schema.py:11`). The code checks that each one encodes to one token (`HF:4B/inference.py:283-285`).
- Forgery guard: if the literal `<decision>` appears anywhere in the rendered user text, the request is **rejected** rather than escaped (`G:src/inputs/schema.py:150-151`). Other chat specials in the state, such as `<|im_end|>`, are not sanitised. Because the tokenizer parses specials by default (`split_special_tokens: False`), they would act as real control tokens (inferred, untested).

### Input layout (one sequence, one pass)

`compile_row` (`G:src/inputs/schema.py:106-163`, identical in `HF:*/inference.py:74-102`) builds three chat messages, which the Qwen3.5 template renders. The assistant message is the last one, so the template emits an empty think block before its content (`HF:4B/chat_template.jinja:100-101`):

```
<|im_start|>system
You are a careful decision assistant. ... return one valid JSON object mapping each field name
to its chosen symbol ... Do not include explanations, Markdown, or extra text.<|im_end|>
<|im_start|>user
[<|vision_start|><|image_pad|>...<|vision_end|>]* (0-8 images, before the text)
Return one answer for every field using the supplied answer symbols.

## State
{ json.dumps(state, indent=2) }              <- strings are JSON-quoted, objects pretty-printed
## Decision schema
team: Which team should handle this request?
    A = billing: Payments and refunds
    B = delivery: Shipping and delivery
urgency: Rate the priority.
    A = 0: Low
    B = 1: Medium
    C = 2: High
refund_requested: Is the customer asking for a refund?
    A = no: The answer is no (negative, or disagree with the claim).
    B = yes: The answer is yes (affirmative, or align with the claim).<|im_end|>
<|im_start|>assistant
<think>

</think>

{
    "team": "<decision>",                     <- logits read at the token BEFORE each <decision>
    "urgency": "<decision>",                     (the opening-quote / `": "` token, inferred from
    "refund_requested": "<decision>"              BPE), restricted to that field's symbols A,B,(C)
}<|im_end|>
```

Readout positions and visibility (causal, nothing masked):

```
tok:   [sys][user: state + full schema of ALL fields][asst: {  "team": " <D1> ", "urgency": " <D2> ", ... ]
read:                                                          ^p1                ^p2
p1 sees: system, state, ALL questions/options, field name "team"
p2 sees: everything p1 sees + <D1> token (its hidden state carries the model's belief about field 1)
         + the name of field 2. Fields are scored jointly and order-dependently: no per-question isolation.
```

- `positions = (input_ids == marker_id).nonzero() - 1`, and the code checks that their number equals the number of fields (`HF:4B/inference.py:232-234`). The forward pass uses `use_cache=False, logits_to_keep=positions`, so only those rows of the LM head are computed (`:244`).
- Gold symbols never enter the input. The `<decision>` token stands in for the answer in both training and inference, so later fields see the marker and **not** a predicted symbol (`G:README.md:74-78`).
- Option and question order are preserved and must not be changed (`G:README.md:80-82`). There is no shuffle at inference, and the docs say nothing on whether training shuffles (unknown).

### Readout math

```
z_f  = LMhead(h[pos_f])[ids(A..symbol_K)]          K = #options of field f   (tied embedding rows)
p_f  = softmax(z_f.float())                        (HF:4B/inference.py:286)
p'_f = softmax(log p_f / T)                        == softmax(z_f / T); argmax-preserving, asserted (:153-168)
```

| type | options compiled | answer |
|---|---|---|
| `choice` | `criteria` object in insertion order → A, B, … (`schema.py:39-40`) | `choice = argmax` (ties broken lexically), `probabilities` by key |
| `score` | list → values `"0","1",…`; or an object with **finite numeric keys** (`schema.py:41-46`, `inference.py:126-134`) | `score = Σ float(value)·p`. With numeric keys this is the expected **value**, not the expected level index. `legend` = value → description |
| `noul` | always `[no, yes]` → A, B; descriptions taken from `no/false/0` and `yes/true/1` keys, or the fixed defaults above (`schema.py:47-57`) | `noul = p(yes)` |

- `confidence` = **max p** for every type (`inference.py:177`). This differs from Jev's closed-form `(p_max − 1/n)/(1 − 1/n)` (choice) and MAD-based (score) formulas, and Jev returns no confidence for noul. The response also adds `decision`, `source`, `timing`, `backend` and `calibration`.
- Limits: **1–16 questions per request**, **1–62 options** (noul is fixed at 2), **≤ 8 images**, `max_length` **8,192 tokens** including image tokens. Over-length input is rejected without truncation (`inference.py:104-142,229-231`). Training uses the same 8,192 default (`G:configs/training/qwen35.py:32`). Jev allows 255 options, any number of questions and 64k tokens.

### Calibration

- One scalar **temperature per checkpoint**, fitted by NLL with convex bisection over T ∈ [0.01, 100] on **1,728 calibration cases**, with **1,693** held-out validation cases. The split is private (`HF:*/README.md` Calibration, `G:README.md:320-353`).
- Values (`G:benchmarks/temperature-presets.json`, which binds each one to shard SHA-256s):

  | size | T |
  |---|---|
  | 0.8B | 2.7478 |
  | 2B | 2.1005 |
  | 4B | 1.9924 |

  T > 1 everywhere, so the raw SFT model is over-confident and the smallest model most of all.
- The presets were fitted with the **XTuner** backend. The HF backend can differ by kernel/BF16 noise, and "occasionally labels" (`G:docs/EVALUATION.md` §1).
- The model is not type-aware: there is no per-type or per-K temperature (Strands and d1 have them), and no learned calibrator.

---

## Training

- **Objective** "autoregressive masked language modeling", `objective_revision: masked-next-token-v4` (`G:scripts/training_config.py`):
  - labels are −100 everywhere except at each `<decision>` position, which carries the gold symbol id (`G:src/inputs/masked.py:61-76`);
  - the collator shifts `input_ids[:, :-1]` / `labels[:, 1:]`, so the logit at marker−1 learns the symbol (`:219-220`);
  - **cross-entropy over the full 248k vocabulary**, not over the legal symbols, with token-level reduction (`G:configs/training/qwen35.py:123-127`, `G:README.md:80`);
  - **hard labels only**, with no soft or teacher targets (`G:docs/DATA.md:272`).
- **Defaults in the released config** (the actual released runs' settings are not published):
  - full FT with AdamW, **lr 2e-6**, wd 0.01, cosine to 2e-7, 3% warm-up, global batch 16, 2 epochs, seed 42, FSDP with full recompute;
  - `pack_max_length` = 8,192 with soft packing;
  - final checkpoint only, with "no benchmark-based selection" (`G:configs/training/qwen35.py:100-141`).
  - `train.sh` uses 4 GPUs by default.
- **Packing isolation**: the packed rows use `cu_seq_lens` varlen attention. A custom `packed_causal_conv1d` zeroes the GDN short-conv history at every row boundary, so packed examples don't leak into each other (`G:src/model/qwen.py` `packed_causal_conv1d`). Text-only batches still run a dummy frozen-vision forward to keep FSDP collectives in step.
- **Data**: undisclosed. "This interface specification does not disclose the models' training data sources, composition, quantities, or mixing proportions" (`G:docs/DATA.md:212-214`). The record format is `{id, state, images, questions, targets{field:{label}}}` (`G:docs/DATA.md:223-249`), and the image support plus the game/browser demos imply visual decision data (inferred). Whether the training data includes the train splits of AG News, ToolACE or WildJailBreak, whose test splits are among the evals, is **unknown**.

---

## Evals

**Self-reported, seven suites** (`G:README.md:102-111`; 10,751 rows / 12,351 decisions, bundled under `G:benchmarks/accuracy-v1/`; Brier and ECE on Jevbench-Hard only, n = 111, max-p ECE with 10 bins):

| model | Easy | Orig | Hard | Typed Dec. | ToolACE | AG News | WildJB | Avg | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jev | 100.00 | 98.61 | 72.07 | 73.35 | 91.29 | 89.57 | 96.29 | 88.74 | 0.358 | 0.095 |
| Kev (version unstated, T 2.96) | 100.00 | 93.06 | 45.05 | 65.60 | 87.42 | 89.82 | 75.97 | 79.56 | 0.738 | 0.262 |
| JevK5 | 100.00 | 97.22 | 73.87 | 64.50 | 80.97 | 89.13 | 90.45 | 85.16 | 0.366 | 0.047 |
| Intern-Decision-0.8B | 97.92 | 80.56 | 52.25 | 77.35 | 94.52 | 88.61 | 64.48 | 79.38 | 0.530 | 0.066 |
| Intern-Decision-2B | 100.00 | 84.72 | 63.96 | 79.35 | 96.45 | 89.96 | 78.33 | 84.68 | 0.437 | 0.100 |
| Intern-Decision-4B | 100.00 | 98.61 | 73.87 | 80.55 | 96.45 | 90.82 | 89.86 | **90.02** | 0.347 | 0.065 |

Caveats:
- The average is dominated in count by AG News (7,600) and WildJailBreak (2,210), but the mean is unweighted across suites.
- Typed Decision (`LocalLLaMA/typed-decisions`) and ToolACE are the suites where Intern-Decision beats Jev. They may be close to the training distribution, but this cannot be checked because the data is private (inferred risk).
- Jevbench is `fstandhartinger/jevbench` @ `7ce310c` (`G:requirements-eval.txt`).

**Known-distribution calibration pilot** (`G:benchmarks/known-distribution-pilot-v1`, 96 cases = 48 paired settings, exact reference distributions, 4B only): Brier/ECE **0.628/0.213 raw → 0.550/0.089 with T**, vs Jev 1.13 at 0.595/0.130. The generator and scorer are bundled. It is evaluation-only and was not used for fitting.

**Decision Index 0.3.2 board** (`notes/hf-raw/_spaces/dix_leaderboard.md`, 2026-10-16):

| rank | DI Full | entrant | ECE |
|---:|---:|---|---:|
| 104 | **38.21** | Intern-Decision-4B | 0.0278 |
| 133 | 22.58 | Intern-Decision-2B | 0.0608 |
| 153 | 11.78 | Intern-Decision-0.8B | 0.0255 |
| ref | 39.5 / 13.66 | Kev 4B r10 / Kev 0.8B r15 | 0.032 / 0.040 |
| ref | 61.53 | Jev 1.14.0 | 0.080 |

On the held-out panel the self-reported +1.3 pp over Jev turns into −23 points. Board ECE is good at all sizes (0.026–0.061), so the temperature transfers better than the accuracy does.

**Latency** (`G:README.md:115-127`): RTX 4090, BF16, SDPA, HF backend with fla 0.4.2; 289 input tokens, 3 fields in one pass. Mean 33.98 / 33.28 / 44.16 ms (0.8B / 2B / 4B) vs Jev 109.70 ms. This compares local model time with Jev's network round trip.

---

## Serving

- **HF in-repo engine**: `from inference import DecisionEngine; engine.predict(request)` (`HF:*/inference.py:249-307`). It runs one request at a time with no batching and no KV/prefix cache (`use_cache=False`). `backend="hf"` is the only backend in the HF copy, and the GitHub version adds an XTuner backend. The requirements pin `torch 2.9.1`, `transformers 5.14.1`.
- **HTTP** (`G:src/service/app.py`, `S:app.py`): `POST /v1/decisions`, alias **`/v1/jev`**, plus `GET /health`, `/docs` and a browser demo. The body is the Jev request (`state`, `questions{type, instructions, criteria}`), with images as base64 bytes (≤ 8 images, 12 MB each, 32 MB total). The response is the Jev envelope `model/answers/usage` with extra fields. It is **not** `/v1/systemone`, and there is no `/v1/models`, so the TypeSafe SDK would need a path change (inferred). It binds to loopback by default, with no auth and no request logging.
- **Optional "thinking handoff"** (`G:src/inference/pipeline.py`, `thinking.py`): this is a System-1 → System-2 escalation. Fields whose max-p falls below `confidence_threshold` (default 0.7) can be streamed to an external OpenAI-compatible chat model (`LLM_BASE_URL`/`LLM_API_KEY`), which picks one label. Such answers return `probabilities: null`, `source: "thinking"`, and noul as a hard 0/1. It is off by default and not used in any reported eval.
- **Space** `internlm/intern-decision`: a Docker app on **CPU basic** serving the 0.8B, with `MAX_INPUT_LENGTH` 8,192 (`S:README.md`).
- **vLLM / other engines**: no integration is shipped. The model is a stock Qwen3.5 checkpoint, though, so any engine that returns prompt logprobs at chosen positions could serve it (inferred).

---

## Licence

- Weights: `license: apache-2.0` on all three cards. `LICENSE` is Apache-2.0, and `LICENSE-QWEN` preserves the upstream Qwen licence (`HF:4B/files/LICENSE`, card "License and acknowledgment").
- Code: Apache-2.0 (`G:LICENSE`). The bundled Jevbench files are MIT. AG News carries an "unknown" licence, and the repo warns about redistributing it (`G:docs/EVALUATION.md` §2).

---

## Notable design choices

**vs Jev**
1. **The LM head is the decision head.** Jev claims outputs "not a language model" with dedicated typed readouts (reconstructed). Intern-Decision is a plain causal LM scoring next-token symbols, so its probabilities are those of a constrained one-token generation computed in a single prefill.
2. **No question isolation.** Jev guarantees independent answers (Hume's sibling-secret test: 0.00). Here all questions sit in one user turn and later fields attend to earlier markers, so adding, removing or reordering questions can change other answers (by construction; not measured by InternLM).
3. **Order is semantic.** Option letters are bound to position, and question order fixes which fields see which. Jev also shows first-option bias, but Intern-Decision is not trained to resist it, and no permutation tests are published.
4. **Much tighter limits**: 16 questions, 62 options and an 8k context, vs unlimited questions, 255 options and 64k for Jev.
5. **Calibration** is one scalar T on top of hard-label CE, roughly what Jev's RLCD is said to replace. Confidence is max-p, not Jev's formula. Score keys can carry numeric values, so `score` is an expected value rather than an expected level.
6. **Multimodal from day one** (frozen Qwen3.5 ViT, ≤ 8 images), comparable to Jev 1.14's image support. There is no `box` type.

**vs kev**
1. **Head**: kev drops the LM head and trains a pointer (`<decide>` query · `</opt>` keys). Intern keeps the tied LM head and reads symbol logits, so it needs **zero new parameters** but caps options at 62 single-character symbols.
2. **Mask**: kev isolates questions exactly, with a block-causal mask on attention-only bases and one causal row per question over a cached state on Qwen3.5 hybrids. Intern runs one shared causal sequence with no masking. That is cheaper (state and schema encoded once, no cache bookkeeping) but not isolated.
3. **Trained params**: kev uses LoRA r16 on a **-Base** model at lr 5e-5, except the 27B full FT at 2e-6. Intern full-fine-tunes the **instruct** model's LM at lr 2e-6 with vision frozen, and the system prompt and chat template are part of the learned interface.
4. **Delimiters**: kev reuses five Qwen specials and escapes user text. Intern adds one `<decision>` special, rejects requests that contain it, and leaves other specials untouched.
5. **Openness**: both publish training code, eval suites and temperatures. kev also publishes its data recipe and train sources, while Intern's training data is fully private.
6. **Results**: the two sit at the same level at 4B on the DI board (38.21 vs 39.5), while Intern's own table puts a Kev at 79.56 vs Intern-4B at 90.02. Intern's per-checkpoint ECE on the board is slightly better at 4B (0.028 vs 0.032).
