# Standard One (Standard Thinking)

A company-built Jev clone in two generations that share the name.

1. **Standard One 3B / 8B (v1 → v2.2).** A LoRA fine-tune of **Ministral 3 Instruct**, with the Pixtral vision tower kept. It reads the **logprobs of single-token option labels** (A–Z, AA, …) from stock SGLang, one prefill per question, behind a separate `jev-adapter` that serves `POST /v1/systemone`.
2. **Standard One SH (3B v3, 8B v4).** The same backbone (further LoRA-tuned and merged) plus a **70.6M-parameter "joint schema head"**. It reads two hidden-state taps for the whole request in one forward pass and **lets questions attend to each other** by design.

Source pins (no GitHub repo for the model; server code ships inside the HF repos):

| ref | what | where |
|---|---|---|
| `HF:8B` = `StandardThinking/StandardOne-8B` main `ac495d1c` (v2.2, 2026-10-04; tags v1, v1.1, v2) | API JSON, tree, card, `release-manifest.json`, `MERGE_REPORT.json`, `docs/BENCHMARKS.md`, `evidence/*.json`, `server/` (jev-adapter, read only, never executed) | `notes/hf-raw/StandardThinking__StandardOne-8B/` |
| `HF:3B` = `…-3B` `70023495` (v2.2); `HF:8BL`/`HF:3BL` = `…-8B-LoRA` `c1665b92`, `…-3B-LoRA` `e89ee1e6` | card, `adapter_config.json`, `merge.py` | `notes/hf-raw/StandardThinking__StandardOne-{3B,3B-LoRA,8B-LoRA}/` |
| `HF:8BSH` = `…-8B-SH` `534cedff` (v4, 2026-10-13); `HF:3BSH` = `…-3B-SH` `7037b583` (v3) | card, `release-manifest.json`, `head/schema_head_config.json`, `code/{head,model,render,scoring,serve_head}.py` | `notes/hf-raw/StandardThinking__StandardOne-{8B,3B}-SH/` |
| `W` | standardthinking.ai/resources/guides/standard-one-sh-decision-index/ (fetched 2026-10-19) | web |

---

## Identity

| | |
|---|---|
| Maker | Standard Thinking, Inc. The site names no people and gives no location (`W`). HF org `StandardThinking`; engine repos on GitHub at `standardthinkingai/standard-one-sglang` and `standard-one-adapter` (Apache-2.0, not cloned) (`HF:8BSH/README.md:182-188`) |
| Repos | 8B, 3B (merged), 8B-LoRA, 3B-LoRA (created 2026-09-24); 8B-SH (10-07), 3B-SH (10-09). Derivatives include `-FP8` (SGLang-validated) and `-GGUF` (Q8_0 … IQ2_M plus vision projector) (`HF:8B/README.md:70-77`, `HF:8BSH/README.md:41`) |
| Versions | Base line: v1 → v1.1 → v2 → v2.1 → **v2.2** (10-04). SH line: 8B v1 (LoRA) → v2 → v3 → **v4** (10-13); 3B-SH **v3** (10-13) (`HF:8B/README.md:30-33`, `HF:8BSH/README.md:22-27`) |
| Popularity | 8B 3,257 downloads, 3B 2,877, 0 likes each; the SH repos have about 200 downloads each (`HF:*/api.json`) |
| Hosted | "The Standard One API serves the SH models"; new users get free credit; prices not published (`W`) |

## Architecture

### Backbone
| Model | Base (revision) | Adaptation | Params | Source |
|---|---|---|---|---|
| 8B v2.2 | `mistralai/Ministral-3-8B-Instruct-2512-BF16` @`f6fae979` (**instruct**, not base) | LoRA r16, α32, dropout 0 on `q,k,v,o,gate,up,down_proj` of the **language model only** (vision tower and projector excluded); 44.56M trainable; merged: 238 tensors changed, max \|Δw\| 0.00211 | 8.92B BF16 | `HF:8B/README.md:229-231`, `MERGE_REPORT.json` |
| 3B v2.2 | `…-3B-Instruct-2512-BF16` @`b6d637be` | LoRA r16, α32 | 3.85B | `HF:3B/README.md:172`, `HF:3BL/adapter_config.json` |
| 8B-SH v4 | Standard One 8B v2.2 (`ac495d1c`) | further **LoRA r64, α128**, fp32 merge, bf16 save; head trained jointly | 8.92B + 70.6M head (fp32) | `HF:8BSH/release-manifest.json` |
| 3B-SH v3 | Standard One 3B v2.2 | same scheme; head taps the final layer and layer 19 | 3.85B + head | `HF:3BSH/README.md:46-47` |

The `mistral3` architecture keeps the Pixtral vision tower; the adapter accepts image data or HTTP URLs (`server/jev_adapter/protocol.py:114-141`). No image decision benchmark is reported (`HF:8B/README.md:66-67,312`).

### Input layout
- **v2.2 `jev-adapter`, `served` wording** (default): `Context:\n<state JSON/text>\n\nQuestion: <instructions>\nOptions:\n A: name: description …`, no system prompt. `native` wording uses `State:` / `Question:` / `Options:` with `A. name: description`, at most 26 options (`HF:8B/server/jev_adapter/protocol.py:222-303`, `HF:8B/README.md:107-119`).
- noul is rendered as two options, `yes`/`no` (true, false order); score as its ordered levels (`protocol.py:153-161,249-250`).
- The labels must tokenise as single tokens at the assistant boundary; otherwise the request is refused (`server/jev_adapter/sglang.py:472-508`).
- **SH (`schema-v1-nosys`)**: one user turn holding a preamble, a **preview** list of all questions ("Questions to answer:"), `State:`, then every question with its options. The chat template has an **empty system message**, and adding Mistral's default system prompt "changes the hidden states" the head reads (`HF:8BSH/code/render.py:1-50,104-133`, `HF:8BSH/README.md:141-146`). Token spans are tagged with 5 roles (other/state/preview/question/option). Three alternative wordings and question/option permutations are used as training augmentation only (`render.py:31-50,174-182`).

### Attention & question isolation
- **v2.2: isolation by separate requests.** `Service.score` makes one job per question (× optional rotations) and runs them concurrently with `asyncio.gather`. Each job is a full prompt sent to SGLang with the state repeated. The README launch line sets `--disable-radix-cache`, so the state is **re-prefilled per question** (`server/jev_adapter/service.py:98-138`, `HF:8B/README.md:89-94`). Questions are fully isolated from one another, but there is no shared-state compute saving. The token count `usage.input_tokens` is per evaluation (inferred).
- **SH: no isolation, deliberately.** The backbone runs one causal sequence over the whole request, so later questions see earlier ones and every question sees the preview list. The head then adds `InteractBlock`s with **relation-typed attention biases** over 8 node relations, including `REL_Q_TO_OTHER_Q`, `REL_OPT_TO_OTHER_OPT` and `REL_OPT_TO_OTHER_Q` (`HF:8BSH/code/head.py:20-23,135-154,233-242`). The card frames this as a feature: "places every question and option in one joint graph" (`HF:8BSH/README.md:49-52`). Adding or removing a question can therefore change sibling answers (inferred; untested here), which contradicts Jev's documented guarantee.
- **Order equivariance (SH).** Nodes carry no positional encoding, and span pooling plus structure-only biases make the head "exactly equivariant to the order of choice/noul options and of questions" (`tests/test_equivariance.py` per docstring). Score levels carry an ordinal encoding and are not equivariant (`head.py:6-9`). The backbone's text order still matters, so end-to-end invariance does not follow (inferred).

### Readout head
- **v2.2:** SGLang `max_new_tokens: 0`, `return_logprob`, `token_ids_logprob = <label ids>`. These are the next-token logprobs of the option labels at the assistant boundary, renormalised by softmax over the labels with temperature (`server/jev_adapter/sglang.py:298-424`). This is the LM head; nothing new is trained.
- **SH** (`JointSchemaHead`, config d = 1024, 16 heads, FFN 2,816, 2 route + 2 interact blocks, max 4,096 nodes) (`HF:8BSH/head/schema_head_config.json`, `code/head.py:168-323`):
  1. **Taps** = the final layer and **layer 25** (3B: 19), mixed, projected, plus a role embedding.
  2. One node per question and per option, by **attention pooling** within its token span, concatenated with the span's last token.
  3. Add qtype and kind embeddings, plus an **ordinal MLP feature for score levels**.
  4. **Route blocks** cross-attend from nodes to the state tokens; **interact blocks** run node self-attention with relation biases.
  5. Scorer MLP on `[o, q, o⊙q]` gives one logit per option, divided by a **learned per-type temperature** `exp(log_temp[type])`.

### Question types
choice (2–255), noul, score (`HF:8BSH/README.md:38`). The v2.2 adapter allows up to 255 labels (`A–Z, AA, AB…`, `--label-scheme upper`) but was evaluated only up to 151 options (`HF:8B/README.md:306-307`). **SH code caps score at 10 levels and requests at 256 questions** (`HF:8BSH/code/render.py:28-31,98`). Answers: `noul` = p(true); `score` = Σ i·p_i plus legend; an optional `permutations` 1–16 runs a **rotation ensemble** whose results are averaged (`protocol.py:104,189-194,372-385`).

### Limits
32,768 tokens per request for both lines (SGLang `--context-length 32768`, `render.py:27`; refused, not truncated, per `W`). The SH requests take a single pass, so the limit is per request, not per question.

### Calibration
- **v2.2:** a fixed served temperature, **T = 1.65** for every type. It was not refit for v2.2; the adapter's own default is 1.0. v2 `native` used per-type T (choice 0.85 / noul 0.85 / score 0.70) "chosen on non-JevBench data" (`HF:8B/release-manifest.json`, `README.md:113-119,303-305`).
  - Reported v2 hard-tier ECE: 0.184, vs Jev raw 0.099. Mean TV to stated distributions: 0.108, vs Jev 0.192 (`HF:8B/README.md:204-207`).
- **SH:** a learned per-type temperature inside the head, plus an optional server `--temperature-file` (`code/scoring.py:98-112`). v3 trained hard items with **soft gold 0.8**; JevBench-hard ECE 0.045 (v2: 0.097) (`HF:8BSH/README.md:125-134`).
- **confidence = 1 − normalised entropy** in both lines (`metadata.confidence_method`). This is **not** Jev's closed-form `(p_max − 1/n)/(1 − 1/n)` (`code/render.py:434-436`, `HF:8B/README.md:145`).

## Training
- **Data (v2):** a 520,754-row mixture of synthetic, format-augmented decision data plus about 40 public datasets converted to decisions, with labels from the datasets and distractors generated by code. Datasets include SQuAD 2.0, BoolQ, MNLI, PAWS-X, ContractNLI, CUAD, HotpotQA, MuSiQue, GSM8K, TempReason, MASSIVE, Aegis 2.0, jailbreak and prompt-injection sets, Lichess, and ClinicalTrials.gov (`HF:8B/README.md:242-294`). Nine languages; Korean is a smaller share.
  - An exact-overlap audit against the JevBench public tiers found 0 scenario matches and 181 instruction matches (0.03%); those rows were kept and disclosed (`:296`).
- v2.2 continues from v2.1 "with additional decision data"; adding A–Z/AA labels moved 53–151-option accuracy from 69.8 to 82.7% (`HF:8B/README.md:156-170`).
- **SH stages** (`HF:8BSH/release-manifest.json`):
  1. v2 merged;
  2. `hardgen-v1`, 2,241 LLM-authored hard items with independent-model gold checks, soft gold 0.8, replay, 136 steps;
  3. a repair pass, 57 steps → v3;
  4. "further training on decision data, mostly our own synthetic items" in two rounds → v4.
  - Public train splits of HoVer, Amazon ESCI and ContractNLI were used (`W`).
- **Not disclosed:** learning rate, epochs, loss form (inferred: CE over label logits or head logits) and hardware for training. Evaluation ran on AMD MI350X and H200.

## Evals
- **v2.2 served** (`HF:8B/README.md:160-172`): JevBench public easy 100 / standard 98.61 / hard 56.76 (Jev 1.13: 100 / 98.61 / 72.07); held-out decision set 82.5; long-document 40.0; **Decision Index 0.2.1 = 41.14** (3B 34.85).
- **v2 served vs Jev, same items:**
  - realistic transfer 90.83 vs 86.67;
  - stated-distribution 81.18 vs 72.97;
  - judge proxy 89.33 vs 90.50;
  - JevBench hard 54.95 vs 72.07.
  - In the offline comparison, the tuned 8B is **below its untuned base** on JevBench hard (55.86 vs 60.36) (`HF:8B/README.md:181-204`).
- **SH:** 8B v4 **DI 0.3 public 57.40** (v3 48.34); 3B v3 51.53 (v2 42.90). Self-scored with the official kit over all 140,178 scoreable requests; runs are in `StandardThinking/StandardOne-SH-decision-index`.
  - 8B v4 JevBench public: easy 100 / original 94.44 / hard **75.68**, above Jev's 72.07 on the hard tier. Long-doc 88.0 (`HF:8BSH/README.md:63-77`, `release-manifest.json`).
  - DI 0.3.2 lists "Standard One 3B SH (v2)" and "Standard One 8B SH (v4)" as LoRA + head, autoregressive (`notes/decision_index_0.3.2_entries.tsv`). The 3B entry trails the HF v3.
- **Speed (v2.2, H200, SGLang, serial, ~280 tokens per decision):** 8B p50 25.8 ms / p95 41.9 ms; 3B 22.6 / 33.2 ms. 8B throughput 29.3k → 39.5k tok/s from concurrency 1 to 64 (`HF:8B/README.md:209-221`).

## Serving
- **v2.2:** stock SGLang 0.5.20 (`--disable-radix-cache`, 32k) plus the `jev-adapter` FastAPI process: `/v1/systemone`, alias `jev-latest`, `--max-concurrency`, optional `return_logprobs`, `permutations`, `temperature_scaling`. The response carries `metadata.adapter_elapsed_ms` and `output_tokens: 0` (`HF:8B/README.md:86-152`).
- **SH:** `code/serve_head.py`, a plain-PyTorch server that batches requests, with the backbone in bf16, the head in fp32 and a 127.0.0.1 bind. For production there is the **"Standard One engine"**, an SGLang fork with "a prefill-only decision mode and the schema-head readout", ROCm and CUDA 13 Dockerfiles (`HF:8BSH/README.md:149-199`).

## Licence
Apache-2.0 for the adapters, merged weights, SH head, `server/` code and cards. The base Ministral 3 Instruct is Apache-2.0 with a note against infringing third-party rights; SGLang and the adapter are Apache-2.0; the JevBench harness and tiers are MIT (`HF:8B/README.md:316-318`, `LICENSE`, `NOTICE`). Training datasets carry their own terms, including several CC BY-SA ones. This is the only family in this batch that is fully permissive end to end.

## What is novel vs Jev and kev
1. **A non-Qwen VLM backbone.** It is a Ministral 3 *Instruct* with the Pixtral tower kept, so images can be passed today (unbenchmarked), matching Jev 1.14's "VLM all along" direction. kev and most clones are text-only Qwen.
2. **Two readouts in one family.** v2.2 is "zero new parameters" (label logprobs from an unmodified SGLang, with isolation by separate requests and no prefix sharing). SH is a large **graph-style head** with learned per-type temperatures, multi-layer taps and ordinal features for score. kev's pointer head is one bilinear map.
3. **Explicitly joint questions.** The SH head adds question↔question and option↔other-option attention. This is the opposite of Jev's and kev's isolation guarantee, and it is traded for many-question and long-document gains. Order equivariance is enforced in the head instead.
4. **Calibration choices differ from Jev:** entropy-based confidence, a fixed served T = 1.65 not refit across versions, and soft-gold 0.8 targets on hard items. Reported calibration is mixed: v2.2 ECE was worse than Jev's, while SH v3/v4 is good on JevBench hard.
5. **Evaluation transparency:** the card discloses benchmark-overlap audits, served-vs-offline gaps (TV ≈ 0.06) and regressions against the untuned base. Training hyperparameters stay undisclosed, unlike kev.
