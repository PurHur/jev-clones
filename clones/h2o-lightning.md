# H2O-Lightning-4B (H2O.ai)

A Jev-style decision model. It is a LoRA fine-tune (r64, merged) of the post-trained **Qwen3.5-4B** chat model. Each question is rendered as its own chat prompt, `record / question / options` with one label per option (`A) name: desc`), and the prompt ends in an `Answer:` prefill. The answer is read at the last prompt position as the softmax of the **LM head's logits restricted to the asked label tokens**. That is one forward pass and one "output token" per question, on **stock vLLM 0.30.0** with a standard-library shim that serves TypeSafe's `POST /v1/systemone`. Since v1.3 the same weights also read prompts **two-way**: the 8 full-attention layers run bidirectionally and the 24 Gated-DeltaNet layers stay causal. In that mode vLLM's pooling runner returns the final hidden state, and the shim applies the label rows of the tied embedding in fp32 itself. This is the "pooled FP32 readout".

Source pins used for this profile:

| ref | what | where |
|---|---|---|
| `HF` = `h2oai/h2o-lightning-4b` @ main `898fa17e` (2026-10-19). Weights on main = **v1.3.1** (tag → `7c64d36a`, 2026-10-17) | API JSON, tree, refs, commits, README, `config.json`, `serve_config.json`, `adapter/adapter_config.json`, `h2o_lightning_shim.py`, `pooled_readout.py`, `box.py`, `assets.py`, `serve.sh`, `vllm_patches/*`, `routing_reproducibility/*`, tests, plus the **header only** of `label_rows.lightning` (first 16 KiB, parsed to `label_rows.header.json`; the partial binary was moved to `.trash/`) | `/home/ai/projects/jev-clones/notes/hf-raw/h2oai__h2o-lightning-4b/` (`files/`, `refs.json`, `commits.json`) |
| `HF@v1.0`, `HF@v1.1` | early `serve_config.json` + README | same dir, `rev-v1.0/`, `rev-v1.1/` |
| `HF:nemo` / `HF:gguf` | sibling repos `h2oai/h2o-lightning-4b-nemotron-nano` and `h2oai/h2o-lightning-4b-gguf` (API + README) | `notes/hf-raw/h2oai__h2o-lightning-4b-{nemotron-nano,gguf}/` |
| `GH:vllm` | `github.com/h2oai/vllm`, branches `h2o/per-request-attention` @`236568a780` and `h2o/gemma4-noncausal-full-attention` @`88cd4a0d82`; shallow, blob-less clone, **commit messages only**, never checked out or run | `/home/ai/projects/jev-clones/sources/h2oai-vllm` |
| press | Business Wire release, 2026-10-12, read through [financialcontent.com](https://www.financialcontent.com/article/bizwire-2026-10-12-h2o-lightning-4b-ranks-first-on-jevbench-open-weights-leaderboard). The "technical blog" it links to was not found. | web |

Citation form: `HF:README.md:568` means file:line in the fetched README. `shim:1696` means `files/h2o_lightning_shim.py` line 1696, and `pr:8` means `files/pooled_readout.py` line 8. Code was read as text only and never executed.

> **Main finding.** H2O-Lightning-4B adds **no head and no parameters**. The safetensors dtype split equals the base's exactly: BF16 4,659,861,248 + F32 3,840 (`HF api.json`; `Qwen/Qwen3.5-4B` API gives the same). The F32 tensors are the base's own; they are probably the Gated-DeltaNet `A_log` and norm weights, 24 × (32 + 128) = 3,840 (inferred). The "pooled FP32 readout" is the ordinary next-token LM head evaluated outside vLLM: `logit = h · E[label]`, where `E` is the tied input embedding stored as fp32 rows `[255, 2560]` in `label_rows.lightning`. The design is a careful *generative-LLM-as-classifier* setup, which is the opposite of kev's pointer head. Isolation between questions is exact because **each question is a separate vLLM request**. One-way mode shares the state through vLLM's prefix cache. Two-way mode shares nothing: the bidirectional state depends on the question, so every question recomputes the whole prompt.

---

## Identity

| | |
|---|---|
| Vendor | H2O.ai. vLLM fork commits are by **Jonathan C. McKinney** (`GH:vllm` log), named "Chief of Technology" in the press release; CEO Sri Ambati is quoted too (press). |
| Weights (HF) | `h2oai/h2o-lightning-4b`: created 2026-10-02, 4,944 downloads / 84 likes on 2026-10-19 (`HF api.json`). It holds merged `model.safetensors` (9.32 GB) plus `adapter/adapter_model.safetensors` (520 MB LoRA) (`HF tree.json`). |
| Siblings | `h2oai/h2o-lightning-4b-gguf` (2026-10-10, llama.cpp; decision support proposed in llama.cpp PR #30361) (`HF:README.md:906-912`). `h2oai/h2o-lightning-4b-nemotron-nano` (2026-10-09, on `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, text only, NVIDIA open model licence). Its card self-reports JevBench public 192/231 and DI 0.3 public 53.20 (`HF:nemo README`). Community MLX ports are by netprojects. |
| Lineage | It describes itself as a decision model for JevBench / TypeSafe's wire format. It cites no Hume or kev lineage. The prompt idea (letters + LM head) is the "Tev1-style" family, not the pointer family (inferred). |
| Versions (tags → date) | v1.0 `ef0315d6` (10-02) · v1.1 `193ad740` (10-04, **the JevBench board submission**) · v1.2 / v1.2.1 (10-04) · v1.2.2 (10-08) · v1.2.3 `97becfd2` (10-09, per-type temperatures, the "measured configuration") · v1.3 `90497bfd` (10-16, dual-mode weights; first commit fails to load) · v1.3.0.1–.5 (packaging fixes) · **v1.3.1** `7c64d36a` (10-17, one more pass at a lower lr) · `image` / `image-preview-{1,2,3}` tags and branch (`HF refs.json`, `commits.json`) |
| Status | Production-framed: release gating through `test_card.py` ("every release is gated on it"), sha-checked sidecar modules, and a start-up guard (`HF:README.md:326-330`, `385-389`, `shim:1024-1058`). |

---

## Architecture

### Backbone

| Item | Value | Source |
|---|---|---|
| Base | `Qwen/Qwen3.5-4B` @`851bf6e8` (the **post-trained/instruct** release, not `-Base`) | `serve_config.json:base_model_revision`, `adapter_config.json` |
| Class | `Qwen3_5ForConditionalGeneration` (vision tower kept: 24 layers, d 1024, patch 16, merge 2) | `config.json` |
| Text stack | 32 layers = 24 `linear_attention` (Gated DeltaNet, 16 K-heads / 32 V-heads × 128, conv 4) + 8 `full_attention` (every 4th; 16 q / 4 KV heads, head_dim 256, output gate, partial RoPE 0.25, mRoPE [11,11,10], θ 1e7). d 2560, FFN 9216, vocab 248,320, **tied embeddings**, MTP 1 layer, `max_position_embeddings` 262,144 | `config.json:text_config` |
| Added config key | `"head_dtype": "float32"` at both config levels. It makes vLLM compute the logits in fp32 (`HF:README.md:346`). | `config.json` |
| Trained params | LoRA **r 64, α 128** (scale 2), no DoRA/rsLoRA, `task_type CAUSAL_LM`. Targets: attention `q,k,v,o_proj`, DeltaNet `in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj`, and MLP `gate,up,down_proj`. The README counts "496 adapter tensors across 248 decoder projections" (`HF:README.md:212-220`, `adapter/adapter_config.json`). The adapter is 520 MB on disk (≈ 130M params in fp32 or ≈ 260M in bf16; the dtype is not given, inferred). It is **merged** into `model.safetensors`. Vision encoder unchanged: "fine-tuning did not change" it (`HF:README.md:901-902`). | as cited |
| New vocabulary / head | None. The safetensors param count is identical to the base. Option labels are existing tokens. | `HF api.json` vs `Qwen/Qwen3.5-4B` API |

### Input encoding (one prompt per question)

The shim builds the prompt (`shim:142-213`, `serve_config.json:prompt`):
- `state` is rendered verbatim if it is a string, otherwise as **compact JSON** (`separators=(",",":")`) (`shim:142`). Images lifted out of `state` leave `[image N]` placeholders.
- Options (`options_of`, `shim:161-186`):
  - **noul** is always `[true, false]` (true first). The descriptions come from `criteria.true/false`, else `"the statement holds"` / `"it does not"`.
  - **choice** uses the criteria keys in request order. A null description is replaced by the name.
  - **score** uses the level texts, with names `"0".."L-1"`.
- Labels: the first K of a fixed 255-entry list (`serve_config.json:labels`):
  - `A–Z`, `a–z` (52);
  - 23 Greek lowercase letters `α…ω`;
  - then **180 common BPE fragments** used as labels (`in, er, on, re, at, …, the, …, and, …, The, …, ell`).
  - Each must be **one token** after `"Answer:" + " "`. `check_backend` verifies this against vLLM's tokenizer at start-up and reads each label's token id (`shim:1545-1578`). The label-row header confirms it, e.g. `A`→357, `B`→417 (`label_rows.header.json`, `label_source: "labels via tokenizer.json after 'Answer:'"`).
- User turn, layout `plain` (`shim:196-203`): `record: <state>\nquestion: <instructions>\noptions:\n<L>) <name>: <desc>` per line.
- Full prompt (`render_chat`, `shim:206-213`, `CHATML` at `shim:101`): Qwen ChatML with the fixed system prompt *"You are a decision engine. You read a record and answer one question about it by choosing exactly one option. Reply with a single letter."*, the user turn, an **empty think block** (`<think>\n\n</think>\n\n`, thinking forced off, `shim:897-899`), then the prefill `Answer:`. No BOS is added (`add_special_tokens: False`).
- Images use chat completions with the image parts ahead of the user text and `continue_final_message`. Before the first image request the shim checks that this renders token-identically to the text prompt (`shim:1751-1768`).
- State limit: `max_state_tokens` 32,000. A longer record keeps its **head and tail halves around a `...` line** but is still billed in full (`shim:1581-1590`, `shim:1745-1749`). `--max-model-len` 40,960. A prompt over the context gets a 422.

```
One question = one vLLM sequence (one-way: causal everywhere; two-way: the 8 full-attn layers bidirectional)

<|im_start|>system\n You are a decision engine ... Reply with a single letter.<|im_end|>\n     ┐ shared head across the
<|im_start|>user\n record: {"ticket":"Checkout returns HTTP 500 ...","plan":"enterprise"}\n  ┘ questions of a request
question: Which team should take it?\n                                                        ┐
options:\n A) payments: Payments\n B) web: Web frontend\n C) support: Customer support         │ per-question tail
<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n Answer:                            ┘
                                                                     ^ readout position (last prompt token)
                                                                       logits over {" A"," B"," C"} only

Q1 ──► request 1: [head | tail_1]      one-way : vLLM prefix cache reuses head KV + DeltaNet state (block-aligned)
Q2 ──► request 2: [head | tail_2]      two-way : head attends to tail_k, so no reuse; each request recomputes everything
Q3 ──► request 3: [head | tail_3]      (sent concurrently from a thread pool; vLLM batches them)
```

### Attention and isolation between questions

- **Isolation is exact by construction.** `decide_questions` sends one completion (one-way) or one embeddings call (two-way) **per question**, concurrently from a thread pool (`shim:1696-1743`). No question ever sees another.
- **One-way state sharing** comes from vLLM's automatic prefix cache over the common head. `usage.input_tokens` bills that head once: `Σ prompt_tokens − shared × (N−1)`, where `shared = common_prefix_len` of the tokenised prompts (`shim:1736-1742`, `shim:1817-1828`). On Qwen3.5 vLLM 0.30 caches the recurrent state only at 528-token block boundaries (272 with a bf16 SSM state). It also defers requests whose cached prefix was written in the same step, which caps throughput at about one decision per scheduler step. Two patches in `vllm_patches/` fix this: one is scheduling-only, the other replaces a boolean-mask assignment with a sync-free `torch.where` (`HF:README.md:723-737`, `vllm_patches/README.md`).
- **Two-way** (`--hf-overrides '{"is_causal": false, "text_config": {"is_causal": false}}'`, `--runner pooling --convert embed`, `seq_pooling_type LAST`, `use_activation false`, **prefix caching and chunked prefill off**) (`serve.sh:95-98`, `HF:README.md:589-607`):
  - The 8 full-attention layers attend over the **whole prompt in both directions**, so record tokens also see the question and options. The 24 DeltaNet layers stay causal.
  - The head's representation therefore depends on the question and cannot be cached. `usage.processed_input_tokens` reports the N full prompts actually read (`shim:1828-1832`).
  - The nested override matters. vLLM fork commit `2d290b4` notes that a flat `is_causal=false` on a `ForConditionalGeneration` folder was "a silent no-op … 99.7% of readouts equal to causal" (`GH:vllm`).
- **One-engine mode** (v1.3.1) runs on H2O's vLLM fork `h2o/per-request-attention` @`236568a780`:
  - `vllm_xargs {"attention_mode": "bidirectional"}` makes one request non-causal through the backends' per-sequence causal flag;
  - such a request is prefill-only (`max_tokens` 1), scheduled whole (never chunked, no block-aligned Mamba split), and never reads or writes the prefix cache;
  - Triton attention backend (`HF:README.md:648-676`; commits `ae551b5`, `236568a` in `GH:vllm`).
  - Parity with the pooling engine: 230/231 argmax, max |dp| 0.020 (`HF:README.md:665-670`).
- **Options within a question** are read jointly as an ordered list. One-way is causal, so later options see earlier ones but not vice versa. Two-way lets every option see every other through 8 of 32 layers. No option-order permutation test is published, so order sensitivity is unmeasured (inferred: present in one-way, as for any letter-label LLM reader).

### Readout

**One-way** (`label_logprobs`, `shim:1186-1207`):
- `POST /v1/completions` with `max_tokens 1`, `temperature 0`, `logprob_token_ids = [ids of the asked labels]`, `return_tokens_as_token_ids`.
- vLLM returns the log-probabilities of exactly those ids at the answer slot, so none can drop out of a top-k.
- At most 128 ids per request in vLLM 0.30, so a question with more than 128 options is read in 2 chunked completions (`LOGPROB_CHUNK`, `shim:94`). The full-vocabulary normaliser cancels after renormalising.

**Two-way, the "pooled FP32 readout"** (`pr:1-19`, `pr:197-232`):
```
h          = final hidden state at the last prompt position (after final norm, before lm_head; fp32 base64 from /v1/embeddings)
logit_l    = h · W[l]                      W = label_rows.lightning "W": fp32 [255, 2560] = rows of
                                           model.language_model.embed_tokens.weight (tied lm_head), one token per label
                                           (softcap tanh if the model had one; Qwen3.5: none)
log P(l)   = logit_l − logsumexp_{l' ∈ asked labels} logit_l'          (restricted softmax, floored at 1e-12)
```
- This is exactly what the one-way path reads, but done outside vLLM because the pooling runner returns no logits.
- Parity of the pooled readout vs vLLM logprobs under the same causal attention: 99.32% of 293 vision questions (median |dp| 0.0005), 99.5–99.8% on text (`HF:README.md:153-157`).
- The label-row file is sha-bound to the served `model.safetensors`. The guard refuses an L2-normalised pooled vector (`use_activation` left on) and checks three fixed probes against `two_way.probe_reference` within 0.05. An engine started without the override answers plausibly but differs by about 0.11, so it is refused with a 503 (`pr:190-194`, `shim:1461-1521`, `HF:README.md:639-646`).
- Name note: the file is `.lightning`, not `.safetensors`, because vLLM's loader globbed the v1.3 `label_rows.safetensors` as a weight shard and failed (`HF commits b6eaf373`).

**Answer formatting** (`format_answer`, `shim:718-749`): `p = softmax(logP / T_type)`, then:

| type | answer | confidence |
|---|---|---|
| `noul` | `noul = p[true]`, then **`commit_noul` floor** (below) | none (as Jev) |
| `choice` | first argmax; `probabilities` by key | `(p_max − 1/K)/(1 − 1/K)`, Jev's formula (`shim:678-683`) |
| `score` | `score = Σ i·p_i`; `legend`; `probabilities` | **`max(p)`**, the modal level's probability. This is *not* Jev's MAD-based formula (`shim:740-744`). v1.2.3 used the normalised maximum (`HF:README.md:401-402`). |

- `effort` tiers `balanced` / `thorough` are accepted for API compatibility but always run `fast` with 1 pass, and the response says so (`shim:1824-1828`).
- An optional per-question `threshold` on noul adds a boolean `decision` (`shim:710-716`).

### Calibration

- **Per-type temperatures** (`serve_config.json:temperature`): choice **0.75**, noul **0.80**, score **0.65**; images **0.65**. All are below 1, so they sharpen.
  - Earlier: a single T of 0.85 (v1.0) and 0.80 (v1.1) (`HF@v1.0`, `HF@v1.1 serve_config.json`). Per-type values arrived in v1.2.3 with "answers unchanged" (`HF commits acaf0d4e`).
  - They were fitted on BF16 one-way output. The `two_way` block **reuses the one-way values** as an "interim calibration", with nothing fitted for two-way (`HF:README.md:149-152`, `607-611`). FP8 also reuses them.
- **`noul_floor` 0.801.** If `max(p, 1−p) < 0.801`, the reported P(yes) is moved to **0.801 on its own side** (0.199 for no). A tie goes to yes, and the argmax never changes (`commit_noul`, `shim:700-707`).
  - The shipped model can therefore **never return a yes/no probability in (0.199, 0.801)**. The card reports "no yes/no answer between 0.2 and 0.8" as a measured property (`HF:README.md:116-118`), but it holds by construction.
  - It is present since v1.0 and is "the evaluated configuration" for JevBench and DI. `SHIM_NOUL_FLOOR=0` gives raw probabilities ("routing mode", recommended behind a fallback) (`HF:README.md:766-784`).
  - *Inferred:* the floor is tuned to a JevBench scoring rule that rewards committed yes/no answers. It throws away exactly the uncertainty signal Jev's calibrated noul carries.
- **Box confidence**: a Platt map `(a 0.7088, b 1.6955)` of the box generation's own digit-token probabilities (`serve_config.json:box.generation_calibration`, `box.py`).
- No calibration loss is disclosed. The card frames calibration as "a first-class goal", with "one forward pass, one output token" (`HF:README.md:91`).

### Per-type handling and limits

| | value | source |
|---|---|---|
| choice options | 1..255; "decides best with up to about 16" | `shim:161-186`, `HF:README.md:943` |
| score levels | ≥ 1, ≤ 255 labels | `shim:177-179` |
| questions per request | no limit in code; one vLLM request each (`SHIM_QUESTION_THREADS` default 64) | `shim:1696-1743`, `HF:README.md:743-745` |
| state | 32,000 tokens (head + tail cut), context 40,960 | `serve_config.json`, `shim:1581` |
| images | ≤ 4 per request, ≤ 1,048,576 px each, ≤ 20 MB; data URIs anywhere in `state`, a top-level `images` list, `https` links (fetched by the shim with SSRF guards), multipart, or Jev-style `assets` references | `HF:README.md:510-522`, `701-707`, `shim:20-31` |
| `box` (Jev 1.14 primitive) | **off by default** (400). When on: one guided-JSON **generation** (Qwen `bbox_2d` 0..1000), confidence from its token probabilities (Platt), `present` = a separate yes/no readout, `found`. Needs a generate-mode (one-way) engine. An `edges` mode (per-edge distribution over 100 bins, expected value, Jev's described method) exists in `box.py` but "needs a model trained on edge readouts; keep `generate` with this release". Measured AUC 0.77, ECE 0.056 on 300 held-out COCO/doc/UI images | `HF:README.md:709-716`, `box.py:1-30`, `serve_config.json:box` |

---

## Training

What is disclosed (`HF:README.md:868-902`) is less than for kev or Strands. There are no hyperparameters, step counts, losses or data files.
- **Method:** a LoRA fine-tune (r64/α128, all attention, DeltaNet and MLP projections) of Qwen3.5-4B in **three stages**, merged afterwards. Each example is a record with **one typed question** and "a graded answer". v1.3 "trained for both attention patterns" (causal and two-way). v1.3.1 is v1.3 "continued for one more pass over the training mix at a lower learning rate" (`HF:README.md:113-115`). The loss is undisclosed; CE on the label token is inferred from the readout.
- **Data mix** (share of examples, by stage 1 / 2 / 3 / all):

  | use case | 1 | 2 | 3 | all |
  |---|---:|---:|---:|---:|
  | documents, policies & rules | 34 | 49 | 36 | **37** |
  | numbers, dates & probability | 24 | 41 | 15 | **22** |
  | games & simulations | 23 | 2 | 6 | **12** |
  | everyday language & knowledge | 8 | 1 | 15 | **10** |
  | support & operations | 6 | 4 | 16 | **10** |
  | safety & guardrails | 6 | 3 | 9 | **7** |
  | coding & tool calls | – | – | 3 | **2** |
  | finance & commerce | – | – | 1 | **0.5** |

- **By type:** choice 54%, yes/no 29%, score 16%; yes/no and score rise in stage 3.
- **Language and modality:** all text; over 99% English. Image ability comes entirely from the frozen base vision tower and the shared text readout.
- **Labels:** "real, licence-clean data, with labels from human annotation or agreement among several independent strong models". So it is teacher-agreement labelling, like kev's CFPB delta; whether the targets were hard or soft is not stated (`HF:README.md:90-91`).
- **Contamination controls:** no JevBench or ImageJevBench data; exact and near-duplicate dedup against every eval set; "no rules keyed to any benchmark item's wording"; ship/no-ship rules written before results (`HF:README.md:84-91`, `949-953`). The routing-threshold questions drop any item sharing a 13-gram (or more than two 8-grams) with public JevBench (`routing_reproducibility/ATTRIBUTION.md`).

---

## Evals

**Decision Index board (DI 0.3.2, `notes/hf-raw/_spaces/dix_leaderboard.md`, 2026-10-16): not listed.** No H2O row exists in the 178-line snapshot or `dix_table.tsv`. Board neighbours on the same base: Kev 4B r10 39.5, Tev1-4B 34.14 (both Qwen3.5-4B-Base).

**Self-reported Decision Index 0.3 public suite** (official kit and scorer, shipped T and floor, `max_state_tokens` raised to 1e6) (`HF:README.md:137-152`):

| revision / mode | text (140,178 rows) | vision (12,210 rows) |
|---|---|---|
| **v1.3.1 two-way** | **57.86** | 67.13 |
| v1.3.1 one-way | 56.72 | **67.51** |
| v1.3 two-way / one-way | 57.06 / 55.88 | 67.86 / 67.57 |
| v1.2.3 one-way | 51.41 | 63.09 |
| v1.1 one-way (board weights) | 51.35 | n/a |
| Nemotron-Nano sibling | 53.20 | n/a |

These are public-split numbers, so they are comparable to IBM's card figure (53.30, 30B) and LemonDecider's (50.31), not to board "Full" scores. Two-way gains about 1.1 on text and nothing on vision.

**JevBench** (benchmarkheaven.com, a different benchmark from DI):
- Composite (v1.6.1, equal-weight harmonic mean of Intelligence / Calibration / Speed / Cost): **v1.1 72.5** vs Jev 1.13.0 71.5 vs Quyet-1.0-Large 31B 71.4.
  - Axes: Intelligence 60.0 (Jev 63.6, Quyet 73.4), Calibration 90.0 (Jev 90.6), Speed 92.6, Cost 60.3.
  - Claimed "#1 open-weights composite 10-07 through 10-19" across board v1.6.1–v1.7.46.
  - It wins on speed and cost. It is behind Jev on intelligence (`HF:README.md:36-56`, press).
- Public 231 items (JevBench CLI, `--adapter typesafe`) (`HF:README.md:116-129`, `179-233`):
  - v1.2.3: 205/231 (88.7%), easy 48/48 · standard 71/72 · hard 86/111;
  - v1.3.1: two-way 203, one-way 198;
  - 0 of 74 yes/no answers in (0.2, 0.8), which is guaranteed by the floor;
  - mean 654 input tokens per decision;
  - five hard items are near-ties that GPU numerics flip.

**Images** (v1.2.3, `HF:README.md:236-268`):
- 8/8 on the published ImageJevBench examples.
- **87.4%, ECE 0.017** on 1,877 multiple-choice questions over 16 licence-clean validation sets built from their own annotations (Open Images, VizWiz, TextVQA, CORD, TAT-QA, DocLayNet, OWID charts, Rico).
- Weakest: counting objects (63.3%) and Rico UI goal (65.0%).

**Routing thresholds** (v1.2.3 one-way, floor off, 3,474 held-out questions from `crh225/plumb-decisions` + `caiovicentino1/eikos-decisions`; predictions and script shipped) (`HF:README.md:270-301`):
- Raw accuracy: choice 82.7%, yes/no 84.7%, score 80.2%.
- At threshold 0.9: 53.1% escalated, 98.5% accurate on the rest, 0.7% confidently wrong.

**MLX parity** (793 held-out questions): BF16 / 8-bit agree with vLLM on 99.5% / 99.0%; 4-bit on about 91–92%. ECE 0.033, Brier 0.242 on vLLM BF16 (`HF:README.md:920-939`).

---

## Serving

- **Stack.** Stock **vLLM 0.30.0** + `h2o_lightning_shim.py` (stdlib only; numpy optional), started by `serve.sh` (`HF:README.md:303-345`). Four layouts:
  - two-way pooling engine (default);
  - one-way generate engine (`TWO_WAY=0`, the v1.2.3 setup);
  - two-way + a second generate engine for Box (`LIGHTNING_BOX_ENABLED=1`, two weight copies at 0.45 GPU utilisation);
  - `ONE_ENGINE=1` on the H2O vLLM fork, with the mode chosen per request.
  - Every response states `attention_mode`. Asking for a mode with no engine is a 400, never a silent fallback.
- **API compatibility.**
  - `POST /v1/systemone`, with alias `/api/alpha/decisions`. Jev request shape including `assets`. The response carries `{model, answers, probability_source: "native", effort_used, usage{input_tokens, output_tokens, processed_input_tokens, latency_ms, images}, attention_mode}`. `output_tokens` = one per answered question (`shim:1817-1844`).
  - `POST /v1/decisions` follows **OpenAI's Decisions API** (`client.decisions.create`; predicate/choice/score), mapped onto the same prompt (`HF:README.md:436-466`, `shim:1893-2028`).
  - Status codes: 422 for unanswerable input, which JevBench scores as one wrong answer instead of a run-stopping failure; 502/503 for backend faults (`shim:59-63`). JevBench runs through `--adapter typesafe`.
- **Hybrid mode.** Serve the base Qwen3.5-4B with `adapter/` as a vLLM LoRA, so one server handles both decisions and free text. One-way only. 230/231 agreement with the merged weights. 94 ms vs 60 ms per decision on an A6000.
  - Pitfall: v1.3–v1.3.0.3 shipped adapter keys without `language_model.`, which vLLM silently ignored, so it served the base model (172/231) (`HF:README.md:786-866`).
- **Latency.**
  - Serial: about 29–30 ms p50 per request at the 72 short standard items (110–163 tokens) on an H100 SXM5 or RTX PRO 4500, 69.9 ms p95 over all 231 on the H100. The longest 3,952-token item takes 85.9 ms on the H100.
  - The JevBench board measured 29 ms for v1.1.
  - Two-way serial on an L40S: 38 ms p50 (`HF:README.md:116-120`, `181-233`).
- **Throughput** (H100, one-way, 3 questions × ~740 tokens per decision):
  - stock: 29 decisions/s;
  - patched BF16: 81.6/s at 64 in flight;
  - patched FP8: 99.2/s (FP8 changes about 3% of answers, mostly near-ties).
  - SGLang `/v1/score` was faster only at saturation and agreed on 97.8% of argmaxes. The two-way pooling engine "was not measured at scale" (`HF:README.md:718-764`).
- **Other runtimes:** GGUF (llama.cpp; a native decision route via PR #30361, or stock llama.cpp + shim) and community MLX q8/q4 (`HF:README.md:904-918`). The HF Space demo runs on ZeroGPU.

---

## Licence

- Weights, adapter, shim and tooling: **Apache-2.0** (`cardData.license`, `LICENSE`, `HF:README.md:957-960`).
- Base Qwen3.5-4B: Apache-2.0. vLLM: Apache-2.0.
- Routing data: Plumb is Apache-2.0; Eikos is **CC BY 4.0** (attribution file shipped).
- The Nemotron-Nano sibling is under the NVIDIA open model licence (`license: other`).

---

## Notable design choices

**vs Jev**
1. **One sequence per question, not one shared pass.** Isolation is exact, as with Jev, but by giving each question its own request. A shared state is "encoded once" only through vLLM's prefix cache, and only in one-way mode. Two-way gives up state sharing entirely, so cost scales as N × (state + question), although billing still counts the state once.
2. **Letters on the LM head**, with up to 255 options through a 255-entry label list whose tail is arbitrary BPE fragments (`the`, `and`, `ing`, …). The labels are plain text in the prompt. Jev's unforgeable option boundaries (Hume) have no counterpart here: injected `X) …` lines in the state are indistinguishable from real options. This is inferred from the prompt format; no forgery test is published.
3. **Two-way attention on the full-attention layers** is the Perplexity / LemonDecider / vLLM-SR d3 idea, applied to a 4B and served on stock vLLM through a config override plus the pooling runner. It works without any custom model code.
4. **Calibration is post-hoc and partly cosmetic.** Per-type temperatures are below 1, and the **noul floor at 0.801** makes yes/no outputs bimodal by construction. Jev returns a continuous calibrated P(yes). Score confidence is `max p` rather than Jev's MAD formula. The choice confidence formula matches Jev's.
5. **Wider API surface than Jev:** images (up to 4, URL fetch), Jev 1.14's `assets` and `box`, OpenAI Decisions, and hybrid decisions+text on one server. Box is a constrained *generation*, not Jev's described per-edge expectation; that path is present but untrained.

**vs kev**
1. **Head:** kev trains a pointer head (W_q, W_k over `</opt>` and `<decide>` hidden states) on a frozen **Base** LM. H2O trains **no head** and uses the **instruct** model's tied LM head over label tokens with a chat template. The H2O approach is cheaper to serve (stock vLLM, GGUF, MLX) but ties options to single-token labels and inherits LM label-prior biases.
2. **Isolation mechanism:** kev on Qwen3.5 uses causal rows continuing a *cached state replica* inside one process. H2O delegates the same thing to vLLM's prefix cache, one HTTP request per question. Only H2O has a bidirectional variant, and kev has no two-way reading.
3. **Adapter scale:** kev uses LoRA r16/α32 at low lr on a frozen backbone (lesson: preserve base knowledge). H2O uses r64/α128 and merges the result. Both adapt the DeltaNet projections. Neither adds vocabulary.
4. **Calibration:** kev stores one temperature in `head.pt`, fitted on held-out datasets since U. H2O ships per-type temperatures plus the noul floor in `serve_config.json`, fitted on BF16 one-way output and reused unrefitted for two-way and FP8.
5. **Evidence:** kev publishes code, data generators, frozen suites and board entries (Kev 4B 39.5 on DI 0.3.2). H2O publishes the serving stack in full but **no training code, data or hyperparameters**. Its headline numbers are self-run (DI public 57.86) or from JevBench (composite #1), not from the DI board. The two DI numbers are not comparable splits.
