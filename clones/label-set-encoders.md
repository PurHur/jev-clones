# Label-set encoders: Knowledgator GLiClass v4 and Fastino GLiNER2.5-Decide

Two families that come from zero-shot classification rather than from Jev, and that now market themselves as
decision models. Both put the **candidate labels inside the input sequence** and score each label from the
hidden state of a marker token. Neither generates text. GLiClass v4 9B/27B goes further: a hybrid Qwen decoder
reads the context into a KV cache, the labels are appended as transient tokens, and a small bidirectional
"scorer" plus a learned per-pair calibrator sit on top.

Source pins used for this profile (compiled 2026-10-19):

| ref | what | where |
|---|---|---|
| `GC:` | `Knowledgator/GLiClass` HEAD `6a00f80` (2026-10-16, "Bump version to 0.1.21"), shallow clone, read only, never executed | `sources/gliclass/` |
| `G2:` | `fastino-ai/GLiNER2` HEAD `b86d35c` (2026-10-09, merge of PR #189 "transformers-v5-support"), shallow clone, read only | `sources/gliner2/` |
| `HF:<repo>` | API JSON (`api.json`, refreshed as `api-refresh.json` with `?expand[]=safetensors,sha,...`), `tree.json`, `files/` (config, README, calibration report, tokenizer/processor config, 27B weight index) | `notes/hf-raw/knowledgator__gliclass-v4-{edge,9b,27b}/`, `notes/hf-raw/fastino__GLiNER2.5-{Decide,Decide-1B,multi-Decide}/` |
| board | Decision Index 0.3.2 snapshot (2026-10-16) | `notes/hf-raw/_spaces/dix/index.json`, `dix_leaderboard.md` |
| papers | SCX Router arXiv 2609.02292 (abs + HTML, via WebFetch); GLiClass arXiv 2508.07662; GLiNER2 arXiv 2507.18546 | web |

No weights were downloaded and no upstream code was run.

> **Main findings.**
> 1. GLiClass v4 9B/27B ("decoder-KV") is structurally almost the same as kev: a causal Qwen backbone, the
>    label text followed by a marker token (`<<LABEL>>` here, `</opt>` in kev), and a closing token (`<<SEP>>`
>    here, `<decide>` in kev) that has read all labels. The differences are on top: a 2-layer bidirectional
>    DeBERTa encoder over the label section, an MLP on `[sep; label]` instead of a dot product, up to 3 gated
>    recurrent refinement steps, and a learned per-pair calibrator. The input order is **instruction before
>    state**, so the state prefix cannot be shared across questions that have different instructions.
> 2. The shipped calibrator was fitted with a calibration script that is **not in the public repo** (objective
>    `mixed`, `rank_preserving`, `balance_confidence`). On its own held-out split it **raises** aggregate
>    pair-level ECE (9B: 0.009 → 0.057) and NLL (0.135 → 0.166). It helps only on the over-confident and mixed
>    sub-groups. The calibration data file is called `decision_index_calibration.json`.
> 3. GLiNER2.5-Decide is a plain GLiNER2 schema encoder (DeBERTa-v3-large / Ettin-1B / mDeBERTa-base). Several
>    "heads" (questions) share one bidirectional sequence with **no isolation**. Each label gets one logit from
>    an MLP on its `[L]` marker. Training uses independent sigmoid BCE, but single-label inference applies a
>    softmax, and there is **no calibration step**. It is strong on its own operational suite (fast-decisions
>    60.2%) and weak on the general Decision Index (9.24, rank 155).

---

## A. Knowledgator GLiClass v4

### Identity

| | |
|---|---|
| Org | Knowledgator Engineering (Kyiv). Authors on the cited papers: Ihor Stepanov, Aleksandr Smechov, Mykhailo Shtopko, Dmytro Vodianytskyi, Oleksandr Lukashov (`HF:gliclass-v4-9b/files/README.md:563-569`) |
| Code | github.com/Knowledgator/GLiClass, Apache-2.0, PyPI `gliclass` 0.1.21 (`GC:pyproject.toml:9`, `GC:LICENSE`) |
| Lineage | GLiClass (arXiv 2508.07662, 2025-08: a uni-encoder zero-shot classifier "inspired by GLiNER"); the decoder-KV architecture comes from **SCX Router** (arXiv 2609.02292, 2026-09-02: a Qwen3-0.6B router whose labels are candidate models) (`HF:gliclass-v4-9b/files/README.md:32-33`) |
| Checkpoints | `knowledgator/gliclass-v4-edge` (created 10-16, sha `90be5550`, 439.8M = 438.7M BF16 + 1.12M F32), `gliclass-v4-9b` (10-13, sha `14d11a92`, 8.98B = 8.975B BF16 + 4.28M F32), `gliclass-v4-27b` (10-13, sha `acd9c277`, 27.00B = 26.99B BF16 + 5.33M F32) (`HF:*/api-refresh.json`) |
| Downloads / likes (10-19) | edge 274 / 11, 9b 114 / 11, 27b 48 / 7 |
| Older lines (not profiled) | `gliclass-instruct-*-v1.0` (2026-02), `gliclass-multilang-*` (2026-04), `gliformer-*` (2026-09/10), all from `?author=knowledgator` |

The F32 parameters are exactly the calibrator. For 9B: `(4·4096+6)·256 + 256² + 2·256 + 2·4096 LayerNorm ≈ 4.28M`, which matches `F32 4,278,786`. The 27B index lists only `calibrator.*`, `model.scorer.*` and `model.decoder_model.{language_model,visual}.*` tensors (`HF:gliclass-v4-27b/files/model.safetensors.index.json`).

### Architecture

#### Backbone

| ckpt | `architecture_type` | backbone | layers / d | trained-on-top modules | ctx | src |
|---|---|---|---|---|---|---|
| edge | `uni-encoder` | `microsoft/deberta-v3-large` (`DebertaV2Model`, relative attention, 512 abs positions, position_buckets 256, vocab 128,004 = +3 special) | 24 / 1024, 16 H | `FeaturesProjector` ×2, `MLPScorer` (2d→1024→512→1), calibrator | card serves up to `max_length=16384` (`HF:edge/files/README.md` server block) | `HF:gliclass-v4-edge/files/config.json:2,17-68,124,145-151` |
| 9b | `decoder-kv` | `Qwen/Qwen3.5-9B` loaded as `Qwen3_5Model` (multimodal, no LM head): 32 L = 24 Gated-DeltaNet + 8 full attention, 16 q / 4 KV heads, head_dim 256, partial RoPE 0.25, 262,144 positions, 27-layer vision tower, vocab resized to 248,080 | 32 / 4096 | `DecoderKVScorer` (2-layer DeBERTa encoder + projectors + MLP + `RecurrentReasoningCell`), calibrator | 262,144 native; card examples `max_length=8192`, DI runs 32,768 | `HF:gliclass-v4-9b/files/config.json:2,18-161,218-255` |
| 27b | `decoder-kv` | `Qwen/Qwen3.8-27B`: 64 L (48 DeltaNet + 16 full), d 5120, 24 q / 4 KV, FFN 17,408 | 64 / 5120 | same | same | `HF:gliclass-v4-27b/files/config.json` (diff vs 9b: only `encoder_model_name`, `hidden_size`, `encoder_config`) |

- **Decoder-KV loading.** With `multimodal: true` and a `Qwen3_5Config`, the backbone is `transformers.models.qwen3_5.Qwen3_5Model`, i.e. text + vision, no LM head (`GC:gliclass/model.py:1009-1020`). Text-only Qwen3.5 configs collapse to `Qwen3_5TextModel` (`:1027-1035`). Qwen3, Gemma 4 and EmbeddingGemma 2 are also accepted; EmbeddingGemma 2 is bidirectional and has no KV cache (`:1006-1007, 1288-1293`). The vocabulary is resized for the three added tokens (`:1058-1061`). Vision/audio towers are frozen by default (`freeze_media_encoders: true`, `:1063-1068`).
- **Special tokens (9B/27B).** `class_token_index 248077` = `<<LABEL>>`, `sep_token_index` = `text_token_index` = 248078 = `<<SEP>>`, `example_token_index 248079` = `<<EXAMPLE>>` (`HF:gliclass-v4-9b/files/config.json:12,165,246,249`). Edge: 128001 / 128002 / 128003 (`HF:edge/files/config.json:11,70,153`).
- **What is trained.** Not stated per checkpoint. Training stage 1 is "large-scale supervised training" with the recurrent steps trained "in the same run" (`HF:9b/files/README.md:540-543`). The library supports a frozen decoder (only the scorer and the new token rows train) or partial unfreezing of the last layers (`GC:train.py:80-120`). SCX Router lists separate learning rates for decoder and scorer (paper §training, via WebFetch). Whether the v4 9B/27B backbones are fully fine-tuned, partly fine-tuned or frozen is **unknown**: the Hub carries a full backbone copy in every case, so file sizes do not answer it. The 9B calibration report names its source `models/qwen3.8-9b-fast/last`, although the config says `Qwen/Qwen3.5-9B` (`HF:9b/files/calibration_report.json` `args.model_path`). This is an internal naming inconsistency; the config is authoritative.

#### Input layout

Decoder-KV (9B/27B), `format_decoder_kv_sequence` (`GC:gliclass/data_processing.py:12-49`):

```
[prompt][examples][media]text <<SEP>> label_1 <<LABEL>> label_2 <<LABEL>> ... label_K <<LABEL>> <<SEP>>
`------------- context (cacheable) ------------'`------------- label section (transient) ------------'
```

- Prompt, examples, media placeholders and text are simply **concatenated with no separator** (`f"{prompt}{examples}{media}{text}"`, `data_processing.py:17`). Few-shot examples are `<<EXAMPLE>>text \nLabels:\n l1, l2…` with a closing `<<SEP>>` (`:353-365`). There is no chat template in the classic pipeline. A `chat_template.jinja` is shipped with the 9B/27B repos for the processor but is not used by `format_decoder_kv_sequence`.
- **The label marker comes after its label text**, as in kev's `</opt>`. The label section is the only input of the scorer (`scorer_full_sequence: false`, `GC:gliclass/config.py:186-188`).
- **Truncation.** The classic pipeline tokenizes the label section first and right-truncates the context to `max_length − len(label_ids)` (`GC:gliclass/pipeline.py:852-861`), so the labels are never cut but the end of the state silently is. Media get a budget via `process_multimodal_with_budget(..., reserved=len(label_ids))` (`:875-885`). The card's Jev server passes `max_length=10**9` and refuses inputs over `MAX_LENGTH` with HTTP 422 instead (`HF:9b/files/README.md:272, 321-323`).
- **Label order.** Training shuffles labels (`shuffle_labels: true`; `GC:data_processing.py:398-399`). Augmentation also removes labels (0.15), adds random labels (0.10), adds descriptions (0.25), synonyms (0.1) and examples (0.25) (`GC:data_processing.py:52-64`).

Uni-encoder (edge), `UniEncoderZeroShotClassificationPipeline.prepare_input` (`GC:gliclass/pipeline.py:602-627`):

```
[CLS] <<LABEL>>label_1 <<LABEL>>label_2 ... <<LABEL>>label_K <<SEP>> prompt text [examples] [SEP]
       ^ marker BEFORE its label (label-first)                          (prompt_first = true)
```

Labels go first, so right truncation (`truncation=True, max_length`) cuts the text, not the labels (`pipeline.py:643-645`).

#### Attention and isolation

```
Decoder-KV, classic forward (what the Jev server and the DI numbers use):
  one causal pass over   context ... <<SEP>> l1 <<LABEL>> l2 <<LABEL>> ... <<SEP>>
                                       |--- label i sees context + labels < i (causal; DeltaNet state carries it too)
  then DeBERTa scorer (bidirectional, 2 layers) over the label section only
                                       |--- labels see each other in both directions, but see the text ONLY
                                            through the backbone states of the label tokens

Decoder-KV, streaming / cached path:
  update_decoder_cache(context chunk)        -> persistent KV (+ DeltaNet conv/recurrent state)
  classify_from_decoder_cache(label section, explicit position_ids, use_cache=False)
                                             -> label tokens read the cache, nothing is written back
```

- **"Transient label tokens."** In the cached path the label section runs with `past_key_values=cache, use_cache=False`, and its hidden states are sliced from the end of the output (`GC:gliclass/model.py:1313-1343`). The label KV is never added to the cache, so the label roster can change per call without contaminating the session state. SCX Router describes exactly this (paper: "label-stage keys and values are not written back to the persistent cache"). The streaming pipeline puts the prompt and examples in the cached prefix and appends only new text chunks (`GC:gliclass/streaming/pipeline.py:36-37, 211, 423`).
- **In the classic forward there is no such isolation trick.** The whole sequence, labels included, is one causal pass (`model.py:1415-1423`), and the label section is cut out afterwards by locating the last `<<SEP>>` before the first `<<LABEL>>` (`_extract_label_section`, `:1208-1286`). The result is the same as the cached path because attention is causal; "transient" only matters when the cache is reused.
- **Across questions.** Under the Jev mapping each question is its **own pass** (`HF:9b/files/README.md:237-240, 336-339`), so questions are isolated by construction. The card suggests batching questions that share a state as separate rows (`:392`). **Inferred:** the cached path could share one state prefix across questions only if they share the instruction, because `prompt` sits *before* the text in the context. Jev, and kev's branch layout, put the state first and the question after it.
- **Within a question.** Options interact twice: causally in the backbone (later labels see earlier ones), then bidirectionally in the scorer encoder. There is no positional trick against order bias. Only training-time shuffling addresses it.

#### Readout head (decoder-KV)

`DecoderKVScorer` (`GC:gliclass/scorers.py:248-495`):

```
H      = last_hidden_state[label section]                 (B, Ls, d)        from the backbone
S_1    = DebertaV2Encoder_2L(H)                            d heads = d/64, FFN 4d, rel. attention p2c/c2p, 512 rel positions   (:266-281)
t      = W_t · S[last <<SEP>>]   (text repr; the closing SEP has read all labels)                   (:313-335, 478)
l_j    = W_l · S[j-th <<LABEL>>]                                                                   (:337-348, 481)
z_j    = MLP([t ; l_j])          MLP: 2d → 1024 → 512 → 1, ReLU (scorer_mlp_hidden_size 1024)       (:291-300, 487-491)

Recurrent refinement (recurrent_steps 3 in 9B/27B config, 1 in edge), RecurrentReasoningCell (:191-245):
  x   = LN(H)
  r   = σ(W_r [h; x]);  h~ = Enc(r⊙h + (1−r)⊙x);  u = σ(W_u [h; h~]);  h' = (1−u)⊙h + u⊙h~
  each step re-scores z from h'; Enc is the SAME 2-layer DeBERTa encoder
  inference halts an example when max_j |p_j^(t) − p_j^(t−1)| < recurrent_halt_threshold (0.01)   (:440-464)
```

- `recurrent_read_text: false` in both configs, so the refinement steps never re-read the text (`HF:9b/files/config.json:238`; `GC:scorers.py:309`). All refinement happens over the label section.
- Normalisation of features is off (`normalize_features: false`), so `logit_scale` is unused on this path.
- The edge readout is GLiClass's original uni-encoder: the `[CLS]`/first-token pooled text vector and the `<<LABEL>>` token states (`embed_class_token: true`, `class_token_pooling: first`) each go through a 2-layer `FeaturesProjector` and then the `MLPScorer` on `[t; l_j]` (`GC:gliclass/model.py:176-267, 467-490`; `GC:scorers.py:72-97`; `GC:gliclass/layers.py:49-63`).

#### Per-type handling (Jev mapping on the card)

| Jev type | GLiClass labels | activation | answer | src |
|---|---|---|---|---|
| `choice` | description per key (key if null), ≥ 2 | softmax over calibrated logits | `choice` = argmax, `confidence` = **p_max** (not Jev's rescaled formula), `probabilities` | `HF:9b/files/README.md:302-307, 324-325, 343-344` |
| `noul` | `[true-desc or "yes", false-desc or "no"]`, **true first** | softmax over the 2 | `noul = p["true"]`, no confidence | `:308-310, 345-346` |
| `score` | level descriptions in order, ≥ 2 | softmax | `score = Σ i·p_i` (expected **index**, not 0..1), `confidence` = p_max | `:311-314, 347-349` |

- A JSON state is rendered as `key: value` lines, and string lists as `- item` lines (`:286-299`).
- The model itself is trained as **multi-label** (`problem_type: multi_label_classification`, focal-BCE per label, `focal_loss_alpha 0.7`, `gamma −1` = no focusing) (`HF:9b/files/config.json:167-169, 225`; `GC:gliclass/loss_functions.py:82-90`; `GC:gliclass/model.py:1121-1134`). The softmax used for `choice`/`score`/`noul` is an inference-time choice (`classification_type="single-label"`, `GC:gliclass/pipeline.py:290`). README stage 1 says single-label and scoring examples are mixed in (`HF:9b/files/README.md:542-543`), but the shipped config's loss is the per-label focal-BCE.
- There is no ordinal structure for `score`; levels are ordinary labels.

#### Calibration

The calibrator math is from `GC:gliclass/calibration.py` (at 6a00f80):

```
features_j = [LN(t); LN(l_j); LN(t)⊙LN(l_j); |LN(t)−LN(l_j)|; φ(z)_j]            (:120-123)
φ(z)_j     = [z_j, |z_j|, z_j − max z, z_j − mean z, mean binary entropy, log K]    (:40-65)
(o1, o2)   = MLP_256(features_j)   (Linear → GELU → Dropout → Linear → GELU → Dropout → Linear)   (:79-87)
β_j        = beta_max · σ(o1)  ∈ (0, 3)          b_j = o2  (calibrator_use_bias: true)          (:126-128)
z'_j       = β_j · z_j + b_j                     calibrated logits replace `logits`; raw kept as `uncalibrated_logits` (GC:model.py:1517-1528)
init       = output layer zeroed so (β, b) = global temperature/Platt fit (β0, b0)              (:90-100)
loss       = NLL(σ(z') or softmax(z') for one-hot texts) + 0.01·(log β − log β0)² + 0.01·(b − b0)²   (:245-278)
```

- The backbone is frozen and representations are cached once (`collect_calibration_examples`, `:162-210`). The public `calibrate.py` reweights pairs so every (raw-confidence bin × correct/wrong) cell has equal loss mass (`GC:calibrate.py:1-10, 251-259`).
- **Shipped fit ≠ public script.** The 9B and 27B `calibration_report.json` args include `objective "mixed"`, `softmax_weight 0.5`, `rank_preserving true`, `balance_confidence true`, `confidence_threshold 0.7`, `skip_long true`, `num_epochs 3`, `lr 1e-4`, `max_length 2048`, `max_examples 10000`, `eval_ratio 0.2`, data `data/decision_index_calibration.json` (`HF:9b/files/calibration_report.json` `args`; the 27B report is identical apart from paths). None of `objective`, `softmax_weight`, `rank_preserving`, `balance_confidence` or `confidence_threshold` exists in `GC:calibrate.py:232-285` or `GC:gliclass/calibration.py`, and the report's `groups` breakdown is not produced by the public `evaluate_calibrator` (`:358-402`). **Inferred:** `mixed` blends the per-pair sigmoid NLL and the softmax cross-entropy with weight 0.5, and `rank_preserving` constrains (β, b) so that the label order within a text cannot change. The second matters because a per-pair bias can otherwise flip rankings and decisions (the docstring warns of this, `calibration.py:10-14`).
- **Reported effect (pair level, per-label sigmoid):**

| | uncal. ECE / NLL / Brier | global scaling ECE | calibrated ECE / NLL / Brier | β mean (min–max) | β0 / b0 |
|---|---|---|---|---|---|
| 9B held-out (1,937 texts) | **0.0094** / **0.135** / 0.040 | 0.025 | 0.057 / 0.166 / 0.047 | 0.39 (0.02–0.61) | 0.540 / −0.484 |
| 27B held-out (1,937 texts) | **0.0079** / **0.135** / 0.038 | 0.025 | 0.070 / 0.174 / 0.048 | 0.36 (0.06–0.49) | 0.581 / −0.510 |
| 9B "overconfident" group (180) | 0.171 | 0.110 | **0.072** | | |
| 9B "normal" group (1,436) | **0.023** | 0.041 | 0.074 | | |

  Source: `HF:gliclass-v4-{9b,27b}/files/calibration_report.json` (`held_out`, `held_out.groups`). The calibrator (β ≈ 0.4, so logits are *softened* about 2.5×) is worse in aggregate on its own held-out split. It improves the over-confident and mixed groups and degrades the large "normal" group. The card's claim that "a score of 0.8 means the label is correct about 80% of the time" (`HF:9b/files/README.md:90`) is not supported by these numbers at the aggregate level. Softmax-level (single-label) calibration, which is what the Jev mapping returns, is not reported.
- **Contamination risk (inferred).** The calibration file name says it is derived from the Decision Index, and the cards report DI scores computed with the calibrated logits (`HF:9b/files/README.md:394-395`).
- Edge ships a calibrator (`use_calibrator: true`, `HF:edge/files/config.json:155`) but no calibration report.

#### Limits

- Labels: `max_num_classes 25` with `max_labels_alloc "dynamic"`: the pipeline sizes the label buffer from the request, so 25 is not a hard cap (`HF:9b/files/config.json:218-219`; `GC:gliclass/pipeline.py:379-384`). The scorer's DeBERTa encoder has `max_relative_positions=512` over the label section (`GC:scorers.py:272-280`), which in practice bounds the useful label-section length (**inferred**: this is a relative-position bucket limit, not a hard error).
- Context: backbone 262k; the classic pipeline right-truncates the context when `max_length` is set, and the card server refuses over 32,768 (9B) or 16,384 (edge) (`HF:9b/files/README.md:267, 322-323`).
- Stated weak spots: multi-step arithmetic, GPQA/HLE, "long multi-question documents", non-English (`HF:9b/files/README.md:552-558`).

### Training

1. **Supervised.** A mixture recasting classification, NLI, QA, reranking, extraction, tool calling, moderation, judging and reasoning as decisions over candidate labels, with instructions, label descriptions and images. Focal loss. Recurrent depth is sampled uniformly from [`recurrent_min_steps`, `recurrent_steps`] = [1, 3] per batch (`HF:9b/files/README.md:540-543`; `GC:gliclass/model.py:1104-1107`). The recurrent loss is the mean step loss plus a hinge `0.1·relu(l_t − 0.95·sg(l_{t−1}))` for t ≥ 3 (`recurrent_improvement_coef 0.1`, `margin 0.05`); the confidence term is off (`recurrent_confidence_coef 0.0`) (`GC:model.py:1156-1206`; `HF:9b/files/config.json:229-234`).
2. **RL.** "PPO-style" over label decisions, with task-metric rewards (F1, accuracy, recall), a value baseline, a clipped ratio, and optional KL and entropy terms (`HF:9b/files/README.md:544-545`). Implementation: `GC:gliclass/training.py:697-873` (`cliprange`, value MSE, `kl_beta`, `entropy_beta`) and `GC:train_rl.py`.
3. **Calibration.** Frozen backbone, per-pair (β, b) MLP, NLL on held-out data (see above).

- Data, step counts, learning rates and hardware for v4 are **not published**. SCX Router (the 0.6B ancestor) reports Qwen3-0.6B, max length 4,096, 3 epochs / 18,300 steps at batch 4, bf16, a second stage at lr 1e-6, trained on 150k verifier-scored + 15k open-ended tasks generated from a 23-family / 115-type ontology (arXiv 2609.02292).
- No Jev distillation is claimed. Jev-agreement suites appear only as evals.

### Evals

All numbers are self-reported on the cards. The GLiClass v4 models are **not on the public Decision Index board snapshot** of 2026-10-16.

**Decision Index 0.3, public part** (37 benchmarks, 140,178 requests; one pass per question through the Jev mapping; RTX PRO 6000; `HF:9b/files/README.md:403-425`):

| model | public skill | raw | Knowledge | Language | Retrieval | Tools | Arts | median ms/request |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Edge 0.4B | 47.49 | 59.70 | 21.4 | 54.6 | **66.7** | 58.1 | 38.6 | 11.0 |
| 9B | 55.56 | 66.10 | 36.6 | 61.6 | 61.6 | 74.8 | 41.5 | 34.0 |
| 27B | 61.74 | 70.76 | 45.7 | 67.6 | 63.0 | 81.0 | 50.3 | 73.7 |
| pplx-decider 27B (board) | 62.25 | 71.66 | 52.0 | 67.5 | 61.3 | 78.9 | 47.0 | |
| Jev 1.13 (board) | 57.96 | 68.55 | **53.9** | 59.2 | 55.4 | 75.1 | 39.1 | |
| Kev 27B (board) | 56.69 | 67.55 | 45.7 | 62.6 | 55.7 | 72.6 | 42.7 | |

- The pattern: strong on retrieval/classification and tools, weak on knowledge. 9B GPQA 25.2 vs Jev 71.4, MMLU-Pro 46.4 vs 80.5, BBH 55.8 vs 89.7 (`:436-446`). The edge model beats every larger model on BANKING77 (90.7) and CLINC (92.2) but scores 0.0 on CRUXEval and 2.1 on GSM8K (`:438-441, 466-467`).
- **KGJevBench** (Knowledgator's own suite, 28 rows; `:497-533`): 27B average 82.2 vs pplx-decider-27b 82.6 and clef 78.6. GLiClass leads on triple relevance (edge 90.6 vs ≤ 47 for decoders), task routing and KG path ranking. It trails on JevJudge text (27B 32.2 vs 52.4 for glide), typed decisions (68.5 vs 74.2) and Jev Distill agreement (61.0 vs 72.7).
- No calibration metric (ECE/Brier) is reported for the DI runs.

### Serving

- Python: `GLiClassModel.from_pretrained` + `ZeroShotClassificationPipeline(classification_type=..., max_length=...)` (`HF:9b/files/README.md:57-72`). Labels accept hierarchical dicts and few-shot examples (`:137-138`).
- Streaming: `StreamingZeroShotClassificationPipeline` keeps one text-only KV cache per session and encodes only the newest chunk. Its strategies (`EveryChunk`, `EveryNTokens`, `OnDelimiter`, `SlidingWindow`, `Composed`) decide when to score. Streaming is text-only (`:198-224`; `GC:gliclass/streaming/`). This is the one place where the architecture's cacheable-prefix property is used.
- A Jev-compatible `POST /v1/systemone` + `GET /v1/models` server is given only as a **card snippet** (FastAPI, one question at a time, `usage.output_tokens: 0`), not as a library entry point (`HF:9b/files/README.md:250-356`). The library server is a Ray Serve deployment with dynamic batching and PolyLoRA adapters (`GC:gliclass/serve/server.py:1-40`).
- Optional FlashDeBERTa kernels for the scorer (`USE_FLASHDEBERTA`, `GC:scorers.py:282-285`).

### Licence

Apache-2.0 for code (`GC:LICENSE`) and all three checkpoints (`HF:*/api-refresh.json cardData.license`). Bases: DeBERTa-v3-large (MIT), Qwen3.5-9B and Qwen3.8-27B (Apache-2.0), bge-small-en-v1.5 (MIT; referenced in `label_model_config`, but used only by the bi-encoder architectures, not by uni-encoder or decoder-kv; `GC:gliclass/config.py:104-113`). Training data are not itemised.

---

## B. Fastino GLiNER2.5-Decide

### Identity

| | |
|---|---|
| Org | Fastino AI (Palo Alto). GLiNER2 authors: Urchade Zaratiana, Gil Pasternak, Oliver Boyd, George Hurn-Maloney, Ash Lewis (arXiv 2507.18546) |
| Code | github.com/fastino-ai/GLiNER2, Apache-2.0 (`G2:LICENSE`); PyPI `gliner2` (`[local]`, `[train]` extras) |
| Checkpoints | `fastino/GLiNER2.5-Decide` (created 09-23, sha `9d1bfb84`, 486.4M F32, base `fastino/gliner2-large-v1`, **119,592 dl / 419 likes**); `GLiNER2.5-Decide-1B` (09-24, sha `67af949e`, 1.19B F32, base `fastino/gliner2-xl-0111`); `GLiNER2.5-multi-Decide` (09-24, sha `e173bca1`, 287.4M F32, base `fastino/gliner2.5-multi-v1`) (`HF:fastino__*/api-refresh.json`) |
| Card sizes | "340M" for Decide (DeBERTa-v3-large without its 128k-row embedding matrix, **inferred**; the API counts 486M), "287M" multi, "1B" |
| Positioning | "a specialist for operational decisions … not a general-purpose model. It does not reason" (`HF:GLiNER2.5-Decide/files/README.md:60, 516-517`) |
| Siblings | Fastino also has an autoregressive **GLiDE** 28B on the DI board (60.21, ECE 0.040, rank 16; no HF weights) and hosted fine-tuning via `api.fastino.ai` (`SKILL.md`). Not profiled here |

### Architecture

#### Backbone

| ckpt | `architecture` | encoder | L / d / H | positions | src |
|---|---|---|---|---|---|
| Decide | `span` (`span_mode markerV0`, `max_width 8`, `count_lstm`) | `microsoft/deberta-v3-large`, vocab 128,011 (+10 special) | 24 / 1024 / 16 | 512 abs., relative attention (buckets 256) | `HF:GLiNER2.5-Decide/files/config.json` |
| Decide-1B | `span` | `jhu-clsp/ettin-enc-from-dec-1b` (ModernBERT: full attention every 3rd layer, others 128-token local, RoPE θ 160k) | 28 / 1792 / 28 | 7,999 | `HF:GLiNER2.5-Decide-1B/files/config.json` |
| multi-Decide | `boundary` (boundary/candidate/record/relation heads, abstention) | `microsoft/mdeberta-v3-base`, vocab 250,112 | 12 / 768 / 12 | `max_len 4096` (word tokens) | `HF:GLiNER2.5-multi-Decide/files/config.json` |

- All weights are trained, encoder included (full fine-tune from the gliner2 bases; the DI board lists "full fine-tune"). LoRA is supported for user fine-tunes (`G2:gliner2/models/span/model.py:960-1060`).
- The classification head is one MLP shared by every label: `d → 2d → 1`, ReLU, no dropout (`G2:gliner2/models/span/model.py:119-127`; boundary variant with dropout 0.1, `G2:gliner2/models/boundary/model.py:1255-1262`).

#### Input layout

GLiNER2 schema serialisation (`G2:gliner2/processor.py:336-359, 1295-1330, 1542-1568`):

```
( [P] task[: prompt] [DESCRIPTION] l_a: desc_a ... [EXAMPLE] x [OUTPUT] y ( [L] l_1 [L] l_2 ... [L] l_K ) ) [SEP_STRUCT]
( [P] task2 ... ( [L] ... ) ) [SEP_STRUCT] ... [SEP_TEXT] w_1 w_2 ... w_N
  ^ one parenthesised schema per head (question), all BEFORE the text
```

- The `[L]` marker precedes its label (label-first, like GLiClass edge). The embedding at the first subword of each `[L]` is the label's representation (`token_pooling: first`). The routing only collects the structural slots (`[P]` at offset+1 and `[L]` at offsets 4, 6, …) so that special tokens inside prompts or descriptions cannot create extra logits (`processor.py:1556-1568`).
- **The prompt and label descriptions live in the `[P]` slot, not next to each label**: `prompt_str = f"{task}: {prompt}"` plus ` [DESCRIPTION] label: desc` for each described label (`processor.py:1306-1323`).
- The text is word-split, and `max_len` caps the number of word tokens by cutting at the right (`processor.py:683-715`). For long inputs `classify_long` chunks the text, aggregates per-label **scores** by max/mean/first, then decodes once (`G2:gliner2/classification/long_text.py:1-41`).
- **Training augmentation** (`processor.py:305-314, 1211-1290`): shuffle labels and heads; drop up to 50% of labels (Beta-sampled); with p = 0.5 rename labels to `label 1..K` (descriptions/examples must carry the meaning); with p = 0.5 re-insert the gold; pick one of {few_shot, descriptions, both, none} per head.

#### Attention and isolation

- One bidirectional encoder pass over all heads plus the text. The only mask is padding (`G2:gliner2/processor.py:766`). **Heads attend to each other and labels attend across heads.** There is no isolation between questions, unlike Jev, where adding or removing a question must not change the others.
- Within a head, labels attend to each other bidirectionally. The parenthesised positions follow request order; training shuffling is the only defence against slot bias.

#### Readout and per-type handling

```
z_j = MLP(h[[L]_j])                              one logit per label, independently        (span/model.py:359-361; runtime.py:896-899)
p   = softmax(z / T)   if single-label            (T = `temperature` argument, default 1.0)   (runtime.py:897-909)
p   = sigmoid(z / T)   if multi_label             then threshold (cls_threshold, default 0.5), argmax fallback   (:925-937)
training loss: BCE(sigmoid(z), y) summed per head  (independent binary targets, even for single-label heads)   (span/model.py:359-365)
```

- **Train/serve mismatch**: logits are trained as independent sigmoids and decoded with a softmax for single-label heads. Nothing in training makes the softmax probabilities calibrated.
- `noul`: no native type. The card uses `["yes","no"]` heads, optionally with a `prompt` (`HF:GLiNER2.5-Decide/files/README.md:268-299, 401-419`).
- `score`: labels `"0"…"10"` as plain classes; "the loss does not know that 6 is closer to 7 than 0 is" (`:461-476, 510`). There is no expected value; the API returns the argmax string.
- `choice`: native; labels may carry descriptions (`labels: {name: desc}`) (`:438-459`).
- The returned confidence is the chosen label's probability, only on request (`include_confidence`). There is no `probabilities` map in the default output shape (`G2:gliner2/inference/runtime.py:49, 938-940`).
- Multi-head constraints: `gliner2/classification/constraints.py` implements a hard-constraint DSL with three-valued (Kleene) evaluation for joint decoding across heads (`G2:gliner2/classification/constraints.py:1-16`). Jev has no equivalent.

#### Calibration

None. There is no temperature, Platt or isotonic step in the checkpoints (the configs have no calibration fields), only the user-supplied `temperature` argument. The DI board measured **ECE 0.088, multi-class Brier 0.705, accuracy 0.434, mean confidence 0.506** over 35,088 scored items. It is under-confident in the low bins and over-confident in 0.7-0.8 (conf 0.746 vs acc 0.491) (`board: dix/index.json`, `engine gliner-decide`, `calibration`).

#### Limits

- Decide: DeBERTa-v3 512 absolute positions (relative attention lets longer inputs run, but they are out of distribution; **inferred**); `max_len` is null in the config. multi: 4,096 word tokens. 1B: 7,999 positions.
- Labels per head: training samples up to `max_num_labels 1000` (halved when descriptions/examples are used) (`processor.py:314, 1249-1255`). The practical limit is the encoder length.

### Training

- Base checkpoints `gliner2-large-v1` / `gliner2-xl-0111` / `gliner2.5-multi-v1`, then a Decide fine-tune. The local path in the config is `/home/urchadezaratiana/checkpoints/checkpoint-45500` (`HF:GLiNER2.5-Decide/files/config.json`), so roughly 45.5k steps for the 340M (**inferred** from the path).
- The Decide data mix is not published. The training row format is `{"input", "output": {"classifications": [{task, labels, true_label, multi_label, prompt, label_descriptions}]}}` (`HF:GLiNER2.5-Decide/files/README.md:480-512`). The boundary model's classification loss is BCE averaged over supervised labels, weighted `classification_loss_weight 1.0` (`G2:gliner2/models/boundary/model.py:1700-1742`).
- User fine-tuning: `ExtractorTrainer(model, TrainingConfig(...)).train("train.jsonl")`, or the hosted Fastino Agent / `api.fastino.ai` training jobs (`SKILL.md:10-24`).

### Evals

- **fast-decisions** (Fastino's own: 17 operational domains, 1–4 heads each, held-out test 300/domain; the public HF dataset is the 100/domain dev split, Apache-2.0, `notes/hf-raw/_datasets/fastino__fast-decisions.README.md`). Exact-match average: Decide 60.2%, Decide-1B 59.6%, JevK5 57.6%, multi 56.7%, "SemIf (Qwen3.5-4B)" 56.4%, GLiFormer large 49.0%, Laya Router 46.6% (`HF:GLiNER2.5-Decide/files/README.md:44-56`). The 1B does not beat the 340M. On Knowledgator's card, the same suite gives gliclass-v4 54-55 and d1 61.8 (`HF:gliclass-v4-9b/files/README.md:504`).
- **Decision Index 0.3.2 board**: GLiNER2.5-Decide **9.24** balanced skill (public 11.48), rank 155 of ~171. Per area, skill: knowledge 0.02, language 0.08, retrieval 0.25, tools 0.15, arts 0.10. Coverage 0.76. Latency median 23.3 ms, p95 229 ms (`board: dix/index.json`).
- No numbers for the multi or 1B on the board.

### Serving

- Local: `AutoExtractor.from_pretrained(...).classify_text(text, {head: labels | {labels, multi_label, cls_threshold, prompt}})`, plus batch and long variants (`G2:gliner2/inference/runtime.py:1863-1972`). CPU or GPU; `quantize()` and `compile()` helpers (`G2:gliner2/models/span/model.py:894-917`).
- No Jev-compatible endpoint in the repo (no `systemone` anywhere in `G2:`). The hosted API is OpenAI-style `POST /v1/chat/completions` with `X-API-Key` (`SKILL.md:12-24`).

### Licence

Apache-2.0 for code and all three checkpoints (`G2:LICENSE`; `HF:fastino__*/api-refresh.json`). Bases: DeBERTa-v3-large and mDeBERTa-v3-base (MIT), Ettin-1B (MIT, per jhu-clsp). The fast-decisions dev split is Apache-2.0.

---

## Notable design choices vs Jev and vs kev

```
                 kev (Qwen, block-causal / rows)          GLiClass v4 decoder-KV                     GLiNER2.5-Decide
sequence         <state> S <q> I <opt>A</opt>...<decide>  I S <<SEP>> A<<LABEL>> B<<LABEL>> <<SEP>>  ([P] t1: I ([L]A [L]B)) [SEP_STRUCT] (...) [SEP_TEXT] S
order            state first, question after               instruction first, state after            all questions first, state last
option marker    after option  (</opt>)                     after label (<<LABEL>>)                   before label ([L])
question token   <decide> after all options                 closing <<SEP>> after all labels          [P] (count head only; not used for scoring)
attention        causal; branches isolated                  causal backbone, then 2-layer             fully bidirectional, heads NOT isolated
                                                            bidirectional scorer over labels
score            (W_q h_decide)·(W_k h_opt)/√d              MLP([W_t h_sep ; W_l h_label]) ×≤3 GRU   MLP(h_[L]) (no question vector)
probability      softmax over options                       sigmoid-trained, softmax at serve         sigmoid-trained, softmax at serve
calibration      one T in head.pt                           per-pair (β, b) MLP, β ∈ (0, 3)           none
questions/pass   many (shared state prefix)                 one (Jev mapping); cache only for streams many, but coupled
```

1. **Label-set-in-input is the same idea as Jev's option list**, so the interface maps directly. The difference is what the label is scored against. kev scores each `</opt>` against a dedicated `<decide>` query vector. GLiClass pairs each `<<LABEL>>` with the closing `<<SEP>>` through an MLP, which is functionally the same role. GLiNER2 has **no question vector at all**: the question is only context that shaped the `[L]` states, and the logit is a function of the label state alone. That is a classifier head, not a pointer.
2. **GLiClass decoder-KV ≈ kev + Zeiger's set head.** The causal-backbone, marker-after-option, closing-token pattern matches kev. The extra bidirectional encoder over the label section matches Zeiger's 2-layer option transformer (`clones/zeiger.md`). The new pieces are the gated recurrent refinement (adaptive depth, halting at Δp < 0.01) and the learned per-pair calibrator. Neither appears in Jev's public description, and neither has an ablation on the cards.
3. **Isolation.**
   - Jev guarantees that sibling questions do not affect each other.
   - GLiClass gets this for free only because the Jev mapping runs **one pass per question**. That costs re-encoding the state per question, with no shared prefix, because the instruction precedes the state.
   - GLiNER2 packs heads into one bidirectional sequence, so adding a head can change the others' answers. This violates Jev's contract by construction (**inferred**; not measured here).
   - The "transient label tokens" trick is a real KV-reuse mechanism, but it caches the *context* for a stream of label queries. Jev's pattern (one state, many questions) would need the instruction moved into the transient section. **Inferred:** that is a small format change for Zeiger-style export, not an architectural one.
4. **Calibration philosophy.**
   - Jev returns probabilities "optimized against outcomes" with closed-form confidence.
   - kev ships one fitted T.
   - GLiClass learns a per-pair inverse temperature and bias from representations and logit statistics. This is the most expressive post-hoc calibrator in the clone set, but its own report shows aggregate ECE and NLL getting *worse* on held-out data while targeted sub-groups improve.
   - GLiNER2 ships none, and the DI board measures it as under-confident in the low bins and over-confident around 0.7–0.8.
   - Both families train independent sigmoids and serve a softmax for single-answer questions. Jev and kev train the softmax they serve.
5. **Per-type semantics are thin.**
   - Neither has a native `noul` or `score`. Both are label sets, and score levels are unordered classes.
   - GLiClass's card server reproduces Jev's response shape, but `confidence` is p_max (not Jev's rescaled formula) and `score` is an expected index.
   - GLiNER2 returns label strings, so a `score` is an argmax digit, not an expectation.
6. **Size versus skill.**
   - The encoder variants (GLiClass edge 0.4B, GLiNER2.5 0.3-1.2B) are very fast (11-23 ms median) and excellent on intent/routing benchmarks (edge BANKING77 90.7 > Jev 79.5).
   - They collapse on knowledge and reasoning (edge GSM8K 2.1, GLiNER2.5 DI knowledge skill 0.02).
   - Moving the same label-set head onto a 27B decoder closes most of the gap (DI public 61.74 vs Jev 1.13 57.96), still with the knowledge deficit (45.7 vs 53.9).
   - This matches kev's lesson that backbone capacity dominates transfer.
7. **Relevance for Zeiger.**
   - GLiClass's `classify_from_decoder_cache` (labels read a frozen context cache with explicit `position_ids` and are never written back) is the cleanest public implementation of "state once, options transient" on a hybrid DeltaNet backbone. It is worth reading next to kev's `shared_prefix.py`.
   - GLiNER2's anti-shortcut augmentation is cheap to copy for option-text robustness: synthetic `label i` names with descriptions carrying the meaning, Beta-sampled label dropping, and gold re-insertion.
