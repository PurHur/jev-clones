# Group: multimodal Jev clones

Compiled 2026-10-19. Nine families that put images, video or audio into the state. Sources: HF API (with
`?expand[]=safetensors`), raw `config.json`, `adapter_config.json`, READMEs and custom code, saved in
`notes/hf-raw/<org>__<model>/`. Shallow GitHub checkouts are in `sources/<org>__<repo>/`. They were read and
never executed. Code citations are `file:line`. HF repo files are cited by name and live in `notes/hf-raw/.../files/`.
Download counts are the HF 30-day counter on 2026-10-19, summed per family (from `notes/families.json`).
**DI** is Decision Index 0.3.2 "DI Full (balanced_skill)", the snapshot of 2026-10-16 with 173 entrants
(`notes/hf-raw/_spaces/dix_leaderboard.md`). For reference, Jev 1.14.0 scores 61.53 there (rank 13, ECE 0.080) and
Jev 1.13 scores 60.11. Anything marked *(inferred)* is our reading, not a statement by the project.

**Jev 1.14 for comparison** (`clones/jev-reference.md`). Images arrive as an `assets` map (uuid → base64 jpg/png)
behind the `TypeSafe-Danger-Zone-Images` header, and the state refers to them by key. The new `box` type returns
4 edges, each "the expected value of its edge's predicted position", which points to a distribution over binned
positions per edge. The docs say image assets "require a vision-language model". **No clone in this group
implements `box`.** The nearest analogues: VideoJev scores time intervals (temporal grounding, a 1-D box built
from per-segment sigmoids), and vev's Doom harness cuts the frame into 8 slices and asks a noul about each.

## Comparison

| Family | Backbone (decoder unless noted) | How media enters | Question isolation | Readout head | Types | Calibration | DI 0.3.2 |
|---|---|---|---|---|---|---|---|
| OneJev 1.5/1.6 (OmniJev) | Qwen3.5-0.8B/4B/9B, Qwen3.8-27B, full FT, vision frozen | `<image:N>`/`<video:N>` placeholders in state → Qwen vision blocks, M-RoPE; in the shared prefix | shared prefix KV, one branch per question (forked cache) | LM-head rows of 256 letter labels (A..Z, AA..) | noul/choice/score | temperature per (type, K) | 59.90 (27B), 53.29 (9B), 49.17 (4B), 29.01 (0.8B) |
| Valen (Valen-Team) | Qwen3.5-0.8B/2B/4B, LoRA merged + visual merger trained | Qwen chat-template image_url/video_url, 16 frames | **none** by default (`shared_state`: all questions in one causal sequence); `question` mode = one row per question | 2-layer MLP-Mixer over 4 pooled role vectors (context, instruction, candidate, decision) | noul/choice/score (score levels scored separately) | single temperature arg | not listed |
| vev (Xiaooolong / CountingSheep) | Qwen3.5-4B/9B, LoRA r16 | `{"image":{"url":data-uri}}` anywhere in the JSON state | one row per question (0.1.0); shared state prefix (0.1.1) | next-token probs on Yes/No, letters, digits, renormalised | noul/choice/score | none shipped; KL anchor to base + Brier 0.1 in training | not listed |
| imajev (mohit67890) | Qwen3.5-2B/4B/9B, LoRA r64 | 0–2 images (≤400k px) as multipart upload | one prompt per question and rotation *(inferred)* | separate 256×2560 linear (255 codes + `unknown`) | noul/choice/score + abstain | one temperature 1.305 | not listed |
| Jev-Omni (akhilaaa3) | Gemma 4 12B-it (unified text+vision+audio) | Gemma image/audio tokens before the text; video = 16 frames as 16 images; audio ≤30 s | one call per question, no sharing | last hidden → norm → Linear(3840, 256) | options list only (noul/score via options) | none shipped (ECE 0.040 claimed) | 42.35 (rank 88, ECE 0.161) |
| VideoJev (MCG-NJU) | Qwen3.5-4B, Qwen3.6-35B-A3B; last layer attention-only | whole video at 2 fps via Qwen video tokens; timestamps `<x.x seconds>` are candidates | questions packed causally after the video, no block mask *(inferred)*; video cached | cosine pointer: `<decide>` query vs `</opt>` keys, scale 10 | grounding (sigmoid per segment) + choice (softmax) | thresholds only | not listed |
| laya-vision (r33drichards / thaitea) | SmolVLM-256M cut to 20 layers (201M); ModernVBERT-250M variant (encoder) | SmolVLM image tokens at the start of the user turn, 512 px | one sequence per question (Laya layout) | Laya head: 2 transformer layers + scorer over option terminator `\n` (causal) or `[MASK]` (encoder) | noul/choice/score | Laya temperature per type | not listed |
| DuplexJev (adventists-ai) | Qwen3-ASR-0.6B encoder + connector + Qwen3-4B / Qwen3-VL-32B | ASR encoder states, 2-frame stack (6.25 tok/s), MLP connector, `<|audio|>` | shared prefix, suffixes under a **block-diagonal mask** | single next-token letter distribution | choice (closed set) | max p only | not listed |
| NeoHorse-Jev (TokenRhythm) | NeoHorse-1-4B (Qwen3.5-4B text) + untouched Qwen3.5-4B vision tower | one image + one question per request | kev runtime (block-causal branches *(inferred)*); `option_isolation: false` | kev pointer head (head_dim 256) | noul/choice/score | temperature 1.0 | 37.97 (rank 105, ECE 0.104) |

**Pattern.** All nine take a pretrained VLM or speech LLM and keep its native media path. None adds a
TypeSafe-style `assets` map; media sit inline in the state. Two kinds of head sit on top:
- **LM-head label tokens**: OneJev, vev, DuplexJev, and imajev (as a separate linear initialised from code tokens *(inferred)*).
- **Learned pointer or mixer heads**: Valen, VideoJev, NeoHorse, laya-vision, Jev-Omni.

The decision head does **not** change for vision. The same text-trained head reads hidden states that have
attended to image tokens. NeoHorse makes this explicit by grafting an untrained vision tower under a text-only
head. Only VideoJev adds a new output type (temporal intervals). Only DuplexJev changes the input encoder
(an ASR encoder plus a trained connector, instead of the base model's own modality path).

---

## OneJev / OneJev 1.5 / 1.6 / PlayJev (OmniJev)

**Identity**
- Code: https://github.com/OmniJev/OneJev (Apache-2.0, ★174, HEAD 1e42405 2026-10-19). Package name is `qev`.
- Weights: https://huggingface.co/OmniJev, Apache-2.0. The family has about 8.5k downloads and 30 likes.
  - OneJev-0.8B/4B: 2026-09-27
  - OneJev1.5-0.8B/4B/9B/27B: 2026-10-12/13
  - OneJev1.6-4B: 2026-10-16, 4.54B params
  - PlayJev-0.8B: 2026-09-19
- Data: https://huggingface.co/datasets/OmniJev/OneJev-Data, 94,707 of the 99,193 training questions, 17.7 GB.
- Do not confuse with **tinnel123/OmniJev** (https://github.com/tinnel123666888/OmniJev, ★165), a separate same-name project. It is a Qwen3.5-0.8B/2B/4B LoRA r32 with a "prefix-branch" path, an ordinal score head and temperatures noul 1.20 / choice 1.12 / score 1.19 (`tinnel123__OmniJev-4B/files/head_meta.json`).

**Backbone.** Full fine-tune of `Qwen3.5ForConditionalGeneration`. The vision tower stays frozen.
- 4B: 32 layers (8 full-attention + 24 linear-attention), hidden 2560, 16 heads / 4 KV heads, max positions 262,144, vocab 248,320.
- Vision: 24 blocks, d 1024, patch 16, spatial merge 2 (`OmniJev__OneJev1.5-4B/files/config.json`).
- Parameter counts: 1.5-4B 5.17B and 1.6-4B 4.54B. The extra 0.63B equals one 248,320×2560 matrix, which suggests 1.5 unties the LM head *(inferred)*.

**Input layout.** Chat prompt with a fixed system line: "Choose exactly one of the listed options. Respond with only its uppercase letter" (`qev/prompt.py:16-19`).
- The state is JSON. `<image:N>` / `<video:N>` placeholders in it are replaced by Qwen vision blocks (`qev/mm_engine.py:1-12`).
- Each question is a suffix "Question: … Options: A. … Answer with one letter".
- Video arrives as pre-extracted frames plus `fps` (`qev/media.py:9-25`).

**Isolation.** The state and media form the shared prefix. "The vision encoder runs once per request, and every question is a text-only branch off the forked cache." Suffix positions continue from the M-RoPE delta, so branches equal full single-question runs (`qev/mm_engine.py:1-9`, `:179-190`). Isolation therefore matches Jev's behaviour.

**Readout.** The engine slices the LM-head rows of the slot tokens: A..Z, then two-letter labels up to 256, with single-token labels only (`qev/prompt.py:21-26`, `qev/mm_engine.py:441-462`). No new head is added. A softmax runs over the K live slots.

**Question types.** noul, choice and score. The answer and confidence formulas copy TypeSafe's (`qev/answers.py`). No `box`.

**Calibration.** `Calibration` holds temperatures keyed `kind` or `kind:K` (`qev/calibrate.py:14-21`). They are fitted by cross-entropy against soft targets (`:43-70`).

**Training.**
- One epoch, lr 5e-6, 16k-token limit, CE + Brier (weight 1.0), option padding/augmentation (`train/configs/onejev_4b_full.yaml`).
- About 320k questions from agent runs, videos, images and public decision-dataset train splits (1.5 card).
- PlayJev: Qwen3.5-0.8B-Base trained by behaviour cloning plus 3 DAgger rounds on 10 browser games. It scores 0.57 of the teacher on average (`OmniJev__PlayJev-0.8B/files/README.md`).

**Evals.**
- DI 0.3.2: 27B 59.90 (rank 20, ECE 0.024, just under Jev 1.13's 60.11), 9B 53.29, 4B 49.17, 0.8B 29.01.
- Self-reported on the DI *public part*: 1.6-4B 62.01, 1.6-27B 69.49 (GitHub README). This is a different split from the full DI.
- Speed on an H200 with a 1280×720 screenshot: 4B 64 ms for 1 question, 104 ms for 10.

**Novel.** The cleanest multimodal port of the Jev contract:
- vision is only a prefix, and question branches stay text-only off a cached multimodal KV;
- the readout is the LM head itself, with no added parameters;
- it serves through llama.cpp GGUF as well.

## Valen (Valen-Team)

**Identity**
- Code: https://github.com/Valen-Team/Valen (Apache-2.0, ★873, HEAD 7b43242 2026-10-16).
- Weights: Valen-Preview-0923 (2026-09-23, Qwen3.5-2B), Valen-0.8B/2B/4B (2026-10-05). Apache-2.0. About 2.4k downloads.
- Authors (per `notes/families.json`): Liuziyu77, also the authors of JevalKit.
- Eval set: https://huggingface.co/datasets/Valen-Team/VisualDecisionBench.

**Backbone.** `ValenQwenForDecisionMaking`, a subclass of Qwen3.5 (`configuration_valen.py`). The LoRA is merged, the visual merger is trained, and the Mixer is kept in FP32.
- 4B: 32 layers (8 full / 24 linear), hidden 2560, 4.54B params.
- 0.8B: 24 layers, hidden 1024.
- `valen_max_length` 32,768.

**Input layout.** The state is Qwen chat messages carrying `image_url` / `video_url`.
- Media settings (`config.json` `valen_media_kwargs`): images use the processor's `longest_edge` 262,144 / `shortest_edge` 65,536; video takes 16 sampled frames.
- Each question is rendered as "Task: <type>\nQuestion: …\nCandidates:\n<key>: <desc>".
- Control tokens are rejected inside user text (`compiler.py:66-68,151`).

**Isolation.** The default `qwen_execution="shared_state"` puts **all questions in one causal sequence**. The code says so itself: "This is ordinary causal attention, not isolated question/Score branches" (`compiler.py:180-183`). Later questions therefore see earlier ones, which breaks Jev's add/remove invariance. `execution="question"` gives one row per question instead.

**Readout.** Interchangeable heads (`heads.py`): bilinear, MLP, role-MLP and the shipped **MixerHead** (`heads.py:108-131`).
- Mean-pooled context span, instruction span and candidate span, plus the decision-token state, are projected to width 256 with role embeddings.
- They pass through 2 MLP-Mixer blocks that mix the 4 roles but "never candidate rows". One logit per candidate follows.
- Candidates are scored independently of each other. They interact only through causal attention in the backbone.

**Question types.**
- noul: the fixed candidates `true` / `false` (`schema.py:12-13`).
- choice: 1–255 options.
- score: 2–255 levels, each level in its own branch, "Score never sees its level ID, nor any other level description" (`compiler.py:132,158`).
- Response formulas follow TypeSafe (`responses.py`).

**Calibration.** A single temperature argument to `predict` (`modeling_valen.py:73-81`). The card ships none.

**Training.** Two-stage SFT: head warm-up, then joint LoRA + Mixer + visual merger, on "millions of samples". It started from Valen-Training-General-100k. The GitHub badge says RLCD is experimental.

**Evals** (GitHub README, 2026-10-06 figures).
- VisualDecisionBench-Image: 0.8B 75.9%, 2B 80.4%, 4B 83.1%.
- VisualDecisionBench-Video: 79.0%, 82.9%, 87.4%.
- Public JevBench: 77.9%, 83.6%, 83.1%, against Intern-Decision-4B at 86.6%.
- Not on the DI.

**Novel.** The only clone with a pluggable family of heads and an ablation-friendly design (role-pooled Mixer against bilinear pointer). It also trains the visual merger. On the other hand, it gives up question isolation by default.

## vev (Xiaooolong / CountingSheep)

**Identity**
- Code: https://github.com/Xiaooolong/vev (Apache-2.0, ★18, HEAD 2d8c403 2026-10-06).
- Weights: https://huggingface.co/CountingSheep/vev-4b and `vev-9b` (merged), plus `-lora` adapters. 2026-09-30, **CC BY-NC 4.0**, because 6 of the 42 data sources are non-commercial. About 400 downloads.

**Backbone.** Qwen3.5-4B (4.54B params, 32 layers, hidden 2560) and Qwen3.5-9B (9.41B). LoRA r16, α32, on all attention, linear-attention and MLP projections of the language model only (`CountingSheep__vev-4b-lora/files/adapter/adapter_config.json`). The vision tower is untouched.

**Input layout.**
- The state is a string or any JSON. An image is `{"image":{"url":"data:image/png;base64,…"}}` anywhere inside it (README).
- `max_pixels` 1,048,576 and `max_row_tokens` 4,096 in training (`vev.json`). The serving limit is 32k tokens.
- Questions use the chat template with answer-format lines.

**Isolation.** 0.1.0 ran one forward pass per question, each row carrying the full state prefix (`train/data.py:2`). 0.1.1 shares the state prefix when that saves recomputation (`vev/model.py:251,304`) and gives the same probabilities. It has a conformance test suite that includes `conformance/test_isolation.py`.

**Readout.** "Next-token probabilities for the answer tokens (Yes/No, option letters, level digits), renormalised over the allowed tokens" (README:230, `vev/readout.py:40-80`). There is no extra head (`head_lr 0.0`).

**Question types.** noul, choice and score. No `box`. The Doom harness cuts the frame into 8 vertical slices with one noul per slice, a poor man's localisation.

**Calibration.**
- Training: Brier 0.1, score weight 0.05, and a **KL anchor** (weight 2.0, `anchor_kl` 0.3) to the adapter-disabled base on an unlabelled anchor set, "to limit drift on question types that are not in the training data" (`vev.json`, `data/anchor/build.py:1-8`).
- No temperature is shipped. The README states ECE ranges and calls the model biased toward "no".

**Training data.** Build `v1-research`: 762,820 records from 42 public sources. 431k are image records, 331k text; 621k are English and 142k Chinese. A 100k subset was used, for 2,500 steps (TRAINING.md:41,66-67).

**Evals.**
- JevBench public 76.6%, kev transfer-v4 77.6%, MMBench-EN 88.1%, POPE 88.9%, MMStar 62.7%.
- Own judgment sets: LlavaGuard policy 0.742, EvalMuse t2i 0.715 (card model-index, README).
- Speed: one 1 MP image takes 78 ms. Not on the DI.

**Novel.** The KL anchor to the zero-shot base is a cheap way to keep a LoRA clone general. The project also has the most explicit licence audit of its training mixture.

## imajev (mohit67890)

**Identity**
- Code: https://github.com/mohit67890/imajev (Apache-2.0, ★398, HEAD 318e8da 2026-10-11).
- Weights: imajev-2b/9b (2026-09-23) and 4b (2026-09-24, revised 2026-10-10). PEFT plus MLX. Apache-2.0. About 4.3k downloads, 118 likes.
- Benchmark: https://huggingface.co/datasets/mohit67890/imajev-bench.

**Backbone.** Qwen3.5-4B, LoRA **r64**, α128, on the language-model projections (`adapter_config.json`). There are 121.9M trainable LoRA params plus a 655k readout (README).

**Input layout.**
- The JSON request carries 0–2 images (resized to at most 400k pixels) as a multipart upload.
- Limits: state ≤32 KB, 1–8 questions, ≤4,096 tokens (refused, not truncated). English only.
- Two-image "reference against target" requests are a trained use case.

**Isolation.** Not documented. The card's numbers use "4 option rotations", i.e. 4 prompts per question, so every question is likely its own prompt *(inferred)*.

**Readout.** "One bias-free linear layer, 256 × 2560, float32 (255 option codes + `unknown`)" (README). The code tokens are A, B, … up to two-letter codes, 256 entries (`decision_readout.json`). So it is a separate readout over code-token positions, likely initialised from the LM head rows *(inferred)*.
- `unknown_probability` and `abstained` come back on every answer. 17–22% of the training targets are `unknown`.

**Question types.** noul, choice (2–255), score (2–10), plus abstention. No `box`.

**Calibration.** One temperature, 1.305, fitted by NLL on 150 template items (`calibration.json`). There is a separate fit for 4-rotation serving (`calibration-rot4.json`) and a per-modality variant (`calibration-modality.json`).

**Training.** About 1M decisions in total, in four stages:
1. a 504k human-labelled set, 296k labelled by the 9B, and 66k photo-against-record decisions;
2. teacher-agreement filtering;
3. hard-mined decisions labelled by Qwen3.6-35B-A3B;
4. a soft-target stage with a 0.3 rationale loss.

The cost was about $676 for the whole project (README).

**Evals.**
- JevBench public hard 72.1% (4 rotations, ECE 0.082). Third-party boards: JevBench v1.4.2.2 #1 at 67.4 against Jev 1.13 at 63.3; Image JevBench #1 at 76.39, ahead of Jev-Omni at 73.10 and NeoHorse at 71.94.
- DecisionBench 79.7%.
- Not on the DI.

**Novel.** The **trained `unknown` slot** for abstention, which is rare in the field, and checking a photo against a record.

## Jev-Omni (akhilaaa3)

**Identity.** https://huggingface.co/akhilaaa3/Jev-Omni, created 2026-09-20, updated 2026-10-13. Apache-2.0. 5,379 downloads, 420 likes. No code repo; the loader `jev_omni.py` ships in the HF repo.

**Backbone.** `Gemma4UnifiedForConditionalGeneration`, base `google/gemma-4-12B-it`, 11.96B params, shipped in BF16 (`config.json`).
- Text: 48 layers, hidden 3840, 16 heads / 8 KV heads, sliding window 1024, max positions 262,144, vocab 262,144.
- A built-in vision tower (patch 16) and audio tower (d 640).
- LoRA r512 / α512, merged on top of an earlier merged r128 run. 2.10B trainable params (`decision_config.json`).

**Input layout** (`jev_omni.py:30-36,73-89`).
- The media content goes first: an image; or a video as 16 uniformly sampled frames passed as 16 images; or audio converted by ffmpeg to 16 kHz mono and capped at 30 s.
- Then the text "{state} --- QUESTION: … OPTIONS: 1. … Reply with only the number".
- One modality per call.

**Isolation.** Not applicable: one question per forward pass, and the state is re-sent for every question. The card prices it that way.

**Readout.** `_Head256` (`jev_omni.py:111-121`): the last-token hidden state of the text decoder, standardised with stored μ/σ buffers, goes through `Linear(3840, 256)`. Slots beyond K are masked to −1e30, then softmax. The slot head reads the position, not the option text: there is no pointer to option tokens.

**Question types.** `predict(state, question, options)` with 2–256 options. Noul and score must be phrased as option lists. "Best supported at ≤20 options."

**Calibration.** Nothing post-hoc ships. The card claims DecisionBench-Medium ECE 0.040.

**Training.** Card: "30,000-question fine-tuning run". Config: 24k items, 750 steps, lr 1e-5, 4 GPUs. The data is not released.

**Evals.**
- DecisionBench-Medium 87.6%, JevBench (matched subset) 86.2%, MMAU 63.1%, MVBench 53.1%.
- DI 0.3.2: **42.35** (rank 88, ECE 0.161).
- Speed on an H200: image 26 ms, 13 s of audio 31 ms, 16-frame video 504 ms.

**Novel.** The only clone in the set with **audio, image and video** on one backbone. Its head is the most basic: a fixed 256-slot classifier on the last token.

## VideoJev (MCG-NJU)

**Identity**
- Code: https://github.com/MCG-NJU/VideoJev (Apache-2.0, ★176, HEAD 7f8b38a 2026-10-18).
- Weights: https://huggingface.co/ZhuYuhan/VideoJev-4B (4.47B) and `VideoJev-35B-A3B` (33.8B). Both created 2026-10-10, Apache-2.0, about 2.9k downloads.
- Paper: arXiv 2610.18615 (2026-10-13, Zhu, …, Limin Wang). Abstract saved in `notes/hf-raw/_papers/`.

**Backbone.** Qwen3.5-4B (32 layers, hidden 2560) and Qwen3.6-35B-A3B (40 layers, hidden 2048, 256 experts / 8 active, FP32 routers).
- **The last decoder layer is cut to attention-only**: its MLP and post-attention norm are deleted (`modeling_videojev.py:19-26,79-82`).
- The vision encoder is frozen.

**Input layout.**
- The video is sampled at 2 fps (up to 2048 frames) and goes through the Qwen video path with 3-D positions.
- Every question suffix reads "Task: temporal grounding | single-choice video QA / Question / Options:". Each candidate is followed by an `</opt>` token (`option_end_token_id` 248078), and the suffix ends in `<decide>` (248077) (`processing_videojev.py:16-29`, `config.json`).
- **For grounding, the candidates are the native timestamp tokens `<x.x seconds>`**, one per sampled segment.

**Isolation.**
- The video is encoded once ("Cached inference"), and as many questions as fit the context are packed into one sequence ("Joint inference", `processing_videojev.py:63-84`).
- A guard checks that "candidate readouts [do not] cross query boundaries". No block mask is built (a single `cu_seq_lens` span), so later questions can attend to earlier question text *(inferred)*.

**Readout.** `CosineDecisionHead` (`modeling_videojev.py:33-44`): L2-normalised q(`<decide>`) · k(`</opt>`) × 10 + bias, with head_dim 512, in FP32.
- Grounding: an **independent sigmoid** per segment, thresholded at 0.25 and smoothed into an interval.
- Choice: a softmax (`:164`).

**Question types.** `grounding` (temporal interval) and `choice`. No noul or score. Temporal grounding is the 1-D counterpart of Jev 1.14's `box`.

**Calibration.** None beyond the decode threshold (`config.json` `decoder.threshold` 0.25).

**Training.** Two stages, one epoch each, global batch 64 packs:
1. head warm-up (BCE / CE);
2. joint backbone + head with BCE + 2 × duration-weighted SoftIoU (README:173-178).

Data: TimeLens-100K, TimeLens2-93K and QA sets. Trained with XTuner.

**Evals** (paper abstract). On three grounding benchmarks, mean mIoU improves over the backbones by +6.7 (4B) and +10.8 (35B), with a 2.8–3.1× speed-up. On four video-QA sets it is comparable to the backbones' Thinking mode at more than 100× the speed. Not on the DI.

**Novel.** A **new decision type** (interval grounding over timestamp-token candidates). It drops the last MLP, since only attention is needed for a pointer readout. The paper is peer-review style and makes the cached/joint inference argument explicitly.

## laya-vision (r33drichards / thaitea)

**Identity**
- Code: https://github.com/r33drichards/laya-vision, a fork of Laya (Apache-2.0, ★93, HEAD 9e1e241 2026-09-30).
- Weights: https://huggingface.co/thaitea/laya-vision (now the 201M model), `-201m`, `-smolvlm-256m-score` (236.6M), `-modernvbert-250m` (267M) and others. 2026-09-19..25. **CC BY-NC-SA 4.0**. 0 downloads and 49 likes as counted.

**Backbone.**
- Shipped: SmolVLM-256M-Instruct (Llama text: hidden 576, 9 heads / 3 KV, vocab 49,280) **cut to 20 of its 30 layers**, found by an autoresearch loop (`backbone/config.json`, README).
- Variant: ModernVBERT (bidirectional, 22 layers, hidden 768).
- Laya's 2-layer decision head and scorer are reused (`vlm_agent_config.json`: `head_layers` 2, `n_act` 2).

**Input layout.** `<|im_start|>User:<image>` then the state text, the question and the options. Each option line ends in a fixed terminator `\n`, whose hidden state is the readout (`laya/vlm.py:6-30,212,257-300`). The image is 512 px, with optional splitting (`image_split_edge`). `max_len` 1024 / `head_max_len` 256.

**Isolation.** One sequence per question, as in Laya. Option attention is `"block"`: a 4-D mask lets the option block attend to itself in both directions while the state and question stay causal (`laya/vlm.py:485`, `modal_app.py:442-446`).

**Readout.**
- `readout="terminator"` (causal SmolVLM) or `"mask"` (`[MASK]` markers on ModernVBERT, `laya/vlm.py:76,133`).
- Laya's head over the markers, then a softmax.
- The code documents the causal order bias and mitigates it with random `option_order` in training and `n_permutations` at inference (`laya/vlm.py:27-30`).

**Question types.** noul, choice, score. The card says the score head is untrained on some checkpoints (`notes/families.json`).

**Calibration.** Laya temperatures per type: `[3.86, 2.10, 3.05]` (choice, score, noul) (`vlm_agent_config.json`). The training option `w_ce_schedule="anneal"` ends training on the proper scoring rule alone.

**Training.** The Cauldron, VLFeedback, A-OKVQA, ScienceQA, VQAv2 and game frames. About 2 h on Modal.

**Evals.**
- A-OKVQA 59.8%, ScienceQA-img 82.4%, VQAv2 yes/no 71.4%.
- 34-set suite: 69.1% overall, ECE 0.041, 63 ms on an L4 (`training_metrics.json`).
- Not on the DI.

**Novel.** The smallest multimodal clone, at about 200M, CPU-plausible *(inferred)*. It runs a **controlled causal-against-bidirectional backbone comparison** on identical data and puts a block-bidirectional option mask on a causal VLM.

## DuplexJev (adventists-ai)

**Identity**
- Code: https://github.com/adventists-ai/duplexjev (Apache-2.0, ★28, HEAD 715889c 2026-10-16).
- Weights: DuplexJev-4B / -4B-Turn / -4B-Para and -32B / -32B-Turn / -32B-Para (2026-09-28..10-14), plus -32B-Vision (preview, 2026-10-10). **CC BY-NC 4.0**. About 650 downloads.
- Paper: arXiv 2610.02638, "Batched Speech Decisions Without Decoding".

**Backbone.** An Ultravox-style `DuplexJevForConditionalGeneration` (`config.json`):
- **Qwen3-ASR-0.6B encoder** (18 layers, d 896), frozen;
- a SwiGLU projector (`stack_factor` 2);
- Qwen3-4B (36 layers, hidden 2560) or Qwen3-VL-32B (64 layers, hidden 5120).

The paper's LLM is frozen. The Turn and Para models add a merged rank-16 LoRA. Total size: 4.22B / 32.96B.

**Input layout.** Audio is turned into encoder hidden states, then 2-frame stacking (6.25 tokens/s) and the MLP connector, and enters as `<|audio|>` in the user turn. The connector reads the last layer or uses cross-attention (the paper's two variants). The 32B-Vision preview adds an image.

**Isolation.** "Context and audio are encoded once and all question suffixes are packed into one row under a **block-diagonal mask**; answers match one-by-one runs up to bf16 noise" (GitHub README:132-133). This is the Jev/kev shape. Under vLLM, the prefix cache plays that role instead.

**Readout.** "Each question lists its options under permuted letters; the answer is the next-token softmax restricted to those letters" (README). There is no added head. Training applies CE to the single answer token, plus distillation for content.

**Question types.** Closed-set choice: turn completion, filler choice, gender, emotion, stress, non-verbal sounds, spoken QA. No typed noul/score API: yes/no questions are 2-option choices *(inferred)*.

**Calibration.** max p is used as the confidence. No temperature.

**Training.** ASR/speech corpora: paralinguistic dialogues, Easy-Turn, NonverbalTTS, EARS, MSPB, AISHELL-1, plus TTS-generated (CosyVoice2) spoken items. v1.1 adds conversation replay to stop the model drifting toward terse replies (4B-Para card).

**Evals.** Speech decisions only:
- Spoken QA 90% against 91% reading the transcript; gender and emotion about 90%. Easy-Turn 91.0 (in-domain).
- Throughput: about 170 decisions/s on one H200; 80 decisions about 8 utterances in about 0.1 s on 8 GPUs (paper).
- Not on the DI.

**Novel.** The only **speech-native** clone. It shows that a single-token readout over a frozen LLM can "hear beyond the transcript" (prosody, speaker, emotion) once supervised on the answer token.

## NeoHorse-Jev-4B (TokenRhythm)

**Identity**
- Code: https://github.com/TokenRhythm/NeoHorse (Apache-2.0, ★1,738, HEAD c494069 2026-09-26). The `jev/` directory holds the runtime, which comes from kev (`model_manifest.json` `runtime_origin`).
- Weights: https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B, created 2026-09-23, Apache-2.0. About 7.1k downloads in the family.

**Backbone.** NeoHorse-1-4B, TokenRhythm's own Qwen3.5-4B-derived base (32 layers, hidden 2560, 262k positions), with a LoRA merged in FP32 (`model_manifest.json`).
- **The vision tower is copied unchanged from Qwen/Qwen3.5-4B**: 297 tensors, 333.5M params, `vision_finetuning: false`, `text_tensors_unchanged: true` (`vision/base_vision_provenance.json`, `config.json`).

**Input layout.** Text requests are kev-style, with option markers and multiple questions. Image requests are limited to "one static image and one question" (`vision/README.md`), with the image as an inline data URL.

**Isolation.** Text uses kev's block-causal branches *(inferred from the kev runtime)*. `option_isolation: false`, so options see each other causally.

**Readout.** kev pointer head, head_dim 256, FP32, stored in `pointer_head.safetensors`. It reads the decision token against the option-marker hidden states.

**Question types.** noul, choice, score. No `box`.

**Calibration.** `temperature: 1.0`, i.e. none applied.

**Training.** Not detailed on the card. It is a kev-recipe fine-tune of NeoHorse-1-4B on text decision data *(inferred)*. The vision path received **zero** decision training.

**Evals.**
- Six-group text aggregate 77.70: JevBench 75.73, Kev 81.92, Nimble 87.23, VitaminC 77.13, MASSIVE 85.43.
- Image-NLI 60.65% over 8,000 examples.
- DI 0.3.2: **37.97** (rank 105, ECE 0.104), below Kev 4B r10 at 39.5.
- Production runtimes for vLLM and SGLang.

**Novel.** It is the clearest test of whether a **text-trained pointer head transfers to vision for free**. The answer looks like "partly": Image-NLI is 60.7%, and Image JevBench is 71.94, against imajev's 76.39.
