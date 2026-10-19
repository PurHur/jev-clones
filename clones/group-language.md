# Group: language-specific clones (TH, TR/DE/EN, JA, RU, PL, TR, DE)

Compiled 2026-10-19. Sources: HF API (`?expand[]=safetensors`, downloads = HF 30-day counter on 2026-10-19), raw
`README.md` / `config.json` / `adapter_config.json` / `tokenizer_config.json` / custom code saved under
`notes/hf-raw/<org>__<model>/`; GitHub repos shallow-cloned read-only into `sources/{jevalt,sokudan,frida-decisions,basal}`
(never executed). Dataset cards under `notes/hf-raw/_datasets/`. "(inferred)" marks our inference; "[ours]" marks our own
measurement. **Decision Index 0.3.2** (snapshot `notes/hf-raw/_spaces/dix_leaderboard.md`, 2026-10-16): **none of the models
in this group is on the board.** Reference rows: Jev 1.14.0 61.53 (#13), Jev 1.13 60.11 (#18), Kev 4B 39.5 (#98),
Intern-Decision-4B 38.21 (#104, the base of JevAlt), Laya 4.43 (#164). `notes/decision_index_0.3.2_entries.tsv` has no
entry for any model below either.

## Comparison

| Family | Lang | Base (backbone) | Enc/dec | Params | Readout | Question isolation | Options max | Ctx (trained / served) | Calibration | Data: native vs translated | Licence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **OpenThai-SystemOne** (iApp) | th, en | Qwen3.5-0.8B-Base text tower + 4.47B-token Thai CPT | causal decoder | 0.75B | 256-way slot head at `<\|ts_answer\|>` (slot 255 = abstain) | **none**: all questions in one causal sequence | 255 | "64k per request" (card) | 1 learned T per type (+ Brier stage) | mostly converted public TH/EN sets + synthetic TH (Qwen3.6-35B-A3B); native Thai | Apache-2.0 |
| **JevAlt** Deem / Karar / **Wähler** (Mert Kaya) | en / tr / **de** | Intern-Decision-4B (Qwen3.5-4B), LoRA r32 | causal decoder (VLM, text only used) | 4.54B | LM-head letter logits at `<decision>` markers | none (Intern-Decision JSON skeleton, causal) | 62 | 8k | T per type×language + conformal sets | **native** teacher-written rows per language + MASSIVE/PAWS-X | Apache-2.0 |
| **sokudan-ja-310m** (GeneLab) | ja | sbintuitions/modernbert-ja-310m | bidirectional encoder | 0.31B | marker scorer (768→1) + cumulative-link ordinal head | per question (one sequence each) | ~6 trained | state ≤1024 tok served, trained avg 134 | bool-only T=2.07 | **native synthetic** (qwen3:30b-a3b), no translation | Apache-2.0 |
| **FRIDA-Decisions** (SberAI) | ru, en | ai-forever/FRIDA (FRED-T5 encoder) | bidirectional encoder | 0.82B | linear head on mean of each option's tokens | **full**: block mask, state cached, options never see each other | unlimited (rows of 16) | state 512 | none (listwise softmax) | 71% RU / 29% EN, mostly human-labelled public data | MIT |
| **basal-1.x** (Remek / si5.pl) | pl, en | speakleash/Bielik-1.5B / 4.5B v3 Instruct | causal decoder (Llama) | 1.6B / 4.76B | letter logits after `{"answer": "` prefill | **full** in 1.5 (SOAM tree mask); 1.0 = one prompt per question | 10 | 8k / 32k | T per type, certified 1%/5% thresholds | **native Polish** generated + code-computed labels, ~21% EN public | Apache-2.0 |
| **basal-omni-1.0** | pl, en (+img/audio zero-shot) | Gemma 4 E4B / 12B / 31B | causal decoder | 7.9B / 12.0B / 31.3B | 255 "code" logits (prompt format 2) | prefix-shared prompts (FORMAT2), branch isolation (inferred same as SOAM) | 255 | 131k / 262k | T per type & bucket, 1- and 2-order files | soft targets from basal-1.5 (640k–815k decisions) | Apache-2.0 |
| **Turnike-JEV** (Berkay Hasret) | tr, en | Qwen3.5-0.8B / 4B / 9B (instruct), LoRA r16 | causal decoder | adapter 10.8M–43.3M | output-head symbol rows at token before each `"?"` | none (≤16 questions in one causal prompt) | 62 (+2-stage elimination) | 1536 / 4096 | **none** (T=1) | public TR/EN + LLM-labelled + synthetic; MT only in a supplementary eval | Apache-2.0 |
| **decisions-de** models (johannhartmann) | **de** | Qwen3.5-2B / Gemma 4 E4B, QLoRA r64 | causal decoder + Clef joint-schema head | 2.27B / 8.0B | Clef head, logit per option | none (Clef: all fields in one sequence, joint head) | – | 4096 | global T folded (1.56) + per-type T | **machine-translated** (Claude Sonnet/Haiku 5.5) from bekko / typed-decisions | Apache-2.0 (data per-source) |

**Tokenizer cost, same support ticket in 7 languages [ours]** (`notes/hf-raw/_tokenizers/`, `tokenizers` 0.23.3 on each
repo's `tokenizer.json`; one ~45-word text, so treat as indicative):

| tokenizer (vocab) | en | **de** | pl | tr | ru | ja | th |
|---|---|---|---|---|---|---|---|
| Qwen3.5 (248k): JevAlt, Turnike, decisions-de | 51 | **57** | 81 | 72 | 59 | 59 | 58 |
| OpenThai's shipped tokenizer (248k + 19 added) | 51 | 57 | 81 | 72 | 59 | 59 | **121** |
| Gemma 4 (262k): basal-omni, decisions-de E4B | 51 | 64 | 80 | 68 | 65 | 60 | 75 |
| Bielik v3 (32k): basal-1.x | 87 | 107 | **53** | 139 | 442 | 340 | 661 |
| FRED-T5 / FRIDA (94k) | 48 | 96 | 138 | 113 | **51** | 155 | 441 |
| ModernBERT-ja (102k) | 58 | 86 | 115 | 110 | 111 | **46** | 382 |

Readings: Qwen3.5 is the cheapest German tokenizer here (1.12× English); Gemma 4 1.25×; the language-native vocabularies
win only on their own language (Bielik 0.65× English on Polish, FRIDA/ModernBERT-ja best on RU/JA) and are 1.7–2× worse on
German. OpenThai's `tokenizer.json` uses the Qwen2-era pre-tokenizer regex without `\p{M}`, so Thai combining vowel/tone
marks are split from their letters; the same Thai text costs 2.1× the tokens of stock Qwen3.5 (diffed: vocab and merges
identical, only `pre_tokenizer` differs). Whether training used this tokenizer too is not stated (inferred: yes, it is the
shipped one, so the model is consistent but Thai context is twice as expensive as necessary).

---

## OpenThai-SystemOne (iApp Technology, Thailand)

- **Identity.** [iapp/OpenThai-SystemOne](https://huggingface.co/iapp/OpenThai-SystemOne) (created 2026-09-20, modified
  10-01, 11,987 dl, 37 likes) + [iapp/OpenThai-SystemOne-Ollama](https://huggingface.co/iapp/OpenThai-SystemOne-Ollama)
  (GGUF, 604 dl). Code [iapp-technology/openthai-systemone](https://github.com/iapp-technology/openthai-systemone)
  (68★, Apache-2.0). Sponsor Siam AI (H100s). Licence Apache-2.0.
- **Backbone.** `OpenThaiSystemOneForDecision` (remote code `modeling.py`), text tower of Qwen3.5-0.8B-Base: 24 layers
  (18 Gated-DeltaNet `linear_attention` + 6 `full_attention`, `full_attention_interval` 4), d 1024, FFN 3584, 8 heads / 2 KV,
  head_dim 256, partial RoPE 0.25, θ 1e7, `max_position_embeddings` 262,144, vocab 248,339. Vision tower removed, LM head
  removed. **752,674,883 params** BF16 (API). Continued-pretrained on 4.47B tokens of Thai (web, Wikipedia, parallel
  TH↔EN, accessibility trees/JSON) before SFT.
- **Input layout** (`formatting.py:1-19`), one causal sequence:
  `<|ts_state|> state ⟶ <|ts_q|><|ts_choice|> instr ⟶ <|ts_opt_0|> name: desc … ⟶ <|ts_answer|>` then the next question.
  257 new control tokens, initialised at the mean of digit embeddings + noise (`modeling.py` `_init_new_token_embeddings`).
  State is tail-kept on truncation (`formatting.py:236-240`).
- **Isolation.** None. Questions are appended one after another under plain causal attention, so question *k* reads
  questions 1..k−1 (`formatting.py:241-247`); training uses `shuffle_questions` (inferred: to wash out order effects).
- **Readout.** `slot_head = Linear(1024, 256)` on the hidden state at each `<|ts_answer|>`; slots ≥ k masked to −inf, slot
  255 (**abstain**) always valid; softmax. noul = 2 slots `no`/`yes`; score = levels as slots. One head for all types.
- **Question types.** choice ≤255, score 2–10, noul. Order-invariant mode averages cyclic option orders in one batch
  (auto for >10 options): flips 18.9% → 3.3% on Thai, banking77 41.7 → 63.3. Ollama build: Ollama renders each question as a
  chat prompt and reads letters A–Z, so a separate fine-tune (≤26 options, no abstain) was shipped.
- **Calibration.** Stage 3: backbone frozen, slot head + `log_temperature[3]` trained 400 steps with CE + Brier
  (`brier_weight 1.0`); v0.3 T = 1.055 / 1.047 / 1.008 (choice/noul/score). Confidence in the code is 1 − normalised entropy.
- **Training data.** ~1.8M public + 98k synthetic records in v0.1; v0.3 adds 177k real train-split records and 78k targeted
  synthetic (Qwen3.6-35B-A3B, blind-checked). **Native Thai** data (Prachathai, Wisesight, Wongnai, MASSIVE-th, XNLI-th),
  not translated; option-order shuffling and abstain examples.
- **Evals (card).** Public 13-subset bench (Nimble protocol) macro **74.3** vs Nimble-9B 74.8 and Jev 1.13.0 76.0;
  MASSIVE-de-DE 88.3 (Jev 86.9). Thai held-out: MASSIVE-th 90.0, Prachathai 98.1, XNLI-th 77.1, Wisesight 51.6 (weak).
  English median ECE 0.15 vs Thai ≤0.05. ~40 ms on H100 per 3-question request; Doom demo at ~41 ms/decision.
  Not on DI 0.3.2.
- **Novel.** Language CPT before decision SFT (the only clone here that adapts the backbone to the language first);
  fixed 256-slot classifier instead of LM letters or a pointer; explicit abstain slot.

## JevAlt: Deem-4B (EN), Karar-4B (TR), Wähler-4B (DE) (Mert Kaya / Eschatia Labs)

- **Identity.** [mertkayacs/Deem-4B](https://huggingface.co/mertkayacs/Deem-4B) (265 dl),
  [Karar-4B](https://huggingface.co/mertkayacs/Karar-4B) (269 dl),
  [**Wahler-4B**](https://huggingface.co/mertkayacs/Wahler-4B) (313 dl), all created 2026-10-01, modified 10-18; GGUF
  repos `*-GGUF`. Code [mertkayacs/jevalt](https://github.com/mertkayacs/jevalt) (6★, Apache-2.0, pin `2aedb3a`
  2026-10-16), data [mertkayacs/jevalt-data](https://huggingface.co/datasets/mertkayacs/jevalt-data), bench
  `mertkayacs/jevalt-bench`, Space `mertkayacs/JevAlt`. Licence Apache-2.0.
- **Backbone.** `Qwen3_5ForConditionalGeneration` (vision tower kept), 32 layers (24 GDN + 8 full), d 2560, 16 heads / 4 KV,
  ctx 262,144, vocab 248,320, **4,539,265,536 params** (identical to Intern-Decision-4B). Merged bf16 weights plus the LoRA in
  `lora/` (r 32, α 32, all language-model attention/MLP/GDN projections, `lora/adapter_config.json`).
- **Base choice.** A bake-off (`docs/how-it-was-built.md`) on typed-decisions / JevBench / MASSIVE-{en,tr,de}: Intern-Decision-4B
  0.804 / 0.712 (hard) / 0.653 (de) vs Kev-4B 0.670 / 0.541 / 0.643 vs Laya 0.361 / 0.342 / 0.423 → start from Intern-Decision.
  So the language choice is "multilingual Qwen3.5 + language-weighted fine-tune", not a native German base.
- **Input layout / readout.** Byte-identical to Intern-Decision's compiler: chat prompt with `## State` and a
  `## Decision schema` listing every field with single-token symbols, assistant turn a JSON skeleton with one `<decision>`
  per field; logits at each marker restricted to that field's symbols (≤62). One forward pass for all fields.
- **Isolation.** None (inherited from Intern-Decision: later fields see earlier fields and their markers).
- **Question types.** choice (≤62), score 2–10, noul, plus `unknown` option with `abstain: true`; optional
  `reasoning: on|auto` writes a ≤256-token trace first (not System One; off by default, and for German it did not help:
  0.605 both ways, Brier worse with auto).
- **Calibration.** `calibration.json`: temperatures per type and **per type×language** (Wähler: choice:de 1.113, noul:de
  1.232, default 1.10), fitted on 3,224 held-out decisions; **split-conformal thresholds** for 80/90/95% answer sets.
  Confidence `(n·p_max − 1)/(n − 1)`.
- **Training data.** Per language, rows **written natively** by open-weight teachers chosen in a native-feel writer
  bake-off (German: DeepSeek V4 and Gemma 4 won; rows written by Gemma 4 26B-A4B and DeepSeek V4.1 Flash), double-labelled by
  teachers from other labs, soft label = mean of two, kept only on argmax agreement. "Nothing is translated from English."
  Plus fix sets F1–F12 (unknown, negation, dates, numbers, padding to 3k tokens, injection, policies) and converted public
  data (MASSIVE, PAWS-X, Open-Jev, typed-decisions); note that MASSIVE-de and PAWS-X-de are themselves human
  translations/localisations of English sources. German run: 1 epoch shared mix (17,363 rows) + 1 epoch German-weighted
  (5,384 rows, 70% German). Whole project 32.2 USD, ~2.7 A100 h.
- **Evals (card, same-pipeline held-out).** Deem EN 94.7% (Kev-4B 84.7), Karar TR 96.8% (87.1), Wähler DE **92.0%** (81.1),
  Brier 0.119 vs 0.298. External: Wähler 10kGNAD 62.5% vs Kev 65.3%, GermEval 2017 64.3% vs 65.3%, JevBench-hard 65.8 vs 54.1;
  live German test 122/130. Not on DI 0.3.2 (base Intern-Decision-4B = 38.21).
- **Novel.** Per-language specialist heads-free LoRAs on a shared decision base; native-writer bake-off with blind raters;
  per-language temperatures + conformal sets; total cost published.

## sokudan-ja-310m (GeneLab / hiroki-abe-58, Japan)

- **Identity.** [GeneLab/sokudan-ja-310m](https://huggingface.co/GeneLab/sokudan-ja-310m) v0.2.1 (created 2026-09-21,
  modified 09-29, 306 dl, 0 likes); code [hiroki-abe-58/sokudan](https://github.com/hiroki-abe-58/sokudan) (4★, Apache-2.0,
  pin `eccded3` 2026-10-08), PyPI `sokudan`. Licence Apache-2.0 (bench CC BY 4.0).
- **Backbone.** sbintuitions/modernbert-ja-310m (MIT): 25 layers, d 768, 12 heads, FFN 3072, 8192 positions,
  `local_attention` 128 with global every 3rd layer, vocab 102,400 (Japanese SentencePiece). `SokudanJointModel`
  **314,614,274 params** = backbone 314,611,968 + scorer 768 + ordinal head 1,538 (`config.json`). Fully fine-tuned.
- **Input layout.** `<s> instructions </s><s> opt₁ <mask> opt₂ <mask> … </s><s> state </s>` (`question_first`,
  `docs/input_order.md`), one sequence **per question**; state re-encoded per question (latency linear in #questions).
  A "separate" state/question encoding with a cross-attention head was built and lost badly (held-out AUROC 0.506 vs 0.872,
  `docs/architecture.md` §1.1); v1 question packing with a block mask was dropped because the 128-token local window would hide
  the state from most layers (§2).
- **Isolation.** Per question by construction (separate sequences). Options inside a question attend to each other
  bidirectionally.
- **Readout.** Hidden state at each `<mask>` → `scorer` (768→1, no bias) → softmax for choice/bool; **score uses a dynamic-K
  cumulative-link head with RPS loss** (`SOKUDAN_SPEC.md:284,498`): the only ordinal-structured score head in this group.
- **Question types.** choice (4–6 trained), score K=2–7, bool (`noul` accepted as alias).
- **Calibration.** Only bool calibrated by default (T = 2.07 on 9,850 held-out rows; ECE 0.181 → 0.105); choice/score
  temperatures made `bench_en`/RPS worse and are off. **v0.2 is a model soup** of 8 seeds (plain fp32 average incl. head).
- **Training data.** **All synthetic, native Japanese**, label-conditioned generation by `qwen3:30b-a3b-instruct-2507`
  (14 domains, 28 attributes), schema randomisation (shuffle, paraphrase, distractors, scale inversion). Adding the Japanese
  translation of typed-decisions (5,799 rows) raised that test to 0.703 but hurt held-out schemas, so it was rejected.
  Jev was deliberately never queried (TypeSafe ToS).
- **Evals.** `bench_ja` (300, synthetic): choice 0.880, score RPS 0.075, bool AUROC 0.844; `bench_en` choice 0.872 but bool at
  majority level. CPU (Core Ultra 9, 24 thr) 371 ms for 3 questions; MLX fp16 ~10 ms on M1 Max. Not on DI 0.3.2.
- **Novel.** Ordinal head; preregistered release rules and negative results (perm-KL, order averaging, data
  randomisation all measured and rejected); Laya-style presentation checks reported honestly (first-slot bias remains).

## FRIDA-Decisions (ai-forever / SberAI, Russia)

- **Identity.** [ai-forever/FRIDA-Decisions](https://huggingface.co/ai-forever/FRIDA-Decisions) (created 2026-10-02,
  modified 10-10, 1,296 dl, 49 likes); code [ai-forever/FRIDA-Decisions](https://github.com/ai-forever/FRIDA-Decisions)
  (29★, pin `2c878d6` 2026-10-10), [Habr article](https://habr.com/ru/companies/sberbank/articles/1092056/), benchmark
  [artemsnegirev/razvilka](https://huggingface.co/datasets/artemsnegirev/razvilka). Licence MIT.
- **Backbone.** `T5EncoderModel` from ai-forever/FRIDA (encoder of FRED-T5-1.7B, RU/EN embedding model): 24 layers,
  d_model 1536, 24 heads, d_ff 4096 gated-GELU, relative position buckets 32 / max distance 128, vocab 93,651.
  **823,401,216 params** BF16 + separate `head.safetensors` (fp32). LoRA r16 on q/k/v/o (4.7M trainable) merged.
- **Input layout** (`README.md` "How it works"): one packed row
  `[state][question 1][opt A][opt B]…[question 2][yes][no]`; positions **restart after the state** for every question so each
  option sits where it would sit alone. Per-type instruction suffixes in Russian (`decisions_config.json`). Limits: state
  512 (default cut 384), instructions 96, option 256, rows of ≤16 options / 1,024 tokens.
- **Isolation.** **Strict, by block mask**: state sees only itself; a question sees state + itself; an option sees state +
  its question + itself, **never another option or another question**. State K/V are question-independent and cached
  (exact, drift 6.7e-6); vLLM plugin uses the prefix cache. 243 intents: 97,685 → 4,871 tokens, 4.65 s → 0.44 s.
- **Readout.** Linear head on the **mean of each candidate's own token states** → margin; softmax over a question's
  candidates. Order-independent by construction (options never interact): the opposite of Jev's joint ordered reading.
- **Question types.** choice, score (level 0..K−1 + expectation), noul (yes/no candidates), plus **`ranking`** (margins for
  thresholding retrieval candidates).
- **Calibration.** No temperature step documented; listwise softmax for choice/ranking, pairwise BCE for noul. The README
  warns margins are not calibrated across requests.
- **Training data.** 1.42M examples / 151 question types, **71% Russian, 29% English**, "mostly human or naturally labelled",
  ~¼ LLM-generated or model-labelled (their estimate), paraphrase-augmented instructions; one epoch (~50 consumer-GPU hours).
- **Evals.** razvilka (735 RU items): **0.893** vs Jev 0.897 (McNemar p 0.84), Qwen3.5-35B-A3B 0.897, decider-2b 0.833,
  GLiNER2.5-multi 0.576, Laya-multilingual 0.559. RTX 5060 Ti 28–34 ms; ONNX int8 CPU 0.9 s for 384 tokens × 3 questions.
  Not on DI 0.3.2.
- **Novel.** The cleanest implementation of Jev's "encode state once, isolate questions" on an encoder, with exact state
  K/V caching and a vLLM pooling plugin; a fourth question type (`ranking`).

## basal-1.0 / 1.5 and basal-omni-1.0 (Remek Kinas / si5.pl, Poland)

- **Identity.** [Remek/basal-1.0-4.5B](https://huggingface.co/Remek/basal-1.0-4.5B) (2026-09-26, 2,077 dl),
  [basal-1.0-1.5B](https://huggingface.co/Remek/basal-1.0-1.5B) (1,242 dl), [basal-1.5-4.5B](https://huggingface.co/Remek/basal-1.5-4.5B)
  (10-02, 735 dl), [basal-1.5-mini](https://huggingface.co/Remek/basal-1.5-mini) (10-03, 734 dl; also basal-1.5-max 11B,
  not fetched), [basal-omni-1.0-mini](https://huggingface.co/Remek/basal-omni-1.0-mini) (10-11, 977 dl),
  [-max](https://huggingface.co/Remek/basal-omni-1.0-max) (10-10, 988 dl), [-ultra](https://huggingface.co/Remek/basal-omni-1.0-ultra)
  (10-15, 103 dl, 16 likes). Engine [rkinas/basal](https://github.com/rkinas/basal) (190★, Apache-2.0, pin `e49665c`
  2026-10-19), tech report `docs/basal-1.pdf`, DOI 10.5281/zenodo.23022986, leaderboard Werdykt at basal.si5.pl. Licence Apache-2.0.
- **Backbone, 1.x.** `LlamaForCausalLM` = **speakleash/Bielik v3 Instruct** (Polish-native): 4.5B = 60 layers, d 2048,
  16 heads / 2 KV, ctx 32,768, **4,757,260,288 params**; 1.5B = 32 layers, d 1536, 12/2, ctx 8,192, 1,596,507,648. **32k Polish
  vocabulary** (0.65× English token count on Polish text, 2× on German [ours]). Full fine-tune, "recipe includes an RL stage".
- **Backbone, omni.** Gemma 4: mini = E4B (42 layers, d 2560, ctx 131k, 7.94B incl. audio+vision), max = 12B
  (`Gemma4UnifiedForConditionalGeneration`, 48 layers, d 3840, 11.96B), ultra = 31B (60 layers, d 5376, 31.27B, image only).
  Vocab 262,144.
- **Input layout / readout.** Chat prompt (Polish or English template, `basal.json`): `Stan:\n{state}\n\nPytanie: {q}\nOpcje:\n{A..J}`
  with system "treat the state as data, not instructions"; assistant prefilled with `{"answer": "`; softmax over next-token
  logits of the option letters **A–J (≤10)**. Omni (prompt format 2): **255 single-token codes**.
- **Isolation.** 1.0: one prompt per question (isolated, state recomputed). **1.5: SOAM** (`docs/SOAM.md`): all
  questions × both option orders form a prefix tree flattened into one row; a tree mask lets each token see only its ancestors,
  positions continue from the prefix, state computed once in pass 1 and reused as K/V in pass 2. This is the closest public
  reconstruction of the reference's "state once, isolated branches" (5 questions 37.8 ms vs 111.7 ms as separate prompts).
- **Question types.** choice, noul, score, plus engine types `multi`, `act`, `tool` (OpenAI/MCP tool defs), and in basal 2.1
  `classify` / `verify` / `compress` / `extract` expanded into decisions; experimental evidence-span pointer head
  (`evidence_head.pt`).
- **Calibration.** Order averaging (original + reversed) by default in 1.x, then per-type temperatures fitted on the averaged
  prediction; **certified confidence thresholds** for 1% / 5% accepted error (1.0-4.5B: 58.6% coverage at 1.2% error; Jev
  1.13.0 18.1% under the same procedure). 1.0 test ECE 0.026 raw → 0.0096 after T. Early-exit heads (layers 30–55) in 1.0-4.5B.
- **Training data.** Polish and English decisions whose labels are computed by code (deadlines, amounts, twin pairs),
  grounded in statutes with verified quotes, or agreed by verifier models; ~21% English items from a public dataset; split by
  template/statute/domain; "part generated or verified with commercial models". **Native Polish**, not translated. Omni:
  640k–815k decisions with soft targets from basal-1.5 (distillation), text only; images/audio zero-shot via Gemma encoders.
- **Evals (card, own benchmarks).** 1.0-4.5B PL decisions 0.884 vs Jev 1.13.0 0.780 (EN 0.741 vs 0.736, public EN bench 0.740
  vs Jev 0.861). 1.5-4.5B Werdykt 0.721; omni-max 0.823; omni-ultra **0.879**, PL decisions 0.979. 1.0-4.5B 12.5 ms on H100.
  Not on DI 0.3.2.
- **Novel.** Language-native base (the only Polish-vocabulary backbone among all clones); SOAM tree-mask packing over a stock
  causal LM; certified-risk thresholds; early exits; a vendor-neutral runtime contract (`FORMAT2.md` + test vectors).

## Turnike-JEV 0.8B / 4B / 9B (Berkay Hasret, Turkey)

- **Identity.** [berkay123001/turnike-jev-4b](https://huggingface.co/berkay123001/turnike-jev-4b) (adapter, 2026-10-13,
  30 dl), [-4b-merged](https://huggingface.co/berkay123001/turnike-jev-4b-merged), [-9b](https://huggingface.co/berkay123001/turnike-jev-9b)
  (10-14), [-9b-merged](https://huggingface.co/berkay123001/turnike-jev-9b-merged), [-0.8b](https://huggingface.co/berkay123001/turnike-jev-0.8b),
  FP8 copies named in the card. No GitHub repo; code is `karar.py` (~400 lines) + `birlestir.py` + `test_karar.py` in the HF repo.
  ~150 dl total. Licence Apache-2.0. Card states it was written with Claude coding agents.
- **Backbone.** Qwen3.5-0.8B / 4B / 9B **instruct** checkpoints (VLM, vision untouched), frozen; LoRA r16 α32 dropout 0.05 on
  all LM attention/MLP/GDN projections (`adapter_config.json`); trainable 10.8M / 32.5M / 43.3M. Merged 4B = 4,539,265,536, 9B
  = 9,409,813,744 params; 9B = 32 layers, d 4096.
- **Input layout / readout.** Intern-Decision-style prompt (`## State` as JSON string, `## Decision schema` with `A = label:
  desc`), assistant skeleton `{"q1": "?", …}`; at the token before each `"?"` the final hidden state is multiplied with the
  output rows of that question's symbols only (≤62, fp32), softmax, **T = 1**. >62 options: chunked two-stage elimination.
- **Isolation.** None: ≤16 questions per causal prompt, each sees the schema and earlier placeholders (not answers). They
  found the model reading "none of the above" from a sibling question and now split such questions into separate prompts.
- **Training.** KL to target distributions (gold smoothed to 0.9, or crowd distributions), 8–16 questions packed per item,
  max 1,536 tokens, 2,831 steps, lr 1e-4, one RTX 5060 Ti (4B) / RTX PRO 6000 (9B), identical recipe for all sizes.
- **Training data.** Public Turkish and English datasets, LLM-labelled public texts and synthetic data (~47% English); not
  released; "not distilled from any commercial decision API". **Native Turkish**; machine translation appears only in a
  supplementary eval (JevBench TR-50); primary Turkish eval `tr_insan` is human-labelled.
- **Evals.** Full JevBench (Praveenrajus/jev-bench, 22,773 rows) 0.681 / **0.774** / 0.783 vs Jev 1.13.0 0.733; tr_insan 0.748 (4B),
  0.763 (9B) vs Jev 0.727, Microsoft-Decision-1 0.714, Kev-4B 0.708, Intern-Decision-4B 0.675. Raw macro ECE 0.084 (Jev 0.113).
  ~109 ms (bf16) per decision on RTX 5060 Ti. Not on DI 0.3.2.
- **Novel.** Small frozen-base LoRA on an *instruct* Qwen with a documented shortcut audit (sibling leakage, option-count
  leakage); human-labelled native test set.

---

## German-language clones (priority)

No German-native backbone is used by any clone found. All German decision models sit on multilingual decoders (Qwen3.5,
Gemma 4), and the two real German families differ mainly in **data provenance**: JevAlt writes German natively with open
teachers; johannhartmann machine-translates English decision corpora with Claude. Search: HF API `search=` german, deutsch,
decisions-de, decision-de, entscheid, german-decision, typed-decisions-de, jev-de, kev-de, laya-de (created ≥ 2026-09-14;
results in scratch, filtered by id); `notes/hf_candidates.tsv` grep; `gh search repos` "Entscheidungsmodell", "german
decision model jev", "systemone deutsch". Beware: `shisa-ai/shisa-de-*` is "Decision Engine", **not** German.

### Wähler-4B (JevAlt, German member)

The German member of the JevAlt card above. Facts specific to German: base Intern-Decision-4B; second epoch on a 70%-German
mix (5,384 rows); German rows (1,408 teacher-written, 107 village-game, 1,257 fix-set, 4,074 converted public: MASSIVE-de 2,500,
PAWS-X-de 2,500 before splits) **written in German** by Gemma 4 26B-A4B and DeepSeek V4.1 Flash, labelled by two other-lab
teachers (Qwen3.5-35B-A3B for German); temperatures `choice:de` 1.113 / `noul:de` 1.232 and `choice:de` conformal thresholds
(0.80 → 0.210, 0.95 → 0.727). Held-out German 92.0% (same pipeline), but **on external German suites it is level with or below
Kev-4B** (10kGNAD 62.5 vs 65.3, GermEval 2017 64.3 vs 65.3), so the in-pipeline gain does not transfer to classic German
classification (inferred: the native synthetic rows teach the decision format and robustness more than German topic semantics).
Q4_K_M GGUF 2.71 GB, 3.03 GB RAM at 4k context; Ollama `mertkayacs/wahler-4b` (letter output, calibration only via the JevAlt
server). Tokenizer: stock Qwen3.5 (German 1.12× English [ours]).

### johannhartmann decisions-de models and datasets

- **Identity.** [johannhartmann/Qwen3.5-2B_qlora_decisions-de](https://huggingface.co/johannhartmann/Qwen3.5-2B_qlora_decisions-de)
  (created 2026-10-14, 35 dl; GGUF repo 638 dl) and [johannhartmann/gemma-4-E4B-it-unsloth-bnb-4bit_qlora_typed-decisions-de](https://huggingface.co/johannhartmann/gemma-4-E4B-it-unsloth-bnb-4bit_qlora_typed-decisions-de)
  (10-13, 22 dl). Datasets [johannhartmann/decisions-de](https://huggingface.co/datasets/johannhartmann/decisions-de)
  (10-14, 166 dl) and [johannhartmann/typed-decisions-de](https://huggingface.co/datasets/johannhartmann/typed-decisions-de).
  Trained in **Unsloth Studio** ("Train as: decision model"). Weights Apache-2.0; data keeps per-source licences.
- **Backbone.** Qwen3.5-2B instruct (24 layers = 18 GDN + 6 full, d 2048, 8/2 heads, vocab 248,320; **2,274,069,824 params**,
  vision kept) / Gemma 4 E4B-it (42 layers, d 2560, 7,996,156,490 params with audio+vision). QLoRA r64 α64 on q/k/v/o/gate/up/down
  (language **and** vision layers), merged into bf16.
- **Head / readout.** A fresh **Cloudflare Clef joint-schema head** (`joint_head_config.json`: width 512, 2 evidence-routing
  layers, 4 decoder layers, 8 heads), shipped with Cloudflare's unmodified `joint_schema_model.py` (`load_release_model`,
  `systemone`). One causal sequence holds the state and every field; the head routes evidence from the whole sequence to each
  option and lets fields attend to each other (see `clones/cloudflare-clef.md`).
- **Isolation.** None, by design of the Clef head (fields interact). Unsloth places the whole schema before the state and
  truncates the state at 4,096 tokens (dataset card).
- **Calibration.** Unsloth's global temperature 1.56 folded into the head; per-type temperatures in
  `unsloth_decision_config.json` (noul 1.13, choice 0.88, score 0.67) applied only by Unsloth's loader. In-domain ECE 0.031;
  underconfident out of domain (mean conf 0.62 at acc 0.79).
- **Training data.** **Machine-translated**, no human review: `decisions-de` = 167,850 cases / 374,186 questions from 90 sources,
  translated from the train splits of `hotchpotch/bekko-system-one-dataset-v0` with the Anthropic Batch API (Claude Sonnet 5.5
  for questions/options and repeated strings, Haiku 5.5 for the rest, Sonnet review pass); option **keys stay English**, labels
  are the English gold; language-bound tasks (BLiMP, WinoGrande, POS…) excluded; ~10,300 strings changed digits by
  localisation. The Qwen 2B model saw the 19,286-row train config of v1 (card); v2 of the dataset has 180,327 rows.
  `typed-decisions-de` = translation of `LocalLLaMA/typed-decisions` (4 business workflows, teacher-distribution gold carried
  over from English).
- **Evals (card).** 2B: decisions-de validation **0.842** (noul 0.940, choice 0.769, score 0.588), German DI 0.3 sample (2,000
  translated questions, choice/noul only, own kit) 0.793, typed-decisions-de test 0.504. Gemma E4B specialist: typed-decisions-de
  test **0.776**. Not on DI 0.3.2.
- **Novel.** The only openly published **German decision training corpora** (two, with licence filtering per source) and the
  first third-party reuse of the Clef head on a small backbone; GGUF served by llama.cpp at `/v1/systemone`.

### Other German-relevant items (task fine-tunes and apps, no card)

- [TextCortex/laya-cybersec](https://huggingface.co/TextCortex/laya-cybersec) (2026-09-28, 136 dl, licence "other"): Laya
  multilingual (mmBERT-base, 321.9M) fine-tuned for **EN+DE prompt-injection / exfiltration** detection (235,622-row PI-scanner
  fine-tune, `rl_agent_config.json`); German full-suite AUROC 0.878 vs Jev 0.956. Sibling
  [TextCortex/clef-cybersecurity](https://huggingface.co/TextCortex/clef-cybersecurity) (10-05, Apache-2.0): full-parameter
  update of selected layers of Clef-flash (9.53B total), German AUROC **0.974** (Jev 0.956), weak on German skills (0.917).
- [Ramg77/laya-sentiment-multilingual](https://huggingface.co/Ramg77/laya-sentiment-multilingual): Laya-multilingual noul
  sentiment fine-tune on EN/ES/DE (900 train rows), 55.7% → 86.0%.
- Apps on hosted Jev or Laya, no own weights: `KI-Klub-Werder-e-V/kommentarwache` (comment moderation; Jev now, Laya later),
  `mrebbert/Jev-CustomerService-Demo`, `JanEggers-hr/jev_grading_demo`, `GodModeAI2025/JevCoreML` (kev-0.6b + laya in Core ML).
- Not German despite the name: `shisa-ai/shisa-de-2` (Japanese lab, "Decision Engine", Gemma 4 26B-A4B; DI 59.77 #21, ECE 0.207).

## Patterns in this group

1. **Base choice per language splits three ways.** (a) Language-native vocabularies where a strong local model exists:
   Bielik (PL), FRED-T5/FRIDA (RU), ModernBERT-ja (JA). These win on token cost in their language and, for basal and
   FRIDA, on their native benchmarks. (b) Multilingual Qwen3.5 adapted to the language: OpenThai with real CPT, JevAlt and
   Turnike with LoRA only. (c) Gemma 4 for the newest basal-omni line, mainly for media and the 255-code readout.
   German has no group (a) clone.
2. **Native data beats translation as the stated goal**, with one exception. JevAlt, sokudan, basal, Turnike and OpenThai
   all build native-language rows (generated by local teachers or converted from native datasets); sokudan tested adding a
   translated set and rejected it. Only the German decisions-de work is fully machine-translated, and its own out-of-domain
   numbers (0.50 on typed-decisions-de for the generalist) show the cost.
3. **Isolation is rare.** Only FRIDA (block mask, encoder) and basal-1.5 (SOAM tree mask, decoder) reproduce Jev's
   "state once, isolated questions"; sokudan and basal-1.0 isolate by re-encoding per question; OpenThai, JevAlt,
   Turnike and the Clef-head German models let questions see each other.
4. **Readouts:** LM-head letters (JevAlt, Turnike, basal), a fixed slot classifier (OpenThai), encoder marker or mean-pool
   scorers (sokudan, FRIDA), a joint head (decisions-de). sokudan is the only ordinal score head.
