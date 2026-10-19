# Liquid AI d1 (d1-3B, d1-omni-600M)

Liquid AI's open-weight decision models. They come from two separate designs:

- **d1-3B** is the LFM2.5-VL-3B conv/attention hybrid decoder, run as a **tree**: the state is the *trunk*, read once, and each question is a *branch* that continues it. A custom kernel layer gives each branch exactly the trunk plus itself, in both the attention layers and the short-convolution layers. The answer is read **from the LM head at each branch's last token**, restricted to verbaliser tokens (yes/no, option letters, score digits).
- **d1-omni-600M** is the bidirectional LFM2.5-Encoder-350M with a 2-layer transformer **decision head** that scores a `<|mask|>` marker per option. It has a SigLIP2 vision prefix and a FastConformer audio prefix, runs **one full row per question**, and applies per-type, per-option-count temperatures.

Both expose `system_one()` / `system_one_batch()` in remote code. llama.cpp serves both at `/v1/systemone`.

Source pins (fetched 2026-10-19, nothing executed, no weights downloaded):

| ref | what | where |
|---|---|---|
| `HF:3b/` | `LiquidAI/d1-3B` files: `config.json`, `modeling_d1.py`, `lfm2_vl.py`, `hybrid.py`, `runner.py`, `prompt.py`, `api.py`, README, tokenizer/processor configs, chat template, LICENSE, `api.json` (with `expand[]=safetensors`), `tree.json` | `notes/hf-raw/LiquidAI__d1-3B/files/` |
| `HF:omni/` | `LiquidAI/d1-omni-600M` files: `config.json`, `modeling_d1.py`, `encoder.py`, `prompt.py`, `vision.py`, `audio.py`, README, tokenizer config, LICENSE, `api.json`, `tree.json` | `notes/hf-raw/LiquidAI__d1-omni-600M/files/` |
| `HF:enc/` | Base `LiquidAI/LFM2.5-Encoder-350M` (`config.json`, `modeling_lfm2_bidirectional.py`) | `notes/hf-raw/LiquidAI__LFM2.5-Encoder-350M/files/` |
| `BLOG` | "Open d1: Edge decision models for text, vision, and audio" (2026-10-07; served at both `/blog/open-d1` and `/blog/d1-open`, byte-identical), plus the two architecture figures | `notes/hf-raw/LiquidAI__d1-3B/web/www.liquid.ai_blog_open-d1.txt`, `d1-3b-arch.png`, `d1-omni-600m-arch.png` |
| `HBLOG` | "Introducing d1: The most capable decision model, now with vision" (hosted d1, 2026-10-05) | `.../web/www.liquid.ai_blog_d1-decision-model.txt` |
| `DOCS` | docs.liquid.ai `decision-models.md`, `d1-3b.md`, `d1-omni-600m.md`, `d1.md` (hosted) | `.../web/docs.liquid.ai_*.md` |
| `SK` | `github.com/Liquid4All/liquid-d1-skills` (first-party agent skill for the hosted API), shallow clone `fa6e725` (2026-10-01), read only | `/home/ai/projects/jev-clones/sources/liquid-d1-skills` |
| `DI` | Decision Index board snapshot, panel 0.3.2, 2026-10-16 | `notes/hf-raw/_spaces/dix/index.json` |

Citation form: `HF:3b/hybrid.py:219` means file:line; `BLOG:34` means line 34 of the text dump. There is no GitHub repo for the open models' code. The code ships only in the HF repos, and the only first-party GitHub repo is the hosted-API skill.

> **Main finding.**
> - **d1-3B is the cleanest "shared state, isolated branches" implementation in the survey on a non-transformer hybrid.** LFM2's short convolutions keep only a 3-tap window and no recurrent state. A branch therefore needs just the trunk's last 2 conv inputs per conv layer (`causal_conv` with `Tree.conv_order`). Attention runs trunk-causal plus branch→trunk plus branch-causal-self, merged by log-sum-exp. Branch positions restart at the trunk length. The result is mathematically identical to running each (state + question) as its own chain, with the state computed once and **nothing cached, copied or padded** (`HF:3b/hybrid.py:1-14`).
> - The readout is plain **LM-head verbalisation**. There is no trained head, and no temperature is applied at inference.
> - **d1-omni-600M does not share the state across questions at all.** Because the encoder is bidirectional, every question is a separate full row (state re-encoded), read by a small transformer head at `<|mask|>` markers. Its kev-style delimiters (`<state> <q> <opt> … </opt> <decide>` and the `<¦name¦>` escape) are the same scheme as kev's.

---

## Identity

| | |
|---|---|
| Vendor | Liquid AI (LFM family; LFM2 technical report arXiv 2511.23404, cited by both cards) |
| Weights (HF) | `LiquidAI/d1-3B` (created 2026-10-05 09:55Z, last modified 10-07; 22,004 downloads / 449 likes), `LiquidAI/d1-omni-600M` (created 10-05 17:18Z, modified 10-07; 14,419 dl / 137 likes) (`HF:*/api.json`) |
| Parameters (`?expand[]=safetensors`) | d1-3B **3,123,483,888** BF16 (one 6.25 GB `model.safetensors`); d1-omni **587,161,089** **F32** (2.35 GB). README split for omni: 381M trunk + decision head, 94M vision encoder, 112M audio encoder (`HF:omni/README.md:60-61`) |
| Other formats (DOCS) | `d1-3B-GGUF` / `d1-omni-600M-GGUF` (llama.cpp ≥ 0.7.0, Q8_0 shown), `d1-3B-MLX-{8bit,4bit,fp16}` (mlx-vlm from git), `d1-3B-w8a8` (INT8 for Ampere+ / Jetson). Community: `litert-community/d1-3B-LiteRT`, `OceanLabs/Coral-2-3B` (`notes/hf-sweep.md:518`) |
| Languages (card metadata) | d1-3B: ar, zh, en, fr, de, hi, id, it, ja, ko, pl, pt, ru, es, th, vi. Omni: en, de, es, fr, it, nl, pl, pt, ar, hi, ja, ru, tr, vi, zh. Omni audio was trained on English speaker↔assistant requests only (`HF:omni/README.md:85-87`) |
| Family | Hosted **d1** and **d1-ultra** (closed, Liquid API / OpenRouter / Vercel), plus the open **d1-3B** and **d1-omni-600M** (`DOCS:decision-models.md:31-66`). d1-ultra is "built on a larger open-weight model with Liquid-specific customizations" (`DOCS:d1.md:21`). The docs do not say that hosted d1 is d1-3B. Hosted d1 decisions take 200–300 ms (`HBLOG:26`) vs 8–16 ms for d1-3B, and are billed per question as separate prompts (`SK:references/api.md:53-55`), so it is probably a different model (inferred) |
| Demos | HF Space `LiquidAI/system-one-arcade` (10 camera/game demos on d1-3B), NVIDIA Isaac Sim hardware-in-the-loop on Jetson (`BLOG:143-144`); Jetson AI Lab guides for both models |

---

## Architecture

### Backbone

| | d1-3B | d1-omni-600M |
|---|---|---|
| Class | `Lfm2VlForConditionalGeneration` (`lfm2_vl`), `auto_map.AutoModel → modeling_d1.D1Model` (`HF:3b/config.json:2-7`) | `D1OmniModel` (`d1_omni`) with `D1OmniConfig` (`HF:omni/config.json:2-9`) |
| Base | `LiquidAI/LFM2.5-VL-3B`. The text backbone was **weight-averaged with LFM2.5-2.6B** to "create a better base model" (`BLOG:34`); the figure labels the stack "LFM2.5-2.6B, 30 layers, causal" | `LiquidAI/LFM2.5-Encoder-350M` (`Lfm2BidirectionalForMaskedLM`), bidirectional attention + non-causal short conv (`HF:enc/modeling_lfm2_bidirectional.py:1-14,116-128`) |
| Layers | **30 = 22 short-conv + 8 GQA attention** (attention at indices 2,5,9,13,17,21,24,27) (`HF:3b/config.json:49-80`) | **16 = 10 conv + 6 attention** (attention at 2,5,8,10,12,14) (`HF:omni/config.json:37-54`) |
| Width / heads | d 2048, 32 q / 8 KV heads (head_dim 64), per-head q/k RMSNorm, full RoPE θ 1e6 (`HF:3b/lfm2_vl.py:22-45`) | d 1024, 16 q / 8 KV (head_dim 64), q/k RMSNorm, RoPE θ 1e6 (`HF:omni/encoder.py:36-59`) |
| Conv operator | `out(C · conv(B · x))`, depthwise **causal**, `conv_L_cache` 3 taps, no bias (`HF:3b/lfm2_vl.py:48-63`) | same gated form, **centred** 3-tap (left, self, right) (`HF:omni/encoder.py:62-82`) |
| MLP | SwiGLU, `intermediate_size` 10,752 used as-is (`block_auto_adjust_ff_dim: false`) (`HF:3b/config.json:33,48`, `lfm2_vl.py:66-80`) | SwiGLU, 2/3·6656 rounded up to a multiple of 256 = 4,608 (`HF:omni/encoder.py:85-97`) |
| Vocab / tying | 128,000, tied embeddings (`HF:3b/config.json:94-99`) | 65,536 (`HF:omni/config.json:31`) |
| Positions | `max_position_embeddings` 32,768 | 128,000 (base), but the model caps at `max_length` 16,384 (`HF:omni/config.json:11,59`) |
| Vision | SigLIP2 NaFlex "shape-optimized 400M": 27 layers, width 1152, patch 16; 2× pixel-unshuffle projector (GELU, hidden 2048); 512-px tiles, 2–10 tiles + thumbnail, 64–256 tokens per tile (`HF:3b/config.json:9-27,99-118`, README:81) | SigLIP2 from LFM2.5-VL-450M: 12 layers, width 768, patch 16; same tiling/projector logic re-implemented (`HF:omni/vision.py`, `config.json:62-75`) |
| Audio | – | 17-layer FastConformer (d 512, 8 heads, conv kernel 9, ff ×4), 128-bin log-mel at 10 ms, 8× subsampling → one embedding per **80 ms**, adapter 512→1024 + residual MLP; LFM2-Audio / NVIDIA Canary lineage; ≤30 s (`HF:omni/audio.py:1-5,17-29,189-249`, `config.json:76-85`) |
| Dtype | BF16 | trained **F32**. fp16 keeps the top answer on all checked rows; **bf16 flips** 0.8% of text and 1.7% of audio top answers (`HF:omni/README.md:170-172`) |

d1-3B's remote code **replaces HF's LFM2 language model** with its own `hybrid.LanguageModel` under HF's module names, so checkpoints load and save unchanged. HF's SigLIP2 tower and projector are kept (`HF:3b/lfm2_vl.py:1-8,88-94`). Any non-LFM2 checkpoint passed to `SystemOne` would fall back to stock transformers (`HF:3b/runner.py:34-50`).

### Input encoding

**d1-3B** (`HF:3b/prompt.py:305-354`; ChatML; `bos` = `<|startoftext|>`, `eos` = `<|im_end|>`, `pad` = `<|pad|>`, `HF:3b/tokenizer_config.json`). No system turn by default (`SYSTEMS["none"]`; an alternative `"isolated"` system prompt exists but is unused, `:15-22`):

```
TRUNK  (prefix_text, shared by every question of the request)
<|startoftext|><|im_start|>user\n
[<image> markup per picture, as the chat template writes it]      <- runner.py:220-227
{state: str as-is | json.dumps(indent=2) for JSON}\n\n            <- state_block, style "json_only" (:246-266)
\nQUESTION:\n

BRANCH (suffix_text, one per question)
choice : {instructions}\n\nOptions:\nA {desc or label}\nB {desc or label}\n\nReply with the option code only.
noul   : {instructions}[\nYes: {criteria.true}\nNo: {criteria.false}]\n\nReply with yes or no only.
score  : {instructions}\n\n0 {level0}\n1 {level1}\n...\n\nReply with a single digit 0-{L-1} only.
<|im_end|>\n<|im_start|>assistant\n▮          <- answer slot = last branch token; empty "lead" on LFM2-VL (:322-330)
```

- **Choice option lines** use `option_style="desc"`, so an option is written as `"{code} {description}"`. **The option key is shown only when the description is null** (with `_` → space) (`:272-276`). `name_desc` would show both, but it is not the default.
- **Codes** (`option_codes`, `:62-70`): the labels themselves if they are already single letters; otherwise `A..Z` for ≤26 options, else `00, 01, …`. Each code must be a single token. A code that is not falls back to a pool: A–Z, 00–99, a–z, `#0..#199`, then two-letter pairs (`:73-103`). The question schema's 255-option limit is therefore bounded in practice by how many single-token aliases the tokenizer has (inferred).
- **Not sanitised.** `<|im_end|>` or `Options:` text inside the state or instructions reaches the model unchanged. Non-string `instructions` and criteria are interpolated with Python `str()` (dict repr), not JSON, because `as_question` passes them through raw (`:138-148,288`) (inferred quirk).

**d1-omni-600M** (`HF:omni/prompt.py:1-10,87-111`). Delimiters are reserved tokens, and the marker is the encoder's `<|mask|>` (`:20-22`):

```
<bos> <|reserved_7|> state  <|reserved_8|> instructions
      <|reserved_9|> <|mask|> option_0 text <|reserved_10|>
      <|reserved_9|> <|mask|> option_1 text <|reserved_10|> ...  <|reserved_11|>
       (<state>)                (<q>)  (<opt>) ^marker        (</opt>)          (<decide>)
```

- Caller text has `<|name|>` rewritten to `<¦name¦>`, so delimiters cannot be forged (`:55-57`). This is the same rule and the same five roles as kev (`clones/kev.md` Input encoding).
- Option texts: choice `"key"` or `"key: description"`; score `"level i: text"`; **noul `["false: …", "true: …"]`**, with defaults "no, the statement does not hold" / "yes, the statement holds" (`:69-84`). After an audio prefix, choice options become `"option_000: desc"` and noul becomes `false: no` / `true: yes`, "as the audio questions were trained" (`:69-80`, `modeling_d1.py:24,95-99`).
- **Truncation budget per row.** Options get `budget = max(96, min(24·K + 32, max_len/2))` tokens. Each option's text is cut to `per = max(2, (budget − 3K)/K)` tokens. Instructions are cut to `max(16, budget)`, and the state is cut on the right to whatever room is left (`:91-107`). With images `max_len` is only **896**, so 50 options would get ~5 text tokens each (computed). `ValueError` if the last marker does not fit (`:109-110`).

### Attention mask

**d1-3B: trunk + branches ("tree"), exact isolation.** `Tree(p, lengths)` packs `[trunk | branch 0 | branch 1 | …]` on one axis. Positions are `0..P-1` for the trunk and **restart at P** for every branch (`HF:3b/hybrid.py:29-40`). Per layer:

| Layer kind | Trunk tokens | Branch tokens | Code |
|---|---|---|---|
| Attention (8 layers) | causal SDPA over the trunk | (a) every branch query over **all P trunk keys**, one flash call with each KV head's query group folded into the query axis so trunk K/V are read once for all questions; (b) each branch over **its own tokens, causally**, via varlen flash with `cu_seqlens`; (a) and (b) merged exactly by log-sum-exp | `hybrid.py:143-153,156-177,180-184,219-236` |
| Short conv (22 layers) | causal depthwise conv | one conv over all branches, each **led by the trunk's last `keep = taps − 1 = 2` inputs** (gather/scatter indices from `Tree.conv_order`) | `hybrid.py:43-55,100-110` |
| Norms, projections, MLP | token-wise over the whole packed axis, once | | `hybrid.py:1-14,258-260` |

```
packed axis   [ t0 t1 ... t(P-1) | a0 a1 a2 a3 | b0 b1 b2 | c0 c1 c2 c3 c4 ]
position        0  1 ...   P-1   |  P P+1 ...  |  P P+1.. |  P  P+1 ...        <- each branch restarts at P
leaves (read)                            ^a3         ^b2              ^c4      <- Tree.leaves = last token per branch

attention (rows = queries, cols = keys)          short conv, 3 taps (receptive field per output)
            trunk   A     B     C                 a0 <- t(P-2), t(P-1), a0     <- led by the trunk's tail
 trunk    [  \      .     .     .  ]              a1 <- t(P-1), a0, a1
 A        [  #      \     .     .  ]              b0 <- t(P-2), t(P-1), b0     <- never sees a*
 B        [  #      .     \     .  ]
 C        [  #      .     .     \  ]              (no recurrent state to copy: LFM2 conv keeps only a 3-tap window)

 '#' = one flash call over the trunk for all branch tokens; '\' = varlen causal per branch; merged by LSE
```

- **Isolation is exact by construction.** Every branch computes what the chain `trunk + branch` would compute alone. The docstring claims "the same outputs as HF's own implementation" (`HF:3b/lfm2_vl.py:5-7`), and the only stated difference is bf16 kernel numerics by batch shape (`runner.py:125-127`). This matches Jev's documented behaviour and kev's block-causal or row designs.
- **Why the conv branches are cheap.** The hybrid problem kev hit on Qwen3.5 does not arise. Gated DeltaNet carries a recurrent state, so kev had to copy cache replicas per row. LFM2's conv is a 3-tap FIR, so branching only needs the last 2 trunk inputs per conv layer, gathered by index (`hybrid.py:100-110`).
- **Sliding-window support.** The code also supports a sliding window, where a branch sees only the trunk's last `window − 1` keys. LFM2 does not use it (`window=None`, `lfm2_vl.py:44`).
- **Batching.**
  - A request with several questions is one tree (state = trunk, questions = branches). Branches are split into consecutive chunks under a `token_budget` of 65,536 branch tokens; each chunk recomputes the trunk (`runner.py:81,145-162,204-208`).
  - **Single-question requests from different states** are packed into one tree whose trunk is their **longest common token prefix** (in practice the `<|startoftext|><|im_start|>user\n` header). Each state then lives inside its own branch, which is still exact (`runner.py:122-135,177-182`).
  - A single question, or an image request with one question, runs as a plain chain (`:129-131,196-199`). Optional `torch.compile(mode="reduce-overhead")` uses CUDA graphs on NVIDIA; it is refused on ROCm, where "the CUDA graphs fault after a few dozen calls" (`:106-110`).

**d1-omni-600M: one bidirectional row per question.** `probabilities_batch` encodes `(state, question)` as a separate sequence for every question (`HF:omni/modeling_d1.py:107-108`). Rows are batched longest-first under 65,536 tokens (`:120-131`).

- Within a row, attention is **fully bidirectional** over the text: the state sees the question and the options, and options see each other, like a cross-encoder. The only masks are padding and the media rule (`HF:omni/encoder.py:130-151`).
- **Media prefix is text-blind.** Media queries cannot attend to text keys (`:134-139`), and the conv's right tap is zeroed at the last media position (`:78,140`). "The prefix is a function of the media alone" (`:6-8`).
- **Cost.** The vision or audio tower runs once per request (`modeling_d1.py:94,97`), but the trunk re-runs over prefix + state for **every question**. Compute grows linearly with questions × (media + state). This is the price of a bidirectional encoder: the state's representation depends on the question, so it cannot be shared.
- Questions are perfectly isolated, since each is its own sequence.

```
d1-omni, Q questions = Q rows:
 row q:  [ media prefix (attends media only) | <bos><state> state <q> instr (<opt><mask> text </opt>)xK <decide> ]
          '------------ bidirectional over text; media never reads text ------------'
                  -> DecisionHead over text positions -> logits at each <mask>
```

### Readout head

**d1-3B: LM-head verbaliser, no trained head.**

```
h_leaf     = final-norm hidden state at each branch's last token           hybrid.py:291-297
z          = lm_head(h_leaf)            (tied embeddings, 128k vocab)     lfm2_vl.py:96-104
logz       = z - logsumexp(z)           (log-softmax over the full vocab) runner.py:137-143
s_option   = max over the option's token forms of logz[t]                  prompt.py:155-186
              noul : yes-forms {yes,Yes,YES} vs no-forms {no,No,NO}
              choice: {code, " code"}         score: {"0".."L-1"}
p          = softmax(s_option)          (renormalised over the options only) prompt.py:187-191
```

The `calibration` hook (`calibration.apply(q, scores)`) exists, but `D1Model.engine` constructs `SystemOne` with no calibration, so **no temperature is applied** (`modeling_d1.py:18-22`, `runner.py:75,99`). Probability mass on non-option tokens is discarded by the renormalisation, so the model's "off-schema" mass is invisible to the caller.

**d1-omni-600M: `DecisionHead`** (`HF:omni/encoder.py:154-167`, `modeling_d1.py:133-164`):

```
x      = trunk_hidden[text positions only] + TypeEmb[choice=0|score=1|noul=2]      (B × L × 1024)
x      = 2 × TransformerEncoderLayer(d 1024, 16 heads, ff 4096, pre-norm, padding mask)
g_k    = x[position of <|mask|> for option k]
logit_k= Linear(GELU(Linear(LayerNorm(g_k))))          one shared MLP
p      = softmax(logit / T[type:bucket])   (text-only requests)   -> noul reversed from [false,true] to [true,false]
```

The marker sits *before* the option text, but because the trunk and head are bidirectional it still reads the option, the instructions and the state. Unlike kev's `<decide>` query, there is no separate question query; each option is scored by its marker alone, after the head's joint layers over the row.

### Question types

| type | d1-3B readout | d1-omni options | answer (both `api.py` / `prompt.py:answer`) | confidence |
|---|---|---|---|---|
| `noul` | P(yes-forms) vs P(no-forms) | `[false: …, true: …]`, reversed after softmax | `noul = P(true)` | none |
| `choice` | single-token codes (A..Z / native letters / 00..) | `key` or `key: desc` | `choice = argmax`, `probabilities` keyed by option name, in request order | **`p_max`** |
| `score` | digits `0..L-1`, which must be single tokens, so **2–10 levels** (`prompt.py:163-170`) | `level i: text` | `score = Σ i·p_i`, `legend`, `probabilities` | **`p_max`** |

- Validation (omni): choice needs ≥2 options, score 2–10 levels, noul criteria optional, as `{true,false}` or `{yes,no}` (`HF:omni/prompt.py:32-40`). d1-3B validates only through the verbaliser (`RuntimeError` when a code cannot be made single-token).
- Option order is request order in both models. **No permutation invariance**: d1-3B reads the options as a lettered list, and omni positions differ per option. The blog says "shuffling answer options" in training mattered (`BLOG:34`), which gives robustness, not invariance.
- Neither model implements Jev's `box`, the `TypeSafe-Danger-Zone` headers, or Jev's closed-form confidence.

### Context and limits

| | d1-3B | d1-omni-600M |
|---|---|---|
| Model window | 32,768 positions (`config.json:81`; README/DOCS "32K") | `max_length` **16,384**, text + image + audio positions together (`config.json:11`) |
| Per-modality text cap | images resized to ≤1 MP before the processor (`runner.py:31,53-65`) | text cut to **896** with images and **15,360** with audio (`config.json:12-13`; `modeling_d1.py:95-103`) |
| Overflow | **No length check in the shipped code** (none in `runner.py`/`prompt.py`). An over-long state is passed to the model as-is (inferred). Hosted d1 returns 422 over 32,768 tokens or 256 KiB per question-prompt (`SK:references/api.md:53-55`) | right-truncation of the state; option/instruction budgets as above; error if media leave fewer than 64 positions (`modeling_d1.py:103-105`) |
| Questions per request | unbounded; branches chunked by the 65,536-token budget | unbounded; rows chunked by 65,536 tokens |
| Media per request | any number of images (one tree trunk) | images **or** one audio clip, not both (`modeling_d1.py:91-92`); audio ≤30 s, padded to ≥0.5 s |
| DI note | – | "requests over the model's 16,384-token default budget were rerun by the lab with the budget lifted to 128,000" (`DI` notes) |

### Calibration

- **d1-3B: none at inference.** Raw softmax over the verbaliser log-probs. Calibration comes only from training (shuffled options, long inputs, shortcut fixes, seed/mixture merges; `BLOG:34`). DI calibration: ECE **0.0468**, Brier 0.390, acc 0.702, 0.82% of answers ≥0.95 wrong (`DI`).
- **d1-omni: post-hoc temperatures in `config.json`**, keyed by `type:option-count bucket` (`HF:omni/config.json:16-27`, `prompt.py:129-131`, `modeling_d1.py:159-161`):

| key | T | | key | T |
|---|---|---|---|---|
| `noul:2` | 1.666 | | `score:3-5` | 1.730 |
| `choice:2` | 1.747 | | `score:6-10` | 1.000 |
| `choice:3-5` | 1.400 | | fallback `choice` / `score` / `noul` | 1.0 |
| `choice:6-10` | 1.175 | | (`score:2` has no entry → 1.0) | |
| `choice:11+` | 1.373 | | | |

  These temperatures are applied **only to text-only requests**; "image and audio answers are the model's softmax as trained" (`modeling_d1.py:95-101`, `README.md:190-191`). All T > 1 means the raw head is over-confident, most of all on binary decisions. DI: ECE **0.128**, Brier 0.586, the worst calibration of the Liquid/Cloudflare models (`DI`).

### Multimodal

- **d1-3B (text + images).**
  - Images are embedded by HF's LFM2-VL tower and scattered into the trunk's `<image>` placeholder slots *before* the tree runs, so all questions share one image encoding (`lfm2_vl.py:96-104`, `runner.py:200-208`).
  - The processor pads each image to 1,024 patches; the code trims the padding ("the same answer for up to half the work") (`runner.py:236-243`).
  - A state of `None` means "the images are the whole state" (`prompt.py:313-319`).
  - Card vision results: mean of 11 image benchmarks 74.1 vs base 73.9; 45.1 with images removed (`HF:3b/README.md:235-256`).
- **d1-omni (text + images, or text + audio).**
  - Vision: LFM2-VL-style smart resize to 64k–256k px² multiples of 32, a tile grid for large images (≤10 tiles + thumbnail), SigLIP2 → 2×2 unshuffle → MLP into the trunk width; several images are concatenated in order (`vision.py:18-113`).
  - Audio: NeMo-style mel front end in fp32, FastConformer, adapter + residual (`audio.py`).
  - Media are a **prefix of embeddings, not tokens**, and the head ignores media positions (`modeling_d1.py:136-146`).
  - Training note from the blog: vision used an adapter plus **LoRA active only when the input had images**, with a frozen vision encoder; the LoRA was later merged (`BLOG:35`).

---

## Training

Liquid released no training code or data. Everything below is from the blog.

- **d1-3B** (`BLOG:34`):
  - Base = **weight average of LFM2.5-2.6B and the text backbone of LFM2.5-VL-3B**.
  - "We then fine-tuned checkpoints with different random seeds and data mixtures before merging them again" (a model soup).
  - "Training on long inputs, shuffling answer options, and fixing shortcuts in the data made a bigger difference than more advanced techniques."
  - The DI board labels it a full fine-tune. No loss, data sources, teacher, RL or distillation are disclosed. The verbaliser readout implies CE on the answer token (inferred).
- **d1-omni-600M** (`BLOG:35`):
  1. Fine-tune LFM2.5-Encoder-350M on decision tasks.
  2. Audio: train a FastConformer encoder + adapter, then fine-tune the audio encoder with the text backbone frozen.
  3. Vision: take LFM2.5-VL-450M's encoder (frozen), train an adapter plus LoRA on the backbone that is active only with images.
  4. Fine-tune the full model, merge the LoRA, and **average the weights with the previous checkpoint** "to regularize the final model".
  5. Fit the per-type temperatures on text afterwards (inferred from the config, not described).
- The omni card excludes HelpSteer2 from its table because "it may overlap with d1-omni-600M's training data" (`HF:omni/README.md:220-221`). That is the only hint of the data mix.

---

## Evals

**Card: Decision Index 0.2.1** ("official scorer, not a leaderboard submission"; `HF:3b/README.md:198-213`, `HF:omni/README.md:199-216`):

| Model | DI | Knowledge | Language | Retrieval | Tools | Arts |
|---|---|---|---|---|---|---|
| **d1-3B** | **48.57** | 23.8 | 56.4 | 52.8 | **74.5** | **36.3** |
| d1-omni-600M | 15.95 | 8.3 | 12.9 | 35.0 | 15.1 | 6.8 |
| Winnow-12B | 50.02 | 33.8 | 56.0 | 54.0 | 71.0 | 30.0 |
| Decider 35B-A3B | 47.11 | 31.8 | 55.5 | 54.7 | 56.5 | 32.6 |
| Decider 4B / 2B | 40.70 / 28.97 | | | | | |

Liquid's claim is "best decision model under 10B on the Decision Index 0.2.1" (`HF:3b/README.md:56-57`). Knowledge (23.8) is the weak spot.

**DI board 0.3.2** (Liquid's own submission on AMD MI325X with `model.system_one`; private tests run by the lab the same way; `DI`):

| | balanced (Full) | public | breadth | acc | ECE | Brier | latency median / p95 |
|---|---|---|---|---|---|---|---|
| d1-3B | **38.90** | 48.99 | 46.22 | 0.702 | 0.0468 | 0.390 | 16.1 / 55.9 ms |
| d1-omni-600M | **9.09** | 17.90 | 16.11 | 0.530 | 0.128 | 0.586 | 10.9 / 34 ms |
| Jev 1.13 (ref) | 60.11 | 57.96 | 57.27 | 0.738 | 0.078 | 0.358 | 524 ms (HTTP) |

- d1-3B category skills: knowledge **0.243**, language 0.565, retrieval 0.528, tools 0.745, arts 0.389.
- As with Clef, the balanced score falls ~10 points below the public one, where Jev's rises. Here the likely cause is model size on private knowledge items, not only overfitting (inferred).

**Benchmarks as decisions** (`HF:3b/README.md:219-233`, `HF:omni/README.md:223-234`, `BLOG:38-84`):

| | d1-omni-600M | d1-3B | Decider 4B | Decider 2B |
|---|---|---|---|---|
| SQuAD 2.0 | 74.0 | **85.3** | 76.0 | 67.7 |
| Civil Comments | **95.8** | 93.0 | 92.8 | 93.6 |
| MASSIVE intent | 86.1 | 87.3 | **88.3** | 81.1 |
| PubMedQA | 61.3 | **66.0** | 63.3 | 65.7 |
| BoolQ | 77.7 | 86.7 | **89.0** | 87.3 |
| XNLI | 74.7 | 85.0 | **88.6** | 85.0 |
| PAWS-X | **79.5** | 76.9 | 69.8 | 59.5 |
| Mean (7) | 78.4 | **82.9** | 81.1 | 77.1 |

- The d1-3B card also lists HelpSteer2 36.7, giving an 8-row mean of 77.1.
- Other results: d1-3B DecisionBench (eng v1, 23.9k rows) 71.8, Fast Decisions dev 69.3, ImajevBench 64.0 (base 66.8). d1-omni Fast Decisions dev 76.9.
- There are no audio decision benchmarks: "Dedicated audio decision benchmarks are currently an open problem" (`BLOG:85`).

**Speed** (d1-3B, warm, one request at a time, median of 20; `HF:3b/README.md:164-192`):

| device | 1 question | 3 q, one pass | 3.4k-token state | 384-px image | 64 states packed |
|---|---|---|---|---|---|
| RTX 4090 (bf16, CUDA graphs) | 8 ms (16 ms eager) | 21 ms | 102 ms | 17 ms | 475/s |
| AMD MI325X | 9 ms | 14 ms | 44 ms | 18 ms | 1,106/s |
| Apple M5 Pro (mps) | 30 ms | 41 ms | 640 ms | 62 ms | 78/s |
| Jetson AGX Thor | 16 ms | 20 ms | 220 ms | 35 ms | 262/s |
| Jetson AGX Orin 64 GB | 26 ms | 35 ms | 560 ms | 83 ms | 110/s |
| Jetson Orin Nano | 50 ms | 73 ms | 1,640 ms | 202 ms | 38/s |

Three questions cost ~1.3× one question (Thor 16 → 20 ms), the tree's shared trunk at work. No speed numbers are given for d1-omni ("early research release", `HF:omni/README.md:193-195`).

---

## Serving / API compatibility

- **Python (remote code).** `AutoModel.from_pretrained(repo, trust_remote_code=True)` and then:
  - `model.system_one(state, {name: question}, images=None[, audio=None])` → `{"answers": {...}, "usage": {"input_tokens": n, "output_tokens": 0}}`;
  - `system_one_batch([...])` for many requests;
  - `probabilities(...)` (omni) for the raw distributions.

  The request is the Jev body minus `model`, as positional arguments; the response omits `model` (`HF:3b/modeling_d1.py:24-31`, `api.py:49-61`; `HF:omni/modeling_d1.py:66-83`).
- **`input_tokens`** for d1-3B = trunk + all branches (`runner.py:209`), close to Jev's "state counted once". For omni it is the sum over rows of (media + text), so the state is counted once **per question** (`modeling_d1.py:109`; README: "counts every position the trunk read").
- **HTTP.** No server ships in the HF repos. llama.cpp ≥ 0.7.0 `llama-server -hf LiquidAI/d1-3B-GGUF:Q8_0` serves `POST /v1/systemone` with `state`, `questions`, and an `images` data-URL extension. Omni needs `-b 4096 -ub 4096` (`DOCS:d1-3b.md:97-135`, `d1-omni-600m.md:116-170`). Not checked here: whether llama.cpp's port reproduces the tree batching and the omni temperatures.
- **Hosted d1 / d1-ultra** (different models, see Identity):
  - `POST https://api.liquid.ai/decisions/v1/systemone`; also OpenRouter `liquid/d1` and Vercel AI Gateway (`HBLOG`, `DOCS:d1.md`).
  - ≤128 questions per request; choice 2–255 options; score 2–10 levels. "Each question is its own prompt carrying the whole state", and usage counts each one (`SK:references/api.md:39-55`).
  - Images: ≤8 per request, ≤10,000 32×32 patches in total, 1.5 tokens per patch, billed again per question (`DOCS:d1.md:926-951`).
  - Priced at $0.04/M input in Liquid's own comparison (`HBLOG` methodology).
  - The skill repo (10-01) still says d1 returns 422 for images and names an image model "LFM2.5-S1-VL" (`SK:references/api.md:41`). Vision on hosted d1 arrived with the 10-05 blog.

---

## Licence

- **LFM Open License v1.0** (`license: other`, `license_name: lfm1.0`; LICENSE 10.6 kB, identical for both models apart from whitespace; `HF:3b/files/LICENSE`).
- Apache-2.0-style grants of copyright and patent rights, with a **Commercial Use Limitation**: commercial use is licensed only for entities **under $10M annual revenue** (the "Threshold"). Qualified non-profits are exempt for non-commercial or research use (`LICENSE:27-33,51-57`).
- The patent licence terminates on patent litigation (`:37`), and the attribution and NOTICE duties apply (`:45-47`).
- The blog's "Download, fine-tune, and deploy without restrictions" (`BLOG:148`) overstates this for large companies.

---

## Changelog

| Date | Event | Source |
|---|---|---|
| ≤2026-10-01 | Hosted **d1** (text-only, "experimental release") on the Liquid API; `liquid-d1-skills` repo initial commit `fa6e725` (10-01) | `HBLOG:23`, `SK` |
| 10-05 | Hosted d1 gains **vision**, d1-ultra listed; "first model to rival Jev on text decisions". `d1-3B` (09:55Z) and `d1-omni-600M` (17:18Z) repos created | `HBLOG`, `HF:*/api.json` |
| 10-07 | "Open d1" blog; cards finalised (last modified 10-07 ~20:23Z); GGUF / MLX / W8A8 / Jetson AI Lab variants; day-one llama.cpp support | `BLOG:17,87,149`, DOCS |
| 10-16 | DI 0.3.2 board lists d1-3B 38.90 and d1-omni 9.09 (Liquid submission) | `DI` |

There is no versioning on the HF repos (no tags) and no revision history beyond the 10-05 → 10-07 card edits.

---

## Notable design choices vs Jev and vs kev

1. **Trunk/branch tree attention on a conv hybrid (d1-3B).** It achieves what Hume reconstructed for Jev (shared state prefix, mutually masked question branches, branch positions continuing the state) on a non-transformer backbone, with an exact merge (LSE) and no KV cache. kev's attention-only models use a dense `[L×L]` additive mask, and its Qwen3.5 models fall back to independent rows with copied DeltaNet/conv caches. d1-3B's 3-tap conv makes branching nearly free. Of the designs surveyed, it is the closest match to Jev's documented "state ingested once, questions in parallel, independent answers".
2. **Verbaliser readout instead of a head (d1-3B).** kev, Clef and most clones train a pointer or a scoring head. d1-3B reads the LM head at the answer slot over single-token codes or digits and renormalises over the options. That keeps the model a normal VLM checkpoint (GGUF/MLX/llama.cpp ports are straightforward) and works across 2–10 score levels. The costs: option-code capacity limits, letter or position bias, hidden off-schema probability mass, and a dependence on the option text being shown as a description rather than its key.
3. **Encoder cross-encoding (d1-omni).** The bidirectional encoder lets the state be read *conditioned on the question*, the opposite trade from Jev and kev, which share the state across questions. It buys quality per parameter at 0.6B but makes cost linear in the number of questions and gives up state-once billing. The delimiter set and `<¦…¦>` escaping match kev's scheme (inferred lineage), but the readout sits on a per-option `<|mask|>` marker scored after a 2-layer joint head, not on `</opt>` against `<decide>`.
4. **Calibration.** d1-3B ships no temperature. d1-omni ships **per-type × option-count temperatures** (kev: one global T; strands-decider: per type), applied only to text. Its media answers are uncalibrated by design, which shows in its DI ECE of 0.128.
5. **Recipe.** Weight averaging and model soups (2.6B + VL text backbone; seed/mixture merges; LoRA merge + averaging for omni), option shuffling and long-input training. This sits close to kev's lessons (low-lr adapters, shuffle augmentation, shortcut-fixing none-pairs). It is the opposite of Clef's Brier + RL objective work.
6. **Edge first.** These are the only big-org clones with published Jetson, Apple and ROCm numbers and INT8/GGUF/MLX releases. d1-3B at 8–16 ms per decision is ~30–60× faster than hosted Jev round trips, at a DI 0.3.2 cost of 21 points (38.90 vs 60.11), concentrated in knowledge (0.24).
7. **Contract fidelity.**
   - Field names match Jev, but `confidence = p_max` (not Jev's formula) in both models.
   - No `model` field is echoed.
   - d1-3B has no context guard.
   - d1-omni silently truncates option texts to a few tokens under image budgets.
   - Question IDs are never sent to the model (both models).
   - Questions are fully isolated in both models, matching Jev, unlike Clef.
