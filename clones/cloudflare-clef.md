# Cloudflare Clef (clef, clef-flash, clef-omni)

Cloudflare's open-weight Jev replication. A Qwen multimodal backbone (Qwen3.8-27B, Qwen3.5-9B or the Qwen3-Omni-30B-A3B thinker) reads **one causal prompt that holds the state and every question**. A ~110–128M-parameter **joint schema head** then routes evidence from the whole sequence to every option of every question, lets the questions attend to **each other**, and emits one logit per option. It serves TypeSafe's `POST /v1/systemone` contract on Workers AI and SGLang.

Source pins used for this profile (fetched 2026-10-19, nothing executed, no weights downloaded):

| ref | what | where |
|---|---|---|
| `HF:clef/` | `Cloudflare/clef` repo files: `config.json`, `joint_head_config.json`, `joint_schema_model.py`, README, tokenizer/processor configs, `model.safetensors.index.json`, LICENSE, `api.json` (with `expand[]=safetensors`), `tree.json`. The SGLang PR pins the weights at `clef@2f3de3dd` | `notes/hf-raw/Cloudflare__clef/` |
| `HF:flash/` | `Cloudflare/clef-flash`, same set. **`joint_schema_model.py` is byte-identical to clef's** (`diff`), so code citations below say `HF:clef/` for both. PR pin `clef-flash@17f0b0ad` | `notes/hf-raw/Cloudflare__clef-flash/` |
| `HF:omni/` | `Cloudflare/clef-omni`, same set; its `joint_schema_model.py` is a 29 kB fork that adds audio/video decoding | `notes/hf-raw/Cloudflare__clef-omni/` |
| `B1` | Blog, "Introducing Clef: our open-source decision models, and new RL fine-tuning platform" (2026-10-01; M. Chen, A. Reneau, K. Flansburg). Text dump, cited by line | `notes/hf-raw/Cloudflare__clef/web/blog.cloudflare.com_clef-decision-models_.txt` |
| `B2` | Blog, "Introducing Clef-omni with full multimodality, plus a faster Clef and a cheaper Clef-flash" (2026-10-09) | `.../web/blog.cloudflare.com_clef-faster-cheaper-multimodal_.txt` |
| `WAI` | Workers AI model pages (`index.md`) and the input/output JSON schemas for `@cf/cloudflare/clef`, `clef-flash`, `clef-omni` | `.../web/developers.cloudflare.com_workers-ai_models_*_index.md`, `clef*_schema-*.json` |
| `PR` / `SGL` | SGLang PR #42721 "[Model] Support Clef and Clef-Flash decision models on /v1/systemone" (opened 2026-10-06 by `areneau-cf`, merged 2026-10-09, merge commit `bd2d73d`). The PR JSON plus three files at the merge commit, read only | `.../web/sglang_pr42721*.json`, `/home/ai/projects/jev-clones/sources/sglang-clef/` |
| `DI` | Decision Index board snapshot (`multimodalart/jev-decision-index`, `data/index.json`, panel 0.3.2, generated 2026-10-16) | `notes/hf-raw/_spaces/dix/index.json` |

Citation form: `HF:clef/joint_schema_model.py:116` means file:line. `B1:991` means line 991 of the blog text dump. There is **no first-party GitHub repo**; the release code is the single `joint_schema_model.py` in each HF repo, and the serving code is the SGLang PR. (Third-party repos found by `gh search`, e.g. `Gjusev/clef-router`, `haginot/clef-quickstart`, are not used here.)

> **Main finding.** Clef does **not** keep Jev's question isolation, and the coupling is deliberate. (1) The backbone sees one plain causal sequence `[system][STATE][FIELD 1]…[FIELD N][suffix]`, so field *k*'s tokens already read fields 1..k−1, including their **question IDs**, which are written into the prompt. (2) The head's memory is the whole sequence, and its "global" vector is the last token, which has read every field. (3) The four field layers are `nn.TransformerDecoderLayer`s whose self-attention runs over **all fields of the record with no mask**. Adding, removing, renaming or reordering one question can therefore change every other answer. Cloudflare's blog describes the design as "allowing individual field parameters to cross-attend with other fields" (`B1:990`), and the system prompt tells the model to "Decide every field jointly" (`HF:clef/joint_schema_model.py:24-27`). Two smaller departures: choice options are **sorted alphabetically by key** before encoding (`:56`), and the release code returns `confidence = p_max`, not Jev's closed-form confidence (`:533,540`). The SGLang route fixes the second one, but the first is in the model.

---

## Identity

| | |
|---|---|
| Vendor | Cloudflare, Workers AI team. Blog authors Michelle Chen, Alex Reneau, Kevin Flansburg (B1), with Kevin Jain on B2. The SGLang integration was written by `areneau-cf` (PR) |
| Weights (HF) | `Cloudflare/clef` (created 2026-09-30 21:15Z, last modified 10-09; 21,421 downloads, 2,218 likes), `Cloudflare/clef-flash` (09-30, 35,576 dl, 841 likes), `Cloudflare/clef-omni` (10-09, 4,856 dl, 127 likes) (`HF:*/api.json`) |
| Parameters (`?expand[]=safetensors`) | clef **27,356,728,560** BF16; flash **9,409,813,744** BF16; omni **35,259,818,545** BF16. The omni count includes the unused talker and code2wav weights. The DI board lists the served sizes as 27.80B / 9.65B / 31.83B (`DI`). The joint heads are separate files: 256.1 MB / 243.5 MB / 218.4 MB `joint_head.safetensors` (`HF:*/tree.json`) |
| Hosted | Workers AI `@cf/cloudflare/clef`, `@cf/cloudflare/clef-flash`, `@cf/cloudflare/clef-omni` (`WAI`, `B1:962`). Docs say the models are also reachable through AI Gateway (`B2`) |
| Lineage | Cloudflare's earlier DiffusionGemma decision experiment, published "in the same week that Jev came out", which exposed LLM logprobs as probabilities (`B1:989`). That model appears as the "DiffusionGemma Jev" column in the card. Clef "builds upon this concept" with a Qwen backbone (`B1:990`) |
| Name | Music clef; "the CF hearkens to Cloudflare" (B1) |
| Status | Production on Workers AI. Cloudflare's own teams use it for threat-intel domain classification, spam-issue closing on the docs repo, EmDash plugin phishing moderation and DLP PII scanning (`B1:836`, B2) |

---

## Architecture

### Backbone

| Model | Base (`base_model`) | Text stack | Vision / audio | Context (config) | Source |
|---|---|---|---|---|---|
| **clef** | `Qwen/Qwen3.8-27B` (post-trained, not -Base) | `qwen3_5`, 64 layers = **48 Gated-DeltaNet linear-attention + 16 full attention** (every 4th layer, indices 3,7,…,63), d 5120, 24 q-heads / 4 KV, head_dim 256, partial RoPE 0.25 with interleaved mRoPE `[11,11,10]`, θ 1e7, vocab 248,320, untied LM head, `mtp_num_hidden_layers 0` | Qwen3.5 ViT: depth 27, width 1152, `out_hidden_size` 5120, patch 16, spatial merge 2, `deepstack_visual_indexes []` | `max_position_embeddings` 262,144 | `HF:clef/config.json:16-118,124-138` |
| **clef-flash** | `Qwen/Qwen3.5-9B` | 32 layers = 24 DeltaNet + 8 full attention, d 4096, 16 q / 4 KV, head_dim 256, same RoPE, vocab 248,320, untied | same ViT, `out_hidden_size` 4096 | 262,144 | `HF:flash/config.json:15-102` |
| **clef-omni** | `Qwen/Qwen3-Omni-30B-A3B-Instruct` (thinker only is used) | `qwen3_omni_moe` thinker: 48 layers, d 2048, 32 q / 4 KV, head_dim 128, **MoE 128 experts, top-8**, expert FFN 768, vocab 152,064, θ 1e6, interleaved mRoPE `[24,20,20]` | thinker `audio_tower` (32 layers) + `visual` (27 blocks); talker (20-layer MoE) and code2wav are **shipped unchanged but not loaded** | 65,536 | `HF:omni/config.json:199-389`, `HF:omni/README.md:41-43` |

- **Weights are merged backbones.** `load_release_model` loads a "merged backbone" with `Qwen3_5ForConditionalGeneration.from_pretrained` and then the head from `joint_head.safetensors` with `strict=True` (`HF:clef/joint_schema_model.py:494-520`). The safetensors index contains `lm_head.weight`, `model.language_model.*` (848 tensors for clef, 424 for flash) and `model.visual.*`, with no MTP tensors (`HF:clef/model.safetensors.index.json`). Per the blog, the backbone was frozen and **rank-256 LoRA adapters** were trained jointly with the head, so the shipped backbone is base + merged LoRA (`B1:991`). The DI board labels all three "full fine-tune" (`DI` meta.kind); that is the board's classification and conflicts with the blog.
- **The LM head is kept, but only as an embedding table.** Nothing is decoded. The head reads `get_output_embeddings().weight` (the untied LM-head matrix) for a lexical prior over option tokens (`HF:clef/joint_schema_model.py:382,490`). Omni replaces `thinker.lm_head` with `Identity`, so `outputs.logits` *are* the final hidden states, and keeps the original matrix aside for the prior (`HF:omni/joint_schema_model.py:578-580,608,612`).
- **No KV cache.** `use_cache=False` in loading and forward (`HF:clef/joint_schema_model.py:481,514`). There is no prefix sharing across requests in the reference code; the SGLang integration also disables the radix cache (PR).
- Text-only records skip the vision tower by calling `model.language_model` directly (`:475-477`).
- **Joint head size** (computed from `joint_head_config.json` and the module definitions, and matching the file sizes to within the 12,376-byte safetensors header): clef **128.06M**, flash **121.76M**, omni **109.18M** parameters in BF16. The differences come from the six `hidden_size → 1024` projections.

### Input encoding

`encode_record` (`HF:clef/joint_schema_model.py:103-196`) builds **one token sequence per request**, using Qwen's chat template markers but no special decision tokens:

```
<|im_start|>system
Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>
<|im_start|>user
STATE:
[<|vision_start|><|image_pad|><|vision_end|> ... <|video_pad|> ...]\n      <- media placeholders, before the state (:158-161)
{state rendered: string as-is, else json.dumps(sort_keys=True, separators=(",",":"))}   <- (:35-43, :162)

SCHEMA FIELDS:

FIELD 1
ID: <question_id>                          <- the caller's key IS model input (:116)
TYPE: choice
INSTRUCTION: [[ rendered instructions ]]   <- question_span (:119-124); empty -> question_id (:121-122)
ALLOWED OPTIONS:
OPTION 1: [[ {"description":"...","option_id":"billing"} ]]   <- option_span = this JSON (:130-136)
OPTION 2: [[ {"description":"...","option_id":"technical"} ]]
END FIELD

FIELD 2
...
<|im_end|>
<|im_start|>assistant
<think>

</think>

JOINT SCHEMA DECISIONS:                    <- last token = "global vector" (:154-157, :358)
```

- Tokenisation: `tokenizer(text, add_special_tokens=False)` piece by piece (`:77-78`). The `<|im_start|>` strings are written in plain text, so they become the real special tokens. The state, instructions and option text are **not sanitised**, so a state containing `<|im_end|>` or a literal `FIELD 3\nID: …` reaches the model unchanged (inferred from the absence of any escaping; compare kev's `<¦name¦>` rewrite).
- Option texts use **mean-pooled spans**, not boundary tokens. Each option's span is the JSON `{"description": …, "option_id": …}` (keys sorted by `render`, so `description` comes first). Every option span therefore also contains the shared literal tokens `{"description":"`, `","option_id":"`, which dilute both the span mean and the lexical mean (inferred).
- **Truncation.** The schema, prefix and suffix must fit in `max_length` (default **16,384** for clef/flash, **64,000** for omni), or `ValueError` is raised (`:165-169`; `HF:omni/joint_schema_model.py:40`). The state is then **silently cut on the right** to fit (`:170`). `max_state_tokens` is an optional extra cap (`:163-164`).
- Request JSON keys beyond `state`, `questions`, `images`, `videos`, `media_kwargs` (and `audio` on omni) are ignored. `id` is only a record label (`:194`).

### Attention mask

The backbone runs **plain causal attention over the packed request**. The only mask is right padding (`collate_records`, `:199-239`). The DeltaNet layers carry their recurrent state through the whole sequence, so every later token depends on every earlier one.

```
Backbone visibility (rows = queries, cols = keys; '\' causal, '#' visible, '.' blocked)
DeltaNet layers give the same visibility through their recurrent state.

              sys+STATE   FIELD 1   FIELD 2   FIELD 3   suffix
 sys+STATE  [    \          .         .         .        .   ]
 FIELD 1    [    #          \         .         .        .   ]
 FIELD 2    [    #          #         \         .        .   ]     <- field 2 has read field 1's ID,
 FIELD 3    [    #          #         #         \        .   ]        type, instructions and options
 suffix     [    #          #         #         #        \   ]     <- global vector g = last token

Jev / kev:  every question block sees the state only (block-causal or separate rows).
```

The head adds two more coupling paths (next section). The memory `M` is every token of the sequence (`:356-357`). The field layers use `TransformerDecoderLayer(fields, memory)` with no `tgt_mask` (`:422-423`), i.e. **full bidirectional self-attention among all N fields**, then cross-attention to `M`.

**Isolation verdict** (from the code; not measured by us):

| Path | What it couples | Code |
|---|---|---|
| Backbone causal prefix | Field *k*'s hidden states depend on fields 1..k−1: their IDs, types, instructions and option texts. Later fields cannot affect earlier field spans in the backbone | `:110-148`, causal LM |
| Head memory `M` | Every option query and every field cross-attends to all tokens, including other questions' text | `:357,400-401,422-423` |
| Global vector `g = H[-1]` | Added to every field (`:418`) and to every prior anchor (`:433-436`). The last token has read the whole schema | `:358` |
| Field self-attention | 4 layers of unmasked attention across the N field vectors | `:313-326,421-423` |
| Option routing layers | **No** option↔option attention: `EvidenceRoutingLayer` is cross-attention only (options → memory). Options of different questions are concatenated into one query set, but they interact only through `M` | `:242-278,399-402` |

Consequences: (a) Jev's guarantee that one can "add or remove questions without changing the others' results" does not hold. (b) A secret placed in a sibling question's instructions is readable by later questions in the backbone and by all questions in the head; Hume's 0.00-vs-0.90 sibling-secret test (`clones/jev-reference.md` §3) would be expected to fail (inferred). (c) Request key order matters: fields are numbered in dict iteration order (`:112`). (d) Question IDs are model input, which Jev documents as "not sent to the model". Cloudflare's training data "permutat[es] field orders" to make the model robust to order (`B1:991`), but that is robustness, not invariance.

### Readout head

`JointSchemaHead` (`HF:clef/joint_schema_model.py:281-459`), config `{hidden_size: 5120|4096|2048, width: 1024, routing_layers: 2, layers: 4, heads: 16, feedforward: 4096}` (`HF:*/joint_head_config.json`). All three models use the same widths. Per record:

```
H   = LayerNorm(last_hidden_state)[:len]                     (T × d)        :353-356
M   = W_mem H                                                (T × 1024)     :357      memory = whole prompt
g   = H[T-1]                                                 last token     :358
q_k = mean H[instruction span of field k]                                   :359-364
c_kj= mean H[option span kj]                                 contextual     :373-378
l_kj= mean E_out[token ids of option span kj]                lexical (LM-head rows) :379-383

-- stage 1: option-specific evidence routing (all options of all fields as one query set)
o_kj = W_oc c_kj + W_ol l_kj + W_oq q_k                                     :388-399
2 x EvidenceRoutingLayer:  o <- o + MHA(LN o, LN M, LN M);  o <- o + FFN(LN o)   :242-278, :400-402

-- stage 2: joint cross-field attention
a_kj = softmax_j( o_kj . W_q q_k / sqrt(1024) )                              :405-414
f_k  = W_q q_k + LN( sum_j a_kj o_kj ) + W_g g + TypeEmb[noul|choice|score]  :415-420
4 x TransformerDecoderLayer(norm_first, GELU, 16 heads):                      :313-326, :421-424
      f <- f + SelfAttn(f over ALL fields)    f <- f + CrossAttn(f, M)    f <- f + FFN
f_k  = LN(f_k)

-- stage 3: schema-bound scoring
prior_kj    = exp(min(s_p, ln100)) * cos( l_kj , q_k + g )                   :433-439   "lexical prior"
joint_kj    = exp(min(s_j, ln100)) * cos( f_k , LN o_kj )
            + MLP_1024( [f_k, o_kj, f_k*o_kj, |f_k - o_kj|] )                :440-454
logit_kj    = prior_kj + sigmoid(gamma) * joint_kj                           :455-457
p_k         = softmax_j(logit_kj)          (no temperature)                  :568
```

- `s_p`, `s_j` and `gamma` are learned scalars initialised at 0, so at initialisation both scales are 1 and the gate is 0.5 (`:335-337`).
- The blog's terms map onto this: "two-stage attention routing", "every valid choice extracts context relevant to the prompt" = stage 1; "field parameters … cross-attend with other fields and back to the original payload" = stage 2; "lexical prior" = `prior`; "schema-bound scoring" = stage 3 (`B1:990`, `B2:854`).
- Unlike kev's pointer head, the representations are **span means**, not boundary tokens, and the only "decision" representation is the instruction span mean, which has *not* read the options (it precedes them causally). Option-awareness enters the field vector through the routed option summary and through the field layers.

### Question types

| type | options fed to the head (`question_options`, `:46-57`) | answer (`systemone_answer`, `:523-543`) | confidence (release code) |
|---|---|---|---|
| `noul` | exactly `[true, false]`, in that order. Default descriptions "The proposition is true or the answer is yes." / "…false or the answer is no.", overridden by `criteria.true/false`; other keys are ignored | `noul = P(true)`, 4 decimals | none |
| `choice` | criteria keys **sorted alphabetically** (`sorted((str(key), value) …)`, `:56`); the response maps back to request order (`:528-534`) | `choice = argmax`, `probabilities` | **`p_max`** (`:533`) |
| `score` | levels `"0".."L-1"` in order, description per level (`:57`) | `score = Σ i·p_i`, `legend`, `probabilities` | **`max_i p_i`** (`:540`) |

- Validation in `systemone()`: `model` must be a string, `state` must be present, ≥1 question, type in {noul, choice, score}, non-empty `criteria` for choice/score (`:552-561`). There is no upper bound on options or questions in code. `instructions` is optional and defaults to the question ID.
- Hosted limits (`WAI` schema): **1–64 questions** per request, IDs `[A-Za-z0-9_.-]` up to 100 characters, choice 2–255 options, score 2–10 levels, `instructions` required (string, object or array).
- `TYPE: <type>` is written into the prompt (`:116`) and a learned `type_embedding` is added per field (`:300,419`).
- **Option order.** Alphabetical sorting removes request-order effects (Jev's documented first-option bias) but substitutes alphabetical-position effects. Keys `"a_refund"` vs `"z_refund"` would see different positions (inferred). Score levels keep their order, as they must.

### Context and limits

| | clef | clef-flash | clef-omni |
|---|---|---|---|
| Backbone window (config) | 262,144 | 262,144 | 65,536 |
| Reference code default `max_length` | 16,384 (`:106`) | 16,384 | **64,000** (`HF:omni/…:40`) |
| Hosted context (WAI) | **65,536** | **24,576** (cut from 64k on 10-09; "only 0.24% of requests exceed 24k", `B2:957`) | **64,000** |
| Claimed trained context | "64k context window (compared to Jev's 32k)" (`B1:839`) | "trained to support a 256k context window" (`B2:957`) | – |
| Overflow behaviour | schema too long → error; state silently right-truncated (`:165-170`). Hosted: media over the window → request fails, otherwise the text state is truncated (`WAI:clef.md:51`) |||
| Questions per request | code: unbounded; hosted 64 | | |
| Media per request (hosted) | ≤4 images, data URLs only, 4 MiB/16 MP each, 13 MiB body | same | ≤4 images, ≤4 audio clips (300 s each), ≤2 videos (60 s each, 2 fps), 16 MiB audio+video |

The DI lab ran clef-omni with the window raised to 65,536; 33 ToolRet requests "whose options alone exceed it" were scored wrong (`DI` clef-omni notes). This is a direct cost of putting all options of all questions in one sequence.

### Calibration

- **Training-side only.** The blog describes label-smoothed cross-entropy plus a **Brier loss** "to refine probability calibration", and **RLCD** as a secondary target (`B1:991`; same recipe for omni, `B2:856`).
- **No temperature or post-hoc calibrator** ships. `joint_head_config.json` has none, and `systemone` applies a raw `softmax(-1)` to the head logits (`:568`).
- **Confidence differs by serving path.**
  - Release `systemone`: `confidence = p_max` for choice and score (`:533,540`).
  - SGLang `/v1/systemone`: per the PR, answers go "through the route's existing `_answer`, so `confidence` uses the same formulas as other models on the route". Those are Jev's formulas: `(n·p_max − 1)/(n − 1)` for choice, and `max(0, 1 − spread/uniform_spread)` for score (`SGL:serving.py:405-427,494-510`).
  - Workers AI says only "derived from the probabilities" (`WAI` output schema); which formula it uses is unknown.
- DI calibration (32 benchmarks, confidence = probability on the chosen option): clef ECE **0.0395**, Brier 0.306, 2.3% of answers ≥0.95 wrong; flash ECE **0.0249** (lowest of the three), Brier 0.333, 0.88%; omni ECE 0.0403. For comparison, Jev 1.13 has ECE 0.078 on the same board (`DI`).

### Multimodal

- **clef / flash (image + video).** Images and video frames go through the Qwen3-VL processor (`Qwen3VLProcessor`, `Qwen2VLImageProcessor`, patch 16, merge 2, `shortest_edge` 65,536 px², video fps 2, 4–768 frames; `HF:clef/processor_config.json`). Placeholder tokens are inserted **after `STATE:\n` and before the state text** (`:88,158-161`). `mm_token_type_ids` are offset into the batch (`:96-99,225-233`). Text-only and multimodal records can share a batch (README). Hosted image billing: one token per 32×32 block after resize to ~65k px²–1 MP, plus 3 marker tokens, ≤1,024 tokens per image (`WAI:clef.md:40-49`).
- **clef-omni (image + audio + video).** Adds loaders that accept PIL images, paths, `http(s)` URLs (fetched with `urllib.request.urlopen`), base64 data URLs or bytes. Audio is decoded with PyAV to 16 kHz mono. Video is sampled at **2 fps**, frames are downscaled to ≤256·32·32 px, padded to an even frame count, and the **soundtrack is used (`use_audio_in_video`) only when every video in the record has one** (`HF:omni/joint_schema_model.py` `_media_bytes`, `load_audio`, `load_video`, `_encode_media`). Images use `{"shortest_edge": 65,536, "longest_edge": 1,048,576}`. Each record is run **unpadded, one per forward**, "so padding never changes a record's probabilities" (`:584-591`).
  - Inferred risk: the reference `systemone` will dereference arbitrary URLs and **local file paths** that appear in a request's media lists, so exposing it directly as an HTTP handler would allow SSRF and local file reads. Workers AI accepts embedded data only ("Remote URLs are not accepted", `WAI`).
- Omni hosted latency: text ~130 ms median, image ~150 ms, audio "a few hundred ms", a 21-s video with sound ~1.5 s (`B2:857`).

```
                    clef / clef-flash                                clef-omni
 images ─► Qwen3.5 ViT (27L) ─┐                images ─► thinker.visual ─────────┐
 video  ─► (2 fps)  ──────────┤                audio  ─► thinker.audio_tower ────┤
                              ▼                video  ─► visual (+ its soundtrack)┤
 [sys][STATE: <media pads> state][FIELD 1..N][suffix]   (same layout)            ▼
              │  one causal prefill, Gated-DeltaNet/attn 3:1          MoE thinker 48L, top-8/128
              ▼                                                                  │
        last_hidden_state ────────────────────► JointSchemaHead ◄────────────────┘
                                                     │
                     one logit per option of every question → softmax per question
```

---

## Training

What is published comes from the two blog posts. **No training code, data, configs or metrics are released** (the HF repos hold only inference files).

- **Parameters trained.** Frozen Qwen3.8-27B (clef) and Qwen3.5-9B (flash); the routing head is "jointly optimized … alongside rank-256 low-rank adapters" (`B1:991`). Omni follows "the same techniques … freezing the Qwen3 backbone, training low-rank adapters (LoRA)" (`B2:856`). The released checkpoints have the adapters merged (inferred from "merged backbone", `HF:clef/joint_schema_model.py:500`, and from there being no adapter files).
- **Losses.** "Label-smoothed cross-entropy for valid schema outputs paired with a Brier loss to refine probability calibration" (`B1:991`).
- **RLCD.** "We also developed Reinforcement Learning for Calibrated Decisions (RLCD) to serve as a secondary optimization target":
  - partial credit for adjacent ordinal choices (score);
  - a reward for "fully precise record outputs", i.e. every field of a record correct, which is a joint, cross-question reward that fits the joint head;
  - a "reference penalty to prevent distribution shift" (a KL-to-reference-style term; inferred) (`B1:991`).
  - The name is TypeSafe's (`clones/jev-reference.md` §3). Cloudflare presents its own RLCD with no relation claimed and no recipe published.
- **Data.** "Our own internal synthetic datasets permutating field orders, prompts, and schema structures" (`B1:991`). No public datasets and no Jev distillation are mentioned. Teachers, sizes and any benchmark train-split use are unknown.
- **Schedule.** "We decided we wanted to do something in the decision model space on a Friday evening, trained the model over the weekend, and launched it on Thursday" (`B2:832`). Inferred: Fri 09-25 → weekend 09-26/27 → launch Thu 10-01.
- **Productised fine-tuning.** Cloudflare offers customer fine-tuning of Clef, first through its FDE team and later self-serve. The pipeline is AI Gateway (dataset capture) → Workers AI rollouts → Containers RL sandbox → a new "Trainer" → redeploy via BYO model (Cog/Replicate lineage) (B1).

---

## Evals

**Card, Decision Index 0.2.1** (Cloudflare's internal run, 43 benchmarks; selected rows, Jev = hosted Jev; `HF:clef/README.md:191-235`, `HF:omni/README.md:177-220`):

| Benchmark | Clef | Clef-flash | Clef-omni | Jev |
|---|---|---|---|---|
| BANKING77 (macro-F1) | 94.2 | 90.9 | 94.8 | 79.7 |
| CLINC150+OOS | 97.4 | 66.8 | 97.7 | 89.3 |
| Home appliance sim (exact) | 83.0 | 97.7 | 69.3 | 52.3 |
| MMLU | 90.3 | 91.8 | 92.7 | 91.7 |
| **MMLU-Pro** | 65.9 | 65.3 | 70.6 | **82.7** |
| **GPQA Diamond** | 48.0 | 51.0 | 47.4 | **78.3** |
| **BBH** | 73.7 | 68.9 | 69.0 | **92.9** |
| GSM8K | 80.8 | 67.3 | 79.0 | 79.9 |
| RAGTruth (halluc. F1) | 79.4 | 35.6 | 42.0 | 76.5 |
| When2Call MCQ | 72.4 | 65.6 | 63.3 | 81.0 |
| ForecastBench (Brier ↓) | 13.9 | 10.6 | 11.7 | 17.4 |
| CLadder | 94.0 | 97.7 | 97.2 | 72.6 |
| Median latency (ms) | 209.3 | 38.8 | – | 524.1 |
| p95 latency (ms) | 238.6 | 122.4 | – | 536.0 |

The pattern: Clef beats Jev on classification, tool and NLU sets, and loses by 15–30 points on knowledge and reasoning (MMLU-Pro, GPQA, BBH, When2Call).

**Card, TypeSafe workflow evals** (`HF:clef/README.md:241-247`, `HF:omni/README.md:226-232`):

| Workflow | Clef | flash | omni | Jev |
|---|---|---|---|---|
| Invoice exact / primary | 64.7 / 86.2 | 57.1 / 73.3 | 60.2 / 82.0 | 61.8 / 83.1 |
| Customer service exact | 76.3 | 77.0 | 71.6 | 76.0 |
| Security incidents exact | 62.9 | 61.7 | 61.7 | 61.7 |
| Agent trace primary | 68.5 | 69.8 | 65.8 | 71.6 |

**DI board 0.3.2** (lab runs, 37-benchmark panel incl. private tests, RTX PRO 6000; `DI`):

| | balanced (Full) | public | breadth | acc | ECE | Brier | median latency |
|---|---|---|---|---|---|---|---|
| clef | **53.08** | **61.71** | 61.04 | 0.778 | 0.0395 | 0.306 | 102 ms |
| clef-omni | 48.28 | 56.54 | 56.04 | 0.772 | 0.0403 | 0.323 | 191 ms |
| clef-flash | 47.61 | 56.15 | 55.17 | 0.756 | 0.0249 | 0.333 | 53 ms |
| Jev 1.13 (ref) | 60.11 | 57.96 | 57.27 | 0.738 | 0.078 | 0.358 | 524 ms (HTTP) |

- Clef's **public** score is above Jev 1.13's (61.71 vs 57.96), but its **balanced** score, which includes the private split, is 7 points below (53.08 vs 60.11). Jev rises from public to balanced; Clef drops 8.6. A plausible reading is fitting to public benchmark distributions; that reading is inferred, and the training data is undisclosed.
- Category skills for clef: knowledge 0.528, language 0.613, retrieval 0.628, tools 0.807, arts 0.489 (`DI`).
- Omni was a lab run of the card's `joint_schema_model` path, not a Cloudflare submission.
- The PR's SGLang parity check: top choice 12/12 vs the reference on payloads of 3–64 questions, 255 options, ≤32k states and 1–4 images; max |Δp| 0.068 (clef) / 0.011 (flash), attributed to BF16 noise (PR).

---

## Serving / API compatibility

- **Contract.** "Fully compatible with Jev and SystemOne" (`HF:clef/README.md:31`).
  - Request `{model, state, questions{id:{type, instructions, criteria}}}` → response `{model, answers, usage{input_tokens, output_tokens}}`.
  - Release code reports `output_tokens: 0` and `input_tokens` = prompt length (`:572-576`). Note that this prompt includes every question ID and the schema scaffolding.
  - Extensions: `images`, `videos` (clef/flash), plus `audio` (omni).
- **Not Jev-identical.**
  - No `box` type, and no `TypeSafe-Danger-Zone-*` headers.
  - Confidence formula differs in the release code.
  - Question IDs are model input.
  - Answers are not isolated across questions.
  - Hosted limit of 64 questions (Jev: unlimited) and a `model` field restricted by regex to `clef|clef-flash` (`WAI` schema).
- **Workers AI.** `POST https://api.cloudflare.com/client/v4/accounts/{id}/ai/run/@cf/cloudflare/clef` or `env.AI.run("@cf/cloudflare/clef", {...})`. The standard Workers AI envelope applies, so this is not the bare `/v1/systemone` path (`WAI:clef.md:55-158`). Pricing: clef **$0.24/M** input tokens; flash **$0.038/M** (was $0.09 at launch; now below Jev's $0.042); omni **$0.15/M**; output free (`B2:954`, `WAI`). Cloudflare commits to not reading, storing or training on requests unless the customer uses fine-tuning (B1).
- **SGLang** (PR #42721, merged 2026-10-09; the blog says it ships in SGLang 0.5.22; image `lmsysorg/sglang:dev-clef`, `sglang serve --model-path Cloudflare/clef`; `HF:clef/README.md:94-118`):
  - It detects a `Qwen3_5ForConditionalGeneration` checkpoint that has `joint_head_config.json` and serves it in embedding mode with **no radix cache, no chunked prefill, no CUDA graphs, TP=1/PP=1** ("the head reads full LM head rows").
  - It rebuilds the trained prompt with a `decision_layout` of spans, and runs the head batched through varlen FlashAttention, with span means from FP32 prefix sums.
  - It refuses `/v1/decisions`, `/encode`, LoRA and `chat_template_kwargs`.
  - Clef-omni on SGLang: "coming soon" (`HF:omni/README.md:150`).
- **Speed (SGLang, 1×H200, BF16, c=1, median of 25; PR):**

| | 3 q (511 tok) | 64 q (6.5k tok) | 3 q × 255 options (8.8k tok) |
|---|---|---|---|
| Clef-Flash | 38 ms | 186 ms | 247 ms |
| Clef | 67 ms | 572 ms | 755 ms |

  Because every option of every question is in one sequence, cost grows with Σ options. 255 options cost about 8.8k tokens of prefill, where Hume found that Jev's latency barely changes from 2 to 200 options.
- Clef's hosted serving was sped up on 10-09 "at the serving infrastructure layer", with no new weights (B2). Cloudflare's threat-intel workflow (fetch + render + classify) took 2.2 s with Clef vs 4.7 s with gpt-oss-120b (`B1:836`).
- **Downstream.** `ArthurZ/clef-flash` repackages flash as the HF `model_type: "jev"` format (`notes/hf-sweep.md:34`). `decision-tune/decisiontune-1.0` (ModernBERT-large) is "distilled from Cloudflare clef-flash soft labels" (`DI` notes).

---

## Licence

- **Apache-2.0** for all three repos (`cardData.license`; the LICENSE file is the standard Apache 2.0 text, 11,544 bytes; `HF:clef/files/LICENSE`). The card says this follows the bases (Qwen3.8-27B, Qwen3.5-9B, Qwen3-Omni-30B-A3B-Instruct, all Apache-2.0).
- No use restrictions beyond Apache-2.0. Hosted use is under Cloudflare's Workers AI terms.

---

## Changelog

| Date | Event | Source |
|---|---|---|
| ~2026-09-15/19 | Cloudflare publishes the DiffusionGemma logprob decision-model experiment ("in the same week that Jev came out") | `B1:989` |
| 09-25 → 09-27 (inferred) | Decision taken "on a Friday evening", model trained over the weekend | `B2:832` |
| 09-30 21:15Z | `Cloudflare/clef` and `clef-flash` repos created | `HF:*/api.json` |
| 10-01 | Launch blog; Workers AI hosting at $0.24 / $0.09 per M; 64k context advertised for both; RL fine-tuning service announced | B1 |
| 10-06 | SGLang PR #42721 opened (closes issue #42409) | PR |
| 10-09 | `clef-omni` released (repo created 04:10Z) with blog B2. Flash hosted price cut to $0.038/M and hosted context cut to 24,576; clef hosted serving moved to SGLang and made faster (no new weights). SGLang PR merged (`bd2d73d`). clef/flash READMEs gain the SGLang section (`lastModified` 10-09 18:46Z) | B2, PR, `HF:*/api.json` |

No new clef/flash weights have been published since release. The PR pins the revisions `clef@2f3de3dd` and `clef-flash@17f0b0ad`.

---

## Notable design choices vs Jev and vs kev

1. **Joint, not isolated, decisions.** This is the defining difference. Jev (behaviourally) and kev (by construction: a block-causal mask, or independent rows on hybrid bases) give each question a view of the state alone. Clef puts all questions in one causal prompt, writes their IDs into it, and adds unmasked cross-field attention in the head. That design suits schemas whose fields are correlated, such as invoice "primary action" plus "exact actions", where Clef beats Jev. It costs Jev's add/remove invariance and opens cross-question prompt injection. The RLCD "fully precise record" reward also optimises record-level, joint correctness.
2. **Hybrid-recurrent backbone with no branching problem.** kev had to abandon its packed mask on Qwen3.5 DeltaNet bases and run one row per question, because recurrent layers cannot honour a block-causal mask (`clones/kev.md` Attention mask). Clef sidesteps the issue by not isolating at all: one causal prefill over the whole request, so the Gated-DeltaNet layers run as a normal LM.
3. **A learned transformer head instead of a pointer.** kev scores `</opt>` hidden states against a `<decide>` token with a bilinear pointer. Clef mean-pools spans and adds a lexical prior from the untied LM-head rows. It runs 2 evidence-routing layers (options → whole prompt), then 4 field layers (fields ↔ fields, fields → prompt), then a cosine-plus-MLP scorer. The head has ~110–128M parameters vs kev's ~2.6M.
4. **No special tokens, no sanitisation.** kev reuses five Qwen FIM/box tokens and rewrites `<|…|>` in user text. Clef uses plain-text scaffolding (`FIELD n`, `ID:`, `OPTION n:`, JSON option objects) inside the Qwen chat template with an empty `<think>` block. User text is not escaped, so option-boundary forgery is possible in principle (inferred; the Jev reference reports Jev's boundaries as unforgeable).
5. **Options sorted alphabetically.** This differs from Jev (which reads request order and leans toward the first option) and from kev (request order, with shuffle augmentation). Score levels keep their order.
6. **Calibration through losses, not temperature.** Label-smoothed CE + Brier + an RLCD-named RL stage, with no post-hoc T (kev ships a fitted T in `head.pt`). Board ECEs (0.025–0.040) are the best of the three profiled families and well below Jev 1.13's 0.078 on the same board.
7. **Confidence is not Jev's in the reference code** (`p_max`). SGLang restores Jev's formulas, and Workers AI is unspecified. Clients that threshold on `confidence` will see different numbers depending on the serving path.
8. **Multimodality first.** The vision towers are kept (clef/flash) and audio+video are added via the Qwen3-Omni thinker (omni). Jev 1.14 only has images behind a danger-zone header; kev is text-only.
9. **Cost profile.** One prefill of state + all questions + all options. Cheap for a few questions over a long state; expensive for many large option sets (64 q → 572 ms on H200 for clef). kev and Jev share the state across questions but keep each question's options in its own branch.
10. **Recipe vs kev.** Both freeze the base and train adapters + head. Clef uses LoRA r256 (kev r16; kev-27B moved to full FT) on synthetic data with Brier/RL objectives, where kev uses hard labels from public data with CE. Clef publishes no data, code or ablations; kev's eval discipline (frozen hashed suites, locked tests) has no counterpart here, and Clef's private-vs-public DI gap is the main warning sign.
