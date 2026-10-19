# Hugging Face sweep: Jev-style decision models, with a focus on big labs (2026-10-19)

Scope: a detailed sweep of the Hugging Face Hub for models that do Jev's job: one forward pass, typed questions
(`noul`, `choice`, `score`), calibrated probabilities and no decoding. The focus is on newer releases from big
companies and labs. Everything here comes from the HF API and the files in each repo (config, README, adapter
config, `*.py` read as text only, never run). No weights were downloaded. The raw JSON is in
`notes/hf-raw/<org>__<model>/` (`api.json` = `/api/models/<id>?expand[]=safetensors,...`, `tree.json`, and
`files/` holding the small text files). Search and listing dumps are in `notes/hf-raw/_search/`,
`_authors/`, `_spaces/` and `_papers/`.

Notation: L = layers, d = hidden size, H/KV = attention heads / KV heads, ctx = `max_position_embeddings`.
"lin/full" is the Qwen3.5 hybrid (Gated DeltaNet linear attention, plus full attention every 4th layer:
`full_attention_interval: 4`). "swa/full" is Gemma-4 sliding window (1024) plus global. "conv/full" is LFM2's
short-conv hybrid. DI is the community **Decision Index** (`multimodalart/jev-decision-index`, data
`data/index.json`, generated 2026-10-16, panel "Decision Index 0.3.2". The full snapshot is in
`hf-raw/_spaces/dix_leaderboard.md`). On that board Jev 1.14.0 scores 61.53 (ECE 0.080) and Jev 1.13 scores 60.11.

## 0. Headline findings

1. **Decision models are now a category in their own right, and several big companies ship open weights.**
   These follow Jev's interface closely (state + typed questions → per-option probabilities, often a
   `/v1/systemone`-compatible HTTP API):
   **Perplexity** (`pplx-decider` v1/v1.1/v1.2/flash-v1.2), **vLLM Semantic Router** (`vllm-sr`, the vLLM project:
   Decision 1.0/2.0, d3, Vela), **Cloudflare** (`clef`, `clef-flash`, `clef-omni`), **Amazon / Strands Agents**
   (`strands-decider-*`, five sizes on Gemma 4 and Qwen3.5, mirrored under `amazon/`), **Liquid AI** (`d1-3B`,
   `d1-omni-600M`), **Shanghai AI Lab / InternLM** (`Intern-Decision-0.8B/2B/4B`), **IBM** (the `decision`
   LoRA/aLoRA in `granitelib-core-r1.0`, composed into `granite-switch-4.1-8b/30b`), **Ant Group / inclusionAI**
   (`LemonDecider`), **H2O.ai** (`h2o-lightning-4b`) and **Together** (`Tev1-*-experimental`).
2. **Meta** published an adjacent system: `facebook/meta-encoder` (MetaEncoder-30B, "Multimodal System One
   Encoder", arXiv 2610.11316). It is a contrastive **bi-encoder** on Meta's `Muse-Glimmer-30B` that scores
   candidates by inner product. The repo is gated (`gated: auto`), so `config.json` could not be fetched
   anonymously. The backbone facts below come from `unsloth/Muse-Glimmer-30B`.
3. **Hugging Face is building a shared format.** `ArthurZ/*` (Arthur Zucker, transformers maintainer) repackages
   d3, pplx-decider, JEV-27B, GEV, clef-flash, kev-4b, Darwin-ZTC and others as `model_type: "jev"`
   (`transformers_version: 5.19.0.dev0`). The config has a `backbone_config`, `num_verbalizers` (255 or 24),
   per-type `offsets`, `question_types: [noul, choice, score]` and `softcap`, and the repos load through a new
   `pipeline("decision")`. This looks like the convergence point for decoder-readout clones.
4. **Specialist label-set families moved into this space:** **Fastino** `GLiNER2.5-Decide` (0.29–1.2B, GLiNER2
   schema heads), **Knowledgator** `gliclass-v4-edge/9b/27b` ("decoder-KV": labels are transient tokens that
   read the backbone KV cache, then a 2-layer DeBERTa label scorer and a calibrator), and **Liquid**
   `LFM2.5-Encoder-350M-Prompt-Router`.
5. **No decision-model release found on HF from** Google, Microsoft, NVIDIA, Alibaba/Qwen (as an org), Apple,
   Salesforce, Cohere, Mistral, AI2, Nous, Jina, Mixedbread, Answer.AI/LightOn, Snowflake, BAAI, DeepSeek,
   Moonshot, Zhipu, Tencent or ByteDance. Their recent releases are embeddings, rerankers or LLMs, many of them the
   **backbones** of the clones (Qwen3.5/3.8, Gemma 4, LFM2.5, ModernBERT/Ettin). Off HF: **Microsoft-Decision-1**
   (Foundry only, Qwen3.5-9B post-train, GA 2026-10-08, no weights), **OpenAI Decisions** (API beta) and hosted
   **Liquid d1** (web sources, see §3).
6. **The architecture has converged.** Almost every strong clone is a Qwen3.5/3.8 hybrid (or Gemma 4) with:
   (a) single-token option codes (`A..Z, AA..`, up to 255) read either from a sliced LM head or from a separate
   `[255, d]` readout; (b) **noncausal full-attention layers** (linear layers stay causal), at Perplexity, vLLM-SR
   d3, LemonDecider and AutoTrust; (c) last-token or per-question marker readout; (d) temperature calibration,
   sometimes per type. The exceptions worth studying are Cloudflare's **joint schema head** (a 4-layer transformer
   over final hidden states), Strands' **pointer head**, vLLM-SR Decision 2.0's **bilinear endpoint head**,
   Liquid d1's **trunk-plus-branch tree attention**, Knowledgator's **decoder-KV label tokens** and Meta's
   **bi-encoder**.
7. **Possible shared code lineage:** pplx-decider-v1.1's `decision_config.json` records `step: 2446`, and its
   source package is `autojev`. LemonDecider's card lists one run of exactly 2,446 steps, and its
   `modeling_lemondecider.py` defaults `decision_protocol` to `'autojev-v1'`. That suggests LemonDecider was built
   on Perplexity's released training/runtime code. This is an inference from the files, not a stated fact.

## 1. Relevance ranking

### 1a. True Jev clones from big companies and labs (the interface matches Jev; ranked by closeness × quality × openness)

| # | org | models (HF) | date | backbone → head | DI 0.3.2 Full (board) | why it matters |
|---:|---|---|---|---|---:|---|
| 1 | Perplexity | `perplexity-ai/pplx-decider-v1-27b`, `-v1.1-27b`, `-v1.2`, `-flash-v1.2` | 10-01 … 10-15 | Qwen3.8-27B / Qwen3.5-4B, **noncausal full-attn**, 255-way readout; v1.2 joint marker readout | 65.55 (v1.2), 54.81 (flash) | closest to Jev, Apache-2.0, code in repo (`autojev`, `pplx_decider`) |
| 2 | vLLM Semantic Router (vLLM project) | `vllm-sr/d3`, `d3-flash/-mini/-nano/-lite/-edge`, `Decision-2.0-*`, `Vela-2.0-0.3B` | 09-21 … 10-10 | d3: Qwen3.8-27B noncausal + 255 code readout. D2.0: LoRA + bilinear endpoint head. Vela: ModernBERT encoder | 66.26 (d3, #2 overall) | best open big-org entry; full family 0.6B–27B |
| 3 | Cloudflare | `Cloudflare/clef`, `clef-flash`, `clef-omni` | 09-30, 10-09 | Qwen3.8-27B / 3.5-9B / Qwen3-Omni-30B-A3B → **joint schema head** (4-layer transformer, 2 routing layers) | 53.08 / 47.61 / 48.28 | distinct head design; audio+video (omni) |
| 4 | Amazon (Strands Agents) | `amazon/strands-decider-{E2B,E4B,12B,26B-A4B}-gemma4-v1-2610`, `-2B-qwen3.5-v1-2610`; `StrandsAgents/strands-decider-2B-hobson-v19/v21` | 10-05 … 10-09 | LoRA r48 on Gemma 4 / Qwen3.5 + **pointer head** (24 slots, dim 768), per-type temperatures | 21.34 (2B hobson-v21); card: 47.14/48.69 DI 0.2.1 (12B/26B) | fully reproducible recipe + data inventory; seed soups |
| 5 | Liquid AI | `LiquidAI/d1-3B`, `d1-omni-600M` | 10-05 | LFM2.5-VL-3B (conv/full) with trunk+branch tree attention; d1-omni: LFM2.5-Encoder-350M + 2-layer decision head | 38.90 / 9.09 | non-transformer hybrid; edge-sized; audio |
| 6 | InternLM (Shanghai AI Lab) | `internlm/Intern-Decision-0.8B/2B/4B` | 09-26 | Qwen3.5 VL, causal, LM-head option symbols at `<decision>` placeholders, all fields in one pass | 38.21 / 22.58 / 11.78 | multi-question JSON-skeleton readout; T=1.99 |
| 7 | IBM | `ibm-granite/granitelib-core-r1.0` (`decision/` LoRA + aLoRA), `granite-switch-4.1-8b/30b` | 10-14/15 | Granite-4.1-8B/30B (dense, full attn) + LoRA r64; aLoRA activates at `### DECISION`, state KV on frozen base | n/a (card: 53.30 public, 30B LoRA) | big-vendor adapter approach; KV reuse across questions |
| 8 | Ant Group (inclusionAI) | `inclusionAI/LemonDecider` | 10-05 | Qwen3.5-4B, noncausal full-attn, FP32 255-way readout, 256K ctx, uncalibrated T=1.0 | n/a (self-rep 50.31 public) | appears to share Perplexity's `autojev` lineage |
| 9 | H2O.ai | `h2oai/h2o-lightning-4b` | 10-02 | Qwen3.5-4B LoRA r64 merged + FP32 pooled readout; "two-way reading"; stock vLLM + shim | not on board (JevBench composite #1) | vLLM-native serving |
| 10 | Together | `togethercomputer/Tev1-4B-experimental` (+0.8B) | 09-23 | Qwen3.5-4B SFT, keeps LM head, 1 letter token, 2–24 options | 34.14 | "train your own Jev for $17" recipe; weakest interface match |

### 1b. Adjacent big-lab architectures worth comparing (not Jev interface, same single-pass scoring job)

| org | model | date | architecture | head / option encoding | relevance |
|---|---|---|---|---|---|
| Meta | `facebook/meta-encoder` (MetaEncoder-30B) | 09-30 | bi-encoder on Muse-Glimmer-30B (52L, d 6656, 32H/2KV, swa 39 / full 13, ctx 131072) | task and candidates encoded separately, inner product; JevBench orig 0.944 (OpenJev 0.986) | **high**: explicitly a "System One" decision encoder |
| Fastino | `fastino/GLiNER2.5-Decide` (+`-1B`, `-multi-`) | 09-23/24 | GLiNER2 schema extractor on DeBERTa-v3-large (24L/1024) / Ettin-enc 1B (ModernBERT 28L/1792) / mDeBERTa-base | label set in-sequence, per-label scoring, multi-head per call | high (DI 9.24, but 119k downloads) |
| Knowledgator | `knowledgator/gliclass-v4-edge`, `-9b`, `-27b` | 10-13/16 | edge: DeBERTa-v3-large uni-encoder; 9b/27b: Qwen3.5-9B / Qwen3.8-27B "decoder-KV" | labels as transient `<<LABEL>>` tokens reading the KV cache, then a 2-layer DeBERTa label scorer, MLP, recurrent refinement and calibrator | high (self-rep DI 55.6 for 9b) |
| Liquid AI | `LiquidAI/LFM2.5-Encoder-350M-Prompt-Router` | 07-13 | `Lfm2BidirForSequenceRouting`, 16L conv/full bidirectional | prompt vs free-text lanes in one pass | medium |
| NVIDIA | `nvidia/Nemotron-3.5-Rerank-8B-BF16` | 10-15 | `Mistral3ForSequenceClassification`, **is_causal false**, avg pooling, Dense 4096→1 | cross-encoder pair score | medium (bidirectionalised decoder) |
| Jina | `jinaai/jina-reranker-v3.5` | 07-14 | Qwen3-0.6B, 3L2G hybrid (swa 16 / full 12) | listwise "last-but-not-late": query + many docs in one pass | medium (listwise ≈ choice) |
| LightOn | `lightonai/LightOn-rerank-PW/LW-{0.8,2,4}B` | 07-08 | Qwen3.5 lin/full | PW: `logit(Yes) − logit(No)` (= noul readout); LW: listwise | medium |
| Tencent | `tencent/R3-rerank-0.6b` | 07-08 | Qwen3-Reranker-0.6B | yes/no logit cross-encoder for skill routing | low/medium |
| Alibaba-NLP | `Alibaba-NLP/core-reranker-2b/8b` | 08-30 | Qwen3-VL | pointwise relevance score | low |
| Amazon | `amazon/poiss-bge-m3-reranker` | 09-18 | XLM-R (bge-reranker-v2-m3) | 1 logit per pair | low |
| Prior Labs | `Prior-Labs/tabpfn_3_5` | 09-09 | tabular ICL transformer | class probabilities in one pass | low (tabular) |
| InternLM | `internlm/AdvancedMathBench-AutoVerifier` | 09-29 | Qwen3.5-MoE 35B-A3B | generative verifier | low |

### 1c. Non-big-org clones that matter for comparison (top of the board / most downloaded)

`startlux-models/StartLux-Decision-v1.1-27B-Instant` (#1, 69.12), `MISA-core/linne-oak-1-0` (Gemma-4-31B, 65.61),
`Mapika/decider-*` (decider-2b has 318k downloads; 0.8B–35B-A3B and 31B), `autotrust/JEV-*` / `GEV-26B-Decide`
(Jev-distilled; JEV-27B-VL has 1.93M downloads), `openjev/openjev`, `bespokelabs/Nimble-v4-max` (self-reported
68.46 public), `OmniJev/OneJev1.5-*`, `tasksource/tasksource-decider-nano` (150M cross-encoder),
`FINAL-Bench/Darwin-27B-ZTC` / `ZTC-Judge-*`. The known clones Kev, Laya and Zeiger are covered in `clones/`.

## 2. Master table: every model fetched in detail (120 repos)

Columns come from `api.json` (created, downloads, cardData.license, safetensors.total) and `config.json` (or `text_config`/`encoder_config`/`backbone_config`). An empty params cell means adapter-only, gated or no safetensors metadata. The verdict is one word plus a hint. CLONE = big org with the Jev interface; clone = community Jev interface; CLONE-ish = label-set single-pass; adjacent = same scoring job, different interface; mirror/base/weak/irrelevant as named.

| model | created | dl | lic | params | architectures / model_type | L | d | H/KV | ctx | layer mix | verdict |
|---|---|---:|---|---:|---|---:|---:|---|---:|---|---|
| Alibaba-NLP/core-reranker-2b | 2026-08-30 | 139 | cc-by-4.0 | 2.13B | Qwen3VLForConditionalGeneration / qwen3_vl | 28 | 2048 | 16/8 | 262144 |  | adjacent (pointwise VL reranker) |
| Alibaba-NLP/core-reranker-8b | 2026-08-30 | 106 | cc-by-4.0 | 8.77B | Qwen3VLForConditionalGeneration / qwen3_vl | 36 | 4096 | 32/8 | 262144 |  | adjacent (pointwise VL reranker) |
| ArthurZ/Darwin-27B-ZTC-v5 | 2026-10-16 | 34 | apache-2.0 | 1.3M | jev | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | mirror (HF `jev` port) |
| ArthurZ/JEV-27B | 2026-10-16 | 20 |  | 0.1M | jev | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | mirror (HF `jev` port) |
| ArthurZ/PhysicalEye-Decide-35B | 2026-10-16 | 28 | apache-2.0 | 0.5M | jev |  | 2048 | / |  |  | mirror (HF `jev` port) |
| ArthurZ/d3 | 2026-10-16 | 42 | apache-2.0 | 1.3M | jev |  | 5120 | / |  |  | mirror (HF `jev` port) |
| Cloudflare/clef | 2026-09-30 | 21421 | apache-2.0 | 27.36B | Qwen3_5ForConditionalGeneration / qwen3_5 | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | CLONE (big co.) |
| Cloudflare/clef-flash | 2026-09-30 | 35576 | apache-2.0 | 9.41B | Qwen3_5ForConditionalGeneration / qwen3_5 | 32 | 4096 | 16/4 | 262144 | lin:24,full:8 | CLONE (big co.) |
| Cloudflare/clef-omni | 2026-10-09 | 4856 | apache-2.0 | 35.26B | Qwen3OmniMoeForConditionalGeneration / qwen3_omni_moe |  |  | / |  |  | CLONE (big co.) |
| Contrastive-LM/CLM-v0.1-8B | 2026-09-21 | 4989 | apache-2.0 |  | clm |  |  | / |  |  | clone-ish (bi-encoder) |
| FINAL-Bench/ZTC-Judge-27B | 2026-09-19 | 534 | apache-2.0 | 27.78B | Qwen3_5ForConditionalGeneration / qwen3_5 | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | clone (judge) |
| FINAL-Bench/ZTC-Judge-9B | 2026-09-20 | 2592 | apache-2.0 | 9.65B | Qwen3_5ForConditionalGeneration / qwen3_5 | 32 | 4096 | 16/4 | 262144 | lin:24,full:8 | clone (judge) |
| HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B | 2026-09-26 | 830 | apache-2.0 | 0.60B | Qwen3Model / qwen3 | 28 | 1024 | 16/8 | 32768 |  | adjacent (embedding) |
| Horizon-Labs/multilingual-zeroshot-base | 2026-09-25 | 783 | apache-2.0 | 0.31B | ModernBertForSequenceClassification / modernbert | 22 | 768 | 12/ | 8192 | full:8,swa:14 | adjacent (NLI zero-shot) |
| JackKozmo29/mailguard-jev-style-1.5b | 2026-09-29 | 59572 | apache-2.0 | 1.54B | Qwen2ForCausalLM / qwen2 | 28 | 1536 | 12/2 | 32768 | full:28 | weak (generative classifier) |
| LIQIIIII/Jeva-4B | 2026-10-09 | 1327 | apache-2.0 |  |  |  |  | / |  |  | clone |
| LiquidAI/LFM2.5-Encoder-350M | 2026-07-27 | 18774 | other | 0.35B | Lfm2BidirectionalForMaskedLM / lfm2 | 16 | 1024 | 16/8 | 128000 | conv:10,full:6 | base (d1-omni trunk) |
| LiquidAI/LFM2.5-Encoder-350M-Prompt-Router | 2026-07-13 | 1309 | other | 0.36B | Lfm2BidirForSequenceRouting / lfm2 | 16 | 1024 | 16/8 | 128000 | conv:10,full:6 | adjacent (zero-shot router head) |
| LiquidAI/d1-3B | 2026-10-05 | 22004 | other | 3.12B | Lfm2VlForConditionalGeneration / lfm2_vl | 30 | 2048 | 32/8 | 32768 | conv:22,full:8 | CLONE (big co.) |
| LiquidAI/d1-omni-600M | 2026-10-05 | 14419 | other | 0.59B | D1OmniModel / d1_omni | 16 | 1024 | 16/8 | 128000 | conv:10,full:6 | CLONE (big co.) |
| PNPM244/linne-oak-1-0 | 2026-10-11 | 167 | apache-2.0 | 31.27B | Gemma4ForConditionalGeneration / gemma4 | 60 | 5376 | 32/16 | 262144 | swa:50,full:10 | clone (top-3 DI) |
| Maincode/matilda-jev-v1.5 | 2026-10-09 | 137 | apache-2.0 | 26.09B | MatildaJevModel / matilda_jev | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | clone |
| Mapika/decider-0.8b | 2026-09-19 | 16150 | apache-2.0 | 0.75B | Qwen3_5ForCausalLM / qwen3_5_text | 24 | 1024 | 8/2 | 262144 | lin:18,full:6 | clone |
| Mapika/decider-12b | 2026-09-29 | 713 | apache-2.0 | 11.96B | Gemma4UnifiedForConditionalGeneration / gemma4_unified | 48 | 3840 | 16/8 | 262144 | swa:40,full:8 | clone |
| Mapika/decider-2b | 2026-09-16 | 318711 | apache-2.0 | 1.88B | Qwen3_5ForCausalLM / qwen3_5_text | 24 | 2048 | 8/2 | 262144 | lin:18,full:6 | clone (most dl small) |
| Mapika/decider-2b-vision | 2026-09-16 | 4591 | apache-2.0 | 2.21B | Qwen3_5ForConditionalGeneration / qwen3_5 | 24 | 2048 | 8/2 | 262144 | lin:18,full:6 | clone |
| Mapika/decider-35b-a3b | 2026-09-20 | 2658 | apache-2.0 | 34.66B | Qwen3_5MoeForCausalLM / qwen3_5_moe_text | 40 | 2048 | 16/2 | 262144 | lin:30,full:10 | clone |
| Mapika/decider-4b | 2026-09-22 | 29745 | apache-2.0 | 4.21B | Qwen3_5ForCausalLM / qwen3_5_text | 32 | 2560 | 16/4 | 262144 | lin:24,full:8 | clone |
| Mapika/decider-chat-gemma4-31b | 2026-09-29 | 643 | apache-2.0 | 31.27B | Gemma4ForConditionalGeneration / gemma4 | 60 | 5376 | 32/16 | 262144 | swa:50,full:10 | clone (no-train readout) |
| OmniJev/OneJev1.5-27B | 2026-10-12 | 116 | apache-2.0 | 27.36B | Qwen3_5ForConditionalGeneration / qwen3_5 | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | clone |
| Prior-Labs/tabpfn_3_5 | 2026-09-09 | 11716 | other |  |  |  |  | / |  |  | adjacent (tabular ICL) |
| StrandsAgents/strands-decider-2B-hobson-v21 | 2026-10-05 | 0 | apache-2.0 |  |  |  |  | / |  |  | CLONE (Amazon) |
| TokenRhythm/NeoHorse-Jev-4B | 2026-09-23 | 2898 | apache-2.0 |  |  |  |  | / |  |  | clone |
| WhaletechAI/W1-JEV | 2026-09-25 | 4187 |  |  |  |  |  | / |  |  | clone (unclear) |
| akhilaaa3/Jev-Omni | 2026-09-20 | 5379 | apache-2.0 | 11.96B | Gemma4UnifiedForConditionalGeneration / gemma4_unified | 48 | 3840 | 16/8 | 262144 | swa:40,full:8 | clone |
| alibiserikbay/JevK5 | 2026-09-22 | 12868 | apache-2.0 | 4.21B | Qwen3_5ForCausalLM / qwen3_5_text | 32 | 2560 | 16/4 | 262144 | lin:24,full:8 | clone |
| amazon/poiss-bge-m3-reranker | 2026-09-18 | 47 | apache-2.0 | 0.57B | XLMRobertaForSequenceClassification / xlm-roberta | 24 | 1024 | 16/ | 8194 |  | adjacent (cross-encoder) |
| amazon/strands-decider-12B-gemma4-v1-2610 | 2026-10-09 | 0 | apache-2.0 |  |  |  |  | / |  |  | CLONE (Amazon) |
| amazon/strands-decider-26B-A4B-gemma4-v1-2610 | 2026-10-09 | 0 | apache-2.0 |  |  |  |  | / |  |  | CLONE (Amazon) |
| amazon/strands-decider-2B-qwen3.5-v1-2610 | 2026-10-09 | 0 | apache-2.0 |  |  |  |  | / |  |  | CLONE (Amazon) |
| amazon/strands-decider-E2B-gemma4-v1-2610 | 2026-10-09 | 0 | apache-2.0 |  |  |  |  | / |  |  | CLONE (Amazon) |
| amazon/strands-decider-E4B-gemma4-v1-2610 | 2026-10-09 | 0 | apache-2.0 |  |  |  |  | / |  |  | CLONE (Amazon) |
| autotrust/AutoTrust-JEV-27B-v1.1 | 2026-10-17 | 5615 | apache-2.0 | 27.36B | Qwen3_5ForConditionalGeneration / qwen3_5 | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | clone (Jev distill) |
| autotrust/GEV-26B-Decide | 2026-10-02 | 1119803 | apache-2.0 | 25.81B | Gemma4ForConditionalGeneration / gemma4 | 30 | 2816 | 16/8 | 262144 | swa:25,full:5 | clone (S1+S2) |
| autotrust/JEV-27B | 2026-09-25 | 239683 | apache-2.0 | 26.90B | Qwen3_5ForCausalLM / qwen3_5_text | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | clone (Jev distill) |
| autotrust/JEV-27B-VL | 2026-09-30 | 1929553 | apache-2.0 | 27.78B | Qwen3_5ForConditionalGeneration / qwen3_5 | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | clone (Jev distill, most dl) |
| autotrust/JEV-35B | 2026-10-11 | 454857 | apache-2.0 | 35.95B | Qwen3_5MoeForConditionalGeneration / qwen3_5_moe | 40 | 2048 | 16/2 | 262144 | lin:30,full:10 | clone |
| autotrust/JEV-9B | 2026-09-23 | 454435 | apache-2.0 | 8.95B | Qwen3_5ForCausalLM / qwen3_5_text | 32 | 4096 | 16/4 | 262144 | lin:24,full:8 | clone (Jev distill) |
| autotrust/JEV-Gemma4-26B-A4B | 2026-09-29 | 33900 | apache-2.0 | 25.81B | Gemma4ForConditionalGeneration / gemma4 | 30 | 2816 | 16/8 | 262144 | swa:25,full:5 | clone |
| bespokelabs/Bespoke-Nimble-9B-v3 | 2026-09-30 | 555 | cc-by-nc-4.0 |  |  |  |  | / |  |  | clone |
| bespokelabs/Nimble-v4-max | 2026-10-18 | 15 | cc-by-nc-4.0 | 27.36B | Qwen3_5ForConditionalGeneration / qwen3_5 | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | clone (self-rep 68.5 public) |
| chaoliangUNSW/Jev-Style-0.8B-Decision-v3 | 2026-09-24 | 1217 | apache-2.0 | 0.75B | Qwen3_5ForCausalLM / qwen3_5_text | 24 | 1024 | 8/2 | 262144 | lin:18,full:6 | clone |
| chaoliangUNSW/Jev-Style-2B-Decision-v3 | 2026-09-27 | 1381 | apache-2.0 | 1.88B | Qwen3_5ForCausalLM / qwen3_5_text | 24 | 2048 | 8/2 | 262144 | lin:18,full:6 | clone |
| com-kotobalabs/open-jev-deberta-v3-large | 2026-09-18 | 3579 | apache-2.0 | 0.43B | DebertaV2Model / deberta-v2 | 24 | 1024 | 16/ | 512 |  | clone (encoder) |
| convaiinnovations/laya-typed-decisions | 2026-09-18 | 3166 | apache-2.0 | 0.42B | LayaTypedDecisions / laya |  |  | / |  |  | clone (Laya, known) |
| decision-tune/decisiontune-1.0 | 2026-10-04 | 263 | apache-2.0 | 0.39B | ModernBertForMaskedLM / modernbert | 28 | 1024 | 16/ | 8192 | full:10,swa:18 | clone (encoder) |
| facebook/meta-encoder | 2026-09-30 | 130 | apache-2.0 | 29.78B |  |  |  | / |  |  | ADJACENT-STRONG (Meta, S1 bi-encoder) |
| fastino/GLiNER2.5-Decide | 2026-09-23 | 119592 | apache-2.0 | 0.49B | Gliner2ForSchemaExtraction / gliner2 | 24 | 1024 | 16/ | 512 |  | CLONE-ish (GLi label-set) |
| fastino/GLiNER2.5-Decide-1B | 2026-09-24 | 2773 | apache-2.0 | 1.19B | Gliner2ForSchemaExtraction / gliner2 | 28 | 1792 | 28/ | 7999 | full:10,swa:18 | CLONE-ish (GLi label-set) |
| fastino/GLiNER2.5-multi-Decide | 2026-09-24 | 19223 | apache-2.0 | 0.29B | Gliner2ForSchemaExtraction / gliner2 | 12 | 768 | 12/ | 512 |  | CLONE-ish (GLi label-set) |
| google/embeddinggemma-2 | 2026-09-14 | 167750 | apache-2.0 | 0.74B | EmbeddingGemma2Model / embedding_gemma2 | 24 | 512 | 4/2 | 262144 | swa:20,full:4 | base/adjacent (embedding) |
| gyanbhambhani/oss-jev-qwen05-decision-noul-bal-v3 | 2026-10-17 | 326 | apache-2.0 | 0.49B | Qwen2ForCausalLM / qwen2 | 24 | 896 | 14/2 | 32768 | full:24 | weak |
| h2oai/h2o-lightning-4b | 2026-10-02 | 4944 | apache-2.0 | 4.66B | Qwen3_5ForConditionalGeneration / qwen3_5 | 32 | 2560 | 16/4 | 262144 | lin:24,full:8 | CLONE (H2O) |
| hotchpotch/bekko-system-one-v0-400m | 2026-09-29 | 0 |  |  |  |  |  | / |  |  | clone (encoder) |
| iapp/OpenThai-SystemOne | 2026-09-20 | 11987 | apache-2.0 | 0.75B | OpenThaiSystemOneForDecision / openthai_systemone | 24 | 1024 | 8/2 | 262144 | lin:18,full:6 | clone |
| ibm-granite/granite-4.1-8b | 2026-04-06 | 94986 | apache-2.0 | 8.79B | GraniteForCausalLM / granite | 40 | 4096 | 32/8 | 131072 |  | base |
| ibm-granite/granite-switch-4.1-30b | 2026-10-14 | 768 | apache-2.0 | 29.99B | GraniteSwitchForCausalLM / granite_switch | 66 | 4096 | 32/8 | 131072 | full:66 | CLONE (IBM, adapter in switch) |
| ibm-granite/granite-switch-4.1-8b | 2026-10-14 | 980 | apache-2.0 | 8.78B | GraniteSwitchForCausalLM / granite_switch | 42 | 4096 | 32/8 | 131072 | full:42 | CLONE (IBM, adapter in switch) |
| ibm-granite/granitelib-core-r1.0 | 2025-12-10 | 3687 | apache-2.0 |  |  |  |  | / |  |  | CLONE (IBM decision LoRA/aLoRA) |
| inclusionAI/LemonDecider | 2026-10-05 | 343 | apache-2.0 | 4.54B | LemonDeciderForDecisionClassification / lemondecider_decision | 32 | 2560 | 16/4 | 262144 | lin:24,full:8 | CLONE (Ant Group) |
| internlm/AdvancedMathBench-AutoVerifier | 2026-09-29 | 239 |  | 35.96B | Qwen3_5MoeForConditionalGeneration / qwen3_5_moe | 40 | 2048 | 16/2 | 262144 | lin:30,full:10 | adjacent (generative verifier) |
| internlm/Intern-Decision-0.8B | 2026-09-26 | 3012 | apache-2.0 | 0.85B | Qwen3_5ForConditionalGeneration / qwen3_5 | 24 | 1024 | 8/2 | 262144 | lin:18,full:6 | CLONE (Shanghai AI Lab) |
| internlm/Intern-Decision-2B | 2026-09-26 | 2226 | apache-2.0 | 2.21B | Qwen3_5ForConditionalGeneration / qwen3_5 | 24 | 2048 | 8/2 | 262144 | lin:18,full:6 | CLONE (Shanghai AI Lab) |
| internlm/Intern-Decision-4B | 2026-09-26 | 4458 | apache-2.0 | 4.54B | Qwen3_5ForConditionalGeneration / qwen3_5 | 32 | 2560 | 16/4 | 262144 | lin:24,full:8 | CLONE (Shanghai AI Lab) |
| itsdevice/modernbert-systemone | 2026-10-14 | 26 |  | 0.15B | ModernBertModel / modernbert | 22 | 768 | 12/ | 8192 | full:8,swa:14 | clone (encoder) |
| jaredpalmer/kev-27b | 2026-09-24 | 2732 | apache-2.0 | 25.62B | Qwen3_5TextModel / qwen3_5_text | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | clone (Kev, known) |
| jaredpalmer/kev-4b | 2026-09-19 | 26121 | apache-2.0 |  |  |  |  | / |  |  | clone (Kev, known) |
| jinaai/jina-reranker-v3.5 | 2026-07-14 | 74872 | cc-by-nc-4.0 | 0.60B | JinaForRanking / qwen3 | 28 | 1024 | 16/8 | 131072 | swa:16,full:12 | adjacent (listwise reranker) |
| juspay/Xor-26B-A4B | 2026-10-01 | 944 | apache-2.0 | 25.81B | Gemma4ForConditionalGeneration / gemma4 | 30 | 2816 | 16/8 | 262144 | swa:25,full:5 | clone |
| knowledgator/gliclass-v4-27b | 2026-10-13 | 48 | apache-2.0 | 27.00B | GLiClassModel / GLiClass |  | 5120 | / |  |  | CLONE-ish (GLiClass decoder-KV) |
| knowledgator/gliclass-v4-9b | 2026-10-13 | 114 | apache-2.0 | 8.98B | GLiClassModel / GLiClass |  | 4096 | / |  |  | CLONE-ish (GLiClass decoder-KV) |
| knowledgator/gliclass-v4-edge | 2026-10-16 | 274 | apache-2.0 | 0.44B | GLiClassModel / GLiClass | 24 | 1024 | 16/ | 512 |  | CLONE-ish (GLiClass encoder) |
| lightonai/LightOn-rerank-LW-2B | 2026-07-08 | 529 | apache-2.0 | 2.21B | Qwen3_5ForConditionalGeneration / qwen3_5 | 24 | 2048 | 8/2 | 262144 | lin:18,full:6 | adjacent (listwise reranker) |
| lightonai/LightOn-rerank-PW-2B | 2026-07-08 | 383 | apache-2.0 | 2.21B | Qwen3_5ForConditionalGeneration / qwen3_5 | 24 | 2048 | 8/2 | 262144 | lin:18,full:6 | adjacent (yes/no logit reranker) |
| mistralai/LIDstral-Arabic | 2026-10-08 | 154 | apache-2.0 |  |  |  |  | / |  |  | irrelevant (fastText LID) |
| mrm8488/embeddinggemma-2-ft-typed-decision | 2026-10-11 | 1172 | apache-2.0 | 0.27B | GemmaDecisionModel / gemma_decision |  |  | / |  |  | clone (encoder) |
| nace-ai/drex-v2.0 | 2026-10-06 | 151 | other | 35.11B | Qwen3_5MoeForConditionalGeneration / qwen3_5_moe | 40 | 2048 | 16/2 | 262144 | lin:30,full:10 | clone |
| nvidia/Ising-Calibration-1.5-31B-BF16 | 2026-07-13 | 4035 | other | 31.27B | Gemma4ForConditionalGeneration / gemma4 | 60 | 5376 | 32/16 | 262144 | swa:50,full:10 | irrelevant (quantum plots VLM) |
| nvidia/Nemotron-3.5-Rerank-8B-BF16 | 2026-10-15 | 1025 | other | 8.38B | Mistral3ForSequenceClassification / mistral3 | 34 | 4096 | 32/8 | 262144 |  | adjacent (bidirectional cross-encoder) |
| onnx-community/Decision-2.0-Sol-2B-ONNX | 2026-10-03 | 51 | apache-2.0 |  | Decision2Model / qwen3_5_text | 24 | 2048 | / |  |  | mirror |
| onnx-community/strands-decider-2B-hobson-v19-ONNX | 2026-10-01 | 518 | apache-2.0 |  | StrandsDeciderModel / qwen3_5_text | 24 | 2048 | / |  | lin:18,full:6 | mirror |
| openjev/OpenJev-Flash-9B | 2026-10-07 | 249 | cc-by-nc-4.0 | 9.41B | Qwen3_5ForConditionalGeneration / qwen3_5 | 32 | 4096 | 16/4 | 262144 | lin:24,full:8 | clone |
| openjev/openjev | 2026-09-20 | 6872 | cc-by-nc-4.0 | 27.36B | Qwen3_5ForConditionalGeneration / qwen3_5 | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | clone |
| perplexity-ai/pplx-decider-flash-v1.2 | 2026-10-15 | 608 | apache-2.0 | 4.54B | Qwen3_5Model / qwen3_5 | 32 | 2560 | 16/4 | 262144 | lin:24,full:8 | CLONE (Perplexity) |
| perplexity-ai/pplx-decider-v1-27b | 2026-10-01 | 3058 | apache-2.0 | 26.09B | Qwen3_5Model / qwen3_5 | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | CLONE (Perplexity) |
| perplexity-ai/pplx-decider-v1.1-27b | 2026-10-05 | 4322 | apache-2.0 | 26.09B | Qwen3_5Model / qwen3_5 | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | CLONE (Perplexity) |
| perplexity-ai/pplx-decider-v1.2 | 2026-10-15 | 387 | apache-2.0 | 26.09B | Qwen3_5Model / qwen3_5 | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | CLONE (Perplexity) |
| php-ai/zeiger-0.6b | 2026-09-21 | 33 | cc-by-sa-4.0 | 0.61B |  |  |  | / |  |  | clone (ours, known) |
| pngwn/system-one-qwen3.5-4b-scorer | 2026-09-16 | 787 | cc-by-nc-4.0 |  |  |  |  | / |  |  | clone (early) |
| samatv256/mini-Jev | 2026-09-21 | 1717 | apache-2.0 | 0.3M |  |  |  | / |  |  | weak |
| scairesearch/kieve-jev | 2026-10-19 | 0 | apache-2.0 |  |  |  |  | / |  |  | empty/stub |
| shibing624/train-jev-27b-lora | 2026-10-16 | 60 | apache-2.0 |  |  |  |  | / |  |  | clone (LoRA) |
| shisa-ai/shisa-de-2 | 2026-10-07 | 1336 | apache-2.0 | 25.81B | Gemma4ForConditionalGeneration / gemma4 | 30 | 2816 | 16/8 | 262144 | swa:25,full:5 | clone |
| social-jev/social-jev-0.6B-preview | 2026-10-15 | 83 |  | 0.60B | SocialJevModel / social_jev | 28 | 1024 | 16/8 | 40960 | full:28 | clone |
| social-jev/social-jev-4B-preview | 2026-10-15 | 48 |  | 4.02B | SocialJevModel / social_jev | 36 | 2560 | 32/8 | 40960 | full:36 | clone |
| startlux-models/StartLux-Decision-v1.1-27B-Instant | 2026-10-11 | 359 | cc-by-nc-4.0 | 27.78B | Qwen3_5ForConditionalGeneration / qwen3_5 | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | clone (#1 DI) |
| sthanika-ai/Sieve-27B | 2026-10-07 | 173 | apache-2.0 |  |  |  |  | / |  |  | clone |
| systemonedev/kenning-large-v0.6 | 2026-10-08 | 64 | apache-2.0 | 0.40B | ModernBertForSequenceClassification / modernbert | 28 | 1024 | 16/ | 2048 | full:10,swa:18 | clone (encoder) |
| systemonedev/kenning-xl-v0.6 | 2026-10-06 | 375 | apache-2.0 | 1.72B | Qwen3ForCausalLM / qwen3 | 28 | 2048 | 16/8 | 40960 | full:28 | clone |
| tasksource/tasksource-decider-nano | 2026-09-28 | 445 | apache-2.0 | 0.15B | ModernBertModel / modernbert | 22 | 768 | 12/ | 7999 | full:8,swa:14 | clone (cross-encoder) |
| tencent/ContextPilot-8B | 2026-08-27 | 767 | other | 8.19B | Qwen3ForCausalLM / qwen3 | 36 | 4096 | 32/8 | 40960 | full:36 | irrelevant (generative agent) |
| tencent/R3-rerank-0.6b | 2026-07-08 | 1430 | apache-2.0 | 0.60B | Qwen3ForCausalLM / qwen3 | 28 | 1024 | 16/8 | 40960 | full:28 | adjacent (yes/no logit reranker) |
| togethercomputer/Tev1-4B-experimental | 2026-09-23 | 5204 |  | 4.66B | Qwen3_5ForConditionalGeneration / qwen3_5 | 32 | 2560 | 16/4 | 262144 | lin:24,full:8 | clone-ish (Together, LM head, 1 token) |
| unsloth/Muse-Glimmer-30B | 2026-08-10 | 6059 | apache-2.0 | 29.78B | MuseGlimmerForConditionalGeneration / muse_glimmer | 52 | 6656 | 32/2 | 131072 | swa:39,full:13 | base (MetaEncoder backbone) |
| vllm-sr/Decision-2.0-Nox-4B | 2026-09-29 | 2490 | apache-2.0 |  | Decision2Model / decision2 |  |  | / |  |  | CLONE (vLLM-SR) |
| vllm-sr/Decision-2.0-Vega-27B | 2026-09-29 | 2997 | apache-2.0 |  | Decision2Model / decision2 |  |  | / |  |  | CLONE (vLLM-SR) |
| vllm-sr/Vela-2.0-0.3B | 2026-09-29 | 3650 | apache-2.0 | 0.31B | Vela2Model / vela2-unified | 22 | 768 | 12/ | 32768 |  | CLONE (vLLM-SR encoder) |
| vllm-sr/d3 | 2026-10-10 | 4852 | apache-2.0 | 26.09B | Qwen3_5Model / qwen3_5 | 64 | 5120 | 24/4 | 262144 | lin:48,full:16 | CLONE (vLLM-SR) |
| vllm-sr/d3-flash | 2026-10-10 | 4426 | apache-2.0 | 8.39B | Qwen3_5Model / qwen3_5 | 32 | 4096 | 16/4 | 262144 | lin:24,full:8 | CLONE (vLLM-SR) |
| vllm-sr/d3-mini | 2026-10-10 | 33131 | apache-2.0 | 4.54B | Qwen3_5Model / qwen3_5 | 32 | 2560 | 16/4 | 262144 | lin:24,full:8 | CLONE (vLLM-SR) |

## 2b. Big-lab orgs screened by listing only (releases since 2026-07, author listing sorted by createdAt)

| org | recent releases seen (Jul–Oct 2026) | verdict |
|---|---|---|
| Qwen | Qwen3.8-27B(-FP8), Qwen3.8-Flash-Next, Qwen3.8-2.4T-A95B, Qwen-Drive-1.0-4B, Qwen-Image-2.1* | no decision model; Qwen3.5/3.8 is the dominant **backbone** of the clones |
| Alibaba-NLP | core-emb-2b/8b, core-reranker-2b/8b, UEmbed-2B/4B/9B | adjacent (reranker fetched) |
| google | embeddinggemma-2, gnm-v3, timesfm-3.0, tipsv1-* | no; embeddinggemma-2 is the base of `mrm8488/embeddinggemma-2-ft-typed-decision` |
| facebook / meta-models | meta-encoder, MoEViE-* | meta-encoder = adjacent-strong |
| meta-llama | none since Aug | no |
| microsoft | bitnet-embedding-0.6b/270m, UserRepGen-4B, AesCode, FrogNano-4B, VibeVoice-ASR | no (Microsoft-Decision-1 is Foundry-only) |
| nvidia | Nemotron-3.5-Rerank-8B, Nemotron-3/3.5-Embed-1B/8B, Ising-Calibration-1.5-31B, Muse-Glimmer-30B-NVFP4 | reranker = adjacent; "Ising-Calibration" = quantum plots, irrelevant |
| ibm-granite | granite-switch-4.1-8b/30b, granite-4.2-*, granite-speech-5.0, granitelib-core-r1.0 (decision) | CLONE (adapter) |
| ibm-research | qaoa_angles.* | no |
| apple | LensVLM-9B | no |
| amazon | strands-decider-* (5), poiss-bge-m3-reranker/retriever | CLONE |
| Salesforce, NousResearch, Snowflake, mixedbread-ai, moonshotai, HuggingFaceTB, sentence-transformers, cross-encoder, MoritzLaurer, urchade, Tencent-Hunyuan, ByteDance-Seed, baidu, openai, LGAI-EXAONE, ServiceNow-AI, nomic-ai, zeroentropy, PleIAs, meituan-longcat, OpenGVLab, sarvamai, infly, intfloat, Unbabel, ai21labs, RLHFlow, ContextualAI | nothing public since Aug 2026 | no |
| CohereLabs | tiny-aya-*, North-Small-Translate, North-Micro-Vision | no |
| mistralai | Timestral-*, LIDstral-Arabic (fastText + isotonic calibration), Voxtral | no |
| allenai | Bolmo, Bwen-8B, AstaBrief, ACE climate | no |
| jinaai | jina-reranker-v3.5 (+GGUF/MLX), jina-ocr-v1 | adjacent |
| answerdotai / lightonai | colbert-muvera-*, LateOn, LightOn-rerank-PW/LW-0.8/2/4B, mDenseOn | adjacent |
| BAAI | AREX-2, Recon2Reason-4B, GaussianMind | no |
| deepseek-ai | DeepSeek-V4.1-Flash, V4-Pro | no |
| zai-org | GLM-5.3(-Flash) | no |
| tencent | WeMM-Embedding-2B/4B/9B, R3-rerank/embedding-0.6b, EVIE, ContextPilot, Hy4-preview, UI-Mate | adjacent (reranker) only |
| ByteDance | Bernini-Diffusers-v2 | no |
| XiaomiMiMo | MiMo-V2.6-* (Distill-Qwen-9B is the base of `nace-ai/drex-v1.5`) | no |
| stepfun-ai | Step-5-Preview | no |
| internlm | Intern-Decision-0.8B/2B/4B, AdvancedMathBench-AutoVerifier | CLONE |
| inclusionAI | LemonDecider, SingGuard-NSFA-* (generative VLM guards) | CLONE (LemonDecider) |
| openbmb | Trident-Qwen3VL-2B, UltraData-Code-L2-Classifier, MathForm-8B | no |
| LiquidAI | d1-3B, d1-omni-600M (not in the public author listing; found via Space/board), LFM2.5-Encoder-230M/350M (+Prompt-Router, Policy-Linter, Spellchecker, Diffusion) | CLONE + adjacent |
| knowledgator | gliclass-v4-edge/9b/27b, gliformer-*, gliner-stream-pii | label-set clone-ish |
| fastino | GLiNER2.5-Decide(-1B, -multi-), gliner2.5-base/small/multi-v1 | label-set clone-ish |
| Prior-Labs | tabpfn_3_5 | adjacent (tabular) |
| perplexity-ai | pplx-decider-*, pplx-embed-v2-*, pplx-computer-qwen-3-8-27b-dflash2 | CLONE |
| Cloudflare | clef, clef-flash, clef-omni | CLONE |
| h2oai | h2o-lightning-4b | CLONE |
| togethercomputer | Tev1-4B/0.8B-experimental | weak clone |
| vllm-sr | Decision-1.0 (Kai/Lex/Sol/Nox/Lux/Eos), Decision-2.0 (Kai/Eos/Sol/Nox/Lux/Vega), d3 (edge/lite/nano/mini/flash/27B), Vela-1.0-Encoder-307M-* (Guard/PII/Halu/FactCheck/Reranker/...), Vela-2.0-0.3B/0.8B/4B/9B | CLONE family |
| bespokelabs | Bespoke-Nimble-9B (v1–v3), Bespoke-Nimble-27B-v3, Nimble-v4-max | clone (startup) |
| HIT-TMG | JevEmbed-Qwen3-Embedding-0.6B/4B, JevEmbed-KaLM-V2.5 | adjacent (embedding) |

Not on HF (web, for completeness): **Microsoft-Decision-1** (Foundry, Qwen3.5-9B post-train, GA 2026-10-08, no
weights or figures), **OpenAI Decisions** (API beta, gpt-6-luna), hosted **Liquid d1**, **Nokia AnyJev**
(github.com/nokia-applied-research/AnyJev: L0 option-logit readout with debiasing on any LLM; tech report
arXiv 2610.00831). Sources: redreamality.com/blog/open-decision-models-2026-survey-benchmarks,
orcarouter.ai/blog/microsoft-decision-1-ga-foundry-no-published-benchmarks, alphasignal.ai (pplx-decider).

## 3. Architecture cards (relevant models; config fields are cited as `file:key`)

Unless noted, all Qwen3.5/3.8 backbones share `head_dim 256`, `vocab_size 248320`, `max_position_embeddings
262144`, `rope_parameters {rope_theta 1e7, partial_rotary_factor 0.25, mrope_interleaved, mrope_section
[11,11,10]}` and `full_attention_interval 4`. Shapes: 27B = 64L, d 5120, 24H/4KV, lin 48 / full 16. 9B = 32L,
d 4096, 16H/4KV, 24/8. 4B = 32L, d 2560, 16H/4KV, 24/8. 2B = 24L, d 2048, 8H/2KV, 18/6. 0.8B = 24L, d 1024,
8H/2KV, 18/6.

### 3.1 Perplexity: pplx-decider (CLONE, decoder, hybrid, noncausal)
- Repos: `perplexity-ai/pplx-decider-v1-27b` (created 2026-10-01, 3,058 dl / 126 likes), `-v1.1-27b` (10-05,
  4,322 / 156), `-v1.2` (10-15, 387 / 27), `-flash-v1.2` (10-15, 608 / 27). Apache-2.0. Not gated, although the
  card text says "private repository". Quantised mirrors exist (`mlx-community`, `Livesport`, `turintech`).
- Params (safetensors API): 26,085,330,160 BF16 for the 27B models, 4,539,265,536 for flash.
  `config.json:architectures = ["Qwen3_5Model"]` (no LM head), `model_type qwen3_5`. 27B: 64L, d 5120, 24H/4KV.
  Flash: 32L, d 2560, 16H/4KV.
- Attention: Qwen3.5 hybrid, 48 linear + 16 full (`layer_types`). The full-attention layers run **noncausal**
  (`decision_config.json:attention_mode = "noncausal_full_attention"` from flash-v1.2 on; the v1.1 README says
  "lifting of the causal mask"). The linear layers keep their native (causal) behaviour.
- Readout: a separate `readout.safetensors`, BF16 `[255, 5120]` (v1.1 card), with rows mapped to vocabulary ids
  in `decision_config.json:token_ids`. Option codes are `A…Z, AA…` (`decision_config.json:codes`, 255 entries).
  `pooling: "last"`. v1/v1.1 answer **one question per pass** (last token). v1.2 and flash use
  `joint_config.json {template "joint-question-inline-readout-v1", marker "<|fim_suffix|>", version 3}`: the
  state followed by all questions in one sequence, with one readout at the marker after each question's options.
  Images are supported, max 8,192 tokens per request (v1.1).
- Calibration: one scalar temperature (`decision_config.json:temperature`: 1.382 for v1.2, 1.554 for flash).
  ECE on DI 0.3.2 is 0.0551 (v1.2) and 0.0523 (flash).
- Training: v1 was built on Bespoke-Nimble converters of human-labelled sets; v1.1 adds tasksource; flash is
  **distilled from the 27B models' full distributions** and model-souped.
- Scores: v1.1 card says DI 61.56 vs Jev 57.9. Board: v1.2 65.55 (#4), flash 54.81.

### 3.2 vLLM Semantic Router: d3 / Decision 2.0 / Vela 2.0 (CLONE family)
- **d3** (`vllm-sr/d3`, 10-10, 4,852 dl / 249 likes, Apache-2.0; `d3-flash` 9B, `d3-mini` 4B with 33k dl, `-nano`
  2B, `-lite` 0.8B, `-edge` 0.6B). Params: 26.09B (d3), 8.39B (flash), 4.54B (mini). `config.json:architectures
  ["Qwen3_5Model"]`. `decision_config.json {format_id "d3-code-readout-v1", attention_mode
  "noncausal_full_attention", pooling "last", readout_dtype float32, temperature 1.0}`. That is the same
  layout as Perplexity (255 code readout; HF port `ArthurZ/d3:num_verbalizers 255`). Text plus images and videos.
  Board: **66.26 (#2), ECE 0.043**. d3-flash scores 55.79.
- **Decision 2.0** (`Decision-2.0-{Vega-27B, Lux-9B, Nox-4B, Sol-2B, Eos-0.8B, Kai-0.6B}`, 09-28/29):
  `config.json:architectures ["Decision2Model"]`, `model_type decision2`. `decision_config.json:architecture
  "qwen3.5-text-endpoints-global-query-shared-bilinear-mlp"`, `head_variant shared`, `max_options 255`,
  `head_compute_dtype float32`, `loss_version "decision2-valid-k-ce-plus-optional-brier-and-replay-kl-v1"`,
  `prompt_version "decision2-segmented-options-global-query-v1"`. **Bilinear endpoint head** over option-segment
  endpoints versus a global query, not LM-head codes. Vega is a LoRA (`adapter_config: r 512, alpha 1024`,
  29.37B incl. adapter, ctx 32,768). Nox is a full checkpoint ("uniform-soup-of-full-checkpoints", ctx 16,384).
  Multi-question in one pass. Board: Vega 55.88, Nox 44.95.
- **Vela 2.0 0.3B** (`vllm-sr/Vela-2.0-0.3B`, 309M, F32): `Vela2Model` / `vela2-unified` on a 22-layer
  bidirectional ModernBERT (d 768, 12H, `global_attn_every_n_layers 3`, `local_attention 128`, YaRN to 8,192).
  `readout "both"`, `proj_dim 256`, span readout for PII and claims, calibration `tau 0.060`, `tau_span 0.070`.
  Question types: Choice, Noul, Score, Span, Set. SystemOne-compatible API.

### 3.3 Cloudflare: Clef (CLONE, joint schema head)
- `Cloudflare/clef` (09-30, 21,421 dl / 2,218 likes, 27.36B), `clef-flash` (09-30, 35,576 dl, 9.41B), `clef-omni`
  (10-09, 35.26B, Qwen3-Omni-30B-A3B thinker: 48L, d 2048, 32H/4KV, 128 experts, top-8, ctx 65,536). Apache-2.0.
- Backbones are kept with their vision encoder (`Qwen3_5ForConditionalGeneration`). Omni has vision and audio,
  with the talker/code2wav removed.
- Head: `joint_head_config.json {width 1024, routing_layers 2, layers 4, heads 16, feedforward 4096}` and
  `joint_schema_model.py` (`EvidenceRoutingLayer`, `JointSchemaHead` with `nn.TransformerDecoderLayer`s). The head
  reads the final hidden states, routes state evidence to each question and **scores all options of all
  questions jointly**: one logit per option, softmax per question. Options are encoded as `option_ids` per
  question in the record.
- Third-party description: trained with a Brier loss + RLCD. Board: clef 53.08 (ECE 0.040), flash 47.61, omni
  48.28. Also served on Workers AI. API "fully compatible with Jev and SystemOne".

### 3.4 Amazon / Strands Agents: strands-decider (CLONE, pointer head)
- `amazon/strands-decider-{E2B,E4B,12B,26B-A4B}-gemma4-v1-2610` and `-2B-qwen3.5-v1-2610`, all created
  2026-10-09, mirrored from `StrandsAgents/*` (also `strands-decider-2B-hobson-v19/v21`, 09-30 / 10-05).
  Apache-2.0. Each is a PEFT LoRA plus `head.safetensors` (no full weights; safetensors API empty).
- `strands_decider_config.json`: `head_type "pointer"`, `pointer_dim 768` (256 in hobson-v21), `num_slots 24`,
  `max_length 4096`, `lora_r 48 / alpha 96` (16/32 in hobson), targets q/k/v/o/gate/up/down,
  `kl_frozen_weight 0.3` (KL anchor to hobson-v19), `ordinal_smoothing 0.1`,
  **`temperature_by_kind {noul, choice, score}`** (E2B: 1.070 / 0.561 / 1.479).
- Bases: `google/gemma-4-{E2B,E4B,12B,26B-A4B}-it`, `Qwen/Qwen3.5-2B-Base`. The adapter goes on the Gemma 4 text
  decoder (`...model.language_model`, `task_type FEATURE_EXTRACTION`), so readout is from hidden states via a
  pointer over option slots, not from the LM head.
- Recipe: teacher Qwen3.5-4B distributions where the teacher agrees with gold, 123k + 12k code rows, 3-seed
  **LoRA soup** (mean update stored as one rank-48 adapter), post-hoc calibration. Code:
  github.com/strands-labs/strands-decider. Serves `/v1/systemone`. Multi-question per request.
- JevBench public (w4096): E2B 180/231 (Brier 0.313, ECE 0.053), E4B 187 (ECE 0.037), 12B 200 (0.061),
  26B-A4B 207 (0.051). DI 0.2.1 per the cards: 29.78 / 39.42 / 47.14 / 48.69. Board (DI 0.3.2): 2B hobson-v21
  21.34.

### 3.5 Liquid AI: d1 (CLONE, conv hybrid, tree attention)
- `LiquidAI/d1-3B` (10-05, 22,004 dl / 449 likes, 3.12B) and `d1-omni-600M` (10-05, 14,419 dl, 587M F32).
  Licence `lfm1.0` (other).
- d1-3B: `Lfm2VlForConditionalGeneration` / `lfm2_vl`, text 30L, d 2048, 32H/8KV, `layer_types` conv 22 / full 8,
  ctx 32,768, vocab 128,000, image tiles 512 px (64–256 tokens). `modeling_d1.py` `D1Model` exposes
  `system_one()` and `system_one_batch()`. `hybrid.py` packs the state as one **trunk** and each question as a
  **branch**: the trunk runs causally, each branch attends to the trunk plus itself (varlen flash), and the
  answer is read at each branch's last token. "Zero output tokens". ~8 ms per decision on an RTX 4090.
- d1-omni-600M: `D1OmniModel` / `d1_omni` on LFM2.5-Encoder-350M (bidirectional, 16L, d 1024, conv 10 / full 6,
  vocab 65,536), `head_layers 2` (`DecisionHead` reading marker positions), `max_length 16384`, a 94M vision
  encoder and a 112M audio encoder. **Per-type, per-option-count temperatures** in `config.json:temperatures`
  (e.g. `choice:2` 1.747, `noul:2` 1.666, `score:6-10` 1.0).
- Board: d1-3B 38.90 (ECE 0.047), d1-omni 9.09. Card claim: DI 0.2.1 48.57 (best under 10B at the time).
  Liquid also runs a hosted d1 API (web).

### 3.6 InternLM (Shanghai AI Lab): Intern-Decision (CLONE, LM-head symbols, multi-field)
- `internlm/Intern-Decision-0.8B/2B/4B` (09-26; 4B: 4,458 dl / 88 likes, 4.54B). Apache-2.0.
  `Qwen3_5ForConditionalGeneration` (vision kept), causal attention, tied embeddings.
- Readout (README + `inference.py`): options map to single-token symbols `A–Z, a–z, 0–9`. The prompt renders the
  state, the schema and a **complete assistant JSON skeleton** with one `<decision>` placeholder per field. One
  causal forward pass. Logits at the position **just before each placeholder**, softmaxed over that field's
  allowed symbols, then a temperature (4B: T = 1.992). So all questions are answered in one pass, but
  causally: later fields see earlier ones.
- Card table: 4B average 90.02 vs Jev 88.74 on their mix, Brier 0.347, ECE 0.065. Board: 4B 38.21 (ECE 0.028).

### 3.7 IBM: Granite decision LoRA / aLoRA (CLONE via adapter)
- `ibm-granite/granitelib-core-r1.0/decision/{granite-4.1-8b,granite-4.1-30b}/{lora,alora}` (repo created
  2025-12-10, decision added by 10-15), composed into `ibm-granite/granite-switch-4.1-8b` / `-30b` (10-14,
  `GraniteSwitchForCausalLM`, 8.78B / 29.99B, 42 / 66 layers, d 4096, 32H/8KV, ctx 131,072, full attention).
  Base `granite-4.1-8b`: 40L, d 4096, 32H/8KV, vocab 100,352. Apache-2.0, sigstore-signed (`model.sig`).
- `adapter_config.json`: `r 64`, `lora_alpha 128`, all attention+MLP projections, `task_type CAUSAL_LM`. The aLoRA
  has `alora_invocation_tokens [14711, 43917, 25189]` (`### DECISION`): the state turn runs on the **frozen base**
  (KV reusable across questions) and the adapter switches on at the question turn.
- Readout: softmax over single-token option-label logits at the first assistant position
  (`io.yaml: max_completion_tokens 1, temperature 0.0`). **One pass per question.** `choice` / `noul` / `score`
  (expected level Σ i·p(i)).
- DI v0.3 public (card): 30B LoRA 53.30, 30B aLoRA 50.91, 8B LoRA 46.00, 8B aLoRA 42.56. The Switch card gives 42.9.

### 3.8 Ant Group / inclusionAI: LemonDecider 4B (CLONE)
- `inclusionAI/LemonDecider` (10-05, 343 dl / 31 likes, 4.54B BF16 + 652,800 F32 readout params, i.e. 255 × 2560).
  Apache-2.0, `trust_remote_code`.
- `config.json`: `architectures ["LemonDeciderForDecisionClassification"]`, `model_type lemondecider_decision`,
  `attention_mode "noncausal_full_attention"` (`text_config.is_causal false`; GDN layers stay causal),
  `pooling last`, `max_options 255`, `code_scheme letters-v1`, `max_input_length 262144`, **`calibrated false`,
  `temperature 1.0`**, `readout_dtype float32` (bias-free 255-way `LemonDeciderReadout`).
- Training: 805k public rows, soft labels for ~600k, objective `valid_option_soft_cross_entropy`, 2 SFT runs
  merged (2,446 and 700 steps), 64 GPUs. Self-reported DI 0.3 public 50.31. See the lineage note in §0.

### 3.9 H2O.ai: H2O-Lightning-4B (CLONE)
- `h2oai/h2o-lightning-4b` (10-02, 4,944 dl / 84 likes, 4.66B incl. vision). Apache-2.0. Qwen3.5-4B,
  `adapter/adapter_config.json r 64, alpha 128` (also merged). `config.json:head_dtype float32`, `pooled_readout.py`.
  `serve_config.json {noul_floor 0.801, max_options 255}`. Runs on **unmodified vLLM** plus a stdlib shim, one
  output token per decision. Supports "two-way" versus "one-way" prompt reading. Claims #1 JevBench open-weights
  composite from 10-07 to 10-19. Not on the DI board.

### 3.10 Together: Tev1 (weak clone)
- `togethercomputer/Tev1-4B-experimental` (09-23, 5,204 dl, 4.66B; also 0.8B). SFT of Qwen3.5-4B. **Keeps the LM
  head**, and the answer is one option letter (2–24 options). The card says it is "not a non-autoregressive
  Jev runtime". Board: 34.14 / 12.02. Recipe at github.com/togethercomputer/tev1.

### 3.11 Meta: MetaEncoder-30B (adjacent, bi-encoder)
- `facebook/meta-encoder` (09-30, 130 dl / 27 likes, 29.78B BF16, Apache-2.0, **gated auto**, so `config.json`
  was not fetched). Files: `modeling_metaencoder.py` (14.7 kB), `vllm_metaencoder.py`, two shards
  (49.9 GB + 9.7 GB).
- Backbone `meta-models/Muse-Glimmer-30B` (via `unsloth/Muse-Glimmer-30B/config.json`:
  `MuseGlimmerForConditionalGeneration`, 52L, d 6656, 32H/**2KV**, head_dim 128, swa 39 / full 13,
  `sliding_window 2048`, ctx 131,072, vocab 202,048).
- Mechanism: the task and each candidate are embedded **separately** with grounding prompts, and the score is an
  inner product, so candidates are cacheable and ANN-scalable. This is the opposite end from cross-attention
  heads. The prompt mirrors Jev's `Task: … Criteria: - label: description`.
- Card: JevBench orig 0.944 / easy 1.0 / hard 0.739 vs OpenJev-27B 0.986 / 1.0 / 0.766. "Jev Intelligence" 81.4
  vs 84.5. No calibration figures.

### 3.12 Fastino: GLiNER2.5-Decide (label-set encoder)
- `fastino/GLiNER2.5-Decide` (09-23, **119,592 dl / 419 likes**, 486M F32): `Gliner2ForSchemaExtraction` /
  `gliner2` on `microsoft/deberta-v3-large` (24L, d 1024, 16H, 512 positions, relative attention),
  `token_pooling first`, `counting_layer count_lstm`, `architecture span`.
- `-1B` (1.19B): encoder `jhu-clsp/ettin-enc-from-dec-1b` (ModernBERT 28L, d 1792, 28H, `local_attention 128`,
  `global_attn_every_n_layers 3`, 7,999 positions).
- `-multi-` (287M): `microsoft/mdeberta-v3-base` (12L/768, vocab 250k), `max_len 4096`, `architecture boundary`.
- The label set sits in the input sequence (GLiNER2 schema), several "heads" (questions) per call, multi-label
  thresholds. Ordinal scoring is done through labels. fast-decisions benchmark avg 60.2% (vs JevK5 57.6%).
  DI board 9.24 (ECE 0.088).

### 3.13 Knowledgator: GLiClass v4 (label-set; decoder-KV)
- `knowledgator/gliclass-v4-edge` (10-16, 440M): `GLiClassModel`, `architecture_type "uni-encoder"`,
  DeBERTa-v3-large, `label_model_name BAAI/bge-small-en-v1.5`, calibrator (`calibrator_hidden_size 256`,
  `beta_max 3.0`). Self-reported DI public 47.5, 11 ms median.
- `-9b` (8.98B) / `-27b` (27.0B): `architecture_type "decoder-kv"` on Qwen3.5-9B / Qwen3.8-27B. The backbone
  writes the instruction, examples, text and images to the KV cache. Labels are **transient `<<LABEL>>` tokens**
  (`class_token_index 248077`) that read the cache without entering it. Then a 2-layer bidirectional DeBERTa label
  scorer, a pairwise MLP, a gated recurrent refinement (≤3 steps, early exit), and a per-pair temperature
  calibrator (`calibrator_objective mixed`, `rank_preserving true`). From SCX Router (arXiv 2609.02292).
  Self-reported DI 55.6, 34 ms. The 27b card copies the 9b text (it says Qwen3.5-9B) but `config.json` says
  Qwen3.8-27B.

### 3.14 HF transformers `jev` model_type (infrastructure)
- `ArthurZ/{d3, d3-*, pplx-decider-v1(.1)-27b, JEV-27B, JEV-35B, JEV-9B, GEV-26B-Decide, clef-flash, kev-4b, von,
  Darwin-27B-ZTC*, PhysicalEye-Decide-35B}` (10-13…10-16). `config.json`: `model_type "jev"`, `backbone_config`
  (original repo, e.g. `vllm-sr/d3` `Qwen3_5Model`), `num_verbalizers` (255 for code readouts, 24 for
  JEV-27B), `offsets {choice, noul, score}`, `question_types`, `softcap`, `keep_in_fp32_modules ["^proj"]`,
  `transformers_version 5.19.0.dev0`, plus `decision_template.jinja` and `decision_config.json:labels`.
  `pipeline("decision", model=...)` takes `{state, questions}`. This is a de facto standard to target for Zeiger
  export.

### 3.15 Adjacent rerankers (brief)
- `nvidia/Nemotron-3.5-Rerank-8B-BF16` (10-15, 8.38B, licence openmdw-1.1): `Mistral3ForSequenceClassification`,
  base Ministral-3-8B, **`is_causal false`**, `pooling avg`, 2_Dense 4096→1, 34L, d 4096, 32H/8KV, ctx 262,144
  (YaRN ×16 from 16k). Multimodal.
- `jinaai/jina-reranker-v3.5` (07-14, 74,872 dl, 0.60B, CC-BY-NC-4.0): `JinaForRanking` on Qwen3-0.6B, swa 16 /
  full 12 (window 1024), listwise "last-but-not-late" (query + candidates in one pass).
- `lightonai/LightOn-rerank-PW/LW-2B` (07-08, Apache-2.0): Qwen3.5-2B hybrid. PW = `logit(Yes) − logit(No)`;
  LW = listwise.
- `tencent/R3-rerank-0.6b`, `Alibaba-NLP/core-reranker-2b/8b`, `amazon/poiss-bge-m3-reranker`: pair scorers, low
  relevance.

### 3.16 Notable community clones (short; full data in the raw dirs)
- `startlux-models/StartLux-Decision-v1.1-27B-Instant`: Qwen3.8-27B VL, `decision_config.json`, CC-BY-NC, board #1
  69.12 (ECE 0.026), paper arXiv 2610.17309 (AutoThink + RL).
- `autotrust/JEV-27B-VL` (1.93M dl): Qwen3.8-27B, distilled from Jev 1.13 output distributions (per the redreamality survey). `JEV-35B`
  (Qwen3.5-35B-A3B MoE, 256 experts / top-8). `GEV-26B-Decide` (Gemma-4-26B-A4B, 128 experts, swa 25 / full 5,
  System-1 with fallback thinking). `AutoTrust-JEV-27B-v1.1`: `decision_config attention_mode
  noncausal_full_attention`, `vllm_readout.json`.
- `Mapika/decider-{0.8b,2b,4b,12b,35b-a3b,31b}`: plain `Qwen3_5ForCausalLM` / Gemma 4. Option letter read at the
  answer slot (Gemma: final-logit softcapping applied). decider-2b has **318,711 dl**. `decider-chat-gemma4-31b`
  is untrained weights + `decider_config.json` (readout only), board 57.62.
- `tasksource/tasksource-decider-nano`: 150M ModernBERT (Ettin reranker 150m), joint cross-encoder, DI public
  30.90.
- `openjev/openjev`: Qwen3.8-27B VL, up to 52 options, CC-BY-NC.
- `iapp/OpenThai-SystemOne`: `OpenThaiSystemOneForDecision` on Qwen3.5-0.8B.
- `social-jev/social-jev-*`: `SocialJevModel` on Qwen3 (full attention), own `head.py`.

## 4. Queries run

HF model search (`/api/models?search=<q>&sort=lastModified|createdAt&direction=-1&limit=100..1000&expand[]=...`,
Link-header paginated; dumps in `hf-raw/_search/`): `jev` (593 hits), `kev` (4,158, mostly unrelated),
`systemone` (56), `system-one` (73), `system1` (28), `typesafe` (6: ruvector intent classifiers, irrelevant), `noul`
(11), `choice-head` (1), `pointer` (57), `zeiger` (1), `decision` (901), `decide` (306), `decider` (170), `judge` (1,497),
`verifier` (389), `reward` (3,000 cap), `calibrat` (913), `calibrated` (100), `reranker` (2,151), `rerank` (2,527),
`zero-shot` (1,590), `gliclass` (634), `gliner` (1,423), `classifier` (3,000 cap), `typed-decision` (80), `open-jev`
(66), `jev-style` (25), `laya` (2,120), `kenning` (5), `gev` (217), `router` (1,487), `guard` (1,862), `scorer` (374),
`semif` (18), `scx` (19), `nimble` (793), `decision-index` (0), `jevbench` (0).

Author listings (`/api/models?author=<org>&sort=createdAt&limit=200`, dumps in `hf-raw/_authors/`): google, facebook,
meta-llama, microsoft, nvidia, Qwen, Alibaba-NLP, ibm-granite, ibm-research, apple, amazon, Salesforce, CohereLabs,
Cohere, CohereForAI, mistralai, allenai, NousResearch, jinaai, mixedbread-ai, answerdotai, lightonai, Snowflake, BAAI,
deepseek-ai, moonshotai, zai-org, THUDM, tencent, Tencent-Hunyuan, ByteDance, ByteDance-Seed, knowledgator,
urchade, fastino, Skywork, internlm, LiquidAI, HuggingFaceTB, cross-encoder, tomaarsen, MoritzLaurer, nomic-ai,
ServiceNow-AI, upstage, LGAI-EXAONE, naver, stepfun-ai, XiaomiMiMo, baidu, openai, ai21labs, autotrust, ArthurZ,
onnx-community, sentence-transformers, Prior-Labs, typesafe, TypeSafe, typesafe-ai (all three empty),
jaredpalmer, NandhaKishorM (empty; Laya lives at convaiinnovations), HIT-TMG, zeroentropy, ContextualAI,
voyageai, PleIAs, openbmb, OpenGVLab, meituan-longcat, inclusionAI, rednote-hilab, google-bert, FacebookAI,
intfloat, Unbabel, RLHFlow, Ray2333, infly, sarvamai, unsloth, perplexity-ai, StrandsAgents, bespokelabs, Mapika,
openjev, systemonedev, FINAL-Bench, social-jev, convaiinnovations, Horizon-Labs, vllm-sr.

Derivatives (`/api/models?filter=base_model:{adapter,finetune,quantized}:<id>`): kev-4b (5 adapters, e.g.
`Flolep2607/kev-4b-guard`, `jtatman/kev-4b-code-verify-*`), kev-0.8b (3), kev-27b (0), laya (100+ finetunes),
Mapika/decider-2b (7), pplx-decider-v1.1-27b (4 quants), Cloudflare/clef-flash (10, incl. `ArthurZ/clef-flash`),
LiquidAI/d1-3B (11, incl. `OceanLabs/Coral-2-3B`, `litert-community/d1-3B-LiteRT`).

Spaces (`/api/spaces?search=`): `decision-index`, `jev`, `systemone`, `decider`. Key find:
`multimodalart/jev-decision-index` (726 likes), whose `data/index.json` + `methodology.json` + README were saved
to `hf-raw/_spaces/dix/` and rendered to `dix_leaderboard.md`. `LiquidAI/system-one-arcade` led to `LiquidAI/d1-3B`.

Collections (`/api/collections?q=decision`): vllm-sr Decision 1.0/2.0/3.0, internlm/intern-decision, Cloudflare/clef,
ggml-org/decision-models, startlux, sieve, amos-1, AnyJev, davanstrien "Jev-style decision models: open data".

Papers (`/api/papers/search?q=`; dumps in `hf-raw/_papers/`): "system one decision", "jev", "decision model
calibrated", "typed decisions", "SCX router", "MetaEncoder". Relevant: 2610.11316 MetaEncoder; 2610.17205 "Releasing
the Inner Jev: A Simple Recipe for Open Decision Models"; 2610.18233 "Consider, Then Choose: Order-Invariant
Decisions"; 2609.33971 "Do System One Decisions Add Up?" (probabilistic coherence); 2610.18773 "Jeva: System One
Models from First Principles"; 2610.02076 "LLM-as-Jev"; 2610.02586 "Labels Override Definitions"; 2609.35865 PACT;
2610.07730 SanSi (looped typed decision model); 2610.00831 AnyJev tech report; 2609.02292 SCX Router; 2610.17309
StartLux-Decision; 2609.23886 this-that-model; 2609.30706 LAVOIR; 2610.06744 ufakzeka-karar (order-invariant option
scoring).

WebSearch: "decision model like Jev TypeSafe System One open release October 2026"; "Jev alternative open weights
decision model calibrated probabilities noul choice score"; "NVIDIA OR Google OR Meta OR Alibaba Qwen decision model
system one Jev open weights Hugging Face October 2026". WebFetch: the redreamality survey and the orcarouter
Microsoft-Decision-1 page (the towardsai article returned 403).

Detail fetch (`hf-raw/<org>__<model>/`: `api.json` with `expand[]=safetensors,cardData,config,siblings,...`,
`tree.json`, `files/*.json|md|py|yaml|jinja|txt` < 3 MB, tokenizer/vocab skipped): 120 repos, listed in §2.
Gated/unfetchable: `facebook/meta-encoder` config and code (gated auto). `scairesearch/kieve-jev` is a stub.

## 5. Recommended deep dives (in priority order)

1. **perplexity-ai/pplx-decider-v1.2 + flash-v1.2**: the open reference design closest to Jev (noncausal full
   attention in a hybrid, a 255-way code readout separate from the LM head, joint multi-question marker readout,
   distillation into 4B). The code ships in the repo (`source/`).
2. **vllm-sr/d3 (+ Decision-2.0 bilinear head, Vela-2.0 encoder)**: #2 on the board. Compare d3's code readout
   with D2.0's endpoint/bilinear head inside the same org.
3. **Cloudflare/clef**: the only big-org model with a learned **joint cross-question transformer head**. Closest
   in spirit to multi-question coupling.
4. **amazon/StrandsAgents strands-decider**: a pointer head, per-type temperatures, LoRA seed soups, and the most
   complete public recipe and data inventory (github strands-labs/strands-decider).
5. **LiquidAI/d1-3B + d1-omni-600M**: non-transformer conv hybrid, trunk/branch tree attention, and an
   encoder-sized model with per-type, per-option-count temperatures. The most relevant for small/edge Zeiger.
6. **knowledgator/gliclass-v4-9b (decoder-KV)** and **facebook/meta-encoder**: two alternative interfaces (label
   tokens that read a cache; a bi-encoder) that scale to large option sets.
7. **HF `jev` model_type / `pipeline("decision")`** (ArthurZ ports): to see whether Zeiger can export to it.
