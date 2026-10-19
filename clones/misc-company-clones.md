# Company clones: Juspay Xor, Nace.AI Drex, StartLux-Decision

Three company-built open decision models that serve TypeSafe's `POST /v1/systemone` contract. The Decision Index lists all three as "autoregressive". This is only partly true. **Xor 1.x** and **StartLux** read the **next-token logits of option letters** from an LM, so they are label-token readouts with no head. **Drex** and **xor-lite 1.1** use a **kev-style pointer head** over option-end hidden states. None of them generates text on the one-pass path; StartLux AutoThink does generate when it is unsure. Compiled 2026-10-19.

| | Juspay Xor | Nace.AI Drex | StartLux-Decision |
|---|---|---|---|
| Flagship / base | Xor 1.2 = Qwen3.6-35B-A3B; Xor 26B-A4B v1.2 = Gemma 4 26B-A4B-it | v2.0 = Qwen3.6-35B-A3B; v1.5 = MiMo-V2.6-Distill-Qwen-9B; DLM = nvidia/Efficient-DLM-8B | v1.1 0.8B–27B (Qwen3.5/3.8 family, inferred), v1 adds 35B-A3B |
| Weights | merged BF16 (LoRA r16 merged) | merged full weights + `head.pt` | full FT + merged adapter, 4-seed average |
| Readout | first-token logprobs of letter labels (A..Z, AA..), forward + reversed order averaged | kev `PointerHead` (`<decide>` · `</opt>`) | logits of letters A–Z at the last prompt position |
| Isolation | one prompt per question, SGLang radix cache shares the state prefix | one causal row per question over a cached state (hybrid); packed block-causal with bidirectional state for DLM | one prompt per question; **prompt-bidirectional** full-attention layers |
| Calibration | per-type temperature (choice 1.1, noul 1.4, score 1.0) | `head.pt` temperature (DLM: 1.0; others not visible without weights) | per-type temperature in `decision_config.json`, fitted on held-out data; RLCD² for AutoThink |
| DI 0.3.2 Full (board, 2026-10-16) | Xor 26B-A4B v1.1 **57.07** (#36); Xor 1.2 **51.15** (#56) | Drex v2.0 **55.94** (#40); v1.5 **49.72** (#59) | v1.1 27B Instant **69.12** (#1) |
| Licence | Apache-2.0 | code Apache-2.0; weights Nace.AI Open RAIL-M (DLM: CC BY-NC 4.0) | code Apache-2.0; weights CC BY-NC 4.0 |

For reference, Jev (jev-1.14.0) scores 61.53 with ECE 0.0799 on the same board (`notes/hf-raw/_spaces/dix_table.tsv`).

**Pins**

| ref | what | where |
|---|---|---|
| `X` | `juspay/xor` main `a286a461`, with README, `RELEASE_PROVENANCE.json` and config. Serving bundle `xor-1.2-serving.tar.gz` extracted only for reading, never run | `notes/hf-raw/juspay__xor/files/` (bundle under `serving/extracted/`) |
| `X26` | `juspay/Xor-26B-A4B` (v1.2 card), `serving/xn_base_server.py`, `evaluation/` | `notes/hf-raw/juspay__Xor-26B-A4B/files/` |
| `XL` / `XO` | `juspay/xor-lite` `b2b5a606` (`decider/`, serving bundle extracted) and `juspay/xor-omni` `a05bba64` | `notes/hf-raw/juspay__xor-lite/`, `juspay__xor-omni/` |
| `DG` | GitHub `nace-ai/drex-decision-models` @ `be8ba7c8` (2026-10-13) | `sources/drex-decision-models` |
| `D2` / `D15` / `DD` | HF `nace-ai/drex-v2.0`, `drex-v1.5` `f80f9d2b`, `drex-dlm` `8f6f8244` | `notes/hf-raw/nace-ai__*` |
| `SG` | GitHub `StartLuxLabs/StartLux-Decision` @ `83fe6c7d` (2026-10-17) | `sources/startlux-decision` |
| `S27` / `S9A` / `S4` / `S35` | HF `StartLux-Decision-v1.1-27B-Instant` `32e7d35d`, `v1.1-9B-AutoThink` `efc649d8`, `v1-4B` `9c0a32e9` and `v1-35B-A3B` `81e59afd`, fetched in full. The other 8 repos are api.json only | `notes/hf-raw/startlux-models__*` |
| `DIX` | Decision Index 0.3.2 snapshot of 2026-10-16 | `notes/hf-raw/_spaces/dix_leaderboard.md`, `notes/decision_index_0.3.2_entries.tsv` |

---

## 1. Juspay Xor

### Identity
- **Maker.** Juspay Technologies, the Indian payments company. HF org `juspay`; there is no GitHub repo. Serving images are published by `prakhar1611/xor-sglang` (`X:README.md`).
- **Releases:**
  - **Xor 1.0** (tag `xor-v1`) and **1.1** (`v1.1`). **Xor 1.2** (branch `release/xor-1.2`) dates from 2026-09-28 (`X:RELEASE_PROVENANCE.json`).
  - **Xor 26B-A4B** v1, v1.1, and v1.2 on main (2026-10-16).
  - **xor-lite** 1.0 and 1.1 (MiniCPM5-2B), **xor-omni** (Gemma 4 26B-A4B), and NVFP4 companions `xor-nvfp4` and `Xor-26B-A4B-NVFP4`.
  - About 5.6k downloads across the family.
- **Lineage.** The Xor 1.2 wrapper's environment variables are still named `OPENJEV_*`, and the provenance file records a `source_parent_commit fed366a6` (`X:serving/server.py:23-40`). The wrapper therefore descends from an OpenJev-style label-readout server (inferred). xor-lite 1.1 bundles **unmodified `strands_decider` code** from strands-labs/strands-decider (`XL:THIRD_PARTY_NOTICES.md:5`).

### Architecture
- **Backbone.**
  - Xor 1.2 is `Qwen3_5MoeForConditionalGeneration` from Qwen3.6-35B-A3B @`995ad96e`: 40 layers in a 3:1 linear-attention : full-attention pattern, 256 experts with 8 active, a 262,144-position window and 35.1B params (`X:config.json`).
  - The weights are a **PEFT LoRA r16 / α32 merged at α24** (scale 1.5) over 310 target tables. The 19 MTP tensors are dropped (`X:RELEASE_PROVENANCE.json` adapter, merge, `known_provenance_gaps`).
  - Xor 26B-A4B v1.2 is a **weight blend**: 0.60 × v1.1 plus 0.40 × a second LoRA-merged checkpoint (`ndnew`) on the non-expert tensors, with the routed experts taken from v1.1. The blend ratio was picked on public Decision Index results (`X26:README.md` "What changed").
- **Input layout.** There is one chat prompt per question:
  - fixed system line "You are a careful classifier.";
  - user turn = `state`, then the question text, then `Options:` with `A: desc` lines, then "Answer with a single letter only." (or "a single option code" above 26 options);
  - assistant prefix `Answer:\n`.
  - Images (≤ 8) or one video are inserted as Qwen vision pads (`X:serving/server.py:174-219`). For 26B-A4B the prefix is `Answer: (`, and the label set is derived from the tokenizer (`X26:serving/xn_base_server.py:8,40-62`).
- **Attention and isolation.** These are ordinary causal LM prompts. Every question, and each of its two option orders, is a separate sequence in one batched SGLang `/generate` call, so isolation is exact. The state prefix is shared only through SGLang's radix KV cache (`X:serving/server.py:174-178`). There is no mask engineering.
- **Readout.**
  - It is a `max_new_tokens 1, temperature 0` call with `token_ids_logprob` restricted to the option label ids. The labels come from **255 single-token markers** A..Z, then AA, AB, … (`X:serving/markers_255.json`).
  - Every question runs **twice, in forward and reversed option order**:
    - for ≥ 3 options, the raw logprobs are averaged and then softmaxed with T;
    - for binary questions, each order is tempered and softmaxed, and the two are averaged (`X:serving/server.py:179-275`).
  - This is a **label-token generator readout, not a trained head** (Xor 1.x, 26B-A4B and Omni). **xor-lite 1.1** is the exception: it switched to a **pointer head** (Strands Decider `head_type: pointer`, `pointer_dim 256`, LoRA r32/α64, 1.1M-param head) on a 0.3/0.7 weight average of two MiniCPM5-2B fine-tunes (`XL:README.md` Lineage, `XL:decider/strands_decider_config.json`).
- **Question types.**
  - **noul** becomes the options `[yes|criteria.true, no|criteria.false]`.
  - **choice** has 2–255 options, plus an optional `unknown` extra option with an abstain threshold (`min_confidence`).
  - **score** is the expected level over the levels (`X:serving/server.py:300-353`); 26B-A4B limits it to 2–10.
- **Limits.**
  - Xor 1.2: request body ≤ 8 MB, max prefill 250,000 tokens (`X:README.md`).
  - 26B-A4B: context 131,072 (`X26:xn_base_server.py:27`).
  - xor-omni: 32,768.
  - xor-lite: 4,096 training length.
- **Calibration.**
  - Xor 1.2 uses per-type temperatures **choice 1.1, noul 1.4, score 1.0**, with fallback 2.0 (`X:RELEASE_PROVENANCE.json` serving.temperature_map).
  - 26B-A4B v1.2 uses choice 1.95, noul 1.0, score 1.0. The choice value was "carried over … not re-fitted on held-out data" (`X26:README.md`).
  - Xor 1.2 **confidence is 1 − H(p)/log n** (normalised entropy), not Jev's formula (`X:serving/server.py:281-285`). 26B-A4B reports confidence = max p (`X26:README.md` limits).
  - xor-lite 1.0 used T = 100; 1.1 uses T = 1.0.

### Training
- Not disclosed for Xor 1.2. The adapter archive arrived as `f12-a24-champion-52.23-v0.2.1.zip` without `trainer_state.json` (`X:RELEASE_PROVENANCE.json`). The "52.23" is probably the selection score on a Decision Index run (inferred).
- 26B-A4B card, "No claim of benchmark-clean training":
  - v1.1 mixes contain **POP909 test rows** from the index;
  - the second checkpoint's pool had 26 exact and about 3,500 near matches against DI 0.3 under a per-field screen;
  - rows have no per-row provenance (`X26:README.md` "Training data and known gaps").
- xor-lite 1.1 lists 30 training files (`XL:decider/train_config.json`):
  - generated, multistep and adequacy sets;
  - `pool_*_gemma` pools (BBH, MMLU-aux, ARC, VAST, iSarcasm, CLadder, CRUX, ESCI, …), which suggests **Gemma-labelled or teacher pools** (inferred);
  - `di9_*` strands sets;
  - `chess24_rows`, which overlaps the index's ChessBench (card note).
- xor-lite loss settings: CE, `kl_frozen_weight 0.3`, `ordinal_smoothing 0.1`.

### Evals
- **DIX 0.3.2 Full:**

  | Model | Score | ECE |
  |---|---|---|
  | Xor 26B-A4B v1.1 | 57.07 | 0.0575 |
  | Xor 1.2 | 51.15 | 0.0236 |
  | Jev | 61.53 | — |

- **Self-run figures:**
  - Xor 26B-A4B v1.2, public DI 0.3: 59.46 on 140,620 requests, median 177 ms, p95 2.27 s on 2× RTX PRO 6000 (`X26:README.md`).
  - xor-lite 1.1, public DI 0.3: 37.51 (raw 53.66) (`XL:README.md`).
  - Xor 1.2 on JEVBench public tiers: 0.900 accuracy (208/231), Brier 0.179, ECE 0.073, p50 69 ms on one H200 (`X:README.md`).

### Serving
- Stdlib Python wrapper (`server.py`) in front of a **patched SGLang** image. The patch fixes a crash in candidate-logprob handling when scoring and generation requests share a batch.
- Docker Compose with `run.sh`, which checksum-verifies every file. Validated TP1 on one H200 (66 GB of weights).
- Extra endpoints: `/v1/systemone/permute` and `/separate`. The wrapper keeps an LRU answer cache. There is no auth; it binds to localhost (`X:README.md`, `X:serving/server.py:461-640`).
- xor-lite runs in PyTorch/FastAPI through the Strands Decider code instead.

### Licence
Apache-2.0 for every repo (card `license`), on Apache-2.0 bases (Qwen, Gemma 4 per notices, MiniCPM5). Benchmark inputs are not redistributed.

### What is novel vs Jev and kev
1. **Two-order debiasing as standard.** Every question is scored in forward and reversed option order and merged. This targets Jev's documented first-option bias (jaggedness #8) at 2× prefill cost, where kev chooses to accept order sensitivity.
2. **No head on the flagship.** Merged LoRA plus single-token label logprobs gives a stock SGLang deployment with prefix caching. It is closer to the SemIf training-free readout than to kev, but post-trained.
3. **Provenance discipline.** It publishes byte-level hashes of base, adapter, merge script and runtime image, plus an explicit "known gaps" list, including admitted benchmark contamination.
4. **Within one family**, Juspay moved its small model (xor-lite) from token readout to a pointer head, while the big models stayed on token readout.

---

## 2. Nace.AI Drex

### Identity
- **Maker.** Nace.AI (nace.ai). Hosted Drex API at drex.nace.ai, also on OpenRouter, DeepInfra and haimaker ([HF](https://huggingface.co/nace-ai/drex-v2.0), [OpenRouter](https://openrouter.ai/models/nace-ai/drex-v2.0)).
- **Launch.** Show HN "Drex: Open jev-like claims win on decision index" (HN 49848470, 2026-09-25, 4 points, linking [nace.ai/drex](https://www.nace.ai/drex)). The nace.ai page compares Drex 1.1 runs of 2026-09-26.
- **Open releases:**
  - **Drex v1.5** (HF 2026-09-28; announced 10-09);
  - **Drex DLM** (2026-10-01);
  - **Drex v2.0** (HF 2026-10-06; announced about 10-12);
  - Q8_0 GGUFs for each.
  - About 10k downloads in total.
- **Code.** GitHub `nace-ai/drex-decision-models` (Apache-2.0), plus a **llama.cpp fork** (branches `drex-v2.0` and `edlm`) and an **Ollama fork** (branches `drex-v2.0` and `nace-edlm`, adding a `CAPABILITY decision` that serves `/v1/systemone`) (`DG:README.md`).
- **Lineage.** It is an explicit **kev derivative**: `kev/api.py`, `model.py` and `checkpoint.py` are adapted from jaredpalmer/kev under Apache-2.0. The original import commit is unknown; the comparison reference is kev `fc4a17e1` (`DG:docs/upstream.md`).

### Architecture
- **Backbone:**
  - v2.0: Qwen3.6-35B-A3B, language model only, with the vision tower dropped on load. 40 layers, 256 experts with 8 active (`DG:models/drex-v2.0/README.md` "How it works", `DG:kev/model.py:89-95`).
  - v1.5: `Qwen3_5ForCausalLM`, 32 layers, 3:1 hybrid, about 9B params.
  - DLM: nvidia/Efficient-DLM-8B, a diffusion LM, loaded through Drex's own `kev/edlm.py` (no remote code).
  - Index entries say "full fine-tune" for v1.5 and v2.0; the cards say only "merged" (`D15:README.md`).
- **Input layout.** The kev encoding: reused special-token delimiters `<state> … <q> instr <opt> … </opt> … <decide>`, with user text sanitised. Drex adds ordered score legends (`DG:docs/upstream.md`).
- **Attention and isolation:**
  - The **hybrid bases (v1.5, v2.0)** use kev's row form: one causal row per question continuing the state. `rows_per_pass` is set by a packed-token budget (`DG:kev/model.py:60-66,219-233`).
  - In llama.cpp, `-np 2` "encodes the state once and shares it across questions" (`DG:models/drex-v1.5/README.md`).
  - **Drex DLM** uses a packed block-causal mask with **bidirectional state attention** (`state_bidir=True`). Branches are causal within themselves and do not see each other, and the state cannot see questions. All questions run in one forward pass when they fit (`DD:README.md` "How it works", `DG:kev/model.py:185-195`).
- **Readout.** kev's `PointerHead` with dp = 256: q = W_q·h[`<decide>`], k_j = W_k·h[`</opt>_j`], softmax(q·k/√dp / T) (`DG:kev/model.py:237-249`). This is **prefill-only with a trained head**, not label-token generation, despite the index's "autoregressive" label.
- **Question types.** noul/choice/score as in kev (noul is `[no, yes]`; choice options in request order; score uses the expected level). A `{"requests": [...]}` batch form is available (`D15:README.md`).
- **Limits.**
  - Default 16,384 tokens per state + question, raisable to **131,072** (v1.5, v2.0) or 32,768 (DLM) through `KEV_CONTEXT` / `SYSTEMONE_CONTEXT`. Over-long requests are refused, not truncated (`DG:docs/context-length.md`, `DG:kev/model.py:22-45`).
  - v1.5 long-context results: 8–32k tokens 89.5%, 32–128k tokens 93.4%, median 2.0 s (`DG:models/drex-v1.5/README.md`).
- **Calibration.** kev's single scalar `head.pt["temperature"]`, applied at eval time (`DG:kev/checkpoint.py:58`). DLM ships T = 1.0 (`DD:README.md:276`). The values for v1.5 and v2.0 sit inside `head.pt` and were not fetched (unknown). Cards are tagged `calibrated-probabilities` with no method described.

### Training
- Little is disclosed. nace.ai: Drex 2.0 "is trained on the **official training splits of the index benchmarks** alongside **verifiable procedural data**", evaluated on held-out splits ([nace.ai/drex](https://www.nace.ai/drex)).
- No training code is released (the repository has the runtime only), and no data cards. Full fine-tune per the index (unverified).

### Evals
- **DIX 0.3.2 Full:** v2.0 **55.94** (ECE 0.0183, the lowest of the three families); v1.5 **49.72** (ECE 0.0897).
- **Self-run, public DI 0.3.1:** v2.0 **61.25** and v1.5 58.08, against Jev 1.13.0 at 57.96 from the board. Nace's claim of "3.3 ahead of Jev" uses this public-only subset; on the Full board both Drex models are below Jev's 61.53.
- **v2.0 other results** (`DG:models/drex-v2.0/README.md`):

  | Benchmark | Drex v2.0 | Jev 1.13.0 |
  |---|---|---|
  | JevBench 231 | 85.7% | 86.6% |
  | DecideBench v1.1 (acc / pair) | 96.3% / 92.5% | 98.0% / 96.0% |
  | Email AUC | 0.984 | 0.981 |

  In an **OpenSpiel game arena against Jev** (8 games × 32 matches), v2.0 scores 60.9% and v1.5 56.8%.
- Drex DLM: 52.31 on DI 0.2 (`DG:models/drex-dlm/README.md`).

### Serving
- Python `serve.py` (CUDA; Apple silicon for DLM).
- llama.cpp fork run as `llama-server --embedding --pooling none` with the pointer head inside the GGUF; head matrices stay F16 in Q8_0.
- Ollama fork with `CAPABILITY decision`.
- Answers match across runners: identical on A10G in bf16 and Q8_0; within 0.03 between Metal and CPU for v2.0.
- FP8 shifts probabilities by 0.015 on average and 0.29 at most ([nace.ai/drex](https://www.nace.ai/drex)).
- Hosted pricing for v1.5 is $0.04–0.05/M input on aggregators (WebSearch summary, unverified).

### Licence
- Code: Apache-2.0 with a kev NOTICE.
- Weights v1.5 and v2.0: **Nace.AI Open RAIL-M** (use restrictions; HF `license: other`).
- Drex DLM: **CC BY-NC 4.0**.

### What is novel vs Jev and kev
1. **kev scaled to a 35B MoE.** It keeps kev's pointer head and row isolation, which shows kev's design holds at 35B-A3B and 128k context.
2. **The first pointer-head decision model on a diffusion LM** (Efficient-DLM-8B). It has a bidirectional state with causal branches and all questions in one pass, so it is closer to Jev's single-forward-pass claim than the row form.
3. **llama.cpp and Ollama `/v1/systemone` support for pointer-head models.** It treats a decision model as an embedding-mode GGUF with a custom head, and makes local CPU and Metal serving first-class.
4. **Evaluation through games against Jev** (chess, checkers and others), with honest DecideBench and JevBench rows where Jev still leads.

---

## 3. StartLux-Decision

### Identity
- **Maker.** StartLux (StartLux Labs), with the University of Hong Kong and SUSTech. Authors: Xialie Zhuang, Zunhai Su, Jia Pan, Ngai Wong, Guanhua Chen and Chuan-Wei Kuo (`SG:README.md` Citation).
- **Technical report.** [arXiv 2610.17309](https://arxiv.org/abs/2610.17309), *AutoThink Decision Models with Reinforcement Learning for Calibrated Deliberative Decisions*.
- **v1** (2026-09-29; tag `v1`): 0.8B, 2B, 4B, 9B and 27B dense, plus 35B-A3B (2026-10-01). Images and a 256K context were added 2026-10-03 (`SG:v1/README.md`).
- **v1.1** (2026-10-10/11):
  - **Instant** 0.8B, 2B, 4B, 9B, 27B;
  - **AutoThink** 4B, 9B, 27B;
  - GGUF and MLX 4/8-bit builds of each.
  - 65 HF repos in total, about 133k downloads, 783★ on GitHub.
- **Hosted.** 27B AutoThink is free at bamboonode.dev.

### Architecture
- **Backbone.**
  - Configs are `Qwen3_5ForConditionalGeneration`: 4B has 32 layers at d = 2560; 27B has 64 layers at d = 5120. All have a 262,144-position window and keep the vision tower (`S4`/`S27:config.json`).
  - v1-35B-A3B is `Qwen3_5MoeForConditionalGeneration` (`S35:config.json`).
  - The bases are Qwen3.5-0.8B/2B/4B/9B, Qwen3.8-27B per the index, and Qwen3.6-35B-A3B. These are inferred from configs and DIX; the cards do not name them.
  - v1.1 Instant recipe: "Stage 1 fine-tuned all weights on about 0.8 million decision examples. Stage 2 trained an adapter with bidirectional prompt attention on about 1.7 million further examples and merged it. The released weights are the uniform average of four such runs" with different seeds (`S27:README.md`).
- **Input layout.** One chat prompt per question:
  - system line "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only.";
  - an Evidence / Question / Options block with `A) …` lines;
  - JSON arrays of 8 or more items get `_index` annotations;
  - the prompt ends with the empty think block `<think>\n\n</think>\n\n` (`SG:startlux_decide/prompt.py:1-40`).
- **Attention and isolation.**
  - Each question is a separate prompt holding the whole state, so isolation is exact by construction.
  - In **Instant** models the full-attention layers (1 in 4) use a **prompt-bidirectional mask**: query q may read key k ⇔ k ≤ q, or both q and k fall before the prompt end p. The Gated-DeltaNet layers stay causal. With this mask, "the state is read with the question in view".
  - The mask is implemented as a vLLM plugin that reruns non-causal FlashAttention over the prompt rows (+2.2% wall time). Prefix caching and chunked prefill must be off.
  - In llama.cpp the GGUFs carry the non-causal flag; the MLX backend applies the mask itself. Plain vLLM, llama.cpp or MLX would silently run a different model (`SG:docs/vllm_bidirectional.md`, `S27:README.md`).
  - AutoThink models are causal.
- **Readout.** Logits of the 26 letter tokens (ids 32–57) at the last prompt position, divided by a per-type T and softmaxed over the listed options. Format `lodestar-letter-readout-v1` (`S27:decision_config.json`). This is a label-token readout from the LM head; there is no head.
  - For more than 26 options, "group rounds" are used: groups of ≤ 25, the top 3 of each go to a final read, and the rest share 0.001 of the probability (`wide_choice` in `decision_config.json`).
  - **AutoThink:** if the first read-out's top p is below `gate` (0.6 by default; 0.9 for the published numbers), the model writes a thought of up to 4,096 tokens and the letters are read again. The two read-outs are mixed as final = 0.25·first + 0.75·after, with the second re-tempered (`SG:startlux_decide/modelcfg.py:22,59-69`, `readout.py:24-36`).
- **Question types.**
  - noul is shown as yes/no.
  - choice takes 2–26 options per pass, or more through group rounds.
  - score takes 2–10 levels and returns the expected level.
  - Every choice reports **`none_probability`**: an internal extra "None of the above" option, renormalised away. With `allow_none`, `choice` may be null. The suggested threshold is 0.2, which catches 66.3% of all-wrong option lists at 2.2% false flags (`SG:README.md` Controls).
  - Confidence uses exactly Jev's closed-form formulas (`SG:startlux_decide/readout.py:39-57`).
- **Limits.** States up to 262,144 tokens and up to 8 images. vLLM is configured with `max-model-len 65536` in the documented command.
- **Calibration.**
  - Per-type temperatures "fitted on held-out data": 27B Instant noul 1.724, choice 1.087, score 1.184; 9B AutoThink 1.147, 0.995, 1.019 (`decision_config.json`).
  - AutoThink reasoning is trained with **RLCD²**: RL with proper scoring rules against a frozen model, plus a soft length penalty. It raises the share of traces that finish by up to 48 pp and cuts length by up to 65% (arXiv abstract).
  - `none_probability` is explicitly "an alarm score, not a calibrated probability".

### Training
- Three stages (arXiv): (1) shared decision capability, (2) Instant / AutoThink specialisation, (3) reasoning SFT + RLCD² for AutoThink. Instant sizes: about 0.8M + 1.7M examples, as above.
- The v1 data includes the **public train splits of 14 Decision Index benchmarks**, with their test items filtered out (`SG:v1/README.md`).
- Recipes and seed averages were chosen by public DI scores, so "they reflect model selection" (`S27:README.md`).
- LoRA fine-tuning and temperature recipes for users are in `v1/finetune/`.

### Evals
- **DIX 0.3.2 Full: v1.1 27B Instant 69.12, rank 1 of 174** (public 72.88, private same-skill 64.82, private new-domain 61.41; ECE 0.0261). This is 7.6 above Jev at 61.53 (`SG:README.md` release notes, `DIX`).
- **Self-run public DI 0.3:**

  | Model | DI 0.3 public | DecisionPivot pair accuracy |
  |---|---|---|
  | 27B AutoThink | 73.37 | 86.9% |
  | 27B Instant | 72.83 | 75.4% |
  | 9B AutoThink | 70.45 | — |
  | 9B Instant | 69.96 | — |
  | 4B Instant | 67.88 | — |
  | 0.8B Instant | 53.75 | — |

  DecisionPivot has 7,800 programmatic contrast pairs (`SG:README.md` Models).
- **v1:** DI 0.2.1 27B 63.88 vs Jev 1.13 57.91. The 27B scores 58.4% over 256 chess games against Jev (`SG:v1/README.md`).
- **Latency.** Instant 25–52 ms median per request on one H200 with vLLM; AutoThink 38–59 ms when it is sure.
- **Quantisation.** Q4_K_M loses up to 4.3 points.

### Serving
- `startlux-decide serve` fronts one vLLM server per GPU and checks that the bidirectional plugin is active.
- Backends: vLLM, llama.cpp, MLX and transformers.
- Production features: API keys, TLS, early 429s, access logs.
- Controls: `think_mode`, `gate`, `think_budget`, `return_thought` (`SG:README.md`).

### Licence
Code Apache-2.0. **Weights CC BY-NC 4.0**; commercial use needs a licence from StartLux (`SG:README.md`, `S27:README.md`).

### What is novel vs Jev and kev
1. **Prompt-bidirectional attention on a hybrid causal LM.** Only the full-attention layers are bidirectional, retrofitted by an adapter stage and served through a vLLM plugin. It attacks the causal-reading weakness (Hume's ordering evidence) without kev's head or mask machinery.
2. **"System 1.5" AutoThink.** It stays one-pass when confident and reasons, then re-reads, below a gate. RLCD² RL rewards calibrated probabilities rather than accuracy, which reinterprets TypeSafe's unpublished RLCD.
3. **A built-in none-of-the-above signal** (`none_probability`) and group rounds that go beyond the 26-letter limit, where Jev uses a "two-stage process above 255".
4. **Seed-averaged full-FT 27B**, top of the board by a wide margin, though it is non-commercial and selected on the public suite.
