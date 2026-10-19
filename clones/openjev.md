# Open-Jev / OpenJev: two independent clones

Two unrelated projects with near-identical names. Compiled 2026-10-19. **Part A** is Zefan Cai's `Zefan-Cai/Open-Jev`, a per-candidate cross-encoder with a scalar Yes-minus-No head. **Part B** is Junhao Shen's `ejhshen/OpenJev`, a per-option branch model with a permutation-equivariant set head and RL post-training. They share no code or weights. Name collisions are listed in Part C, so always cite the full repo path.

Source pins:

| ref | what | where |
|---|---|---|
| `Z` = `Zefan-Cai/Open-Jev` @`bd411888` (2026-10-04, merge PR #51) | shallow, blob-filtered clone; read only, never executed | `/home/ai/projects/jev-clones/sources/zefan-open-jev` |
| `HFZ2`, `HFZ9`, `HFZ27` = `ZefanCai/Open-Jev-2B` @`0c7aa498`, `-9B` @`47e96688`, `-27B-v1.1` @`28cf7306` | API JSON, tree, README, `package/checkpoint/{model,temperature}.json`, adapter_config, provenance and verification JSON (no weights) | `notes/hf-raw/ZefanCai__Open-Jev-*/` |
| `F` = `lyuyiqi/open-jev-fast` README (HEAD as of 2026-10-19; pushed 2026-09-28) | README only | `notes/hf-raw/github__lyuyiqi__open-jev-fast/README.md` |
| `E` = `ejhshen/OpenJev` @`e279c448` (2026-09-29) | shallow clone; read only | `/home/ai/projects/jev-clones/sources/ejhshen-openjev` |
| `HFE` = `shenjunhao/OpenJev-4B` @`7236cd17` | README, `openjev_config.json`, `calibration.json`, `manifest.json`, `evaluation.json`, backbone `config.json` (no weights) | `notes/hf-raw/shenjunhao__OpenJev-4B/` |

---

# Part A: Zefan-Cai/Open-Jev (Open-Jev-2B / 9B / 27B-v1.1)

## Identity

| | |
|---|---|
| Repo | github.com/Zefan-Cai/Open-Jev. Created 2026-09-20, ★418, code MIT (`gh api`; `Z:LICENSE`). Site zefan-cai.github.io/open-jev |
| Weights | `ZefanCai/Open-Jev-2B`, `-9B` (both 2026-09-20), `-27B-v1.1` (2026-09-22/23). Each holds a LoRA adapter, `head.pt`, `temperature.json` and provenance. **No base weights.** HF reports 0 downloads (`HFZ*/api.json`). Datasets: `ZefanCai/Open-Jev`, `ZefanCai/Open-Jev-v1.1` |
| Lineage | "Independent implementation inspired by TypeSafe's Jev. It does not reproduce proprietary RLCD" (`Z:README.md` intro). No kev or Hume code is reused |
| Status | 27B v1.1 is the flagship. "New 2B training stopped; new 9B training did not start" (`Z:README.md` results). Later v6/v7/v8 adaptations are experimental and the released weights are unchanged |
| Decision Index 0.3.2 | Not listed (no `Zefan`/`Open-Jev` row in `notes/decision_index_0.3.2_entries.tsv`). benchmarkheaven JevBench v1.5.4, observed 2026-10-02: 9B #46 (composite 24.356), 2B #65 (9.073). No 27B entry (`Z:docs/evaluation-ledger.md:42-76`) |

## Architecture

### Backbone
| ckpt | base (pinned revision) | LoRA | trained params |
|---|---|---|---|
| 2B | `Qwen/Qwen3.5-2B` @`15852e8c` | r 8, α 16, dropout 0 | ~10 MB adapter |
| 9B | `Qwen/Qwen3.5-9B` @`c2022362` | r 8, α 16 | ~23.6 MB adapter |
| 27B-v1.1 | `Qwen/Qwen3.8-27B` @`1d4bf0f2` (post-trained, not -Base) | r 8, α 16 | 15,471,617 (`HFZ27:README.md`) |

- The backbone is loaded as `AutoModelForImageTextToText`, then only `full.model.language_model` is kept. The vision tower, MTP and `lm_head` are discarded after init (`Z:jev/model.py:101-130`). The backbone is frozen, runs in bf16, and uses SDPA.
- LoRA targets are `q,k,v,o_proj` plus the Gated-DeltaNet `in_proj_qkv` and `out_proj`. MLPs are not adapted (`Z:jev/model.py:189`; `HFZ27:adapter_config.json`).

### Input layout
- One question becomes **K independent prompts, one per candidate**, rendered through the Qwen chat template (user turn, `enable_thinking=False`) (`Z:jev/model.py:226-236`):
  `Context:\n{state}\n\nQuestion: {instructions}\nProposed answer: {option}\nIs this proposed answer correct? Answer Yes or No.` (`Z:jev/api.py:78-85`)
- A noul question is one prompt ending "Is the answer to this question yes?". Criteria are appended as "Yes means: … / No means: …" (`Z:jev/api.py:66-73`).
- Choice candidates are rendered as `name` or `name: description`. Score levels pass only their text, so "Candidate order, question IDs, adjacent score levels and targets are absent" (`Z:jev/api.py:83`).
- JSON state is rendered with `json.dumps(sort_keys=True)`. No special delimiter tokens are used, so the boundaries are plain text (inferred: forgeable by the content of the state).

### Attention & question isolation
- **Isolation is exact by construction.** Every (question, candidate) pair is a separate causal sequence, so neither questions nor options can see each other. This is the strongest isolation among the clones, but options are **never read jointly**. That differs from Jev, which Hume found reads its options jointly (`clones/jev-reference.md` §3).
- As a consequence, the output is **order-invariant**, and "none of the above"-style options cannot reference their siblings (inferred).
- Optional `--prefix-cache` prefills the shared state and question once, then branches for each candidate suffix, keeping separate attention, conv and recurrent states. It is off by default because the Open-Jev-2B CUDA comparison "exceeded the probability tolerance on 9/11 workloads" even though all 440 selected decisions matched (`Z:README.md` Train section; `Z:docs/prefix-caching.md`).

### Readout head
- `nn.Linear(hidden, 1)` in fp32 on the final-position hidden state of each prompt. It is **initialised to `lm_head[Yes] − lm_head[No]`** with bias 0 (`Z:jev/model.py:107-126,182-185`), so at step 0 the head equals the base model's zero-shot Yes/No logit difference.
- Choice and score: the K scalar logits are softmaxed after dividing by the temperature T. Noul: logits `[0, s]`, i.e. sigmoid(s/T) (`Z:jev/model.py:263-269`). Large choices are scored in bounded candidate batches and normalised only after reassembly (`Z:README.md`).

### Question types
| type | candidates | output | confidence |
|---|---|---|---|
| noul | 1 prompt | `noul = p_true` | none |
| choice | 1..255 keys (`Z:jev/api.py:48`) | argmax, probabilities | `(p_max − 1/n)/(1 − 1/n)` ("official system-one-adapter formula", `Z:jev/metrics.py:69-78`) |
| score | 2..10 levels | `Σ i·p_i`, legend | TypeSafe uniform-MAD formula (`Z:jev/metrics.py:81-91`) |

### Limits
- **4,096 tokens per candidate prompt** (the saved `max_length`). Longer input is rejected, never truncated (`HFZ*/package/checkpoint/model.json`; `Z:jev/model.py:239-240`). JevBench runs used 16,384 (`HFZ27:README.md`).
- Cost scales as K × (state + question) without the prefix cache.

### Calibration
- A **single global temperature** is fitted on 512 calibration rows by CE, using a log grid plus golden-section search (`Z:jev/metrics.py:167+`). It is saved per checkpoint: 2B T = 1.519, 9B T = 1.897, 27B T = 2.534 (`HFZ*/temperature.json`).
- The training loss is itself partly a proper score: CE plus a 0.1 × Brier term.

## Training
- Objective: **soft cross-entropy + 0.1 × Brier** over each question's candidate softmax (`Z:jev/train.py:415-416`). Defaults: lr 5e-5 (LoRA), head-lr 1e-4 (`:556-558`). Each candidate is an independent sequence (`method: independent_candidate_lora_nll_brier`).
- **2B / 9B**: one pass over the frozen `release-v2` mix, 80,816 training rows (115,821 records / 10,771 groups in all), 20,204 steps at global batch 4 (`HFZ9:README.md`).
  - Sources are all programmatic or synthetic: customer cases, Doom, Wikispeedia, workflows, games, painting geometry and reasoning controls (`Z:README.md` Train).
  - Public redistributable split: 79,116 rows. 1,700 Wiki rows are withheld for licensing.
- **27B v1.1**: warm-started from an earlier 27B checkpoint at step 26,500 (trained on the browser/drone expansion), then one pass over `community-hard-mix-v2-final`.
  - Data: 148,639 rows = 82,045 WANLI NLI + 30,720 new synthetic community tasks + 35,874 replay.
  - 37,160 steps at batch 4; lr 2e-5 LoRA / 5e-5 head; 4-rank FSDP2 (`HFZ27:README.md` Training).
- Labels are code-generated or taken from public data (WANLI, CC-BY-4.0). No Jev outputs are used as labels (inferred from provenance manifests; Jev is used only as an eval baseline).
- An audit of other "OpenJev" HF datasets "found no accessible training corpus and imported zero upstream rows" (`Z:docs/openjev-hf-data.md`).

## Evals
- **Internal holdouts** (synthetic, same generators), hard accuracy (`Z:README.md` table; `HFZ*/README.md`):

  | | Old Test | Old OOD | Expanded Test | Expanded OOD | JevBench public 231 | Hard 111 |
  |---|---|---|---|---|---|---|
  | 2B | 94.71% | 86.02% | – | – | 64.94% | 41.44% |
  | 9B | 97.54% (ECE 0.008) | 91.97% (ECE 0.037) | – | – | 77.49% | 59.46% |
  | 27B-v1.1 | 98.31% | 95.98% | 96.65% | 96.44% | **85.28%** | 72.07% |
  | Jev 1.13.0 | | | | | 86.58% | 72.97% |

- No full-data base-model baseline exists, so the "training gain" is not isolated (author's own caveat).
- Weak spots the author reports:
  - natural support routing: 2B scores 64.8% vs BM25 82.4%;
  - browser snapshots: 2B scores below the always-BLOCKED baseline;
  - drone snapshots: all models score below a constant baseline;
  - the Snake and platformer policies collect no food and complete no episodes (`Z:README.md`).
- **Latency vs Jev**, same client, 11 workloads: customer service p50 85 ms (local) vs 295 ms (Jev over HTTPS). At 1,024 state tokens × 32 candidates, Open-Jev is slower: 1,016 ms vs 301 ms. The hardware differs (`Z:README.md` Inference latency).

## Serving
- `python -m jev.server` serves `POST /v1/systemone` (TypeSafe shape) and `/health`, plus a text/CSV workbench, a developer lab and a painting demo. It binds to loopback, port 8791, `--batch-size 1` by default (`HFZ9:README.md`; `Z:README.md`). Docker images bake in 2B (GPU and CPU variants); there is an HF Space CPU workbench (2B); 9B fits bf16 on one 16 GB card.
- Acceleration:
  - **`triton-tail`** (all sizes): a fused final-row RMSNorm plus fp32 head. Whole-model speedup is only 1.006–1.020× (`Z:docs/fast-backend.md`).
  - **`fast-cuda`** (27B only, sm90+): vendors **lyuyiqi/open-jev-fast** @`c52b8bb9` (MIT). That project merges LoRA, fuses kernels, captures CUDA graphs and builds a request-local prefix tree.
    - Its own numbers on a B300: the example request takes 17.3 ms vs 103.7 ms for FLA-PyTorch; JevBench mean is 42 ms vs 258 ms (6.1×) at 197/231 vs 198/231 (`F:README.md`).
    - In Open-Jev's H100 integration it **failed the probability tolerance** (0.00593 vs 1e-4), so it stays experimental (`Z:README.md`).

## Licence
- Code: MIT. Adapters and heads: Apache-2.0 with the Qwen attribution (`HFZ*/LICENSE`, `UPSTREAM.md`). Generated controls: CC0. WANLI: CC-BY-4.0. The Qwen bases (Apache-2.0) must be fetched separately.

## What is novel vs Jev and kev
1. **Reranker-style cross-encoder, not a branch mask.** Each candidate is a separate full prompt scored by a scalar head **seeded from the LM's own Yes−No logit**. The model starts as a zero-shot verifier, and LoRA plus the head sharpen it. kev instead packs all options into one branch and points at the `</opt>` tokens.
2. **Order-invariant and fully isolated by construction**, at the cost of K× compute and no cross-option reasoning. Jev reads its options jointly; kev reads them in order.
3. **Proper-score training (CE + Brier) plus a saved T**, with unusually heavy provenance and audit tooling (hash-bound packages, zero-missing audits, refusal to serve a partly offloaded model).
4. Broad **synthetic agent/game/workflow data** rather than public NLP sets. The 27B is close to Jev on public JevBench (197 vs 200 of 231).

---

# Part B: ejhshen/OpenJev (OpenJev-4B)

## Identity

| | |
|---|---|
| Repo | github.com/ejhshen/OpenJev, created and last pushed 2026-09-29, ★23, MIT (`gh api`; `E:LICENSE`). Author Junhao Shen |
| Weights | `shenjunhao/OpenJev-4B` (2026-09-27 → 09-29). **Full merged backbone** (3 safetensors shards, ~8.4 GB, bf16 per `backbone/config.json`) + `decision_head.safetensors` (37.8 MB) + tokenizer (`HFE:tree.json`, `manifest.json` stage `decision-rlcd`). Downloads hidden or 0 |
| Data | `shenjunhao/OpenJevData-140k` (100,345 SFT + 46,393 RL rows); eval set `shenjunhao/mmdm` (`E:docs/TRAINING.md`) |
| Subtitle | "Learning Open-Vocabulary Probabilistic Decision Models from Pretrained Language Models" (`E:README.md:3`). No paper was found |
| Decision Index 0.3.2 | Not listed |

## Architecture

### Backbone
- Text backbone of **Qwen3.5-4B** (hybrid Gated-DeltaNet), hidden size 2560 (`HFE:openjev_config.json`). **Full-parameter** training in both stages (`E:docs/TRAINING.md:3`). Compute contract `bf16-backbone-fp32-head-v1`; FLA gated-delta kernels (`fla-core 0.5.2`).

### Input layout
- Plain-text tags, no special tokens (`E:src/openjev/models/decision/serializer.py:65-91`):
  `<STATE>{state}</STATE>\n<QUESTION>{q}</QUESTION>\n<OPTION><NAME>{name}</NAME>[<CRITERIA>{c}</CRITERIA>]</OPTION>`
- One **branch per option** = state + question + that option. `question_end` is the last prefix token and `option_end` is the `</OPTION>` end token. No BOS and no chat template.
- Noul becomes two options, `False` and `True`, with the optional criteria; score becomes its levels (`E:src/openjev/api/typesafe.py:54-68`).

### Attention & question isolation
- Questions are **separate decisions**. Options are **separate causal branches over a shared prefix**: "expanded" mode re-runs each branch, and shared-prefix mode prefills the state once, forks it per question, then forks per option. Caches are request-local (`E:src/openjev/runtime/prefix.py:1-30`). No option sees another option inside the backbone.
- Cross-option information enters **only in the head** (see below). Order sensitivity therefore exists only through numerics: "predictions follow the options when they are reordered, up to numerical precision" (`E:README.md` §2).

### Readout head (`DecisionHead`, `E:src/openjev/models/decision/head.py`)
```
q   = W_q · h[question_end]              (mean over the K identical branch prefixes; pooling.py:208)
o_j = W_o · h[option_end_j]               2560 → 512
o   = OptionSetInteractor(o + q)          2 pre-norm MHA+FFN blocks, 8 heads, ffn 2048, NO positional encoding
z_j = (C_q q) · (C_o o_j) / sqrt(512)     bilinear compatibility, fp32
p   = softmax(z / T)
```
- The interactor is a permutation-equivariant transformer over the option set (`set_interactor.py`), so the model can still capture "overlap, competition and references to other options" (`E:README.md` §2). The head is 37.8 MB and initialised independently of the backbone (Xavier; final bilinear gain 0.1).

### Question types
| type | options | output | confidence |
|---|---|---|---|
| choice | 1..255 | argmax, probabilities | **1 − H(p)/log K** (normalised entropy), not TypeSafe's formula (`E:src/openjev/api/typesafe.py:82`) |
| noul | False/True branches | `noul = p[1]` | none |
| score | 1..10 levels | expected index, legend | 1 − H/log K |

### Limits
- 8,192 tokens per branch and 65,536 "logical" tokens per decision (prefix + all options). No truncation (`HFE:openjev_config.json`).

### Calibration
- `calibration.json` = `{"temperature": 1.0, "fitted": false}`, so **no post-hoc temperature** is applied. Calibration comes from training on soft targets and the RL proper-score reward (below).

## Training
- **Stage 1, SFT**: decision cross-entropy over hard or soft target distributions. Global batch 1,024 decisions; lr 4e-6 backbone / 4e-5 head; 10 warm-up steps then constant; 1 epoch, i.e. 98 steps on the public data; 8 GPUs, FSDP2 via verl-agent @`20bd331b` (`E:training/sft.json`; `E:docs/TRAINING.md`).
- **Stage 2, REINFORCE-Analysis ("RLCD" module)**: the policy is the option distribution itself, so the RL acts on decisions and samples no generated tokens.
  - 16 sampled actions per decision with 10% uniform exploration; binary outcomes are drawn from a hidden outcome tape.
  - Reward per sampled action: `α(c − p) + (1−α)·c/p` with α = 0.5 (`E:src/openjev/rlcd/estimators.py:104-113`). This mixes a Brier-like and a log-score-like gradient (inferred), plus an analytic term for known probabilities.
  - Variance and drift control: importance correction, a leave-one-out baseline, and KL to a frozen SFT reference with coefficient 0.01.
  - 25% of SFT contexts are replayed under the same RL loss. lr 7.5e-7 / 7.5e-6; 71,479 contexts, 70 steps (`E:training/rl.json`; `E:docs/TRAINING.md`).
- The public recipe starts from the released OpenJev-4B artifact. How the original head was first attached to raw Qwen3.5-4B is not shown (inferred).

## Evals (HF card §3; `HFE:evaluation.json`)
| | OpenJev-4B | Jev 1.13 | Decider 4B v2.1 | JevK5 4B v0.3 | Intern-Decision-4B |
|---|---|---|---|---|---|
| MMDM Hard (n = 4,351) acc / Brier | 85.57 / 0.186 | **86.37 / 0.173** | 78.88 / 0.283 | 74.70 / 0.333 | 66.95 / 0.407 |
| MMDM Soft (n = 610) Brier | **0.039** | 0.112 | 0.139 | 0.109 | 0.123 |
| JevBench public 231 acc / Brier | **88.31 / 0.176** | 86.15 / 0.182 | 82.68 | 87.45 | 87.01 |

- MMDM is the author's own collection, and its published split merges dev and test: "not a new blind evaluation" (author's caveat). Training includes scenarios "informed by public decision-task patterns". Treat the Soft-Brier lead as in-distribution (inferred).
- The strongest category is "Option logic": 51.9% vs Jev 41.1%, consistent with the set interactor (inferred).
- Jev scores 86.15% here vs 86.58% in Zefan's run of the same suite: different JevBench versions (v1.3.0 here).

## Serving
- `openjev.runtime.engine.DecisionEngine` (`execution_mode="expanded"` or shared-prefix) plus an HTTP server with the TypeSafe request shape (`E:docs/INFERENCE.md`; `E:src/openjev/runtime/server.py`). Validated on PyTorch 2.11 + cu129. No latency figures are published.

## Licence
- MIT for code and weights (`HFE` cardData `mit`). The Qwen3.5-4B base is Apache-2.0. The dataset licence is not checked here.

## What is novel vs Jev and kev
1. **Option-set interactor**: per-option branches isolate the backbone, and a small positional-free transformer restores cross-option reasoning **while staying permutation-equivariant**. This is a middle path between kev (joint ordered reading, order-sensitive) and Part A (fully independent).
2. **Bilinear question–option compatibility** in a 512-d set space, instead of kev's dot-product pointer.
3. **Decision-level RL with an outcome tape** ("REINFORCE-A", reward mixing α(c−p) and c/p), the most concrete public attempt at an RLCD-like recipe, with no temperature step at all.
4. **Full-parameter** post-training of a 4B hybrid model, shipped as merged weights.

---

# Part C: other projects using the OpenJev name (do not conflate)

| Full path | What | Source |
|---|---|---|
| HF `AlexWortega/openjev` | The first "openjev" (2026-09-16): Qwen3.5-4B NLI cross-encoder; 649 likes, downloads hidden | `notes/discovery.md` #120 |
| GH `TheoLeeCJ/SemIf-OpenJev` (also `TheoLeeCJ/openjev`) | SemIf, formerly OpenJev: training-free readout of declared option logits from a frozen Qwen3.5-4B | discovery.md L280 |
| HF `openjev/openjev`, `openjev/OpenJev-Flash-9B` | openjev org: 27.4B and 9B decision checkpoints, weights CC BY-NC 4.0 | discovery.md #43 |
| GH `alanhuangyoo/OpenJev` / HF `alanhuangya/OpenJev-{1.7B,4B,8B}` | Qwen3 decision models, architecture not verified | #84 |
| HF `v6543210/openJev-1.5B` | Qwen2.5-Coder-1.5B "non-autoregressive System 1" | #94 |
| GH `IamBusy/OpenJev-Vision` / HF `IamBusy/OpenJev-0.6B` | Encode an image once, answer many questions | #138 |
| GH `ekzhang/openjev-sglang`, `razorback16/openjev`, `JoshuaSP/open-jev`, `daseinlabs/open-jev`, `nico-martin/open-jev`, `cappuch/openjev.cpp` | Servers, readouts and ports (SGLang, DiffusionGemma, Gemma 3 log-prob, browser ONNX, llama.cpp fork) | discovery.md L284-326 |
| HF `com-kotobalabs/open-jev-deberta-v3-large`; GH `Heman10x-NGU/openJev-verdict-2.0`; HF `dknguyen2304/openjev-rlcd-qwen3-4b-v1` | DeBERTa span head; GLiClass Verdict; an in-progress RLCD attempt | #161, #159, L493 |
| "OpenJev" hosted on Codiv; Kotoba "OpenJev" | Gateway or hosted variants; Zefan's JevBench report names both as distinct systems | discovery.md L376; `Z:docs/jevbench-public.md:7` |
