# NanoJev (TianyuCodings / C-Tianyu)

A 0.6B "nano replica" of Jev on Qwen3-0.6B. It runs **one forward row per candidate** (state + question + one candidate), reads the last-token hidden state through a scalar head, adds a **permutation-invariant set-attention correction** across a choice's candidates, and takes a softmax. Despite the Jev-style typed interface and 2.5k GitHub stars, the released checkpoint is **not general-purpose**. It is trained only on four game-policy tasks (Maze, Snake, ViZDoom Basic, ViZDoom Predict Position), partly on **Jev's own API output distributions**. It does not serve `/v1/systemone`.

| ref | what | where |
|---|---|---|
| `G` = GitHub `TianyuCodings/NanoJev` @ `45b62928` (2026-10-15, "READMEs: add the question count of System1-Arena") | shallow clone, read only, never executed | `/home/ai/projects/jev-clones/sources/nanojev` |
| `HF` = `C-Tianyu/NanoJev` @ `047b927b` (release tag `unified-games-v1`) | API JSON, tree, README, TRAINING_RECIPE.md, config.json, summary.json, variants/*, the vendored `source/` scripts (no weights) | `/home/ai/projects/jev-clones/notes/hf-raw/C-Tianyu__NanoJev/` |

---

## Identity

| | |
|---|---|
| Maker | Tianyu Chen (GitHub TianyuCodings, HF C-Tianyu). Training paths in the config show a co-worker account `rwang` (`HF:config.json` `input`) |
| Created | GitHub 2026-09-17; HF 2026-09-17 → 2026-09-20 (`HF:api.json`). GitHub has 2,565 stars; HF has 8,298 downloads and 88 likes (`notes/families.json`) |
| Release | `unified-games-v1`: step 400 of the `hard_lr1e5` run, seed 17, one checkpoint for all four games (`G:README.en.md:84-93`, `HF:TRAINING_RECIPE.md:3-7`) |
| Related | Same author: **System1-Arena** (arXiv 2610.21244, about 100K questions, a Jev-like benchmark with a leaderboard) and **JevHarness** (`G:README.en.md:7-9`). Derivatives include `SUPER321/jevflash-doom-basic-0.6b`, `yannelli/nanojev-code-review-decisions`, `chenyangcun/NanoJev-MLX`, `samguan2020/chinese-chess-jev` and `CYMCharming/S1Q` (`notes/hf_candidates.tsv`, `notes/github_candidates.tsv`). `sdmlai/nano-jev` is an unrelated project |
| Decision Index 0.3.2 | not listed |
| General-purpose? | **No** for the released weights: the data is 18,760 game questions, 16,333 of them ViZDoom (`G:README.en.md:68-78`). The pre-game "toy/workflow" checkpoints were trained on small Jev-labelled decision sets, but they are not the release (`G:docs/DEVELOPMENT_RESULTS.md:11-13`) |

## Architecture

### Backbone
- **Qwen/Qwen3-0.6B** (post-trained) @ `c1899de2`, loaded as `AutoModel` (no LM head) (`HF:config.json` `model`, `resolved_model_revision`; `G:scripts/train_pipeline_decisions.py:439-440`).
- **Full fine-tune, no LoRA.** Backbone lr 1e-5, heads 1e-4 (head warm-up 1e-3). FP32 master weights with bf16 autocast and gradient checkpointing, trained on an A100 80GB (`HF:config.json`).

### Input layout
- Each candidate gets its own **leaf** token sequence (`G:scripts/train_toy_decisions.py:45-57`, `G:scripts/predict_toy_decisions.py:110-111`): `"State:\n{state}\n" + "Question type: {typ}\nQuestion:\n{instructions}\n" [+ "False/True criterion: …"] + "Candidate:\n{text}\nDecision:" + <eos>`.
- Plain-text headers only: no special tokens and no chat template. Choice leaves are `key: description`. Score leaves hold **only their own level text**, with no index and no neighbouring levels. Boolean has **one** leaf, `"The proposition is true."` (`G:scripts/train_toy_decisions.py:33-43`).
- Dict/list states are serialised with Python `str(...)`, which is not canonical JSON. The project's own contract audit flags this (`G:docs/TYPESAFE_CONTRACT.md:7`). Question ids never enter the model (`:8`).

### Attention & question isolation
- Every leaf is an independent causal row, so isolation holds by construction **at both levels**: no cross-question attention, and no candidate sees another candidate's tokens inside the backbone (`G:docs/TYPESAFE_CONTRACT.md:9`).
- The state is **repeated in every leaf**, with no shared-prefix computation ("batching does not imply shared-prefix computation"). Shared-prefix inference is an open roadmap item (`G:README.en.md:154`). A tree-attention design was checked only on a pure-Python toy (`G:research/tree_attention_check.json`).

### Readout head
`DecisionModel` (`G:scripts/train_toy_decisions.py:86-141`):

```
h_j   = LayerNorm(h_last(leaf_j))             end-of-sequence hidden state
z_j   = w·h_j + b                              scalar head
choice only (set_head="attention"):
  u_j = W_p [h_j ; log K]                     128-d, 4-head nn.MultiheadAttention over the K candidates (no positions)
  z_j += W_o tanh(u_j + MHA(u)_j)              zero-initialised residual
boolean: logits [0, z]  -> p_true = sigmoid(z)
p = softmax(z over K)
```

- The set head is permutation-equivariant, with no positional input. It sees `log K`, so the number of candidates can shift probabilities. It applies to **choice only**; score levels are scored independently and then softmaxed (`G:docs/TYPESAFE_CONTRACT.md:12`).

### Question types
| type (local name) | candidates | output |
|---|---|---|
| `boolean` (not `noul`; local `noul` **fails validation**) | 1 leaf | `p_true`, a 2-entry distribution, a thresholded value |
| `choice` | 2–255 (`G:scripts/predict_toy_decisions.py:72-73`) | softmax distribution and argmax; **no `confidence`** |
| `score` | 2–10 (`:79`) | `score = Σ i·p_i`, `level` = argmax; **no `legend`, no `confidence`** |

(`G:docs/TYPESAFE_CONTRACT.md:10-14`)

### Limits
- Every leaf must fit `max_length`; inputs are never truncated silently (`G:scripts/predict_toy_decisions.py:114-115`). The release was trained at 8,192 (`HF:config.json` `max_length`).
- The demo server caps a request at 32 states, 96 questions and 256 candidate paths (`G:scripts/serve_decisions.py:43-44`).

### Calibration
- **None shipped**: `temperature 1.0`, `temperature_fitted: false` (`HF:config.json`). A `calibration` split exists (1,708 questions) but is not used for scaling.
- Research only: `docs/RLCD_EXPERIMENT.md` derives a sampled **pairwise proper-reward policy gradient**. It draws M samples from p and uses R = (2/M)Σ1[A_i=Y] − agreement penalty, whose expectation is ‖q‖² − ‖p−q‖², an unbiased stochastic gradient of the Brier score (`G:docs/RLCD_EXPERIMENT.md:15-30`). It is checked exactly in float64 and run on a CPU MLP and on Qwen, and "does not establish that the sampled method is better than direct controls". It is **not active** in the release (`HF:TRAINING_RECIPE.md:6`).

## Training
- **Earlier stages (inferred chain).** Toy and workflow decision sets were labelled by **Jev through Vercel AI Gateway** (`teacher:'jev', model:'typesafe-ai/jev'`, with cost tracking) (`G:scripts/label_toy_decisions.mjs:47`). Targets are Jev's native probabilities at their declared rounding, or gold labels (`G:scripts/train_toy_decisions.py:59-72,144-153`). Then followed a Basic-supervised game stage, which is the `training_initialization/` bundle (`HF:TRAINING_RECIPE.md:12-13`).
- **Release run `hard_lr1e5`** (`HF:config.json`, `HF:TRAINING_RECIPE.md`):
  - 600 updates; step 400 selected on minimum weighted dev CE.
  - Each batch holds 24 questions (8 Maze, 8 Snake, 4 Basic, 4 Predict Position), loss-weighted 1/3, 1/3, 1/6, 1/6.
  - Weight decay 0.01. Seeded with replacement from 10,893 eligible train questions.
- **Targets by task:**
  - Maze and Snake use **`api_policy_distribution`**, i.e. Jev API action distributions (distillation).
  - ViZDoom uses **`expert_action`** one-hot labels from 896 recorded Predict Position expert episodes plus APPO-based Basic supervision (`HF:config.json` `policy_questions_by_split_task_target_kind`).
  - Loss: complete-question categorical CE. A TD/Brier outcome loss is implemented but off (`td_weight 0`).
- **Variants on HF:** `events_brier`, `events_ce`, `events_paired` (the RLCD-style estimator), `games_api`, `games_gold`, `local_atomic` (`HF:variants/*`).

## Evals
Held-out gameplay, 274-case test set, same observation interface, ε-greedy 0.1, seed 17 (`G:README.en.md:56-64`):

| | Maze | Snake | ViZDoom Basic | Predict Position |
|---|---|---|---|---|
| **NanoJev** | 4/10 | 8/8 | **128/128** | **27/128** |
| Jev (API) | **7/10** | 8/8 | 56/128 | 11/128 |
| Untuned Qwen3-0.6B | 2/10 | 0/8 | 56/128 | 11/128 |

- Showcases use code planners on top of the model's atomic judgments: 50×50 maze in 225 attempts (Jev 2,738), and Snake with 30 food over 256 steps (`G:README.en.md:24,40`).
- Dev at step 400: Maze CE 0.855, KL to the Jev target 0.047, argmax agreement with Jev 0.685. Shooting CE 0.407 (`HF:summary.json`).
- **No general NLP/decision benchmark**: no MMLU, no kev suites, no Decision Index, no JevBench. Its whole-map direct-action pilot gave 0 maze successes for every learned system, Jev included (`G:docs/DEVELOPMENT_RESULTS.md:23-29`). Broad evaluation moved to the separate System1-Arena project (inferred: NanoJev is the author's testbed, not the benchmarked model).

## Serving
- `scripts/serve_decisions.py` is a stdlib `HTTPServer` on 127.0.0.1:8765. It serves **`POST /api/evaluate`** (batched states/questions) and `/api/health`, plus static web replays (`G:scripts/serve_decisions.py:22-44`, `G:README.en.md:128-134`). **No `/v1/systemone`.** The response lacks `confidence` and `legend`, and the type is `boolean`, not `noul`, so it is not wire-compatible with the TypeSafe SDK (`G:docs/TYPESAFE_CONTRACT.md:10-14`). The Jev route in `teachers.mjs` uses the Vercel `ai` SDK's `experimental_evaluate` (`G:scripts/teachers.mjs:1`).
- CUDA only (`--disable-native-triton` flag). No published latency figure for the release.

## Licence
- Code: **MIT** (`G:LICENSE`, `G:SOURCE_MANIFEST.json` `license_scope`). The HF model card has **no licence field** (`HF:api.json` cardData). The base Qwen3-0.6B is Apache-2.0. Since part of the targets are Jev outputs, TypeSafe's terms may apply to the distilled behaviour (inferred, not stated).

## What is novel vs Jev and kev
1. **Per-candidate rows plus a set-attention head.** Unlike Kev's packed options read by a pointer from `<decide>`, each candidate is encoded alone with the state and question, then a small permutation-equivariant MHA mixes the candidate vectors with `log K`. Option-order invariance holds by construction (Jev and Kev are order-sensitive), but the cost is K full passes per choice, with the state repeated K times.
2. **Boolean as one path + sigmoid**; score as independent level leaves. This avoids Kev's `[no, yes]` two-option encoding.
3. **Games as the decision domain.** It is the only prominent clone that tests Jev-style typed decisions as an RL-style action policy, and it beats Jev on ViZDoom after training on Jev's own Maze/Snake distributions plus expert shooting data.
4. **Explicit Jev distillation.** Kev uses hard labels only; NanoJev trains directly on Jev probabilities via the Vercel gateway.
5. **RLCD research.** It contains the most careful public derivation of an RLCD-like sampled proper-scoring policy gradient, though it is not in the release. There is no temperature scaling at all.
6. **Not a general Jev replacement.** Game-only weights, a non-Jev API (`/api/evaluate`, `boolean`), no confidence or legend, Python-repr JSON state, no shared prefix.
