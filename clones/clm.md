# CLM: Contrastive Language Models (Contrastive-LM/CLM)

A dual-encoder ("two-tower") System One model. A **frozen Qwen3-8B** embeds the state (with the question) and each candidate action separately, using last-token pooling. Two small MLP projection heads map both embeddings into a 512-d space, and the answer is a softmax over scaled cosine similarities. Trained with InfoNCE. It serves TypeSafe's `POST /v1/systemone` shape. Compiled 2026-10-19.

Source pins:

| ref | what | where |
|---|---|---|
| `G` = `Contrastive-LM/CLM` @`d5f9ef0f` (2026-10-05, "Revise citation title in README") | shallow, blob-filtered clone; read only, never executed | `/home/ai/projects/jev-clones/sources/clm` |
| `HF` = `Contrastive-LM/CLM-v0.1-8B` @`e939398d` | api.json, tree.json, README, config.json (the `CLM_v0.1-8B.pt` head, 75.6 MB, was not downloaded) | `notes/hf-raw/Contrastive-LM__CLM-v0.1-8B/` |
| `HFD` = `Contrastive-LM/deepswe-clm-heads-8k` @`c60876f3` | README, config, split/verification/task JSON (no `best_head.pt`) | `notes/hf-raw/Contrastive-LM__deepswe-clm-heads-8k/` |
| Web | HN 49890520 (2 points, no comments; links VentureBeat "Stanford and Nvidia's open Jev-like model"); secondary press (completeaitraining.com citing VentureBeat, cryptobriefing, pasqualepillitteri.it); blog contrastive-lm.notion.site (did not render for us) | — |

---

## Identity

| | |
|---|---|
| Repo | github.com/Contrastive-LM/CLM. Created 2026-09-23, ★3,072, Apache-2.0 (`gh api`). PyPI `contrastive-lm` (`G:README.md` Installation) |
| Authors | Jacky Kwok, Hangoo Kang, Tarun Suresh, Jon Saad-Falcon, Marco Pavone, Christopher Ré, Azalia Mirhoseini (`G:README.md` Citation). Press calls it "Stanford and Nvidia" (Pavone is Stanford/NVIDIA; Ré and Mirhoseini are Stanford) |
| Paper | **None.** The citation is a Notion blog (`@misc … note={Notion Blog}`). No arXiv ID was found |
| Weights | `Contrastive-LM/CLM-v0.1-8B`: heads only, 4,989 downloads, 780 likes, created 2026-09-21. `Contrastive-LM/deepswe-clm-heads-8k`: a task-fine-tuned head, MIT. Embedding datasets: `Contrastive-LM/deepswe-clm-embeddings-8k` (`G:README.md`) |
| Model names | `clm-latest` (release date 2026-09-19 per `/v1/models`), `clm-raw` (ablation: cosine in the raw encoder space) |
| Roadmap | Larger backbones and multimodal (`G:README.md` Roadmap). Press mentions a "CLM-35B-A3B" multimodal MoE for early October; nothing was found on HF (unverified) |
| Decision Index 0.3.2 | `CLM-v0.1-8B`, kind "full fine-tune", technique "autoregressive", 8.19B, base `Qwen/Qwen3-8B-Base`. **This conflicts with the card and code**: frozen `Qwen/Qwen3-8B` (not -Base), heads only (`HF:config.json`; `G:src/clm/heads.py`) |

## Architecture

### Backbone
- `Qwen/Qwen3-8B`, **frozen**, served by vLLM as a pooling model (`vllm serve Qwen/Qwen3-8B --runner pooling`). Last-token pooling gives a 4096-d normalised embedding (`HF:config.json`; `G:README.md` Quickstart; `G:src/clm/embedder.py`).
- The backbone gets no LoRA and no fine-tuning. Only the heads train ("Only the heads are trained, so fine-tuning is cheap", `HF:README.md`).

### Input layout
- **State tower input** = `to_text(state) + "\n\n" + to_text(instructions)`, "context first, question last". Objects are rendered as `key: value` prose and arrays as `- item` lines, "never JSON, the heads are trained on prose" (`G:src/clm/schema.py:27-65`; `G:README.md` API).
- **Action tower input** is set per candidate (`G:src/clm/schema.py:75-101`):
  - choice: the option's **description if present, else its key**, verbatim. The key is not combined with the description.
  - score: each level's text.
  - noul: `"true: Yes. This is true: {instructions}"` / `"false: No. This is false: {instructions}"`, or the caller's true/false criteria.
- Candidates never see the state, and the state never sees the candidates. This is late interaction at the vector level (a bi-encoder).

### Attention & question isolation
- **Total isolation by construction.** Every question is a separate state-tower call, and every candidate is a separate action-tower call. No attention crosses questions, options, or state and option.
- Option order cannot matter (permutation-invariant), but options also cannot condition on the state's details or on each other (inferred: no "none of the above" reasoning, and weak on fine-grained conditional decisions).
- Each question re-embeds the whole state, because the question is appended to it. The state is not shared across questions inside the encoder; only exact repeats hit the cache (inferred from `build_pairs`, `G:src/clm/schema.py:104-112`).

### Readout head (`G:src/clm/heads.py`)
```
s   = state_head(E(state ⊕ question))      MLP 4096 → width → … → 512 (GELU; optional LayerNorm/residual)
a_j = action_head(E(candidate_j))          separate MLP, same shape
z_j = exp(logit_scale) · cos(s, a_j)        logit_scale = log(1/τ), init log(1/0.07), clamped ≤ 100
p   = softmax_j(z_j / T_request)            T_request optional, (0,100], default 1
```
- README: "a 20M-parameter trainable projection head" per encoder. The checkpoint is 75.6 MB (`HF:tree.json`), which is about 18.9M params at fp32 for *both* heads, so either the heads are smaller than stated or they are stored in 16-bit (inferred; the weights were not opened). One head pair serves all three question types.

### Question types
| type | candidates | answer | confidence |
|---|---|---|---|
| noul | 2 synthetic texts (false, true) | `noul = p_true` | none |
| choice | ≥1 option (no upper bound in schema) | argmax, probabilities | **p_max − mean(p_others)**, not TypeSafe's formula (`G:src/clm/schema.py:122-128`) |
| score | ≥2 ordered levels | `Σ i·p_i`, legend | same as choice |

The score head has no ordinal structure; levels are independent candidates.

### Limits
- States are **truncated at 2,048 tokens** by default. To raise the limit, raise both `--max-model-len` (vLLM) and `clm-serve --max-tokens` together (`G:README.md` Serve).
- No documented cap on candidate or question count. `billing_units` = number of questions.

### Calibration
- Calibration is learned only through the InfoNCE temperature (`logit_scale`). There is **no post-hoc calibration fit** and no calibration metric is published (ECE, Brier). The per-request `temperature` field is a caller knob (`G:README.md` API).
- InfoNCE probabilities are retrieval scores over in-batch negatives, not outcome-calibrated probabilities (inferred).
- The noul probability comes from a 2-way contrast between two synthetic sentences. Its calibration is unvalidated (inferred).

## Training
- **Loss**: bidirectional in-batch InfoNCE over B×B state–action similarities. Mid-training adds k hard negatives per state in the s→a direction (`G:README.md` Training Algorithm).
- **Data, three stages** (`G:README.md` Data Recipe):
  1. Pre-training: ~60M Nemotron DQA question→answer pairs.
  2. Mid-training: ~30M synthetic hard negatives from Gemini 2.5 Flash-Lite.
  3. Post-training: ~1M agent trajectories (Agent Data Protocol, Endless-Terminals, LiteCoder-Terminal-SFT), mixed 60% agentic / 40% Nemotron replay.
- **Ablations reported**:
  - Without replay, hard-negative top-1 accuracy drops from 69% to 56.2%; with replay it holds at 68.5%.
  - Pre-training then mid-training reaches 69.2%, vs 62.4% with hard negatives from the start. Pre-training alone gives 52.1%.
- **Scaling laws** (README, figures in the blog): test InfoNCE loss follows a power law in compute, data, head size and encoder size. Encoder size gives the biggest gains. At the compute-optimal point head size scales as N* ∝ D^1.02, about 310 tokens per parameter. Experiments use Nemotron DQA (`G:README.md` Scaling Laws).
- **Fine-tuning** (`G:train/finetune.py`): heads only, on precomputed embeddings.
  - Optimiser: AdamW, OneCycle cosine schedule, lr default `2e-3·sqrt(1024/width)·sqrt(B/1024)` (`:289-306`).
  - Tasks: `clm` (InfoNCE on trajectories) or `choice` (typed decisions, e.g. `LocalLLaMA/typed-decisions`). `--loss softce` uses a per-question softmax.
  - `docs/FINETUNING.md` is an "autonomous LLM fine-tuning" loop protocol (an agent edits `finetune.py` and logs to `results.tsv`).
- No Jev outputs are used in training (inferred; none are listed in the data recipe).

## Evals
- **Zero-shot vs Jev**: T-Rex, BFCL v4 tool calling, WikiRacing and Super Mario. Results are in a chart only (`G:assets/zero-shot.png`). The claim is "on par … up to 9× faster", with the biggest speedups at many candidates or reused actions.
- **T-Rex**: the only shipped raw numbers (`G:examples/t_rex/results/*_realtime.json`; 5 seeds × 60 s, 6 requests in flight, safety shield on):

  | | best score | survived | p50 latency | model p50 | decisions | agreement with planner | shield interventions |
  |---|---|---|---|---|---|---|---|
  | CLM (local) | 697 | 5/5 | **16.5 ms** | 2.6 ms | 3,341.8 | **0.658** | **4,883** |
  | Jev 1.13.0 (api.typesafe.ai) | 697 | 5/5 | 149.8 ms | 131.9 ms | 1,119 | 0.987 | 28 |

  The "parity" holds with the shield active. CLM agrees with the planner much less often and needed about 174× more shield overrides, so the 9× latency headline hides a large decision-quality gap on this task. Jev's latency also includes a WAN round trip (inferred from the JSON).
- **As a verifier (best-of-N)**:
  - DeepSWE: 31/38 = 81.6% with Bo4 Opus 5 candidates, task-disjoint 75/38 split, mean of the last 12 step scores. Pass@1 is 28/38 = 73.7% and the oracle is 34/38 = 89.5% (`HFD:verification.json`, `README.md`). That is +3 tasks over random pick on 38 tasks.
  - Terminal-Bench 2.1: 87.6% on 30 held-out tasks with Fable 5 candidates.
  - 4.1–5.7× faster than Jev. "Jev fails to serve as a verifier for these long-horizon tasks, scoring below pass@1" (`G:README.md` Agentic). The "SOTA" label is self-reported, and the sets are small.
- With about 1k candidates, CLM is claimed 13× faster than Jev (`HF:README.md`).
- No TypeSafe-style typed-decision accuracy, calibration, JevBench or Decision Index composite score is published (as of `G`).

## Serving
- Two processes:
  1. **vLLM** Qwen3-8B pooling server on the GPU (:8090, `/v1/embeddings`);
  2. **`clm-serve`** (FastAPI, :8700) holding the heads (CPU or GPU, hot-reloaded on file change).
- Endpoints: `POST /v1/systemone`, `POST /v1/rank`, `GET /v1/models`, `/health`, and a playground UI at `/`. Optional bearer auth via `CLM_API_KEY`; `X-CLM-Latency-Ms` header (`G:README.md` API Reference; `G:src/clm/server.py`).
- **Vector cache**: one device-memory slab reserved at start-up (default 2% of the device, "the way vLLM claims its KV cache") holds state and action projections, keyed by head generation, with LRU eviction.
  - RTX 4090 server-side p50: a new state costs ~28 ms; a revisited state costs 0.6–0.7 ms (3 or 50 actions).
  - `usage.input_tokens` counts only cache misses (`G:README.md` vector cache; `G:src/clm/cache.py`).
- The README example gives 58 ms latency for 3 questions on a warm cache.

## Licence
- Code: Apache-2.0 (`G:LICENSE`). CLM-v0.1-8B head: Apache-2.0 (`HF` cardData). The DeepSWE head is tagged MIT. The Qwen3-8B base (Apache-2.0) is fetched separately.

## What is novel vs Jev and kev
1. **A different paradigm: a dual encoder with InfoNCE**, not a cross-encoder or prefill-plus-pointer. State and action are embedded independently, so **action embeddings are cacheable across states and requests**. Cost is one encoder pass per new text plus a dot product. This is the only family here whose cost does not grow with K × state length.
2. **The frozen backbone is served by stock vLLM pooling**; the whole trainable model is ~75 MB of heads. Fine-tuning runs on cached embeddings in minutes, and new heads hot-swap.
3. **Agent-trajectory pre-training** (Nemotron QA → hard negatives → ADP traces) and **published scaling laws** for the contrastive verifier objective, which no other clone reports.
4. **Weak points vs Jev and kev**:
   - No cross-attention between state and options (inferred: weak on conditional or policy decisions, negation, and options that only make sense in context).
   - 2,048-token default state window.
   - Non-standard confidence formula, no calibration evidence, and noul built from synthetic true/false sentences.
   - The T-Rex parity relies on an external safety shield.
   - The registry entry (Decision Index) misdescribes it as a full fine-tune of Qwen3-8B-Base.
