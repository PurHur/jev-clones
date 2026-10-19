# Standard One — full benchmark report

> **Note.** The v2 figures for both sizes are on their model cards (Standard One 8B and Standard One 3B). The benchmark figures in
> this report are still those of v1.1; the training-data provenance and overlap-audit sections already describe the v2 weights.

This report covers both released systems, **Standard One 8B** (`StandardThinking/StandardOne-8B` / `-LoRA`) and
**Standard One 3B** (`StandardThinking/StandardOne-3B` / `-LoRA`). It is our own measurement, not an official JevBench
leaderboard score. The only comparisons in this report are: the untuned
base model (accuracy, see § Decision quality vs. the untuned base), Qwen-based models served on the
same H200 stack (speed only, untuned, no accuracy comparison implied), and one authorized peer
comparison against TypeSafe Jev 1.13 (see § Measured against Jev 1.13).

## Contents

- [Suite name key](#suite-name-key)
- [Scope](#scope)
- [Measurement paths: offline HF runner vs. served endpoint](#measurement-paths-offline-hf-runner-vs-served-endpoint)
- [Text suites](#text-suites)
- [Served parity](#served-parity)
- [Public classification and decision suites](#public-classification-and-decision-suites)
- [Latency and throughput](#latency-and-throughput)
- [Measured against Jev 1.13](#measured-against-jev-113)
- [Calibration](#calibration)
- [Decision quality vs. the untuned base](#decision-quality-vs-the-untuned-base)
- [Data provenance and licences](#data-provenance-and-licences)
- [Training data provenance](#training-data-provenance)
- [Public-overlap audit and disclosure](#public-overlap-audit-and-disclosure)
- [Evidence file hashes](#evidence-file-hashes)

## Suite name key

The model cards use readable names for four internal evaluation suites; this report keeps the original
identifiers so figures trace back to `release-manifest.json` and `evidence/`. Mapping: `judge-proxy-v1`
= judge proxy (600 questions: routing + answer adequacy); `internal-eval-v2-realistic` = realistic
transfer set (600); `distprob-val-v2` = stated-distribution probability (1,036);
`internal-eval-hardproxy-v1` = hard proxy (600). JevBench's own public-tier names (easy / standard /
hard) are used as-is in both places.

## Scope

- Both candidates: merged BF16 weights, LoRA r=16/α=32 on `q/k/v/o/gate/up/down` of the language model
  only (vision tower + projector excluded), same five-stage training recipe on the **identical final
  training mixture** (34 cohorts, 382,576 rows) — see `MODEL_CARD_8B.md` / `MODEL_CARD_3B.md` §
  *Training* for the recipe and per-cohort breakdown, and `release-manifest.json` for the exact
  per-cohort row counts and hashes. The 3B is a smaller base/adapter pair, not a reduced-data variant.
  **Image input is supported** (the Pixtral vision tower and projector are retained; images are passed
  as data URLs), and image-side data was included in the final training stage — this report does not
  detail image or content-moderation training or results further.
- Release serving configuration for both: native wording, **no system prompt**
  (`--prompt-wording native --native-system-prompt none`), single option order, server-side default
  temperature **T = 1.65 (8B)** / **T = 1.55 (3B)** — both refit directly on served probabilities
  (see § Calibration). See `QUICKSTART.md` for the exact commands.
- What is **not** covered here: any official JevBench sealed-set score (we have none yet), any additional peer-system comparison beyond Jev 1.13, and image/content-moderation
  training detail beyond the one sentence above.

## Measurement paths: offline HF runner vs. served endpoint

Two different measurement paths appear throughout this report and are not interchangeable:

1. **Offline HF runner**: `transformers`-based scoring harness, computes label-logit probabilities
   directly from a loaded checkpoint (adapter or merged), no network serving stack involved.
2. **Served endpoint**: stock SGLang 0.5.20 + `jev-adapter`, `POST /v1/systemone`, the same path an
   external caller would use — see `QUICKSTART.md` for how to reproduce this endpoint.

**These two paths do not agree exactly.** Served probabilities differ measurably from offline-runner
probabilities for both candidates: mean total-variation distance ≈0.07–0.09 on the hard tier, with 8–11 of
111 argmax label changes, even though the rendered prompts were verified byte-identical between the two
paths. **Treat served numbers as authoritative and offline numbers as indicative.**

Temperature is **final and served-path-fit** for both candidates: T = 1.65 (8B) and T = 1.55 (3B) were
refit directly on served probabilities from the running endpoint, on
`internal-eval-hardproxy-calib-v1` + `distprob-calib-v2`, no system prompt. This supersedes an earlier
offline-harness fit (T = 1.55 / T = 1.5).

Most of the offline text-suite table below reflects the **with-system-prompt** configuration, while the
release ships **without** the system prompt. Where we independently re-measured the no-system-prompt
configuration (hard, original, judge-proxy-v1, internal-eval-v2-realistic, distprob-val-v2, easy), both
columns are shown.

## Text suites

### Standard One 8B

| Suite | n | With system prompt | No system prompt (shipped) | Notes |
|---|---|---|---|---|
| JevBench public easy | 48 | 100.00 % | **100.00 %** | |
| JevBench public original | 72 | 95.83 % | **97.22 %** | |
| JevBench public hard | 111 | 51.35 % | **52.25 %** | never used for training or T-fitting; informed the prompt-wording choice |
| judge-proxy-v1 (routing + adequacy) | 600 | 89.00 % | **88.83 %** | leaderboard judge text not public; proxy only |
| internal-eval-v2-realistic | 600 | 91.33 % | **90.00 %** | |
| distprob-val-v2 | 1,036 | 80.89 % | **81.56 %** | |
| internal-eval-v1 | 600 | 76.50 % | not separately re-measured | |
| internal-eval-hardproxy-v1 | 600 | 53.00 % | not separately re-measured | hard-tier proxy |
| games-harness-eval-v1 | 360 | 96.94 % | not separately re-measured | othello/chess/tic-tac-toe/snake/graph, text |
| tetris-drought-eval-v1 (text) | 120 | 67.50 % | not separately re-measured | |

### Standard One 3B

| Suite | n | With system prompt | No system prompt (shipped) | Notes |
|---|---|---|---|---|
| JevBench public easy | 48 | 100.00 % | **100.00 %** | |
| JevBench public original | 72 | 93.06 % | **93.06 %** | unchanged with/without system prompt |
| JevBench public hard | 111 | 48.65 % | **46.85 %** | never used for training or T-fitting; informed the prompt-wording choice |
| judge-proxy-v1 (routing + adequacy) | 600 | 89.00 % | **88.83 %** | |
| internal-eval-v2-realistic | 600 | 90.00 % | **90.67 %** | |
| distprob-val-v2 | 1,036 | 81.18 % | **79.83 %** | |
| internal-eval-v1 | 600 | 71.67 % | not separately re-measured | |
| internal-eval-hardproxy-v1 | 600 | 43.50 % | not separately re-measured | hard-tier proxy |
| games-harness-eval-v1 | 360 | 96.39 % | not separately re-measured | |
| tetris-drought-eval-v1 (text) | 120 | 68.33 % | not separately re-measured | |

## Served parity

Both offline and served measurement paths were run; served numbers below are authoritative, on the
final served-path temperature fit:

| | Standard One 8B (no system prompt, T=1.65) | Standard One 3B (no system prompt, T=1.55) |
|---|---|---|
| easy (48) | 100.00 % | 100.00 % |
| original (72) | 94.44 % | 88.89 % |
| hard (111) | 54.95 % | 45.95 % |
| judge-proxy-v1 | 90.50 % | 89.00 % |
| internal-eval-hardproxy-v1 | 52.50 % | 42.17 % |
| internal-eval-v2-realistic | 90.33 % | 90.83 % |
| distprob-val-v2 | 81.37 % | 80.50 % |
| Calibration axis (leaderboard formula) | 78.1 | 73.3 |

As noted above, served numbers differ somewhat from the corresponding offline numbers for both
candidates (mean TV ≈0.07–0.09 hard tier, 8–11/111 argmax changes) — these are the figures used for the
Accuracy does not depend on temperature, so the earlier T=1.0-forced served run
(before the temperature refit) reported the same easy/original/hard accuracy: 8B 100.00/94.44/54.95, 3B 100.00/88.89/45.95 no-system-prompt.

## Public classification and decision suites

400 cases per suite, seed 13, sampled from public test splits, one request per case, served endpoints
(merged BF16, SGLang 0.5.20, `jev-adapter`, native wording, no system prompt, single option order),
measured 25 September 2026. Standard One only; no comparison column.

| Suite | Standard One 8B | Standard One 3B |
|---|---|---|
| AG News (4 topics) | 84.2 % | 83.0 % |
| DAIR Emotion (6 labels) | 56.5 % | 58.8 % |
| Typed decisions (400 cases, 2,000 decisions) | 69.3 % | 69.0 % |
| MASSIVE intent, 20 options, 9 languages, mean (en 88.0 / 85.5, ko 88.2 / 83.8) | 85.6 % | 83.3 % |
| XNLI, 3 labels, 6 languages, mean | 60.1 % | 62.0 % |
| Email spam | 92.2 % | 93.0 % |
| Phishing | 86.0 % | 89.8 % |
| Moderation (toxic vs benign) | 85.0 % | 81.5 % |
| Model routing (domain) | 93.0 % | 83.7 % |

Option-order flip rate (share of cases whose answer changes under a permuted option order) ranges from
3.3 % (AG News) to 13 % (MASSIVE en) for the 8B. Full per-language and per-workflow numbers, macro-F1
and ECE: [`public-classification-suites.md`](public-classification-suites.md).

## Latency and throughput

![Bar chart of raw p50/p95 decision latency (ms) on one H200, SGLang 0.5.20, serial, 242-decision profile: Standard One 3B 22.6/33.2 ms, Standard One 8B 25.8/41.9 ms, Qwen3-8B 28.5/57.2 ms, Qwen3.5-4B 48.8/72.7 ms.](assets/02-latency-vs-qwen.png)

*Same H200, SGLang 0.5.20, `jev-adapter`, 242-decision profile, serial, ≈280 input tokens/decision.
Qwen checkpoints are served **untuned**, as a **speed-only reference — no accuracy comparison is
implied**.*

| Model | p50 | p95 | Input tokens |
|---|---|---|---|
| Standard One 3B | 22.6 ms | 33.2 ms | ≈280 |
| Standard One 8B | 25.8 ms | 41.9 ms | ≈280 |
| Qwen3-8B (untuned) | 28.5 ms | 57.2 ms | 278 |
| Qwen3.5-4B (untuned) | 48.8 ms | 72.7 ms | 283 |

242-decision profile, one H200, no system prompt:

| | Standard One 8B | Standard One 3B |
|---|---|---|
| Raw p50 / p95 | 25.8 ms / 41.9 ms | 22.6 ms / 33.2 ms |
| Leaderboard-adjusted (×2 + 0.15 s) p50 / p95 | 0.202 s / 0.234 s | 0.195 s / 0.216 s |
| Speed axis | 93.3 | 93.7 |
| Mean input tokens/decision | 280 | 280 (hard tier alone: 1,242) |
| With system prompt: mean tokens / Speed axis | 803 / — | 803 / 93.6 |

Throughput at concurrency 8, same 242-decision/280-token profile: **≈40 decisions/s (32.3k tokens/s)**
on one H200, measured for the 8B only. On the heavier hard+standard, 1,322-token/request mix (below):
**29.3k tok/s at concurrency 1 → 39.5k tok/s at concurrency 64**.

![Line chart of Standard One 8B prefill throughput on one H200 rising from 29.3k tokens/s at concurrency 1 to 39.5k tokens/s at concurrency 64, with p50 latency growing from 31 ms to 1,998 ms over the same range.](assets/04-throughput.png)

*Standard One 8B merged BF16, SGLang 0.5.20, one H200, **hard+standard mix at 1,322 tokens/request** — a
heavier, different profile than the 280-token/decision figure above, so the two throughput numbers
(37.0k tok/s here at concurrency 8 vs. 32.3k tok/s above) are not directly comparable; both are genuine
measurements of the same hardware under different request mixes. **8B only.**

| Concurrency | Throughput (k tok/s) | p50 (ms) |
|---|---|---|
| 1 | 29.3 | 31 |
| 8 | 37.0 | 282 |
| 32 | 39.3 | 1,033 |
| 64 | 39.5 | 1,998 |

## Measured against Jev 1.13

![Grouped bar chart of accuracy by suite for Standard One 8B, Standard One 3B and Jev 1.13, covering stated-distribution probability, realistic transfer set, judge proxy, JevBench easy/standard/hard and the hard-tier proxy.](assets/06-vs-jev.png)

The one peer comparison in this report: we measured **TypeSafe Jev 1.13** (`typesafe/jev-1.13-20260917`)
through OpenRouter's `/v1/systemone`, on the same items, same option order, raw probabilities as
returned (no temperature applied), 3,067 questions, 0 failures, measured 24 September 2026 from the
same host used for our own runs.

| Suite (n) | Jev 1.13 | Standard One 8B | Standard One 3B |
|---|---|---|---|
| distprob-val-v2 (1,036) | 72.97 % | **81.37 %** | 80.50 % |
| internal-eval-v2-realistic (600) | 86.67 % | 90.33 % | **90.83 %** |
| judge-proxy-v1 (600) | **90.50 %** | **90.50 %** | 89.00 % |
| JevBench easy (48) | 100.00 % | 100.00 % | 100.00 % |
| JevBench standard (72) | **98.61 %** | 94.44 % | 88.89 % |
| internal-eval-hardproxy-v1 (600) | **54.83 %** | 52.50 % | 42.17 % |
| JevBench hard (111) | **72.07 %** | 54.95 % | 45.95 % |

Jev leads on the public hard tier and the hard proxy; Standard One leads on stated-distribution probability and
the realistic transfer set; judge and easy tie.

Calibration on the same items, Jev raw (no temperature applied):

| | Jev 1.13 | Standard One 8B | Standard One 3B |
|---|---|---|---|
| Hard-tier ECE (10-bin, top label) | **0.099** | 0.231 raw (0.147 at T=1.65) | 0.286 raw (0.198 at T=1.55) |
| Mean TV to stated distributions | **0.192** | 0.109 raw (0.145 at T) | 0.112 raw (0.139 at T) |

Latency, same 242-decision profile, same host:

| | p50 | p95 | Input tokens/decision |
|---|---|---|---|
| Jev 1.13 (hosted API via OpenRouter) | 227 ms | 321 ms | ≈538 (OpenRouter-reported) |
| Standard One 8B | 25.8 ms | 41.9 ms | ≈280 |
| Standard One 3B | 22.6 ms | 33.2 ms | ≈280 |

Jev's latency includes network round-trip to a hosted API; Standard One's is local-GPU serial on the same host.

## Calibration

- **T fit (release, no system prompt, served path, final)**: 8B T = 1.65, 3B T = 1.55, both refit
  directly on served probabilities from the running endpoint, on
  `internal-eval-hardproxy-calib-v1` + `distprob-calib-v2`. This is the final, shipped configuration.
- **Historical reference (offline-harness fit, superseded)**: 8B T = 1.55 (no system prompt) / T = 1.45
  (with system prompt, merged-checkpoint offline check, hard-tier ECE 0.269 at T=1, 0.206 at T=1.45);
  3B T = 1.5 (no system prompt) / T = 1.35 (with system prompt, merged-checkpoint offline check,
  hard-tier ECE 0.269 at T=1, 0.212 at T=1.35). These predate the served-path refit and are kept for
  record only.
- **Calibration axis (leaderboard formula)**: 8B **78.1**, 3B **73.3** (see § Served parity).
- **Distribution (TV) fidelity**, `distprob-val-v2`, offline, with system prompt, at T=1: 8B mean TV ≈0.103; 3B mean TV ≈0.110. See § *Measured against Jev 1.13* for the
  served-path TV comparison against Jev.

## Decision quality vs. the untuned base

![Diverging bar chart of accuracy change vs the untuned base per suite, Standard One 8B and 3B, ordered by gain.](assets/05-gain-over-base.png)

*Accuracy change versus the untuned Ministral 3 base, same prompt, same offline runner, no system prompt, T=1. Public JevBench tiers are the 231 public items; the other suites are held-out proxies built from public data. The hard tier is placed last on purpose (see below).*

The only base-model comparison in this report. On the public **hard** tier, the untuned Ministral 3 8B
base is measured **ahead of Standard One 8B** (60.4 % vs. 52.3 %, a 9-of-111-item gap) — hard is 30 % of the
v1.3.0 Intelligence axis, so this is material, not a rounding artifact. All rows below come from one
run of the same offline runner (native wording, no system prompt, T=1, single option order); the untuned 3B base is behind on every suite except the public hard tier (by 1 of 111 items), the untuned 8B base is ahead only on the public hard tier.

| Suite | Base 8B | **Standard One 8B** | Base 3B | **Standard One 3B** |
|---|---|---|---|---|
| distprob-val-v2 (stated-distribution probability, 1,036) | 34.85 % | **81.56 %** (+46.7) | 31.56 % | **79.83 %** (+48.3) |
| internal-eval-v2-realistic (realistic transfer set, 600) | 72.83 % | **90.00 %** (+17.2) | 62.50 % | **90.67 %** (+28.2) |
| JevBench public standard tier (72) | 79.17 % | **97.22 %** (+18.0) | 70.83 % | **93.06 %** (+22.2) |
| judge-proxy-v1 (routing + adequacy, 600) | 79.33 % | **88.83 %** (+9.5) | 63.17 % | **88.83 %** (+25.7) |
| JevBench public easy tier (48) | 100.00 % | **100.00 %** (+0.0) | 97.92 % | **100.00 %** (+2.1) |
| JevBench public hard tier (111) | 60.36 % | **52.25 %** (-8.1) | 47.75 % | **46.85 %** (-0.9) |


## Data provenance and licences

| Source | Licence |
|---|---|
| Every synthetic cohort (deterministic generators + visible checkers; subset teacher-reviewed) | Apache-2.0 (ours) |
| Base models `mistralai/Ministral-3-8B-Instruct-2512(-BF16)` / `-3B-` | Apache-2.0 |
| `jev-adapter` server code | Apache-2.0 |
| JevBench harness + public tiers (evaluation only, commit `fd51755e…`) | MIT |

All training-cohort details (per-cohort row counts, sha256) are in § *Training data provenance* below
and in `release-manifest.json`, identical for both candidates.

## Training data provenance

The following cohort families make up the 46-cohort, 520,754-row training mixture used by
both Standard One 8B and Standard One 3B. Apart from the public-dataset items listed below, every row is synthetic (deterministic generators with visible checkers; a subset was teacher-reviewed) or format-augmented. No JevBench evaluation scenario appears in any
training cohort.

| Cohort family | What it covers | Licence |
|---|---|---|
| Document & field normalisation (`document-v3`, `exact`, `field-roles`, `synthetic-en`) | Extracting and normalising fields from documents and records | Apache-2.0 (ours) |
| Judge / routing & adequacy (`judge-proxy-v1`, `judge-realistic-train`, `judge-multilingual-train`) | Routing decisions and answer-adequacy checks | Apache-2.0 (ours) |
| Stated-distribution probability (`distprob-scale30k`, `distprob-multilingual`, `distprob-pilot2k`, `narrative-prob-train`, `probability`) | Estimating a probability distribution over stated outcomes | Apache-2.0 (ours) |
| Verification & scoring (`noul-score-10k`) | Yes/no verification and ordinal scoring | Apache-2.0 (ours) |
| Adequacy-rubric style (`hardstyle-train`) | Hard-style adequacy questions (see § Public-overlap audit and disclosure) | Apache-2.0 (ours) |
| Abstention & robustness (`abstain`, `paraphrase`, `trap`, `hard-natural`) | Abstaining when the state does not settle the question, paraphrased and adversarial wording, harder natural-language decisions | Apache-2.0 (ours) |
| Game & board-state (`games-text-train`, `games-harness-train`, `games-harness-multilingual`) | Othello, chess, tic-tac-toe, snake and graph-problem decisions | Apache-2.0 (ours) |
| Tetris board-state (`games-text-train-tetris`, `games-harness-tetris`, `tetris-drought-train`) | Tetris-specific board-state decisions | Apache-2.0 (ours) |
| Synthetic and format-augmented decision data (v2) | Decision items in varied layouts | Apache-2.0 (ours) |
| Public-dataset decision items | Decision items converted from public datasets (table below) | Each dataset's own licence (table below) |
| JevBench public tiers (evaluation only, not training) | Easy / standard / hard held-out accuracy measurement | MIT |

Public datasets used (train splits where a dataset has one; CUAD: its single release file). Labels come from the datasets;
distractor options and derived labels are generated by code. Licences as stated by each dataset; attribution to the dataset authors.

| Dataset / upstream id | Licence | Cohort |
|---|---|---|
| SQuAD 2.0<br>`rajpurkar/squad_v2` | CC BY-SA 4.0 | `pub-abstain-v1` |
| ARC<br>`allenai/ai2_arc` | CC BY-SA 4.0 | `pub-breadth-v1` |
| BoolQ<br>`google/boolq` | CC BY-SA 3.0 | `pub-breadth-v1` |
| CommonsenseQA<br>`tau/commonsense_qa` | MIT | `pub-breadth-v1` |
| HellaSwag<br>`Rowan/hellaswag` | MIT | `pub-breadth-v1` |
| Banking77<br>`PolyAI/banking77` | CC BY 4.0 | `pub-classify-v1` |
| Bias in Bios<br>`LabHC/bias_in_bios` | MIT | `pub-classify-v1` |
| Bitext customer support<br>`bitext/Bitext-customer-support-llm-chatbot-training-dataset` | CDLA-Sharing-1.0 | `pub-classify-v1` |
| CLINC150<br>`clinc/clinc_oos` | CC BY 3.0 | `pub-classify-v1` |
| Amazon Counterfactual<br>`mteb/amazon_counterfactual` | CC BY 4.0 | `pub-classify-v1` |
| DBpedia-14<br>`fancyzhx/dbpedia_14` | CC BY-SA 3.0 | `pub-classify-v1` |
| Dolly 15k<br>`databricks/databricks-dolly-15k` | CC BY-SA 3.0 | `pub-classify-v1`, `safety-judge-v1` |
| GoEmotions<br>`google-research-datasets/go_emotions` | Apache-2.0 | `pub-classify-v1` |
| MASSIVE<br>`AmazonScience/massive` | CC BY 4.0 | `pub-classify-v1` |
| Twitter Financial News Sentiment<br>`zeroshot/twitter-financial-news-sentiment` | MIT | `pub-classify-v1` |
| HelpSteer3<br>`nvidia/HelpSteer3` | CC BY 4.0 | `pub-judge-v1` |
| HelpSteer2<br>`nvidia/HelpSteer2` | CC BY 4.0 | `pub-judge-v1` |
| 2WikiMultihopQA<br>`framolfese/2WikiMultihopQA` | Apache-2.0 | `pub-multihop-v1` |
| HotpotQA<br>`hotpotqa/hotpot_qa` | CC BY-SA 4.0 | `pub-multihop-v1` |
| MuSiQue<br>`dgslibisey/MuSiQue` | CC BY 4.0 | `pub-multihop-v1` |
| QASC<br>`allenai/qasc` | CC BY 4.0 | `pub-multihop-v1` |
| DROP<br>`ucinlp/drop` | CC BY-SA 4.0 | `pub-numeric-v1` |
| GSM8K<br>`openai/gsm8k` | MIT | `pub-numeric-v1` |
| TempReason<br>`tonytan48/TempReason` | CC BY-SA 3.0 | `pub-numeric-v1` |
| MultiNLI<br>`nyu-mll/multi_nli` | OANC / CC BY-SA 3.0 / CC BY 3.0 | `pub-paraphrase-v1` |
| PAWS<br>`google-research-datasets/paws` | Google terms, free for any purpose | `pub-paraphrase-v1` |
| PAWS-X<br>`google-research-datasets/paws-x` | Google terms, free for any purpose | `pub-paraphrase-v1` |
| SNLI<br>`stanfordnlp/snli` | CC BY-SA 4.0 | `pub-paraphrase-v1` |
| WANLI<br>`alisawuffles/WANLI` | CC BY 4.0 | `pub-paraphrase-v1` |
| ContractNLI<br>`stanfordnlp.github.io/contract-nli` | CC BY 4.0 | `pub-rules-v1` |
| CUAD<br>`theatticusproject/cuad` | CC BY 4.0 | `pub-rules-v1` |
| ShARC<br>`sharc-data.github.io` | CC BY-SA 3.0 | `pub-rules-v1` |
| Jailbreak classification<br>`jackhhao/jailbreak-classification` | Apache-2.0 | `safety-judge-v1` |
| Prompt injections<br>`deepset/prompt-injections` | Apache-2.0 | `safety-judge-v1` |
| Aegis AI Content Safety 2.0<br>`nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | CC BY 4.0 | `content-safety-text-v1` |
| Jigsaw Toxic Comment Classification (mirror of the Kaggle data)<br>`tasksource/jigsaw_toxicity` | CC0 (data); comment text CC BY-SA 3.0 (Wikipedia) | `content-safety-text-v1` |
| Measuring Hate Speech<br>`ucberkeley-dlab/measuring-hate-speech` | CC BY 4.0 | `content-safety-text-v1` |
| Image safety classes<br>`deepghs/nsfw_detect` | MIT | `content-safety-image-v1`, `content-safety-image-2d-v1` |

## Public-overlap audit and disclosure

Run on the 46-cohort, 520,754-row training data that **both candidates train on**:

- **0 exact state matches** — no training row's state/scenario duplicates any public JevBench item.
- **181 exact instruction matches**: `hardstyle-train-v1` (87 of 3,900 rows) and `hardstyle-train-v2`
  (94 of 3,879 rows) contain rows whose entire instruction is byte-identical, after normalization, to
  one public `jevbench-hard` instruction — a 58-character generic adequacy-rubric question ("does the
  response fully and correctly satisfy the request?"), reused as the complete instruction rather than
  as a substring.
- 24 distinct shared word 8-grams (20,400 rows), all in known families: the rubric sentence above (also in `judge-realistic-train-v3`, `judge-multilingual-train-v1`, `judge-proxy-v1` and the format-augmented cohort), an insurance "cancellation takes effect" boilerplate clause, a "coverage reviewer" claims-adjudication phrase, and generic contract boilerplate (for example "controlled by or under common control with") in one public-dataset cohort.
- All other 44 cohorts: 0 exact state and 0 exact instruction matches.

**These 181 rows are kept and disclosed here, not regenerated.** They are 0.03 % of the mixture, and
the exposure is limited to one hard-tier *question's* generic wording, never a *state* or an answer.

## Evidence file hashes

`evidence/` (8B):

| File | sha256 |
|---|---|
| `adapter_model.safetensors` | `9551923c162970ea8bc2c3a5faf3ea3762b48591c9f460064e531d09427fa41a` |
| `adapter_config.json` | `545101ebcafd9fb90723284518d744ca01de46534e04a5dfd4e8f86bc4de0ef2` |
| `manifest.json` | `62bd99b6d4a49926840c258874a3b804a868626761fa57a39e8bfe10e3192904` |
| `MERGE_REPORT.json` | `1f39839e969d874d397d3842eb702f39516ec784f546ce18220e9836bffb2793` |
| `calibration-leaderboard.json` | `a28ada7681a3ae8edcce3f7885dec63ecfb9bc6a9c9794204fa040ae1dd1065f` |
| `latency-report-shared-gpu.json` | `f6734e5cb4a210e9284632ef97c922876f4aaced3106461ec9cede475a640fa9` |
| `served-easy-report.json` | `f68aa8f1d1bbc60b045592713188563643e07d0e8d8337df4df9387bf87254f3` |
| `served-original-report.json` | `b87e0a9a48ebb9fa933fe0db470b506cbe1850824abf65dfebf6ad6c5f58cdda` |
| `served-hard-report.json` | `c0eb0fde38c44b1f3f62f6a79cc45c3e89cad1f782d1c6cc40526f69e8595ff6` |

`evidence/` (3B):

| File | sha256 |
|---|---|
| `adapter_model.safetensors` | `7bc8bfff4be52e0c9c6f605faf1eebc5e8a966ce103793c7f6970e29056ef5d3` |
| `adapter_config.json` | `43b99a646a535b15e0f8bceebe2cfff03dd8cc519f5e21f68856b885657d7179` |
| `manifest.json` | `ecfd448e212e09df55c86c93233abe857282b8d9d6154dceddb681663b6ebea2` |
| `calibration-leaderboard.json` | `de13ebd925123ade9f065d7ae3a3955e7f55ddc29ce1d598624039fcdc75567e` |
| `offline-native-result.json` | `2a53c2dc7c82dcf971110e2f788c750c63a09c1c0ea844a371e6abdd087adf24` |

`docs/assets/` (figures referenced by this report):

| File | sha256 |
|---|---|
| `02-latency-vs-qwen.png` | `d383eb64f146c6be069455d1631e10ef120ffb534f0000f7470e715e9fe99872` |
| `04-throughput.png` | `bdb8cb392f6f288fe7ce7fde78e39759cd688340be4fe605890dc79c58e83f73` |

The 3B evidence set does not yet include a separate `MERGE_REPORT.json` or served-endpoint report
files; the merge-tensor numbers (182 tensors changed, max |Δ| 0.0023) and served numbers (§ Served
parity) are recorded narratively in `MODEL_CARD_3B.md`.
