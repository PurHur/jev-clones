---
license: apache-2.0
language:
- en
- ko
tags:
- ztc
- answer-verification
- hallucination-detection
- zero-token
- confidence-estimation
library_name: transformers
pipeline_tag: text-classification
---

<!-- ZTC-ECOSYSTEM:START -->
### ZTC ecosystem

[![Darwin--397B--ZTC](https://img.shields.io/badge/Model-Darwin--397B--ZTC-0e7b72?style=flat-square)](https://huggingface.co/FINAL-Bench/Darwin-397B-ZTC) ![ZTC--Judge--27B](https://img.shields.io/badge/Model-ZTC--Judge--27B-c9ccd1?style=flat-square&label=you%20are%20here&labelColor=444) [![Gate%20Arcade](https://img.shields.io/badge/Demo-Gate%20Arcade-9a7b1f?style=flat-square)](https://huggingface.co/spaces/FINAL-Bench/gate-tetris) [![Typed%20Decision%20Leaderboard](https://img.shields.io/badge/Leaderboard-Typed%20Decision%20Leaderboard-3f6f8f?style=flat-square)](https://huggingface.co/spaces/mayafree/typed-decision-leaderboard) [![Verifier%20Playground](https://img.shields.io/badge/Playground-Verifier%20Playground-5a6672?style=flat-square)](https://huggingface.co/spaces/mayafree/verifier-playground) [![vidraft.net](https://img.shields.io/badge/Home-vidraft.net-161d24?style=flat-square)](https://vidraft.net)

**Zero-Token Confidence (ZTC)** reads a model's own internal state to judge whether an answer is right — **generating zero tokens**. `Darwin-397B-ZTC` ships a self-readout probe; `ZTC-Judge-27B` judges other models' answers; the arcade shows the gate deciding *execute or hold* in real time.
<!-- ZTC-ECOSYSTEM:END -->

# ZTC-Judge-27B

**Ask a model how confident it is and you get a coin flip (AUC 0.500). This reads the same forward pass and gets 0.7289 on domains it has never seen — above the surface baseline in all five.**

ZTC-Judge-27B takes a question and an answer written by *any* model and scores whether that answer
can be trusted — **without generating a single token**.

> **ZTC** — Zero-Token Confidence
> **Judge** — it evaluates *someone else's* answer, not its own

---

## How it works

```
[question + answer]  →  one forward pass through the 27B model
                     →  final-layer hidden state at the last position (5,120-d)
                     →  one dot product with the probe
                     →  score
```

**Generated tokens: 0.** No access to the answering model's weights, logits or log-probabilities is
required — the text of the answer is the only input. Because nothing is decoded, latency is one
forward pass and batching translates directly into throughput.

## Usage

```python
from ztc_judge import ZTCJudge

judge = ZTCJudge.from_pretrained("FINAL-Bench/ZTC-Judge-27B")

judge.score(
    "Which defensive compound does the insect release when it meets a predator?",
    "C. Allomone",
    domain="scientific_reasoning",
)
# {'score': -0.61, 'verdict': 'review', 'domain_auc': 0.7410, 'generated_tokens': 0}
```

Batched scoring (recommended — there is no decode step, so batch size is throughput):

```python
judge.score_batch([(q1, a1), (q2, a2), ...], domain="scientific_reasoning")
```

The score is an unbounded real number; higher means more likely correct. It is a **ranking signal**,
not a calibrated probability. Pick a threshold from your own review budget (see below).

---

## Evaluation

Two independent measurements are reported. They use different test sets and different
protocols, and are **not** interchangeable — read the protocol line before quoting a number.

### A. Independent leaderboard — 2,018 items, leave-one-domain-out

The **Typed Decision Leaderboard** scores answer verifiers from several vendors on one identical
item set with identical labels: <https://huggingface.co/spaces/mayafree/typed-decision-leaderboard>

| System | AUC |
|---|---|
| Darwin-397B-ZTC | 0.7272 |
| JEV (TypeSafe AI) | 0.7350 |
| **ZTC-Judge-27B** | **0.7289** |
| GPT-5.2 asked directly | 0.7148 |
| open-jev 4B | 0.6844 |
| *Answer length and formatting only* | *0.6223* |

## A second probe ships with this model, for answers from models we have not seen

The table above is measured on a set where **79% of the answers come from one model family**. A probe
fitted on it can end up specialised to that family rather than to correctness. We checked, using a
model that appears in neither the fitting nor the scoring — Claude Haiku 4.5 — answering the same
2,018 questions. Those 1,939 answers were graded with the same code and the same domain weights.

| On answers from an unseen model | AUC |
|---|---|
| **`ztc_curve_probe_v2.npz`** | **0.7752** |
| JEV | 0.7521 |
| `ztc_curve_probe.npz` (the probe behind the table above) | 0.7349 |
| *Answer length and formatting only* | *0.4334* |

| Paired comparison | Difference | 95% interval | |
|---|---|---|---|
| **v2 over JEV** | **+0.0230** | **[+0.0056, +0.0412]** | separable |
| v2 over v1 | +0.0402 | [+0.0247, +0.0562] | separable |
| v1 over JEV | −0.0172 | [−0.0356, +0.0001] | not separable |

3,000 paired bootstrap resamples, weighted by domain.

**Which file to use.** `ztc_curve_probe.npz` produces the reported figure on the benchmark above and
is the reference for it. `ztc_curve_probe_v2.npz` is fitted on answers from several models and is the
one to use on answers written by a model that is not in that benchmark — which is the usual case in
deployment.

🔴 **The two do not agree, and the direction reverses.** Scored on the 2,018-question benchmark under
one identical method, v2 reaches 0.7066 where v1 reaches 0.7236. Specialisation helps on the
distribution it was specialised to and costs elsewhere. Both files are published so the trade can be
made deliberately rather than discovered.


> **Revised 2026-09-21.** Earlier revisions of this card reported **0.7364** for Darwin-397B-ZTC
> and **0.7282** for ZTC-Judge-27B. Those figures were produced by a run whose standardisation
> statistics were computed over all five domains, including the held-out one, which leaks a small
> amount of the evaluation domain into every figure. Re-run with the statistics fitted inside the
> training domains only, the figures are **0.7272** and **0.7289**. Cite the current values.

| Patronus Lynx 8B | 0.5179 |
| *The answering model's own stated confidence* | *0.5000* |

**Per domain, and against the surface baseline in the same domain:**

| Domain | Baseline | **ZTC-Judge-27B** | Items | Wrong |
|---|---|---|---|---|
| Professional exams (law · math · biology) | 0.7138 | **0.8462** | 400 | 80 |
| Scientific reasoning | 0.7272 | **0.7410** | 198 | 20 |
| Biology & medicine | 0.5908 | **0.7154** | 917 | 169 |
| Disaster & safety procedures | 0.5949 | **0.6961** | 225 | 109 |
| General multi-step reasoning | 0.5420 | **0.6172** | 278 | 130 |
| **Size-weighted mean** | **0.6223** | **0.7289** | 2,018 | 508 |

**Above the surface baseline in all five domains.** That baseline — a model reading nothing but
answer length, digit count and formatting — is the bar that matters; a verifier below it is
detecting shape, not correctness.

**Protocol.** Every figure comes from a domain the probe never saw. Hyper-parameters are selected
inside the training domains only. Scores are computed per domain and then size-weighted; pooling
all items into one AUC inflates the result, because score scales differ between domains.

**Self-readout.** Given only the question, this model answers on its own and the same forward pass
tells whether it was right: **0.7322** (3 domains, 1,595 items). Tools that see only text from
outside a model cannot do this at all.

### B. Repository-internal measurement — 400 items, random folds

| | |
|---|---|
| Judge | Qwen3.5-27B, revision `fc05daec18b0` (weights unmodified) |
| Items | 400 professional-exam items, out-of-fold |
| Result | **0.7999** |
| Permutation null | z = 25.68 |

This number is higher than the leaderboard figure because the protocol is easier: one domain,
random folds instead of held-out domains. Quote **0.7289** when comparing against other systems.

### API — drop-in for an existing JEV integration

The endpoint takes the same request shape and returns the same response shape, so switching an
existing integration is a URL change.

```bash
POST /v1/evaluate
Authorization: Bearer <token>

{"model": "vidraft/ztc",
 "state": {"question": "...", "answer": "..."},
 "questions": {"correct": {"type": "boolean",
                           "instructions": "Is the ANSWER factually correct?"}}}
```

```json
{"model": "vidraft/ztc-judge-27b",
 "answers": {"correct": {
    "probability": 0.1043,
    "verdict": "review",
    "score": -0.72,
    "position": 0.268,

    "band": "low",
    "action": "hold_or_escalate",
    "measured": {
      "band_accuracy": 0.485,
      "base_accuracy": 0.748,
      "if_lowest_20pct_dropped": 0.814,
      "escalate_gain_at_20pct_budget": 0.0134,
      "do_not": "resample_same_model",
      "why_not": "measured: fixes 6.7% of wrong answers, breaks 13.1% of right ones"}}},
 "usage": {"generated_tokens": 0}}
```

`type` accepts `boolean` and `noul`. Existing clients read `answers.<key>.probability` and ignore
the rest; the additional fields are there for clients that want to act on the score rather than
merely record it. **0.19 s per call, zero generated tokens.**

### What `probability` means

The raw score is unbounded. The shipped calibration maps it to P(answer is correct), fitted
**leave-one-domain-out** — the mapping never sees the domain it is applied to.

| | Expected calibration error |
|---|---|
| **ZTC-Judge-27B (after calibration)** | **0.0245** |
| JEV, as shipped | 0.0381 |
| JEV, after the same calibration | 0.0261 |
| Laya-Multilingual, as shipped | 0.4985 |
| Laya-Typed-Decisions, as shipped | 0.2641 |

Measured on the same 2,018 items. **ZTC and JEV are effectively tied on calibration; the difference
of 0.0016 is not meaningful.** Figures published elsewhere for these systems were measured on other
test sets and do not reproduce here.

🔴 Calibration is uneven across domains: 0.0225 on biology & medicine, but 0.2941 on scientific
reasoning and 0.2381 on general reasoning. Treat the probability as reliable in the first case and
as a ranking signal only in the other two.

## What to do when the score is low

The score alone is not actionable, so the response carries the measured consequence of each choice.

| Band | Share | **Actual accuracy of answers in this band** | Recommended |
|---|---|---|---|
| `low` | 20% | **48.5%** | hold, or escalate to a stronger model |
| `mid` | 40% | 73.2% | escalate if budget allows |
| `high` | 40% | **89.6%** | accept |

**Three things that work, measured on 2,018 items:**

| Action | Effect |
|---|---|
| **Drop the lowest-scoring 20%** | accuracy of what remains: **74.8% → 81.4%** |
| **Escalate the lowest 20% to a stronger model** | **+1.34 pp** end-to-end |
| **Send the lowest 20% to human review** | catches **47.2%** of all errors — 2.4× random |

### Generate several candidates and let the verifier pick

When the score is low, the most effective next step is not to escalate — it is to **produce more
candidate answers and select between them**. Measured on 346 questions with five candidates each
(1,730 candidates), all scored through this endpoint:

| Policy | Accuracy | Output tokens | vs. one attempt |
|---|---|---|---|
| One attempt | 49.13% | 72 | — |
| Majority vote over 3 | 46.82% | 212 | **−2.31 pp** |
| Majority vote over 5 | 46.53% | 353 | **−2.60 pp** |
| **Pick best of 5 with this model** | **51.45%** | 353 | **+2.31 pp** |
| Pick best of 5 with JEV | 53.47% | 353 | +4.34 pp |
| Pick one of 5 at random | 46.82% | 353 | −2.31 pp |
| *Oracle — any correct candidate counts* | *63.87%* | *353* | *+14.74 pp* |

**The same five candidates swing by 5 points depending on how one is chosen.** Majority voting is
worse than not resampling at all: when a model prefers a wrong answer, more samples make that wrong
consensus more certain. A verifier that ranks the candidates is what turns extra samples into
accuracy.

**Spend the budget only where it is needed.** Generating extra candidates only for low-scoring
first attempts captures most of the gain at a fraction of the cost:

| Triggered on | Accuracy | Output tokens | vs. one attempt |
|---|---|---|---|
| 10% of items | 49.71% | 80 | +0.58 pp |
| **30% of items** | **50.87%** | **132** | **+1.73 pp** |
| 100% of items | 51.45% | 353 | +2.31 pp |

**At a 30% trigger rate you get three quarters of the benefit for 1.8× the tokens**, where always
generating costs 4.9× for 1.3× the benefit.

*Scope: one generator (GPT-4o-mini), one item set, five candidates. The oracle row shows the
headroom that remains — a correct candidate is present far more often than any policy recovers it.*

**One thing that does not work:**

🔴 **Do not take a majority vote over resamples.** Measured: five resamples with majority voting score **46.53%** where a single attempt scores **49.13%**. More candidates make a wrong consensus more certain unless something picks between them — see the table above.



*Escalation pays for itself through precision, not recall. Re-answering repairs about 38% of wrong
answers and damages about 30% of right ones, so a gate is only worth its budget if it mostly calls
answers that are actually wrong.*

---

## Two probes ship with this model

| File | Size | Produces |
|---|---|---|
| `ztc_probe.npz` | 59 KB | linear readout — one dot product |
| `ztc_curve_probe.npz` | 5.3 MB | **the leaderboard figure, 0.7289** — 256 anchors, RBF kernel |
| `ztc_curve_probe_v2.npz` | 21 MB | **0.7752 on answers from an unseen model** — 1,024 anchors, same call shape |
| `usage_curve.py` | — | runnable example for both |

Both read the same input: the final-layer hidden state at the last position, from a single forward
pass with zero generated tokens. The curved probe is the one to use when the number matters; the
linear probe is there for deployments where 59 KB and a single dot product are worth the difference.

### Review budget

If you can only re-check part of your traffic, sort by score ascending and review the lowest.

| Reviewed | Errors caught | vs. random |
|---|---|---|
| 5% | 14.0% | 2.8× |
| 10% | 28.0% | 2.8× |
| 15% | 38.4% | 2.6× |
| 20% | 47.2% | 2.4× |
| 30% | 62.0% | 2.1× |
| 40% | 72.6% | 1.8× |

---

## Intended use

- **Triage of LLM output at scale** — route low-scoring answers to a human, a larger model, or a
  retrieval step, and let the rest through.
- **Regression monitoring** — track the score distribution of a deployed system over time.
- **Cheap pre-filter in front of an expensive verifier** — costs one forward pass, no decoding.

## Out of scope

- **Not a grounding checker.** It does not take a source document and decide whether the answer
  follows from it. Tools built for that task target a different axis.
- **Not a safety, toxicity or policy classifier.**
- **Not a calibrated probability.** Use it to rank and threshold, not as a likelihood.
- **Not a substitute for human review in high-stakes decisions.**

## Limitations

- **Domain coverage.** Scores are only meaningful for the five domains above. When `domain_auc`
  returns `None`, the domain was never measured and the score carries no published guarantee.
- **Weakest domain is 0.6172 (general multi-step reasoning).** General multi-step reasoning is the hardest case and is listed first
  in this section on purpose.
- **Sensitive to which model wrote the answer.** The probe is fitted on answers from four models.
  Adding 1,772 answers from a single additional model shifted the mixture and **lowered** the
  size-weighted score from 0.7278 to 0.7177 — professional exams rose to 0.8575 while every other
  domain fell. Treat "works on any model's output" as a design goal, not a measured guarantee: if
  your generator differs sharply from the training mixture, measure before relying on the number.
- **Revision lock.** The probe is fitted to one specific revision of the base model. Running it on a
  different Qwen3.5-27B revision produces **no error and silently wrong scores**. This repository
  ships the matching weights so that failure mode cannot occur.
- **It reports the verifier's judgement**, which is not the same thing as the answering model's own
  confidence — that quantity measures 0.500 here.
- **Long answers** are read within the model's context window; answers exceeding it are truncated
  before scoring.

## Lineage

| | |
|---|---|
| Base model | Qwen/Qwen3.5-27B, revision `fc05daec18b0`, Apache-2.0 |
| Modification to base weights | none — the probe is a separate 59 KB file |
| Added by FINAL-Bench | probe, inference code, evaluation protocol and tables |

## What this repository contains

| | |
|---|---|
| Included | Base weights · tokenizer · probe · inference code · evaluation configuration |
| Not included | Training corpus · hidden-state matrices · fitting pipeline |

The probe is the product; the material it was fitted from is not distributed.

## Independent comparison

This model is listed on the **Typed Decision Leaderboard**, which measures answer verifiers from
several vendors on one identical item set with identical labels, and publishes scores, labels and
grading code:

<https://huggingface.co/spaces/mayafree/typed-decision-leaderboard>

## License

The base model, Qwen3.5-27B, is Apache-2.0 and redistributable. The probe, the inference code and
the evaluation tables are assets of FINAL-Bench / VIDRAFT.

## Citation

```bibtex
@misc{ztc_judge_27b_2026,
  title  = {ZTC-Judge-27B: zero-token answer verification from a single forward pass},
  author = {FINAL-Bench},
  year   = {2026},
  url    = {https://huggingface.co/FINAL-Bench/ZTC-Judge-27B}
}
```
