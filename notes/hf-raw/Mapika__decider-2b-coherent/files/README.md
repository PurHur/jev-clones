---
license: apache-2.0
base_model: Mapika/decider-2b
library_name: pytorch
tags:
  - decider
  - bookiebench
  - calibration
  - coherence
  - multiple-choice
  - probabilistic-reasoning
---

# decider-2b-coherent (v1.1)

**decider-2b-coherent** is [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b) with two small add-ons:

1. **A learned coupling.** When you ask several questions about one context, you get a single joint distribution over all the answers. Every marginal, conjunction, negation and conditional comes from that one joint.
2. **A sims-calibrated marginal adapter.** It runs only when a domain gate decides the input looks like a BookieBench-style probabilistic-reasoning state.

The joint is I-projected onto the served marginals using factored iterative proportional fitting (IPF). So:

- **The answers are coherent by construction.** No Dutch book can be made against them. BookieBench measures dutch = 0 on every group.
- **On real (non-sims) inputs the marginals are stock decider-2b's.** The gate sends almost every real input down decider's own path at its served temperature T = 1.145. Regression-set accuracy and NLL match decider-2b to within 0.0002.

This repository holds only the add-on weights: 9.5M parameters, 38 MB. The frozen base is loaded from `Mapika/decider-2b` at a pinned revision (`config.json` → `base_revision` = `533964dae8be954c5b5e19fa4948e48408094c1e`). The base weights are never modified.

## How it works

```
frozen decider-2b forward (decider's own state_first prompt, token-identical)
  -> letter logits l_i for each question i, hidden states h
  -> domain gate  p_sims = sigmoid(w · standardize([mean slot hidden, mean context hidden]) + b)
       p_sims <  0.5  (real): q_i = softmax(l_i / 1.145)                              = stock decider-2b
       p_sims >= 0.5  (sims): q_i = softmax(a·l_i + <U h_slot_i, V h_option_ij>/√d_a)  (the adapter)
  -> coupling: a mixture of M = 8 products, with component logits log q_i + learned offsets
  -> factored IPF onto q_1..q_n (float64, tolerance 1e-12): the joint's marginals are exactly q_i
```

- A single-question input skips the coupling, so its joint is `q` itself.
- A mixture of products stays a mixture of products under IPF scaling. Each IPF step costs O(M·K_i), and the full joint table is never built, whatever the number of questions.

## Usage

The code lives in this repo (`decider_coherent/`). It depends only on `torch`, `transformers` (tested with 5.17, torch 2.14) and `huggingface_hub`.

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("Mapika/decider-2b-coherent")
sys.path.insert(0, path)
from decider_coherent import load

m = load()   # downloads Mapika/decider-2b at the pinned revision + this repo's model.pt; uses CUDA if available

ctx = "Customer: I was charged twice for my March invoice and need this fixed today, it's blocking payroll."
p = m.predict(ctx, [
    ("What is the ticket about?", ["billing", "technical issue", "account access", "other"]),
    ("What is the priority?",     ["low", "medium", "high"]),
])
p.marginals          # [[P(billing), ...], [P(low), P(medium), P(high)]]  (options in the order given)
p.joint              # numpy array [4, 3]: P(topic, priority)
p.p_sims             # domain-gate score; < 0.5 here, so the marginals are stock decider-2b's
p.answer({"kind": "cond", "event": {"q1": [0]}, "given": {"q2": [2]}})       # P(billing | high priority)
p.answer({"kind": "noul", "event": {"q1": [0], "q2": [2]}, "neg": True})     # P(not (billing and high))
```

Questions are named `q1..qn` for `answer()`. Query kinds:

- `marginal`: `{"kind": "marginal", "var": "q1"}`
- `noul` (conjunction, or its negation with `"neg": true`): `{"kind": "noul", "event": {name: [option indices]}}`
- `cond` (conditional): `{"kind": "cond", "event": {...}, "given": {...}}`

These are the BookieBench query semantics. For a BookieBench-format instance (`prelude`, `variables`, `queries`, `steps`):

```python
last = len(inst["steps"]) - 1
step_of = lambda q: last if q.get("step") is None else (last + q["step"] if q["step"] < 0 else q["step"])
pred = m.predict_instance(inst)                        # evidence up to the last step
answers = {q["id"]: pred.answer(q) for q in inst["queries"] if step_of(q) == last}
# earlier-step queries: m.predict_instance(inst, step=k)
```

`m.predict_batch([(context, questions), ...], bs=32)` batches inputs. For full control, `m.build(...)` and `m(items)` return `MixtureJoint` objects.

### Joint vs marginal-only use

- `predict(..., joint=True)` (the default) runs the coupling and IPF. Every cross-question answer is then available and coherent.
- `predict(..., joint=False)` skips the coupling and returns the product of the marginals. **The marginals are identical either way**, because IPF reproduces them exactly. So if you only need per-question probabilities, use `joint=False`: it costs the same as stock decider-2b plus the tiny gate (and the adapter on sims inputs).
- Don't combine answers from `joint=False` across questions. The product joint treats the questions as independent.

### Temperature and domain override

- `temperature=` sets the real-path temperature. The default is 1.145, decider-2b's served value.
- `domain="real"` or `domain="sims"` overrides the gate.
- BookieBench's leaderboard protocol fits one post-hoc factor on train calibration data. For this model it is t = 1.033 (`config.json` → `leaderboard_temperature_factor`), applied to the joint as p ∝ joint^(1/t). The model is served untempered; the factor changes the scores only marginally (see below).

## Domain gate

The gate is a logistic probe on frozen decider-2b features (mean hidden state over the answer slots, mean hidden state over the context tokens). It was trained on 11,600 sims train states and 12,000 rows from the decider mixture train half, with threshold 0.5. The adapter runs only when p_sims ≥ 0.5; the coupling and IPF run everywhere.

| Data | Share gated as "sims" |
|---|---|
| Held-back train rows (4,720) | AUC 1.000, 0 errors |
| decider regression set (120,424 real rows, eval) | 0.05% (63 rows) |
| Held-out real calibration slice (20,000 mixture train rows) | 0.0% |
| BookieBench release sims groups (test, test_prior, heldout, val, dev; eval) | 100% |
| BookieBench realcoh subset (7,500 real-data records; eval) | 1.65%: math_problems 34%, mmlu_pro 7%, code_defects 1%, all other sources 0% |

The eval-set rates are reported only. Nothing was selected on them.

## Training

- **Base:** frozen, never updated. Only the coupling, the adapter and the gate were trained.
- **Sims data:** BookieBench train splits only: `data/release/train` (22 families, including the randbn/randhmm procedural priors) plus the v1 `train_nuisance` set, with a random evidence step per sample. Excluded:
  - every id in the leaderboard calibration sample
  - a 50-per-family dev slice used for checkpoint selection
  - all 9 HOLDOUT_TRAIN families (epidemic, forensic, raters, recapture, montyhall, search, queue, prog_domain, tab_stream), val (genetics, tracking) and every eval file
- **Real data:** the train half of the decider supervised mixture (multi-question rows, gold-tuple joint NLL, used for the coupling). The eval half is the regression set and was never used. A 20,000-row held-out slice was excluded.
- **Loss:** on sims rows, KL(exact posterior joint ‖ IPF(coupling, adapted marginals)). On real multi-question rows, −log joint(gold tuple) through decider's own marginals.
- **Schedule:** coupling initialised from the v1.0 hybrid. 3,000 steps, batch 32 (25% real rows), AdamW with lr 5e-4 and cosine decay. One GPU, about 55 minutes.
- **Selection:** the checkpoint with the best joint KL on the train dev slice (step 3000).
- **Seeds:** one.

## Evaluation

### BookieBench v1.1.0 leaderboard

Shared leaderboard subset, scored by the leaderboard's own code. `-Tfit` rows use the leaderboard's single fitted factor t (1.033 here, 3.372 for v1.0); raw rows are untempered. All groups have answered_frac 1.0 and dutch = 0 (≤ 1e-16, dutch@0.01 = 0).

skill = 1 − KL / KL(uniform) against the exact posterior.

| group | model | skill | skill_prior | kl_marg | sens |
|---|---|---|---|---|---|
| new_mechanics (headline, no train data) | **decider-2b-coherent v1.1 (-Tfit)** | **0.239** | 0.239 | 0.235 | −0.002 |
| | decider-2b-coherent v1.1 (raw) | 0.236 | 0.236 | 0.236 | −0.007 |
| | v1.0 hybrid (-Tfit) | 0.183 | 0.183 | 0.257 | 0.027 |
| | decider-2b (-Tfit) | −0.107 | | | |
| surface_transfer | **v1.1 (-Tfit)** | **0.264** | 0.257 | 0.126 | −0.090 |
| | v1.1 (raw) | 0.260 | 0.254 | 0.126 | −0.100 |
| | v1.0 hybrid (-Tfit) | 0.222 | 0.212 | 0.140 | −0.047 |
| in_family | **v1.1 (-Tfit)** | **0.493** | 0.449 | 0.128 | 0.128 |
| | v1.1 (raw) | 0.494 | 0.451 | 0.127 | 0.124 |
| | v1.0 hybrid (-Tfit) | 0.217 | 0.098 | 0.210 | 0.058 |

For comparison, stock decider-2b has dutch 0.85 on new_mechanics, because its answers to different queries are produced separately and can contradict each other.

**realcoh** (real-data coherence set, all 25 sources, 1,500 states):

| model | acc | ece | logscore | dutch |
|---|---|---|---|---|
| **v1.1 (-Tfit)** | 0.671 | **0.081** | **−0.833** | 0 |
| v1.1 (raw) | 0.671 | 0.087 | −0.839 | 0 |
| v1.0 hybrid (-Tfit) | 0.672 | 0.201 | −0.981 | 0 |

### Regression gate against decider-2b

The decider regression set (the eval half of the decider mixture), with the temperature fitted on in-task tasks as in decider's own protocol:

| | acc | NLL | ECE |
|---|---|---|---|
| in-task: decider-2b-coherent | 0.8017 | 0.4809 | 0.0384 |
| in-task: decider-2b | 0.8017 | 0.4808 | 0.0384 |
| held-out tasks: decider-2b-coherent | 0.7518 | 0.6272 | 0.0844 |
| held-out tasks: decider-2b | 0.7518 | 0.6270 | 0.0844 |

The fitted temperature is identical (1.1545). The residual difference comes from the 63 gated rows.

## Caveats

- **Sensitivity drops on transfer groups.** The adapter mainly fixes absolute calibration, not how the model responds to evidence. sens goes from 0.027 to −0.002 on new_mechanics and from −0.047 to −0.090 on surface_transfer, meaning the answers move less (or in the wrong direction) when evidence changes.
- **Some families get worse** than with the v1.0 hybrid (skill, -Tfit):

  | family | v1.0 hybrid | v1.1 |
  |---|---|---|
  | epidemic (new_mechanics) | −0.031 | −0.187 |
  | heldout/spam | 0.138 | 0.032 |
  | heldout/hiring | 0.330 | 0.250 |

  The group means rise because gains elsewhere are larger (e.g. poker, gauge, matching, factory).
- **One seed.** Seed-to-seed variation has not been measured.
- **Gate misfires on some real data.** realcoh `math_problems` is gated as sims 34% of the time (mmlu_pro 7%), so those inputs get the adapter instead of stock decider marginals. Real inputs that look like probability puzzles can be routed the same way. Pass `domain="real"` to force decider's path.
- **In-family gains are partly memorised surface statistics.** The in_family jump reflects training on those families' train data.
- **Option order is inherited.** The model reads decider's prompt, so it keeps decider's option-order sensitivity.
- **Context truncation is inherited.** Contexts longer than 1,536 tokens are cut from the end, as in decider-2b.

## License and attribution

Apache-2.0 (see `LICENSE`). The base model [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b) is also Apache-2.0, and the prompt helpers in `decider_coherent/modeling.py` are vendored from [github.com/Mapika/decider](https://github.com/Mapika/decider) (Apache-2.0).

- Author: Mark Marosi
- Benchmark: BookieBench, [github.com/Mapika/bookiebench](https://github.com/Mapika/bookiebench)
- Base model and code: [Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b), [github.com/Mapika/decider](https://github.com/Mapika/decider)
