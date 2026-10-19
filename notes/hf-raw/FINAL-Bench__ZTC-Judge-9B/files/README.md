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

# ZTC-Judge-9B

**Answer verification from a single forward pass, with zero generated tokens — at 9B.**

ZTC-Judge-9B takes a question and an answer written by *any* model and scores whether that
answer can be trusted. It is the third rung of a four-point size ladder measured under one
identical protocol, and it is published so the shape of that ladder can be checked rather than
asserted.

> **ZTC** — Zero-Token Confidence · **Judge** — it evaluates *someone else's* answer, not its own

---

## Read this before deploying

This model is **not** the strongest member of the family, and the card says so with numbers.

| Model | Leaderboard AUC |
|---|---|
| Darwin-397B-ZTC | 0.7272 |
| ZTC-Judge-27B | 0.7289 |
| **ZTC-Judge-9B** | **0.6506** |
| **ZTC-Judge-4B** | **0.6360** |
| *Answer length and formatting only* | *0.6223* |

## A second probe ships with this model, for answers from models we have not seen

The benchmark above is measured on a set where **79% of the answers come from a single model
family**. A probe fitted on it can end up specialised to that family rather than to correctness. We
checked with a model used in neither the fitting nor the scoring — Claude Haiku 4.5 — answering the
same 2,018 questions. Those 1,939 answers were graded with the same code and the same domain weights.

| On answers from an unseen model | AUC |
|---|---|
| **`ztc_curve_probe_v2.npz`** | **0.6931** |
| `ztc_curve_probe.npz` (behind the table above) | 0.6708 |
| *Answer length and formatting only* | *0.4334* |

`ztc_curve_probe_v2.npz` is fitted on answers from several models. It clears three gates: it improves
on the shipped probe (**+0.0223**), it clears the surface baseline, and against 8 label-permutation
runs it stands at **z = 10.48**. The file is 16.8 MB, 1,024 anchors, and takes the same call shape as the
original — the two are interchangeable in code.

**Which file to use.** `ztc_curve_probe.npz` reproduces the benchmark figure above and is the
reference for it. `ztc_curve_probe_v2.npz` is the one to use on answers written by a model outside
that benchmark, which is the usual case in deployment.

🔴 **The same trade appears at every size.** Specialisation helps on the distribution it was fitted to
and costs elsewhere. Both files are published so the choice can be made deliberately.

🔑 **Across this ladder, 9B and 4B land in the same place under v2 — 0.6931 and 0.6932.** Doubling the
parameter count buys nothing on this axis. Prefer the 4B unless something else requires the larger model.


> **Revised 2026-09-21.** Earlier revisions of this card reported **0.7364** for Darwin-397B-ZTC
> and **0.7282** for ZTC-Judge-27B. Those figures were produced by a run whose standardisation
> statistics were computed over all five domains, including the held-out one, which leaks a small
> amount of the evaluation domain into every figure. Re-run with the statistics fitted inside the
> training domains only, the figures are **0.7272** and **0.7289**. Cite the current values.


**The ladder does not decline smoothly — it steps.** Between 9B and 27B the score moves 0.078,
while between 4B and 9B it moves 0.015. Whatever carries verification quality is largely absent
below 27B on this axis.

**Where this model is worth deploying is one specific place**, and it is a real one:

| Domain | Surface baseline | **9B** | Margin |
|---|---|---|---|
| **Professional exams (law · math · biology)** | 0.7138 | **0.7623** | ****+0.0485**** |
| Scientific reasoning | 0.7272 | 0.6764 | 🔴 -0.0508 |
| Biology & medicine | 0.5908 | 0.6310 | **+0.0402** |
| Disaster & safety procedures | 0.5949 | 0.5862 | 🔴 -0.0087 |
| General multi-step reasoning | 0.5420 | 0.5881 | **+0.0461** |
| **Size-weighted mean** | **0.6223** | **0.6506** | ****+0.0283**** |

🔴 **Do not use this model for disaster and safety content.** In that domain it does not clear the
surface baseline, which means it is reading answer shape rather than correctness there.

✅ **Professional-exam style content is where it earns its size.** It runs on a laptop, on CPU, and
inside networks that never reach the internet — places a hosted API cannot go.

## How it works

```
[question + answer]  →  one forward pass
                     →  final-layer hidden state at the last position (4096-d)
                     →  probe
                     →  score
```

**Generated tokens: 0.** No access to the answering model's weights or logits is required; the text
of the answer is the only input. Latency is one forward pass, and batching converts directly into
throughput.

## Usage

```python
import json
import numpy as np, torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel, AutoTokenizer

REPO = "FINAL-Bench/ZTC-Judge-9B"
cfg  = json.load(open(hf_hub_download(REPO, "ztc_config.json"), encoding="utf-8"))
path = snapshot_download(REPO)

tok   = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()

def hidden(question, answer):
    b = tok([cfg["template"] % (question, answer)], return_tensors="pt",
            truncation=True, max_length=cfg["max_length"])
    dev = next(model.parameters()).device
    with torch.no_grad():
        h = model(input_ids=b["input_ids"].to(dev),
                  attention_mask=b["attention_mask"].to(dev)).last_hidden_state
    return h[0, int(b["attention_mask"].sum()) - 1].float().numpy().astype(np.float64)

# linear probe — one dot product
p = np.load(hf_hub_download(REPO, "ztc_probe.npz"))
v = hidden("Which defensive chemical does an insect release?", "C. Allomone")
print(float(((v - p["mu"]) / p["sd"]) @ p["w"]))
```

The score is an unbounded real number; higher means more likely correct. It is a **ranking signal**,
not a calibrated probability — choose a threshold from your own review budget.

## Two probes ship with this model

| File | Produces |
|---|---|
| `ztc_probe.npz` | linear readout — one dot product |
| `ztc_curve_probe.npz` | **the reported figure 0.6506** — 256 anchors, RBF kernel |

Both read the same input. The curved probe is the one to use when the number matters.

## API — drop-in for an existing JEV integration

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
{"model": "vidraft/ztc-judge-9b",
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

## Protocol

| | |
|---|---|
| Items | 2,018 · 508 incorrect · 5 domains · answers written by 4 different models |
| Metric | AUC — how well wrong answers sort to the bottom. Threshold-free. 0.5 = coin flip |
| Selection | **Leave-one-domain-out.** Every figure comes from a domain the probe never saw; hyper-parameters are chosen inside the training domains only |
| Aggregation | Per domain, then size-weighted. Pooling all items into one AUC inflates the result |

The same protocol, item set and grading code are applied to every rung of the ladder and to the
other systems on the independent leaderboard:
<https://huggingface.co/spaces/mayafree/typed-decision-leaderboard>

## Out of scope

- **Not a grounding checker.** It does not take a source document and decide whether the answer
  follows from it.
- **Not a safety, toxicity or policy classifier.**
- **Not a calibrated probability.** Use it to rank and threshold.
- **Not a general-purpose verifier at this size.** See the domain table above.

## Limitations

- **Domain coverage.** Scores are meaningful only for the five domains listed. Outside them nothing
  has been measured and no guarantee is published.
- **Below the surface baseline on disaster and safety.** Stated in the table rather than omitted.
- **Sensitive to which model wrote the answer.** The probe is fitted on answers from four models.
  Adding 1,772 answers from a single additional model shifted the mixture and **lowered** the
  size-weighted score from 0.7278 to 0.7177 — professional exams rose to 0.8575 while every other
  domain fell. Treat "works on any model's output" as a design goal, not a measured guarantee: if
  your generator differs sharply from the training mixture, measure before relying on the number.
- **Revision lock.** The probe is fitted to one specific revision of the base model. Running it on a
  different revision produces **no error and silently wrong scores**; this repository ships the
  matching weights so that failure mode cannot occur.
- **It reports the verifier's judgement**, which is not the answering model's own confidence — that
  quantity measures 0.5000 on this set.

## Lineage

| | |
|---|---|
| Base model | `Qwen/Qwen3.5-9B`, Apache-2.0 |
| Modification to base weights | none — the probes are separate files |
| Added by FINAL-Bench | probes, inference code, evaluation protocol and tables |

## What this repository contains

| | |
|---|---|
| Included | Base weights · tokenizer · linear probe · curved probe · configuration |
| Not included | Training corpus · hidden-state matrices · fitting pipeline |

## The rest of the ladder

[ZTC-Judge-27B](https://huggingface.co/FINAL-Bench/ZTC-Judge-27B) ·
[Darwin-397B-ZTC](https://huggingface.co/FINAL-Bench/Darwin-397B-ZTC) ·
[ZTC-Judge-9B](https://huggingface.co/FINAL-Bench/ZTC-Judge-9B) ·
[ZTC-Judge-4B](https://huggingface.co/FINAL-Bench/ZTC-Judge-4B)

## License

The base model is Apache-2.0 and redistributable. The probes, the inference code and the evaluation
tables are assets of FINAL-Bench / VIDRAFT.
