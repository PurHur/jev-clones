---
license: apache-2.0
base_model: tasksource/ModernBERT-large-nli
pipeline_tag: zero-shot-classification
tags: [system-one, kenning, decision-model, calibrated]
---

# kenning-large-v0.6

Kenning is a **System One** decision model: given program state and typed questions (`noul` yes/no, `choice`, `score`) it returns typed answers with calibrated probabilities in one forward pass. It does not generate text. It was built with [SystemOne Builder](https://github.com/systemonedev/systemone-builder).

## Model

- Base model: `tasksource/ModernBERT-large-nli` (see its model card)
- Architecture: cross-encoder; every candidate answer is scored against the state, and each question's distribution is `softmax(scores / T)` with per-type temperatures fitted on held-out data
- Temperatures: `{"choice": 1.8221, "score": 1.7333, "noul": 1.8221}`
- Max tokens per (state, answer) pair: 2048
- Created: 2026-10-08T23:04:00Z

## Training data

| Source | Rows | Licence |
|---|---|---|
| `nyu-mll/multi_nli` | 8000 | OANC (permissive); fiction genre excluded |
| `clinc/clinc_oos` | 3000 | CC-BY-3.0 |
| `fancyzhx/amazon_polarity` | 2000 | Apache-2.0 |
| `google/civil_comments` | 2000 | CC0-1.0 |
| `nvidia/HelpSteer2` | 3000 | CC-BY-4.0 |
| `jackhhao/jailbreak-classification` | 1000 | Apache-2.0 |
| `deepset/prompt-injections` | 500 | Apache-2.0 |
| `allenai/ropes` | 3000 | CC-BY-4.0 |
| `structured expense (records): rule-generated, exact labels` | 800 | generated (kenning/structured.py) |
| `structured shipping (records): rule-generated, exact labels` | 800 | generated (kenning/structured.py) |
| `structured leave (records): rule-generated, exact labels` | 800 | generated (kenning/structured.py) |
| `structured inventory (records): rule-generated, exact labels` | 800 | generated (kenning/structured.py) |
| `structured incident (records): rule-generated, exact labels` | 800 | generated (kenning/structured.py) |
| `structured eligibility (records): rule-generated, exact labels` | 800 | generated (kenning/structured.py) |
| `structured loan (records): rule-generated, exact labels` | 800 | generated (kenning/structured.py) |
| `structured subscription (records): rule-generated, exact labels` | 800 | generated (kenning/structured.py) |
| `structured table (table): rule-generated, exact labels` | 800 | generated (kenning/structured.py) |
| `structured tool_call (agent): rule-generated, exact labels` | 800 | generated (kenning/structured.py) |
| `structured agent_trace (agent): rule-generated, exact labels` | 800 | generated (kenning/structured.py) |
| `structured access_log (logs): rule-generated, exact labels` | 800 | generated (kenning/structured.py) |
| `structured deploy_log (logs): rule-generated, exact labels` | 800 | generated (kenning/structured.py) |
| `synthetic: written by Qwen/Qwen2.5-7B-Instruct from labelled scenarios` | 1985 | generated (teacher: Qwen2.5-7B-Instruct, Apache-2.0) |
| `synthetic: 10 label-conditioned tasks written by Qwen/Qwen2.5-7B-Instruct` | 2793 | generated (teacher: Qwen2.5-7B-Instruct, Apache-2.0) |

## Held-out results (split of the training pool)

| | accuracy | Brier | ECE |
|---|---|---|---|
| zero-shot (before training) | 0.578 | 0.607 | 0.220 |
| trained | 0.905 | 0.141 | 0.040 |
| trained + calibrated | 0.905 | 0.133 | 0.006 |

These are in-distribution numbers. Benchmark on your own data (and recalibrate on a few hundred labelled examples from it) before letting the model act automatically.

## Benchmarks

Recorded with `systemone bench` on suites never used for training. *Automated* is the share of items decided without a person (p >= 0.9 or <= 0.1); a *threat auto-closed* is a positive item the model was sure was negative.

| Suite | Items | Accuracy | Brier | ECE | Automated | Threats auto-closed | Latency p50 |
|---|---|---|---|---|---|---|---|
| general | 1328 | 0.660 / 0.900 / 0.604 / 0.820 / 0.700 / 0.900 / 0.940 / 0.750 / 0.979 / 1.000 / 0.940 / 0.583 / 0.938 / 0.680 / 1.000 / 0.660 / 0.900 / 0.920 / 0.560 / 0.540 / 0.833 / 0.812 / 0.560 / 0.880 / 0.600 / 0.760 / 0.820 | 0.214 | 0.179 | - | - | 589 ms |

Where a suite asks several questions, accuracy lists each (out of domain: spam / emotion / news topic; layouts: one per layout). Comparisons with Cloudflare Clef and TypeSafe Jev on the same suites are in the [Kenning docs](https://github.com/systemonedev/systemone-builder/blob/main/docs/kenning.md).

## Use

```bash
pip install "systemone-client[local]"
```

```python
from systemone import Kenning, Noul, Choice

model = Kenning.from_pretrained("systemonedev/kenning-large-v0.6")  # in-process, runs on CPU or GPU
answer = model.system_one(state={"ticket": "I was charged twice."},
                          questions={"billing": Noul("Is this about billing?")})
print(answer.nouls["billing"].noul)
```

Or serve it with SystemOne Builder's `kenning` service and call `POST /v1/systemone` with `systemone.Client`.

## Limitations

- Calibration was fitted on the training distribution; probabilities on other data are scores until recalibrated.
- Arithmetic, dates and long or contradictory states degrade accuracy.
- Determinism: identical requests give identical answers on the same hardware and software; results can differ across GPUs or library versions.

## Licence

The weights are released under the **Apache License 2.0** (LICENSE). Upstream licences of the base model and of every training source are listed in NOTICE.md; some are share-alike (CC-BY-SA-3.0), so keep NOTICE.md with the weights.

Kenning implements a wire format compatible with TypeSafe AI's System One API. It is not affiliated with or endorsed by TypeSafe AI, and was not trained on TypeSafe outputs.
