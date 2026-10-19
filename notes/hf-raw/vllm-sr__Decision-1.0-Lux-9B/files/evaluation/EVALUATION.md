# Evaluation

The comparison covers **3,766 scored decisions across 54 tasks** and all 15 displayed models. All models use the same requested examples, candidate order and scoring rules. Each model keeps its native admission, truncation and probability semantics.

| Model | Size | Decisions | Composition | Reading | Inference | Transfer | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lux-9B | 9B | **84.38** | **52.75** | 90.16 | **91.46** | 77.72 | **77.40** |
| Nox-4B | 4B | **83.00** | **51.79** | 79.06 | **86.25** | 69.60 | **73.09** |
| Kev-9B | 9B | 76.75 | 45.75 | 86.72 | 83.54 | 79.25 | 71.89 |
| Kev-4B | 4B | 71.90 | 48.54 | 81.88 | 84.58 | 76.10 | 70.09 |
| Qwen3.5-9B | 9B | 73.91 | 44.62 | 89.84 | 79.58 | 73.23 | 69.73 |
| Decider | 2B | 64.01 | 46.58 | 92.03 | 84.38 | 69.31 | 67.71 |
| Qwen3.5-4B | 4B | 69.89 | 43.33 | 87.97 | 79.79 | 68.83 | 67.29 |
| Sol-2B | 2B | 73.75 | 46.08 | 76.56 | 84.17 | 57.07 | 66.32 |
| Eos-0.8B | 0.8B | 65.94 | 46.04 | 70.31 | 81.67 | 52.01 | 61.89 |
| Kev-0.8B | 0.8B | 60.14 | 42.29 | 67.81 | 68.75 | 61.19 | 58.28 |
| Qwen3.5-2B | 2B | 57.12 | 39.00 | 73.75 | 72.29 | 56.31 | 57.24 |
| Kai-0.6B | 0.6B | 57.96 | 40.83 | 54.69 | 69.79 | 48.37 | 53.52 |
| Laya · English | 0.421B | 56.54 | 35.33 | 51.41 | 63.75 | 53.06 | 51.03 |
| Laya · Multilingual | 0.322B | 47.25 | 38.92 | 50.78 | 57.29 | 47.13 | 47.19 |
| Jev | — | 79.10 | 66.38 | 94.53 | 89.79 | 87.19 | 81.05 |

Accuracy (%). Open model rows are sorted by overall score; Jev is the frontier reference at the end. Bold marks Decision-family cells strictly above every external open or untuned reference in that metric; Jev and the other Decision models are excluded from that threshold.

## Scope and weighting

The overall score weights **Decisions 30%, Composition 25%, Reading 15%, Inference 15%, Transfer 15%**. The first four panels contain 880, 880, 480 and 480 questions and retain their original family/source weights. Transfer is micro-accuracy over 1,046 clean knowable questions from the frozen upstream transfer test; 110 missing-evidence questions and 108 variants remain separate diagnostics. No latency, calibration error or consistency score is averaged into accuracy.

Reported benchmarks cover English and Chinese; these are evaluation languages, not a restriction on accepted input languages.

These are **outcome-informed product-priority weights, chosen after observing benchmark results**. The data are observed regression tests, not a fresh blind test. Reweighting is not a training improvement. [Weight sensitivity](SENSITIVITY.md) retains the prior weighting and original four-panel comparison for the same model weights. Training, checkpoint selection and calibration do not use these test labels.

## Full results

[All 54 task rows](TASKS.md) preserve every original decision, composition, reading and inference task plus all 27 transfer tasks. [Diagnostics](DIAGNOSTICS.md) separately report probability quality, option-order sensitivity, missing evidence, native coverage and uncertainty. [Exact statistics](benchmark.json) include counts and confidence intervals.

## Model and API scope

Decision models return typed Choice, Noul and Score answers in the SystemOne format. Encoder and decoder references are evaluated through their published native interfaces. Untuned Qwen models use the frozen letter-logit readout, with no generated-text parsing. Kev uses the matched BF16 backbone with FP32 head and shipped temperature; date-fact injection is disabled. This differs from the authors’ default FP32 reproduction. Laya English and multilingual are measured separately. Kai-0.6B and Eos-0.8B use their independently verified published native interfaces on the same 54 tasks and requested denominators. Jev is a recorded hosted-service snapshot.

The tests assess decisions from the supplied state and fixed choices. They do not establish live fact retrieval, universally superior reasoning or cross-hardware speed rankings. [Immutable identities and evaluation provenance](evaluation-provenance.json).
