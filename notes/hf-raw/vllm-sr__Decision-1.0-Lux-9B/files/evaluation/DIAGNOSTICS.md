# Diagnostics

These axes remain separate from headline accuracy. Probability metrics use the shipped distribution; no temperature is fitted on this benchmark. An unavailable full-population metric is shown as —, without replacing it with a successful-only average.

## Probability quality on transfer

| Model | Valid / requested | Brier ↓ | NLL ↓ | ECE % ↓ | Coverage at ≤5% error % ↑ | AURC ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Lux-9B | 1046/1046 | 0.2911 | 0.5870 | 4.79 | 63.38 | 0.0579 |
| Nox-4B | 1046/1046 | 0.4169 | 0.9079 | 11.80 | 36.42 | 0.1109 |
| Kev-9B | 1046/1046 | 0.2948 | 0.6028 | 4.45 | 58.03 | 0.0586 |
| Kev-4B | 1046/1046 | 0.3145 | 0.6434 | 2.61 | 56.79 | 0.0659 |
| Qwen3.5-9B | 1046/1046 | 0.3673 | 0.7597 | 9.75 | 47.42 | 0.0872 |
| Decider | 1046/1046 | 0.4122 | 0.7969 | 8.09 | 35.18 | 0.1172 |
| Qwen3.5-4B | 1046/1046 | 0.4199 | 0.8186 | 8.98 | 34.70 | 0.1233 |
| Sol-2B | 1046/1046 | 0.5454 | 1.1094 | 11.25 | 16.16 | 0.2073 |
| Eos-0.8B | 1046/1046 | 0.5847 | 1.1406 | 13.96 | 10.23 | 0.2611 |
| Kev-0.8B | 1046/1046 | 0.4839 | 0.9489 | 2.58 | 21.61 | 0.1782 |
| Qwen3.5-2B | 1046/1046 | 0.5708 | 1.2327 | 12.30 | 11.76 | 0.2699 |
| Kai-0.6B | 1046/1046 | 0.6392 | 1.2391 | 14.36 | 4.02 | 0.3345 |
| Laya · English | 1046/1046 | 0.5832 | 1.2275 | 12.90 | 0.48 | 0.2780 |
| Laya · Multilingual | 1046/1046 | 0.6873 | 1.4653 | 21.36 | 0.00 | 0.3460 |
| Jev | 1046/1046 | 0.1912 | 0.5743 | 3.35 | 76.96 | 0.0346 |

Coverage at an error threshold keeps whole confidence-tie groups together. These are empirical observed-sample results, not a deployment error guarantee.

## Option-order robustness

| Model | Valid / requested pairs | Both correct % ↑ | Semantic flip % ↓ | Mean half-L1 ↓ |
|---|---:|---:|---:|---:|
| Lux-9B | 36/36 | 77.78 | 8.33 | 0.0823 |
| Nox-4B | 36/36 | 63.89 | 16.67 | 0.0795 |
| Kev-9B | 36/36 | 80.56 | 2.78 | 0.0615 |
| Kev-4B | 36/36 | 77.78 | 5.56 | 0.0667 |
| Qwen3.5-9B | 36/36 | 75.00 | 11.11 | 0.1157 |
| Decider | 36/36 | 83.33 | 11.11 | 0.0701 |
| Qwen3.5-4B | 36/36 | 77.78 | 13.89 | 0.1390 |
| Sol-2B | 36/36 | 41.67 | 38.89 | 0.0694 |
| Eos-0.8B | 36/36 | 55.56 | 11.11 | 0.1505 |
| Kev-0.8B | 36/36 | 55.56 | 16.67 | 0.0808 |
| Qwen3.5-2B | 36/36 | 55.56 | 30.56 | 0.2163 |
| Kai-0.6B | 36/36 | 44.44 | 13.89 | 0.1147 |
| Laya · English | 36/36 | 44.44 | 19.44 | 0.0928 |
| Laya · Multilingual | 36/36 | 50.00 | 22.22 | 0.1491 |
| Jev | 36/36 | 86.11 | 0.00 | 0.0208 |

The 36 paired permutations test the same semantics under changed option order. Stable answers can still be wrong; both-correct rate therefore accompanies flip rate. Refused pairs remain in the both-correct denominator.

## Missing evidence

| Model | Valid / requested | Intact/control accuracy % ↑ | Mean max P % ↓ | P≥0.9 share % ↓ | Normalized entropy ↑ | Paired confidence drop pp ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Lux-9B | 110/110 | 86.36 | 63.60 | 20.91 | 0.7358 | 26.59 |
| Nox-4B | 110/110 | 72.73 | 78.65 | 27.27 | 0.5072 | 12.64 |
| Kev-9B | 110/110 | 91.82 | 39.61 | 0.00 | 0.9981 | 53.19 |
| Kev-4B | 110/110 | 91.82 | 41.47 | 0.00 | 0.9918 | 51.57 |
| Qwen3.5-9B | 110/110 | 80.91 | 67.62 | 7.27 | 0.7585 | 19.09 |
| Decider | 110/110 | 69.09 | 68.82 | 16.36 | 0.6772 | 13.59 |
| Qwen3.5-4B | 110/110 | 72.73 | 59.89 | 0.91 | 0.8120 | 22.35 |
| Sol-2B | 110/110 | 65.45 | 77.19 | 24.55 | 0.5424 | 5.68 |
| Eos-0.8B | 110/110 | 55.45 | 63.01 | 9.09 | 0.7772 | 1.93 |
| Kev-0.8B | 110/110 | 87.27 | 42.11 | 0.00 | 0.9796 | 40.34 |
| Qwen3.5-2B | 110/110 | 51.82 | 65.19 | 9.09 | 0.7370 | 4.68 |
| Kai-0.6B | 110/110 | 37.27 | 52.11 | 18.18 | 0.8263 | 0.62 |
| Laya · English | 110/110 | 49.09 | 67.96 | 0.00 | 0.7497 | -5.58 |
| Laya · Multilingual | 110/110 | 37.27 | 75.70 | 29.09 | 0.5307 | -0.49 |
| Jev | 110/110 | 93.64 | 62.04 | 16.36 | 0.7657 | 30.30 |

The 110 unknowable examples have no scored true class and are excluded from accuracy. Confidence is compared with matched evidence-bearing controls. These variants can combine evidence removal, candidate deletion and option permutation. Their confidence shifts are descriptive and do not isolate pure abstention or evidence sensitivity; these are not correctness scores.

## Native contract coverage

| Model | Original probability rows | Transfer probability rows | Transfer truncated questions |
|---|---:|---:|---:|
| Lux-9B | 2720/2720 | 1264/1264 | 0 |
| Nox-4B | 2720/2720 | 1264/1264 | 0 |
| Kev-9B | 2720/2720 | 1264/1264 | 0 |
| Kev-4B | 2720/2720 | 1264/1264 | 0 |
| Qwen3.5-9B | 2720/2720 | 1264/1264 | 0 |
| Decider | 2720/2720 | 1264/1264 | 0 |
| Qwen3.5-4B | 2720/2720 | 1264/1264 | 0 |
| Sol-2B | 2720/2720 | 1264/1264 | 0 |
| Eos-0.8B | 2720/2720 | 1264/1264 | 0 |
| Kev-0.8B | 2720/2720 | 1264/1264 | 0 |
| Qwen3.5-2B | 2720/2720 | 1264/1264 | 0 |
| Kai-0.6B | 2720/2720 | 1264/1264 | 0 |
| Laya · English | 2720/2720 | 1264/1264 | 34 |
| Laya · Multilingual | 2720/2720 | 1264/1264 | 14 |
| Jev | 2720/2720 | 1264/1264 | — / not observable |

Transfer coverage includes all 1,264 questions: clean, unknown-evidence and order variants. Accuracy uses 1,046 clean knowable questions. Native refusals count as incorrect when accuracy applies, and absent probabilities are never invented. Laya retains upstream truncation. Server-side truncation for Jev cannot be observed.

## Uncertainty

| Model | Overall % | 95% component-bootstrap interval |
|---|---:|---:|
| Lux-9B | 77.40 | 76.01–78.77 |
| Nox-4B | 73.09 | 71.57–74.56 |
| Kev-9B | 71.89 | 70.42–73.35 |
| Kev-4B | 70.09 | 68.45–71.63 |
| Qwen3.5-9B | 69.73 | 68.27–71.20 |
| Decider | 67.71 | 66.10–69.34 |
| Qwen3.5-4B | 67.29 | 65.89–68.69 |
| Sol-2B | 66.32 | 64.77–67.85 |
| Eos-0.8B | 61.89 | 60.26–63.53 |
| Kev-0.8B | 58.28 | 56.64–59.89 |
| Qwen3.5-2B | 57.24 | 55.74–58.76 |
| Kai-0.6B | 53.52 | 51.85–55.25 |
| Laya · English | 51.03 | 49.43–52.68 |
| Laya · Multilingual | 47.19 | 45.58–48.82 |
| Jev | 81.05 | 79.70–82.35 |

Intervals use 10,000 paired whole-source-component bootstrap draws. Related source, variant, parent, control and exact-state records stay together; all models receive the same draws. Five panels are sampled independently. These descriptive intervals do not create a positive-confidence-interval release requirement.
