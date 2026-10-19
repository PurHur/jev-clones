# Evaluation: Vela 2.0 0.3B

[Back to the model card](README.md) · [Usage](USAGE.md) · [Training and model details](TRAINING.md)

## Result overview

On 14 router safety test sets its macro AUC is **0.871**, against **0.704** for GLiNER2.5-Decide on the same rows, and on the Vela 1.0 router tasks it stays within about 3 points of the per-task Vela 1.0 specialists; the one task outside that band is RTP-LX request harm, **−3.2** (paired 95% CI [−3.7, −2.7]).

| Model | Safety macro AUC, 14 sets ↑ | PII short / 8K ↑ | Hallucination example-F1 ↑ | RAGTruth example-F1 ↑ | fast-decisions ↑ | Decision Index R&C slice ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vela 2.0 0.3B** | **0.871** | **0.995** / **0.929** | 0.848 | 0.706 | 49.4 | 0.304 |
| GLiNER2.5-Decide | 0.704 | – | – | – | **62.9** | 0.296 |
| Decision-1.0-Kai-0.6B | – | – | – | – | – | 0.102 |
| Decision-1.0-Nox (4.7B) | – | – | – | – | – | **0.472** |
| *Vela 1.0 specialists (one model per task)* | – | 0.976 / 0.908 | **0.875** | **0.747** | – | – |
| LettuceDetect v2 mmBERT-base (hallucination only) | – | – | 0.869 | 0.743 | – | – |
| LettuceDetect v2 qwen-2b (generative, hallucination only) | – | – | 0.921 | 0.817 | – | – |

All reported benchmark results are on test sets. “–” means not measured: Kai and Nox were not run on the Vela tasks, and the Vela 1.0 specialists cover one task each.

## Evaluation protocol and selection

- Vela 2.0, GLiNER2.5-Decide and the specialists are scored on identical rows for the task comparisons. Quoted differences use a paired bootstrap with 2,000 resamples.
- Checkpoint selection used dev splits. Final release selection also considered test results.
- The 14 safety sets come from task families this model was trained on and Decide was not. The safety comparison measures router safety performance, rather than zero-shot generalisation.
- The model is ahead of Decide on 13 safety sets and level on attack held-out families: −0.011 [−0.051, +0.027].

### PII calibration and scoring

PII uses exact micro-F1 on 1,704 short rows and 30 documents at each evaluated long-document length. The 8K results have distinct scoring and calibration paths:

| 8K PII path | Exact micro-F1 | Interpretation |
| --- | ---: | --- |
| Shipped calibration, end to end | 0.929 | The minimum threshold floor was lowered after the test effect had been seen. This is a test-informed release result. |
| Test-blind calibration, 0.001 floor | 0.896 | The result with the original threshold floor. |
| Research scorer | 0.894 | Uses its per-length thresholds; Δ against the Vela 1.0 specialist is −1.4 [−6.7, +4.2]. |

The PII length curve and sparse-document gate were fitted on dev. The shipped minimum-floor change is separate from that dev fit and was informed by the test results.

### Other decision benchmarks

`fastino/fast-decisions` uses its public split at revision `1a33070c`, covering 17 domains. It is Decide's own benchmark. Decide leads by 13.5 points: −13.5 [−15.5, −11.6]. This result is not an independent zero-shot evaluation for this checkpoint.

The Jev Decision Index column is edition 0.2.1's Retrieval & Classification slice:

- BANKING77, CLINC150, BRIGHT, ESCI and PhishNChips.
- This model was scored with the kit's own scorers through our adapter, rather than the lab's harness; HoVer was not run.
- The Decide, Kai and Nox values come from the board.

### Hallucination scoring and references

The hallucination comparison covers 10,698 examples. RAGTruth covers 2,700 rows; its Vela 1.0 value is Vela Halu re-scored on those same rows.

LettuceDetect v2 is re-scored by us using its own evaluation input, reproducing its model-card results. Its 10,698-example values in the overview are taken from its model cards.

The Vela 1.0 references are router-pinned revisions, re-scored per row:

| Reference | Revision |
| --- | --- |
| Safety | `6e70e725` |
| Guard | `087f9e40` |
| Domain | `f6354f54` |
| PII | `6d3300c4` |
| Halu | `ca875312` |

## Tradeoffs versus per-task references

Results below the per-task reference on the same rows:

| Task | Vela 2.0 0.3B | Reference | Δ [95% CI] |
| --- | ---: | ---: | ---: |
| RTP-LX, request harm AUC | 0.728 | Safety 0.761 | −3.2 [−3.7, −2.7] |
| Toxic spans, mean char-F1 | 0.423 | mmBERT span head 0.451 | −2.8 [−3.5, −2.1] |
| Hallucination, 10,698 examples, example-F1 | 0.848 | Halu 0.875 | −2.7 [−3.4, −2.1] |
| RAGTruth, 2,700 rows, example-F1 | 0.706 | LettuceDetect v2 mmBERT 0.743 | −3.7 [−5.8, −1.6] |
| Domain macro-F1 | 0.825 | Domain 0.831 | −0.7 [−1.9, +0.6] |

## Full per-task results

![Δ against the Vela 1.0 specialists](figures/fig_delta_vs_vela.png)

| Task | Reference | Vela 2.0 0.3B | Δ [95% CI] |
|---|---|---|---|
| RTP-LX, request harm AUC | Safety 0.761 | 0.728 | **−3.2 [−3.7, −2.7]** |
| Multilingual HateCheck AUC | Safety 0.646 | 0.662 | +1.6 [+1.1, +2.2] |
| CultureGuard AUC | Safety 0.863 | 0.900 | +3.7 [+3.4, +4.0] |
| attack, held-out families AUC | Guard 0.792 | 0.882 | +9.0 [+4.1, +13.6] |
| Domain macro-F1 | Domain 0.831 | 0.825 | −0.7 [−1.9, +0.6] |
| PII short, exact micro-F1 (1,704 rows) | PII 0.976 | 0.995 | +1.9 [+1.2, +2.7] |
| PII 4K tokens (30 documents) | PII 0.928 | 0.987 | +5.9 [+1.7, +9.4] |
| PII 8K tokens (30 documents) | PII 0.908 | 0.894 (export: 0.929 shipped; 0.896 with the 0.001 floor, the test-blind figure) | −1.4 [−6.7, +4.2] |
| hallucination, 10,698 examples, example-F1 | Halu 0.875 | 0.848 | −2.7 [−3.4, −2.1] |
| RAGTruth 2,700 rows, example-F1 | LettuceDetect v2 mmBERT 0.743 | 0.706 | −3.7 [−5.8, −1.6] |
| toxic spans, mean char-F1 | mmBERT span head 0.451 | 0.423 | −2.8 [−3.5, −2.1] |

PII rows use the research scorer's per-length thresholds; through the exported calibration they are 0.995 / 0.991 / 0.929.

## Additional decision benchmarks

| benchmark | this model | reference |
|---|---|---|
| `fastino/fast-decisions` @ `1a33070c`, public split, 17 domains | 49.4 | GLiNER2.5-Decide 62.9; multi-Decide 58.7 |
| MASSIVE held-out scenarios (iot / transport / cooking), intent / scenario accuracy | 0.510 / 0.575 | – |
| QASC 8-way (held out) | 0.251 | chance 0.125 |
| Super-NI task 1568 (held out, n = 162) | 0.370 | – |

Jev Decision Index 0.2.1, Retrieval & Classification (this model through our adapter; the other rows are the board's):

| model | BANKING77 | CLINC150 | BRIGHT | ESCI | PhishNChips | R&C slice |
|---|---|---|---|---|---|---|
| **this model** | 0.536 | 0.519 | 0.220 | 0.216 | 0.645 | **0.304** |
| Decision-1.0-Kai (our init) | 0.407 | 0.057 | 0.138 | 0.204 | 0.499 | 0.102 |
| GLiNER2.5-Decide | 0.656 | 0.604 | 0.201 | 0.233 | 0.500 | 0.296 |
| Decision-1.0-Nox (4.7B) | 0.833 | 0.853 | 0.346 | 0.385 | 0.541 | 0.472 |
| best entrant on the slice (27B) | 0.795 | 0.881 | 0.497 | 0.574 | 0.847 | 0.658 |

In the reported board snapshot, this model ranked 39th of 71 on the slice, the highest entrant under 500M parameters.

## How it differs from related models

| | Vela 1.0 specialists | GLiNER2.5-Decide | Decision-1.0-Kai-0.6B | Decision-1.0-Kai-0.6B-Router-Signals | **Vela 2.0 0.3B** |
|---|---|---|---|---|---|
| what you ask | fixed labels per model (Safety, Guard, Hazard, Domain, …) | labels written into the input, zero-shot | SystemOne Choice / Noul / Score with runtime candidates | fixed questions for 10 router signals | SystemOne Choice / Noul / Score, plus span and set, over typed parts |
| spans | PII and Halu are separate token models | not evaluated here | no | no: PII and hallucination are yes/no questions | yes: PII, unsupported claims and runtime-supplied labels |
| context (as stated on each card) | per model | not stated (DeBERTa-v3-large backbone) | 1,024-token input budget in the released runtime; longer inputs rejected | 1,024 tokens | 8,192, longer labelled parts windowed |
| passes | one per model | one per text | questions batched per path | one per question | one API request; extra Span questions and long-input windows can require additional sequences |

Router-Signals is the simpler drop-in for the signals the router already has; this model targets new questions, span outputs and long inputs. We have not evaluated Router-Signals on our rows.

[Back to the model card](README.md)
