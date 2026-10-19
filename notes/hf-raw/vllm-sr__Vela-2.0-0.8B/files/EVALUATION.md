# Evaluation: Vela 2.0 0.8B

[Model card](README.md) · [Usage](USAGE.md) · [Training](TRAINING.md)

The tables retain the full released results, including comparisons and task breakdowns. Test texts are held out of training; calibration and release-selection protocols are described separately below.

## Family results

<p align="center"><img src="https://huggingface.co/vllm-sr/Vela-2.0-0.8B/resolve/main/assets/vela2-family.jpg" alt="Four sizes, one family: 0.3B, 0.8B, 4B, 9B" width="80%" /></p>

| Model | Safety (AUC, 14 sets) | PII, 8K (F1; Vela 2.0 shipped calibration) | Hallucination (F1) | RAGTruth (F1) | fast-decisions | Jev Index (Vela 2.0: `noul_calibration=True`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vela 2.0 0.8B** | 0.875 | 0.603 | 0.828 | 0.652 | 56.8 | 16.01 |
| [Vela 2.0 0.3B (encoder)](https://huggingface.co/vllm-sr/Vela-2.0-0.3B) | 0.871 | 0.929 | 0.848 | 0.706 | 49.4 | – |
| [Vela 2.0 4B](https://huggingface.co/vllm-sr/Vela-2.0-4B) | **0.921** | 0.906 | 0.874 | 0.770 | 61.5 | 31.91 |
| [Vela 2.0 9B](https://huggingface.co/vllm-sr/Vela-2.0-9B) | 0.921 | **0.940** | 0.885 | 0.774 | 62.5 | **41.63** |
| Vela 1.0 specialists (one model per task) | – | 0.908 | 0.875 | 0.747 | – | – |
| [LettuceDetect v2 qwen-2b](https://huggingface.co/KRLabsOrg/lettucedect-v2-qwen-2b) (generative, hallucination only) | – | – | **0.921** | **0.817** | – | – |
| [LettuceDetect v2 mmBERT-base](https://huggingface.co/KRLabsOrg/lettucedect-v2-mmbert-base) (hallucination only) | – | – | 0.869 | 0.743 | – | – |
| [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | 0.704 | – | – | – | **62.9** | – |
| [Decision-2.0-Eos-0.8B](https://huggingface.co/vllm-sr/Decision-2.0-Eos-0.8B) (our base, Decision 2.0) | – | – | – | – | – | 20.14 |

The decoder checkpoints, temperatures and thresholds were selected on dev data only. The **0.3B encoder follows a different release-selection protocol**: final release selection also considered test results. Its 0.929 shipped 8K PII result uses a threshold floor lowered after the test effect was seen; the test-blind 0.001 floor gives 0.896, and the research scorer gives 0.894. The 0.3B fast-decisions result is not an independent zero-shot evaluation for that checkpoint. See the [0.3B evaluation disclosures](https://huggingface.co/vllm-sr/Vela-2.0-0.3B/blob/main/EVALUATION.md).

| Comparison detail | Protocol |
| --- | --- |
| LettuceDetect v2, RAGTruth | Re-scored by us on the same 2,700 rows, using its own evaluation input; reproduces its model cards |
| LettuceDetect v2, 10,698 examples | Values from its model cards on the same test set |
| Jev Decision Index | Edition 0.2.1, 38 general decision benchmarks; optional Noul calibration enabled |
| This decoder, default Jev mode | 16.01 with optional Noul calibration off |
| Decision 2.0 bases | Same harness, reproducing released cards within 0.1: Eos 20.14, Nox 42.55, Lux 46.23 |

The routing recipe trades part of the base model's general-decision score for the router tasks. The family table keeps the base and specialist references visible so that this tradeoff can be assessed.

## Router tasks against per-task references

PII rows in this table use the research scorer's per-length dev thresholds. All other metrics retain the released task definitions. The delta column is paired against the named task reference.

| Task | Reference model | Reference score | **Vela 2.0 0.8B (research PII thresholds)** | Δ points [95% CI] | Vela 2.0 4B | Vela 2.0 9B | Vela 2.0 0.3B |
|---|---|---:|---:|---:|---:|---:|---:|
| Request harm, RTP-LX (AUC) | Safety | 0.761 | 0.710 | −5.1 [−5.7, −4.5] | 0.774 | **0.801** | 0.728 |
| Hate speech, Multilingual HateCheck (AUC) | Safety | 0.646 | 0.653 | +0.7 [+0.1, +1.4] | 0.769 | **0.855** | 0.662 |
| Request harm, CultureGuard (AUC) | Safety | 0.863 | 0.850 | −1.3 [−1.6, −0.9] | 0.898 | **0.902** | 0.900 |
| Prompt attacks, unseen families (AUC) | Guard | 0.792 | 0.940 | +14.8 [+10.0, +19.3] | 0.985 | **0.989** | 0.882 |
| Domain (macro-F1) | Domain | 0.831 | 0.759 | −7.2 [−9.0, −5.5] | 0.820 | **0.844** | 0.825 |
| Modality (macro-F1) | Modality | 0.889 | **1.000** | +11.1 [+9.4, +12.7] | **1.000** | **1.000** | 0.961 |
| Fact-check needed (macro-F1) | FactCheck | 0.501 | 0.999 | +49.7 [+45.9, +53.4] | **1.000** | **1.000** | 0.997 |
| Feedback (macro-F1) | Feedback | 0.425 | 0.904 | +47.9 [+45.3, +50.5] | 0.920 | 0.939 | **0.989** |
| PII, short texts (F1) | PII | 0.976 | 0.984 | +0.8 [+0.1, +1.7] | 0.987 | 0.985 | **0.995** |
| PII, 4K-token documents (F1) | PII | 0.928 | 0.624 | −30.4 [−35.4, −26.2] | 0.898 | 0.965 | **0.987** |
| PII, 8K-token documents (F1) | PII | 0.908 | 0.603 | −30.5 [−35.1, −26.7] | 0.905 | **0.940** | 0.894 |
| Hallucination, 10,698 examples (F1) | Halu | 0.875 | 0.828 | −4.7 [−5.5, −4.0] | 0.874 | **0.885** | 0.848 |
| RAGTruth (F1) | LettuceDetect v2 mmBERT | 0.743 | 0.652 | −9.1 [−11.6, −6.6] | 0.770 | **0.774** | 0.706 |
| Toxic spans (char-F1) | mmBERT span head | **0.451** | 0.366 | −8.5 [−9.2, −7.8] | 0.379 | 0.446 | 0.423 |

### PII calibration modes

| Mode for this checkpoint | Short / 4K / 8K F1 |
| --- | --- |
| Research scorer, per-length dev thresholds | 0.984 / 0.624 / 0.603 |
| Shipped calibration, end to end | 0.984 / 0.624 / 0.603 |

The family summary uses shipped 8K calibration; the router comparison uses research-scoring thresholds. Both decoder modes are dev-selected. The 0.3B encoder's research and shipped results, and its test-informed release selection, are disclosed next to the family table above.

## Safety, all 14 sets (AUC)

| set | **Vela 2.0 0.8B** | Vela 2.0 4B | Vela 2.0 9B | Vela 2.0 0.3B | Vela 1.0 | GLiNER2.5-Decide |
|---|---:|---:|---:|---:|---:|---:|
| RTP-LX, request harm | 0.710 | 0.774 | **0.801** | 0.728 | 0.761 | 0.642 |
| Multilingual HateCheck | 0.653 | 0.769 | **0.855** | 0.662 | 0.646 | 0.602 |
| XSTest, request harm | 0.854 | 0.967 | **0.990** | 0.890 | 0.782 | 0.807 |
| CultureGuard, request harm | 0.850 | 0.898 | **0.902** | 0.900 | 0.863 | 0.674 |
| Prompt attacks, unseen families | 0.940 | 0.985 | **0.989** | 0.882 | 0.792 | 0.893 |
| Prompt attacks, known families | 0.948 | 0.956 | **0.962** | 0.950 | 0.820 | 0.569 |
| deepset prompt injections | 0.970 | 0.989 | **0.990** | 0.957 | 0.779 | 0.761 |
| CultureGuard, response harm | 0.860 | 0.937 | **0.941** | 0.916 | 0.828 | 0.624 |
| AEGIS 2.0, response harm | 0.883 | **0.934** | 0.931 | 0.917 | 0.872 | 0.660 |
| PolyGuard, response harm | 0.901 | 0.925 | **0.957** | 0.936 | 0.884 | 0.759 |
| Do-Not-Answer, response harm | 0.883 | **0.931** | 0.777 | 0.656 | 0.418 | 0.416 |
| Do-Not-Answer, refusal | **0.918** | 0.893 | 0.913 | 0.896 | – | 0.886 |
| XSTest, refusal | 0.973 | **0.992** | 0.989 | 0.955 | – | 0.887 |
| PolyGuard, refusal | 0.907 | 0.945 | 0.894 | **0.947** | – | 0.683 |
| **mean** | 0.875 | **0.921** | 0.921 | 0.871 | – | 0.704 |

These comparisons cover trained safety families. They are not zero-shot comparisons against GLiNER2.5-Decide, which was not trained on these families. Vela 1.0 uses Safety for harm and Guard for attacks; it has no refusal output.

## General and held-out decisions

| benchmark | **Vela 2.0 0.8B** | Vela 2.0 4B | Vela 2.0 9B | Vela 2.0 0.3B | reference |
|---|---:|---:|---:|---:|---|
| `fastino/fast-decisions` (17 domains) | 56.8 | 61.5 | 62.5 | 49.4 | GLiNER2.5-Decide 62.9 |
| MASSIVE, unseen scenarios: intent / scenario accuracy | 0.544 / 0.632 | 0.727 / 0.814 | 0.763 / 0.871 | 0.510 / 0.575 | – |
| QASC, 8-way science QA (held out) | 0.434 | 0.640 | 0.644 | 0.251 | chance 0.125 |
| Super-NI task 1568 (held out) | 0.340 | 0.494 | 0.574 | 0.370 | – |

Differences are paired bootstrap 95% intervals with 2,000 resamples. This decoder's checkpoint, temperatures and thresholds were chosen on dev data only. The 0.3B fast-decisions figure follows the encoder's separately disclosed release-selection protocol and is not an independent zero-shot result for that checkpoint.

## Open extraction benchmarks

The broad span head is evaluated on open-label NER, relations, extractive QA and evidence. **SQuAD v2 dev is not zero-shot for this model**: its training split is in the broad-head data. NER and ACL-Verbatim are evaluated with the held-out protocols below.

| Model | NER, 7 sets (F1) | Relations, CoNLL04 (F1) | SQuAD v2 dev (F1) | ACL-Verbatim evidence (word F1) | Latency, 3K-token context (ms) |
|---|---:|---:|---:|---:|---:|
| **Vela 2.0 0.8B, broad span head** | 28.9 | 3.0 | 27.1 | 23.6 | 477 |
| [GLiNER2.5-base](https://huggingface.co/fastino/gliner2.5-base-v1) | 55.9 | 8.5 | 45.5 | 2.8 | **360** |
| [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | 51.5 | 1.5 | 46.0 | 2.3 | 415 |
| [GLiFormer-base](https://huggingface.co/knowledgator/gliformer-base-v1) | 60.4 | **8.6** | 53.1 | 2.7 | 386 |
| [GLiFormer-large](https://huggingface.co/knowledgator/gliformer-large-v1) | **63.6** | 6.2 | 47.0 | 7.0 | 930 |
| [GLiNER large-v2.5](https://huggingface.co/gliner-community/gliner_large-v2.5) | 61.3 | 7.5 | **54.9** | 4.6 | 496 |
| [ACL-Verbatim ModernBERT](https://huggingface.co/KRLabsOrg/acl-verbatim-modernbert) (trained in-domain, evidence only) | – | – | – | **53.4** | – |

NER per set (CrossNER and MIT, strict micro-F1, label names only):

| Model | AI | Literature | Music | Politics | Science | Movie | Restaurant | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Vela 2.0 0.8B, broad span head** | 27.3 | 21.0 | 28.4 | 39.0 | 40.1 | 31.9 | 14.8 | 28.9 |
| [GLiNER2.5-base](https://huggingface.co/fastino/gliner2.5-base-v1) | 52.5 | 52.2 | 57.4 | 63.2 | 59.3 | **60.6** | 46.4 | 55.9 |
| [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | 48.8 | 48.0 | 56.2 | 63.5 | 47.0 | 55.5 | 41.6 | 51.5 |
| [GLiFormer-base](https://huggingface.co/knowledgator/gliformer-base-v1) | **56.4** | 62.6 | 67.6 | 71.4 | 67.5 | 56.8 | 40.5 | 60.4 |
| [GLiFormer-large](https://huggingface.co/knowledgator/gliformer-large-v1) | 54.2 | 60.3 | 66.7 | 71.8 | **70.0** | 60.4 | **61.9** | **63.6** |
| [GLiNER large-v2.5](https://huggingface.co/gliner-community/gliner_large-v2.5) | **56.4** | **65.6** | **71.4** | **72.4** | 65.0 | 53.7 | 44.9 | 61.3 |

| Evaluation | Protocol |
| --- | --- |
| Shared harness | Every model is scored by us on the same rows; reproduces the GLiFormer-base card on all seven NER sets and the gliner_large-v2.5 card average within 0.2 |
| NER | CrossNER and MIT; dataset's full label list per sentence, label names only; strict entity-level micro-F1 following GLiNER evaluation |
| Thresholds | GLiNER-family baselines: 0.5 defaults; this model: shipped broad-head threshold 0.35 |
| Relations | Per `(entity, relation type)`, ask “Which spans are the relation of entity?” over the sentence; tail-span micro-F1 on CoNLL04, SciERC, NYT11 and FewRel (one split) |
| Relation baselines | GLiNER-family models receive the same query as an entity label |
| SQuAD v2 | Train split is in the broad-head data; dev F1 is not zero-shot |
| ACL-Verbatim | Held out; paragraph-scale evidence in scientific papers |
| ACL-Verbatim ModernBERT reference | Trained on ACL-Verbatim itself, evidence only |
| Latency | One question per call, approximately 3K-token multi-passage context, mean over 100 questions after five warm-up calls, one A40 GPU for every model |

### Router versus broad head on zero-shot dev

These ablations are on held-out **dev sets**, distinct from the benchmark test results above. Values use the original 0–1 F1 scale.

| Task | Router span head | Broad span head |
| --- | ---: | ---: |
| NER | 0.113 | 0.259 |
| Relations | 0.107 | 0.229 |
| Verbatim evidence | 0.229 | 0.308 |

The broad head was added with the backbone, candidate head and router span head frozen. Router outputs were checked to be bitwise identical on 700 rows; see [training stages](TRAINING.md#training-stages).
