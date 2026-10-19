# Fresh held-out set

A second held-out set, built before these weights were trained and scored once at the end: 44 files and 57,805 rows from 37 public corpora that the suite in [semantic-router#4305](https://github.com/vllm-project/semantic-router/issues/4305) never touched, with exact and near duplicates (MinHash, Jaccard 0.8) of any suite row removed (rows added to training later were checked against it the same way), frozen before any of the changes behind these weights was tried. It is used for evaluation only; nothing was chosen on it. The encoders are the ones on the model card, and the previous weights of this repository are scored as the Decision runtime in [semantic-router#4086](https://github.com/vllm-project/semantic-router/pull/4086) serves them.

Mean over each signal's files with both classes (domain and feedback by accuracy):

| signal | files | metric | this model | previous weights (served) | Vela, matched budget | Vela, documented recipe | mmBERT-32K |
|---|---:|---|---:|---:|---:|---:|---:|
| domain | 5 | mean accuracy | 0.497 | 0.523 | 0.508 | 0.510 | 0.515 |
| feedback | 2 | mean accuracy | 0.882 | 0.858 | 0.869 | 0.862 | 0.871 |
| jailbreak | 2 | mean AUC | 0.691 | 0.735 | 0.633 | 0.675 | 0.661 |
| | | partial AUC to 5 percent false positives | 0.519 | 0.519 | 0.522 | 0.508 | 0.552 |
| safety | 5 | mean AUC | 0.898 | 0.889 | 0.905 | 0.891 | 0.897 |
| | | partial AUC to 5 percent false positives | 0.714 | 0.703 | 0.732 | 0.710 | 0.714 |
| hazard | 2 | mean macro AUC | 0.944 | 0.942 | 0.919 | 0.926 | 0.929 |
| | | partial AUC to 5 percent false positives | 0.805 | 0.807 | 0.749 | 0.762 | 0.761 |
| fact check | 3 | mean AUC | 0.812 | 0.809 | 0.817 | 0.816 | 0.835 |
| | | partial AUC to 5 percent false positives | 0.587 | 0.579 | 0.658 | 0.686 | 0.685 |
| PII | 4 | mean AUC | 0.978 | 0.962 | 0.908 | 0.945 | 0.920 |
| | | partial AUC to 5 percent false positives | 0.920 | 0.885 | 0.846 | 0.898 | 0.879 |
| hallucination | 6 | mean AUC | 0.727 | 0.732 | 0.708 | 0.739 | 0.722 |
| | | partial AUC to 5 percent false positives | 0.559 | 0.557 | 0.569 | 0.574 | 0.566 |
| tool need | 3 | mean AUC | 0.769 | 0.770 | 0.767 | 0.773 | 0.773 |
| | | partial AUC to 5 percent false positives | 0.565 | 0.545 | 0.542 | 0.540 | 0.544 |

Paired file by file with a bootstrap over groups and a Benjamini-Hochberg correction at 5 percent, this model wins, ties and loses 10 / 17 / 5 against the matched-budget Vela, 10 / 19 / 3 against the documented recipe and 9 / 20 / 3 against mmBERT-32K on 32 files; 17 / 7 / 5 against the released Vela checkpoints on the 29 files they cover and 3 / 1 / 5 against Vela Shield on its 9. Against the previous weights of this repository it is 5 / 22 / 5.

It has the highest mean of the five on hazard, PII and feedback, and leads the three encoders on jailbreak. It trails all three encoders on domain and on fact check, most at low false-positive rates (partial AUC 0.071 to 0.099 behind), and the matched-budget Vela on safety (0.007 AUC). Against the previous weights it gains on PII, feedback and safety and loses on jailbreak and domain.

Files with one class, at the threshold that gives this model 5 percent false positives on dev: it flags 63% of MGSM math questions and none of MBPP's coding tasks as needing a fact check, and 82% of Mintaka and 45% of TruthfulQA questions; it catches 98% of Tensor Trust attacks, 49% of CyberSecEval's indirect injections, 48% of the IPI arena attacks and 36% of b3's; 92% of CatQA's harmful questions; and routes 100% of Emu Edit and 97% of OneIG-Bench Chinese requests to image output while sending 1.4% of text requests there.
