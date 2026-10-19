# Evaluation

Every model below scored the same rows. "Previous weights" is the first release of this repository, which also trained on each task's in-distribution test split. The Vela and mmBERT columns are the Vela base and mmBERT-32K trained on the suite's training split with the router repository's trainers: Vela with as many examples per task as the Decision model saw (learning rate and run picked on dev) and with the documented recipe (run picked on dev), mmBERT with the documented recipe. "Released models" are the published Vela task checkpoints, Vela Shield and the older mmbert32k classifiers, scored under this suite's labels. Files named `hold-` were not trained on by any of these models, except the slices marked in the note. Files named `test-` are the test splits of training corpora. The in-distribution test is not listed, since the previous weights trained on it. A file with one class reports recall or specificity at each model's own dev threshold. Hazard is scored over the eleven labels every model outputs. The note names a released model that trained on the same source.

| task | test set | rows | metric | this model | previous weights | Vela, matched budget | Vela, documented recipe | mmBERT | released models | note |
|---|---|---|---|---|---|---|---|---|---|---|
| domain | hold-arena-expert | 987 | accuracy | 0.740 | 0.740 | 0.769 | 0.728 | 0.716 | Vela 0.691; mmbert32k 0.584 |  |
| domain | hold-mmlu-cf | 1,988 | accuracy | 0.591 | 0.601 | 0.615 | 0.622 | 0.619 | Vela 0.529; mmbert32k 0.496 |  |
| domain | hold-mmlu-pro | 1,988 | accuracy | 0.668 | 0.662 | 0.671 | 0.641 | 0.648 | Vela 0.725; mmbert32k 0.925 | Vela Domain's recipe trains on the MMLU material behind 6,810 of its questions (see #1485); mmbert32k-intent trained on it |
| domain | hold-mmlu-prox | 1,988 | accuracy | 0.561 | 0.562 | 0.603 | 0.586 | 0.568 | Vela 0.660; mmbert32k 0.747 | translations of the MMLU-Pro test |
| domain | hold-supergpqa | 1,882 | accuracy | 0.713 | 0.728 | 0.734 | 0.725 | 0.736 | Vela 0.587; mmbert32k 0.606 |  |
| domain | hold-yahoo | 2,000 | accuracy | 0.547 | 0.549 | 0.535 | 0.492 | 0.482 | Vela 0.522; mmbert32k 0.451 |  |
| domain | test-exams | 1,927 | accuracy | 0.815 | 0.808 | 0.858 | 0.803 | 0.809 | Vela 0.516; mmbert32k 0.559 |  |
| fact_check | hold-factbench | 997 | recall at dev threshold | 0.506 | 0.394 | 0.354 | 0.359 | 0.321 | Vela 0.092; mmbert32k 0.060 |  |
| fact_check | hold-no_robots | 2,000 | AUC | 0.913 | 0.921 | 0.885 | 0.884 | 0.936 | Vela 0.962; mmbert32k 0.827 |  |
| fact_check | hold-shipped-factcheck | 2,000 | AUC | 0.844 | 0.865 | 0.839 | 0.803 | 0.845 | Vela 0.724; mmbert32k 0.925 | mmbert32k fact-check: its own dataset |
| fact_check | hold-simpleqa | 2,000 | recall at dev threshold | 0.977 | 0.945 | 0.559 | 0.632 | 0.589 | Vela 0.041; mmbert32k 0.912 |  |
| fact_check | hold-wildbench | 414 | AUC | 0.888 | 0.891 | 0.862 | 0.843 | 0.880 | Vela 0.852; mmbert32k 0.656 |  |
| feedback | hold-shipped-feedback | 1,715 | accuracy | 0.350 | 0.358 | 0.367 | 0.370 | 0.361 | Vela 0.311; mmbert32k 0.996 | mmbert32k feedback: its own training data |
| feedback | test-sgd | 1,998 | accuracy | 0.973 | 0.965 | 0.957 | 0.948 | 0.951 | Vela 0.688; mmbert32k 0.359 |  |
| hallucination | hold-attributionbench-ood | 1,390 | AUC | 0.861 | 0.862 | 0.831 | 0.876 | 0.858 | Vela 0.780 | slice: AttributionBench OOD subsets |
| hallucination | hold-hallumix | 2,000 | AUC | 0.739 | 0.738 | 0.613 | 0.655 | 0.602 | Vela 0.755 |  |
| hallucination | hold-halubench | 1,872 | AUC | 0.601 | 0.613 | 0.552 | 0.520 | 0.510 | Vela 0.527 |  |
| hallucination | hold-summedits | 578 | AUC | 0.774 | 0.774 | 0.692 | 0.686 | 0.699 | Vela 0.823 |  |
| hallucination | test-psiloqa | 1,265 | AUC | 0.872 | 0.859 | 0.843 | 0.813 | 0.821 | Vela 0.884 | Vela Halu: the train split |
| hallucination | test-ragbench | 1,311 | AUC | 0.628 | 0.620 | 0.580 | 0.604 | 0.599 | Vela 0.605 |  |
| hallucination | test-ragtruth | 1,528 | AUC | 0.669 | 0.680 | 0.646 | 0.671 | 0.676 | Vela 0.864 | Vela Halu: the train split |
| hazard | hold-ailuminate | 2,000 | macro AUC, 11 labels | 0.893 | 0.884 | 0.868 | 0.882 | 0.882 | Vela 0.837; Shield 0.808 |  |
| hazard | hold-aya-redteaming | 1,853 | macro AUC, 11 labels | 0.866 | 0.855 | 0.887 | 0.888 | 0.888 | Vela 0.742; Shield 0.744 | slice: two Aya languages |
| hazard | hold-beavertails-eval | 450 | macro AUC, 11 labels | 0.903 | 0.887 | 0.897 | 0.900 | 0.904 | Vela 0.863; Shield 0.850 |  |
| hazard | hold-mlcommons-synth | 2,000 | macro AUC, 11 labels | 0.972 | 0.960 | 0.967 | 0.957 | 0.958 | Vela 0.961; Shield 0.981 |  |
| hazard | test-aegis2 | 1,693 | macro AUC, 11 labels | 0.956 | 0.958 | 0.944 | 0.945 | 0.946 | Vela 0.880; Shield 0.947 | Shield and Vela Hazard: the train split |
| hazard | test-nemotron-v3 | 2,000 | macro AUC, 11 labels | 0.940 | 0.940 | 0.935 | 0.925 | 0.927 | Vela 0.832; Shield 0.915 | Shield and Vela Hazard: the train split |
| jailbreak | hold-bipia | 637 | AUC | 0.580 | 0.541 | 0.423 | 0.574 | 0.611 | Vela 0.664; Shield 0.629; mmbert32k 0.532 |  |
| jailbreak | hold-hackaprompt-late-levels | 2,000 | recall at dev threshold | 0.996 | 0.992 | 0.999 | 0.993 | 0.781 | Vela 0.001; Shield 0.961; mmbert32k 0.001 | slice: HackAPrompt levels 9-10 |
| jailbreak | hold-jailbreakhub-late | 1,303 | AUC | 0.826 | 0.824 | 0.807 | 0.795 | 0.746 | Vela 0.652; Shield 0.712; mmbert32k 0.752 | slice: JailbreakHub after June 2023 |
| jailbreak | hold-llmail | 1,152 | AUC | 0.969 | 0.979 | 0.760 | 0.791 | 0.970 | Vela 0.972; Shield 1.000; mmbert32k 0.867 | Shield: same dataset, phases unstated; Vela Guard recipe: phase 1 |
| jailbreak | hold-notinject | 339 | specificity at dev threshold | 0.699 | 0.693 | 0.717 | 0.785 | 0.782 | Vela 0.959; Shield 0.947; mmbert32k 1.000 |  |
| jailbreak | hold-promptshield-test | 2,000 | AUC | 0.767 | 0.691 | 0.677 | 0.695 | 0.763 | Vela 0.763; Shield 0.808; mmbert32k 0.705 |  |
| jailbreak | hold-toxicchat | 1,152 | AUC | 0.906 | 0.893 | 0.888 | 0.933 | 0.939 | Vela 0.915; Shield 0.928; mmbert32k 0.969 | mmbert32k jailbreak: the train half |
| modality | hold-alpaca | 2,000 | specificity at dev threshold | 0.910 | 0.867 | 0.893 | 0.961 | 0.956 | Vela 0.989; mmbert32k 0.997 | mmbert32k modality (card) |
| modality | hold-anyinstruct | 2,000 | AUC | 0.980 | 0.985 | 0.970 | 0.982 | 0.983 | Vela 0.984; mmbert32k 0.884 |  |
| modality | hold-arena-t2i-hard | 210 | recall at dev threshold | 0.971 | 0.971 | 0.938 | 0.967 | 0.957 | Vela 0.238; mmbert32k 0.776 |  |
| modality | hold-gedit-bench | 1,189 | recall at dev threshold | 0.974 | 0.992 | 0.998 | 0.983 | 0.981 | Vela 0.148; mmbert32k 0.031 |  |
| modality | hold-parti | 1,559 | recall at dev threshold | 0.999 | 0.999 | 0.999 | 0.997 | 0.999 | Vela 0.135; mmbert32k 0.158 | mmbert32k modality (card) |
| modality | hold-realmix | 3,769 | AUC | 0.994 | 0.994 | 0.988 | 0.986 | 0.985 | Vela 0.892; mmbert32k 0.851 |  |
| modality | hold-search-arena | 2,000 | specificity at dev threshold | 0.909 | 0.900 | 0.859 | 0.871 | 0.873 | Vela 0.988; mmbert32k 0.994 |  |
| modality | hold-shipped-modality | 2,000 | AUC | 0.995 | 0.993 | 0.972 | 0.995 | 0.994 | Vela 0.975; mmbert32k 0.982 | mmbert32k modality: its dataset |
| modality | test-dolly | 2,000 | specificity at dev threshold | 0.965 | 0.936 | 0.918 | 0.982 | 0.981 | Vela 1.000; mmbert32k 0.998 |  |
| modality | test-realedit | 2,000 | recall at dev threshold | 1.000 | 1.000 | 0.999 | 0.997 | 0.998 | Vela 0.066; mmbert32k 0.011 |  |
| pii | hold-ai4privacy | 2,000 | AUC | 0.977 | 0.964 | 0.954 | 0.973 | 0.964 | Vela 0.981; mmbert32k 0.954 |  |
| pii | hold-kaggle-essays | 1,125 | AUC | 0.987 | 0.987 | 0.984 | 0.992 | 0.994 | Vela 0.992; mmbert32k 0.983 |  |
| pii | hold-pii-prompts | 2,000 | AUC | 0.982 | 0.981 | 0.981 | 0.981 | 0.977 | Vela 0.987; mmbert32k 0.975 |  |
| pii | hold-wildchat | 2,000 | specificity at dev threshold | 0.814 | 0.842 | 0.822 | 0.828 | 0.842 | Vela 0.971; mmbert32k 0.976 |  |
| pii | test-abcd | 1,486 | AUC | 0.999 | 0.999 | 0.999 | 0.999 | 0.998 | Vela 0.990; mmbert32k 0.986 |  |
| pii | test-mapa | 1,679 | AUC | 0.984 | 0.978 | 0.934 | 0.964 | 0.960 | Vela 0.937; mmbert32k 0.863 |  |
| pii | test-tab | 1,461 | AUC | 0.994 | 0.994 | 0.988 | 0.991 | 0.987 | Vela 0.982; mmbert32k 0.975 |  |
| safety | hold-coconot | 897 | AUC | 0.989 | 0.892 | 0.997 | 0.995 | 0.995 | Vela 0.883; Shield 0.953 | slice of CoCoNot, whose training split this model and the encoders trained on (the previous weights left CoCoNot out) |
| safety | hold-coconot-contrast | 379 | specificity at dev threshold | 0.921 | 0.897 | 0.945 | 0.902 | 0.876 | Vela 0.887; Shield 0.955 | CoCoNot contrast set (the previous weights left CoCoNot out) |
| safety | hold-jbb | 200 | AUC | 0.921 | 0.899 | 0.875 | 0.879 | 0.873 | Vela 0.866; Shield 0.918 |  |
| safety | hold-openai-moderation | 1,473 | AUC | 0.896 | 0.891 | 0.889 | 0.878 | 0.885 | Vela 0.871; Shield 0.916 |  |
| safety | hold-orbench-hard | 1,319 | specificity at dev threshold | 0.646 | 0.700 | 0.557 | 0.732 | 0.694 | Vela 0.917; Shield 0.469 | slice: OR-Bench hard |
| safety | hold-toxicchat | 1,657 | AUC | 0.967 | 0.959 | 0.953 | 0.955 | 0.956 | Vela 0.906; Shield 0.971 | Shield: 179 rows through PolyGuardMix and Salad |
| safety | hold-xstest | 450 | AUC | 0.891 | 0.862 | 0.903 | 0.828 | 0.823 | Vela 0.782; Shield 0.926 |  |
| safety | test-aegis2 | 1,749 | AUC | 0.927 | 0.924 | 0.913 | 0.917 | 0.917 | Vela 0.919; Shield 0.943 | Shield and Vela Safety: the train split, which repeats test prompts |
| safety | test-nemotron-v3 | 2,000 | AUC | 0.902 | 0.894 | 0.890 | 0.882 | 0.886 | Vela 0.890; Shield 0.916 | Shield and Vela Safety: the train split |
| safety | test-polyguardprompts | 2,000 | AUC | 0.894 | 0.894 | 0.903 | 0.859 | 0.863 | Vela 0.799; Shield 0.924 | Shield: PolyGuardMix train |
| safety | test-wildguardmix | 1,696 | AUC | 0.927 | 0.925 | 0.925 | 0.908 | 0.909 | Vela 0.806; Shield 0.937 | Shield: through PolyGuardMix |
| safety | test-wildjailbreak-eval | 1,206 | AUC | 0.940 | 0.930 | 0.949 | 0.923 | 0.927 | Vela 0.864; Shield 0.938 |  |
| tool_need | hold-bfcl | 1,949 | AUC | 0.903 | 0.900 | 0.832 | 0.837 | 0.815 |  |  |
| tool_need | hold-mix | 3,949 | AUC | 0.907 | 0.910 | 0.843 | 0.838 | 0.840 |  |  |
| tool_need | hold-xlam-irrelevance | 2,000 | specificity at dev threshold | 0.989 | 0.983 | 0.968 | 0.952 | 0.968 |  |  |
| tool_need | test-when2call-test | 1,217 | specificity at dev threshold | 0.960 | 0.929 | 0.891 | 0.915 | 0.910 |  |  |

Paired over the files with both classes, with a bootstrap over groups and a Benjamini-Hochberg correction at 5 percent, this model wins, ties and loses 26 / 24 / 7 against the matched-budget Vela, 32 / 19 / 6 against the documented Vela recipe and 28 / 22 / 7 against mmBERT on 57 files (the files above with both classes except the shipped-classifier files and BFCL, plus each task's in-distribution test, which none of these four trained on), 36 / 12 / 7 against the released Vela checkpoints on the 55 files they cover and 7 / 9 / 8 against Vela Shield on its 24. The previous weights are not paired here, since they trained on the in-distribution tests; on the fresh set ([FRESH.md](FRESH.md)) this model is 5 / 22 / 5 against them.

## Serving through the Decision runtime

It is meant to be served by the vLLM Semantic Router Decision runtime ([semantic-router#4086](https://github.com/vllm-project/semantic-router/pull/4086), not merged yet). Loaded with that runtime's `VelaTorchRuntime` and the Kai profile, on 200 in-distribution test rows each of domain, modality, feedback, jailbreak, safety, fact check and PII, its probabilities match the training runtime within 5e-06 and no answer changes. That runtime sends each Choice option as `KEY: description`, the form these weights were trained on.

## Latency and memory

Measured on the training-split run, which has the same architecture. Median per request, fp32, one request at a time, 205 requests, with 3 Choice and 5 Noul questions in two calls against eight Vela checkpoints (the seven released request-time ones and a tool-need model trained the same way):

| per request, median | CPU (Xeon Platinum 8568Y+, 8 threads) | GPU (MI300X) |
|---|---|---|
| this model, both calls in sequence | 409.0 ms | 47.2 ms |
| this model, slower call (calls in parallel) | 237.6 ms | 23.9 ms |
| eight Vela checkpoints in sequence | 189.0 ms | 82.0 ms |
| eight Vela checkpoints, slowest one | 25.4 ms | 10.9 ms |

Every question re-reads the request, so cost grows with the number of questions asked. Batched on the GPU with requests sorted by length, it serves 100.8 requests per second through [semantic-router#4086](https://github.com/vllm-project/semantic-router/issues/4086)'s executor, against 102.6 for the eight checkpoints. Through [semantic-router#4086](https://github.com/vllm-project/semantic-router/issues/4086)'s scheduler, which batches in arrival order, it serves about 37, and 66 at batch 64 with [semantic-router#4310](https://github.com/vllm-project/semantic-router/pull/4310). Peak GPU memory was 5.35 GiB, against 10.27 GiB for the eight checkpoints.
