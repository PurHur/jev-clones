# Source inventory

Paths are relative to the repository root unless they are links. A module path such as
`data/recipes.py` is relative to `src/strands_decider/`, but a data file such as `data/train_v5.jsonl`,
and the scripts under `data/generators/` and `data/checks/`, are relative to the repository
root.

This table lists every input of the current recipe (`recipe.sh`), the v19 route and the v20
route. The licence column gives the licence or terms that each source states, with a link to
the page that states them. Where a Hugging Face card and the original release disagree, the
column gives the two. Card fields are informational, because a card licence is the uploader's
claim.

| Source | Pinned revision | Role | Transformation | Committed or downloaded | Licence | Attribution |
| --- | --- | --- | --- | --- | --- | --- |
| 21 short-task classification datasets on Hugging Face ([Short-task datasets](#short-task-datasets)) | No: `load_dataset` gets no revision | Train | `strands-decider data build` (`data/recipes.py`) turns each row into a typed question. Large label sets are subsampled to at most 24 options, and each task is capped at a fixed row count. | Downloaded | Per dataset, in [Short-task datasets](#short-task-datasets). | The dataset card of each Hugging Face id |
| 7 held-out short tasks ([Short-task datasets](#short-task-datasets)) | No | Held out: `emotion`, `hate_severity`, `massive_intent` and `sarcasm` calibrate and evaluate. The three RuleTaker depths are held out of training only. | The same build. `--holdout` writes them to `data/train_v5.holdout.jsonl`. `data/holdout_v5_norule.jsonl` drops the RuleTaker depths. | Downloaded | Per dataset, in [Short-task datasets](#short-task-datasets). `emotion` is for educational and research use only. | The dataset card of each Hugging Face id |
| ContractNLI (release zip) | No (hash-checked after `fetch`) | Train and evaluation | `data/multistep.py`: 17 claims per NDA, balanced per claim | Downloaded by `recipe.sh fetch` | CC BY 4.0 ([project page](https://stanfordnlp.github.io/contract-nli/)) | Koreeda & Manning (Findings of EMNLP 2021), Hitachi America |
| MuSiQue, full v1.0 (release zip) | No (hash-checked after `fetch`) | Train and evaluation | `data/multistep.py`: each question with its answerable and unanswerable versions | Downloaded by `recipe.sh fetch` | CC BY 4.0 ([repository](https://github.com/StonyBrookNLP/musique)) | Trivedi et al., TACL 2022 |
| BoardgameQA (`tasksource/Boardgame-QA`) | No | Train and evaluation | `data/multistep.py`: at most a quarter of the multi-step rows | Downloaded | CC BY 4.0 ([card](https://huggingface.co/datasets/tasksource/Boardgame-QA)) | Kazemi et al., NeurIPS 2023 |
| tasksource-jev-typed-decisions, `full` config (`tasksource/tasksource-jev-typed-decisions`), v25 | Yes: `d2ab1d12be4463fb7ac1f877ed2193b3bb59642a` (sha256 of each parquet file and of the output in the manifest) | Train (v25 data round) | `data/tasksource_typed.py`: 127 English sources whose every listed licence is CC0, MIT, Apache, BSD, CC BY or CC BY-SA, after source exclusions and a hand audit; 3,000 rows per source; rows overlapping evaluation sets by 13-gram dropped. `data/tasksource_typed_v25.manifest.json` lists every source with its licence, verdict and counts. | Downloaded and converted, not committed | Per source (manifest); 47,723 of the rows are CC BY-SA | Sileo, LREC-COLING 2024, and the source of each row (`task` = `ts/<source>`) |
| HotpotQA (`hotpotqa/hotpot_qa`, distractor) | No | Evaluation only | `data/multistep.py`: comparison questions as yes/no or a choice between two titles | Downloaded | CC BY-SA 4.0, ShareAlike ([card](https://huggingface.co/datasets/hotpotqa/hotpot_qa)) | Yang et al., EMNLP 2018 |
| HelpSteer2 (`nvidia/HelpSteer2`) | No: `resolve/main` (hash-checked after `fetch`) | Train and evaluation | `data/adequacy.py`: ratings thresholded to adequate or inadequate, classes balanced | Downloaded by `recipe.sh fetch`, not redistributed | CC BY 4.0 ([card](https://huggingface.co/datasets/nvidia/HelpSteer2)) | Wang et al. (NVIDIA, 2024), arXiv:2406.08673 |
| Generated document questions, v16 and v18 (`data/synthetic/generated_v16*.jsonl`, `generated_v18*.jsonl`) | Yes: committed files | Train, and evaluation on held-out domains | Written by Qwen3.6-27B and kept when Qwen3.5-397B-A17B agreed, through OpenRouter. `data/generated.py` balances them. | Committed, with the raw exports in `data/generators/gen_*/` | Models: Apache-2.0 ([Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)). The model licence sets no terms on the generated text. The committed files are under the repository licence, Apache-2.0 ([LICENSE](../LICENSE)). | Qwen3.6-27B, Qwen3.5-397B-A17B |
| Generated adequacy items, v19 (`data/synthetic/adequacy_gen*.jsonl`) | Yes: committed files | Train and evaluation | The same writer and verifier. Each inadequate response carries one assigned defect. | Committed (`data/generators/gen_adequacy/`) | Models: Apache-2.0 ([Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)). The model licence sets no terms on the generated text. The committed files are under the repository licence, Apache-2.0 ([LICENSE](../LICENSE)). | Qwen3.6-27B, Qwen3.5-397B-A17B |
| Instruction-flip pairs, v20 (`data/synthetic/flips_v20*.jsonl`) | Yes: committed files | Train (v20) and evaluation | The same writer and verifier. Each short policy is asked twice, with instructions that reverse the answer. | Committed (`data/generators/gen_flips/`) | Models: Apache-2.0 ([Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)). The model licence sets no terms on the generated text. The committed files are under the repository licence, Apache-2.0 ([LICENSE](../LICENSE)). | Qwen3.6-27B, Qwen3.5-397B-A17B |
| Question paraphrases, v20 (`data/generators/gen_paraphrases/paraphrases.jsonl`) | Yes: committed file | Train (v20) and the paired consistency evaluation | Written by Qwen3.6-27B and checked by Qwen3.5-397B-A17B. `recipe.sh generated` attaches them as instruction variants. | Committed | Models: Apache-2.0 ([Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)). The model licence sets no terms on the generated text. The committed files are under the repository licence, Apache-2.0 ([LICENSE](../LICENSE)). | Qwen3.6-27B, Qwen3.5-397B-A17B |
| Catch-all rows, v20 (`data/catchall_v20*.jsonl`) | Follows the short-task corpus | Train (v20) and evaluation | `data/catchall.py` adds "other" or "none of these" options to the corpus's choice rows | Derived by `recipe.sh catchall`, not committed | Follows the short-task datasets | Not applicable |
| Frozen-teacher distributions from `Qwen/Qwen3.5-4B` | Yes: revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (`data/teacher.py`) | Training targets | v19 route: `recipe.sh teacher` labels the multi-step rows, and the parent trains toward them. v20 route: `distill` uses the committed labels of the short-task corpus (`data/synthetic/teacher_v5_qwen35-4b.jsonl`). | The model is downloaded. The short-task labels are committed, without the 21,000 `score` rows (Yelp, SST-5, app_reviews, formality); `python -m strands_decider.data.teacher` regenerates them. `data/synthetic/teacher_multistep_v14.jsonl` is committed too, but no step reads it. | Model: Apache-2.0 ([card](https://huggingface.co/Qwen/Qwen3.5-4B)). The distributions label the short-task and multi-step rows, so the terms of those rows also apply. | Qwen |
| Frozen-teacher distributions from `google/gemma-4-31B-it` (g4) | Yes: revision `842da3794eaa0b77d5f08bae87a17459d91ff475` (`training/recipe.sh teacher31b`) | Training targets (g4 only) | `teacher31b` merges the committed labels of the short-task, generated and adequacy rows (`data/synthetic/teacher_gemma4-31b-it_*.jsonl`) with v14's replay distributions into `data/teacher_g4.jsonl` (`training/merge_teacher_g4.py`) | Committed, without the 21,000 `score` rows of the short-task corpus, as for the Qwen3.5-4B labels; `RELABEL=1 training/recipe.sh teacher31b` regenerates them with the model | Apache-2.0 | Google |
| Code-task rows, v27 build (g26 `+code`: `data/code_v27.jsonl`, `data/code_v27_eval.jsonl`) | Yes: the build's sources are pinned (dataset revisions, repository commits; its `manifest.json`) | Train (g26 `+code` arms) and the code held-out evaluation | `research/scripts/g26_code_data.py` takes the build's 10 % mix (`mix/code10.jsonl`) without the rows past 4,096 tokens, and 1,996 held-out rows not in it. Sources: generated programs with execution-checked answers; functions from 286 permissive GitHub repositories; CuBERT (5 tasks); CommitPackFT; CodeReviewer; CodeContests; Juliet C/C++ 1.3 | Built by the v27 code-data build, s3 `data-v27-code/` (its `v27-code.md` has the per-source table, decontamination and licence gate) | Generated programs: none (written here); repositories: MIT / Apache-2.0 / BSD-2/3-Clause / ISC per repository; CuBERT: Apache-2.0; CommitPackFT: MIT (+ per-row repository licence); CodeReviewer: CC BY 4.0; CodeContests: CC BY 4.0; Juliet: public domain (NIST) / CC0-1.0 | Per repository (commit, licence file in the build's `repos.json`); Google (CuBERT); BigCode (CommitPackFT); Microsoft (CodeReviewer); DeepMind (CodeContests); NIST (Juliet) |
| Replay distributions | v19 route: regenerated. v20 route: yes, committed file. | Training targets | v19 route: `recipe.sh replay` labels the multi-step rows with the parent checkpoint's own distributions, and v19 trains toward them. v20 route: `distill` uses v14's distributions (`data/synthetic/replay_v14_multistep.jsonl`). | Regenerated, or committed | Not applicable: the outputs of this project's own checkpoints | Not applicable |
| Base model `Qwen/Qwen3.5-2B-Base` | No: the loader passes no revision ([Model artifact](../docs/inference.md#model-artifact)) | The torso for training and inference | LoRA adapters and the readout are trained on it | Downloaded from Hugging Face | Apache-2.0 ([card](https://huggingface.co/Qwen/Qwen3.5-2B-Base)) | Qwen |
| Base model `google/gemma-4-E2B-it` (g4, g26) | No: the loader passes no revision (Hub `main` on 2026-10-06: `3e22461f65e89153144f8adb70e3b8c2cc9845a7`) | The torso for training and inference (`configs/experiments/g4.yaml`, `g26-e2b*.yaml`) | LoRA adapters and the readout are trained on its text decoder; the vision and audio encoders are dropped at load | Downloaded from Hugging Face (not gated) | Apache-2.0 ([Gemma 4 license page](https://ai.google.dev/gemma/docs/gemma_4_license), checked 2026-10-06; earlier Gemma generations used the Gemma Terms of Use instead) | Google |
| Base model `google/gemma-4-E4B-it` (g4-e4b, g26) | No: the loader passes no revision (Hub `main` on 2026-10-06: `ee0ef6023621cff504d758262d4e04895a5af4a2`) | The torso for training and inference (`configs/experiments/g4-e4b.yaml`, `g26-e4b*.yaml`) | LoRA adapters and the readout are trained on its text decoder; the vision and audio encoders are dropped at load | Downloaded from Hugging Face (not gated) | Apache-2.0 ([Gemma 4 license page](https://ai.google.dev/gemma/docs/gemma_4_license), checked 2026-10-06; earlier Gemma generations used the Gemma Terms of Use instead) | Google |
| Base model `google/gemma-4-12B-it` (g26) | No: the loader passes no revision (Hub `main` on 2026-10-06: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`) | The torso for training and inference (`configs/experiments/g26-12b*.yaml`) | LoRA adapters and the readout are trained on its text decoder; the vision and audio encoders are dropped at load | Downloaded from Hugging Face (not gated) | Apache-2.0 ([Gemma 4 license page](https://ai.google.dev/gemma/docs/gemma_4_license), checked 2026-10-06; earlier Gemma generations used the Gemma Terms of Use instead) | Google |
| Base model `google/gemma-4-26B-A4B-it` (g26) | No: the loader passes no revision (Hub `main` on 2026-10-06: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`) | The torso for training and inference (`configs/experiments/g26-26b-a4b*.yaml`) | LoRA adapters and the readout are trained on its text decoder; the vision and audio encoders are dropped at load | Downloaded from Hugging Face (not gated) | Apache-2.0 ([Gemma 4 license page](https://ai.google.dev/gemma/docs/gemma_4_license), checked 2026-10-06; earlier Gemma generations used the Gemma Terms of Use instead) | Google |

## Short-task datasets

`strands-decider data build` reads these Hugging Face datasets through `data/recipes.py`. The
licence column gives the card's licence field first, then the terms of the original release
when the card says `unknown`, `other` or nothing. All links were read on 2026-10-03.
"ShareAlike" marks a CC BY-SA source.

| Recipe | Hugging Face id (config) | Question type | Role | Licence |
| --- | --- | --- | --- | --- |
| `ag_news` | `fancyzhx/ag_news` | choice | train | Card: unknown. Original: non-commercial use only ([card](https://huggingface.co/datasets/fancyzhx/ag_news), [AG corpus page](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)). |
| `banking77` | `legacy-datasets/banking77` | choice | train | CC BY 4.0 ([card](https://huggingface.co/datasets/legacy-datasets/banking77), [LICENSE](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE)) |
| `clinc150` | `clinc/clinc_oos` (`plus`) | choice | train | CC BY 3.0 ([card](https://huggingface.co/datasets/clinc/clinc_oos), [LICENSE](https://github.com/clinc/oos-eval/blob/master/LICENSE)) |
| `dbpedia` | `fancyzhx/dbpedia_14` | choice | train | CC BY-SA 3.0, ShareAlike ([card](https://huggingface.co/datasets/fancyzhx/dbpedia_14)) |
| `lang_id` | `papluca/language-identification` | choice | train | Card: none. Its sources include XNLI, CC BY-NC 4.0 ([LICENSE](https://github.com/facebookresearch/XNLI/blob/main/LICENSE)), and the Multilingual Amazon Reviews Corpus, academic research only ([card](https://huggingface.co/datasets/defunct-datasets/amazon_reviews_multi)). |
| `yahoo_topics` | `community-datasets/yahoo_answers_topics` | choice | train | Card: unknown. Original: non-commercial research under a signed Yahoo data agreement ([card](https://huggingface.co/datasets/community-datasets/yahoo_answers_topics), [dataset readme](https://github.com/SALT-NLP/MixText/blob/master/data/yahoo_answers_csv/readme.md)). |
| `spam` | `ucirvine/sms_spam` | noul | train | Card: unknown. Original: CC BY 4.0 ([card](https://huggingface.co/datasets/ucirvine/sms_spam), [UCI page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)) |
| `toxicity` | `google/civil_comments` | noul | train | CC0 1.0 ([card](https://huggingface.co/datasets/google/civil_comments)) |
| `mnli_entail` | `nyu-mll/glue` (`mnli`) | noul | train | Card: other, per task. MultiNLI card: CC BY 3.0, CC BY-SA 3.0, MIT and other (the OANC licence). ShareAlike for one fiction source ([GLUE card](https://huggingface.co/datasets/nyu-mll/glue), [MultiNLI card](https://huggingface.co/datasets/nyu-mll/multi_nli)). |
| `boolq` | `google/boolq` | noul | train | CC BY-SA 3.0, ShareAlike ([card](https://huggingface.co/datasets/google/boolq), [repository](https://github.com/google-research-datasets/boolean-questions)) |
| `paws` | `google-research-datasets/paws` (`labeled_final`) | noul | train | Card: other. Original: custom terms, "may be freely used for any purpose" ([card](https://huggingface.co/datasets/google-research-datasets/paws), [LICENSE](https://github.com/google-research-datasets/paws/blob/master/LICENSE)). The sentences come from Wikipedia. |
| `vitaminc` | `tals/vitaminc` | noul | train | CC BY-SA 3.0, ShareAlike ([card](https://huggingface.co/datasets/tals/vitaminc)) |
| `wnli` | `nyu-mll/glue` (`wnli`) | noul | train | Card: other, per task. Original: the Winograd Schema collection, CC BY 4.0 ([GLUE card](https://huggingface.co/datasets/nyu-mll/glue), [collection page](https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html)) |
| `pubmed_qa` | `qiaojin/PubMedQA` (`pqa_labeled`) | noul | train | MIT ([card](https://huggingface.co/datasets/qiaojin/PubMedQA), [LICENSE](https://github.com/pubmedqa/pubmedqa/blob/master/LICENSE)) |
| `yelp_stars` | `Yelp/yelp_review_full` | score | train | Card: other, which points to the Yelp Dataset Terms of Use: non-commercial use only, no redistribution ([card](https://huggingface.co/datasets/Yelp/yelp_review_full), [terms](https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf)). |
| `sst5_sentiment` | `SetFit/sst5` | score | train | Unknown after checking. The card, the [SST page](https://nlp.stanford.edu/sentiment/) and the [`stanfordnlp/sst` card](https://huggingface.co/datasets/stanfordnlp/sst) state no licence ([card](https://huggingface.co/datasets/SetFit/sst5)). |
| `app_reviews` | `sealuzh/app_reviews` | score | train | Unknown after checking. The card says unknown, and the [repository](https://github.com/sealuzh/user_quality) states no licence ([card](https://huggingface.co/datasets/sealuzh/app_reviews)). |
| `formality` | `osyvokon/pavlick-formality-scores` | score | train | Card: CC BY 3.0. The [paper](https://aclanthology.org/Q16-1005/) states no data licence, and part of the text comes from Yahoo Answers ([card](https://huggingface.co/datasets/osyvokon/pavlick-formality-scores)). |
| `ruletaker_d0`, `ruletaker_d1`, `ruletaker_d2` | `tasksource/ruletaker` (depths 0, 1 and 2) | noul | train | Apache-2.0 ([card](https://huggingface.co/datasets/tasksource/ruletaker), [AllenAI repository](https://github.com/allenai/ruletaker)) |
| `emotion` | `dair-ai/emotion` | choice | held out: calibration and evaluation | Card: other, "educational and research purposes only" ([card](https://huggingface.co/datasets/dair-ai/emotion), [repository](https://github.com/dair-ai/emotion_dataset)). |
| `massive_intent` | `mteb/amazon_massive_intent` (`en`) | choice | held out: calibration and evaluation | Card: Apache-2.0. Original: CC BY 4.0 ([card](https://huggingface.co/datasets/mteb/amazon_massive_intent), [NOTICE.md](https://github.com/alexa/massive/blob/main/NOTICE.md)) |
| `sarcasm` | `raquiba/Sarcasm_News_Headline` | noul | held out: calibration and evaluation | Card: none. Original: CC BY 4.0 on the author's Kaggle page. The headlines come from TheOnion and HuffPost ([card](https://huggingface.co/datasets/raquiba/Sarcasm_News_Headline), [Kaggle page](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)). |
| `hate_severity` | `ucberkeley-dlab/measuring-hate-speech` | score | held out: calibration and evaluation | CC BY 4.0 ([card](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)) |
| `ruletaker_d3`, `ruletaker_d5`, `ruletaker_natlang` | `tasksource/ruletaker` (depths 3 and 5, NatLang) | noul | held out of training, not used for calibration | Apache-2.0 ([card](https://huggingface.co/datasets/tasksource/ruletaker)) |

### Registered probes, not in the current recipe

`data/recipes.py` also registers seven held-out probes. The current recipe does not build them.
[research/history.md](../research/history.md) reports their transfer results.

| Recipe | Hugging Face id (config) | Licence |
| --- | --- | --- |
| `newsgroups` | `SetFit/20_newsgroups` | Card: none. Original: CC BY 4.0 ([card](https://huggingface.co/datasets/SetFit/20_newsgroups), [UCI page](https://archive.ics.uci.edu/dataset/113/twenty+newsgroups)) |
| `trec_qc` | `SetFit/TREC-QC` | Unknown after checking. The card, the [CogComp page](https://cogcomp.seas.upenn.edu/Data/QA/QC/) and the [`CogComp/trec` card](https://huggingface.co/datasets/CogComp/trec) state none ([card](https://huggingface.co/datasets/SetFit/TREC-QC)). |
| `arxiv_class` | `ccdv/arxiv-classification` (`no_ref`) | Unknown after checking. The card states none. The text is the full text of arXiv papers, and each paper has the licence that its authors selected ([card](https://huggingface.co/datasets/ccdv/arxiv-classification), [arXiv licences](https://info.arxiv.org/help/license/index.html)). |
| `rte` | `nyu-mll/glue` (`rte`) | Unknown after checking. The GLUE card refers to the original licences, and the RTE releases state none ([GLUE card](https://huggingface.co/datasets/nyu-mll/glue)). |
| `qnli` | `nyu-mll/glue` (`qnli`) | CC BY-SA 4.0, ShareAlike, from SQuAD ([SQuAD page](https://rajpurkar.github.io/SQuAD-explorer/)) |
| `cola` | `nyu-mll/glue` (`cola`) | Unknown after checking. The README of the [CoLA release](https://nyu-mll.github.io/CoLA/) states no licence. |
| `subjectivity` | `SetFit/subj` | Unknown after checking. The card and the [movie-review data page](https://www.cs.cornell.edu/people/pabo/movie-review-data/) state none ([card](https://huggingface.co/datasets/SetFit/subj)). |

## Calibration rows with human label distributions (v25)

`data/calibration.py` converts five sources whose items carry several raters' labels. Each
row keeps one gold `label`, and the raters' distribution goes into a target file for
`TrainConfig.teacher_file`. No configuration trains on these rows yet. The rows are not
committed. [`calibration_v25.json`](calibration_v25.json) records, for each output, the rows
by kind, the sha256, the rows dropped and why, the licence and the command, and the sha256
of every downloaded input.

| Output | Source | Licence | ShareAlike | Rows by kind |
| --- | --- | --- | --- | --- |
| `civil_mid` | Civil Comments (`google/civil_comments`), the 0.2-0.5 band that `toxicity` drops | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | No | 6,000 noul |
| `civil_existing` | The same, for the 5,276 `toxicity` rows already in `train_v5.jsonl` (`relabel`) | CC0 1.0 | No | 5,276 noul |
| `wiki_toxicity`, `wiki_toxicity_score`, `wiki_aggression_score`, `wiki_attack` | [Wikipedia Talk Labels](https://meta.wikimedia.org/wiki/Research:Detox/Data_Release) (figshare 4563973, 4267550, 4054689), train and dev splits | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | No | 16,000 noul, 16,000 score |
| `hs2_dist` | HelpSteer2 `disagreements/disagreements.jsonl.gz` (`nvidia/HelpSteer2`), train-split prompts only | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | No | 6,000 noul, 4,864 score |
| `hs2_existing` | The same, for the 4,866 rows of `adequacy_hs2.jsonl` (`relabel`) | CC BY 4.0 | No | 4,866 noul |
| `unli` | UNLI (`Zhengping/UNLI`), train split | MIT (card); text from SNLI, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Yes | 8,000 noul, 4,000 score |
| `nli_dist` | SNLI 1.0 (train, dev) and MultiNLI 1.0 (dev matched, mismatched) release JSONL, `annotator_labels` | SNLI [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); MultiNLI [OANC, CC BY-SA 3.0, CC BY 3.0](https://cims.nyu.edu/~sbowman/multinli/) | Yes | 10,000 choice, 5,000 noul |

Every candidate row is checked against held-out text before it is kept: the JevBench public
items at commit `1bcc55e`, the held-out short tasks, RuleTaker depths 3 and 5 and NatLang,
HotpotQA, ANLI, the committed `data/synthetic/*_eval.jsonl` files and HelpSteer2 validation.
ChaosNLI (CC BY-NC 4.0) is evaluation only: its ids are read to keep its items out, and
nothing is converted from it.

```bash
CAL=../data-v25/calibration   # any directory outside the repository
python -m strands_decider.data.calibration heldout --raw $CAL/raw --out $CAL/raw/heldout/texts.jsonl
for s in civil wikitalk hs2 unli nli; do
  python -m strands_decider.data.calibration $s --raw $CAL/raw --out $CAL/rows
done
# targets for rows already in the corpus (likewise hs2_existing for data/adequacy_hs2.jsonl,
# --task adequacy_hs2), then one teacher file for v20's train_files order plus the new rows.
# teacher_v20's indices already span train_v5 and multistep_v14; later targets replace earlier ones.
python -m strands_decider.data.calibration relabel --corpus data/train_v5.jsonl --task toxicity \
  --rows $CAL/rows/civil_existing.jsonl --out $CAL/rows/train_v5_toxicity_soft.jsonl
python -m strands_decider.data.calibration merge \
  data/train_v5.jsonl:data/teacher_v20.jsonl,$CAL/rows/train_v5_toxicity_soft.jsonl \
  data/multistep_v14.jsonl data/generated_v16p.jsonl data/generated_v18p.jsonl \
  data/adequacy_hs2.jsonl:$CAL/rows/adequacy_hs2_soft.jsonl data/adequacy_gen.jsonl \
  data/catchall_v20.jsonl data/flips_v20.jsonl \
  $CAL/rows/civil_mid.jsonl:$CAL/rows/civil_mid_soft.jsonl --out data/teacher_v25.jsonl
# then append the same row files to train_files, in the same order
```

`raw/` must hold the downloads listed in `calibration_v25.json` (`inputs_sha256`) and the
JevBench tree unpacked under `raw/jevbench/`.

## Planned sources (not yet in the recipe)

These candidates come from two read-only data reports of 2026-10-03, `CALIBRATION-DATA.md`
(soft labels) and `SCALE-DATA.md` (volume and breadth). They are not in this repository. "SA" = ShareAlike.
Each licence was read on the linked page on 2026-10-03.

| Source | Report | Licence (link) | SA | Attribution | Condition |
| --- | --- | --- | --- | --- | --- |
| Wikipedia Talk Labels: Toxicity, Aggression, Personal Attacks | Calibration | CC0 ([figshare](https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Toxicity/4563973)) | No | Wulczyn, Thain & Dixon, WWW 2017 (Wikimedia Foundation, Jigsaw) | None |
| HelpSteer2 `disagreements` file (`nvidia/HelpSteer2`) | Calibration | CC BY 4.0 ([card](https://huggingface.co/datasets/nvidia/HelpSteer2)) | No | Wang et al. (NVIDIA, 2024), arXiv:2406.08673 | None |
| UNLI (`Zhengping/UNLI`) | Calibration | MIT on the card, uploaded by a co-author. The text is SNLI, CC BY-SA 4.0 ([card](https://huggingface.co/datasets/Zhengping/UNLI), [SNLI](https://nlp.stanford.edu/projects/snli/)) | Yes | Chen et al., ACL 2020. SNLI: Bowman et al., EMNLP 2015 | The project page states no licence |
| SNLI and MultiNLI annotator labels (original release JSONL) | Calibration | SNLI: CC BY-SA 4.0 ([card](https://huggingface.co/datasets/stanfordnlp/snli)). MultiNLI: CC BY 3.0, CC BY-SA 3.0, MIT and OANC ([card](https://huggingface.co/datasets/nyu-mll/multi_nli)) | Yes | Bowman et al., EMNLP 2015. Williams, Nangia & Bowman, NAACL 2018 | None |
| BBQ (`heegyu/bbq`) | Calibration | CC BY 4.0 ([repository](https://github.com/nyu-mll/BBQ), [card](https://huggingface.co/datasets/heegyu/bbq)) | No | Parrish et al., Findings of ACL 2022 | None |
| DICES | Calibration | CC BY 4.0 ([repository](https://github.com/google-research-datasets/dices-dataset)) | No | Aroyo et al., NeurIPS 2023 (Google) | None |
| AmbiEnt (`tasksource/ambient`) | Calibration | CC BY 4.0 ([repository](https://github.com/alisawuffles/ambient)) | No | Liu et al., EMNLP 2023 | None |
| δ-NLI, Defeasible NLI (`tasksource/defeasible-nli`) | Calibration | MIT ([repository](https://github.com/rudinger/defeasible-nli)). The card says Apache-2.0. The SNLI subset is CC BY-SA 4.0 | Yes, SNLI subset | Rudinger et al., Findings of EMNLP 2020 | None |
| HelpSteer3 `preference` (`nvidia/HelpSteer3`) | Calibration, scale | CC BY 4.0 ([card](https://huggingface.co/datasets/nvidia/HelpSteer3)) | No | NVIDIA, 2025 | None |
| FOLIO (`tasksource/folio`) | Calibration | CC BY-SA 4.0 ([repository](https://github.com/Yale-LILY/FOLIO)) | Yes | Han et al., 2022, arXiv:2209.00840 | None |
| SQuAD 2.0 (`rajpurkar/squad_v2`) | Calibration, scale | CC BY-SA 4.0 ([card](https://huggingface.co/datasets/rajpurkar/squad_v2)) | Yes | Rajpurkar, Jia & Liang, ACL 2018 | None |
| tasksource-jev-typed-decisions, `full` config | Scale | Per source. The card says other, and gives a `license_use` field for each row ([card](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions)) | Yes, for about 150k rows | tasksource (Sileo), and the attribution of each source | Drop the "unspecified", NC and GPL or AGPL sources. The `license_use` field is best effort |
| CFPB consumer complaints (`BEE-spoke-data/consumer-finance-complaints`) | Scale | CC0 ([card](https://huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints)) | No | Consumer Financial Protection Bureau | None |
| hh-rlhf (`Anthropic/hh-rlhf`) | Scale | MIT ([card](https://huggingface.co/datasets/Anthropic/hh-rlhf)) | No | Bai et al. (Anthropic, 2022) | The responses are Anthropic model outputs, released by Anthropic under MIT |
| HelpSteer v1 (`nvidia/HelpSteer`) | Scale | CC BY 4.0 ([card](https://huggingface.co/datasets/nvidia/HelpSteer)) | No | Wang et al. (NVIDIA, 2023) | None |
| OASST2 (`OpenAssistant/oasst2`) | Scale | Apache-2.0 ([card](https://huggingface.co/datasets/OpenAssistant/oasst2)) | No | Köpf et al., 2023 (OpenAssistant) | None |
| DocNLI (`tasksource/doc-nli`) | Scale | BSD-3-Clause ([repository](https://github.com/salesforce/DocNLI), [card](https://huggingface.co/datasets/tasksource/doc-nli)) | No | Yin, Radev & Xiong, Findings of ACL 2021 (Salesforce) | Part of the text is CNN/DailyMail news |
| CUAD (`theatticusproject/cuad-qa`) | Scale | CC BY 4.0 ([card](https://huggingface.co/datasets/theatticusproject/cuad-qa)) | No | Hendrycks et al., NeurIPS 2021 (The Atticus Project) | The card makes no representations about the rights in the contracts |
| MAUD (`theatticusproject/maud`) | Scale | CC BY 4.0 ([card](https://huggingface.co/datasets/theatticusproject/maud)) | No | Wang et al., EMNLP 2023 (The Atticus Project) | None |
| CaseHOLD and LEDGAR, from LexGLUE (`coastalcph/lex_glue`) | Scale | CC BY 4.0 ([card](https://huggingface.co/datasets/coastalcph/lex_glue)) | No | Chalkidis et al., ACL 2022. Zheng et al., ICAIL 2021. Tuggener et al., LREC 2020 | None |
| 2WikiMultihopQA | Scale | Apache-2.0 ([repository](https://github.com/Alab-NII/2wikimultihop)). The Wikipedia text is CC BY-SA | Yes | Ho et al., COLING 2020 | None |
| glaive-function-calling-v2 (`glaiveai/glaive-function-calling-v2`) | Scale | Apache-2.0 ([card](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)) | No | Glaive AI | The card does not name the generator model. |
| hermes-function-calling-v1 (`NousResearch/hermes-function-calling-v1`) | Scale | Apache-2.0 ([card](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)) | No | Nous Research | The card does not name the generator models. |
| FEVER-NLI (`pietrolesci/nli_fever`) | Scale | Wikipedia terms, or CC BY-SA 3.0 ([`fever/fever` card](https://huggingface.co/datasets/fever/fever)) | Yes | Thorne et al., NAACL 2018 | The `fever/fever` card also lists GPL-3.0. |
| SNLI, more rows (`stanfordnlp/snli`) | Scale | CC BY-SA 4.0 ([card](https://huggingface.co/datasets/stanfordnlp/snli)) | Yes | Bowman et al., EMNLP 2015 | None |
| MedMCQA (`openlifescienceai/medmcqa`) | Scale | Apache-2.0 ([card](https://huggingface.co/datasets/openlifescienceai/medmcqa)) | No | Pal et al., CHIL 2022 | None |
| AQuA-RAT (`deepmind/aqua_rat`) | Scale | Apache-2.0 ([card](https://huggingface.co/datasets/deepmind/aqua_rat)) | No | Ling et al., ACL 2017 (DeepMind) | None |
| ALFWorld expert trajectories | Scale | MIT ([repository](https://github.com/alfworld/alfworld)) | No | Shridhar et al., ICLR 2021 | None |
| Mind2Web, train split (`osunlp/Mind2Web`) | Scale | CC BY 4.0 ([card](https://huggingface.co/datasets/osunlp/Mind2Web)) | No | Deng et al., NeurIPS 2023 | None |
| tasksource procedural-typed-decisions | Scale | Apache-2.0 ([card](https://huggingface.co/datasets/tasksource/procedural-typed-decisions)) | No | tasksource (Sileo) | None |
| ShARC (`tasksource/sharc`) | Scale | CC BY-SA 3.0 ([data README](README.md)). The tasksource card states none | Yes | Saeidi et al., EMNLP 2018 | None |
| Natural Questions (`google-research-datasets/natural_questions`), reserve | Scale | CC BY-SA 3.0 ([card](https://huggingface.co/datasets/google-research-datasets/natural_questions)) | Yes | Kwiatkowski et al., TACL 2019 | None |
| BIGPATENT (`NortheasternUniversity/big_patent`), reserve | Scale | CC BY 4.0 ([card](https://huggingface.co/datasets/NortheasternUniversity/big_patent)) | No | Sharma et al., ACL 2019 | None |

The two reports also propose more rows or soft labels from current sources: `civil_comments`
(CC0), `dbpedia_14` (CC BY-SA 3.0), MultiNLI and VitaminC. These rows keep the licence of the
current source.
