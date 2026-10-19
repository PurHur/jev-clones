---
license: apache-2.0
language:
  - en
  - es
  - de
base_model: convaiinnovations/laya-multilingual
datasets:
  - tyqiangz/multilingual-sentiments
pipeline_tag: text-classification
library_name: laya
inference: false
tags:
  - decision-model
  - sentiment-analysis
  - multilingual
  - coreml
  - on-device
  - calibrated
  - hybrid-inference
  - apple-silicon
  - laya
metrics:
  - accuracy
  - ece
model-index:
  - name: laya-sentiment-multilingual
    results:
      - task:
          type: text-classification
          name: Binary Sentiment Classification
        dataset:
          type: tyqiangz/multilingual-sentiments
          name: Multilingual Sentiments — full non-neutral test set (580 per language)
          split: test
        metrics:
          - type: accuracy
            value: 0.860
            name: Accuracy (n=1740)
            verified: false
          - type: ece
            value: 0.0415
            name: Expected Calibration Error, cross-fitted (n=1740)
            verified: false
      - task:
          type: text-classification
          name: Binary Sentiment Classification — published 300-example holdout
        dataset:
          type: tyqiangz/multilingual-sentiments
          name: Multilingual Sentiments (holdout, neutral removed)
          split: test
        metrics:
          - type: accuracy
            value: 0.880
            name: Accuracy (n=300)
            verified: false
      - task:
          type: text-classification
          name: Binary Sentiment Classification (English)
        dataset:
          type: tyqiangz/multilingual-sentiments
          name: Multilingual Sentiments — EN test (non-neutral)
          split: test
        metrics:
          - type: accuracy
            value: 0.902
            name: Accuracy (n=580)
            verified: false
      - task:
          type: text-classification
          name: Binary Sentiment Classification (Spanish)
        dataset:
          type: tyqiangz/multilingual-sentiments
          name: Multilingual Sentiments — ES test (non-neutral)
          split: test
        metrics:
          - type: accuracy
            value: 0.834
            name: Accuracy (n=580)
            verified: false
      - task:
          type: text-classification
          name: Binary Sentiment Classification (German)
        dataset:
          type: tyqiangz/multilingual-sentiments
          name: Multilingual Sentiments — DE test (non-neutral)
          split: test
        metrics:
          - type: accuracy
            value: 0.843
            name: Accuracy (n=580)
            verified: false
---

# Laya Sentiment Multilingual (EN/ES/DE)

A fine-tuned Laya decision model for binary sentiment classification in English, Spanish, and German. It raises accuracy from 55.7% (zero-shot base model) to 86.0% on the full non-neutral test set, with temperature scaling that helps less than the small holdout suggested.

## Model Details

- Base model: `convaiinnovations/laya-multilingual` (322M params, mmBERT-base encoder)
- Task: binary sentiment via the `noul` primitive — P(text expresses positive sentiment). The neutral class was removed from the training data.
- Languages: EN, ES, DE (Latin script only)
- Weights: `model.safetensors` — 643,835,524 bytes, sha256 `1701a2f936d82eaa7a5e3f4da8c236d6f0fe47d03fad9a49f9e5f5a2483d08c9`
- Deployment: CoreML FP16 on Apple Silicon (macOS 15+ / iOS 18+) — see `coreml/`
- License: Apache 2.0

## Training Details

- Dataset: `tyqiangz/multilingual-sentiments` (Apache 2.0), revision `a3080a58`, neutral removed
- Split: 900 train / 300 holdout, balanced 150/150 per language (seed 42). Manifest, hashes and build script in `split/`
- Method: RLCD fine-tuning, LR 3e-5, best epoch 1 of a 15-epoch budget (early stopping, patience 4)
- Calibration: temperature T=0.9 (`calibration.json`) — see the caveats below

## Evaluation Results

Two evaluation sets, both from the `test` split (disjoint from training; verified 0 exact text overlap, 0 duplicate texts inside train):

- **Holdout 300** (100/language): the sample used during development and for checkpoint selection. This is the number comparable with earlier releases.
- **Full test 1740** (580/language, all non-neutral rows): the honest headline. It contains the 300.

Confidence intervals are 95% Wilson.

### Accuracy

| Set | n | Accuracy | 95% CI |
|---|---|---|---|
| Zero-shot base (holdout 300) | 300 | 55.7% | [50.0–61.2%] |
| Fine-tuned (holdout 300) | 300 | 88.0% | [83.8–91.2%] |
| **Fine-tuned (full test)** | **1740** | **86.0%** | **[84.3–87.5%]** |

The 88.0% on the 300-example holdout is about **2 points optimistic**: on 1740 examples the same model scores 86.0%. Both intervals overlap, but the smaller sample was the more favourable one. Differences between 88.0% and 86.0% are not meaningful; the 1740-example estimate is simply tighter.

### Per language (full test, n=580 each)

| Language | Accuracy | 95% CI |
|---|---|---|
| English (en) | 90.2% | [87.5–92.3%] |
| German (de) | 84.3% | [81.1–87.0%] |
| Spanish (es) | 83.4% | [80.2–86.3%] |

With 580 examples per language the ordering is now meaningful: English is ahead of German and Spanish, which are within a point of each other.

### Per class (full test)

| Class | Accuracy |
|---|---|
| Positive | 86.8% (755/870) |
| Negative | 85.2% (741/870) |

On the 300-example holdout this looked perfectly symmetric (88.0%/88.0%). At n=1740 a small negative-class advantage appears; the symmetry was a small-sample artefact.

### Calibration

T=0.9 was fitted by minimising NLL on the 300 holdout. Two checks were run on the full test set.

**1. Cross-fitted temperature** (5 folds: each case scored with a T fitted without it):

| Set | ECE @T=1.0 | ECE @T=0.9 | ECE cross-fitted (T≈1.0) |
|---|---|---|---|
| Holdout 300 | 0.0534 | 0.0254 | 0.0271 |
| **Full test 1740** | **0.0416** | 0.0338 | **0.0415** |

**2. Transferred temperature:** T fitted on the 300 holdout (= 0.9), applied to the 1,440 test examples it never saw:

| | ECE | NLL | Brier |
|---|---|---|---|
| raw (T=1.0) | 0.0471 | 0.3336 | 0.1022 |
| with T=0.9 from the 300 | 0.0373 | 0.3354 | 0.1032 |

What this means, plainly:

- On the 300-example holdout, temperature scaling looks like a large win (ECE 0.0534 → 0.0254, a 52% reduction).
- On 1,440 unseen examples the same correction buys much less (0.0471 → 0.0373, 21%) and makes both NLL and Brier slightly **worse**.
- A temperature fitted on the full test set is ≈1.0, i.e. *no correction at all*.

**The published ECE improvement is mostly an artefact of the small holdout.** Use `T=0.9` as a mild, harmless correction, do not treat 0.0254 as the model's calibration, and do not build hard confidence thresholds on the difference.

### Data-scaling experiment

The source dataset has 1,839 train + 324 validation rows per language for EN/ES/DE (the often-quoted 591k figure is the dataset total across **all** languages, not the EN/ES/DE pool). Training data was therefore scaled from 900 to **4,326** examples (1,442/language, a record-level superset of the 900) and the same recipe was re-run.

| Set | v1 (900) | v2 (4,326) | Difference | Paired McNemar |
|---|---|---|---|---|
| Holdout 300 | 88.0% | 88.0% | +0.0 | p = 1.0 |
| Clean 1,440 (never used for selection) | 85.6% | 86.3% | +0.8 | p = 0.42 |
| Full test 1,740 | 86.0% | 86.6% | +0.6 | p = 0.46 |

**4.8× the training data produced no statistically detectable improvement.** The limiting factor is the training recipe, not data volume: with 4,326 examples one epoch means 541 optimizer steps versus 113 for the 900-example run, so the best v2 checkpoint saw ~9.6× more gradient steps at a constant LR of 3e-5 and memorised the training set (97–98% train accuracy versus 87% validation). The extra data also bought sharper but *worse* probabilities: at the same T=0.9, v2's ECE is 0.089 against v1's 0.034. The released weights remain **v1**; the v2 run is published separately as [`Ramg77/laya-sentiment-multilingual-v2`](https://huggingface.co/Ramg77/laya-sentiment-multilingual-v2) and is explicitly not recommended over v1. Full logs and scripts: `experiments/data-scaling/`.

### External benchmark and label audit

The evaluation data turns out to be the **UMSAB benchmark** (identical in content to
`cardiffnlp/tweet_sentiment_multilingual`, verified at 100% text overlap), which makes the numbers
comparable with published work — and makes a head-to-head possible, because
**`cardiffnlp/twitter-xlm-roberta-base-sentiment` is fine-tuned on that same benchmark**.

| Model | Accuracy (n=1,740) | 95% CI | ECE |
|---|---|---|---|
| Cardiff XLM-R (270M, 3 classes, reduced to binary) | **89.7%** | [88.2, 91.1] | 0.0097 |
| This model | 86.0% | [84.3, 87.5] | 0.0416 |

Paired McNemar p = 1.3e-05, and the baseline wins in every language and at every coverage level.
Contamination was ruled out first: its three-class macro-F1 on the full test split (73.0 / 75.1 /
69.3 for en/de/es) sits inside the range published by the XLM-T authors.

**Isolating data from recipe.** Fine-tuning that same baseline on **our own data** settles the
cause. With our exact 900 examples it reaches **88.8%** (+2.8 points over this model, p = 0.0013)
and with our full 4,326-example pool **89.1%** — statistically indistinguishable from the same
baseline trained on ~17,000 examples across eight languages, and in 57 optimizer steps where ours
uses 113. The 900 examples were never the problem.

**If you need a multilingual sentiment classifier, use the baseline.** This model is a documented
experiment, not a competitive release. Its value is the audit around it: the ~86% plateau is a
property of its training procedure, not of the data.

Two more findings from the same round:

- **The labels are not the ceiling.** Label noise accounts for at most ~1% of the test set (only
  10 cases are wrong with confidence >= 0.95, and 8 of those are also missed by the independently
  trained 4.8x model). The residual error is sarcasm, mixed sentiment and complaints phrased with
  positive words; the contrast marker (*but/aber/pero*) is the only objective predictor we found.
- **Five seeds per configuration** show the recipe averaging **84.98%** (range 84.0–86.1), so the
  86.0% headline sits at the top of its own seed distribution. The one recipe change that survives
  testing is an LR schedule (warmup + cosine): **+1.0 point** [CI +0.33, +1.72].

A follow-up experiment equalised the optimization budget instead of the epoch count: with 113 steps, 4,326 examples and 900 examples tie (87.0% each on the holdout) and the exact v1 recipe on 4.8× the data lands 1.4 points *below* v1. Freezing the encoder is clearly worse — head-only training needs 17.7× the steps to stay 2 points behind. See `experiments/recipe-v3/`.

## Reproducibility

```bash
git clone https://huggingface.co/Ramg77/laya-sentiment-multilingual && cd laya-sentiment-multilingual
pip install laya torch transformers safetensors numpy

# published holdout (300)
python eval_sentiment.py --model . --holdout split/holdout_laya_sentiment.jsonl

# full test set (1740)
python eval_sentiment.py --model . --holdout split/test_extended.jsonl
```

Included in this repo:

- `eval_sentiment.py` — accuracy with Wilson CIs, ECE (raw / shipped / cross-fitted) with bootstrap CIs, per-language and per-class breakdowns
- `split/` — `train_laya_sentiment.jsonl` (900), `holdout_laya_sentiment.jsonl` (300), `test_extended.jsonl` (1,740), `build_sentiment.py` (seed 42), `manifest.json` (sha256 of every file, pinned dataset revision, leakage checks)
- `calibration.json` — the fitted T and the before/after metrics
- `coreml/` — CoreML conversion config, including the source weight hash
- `experiments/data-scaling/` — the 4,326-example split, training script, per-epoch log and paired comparison
- `experiments/recipe-v3/` — equal-steps, frozen-encoder and LR-schedule arms with their logs and the paired comparison
- `experiments/multiseed/` — the four configurations × five seeds, with per-run logits
- `experiments/external-baseline/` — the Cardiff comparison, including the contamination check
- `experiments/data-vs-recipe/` — the baseline fine-tuned on our own 900 and 4,326 examples, which isolates the plateau as a recipe effect
- `experiments/error-audit/` — all 244 errors with text, label, prediction and confidence
- `paper/code/` — the analysis and figure scripts used to produce the paper
- `reports/` — full written report (Markdown and PDF) covering every measurement in this card
- `paper/` — preprint (Markdown and PDF) with the two controlled experiments, the figures and an appendix listing every correction made to the earlier documentation

Reproduced with laya 0.3.11, torch 2.14.0, transformers 5.17.0, Python 3.12.7 (Apple Silicon, MPS).

## Deployment

- A CoreML FP16 build is published as a separate repository: [`Ramg77/laya-sentiment-multilingual-coreml`](https://huggingface.co/Ramg77/laya-sentiment-multilingual-coreml) (647 MiB; Neural Engine and GPU on macOS 15+ / iOS 18+). It is derived from the exact weights published here: `coreml/coreml_config.json` records `source_weights_sha256: 1701a2f9…08c9`, identical to the `model.safetensors` above. It is also reproducible with `laya-coreml convert <model_dir> <out> --max-length 512 --precision float16 --attention sdpa`.
- Local hybrid harness: FastAPI gateway on 127.0.0.1:8090, exposed to an agent as a native `laya_sentiment` tool. When calibrated confidence < 0.75 the harness delegates the case to a cloud model.
- On the full test set, 12.1% of cases fall below that 0.75 threshold (210/1740); on the holdout it is 13.0% (39/300). Note that this gate is only as good as the calibration: the more-data v2 run is much more confident — it would delegate only 1.8% (32/1740) — while being *worse* calibrated in the raw sense (ECE 0.089 versus 0.034 at the same temperature). An overconfident model silently defeats a confidence threshold.
- Latency: 16 ms steady-state and 375 ms warmup, measured by the author on an M3 Air (ANE). Not independently verified.

## Usage

```python
import laya

agent = laya.load("Ramg77/laya-sentiment-multilingual")
result = agent.predict(
    "I love this product, it changed my life!",
    {
        "sentiment": {
            "type": "noul",
            "instructions": "Does this text express positive sentiment?"
        }
    }
)
print(result["answers"]["sentiment"])
```

For temperature scaling, use the included `calibration.json` (T=0.9) with the caveats above.

Set `USE_TF=0` when TensorFlow is installed in the same environment: the abseil runtime can deadlock Laya's model initialisation.

## Limitations

- **Binary only**: positive vs. negative. No neutral or mixed sentiment in a single pass.
- **Three languages, Latin script only**: only EN/ES/DE were trained. Other languages are unvalidated. This model does **not** inherit the multilingual coverage of the Laya base checkpoint: the base model's documented failure modes on non-Latin scripts (e.g. the calibration collapse reported on Khmer and Hebrew) were never re-tested after fine-tuning, and nothing here suggests they were fixed.
- **Small data pool**: for EN/ES/DE the dataset only contains 1,839 train + 324 validation rows per language. The 900-example training set used ~21% of the available non-neutral rows (300 of 1,442 per language); the true ceiling for this dataset and these languages is 1,442 examples per language, not hundreds of thousands. The 591k figure often quoted for this dataset is its total across all languages.
- **Not competitive with the standard baseline**: Cardiff XLM-R, fine-tuned on the same benchmark, scores 89.7% against 86.0% (p = 1.3e-05) with fewer parameters and better calibration. The comparison does not isolate the cause (the baseline saw ~16x more training data).
- **More data did not help within this recipe**: with the optimization budget equalised, 4,326 and 900 examples tie (five seeds: -0.3 points, 95% CI [-1.3, +0.7]) and freezing the encoder is clearly worse. Another model trained on the same data reaches ~90%, so this is a statement about the recipe, not about the data.
- **Short text**: training data is short social-media messages (mean ~20 tokens). Long-form text is not evaluated.
- **Domain shift**: trained on social-media sentiment; performance on formal business text is unknown.
- **Calibration is weakly supported**: the T=0.9 correction does not reproduce its holdout magnitude out of sample, and a temperature fitted on the full test set is ≈1.0.
- **Sentiment != intent**: this detects expressed sentiment, not customer intent. Do not use it for routing or escalation decisions.
- **Small training set**: 900 examples is a small-scale result, not a competitive benchmark.

## Ethical Considerations

- Sentiment classifiers can encode demographic bias present in the training data. The multilingual-sentiments dataset was not audited for bias across demographic groups.
- Do not use for high-stakes decisions (hiring, credit, medical triage).
- Intended for content analysis and user-facing features with human oversight.

## License

**Apache 2.0**, and the reason is not a preference: **this model is a derivative work.**

- The weights are a fine-tune of `convaiinnovations/laya-multilingual`, which is **Apache 2.0**. A
  derivative of an Apache-2.0 model keeps that license, so Apache 2.0 is the only thing this card
  could have said without contradicting its own base.
- The initial intent was **GPL-2.0**, to follow the Laya project's path. Verifying it changed the
  answer: Laya —its GitHub repository, the `laya` and `laya-coreml` packages, and the base model—
  declares Apache 2.0, not GPL. The FSF considers **Apache 2.0 incompatible with GPL-2.0** (the
  patent-termination clause), though compatible with GPL-3.0, so releasing these weights under
  GPL-2.0 while they derive from an Apache-2.0 model would have been a grey area at best.

**The training data carries its own terms.** `tyqiangz/multilingual-sentiments` declares Apache 2.0
and is redistributed here as it came. Its content is identical to the
`cardiffnlp/tweet_sentiment_multilingual` benchmark, which **declares no license**, and the ultimate
provenance is three annotation projects with their own conditions:

| Language | Origin |
|---|---|
| English | SemEval-2017 Task 4 |
| German | SB-10K |
| Spanish | InterTASS 2017 (6 classes collapsed to 3) |

The direct source is permissive and attributed; the chain behind it is not fully documented upstream.
If this is going to be used commercially, review that chain first.

**What this section promises, and what it does not.** It covers the weights published here and the
code in the [companion harness](https://github.com/Ra77Data/laya-hybrid-harness), which is Apache 2.0
for the same reason. It grants nothing over the base model that Laya's own license does not grant, and
it says nothing about the terms of the annotation projects listed above.

## Citation

```bibtex
@misc{laya-sentiment-multilingual,
  author = {Ramg77},
  title = {Laya Sentiment Multilingual (EN/ES/DE)},
  year = {2026},
  publisher = {Hugging Face}
}

@misc{laya,
  author = {Nandha Kishor M},
  title = {Laya: Non-autoregressive System 1 decision engine},
  year = {2026},
  howpublished = {https://github.com/NandhaKishorM/laya}
}

@misc{multilingual-sentiments,
  author = {tyqiangz},
  title = {Multilingual Sentiments},
  year = {2023},
  howpublished = {https://huggingface.co/datasets/tyqiangz/multilingual-sentiments}
}
```

## Acknowledgments

- Laya by Nandha Kishor M and Convai Innovations — Apache 2.0.
- multilingual-sentiments by tyqiangz — Apache 2.0.
- Fine-tuning pipeline, calibration, and deployment built on the laya-coreml ecosystem.
