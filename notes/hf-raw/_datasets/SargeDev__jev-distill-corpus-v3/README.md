---
license: apache-2.0
language:
  - en
task_categories:
  - text-classification
tags:
  - system-one
  - typed-decisions
  - distillation
  - calibrated-probabilities
  - noul
  - choice
  - score
size_categories:
  - 100K<n<1M
pretty_name: Jev Distill Corpus v3
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: validation
        path: validation.jsonl
      - split: calibration
        path: calibration.jsonl
      - split: test
        path: test.jsonl
      - split: ood
        path: ood.jsonl
      - split: test_set_30k
        path: test_set_30k.jsonl
---

# Jev Distill Corpus v3

A 740,957-row calibrated typed-decision corpus in the TypeSafe System One schema
(`noul` / `choice` / `score`). Built for training small local "System One" judges
(fine-tuned BERT-class encoders or small Qwens) that read a state and a typed
question and return a calibrated probability distribution in one forward pass.

## Streams

| stream | rows | origin |
|---|---:|---|
| `yuri_v3` | 498,010 | Synthetic operational scenarios across 53 domains (business, technical, agent, knowledge, spatial, structured, medical, biology, genomics, physics, chemistry, theology, science), 3 primitives per domain. Labels distilled from **Jev 1.13** (TypeSafe) via OpenRouter. |
| `yuri_v1` | 148,154 | Memory-relevance `noul` pairs distilled from a 32B teacher over 18 open-domain QA datasets (HotpotQA, PubMedQA, MLQA, QASports, NarrativeQA, LSAT, TyDi, etc.). Query embedded in the question text. |
| `openjev_v2` | 94,801 | Rows from [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) `release-v2-redistributable` (CC0), reschema'd to the unified format. Its `ood` rows are preserved in the `ood` split. |

## Splits

| split | rows | notes |
|---|---:|---|
| `train` | 655,806 | training pool |
| `validation` | 14,111 | |
| `calibration` | 13,766 | reserved for temperature fitting / calibration |
| `test` | 14,261 | original random test slice |
| `ood` | 13,058 | Open-Jev OOD rows |
| `test_set_30k` | 29,955 | **held-out evaluation set** — stratified across every domain family and all 3 primitives, leakage-checked against the earlier 10k benchmark |

`test_set_30k` was carved out of train/validation/calibration after splitting, so
those splits no longer contain it. If you train on `train`, evaluate on `test_set_30k`.

## Row schema

```json
{
  "id": "v3_ab12cd34_c",
  "kind": "noul | choice | score",
  "options": ["false", "true"],
  "target": [0.11, 0.89],
  "state": "scenario text",
  "question": "typed question about the state",
  "domain": "incident_triage",
  "family": "technical",
  "source": "yuri_v3 | yuri_v1 | openjev_v2"
}
```

- `noul`: 2-way distribution `[P(false), P(true)]`
- `choice`: distribution aligned with `options`
- `score`: distribution over ordered levels `0..5` (labels distilled from Jev's
  expected-score distribution)

All targets are normalized probability distributions (sum to 1).

## Label quality (yuri_v3 stream)

- Teacher: Jev 1.13 via the OpenRouter Decisions API, full distributions per question
- Degenerate labels (max-prob ≥ 0.999 with confidence ≥ 0.95): filtered (1.5%)
- noul calibration: mean 0.45, stdev 0.28 — the teacher discriminates rather than rubber-stamps
- Dedup: exact-duplicate rate ~0% (sha256 of normalized state+question), cross-stream
  collisions removed at merge

## Provenance & privacy

- All rows are **synthetic or derived from public datasets**; emails in scraped
  passages were redacted. Credentials/secret patterns were scanned for (zero hits).
- No rows originate from live agent sessions or personal data.

## Attribution

This dataset was created and maintained by SargeDev and contributors for open research into learned System One / calibrated decision-making. If you use this dataset for training, fine-tuning, evaluation, or derivative models, please credit SargeDev/jev-distill-corpus-v3 and link back to this dataset.

## Intended use

Train a small judge (e.g. fine-tuned encoder or 0.5B–4B LM) to mirror System One
typed decisions locally. Example: `noul` head = P(true) from the target's second
element; `choice`/`score` train against the full distribution (KL/RPS losses).

## License

Apache-2.0. The `openjev_v2` stream is additionally CC0 at its source. And if you use this dataset, credit it in your final product. 