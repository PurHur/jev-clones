# Routing thresholds: per-question predictions

This folder reproduces the [Routing thresholds](../README.md#routing-thresholds) table of H2O-Lightning-4B and lets you
compute it at any other threshold, temperature or escalation budget.

## What is here

| file | what |
|---|---|
| `predictions.jsonl` | one row per question (3,474): `dataset`, `source_id` (the `id` of the row in the original dataset), `converted_row`, `question_type` (`choice`, `noul` = yes/no, `score`), `options`, `gold` / `gold_indices`, `raw_logprobs_t1` (the label log-probabilities at temperature 1, renormalised over the options, in `options` order), `served_temperature` and `served_probabilities` (the shim's answer at the shipped temperature, yes/no floor off), `top_answer`, `top_probability` and `correct` (from `raw_logprobs_t1` at the shipped temperature, the table's method) |
| `questions/plumb_s2k.jsonl`, `questions/eikos.jsonl` | the questions as the model saw them (document, instructions, criteria, options, gold), with `dataset`, `license` and `source_id` on every row |
| `routing_thresholds.py` | the table from `predictions.jsonl`; any thresholds, temperatures or budgets (standard library only) |
| `ATTRIBUTION.md`, `questions/LICENSE-plumb-decisions-Apache-2.0.txt` | sources and licences of the questions |
| `SHA256SUMS` | the sha256 of every file here |

```bash
python3 routing_thresholds.py                                  # the card's table (thresholds 0.8 and 0.9)
python3 routing_thresholds.py --thresholds 0.5 0.6 0.7 0.8 0.9 0.95
python3 routing_thresholds.py --escalate 0.25                  # per type: the highest threshold that escalates at most 25 %
python3 routing_thresholds.py --max-wrong 0.01                 # per type: the lowest threshold with at most 1 % confidently wrong
python3 routing_thresholds.py --temps choice=1,noul=1,score=1  # other temperatures
python3 routing_thresholds.py --dataset eikos                  # one source dataset
```

A question is **escalated** when its top probability is below the threshold, and **confidently wrong** when its top
probability is at or above the threshold but its top answer is not in the gold set; both are shares of all questions of
that type. Probabilities are `softmax(raw_logprobs_t1 / T)` with the per-type temperature T.

## How the predictions were made

- Model: `h2oai/h2o-lightning-4b` at tag `v1.2.3` (commit `97becfd219bcd85b0d6e800cc9544c91627656bc`), the merged weights (`model.safetensors` sha256
  `2d51561f221e1549d91bd1c38b3512a1435dd32950d454a9f040f1d4d11497b6`), served as in [Run it](../README.md#run-it) on one RTX PRO 4500 Blackwell (32 GB), vLLM 0.30.0:

  ```bash
  VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve ./h2o-lightning-4b --served-model-name h2oai/h2o-lightning-4b --host 127.0.0.1 --port 8000 \
    --max-model-len 40960 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image": 4, "video": 0}' --mm-processor-kwargs '{"max_pixels": 1605632}'
  ```

- Two copies of the shipped shim against that server, both in [routing mode](../README.md#optional-routing-mode)
  (`SHIM_NOUL_FLOOR=0`, the yes/no floor off):

  ```bash
  # served probabilities: shipped temperatures (choice 0.75, yes/no 0.8, score 0.65)
  SHIM_NOUL_FLOOR=0 python3 h2o_lightning_shim.py --config serve_config.json --vllm http://127.0.0.1:8000 --port 8741
  # raw_logprobs_t1: the same shim at temperature 1
  SHIM_TEMPERATURE=1 SHIM_NOUL_FLOOR=0 python3 h2o_lightning_shim.py --config serve_config.json --vllm http://127.0.0.1:8000 --port 8742
  ```

  Each question was sent as one `POST /v1/systemone` request (`state` = the document; one question with its
  instructions and criteria), 32 at a time.
- The two columns come from two separate passes over the same server, 32 requests at a time. vLLM batches concurrent
  requests, so repeated reads differ slightly: between the two passes the median absolute probability difference is
  0.0005, the 99th percentile 0.016, and 9 of 3,474 top answers differ.
- A re-read therefore moves the small score cells (253 questions) by about one question. Example: at threshold 0.9
  about 100 score questions are kept, so one question is about 1 point of "accuracy on the rest". The card shows 98.0 %
  there; this re-read gives 99.0 %, because one wrong score question sits right at the threshold (0.896 here, just
  above 0.9 in the card's read). Every other cell agrees with the card within 0.5 points.

## The card's table recomputed from these files

| question type (n) | threshold | escalated | accuracy on the rest | confidently wrong |
|---|---|---|---|---|
| choice (1,854) | 0.8 | 34.1 % | 95.4 % | 3.0 % |
| choice | 0.9 | 47.5 % | 98.2 % | 1.0 % |
| yes/no (1,367) | 0.8 | 41.0 % | 96.0 % | 2.3 % |
| yes/no | 0.9 | 59.5 % | 99.1 % | 0.4 % |
| score (253) | 0.8 | 46.6 % | 96.3 % | 2.0 % |
| score | 0.9 | 60.1 % | 99.0 % | 0.4 % |
| all (3,474) | 0.8 | 37.7 % | 95.7 % | 2.7 % |
| all | 0.9 | 53.2 % | 98.5 % | 0.7 % |

Without a threshold: choice 82.7 %, yes/no 84.7 %, score 80.6 %.

## Sources

2,000 questions from Plumb ([crh225/plumb-decisions](https://huggingface.co/datasets/crh225/plumb-decisions),
Apache-2.0, revision `718a9f006f3beaecfa7f66a819553f99ed7b94f6`: all 131 test-split questions and 1,869 train-split
questions) and 1,474 from Eikos
([caiovicentino1/eikos-decisions](https://huggingface.co/datasets/caiovicentino1/eikos-decisions), CC-BY-4.0, revision
`cebb4a26687cb14742fd07c0ebe17387ee51d369`: the English rows of the core held-out and validation splits). Neither dataset was used to train the model. See
ATTRIBUTION.md for the changes made in converting them.
