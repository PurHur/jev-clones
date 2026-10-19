---
language: en
license: other
license_name: lfm1.0
license_link: LICENSE
library_name: peft
base_model: LiquidAI/LFM2.5-350M
base_model_relation: adapter
pipeline_tag: text-classification
tags:
  - decision-model
  - lora
  - multiple-choice
  - typesafe
  - lfm2
datasets:
  - legacy-datasets/banking77
  - google/boolq
  - fancyzhx/ag_news
  - nyu-mll/multi_nli
  - SetFit/sst5
  - Yelp/yelp_review_full
  - CogComp/trec
  - fancyzhx/dbpedia_14
  - SetFit/amazon_reviews_multi_en
  - stanfordnlp/imdb
metrics:
  - accuracy
---

# lev-350m

lev's a clone of [kev](https://github.com/jaredpalmer/kev), Jared Palmer's Jev-style decision model, with the Qwen backbone replaced by LiquidAI's [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M).  Mostly a prototype and a fun weekend experiment.

Just like Jev, you give it one document / the state and a set of typed questions, and it returns a probability for every option of every question from a single forward pass.  This is not an LLM so it doesn't do any text generation.

![How one request goes through lev](lev-architecture.png)

- Code: [github.com/franckverrot/lev](https://github.com/franckverrot/lev)
- Demo: [huggingface.co/spaces/franckverrot/lev](https://huggingface.co/spaces/franckverrot/lev)

## Run it

This repo holds a LoRA adapter and a small pointer head (6.5M parameters) for LFM2.5-350M.  The `lev` package loads them:

```bash
git clone https://github.com/franckverrot/lev && cd lev
uv sync --extra serve
uv run --extra serve python -m lev.serve --run franckverrot/lev-350m --port 8009
```

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "model": "lev-latest",
  "questions": {
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                 "shipping": "Delivery status, delays, lost packages",
                                 "billing": "Charges, invoices, payment problems"}},
    "escalate":    {"type": "noul",  "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }}'
```

## Training

Trained on kev's frozen `decision-v4` suite: 1,000 records from each of Banking77, BoolQ, AG News, MNLI, SST-5, Yelp, TREC, DBpedia-14, Amazon reviews and IMDB, plus 896 programmatic policy records.  Two epochs, 42 minutes.  The command is in the repo's README.

## Results

| | lev-350m | kev-0.6b | Jev |
|---|---|---|---|
| parameters | 361M | 607M | hosted |
| accuracy, in-distribution (`decision-v4`) | 0.773 | 0.805 | 0.845 |
| accuracy, out-of-domain (`transfer-v4`) | 0.546 | 0.598 | 0.857 |
| confidently wrong out of domain (p ≥ 0.9), as served | 0.8% | 5.2% | 3.7% |
| request with 3 questions | 25 ms | 46 ms | |
| 24 questions on one state | 110 ms | 223 ms | |
| weights in memory | 1.44 GB | 2.43 GB | |
| training, seconds per record | 0.25 | 0.44 | |

All done on a M2 Max 96GB.

## Credits

[kev](https://github.com/jaredpalmer/kev) by Jared Palmer: the design, the code this started from, the frozen suites and the Jev measurements.  [Archer Hume](https://archerhume.com/posts/jevs-architecture-unmasked) for the architecture reconstruction.  [TypeSafe](https://docs.typesafe.ai/api) for the API contract.  [Liquid AI](https://www.liquid.ai) for the backbone.

## License

The adapter and head are Apache-2.0.  LFM2.5-350M is distributed by Liquid AI under the LFM Open License v1.0; a copy is included as `LICENSE`.  Datasets carry their own licenses.
