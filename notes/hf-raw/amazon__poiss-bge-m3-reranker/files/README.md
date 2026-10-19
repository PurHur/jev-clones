---
license: apache-2.0
base_model: BAAI/bge-reranker-v2-m3
library_name: transformers
pipeline_tag: text-classification
datasets:
- amazon/poiss
language:
- en
- es
- fr
- pt
- de
- it
- nl
tags:
- reranker
- cross-encoder
- point-of-interest
- poi-search
- geospatial
- multilingual
---

# POISS BGE-Reranker-v2-m3

`BAAI/bge-reranker-v2-m3` fine-tuned on [POISS](https://huggingface.co/datasets/amazon/poiss)
to rerank point-of-interest candidates. It is a cross-encoder: query and candidate are encoded
jointly, so unlike the bi-encoder it can exploit pairwise features like the distance between
the user and the POI. Taking advantage of the cross-encoder's stronger expressive capabilities
compared with a bi-encoder, the reranker was trained to incorporate the POI's quality score
in addition to distance. Both distance and quality score are represented as natural-language buckets.

Intended position in the pipeline: retrieve with
[`amazon/poiss-bge-m3-retriever`](https://huggingface.co/amazon/poiss-bge-m3-retriever), then
rerank its top-k with this model. Output is a single logit per pair; higher is more relevant.

The numbers below were measured on the pool the paper uses — the retriever's **top-1000 from the
full 72M-POI corpus** — which is not distributed with the dataset. Reranking the dataset's
`candidates` column instead is an easier setting (every graded positive is present by
construction) and gives different, non-comparable numbers. To reproduce this table, rebuild the
pool by embedding the corpus with the retriever above.

## Results

Reranking the fine-tuned retriever's top-1000 on the POISS test split (42,018 queries,
percentages):

| System | P@5 | P@20 | MRR | N@5 | N@20 |
|---|---|---|---|---|---|
| BGE-M3 retriever (bi-encoder order) | 57.6 | 35.5 | 85.5 | 61.5 | 62.6 |
| BGE-Reranker-v2-m3 zero-shot | 24.6 | 14.3 | 50.0 | 32.1 | 34.8 |
| **this model** | **63.0** | **38.3** | **89.7** | **66.8** | **66.6** |

Per-intent nDCG@20: `Search` 76.8, `Detail` 78.9, `Recommend` 56.3, `Things-to-do` 55.0.
Precision is higher on open-ended intents simply because their relevant sets are larger; the
rank-aware metrics show the ordering there is the harder problem.

## Usage

The model scores a **text pair**: a query string and a POI string. Field order, prompt labels,
and the bucket vocabularies below are part of the trained format.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "amazon/poiss-bge-m3-reranker"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()

SCORE_BUCKETS = [  # upper bound (exclusive), label
    (1.0, "Very Poor"), (1.5, "Poor"), (2.0, "Somewhat Poor"), (2.5, "Fair"),
    (3.0, "Average"), (3.5, "Good"), (4.0, "Very Good"), (4.5, "Excellent"),
    (5.0, "Outstanding"), (5.1, "Perfect"),
]
DISTANCE_BUCKETS = [  # upper bound in km (exclusive), label
    (2.0, "extremely close"), (4.0, "very close"), (6.0, "close"), (8.0, "nearby"),
    (10.0, "moderately close"), (15.0, "moderately far"), (20.0, "far"),
    (30.0, "quite far"), (50.0, "very far"), (320.0, "extremely far"),
]


def bucket(value, buckets):
    for upper, label in buckets:
        if value < upper:
            return label
    return buckets[-1][1]


def format_query(text):
    return f"Query: {text[:100]}"


def format_poi(name, categories, distance_km, score, address):
    # categories: a list; the first 10 are kept and joined by a space
    fields = [f"Name: {name[:60]}", f"Categories: {' '.join(categories[:10])}"]
    if distance_km is not None and distance_km >= 0:
        fields.append(f"Distance: {bucket(distance_km, DISTANCE_BUCKETS)}")
    fields.append(
        f"Score: {bucket(score, SCORE_BUCKETS)}" if score and score > 0 else "Score: None"
    )
    fields.append(f"Address: {address[:60]}")
    return "; ".join(fields)


query = format_query("pizzeria near the colosseum")
pois = [
    format_poi("Pizzeria Da Enzo", ["pizza_restaurant", "restaurant"], 0.6, 4.3,
               "Via dei Fori Imperiali 1, Rome"),
    format_poi("Oslo Dental Clinic", ["dentist"], 2100.0, 3.1, "Storgata 10, Oslo"),
]

batch = tokenizer([query] * len(pois), pois, padding=True, truncation=True,
                  max_length=256, return_tensors="pt")
with torch.no_grad():
    scores = model(**batch).logits.squeeze(-1)
print(scores)  # higher = more relevant
```

Details that matter:

- Field order is fixed: `Name`, `Categories`, `Distance`, `Score`, `Address`, joined by `"; "`.
- `Categories` is the first 10 category labels joined by a space; the query is truncated at 100
  characters, and anything after a `"; Position:"` marker is stripped from it.
- `Score` is the POISS 0–5 quality score, shipped with the dataset as
  [`data/poi_quality_score.parquet`](https://huggingface.co/datasets/amazon/poiss/blob/main/data/poi_quality_score.parquet)
  (it is not an Overture field). When it is missing or non-positive the field is still emitted,
  as `Score: None`. Values above 5 are halved and clamped.
- The score is **sparse**: it covers 30.7% of query–candidate pairs in POISS (43.7% of the
  graded positives). `Score: None` is therefore the common case, and is what this model saw for
  about two thirds of its training pairs — not a degraded mode.
- `Distance` is the haversine distance in km between the user coordinate and the POI, and is
  omitted entirely when unavailable.
- Truncation: 256 tokens for the pair; the source cutoffs are 100 characters for the query, 60
  for the name and address, 10 categories.

## Training

Fine-tuned from `BAAI/bge-reranker-v2-m3` on the POISS train split (254,089 training and 13,374
validation records), with POI content resolved from Overture Places release `2026-03-18`.

| | |
|---|---|
| Objective | binary cross-entropy on the pair logit |
| Hard negatives | 100 mined per query, 16 sampled per step |
| Optimizer | AdamW, lr 1e-6, gradient clipping 1.0 |
| Batch size | 4 per GPU × 8 GPUs |
| Max sequence length | 256 |
| Precision / hardware | mixed precision, 8× NVIDIA A100 40GB |
| Schedule | up to 20 epochs, best checkpoint by validation nDCG@20 |
| Selected checkpoint | epoch 18, validation nDCG@20 0.792 |

Negatives are the candidates the LLM judge marked non-relevant, padded from the reranked
top-300 starting at rank 30. Positives are the graded `relevant` POIs.

The quality score the model consumes is **LLM-generated from Overture POI metadata**.

## Reproducibility note

The published weights are the checkpoint used for the paper, re-expressed in the stock
`XLMRobertaForSequenceClassification` layout; all 393 tensors are bit-identical to the training
checkpoint.

## Intended use and limitations

Intended for research on POI search reranking. Reranking quality is bounded by the candidate
set: relevant POIs the retriever never surfaces cannot be recovered, and on POISS the
fine-tuned retriever's R@1k is 95.6, so a few percent of graded POIs are out of reach by
construction. Coverage is limited to 7 languages concentrated in the Americas and Europe, so
behaviour outside those locales is untested.

## Citation

```bibtex
@inproceedings{maritan-etal-2026-poiss,
    title = "{POISS}: A Large-Scale Multilingual Dataset for Point-of-Interest Search",
    author = "Maritan, Nicola  and
      Moschitti, Alessandro  and
      Borazio, Federico  and
      Zhou, Xiaokun  and
      Bai, Zhengwei",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    note = "To appear",
}
```

## License

These weights are a derivative work of
[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3), distributed by BAAI
under the **Apache License 2.0**, and are released under the same license. A copy is included
as `LICENSE`, and `NOTICE` records the required statement of modifications: all weights were
modified by supervised fine-tuning on the POISS train split with a binary cross-entropy
objective, with no change to architecture, configuration, vocabulary or tokenizer.

The base model itself derives from `BAAI/bge-m3` (MIT), which derives from XLM-RoBERTa (MIT).

The training data, [POISS](https://huggingface.co/datasets/amazon/poiss), is Apache-2.0 and
places no additional restriction on these weights.
