---
license: apache-2.0
language:
- en
- fr
base_model:
- Qwen/Qwen3.5-2B
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- reranker
- cross-encoder
- multimodal
- text-ranking
- document-reranking
- vidore
- beir
- sentence-transformers
datasets:
- vidore/colpali_train_set
- lightonai/embeddings-fine-tuning
---

<div align="center">
<img src="rerank.png" alt="Reranking: the second stage that puts the best text and image candidates on top" width="560">

[![Website](https://img.shields.io/badge/LightOn-Website-blue?logo=google-chrome)](https://lighton.ai)
[![LinkedIn](https://img.shields.io/badge/LightOn-LinkedIn-0A66C2?logo=linkedin)](https://www.linkedin.com/company/lighton/)
[![X](https://img.shields.io/badge/@LightOnIO-X-black?logo=x)](https://x.com/LightOnIO)

📝 [Blog post](https://huggingface.co/blog/lightonai/lighton-rerank)

</div>

<h1 align="center">LightOn-rerank-PW-2B</h1>
<h3 align="center">Unified Text + Visual Document Reranker by LightOn</h3>

<p align="center">
<a href="https://huggingface.co/lightonai/LightOn-rerank-PW-0.8B">PW-0.8B</a> |
<a href="https://huggingface.co/lightonai/LightOn-rerank-LW-0.8B">LW-0.8B</a> |
<a href="https://huggingface.co/lightonai/LightOn-rerank-PW-2B">PW-2B</a> |
<a href="https://huggingface.co/lightonai/LightOn-rerank-LW-2B">LW-2B</a> |
<a href="https://huggingface.co/lightonai/LightOn-rerank-PW-4B">PW-4B</a> |
<a href="https://huggingface.co/lightonai/LightOn-rerank-LW-4B">LW-4B</a>
</p>

---

## About the LightOn-rerank family

Production retrieval pipelines usually need two rerankers: one for text passages and one for visual documents (PDF pages, slides, scans). **LightOn-rerank** models are *unified* cross-encoder rerankers: a single model scores both text passages and document page images against a query, on top of any first-stage retriever (BM25, dense embeddings, or ColPali-family late-interaction models).

The models are built on Qwen3.5 backbone (hybrid linear + full attention) and jointly fine-tuned on text and visual reranking data with mixed-modality batches (LoRA, merged into the released weights). Training data is English-only; French performance transfers zero-shot from the multilingual backbone.

The family comes in two scoring flavours × three sizes (0.8B / 2B / 4B):

- **PW (pointwise)**: each candidate is scored independently. The model judges whether the document answers the query, and the score is `logit("Yes") − logit("No")`. One forward pass per candidate and no generation.
- **LW (listwise)**: generative listwise ranking, where 4 candidates are placed in a single prompt and the model generates a permutation (`[2] > [4] > [1] > [3]`). Larger candidate pools are ranked with a sliding window (window 4, stride 2, bottom-to-top). Cross-document attention makes LW markedly stronger on hard visual reranking, and unlike pointwise scoring it keeps improving with backbone size.

**LightOn-rerank-PW-2B** is the pointwise flagship of the family and its latency floor: the fastest model in the grid that clearly beats the first-stage retriever. On ViDoRe V3 it improves over our previous vision reranker (MonoQwen2-VL-v0.1) by 2.1 nDCG points and outperforms both Qwen3-VL-Reranker-2B and jina-reranker-m0, while keeping the serving simplicity of independent per-candidate scoring. If you can afford ~3× the per-query latency, [LightOn-rerank-LW-2B](https://huggingface.co/lightonai/LightOn-rerank-LW-2B) adds another +2.8 nDCG points at the same size.

## Results

**ViDoRe V3** (visual document reranking, 8 domains × EN/FR queries), overall nDCG@10, ColQwen2.5-v0.2 first stage, retrieve 100 / rerank 100. All models, including baselines, were re-evaluated under this same two-stage protocol, so numbers are mutually comparable but not comparable to vendor-reported end-to-end results.

| Model | Params | Scoring | ViDoRe V3 overall nDCG@10 |
|---|---|---|---|
| [LightOn-rerank-LW-4B](https://huggingface.co/lightonai/LightOn-rerank-LW-4B) | 4.5B | listwise | 64.69 |
| *Qwen3-VL-Reranker-8B* | *8B* | *pointwise (pooling)* | *64.23* |
| [LightOn-rerank-LW-2B](https://huggingface.co/lightonai/LightOn-rerank-LW-2B) | 2.2B | listwise | 62.66 |
| **LightOn-rerank-PW-2B (this model)** | 2.2B | pointwise | **59.87** |
| [LightOn-rerank-PW-4B](https://huggingface.co/lightonai/LightOn-rerank-PW-4B) | 4.5B | pointwise | 59.80 |
| *jina-reranker-m0* | *2.4B* | *pointwise* | *59.40* |
| *Qwen3-VL-Reranker-2B* | *2B* | *pointwise (pooling)* | *59.18* |
| [LightOn-rerank-LW-0.8B](https://huggingface.co/lightonai/LightOn-rerank-LW-0.8B) | 0.85B | listwise | 58.25 |
| *MonoQwen2-VL-v0.1* | *2B* | *pointwise* | *57.76* |
| *First-stage only (ColQwen2.5, no rerank)* | — | — | *55.60* |
| [LightOn-rerank-PW-0.8B](https://huggingface.co/lightonai/LightOn-rerank-PW-0.8B) | 0.85B | pointwise | 48.20 |

### ViDoRe V3 detail (nDCG@10, ColQwen2.5 first stage, rerank-100)

| Domain | EN | FR |
|---|---|---|
| finance_en | 61.13 | 57.16 |
| finance_fr | 49.34 | 52.60 |
| computer_science | 73.98 | 70.47 |
| hr | 66.65 | 61.39 |
| energy | 69.21 | 70.13 |
| industrial | 55.20 | 48.64 |
| pharmaceuticals | 63.85 | 60.39 |
| physics | 48.51 | 49.27 |
| **mean** | **60.98** | **58.76** |

Overall nDCG@10: **59.87** (EN 60.98 / FR 58.76) — +4.3 points over the first stage alone (55.60).

### BEIR results (text reranking)

13 datasets, nDCG@10, BM25 first stage, retrieve 100 / rerank 100, all models re-evaluated under the same protocol. ⚠️ marks datasets in the text training mix (NQ, MSMARCO); the decontaminated mean excludes them.

| Dataset | nDCG@10 |
|---|---|
| fever | 81.00 |
| scifact | 77.55 |
| trec-covid | 83.08 |
| hotpotqa | 71.32 |
| nq ⚠️ | 57.37 |
| dbpedia | 42.21 |
| arguana | 35.53 |
| fiqa | 36.38 |
| msmarco ⚠️ | 37.10 |
| nfcorpus | 35.86 |
| touche-2020 | 37.92 |
| climate-fever | 22.01 |
| scidocs | 17.54 |
| **Mean (13)** | **48.84** |
| **Decontaminated mean (11, excl. ⚠️)** | **49.13** |

Under the same protocol, jina-reranker-m0 scores 51.97 decontaminated mean and Qwen3-VL-Reranker-2B 49.08. This model sits between the two, and is the slightly stronger of our two recipes on text (LW-2B: 48.12) — the mirror image of vision, where listwise wins at every size. jina-reranker-m0 is the only baseline close on both modalities, and it trails on ViDoRe V3 (59.40 vs 59.87).

## Model Details

- **Model type:** multimodal cross-encoder reranker (**pointwise**: each candidate is scored independently as `logit("Yes") − logit("No")`)
- **Base model:** [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (Qwen3.5 hybrid linear + full attention VLM)
- **Parameters:** ≈2.2B (bfloat16, 4.4 GB)
- **Inputs:** query (text) + candidate document(s): text passage **or** page image
- **Fine-tuning:** joint text+vision LoRA (r=32, α=32, rsLoRA, merged into the released weights), mixed-modality batches (2 text + 2 vision groups per micro-batch), vision loss weight 1.3, lr 1e-4, 1 epoch (465 steps)
- **Data:** same 213k groups as the listwise models — 107k text groups (NQ, TriviaQA, MS MARCO; each a `[pos, neg_0, neg_1, neg_2]` 4-list with hard negatives mined via the NV-Retriever approach with GTE-ModernBERT) + 106k vision groups (ColPali train set with negatives mined by Nomic). For pointwise training each group is flattened into (query, document, Yes/No) triples.
- **Languages:** English (training), French (zero-shot transfer)
- **Requirements:** `transformers >= 5.4.0` (`qwen3_5` architecture); `sentence-transformers >= 5.4.0` for the CrossEncoder usage

## Usage: pointwise reranking

Each candidate is scored independently as `logit("Yes") − logit("No")` at the first generated position; sort candidates by descending score. The model was trained with a fixed system prompt and user template: use them verbatim for best results.

### Using Sentence Transformers

Install Sentence Transformers (`>= 5.4.0`):

```bash
pip install "sentence-transformers[image]"
```

The trained system prompt and user templates are baked into the bundled `reranker` chat template, so query-document pairs are formatted correctly out of the box:

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("lightonai/LightOn-rerank-PW-2B")

query = "What is late interaction in neural information retrieval?"
documents = [
    "ColBERT computes token-level query-document interactions at search time...",
    "The Eiffel Tower is located on the Champ de Mars in Paris.",
]

pairs = [(query, doc) for doc in documents]
scores = model.predict(pairs)
print(scores)
# [-4.875 -6.75 ]

rankings = model.rank(query, documents)
print(rankings)
# [{'corpus_id': 0, 'score': -4.875}, {'corpus_id': 1, 'score': -6.75}]
```

To rerank **page images**, pass a `PIL.Image` (or an image URL or file path string) as the document. Text and image candidates can be mixed in the same call:

```python
from PIL import Image

pairs = [
    (query, Image.open("page_1.png")),
    (query, "https://example.com/page_2.png"),
    (query, "A text passage candidate for the same query."),
]
scores = model.predict(pairs)
```

Scores are raw `logit("Yes") − logit("No")` differences. You can map them to 0...1 probabilities with `model.predict(pairs, activation_fn=torch.nn.Sigmoid())`.

### Using Transformers

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "lightonai/LightOn-rerank-PW-2B"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # optional, remove if flash-attn is not installed
    device_map="cuda",
).eval()
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "left"  # scores are read at the last position

YES_TOKEN_ID = 9175  # "Yes"
NO_TOKEN_ID = 2665   # "No"

SYSTEM_PROMPT = "Judge whether the document is relevant to the query. Answer Yes or No."
USER_TEMPLATE = (
    "Given a query, determine if the document is relevant. "
    "The query is: {query}\n\nDocument: {doc}"
)

query = "What is late interaction in neural information retrieval?"
documents = [
    "ColBERT computes token-level query-document interactions at search time...",
    "The Eiffel Tower is located on the Champ de Mars in Paris.",
]

texts = [
    processor.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(query=query, doc=doc)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    for doc in documents
]
inputs = processor(
    text=texts, return_tensors="pt", padding=True, truncation=True, max_length=2048
).to(model.device)
with torch.inference_mode():
    logits = model(**inputs).logits[:, -1]
scores = (logits[:, YES_TOKEN_ID] - logits[:, NO_TOKEN_ID]).tolist()
# [-4.875, -6.75]

ranked = sorted(zip(scores, documents), reverse=True)
```

To score a **page image** instead of a text passage, replace the user message with:

```python
VISION_TEMPLATE = "Given a query, determine if the document image is relevant. The query is: {query}"

{"role": "user", "content": [
    {"type": "image", "image": page_image},  # PIL.Image
    {"type": "text", "text": VISION_TEMPLATE.format(query=query)},
]}
```

and pass `images=[page_image, ...]` to the processor call (keep the same system prompt).

### Serving with vLLM

```bash
vllm serve lightonai/LightOn-rerank-PW-2B --trust-remote-code --max-model-len 16384
```


```python
resp = client.chat.completions.create(
    model="lightonai/LightOn-rerank-PW-2B",
    messages=messages,          # same system + user messages as above
    max_tokens=1,
    logprobs=True,
    top_logprobs=20,
    temperature=0.0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
top = resp.choices[0].logprobs.content[0].top_logprobs
lp = {t.token: t.logprob for t in top}
score = lp.get("Yes", -100.0) - lp.get("No", -100.0)
```

Qwen3.5 chat templates enable thinking by default; without `enable_thinking: False` the first generated token is a `<think>` token and the Yes/No logprobs are distorted. Full-page document images can exceed 8k tokens, so keep `--max-model-len` at 16384 or higher when reranking page images.

## Notes & limitations

- Pointwise scoring does not improve with backbone scale (the 4B pointwise model ties this one at 59.80). If you can afford ~3× the per-query latency, [LightOn-rerank-LW-2B](https://huggingface.co/lightonai/LightOn-rerank-LW-2B) is +2.8 nDCG@10 on ViDoRe V3 at the same size.
- Training data is English-only. French works zero-shot (the backbone is multilingual) but is slightly behind English on average.
- BEIR contamination flag: NQ and MSMARCO are part of the text training data; headline text figures use decontaminated means that exclude them.

## The LightOn-rerank family

| Model | Backbone | Scoring | ViDoRe V3 overall nDCG@10 |
|---|---|---|---|
| [LightOn-rerank-PW-0.8B](https://huggingface.co/lightonai/LightOn-rerank-PW-0.8B) | Qwen3.5-0.8B | pointwise | 48.20 |
| [LightOn-rerank-LW-0.8B](https://huggingface.co/lightonai/LightOn-rerank-LW-0.8B) | Qwen3.5-0.8B | listwise | 58.25 |
| [LightOn-rerank-PW-2B](https://huggingface.co/lightonai/LightOn-rerank-PW-2B) | Qwen3.5-2B | pointwise | 59.87 |
| [LightOn-rerank-LW-2B](https://huggingface.co/lightonai/LightOn-rerank-LW-2B) | Qwen3.5-2B | listwise | 62.66 |
| [LightOn-rerank-PW-4B](https://huggingface.co/lightonai/LightOn-rerank-PW-4B) | Qwen3.5-4B | pointwise | 59.80 |
| [LightOn-rerank-LW-4B](https://huggingface.co/lightonai/LightOn-rerank-LW-4B) | Qwen3.5-4B | listwise | **64.69** |

Rule of thumb: **LW** models are stronger at every size (and the gap grows with size); **PW** models are cheaper to serve and score candidates independently. For the best quality pick [LW-4B](https://huggingface.co/lightonai/LightOn-rerank-LW-4B); for the best quality/cost trade-off pick [LW-2B](https://huggingface.co/lightonai/LightOn-rerank-LW-2B); for maximum throughput on text-heavy workloads pick a PW model.

## Citation

```
@misc{ananya2026lightonrerank,
  title={One Adapter, Both Modalities: Field Notes from Building and Serving a Multimodal Reranker},
  author={Ananya, Ishrat Jahan and Chatelain, Amelie},
  year={2026},
  howpublished={\url{https://huggingface.co/blog/lightonai/lighton-rerank}},
}
```