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

<h1 align="center">LightOn-rerank-LW-2B</h1>
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

**LightOn-rerank-LW-2B** is the best 2B-class model of the family: 62.66 nDCG@10 on ViDoRe V3, improving over the ColQwen2.5 first stage by 7.1 nDCG points and leading every open 2B-class multimodal reranker we measured. Putting 4 candidates in a single forward pass lets documents attend to each other, which fixes the discriminative weakness of independent per-candidate scoring: +2.8 nDCG@10 over the pointwise sibling with identical backbone and data, winning 13 of 16 splits.

## Results

**ViDoRe V3** (visual document reranking, 8 domains × EN/FR queries), overall nDCG@10, ColQwen2.5-v0.2 first stage, retrieve 100 / rerank 100. All models, including baselines, were re-evaluated under this same two-stage protocol, so numbers are mutually comparable but not comparable to vendor-reported end-to-end results.

| Model | Params | Scoring | ViDoRe V3 overall nDCG@10 |
|---|---|---|---|
| [LightOn-rerank-LW-4B](https://huggingface.co/lightonai/LightOn-rerank-LW-4B) | 4.5B | listwise | 64.69 |
| *Qwen3-VL-Reranker-8B* | *8B* | *pointwise (pooling)* | *64.23* |
| **LightOn-rerank-LW-2B (this model)** | 2.2B | listwise | **62.66** |
| [LightOn-rerank-PW-2B](https://huggingface.co/lightonai/LightOn-rerank-PW-2B) | 2.2B | pointwise | 59.87 |
| [LightOn-rerank-PW-4B](https://huggingface.co/lightonai/LightOn-rerank-PW-4B) | 4.5B | pointwise | 59.80 |
| *jina-reranker-m0* | *2.4B* | *pointwise* | *59.40* |
| *Qwen3-VL-Reranker-2B* | *2B* | *pointwise (pooling)* | *59.18* |
| [LightOn-rerank-LW-0.8B](https://huggingface.co/lightonai/LightOn-rerank-LW-0.8B) | 0.85B | listwise | 58.25 |
| *MonoQwen2-VL-v0.1* | *2B* | *pointwise* | *57.76* |
| *First-stage only (ColQwen2.5, no rerank)* | — | — | *55.60* |
| [LightOn-rerank-PW-0.8B](https://huggingface.co/lightonai/LightOn-rerank-PW-0.8B) | 0.85B | pointwise | 48.20 |

### ViDoRe V3 detail (nDCG@10, ColQwen2.5 first stage, rerank-100, sliding window 4/2)

| Domain | EN | FR |
|---|---|---|
| finance_en | 71.43 | 62.02 |
| finance_fr | 47.59 | 51.08 |
| computer_science | 82.26 | 79.23 |
| hr | 68.75 | 61.95 |
| energy | 67.11 | 70.67 |
| industrial | 58.03 | 50.27 |
| pharmaceuticals | 68.10 | 65.36 |
| physics | 48.84 | 49.88 |
| **mean** | **64.01** | **61.31** |

Overall nDCG@10: **62.66** (EN 64.01 / FR 61.31), the best 2B-class result under this protocol: +3.5 over the official Qwen3-VL-Reranker-2B (59.18) and +3.3 over jina-reranker-m0 (59.40). The training data is English-only, so the FR column is pure zero-shot cross-lingual transfer; FR even beats EN on three domains (finance_fr, energy, physics).

### BEIR results (text reranking)

13 datasets, nDCG@10, BM25 first stage, retrieve 100 / rerank 100, all models re-evaluated under the same protocol (this model runs the same W=4 stride=2 sliding window it uses for document pages; baselines score through their native text paths). ⚠️ marks datasets in the text training mix (NQ, MSMARCO); the clean mean excludes them.

| Dataset | LightOn-rerank-LW-2B | Qwen3-VL-Reranker-2B | jina-reranker-m0 |
|---|---|---|---|
| fever | **78.92** | 73.34 | 80.51 |
| scifact | 75.94 | 77.33 | 79.68 |
| trec-covid | 71.78 | **83.47** | 83.25 |
| hotpotqa | 71.92 | 75.00 | 76.09 |
| nq ⚠️ | 54.84 | 57.44 | 60.51 |
| dbpedia | 39.48 | 39.81 | 45.66 |
| arguana | **41.25** | 36.78 | 40.68 |
| fiqa | 37.54 | 37.50 | 44.64 |
| msmarco ⚠️ | 36.95 | 38.44 | 38.21 |
| nfcorpus | 35.14 | 37.99 | 37.35 |
| touche-2020 | 34.83 | 33.75 | 36.09 |
| climate-fever | 23.81 | 25.21 | 26.61 |
| scidocs | 18.76 | 19.69 | 21.11 |
| **Mean (13)** | 47.78 | 48.90 | **51.57** |
| **Clean mean (11, excl. ⚠️)** | 48.12 | 49.08 | **51.97** |

The ordering flips relative to ViDoRe V3: on vision this model leads the 2B class; on text jina-reranker-m0 leads decisively, and this model is roughly tied with Qwen3-VL-Reranker-2B. On text the pointwise sibling is slightly stronger ([PW-2B](https://huggingface.co/lightonai/LightOn-rerank-PW-2B): 49.13 clean mean) — the mirror image of vision, where listwise wins at every size.

## Model Details

- **Model type:** multimodal cross-encoder reranker (**generative listwise**: 4 candidates per prompt, ranked by generating a permutation; sliding window (4, stride 2) for larger pools)
- **Base model:** [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (Qwen3.5 hybrid linear + full attention VLM)
- **Parameters:** ≈2.2B (bfloat16, 4.4 GB)
- **Inputs:** query (text) + candidate document(s): text passage **or** page image
- **Fine-tuning:** joint text+vision LoRA (r=32, α=32, rsLoRA, merged into the released weights), mixed-modality batches (2 text + 2 vision groups per micro-batch), vision loss weight 1.3, lr 5e-5, warmup 30%, 1 epoch (419 steps), training images resized to 512×512
- **Data:** 213k listwise groups — 107k text groups (NQ, TriviaQA, MS MARCO; each a `[pos, neg_0, neg_1, neg_2]` 4-list with hard negatives mined via the NV-Retriever approach with GTE-ModernBERT) + 106k vision groups (ColPali train set with negatives mined by Nomic). The gold permutation (`pos > neg_0 > neg_1 > neg_2`) is constructed directly from the mining metadata; training is cross-entropy on the permutation tokens only.
- **Languages:** English (training), French (zero-shot transfer)
- **Requirements:** `transformers >= 5.4.0` (`qwen3_5` architecture)

## Usage: generative listwise reranking

The model ranks **4 candidates per prompt** by generating a permutation string such as `[2] > [1] > [4] > [3]`. Candidate pools larger than 4 are ranked with a sliding window (window 4, stride 2) moving from the bottom of the list to the top, so the best candidates bubble up to the front. If a generation cannot be parsed, fall back to the input order (observed fallback rate in our evals: ≈0.01%).

```python
import re
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "lightonai/LightOn-rerank-LW-2B"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # optional, remove if flash-attn is not installed
    device_map="cuda",
).eval()
processor = AutoProcessor.from_pretrained(model_id)

PROMPT = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
PERM_RE = re.compile(r"\[(\d)\]\s*>\s*\[(\d)\]\s*>\s*\[(\d)\]\s*>\s*\[(\d)\]")


def rank_window(query: str, docs: list[str]) -> list[int]:
    """Rank exactly 4 text passages; returns window indices, most relevant first."""
    body = "\n".join(f"[{i + 1}]: {d}" for i, d in enumerate(docs))
    user = f"Query: {query}\n\nRank these passages from most to least relevant:\n{body}\n\nRanking:"
    inputs = processor(text=[PROMPT.format(user=user)], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=30, do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    completion = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    m = PERM_RE.search(completion)
    return [int(g) - 1 for g in m.groups()] if m else list(range(4))


def rerank(query: str, docs: list, window: int = 4, stride: int = 2) -> list[int]:
    """Sliding-window listwise rerank; returns document indices, most relevant first."""
    order = list(range(len(docs)))
    positions = list(range(max(0, len(docs) - window), -1, -stride))
    if positions and positions[-1] != 0:
        positions.append(0)
    for pos in positions:
        end = min(pos + window, len(docs))
        p = max(0, end - window)
        if end - p < 2:
            continue
        perm = rank_window(query, [docs[i] for i in order[p:end]])
        order[p:end] = [order[p + j] for j in perm]
    return order


query = "What is late interaction in neural information retrieval?"
documents = ["passage 1 ...", "passage 2 ...", "passage 3 ...", "passage 4 ...", "passage 5 ..."]
print(rerank(query, documents))
```

For **page images**, build the window prompt with interleaved image placeholders instead:

```python
def rank_window_images(query: str, images: list) -> list[int]:  # 4 PIL images
    content = [{"type": "text", "text": f"Query: {query}\n\nRank these documents from most to least relevant:\n"}]
    for i, img in enumerate(images):
        content += [
            {"type": "text", "text": f"[{i + 1}]: "},
            {"type": "image", "image": img},
            {"type": "text", "text": "\n"},
        ]
    content.append({"type": "text", "text": "\nRanking:"})
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=list(images), return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=30, do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    completion = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    m = PERM_RE.search(completion)
    return [int(g) - 1 for g in m.groups()] if m else list(range(4))
```

If a window has fewer than 4 candidates, pad it by repeating the last candidate and drop the duplicates from the returned order. Our evaluations ran this model with HF `generate()`.

## Notes & limitations

- Cross-document attention is the key architectural ingredient: a listwise *loss* over independently scored documents regressed vision performance well below the pointwise baseline in our ablations; generative listwise (all 4 documents in one prompt) is what flips the sign. For the same reason, don't read this model pointwise at deploy time (e.g. a Yes/No readout) — on our smoke split that cost roughly three quarters of the rerank lift.
- **Serve at high image resolution.** The model was trained at 512×512, but inference quality keeps improving with resolution: 62.66 at the processor's default (~14.6k ViT patches/page) vs 61.90 at 1M pixels (−0.8) and 57.06 at the 512² training resolution (−5.6). 1M pixels is a good operating point if ViT compute is a concern.
- **To cut latency, shrink the candidate pool, not the scoring method**: reranking the top-20 candidates instead of 100 keeps 85% of the rerank lift at ~5× fewer windows (top-10: 58% at ~12× fewer), measured over the full benchmark.
- Training data is English-only. French works zero-shot (the backbone is multilingual) but is slightly behind English on average.
- BEIR contamination flag: NQ and MSMARCO are part of the text training data; headline text figures use clean means that exclude them.

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