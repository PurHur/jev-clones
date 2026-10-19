---
license: cc-by-4.0
pipeline_tag: feature-extraction
library_name: transformers
tags:
- embeddings
- multimodal
- retrieval
- compositional-reasoning
- vision
- reranker-distillation
---

<h1 align="center">Core-Embed: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.04083"><img src="https://img.shields.io/badge/arXiv-CORE-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/"><img src="https://img.shields.io/badge/GitHub-CODE-black.svg" alt="GitHub"></a>
  <img src="https://img.shields.io/badge/License-CC--BY--4.0-blue.svg" alt="License: CC-BY-4.0">
</p>

Core-Embed is an MLLM-based multimodal embedding model that resolves fine-grained attribute-object bindings by distilling a reranker's compositional judgments into the embedding space.

## Model Family

| Model | Backbone | Parameters | Outputs | Modalities |
|---|---|---:|---|---|
| core-emb-2b | VL-Emb (Qwen3-VL) | 2B | Dense embedding | Text, image |
| core-emb-8b | VL-Emb (Qwen3-VL) | 8B | Dense embedding | Text, image |
| core-reranker-2b | Qwen3-VL-Reranker | 2B | Relevance score | Text, image |
| core-reranker-8b | Qwen3-VL-Reranker | 8B | Relevance score | Text, image |

## Highlights

- **Compositional retrieval**: distinguishes scenes with the same concepts but different attribute-object bindings (e.g., "a white plate and a black chair" vs. "a black plate and a white chair").
- **Rank-KL distillation**: trained to reproduce the reranker teacher's fine-grained ranking over a five-level compositional matching spectrum (full match, partial presence, attribute error, object error, full mismatch), instead of collapsing all negatives into one class as InfoNCE does.
- **General retrieval preserved**: continual training from a strong MLLM embedding backbone, so COCO and Flickr30k retrieval quality is retained while compositional accuracy improves.

## Training

Candidate lists are synthesized from LAION-400M seed images: Qwen3-VL-32B extracts structured scene representations and generates queries plus five captions spanning the matching levels, Z-Image-Turbo generates the candidate images, and MLLM-based verification filters low-quality tuples. The student (VL-Emb) is then trained with the Rank-KL objective — a KL divergence between temperature-softened teacher reranker scores and student cosine similarities over each candidate list. Core-Reranker models are fine-tuned from Qwen3-VL-Reranker on the same synthesized data.

## Results

On compositional reasoning benchmarks (COLA, SugarCrepe++, NegBench):

- **Core-Reranker-8B**: 82.7% total average, +10.7 points over Jina-Reranker, while recovering negation sensitivity that standard reranker fine-tuning erodes.
- **Core-Embed-8B**: best total average (0.666) among all evaluated embedding models, +5.7 points over its VL-Emb-8B backbone.
- Gains transfer to the MCMR multi-condition retrieval benchmark (R@1 0.375 → 0.412) with COCO/Flickr30k performance fully preserved.

See the paper for the full evaluation across 12 embedding baselines and 5 reranker baselines.

## Usage

Requires a recent `transformers` build with Qwen3-VL support:

```bash
pip install "transformers>=4.57.0" torch qwen-vl-utils pillow
```

The model is loaded through the wrapper classes provided in the [GitHub repository](https://github.com/):

```python
import torch
from models.qwen3_vl.qwen3_vl_wrapper import Qwen3VLForEmbedding, Qwen3VLForEmbeddingProcessor

model_path = "core-emb-8b"
model = Qwen3VLForEmbedding.from_pretrained(
    model_path, device_map="cuda", torch_dtype=torch.bfloat16
)
processor = Qwen3VLForEmbeddingProcessor.from_pretrained(
    model_path,
    instruction_standalone=True,
    max_length=3024,
    min_pixels=32*32*4,
    max_pixels=32*32*1280,
    total_pixels=32*32*4500,
    num_frames=48,
)

query_inst = "Find me an everyday image that matches the given caption: "
psg_inst = "Represent the given image."

queries = ["a white plate and a black chair"]
inputs = processor(
    texts=queries, images=[None], videos=[None],
    instruction=[query_inst], is_query=True,
)
with torch.inference_mode():
    query_emb = model(**inputs.to(model.device)).last_hidden_state[:, -1]

images = ["path/to/image.jpg"]
inputs = processor(
    texts=[None], images=images, videos=[None],
    instruction=[psg_inst], is_query=False,
)
with torch.inference_mode():
    img_emb = model(**inputs.to(model.device)).last_hidden_state[:, -1]

print(query_emb @ img_emb.T)
```

For FlashAttention acceleration, pass `attn_implementation="flash_attention_2"` when loading the model.

For evaluation on COLA, SugarCrepe++, NegBench, COCO, Flickr30k, and MCMR, use the evaluation code in the [GitHub repository](https://github.com/).

## Citation

If you use Core-Embed, please cite the paper:

```bibtex
@misc{song2026core,
      title={CORE: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation}, 
      author={Tingyu Song and Mingxin Li and Yanzhao Zhang and Dingkun Long and Chu Liu and Pengjun Xie and Yilun Zhao and Shu Wu},
      year={2026},
      eprint={2609.04083},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.04083}, 
}
```

## Acknowledgements
We thank the authors of [COLA](https://arxiv.org/abs/2305.03689), [SugarCrepe++](https://arxiv.org/abs/2406.11171), and [NegBench](https://github.com/m1k2zoo/negbench) for their benchmarks, and the [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) project for the evaluation framework.
