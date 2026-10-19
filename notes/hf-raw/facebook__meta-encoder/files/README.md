---
license: apache-2.0
base_model: meta-models/Muse-Glimmer-30B
base_model_relation: finetune
library_name: transformers
pipeline_tag: feature-extraction
tags:
  - metaencoder
  - multimodal representation
  - similarity
  - image-text-retrieval
  - classification
  - decision making
---

# MetaEncoder-30B

[![arXiv](https://img.shields.io/badge/arXiv-2610.11316-b31b1b?logo=arxiv)](https://arxiv.org/abs/2610.11316)
[![Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-meta--encoder--space-blue)](https://huggingface.co/spaces/facebook/meta-encoder-space)
[![GitHub](https://img.shields.io/badge/GitHub-meta--encoder--eval-181717?logo=github)](https://github.com/facebook/meta-encoder-eval)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](./LICENSE)

**Multimodal System One Encoder** with a natural language interface for instruction-following encoding: give it a *task* and a list of *candidates*,
and it scores the candidates by task specification. Both a task and each candidate are expressed in natural
language — instructions, queries, questions, state descriptions, criteria — with image and
video as side information.

Key features:

- **Natural language interface.** No rigid schemas. Describe the
  task and the candidates in free-form text.
- **Highly-efficient cacheable representations.** Candidates and task are encoded separately by prompt instructions that ground each other.
- **Scales to arbitrarily large candidate sets.** Matching is an inner product over representations, so an ANN index can serve
  millions of candidates without re-running the model.

The model is built by contrastively fine-tuning
[`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B).
It is described in the paper
[MetaEncoder: Exploring the Limit of Bi-Encoders for Multimodal System One Decision Making with Natural Language Interface](https://arxiv.org/abs/2610.11316).

## Results

| Category | Benchmark | **MetaEncoder** | OpenJev (27B) | WeMM-Embedding (9B) | Qwen3-VL-Embedding (8B) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Decision making | JevBench (orig) | 0.9444 | **0.9861** | 0.6250 | 0.6389 |
|  | JevBench (easy) | **1.0** | **1.0** | 0.9792 | 0.9375 |
|  | JevBench (hard) | 0.7387 | **0.7658** | 0.4865 | 0.4324 |
|  | Jev Intelligence | 81.4 | **84.5** | 45.8 | 42.1 |
| Multimodal decision making | ImaJev | **0.8555** | - | 0.8117 | 0.7792 |
| General knowledge understanding | MMLU | **0.7767** | - | 0.7325 | 0.6544 |
| Multimodal understanding | MMMU | **0.5788** | - | 0.5656 | 0.5062 |
|  | VideoMMMU | **0.59** | - | 0.4846 | 0.4187 |
|  | NaturalBench | **0.813** | - | 0.8017 | 0.7097 |
|  | TempCompass | 0.7346 | - | **0.7527** | 0.7172 |
| Multilingual text retrieval | NanoBEIR | **0.6634** | - | 0.6175 | 0.6061 |
| Multimodal retrieval | MMEB V3 (image) | 0.7897 | - | **0.8099** | 0.7796 |
|  | MMEB V3 (video) | 0.6486 | - | **0.7431** | 0.6715 |
|  | MMEB V3 (visdoc) | 0.8149 | - | **0.8334** | 0.8236 |

Bold marks the best score in each row.

For MMLU and MMMU, the option list is added into the task prompt.

## Quick start

```bash
pip install "transformers>=5.15" torch "torchvision<0.27" accelerate pillow
```

`modeling_metaencoder.py` lives in this repo rather than in a package, so fetch the repo once
and put it on your path:

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("facebook/meta-encoder")
sys.path.insert(0, path)
```

## Example
```python
from PIL import Image
from modeling_metaencoder import MetaEncoder

model = MetaEncoder.from_pretrained(path)

# Both a task and a candidate can be multimodal
results = model.match(
    task={
        "image": Image.open("reference_jacket.jpg"),
        "text": "Which of the following media has the same style jacket but in red, with a hood",
    },
    candidates=[
        {"image": Image.open("p1.jpg"), "text": "navy windbreaker, no hood"},
        {"image": Image.open("p2.jpg"), "text": "red hooded parka"},
        {"image": Image.open("p3.jpg"), "text": "black leather biker jacket"},
        {"video": runway_clip, "text": "autumn outerwear runway segment"},
    ],
)

best = results[0]
best.index, round(best.score, 4), best.candidate
```

`match` returns `Match(index, score, candidate)`, best first. Both sides accept text, images
and video in any combination:

- The task's `text` carries the question, query, state, instruction — it is what grounds the task
  to the candidates.
- Each candidate's `text` grounds it to the task, and instruction can also be added into the text to control how that candidate is encoded.

### Another example

```python
action = model.match(
    task={
        "image": Image.open("screen.png"),
        "text": "Goal: cancel the pending order. Choose the next UI action.",
    },
    candidates=[
        {"text": "click the 'Orders' tab in the left sidebar"},
        {"text": "click the red 'Cancel order' button"},
        {"text": "scroll down to the shipping details"},
        {"text": "close the dialog without saving"},
    ],
    top_k=1,
)[0]

action.candidate["text"], round(action.score, 4)
```

### Prompt format used in evaluation

For prompt with closed set, it is always recommended to add the options available in prompt, which improves MMLU and MMMU:
```text
{state}

Task: {instruction}
Criteria:
- {label}: {what the label means}
- {label}: {what the label means}
```

For candidate, there are two scenarios. For the JEV benchmarks, the candidates are the labels with descriptions in the criteria block.
For other datasets, we keep the description in the candidates:
```text
candidates: "white, glycolytic, slow contracting."
            "red, oxidative, fast contracting."
            "red, oxidative, slow contracting."
            ...
```

## Reusing a candidate pool

`match` encodes candidates on every call. For a fixed corpus, embed it once and pass the
tensor back — the scores are identical, and only the task is encoded per query:

```python
index = model.encode(corpus)                        # once
hits = model.match({"text": question}, index, top_k=5)
corpus[hits[0].index]
```

**→ [USAGE.md](./USAGE.md)** covers the full task/candidate spec, instructions, video vs.
multi-image, the exact prompt format, and troubleshooting.

## Serving with vLLM

Text and image inputs can be served with [vLLM](https://github.com/vllm-project/vllm) on its
Transformers backend. Video is not supported there; encode video with `MetaEncoder` as above.

```bash
pip install "vllm==0.23.0" "transformers==5.15.1" "torchvision<0.27"
```

```python
import os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # the model class below is registered in this process

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM
from vllm.config import PoolerConfig

path = snapshot_download("facebook/meta-encoder")
sys.path.insert(0, path)
import vllm_metaencoder

vllm_metaencoder.register()
processor = AutoProcessor.from_pretrained(path)
llm = LLM(
    model=path, runner="pooling", dtype="bfloat16", max_model_len=8192,
    enforce_eager=True,  # torch.compile does not handle this model's patched modules yet
    limit_mm_per_prompt={"image": 2}, attention_backend="FLASH_ATTN",
    pooler_config=PoolerConfig(pooling_type="LAST", use_activation=True),  # last token, L2-normalised
)

task = {"text": "Which photo shows a bicycle?"}
candidates = [{"image": Image.open("a.jpg")}, {"text": "a red bicycle leaning on a wall"}]
outputs = llm.embed(
    [vllm_metaencoder.to_prompt(processor, item) for item in [task, *candidates]],
    tokenization_kwargs=vllm_metaencoder.TOKENIZATION_KWARGS,
)
emb = np.stack([o.outputs.embedding for o in outputs])
scores = emb[1:] @ emb[0]   # cosine similarity of each candidate to the task
```

`vllm_metaencoder.py` ships in this repo and is required: vLLM's generic backend drops the
weight-less norm inside MuseGlimmer's token embedding and resizes its other weight-less norms,
which `register()` restores, and `to_prompt()` builds the exact MetaEncoder prompt with a single
BOS token.

Parity against `MetaEncoder` (bf16, one GPU): cosine similarity 0.9998 on text and 0.993 on images;
on 200 ImageNet-1K images ranked against all 1,000 labels, 85.5% top-1 with vLLM vs 84.5% with
`MetaEncoder`, with the same prediction on 97.5% of images. Tested offline on a single GPU with
the settings above; multi-GPU and `vllm serve` have not been verified yet.

## License and use

Licensed under [Apache 2.0](./LICENSE), inherited from the base model. Use is additionally
subject to the base model's [Usage Policy](./USAGE_POLICY.md). See [NOTICE](./NOTICE) for a
statement of what was changed relative to the base model.

## Citation

```bibtex
@misc{cheng2026metaencoder,
  title         = {MetaEncoder: Exploring the Limit of Bi-Encoders for Multimodal System One Decision Making with Natural Language Interface},
  author        = {Jianpeng Cheng and Guangyu Sun and Aashu Singh and Benyu Zhang and Haixing Dai and Hossein Mansour and Jiangfan Zhang and Shlok Kumar Mishra and Wei Sun and Xuanming Cui and Yanli Liu and Qi Guo and Max Xiangjun Fan and Jun Xiao},
  year          = {2026},
  eprint        = {2610.11316},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.11316}
}
```
