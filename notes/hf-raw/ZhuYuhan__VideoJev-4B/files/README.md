---
library_name: transformers
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
base_model_relation: finetune
tags:
- video
- video-question-answering
- temporal-grounding
- custom_code
---

# VideoJev-4B

A video System One model based on [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), using direct candidate scoring for **temporal grounding** and **multiple-choice video QA**.

[Paper](https://arxiv.org/abs/2610.18615) · [Homepage](https://mcg-nju.github.io/VideoJev) · [Code](https://github.com/MCG-NJU/VideoJev) · [Usage guide](USAGE.md)

## Benchmark Results

![VideoJev-4B temporal grounding quality and latency](assets/benchmark-grounding.png)

![VideoJev-4B video QA quality and latency](assets/benchmark-qa.png)

## Quick start

Tested with Python 3.12, PyTorch 2.8 / CUDA 12.8, and an H200. Install a matching PyTorch build first.

```bash
pip install transformers==5.14.1 accelerate safetensors numpy pillow av==18.1.0 \
  flash-linear-attention==0.4.2 fla-core==0.4.2
pip install flash-attn causal-conv1d --no-build-isolation
```

Run grounding and QA on [Eating spaghetti](https://huggingface.co/datasets/ZhuYuhan/VideoJev-Demo/resolve/7e2b9685bca592c3b929268f6fcf82c4ad74b267/eating_spaghetti.mp4) (10 seconds), at 2 FPS.

```python
import json
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoProcessor

video = hf_hub_download(
    repo_id="ZhuYuhan/VideoJev-Demo", filename="eating_spaghetti.mp4",
    repo_type="dataset", revision="7e2b9685bca592c3b929268f6fcf82c4ad74b267",
)
repo = "ZhuYuhan/VideoJev-4B"
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
model = AutoModel.from_pretrained(
    repo, trust_remote_code=True, dtype=torch.bfloat16,
    device_map={"": "cuda:0"}, attn_implementation="flash_attention_2",
)
model.prepare_for_inference()

questions = [
    {"id": "grounding-1", "task": "grounding",
     "query": "The woman in green eats pasta."},
    {"id": "qa-1", "task": "choice",
     "query": "What is the woman in green eating?",
     "options": ["A. Pasta.", "B. Rice.", "C. Soup."]},
]
results = model.predict_video(
    processor, video, questions,
    fps=2.0, max_frames=2048, context=8192,
)
print(json.dumps(results, indent=2))
```

## Output

Actual output from the video and code above:

```json
[
  {
    "id": "grounding-1",
    "task": "grounding",
    "probabilities": [
      0.005761092994362116,
      0.010865824297070503,
      0.026081746444106102,
      0.9046059250831604,
      0.9879851937294006,
      0.9887627959251404,
      0.9896195530891418,
      0.9849089980125427,
      0.833530068397522,
      0.33037981390953064
    ],
    "interval": [
      3.0,
      10.0
    ]
  },
  {
    "id": "qa-1",
    "task": "choice",
    "probabilities": [
      0.9999161958694458,
      5.755242091254331e-05,
      2.617507561808452e-05
    ],
    "answer_index": 0,
    "answer": "A. Pasta."
  }
]
```

`interval` is in seconds; `answer_index` is zero-based. `probabilities` contains independent sigmoid scores for grounding and softmax scores over QA options.

Grounding defaults to `threshold=0.25`; pass `threshold=0.4` to `predict_video` or `predict_features` to override it for one call.

See the [usage guide](USAGE.md) for task-specific settings and output details.

## Citation

```bibtex
@article{zhu2026videojev,
  title={VideoJev: System One Model for Video Understanding and Interaction},
  author={Zhu, Yuhan and Zeng, Xiangyu and Si, Qingyi and Yang, Chenxu and Li, Yifei and Qin, Luozheng and Zhou, Junhao and Yao, Dingyu and Zhang, Jun and Qin, Chuanyu and others},
  journal={arXiv preprint arXiv:2610.18615},
  year={2026}
}
```
