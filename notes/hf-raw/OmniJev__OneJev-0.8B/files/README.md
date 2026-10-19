---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
library_name: transformers
datasets:
  - OmniJev/OneJev-Data
pipeline_tag: image-text-to-text
tags: [onejev, system-one, decision-model, calibration, multimodal, gui-agent, video]
---


<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
    <img alt="OneJev-0.8B, a Multimodal System One Decision Model" src="assets/banner.svg" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://huggingface.co/collections/OmniJev/onejev"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging_Face-OneJev-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
  <a href="https://huggingface.co/spaces/BradNLP/OneJev"><img alt="Demo" src="https://img.shields.io/badge/Demo-Try_it-F83800?style=flat-square&logo=gradio&logoColor=white"></a>
  <a href="https://github.com/OmniJev/OneJev"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-OmniJev%2FOneJev-181717?style=flat-square&logo=github&logoColor=white"></a>
  <a href="https://omnijev.github.io/OneJev/"><img alt="Website" src="https://img.shields.io/badge/Website-OneJev-0A84FF?style=flat-square&logo=googlechrome&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/OmniJev/OneJev-Data"><img alt="Data" src="https://img.shields.io/badge/Data-OneJev--Data-FF9D00?style=flat-square&logo=huggingface&logoColor=white"></a>
  <a href="https://github.com/OmniJev/OneJev/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-D22128?style=flat-square&logo=apache&logoColor=white"></a>
</p>

OneJev-0.8B is the 0.8B model of OneJev, a multimodal System One decision model. Give it a screenshot, a photo, a video or plain text along with a few typed questions, and it returns a calibrated probability for every option in a single forward pass. It is a full fine-tune of [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) on 99,193 questions drawn from real agent runs, videos and images.

## Results

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/results-dark.svg">
    <img alt="Results table: OneJev-0.8B against Jev 1.13, Jev-Omni 12B and Qwen3.8-27B thinking; best score in each row in magenta, second best in light pink" src="assets/results.svg" width="100%">
  </picture>
</p>

Accuracy in percent. The OneJev test set holds questions of the training kinds that no model saw in training. Jev 1.13's scores are its published ones, and it reads text only.

On one H200 with a 1280x720 screenshot, OneJev-0.8B answers 1 question in 31 ms and 10 questions in one request in 51 ms (5.1 ms per question).

## Quick start

```bash
pip install "qev[torch] @ git+https://github.com/OmniJev/OneJev.git"
qev serve --model OmniJev/OneJev-0.8B
```

```python
from qev import Client, Choice, Noul
from qev.media import data_uri

r = Client("http://localhost:8000").system_one(
    state={"task": "Pay the open invoice from ACME", "screen": "<image:1>"},
    media=[{"type": "image", "data": data_uri("screenshot.png")}],
    questions={"done": Noul("The invoice has been paid"),
               "next": Choice("What should the agent do next?", {"click": "click an element", "stop": "stop"})},
)
```

The server speaks TypeSafe's System One API plus a `media` field for images and video. More examples, the latency
benchmark and the code are on [GitHub](https://github.com/OmniJev/OneJev).

OneJev-0.8B also runs on llama.cpp from the [GGUF build](https://huggingface.co/mradermacher/OneJev-0.8B-GGUF) by mradermacher,
without PyTorch. Images work there; video needs the PyTorch server.

```bash
brew install llama.cpp
pip install git+https://github.com/OmniJev/OneJev.git
qev serve --gguf mradermacher/OneJev-0.8B-GGUF:Q8_0
```

## All sizes

| Model | Base | Weights |
|---|---|---:|
| **OneJev-0.8B** | Qwen3.5-0.8B | 2.2 GB |
| [OneJev-4B](https://huggingface.co/OmniJev/OneJev-4B) | Qwen3.5-4B | 10.4 GB |
| [OneJev-9B](https://huggingface.co/OmniJev/OneJev-9B) | Qwen3.5-9B | 18.8 GB |
| [OneJev-27B](https://huggingface.co/OmniJev/OneJev-27B) | Qwen3.8-27B | 54.7 GB |
| [OneJev-27B-FP8](https://huggingface.co/OmniJev/OneJev-27B-FP8) | OneJev-27B in 8-bit | 30.4 GB |

All sizes are in the [OneJev collection](https://huggingface.co/collections/OmniJev/onejev).

## Citation

```bibtex
@misc{onejev2026,
  title        = {{OneJev}: A Multimodal System One Decision Model},
  author       = {{OmniJev Team}},
  year         = {2026},
  howpublished = {\url{https://github.com/OmniJev/OneJev}}
}
```

## License

Apache 2.0
