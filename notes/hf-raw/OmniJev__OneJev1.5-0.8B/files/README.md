---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
library_name: transformers
pipeline_tag: image-text-to-text
tags: [onejev, system-one, decision-model, calibration, multimodal, gui-agent, video]
---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
    <img alt="OneJev1.5-0.8B, a Multimodal System One Decision Model" src="assets/banner.svg" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://huggingface.co/OmniJev"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging_Face-OmniJev-FFD21E?style=flat-square&logo=huggingface&logoColor=black"></a>
  <a href="https://github.com/OmniJev/OneJev"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-OmniJev%2FOneJev-181717?style=flat-square&logo=github&logoColor=white"></a>
  <a href="https://omnijev.github.io/OneJev/"><img alt="Website" src="https://img.shields.io/badge/Website-OneJev-0A84FF?style=flat-square&logo=googlechrome&logoColor=white"></a>
  <a href="https://github.com/OmniJev/OneJev/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-D22128?style=flat-square&logo=apache&logoColor=white"></a>
</p>

OneJev1.5-0.8B is the 0.8B model of OneJev 1.5, a multimodal System One decision model. Give it a screenshot, a photo, a video or plain text along with a few typed questions, and it returns a calibrated probability for every option in a single forward pass. It is a full fine-tune of [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) on about 320,000 questions drawn from real agent runs, videos, images and the training splits of public decision datasets.

## Decision Index

On the public part of [Decision Index 0.3](https://huggingface.co/spaces/multimodalart/jev-decision-index) (37 benchmarks, 140,178 requests, run and scored with the board's own kit), OneJev1.5-0.8B scores 42.71, the highest of any model up to 1.5B parameters on the board, and it leads its class in all five areas.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/di-class-table-dark.svg">
    <img alt="Decision Index public part, table: OneJev1.5-0.8B 42.7 against Crivo-150M 36.4, EXAONE-4.0 1.2B 30.7, jiwo 0.8B v2 29.4 and DecisionTune 1.0 28.9; OneJev1.5-0.8B is best overall and in each of the five areas, best value in each row in magenta, second best in light pink" src="assets/di-class-table.svg" width="100%">
  </picture>
</p>

It also scores above 47 larger models on the board, models of 4B to 28B parameters among them.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/di-larger-dark.svg">
    <img alt="Decision Index public part: OneJev1.5-0.8B 42.7 at 0.9B parameters against Sieve 9B 41.9, Jev-Omni 41.0, djev 40.9, Winnow-E4B 40.9, Bespoke Nimble 9B v2 39.9 and Solomon v1.1 38.1 at 8B to 28B" src="assets/di-larger.svg" width="100%">
  </picture>
</p>

Other models' scores are their public-part scores on the board on 12 October 2026.

## Results

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/results-dark.svg">
    <img alt="Results table: OneJev1.5-0.8B against Jev 1.13, Jev-Omni 12B and Qwen3.8-27B thinking; best score in each row in magenta, second best in light pink" src="assets/results.svg" width="100%">
  </picture>
</p>

Accuracy in percent. The OneJev test set holds questions of the training kinds that no model saw in training. Jev 1.13's scores are its published ones, and it reads text only.

## Quick start

```bash
pip install "qev[torch] @ git+https://github.com/OmniJev/OneJev.git"
qev serve --model OmniJev/OneJev1.5-0.8B
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

The server speaks TypeSafe's System One API plus a `media` field for images and video. More examples and the code are on [GitHub](https://github.com/OmniJev/OneJev).

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
