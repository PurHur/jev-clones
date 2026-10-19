---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: finetune
new_version: perplexity-ai/pplx-decider-v1.1-27b
library_name: pytorch
pipeline_tag: text-classification
tags:
  - classification
  - multimodal
  - custom-code
---

# pplx-decider-v1-27b

pplx-decider-v1-27b is a decision model fine-tuned from Qwen3.8-27B.

Accuracy across 11 benchmarks. The pplx-decider-v1-27b results were measured through the Perplexity API.

| Benchmark | Jev | Qwen3.8-27B | pplx-decider-v1-27b |
| --- | ---: | ---: | ---: |
| WinoGrande | **90.70%** | 73.10% | 83.30% |
| FinancialPhraseBank | 76.98% | 75.68% | **84.18%** |
| RAGTruth | 77.27% | 61.53% | **88.80%** |
| JudgeBench | **78.57%** | 68.86% | 78.29% |
| BBH | **94.27%** | 72.80% | 82.80% |
| JevBench public hard | **73.27%** | 72.28% | 70.30% |
| TabFact | 89.80% | 78.60% | **90.60%** |
| ContractNLI | 77.45% | **80.78%** | **80.78%** |
| Circa | 84.60% | 87.00% | **89.20%** |
| Belebele | **95.00%** | 93.20% | 94.00% |
| TruthfulQA binary | **92.00%** | 82.80% | 85.40% |
| Overall | 84.51% | 74.76% | **85.71%** |

Bold marks the best score in each row.

## Usage

Python 3.12+ and a CUDA GPU with room for approximately 49 GiB of weights plus working memory.

Download and run the [inference example](inference.py) with uv:

```bash
uvx --from huggingface-hub hf download perplexity-ai/pplx-decider-v1-27b inference.py --local-dir .
uv run inference.py
```

uv installs the dependencies; the script downloads the model from Hugging Face. In an environment with these dependencies installed, use `Decider` directly:

```python
from inference import Decider

model = Decider.from_pretrained("perplexity-ai/pplx-decider-v1-27b")
result = model.predict(
    "My Stripe integration keeps failing. Please help ASAP.",
    {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "Charges and refunds",
            "technical_support": "Integration errors",
            "sales": "Questions about buying a product",
        },
    },
)
print(result)  # Selected choice and calibrated probabilities.
```

Use `{"type": "noul", "instructions": "Does this message express urgency?"}` for a yes/no probability. For images, pass `images=["screenshot.png"]` to `predict`, or run:

```bash
uv run inference.py --image screenshot.png
```

## Acknowledgment

Our data pipeline builds on [Bespoke Nimble](https://github.com/bespokelabsai/nimble)
by Bespoke Labs. The data builder in `source/` uses Nimble's converters to turn
human-labelled public datasets into Choice and Noul decisions. It also uses Nimble's
published benchmark subsets, pinned by record ID and checksum, to keep those benchmark
records and related examples out of training. Nimble is downloaded at a pinned revision
when the data is built and is not redistributed here. If you use this part of the
pipeline, please also cite Nimble:

```bibtex
@misc{nimble2026,
  author = {{Bespoke Labs} and Sathiamoorthy, Maheswaran},
  title = {Nimble},
  year = {2026},
  howpublished = {\url{https://github.com/bespokelabsai/nimble}},
  note = {Model: https://huggingface.co/bespokelabs/Bespoke-Nimble-9B}
}
```
