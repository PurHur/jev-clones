---
license: apache-2.0
base_model: vllm-sr/d3
pipeline_tag: zero-shot-classification
library_name: transformers
tags: [decision-model, system-one]
---

# d3 for 🤗 Transformers

The 255-way answer-code readout of [vllm-sr/d3](https://huggingface.co/vllm-sr/d3) as a JEV decision head, for the `decision`
pipeline. This repo holds the readout, the tokenizer and the prompt; the Qwen3.5 backbone loads from the original repo,
with bidirectional full attention as it was trained.

## Questions

Each question is answered with a probability for every option: `noul` (yes/no), `choice` and `score`.

```python
from transformers import pipeline

decide = pipeline("decision", model="ArthurZ/d3")
answers = decide(
    {
        "state": {"customer": "Lena", "message": "I was charged twice for my order and I want my money back today."},
        "questions": {
            "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
            "tone": {"type": "choice", "instructions": "What is the tone?", "criteria": {"calm": "Polite", "angry": None}},
            "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["Can wait", "This week", "Today", "Now"]},
        },
    }
)
answers["refund"]["noul"]  # probability of true
answers["tone"]["choice"], answers["tone"]["probabilities"]
answers["urgency"]["score"]  # expected level, 0 to 3
```

## Same questions, new states

A bare state is asked the questions of the last request, or the ones given up front:

```python
decide = pipeline(
    "decision",
    model="ArthurZ/d3",
    questions={"spam": {"type": "noul", "instructions": "Is this spam?"}},
)
decide("WIN A FREE IPHONE!!! click here to claim your prize")
decide("Hi! The parcel arrived, thanks a lot.")
```

## Images

Images are `assets`, as in the System One API: a placeholder in the state is replaced with its image. An asset is a
PIL image, a path, a URL or `{"type": "image", "format": "png", "data_base64": ...}`.

```python
decide(
    {
        "state": "The customer sent photo-1 and asks if the product is damaged.",
        "assets": {"photo-1": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"},
        "questions": {"damaged": {"type": "noul", "instructions": "Is the product in the photo damaged?"}},
    }
)
```

Converted with
[`convert_decider_to_hf.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/jev/convert_decider_to_hf.py).
Its prompts are token-for-token those of the original code (`decision_test.json`), except that control tokens written
in a request are escaped instead of read as such.
