---
base_model: Qwen/Qwen3-0.6B
library_name: transformers
tags:
- social-jev
- classification
---
# SocialJev 0.6B — Preview

SocialJev is a family of classification foundation models for the social sciences, inspired by [TypeSafe’s Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). The models combine the flexibility of modern language models pretrained on trillions of tokens of text with the speed of lightweight NLP methods. Define your task through a simple typed interface and SocialJev returns typed outputs.

[Website](https://social-jev.github.io/) · [Documentation](https://social-jev.github.io/docs/) · [GitHub](https://github.com/social-jev/social-jev)

**Note:** SocialJev is under active development. Interfaces may change between early releases.

## Try with Transformers

The model can be used directly through Transformers without installing the SocialJev package:

```bash
uv pip install torch transformers pydantic
```

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "social-jev/social-jev-0.6B-preview",
    trust_remote_code=True,
)
query = model.Query(
    text="Which policy area does this document concern?",
    options=["Transport", "Education", "Healthcare"],
)
requests = [
    model.Request(state="The council expanded the bus network.", queries=[query]),
    model.Request(state="The government recruited more teachers.", queries=[query]),
]

results = model.query(requests)
print(results[0][0].choice)
print(results[0][0].probabilities)
```

A possible output:

```text
Transport
{'Transport': 0.98, 'Education': 0.01, 'Healthcare': 0.01}
```

Results are ordered first by document, then by query for that document. Each result provides `.choice` and `.probabilities`; the probabilities above are illustrative.

## Larger workloads with SocialJev and vLLM

The SocialJev package provides the same typed interface across inference backends. Its standard installation uses Transformers:

```bash
uv pip install social-jev
```

For larger workloads, install the optional vLLM dependencies:

```bash
uv pip install "social-jev[vllm]"
```

The following example uses local vLLM with synchronous submission:

```python
from social_jev import Query, Request, SocialJev

model = SocialJev.from_pretrained(
    "social-jev/social-jev-0.6B-preview",
    backend="vllm",
)
query = Query(
    text="Which policy area does this document concern?",
    options=["Transport", "Education", "Healthcare"],
)
requests = [
    Request(state="The council expanded the bus network.", queries=[query]),
    Request(state="The government recruited more teachers.", queries=[query]),
]

try:
    results = model.query(requests)
    print(results[0][0].choice)
    print(results[0][0].probabilities)
finally:
    model.stop()
```

Both local and server-based vLLM support synchronous `.query()` and asynchronous `.aquery()`. Concurrent asynchronous submissions can overlap batch preparation and processing with inference.

See the [inference guide](https://social-jev.github.io/docs/v0.1.0a4/inference/) for execution patterns and configuration, and the [API reference](https://social-jev.github.io/docs/v0.1.0a4/api/) for inputs, outputs, and loading arguments.
