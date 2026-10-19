---
pipeline_tag: zero-shot-classification
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
library_name: transformers
tags:
- zero-shot-classification
- decision-model
- classification
- system-one
- multimodal
- vision
- video
- safetensors
---

![d3-edge: Decision 3.0](assets/banner.png)

# d3-edge

**d3-edge** is the 0.6B multimodal foundation decision model of Decision 3.0, the decision models of [vLLM Semantic Router](https://github.com/vllm-project/semantic-router). Give it an input (text or JSON, optionally with images and videos) and the questions you need answered: pick one of several options, say yes or no, or rate on a scale. It answers them all in one call and returns a probability for every answer, without generating text.

**Try it free** at [decision.vllm-sr.ai](https://decision.vllm-sr.ai) · Official Space: [vllm-sr/Decision-3.0](https://huggingface.co/spaces/vllm-sr/Decision-3.0)

| | |
| --- | --- |
| **Parameters** | 0.59B, including the 0.10B vision encoder |
| **Inputs** | Text or JSON, plus images and videos (several per request) |
| **Decision types** | Choice · Yes / No · Score |
| **License** | Apache-2.0 |

## Highlights

- **Jev Decision Index 0.3.2: 20.2 on the official text board**, second among the entries of similar size; public suite 28.82, measured with the official 0.3 kit on the released weights: all 140,178 public requests answered, none unsupported.
- **+6.6 over Decision 2.0** (its 0.6B model: 13.6 on the official board): +12.1 on the public suite, ahead in all five of its areas, and +7.4 and +1.1 on the private same-skill tests and new-domain tasks.
- **Reads images:** multiple images per request (PNG, JPEG or WebP), given as paths, URLs, PIL images or base64 data URLs; every question of the request sees all of them.
- **Reads videos:** multiple videos per request (MP4, WebM, MOV or MKV), given as paths, URLs, base64 data URLs or frame arrays, read at 2 frames per second; **Perception Test (multiple-choice video QA): 46.6** (internal evaluation).
- **Speed:** a median of 6.6 ms for a text request, 58.6 ms for a request with an image and 274 ms for a request with a 10-second video, on one AMD Instinct MI325X GPU, one request at a time with AutoModel.
- **Serving:** 861 requests/s with p99 under 1 s on one AMD Instinct MI325X with vLLM Semantic Router, 6.1× AutoModel (one-question text, with batching, now the default); with one request per forward, the former default, it gave AutoModel's answers byte for byte when both used the same kernel settings ([Serving performance](#serving-performance)).
- **Many questions, one call:** Choice, Yes / No and Score questions about the same input are answered together, each from its own forward pass over the input, with a probability for every option.

## Quickstart

```bash
pip install "transformers==5.17.0" torch torchvision pillow opencv-python-headless safetensors accelerate
pip install flash-linear-attention  # optional: fast GPU kernels for the linear-attention layers
```

```python
import json

from huggingface_hub import hf_hub_download
from transformers import AutoModel

model = AutoModel.from_pretrained("vllm-sr/d3-edge", trust_remote_code=True)

# Text
result = model.system_one(
    state="The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
    questions={
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "returns": "Refunds, replacements and damaged deliveries",
                "billing": "Payments, invoices and charges",
                "technical": "Product setup and faults"
            }
        },
        "receipt": {
            "type": "noul",
            "instructions": "Does the customer have a receipt?"
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": [
                "Routine",
                "Soon",
                "Today"
            ]
        }
    },
)
print(json.dumps(result["answers"], indent=2))

# Text and an image (or several)
receipt = hf_hub_download("vllm-sr/d3-edge", "assets/example-receipt.png")
result = model.system_one(
    state="The customer says the blender arrived cracked and attached the receipt.",
    images=[receipt],  # local paths, http(s) URLs, PIL images or base64 data URLs
    questions={
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "returns": "Refunds, replacements and damaged deliveries",
                "billing": "Payments, invoices and charges",
                "technical": "Product setup and faults"
            }
        },
        "on_receipt": {
            "type": "noul",
            "instructions": "Does the receipt list the blender?"
        },
        "payment": {
            "type": "choice",
            "instructions": "How was the order paid?",
            "criteria": {
                "card": None,
                "cash": None,
                "gift card": None
            }
        }
    },
)
print(json.dumps(result["answers"], indent=2))

# Text and a video (or several)
clip = hf_hub_download("vllm-sr/d3-edge", "assets/example-video.mp4")
result = model.system_one(
    state="A short clip from a test camera.",
    videos=[clip],  # local paths, http(s) URLs, base64 data URLs or frame arrays
    questions={
        "direction": {
            "type": "choice",
            "instructions": "Which way does the square move?",
            "criteria": {
                "right": "From left to right",
                "left": "From right to left",
                "still": "It does not move"
            }
        },
        "color_change": {
            "type": "noul",
            "instructions": "Does the square change color?"
        }
    },
)
print(json.dumps(result["answers"], indent=2))

# Or as a pipeline:
# transformers.pipeline("decision", model="vllm-sr/d3-edge", trust_remote_code=True)(state=..., questions=..., images=..., videos=...)
```

Images go before the text of the request, each read at up to 1.6 megapixels; videos follow the images, read at 2 frames per second (at most 32 frames spread over the whole video, each at up to 0.2 megapixels). Every question of the request sees all of them.

## Run with vLLM Semantic Router

[vLLM Semantic Router](https://github.com/vllm-project/semantic-router) serves Decision 3.0 natively in Engine mode, which runs one model behind the System One API on your own machine. Its answers matched this package's own `d3_server.py` exactly, for text, images and videos, with one request per forward, the former default (measured on AMD Instinct MI325X); with batching, now the default, a few near-tie decisions change ([Serving performance](#serving-performance)).

```bash
curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --channel dev --no-launch
export PATH="$HOME/.local/bin:$PATH"
vllm-sr serve -e vllm-sr/d3-edge
```

The platform is detected automatically; choose one with `--platform auto|cpu|cuda|rocm` (default `auto`).

Engine mode and Decision 3.0 are in the development channel (`--channel dev`) until a stable vLLM Semantic Router release includes them. The first start pulls the image for your machine and downloads the model. Then ask questions about a request:

```bash
curl -s localhost:8899/v1/systemone -H 'content-type: application/json' -d '{
  "model": "vllm-sr/d3-edge",
  "state": "Write a Python function that merges two sorted lists.",
  "questions": {
    "kind": {"type": "choice", "instructions": "What kind of work is this?",
             "criteria": {"code": "Writing or fixing code", "math": "Mathematics", "chat": "Anything else"}},
    "reasoning": {"type": "noul", "instructions": "Does answering this need multi-step reasoning?"}
  }
}'
```

Send images as base64 data URLs (PNG, JPEG or WebP) in `images` and videos (MP4, WebM, QuickTime or Matroska) in `videos`; every question sees them:

```bash
curl -s localhost:8899/v1/systemone -H 'content-type: application/json' --data-binary @- <<EOF
{
  "model": "vllm-sr/d3-edge",
  "state": "What does the attached image show?",
  "images": ["data:image/png;base64,$(base64 -w0 chart.png)"],
  "questions": {
    "kind": {"type": "choice", "instructions": "What kind of image is this?",
             "criteria": {"chart": "A chart or plot", "photo": "A photograph", "document": "A document"}}
  }
}
EOF
```

```bash
curl -s localhost:8899/v1/systemone -H 'content-type: application/json' --data-binary @- <<EOF
{
  "model": "vllm-sr/d3-edge",
  "state": "What happens in the clip?",
  "videos": ["data:video/mp4;base64,$(base64 -w0 clip.mp4)"],
  "questions": {
    "moving": {"type": "noul", "instructions": "Does something move across the scene?"}
  }
}
EOF
```

`vllm-sr stop` stops it. See [Router and Engine modes](https://github.com/vllm-project/semantic-router/blob/main/src/vllm-sr/INSTANCE_MODES.md#ask-about-images-and-videos) for limits and options.

### Serving performance

**On one AMD Instinct MI325X, vLLM Semantic Router serves d3-edge at 861 requests/s with p99 under 1 s, 6.1× AutoModel** (one-question text, with batching, now the default), and with one request per forward, the former default, it gave AutoModel's answers byte for byte when both used the same kernel settings.

![Serving throughput of d3-edge on one AMD Instinct MI325X: AutoModel, the package server and vLLM Semantic Router Engine mode](assets/serving-throughput.png)

AutoModel runs in your own process, one call at a time (the Quickstart code); the package server is its `python d3_server.py`; Engine mode is `vllm-sr serve -e`, measured with one request per forward (*exact* in the chart, the former default) and with batching. In vLLM Semantic Router from main (its next development package and release), Engine mode batches concurrent requests by default and takes vLLM-style [engine arguments](https://github.com/vllm-project/semantic-router/blob/main/website/docs/model-runtime/engine-args.md): `--max-num-batched-tokens`, `--max-num-seqs`, `--batch-window-ms` and `--dtype`.

Where Engine mode helps, and what it costs:

- **Throughput:** 6.1× AutoModel's requests per second on one-question text, 5.0× on 8-question images and 2.4× on 10-second videos, with the better of the two Engine setups in each row.
- **Start-up:** the first answer comes 11.6 s after launch with empty kernel caches, against 24.9 s for AutoModel, whose first image then takes another 1.13 s, and 36.5 s for the package server.
- **Many questions and long inputs:** one request at a time, Engine mode with one request per forward answered 16 text questions in 21.4 ms (AutoModel: 22.4 ms), 8 questions about one image in 215 ms (267 ms) and a 32k-token text in 268 ms (281 ms); batching runs the 8 image questions together in 111 ms, 58 % less than AutoModel.
- **GPU memory:** 15.9 GiB per replica at peak with one request per forward, above the package server's 12.1 GiB, and 19.9 GiB with batching; AutoModel, serving one request at a time, peaked at 5.2 GiB. Engine mode keeps up to 2 GiB of image features cached.
- **Answers:** Engine mode with one request per forward returned answers byte-identical to AutoModel's on all 7 request types (text with 1, 4 or 16 questions, an image with 1 or 8, the video and a 32k-token text) when AutoModel used the same kernel settings: 140 Engine replies and 21 AutoModel calls compared. With the kernel autotuning AutoModel runs in each new process, 6 of the 7 were byte-identical and no decision changed (largest probability difference 0.009). Batching changed 0.151 % of decisions on the complete public suite (487 of 323,259 questions, near-ties), with gold accuracy 61.702 → 61.694 % and the index 28.82 → 28.82.

A single short request is fastest in your own process: AutoModel answers one text question 1.3 ms sooner than Engine mode and one new image 26 ms sooner, the cost of going through the router to the model worker and its media path. Engine mode's gains come from serving many requests at once.

<details>
<summary>Single-request latency and the numbers behind the chart</summary>

One request at a time, warm (p50 / p90, ms):

| Request | AutoModel | Package server | Engine, one request per forward | Engine, batching (now the default) |
| --- | ---: | ---: | ---: | ---: |
| Text, 1 question | 6.6 / 7.2 | 7.0 / 7.4 | 7.9 / 8.4 | 9.9 / 10.8 |
| Text, 16 questions | 22.4 / 24.1 | 22.1 / 22.4 | 21.4 / 22.1 | 20.6 / 21.0 |
| Image (1.6 MP), 1 question | 58.6 / 59.7 | 61.8 / 63.4 | 84.9 / 87.0 | 87.7 / 93.0 |
| Image (1.6 MP), 8 questions | 267 / 269 | 261 / 267 | 215 / 219 | 111 / 113 |
| Video (10 s, 1080p), 1 question | 274 / 279 | 256 / 265 | 294 / 309 | 309 / 324 |
| 32k-token text, 1 question | 281 / 285 | 285 / 308 | 268 / 269 | 274 / 282 |

Requests per second on one GPU, closed loop with up to 64 clients (p99 under 1 s, except *):

| | AutoModel † | Package server | Engine, one request per forward | Engine, batching (now the default) |
| --- | ---: | ---: | ---: | ---: |
| Text, 1 question | 142 | 137 | 127 | 861 |
| Image (1.6 MP), 8 questions | 3.7 | 3.9 | 6.2 | 18.8 |
| Video (10 s, 1080p), 1 question | 3.7 | 7.2 | 7.7 | 8.7 |
| Text, 1 question, saturation | 142 | 137 | 127 | 861 |

† Back-to-back calls in one process (1,000 / mean latency). * p99 over 1 s at every load; the highest rate is shown.

</details>

<sub>Measured on 2026-10-13, one AMD Instinct MI325X per replica and each side alone on the GPU: ROCm 7.2 with PyTorch 2.12; AutoModel and the package server with Transformers 5.17.0; vLLM Semantic Router Engine mode built from source, with serving optimizations that are newer than its published images. Requests: a support ticket of about 330 tokens with 1, 4 or 16 questions; a 1,456 × 1,092 PNG (1.6 MP) with 1 or 8 questions; a 10-second 1080p MP4, read at 2 frames per second; a 32k-token text; every image and video request carries new bytes. One request at a time: 3 warm-up calls, then up to 20 timed calls. Under load: 1, 2, 4, … 64 closed-loop clients, each level warmed up and then counted for 15 to 60 s; Engine mode admits 32 requests in flight per worker and answers the rest at once with 503. Start-up: model files on local disk; Engine mode from the start of its router process, without the CLI's image pull.</sub>

## Evaluation

### Text: Jev Decision Index 0.3.2

| Model | Jev Decision Index ↑ | Public ↑ | Same-skill tests ↑ | New-domain tasks ↑ |
| --- | ---: | ---: | ---: | ---: |
| **d3-edge** | **20.2** | **28.8** | **24.3** | **12.9** |
| jiwo 0.8B v2 | 25.6 | 29.4 | 30.6 | 17.9 |
| Decision 2.0 (0.6B) | 13.6 | 16.7 | 16.9 | 11.8 |
| vera-spark | 11.5 | 19.6 | 15.1 | 7.3 |
| GLiNER2.5-Decide | 9.2 | 11.5 | 12.1 | 9.6 |
| LiquidAI d1-omni-600M | 9.1 | 17.9 | 12.9 | 5.1 |

d3-edge is not on the official text board's Pareto frontier: Crivo-150M scores 22.2 with 0.15 billion parameters. Among the eight entries within 0.75 to 1.33 times its size, d3-edge ranks second, behind jiwo 0.8B v2 at 25.6. The other five Decision 3.0 models are on the frontier.

![Jev Decision Index against model size](assets/index-pareto.png)

![Jev Decision Index by area: d3-edge and Decision 2.0 (0.6B)](assets/index-areas.png)

### Images: Jev Decision Index vision board 0.3.2

| Model | Vision Index ↑ | Public ↑ | Private ↑ |
| --- | ---: | ---: | ---: |
| **d3-edge** | **31.7** | **36.6** | **26.8** |
| LiquidAI d1-omni-600M ‡ | — | 4.8 | — |

‡ LiquidAI d1-omni-600M is not on the official vision board; its public score is our run of the official 0.3.2 vision kit (public part) with the model's own `system_one`, on AMD Instinct MI325X, 2026-10-13; d3-edge measured the same way scores 36.7 (board: 36.6).

On the official vision board, the six Decision 3.0 models form the whole Pareto frontier: no other entry is on it. d3-edge is the smallest entry on the board.

d3-edge on the public vision benchmarks († approximate rebuild):

| Benchmark | d3-edge |
| --- | ---: |
| CV-Bench | 54.2 |
| BLINK | 35.1 |
| RealWorldQA | 24.8 |
| CharXiv † | 40.9 |
| InfographicVQA † | 60.7 |
| Mind2Web † | 41.2 |
| Winoground | 29.0 |
| KIE (CORD+FUNSD) † | 86.4 |
| Moderation (Hateful Memes) | 6.8 |
| R-Bench-M | 10.3 |
| MMMU-Pro vision | 4.9 |

### Videos: Perception Test

| Perception Test (validation) | d3-edge |
| --- | ---: |
| **All 19,140 questions** | **46.6** |
| Memory | 40.6 |
| Abstraction | 36.2 |
| Physics | 38.3 |
| Semantics | 67.3 |

Multiple-choice video question answering (three options per question, chance 33.3): top-1 accuracy (%), videos read with the defaults above.

<sub>Jev Decision Index: official leaderboard, text and vision boards 0.3.2, 2026-10-12; the boards evaluated v3.1.0, which has the same weights as this version. Size bands and the Pareto frontier use the parameter counts listed by the boards. d3-edge on the public vision benchmarks and the Perception Test: internal evaluation.</sub>

## License

Apache-2.0 ([LICENSE](LICENSE)). Built on [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Apache-2.0).

## Citation

```bibtex
@misc{d3_edge_2026,
  title        = {{d3-edge}: A Multimodal Foundation Decision Model},
  author       = {{vLLM Semantic Router Team}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/vllm-sr/d3-edge}}
}
```

Trained on AMD Instinct MI325X GPUs.
