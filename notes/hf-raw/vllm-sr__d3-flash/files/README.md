---
pipeline_tag: zero-shot-classification
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
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

![d3-flash: Decision 3.0](assets/banner.png)

# d3-flash

**d3-flash** is the 9B multimodal foundation decision model of Decision 3.0, the decision models of [vLLM Semantic Router](https://github.com/vllm-project/semantic-router). Give it an input (text or JSON, optionally with images and videos) and the questions you need answered: pick one of several options, say yes or no, or rate on a scale. It answers them all in one call and returns a probability for every answer, without generating text.

**Try it free** at [decision.vllm-sr.ai](https://decision.vllm-sr.ai) · Official Space: [vllm-sr/Decision-3.0](https://huggingface.co/spaces/vllm-sr/Decision-3.0)

| | |
| --- | --- |
| **Parameters** | 8.39B, including the 0.46B vision encoder |
| **Inputs** | Text or JSON, plus images and videos (several per request) |
| **Decision types** | Choice · Yes / No · Score |
| **License** | Apache-2.0 |

## Highlights

- **Jev Decision Index 0.3.2: 55.8 on the official text board**, first among the entries of similar size and on the board's Pareto frontier; public suite 57.79, measured with the official 0.3 kit on the released weights: all 140,178 public requests answered, none unsupported.
- **+7.6 over Decision 2.0** (its 9B model: 48.2 on the official board): +11.0 on the public suite, ahead in all five of its areas, and +7.3 and +4.2 on the private same-skill tests and new-domain tasks.
- **Reads images:** multiple images per request (PNG, JPEG or WebP), given as paths, URLs, PIL images or base64 data URLs; every question of the request sees all of them.
- **Reads videos:** multiple videos per request (MP4, WebM, MOV or MKV), given as paths, URLs, base64 data URLs or frame arrays, read at 2 frames per second; **Perception Test (multiple-choice video QA): 73.3** (internal evaluation).
- **Speed:** a median of 19.2 ms for a text request, 166 ms for a request with an image and 371 ms for a request with a 10-second video, on one AMD Instinct MI325X GPU, one request at a time with AutoModel.
- **Serving:** 95 requests/s with p99 under 1 s on one AMD Instinct MI325X with vLLM Semantic Router, 1.8× AutoModel (one-question text, with batching, now the default); with one request per forward, the former default, it gave answers identical to AutoModel's ([Serving performance](#serving-performance)).
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

model = AutoModel.from_pretrained("vllm-sr/d3-flash", trust_remote_code=True)

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
receipt = hf_hub_download("vllm-sr/d3-flash", "assets/example-receipt.png")
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
clip = hf_hub_download("vllm-sr/d3-flash", "assets/example-video.mp4")
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
# transformers.pipeline("decision", model="vllm-sr/d3-flash", trust_remote_code=True)(state=..., questions=..., images=..., videos=...)
```

Images go before the text of the request, each read at up to 1.6 megapixels; videos follow the images, read at 2 frames per second (at most 32 frames spread over the whole video, each at up to 0.2 megapixels). Every question of the request sees all of them.

## Run with vLLM Semantic Router

[vLLM Semantic Router](https://github.com/vllm-project/semantic-router) serves Decision 3.0 natively in Engine mode, which runs one model behind the System One API on your own machine. Its answers matched this package's own `d3_server.py` exactly, for text, images and videos, with one request per forward, the former default (measured on AMD Instinct MI325X); with batching, now the default, a few near-tie decisions change ([Serving performance](#serving-performance)).

```bash
curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --channel dev --no-launch
export PATH="$HOME/.local/bin:$PATH"
vllm-sr serve -e vllm-sr/d3-flash
```

The platform is detected automatically; choose one with `--platform auto|cpu|cuda|rocm` (default `auto`).

Engine mode and Decision 3.0 are in the development channel (`--channel dev`) until a stable vLLM Semantic Router release includes them. The first start pulls the image for your machine and downloads the model. Then ask questions about a request:

```bash
curl -s localhost:8899/v1/systemone -H 'content-type: application/json' -d '{
  "model": "vllm-sr/d3-flash",
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
  "model": "vllm-sr/d3-flash",
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
  "model": "vllm-sr/d3-flash",
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

**On one AMD Instinct MI325X, vLLM Semantic Router serves d3-flash at 95 requests/s with p99 under 1 s, 1.8× AutoModel** (one-question text, with batching, now the default), and with one request per forward, the former default, it gave answers identical to AutoModel's.

![Serving throughput of d3-flash on one AMD Instinct MI325X: AutoModel, the package server and vLLM Semantic Router Engine mode](assets/serving-throughput.png)

AutoModel runs in your own process, one call at a time (the Quickstart code); the package server is its `python d3_server.py`; Engine mode is `vllm-sr serve -e`, measured with one request per forward (*exact* in the chart, the former default) and with batching. In vLLM Semantic Router from main (its next development package and release), Engine mode batches concurrent requests by default and takes vLLM-style [engine arguments](https://github.com/vllm-project/semantic-router/blob/main/website/docs/model-runtime/engine-args.md): `--max-num-batched-tokens`, `--max-num-seqs`, `--batch-window-ms` and `--dtype`.

Where Engine mode helps, and what it costs:

- **Throughput:** 1.8× AutoModel's requests per second on one-question text, 2.2× on 8-question images and 2.7× on 10-second videos, with the better of the two Engine setups in each row (8-question images: only batching keeps p99 under 1 s there).
- **Start-up:** the first answer comes 15.1 s after launch (one request per forward; 15.4 s with batching) with empty kernel caches, against 28.6 s for AutoModel, whose first image then takes another 1.24 s, and 53.7 s for the package server.
- **Many questions and long inputs:** one request at a time, Engine mode with one request per forward answered 16 text questions in 172 ms (AutoModel: 176 ms), 8 questions about one image in 1.01 s (1.09 s) and a 32k-token text in 1.36 s (1.39 s); batching runs the 8 image questions together in 533 ms, 51 % less than AutoModel.
- **GPU memory:** 29.1 GiB per replica at peak with one request per forward, below the package server's 30.4 GiB, and 27.0 GiB with batching; AutoModel, serving one request at a time, peaked at 23.8 GiB. Engine mode keeps up to 2 GiB of image features cached.
- **Answers:** Engine mode with one request per forward returned answers byte-identical to AutoModel's on all 7 request types (text with 1, 4 or 16 questions, an image with 1 or 8, the video and a 32k-token text) when AutoModel used the same kernel settings: 140 Engine replies and 21 AutoModel calls compared. With the kernel autotuning AutoModel runs in each new process, all 7 were byte-identical too. Batching changed 0.075 % of decisions on a 4,000-request sample of the public suite (22 of 29,339 questions, near-ties), with gold accuracy 91.565 → 91.586 %.

A single short request is fastest in your own process: AutoModel answers one text question 3.4 ms sooner than Engine mode and one new image 26 ms sooner, the cost of going through the router to the model worker and its media path. Engine mode's gains come from serving many requests at once.

<details>
<summary>Single-request latency and the numbers behind the chart</summary>

One request at a time, warm (p50 / p90, ms):

| Request | AutoModel | Package server | Engine, one request per forward | Engine, batching (now the default) |
| --- | ---: | ---: | ---: | ---: |
| Text, 1 question | 19.2 / 19.4 | 20.0 / 20.8 | 22.7 / 24.2 | 26.7 / 29.8 |
| Text, 16 questions | 176 / 178 | 178 / 179 | 172 / 173 | 171 / 172 |
| Image (1.6 MP), 1 question | 166 / 167 | 173 / 175 | 192 / 195 | 200 / 209 |
| Image (1.6 MP), 8 questions | 1.09 s / 1.11 s | 1.08 s / 1.09 s | 1.01 s / 1.02 s | 533 / 536 |
| Video (10 s, 1080p), 1 question | 371 / 375 | 372 / 389 | 371 / 381 | 394 / 406 |
| 32k-token text, 1 question | 1.39 s / 1.40 s | 1.38 s / 1.39 s | 1.36 s / 1.37 s | 1.36 s / 1.36 s |

Requests per second on one GPU, closed loop with up to 64 clients (p99 under 1 s, except *):

| | AutoModel † | Package server | Engine, one request per forward | Engine, batching (now the default) |
| --- | ---: | ---: | ---: | ---: |
| Text, 1 question | 52 | 50 | 47 | 95 |
| Image (1.6 MP), 8 questions | 0.92 * | 0.93 * | 1.0 * | 2.1 |
| Video (10 s, 1080p), 1 question | 2.7 | 4.5 | 7.3 | 6.9 |
| Text, 1 question, saturation | 52 | 50 | 47 | 95 |

† Back-to-back calls in one process (1,000 / mean latency). * p99 over 1 s at every load; the highest rate is shown.

</details>

<sub>Measured on 2026-10-13, one AMD Instinct MI325X per replica and each side alone on the GPU: ROCm 7.2 with PyTorch 2.12; AutoModel and the package server with Transformers 5.17.0; vLLM Semantic Router Engine mode built from source, with serving optimizations that are newer than its published images. Requests: a support ticket of about 330 tokens with 1, 4 or 16 questions; a 1,456 × 1,092 PNG (1.6 MP) with 1 or 8 questions; a 10-second 1080p MP4, read at 2 frames per second; a 32k-token text; every image and video request carries new bytes. One request at a time: 3 warm-up calls, then up to 20 timed calls. Under load: 1, 2, 4, … 64 closed-loop clients, each level warmed up and then counted for 15 to 60 s; Engine mode admits 32 requests in flight per worker and answers the rest at once with 503. Start-up: model files on local disk; Engine mode from the start of its router process, without the CLI's image pull.</sub>

## Evaluation

### Text: Jev Decision Index 0.3.2

| Model | Jev Decision Index ↑ | Public ↑ | Same-skill tests ↑ | New-domain tasks ↑ |
| --- | ---: | ---: | ---: | ---: |
| **d3-flash** | **55.8** | **57.8** | **54.6** | **48.6** |
| Bespoke Nimble 9B v3 | 54.7 | 57.2 | 52.3 | 49.1 |
| Sieve 9B Plus | 50.9 | 53.4 | 49.3 | 45.4 |
| Nace.AI Drex v1.5 | 49.7 | 58.1 | 49.7 | 39.7 |
| JevK5 9B v0.3.3 | 48.4 | 47.6 | 48.2 | 43.3 |
| Decision 2.0 (9B) | 48.2 | 46.8 | 47.4 | 44.5 |

On the official text board, d3-flash is on the Pareto frontier, where no entry with as many parameters or fewer scores higher, and ranks first among the 18 entries within 0.75 to 1.33 times its size. Five of the six Decision 3.0 models are on that frontier; d3-edge, second among the entries of its size, is not.

![Jev Decision Index against model size](assets/index-pareto.png)

![Jev Decision Index by area: d3-flash and Decision 2.0 (9B)](assets/index-areas.png)

### Images: Jev Decision Index vision board 0.3.2

| Model | Vision Index ↑ | Public ↑ | Private ↑ |
| --- | ---: | ---: | ---: |
| **d3-flash** | **66.9** | **69.9** | **63.9** |
| JPT-9B | 61.7 | 65.4 | 58.0 |
| Winnow-E4B | 42.1 | 48.1 | 36.1 |

On the official vision board, the six Decision 3.0 models form the whole Pareto frontier: no other entry is on it. d3-flash ranks first among the three entries within 0.75 to 1.33 times its size, and fourth of 33 overall.

d3-flash on the public vision benchmarks († approximate rebuild):

| Benchmark | d3-flash |
| --- | ---: |
| CV-Bench | 75.2 |
| BLINK | 55.5 |
| RealWorldQA | 63.4 |
| CharXiv † | 83.0 |
| InfographicVQA † | 95.1 |
| Mind2Web † | 85.1 |
| Winoground | 83.3 |
| KIE (CORD+FUNSD) † | 96.3 |
| Moderation (Hateful Memes) | 41.3 |
| R-Bench-M | 20.8 |
| MMMU-Pro vision | 38.7 |

### Videos: Perception Test

| Perception Test (validation) | d3-flash |
| --- | ---: |
| **All 19,140 questions** | **73.3** |
| Memory | 75.5 |
| Abstraction | 68.7 |
| Physics | 66.1 |
| Semantics | 86.1 |

Multiple-choice video question answering (three options per question, chance 33.3): top-1 accuracy (%), videos read with the defaults above.

<sub>Jev Decision Index: official leaderboard, text and vision boards 0.3.2, 2026-10-12; the boards evaluated v3.1.0, which has the same weights as this version. Size bands and the Pareto frontier use the parameter counts listed by the boards. d3-flash on the public vision benchmarks and the Perception Test: internal evaluation.</sub>

## License

Apache-2.0 ([LICENSE](LICENSE)). Built on [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (Apache-2.0).

## Citation

```bibtex
@misc{d3_flash_2026,
  title        = {{d3-flash}: A Multimodal Foundation Decision Model},
  author       = {{vLLM Semantic Router Team}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/vllm-sr/d3-flash}}
}
```

Trained on AMD Instinct MI325X GPUs.
