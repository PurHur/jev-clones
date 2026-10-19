---
pipeline_tag: zero-shot-classification
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
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

![d3: Decision 3.0](assets/banner.png)

# d3

**d3** is the 27B multimodal foundation decision model of Decision 3.0, the decision models of [vLLM Semantic Router](https://github.com/vllm-project/semantic-router). Give it an input (text or JSON, optionally with images and videos) and the questions you need answered: pick one of several options, say yes or no, or rate on a scale. It answers them all in one call and returns a probability for every answer, without generating text.

**Try it free** at [decision.vllm-sr.ai](https://decision.vllm-sr.ai) · Official Space: [vllm-sr/Decision-3.0](https://huggingface.co/spaces/vllm-sr/Decision-3.0)

| | |
| --- | --- |
| **Parameters** | 26.09B, including the 0.46B vision encoder |
| **Inputs** | Text or JSON, plus images and videos (several per request) |
| **Decision types** | Choice · Yes / No · Score |
| **License** | Apache-2.0 |

## Highlights

- **Jev Decision Index 0.3.2: 66.3, first on the official text board** and on its Pareto frontier; public suite 65.45, measured with the official 0.3 kit on the released weights: all 140,178 public requests answered, none unsupported.
- **+10.4 over Decision 2.0** (its 27B model: 55.9 on the official board): +8.5 on the public suite, ahead in all five of its areas, and +7.1 and +12.6 on the private same-skill tests and new-domain tasks.
- **Reads images:** multiple images per request (PNG, JPEG or WebP), given as paths, URLs, PIL images or base64 data URLs; every question of the request sees all of them.
- **Reads videos:** multiple videos per request (MP4, WebM, MOV or MKV), given as paths, URLs, base64 data URLs or frame arrays, read at 2 frames per second; **Perception Test (multiple-choice video QA): 77.4** (internal evaluation).
- **Speed:** text requests take a median of 54.4 ms, requests with an image a median of 290 ms, and requests with a 10-second video a median of 500 ms, on one AMD Instinct MI325X, one request at a time with AutoModel.
- **Serving:** 27 requests/s with p99 under 1 s on one AMD Instinct MI325X with vLLM Semantic Router, 1.5× AutoModel (one-question text, with batching, now the default); with one request per forward, the former default, it gave AutoModel's answers byte for byte when both used the same kernel settings ([Serving performance](#serving-performance)).
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

model = AutoModel.from_pretrained("vllm-sr/d3", trust_remote_code=True)

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
receipt = hf_hub_download("vllm-sr/d3", "assets/example-receipt.png")
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
clip = hf_hub_download("vllm-sr/d3", "assets/example-video.mp4")
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
# transformers.pipeline("decision", model="vllm-sr/d3", trust_remote_code=True)(state=..., questions=..., images=..., videos=...)
```

Images go before the text of the request, each read at up to 1.6 megapixels; videos follow the images, read at 2 frames per second (at most 32 frames spread over the whole video, each at up to 0.2 megapixels). Every question of the request sees all of them.

## Run with vLLM

[vLLM](https://github.com/vllm-project/vllm) serves d3 natively: `vllm serve` loads this repository as it is, with no conversion and no extra flags, and answers on vLLM's System One endpoint, `/v1/systemone`. The support is on vLLM's main branch since 2026-10-14 ([vllm-project/vllm#61565](https://github.com/vllm-project/vllm/pull/61565), commit [`89cab55`](https://github.com/vllm-project/vllm/commit/89cab550862e4b7f08d70c19682121362d7dd0c5)), and no vLLM release includes it yet (the latest is v0.31.0). Until one does, install a current nightly build with [uv](https://docs.astral.sh/uv/); vLLM's nightly indexes do not work with plain `pip`:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

# NVIDIA GPUs (CUDA)
uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

# AMD GPUs (ROCm 7.2; glibc 2.39 or newer, as on Ubuntu 24.04)
sudo apt install -y libopenmpi3t64 libdw1t64
export VLLM_ROCM_VARIANT=$(curl -s https://wheels.vllm.ai/rocm/nightly | grep -oP 'rocm\d+' | head -1 | sed 's/%2B/+/g')
uv pip install --pre vllm --extra-index-url https://wheels.vllm.ai/rocm/nightly/${VLLM_ROCM_VARIANT} --index-strategy unsafe-best-match
```

The ROCm wheels load OpenMPI and libdw at import, hence the `apt` line (Ubuntu 24.04 package names). Start the server; the first start downloads the model. Then ask questions about a request:

```bash
vllm serve vllm-sr/d3
```

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "model": "vllm-sr/d3",
  "state": "Write a Python function that merges two sorted lists.",
  "questions": {
    "kind": {"type": "choice", "instructions": "What kind of work is this?",
             "criteria": {"code": "Writing or fixing code", "math": "Mathematics", "chat": "Anything else"}},
    "reasoning": {"type": "noul", "instructions": "Does answering this need multi-step reasoning?"}
  }
}'
```

The reply gives each question its pick and each option a probability (d3 on one AMD Instinct MI300X; `id`, `created` and `diagnostics` left out, probabilities rounded):

```json
{
  "object": "structured_decision",
  "model": "vllm-sr/d3",
  "answers": {
    "kind": {"type": "choice", "choice": "code", "probabilities": {"code": 0.9996, "math": 0.0002, "chat": 0.0002}, "confidence": 0.9996},
    "reasoning": {"type": "noul", "noul": 0.5808, "probabilities": {"yes": 0.5808, "no": 0.4192}, "confidence": 0.5808}
  },
  "usage": {"input_tokens": 213, "output_tokens": 0}
}
```

The probabilities are this package's, as `model.system_one()` gives them, and `confidence` is the chosen option's probability; a yes / no answer's `noul` is the probability of yes. The image and video requests of the next section work unchanged on port 8000: vLLM places the media before the text and reads them with this package's budgets. A top-level `instructions` field returns 400, since the prompt is fixed. vLLM reserves 92 % of the GPU's memory by default; lower `--gpu-memory-utilization` to share the GPU.

**Verified on AMD Instinct MI300X (ROCm).** These commands were run as written in a new Ubuntu 24.04 environment with ROCm 7.2 and the ROCm nightly of 2026-10-15 (`0.32.0rc1.dev190+g3f1bcfd82`). Against this package's runtime (AutoModel with the versions in `requirements.txt`), vLLM made the same decisions on the request above and on the image and video requests of the next section, and on 415 of 416 further questions (text, JSON, a 40-option choice, an image and a video); the other one is a near-tie whose two most likely answers are 0.002 apart. The median probability difference was 0.0025. The CUDA command installs a nightly that includes d3, but it has not been run on an NVIDIA GPU yet.

In the measurements of the vLLM pull request, on one AMD Instinct MI325X, vLLM changed none of the 416 decisions and served 1.4× and 1.8× the requests per second of this package's own server (`d3_server.py`) at 8 and 64 concurrent clients; one request at a time it was 3 % slower. It starts more slowly than AutoModel: 215 s to the first answer with empty caches, against 40 s; `--max-model-len 32768` brings that to 139 s.

Request fields and limits: [Decision 3.0 in the vLLM docs](https://docs.vllm.ai/en/latest/models/structured_decisions_models/#decision-30).

## Run with vLLM Semantic Router

[vLLM Semantic Router](https://github.com/vllm-project/semantic-router) serves Decision 3.0 natively in Engine mode, which runs one model behind the System One API on your own machine. Its answers matched this package's own `d3_server.py` exactly, for text, images and videos, with one request per forward, the former default (measured on AMD Instinct MI325X); with batching, now the default, a few near-tie decisions change ([Serving performance](#serving-performance)).

```bash
curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --channel dev --no-launch
export PATH="$HOME/.local/bin:$PATH"
vllm-sr serve -e vllm-sr/d3
```

The platform is detected automatically; choose one with `--platform auto|cpu|cuda|rocm` (default `auto`).

Engine mode and Decision 3.0 are in the development channel (`--channel dev`) until a stable vLLM Semantic Router release includes them. The first start pulls the image for your machine and downloads the model. Then ask questions about a request:

```bash
curl -s localhost:8899/v1/systemone -H 'content-type: application/json' -d '{
  "model": "vllm-sr/d3",
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
  "model": "vllm-sr/d3",
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
  "model": "vllm-sr/d3",
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

**On one AMD Instinct MI325X, vLLM Semantic Router serves d3 at 27 requests/s with p99 under 1 s, 1.5× AutoModel** (one-question text, with batching, now the default), and with one request per forward, the former default, it gave AutoModel's answers byte for byte when both used the same kernel settings.

![Serving throughput of d3 on one AMD Instinct MI325X: AutoModel, the package server and vLLM Semantic Router Engine mode](assets/serving-throughput.png)

AutoModel runs in your own process, one call at a time (the Quickstart code); the package server is its `python d3_server.py`; Engine mode is `vllm-sr serve -e`, measured with one request per forward (*exact* in the chart, the former default) and with batching. In vLLM Semantic Router from main (its next development package and release), Engine mode batches concurrent requests by default and takes vLLM-style [engine arguments](https://github.com/vllm-project/semantic-router/blob/main/website/docs/model-runtime/engine-args.md): `--max-num-batched-tokens`, `--max-num-seqs`, `--batch-window-ms` and `--dtype`.

Where Engine mode helps, and what it costs:

- **Throughput:** 1.5× AutoModel's requests per second on one-question text, 1.3× on 8-question images and 1.8× on 10-second videos, with the better of the two Engine setups in each row (8-question images: highest rates, since no side keeps p99 under 1 s there).
- **Start-up:** the first answer comes 19.4 s after launch (one request per forward; 20.4 s with batching) with empty kernel caches, against 38.4 s for AutoModel, whose first image then takes another 6.20 s, and 80.8 s for the package server.
- **Many questions and long inputs:** one request at a time, Engine mode with one request per forward answered 16 text questions in 548 ms (AutoModel: 572 ms), 8 questions about one image in 1.87 s (1.95 s) and a 32k-token text in 4.21 s (4.23 s); batching runs the 8 image questions together in 1.56 s, 20 % less than AutoModel.
- **GPU memory:** 65.4 GiB per replica at peak with one request per forward, about the same as the package server's 65.2 GiB, and 62.1 GiB with batching; AutoModel, serving one request at a time, peaked at 61.8 GiB. Engine mode keeps up to 2 GiB of image features cached.
- **Answers:** Engine mode with one request per forward returned answers byte-identical to AutoModel's on all 7 request types (text with 1, 4 or 16 questions, an image with 1 or 8, the video and a 32k-token text) when AutoModel used the same kernel settings: 135 Engine replies and 21 AutoModel calls compared. With the kernel autotuning AutoModel runs in each new process, 5 of the 7 were byte-identical and no decision changed (largest probability difference 0.005). Batching changed 0.048 % of decisions on a 4,000-request sample of the public suite (14 of 29,339 questions, near-ties), with gold accuracy 92.109 → 92.126 %.

A single short request is fastest in your own process: AutoModel answers one text question 0.9 ms sooner than Engine mode and one new image 22 ms sooner, the cost of going through the router to the model worker and its media path. Engine mode's gains come from serving many requests at once.

<details>
<summary>Single-request latency and the numbers behind the chart</summary>

One request at a time, warm (p50 / p90, ms):

| Request | AutoModel | Package server | Engine, one request per forward | Engine, batching (now the default) |
| --- | ---: | ---: | ---: | ---: |
| Text, 1 question | 54.4 / 54.6 | 55.0 / 55.3 | 55.3 / 55.6 | 57.2 / 57.9 |
| Text, 16 questions | 572 / 574 | 571 / 572 | 548 / 549 | 548 / 549 |
| Image (1.6 MP), 1 question | 290 / 292 | 295 / 297 | 312 / 314 | 315 / 316 |
| Image (1.6 MP), 8 questions | 1.95 s / 1.95 s | 1.95 s / 1.97 s | 1.87 s / 1.88 s | 1.56 s / 1.57 s |
| Video (10 s, 1080p), 1 question | 500 / 518 | 509 / 523 | 551 / 579 | 533 / 548 |
| 32k-token text, 1 question | 4.23 s / 4.24 s | 4.30 s / 4.30 s | 4.21 s / 4.23 s | 4.22 s / 4.23 s |

Requests per second on one GPU, closed loop with up to 64 clients (p99 under 1 s, except *):

| | AutoModel † | Package server | Engine, one request per forward | Engine, batching (now the default) |
| --- | ---: | ---: | ---: | ---: |
| Text, 1 question | 18.4 | 17.7 | 18.1 | 27 |
| Image (1.6 MP), 8 questions | 0.51 * | 0.53 * | 0.53 * | 0.67 * |
| Video (10 s, 1080p), 1 question | 2.0 | 2.8 | 3.7 | 3.7 |
| Text, 1 question, saturation | 18.4 | 17.7 | 18.1 | 29 |

† Back-to-back calls in one process (1,000 / mean latency). * p99 over 1 s at every load; the highest rate is shown.

</details>

<sub>Measured on 2026-10-13, one AMD Instinct MI325X per replica and each side alone on the GPU: ROCm 7.2 with PyTorch 2.12; AutoModel and the package server with Transformers 5.17.0; vLLM Semantic Router Engine mode built from source, with serving optimizations that are newer than its published images. Requests: a support ticket of about 330 tokens with 1, 4 or 16 questions; a 1,456 × 1,092 PNG (1.6 MP) with 1 or 8 questions; a 10-second 1080p MP4, read at 2 frames per second; a 32k-token text; every image and video request carries new bytes. One request at a time: 3 warm-up calls, then up to 20 timed calls. Under load: 1, 2, 4, … 64 closed-loop clients, each level warmed up and then counted for 15 to 60 s; Engine mode admits 32 requests in flight per worker and answers the rest at once with 503. Start-up: model files on local disk; Engine mode from the start of its router process, without the CLI's image pull.</sub>

## Evaluation

### Text: Jev Decision Index 0.3.2

| Model | Jev Decision Index ↑ | Public ↑ | Same-skill tests ↑ | New-domain tasks ↑ |
| --- | ---: | ---: | ---: | ---: |
| **d3** | **66.3** | **65.5** | **62.8** | **60.5** |
| Decider 31B | 64.6 | 64.3 | 61.8 | 58.3 |
| Perplexity Decider v1.1 (27B) | 62.8 | 62.3 | 61.1 | 55.6 |
| Darwin-27B-ZTC | 62.5 | 62.6 | 58.7 | 57.7 |
| AnyJev L0 · Gemma-4-31B | 60.2 | 59.8 | 59.1 | 53.0 |
| Decision 2.0 (27B) | 55.9 | 57.0 | 55.7 | 47.9 |

d3 ranks first of the 149 entries on the official text board and is on its Pareto frontier: no entry with as many parameters or fewer scores higher. Five of the six Decision 3.0 models are on that frontier; d3-edge, second among the entries of its size, is not.

![Jev Decision Index against model size](assets/index-pareto.png)

![Jev Decision Index by area: d3 and Decision 2.0 (27B)](assets/index-areas.png)

### Images: Jev Decision Index vision board 0.3.2

| Model | Vision Index ↑ | Public ↑ | Private ↑ |
| --- | ---: | ---: | ---: |
| **d3** | **71.3** | **74.1** | **68.5** |
| Perplexity Decider v1.1 (27B) | 70.6 | 73.2 | 67.9 |
| JEV-27B-VL | 69.6 | 72.8 | 66.4 |
| Solomon v1.1 (27B) | 66.8 | 69.9 | 63.8 |

On the official vision board, the six Decision 3.0 models form the whole Pareto frontier: no other entry is on it. d3 ranks first of the 33 entries.

d3 on the public vision benchmarks († approximate rebuild):

| Benchmark | d3 |
| --- | ---: |
| CV-Bench | 76.2 |
| BLINK | 59.9 |
| RealWorldQA | 73.4 |
| CharXiv † | 91.7 |
| InfographicVQA † | 96.7 |
| Mind2Web † | 90.0 |
| Winoground | 85.8 |
| KIE (CORD+FUNSD) † | 97.7 |
| Moderation (Hateful Memes) | 43.9 |
| R-Bench-M | 33.0 |
| MMMU-Pro vision | 47.0 |

### Videos: Perception Test

| Perception Test (validation) | d3 |
| --- | ---: |
| **All 19,140 questions** | **77.4** |
| Memory | 80.3 |
| Abstraction | 71.2 |
| Physics | 72.5 |
| Semantics | 88.8 |

Multiple-choice video question answering (three options per question, chance 33.3): top-1 accuracy (%), videos read with the defaults above.

<sub>Jev Decision Index: official leaderboard, text and vision boards 0.3.2, 2026-10-12; the boards evaluated v3.1.0, which has the same weights as this version. Size bands and the Pareto frontier use the parameter counts listed by the boards. d3 on the public vision benchmarks and the Perception Test: internal evaluation.</sub>

## License

Apache-2.0 ([LICENSE](LICENSE)). Built on [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0).

## Citation

```bibtex
@misc{d3_2026,
  title        = {{d3}: A Multimodal Foundation Decision Model},
  author       = {{vLLM Semantic Router Team}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/vllm-sr/d3}}
}
```

Trained on AMD Instinct MI325X GPUs.
