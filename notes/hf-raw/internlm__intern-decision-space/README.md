---
title: Intern-Decision
emoji: 🎯
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
startup_duration_timeout: 1h
license: apache-2.0
models:
- InternLM/Intern-Decision-0.8B
short_description: Multimodal decisions with calibrated probabilities
---

# Intern-Decision

[GitHub](https://github.com/internlm/Intern-Decision) · [Model weights](https://huggingface.co/InternLM/Intern-Decision-0.8B)

A standalone Hugging Face Docker Space for **Intern-Decision-0.8B**, fine-tuned
from the Qwen3.5-0.8B instruction-tuned/chat checkpoint. Enter a state, define
`choice`, `score`, and `noul` questions, and optionally upload images. The demo
returns all local decisions and calibrated probabilities in a single forward.
The browser interface includes editable examples, probability bars, timing,
JSON responses, image previews, and session history.

## Deploy to Hugging Face Spaces

1. Create a Space and select the **Docker** SDK with a blank template.
2. Upload the contents of this folder to the Space repository root, including
   the Dockerfile, README, Python files, and `static/` directory.
3. Select **CPU** hardware in Space Settings. The demo defaults to `DEVICE=cpu`.
   Long or image-heavy requests may require more RAM than the free CPU tier.
4. Make `InternLM/Intern-Decision-0.8B` available on the Hub before starting the
   Space. For a private or gated repository, add a read-authorized `HF_TOKEN`
   as a **Space secret**. Public model downloads need no token.

The container listens on `0.0.0.0:7860`. At startup it downloads only model
weights and processor/tokenizer assets into the Hugging Face cache, then loads
the model once with native Transformers. It does not execute the model
repository's Python files. Startup waits for the checkpoint to load before
serving the application; subsequent requests reuse it.

No model weights, datasets, training code, or original project checkout are
needed in this Space repository. This Docker Space defaults to CPU;
this is not a ZeroGPU integration. CPU inference is slower than GPU inference,
and the free CPU tier may
run out of RAM for long or image-heavy requests. The first download requires roughly 2 GB for the model assets,
in addition to the container dependencies. Ephemeral caches may be downloaded
again when the Space is rebuilt or restarted on a new host.

## Settings

Set variables and secrets through the Space Settings page; never commit keys.

| Setting | Default | Purpose |
|---|---|---|
| `MODEL_REVISION` | `main` | Hub revision; set a commit SHA after uploading the model for a pinned deployment |
| `HF_TOKEN` (secret) | unset | Read access for private/gated model downloads |
| `MAX_INPUT_LENGTH` | `8192` | Token limit, including image tokens; longer inputs are rejected |
| `ATTN_IMPLEMENTATION` | `sdpa` | Native HF attention implementation: `sdpa` or `eager` |
| `MODEL_DTYPE` | `bfloat16` | `bfloat16`, `float16`, or `float32`; default is the tested GPU setting |
| `DEVICE` | `cpu` | Runs on CPU by default; no NVIDIA driver is required |
| `LOCAL_MODEL_PATH` | unset | Optional complete local checkpoint, for local testing without a Hub download |

Calibration uses **T=2.747760550703**, matching the released 0.8B checkpoint. This
rescales candidate probabilities; it is not a text-generation temperature.
Local inference uses one causal forward over the decision skeleton and reads
the logits immediately before each `<decision>` marker. It returns `choice`,
the probability of `yes` for `noul`, and the expected numeric value for `score`.

## API

`POST /v1/jev` and `POST /v1/decisions` accept a single Jev-style JSON request
and return `model`, `answers`, `usage`, timing, and calibration metadata. For example, from any client:

```python
import requests

request = {
    "state": "The order arrived on time.",
    "questions": {
        "on_time": {
            "type": "noul",
            "instructions": "Did the order arrive on time?",
        }
    },
}
response = requests.post(
    "https://YOUR-SPACE.hf.space/v1/jev", json=request, timeout=120
)
response.raise_for_status()
print(response.json()["answers"])
```

Requests can contain up to 16 questions and 62 choices per field. Use the
browser to upload images, or supply `images` as a list of objects with a
`data` key containing a base64 data URL. Image paths and arbitrary remote URLs
are rejected at the HTTP boundary. Limits: eight images, 12 MB per encoded
image payload, 32 MB combined, and 16 million pixels per image. The server
rejects animated images and normalizes uploads to temporary PNGs.

`GET /health` reports readiness. `GET /demo-meta` reports model identity,
calibration temperature, image upload limits, and image preprocessing settings.

Images preserve their aspect ratio and are aligned to the model's visual patch
grid. The demo does not upscale small images. It caps the processed image at
4,194,304 pixels, or roughly 4,096 visual tokens, leaving room for the state
and questions within the 8,192-token request limit. Multiple images and long
text can still exceed the limit and will be rejected without truncation.

## Local Docker use

The default is `DEVICE=cpu`. Remove any previously configured `DEVICE=cuda`
variable or change it to `cpu`. BF16 is retained to limit model memory; CPU
performance depends on the host. CPU predictions have not been checked for
numerical parity with the GPU evaluation results.

```bash
docker build -t intern-decision-space .
docker run --rm -p 7860:7860 intern-decision-space
```

Open `http://localhost:7860`. The same model download and secrets configuration
applies. An existing local model can be mounted read-only into the container
and selected using `LOCAL_MODEL_PATH`.

## Data handling

The application does not write request/response logs. Browser session history
stays in memory until the page closes and can be exported by the visitor.
Temporary uploaded files are deleted after local prediction, including on
failure. Requests are processed by the model in the Space. Hosting-platform
policies also apply.

## License

[LICENSE](LICENSE) contains the Apache-2.0 text used for this release.
[LICENSE-QWEN](LICENSE-QWEN) preserves the Qwen upstream license. The demo uses
the Intern-Decision model, derived from Qwen3.5, and the original Intern-Decision
demo interface adapted for standalone Hugging Face Spaces deployment.
