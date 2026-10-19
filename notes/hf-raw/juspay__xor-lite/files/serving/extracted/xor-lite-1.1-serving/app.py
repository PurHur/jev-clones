"""XOR Lite 1.1 API: the vendored strands_decider System One server with the release settings.

The model directory holds the base at its root and the release checkpoint in decider/.
decider/strands_decider_config.json names the base as "..", so the server loads from
inside decider/. The LoRA adapter is applied at runtime by PEFT; it is never merged into
the bf16 base weights, because a merge rounds away part of the adapter delta.
"""

import os

import uvicorn

MODEL_DIR = os.environ.get("XOR_LITE_MODEL_DIR", "/models/xor-lite")
MODEL_ID = os.environ.get("XOR_LITE_MODEL_ID", "xor-lite")
RELEASE_DATE = os.environ.get("XOR_LITE_RELEASE_DATE", "2026-10-14")
PORT = int(os.environ.get("XOR_LITE_PORT", "30002"))

os.chdir(os.path.join(MODEL_DIR, "decider"))

from strands_decider.server import create_app  # noqa: E402

# The prefix cache is off: the Decision Index run that the release reports scored each
# request with plain batched encoding.
app = create_app(".", device="cuda", use_prefix_cache=False, model_name=MODEL_ID)


@app.get("/v1/models")
def models() -> dict:
    return {
        "data": [{"id": MODEL_ID, "underlying": MODEL_ID, "aliases": [MODEL_ID]}],
        "models": [{
            "name": MODEL_ID,
            "description": "General-purpose system-one model.",
            "release_date": RELEASE_DATE,
        }],
    }


if __name__ == "__main__":
    # One worker: the model owns the GPU.
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)
