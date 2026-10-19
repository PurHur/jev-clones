"""Server settings: every value comes from an environment variable (see serve.env), CLI flags override.

Context and batch limits default to what fits the GPU: a 27B model in bf16 needs ~54 GB for weights,
so a 96 GB RTX PRO 6000 gets 32k-token requests / 16k-token batches, while 288 GB cards get 131k.
"""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHECKPOINT = str(Path(__file__).resolve().parents[2])


@dataclass(frozen=True)
class Settings:
    checkpoint: str
    model_name: str
    host: str
    port: int
    api_key: str | None
    max_context: int
    token_budget: int
    batch_size: int
    queue_seconds: float
    warmup: bool
    device: str | None


def gpu_limits() -> tuple[int, int]:
    """(max context tokens, batch token budget) sized to the visible GPU."""
    import torch

    if not torch.cuda.is_available():
        return 8192, 8192
    gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if gib >= 200:
        return 131072, 131072
    if gib >= 80:
        return 32768, 16384
    return 8192, 8192


def load() -> Settings:
    checkpoint = os.getenv("MJ_CHECKPOINT", DEFAULT_CHECKPOINT)
    key = os.getenv("MJ_API_KEY") or None
    key_file = os.getenv("MJ_API_KEY_FILE")
    if key is None and key_file and Path(key_file).is_file():
        key = Path(key_file).read_text().strip() or None
    auto_context, auto_budget = gpu_limits() if not (os.getenv("MJ_MAX_CONTEXT") and os.getenv("MJ_TOKEN_BUDGET")) else (0, 0)
    return Settings(
        checkpoint=checkpoint,
        model_name=os.getenv("MJ_MODEL_NAME") or Path(checkpoint).resolve().name,
        host=os.getenv("MJ_HOST", "127.0.0.1"),
        port=int(os.getenv("MJ_PORT", "8000")),
        api_key=key,
        max_context=int(os.getenv("MJ_MAX_CONTEXT") or auto_context),
        token_budget=int(os.getenv("MJ_TOKEN_BUDGET") or auto_budget),
        batch_size=int(os.getenv("MJ_BATCH_SIZE", "64")),
        queue_seconds=float(os.getenv("MJ_QUEUE_SECONDS", "60")),
        warmup=os.getenv("MJ_WARMUP", "1") not in ("0", "false", "no"),
        device=os.getenv("MJ_DEVICE") or None,
    )
