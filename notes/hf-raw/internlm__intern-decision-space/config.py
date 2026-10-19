"""Space settings and download of data-only model assets from Hugging Face."""
import os
from pathlib import Path
from typing import Literal

from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict, Field

MODEL_ID = "InternLM/Intern-Decision-0.8B"


def resolve_device(requested):
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but no NVIDIA GPU is available. Select GPU hardware "
            "in Space Settings, or set DEVICE=auto (CPU fallback) or DEVICE=cpu."
        )
    return str(device)


class SpaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_revision: str = "main"
    local_model_path: str | None = None
    max_length: int = Field(default=8192, ge=1, le=8192)
    device: str = "cpu"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: Literal["sdpa", "eager"] = "sdpa"

    @classmethod
    def from_env(cls):
        return cls(
            model_revision=os.environ.get("MODEL_REVISION", "main"),
            local_model_path=os.environ.get("LOCAL_MODEL_PATH") or None,
            max_length=int(os.environ.get("MAX_INPUT_LENGTH", "8192")),
            device=os.environ.get("DEVICE", "cpu"),
            dtype=os.environ.get("MODEL_DTYPE", "bfloat16"),
            attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
        )


def resolve_checkpoint(config):
    if config.local_model_path:
        checkpoint = Path(config.local_model_path).resolve()
        if not (checkpoint / "model.safetensors.index.json").is_file():
            raise ValueError("LOCAL_MODEL_PATH must contain a complete HF checkpoint")
        return str(checkpoint)
    # Do not download or execute Python code from the model repository.
    return snapshot_download(
        repo_id=MODEL_ID,
        revision=config.model_revision,
        token=os.environ.get("HF_TOKEN") or None,
        allow_patterns=[
            "*.safetensors", "model.safetensors.index.json", "config.json",
            "tokenizer.json", "tokenizer_config.json", "added_tokens.json",
            "special_tokens_map.json", "vocab.json", "merges.txt",
            "chat_template.jinja", "chat_templates/*.jinja",
            "preprocessor_config.json", "processor_config.json",
            "video_preprocessor_config.json", "generation_config.json",
        ],
    )
