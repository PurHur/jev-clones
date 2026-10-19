"""Fetch what the conversion needs: the decider checkpoint (LoRA + head), its base
torso at the pinned revision, and onnx-community's Qwen3.5-2B OPT text graphs."""
from huggingface_hub import snapshot_download

DECIDER = "StrandsAgents/strands-decider-2B-hobson-v19"
BASE = "Qwen/Qwen3.5-2B-Base"
BASE_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"  # hf_export.BASE_REVISION
ONNX = "onnx-community/Qwen3.5-2B-ONNX-OPT"

print(snapshot_download(DECIDER, ignore_patterns=["eval/*", "training/*"]))
print(snapshot_download(BASE, revision=BASE_REVISION))
print(snapshot_download(ONNX, local_dir="base_onnx", allow_patterns=[
    "*.json", "*.jinja",
    "onnx/decoder_model_merged.onnx*", "onnx/decoder_model_merged_q4f16.onnx*",
    "onnx/embed_tokens.onnx*", "onnx/embed_tokens_q4f16.onnx*",
]))
