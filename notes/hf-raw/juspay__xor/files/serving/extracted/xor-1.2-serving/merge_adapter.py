import os
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: merge_adapter.py <adapter-dir> <output-dir>")

    adapter_dir = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    if not adapter_dir.is_dir():
        raise SystemExit(f"adapter directory does not exist: {adapter_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {output_dir}")

    adapter_config = PeftConfig.from_pretrained(adapter_dir, local_files_only=True)
    if adapter_config.base_model_name_or_path != "Qwen/Qwen3.6-35B-A3B":
        raise SystemExit(f"unexpected base model: {adapter_config.base_model_name_or_path}")

    base_path = Path(os.environ.get("BASE_MODEL_PATH", adapter_config.base_model_name_or_path))
    local_base = base_path.exists()
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        base_path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=local_base,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        is_trainable=False,
        local_files_only=True,
    )
    merged_model = peft_model.merge_and_unload(safe_merge=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir, max_shard_size="5GB", safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    for name in (
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "configuration.json",
        "generation_config.json",
        "LICENSE",
    ):
        adapter_source = adapter_dir / name
        base_source = base_path / name
        source = adapter_source if adapter_source.is_file() else base_source
        if source.is_file():
            shutil.copy2(source, output_dir / name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
