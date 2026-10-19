"""Assemble the publishable repo (local only) in out/repo: config.json with a `decider`
section for open-jev, the decider's tokenizer, and the fused graph(s)."""
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

OUT = Path("out")
REPO = OUT / "repo"
DECIDER = Path(snapshot_download("StrandsAgents/strands-decider-2B-hobson-v19", ignore_patterns=["eval/*", "training/*"]))
TORSO = Path(snapshot_download("Qwen/Qwen3.5-2B-Base", revision="b1485b2fa6dfa1287294f269f5fb618e03d52d7c"))

(REPO / "onnx").mkdir(parents=True, exist_ok=True)
FILES = {"q4f16": "model_q4f16", "q8": "model_quantized"}
dtypes = [d for d, n in FILES.items() if (OUT / f"onnx/{n}.onnx").exists()]
for d in dtypes:
    for f in (OUT / "onnx").glob(f"{FILES[d]}.onnx*"):
        target = REPO / "onnx" / f.name
        if target.exists():
            target.unlink()
        os.link(f, target)  # hard link: no second copy on a full disk

text = json.loads((TORSO / "config.json").read_text())["text_config"]
head = json.loads((OUT / "decider_config.json").read_text())
config = {
    "architectures": ["StrandsDeciderModel"],
    "model_type": "qwen3_5_text",
    **{k: text[k] for k in ("hidden_size", "num_hidden_layers", "vocab_size", "layer_types")},
    "pad_token_id": 248044,
    "decider": {
        "description": "strands-decider-2B (hobson v19): Qwen3.5-2B-Base + rank-16 LoRA + a pointer head. One prompt per question: <state>…</state> then the rendered question ending at <answer>. The logit of option k is the scaled dot product of the head's q(<answer> state) and k(last token of option k's line); divide by temperature_by_kind[type] and softmax.",
        "source": "StrandsAgents/strands-decider-2B-hobson-v19",
        "max_length": head["max_length"],
        "max_question_fraction": 0.75,
        "temperature": head["temperature"],
        "temperature_by_kind": head["temperature_by_kind"],
        "ordinal_smoothing": head["ordinal_smoothing"],
        "inputs": {"answer_pos": "index of the last real token (<answer>) per row", "option_pos": "index of each option line's last token, per row"},
    },
    "transformers.js_config": {
        "use_external_data_format": {f"{FILES[d]}.onnx": 1 for d in dtypes},
        "dtype": "q4f16",
    },
}
(REPO / "config.json").write_text(json.dumps(config, indent=2) + "\n")
for name in ("tokenizer.json", "tokenizer_config.json"):
    (REPO / name).unlink(missing_ok=True)
    shutil.copyfile(DECIDER / name, REPO / name)
print(sorted(str(p.relative_to(REPO)) for p in REPO.rglob("*") if p.is_file()))
