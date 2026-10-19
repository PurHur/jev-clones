"""Assemble out/<MODEL>/repo for onnx-community: config.json with a `decision2` section
for open-jev, the model's tokenizer, the graphs (hard links) and the conversion scripts."""
import json, os, shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL = os.environ["MODEL"]
SRC = Path(snapshot_download(f"vllm-sr/Decision-2.0-{MODEL}", ignore_patterns=["assets/*"]))
OUT = Path(f"out/{MODEL}")
REPO = OUT / "repo"
(REPO / "onnx").mkdir(parents=True, exist_ok=True)
FILES = {"q8": "model_quantized", "q4f16": "model_q4f16"}
dtypes = [d for d, n in FILES.items() if (OUT / "onnx" / f"{n}.onnx").exists()]
for d in dtypes:
    for f in (OUT / "onnx").glob(f"{FILES[d]}.onnx*"):
        t = REPO / "onnx" / f.name
        t.unlink(missing_ok=True)
        os.link(f, t)
manifest = json.loads((SRC / "MODEL_MANIFEST.json").read_text())
backbone = json.loads((SRC / "backbone/config.json").read_text())
text = backbone.get("text_config", backbone)
bias = (manifest.get("score_bias") or {}).get("offsets") or {}
config = {
    "architectures": ["Decision2Model"],
    "model_type": "qwen3" if backbone.get("model_type") == "qwen3" else "qwen3_5_text",
    **{k: text[k] for k in ("hidden_size", "num_hidden_layers", "vocab_size") if k in text},
    "decision2": {
        "description": "Decision 2.0 (vLLM Semantic Router): one prompt per question (Context / Task type / Question / Options as JSON {description, key} segments / Decision:). The graph returns the CandidateHead score of each option's last token against the final token; add score_bias[L] for an L-level score, then softmax (temperature 1).",
        "source": f"vllm-sr/Decision-2.0-{MODEL}",
        "max_input_tokens": manifest["max_input_tokens"],
        "score_bias": bias,
        "dtypes": dtypes,
    },
    "transformers.js_config": {"use_external_data_format": {f"{FILES[d]}.onnx": 1 for d in dtypes}, "dtype": "q8"},
}
(REPO / "config.json").write_text(json.dumps(config, indent=2) + "\n")
for name in ("tokenizer.json", "tokenizer_config.json"):
    (REPO / name).unlink(missing_ok=True)
    shutil.copyfile(SRC / name, REPO / name)
conv = REPO / "conversion"
(conv / "fixtures").mkdir(parents=True, exist_ok=True)
for f in ("build.py", "export_qwen3.py", "quantize_qwen3.py", "reference.py", "parity.py", "make_repo.py", "qwen3_5_opt_mapping.json"):
    shutil.copyfile(f, conv / f)
for f in ("fixtures/requests.json", f"fixtures/reference-{MODEL}.json"):
    shutil.copyfile(f, conv / f)
print(MODEL, dtypes, sorted(str(p.relative_to(REPO)) for p in REPO.glob("*")), file=sys.stderr)
