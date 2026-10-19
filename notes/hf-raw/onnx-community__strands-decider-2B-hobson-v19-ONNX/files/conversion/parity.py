"""Check the ONNX build against the PyTorch reference (fixtures/reference.json).

Runs embed_tokens + decoder_model_merged (empty caches, one question per pass) on the
reference token ids, applies the pointer head from head.npy, and compares the
probabilities. PRECISION=fp32 (default) or q4f16; the q4f16 graph on the CPU EP is
only a rough check (see the OneJev notes), the browser check is the real one."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

OUT = Path(os.environ.get("OUT", "out"))
PRECISION = os.environ.get("PRECISION", "fp32")
suffix = "" if PRECISION == "fp32" else f"_{PRECISION}"
cfg = json.loads((OUT / "decider_config.json").read_text())
H, D = cfg["hidden_size"], cfg["pointer_dim"]
flat = np.load(OUT / "head.npy")
sizes = [H, H, D * H, D, D * H, D]
parts = np.split(flat, np.cumsum(sizes)[:-1])
norm_w, norm_b, q_w, q_b, k_w, k_b = parts
q_w, k_w = q_w.reshape(D, H), k_w.reshape(D, H)

so = ort.SessionOptions()
so.log_severity_level = 3
embed = ort.InferenceSession(str(OUT / f"onnx/embed_tokens{suffix}.onnx"), so, providers=["CPUExecutionProvider"])
decoder = ort.InferenceSession(str(OUT / f"onnx/decoder_model_merged{suffix}.onnx"), so, providers=["CPUExecutionProvider"])
inputs = {i.name: i for i in decoder.get_inputs()}
dtype = np.float16 if "float16" in inputs["inputs_embeds"].type else np.float32


def empty_caches():
    feeds = {}
    for name, info in inputs.items():
        if name.startswith("past_"):
            shape = [1 if d == "batch_size" else (0 if isinstance(d, str) else d) for d in info.shape]
            feeds[name] = np.zeros(shape, np.float16 if "float16" in info.type else np.float32)
    return feeds


def layer_norm(x):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + cfg["layer_norm_eps"]) * norm_w + norm_b


def head(hidden, opt_idx, kind):
    h = layer_norm(hidden.astype(np.float32))
    d = h[-1] @ q_w.T + q_b
    o = h[opt_idx] @ k_w.T + k_b
    logits = (o @ d) * D ** -0.5 / cfg["temperature_by_kind"].get(kind, cfg["temperature"])
    e = np.exp(logits - logits.max())
    return e / e.sum()


worst = 0.0
flips = 0
for record in json.loads(Path("fixtures/reference.json").read_text()):
    for case in record["cases"]:
        ids = np.array([case["state_ids"] + case["question_ids"]], np.int64)
        L = ids.shape[1]
        (emb,) = embed.run(None, {"input_ids": ids})
        feeds = {
            "inputs_embeds": emb.astype(dtype),
            "attention_mask": np.ones((1, L), np.int64),
            "position_ids": np.tile(np.arange(L, dtype=np.int64), (3, 1, 1)),
            "num_logits_to_keep": np.array(0, np.int64),
            **empty_caches(),
        }
        hidden = decoder.run(["last_hidden_state"], feeds)[0][0]
        assert hidden.shape[0] == L, hidden.shape
        probs = head(hidden, case["opt_idx"], case["kind"])
        ref = np.array(case["probs"])
        diff = float(np.abs(probs - ref).max())
        worst = max(worst, diff)
        flips += int(probs.argmax() != ref.argmax())
        print(f"{record['id']:>10} {case['question']:<12} max|dp|={diff:.4f}", np.round(probs, 4).tolist(), np.round(ref, 4).tolist(), file=sys.stderr)
print(f"{PRECISION}: worst {worst:.4f}, argmax flips {flips}", file=sys.stderr)
