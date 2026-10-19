"""Run a fused graph on the reference prompts (one right-padded batch per record) and
compare probabilities with fixtures/reference-<MODEL>.json."""
import json, os, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

MODEL, NAME = os.environ["MODEL"], os.environ.get("NAME", "model_quantized")
bias = {}
mf = json.loads(Path(f"fixtures/reference-{MODEL}.json").read_text())
so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(f"out/{MODEL}/onnx/{NAME}.onnx", so, providers=["CPUExecutionProvider"])
offsets = json.loads(os.environ.get("SCORE_BIAS", "{}"))
worst, flips, dump = 0.0, 0, {}
for rec in mf:
    cs = rec["cases"]
    L, K, B = max(len(c["ids"]) for c in cs), max(len(c["option_pos"]) for c in cs), len(cs)
    ids = np.zeros((B, L), np.int64); mask = np.zeros((B, L), np.int64)
    ans = np.zeros(B, np.int64); opt = np.zeros((B, K), np.int64)
    for b, c in enumerate(cs):
        ids[b, :len(c["ids"])] = c["ids"]; mask[b, :len(c["ids"])] = 1
        ans[b] = c["answer_pos"]; opt[b, :len(c["option_pos"])] = c["option_pos"]
    (lg,) = sess.run(None, {"input_ids": ids, "attention_mask": mask, "answer_pos": ans, "option_pos": opt})
    for b, c in enumerate(cs):
        z = lg[b, :len(c["keys"])].astype(np.float64)
        if c["kind"] == "score" and str(len(z)) in offsets:
            z = z + np.array(offsets[str(len(z))])
        p = np.exp(z - z.max()); p /= p.sum()
        a = c["answer"]
        ref = np.array([1 - a["noul"], a["noul"]]) if c["kind"] == "noul" else np.array([a["probabilities"][k] for k in c["keys"]])
        d = float(np.abs(p - ref).max()); worst = max(worst, d); flips += int(p.argmax() != ref.argmax())
        dump[f"{rec['id']}/{c['question']}"] = p.tolist()
        print(f"{rec['id']:>10} {c['question']:<12} {d:.4f}", file=sys.stderr)
Path(f"out/{MODEL}/parity_{NAME}.json").write_text(json.dumps(dump))
print(f"{MODEL} {NAME}: worst {worst:.4f}, flips {flips}", file=sys.stderr)
