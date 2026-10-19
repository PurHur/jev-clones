"""Run the fused graph on the reference prompts, all questions of a record in one
right-padded batch, and compare with fixtures/reference.json."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

OUT = Path(os.environ.get("OUT", "out"))
PRECISION = os.environ.get("PRECISION", "q4f16")
suffix = "" if PRECISION == "fp32" else f"_{PRECISION}"
cfg = json.loads((OUT / "decider_config.json").read_text())
so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(str(OUT / f"onnx/model{suffix}.onnx"), so, providers=["CPUExecutionProvider"])
print([i.name for i in sess.get_inputs()], [o.name for o in sess.get_outputs()], file=sys.stderr)

worst, flips = 0.0, 0
dump = {}
for record in json.loads(Path("fixtures/reference.json").read_text()):
    cases = record["cases"]
    seqs = [c["state_ids"] + c["question_ids"] for c in cases]
    L, B, K = max(map(len, seqs)), len(seqs), max(len(c["opt_idx"]) for c in cases)
    ids = np.zeros((B, L), np.int64); mask = np.zeros((B, L), np.int64)
    answer = np.zeros(B, np.int64); options = np.zeros((B, K), np.int64)
    for b, (s, c) in enumerate(zip(seqs, cases)):
        ids[b, : len(s)] = s; mask[b, : len(s)] = 1
        answer[b] = len(s) - 1
        options[b, : len(c["opt_idx"])] = c["opt_idx"]
    (logits,) = sess.run(None, {"input_ids": ids, "attention_mask": mask, "answer_pos": answer, "option_pos": options})
    for b, c in enumerate(cases):
        k = len(c["opt_idx"])
        z = logits[b, :k] / cfg["temperature_by_kind"].get(c["kind"], cfg["temperature"])
        p = np.exp(z - z.max()); p /= p.sum()
        ref = np.array(c["probs"])
        dump[f"{record['id']}/{c['question']}"] = p.tolist()
        d = float(np.abs(p - ref).max()); worst = max(worst, d); flips += int(p.argmax() != ref.argmax())
        print(f"{record['id']:>10} {c['question']:<12} max|dp|={d:.4f}", file=sys.stderr)
(OUT / f"parity_{PRECISION}.json").write_text(json.dumps(dump, indent=1))
print(f"fused {PRECISION}: worst {worst:.4f}, argmax flips {flips}", file=sys.stderr)
