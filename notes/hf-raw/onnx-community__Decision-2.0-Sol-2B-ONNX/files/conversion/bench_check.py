"""Packaged runtime (CPU fp32) vs the q8 ONNX graph on the first N typed-decisions
states, same prompts (built by the package's own encoder), to separate port error
from model quality on long benchmark prompts."""
import json, os, sys
from pathlib import Path
import numpy as np, onnxruntime as ort
from huggingface_hub import snapshot_download

MODEL, N = os.environ["MODEL"], int(os.environ.get("N", "6"))
SRC = snapshot_download(f"vllm-sr/Decision-2.0-{MODEL}", ignore_patterns=["assets/*"])
sys.path.insert(0, SRC)
import decision2.api as d2api
d2api.verify_bundle = lambda path: json.loads((Path(path).resolve() / "MODEL_MANIFEST.json").read_text())
from decision2._vendor.dev2model.decision_model import encode
from decision2._vendor.dev2model.infer import question_to_row
model = d2api.Decision2.from_pretrained(SRC, device="cpu")
so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(f"out/{MODEL}/onnx/model_quantized.onnx", so, providers=["CPUExecutionProvider"])
bias = (json.loads(Path(f"out/{MODEL}/repo/config.json").read_text())["decision2"]["score_bias"])
rows = json.loads(Path.home().joinpath("work/rnd/gliner2/work/open-jev-demo-combined/public/typed-decisions-test.json").read_text())[:N]
worst, flips, n = 0.0, 0, 0
for row in rows:
    raw = json.loads(row["questions"])
    qs = {k: ({"type": "noul", "instructions": q["instructions"], **({"criteria": q["criteria"]} if q.get("criteria") else {})} if q["type"] == "noul" else {"type": q["type"], "instructions": q["instructions"], "criteria": q["criteria"]}) for k, q in raw.items()}
    ans = model.system_one(state=row["state"], questions=qs)["answers"]
    for k, q in qs.items():
        enc = encode(question_to_row({"id": "r", "state": row["state"]}, k, q), model.backend.tokenizer, model.backend.cap)
        ids = np.array([enc["ids"]])
        (lg,) = sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids), "answer_pos": np.array([enc["query_position"]]), "option_pos": np.array([enc["candidate_positions"]])})
        z = lg[0].astype(np.float64)
        if q["type"] == "score" and str(len(z)) in bias: z = z + np.array(bias[str(len(z))])
        p = np.exp(z - z.max()); p /= p.sum()
        a = ans[k]
        r = np.array([1 - a["noul"], a["noul"]]) if q["type"] == "noul" else np.array([a["probabilities"][key] for key in enc["keys"]])
        worst = max(worst, float(np.abs(p - r).max())); flips += int(p.argmax() != r.argmax()); n += 1
print(f"{MODEL}: {n} benchmark questions, q8 vs packaged runtime worst {worst:.4f}, top answer changed {flips}", file=sys.stderr)
