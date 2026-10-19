"""fp32 / q4f16 ONNX (fused graph for q4f16, unfused for fp32) on the first N
typed-decisions states, prompts built with the upstream prompting + tokenizer, to
separate quantization error from port error against fixtures/bench_check.json."""
import json, os, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from strands_decider.prompting import render_question, render_state
from strands_decider.schema import SystemOneRequest
from strands_decider.infer import _option_token_index

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
PRECISION = os.environ.get("PRECISION", "fp32")
cfg = json.loads(Path("out/decider_config.json").read_text())
tok = AutoTokenizer.from_pretrained("StrandsAgents/strands-decider-2B-hobson-v19")
rows = json.loads(Path.home().joinpath("work/rnd/gliner2/work/open-jev-demo-combined/public/typed-decisions-test.json").read_text())[:N]
ref = json.loads(Path("fixtures/bench_check.json").read_text())
so = ort.SessionOptions(); so.log_severity_level = 3
if PRECISION == "fp32":
    embed = ort.InferenceSession("out/onnx/embed_tokens.onnx", so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession("out/onnx/decoder_model_merged.onnx", so, providers=["CPUExecutionProvider"])
    H, D = 2048, 256
    flat = np.load("out/head.npy"); nw, nb, qw, qb, kw, kb = np.split(flat, np.cumsum([H, H, D*H, D, D*H, D])[:-1]); qw = qw.reshape(D, H); kw = kw.reshape(D, H)
else:
    fused = ort.InferenceSession(f"out/onnx/model_{PRECISION}.onnx", so, providers=["CPUExecutionProvider"])

def ln(x):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * nw + nb

worst = 0; flips = 0; n = 0
for i, row in enumerate(rows):
    raw = json.loads(row["questions"])
    qs = {k: ({"type": "noul", "instructions": q["instructions"], **({"criteria": q["criteria"]} if q.get("criteria") else {})} if q["type"] == "noul" else {"type": q["type"], "instructions": q["instructions"], "criteria": q["criteria"]}) for k, q in raw.items()}
    req = SystemOneRequest(state=row["state"], questions=qs)
    st = render_state(req.state)
    s = tok(st, add_special_tokens=True)["input_ids"]
    for name, q in req.questions.items():
        rq = render_question(q)
        enc = tok(rq.text, add_special_tokens=False, return_offsets_mapping=True)
        opt = [len(s) + j for j in _option_token_index(enc["offset_mapping"], rq.option_spans, 0)]
        ids = np.array([s + enc["input_ids"]], np.int64); L = ids.shape[1]
        if PRECISION == "fp32":
            (e,) = embed.run(None, {"input_ids": ids})
            feeds = {"inputs_embeds": e, "attention_mask": np.ones((1, L), np.int64), "position_ids": np.tile(np.arange(L), (3, 1, 1)), "num_logits_to_keep": np.array(0, np.int64)}
            for inp in dec.get_inputs():
                if inp.name.startswith("past_"):
                    feeds[inp.name] = np.zeros([1 if d == "batch_size" else (0 if isinstance(d, str) else d) for d in inp.shape], np.float32)
            h = ln(dec.run(["last_hidden_state"], feeds)[0][0])
            logits = (h[opt] @ kw.T + kb) @ (h[-1] @ qw.T + qb) * D ** -0.5
        else:
            (lg,) = fused.run(None, {"input_ids": ids, "attention_mask": np.ones((1, L), np.int64), "answer_pos": np.array([L - 1]), "option_pos": np.array([opt])})
            logits = lg[0]
        z = logits / cfg["temperature_by_kind"][rq.kind]; p = np.exp(z - z.max()); p /= p.sum()
        a = ref[i]["answers"][name]
        if a["type"] == "noul":
            rv = np.array([1 - a["noul"], a["noul"]])
        else:
            rv = np.array([a["probabilities"][lbl] for lbl in rq.slot_labels])
        d = float(np.abs(p - rv).max()); worst = max(worst, d); flips += int(p.argmax() != rv.argmax()); n += 1
        print(i, name, L, f"{d:.4f}", file=sys.stderr)
print(f"{PRECISION}: {n} questions, worst {worst:.4f}, flips {flips}", file=sys.stderr)
