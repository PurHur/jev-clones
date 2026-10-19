"""Reference outputs from the upstream engine (PyTorch, CPU fp32), one question at a time
on the full prompt, which is what the browser does. For each question it records the
token ids, the option positions and the probabilities, so the ONNX graph and the
TypeScript port can each be checked separately. Also records the engine's own response
with the shared-prefix cache, to show how far that path sits from the full prompt."""
import json
import sys
from pathlib import Path

import dataclasses

import torch
from strands_decider.infer import load_engine
from strands_decider.prompting import render_question, render_state
from strands_decider.schema import SystemOneRequest

CKPT = "StrandsAgents/strands-decider-2B-hobson-v19"
requests = json.loads(Path("fixtures/requests.json").read_text())
engine = load_engine(CKPT, device="cpu", use_prefix_cache=False)
model = engine.model
out = []
for r in requests:
    req = SystemOneRequest(state=r["state"], questions=r["questions"])
    state_text = render_state(req.state)
    cases = []
    for name, q in req.questions.items():
        rq = render_question(q)
        s, (qi,) = engine._fit(state_text, [rq.text])
        opt_idx = engine._option_idx([rq], len(s))
        ids = torch.tensor([s + qi])
        with torch.inference_mode():
            hidden = model.encode(ids, torch.ones_like(ids)).float()
            res = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                        n_slots=torch.tensor([rq.n_slots]),
                        temperature=engine._temperatures([rq.kind]), opt_idx=opt_idx)
        cases.append({
            "question": name, "kind": rq.kind, "labels": list(rq.slot_labels),
            "prompt": state_text + rq.text, "state_ids": s, "question_ids": qi,
            "opt_idx": opt_idx[0].tolist(),
            "probs": res["log_probs"].exp()[0].tolist(),
            "hidden_answer_head": hidden[0, -1, :8].tolist(),
        })
        print(r["id"], name, [round(p, 4) for p in cases[-1]["probs"]], file=sys.stderr)
    full = engine.evaluate(req).model_dump()
    engine.cfg = dataclasses.replace(engine.cfg, use_prefix_cache=True)
    shared = engine.evaluate(req).model_dump()
    engine.cfg = dataclasses.replace(engine.cfg, use_prefix_cache=False)
    out.append({"id": r["id"], "request": r, "cases": cases, "response": full, "response_shared_prefix": shared})
Path("fixtures/reference.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("wrote fixtures/reference.json", file=sys.stderr)
