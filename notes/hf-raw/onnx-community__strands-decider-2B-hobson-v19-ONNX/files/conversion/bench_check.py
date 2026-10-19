"""Upstream engine (PyTorch, CPU fp32) on the first N typed-decisions states, the same
questions the open-jev benchmark asks, to compare with the browser's answers."""
import json
import sys
from pathlib import Path

from strands_decider.infer import load_engine
from strands_decider.schema import SystemOneRequest

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
rows = json.loads(Path.home().joinpath("work/rnd/gliner2/work/open-jev-demo-combined/public/typed-decisions-test.json").read_text())[:N]
engine = load_engine("StrandsAgents/strands-decider-2B-hobson-v19", device="cpu")
out = []
for i, row in enumerate(rows):
    raw = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
    gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
    questions = {}
    for qid, q in raw.items():
        if q["type"] == "choice":
            questions[qid] = {"type": "choice", "instructions": q["instructions"], "criteria": q["criteria"]}
        elif q["type"] == "score":
            questions[qid] = {"type": "score", "instructions": q["instructions"], "criteria": q["criteria"]}
        else:
            questions[qid] = {"type": "noul", "instructions": q["instructions"], **({"criteria": q["criteria"]} if q.get("criteria") else {})}
    resp = engine.evaluate(SystemOneRequest(state=row["state"], questions=questions)).model_dump()
    out.append({"i": i, "answers": resp["answers"], "gold": {k: v["label"] for k, v in gold.items()}, "types": {k: q["type"] for k, q in raw.items()}})
    for qid, a in resp["answers"].items():
        print(i, qid, raw[qid]["type"], "gold", gold[qid]["label"], "->", a.get("noul", a.get("choice", a.get("score"))), file=sys.stderr)
Path("fixtures/bench_check.json").write_text(json.dumps(out, indent=1))
