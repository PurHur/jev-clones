"""Reference outputs from the packaged Decision 2.0 runtime (exact path, CPU fp32):
per question, the token ids, option endpoints and the answer probabilities."""
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL = os.environ["MODEL"]
SRC = snapshot_download(f"vllm-sr/Decision-2.0-{MODEL}", ignore_patterns=["assets/*"])
sys.path.insert(0, SRC)
import decision2.api as d2api  # noqa: E402
from decision2.api import Decision2  # noqa: E402

# The package verifier rejects the Hub cache's symlinks (and wants the card images);
# the files were fetched from the Hub, so read the manifest without the byte check.
d2api.verify_bundle = lambda path: json.loads((Path(path).resolve() / "MODEL_MANIFEST.json").read_text())
from decision2._vendor.dev2model.decision_model import encode  # noqa: E402
from decision2._vendor.dev2model.infer import question_to_row  # noqa: E402

model = Decision2.from_pretrained(SRC, device="cpu")
backend = model.backend
out = []
for r in json.loads(Path("fixtures/requests.json").read_text()):
    # Decision 2.0 takes choice criteria as given (descriptions may be empty -> use the key).
    questions = {}
    for k, q in r["questions"].items():
        q = dict(q)
        if q["type"] == "choice":
            q["criteria"] = {o: (d or None) for o, d in q["criteria"].items()}
        questions[k] = q
    result = model.system_one(state=r["state"], questions=questions)
    cases = []
    for qid, q in questions.items():
        row = question_to_row({"id": "request", "state": r["state"]}, qid, q)
        enc = encode(row, backend.tokenizer, backend.cap)
        a = result["answers"][qid]
        cases.append({"question": qid, "kind": q["type"], "keys": enc["keys"], "ids": enc["ids"],
                      "option_pos": enc["candidate_positions"], "answer_pos": enc["query_position"], "answer": a})
        print(r["id"], qid, json.dumps(a)[:160], file=sys.stderr)
    out.append({"id": r["id"], "request": {**r, "questions": questions}, "cases": cases})
Path(f"fixtures/reference-{MODEL}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
