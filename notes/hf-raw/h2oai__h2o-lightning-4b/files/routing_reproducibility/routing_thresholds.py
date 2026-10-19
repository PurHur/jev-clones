#!/usr/bin/env python3
"""Routing thresholds for H2O-Lightning-4B from the per-question predictions in predictions.jsonl. Python 3.8+, standard
library only.

For a question type and a confidence threshold t, a question is ESCALATED when its top probability is below t; it is
CONFIDENTLY WRONG when its top probability is at or above t and the top answer is not in the gold set. Both are shares of
all questions of that type. "Accuracy on the rest" is the accuracy of the questions that are not escalated.

Probabilities are softmax(raw_logprobs_t1 / T) per question, with T the per-type temperature (the shipped temperatures
by default: choice 0.75, yes/no 0.8, score 0.65). That is exactly what the served shim computes with the yes/no floor
off (routing mode); --served uses the recorded served probabilities instead (the same numbers up to float rounding).

    python3 routing_thresholds.py                                  # the model card's table (thresholds 0.8 and 0.9)
    python3 routing_thresholds.py --thresholds 0.5 0.6 0.7 0.8 0.9 0.95
    python3 routing_thresholds.py --escalate 0.25                  # per type: the highest threshold that escalates at most 25 %
    python3 routing_thresholds.py --max-wrong 0.01                 # per type: the lowest threshold that keeps confidently wrong <= 1 %
    python3 routing_thresholds.py --temps choice=1,noul=1,score=1  # other temperatures
    python3 routing_thresholds.py --dataset plumb --json out.json  # one source dataset; machine-readable output
"""
import argparse, json, math, os

SHIPPED = {"choice": 0.75, "noul": 0.8, "score": 0.65}
TYPES = ("choice", "noul", "score")
LABEL = {"choice": "choice", "noul": "yes/no", "score": "score", "all": "all"}


def load(path, dataset=None):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if dataset and dataset.lower() not in r["dataset"].lower():
                continue
            rows.append(r)
    return rows


def softmax(lp, T):
    x = [v / T for v in lp]
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]


def scored(rows, temps, served=False):
    """(type, top probability, correct) per question"""
    out = []
    for r in rows:
        p = r["served_probabilities"] if served else softmax(r["raw_logprobs_t1"], temps[r["question_type"]])
        k = max(range(len(p)), key=p.__getitem__)
        out.append((r["question_type"], p[k], k in set(r["gold_indices"])))
    return out


def cell(items, t):
    n = len(items)
    kept = [c for _, p, c in items if p >= t]
    return {"threshold": t, "escalated": (n - len(kept)) / n, "accuracy_on_rest": (sum(kept) / len(kept)) if kept else None,
            "confidently_wrong": sum(1 for c in kept if not c) / n}


def pick(items, ok, highest=False):
    """the lowest (or highest) threshold, among 0 and the observed top probabilities, whose cell satisfies ok"""
    ts = [0.0] + sorted({p for _, p, _ in items})
    for t in (reversed(ts) if highest else ts):
        c = cell(items, t)
        if ok(c):
            return c
    return None


def pct(v):
    return "-" if v is None else f"{100 * v:.1f} %"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--predictions", default=os.path.join(here, "predictions.jsonl"))
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.8, 0.9])
    ap.add_argument("--temps", default="", help="per-type temperatures, e.g. choice=0.75,noul=0.8,score=0.65")
    ap.add_argument("--served", action="store_true", help="use the recorded served probabilities (shipped temperatures)")
    ap.add_argument("--escalate", type=float, help="budget: the largest share of questions you are willing to escalate")
    ap.add_argument("--max-wrong", type=float, help="budget: the largest share of confidently wrong answers you accept")
    ap.add_argument("--dataset", help="only the rows whose dataset name contains this (plumb or eikos)")
    ap.add_argument("--json", help="also write the result here")
    a = ap.parse_args()
    temps = dict(SHIPPED)
    for kv in filter(None, a.temps.split(",")):
        k, v = kv.split("=")
        temps[k.strip()] = float(v)
    if a.served and a.temps:
        ap.error("--served uses the shipped temperatures; drop --temps")
    items = scored(load(a.predictions, a.dataset), temps, a.served)
    groups = {qt: [x for x in items if x[0] == qt] for qt in TYPES}
    groups["all"] = items
    print(f"{len(items)} questions; temperatures {'served (shipped)' if a.served else temps}; yes/no floor off")
    out = {"temperatures": "served" if a.served else temps, "groups": {}}
    for g, its in groups.items():
        if not its:
            continue
        acc = sum(c for _, _, c in its) / len(its)
        res = {"n": len(its), "accuracy": acc, "by_threshold": [cell(its, t) for t in a.thresholds]}
        if a.escalate is not None:
            res["escalation_budget"] = pick(its, lambda c: c["escalated"] <= a.escalate + 1e-12, highest=True)
        if a.max_wrong is not None:
            res["wrong_budget"] = pick(its, lambda c: c["confidently_wrong"] <= a.max_wrong + 1e-12)
        out["groups"][g] = res
    print("\n| question type (n) | threshold | escalated | accuracy on the rest | confidently wrong |\n|---|---|---|---|---|")
    for g, res in out["groups"].items():
        for i, c in enumerate(res["by_threshold"]):
            name = f"{LABEL[g]} ({res['n']:,})" if i == 0 else LABEL[g]
            print(f"| {name} | {c['threshold']:g} | {pct(c['escalated'])} | {pct(c['accuracy_on_rest'])} | {pct(c['confidently_wrong'])} |")
    print("\nWithout a threshold: " + ", ".join(f"{LABEL[g]} {pct(r['accuracy'])}" for g, r in out["groups"].items()))
    for key, what in (("escalation_budget", "Highest threshold per type escalating at most " + pct(a.escalate)),
                      ("wrong_budget", "Lowest threshold per type with confidently wrong at most " + pct(a.max_wrong))):
        if any(key in r for r in out["groups"].values()):
            print(f"\n{what}:")
            for g, r in out["groups"].items():
                c = r.get(key)
                print(f"  {LABEL[g]}: " + ("not reachable" if c is None else
                      f"t = {c['threshold']:.4f}: escalated {pct(c['escalated'])}, accuracy on the rest "
                      f"{pct(c['accuracy_on_rest'])}, confidently wrong {pct(c['confidently_wrong'])}"))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
