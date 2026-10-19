"""Build site/index.html: the dashboard, with every dataset embedded.

Sources (all in this repo):
  data/models.json                         curated architecture facts per profiled model (from clones/*.md)
  notes/hf-raw/_spaces/dix_table.tsv       Decision Index 0.3.2 board snapshot (2026-10-16)
  notes/board_dates.json                   HF creation date per board repo (scripts/board_dates.py)
  notes/families.json                      197 families found by the discovery sweep
  notes/arch/arch_table.tsv                config fields of 479 HF repos (scripts/hf_arch.py)
  clones/*.md, COMPARISON.md               profiles shown in the reader

Usage: python3 -I scripts/build_site.py
"""

import ast
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# profiled model id -> (family name substring in families.json, board weights-URL substring or None)
LINKS = {
    "jev": (None, None), "zeiger": ("Zeiger", None),
    "pplx": ("pplx-decider", "perplexity-ai/pplx-decider-v1.2"),
    "pplx-flash": ("pplx-decider", "perplexity-ai/pplx-decider-flash-v1.2"),
    "lemon": ("Lemon", None), "d3": ("Decision 1.0", "vllm-sr/d3/"),
    "d2": ("Decision 1.0", "Decision-2.0-Vega"), "vela": ("Decision 1.0", None),
    "clef": ("Clef", "Cloudflare/clef"), "d1": ("d1-3B", "LiquidAI/d1-3B"), "d1omni": ("d1-3B", "d1-omni"),
    "strands": ("Strands", None), "granite": ("granite", None), "intern": ("Intern-Decision", "Intern-Decision-4B"),
    "h2o": ("H2O-Lightning", None), "tev1": ("Tev1", "Tev1-4B"), "meta": ("MetaEncoder", None),
    "kev": ("kev", "jaredpalmer/kev-27b"), "laya": ("Laya", None), "vegaml": ("vegaml", None),
    "autotrust": ("JEV-9B", "autotrust/JEV-27B"), "decider": ("Decider (0.8B", "Mapika/decider-31b"),
    "webjev": ("WebJev", None), "startlux": ("StartLux", "StartLux-Decision-v1.1-27B"),
    "xor": ("Xor", "juspay/Xor-26B-A4B"), "drex": ("Drex", "nace-ai/drex-v2.0"), "jeeves": ("Jeeves", None),
    "openjev-e": ("OpenJev-4B (ejhshen)", None), "openjev-z": ("Open-Jev 2B", None), "lev": ("Lev (interfaze)", "interfaze-ai/lev"),
    "lev-loop": ("looped", None), "von": ("Von", None), "clm": ("CLM", None), "nimble": ("Nimble", None),
    "standardone": ("Standard One", "StandardOne-3B-SH"), "basal": ("Basal", None), "crivo": ("rivo", "tardellirs/crivo-150m"),
    "videojev": ("VideoJev", None),
}


# release dates stated in the profiles for that exact model; every other model takes its family's first upload
DATED = {"jev", "pplx", "pplx-flash", "d3", "d2", "vela", "clef", "d1", "d1omni", "strands", "granite", "intern",
         "h2o", "tev1", "meta", "kev", "vegaml", "lemon"}


def board() -> list[dict]:
    rows = []
    for line in (ROOT / "notes/hf-raw/_spaces/dix_table.tsv").read_text().splitlines():
        cells = line.split("\t")
        if len(cells) == 1 and line.startswith("JEV "):
            # header line of the snapshot carries the current Jev reference: "JEV <score> <ece>"
            _, score, ece = line.split()
            rows.append({"score": float(score), "slug": "jev-1.14.0", "name": "Jev 1.14", "org": "TypeSafe",
                         "kind": "hosted API", "params": None, "base": "", "url": "", "ece": float(ece)})
            continue
        if len(cells) < 10:
            continue
        try:
            org = ast.literal_eval(cells[3]).get("name", "") if cells[3].startswith("{") else ""
        except (ValueError, SyntaxError):
            org = ""
        params = float(cells[6]) / 1e9 if cells[6] not in ("", "None") else None
        rows.append({"score": float(cells[0]), "slug": cells[1], "name": cells[2], "org": org, "kind": cells[4],
                     "params": round(params, 2) if params else None, "base": cells[7] if cells[7] != "None" else "",
                     "url": cells[8] if cells[8] != "None" else "",
                     "ece": float(cells[9]) if cells[9] not in ("", "None") else None})
    dates = json.loads((ROOT / "notes/board_dates.json").read_text())
    for r in rows:
        repo = "/".join(r["url"].split("/")[3:5]) if r["url"].startswith("https://huggingface.co/") else ""
        r["created"] = dates.get(repo)
    rows.sort(key=lambda r: -r["score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def families() -> list[dict]:
    out = []
    for f in json.loads((ROOT / "notes/families.json").read_text()):
        out.append({"name": f["name"], "maker": f.get("maker", ""), "kind": f["kind"], "first": f.get("first", ""),
                    "dl": f.get("dl") or 0, "likes": f.get("likes") or 0, "main": f.get("main", ""),
                    "arch": f.get("arch", ""), "repos": f.get("n_repos") or 0, "stars": f.get("stars") or 0})
    return out


def arch_stats() -> dict:
    rows = list(csv.DictReader(open(ROOT / "notes/arch/arch_table.tsv"), delimiter="\t"))
    names = {"qwen3_5_text": "Qwen3.5 / 3.8 hybrid", "qwen3_5_moe_text": "Qwen3.5 MoE", "gemma4_text": "Gemma 4",
             "gemma4_unified_text": "Gemma 4", "qwen3": "Qwen3", "qwen2": "Qwen2.5", "modernbert": "ModernBERT",
             "deberta-v2": "DeBERTa", "llama": "Llama", "ministral3": "Ministral 3", "gliner2": "GLiNER2"}
    backbone = collections.Counter()
    for r in rows:
        t = r["text_model_type"]
        if t:
            backbone[names.get(t, "Other")] += 1
    mix = collections.Counter()
    for r in rows:
        if not r["model_type"]:
            continue
        m = r["layer_mix"]
        if m.startswith("linear_attention"):
            mix["DeltaNet + full attention"] += 1
        elif m.startswith("sliding"):
            mix["Sliding + full attention"] += 1
        elif "conv" in m:
            mix["Conv + attention"] += 1
        else:
            mix["Full attention only"] += 1
    lora = collections.Counter(r["lora_r"] for r in rows if r["lora_r"])
    return {"repos": len(rows), "with_config": sum(1 for r in rows if r["model_type"]),
            "noncausal": sum(1 for r in rows if r["is_causal"] == "False"),
            "backbone": backbone.most_common(), "mix": mix.most_common(),
            "lora": sorted(lora.items(), key=lambda kv: int(kv[0]))}


def main() -> None:
    models = json.loads((ROOT / "data/models.json").read_text())
    fam = families()
    brd = board()
    for m in models:
        fsub, bsub = LINKS.get(m["id"], (None, None))
        f = next((f for f in fam if fsub and fsub.lower() in f["name"].lower()), None)
        if m["id"] not in DATED:
            m["released"] = f["first"] if f else None
        if f:
            m["downloads"] = f["dl"]
            m["family"] = f["name"]
        b = next((b for b in brd if bsub and bsub.lower() in (b["url"] + b["name"]).lower()), None)
        if m["id"] == "jev":
            b = next(b for b in brd if b["slug"] == "jev-1.14.0")
        if b:
            m["di"], m["di_rank"], m["ece"], m["di_source"] = b["score"], b["rank"], b["ece"], "board"
            m["board_name"] = b["name"]
        elif m.get("di") is not None:
            m["di_source"] = "self-reported"
            m["di_rank"] = None
    profiles = {p.name: p.read_text() for p in sorted((ROOT / "clones").glob("*.md"))}
    profiles["COMPARISON.md"] = (ROOT / "COMPARISON.md").read_text()
    data = {"models": models, "board": brd, "families": fam, "arch": arch_stats(), "profiles": profiles,
            "snapshot": "Decision Index 0.3.2, snapshot 2026-10-16", "built": "2026-10-19"}
    template = (ROOT / "site/template.html").read_text()
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out = template.replace("/*__DATA__*/null", blob)
    (ROOT / "site/index.html").write_text(out)
    print(f"site/index.html: {len(models)} models, {len(brd)} board rows, {len(fam)} families, "
          f"{len(profiles)} profiles, {len(out) // 1024} KB")


if __name__ == "__main__":
    main()
