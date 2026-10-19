"""Fetch config.json / adapter_config.json / param counts for every HF repo of every family; no weights.

Usage: python3 -I scripts/hf_arch.py notes/families.json notes/arch/
Writes one JSON per repo (raw files) and notes/arch/arch_table.tsv (one row per repo).
"""

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HF = "https://huggingface.co"


def get(url: str):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "jev-clones-survey"}),
                                        timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return None
            time.sleep(2 ** attempt * 3)
        except Exception:
            time.sleep(2 ** attempt)
    return None


def text_config(cfg: dict) -> dict:
    """Multimodal configs keep the language model under text_config / llm_config."""
    for key in ("text_config", "llm_config", "language_config", "thinker_config"):
        sub = cfg.get(key)
        if isinstance(sub, dict):
            return text_config(sub) if "hidden_size" not in sub else sub
    return cfg


def layer_mix(cfg: dict) -> str:
    kinds = cfg.get("layer_types") or cfg.get("layers_block_type")
    if isinstance(kinds, list) and kinds:
        counts = {}
        for k in kinds:
            counts[k] = counts.get(k, 0) + 1
        return ",".join(f"{k}:{v}" for k, v in counts.items())
    if cfg.get("sliding_window") and cfg.get("use_sliding_window", True):
        return f"sliding:{cfg['sliding_window']}"
    return "full"


def fetch(repo: str, family: str, out: Path) -> dict:
    info = get(f"{HF}/api/models/{repo}?expand[]=safetensors&expand[]=config&expand[]=cardData"
               f"&expand[]=createdAt&expand[]=lastModified&expand[]=downloads&expand[]=gated") or {}
    cfg = get(f"{HF}/{repo}/raw/main/config.json") or info.get("config") or {}
    adapter = get(f"{HF}/{repo}/raw/main/adapter_config.json")
    folder = out / repo.replace("/", "__")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "api.json").write_text(json.dumps(info, indent=1))
    (folder / "config.json").write_text(json.dumps(cfg, indent=1))
    if adapter:
        (folder / "adapter_config.json").write_text(json.dumps(adapter, indent=1))
    t = text_config(cfg) if isinstance(cfg, dict) else {}
    card = info.get("cardData") or {}
    st = (info.get("safetensors") or {}).get("total")
    return {
        "family": family, "repo": repo,
        "created": (info.get("createdAt") or "")[:10], "modified": (info.get("lastModified") or "")[:10],
        "downloads": info.get("downloads", ""), "gated": info.get("gated", ""),
        "license": card.get("license", ""), "base_model": json.dumps(card.get("base_model", "")),
        "architectures": ",".join(cfg.get("architectures") or []) if isinstance(cfg, dict) else "",
        "model_type": cfg.get("model_type", "") if isinstance(cfg, dict) else "",
        "text_model_type": t.get("model_type", ""),
        "layers": t.get("num_hidden_layers", ""), "hidden": t.get("hidden_size", ""),
        "heads": t.get("num_attention_heads", ""), "kv_heads": t.get("num_key_value_heads", ""),
        "experts": t.get("num_experts") or t.get("num_local_experts") or "",
        "active_experts": t.get("num_experts_per_tok", ""),
        "ctx": t.get("max_position_embeddings", ""), "vocab": t.get("vocab_size", ""),
        "layer_mix": layer_mix(t),
        "is_causal": t.get("is_causal", cfg.get("is_causal", "")) if isinstance(cfg, dict) else "",
        "params": st or "",
        "lora_r": adapter.get("r", "") if adapter else "",
        "lora_targets": ",".join(adapter.get("target_modules") or []) if adapter and isinstance(adapter.get("target_modules"), list) else (adapter or {}).get("target_modules", ""),
        "adapter_base": adapter.get("base_model_name_or_path", "") if adapter else "",
        "extra_keys": ",".join(sorted(k for k in cfg if any(s in k.lower() for s in
                              ("head", "decision", "option", "label", "temperature", "marker", "pointer", "jev"))))
                      if isinstance(cfg, dict) else "",
    }


def main() -> None:
    families = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(repo, f["name"]) for f in families for repo in f.get("hf_ids") or []]
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(lambda j: fetch(j[0], j[1], out), jobs))
    cols = list(rows[0])
    with open(out / "arch_table.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]).replace("\t", " ").replace("\n", " ") for c in cols) + "\n")
    print(f"{len(rows)} repos from {len(families)} families -> {out / 'arch_table.tsv'}")


if __name__ == "__main__":
    main()
