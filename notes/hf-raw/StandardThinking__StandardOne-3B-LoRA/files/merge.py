#!/usr/bin/env python3
"""Merge this LoRA adapter into its pinned base checkpoint for release.

Downloads the base model at the pinned revision from Hugging Face, merges
the adapter with PEFT, and writes HF-format shards + tokenizer/processor
files plus a tensor-level report (which parameters changed).

Usage:
    python merge.py --out /path/to/merged
"""
import argparse
import hashlib
import json
import os
import shutil
import time

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

BASE_REPO = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
BASE_REVISION = "b6d637bef2393152b3da2b2fde72eecdee30557e"
ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.time()

snapshot = snapshot_download(repo_id=BASE_REPO, revision=BASE_REVISION)
base = Mistral3ForConditionalGeneration.from_pretrained(
    snapshot, torch_dtype=torch.bfloat16, device_map="cpu"
)
ref = {n: p.detach().clone() for n, p in base.named_parameters()}
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
merged = model.merge_and_unload()

changed, unchanged = [], 0
for n, p in merged.named_parameters():
    r = ref.get(n)
    if r is None:
        changed.append((n, "NEW"))
        continue
    if r.shape != p.shape or not torch.equal(r, p):
        changed.append((n, float((p.float() - r.float()).abs().max())))
    else:
        unchanged += 1
bad = [
    n
    for n, _ in changed
    if (
        "vision_tower" in n
        or "multi_modal_projector" in n
        or not any(
            k in n
            for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        )
    )
]

os.makedirs(a.out, exist_ok=True)
merged.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
for f in (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tekken.json",
    "chat_template.jinja",
    "processor_config.json",
    "generation_config.json",
    "SYSTEM_PROMPT.txt",
    "params.json",
):
    src = os.path.join(snapshot, f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(a.out, f))
try:
    AutoProcessor.from_pretrained(snapshot).save_pretrained(a.out)
except Exception as e:
    print("processor save skipped:", e)

shas = {}
for f in sorted(os.listdir(a.out)):
    p = os.path.join(a.out, f)
    if os.path.isfile(p):
        shas[f] = hashlib.sha256(open(p, "rb").read()).hexdigest()

report = {
    "base_repo": BASE_REPO,
    "base_revision": BASE_REVISION,
    "adapter": ADAPTER_DIR,
    "adapter_sha256": hashlib.sha256(
        open(os.path.join(ADAPTER_DIR, "adapter_model.safetensors"), "rb").read()
    ).hexdigest(),
    "changed_params": len(changed),
    "unchanged_params": unchanged,
    "changed_outside_lm_projections": bad,
    "max_abs_delta": max((d for _, d in changed if isinstance(d, float)), default=0.0),
    "changed_sample": changed[:8],
    "file_sha256": shas,
    "seconds": round(time.time() - t0, 1),
}
json.dump(report, open(os.path.join(a.out, "MERGE_REPORT.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in report.items() if k != "file_sha256"}, indent=1))
