# -*- coding: utf-8 -*-
"""ZTC-Judge-27B — curved probe, the one that produces the leaderboard number.

Two probes ship with this model and they are not interchangeable:

    ztc_probe.npz         linear, 59 KB   -> one dot product
    ztc_curve_probe.npz   curved, 5.3 MB  -> 256 anchors, RBF kernel

The leaderboard figure (0.7282, per-domain leave-one-domain-out) is produced by the
curved probe. Use this file if you want to reproduce that number.

Both read the same input: the final-layer hidden state at the last position, 5,120-d,
from a single forward pass with zero generated tokens.
"""
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO = "FINAL-Bench/ZTC-Judge-27B"
TEMPLATE = ("다음은 어떤 문제와 그에 대한 답변이다. 이 답변이 옳은지 판단하라.\n\n"
            "[문제]\n%s\n\n[답변]\n%s\n\n이 답변은 옳은가?")


def hidden(question, answer, model, tok, max_length=2048):
    """One forward pass. No tokens are generated."""
    text = TEMPLATE % (question.strip(), answer.strip())
    b = tok([text], return_tensors="pt", truncation=True, max_length=max_length)
    dev = next(model.parameters()).device
    with torch.no_grad():
        ids = b["input_ids"].to(dev)
        am = b["attention_mask"].to(dev)
        h = model(input_ids=ids, attention_mask=am).last_hidden_state
    return h[0, int(am.sum()) - 1].float().cpu().numpy().astype(np.float64)


def score_curved(vec, probe):
    """Curved readout: distance to 256 anchors through an RBF kernel, then one dot product.

    Higher means more likely correct. The value is a ranking signal, not a calibrated
    probability — pick a threshold from your own review budget.
    """
    z = (vec - probe["mu"]) / probe["sd"]
    a = probe["anchors"].astype(np.float64)
    d2 = np.sum(a * a, 1) + float(z @ z) - 2.0 * (a @ z)
    k = np.exp(-np.maximum(d2, 0.0) / (float(probe["med"]) * float(probe["gamma"])))
    return float(k @ probe["alpha"].astype(np.float64))


def score_linear(vec, probe):
    """Linear readout: a single dot product, 59 KB on disk."""
    return float(((vec - probe["mu"]) / probe["sd"]) @ probe["w"])


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download, snapshot_download

    path = snapshot_download(REPO)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # 27B in bf16 is ~54 GB. Quantising changes the hidden state and silently breaks
    # the probe, so keep the precision and let the overflow sit on CPU instead.
    model = AutoModel.from_pretrained(
        path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="auto", max_memory={0: "44GiB", "cpu": "72GiB"}).eval()

    curve = dict(np.load(hf_hub_download(REPO, "ztc_curve_probe.npz")))

    q = "By what mechanism do statins lower LDL cholesterol?"
    a = ("Statins bind directly to circulating LDL particles and mark them for "
         "clearance by macrophages.")
    v = hidden(q, a, model, tok)
    print("curved score: %.3f  (lower = more likely wrong)" % score_curved(v, curve))
