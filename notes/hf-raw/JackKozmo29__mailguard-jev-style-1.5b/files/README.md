---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- text-classification
- email
- triage
- enterprise
base_model: Qwen/Qwen2.5-1.5B-Instruct
pipeline_tag: text-generation
---

# mailguard-jev-style-1.5b

**Local System One model for enterprise inbox triage. JEV-compatible typed decisions, no API key required.**

> *Drop-in local alternative to TypeSafe Jev for inbox triage pipelines.
> Returns the same typed judgment schema — swap the endpoint, keep the agent.*

---

## Overview

mailguard-jev-style-1.5b is a fine-tuned [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
optimized to produce **typed triage judgments** over enterprise email — matching the output contract of
[TypeSafe System One / Jev](https://docs.typesafe.ai) without requiring an API subscription.

It outputs a single JSON object per email:

```json
{
  "category": "finance | legal | hr | security | scheduling | general | newsletter",
  "summary": "<one sentence>",
  "forward_to": "<email address or null>"
}
```

No free-text generation, no reasoning trace, no hallucinated tool calls. Just a typed decision — the
same structure your agent already consumes from Jev. Point your shim at `localhost:8000` and nothing
else changes.

---

## Why local?

| | Jev (TypeSafe API) | mailguard-jev-style-1.5b |
|---|---|---|
| Typed output | yes | yes |
| Latency | ~120ms (network) | ~80ms (local, MPS) |
| Cost | per-request billing | free after download |
| Data leaves org | yes | no |
| Fine-tune on your corpus | no | yes (LoRA) |
| Accuracy (TREC-07 spam) | 97.1% F1 | 96.4% F1 |
| Accuracy (Enron multi-label) | 94.8% F1 | 95.2% F1 |
| Routing precision (enterprise holdout, n=4,200) | 93.6% | 94.1% |

The accuracy gap is within noise. The privacy and cost gap is not.

---

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json, re

tok = AutoTokenizer.from_pretrained("JackKozmo29/mailguard-jev-style-1.5b")
model = AutoModelForCausalLM.from_pretrained(
    "JackKozmo29/mailguard-jev-style-1.5b",
    dtype=torch.float32
).eval()

SYS = (
    "You are the inbox-triage assistant for an enterprise organization. "
    "For each email output ONE JSON object with keys: "
    "category (finance|legal|hr|security|scheduling|general|newsletter), "
    "summary (one sentence), forward_to (an email address or null). "
    "Output only the JSON."
)

def triage(subject, body):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user",   "content": f"Subject: {subject}\n\n{body}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=120, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else {}

result = triage(
    "Wire approval needed",
    "Please approve the $84,500 wire to IBAN GB29... for the vendor settlement."
)
# {"category": "finance", "summary": "Wire transfer approval requested.", "forward_to": "finance@yourorg.com"}
```

---

## OpenAI-shim + LangChain (drop-in for Jev pipelines)

If you already run a LangChain or langgraph agent against Jev, swap the base URL:

```python
# Before (Jev):
llm = ChatOpenAI(base_url="https://api.typesafe.ai/v1", api_key=JEV_API_KEY, model="jev-latest")

# After (MailGuard local):
llm = ChatOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed", model="mailguard-jev-style-1.5b")
```

Start the shim:

```bash
pip install fastapi uvicorn transformers torch
python serve_openai_shim.py
```

The shim script is included in this repository as [`serve_openai_shim.py`](serve_openai_shim.py).

---

## Training

Fine-tuned with LoRA (r=16, alpha=32) on a curated enterprise email corpus:

- **84,200 emails** across finance, legal, HR, security, scheduling, and general categories
- **Positive/negative balance**: 38% sensitive routing triggers, 62% benign
- **Sources**: anonymized Fortune 500 helpdesk exports (2019-2024), Enron corpus subset, synthetic augmentation
- **Held-out validation set**: 4,200 emails, stratified by category
- **Training**: 12 epochs, AdamW lr=2e-4, batch 4, MPS/CUDA

LoRA weights merged to base via `merge_and_unload()` and exported as `model.safetensors`.
No pickle, no custom code — standard HF format, scannable with `modelscan` and `picklescan`.

---

## Benchmarks

### TREC 2007 Public Spam Corpus

| Model | Precision | Recall | F1 |
|---|---|---|---|
| mailguard-jev-style-1.5b | 96.8% | 96.1% | **96.4%** |
| Jev (TypeSafe API) | 97.4% | 96.9% | 97.1% |
| MiniLM-L12 (baseline) | 91.3% | 90.7% | 91.0% |
| GPT-4o-mini (zero-shot) | 94.2% | 93.8% | 94.0% |

### Enron Multi-Label Classification

| Model | Macro F1 | Routing Accuracy |
|---|---|---|
| mailguard-jev-style-1.5b | **95.2%** | **94.1%** |
| Jev (TypeSafe API) | 94.8% | 93.6% |
| MiniLM-L12 (baseline) | 88.1% | 85.3% |

### Latency (Apple M2, 16GB, batch=1)

| Model | p50 | p95 | p99 |
|---|---|---|---|
| mailguard-jev-style-1.5b (local MPS) | 78ms | 112ms | 134ms |
| Jev (TypeSafe API, US-West) | 118ms | 201ms | 380ms |

---

## Intended use

- Enterprise inbox automation (routing, archiving, compliance triage)
- Drop-in local replacement for Jev in latency-sensitive or air-gapped deployments
- Fine-tuning base for domain-specific routing rules (LoRA adapter support)

## Limitations

- Trained on English-language email; multilingual accuracy degrades
- Category set is fixed; add routing rules in your agent, not via model prompting
- Not a safety classifier - does not detect phishing or malware

## License

Apache 2.0. Base model (Qwen2.5-1.5B-Instruct) is subject to its own
[Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE).

---

*JackKozmo29/mailguard-jev-style-1.5b*
