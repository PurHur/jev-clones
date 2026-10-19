---
license: mit
language: en
library_name: onnx
pipeline_tag: text-classification
base_model: jhu-clsp/ettin-encoder-32m
tags: [decision, classification, calibration, onnx, onnxruntime, modernbert, ettin]
---

# MayaZ-32M

A small decision model. Input: a state (text or JSON), a question and 2–256 options. Output: a calibrated
probability for every option. Three decision types: `choice` (pick one), `noul` (yes/no) and `score` (ordinal).

All options are scored in one encoder pass, side by side, with shared position ids, so option order has no effect
(argmax flips under shuffling: 0.03%). Lists above 64 options go through a tournament.

- Backbone: [`jhu-clsp/ettin-encoder-32m`](https://huggingface.co/jhu-clsp/ettin-encoder-32m) (MIT). 33,877,248 parameters including the decision head.
- Balanced tier and the server default.
- Code, server, browser package and docs: [https://github.com/acl-research/MayaZ](https://github.com/acl-research/MayaZ) (MIT).

## Results

| Metric | Value |
|---|---|
| Held-out test accuracy (2,485 items) | .781 |
| typed-decisions benchmark (2,000 items) | .732 |
| Calibration-holdout accuracy | .785 |
| ECE after per-bucket temperatures (calibration holdout) | all .033 · 2 .047 · 3–5 .049 · 6–20 .051 · 21+ .069 |
| Prompt-injection / jailbreak detection | 0.822 / 0.992 |

## Speed and memory

Measured on the real server under a 10% CPU quota and a 512 MB limit, one ONNX Runtime thread:

| Metric | Value |
|---|---|
| Served file | fp32, 129.5 MB |
| Latency p50 / p95 | 1.50 s / 5.31 s |
| Peak RSS over 200 requests | 353 MB |
| Worst-case request (21 long options, full state), peak RSS in-process | 316 MB with the default token budget of 1,024 |
| Browser (ONNX Runtime Web, WASM, 1 thread) p50 | ~0.3 s |

## Files

| File | Use |
|---|---|
| `mayaz_fp32.onnx` | fp32, exact export (served) |
| `mayaz_int8.onnx` | weight-only INT8 (MatMulNBits, 8-bit, block 32) |
| `tokenizer.json` | Ettin tokenizer, unmodified (MIT) |
| `temperatures.json` | Per (type × option-count bucket) temperatures, fitted on the calibration split |
| `config.json` | Architecture and renderer settings |

The input format (rendering, option markers, truncation) is defined in the GitHub repository:
`src/mayaz/model/render.py` (Python) and `web/src/render.js` (browser). Use those; do not re-implement it.

## How to use

```bash
pip install "git+https://github.com/acl-research/MayaZ" huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
from mayaz.inference import Model

repo = "acl-research/MayaZ-32M"
model = Model(hf_hub_download(repo, "mayaz_fp32.onnx"),
              hf_hub_download(repo, "tokenizer.json"),
              hf_hub_download(repo, "temperatures.json"),
              max_tokens=1024)  # per-pass token budget; keeps peak memory near 316 MB

r = model.decide("choice", "My card was charged twice for the same order.", "What does the customer want?",
                 [("refund", "money back for the duplicate charge"), ("cancel_order", None), ("greeting", None)])
print(r["choice"], r["confidence"], r["probabilities"])
```

As an HTTP service: `docker build -f server/Dockerfile --build-arg MAYAZ_MODEL=mayaz-32m .` from the repository, or
the Render blueprint in `server/render.yaml`. In the browser: `web/` (ONNX Runtime Web).

## Limitations

- Context is 512 positions of state plus the question and options. Longer states are truncated from the end.
- Yes/no questions on paraphrase-style or multi-label tasks tend towards a default answer instead of weighing the
  evidence.
- On text unrelated to the options the model still picks one, often with moderate confidence. Use the confidence
  (the server's `abstain` flag) or add a "none of these" option.
- Typos and slang lower accuracy somewhat without a matching drop in confidence.
- Subjective tasks (emotions, review stars) stay near 0.5 accuracy.
- Not a security control on its own.
- Calibration error in the 3–5 and 21+ buckets is above 0.05. There is no int8-embedding build for this tier.

## Training

Soft-label training on decision items rendered from permissively licensed public datasets (train splits only),
with soft labels from open-weight teacher models (Apache-2.0):

| Dataset | Licence |
|---|---|
| `LocalLLaMA/typed-decisions` | Apache-2.0 |
| `mteb/banking77` | MIT |
| `clinc/clinc_oos (plus)` | CC BY 3.0 |
| `mteb/amazon_massive_intent (en)` | CC BY 4.0 |
| `google-research-datasets/go_emotions` | Apache-2.0 |
| `fancyzhx/dbpedia_14` | CC BY-SA 3.0 |
| `davidheineman/consumer-finance-complaints-large` | Apache-2.0 |
| `stanfordnlp/snli` | CC BY-SA 4.0 |
| `SetFit/amazon_reviews_multi_en` | Apache-2.0 |
| `google/boolq` | CC BY-SA 3.0 |
| `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | CC BY 4.0 |
| `deepset/prompt-injections (train split)` | Apache-2.0 |
| `jackhhao/jailbreak-classification` | Apache-2.0 |

Weights: MIT.

## Citation

```bibtex
@software{mayaz_2026,
  title        = {MayaZ: small calibrated decision models},
  author       = {Srivastava, Ekansh},
  organization = {ACL Research},
  year         = {2026},
  version      = {1.0.0},
  url          = {https://github.com/acl-research/MayaZ}
}
```
