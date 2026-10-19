---
license: apache-2.0
library_name: gguf
base_model: johannhartmann/Qwen3.5-2B_qlora_decisions-de
base_model_relation: quantized
datasets:
- johannhartmann/decisions-de
language:
- de
tags:
- gguf
- llama.cpp
- clef
- systemone
- decision-model
- structured-decisions
- qwen3.5
- unsloth
- german
---

# Qwen3.5-2B_qlora_decisions-de-GGUF

GGUF files of [`johannhartmann/Qwen3.5-2B_qlora_decisions-de`](https://huggingface.co/johannhartmann/Qwen3.5-2B_qlora_decisions-de),
a small German Clef decision model (Qwen3.5-2B with a joint-schema head, trained on
[`johannhartmann/decisions-de`](https://huggingface.co/datasets/johannhartmann/decisions-de)), for
llama.cpp's decision server endpoint `/v1/systemone`. Given a state and typed questions (`choice`,
`noul`, `score`), it returns a probability for every option of every question in one prefill, in the
System One / Decision API format. There is no text generation.

| File | Size | Notes |
|---|---:|---|
| `model-Q8_0.gguf` | 2.0 GB | recommended, closest to PyTorch |
| `model-Q6_K.gguf` | 1.6 GB | |
| `model-Q4_K_M.gguf` | 1.3 GB | k-quants can shift probabilities and flip close answers |
| `model-BF16.gguf` | 3.8 GB | unquantized |
| `mmproj-Q8_0.gguf`, `mmproj-BF16.gguf` | 0.4 / 0.7 GB | vision tower, only needed for `images`/`videos` in the request |
| `export.json` | | Unsloth Studio's index of the files |

Exported with Unsloth's decision GGUF exporter and the llama.cpp Clef converter. The per-type
temperatures Unsloth fitted after training (noul 1.13, choice 0.88, score 0.67) are stored in the GGUF
metadata, so llama.cpp serves calibrated probabilities.

## Usage

Needs llama.cpp **b11443 or newer** (tested with b11541).

```bash
llama-server -m model-Q8_0.gguf --mmproj mmproj-Q8_0.gguf -ngl 99 \
  -np 2 -c 32768 -b 16384 -ub 16384 --port 8080
```

The whole request is processed in one batch, so `-b`/`-ub` must be at least the longest request in
tokens. With the default 512, requests over 512 tokens fail with HTTP 500 ("input is too large to
process"). `-c` is shared by the `-np` slots.

```bash
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "decider",
  "state": {"rechnung": {"betrag": 1250.0, "status": "seit 30 Tagen überfällig"}},
  "questions": {
    "status": {"type": "choice", "instructions": "Welchen Status hat die Rechnung?",
               "criteria": {"paid": "bezahlt", "overdue": "überfällig", "draft": "Entwurf"}},
    "gross":  {"type": "noul", "instructions": "Ist der Betrag höher als 1000 EUR?"},
    "dring":  {"type": "score", "instructions": "Wie dringend ist eine Mahnung?",
               "criteria": ["gar nicht", "etwas", "sehr"]}
  }}'
```

```json
{"answers": {
  "status": {"type": "choice", "choice": "overdue", "probabilities": {"draft": 0.0002, "overdue": 0.9994, "paid": 0.0005}, ...},
  "gross":  {"type": "noul", "noul": 0.9991},
  "dring":  {"type": "score", "score": 1.70, "probabilities": {"0": 0.042, "1": 0.219, "2": 0.739}, ...}},
 "usage": {"input_tokens": 339, "output_tokens": 0}}
```

## Results

Q8_0 through `llama-server /v1/systemone`, compared with the bf16 PyTorch model:

| Eval set | Decisions | Q8_0 accuracy | PyTorch accuracy | Q8_0 ECE | PyTorch ECE* |
|---|---:|---:|---:|---:|---:|
| `decisions-de` validation | 1,215 | 0.845 | 0.842 | 0.023 | 0.021 |
| `typed-decisions-de` test | 2,000 | 0.509 | 0.504 | 0.139 | 0.164 |

\* PyTorch with the same per-type temperatures that the GGUF carries.

The other quantizations were not evaluated. See the
[model card of the PyTorch model](https://huggingface.co/johannhartmann/Qwen3.5-2B_qlora_decisions-de)
for training details, the German Decision Index sample, and limitations: the model is a broad
generalist and weak on the four business workflows of `typed-decisions-de`.

## License

Apache-2.0, like the base model Qwen3.5-2B. The training data keeps per-source upstream licenses; see
the [dataset card](https://huggingface.co/datasets/johannhartmann/decisions-de).
