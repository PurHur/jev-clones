---
license: apache-2.0
language: [th, en]
base_model: iapp/OpenThai-SystemOne
base_model_relation: finetune
library_name: gguf
pipeline_tag: text-classification
tags: [system-one, decision-model, thai, qwen3.5, gguf, ollama, llama.cpp]
---

# OpenThai-SystemOne v0.3 for Ollama

[OpenThai-SystemOne](https://huggingface.co/iapp/OpenThai-SystemOne) is an open Thai + English **System One decision
model** (0.8B, Apache-2.0). It does not generate text: given a *state* (text or JSON) and typed questions it returns
probabilities: `choice` between named options, `noul` (yes/no) and `score` on an ordered scale.

This repo is its **Ollama build** for Ollama's System One API (`POST /v1/systemone`, **Ollama ≥ 0.35**), the same API
Ollama serves Nimble and Tev1 with. Everything runs on your machine; no API key.

```bash
ollama pull iapp/openthai-systemone                       # ollama.com: 0.8b (= 0.8b-q8_0), 0.8b-q4_K_M, 0.8b-bf16
ollama pull hf.co/iapp/OpenThai-SystemOne-Ollama:Q8_0     # the same files from this repo
```

The curl example below uses the hf.co name; with the ollama.com pull, use `"model": "iapp/openthai-systemone"`.

```bash
curl http://localhost:11434/v1/systemone -d '{
  "model": "hf.co/iapp/OpenThai-SystemOne-Ollama:Q8_0",
  "state": {"ticket": "ลูกค้าแจ้งว่าโดนหักเงินซ้ำสองครั้ง ขอเงินคืนด่วน โทรมาสามรอบแล้ว"},
  "questions": {
    "department": {"type": "choice", "instructions": "ทีมใดควรรับผิดชอบ",
                   "criteria": {"billing": "การเงิน/ค่าบริการ", "technical": "ระบบใช้งานไม่ได้", "sales": null}},
    "frustration": {"type": "score", "instructions": "ลูกค้าหงุดหงิดแค่ไหน",
                    "criteria": ["ใจเย็น", "หงุดหงิดแต่สุภาพ", "โกรธมาก"]},
    "refund_requested": {"type": "noul", "instructions": "ลูกค้าขอเงินคืนอย่างชัดเจนหรือไม่"}
  }
}'
```

```json
{"model": "hf.co/iapp/OpenThai-SystemOne-Ollama:Q8_0",
 "answers": {
   "department": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9515, "technical": 0.0334, "sales": 0.0151}, "confidence": 0.7959},
   "frustration": {"type": "score", "score": 1.8771, "legend": {"0": "ใจเย็น", "1": "หงุดหงิดแต่สุภาพ", "2": "โกรธมาก"},
                   "probabilities": {"0": 0.0216, "1": 0.0797, "2": 0.8987}, "confidence": 0.6537},
   "refund_requested": {"type": "noul", "noul": 0.9805}},
 "usage": {"input_tokens": 741, "output_tokens": 4}}
```

(Probabilities rounded. The whole request is part of every question's prompt, so the same question can score slightly
differently next to other questions.)

## How this build differs from the main repo

Ollama does not run OpenThai-SystemOne's 256-slot decision head. For each question it renders one chat prompt (the whole
request as JSON plus `Requested field: "<name>"`, with the model's Qwen3.5 chat template and thinking off) and reads the
next-token probabilities of the answer letters `A`–`Z`. The weights here are therefore **v0.3 fine-tuned for that
prompt**:

* 3,000 steps (192k questions) on the v0.3 training mix, rendered exactly as Ollama 0.35 renders them (a byte-exact port
  of Ollama's `decision/systemone.go`, checked against Go) and tokenized by llama.cpp, as Ollama's runner does.
* Loss = cross-entropy over the question's candidate letters only, which is the softmax Ollama computes.
* One temperature, fitted on held-out records, is folded into the final norm (Ollama always scores at temperature 1).
* The GGUF files are a plain Qwen3.5 (`qwen35`) text model with tied embeddings; the Modelfile / `system` file sets the
  system prompt the model was trained with and `num_ctx 8192`.

Compared with the main repo's own API (`pip install openthai-systemone`): Ollama allows **2–26 options** per question
(the main API: 255), runs one prompt per question (the shared prefix is cached), and has no order-invariant mode and no
abstain answer.

## Evaluation (through Ollama 0.35)

All columns were run by us through Ollama's `/v1/systemone` on the same records: the first 800 of each set, keeping only
records whose questions have ≤ 26 options (Ollama's limit; drops banking77 and the 60-way MASSIVE-th intents). The first
column is the original v0.3 weights with their 256-slot head on the same records, for reference. `choice` / `noul` =
accuracy, `score` = exact level. Harness: `scripts/25_competitor_eval.py --model ollama:<name> --max-options 26` in the
[GitHub repo](https://github.com/iapp-technology/openthai-systemone).

**Public 13 subsets (Bespoke Nimble's public benchmark)**

| set (type, n) | v0.3, main repo's API | **this repo, Q8_0** | this repo, Q4_K_M | Tev1 0.8B | Tev1 4B | Nimble 9B |
|---|---|---|---|---|---|---|
| aegis2 (noul, n=250) | **83.2** | 82.0 | 81.2 | 60.8 | 80.8 | **83.2** |
| boolq (noul, n=300) | 79.7 | 79.7 | 81.0 | 77.7 | 85.3 | **86.3** |
| civil_comments (noul, n=300) | **79.0** | 76.0 | 77.3 | 76.7 | 73.3 | 78.0 |
| helpsteer2 (score, n=250) | **41.6** | 39.2 | 40.0 | 33.2 (1 err) | 36.8 (1 err) | 33.6 |
| massive-de-DE (choice, n=350) | **88.6** | 88.0 | 87.4 | 69.7 | 83.1 | 83.1 |
| massive-en-US (choice, n=350) | **89.1** | 88.9 | 88.6 | 78.9 | 85.4 | 84.0 |
| multinli (choice, n=299) | 88.6 | 86.0 | 84.3 | 75.6 | **92.0** | 90.0 |
| paws (noul, n=250) | **94.0** | 92.8 | 92.8 | 65.2 | 82.4 | 74.0 |
| pubmedqa (choice, n=250) | 64.0 | 65.6 | 65.6 | 61.2 | 74.4 | **77.2** |
| squad2 (noul, n=299) | **89.3** | 89.0 | 86.3 | 70.9 | 76.3 | 74.2 |
| summeval-consistency (score, n=144) | 75.0 | 76.4 | 74.3 | **84.0** | 79.9 | 81.9 |
| summeval-relevance (score, n=240) | 21.7 | 28.7 | 27.9 | 13.8 | **50.0** | 48.3 |
| vitaminc-dev (choice, n=599) | 72.5 | 74.1 | 75.0 | 68.8 | 74.3 | **79.0** |
| **macro** | **74.3** | **74.3** | **74.0** | **64.3** | **74.9** | **74.8** |


**Thai sets (8 sets, 12 rows; wisesight and SIB-200 held out of training)**

| set (type, n) | v0.3, main repo's API | **this repo, Q8_0** | this repo, Q4_K_M | Tev1 0.8B | Tev1 4B | Nimble 9B |
|---|---|---|---|---|---|---|
| contrastive_th (choice, n=296) | 80.7 | 83.8 | 82.4 | 75.7 | 92.2 | **93.2** |
| contrastive_th (noul, n=248) | 83.5 | 84.7 | 84.3 | 79.0 | 94.8 | **97.2** |
| contrastive_th (score, n=56) | 78.6 | 78.6 | 75.0 | 60.7 | **91.1** | 83.9 |
| massive_th (choice, n=588) | **94.6** | 92.7 | 92.3 | 71.6 | 88.8 | 90.8 |
| prachathai (choice, n=413) | **98.5** | 97.8 | 97.6 | 56.7 | 61.7 | 61.5 |
| prachathai (noul, n=1568) | 93.4 | **95.2** | 95.0 | 68.8 | 74.2 | 66.3 |
| sib200_th (choice, n=204) | 77.9 | 78.9 | 76.5 | 83.3 | 86.3 | **88.7** |
| wisesight (choice, n=800) | 49.0 | **49.9** | 49.8 | 40.8 | 48.0 | 48.5 |
| wongnai (score, n=800) | **64.5** | 64.0 | 62.3 | 38.5 | 56.1 | 52.1 |
| xlam_tools (choice, n=800) | **99.4** | **99.4** | **99.4** | 90.1 | 97.1 | 97.2 |
| xnli_th (choice, n=800) | **79.8** | 79.2 | 78.5 | 66.9 | 76.5 | 76.0 |
| xnli_th (noul, n=800) | **86.8** | 85.9 | 85.6 | 20.9 | 38.4 | 84.0 |
| **macro** | **82.2** | **82.5** | **81.6** | **62.7** | **75.4** | **78.3** |

**Latency** (median end to end through Ollama, one model loaded, idle H100, Thai requests): 1 question 23 ms (Q8_0), 23 ms
(Q4_K_M), 26 ms (BF16); 3 questions 136 / 127 / 142 ms. Same setup: Tev1 0.8B 22 / 102 ms, Tev1 4B 66 / 406 ms,
Nimble 9B 69 / 392 ms. Ollama runs one prompt per question, reusing the shared prefix.

BF16 scores the same as Q8_0 (public 74.3, Thai 82.5).

## Files

| file | size | public / Thai macro through Ollama |
|---|---|---|
| `OpenThai-SystemOne-v0.3-Ollama-Q8_0.gguf` | 812 MB | 74.3 / 82.5 (recommended) |
| `OpenThai-SystemOne-v0.3-Ollama-Q4_K_M.gguf` | 529 MB | 74.0 / 81.6 |
| `OpenThai-SystemOne-v0.3-Ollama-BF16.gguf` | 1517 MB | 74.3 / 82.5 |
| `Modelfile`, `Modelfile.Q4_K_M`, `Modelfile.BF16` | – | for `ollama create` from a local file |
| `system`, `params` | – | system prompt and `num_ctx 8192`, applied by `ollama pull hf.co/...` |

`Modelfile` builds the same model from a local file (`ollama create openthai-systemone -f Modelfile`); `system` and
`params` are what `ollama pull hf.co/...` applies.

## Limits

* Up to 26 options per question (Ollama). For more options, bucket them, or use the main repo's API (255 options).
* Not a chat model: `ollama run` will produce text, but the model was only trained to answer `/v1/systemone` prompts.
* A small model: use `confidence` and route low-confidence decisions to a bigger model or a person.

## License

Apache-2.0. Built by iApp Technology / OpenThai on Qwen3.5-0.8B-Base (Apache-2.0). Not affiliated with TypeSafe AI,
Bespoke Labs or Together AI.
