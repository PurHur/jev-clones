---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
pipeline_tag: text-generation
tags: [system-one, kenning, kenning-xl, decision-model, calibrated, experimental]
---

# kenning-xl-v0.6 (experimental)

Kenning-XL is a **System One decision model built on a decoder**. It answers the same wire format as Kenning (`POST /v1/systemone`; typed questions → calibrated probabilities, no text), but instead of a cross-encoder it uses a small decoder (Qwen3-1.7B) with a **constrained answer-token readout**: it renders (state, question) and reads the next-token distribution over the answer tokens in one pass, so the full language model reasons over the whole state. An optional **deliberate mode** generates a short reasoning trace, then reads the answer — for numeric, policy and table decisions a single pass can't do.

This is the experimental v0.6 engine. The published cross-encoder (`kenning-large-v0.5`) is still the fast default; Kenning-XL is complementary — **strongest on agent and structured-record decisions**.

## How it's served

It is served by **SystemOne Builder's `kenning-xl` engine** (`systemone_builder.kenning.xl_serve`), which answers one-pass and **escalates to deliberate mode only for low-confidence, short/structured questions** — so the extra latency is spent only where reasoning helps. See the [design notes](https://github.com/systemonedev/systemone-builder/blob/main/docs/kenning-xl-design.md). It is not a plain `transformers` classifier; load it through the builder.

## Training

- Base: `Qwen/Qwen3-1.7B` (Apache-2.0), LoRA fine-tuned (merged), bf16, 1,536-token context.
- **Readout distillation** on the v0.5 structured corpus (records, tables, agent steps, logs, text, safety; exact or public labels), plus **gold-trace deliberate training**: the builder's rule generators compute each label, so they emit the exact arithmetic/threshold/membership reasoning chain for free (`kenning/traces.py`), and the model learns to reason-then-answer on those.
- Not trained on any TypeSafe output.

## Benchmark (general suite, per-family mean, 1,328 held-out items)

As served (confidence + size-gated escalation), vs the cross-encoder v0.5 and — for comparison only — Clef (`clef-flash`) and Jev (`jev-latest`):

| Family | kenning-xl v0.6 | kenning v0.5 | Clef | Jev |
|---|---|---|---|---|
| **Macro** | **0.688** | 0.653 | 0.791 | 0.830 |
| Agent (tool calls, task completion) | **0.860** | 0.727 | 0.793 | 0.900 |
| Conversation | 1.000 | 0.979 | 1.000 | 1.000 |
| Records (rules over JSON) | 0.663 | 0.587 | 0.857 | 0.921 |
| Tables | 0.660 | 0.520 | 0.860 | 0.940 |
| Text | 0.752 | 0.756 | 0.841 | 0.834 |
| Quality (helpful, correct) | 0.439 | 0.479 | 0.447 | 0.498 |
| Logs | 0.440 | 0.520 | 0.740 | 0.720 |

**Beats Clef on agent decisions**, ties conversation, and the gold-trace deliberate reasoning more than doubles the hardest arithmetic task (daily-limit checks 0.42 → 0.81, transferring to held-out tasks). It does **not** match Clef overall, and currently trails even the cross-encoder on logs, quality and text — deliberate reasoning isn't trace-trained for log-counting yet. Pick the engine for your task: Kenning-XL for agent/structured decisions, the v0.5 cross-encoder for text/safety/logs.

## Licence

Weights under the **Apache License 2.0** (merged LoRA adapter on Apache-2.0 Qwen3-1.7B). Training data is permissively licensed or generated; see NOTICE.md. Kenning implements a wire format compatible with TypeSafe AI's System One API; not affiliated with or endorsed by TypeSafe AI, and not trained on its outputs.
