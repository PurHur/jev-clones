---
license: apache-2.0
bases:
  - Qwen/Qwen2.5-0.5B-Instruct
tags:
  - reranker
  - memory-gating
  - distillation
  - lora
  - system-one
---

# Jev-Gate Student B — LoRA Memory-Relevance Judge

A LoRA adapter (r=16, α=32, on q_proj/v_proj) on `Qwen/Qwen2.5-0.5B-Instruct`, distilled from the Jev typed-judgment API into a compact local judge for agent-memory gating.

**What it does:** given a query and a candidate memory passage, outputs P(relevant) as the calibrated `yes` probability read from the final-token logits of `yes` vs `no`. Used to filter which vector-recalled memories get injected into agent context (vector recall → cross-encoder band gate → this judge for gray-band cases).

**Training:** LoRA, 1 epoch, LR 1e-4, on rows sampled from [SargeDev/jev-distill-corpus](https://huggingface.co/datasets/SargeDev/jev-distill-corpus) — paired relevance judgments (Jev graded 0–7 expected-value scores + binary labels) distilled from typed judgment calls by a larger teacher model. Prompt format:

```
Memory: {passage ≤600 chars}
Query: {query}
Question: Is this memory relevant for answering the query? Answer yes or no with confidence.
```

## Full 10k Held-Out Benchmark (Sep 20, 2026)

A 10,000-row held-out test set (`test_10k.jsonl`, sampled from the same public-dataset manifest pool as the corpus, seed 42, **hard-disjoint** from all 142,909 train/dev/test rows by `(query, text[:600])` key). Gold labels = 32B teacher scores rescaled /7. All three judges scored with the **identical prompt** above, yes/no logit softmax.

| Judge | MAE | Pearson r | Binary agree @0.5 | Gray-band MAE (n=1,669) | Latency |
|---|---|---|---|---|---|
| **Jev 1.13 (API, OpenRouter)** | **0.187** | **0.787** | **84.4%** | **0.205** | ~1,007 ms |
| **Student B (this adapter, local)** | 0.219 | 0.709 | 81.7% | 0.275 | 23 ms (batch-32, RTX 3060) |
| Vanilla Qwen2.5-0.5B-Instruct | 0.498 | −0.005 | 44.4% | 0.329 | 22 ms |

### Distillation fidelity (the key number)

**Student B vs its own teacher (Jev 1.13): 86.4% agreement, Pearson r = 0.824, MAE 0.144** across all 10k rows. The 1.1M-parameter LoRA reproduces a production System One model's judgments within ~3 points of binary agreement, at **43× lower latency** and zero API cost.

Where they disagree (1,364 items), Jev is right vs gold 60% of the time — a real but modest teacher edge. The gray band (teacher y ∈ [0.3, 0.7]) is where both models live in production (the band gate escalates only these cases); student gray-band MAE 0.275 vs teacher 0.205.

### Small test slices (earlier evals)

| Eval | Student B | Vanilla 0.5B | Student A |
|---|---|---|---|
| Original test (5,605) | 0.148 / 0.836 / 86.0% | 0.443 / −0.007 / 51.8% | 0.112 / 0.886 / 87.6% |
| Held-out (n=60) | 0.187 / 0.791 / 90.0% | 0.536 / −0.067 / 38.3% | 0.273 / 0.445 / 60.0% |

### Takeaways

1. **Distillation works**: the LoRA captures most of the teacher's judgment; the win is from distillation, not model size (vanilla 1.5B barely helps).
2. **Gray band is the battleground**: both models are most uncertain there; that's exactly where the cascade routes escalation.
3. **Latency economics**: local student ≈ sub-second for a 25-candidate cascade judgment (batched); API teacher ≈ 1 s/item — fine for referee calls, wrong for every-candidate scoring.
4. **10k-set construction**: seeded sample from fresh (never-trained) manifest rows, disjoint by content key — reproducible via seed 42.

## How the pipeline works (production)

```
Qdrant vector recall (25 candidates)
  → bge-reranker-base int8 ONNX cross-encoder (local band gate: keep ≥0.08 / drop ≤0.01)
  → gray band [0.01, 0.08] → Student B (this adapter) judges locally
  → final TOP-K injection (default 8)
Fail-open everywhere: any stage error → keep everything (no forgetting).
```

Every decision is logged with (vector_score, local_score, student_score, verdict) to a local JSONL log — kept private (closed-source project only; never published) and used as retraining corpus for future internal student iterations.

## What's published

| Repo | Contents |
|---|---|
| [SargeDev/jev-gate-student-b](https://huggingface.co/SargeDev/jev-gate-student-b) | This adapter (adapter_config.json + adapter_model.safetensors + tokenizer) |
| [SargeDev/jev-distill-corpus](https://huggingface.co/datasets/SargeDev/jev-distill-corpus) | 148,160 rows of paired relevance judgments (teacher-scored) |

## Future directions

1. **Expand the training corpus**: the current training set was 60k rows from the pre-production corpus; internal-only private runs can additionally mine local production logs for real query-distribution coverage. (Production data stays private — never published.)
2. **Scale the student**: the same recipe on **Qwen2.5-1.5B-Instruct** (LoRA r=16–32) should close most of the remaining ~3-point gap to Jev; a **Qwen3-4B / Qwen3-8B LoRA** is the next rung — expect MAE < 0.15 and agreement > 90% if the trend holds. All still local-friendly (1.5B ≈ 3 GB bf16).
3. **Hybrid referee cascade**: keep Student B local for the gray band; route only true near-ties (student score ∈ [0.45, 0.55]) to the Jev API for a final call. Expected: teacher accuracy at ~2% of the call volume.
4. **Multi-judge ensemble**: Jev API + Student B + bge-reranker — average scores for the injected set, use disagreement as an uncertainty signal.
5. **Merge to GGUF**: merge the LoRA into base weights for llama.cpp single-file serving (removes PeftModel dependency; warm path already ~1 s, GGUF would make it ~50 ms on CPU-only hosts).

## License

Apache-2.0 (matches base model).