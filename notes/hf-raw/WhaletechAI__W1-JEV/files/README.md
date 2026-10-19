---
tags:
  - diffusion
  - pytorch
  - structured-prediction
  - w1-jev
---

# W1-JEV

### Millisecond decisions. Less than 1¢ in GPU cost per 1,000 requests.

**33 ms median latency · $0.00567 per 1,000 requests · 30 emails per second**

![W1-JEV demo](./W1-jev.gif)

[中文模型介绍与评测报告](./report.zh.md)

W1-JEV is a **4B-class language diffusion model for structured decision tasks**, built on our W1 DLM. We applied post-training output-format alignment to reproduce JEV-style decision tasks, targeting decision workflows and high-volume automation.

On **231 matching public adaptation questions**, W1-JEV answers **184 correctly**, reaching **92% of the accuracy of the Jev 1.13.0 public record**. Our reported cost comparison places GPU occupancy cost at approximately **one seventh of the reference setup**.

The measured run completed all 231 requests in **10.8373 seconds**, with **100% strict output-format validity**. At a GPU rate of $0.435/hour, the allocated cost is **$0.00567 per 1,000 requests**—about **0.57 US cents**.

![W1-JEV and Jev 1.13.0 comparison](./assets/w1-jev-comparison.png)

## Built for high-volume decisions

When every input requires a decision, the latency and cost of each request add up. W1-JEV focuses on that repeated step to make batch decisions faster and more economical.

- **Short waits.** The reported run achieved **33.13 ms p50** end-to-end request latency and **21.32 requests/second** throughput.
- **Low GPU cost per decision.** Measured GPU occupancy cost was **$0.00567 per 1,000 requests** at the stated hourly rate.
- **Outputs that fit the workflow.** All **231 completed requests** passed strict format validation in this run.
- **A local deployment path.** The release includes a pure PyTorch model, a W1-specific runtime, a local HTTP service, and an executable choice-question client.

## Accuracy: 92% of the reference accuracy

Results below cover **231 matching questions from the public adaptation set**. W1-JEV results are publisher-reported; Jev 1.13.0 results come from the public record.

| Metric | W1-JEV — publisher-reported | Jev 1.13.0 — public record |
| --- | ---: | ---: |
| Correct answers | **184 / 231** | 200 / 231 |
| Raw accuracy | **79.65%** | 86.58% |
| Simple / easy subset | **48 / 48** | 48 / 48 |

W1-JEV matches the reference on the easy subset. Across the full matching set, it answers 16 fewer questions correctly, a **6.93 percentage-point accuracy gap**.

**“92%” is a relative accuracy ratio:** `(184 / 231) / (200 / 231) = 92%`. W1-JEV's absolute accuracy is **79.65%**; the ratio is specific to this set.

## Performance and GPU cost

This run used a single **NVIDIA GeForce RTX 5090**, with **2 concurrent clients**, one GPU worker and **maximum batch size 2**. The run comprised **116 batches** (average batch **1.99**), with up to **5 ms batching wait**.

| Metric | Reported result |
| --- | ---: |
| End-to-end request latency, p50 / p95 | **33.13 / 374.02 ms** |
| Throughput | **21.32 requests/s** |
| Wall-clock duration | **10.8373 s** |
| Requests completed | **231 / 231** |
| Strict format validity | **100%** |
| Peak PyTorch allocated VRAM | **7.45 GiB** |
| GPU hourly rate | **$0.435 / GPU-hour** |
| GPU occupancy cost per 1,000 requests | **$0.00567** |

The cost is calculated from the reported run duration:

```text
GPU occupancy cost per 1,000 requests
= (10.8373 / 3,600) × $0.435 × (1,000 / 231)
≈ $0.00567
```

This is the GPU occupancy cost allocated to the run, excluding other service overhead. It is not a hosted API price. Peak PyTorch allocated VRAM is an allocator measurement, not the total VRAM required for deployment.

## Model architecture

W1-JEV uses a custom PyTorch **`LangDiT` language diffusion transformer**, with bidirectional attention, timestep conditioning, and adaptive layer normalization.

| Property | Value |
| --- | --- |
| Model | Custom PyTorch `LangDiT` |
| Parameters | **3,739,097,600 — approximately 3.74B, in the 4B class** |
| Transformer blocks | 48 |
| Vocabulary size | 64,512 |
| Hidden / attention / FFN dimensions | 2,048 / 3,072 / 7,168 |
| Attention heads / head dimension | 24 / 128 |
| Attention | Bidirectional, RoPE, no causal mask |
| Conditioning | Timestep embedding and adaptive layer normalization |
| Default runtime input limit | **4,096 tokens**, configurable via `model.max_seq_len` |
| Default inference dtype | **BF16** |

The checkpoint is stored in **FP32**; inference uses **BF16** by default. The runtime input limit is configured through `model.max_seq_len` in `config.json`; RoPE positions are computed dynamically from the actual sequence length.

## Runtime entry points

See [Usage](./usage.md) for download, installation, and a first request.

Use the repository's W1-specific runtime for inference. The main entry points are:

- **`structured_server.py`** — W1 runtime and local HTTP service.
- **`examples/ask.py`** — executable client for choice questions.
- **`config.json`** — model architecture and runtime context configuration.

## Repository contents

| File | Purpose |
| --- | --- |
| `w1-jev.pt` | FP32 model-only checkpoint; no optimizer, EMA, or other training state. |
| `model.py` | Pure PyTorch W1 language diffusion transformer. |
| `config.json` | Architecture and runtime context configuration. |
| `tokenizer.json`, `tokenizer_config.json` | W1 tokenizer and special-token metadata. |
| `checkpoint.py` | Strict validation of model-state keys, shapes, and finite values. |
| `batch.py` | Variable-length batching and answer-slot logits. |
| `djev_template.py` | Extracted djev schema, prompt, template, and scheduling helpers. |
| `structured_server.py` | W1-specific runtime and local HTTP service. |
| `examples/ask.py` | Executable choice-question client. |
| `README.md` | English model card. |
| `report.zh.md` | Chinese model card, benchmark report, and repository guide. |
