---
license: apache-2.0
library_name: transformers.js
base_model: vllm-sr/Decision-2.0-Sol-2B
base_model_relation: quantized
pipeline_tag: text-classification
tags: [decision-model, system-one, typed-decisions, onnx, webgpu, transformers.js, open-jev]
---

# Decision-2.0-Sol-2B (ONNX)

ONNX conversion of [vllm-sr/Decision-2.0-Sol-2B](https://huggingface.co/vllm-sr/Decision-2.0-Sol-2B), a [Decision 2.0](https://huggingface.co/collections/vllm-sr/decision-20-6ab7cf7bdfb506bf8269cb00) model from [vLLM Semantic Router](https://github.com/vllm-project/semantic-router): give it a state and typed questions (`choice`, `noul`, `score`) and it returns a probability for every option. For the model, its training and evaluation, see the [original card](https://huggingface.co/vllm-sr/Decision-2.0-Sol-2B).

It runs in the browser on WebGPU with [open-jev](https://github.com/nico-martin/open-jev) (`model: "decision2-sol-2b"`), or with ONNX Runtime anywhere.

## How it was made

Decision 2.0 is a full fine-tune of Qwen3.5-2B with a candidate head on the final hidden states. The weights were put into the graphs of [onnx-community/Qwen3.5-2B-ONNX-OPT](https://huggingface.co/onnx-community/Qwen3.5-2B-ONNX-OPT) (fused `LinearAttention` / `CausalConvWithState`), mapped by name with transforms found by value for the same graphs (transpose, `1 + weight` for the zero-centred RMSNorm, `-exp(A_log)`), and quantized with ONNX Runtime's own kernels. The LM head is dropped; embeddings, decoder (empty caches and position ids built in-graph) and head are fused into one graph.

```
input_ids [B, L], attention_mask [B, L], answer_pos [B], option_pos [B, K]  ->  logits [B, K]
```

Build one prompt per question as the package's `decision_model.segments` / `encode` do (`Context:` … `Task type:` … `Question:` … `Options:`, one `<option>` segment per option holding `{"description": …, "key": …}` as sorted, compact JSON, then the `Decision:` suffix), tokenizing each segment separately; `option_pos` is the last token of each option segment and `answer_pos` the last token. Add `decision2.score_bias[L]` from `config.json` to the logits of an L-level score question (none for this model), then softmax (temperature 1).

Scripts and reference cases are in `conversion/`.

## Files

| dtype (Transformers.js) | File | Weights | Size |
|---|---|---|---|
| `q8` (default) | `onnx/model_quantized.onnx` | 8-bit block-32 weights, 4-bit embeddings, fp16 activations | 1.81 GB |
| `q4f16` | `onnx/model_q4f16.onnx` | 4-bit block-32 weights, fp16 activations | 1.10 GB |

`q8` here is 8-bit block quantization (`MatMulNBits`, `bits=8`), not the usual dynamic int8 export; it uses the `q8` slot because Transformers.js has no `q8f16`.

## Checks

Against the packaged runtime (`decision2.api.Decision2`, exact path, CPU fp32), on 11 questions over 6 states: largest difference in any option's probability and how many top answers changed.

| Runtime | Max difference | Top answer changed |
|---|---|---|
| `q8`, ONNX Runtime CPU (fixtures) | 0.054 | 0 of 11 |
| `q8` on 30 typed-decisions questions | 0.021 | 1 of 30 |
| `q4f16`, ONNX Runtime CPU (fixtures) | 0.200 | 1 of 11 |

open-jev in Node reproduces the Python run of the same graph exactly (same token ids, option positions and score offsets).

## In the browser

On a MacBook Pro (M3 Pro, Chrome, WebGPU), on the 400 states × 5 questions of [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (test split; top answer = annotator label, as in the [open-jev demo](https://huggingface.co/spaces/shreyask/open-jev-demo)'s benchmark):

| dtype | choice | score | noul | overall | median per state |
|---|---|---|---|---|---|
| `q8` | 301/600 | 379/800 | 440/600 | 56.0% | 2.5 s |

## License

Apache-2.0, as the original model.
