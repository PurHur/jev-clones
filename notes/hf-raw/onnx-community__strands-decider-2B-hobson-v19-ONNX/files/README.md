---
license: apache-2.0
library_name: transformers.js
base_model: StrandsAgents/strands-decider-2B-hobson-v19
base_model_relation: quantized
pipeline_tag: text-classification
tags: [strands-decider, decision-model, system-one, typed-decisions, onnx, webgpu, transformers.js, open-jev]
---

# strands-decider-2B (hobson v19), ONNX

ONNX conversion of [StrandsAgents/strands-decider-2B-hobson-v19](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19), Strands Labs' decision model: give it a state and typed questions (`choice`, `score`, `noul`), and it returns a calibrated probability for every option, in one forward pass per question. For the model, its training data, its evaluation and its limits, see the [original card](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19) and the [strands-decider repository](https://github.com/strands-labs/strands-decider).

It runs in the browser on WebGPU with [open-jev](https://github.com/nico-martin/open-jev) (`model: "strands-decider-2b"`), or with ONNX Runtime anywhere.

## How it was made

strands-decider-2B is [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) (at the revision the adapter was trained on) with a rank-16 LoRA adapter and a pointer head of about a million parameters on the final hidden states. This repo:

- merges the LoRA into the base weights (`W + alpha/r · B·A`, alpha/r = 2);
- puts them into the graphs of [onnx-community/Qwen3.5-2B-ONNX-OPT](https://huggingface.co/onnx-community/Qwen3.5-2B-ONNX-OPT) (fused `LinearAttention` / `CausalConvWithState`), mapped by name with the transforms found by value for the same graphs (transpose, `1 + weight` for the zero-centred RMSNorm, `-exp(A_log)`); every swapped weight has cosine similarity ≥ 0.993 to the instruct weight it replaces;
- drops the LM head (the decider never reads logits) and exposes the final hidden states;
- fuses embeddings, decoder (with empty caches and position ids built in the graph) and the pointer head into one graph:

```
input_ids [B, L], attention_mask [B, L], answer_pos [B], option_pos [B, K]  ->  logits [B, K]
```

`logits` are before temperature: divide by `decider.temperature_by_kind[type]` from `config.json` (noul 0.911, choice 0.734, score 1.328) and take a softmax over the question's options.

Scripts and the reference cases are in `conversion/`.

## Files

| dtype (Transformers.js) | File | Weights | Size |
|---|---|---|---|
| `q8` (default) | `onnx/model_quantized.onnx` | decoder 8-bit, block 32 (fp16 activations); embeddings 4-bit | 1.80 GB |
| `q4f16` | `onnx/model_q4f16.onnx` | decoder and embeddings 4-bit, block 32 | 1.09 GB |

Both need WebGPU with `shader-f16` in the browser (ONNX Runtime's CPU provider runs them too). `q8` here is 8-bit block quantization with ONNX Runtime's own kernel (`MatMulNBits`, `bits=8`), not the usual dynamic int8 export; it uses the `q8` slot because Transformers.js has no `q8f16`.

## Checks

Against the upstream engine (`strands_decider.infer`, PyTorch, CPU fp32), largest difference in any option's probability and how often the top answer changed:

| Runtime | 11 questions on 6 short states | 40 questions on 8 typed-decisions states |
|---|---|---|
| fp32 ONNX (ONNX Runtime, CPU) | 0.0009, 0 changed | 0.0013 (10 questions), 0 changed |
| `q8`, WebGPU (Chrome, M3 Pro) | 0.016, 0 changed | 0.026, 0 changed |
| `q4f16`, WebGPU | 0.072, 0 changed | 0.146, 5 changed |

`q4f16` moves near-ties on long states; prefer `q8` unless the download size matters more.

The open-jev integration was also checked in Node against the Python run of the same graph: identical probabilities (prompt rendering, tokenization, option positions and temperatures match `prompting.py` and `infer.py`).

## Speed and accuracy in the browser

On a MacBook Pro (M3 Pro, Chrome, WebGPU): about 130 ms for one short question; questions on one state run as a batch. On the 400 states × 5 questions of [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (test split, top answer = annotator label, as in the [open-jev demo](https://huggingface.co/spaces/shreyask/open-jev-demo)'s benchmark):

| dtype | choice | score | noul | overall | median per state |
|---|---|---|---|---|---|
| `q4f16` | 334/600 | 433/800 | 401/600 | 58.4% | 2.2 s |
| `q8` | 337/600 | 458/800 | 392/600 | 59.4% | 2.2 s |

The model was not trained on typed-decisions. Its confidence holds up there too: answers at 0.9 or more are right 94% of the time (`q8`).

## License

Apache-2.0, as the original model and Qwen3.5-2B-Base.
