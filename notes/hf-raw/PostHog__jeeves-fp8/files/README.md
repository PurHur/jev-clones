---
license: apache-2.0
base_model: PostHog/jeeves
base_model_relation: quantized
language:
- en
pipeline_tag: text-classification
tags:
- decision
- calibration
- reasoning
- speculative-decoding
- qwen3.5
- fp8
---

# Jeeves-9B FP8

FP8 weights of [PostHog/jeeves](https://huggingface.co/PostHog/jeeves), a Jev-like decision model that reasons before it decides when it is unsure. This is Jeeves v2; v1 is the `v1` tag of this repository. Every large linear layer of the model and of the block-4 drafter is stored in FP8 (e4m3fn), with one float32 scale per output row. The embeddings, the norms, the Gated DeltaNet gates and the pointer head stay as in the bf16 release. The download is 11.6 GB. The bf16 model and its block-4 drafter are 21 GB.

`--precision fp8` with the bf16 weights gives the same outputs. These weights load with the [Jeeves code](https://github.com/PostHog/jeeves), not with `transformers`.

## Files

| file | contents |
|---|---|
| `model-*.safetensors`, `config.json` | fused Qwen3.5-9B: each quantized layer's `*.weight` in F8_E4M3 with a float32 `*.weight_scale` per output row, other tensors bf16 |
| `drafter_k4.safetensors` | block-4 drafter, projections in FP8 the same way |
| `head.pt` | pointer head, unchanged |
| `export.json` | temperatures (1.522 direct, 5.420 after reasoning), prompt format and training config, with `"precision": "fp8"` |
| `tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`, `chat_template.jinja` | Qwen3.5 tokenizer |
| `LICENSE` | Apache-2.0, inherited from Qwen3.5-9B |

## Usage

```bash
hf download PostHog/jeeves-fp8 --local-dir jeeves-fp8
git clone https://github.com/PostHog/jeeves && cd jeeves
python -m inference.serve --model ../jeeves-fp8 --drafter ../jeeves-fp8/drafter_k4.safetensors --precision fp8 --port 8009
```

The engine serves these weights on CUDA GPUs with compute capability 8.9 or higher and on Apple Silicon Macs. On a Mac with 48 GB, add `--max-rows 4 --max-len 4096`. These weights only run with `--precision fp8`.

The request and response follow Jev's `/v1/systemone` format. See the [bf16 model card](https://huggingface.co/PostHog/jeeves) and the GitHub README for the API, the options and the results.

## Accuracy and speed

FP8 changes the outputs slightly against bf16. On dev questions, accuracy and NLL did not change measurably. The serving results in the bf16 model card (325 dev questions, accuracy 0.822 reasoning when unsure) come from `--precision fp8` on an H100. The speed rows below were measured with v1.

| | bf16 | FP8 |
|---|---|---|
| GPU memory for the weights | 21 GB | 11.5 GB |
| `speed.py` on an L40S (12 dev requests) | 9.5 s | 7.9 s |
| one question thinking on an M4 Pro (48 GB) | about 20 tokens/s | about 38 tokens/s |

## How they were made

```bash
hf download PostHog/jeeves --revision 27c0c79f6c8b68abba1b1edcaa78cd4a2a6a7431 --local-dir jeeves-weights
python export_fp8.py jeeves-weights --drafter jeeves-weights/drafter_k4.safetensors --out jeeves-fp8
```

with `export_fp8.py` from the GitHub repository. The drafter is still v1's. Each row's scale is the row's largest absolute weight divided by 448, the largest e4m3fn value. Each weight is divided by its row's scale and rounded to the nearest e4m3fn value.

## License

The weights are derived from Qwen3.5-9B and are released under its Apache-2.0 license (`LICENSE`). The Jeeves code is MIT licensed.
