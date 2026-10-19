# Bekko System One v0 400M — browser ONNX

This directory contains an ONNX conversion for the dedicated **Bekko System One browser runtime**. It supports Choice, Noul, and Score typed decisions. The companion Node.js runtime can also load this directory.

This is not a Sentence Transformers ONNX backend export or a general-purpose sentence embedding model. Use the model repository's root files and `inference_v0.py` for Python / Sentence Transformers inference.

## Files and precision

| File | Purpose |
| --- | --- |
| `model.onnx` | Shared-prefix encoder and three prediction heads |
| `manifest.json` | Model filename, hashes, task order, token budgets, quantization policy, and source revision |
| `tokenizer.json` | Tokenizer vocabulary and processing rules |
| `tokenizer_config.json` | Special tokens used by the dedicated runtime |

Only the token embedding table is stored as row-wise symmetric INT8. Transformer blocks and prediction heads remain FP32; this is not fully INT8 inference. The tokenizer is unchanged. The ONNX file is 1426.2 MB (decimal), excluding tokenizer and runtime downloads.

## Loading

Use the [Bekko browser runtime and usage guide](https://github.com/hotchpotch/bekko-system-one/tree/main/browser). Set the model base URL to:

```text
https://huggingface.co/hotchpotch/bekko-system-one-v0-400m/resolve/<COMMIT_REVISION>/onnx_browser/
```

Replace `<COMMIT_REVISION>` with a commit containing this directory and pin that same revision for all files. The runtime reads `manifest.json`, both tokenizer files, and the model named in the manifest. It handles typed-decision input rendering, candidate grouping, and output probabilities. Loading `model.onnx` alone does not provide those application-level operations.

Browser execution uses ONNX Runtime Web. Available execution providers, memory use, and practical input sizes depend on the browser and device. Download size is not peak runtime memory; larger models may exceed a device's memory budget. Node.js usage is documented in the same guide.

## Conversion and verification

- [FP32 ONNX export script](https://github.com/hotchpotch/bekko-system-one/blob/main/browser/scripts/export_onnx.py)
- [Embedding INT8 conversion script](https://github.com/hotchpotch/bekko-system-one/blob/main/browser/scripts/quantize_onnx.py) (`--embedding-only`)
- [ONNX verification script](https://github.com/hotchpotch/bekko-system-one/blob/main/browser/scripts/verify-export.js)

The INT8 output is distributed here as `model.onnx`; the manifest records its quantization policy. FP32 reference files and parity fixtures are retained separately for verification.

Source model revision: `4796badc342d3fe431085af5af66ab018d0d077c`. The export checkpoint weights and tokenizer were checked against this revision. On 5 synthetic fixtures, Node.js / ONNX Runtime verification measured a maximum FP32 logit error of 1.0371208e-05 against the Python reference, and a maximum absolute probability difference of 0.0010780631 between FP32 and embedding-INT8 ONNX. These checks are not a full quality evaluation or a guarantee of browser/device compatibility.
