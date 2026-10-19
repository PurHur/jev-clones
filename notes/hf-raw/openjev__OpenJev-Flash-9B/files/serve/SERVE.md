# Serve OpenJev Flash 9B

This repository holds the merged 16-bit weights at its root, so there is no merge step. Python 3.12 was used for the serving environments. The unchanged helper must hash to `81a22f1b1b8912a465059207ef9f60b7c6c16b4de6372305d867efbe38a1987a` (`shasum -a 256 -c SHA256SUMS` checks every file).

## NVIDIA: vLLM

The frozen-10k run used vLLM 0.29.0 on an H100 SXM with online FP8 quantization.

```bash
python3.12 -m venv .venv-nvidia
source .venv-nvidia/bin/activate
python -m pip install 'vllm==0.29.0' 'openai==3.26.0' 'httpx==0.28.1'
hf download openjev/OpenJev-Flash-9B --local-dir OpenJev-Flash-9B

vllm serve ./OpenJev-Flash-9B --host 127.0.0.1 --served-model-name qwen --port 8000 \
  --enable-prefix-caching --max-model-len 16384 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":1}' --trust-remote-code --max-num-seqs 16 \
  --max-logprobs 64 --gdn-prefill-backend triton --quantization fp8
```

In a second terminal, activate the same environment and start the decision API:

```bash
source .venv-nvidia/bin/activate
VLLM=http://127.0.0.1:8000/v1 TOKENIZER=./OpenJev-Flash-9B SHIM_MODEL=./OpenJev-Flash-9B \
READOUT_T=1.07 READOUT_NOUL_T=1.074766 READOUT_NOUL_BIAS=0 \
READOUT_TARGETED=1 READOUT_INSTR_STYLE=pyrepr READOUT_PERMS=1 \
SHIM_STAGGER=1 SHIM_COMPACT=0 SHIM_COMPACT_CAP=0 SHIM_LAYOUT= SHIM_PAD=0 SHIM_LOOP_BREAK=0 \
python OpenJev-Flash-9B/helper/shim.py --host 127.0.0.1 --port 3000
```

Keep `--served-model-name qwen` and `--max-logprobs 64`: the frozen helper requests the exact candidate token scores from that model name. The helper is unchanged, so the environment assignments above select the 9B calibration instead of its built-in defaults. Optional prompt transforms and loop-breaking are disabled for the recorded protocol.

## Apple silicon: MLX 8-bit

Use [openjev/OpenJev-Flash-9B-MLX](https://huggingface.co/openjev/OpenJev-Flash-9B-MLX), the 8-bit build (group size 64) behind the JevBench numbers. The MLX environment used MLX 0.32.2, mlx-lm 0.31.3, Transformers 5.17.0, OpenAI 3.16.2 and HTTPX 0.28.1.

```bash
python3.12 -m venv .venv-mlx
source .venv-mlx/bin/activate
python -m pip install 'mlx==0.32.2' 'mlx-lm==0.31.3' \
  'transformers==5.17.0' 'openai==3.16.2' 'httpx==0.28.1' 'huggingface-hub==1.32.0'
hf download openjev/OpenJev-Flash-9B-MLX --local-dir mlx8
hf download openjev/OpenJev-Flash-9B helper/shim.py helper/shim_mlx.py --local-dir api

TOKENIZER=./mlx8 SHIM_MODEL=./mlx8 \
READOUT_T=1.07 READOUT_NOUL_T=1.074766 READOUT_NOUL_BIAS=0 \
READOUT_TARGETED=1 READOUT_INSTR_STYLE=pyrepr READOUT_PERMS=1 \
SHIM_STAGGER=1 SHIM_COMPACT=0 SHIM_COMPACT_CAP=0 SHIM_LAYOUT= SHIM_PAD=0 SHIM_LOOP_BREAK=0 \
python api/helper/shim_mlx.py --helper api/helper/shim.py --model ./mlx8 \
  --host 127.0.0.1 --port 3000
```

This is the text-only MLX path used for the reported JevBench score. It uses the checkpoint's chat template with thinking disabled and computes full-vocabulary log-softmax at the first output position; the frozen helper selects and calibrates the option probabilities. Keep `--prefix-cache` off to reproduce that path. The shim serializes model calls with a lock; no concurrent-throughput claim is made.

For a local choice and yes/no smoke check, use the same command with `--selfcheck` appended. It prints the helper hash and calibration alongside the answers and exits. It is a smoke check, not an evaluation score.

## Check the API

```bash
curl -s http://127.0.0.1:3000/v1/version
curl -s http://127.0.0.1:3000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"openjev-flash-9b","state":"The customer was charged twice.","questions":{"route":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":null,"shipping":null,"technical":null}}}}'
```

The version response must show `T: 1.07`, `noul_t: 1.074766`, `noul_bias: 0`, targeted readout enabled, `instr_style: "pyrepr"`, and helper SHA256 beginning `81a22f1b`. The decision response contains `answers.route.choice`, option probabilities and confidence.

Choice uses a non-empty option map; yes/no uses `type: "noul"`; score uses `type: "score"` and an ordered array in `criteria`. Each question with up to 52 options is one forward pass. Larger option sets use approximate multi-pass readout. MLX does not support image input or the helper's vLLM chat/prewarm passthrough endpoints.

These commands bind both servers to loopback. If exposing the helper, set `SHIM_TOKEN` to a secret supplied by your deployment and send it as `Authorization: Bearer ...`; terminate TLS and enforce request-size limits at your gateway. Keep the model backend private. The frozen helper is a compatibility server, not a complete public API gateway.
