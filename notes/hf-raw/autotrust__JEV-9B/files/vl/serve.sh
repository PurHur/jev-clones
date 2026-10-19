#!/usr/bin/env bash
# JEV-9B with vision: the multimodal Qwen/Qwen3.5-9B (unchanged, including its vision encoder) + JEV-9B's System 1
# adapter and decision head (vl/adapter_vllm: the same weights as adapter_vllm/, with the layer names moved under
# model.language_model). JEV-9B's language weights are bit-identical to Qwen3.5-9B's, so text decisions are unchanged.
#   System 1: POST /v1/decide {kind, state, question, options} -> calibrated probabilities (text and images)
#             or model "jev-decision" through /v1/completions and /v1/chat/completions
#   System 2: model "autotrust/JEV-9B" through /v1/chat/completions (text and images)
# --max-num-seqs 8 is required: with more than 8 sequences in one batch, vLLM's LoRA path for this multimodal model class
# returns wrong System 1 probabilities. --trust-request-chat-template lets image decisions use the raw decision template.
set -e
JEV_DIR=${JEV_DIR:-JEV-9B}
BASE_DIR=${BASE_DIR:-Qwen3.5-9B}
[ -f "$JEV_DIR/vl/serve_decide.py" ] || hf download autotrust/JEV-9B --include "vl/*" --local-dir "$JEV_DIR"
[ -f "$BASE_DIR/config.json" ] || hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir "$BASE_DIR"
exec python3 "$JEV_DIR/vl/serve_decide.py" --model "$BASE_DIR" --served-model-name autotrust/JEV-9B \
  --enable-lora --max-lora-rank 32 --lora-modules jev-decision="$JEV_DIR/vl/adapter_vllm" \
  --logprobs-mode processed_logprobs --max-model-len ${MAX_MODEL_LEN:-32768} --enable-prefix-caching --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image": 8}' --max-num-seqs 8 --max-logprobs 256 --trust-request-chat-template \
  --port ${PORT:-8000} ${EXTRA_ARGS}
