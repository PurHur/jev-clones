#!/usr/bin/env bash
# Serve autotrust/JEV-35B with vLLM: System 1 typed decisions (POST /v1/decide), text and images.
# The System 1 decision model is merged into the weights (no LoRA): the head bias and per-kind temperatures are applied
# by serve_decide.py (adapter_vllm/decision_head.json, calibration.json). serve_decide.py is the standard vLLM OpenAI
# server (same flags) with /v1/decide added. MAX_MODEL_LEN: up to 262144 (the backbone's native context).
set -e
MODEL_DIR=${MODEL_DIR:-JEV-35B}
[ -d "$MODEL_DIR" ] || hf download autotrust/JEV-35B --local-dir "$MODEL_DIR"
export JEV_DECIDE_MERGED_HEAD="$MODEL_DIR/adapter_vllm/decision_head.json" JEV_DECIDE_SYSTEM2=0
exec python3 "$MODEL_DIR/serve_decide.py" --model "$MODEL_DIR" --served-model-name autotrust/JEV-35B \
  --logprobs-mode processed_logprobs --max-logprobs 256 --max-model-len ${MAX_MODEL_LEN:-65536} \
  --enable-prefix-caching --mamba-cache-mode align --max-num-seqs ${MAX_NUM_SEQS:-64} --trust-request-chat-template \
  --limit-mm-per-prompt '{"image": 8}' --port ${PORT:-8000}
