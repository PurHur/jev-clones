#!/usr/bin/env bash
# Serve autotrust/GEV-26B-Decide with vLLM: one engine, both systems, text and images.
#   System 2: the unmodified gemma-4-26B-A4B-it (served name autotrust/GEV-26B-Decide) on the OpenAI endpoints
#   System 1: POST /v1/decide (2-256 options, adaptive thinking), or the LoRA module "jev-decision" directly
# serve_decide.py is the standard vLLM OpenAI server (same flags) with /v1/decide added.
# Requirements: a vLLM build with Gemma-4 support plus patches/vllm-gemma4-lm-head-lora.patch (LoRA on Gemma-4's tied
# lm_head, vocabulary 262,144), tested with a vLLM development build from September 2026.
# MAX_MODEL_LEN: up to 262144 (the backbone's native context).
# MTP=1: speculative decoding with google/gemma-4-26B-A4B-it-assistant (System 2 about 1.8-1.9x faster; lower System 1
#        throughput at high concurrency).
set -e
MODEL_DIR=${MODEL_DIR:-GEV-26B-Decide}
[ -d "$MODEL_DIR" ] || hf download autotrust/GEV-26B-Decide --local-dir "$MODEL_DIR"
EXTRA=()
if [ "${MTP:-0}" = "1" ]; then
  EXTRA=(--speculative-config '{"model": "google/gemma-4-26B-A4B-it-assistant", "num_speculative_tokens": 4}')
fi
exec python3 "$MODEL_DIR/serve_decide.py" --model "$MODEL_DIR" --served-model-name autotrust/GEV-26B-Decide \
  --enable-lora --max-lora-rank 32 --lora-modules jev-decision="$MODEL_DIR/adapter_vllm" \
  --logprobs-mode processed_logprobs --max-logprobs 256 --max-model-len ${MAX_MODEL_LEN:-65536} --enable-prefix-caching \
  --max-num-seqs 256 --trust-request-chat-template --limit-mm-per-prompt '{"image": 8}' \
  "${EXTRA[@]}" --port ${PORT:-8000}
