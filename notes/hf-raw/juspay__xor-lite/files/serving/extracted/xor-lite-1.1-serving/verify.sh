#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/model}"
if [[ "$MODEL_DIR" != /* ]]; then
  MODEL_DIR="$SCRIPT_DIR/${MODEL_DIR#./}"
fi

required=(
  model.safetensors
  config.json
  generation_config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  chat_template.jinja
  decider/strands_decider_config.json
  decider/head.safetensors
  decider/lora/adapter_config.json
  decider/lora/adapter_model.safetensors
  decider/tokenizer.json
  decider/tokenizer_config.json
  decider/chat_template.jinja
)

for file in "${required[@]}"; do
  [[ -f "$MODEL_DIR/$file" ]] || { echo "Missing model file: $MODEL_DIR/$file" >&2; exit 1; }
done

if [[ -e "$MODEL_DIR/decider/slot_head.pt" ]]; then
  echo "decider/ holds both head.safetensors and slot_head.pt; remove slot_head.pt." >&2
  exit 1
fi

(cd "$MODEL_DIR" && sha256sum -c "$SCRIPT_DIR/checksums.sha256")
echo "Model files verified."
