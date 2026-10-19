#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/model}"
if [[ "$MODEL_DIR" != /* ]]; then
  MODEL_DIR="$SCRIPT_DIR/${MODEL_DIR#./}"
fi

required=(
  model-00001-of-00016.safetensors
  model-00002-of-00016.safetensors
  model-00003-of-00016.safetensors
  model-00004-of-00016.safetensors
  model-00005-of-00016.safetensors
  model-00006-of-00016.safetensors
  model-00007-of-00016.safetensors
  model-00008-of-00016.safetensors
  model-00009-of-00016.safetensors
  model-00010-of-00016.safetensors
  model-00011-of-00016.safetensors
  model-00012-of-00016.safetensors
  model-00013-of-00016.safetensors
  model-00014-of-00016.safetensors
  model-00015-of-00016.safetensors
  model-00016-of-00016.safetensors
  model.safetensors.index.json
  config.json
  configuration.json
  generation_config.json
  merges.txt
  tokenizer.json
  tokenizer_config.json
  vocab.json
  chat_template.jinja
  preprocessor_config.json
  video_preprocessor_config.json
)

for file in "${required[@]}"; do
  [[ -f "$MODEL_DIR/$file" ]] || { echo "Missing model file: $MODEL_DIR/$file" >&2; exit 1; }
done

if grep -q REPLACE_WITH_RELEASE_SHA256 "$SCRIPT_DIR/checksums.sha256"; then
  echo "Release checksums have not been finalized." >&2
  exit 1
fi

(cd "$MODEL_DIR" && sha256sum -c "$SCRIPT_DIR/checksums.sha256")
echo "Model files verified."
