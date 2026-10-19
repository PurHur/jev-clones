#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/model}"
if [[ "$MODEL_DIR" != /* ]]; then
  MODEL_DIR="$SCRIPT_DIR/${MODEL_DIR#./}"
fi
export MODEL_DIR

docker compose down
