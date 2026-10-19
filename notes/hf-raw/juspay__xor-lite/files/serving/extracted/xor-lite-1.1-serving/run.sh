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
export API_PORT="${API_PORT:-30002}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for command in docker nvidia-smi python3 sha256sum; do
  command -v "$command" >/dev/null || { echo "Missing required command: $command" >&2; exit 1; }
done

docker compose version >/dev/null
docker info >/dev/null

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "This release currently requires an x86_64 Linux host." >&2
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  echo "Place the released model files there or set MODEL_DIR in .env." >&2
  exit 1
fi

[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || {
  echo "Set CUDA_VISIBLE_DEVICES to one GPU index; the release runs on a single GPU." >&2
  exit 1
}
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
(( CUDA_VISIBLE_DEVICES < GPU_COUNT )) || {
  echo "GPU index $CUDA_VISIBLE_DEVICES is unavailable; host has $GPU_COUNT GPUs." >&2
  exit 1
}

./verify.sh

python3 - "$API_PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket() as listener:
    try:
        listener.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"API port {port} is unavailable: {exc}")
PY

docker compose up -d --build xor-lite-api

echo "Waiting for the XOR Lite API..."
for _ in $(seq 1 180); do
  if docker compose exec -T xor-lite-api python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:30002/health', timeout=2).read()" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

docker compose exec -T xor-lite-api python3 /app/smoke_test.py http://127.0.0.1:30002
python3 ./smoke_test.py "http://127.0.0.1:$API_PORT"
echo "XOR Lite is ready at http://127.0.0.1:$API_PORT/v1/systemone"
