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
export TP_SIZE="${TP_SIZE:-1}"
export DP_SIZE="${DP_SIZE:-1}"
export MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-250000}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"

for command in docker nvidia-smi python3; do
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

IFS=',' read -r -a SELECTED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
REQUIRED_GPUS=$(( TP_SIZE * DP_SIZE ))
if (( ${#SELECTED_GPUS[@]} < REQUIRED_GPUS )); then
  echo "TP_SIZE=$TP_SIZE and DP_SIZE=$DP_SIZE require at least $REQUIRED_GPUS entries in CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
for gpu in "${SELECTED_GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid CUDA_VISIBLE_DEVICES entry: $gpu" >&2; exit 1; }
  (( gpu < GPU_COUNT )) || { echo "GPU index $gpu is unavailable; host has $GPU_COUNT GPUs." >&2; exit 1; }
done

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

docker compose up -d sglang
echo "Waiting for SGLang..."
for _ in $(seq 1 360); do
  if docker compose exec -T sglang python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:30000/health', timeout=2).read()" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

docker compose exec -T sglang python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:30000/health', timeout=5).read()" >/dev/null
docker compose up -d --build jev-api

echo "Waiting for the JEV API..."
for _ in $(seq 1 60); do
  if docker compose exec -T jev-api python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:30002/health', timeout=2).read()" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

docker compose exec -T jev-api python3 /app/smoke_test.py http://127.0.0.1:30002
python3 ./smoke_test.py "http://127.0.0.1:$API_PORT"
echo "Xor is ready at http://127.0.0.1:$API_PORT/v1/systemone"
