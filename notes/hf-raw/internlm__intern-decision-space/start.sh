#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
unset PYTHONPATH
# One process owns the model; DecisionPipeline serializes local GPU inference.
exec python -m uvicorn app:app --host 0.0.0.0 --port 7860 --workers 1 --no-access-log
