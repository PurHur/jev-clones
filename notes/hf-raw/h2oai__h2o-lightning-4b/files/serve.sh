#!/bin/bash
# H2O-Lightning-4B: stock vLLM + the shim, on one GPU. Run it from a download of the model repo:
#     hf download h2oai/h2o-lightning-4b --local-dir h2o-lightning-4b && bash h2o-lightning-4b/serve.sh
# Then POST decisions to http://127.0.0.1:${PORT:-8741}/v1/systemone. Ctrl-C stops both.
# MODEL defaults to the directory this script is in (the downloaded weights); VLLM_PORT, PORT, MAX_LEN, GPU_UTIL override.
# A model folder with a vision tower (config.json has "vision_config") also gets vLLM's image limits: MAX_IMAGES images
# per prompt (default 4), no video, and images scaled to at most MAX_PIXELS pixels (default: serve_config.json's
# image.max_pixels, the cap the shim sends with each request; 1605632 when the config sets none).
# Before it reports ready, it sends a warm-up request to vLLM, so the first real request is not the slow one.
# VLLM_ARGS adds flags to `vllm serve` (space-separated; values cannot contain spaces), for example the
# high-concurrency ones in the card's "Serving at scale".
# VLLM_PATCHES=1 first applies the two vLLM 0.30.0 source patches in vllm_patches/ to the vLLM that runs here (see the
# card): it edits that environment's vLLM install, so use a dedicated virtual environment. Default off (stock vLLM).
# ATTENTION MODE. TWO_WAY=1 (the default since v1.3; also ATTENTION=two-way) serves two-way (bidirectional) attention,
# the recommended mode: ONE vLLM on the pooling runner with the override '{"is_causal": false, "text_config":
# {"is_causal": false}}', and the shim reading the final hidden state with the shipped label rows. It needs the same
# GPU memory as one-way. TWO_WAY=0 (or ATTENTION=one-way) is the one-way path: one stock vLLM in generate mode, as
# before v1.3. With TWO_WAY=1, Box questions (LIGHTNING_BOX_ENABLED=1; a generation, which the pooling runner cannot
# do) start a SECOND vLLM in generate mode on VLLM_GEN_PORT (default 8001) when the GPU has room for it
# (BOX_ENGINE_MIB free, default 14000, after the two-way engine is up; GPU_UTIL is then 0.45 per engine unless set);
# if there is no room, or that engine fails to start, the decisions keep serving two-way and a box question gets a
# clear 400 (the line below says so). A one-way request ("attention_mode": "causal") on the one-engine two-way
# deployment is a 400 as well. Stock vLLM 0.30, no patch needed for either mode.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-$HERE}"; VLLM_PORT="${VLLM_PORT:-8000}"; PORT="${PORT:-8741}"
case "${ATTENTION:-}" in "") ;; one-way|causal) TWO_WAY=0;; two-way|bidirectional) TWO_WAY=1;; *) echo "ATTENTION must be one-way or two-way"; exit 1;; esac
TWO_WAY="${TWO_WAY:-1}"
case "$TWO_WAY" in 0|1) ;; *) echo "TWO_WAY must be 0 or 1"; exit 1;; esac
# a folder from the first v1.3 download still holds label_rows.safetensors, which vLLM reads as a weight shard and
# fails on ("no module or parameter named 'W'"): refuse in every mode until it is deleted
[ -e "$MODEL/label_rows.safetensors" ] && { echo "delete $MODEL/label_rows.safetensors (an older download; vLLM reads every *.safetensors as weights); the label rows now ship as label_rows.lightning"; exit 1; }
# ONE_ENGINE=1 (v1.3.1, needs the vLLM build with per-request attention, github.com/h2oai/vllm branch
# h2o/per-request-attention @236568a780): ONE generate-mode engine serves one-way readouts, two-way readouts (the shim sends
# vllm_xargs attention_mode=bidirectional per request) and Box; no pooling engine, no label rows. The engine runs
# with --attention-backend TRITON_ATTN and --max-num-batched-tokens = MAX_LEN (a two-way prompt is prefilled whole).
case "${ONE_ENGINE:-0}" in 0|1) ;; *) echo "ONE_ENGINE must be 0 or 1"; exit 1;; esac
ONE_ENGINE="${ONE_ENGINE:-0}"
GEN_PORT="${VLLM_GEN_PORT:-8001}"; BOX_ENGINE_MIB="${BOX_ENGINE_MIB:-14000}"
case "${LIGHTNING_BOX_ENABLED:-0}" in 1|true|yes|on) BOX=1;; *) BOX=0;; esac
if [ "$TWO_WAY" = 1 ]; then
  [ -f "$HERE/label_rows.lightning" ] || { echo "TWO_WAY=1 needs label_rows.lightning beside serve.sh (shipped with the model); TWO_WAY=0 serves one-way without it"; exit 1; }
  if [ "$BOX" = 1 ]; then GPU_UTIL="${GPU_UTIL:-0.45}"; fi
fi
# VLLM_PATCHES=1: vLLM must be 0.30.0; each patch's sha256 must match vllm_patches/SHA256SUMS and it must apply exactly
# (no fuzz); a patch already in place is left as it is; one that neither is in place nor applies stops the script.
# Every file a patch touches becomes a private copy first: uv virtual environments hard-link installed files to a
# shared cache, and patching through the link would change every environment linked to it.
case "${VLLM_PATCHES:-0}" in 0|1) ;; *) echo "VLLM_PATCHES must be 0 or 1"; exit 1;; esac
if [ "${VLLM_PATCHES:-0}" = 1 ]; then
  command -v patch > /dev/null || { echo "VLLM_PATCHES=1 needs the patch program (for example: apt-get install patch)"; exit 1; }
  SP="$(python3 -c 'import importlib.util as u, importlib.metadata as m, os, sys; v = m.version("vllm"); sys.exit("VLLM_PATCHES=1: vLLM " + v + " found; the patches are for 0.30.0") if not v.startswith("0.30.0") else print(os.path.dirname(os.path.dirname(u.find_spec("vllm").origin)))')"
  (cd "$HERE/vllm_patches" && sha256sum -c --quiet SHA256SUMS) || { echo "vllm_patches/: sha256 mismatch"; exit 1; }
  for P in "$HERE"/vllm_patches/*.patch; do
    if patch -p1 -R -s -f -F0 --dry-run -d "$SP" < "$P" > /dev/null 2>&1; then
      echo "vLLM patch $(basename "$P"): already in place"
    elif patch -p1 -N -s -f -F0 --dry-run -d "$SP" < "$P" > /dev/null 2>&1; then
      for T in $(grep '^+++ b/' "$P" | sed 's#^+++ b/##'); do
        if [ -e "$SP/$T" ] && [ "$(stat -c %h "$SP/$T")" -gt 1 ]; then
          cp --remove-destination -p "$SP/$T" "$SP/$T.tmp.$$" && mv -f "$SP/$T.tmp.$$" "$SP/$T" \
            || { echo "vLLM patch: cannot make $T a private copy"; exit 1; }
        fi
      done
      patch -p1 -N -s -f -F0 -d "$SP" < "$P" > /dev/null || { echo "vLLM patch $(basename "$P"): apply failed"; exit 1; }
      find "$SP/vllm" -name __pycache__ -prune -exec rm -rf {} + 2> /dev/null || true
      echo "vLLM patch $(basename "$P"): applied"
    else
      echo "vLLM patch $(basename "$P"): does not apply to the vLLM in $SP (it is for vLLM 0.30.0)"; exit 1
    fi
  done
fi
EXTRA_VLLM_ARGS="${VLLM_ARGS:-}"
unset VLLM_PATCHES VLLM_ARGS VLLM_GEN_PORT  # vLLM warns about VLLM_* variables it does not know
MM=()
if grep -q '"vision_config"' "$MODEL/config.json" 2>/dev/null; then
  # vLLM's own cap is the one the shim sends with every image request: when the two differ, vLLM's first image request
  # is very slow
  CFG_PIXELS="$(python3 -c 'import json, sys; print((json.load(open(sys.argv[1])).get("image") or {}).get("max_pixels") or "")' "$HERE/serve_config.json" 2>/dev/null || true)"
  MAX_PIXELS="${MAX_PIXELS:-${CFG_PIXELS:-1605632}}"
  MM=(--limit-mm-per-prompt "{\"image\": ${MAX_IMAGES:-4}, \"video\": 0}"
      --mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS}}")
fi
# the decisions read log-probabilities (one-way) or a pooled state (two-way) at temperature 0 and never sample, so
# FlashInfer's sampler (a JIT build that fails with an older system CUDA toolkit) is off unless
# VLLM_USE_FLASHINFER_SAMPLER says otherwise
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
SERVED=(--served-model-name h2oai/h2o-lightning-4b --host 127.0.0.1 --max-model-len "${MAX_LEN:-40960}")
gen_engine() {  # $1 = port: a stock generate-mode vLLM (one-way readouts, Box generation)
  vllm serve "$MODEL" "${SERVED[@]}" --port "$1" --gpu-memory-utilization "${GPU_UTIL:-0.90}" ${MM[@]+"${MM[@]}"} ${EXTRA_VLLM_ARGS}
}
two_way_engine() {  # $1 = port: the pooling runner with the full-attention layers two-way; the final hidden state at the
  # answer slot comes back as an "embedding" (last position, not normalised) and the shim reads the labels from it.
  # The override at both config levels: vLLM reads the inner one for the layers and the outer one to turn off chunked
  # prefill and prefix caching, which a two-way prompt must not have (both also set explicitly).
  vllm serve "$MODEL" "${SERVED[@]}" --port "$1" --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
    --runner pooling --convert embed --pooler-config '{"task": "embed", "seq_pooling_type": "LAST", "use_activation": false}' \
    --hf-overrides '{"is_causal": false, "text_config": {"is_causal": false}}' \
    --no-enable-prefix-caching --no-enable-chunked-prefill ${MM[@]+"${MM[@]}"} ${EXTRA_VLLM_ARGS}
}
wait_engine() {  # $1 = pid, $2 = port, $3 = name: until /health answers; a dead process is a failure (return 1)
  until curl -sf "http://127.0.0.1:$2/health" > /dev/null; do
    kill -0 "$1" 2>/dev/null || { echo "$3 exited"; return 1; }
    sleep 5
  done
}
PIDS=(); SHIM_ARGS=()
trap 'kill ${PIDS[@]+"${PIDS[@]}"} 2>/dev/null; wait' EXIT INT TERM
if [ "$ONE_ENGINE" = 1 ]; then
  vllm serve "$MODEL" "${SERVED[@]}" --port "$VLLM_PORT" --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
    --attention-backend TRITON_ATTN --max-num-batched-tokens "${MAX_LEN:-40960}" ${MM[@]+"${MM[@]}"} ${EXTRA_VLLM_ARGS} & GEN_PID=$!; PIDS+=("$GEN_PID")
  wait_engine "$GEN_PID" "$VLLM_PORT" "vLLM" || exit 1
  SHIM_ARGS+=(--vllm "http://127.0.0.1:$VLLM_PORT")
  export SHIM_TWO_WAY_FLAG=1
elif [ "$TWO_WAY" = 1 ]; then
  two_way_engine "$VLLM_PORT" & TWO_PID=$!; PIDS+=("$TWO_PID")
  wait_engine "$TWO_PID" "$VLLM_PORT" "the two-way vLLM" || exit 1
  SHIM_ARGS+=(--vllm-two-way "http://127.0.0.1:$VLLM_PORT")
  if [ "$BOX" = 1 ]; then
    # the free memory of the GPU this script serves on (the first one in CUDA_VISIBLE_DEVICES when that is set, else
    # the first GPU); a hung nvidia-smi is bounded, so an unanswered query only leaves the Box engine off
    GPU_ID="${CUDA_VISIBLE_DEVICES:-}"; GPU_ID="${GPU_ID%%,*}"
    FREE="$(timeout 60 nvidia-smi ${GPU_ID:+--id="$GPU_ID"} --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')" || FREE=""
    if [ -n "$FREE" ] && [ "$FREE" -ge "$BOX_ENGINE_MIB" ]; then
      gen_engine "$GEN_PORT" & GEN_PID=$!; PIDS+=("$GEN_PID")
      if wait_engine "$GEN_PID" "$GEN_PORT" "the Box (generate-mode) vLLM"; then
        SHIM_ARGS+=(--vllm "http://127.0.0.1:$GEN_PORT")
      else
        echo "WARNING: the Box (generate-mode) vLLM did not start; decisions serve two-way, a box question gets HTTP 400 (use TWO_WAY=0 for one-way with Box, or a larger GPU)" >&2
      fi
    else
      echo "WARNING: LIGHTNING_BOX_ENABLED=1 with TWO_WAY=1 needs ${BOX_ENGINE_MIB} MiB free for a second engine; ${FREE:-unknown} MiB free: decisions serve two-way, a box question gets HTTP 400 (use TWO_WAY=0 for one-way with Box, or a larger GPU)" >&2
    fi
  fi
else
  gen_engine "$VLLM_PORT" & GEN_PID=$!; PIDS+=("$GEN_PID")
  SHIM_ARGS+=(--vllm "http://127.0.0.1:$VLLM_PORT")
fi
python3 "$HERE/h2o_lightning_shim.py" --config "$HERE/serve_config.json" --model-dir "$MODEL" --port "$PORT" "${SHIM_ARGS[@]}" &
SHIM_PID=$!; PIDS+=("$SHIM_PID")
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  for P in "${PIDS[@]}"; do kill -0 "$P" 2>/dev/null || { echo "a server exited"; exit 1; }; done
  sleep 5
done
# warm-up through the shim in the deployment's default mode, so the first real request is not the slow one (and, for
# two-way, so the shim's start-up check of the two-way engine runs now and a misconfigured engine stops the script).
# One text request, plus one with a small image when the model reads images. A failure of the image warm-up is only
# a warning; a failed two-way check stops the script.
warm_shim() {
  curl -sf -m 900 "http://127.0.0.1:$PORT/v1/systemone" -H 'Content-Type: application/json' -d "{\"state\": $1,
    \"questions\": {\"decision\": {\"type\": \"noul\", \"instructions\": \"Is this a warm-up?\",
    \"criteria\": {\"true\": \"yes\", \"false\": \"no\"}}}}" > /dev/null
}
warm_shim '"warm-up"' || { echo "the warm-up request failed (see the shim's lines above)"; exit 1; }
if [ ${#MM[@]} -gt 0 ]; then
  warm_shim '{"images": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAT0lEQVR42u3PQQ0AAAgEILV/sItlCh9u0IBOUp9NPScgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBwbwFY4wLYRBP0hgAAAABJRU5ErkJggg=="], "note": "warm-up"}' \
    || echo "warning: the image warm-up request failed; serving anyway" >&2
fi
if [ "$ONE_ENGINE" = 1 ]; then
  echo "H2O-Lightning-4B ready on http://127.0.0.1:$PORT/v1/systemone (one engine, both attention modes per request; Box $([ "$BOX" = 1 ] && echo on || echo off))"
elif [ "$TWO_WAY" = 1 ]; then
  echo "H2O-Lightning-4B ready on http://127.0.0.1:$PORT/v1/systemone (two-way attention; Box $([ "$BOX" = 1 ] && [ -n "${GEN_PID:-}" ] && kill -0 "$GEN_PID" 2>/dev/null && echo on || echo off))"
else
  echo "H2O-Lightning-4B ready on http://127.0.0.1:$PORT/v1/systemone (one-way attention)"
fi
wait
