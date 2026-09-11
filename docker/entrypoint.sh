#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-serve}"
if [[ $# -gt 0 ]]; then
  shift
fi

MODEL_PATH="${MODEL_PATH:-/models/nvidia/Qwen3.6-35B-A3B-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-35B-A3B-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-1919}"
MEMORY_RATIO="${MEMORY_RATIO:-0.88}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
MAX_PREFILL_LENGTH="${MAX_PREFILL_LENGTH:-2048}"
KV_RESERVE_TOKENS="${KV_RESERVE_TOKENS:-2048}"
MOE_BACKEND="${MOE_BACKEND:-offload}"
NVFP4_BACKEND="${NVFP4_BACKEND:-triton}"

case "${cmd}" in
  serve)
    exec ft serve \
      --model "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --host "${HOST}" \
      --port "${PORT}" \
      --memory-ratio "${MEMORY_RATIO}" \
      --max-running-requests "${MAX_RUNNING_REQUESTS}" \
      --max-prefill-length "${MAX_PREFILL_LENGTH}" \
      --kv-reserve-tokens "${KV_RESERVE_TOKENS}" \
      --moe-backend "${MOE_BACKEND}" \
      --nvfp4-backend "${NVFP4_BACKEND}" \
      ${FT_EXTRA_ARGS:-} \
      "$@"
    ;;
  shell)
    exec ft shell --server "http://127.0.0.1:${PORT}" "$@"
    ;;
  bench)
    exec ft bench bw "$@"
    ;;
  bash|sh)
    exec "${cmd}" "$@"
    ;;
  *)
    exec "${cmd}" "$@"
    ;;
esac
