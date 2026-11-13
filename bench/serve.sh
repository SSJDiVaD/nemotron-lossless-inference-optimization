#!/usr/bin/env bash
# Start one inference server pinned to one GPU and block until it answers (or dies / times out).
#   ./serve.sh <gpu> <port> <label> <vllm|sglang> "<extra flags>" ["<ENV=VAL ...>"]
# Example:
#   ./serve.sh 3 8003 mtp_k3 vllm "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
# Stop it with: ./stop.sh <port>
set -uo pipefail
source "$(dirname "$0")/env.sh"
GPU=$1; PORT=$2; LABEL=$3; ENGINE=$4; FLAGS=${5:-}; ENVS=${6:-}
LOG="$RESULTS/logs/${LABEL}.server.log"

eval "set -- $FLAGS"   # split flags honouring quotes
if [[ $ENGINE == vllm ]]; then
  eval "set -- $VLLM_COMMON_FLAGS \"\$@\""
  CMD=("$VLLM_VENV/bin/vllm" serve "$MODEL_DIR" --served-model-name "$SERVED_NAME" --port "$PORT" "$@")
elif [[ $ENGINE == sglang ]]; then
  eval "set -- $SGLANG_COMMON_FLAGS \"\$@\""
  CMD=("$SGLANG_VENV/bin/python" -m sglang.launch_server --model-path "$MODEL_DIR" --served-model-name "$SERVED_NAME"
       --host 0.0.0.0 --port "$PORT" "$@")
else
  echo "unknown engine $ENGINE" >&2; exit 2
fi

echo "[$(date +%T)] gpu=$GPU port=$PORT $LABEL: ${CMD[*]}" | tee "$LOG"
# VLLM_SERVER_DEV_MODE=1 exposes /reset_prefix_cache (handy for manual checks; bench.py uses a nonce instead).
env CUDA_VISIBLE_DEVICES="$GPU" VLLM_SERVER_DEV_MODE=1 $ENVS "${CMD[@]}" >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$RESULTS/pids/$PORT.pid"

for ((t = 0; t < SERVE_TIMEOUT_S; t += 5)); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[$(date +%T)] $LABEL: server DIED during startup. Tail of $LOG:" >&2; tail -n 25 "$LOG" >&2; exit 1
  fi
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null; then
    echo "[$(date +%T)] $LABEL: ready on :$PORT after ${t}s"; exit 0
  fi
  sleep 5
done
echo "$LABEL: startup timed out after ${SERVE_TIMEOUT_S}s" >&2
"$(dirname "$0")/stop.sh" "$PORT"; exit 1
