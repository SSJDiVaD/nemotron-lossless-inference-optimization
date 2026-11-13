#!/usr/bin/env bash
# Run every config in a lane file, unattended: start server -> short bench -> scrape /metrics -> stop.
# A config that fails to start is logged and SKIPPED, so one bad flag never stalls the lane.
#   ./sweep.sh <gpu> <port> configs/laneN.sh [short|full]
# KEEP_LAST=1 leaves the last config's server running (lane 0 baseline, lane 7 final).
set -uo pipefail
source "$(dirname "$0")/env.sh"
GPU=$1; PORT=$2; CONF=$3; MODE=${4:-short}
source "$CONF"   # defines CONFIGS=( "label|engine|ENV=VAL ...|flags" ... )

n=${#CONFIGS[@]}
for i in "${!CONFIGS[@]}"; do
  IFS='|' read -r LABEL ENGINE ENVS FLAGS <<< "${CONFIGS[$i]}"
  echo "=== [$(date +%T)] lane gpu$GPU config $((i + 1))/$n: $LABEL"
  if ! "$BENCH_DIR/serve.sh" "$GPU" "$PORT" "$LABEL" "$ENGINE" "$FLAGS" "$ENVS"; then
    echo "$(date +%T) $LABEL FAILED_TO_START" >> "$RESULTS/failures.txt"; continue
  fi
  "$BENCH_DIR/bench_suite.sh" "http://localhost:$PORT/v1" "$SERVED_NAME" "$LABEL" "$MODE"
  mkdir -p "$RESULTS/$LABEL"
  curl -s "http://localhost:$PORT/metrics" > "$RESULTS/$LABEL/metrics.prom" 2>/dev/null
  # Spec-decode acceptance, if any (vLLM metric names; harmless if absent).
  grep -E '^(vllm:spec_decode_num_(accepted|draft)_tokens_total|vllm:prefix_cache_(hits|queries)_total)' \
    "$RESULTS/$LABEL/metrics.prom" 2>/dev/null | tee "$RESULTS/$LABEL/key_metrics.txt"
  if [[ ${KEEP_LAST:-0} == 1 && $i == $((n - 1)) ]]; then
    echo "=== $LABEL left running on :$PORT"; exit 0
  fi
  "$BENCH_DIR/stop.sh" "$PORT"
done
echo "=== [$(date +%T)] lane gpu$GPU done"
