#!/usr/bin/env bash
# Start the SAME baseline config on all 8 GPUs, benchmark them simultaneously, stop them.
# The spread across lanes is the noise floor: a lane "win" smaller than this spread is not a win.
# Doubles as the kernel/compile-cache warm-up so later lane startups are faster.
set -uo pipefail
source "$(dirname "$0")/env.sh"
NGPU=${NGPU:-8}
for g in $(seq 0 $((NGPU - 1))); do
  "$BENCH_DIR/serve.sh" "$g" $((8000 + g)) "noise_g$g" vllm "--disable-prefix-caching" &
  sleep 20   # stagger so the first process populates the compile caches
done
wait
for g in $(seq 0 $((NGPU - 1))); do
  "$BENCH_DIR/bench_suite.sh" "http://localhost:$((8000 + g))/v1" "$SERVED_NAME" "noise_g$g" short &
done
wait
for g in $(seq 0 $((NGPU - 1))); do "$BENCH_DIR/stop.sh" $((8000 + g)) & done
wait
"$VLLM_VENV/bin/python" "$BENCH_DIR/summarize.py" --results "$RESULTS" --only noise_g
