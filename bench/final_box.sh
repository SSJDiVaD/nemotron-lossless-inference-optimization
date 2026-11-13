#!/usr/bin/env bash
# Final phase on ONE single-GPU box. Runs unattended after the sweep:
#   1. baseline server: greedy outputs at c=8 (base_a) and c=1 (base_b, different batching => FP-drift floor); full bench
#   2. final server   : greedy outputs at c=8; LiveCodeBench accuracy (60 problems); full bench
#   3. optional extra : full bench of one more config (e.g. prefix-caching-only on the H100)
#   4. offline greedy comparisons
# Usage (env): P=h100 FLAGS=h100_flags.sh FINAL="<flags>" [EXTRA_LABEL=... EXTRA="<flags>"] ./final_box.sh
set -uo pipefail
cd "$(dirname "$0")"; source env.sh; source "$FLAGS"
PY="$VLLM_VENV/bin/python"; URL=http://localhost:8000/v1
BASE="--disable-prefix-caching"
greedy() {  # label concurrency
  "$PY" bench.py --base-url $URL --model "$SERVED_NAME" --label "$1" --out-dir "$RESULTS" --trace "$TRACES/lcb.jsonl" \
    --max-sessions 40 --max-tokens 256 --temperature 0 --concurrency "$2" --save-outputs --warmup 1 \
    --extra-body "$BENCH_EXTRA_BODY"
}
echo "=== [$(date +%T)] final phase for $P"
./serve.sh 0 8000 "${P}_final_base" vllm "$BASE" || exit 1
greedy "${P}_greedy_base_a" 8
greedy "${P}_greedy_base_b" 1
./bench_suite.sh $URL "$SERVED_NAME" "${P}_full_baseline" full
./stop.sh 8000

./serve.sh 0 8000 "${P}_final" vllm "$FINAL" || exit 1
greedy "${P}_greedy_final" 8
./bench_suite.sh $URL "$SERVED_NAME" "${P}_final_lcb" lcb
./bench_suite.sh $URL "$SERVED_NAME" "${P}_full_final" full
curl -s http://localhost:8000/metrics > "$RESULTS/${P}_full_final/metrics.prom" 2>/dev/null
./stop.sh 8000

if [[ -n "${EXTRA:-}" ]]; then
  ./serve.sh 0 8000 "$EXTRA_LABEL" vllm "$EXTRA" && ./bench_suite.sh $URL "$SERVED_NAME" "${EXTRA_LABEL}" full; ./stop.sh 8000
fi

R="$RESULTS"
"$PY" compare_outputs.py "$R/${P}_greedy_base_a/lcb_c8.jsonl" "$R/${P}_greedy_base_b/lcb_c1.jsonl" --name "${P}_base_vs_base" --results "$R"
"$PY" compare_outputs.py "$R/${P}_greedy_base_a/lcb_c8.jsonl" "$R/${P}_greedy_final/lcb_c8.jsonl" --name "${P}_base_vs_final" --results "$R"
echo "=== [$(date +%T)] FINAL DONE for $P"
