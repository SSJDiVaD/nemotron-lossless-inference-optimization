#!/usr/bin/env bash
# Run the standard benchmark suite against ANY OpenAI-compatible endpoint (our lanes or Fireworks).
#   ./bench_suite.sh <base_url> <model> <label> <short|full|lcb> [api_key_env]
#   short: quick per-config sweep (~5 min)          -> used by sweep.sh
#   full : final comparison (~20 min)               -> final config, baseline, Fireworks
#   lcb  : LiveCodeBench accuracy run + grading      -> final config vs Fireworks
set -uo pipefail
source "$(dirname "$0")/env.sh"
URL=$1; MODEL=$2; LABEL=$3; MODE=$4; KEYENV=${5:-}
PY="$VLLM_VENV/bin/python"; [[ -x $PY ]] || PY=python3
B=("$PY" "$BENCH_DIR/bench.py" --base-url "$URL" --model "$MODEL" --label "$LABEL" --out-dir "$RESULTS"
   --extra-body "$BENCH_EXTRA_BODY")
[[ -n $KEYENV ]] && B+=(--api-key-env "$KEYENV")

case $MODE in
  short)
    "${B[@]}" --trace "$TRACES/agentic_short.jsonl" --concurrency 1 8 --max-tokens 256
    "${B[@]}" --trace "$TRACES/lcb_short.jsonl"     --concurrency 1 8 --max-tokens 2048
    ;;
  full)
    "${B[@]}" --trace "$TRACES/agentic.jsonl" --concurrency 1 --max-sessions 8 --max-tokens 256
    "${B[@]}" --trace "$TRACES/agentic.jsonl" --concurrency 8 32 --max-tokens 256 --warmup 0
    "${B[@]}" --trace "$TRACES/lcb.jsonl"     --concurrency 1 --max-sessions 8 --max-tokens 4096 --warmup 0
    "${B[@]}" --trace "$TRACES/lcb.jsonl"     --concurrency 8 --max-sessions 24 --max-tokens 4096 --warmup 0
    ;;
  lcb)
    # Accuracy run. Use the model card's recommended sampling for reasoning if it specifies one;
    # whatever you choose, use the SAME settings for every endpoint.
    "${B[@]}" --trace "$TRACES/lcb.jsonl" --concurrency 16 --max-tokens "${LCB_MAX_TOKENS:-8192}" --save-outputs \
      --temperature "${LCB_TEMPERATURE:-1.0}" --top-p "${LCB_TOP_P:-0.95}" --warmup 0
    "$PY" "$BENCH_DIR/lcb_grade.py" --tests "$TRACES/lcb.jsonl.tests" --outputs "$RESULTS/$LABEL/lcb_c16.jsonl"
    ;;
  *) echo "mode must be short|full|lcb" >&2; exit 2 ;;
esac
