#!/usr/bin/env bash
# Redo H100 greedy captures with reasoning text saved (the first pass saved only content, mostly empty).
cd ~/lambda-bench
until grep -q "FINAL DONE" results/logs/final_h100.log 2>/dev/null; do sleep 15; done
source env.sh; source h100_flags.sh
PY="$VLLM_VENV/bin/python"; URL=http://localhost:8000/v1; R=results
greedy() { "$PY" bench.py --base-url $URL --model "$SERVED_NAME" --label "$1" --out-dir $R --trace "$TRACES/lcb.jsonl" \
  --max-sessions 40 --max-tokens 256 --temperature 0 --concurrency "$2" --save-outputs --warmup 1 --extra-body "$BENCH_EXTRA_BODY"; }
rm -rf $R/h100_greedy_*
./serve.sh 0 8000 h100_greedy_baseserver vllm "--disable-prefix-caching" && greedy h100_greedy_base_a 8 && greedy h100_greedy_base_b 1; ./stop.sh 8000
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
SPEC="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
./serve.sh 0 8000 h100_greedy_finalserver vllm "$PC $SPEC" && greedy h100_greedy_final 8; ./stop.sh 8000
"$PY" compare_outputs.py $R/h100_greedy_base_a/lcb_c8.jsonl $R/h100_greedy_base_b/lcb_c1.jsonl --name h100_base_vs_base --results $R
"$PY" compare_outputs.py $R/h100_greedy_base_a/lcb_c8.jsonl $R/h100_greedy_final/lcb_c8.jsonl --name h100_base_vs_final --results $R
echo GREEDY_REDO_DONE
