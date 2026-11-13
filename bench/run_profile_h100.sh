#!/usr/bin/env bash
# Profile single-user decode on the interactive config: where does each ~2.8 ms decode step go?
cd ~/lambda-bench; source env.sh; source h100_flags.sh
D=$PWD/results/profile_h100; rm -rf $D; mkdir -p $D
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
./serve.sh 0 8000 h100_profile vllm "$PC --profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$D\"}'" || exit 1
"$VLLM_VENV/bin/python" bench.py --base-url http://localhost:8000/v1 --model "$SERVED_NAME" --label h100_profile_warm \
  --out-dir results --trace "$TRACES/lcb_short.jsonl" --max-sessions 1 --max-tokens 64 --concurrency 1 --extra-body "$BENCH_EXTRA_BODY"
curl -s -X POST http://localhost:8000/start_profile
"$VLLM_VENV/bin/python" bench.py --base-url http://localhost:8000/v1 --model "$SERVED_NAME" --label h100_profile_run \
  --out-dir results --trace "$TRACES/lcb_short.jsonl" --max-sessions 2 --max-tokens 128 --concurrency 1 --warmup 0 --extra-body "$BENCH_EXTRA_BODY"
curl -s -X POST http://localhost:8000/stop_profile; sleep 20
./stop.sh 8000
"$VLLM_VENV/bin/python" analyze_profile.py "$D" | tee results/profile_h100_summary.txt
echo PROFILE_DONE
