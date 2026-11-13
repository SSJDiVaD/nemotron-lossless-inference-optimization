#!/usr/bin/env bash
# H100 SXM: BASELINE config LiveCodeBench at the 8k budget (pairs with h100_final_lcb on the same GPU).
cd ~/lambda-bench; source env.sh; source h100_flags.sh
./serve.sh 0 8000 h100_base_lcbserver vllm "--disable-prefix-caching" || exit 1
./bench_suite.sh http://localhost:8000/v1 "$SERVED_NAME" h100_base_lcb lcb 2>&1 | tee results/logs/base_lcb8k.log
./stop.sh 8000; echo BASE8K_DONE | tee -a results/logs/base_lcb8k.log
