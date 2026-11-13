#!/usr/bin/env bash
# Accuracy box, 2nd job: BASELINE config at the same 32k budget => baseline-vs-optimized accuracy on the same GPU.
cd ~/lambda-bench
until grep -q "LCB32K_DONE" results/logs/lcb32k_job.log 2>/dev/null; do sleep 15; done
source env.sh; source h100_flags.sh
./serve.sh 0 8000 h100pcie_base vllm "--disable-prefix-caching" || exit 1
LCB_MAX_TOKENS=32768 ./bench_suite.sh http://localhost:8000/v1 "$SERVED_NAME" h100pcie_base_lcb32k lcb 2>&1 | tee results/logs/base_lcb32k.log
./stop.sh 8000; echo BASE32K_DONE
