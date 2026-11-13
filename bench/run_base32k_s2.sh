#!/usr/bin/env bash
cd ~/lambda-bench
until grep -q "BASE32K_DONE" results/logs/base_lcb32k.log 2>/dev/null; do sleep 10; done
source env.sh; source h100_flags.sh
./serve.sh 0 8000 h100pcie_base_s2 vllm "--disable-prefix-caching" || exit 1
LEFT=$(( $(date -d "22:48" +%s) - $(date +%s) )); [ $LEFT -lt 60 ] && LEFT=60
LCB_MAX_TOKENS=32768 timeout $LEFT ./bench_suite.sh http://localhost:8000/v1 "$SERVED_NAME" h100pcie_base_lcb32k_s2 lcb 2>&1 | tee results/logs/base_lcb32k_s2.log
./stop.sh 8000; echo BASE32K_S2_DONE | tee -a results/logs/base_lcb32k_s2.log
