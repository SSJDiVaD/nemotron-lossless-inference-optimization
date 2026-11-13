#!/usr/bin/env bash
# Second H100 PCIe box: BASELINE config at the 32k budget (same GPU type as the final-config 32k run).
cd ~/lambda-bench; source env.sh; source h100_flags.sh
./serve.sh 0 8000 h100pcie_base vllm "--disable-prefix-caching" || exit 1
LCB_MAX_TOKENS=32768 ./bench_suite.sh http://localhost:8000/v1 "$SERVED_NAME" h100pcie_base_lcb32k lcb 2>&1 | tee results/logs/base_lcb32k.log
./stop.sh 8000; echo BASE32K_DONE | tee -a results/logs/base_lcb32k.log
