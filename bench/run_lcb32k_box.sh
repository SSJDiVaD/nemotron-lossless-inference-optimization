#!/usr/bin/env bash
# Accuracy box: LiveCodeBench at a 32k reasoning budget on our final config (prefix caching + MTP k3).
cd ~/lambda-bench
until grep -q "setup complete" setup.log 2>/dev/null; do sleep 15; done
source env.sh; source h100_flags.sh
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
SPEC="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
./serve.sh 0 8000 h100pcie_final vllm "$PC $SPEC" || exit 1
LCB_MAX_TOKENS=32768 ./bench_suite.sh http://localhost:8000/v1 "$SERVED_NAME" h100pcie_final_lcb32k lcb 2>&1 | tee results/logs/lcb32k.log
./stop.sh 8000; echo LCB32K_DONE
