#!/usr/bin/env bash
# GH200: full-trace benchmark of the combined-best config with MTP k=6 (compare with gh200_full_final, k=3).
cd ~/lambda-bench; source env.sh; source gh200_flags.sh
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
SPEC="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":6}'"
./serve.sh 0 8000 gh200_full_k6_server vllm "$PC $SPEC --async-scheduling --max-num-batched-tokens 8192 --max-num-seqs 64" || exit 1
LEFT=$(( $(date -d "22:55" +%s) - $(date +%s) )); [ $LEFT -lt 60 ] && LEFT=60
timeout $LEFT ./bench_suite.sh http://localhost:8000/v1 "$SERVED_NAME" gh200_full_k6 full 2>&1 | tee results/logs/full_k6.log
./stop.sh 8000; echo FULLK6_DONE | tee -a results/logs/full_k6.log
