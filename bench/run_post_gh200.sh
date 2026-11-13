#!/usr/bin/env bash
# GH200 after its final phase: performance modes on the combined-best config, then a 2nd 32k accuracy sample.
cd ~/lambda-bench
until grep -q "FINAL DONE" results/logs/final_gh200.log 2>/dev/null; do sleep 15; done
cat > configs/gh200_post.sh <<'CONF'
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
BEST="$PC --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' --async-scheduling --max-num-batched-tokens 8192 --max-num-seqs 64"
CONFIGS=(
  "gh200_best_throughput|vllm||$BEST --performance-mode throughput"
  "gh200_best_interactivity|vllm||$BEST --performance-mode interactivity"
)
CONF
source gh200_flags.sh && ./sweep.sh 0 8000 configs/gh200_post.sh short 2>&1 | tee results/logs/sweep_gh200_post.log
source env.sh; source gh200_flags.sh
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
SPEC="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
./serve.sh 0 8000 gh200_lcb32k_server vllm "$PC $SPEC --async-scheduling --max-num-batched-tokens 8192 --max-num-seqs 64" && \
  LCB_MAX_TOKENS=32768 ./bench_suite.sh http://localhost:8000/v1 "$SERVED_NAME" gh200_final_lcb32k lcb 2>&1 | tee results/logs/gh200_lcb32k.log
./stop.sh 8000; echo POST_DONE
