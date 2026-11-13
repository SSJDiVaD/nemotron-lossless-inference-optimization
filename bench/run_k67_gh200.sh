#!/usr/bin/env bash
# GH200: extend the MTP depth trend on the combined-best config (k=5 beat k=3). Hard stop 17:48 local (22:48 UTC).
cd ~/lambda-bench
until grep -q "HEADLINE_DONE" results/logs/sweep_gh200_headline.log 2>/dev/null; do sleep 5; done
cat > configs/gh200_k67.sh <<'CONF'
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
mtp() { echo "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$1}'"; }
BIG="--async-scheduling --max-num-batched-tokens 8192 --max-num-seqs 64"
CONFIGS=(
  "gh200_best_k6|vllm||$PC $(mtp 6) $BIG"
  "gh200_best_k7|vllm||$PC $(mtp 7) $BIG"
)
CONF
source env.sh; source gh200_flags.sh
LEFT=$(( $(date -d "22:48" +%s) - $(date +%s) )); [ $LEFT -lt 60 ] && LEFT=60
timeout $LEFT ./sweep.sh 0 8000 configs/gh200_k67.sh short 2>&1 | tee results/logs/sweep_gh200_k67.log
./stop.sh 8000; echo K67_DONE | tee -a results/logs/sweep_gh200_k67.log
