#!/usr/bin/env bash
# GH200 after the 2nd 32k sample: try to push the headline config further. Hard stop 17:48 local (22:48 UTC).
cd ~/lambda-bench
until grep -q "POST_DONE" results/logs/gh200_lcb32k.log 2>/dev/null || tmux capture-pane -p -t main:post -S -50 | grep -q POST_DONE; do sleep 10; done
cat > configs/gh200_headline.sh <<'CONF'
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
mtp() { echo "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$1}'"; }
CONFIGS=(
  "gh200_best_k5|vllm||$PC $(mtp 5) --async-scheduling --max-num-batched-tokens 8192 --max-num-seqs 64"
  "gh200_best_chunk16k|vllm||$PC $(mtp 3) --async-scheduling --max-num-batched-tokens 16384 --max-num-seqs 64"
)
CONF
source env.sh; source gh200_flags.sh
LEFT=$(( $(date -d "22:48" +%s) - $(date +%s) )); [ $LEFT -lt 60 ] && LEFT=60
timeout $LEFT ./sweep.sh 0 8000 configs/gh200_headline.sh short 2>&1 | tee results/logs/sweep_gh200_headline.log
./stop.sh 8000; echo HEADLINE_DONE | tee -a results/logs/sweep_gh200_headline.log
