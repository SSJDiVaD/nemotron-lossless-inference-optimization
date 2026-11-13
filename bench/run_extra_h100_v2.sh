#!/usr/bin/env bash
# H100 extras (after greedy redo): performance modes, interference fixes, then a decode-step profile.
cd ~/lambda-bench
until grep -q "GREEDY_REDO_DONE" results/logs/greedy_redo_h100.log 2>/dev/null; do sleep 15; done
cat > configs/h100_extra_v2.sh <<'CONF'
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
SPEC3="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
CONFIGS=(
  "h100_combo_interactivity|vllm||$PC $SPEC3 --performance-mode interactivity"
  "h100_pc_throughput|vllm||$PC --performance-mode throughput"
  "h100_combo_pc_mtp2|vllm||$PC --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'"
)
CONF
source h100_flags.sh && ./sweep.sh 0 8000 configs/h100_extra_v2.sh short 2>&1 | tee results/logs/sweep_h100_extra.log
./run_profile_h100.sh 2>&1 | tee results/logs/profile_h100.log
