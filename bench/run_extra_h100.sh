#!/usr/bin/env bash
# After the H100 final: can a shallower draft (k=2, fewer Mamba checkpoints) or a different MoE kernel
# reduce the prefix-caching/MTP memory interference seen at c=8?
cd ~/lambda-bench
until grep -q "GREEDY_REDO_DONE" results/logs/greedy_redo_h100.log 2>/dev/null; do sleep 15; done
cat > configs/h100_extra.sh <<'CONF'
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
CONFIGS=(
  "h100_combo_pc_mtp2|vllm||$PC --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'"
  "h100_combo_pc_mtp3_marlin|vllm||$PC --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' --moe-backend marlin"
)
CONF
source h100_flags.sh && ./sweep.sh 0 8000 configs/h100_extra.sh short 2>&1 | tee results/logs/sweep_h100_extra.log
