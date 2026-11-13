# GH200 (96 GB HBM3e) box — speculation-depth sweep, scheduling, and what the extra HBM buys.
# Speedups are vs THIS box's baseline only (different hardware from the H100).
#   ./sweep.sh 0 8000 configs/gh200.sh short
# Common flags (env.sh) keep the same caps as the H100 for comparability; later flags override earlier ones.
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
mtp() { echo "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$1}'"; }
CONFIGS=(
  "gh200_baseline_naive|vllm||--disable-prefix-caching"
  "gh200_mtp_k2|vllm||--disable-prefix-caching $(mtp 2)"
  "gh200_mtp_k3|vllm||--disable-prefix-caching $(mtp 3)"
  "gh200_mtp_k5|vllm||--disable-prefix-caching $(mtp 5)"
  "gh200_combo_pc_mtp_async|vllm||$PC $(mtp 3) --async-scheduling"
  "gh200_combo_bigbatch|vllm||$PC $(mtp 3) --max-num-batched-tokens 8192 --max-num-seqs 64"   # uses the extra 16 GB
  "gh200_baseline_naive_rep|vllm||--disable-prefix-caching"
)
