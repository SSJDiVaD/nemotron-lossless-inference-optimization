# H100 SXM box — the core ablation story. Speedups are vs THIS box's baseline only.
#   ./sweep.sh 0 8000 configs/h100.sh short
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
mtp() { echo "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$1}'"; }
CONFIGS=(
  "h100_baseline_naive|vllm||--disable-prefix-caching"
  "h100_pc|vllm||$PC"
  "h100_mtp_k3|vllm||--disable-prefix-caching $(mtp 3)"
  "h100_combo_pc_mtp|vllm||$PC $(mtp 3)"
  "h100_baseline_naive_rep|vllm||--disable-prefix-caching"   # repeat => noise floor
)
