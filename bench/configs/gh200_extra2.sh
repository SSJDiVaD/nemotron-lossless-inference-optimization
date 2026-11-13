# Kernel-level experiments on top of the best GH200 combo (prefix caching + MTP k3 + async).
# All lossless: same weights, different kernels. Excluded as lossy: VLLM_MARLIN_INPUT_DTYPE=int8/fp8 (quantizes
# activations) and VLLM_HUMMING_USE_F16_ACCUM (lower-precision accumulation).
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
COMBO="$PC --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' --async-scheduling"
CONFIGS=(
  "gh200_combo_moe_marlin|vllm||$COMBO --moe-backend marlin"
  "gh200_combo_humming_grouped|vllm|VLLM_HUMMING_MOE_GEMM_TYPE=grouped|$COMBO"
  "gh200_combo_O3|vllm||$COMBO -O3"
)
