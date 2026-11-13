# 1x B200 box: baseline + the combined best-of-sweeps config (Blackwell runs NVFP4 natively).
# Speedups are vs THIS box's baseline only. Model card recommends MTP k=5 on Blackwell, so run both 3 and 5.
#   source gh200_flags.sh && ./sweep.sh 0 8000 configs/b200.sh short
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
mtp() { echo "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$1}'"; }
BIG="--max-num-batched-tokens 8192 --max-num-seqs 64 --async-scheduling"
CONFIGS=(
  "b200_baseline_naive|vllm||--disable-prefix-caching"
  "b200_best_k3|vllm||$PC $(mtp 3) $BIG"
  "b200_best_k5|vllm||$PC $(mtp 5) $BIG"
)
