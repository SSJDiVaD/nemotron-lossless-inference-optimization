# Queued after configs/gh200.sh: prefix caching alone, to test whether the GH200's 19 GiB KV cache keeps caching
# effective at concurrency 8 (it thrashed on the H100's ~1 GiB).
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
CONFIGS=(
  "gh200_pc|vllm||$PC"
)
