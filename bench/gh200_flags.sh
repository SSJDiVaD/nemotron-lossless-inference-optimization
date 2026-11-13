# GH200-only common flags: --language-model-only skips video/image profiling, which hits an aarch64 gcc -march bug.
export VLLM_COMMON_FLAGS="--trust-remote-code --max-model-len 40960 --gpu-memory-utilization 0.95 --max-num-batched-tokens 2048 --max-num-seqs 32 --reasoning-parser nemotron_v3 --language-model-only"
