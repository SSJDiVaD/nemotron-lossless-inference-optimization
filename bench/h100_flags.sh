# H100 v2 common flags: --language-model-only (skip unused vision encoder, ~1.6 GiB) + 0.97 util so MTP's draft heads fit.
export VLLM_COMMON_FLAGS="--trust-remote-code --max-model-len 40960 --gpu-memory-utilization 0.97 --max-num-batched-tokens 2048 --max-num-seqs 32 --reasoning-parser nemotron_v3 --language-model-only"
