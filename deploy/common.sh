# Shared by the serve_*.sh scripts. Assumes ./setup.sh has been run on this machine.
MODEL_DIR="${MODEL_DIR:-$HOME/models/nemotron-3.5-super-vl-nvfp4}"   # nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
VLLM="${VLLM_VENV:-$HOME/venv-vllm}/bin/vllm"
SITE="${VLLM_VENV:-$HOME/venv-vllm}/lib/python3.12/site-packages"
# vLLM's HUMMING NVFP4 kernels (used on Hopper) JIT-compile with NVRTC, which dlopens libnvrtc-builtins by name.
export LD_LIBRARY_PATH="$SITE/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Drivers older than 580 (CUDA < 13, e.g. Lambda's GH200 / H100 PCIe images) need NVIDIA's forward-compat libcuda.
if [[ -d /usr/local/cuda-13.0/compat ]]; then export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Flags shared by every config. All lossless:
#  --language-model-only : skip the unused vision encoder (frees ~1.6 GiB of weights + its activation reserve => 6x KV
#                          cache on an 80 GB H100); also avoids an aarch64 gcc bug in video-input profiling on GH200.
#  KV cache dtype is left at the checkpoint's own setting (FP8 with calibrated scales, from hf_quant_config.json).
#  Mamba SSM state is left at its default precision (the model card's float16 + stochastic rounding is NOT used).
COMMON=(--served-model-name nemotron --trust-remote-code --max-model-len 40960 --reasoning-parser nemotron_v3
        --language-model-only --host 0.0.0.0 --port "${PORT:-8000}")
# Prefix caching for the hybrid Mamba model: 64-token blocks, cache hits at 64-token granularity, and Mamba state
# checkpoints at the shared-prefix junction (needed for hits when MTP speculation is on).
PC=(--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64)
MTP3=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
