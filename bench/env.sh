# Shared settings. `source env.sh` from every script. Edit here, not in the scripts.
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BENCH_DIR

export MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
export MODEL_DIR="${MODEL_DIR:-$HOME/models/nemotron-3.5-super-vl-nvfp4}"
export SERVED_NAME="nemotron"

export VLLM_VENV="${VLLM_VENV:-$HOME/venv-vllm}"
export SGLANG_VENV="${SGLANG_VENV:-$HOME/venv-sglang}"
export RESULTS="${RESULTS:-$BENCH_DIR/results}"
export TRACES="${TRACES:-$BENCH_DIR/traces}"

# Flags every vLLM lane gets. Text-only (vision encoder off) is lossless for a text benchmark.
# --reasoning-parser nemotron_v3 is from vLLM's Nemotron 3.5 Super VL launch blog.
# 1x H100 80GB: ~65 GB of NVFP4 weights leaves little room, so cap context at 40k and use more of the HBM.
# Hopper has no native FP4: vLLM should pick a weight-only FP4 kernel (dequantize to BF16 on the fly) —
# same weights, so still lossless. Check the server log for which kernel/backend it picked.
export VLLM_COMMON_FLAGS="${VLLM_COMMON_FLAGS:---trust-remote-code --max-model-len 40960 --gpu-memory-utilization 0.95 --max-num-batched-tokens 2048 --max-num-seqs 32 --reasoning-parser nemotron_v3 --limit-mm-per-prompt '{\"image\":0}'}"
export SGLANG_COMMON_FLAGS="--trust-remote-code --context-length 65536"

# Merged into every bench request (both our lanes and Fireworks). Use it to pin thinking on/off the same way
# for every endpoint. Check the model card for the exact chat_template_kwargs key before the clock starts.
export BENCH_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":true}}'

# Server startup timeout (first boot compiles kernels + captures CUDA graphs).
export SERVE_TIMEOUT_S=1500

mkdir -p "$RESULTS/logs" "$RESULTS/pids" "$TRACES"

# NVRTC (used by vLLM's HUMMING FP4 kernels on Hopper) dlopens libnvrtc-builtins by name; the pip-installed CUDA 13
# libs live inside the venv, which isn't on the loader path. Without this: "failed to open libnvrtc-builtins.so.13.0".
export LD_LIBRARY_PATH="$VLLM_VENV/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Reduces fragmentation when the HBM is this full (allocator setting only; no effect on outputs).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hosts whose NVIDIA driver is older than the CUDA 13 build (e.g. GH200 with driver 570 / CUDA 12.8) need NVIDIA's
# forward-compat libcuda: `sudo apt-get install cuda-compat-13-0` (from NVIDIA's CUDA apt repo). Only used if present.
if [[ -d /usr/local/cuda-13.0/compat ]]; then export LD_LIBRARY_PATH="/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH"; fi
true  # env.sh is sourced under `set -e`; it must always end with status 0
