#!/usr/bin/env bash
# One-time node setup for the deploy/serve_*.sh scripts (tested on Lambda: H100 SXM x86_64, H100 PCIe x86_64, GH200 aarch64;
# Ubuntu 22.04, Lambda Stack image). ~10-20 min, mostly the ~70 GB weight download.
set -euo pipefail
VLLM_VERSION="${VLLM_VERSION:-0.33.0}"
VENV="${VLLM_VENV:-$HOME/venv-vllm}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/nemotron-3.5-super-vl-nvfp4}"
MODEL_REV="${MODEL_REV:-3acb09951cff50cda7be0200ebbad2c15a3e63ba}"   # revision benchmarked

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# vLLM 0.33 wheels target CUDA 13. Drivers < 580 (CUDA 12.x) need NVIDIA's forward-compat package (no reboot needed).
DRV_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1)
if [[ ${DRV_CUDA:-0} -lt 13 ]]; then
  . /etc/os-release; ARCH=$([[ $(uname -m) == aarch64 ]] && echo sbsa || echo x86_64)
  wget -q -O /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/$ARCH/cuda-keyring_1.1-1_all.deb"
  sudo dpkg -i /tmp/cuda-keyring.deb >/dev/null && sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0 >/dev/null && echo "installed cuda-compat-13-0"
fi
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv venv -q -p 3.12 "$VENV"
uv pip install -q -p "$VENV" "vllm==$VLLM_VERSION" pip "huggingface_hub[hf_transfer]"
# Without these, vLLM silently falls back to slower non-FlashInfer kernels (Lambda Stack has no nvcc).
"$VENV/bin/vllm" download-kernels 2>&1 | tail -n 2
HF_HUB_ENABLE_HF_TRANSFER=1 "$VENV/bin/hf" download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 \
  --revision "$MODEL_REV" --local-dir "$MODEL_DIR"
echo "done. Start one of: ./serve_h100_interactive.sh | ./serve_h100_multiuser.sh | ./serve_gh200_best.sh"
