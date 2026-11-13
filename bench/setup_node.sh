#!/usr/bin/env bash
# Run ONCE on the fresh Lambda node (Ubuntu + Lambda Stack). ~15-20 min, mostly the weight download.
#   VLLM_VERSION=0.x.y ./setup_node.sh    # pin a version you tested pre-clock (recommended)
set -euo pipefail
source "$(dirname "$0")/env.sh"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
df -h "$HOME" | tail -1   # need ~150 GB free: weights + venvs + caches

sudo apt-get update -qq && sudo apt-get install -y -qq tmux jq >/dev/null
# vLLM's wheels are built for CUDA 13; drivers older than 580 (CUDA < 13) need NVIDIA's forward-compat libcuda.
DRV_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1)
if [[ ${DRV_CUDA:-0} -lt 13 ]]; then
  . /etc/os-release; ARCH=$([[ $(uname -m) == aarch64 ]] && echo sbsa || echo x86_64)
  wget -q -O /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/$ARCH/cuda-keyring_1.1-1_all.deb" \
    && sudo dpkg -i /tmp/cuda-keyring.deb >/dev/null && sudo apt-get update -qq \
    && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0 >/dev/null && echo "installed cuda-compat-13-0"
fi
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

# 1) vLLM env + weight download in parallel (download is the long pole).
uv venv -q -p 3.12 "$VLLM_VENV"
uv pip install -q -p "$VLLM_VENV" "huggingface_hub[hf_transfer]" httpx numpy matplotlib
( HF_HUB_ENABLE_HF_TRANSFER=1 "$VLLM_VENV/bin/hf" download "$MODEL_ID" --local-dir "$MODEL_DIR" \
    > "$RESULTS/logs/download.log" 2>&1 && echo "download done" ) &
DL=$!
uv pip install -q -p "$VLLM_VENV" "vllm${VLLM_VERSION:+==$VLLM_VERSION}" pip ${LMCACHE:+lmcache}
# Without these, vLLM silently falls back to slower non-FlashInfer kernels (no nvcc on Lambda Stack).
"$VLLM_VENV/bin/vllm" download-kernels 2>&1 | tail -n 3

# 2) SGLang env (lane 5) in the background; it's a stretch lane, don't block on it.
( uv venv -q -p 3.12 "$SGLANG_VENV" && uv pip install -q -p "$SGLANG_VENV" "sglang[all]" httpx \
    > "$RESULTS/logs/sglang_install.log" 2>&1 && echo "sglang installed" ) &

# 3) Traces (CPU only, runs while weights download). Skip if the kit already ships them: every box must benchmark
#    byte-identical traces, and regenerating depends on the local Python stdlib version.
PY="$VLLM_VENV/bin/python"
if [[ -s "$TRACES/agentic.jsonl" && -s "$TRACES/lcb.jsonl.tests" ]]; then
  echo "traces already present, not regenerating"; md5sum "$TRACES"/*.jsonl*
else
  $PY "$BENCH_DIR/make_agentic_trace.py" --out "$TRACES/agentic.jsonl" --sessions 32 --turns 5 --shared-tokens 16000 --session-tokens 6000
  $PY "$BENCH_DIR/make_agentic_trace.py" --out "$TRACES/agentic_short.jsonl" --sessions 8 --turns 3 --shared-tokens 16000 --session-tokens 6000 --seed 1
  $PY "$BENCH_DIR/make_lcb_trace.py" --out "$TRACES/lcb.jsonl" --n 60
  head -n 8 "$TRACES/lcb.jsonl" > "$TRACES/lcb_short.jsonl"
fi

wait $DL
du -sh "$MODEL_DIR"
"$VLLM_VENV/bin/vllm" --version
echo "setup complete. Next: ./serve.sh 0 8000 smoke vllm '' && curl localhost:8000/v1/models"
