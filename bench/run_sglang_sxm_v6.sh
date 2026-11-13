#!/usr/bin/env bash
# SGLang final attempt: JIT toolchain from pip (ninja + nvcc 13) on PATH / CUDA_HOME. Hard stop 17:48 local.
cd ~/lambda-bench
until grep -q "SGLANG_DONE" results/logs/sweep_h100_sglang_v5.log 2>/dev/null; do sleep 5; done
NV=$HOME/venv-sglang/lib/python3.12/site-packages/nvidia/cu13
export PATH="$HOME/venv-sglang/bin:$NV/bin:$PATH" CUDA_HOME="$NV"
COMMON="--context-length 40960 --mem-fraction-static 0.96 --chunked-prefill-size 2048 --max-running-requests 16 --mamba-radix-cache-strategy extra_buffer"
cat > configs/h100_sglang_v6.sh <<CONF
CONFIGS=(
  "h100_sglang_radix|sglang|CUDA_HOME=$NV|$COMMON"
  "h100_sglang_radix_mtp3|sglang|CUDA_HOME=$NV|$COMMON --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
)
CONF
source env.sh
LEFT=$(( $(date -d "22:48" +%s) - $(date +%s) )); [ $LEFT -lt 60 ] && LEFT=60
timeout $LEFT ./sweep.sh 0 8000 configs/h100_sglang_v6.sh short 2>&1 | tee results/logs/sweep_h100_sglang_v6.log
./stop.sh 8000; echo SGLANG_DONE | tee -a results/logs/sweep_h100_sglang_v6.log
