#!/usr/bin/env bash
# SGLang retry #3: disable DeepGEMM JIT (needs nvcc >= 12.9, absent on Lambda Stack). Hard stop 17:48 local.
cd ~/lambda-bench
until grep -q "SGLANG_DONE" results/logs/sweep_h100_sglang_v3.log 2>/dev/null; do sleep 5; done
COMMON="--context-length 40960 --mem-fraction-static 0.90 --chunked-prefill-size 2048 --max-running-requests 32 --mamba-radix-cache-strategy extra_buffer"
ENVS="SGLANG_ENABLE_JIT_DEEPGEMM=0 SGL_ENABLE_JIT_DEEPGEMM=0"
cat > configs/h100_sglang_v4.sh <<CONF
CONFIGS=(
  "h100_sglang_radix|sglang|$ENVS|$COMMON"
  "h100_sglang_radix_mtp3|sglang|$ENVS|$COMMON --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
)
CONF
source env.sh
LEFT=$(( $(date -d "22:48" +%s) - $(date +%s) )); [ $LEFT -lt 60 ] && LEFT=60
timeout $LEFT ./sweep.sh 0 8000 configs/h100_sglang_v4.sh short 2>&1 | tee results/logs/sweep_h100_sglang_v4.log
./stop.sh 8000; echo SGLANG_DONE | tee -a results/logs/sweep_h100_sglang_v4.log
