#!/usr/bin/env bash
# SGLang retry without --language-only (in SGLang that flag means encoder-disaggregation mode, unsupported for this
# model; it is not a text-only switch). The vision encoder (~1.6 GiB) is loaded but unused. Hard stop 17:22 local.
cd ~/lambda-bench
until grep -q "SGLANG_DONE" results/logs/sweep_h100_sglang.log 2>/dev/null; do sleep 5; done
COMMON="--context-length 40960 --mem-fraction-static 0.90 --chunked-prefill-size 2048 --max-running-requests 32 --mamba-radix-cache-strategy extra_buffer"
cat > configs/h100_sglang_v3.sh <<CONF
CONFIGS=(
  "h100_sglang_radix|sglang||$COMMON"
  "h100_sglang_radix_mtp3|sglang||$COMMON --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
)
CONF
source env.sh
LEFT=$(( $(date -d "${SGLANG_STOP_UTC:-22:22}" +%s) - $(date +%s) )); [ $LEFT -lt 60 ] && LEFT=60
timeout $LEFT ./sweep.sh 0 8000 configs/h100_sglang_v3.sh short 2>&1 | tee results/logs/sweep_h100_sglang_v3.log
./stop.sh 8000; echo SGLANG_DONE | tee -a results/logs/sweep_h100_sglang_v3.log
