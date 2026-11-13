#!/usr/bin/env bash
# H100 SXM, after the 8k baseline: SGLang engine comparison, matched to the vLLM multi-user config
# (radix/prefix cache with Mamba-aware strategy from the model card, text-only, 40k context, 2048-token prefill
# chunks, 32 running requests). Not used (lossy): --enable-int8-mamba-checkpoint. Hard cap: 20 min total.
cd ~/lambda-bench
until grep -q "BASE8K_DONE" results/logs/base_lcb8k.log 2>/dev/null; do sleep 10; done
cat > configs/h100_sglang.sh <<'CONF'
CONFIGS=(
  "h100_sglang_radix|sglang||--language-only --context-length 40960 --mem-fraction-static 0.90 --chunked-prefill-size 2048 --max-running-requests 32 --mamba-radix-cache-strategy extra_buffer"
)
CONF
source env.sh
timeout 1200 ./sweep.sh 0 8000 configs/h100_sglang.sh short 2>&1 | tee results/logs/sweep_h100_sglang.log
./stop.sh 8000; echo SGLANG_DONE | tee -a results/logs/sweep_h100_sglang.log
