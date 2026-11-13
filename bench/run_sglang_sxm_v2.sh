#!/usr/bin/env bash
# H100 SXM, after the 8k baseline: SGLang, matched to the vLLM configs as closely as SGLang's options allow.
#  1) multi-user config: radix cache (Mamba extra_buffer strategy), text-only, 40k ctx, 2048 chunk, 32 running reqs
#  2) interactive config: + MTP k=3 (SGLang EAGLE-style flags per the model card: 3 steps, top-k 1, 4 draft tokens)
# mem-fraction-static 0.90 (SGLang semantics differ from vLLM's 0.97). Not used (lossy): --enable-int8-mamba-checkpoint.
# Hard stop at 17:22 local (22:22 UTC) so results can be collected before the deadline.
cd ~/lambda-bench
until grep -q "BASE8K_DONE" results/logs/base_lcb8k.log 2>/dev/null; do sleep 10; done
COMMON="--language-only --context-length 40960 --mem-fraction-static 0.90 --chunked-prefill-size 2048 --max-running-requests 32 --mamba-radix-cache-strategy extra_buffer"
cat > configs/h100_sglang.sh <<CONF
CONFIGS=(
  "h100_sglang_radix|sglang||$COMMON"
  "h100_sglang_radix_mtp3|sglang||$COMMON --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
)
CONF
source env.sh
LEFT=$(( $(date -d "22:22" +%s) - $(date +%s) )); [ $LEFT -lt 60 ] && LEFT=60
timeout $LEFT ./sweep.sh 0 8000 configs/h100_sglang.sh short 2>&1 | tee results/logs/sweep_h100_sglang.log
./stop.sh 8000; echo SGLANG_DONE | tee -a results/logs/sweep_h100_sglang.log
