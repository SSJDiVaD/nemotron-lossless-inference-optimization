#!/usr/bin/env bash
# 1x H100 80GB, ONE user at a time (e.g. a developer's coding agent).
# Measured vs baseline (agentic trace, c=1): TTFT 1.41s -> 0.12s (12x), decode 153 -> 351 tok/s (2.3x).
# Do NOT use for multi-user serving: MTP's draft heads take KV memory and prefix-cache hits collapse (c=32 TTFT 21s).
source "$(dirname "$0")/common.sh"
exec "$VLLM" serve "$MODEL_DIR" "${COMMON[@]}" "${PC[@]}" "${MTP3[@]}" \
  --gpu-memory-utilization 0.97 --max-num-batched-tokens 2048 --max-num-seqs 32
