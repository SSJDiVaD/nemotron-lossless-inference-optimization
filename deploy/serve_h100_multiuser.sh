#!/usr/bin/env bash
# 1x H100 80GB, MANY concurrent users (8-32 agent sessions).
# Measured vs baseline (agentic trace): c=8 TTFT 1.51s -> 0.30s (5.0x), aggregate 150 -> 378 tok/s (2.5x);
#                                      c=32 TTFT 11.5s -> 1.6s (7.1x), aggregate 178 -> 371 tok/s (2.1x).
source "$(dirname "$0")/common.sh"
exec "$VLLM" serve "$MODEL_DIR" "${COMMON[@]}" "${PC[@]}" \
  --gpu-memory-utilization 0.97 --max-num-batched-tokens 2048 --max-num-seqs 32
