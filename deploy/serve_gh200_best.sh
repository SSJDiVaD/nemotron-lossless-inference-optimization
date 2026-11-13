#!/usr/bin/env bash
# 1x GH200 (96 GB HBM3e): enough KV memory that prefix caching and MTP reinforce each other instead of competing.
# Measured vs baseline (agentic trace): c=1 TTFT 1.45s -> 0.12s (12x), decode 161 -> ~370 tok/s (2.3x);
#                                      c=8 TTFT 2.1s -> 0.25s, aggregate 145 -> 526 tok/s (3.6x).
# Option: MTP k=5 instead of 3 measured better on the short agent trace (c=1 decode 417 vs 367 tok/s; c=8 aggregate
# 554 vs 526 tok/s) and equal on code generation. Set MTP_K=5 to use it (full-trace headline numbers above are k=3).
source "$(dirname "$0")/common.sh"
exec "$VLLM" serve "$MODEL_DIR" "${COMMON[@]}" "${PC[@]}" --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K:-3}}" --async-scheduling \
  --gpu-memory-utilization 0.95 --max-num-batched-tokens 8192 --max-num-seqs 64
