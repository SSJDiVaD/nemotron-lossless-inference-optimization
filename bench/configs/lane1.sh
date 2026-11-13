# Lane 1 — prefix caching (the #1 TTFT lever) + chunked-prefill budget.
# Hybrid Mamba models need (experimental upstream; requires chunked prefill).  VERIFY
PC="--enable-prefix-caching"
CONFIGS=(
  "pc_align|vllm||$PC"
  "pc_align_mbt2k|vllm||$PC --max-num-batched-tokens 2048"
  "pc_align_mbt8k|vllm||$PC --max-num-batched-tokens 8192"
  "pc_align_mbt16k|vllm||$PC --max-num-batched-tokens 16384"
)
