# Lane 2 — CPU KV offload as a second cache tier.
# On a full B200 the whole agentic working set probably fits in HBM (hybrid Mamba => small KV), so offload
# would show nothing. To demonstrate the effect, shrink the GPU KV pool with --gpu-memory-utilization so the
# 32-session trace overflows it, then compare with/without offload. Use MODE=full for this lane (32 sessions).
PC="--enable-prefix-caching"
SMALL="--gpu-memory-utilization 0.55"
# vLLM native CPU offloading (size in GiB).  VERIFY flag names on your version.
OFF="--kv-offloading-backend native --kv-offloading-size 96"
CONFIGS=(
  "pc_smallkv|vllm||$PC $SMALL"
  "pc_smallkv_offload|vllm||$PC $SMALL $OFF"
  "pc_fullkv_offload|vllm||$PC $OFF"
)
