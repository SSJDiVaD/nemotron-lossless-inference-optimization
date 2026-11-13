# Lane 6 — kernels, compilation and scheduler overhead (small-active-param MoE decode is often CPU/launch bound).
#  - FlashInfer NVFP4 MoE kernels on Blackwell, latency-tuned variant.        VERIFY env var names
#  - -O3: vLLM's highest compilation level (longer startup).                     VERIFY
#  - async scheduling overlaps CPU scheduling with GPU execution.
BASE="--disable-prefix-caching"
CONFIGS=(
  "fi_moe_latency|vllm|VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_FLASHINFER_MOE_BACKEND=latency|$BASE"
  "compile_O3|vllm||$BASE -O3"
  "fi_latency_O3_async|vllm|VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_FLASHINFER_MOE_BACKEND=latency|$BASE -O3 --async-scheduling"
)
