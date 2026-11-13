# Lane 0 — reference. Run with KEEP_LAST=1 so baseline_naive stays up for greedy_equiv.py.
CONFIGS=(
  # Whatever this vLLM version does out of the box (report it: it's what a naive user gets).
  "vllm_defaults|vllm||"
  # Ablation baseline: no prefix caching, no speculation. Every lane's gain is measured against this.
  "baseline_naive|vllm||--disable-prefix-caching"
)
