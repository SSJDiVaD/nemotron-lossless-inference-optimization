# Lane 4 — n-gram / prompt-lookup speculation (no extra weights; diffs copy text from the prompt).
# Speculation on hybrid Mamba models needs state rollback; n-gram may be unsupported for this architecture.
# If both configs fail to start, this lane falls through to the async-scheduling check at the end.  VERIFY
ng() { echo "--disable-prefix-caching --speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":$1,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}'"; }
CONFIGS=(
  "ngram_k4|vllm||$(ng 4)"
  "ngram_k8|vllm||$(ng 8)"
  "async_sched|vllm||--disable-prefix-caching --async-scheduling"
)
