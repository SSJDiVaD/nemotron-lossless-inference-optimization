# Lane 3 — MTP speculative decoding with the model's own heads (the #1 TPS lever). Isolated from prefix
# caching on purpose so the gain is attributable; lane 7 tests the combination.
# `method mtp` + num_speculative_tokens is from vLLM's Nemotron 3.5 Super VL blog (they used 5).
mtp() { echo "--disable-prefix-caching --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$1}'"; }
CONFIGS=(
  "mtp_k1|vllm||$(mtp 1)"
  "mtp_k3|vllm||$(mtp 3)"
  "mtp_k5|vllm||$(mtp 5)"
)
