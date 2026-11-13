# Lane 7 — combination + INTERACTION testing. Add one lever at a time on top of the previous winner, so an
# interference shows up as a regression at a specific step instead of a muddled final number.
# EDIT as lane winners come in (from ~1:15), then re-run:  KEEP_LAST=1 ./sweep.sh 7 8007 configs/lane7.sh
# Known interactions to watch:
#  - prefix caching + MTP on a hybrid Mamba model: both manage Mamba state -> may be unsupported (combo_pc_mtp fails to start)
#  - MTP at high concurrency: compare c=8 vs lane 3; spec can lower aggregate tok/s when the GPU is busy
#  - chunked-prefill budget: optimum shifts once prefix caching shrinks prefills -> retune here, not in lane 1
#  - async scheduling + speculation: historically incompatible in vLLM
PC="--enable-prefix-caching"
SPEC="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
CONFIGS=(
  "combo_pc_mtp|vllm||$PC $SPEC"
  "combo_pc_mtp_async|vllm||$PC $SPEC --async-scheduling"
  "combo_pc_mtp_async_mbt8k|vllm||$PC $SPEC --async-scheduling --max-num-batched-tokens 8192"
)
