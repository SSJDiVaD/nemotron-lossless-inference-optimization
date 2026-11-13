# Single-GPU mode (1x H100 SXM): all lanes collapsed into ONE sequential sweep, highest-value first.
# Each config costs ~8-12 min (startup + short bench), so ~8 configs fit in the 0:40-2:00 window.
# Run:  ./sweep.sh 0 8000 configs/single_gpu.sh short
# Flags follow the Nemotron 3.5 Super VL model card. Deliberately NOT used: --mamba-ssm-cache-dtype float16
# (+ stochastic rounding) — that lowers SSM-state precision, i.e. not lossless.
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
mtp() { echo "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$1}'"; }
CONFIGS=(
  "baseline_naive|vllm||--disable-prefix-caching"   # reference for every speedup
  "pc_align|vllm||$PC"                                # TTFT lever
  "mtp_k3|vllm||--disable-prefix-caching $(mtp 3)"  # TPS lever (model card: k=3 on H100)
  "combo_pc_mtp|vllm||$PC $(mtp 3)"                   # do they combine? (Mamba state interaction)
  "mtp_k2|vllm||--disable-prefix-caching $(mtp 2)"  # spec-depth sweep around the card's value
  "mtp_k5|vllm||--disable-prefix-caching $(mtp 5)"
  "combo_pc_mtp_async|vllm||$PC $(mtp 3) --async-scheduling"
  "baseline_naive_rep|vllm||--disable-prefix-caching" # repeat => single-GPU noise floor
)
