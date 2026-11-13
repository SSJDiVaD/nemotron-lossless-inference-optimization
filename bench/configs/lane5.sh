# Lane 5 — SGLang, the alternative engine (radix prefix cache is on by default).
# SGLang's speculative flags for Nemotron-H-style MTP are the least certain part of this kit.  VERIFY against
# SGLang's Nemotron docs; if the spec config fails, the lane still produces the plain-SGLang data point.
CONFIGS=(
  "sglang_default|sglang||"
  "sglang_mtp|sglang||--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
)
