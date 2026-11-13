# Notes: plan vs. what actually happened

The original plan assumed one 8×B200 Lambda node split into 8 parallel single-GPU "lanes", benchmarked head-to-head
against a Fireworks dedicated deployment of the same checkpoint, within a 4-hour window and a $200 budget covering
all spend (GPU rental plus any inference-provider API usage). This file records every
deviation from that plan, why it happened, and how it affects the results.

## Constraints that changed

| Planned | Actual | Why | Effect on results |
|---|---|---|---|
| 8×B200 node (8 parallel lanes) | **1× H100 SXM**, later **+1× GH200** and **+1× H100 PCIe** | Lambda had no B200/H200/H100 multi-GPU capacity; only A10/A100 were listed at the start. Single GPUs were grabbed by a polling script (`bench/snap_gpus.py`) as they appeared | Lanes ran sequentially per box; each box has its **own baseline**, and speedups are only reported against the same GPU's baseline |
| DeepSeek V4.1 Flash | **Nemotron 3.5 Super VL, NVFP4** | DeepSeek V4.1 Flash is ~511 GB (needs 8× H200/B200) and has no MTP heads; Nemotron fits one GPU and ships MTP heads | — |
| Fireworks dedicated deployment as the comparison | **Together, via OpenRouter** (provider pinned, no fallbacks) | Fireworks offered this model only on dedicated 4× B200 (~$52/hr); not serverless | Provider hardware unknown (likely Blackwell), shared multi-tenant, includes OpenRouter routing overhead. Same NVFP4 precision. Used as a reference point, not an apples-to-apples hardware comparison |
| $200 total budget (Lambda + provider API) | **~$25–30 spent** (est.: ~$22 Lambda GPU-hours + a few dollars of OpenRouter tokens) | Single GPUs are cheap; time and GPU availability, not money, were the constraints | — |

## Setup problems hit (and fixed) — all now automated in `deploy/setup.sh` / `bench/setup_node.sh`

1. **FlashInfer kernels silently disabled** — Lambda Stack has no `nvcc`, so vLLM fell back to slower kernels.
   Fix: `vllm download-kernels` (needed `pip` inside the uv venv).
2. **HUMMING NVFP4 kernel failed to JIT** (`libnvrtc-builtins.so.13.0` not found) — fix: put the venv's
   `nvidia/cu13/lib` on `LD_LIBRARY_PATH`.
3. **Out of memory on 1× H100 80 GB** — 70 GB of weights + 8k-token prefill profiling. Fix: cap prefill chunks to
   2048 tokens and 32 concurrent sequences (scheduling limits only; lossless).
4. **MTP didn't fit on the H100** (KV cache −1.1 GiB). Fix: `--language-model-only` (skip the unused vision encoder)
   + 0.97 memory utilization. Side effect: KV cache grew **6×** (1.0 → 6.3 GiB), and c=8 TTFT improved 3.5× on its own.
   All H100 results were re-run with these settings (the earlier "v1" results are archived in `results/h100_v1_archive`).
5. **GH200: driver 570 (CUDA 12.8) vs CUDA 13 wheels** — fix: NVIDIA's `cuda-compat-13-0` forward-compatibility
   package (no driver upgrade, no reboot). Same on the H100 PCIe box.
6. **GH200 (aarch64): vLLM's video-input profiling triggers a gcc `-march` error** — fix: `--language-model-only`.
7. **vLLM 0.33 flag changes vs. the plan**: `--no-enable-prefix-caching` → `--disable-prefix-caching`;
   `--mamba-cache-mode` deprecated (no-op). Model-card flags for hybrid prefix caching used instead.
8. **Trace token counts**: the first synthetic agent traces were sized by a character heuristic and exceeded the 40k
   context; regenerated with the model's tokenizer (all prompts 21–22k tokens). All boxes used byte-identical traces.

## Bugs in my own tooling (caught and fixed during the run)

- **Greedy-equivalence check was initially vacuous**: with a 256-token cap the model spends every token reasoning, but
  the client saved only the final `content` (empty for ~90% of prompts). Fixed to save and compare reasoning + content;
  H100 captures were redone.
- **`env.sh` ended with a failing `[[ … ]] &&` test**, which made scripts under `set -e` exit silently (cost the
  accuracy box ~18 min).
- **Overwriting a running bash script** corrupts it (bash reads scripts incrementally) — one GH200 job had to be relaunched.

## Methodology decisions worth knowing

- **Lossless definition**: published checkpoint unchanged (NVFP4 experts, FP8 projections, FP8 KV cache with the
  checkpoint's calibrated scales). Excluded as lossy: FP16 Mamba SSM state (model card suggests it), FP8/INT8 MARLIN
  activations, FP16 accumulation.
- **Token-exact greedy matching is not a usable lossless test for this model**: the *same* baseline config diverges
  from itself across batch sizes (GH200: 1/40 exact matches, divergence ~8% into the output) because batch shape
  changes kernel reduction order and long reasoning has many near-ties. Losslessness is argued by construction
  (same weights/KV format; MTP uses rejection sampling, which preserves the output distribution; prefix caching
  reuses KV computed for the identical prefix) and verified empirically by task accuracy at equal budgets.
- **LiveCodeBench**: 60 most recent problems of the public v6 set (2025-03-08 … 2025-04-06; 30 hard / 18 medium / 12
  easy), one sample at temperature 1.0 / top-p 0.95 (model-card sampling), graded by a re-implementation of the LCB
  harness (same grader for every endpoint). Used for **relative** comparison, not to reproduce the model card's 87.4.
  An 8k reasoning budget truncates ~half of the answers; the 32k runs truncate ~a third.
- **Agent trace is synthetic** (not an official dataset): 15.5k-token prefix shared by all sessions + 6k per-session
  files + multi-turn diff edits, canned assistant turns for byte-identical prompts. Prefix-caching gains depend on the
  shared-prefix fraction, which is an assumption.
- **Noise floor**: repeated baselines on the same box agree within ~1.5% (H100) and ~0.3% (GH200).

## Planned but not done

- B200 run (native FP4): no capacity appeared on Lambda during the window.
- SGLang comparison, n-gram speculation, CPU KV offload lane (no spare GPUs; offload irrelevant once the KV cache fit).
- SWE-bench-style real-trajectory replay; vLLM batch-invariant mode for bit-exact verification.
