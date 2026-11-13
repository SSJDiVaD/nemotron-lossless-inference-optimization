# Deploy: Nemotron 3.5 Super VL (NVFP4) on a single Hopper GPU — lossless optimized configs

Pick the script for your hardware and traffic. All configs use the published NVFP4 checkpoint unchanged (no extra
quantization; the FP8 KV cache is the checkpoint's own setting). Serves an OpenAI-compatible API on `:8000`,
model name `nemotron`, reasoning on by default (`chat_template_kwargs: {"enable_thinking": false}` to disable).

| Script | GPU | Traffic | What's on | Measured vs. baseline (same GPU) |
|---|---|---|---|---|
| `serve_h100_interactive.sh` | H100 80GB | 1 user (coding agent) | prefix caching + MTP k=3 | TTFT **12×** faster, decode **2.3×** |
| `serve_h100_multiuser.sh` | H100 80GB | 8–32 users | prefix caching | TTFT **5–7×** faster, throughput **2.1–2.5×** |
| `serve_gh200_best.sh` | GH200 96GB | 1–8+ users | prefix caching + MTP k=3 + async + 8k prefill chunks / 64 seqs | TTFT **~10–12×**, throughput **3.6×** at 8 users |

```bash
./setup.sh                      # once: CUDA forward-compat if needed, vLLM 0.33.0, FlashInfer kernels, weights
./serve_h100_multiuser.sh       # or one of the others
```

**Why two H100 configs?** On an 80 GB H100 the model's weights leave ~6 GiB for KV cache. MTP's draft heads take
~1.7 GiB of that, so with many sessions the cached prefixes get evicted and prefix caching stops paying off
(c=32 TTFT: 1.6 s without MTP vs 21 s with it). With one user there is room for both. On the GH200 (19 GiB of KV
cache) the two optimizations reinforce each other at every concurrency tested.

**Deliberately not used (not lossless):** `--mamba-ssm-cache-dtype float16` + stochastic rounding (model card),
`VLLM_MARLIN_INPUT_DTYPE=int8|fp8`, `VLLM_HUMMING_USE_F16_ACCUM`, any further KV/weight quantization.

**Tried, not better than vLLM's defaults:** `--moe-backend marlin` (−15% throughput under load),
`VLLM_HUMMING_MOE_GEMM_TYPE=grouped` (−12% single-user decode), `-O3` (within noise).

Full methodology and numbers: see `../REPORT.md`.
