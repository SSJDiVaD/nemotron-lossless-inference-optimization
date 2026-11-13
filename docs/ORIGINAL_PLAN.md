> **Outcome note (added after the run).** This is the plan as approved *before* the run. Reality diverged:
> Lambda had no 8×B200 (or any multi-GPU) capacity, so the work ran on single GPUs (1× H100 SXM, 1× GH200,
> 2× H100 PCIe) with per-box baselines; Fireworks only offered this model on dedicated 4× B200, so the provider
> reference became Together via OpenRouter; the $200 budget covered GPUs and provider API together (~$25–30 spent).
> See [`../NOTES.md`](../NOTES.md) for every deviation and [`../REPORT.md`](../REPORT.md) for results.

# Plan: Beat Fireworks on lossless coding inference, in 4 hours (Lambda ≤$200 + Fireworks)

**Decided:** Nemotron 3.5 Super VL, **NVFP4 checkpoint**. Lambda budget **$200 hard cap**; Fireworks costs are separate.

## Context
Interview challenge with **4 hours of wall-clock time**:
- Serve an open model from Hugging Face. The choices are DeepSeek V4.1 Flash or NVIDIA Nemotron 3.5 Super VL.
- Optimize TTFT and TPS for coding workloads.
- Stay **lossless**: no quantization beyond the checkpoints published on HF.
- Run on Lambda, with a budget of up to $300.
- Show the results head-to-head against Fireworks.

The budget allows **parallel lanes**: several independent GPU servers that each test a different optimization at the same time. The winners are combined at the end.

## Verified facts (web-checked 2026-11-13)
| | DeepSeek V4.1 Flash (`-1017` = current) | Nemotron 3.5 Super VL 120B-A12B |
|---|---|---|
| Size | 552B backbone + 196B Engram; 8B active/prompt token, 16B/output token | 120B MoE, 12B active, hybrid Mamba-Transformer |
| HF weights | ~511 GB, MXFP4 experts + MXFP8 rest (the only release) | BF16 (~240 GB) **and NVFP4 (~65 GB)** |
| Minimum GPUs | 8×H200 or 2–8×B200 with Engram offloaded to CPU | **1×B200** (NVFP4), or 1×H200 / 2×H100 |
| Speculative decoding | **No MTP** (dropped in V4.1); built-in "DSpark" draft head with block rejection sampling | **Built-in MTP heads** (`--speculative-config.method mtp`) |
| Prefix caching | Supported | Mamba layers need `--mamba-cache-mode align`, which is **experimental**; hits are block-aligned |
| Fireworks | **Serverless**, $0.30 in / $0.006 cached / $1.20 out per 1M tokens | **On-demand only** (dedicated GPUs; B200 at $13/hr, billed per second) |
| Lambda | 8×B200 ≈ $6.69/GPU-hr ≈ **$54/hr** | 1×B200 ≈ $6.99/hr; 8×B200 node ≈ $54/hr |

Gotchas:
- vLLM's DeepSeek V4.1 recipe defaults to `--kv-cache-dtype fp8`, which counts as extra quantization under the rules. Run the baseline with a BF16 KV cache, and ask the interviewer whether FP8 KV is allowed.
- Lambda's on-demand 2× and 4× B200 shapes are unconfirmed. 8×B200 is confirmed; 1×B200 was announced but may be out of stock.

## Model choice → **Nemotron 3.5 Super VL, NVFP4 checkpoint, 1 B200 per lane**
- **Fast iteration:** the checkpoint is 65 GB and fits on one GPU, so restarts take minutes. DeepSeek is 511 GB and its first boot can take a long time.
- **Parallel lanes fit the budget:** DeepSeek uses a whole 8-GPU node per server, so one server for 4 hours would eat the budget and leave no room for parallel lanes.
- **Clean comparison against Fireworks:** deploy the same NVFP4 checkpoint on a Fireworks on-demand 1×B200. Same model, same precision and same GPU means any difference comes from the serving stack. That is a stronger interview story than comparing against Fireworks' multi-tenant serverless endpoint.
- **Built-in MTP heads:** this is the largest lossless TPS lever, and it works out of the box.
- **Risk:** prefix caching on the Mamba layers is experimental, so validate it in the first hour. Fallback: DeepSeek V4.1 Flash on one 8×B200 node against Fireworks serverless, with no parallel lanes.
- **Check with the interviewer:** NVFP4 is a published HF checkpoint, so it meets the stated rule. If they want "highest published precision", use BF16 on 2×B200 per lane (halves the number of lanes).

## Parallel-lane setup (the "spend a bit more" answer)
**Hardware:** one **8×B200 node** (~$54/hr). Each GPU is pinned to its own vLLM server with `CUDA_VISIBLE_DEVICES=i` and its own port. This gives 8 independent single-GPU lanes from one download and one setup on local NVMe. If 1×B200 instances are in stock, 4 of them work too, but that means 4 separate setups.

| Lane (GPU) | Purpose |
|---|---|
| 0 | **Baseline**: stock vLLM. Kept running all session as the reference, and used for the greedy-equivalence check. |
| 1 | **Prefix caching** (`align` mode) + chunked-prefill budget sweep (2k / 8k / 16k) |
| 2 | Prefix caching + **CPU KV offload** (LMCache / vLLM offloading) at different session counts |
| 3 | **MTP**, `num_speculative_tokens` 1 / 3 / 5 |
| 4 | MTP + **n-gram / prompt-lookup** speculation (if they can be combined; otherwise n-gram alone) |
| 5 | **SGLang** with the same model (hybrid-model radix cache + MTP), to compare engines |
| 6 | Fused-MoE / FlashInfer **autotune**, then the tuned kernels with MTP |
| 7 | **Combined config**: merges lane winners as they arrive. Runs the final bench and LiveCodeBench. |

Rigor:
- At 0:30, run the same short baseline on **all 8 lanes**. That measures the noise floor between lanes, so a 5% gain can be told apart from noise.
- Lanes share the host CPU and RAM, so lane 2's offload traffic could disturb the others. Run it at a quiet moment, or note that it may be affected.

## Benchmarks (unchanged rationale)
- **LiveCodeBench** subset of ~60 recent problems, `max_tokens` ≈ 8k: the accuracy (pass@1) and decode TPS workload.
  - Not SWE-bench: needs an agent harness and Docker environments, takes too long, its pass rate is noisy, and tool execution hides the inference speedups.
  - Not HumanEval/MBPP: saturated, likely contaminated, and its prompts and outputs are too short to stress the engine.
- **Synthetic agentic trace** for TTFT: a ~20k-token shared prefix with multi-turn growth and diff-style outputs. Shared prefixes must be ≥ several Mamba blocks long, so align-mode prefix caching actually hits.
- Run with `vllm bench serve` against any OpenAI-compatible base URL, at concurrency 1 and 8. Report TTFT p50/p99, TPOT, per-user and aggregate tok/s, and pass@1.

## 4-hour timeline
**Before the clock starts:**
- Set up Lambda: account, SSH key, payment card.
- Set up Fireworks: account and API key. Read their on-demand deployment docs.
- Pre-write the scripts:
  - `lane.sh <gpu> <port> <flags>`
  - `bench.sh <base_url>`
  - the agentic-trace generator
  - the LiveCodeBench loader
- Test the scripts against any small model.

| Time | Work |
|---|---|
| 0:00–0:20 | Launch the 8×B200 node and download NVFP4 (~65 GB). **Create the Fireworks on-demand B200 deployment** of the same checkpoint (it scales to zero after an hour idle). Smoke-test one lane. |
| 0:20–0:40 | Start lanes 0–7 with their configs. Run the noise-floor baseline on all lanes, plus Fireworks. |
| 0:40–2:00 | Each lane runs its sweep (scripted, unattended). Lane 7 merges winners as soon as they appear. |
| 2:00–2:40 | Final combined config on 2–3 lanes (repeats run in parallel). Run the full bench vs lane 0 vs Fireworks dedicated (+ Fireworks serverless if relevant). |
| 2:40–3:10 | LiveCodeBench pass@1: final config vs Fireworks, run in parallel. Greedy-equivalence check of the final config vs lane 0 (~50 prompts). Spread the LiveCodeBench problems across lanes 3–7 to finish faster. |
| 3:10–3:15 | `rsync` all results and logs to your laptop. **Terminate the Lambda node and delete the Fireworks deployment.** |
| 3:15–4:00 | Write-up (offline): lane ablation table, charts vs Fireworks, cost. |

## Cost plan (Lambda $200 cap; Fireworks separate)
- **Lambda 8×B200 node:** ~$54–56/hr (check the live rate at launch). Billing runs from launch to **termination**; stopping servers does not stop billing.
- **The node runs only 0:00 → 3:15** (3.25 h ≈ **$175–182**), leaving about $20 of buffer. The write-up (3:15–4:00) needs no GPU, so copy results off the node with `rsync` first, then terminate.
- **Hard stop: terminate by 3:20 no matter what.** Set a phone timer. Lambda does not auto-stop instances.
- **Pre-clock setup costs $0 of GPU time:**
  - Develop the scripts locally.
  - Optionally pre-download the weights (~65 GB) into a Lambda persistent filesystem in the launch region. Storage cost is negligible. This saves ~10 minutes of node time, but it pins you to that region's B200 availability.
- **Fallback if 8×B200 is unavailable:** 4× 1×B200 instances (~$28/hr, ~$100 total) give 4 lanes. Drop lanes 4, 5 and 6, and merge lane 2 into lane 1.
- **Fireworks (outside the cap):** on-demand B200 ~2.5 h × $13 ≈ $33, plus about $10 of tokens.

## Optimizations, ranked by expected gain (each one has its own lane)
1. **Prefix caching:** TTFT 5–20× on the agentic trace, if align mode works.
2. **MTP speculation:** per-user TPS 1.5–2.5×.
3. **N-gram speculation:** more TPS on edit/diff outputs.
4. **SGLang engine:** 1–1.5×, depending on hybrid-model support.
5. **Chunked-prefill budget:** a p99 TTFT/ITL trade-off.
6. **MoE kernel autotune:** 10–30% on the MoE layers.
7. **CPU KV offload:** helps TTFT once the sessions exceed the HBM cache.

**How the levers interact (lane 7 tests this by adding one lever at a time):**
- **Mostly orthogonal:** TTFT levers (prefix caching, offload, chunked prefill) act on prefill. TPS levers (MTP, n-gram, kernels, scheduling) act on decode.
- **Substitutes:** MTP vs n-gram (one draft method at a time), and vLLM vs SGLang.
- **Interference or compatibility risks:**
  - Prefix caching + MTP both manage Mamba state on this hybrid model, so the combination may be unsupported.
  - MTP can lower aggregate throughput at high concurrency.
  - The best chunked-prefill size shifts once prefix caching is on.
  - Async scheduling + speculation was historically incompatible in vLLM.
- **End-to-end gains don't multiply:** turn latency ≈ TTFT + output tokens × TPOT (Amdahl's law).

**Excluded because they would break losslessness:**
- FP8/FP4 KV cache
- re-quantizing BF16 → NVFP4 yourself (the NVFP4 checkpoint must be the published one)
- pruning
- lossy speculation

## Fair comparison vs Fireworks
- Same checkpoint, same GPU type (dedicated deployment).
- Report TTFT raw and adjusted for network round-trip time (RTT).
- Our win is per-user latency at low concurrency. Fireworks is tuned for throughput.

## Verification
- Lane noise floor measured first. Final numbers are the median of the parallel repeats.
- Greedy outputs should match lane 0 apart from documented floating-point drift. LiveCodeBench pass@1 should be within noise of Fireworks and the model card.
- Spend: check the Lambda dashboard at 2:00. Expect ~$110; if it is above $120, cut the 2:00–2:40 block short. Check again after terminating.
