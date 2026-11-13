# bench/provider: dedicated-endpoint comparison

This compares our single-GPU vLLM configs against a provider's **dedicated** endpoint for the same NVFP4 checkpoint: one replica, fixed hardware, autoscaling off. That replaces an earlier OpenRouter comparison, whose provider pin was silently dropped (see `../../NOTES.md`).

It runs unattended through GitHub Actions (`.github/workflows/provider-bench.yml`), so your computer can be off.

## Run it
1. **Add repo secrets** under Settings → Secrets and variables → Actions:
   - `FIREWORKS_API_KEY`
   - `FIREWORKS_ACCOUNT_ID`
   - `TOGETHER_API_KEY`
2. **Allow the workflow to open PRs:** Settings → Actions → General → "Allow GitHub Actions to create and approve pull requests".
3. **Run the workflow** from the Actions tab: **Provider bench → Run workflow**.
   - **`discover`** (free): lists each provider's hardware shapes for the model in the job summary. Pick a Hopper (H100, then H200) NVFP4 shape.
   - **`smoke`** (~15 min, a few dollars): set `provider` and `shape` (and `model` for Together). This validates endpoint create, benchmark and teardown.
   - **`full`**: short + full agent traces, LiveCodeBench speed, and LiveCodeBench accuracy at a 32k budget (hard cap 4 h). It opens a PR whose `REPORT.md` provider section is replaced by the new comparison table.

## Safety: layers against a forgotten paid GPU
1. **No shape, no endpoint.** `smoke` and `full` refuse to create anything without an explicit `shape` from `discover`.
2. **State written before the create call.** The endpoint's name is chosen and saved before the API call, so teardown works even if the create response is lost.
3. **Two teardown paths.** The run's shell `trap` deletes the endpoint on any exit. An independent `teardown` job (`if: always()`) deletes it again from the uploaded state, even if the benchmark failed, timed out or was cancelled. Deletes are retried with backoff and treat "already gone" as success.
4. **Hourly janitor** (`provider-janitor.yml`). It deletes any `provider-bench-*` endpoint older than 6 hours at either provider (a run is capped at ~5 hours), and only those: it never touches other endpoints.
5. **Provider-side scale-to-zero (Fireworks).** Max 1 replica, min 0, 30-minute idle window. If every path above fails, Fireworks releases the GPU after 30 idle minutes anyway.
6. **Prepaid credit with auto-recharge off.** This is the hard cap: the worst case can't cost more than the credit loaded.

The client also refuses to compare the wrong thing: every response must report a Nemotron model (`--expect-model`), otherwise the run fails with exit code 3. And `concurrency: provider-bench` means only one paid endpoint exists at a time.

## Files
| File | Purpose |
|---|---|
| `endpoint.py` | `discover / create / wait / describe / delete`: Fireworks via REST, Together via the `tg` CLI, plus `mock` |
| `run_provider_bench.sh` | create (or reuse) → wait → RTT → suites (same `bench_suite.sh`, traces, sampling) → compare → delete |
| `compare.py` | Markdown table: provider vs our H100 baseline / prefix caching / prefix caching + MTP, plus LiveCodeBench 32k agreement |
| `mock_server.py` | Local OpenAI-compatible mock for free end-to-end tests |

Local test with no spend: start `python3 bench/provider/mock_server.py 18765`, then run `PROVIDER=mock MODE=full RESULTS=/tmp/pb STATE_FILE=/tmp/pb.json bash bench/provider/run_provider_bench.sh`.

**Not yet verified against the live providers (done by `discover`/`smoke`):** the Together CLI's install name and JSON field names, and the Fireworks REST delete path. Both adapters log raw responses, so a mismatch shows up in the discover/smoke logs before any long, expensive run.
