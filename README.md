# Lossless inference optimization — Nemotron 3.5 Super VL (NVFP4) on single Hopper GPUs

**Start with [`REPORT.md`](REPORT.md).**

| Path | What |
|---|---|
| [`REPORT.md`](REPORT.md) | Results, methodology, losslessness argument, what worked / didn't / not tried |
| [`deploy/`](deploy/) | Setup + serve scripts for the best configs (H100 interactive, H100 multi-user, GH200) |
| [`bench/`](bench/) | The experiment harness (traces, client, sweeps, grading, GPU auto-provisioning) |
| [`results/`](results/) | Raw per-request results, per-config summaries, LCB grades, profiles, charts — per GPU box |
| [`env/`](env/) | Exact GPU / driver / package versions and model revision per box |
| [`NOTES.md`](NOTES.md) | Deviations from the plan, setup fixes, tooling bugs caught |
| [`docs/ORIGINAL_PLAN.md`](docs/ORIGINAL_PLAN.md) | The plan as written before the run |
