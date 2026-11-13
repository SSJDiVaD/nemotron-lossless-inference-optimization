#!/usr/bin/env python3
"""Report charts: per-GPU ablations (short agent trace) and full-trace baseline vs best, from collected summaries."""
import json, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = sys.argv[1] if len(sys.argv) > 1 else "collected"; OUT = os.path.join(C, "charts"); os.makedirs(OUT, exist_ok=True)
def get(box, label, trace, conc, key):
    p = os.path.join(C, box, label, "summary.jsonl")
    if not os.path.exists(p): return None
    rows = [r for r in map(json.loads, open(p)) if r["trace"] == trace and r["concurrency"] == conc]
    return rows[-1].get(key) if rows else None
BLUE, ORANGE, GRAY = "#00798c", "#edae49", "#8d99ae"
def bars(ax, names, vals, title, unit, best_hi=True):
    vals = [v or 0 for v in vals]; base = vals[0] or 1
    cols = [GRAY] + [BLUE] * (len(vals) - 1)
    b = ax.barh(names[::-1], vals[::-1], color=cols[::-1])
    for rect, v in zip(b, vals[::-1]):
        x = (base / v if not best_hi else v / base) if v else 0
        ax.text(rect.get_width(), rect.get_y() + rect.get_height()/2, f" {v:.2f}{unit}  ({x:.1f}×)" if unit=="s" else f" {v:.0f}{unit}  ({x:.1f}×)", va="center", fontsize=8)
    ax.set_title(title, fontsize=10); ax.tick_params(labelsize=8); ax.margins(x=0.35)
# GH200 ablation, short agent trace
steps = [("baseline", "gh200_baseline_naive"), ("+ prefix caching", "gh200_pc"), ("+ MTP k=3 + async", "gh200_combo_pc_mtp_async"),
         ("+ 8k chunks / 64 seqs", "gh200_combo_bigbatch"), ("… with MTP k=5", "gh200_best_k5"), ("… with MTP k=6", "gh200_best_k6")]
fig, axs = plt.subplots(1, 3, figsize=(15, 3.2))
bars(axs[0], [s for s, _ in steps], [get("gh200", l, "agentic_short", 1, "ttft_p50") for _, l in steps], "GH200 · TTFT p50, 1 user (lower is better)", "s", best_hi=False)
bars(axs[1], [s for s, _ in steps], [get("gh200", l, "agentic_short", 1, "decode_tps_per_user_p50") for _, l in steps], "GH200 · decode tok/s per user, 1 user", " tok/s")
bars(axs[2], [s for s, _ in steps], [get("gh200", l, "agentic_short", 8, "output_tps_aggregate") for _, l in steps], "GH200 · aggregate tok/s, 8 users", " tok/s")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "gh200_ablation.png"), dpi=130); plt.close(fig)
# H100: interference — full trace at c=1/8/32
cfg = [("baseline", "h100_full_baseline"), ("prefix caching", "h100_full_pc"), ("prefix caching + MTP k=3", "h100_full_final")]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))
import numpy as np
x = np.arange(3); w = 0.26
for i, (name, lab) in enumerate(cfg):
    t = [get("h100", lab, "agentic", c, "ttft_p50") or 0 for c in (1, 8, 32)]
    a = [get("h100", lab, "agentic", c, "output_tps_aggregate") or 0 for c in (1, 8, 32)]
    col = [GRAY, BLUE, ORANGE][i]
    axs[0].bar(x + (i-1)*w, t, w, label=name, color=col); axs[1].bar(x + (i-1)*w, a, w, label=name, color=col)
for ax, t in zip(axs, ["H100 · TTFT p50 (s, lower is better)", "H100 · aggregate tok/s"]):
    ax.set_xticks(x, ["1 user", "8 users", "32 users"]); ax.set_title(t, fontsize=10); ax.tick_params(labelsize=8)
axs[0].set_yscale("log"); axs[1].legend(fontsize=8)
fig.suptitle("H100 80 GB, full agent trace: MTP helps 1 user but hurts under load (memory interference)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "h100_interference.png"), dpi=130); plt.close(fig)
print("charts:", os.listdir(OUT))
