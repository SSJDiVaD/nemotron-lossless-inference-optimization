#!/usr/bin/env python3
"""Collect results/*/summary.jsonl (+ lcb_grade.json, greedy_equiv*.json) into tables and charts.

  python summarize.py --results results                 # writes results/REPORT.md (+ PNG charts if matplotlib)
  python summarize.py --results results --brief         # compact live view (used by the tmux watch window)
  python summarize.py --results results --only noise_g  # noise floor across the 8 identical lanes
"""
import argparse, glob, json, os, statistics

def baseline_for(label):
    """Speedups are only ever vs the SAME hardware's baseline: h100_* vs h100_baseline_naive, gh200_* vs gh200_..."""
    pre = label.split("_")[0]
    return f"{pre}_baseline_naive" if pre in ("h100", "gh200", "b200", "h200") else "baseline_naive"
COLS = [("ttft_p50", "TTFT p50 s"), ("ttft_p99", "TTFT p99 s"), ("ttft_later_turns_p50", "TTFT later-turn p50 s"),
        ("tpot_p50", "TPOT p50 s"), ("decode_tps_per_user_p50", "user tok/s"), ("output_tps_aggregate", "agg tok/s"),
        ("cached_prompt_frac", "cached frac")]


def load(results, only):
    rows = {}
    for path in glob.glob(os.path.join(results, "*", "summary.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if only and not r["label"].startswith(only):
                continue
            rows[(r["label"], r["trace"], r["concurrency"])] = r  # last run wins
    return rows


def f(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def tables(rows, brief):
    out = []
    groups = sorted({(t, c) for _, t, c in rows})
    for trace, conc in groups:
        sub = sorted((r for (l, t, c), r in rows.items() if t == trace and c == conc), key=lambda r: r["label"])
        out.append(f"\n### {trace} @ concurrency {conc}\n")
        hdr = ["label"] + [h for _, h in COLS] + ["TTFT x", "user tok/s x"]
        out.append("| " + " | ".join(hdr) + " |")
        out.append("|" + "---|" * len(hdr))
        for r in sub:
            cells = [r["label"]] + [f(r.get(k)) for k, _ in COLS]
            base = rows.get((baseline_for(r["label"]), trace, conc))
            if not base:
                cells += ["-", "-"]
            else:
                tt = base["ttft_p50"] / r["ttft_p50"] if r.get("ttft_p50") and base.get("ttft_p50") else None
                tp = (r["decode_tps_per_user_p50"] / base["decode_tps_per_user_p50"]
                      if r.get("decode_tps_per_user_p50") and base.get("decode_tps_per_user_p50") else None)
                cells += [f(tt, 2), f(tp, 2)]
            out.append("| " + " | ".join(cells) + " |")
            if brief and len(out) > 200:
                break
    return "\n".join(out)


def noise(rows):
    out = ["\n### Noise floor (identical config on every lane)\n", "| trace | conc | metric | min | max | spread % |",
           "|---|---|---|---|---|---|"]
    for trace, conc in sorted({(t, c) for _, t, c in rows}):
        sub = [r for (l, t, c), r in rows.items() if t == trace and c == conc]
        for k, h in COLS[:6]:
            vals = [r[k] for r in sub if r.get(k) is not None]
            if len(vals) > 1:
                m = statistics.mean(vals)
                out.append(f"| {trace} | {conc} | {h} | {min(vals):.3f} | {max(vals):.3f} | {100 * (max(vals) - min(vals)) / m:.1f} |")
    return "\n".join(out)


def extras(results):
    out = []
    grades = sorted(glob.glob(os.path.join(results, "*", "lcb_grade.json")))
    if grades:
        out += ["\n### LiveCodeBench pass@1 (same grader for every endpoint)\n", "| run | pass@1 | passed/n | by difficulty |",
                "|---|---|---|---|"]
        for g in grades:
            d = json.load(open(g))
            out.append(f"| {os.path.basename(os.path.dirname(g))} | {d['pass@1']:.3f} | {d['passed']}/{d['n']} | {d['by_difficulty']} |")
    for g in sorted(glob.glob(os.path.join(results, "greedy_*.json"))):
        d = json.load(open(g))
        out.append(f"\n**Greedy equivalence** `{os.path.basename(g)}`: {d['identical']}/{d['n']} identical "
                   f"({100 * d['identical_frac']:.0f}%), median divergence at {f(d['median_divergence_position_frac'], 2)} of output")
    return "\n".join(out)


def charts(rows, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made = []
    for metric, title in [("ttft_p50", "TTFT p50 (s, lower is better)"),
                          ("decode_tps_per_user_p50", "Per-user decode tok/s (higher is better)")]:
        for trace, conc in sorted({(t, c) for _, t, c in rows}):
            sub = sorted(((l, r[metric]) for (l, t, c), r in rows.items()
                          if t == trace and c == conc and r.get(metric) is not None), key=lambda x: x[1])
            if len(sub) < 2:
                continue
            fig, ax = plt.subplots(figsize=(8, 0.35 * len(sub) + 1.2))
            ax.barh([l for l, _ in sub], [v for _, v in sub],
                    color=["#d1495b" if "fireworks" in l else "#00798c" if l == baseline_for(l) else "#edae49" for l, _ in sub])
            ax.set_title(f"{title} - {trace} @ c={conc}", fontsize=10)
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            p = os.path.join(results, f"chart_{metric}_{trace}_c{conc}.png")
            fig.savefig(p, dpi=130)
            plt.close(fig)
            made.append(os.path.basename(p))
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--only", default=None, help="label prefix filter")
    ap.add_argument("--brief", action="store_true")
    a = ap.parse_args()
    rows = load(a.results, a.only)
    if not rows:
        print("no results yet"); return
    body = noise(rows) if a.only == "noise_g" else tables(rows, a.brief)
    if a.brief or a.only:
        print(body); return
    imgs = charts(rows, a.results)
    report = "# Benchmark report\n" + body + "\n" + extras(a.results) + "\n" + "".join(f"\n![]({i})\n" for i in imgs)
    open(os.path.join(a.results, "REPORT.md"), "w").write(report)
    print(report)


if __name__ == "__main__":
    main()
