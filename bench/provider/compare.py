#!/usr/bin/env python3
"""Build the provider-vs-ours comparison (markdown) from a provider run directory and our committed results.

  compare.py --provider-results results/provider_fireworks_<stamp> --ours results > COMPARISON.md

Ours (1x H100 80GB, vLLM 0.33, same traces/client):
  baseline        results/h100/h100_full_baseline      (no prefix caching, no MTP)
  multi-user best results/h100/h100_full_pc            (prefix caching)
  interactive     results/h100/h100_full_final         (prefix caching + MTP k=3)
  LCB 32k         results/h100pcie2/h100pcie_base_lcb32k, results/h100pcie/h100pcie_final_lcb32k
"""
import argparse, glob, json, os, re, statistics


def summaries(d):
    rows = {}
    for p in glob.glob(os.path.join(d, "**", "summary.jsonl"), recursive=True):
        for r in map(json.loads, open(p)):
            rows[(r["trace"], r["concurrency"])] = r
    return rows


def label_dirs(root, suffix):
    return sorted(glob.glob(os.path.join(root, f"*{suffix}")))


def rtt_seconds(run):
    p = os.path.join(run, "rtt.txt")
    if not os.path.exists(p):
        return None
    vals = [float(m) for m in re.findall(r"connect=([0-9.]+)", open(p).read())]
    return statistics.median(vals) if vals else None


def fmt(v, nd=2, suffix=""):
    return "-" if v is None else f"{v:.{nd}f}{suffix}"


def grade(p):
    if not p or not os.path.exists(p):
        return None
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider-results", required=True)
    ap.add_argument("--ours", default="results")
    a = ap.parse_args()
    run = a.provider_results
    ep = {}
    if os.path.exists(os.path.join(run, "endpoint.json")):
        try:
            ep = json.load(open(os.path.join(run, "endpoint.json")))
        except json.JSONDecodeError:
            pass
    st = ep.get("state", {})
    rtt = rtt_seconds(run)
    prov = {}
    for sub in ("short", "full", "smoke"):
        for d in label_dirs(run, f"_{sub}"):
            prov.update({(k[0], k[1], sub): v for k, v in summaries(d).items()})

    ours = {"baseline": summaries(os.path.join(a.ours, "h100", "h100_full_baseline")),
            "ours: prefix caching (multi-user)": summaries(os.path.join(a.ours, "h100", "h100_full_pc")),
            "ours: prefix caching + MTP k=3 (interactive)": summaries(os.path.join(a.ours, "h100", "h100_full_final"))}

    out = [f"### Dedicated-endpoint comparison: {st.get('provider', '?')} vs our single H100",
           "",
           f"- **Provider endpoint:** `{st.get('model', '?')}`; shape `{st.get('shape', '?')}`; one replica, autoscaling off.",
           f"- **Network round trip from the benchmark runner:** {fmt(rtt * 1000 if rtt else None, 0, ' ms')} (median TCP connect). "
           "*Adj.* columns subtract it from the provider's TTFT. Ours were measured with the client on the same host.",
           "- **Same traces, client, sampling and thinking settings as our runs.** Every provider response was checked to report a Nemotron model (`--expect-model`).",
           ""]
    served = {r.get("served_by") and json.dumps(r.get("served_by")) for r in prov.values()}
    if served - {None}:
        out.append(f"- **Models reported by provider responses:** {', '.join(sorted(s for s in served if s))}")
        out.append("")

    out += ["#### Full agent trace (32 sessions × 5 turns, ~22k-token prompts)", "",
            "| Concurrency | Config | TTFT p50 | TTFT p50 (adj.) | TTFT p99 | Per-user decode | Aggregate |",
            "|---|---|---|---|---|---|---|"]
    for c in (1, 8, 32):
        p = prov.get(("agentic", c, "full"))
        if p:
            adj = (p["ttft_p50"] - rtt) if (rtt and p.get("ttft_p50")) else None
            out.append(f"| {c} | **provider (dedicated)** | {fmt(p.get('ttft_p50'))} s | {fmt(adj)} s | {fmt(p.get('ttft_p99'))} s | "
                       f"{fmt(p.get('decode_tps_per_user_p50'), 0)} tok/s | {fmt(p.get('output_tps_aggregate'), 0)} tok/s |")
        for name, rows in ours.items():
            r = rows.get(("agentic", c))
            if r:
                out.append(f"| {c} | {name} | {fmt(r.get('ttft_p50'))} s | — | {fmt(r.get('ttft_p99'))} s | "
                           f"{fmt(r.get('decode_tps_per_user_p50'), 0)} tok/s | {fmt(r.get('output_tps_aggregate'), 0)} tok/s |")
    out += ["", "#### LiveCodeBench speed (full trace)", "",
            "| Concurrency | Config | TTFT p50 | Per-user decode | Aggregate |", "|---|---|---|---|---|"]
    for c in (1, 8):
        p = prov.get(("lcb", c, "full"))
        if p:
            out.append(f"| {c} | **provider (dedicated)** | {fmt(p.get('ttft_p50'))} s | {fmt(p.get('decode_tps_per_user_p50'), 0)} tok/s | "
                       f"{fmt(p.get('output_tps_aggregate'), 0)} tok/s |")
        for name, rows in ours.items():
            r = rows.get(("lcb", c))
            if r:
                out.append(f"| {c} | {name} | {fmt(r.get('ttft_p50'))} s | {fmt(r.get('decode_tps_per_user_p50'), 0)} tok/s | "
                           f"{fmt(r.get('output_tps_aggregate'), 0)} tok/s |")

    # accuracy at the 32k budget, with per-problem agreement
    g_prov = grade(next(iter(glob.glob(os.path.join(run, "*_lcb32k", "lcb_grade.json"))), None))
    g_base = grade(os.path.join(a.ours, "h100pcie2", "h100pcie_base_lcb32k", "lcb_grade.json"))
    g_fin = grade(os.path.join(a.ours, "h100pcie", "h100pcie_final_lcb32k", "lcb_grade.json"))
    if g_prov:
        out += ["", "#### LiveCodeBench accuracy, 32k reasoning budget (60 problems, temperature 1.0, 1 sample)", "",
                "| Config | pass@1 | Easy · medium · hard | No code block | Same pass/fail as our baseline | Same as our optimized |",
                "|---|---|---|---|---|---|"]
        def agree(x, y):
            if not (x and y):
                return "-"
            px, py = x["per_problem"], y["per_problem"]
            ids = set(px) & set(py)
            return f"{sum(px[i] == py[i] for i in ids)}/{len(ids)}"
        for name, g in (("**provider (dedicated)**", g_prov), ("ours: baseline", g_base), ("ours: prefix caching + MTP k=3", g_fin)):
            if g:
                bd = g.get("by_difficulty", {})
                out.append(f"| {name} | {100 * g['pass@1']:.1f}% | {bd.get('easy', '-')} · {bd.get('medium', '-')} · {bd.get('hard', '-')} | "
                           f"{g.get('no_code_block', '-')} | {agree(g, g_base)} | {agree(g, g_fin)} |")
    smoke = prov.get(("agentic_short", 1, "smoke"))
    if smoke and not prov.get(("agentic", 1, "full")):
        out += ["", f"*Smoke run only:* short trace, c=1, 2 sessions. TTFT p50 {fmt(smoke.get('ttft_p50'))} s, "
                f"per-user decode {fmt(smoke.get('decode_tps_per_user_p50'), 0)} tok/s, served by {smoke.get('served_by')}."]
    print("\n".join(out))


if __name__ == "__main__":
    main()
