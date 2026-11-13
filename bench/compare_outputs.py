#!/usr/bin/env python3
"""Offline greedy-equivalence check for single-GPU boxes: compare temperature-0 outputs that bench.py saved
(--save-outputs) from two server configs run one after another on the SAME GPU.

  python compare_outputs.py results/h100_greedy_base_a/lcb_c8.jsonl results/h100_greedy_base_b/lcb_c8.jsonl --name base_vs_base
  python compare_outputs.py results/h100_greedy_base_a/lcb_c8.jsonl results/h100_greedy_final/lcb_c8.jsonl  --name base_vs_final

Run base-vs-base too: two runs of the identical baseline config give the floating-point drift floor (batch
composition changes kernel reduction order). Base-vs-final is lossless if it is no worse than that floor.
Writes results/greedy_<name>.json (picked up by summarize.py).
"""
import argparse, json, os


def load(path):
    # Compare the FULL generation: reasoning (thinking) + final content. With small max_tokens a reasoning model's
    # content is often empty, so comparing content alone would be vacuous.
    return {r["session"]: (r.get("reasoning") or "") + "\n<<content>>\n" + (r.get("text") or "")
            for r in map(json.loads, open(path)) if r.get("ok")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("cand")
    ap.add_argument("--name", required=True)
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    r, c = load(a.ref), load(a.cand)
    ids = sorted(set(r) & set(c))
    per, same, fracs = [], 0, []
    for i in ids:
        x, y = r[i], c[i]
        if x == y:
            same += 1; per.append({"id": i, "identical": True}); continue
        k = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        fracs.append(k / max(1, len(x)))
        per.append({"id": i, "identical": False, "first_diff_char": k, "ref_len": len(x), "cand_len": len(y)})
    fracs.sort()
    s = {"ref": a.ref, "cand": a.cand, "n": len(ids), "identical": same, "identical_frac": same / max(1, len(ids)),
         "median_divergence_position_frac": fracs[len(fracs) // 2] if fracs else None}
    print(json.dumps(s))
    json.dump({**s, "per_prompt": per}, open(os.path.join(a.results, f"greedy_{a.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
