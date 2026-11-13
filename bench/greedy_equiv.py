#!/usr/bin/env python3
"""Lossless check: send the same prompts at temperature 0 to a reference and a candidate endpoint and
compare the generated text.

Expected result: most outputs identical. Divergences that start late and are rare come from batch-size-
dependent floating-point reduction order (different batch shapes => different kernel tilings), which also
happens between two runs of the *baseline* itself. To prove that, run this with --ref and --cand both set
to the baseline lane: that gives the noise floor to compare against.

Usage: python greedy_equiv.py --ref http://localhost:8000/v1 --cand http://localhost:8007/v1 \
           --model nemotron --trace traces/lcb.jsonl --n 50
"""
import argparse, asyncio, json, os
import httpx


async def gen(client, base, model, msgs, max_tokens, key, extra):
    h = {"Authorization": f"Bearer {key}"} if key else {}
    r = await client.post(base.rstrip("/") + "/chat/completions", headers=h,
                          json={"model": model, "messages": msgs, "max_tokens": max_tokens,
                                "temperature": 0.0, **extra})
    r.raise_for_status()
    m = r.json()["choices"][0]["message"]
    return (m.get("reasoning_content") or m.get("reasoning") or "") + "\n<<content>>\n" + (m.get("content") or "")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cand-model", default=None)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--extra-body", type=json.loads, default={})
    ap.add_argument("--out", default="results/greedy_equiv.json")
    a = ap.parse_args()

    prompts = []
    for line in open(a.trace):
        s = json.loads(line)
        prompts.append((s["id"], s["messages"] + [{"role": "user", "content": s["turns"][0]["user"]}]))
    prompts = prompts[: a.n]
    sem = asyncio.Semaphore(a.concurrency)

    async with httpx.AsyncClient(timeout=600) as client:
        async def both(pid, msgs):
            async with sem:
                r = await gen(client, a.ref, a.model, msgs, a.max_tokens, None, a.extra_body)
                c = await gen(client, a.cand, a.cand_model or a.model, msgs, a.max_tokens, None, a.extra_body)
                common = next((i for i, (x, y) in enumerate(zip(r, c)) if x != y), min(len(r), len(c)))
                return {"id": pid, "identical": r == c, "first_diff_char": None if r == c else common,
                        "ref_len": len(r), "cand_len": len(c)}
        res = await asyncio.gather(*(both(p, m) for p, m in prompts))

    same = sum(x["identical"] for x in res)
    diffs = [x["first_diff_char"] / max(1, x["ref_len"]) for x in res if not x["identical"]]
    summary = {"ref": a.ref, "cand": a.cand, "n": len(res), "identical": same, "identical_frac": same / len(res),
               "median_divergence_position_frac": sorted(diffs)[len(diffs) // 2] if diffs else None}
    print(json.dumps(summary))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({**summary, "per_prompt": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
