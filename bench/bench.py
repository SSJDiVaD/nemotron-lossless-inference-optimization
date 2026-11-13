#!/usr/bin/env python3
"""Endpoint-agnostic streaming benchmark client (vLLM, SGLang, Fireworks: any OpenAI-compatible /chat/completions).

Trace format (JSONL, one session per line):
  {"id": str, "messages": [...initial messages...], "turns": [{"user": str, "canned_reply": str|null}, ...]}
Turns within a session run sequentially (like a real agent); `--concurrency` sessions run at once.
After each turn the *canned* reply is appended to the history (not the model's real output), so every
endpoint sees byte-identical prompts. Single-turn traces (LiveCodeBench) are just sessions with one turn.

Cache fairness: by default a per-run nonce is prepended to the first message. Sessions in the same run
still share their prefix with each other (that's the point), but nothing is reused from a PREVIOUS run,
on any endpoint, including ones we can't flush (Fireworks).

Metrics per request: TTFT (first content OR reasoning token), e2e, output tokens (from usage when the
server reports it), TPOT, per-user decode tok/s, cached prompt tokens (if the server reports them).
"""
import argparse, asyncio, json, os, statistics, sys, time, uuid
import httpx


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


async def one_request(client, url, headers, body):
    rec = {"ok": False}
    t0 = time.perf_counter()
    ttft, n_chunks, text, reasoning, usage = None, 0, [], [], None
    try:
        async with client.stream("POST", url, headers=headers, json=body) as r:
            if r.status_code != 200:
                rec["error"] = f"HTTP {r.status_code}: {(await r.aread())[:300]!r}"
                return rec
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        n_chunks += 1
                        if d.get("content"):
                            text.append(d["content"])
                        r_piece = d.get("reasoning_content") or d.get("reasoning")
                        if r_piece:
                            reasoning.append(r_piece)
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        rec["error"] = repr(e)[:300]
        return rec
    e2e = time.perf_counter() - t0
    n_out = (usage or {}).get("completion_tokens") or n_chunks
    rec.update(ok=ttft is not None, ttft=ttft, e2e=e2e, n_out=n_out,
               prompt_tokens=(usage or {}).get("prompt_tokens"),
               cached_tokens=((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
               text="".join(text), reasoning="".join(reasoning))
    if ttft is not None and n_out > 1:
        rec["tpot"] = (e2e - ttft) / (n_out - 1)
        rec["decode_tps"] = (n_out - 1) / (e2e - ttft) if e2e > ttft else None
    if ttft is None:
        rec["error"] = "no tokens received"
    return rec


PARTIAL = None  # file handle: every finished request is appended immediately, so a cut-off run is still usable


async def run_session(sess, a, client, url, headers, nonce, sem, records):
    async with sem:
        msgs = [dict(m) for m in sess["messages"]]
        if a.nonce and msgs:
            msgs[0]["content"] = f"[run {nonce}]\n" + msgs[0]["content"]
        for ti, turn in enumerate(sess["turns"][: a.max_turns]):
            msgs.append({"role": "user", "content": turn["user"]})
            body = {"model": a.model, "messages": msgs, "max_tokens": a.max_tokens, "temperature": a.temperature,
                    "stream": True, "stream_options": {"include_usage": True}}
            if a.top_p is not None:
                body["top_p"] = a.top_p
            if a.seed is not None:
                body["seed"] = a.seed
            body.update(a.extra_body)
            rec = await one_request(client, url, headers, body)
            rec.update(session=sess["id"], turn=ti, t_start=time.time())
            text = rec.get("text", "")
            if not a.save_outputs:
                rec.pop("text", None)
                rec.pop("reasoning", None)
            records.append(rec)
            if PARTIAL:
                PARTIAL.write(json.dumps(rec) + "\n"); PARTIAL.flush()
            if not rec["ok"]:
                print(f"  ! {sess['id']} turn {ti}: {rec.get('error')}", file=sys.stderr)
                return
            msgs.append({"role": "assistant", "content": turn["canned_reply"] or text})


def summarize(records, wall, label, trace, conc):
    ok = [r for r in records if r["ok"]]
    def col(k): return [r[k] for r in ok if r.get(k) is not None]
    ttft, tpot, dtps = col("ttft"), col("tpot"), col("decode_tps")
    first = [r["ttft"] for r in ok if r["turn"] == 0]
    later = [r["ttft"] for r in ok if r["turn"] > 0]
    pt, ct = sum(col("prompt_tokens")), sum(col("cached_tokens"))
    return {
        "label": label, "trace": trace, "concurrency": conc, "n_ok": len(ok), "n_err": len(records) - len(ok),
        "wall_s": round(wall, 2),
        "ttft_p50": pct(ttft, 50), "ttft_p90": pct(ttft, 90), "ttft_p99": pct(ttft, 99),
        "ttft_first_turn_p50": pct(first, 50), "ttft_later_turns_p50": pct(later, 50),
        "tpot_p50": pct(tpot, 50), "tpot_p99": pct(tpot, 99),
        "decode_tps_per_user_p50": pct(dtps, 50),
        "output_tps_aggregate": sum(col("n_out")) / wall if wall else None,
        "mean_prompt_tokens": statistics.mean(col("prompt_tokens")) if col("prompt_tokens") else None,
        "mean_output_tokens": statistics.mean(col("n_out")) if ok else None,
        "cached_prompt_frac": (ct / pt) if pt and col("cached_tokens") else None,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g. http://localhost:8000/v1 or https://api.fireworks.ai/inference/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default=None, help="name of env var holding the API key (e.g. FIREWORKS_API_KEY)")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--max-sessions", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--extra-body", type=json.loads, default={}, help='JSON merged into every request, e.g. \'{"chat_template_kwargs":{"enable_thinking":false}}\'')
    ap.add_argument("--no-nonce", dest="nonce", action="store_false", help="allow cache reuse across runs")
    ap.add_argument("--warmup", type=int, default=2, help="tiny requests before timing (not part of the trace)")
    ap.add_argument("--save-outputs", action="store_true", help="keep generated text (needed for lcb_grade.py)")
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()

    sessions = [json.loads(l) for l in open(a.trace)][: a.max_sessions]
    trace_name = os.path.splitext(os.path.basename(a.trace))[0]
    url = a.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ[a.api_key_env]}"} if a.api_key_env else {}
    out_dir = os.path.join(a.out_dir, a.label)
    os.makedirs(out_dir, exist_ok=True)

    limits = httpx.Limits(max_connections=max(a.concurrency) + 4, max_keepalive_connections=max(a.concurrency) + 4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(a.timeout, connect=30), limits=limits) as client:
        for _ in range(a.warmup):
            await one_request(client, url, headers, {"model": a.model, "max_tokens": 8, "stream": True,
                                                     "messages": [{"role": "user", "content": "Say hi."}], **a.extra_body})
        for conc in a.concurrency:
            nonce = uuid.uuid4().hex[:12]
            records, sem = [], asyncio.Semaphore(conc)
            global PARTIAL
            PARTIAL = open(os.path.join(out_dir, f"{trace_name}_c{conc}.partial.jsonl"), "w")
            t0 = time.perf_counter()
            await asyncio.gather(*(run_session(s, a, client, url, headers, nonce, sem, records) for s in sessions))
            wall = time.perf_counter() - t0
            with open(os.path.join(out_dir, f"{trace_name}_c{conc}.jsonl"), "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            s = summarize(records, wall, a.label, trace_name, conc)
            s.update(base_url=a.base_url, model=a.model, max_tokens=a.max_tokens, temperature=a.temperature,
                     extra_body=a.extra_body, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            with open(os.path.join(out_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(s) + "\n")
            fmt = lambda v: "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
            print(f"[{a.label}] {trace_name} c={conc}: ok={s['n_ok']} err={s['n_err']} "
                  f"TTFT p50={fmt(s['ttft_p50'])}s p99={fmt(s['ttft_p99'])}s "
                  f"(turn0 {fmt(s['ttft_first_turn_p50'])}s, later {fmt(s['ttft_later_turns_p50'])}s) "
                  f"TPOT p50={fmt(s['tpot_p50'])}s user={fmt(s['decode_tps_per_user_p50'])} tok/s "
                  f"agg={fmt(s['output_tps_aggregate'])} tok/s cached={fmt(s['cached_prompt_frac'])}")


if __name__ == "__main__":
    asyncio.run(main())
