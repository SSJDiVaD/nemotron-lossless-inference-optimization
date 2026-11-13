#!/usr/bin/env python3
"""Grade LiveCodeBench outputs saved by `bench.py --save-outputs` (pass@1, one sample per problem).

A compact re-implementation of LiveCodeBench's grading (stdin/stdout and LeetCode-style functional tests).
It is not the official lcb_runner, so absolute scores can differ slightly from published numbers; what
matters here is that EVERY endpoint (baseline, optimized, Fireworks) is graded by the same code.

Runs model-generated code in subprocesses with timeouts. Run it on the disposable GPU node, not your laptop.
Note: LiveCodeBench stores private tests as base64(zlib(pickle(json))), so loading them unpickles dataset data,
exactly like the official runner does.

Usage: python lcb_grade.py --tests traces/lcb.jsonl.tests --outputs results/<label>/lcb_c8.jsonl [--outputs ...]
"""
import argparse, base64, json, os, pickle, re, subprocess, sys, tempfile, zlib
from concurrent.futures import ThreadPoolExecutor

PRELUDE = """import sys, os, re, math, heapq, bisect, random, string, itertools, functools, collections, operator
from typing import *
from collections import *
from itertools import *
from functools import *
from heapq import *
from bisect import *
from math import *
sys.setrecursionlimit(1 << 25)
"""

FUNC_HARNESS = r"""
import json, sys
_tests = json.load(open(sys.argv[1]))
_sol = Solution()
_fn = getattr(_sol, sys.argv[2])
for _t in _tests:
    _args = [json.loads(x) for x in _t["input"].split("\n") if x.strip()]
    _got = _fn(*_args)
    _exp = json.loads(_t["output"])
    if isinstance(_got, tuple): _got = list(_got)
    if _got != _exp:
        print("FAIL", json.dumps(_got)[:200], json.dumps(_exp)[:200]); sys.exit(1)
print("PASS")
"""


def load_tests(row):
    pub = json.loads(row["public_test_cases"])
    try:
        priv = json.loads(row["private_test_cases"])
    except json.JSONDecodeError:
        priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(row["private_test_cases"].encode()))))
    return pub + priv


def extract_code(text):
    blocks = re.findall(r"```(?:python|py|Python3?)?\s*\n(.*?)```", text or "", re.S)
    return blocks[-1] if blocks else None


def outputs_match(got, exp):
    g, e = got.strip().split("\n"), exp.strip().split("\n")
    if len(g) != len(e):
        return False
    for gl, el in zip(g, e):
        if gl.strip() == el.strip():
            continue
        gt, et = gl.split(), el.split()
        if len(gt) != len(et):
            return False
        for x, y in zip(gt, et):
            if x == y:
                continue
            try:
                if abs(float(x) - float(y)) > 1e-6 * max(1.0, abs(float(y))):
                    return False
            except ValueError:
                return False
    return True


def grade_one(code, row, timeout):
    tests = load_tests(row)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "sol.py")
        if tests and tests[0]["testtype"] == "functional":
            func = json.loads(row["metadata"])["func_name"]
            tf = os.path.join(d, "tests.json")
            json.dump(tests, open(tf, "w"))
            open(src, "w").write(PRELUDE + code + "\n" + FUNC_HARNESS)
            try:
                r = subprocess.run([sys.executable, src, tf, func], capture_output=True, text=True,
                                   timeout=timeout * max(1, len(tests)) ** 0.5 * 2, cwd=d)
                return r.returncode == 0 and r.stdout.strip().endswith("PASS")
            except subprocess.TimeoutExpired:
                return False
        open(src, "w").write(PRELUDE + code)
        for t in tests:
            try:
                r = subprocess.run([sys.executable, src], input=t["input"], capture_output=True, text=True,
                                   timeout=timeout, cwd=d)
            except subprocess.TimeoutExpired:
                return False
            if r.returncode != 0 or not outputs_match(r.stdout, t["output"]):
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True)
    ap.add_argument("--outputs", required=True, nargs="+")
    ap.add_argument("--timeout", type=float, default=6.0, help="seconds per stdin test")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    rows = {r["question_id"]: r for r in map(json.loads, open(a.tests))}
    for path in a.outputs:
        outs = {}
        for rec in map(json.loads, open(path)):
            outs[rec["session"]] = rec.get("text")
        missing = [q for q in rows if q not in outs]

        def job(q):
            code = extract_code(outs.get(q))
            return q, (code is not None and grade_one(code, rows[q], a.timeout)), code is None

        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(job, rows))
        passed = sum(p for _, p, _ in res)
        no_code = sum(n for _, _, n in res)
        by_diff = {}
        for q, p, _ in res:
            d = rows[q]["difficulty"]
            by_diff.setdefault(d, [0, 0])
            by_diff[d][0] += p
            by_diff[d][1] += 1
        summary = {"outputs": path, "n": len(rows), "pass@1": passed / len(rows), "passed": passed,
                   "no_code_block": no_code, "missing": len(missing),
                   "by_difficulty": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_diff.items())}}
        print(json.dumps(summary))
        with open(os.path.join(os.path.dirname(path), "lcb_grade.json"), "w") as f:
            json.dump({**summary, "per_problem": {q: p for q, p, _ in res}}, f, indent=1)


if __name__ == "__main__":
    main()
