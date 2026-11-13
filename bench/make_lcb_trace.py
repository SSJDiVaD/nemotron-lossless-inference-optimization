#!/usr/bin/env python3
"""Build a LiveCodeBench (code generation) subset: the N most recent problems.

Writes two files:
  <out>          prompts in the bench.py trace format (single-turn sessions)
  <out>.tests    test cases + metadata for lcb_grade.py

Data source: HF dataset repo livecodebench/code_generation_lite (raw test*.jsonl files).
The prompt follows LiveCodeBench's generic chat-model format.
"""
import argparse, json
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "livecodebench/code_generation_lite"

SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) "
          "and will generate a correct Python program that matches the specification and passes all tests.")

FMT_STARTER = ("You will use the following starter code to write the solution to the problem and enclose "
               "your code within delimiters.\n```python\n{starter}\n```")
FMT_STDIN = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test "
             "on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
             "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
             "```python\n# YOUR CODE HERE\n```")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--since", default=None, help="only problems with contest_date >= YYYY-MM-DD")
    a = ap.parse_args()

    # test.jsonl, test2.jsonl, ... testN.jsonl: newer releases are in higher-numbered files (~4.5 GB total),
    # so download newest-first and stop once we have enough problems.
    names = [f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith("test") and f.endswith(".jsonl")]
    files_newest_first = sorted(names, key=lambda f: int(f[4:-6] or 1), reverse=True)
    rows, files = [], []
    for f in files_newest_first:
        files.append(f)
        with open(hf_hub_download(REPO, f, repo_type="dataset")) as fh:
            rows += [json.loads(l) for l in fh]
        if len([r for r in rows if not a.since or r["contest_date"] >= a.since]) >= a.n:
            break
    rows = [r for r in rows if not a.since or r["contest_date"] >= a.since]
    rows.sort(key=lambda r: r["contest_date"], reverse=True)
    rows = rows[: a.n]

    with open(a.out, "w") as po, open(a.out + ".tests", "w") as to:
        for r in rows:
            fmt = FMT_STARTER.format(starter=r["starter_code"]) if r["starter_code"].strip() else FMT_STDIN
            user = f"### Question:\n{r['question_content']}\n\n### Format: {fmt}\n\n### Answer: (use the provided format with backticks)\n\n"
            po.write(json.dumps({"id": r["question_id"],
                                 "messages": [{"role": "system", "content": SYSTEM}],
                                 "turns": [{"user": user, "canned_reply": None}]}) + "\n")
            to.write(json.dumps({k: r[k] for k in ("question_id", "platform", "difficulty", "contest_date",
                                                   "starter_code", "public_test_cases", "private_test_cases",
                                                   "metadata")}) + "\n")
    dates = [r["contest_date"][:10] for r in rows]
    print(f"wrote {len(rows)} problems ({min(dates)} .. {max(dates)}) from {files} to {a.out}")


if __name__ == "__main__":
    main()
