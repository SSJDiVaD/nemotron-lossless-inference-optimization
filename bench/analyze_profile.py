#!/usr/bin/env python3
"""Summarize GPU kernel time from a vLLM torch-profiler trace directory by kernel family."""
import glob, gzip, json, re, sys, collections
d = sys.argv[1]
files = sorted(glob.glob(f"{d}/**/*.json*", recursive=True))
if not files:
    sys.exit(f"no trace files in {d}")
cats = [("MoE experts (FP4 W4A16 GEMM)", r"humming|marlin|moe|expert|topk|fused_moe"),
        ("Mamba / SSM", r"mamba|ssd|selective|causal_conv|chunk_scan|chunk_state|state_passing|ssm"),
        ("Attention", r"flash|attn|fmha|mha|paged"),
        ("Dense GEMM (FP8/BF16)", r"gemm|cutlass|sm90|cublas|matmul|nvjet|wgmma"),
        ("Sampling / spec-decode", r"sampl|argmax|softmax|rejection|top_?k|top_?p|draft"),
        ("Norm / elementwise / other", r".")]
tot = collections.Counter(); names = collections.Counter(); n_kernels = 0
for f in files:
    data = json.load(gzip.open(f) if f.endswith(".gz") else open(f))
    for ev in data.get("traceEvents", []):
        if ev.get("cat") in ("kernel", "gpu_memcpy") and "dur" in ev:
            n_kernels += 1
            nm = ev["name"].lower()
            for c, pat in cats:
                if re.search(pat, nm):
                    tot[c] += ev["dur"]; break
            names[ev["name"][:80]] += ev["dur"]
total = sum(tot.values()) or 1
print(f"trace files: {len(files)}  kernels: {n_kernels}  total GPU kernel time: {total/1000:.1f} ms")
for c, _ in cats:
    print(f"  {c:32s} {tot[c]/1000:9.1f} ms  {100*tot[c]/total:5.1f}%")
print("top kernels:")
for nm, t in names.most_common(12):
    print(f"  {100*t/total:5.1f}%  {nm}")
