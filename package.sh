#!/usr/bin/env bash
# Assemble the submission repo layout from the working folders, then scan for secrets / IPs.
set -euo pipefail
cd "$(dirname "$0")"
# top-level docs + deploy + env live in submission/ while drafting
[ -d submission ] && { mv -f submission/NOTES.md . 2>/dev/null || true; rsync -a submission/deploy/ deploy/; rsync -a submission/env/ env/; rm -rf submission; }
# bench/ = the harness, without local results/collections/caches
rsync -a --delete --exclude collected --exclude results --exclude __pycache__ lambda-bench/ bench/
# results/ = per-box raw results (+ merged summary/charts), without huge raw profiler traces
mkdir -p results
for b in h100 gh200 h100pcie h100pcie2 h100_v1_archive; do
  [ -d "lambda-bench/collected/$b" ] && rsync -a --exclude '*.pt.trace.json.gz' --exclude pids "lambda-bench/collected/$b/" "results/$b/"
done
[ -d lambda-bench/collected/charts ] && rsync -a lambda-bench/collected/charts/ results/charts/
[ -d lambda-bench/collected/all ] && rsync -a --include 'REPORT.md' --include 'chart_*.png' --exclude '*' lambda-bench/collected/all/ results/summary/
# scrub instance IPs (public, but no reason to publish them) from text files
grep -rlE '192\.222\.[0-9]+\.[0-9]+|209\.20\.[0-9]+\.[0-9]+|192-222-[0-9]+-[0-9]+' --include='*' bench deploy results env 2>/dev/null \
  | xargs -r sed -i -E 's/192\.222\.[0-9]+\.[0-9]+/<instance-ip>/g; s/209\.20\.[0-9]+\.[0-9]+/<instance-ip>/g; s/192-222-[0-9]+-[0-9]+/instance/g'
for f in results/*/profile_h100/192-222-*; do [ -e "$f" ] && mv "$f" "${f/192-222-53-82/instance}"; done 2>/dev/null || true
echo "--- secret scan (should be empty):"
grep -rnE 'sk-or-[A-Za-z0-9_-]{10,}|secret_[a-z0-9_-]+_[a-f0-9]{20,}|BEGIN (OPENSSH|RSA) PRIVATE KEY|OPENROUTER_API_KEY=[^ "$]+|LAMBDA_API_KEY=[^ "$]+' . --exclude-dir=.git || true
echo "--- size:"; du -sh bench deploy results env 2>/dev/null; find . -path ./.git -prune -o -type f -size +50M -print
