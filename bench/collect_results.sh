#!/usr/bin/env bash
# Pull results (and server logs) from every box into ./collected/<box>/, then build one combined report.
set -uo pipefail
cd "$(dirname "$0")"
K="-i $HOME/.ssh/lambda_ed25519 -o ConnectTimeout=15"
declare -A BOXES=( [h100]=<instance-ip> [gh200]=<instance-ip> [h100pcie2]=<instance-ip> )  # h100pcie (<instance-ip>) already pulled and terminated
mkdir -p collected/all
for name in "${!BOXES[@]}"; do
  ip=${BOXES[$name]}
  rsync -a -e "ssh $K" "ubuntu@$ip:lambda-bench/results/" "collected/$name/" && echo "pulled $name ($ip)"
  [[ $name == h100 ]] && rsync -a -e "ssh $K" "ubuntu@$ip:archive_h100_v1/" "collected/h100_v1_archive/" 2>/dev/null
  # merge into one tree for summarize.py (labels are already box-prefixed: h100_*, gh200_*, h100pcie_*, together_*)
  rsync -a --exclude logs --exclude pids "collected/$name/" collected/all/
done
[ -d collected/h100pcie ] && rsync -a --exclude logs --exclude pids collected/h100pcie/ collected/all/
"${PY:-python3}" summarize.py --results collected/all > /dev/null && echo "report: collected/all/REPORT.md"
