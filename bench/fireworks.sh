#!/usr/bin/env bash
# Fireworks side of the comparison: a DEDICATED on-demand 1x B200 deployment of the same NVFP4 checkpoint.
# (Nemotron 3.5 Super VL is on-demand only on Fireworks; billed per GPU-second, ~$13/hr for B200.)
#
#   export FIREWORKS_API_KEY=...  FIREWORKS_ACCOUNT=<your-account-id>
#   ./fireworks.sh create      # at 0:00 (deployment takes a while to come up)
#   ./fireworks.sh status
#   ./fireworks.sh bench short|full|lcb
#   ./fireworks.sh delete      # at 3:15 -- don't forget, it bills until deleted / scaled to zero
#
# VERIFY pre-clock: install firectl (https://docs.fireworks.ai), run `firectl create deployment --help`, and
# confirm the model id and accelerator-type spelling below against the model page / docs.
set -euo pipefail
source "$(dirname "$0")/env.sh"
FW_MODEL="accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-nvfp4"
FW_ACCEL="${FW_ACCEL:-NVIDIA_B200_180GB}"
FW_URL="https://api.fireworks.ai/inference/v1"
STATE="$RESULTS/fireworks_deployment_id"

case ${1:-} in
  create)
    firectl create deployment "$FW_MODEL" --accelerator-type "$FW_ACCEL" --accelerator-count 1 \
      --min-replica-count 1 --max-replica-count 1 | tee "$RESULTS/logs/fireworks_create.log"
    # Grab the deployment id from the output (format: accounts/<acct>/deployments/<id>).
    grep -oE 'accounts/[^/ ]+/deployments/[A-Za-z0-9-]+' "$RESULTS/logs/fireworks_create.log" | head -1 > "$STATE"
    echo "deployment: $(cat "$STATE")" ;;
  status)
    firectl get deployment "$(basename "$(cat "$STATE")")" ;;
  bench)
    # Fireworks routes to a dedicated deployment via "<model>#<deployment>".
    "$BENCH_DIR/bench_suite.sh" "$FW_URL" "$FW_MODEL#$(cat "$STATE")" "fireworks_dedicated_${2:-short}_$(date +%H%M)" \
      "${2:-short}" FIREWORKS_API_KEY ;;
  rtt)
    # Network round trip to Fireworks from this machine; subtract from Fireworks TTFT for the adjusted number.
    for _ in 1 2 3 4 5; do curl -s -o /dev/null -w '%{time_connect} %{time_appconnect}\n' "$FW_URL/models" \
      -H "Authorization: Bearer $FIREWORKS_API_KEY"; done ;;
  delete)
    firectl delete deployment "$(basename "$(cat "$STATE")")" && echo "deleted" ;;
  *) echo "usage: $0 create|status|bench <short|full|lcb>|rtt|delete" >&2; exit 2 ;;
esac
