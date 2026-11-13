#!/usr/bin/env bash
# Provider comparison run against a DEDICATED endpoint (one replica, fixed hardware). SPENDS MONEY unless PROVIDER=mock.
#   PROVIDER=fireworks SHAPE=<shape-version> MODE=smoke|full [HOURLY_USD=..] bench/provider/run_provider_bench.sh
#   PROVIDER=together  SHAPE=<cr_config> MODEL=<modelName> MODE=...
# Uses the same client, traces and settings as our own runs (bench_suite.sh). The endpoint is ALWAYS deleted on exit
# (trap); the state file is also kept for the CI workflow's independent always() teardown job.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; BENCH="$(cd "$HERE/.." && pwd)"; REPO="$(cd "$BENCH/.." && pwd)"
: "${PROVIDER:?set PROVIDER}"; : "${MODE:=smoke}"; : "${SHAPE:=}"; : "${MODEL:=}"; : "${HOURLY_USD:=0}"
: "${MAX_HOURS:=4}"
STAMP=$(date -u +%Y%m%dT%H%MZ)
export RESULTS="${RESULTS:-$REPO/results/provider_${PROVIDER}_${STAMP}}"
mkdir -p "$RESULTS/logs"
STATE="${STATE_FILE:-$REPO/endpoint_state.json}"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/null}"
note() { echo "[$(date -u +%H:%M:%S)] $*"; echo "- $(date -u +%H:%M:%S) $*" >> "$SUMMARY"; }
EP="python3 $HERE/endpoint.py"
T0=$(date +%s)

cleanup() {
  local rc=$?
  note "teardown: deleting endpoint"
  $EP delete --state-file "$STATE" || note "**DELETE FAILED - check the provider console**"
  local hrs; hrs=$(python3 -c "print(round(($(date +%s)-$T0)/3600, 2))")
  note "elapsed ${hrs} h; estimated endpoint spend \$$(python3 -c "print(round($hrs*$HOURLY_USD, 2))") at \$$HOURLY_USD/h"
  exit $rc
}

# 1. create (+ teardown registered immediately)
create_args=(--provider "$PROVIDER" --state-file "$STATE")
[[ -n $SHAPE ]] && create_args+=(--shape "$SHAPE"); [[ -n $MODEL ]] && create_args+=(--model "$MODEL")
if [[ ${SKIP_CREATE:-0} == 1 && -f $STATE ]]; then
  note "using existing endpoint from $STATE (created by an earlier CI step)"
  python3 -c "import json;s=json.load(open('$STATE'));print(json.dumps({k:s.get(k) for k in ('provider','base_url','model')}))" > "$RESULTS/create.json"
else
  $EP create "${create_args[@]}" > "$RESULTS/create.json" || { note "create FAILED"; exit 1; }
fi
trap cleanup EXIT INT TERM
note "endpoint created: $(cat "$RESULTS/create.json")"

# 2. wait until a real completion succeeds, then record endpoint metadata
$EP wait --state-file "$STATE" --timeout-min "${WAIT_MIN:-60}" > "$RESULTS/ready.json" || { note "endpoint never became ready"; exit 1; }
$EP describe --state-file "$STATE" > "$RESULTS/endpoint.json"
note "endpoint ready: $(cat "$RESULTS/ready.json")"

BASE_URL=$(python3 -c "import json;print(json.load(open('$STATE'))['base_url'])")
EMODEL=$(python3 -c "import json;print(json.load(open('$STATE'))['model'])")
KEYENV=$(python3 -c "import json;print(json.load(open('$STATE')).get('api_key_env') or '')")
# Every response must report a Nemotron model; mismatch => run fails (exit 3) instead of silently comparing the wrong thing.
export BENCH_EXPECT_MODEL="${BENCH_EXPECT_MODEL:-(?i)nemotron|provider-bench}"
export BENCH_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":true}}'   # same as our runs; no routing fields

# 3. network round trip from this runner (subtract from TTFT for the adjusted number)
for _ in 1 2 3 4 5; do
  curl -s -o /dev/null -w 'connect=%{time_connect} tls=%{time_appconnect} ttfb=%{time_starttransfer}\n' "${BASE_URL%/}/models" \
    ${KEYENV:+-H "Authorization: Bearer ${!KEYENV}"}
done | tee "$RESULTS/rtt.txt"

# 4. suites (same as our runs), under a hard wall-clock cap
LABEL="provider_${PROVIDER}"
suite() { timeout "$(python3 -c "print(int(max(60, $MAX_HOURS*3600 - ($(date +%s)-$T0))))")" \
            "$BENCH/bench_suite.sh" "$BASE_URL" "$EMODEL" "${LABEL}_$1" "$2" $KEYENV 2>&1 | tee -a "$RESULTS/logs/suites.log"; }
case $MODE in
  smoke)
    python3 "$BENCH/bench.py" --base-url "$BASE_URL" --model "$EMODEL" --label "${LABEL}_smoke" --out-dir "$RESULTS" \
      --trace "$BENCH/traces/agentic_short.jsonl" --max-sessions 2 --concurrency 1 --max-tokens 256 \
      --extra-body "$BENCH_EXTRA_BODY" ${KEYENV:+--api-key-env $KEYENV} 2>&1 | tee -a "$RESULTS/logs/suites.log"
    rc=${PIPESTATUS[0]} ;;
  full)
    suite short short; suite full full
    LCB_MAX_TOKENS=32768 suite lcb32k lcb
    rc=0 ;;
  *) note "unknown MODE=$MODE"; exit 2 ;;
esac
grep -q "FATAL" "$RESULTS/logs/suites.log" && { note "**expect-model check FAILED - results not comparable**"; rc=3; }
note "suites finished (mode=$MODE, rc=$rc)"

# 5. comparison table vs our runs
python3 "$HERE/compare.py" --provider-results "$RESULTS" --ours "$REPO/results" > "$RESULTS/COMPARISON.md" 2>>"$RESULTS/logs/compare.log" \
  && note "comparison written: $(basename "$RESULTS")/COMPARISON.md"
exit $rc
