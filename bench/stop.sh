#!/usr/bin/env bash
# Stop the server on <port> and wait until its process tree is gone.
source "$(dirname "$0")/env.sh"
PORT=$1; PIDF="$RESULTS/pids/$PORT.pid"
[[ -f $PIDF ]] || exit 0
PID=$(cat "$PIDF")
pkill -TERM -P "$PID" 2>/dev/null; kill -TERM "$PID" 2>/dev/null
for _ in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
pkill -KILL -P "$PID" 2>/dev/null; kill -KILL "$PID" 2>/dev/null
rm -f "$PIDF"
sleep 5   # let the GPU memory actually free before the next server starts
