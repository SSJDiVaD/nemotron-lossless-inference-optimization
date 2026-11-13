#!/usr/bin/env bash
# Launch all lane sweeps in a tmux session "lanes" (one window per lane). Survives SSH drops.
#   ./launch_lanes.sh            then:  tmux attach -t lanes   (Ctrl-b n / p to switch windows)
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$BENCH_DIR"
tmux new-session -d -s lanes -n lane0 "KEEP_LAST=1 ./sweep.sh 0 8000 configs/lane0.sh short 2>&1 | tee $RESULTS/logs/lane0.log; bash"
sleep 15
for g in 1 3 4 5 6; do
  tmux new-window -t lanes -n "lane$g" "./sweep.sh $g $((8000 + g)) configs/lane$g.sh short 2>&1 | tee $RESULTS/logs/lane$g.log; bash"
  sleep 15
done
# Lane 2 needs the 32-session trace to overflow its shrunken KV pool.
tmux new-window -t lanes -n lane2 "./sweep.sh 2 8002 configs/lane2.sh full 2>&1 | tee $RESULTS/logs/lane2.log; bash"
sleep 15
tmux new-window -t lanes -n lane7 "KEEP_LAST=1 ./sweep.sh 7 8007 configs/lane7.sh short 2>&1 | tee $RESULTS/logs/lane7.log; bash"
tmux new-window -t lanes -n watch "watch -n 30 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv; echo; tail -n 3 $RESULTS/failures.txt 2>/dev/null; echo; $VLLM_VENV/bin/python summarize.py --results $RESULTS --brief'"
echo "lanes launched: tmux attach -t lanes"
