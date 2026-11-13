#!/usr/bin/env bash
cd ~/lambda-bench
until [ "$(grep -c 'lane gpu0 done' results/logs/sweep_gh200.log)" -ge 2 ]; do sleep 15; done
source gh200_flags.sh && ./sweep.sh 0 8000 configs/gh200_extra2.sh short 2>&1 | tee -a results/logs/sweep_gh200.log
