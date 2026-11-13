#!/usr/bin/env bash
# GH200 final: wait for sweep + queued extras, then final = combined best (PC + MTP k3 + async + big batches).
cd ~/lambda-bench
until [ "$(grep -c 'lane gpu0 done' results/logs/sweep_gh200.log)" -ge "${NEED_DONE:-2}" ]; do sleep 15; done
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
SPEC="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
P=gh200 FLAGS=gh200_flags.sh FINAL="$PC $SPEC --async-scheduling --max-num-batched-tokens 8192 --max-num-seqs 64" ./final_box.sh 2>&1 | tee results/logs/final_gh200.log
