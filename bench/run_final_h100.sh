#!/usr/bin/env bash
# H100 final: wait for sweep, then final = prefix caching + MTP k3; extra full bench = prefix caching only.
cd ~/lambda-bench
until grep -q "lane gpu0 done" results/logs/sweep_h100.log; do sleep 15; done
PC="--enable-prefix-caching --enable-mamba-shared-prefix-checkpoint --block-size 64 --prefix-match-unit 64"
SPEC="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"
P=h100 FLAGS=h100_flags.sh FINAL="$PC $SPEC" EXTRA_LABEL=h100_full_pc EXTRA="$PC" ./final_box.sh 2>&1 | tee results/logs/final_h100.log
