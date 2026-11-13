#!/usr/bin/env bash
cd ~/lambda-bench && source ~/.openrouter
export BENCH_EXTRA_BODY='{"provider":{"order":["Together"],"allow_fallbacks":false},"chat_template_kwargs":{"enable_thinking":true}}'
LCB_MAX_TOKENS=32768 ./bench_suite.sh https://openrouter.ai/api/v1 nvidia/nemotron-3.5-super-vl together_lcb32k lcb OPENROUTER_API_KEY 2>&1 | tee results/logs/together_lcb32k.log
