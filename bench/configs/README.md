# Lane config files

Each `laneN.sh` defines `CONFIGS=( "label|engine|ENV=VAL ...|flags" ... )`, which `sweep.sh` runs in order.

- **Quoting:** wrap JSON flag values in single quotes and escape their double quotes, like this:
  `"mtp_k3|vllm||--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"`
- **Labels:** each label must be unique across all lanes, because it names the results directory.
- **Failures:** a config whose server fails to start is written to `results/failures.txt` and skipped.

Flags marked `VERIFY` are from memory of recent vLLM/SGLang releases. Before the clock starts, check them with
`vllm serve --help=all | grep -A3 <flag>` (or the SGLang `--help`) on the version you pinned. If one turns out
wrong, the sweep loses that config (a few minutes), not the whole lane.
