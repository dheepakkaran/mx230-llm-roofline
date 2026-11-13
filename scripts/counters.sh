#!/usr/bin/env bash
# Hardware counters for the fused vs unfused Q4_0 matvec kernels. On driver 580 nvprof --metrics failed even as root.
# Run:  sudo bash scripts/counters.sh
# Writes results/raw/counters-{fused,unfused}.csv (owned by the calling user) and prints a summary.
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
ENV="$P/.pixi/envs/default"
NVPROF="$ENV/bin/nvprof"
BIN="$P/build/llama/bin/llama-completion"
MODEL="$P/models/qwen2.5-0.5b-instruct-q4_0.gguf"
OUT="$P/results/raw"
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
export LD_LIBRARY_PATH="$ENV/lib:$ENV/extras/CUPTI/lib64:$P/build/llama/bin:${LD_LIBRARY_PATH:-}"
METRICS=achieved_occupancy,sm_efficiency,issue_slot_utilization,eligible_warps_per_cycle,dram_read_throughput,gld_efficiency,inst_executed,stall_memory_dependency,stall_exec_dependency,stall_not_selected,stall_pipe_busy,stall_sync,local_load_transactions,local_store_transactions

for mode in fused unfused; do
  if [ $mode = unfused ]; then export GGML_CUDA_DISABLE_FUSION=1; else unset GGML_CUDA_DISABLE_FUSION; fi
  echo "== $mode (takes 1-3 min: each kernel is replayed per metric)"
  timeout 900 "$NVPROF" --kernels "mul_mat_vec_q" --metrics "$METRICS" --csv --log-file "$OUT/counters-$mode.csv" \
    "$BIN" -m "$MODEL" -ngl 99 -fa on -p "Hi" -n 2 --temp 0 -no-cnv --no-display-prompt < /dev/null > /dev/null 2> "$OUT/counters-$mode.err"
  echo "   exit $?"
done
chown "${SUDO_USER:-root}:" "$OUT"/counters-*.csv "$OUT"/counters-*.err 2>/dev/null
grep -h -m1 -E "Error|error" "$OUT"/counters-*.csv | head -2
echo "done. See results/raw/counters-*.csv"
