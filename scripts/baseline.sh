#!/usr/bin/env bash
# Step 2 baseline: llama-bench, all layers on GPU, pp512 + tg128, 3 reps per model.
# Logs GPU temperature before/after each model; stops if the GPU passes 85 C.
cd "$(dirname "$0")/.."
B=build/llama/bin/llama-bench
for m in qwen2.5-0.5b-instruct-{fp16,q8_0,q4_k_m,q4_0} qwen2.5-1.5b-instruct-{q8_0,q4_k_m,q4_0}; do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
  echo "$m temp_before=$t" | tee -a results/raw/baseline-temps.log
  [ "$t" -gt 85 ] && { echo "too hot, stopping"; exit 1; }
  timeout 300 $B -m models/$m.gguf -ngl 99 -p 512 -n 128 -r 3 -o json > results/raw/baseline-$m.json 2> results/raw/baseline-$m.err \
    || echo "$m FAILED (see .err)" | tee -a results/raw/baseline-temps.log
  echo "$m temp_after=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" | tee -a results/raw/baseline-temps.log
  sleep 5
done
