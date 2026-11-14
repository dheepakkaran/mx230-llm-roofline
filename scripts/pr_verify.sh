#!/usr/bin/env bash
# Verify the one-line Pascal fusion-guard fix on the PR branch (sandbox/llama-cpp-pr, latest master + fix):
# build fixed → bench + MUL_MAT_FUSION tests + greedy outputs; revert → rebuild → bench + greedy; restore fix.
set -u
P=~/workspace/projects/mx230-llm-roofline; S=~/workspace/sandbox/llama-cpp-pr; O=$P/results/raw/pr-verify
export PIXI_HOME=~/workspace/.tools/pixi PIXI_CACHE_DIR=~/workspace/.tools/pixi-cache
PX="$HOME/workspace/.tools/bin/pixi run --manifest-path $P/pixi.toml"
cool(){ until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 62 ]; do sleep 5; done; }
build(){ $PX cmake --build $S/build -j 4 --target llama-bench llama-completion test-backend-ops > $O/build-$1.log 2>&1 || { echo "build $1 failed"; exit 1; }; }
bench(){ for m in qwen2.5-0.5b-instruct-q4_0 qwen2.5-0.5b-instruct-q8_0 qwen2.5-0.5b-instruct-q4_k_m qwen2.5-1.5b-instruct-q4_0; do
  cool; $S/build/bin/llama-bench -m $P/models/$m.gguf -ngl 99 -fa on -p 512 -n 128 -r 3 -o json > $O/bench-$1-$m.json 2>/dev/null; done; }
greedy(){ for p in code prose; do $S/build/bin/llama-completion -m $P/models/qwen2.5-1.5b-instruct-q4_0.gguf -ngl 99 -fa on -f $P/scripts/prompts/$p.txt -n 128 --temp 0 -no-cnv --no-display-prompt < /dev/null > $O/greedy-$1-$p.txt 2>/dev/null; done; }
unset GGML_CUDA_DISABLE_FUSION GGML_CUDA_MX_Q4 GGML_CUDA_FORCE_GRAPHS_PRE_VOLTA
cd $S && git rev-parse --short HEAD > $O/commit.txt && git diff --stat >> $O/commit.txt
$PX cmake -S $S -B $S/build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets > $O/cmake.log 2>&1 || { echo cmake failed; exit 1; }
echo "[1/6] build fixed"; build fixed
echo "[2/6] bench fixed"; bench fixed; greedy fixed
echo "[3/6] tests fixed"; cool; timeout 1500 $S/build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT_FUSION > $O/test-fixed.log 2>&1; tail -3 $O/test-fixed.log | tr -d '\033'
echo "[4/6] build upstream"; git -C $S stash push -q -m pr-fix; build upstream
echo "[5/6] bench upstream"; bench upstream; greedy upstream
echo "[6/6] restore fix"; git -C $S stash pop -q; build fixed2; git -C $S diff --stat
echo done
