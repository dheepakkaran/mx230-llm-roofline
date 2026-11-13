### Name and Version

```
version: 1.0.0-dev (build 1, commit b79696f)
built with GNU 13.4.0 for Linux x86_64
```

Built with CUDA 12.9 (CUDA 13 cannot target sm_61), `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61`.

### Operating systems

Linux (Ubuntu, kernel 7.0, NVIDIA driver 580.178.04)

### GGML backends

CUDA

### Hardware

NVIDIA GeForce MX230 laptop GPU (Pascal GP108, compute capability 6.1, 2 SMs, 2 GB GDDR5, 64-bit bus) + Intel i7-8565U.

### Models

Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct, official GGUFs from `Qwen/Qwen2.5-*-Instruct-GGUF` (Q4_0, Q8_0).

### Problem description & steps to reproduce

On this Pascal GPU, token generation is **20–36% faster with `GGML_CUDA_DISABLE_FUSION=1`**, and the output is unchanged. The default (fusion on) is slower than not fusing.

```
llama-bench -m qwen2.5-0.5b-instruct-q4_0.gguf,qwen2.5-1.5b-instruct-q4_0.gguf -ngl 99 -fa on -p 0 -n 128 -r 3
GGML_CUDA_DISABLE_FUSION=1 llama-bench <same args>
```

| model | fusion on (default) | `GGML_CUDA_DISABLE_FUSION=1` | change |
|---|---|---|---|
| Qwen2.5-0.5B Q4_0 | 44.0 t/s | 59.5 t/s | +35% |
| Qwen2.5-0.5B Q8_0 | 41.3 t/s | 49.4 t/s | +20% |
| Qwen2.5-1.5B Q4_0 | 19.8 t/s | 25.8 t/s | +30% |

Greedy output (`--temp 0`, 128 tokens, two prompts, 1.5B Q4_0) is byte-for-byte identical with fusion on and off.

**Where the time goes (nvprof `--print-gpu-trace`, 0.5B Q4_0, 16 decode steps; kernel roles inferred from grid size):**

| kernel | template | grid.x | regs/thread | avg time |
|---|---|---|---|---|
| FFN gate+up, fused | `mul_mat_vec_q<Q4_0, 1, true, true, false>` | 1216 | 80 | **475.5 µs** |
| FFN gate, unfused | `mul_mat_vec_q<Q4_0, 1, false, true, false>` | 1216 | 66 | 127.0 µs |
| FFN up, unfused | same | 1216 | 66 | 127.0 µs |
| 896-row matvecs (attn q/out), fused variant | `mul_mat_vec_q<Q4_0, 1, true, true, false>` | 224 | 80 | 47.6 µs |
| same matvecs, unfused | `mul_mat_vec_q<Q4_0, 1, false, true, false>` | 224 | 66 | 26.4 µs |

So the fused gate+up kernel takes 475 µs where the two separate kernels take 2 × 127 = 254 µs (1.87× slower). It moves ~4.9 MB, i.e. ~10 GB/s, on a GPU where a plain read kernel measures 36.5 GB/s and the unfused/LM-head matvecs reach 21–31 GB/s. With fusion on, this one kernel is about half of each token's time on the 0.5B model.

The fused variant uses 80 registers/thread vs 66 unfused. I could not collect occupancy/stall counters to confirm the cause: `nvprof --metrics` fails with `CUDA profiling error` on this driver even as root, and Nsight Compute does not support Pascal.

**Notes**

- CUDA graphs are off for pre-Volta (`ggml_cuda_graph_set_enabled`). As a check I force-enabled them locally and confirmed they replay (126 of 128 decode steps); tokens/s did not change (44.0 vs 43.8), so launch overhead is not the cause here. That local patch is env-gated and was inactive for every number above.
- A possible fix: skip the mmvq fusion paths (`ggml_cuda_should_fuse_mul_mat_vec_q` and the gate/up GLU fusion) for `cc < GGML_CUDA_CC_VOLTA`, or at least for devices with few SMs. I am happy to test a patch on this GPU.

### First Bad Commit

Not bisected.

### Relevant log output

```
# fusion on (default)
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen2 1B Q4_0                  | 403.20 MiB |   630.17 M | CUDA       |  99 |   1 |           tg128 |         44.02 ± 0.11 |
| qwen2 1.5B Q4_0                | 1011.16 MiB |     1.78 B | CUDA       |  99 |   1 |           tg128 |         19.84 ± 0.03 |

build: b79696f (1)

# GGML_CUDA_DISABLE_FUSION=1
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen2 1B Q4_0                  | 403.20 MiB |   630.17 M | CUDA       |  99 |   1 |           tg128 |         59.47 ± 0.05 |
| qwen2 1.5B Q4_0                | 1011.16 MiB |     1.78 B | CUDA       |  99 |   1 |           tg128 |         25.81 ± 0.01 |

build: b79696f (1)
```
