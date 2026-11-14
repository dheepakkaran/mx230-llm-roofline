# Results

GPU: GeForce MX230 (sm_61, 2 SMs, 2 GB). llama.cpp b79696f, CUDA 12.9, all layers on GPU.
Every number below is from `results/raw/` (gitignored) via `scripts/roofline.py`.

## Step 1: bandwidth ceiling (`kernels/bw.cu`)

| test | GB/s |
|---|---|
| driver spec (3504 MHz × 2 × 64-bit) | 56.1 |
| read-only stream, measured | **36.5** (65% of spec) |
| copy, measured | 33.9 |

Decode ceiling (tok/s) = 36.5 GB/s ÷ bytes read per token. These GGUF files store `token_embd` and
`output` (LM head) separately, so a token reads every weight except `token_embd` (only one row of it).
`scripts/bytes_per_token.py` computes this from the GGUF tensors (`results/bytes_per_token.json`).
(Correction, 2026-11-13: the first version of this table divided by the whole file size, which made
the ceilings too low and the % too high.)

## Step 2: llama.cpp baseline (pp512, tg128, 3 reps, 2026-11-13)

| model | file MiB | prefill tok/s (pp512) | decode tok/s (tg128) | ceiling tok/s (36.5 GB/s ÷ bytes/token) | % of ceiling |
|---|---|---|---|---|---|
| 0.5b-fp16 | 1202 | 606 | 19.9 ± 0.1 | 36.9 | 54% |
| 0.5b-q4_0 | 403 | 1042 | 43.5 ± 0.0 | 105.4 | 41% |
| 0.5b-q4_k_m | 463 | 992 | 39.4 ± 0.1 | 93.1 | 42% |
| 0.5b-q8_0 | 639 | 1009 | 41.4 ± 0.0 | 69.5 | 60% |
| 1.5b-q4_0 | 1011 | 361 | 19.5 ± 0.1 | 39.3 | 50% |
| 1.5b-q4_k_m | 1060 | 336 | 16.6 ± 0.2 | 37.2 | 45% |
| 1.5b-q8_0 | 1801 | 346 | 14.2 ± 0.0 | 22.2 | 64% |

GPU temperature stayed between 55 and 79 °C (`results/raw/baseline-temps.log`).

### What it shows

- **Every quant type is far from the wall, and Q4 is furthest (41–50%).** For 0.5B, Q4_0 reads 34%
  fewer bytes than Q8_0 but is only 5% faster (43.5 vs 41.4 tok/s). Step 4 finds why.
- FP16 is the slowest (no fast FP16 on this chip, and 3× the bytes of Q4).
- 1.5B Q8_0 (1.8 GB) still fits fully in 2 GB at this context size.

## Step 3: tricks already in llama.cpp

All Step 3 runs used llama.cpp defaults, i.e. with kernel fusion on (see 4c: turning it off is faster).

### 3a. Flash attention + INT8 KV cache (1.5B Q4_0, tg64, 2 reps)

| setting | decode @ depth 0 | decode @ depth 4096 | prefill pp512 @ 4096 |
|---|---|---|---|
| FA off, KV f16 | 18.4 | 9.5 | 218 |
| FA on, KV f16 | 19.7 | **16.4** | 214 |
| FA on, KV q8_0 | 19.3 | 12.5 | 210 |

- Flash attention almost doubles decode at a 4K context (9.5 → 16.4 tok/s). FA has no Pascal path in the
  FlashAttention library, but llama.cpp's own FA kernels run on sm_61.
- INT8 KV halves KV memory (Qwen2.5-1.5B: 28 layers × 2 KV heads × 128 dims → 28 KiB/token in f16,
  ~15 KiB in q8_0; 112 MiB vs 60 MiB at 4K) but is **24% slower** at 4K. Same pattern as Q4 weights:
  on 2 SMs, unpacking costs more than the bytes it saves. Use it only when the context does not fit.
- GPU reached 81 °C in this run; cool-down added before the next one.

### 3b. Speculative decoding (target 1.5B Q4_0, draft 0.5B Q4_0, both on GPU, greedy, 256 tokens)

| prompt | target only | speculative | draft accept rate |
|---|---|---|---|
| code (Python function) | 19.7 tok/s | 20.2 (n_draft 3) · 20.1 (2) · 20.3 (5) · 18.4 (8) | 80% · 81% · 75% · 58% |
| prose (why the sky is blue) | 19.7 | **14.0** (n_draft 3) | 45% |

- **Almost no gain on code (+3%), a 29% loss on prose.** The draft is too expensive here: 0.5B Q4_0 runs
  at 43 tok/s (~23 ms/token) against the target's 51 ms/token, so each drafted token costs ~45% of a
  target step, and both models share the same 36.5 GB/s bus. Speculative decoding pays off when the
  draft is ≥10× cheaper; on this GPU that ratio is ~2×.
- Note for this llama.cpp build: speculative needs `--spec-type draft-simple` (default is `none`).

### 3c. CPU/GPU layer split + batch size (Qwen2.5-3B Q4_0, 1905 MiB, 36 layers + output)

Short context (pp128, tg32, FA on):

| GPU layers | 0 | 12 | 20 | 26 | 30 | 33 | 35 | 37 (all) |
|---|---|---|---|---|---|---|---|---|
| decode tok/s | 3.2 | 3.1 | 5.0 | 7.6 | 7.7 | 9.0 | 10.8 | **12.0** |
| prefill tok/s | 94 | 98 | 133 | 149 | 161 | 164 | 181 | 189 |

- Speed grows with every layer moved to the GPU; there is no sweet spot below "all". At 12.0 tok/s the
  full-GPU 3B is at 60% of its 20.1 tok/s ceiling (36.5 GB/s ÷ 1.82 GB read per token).
- **A 3B model fits in 2 GB, but only at a short context.** At depth 2048/4096 with default batch
  (ub 512) it runs out of memory: the 315 MiB compute buffer does not fit. Even 33 layers fails.

Long context, all 37 layers on GPU, FA on:

| KV cache | ubatch | fits at 4K? | decode @2K | decode @4K | prefill @4K |
|---|---|---|---|---|---|
| f16 | 512 | no (OOM) | – | – | – |
| f16 | 128 | no (OOM) | – | – | – |
| **q8_0** | **128** | **yes** | **10.6** | **9.1** | 108 |
| q8_0 | 64 | yes | 10.3 | 9.1 | 104 |

- **Tricks combine:** INT8 KV was a 24% slowdown on its own (3a), but together with a smaller prefill
  batch it is what lets the whole 3B model plus a 4K context live in 2 GB. Smaller batch costs little
  prefill speed (ub 128 → 64 = −4%).
- Thermal: these runs pushed the GPU to 86–90 °C. Later runs are capped shorter.

### 3d. Prefix (prompt) cache (1.5B Q4_0, 1,673-token system prompt + question, 16 new tokens)

| run | prompt processing | total |
|---|---|---|
| no cache | 5.06 s (331 tok/s) | 5.90 s |
| cache, first run (writes it) | 5.07 s | 6.01 s |
| **cache, second run** | **~0 s** (state loaded from a 46 MiB file) | **0.93 s** |

- A long fixed system prompt costs 5 s before the first word on this GPU. Caching it cuts the whole
  answer from 5.9 s to 0.9 s (6.4×). This is the biggest user-visible win in Step 3.

### 3e. Draft-free speculative decoding (n-gram), code prompt

| method | tok/s | accept |
|---|---|---|
| target only | 19.7 | – |
| ngram-simple | 19.3 | 0% |
| ngram-cache | 19.1 | 6% |

- N-gram drafting needs repeated text (editing, RAG quotes). On fresh generation it drafts almost
  nothing, so it costs a little and gains nothing.

## Step 3 summary

| trick | best result on MX230 | verdict |
|---|---|---|
| Flash attention | decode @4K 9.5 → 16.4 tok/s (1.7×) | always on |
| INT8 KV cache | −24% speed alone, but halves KV memory | only to fit long context |
| Smaller prefill batch (ub 128) | frees ~240 MiB, −4% prefill | needed to fit 3B @4K |
| Layer split | all-GPU always fastest; 3B: 3.2 → 12.0 tok/s | put everything on GPU if it fits |
| Speculative (0.5B draft) | +3% code, −29% prose | not worth it on one small GPU |
| N-gram speculative | 0–6% accept | not for fresh text |
| Prefix cache | 5.9 s → 0.9 s answer time | big win for fixed prompts |

The unsolved part was Step 2's finding: Q4 decode at only 41–50% of the bandwidth ceiling. Step 4
profiles the kernels to find where the time goes.

## Step 4: where does the time go? (profiling, 2026-11-13)

### 4a. CUDA graphs are switched off for Pascal; turning them on changes nothing

llama.cpp disables CUDA graphs for every GPU older than Volta (`cc < GGML_CUDA_CC_VOLTA` in
`ggml_cuda_graph_set_enabled`). `patches/01-cuda-graphs-pre-volta.patch` adds an opt-in env var
(`GGML_CUDA_FORCE_GRAPHS_PRE_VOLTA`) plus a counter that proves graphs really run (126 of 128 decode
steps replayed from a graph).

| model | graphs off | graphs on |
|---|---|---|
| 0.5B Q4_0 | 44.0 | 43.8 |
| 0.5B Q8_0 | 41.3 | 41.2 |
| 1.5B Q4_0 | 19.7 | 19.7 |

So kernel launch overhead is not the problem on this GPU.

### 4b. nvprof (works on Pascal; Nsight Compute does not) — 0.5B Q4_0, per decoded token

The GPU is busy for the whole 23.7 ms of each token, so the kernels themselves are slow.

| matvec (inferred from grid size) | bytes | time | bandwidth |
|---|---|---|---|
| FFN gate + up, **fused** (×24 layers) | 4.90 MB | 475 µs | **10.3 GB/s (28%)** |
| FFN down (×24) | 2.45 MB | 115 µs | 21.4 GB/s |
| attn q / out, fused with bias/add (×48) | 0.45 MB | 48 µs | 9.5 GB/s |
| LM head, Q8_0 (×1) | 144.6 MB | 4,654 µs | **31.1 GB/s (85%)** |

The fused gate+up kernel alone takes 11.4 ms per token, about half of the token time.

### 4c. Turning kernel fusion off: +20% to +36% with identical output

llama.cpp already has a switch, `GGML_CUDA_DISABLE_FUSION=1`.

| model | fusion on (default) | **fusion off** | gain | % of ceiling (off) |
|---|---|---|---|---|
| 0.5B Q4_0 | 44.0 | **59.7** | **+36%** | 57% |
| 0.5B Q8_0 | 41.3 | **49.4** | +20% | 71% |
| 1.5B Q4_0 | 19.7 | **25.8** | **+31%** | 66% |

- Greedy output is **byte-for-byte identical** with fusion on and off (code and prose prompts, 1.5B Q4_0,
  128 tokens). Fusion changes no math, only how kernels are grouped.
- Unfused, gate and up are two kernels at 127 µs each (254 µs total) against 475 µs fused: the fused
  kernel is 1.87× slower than doing the work separately. The fused kernel uses 80 registers per thread
  against 66 unfused (nvprof trace). The root cause is not proven: `nvprof --metrics` fails with
  `CUDA profiling error` on driver 580 even as root (`scripts/counters.sh`), and Nsight Compute has
  no Pascal support.
- Reported upstream on 2026-11-13: https://github.com/ggml-org/llama.cpp/issues/32229
  (text in `results/upstream-issue-fusion-pascal.md`).

## Step 5: own Q4_0 matvec kernels (`kernels/q4mv.cu`, 2026-11-13)

Seven kernels, each checked against a CPU double-precision reference, timed over rotating copies of
the weights so the 256 KB L2 cannot help (DRAM numbers). Time includes quantizing x to int8 for the
DP4A kernels (v3–v7). % = share of the 36.5 GB/s ceiling. llama.cpp column = its `mul_mat_vec_q`
time per call inside a real decode (nvprof trace, `GGML_CUDA_DISABLE_FUSION=1`), so the two sides are
measured differently; treat the comparison as close, not exact.

| shape | v1_naive | v2_warp | v3_dp4a | v4_repack | v5_repack2 | v6_smem_gguf | v7_smem_gguf_2row | llama.cpp (in-situ, unfused) |
|---|---|---|---|---|---|---|---|---|
| 0.5B attn q/o   896x896 | 76.5 µs (16%) | 163.9 µs (8%) | 45.7 µs (27%) | 21.9 µs (56%) | 19.3 µs (64%) | 26.8 µs (46%) | 24.5 µs (50%) | 26.4 µs |
| 0.5B attn k/v   128x896 | 29.4 µs (6%) | 21.8 µs (8%) | 9.3 µs (19%) | 6.7 µs (26%) | 6.1 µs (29%) | 7.0 µs (25%) | 6.4 µs (27%) | 5.0 µs |
| 0.5B ffn up    4864x896 | 1221.6 µs (6%) | 738.4 µs (9%) | 208.8 µs (32%) | 95.3 µs (70%) | 79.4 µs (85%) | 124.1 µs (54%) | 110.2 µs (61%) | 127.0 µs |
| 0.5B ffn down  896x4864 | 364.4 µs (18%) | 741.4 µs (9%) | 117.9 µs (57%) | 76.5 µs (88%) | 75.0 µs (90%) | 104.9 µs (64%) | 97.2 µs (69%) | 106.1 µs |
| 1.5B attn q/o 1536x1536 | 270.8 µs (13%) | 403.5 µs (9%) | 64.3 µs (56%) | 49.8 µs (73%) | 47.3 µs (77%) | 66.6 µs (55%) | 65.0 µs (56%) | – |
| 1.5B ffn up   8960x1536 | 6121.4 µs (4%) | 2343.2 µs (9%) | 349.9 µs (61%) | 263.5 µs (80%) | 253.8 µs (84%) | 361.5 µs (59%) | 334.0 µs (64%) | – |
| 1.5B ffn down 1536x8960 | 1734.6 µs (12%) | 2404.5 µs (9%) | 370.2 µs (57%) | 226.8 µs (94%) | 225.0 µs (94%) | 314.3 µs (68%) | 282.8 µs (75%) | – |
| 0.5B LM head 151936x896 | 34273.9 µs (6%) | 23181.4 µs (9%) | 6398.4 µs (33%) | 2824.9 µs (74%) | 2318.6 µs (90%) | 3365.0 µs (62%) | 3285.2 µs (64%) | – |

per-layer matvec time, 0.5B: llama.cpp unfused 422.9 µs (in-situ)
  v5_repack2: 277.7 µs/layer → saves 3.49 ms/token → projected 75.4 tok/s (fusion-off measured 59.7)
  v7_smem_gguf_2row: 372.4 µs/layer → saves 1.21 ms/token → projected 64.4 tok/s (fusion-off measured 59.7)

| kernel | idea | registers |
|---|---|---|
| v1 naive | 1 thread per row, float math | 32 |
| v2 warp | 1 warp per row, float math, byte loads | 32 |
| v3 dp4a | x → int8 once; `__dp4a` does 4 int8 multiply-adds; GGUF layout, 2-byte loads | 32 |
| v4 repack | weights re-laid out: nibbles in an aligned array → one 16-byte load per block | 40 |
| v5 repack ×2 | v4 + 2 rows per warp, both rows' loads issued before the math | 40 |
| v6 smem | stock GGUF layout; warp copies 32 blocks with aligned 16-byte loads into shared memory | – |
| v7 smem ×2 | v6 + 2 rows per warp | – |

What it shows:

- **The memory layout is the biggest lever.** Same math (DP4A), only the layout changes: v3 on the
  18-byte GGUF blocks reaches 27–61%; v5 on an aligned layout reaches 84–94% on the big matrices.
  18-byte blocks force 2-byte loads; aligned blocks allow one 16-byte load per 32 weights.
- **DP4A matters too:** float math (v2) is stuck at ~9% whatever the shape; DP4A (v3) is 3–7× faster.
- **v2 is slower than naive** on most shapes: each lane does 16 one-byte loads per block, and Pascal
  does not keep plain global loads in L1, so every byte load goes to L2.
- **vs llama.cpp (unfused):** v5 is 1.3–1.6× faster on the 0.5B FFN and attention matrices; v7 (stock
  layout, could drop into ggml-cuda without repacking) is 1.08–1.15× faster. The tiny 128×896 k/v
  matvec is slower in our kernels (6.1 vs 5.0 µs): one warp per row leaves the 2 SMs half idle there.
- int8 x adds ~0.5% relative error (`rel_err` ≈ 5e-3), the same trade llama.cpp's own q8_1 path makes.

**Projected** 0.5B Q4_0 decode if llama.cpp's layer matvecs were replaced (LM head unchanged; not
measured end to end, this is arithmetic on the per-call times above):

| setup | per-layer matvec time | tok/s |
|---|---|---|
| llama.cpp default (fusion on), measured | – | 44.0 |
| llama.cpp fusion off, measured | 422.9 µs | 59.7 |
| + v7 kernels (stock layout), projected | 372.4 µs | ~64 |
| + v5 kernels (repacked layout), projected | 277.7 µs | ~75 (71% of the 105 tok/s ceiling) |

Next step to make it real: load-time repacking of Q4_0 weights in ggml-cuda for sm_61 and an
end-to-end tok/s + perplexity check.

## Step 6: the kernels inside llama.cpp, end to end (2026-11-13)

`patches/02-mx-q4-pascal-matvec.patch` adds `ggml/src/ggml-cuda/mx-q4.cu` and a hook in
`ggml_cuda_mul_mat_vec_q`: plain (unfused, non-MoE, one-column) Q4_0 matvecs go to our kernel when
`GGML_CUDA_MX_Q4` is set. It reads llama.cpp's own int8 (q8_1) copy of x, so nothing else changes.

- `smem`: v7 (stock layout through shared memory), no extra memory.
- `repack`: v5; on first use each Q4_0 weight gets an aligned copy (+0.89× of the Q4_0 weight size in
  VRAM: ~180 MB for 0.5B, ~650 MB for 1.5B; both still fit in 2 GB).

### Decode speed (tg128, 3 reps, FA on; all with `GGML_CUDA_DISABLE_FUSION=1` except the first column)

| model | llama.cpp default | fusion off | + `smem` | **+ `repack`** | ceiling | total gain |
|---|---|---|---|---|---|---|
| 0.5B Q4_0 | 44.0 | 60.1 | 65.1 | **75.0** (71%) | 105.4 | **1.70×** |
| 1.5B Q4_0 | 19.7 | 25.9 | 25.5 | **31.1** (79%) | 39.3 | **1.58×** |

The Step 5 projection for 0.5B was ~75 (repack) and ~64 (smem); measured 75.0 and 65.1.
`smem` does not help 1.5B (25.5 vs 25.9).

### Quality

Perplexity, wikitext-2 test, 8 chunks × 256 tokens, **ubatch 1** (forces the matvec path):

| model | llama.cpp (fusion on) | fusion off | smem | repack |
|---|---|---|---|---|
| 0.5B Q4_0 | 23.138 ± 2.335 | 23.138 | 23.102 | 23.102 ± 2.329 |
| 1.5B Q4_0 | 14.437 ± 1.329 | 14.437 | 14.412 | 14.412 ± 1.326 |

- No quality loss: the differences (−0.03) are far inside the error bars. They come from adding the
  same products in a different order. That the number moves at all also shows our kernel really ran.
- Greedy text, 1.5B, 128 tokens: the code prompt is identical word for word; the prose prompt is
  identical for 6 words, then picks a different near-tied word ("Earth's" vs "sun's") and continues
  with a different, equally fluent answer. Bit-exact output is not expected when the summation order
  changes.

## Step 7: root cause of the fusion slowdown, and a one-line fix (2026-11-13)

llama.cpp already turns mmvq fusion off for Pascal in `ggml_cuda_should_fuse_mul_mat_vec_q`
("fusion is not universally faster on Pascal"), but the guard is `cc <= GGML_CUDA_CC_PASCAL`, and
`GGML_CUDA_CC_PASCAL` is 600 (GP100 / Tesla P100). Consumer Pascal (GTX 10-series, MX150/230/250) is
compute capability **6.1 (610)**, so the guard never fires on them. Fix (`patches/03-fix-pascal-fusion-guard.patch`):
`cc < GGML_CUDA_CC_VOLTA`. Still present in upstream master on 2026-11-13; no open PR.

Same session, clean upstream b79696f with and without the one-line change (pp512 + tg128, 3 reps, FA on):

| model | upstream tg128 | **fixed tg128** | change | fixed + `GGML_CUDA_DISABLE_FUSION=1` | pp512 upstream → fixed |
|---|---|---|---|---|---|
| 0.5B Q4_0 | 47.7 | **67.8** | **+42%** | 67.3 | 1099 → 1108 |
| 0.5B Q8_0 | 46.8 | **60.2** | +29% | 59.6 | 1069 → 1071 |
| 0.5B Q4_K_M | 42.6 | **59.7** | **+40%** | 58.5 | 1038 → 1039 |
| 1.5B Q4_0 | 21.5 | **30.5** | **+42%** | 30.2 | 383 → 383 |

- The fix is slightly faster than disabling all fusion, because the other fusions (norms, adds) stay on.
- Absolute speeds in this session are ~8% higher than in Steps 2–6 (laptop power state); compare
  within a table, not across tables.
- `test-backend-ops test -b CUDA0 -o MUL_MAT_FUSION`: **1601/1601 passed** with the fix.

Re-verified on latest upstream master ded522f (the PR branch) with `scripts/pr_verify.sh`, same session:
0.5B Q4_0 43.9 → **60.7** (+38%), 0.5B Q8_0 41.6 → 50.1 (+20%), 0.5B Q4_K_M 39.8 → 53.5 (+34%),
1.5B Q4_0 19.8 → 26.1 (+32%); pp512 within ±1%; greedy output with the fix **identical** to upstream
(1.5B, code + prose, 128 tokens); `MUL_MAT_FUSION` 1601/1601 passed.
