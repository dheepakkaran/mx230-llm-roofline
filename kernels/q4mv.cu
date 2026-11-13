// q4mv.cu — Q4_0 matrix x vector kernels for the MX230 (sm_61), from naive to tuned.
//
// y[m] = W[m,k] · x[k], W stored in llama.cpp's Q4_0 format:
//   one block = 32 weights = fp16 scale d + 16 bytes; byte j holds weight j (low nibble)
//   and weight j+16 (high nibble); weight = (nibble - 8) * d.          18 bytes / 32 weights
//
// Kernels
//   v1 naive    one thread per row, dequantize to float, FMA
//   v2 warp     one warp (32 threads) per row, lanes take consecutive blocks, shuffle reduction
//   v3 dp4a     x quantized to int8 once; 4 lanes per block, __dp4a does 4 int8 multiply-adds
//   v4 repack   weights re-laid out offline: nibbles in an aligned array (one 16-byte load per
//               block) + scales in their own array; one lane per block, 8 dp4a per load
//   v5 repack2  v4 with 2 rows per warp (x loaded once, used twice) and 2 blocks in flight per lane
//   v6 smem     stock GGUF layout: warp copies 32 blocks with aligned 16-byte loads into shared memory
//   v7 smem x2  v6 with 2 rows per warp
//
// Every kernel is checked against a CPU double-precision reference. Timing rotates through
// several copies of W so the 512 KB L2 cannot serve repeated runs (we want DRAM numbers).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int QK = 32;
struct block_q4_0 { __half d; uint8_t qs[QK / 2]; };
static_assert(sizeof(block_q4_0) == 18, "Q4_0 block must be 18 bytes like llama.cpp");

// ---------------------------------------------------------------- warp helpers
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}
__device__ __forceinline__ int warp_isum(int v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// ---------------------------------------------------------------- v1 naive
__global__ void mv_naive(const block_q4_0* __restrict__ W, const float* __restrict__ x,
                         float* __restrict__ y, int m, int nb) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= m) return;
  const block_q4_0* row = W + (size_t)r * nb;
  float acc = 0.f;
  for (int b = 0; b < nb; b++) {
    float d = __half2float(row[b].d);
    const float* xb = x + b * QK;
    for (int j = 0; j < QK / 2; j++) {
      uint8_t q = row[b].qs[j];
      acc += d * ((int)(q & 15) - 8) * xb[j] + d * ((int)(q >> 4) - 8) * xb[j + 16];
    }
  }
  y[r] = acc;
}

// ---------------------------------------------------------------- v2 warp per row
__global__ void mv_warp(const block_q4_0* __restrict__ W, const float* __restrict__ x,
                        float* __restrict__ y, int m, int nb) {
  int row_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (row_id >= m) return;
  const block_q4_0* row = W + (size_t)row_id * nb;
  float acc = 0.f;
  for (int b = lane; b < nb; b += 32) {          // lanes read 32 neighbouring blocks: coalesced
    float d = __half2float(row[b].d);
    const float* xb = x + b * QK;
    for (int j = 0; j < QK / 2; j++) {
      uint8_t q = row[b].qs[j];
      acc += d * (((int)(q & 15) - 8) * xb[j] + ((int)(q >> 4) - 8) * xb[j + 16]);
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row_id] = acc;
}

// ---------------------------------------------------------------- x -> int8 (shared by v3-v5)
// One warp per 32-value block: d = max|x| / 127, q = round(x / d), s = d * sum(q).
// s lets the kernels apply the "-8" offset of Q4_0 once per block instead of per weight:
//   sum((q4 - 8) * d4 * q8 * d8) = d4 * (d8 * sum(q4 * q8) - 8 * s)
__global__ void quant_x(const float* __restrict__ x, int8_t* __restrict__ xq,
                        float* __restrict__ xd, float* __restrict__ xs, int nb) {
  int b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (b >= nb) return;
  float v = x[b * QK + lane];
  float amax = warp_max(fabsf(v));
  float d = amax / 127.f;
  int q = d > 0.f ? __float2int_rn(v / d) : 0;
  int s = warp_isum(q);
  xq[b * QK + lane] = (int8_t)q;
  if (lane == 0) { xd[b] = d; xs[b] = d * s; }
}

// ---------------------------------------------------------------- v3 dp4a on the GGUF layout
// 4 lanes share one block (each takes 4 of its 16 bytes), so a warp covers 8 blocks per step.
// The 18-byte block is only 2-byte aligned, so the 4 bytes are read as two 16-bit loads.
__global__ void mv_dp4a(const block_q4_0* __restrict__ W, const int8_t* __restrict__ xq,
                        const float* __restrict__ xd, const float* __restrict__ xs,
                        float* __restrict__ y, int m, int nb) {
  int row_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (row_id >= m) return;
  const block_q4_0* row = W + (size_t)row_id * nb;
  const int i = lane & 3;                        // which 4-byte slice of the block
  float acc = 0.f;
  for (int b = lane >> 2; b < nb; b += 8) {
    const uint16_t* p = (const uint16_t*)row[b].qs + 2 * i;
    int v = (int)p[0] | ((int)p[1] << 16);
    const int* xi = (const int*)(xq + b * QK);
    int lo = v & 0x0F0F0F0F, hi = (v >> 4) & 0x0F0F0F0F;   // weights 4i..4i+3 and 4i+16..4i+19
    int sumi = __dp4a(lo, xi[i], 0);
    sumi = __dp4a(hi, xi[i + 4], sumi);
    float d4 = __half2float(row[b].d);
    acc += d4 * (xd[b] * sumi - (i == 0 ? 8.f * xs[b] : 0.f));
  }
  acc = warp_sum(acc);
  if (lane == 0) y[row_id] = acc;
}

// ---------------------------------------------------------------- v4 repacked layout
// Wq: m*nb int4 (the 16 nibble bytes of each block, 16-byte aligned) ; Wd: m*nb half scales.
// One lane per block: one 128-bit load brings 32 weights, 8 dp4a consume them.
__device__ __forceinline__ float block_dot(int4 w, const int8_t* __restrict__ xq, float xd_b,
                                           float xs_b, float d4, int b) {
  const int4* xv = (const int4*)(xq + b * QK);
  int4 x0 = __ldg(xv), x1 = __ldg(xv + 1);       // x0: values 0..15, x1: values 16..31
  int s = 0;
  s = __dp4a(w.x & 0x0F0F0F0F, x0.x, s); s = __dp4a((w.x >> 4) & 0x0F0F0F0F, x1.x, s);
  s = __dp4a(w.y & 0x0F0F0F0F, x0.y, s); s = __dp4a((w.y >> 4) & 0x0F0F0F0F, x1.y, s);
  s = __dp4a(w.z & 0x0F0F0F0F, x0.z, s); s = __dp4a((w.z >> 4) & 0x0F0F0F0F, x1.z, s);
  s = __dp4a(w.w & 0x0F0F0F0F, x0.w, s); s = __dp4a((w.w >> 4) & 0x0F0F0F0F, x1.w, s);
  return d4 * (xd_b * s - 8.f * xs_b);
}

__global__ void mv_repack(const int4* __restrict__ Wq, const __half* __restrict__ Wd,
                          const int8_t* __restrict__ xq, const float* __restrict__ xd,
                          const float* __restrict__ xs, float* __restrict__ y, int m, int nb) {
  int row_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  if (row_id >= m) return;
  const int4* wq = Wq + (size_t)row_id * nb;
  const __half* wd = Wd + (size_t)row_id * nb;
  float acc = 0.f;
  for (int b = lane; b < nb; b += 32)
    acc += block_dot(__ldg(wq + b), xq, __ldg(xd + b), __ldg(xs + b), __half2float(wd[b]), b);
  acc = warp_sum(acc);
  if (lane == 0) y[row_id] = acc;
}

// ---------------------------------------------------------------- v5 repack, 2 rows/warp, 2 loads in flight
__global__ void mv_repack2(const int4* __restrict__ Wq, const __half* __restrict__ Wd,
                           const int8_t* __restrict__ xq, const float* __restrict__ xd,
                           const float* __restrict__ xs, float* __restrict__ y, int m, int nb) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
  int r0 = warp * 2;
  if (r0 >= m) return;
  bool has1 = r0 + 1 < m;
  const int4* w0 = Wq + (size_t)r0 * nb;  const __half* d0 = Wd + (size_t)r0 * nb;
  const int4* w1 = w0 + nb;                const __half* d1 = d0 + nb;
  float a0 = 0.f, a1 = 0.f;
  for (int b = lane; b < nb; b += 32) {
    int4 q0 = __ldg(w0 + b);                     // both loads issued before any math
    int4 q1 = has1 ? __ldg(w1 + b) : make_int4(0, 0, 0, 0);
    float xdb = __ldg(xd + b), xsb = __ldg(xs + b);
    a0 += block_dot(q0, xq, xdb, xsb, __half2float(d0[b]), b);
    if (has1) a1 += block_dot(q1, xq, xdb, xsb, __half2float(d1[b]), b);
  }
  a0 = warp_sum(a0); a1 = warp_sum(a1);
  if (lane == 0) { y[r0] = a0; if (has1) y[r0 + 1] = a1; }
}

// ---------------------------------------------------------------- host side
// ---------------------------------------------------------------- v6/v7 stock GGUF layout, staged through shared memory
// The 18-byte blocks stop us from doing 16-byte loads straight into registers. But 32 neighbouring
// blocks are one contiguous 576-byte run, so the warp copies that run with aligned 16-byte loads
// (rounded out to 16-byte edges) into shared memory, then each lane picks its block from there.
// Same file format as llama.cpp, so this kernel could drop into ggml-cuda without repacking.
// ROWS rows per warp: x values and scales are loaded once and reused ROWS times.
// Needs up to 15 bytes of readable padding after W (llama.cpp pads tensor rows anyway).
template <int ROWS>
__global__ void mv_smem(const block_q4_0* __restrict__ W, const int8_t* __restrict__ xq,
                        const float* __restrict__ xd, const float* __restrict__ xs,
                        float* __restrict__ y, int m, int nb) {
  __shared__ int4 sbuf[4][ROWS][38];              // 4 warps per block, 38*16 = 608 >= 576 + 30
  const int w = threadIdx.x / 32, lane = threadIdx.x & 31;
  const int r0 = (blockIdx.x * 4 + w) * ROWS;
  if (r0 >= m) return;
  float acc[ROWS];
  #pragma unroll
  for (int r = 0; r < ROWS; r++) acc[r] = 0.f;
  for (int b0 = 0; b0 < nb; b0 += 32) {
    const int nblk = min(32, nb - b0);
    int off[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {               // issue all rows' loads first
      if (r0 + r >= m) { off[r] = 0; continue; }
      const char* s = (const char*)(W + (size_t)(r0 + r) * nb + b0);
      const uintptr_t sa = (uintptr_t)s & ~(uintptr_t)15;
      const uintptr_t ea = ((uintptr_t)(s + nblk * 18) + 15) & ~(uintptr_t)15;
      const int n16 = (int)((ea - sa) / 16);
      off[r] = (int)((uintptr_t)s - sa);
      for (int t = lane; t < n16; t += 32) sbuf[w][r][t] = __ldg((const int4*)sa + t);
    }
    __syncwarp();
    if (lane < nblk) {
      const int b = b0 + lane;
      const float xdb = __ldg(xd + b), xsb = __ldg(xs + b);
      #pragma unroll
      for (int r = 0; r < ROWS; r++) {
        if (r0 + r >= m) continue;
        const char* bp = (const char*)sbuf[w][r] + off[r] + lane * 18;   // 2-byte aligned
        const uint16_t* q16 = (const uint16_t*)(bp + 2);
        int4 wv;
        wv.x = q16[0] | (q16[1] << 16); wv.y = q16[2] | (q16[3] << 16);
        wv.z = q16[4] | (q16[5] << 16); wv.w = q16[6] | (q16[7] << 16);
        acc[r] += block_dot(wv, xq, xdb, xsb, __half2float(*(const __half*)bp), b);
      }
    }
    __syncwarp();
  }
  #pragma unroll
  for (int r = 0; r < ROWS; r++) {
    float a = warp_sum(acc[r]);
    if (lane == 0 && r0 + r < m) y[r0 + r] = a;
  }
}

static float half_to_float_host(__half h) { return __half2float(h); }

// llama.cpp's Q4_0 quantizer (quantize_row_q4_0_ref)
static void quantize_q4_0(const float* x, block_q4_0* out, int n) {
  for (int b = 0; b < n / QK; b++) {
    float amax = 0, maxv = 0;
    for (int j = 0; j < QK; j++) { float v = x[b * QK + j]; if (fabsf(v) > amax) { amax = fabsf(v); maxv = v; } }
    float d = maxv / -8.f, id = d ? 1.f / d : 0.f;
    out[b].d = __float2half(d);
    for (int j = 0; j < QK / 2; j++) {
      int q0 = std::min(15, (int)(x[b * QK + j] * id + 8.5f));
      int q1 = std::min(15, (int)(x[b * QK + j + 16] * id + 8.5f));
      out[b].qs[j] = (uint8_t)(q0 | (q1 << 4));
    }
  }
}

struct Shape { const char* name; int m, k; double llama_us; };   // llama_us: in-situ nvprof, unfused

int main(int argc, char** argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 200;
  // Qwen2.5-0.5B / 1.5B decode matvecs. llama_us = llama.cpp mul_mat_vec_q time per call from
  // nvprof traces with GGML_CUDA_DISABLE_FUSION=1 (0.5B only; 0 = not measured).
  std::vector<Shape> shapes = {
    {"0.5B attn q/o   896x896",    896,   896,  26.4},
    {"0.5B attn k/v   128x896",    128,   896,   5.0},
    {"0.5B ffn up    4864x896",   4864,   896, 127.0},
    {"0.5B ffn down  896x4864",    896,  4864, 106.1},
    {"1.5B attn q/o 1536x1536",   1536,  1536,   0},
    {"1.5B ffn up   8960x1536",   8960,  1536,   0},
    {"1.5B ffn down 1536x8960",   1536,  8960,   0},
    {"0.5B LM head 151936x896", 151936,   896,   0},
  };
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("# %s sm_%d%d, %d SMs, L2 %d KB, reps %d\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount, prop.l2CacheSize / 1024, reps);
  printf("shape,kernel,us_per_call,quant_x_us,GBps_weights,pct_of_36.5,rel_err,llama_us\n");

  std::mt19937 rng(42); std::normal_distribution<float> nd(0.f, 1.f);
  cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

  for (auto& S : shapes) {
    const int m = S.m, k = S.k, nb = k / QK;
    const size_t nblk = (size_t)m * nb;
    const double wbytes = nblk * 18.0;
    // copies so that all copies together are >= 64 MB (defeats the 512 KB L2), at most 32
    int copies = std::max(1, std::min(32, (int)ceil(64e6 / wbytes)));

    std::vector<float> wf((size_t)m * k), x(k);
    for (auto& v : wf) v = nd(rng) * 0.05f;
    for (auto& v : x) v = nd(rng);
    std::vector<block_q4_0> wq(nblk);
    for (int r = 0; r < m; r++) quantize_q4_0(&wf[(size_t)r * k], &wq[(size_t)r * nb], k);

    // reference on the dequantized weights, in double
    std::vector<double> yref(m);
    for (int r = 0; r < m; r++) {
      double acc = 0;
      for (int b = 0; b < nb; b++) {
        const block_q4_0& B = wq[(size_t)r * nb + b];
        double d = half_to_float_host(B.d);
        for (int j = 0; j < 16; j++) {
          acc += d * ((B.qs[j] & 15) - 8) * x[b * QK + j];
          acc += d * ((B.qs[j] >> 4) - 8) * x[b * QK + j + 16];
        }
      }
      yref[r] = acc;
    }
    // repacked copy
    std::vector<int4> rq(nblk); std::vector<__half> rd(nblk);
    for (size_t i = 0; i < nblk; i++) { memcpy(&rq[i], wq[i].qs, 16); rd[i] = wq[i].d; }

    block_q4_0* dW; int4* dRq; __half* dRd; float *dx, *dy, *dxd, *dxs; int8_t* dxq;
    CK(cudaMalloc(&dW, nblk * 18 * copies + 64));   // +64: padding for v6/v7 rounded loads
    CK(cudaMalloc(&dRq, nblk * 16 * copies));
    CK(cudaMalloc(&dRd, nblk * 2 * copies));
    for (int c = 0; c < copies; c++) {
      CK(cudaMemcpy(dW + c * nblk, wq.data(), nblk * 18, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dRq + c * nblk, rq.data(), nblk * 16, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dRd + c * nblk, rd.data(), nblk * 2, cudaMemcpyHostToDevice));
    }
    CK(cudaMalloc(&dx, k * 4)); CK(cudaMalloc(&dy, m * 4));
    CK(cudaMalloc(&dxq, k)); CK(cudaMalloc(&dxd, nb * 4)); CK(cudaMalloc(&dxs, nb * 4));
    CK(cudaMemcpy(dx, x.data(), k * 4, cudaMemcpyHostToDevice));

    auto check = [&]() {
      std::vector<float> y(m); CK(cudaMemcpy(y.data(), dy, m * 4, cudaMemcpyDeviceToHost));
      double num = 0, den = 0;
      for (int r = 0; r < m; r++) { num += (y[r] - yref[r]) * (y[r] - yref[r]); den += yref[r] * yref[r]; }
      return sqrt(num / den);
    };
    auto timeit = [&](auto launch) {
      for (int i = 0; i < 5; i++) launch(i % copies);
      CK(cudaDeviceSynchronize());
      cudaEventRecord(t0);
      for (int i = 0; i < reps; i++) launch(i % copies);
      cudaEventRecord(t1); CK(cudaEventSynchronize(t1));
      float ms; cudaEventElapsedTime(&ms, t0, t1);
      return 1e3 * ms / reps;
    };
    const int T = 128;                              // threads per block (4 warps)
    auto grid_rows = [&](int rows_per_block) { return (m + rows_per_block - 1) / rows_per_block; };

    double qx_us = timeit([&](int) { quant_x<<<(nb + 3) / 4, T>>>(dx, dxq, dxd, dxs, nb); });

    auto report = [&](const char* name, double us, bool uses_qx) {
      double tot = us + (uses_qx ? qx_us : 0);
      double gbps = wbytes / tot / 1e3;
      printf("%s,%s,%.2f,%.2f,%.2f,%.1f,%.2e,%.1f\n", S.name, name, us, uses_qx ? qx_us : 0.0, gbps,
             100 * gbps / 36.5, check(), S.llama_us);
      fflush(stdout);
    };

    // naive is very slow on the LM head; fewer reps there
    int saved = reps; if (m > 100000) reps = std::max(5, reps / 20);
    report("v1_naive", timeit([&](int c) { mv_naive<<<(m + T - 1) / T, T>>>(dW + c * nblk, dx, dy, m, nb); }), false);
    reps = saved;
    report("v2_warp", timeit([&](int c) { mv_warp<<<grid_rows(T / 32), T>>>(dW + c * nblk, dx, dy, m, nb); }), false);
    quant_x<<<(nb + 3) / 4, T>>>(dx, dxq, dxd, dxs, nb);
    report("v3_dp4a", timeit([&](int c) { mv_dp4a<<<grid_rows(T / 32), T>>>(dW + c * nblk, dxq, dxd, dxs, dy, m, nb); }), true);
    report("v4_repack", timeit([&](int c) { mv_repack<<<grid_rows(T / 32), T>>>(dRq + c * nblk, dRd + c * nblk, dxq, dxd, dxs, dy, m, nb); }), true);
    report("v5_repack2", timeit([&](int c) { mv_repack2<<<grid_rows(2 * T / 32), T>>>(dRq + c * nblk, dRd + c * nblk, dxq, dxd, dxs, dy, m, nb); }), true);
    report("v6_smem_gguf", timeit([&](int c) { mv_smem<1><<<grid_rows(4), T>>>(dW + c * nblk, dxq, dxd, dxs, dy, m, nb); }), true);
    report("v7_smem_gguf_2row", timeit([&](int c) { mv_smem<2><<<grid_rows(8), T>>>(dW + c * nblk, dxq, dxd, dxs, dy, m, nb); }), true);

    cudaFree(dW); cudaFree(dRq); cudaFree(dRd); cudaFree(dx); cudaFree(dy);
    cudaFree(dxq); cudaFree(dxd); cudaFree(dxs);
  }
  return 0;
}
