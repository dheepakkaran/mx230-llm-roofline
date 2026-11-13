// Memory bandwidth probe for the roofline ceiling.
// LLM decode reads every weight once per token, so a read-only streaming kernel
// is the honest ceiling (copy is also reported for reference).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

// Read-only: sum int4 vectors (16 B per load), write one value per block so the
// compiler cannot drop the loads.
__global__ void read_kernel(const int4* __restrict__ a, size_t n, int* out) {
  int acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    int4 v = __ldg(a + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  __shared__ int s[1024];
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (threadIdx.x < k) s[threadIdx.x] ^= s[threadIdx.x + k];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}

__global__ void copy_kernel(const int4* __restrict__ a, int4* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    b[i] = a[i];
}

int main(int argc, char** argv) {
  size_t mb = argc > 1 ? atoi(argv[1]) : 512;          // buffer size per array
  size_t bytes = mb << 20, n = bytes / sizeof(int4);
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  int memClkKHz = 0, busBits = 0;
  cudaDeviceGetAttribute(&memClkKHz, cudaDevAttrMemoryClockRate, 0);
  cudaDeviceGetAttribute(&busBits, cudaDevAttrGlobalMemoryBusWidth, 0);
  double peak = 2.0 * memClkKHz * 1e3 * (busBits / 8) / 1e9;  // DDR: 2 transfers/clock
  printf("device=%s sm_%d%d SMs=%d bus=%d-bit memclk=%d MHz spec_peak=%.1f GB/s\n",
         p.name, p.major, p.minor, p.multiProcessorCount, busBits, memClkKHz / 1000, peak);

  int4 *a, *b; int* out;
  CK(cudaMalloc(&a, bytes)); CK(cudaMalloc(&b, bytes));
  CK(cudaMemset(a, 1, bytes));
  int threads = 256, blocks = p.multiProcessorCount * 32;
  CK(cudaMalloc(&out, blocks * sizeof(int)));
  cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

  auto bench = [&](const char* name, bool copy) {
    std::vector<float> ms;
    for (int r = 0; r < 23; r++) {                       // 3 warm-up + 20 timed
      cudaEventRecord(t0);
      if (copy) copy_kernel<<<blocks, threads>>>(a, b, n);
      else      read_kernel<<<blocks, threads>>>(a, n, out);
      cudaEventRecord(t1); CK(cudaEventSynchronize(t1));
      float m; cudaEventElapsedTime(&m, t0, t1);
      if (r >= 3) ms.push_back(m);
    }
    std::sort(ms.begin(), ms.end());
    double moved = copy ? 2.0 * bytes : (double)bytes;
    double med = ms[ms.size() / 2], best = ms[0];
    printf("%-5s %zu MB: median %.2f GB/s  best %.2f GB/s  (%.0f%% of spec)\n", name, mb,
           moved / med / 1e6, moved / best / 1e6, 100.0 * moved / med / 1e6 / peak);
  };
  bench("read", false);
  bench("copy", true);
  return 0;
}
