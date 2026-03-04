#include "common.cuh"
#include <math.h>

extern "C" __global__ void fused_cbs_fp8_placeholder(const float* x,
                                                      const float* scale,
                                                      const float* bias,
                                                      float* y,
                                                      int channels,
                                                      int numel) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }

  const int c = idx % channels;
  const float normalized = x[idx] * scale[c] + bias[c];
  const float sigmoid = 1.0f / (1.0f + expf(-normalized));
  y[idx] = normalized * sigmoid;
}
