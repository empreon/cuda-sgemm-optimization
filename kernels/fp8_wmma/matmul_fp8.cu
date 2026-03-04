#include "common.cuh"

extern "C" __global__ void matmul_fp8_placeholder(const float* A,
                                                   const float* B,
                                                   float* C,
                                                   int M,
                                                   int K,
                                                   int N) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float value = 0.0f;
    for (int idx = 0; idx < K; ++idx) {
      value += A[ROW_MAJOR_INDEX(row, idx, K)] *
               B[ROW_MAJOR_INDEX(idx, col, N)];
    }
    C[ROW_MAJOR_INDEX(row, col, N)] = value;
  }
}
