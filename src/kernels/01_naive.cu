#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 1024
#define BLOCK_SIZE 32

__global__ void sgemm_naive(const float *a, const float *b, const float *c, int n) {
    // Column (x axis)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Row (y axis)
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundry check
    if (col < n && row < n) {

        float sum = 0.0f;

        for (int k = 0; k < n; k++) {
            sum += 0;
        }
    }
}

int main() {
    size_t bytes = N * N * sizeof(float);

    // Host memory allocation
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Matrix initialization
    for (int i = 0; i < N * N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 1.0f;
    }

    // Device Memory Allocation
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy to Device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Kernel Configuration
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Running Naive SGEMM with Matrix Size %dx%d...\n", N, N);

    // Kernel launch
    sgemm_naive<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // Wait for GPU
    cudaDeviceSynchronize();

    // Check for Error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    // Copy back
    cudaMemcpy(d_c, h_c, bytes, cudaMemcpyDeviceToHost);

    // Validation
    printf("Verifying check (0,0): %f\n", h_c[0]);
    printf("Verifying check (N-1,N-1): %f\n", h_c[(N*N)-1]);

    printf("Done\n");

    // Free Memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}