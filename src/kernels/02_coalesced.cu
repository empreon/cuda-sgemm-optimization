#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define BLOCK_SIZE 32

__global__ void sgemm_coalesced(const float *a, const float *b, float *c, int n) {
    // Column (x axis)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Row (y axis)
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundry check
    if (col < n && row < n) {

        float sum = 0.0f;
        int index = (row * n) + col;

        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }

        c[index] = sum; 
    }
}

int run_coalesced(int N) {
    printf("\n=== RUNNING COALESCED SGEMM (N=%d) ===\n", N);

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

    // Grid & Block Config
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* Benchmark Start */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Running Naive SGEMM with Matrix Size %dx%d...\n", N, N);

    cudaEventRecord(start);

    // Kernel launch
    sgemm_coalesced<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop);

    // Wait for GPU
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    /* Benchmark Finish */

    // GFLOPS Calculation (2 * N^3 operations are performed: Multiplication + Addition)
    // We divided by 1e-3 because we are converting ms to seconds. We divided by 1e9 for Giga.
    double flops = 2.0 * N * N * N;
    double gflops = (flops / 1e9) / (milliseconds / 1000.0);

    printf("Time: %f ms\n", milliseconds);
    printf("Performance: %f GFLOPS\n", gflops);

    // Check for Error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    // Copy back
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Validation
    printf("Verifying check (0,0): %f\n", h_c[0]);
    printf("Verifying check (N-1,N-1): %f\n", h_c[(N*N)-1]);

    printf("Done\n");

    // Free Memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}