#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define TILE_SIZE 32

__global__ void sgemm_tiled(const float *A, const float *B, float *C, int n) {
    // Shared memory initialization (Same speed with L1 Cache)
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Thread and Block indexs
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    // Rows and columns of C (Result) matrix
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    // m: Which tile we are on?
    for (int m = 0; m < n / TILE_SIZE; ++m) {

        // --- PHASE 1: Load Data to Shared Memory ---
        // Each thread get 1 element from Global and put it in Shared memory.
        
        // Load from A
        As[ty][tx] = A[row * n + (m * TILE_SIZE + tx)];

        // Load from B
        Bs[ty][tx] = B[(m * TILE_SIZE + ty) * n + col];

        // Wait for each thread to carry 1 element.
        __syncthreads();

        // --- PHASE 2: Compute using Shared Memory ---
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Wait for all calculations to finish
        __syncthreads();
    }

    // Write the result to Global Memory
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int run_tiling(int N) {
    printf("\n=== RUNNING NAIVE SGEMM (N=%d) ===\n", N);
    
    size_t bytes = N * N * sizeof(float);

    // Host Memory Allocation
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Matrix initialization
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // Device Memory Allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Grid & Block Config
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // --- Benchmark Start ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Running TILED SGEMM with Matrix Size %dx%d...\n", N, N);

    cudaEventRecord(start);
    
    // Kernel Launch
    sgemm_tiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // --- Benchmark End ---

    // GFLOPS Calculation
    double flops = 2.0 * (double)N * (double)N * (double)N;
    double gflops = (flops / 1e9) / (milliseconds / 1000.0);

    printf("Time: %f ms\n", milliseconds);
    printf("Performance: %f GFLOPS\n", gflops);

    // Error Check
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
    }

    // Validation
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("Verifying check (0,0): %f\n", h_C[0]);
    printf("Verifying check (N-1,N-1): %f\n", h_C[(N*N)-1]);

    printf("Done\n");

    // Free
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}