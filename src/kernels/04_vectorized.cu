#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define TILE_SIZE 32

__global__ void sgemm_vectorized(const float *A, const float *B, float *C, int n) {

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Thread and Block indexs
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x; // 8 thread
    int ty = threadIdx.y; // 32 thread

    // Rows and columns of C (Result) matrix
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + (tx * 4);

    float results[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int m = 0; m < n / TILE_SIZE; m++) {

        int A_idx = row * n + (m * TILE_SIZE + tx * 4);
        int B_idx = (m * TILE_SIZE + ty) * n + col;
        
        float4 tmpA = *reinterpret_cast<const float4*>(&A[A_idx]);
        float4 tmpB = *reinterpret_cast<const float4*>(&B[B_idx]);

        // Load from A
        As[ty][tx * 4 + 0] = tmpA.x;
        As[ty][tx * 4 + 1] = tmpA.y;
        As[ty][tx * 4 + 2] = tmpA.z;
        As[ty][tx * 4 + 3] = tmpA.w;

        // Load from B
        Bs[ty][tx * 4 + 0] = tmpB.x;
        Bs[ty][tx * 4 + 1] = tmpB.y;
        Bs[ty][tx * 4 + 2] = tmpB.z;
        Bs[ty][tx * 4 + 3] = tmpB.w;

        // Wait for each thread to carry 1 element.
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {

            // One value from A
            float a_val = As[ty][k];

            // Four sequential value from B
            float b_val0 = Bs[k][tx * 4 + 0];
            float b_val1 = Bs[k][tx * 4 + 1];
            float b_val2 = Bs[k][tx * 4 + 2];
            float b_val3 = Bs[k][tx * 4 + 3];

            results[0] += a_val * b_val0;
            results[1] += a_val * b_val1;
            results[2] += a_val * b_val2;
            results[3] += a_val * b_val3;
        }

        // Wait for all calculations to finish
        __syncthreads();
    }

    // Write the result to Global Memory
    int C_idx = row * n + col; // Target starting address

    float4 res_vec;
    res_vec.x = results[0];
    res_vec.y = results[1];
    res_vec.z = results[2];
    res_vec.w = results[3];

    *reinterpret_cast<float4*>(&C[C_idx]) = res_vec;
}

int run_vectorized(int N) {
    printf("\n=== RUNNING VECTORIZED SGEMM (N=%d) ===\n", N);
    
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
    dim3 blockSize(TILE_SIZE / 4, TILE_SIZE);
    dim3 gridSize(N / TILE_SIZE, N / TILE_SIZE);

    // --- Benchmark Start ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Running TILED SGEMM with Matrix Size %dx%d...\n", N, N);

    cudaEventRecord(start);
    
    // Kernel Launch
    sgemm_vectorized<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

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