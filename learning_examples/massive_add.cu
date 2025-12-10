#include <stdio.h>
#include <cuda_runtime.h>

// Some logic with vector_add but with big numbers
#define N 100000

__global__ void vectorAdd(int *a, int *b, int *c) {
    // Global ID calculation: i = (block ID * block size) + thread ID
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundry check
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    size_t size = N * sizeof(int);

    // Host memory
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);

    for(int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // Device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy to Device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Configuration
    int threadsPerBlock = 256;

    // Ceiling rounding formula: (Sum + Divisor - 1) / Divisor
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Kernel Starting: %d Block, each block has %d Thread.\n", blocksPerGrid, threadsPerBlock);

    // Kernel launch
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

    // Error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verification
    bool success = true;
    for(int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("ERROR! index %d: %d + %d != %d\n", i, h_a[i], h_b[i], h_c[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Success!\n");
        printf("Last result: %d + %d = %d\n", h_a[N-1], h_b[N-1], h_c[N-1]);
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}