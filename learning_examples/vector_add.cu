#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

__global__ void vectorAdd(int *a, int *b, int *c) {
    int i = threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    // Host memory
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);

    for(int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device memory
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy to Device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Kernel launch
    vectorAdd<<<1, N>>>(d_a, d_b, d_c);

    // Error checks
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Copy back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verification
    printf("Results:\n");
    for(int i = 0; i < N; i++) {
        printf("[%d] %d + %d = %d\n", i, h_a[i], h_b[i], h_c[i]);
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}