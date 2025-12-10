#include <stdio.h>
#include <cuda_runtime.h>

#define N 2048 // 2048 x 2048 matris

__global__ void matrixAdd(int *a, int *b, int *c, int width) {
    // 2D Thread ID calculation
    // Column (x axis)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Row (y axis)
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundry check
    if (col < width && row < width) {
        // index = (row x width) + col
        int index = (row * width) + col;

        c[index] = a[index] + b[index];
    }
}

int main() {
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    size_t size = N * N * sizeof(int);

    // Host memory
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);

    for(int i = 0; i < N * N; i++) {
        h_a[i] = 1;
        h_b[i] = 2;
    }

    // Device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy to Device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Configuration (2D)
    dim3 threadsPerBlock(16, 16);

    // Grid dimension calculations
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    printf("Kernel Starting: Grid(%d, %d), Block(%d, %d)\n",
           numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // Kernel launch
    matrixAdd<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verification
    int testRow = 123, testCol = 456;
    int testIndex = testRow * N + testCol;

    if (h_c[testIndex] == 3) {
        printf("Success!\n");
    } else {
        printf("Error! Expected 3, found %d\n", h_c[testIndex]);
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}