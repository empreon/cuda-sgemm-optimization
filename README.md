# CUDA SGEMM Optimization

This repository contains various implementations of Single Precision General Matrix Multiplication (SGEMM) using CUDA, optimized step-by-step.

## Structure

- `src/kernels/`: Contains different kernel implementations.
    - `01_naive.cu`: Naive implementation (Global Memory).
    - `02_coalesced.cu`: Memory coalescing improvements.
    - `03_tiling.cu`: Shared Memory Tiling.
    - `04_vectorized.cu`: Float4 vectorization.
- `src/utils.cu`: Utility functions.
- `src/main.cu`: Main benchmark loop.
- `scripts/plot_benchmark.py`: Script to plot benchmark results.
- `learning_examples/`: Folder for learning examples.

## Building

```bash
mkdir build
cd build
cmake ..
cmake --build .
```
