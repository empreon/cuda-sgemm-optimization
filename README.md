# CUDA SGEMM Optimization

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

This repository explores custom CUDA kernel development for high-performance computing. It implements progressively optimized single-precision matrix multiplication (SGEMM) kernels. The project includes a PyCUDA-based execution engine, a benchmark suite against cuBLAS, and a brute-force autotuner to find optimal tile sizes and block dimensions.

## Features

* **PyCUDA Engine:** A Python wrapper for compiling and launching CUDA kernels with memory pooling.
* **Progressive Optimizations:** Step-by-step kernel improvements targeting memory bandwidth, shared memory, and instruction-level parallelism.
* **cuBLAS Benchmarking:** Validates kernel output accuracy and compares performance metrics against NVIDIA's cuBLAS library.
* **Autotuner:** A brute-force tuning script that sweeps through tile dimensions and thread block configurations to identify the most efficient parameters.

## Kernel Implementations

The `matmul.cu` file contains the following kernel variants:

1. **Naive:** Standard nested loop implementation with global memory accesses.
2. **Tiled:** Utilizes shared memory to reduce global memory bandwidth requirements.
3. **Vectorized 2D:** Employs float4 vectorized loads and stores with 2D thread blocking.
4. **Pipelined 2D:** Implements double buffering in shared memory to overlap global memory fetches with computation.
5. **Asymmetric Pipelined 2D:** Allows independent tuning of M, N, and K tile dimensions.
6. **Register Pipelined 2D:** Maximizes instruction-level parallelism by pipelining at the register level and loading multiple vectors per thread.

## Requirements

* NumPy
* PyCUDA
* scikit-cuda (required for cuBLAS benchmarking)

## Usage

### Benchmarking against cuBLAS

Run the benchmark script to compare the active custom kernel against cuBLAS. You can specify the matrix dimensions via command-line arguments.

```bash
python bench_cublas.py --m 1024 --k 512 --n 2048
```

### Autotuning

Run the autotuner to find the best block configurations for all implemented kernels across various matrix sizes.

```bash
python autotune.py
```

The autotuner evaluates different configurations and safely skips those exceeding shared memory limits. It outputs JSON files containing the performance metrics (GFLOPS, latency, memory usage) for the tested configurations, sorted by performance.

## Project Structure

* `engine.py`: PyCUDA execution engine, compiler interface, and memory manager.
* `utils.py`: Profiling and GFLOPS calculation utilities.
* `matmul.cu`: CUDA source code containing all SGEMM implementations.
* `bench_cublas.py`: Benchmarking script comparing custom kernels to the cuBLAS baseline.
* `autotune.py`: Automated configuration tuning script.
