from __future__ import annotations

import json
import itertools
from pathlib import Path
import sys
import time

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import Engine
from src.utils import gpu_benchmark, gemm_gflops

def ceil_div(a, b):
    return (a + b - 1) // b

def generate_test_matrices(num_tests=10):
    np.random.seed(42)
    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (512, 1024, 2048),
        (2048, 1024, 512),
        (1024, 512, 4096),
        (4096, 512, 1024),
        (1536, 1536, 1536),
        (2560, 2560, 2560)
    ]
    
    matrices = []
    for m, k, n in sizes:
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        matrices.append((m, k, n, a, b))
    return matrices

def benchmark_cublas(handle, m, k, n, a_gpu, b_gpu, c_gpu):
    import skcuda.cublas as cublas
    alpha = np.float32(1.0)
    beta = np.float32(0.0)
    
    def fn():
        cublas.cublasSgemm(
            handle, "n", "n", n, m, k, alpha, 
            int(b_gpu), n, int(a_gpu), k, beta, int(c_gpu), n
        )
        
    res = gpu_benchmark(fn, warmup=3, repeat=15)
    mean_ms = res["mean_ms"]
    gflops = gemm_gflops(m, k, n, mean_ms)
    return mean_ms, gflops

def run_autotune():
    MAX_SHARED_MEM_BYTES = 99 * 1024 
    VEC_WIDTH = 4

    device = cuda.Device(0)
    HW_WARP_SIZE = device.get_attribute(cuda.device_attribute.WARP_SIZE)
    
    tile_sizes = [32, 64, 128]
    asym_m_opts = [64, 128]
    asym_n_opts = [64, 128]
    asym_k_opts = [32, 64]
    
    vblock_rows_options = [4, 8, 16]
    
    kernels = [
        "run_matmul_naive", 
        "run_matmul_tiled", 
        "run_matmul_vectorized_2d", 
        "run_matmul_pipelined_2d",
        "run_matmul_asymetric_pipelined_2d",
        "run_matmul_register_pipelined_2d"
    ]
    
    matrices = generate_test_matrices()
    engine = Engine()
    results = {}
    
    TEST_TAG = "results"

    try:
        import skcuda.cublas as cublas
        CUBLAS_AVAILABLE = True
    except ImportError:
        CUBLAS_AVAILABLE = False
        print("\n[Error] skcuda.cublas not found. Please run `pip install scikit-cuda`.\n")

    if CUBLAS_AVAILABLE:
        print("\n--- Tuning Kernel: NVIDIA cuBLAS (Baseline) ---")
        handle = cublas.cublasCreate()
        cublas_gflops = []
        cublas_ms = []
        try:
            for m, k, n, a, b in matrices:
                try:
                    a_gpu = engine.upload(a)
                    b_gpu = engine.upload(b)
                    c_gpu = engine.alloc("c", m * n * 4, reuse=True)
                    mean_ms, gflops = benchmark_cublas(handle, m, k, n, a_gpu, b_gpu, c_gpu)
                    cublas_ms.append(mean_ms)
                    cublas_gflops.append(gflops)
                    time.sleep(0.1) # GPU cooling delay between tests
                except cuda.MemoryError:
                    print(f" [cuBLAS Skip] Insufficient GPU Memory - Matrix: {m}x{k}x{n}")
                    break
        finally:
            cublas.cublasDestroy(handle)
        
        if cublas_gflops:
            avg_cublas_gflops = sum(cublas_gflops) / len(cublas_gflops)
            avg_cublas_ms = sum(cublas_ms) / len(cublas_ms)
            results["cuBLAS_Baseline"] = {
                "avg_gflops": avg_cublas_gflops,
                "avg_ms": avg_cublas_ms
            }
            print(f" -> cuBLAS Avg GFLOPS: {avg_cublas_gflops:.2f} | Avg Time: {avg_cublas_ms:.2f} ms")
            
            cublas_filename = f"fp32_cuBLAS_{TEST_TAG}.json"
            with open(cublas_filename, "w") as f:
                json.dump(results["cuBLAS_Baseline"], f, indent=4)

    for kernel_name in kernels:
        results[kernel_name] = []
        print(f"\n--- Tuning Kernel: {kernel_name} ---")
        
        if "asymetric" in kernel_name or "register" in kernel_name:
            configs = list(itertools.product(asym_m_opts, asym_n_opts, asym_k_opts, vblock_rows_options))
        else:
            configs = [(t, t, t, v) for t, v in itertools.product(tile_sizes, vblock_rows_options)]
        
        for tile_m, tile_n, tile_k, vblock_rows in configs:
            
            if "naive" in kernel_name or "tiled" in kernel_name:
                block_dim = (tile_m, tile_n, 1)
                num_threads = tile_m * tile_n
            else:
                if "register" in kernel_name:
                    if tile_n % (VEC_WIDTH * 2) != 0 or tile_m % vblock_rows != 0: continue
                    threads_x = tile_n // (VEC_WIDTH * 2)
                else:
                    if tile_n % VEC_WIDTH != 0 or tile_m % vblock_rows != 0: continue
                    threads_x = tile_n // VEC_WIDTH
                    
                threads_y = tile_m // vblock_rows
                block_dim = (threads_x, threads_y, 1)
                num_threads = threads_x * threads_y
            
            if num_threads > 1024:
                continue
                
            if "naive" in kernel_name:
                estimated_shared_mem = 0
            elif "asymetric" in kernel_name or "register" in kernel_name:
                estimated_shared_mem = (2 * tile_m * (tile_k + 1) + 2 * tile_k * (tile_n + 1)) * 4
            elif "pipelined" in kernel_name:
                estimated_shared_mem = 2 * 2 * (tile_m * (tile_n + 1)) * 4
            elif "vectorized" in kernel_name:
                estimated_shared_mem = 2 * (tile_m * (tile_n + 1)) * 4
            elif "tiled" in kernel_name:
                estimated_shared_mem = 2 * (tile_m * tile_n) * 4
            else:
                estimated_shared_mem = 0

            if estimated_shared_mem > MAX_SHARED_MEM_BYTES:
                print(f"Skipping Config -> M:{tile_m} N:{tile_n} K:{tile_k}, VBLOCK:{vblock_rows} (Requires {estimated_shared_mem/1024:.1f}KB Shared Mem)")
                continue
                
            print(f"Testing Config -> M:{tile_m} N:{tile_n} K:{tile_k}, VBLOCK:{vblock_rows}, BlockDim:{block_dim}")
            
            engine.VEC_TILE = tile_m
            engine.VEC_TILE_M = tile_m
            engine.VEC_TILE_N = tile_n
            engine.VEC_TILE_K = tile_k
            engine.VBLOCK_ROWS = vblock_rows
            
            engine._modules.clear()
            engine._kernel_cache.clear()
            
            try:
                func = engine.get_kernel(kernel_name, module_name="fp32/matmul.cu")

                from pycuda import driver as cuda_driver
                func.set_attribute(cuda_driver.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, MAX_SHARED_MEM_BYTES)
                
                regs_per_thread = func.num_regs
                local_mem_spill = func.local_size_bytes
                static_shared = func.shared_size_bytes
                
            except Exception as e:
                print(f"Compilation failed: {e}")
                continue

            config_gflops = []
            config_ms = []
            
            for m, k, n, a, b in matrices:
                if m % tile_m != 0 or n % tile_n != 0 or k % tile_k != 0:
                    continue

                try:
                    a_gpu = engine.upload(a)
                    b_gpu = engine.upload(b)
                    c_gpu = engine.alloc("c", m * n * 4, reuse=True)
                    
                    grid_dim = (ceil_div(n, tile_n), ceil_div(m, tile_m), 1)

                    def benchmark_fn():
                        if "naive" in kernel_name: func(a_gpu, b_gpu, c_gpu, np.int32(m), np.int32(k), np.int32(n), block=block_dim, grid=grid_dim)
                        else: func(a_gpu, b_gpu, c_gpu, np.int32(m), np.int32(k), np.int32(n), block=block_dim, grid=grid_dim, shared=int(estimated_shared_mem))

                    bench_res = gpu_benchmark(benchmark_fn, warmup=3, repeat=15)
                    mean_ms = bench_res["mean_ms"]
                    gflops = gemm_gflops(m, k, n, mean_ms)
                    config_gflops.append(gflops)
                    config_ms.append(mean_ms)
                    
                    time.sleep(0.15) # GPU cooling delay between tests

                except cuda.MemoryError:
                    print(f"    [ERROR] Insufficient GPU Memory (OOM) - Matrix: {m}x{k}x{n}. Skipping...")
                    break
                except cuda.LogicError as e:
                    print(f"    [ERROR] CUDA Limit Violation (Invalid Argument etc.): {e}. Skipping...")
                    break 
                except Exception as e:
                    print(f"    [ERROR] Unexpected Runtime Error: {e}. Skipping...")
                    break
                    
            if config_gflops:
                avg_gflops = sum(config_gflops) / len(config_gflops)
                avg_ms = sum(config_ms) / len(config_ms)
                
                results[kernel_name].append({
                    "VEC_TILE_M": tile_m,
                    "VEC_TILE_N": tile_n,
                    "VEC_TILE_K": tile_k,
                    "VBLOCK_ROWS": vblock_rows,
                    "block_dim": block_dim,
                    "avg_gflops": avg_gflops,
                    "avg_ms": avg_ms,
                    "academics": {
                        "threads_per_block": num_threads,
                        "warps_per_block": ceil_div(num_threads, HW_WARP_SIZE),
                        "registers_per_thread": regs_per_thread,
                        "local_mem_spill_bytes": local_mem_spill, # if >0 then there is register spill
                        "static_shared_mem_bytes": static_shared,
                        "dynamic_shared_mem_bytes": estimated_shared_mem,
                        "total_shared_mem_bytes": static_shared + estimated_shared_mem
                    }
                })
                print(f" -> Avg GFLOPS: {avg_gflops:.2f} | Regs: {regs_per_thread} | Spill: {local_mem_spill}B")

        if results[kernel_name]:
            results[kernel_name] = sorted(results[kernel_name], key=lambda x: x["avg_gflops"], reverse=True)
            kernel_filename = f"fp32_{kernel_name}_{TEST_TAG}.json"
            
            with open(kernel_filename, "w") as f:
                json.dump(results[kernel_name], f, indent=4)
            print(f"[*] {kernel_name} tuning results saved to {kernel_filename}.\n")

    print("\nTuning completed. All kernels saved to their respective .json files.")

if __name__ == "__main__":
    run_autotune()