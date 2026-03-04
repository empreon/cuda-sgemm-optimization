from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import Engine
from src.utils import gemm_gflops, gpu_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark active FP32 CUDA matmul kernel entrypoint."
    )
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    m, k, n = args.m, args.k, args.n

    a_host = np.random.randn(m, k).astype(np.float32)
    b_host = np.random.randn(k, n).astype(np.float32)
    c_ref = a_host @ b_host

    engine = Engine()
    a_gpu = engine.upload(a_host, name="A")
    b_gpu = engine.upload(b_host, name="B")
    c_gpu = engine.alloc("C", m * n * np.dtype(np.float32).itemsize, reuse=False)

    block = (Engine.VEC_TILE // Engine.VEC_WIDTH, Engine.VEC_TILE // Engine.VBLOCK_ROWS, 1)
    stats = gpu_benchmark(
        lambda: engine.matmul(a_gpu, b_gpu, c_gpu, m=m, k=k, n=n, block=block),
        warmup=args.warmup,
        repeat=args.repeat,
    )

    c_host = engine.download(c_gpu, shape=(m, n), dtype=np.float32)
    max_abs_diff = float(np.max(np.abs(c_host - c_ref)))
    is_close = bool(np.allclose(c_host, c_ref, atol=args.atol))

    print("=== Active Custom Kernel Benchmark ===")
    print(f"Shapes: A=({m}, {k}), B=({k}, {n}), C=({m}, {n})")
    print(f"block={block}")
    print(f"mean_ms={float(stats['mean_ms']):.4f}")
    print(f"gflops={gemm_gflops(m, k, n, float(stats['mean_ms'])):.2f}")
    print(f"allclose={is_close}")
    print(f"max_abs_diff={max_abs_diff:.6e}")


if __name__ == "__main__":
    main()
