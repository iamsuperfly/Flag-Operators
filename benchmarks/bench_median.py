"""
Performance benchmark: optimized median vs torch.median baseline.

Run with:
    python benchmarks/bench_median.py
"""

import sys
import os
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from operators.median import median
from benchmarks.bench_utils import benchmark_op

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def make_baseline_global(dtype):
    def fn(size):
        t = torch.randn(size, dtype=dtype, device=DEVICE)
        return torch.median(t)
    return fn


def make_optimized_global(dtype):
    def fn(size):
        t = torch.randn(size, dtype=dtype, device=DEVICE)
        return median(t)
    return fn


def make_baseline_dim(shape, dim, dtype):
    def fn(_):
        t = torch.randn(*shape, dtype=dtype, device=DEVICE)
        return torch.median(t, dim=dim)
    return fn


def make_optimized_dim(shape, dim, dtype):
    def fn(_):
        t = torch.randn(*shape, dtype=dtype, device=DEVICE)
        return median(t, dim=dim)
    return fn


# ---------------------------------------------------------------------------
# Benchmark 1: Global median across sizes
# ---------------------------------------------------------------------------

def bench_global_median():
    print("\n" + "=" * 60)
    print("BENCHMARK: Global median (no dim), float32")
    print("=" * 60)

    sizes = [256, 1024, 4096, 16384, 65536, 262144, 1048576]
    dtype = torch.float32

    results = benchmark_op(
        name="median_global",
        baseline_fn=make_baseline_global(dtype),
        optimized_fn=make_optimized_global(dtype),
        input_sizes=sizes,
    )
    return results


# ---------------------------------------------------------------------------
# Benchmark 2: Dim-reduction median (small N — in-register sort path)
# ---------------------------------------------------------------------------

def bench_dim_small_n():
    print("\n" + "=" * 60)
    print("BENCHMARK: Dim median, small N (in-register sort), float32")
    print("=" * 60)

    configs = [
        ((1024, 7), 1),
        ((1024, 15), 1),
        ((1024, 31), 1),
        ((1024, 63), 1),
        ((1024, 127), 1),
        ((1024, 255), 1),
        ((1024, 511), 1),
    ]

    rows = []
    for (shape, dim) in configs:
        N = shape[dim]
        base_ms = _time(make_baseline_dim(shape, dim, torch.float32), None)
        opt_ms  = _time(make_optimized_dim(shape, dim, torch.float32), None)
        speedup = base_ms / opt_ms if opt_ms > 0 else float("inf")
        rows.append({
            "shape": str(shape),
            "dim": dim,
            "N": N,
            "baseline_ms": round(base_ms, 4),
            "optimized_ms": round(opt_ms, 4),
            "speedup": round(speedup, 3),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Benchmark 3: Dim-reduction median (large N — radix-select path)
# ---------------------------------------------------------------------------

def bench_dim_large_n():
    print("\n" + "=" * 60)
    print("BENCHMARK: Dim median, large N (radix-select), float32")
    print("=" * 60)

    configs = [
        ((512, 1024), 1),
        ((256, 4096), 1),
        ((128, 16384), 1),
        ((64, 65536), 1),
    ]

    rows = []
    for (shape, dim) in configs:
        N = shape[dim]
        base_ms = _time(make_baseline_dim(shape, dim, torch.float32), None)
        opt_ms  = _time(make_optimized_dim(shape, dim, torch.float32), None)
        speedup = base_ms / opt_ms if opt_ms > 0 else float("inf")
        rows.append({
            "shape": str(shape),
            "dim": dim,
            "N": N,
            "baseline_ms": round(base_ms, 4),
            "optimized_ms": round(opt_ms, 4),
            "speedup": round(speedup, 3),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Benchmark 4: dtype sweep
# ---------------------------------------------------------------------------

def bench_dtype_sweep():
    print("\n" + "=" * 60)
    print("BENCHMARK: Dtype sweep, shape=(1024, 127), dim=1")
    print("=" * 60)

    dtypes = [torch.float32, torch.float16]
    shape = (1024, 127)
    dim = 1

    rows = []
    for dtype in dtypes:
        base_ms = _time(make_baseline_dim(shape, dim, dtype), None)
        opt_ms  = _time(make_optimized_dim(shape, dim, dtype), None)
        speedup = base_ms / opt_ms if opt_ms > 0 else float("inf")
        rows.append({
            "dtype": str(dtype),
            "shape": str(shape),
            "baseline_ms": round(base_ms, 4),
            "optimized_ms": round(opt_ms, 4),
            "speedup": round(speedup, 3),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def _time(fn, size, warmup=25, rep=100):
    import time
    if torch.cuda.is_available():
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        for _ in range(warmup):
            fn(size)
        torch.cuda.synchronize()
        for i in range(rep):
            starts[i].record()
            fn(size)
            ends[i].record()
        torch.cuda.synchronize()
        times = sorted([s.elapsed_time(e) for s, e in zip(starts, ends)])
    else:
        for _ in range(warmup):
            fn(size)
        times = []
        for _ in range(rep):
            t0 = time.perf_counter()
            fn(size)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\nDevice: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    r1 = bench_global_median()
    r2 = bench_dim_small_n()
    r3 = bench_dim_large_n()
    r4 = bench_dtype_sweep()

    # Save results
    all_results = pd.concat([r1, r2, r3, r4], ignore_index=True)
    out_path = os.path.join(os.path.dirname(__file__), "results_median.csv")
    all_results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
