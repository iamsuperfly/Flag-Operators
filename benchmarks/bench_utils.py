"""
Shared benchmark utilities for Flag-Operators.
"""
import torch
import pandas as pd
from typing import Callable, Dict, List


def benchmark_op(
    name: str,
    baseline_fn: Callable,
    optimized_fn: Callable,
    input_sizes: List,
    warmup: int = 25,
    rep: int = 100,
) -> pd.DataFrame:
    """
    Benchmark baseline vs optimized implementation across multiple input sizes.
    Returns a DataFrame with columns: size, baseline_ms, optimized_ms, speedup.
    """
    rows = []
    for size in input_sizes:
        base_ms = _time_fn(baseline_fn, size, warmup, rep)
        opt_ms = _time_fn(optimized_fn, size, warmup, rep)
        speedup = base_ms / opt_ms if opt_ms > 0 else float("inf")
        rows.append(
            {
                "operator": name,
                "size": str(size),
                "baseline_ms": round(base_ms, 4),
                "optimized_ms": round(opt_ms, 4),
                "speedup": round(speedup, 3),
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def _time_fn(fn: Callable, size, warmup: int, rep: int) -> float:
    import time

    if torch.cuda.is_available():
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        for _ in range(warmup):
            fn(size)
        torch.cuda.synchronize()
        for i in range(rep):
            start_events[i].record()
            fn(size)
            end_events[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
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
