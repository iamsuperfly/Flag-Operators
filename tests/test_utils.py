"""
Shared test utilities for Flag-Operators accuracy tests.
"""
import torch


def allclose(a: torch.Tensor, b: torch.Tensor, rtol=1e-3, atol=1e-3) -> bool:
    """Check if two tensors are numerically close."""
    return torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return max absolute difference between two tensors."""
    return (a.float() - b.float()).abs().max().item()


def benchmark_ms(fn, warmup=25, rep=100) -> float:
    """
    Time a callable in milliseconds (median of `rep` runs after `warmup`).
    Falls back to CPU timing when CUDA is unavailable.
    """
    import time

    if torch.cuda.is_available():
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        # warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        for i in range(rep):
            start_events[i].record()
            fn()
            end_events[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    else:
        for _ in range(warmup):
            fn()
        times = []
        for _ in range(rep):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]  # median
