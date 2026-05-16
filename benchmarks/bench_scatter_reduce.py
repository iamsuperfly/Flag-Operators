"""
Performance benchmark: optimized scatter_reduce vs FlagGems codegen baseline.

Measures:
  - Median kernel latency (ms)
  - Effective throughput (GB/s)
  - Speedup vs baseline
  - Peak memory delta (MB)

Run with:
    python benchmarks/bench_scatter_reduce.py
"""

import sys
import os
import time
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from operators.scatter_reduce import scatter_reduce_, scatter_reduce

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# Baseline: FlagGems codegen version (if available) or torch fallback
# ──────────────────────────────────────────────────────────────────────────────

def _get_baseline():
    """Return the baseline scatter_reduce_ to compare against."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "FlagGems", "src"))
        from flag_gems.ops.scatter_reduce_ import scatter_reduce_ as fg_sr
        print("Baseline: FlagGems codegen implementation")
        return fg_sr
    except ImportError:
        print("Baseline: torch.Tensor.scatter_reduce_ (PyTorch native)")
        def torch_baseline(inp, dim, index, src, reduce, *, include_self=True):
            out = inp.clone()
            out.scatter_reduce_(dim, index, src, reduce, include_self=include_self)
            return out
        return torch_baseline


baseline_fn = _get_baseline()


# ──────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _time_ms(fn, warmup=30, rep=200):
    """Median kernel time in milliseconds."""
    if torch.cuda.is_available():
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        for i in range(rep):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
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


def _peak_memory_mb(fn):
    """Extra GPU memory allocated during fn() in MB."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    after = torch.cuda.max_memory_allocated()
    return (after - before) / 1e6


def _throughput_gbs(n_bytes, ms):
    """Effective memory bandwidth in GB/s."""
    return (n_bytes / 1e9) / (ms / 1000) if ms > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Individual benchmarks
# ──────────────────────────────────────────────────────────────────────────────

def _make(shape, dim, dtype=torch.float32, idx_frac=0.7):
    idx_shape = list(shape)
    idx_shape[dim] = max(1, int(shape[dim] * idx_frac))
    inp   = torch.rand(shape, dtype=dtype, device=DEVICE)
    src   = torch.rand(idx_shape, dtype=dtype, device=DEVICE)
    index = torch.randint(0, shape[dim], idx_shape, device=DEVICE)
    return inp, index, src


def bench_reduce_modes():
    """Compare all 5 reduce modes on a 512x512 float32 tensor."""
    print("\n" + "=" * 68)
    print("BENCHMARK: All reduce modes  — shape=(512, 512), dim=1, float32")
    print("=" * 68)

    shape, dim = (512, 512), 1
    inp, index, src = _make(shape, dim)
    n_bytes = (inp.numel() + src.numel() + index.numel() * 4) * 4  # rough

    rows = []
    for reduce in ["sum", "prod", "amax", "amin", "mean"]:
        base_ms = _time_ms(lambda: baseline_fn(inp.clone(), dim, index, src, reduce))
        opt_ms  = _time_ms(lambda: scatter_reduce_(inp.clone(), dim, index, src, reduce))
        speedup = base_ms / opt_ms if opt_ms > 0 else float("inf")
        bw      = _throughput_gbs(n_bytes, opt_ms)
        rows.append({
            "reduce": reduce,
            "baseline_ms": round(base_ms, 4),
            "optimized_ms": round(opt_ms, 4),
            "speedup": round(speedup, 3),
            "throughput_GBs": round(bw, 2),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_shapes():
    """Scaling across tensor sizes."""
    print("\n" + "=" * 68)
    print("BENCHMARK: Shape scaling — reduce=sum, dim=1, float32")
    print("=" * 68)

    configs = [
        (64, 64),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 512),
        (512, 2048),
        (64, 64, 64),
        (32, 64, 64),
    ]

    rows = []
    for shape in configs:
        dim = 1
        inp, index, src = _make(shape, dim)
        n_bytes = (inp.numel() + src.numel()) * inp.element_size()
        base_ms = _time_ms(lambda: baseline_fn(inp.clone(), dim, index, src, "sum"))
        opt_ms  = _time_ms(lambda: scatter_reduce_(inp.clone(), dim, index, src, "sum"))
        speedup = base_ms / opt_ms if opt_ms > 0 else float("inf")
        bw      = _throughput_gbs(n_bytes, opt_ms)
        rows.append({
            "shape": str(shape),
            "N_index": index.numel(),
            "baseline_ms": round(base_ms, 4),
            "optimized_ms": round(opt_ms, 4),
            "speedup": round(speedup, 3),
            "throughput_GBs": round(bw, 2),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_dtypes():
    """float32 vs float16."""
    print("\n" + "=" * 68)
    print("BENCHMARK: Dtype sweep — shape=(512, 512), dim=1, reduce=sum")
    print("=" * 68)

    shape, dim = (512, 512), 1
    rows = []
    for dtype in [torch.float32, torch.float16]:
        inp, index, src = _make(shape, dim, dtype=dtype)
        base_ms = _time_ms(lambda: baseline_fn(inp.clone(), dim, index, src, "sum"))
        opt_ms  = _time_ms(lambda: scatter_reduce_(inp.clone(), dim, index, src, "sum"))
        rows.append({
            "dtype": str(dtype),
            "baseline_ms": round(base_ms, 4),
            "optimized_ms": round(opt_ms, 4),
            "speedup": round(base_ms / opt_ms if opt_ms > 0 else 0, 3),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_index_density():
    """How does index density (few vs many scatter targets) affect perf?"""
    print("\n" + "=" * 68)
    print("BENCHMARK: Index density — shape=(512, 512), dim=1, reduce=sum")
    print("=" * 68)

    shape, dim = (512, 512), 1
    rows = []
    for frac in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        inp, index, src = _make(shape, dim, idx_frac=frac)
        base_ms = _time_ms(lambda: baseline_fn(inp.clone(), dim, index, src, "sum"))
        opt_ms  = _time_ms(lambda: scatter_reduce_(inp.clone(), dim, index, src, "sum"))
        rows.append({
            "idx_frac": frac,
            "N_index": index.numel(),
            "baseline_ms": round(base_ms, 4),
            "optimized_ms": round(opt_ms, 4),
            "speedup": round(base_ms / opt_ms if opt_ms > 0 else 0, 3),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_memory():
    """Peak GPU memory delta for optimized vs baseline."""
    print("\n" + "=" * 68)
    print("BENCHMARK: Peak memory delta (MB) — shape=(1024,1024), dim=1")
    print("=" * 68)

    shape, dim = (1024, 1024), 1
    inp, index, src = _make(shape, dim)

    rows = []
    for label, fn in [
        ("baseline",  lambda: baseline_fn(inp.clone(), dim, index, src, "sum")),
        ("optimized", lambda: scatter_reduce_(inp.clone(), dim, index, src, "sum")),
    ]:
        mb = _peak_memory_mb(fn)
        rows.append({"variant": label, "peak_memory_delta_MB": round(mb, 2)})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_variants():
    """Compare the three public variants: inplace / outofplace / out."""
    from operators.scatter_reduce import scatter_reduce_out
    print("\n" + "=" * 68)
    print("BENCHMARK: Three public variants — shape=(512,512), dim=1, sum")
    print("=" * 68)

    shape, dim = (512, 512), 1
    inp, index, src = _make(shape, dim)
    out_buf = torch.empty_like(inp)

    rows = []
    for label, fn in [
        ("scatter_reduce_  (in-place)",      lambda: scatter_reduce_(inp.clone(), dim, index, src, "sum")),
        ("scatter_reduce   (out-of-place)",  lambda: scatter_reduce(inp, dim, index, src, "sum")),
        ("scatter_reduce_out (explicit out)", lambda: scatter_reduce_out(out_buf, inp, dim, index, src, "sum")),
    ]:
        ms = _time_ms(fn)
        rows.append({"variant": label, "latency_ms": round(ms, 4)})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nDevice : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"CUDA   : {torch.version.cuda}")
    print(f"Torch  : {torch.__version__}")

    r1 = bench_reduce_modes()
    r2 = bench_shapes()
    r3 = bench_dtypes()
    r4 = bench_index_density()
    r5 = bench_memory()
    r6 = bench_variants()

    out_csv = os.path.join(os.path.dirname(__file__), "results_scatter_reduce.csv")
    pd.concat([r1, r2, r3, r4, r5, r6], ignore_index=True).to_csv(out_csv, index=False)
    print(f"\nAll results saved to {out_csv}")
