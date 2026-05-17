"""
Performance benchmark: chunk_gated_delta_rule
  Optimized Triton kernel vs PyTorch sequential reference.

Metrics:
  - Median latency (ms)
  - Effective throughput (GFLOP/s)
  - Memory allocated (MB)
  - Speedup vs reference

Run with:
    python benchmarks/bench_chunk_gated_delta_rule.py
"""

import sys, os, time
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from operators.chunk_gated_delta_rule import chunk_gated_delta_rule, _ref_forward

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make(B, H, L, D, dtype=torch.float32):
    q    = torch.randn(B, H, L, D, dtype=dtype, device=DEVICE)
    k    = torch.randn(B, H, L, D, dtype=dtype, device=DEVICE)
    k    = k / (k.norm(dim=-1, keepdim=True).clamp(min=1e-6))
    v    = torch.randn(B, H, L, D, dtype=dtype, device=DEVICE)
    beta = torch.sigmoid(torch.randn(B, H, L, dtype=dtype, device=DEVICE))
    g    = torch.sigmoid(torch.randn(B, H, L, dtype=dtype, device=DEVICE))
    return q, k, v, beta, g


def _time_ms(fn, warmup=10, rep=50):
    if torch.cuda.is_available():
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        for i in range(rep):
            starts[i].record(); fn(); ends[i].record()
        torch.cuda.synchronize()
        times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    else:
        for _ in range(warmup):
            fn()
        times = []
        for _ in range(rep):
            t0 = time.perf_counter(); fn()
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
    return times[len(times) // 2]


def _flops(B, H, L, D):
    """Approximate FLOPs per forward pass: 3 * D² ops per timestep."""
    return B * H * L * 3 * D * D


def _peak_mb(fn):
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / 1e6


def bench_vs_reference():
    print("\n" + "=" * 72)
    print("BENCHMARK: Triton kernel vs PyTorch reference")
    print("=" * 72)

    configs = [
        (1, 1, 64,  32),
        (1, 1, 128, 32),
        (1, 1, 256, 32),
        (2, 4, 128, 32),
        (2, 4, 256, 32),
        (2, 8, 128, 64),
        (1, 1, 128, 64),
        (4, 8, 64,  64),
    ]

    rows = []
    for B, H, L, D in configs:
        q, k, v, beta, g = _make(B, H, L, D)

        ref_ms  = _time_ms(lambda: _ref_forward(q.cpu(), k.cpu(), v.cpu(), beta.cpu(), g.cpu()))
        opt_ms  = _time_ms(lambda: chunk_gated_delta_rule(q, k, v, beta, g))
        speedup = ref_ms / opt_ms if opt_ms > 0 else float("inf")
        gflops  = _flops(B, H, L, D) / 1e9
        tflops  = gflops / (opt_ms / 1000)
        mem_mb  = _peak_mb(lambda: chunk_gated_delta_rule(q, k, v, beta, g))

        rows.append({
            "B": B, "H": H, "L": L, "D": D,
            "ref_ms": round(ref_ms, 3),
            "opt_ms": round(opt_ms, 3),
            "speedup": round(speedup, 2),
            "TFLOP/s": round(tflops, 3),
            "mem_MB":  round(mem_mb, 2),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_sequence_scaling():
    print("\n" + "=" * 72)
    print("BENCHMARK: Sequence length scaling — B=1 H=4 D=64")
    print("=" * 72)

    rows = []
    for L in [32, 64, 128, 256, 512, 1024]:
        B, H, D = 1, 4, 64
        q, k, v, beta, g = _make(B, H, L, D)
        opt_ms  = _time_ms(lambda: chunk_gated_delta_rule(q, k, v, beta, g))
        ref_ms  = _time_ms(lambda: _ref_forward(q.cpu(), k.cpu(), v.cpu(), beta.cpu(), g.cpu()))
        gflops  = _flops(B, H, L, D) / 1e9
        tflops  = gflops / (opt_ms / 1000)
        rows.append({
            "L": L,
            "ref_ms": round(ref_ms, 3),
            "opt_ms": round(opt_ms, 3),
            "speedup": round(ref_ms / opt_ms if opt_ms > 0 else 0, 2),
            "TFLOP/s": round(tflops, 3),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_head_dim_scaling():
    print("\n" + "=" * 72)
    print("BENCHMARK: Head-dim scaling — B=2 H=4 L=128")
    print("=" * 72)

    rows = []
    for D in [16, 32, 64]:
        B, H, L = 2, 4, 128
        q, k, v, beta, g = _make(B, H, L, D)
        opt_ms  = _time_ms(lambda: chunk_gated_delta_rule(q, k, v, beta, g))
        ref_ms  = _time_ms(lambda: _ref_forward(q.cpu(), k.cpu(), v.cpu(), beta.cpu(), g.cpu()))
        gflops  = _flops(B, H, L, D) / 1e9
        rows.append({
            "D": D,
            "state_size": f"{D}x{D}",
            "ref_ms":  round(ref_ms, 3),
            "opt_ms":  round(opt_ms, 3),
            "speedup": round(ref_ms / opt_ms if opt_ms > 0 else 0, 2),
            "GFLOP/s": round(gflops / (opt_ms / 1000), 2),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_batch_head():
    print("\n" + "=" * 72)
    print("BENCHMARK: Batch×Head parallelism — L=128 D=32")
    print("=" * 72)

    rows = []
    for B, H in [(1, 1), (1, 4), (2, 4), (4, 8), (8, 8)]:
        L, D = 128, 32
        q, k, v, beta, g = _make(B, H, L, D)
        opt_ms = _time_ms(lambda: chunk_gated_delta_rule(q, k, v, beta, g))
        ref_ms = _time_ms(lambda: _ref_forward(q.cpu(), k.cpu(), v.cpu(), beta.cpu(), g.cpu()))
        rows.append({
            "B": B, "H": H, "B*H": B * H,
            "ref_ms":  round(ref_ms, 3),
            "opt_ms":  round(opt_ms, 3),
            "speedup": round(ref_ms / opt_ms if opt_ms > 0 else 0, 2),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    print(f"\nDevice : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"CUDA   : {torch.version.cuda}")
    print(f"Torch  : {torch.__version__}")

    r1 = bench_vs_reference()
    r2 = bench_sequence_scaling()
    r3 = bench_head_dim_scaling()
    r4 = bench_batch_head()

    out = os.path.join(os.path.dirname(__file__), "results_chunk_gated_delta_rule.csv")
    pd.concat([r1, r2, r3, r4], ignore_index=True).to_csv(out, index=False)
    print(f"\nResults saved to {out}")
