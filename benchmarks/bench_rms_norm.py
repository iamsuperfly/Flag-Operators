"""
Performance benchmark: rms_norm
  Optimized vectorized kernel vs FlagGems baseline.

Metrics:
  - Median latency (ms)
  - Effective memory bandwidth (GB/s)
  - Speedup vs baseline

Run with:
    python benchmarks/bench_rms_norm.py
"""

import sys, os, time
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from operators.rms_norm import rms_norm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_baseline():
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "FlagGems", "src"))
        from flag_gems.ops.rms_norm import rms_norm as fg_rms
        print("Baseline: FlagGems rms_norm")
        return fg_rms
    except ImportError:
        print("Baseline: torch.nn.functional.rms_norm (PyTorch native)")
        import torch.nn.functional as F
        def torch_rms(x, norm_shape, w, eps=1e-5):
            try:
                return F.rms_norm(x, norm_shape, weight=w, eps=eps)
            except AttributeError:
                rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
                return ((x.float() / rms) * w.float()).to(x.dtype)
        return torch_rms


baseline_fn = _get_baseline()


def _time_ms(fn, warmup=30, rep=200):
    if torch.cuda.is_available():
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        for i in range(rep):
            starts[i].record(); fn(); ends[i].record()
        torch.cuda.synchronize()
        times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    else:
        for _ in range(warmup): fn()
        times = []
        for _ in range(rep):
            t0 = time.perf_counter(); fn()
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
    return times[len(times) // 2]


def _bw_gbs(M, N, elem_bytes, ms):
    # Read x + w, write y: (M*N + N + M*N) elements
    n_bytes = (2 * M * N + N) * elem_bytes
    return (n_bytes / 1e9) / (ms / 1000) if ms > 0 else 0.0


def bench_hidden_dim_scaling():
    print("\n" + "=" * 68)
    print("BENCHMARK: Hidden-dim scaling — M=1024, float32")
    print("=" * 68)

    rows = []
    for N in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        M = 1024
        x = torch.randn(M, N, device=DEVICE)
        w = torch.ones(N, device=DEVICE)

        base_ms = _time_ms(lambda: baseline_fn(x, (N,), w))
        opt_ms  = _time_ms(lambda: rms_norm(x, (N,), w))
        bw      = _bw_gbs(M, N, 4, opt_ms)
        speedup = base_ms / opt_ms if opt_ms > 0 else 0

        rows.append({
            "N": N,
            "baseline_ms": round(base_ms, 4),
            "optimized_ms": round(opt_ms, 4),
            "speedup": round(speedup, 3),
            "BW_GBs": round(bw, 2),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_batch_size():
    print("\n" + "=" * 68)
    print("BENCHMARK: Batch scaling — N=4096, float32")
    print("=" * 68)

    rows = []
    for M in [1, 4, 16, 64, 256, 1024, 4096, 16384]:
        N = 4096
        x = torch.randn(M, N, device=DEVICE)
        w = torch.ones(N, device=DEVICE)

        base_ms = _time_ms(lambda: baseline_fn(x, (N,), w))
        opt_ms  = _time_ms(lambda: rms_norm(x, (N,), w))
        bw      = _bw_gbs(M, N, 4, opt_ms)

        rows.append({
            "M": M,
            "baseline_ms": round(base_ms, 4),
            "optimized_ms": round(opt_ms, 4),
            "speedup": round(base_ms / opt_ms if opt_ms > 0 else 0, 3),
            "BW_GBs": round(bw, 2),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_dtypes():
    print("\n" + "=" * 68)
    print("BENCHMARK: dtype sweep — M=1024, N=4096")
    print("=" * 68)

    M, N = 1024, 4096
    rows = []
    for dtype, name in [(torch.float32, "float32"), (torch.float16, "float16"),
                        (torch.bfloat16, "bfloat16")]:
        if not torch.cuda.is_available() and dtype != torch.float32:
            continue
        x = torch.randn(M, N, device=DEVICE, dtype=dtype)
        w = torch.ones(N, device=DEVICE, dtype=dtype)
        base_ms = _time_ms(lambda: baseline_fn(x, (N,), w))
        opt_ms  = _time_ms(lambda: rms_norm(x, (N,), w))
        bw      = _bw_gbs(M, N, x.element_size(), opt_ms)
        rows.append({
            "dtype": name,
            "baseline_ms": round(base_ms, 4),
            "optimized_ms": round(opt_ms, 4),
            "speedup": round(base_ms / opt_ms if opt_ms > 0 else 0, 3),
            "BW_GBs": round(bw, 2),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_transformer_shapes():
    print("\n" + "=" * 68)
    print("BENCHMARK: Typical transformer shapes")
    print("=" * 68)

    configs = [
        ("BERT-base",   512, 768,   "float32"),
        ("BERT-large",  512, 1024,  "float32"),
        ("LLaMA-7B",   4096, 4096,  "float32"),
        ("LLaMA-70B",  4096, 8192,  "float32"),
        ("GPT-3",      2048, 12288, "float32"),
        ("Mistral-7B", 4096, 4096,  "float32"),
    ]

    rows = []
    for model, M, N, dtype_name in configs:
        dtype = torch.float32
        x = torch.randn(M, N, device=DEVICE, dtype=dtype)
        w = torch.ones(N, device=DEVICE, dtype=dtype)
        base_ms = _time_ms(lambda: baseline_fn(x, (N,), w))
        opt_ms  = _time_ms(lambda: rms_norm(x, (N,), w))
        bw      = _bw_gbs(M, N, 4, opt_ms)
        rows.append({
            "model": model,
            "M": M, "N": N,
            "baseline_ms":  round(base_ms, 4),
            "optimized_ms": round(opt_ms, 4),
            "speedup":      round(base_ms / opt_ms if opt_ms > 0 else 0, 3),
            "BW_GBs":       round(bw, 2),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    print(f"\nDevice : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Torch  : {torch.__version__}")

    r1 = bench_hidden_dim_scaling()
    r2 = bench_batch_size()
    r3 = bench_dtypes()
    r4 = bench_transformer_shapes()

    out = os.path.join(os.path.dirname(__file__), "results_rms_norm.csv")
    pd.concat([r1, r2, r3, r4], ignore_index=True).to_csv(out, index=False)
    print(f"\nResults saved to {out}")
