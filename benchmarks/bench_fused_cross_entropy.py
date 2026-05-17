"""
Performance benchmark: fused_cross_entropy
  Fused log-softmax + NLL vs PyTorch native cross_entropy.

Metrics:
  - Median forward latency (ms)
  - Median backward latency (ms)
  - Peak GPU memory (MB)
  - Effective memory bandwidth (GB/s)
  - Speedup

Run with:
    python benchmarks/bench_fused_cross_entropy.py
"""

import sys, os, time
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from operators.fused_cross_entropy import fused_cross_entropy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _ref(logits, target, reduction="mean"):
    return torch.nn.functional.cross_entropy(logits, target, reduction=reduction)


def _time_ms(fn, warmup=20, rep=100):
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


def _peak_mb(fn):
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / 1e6


def _bw_gbs(B, V, elem_bytes, ms):
    # Read logits[B,V] once + target[B] + write loss: ~2*B*V
    n_bytes = 2 * B * V * elem_bytes
    return (n_bytes / 1e9) / (ms / 1000) if ms > 0 else 0.0


def bench_vocab_scaling():
    print("\n" + "=" * 72)
    print("BENCHMARK: Forward — vocabulary scaling, B=32, float32")
    print("=" * 72)

    rows = []
    for V in [1000, 10000, 32000, 50257, 65536, 131072]:
        B = 32
        logits = torch.randn(B, V, device=DEVICE)
        target = torch.randint(0, V, (B,), device=DEVICE)

        base_ms = _time_ms(lambda: _ref(logits, target))
        opt_ms  = _time_ms(lambda: fused_cross_entropy(logits, target))
        bw      = _bw_gbs(B, V, 4, opt_ms)
        mem     = _peak_mb(lambda: fused_cross_entropy(logits, target))

        rows.append({
            "V": V, "B": B,
            "baseline_ms": round(base_ms, 4),
            "fused_ms": round(opt_ms, 4),
            "speedup": round(base_ms / opt_ms if opt_ms > 0 else 0, 3),
            "BW_GBs": round(bw, 2),
            "peak_mem_MB": round(mem, 2),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_batch_scaling():
    print("\n" + "=" * 72)
    print("BENCHMARK: Forward — batch scaling, V=32000 (LLaMA vocab)")
    print("=" * 72)

    rows = []
    for B in [1, 4, 8, 16, 32, 64, 128, 256]:
        V = 32000
        logits = torch.randn(B, V, device=DEVICE)
        target = torch.randint(0, V, (B,), device=DEVICE)

        base_ms = _time_ms(lambda: _ref(logits, target))
        opt_ms  = _time_ms(lambda: fused_cross_entropy(logits, target))

        rows.append({
            "B": B,
            "baseline_ms": round(base_ms, 4),
            "fused_ms": round(opt_ms, 4),
            "speedup": round(base_ms / opt_ms if opt_ms > 0 else 0, 3),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_forward_backward():
    print("\n" + "=" * 72)
    print("BENCHMARK: Forward + Backward — typical LLM training configs")
    print("=" * 72)

    configs = [
        ("GPT-2",    32,  50257),
        ("LLaMA-7B", 16,  32000),
        ("LLaMA-70B", 8,  32000),
        ("Mistral",  32,  32000),
        ("Gemma",    32, 256000),
    ]

    rows = []
    for model, B, V in configs:
        logits_base = torch.randn(B, V, device=DEVICE, requires_grad=False)
        target      = torch.randint(0, V, (B,), device=DEVICE)

        def fwd_bwd_ref():
            l = logits_base.clone().detach().requires_grad_(True)
            _ref(l, target).backward()

        def fwd_bwd_opt():
            l = logits_base.clone().detach().requires_grad_(True)
            fused_cross_entropy(l, target).backward()

        base_ms = _time_ms(fwd_bwd_ref, warmup=10, rep=50)
        opt_ms  = _time_ms(fwd_bwd_opt, warmup=10, rep=50)
        mem_ref = _peak_mb(fwd_bwd_ref)
        mem_opt = _peak_mb(fwd_bwd_opt)

        rows.append({
            "model": model, "B": B, "V": V,
            "ref_fwd_bwd_ms":  round(base_ms, 3),
            "fused_fwd_bwd_ms": round(opt_ms, 3),
            "speedup": round(base_ms / opt_ms if opt_ms > 0 else 0, 3),
            "ref_mem_MB":   round(mem_ref, 1),
            "fused_mem_MB": round(mem_opt, 1),
            "mem_saved_MB": round(mem_ref - mem_opt, 1),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_reduction_modes():
    print("\n" + "=" * 72)
    print("BENCHMARK: Reduction modes — B=32, V=32000")
    print("=" * 72)

    B, V = 32, 32000
    logits = torch.randn(B, V, device=DEVICE)
    target = torch.randint(0, V, (B,), device=DEVICE)

    rows = []
    for reduction in ["none", "mean", "sum"]:
        base_ms = _time_ms(lambda: _ref(logits, target, reduction))
        opt_ms  = _time_ms(lambda: fused_cross_entropy(logits, target, reduction=reduction))
        rows.append({
            "reduction": reduction,
            "baseline_ms": round(base_ms, 4),
            "fused_ms": round(opt_ms, 4),
            "speedup": round(base_ms / opt_ms if opt_ms > 0 else 0, 3),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def bench_ignore_index():
    print("\n" + "=" * 72)
    print("BENCHMARK: ignore_index overhead — B=64, V=32000")
    print("=" * 72)

    B, V = 64, 32000
    logits = torch.randn(B, V, device=DEVICE)
    target = torch.randint(0, V, (B,), device=DEVICE)
    target_ignored = target.clone()
    target_ignored[::4] = -100  # 25% ignored

    rows = []
    for label, tgt in [("no ignore", target), ("25% ignored", target_ignored)]:
        opt_ms = _time_ms(lambda: fused_cross_entropy(logits, tgt, ignore_index=-100))
        rows.append({"config": label, "fused_ms": round(opt_ms, 4)})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    print(f"\nDevice : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Torch  : {torch.__version__}")

    r1 = bench_vocab_scaling()
    r2 = bench_batch_scaling()
    r3 = bench_forward_backward()
    r4 = bench_reduction_modes()
    r5 = bench_ignore_index()

    out = os.path.join(os.path.dirname(__file__), "results_fused_cross_entropy.csv")
    pd.concat([r1, r2, r3, r4, r5], ignore_index=True).to_csv(out, index=False)
    print(f"\nResults saved to {out}")
