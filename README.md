# Flag-Operators

**Official submission for FlagOS Open Computing Global Challenge (Season 1) – Track 1: Operator Development**

Optimized and high-performance operators for **FlagGems** (the core operator library of FlagOS).

---

## Goal

Improve selected operators from the official 20-task list to make them:
- **Faster** — lower latency via better Triton tiling, vectorization, and kernel fusion
- **More memory-efficient** — smarter use of shared memory, reduced register pressure
- **Production-ready** — clean, well-tested, and benchmarked contributions for the FlagOS ecosystem

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Kernel DSL | [Triton](https://github.com/openai/triton) 3.x |
| Operator Library | [FlagGems](https://github.com/FlagOpen/FlagGems) |
| Deep Learning | PyTorch 2.5+ (CUDA 12.1) |
| Benchmarking | Custom harness + FlagGems benchmark suite |
| Testing | pytest + torch.testing |

---

## Repository Structure

```
Flag-Operators/
├── operators/                    # Optimized Triton operator implementations
│   ├── median.py                 # Operator 01
│   ├── scatter_reduce.py         # Operator 02
│   ├── chunk_gated_delta_rule.py # Operator 03
│   ├── rms_norm.py               # Operator 04
│   └── fused_cross_entropy.py   # Operator 05
├── tests/                       # Accuracy & correctness tests (pytest)
│   ├── test_utils.py
│   ├── test_median.py
│   ├── test_scatter_reduce.py
│   ├── test_chunk_gated_delta_rule.py
│   ├── test_rms_norm.py
│   └── test_fused_cross_entropy.py
├── benchmarks/                  # Performance benchmarks vs baseline
│   ├── bench_utils.py
│   ├── bench_median.py
│   ├── bench_scatter_reduce.py
│   ├── bench_chunk_gated_delta_rule.py
│   ├── bench_rms_norm.py
│   └── bench_fused_cross_entropy.py
├── docs/
│   └── technical_report.md
├── FlagGems/                    # Official FlagGems repo (reference, not modified)
└── README.md
```

---

## Operators

| # | Operator | Status | Key Optimization | Impact |
|---|----------|--------|-----------------|--------|
| 01 | `median` | ✅ Completed | In-register odd-even sort (N≤512) + radix-select (N>512) | Avoids full sort for large N |
| 02 | `scatter_reduce` (full trio) | ✅ Completed | Static `@triton.jit` replaces codegen; `tl.atomic_add/max/min`; 5-config autotune | No runtime I/O; native atomics |
| 03 | `chunk_gated_delta_rule` | ✅ Completed | Fused state-space kernel; D²-state in registers; zero intermediate tensors | Net-new (not in FlagGems) |
| 04 | `rms_norm` | ✅ Completed | Vectorized loads; 9-config autotune; `evict_first/last` cache policy; single-pass for N≤4096 | Lower memory traffic |
| 05 | `fused_cross_entropy` | ✅ Completed | Online max+LSE in one pass; fused backward recomputes softmax on-the-fly; no O(B×V) intermediate | ~2× memory saving for V=128K |
| 06–20 | TBD | ⏳ Pending | — | — |

---

## Operator Details

### Op 03 — `chunk_gated_delta_rule` (Net-new, missing from FlagGems)

The Gated Delta Rule recurrence used in DeltaNet / GLA state-space models:
```
h[t] = g[t] * h[t-1]  +  β[t] * (v[t] - h[t-1] @ k[t]) ⊗ k[t]
o[t] = h[t] @ q[t]
```

**Three public inputs:** `q, k, v` ∈ R^{B×H×L×D}, `beta, g` ∈ R^{B×H×L}

**Key optimizations:**
- One Triton program per (batch, head) pair — full `D×D` state kept in registers/L1
- Zero global-memory traffic for the state matrix between timesteps
- Arithmetic intensity = O(D) → compute-bound for D ≥ 16
- `float32` accumulation for numerical stability across long recurrences

### Op 04 — `rms_norm` (Replaces FlagGems 2-pass kernel)

**FlagGems baseline:** 2-pass kernel — reads `x` twice from global memory.

**Our approach:**
- **Small N (≤4096):** single-pass kernel — entire row in registers, one global read
- **Large N:** vectorized 2-pass with `evict_first` on pass 1, `evict_last` on pass 2 for L2 reuse
- **9 autotuning configs** vs FlagGems `runtime.get_tuned_config` (offline lookup)
- `inv_rms` saved for backward (API-compatible with FlagGems)

### Op 05 — `fused_cross_entropy` (Replaces PyTorch built-in)

**Motivation:** For LLM training with V=128K vocab, standard CE materialises a 2 GB activation per batch step.

**Our approach:**
- **Online max + log-sum-exp in a single pass** — no intermediate softmax stored
- **Tiled streaming** over V with configurable TILE_V (autotuned 512–4096)
- **Fused backward** recomputes `exp(logit - lse)` on-the-fly — no O(B×V) gradient buffer
- Supports `ignore_index`, `label_smoothing`, `reduction='none'/'mean'/'sum'`
- **Memory saving:** ~`B × V × 4` bytes per forward (e.g. 256 MB for B=32, V=128K)

---

## How to Run

### Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install triton numpy pandas matplotlib
```

### Run tests

```bash
pytest tests/ -v
```

### Run benchmarks

```bash
python benchmarks/bench_median.py
python benchmarks/bench_scatter_reduce.py
python benchmarks/bench_chunk_gated_delta_rule.py
python benchmarks/bench_rms_norm.py
python benchmarks/bench_fused_cross_entropy.py
```

---

## Progress

| Phase | Status |
|-------|--------|
| Environment Setup | ✅ Done |
| FlagGems Reference Clone | ✅ Done |
| Operator 01: `median` | ✅ Completed |
| Operator 02: `scatter_reduce` (full trio) | ✅ Completed |
| Operator 03: `chunk_gated_delta_rule` | ✅ Completed |
| Operator 04: `rms_norm` | ✅ Completed |
| Operator 05: `fused_cross_entropy` | ✅ Completed |
| Operators 06–20 | ⏳ In Progress |

---

## Contributing

This repo is a hackathon submission. All improvements are upstream-compatible with FlagGems and will be contributed back upon challenge completion.

---

## References

- [FlagOS Challenge Page](https://flagos.io/)
- [FlagGems Repository](https://github.com/FlagOpen/FlagGems)
- [Triton Documentation](https://triton-lang.org/)
- [DeltaNet Paper](https://arxiv.org/abs/2406.06484)
- [GLA Paper](https://arxiv.org/abs/2312.06635)

---

## License

Apache License 2.0 — compatible with FlagGems upstream license.
