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
├── operators/          # Optimized Triton operator implementations
│   ├── median.py       # Operator 01
│   └── scatter_reduce.py  # Operator 02
├── tests/             # Accuracy & correctness tests (pytest)
│   ├── test_utils.py
│   ├── test_median.py
│   └── test_scatter_reduce.py
├── benchmarks/        # Performance benchmarks vs baseline
│   ├── bench_utils.py
│   ├── bench_median.py
│   └── bench_scatter_reduce.py
├── docs/              # Technical report and design notes
│   └── technical_report.md
├── FlagGems/          # Official FlagGems repo (reference, not modified)
└── README.md
```

---

## Operators

Track 1 covers 20 operators from the FlagGems official task list.

| # | Operator | Status | Optimization Strategy | Key Speedup |
|---|----------|--------|-----------------------|-------------|
| 01 | `median` | ✅ Completed | In-register odd-even sort (N≤512) + radix-select (N>512) | Avoids full sort for large N |
| 02 | `scatter_reduce` (full trio) | ✅ Completed | Static Triton kernel + autotuning; native `atomic_add`/`atomic_max`/`atomic_min`; eliminates codegen overhead | No runtime file I/O; native atomics for int types |
| 03 | `chunk_gated_delta_rule` | ⏳ Next | — | — |
| 04 | `ctc_loss` | ⏳ Pending | — | — |
| 05–20 | TBD | ⏳ Pending | — | — |

### Operator 02 — `scatter_reduce` Details

**Three variants implemented:**
- `scatter_reduce_(inp, dim, index, src, reduce)` — in-place
- `scatter_reduce(inp, dim, index, src, reduce)` — out-of-place
- `scatter_reduce_out(out, inp, dim, index, src, reduce)` — explicit output buffer

**What we replaced:** The FlagGems codegen version generates Python source at runtime, writes it to a temp file, and imports it via `importlib` — incurring disk I/O and import overhead on every fresh session.

**Our approach:**
- Single static `@triton.jit` kernel with `tl.constexpr` specialization (NDIM, reduce type, dtype, int32/int64 offsets)
- `@triton.autotune` across 5 configs (BLOCK 64–1024, num_warps 2–8)
- `tl.atomic_add` with `sem="relaxed"` for sum — zero CAS contention
- `tl.atomic_max` / `tl.atomic_min` for integer amax/amin — native hardware path
- Float amax/amin: CAS with per-element early-exit flag (skips CAS when `src ≤ current`)
- All 5 reduce modes: `sum`, `prod`, `amax`, `amin`, `mean`
- `include_self=True/False` handled correctly for all modes

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
```

---

## Progress

| Phase | Status |
|-------|--------|
| Environment Setup | ✅ Done |
| FlagGems Reference Clone | ✅ Done |
| Operator 01: `median` | ✅ Completed |
| Operator 02: `scatter_reduce` (full trio) | ✅ Completed |
| Operator 03: `chunk_gated_delta_rule` | ⏳ Next |
| Operators 04–20 | ⏳ Pending |

---

## Contributing

This repo is a hackathon submission. All improvements are upstream-compatible with FlagGems and will be contributed back upon challenge completion.

---

## References

- [FlagOS Challenge Page](https://flagos.io/)
- [FlagGems Repository](https://github.com/FlagOpen/FlagGems)
- [Triton Documentation](https://triton-lang.org/)
- [OpenAI Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)

---

## License

Apache License 2.0 — compatible with FlagGems upstream license.
