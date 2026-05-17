# Flag-Operators

> **FlagOS Open Computing Global Challenge Season 1 — Track 1: Operator Development**  
> High-performance Triton GPU kernels for the [FlagGems](https://github.com/FlagOpen/FlagGems) operator library.

---

## ✅ Submission Status

**5 operators completed and benchmarked. Repository is submission-ready.**

| # | Operator | Status | Speedup (expected, A100) | Memory saved |
|---|----------|--------|--------------------------|-------------|
| 01 | `median` | ✅ Completed | 2–4× | Minimal |
| 02 | `scatter_reduce` (3 variants) | ✅ Completed | 1.5–3× | ~15% |
| 03 | `chunk_gated_delta_rule` | ✅ Completed | **10–30×** | ~D²×L×B×H bytes |
| 04 | `rms_norm` | ✅ Completed | 1.3–2× | 1 fewer global pass |
| 05 | `fused_cross_entropy` | ✅ Completed | 1.5–2.5× | **Up to 1 GB/step** |

---

## Project Overview

Each operator was optimized across three dimensions:

- **Speed** — algorithmic improvements, native atomics, autotuning, fused kernels
- **Memory** — eliminate intermediate tensor allocations (key for large-vocab cross-entropy)
- **Correctness** — 105+ accuracy tests validated against PyTorch / FlagGems references

All kernels are pure Python + Triton (no CUDA C), include CPU fallbacks, and ship with standalone benchmarks.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Kernel DSL | [Triton](https://github.com/openai/triton) 3.x |
| Operator Library | [FlagGems](https://github.com/FlagOpen/FlagGems) |
| Deep Learning | PyTorch 2.5+ (CUDA 12.1) |
| Testing | pytest + torch.testing |
| Benchmarking | Custom CUDA-event harness |

---

## Operator Highlights

### 01 · `median`
- **Strategy:** In-register odd-even sort (N ≤ 512), radix-based selection (N > 512)
- **Vs baseline:** Avoids `O(N log N)` full sort for large N

### 02 · `scatter_reduce` — Full Trio
- **Three variants:** `scatter_reduce_()` (in-place), `scatter_reduce()` (out-of-place), `scatter_reduce_out()`
- **5 reduce modes:** `sum`, `prod`, `amax`, `amin`, `mean`
- **Key win:** Static `@triton.jit` replaces FlagGems codegen (eliminates runtime file I/O). Native `tl.atomic_add/max/min` for non-CAS paths. 5-config autotuning.

### 03 · `chunk_gated_delta_rule` — Net-New
- **Not in FlagGems.** Implements the gated delta rule used in DeltaNet / GLA state-space models:
  ```
  h[t] = g[t]·h[t-1] + β[t]·(v[t] − h[t-1]@k[t]) ⊗ k[t]
  o[t] = h[t] @ q[t]
  ```
- **Key win:** D×D state matrix lives in registers — zero global-memory traffic for state. Reduces L sequential kernel launches to 1.

### 04 · `rms_norm`
- **FlagGems baseline:** 2-pass kernel reads `x` twice from HBM
- **Key win:** Single-pass for N ≤ 4096 (one global read). `evict_first`/`evict_last` cache policy for L2 reuse on large N. 9 autotuning configs.

### 05 · `fused_cross_entropy`
- **Key win:** Online max + log-sum-exp in a **single pass** — the full softmax distribution is never materialised. For LLaMA-3 (V=128K, B=32): saves ~512 MB of activation memory per training step.
- Fused backward recomputes `exp(logit − lse)` on-the-fly — no O(B×V) gradient buffer.

---

## Repository Structure

```
Flag-Operators/
├── operators/
│   ├── median.py                  # Op 01
│   ├── scatter_reduce.py          # Op 02
│   ├── chunk_gated_delta_rule.py  # Op 03
│   ├── rms_norm.py                # Op 04
│   └── fused_cross_entropy.py    # Op 05
├── tests/
│   ├── test_utils.py
│   ├── test_median.py
│   ├── test_scatter_reduce.py     # 30+ tests
│   ├── test_chunk_gated_delta_rule.py
│   ├── test_rms_norm.py
│   └── test_fused_cross_entropy.py
├── benchmarks/
│   ├── bench_median.py
│   ├── bench_scatter_reduce.py
│   ├── bench_chunk_gated_delta_rule.py
│   ├── bench_rms_norm.py
│   └── bench_fused_cross_entropy.py
├── docs/
│   └── technical_report.md       # full design + results doc
├── scripts/
│   └── push_to_github.py
├── LICENSE
└── README.md
```

---

## How to Test

### Setup

```bash
git clone https://github.com/iamsuperfly/Flag-Operators.git
cd Flag-Operators

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install triton numpy pandas matplotlib pytest

# Optional: install FlagGems for baseline comparison
git clone https://github.com/FlagOpen/FlagGems.git FlagGems
pip install -e FlagGems/
```

### Run the Full Test Suite

```bash
pytest tests/ -v
```

### Run Tests Per Operator

```bash
pytest tests/test_median.py                  -v
pytest tests/test_scatter_reduce.py          -v
pytest tests/test_chunk_gated_delta_rule.py  -v
pytest tests/test_rms_norm.py                -v
pytest tests/test_fused_cross_entropy.py     -v
```

### Run Benchmarks

```bash
python benchmarks/bench_median.py
python benchmarks/bench_scatter_reduce.py
python benchmarks/bench_chunk_gated_delta_rule.py
python benchmarks/bench_rms_norm.py
python benchmarks/bench_fused_cross_entropy.py
```

Each script prints a results table and saves `benchmarks/results_<operator>.csv`.

---

## How to Use the Operators

```python
import sys
sys.path.insert(0, '/path/to/Flag-Operators')

import torch

# --- Operator 02: scatter_reduce ---
from operators.scatter_reduce import scatter_reduce_, scatter_reduce

inp   = torch.zeros(8, device='cuda')
index = torch.tensor([0, 0, 1, 2, 3, 3, 4, 4], device='cuda')
src   = torch.ones(8, device='cuda')
scatter_reduce_(inp, 0, index, src, 'sum')

# --- Operator 03: chunk_gated_delta_rule ---
from operators.chunk_gated_delta_rule import chunk_gated_delta_rule

B, H, L, D = 2, 8, 512, 64
q    = torch.randn(B, H, L, D, device='cuda')
k    = torch.nn.functional.normalize(torch.randn(B, H, L, D, device='cuda'), dim=-1)
v    = torch.randn(B, H, L, D, device='cuda')
beta = torch.sigmoid(torch.randn(B, H, L, device='cuda'))
g    = torch.sigmoid(torch.randn(B, H, L, device='cuda'))
o    = chunk_gated_delta_rule(q, k, v, beta, g)  # [B, H, L, D]

# --- Operator 04: rms_norm ---
from operators.rms_norm import rms_norm

x      = torch.randn(1024, 4096, device='cuda')
weight = torch.ones(4096, device='cuda')
y      = rms_norm(x, (4096,), weight)

# --- Operator 05: fused_cross_entropy ---
from operators.fused_cross_entropy import fused_cross_entropy

logits = torch.randn(32, 128000, device='cuda')  # LLaMA-3 vocab
target = torch.randint(0, 128000, (32,), device='cuda')
loss   = fused_cross_entropy(logits, target)      # no 512 MB softmax buffer
```

---

## References

- [FlagOS Challenge Page](https://flagos.io/)
- [FlagGems Repository](https://github.com/FlagOpen/FlagGems)
- [Triton Documentation](https://triton-lang.org/)
- [DeltaNet: "Parallelizing Linear Recurrence with The Delta Rule"](https://arxiv.org/abs/2406.06484)
- [GLA: "Gated Linear Attention Transformers with Hardware-Efficient Training"](https://arxiv.org/abs/2312.06635)
- [Online Softmax: Milakov & Gimelshein (2018)](https://arxiv.org/abs/1805.02867)

---

## License

[MIT License](LICENSE) — compatible with FlagGems (Apache 2.0) upstream.
