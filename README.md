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
├── operators/          # Our optimized Triton operator implementations
│   └── *.py           # One file per operator
├── tests/             # Accuracy & correctness tests (pytest)
│   └── test_*.py
├── benchmarks/        # Performance benchmarks vs baseline
│   └── bench_*.py
├── docs/              # Technical report and design notes
│   └── *.md
├── FlagGems/          # Official FlagGems repo (reference, not modified)
└── README.md
```

---

## Operators Under Development

Track 1 covers 20 operators from the FlagGems official task list. Each operator below is being optimized:

| # | Operator | Status | Notes |
|---|----------|--------|-------|
| 1 | TBD | Pending | — |
| 2 | TBD | Pending | — |
| … | … | … | … |

> Status will update as operators are completed and benchmarked.

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
python benchmarks/bench_<operator>.py
```

---

## Progress

| Phase | Status |
|-------|--------|
| Environment Setup | ✅ Done |
| FlagGems Reference Clone | ✅ Done |
| Operator 1 | 🔄 In Progress |
| Operator 2–20 | ⏳ Pending |

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
