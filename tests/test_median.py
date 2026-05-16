"""
Accuracy tests for the optimized median operator.
Validates against torch.median (CPU reference) across dtypes, shapes, and dims.

Run with:
    pytest tests/test_median.py -v
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from operators.median import median
from tests.test_utils import allclose, max_diff

# ── Fixtures ──────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPES = [torch.float32, torch.float16]  # bfloat16 added if supported


def _ref_median_global(t):
    return torch.median(t.cpu().float()).to(t.dtype)


def _ref_median_dim(t, dim, keepdim):
    return torch.median(t.cpu().float(), dim=dim, keepdim=keepdim)


# ── Global median (no dim) ────────────────────────────────────────────────────

class TestMedianGlobal:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("shape", [
        (1,), (7,), (16,), (127,), (128,), (1024,), (4096,),
        (3, 5), (16, 32), (4, 128, 128),
    ])
    def test_global_median_value(self, shape, dtype):
        t = torch.randn(*shape, dtype=dtype, device=DEVICE)
        got = median(t)
        ref = _ref_median_global(t)
        assert allclose(got.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2), (
            f"shape={shape} dtype={dtype}: got={got.item():.6f} ref={ref.item():.6f} "
            f"diff={abs(got.item()-ref.item()):.2e}"
        )

    def test_global_median_odd_count(self):
        """Odd number of elements — median is exact middle."""
        t = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0], device=DEVICE)
        assert median(t).item() == pytest.approx(3.0)

    def test_global_median_even_count(self):
        """Even N — PyTorch returns lower median."""
        t = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        # PyTorch lower-median = element at index (N-1)//2 of sorted = 2.0
        assert median(t).item() == pytest.approx(2.0)

    def test_global_median_single(self):
        t = torch.tensor([42.0], device=DEVICE)
        assert median(t).item() == pytest.approx(42.0)

    def test_global_median_negative(self):
        t = torch.tensor([-5.0, -1.0, -3.0], device=DEVICE)
        assert median(t).item() == pytest.approx(-3.0)


# ── Dim-reduction median ──────────────────────────────────────────────────────

class TestMedianDim:
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("shape,dim", [
        ((7,), 0),
        ((16,), 0),
        ((3, 5), 0),
        ((3, 5), 1),
        ((3, 5), -1),
        ((4, 8, 16), 2),
        ((4, 8, 16), 1),
        ((4, 8, 16), 0),
        ((2, 4, 8, 16), -1),
    ])
    def test_dim_values(self, shape, dim, dtype):
        t = torch.randn(*shape, dtype=dtype, device=DEVICE)
        got = median(t, dim=dim)
        ref = _ref_median_dim(t.float(), dim, keepdim=False)

        assert got.values.shape == ref.values.shape, (
            f"values shape mismatch: {got.values.shape} vs {ref.values.shape}"
        )
        assert allclose(got.values.cpu().float(), ref.values.cpu().float(), rtol=1e-3, atol=1e-3), (
            f"shape={shape} dim={dim}: max_diff={max_diff(got.values.cpu().float(), ref.values.cpu().float()):.2e}"
        )

    @pytest.mark.parametrize("shape,dim", [
        ((7,), 0),
        ((3, 5), 1),
        ((4, 8, 16), 2),
    ])
    def test_dim_indices(self, shape, dim):
        """Indices must point to the correct value in the original tensor."""
        t = torch.randn(*shape, dtype=torch.float32, device=DEVICE)
        got = median(t, dim=dim)
        ref = _ref_median_dim(t, dim, keepdim=False)

        # Values at returned indices must match returned values
        gathered = t.gather(dim, got.indices.clamp(0, t.shape[dim] - 1).unsqueeze(dim)).squeeze(dim)
        assert allclose(gathered.cpu().float(), got.values.cpu().float(), rtol=1e-5, atol=1e-5), (
            f"Index consistency failed for shape={shape} dim={dim}"
        )

    @pytest.mark.parametrize("shape,dim", [
        ((3, 5), 1),
        ((4, 8), 0),
    ])
    def test_keepdim(self, shape, dim):
        t = torch.randn(*shape, dtype=torch.float32, device=DEVICE)
        got = median(t, dim=dim, keepdim=True)
        ref = _ref_median_dim(t, dim, keepdim=True)
        assert got.values.shape == ref.values.shape, (
            f"keepdim shape: {got.values.shape} vs {ref.values.shape}"
        )

    def test_dim_large_n(self):
        """Test with N > SMALL_N_THRESHOLD to exercise the radix-select path."""
        t = torch.randn(8, 1024, dtype=torch.float32, device=DEVICE)
        got = median(t, dim=1)
        ref = _ref_median_dim(t, 1, keepdim=False)
        assert allclose(got.values.cpu(), ref.values.cpu(), rtol=1e-3, atol=1e-3), (
            f"Large-N median mismatch: max_diff={max_diff(got.values.cpu(), ref.values.cpu()):.2e}"
        )

    def test_dim_1d(self):
        t = torch.tensor([5.0, 3.0, 1.0, 4.0, 2.0], device=DEVICE)
        got = median(t, dim=0)
        assert got.values.item() == pytest.approx(3.0)


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestMedianEdge:
    def test_all_same_values(self):
        t = torch.full((16,), 7.0, device=DEVICE)
        assert median(t).item() == pytest.approx(7.0)

    def test_two_elements(self):
        t = torch.tensor([2.0, 8.0], device=DEVICE)
        # lower-median = sorted[0] = 2.0
        assert median(t).item() == pytest.approx(2.0)

    def test_float16_precision(self):
        t = torch.randn(64, dtype=torch.float16, device=DEVICE)
        got = median(t)
        ref = _ref_median_global(t)
        # Allow loose tolerance for fp16
        assert abs(got.item() - ref.item()) < 0.1, (
            f"fp16 global median: got={got.item()} ref={ref.item()}"
        )

    def test_3d_dim_reduction(self):
        t = torch.randn(2, 3, 7, dtype=torch.float32, device=DEVICE)
        for dim in [0, 1, 2, -1]:
            got = median(t, dim=dim)
            ref = _ref_median_dim(t, dim, keepdim=False)
            assert allclose(got.values.cpu(), ref.values.cpu(), rtol=1e-3, atol=1e-3)


# ── CPU fallback ──────────────────────────────────────────────────────────────

class TestMedianCPU:
    def test_cpu_global(self):
        t = torch.randn(10)  # CPU tensor
        got = median(t)
        ref = torch.median(t)
        assert allclose(got, ref, rtol=1e-5, atol=1e-5)

    def test_cpu_dim(self):
        t = torch.randn(4, 5)  # CPU tensor
        got = median(t, dim=1)
        ref = torch.median(t, dim=1)
        assert allclose(got.values, ref.values, rtol=1e-5, atol=1e-5)
