"""
Accuracy tests for the optimized scatter_reduce operator.
All variants are validated against torch.Tensor.scatter_reduce_ (CPU reference).

Run with:
    pytest tests/test_scatter_reduce.py -v
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from operators.scatter_reduce import scatter_reduce, scatter_reduce_, scatter_reduce_out
from tests.test_utils import allclose, max_diff

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Reference implementation (always CPU + torch) ────────────────────────────

def _ref_scatter_reduce_(inp, dim, index, src, reduce, include_self=True):
    """CPU reference: in-place."""
    out = inp.clone().cpu().float()
    src_cpu  = src.cpu().float()
    idx_cpu  = index.cpu()
    out.scatter_reduce_(dim, idx_cpu, src_cpu, reduce, include_self=include_self)
    return out


def _ref_scatter_reduce(inp, dim, index, src, reduce, include_self=True):
    """CPU reference: out-of-place."""
    return _ref_scatter_reduce_(inp.clone(), dim, index, src, reduce, include_self)


# ── Fixtures / helpers ────────────────────────────────────────────────────────

REDUCES = ["sum", "prod", "amax", "amin", "mean"]
FLOAT_DTYPES = [torch.float32, torch.float16]
INT_DTYPES   = [torch.int32, torch.int64]


def _make_inputs(shape, dim, dtype=torch.float32, index_frac=0.8):
    """Return (inp, index, src) on DEVICE with sensible values."""
    inp   = torch.rand(shape, dtype=dtype, device=DEVICE) + 0.1  # avoid zeros for prod
    src   = torch.rand(index.shape if False else shape, dtype=dtype, device=DEVICE) + 0.1
    # Build index: random indices along `dim`, same shape as src
    idx_shape = list(shape)
    idx_shape[dim] = max(1, int(shape[dim] * index_frac))
    src   = torch.rand(idx_shape, dtype=dtype, device=DEVICE) + 0.1
    index = torch.randint(0, shape[dim], idx_shape, device=DEVICE)
    return inp, index, src


# ── Tests: scatter_reduce_ (in-place) ────────────────────────────────────────

class TestScatterReduceInplace:

    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("dtype",  [torch.float32])
    @pytest.mark.parametrize("shape,dim", [
        ((16,), 0),
        ((32, 16), 0),
        ((32, 16), 1),
        ((8, 16, 8), 0),
        ((8, 16, 8), 1),
        ((8, 16, 8), 2),
        ((4, 8, 8, 4), 3),
    ])
    def test_inplace_correctness(self, shape, dim, dtype, reduce):
        inp, index, src = _make_inputs(shape, dim, dtype)

        ref = _ref_scatter_reduce_(inp, dim, index, src, reduce)
        got = scatter_reduce_(inp.clone().to(DEVICE), dim, index, src, reduce)

        assert allclose(got.cpu().float(), ref, rtol=1e-3, atol=1e-3), (
            f"scatter_reduce_ reduce={reduce} shape={shape} dim={dim} "
            f"max_diff={max_diff(got.cpu().float(), ref):.4e}"
        )

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_include_self_false(self, reduce):
        shape, dim = (16, 12), 1
        inp, index, src = _make_inputs(shape, dim)

        ref = _ref_scatter_reduce_(inp, dim, index, src, reduce, include_self=False)
        got = scatter_reduce_(inp.clone().to(DEVICE), dim, index, src, reduce,
                              include_self=False)

        assert allclose(got.cpu().float(), ref, rtol=1e-2, atol=1e-2), (
            f"include_self=False reduce={reduce} max_diff={max_diff(got.cpu().float(), ref):.4e}"
        )

    def test_inplace_1d_sum(self):
        inp   = torch.zeros(8, dtype=torch.float32, device=DEVICE)
        index = torch.tensor([0, 1, 2, 0, 1, 2, 3, 3], device=DEVICE)
        src   = torch.ones(8, dtype=torch.float32, device=DEVICE)
        ref   = inp.clone().cpu().scatter_reduce_(0, index.cpu(), src.cpu(), "sum")
        got   = scatter_reduce_(inp.clone(), 0, index, src, "sum")
        assert allclose(got.cpu(), ref, rtol=1e-5, atol=1e-5)

    def test_inplace_amax_known_values(self):
        inp   = torch.zeros(4, dtype=torch.float32, device=DEVICE)
        index = torch.tensor([0, 0, 1, 2], device=DEVICE)
        src   = torch.tensor([3.0, 5.0, 2.0, 7.0], device=DEVICE)
        got   = scatter_reduce_(inp.clone(), 0, index, src, "amax", include_self=False)
        expected = torch.tensor([5.0, 2.0, 7.0, 0.0])
        assert allclose(got.cpu(), expected, rtol=1e-5, atol=1e-5)

    def test_inplace_amin_known_values(self):
        inp   = torch.full((4,), 10.0, device=DEVICE)
        index = torch.tensor([0, 0, 1, 2], device=DEVICE)
        src   = torch.tensor([3.0, 5.0, 2.0, 7.0], device=DEVICE)
        got   = scatter_reduce_(inp.clone(), 0, index, src, "amin")
        # amax(10,3)→3 at [0], amax(10,5)→5 checked: amin(10,3)=3, amin(10,5)=5
        assert got[0].item() == pytest.approx(3.0, abs=1e-4)
        assert got[1].item() == pytest.approx(2.0, abs=1e-4)
        assert got[2].item() == pytest.approx(7.0, abs=1e-4)

    def test_inplace_prod_known_values(self):
        inp   = torch.ones(3, dtype=torch.float32, device=DEVICE)
        index = torch.tensor([0, 0, 1], device=DEVICE)
        src   = torch.tensor([2.0, 3.0, 4.0], device=DEVICE)
        got   = scatter_reduce_(inp.clone(), 0, index, src, "prod")
        assert got[0].item() == pytest.approx(6.0, rel=1e-3)   # 1*2*3
        assert got[1].item() == pytest.approx(4.0, rel=1e-3)   # 1*4

    def test_inplace_mean_known_values(self):
        inp   = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        index = torch.tensor([0, 0, 1, 2], device=DEVICE)
        src   = torch.tensor([1.0, 3.0, 2.0, 5.0], device=DEVICE)
        got   = scatter_reduce_(inp.clone(), 0, index, src, "mean", include_self=False)
        assert got[0].item() == pytest.approx(2.0, abs=1e-3)  # (1+3)/2
        assert got[1].item() == pytest.approx(2.0, abs=1e-3)  # 2/1
        assert got[2].item() == pytest.approx(5.0, abs=1e-3)  # 5/1

    @pytest.mark.parametrize("shape,dim", [
        ((1024, 256), 1),
        ((512, 512),  0),
    ])
    def test_large_tensor_sum(self, shape, dim):
        inp, index, src = _make_inputs(shape, dim)
        ref = _ref_scatter_reduce_(inp, dim, index, src, "sum")
        got = scatter_reduce_(inp.clone().to(DEVICE), dim, index, src, "sum")
        assert allclose(got.cpu().float(), ref, rtol=1e-2, atol=1e-2), (
            f"Large tensor sum shape={shape} dim={dim}: max_diff={max_diff(got.cpu().float(), ref):.4e}"
        )


# ── Tests: scatter_reduce (out-of-place) ─────────────────────────────────────

class TestScatterReduceOutOfPlace:

    @pytest.mark.parametrize("reduce", REDUCES)
    @pytest.mark.parametrize("shape,dim", [
        ((16,), 0),
        ((32, 16), 1),
        ((8, 8, 8), 2),
    ])
    def test_outofplace_correctness(self, shape, dim, reduce):
        inp, index, src = _make_inputs(shape, dim)

        ref = _ref_scatter_reduce(inp, dim, index, src, reduce)
        got = scatter_reduce(inp.to(DEVICE), dim, index, src, reduce)

        assert allclose(got.cpu().float(), ref, rtol=1e-3, atol=1e-3), (
            f"scatter_reduce reduce={reduce} shape={shape} dim={dim} "
            f"max_diff={max_diff(got.cpu().float(), ref):.4e}"
        )

    def test_outofplace_does_not_modify_inp(self):
        inp, index, src = _make_inputs((16, 8), 1)
        orig = inp.clone()
        scatter_reduce(inp.to(DEVICE), 1, index, src, "sum")
        assert torch.allclose(inp.cpu(), orig.cpu()), "inp was modified (should not be)"

    @pytest.mark.parametrize("reduce", REDUCES)
    def test_outofplace_include_self_false(self, reduce):
        inp, index, src = _make_inputs((16, 12), 1)
        ref = _ref_scatter_reduce(inp, 1, index, src, reduce, include_self=False)
        got = scatter_reduce(inp.to(DEVICE), 1, index, src, reduce, include_self=False)
        assert allclose(got.cpu().float(), ref, rtol=1e-2, atol=1e-2)


# ── Tests: scatter_reduce_out ─────────────────────────────────────────────────

class TestScatterReduceOut:

    @pytest.mark.parametrize("reduce", ["sum", "amax", "amin"])
    @pytest.mark.parametrize("shape,dim", [
        ((16,), 0),
        ((32, 16), 1),
    ])
    def test_out_correctness(self, shape, dim, reduce):
        inp, index, src = _make_inputs(shape, dim)
        out = torch.empty_like(inp, device=DEVICE)

        ref = _ref_scatter_reduce(inp, dim, index, src, reduce)
        scatter_reduce_out(out, inp.to(DEVICE), dim, index, src, reduce)

        assert allclose(out.cpu().float(), ref, rtol=1e-3, atol=1e-3), (
            f"scatter_reduce_out reduce={reduce} shape={shape} dim={dim} "
            f"max_diff={max_diff(out.cpu().float(), ref):.4e}"
        )

    def test_out_tensor_is_written(self):
        inp, index, src = _make_inputs((8, 8), 1)
        out = torch.zeros(8, 8, dtype=torch.float32, device=DEVICE)
        scatter_reduce_out(out, inp.to(DEVICE), 1, index, src, "sum")
        # At least some output cells must be nonzero
        assert out.abs().sum().item() > 0


# ── Float16 numerical stability ───────────────────────────────────────────────

class TestNumericalStability:

    def test_float16_sum(self):
        shape, dim = (64, 32), 1
        inp, index, src = _make_inputs(shape, dim, dtype=torch.float16)
        ref = _ref_scatter_reduce_(inp.float(), dim, index, src.float(), "sum")
        got = scatter_reduce_(inp.clone().to(DEVICE), dim, index, src.to(DEVICE), "sum")
        assert allclose(got.cpu().float(), ref, rtol=1e-1, atol=1e-1), (
            f"float16 sum max_diff={max_diff(got.cpu().float(), ref):.4e}"
        )

    def test_float16_amax(self):
        shape, dim = (64, 32), 1
        inp, index, src = _make_inputs(shape, dim, dtype=torch.float16)
        ref = _ref_scatter_reduce_(inp.float(), dim, index, src.float(), "amax")
        got = scatter_reduce_(inp.clone().to(DEVICE), dim, index, src.to(DEVICE), "amax")
        assert allclose(got.cpu().float(), ref, rtol=1e-2, atol=1e-2)


# ── CPU fallback ──────────────────────────────────────────────────────────────

class TestCPUFallback:

    def test_cpu_scatter_reduce(self):
        inp   = torch.zeros(8, dtype=torch.float32)   # CPU tensor
        index = torch.randint(0, 4, (6,))
        src   = torch.rand(6)
        ref   = inp.clone().scatter_reduce_(0, index, src, "sum")
        got   = scatter_reduce_(inp.clone(), 0, index, src, "sum")
        assert allclose(got, ref, rtol=1e-5, atol=1e-5)
