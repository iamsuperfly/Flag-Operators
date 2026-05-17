"""
Accuracy tests for the optimized rms_norm operator.
Reference: torch.nn.functional.rms_norm (PyTorch native).

Run with:
    pytest tests/test_rms_norm.py -v
"""

import pytest
import torch
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from operators.rms_norm import rms_norm, rms_norm_forward, rms_norm_out
from tests.test_utils import allclose, max_diff

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _ref(x, norm_shape, w, eps=1e-5):
    """Reference: PyTorch native rms_norm."""
    # torch.nn.functional.rms_norm requires PyTorch >= 2.4
    try:
        return F.rms_norm(x, norm_shape, weight=w, eps=eps)
    except AttributeError:
        # Manual fallback for older PyTorch
        N = x.shape[-1]
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        return ((x.float() / rms) * w.float()).to(x.dtype)


# ── Basic correctness ─────────────────────────────────────────────────────────

class TestRmsNormCorrectness:

    @pytest.mark.parametrize("N", [64, 128, 256, 512, 1024, 2048, 4096])
    @pytest.mark.parametrize("M", [1, 4, 16, 64])
    def test_small_n_float32(self, M, N):
        x = torch.randn(M, N, device=DEVICE)
        w = torch.ones(N, device=DEVICE)

        ref = _ref(x.cpu(), (N,), w.cpu()).to(DEVICE)
        got = rms_norm(x, (N,), w)

        assert allclose(got, ref, rtol=1e-4, atol=1e-4), (
            f"M={M} N={N}: max_diff={max_diff(got, ref):.4e}"
        )

    @pytest.mark.parametrize("N", [5000, 8192, 16384, 32768])
    def test_large_n_float32(self, N):
        M = 8
        x = torch.randn(M, N, device=DEVICE)
        w = torch.randn(N, device=DEVICE)

        ref = _ref(x.cpu(), (N,), w.cpu()).to(DEVICE)
        got = rms_norm(x, (N,), w)

        assert allclose(got, ref, rtol=1e-3, atol=1e-3), (
            f"Large N={N}: max_diff={max_diff(got, ref):.4e}"
        )

    @pytest.mark.parametrize("N", [128, 1024, 4096])
    def test_float16(self, N):
        M = 16
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
        w = torch.ones(N, device=DEVICE, dtype=torch.float16)

        ref = _ref(x.float().cpu(), (N,), w.float().cpu()).to(DEVICE).half()
        got = rms_norm(x, (N,), w)

        assert allclose(got.float(), ref.float(), rtol=1e-1, atol=1e-1), (
            f"float16 N={N}: max_diff={max_diff(got.float(), ref.float()):.4e}"
        )

    @pytest.mark.parametrize("N", [128, 1024, 4096])
    def test_bfloat16(self, N):
        if not torch.cuda.is_available():
            pytest.skip("bfloat16 requires CUDA")
        M = 16
        x = torch.randn(M, N, device=DEVICE, dtype=torch.bfloat16)
        w = torch.ones(N, device=DEVICE, dtype=torch.bfloat16)

        ref = _ref(x.float().cpu(), (N,), w.float().cpu()).to(DEVICE).bfloat16()
        got = rms_norm(x, (N,), w)

        assert allclose(got.float(), ref.float(), rtol=1e-1, atol=1e-1)

    def test_non_unit_weight(self):
        M, N = 16, 512
        x = torch.randn(M, N, device=DEVICE)
        w = torch.rand(N, device=DEVICE) * 2.0  # random weights

        ref = _ref(x.cpu(), (N,), w.cpu()).to(DEVICE)
        got = rms_norm(x, (N,), w)

        assert allclose(got, ref, rtol=1e-4, atol=1e-4)

    def test_3d_input(self):
        """Batch × Seq × Hidden — standard transformer shape."""
        B, T, H = 4, 128, 512
        x = torch.randn(B, T, H, device=DEVICE)
        w = torch.ones(H, device=DEVICE)

        ref = _ref(x.view(-1, H).cpu(), (H,), w.cpu()).view(B, T, H).to(DEVICE)
        got = rms_norm(x.view(-1, H), (H,), w).view(B, T, H)

        assert allclose(got, ref, rtol=1e-4, atol=1e-4)

    def test_single_row(self):
        N = 1024
        x = torch.randn(1, N, device=DEVICE)
        w = torch.ones(N, device=DEVICE)
        ref = _ref(x.cpu(), (N,), w.cpu()).to(DEVICE)
        got = rms_norm(x, (N,), w)
        assert allclose(got, ref, rtol=1e-5, atol=1e-5)

    def test_eps_effect(self):
        """Small eps vs large eps should differ for near-zero input."""
        M, N = 4, 256
        x = torch.randn(M, N, device=DEVICE) * 1e-3  # near zero
        w = torch.ones(N, device=DEVICE)

        got_small = rms_norm(x, (N,), w, eps=1e-8)
        got_large = rms_norm(x, (N,), w, eps=1.0)

        # Results should differ (eps dominates when x is tiny)
        assert not torch.allclose(got_small, got_large)

    @pytest.mark.parametrize("N", [64, 512, 4096])
    def test_inv_rms_saved(self, N):
        """rms_norm_forward must return correct inv_rms for backward."""
        M = 8
        x = torch.randn(M, N, device=DEVICE)
        w = torch.ones(N, device=DEVICE)

        _, inv_rms = rms_norm_forward(x, (N,), w)

        # Verify: rms = 1/inv_rms matches manual computation
        rms_manual = x.float().pow(2).mean(dim=-1).add(1e-5).sqrt()
        inv_manual  = 1.0 / rms_manual

        assert allclose(inv_rms.cpu(), inv_manual.cpu(), rtol=1e-4, atol=1e-4)


# ── rms_norm_out variant ──────────────────────────────────────────────────────

class TestRmsNormOut:

    def test_out_matches_forward(self):
        M, N = 8, 512
        x = torch.randn(M, N, device=DEVICE)
        w = torch.ones(N, device=DEVICE)

        y_ref = rms_norm(x, (N,), w)
        y_out = torch.empty_like(x)
        rms_norm_out(y_out, x, (N,), w)

        assert torch.allclose(y_out, y_ref, atol=1e-6)

    def test_out_does_not_modify_x(self):
        M, N = 4, 256
        x    = torch.randn(M, N, device=DEVICE)
        orig = x.clone()
        out  = torch.empty_like(x)
        rms_norm_out(out, x, (N,), torch.ones(N, device=DEVICE))
        assert torch.allclose(x, orig)


# ── Boundary cases ────────────────────────────────────────────────────────────

class TestBoundaryCases:

    def test_identical_rows_identical_outputs(self):
        M, N = 8, 256
        x = torch.randn(1, N, device=DEVICE).expand(M, N).contiguous()
        w = torch.ones(N, device=DEVICE)
        y = rms_norm(x, (N,), w)
        # All rows should be identical
        for i in range(1, M):
            assert torch.allclose(y[0], y[i], atol=1e-5), f"Row {i} differs from row 0"

    def test_non_power_of_two_N(self):
        M = 4
        for N in [100, 513, 1001, 3000]:
            x = torch.randn(M, N, device=DEVICE)
            w = torch.ones(N, device=DEVICE)
            ref = _ref(x.cpu(), (N,), w.cpu()).to(DEVICE)
            got = rms_norm(x, (N,), w)
            assert allclose(got, ref, rtol=1e-3, atol=1e-3), \
                f"Non-power-of-2 N={N}: max_diff={max_diff(got, ref):.4e}"

    @pytest.mark.parametrize("N", [128, 4096, 8192])
    def test_all_ones_input(self, N):
        M = 4
        x = torch.ones(M, N, device=DEVICE)
        w = torch.ones(N, device=DEVICE)
        # rms(ones) = 1 → y = x = 1
        y = rms_norm(x, (N,), w)
        assert allclose(y, x, rtol=1e-5, atol=1e-5)
