"""
Accuracy tests for chunk_gated_delta_rule.
Reference: pure-PyTorch sequential implementation in _ref_forward.

Run with:
    pytest tests/test_chunk_gated_delta_rule.py -v
"""

import pytest
import torch
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from operators.chunk_gated_delta_rule import chunk_gated_delta_rule, _ref_forward
from tests.test_utils import allclose, max_diff

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make(B, H, L, D, dtype=torch.float32):
    q    = torch.randn(B, H, L, D, dtype=dtype, device=DEVICE)
    k    = torch.randn(B, H, L, D, dtype=dtype, device=DEVICE)
    # Normalise k (required by delta rule for stability)
    k    = k / (k.norm(dim=-1, keepdim=True).clamp(min=1e-6))
    v    = torch.randn(B, H, L, D, dtype=dtype, device=DEVICE)
    beta = torch.sigmoid(torch.randn(B, H, L, dtype=dtype, device=DEVICE))
    g    = torch.sigmoid(torch.randn(B, H, L, dtype=dtype, device=DEVICE))
    return q, k, v, beta, g


# ── Reference correctness ─────────────────────────────────────────────────────

class TestDeltaRuleCorrectness:

    @pytest.mark.parametrize("D", [16, 32, 64])
    @pytest.mark.parametrize("L", [8, 16, 32, 64])
    @pytest.mark.parametrize("B,H", [(1, 1), (2, 4)])
    def test_against_reference_float32(self, B, H, L, D):
        q, k, v, beta, g = _make(B, H, L, D)
        ref = _ref_forward(q.cpu(), k.cpu(), v.cpu(), beta.cpu(), g.cpu()).to(DEVICE)
        got = chunk_gated_delta_rule(q, k, v, beta, g)
        assert allclose(got, ref, rtol=1e-3, atol=1e-3), (
            f"B={B} H={H} L={L} D={D}: max_diff={max_diff(got, ref):.4e}"
        )

    @pytest.mark.parametrize("D", [16, 32])
    def test_single_step(self, D):
        """L=1 should give h[0] directly."""
        B, H, L = 1, 1, 1
        q, k, v, beta, g = _make(B, H, L, D)

        ref = _ref_forward(q.cpu(), k.cpu(), v.cpu(), beta.cpu(), g.cpu())
        got = chunk_gated_delta_rule(q, k, v, beta, g)

        assert allclose(got, ref.to(DEVICE), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("D", [16, 32, 64])
    def test_zero_beta(self, D):
        """With β=0 the state never changes: h[t] = g[t]*g[t-1]*...*h[0] = 0."""
        B, H, L = 2, 2, 16
        q, k, v, beta, g = _make(B, H, L, D)
        beta_zero = torch.zeros_like(beta)

        got = chunk_gated_delta_rule(q, k, v, beta_zero, g)
        # All outputs should be zero (h stays at zero initial state)
        assert got.abs().max().item() < 1e-5, \
            f"With β=0 all outputs should be 0, max={got.abs().max().item()}"

    @pytest.mark.parametrize("D", [16, 32, 64])
    def test_gate_zero(self, D):
        """With g=0 the state is fully reset at every step."""
        B, H, L = 2, 2, 16
        q, k, v, beta, g = _make(B, H, L, D)
        g_zero = torch.zeros_like(g)

        ref = _ref_forward(q.cpu(), k.cpu(), v.cpu(), beta.cpu(), g_zero.cpu())
        got = chunk_gated_delta_rule(q, k, v, beta, g_zero)

        assert allclose(got, ref.to(DEVICE), rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("D", [16, 32, 64])
    def test_output_shape(self, D):
        B, H, L = 3, 4, 20
        q, k, v, beta, g = _make(B, H, L, D)
        got = chunk_gated_delta_rule(q, k, v, beta, g)
        assert got.shape == (B, H, L, D), \
            f"Expected shape ({B},{H},{L},{D}), got {got.shape}"

    @pytest.mark.parametrize("D", [16, 32, 64])
    def test_large_sequence(self, D):
        """Longer sequence — ensures loop correctness."""
        B, H, L = 1, 2, 256
        q, k, v, beta, g = _make(B, H, L, D)
        ref = _ref_forward(q.cpu(), k.cpu(), v.cpu(), beta.cpu(), g.cpu())
        got = chunk_gated_delta_rule(q, k, v, beta, g)
        assert allclose(got, ref.to(DEVICE), rtol=1e-2, atol=1e-2), (
            f"L=256 D={D}: max_diff={max_diff(got, ref.to(DEVICE)):.4e}"
        )


# ── Numerical stability ───────────────────────────────────────────────────────

class TestNumericalStability:

    def test_near_unit_gate(self):
        """Gate ≈ 1 → very slow decay → long-range dependencies."""
        B, H, L, D = 1, 1, 64, 32
        q, k, v, beta, g = _make(B, H, L, D)
        g_near_one = torch.full_like(g, 0.999)

        ref = _ref_forward(q.cpu(), k.cpu(), v.cpu(), beta.cpu(), g_near_one.cpu())
        got = chunk_gated_delta_rule(q, k, v, beta, g_near_one)

        assert not torch.isnan(got).any(), "NaN in output with near-unit gate"
        assert allclose(got, ref.to(DEVICE), rtol=1e-2, atol=1e-2)

    def test_high_beta(self):
        """β near 1 → strong delta-rule correction."""
        B, H, L, D = 1, 1, 32, 32
        q, k, v, beta, g = _make(B, H, L, D)
        beta_high = torch.full_like(beta, 0.99)

        ref = _ref_forward(q.cpu(), k.cpu(), v.cpu(), beta_high.cpu(), g.cpu())
        got = chunk_gated_delta_rule(q, k, v, beta_high, g)

        assert not torch.isnan(got).any()
        assert allclose(got, ref.to(DEVICE), rtol=1e-2, atol=1e-2)


# ── CPU fallback ──────────────────────────────────────────────────────────────

class TestCPUFallback:

    def test_cpu_reference_matches_trivial(self):
        """Single step CPU: h = β*v⊗k, o = h@q."""
        B, H, L, D = 1, 1, 1, 8
        q    = torch.ones(B, H, L, D)
        k    = torch.zeros(B, H, L, D); k[..., 0] = 1.0
        v    = torch.zeros(B, H, L, D); v[..., 0] = 2.0
        beta = torch.ones(B, H, L)
        g    = torch.zeros(B, H, L)  # no gate (g=0 means no history)

        o = chunk_gated_delta_rule(q, k, v, beta, g)
        # h = 0*0 + 1*(2-0)*[1,0,...,0]^T = [[2,0,...],[0,...]]
        # o = h @ q = [2,0,...] → sum = 2
        assert o[0, 0, 0, 0].item() == pytest.approx(2.0, abs=1e-4)
