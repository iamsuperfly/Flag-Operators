"""
Accuracy tests for the fused cross-entropy operator.
Reference: torch.nn.functional.cross_entropy.

Run with:
    pytest tests/test_fused_cross_entropy.py -v
"""

import pytest
import torch
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from operators.fused_cross_entropy import fused_cross_entropy, cross_entropy_loss
from tests.test_utils import allclose, max_diff

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _ref(logits, target, reduction="mean", ignore_index=-100, label_smoothing=0.0):
    return F.cross_entropy(
        logits.float(), target,
        reduction=reduction,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


# ── Forward: correctness across shapes ───────────────────────────────────────

class TestCrossEntropyForward:

    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("B,V", [
        (1, 10),
        (8, 100),
        (32, 1000),
        (128, 50257),     # GPT-2 vocab
    ])
    def test_correctness(self, B, V, reduction):
        logits = torch.randn(B, V, device=DEVICE)
        target = torch.randint(0, V, (B,), device=DEVICE)

        ref = _ref(logits.cpu(), target.cpu(), reduction)
        got = fused_cross_entropy(logits, target, reduction=reduction)

        tol = {"rtol": 1e-3, "atol": 1e-3}
        if reduction == "none":
            assert allclose(got.cpu().float(), ref, **tol), \
                f"B={B} V={V} reduction={reduction}: max_diff={max_diff(got.cpu().float(), ref):.4e}"
        else:
            assert abs(got.cpu().float().item() - ref.item()) < 1e-2, \
                f"B={B} V={V} reduction={reduction}: got={got.item():.6f} ref={ref.item():.6f}"

    @pytest.mark.parametrize("B,V", [(16, 512), (32, 1024)])
    def test_ignore_index(self, B, V):
        logits = torch.randn(B, V, device=DEVICE)
        target = torch.randint(0, V, (B,), device=DEVICE)
        target[0] = -100  # ignore first sample

        ref = _ref(logits.cpu(), target.cpu(), reduction="mean", ignore_index=-100)
        got = fused_cross_entropy(logits, target, reduction="mean", ignore_index=-100)

        assert abs(got.cpu().item() - ref.item()) < 1e-2, \
            f"ignore_index: got={got.item():.6f} ref={ref.item():.6f}"

    def test_all_ignored(self):
        """All samples ignored → loss should be 0."""
        B, V = 8, 100
        logits = torch.randn(B, V, device=DEVICE)
        target = torch.full((B,), -100, device=DEVICE)
        got = fused_cross_entropy(logits, target, reduction="mean", ignore_index=-100)
        assert got.item() == pytest.approx(0.0, abs=1e-4)

    def test_label_smoothing(self):
        B, V = 32, 1000
        logits = torch.randn(B, V, device=DEVICE)
        target = torch.randint(0, V, (B,), device=DEVICE)

        ref = _ref(logits.cpu(), target.cpu(), label_smoothing=0.1)
        got = fused_cross_entropy(logits, target, label_smoothing=0.1)

        assert abs(got.cpu().item() - ref.item()) < 5e-2, \
            f"label_smoothing: got={got.item():.6f} ref={ref.item():.6f}"

    def test_perfect_prediction(self):
        """If logits[target] >> rest, loss ≈ 0."""
        B, V = 4, 100
        logits = torch.full((B, V), -10.0, device=DEVICE)
        target = torch.randint(0, V, (B,), device=DEVICE)
        for b in range(B):
            logits[b, target[b]] = 100.0  # very confident

        loss = fused_cross_entropy(logits, target)
        assert loss.item() < 1e-3, f"Expected near-zero loss, got {loss.item()}"

    def test_uniform_logits(self):
        """Uniform logits → loss ≈ log(V)."""
        B, V = 8, 1000
        logits = torch.zeros(B, V, device=DEVICE)
        target = torch.randint(0, V, (B,), device=DEVICE)

        got = fused_cross_entropy(logits, target)
        import math
        expected = math.log(V)
        assert abs(got.item() - expected) < 1e-3, \
            f"Uniform logits: got={got.item():.4f}, expected log({V})={expected:.4f}"

    @pytest.mark.parametrize("V", [32768, 131072])
    def test_large_vocab(self, V):
        """Large vocabulary — checks correctness and numerical stability."""
        B = 4
        logits = torch.randn(B, V, device=DEVICE)
        target = torch.randint(0, V, (B,), device=DEVICE)

        ref = _ref(logits.cpu(), target.cpu())
        got = fused_cross_entropy(logits, target)

        assert abs(got.cpu().item() - ref.item()) < 0.1, \
            f"Large vocab V={V}: got={got.item():.6f} ref={ref.item():.6f}"


# ── Backward: gradient correctness ───────────────────────────────────────────

class TestCrossEntropyBackward:

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("B,V", [(4, 100), (8, 1000), (16, 10000)])
    def test_gradient_correctness(self, B, V, reduction):
        logits = torch.randn(B, V, device=DEVICE, requires_grad=True)
        target = torch.randint(0, V, (B,), device=DEVICE)

        # Reference gradient
        logits_ref = logits.detach().clone().requires_grad_(True)
        ref_loss = _ref(logits_ref.float(), target, reduction)
        ref_loss.backward()
        ref_grad = logits_ref.grad.float()

        # Our gradient
        logits_got = logits.detach().clone().requires_grad_(True)
        got_loss = fused_cross_entropy(logits_got, target, reduction=reduction)
        got_loss.backward()
        got_grad = logits_got.grad.float()

        assert allclose(got_grad.cpu(), ref_grad.cpu(), rtol=1e-3, atol=1e-3), (
            f"Gradient B={B} V={V} reduction={reduction}: "
            f"max_diff={max_diff(got_grad.cpu(), ref_grad.cpu()):.4e}"
        )

    def test_gradient_ignore_index(self):
        B, V = 8, 100
        logits = torch.randn(B, V, device=DEVICE, requires_grad=True)
        target = torch.randint(0, V, (B,), device=DEVICE)
        target[0] = -100

        logits_got = logits.detach().clone().requires_grad_(True)
        fused_cross_entropy(logits_got, target, reduction="mean", ignore_index=-100).backward()

        # First row gradient must be zero (ignored sample)
        assert logits_got.grad[0].abs().max().item() < 1e-5, \
            "Gradient for ignored sample must be zero"

    def test_gradient_sum_none(self):
        """'none' reduction + manual sum should match 'sum' reduction directly."""
        B, V = 8, 100
        logits = torch.randn(B, V, device=DEVICE)
        target = torch.randint(0, V, (B,), device=DEVICE)

        logits_a = logits.clone().requires_grad_(True)
        fused_cross_entropy(logits_a, target, reduction="sum").backward()

        logits_b = logits.clone().requires_grad_(True)
        fused_cross_entropy(logits_b, target, reduction="none").sum().backward()

        assert allclose(logits_a.grad.cpu(), logits_b.grad.cpu(), rtol=1e-4, atol=1e-4)


# ── drop-in API ───────────────────────────────────────────────────────────────

class TestDropInAPI:

    def test_cross_entropy_loss_matches(self):
        B, V = 16, 500
        logits = torch.randn(B, V, device=DEVICE)
        target = torch.randint(0, V, (B,), device=DEVICE)

        ref = _ref(logits.cpu(), target.cpu())
        got = cross_entropy_loss(logits, target)
        assert abs(got.cpu().item() - ref.item()) < 1e-2


# ── CPU fallback ──────────────────────────────────────────────────────────────

class TestCPUFallback:

    def test_cpu_cross_entropy(self):
        logits = torch.randn(8, 100)   # CPU tensor
        target = torch.randint(0, 100, (8,))
        ref    = _ref(logits, target)
        got    = fused_cross_entropy(logits, target)
        assert abs(got.item() - ref.item()) < 1e-4
