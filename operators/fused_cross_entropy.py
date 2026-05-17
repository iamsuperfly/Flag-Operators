"""
Flag-Operators: Fused Cross-Entropy Loss (log-softmax + NLL in one pass)
FlagOS Open Computing Global Challenge Season 1 — Track 1

FlagGems relies on PyTorch's built-in cross-entropy which:
  1. Materialises the full softmax distribution (O(B×V) memory)
  2. Computes log-softmax in one kernel and NLL in another

Our kernel fuses both into a SINGLE pass, computing the online max and
log-sum-exp simultaneously with no intermediate storage.

=== Why this matters ===

For LLM training with large vocabularies (LLaMA-3: V=128K, GPT-4: V=100K+),
cross-entropy is called every forward/backward pass on tensors of shape
[B×T, V] — for a batch with B=8, T=2048, V=128K that's a 2 GB activation
just for the softmax distribution.  Our kernel never materialises it.

=== Algorithm ===

For each sample (one row of logits):

  Online one-pass max + log-sum-exp (numerically stable):
    running_max = -inf
    running_sum = 0
    for each tile of V elements:
        tile_max = max(logits[tile])
        new_max  = max(running_max, tile_max)
        running_sum = running_sum * exp(running_max - new_max) +
                      sum(exp(logits[tile] - new_max))
        running_max = new_max
    log_sum_exp = log(running_sum) + running_max

  Loss:
    loss_i = -logits[target_i] + log_sum_exp_i    (cross-entropy ≡ NLL of softmax)

=== Optimizations ===

1. SINGLE GLOBAL READ of logits (vs 2 reads in separate log-softmax + NLL)
2. ONLINE MAX+LSE: no need to store all logits; O(1) memory per sample
3. VECTORIZED tile loads (TILE_V elements per step) with autotuning
4. PARALLEL across samples (grid = ceil(B / BLOCK_B) with BLOCK_B rows per CTA)
5. Gradient kernel: recomputes softmax on-the-fly from logits — saves activation
6. ignore_index support (mask out target=-100)
7. Reduction modes: none / mean / sum
"""

import logging
import math
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Forward: fused one-pass cross-entropy loss
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"TILE_V": 512,  "BLOCK_B": 1}, num_warps=4),
        triton.Config({"TILE_V": 1024, "BLOCK_B": 1}, num_warps=8),
        triton.Config({"TILE_V": 2048, "BLOCK_B": 1}, num_warps=8),
        triton.Config({"TILE_V": 4096, "BLOCK_B": 1}, num_warps=16),
        triton.Config({"TILE_V": 512,  "BLOCK_B": 4}, num_warps=4),
        triton.Config({"TILE_V": 1024, "BLOCK_B": 4}, num_warps=8),
        triton.Config({"TILE_V": 2048, "BLOCK_B": 4}, num_warps=8),
        triton.Config({"TILE_V": 512,  "BLOCK_B": 8}, num_warps=4),
        triton.Config({"TILE_V": 1024, "BLOCK_B": 8}, num_warps=8),
    ],
    key=["V"],
)
@triton.jit
def _fused_ce_fwd_kernel(
    logits_ptr,   # [B, V]  input logits
    target_ptr,   # [B]     class indices (int64)
    loss_ptr,     # [B]     per-sample loss output
    lse_ptr,      # [B]     log-sum-exp (saved for backward)
    B, V,
    ignore_index,
    TILE_V:  tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """
    Each program processes BLOCK_B consecutive rows of the [B, V] logits matrix.

    Online max + LSE algorithm (single pass over V):
      - Maintains (running_max, running_sum_of_exp) per row
      - After all tiles: lse = log(running_sum) + running_max
      - loss = -logits[target] + lse
    """
    pid   = tl.program_id(0)
    b_ids = pid * BLOCK_B + tl.arange(0, BLOCK_B)   # [BLOCK_B]
    b_mask = b_ids < B

    # Per-row accumulators
    running_max = tl.full([BLOCK_B], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_B], dtype=tl.float32)
    target_logit = tl.zeros([BLOCK_B], dtype=tl.float32)

    # Load targets
    tgt = tl.load(target_ptr + b_ids, mask=b_mask, other=ignore_index)  # [BLOCK_B]

    # Stream through V in tiles
    n_tiles = tl.cdiv(V, TILE_V)
    for step in tl.range(0, n_tiles):
        v_start = step * TILE_V
        v_idx   = v_start + tl.arange(0, TILE_V)          # [TILE_V]
        v_mask  = v_idx < V

        # Load logits tile: shape [BLOCK_B, TILE_V]
        row_off = b_ids[:, None] * V + v_idx[None, :]     # [BLOCK_B, TILE_V]
        logits  = tl.load(
            logits_ptr + row_off,
            mask=b_mask[:, None] & v_mask[None, :],
            other=float("-inf"),
        ).to(tl.float32)                                   # [BLOCK_B, TILE_V]

        # Tile max
        tile_max = tl.max(logits, axis=1)                  # [BLOCK_B]

        # Update running max and running sum using the log-sum-exp identity:
        # exp(a) + exp(b) = exp(new_max) * [exp(a-new_max) + exp(b-new_max)]
        new_max     = tl.maximum(running_max, tile_max)
        running_sum = (running_sum * tl.exp(running_max - new_max)
                       + tl.sum(tl.exp(logits - new_max[:, None]), axis=1))
        running_max = new_max

        # Capture the logit at the target class (if it falls in this tile)
        tgt_in_tile = (tgt[:, None] == v_idx[None, :]) & v_mask[None, :]  # [BLOCK_B, TILE_V]
        target_logit += tl.sum(
            tl.where(tgt_in_tile, logits, 0.0), axis=1
        )

    # Compute log-sum-exp and loss
    lse  = tl.log(running_sum) + running_max           # [BLOCK_B]
    loss = -target_logit + lse                          # [BLOCK_B]

    # ignore_index: set loss to 0 for ignored targets
    ignored = (tgt == ignore_index)
    loss     = tl.where(ignored, 0.0, loss)

    tl.store(loss_ptr + b_ids, loss, mask=b_mask)
    tl.store(lse_ptr  + b_ids, lse,  mask=b_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Backward: fused gradient kernel  (no stored softmax needed)
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"TILE_V": 512,  "BLOCK_B": 1}, num_warps=4),
        triton.Config({"TILE_V": 1024, "BLOCK_B": 1}, num_warps=8),
        triton.Config({"TILE_V": 2048, "BLOCK_B": 1}, num_warps=8),
        triton.Config({"TILE_V": 4096, "BLOCK_B": 1}, num_warps=16),
        triton.Config({"TILE_V": 512,  "BLOCK_B": 4}, num_warps=4),
        triton.Config({"TILE_V": 1024, "BLOCK_B": 4}, num_warps=8),
    ],
    key=["V"],
)
@triton.jit
def _fused_ce_bwd_kernel(
    dlogits_ptr,  # [B, V]  output gradient wrt logits
    logits_ptr,   # [B, V]  original logits (re-read; not stored softmax)
    lse_ptr,      # [B]     log-sum-exp from forward
    target_ptr,   # [B]     class indices
    dout_ptr,     # [B] or scalar  upstream gradient
    B, V,
    ignore_index,
    n_valid,          # denominator for 'mean' reduction (or -1 for 'none'/'sum')
    SCALAR_DOUT: tl.constexpr,   # whether dout is a scalar (reduction=mean/sum)
    TILE_V:  tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """
    Gradient of cross-entropy:
      dL/d(logit_j) = softmax_j  -  1[j == target]

    We recompute softmax_j = exp(logit_j - lse) on the fly — no need to store
    the O(B×V) softmax from the forward pass.
    """
    pid    = tl.program_id(0)
    b_ids  = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_ids < B

    tgt = tl.load(target_ptr + b_ids, mask=b_mask, other=ignore_index)
    lse = tl.load(lse_ptr    + b_ids, mask=b_mask, other=0.0).to(tl.float32)

    if SCALAR_DOUT:
        dout = tl.load(dout_ptr).to(tl.float32)                # scalar
    else:
        dout = tl.load(dout_ptr + b_ids, mask=b_mask, other=0.0).to(tl.float32)

    # Scale: divide by n_valid for 'mean' reduction
    if n_valid > 0:
        dout = dout / n_valid

    ignored = (tgt == ignore_index)

    n_tiles = tl.cdiv(V, TILE_V)
    for step in tl.range(0, n_tiles):
        v_start = step * TILE_V
        v_idx   = v_start + tl.arange(0, TILE_V)
        v_mask  = v_idx < V

        row_off = b_ids[:, None] * V + v_idx[None, :]
        logits  = tl.load(
            logits_ptr + row_off,
            mask=b_mask[:, None] & v_mask[None, :],
            other=float("-inf"),
        ).to(tl.float32)

        # softmax_j = exp(logit_j - lse)
        softmax_j = tl.exp(logits - lse[:, None])

        # Subtract one-hot at target position
        is_target = (tgt[:, None] == v_idx[None, :]) & v_mask[None, :]
        softmax_j = softmax_j - is_target.to(tl.float32)

        # Scale by upstream gradient and zero out ignored samples
        grad = dout[:, None] * softmax_j
        grad = tl.where(ignored[:, None] | ~(b_mask[:, None] & v_mask[None, :]),
                        0.0, grad)

        tl.store(
            dlogits_ptr + row_off,
            grad.to(dlogits_ptr.dtype.element_ty),
            mask=b_mask[:, None] & v_mask[None, :],
        )


# ──────────────────────────────────────────────────────────────────────────────
# Autograd Function
# ──────────────────────────────────────────────────────────────────────────────

class _FusedCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, target, reduction, ignore_index, label_smoothing):
        B, V = logits.shape
        assert target.shape == (B,), f"target shape {target.shape}, expected ({B},)"

        loss = torch.empty(B, dtype=torch.float32, device=logits.device)
        lse  = torch.empty(B, dtype=torch.float32, device=logits.device)

        grid = lambda meta: (triton.cdiv(B, meta["BLOCK_B"]),)
        _fused_ce_fwd_kernel[grid](
            logits, target, loss, lse,
            B, V, ignore_index,
        )

        # Apply reduction
        valid_mask = (target != ignore_index)
        n_valid    = int(valid_mask.sum().item())

        if reduction == "none":
            out = loss
        elif reduction == "sum":
            out = loss.sum()
        else:  # mean
            out = loss.sum() / max(n_valid, 1)

        ctx.save_for_backward(logits, target, lse)
        ctx.reduction    = reduction
        ctx.ignore_index = ignore_index
        ctx.n_valid      = n_valid
        ctx.B            = B
        ctx.V            = V
        return out

    @staticmethod
    def backward(ctx, grad_output):
        logits, target, lse = ctx.saved_tensors
        B, V   = ctx.B, ctx.V
        n_valid = ctx.n_valid

        dlogits = torch.empty_like(logits)
        scalar_dout = grad_output.ndim == 0 or (ctx.reduction in ("mean", "sum"))

        dout_storage = grad_output.contiguous()

        grid = lambda meta: (triton.cdiv(B, meta["BLOCK_B"]),)
        _fused_ce_bwd_kernel[grid](
            dlogits, logits, lse, target,
            dout_storage,
            B, V, ctx.ignore_index,
            n_valid if ctx.reduction == "mean" else -1,
            SCALAR_DOUT=scalar_dout,
        )
        return dlogits, None, None, None, None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def fused_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Fused cross-entropy loss.

    Args:
        logits        : [B, V]  unnormalised logits
        target        : [B]     class indices (dtype int64)
        reduction     : 'none' | 'mean' | 'sum'
        ignore_index  : class index to ignore in loss+gradient
        label_smoothing: not yet fused (applied as post-processing for now)

    Returns:
        Scalar loss (or [B] tensor when reduction='none').
    """
    logger.debug("FLAG-OPS FUSED_CROSS_ENTROPY")

    assert logits.ndim == 2, f"logits must be 2-D [B,V], got shape {logits.shape}"
    assert reduction in ("none", "mean", "sum"), f"Unknown reduction: {reduction}"

    if not logits.is_cuda:
        return torch.nn.functional.cross_entropy(
            logits, target, reduction=reduction,
            ignore_index=ignore_index, label_smoothing=label_smoothing,
        )

    logits = logits.contiguous()
    target = target.contiguous()

    if label_smoothing > 0.0:
        # Adjust target logit contribution: (1 - ε) * CE + ε * uniform_CE
        loss_raw = _FusedCrossEntropy.apply(
            logits, target, reduction, ignore_index, 0.0
        )
        # Smooth term: -(1/V) * sum_j logits_j  +  lse
        # = lse - mean(logits)
        mean_logits = logits.float().mean(dim=-1)  # [B]
        lse = torch.logsumexp(logits.float(), dim=-1)  # [B]
        uniform_ce = lse - mean_logits
        valid_mask = (target != ignore_index)
        if reduction == "none":
            uniform_ce = uniform_ce * valid_mask
        elif reduction == "mean":
            uniform_ce = uniform_ce[valid_mask].mean()
        else:
            uniform_ce = uniform_ce[valid_mask].sum()
        return (1.0 - label_smoothing) * loss_raw + label_smoothing * uniform_ce

    return _FusedCrossEntropy.apply(
        logits, target, reduction, ignore_index, label_smoothing
    )


def cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight=None,
    reduction: str = "mean",
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Drop-in replacement for torch.nn.functional.cross_entropy.
    `weight` is passed through to PyTorch when provided (non-fused path).
    """
    if weight is not None:
        return torch.nn.functional.cross_entropy(
            logits, target, weight=weight, reduction=reduction,
            ignore_index=ignore_index, label_smoothing=label_smoothing,
        )
    return fused_cross_entropy(
        logits, target, reduction=reduction,
        ignore_index=ignore_index, label_smoothing=label_smoothing,
    )
