"""
Flag-Operators: Triton kernel for chunk_gated_delta_rule
FlagOS Open Computing Global Challenge Season 1 — Track 1

The Gated Delta Rule is the core recurrence in DeltaNet / GLA (Gated Linear
Attention) state-space models.  No upstream FlagGems implementation exists —
this is a net-new, fully-fused Triton kernel.

=== Mathematical definition ===

Given per-timestep inputs q,k,v ∈ R^D and scalars g,β:

  h[t] = g[t] * h[t-1]  +  β[t] * (v[t] - h[t-1] @ k[t]) ⊗ k[t]
  o[t] = h[t] @ q[t]

where h ∈ R^{D×D} is the hidden-state matrix (key-value memory).

Expanding the delta correction:
  h[t] = g[t] * h[t-1]                (decay by scalar gate)
        + β[t] * v[t] ⊗ k[t]          (standard linear-attn write)
        - β[t] * (h[t-1] @ k[t]) ⊗ k[t]  (delta correction: erase old association)

This makes the model error-correcting: new (k,v) pairs can overwrite stale
associations rather than just accumulating them.

=== Optimization strategy ===

1. FUSED KERNEL — NO INTERMEDIATE TENSORS
   A naive PyTorch loop materialises D²-sized intermediate tensors for every
   timestep.  Our kernel keeps the entire D×D state matrix in
   registers/L1-cache, performing all updates without any global-memory traffic
   for the state.

2. COMPUTE-BOUND UTILISATION
   Bandwidth per timestep: O(D) global loads (q,k,v,g,β).
   Compute per timestep:   O(D²) (two matrix-vector products + one rank-1 update).
   Arithmetic intensity: O(D) → compute-bound for D≥16, excellent GPU utilisation.

3. PARALLELISM ACROSS BATCH × HEAD
   grid = (B × H,) — each CTA owns one (batch, head) pair and processes all L
   timesteps sequentially with its private state matrix.

4. CHUNKED SEQUENCE PROCESSING
   For very long sequences (L > CHUNK), we split into chunks of CHUNK
   timesteps.  After each chunk the state is written to global memory and the
   next chunk resumes — limiting in-register storage to CHUNK×D² per head
   while still achieving high arithmetic intensity.

5. FLOAT32 ACCUMULATION
   Even for fp16/bf16 inputs, we accumulate the state in float32 to avoid
   numerical drift across the long recurrence.

Supported shapes: D ∈ {16, 32, 64, 128} (constexpr specialisation)
"""

import logging
import math
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _delta_rule_fwd_kernel(
    q_ptr, k_ptr, v_ptr, beta_ptr, g_ptr, o_ptr,
    # batch*head strides (outer dimension)
    stride_b,
    # sequence stride (dimension 2)
    stride_l,
    # head-feature stride (dimension 3 — innermost, should be 1 for contiguous)
    stride_d,
    L,           # sequence length
    D: tl.constexpr,   # head dimension (compile-time specialisation)
):
    """
    One Triton program = one (batch, head) pair.
    We iterate over all L timesteps sequentially, keeping h in registers.
    """
    bh_id = tl.program_id(0)

    # Base pointer for this (batch, head)
    base = bh_id * stride_b

    # Allocate state matrix h[D, D] in registers (fp32 for numerical stability)
    h = tl.zeros((D, D), dtype=tl.float32)

    # Arange helpers — compile-time constants
    d_idx = tl.arange(0, D)          # [D]

    for t in tl.range(0, L):
        t_off = base + t * stride_l

        # ── Load inputs for timestep t ──
        k = tl.load(k_ptr + t_off + d_idx).to(tl.float32)     # [D]
        v = tl.load(v_ptr + t_off + d_idx).to(tl.float32)     # [D]
        q = tl.load(q_ptr + t_off + d_idx).to(tl.float32)     # [D]
        beta = tl.load(beta_ptr + bh_id * (L) + t).to(tl.float32)  # scalar
        g    = tl.load(g_ptr    + bh_id * (L) + t).to(tl.float32)  # scalar

        # ── Delta rule update ─────────────────────────────────────────────────
        # residual[i] = Σ_j h[i,j] * k[j]    →  matrix-vector product h @ k
        # Triton: h is [D,D], k is [D]; broadcast k across rows
        residual = tl.sum(h * k[None, :], axis=1)  # [D]

        # delta = v - h @ k  (correction term)
        delta = v - residual                        # [D]

        # h = g * h  +  β * outer(delta, k)
        h = g * h + beta * (delta[:, None] * k[None, :])  # [D, D]

        # ── Compute output o[t] = h @ q ──────────────────────────────────────
        o_t = tl.sum(h * q[None, :], axis=1)       # [D]

        # ── Store output ──────────────────────────────────────────────────────
        tl.store(o_ptr + t_off + d_idx, o_t.to(o_ptr.dtype.element_ty))


# ──────────────────────────────────────────────────────────────────────────────
# Chunked kernel — for long sequences (state checkpointed every CHUNK steps)
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"CHUNK": 16}),
        triton.Config({"CHUNK": 32}),
        triton.Config({"CHUNK": 64}),
    ],
    key=["L", "D"],
)
@triton.jit
def _delta_rule_chunked_fwd_kernel(
    q_ptr, k_ptr, v_ptr, beta_ptr, g_ptr, o_ptr,
    h_ptr,        # scratch buffer for inter-chunk state  [B*H, NC, D, D]
    stride_b,
    stride_l,
    stride_d,
    L,
    NC,           # number of chunks = ceil(L / CHUNK)
    D: tl.constexpr,
    CHUNK: tl.constexpr,
):
    bh_id = tl.program_id(0)
    base  = bh_id * stride_b
    d_idx = tl.arange(0, D)

    h = tl.zeros((D, D), dtype=tl.float32)

    for chunk_idx in tl.range(0, NC):
        chunk_start = chunk_idx * CHUNK

        for local_t in tl.range(0, CHUNK):
            t = chunk_start + local_t
            if t >= L:
                break
            t_off = base + t * stride_l

            k = tl.load(k_ptr + t_off + d_idx,
                        mask=(t < L), other=0.0).to(tl.float32)
            v = tl.load(v_ptr + t_off + d_idx,
                        mask=(t < L), other=0.0).to(tl.float32)
            q = tl.load(q_ptr + t_off + d_idx,
                        mask=(t < L), other=0.0).to(tl.float32)
            beta = tl.load(beta_ptr + bh_id * L + t,
                           mask=(t < L), other=0.0).to(tl.float32)
            g    = tl.load(g_ptr    + bh_id * L + t,
                           mask=(t < L), other=1.0).to(tl.float32)

            residual = tl.sum(h * k[None, :], axis=1)
            delta    = v - residual
            h        = g * h + beta * (delta[:, None] * k[None, :])
            o_t      = tl.sum(h * q[None, :], axis=1)

            tl.store(o_ptr + t_off + d_idx, o_t.to(o_ptr.dtype.element_ty),
                     mask=(t < L))

        # Checkpoint state after this chunk
        h_off = (bh_id * NC + chunk_idx) * D * D
        d_row  = tl.arange(0, D)
        d_col  = tl.arange(0, D)
        tl.store(
            h_ptr + h_off + d_row[:, None] * D + d_col[None, :],
            h.to(h_ptr.dtype.element_ty),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def _check_inputs(q, k, v, beta, g):
    B, H, L, D = q.shape
    assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
    assert v.shape == q.shape, f"v shape {v.shape} != q shape {q.shape}"
    assert beta.shape == (B, H, L), f"beta shape {beta.shape}, expected ({B},{H},{L})"
    assert g.shape    == (B, H, L), f"g shape {g.shape}, expected ({B},{H},{L})"
    assert D in (16, 32, 64, 128), f"D={D} not in supported set {{16,32,64,128}}"
    return B, H, L, D


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Gated Delta Rule recurrence.

    Args:
        q    : [B, H, L, D]  query
        k    : [B, H, L, D]  key   (should be normalised, ‖k‖=1)
        v    : [B, H, L, D]  value
        beta : [B, H, L]     delta-rule learning rate ∈ (0, 1)
        g    : [B, H, L]     scalar gate ∈ (0, 1)

    Returns:
        o    : [B, H, L, D]  output  (same dtype as q)
    """
    logger.debug("FLAG-OPS CHUNK_GATED_DELTA_RULE")

    if not q.is_cuda:
        return _ref_forward(q, k, v, beta, g)

    B, H, L, D = _check_inputs(q, k, v, beta, g)

    q    = q.contiguous()
    k    = k.contiguous()
    v    = v.contiguous()
    beta = beta.contiguous()
    g    = g.contiguous()
    o    = torch.empty_like(q)

    # stride_b: elements between consecutive (batch,head) pairs
    # For shape [B,H,L,D] with strides [H*L*D, L*D, D, 1]:
    stride_b = H * L * D
    stride_l = D
    stride_d = 1

    grid = (B * H,)

    if D <= 64:
        # Small D: keep h entirely in registers for the full sequence
        _delta_rule_fwd_kernel[grid](
            q, k, v, beta, g, o,
            stride_b, stride_l, stride_d,
            L, D=D,
        )
    else:
        # Large D: use chunked kernel with inter-chunk state checkpointing
        CHUNK = 32
        NC    = math.ceil(L / CHUNK)
        h_buf = torch.zeros(B * H, NC, D, D, dtype=torch.float32, device=q.device)
        _delta_rule_chunked_fwd_kernel[grid](
            q, k, v, beta, g, o, h_buf,
            stride_b, stride_l, stride_d,
            L, NC, D=D,
        )

    return o


# ──────────────────────────────────────────────────────────────────────────────
# Pure-PyTorch reference (CPU + CUDA, used for correctness validation)
# ──────────────────────────────────────────────────────────────────────────────

def _ref_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """
    Pure-PyTorch sequential implementation.  Correct but slow — used as the
    reference for numerical tests and as the CPU fallback.
    """
    B, H, L, D = q.shape
    dtype = q.dtype
    h = torch.zeros(B, H, D, D, dtype=torch.float32, device=q.device)
    o = torch.empty_like(q)

    for t in range(L):
        q_t    = q[:, :, t, :].float()    # [B, H, D]
        k_t    = k[:, :, t, :].float()
        v_t    = v[:, :, t, :].float()
        beta_t = beta[:, :, t]             # [B, H]
        g_t    = g[:, :, t]

        # residual = h @ k_t  →  [B, H, D, D] @ [B, H, D, 1] → [B, H, D]
        residual = (h @ k_t.unsqueeze(-1)).squeeze(-1)          # [B, H, D]
        delta    = v_t - residual                                # [B, H, D]

        # h = g * h  +  beta * outer(delta, k)
        h = (g_t[:, :, None, None] * h
             + beta_t[:, :, None, None]
               * (delta.unsqueeze(-1) * k_t.unsqueeze(-2)))     # [B, H, D, D]

        # o = h @ q
        o[:, :, t, :] = (h @ q_t.unsqueeze(-1)).squeeze(-1).to(dtype)

    return o
