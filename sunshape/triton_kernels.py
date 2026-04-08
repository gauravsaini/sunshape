"""
SunShape block-VQ attention score kernels.

Provides a unified entry point for computing attention scores against
SunShape block-quantized keys. This module includes a Triton fast path
for the primary deployment target: a single decode-step query tile against
SunShape block-VQ keys, computing scores *directly* from quantized indices
and pre-decoded centroids without materializing the full dequantized key
matrix. Any unsupported shape or missing Triton/CUDA runtime falls back to
the PyTorch reference implementation.

SunShape block layout
---------------------
- head_dim is partitioned into ``n_blocks`` contiguous blocks of ``block_dim``.
- The codebook is shaped ``[n_blocks, n_centroids, block_dim]``.
- Quantized keys are stored as index tensors ``[n_tokens, n_blocks]`` where
  each element selects a centroid from the per-block codebook.

The Triton kernel fuses the lookup + dot-product path so the memory traffic
bottleneck shifts from reading ``[n_tokens, head_dim]`` FP32 keys to reading
``[n_tokens, n_blocks]`` INT32 indices + a much smaller codebook.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("sunshape.triton")
_TRITON_FAILURE_LOGGED = False

_HAS_TRITON = False
try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    _HAS_TRITON = True
except ImportError:
    pass


# ------------------------------------------------------------------
# Triton kernel: fused block-VQ dequant + matmul (LEGACY)
# ------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _sunshape_scores_kernel(
        query_ptr,
        indices_ptr,
        centroids_ptr,
        out_ptr,
        # shape scalars
        n_qh,
        n_tokens,
        n_centroids,
        # leading-dim strides (inner dims = 1 for contiguous tensors)
        q_stride_h,      # query:   [n_qh, head_dim]     -> stride(0)
        idx_stride_t,    # indices: [n_tokens, n_blocks]  -> stride(0)
        out_stride_h,    # output:  [n_qh, n_tokens]      -> stride(0)
        # constexpr tile sizes
        n_blocks: tl.constexpr,
        block_dim: tl.constexpr,
        block_q: tl.constexpr,
        block_t: tl.constexpr,
    ):
        """
        Legacy kernel: fused dequant + dot product.
        Kept as fallback; prefer _sunshape_precomputed_scores_kernel.
        """
        pid_q = tl.program_id(0)
        pid_t = tl.program_id(1)

        q_offsets = pid_q * block_q + tl.arange(0, block_q)   # [block_q]
        t_offsets = pid_t * block_t + tl.arange(0, block_t)   # [block_t]

        q_mask = q_offsets < n_qh
        t_mask = t_offsets < n_tokens

        acc = tl.zeros((block_q, block_t), dtype=tl.float32)

        for b in range(0, n_blocks):
            token_indices = tl.load(
                indices_ptr + t_offsets * idx_stride_t + b,
                mask=t_mask,
                other=0,
            ).to(tl.int32)

            cent_base = b * (n_centroids * block_dim)
            for d in range(0, block_dim):
                global_d = b * block_dim + d

                q_vals = tl.load(
                    query_ptr + q_offsets * q_stride_h + global_d,
                    mask=q_mask,
                    other=0.0,
                ).to(tl.float32)

                c_vals = tl.load(
                    centroids_ptr + cent_base + token_indices * block_dim + d,
                    mask=t_mask,
                    other=0.0,
                ).to(tl.float32)

                acc += q_vals[:, None] * c_vals[None, :]

        tl.store(
            out_ptr
            + q_offsets[:, None] * out_stride_h
            + t_offsets[None, :],
            acc,
            mask=q_mask[:, None] & t_mask[None, :],
        )


# ------------------------------------------------------------------
# Triton kernel: PRECOMPUTED query-centroid dot scores (NEW)
# ------------------------------------------------------------------
# Key insight: since SunShape uses a fixed block codebook, we can
# precompute dot(query_block_b, centroid[b, c]) for ALL centroids
# outside the kernel. The kernel then does PURE GATHER + ADD:
#   score[q, t] = sum_b  qdots[q, b, indices[t, b]]
# This eliminates the inner block_dim loop entirely.
# TurboQuant CANNOT do this because its scalar codebook has only
# 2 entries per coordinate — the precompute doesn't amortize.
# SunShape has 256 entries per 8-dim BLOCK — precompute is small
# and the gather kernel is dramatically faster.
# ------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _sunshape_precomputed_scores_kernel(
        qdots_ptr,       # [n_qh, n_blocks, n_centroids]  precomputed
        indices_ptr,     # [n_tokens, n_blocks]            int32
        out_ptr,         # [n_qh, n_tokens]                float32
        # shape scalars
        n_qh,
        n_tokens,
        n_centroids,
        # strides
        qd_stride_h,     # qdots stride(0) = n_blocks * n_centroids
        qd_stride_b,     # qdots stride(1) = n_centroids
        idx_stride_t,    # indices stride(0) = n_blocks
        out_stride_h,    # output stride(0) = n_tokens
        # constexpr
        n_blocks: tl.constexpr,
        block_q: tl.constexpr,
        block_t: tl.constexpr,
    ):
        """
        Pure-gather attention score kernel.

        ``scores[q, t] = sum_b qdots[q, b, indices[t, b]]``

        No FMA in the hot loop — just index loads and table lookups.
        This is the structural advantage of block VQ over scalar quant:
        the block dot products can be precomputed and the kernel reduces
        to pure memory gather.

        Grid: (cdiv(n_qh, block_q), cdiv(n_tokens, block_t))
        """
        pid_q = tl.program_id(0)
        pid_t = tl.program_id(1)

        q_offsets = pid_q * block_q + tl.arange(0, block_q)   # [block_q]
        t_offsets = pid_t * block_t + tl.arange(0, block_t)   # [block_t]

        q_mask = q_offsets < n_qh
        t_mask = t_offsets < n_tokens

        acc = tl.zeros((block_q, block_t), dtype=tl.float32)

        for b in range(0, n_blocks):
            # Load centroid indices for this block: [block_t]
            token_indices = tl.load(
                indices_ptr + t_offsets * idx_stride_t + b,
                mask=t_mask,
                other=0,
            ).to(tl.int32)

            # Gather precomputed dots: qdots[q, b, idx] -> [block_q, block_t]
            # qdots layout: [n_qh, n_blocks, n_centroids], contiguous
            # address = q * qd_stride_h + b * qd_stride_b + idx
            dot_vals = tl.load(
                qdots_ptr
                + q_offsets[:, None] * qd_stride_h
                + b * qd_stride_b
                + token_indices[None, :],
                mask=q_mask[:, None] & t_mask[None, :],
                other=0.0,
            )

            acc += dot_vals

        tl.store(
            out_ptr
            + q_offsets[:, None] * out_stride_h
            + t_offsets[None, :],
            acc,
            mask=q_mask[:, None] & t_mask[None, :],
        )


# ------------------------------------------------------------------
# Packed-index Triton kernel variant (for bit-packed storage)
# ------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _sunshape_packed_scores_kernel(
        query_ptr,
        packed_ptr,
        centroids_ptr,
        out_ptr,
        # shape scalars
        n_qh,
        n_tokens,
        n_centroids,
        n_blocks,
        # leading-dim strides
        q_stride_h,
        packed_stride_t,
        out_stride_h,
        # constexpr params
        bits: tl.constexpr,
        vals_per_elem: tl.constexpr,
        block_dim: tl.constexpr,
        block_q: tl.constexpr,
        block_t: tl.constexpr,
    ):
        """
        Variant for bit-packed index storage (future use).

        When indices are packed into wider integers (e.g. 8-bit indices packed
        4-per-int32), this kernel unpacks on the fly.
        """
        pid_q = tl.program_id(0)
        pid_t = tl.program_id(1)

        q_offsets = pid_q * block_q + tl.arange(0, block_q)
        t_offsets = pid_t * block_t + tl.arange(0, block_t)

        q_mask = q_offsets < n_qh
        t_mask = t_offsets < n_tokens

        acc = tl.zeros((block_q, block_t), dtype=tl.float32)
        index_mask = (1 << bits) - 1
        packed_dim = (n_blocks + vals_per_elem - 1) // vals_per_elem

        for p in range(0, packed_dim):
            # packed is contiguous [n_tokens, packed_dim], inner stride = 1
            packed_vals = tl.load(
                packed_ptr + t_offsets * packed_stride_t + p,
                mask=t_mask,
                other=0,
            ).to(tl.int32)

            for v in range(0, vals_per_elem):
                b_idx = p * vals_per_elem + v
                if b_idx < n_blocks:
                    indices = (packed_vals >> (v * bits)) & index_mask

                    # per-element dot product (avoid 3D tl.sum)
                    cent_base = b_idx * (n_centroids * block_dim)
                    for d in range(0, block_dim):
                        global_d = b_idx * block_dim + d

                        q_vals = tl.load(
                            query_ptr + q_offsets * q_stride_h + global_d,
                            mask=q_mask,
                            other=0.0,
                        ).to(tl.float32)

                        c_vals = tl.load(
                            centroids_ptr + cent_base + indices * block_dim + d,
                            mask=t_mask,
                            other=0.0,
                        ).to(tl.float32)

                        acc += q_vals[:, None] * c_vals[None, :]

        tl.store(
            out_ptr
            + q_offsets[:, None] * out_stride_h
            + t_offsets[None, :],
            acc,
            mask=q_mask[:, None] & t_mask[None, :],
        )


# ------------------------------------------------------------------
# Reference implementation (always available)
# ------------------------------------------------------------------


def _torch_attention_scores(
    codec,
    query: torch.Tensor,
    quantized_key,
) -> torch.Tensor:
    """PyTorch reference: dequantize then matmul.

    This is the fallback path used when Triton is not available or the
    input shapes don't match the narrow kernel target.
    """
    k_hat = codec.dequantize(quantized_key)
    return torch.matmul(query.float(), k_hat.float().transpose(-2, -1))


# ------------------------------------------------------------------
# Shape validation
# ------------------------------------------------------------------


def _supports_triton_fast_path(
    codec,
    query: torch.Tensor,
    quantized_key,
) -> bool:
    return _triton_fast_path_unavailable_reason(codec, query, quantized_key) is None


def _triton_fast_path_unavailable_reason(
    codec,
    query: torch.Tensor,
    quantized_key,
) -> str | None:
    """
    Narrow first kernel target:
    - query: [num_query_heads, head_dim]
    - indices: [num_tokens, n_blocks]
    - centroids: [n_blocks, n_centroids, block_dim]
    """
    if not _HAS_TRITON:
        return "Triton is not installed or failed to import."
    if "_sunshape_scores_kernel" not in globals():
        return "Triton kernel _sunshape_scores_kernel is not defined."
    if not hasattr(quantized_key, "indices"):
        return "Quantized key must expose an .indices tensor."
    if getattr(quantized_key, "passthrough_blocks", None) is not None:
        return "Mixed-precision passthrough blocks are not yet supported by the Triton fast path."
    if not query.is_cuda:
        return "Query tensor must be on CUDA for the Triton fast path."
    if not quantized_key.indices.is_cuda:
        return "Quantized key indices must be on CUDA for the Triton fast path."
    if query.ndim != 2:
        return f"Expected query.ndim == 2, got {query.ndim}."
    if quantized_key.indices.ndim != 2:
        return f"Expected quantized_key.indices.ndim == 2, got {quantized_key.indices.ndim}."
    if quantized_key.indices.shape[1] != codec.n_blocks:
        return (
            "Quantized key block count does not match codec: "
            f"indices.shape[1]={quantized_key.indices.shape[1]}, codec.n_blocks={codec.n_blocks}."
        )
    if query.shape[1] != codec.head_dim:
        return f"Query head dim {query.shape[1]} does not match codec.head_dim {codec.head_dim}."
    if codec.centroids.ndim != 3:
        return f"Expected codec.centroids.ndim == 3, got {codec.centroids.ndim}."
    return None


def require_triton_fast_path(
    codec,
    query: torch.Tensor,
    quantized_key,
) -> None:
    reason = _triton_fast_path_unavailable_reason(codec, query, quantized_key)
    if reason is not None:
        raise RuntimeError(reason)


# ------------------------------------------------------------------
# Precomputed query-centroid dots (host-side)
# ------------------------------------------------------------------


def _precompute_query_centroid_dots(
    query: torch.Tensor,
    centroids: torch.Tensor,
    n_blocks: int,
    block_dim: int,
) -> torch.Tensor:
    """
    Precompute dot(query_block_b, centroid[b, c, :]) for all (q, b, c).

    Parameters
    ----------
    query : (n_qh, head_dim) float32, already in codec basis
    centroids : (n_blocks, n_centroids, block_dim) float32

    Returns
    -------
    qdots : (n_qh, n_blocks, n_centroids) float32
        qdots[q, b, c] = dot(query[q, b*bd:(b+1)*bd], centroids[b, c, :])
    """
    n_qh = query.shape[0]
    n_centroids = centroids.shape[1]
    # Reshape query into blocks: (n_qh, n_blocks, block_dim)
    q_blocks = query.view(n_qh, n_blocks, block_dim)
    # Batch matmul: (n_qh, n_blocks, block_dim) @ (n_blocks, block_dim, n_centroids)
    # -> (n_qh, n_blocks, n_centroids)
    # Use einsum for clarity: qdots[q, b, c] = sum_d q_blocks[q, b, d] * centroids[b, c, d]
    qdots = torch.einsum("qbd,bcd->qbc", q_blocks, centroids)
    return qdots.contiguous()


# ------------------------------------------------------------------
# Triton dispatch
# ------------------------------------------------------------------


def _triton_attention_scores(
    codec,
    query: torch.Tensor,
    quantized_key,
) -> torch.Tensor:
    """Single-step Triton fast path using precomputed query-centroid dots.

    The hot kernel is now pure gather+add — no FMA in the inner loop.
    The precompute step (einsum over n_qh × n_blocks × n_centroids × block_dim)
    is done once per decode step and is small relative to the token sweep.
    """
    if hasattr(codec, "_forward_transform"):
        query = codec._forward_transform(query)
    if query.dtype != torch.float32:
        query = query.to(torch.float32)
    if not query.is_contiguous():
        query = query.contiguous()

    indices = quantized_key.indices
    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)
    if not indices.is_contiguous():
        indices = indices.contiguous()

    centroids = codec.centroids
    if centroids.dtype != torch.float32:
        centroids = centroids.to(torch.float32)
    if not centroids.is_contiguous():
        centroids = centroids.contiguous()

    n_qh, d_model = query.shape
    n_tokens, _ = indices.shape
    n_centroids = centroids.shape[1]

    # --- Precompute query-centroid block dot products ---
    # Cost: n_qh × n_blocks × n_centroids × block_dim FMAs (once)
    # For n_qh=8, n_blocks=16, n_centroids=256, block_dim=8: ~262K FMAs
    # This is tiny compared to the N-token sweep.
    qdots = _precompute_query_centroid_dots(
        query, centroids, codec.n_blocks, codec.block_dim,
    )

    output = torch.empty((n_qh, n_tokens), device=query.device, dtype=torch.float32)

    block_q = int(getattr(codec, "_triton_block_q", 8))
    block_t = int(getattr(codec, "_triton_block_t", 64))
    num_warps = int(getattr(codec, "_triton_num_warps", 4))

    grid = (triton.cdiv(n_qh, block_q), triton.cdiv(n_tokens, block_t))

    _sunshape_precomputed_scores_kernel[grid](
        qdots,
        indices,
        output,
        n_qh,
        n_tokens,
        n_centroids,
        qdots.stride(0),
        qdots.stride(1),
        indices.stride(0),
        output.stride(0),
        n_blocks=codec.n_blocks,
        block_q=block_q,
        block_t=block_t,
        num_warps=num_warps,
    )
    return output


def _triton_attention_scores_legacy(
    codec,
    query: torch.Tensor,
    quantized_key,
) -> torch.Tensor:
    """Legacy Triton path using the old fused dequant+matmul kernel."""
    if hasattr(codec, "_forward_transform"):
        query = codec._forward_transform(query)
    if query.dtype != torch.float32:
        query = query.to(torch.float32)
    if not query.is_contiguous():
        query = query.contiguous()

    indices = quantized_key.indices
    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)
    if not indices.is_contiguous():
        indices = indices.contiguous()

    centroids = codec.centroids
    if centroids.dtype != torch.float32:
        centroids = centroids.to(torch.float32)
    if not centroids.is_contiguous():
        centroids = centroids.contiguous()

    n_qh, d_model = query.shape
    n_tokens, _ = indices.shape
    n_centroids = centroids.shape[1]

    output = torch.empty((n_qh, n_tokens), device=query.device, dtype=torch.float32)

    block_q = int(getattr(codec, "_triton_block_q", 8))
    block_t = int(getattr(codec, "_triton_block_t", 64))
    num_warps = int(getattr(codec, "_triton_num_warps", 4))

    grid = (triton.cdiv(n_qh, block_q), triton.cdiv(n_tokens, block_t))

    _sunshape_scores_kernel[grid](
        query,
        indices,
        centroids,
        output,
        n_qh,
        n_tokens,
        n_centroids,
        query.stride(0),
        indices.stride(0),
        output.stride(0),
        n_blocks=codec.n_blocks,
        block_dim=codec.block_dim,
        block_q=block_q,
        block_t=block_t,
        num_warps=num_warps,
    )
    return output


def _triton_kernel_variant(codec) -> str:
    variant = str(getattr(codec, "_triton_kernel_variant", "legacy")).strip().lower()
    if variant not in {"legacy", "precomputed"}:
        return "legacy"
    return variant


def sunshape_attention_scores_strict(
    codec,
    query: torch.Tensor,
    quantized_key,
) -> torch.Tensor:
    """Run the Triton fast path only, or raise a descriptive error.

    The query is interpreted in the original model basis and transformed
    internally into the codec's quantization basis when needed.
    """
    require_triton_fast_path(codec, query, quantized_key)
    if _triton_kernel_variant(codec) == "precomputed":
        return _triton_attention_scores(codec, query, quantized_key)
    return _triton_attention_scores_legacy(codec, query, quantized_key)


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------


def sunshape_attention_scores(
    codec,
    query: torch.Tensor,
    quantized_key,
) -> torch.Tensor:
    """
    Compute ``<query, k_hat>`` against SunShape block-quantized keys.

    Triton is used for the narrow validated kernel target:
    a single decode-step query tile against a 2D block-VQ index matrix.
    All other cases fall back to the PyTorch reference path.

    Parameters
    ----------
    codec : SunShapeBlockCodec
        Calibrated codec with ``.centroids``, ``.n_blocks``, ``.block_dim``,
        ``.head_dim`` attributes.
    query : torch.Tensor
        Query tensor of shape ``[num_query_heads, head_dim]`` in the original
        model basis.
    quantized_key : SunShapeQuantized
        Named tuple with ``.indices`` of shape ``[num_tokens, n_blocks]``.

    Returns
    -------
    torch.Tensor
        Attention scores of shape ``[num_query_heads, num_tokens]``.
    """
    global _TRITON_FAILURE_LOGGED
    if _supports_triton_fast_path(codec, query, quantized_key):
        try:
            return sunshape_attention_scores_strict(codec, query, quantized_key)
        except Exception as exc:  # pragma: no cover
            if not _TRITON_FAILURE_LOGGED:
                logger.warning(
                    "Falling back to Torch attention path after Triton failure: %s", exc
                )
                _TRITON_FAILURE_LOGGED = True

    return _torch_attention_scores(codec, query, quantized_key)
