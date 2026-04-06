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
# Triton kernel: fused block-VQ dequant + matmul
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
        d_model,
        # strides
        q_stride_h,
        q_stride_d,
        idx_stride_t,
        idx_stride_b,
        cent_stride_b,
        cent_stride_c,
        cent_stride_d,
        out_stride_h,
        out_stride_t,
        # constexpr tile sizes
        n_blocks: tl.constexpr,
        block_dim: tl.constexpr,
        block_q: tl.constexpr,
        block_t: tl.constexpr,
    ):
        """
        Compute ``scores[q, t] = sum_d query[q, d] * centroids[block_of(d), idx[t, block_of(d)], d_within_block]``

        Grid: (cdiv(n_qh, block_q), cdiv(n_tokens, block_t))
        """
        pid_q = tl.program_id(0)
        pid_t = tl.program_id(1)

        q_offsets = pid_q * block_q + tl.arange(0, block_q)
        t_offsets = pid_t * block_t + tl.arange(0, block_t)

        q_mask = q_offsets < n_qh
        t_mask = t_offsets < n_tokens

        acc = tl.zeros((block_q, block_t), dtype=tl.float32)

        for b in range(0, n_blocks):
            # Load per-token centroid indices for this block: [block_t]
            token_indices = tl.load(
                indices_ptr + (t_offsets * idx_stride_t) + (b * idx_stride_b),
                mask=t_mask,
                other=0,
            ).to(tl.int32)

            # For each dimension within this block
            for d in range(0, block_dim):
                # Gather centroid values: centroids[b, token_indices, d]
                c_vals = tl.load(
                    centroids_ptr
                    + (b * cent_stride_b)
                    + (token_indices * cent_stride_c)
                    + (d * cent_stride_d),
                    mask=t_mask,
                    other=0.0,
                ).to(tl.float32)

                # Load query values: query[q_offsets, b * block_dim + d]
                global_d = b * block_dim + d
                q_vals = tl.load(
                    query_ptr + (q_offsets * q_stride_h) + (global_d * q_stride_d),
                    mask=q_mask,
                    other=0.0,
                ).to(tl.float32)

                # Outer product accumulation
                acc += q_vals[:, None] * c_vals[None, :]

        tl.store(
            out_ptr
            + (q_offsets[:, None] * out_stride_h)
            + (t_offsets[None, :] * out_stride_t),
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
        d_model,
        n_blocks,
        # strides
        q_stride_h,
        q_stride_d,
        packed_stride_t,
        packed_stride_p,
        cent_stride_b,
        cent_stride_c,
        cent_stride_d,
        out_stride_h,
        out_stride_t,
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
            packed_vals = tl.load(
                packed_ptr + (t_offsets * packed_stride_t) + (p * packed_stride_p),
                mask=t_mask,
                other=0,
            ).to(tl.int32)

            for v in range(0, vals_per_elem):
                b_idx = p * vals_per_elem + v
                if b_idx < n_blocks:
                    indices = (packed_vals >> (v * bits)) & index_mask

                    for d in range(0, block_dim):
                        c_vals = tl.load(
                            centroids_ptr
                            + (b_idx * cent_stride_b)
                            + (indices * cent_stride_c)
                            + (d * cent_stride_d),
                            mask=t_mask,
                            other=0.0,
                        ).to(tl.float32)

                        global_d = b_idx * block_dim + d
                        q_vals = tl.load(
                            query_ptr
                            + (q_offsets * q_stride_h)
                            + (global_d * q_stride_d),
                            mask=q_mask,
                            other=0.0,
                        ).to(tl.float32)

                        acc += q_vals[:, None] * c_vals[None, :]

        tl.store(
            out_ptr
            + (q_offsets[:, None] * out_stride_h)
            + (t_offsets[None, :] * out_stride_t),
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
# Triton dispatch
# ------------------------------------------------------------------


def _triton_attention_scores(
    codec,
    query: torch.Tensor,
    quantized_key,
) -> torch.Tensor:
    """Single-step Triton fast path matching the Torch reference exactly."""
    if hasattr(codec, "_forward_transform"):
        query = codec._forward_transform(query)
    query = query.contiguous().to(torch.float32)
    indices = quantized_key.indices.contiguous().to(torch.int32)
    centroids = codec.centroids.contiguous().to(torch.float32)

    n_qh, d_model = query.shape
    n_tokens, _ = indices.shape

    output = torch.empty((n_qh, n_tokens), device=query.device, dtype=torch.float32)

    block_q = 8
    block_t = 64

    grid = (triton.cdiv(n_qh, block_q), triton.cdiv(n_tokens, block_t))

    _sunshape_scores_kernel[grid](
        query,
        indices,
        centroids,
        output,
        n_qh,
        n_tokens,
        d_model,
        query.stride(0),
        query.stride(1),
        indices.stride(0),
        indices.stride(1),
        centroids.stride(0),
        centroids.stride(1),
        centroids.stride(2),
        output.stride(0),
        output.stride(1),
        n_blocks=codec.n_blocks,
        block_dim=codec.block_dim,
        block_q=block_q,
        block_t=block_t,
        num_warps=4,
    )
    return output


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
    return _triton_attention_scores(codec, query, quantized_key)


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
            return _triton_attention_scores(codec, query, quantized_key)
        except Exception as exc:  # pragma: no cover
            if not _TRITON_FAILURE_LOGGED:
                logger.warning(
                    "Falling back to Torch attention path after Triton failure: %s", exc
                )
                _TRITON_FAILURE_LOGGED = True

    return _torch_attention_scores(codec, query, quantized_key)
