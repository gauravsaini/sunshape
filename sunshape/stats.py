from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch


def bits_to_centroids(bits_per_dim: float, block_dim: int) -> int:
    bits_per_block = bits_per_dim * block_dim
    rounded = round(bits_per_block)
    if abs(bits_per_block - rounded) > 1e-6:
        raise ValueError(
            f"bits_per_dim={bits_per_dim} with block_dim={block_dim} gives non-integer bits/block={bits_per_block}"
        )
    return 2 ** int(rounded)


def default_block_dim(bits_per_dim: float) -> int:
    if abs(bits_per_dim - 1.0) < 1e-6:
        return 8
    if abs(bits_per_dim - 3.0) < 1e-6 or abs(bits_per_dim - 4.0) < 1e-6:
        return 2
    raise ValueError(f"Unsupported rate for default block-dim mapping: {bits_per_dim}")


def pct_reduction(compressed: float, baseline: float) -> float:
    return (1.0 - compressed / baseline) * 100.0


def load_trace_meta(trace_path: str | Path) -> dict:
    obj = torch.load(trace_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "__meta__" in obj:
        return dict(obj["__meta__"])
    raise ValueError(f"Trace artifact {trace_path} is missing metadata")


@dataclass
class CompressionStats:
    per_head: pd.DataFrame
    per_context: pd.DataFrame


def build_compression_stats(
    *,
    model_name: str,
    trace_meta: dict,
    bits_per_dim_list: list[float],
    contexts: list[int],
    key_baseline_bits: int = 16,
    value_bits: int = 16,
    centroid_bits: int = 32,
) -> CompressionStats:
    head_dims = sorted({int(v) for v in trace_meta.get("head_dims", {}).values()})
    if not head_dims:
        raise ValueError("Trace metadata is missing head_dims")

    per_head_rows: list[dict] = []
    context_rows: list[dict] = []

    for head_dim in head_dims:
        for bits_per_dim in bits_per_dim_list:
            block_dim = default_block_dim(bits_per_dim)
            if head_dim % block_dim != 0:
                continue
            n_blocks = head_dim // block_dim
            n_centroids = bits_to_centroids(bits_per_dim, block_dim)

            key_fp_baseline_bits = head_dim * key_baseline_bits
            key_quant_bits = head_dim * bits_per_dim
            key_fp_bytes = key_fp_baseline_bits / 8.0
            key_quant_bytes = key_quant_bits / 8.0

            value_fp_bytes = head_dim * value_bits / 8.0
            kv_fp_bytes = key_fp_bytes + value_fp_bytes
            kv_quant_bytes = key_quant_bytes + value_fp_bytes

            codebook_values = n_blocks * n_centroids * block_dim
            codebook_bytes = codebook_values * centroid_bits / 8.0

            per_head_rows.append(
                {
                    "model_name": model_name,
                    "head_dim": head_dim,
                    "bits_per_dim": bits_per_dim,
                    "block_dim": block_dim,
                    "n_blocks": n_blocks,
                    "n_centroids": n_centroids,
                    "key_baseline_bytes_per_head_token": key_fp_bytes,
                    "key_quant_bytes_per_head_token": key_quant_bytes,
                    "key_only_compression_x": key_fp_bytes / key_quant_bytes,
                    "key_only_reduction_pct": pct_reduction(key_quant_bytes, key_fp_bytes),
                    "kv_baseline_bytes_per_head_token": kv_fp_bytes,
                    "kv_quant_bytes_per_head_token_if_values_fp": kv_quant_bytes,
                    "kv_total_compression_x_if_values_fp": kv_fp_bytes / kv_quant_bytes,
                    "kv_total_reduction_pct_if_values_fp": pct_reduction(kv_quant_bytes, kv_fp_bytes),
                    "codebook_bytes_per_head": codebook_bytes,
                }
            )

            for context_tokens in contexts:
                key_baseline_context_bytes = key_fp_bytes * context_tokens
                key_quant_context_bytes = key_quant_bytes * context_tokens
                kv_baseline_context_bytes = kv_fp_bytes * context_tokens
                kv_quant_context_bytes = kv_quant_bytes * context_tokens
                context_rows.append(
                    {
                        "model_name": model_name,
                        "head_dim": head_dim,
                        "context_tokens": context_tokens,
                        "bits_per_dim": bits_per_dim,
                        "block_dim": block_dim,
                        "n_centroids": n_centroids,
                        "key_baseline_context_bytes_per_head": key_baseline_context_bytes,
                        "key_quant_context_bytes_per_head": key_quant_context_bytes,
                        "key_only_context_compression_x": key_baseline_context_bytes / key_quant_context_bytes,
                        "kv_baseline_context_bytes_per_head": kv_baseline_context_bytes,
                        "kv_quant_context_bytes_per_head_if_values_fp": kv_quant_context_bytes,
                        "kv_context_compression_x_if_values_fp": kv_baseline_context_bytes / kv_quant_context_bytes,
                        "codebook_bytes_per_head": codebook_bytes,
                        "codebook_overhead_pct_vs_quant_key_context": (codebook_bytes / key_quant_context_bytes) * 100.0,
                        "codebook_overhead_pct_vs_quant_kv_context": (codebook_bytes / kv_quant_context_bytes) * 100.0,
                    }
                )

    return CompressionStats(
        per_head=pd.DataFrame(per_head_rows).sort_values(["head_dim", "bits_per_dim"]),
        per_context=pd.DataFrame(context_rows).sort_values(["head_dim", "context_tokens", "bits_per_dim"]),
    )
