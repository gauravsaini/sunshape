"""
SunShape BOCI Diagnostic — one-call analysis for any model.

Usage:

    from sunshape.diagnose import diagnose_model

    # Option A: pass a model name → auto-extract traces
    report = diagnose_model("Qwen/Qwen3.5-4B")

    # Option B: pass pre-extracted queries per layer
    report = diagnose_model(queries={"layer_8": q_tensor, "layer_16": q_tensor})

    # The report tells you exactly what to do
    for layer in report.layers:
        print(layer)

    # Or just print it
    print(report)

Theory reference:
    The diagnostic computes a block-outlier compatibility score (BOCI) for
    each layer by combining block-compatibility improvement with outlier
    concentration under the same permutation. The structural BCI term and the
    rate-dependent noise floor ε(R) ~ 2^{-2R} remain the mechanistic pieces
    underneath the practical diagnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch


# ─── Core BCI computation ─────────────────────────────────────────────────


def _covariance(q: torch.Tensor) -> torch.Tensor:
    """Compute the query covariance matrix Σ_Q."""
    q_centered = q - q.mean(dim=0, keepdim=True)
    return (q_centered.T @ q_centered) / max(1, q_centered.shape[0] - 1)


def _block_mask(d: int, block_dim: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(d, device=device)
    return (idx[:, None] // block_dim) == (idx[None, :] // block_dim)


def compute_bci(sigma: torch.Tensor, block_dim: int) -> float:
    """
    Block-Compatibility Index.

    BCI = ||Σ - blkdiag(Σ)||²_F / ||Σ||²_F

    BCI = 0 → perfectly block-diagonal (codec loses nothing)
    BCI = 1 → all energy is cross-block (codec discards everything)
    """
    d = sigma.shape[0]
    mask = _block_mask(d, block_dim, sigma.device)
    total_energy = (sigma ** 2).sum()
    on_block_energy = (sigma * mask.float()) ** 2
    on_block = on_block_energy.sum()
    bci = 1.0 - (on_block / total_energy.clamp(min=1e-12))
    return float(bci.item())


def compute_off_block_mass(sigma: torch.Tensor, block_dim: int) -> float:
    """Fraction of |Σ_Q| energy in off-block entries (L1 version)."""
    d = sigma.shape[0]
    mask = _block_mask(d, block_dim, sigma.device)
    total = sigma.abs().sum().clamp(min=1e-12)
    off = sigma.abs()[~mask].sum()
    return float((off / total).item())


def _positive_excess_kurtosis(q: torch.Tensor) -> torch.Tensor:
    centered = q.float() - q.float().mean(dim=0, keepdim=True)
    var = centered.square().mean(dim=0).clamp(min=1e-12)
    fourth = centered.pow(4).mean(dim=0)
    excess = fourth / var.square() - 3.0
    return excess.clamp(min=0.0)


def _outlier_block_share(excess_kurtosis: torch.Tensor, block_dim: int) -> float:
    if excess_kurtosis.numel() == 0:
        return 0.0
    n_blocks = excess_kurtosis.numel() // block_dim
    if n_blocks == 0:
        return 0.0
    block_mass = excess_kurtosis.view(n_blocks, block_dim).sum(dim=1)
    total = block_mass.sum().clamp(min=1e-12)
    return float((block_mass.max() / total).item())


def _boci_score(relative_bci_reduction: float, concentration_gain: float) -> float:
    return float(relative_bci_reduction + max(0.0, concentration_gain))


def _greedy_block_perm(sigma: torch.Tensor, block_dim: int) -> torch.Tensor:
    """Greedy affinity-based block packing (same as codec)."""
    d = sigma.shape[0]
    affinity = sigma.abs().clone()
    affinity.fill_diagonal_(0)
    remaining = set(range(d))
    ordered: list[int] = []

    while remaining:
        rem_list = sorted(remaining)
        if len(rem_list) <= block_dim:
            ordered.extend(rem_list)
            break
        row_scores = affinity[rem_list][:, rem_list].sum(dim=1)
        seed = rem_list[int(row_scores.argmax().item())]
        block = [seed]
        remaining.remove(seed)

        while len(block) < block_dim and remaining:
            cand_list = sorted(remaining)
            block_t = torch.tensor(block, device=affinity.device, dtype=torch.long)
            cand_t = torch.tensor(cand_list, device=affinity.device, dtype=torch.long)
            gains = affinity[cand_t][:, block_t].sum(dim=1)
            best = cand_list[int(gains.argmax().item())]
            block.append(best)
            remaining.remove(best)

        ordered.extend(block)

    return torch.tensor(ordered, device=sigma.device, dtype=torch.long)


def _permute_sigma(sigma: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return sigma[perm][:, perm]


# ─── Noise floor estimation ───────────────────────────────────────────────


def estimate_noise_floor(bits_per_dim: float, block_dim: int) -> float:
    """
    Estimate the per-block VQ noise floor at a given rate.

    For a rate-R VQ with block size m, the distortion scales as:
        ε(R, m) ~ 2^{-2R}

    This is a normalized scalar — the "amount of quantization noise"
    that structural improvement must overcome.
    """
    return 2.0 ** (-2.0 * bits_per_dim)


# ─── Per-layer result ──────────────────────────────────────────────────────


@dataclass
class LayerDiagnostic:
    """Diagnostic result for a single layer."""

    layer_idx: int | str
    head_dim: int
    block_dim: int

    # BCI measurements
    bci_identity: float
    bci_profileperm: float
    delta_bci: float  # negative = PP improved
    relative_bci_reduction: float

    # Off-block mass (L1)
    off_block_identity: float
    off_block_profileperm: float
    delta_off_block: float

    # Anisotropy
    condition_number: float  # σ_max / σ_min of Σ_Q
    sv_ratio_top5: float  # σ_1 / σ_5

    # Outlier diagnostics
    mean_positive_excess_kurtosis: float
    max_positive_excess_kurtosis: float
    outlier_block_share_identity: float
    outlier_block_share_profileperm: float
    delta_outlier_block_share: float

    # Practical diagnostic
    boci_score: float

    # Theory predictions
    noise_floors: dict[float, float]  # rate → ε(R)
    recommendations: dict[float, str]  # rate → recommendation

    # Layer metadata
    attention_type: str = ""  # "full_attention", "linear_attention", etc.

    def __str__(self) -> str:
        rel_reduction = self.relative_bci_reduction * 100
        attn_label = f" [{self.attention_type}]" if self.attention_type else ""
        lines = [
            f"Layer {self.layer_idx} (d={self.head_dim}, block={self.block_dim}){attn_label}",
            f"  BCI:  identity={self.bci_identity:.4f}  →  profileperm={self.bci_profileperm:.4f}  (Δ={self.delta_bci:+.4f}, {rel_reduction:.1f}% reduction)",
            f"  Off-block mass: {self.off_block_identity:.4f} → {self.off_block_profileperm:.4f} (Δ={self.delta_off_block:+.4f})",
            f"  Anisotropy: cond={self.condition_number:.1f}, σ₁/σ₅={self.sv_ratio_top5:.2f}",
            f"  Kurtosis: mean+={self.mean_positive_excess_kurtosis:.2f}, max+={self.max_positive_excess_kurtosis:.2f}",
            f"  Outlier block share: identity={self.outlier_block_share_identity:.3f} → profileperm={self.outlier_block_share_profileperm:.3f} (Δ={self.delta_outlier_block_share:+.3f})",
            f"  BOCI score: {self.boci_score:.4f}",
            f"  Recommendations:",
        ]
        for rate, rec in sorted(self.recommendations.items()):
            nf = self.noise_floors[rate]
            lines.append(f"    {rate:.0f} bit/dim (ε={nf:.4f}): {rec}")
        return "\n".join(lines)


# ─── Full report ───────────────────────────────────────────────────────────


@dataclass
class ModelDiagnostic:
    """Full diagnostic report for a model."""

    model_name: str
    layers: list[LayerDiagnostic] = field(default_factory=list)

    # Theory predictions (dense mixing)
    dense_mixing_expected_bci: float = 0.0  # 1 - m/d

    def summary(self) -> str:
        lines = [
            f"═══ SunShape Diagnostic: {self.model_name} ═══",
            f"  Dense mixing expected BCI: {self.dense_mixing_expected_bci:.4f} (AVOID under block codecs)",
            "",
        ]
        for layer in self.layers:
            lines.append(str(layer))
            lines.append("")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict:
        """Export as a dict for JSON serialization."""
        return {
            "model_name": self.model_name,
            "dense_mixing_expected_bci": self.dense_mixing_expected_bci,
            "rankings": {
                "boci": rank_layers(self, strategy="boci"),
            },
            "layers": [
                {
                    "layer": l.layer_idx,
                    "head_dim": l.head_dim,
                    "block_dim": l.block_dim,
                    "bci_identity": l.bci_identity,
                    "bci_profileperm": l.bci_profileperm,
                    "delta_bci": l.delta_bci,
                    "relative_bci_reduction": l.relative_bci_reduction,
                    "off_block_identity": l.off_block_identity,
                    "off_block_profileperm": l.off_block_profileperm,
                    "mean_positive_excess_kurtosis": l.mean_positive_excess_kurtosis,
                    "max_positive_excess_kurtosis": l.max_positive_excess_kurtosis,
                    "outlier_block_share_identity": l.outlier_block_share_identity,
                    "outlier_block_share_profileperm": l.outlier_block_share_profileperm,
                    "delta_outlier_block_share": l.delta_outlier_block_share,
                    "boci_score": l.boci_score,
                    "condition_number": l.condition_number,
                    "sv_ratio_top5": l.sv_ratio_top5,
                    "recommendations": l.recommendations,
                }
                for l in self.layers
            ],
        }


def layer_selection_score(layer: LayerDiagnostic, strategy: str = "boci") -> float:
    if strategy == "boci":
        return float(layer.boci_score)
    raise ValueError(f"Unsupported layer selection strategy: {strategy}")


def rank_layers(report: ModelDiagnostic, strategy: str = "boci") -> list[dict]:
    ranked = []
    for layer in report.layers:
        ranked.append(
            {
                "layer": int(layer.layer_idx) if isinstance(layer.layer_idx, int) or str(layer.layer_idx).isdigit() else layer.layer_idx,
                "score": layer_selection_score(layer, strategy=strategy),
                "relative_bci_reduction": float(layer.relative_bci_reduction),
                "delta_bci": float(layer.delta_bci),
                "delta_outlier_block_share": float(layer.delta_outlier_block_share),
                "condition_number": float(layer.condition_number),
                "attention_type": layer.attention_type,
            }
        )
    ranked.sort(key=lambda row: (row["score"], row["relative_bci_reduction"], row["condition_number"]), reverse=True)
    return ranked


# ─── Recommendation logic ─────────────────────────────────────────────────


def _recommend(
    delta_bci: float,
    bci_identity: float,
    bits_per_dim: float,
    block_dim: int,
    condition_number: float,
) -> str:
    """
    Apply Corollary 1: ProfilePerm helps when the relative BCI reduction
    exceeds the codec's rate-dependent noise floor.

    Uses relative BCI reduction (|ΔBCI| / BCI_identity) to avoid being
    inflated by large covariance norms. The noise floor ε(R) ~ 2^{-2R}
    sets the threshold that structural improvement must clear.
    """
    noise_floor = estimate_noise_floor(bits_per_dim, block_dim)

    # Relative BCI reduction: how much of the original leakage did PP remove?
    relative_reduction = abs(delta_bci) / max(bci_identity, 1e-8)

    # Scale by how hard the quantization problem is (inverse noise floor)
    # At low bitrate, noise_floor is large → harder to see structural gains
    # At high bitrate, noise_floor is small → easier to see gains
    effective_signal = relative_reduction / max(noise_floor, 1e-12)

    # Strong: PP removes >10% of BCI AND anisotropy gives room to exploit
    if relative_reduction > 0.10 and condition_number > 5.0:
        if bits_per_dim <= 1.0:
            return "SunShape-Pro (large BCI reduction + high anisotropy)"
        return "SunShape (large BCI reduction, alignment reliable)"

    # Moderate: PP removes 3-10% of BCI
    if relative_reduction > 0.03:
        if bits_per_dim <= 1.0:
            return "SunShape-Pro or Baseline (moderate BCI reduction — test both)"
        if bits_per_dim <= 3.0:
            return "SunShape (moderate BCI reduction, low noise floor helps)"
        return "SunShape (marginal — low noise floor may expose small gains)"

    # Weak: PP removes 1-3% of BCI
    if relative_reduction > 0.01:
        if bits_per_dim >= 3.0:
            return "SunShape or Baseline (small BCI reduction — may help at low noise floor)"
        return "Baseline BlockVQ (BCI reduction too small vs noise floor)"

    # Very weak / near-isotropic
    return "Baseline BlockVQ (negligible BCI reduction — alignment won't help)"


# ─── Main diagnostic function ─────────────────────────────────────────────


def diagnose_layer(
    q: torch.Tensor,
    layer_idx: int | str = 0,
    block_dim: int = 8,
    target_rates: list[float] | None = None,
) -> LayerDiagnostic:
    """
    Diagnose a single layer from its query vectors.

    Parameters
    ----------
    q : torch.Tensor
        Query vectors, shape (N, head_dim). Can be from any head;
        for multi-head models, pass queries from a single representative head.
    layer_idx : int or str
        Layer identifier for the report.
    block_dim : int
        Block size of the target codec (default 8).
    target_rates : list of float
        Bitrates to evaluate (default [1.0, 3.0, 4.0]).
    """
    if target_rates is None:
        target_rates = [1.0, 3.0, 4.0]

    q = q.float()
    head_dim = q.shape[-1]

    # Compute Σ_Q
    sigma_q = _covariance(q)
    sigma_norm = sigma_q.norm(p="fro").item()

    # Anisotropy
    eigvals = torch.linalg.eigvalsh(sigma_q)
    eigvals_sorted = eigvals.flip(0)  # descending
    cond = float((eigvals_sorted[0] / eigvals_sorted[-1].clamp(min=1e-12)).item())
    sv5 = float((eigvals_sorted[0] / eigvals_sorted[min(4, len(eigvals_sorted) - 1)].clamp(min=1e-12)).item())

    # BCI under identity
    bci_id = compute_bci(sigma_q, block_dim)
    ob_id = compute_off_block_mass(sigma_q, block_dim)

    # BCI under ProfilePerm
    perm = _greedy_block_perm(sigma_q, block_dim)
    sigma_pp = _permute_sigma(sigma_q, perm)
    bci_pp = compute_bci(sigma_pp, block_dim)
    ob_pp = compute_off_block_mass(sigma_pp, block_dim)

    delta_bci = bci_pp - bci_id
    relative_bci_reduction = abs(delta_bci) / max(bci_id, 1e-8)
    delta_ob = ob_pp - ob_id

    positive_excess_kurtosis = _positive_excess_kurtosis(q)
    kurt_mean = float(positive_excess_kurtosis.mean().item())
    kurt_max = float(positive_excess_kurtosis.max().item())
    outlier_share_identity = _outlier_block_share(positive_excess_kurtosis, block_dim)
    outlier_share_profileperm = _outlier_block_share(positive_excess_kurtosis[perm], block_dim)
    delta_outlier_share = outlier_share_profileperm - outlier_share_identity

    boci_score = _boci_score(relative_bci_reduction, delta_outlier_share)

    # Recommendations per rate
    noise_floors = {}
    recommendations = {}
    for rate in target_rates:
        nf = estimate_noise_floor(rate, block_dim)
        noise_floors[rate] = nf
        recommendations[rate] = _recommend(delta_bci, bci_id, rate, block_dim, cond)

    return LayerDiagnostic(
        layer_idx=layer_idx,
        head_dim=head_dim,
        block_dim=block_dim,
        bci_identity=bci_id,
        bci_profileperm=bci_pp,
        delta_bci=delta_bci,
        relative_bci_reduction=relative_bci_reduction,
        off_block_identity=ob_id,
        off_block_profileperm=ob_pp,
        delta_off_block=delta_ob,
        condition_number=cond,
        sv_ratio_top5=sv5,
        mean_positive_excess_kurtosis=kurt_mean,
        max_positive_excess_kurtosis=kurt_max,
        outlier_block_share_identity=outlier_share_identity,
        outlier_block_share_profileperm=outlier_share_profileperm,
        delta_outlier_block_share=delta_outlier_share,
        boci_score=boci_score,
        noise_floors=noise_floors,
        recommendations=recommendations,
    )


def diagnose_from_traces(
    traces: dict[int, dict[str, torch.Tensor]],
    model_name: str = "unknown",
    block_dim: int = 8,
    target_rates: list[float] | None = None,
    max_queries: int = 8192,
) -> ModelDiagnostic:
    """
    Diagnose a model from pre-extracted Q/K traces.

    Parameters
    ----------
    traces : dict
        {layer_idx: {"q": Tensor(N, d), "k": Tensor(N, d)}}
        As produced by trace_extract.build_trace_artifact().
    model_name : str
        Model identifier for the report.
    block_dim : int
        Block size of the target codec.
    target_rates : list of float
        Bitrates to evaluate.
    max_queries : int
        Subsample queries if more than this (for speed).
    """
    layer_results = []
    head_dim = None

    for layer_idx in sorted(traces.keys()):
        q = traces[layer_idx]["q"]
        if len(q) > max_queries:
            q = q[:max_queries]
        head_dim = q.shape[-1]
        diag = diagnose_layer(q, layer_idx=layer_idx, block_dim=block_dim, target_rates=target_rates)
        layer_results.append(diag)

    report = ModelDiagnostic(
        model_name=model_name,
        layers=layer_results,
        dense_mixing_expected_bci=1.0 - block_dim / (head_dim or 128),
    )
    return report


def _find_decoder_layers(model) -> list:
    """
    Find the sequential list of decoder layers in any HuggingFace model.

    Handles common patterns:
      - model.model.layers         (Llama, Qwen, Mistral, Gemma, ...)
      - model.transformer.h        (GPT-2, GPT-J, ...)
      - model.gpt_neox.layers      (GPT-NeoX, Pythia, ...)
      - model.transformer.blocks   (MPT, ...)
      - model.language_model.model.layers  (multimodal wrappers)
    """
    candidates = [
        ("model", "model", "layers"),
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("transformer", "blocks"),
        ("language_model", "model", "layers"),
    ]

    for path in candidates:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return list(obj)
        except AttributeError:
            continue

    # Fallback: walk all named modules for anything that looks like a ModuleList
    # of layers containing attention
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 2:
            # Check if children have something that looks like attention
            first = module[0]
            for child_name, child in first.named_modules():
                if hasattr(child, "weight") and any(
                    kw in child_name for kw in ("q_proj", "query", "q_attn", "Wqkv")
                ):
                    return list(module)

    raise RuntimeError(
        f"Could not find decoder layers in {type(model).__name__}. "
        "Pass pre-extracted query tensors via diagnose_layer() or diagnose_from_traces() instead."
    )


def _get_head_dim(attn_module) -> int:
    """Best-effort head_dim extraction from any attention module."""
    # Direct attribute
    for attr in ("head_dim", "head_k_dim", "d_head", "key_dim_per_head"):
        val = getattr(attn_module, attr, None)
        if val is not None and isinstance(val, int) and val > 0:
            return val
    # Derive from q_proj shape and num_heads
    q_proj = getattr(attn_module, "q_proj", None)
    num_heads = getattr(attn_module, "num_heads", None) or getattr(attn_module, "num_k_heads", None)
    if q_proj is not None and hasattr(q_proj, "weight") and num_heads:
        return q_proj.weight.shape[0] // num_heads
    return 128  # safe default for most models


def _find_q_proj(layer_module) -> tuple[torch.nn.Module, str, int, str]:
    """
    Find the query projection in a single decoder layer.

    Returns (projection_module, description, head_dim, attention_type).

    Handles:
      Standard attention:
        - layer.self_attn.q_proj           (Llama, Qwen, Mistral, Gemma)
        - layer.attn.q_proj               (some variants)
        - layer.attention.query            (BLOOM-style)
        - layer.attn.c_attn               (GPT-2, fused QKV)
        - layer.attention.Wqkv             (MPT, fused QKV)
      Linear / hybrid attention:
        - layer.linear_attn.in_proj_qkv   (Qwen3.5 GatedDeltaNet)
        - layer.temporal_block.in_proj     (Mamba-style hybrids)
      Fallback:
        - Any child module with 'q_proj' or 'in_proj_qkv' in name
    """
    # ── Standard attention paths ─────────────────────────────────────
    for attn_attr in ("self_attn", "attn", "attention", "self_attention"):
        attn = getattr(layer_module, attn_attr, None)
        if attn is None:
            continue

        head_dim = _get_head_dim(attn)

        # Direct q_proj
        q_proj = getattr(attn, "q_proj", None)
        if q_proj is not None and hasattr(q_proj, "weight"):
            return q_proj, "q_proj", head_dim, "full_attention"

        # query (BLOOM-style)
        q_proj = getattr(attn, "query", None)
        if q_proj is not None and hasattr(q_proj, "weight"):
            return q_proj, "query", head_dim, "full_attention"

        # Fused QKV: c_attn (GPT-2) or Wqkv (MPT)
        for fused_name in ("c_attn", "Wqkv", "qkv_proj", "query_key_value"):
            fused = getattr(attn, fused_name, None)
            if fused is not None and hasattr(fused, "weight"):
                return fused, f"{fused_name}:fused", head_dim, "full_attention"

    # ── Linear / hybrid attention paths ──────────────────────────────
    for attn_attr in ("linear_attn", "temporal_block", "recurrence", "ssm"):
        attn = getattr(layer_module, attn_attr, None)
        if attn is None:
            continue

        head_dim = _get_head_dim(attn)

        # Fused QKV projection (GatedDeltaNet, RWKV, etc.)
        for proj_name in ("in_proj_qkv", "in_proj", "qkv_proj", "in_projection"):
            proj = getattr(attn, proj_name, None)
            if proj is not None and hasattr(proj, "weight"):
                # Determine Q fraction from key_dim if available
                key_dim = getattr(attn, "key_dim", None)
                if key_dim is not None:
                    desc = f"{proj_name}:fused_linear(q_dim={key_dim})"
                else:
                    desc = f"{proj_name}:fused"
                return proj, desc, head_dim, "linear_attention"

        # Direct q_proj in linear attention
        q_proj = getattr(attn, "q_proj", None)
        if q_proj is not None and hasattr(q_proj, "weight"):
            return q_proj, "q_proj", head_dim, "linear_attention"

    # ── Fallback: scan ALL named children ────────────────────────────
    for name, mod in layer_module.named_modules():
        if mod is layer_module:
            continue
        if not hasattr(mod, "weight"):
            continue
        # Look for any query-like projection
        name_lower = name.lower()
        if "q_proj" in name_lower:
            return mod, name, 128, "unknown"
        if "in_proj_qkv" in name_lower or "in_proj" in name_lower:
            return mod, f"{name}:fused", 128, "unknown"

    raise RuntimeError(
        f"Could not find query projection in {type(layer_module).__name__}. "
        f"Children: {[n for n, _ in layer_module.named_children()]}. "
        "Pass pre-extracted query tensors via diagnose_layer() instead."
    )


def diagnose_model(
    model_name: str,
    layers: list[int] | None = None,
    block_dim: int = 8,
    target_rates: list[float] | None = None,
    num_samples: int = 16,
    seq_len: int = 512,
    trust_remote_code: bool = True,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    dataset_split: str = "validation",
) -> ModelDiagnostic:
    """
    One-call diagnostic: load ANY model, extract queries, compute BCI, recommend.

    This function is architecture-agnostic. It dynamically discovers decoder
    layers and query projections by inspecting the module tree. No hardcoded
    model families — works with any HuggingFace CausalLM.

    Parameters
    ----------
    model_name : str
        HuggingFace model name (e.g., "Qwen/Qwen3.5-4B", "meta-llama/...",
        "mistralai/...", "google/gemma-...", "EleutherAI/pythia-...", etc.).
    layers : list of int, optional
        Which decoder layer indices to analyze. If None, picks 4 evenly
        spaced layers automatically from the model's architecture.
    block_dim : int
        Target codec block size (default 8).
    target_rates : list of float
        Bitrates to evaluate (default [1.0, 3.0, 4.0]).
    num_samples : int
        Number of calibration texts from the dataset (default 16).
    seq_len : int
        Max sequence length for calibration (default 512).
    trust_remote_code : bool
        Whether to trust remote code for model loading.
    dataset_name : str
        HuggingFace dataset for calibration data (default "wikitext").
    dataset_config : str
        Dataset config (default "wikitext-2-raw-v1").
    dataset_split : str
        Dataset split (default "validation").

    Returns
    -------
    ModelDiagnostic
        Full report with per-layer BCI analysis and recommendations.

    Example
    -------
    >>> report = diagnose_model("Qwen/Qwen3.5-0.8B")
    >>> print(report)

    >>> report = diagnose_model("meta-llama/Llama-3.1-8B-Instruct")
    >>> print(report)

    >>> report = diagnose_model("EleutherAI/pythia-410m")
    >>> print(report)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Load model ──────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[diagnose] Loading {model_name} on {device}...")
    load_kwargs = {"torch_dtype": dtype, "trust_remote_code": trust_remote_code}
    if device == "cuda":
        load_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Discover architecture ───────────────────────────────────────────
    decoder_layers = _find_decoder_layers(model)
    n_layers = len(decoder_layers)
    print(f"[diagnose] Found {n_layers} decoder layers in {type(model).__name__}")

    if layers is None:
        # Pick 4 evenly spaced layers
        layers = [int(n_layers * f) for f in [0.1, 0.3, 0.6, 0.85]]
        layers = sorted(set(max(0, min(l, n_layers - 1)) for l in layers))
    else:
        layers = [l for l in layers if 0 <= l < n_layers]

    # Discover query projections for each target layer
    layer_info = {}
    for idx in layers:
        try:
            q_proj_mod, desc, head_dim, attn_type = _find_q_proj(decoder_layers[idx])
            layer_info[idx] = {"module": q_proj_mod, "desc": desc, "head_dim": head_dim, "attn_type": attn_type}
            print(f"[diagnose] Layer {idx}: found {desc} (head_dim≈{head_dim}, {attn_type})")
        except RuntimeError as e:
            print(f"[diagnose] Layer {idx}: SKIPPED — {e}")

    if not layer_info:
        raise RuntimeError("Could not find query projections in any requested layer.")

    # ── Hook query projections ──────────────────────────────────────────
    q_captures: dict[int, list[torch.Tensor]] = {idx: [] for idx in layer_info}
    hooks = []

    for idx, info in layer_info.items():
        q_mod = info["module"]
        desc = info["desc"]
        head_dim = info["head_dim"]

        def make_hook(layer_idx: int, projection_desc: str, hdim: int):
            def hook_fn(module, input, output):
                # output shape: (batch, seq_len, out_features)
                out = output
                if isinstance(out, tuple):
                    out = out[0]
                out = out.detach().cpu().float()

                if ":fused" in projection_desc:
                    # Extract Q portion from fused projection
                    total = out.shape[-1]
                    # For linear attention with known q_dim, extract that
                    # e.g. "in_proj_qkv:fused_linear(q_dim=2048)"
                    import re
                    q_dim_match = re.search(r'q_dim=(\d+)', projection_desc)
                    if q_dim_match:
                        q_dim = int(q_dim_match.group(1))
                        out = out[..., :q_dim]
                    else:
                        # Default: assume Q is first third
                        out = out[..., : total // 3]

                # Reshape to (N, head_dim): flatten batch & seq, split heads
                flat = out.reshape(-1, out.shape[-1])
                out_dim = flat.shape[-1]
                if out_dim > hdim and out_dim % hdim == 0:
                    # Multi-head: reshape to (N*num_heads, head_dim)
                    flat = flat.reshape(-1, hdim)

                q_captures[layer_idx].append(flat)

            return hook_fn

        hooks.append(q_mod.register_forward_hook(make_hook(idx, desc, head_dim)))

    # ── Run calibration forward passes ──────────────────────────────────
    print(f"[diagnose] Running {num_samples} calibration forward passes...")
    from datasets import load_dataset
    ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    texts = [x for x in ds["text"] if isinstance(x, str) and x.strip()][:num_samples]

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            model(**enc)

    for h in hooks:
        h.remove()

    # ── Build traces dict and diagnose ──────────────────────────────────
    traces: dict[int, dict[str, torch.Tensor]] = {}
    for idx, chunks in q_captures.items():
        if chunks:
            q_all = torch.cat(chunks, dim=0)
            traces[idx] = {"q": q_all}
            print(f"[diagnose] Layer {idx}: captured {q_all.shape[0]} query vectors, dim={q_all.shape[1]}")

    # Clean up model to free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return diagnose_from_traces(
        traces,
        model_name=model_name,
        block_dim=block_dim,
        target_rates=target_rates,
    )
