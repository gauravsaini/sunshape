from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def log_shape(eigvals: torch.Tensor, c: float = 5.0, max_cond: float = 100.0) -> torch.Tensor:
    """
    Log-compressed spectral shaping used by TL-SMAQ/SunShape.

    Applies a log1p compression to the eigenvalues, then normalizes the
    log-spectrum to have zero mean (geometric mean = 1). The result is
    the per-eigenvalue scaling factor for the metric.

    Parameters
    ----------
    eigvals : torch.Tensor
        Eigenvalues of the sensitivity-weighted covariance, sorted ascending.
    c : float
        Compression constant (higher = more aggressive shaping).
    max_cond : float
        Maximum condition number allowed for the shaped eigenvalues.
        This prevents the metric inverse from amplifying reconstruction
        errors when some eigenvalue directions have near-zero sensitivity.
    """
    shaped = torch.log1p(c * eigvals.clamp(min=0))
    shaped = shaped.clamp(min=1e-8)
    log_shaped = torch.log(shaped)
    log_shaped = log_shaped - log_shaped.mean()
    result = torch.exp(log_shaped / 2.0)

    # Enforce condition number bound to prevent E_inv blowup at high dims
    if max_cond > 0 and result.numel() > 1:
        max_val = result.max()
        min_allowed = max_val / max_cond
        result = result.clamp(min=min_allowed.item())

    return result


def build_tlsunshape_metric(
    q_cal: torch.Tensor,
    k_cal: torch.Tensor,
    c: float = 5.0,
    max_queries: int = 128,
    max_keys: int = 512,
    max_cond: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the task-local metric used by SunShape.

    Queries are reweighted by an attention sensitivity proxy ``p(1-p)`` before
    constructing the covariance.

    Parameters
    ----------
    max_cond : float
        Maximum condition number for the shaped metric. Prevents the
        metric inverse from amplifying errors when some eigen-directions
        have near-zero query sensitivity. Default 100.0 keeps the inverse
        well-behaved at all practical head dimensions (64–256).
    """
    d_head = q_cal.shape[-1]
    q_sub = q_cal[: min(max_queries, len(q_cal))].float()
    k_sub = k_cal[: min(max_keys, len(k_cal))].float()

    logits = (q_sub @ k_sub.T) / math.sqrt(d_head)
    p = F.softmax(logits, dim=-1)
    sensitivity = (p * (1.0 - p)).sum(dim=-1)

    q_weighted = q_sub * sensitivity.sqrt().unsqueeze(1)
    sigma = (q_weighted.T @ q_weighted) / max(1, len(q_sub))

    evals, evecs = torch.linalg.eigh(sigma)
    shaped = log_shape(evals, c=c, max_cond=max_cond)
    sqrt_diag = torch.diag(shaped)
    inv_sqrt_diag = torch.diag(1.0 / shaped.clamp(min=1e-8))
    e = evecs @ sqrt_diag @ evecs.T
    e_inv = evecs @ inv_sqrt_diag @ evecs.T
    return e, e_inv

