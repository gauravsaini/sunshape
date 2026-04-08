from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def refine_centroids_strict(
    q_cal: torch.Tensor,
    k_cal: torch.Tensor,
    centroids: torch.Tensor,
    e_metric: torch.Tensor,
    block_dim: int,
    n_steps: int = 3,
    step_size: float = 0.1,
    block_affinity: torch.Tensor | None = None,
    active_blocks: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Refine block centroids against an attention-aware objective without changing
    the codebook size.

    This is a small package version of the notebook DSQ refinement.
    """
    head_dim = k_cal.shape[-1]
    n_blocks = head_dim // block_dim
    q_sub = q_cal[: min(128, len(q_cal))].float()
    k_sub = k_cal[: min(1024, len(k_cal))].float()
    scale = math.sqrt(head_dim)

    refined = centroids.clone().float()
    k_hat = torch.zeros_like(k_sub)

    if block_affinity is None:
        block_affinity = torch.ones(n_blocks, device=refined.device, dtype=refined.dtype)
    else:
        block_affinity = block_affinity.to(refined.device, dtype=refined.dtype)

    if active_blocks is None:
        active_blocks = torch.ones(n_blocks, device=refined.device, dtype=torch.bool)
    else:
        active_blocks = active_blocks.to(refined.device, dtype=torch.bool)

    for b in range(n_blocks):
        sl = slice(b * block_dim, (b + 1) * block_dim)
        e_blk = e_metric[sl, sl]
        dists = torch.cdist(k_sub[:, sl] @ e_blk.T, refined[b] @ e_blk.T)
        assigns = dists.argmin(dim=-1)
        k_hat[:, sl] = refined[b][assigns]

    for _ in range(n_steps):
        p_orig = F.softmax((q_sub @ k_sub.T) / scale, dim=-1)
        p_hat = F.softmax((q_sub @ k_hat.T) / scale, dim=-1)
        err = p_hat - p_orig
        sens = p_hat * (1.0 - p_hat)
        grad_attn = ((err * sens).T @ q_sub) / scale
        total_grad = (k_hat - k_sub) + 0.5 * grad_attn

        for b in range(n_blocks):
            if not bool(active_blocks[b]):
                continue
            sl = slice(b * block_dim, (b + 1) * block_dim)
            e_blk = e_metric[sl, sl]
            dists = torch.cdist(k_sub[:, sl] @ e_blk.T, refined[b] @ e_blk.T)
            assigns = dists.argmin(dim=-1)

            new_cents = refined[b].clone()
            block_step = float(step_size) * float(block_affinity[b].item())
            for ci in range(refined.shape[1]):
                mask = assigns == ci
                if mask.any():
                    new_cents[ci] -= block_step * total_grad[mask, sl].mean(dim=0)
            refined[b] = new_cents
            k_hat[:, sl] = new_cents[assigns]

    return refined
