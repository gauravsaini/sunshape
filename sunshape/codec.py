from __future__ import annotations

import math
from typing import NamedTuple

import torch

from sunshape.dsq import refine_centroids_strict
from sunshape.metrics import build_tlsunshape_metric


def _generate_rotation(d: int, seed: int = 0, device: torch.device | None = None) -> torch.Tensor:
    """Generate a random orthogonal rotation matrix via QR decomposition.

    Same construction as TurboQuant: guarantees a uniform Haar-random
    rotation so that each coordinate of the rotated vector follows a
    Beta(d/2, d/2) distribution, and distinct coordinates are nearly
    independent in high dimensions.
    """
    rng = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(d, d, generator=rng, device=device)
    Q, R = torch.linalg.qr(A)
    diag_signs = torch.sign(torch.diag(R))
    Q = Q * diag_signs.unsqueeze(0)
    return Q.float()


class SunShapeQuantized(NamedTuple):
    indices: torch.Tensor
    head_dim: int
    block_dim: int
    n_blocks: int
    passthrough_blocks: torch.Tensor | None = None
    passthrough_values: torch.Tensor | None = None


def _kmeans(data: torch.Tensor, n_centroids: int, n_iters: int = 25, seed: int = 0) -> torch.Tensor:
    n, d = data.shape
    rng = torch.Generator(device=data.device).manual_seed(seed)
    if n <= n_centroids:
        out = torch.zeros(n_centroids, d, device=data.device, dtype=data.dtype)
        out[:n] = data
        return out

    centroids = data[torch.randperm(n, generator=rng, device=data.device)[:n_centroids]].clone()
    for _ in range(n_iters):
        assigns = torch.cdist(data, centroids).argmin(dim=1)
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(n_centroids, device=data.device)
        new_centroids.index_add_(0, assigns, data)
        counts.index_add_(0, assigns, torch.ones(n, device=data.device))
        mask = counts > 0
        new_centroids[mask] /= counts[mask].unsqueeze(1)
        new_centroids[~mask] = centroids[~mask]
        centroids = new_centroids
    return centroids


def _metric_weighted_kmeans(
    data: torch.Tensor,
    metric: torch.Tensor,
    n_centroids: int,
    n_iters: int = 25,
    seed: int = 0,
) -> torch.Tensor:
    """
    Compatibility helper for callers that want blockwise k-means under the
    learned TL-SMAQ/SunShape metric. The metric here is the linear
    preconditioner block E_b, so we cluster in shaped coordinates and map the
    centroids back into the original key space.
    """
    shaped = data @ metric.T
    cents_shaped = _kmeans(shaped, n_centroids, n_iters=n_iters, seed=seed)
    metric_inv = torch.linalg.pinv(metric)
    return cents_shaped @ metric_inv.T


def _compact_index_dtype(n_centroids: int) -> torch.dtype:
    if n_centroids <= 256:
        return torch.uint8
    if n_centroids <= 65535:
        return torch.int16
    return torch.int32


def _invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm), device=perm.device)
    return inv


def _apply_permutation(x: torch.Tensor, perm: torch.Tensor | None) -> torch.Tensor:
    if perm is None or perm.numel() == 0:
        return x
    return x[..., perm]


def _covariance_block_permutation(q_cal: torch.Tensor, block_dim: int) -> torch.Tensor:
    q_centered = q_cal - q_cal.mean(dim=0, keepdim=True)
    cov = (q_centered.T @ q_centered) / max(1, q_cal.shape[0] - 1)
    d = cov.shape[0]
    affinity = cov.abs().clone()
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
            block_tensor = torch.tensor(block, device=affinity.device, dtype=torch.long)
            cand_tensor = torch.tensor(cand_list, device=affinity.device, dtype=torch.long)
            gains = affinity[cand_tensor][:, block_tensor].sum(dim=1)
            best = cand_list[int(gains.argmax().item())]
            block.append(best)
            remaining.remove(best)

        ordered.extend(block)

    return torch.tensor(ordered, device=q_cal.device, dtype=torch.long)


def _block_local_cov_metric(q_cal: torch.Tensor, perm: torch.Tensor, block_dim: int) -> torch.Tensor:
    q_perm = _apply_permutation(q_cal, perm)
    q_centered = q_perm - q_perm.mean(dim=0, keepdim=True)
    cov = (q_centered.T @ q_centered) / max(1, q_perm.shape[0] - 1)
    local_metric = torch.zeros_like(cov)
    d = cov.shape[0]
    n_blocks = d // block_dim
    for b in range(n_blocks):
        sl = slice(b * block_dim, (b + 1) * block_dim)
        local_metric[sl, sl] = cov[sl, sl]
    return local_metric


def _block_affinity_gate(q_cal: torch.Tensor, block_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    centered = q_cal.float() - q_cal.float().mean(dim=0, keepdim=True)
    per_dim_var = centered.square().mean(dim=0)
    n_blocks = q_cal.shape[-1] // block_dim
    block_mass = per_dim_var.view(n_blocks, block_dim).sum(dim=1)
    total_mass = block_mass.sum().clamp(min=1e-12)
    block_affinity = block_mass / total_mass

    ranked = torch.argsort(block_affinity, descending=True)
    cumulative = torch.cumsum(block_affinity[ranked], dim=0)
    cutoff = int((cumulative < 0.5).sum().item()) + 1
    cutoff = max(1, min(cutoff, n_blocks))

    active = torch.zeros(n_blocks, device=q_cal.device, dtype=torch.bool)
    active[ranked[:cutoff]] = True

    active_affinity = block_affinity.clone()
    active_affinity[~active] = 0.0
    active_scale = active_affinity.max().clamp(min=1e-12)
    active_affinity = active_affinity / active_scale
    return active_affinity, active


def _positive_excess_kurtosis(q: torch.Tensor) -> torch.Tensor:
    centered = q.float() - q.float().mean(dim=0, keepdim=True)
    var = centered.square().mean(dim=0).clamp(min=1e-12)
    fourth = centered.pow(4).mean(dim=0)
    excess = fourth / var.square() - 3.0
    return excess.clamp(min=0.0)


def _mixed_precision_block_mask(q_cal: torch.Tensor, block_dim: int, mass_threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    kurtosis = _positive_excess_kurtosis(q_cal)
    n_blocks = q_cal.shape[-1] // block_dim
    block_mass = kurtosis.view(n_blocks, block_dim).sum(dim=1)
    total_mass = block_mass.sum().clamp(min=1e-12)
    block_mass = block_mass / total_mass
    ranked = torch.argsort(block_mass, descending=True)
    cumulative = torch.cumsum(block_mass[ranked], dim=0)
    cutoff = int((cumulative < mass_threshold).sum().item()) + 1
    cutoff = max(1, min(cutoff, n_blocks))
    heavy = torch.zeros(n_blocks, device=q_cal.device, dtype=torch.bool)
    heavy[ranked[:cutoff]] = True
    return heavy, block_mass


class SunShapeBlockCodec(torch.nn.Module):
    """
    Mainline block codec aligned with the current SunShape paper direction.

    Recommended modes:
    - ``profileperm_baseline``: fixed offline ProfilePerm(SigmaQ) + plain block VQ
    - ``profileperm_localmetric_dsq``: ProfilePerm(SigmaQ) + local metric + DSQ
    - ``profileperm_mixed_precision``: ProfilePerm(SigmaQ) + high-precision heavy blocks + VQ on light blocks

    Legacy ablation modes are kept for reproducibility:
    - ``legacy_strict``: TL-SMAQ-style full metric shaping
    - ``rotated``: dense random rotation
    """

    def __init__(
        self,
        head_dim: int,
        block_dim: int = 8,
        n_centroids: int = 256,
        c: float = 5.0,
        n_refine_dsq: int = 3,
        mode: str = "profileperm_baseline",
        use_rotation: bool = False,
        rotation_seed: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        if head_dim % block_dim != 0:
            raise ValueError("head_dim must be divisible by block_dim")
        self.head_dim = head_dim
        self.block_dim = block_dim
        self.n_blocks = head_dim // block_dim
        self.n_centroids = n_centroids
        self.c = c
        self.mode = "rotated" if use_rotation else mode
        self.n_refine_dsq = 3 if n_refine_dsq is None else int(n_refine_dsq)
        self.use_rotation = self.mode == "rotated"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.register_buffer("E", torch.eye(head_dim, device=self.device))
        self.register_buffer("E_inv", torch.eye(head_dim, device=self.device))
        self.register_buffer("centroids", torch.empty(self.n_blocks, n_centroids, block_dim, device=self.device))
        self.register_buffer("dsq_block_affinity", torch.ones(self.n_blocks, device=self.device))
        self.register_buffer("dsq_active_blocks", torch.ones(self.n_blocks, device=self.device, dtype=torch.bool))
        self.register_buffer("mixed_high_precision_blocks", torch.zeros(self.n_blocks, device=self.device, dtype=torch.bool))
        self.register_buffer("mixed_block_mass", torch.zeros(self.n_blocks, device=self.device))
        identity_perm = torch.arange(head_dim, device=self.device, dtype=torch.long)
        self.register_buffer("permutation", identity_perm.clone())
        self.register_buffer("inv_permutation", identity_perm.clone())

        if self.use_rotation:
            rotation = _generate_rotation(head_dim, seed=rotation_seed, device=self.device)
        else:
            rotation = torch.eye(head_dim, device=self.device)
        self.register_buffer("rotation", rotation)

    def _forward_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in {"profileperm_baseline", "profileperm_localmetric_dsq", "profileperm_mixed_precision"}:
            return _apply_permutation(x, self.permutation)
        if self.use_rotation:
            return x @ self.rotation.T
        return x

    def _inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in {"profileperm_baseline", "profileperm_localmetric_dsq", "profileperm_mixed_precision"}:
            return _apply_permutation(x, self.inv_permutation)
        if self.use_rotation:
            return x @ self.rotation
        return x

    def fit(
        self,
        q_cal: torch.Tensor,
        k_cal: torch.Tensor,
        kmeans_iters: int = 25,
        seed: int = 0,
    ) -> "SunShapeBlockCodec":
        q_cal = q_cal.to(self.device).float()
        k_cal = k_cal.to(self.device).float()

        if self.mode == "rotated":
            q_cal = q_cal @ self.rotation.T
            k_cal = k_cal @ self.rotation.T
            self.E.copy_(torch.eye(self.head_dim, device=self.device))
            self.E_inv.copy_(torch.eye(self.head_dim, device=self.device))
            self.permutation.copy_(torch.arange(self.head_dim, device=self.device))
        elif self.mode == "identity_baseline":
            self.E.copy_(torch.eye(self.head_dim, device=self.device))
            self.E_inv.copy_(torch.eye(self.head_dim, device=self.device))
            self.permutation.copy_(torch.arange(self.head_dim, device=self.device))
            self.inv_permutation.copy_(torch.arange(self.head_dim, device=self.device))
        elif self.mode == "profileperm_baseline":
            perm = _covariance_block_permutation(q_cal, self.block_dim).to(self.device)
            inv_perm = _invert_permutation(perm)
            self.permutation.copy_(perm)
            self.inv_permutation.copy_(inv_perm)
            q_cal = _apply_permutation(q_cal, perm)
            k_cal = _apply_permutation(k_cal, perm)
            self.E.copy_(torch.eye(self.head_dim, device=self.device))
            self.E_inv.copy_(torch.eye(self.head_dim, device=self.device))
        elif self.mode == "profileperm_localmetric_dsq":
            perm = _covariance_block_permutation(q_cal, self.block_dim).to(self.device)
            inv_perm = _invert_permutation(perm)
            self.permutation.copy_(perm)
            self.inv_permutation.copy_(inv_perm)
            q_cal = _apply_permutation(q_cal, perm)
            k_cal = _apply_permutation(k_cal, perm)
            e = _block_local_cov_metric(q_cal, torch.arange(self.head_dim, device=self.device), self.block_dim)
            self.E.copy_(e)
            self.E_inv.copy_(torch.linalg.pinv(e))
            dsq_block_affinity, dsq_active_blocks = _block_affinity_gate(q_cal, self.block_dim)
            self.dsq_block_affinity.copy_(dsq_block_affinity.to(self.device))
            self.dsq_active_blocks.copy_(dsq_active_blocks.to(self.device))
        elif self.mode == "profileperm_mixed_precision":
            perm = _covariance_block_permutation(q_cal, self.block_dim).to(self.device)
            inv_perm = _invert_permutation(perm)
            self.permutation.copy_(perm)
            self.inv_permutation.copy_(inv_perm)
            q_cal = _apply_permutation(q_cal, perm)
            k_cal = _apply_permutation(k_cal, perm)
            self.E.copy_(torch.eye(self.head_dim, device=self.device))
            self.E_inv.copy_(torch.eye(self.head_dim, device=self.device))
            heavy_blocks, block_mass = _mixed_precision_block_mask(q_cal, self.block_dim, mass_threshold=0.5)
            self.mixed_high_precision_blocks.copy_(heavy_blocks.to(self.device))
            self.mixed_block_mass.copy_(block_mass.to(self.device))
        elif self.mode == "legacy_strict":
            self.permutation.copy_(torch.arange(self.head_dim, device=self.device))
            self.inv_permutation.copy_(torch.arange(self.head_dim, device=self.device))
            e, e_inv = build_tlsunshape_metric(q_cal, k_cal, c=self.c)
            self.E.copy_(e)
            self.E_inv.copy_(e_inv)
        else:
            raise ValueError(f"Unknown SunShapeBlockCodec mode: {self.mode}")

        if self.mode != "profileperm_localmetric_dsq":
            self.dsq_block_affinity.fill_(1.0)
            self.dsq_active_blocks.fill_(True)
        if self.mode != "profileperm_mixed_precision":
            self.mixed_high_precision_blocks.fill_(False)
            self.mixed_block_mass.zero_()

        working_centroids = []
        for b in range(self.n_blocks):
            sl = slice(b * self.block_dim, (b + 1) * self.block_dim)
            if self.mode == "profileperm_mixed_precision" and bool(self.mixed_high_precision_blocks[b].item()):
                working_centroids.append(torch.zeros(self.n_centroids, self.block_dim, device=self.device))
                continue
            e_blk = self.E[sl, sl]
            e_inv_blk = self.E_inv[sl, sl]
            c_shaped = _kmeans(k_cal[:, sl] @ e_blk.T, self.n_centroids, n_iters=kmeans_iters, seed=seed + b)
            c_orig = c_shaped @ e_inv_blk.T
            working_centroids.append(c_orig)

        working = torch.stack(working_centroids)
        if self.mode in ("profileperm_baseline", "identity_baseline"):
            self.centroids.copy_(working)
            return self
        if self.n_refine_dsq > 0:
            working = refine_centroids_strict(
                q_cal=q_cal,
                k_cal=k_cal,
                centroids=working,
                e_metric=self.E,
                block_dim=self.block_dim,
                n_steps=self.n_refine_dsq,
                block_affinity=self.dsq_block_affinity,
                active_blocks=self.dsq_active_blocks,
            )
        self.centroids.copy_(working)
        return self

    def quantize(self, keys: torch.Tensor) -> SunShapeQuantized:
        keys = keys.to(self.device).float()
        keys = self._forward_transform(keys)
        indices = torch.zeros(keys.shape[0], self.n_blocks, dtype=torch.long, device=self.device)
        passthrough_values = None
        if self.mode == "profileperm_mixed_precision" and bool(self.mixed_high_precision_blocks.any().item()):
            passthrough_values = torch.zeros(keys.shape[0], self.n_blocks, self.block_dim, device=self.device, dtype=keys.dtype)
        for b in range(self.n_blocks):
            sl = slice(b * self.block_dim, (b + 1) * self.block_dim)
            if self.mode == "profileperm_mixed_precision" and bool(self.mixed_high_precision_blocks[b].item()):
                if passthrough_values is not None:
                    passthrough_values[:, b, :] = keys[:, sl]
                continue
            e_blk = self.E[sl, sl]
            dists = torch.cdist(keys[:, sl] @ e_blk.T, self.centroids[b] @ e_blk.T)
            indices[:, b] = dists.argmin(dim=-1)
        return SunShapeQuantized(
            indices=indices.to(_compact_index_dtype(self.n_centroids)),
            head_dim=self.head_dim,
            block_dim=self.block_dim,
            n_blocks=self.n_blocks,
            passthrough_blocks=self.mixed_high_precision_blocks.detach().clone() if self.mode == "profileperm_mixed_precision" else None,
            passthrough_values=passthrough_values,
        )

    def dequantize(self, quantized: SunShapeQuantized) -> torch.Tensor:
        k_hat = torch.zeros(quantized.indices.shape[0], self.head_dim, device=self.device, dtype=self.centroids.dtype)
        indices = quantized.indices.long()
        for b in range(self.n_blocks):
            sl = slice(b * self.block_dim, (b + 1) * self.block_dim)
            if quantized.passthrough_blocks is not None and bool(quantized.passthrough_blocks[b].item()):
                if quantized.passthrough_values is None:
                    raise ValueError("Mixed-precision quantized keys are missing passthrough values.")
                k_hat[:, sl] = quantized.passthrough_values[:, b, :]
            else:
                k_hat[:, sl] = self.centroids[b][indices[:, b]]
        return self._inverse_transform(k_hat)

    def forward(self, keys: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(keys))

    def attention_scores(self, query: torch.Tensor, quantized: SunShapeQuantized) -> torch.Tensor:
        from sunshape.triton_kernels import (
            _supports_triton_fast_path,
            _triton_attention_scores,
            _torch_attention_scores,
        )

        query = query.to(self.device).float()

        # Triton fast path operates in the codec's transformed space.
        # For mainline SunShape this is the fixed offline permutation space;
        # for the legacy ablation it can also be a dense rotated space.
        if _supports_triton_fast_path(self, query, quantized):
            try:
                return _triton_attention_scores(self, query, quantized)
            except Exception:
                pass

        # Fallback path dequantizes back into the original basis.
        return _torch_attention_scores(self, query, quantized)

    def heldout_logit_mse(self, q_test: torch.Tensor, k_test: torch.Tensor) -> float:
        q_test = q_test.to(self.device).float()
        k_test = k_test.to(self.device).float()
        k_hat = self.forward(k_test)
        delta = k_test - k_hat
        return ((q_test * delta).sum(dim=1).square()).mean().item()

    def kl_attention(self, q_test: torch.Tensor, k_test: torch.Tensor) -> float:
        import torch.nn.functional as F

        q_test = q_test.to(self.device).float()
        k_test = k_test.to(self.device).float()
        k_hat = self.forward(k_test)
        scale = math.sqrt(self.head_dim)
        logits_orig = (q_test @ k_test.T) / scale
        logits_hat = (q_test @ k_hat.T) / scale
        p = F.softmax(logits_orig, dim=-1)
        qd = F.softmax(logits_hat, dim=-1)
        return (p * (torch.log(p + 1e-9) - torch.log(qd + 1e-9))).sum(dim=-1).mean().item()
