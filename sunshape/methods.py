from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from sunshape.cache import SunShapeCache
from sunshape.hf import default_block_dim, resolve_mode_alias
from sunshape.turbo_cache import TurboQuantCache


CacheFactory = Callable[[], Any | None]


@dataclass(frozen=True)
class MethodSpec:
    spec: str
    method: str
    bits_per_dim: float | None
    label: str
    family: str


SUNSHAPE_PRIMARY_METHODS = ("sunshape_base", "sunshape_mixed", "sunshape_pro")

_FAMILY_BY_METHOD = {
    "native_fp": "reference",
    "identity_cache": "reference",
    "profileperm_baseline": "sunshape",
    "profileperm_mixed_precision": "sunshape",
    "profileperm_localmetric_dsq": "sunshape",
    "turboquant_mse": "counterpart",
    "turboquant_prod": "counterpart",
}


def canonical_method_name(name: str) -> str:
    name = str(name).strip()
    if name in {"native_fp", "identity_cache"} or name.startswith("turboquant_"):
        return name
    return resolve_mode_alias(name)


def method_family(name: str) -> str:
    return _FAMILY_BY_METHOD.get(canonical_method_name(name), "custom")


def parse_method_spec(spec: str) -> MethodSpec:
    spec = str(spec).strip()
    if spec in {"native_fp", "identity_cache"}:
        return MethodSpec(
            spec=spec,
            method=spec,
            bits_per_dim=None,
            label=spec,
            family=method_family(spec),
        )
    if "@" not in spec:
        raise ValueError(f"Method spec must look like 'sunshape_base@2.0' or 'turboquant_mse@3.0', got: {spec}")
    name, bits = spec.split("@", 1)
    bits_value = float(bits)
    canonical = canonical_method_name(name)
    return MethodSpec(
        spec=spec,
        method=canonical,
        bits_per_dim=bits_value,
        label=f"{canonical}@{bits_value:g}",
        family=method_family(canonical),
    )


def build_cache_factory(
    *,
    method: str,
    bits_per_dim: float | None,
    layers: list[int],
    traces: dict,
    trace_meta: dict,
    model,
    num_layers: int,
    dsq_steps: int,
    cal_points: int,
    seed: int,
    device: torch.device,
) -> CacheFactory:
    method = canonical_method_name(method)
    if method == "native_fp":
        return lambda: None
    if method == "identity_cache":
        return lambda: SunShapeCache.identity(
            patched_layers=layers,
            config=model.config,
            num_layers=num_layers,
        )
    if method in {"turboquant_mse", "turboquant_prod"}:
        if bits_per_dim is None:
            raise ValueError(f"{method} requires bits_per_dim")
        tq_bits = int(round(bits_per_dim))
        if abs(bits_per_dim - tq_bits) > 1e-6:
            raise ValueError(f"{method} requires integer bits_per_dim, got {bits_per_dim}")
        head_dims = {int(k): int(v) for k, v in (trace_meta.get("head_dims", {}) or {}).items()}
        return lambda: TurboQuantCache.for_layers(
            head_dims=head_dims,
            layers=layers,
            config=model.config,
            num_layers=num_layers,
            bits=tq_bits,
            mode=method,
            seed=seed,
            device=device,
        )

    if bits_per_dim is None:
        raise ValueError(f"{method} requires bits_per_dim")
    cache_template = SunShapeCache.from_traces(
        traces,
        layers=layers,
        config=model.config,
        num_layers=num_layers,
        block_dim=int(default_block_dim(float(bits_per_dim))),
        bits_per_dim=float(bits_per_dim),
        mode=method,
        n_refine_dsq=int(dsq_steps),
        cal_points=int(cal_points),
        seed=int(seed),
        device=device,
    )
    fitted_codecs = {}
    for i, layer in enumerate(cache_template.layers):
        if hasattr(layer, "codec") and layer.codec is not None:
            fitted_codecs[i] = layer.codec
    return lambda: SunShapeCache.from_codecs(
        codecs_dict=fitted_codecs,
        config=model.config,
        num_layers=num_layers,
    )


def default_method_grid(include_counterparts: bool = True) -> list[str]:
    methods = ["native_fp", "identity_cache", "sunshape_base@3.0", "sunshape_mixed@3.0", "sunshape_pro@3.0"]
    if include_counterparts:
        methods.append("turboquant_mse@3.0")
    return methods
