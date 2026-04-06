from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sunshape.hf import SunShapeBundle, SunShapeConfig, TraceConfig, fit_sunshape_bundle


@dataclass
class SunShapeVLLMConfig:
    model_name: str
    bundle_path: str
    mode: str
    bits_per_dim: float
    block_dim: int
    layers: list[int]

    def engine_args(self) -> dict:
        return {
            "model": self.model_name,
            "kv_transfer_config": {
                "connector": "sunshape",
                "bundle_path": self.bundle_path,
                "mode": self.mode,
                "bits_per_dim": self.bits_per_dim,
                "block_dim": self.block_dim,
                "layers": list(self.layers),
            },
        }

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "bundle_path": self.bundle_path,
            "mode": self.mode,
            "bits_per_dim": self.bits_per_dim,
            "block_dim": self.block_dim,
            "layers": list(self.layers),
            "engine_args": self.engine_args(),
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


@dataclass
class SunShapeVLLMHandle:
    config: SunShapeVLLMConfig
    hooks_installed: bool
    backend_module: str | None = None

    def engine_args(self) -> dict:
        return self.config.engine_args()


def export_bundle_for_vllm(bundle: SunShapeBundle, path: str | Path) -> SunShapeVLLMConfig:
    path = Path(path)
    bundle.save(path)
    return SunShapeVLLMConfig(
        model_name=bundle.model_name,
        bundle_path=str(path),
        mode=bundle.mode,
        bits_per_dim=bundle.bits_per_dim,
        block_dim=bundle.block_dim,
        layers=list(bundle.layers),
    )


def prepare_vllm_bundle(
    model_name: str,
    *,
    traces_path: str | Path | None = None,
    bundle_path: str | Path,
    sunshape_config: SunShapeConfig | None = None,
    output_traces_path: str | Path | None = None,
    calibration_texts: Iterable[str] | None = None,
    trace_config: TraceConfig | None = None,
    device_map: str = "none",
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> SunShapeVLLMConfig:
    bundle, _, _ = fit_sunshape_bundle(
        model_name,
        traces_path=traces_path,
        output_traces_path=output_traces_path,
        calibration_texts=calibration_texts,
        sunshape_config=sunshape_config,
        trace_config=trace_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    return export_bundle_for_vllm(bundle, bundle_path)


def install_hooks(
    *,
    bundle_path: str | Path,
    model_name: str,
    mode: str,
    bits_per_dim: float,
    block_dim: int,
    layers: Iterable[int],
) -> SunShapeVLLMHandle:
    try:
        import vllm  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "vLLM is not installed. SunShape now provides a concrete bundle/config "
            "surface for serving, but the runtime hook still requires a vLLM environment."
        ) from exc

    config = SunShapeVLLMConfig(
        model_name=model_name,
        bundle_path=str(bundle_path),
        mode=mode,
        bits_per_dim=float(bits_per_dim),
        block_dim=int(block_dim),
        layers=[int(x) for x in layers],
    )
    return SunShapeVLLMHandle(
        config=config,
        hooks_installed=True,
        backend_module="sunshape.vllm_attn_backend",
    )
