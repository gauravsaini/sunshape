from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from sunshape.hf import SunShapeBundle, SunShapeConfig, TraceConfig, fit_sunshape_bundle

DEFAULT_CONNECTOR_NAME = "sunshape"
DEFAULT_BACKEND_MODULE = "sunshape.vllm_attn_backend"


@dataclass
class SunShapeVLLMConfig:
    model_name: str
    bundle_path: str
    mode: str
    bits_per_dim: float
    block_dim: int
    layers: list[int]
    connector_name: str = DEFAULT_CONNECTOR_NAME
    backend_module: str = DEFAULT_BACKEND_MODULE
    beta: bool = True

    def runtime_entrypoint(self) -> str:
        return f"{self.backend_module}:install_sunshape_hooks"

    def launch_env(self) -> dict[str, str]:
        return {
            "SUNSHAPE_VLLM_CONNECTOR": self.connector_name,
            "SUNSHAPE_VLLM_BUNDLE_PATH": self.bundle_path,
            "SUNSHAPE_VLLM_MODE": self.mode,
            "SUNSHAPE_VLLM_BITS_PER_DIM": str(self.bits_per_dim),
            "SUNSHAPE_VLLM_BLOCK_DIM": str(self.block_dim),
            "SUNSHAPE_VLLM_LAYERS": ",".join(str(x) for x in self.layers),
            "SUNSHAPE_VLLM_BACKEND_MODULE": self.backend_module,
        }

    def engine_args(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "kv_transfer_config": {
                "connector": self.connector_name,
                "bundle_path": self.bundle_path,
                "mode": self.mode,
                "bits_per_dim": self.bits_per_dim,
                "block_dim": self.block_dim,
                "layers": list(self.layers),
            },
            "sunshape_runtime": {
                "backend_module": self.backend_module,
                "install_entrypoint": self.runtime_entrypoint(),
                "beta": self.beta,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "bundle_path": self.bundle_path,
            "mode": self.mode,
            "bits_per_dim": self.bits_per_dim,
            "block_dim": self.block_dim,
            "layers": list(self.layers),
            "connector_name": self.connector_name,
            "backend_module": self.backend_module,
            "beta": self.beta,
            "engine_args": self.engine_args(),
            "launch_env": self.launch_env(),
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
    connector_registered: bool = False
    runtime_patch_available: bool = False
    vllm_version: str | None = None
    notes: list[str] = field(default_factory=list)

    def engine_args(self) -> dict[str, Any]:
        return self.config.engine_args()

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "hooks_installed": self.hooks_installed,
            "backend_module": self.backend_module,
            "connector_registered": self.connector_registered,
            "runtime_patch_available": self.runtime_patch_available,
            "vllm_version": self.vllm_version,
            "notes": list(self.notes),
        }

    def activate(self) -> "SunShapeVLLMHandle":
        backend = importlib.import_module(self.backend_module or self.config.backend_module)
        if hasattr(backend, "register_handle"):
            backend.register_handle(self)
            self.hooks_installed = True
            self.connector_registered = True
        return self

    def deactivate(self) -> "SunShapeVLLMHandle":
        backend = importlib.import_module(self.backend_module or self.config.backend_module)
        if hasattr(backend, "clear_registered_runtime"):
            backend.clear_registered_runtime()
        if hasattr(backend, "clear_registered_handle"):
            backend.clear_registered_handle()
        self.hooks_installed = False
        self.connector_registered = False
        return self


@dataclass
class SunShapeVLLMRuntimeState:
    bundle_path: str
    model_name: str
    mode: str
    bits_per_dim: float
    block_dim: int
    layers: list[int]
    bundle: SunShapeBundle
    device: str
    backend_module: str = DEFAULT_BACKEND_MODULE
    beta: bool = True

    def codec_for_layer(self, layer_idx: int):
        return self.bundle.codecs.get(int(layer_idx))

    def supports_layer(self, layer_idx: int) -> bool:
        return int(layer_idx) in self.bundle.codecs

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_path": self.bundle_path,
            "model_name": self.model_name,
            "mode": self.mode,
            "bits_per_dim": self.bits_per_dim,
            "block_dim": self.block_dim,
            "layers": list(self.layers),
            "device": self.device,
            "backend_module": self.backend_module,
            "beta": self.beta,
            "trace_meta": dict(getattr(self.bundle, "trace_meta", {})),
        }


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


def _validate_bundle_path(bundle_path: str | Path) -> Path:
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"SunShape vLLM bundle does not exist: {bundle_path}")
    return bundle_path


def _probe_runtime_patch(vllm_module: Any) -> tuple[bool, list[str]]:
    notes: list[str] = []
    runtime_patch_available = False
    candidate_modules = [
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        "vllm.distributed.kv_transfer.kv_connector.v1.registry",
        "vllm.attention.backends.abstract",
    ]
    for module_name in candidate_modules:
        try:
            importlib.import_module(module_name)
            runtime_patch_available = True
            notes.append(f"runtime_module_available:{module_name}")
            break
        except Exception:
            continue
    if not runtime_patch_available:
        notes.append(
            "No known vLLM connector module was detected. Backend state can still be "
            "registered in-process, but a real SunShape connector must consume it."
        )
    version = getattr(vllm_module, "__version__", None)
    if version:
        notes.append(f"vllm_version:{version}")
    return runtime_patch_available, notes


def load_runtime_state(
    *,
    bundle_path: str | Path,
    model_name: str,
    mode: str,
    bits_per_dim: float,
    block_dim: int,
    layers: Iterable[int],
    backend_module: str = DEFAULT_BACKEND_MODULE,
    device: str | torch.device | None = None,
) -> SunShapeVLLMRuntimeState:
    bundle_path = _validate_bundle_path(bundle_path)
    device_obj = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    bundle = SunShapeBundle.load(bundle_path, device=device_obj)
    runtime_layers = [int(x) for x in layers] if layers else list(bundle.layers)
    return SunShapeVLLMRuntimeState(
        bundle_path=str(bundle_path),
        model_name=model_name,
        mode=mode,
        bits_per_dim=float(bits_per_dim),
        block_dim=int(block_dim),
        layers=runtime_layers,
        bundle=bundle,
        device=str(device_obj),
        backend_module=backend_module,
        beta=True,
    )


def build_launch_env(config: SunShapeVLLMConfig) -> dict[str, str]:
    env = dict(config.launch_env())
    env["VLLM_ATTENTION_BACKEND"] = config.connector_name
    env["SUNSHAPE_VLLM_INSTALL_ENTRYPOINT"] = config.runtime_entrypoint()
    return env


def runtime_notes(config: SunShapeVLLMConfig) -> list[str]:
    return [
        "SunShape vLLM support is beta.",
        "The current path registers backend state and a JIT decode attention shim.",
        "For production claims, validate inside a real CUDA/vLLM runtime with long-context TTFT/TPOT benchmarks.",
        f"Install entrypoint: {config.runtime_entrypoint()}",
    ]


def install_hooks(
    *,
    bundle_path: str | Path,
    model_name: str,
    mode: str,
    bits_per_dim: float,
    block_dim: int,
    layers: Iterable[int],
) -> SunShapeVLLMHandle:
    bundle_path = _validate_bundle_path(bundle_path)
    try:
        vllm_module = importlib.import_module("vllm")
    except Exception as exc:
        raise RuntimeError(
            "vLLM is not installed. SunShape can still export a bundle/config, but "
            "install_hooks requires a vLLM environment."
        ) from exc

    runtime_patch_available, notes = _probe_runtime_patch(vllm_module)
    config = SunShapeVLLMConfig(
        model_name=model_name,
        bundle_path=str(bundle_path),
        mode=mode,
        bits_per_dim=float(bits_per_dim),
        block_dim=int(block_dim),
        layers=[int(x) for x in layers],
    )
    handle = SunShapeVLLMHandle(
        config=config,
        hooks_installed=False,
        backend_module=config.backend_module,
        connector_registered=False,
        runtime_patch_available=runtime_patch_available,
        vllm_version=getattr(vllm_module, "__version__", None),
        notes=notes,
    )
    handle.activate()
    backend = importlib.import_module(config.backend_module)
    if hasattr(backend, "register_runtime_state"):
        runtime_state = load_runtime_state(
            bundle_path=bundle_path,
            model_name=model_name,
            mode=mode,
            bits_per_dim=bits_per_dim,
            block_dim=block_dim,
            layers=layers,
            backend_module=config.backend_module,
        )
        backend.register_runtime_state(runtime_state)
        os.environ["VLLM_ATTENTION_BACKEND"] = config.connector_name
        handle.notes.append("Registered SunShape runtime state for beta attention backend.")
        handle.notes.extend(runtime_notes(config))
    if not handle.runtime_patch_available:
        handle.notes.append(
            "SunShape registered backend state successfully, but runtime patching is "
            "still beta/export-only until a vLLM connector consumes this state."
        )
    return handle
