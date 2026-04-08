"""SunShape attention backend shim.

This is the serving-facing surface for the SunShape beta vLLM integration.
Today it exposes:

- mode toggles for capture / active / off
- typed backend registration helpers
- a typed hook installer that installs in-process backend state
- a config object that can be handed to a vLLM launcher path
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from sunshape.integration.vllm import (
    SunShapeVLLMConfig,
    SunShapeVLLMHandle,
    SunShapeVLLMRuntimeState,
    install_hooks,
)
from sunshape.triton_kernels import sunshape_attention_scores

MODE_ACTIVE = "active"
MODE_CAPTURE = "capture"
MODE_OFF = "off"

_MODE = MODE_OFF
_ACTIVE_HANDLE: SunShapeVLLMHandle | None = None
_RUNTIME_STATE: SunShapeVLLMRuntimeState | None = None


@dataclass(frozen=True)
class SunShapeQuantizedKeys:
    """Per-head quantized key cache payload for the beta backend.

    This is the narrow runtime-facing object that lets the backend consume
    block-VQ keys directly on the attention hot path without first materializing
    dense FP key tensors.
    """

    quantized_heads: tuple[Any, ...]
    batch_size: int
    num_kv_heads: int
    seq_len: int

    def head(self, batch_idx: int, kv_head_idx: int) -> Any:
        return self.quantized_heads[batch_idx * self.num_kv_heads + kv_head_idx]


def set_mode(mode: str):
    global _MODE
    _MODE = mode


def get_mode() -> str:
    return _MODE


def register_handle(handle: SunShapeVLLMHandle) -> SunShapeVLLMHandle:
    global _ACTIVE_HANDLE
    _ACTIVE_HANDLE = handle
    set_mode(MODE_ACTIVE)
    return handle


def register_runtime_state(state: SunShapeVLLMRuntimeState) -> SunShapeVLLMRuntimeState:
    global _RUNTIME_STATE
    _RUNTIME_STATE = state
    set_mode(MODE_ACTIVE)
    return state


def clear_registered_handle() -> None:
    global _ACTIVE_HANDLE
    _ACTIVE_HANDLE = None
    if _RUNTIME_STATE is None:
        set_mode(MODE_OFF)


def clear_registered_runtime() -> None:
    global _RUNTIME_STATE
    _RUNTIME_STATE = None
    if _ACTIVE_HANDLE is None:
        set_mode(MODE_OFF)


def get_registered_handle() -> SunShapeVLLMHandle | None:
    return _ACTIVE_HANDLE


def get_registered_runtime() -> SunShapeVLLMRuntimeState | None:
    return _RUNTIME_STATE


def quantize_key_states(layer_idx: int, key_states: torch.Tensor) -> SunShapeQuantizedKeys:
    runtime = get_registered_runtime()
    if runtime is None:
        raise RuntimeError("SunShape runtime state is not registered.")
    codec = runtime.codec_for_layer(int(layer_idx))
    if codec is None:
        raise RuntimeError(f"No SunShape codec is registered for layer {layer_idx}.")
    if key_states.ndim != 4:
        raise RuntimeError(
            "quantize_key_states expects key_states with shape [batch, kv_heads, seq_len, head_dim]."
        )

    batch_size, num_kv_heads, seq_len, _head_dim = key_states.shape
    quantized_heads: list[Any] = []
    for batch_idx in range(batch_size):
        for kv_head_idx in range(num_kv_heads):
            quantized_heads.append(codec.quantize(key_states[batch_idx, kv_head_idx].float()))
    return SunShapeQuantizedKeys(
        quantized_heads=tuple(quantized_heads),
        batch_size=int(batch_size),
        num_kv_heads=int(num_kv_heads),
        seq_len=int(seq_len),
    )


def _decode_keys_if_needed(layer_idx: int | None, key_states: torch.Tensor) -> torch.Tensor:
    runtime = get_registered_runtime()
    if runtime is None or layer_idx is None:
        return key_states
    codec = runtime.codec_for_layer(int(layer_idx))
    if codec is None or key_states.ndim < 2:
        return key_states
    orig_shape = key_states.shape
    orig_dtype = key_states.dtype
    flat = key_states.reshape(-1, orig_shape[-1])
    decoded = codec(flat.float())
    return decoded.to(orig_dtype).reshape(orig_shape)


def _dequantize_key_payload_if_needed(layer_idx: int | None, key_states: Any) -> Any:
    if not isinstance(key_states, SunShapeQuantizedKeys):
        return key_states
    runtime = get_registered_runtime()
    if runtime is None or layer_idx is None:
        raise RuntimeError("SunShape runtime state is required to dequantize quantized key payloads.")
    codec = runtime.codec_for_layer(int(layer_idx))
    if codec is None:
        raise RuntimeError(f"No SunShape codec is registered for layer {layer_idx}.")

    restored_batches = []
    for batch_idx in range(key_states.batch_size):
        restored_heads = []
        for kv_head_idx in range(key_states.num_kv_heads):
            restored_heads.append(codec.dequantize(key_states.head(batch_idx, kv_head_idx)))
        restored_batches.append(torch.stack(restored_heads, dim=0))
    return torch.stack(restored_batches, dim=0)


def _quantized_fast_path_unavailable_reason(
    *,
    runtime_state: SunShapeVLLMRuntimeState | None,
    layer_idx: int | None,
    query: torch.Tensor,
    key: Any,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
) -> str | None:
    if runtime_state is None or layer_idx is None:
        return "No registered runtime state for quantized attention."
    if not isinstance(key, SunShapeQuantizedKeys):
        return "Key payload is not SunShapeQuantizedKeys."
    codec = runtime_state.codec_for_layer(int(layer_idx))
    if codec is None:
        return f"No codec for layer {layer_idx}."
    if query.ndim != 4 or value.ndim != 4:
        return "Quantized attention expects query/value tensors with shape [batch, heads, seq, dim]."
    if key.batch_size != int(query.shape[0]):
        return "Quantized key batch size does not match query batch size."
    if key.batch_size != int(value.shape[0]):
        return "Quantized key batch size does not match value batch size."
    if key.num_kv_heads != int(value.shape[1]):
        return "Quantized key kv-head count does not match value kv-head count."
    if key.seq_len != int(value.shape[2]):
        return "Quantized key sequence length does not match value sequence length."
    if int(query.shape[-1]) != int(codec.head_dim):
        return "Query head_dim does not match codec head_dim."
    if int(query.shape[1]) % int(key.num_kv_heads) != 0:
        return "Query head count must be divisible by kv-head count."
    if is_causal and int(query.shape[2]) != 1:
        return "Causal quantized fast path currently targets decode (query_len == 1)."
    if attn_mask is not None and attn_mask.ndim not in {2, 3, 4}:
        return "Unsupported attention-mask rank for quantized fast path."
    return None


def _select_attn_mask_slice(
    attn_mask: torch.Tensor | None,
    *,
    batch_idx: int,
    q_head_start: int,
    q_head_end: int,
    batch_size: int,
) -> torch.Tensor | None:
    if attn_mask is None:
        return None
    if attn_mask.ndim == 2:
        return attn_mask.unsqueeze(0)
    if attn_mask.ndim == 3:
        if attn_mask.shape[0] == batch_size:
            return attn_mask[batch_idx].unsqueeze(0)
        return attn_mask[q_head_start:q_head_end]
    return attn_mask[batch_idx, q_head_start:q_head_end]


def _apply_quantized_attention(
    *,
    codec,
    query: torch.Tensor,
    key: SunShapeQuantizedKeys,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    scale: float | None,
) -> torch.Tensor:
    batch_size, num_query_heads, query_len, _head_dim = query.shape
    _batch_value, num_kv_heads, seq_len, value_dim = value.shape
    query_per_kv = num_query_heads // num_kv_heads
    output = torch.empty((batch_size, num_query_heads, query_len, value_dim), device=query.device, dtype=value.dtype)

    for batch_idx in range(batch_size):
        for kv_head_idx in range(num_kv_heads):
            q_start = kv_head_idx * query_per_kv
            q_end = q_start + query_per_kv
            query_slice = query[batch_idx, q_start:q_end]
            if scale is not None:
                query_slice = query_slice * float(scale)
            query_flat = query_slice.reshape(-1, query_slice.shape[-1])
            scores = sunshape_attention_scores(codec, query_flat, key.head(batch_idx, kv_head_idx))
            scores = scores.view(query_per_kv, query_len, seq_len)

            mask_slice = _select_attn_mask_slice(
                attn_mask,
                batch_idx=batch_idx,
                q_head_start=q_start,
                q_head_end=q_end,
                batch_size=batch_size,
            )
            if mask_slice is not None:
                scores = scores + mask_slice.to(device=scores.device, dtype=scores.dtype)

            probs = torch.softmax(scores, dim=-1).to(value.dtype)
            output[batch_idx, q_start:q_end] = torch.matmul(probs, value[batch_idx, kv_head_idx])
    return output


class SunShapeAttentionImpl:
    """Beta attention shim for vLLM integration.

    This class intentionally keeps the contract permissive so it can be used as
    a light wrapper around whatever backend object vLLM provides in the target
    runtime. If a delegate backend exposes `forward`, we decode SunShape-managed
    keys just-in-time and then hand off to the delegate. Otherwise we fall back
    to PyTorch SDPA for 4D `[B, H, S, D]` tensors.
    """

    def __init__(self, *, delegate: Any = None, layer_idx: int | None = None, runtime_state: SunShapeVLLMRuntimeState | None = None):
        self.delegate = delegate
        self.layer_idx = layer_idx
        self.runtime_state = runtime_state or get_registered_runtime()

    def forward(
        self,
        query: torch.Tensor,
        key: Any,
        value: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        scale: float | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        layer = self.layer_idx if layer_idx is None else layer_idx
        runtime_state = self.runtime_state or get_registered_runtime()
        reason = _quantized_fast_path_unavailable_reason(
            runtime_state=runtime_state,
            layer_idx=layer,
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
        if reason is None:
            codec = runtime_state.codec_for_layer(int(layer)) if runtime_state is not None else None
            if codec is None:
                raise RuntimeError(f"No codec is registered for layer {layer}.")
            return _apply_quantized_attention(
                codec=codec,
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                scale=scale,
            )
        key = _dequantize_key_payload_if_needed(layer, key)
        key = _decode_keys_if_needed(layer, key)
        delegate = self.delegate
        if delegate is not None:
            if hasattr(delegate, "forward"):
                return delegate.forward(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                    scale=scale,
                    layer_idx=layer,
                    **kwargs,
                )
            if callable(delegate):
                return delegate(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                    scale=scale,
                    layer_idx=layer,
                    **kwargs,
                )
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise RuntimeError(
                "SunShapeAttentionImpl fallback expects 4D [batch, heads, seq, dim] tensors "
                "when no delegate backend is provided."
            )
        if scale is not None:
            query = query * float(scale)
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )

    __call__ = forward


class SunShapeAttentionBackend:
    """Tiny beta backend/factory surface for vLLM experiments."""

    backend_name = "sunshape"

    def __init__(self, *, delegate_backend: Any = None):
        self.delegate_backend = delegate_backend

    def make_impl(self, *, layer_idx: int | None = None, delegate: Any = None) -> SunShapeAttentionImpl:
        return SunShapeAttentionImpl(
            delegate=delegate if delegate is not None else self.delegate_backend,
            layer_idx=layer_idx,
            runtime_state=get_registered_runtime(),
        )

    def describe(self) -> dict[str, Any]:
        return describe_backend()


def get_backend_factory(*, delegate_backend: Any = None) -> SunShapeAttentionBackend:
    return SunShapeAttentionBackend(delegate_backend=delegate_backend)


def ensure_backend_registered() -> dict[str, Any]:
    runtime = get_registered_runtime()
    handle = get_registered_handle()
    return {
        "backend": "sunshape",
        "mode": get_mode(),
        "registered": handle is not None,
        "runtime_loaded": runtime is not None,
        "install_entrypoint": "sunshape.vllm_attn_backend:install_sunshape_hooks",
        "env_backend": os.environ.get("VLLM_ATTENTION_BACKEND"),
    }


def describe_backend() -> dict[str, Any]:
    handle = get_registered_handle()
    if handle is None:
        return {
            "mode": get_mode(),
            "registered": False,
            "config": None,
            "runtime": None,
        }
    return {
        "mode": get_mode(),
        "registered": True,
        "config": handle.config.to_dict(),
        "hooks_installed": handle.hooks_installed,
        "connector_registered": handle.connector_registered,
        "runtime_patch_available": handle.runtime_patch_available,
        "notes": list(handle.notes),
        "runtime": _RUNTIME_STATE.to_dict() if _RUNTIME_STATE is not None else None,
    }


def install_sunshape_hooks(*args, **kwargs) -> SunShapeVLLMHandle:
    return register_handle(install_hooks(*args, **kwargs))


__all__ = [
    "MODE_ACTIVE",
    "MODE_CAPTURE",
    "MODE_OFF",
    "SunShapeVLLMConfig",
    "SunShapeVLLMHandle",
    "SunShapeVLLMRuntimeState",
    "SunShapeAttentionBackend",
    "SunShapeAttentionImpl",
    "SunShapeQuantizedKeys",
    "clear_registered_handle",
    "clear_registered_runtime",
    "describe_backend",
    "ensure_backend_registered",
    "get_mode",
    "get_registered_handle",
    "get_registered_runtime",
    "get_backend_factory",
    "install_sunshape_hooks",
    "quantize_key_states",
    "register_handle",
    "register_runtime_state",
    "set_mode",
]
