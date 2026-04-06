"""SunShape attention backend shim.

This is the serving-facing surface for a future vLLM runtime integration.
Today it exposes:

- mode toggles for capture / active / off
- a typed hook installer
- a config object that can be handed to a vLLM launcher path
"""

from __future__ import annotations

from sunshape.integration.vllm import SunShapeVLLMConfig, SunShapeVLLMHandle, install_hooks

MODE_ACTIVE = "active"
MODE_CAPTURE = "capture"
MODE_OFF = "off"

_MODE = MODE_OFF


def set_mode(mode: str):
    global _MODE
    _MODE = mode


def get_mode() -> str:
    return _MODE


def install_sunshape_hooks(*args, **kwargs) -> SunShapeVLLMHandle:
    return install_hooks(*args, **kwargs)


__all__ = [
    "MODE_ACTIVE",
    "MODE_CAPTURE",
    "MODE_OFF",
    "SunShapeVLLMConfig",
    "SunShapeVLLMHandle",
    "get_mode",
    "install_sunshape_hooks",
    "set_mode",
]
