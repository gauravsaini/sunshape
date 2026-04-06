"""SunShapeCache: a DynamicCache subclass that quantizes keys on the fly.

This is the "zero-wrapper" evaluation path.  The model's own native
attention forward pass runs unmodified — we only intercept the KV storage
via the standard HuggingFace Cache API.  This eliminates the wrapper
fidelity issues discovered in the patched-attention evaluation path.

Usage
-----
>>> from sunshape.cache import SunShapeCache
>>> cache = SunShapeCache.from_traces(traces, layers=[3, 11], ...)
>>> out = model(input_ids, past_key_values=cache, labels=input_ids)
>>> ppl = math.exp(out.loss.item())
"""
from __future__ import annotations

import math
from typing import Any

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


class SunShapeLayer(DynamicLayer):
    """Drop-in replacement for DynamicLayer that quantise-dequantises keys
    through a fitted SunShapeBlockCodec before storing them."""

    def __init__(self, codec=None):
        super().__init__()
        self.codec = codec          # SunShapeBlockCodec | None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.codec is not None:
            # key_states shape: [batch, num_kv_heads, seq_len, head_dim]
            orig_shape = key_states.shape
            orig_dtype = key_states.dtype
            flat = key_states.reshape(-1, orig_shape[-1])   # [B*H*S, D]
            flat_q = self.codec(flat.float())                # quantise → dequantise
            key_states = flat_q.to(orig_dtype).view(orig_shape)
        return super().update(key_states, value_states, *args, **kwargs)


class IdentityLayer(DynamicLayer):
    """A control layer: same path as SunShapeLayer but with identity
    quantization (no-op).  Used to verify wrapper-free fidelity."""
    pass  # update() is inherited unchanged from DynamicLayer


class SunShapeCache(DynamicCache):
    """DynamicCache whose selected layers quantise keys via SunShape codecs.

    Layers that are NOT in ``layer_codecs`` fall through to the normal
    DynamicLayer path (unquantised FP storage).
    """

    def __init__(
        self,
        config=None,
        num_layers: int | None = None,
        layer_codecs: dict[int, Any] | None = None,
        identity_layers: set[int] | None = None,
        **kwargs,
    ):
        # Let DynamicCache initialize the correct layer list (LinearAttention vs DynamicLayer vs SlidingWindow)
        # using the config (if provided)
        if config is not None:
            kwargs["config"] = config
        super().__init__(**kwargs)

        layer_codecs = layer_codecs or {}
        identity_layers = identity_layers or set()

        # If base class didn't build layers (e.g. no config provided and no ddp_cache_data)
        if len(self.layers) == 0:
            if num_layers is None:
                raise ValueError("Must provide either config or num_layers")
            for _ in range(num_layers):
                self.layers.append(DynamicLayer())

        # Now swap out the target layers
        for i in range(len(self.layers)):
            if i in layer_codecs:
                codec = layer_codecs[i]
                # Keep sliding window properties if the original layer had them
                sliding = getattr(self.layers[i], "sliding_window", None)
                new_layer = SunShapeLayer(codec=codec)
                if sliding is not None:
                    new_layer.sliding_window = sliding
                self.layers[i] = new_layer
            elif i in identity_layers:
                sliding = getattr(self.layers[i], "sliding_window", None)
                new_layer = IdentityLayer()
                if sliding is not None:
                    new_layer.sliding_window = sliding
                self.layers[i] = new_layer

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_codecs(
        cls,
        codecs_dict: dict[int, Any],
        config=None,
        num_layers: int | None = None,
    ) -> "SunShapeCache":
        """Build a cache with pre-fitted codecs for specific layers."""
        return cls(config=config, num_layers=num_layers, layer_codecs=codecs_dict)

    @classmethod
    def identity(cls, patched_layers: list[int], config=None, num_layers: int | None = None) -> "SunShapeCache":
        """Build an identity-only cache for the wrapper fidelity control test.

        Every layer stores keys in FP without any quantisation, but the
        ``patched_layers`` go through the IdentityLayer path so we can
        confirm the DynamicLayer path itself is numerically faithful.
        """
        return cls(
            config=config,
            num_layers=num_layers,
            identity_layers=set(patched_layers),
        )

    @classmethod
    def from_traces(
        cls,
        traces: dict,
        layers: list[int],
        config=None,
        num_layers: int | None = None,
        *,
        block_dim: int = 8,
        bits_per_dim: float = 1.0,
        mode: str = "sunshape_baseline",
        n_refine_dsq: int = 3,
        cal_points: int = 4096,
        seed: int = 42,
        device: torch.device | None = None,
    ) -> "SunShapeCache":
        """Fit codecs from trace data and build a ready-to-use cache.

        Parameters
        ----------
        traces : dict
            Loaded trace artifact (the ``layers`` dict from
            ``load_trace_artifact``).
        layers : list[int]
            Which layers to quantise.
        num_layers : int
            Total number of decoder layers in the model.
        """
        from sunshape.codec import SunShapeBlockCodec

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )

        n_centroids = 2 ** int(round(bits_per_dim * block_dim))
        codecs: dict[int, SunShapeBlockCodec] = {}

        for layer_idx in layers:
            layer_data = traces[layer_idx]
            q_cal = layer_data["q"][:cal_points].float().to(device)
            k_cal = layer_data["k"][:cal_points].float().to(device)
            head_dim = q_cal.shape[-1]

            codec = SunShapeBlockCodec(
                head_dim=head_dim,
                block_dim=block_dim,
                n_centroids=n_centroids,
                n_refine_dsq=n_refine_dsq,
                mode=mode,
                device=device,
            )
            codec.fit(q_cal, k_cal, kmeans_iters=15, seed=seed)
            codecs[layer_idx] = codec

        return cls.from_codecs(codecs_dict=codecs, config=config, num_layers=num_layers)

    def has_previous_state(self, layer_idx: int | None = None) -> bool:
        """Required by Qwen3.5 hybrid models in newer transformers versions."""
        return self.get_seq_length() > 0
