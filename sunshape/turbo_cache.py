from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from sunshape.turbo_baseline import TurboQuantMSE, TurboQuantProd


class TurboQuantLayer(DynamicLayer):
    def __init__(self, quantizer: Any = None):
        super().__init__()
        self.quantizer = quantizer

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.quantizer is not None:
            orig_shape = key_states.shape
            orig_dtype = key_states.dtype
            flat = key_states.reshape(-1, orig_shape[-1])
            flat_q = self.quantizer(flat.float())
            key_states = flat_q.to(orig_dtype).view(orig_shape)
        return super().update(key_states, value_states, *args, **kwargs)


class TurboQuantCache(DynamicCache):
    def __init__(
        self,
        config=None,
        num_layers: int | None = None,
        layer_quantizers: dict[int, Any] | None = None,
        **kwargs,
    ):
        if config is not None:
            kwargs["config"] = config
        super().__init__(**kwargs)

        layer_quantizers = layer_quantizers or {}

        if len(self.layers) == 0:
            if num_layers is None:
                raise ValueError("Must provide either config or num_layers")
            for _ in range(num_layers):
                self.layers.append(DynamicLayer())

        for i in range(len(self.layers)):
            if i in layer_quantizers:
                sliding = getattr(self.layers[i], "sliding_window", None)
                new_layer = TurboQuantLayer(quantizer=layer_quantizers[i])
                if sliding is not None:
                    new_layer.sliding_window = sliding
                self.layers[i] = new_layer

    @classmethod
    def from_quantizers(
        cls,
        quantizers_dict: dict[int, Any],
        config=None,
        num_layers: int | None = None,
    ) -> "TurboQuantCache":
        return cls(config=config, num_layers=num_layers, layer_quantizers=quantizers_dict)

    @classmethod
    def for_layers(
        cls,
        *,
        head_dims: dict[int, int],
        layers: list[int],
        config=None,
        num_layers: int | None = None,
        bits: int = 3,
        mode: str = "turboquant_mse",
        seed: int = 42,
        device: torch.device | None = None,
    ) -> "TurboQuantCache":
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        quantizers: dict[int, Any] = {}
        for layer_idx in layers:
            head_dim = int(head_dims[layer_idx])
            if mode == "turboquant_prod":
                quantizers[layer_idx] = TurboQuantProd(head_dim, bits=int(bits), seed=seed + layer_idx, device=device)
            elif mode == "turboquant_mse":
                quantizers[layer_idx] = TurboQuantMSE(head_dim, bits=int(bits), seed=seed + layer_idx, device=device)
            else:
                raise ValueError(f"Unsupported TurboQuant cache mode: {mode}")
        return cls.from_quantizers(quantizers_dict=quantizers, config=config, num_layers=num_layers)

    def has_previous_state(self, layer_idx: int | None = None) -> bool:
        return self.get_seq_length() > 0
