from __future__ import annotations

import random
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb as gemma4_apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_apply_rotary_pos_emb
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as qwen2_apply_rotary_pos_emb

from sunshape.cache import SunShapeCache
from sunshape.codec import SunShapeBlockCodec

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb as qwen3_5_apply_rotary_pos_emb
except Exception:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as qwen3_5_apply_rotary_pos_emb

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb
except Exception:
    qwen3_apply_rotary_pos_emb = qwen3_5_apply_rotary_pos_emb


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_hf_model_ref(model_ref: str) -> str:
    ref = str(model_ref).strip()
    if not ref:
        raise ValueError("model_ref must be a non-empty string")

    is_hf_url = False
    for prefix in (
        "https://huggingface.co/",
        "http://huggingface.co/",
        "https://hf.co/",
        "http://hf.co/",
        "hf.co/",
    ):
        if ref.startswith(prefix):
            ref = ref[len(prefix):]
            is_hf_url = True
            break

    ref = ref.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    if is_hf_url:
        for marker in ("/tree/", "/blob/"):
            if marker in ref:
                ref = ref.split(marker, 1)[0]
    return ref


def _hf_pretrained_kwargs(*, trust_remote_code: bool = False, local_files_only: bool = False) -> dict:
    kwargs = {"trust_remote_code": trust_remote_code}
    if local_files_only:
        kwargs["local_files_only"] = True
    return kwargs


@contextmanager
def _hf_offline_context(enabled: bool):
    if not enabled:
        yield
        return
    previous = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
        "SUNSHAPE_HF_LOCAL_ONLY": os.environ.get("SUNSHAPE_HF_LOCAL_ONLY"),
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["SUNSHAPE_HF_LOCAL_ONLY"] = "1"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    mapping = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = mapping[name]
    if device.type == "cpu" and dtype == torch.float16:
        return torch.float32
    return dtype


def _detect_model_type(model) -> str:
    model_type = str(getattr(model.config, "model_type", "")).lower()
    if "gemma4" in model_type or "gemma" in model_type:
        return "gemma4"
    if "qwen3_5" in model_type:
        return "qwen3_5"
    if "qwen3" in model_type:
        return "qwen3"
    if "qwen" in model_type:
        return "qwen2"
    return "llama"


def _get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        return model.language_model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not locate decoder layers for this model.")


def _get_num_layers(model) -> int:
    return len(_get_decoder_layers(model))


def _infer_supported_kv_layers(model) -> list[int]:
    layers = _get_decoder_layers(model)
    return [idx for idx, layer in enumerate(layers) if hasattr(layer, "self_attn")]


def _get_attention_layout(module, hidden_size_hint: int | None = None) -> tuple[int, int, int]:
    head_dim = int(getattr(module, "head_dim"))
    q_out = int(module.q_proj.weight.shape[0])
    k_out = int(module.k_proj.weight.shape[0])
    if hasattr(module, "q_norm") and hasattr(module, "k_norm"):
        num_heads = q_out // (2 * head_dim)
    else:
        num_heads = q_out // head_dim
    num_kv_heads = k_out // head_dim
    hidden_size = int(getattr(module, "o_proj").weight.shape[0] or hidden_size_hint or getattr(module, "hidden_size", 0) or q_out)
    return num_heads, num_kv_heads, hidden_size


def _apply_rope_pair(model_type: str, query_states: torch.Tensor, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    if model_type == "gemma4":
        return gemma4_apply_rotary_pos_emb(query_states, cos, sin), gemma4_apply_rotary_pos_emb(key_states, cos, sin)
    if model_type == "qwen3_5":
        return qwen3_5_apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if model_type == "qwen3":
        return qwen3_apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if model_type == "qwen2":
        return qwen2_apply_rotary_pos_emb(query_states, key_states, cos, sin)
    return llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)


def _concat_rows(chunks: list[torch.Tensor]) -> torch.Tensor | None:
    if not chunks:
        return None
    return torch.cat(chunks, dim=0)


def _load_texts(calibration_texts: Iterable[str] | None, *, dataset_name: str, dataset_config: str, split: str, num_samples: int) -> list[str]:
    if calibration_texts is not None:
        texts = [text for text in calibration_texts if isinstance(text, str) and text.strip()]
        return texts[:num_samples]
    ds = load_dataset(dataset_name, dataset_config, split=split)
    texts = [x for x in ds["text"] if isinstance(x, str) and x.strip()]
    return texts[:num_samples]


def resolve_mode_alias(mode: str) -> str:
    aliases = {
        "sunshape_base": "profileperm_baseline",
        "profileperm_baseline": "profileperm_baseline",
        "profileperm_sigmaq": "profileperm_baseline",
        "baseline": "identity_baseline",
        "identity_baseline": "identity_baseline",
        "sunshape_pro": "profileperm_localmetric_dsq",
        "profileperm_localmetric_dsq": "profileperm_localmetric_dsq",
        "profileperm_sigmaq_localmetric_dsq": "profileperm_localmetric_dsq",
        "rotated": "rotated",
    }
    if mode not in aliases:
        raise ValueError(f"Unknown SunShape mode alias: {mode}")
    return aliases[mode]


def default_block_dim(bits_per_dim: float) -> int:
    return 8 if bits_per_dim <= 1.0 else 2


@dataclass
class SunShapeConfig:
    layers: list[int] | None = None
    bits_per_dim: float = 4.0
    block_dim: int | None = None
    mode: str = "sunshape_base"
    cal_points: int = 4096
    dsq_steps: int = 3
    kmeans_iters: int = 15
    seed: int = 42

    @property
    def resolved_mode(self) -> str:
        return resolve_mode_alias(self.mode)

    @property
    def resolved_block_dim(self) -> int:
        return default_block_dim(self.bits_per_dim) if self.block_dim is None else int(self.block_dim)


@dataclass
class TraceConfig:
    num_samples: int = 16
    seq_len: int = 512
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "validation"


@dataclass
class SunShapeBundle:
    model_name: str
    layers: list[int]
    bits_per_dim: float
    block_dim: int
    mode: str
    cal_points: int
    dsq_steps: int
    seed: int
    codecs: dict[int, SunShapeBlockCodec]
    trace_meta: dict

    def make_cache(self, *, config=None, num_layers: int | None = None) -> SunShapeCache:
        return SunShapeCache.from_codecs(codecs_dict=self.codecs, config=config, num_layers=num_layers)

    def wrap_model(self, model) -> "SunShapeCausalLM":
        return SunShapeCausalLM(model=model, bundle=self)

    def state_dict(self) -> dict:
        codec_state = {}
        for layer_idx, codec in self.codecs.items():
            codec_state[str(layer_idx)] = {
                "state_dict": {k: v.detach().cpu() for k, v in codec.state_dict().items()},
                "head_dim": codec.head_dim,
                "block_dim": codec.block_dim,
                "n_centroids": codec.n_centroids,
                "c": codec.c,
                "n_refine_dsq": codec.n_refine_dsq,
                "mode": codec.mode,
            }
        return {
            "format_version": 1,
            "model_name": self.model_name,
            "layers": list(self.layers),
            "bits_per_dim": self.bits_per_dim,
            "block_dim": self.block_dim,
            "mode": self.mode,
            "cal_points": self.cal_points,
            "dsq_steps": self.dsq_steps,
            "seed": self.seed,
            "trace_meta": dict(self.trace_meta),
            "codecs": codec_state,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    @classmethod
    def load(cls, path: str | Path, *, device: torch.device | None = None) -> "SunShapeBundle":
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict) or "codecs" not in obj:
            raise ValueError("Invalid SunShape bundle file.")
        if device is None:
            device = _choose_device()

        codecs: dict[int, SunShapeBlockCodec] = {}
        for layer_idx_str, payload in obj["codecs"].items():
            codec = SunShapeBlockCodec(
                head_dim=int(payload["head_dim"]),
                block_dim=int(payload["block_dim"]),
                n_centroids=int(payload["n_centroids"]),
                c=float(payload.get("c", 5.0)),
                n_refine_dsq=int(payload.get("n_refine_dsq", 3)),
                mode=str(payload["mode"]),
                device=device,
            )
            codec.load_state_dict(payload["state_dict"])
            codec.to(device)
            codecs[int(layer_idx_str)] = codec
        return cls(
            model_name=str(obj["model_name"]),
            layers=[int(x) for x in obj["layers"]],
            bits_per_dim=float(obj["bits_per_dim"]),
            block_dim=int(obj["block_dim"]),
            mode=str(obj["mode"]),
            cal_points=int(obj["cal_points"]),
            dsq_steps=int(obj["dsq_steps"]),
            seed=int(obj["seed"]),
            codecs=codecs,
            trace_meta=dict(obj.get("trace_meta", {})),
        )


class SunShapeCausalLM(torch.nn.Module):
    def __init__(self, model, bundle: SunShapeBundle):
        super().__init__()
        self.model = model
        self.bundle = bundle

    def _inject_cache(self, kwargs: dict) -> dict:
        updated = dict(kwargs)
        if updated.get("past_key_values") is None and updated.get("use_cache", True):
            updated["past_key_values"] = self.bundle.make_cache(
                config=self.model.config,
                num_layers=_get_num_layers(self.model),
            )
        return updated

    def forward(self, *args, **kwargs):
        return self.model(*args, **self._inject_cache(kwargs))

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **self._inject_cache(kwargs))

    def unwrap(self):
        return self.model

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class SunShapeForCausalLM(SunShapeCausalLM):
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        traces_path: str | Path | None = None,
        bundle_path: str | Path | None = None,
        output_traces_path: str | Path | None = None,
        output_bundle_path: str | Path | None = None,
        calibration_texts: Iterable[str] | None = None,
        sunshape_config: SunShapeConfig | None = None,
        trace_config: TraceConfig | None = None,
        device_map: str = "none",
        torch_dtype: str = "float16",
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> tuple["SunShapeForCausalLM", object, SunShapeBundle]:
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if bundle_path is not None:
            bundle = SunShapeBundle.load(bundle_path, device=model.device)
        else:
            bundle, _, _ = fit_sunshape_bundle(
                model_name,
                model=model,
                tokenizer=tokenizer,
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
        if output_bundle_path is not None:
            bundle.save(output_bundle_path)
        return cls(model=model, bundle=bundle), tokenizer, bundle


def load_model_and_tokenizer(
    model_name: str,
    *,
    device_map: str = "none",
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
):
    model_name = normalize_hf_model_ref(model_name)
    device = _choose_device()
    dtype = _resolve_dtype(torch_dtype, device)
    load_kwargs = {
        "dtype": dtype,
        **_hf_pretrained_kwargs(trust_remote_code=trust_remote_code, local_files_only=local_files_only),
    }
    if device_map != "none":
        load_kwargs["device_map"] = device_map
    with _hf_offline_context(local_files_only):
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except ValueError as exc:
            # Mistral 3 / Ministral 3 checkpoints can be published as
            # conditional-generation models (multimodal wrapper) while still
            # supporting text-only forward passes for our eval stack.
            if AutoModelForImageTextToText is None or "Mistral3Config" not in str(exc):
                raise
            model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        if device_map == "none":
            model = model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **_hf_pretrained_kwargs(trust_remote_code=trust_remote_code, local_files_only=local_files_only),
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def extract_trace_artifact(
    *,
    model,
    tokenizer,
    layers: list[int],
    trace_config: TraceConfig,
    calibration_texts: Iterable[str] | None = None,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict]:
    model_type = _detect_model_type(model)
    decoder_layers = _get_decoder_layers(model)

    unsupported = [layer_idx for layer_idx in layers if not hasattr(decoder_layers[layer_idx], "self_attn")]
    if unsupported:
        raise ValueError(
            f"Requested layers {unsupported} do not expose standard KV self-attention. "
            "For hybrid models, pass only the full-attention layers."
        )

    traces: dict[int, dict[str, list[torch.Tensor]]] = {layer: {"q": [], "k": []} for layer in layers}
    hooks = []

    def make_attn_hook(layer_idx: int):
        def hook(module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            if hidden_states is None:
                raise RuntimeError("Could not recover hidden_states for SunShape trace extraction.")
            bsz, q_len, _ = hidden_states.shape
            if hasattr(module, "q_norm") and hasattr(module, "k_norm"):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, module.head_dim)
                query_states, _gate = torch.chunk(
                    module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim * 2),
                    2,
                    dim=-1,
                )
                query_states = module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            else:
                num_heads, num_kv_heads, _ = _get_attention_layout(module, hidden_size_hint=hidden_states.shape[-1])
                query_states = module.q_proj(hidden_states).view(bsz, q_len, num_heads, module.head_dim).transpose(1, 2)
                key_states = module.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, module.head_dim).transpose(1, 2)

            position_embeddings = kwargs.get("position_embeddings", None)
            position_ids = kwargs.get("position_ids", None)
            if position_embeddings is None:
                rotary = getattr(module, "rotary_emb", None)
                if rotary is None:
                    raise AttributeError("attention module needs position_embeddings or rotary_emb")
                position_embeddings = rotary(hidden_states, position_ids)
            cos, sin = position_embeddings
            query_states, key_states = _apply_rope_pair(model_type, query_states, key_states, cos, sin)

            q_flat = query_states.detach().cpu().float().permute(0, 2, 1, 3).reshape(-1, module.head_dim)
            k_flat = key_states.detach().cpu().float().permute(0, 2, 1, 3).reshape(-1, module.head_dim)
            traces[layer_idx]["q"].append(q_flat)
            traces[layer_idx]["k"].append(k_flat)

        return hook

    for layer_idx in layers:
        hooks.append(decoder_layers[layer_idx].self_attn.register_forward_hook(make_attn_hook(layer_idx), with_kwargs=True))

    texts = _load_texts(
        calibration_texts,
        dataset_name=trace_config.dataset_name,
        dataset_config=trace_config.dataset_config,
        split=trace_config.split,
        num_samples=trace_config.num_samples,
    )
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=trace_config.seq_len)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            model(**enc)

    for hook in hooks:
        hook.remove()

    packed: dict[int, dict[str, torch.Tensor]] = {}
    head_dims: dict[str, int] = {}
    for layer_idx in layers:
        q = _concat_rows(traces[layer_idx]["q"])
        k = _concat_rows(traces[layer_idx]["k"])
        if q is None or k is None:
            continue
        packed[layer_idx] = {"q": q, "k": k}
        head_dims[str(layer_idx)] = int(k.shape[-1])

    meta = {
        "model_name": getattr(model.config, "_name_or_path", "unknown"),
        "model_type": model_type,
        "seq_len": trace_config.seq_len,
        "num_samples": trace_config.num_samples,
        "layers_to_test": list(layers),
        "trace_space": "per_head_post_rope",
        "head_dims": head_dims,
        "format_version": 2,
    }
    return packed, meta


def load_trace_artifact(trace_path: str | Path) -> tuple[dict[int, dict[str, torch.Tensor]], dict]:
    obj = torch.load(trace_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "__meta__" in obj and "layers" in obj:
        return obj["layers"], dict(obj["__meta__"])
    raise ValueError("Trace file missing metadata or wrong format.")


def save_trace_artifact(traces: dict[int, dict[str, torch.Tensor]], meta: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"__meta__": meta, "layers": traces}, output_path)
    return output_path


def fit_bundle_from_traces(
    *,
    model_name: str,
    traces: dict[int, dict[str, torch.Tensor]],
    trace_meta: dict,
    sunshape_config: SunShapeConfig,
    device: torch.device | None = None,
) -> SunShapeBundle:
    if sunshape_config.layers is None:
        layers = sorted(traces.keys())
    else:
        layers = list(sunshape_config.layers)
    if device is None:
        device = _choose_device()

    block_dim = sunshape_config.resolved_block_dim
    n_centroids = 2 ** int(round(sunshape_config.bits_per_dim * block_dim))
    codecs: dict[int, SunShapeBlockCodec] = {}

    for layer_idx in layers:
        layer_data = traces[layer_idx]
        q_cal = layer_data["q"][: sunshape_config.cal_points].float().to(device)
        k_cal = layer_data["k"][: sunshape_config.cal_points].float().to(device)
        head_dim = q_cal.shape[-1]

        codec = SunShapeBlockCodec(
            head_dim=head_dim,
            block_dim=block_dim,
            n_centroids=n_centroids,
            n_refine_dsq=sunshape_config.dsq_steps,
            mode=sunshape_config.resolved_mode,
            device=device,
        )
        codec.fit(q_cal, k_cal, kmeans_iters=sunshape_config.kmeans_iters, seed=sunshape_config.seed)
        codecs[layer_idx] = codec

    return SunShapeBundle(
        model_name=model_name,
        layers=layers,
        bits_per_dim=sunshape_config.bits_per_dim,
        block_dim=block_dim,
        mode=sunshape_config.resolved_mode,
        cal_points=sunshape_config.cal_points,
        dsq_steps=sunshape_config.dsq_steps,
        seed=sunshape_config.seed,
        codecs=codecs,
        trace_meta=dict(trace_meta),
    )


def fit_sunshape_bundle(
    model_name: str,
    *,
    model=None,
    tokenizer=None,
    traces_path: str | Path | None = None,
    output_traces_path: str | Path | None = None,
    calibration_texts: Iterable[str] | None = None,
    sunshape_config: SunShapeConfig | None = None,
    trace_config: TraceConfig | None = None,
    device_map: str = "none",
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> tuple[SunShapeBundle, object | None, object | None]:
    sunshape_config = SunShapeConfig() if sunshape_config is None else sunshape_config
    trace_config = TraceConfig() if trace_config is None else trace_config
    _set_seed(sunshape_config.seed)

    owns_model = model is None or tokenizer is None
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

    if traces_path is not None:
        traces, trace_meta = load_trace_artifact(traces_path)
    else:
        layers = sunshape_config.layers or _infer_supported_kv_layers(model)
        sunshape_config = SunShapeConfig(
            layers=layers,
            bits_per_dim=sunshape_config.bits_per_dim,
            block_dim=sunshape_config.block_dim,
            mode=sunshape_config.mode,
            cal_points=sunshape_config.cal_points,
            dsq_steps=sunshape_config.dsq_steps,
            kmeans_iters=sunshape_config.kmeans_iters,
            seed=sunshape_config.seed,
        )
        traces, trace_meta = extract_trace_artifact(
            model=model,
            tokenizer=tokenizer,
            layers=layers,
            trace_config=trace_config,
            calibration_texts=calibration_texts,
        )
        if output_traces_path is not None:
            save_trace_artifact(traces, trace_meta, output_traces_path)

    bundle = fit_bundle_from_traces(
        model_name=model_name,
        traces=traces,
        trace_meta=trace_meta,
        sunshape_config=sunshape_config,
        device=model.device if hasattr(model, "device") else _choose_device(),
    )
    return bundle, (model if owns_model else None), (tokenizer if owns_model else None)


def prepare_sunshape_model(
    model_name: str,
    *,
    traces_path: str | Path | None = None,
    output_traces_path: str | Path | None = None,
    calibration_texts: Iterable[str] | None = None,
    sunshape_config: SunShapeConfig | None = None,
    trace_config: TraceConfig | None = None,
    device_map: str = "none",
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> tuple[SunShapeCausalLM, object, SunShapeBundle]:
    bundle, model, tokenizer = fit_sunshape_bundle(
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
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    return SunShapeForCausalLM(model=model, bundle=bundle), tokenizer, bundle
