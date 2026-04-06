from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from sunshape.cache import SunShapeCache
from sunshape.hf import load_model_and_tokenizer, load_trace_artifact, resolve_mode_alias
from sunshape.turbo_cache import TurboQuantCache


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_eval_texts(domain: str, max_texts: int) -> list[str]:
    domain = domain.lower()
    if domain == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [x for x in ds["text"] if isinstance(x, str) and x.strip()]
        return texts[:max_texts]
    raise ValueError(f"Unsupported eval domain: {domain}")


def get_num_layers(model) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return len(model.model.language_model.layers)
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        return len(model.language_model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise AttributeError("Could not determine the number of decoder layers.")


def get_cache_config(model):
    config = getattr(model, "config", None)
    if config is None:
        return None
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "num_hidden_layers"):
        return text_config
    return config


def compute_ppl_with_cache(
    model,
    tokenizer,
    texts: list[str],
    max_len: int,
    cache_factory,
) -> tuple[float, float]:
    losses = []
    device = model.device
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        if input_ids.shape[1] < 4:
            continue

        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        cache = cache_factory()
        if cache is not None:
            kwargs["past_key_values"] = cache

        with torch.no_grad():
            out = model(**kwargs)
        losses.append(float(out.loss.item()))

    mean_nll = float(np.mean(losses))
    return mean_nll, float(math.exp(mean_nll))


def run_cache_eval(
    model_name: str,
    traces_path: str,
    layers: list[int],
    modes: list[str],
    eval_domain: str,
    max_eval_texts: int,
    ctx_len: int,
    block_dim: int,
    bits_per_dim: float,
    seed: int,
    dsq_steps: int,
    cal_points: int,
    device_map: str,
    torch_dtype: str,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    warmup_runs: int = -1,
) -> pd.DataFrame:
    set_seed(seed)
    device = choose_device()
    traces, _trace_meta = load_trace_artifact(traces_path)
    texts = load_eval_texts(eval_domain, max_eval_texts)
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    return run_cache_eval_loaded(
        model=model,
        tokenizer=tokenizer,
        traces=traces,
        trace_meta=_trace_meta,
        texts=texts,
        layers=layers,
        modes=modes,
        eval_domain=eval_domain,
        ctx_len=ctx_len,
        block_dim=block_dim,
        bits_per_dim=bits_per_dim,
        seed=seed,
        dsq_steps=dsq_steps,
        cal_points=cal_points,
        device=device,
        warmup_runs=warmup_runs,
    )


def run_cache_eval_loaded(
    *,
    model,
    tokenizer,
    traces: dict,
    trace_meta: dict,
    texts: list[str],
    layers: list[int],
    modes: list[str],
    eval_domain: str,
    ctx_len: int,
    block_dim: int,
    bits_per_dim: float,
    seed: int,
    dsq_steps: int,
    cal_points: int,
    device: torch.device | None = None,
    warmup_runs: int = -1,
) -> pd.DataFrame:
    if device is None:
        device = choose_device()
    warmup = int(warmup_runs)
    if warmup < 0:
        warmup = 1 if device.type == "mps" else 0
    num_layers = get_num_layers(model)
    cache_config = get_cache_config(model)
    rows: list[dict[str, object]] = []

    for _ in range(warmup):
        _ = compute_ppl_with_cache(model, tokenizer, texts, ctx_len, cache_factory=lambda: None)

    nll, ppl = compute_ppl_with_cache(model, tokenizer, texts, ctx_len, cache_factory=lambda: None)
    rows.append({
        "method": "native_fp",
        "layers": ",".join(map(str, layers)),
        "seed": seed,
        "bits_per_dim": bits_per_dim,
        "eval_domain": eval_domain,
        "nll": nll,
        "ppl": ppl,
        "delta_ppl": 0.0,
        "delta_nll": 0.0,
        "fidelity_ok": True,
    })
    base_ppl = ppl
    base_nll = nll

    nll_id, ppl_id = compute_ppl_with_cache(
        model,
        tokenizer,
        texts,
        ctx_len,
        cache_factory=lambda: SunShapeCache.identity(
            patched_layers=layers,
            config=cache_config,
            num_layers=num_layers,
        ),
    )
    identity_delta = abs(ppl_id - base_ppl)
    fidelity_ok = identity_delta < 0.5
    rows.append({
        "method": "identity_cache",
        "layers": ",".join(map(str, layers)),
        "seed": seed,
        "bits_per_dim": bits_per_dim,
        "eval_domain": eval_domain,
        "nll": nll_id,
        "ppl": ppl_id,
        "delta_ppl": ppl_id - base_ppl,
        "delta_nll": nll_id - base_nll,
        "fidelity_ok": fidelity_ok,
    })

    head_dims = {int(layer_idx): int(meta_dim) for layer_idx, meta_dim in (trace_meta.get("head_dims", {}) or {}).items()}
    for mode in modes:
        if mode in {"turboquant_mse", "turboquant_prod"}:
            tq_bits = int(round(bits_per_dim))
            if abs(float(bits_per_dim) - tq_bits) > 1e-6:
                raise ValueError(f"{mode} requires integer bits_per_dim, got {bits_per_dim}")
            cache_factory = lambda mode=mode, tq_bits=tq_bits: TurboQuantCache.for_layers(
                head_dims=head_dims,
                layers=layers,
                config=cache_config,
                num_layers=num_layers,
                bits=tq_bits,
                mode=mode,
                seed=seed,
                device=device,
            )
            resolved_mode = mode
        else:
            resolved_mode = resolve_mode_alias(mode)
            cache_template = SunShapeCache.from_traces(
                traces,
                layers=layers,
                config=cache_config,
                num_layers=num_layers,
                block_dim=block_dim,
                bits_per_dim=bits_per_dim,
                mode=resolved_mode,
                n_refine_dsq=dsq_steps,
                cal_points=cal_points,
                seed=seed,
                device=device,
            )
            fitted_codecs = {}
            for i, layer in enumerate(cache_template.layers):
                if hasattr(layer, "codec") and layer.codec is not None:
                    fitted_codecs[i] = layer.codec
            cache_factory = lambda fitted_codecs=fitted_codecs: SunShapeCache.from_codecs(
                codecs_dict=fitted_codecs,
                config=cache_config,
                num_layers=num_layers,
            )

        nll_q, ppl_q = compute_ppl_with_cache(
            model,
            tokenizer,
            texts,
            ctx_len,
            cache_factory=cache_factory,
        )
        rows.append({
            "method": resolved_mode,
            "layers": ",".join(map(str, layers)),
            "seed": seed,
            "bits_per_dim": bits_per_dim,
            "eval_domain": eval_domain,
            "nll": nll_q,
            "ppl": ppl_q,
            "delta_ppl": ppl_q - base_ppl,
            "delta_nll": nll_q - base_nll,
            "fidelity_ok": fidelity_ok,
        })

    return pd.DataFrame(rows)


def save_eval_outputs(df: pd.DataFrame, output_csv: str = "", output_json: str = "") -> None:
    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(df.to_json(orient="records", indent=2))
