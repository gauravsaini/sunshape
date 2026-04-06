# SunShape

SunShape is a lightweight research-to-production toolkit for KV-cache compression in transformer decoding.
This repository contains the core `sunshape` Python package (the `sunshape/` subdirectory from the larger workspace) with APIs for:

- fitting SunShape codecs from traces
- evaluating quality/perplexity impact with cache-level patching
- estimating memory/compression tradeoffs
- exporting runtime bundles
- comparing against TurboQuant-style baselines

## Highlights

- Cache-native quantization path (`SunShapeCache`) that avoids custom attention rewrites
- Bundle workflow (`SunShapeBundle`) for reproducible fit/export/load
- CLI for fit/eval/stats/diagnose/serve
- `--model` accepts either a Hugging Face repo ID or a full HF URL
- Support for modern HF model families, including recent Ministral/Mistral-3 style configs

## Repository Layout

- `hf.py`: model loading, trace extraction, bundle fitting/loading
- `cache.py`: dynamic cache wrapper for on-the-fly key quantization
- `eval.py`: PPL/NLL evaluation with native vs quantized cache paths
- `stats.py`: compression accounting and KV memory projections
- `turbo_baseline.py` and `turbo_cache.py`: comparison baselines
- `cli.py`: command-line entrypoints for operational workflows

## Setup

This repo is source-only package code. Install runtime dependencies in your environment first:

```bash
pip install torch transformers datasets pandas numpy
```

Then make the package importable from the parent directory of this checkout:

```bash
cd ..
python -m sunshape.cli --help
```

If your directory name is not `sunshape`, set `PYTHONPATH` so the package can be resolved.

## Quickstart

### Model Input (Repo ID or URL)

All CLI commands that take `--model` now accept either format:

- `Qwen/Qwen3-0.6B`
- `https://huggingface.co/Qwen/Qwen3-0.6B`
- `https://huggingface.co/mistralai/Ministral-3-8B-Base-2512`

### 1) Diagnose a model

```bash
cd ..
python -m sunshape.cli diagnose \
  --model Qwen/Qwen3-0.6B \
  --block-dim 8
```

### 2) Fit a bundle

```bash
cd ..
python -m sunshape.cli fit \
  --model Qwen/Qwen3-0.6B \
  --bundle-path ./artifacts/qwen3_06b.sunshape.pt \
  --mode sunshape_base \
  --bits-per-dim 3.0 \
  --layers 3 7 11 15
```

### 3) Evaluate cache quality

```bash
cd ..
python -m sunshape.cli eval \
  --model Qwen/Qwen3-0.6B \
  --traces-path ./traces_by_layer_qwen3_06b_multilayer.pt \
  --layers 3 7 11 15 \
  --modes sunshape_base turboquant_mse turboquant_prod \
  --bits-per-dim 3.0 \
  --ctx-len 256 \
  --max-eval-texts 32
```

### 4) Compute compression stats

```bash
cd ..
python -m sunshape.cli stats \
  --traces-path ./traces_by_layer_qwen3_06b_multilayer.pt \
  --bits-per-dim 1.0 2.0 3.0 4.0 \
  --contexts 2048 4096
```

## Notes

- HF model URLs are normalized automatically (including `hf.co` short links and `/tree/...` URLs).
- `--local-files-only` is supported in model-loading paths for offline workflows.
- For multimodal checkpoints used in text-only eval, SunShape falls back to compatible HF auto-model loaders when needed.
