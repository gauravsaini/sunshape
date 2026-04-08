# SunShape

SunShape studies KV-cache compression under a fixed block codec. The current
paper-facing story is:

- dense mixing is mismatched to fixed block codecs because it spreads
  query-relevant coupling across blocks
- `ProfilePerm(SigmaQ)` is the reusable structural signal
- `SunShape` is the mainline method
- generation decides final method choice; proxy metrics are supporting evidence

## Separation Of Concerns

The repo is now organized around a simple rule:

- `sunshape/` is the production-facing library surface
- `scripts/eval/`, `scripts/paper_plots/`, and `bash_scripts/` are research and
  evaluation consumers of that library

The intent is that new experiments should call the public `sunshape` API rather
than re-implementing model patching or codec fitting logic inside scripts.

## Current Method Names

| Paper Name | Runtime Name | Meaning |
|:--|:--|:--|
| `Baseline BlockVQ` | `baseline_blockvq` | Plain blockwise VQ in the original layout |
| `SunShape` | `profileperm_sigmaq` | `ProfilePerm(SigmaQ)` plus plain block VQ |
| `SunShape-Mixed` | `profileperm_mixed_precision` | `ProfilePerm(SigmaQ)` plus high-precision heavy blocks plus VQ on light blocks |
| `SunShape-Pro` | `profileperm_sigmaq_localmetric_dsq` | `ProfilePerm(SigmaQ)` plus local metric plus DSQ |
| `Baseline FP` | `baseline_fp` | Uncompressed reference |

## Current Status Of `SunShape-Mixed`

As of `2026-04-07`, the local validation result is: do **not** deprecate
`SunShape-Pro` yet.

- Validation artifact: `tmp/mixed_vs_pro_validate_qwen05b/pareto_summary.csv`
- Setup: `Qwen/Qwen2.5-0.5B-Instruct` with `runs/traces/traces_by_layer_qwen.pt`
  on layer `8`
- Cache-PPL result: `SunShape-Mixed` beat `SunShape-Pro` at `1.0` bit/dim
  (`+0.494` vs `+0.687` delta PPL) and `3.0` bit/dim (`-0.059` vs `-0.033`),
  but lost at `4.0` bit/dim (`+0.203` vs `+0.086`)
- `SunShape-Mixed` is also not rate-fair in the current implementation because
  it keeps some blocks in high precision; on this validated layer its effective
  key rate was about `4.75` bits/dim at nominal `1.0`, `4.625` at nominal
  `3.0`, and `5.5` at nominal `4.0`
- Current product-facing decision: keep `SunShape`, `SunShape-Mixed`, and
  `SunShape-Pro` all available until `SunShape-Mixed` wins cleanly at matched
  effective rate

## Repository Layout

```text
sunshape/
├── bash_scripts/          # shell entrypoints
├── examples/              # product-facing quickstarts
├── docs/                  # notes and supporting writeups
├── scripts/
│   ├── eval/             # evaluation pipelines
│   ├── paper_plots/      # mechanism / paper utilities
│   └── notebooks/        # notebook builders
├── sunshape/             # core library code
├── pyproject.toml
└── uv.lock
```

## Environment

Python is managed with `uv`.

```bash
cd /turboquant/0204/sunshape
uv sync
```

The project currently uses exact dependency pins for reproducibility of the
paper runs.

For a quick freeze/check pass before a talk or demo, see
[CONFERENCE_DEMO_CHECKLIST.md](/turboquant/0204/sunshape/CONFERENCE_DEMO_CHECKLIST.md).

## Install -> Fit -> Eval -> Serve

For the library-first path, these are the main commands:

```bash
cd /turboquant/0204/sunshape
uv run sunshape fit --model Qwen/Qwen3.5-0.8B --bundle-path local_runs/qwen08b_sunshape.pt
uv run sunshape eval --model Qwen/Qwen3.5-0.8B \
  --traces-path local_runs/traces/traces_by_layer_qwen08b.pt \
  --layers 3 7 11 15 19 23 \
  --bits-per-dim 1.0 \
  --local-files-only
uv run sunshape stats --traces-path local_runs/traces/traces_by_layer_qwen08b.pt
uv run sunshape serve --model Qwen/Qwen3.5-0.8B \
  --bundle-path local_runs/qwen08b_sunshape.pt \
  --local-files-only
uv run sunshape export-vllm --model Qwen/Qwen3.5-0.8B \
  --traces-path local_runs/traces/traces_by_layer_qwen08b.pt \
  --bundle-path local_runs/qwen08b_vllm_bundle.pt
uv run sunshape bundle-info --bundle-path local_runs/qwen08b_sunshape.pt
```

Runnable examples:

```bash
uv run python examples/hf_quickstart.py
uv run python examples/serve_quickstart.py
uv run python examples/vllm_export.py
```

## CLI

The package now exposes a CLI:

```bash
cd /turboquant/0204/sunshape
uv run sunshape --help
```

Main commands:

```bash
uv run sunshape fit --model Qwen/Qwen3.5-0.8B --bundle-path local_runs/qwen08b_sunshape.pt
uv run sunshape diagnose --model Qwen/Qwen3.5-4B
uv run sunshape stats --traces-path local_runs/traces/traces_by_layer_qwen08b.pt
uv run sunshape eval --model Qwen/Qwen3.5-0.8B \
  --traces-path local_runs/traces/traces_by_layer_qwen08b.pt \
  --layers 3 7 11 15 19 23 \
  --bits-per-dim 1.0 \
  --local-files-only
uv run sunshape serve --model Qwen/Qwen3.5-0.8B \
  --bundle-path local_runs/qwen08b_sunshape.pt \
  --local-files-only
uv run sunshape export-vllm --model Qwen/Qwen3.5-0.8B \
  --traces-path local_runs/traces/traces_by_layer_qwen08b.pt \
  --bundle-path local_runs/qwen08b_vllm_bundle.pt
uv run sunshape bundle-info --bundle-path local_runs/qwen08b_sunshape.pt
```

`sunshape eval` is the current quality-degradation command. It runs the clean
cache-based teacher-forced evaluation path and reports:

- `native_fp`
- `identity_cache` fidelity control
- quantized method rows
- `delta_ppl` and `delta_nll` versus native fp

`sunshape serve` is the current demo-serving command. It exposes a small HTTP
API with `/health` and `/generate` for the HF-backed SunShape runtime.

## Main Entry Points

### Library-first Hugging Face usage

```python
from sunshape import SunShapeConfig, SunShapeForCausalLM

model, tokenizer, bundle = SunShapeForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B",
    sunshape_config=SunShapeConfig(
        mode="sunshape_base",
        bits_per_dim=4.0,
        layers=[3, 7, 11, 15, 19, 23],
    ),
)

inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=24)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

This path:

- loads the model/tokenizer
- auto-fits SunShape codecs from calibration text
- wraps the Hugging Face model so `forward()` / `generate()` inject a fresh
  `SunShapeCache` automatically

If you already have traces, use `fit_sunshape_bundle(...)` instead.

### vLLM-facing bundle export

```python
from sunshape import SunShapeConfig, prepare_vllm_bundle

config = prepare_vllm_bundle(
    "Qwen/Qwen3.5-0.8B",
    traces_path="local_runs/traces/traces_by_layer_qwen08b.pt",
    bundle_path="local_runs/sunshape_vllm_bundle.pt",
    sunshape_config=SunShapeConfig(mode="sunshape_base", bits_per_dim=4.0),
)

print(config.engine_args())
```

This is the current vLLM-facing contract:

- fit and export a reusable SunShape bundle
- produce structured engine config for a vLLM runtime
- expose a beta attention-backend registration path for Colab/CUDA experiments

This is intentionally export/config only. `sunshape serve` is the currently
supported demo server for HF-backed inference; the vLLM runtime hook remains a
separate integration surface.

### Colab / CUDA deployment benchmark

For real TTFT / TPOT measurements, use the CUDA path and the dedicated script:

```bash
uv run python scripts/eval/vllm_deployment_benchmark.py \
  --model Qwen/Qwen3.5-0.8B \
  --bundle-path local_runs/qwen08b_vllm_bundle.pt \
  --traces-path local_runs/traces/traces_by_layer_qwen08b.pt \
  --output-json local_runs/vllm_qwen08b_benchmark.json
```

This is intended for Colab or another NVIDIA environment where `vllm` is
available.

### Full local bench

```bash
cd /turboquant/0204/sunshape
bash ./bash_scripts/run_local_bench.sh qwen08b
```

This runs:

- trace extraction
- real-trace proxy evaluation
- cache-based Wikitext PPL
- NIAH / grounded-QA
- deployment microbench
- Pareto compilation

### Block-structure mechanism suite

```bash
cd /turboquant/0204/sunshape
bash ./bash_scripts/run_block_structure_suite.sh qwen08b
```

This is the canonical mechanism-study path. It compares:

- `identity`
- `profileperm_sigmaq`
- `dense_random_rotation`
- `hadamard`

under the same fixed block codec and writes:

- `block_structure_suite_summary.csv`
- `block_structure_suite_layerwise.csv`
- `block_structure_suite_report.md`

### Clean cache-based Wikitext PPL

```bash
cd /turboquant/0204/sunshape
uv run python -m scripts.cache_eval \
  --model-name Qwen/Qwen3.5-0.8B \
  --traces-path local_runs/traces/traces_by_layer_qwen08b.pt \
  --layers 3 \
  --eval-domain wikitext \
  --max-eval-texts 16
```

This path uses `SunShapeCache` and avoids the old patched-attention wrapper.

### Minimal runnable example

```bash
cd /turboquant/0204/sunshape
uv run python examples/hf_quickstart.py
uv run python examples/serve_quickstart.py
```

For the serving-facing export example:

```bash
cd /turboquant/0204/sunshape
uv run python examples/vllm_export.py
```

## Quick Diagnostic

```python
from sunshape import diagnose_model

report = diagnose_model("Qwen/Qwen3.5-4B")
print(report)
```

## Current Evaluation Doctrine

Use metrics in this order:

1. generation quality
2. degradation vs `baseline_fp`
3. proxy metrics such as logit MSE / KL / attention agreement
4. memory and latency tradeoffs

Pareto-style comparison is preferred over collapsing everything to one number.

## Notes

- Colab notebooks under `colabs/` are optional helpers and are not required for
  local reproduction.
- `local_runs/` is intentionally ignored; paper evidence should be regenerated
  from the checked-in scripts, not committed run artifacts.

Restored vllm_attn_backend.py
{
  "benchmark_version": "2026-04-07-decode-full-cache-v2",
  "device": "cuda",
  "num_tokens": 4096,
  "query_len": 1,
  "native_dense_ms": 0.17654815673828125,
  "native_dense_std_ms": 0.04040263472941984,
  "sunshape_quantized_hot_ms": 0.3221126556396484,
  "sunshape_quantized_hot_std_ms": 0.007147611025452602,
  "sunshape_dequantized_ref_ms": 0.14155776023864747,
  "sunshape_dequantized_ref_std_ms": 0.0038512209929443607,
  "speedup_hot_over_dequantized_x": 0.43946662063787384,
  "max_abs_diff_vs_dequantized_ref": 4.284083843231201e-08,
  "mean_abs_diff_vs_dequantized_ref": 9.481539109401638e-09,
  "correctness_ok": true
}