# Examples

These are the product-facing examples for the `sunshape` library.

Run them from the repo root:

```bash
cd /sunshape
uv run python examples/hf_quickstart.py
uv run python examples/serve_quickstart.py
uv run python examples/vllm_export.py
```

The `scripts/` tree still exists for research and paper workflows, but the
files in this folder are the ones meant to demonstrate the public library API.

`hf_quickstart.py` shows the library-first `fit -> generate` path.
`serve_quickstart.py` shows the HF-backed demo server path.
`vllm_export.py` shows the export-only vLLM bundle/config path.
