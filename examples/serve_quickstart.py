from __future__ import annotations

from sunshape import SunShapeConfig
from sunshape.server import build_runtime, serve_runtime


def main() -> None:
    runtime = build_runtime(
        model_name="Qwen/Qwen3.5-0.8B",
        bundle_path="local_runs/qwen08b_sunshape.pt",
        local_files_only=True,
        sunshape_config=SunShapeConfig(
            mode="sunshape_base",
            bits_per_dim=4.0,
            layers=[3, 7, 11, 15, 19, 23],
        ),
    )
    print("serving bundle layers:", runtime.bundle.layers)
    serve_runtime(runtime, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
