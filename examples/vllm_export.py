from __future__ import annotations

from sunshape import SunShapeConfig, prepare_vllm_bundle


def main() -> None:
    config = prepare_vllm_bundle(
        "Qwen/Qwen3.5-0.8B",
        traces_path="local_runs/traces/traces_by_layer_qwen08b.pt",
        bundle_path="local_runs/qwen08b_vllm_bundle.pt",
        sunshape_config=SunShapeConfig(
            mode="sunshape_base",
            bits_per_dim=4.0,
            layers=[3, 7, 11, 15, 19, 23],
        ),
        local_files_only=True,
    )
    print(config.to_dict())


if __name__ == "__main__":
    main()
