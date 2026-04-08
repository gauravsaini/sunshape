from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from sunshape.diagnose import diagnose_from_traces, diagnose_model
from sunshape.eval import run_cache_eval, save_eval_outputs
from sunshape.hf import (
    SunShapeBundle,
    SunShapeConfig,
    SunShapeForCausalLM,
    TraceConfig,
    default_block_dim,
    fit_sunshape_bundle,
    load_trace_artifact,
)
from sunshape.server import SunShapeServeConfig
from sunshape.server import build_runtime, serve_runtime
from sunshape.stats import build_compression_stats, load_trace_meta


def _add_pareto_arguments(parser: argparse.ArgumentParser) -> None:
    add_pareto_arguments, _ = _load_pareto_driver()

    add_pareto_arguments(parser)


def _load_pareto_driver():
    module_name = "sunshape._pareto_compare_driver"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached.add_pareto_arguments, cached.run_pareto_compare
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "eval" / "pareto_compare_driver.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Could not load Pareto driver from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.add_pareto_arguments, module.run_pareto_compare


@dataclass
class _LocalVLLMConfig:
    model_name: str
    bundle_path: str
    mode: str
    bits_per_dim: float
    block_dim: int
    layers: list[int]

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "bundle_path": self.bundle_path,
            "mode": self.mode,
            "bits_per_dim": self.bits_per_dim,
            "block_dim": self.block_dim,
            "layers": list(self.layers),
            "engine_args": {
                "model": self.model_name,
                "kv_transfer_config": {
                    "connector": "sunshape",
                    "bundle_path": self.bundle_path,
                    "mode": self.mode,
                    "bits_per_dim": self.bits_per_dim,
                    "block_dim": self.block_dim,
                    "layers": list(self.layers),
                },
            },
        }

    def save_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2))
        return output_path


def _add_common_sunshape_args(parser: argparse.ArgumentParser, *, include_layers: bool = True) -> None:
    parser.add_argument("--mode", default="sunshape_base", choices=["sunshape_base", "sunshape_mixed", "sunshape_pro"])
    parser.add_argument("--bits-per-dim", type=float, default=4.0)
    parser.add_argument("--block-dim", type=int, default=0, help="Defaults to 8 at 1-bit and 2 otherwise.")
    if include_layers:
        parser.add_argument("--layers", nargs="*", type=int, default=None)
    parser.add_argument("--cal-points", type=int, default=4096)
    parser.add_argument("--dsq-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)


def _sunshape_config_from_args(args: argparse.Namespace) -> SunShapeConfig:
    return SunShapeConfig(
        layers=args.layers,
        bits_per_dim=args.bits_per_dim,
        block_dim=None if int(args.block_dim) == 0 else int(args.block_dim),
        mode=args.mode,
        cal_points=args.cal_points,
        dsq_steps=args.dsq_steps,
        seed=args.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sunshape", description="Production-facing CLI for the SunShape library.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit SunShape and save a reusable bundle.")
    fit_parser.add_argument("--model", required=True)
    fit_parser.add_argument("--bundle-path", required=True)
    fit_parser.add_argument("--traces-path", default="")
    fit_parser.add_argument("--output-traces-path", default="")
    fit_parser.add_argument("--device-map", default="none")
    fit_parser.add_argument("--torch-dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    fit_parser.add_argument("--trust-remote-code", action="store_true")
    fit_parser.add_argument("--local-files-only", action="store_true")
    fit_parser.add_argument("--num-samples", type=int, default=16)
    fit_parser.add_argument("--seq-len", type=int, default=512)
    fit_parser.add_argument("--dataset-name", default="wikitext")
    fit_parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    fit_parser.add_argument("--split", default="validation")
    _add_common_sunshape_args(fit_parser)

    diagnose_parser = subparsers.add_parser("diagnose", help="Diagnose a model or an existing trace artifact.")
    diagnose_parser.add_argument("--model", default="")
    diagnose_parser.add_argument("--traces-path", default="")
    diagnose_parser.add_argument("--block-dim", type=int, default=8)
    diagnose_parser.add_argument("--max-queries", type=int, default=8192)
    diagnose_parser.add_argument("--output-json", default="")

    vllm_parser = subparsers.add_parser("export-vllm", help="Export a SunShape bundle and vLLM config payload.")
    vllm_parser.add_argument("--model", required=True)
    vllm_parser.add_argument("--traces-path", default="")
    vllm_parser.add_argument("--bundle-path", required=True)
    vllm_parser.add_argument("--output-traces-path", default="")
    vllm_parser.add_argument("--device-map", default="none")
    vllm_parser.add_argument("--torch-dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    vllm_parser.add_argument("--trust-remote-code", action="store_true")
    vllm_parser.add_argument("--local-files-only", action="store_true")
    vllm_parser.add_argument("--num-samples", type=int, default=16)
    vllm_parser.add_argument("--seq-len", type=int, default=512)
    vllm_parser.add_argument("--dataset-name", default="wikitext")
    vllm_parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    vllm_parser.add_argument("--split", default="validation")
    vllm_parser.add_argument("--output-json", default="")
    _add_common_sunshape_args(vllm_parser)

    stats_parser = subparsers.add_parser("stats", help="Estimate KV/cache memory savings for a trace artifact.")
    stats_parser.add_argument("--model", default="")
    stats_parser.add_argument("--traces-path", required=True)
    stats_parser.add_argument("--bits-per-dim", nargs="+", type=float, default=[1.0, 3.0, 4.0])
    stats_parser.add_argument("--contexts", nargs="+", type=int, default=[2048, 4096])
    stats_parser.add_argument("--key-baseline-bits", type=int, default=16)
    stats_parser.add_argument("--value-bits", type=int, default=16)
    stats_parser.add_argument("--centroid-bits", type=int, default=32)
    stats_parser.add_argument("--output-csv", default="")
    stats_parser.add_argument("--output-context-csv", default="")

    eval_parser = subparsers.add_parser("eval", help="Run clean cache-based quality evaluation and report degradation vs native fp.")
    eval_parser.add_argument("--model", required=True)
    eval_parser.add_argument("--traces-path", required=True)
    eval_parser.add_argument("--layers", nargs="+", type=int, default=[3])
    eval_parser.add_argument("--modes", nargs="+", default=["sunshape_base", "sunshape_pro"])
    eval_parser.add_argument("--eval-domain", default="wikitext")
    eval_parser.add_argument("--max-eval-texts", type=int, default=16)
    eval_parser.add_argument("--ctx-len", type=int, default=512)
    eval_parser.add_argument("--device-map", default="none")
    eval_parser.add_argument("--torch-dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    eval_parser.add_argument("--trust-remote-code", action="store_true")
    eval_parser.add_argument("--local-files-only", action="store_true")
    eval_parser.add_argument("--output-csv", default="")
    eval_parser.add_argument("--output-json", default="")
    _add_common_sunshape_args(eval_parser, include_layers=False)

    bundle_parser = subparsers.add_parser("bundle-info", help="Inspect a saved SunShape bundle.")
    bundle_parser.add_argument("--bundle-path", required=True)
    bundle_parser.add_argument("--output-json", default="")

    pareto_parser = subparsers.add_parser("pareto", help="Run a resumable SunShape vs counterpart Pareto sweep.")
    _add_pareto_arguments(pareto_parser)

    serve_parser = subparsers.add_parser("serve", help="Serve a SunShape-wrapped HF model over a small HTTP API.")
    serve_parser.add_argument("--model", required=True)
    serve_parser.add_argument("--bundle-path", default="")
    serve_parser.add_argument("--traces-path", default="")
    serve_parser.add_argument("--output-traces-path", default="")
    serve_parser.add_argument("--output-bundle-path", default="")
    serve_parser.add_argument("--device-map", default="none")
    serve_parser.add_argument("--torch-dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    serve_parser.add_argument("--trust-remote-code", action="store_true")
    serve_parser.add_argument("--local-files-only", action="store_true")
    serve_parser.add_argument("--num-samples", type=int, default=16)
    serve_parser.add_argument("--seq-len", type=int, default=512)
    serve_parser.add_argument("--dataset-name", default="wikitext")
    serve_parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    serve_parser.add_argument("--split", default="validation")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--request-timeout-s", type=float, default=30.0)
    serve_parser.add_argument("--max-request-bytes", type=int, default=16_384)
    serve_parser.add_argument("--max-prompt-chars", type=int, default=4_096)
    serve_parser.add_argument("--max-new-tokens", type=int, default=64)
    serve_parser.add_argument("--max-inflight-requests", type=int, default=1)
    _add_common_sunshape_args(serve_parser)

    return parser


def cmd_fit(args: argparse.Namespace) -> int:
    trace_config = TraceConfig(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
    )
    model, tokenizer, bundle = SunShapeForCausalLM.from_pretrained(
        args.model,
        traces_path=args.traces_path or None,
        output_traces_path=args.output_traces_path or None,
        output_bundle_path=args.bundle_path,
        sunshape_config=_sunshape_config_from_args(args),
        trace_config=trace_config,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    print(json.dumps({
        "model": args.model,
        "bundle_path": str(Path(args.bundle_path)),
        "layers": bundle.layers,
        "mode": bundle.mode,
        "bits_per_dim": bundle.bits_per_dim,
        "block_dim": bundle.block_dim,
        "trace_meta": bundle.trace_meta,
    }, indent=2))
    _ = model, tokenizer
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    if bool(args.model) == bool(args.traces_path):
        raise SystemExit("Pass exactly one of --model or --traces-path.")
    if args.traces_path:
        traces, meta = load_trace_artifact(args.traces_path)
        report = diagnose_from_traces(
            traces,
            model_name=str(meta.get("model_name", "unknown")),
            block_dim=args.block_dim,
            max_queries=args.max_queries,
        )
    else:
        report = diagnose_model(args.model, block_dim=args.block_dim, max_queries=args.max_queries)
    print(report)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nSaved JSON report to {output_path}")
    return 0


def cmd_export_vllm(args: argparse.Namespace) -> int:
    trace_config = TraceConfig(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
    )
    bundle, _, _ = fit_sunshape_bundle(
        args.model,
        traces_path=args.traces_path or None,
        output_traces_path=args.output_traces_path or None,
        sunshape_config=_sunshape_config_from_args(args),
        trace_config=trace_config,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    bundle.save(args.bundle_path)
    config = _LocalVLLMConfig(
        model_name=bundle.model_name,
        bundle_path=str(Path(args.bundle_path)),
        mode=bundle.mode,
        bits_per_dim=bundle.bits_per_dim,
        block_dim=bundle.block_dim,
        layers=list(bundle.layers),
    )
    payload = config.to_dict()
    print(json.dumps(payload, indent=2))
    if args.output_json:
        output_path = config.save_json(args.output_json)
        print(f"\nSaved vLLM config to {output_path}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    trace_meta = load_trace_meta(args.traces_path)
    model_name = args.model or str(trace_meta.get("model_name", "unknown"))
    stats = build_compression_stats(
        model_name=model_name,
        trace_meta=trace_meta,
        bits_per_dim_list=list(args.bits_per_dim),
        contexts=list(args.contexts),
        key_baseline_bits=args.key_baseline_bits,
        value_bits=args.value_bits,
        centroid_bits=args.centroid_bits,
    )
    print("Per-head token storage:")
    print(stats.per_head.to_string(index=False))
    print("\nContext-level storage per head:")
    print(stats.per_context.to_string(index=False))
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        stats.per_head.to_csv(out, index=False)
        print(f"\nSaved per-head stats to {out}")
    if args.output_context_csv:
        out = Path(args.output_context_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        stats.per_context.to_csv(out, index=False)
        print(f"Saved context stats to {out}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    block_dim = int(args.block_dim) if int(args.block_dim) > 0 else int(default_block_dim(float(args.bits_per_dim)))
    df = run_cache_eval(
        model_name=args.model,
        traces_path=args.traces_path,
        layers=list(args.layers),
        modes=list(args.modes),
        eval_domain=args.eval_domain,
        max_eval_texts=args.max_eval_texts,
        ctx_len=args.ctx_len,
        block_dim=block_dim,
        bits_per_dim=float(args.bits_per_dim),
        seed=int(args.seed),
        dsq_steps=int(args.dsq_steps),
        cal_points=int(args.cal_points),
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    print(df.to_string(index=False))
    save_eval_outputs(df, output_csv=args.output_csv, output_json=args.output_json)
    if args.output_csv:
        print(f"\nSaved CSV to {Path(args.output_csv)}")
    if args.output_json:
        print(f"Saved JSON to {Path(args.output_json)}")
    return 0


def cmd_bundle_info(args: argparse.Namespace) -> int:
    bundle = SunShapeBundle.load(args.bundle_path)
    payload = {
        "model_name": bundle.model_name,
        "layers": list(bundle.layers),
        "mode": bundle.mode,
        "bits_per_dim": bundle.bits_per_dim,
        "block_dim": bundle.block_dim,
        "cal_points": bundle.cal_points,
        "dsq_steps": bundle.dsq_steps,
        "seed": bundle.seed,
        "trace_meta": dict(bundle.trace_meta),
    }
    print(json.dumps(payload, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved bundle metadata to {output_path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    trace_config = TraceConfig(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
    )
    runtime = build_runtime(
        model_name=args.model,
        bundle_path=args.bundle_path or None,
        traces_path=args.traces_path or None,
        output_traces_path=args.output_traces_path or None,
        output_bundle_path=args.output_bundle_path or None,
        sunshape_config=_sunshape_config_from_args(args),
        trace_config=trace_config,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    serve_runtime(
        runtime,
        host=args.host,
        port=int(args.port),
        config=SunShapeServeConfig(
            request_timeout_s=float(args.request_timeout_s),
            max_request_bytes=int(args.max_request_bytes),
            max_prompt_chars=int(args.max_prompt_chars),
            max_new_tokens=int(args.max_new_tokens),
            max_inflight_requests=int(args.max_inflight_requests),
        ),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fit":
        return cmd_fit(args)
    if args.command == "diagnose":
        return cmd_diagnose(args)
    if args.command == "export-vllm":
        return cmd_export_vllm(args)
    if args.command == "stats":
        return cmd_stats(args)
    if args.command == "eval":
        return cmd_eval(args)
    if args.command == "bundle-info":
        return cmd_bundle_info(args)
    if args.command == "pareto":
        _, run_pareto_compare = _load_pareto_driver()
        run_pareto_compare(args)
        return 0
    if args.command == "serve":
        return cmd_serve(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
