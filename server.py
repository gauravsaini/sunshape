from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from sunshape.hf import SunShapeConfig, SunShapeForCausalLM, TraceConfig


@dataclass
class SunShapeRuntime:
    model: Any
    tokenizer: Any
    bundle: Any
    model_name: str

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool | None = None,
    ) -> dict[str, Any]:
        prompt = str(prompt)
        if do_sample is None:
            do_sample = temperature > 0.0
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(max(temperature, 1e-5)) if do_sample else None,
            do_sample=bool(do_sample),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
        return {
            "prompt": prompt,
            "text": full_text,
            "generated_text": generated_text,
            "model_name": self.model_name,
            "layers": list(self.bundle.layers),
            "mode": self.bundle.mode,
            "bits_per_dim": self.bundle.bits_per_dim,
            "block_dim": self.bundle.block_dim,
        }


def build_runtime(
    *,
    model_name: str,
    bundle_path: str | Path | None = None,
    traces_path: str | Path | None = None,
    output_traces_path: str | Path | None = None,
    output_bundle_path: str | Path | None = None,
    sunshape_config: SunShapeConfig | None = None,
    trace_config: TraceConfig | None = None,
    device_map: str = "none",
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> SunShapeRuntime:
    model, tokenizer, bundle = SunShapeForCausalLM.from_pretrained(
        model_name,
        traces_path=traces_path,
        bundle_path=bundle_path,
        output_traces_path=output_traces_path,
        output_bundle_path=output_bundle_path,
        sunshape_config=sunshape_config,
        trace_config=trace_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    return SunShapeRuntime(model=model, tokenizer=tokenizer, bundle=bundle, model_name=model_name)


def serve_runtime(runtime: SunShapeRuntime, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SunShapeHTTP/0.1"

        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._write_json(
                    200,
                    {
                        "ok": True,
                        "model_name": runtime.model_name,
                        "layers": list(runtime.bundle.layers),
                        "mode": runtime.bundle.mode,
                        "bits_per_dim": runtime.bundle.bits_per_dim,
                        "block_dim": runtime.bundle.block_dim,
                    },
                )
                return
            self._write_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/generate":
                self._write_json(404, {"ok": False, "error": "not_found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                self._write_json(400, {"ok": False, "error": "invalid_json"})
                return

            prompt = str(payload.get("prompt", ""))
            if not prompt:
                self._write_json(400, {"ok": False, "error": "prompt_required"})
                return

            try:
                result = runtime.generate(
                    prompt,
                    max_new_tokens=int(payload.get("max_new_tokens", 64)),
                    temperature=float(payload.get("temperature", 0.0)),
                    do_sample=payload.get("do_sample"),
                )
            except Exception as exc:
                self._write_json(500, {"ok": False, "error": str(exc)})
                return
            self._write_json(200, {"ok": True, **result})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    httpd = ThreadingHTTPServer((host, int(port)), Handler)
    print(
        json.dumps(
            {
                "status": "serving",
                "host": host,
                "port": int(port),
                "health_url": f"http://{host}:{int(port)}/health",
                "generate_url": f"http://{host}:{int(port)}/generate",
                "model_name": runtime.model_name,
                "layers": list(runtime.bundle.layers),
                "mode": runtime.bundle.mode,
                "bits_per_dim": runtime.bundle.bits_per_dim,
            }
        )
    )
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
