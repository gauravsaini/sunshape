from __future__ import annotations

import logging
import json
import os
import time
import uuid
import threading
from dataclasses import dataclass, field
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
    started_at: float = field(default_factory=time.time)

    def health_payload(self, *, request_limits: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "ok": True,
            "model_name": self.model_name,
            "layers": list(self.bundle.layers),
            "mode": self.bundle.mode,
            "bits_per_dim": self.bundle.bits_per_dim,
            "block_dim": self.bundle.block_dim,
            "pid": os.getpid(),
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "trace_meta": dict(getattr(self.bundle, "trace_meta", {})),
        }
        if request_limits is not None:
            payload["request_limits"] = dict(request_limits)
        return payload

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


@dataclass
class SunShapeServeConfig:
    request_timeout_s: float = 30.0
    max_request_bytes: int = 16_384
    max_prompt_chars: int = 4_096
    max_new_tokens: int = 64
    max_inflight_requests: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_timeout_s": float(self.request_timeout_s),
            "max_request_bytes": int(self.max_request_bytes),
            "max_prompt_chars": int(self.max_prompt_chars),
            "max_new_tokens": int(self.max_new_tokens),
            "max_inflight_requests": int(self.max_inflight_requests),
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


def serve_runtime(
    runtime: SunShapeRuntime,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    config: SunShapeServeConfig | None = None,
) -> None:
    config = config or SunShapeServeConfig()
    logger = logging.getLogger("sunshape.server")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    inflight = threading.BoundedSemaphore(max(1, int(config.max_inflight_requests)))

    class Handler(BaseHTTPRequestHandler):
        server_version = "SunShapeHTTP/0.1"
        protocol_version = "HTTP/1.1"

        def setup(self) -> None:
            super().setup()
            try:
                self.connection.settimeout(float(config.request_timeout_s))
            except Exception:
                pass

        def _request_id(self) -> str:
            request_id = self.headers.get("X-Request-ID", "").strip()
            return request_id or uuid.uuid4().hex[:12]

        def _log(self, event: str, **fields: Any) -> None:
            logger.info(
                json.dumps(
                    {
                        "event": event,
                        "path": self.path,
                        "client": self.client_address[0] if self.client_address else None,
                        **fields,
                    },
                    default=str,
                )
            )

        def _write_json(self, status: int, payload: dict[str, Any], *, request_id: str | None = None) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if request_id:
                self.send_header("X-Request-ID", request_id)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            request_id = self._request_id()
            if self.path != "/health":
                self._write_json(404, {"ok": False, "error": "not_found", "request_id": request_id}, request_id=request_id)
                self._log("request_end", request_id=request_id, status=404)
                return
            payload = runtime.health_payload(request_limits=config.to_dict())
            payload["request_id"] = request_id
            self._write_json(200, payload, request_id=request_id)
            self._log("request_end", request_id=request_id, status=200)

        def do_POST(self) -> None:  # noqa: N802
            request_id = self._request_id()
            if self.path != "/generate":
                self._write_json(404, {"ok": False, "error": "not_found", "request_id": request_id}, request_id=request_id)
                self._log("request_end", request_id=request_id, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                self._write_json(400, {"ok": False, "error": "invalid_content_length", "request_id": request_id}, request_id=request_id)
                self._log("request_end", request_id=request_id, status=400)
                return
            if length > int(config.max_request_bytes):
                self._write_json(413, {"ok": False, "error": "request_too_large", "request_id": request_id}, request_id=request_id)
                self._log("request_end", request_id=request_id, status=413, content_length=length)
                return

            acquired = inflight.acquire(blocking=False)
            if not acquired:
                self._write_json(503, {"ok": False, "error": "server_busy", "request_id": request_id}, request_id=request_id)
                self._log("request_end", request_id=request_id, status=503)
                return

            start = time.monotonic()
            self._log("request_start", request_id=request_id, content_length=length)
            try:
                raw = self.rfile.read(length) if length > 0 else b"{}"
                if len(raw) > int(config.max_request_bytes):
                    self._write_json(413, {"ok": False, "error": "request_too_large", "request_id": request_id}, request_id=request_id)
                    self._log("request_end", request_id=request_id, status=413)
                    return
                payload = json.loads(raw.decode("utf-8"))
                prompt = str(payload.get("prompt", ""))
                if not prompt:
                    self._write_json(400, {"ok": False, "error": "prompt_required", "request_id": request_id}, request_id=request_id)
                    self._log("request_end", request_id=request_id, status=400)
                    return
                if len(prompt) > int(config.max_prompt_chars):
                    self._write_json(413, {"ok": False, "error": "prompt_too_long", "request_id": request_id}, request_id=request_id)
                    self._log("request_end", request_id=request_id, status=413, prompt_chars=len(prompt))
                    return

                max_new_tokens = int(payload.get("max_new_tokens", config.max_new_tokens))
                if max_new_tokens > int(config.max_new_tokens):
                    self._write_json(
                        400,
                        {"ok": False, "error": "max_new_tokens_too_large", "request_id": request_id},
                        request_id=request_id,
                    )
                    self._log("request_end", request_id=request_id, status=400, max_new_tokens=max_new_tokens)
                    return

                try:
                    temperature = float(payload.get("temperature", 0.0))
                except (TypeError, ValueError):
                    self._write_json(
                        400,
                        {"ok": False, "error": "invalid_temperature", "request_id": request_id},
                        request_id=request_id,
                    )
                    self._log("request_end", request_id=request_id, status=400, error="invalid_temperature")
                    return

                result = runtime.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=payload.get("do_sample"),
                )
            except json.JSONDecodeError:
                self._write_json(400, {"ok": False, "error": "invalid_json", "request_id": request_id}, request_id=request_id)
                self._log("request_end", request_id=request_id, status=400)
                return
            except ValueError as exc:
                self._write_json(
                    400,
                    {"ok": False, "error": "bad_request", "detail": str(exc), "request_id": request_id},
                    request_id=request_id,
                )
                self._log("request_end", request_id=request_id, status=400, error="bad_request", elapsed_ms=round((time.monotonic() - start) * 1000, 2))
                return
            except Exception:
                self._write_json(500, {"ok": False, "error": "internal_error", "request_id": request_id}, request_id=request_id)
                self._log("request_end", request_id=request_id, status=500, elapsed_ms=round((time.monotonic() - start) * 1000, 2))
                return
            finally:
                inflight.release()

            elapsed_ms = round((time.monotonic() - start) * 1000, 2)
            self._write_json(200, {"ok": True, "request_id": request_id, **result}, request_id=request_id)
            self._log(
                "request_end",
                request_id=request_id,
                status=200,
                elapsed_ms=elapsed_ms,
                prompt_chars=len(prompt),
                max_new_tokens=max_new_tokens,
                temperature=float(payload.get("temperature", 0.0)),
            )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    httpd = ThreadingHTTPServer((host, int(port)), Handler)
    httpd.daemon_threads = True
    httpd.allow_reuse_address = True
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
                "request_limits": config.to_dict(),
            }
        )
    )
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
