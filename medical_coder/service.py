from __future__ import annotations

import base64
import binascii
import hmac
import json
import os
import tempfile
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .cli import json_default
from .orchestrator import CodingWorkflow
from .providers import ProviderError
from .serde import context_from_dict


class ServiceConfigurationError(RuntimeError):
    pass


def build_workflow() -> CodingWorkflow:
    required = {
        "snapshot": os.environ.get("CODER_SNAPSHOT", ""),
        "service_token": os.environ.get("CODER_SERVICE_TOKEN", ""),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ServiceConfigurationError("missing service configuration: " + ", ".join(missing))
    return CodingWorkflow(
        Path(required["snapshot"]),
        Path(os.environ.get("CODER_PROVIDER_CONFIG", "config/providers.json")),
        Path(os.environ.get("CODER_ROLES_CONFIG", "config/system_roles.json")),
        Path(os.environ.get("CODER_SCOPE_CONFIG", "source-packs/scopes/medium-private-surgical-practice.json")),
        Path(os.environ.get("CODER_AUDIT_DATABASE", "var/audit.sqlite")),
    )


class CodingHandler(BaseHTTPRequestHandler):
    workflow: CodingWorkflow
    token: str
    max_request_bytes: int

    def log_message(self, format: str, *args: Any) -> None:
        # Access logs intentionally exclude request bodies and patient identifiers.
        super().log_message(format, *args)

    def _json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, sort_keys=True, default=json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        supplied = self.headers.get("authorization", "")
        return supplied.startswith("Bearer ") and hmac.compare_digest(supplied[7:], self.token)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._json(200, {"status": "ok"})
            return
        if self.path == "/readyz":
            self._json(200, {"status": "ready", "snapshot_id": self.workflow.snapshot.snapshot_id})
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/v1/code":
            self._json(404, {"error": "not_found"})
            return
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
        except ValueError:
            self._json(400, {"error": "invalid_content_length"})
            return
        if length <= 0 or length > self.max_request_bytes:
            self._json(413, {"error": "request_size_out_of_bounds"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            encounter_id = str(payload["encounter_id"]).strip()
            if not encounter_id or len(encounter_id) > 128:
                raise ValueError("invalid encounter identity")
            document = base64.b64decode(payload["document_base64"], validate=True)
            if not document:
                raise ValueError("empty document")
            suffix = Path(str(payload.get("filename", "note.pdf"))).suffix.casefold()
            if suffix not in {".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                raise ValueError("unsupported filename extension")
            context = context_from_dict(payload["context"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, binascii.Error):
            self._json(400, {"error": "invalid_request"})
            return
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(prefix="coder-note-", suffix=suffix)
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(document)
                handle.flush()
                os.fsync(handle.fileno())
            result = self.workflow.code(encounter_id, temporary_path, context)
        except Exception as error:
            try:
                self.workflow.audit.record_failure(encounter_id, error)
            except Exception as audit_error:
                self._json(500, {"error": "audit_failure", "type": type(audit_error).__name__})
                return
            status = 503 if isinstance(error, (ProviderError, TimeoutError)) else 500
            self._json(status, {"error": "coding_workflow_failed", "type": type(error).__name__})
        else:
            self._json(200, asdict(result))
        finally:
            if temporary_path:
                temporary_path.unlink(missing_ok=True)


def main() -> None:
    workflow = build_workflow()
    handler = type(
        "ConfiguredCodingHandler",
        (CodingHandler,),
        {
            "workflow": workflow,
            "token": os.environ["CODER_SERVICE_TOKEN"],
            "max_request_bytes": int(os.environ.get("CODER_MAX_REQUEST_BYTES", str(32 * 1024 * 1024))),
        },
    )
    host = os.environ.get("CODER_LISTEN_HOST", "127.0.0.1")
    port = int(os.environ.get("CODER_LISTEN_PORT", "8080"))
    ThreadingHTTPServer((host, port), handler).serve_forever()


if __name__ == "__main__":
    main()
