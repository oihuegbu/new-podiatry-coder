"""Checksummed provenance for authoritative coding inputs."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from app.core import config

_AUTHORITATIVE = {
    "icd10_codes": config.ICD10_FILE,
    "cpt_codes": config.CPT_FILE,
    "hcpcs_codes": config.HCPCS_FILE,
    "ncci_edits": config.NCCI_FILE,
    "mue_limits": config.MUE_FILE,
    "coverage_policy": config.LCD_FILE,
    "validator_rules": config.DATA_DIR / "rules" / "validator_rules.json",
}


def _authoritative_paths() -> dict[str, Path]:
    paths = dict(_AUTHORITATIVE)
    for path in sorted(config.CODES_DIR.glob("*.json")):
        paths.setdefault(f"codes/{path.name}", path)
    for path in sorted((config.DATA_DIR / "rules").glob("*.json")):
        paths.setdefault(f"rules/{path.name}", path)
    runtime = {
        "compliance_database": config.DATA_DIR / "compliance.db",
        "validator_implementation": config.BASE_DIR / "app" / "validation" /
                                    "validator.py",
        "scrubber_implementation": config.BASE_DIR / "app" / "compliance" /
                                   "engine.py",
        "release_gate_implementation": config.BASE_DIR / "app" / "release" /
                                       "claim_readiness.py",
    }
    paths.update(runtime)
    return paths


def sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=32)
def _sha256_cached(path_text: str, mtime_ns: int, size: int) -> str:
    """Hash once per immutable filesystem identity; edits invalidate it."""
    path = Path(path_text)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_source_manifest() -> dict:
    records = []
    errors = []
    for source_id, path in _authoritative_paths().items():
        try:
            stat = path.stat()
            records.append({"source_id": source_id,
                            "path": str(path.relative_to(config.BASE_DIR)),
                            "sha256": sha256_file(path),
                            "size": stat.st_size})
        except Exception as exc:
            errors.append(f"{source_id}: {exc}")
    body = {"records": records, "errors": errors}
    body["fingerprint"] = manifest_fingerprint(body)
    return body


def manifest_fingerprint(manifest: dict) -> str:
    body = {"records": manifest.get("records") or [],
            "errors": manifest.get("errors") or []}
    return "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
