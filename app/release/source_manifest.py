"""Checksummed provenance for authoritative coding inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import calendar
from datetime import date
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
    # Governed terminology is not code authority, but it is a release-bearing
    # interpretation source and must be bound into the same immutable manifest.
    "terminology_registry": config.TERMINOLOGY_REGISTRY_FILE,
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
        "terminology_implementation": config.BASE_DIR / "app" /
                                      "terminology" / "normalizer.py",
        "submission_configuration": Path(os.getenv(
            "PRACTICE_CONFIG_PATH",
            str(config.DATA_DIR / "practice_config.json"))),
    }
    paths.update(runtime)
    return paths


def sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=256)
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
    errors = _checkpoint_database()
    for source_id, path in _authoritative_paths().items():
        try:
            stat = path.stat()
            try:
                display_path = str(path.relative_to(config.BASE_DIR))
            except ValueError:
                display_path = f"external/{path.name}"
            records.append({"source_id": source_id,
                            "path": display_path,
                            "sha256": sha256_file(path),
                            "size": stat.st_size,
                            **_release_metadata(source_id)})
        except Exception as exc:
            errors.append(f"{source_id}: {exc}")
    errors.extend(_database_source_errors())
    body = {"records": records, "errors": errors}
    body["fingerprint"] = manifest_fingerprint(body)
    return body


def _checkpoint_database() -> list[str]:
    """Canonicalize committed WAL frames before hashing compliance.db.

    This is a storage checkpoint, not a data change.  If another writer owns
    the WAL, the manifest fails closed instead of hashing a stale main file.
    """
    db_path = config.DATA_DIR / "compliance.db"
    if not db_path.exists():
        return ["compliance_database: absent"]
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        busy, frames, checkpointed = conn.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        conn.close()
    except sqlite3.Error as exc:
        return [f"compliance_database checkpoint failed: {exc}"]
    if busy or frames != checkpointed:
        return ["compliance_database checkpoint incomplete; concurrent writer active"]
    return []


def _database_source_errors() -> list[str]:
    """Detect a database built from different source-file identities."""
    db_path = config.DATA_DIR / "compliance.db"
    if not db_path.exists():
        return ["compliance_database: absent"]
    sources = {
        "icd10_codes": [config.ICD10_FILE],
        "cpt_codes": [config.CPT_FILE],
        "hcpcs_codes": [config.HCPCS_FILE],
        "ncci": [config.NCCI_FILE],
        "mue": [config.MUE_FILE],
        "lcd": [config.LCD_FILE, config.MCD_COVERAGE_CACHE_FILE],
    }

    def identity(paths: list[Path]) -> str:
        parts = []
        for path in paths:
            try:
                stat = path.stat()
                parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
            except OSError:
                parts.append(f"{path.name}:missing")
        return "|".join(parts)

    try:
        conn = sqlite3.connect(db_path, timeout=30)
        rows = conn.execute(
            "SELECT source_id, fingerprint FROM data_file_fingerprint"
        ).fetchall()
        busy, frames, checkpointed = conn.execute(
            "PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        conn.close()
    except sqlite3.Error as exc:
        return [f"compliance_database provenance unavailable: {exc}"]
    recorded = dict(rows)
    errors = [
        f"compliance_database source mismatch: {source_id}"
        for source_id, paths in sources.items()
        if recorded.get(source_id) != identity(paths)
    ]
    if busy or frames != checkpointed or frames:
        errors.append("compliance_database changed while manifest was built")
    return errors


def manifest_fingerprint(manifest: dict) -> str:
    body = {"records": manifest.get("records") or [],
            "errors": manifest.get("errors") or []}
    return "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def valid_record(record: dict) -> bool:
    """Validate the minimum identity carried by every source record."""
    return bool(record.get("source_id") and record.get("path") and
                isinstance(record.get("size"), int) and record["size"] >= 0 and
                _SHA256_RE.fullmatch(str(record.get("sha256") or "")))


def _release_metadata(source_id: str) -> dict:
    """Effective/version metadata from the database built from the source.

    Whole-file checksums identify the exact bytes. These fields explain which
    date window/version those bytes represent without loading very large JSON
    sources into memory a second time.
    """
    metadata_sources = {"ncci_edits", "mue_limits", "coverage_policy",
                        "icd10_codes", "cpt_codes", "hcpcs_codes"}
    if source_id not in metadata_sources:
        return {"effective_from": "", "effective_to": "", "version": ""}
    db_path = config.DATA_DIR / "compliance.db"
    if not db_path.exists():
        return {"effective_from": "", "effective_to": "", "version": ""}
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        row = None
        version_row = None
        if source_id == "ncci_edits":
            row = conn.execute(
                "SELECT MIN(effective_from), MAX(effective_to) FROM ncci_ptp"
            ).fetchone()
        elif source_id == "mue_limits":
            row = conn.execute(
                "SELECT MIN(effective_from), MAX(effective_to) FROM mue"
            ).fetchone()
        elif source_id in {"icd10_codes", "cpt_codes", "hcpcs_codes"}:
            system = {"icd10_codes": "ICD10", "cpt_codes": "CPT",
                      "hcpcs_codes": "HCPCS"}[source_id]
            row = conn.execute(
                "SELECT MIN(effective_from), MAX(effective_to) FROM code_set "
                "WHERE code_system=?", (system,)).fetchone()
        else:
            row = None
        try:
            seed_id = {
                "ncci_edits": "seed:ncci", "mue_limits": "seed:mue",
                "coverage_policy": "seed:lcd",
                "icd10_codes": "seed:icd10_codes",
                "cpt_codes": "seed:cpt_codes",
                "hcpcs_codes": "seed:hcpcs_codes",
            }.get(source_id, "")
            version_row = conn.execute(
                "SELECT effective_from, ingested_at FROM data_source_version "
                "WHERE source_id IN (?, ?, ?) ORDER BY ingested_at DESC LIMIT 1",
                (source_id, source_id.removesuffix("_edits"), seed_id),
            ).fetchone()
        except sqlite3.Error:
            # Older stores may predate the refresh-version registry.  That
            # must not discard effective bounds successfully read above.
            version_row = None
        metadata = {
            "effective_from": str((row or ("", ""))[0] or ""),
            "effective_to": str((row or ("", ""))[1] or ""),
            "version": ("/".join(str(v or "") for v in version_row)
                        if version_row else ""),
        }
        metadata.update(_edition_release_window(source_id))
        return metadata
    except (OSError, sqlite3.Error):
        return {"effective_from": "", "effective_to": "", "version": ""}
    finally:
        if conn is not None:
            conn.close()


def _quarter_window(year: int, month: int) -> tuple[str, str]:
    quarter_month = ((month - 1) // 3) * 3 + 1
    end_month = quarter_month + 2
    return (date(year, quarter_month, 1).isoformat(),
            date(year, end_month, calendar.monthrange(year, end_month)[1]).isoformat())


def _edition_release_window(source_id: str) -> dict:
    """Claim-date coverage of the exact licensed/published snapshot.

    Code lifecycle dates and release freshness are separate concepts.  A
    decades-old HCPCS code may remain active, while an April quarterly file
    is still insufficient authority for an August claim.  The manifest
    records both so the release certificate can enforce the latter.
    """
    try:
        if source_id == "cpt_codes":
            data = json.loads(config.CPT_FILE.read_text())
            year = int((data.get("metadata") or {}).get("year"))
            return {"release_effective_from": date(year, 1, 1).isoformat(),
                    "release_effective_to": date(year, 12, 31).isoformat(),
                    "release_basis": "licensed CPT edition"}
        if source_id == "icd10_codes":
            rows = json.loads(config.ICD10_FILE.read_text())
            fy = int(next(str(row.get("fy")) for row in rows if row.get("fy")))
            return {"release_effective_from": date(fy - 1, 10, 1).isoformat(),
                    "release_effective_to": date(fy, 9, 30).isoformat(),
                    "release_basis": "ICD-10-CM fiscal-year edition"}
        if source_id == "hcpcs_codes":
            rows = json.loads(config.HCPCS_FILE.read_text())
            source_name = str(((rows[0].get("metadata") or {}).get("source_file")))
            match = re.search(r"(20\d{2})[_-]?(JAN|APR|JUL|OCT)", source_name,
                              re.IGNORECASE)
            if not match:
                match = re.search(r"(JAN|APR|JUL|OCT)[_-]?(20\d{2})", source_name,
                                  re.IGNORECASE)
                if match:
                    month_name, year_text = match.groups()
                else:
                    return {}
            else:
                year_text, month_name = match.groups()
            month = {"JAN": 1, "APR": 4, "JUL": 7, "OCT": 10}[month_name.upper()]
            start, end = _quarter_window(int(year_text), month)
            return {"release_effective_from": start,
                    "release_effective_to": end,
                    "release_basis": "CMS quarterly HCPCS release"}
    except (OSError, ValueError, TypeError, StopIteration, IndexError):
        return {}
    return {}
