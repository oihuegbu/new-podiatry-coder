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
    "pfs_indicators": config.GLOBAL_PERIODS_FILE,
    "coverage_policy": config.LCD_FILE,
    "validator_rules": config.DATA_DIR / "rules" / "validator_rules.json",
    # Governed terminology is not code authority, but it is a release-bearing
    # interpretation source and must be bound into the same immutable manifest.
    "terminology_registry": config.TERMINOLOGY_REGISTRY_FILE,
    "terminology_source_catalog": config.TERMINOLOGY_SOURCE_CATALOG_FILE,
    "retrieval_lexicon_catalog": config.RETRIEVAL_LEXICON_CATALOG_FILE,
    "source_requirements": config.SOURCE_REQUIREMENTS_FILE,
    "mcd_coverage_cache": config.MCD_COVERAGE_CACHE_FILE,
}


def _authoritative_paths() -> dict[str, Path]:
    paths = dict(_AUTHORITATIVE)
    # The release certificate must bind the registry actually consulted by
    # scope evaluation, including deployments that keep signed scopes outside
    # the repository. Binding the repository default here would create a
    # split-brain authorization proof.
    paths["autonomous_scope_registry"] = Path(os.getenv(
        "AUTONOMOUS_SCOPE_REGISTRY",
        str(config.DATA_DIR / "release" / "autonomous_scopes.json")))
    for path in sorted(config.CODES_DIR.glob("*.json")):
        paths.setdefault(f"codes/{path.name}", path)
    for path in sorted((config.DATA_DIR / "rules").glob("*.json")):
        paths.setdefault(f"rules/{path.name}", path)
    for path in sorted(config.TERMINOLOGY_PACK_DIR.glob("*.json")):
        paths.setdefault(f"terminology_pack/{path.name}", path)
    try:
        catalog = json.loads(config.TERMINOLOGY_SOURCE_CATALOG_FILE.read_text())
        for source in catalog.get("sources") or []:
            source_id = str(source.get("id") or "unknown")
            for index, relative in enumerate(source.get("paths") or [], 1):
                path = (config.BASE_DIR / str(relative)).resolve()
                if not path.is_relative_to(config.BASE_DIR.resolve()):
                    raise ValueError(
                        f"terminology source escapes repository: {relative}")
                paths.setdefault(
                    f"terminology_input/{source_id}/{index}:{path.name}", path)
    except Exception:
        # Preserve build_source_manifest's structured fail-closed behavior:
        # an invalid catalog becomes a missing sentinel record and therefore
        # a manifest error instead of escaping as an uncaught exception.
        paths["terminology_catalog_inputs_invalid"] = (
            config.BASE_DIR / ".invalid-terminology-source-catalog")
    try:
        catalog = json.loads(config.RETRIEVAL_LEXICON_CATALOG_FILE.read_text())
        for pack in catalog.get("packs") or []:
            pack_id = str(pack.get("id") or "unknown")
            roles = ["path", "code_source"]
            if pack.get("candidate_source"):
                roles.append("candidate_source")
            for role in roles:
                relative = str(pack.get(role) or "")
                path = (config.BASE_DIR / relative).resolve()
                if not relative or not path.is_relative_to(
                        config.BASE_DIR.resolve()):
                    raise ValueError(
                        f"retrieval lexicon path escapes repository: {relative}")
                paths.setdefault(
                    f"retrieval_lexicon/{pack_id}/{role}:{path.name}", path)
    except Exception:
        paths["retrieval_lexicon_catalog_inputs_invalid"] = (
            config.BASE_DIR / ".invalid-retrieval-lexicon-catalog")
    runtime = {
        "compliance_database": config.DATA_DIR / "compliance.db",
        "validator_implementation": config.BASE_DIR / "app" / "validation" /
                                    "validator.py",
        "consistency_implementation": config.BASE_DIR / "app" / "validation" /
                                      "consistency.py",
        "scrubber_implementation": config.BASE_DIR / "app" / "compliance" /
                                   "engine.py",
        "compliance_datastore_implementation": config.BASE_DIR / "app" /
                                               "compliance" / "datastore" /
                                               "store.py",
        "payer_registry_implementation": config.BASE_DIR / "app" /
                                         "compliance" / "payer_registry.py",
        "release_gate_implementation": config.BASE_DIR / "app" / "release" /
                                       "claim_readiness.py",
        "mutation_ledger_implementation": config.BASE_DIR / "app" / "release" /
                                          "mutation_ledger.py",
        "scope_bootstrap_implementation": config.BASE_DIR / "app" / "release" /
                                          "scope_bootstrap.py",
        "scope_authorization_implementation": config.BASE_DIR / "app" /
                                              "release" / "scope_registry.py",
        "identifier_validation_implementation": config.BASE_DIR / "app" /
                                                "core" / "identifiers.py",
        "model_execution_implementation": config.BASE_DIR / "app" / "core" /
                                          "model_profiles.py",
        "terminology_implementation": config.BASE_DIR / "app" /
                                      "terminology" / "normalizer.py",
        "terminology_builder_implementation": config.BASE_DIR / "tools" /
                                              "build_terminology_pack.py",
        "retrieval_lexicon_implementation": config.BASE_DIR / "app" / "rag" /
                                            "retrieval_lexicon.py",
        "retrieval_lexicon_importer": config.BASE_DIR / "tools" /
                                      "import_retrieval_lexicon.py",
        "retrieval_lexicon_corroborator": config.BASE_DIR / "tools" /
                                          "corroborate_retrieval_lexicon.py",
        "retrieval_index_implementation": config.BASE_DIR / "app" / "rag" /
                                          "vector_store.py",
        "candidate_retriever_implementation": config.BASE_DIR / "app" / "rag" /
                                              "retriever.py",
        "result_cache_implementation": config.BASE_DIR / "app" / "core" /
                                       "cache.py",
        "source_preflight_implementation": config.BASE_DIR / "app" /
                                           "compliance" / "refresh" /
                                           "preflight.py",
        "clinical_facts_implementation": config.BASE_DIR / "app" /
                                         "clinical_facts" / "builder.py",
        "clinical_audit_implementation": config.BASE_DIR / "tools" /
                                         "clinical_auditor.py",
        "record_coherence_implementation": config.BASE_DIR / "tools" /
                                           "record_coherence.py",
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
            "PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        if not busy and frames == checkpointed and frames > 0:
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
        "global_periods": [config.GLOBAL_PERIODS_FILE],
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
    """Return cached metadata keyed by every file that can affect it."""
    source_paths = {
        "icd10_codes": (config.ICD10_FILE,),
        "cpt_codes": (config.CPT_FILE,),
        "hcpcs_codes": (config.HCPCS_FILE,),
        "ncci_edits": (config.NCCI_FILE,),
        "mue_limits": (config.MUE_FILE,),
        "pfs_indicators": (config.GLOBAL_PERIODS_FILE,),
        "coverage_policy": (config.LCD_FILE,),
        "mcd_coverage_cache": (config.MCD_COVERAGE_CACHE_FILE,),
    }
    paths = source_paths.get(source_id, ())
    if source_id != "mcd_coverage_cache":
        paths = (*paths, config.DATA_DIR / "compliance.db")
    identity = []
    for path in paths:
        try:
            stat = path.stat()
            identity.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            identity.append((str(path), -1, -1))
    return dict(_release_metadata_cached(source_id, tuple(identity)))


@lru_cache(maxsize=64)
def _release_metadata_cached(source_id: str, _identity: tuple) -> dict:
    """Effective/version metadata from the database built from the source.

    Whole-file checksums identify the exact bytes. These fields explain which
    date window/version those bytes represent without loading very large JSON
    sources into memory a second time. ``_identity`` is deliberately unused
    in the body; it invalidates this cache whenever a backing source changes.
    """
    metadata_sources = {"ncci_edits", "mue_limits", "pfs_indicators",
                        "coverage_policy",
                        "mcd_coverage_cache",
                        "icd10_codes", "cpt_codes", "hcpcs_codes"}
    if source_id not in metadata_sources:
        return {"effective_from": "", "effective_to": "", "version": ""}
    if source_id == "mcd_coverage_cache":
        try:
            data = json.loads(config.MCD_COVERAGE_CACHE_FILE.read_text())
            return {"effective_from": str(data.get("effective_from") or ""),
                    "effective_to": "",
                    "version": str(data.get("fetched_at") or ""),
                    "fetched_at": str(data.get("fetched_at") or "")}
        except (OSError, ValueError, TypeError):
            return {"effective_from": "", "effective_to": "", "version": "",
                    "fetched_at": ""}
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
        elif source_id == "pfs_indicators":
            row = conn.execute(
                "SELECT MIN(effective_from), MAX(effective_to) "
                "FROM global_period WHERE effective_from<>'1900-01-01'"
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
                "pfs_indicators": "seed:global_periods",
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
        metadata["release_windows"] = _database_release_windows(
            source_id, conn)
        metadata.update(_edition_release_window(source_id, row))
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


def _database_release_windows(source_id: str, conn) -> list[dict]:
    refresh_id = {"ncci_edits": "ncci_ptp", "mue_limits": "mue",
                  "pfs_indicators": "pfs_global"}.get(source_id)
    if not refresh_id:
        return []
    releases = []
    try:
        rows = conn.execute(
            "SELECT DISTINCT effective_from FROM data_source_version "
            "WHERE source_id=? AND effective_from<>''",
            (refresh_id,)).fetchall()
        releases.extend(str(row[0]) for row in rows)
    except sqlite3.Error:
        pass
    if source_id == "pfs_indicators":
        published = _pfs_source_release_date()
    else:
        published = _parse_source_release_date(source_id, "")
    if source_id == "ncci_edits" and not published:
        # Some retained CMS NCCI filenames expose only an internal release
        # revision (for example ``v322``), not a calendar quarter. The newest
        # edit-effective date in that complete snapshot is the conservative
        # lower bound of the release it can prove; refresh provenance supplies
        # explicit quarters for every later downloaded snapshot.
        row = conn.execute(
            "SELECT MAX(effective_from) FROM ncci_ptp").fetchone()
        try:
            published = date.fromisoformat(str((row or [""])[0] or ""))
        except ValueError:
            published = None
    if published:
        releases.append(published.isoformat())
    windows = set()
    for value in releases:
        try:
            release = date.fromisoformat(value)
        except ValueError:
            continue
        windows.add(_quarter_window(release.year, release.month))
    return [{"effective_from": start, "effective_to": end}
            for start, end in sorted(windows)]


def _head_text(path: Path, limit: int = 256 * 1024) -> str:
    with path.open("rb") as handle:
        return handle.read(limit).decode("utf-8", errors="replace")


def _pfs_source_release_date() -> date | None:
    try:
        head = _head_text(config.GLOBAL_PERIODS_FILE)
    except OSError:
        return None
    match = re.search(
        r"(20\d{2})[_\s-]*(Jan(?:uary)?|Apr(?:il)?|Jul(?:y)?|Oct(?:ober)?)",
        head, re.IGNORECASE)
    if match:
        year, month_name = match.groups()
        month = {"jan": 1, "apr": 4, "jul": 7, "oct": 10}[
            month_name[:3].lower()]
        return date(int(year), month, 1)
    match = re.search(r"\bRVU(\d{2})([A-D])\b", head, re.IGNORECASE)
    if match:
        year, letter = match.groups()
        month = (ord(letter.upper()) - ord("A")) * 3 + 1
        return date(2000 + int(year), month, 1)
    return None


def _edition_release_window(source_id: str, db_window=None) -> dict:
    """Claim-date coverage of the exact licensed/published snapshot.

    Code lifecycle dates and release freshness are separate concepts.  A
    decades-old HCPCS code may remain active, while an April quarterly file
    is still insufficient authority for an August claim.  The manifest
    records both so the release certificate can enforce the latter.
    """
    try:
        if source_id == "cpt_codes":
            # CPT rows are bounded to their licensed edition by ingestion;
            # reading the DB bound avoids reparsing the multi-hundred-MB JSON
            # on every manifest verification.
            year = int(str((db_window or ("", ""))[0])[:4])
            return {"release_effective_from": date(year, 1, 1).isoformat(),
                    "release_effective_to": date(year, 12, 31).isoformat(),
                    "release_basis": "licensed CPT edition"}
        if source_id == "icd10_codes":
            start = date.fromisoformat(str((db_window or ("", ""))[0]))
            return {"release_effective_from": start.isoformat(),
                    "release_effective_to": date(
                        start.year + 1, 9, 30).isoformat(),
                    "release_basis": "ICD-10-CM fiscal-year edition"}
        if source_id == "hcpcs_codes":
            head = _head_text(config.HCPCS_FILE)
            match_name = re.search(
                r'"source_file"\s*:\s*"([^"]+)"', head, re.IGNORECASE)
            source_name = match_name.group(1) if match_name else ""
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
        if source_id in {"ncci_edits", "mue_limits", "pfs_indicators"}:
            table = {"ncci_edits": "ncci_ptp", "mue_limits": "mue",
                     "pfs_indicators": "global_period"}[source_id]
            conn = sqlite3.connect(
                f"file:{config.DATA_DIR / 'compliance.db'}?mode=ro", uri=True)
            try:
                row = conn.execute(
                    f"SELECT MAX(effective_from) FROM {table}").fetchone()
            finally:
                conn.close()
            fallback = str((row or [""])[0] or "")
            release = (_parse_source_release_date(source_id, fallback)
                       if source_id != "pfs_indicators" else
                       date.fromisoformat(fallback))
            if not release:
                return {}
            start, end = _quarter_window(release.year, release.month)
            return {"release_effective_from": start,
                    "release_effective_to": end,
                    "release_basis": (
                        "CMS quarterly PFS release"
                        if source_id == "pfs_indicators" else
                        "CMS quarterly edit release")}
    except (OSError, ValueError, TypeError, StopIteration, IndexError):
        return {}
    return {}


def _parse_source_release_date(source_id: str, fallback: str) -> date | None:
    """Resolve the published release date, correcting row-level revision dates.

    The bundled MUE rows carry 2026-03-31 while their source filename says
    ``Eff_04-01-2026``. The filename is the release authority; the row date is
    a revision marker and must not shift the quarterly applicability window.
    """
    path = config.MUE_FILE if source_id == "mue_limits" else config.NCCI_FILE
    published = None
    try:
        head = _head_text(path)
        match = re.search(
            r"(?:Eff[_-]?)?(\d{2})[-_](\d{2})[-_](20\d{2})", head,
            re.IGNORECASE)
        if match:
            month, day, year = map(int, match.groups())
            published = date(year, month, day)
        else:
            match = re.search(r"(20\d{2})q([1-4])", head, re.IGNORECASE)
            if match:
                year, quarter = map(int, match.groups())
                published = date(year, (quarter - 1) * 3 + 1, 1)
    except (OSError, ValueError, TypeError, IndexError):
        pass
    try:
        database_release = date.fromisoformat(fallback)
    except ValueError:
        database_release = None
    # The pinned seed filename corrects transformed seed rows that use the
    # prior day, while a later additive refresh recorded in compliance.db
    # must supersede the seed. Choosing the newer proven date handles both.
    candidates = [value for value in (published, database_release) if value]
    return max(candidates) if candidates else None
