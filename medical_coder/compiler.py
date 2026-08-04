from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
COMPILER_VERSION = "0.1.1"


class SourceIntegrityError(RuntimeError):
    """The configured source pack cannot be compiled without guessing."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dig(value: dict[str, Any], dotted_path: str | None) -> Any:
    if not dotted_path:
        return None
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def normalize_date(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    formats = ("%Y-%m-%d", "%Y%m%d", "%Y-%m", "%Y")
    for pattern in formats:
        try:
            parsed = datetime.strptime(text, pattern)
            if pattern == "%Y":
                return f"{parsed.year:04d}-01-01"
            if pattern == "%Y-%m":
                return f"{parsed.year:04d}-{parsed.month:02d}-01"
            return parsed.date().isoformat()
        except ValueError:
            continue
    raise SourceIntegrityError(f"unsupported effective-date value: {text!r}")


def configured_date(row: dict[str, Any], config: dict[str, Any], prefix: str) -> str | None:
    filename_field = config.get(f"{prefix}_filename_field")
    filename_pattern = config.get(f"{prefix}_filename_regex")
    if filename_field and filename_pattern:
        match = re.search(filename_pattern, str(row.get(filename_field, "")))
        if not match:
            raise SourceIntegrityError(f"configured date was not present in {filename_field}")
        parts = {name: int(value) for name, value in match.groupdict().items()}
        return datetime(parts["year"], parts["month"], parts["day"]).date().isoformat()
    return normalize_date(row.get(config.get(f"{prefix}_field", "")))


def rows_from_document(document: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = document
    container = config.get("container")
    if container:
        if not isinstance(document, dict) or container not in document:
            raise SourceIntegrityError(f"missing configured container {container!r}")
        rows = document[container]
    if not isinstance(rows, list):
        raise SourceIntegrityError("artifact source must resolve to a list")
    if not all(isinstance(row, dict) for row in rows):
        raise SourceIntegrityError("artifact rows must be objects")
    return rows


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA foreign_keys=ON;
        CREATE TABLE snapshot_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE artifacts (
            artifact_id TEXT PRIMARY KEY,
            authority TEXT NOT NULL,
            system TEXT NOT NULL,
            version TEXT NOT NULL,
            code TEXT NOT NULL,
            display TEXT NOT NULL,
            concept_id TEXT,
            effective_start TEXT,
            effective_end TEXT,
            status TEXT NOT NULL,
            billable INTEGER NOT NULL CHECK (billable IN (0, 1)),
            properties_json TEXT NOT NULL,
            requirements_json TEXT,
            source_path TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            source_locator TEXT NOT NULL,
            UNIQUE(system, version, code)
        );
        CREATE INDEX artifacts_identity ON artifacts(system, code);
        CREATE INDEX artifacts_temporal ON artifacts(system, effective_start, effective_end);
        CREATE VIRTUAL TABLE artifact_search USING fts5(
            artifact_id UNINDEXED,
            searchable_text,
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE TABLE constraints (
            constraint_id TEXT PRIMARY KEY,
            authority TEXT NOT NULL,
            kind TEXT NOT NULL,
            left_code TEXT,
            right_code TEXT,
            target_code TEXT,
            numeric_value REAL,
            effective_start TEXT,
            effective_end TEXT,
            properties_json TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            source_locator TEXT NOT NULL
        );
        CREATE INDEX constraints_pair ON constraints(kind, left_code, right_code);
        CREATE INDEX constraints_left ON constraints(kind, left_code);
        CREATE INDEX constraints_right ON constraints(kind, right_code);
        CREATE INDEX constraints_target ON constraints(kind, target_code);
        CREATE TABLE policy_documents (
            document_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            body TEXT NOT NULL
        );
        """
    )


class SourceCompiler:
    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root.resolve()

    def compile(self, pack_path: Path, output_root: Path) -> Path:
        pack_path = pack_path.resolve()
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        if pack.get("schema_version") != SCHEMA_VERSION:
            raise SourceIntegrityError("unsupported source-pack schema")
        sources = self._source_inventory(pack)
        identity = {
            "compiler_schema": SCHEMA_VERSION,
            "compiler_version": COMPILER_VERSION,
            "pack": pack,
            "sources": sources,
        }
        snapshot_id = sha256_bytes(canonical_json(identity))
        output_root = output_root.resolve()
        destination = output_root / snapshot_id
        if destination.exists():
            self._verify_existing(destination, snapshot_id)
            return destination

        output_root.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=output_root))
        try:
            database_path = temp / "terminology.sqlite"
            connection = sqlite3.connect(database_path)
            try:
                create_schema(connection)
                self._compile_artifacts(connection, pack, sources)
                self._compile_constraints(connection, pack, sources)
                self._compile_policies(connection, pack, sources)
                connection.execute(
                    "INSERT INTO snapshot_metadata(key, value) VALUES (?, ?)",
                    ("snapshot_id", snapshot_id),
                )
                connection.commit()
                integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
                if integrity != "ok":
                    raise SourceIntegrityError(f"sqlite integrity check failed: {integrity}")
            finally:
                connection.close()

            manifest = {
                "snapshot_id": snapshot_id,
                "pack_id": pack["pack_id"],
                "schema_version": SCHEMA_VERSION,
                "compiler_version": COMPILER_VERSION,
                "compiled_at": datetime.now(timezone.utc).isoformat(),
                "capabilities": pack.get("capabilities", {}),
                "sources": sources,
                "database_sha256": sha256_file(database_path),
            }
            (temp / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
            os.chmod(database_path, 0o444)
            os.chmod(temp / "manifest.json", 0o444)
            os.chmod(temp, 0o555)
            os.replace(temp, destination)
            return destination
        except Exception:
            if temp.exists():
                shutil.rmtree(temp, ignore_errors=True)
            raise

    def _resolve(self, configured_path: str) -> Path:
        path = (self.repository_root / configured_path).resolve()
        if self.repository_root not in path.parents:
            raise SourceIntegrityError("source path escapes repository")
        if not path.is_file():
            raise SourceIntegrityError(f"required source missing: {configured_path}")
        return path

    def _source_inventory(self, pack: dict[str, Any]) -> list[dict[str, Any]]:
        paths: set[str] = set()
        for section in ("artifacts", "constraints", "enrichments"):
            paths.update(item["path"] for item in pack.get(section, []))
        paths.update(pack.get("policy_documents", []))
        inventory = []
        for configured_path in sorted(paths):
            path = self._resolve(configured_path)
            inventory.append(
                {
                    "path": configured_path,
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
        return inventory

    @staticmethod
    def _hash_for(sources: list[dict[str, Any]], path: str) -> str:
        for source in sources:
            if source["path"] == path:
                return str(source["sha256"])
        raise SourceIntegrityError(f"source not inventoried: {path}")

    def _compile_artifacts(
        self,
        connection: sqlite3.Connection,
        pack: dict[str, Any],
        sources: list[dict[str, Any]],
    ) -> None:
        for config in pack.get("artifacts", []):
            configured_path = config["path"]
            source_hash = self._hash_for(sources, configured_path)
            document = json.loads(self._resolve(configured_path).read_text(encoding="utf-8"))
            rows = rows_from_document(document, config)
            document_version = config.get("version") or dig(document, config.get("version_path"))
            for index, row in enumerate(rows):
                code = str(row.get(config["code_field"], "")).strip()
                display = next(
                    (str(row.get(field, "")).strip() for field in config["display_fields"] if str(row.get(field, "")).strip()),
                    "",
                )
                version = config.get("version") or row.get(config.get("version_field", "")) or document_version
                if not code or not display or not version:
                    raise SourceIntegrityError(f"incomplete artifact at {configured_path}[{index}]")
                authority = str(config["authority"])
                system = str(config["system"])
                artifact_id = "/".join((authority, system, str(version), code))
                start = configured_date(row, config, "effective_start")
                end = configured_date(row, config, "effective_end")
                if start and end and end < start:
                    if config.get("stale_end_before_start") == "drop_end":
                        end = None
                    else:
                        raise SourceIntegrityError(f"reversed active period at {configured_path}[{index}]")
                status = str(row.get(config.get("status_field", ""), "active") or "active").lower()
                billable = bool(row.get(config.get("billable_field", ""), config.get("billable_default", False)))
                requirements = row.get(config.get("requirements_field", "")) if config.get("requirements_field") else None
                properties = {
                    key: value
                    for key, value in row.items()
                    if key not in {config["code_field"], *config["display_fields"]}
                }
                locator = f"$[{index}]" if not config.get("container") else f"$.{config['container']}[{index}]"
                connection.execute(
                    """
                    INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact_id,
                        authority,
                        system,
                        str(version),
                        code,
                        display,
                        row.get(config.get("concept_field", "")),
                        start,
                        end,
                        status,
                        int(billable),
                        json.dumps(properties, sort_keys=True),
                        json.dumps(requirements, sort_keys=True) if requirements is not None else None,
                        configured_path,
                        source_hash,
                        locator,
                    ),
                )
                searchable = " ".join(
                    part for part in (display, str(row.get(config.get("consumer_field", ""), ""))) if part
                )
                connection.execute(
                    "INSERT INTO artifact_search(artifact_id, searchable_text) VALUES (?, ?)",
                    (artifact_id, searchable),
                )

    def _compile_constraints(
        self,
        connection: sqlite3.Connection,
        pack: dict[str, Any],
        sources: list[dict[str, Any]],
    ) -> None:
        for config in pack.get("constraints", []):
            configured_path = config["path"]
            source_hash = self._hash_for(sources, configured_path)
            document = json.loads(self._resolve(configured_path).read_text(encoding="utf-8"))
            if not isinstance(document, list):
                raise SourceIntegrityError(f"constraint source is not a list: {configured_path}")
            for index, row in enumerate(document):
                kind = config["kind"]
                left = row.get(config.get("left_field", ""))
                right = row.get(config.get("right_field", ""))
                target = row.get(config.get("code_field", ""))
                numeric = row.get(config.get("value_field", ""))
                properties = {key: row.get(key) for key in config.get("properties", [])}
                locator = f"$[{index}]"
                payload = {
                    "authority": config["authority"],
                    "kind": kind,
                    "left": left,
                    "right": right,
                    "target": target,
                    "start": row.get(config.get("effective_start_field", "")),
                    "end": row.get(config.get("effective_end_field", "")),
                    "properties": properties,
                    "source": configured_path,
                    "locator": locator,
                }
                constraint_id = sha256_bytes(canonical_json(payload))
                connection.execute(
                    "INSERT INTO constraints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        constraint_id,
                        config["authority"],
                        kind,
                        left,
                        right,
                        target,
                        float(numeric) if numeric not in (None, "") else None,
                        configured_date(row, config, "effective_start"),
                        configured_date(row, config, "effective_end"),
                        json.dumps(properties, sort_keys=True),
                        configured_path,
                        source_hash,
                        locator,
                    ),
                )

    def _compile_policies(
        self,
        connection: sqlite3.Connection,
        pack: dict[str, Any],
        sources: list[dict[str, Any]],
    ) -> None:
        for configured_path in pack.get("policy_documents", []):
            source_hash = self._hash_for(sources, configured_path)
            body = self._resolve(configured_path).read_text(encoding="utf-8")
            connection.execute(
                "INSERT INTO policy_documents VALUES (?, ?, ?, ?)",
                (source_hash, configured_path, source_hash, body),
            )

    @staticmethod
    def _verify_existing(destination: Path, expected_id: str) -> None:
        manifest_path = destination / "manifest.json"
        database_path = destination / "terminology.sqlite"
        if not manifest_path.is_file() or not database_path.is_file():
            raise SourceIntegrityError("existing snapshot is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("snapshot_id") != expected_id:
            raise SourceIntegrityError("snapshot identity mismatch")
        if manifest.get("database_sha256") != sha256_file(database_path):
            raise SourceIntegrityError("snapshot database hash mismatch")

