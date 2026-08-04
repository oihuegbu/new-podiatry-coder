from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 3
COMPILER_VERSION = "0.3.0"


class SourceIntegrityError(RuntimeError):
    """The configured source pack cannot be compiled without guessing."""


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_compiler_tree(path: Path) -> None:
    """Remove a compiler-owned tree even after it was made read-only."""
    if not path.exists():
        return
    for directory, child_directories, _files in os.walk(path):
        os.chmod(directory, 0o700)
        for child in child_directories:
            os.chmod(Path(directory) / child, 0o700)
    shutil.rmtree(path)


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
    formats = ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y", "%Y-%m", "%Y")
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
    if config.get("code_keyed"):
        if not isinstance(rows, dict):
            raise SourceIntegrityError("code-keyed source must resolve to an object")
        key_field = config.get("key_field_from_map", config.get("code_field", "code"))
        mapped = []
        for key, value in rows.items():
            if not isinstance(value, dict):
                raise SourceIntegrityError("code-keyed values must be objects")
            mapped.append({key_field: key, **value})
        rows = mapped
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
        CREATE TABLE lookup_terms (
            lookup_id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
            term TEXT NOT NULL,
            term_kind TEXT NOT NULL,
            authority TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            source_locator TEXT NOT NULL
        );
        CREATE INDEX lookup_terms_artifact ON lookup_terms(artifact_id);
        CREATE VIRTUAL TABLE lookup_search USING fts5(
            lookup_id UNINDEXED,
            term,
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE TABLE artifact_semantics (
            semantic_id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            source_locator TEXT NOT NULL
        );
        CREATE INDEX artifact_semantics_artifact ON artifact_semantics(artifact_id, kind);
        CREATE TABLE coverage_policies (
            policy_key TEXT PRIMARY KEY,
            authority TEXT NOT NULL,
            policy_id TEXT NOT NULL,
            title TEXT NOT NULL,
            effective_start TEXT,
            effective_end TEXT,
            jurisdictions_json TEXT NOT NULL,
            payer_types_json TEXT NOT NULL,
            payer_identifiers_json TEXT NOT NULL,
            contract_profiles_json TEXT NOT NULL,
            properties_json TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            source_locator TEXT NOT NULL
        );
        CREATE TABLE coverage_procedures (
            policy_key TEXT NOT NULL REFERENCES coverage_policies(policy_key),
            normalized_code TEXT NOT NULL,
            PRIMARY KEY(policy_key, normalized_code)
        );
        CREATE INDEX coverage_procedures_code ON coverage_procedures(normalized_code);
        CREATE TABLE coverage_diagnoses (
            policy_key TEXT NOT NULL REFERENCES coverage_policies(policy_key),
            normalized_code TEXT NOT NULL,
            disposition TEXT NOT NULL,
            procedure_scope TEXT NOT NULL DEFAULT '',
            group_id TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT 'unspecified',
            PRIMARY KEY(policy_key, normalized_code, disposition, procedure_scope, group_id, role)
        );
        CREATE INDEX coverage_diagnoses_code ON coverage_diagnoses(normalized_code, procedure_scope);
        CREATE TABLE artifact_tokens (
            artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
            token TEXT NOT NULL,
            material INTEGER NOT NULL CHECK(material IN (0, 1)),
            PRIMARY KEY(artifact_id, token)
        );
        CREATE INDEX artifact_tokens_material ON artifact_tokens(artifact_id, material);
        CREATE TABLE token_statistics (
            system TEXT NOT NULL,
            token TEXT NOT NULL,
            document_frequency INTEGER NOT NULL,
            system_artifact_count INTEGER NOT NULL,
            PRIMARY KEY(system, token)
        );
        CREATE TABLE jurisdiction_aliases (
            canonical TEXT NOT NULL,
            alias TEXT NOT NULL,
            authority TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            source_locator TEXT NOT NULL,
            PRIMARY KEY(canonical, alias, authority)
        );
        CREATE INDEX jurisdiction_alias_lookup ON jurisdiction_aliases(alias);
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
            "compiler_source_sha256": sha256_file(Path(__file__).resolve()),
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
                self._compile_enrichments(connection, pack, sources)
                self._compile_lookup_maps(connection, pack, sources)
                self._compile_semantic_maps(connection, pack, sources)
                self._compile_descriptor_profiles(connection, pack)
                self._compile_constraints(connection, pack, sources)
                self._compile_coverage(connection, pack, sources)
                self._compile_jurisdictions(connection, pack, sources)
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
                "compiler_source_sha256": identity["compiler_source_sha256"],
                "compiled_at": datetime.now(timezone.utc).isoformat(),
                "capabilities": pack.get("capabilities", {}),
                "semantic_derivations": pack.get("semantic_derivations", {}),
                "sources": sources,
                "database_sha256": sha256_file(database_path),
            }
            (temp / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
            os.chmod(database_path, 0o444)
            os.chmod(temp / "manifest.json", 0o444)
            os.chmod(temp, 0o555)
            try:
                os.replace(temp, destination)
            except FileExistsError:
                # Another compiler published the same content-addressed build
                # while this process was working.  Accept it only after full
                # identity/hash verification, and never leave our read-only
                # staging directory behind.
                remove_compiler_tree(temp)
                self._verify_existing(destination, snapshot_id)
            return destination
        except BaseException:
            if temp.exists():
                remove_compiler_tree(temp)
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
        for section in (
            "artifacts", "constraints", "enrichments", "lookup_maps", "semantic_maps",
            "coverage_sources", "jurisdiction_sources",
        ):
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
                        dig(row, config.get("concept_field")),
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
                    part
                    for part in (
                        display,
                        *(str(row.get(field, "")).strip() for field in config.get("search_fields", [])),
                    )
                    if part
                )
                connection.execute(
                    "INSERT INTO artifact_search(artifact_id, searchable_text) VALUES (?, ?)",
                    (artifact_id, searchable),
                )

    @staticmethod
    def _normalized_code(value: Any) -> str:
        return "".join(character for character in str(value).upper() if character.isalnum())

    def _artifact_map(self, connection: sqlite3.Connection, system: str) -> dict[str, str]:
        return {
            self._normalized_code(row[1]): row[0]
            for row in connection.execute(
                "SELECT artifact_id, code FROM artifacts WHERE system = ?",
                (system,),
            )
        }

    def _compile_enrichments(
        self,
        connection: sqlite3.Connection,
        pack: dict[str, Any],
        sources: list[dict[str, Any]],
    ) -> None:
        for config in pack.get("enrichments", []):
            configured_path = config["path"]
            source_hash = self._hash_for(sources, configured_path)
            document = json.loads(self._resolve(configured_path).read_text(encoding="utf-8"))
            rows = rows_from_document(document, {**config, "key_field_from_map": "code"})
            artifact_map = self._artifact_map(connection, str(config["system"]))
            for index, row in enumerate(rows):
                artifact_id = artifact_map.get(self._normalized_code(row.get(config.get("code_field", "code"))))
                if not artifact_id:
                    continue
                payload = {key: row.get(key) for key in config.get("properties", [])}
                locator = f"$.{config.get('container')}[{index}]" if config.get("container") else f"$[{index}]"
                identity = sha256_bytes(canonical_json([artifact_id, config.get("kind", "properties"), payload, source_hash, locator]))
                connection.execute(
                    "INSERT INTO artifact_semantics VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (identity, artifact_id, config.get("kind", "properties"), json.dumps(payload, sort_keys=True), configured_path, source_hash, locator),
                )

    def _compile_lookup_maps(
        self,
        connection: sqlite3.Connection,
        pack: dict[str, Any],
        sources: list[dict[str, Any]],
    ) -> None:
        for config in pack.get("lookup_maps", []):
            configured_path = config["path"]
            source_hash = self._hash_for(sources, configured_path)
            document = json.loads(self._resolve(configured_path).read_text(encoding="utf-8"))
            mapping = dig(document, config.get("container")) if config.get("container") else document
            if not isinstance(mapping, dict):
                raise SourceIntegrityError("lookup map must resolve to an object")
            artifact_map = self._artifact_map(connection, str(config["system"]))
            for code, values in mapping.items():
                artifact_id = artifact_map.get(self._normalized_code(code))
                if not artifact_id:
                    continue
                terms = values if isinstance(values, list) else values.get(config.get("terms_field", "terms"), []) if isinstance(values, dict) else []
                for index, term in enumerate(terms):
                    if not str(term).strip():
                        continue
                    locator = f"$.{config.get('container')}.{code}[{index}]" if config.get("container") else f"$.{code}[{index}]"
                    identity = sha256_bytes(canonical_json([artifact_id, term, config["term_kind"], source_hash, locator]))
                    connection.execute(
                        "INSERT OR IGNORE INTO lookup_terms VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (identity, artifact_id, str(term), config["term_kind"], config["authority"], configured_path, source_hash, locator),
                    )
                    connection.execute("INSERT OR IGNORE INTO lookup_search VALUES (?, ?)", (identity, str(term)))

    def _compile_semantic_maps(
        self,
        connection: sqlite3.Connection,
        pack: dict[str, Any],
        sources: list[dict[str, Any]],
    ) -> None:
        for config in pack.get("semantic_maps", []):
            configured_path = config["path"]
            source_hash = self._hash_for(sources, configured_path)
            document = json.loads(self._resolve(configured_path).read_text(encoding="utf-8"))
            mapping = dig(document, config.get("container")) if config.get("container") else document
            if not isinstance(mapping, dict):
                raise SourceIntegrityError("semantic map must resolve to an object")
            artifact_map = self._artifact_map(connection, str(config["system"]))
            for code, value in mapping.items():
                artifact_id = artifact_map.get(self._normalized_code(code))
                if not artifact_id:
                    continue
                payload = value if isinstance(value, dict) else {"value": value}
                locator = f"$.{config.get('container')}.{code}" if config.get("container") else f"$.{code}"
                identity = sha256_bytes(canonical_json([artifact_id, config["kind"], payload, source_hash, locator]))
                connection.execute(
                    "INSERT OR IGNORE INTO artifact_semantics VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (identity, artifact_id, config["kind"], json.dumps(payload, sort_keys=True), configured_path, source_hash, locator),
                )

    @staticmethod
    def _field_values(row: dict[str, Any], fields: list[str]) -> set[str]:
        values: set[str] = set()
        for field in fields:
            raw = row.get(field, [])
            if not isinstance(raw, list):
                raw = [raw]
            if any(isinstance(item, (dict, list)) for item in raw):
                raise SourceIntegrityError(f"field {field!r} is structured but no structured mapping is configured")
            values.update(str(item).strip() for item in raw if str(item).strip())
        return values

    def _compile_coverage(
        self,
        connection: sqlite3.Connection,
        pack: dict[str, Any],
        sources: list[dict[str, Any]],
    ) -> None:
        for config in pack.get("coverage_sources", []):
            configured_path = config["path"]
            source_hash = self._hash_for(sources, configured_path)
            document = json.loads(self._resolve(configured_path).read_text(encoding="utf-8"))
            rows = rows_from_document(document, config)
            for index, row in enumerate(rows):
                policy_id = str(row.get(config["policy_id_field"], "")).strip()
                if not policy_id:
                    raise SourceIntegrityError(f"coverage row lacks policy identity at {configured_path}[{index}]")
                locator = f"$[{index}]"
                policy_key = sha256_bytes(canonical_json([config["authority"], policy_id, source_hash, locator]))
                jurisdictions = self._field_values(row, config.get("jurisdiction_fields", []))
                payer_types = set(config.get("payer_types", [])) | self._field_values(row, config.get("payer_type_fields", []))
                payer_identifiers = set(config.get("payer_identifiers", [])) | self._field_values(row, config.get("payer_identifier_fields", []))
                contract_profiles = set(config.get("contract_profiles", [])) | self._field_values(row, config.get("contract_profile_fields", []))
                properties = {key: row.get(key) for key in config.get("properties", [])}
                connection.execute(
                    "INSERT INTO coverage_policies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        policy_key, config["authority"], policy_id,
                        str(row.get(config.get("title_field", ""), policy_id)),
                        configured_date(row, config, "effective_start"),
                        configured_date(row, config, "effective_end"),
                        json.dumps(sorted(jurisdictions)),
                        json.dumps(sorted(payer_types)),
                        json.dumps(sorted(payer_identifiers)),
                        json.dumps(sorted(contract_profiles)),
                        json.dumps(properties, sort_keys=True),
                        configured_path, source_hash, locator,
                    ),
                )
                for value in self._field_values(row, config.get("procedure_fields", [])):
                    connection.execute("INSERT OR IGNORE INTO coverage_procedures VALUES (?, ?)", (policy_key, self._normalized_code(value)))
                for disposition, fields in config.get("diagnosis_fields", {}).items():
                    for value in self._field_values(row, fields):
                        connection.execute(
                            "INSERT OR IGNORE INTO coverage_diagnoses VALUES (?, ?, ?, '', '', 'unspecified')",
                            (policy_key, self._normalized_code(value), disposition),
                        )
                for group_config in config.get("diagnosis_group_fields", []):
                    groups = row.get(group_config["field"], [])
                    if not isinstance(groups, list) or any(not isinstance(group, dict) for group in groups):
                        raise SourceIntegrityError(
                            f"coverage diagnosis groups changed shape at {configured_path}[{index}]"
                        )
                    for group in groups:
                        codes = group.get(group_config["codes_field"], [])
                        scopes = group.get(group_config["procedure_scope_field"], [])
                        if not isinstance(codes, list) or not isinstance(scopes, list):
                            raise SourceIntegrityError(
                                f"coverage diagnosis group fields changed shape at {configured_path}[{index}]"
                            )
                        normalized_scopes = {self._normalized_code(value) for value in scopes if self._normalized_code(value)}
                        if not normalized_scopes:
                            normalized_scopes = {""}
                        group_id = str(group.get(group_config.get("group_id_field", ""), "")).strip()
                        role = str(group.get(group_config.get("role_field", ""), "unspecified")).strip() or "unspecified"
                        allowed_roles = set(group_config.get("allowed_roles", ["unspecified"]))
                        if role not in allowed_roles:
                            raise SourceIntegrityError(
                                f"unknown coverage diagnosis role {role!r} at {configured_path}[{index}]"
                            )
                        for code in codes:
                            normalized_diagnosis = self._normalized_code(code)
                            if not normalized_diagnosis:
                                continue
                            for procedure_scope in normalized_scopes:
                                connection.execute(
                                    "INSERT OR IGNORE INTO coverage_diagnoses VALUES (?, ?, ?, ?, ?, ?)",
                                    (
                                        policy_key, normalized_diagnosis, group_config["disposition"],
                                        procedure_scope, group_id, role,
                                    ),
                                )

    @staticmethod
    def _descriptor_tokens(value: str, minimum_length: int) -> set[str]:
        return {
            token for token in re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)
            if len(token) >= minimum_length
        }

    def _compile_descriptor_profiles(self, connection: sqlite3.Connection, pack: dict[str, Any]) -> None:
        config = pack.get("semantic_derivations", {}).get("descriptor_token_requirements")
        if not config or not config.get("enabled"):
            return
        maximum_ratio = float(config["maximum_document_frequency_ratio"])
        minimum_length = int(config.get("minimum_token_length", 2))
        if not 0 < maximum_ratio < 1 or minimum_length < 1:
            raise SourceIntegrityError("invalid descriptor-token derivation configuration")
        rows = connection.execute("SELECT artifact_id, system, display FROM artifacts").fetchall()
        by_artifact: dict[str, tuple[str, set[str]]] = {}
        frequencies: dict[tuple[str, str], int] = {}
        totals: dict[str, int] = {}
        for artifact_id, system, display in rows:
            tokens = self._descriptor_tokens(display, minimum_length)
            by_artifact[artifact_id] = (system, tokens)
            totals[system] = totals.get(system, 0) + 1
            for token in tokens:
                key = (system, token)
                frequencies[key] = frequencies.get(key, 0) + 1
        connection.executemany(
            "INSERT INTO token_statistics VALUES (?, ?, ?, ?)",
            ((system, token, count, totals[system]) for (system, token), count in sorted(frequencies.items())),
        )
        connection.executemany(
            "INSERT INTO artifact_tokens VALUES (?, ?, ?)",
            (
                (artifact_id, token, int(frequencies[(system, token)] / totals[system] <= maximum_ratio))
                for artifact_id, (system, tokens) in by_artifact.items() for token in sorted(tokens)
            ),
        )

    def _compile_jurisdictions(
        self,
        connection: sqlite3.Connection,
        pack: dict[str, Any],
        sources: list[dict[str, Any]],
    ) -> None:
        for config in pack.get("jurisdiction_sources", []):
            configured_path = config["path"]
            source_hash = self._hash_for(sources, configured_path)
            document = json.loads(self._resolve(configured_path).read_text(encoding="utf-8"))
            rows = rows_from_document(document, config)
            for index, row in enumerate(rows):
                canonical = str(row.get(config["canonical_field"], "")).casefold().strip()
                if not canonical:
                    raise SourceIntegrityError("jurisdiction source row lacks canonical identity")
                aliases = self._field_values(row, config.get("alias_fields", [])) | {canonical}
                locator = f"$[{index}]" if not config.get("container") else f"$.{config['container']}[{index}]"
                for alias in aliases:
                    connection.execute(
                        "INSERT OR IGNORE INTO jurisdiction_aliases VALUES (?, ?, ?, ?, ?, ?)",
                        (canonical, alias.casefold().strip(), config["authority"], configured_path, source_hash, locator),
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

