from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from .compiler import sha256_file
from .models import Candidate, EvidenceFact, EvidenceGraph, Normalization


class SnapshotIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    authority: str
    system: str
    version: str
    code: str
    display: str
    concept_id: str | None
    effective_start: str | None
    effective_end: str | None
    status: str
    billable: bool
    properties: dict[str, Any]
    requirements: list[dict[str, Any]] | None
    source_path: str
    source_hash: str
    source_locator: str


class TerminologySnapshot:
    def __init__(self, snapshot_directory: Path) -> None:
        self.directory = snapshot_directory.resolve()
        manifest_path = self.directory / "manifest.json"
        database_path = self.directory / "terminology.sqlite"
        if not manifest_path.is_file() or not database_path.is_file():
            raise SnapshotIntegrityError("snapshot is incomplete")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("database_sha256") != sha256_file(database_path):
            raise SnapshotIntegrityError("snapshot database hash mismatch")
        self.snapshot_id = str(self.manifest["snapshot_id"])
        self._database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self._database_path}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _fts_query(text: str) -> str:
        tokens = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
        if not tokens:
            raise ValueError("candidate search requires lexical evidence")
        return " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))

    def search(
        self,
        evidence: EvidenceGraph,
        effective_date: date,
        system_scope: set[str],
        limit_per_fact: int = 20,
    ) -> list[Candidate]:
        if not system_scope:
            raise ValueError("closed-world search requires an explicit system scope")
        combined: dict[str, Candidate] = {}
        placeholders = ",".join("?" for _ in system_scope)
        with self._connect() as connection:
            for fact in evidence.facts:
                query = self._fts_query(" ".join((fact.text, *map(str, fact.attributes.values()))))
                rows = connection.execute(
                    f"""
                    SELECT a.artifact_id, a.system, a.version, a.display, bm25(artifact_search) AS rank
                    FROM artifact_search
                    JOIN artifacts a USING (artifact_id)
                    WHERE artifact_search MATCH ?
                      AND a.system IN ({placeholders})
                      AND (a.effective_start IS NULL OR a.effective_start <= ?)
                      AND (a.effective_end IS NULL OR a.effective_end >= ?)
                      AND a.status = 'active'
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, *sorted(system_scope), effective_date.isoformat(), effective_date.isoformat(), limit_per_fact),
                ).fetchall()
                for row in rows:
                    score = -float(row["rank"])
                    existing = combined.get(row["artifact_id"])
                    fact_ids = set(existing.supporting_fact_ids if existing else ())
                    fact_ids.add(fact.fact_id)
                    combined[row["artifact_id"]] = Candidate(
                        artifact_id=row["artifact_id"],
                        system=row["system"],
                        version=row["version"],
                        display=row["display"],
                        score=max(score, existing.score if existing else score),
                        supporting_fact_ids=tuple(sorted(fact_ids)),
                    )
        return sorted(combined.values(), key=lambda item: (-item.score, item.artifact_id))

    def normalize_exact(
        self,
        fact: EvidenceFact,
        effective_date: date,
        system_scope: set[str],
    ) -> Normalization | None:
        candidates = self.search(
            EvidenceGraph(
                document_id=fact.source_spans[0].document_id,
                document_hash="normalization-only",
                facts=(fact,),
            ),
            effective_date,
            system_scope,
            limit_per_fact=50,
        )
        exact = []
        normalized_text = " ".join(fact.text.casefold().split())
        for candidate in candidates:
            artifact = self.artifact(candidate.artifact_id)
            if artifact and " ".join(artifact.display.casefold().split()) == normalized_text:
                exact.append(artifact)
        if len(exact) != 1:
            return None
        artifact = exact[0]
        return Normalization(
            concept_id=artifact.concept_id or artifact.artifact_id,
            system=artifact.system,
            source=f"snapshot:{self.snapshot_id}",
            confidence=1.0,
            alternatives=(),
        )

    def artifact(self, artifact_id: str) -> Artifact | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        return Artifact(
            artifact_id=row["artifact_id"],
            authority=row["authority"],
            system=row["system"],
            version=row["version"],
            code=row["code"],
            display=row["display"],
            concept_id=row["concept_id"],
            effective_start=row["effective_start"],
            effective_end=row["effective_end"],
            status=row["status"],
            billable=bool(row["billable"]),
            properties=json.loads(row["properties_json"]),
            requirements=json.loads(row["requirements_json"]) if row["requirements_json"] else None,
            source_path=row["source_path"],
            source_hash=row["source_hash"],
            source_locator=row["source_locator"],
        )

    def active_on(self, artifact: Artifact, service_date: date) -> bool:
        value = service_date.isoformat()
        return (
            artifact.status == "active"
            and (artifact.effective_start is None or artifact.effective_start <= value)
            and (artifact.effective_end is None or artifact.effective_end >= value)
        )

    def constraints(
        self,
        kinds: Iterable[str],
        codes: Iterable[str],
        service_date: date,
    ) -> list[dict[str, Any]]:
        kind_values = tuple(sorted(set(kinds)))
        code_values = tuple(sorted(set(codes)))
        if not kind_values or not code_values:
            return []
        kind_slots = ",".join("?" for _ in kind_values)
        code_slots = ",".join("?" for _ in code_values)
        value = service_date.isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM constraints
                WHERE kind IN ({kind_slots})
                  AND (
                    target_code IN ({code_slots})
                    OR left_code IN ({code_slots})
                    OR right_code IN ({code_slots})
                  )
                  AND (effective_start IS NULL OR effective_start <= ?)
                  AND (effective_end IS NULL OR effective_end >= ?)
                """,
                (*kind_values, *code_values, *code_values, *code_values, value, value),
            ).fetchall()
        return [
            {
                **dict(row),
                "properties": json.loads(row["properties_json"]),
            }
            for row in rows
        ]

