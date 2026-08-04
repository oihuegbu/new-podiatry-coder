from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .compiler import canonical_json


class AuditIntegrityError(RuntimeError):
    pass


class AuditStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS encounters (
                    encounter_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    document_hash TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    encounter_id TEXT NOT NULL REFERENCES encounters(encounter_id),
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(encounter_id, sequence),
                    UNIQUE(event_hash)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def begin(self, encounter_id: str, document_hash: str, snapshot_id: str, context_hash: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT * FROM encounters WHERE encounter_id = ?", (encounter_id,)).fetchone()
            if existing:
                if (existing["document_hash"], existing["snapshot_id"], existing["context_hash"]) != (document_hash, snapshot_id, context_hash):
                    raise AuditIntegrityError("encounter identity was reused with different immutable inputs")
                event = connection.execute(
                    "SELECT 1 FROM audit_events WHERE encounter_id = ? LIMIT 1", (encounter_id,)
                ).fetchone()
                if not event:
                    self._insert_event(
                        connection, encounter_id, "RECEIVED",
                        {"document_hash": document_hash, "snapshot_id": snapshot_id, "context_hash": context_hash},
                        now,
                    )
                elif not existing["result_json"]:
                    self._insert_event(
                        connection, encounter_id, "RETRY_STARTED",
                        {"document_hash": document_hash, "snapshot_id": snapshot_id, "context_hash": context_hash},
                        now,
                    )
                return
            connection.execute(
                "INSERT INTO encounters VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
                (encounter_id, "RECEIVED", document_hash, snapshot_id, context_hash, now, now),
            )
            self._insert_event(
                connection, encounter_id, "RECEIVED",
                {"document_hash": document_hash, "snapshot_id": snapshot_id, "context_hash": context_hash},
                now,
            )

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        encounter_id: str,
        event_type: str,
        payload: dict[str, Any],
        now: str,
    ) -> str:
        payload_bytes = canonical_json(payload)
        previous = connection.execute(
            "SELECT sequence, event_hash FROM audit_events WHERE encounter_id = ? ORDER BY sequence DESC LIMIT 1",
            (encounter_id,),
        ).fetchone()
        sequence = int(previous["sequence"]) + 1 if previous else 1
        previous_hash = str(previous["event_hash"]) if previous else "0" * 64
        event_hash = hashlib.sha256(
            canonical_json([encounter_id, sequence, event_type, json.loads(payload_bytes), previous_hash])
        ).hexdigest()
        connection.execute(
            "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?)",
            (encounter_id, sequence, event_type, payload_bytes.decode("utf-8"), previous_hash, event_hash, now),
        )
        updated = connection.execute(
            "UPDATE encounters SET state = ?, updated_at = ? WHERE encounter_id = ?",
            (event_type, now, encounter_id),
        )
        if updated.rowcount != 1:
            raise AuditIntegrityError("audit event references an unknown encounter")
        return event_hash

    def append(self, encounter_id: str, event_type: str, payload: dict[str, Any]) -> str:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._insert_event(connection, encounter_id, event_type, payload, now)

    def finish(self, encounter_id: str, state: str, result: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        result_json = canonical_json(result)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT result_json FROM encounters WHERE encounter_id = ?", (encounter_id,)
            ).fetchone()
            if not existing:
                raise AuditIntegrityError("cannot finish an unknown encounter")
            if existing["result_json"]:
                if existing["result_json"] != result_json.decode("utf-8"):
                    raise AuditIntegrityError("completed encounter result is immutable")
                return
            self._insert_event(
                connection, encounter_id, state,
                {"result_hash": hashlib.sha256(result_json).hexdigest()}, now,
            )
            updated = connection.execute(
                "UPDATE encounters SET state = ?, result_json = ?, updated_at = ? WHERE encounter_id = ?",
                (state, result_json.decode("utf-8"), now, encounter_id),
            )
            if updated.rowcount != 1:
                raise AuditIntegrityError("completed encounter disappeared")

    def verify_chain(self, encounter_id: str) -> None:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM audit_events WHERE encounter_id = ? ORDER BY sequence", (encounter_id,)).fetchall()
        previous_hash = "0" * 64
        for expected_sequence, row in enumerate(rows, 1):
            if row["sequence"] != expected_sequence or row["previous_hash"] != previous_hash:
                raise AuditIntegrityError("audit chain order is broken")
            expected_hash = hashlib.sha256(
                canonical_json([encounter_id, expected_sequence, row["event_type"], json.loads(row["payload_json"]), previous_hash])
            ).hexdigest()
            if expected_hash != row["event_hash"]:
                raise AuditIntegrityError("audit event hash mismatch")
            previous_hash = expected_hash

    def result(self, encounter_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT result_json FROM encounters WHERE encounter_id = ?", (encounter_id,)).fetchone()
        return json.loads(row["result_json"]) if row and row["result_json"] else None

    def record_failure(self, encounter_id: str, error: Exception) -> bool:
        with self._connect() as connection:
            encounter = connection.execute(
                "SELECT result_json FROM encounters WHERE encounter_id = ?", (encounter_id,)
            ).fetchone()
        if not encounter or encounter["result_json"]:
            return False
        self.append(encounter_id, "TECHNICAL_FAILURE", {"error_type": type(error).__name__})
        return True
