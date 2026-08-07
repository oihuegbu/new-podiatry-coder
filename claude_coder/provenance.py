"""Phase-0 evidence provenance + relation kernel (SHADOW — no release impact yet).

Two graph-compatible foundations, both behind an append-only repository seam so the
Phase-3 durable store swaps in without touching callers:

  1. EVIDENCE ANCHORING. Every quoted evidence span is anchored to an EXACT offset in
     the source and the slice is re-verified (note_text[start:end] == exact_text) — a
     plausible-looking quotation that is not verbatim in the source does NOT anchor.
     This never fabricates offsets; an unanchorable span is reported, never guessed.

  2. RELATION KERNEL. `RelationAssertion` is a first-class, content-addressed edge
     (subject, predicate, object) with identity + merge: re-assertions of the same edge
     ACCUMULATE evidentiary support instead of creating parallel edges, and any state
     disagreement collapses to UNCERTAIN (weak/conflicting relationships never become a
     confident PART_OF). No eligibility or billability decision lives here — this only
     records what the documentation asserts.

Agnostic: pure text/graph mechanics; no medical code, term, or scenario.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import EvidenceSpan, RelationAssertion, RelationState


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------- evidence anchoring
def anchor_offsets(note_text: str, exact_text: str) -> tuple[int, int] | None:
    """The [start, end) of the FIRST exact occurrence of `exact_text`, re-verified so
    the returned slice equals the quote. None when the quote is not verbatim present —
    never a fuzzy or approximate match (defensibility requires the exact span)."""
    if not exact_text:
        return None
    i = note_text.find(exact_text)
    if i < 0:
        return None
    j = i + len(exact_text)
    if note_text[i:j] != exact_text:          # invariant guard (should never trip)
        return None
    return i, j


def _span_id(document_sha256: str, document_version: str, span: EvidenceSpan,
             start: int, end: int) -> str:
    raw = "|".join((document_sha256, document_version, str(start), str(end),
                    _sha(span.text), str(span.section or ""), str(span.page or "")))
    return _sha(raw)


def anchor_span(note_text: str, span: EvidenceSpan, *, start_hint: int | None = None,
                document_version: str | None = None) -> EvidenceSpan:
    """Return a copy of `span` carrying its verified offsets + content hash, or an
    UNANCHORED copy (anchored=False) when the quote is not verbatim in the source."""
    doc_hash = _sha(note_text)
    version = str(document_version or doc_hash)
    hint = span.start if start_hint is None else start_hint
    pos = None
    if isinstance(hint, int) and hint >= 0:
        end = hint + len(span.text)
        if note_text[hint:end] == span.text:
            pos = (hint, end)
    if pos is None:
        pos = anchor_offsets(note_text, span.text)
    if pos is None:
        return replace(span, start=None, end=None,
                       text_sha256=_sha(span.text), anchored=False,
                       document_sha256=doc_hash, document_version=version, span_id=None)
    i, j = pos
    return replace(span, start=i, end=j, text_sha256=_sha(span.text), anchored=True,
                   document_sha256=doc_hash, document_version=version,
                   span_id=_span_id(doc_hash, version, span, i, j))


def anchor_facts(note_text: str, facts: list, document_version: str | None = None) -> list:
    """Anchor every fact's evidence spans IN PLACE (facts are mutable; spans are frozen,
    so each span is replaced with an anchored copy). Transparent to billing — only adds
    offsets/hash; the span text is unchanged. Repeated quotations are assigned
    successive exact occurrences when available, while an extractor-supplied offset
    is accepted only after exact slice verification."""
    occurrences: dict[str, list[int]] = {}
    used: dict[str, int] = {}
    for f in facts:
        anchored = []
        for s in (f.evidence or []):
            hint = s.start
            if hint is None and s.text:
                if s.text not in occurrences:
                    positions, start = [], 0
                    while True:
                        pos = note_text.find(s.text, start)
                        if pos < 0:
                            break
                        positions.append(pos)
                        start = pos + max(1, len(s.text))
                    occurrences[s.text] = positions
                idx = used.get(s.text, 0)
                positions = occurrences[s.text]
                if idx < len(positions):
                    hint = positions[idx]
                    used[s.text] = idx + 1
            anchored.append(anchor_span(note_text, s, start_hint=hint,
                                        document_version=document_version))
        f.evidence = anchored
    return facts


def anchoring_report(facts: list) -> dict[str, Any]:
    """Shadow coverage report: how many evidence spans anchored, and which did not."""
    total = anchored = 0
    unanchored: list[dict] = []
    for f in facts:
        for s in (f.evidence or []):
            total += 1
            if s.anchored:
                anchored += 1
            else:
                unanchored.append({"fact_id": f.fact_id, "text": s.text[:80]})
    return {
        "spans_total": total,
        "spans_anchored": anchored,
        "coverage": (anchored / total) if total else 1.0,
        "unanchored": unanchored,
    }


# ------------------------------------------------------------------ relation kernel
def relation_key(subject_event_id: str, predicate: str, object_event_id: str) -> str:
    """Content-addressed identity for an edge — so the same (subject, predicate, object)
    asserted by different passes is ONE edge with accumulated support, not duplicates."""
    raw = f"{subject_event_id}|{predicate}|{object_event_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def merge_relations(relations: list[RelationAssertion]) -> list[RelationAssertion]:
    """Deduplicate edges by content identity. Re-assertions union their evidence and
    raise support; ANY state disagreement collapses to UNCERTAIN so a conflicting or
    weakly-asserted relationship never survives as a confident edge. Inputs are not
    mutated."""
    by_id: dict[str, RelationAssertion] = {}
    for r in relations:
        rid = r.relation_id
        if rid not in by_id:
            by_id[rid] = replace(r, evidence_span_ids=list(r.evidence_span_ids))
            continue
        m = by_id[rid]
        m.evidence_span_ids = list(dict.fromkeys(
            list(m.evidence_span_ids) + list(r.evidence_span_ids)))
        m.support += r.support
        m.confidence = max(m.confidence, r.confidence)
        if m.state is not r.state:
            m.state = RelationState.UNCERTAIN
    return list(by_id.values())


class RelationIntegrityError(ValueError):
    """A relation graph cannot safely drive eligibility."""


_SYMMETRIC = {"separate_from", "same_episode_as"}


def bind_relation_evidence(relations: list[RelationAssertion], facts: list) -> list[RelationAssertion]:
    """Resolve extractor event references to verified location-specific span ids."""
    by_event = {f.fact_id: [s.span_id for s in (f.evidence or []) if s.anchored and s.span_id]
                for f in facts}
    bound = []
    for rel in relations or []:
        span_ids: list[str] = []
        for ref in rel.evidence_span_ids or []:
            if str(ref).startswith("event:"):
                span_ids.extend(by_event.get(str(ref).split(":", 1)[1], []))
            else:
                span_ids.append(str(ref))
        bound.append(replace(rel, evidence_span_ids=list(dict.fromkeys(span_ids))))
    return bound


def validate_relations(relations: list[RelationAssertion], facts: list) -> list[RelationAssertion]:
    """Validate graph identity before a relation can suppress or separate an event.

    Invalid input fails closed rather than being silently dropped, because dropping a
    malformed edge can turn an unknown relationship into an apparently eligible line.
    """
    event_ids = {f.fact_id for f in facts if f.fact_id}
    span_ids = {s.span_id for f in facts for s in (f.evidence or []) if s.span_id}
    normalized: list[RelationAssertion] = []
    for rel in relations or []:
        if rel.subject_event_id not in event_ids or rel.object_event_id not in event_ids:
            raise RelationIntegrityError(
                f"relation {rel.relation_id} references an unknown clinical event")
        pred = rel.predicate.value if hasattr(rel.predicate, "value") else str(rel.predicate)
        if rel.subject_event_id == rel.object_event_id:
            raise RelationIntegrityError(f"relation {rel.relation_id} is self-referential")
        if not 0.0 <= float(rel.confidence) <= 1.0:
            raise RelationIntegrityError(f"relation {rel.relation_id} has invalid confidence")
        if not rel.evidence_span_ids:
            raise RelationIntegrityError(f"relation {rel.relation_id} has no anchored evidence")
        if set(rel.evidence_span_ids or []) - span_ids:
            raise RelationIntegrityError(
                f"relation {rel.relation_id} references unverified evidence spans")
        current = rel
        if pred in _SYMMETRIC and rel.subject_event_id > rel.object_event_id:
            current = replace(rel, subject_event_id=rel.object_event_id,
                              object_event_id=rel.subject_event_id)
        normalized.append(current)
    return merge_relations(normalized)


# ---------------------------------------------------- append-only repository seam
class AuditRepository:
    """Append-only shadow audit sink. Phase 3 replaces the implementation with a durable
    store (PostgreSQL) behind THIS interface, so Phase-0 callers never change."""

    def append(self, encounter_id: str, kind: str, record: dict) -> str:  # pragma: no cover
        raise NotImplementedError


class NullAuditRepository(AuditRepository):
    """Default no-op sink (tests / environments without a writable output dir)."""

    def append(self, encounter_id: str, kind: str, record: dict) -> str:
        payload = {"encounter_id": encounter_id, "kind": kind, "record": record}
        return _sha(json.dumps(payload, sort_keys=True, default=str))


class SqliteAuditRepository(AuditRepository):
    """Phase 3: durable, hash-chained, append-only provenance store in a DEDICATED SQLite
    database (WAL, synchronous=FULL, INSERT-only enforced by triggers), kept separate from
    the authoritative compliance.db (different lifecycle/retention). Same contract as
    JsonlAuditRepository: append() returns the entry's record_sha256, entries are per-
    encounter hash-chained, and in strict mode a write failure RAISES so the pipeline routes
    SYSTEM_HOLD (release provenance is fail-closed, never an empty-success fallback).

    Swap PostgreSQL in behind AuditRepository only if a scale trigger appears (multi-host
    concurrent writers, high write concurrency, or HA-replication/PITR compliance) -- the
    seam and the hash-chained record shape are unchanged, so it is not a rewrite.
    """

    _DDL = (
        "CREATE TABLE IF NOT EXISTS audit_log ("
        " seq INTEGER PRIMARY KEY AUTOINCREMENT,"
        " encounter_id TEXT NOT NULL,"
        " kind TEXT NOT NULL,"
        " recorded_at TEXT NOT NULL,"
        " control_mode TEXT NOT NULL,"
        " previous_record_sha256 TEXT NOT NULL,"
        " record_json TEXT NOT NULL,"
        " record_sha256 TEXT NOT NULL)",
        "CREATE INDEX IF NOT EXISTS idx_audit_encounter ON audit_log(encounter_id, seq)",
        "CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log"
        " BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END",
        "CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log"
        " BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END",
    )

    def __init__(self, db_path, *, strict: bool = True) -> None:
        self.db_path = Path(db_path)
        self.strict = strict

    def _connect(self):
        import sqlite3
        if self.db_path.parent and str(self.db_path.parent):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None -> manual transaction control, so append() can take the
        # write lock with BEGIN IMMEDIATE before reading the previous hash (Codex F6-R4).
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=FULL")
        for stmt in self._DDL:
            conn.execute(stmt)
        return conn

    def append(self, encounter_id: str, kind: str, record: dict) -> str:
        try:
            conn = self._connect()
            try:
                # Serialize the read-previous + append so two concurrent writers cannot both
                # observe the same predecessor and fork the per-encounter hash chain. The
                # RESERVED lock is taken at BEGIN IMMEDIATE (before the SELECT), and other
                # writers block up to busy_timeout instead of racing. (Codex F6-R4.)
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT record_sha256 FROM audit_log WHERE encounter_id=?"
                    " ORDER BY seq DESC LIMIT 1", (str(encounter_id),)).fetchone()
                previous = row[0] if row else ""
                entry = {
                    "encounter_id": encounter_id,
                    "kind": kind,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "control_mode": "ENFORCED_FAIL_CLOSED",
                    "previous_record_sha256": previous,
                    "record": record,
                }
                entry["record_sha256"] = _sha(json.dumps(entry, sort_keys=True, default=str))
                conn.execute(
                    "INSERT INTO audit_log(encounter_id, kind, recorded_at, control_mode,"
                    " previous_record_sha256, record_json, record_sha256)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (encounter_id, kind, entry["recorded_at"], entry["control_mode"],
                     previous, json.dumps(record, sort_keys=True, default=str),
                     entry["record_sha256"]))
                conn.commit()
                return entry["record_sha256"]
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
            finally:
                conn.close()
        except Exception:
            if self.strict:
                raise
            return ""

    def records(self, encounter_id: str | None = None) -> list[dict]:
        """Read the chained records back (verification/audit; not release-gating)."""
        conn = self._connect()
        try:
            q = ("SELECT encounter_id, kind, recorded_at, control_mode,"
                 " previous_record_sha256, record_json, record_sha256 FROM audit_log")
            args: tuple = ()
            if encounter_id is not None:
                q += " WHERE encounter_id=?"
                args = (str(encounter_id),)
            q += " ORDER BY seq"
            return [{"encounter_id": r[0], "kind": r[1], "recorded_at": r[2],
                     "control_mode": r[3], "previous_record_sha256": r[4],
                     "record": json.loads(r[5]), "record_sha256": r[6]}
                    for r in conn.execute(q, args).fetchall()]
        finally:
            conn.close()


    def verify_chain(self, encounter_id: str | None = None) -> list[str]:
        """Return a list of integrity problems (empty => intact). Detects hash tampering (a
        stored record_sha256 that does not equal the recomputed hash of its fields), broken
        or forked links (previous_record_sha256 != the prior record\'s hash within an
        encounter), and reordering/gaps (seq not strictly increasing). (Codex F6-R4.)"""
        conn = self._connect()
        try:
            q = ("SELECT seq, encounter_id, kind, recorded_at, control_mode,"
                 " previous_record_sha256, record_json, record_sha256 FROM audit_log")
            args: tuple = ()
            if encounter_id is not None:
                q += " WHERE encounter_id=?"
                args = (str(encounter_id),)
            q += " ORDER BY seq"
            rows = conn.execute(q, args).fetchall()
        finally:
            conn.close()
        problems: list[str] = []
        last_sha: dict[str, str] = {}
        last_seq = -1
        for (seq, enc, kind, at, mode, prev, rj, sha) in rows:
            if seq <= last_seq:
                problems.append(f"seq {seq}: not strictly increasing")
            last_seq = seq
            entry = {"encounter_id": enc, "kind": kind, "recorded_at": at,
                     "control_mode": mode, "previous_record_sha256": prev,
                     "record": json.loads(rj)}
            if _sha(json.dumps(entry, sort_keys=True, default=str)) != sha:
                problems.append(f"seq {seq} enc {enc}: record hash mismatch (tampered)")
            expected_prev = last_sha.get(enc, "")
            if prev != expected_prev:
                problems.append(f"seq {seq} enc {enc}: broken/forked chain link")
            last_sha[enc] = sha
        return problems


class JsonlAuditRepository(AuditRepository):
    """Per-encounter hash-chained JSONL artifact.

    Enforced callers require a successful durable append before release. A failure
    raises and the pipeline routes SYSTEM_HOLD; there is no empty-success fallback.
    """

    def __init__(self, root, *, strict: bool = True) -> None:
        self.root = Path(root)
        self.strict = strict

    def append(self, encounter_id: str, kind: str, record: dict) -> str:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(encounter_id))
            path = self.root / f"{safe}.jsonl"
            previous = ""
            if path.exists() and path.stat().st_size:
                with open(path, encoding="utf-8") as prior:
                    tail = [line for line in prior.read().splitlines() if line.strip()]
                if not tail:
                    raise OSError("audit chain has no readable final record")
                last = json.loads(tail[-1])
                # One-time deterministic bridge from pre-chain shadow records.
                previous = str(last.get("record_sha256") or
                               _sha(json.dumps(last, sort_keys=True, default=str)))
            entry = {
                "encounter_id": encounter_id,
                "kind": kind,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "control_mode": "ENFORCED_FAIL_CLOSED",
                "previous_record_sha256": previous,
                "record": record,
            }
            entry["record_sha256"] = _sha(json.dumps(entry, sort_keys=True, default=str))
            line = json.dumps(entry, sort_keys=True, default=str)
            with open(path, "a") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            return entry["record_sha256"]
        except Exception:
            if self.strict:
                raise
            return ""
