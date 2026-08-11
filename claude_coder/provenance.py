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
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import EvidenceSpan, RelationAssertion, RelationState


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _witness_payload(entry: dict) -> dict:
    """The sealed portion of a witness entry — everything except the seal itself."""
    return {k: v for k, v in entry.items() if k != "seal"}


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


# Reconciliation statuses this module can establish DETERMINISTICALLY. They are written by
# `reconcile_relations` only; nothing an extraction model emits can set them, because the
# whole point is that an edge asserted by the same model that authored the events is not
# independent support for that edge. (Codex F6-R3.)
UNRECONCILED = "unreconciled"
CORROBORATED = "corroborated"            # independently re-asserted (support >= threshold)
SOURCE_COLOCATED = "source_colocated"    # one verified verbatim span documents BOTH endpoints


def reconcile_relations(relations: list[RelationAssertion], facts: list,
                        *, min_independent_assertions: int = 2) -> list[RelationAssertion]:
    """Stamp each edge with how -- if at all -- it was independently reconciled.

    Two deterministic, re-checkable criteria, in priority order:

      1. CORROBORATED -- the identical (subject, predicate, object) edge was asserted at least
         `min_independent_assertions` times and merged, i.e. more than one pass agreed.
      2. SOURCE_COLOCATED -- at least one ANCHORED evidence span is shared by BOTH endpoint
         events, i.e. a single verbatim passage of the source document states the two events
         together. That is a property of the document, not of the model's self-confidence.

    Anything else stays UNRECONCILED. This never raises confidence and never changes state; it
    only records a verifiable provenance fact that release controls may then require.
    """
    spans_by_event = {f.fact_id: {s.span_id for s in (f.evidence or [])
                                  if getattr(s, "anchored", False) and s.span_id}
                      for f in facts if getattr(f, "fact_id", None)}
    out: list[RelationAssertion] = []
    for rel in relations or []:
        status = UNRECONCILED
        subject_spans = spans_by_event.get(rel.subject_event_id, set())
        object_spans = spans_by_event.get(rel.object_event_id, set())
        shared = subject_spans & object_spans & set(rel.evidence_span_ids or [])
        if int(getattr(rel, "support", 1) or 1) >= int(min_independent_assertions):
            status = CORROBORATED
        elif shared:
            status = SOURCE_COLOCATED
        out.append(replace(rel, reconciliation_status=status))
    return out


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
    # Merge first (so re-assertions accumulate support and conflicting states collapse to
    # UNCERTAIN), THEN record how each surviving edge was reconciled. A release control that
    # requires reconciliation therefore reads a value this deterministic layer wrote, never
    # one the extraction model supplied. (Codex F6-R3.)
    return reconcile_relations(merge_relations(normalized), facts)


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
        " record_sha256 TEXT NOT NULL,"
        # The witness record that sealed this row, so the anchoring is BIDIRECTIONAL: the
        # witness proves the row exists, and the row proves the witness line exists. Erasing
        # the terminal record now requires consistently truncating BOTH sides. (Codex F6-R4-A.)
        " witness_sha256 TEXT)",
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
        # busy_timeout FIRST: switching journal mode and running the DDL below both take
        # locks, and until the busy handler is installed a concurrent opener gets an immediate
        # SQLITE_BUSY ("database is locked") instead of waiting. Ordering this after the
        # journal-mode switch made simultaneous first-opens fail. (Post-fix review.)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        for stmt in self._DDL:
            conn.execute(stmt)
        # Forward migration for stores created before the witness column existed. ADD COLUMN
        # is DDL, so the append-only UPDATE/DELETE triggers do not (and must not) block it;
        # pre-existing rows keep a NULL witness and verify_chain reports them as unwitnessed
        # rather than silently accepting them.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(audit_log)").fetchall()}
        if "witness_sha256" not in cols:
            try:
                conn.execute("ALTER TABLE audit_log ADD COLUMN witness_sha256 TEXT")
            except sqlite3.OperationalError as exc:
                # A concurrent opener may have migrated between the probe and here. Adopt its
                # result; anything else is a real schema failure and must surface.
                if "duplicate column name" not in str(exc).lower():
                    raise
        return conn

    def append(self, encounter_id: str, kind: str, record: dict) -> str:
        try:
            conn = self._connect()
            try:
                # Serialize the read-previous + append so two concurrent writers cannot both
                # observe the same predecessor and fork the per-encounter hash chain. The
                # RESERVED lock is taken at BEGIN IMMEDIATE (before the SELECT), and other
                # writers block up to busy_timeout instead of racing. (Codex F6-R4.) The same
                # lock serializes the witness append, so the witness chain cannot fork either.
                conn.execute("BEGIN IMMEDIATE")
                # Fail CLOSED before extending a chain whose external witness was destroyed
                # or rewritten: a writer must never quietly continue on top of a broken
                # anchor. O(1) -- it compares the DB's terminal row against the witness tail.
                in_doubt_over = self._assert_witness_extendable(conn)
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
                # ORDERING IS THE ATOMICITY CONTRACT (Codex F6-R4-A): the witness record is
                # written and fsynced BEFORE the row is committed. Therefore
                #   - witness write fails  -> rollback; NO committed row and no witness: clean;
                #   - commit fails/crashes -> the witness is ahead by exactly one record, an
                #     explicitly recoverable IN-DOUBT tail that `_assert_witness_extendable`
                #     tolerates once and `verify_chain` reports by name.
                # The previous order (commit, then witness) could leave a committed row with
                # no witness at all -- indistinguishable from a truncated witness.
                witness_sha = self._append_head_witness(
                    encounter_id, entry["record_sha256"], in_doubt_over=in_doubt_over)
                conn.execute(
                    "INSERT INTO audit_log(encounter_id, kind, recorded_at, control_mode,"
                    " previous_record_sha256, record_json, record_sha256, witness_sha256)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (encounter_id, kind, entry["recorded_at"], entry["control_mode"],
                     previous, json.dumps(record, sort_keys=True, default=str),
                     entry["record_sha256"], witness_sha))
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


    # ------------------------------------------------------- external terminal-head witness
    # The witness is a SEALED, hash-chained, append-only journal kept outside the audit log.
    # Every entry is an HMAC-SHA256 over its canonical payload AND its predecessor's seal, so
    # a line cannot be edited, reordered, removed from the middle, or forged without the key.
    # The audit row stores the seal of the witness entry that sealed it, so the anchoring runs
    # in BOTH directions. Missing / empty / corrupt / duplicated / partial witness state is an
    # INTEGRITY FAILURE whenever durable rows exist -- never an empty-success `{}`.
    #
    # TRUST BOUNDARY: the HMAC key is read from the PROVENANCE_WITNESS_KEY environment
    # variable when set, which puts key custody OUTSIDE the filesystem the audit writer owns
    # -- a process that can reach the database and the witness file still cannot forge or
    # shorten the journal without the key. When no key is provisioned, one is generated once
    # into a 0600 sidecar so the seal is still tamper-evident against every writer that does
    # not hold it; that weaker default is recorded, not silently assumed, by `witness_status`.
    _WITNESS_VERSION = "audit-head-witness-v2"

    def _witness_path(self):
        return Path(str(self.db_path) + ".heads")

    def _witness_key_path(self):
        return Path(str(self.db_path) + ".witnesskey")

    def _witness_key(self, *, create: bool = False) -> bytes:
        """The sealing key: externally provisioned when available, else a 0600 sidecar.

        `create` is False on every READ path. Verification must never mint a key: doing so
        would let a store whose key was deleted be re-sealed under a fresh key, and would
        make a read-only integrity check write to disk. A missing key with entries present
        is an integrity failure, reported by `read_witness`.
        """
        env = os.getenv("PROVENANCE_WITNESS_KEY")
        if env and env.strip():
            return env.strip().encode("utf-8")
        kp = self._witness_key_path()
        if kp.exists():
            data = kp.read_bytes().strip()
            if not data:
                raise OSError(f"audit witness key at {kp} is empty")
            return data
        if not create:
            raise OSError(
                f"audit witness seal key is unavailable ({kp} is absent and "
                f"PROVENANCE_WITNESS_KEY is unset); witness entries cannot be verified")
        import secrets
        kp.parent.mkdir(parents=True, exist_ok=True)
        # PUBLISH ATOMICALLY. Creating the key file with O_EXCL and *then* writing it leaves a
        # window in which a racing reader sees a real but EMPTY key file and fails. So the key
        # is written to a private temp file, fsynced, and linked into place: `os.link` is
        # atomic and refuses to clobber, so the key file only ever appears complete, and a
        # racer that loses simply adopts the winner's key rather than overwriting a key that
        # may already have sealed entries. (Post-fix review.)
        key = secrets.token_hex(32).encode("ascii")
        tmp = kp.parent / f"{kp.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, key)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(str(tmp), str(kp))
        except FileExistsError:
            pass                       # another writer published first: adopt theirs
        finally:
            try:
                os.unlink(str(tmp))
            except OSError:
                pass
        data = kp.read_bytes().strip()
        if not data:
            raise OSError(f"audit witness key at {kp} is empty")
        return data

    def key_custody(self) -> str:
        """`external` when the seal key is provisioned outside this store's own directory."""
        env = os.getenv("PROVENANCE_WITNESS_KEY")
        return "external" if env and env.strip() else "adjacent-keyfile"

    def _seal(self, payload: dict, *, create_key: bool = False) -> str:
        import hmac
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hmac.new(self._witness_key(create=create_key), canonical.encode("utf-8"),
                        hashlib.sha256).hexdigest()

    def _witness_tail(self) -> dict | None:
        """The last witness entry, or None when the journal does not exist yet.

        Raises when the journal exists but its tail is unreadable/unsealed -- a corrupt tail
        must stop the writer, not be skipped."""
        wp = self._witness_path()
        if not wp.exists():
            return None
        lines = [ln for ln in wp.read_text().splitlines() if ln.strip()]
        if not lines:
            raise OSError(f"audit head witness {wp} exists but is empty")
        try:
            tail = json.loads(lines[-1])
        except Exception as exc:
            raise OSError(f"audit head witness {wp} has a corrupt final entry: {exc}") from exc
        if not isinstance(tail, dict) or self._seal(_witness_payload(tail)) != tail.get("seal"):
            raise OSError(f"audit head witness {wp} final entry fails its seal")
        return tail

    def _assert_witness_extendable(self, conn) -> str | None:
        """Refuse to extend a chain whose external witness no longer matches the log.

        Returns the seal of an observed IN-DOUBT tail (else None) so the next entry can record
        that it was written over one. Compares the log's GLOBAL terminal row against the
        witness tail, which is O(1) on both sides. Exactly two states are healthy:
          - aligned: the terminal row's witness seal IS the witness tail (normal), or
          - in-doubt by one: the tail's predecessor is the terminal row's seal, i.e. a write
            crashed between the witness fsync and the commit (documented, recoverable).
        Anything else means the witness was truncated, replaced, or rewritten, or that rows
        exist with no witness at all -- both fail closed.

        KNOWN, DELIBERATE LIMIT: at write time a crashed commit and a deleted terminal ROW are
        indistinguishable from O(1) local state, so exactly ONE in-doubt tail is tolerated (a
        crash must be recoverable). The anomaly is not lost: the divergence is permanent, the
        next entry records `in_doubt_over`, and `verify_chain` reports it by name for as long
        as the store exists.
        """
        row = conn.execute(
            "SELECT record_sha256, witness_sha256 FROM audit_log ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        tail = self._witness_tail()
        if row is None:
            if tail is not None:
                raise OSError(
                    "audit head witness records entries but the audit log is empty -- the "
                    "durable log was truncated or replaced")
            return None
        record_sha, witness_sha = row[0], row[1]
        if not witness_sha:
            # DEPLOY BOUNDARY. A store written before this control existed has rows with no
            # seal and no journal. That is a legacy prefix, not an attack, and bricking every
            # future append on it would be a self-inflicted outage -- so such a store may
            # ADOPT the witness starting now. But if a journal exists, or any row IS sealed,
            # then an unsealed terminal row means someone appended around the witness (or
            # destroyed it), which fails closed.
            sealed_exists = conn.execute(
                "SELECT 1 FROM audit_log WHERE witness_sha256 IS NOT NULL"
                " AND witness_sha256 <> '' LIMIT 1").fetchone()
            if tail is not None or sealed_exists:
                raise OSError(
                    "the terminal audit row carries no external witness seal although this "
                    "store is witnessed; refusing to extend a chain that was appended to "
                    "around the witness")
            return None       # legacy pre-witness store: the journal starts with this append
        if tail is None:
            raise OSError("audit rows exist but the external head witness is missing")
        if tail.get("seal") == witness_sha:
            return None
        if tail.get("previous_seal") == witness_sha and tail.get("record_sha256") != record_sha:
            return str(tail.get("seal"))    # in-doubt tail: witness ahead by exactly one entry
        raise OSError(
            "external head witness does not match the audit log's terminal record "
            "(witness truncated, rewritten, or out of order)")

    def _append_head_witness(self, encounter_id: str, record_sha256: str,
                             in_doubt_over: str | None = None) -> str:
        """Seal and durably append the new terminal head, returning its seal.

        Written BEFORE the row is committed and fsynced, so a witness failure aborts the
        whole append instead of leaving a committed row nothing vouches for. When it is
        written over an in-doubt tail, that fact is SEALED INTO the entry, so the anomaly is
        permanently recorded in the journal instead of only inferable from a divergence."""
        tail = self._witness_tail()
        entry = {
            "version": self._WITNESS_VERSION,
            "encounter_id": str(encounter_id),
            "record_sha256": str(record_sha256),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "previous_seal": str(tail.get("seal")) if tail else "",
        }
        if in_doubt_over:
            entry["in_doubt_over"] = str(in_doubt_over)
        entry["seal"] = self._seal(entry, create_key=True)
        wp = self._witness_path()
        wp.parent.mkdir(parents=True, exist_ok=True)
        with open(wp, "a") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return entry["seal"]

    def read_witness(self) -> tuple[list[dict], list[str]]:
        """(entries, problems) for the whole journal. A corrupt, unsealed, mis-chained or
        duplicated entry is REPORTED, never skipped."""
        problems: list[str] = []
        wp = self._witness_path()
        if not wp.exists():
            return [], ["external head witness file is missing"]
        try:
            raw = wp.read_text()
        except Exception as exc:
            return [], [f"external head witness is unreadable: {exc}"]
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return [], ["external head witness is empty"]
        entries: list[dict] = []
        previous_seal = ""
        seen_seals: set[str] = set()
        for i, line in enumerate(lines):
            try:
                w = json.loads(line)
                if not isinstance(w, dict):
                    raise ValueError("not an object")
            except Exception as exc:
                problems.append(f"witness entry {i}: corrupt/unparseable ({exc})")
                previous_seal = None          # the chain can no longer be followed
                continue
            try:
                expected = self._seal(_witness_payload(w))
            except Exception as exc:
                problems.append(f"witness entry {i}: seal cannot be verified ({exc})")
                entries.append(w)
                previous_seal = w.get("seal")
                continue
            if expected != w.get("seal"):
                problems.append(f"witness entry {i}: seal mismatch (forged or edited)")
            if previous_seal is not None and w.get("previous_seal", "") != previous_seal:
                problems.append(f"witness entry {i}: broken witness chain link")
            if w.get("seal") in seen_seals:
                problems.append(f"witness entry {i}: duplicate witness seal")
            seen_seals.add(w.get("seal"))
            previous_seal = w.get("seal")
            entries.append(w)
        return entries, problems

    def witness_status(self) -> dict:
        """Observability for the witness control: version, custody, entry count, problems."""
        entries, problems = self.read_witness()
        return {"version": self._WITNESS_VERSION, "key_custody": self.key_custody(),
                "entries": len(entries), "problems": problems,
                "path": str(self._witness_path())}

    def verify_chain(self, encounter_id: str | None = None) -> list[str]:
        """Return a list of integrity problems (empty => intact). Detects hash tampering (a
        stored record_sha256 that does not equal the recomputed hash of its fields), broken
        or forked links (previous_record_sha256 != the prior record\'s hash within an
        encounter), and reordering/gaps (seq not strictly increasing). (Codex F6-R4.)"""
        conn = self._connect()
        try:
            q = ("SELECT seq, encounter_id, kind, recorded_at, control_mode,"
                 " previous_record_sha256, record_json, record_sha256, witness_sha256"
                 " FROM audit_log")
            args: tuple = ()
            if encounter_id is not None:
                q += " WHERE encounter_id=?"
                args = (str(encounter_id),)
            q += " ORDER BY seq"
            rows = conn.execute(q, args).fetchall()
            any_rows = bool(conn.execute(
                "SELECT 1 FROM audit_log LIMIT 1").fetchone())
        finally:
            conn.close()
        problems: list[str] = []
        last_sha: dict[str, str] = {}
        present: dict[str, list] = {}
        row_seals: dict[str, tuple] = {}
        last_seq = -1
        for (seq, enc, kind, at, mode, prev, rj, sha, wsha) in rows:
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
            present.setdefault(enc, []).append((seq, sha))
            row_seals[sha] = (seq, enc, wsha)
        # ------------------------------------------------- external-witness cross-check
        # Bidirectional and FAIL-CLOSED (Codex F6-R4-A). A missing, empty, corrupt, unsealed,
        # duplicated or partial witness is an integrity failure whenever durable rows exist —
        # the previous implementation returned {} for an absent witness, so deleting the
        # sidecar reopened the very tail-truncation blind spot it was added to close.
        entries, witness_problems = self.read_witness()
        if not any_rows and not entries:
            return problems                     # nothing durable yet: nothing to witness
        problems.extend(witness_problems)
        if any_rows and not entries:
            problems.append("durable audit rows exist but the external head witness does not")
            return problems
        scoped = [w for w in entries
                  if encounter_id is None or str(w.get("encounter_id")) == str(encounter_id)]
        witnessed = {str(w.get("record_sha256")): w for w in scoped}
        if len(witnessed) != len(scoped):
            problems.append("external head witness contains duplicate record entries")
        # every durable row must be sealed, and its seal must still be IN the witness
        for sha, (seq, enc, wsha) in row_seals.items():
            if not wsha:
                problems.append(f"seq {seq} enc {enc}: audit row carries no witness seal")
                continue
            w = witnessed.get(sha)
            if w is None:
                problems.append(
                    f"seq {seq} enc {enc}: witness entry for this record is absent "
                    f"(witness truncated or rewritten)")
            elif w.get("seal") != wsha:
                problems.append(
                    f"seq {seq} enc {enc}: witness seal does not match the audit row")
        # every witnessed record must still exist in the log; the single trailing in-doubt
        # entry (witness fsynced, commit lost) is named as such rather than hidden
        tail_sha = str(scoped[-1].get("record_sha256")) if scoped else None
        for w in scoped:
            if w.get("in_doubt_over"):
                problems.append(
                    f"enc {w.get('encounter_id')}: this entry was written over an IN-DOUBT "
                    f"witness tail ({str(w['in_doubt_over'])[:16]}…) -- a commit was lost or "
                    f"a terminal record was removed at that point")
        for w in scoped:
            sha = str(w.get("record_sha256"))
            if sha in row_seals:
                continue
            enc = w.get("encounter_id")
            if sha == tail_sha:
                problems.append(
                    f"enc {enc}: witnessed terminal record is absent from the audit log "
                    f"(in-doubt write, or terminal-record truncation)")
            else:
                problems.append(
                    f"enc {enc}: witnessed audit record is missing from the log "
                    f"(tail/interior truncation)")
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
