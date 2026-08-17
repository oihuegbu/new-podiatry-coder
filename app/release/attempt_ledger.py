"""Append-only processing-attempt ledger — output attempts are atomic and superseding.

WHY THIS EXISTS — issue #6 F6-R6-B, product directive section 7
================================================================================
`run.py` used to deliver each note's result with a direct JSON write::

    with open(OUTPUT_DIR / f"{stem}_results.json", "w") as fh:   # truncate + write
        json.dump(payload, fh)

Open-for-write truncates FIRST. A new attempt that dies between the truncate and
the final byte therefore leaves either a half-written file or — when the failure
happens before the open, as a disk-full/permission/serialization failure does —
the PREVIOUS attempt's file, intact and unmarked. That older file may carry a
releasable claim, and every downstream consumer (`tools/claims_registry.py`
ingest, `tools/claim_submitter.py`, the `all_results.json` aggregate) read it as
"the current result for this note". A failed re-run silently kept an older
release eligible for submission.

The fix is not "write more carefully". It is that a result on disk is no longer
self-authorizing: **an artifact is consumable only while it is the output of the
CURRENT attempt, and every new attempt supersedes its predecessor before it can
fail**. That ordering is the whole property. Superseding at the START of an
attempt is what makes an output failure fail-closed: by the time anything can go
wrong, the prior success is already not current.

THE LEDGER
================================================================================
Under the results directory::

    results/
      attempts/                          <- the directory's GOVERNANCE marker
        <document_id>/
          ledger.jsonl                   append-only attempt history (one JSON
                                         object per state transition, fsynced)
          current.json                   THE POINTER — atomically replaced; the
                                         single race-free answer to "what is the
                                         current attempt for this encounter?"
          <attempt_id>.json              that attempt's artifact, retained
      <document_id>_results.json         the CURRENT attempt's artifact, published
      all_results.json                   aggregate of current COMPLETED attempts

States (`AttemptState`):

    IN_PROGRESS   an attempt is open. Nothing is consumable for this encounter.
    COMPLETED     an artifact was durably written AND published for this exact
                  document version. The ONLY consumable state.
    SYSTEM_RETRY  the encounter could not be processed (a dependency failed).
                  A tombstone WITH an artifact: the failure bundle is published
                  so the note is visible, but it is never served as a current
                  result — this is system work, not a coding conclusion.
    FAILED        the attempt could not durably produce an artifact at all (the
                  output failure this module exists for). A tombstone: the
                  ledger says so out loud instead of leaving silence, and the
                  published path is overwritten with an explicitly non-releasable
                  bundle (or, if even that cannot be written, removed).

WHAT THE POINTER DOES AND DOES NOT GOVERN
================================================================================
It governs SUPERSESSION — which attempt is current, and whether that attempt
durably completed for the document version it was opened against. It deliberately
does NOT compare the published artifact's BYTES against the retained attempt
copy:

  * content integrity is already owned, end to end, by the claim fingerprint and
    the release certificate (`app/contracts/claim_bundle.py`), which every
    consumer verifies and which detect an edited claim by name;
  * a coder's correction and the adjudicator legitimately rewrite a published
    artifact in place. A byte comparison here would refuse those for
    "supersession" — a false, unrelated reason — while adding nothing the
    certificate does not already catch.

What it DOES check about content is the one thing supersession is about: the
published artifact must declare the SAME document version the current attempt was
opened for. That is what stops an old-document-version success from being served
as current after the document was revised.

MIGRATION / GOVERNANCE SCOPE
================================================================================
A results directory is *governed* once `attempts/` exists in it — which `run.py`
creates on its first note. Consumers pointed at a governed directory resolve
every current result through the pointer, and an artifact with no current
COMPLETED attempt is refused BY NAME. A directory with no `attempts/` predates
this ledger (or belongs to a tool that materializes result files itself, e.g.
`claims_registry export-gold`); it is read exactly as before. The marker is the
directory, not the document, so removing ONE document's ledger cannot quietly
return that document to ungoverned reads.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from app.core.config import (
    AGGREGATE_RESULTS, ATTEMPT_DIRNAME, ATTEMPT_HISTORY, ATTEMPT_POINTER,
    RESULT_SUFFIX)

#: The run-output layout, from the module that declares every path this deployment
#: composes (`app/core/config.py`). Named here for readability, never spelled here:
#: this module, `run.py`, the claims registry and the 837P submitter have to address
#: exactly the same files, and a filename convention copied into four modules is how
#: two of them come to address different ones.
#:
#: `LEDGER_DIRNAME` is also the GOVERNANCE MARKER: its existence in a results
#: directory is what makes that directory ledger-governed.
LEDGER_DIRNAME = ATTEMPT_DIRNAME
POINTER_NAME = ATTEMPT_POINTER
HISTORY_NAME = ATTEMPT_HISTORY
ARTIFACT_SUFFIX = RESULT_SUFFIX
AGGREGATE_NAME = AGGREGATE_RESULTS

POINTER_SCHEMA = "attempt_pointer/1"
RECORD_SCHEMA = "attempt_record/1"


class AttemptState(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    SYSTEM_RETRY = "SYSTEM_RETRY"
    FAILED = "FAILED"


#: The only state whose artifact may be served as the current result. Stated as a
#: set rather than an `is COMPLETED` test so a future state cannot become
#: consumable by accident — it has to be added here, in a reviewed change.
CONSUMABLE_STATES = frozenset({AttemptState.COMPLETED})


class AttemptLedgerError(RuntimeError):
    """The ledger could not do what was asked. Never returned as a value."""


class AttemptWriteError(AttemptLedgerError):
    """A ledger record, pointer or artifact could not be durably written."""


class AttemptSuperseded(AttemptLedgerError):
    """A newer attempt owns this encounter's pointer; this one may not publish."""


# --------------------------------------------------------------------------
# durable primitives
# --------------------------------------------------------------------------

def _fsync_dir(directory: Path) -> None:
    """Make a rename/creation itself durable, not just the file's contents.

    `os.replace` is atomic with respect to readers, but the DIRECTORY ENTRY it
    changes is only guaranteed to survive a crash after the directory is
    fsynced. Without this the pointer can be observed to move back after a power
    loss — which is exactly the stale-result resurrection this module forbids.
    """
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: Path, payload, *, indent: int | None = None) -> None:
    """Temp file in the SAME directory, flush, fsync, `os.replace`, fsync dir.

    Same pattern as `claude_coder/checkpoint.py`'s anchor writer, for the same
    reason: a reader either sees the whole previous file or the whole new one,
    and never a truncated prefix of either. The temp file is removed on any
    failure so a crashed attempt cannot leave partial JSON behind that a later
    glob might pick up.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=indent, default=str, sort_keys=indent is None)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".attempt-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


def document_version_of(path: Path) -> str:
    """The immutable identity of a SOURCE document, by the project's convention.

    Deliberately delegated to the source-evidence compiler's own digest function
    rather than recomputed with a local convention: the value recorded here is
    compared, at completion, against the `document_version` the compiled bundle
    declares, and two independent spellings of "sha256 of the bytes" would make
    that comparison meaningless the first time either changed.
    """
    from app.ingestion.source_evidence import document_digest
    return document_digest(Path(path).read_bytes())


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_dirname(document_id: str) -> str:
    """A document id is a filename stem here, but it arrives from a caller.

    Sanitized (and salted with a digest of the original) so no id can escape the
    ledger root or collide with another after sanitization.
    """
    text = str(document_id or "")
    if not text:
        raise AttemptLedgerError("an attempt needs a non-empty document id")
    safe = _UNSAFE.sub("_", text)[:120]
    if safe == text:
        return safe
    return f"{safe}.{hashlib.sha256(text.encode()).hexdigest()[:12]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# values
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Attempt:
    """One opened processing attempt. Returned by `begin`, consumed by the
    terminal transitions — a caller cannot invent one."""

    document_id: str
    document_version: str
    attempt_id: str
    sequence: int
    opened_at: str


@dataclass(frozen=True)
class CurrentResult:
    """The current, consumable artifact for one encounter."""

    document_id: str
    path: Path
    attempt_id: str
    document_version: str


# --------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------

class AttemptLedger:
    """The append-only attempt store for one results directory.

    Single-writer by design: one batch process owns a results directory at a
    time. Concurrency is nonetheless not left to chance — every terminal
    transition verifies it still OWNS the pointer before moving it, so a second
    process cannot resurrect an older attempt over a newer one. It refuses
    (`AttemptSuperseded`) instead.
    """

    def __init__(self, results_dir) -> None:
        self.results_dir = Path(results_dir)
        self.root = self.results_dir / LEDGER_DIRNAME

    # ------------------------------------------------------------- locations
    def _doc_root(self, document_id: str) -> Path:
        return self.root / _safe_dirname(document_id)

    def _pointer_path(self, document_id: str) -> Path:
        return self._doc_root(document_id) / POINTER_NAME

    def _history_path(self, document_id: str) -> Path:
        return self._doc_root(document_id) / HISTORY_NAME

    def published_path(self, document_id: str) -> Path:
        """Where the CURRENT artifact for this encounter is published.

        Unchanged from the pre-ledger naming: every existing consumer, tool and
        operator habit addresses a note's result by this path, and moving it
        would have been a second, unrelated migration bundled into a durability
        fix.
        """
        return self.results_dir / f"{document_id}{ARTIFACT_SUFFIX}"

    def governed(self) -> bool:
        return self.root.is_dir()

    # -------------------------------------------------------------- history
    def history(self, document_id: str) -> list[dict]:
        """Every recorded transition for this encounter, oldest first.

        A line that will not parse is FATAL rather than skipped. Torn JSONL is
        only producible by a crash mid-append, which by construction happens
        inside an attempt that never reached a terminal state — so refusing is
        free in practice, and the alternative (skip and carry on) is the silent
        degradation this repository treats as a defect class.
        """
        path = self._history_path(document_id)
        if not path.exists():
            return []
        try:
            raw = path.read_text()
        except OSError as exc:
            raise AttemptLedgerError(
                f"attempt history unreadable for {document_id}: {exc}") from exc
        records = []
        for number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise AttemptLedgerError(
                    f"attempt history for {document_id} is damaged at line "
                    f"{number} ({exc}); no result for this encounter can be "
                    f"served until it is reconciled") from exc
            if not isinstance(record, dict):
                raise AttemptLedgerError(
                    f"attempt history for {document_id} line {number} is not a "
                    f"JSON object")
            records.append(record)
        return records

    def _append_history(self, document_id: str, record: dict) -> None:
        path = self._history_path(document_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as fh:
                fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            _fsync_dir(path.parent)
        except OSError as exc:
            raise AttemptWriteError(
                f"attempt history for {document_id} could not be appended: "
                f"{exc}") from exc

    # -------------------------------------------------------------- pointer
    def pointer(self, document_id: str) -> dict | None:
        """The current-attempt pointer, or None when this encounter has none.

        One `read_text` of one atomically replaced file: the read is race-free
        against a concurrent write by construction, which is the property the
        directive asks for ("an atomic current-attempt pointer").
        """
        path = self._pointer_path(document_id)
        try:
            raw = path.read_text()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise AttemptLedgerError(
                f"current-attempt pointer unreadable for {document_id}: "
                f"{exc}") from exc
        try:
            record = json.loads(raw)
        except ValueError as exc:
            raise AttemptLedgerError(
                f"current-attempt pointer for {document_id} is not valid JSON "
                f"({exc})") from exc
        declared = record.get("schema") if isinstance(record, dict) else None
        if declared != POINTER_SCHEMA:
            raise AttemptLedgerError(
                f"current-attempt pointer for {document_id} declares schema "
                f"{declared!r}, not {POINTER_SCHEMA!r}")
        if record.get("state") not in {s.value for s in AttemptState}:
            raise AttemptLedgerError(
                f"current-attempt pointer for {document_id} declares unknown "
                f"state {record.get('state')!r}")
        return record

    def _write_pointer(self, document_id: str, record: dict) -> None:
        try:
            atomic_write_json(self._pointer_path(document_id), record)
        except OSError as exc:
            raise AttemptWriteError(
                f"current-attempt pointer for {document_id} could not be "
                f"written: {exc}") from exc

    def _invalidate(self, document_id: str) -> None:
        """Last resort when NOTHING can be written: make the encounter unservable.

        Removing an entry needs no free space, so it is the one action still
        available on a full filesystem. It is not a nicer failure — it destroys
        the published copy of a prior result — but the prior result is retained
        in the attempt store either way, and leaving a releasable artifact
        addressable while its supersession could not be recorded is the exact
        defect this module exists to remove.
        """
        problems = []
        for path in (self._pointer_path(document_id),
                     self.published_path(document_id)):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                problems.append(f"{path.name}: {exc}")
        if problems:
            raise AttemptWriteError(
                f"{document_id}: an attempt could neither be recorded nor "
                f"invalidated, so an older result may still be addressable — "
                f"{'; '.join(problems)}")

    # ----------------------------------------------------------- transitions
    def begin(self, document_id: str, document_version: str) -> Attempt:
        """Open an attempt — and SUPERSEDE the previous result, before any work.

        Ordering is the point. The pointer moves to IN_PROGRESS first, so from
        this instant no consumer can be served this encounter's older artifact,
        whatever happens next: the process can die, the disk can fill, the
        coder can raise. There is no window in which a new attempt exists and
        an old success is still current.
        """
        records = self.history(document_id)
        sequence = max((int(r.get("sequence") or 0) for r in records), default=0) + 1
        attempt = Attempt(
            document_id=str(document_id),
            document_version=str(document_version or ""),
            attempt_id=f"{sequence:04d}-{uuid.uuid4().hex[:12]}",
            sequence=sequence,
            opened_at=_now(),
        )
        entry = self._record(attempt, AttemptState.IN_PROGRESS)
        try:
            self._append_history(document_id, entry)
            self._write_pointer(document_id, self._pointer_record(attempt,
                                                                 AttemptState.IN_PROGRESS))
        except AttemptLedgerError:
            # The supersession could not be recorded. The prior pointer/artifact
            # must not survive as current on the strength of that failure.
            self._invalidate(document_id)
            raise
        return attempt

    def complete(self, attempt: Attempt, payload: dict) -> Path:
        """Durably store and publish this attempt's artifact, then commit it.

        The commit is the pointer move, and it is LAST: every byte the consumer
        will read is already fsynced to disk before the pointer says the result
        exists. A crash anywhere earlier leaves the attempt IN_PROGRESS — not
        consumable, and explicitly not a success.
        """
        declared = _declared_document_version(payload)
        if attempt.document_version and declared != attempt.document_version:
            # Either the document changed underneath the attempt, or the result
            # declares no source document at all. Publishing the first would bind
            # a claim to a version this attempt never opened against; publishing
            # the second would produce an artifact `resolve_current` could never
            # serve, which is a silently unusable success. Both refuse here,
            # where the reason is still known.
            raise AttemptWriteError(
                f"{attempt.document_id}: attempt {attempt.attempt_id} was opened "
                f"for document version {attempt.document_version} but the result "
                f"declares {declared or 'none'}; the document changed while it "
                f"was being processed")
        return self._publish(attempt, payload, AttemptState.COMPLETED, error="")

    def system_retry(self, attempt: Attempt, payload: dict, *, error: str) -> Path:
        """The encounter could not be processed: publish the failure bundle as a
        tombstone WITH an artifact.

        The note stays visible in the results directory (a note that produced
        nothing on disk is invisible, which is its own silent failure), and the
        attempt is recorded SYSTEM_RETRY — dependency/system work, never a
        coding conclusion, and never served as a current result.
        """
        return self._publish(attempt, payload, AttemptState.SYSTEM_RETRY,
                             error=str(error or ""))

    def fail(self, attempt: Attempt, *, error: str,
             tombstone: dict | None = None) -> None:
        """The attempt could not durably produce an artifact — say so out loud.

        This is the F6-R6-B path. The published location is overwritten with an
        explicitly non-releasable tombstone bundle when one can still be written
        (so even a consumer that knows nothing about this ledger cannot read the
        superseded success), and removed when it cannot. Either way the pointer
        ends FAILED, and a FAILED pointer is not consumable.
        """
        entry = self._record(attempt, AttemptState.FAILED, error=str(error or ""))
        published: str | None = None
        if tombstone is not None:
            try:
                atomic_write_json(self.published_path(attempt.document_id),
                                  tombstone, indent=2)
                published = self.published_path(attempt.document_id).name
            except OSError:
                published = None
        stale: str = ""
        if published is None:
            try:
                self.published_path(attempt.document_id).unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                stale = str(exc)
        # The tombstone is recorded BEFORE the stale-artifact problem is raised.
        # Losing the FAILED record because the cleanup failed would leave the
        # encounter looking merely interrupted, which is the weaker and less
        # actionable of the two states.
        entry["artifact"] = published or ""
        entry["stale_artifact_problem"] = stale
        self._append_history(attempt.document_id, entry)
        if self._owns(attempt):
            self._write_pointer(attempt.document_id,
                                self._pointer_record(attempt, AttemptState.FAILED,
                                                     artifact=published or "",
                                                     error=str(error or "")))
        if stale:
            raise AttemptWriteError(
                f"{attempt.document_id}: the failed attempt is recorded, but its "
                f"stale published artifact could neither be replaced nor removed "
                f"({stale})")

    # ------------------------------------------------------------- internals
    def _owns(self, attempt: Attempt) -> bool:
        current = self.pointer(attempt.document_id)
        return bool(current) and current.get("attempt_id") == attempt.attempt_id

    def _publish(self, attempt: Attempt, payload: dict, state: AttemptState,
                 *, error: str) -> Path:
        if not self._owns(attempt):
            raise AttemptSuperseded(
                f"{attempt.document_id}: attempt {attempt.attempt_id} no longer "
                f"owns the current-attempt pointer; refusing to publish over a "
                f"newer attempt")
        retained = self._doc_root(attempt.document_id) / f"{attempt.attempt_id}.json"
        published = self.published_path(attempt.document_id)
        try:
            atomic_write_json(retained, payload, indent=2)
            atomic_write_json(published, payload, indent=2)
        except OSError as exc:
            raise AttemptWriteError(
                f"{attempt.document_id}: attempt {attempt.attempt_id} artifact "
                f"could not be written ({exc})") from exc
        entry = self._record(attempt, state, error=error)
        entry["artifact"] = published.name
        entry["retained_artifact"] = retained.name
        entry["artifact_sha256"] = _payload_digest(payload)
        self._append_history(attempt.document_id, entry)
        self._write_pointer(
            attempt.document_id,
            self._pointer_record(attempt, state, artifact=published.name,
                                 retained=retained.name, error=error,
                                 artifact_sha256=entry["artifact_sha256"]))
        return published

    def _record(self, attempt: Attempt, state: AttemptState, *,
                error: str = "") -> dict:
        return {
            "schema": RECORD_SCHEMA,
            "document_id": attempt.document_id,
            "document_version": attempt.document_version,
            "attempt_id": attempt.attempt_id,
            "sequence": attempt.sequence,
            "state": state.value,
            "opened_at": attempt.opened_at,
            "recorded_at": _now(),
            "error": error,
        }

    def _pointer_record(self, attempt: Attempt, state: AttemptState, *,
                        artifact: str = "", retained: str = "",
                        error: str = "", artifact_sha256: str = "") -> dict:
        return {
            "schema": POINTER_SCHEMA,
            "document_id": attempt.document_id,
            "document_version": attempt.document_version,
            "attempt_id": attempt.attempt_id,
            "sequence": attempt.sequence,
            "state": state.value,
            "artifact": artifact,
            "retained_artifact": retained,
            "artifact_sha256": artifact_sha256,
            "opened_at": attempt.opened_at,
            "recorded_at": _now(),
            "error": error,
        }

# --------------------------------------------------------------------------
# payload helpers
# --------------------------------------------------------------------------

def _declared_document_version(payload) -> str:
    """The source-document identity the artifact itself declares.

    Read off the canonical `ClaimBundle` shape only. An artifact that declares
    none returns "" and is handled by the caller — never defaulted to the
    pointer's value, which would make the comparison self-satisfying.
    """
    if not isinstance(payload, dict):
        return ""
    encounter = payload.get("encounter")
    if not isinstance(encounter, dict):
        return ""
    document = encounter.get("source_document")
    if not isinstance(document, dict):
        return ""
    return str(document.get("document_version") or "").strip()


def _payload_digest(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------
# what a consumer is allowed to read
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CurrentResults:
    """Everything a consumer of a results directory may read, and why not."""

    governed: bool
    results: tuple[CurrentResult, ...]
    #: document_id -> the exact reason its artifact is NOT current. Consumers
    #: surface these; a refusal that no one can see is a silent skip.
    refusals: dict[str, str]

    def paths(self) -> list[Path]:
        return [r.path for r in self.results]


def _document_ids(ledger: AttemptLedger) -> dict[str, Path]:
    """Every encounter with an attempt store, mapped to its ledger directory.

    The id comes from the RECORDS (pointer, else the first history entry), not
    from the directory name — the directory name is a sanitized derivative and
    is not reversible.
    """
    found: dict[str, Path] = {}
    for entry in sorted(ledger.root.iterdir()):
        if not entry.is_dir():
            continue
        document_id = ""
        try:
            raw = (entry / POINTER_NAME).read_text()
            document_id = str((json.loads(raw) or {}).get("document_id") or "")
        except (OSError, ValueError, AttributeError):
            document_id = ""
        if not document_id:
            try:
                first = (entry / HISTORY_NAME).read_text().splitlines()[0]
                document_id = str((json.loads(first) or {}).get("document_id") or "")
            except (OSError, ValueError, IndexError, AttributeError):
                document_id = ""
        found[document_id or entry.name] = entry
    return found


def _published_ids(results_dir: Path) -> set[str]:
    return {p.name[: -len(ARTIFACT_SUFFIX)]
            for p in sorted(results_dir.glob(f"*{ARTIFACT_SUFFIX}"))
            if p.name != AGGREGATE_NAME}


def resolve_current(results_dir) -> CurrentResults:
    """THE answer to "which artifacts in this directory may be consumed?".

    In a governed directory an artifact is current only when the encounter's
    pointer names a COMPLETED attempt, that attempt's artifact is where the
    pointer says it is, and the artifact declares the same document version the
    attempt was opened for. Everything else is refused BY NAME — an in-flight
    attempt, a failed one, a system-retry tombstone, a result file with no
    attempt record at all, and a completed attempt for a document version that
    has since been superseded.

    In an ungoverned directory (no `attempts/`) every `*_results.json` is
    returned, exactly as before this ledger existed.
    """
    ledger = AttemptLedger(results_dir)
    published = _published_ids(ledger.results_dir)
    if not ledger.governed():
        return CurrentResults(
            governed=False,
            results=tuple(CurrentResult(document_id=doc,
                                        path=ledger.published_path(doc),
                                        attempt_id="", document_version="")
                          for doc in sorted(published)),
            refusals={})

    results: list[CurrentResult] = []
    refusals: dict[str, str] = {}
    for document_id in sorted(set(_document_ids(ledger)) | published):
        ok, reason, current = _resolve_one(ledger, document_id)
        if ok and current is not None:
            results.append(current)
        else:
            refusals[document_id] = reason
    return CurrentResults(governed=True, results=tuple(results), refusals=refusals)


def _resolve_one(ledger: AttemptLedger,
                 document_id: str) -> tuple[bool, str, CurrentResult | None]:
    try:
        history = ledger.history(document_id)   # a damaged history is fatal, not skipped
        pointer = ledger.pointer(document_id)
    except AttemptLedgerError as exc:
        return False, str(exc), None
    if pointer is None and history:
        return False, ("the current-attempt pointer for this encounter is gone "
                       f"({len(history)} attempt record(s) remain), so no result "
                       "file here can be shown to be current; re-run the note"), None
    if pointer is None:
        return False, ("no processing attempt is recorded for this encounter, so "
                       "no result file here can be shown to be current; re-run "
                       "the note to open one"), None
    state = pointer.get("state")
    attempt_id = str(pointer.get("attempt_id") or "")
    if state not in {s.value for s in CONSUMABLE_STATES}:
        return False, (f"the current attempt ({attempt_id}) is {state}, not "
                       f"COMPLETED; any earlier result for this encounter has "
                       f"been superseded"), None
    path = ledger.published_path(document_id)
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return False, (f"attempt {attempt_id} completed but its published "
                       f"artifact {path.name} is missing"), None
    except (OSError, ValueError) as exc:
        return False, (f"the published artifact for attempt {attempt_id} could "
                       f"not be read ({exc})"), None
    expected = str(pointer.get("document_version") or "")
    declared = _declared_document_version(payload)
    if expected and declared != expected:
        return False, (f"{path.name} declares document version "
                       f"{declared or 'none'} but the current attempt "
                       f"({attempt_id}) is for {expected}; it is a result for a "
                       f"different version of this document"), None
    return True, "", CurrentResult(document_id=document_id, path=path,
                                   attempt_id=attempt_id,
                                   document_version=expected)


def consumable(results_dir, document_id: str) -> tuple[bool, str]:
    """May this one encounter's published artifact be consumed right now?

    The per-document form of `resolve_current`, for consumers that already know
    which note they are asking about (the 837P submitter reaches a result file
    from a registry event, not from a directory scan).
    """
    ledger = AttemptLedger(results_dir)
    if not ledger.governed():
        return True, ""
    ok, reason, _ = _resolve_one(ledger, document_id)
    return ok, reason
