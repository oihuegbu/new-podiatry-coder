"""Terminal-head CHECKPOINT ANCHOR — the seam for a trust boundary outside the
mutable durable-store + witness-journal pair.

WHY THIS EXISTS
---------------
The audit store is hash-chained and its head is sealed into an HMAC witness journal.
That pair detects edits, reordering and ONE-SIDED truncation. It cannot detect a
CONSISTENT truncation: remove the terminal journal entry AND the terminal durable row
together and everything that is left is internally consistent — the remaining prefix
verifies, and the legitimate writer (which holds the seal key) happily extends the
shortened chain. No key material is needed to do this, so a stronger key does not fix
it. The only thing that can is an expected terminal position recorded OUTSIDE both
mutable objects, in a store the audit writer cannot rewrite.

WHAT THIS MODULE PROVIDES
-------------------------
`CheckpointAnchor` — a tiny, pluggable interface holding, per store, the last known
good (monotonic sequence, seal) of the witness journal. The repository writes the
checkpoint after fsyncing a journal entry and before committing the row, and verifies
it before extending the chain, so a shortened journal is caught on the RELEASE PATH
(the append raises → the pipeline routes SYSTEM_HOLD) rather than in an audit report.

TRUST-BOUNDARY HONESTY
----------------------
An anchor is only as strong as the store behind it, so every backend declares
`external`. It is True ONLY for a backend whose write path is separately controlled —
an append-only/object-locked/WORM store the audit writer can add to but cannot
overwrite, truncate or delete.

  * `DisabledCheckpointAnchor` (the default): no anchor at all. Consistent truncation
    is NOT detectable. Reported, never silently assumed.
  * `LocalFileCheckpointAnchor`: a REFERENCE backend. It implements the full contract
    (monotonicity, rollback refusal, unavailability) and is what the regressions drive,
    but it lives on the same filesystem, under the same identity, as the store it
    anchors — anyone who can truncate the journal can also roll it back. It is
    `external = False` and MUST NOT be represented as a real trust boundary.

Provisioning a genuinely external backend is an infrastructure decision (a bucket with
object-lock/versioning plus a write-but-not-overwrite grant, or an equivalent
separately-administered append-only service), deliberately NOT taken here. Selecting
one is a one-line configuration change plus a backend class implementing this
interface; the enforcement logic above it does not change.

Agnostic: pure storage/sequence mechanics; no medical code, term, or scenario.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

CHECKPOINT_VERSION = "terminal-head-checkpoint-v1"

#: Environment variable naming the backend. Unset/none -> no anchor (reported).
ANCHOR_ENV = "PROVENANCE_CHECKPOINT_ANCHOR"
#: When truthy, a store with NO configured anchor fails closed instead of running
#: self-anchored. This is the switch that turns the control on once a real backend
#: exists, without a code change.
REQUIRED_ENV = "PROVENANCE_CHECKPOINT_REQUIRED"
#: One-time, explicit operator consent to anchor a journal that predates the anchor.
#: Without it, "the anchor knows nothing about a non-empty journal" fails closed,
#: because that is exactly what deleting the anchored checkpoint looks like.
ADOPT_ENV = "PROVENANCE_CHECKPOINT_ADOPT"


class CheckpointError(RuntimeError):
    """Base class: any terminal-head checkpoint failure. Always fail-closed."""


class CheckpointConfigError(CheckpointError):
    """The anchor configuration is unusable. Raised rather than defaulting to 'off':
    a typo in the backend spec must not silently disable an integrity control."""


class AnchorUnavailable(CheckpointError):
    """The configured anchor could not be read or written. UNVERIFIABLE is a failure,
    never an implicit pass."""


class AnchorFormatError(CheckpointError):
    """The stored checkpoint is not a well-formed record."""


class AnchorRollback(CheckpointError):
    """A write would move the anchor backwards, or fork it at the same sequence."""


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Checkpoint:
    """The last known good terminal head of one store's witness journal.

    `seq` is a strictly increasing per-store counter (journal entries are numbered from
    1). `seal` is that entry's HMAC seal. Together they say "this store's journal had at
    least `seq` entries, and entry `seq` was exactly this one" — the statement a
    consistently truncated pair cannot satisfy.

    `signature` is written and checked by the caller that owns the sealing key, so a
    third party who can write the anchor store still cannot fabricate a checkpoint.
    """

    store_id: str
    seq: int
    seal: str
    record_sha256: str
    witness_version: str
    written_at: str
    signature: str = ""

    def payload(self) -> dict:
        """The signed portion — everything except the signature itself."""
        return {
            "version": CHECKPOINT_VERSION,
            "store_id": self.store_id,
            "seq": int(self.seq),
            "seal": self.seal,
            "record_sha256": self.record_sha256,
            "witness_version": self.witness_version,
            "written_at": self.written_at,
        }

    def as_record(self) -> dict:
        record = self.payload()
        record["signature"] = self.signature
        return record

    @classmethod
    def from_record(cls, record) -> "Checkpoint":
        """Parse a stored record, validating every field. A malformed checkpoint is an
        error, never a 'no checkpoint' (which would read as an empty pass)."""
        if not isinstance(record, dict):
            raise AnchorFormatError("checkpoint record must be a JSON object")
        version = record.get("version")
        if version != CHECKPOINT_VERSION:
            raise AnchorFormatError(
                f"unsupported checkpoint version {version!r} (expected {CHECKPOINT_VERSION})")
        seq = record.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise AnchorFormatError("checkpoint seq must be an integer >= 1")
        for field in ("store_id", "seal", "record_sha256", "witness_version",
                      "written_at", "signature"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise AnchorFormatError(f"checkpoint field {field!r} must be a non-empty string")
        return cls(store_id=record["store_id"], seq=seq, seal=record["seal"],
                   record_sha256=record["record_sha256"],
                   witness_version=record["witness_version"],
                   written_at=record["written_at"], signature=record["signature"])


class CheckpointAnchor:
    """Store the last known good terminal head for a store id, outside that store."""

    backend_id = "abstract"
    #: Is the anchor configured at all?
    configured = False
    #: Is the backing store a genuine, separately controlled trust boundary? Only a
    #: backend the audit writer can APPEND to but cannot overwrite/delete may say True.
    external = False

    def read(self, store_id: str) -> dict | None:  # pragma: no cover - interface
        """The stored record, or None when this store has never been anchored.
        Raises `AnchorUnavailable` when the answer is unknown."""
        raise NotImplementedError

    def write(self, record: dict) -> None:  # pragma: no cover - interface
        """Persist a checkpoint durably. Raises `AnchorRollback` if it would move the
        anchor backwards, `AnchorUnavailable` if it cannot be persisted."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"backend": self.backend_id, "configured": bool(self.configured),
                "external_trust_boundary": bool(self.external)}


class DisabledCheckpointAnchor(CheckpointAnchor):
    """No anchor. The witness journal remains SELF-ANCHORED: a consistent truncation of
    the durable tail and the journal tail together cannot be detected. This is the
    default because no external store is provisioned; it is reported by
    `describe()` so no artifact can imply a guarantee that is not in force."""

    backend_id = "disabled"
    configured = False
    external = False

    def read(self, store_id: str) -> dict | None:
        return None

    def write(self, record: dict) -> None:
        return None

    def describe(self) -> dict:
        info = super().describe()
        info["limitation"] = (
            "no terminal-head checkpoint anchor is configured; the witness journal is "
            "self-anchored, so a consistent durable-tail + journal-tail truncation is "
            "NOT detectable")
        return info


class LocalFileCheckpointAnchor(CheckpointAnchor):
    """REFERENCE BACKEND — NOT a real trust boundary.

    One JSON object per store id under `root`, written atomically and fsynced, with
    monotonicity enforced on write. It implements the complete contract so the
    enforcement path can be exercised and regression-tested end to end, and so a real
    backend is a drop-in replacement.

    It is `external = False` on purpose: it sits on the same filesystem, under the same
    identity, as the store it anchors, so a writer able to truncate the journal is also
    able to roll it back. Point `root` at separately controlled storage and it is still
    only as external as that storage's ACLs make it — which is why the honest fix is a
    backend whose store refuses overwrites, not merely a different path.
    """

    backend_id = "local-file"
    configured = True
    external = False

    def __init__(self, root) -> None:
        self.root = Path(root)

    def _path(self, store_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(store_id))
        if not safe:
            raise CheckpointConfigError("store id resolves to an empty checkpoint name")
        return self.root / f"{safe}.checkpoint.json"

    def read(self, store_id: str) -> dict | None:
        path = self._path(store_id)
        try:
            if not path.exists():
                return None
            raw = path.read_text()
        except OSError as exc:
            raise AnchorUnavailable(f"checkpoint anchor unreadable at {path}: {exc}") from exc
        try:
            record = json.loads(raw)
        except Exception as exc:
            raise AnchorFormatError(f"checkpoint at {path} is not valid JSON: {exc}") from exc
        Checkpoint.from_record(record)          # validate eagerly; malformed != absent
        return record

    def write(self, record: dict) -> None:
        incoming = Checkpoint.from_record(record)
        existing = self.read(incoming.store_id)
        if existing is not None:
            current = Checkpoint.from_record(existing)
            if incoming.seq < current.seq:
                raise AnchorRollback(
                    f"refusing to move the checkpoint anchor backwards "
                    f"({current.seq} -> {incoming.seq})")
            if incoming.seq == current.seq and incoming.seal != current.seal:
                raise AnchorRollback(
                    f"refusing to fork the checkpoint anchor at seq {current.seq}")
        path = self._path(incoming.store_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".ckpt-")
            try:
                with os.fdopen(fd, "w") as fh:
                    fh.write(json.dumps(record, sort_keys=True))
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, str(path))
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError as exc:
            raise AnchorUnavailable(f"checkpoint anchor unwritable at {path}: {exc}") from exc

    def describe(self) -> dict:
        info = super().describe()
        info["root"] = str(self.root)
        info["limitation"] = (
            "reference backend: same filesystem and identity as the store it anchors, "
            "so it is tamper-EVIDENT against an unprivileged writer but is not a "
            "separately controlled append-only store")
        return info


def resolve_checkpoint_anchor(spec: str | None = None) -> CheckpointAnchor:
    """Build the configured anchor. Fail-closed on an unrecognised spec.

    Accepted:
      * unset / "" / "none" / "disabled" / "off" -> `DisabledCheckpointAnchor`
      * "file:<directory>"                       -> `LocalFileCheckpointAnchor`

    Anything else raises, INCLUDING remote-store schemes: naming a backend this build
    cannot honour must stop the writer, not read as 'no anchor configured'.
    """
    raw = os.getenv(ANCHOR_ENV, "") if spec is None else spec
    text = str(raw or "").strip()
    if not text or text.lower() in ("none", "disabled", "off"):
        return DisabledCheckpointAnchor()
    if text.lower().startswith("file:"):
        root = text[len("file:"):].strip()
        if not root:
            raise CheckpointConfigError(f"{ANCHOR_ENV}='file:' names no directory")
        return LocalFileCheckpointAnchor(root)
    raise CheckpointConfigError(
        f"{ANCHOR_ENV}={text!r} names no backend implemented by this build "
        f"(supported: 'none', 'file:<directory>'). A separately controlled append-only "
        f"backend has not been provisioned; refusing to run as if an anchor existed")


def checkpoint_required() -> bool:
    """True when running WITHOUT a configured anchor must fail closed."""
    return _truthy(os.getenv(REQUIRED_ENV))


def checkpoint_adoption_allowed() -> bool:
    """True when a journal that predates the anchor may be adopted (operator consent)."""
    return _truthy(os.getenv(ADOPT_ENV))
