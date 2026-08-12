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
  * `S3CheckpointAnchor`: the REAL backend. Checkpoints live in a separately
    administered object store the audit writer may add to but may not delete from or
    rewrite (see `terraform/s3_checkpoint.tf`). It is `external = True`; what earns
    that claim is documented on the class and proved by its regressions, which run
    against the live bucket rather than a mock.

HOW IT IS TURNED ON (checked-in configuration — no code change)
---------------------------------------------------------------
    PROVENANCE_CHECKPOINT_ANCHOR='s3://<bucket>/checkpoints?region=us-east-1'
    PROVENANCE_CHECKPOINT_REQUIRED=1     # an unanchored run then fails closed
    PROVENANCE_CHECKPOINT_ADOPT=1        # ONE reviewed run, only to adopt a journal that
                                         # predates the anchor; unset it again afterwards

The deployed runtime sets the first two from source, and deliberately from two DIFFERENT
places (Codex F6-R4-A finding A):

  * `PROVENANCE_CHECKPOINT_REQUIRED=1` is a constant in `docker-compose.yml`, so it cannot
    be lost by a stale or unrefreshed `.env`;
  * `PROVENANCE_CHECKPOINT_ANCHOR` is derived from the bucket resource in
    `terraform/secrets.tf` and delivered through the runtime secret, so it is never a
    hand-copied bucket name that a clean `terraform apply` would silently invalidate.

The split is the fail-safe direction: config drift can only leave the run REQUIRED with no
anchor, which holds the release. It can never leave it silently unanchored.

The bucket and its grants are `terraform/s3_checkpoint.tf`; the key prefix must be one the
role is granted, and credentials come from the instance role (nothing here reads any). A
spec this build cannot honour, or an anchor it cannot reach, REFUSES — it never degrades
to "no anchor configured".

Agnostic: pure storage/sequence mechanics; no medical code, term, or scenario.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

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


#: Bounded network patience. An anchor that hangs is an anchor that cannot answer, and an
#: unanswered anchor must FAIL rather than stall the release path indefinitely.
S3_CONNECT_TIMEOUT_SECONDS = 5
S3_READ_TIMEOUT_SECONDS = 10
S3_MAX_ATTEMPTS = 3
#: Hard bound on the forward walk `S3CheckpointAnchor.read` performs over immutable
#: per-sequence objects. Healthy operation costs exactly one probe (the 404 for "no
#: checkpoint beyond the head"); more than that means the mutable head pointer was moved
#: backwards, which is an attack, and exhausting the bound REFUSES rather than guesses.
S3_MAX_FORWARD_PROBES = 1024


#: Process-and-region scoped AWS client reuse. Production resolves a FRESH anchor object
#: per call (so a configuration change takes effect without a restart, and a broken spec
#: fails closed at the point of use) — which, with a per-object client, meant building a
#: botocore session, reloading the S3 service model and re-resolving instance-role
#: credentials several times per audit record. That is not merely slow: instance metadata
#: is rate-limited, so hammering it under load returns throttles, and every throttle here
#: is an UNVERIFIABLE anchor that correctly but needlessly holds a release. The client is
#: therefore shared. Keyed by PID as well as region because a boto3 client carries open
#: sockets that must never be inherited across a fork into a worker process.
_S3_CLIENTS: dict[tuple, object] = {}
_S3_CLIENT_LOCK = threading.Lock()
#: Result of the one-time "can this SDK make write-once puts?" probe (see
#: `S3CheckpointAnchor._assert_conditional_writes_available`). A property of the installed
#: SDK, so it is invariant for the life of the process.
_CONDITIONAL_WRITE_SUPPORT: bool | None = None


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _safe_store_name(store_id: str) -> str:
    """A store id reduced to a name safe to use as a path/key component.

    Shared by every backend so one store resolves to the same identity whichever backend
    anchors it, and so a store id can never escape its namespace (`../`, absolute paths,
    S3 key delimiters) in either.
    """
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(store_id))
    safe = safe.strip(".")
    if not safe:
        raise CheckpointConfigError("store id resolves to an empty checkpoint name")
    return safe


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
        return self.root / f"{_safe_store_name(store_id)}.checkpoint.json"

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
        # Backend-agnostic name for "where the checkpoints actually live", so the durable
        # release record can state WHICH store anchored a release without the pipeline
        # having to know one backend's field names.
        info["location"] = str(self.root)
        info["limitation"] = (
            "reference backend: same filesystem and identity as the store it anchors, "
            "so it is tamper-EVIDENT against an unprivileged writer but is not a "
            "separately controlled append-only store")
        return info


class S3CheckpointAnchor(CheckpointAnchor):
    """REAL BACKEND — a separately administered, append-only object store.

    WHAT MAKES THIS A TRUST BOUNDARY (`external = True`)
    ----------------------------------------------------
    Three mechanisms, of which the FIRST is the load-bearing one:

      1. THE APPLICATION IDENTITY CANNOT DELETE. `terraform/s3_checkpoint.tf` attaches an
         explicit `Deny` on `DeleteObject`/`DeleteObjectVersion`/retention-bypass/
         bucket-policy/versioning actions for this bucket. An explicit Deny outranks
         every Allow from every policy, present or future, so erasing an anchored
         checkpoint requires credentials the pipeline does not hold. This is what the
         local reference backend structurally cannot offer and is proved, against the
         live bucket, by `tests/test_checkpoint_s3.py`.
      2. HISTORY SURVIVES A WRITE. The bucket is versioned and version-deletion is denied,
         so even an overwrite preserves what it overwrote for an auditor with credentials
         this role does not have.
      3. WRITE-ONCE OBJECTS. Every checkpoint sequence is its own object, written with a
         conditional put (`If-None-Match: *`) that the STORE refuses when the key already
         exists — not our own bookkeeping. This is what stops an ordinary rollback/replay
         from succeeding at all rather than merely leaving evidence.

    WHAT ADMINISTERS THE BOUNDARY. The Deny above must live outside the principal it
    constrains, or it is advice rather than a boundary. It does: the runtime role holds no
    IAM permissions over itself, and the broader terraform-capable permissions live on a
    separate role assumable only by the account root, never by the box's instance
    credentials (`terraform/iam_terraform_access.tf`, Codex F6-R4-A finding B). The
    anchor's grant is therefore the whole of what the writer can do to this bucket:
    `PutObject`/`GetObject` under one prefix, and nothing else anywhere in it.

    HONEST LIMIT: (3) is enforced by the request this code sends, not by a bucket policy or
    an Object-Lock retention rule. It does not weaken (1) or (2), which are policy-enforced
    and are the properties the control depends on: an attacker running as this role can add
    to the anchored history and, with effort, obscure the newest pointer, but cannot remove
    or destroy what was anchored.

    The mutable `head.json` pointer is deliberately NOT trusted. It is a starting FLOOR
    for `read()`, which then walks forward over the write-once objects, so rolling the
    pointer backwards does not lower the anchor: the walk finds the immutable evidence of
    every sequence past it, reports the true head, and the caller refuses the truncated
    journal. That is the whole point — the pointer is an optimisation, the objects are
    the record.

    Key layout under `<prefix>`:
      * `<store>/head.json`               — untrusted pointer to the last written head
      * `<store>/seq/<seq:012d>.json`     — write-once, one per sequence, the real record

    UNVERIFIABLE IS A FAILURE. Every network error, credential error, throttle, and
    permission denial raises `AnchorUnavailable`. Nothing here can return `None` (which
    the caller reads as "never anchored") for any reason other than the store positively
    answering `NoSuchKey` — notably not for a 404 in general, since `NoSuchBucket` is one
    too and a vanished anchor store must hold the release, not disable the control.

    LATENCY IS BOUNDED, and deliberately: the anchor is consulted and advanced inside the
    repository's `BEGIN IMMEDIATE` section (that ordering is the atomicity contract this
    control rests on), so every second spent here is a second the audit store's write lock
    is held. The timeouts and attempt count below cap that; a store that cannot answer
    within them fails the append rather than stalling the release path, which is the
    correct trade — a held release is recoverable, a hung one is not.
    """

    backend_id = "s3-object-store"
    configured = True
    external = True

    def __init__(self, bucket: str, prefix: str, *, region: str | None = None,
                 client=None) -> None:
        self.bucket = str(bucket).strip()
        if not self.bucket:
            raise CheckpointConfigError("checkpoint anchor names no S3 bucket")
        prefix = str(prefix or "").strip().lstrip("/")
        if not prefix:
            # Refused at CONFIGURATION time rather than at the first write: the anchor's
            # IAM statement scopes its grant to a prefix, so a bucket-root spec writes
            # outside the reviewed grant and is answered AccessDenied on the RELEASE path --
            # diagnosed hours later and far from the typo that caused it. (Proved live: the
            # runtime role is denied PutObject outside the granted prefix.)
            raise CheckpointConfigError(
                "checkpoint anchor names no S3 key prefix; the anchor grant is "
                "prefix-scoped, so a bucket-root spec cannot be honoured")
        self.prefix = prefix if prefix.endswith("/") else prefix + "/"
        self.region = (str(region).strip() or None) if region else None
        self._client = client
        if client is None:
            self._assert_conditional_writes_available()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_spec(cls, spec: str) -> "S3CheckpointAnchor":
        """Parse `s3://<bucket>/<prefix>[?region=<region>]`.

        Unknown query parameters are an ERROR, not ignored: a silently dropped option is
        how an operator ends up believing they configured something they did not.
        """
        parts = urlsplit(str(spec).strip())
        if parts.scheme.lower() != "s3":
            raise CheckpointConfigError(f"{spec!r} is not an s3:// checkpoint anchor spec")
        if parts.fragment:
            raise CheckpointConfigError(f"{spec!r} carries an unsupported URL fragment")
        options = dict(parse_qsl(parts.query, keep_blank_values=True))
        unknown = sorted(set(options) - {"region"})
        if unknown:
            raise CheckpointConfigError(
                f"{spec!r} carries unsupported option(s) {unknown}; refusing to run with "
                f"configuration this build would silently ignore")
        region = options.get("region") or os.getenv("AWS_REGION") \
            or os.getenv("AWS_DEFAULT_REGION") or None
        return cls(parts.netloc, parts.path, region=region)

    @staticmethod
    def _assert_conditional_writes_available() -> None:
        """Fail at CONFIGURATION time if the installed SDK cannot make write-once puts.

        One leg of this backend's `external = True` claim is the conditional put. An SDK
        too old to send `If-None-Match` would still function -- and would silently
        downgrade the anchor to last-writer-wins. Checked against the local service model,
        so it costs no network call and no credentials, and memoised because the answer is
        a property of the installed SDK.
        """
        global _CONDITIONAL_WRITE_SUPPORT
        if _CONDITIONAL_WRITE_SUPPORT:
            return
        try:
            import botocore.session
        except ImportError as exc:
            raise CheckpointConfigError(
                "the S3 checkpoint anchor is configured but boto3/botocore is not "
                "installed in this build; refusing to run as if an anchor existed") from exc
        try:
            model = botocore.session.get_session().get_service_model("s3")
            members = model.operation_model("PutObject").input_shape.members
        except Exception as exc:                       # pragma: no cover - defensive
            raise CheckpointConfigError(
                f"could not inspect the installed AWS SDK's S3 model: {exc}") from exc
        if "IfNoneMatch" not in members:
            raise CheckpointConfigError(
                "the installed AWS SDK cannot send conditional (If-None-Match) puts, so "
                "checkpoint objects could be overwritten rather than written once; "
                "refusing to present this as an external trust boundary")
        _CONDITIONAL_WRITE_SUPPORT = True

    def _s3(self):
        if self._client is not None:
            return self._client
        key = (os.getpid(), self.region or "")
        with _S3_CLIENT_LOCK:
            client = _S3_CLIENTS.get(key)
            if client is None:
                try:
                    import boto3
                    from botocore.config import Config
                    client = boto3.client(
                        "s3", region_name=self.region,
                        config=Config(connect_timeout=S3_CONNECT_TIMEOUT_SECONDS,
                                      read_timeout=S3_READ_TIMEOUT_SECONDS,
                                      retries={"max_attempts": S3_MAX_ATTEMPTS,
                                               "mode": "standard"}))
                except ImportError as exc:             # pragma: no cover - guarded above
                    raise CheckpointConfigError(
                        f"the S3 checkpoint anchor requires boto3: {exc}") from exc
                except Exception as exc:
                    # Region resolution, endpoint resolution, a broken instance-metadata
                    # service: all mean the anchor cannot be consulted. UNVERIFIABLE, and
                    # deliberately NOT cached -- a transient failure must not poison every
                    # later attempt in this process.
                    raise AnchorUnavailable(
                        f"the S3 checkpoint anchor client could not be built: "
                        f"{type(exc).__name__}: {exc}") from exc
                _S3_CLIENTS[key] = client
        return client

    # ------------------------------------------------------------------------- keys
    def _head_key(self, store_id: str) -> str:
        return f"{self.prefix}{_safe_store_name(store_id)}/head.json"

    def _seq_key(self, store_id: str, seq: int) -> str:
        # Zero-padded so the objects also sort lexicographically for a human (or an
        # auditor with list permission this role deliberately does not have).
        return f"{self.prefix}{_safe_store_name(store_id)}/seq/{int(seq):012d}.json"

    # ------------------------------------------------------------- transport helpers
    @staticmethod
    def _error_code(exc) -> str:
        """The S3 error code only. Deliberately narrow: the full botocore response is not
        logged or raised anywhere, so request ids, endpoints and header material never
        reach an audit artifact or a log line."""
        try:
            return str((getattr(exc, "response", None) or {}).get("Error", {}).get("Code") or "")
        except Exception:                              # pragma: no cover - defensive
            return ""

    def _get(self, key: str) -> dict | None:
        """The record at `key`, or None ONLY when the store positively answers `NoSuchKey`
        for a key this role is permitted to read. Everything else raises."""
        from botocore.exceptions import BotoCoreError, ClientError
        try:
            body = self._s3().get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as exc:
            code = self._error_code(exc)
            # STRICTLY `NoSuchKey`, matched on the error code and never on the HTTP
            # status. `NoSuchBucket` is ALSO a 404, and a status-based test folded a
            # missing/renamed/wrong-account bucket into "this store was never anchored" --
            # i.e. a vanished anchor store would have silently disabled the control
            # instead of holding the release. (Caught by the live regression against a
            # bucket that does not exist; it is exactly the silent-empty-success class
            # this codebase keeps finding.) Everything else -- AccessDenied, NoSuchBucket,
            # SlowDown, expired credentials, region redirect -- means UNKNOWN, and
            # unknown must hold the release.
            if code == "NoSuchKey":
                return None
            raise AnchorUnavailable(
                f"checkpoint anchor could not read s3://{self.bucket}/{key} "
                f"({code or 'ClientError'})") from exc
        except BotoCoreError as exc:
            raise AnchorUnavailable(
                f"checkpoint anchor could not read s3://{self.bucket}/{key} "
                f"({type(exc).__name__})") from exc
        try:
            record = json.loads(body)
        except Exception as exc:
            raise AnchorFormatError(
                f"checkpoint at s3://{self.bucket}/{key} is not valid JSON: {exc}") from exc
        return record

    def _put(self, key: str, body: bytes, *, only_if_absent: bool) -> bool:
        """Write `body` at `key`. With `only_if_absent`, the STORE refuses an existing
        key (412) and this returns False; the caller decides whether that is benign
        idempotence or a fork attempt. Server-side encryption is deliberately not sent:
        the bucket's own default rule owns that, so encryption cannot drift from what
        infrastructure declares."""
        from botocore.exceptions import BotoCoreError, ClientError
        extra = {"IfNoneMatch": "*"} if only_if_absent else {}
        try:
            self._s3().put_object(Bucket=self.bucket, Key=key, Body=body,
                                  ContentType="application/json", **extra)
            return True
        except ClientError as exc:
            code = self._error_code(exc)
            if only_if_absent and code == "PreconditionFailed":
                return False
            raise AnchorUnavailable(
                f"checkpoint anchor could not write s3://{self.bucket}/{key} "
                f"({code or 'ClientError'})") from exc
        except BotoCoreError as exc:
            raise AnchorUnavailable(
                f"checkpoint anchor could not write s3://{self.bucket}/{key} "
                f"({type(exc).__name__})") from exc

    def _checkpoint_at(self, store_id: str, seq: int) -> dict | None:
        """One write-once sequence object, validated against the key it was found under.

        The key/content agreement matters: without it a record could be planted at a high
        sequence while CLAIMING a low one, which would make the forward walk report a head
        the store does not actually attest to."""
        record = self._get(self._seq_key(store_id, seq))
        if record is None:
            return None
        cp = Checkpoint.from_record(record)
        if cp.seq != int(seq):
            raise AnchorFormatError(
                f"checkpoint object for sequence {seq} declares sequence {cp.seq}")
        if cp.store_id != str(store_id):
            raise AnchorFormatError(
                f"checkpoint object under store {store_id!r} declares store {cp.store_id!r}")
        return record

    # -------------------------------------------------------------------- interface
    def read(self, store_id: str) -> dict | None:
        """The HIGHEST sequence this store can prove is anchored.

        The mutable head pointer supplies a starting floor and nothing else; the answer
        comes from walking the write-once objects forward from it. Consequently an
        attacker who rewrites the pointer (the only mutable thing here) cannot lower the
        anchor, and the caller still sees an anchor ahead of the truncated journal.
        """
        best = self._get(self._head_key(store_id))
        floor = 0
        if best is not None:
            head = Checkpoint.from_record(best)        # malformed != absent
            if head.store_id != str(store_id):
                raise AnchorFormatError(
                    f"checkpoint head under store {store_id!r} declares store "
                    f"{head.store_id!r}")
            floor = head.seq
        seq = floor + 1
        for _ in range(S3_MAX_FORWARD_PROBES):
            found = self._checkpoint_at(store_id, seq)
            if found is None:
                return best
            best, seq = found, seq + 1
        raise AnchorUnavailable(
            f"the checkpoint anchor holds at least {seq - 1} sequences for store "
            f"{store_id!r} but its head pointer claims {floor}; refusing to guess a "
            f"terminal head after {S3_MAX_FORWARD_PROBES} probes")

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
        body = json.dumps(record, sort_keys=True).encode("utf-8")
        # The write-once object FIRST: it is the record. If the head pointer write then
        # fails the append aborts, and the next `read()` still finds this sequence by
        # walking forward from the stale pointer -- so a crash here leaves the anchor
        # correct, merely without its shortcut.
        seq_key = self._seq_key(incoming.store_id, incoming.seq)
        if not self._put(seq_key, body, only_if_absent=True):
            stored = self._checkpoint_at(incoming.store_id, incoming.seq)
            if stored is None:                         # pragma: no cover - racing delete
                raise AnchorUnavailable(
                    f"s3://{self.bucket}/{seq_key} rejected a write-once put but cannot "
                    f"be read back")
            if json.dumps(stored, sort_keys=True) != json.dumps(record, sort_keys=True):
                raise AnchorRollback(
                    f"refusing to overwrite the write-once checkpoint already anchored at "
                    f"seq {incoming.seq}")
            # Byte-identical re-put: the store already holds exactly this checkpoint.
        self._put(self._head_key(incoming.store_id), body, only_if_absent=False)

    def describe(self) -> dict:
        info = super().describe()
        info["location"] = f"s3://{self.bucket}/{self.prefix}"
        info["region"] = self.region or "sdk-default"
        info["guarantee"] = (
            "checkpoints are write-once objects in a separately administered, versioned "
            "bucket on which the audit writer's own role is explicitly denied delete, "
            "version-delete and retention-bypass, so the identity that appends to the "
            "audit log can extend the anchored history but cannot erase it")
        return info


def resolve_checkpoint_anchor(spec: str | None = None) -> CheckpointAnchor:
    """Build the configured anchor. Fail-closed on an unrecognised spec.

    Accepted:
      * unset / "" / "none" / "disabled" / "off"    -> `DisabledCheckpointAnchor`
      * "file:<directory>"                          -> `LocalFileCheckpointAnchor`
      * "s3://<bucket>/<prefix>[?region=<region>]"  -> `S3CheckpointAnchor`

    Anything else raises: naming a backend this build cannot honour must stop the writer,
    not read as 'no anchor configured'.
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
    if text.lower().startswith("s3://"):
        return S3CheckpointAnchor.from_spec(text)
    raise CheckpointConfigError(
        f"{ANCHOR_ENV}={text!r} names no backend implemented by this build (supported: "
        f"'none', 'file:<directory>', 's3://<bucket>/<prefix>'); refusing to run as if "
        f"an anchor existed")


def checkpoint_required() -> bool:
    """True when running WITHOUT a configured anchor must fail closed."""
    return _truthy(os.getenv(REQUIRED_ENV))


def checkpoint_adoption_allowed() -> bool:
    """True when a journal that predates the anchor may be adopted (operator consent)."""
    return _truthy(os.getenv(ADOPT_ENV))
