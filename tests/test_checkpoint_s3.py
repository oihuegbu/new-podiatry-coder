"""LIVE regressions for `S3CheckpointAnchor` — the real external terminal-head anchor.

WHY THESE ARE NOT MOCKED
------------------------
Every property this backend exists for is a property of the STORE, not of our code:
that a `PutObject` cannot overwrite an existing sequence, that a `DeleteObject` is
refused, that a read outside the granted prefix is denied rather than empty. A mocked S3
would assert only that we call the API we think we call — which is precisely the belief
that was wrong the last time (the app role held `PowerUserAccess`, so it could delete
from the anchor bucket until an explicit `Deny` was added; a mock would have stayed
green through that entire window). So these run against the actual bucket.

RUNNING THEM
------------
They are skipped unless `PROVENANCE_CHECKPOINT_S3_TEST_URI` names a real anchor spec,
because the hermetic test image has no route to AWS and a test that cannot reach its
subject must say so rather than fail as if the code were broken:

    PROVENANCE_CHECKPOINT_S3_TEST_URI='s3://<bucket>/checkpoints?region=us-east-1' \
        python -m pytest tests/test_checkpoint_s3.py -q

Credentials come from the instance role; nothing here configures any.

Each test namespaces itself under a fresh store id, so runs never collide -- necessary
because the objects are write-once and this role cannot (by design) clean up after
itself. The residue is a few hundred bytes of JSON per run.

Agnostic: pure storage/sequence mechanics; no medical code, term, or scenario.
"""
import json
import os
import uuid

import pytest

_URI = (os.environ.get("PROVENANCE_CHECKPOINT_S3_TEST_URI") or "").strip()
if not _URI:
    pytest.skip(
        "set PROVENANCE_CHECKPOINT_S3_TEST_URI='s3://<bucket>/<prefix>?region=<region>' to "
        "run the LIVE external-checkpoint-anchor regressions (they require real AWS "
        "reachability and the instance role; there is no offline substitute for them)",
        allow_module_level=True)


def _anchor():
    from claude_coder.checkpoint import S3CheckpointAnchor, resolve_checkpoint_anchor
    anchor = resolve_checkpoint_anchor(_URI)
    assert isinstance(anchor, S3CheckpointAnchor)
    return anchor


def _store_id(tag: str) -> str:
    return f"live-{tag}-{uuid.uuid4().hex}.db"


def _repo(tmp_path, store_id, *, name="live.db"):
    """A repository bound to the REAL anchor, under a unique store id. A fresh repository
    and a fresh anchor object each time is what a process restart looks like."""
    from claude_coder.provenance import SqliteAuditRepository
    os.environ["PROVENANCE_STORE_ID"] = store_id
    return SqliteAuditRepository(tmp_path / name, checkpoint_anchor=_anchor())


_ENV_KEYS = ("PROVENANCE_STORE_ID", "PROVENANCE_CHECKPOINT_ANCHOR",
             "PROVENANCE_CHECKPOINT_REQUIRED", "PROVENANCE_CHECKPOINT_ADOPT")


@pytest.fixture(autouse=True)
def _isolated_anchor_env():
    """Snapshot/clear/restore. These tests set the store id through the environment (the
    way production does), so without a restore they would leak configuration into every
    later test in the session -- an isolation bug that reads as a mysterious failure
    somewhere else entirely."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for key in _ENV_KEYS:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _checkpoint(store_id, seq, seal="a" * 64):
    from claude_coder.checkpoint import Checkpoint
    return Checkpoint(store_id=store_id, seq=seq, seal=seal, record_sha256="b" * 64,
                      witness_version="audit-head-witness-v3",
                      written_at="2026-08-11T00:00:00+00:00", signature="sig-" + seal[:8])


# ------------------------------------------------ absence must be OBSERVABLE, not guessed
def test_the_deployed_identity_can_tell_absent_from_denied():
    """The precondition every other test here rests on, pinned as its own regression.

    S3 answers GetObject for a key that does not exist with 403 AccessDenied -- not 404
    NoSuchKey -- unless the caller holds `s3:ListBucket` on the bucket. This backend refuses
    to read 403 as "never anchored" (that would be the silent-empty-success this whole
    module exists to prevent), so without that grant `read()` raises on every first read and
    every encounter holds. The failure is total, and it is invisible in any mocked test and
    in any policy review that only looks at Get/Put/Delete.

    This is not hypothetical: it is what removing PowerUserAccess from the runtime role
    (F6-R4-A finding B) actually did -- PowerUserAccess had been supplying ListBucket
    bucket-wide as a side effect, so narrowing the role correctly also, silently, removed
    the anchor's ability to observe absence. `terraform/s3_checkpoint.tf` now grants it
    explicitly; this test is what tells you if it is ever dropped again.
    """
    anchor = _anchor()
    assert anchor.read(_store_id("absence")) is None


# ---------------------------------------------------------------- the basic round trip
def test_a_checkpoint_written_to_the_real_bucket_reads_back_identically():
    """The floor: a real PutObject followed by a real GetObject returns the same record,
    and an unanchored store reads back as None (never as an error, never as a stale hit
    from another store's namespace)."""
    anchor = _anchor()
    store = _store_id("roundtrip")
    assert anchor.read(store) is None                       # never anchored yet
    written = _checkpoint(store, 1).as_record()
    anchor.write(written)
    assert _anchor().read(store) == written                 # fresh client, fresh object
    later = _checkpoint(store, 2, seal="c" * 64).as_record()
    anchor.write(later)
    assert _anchor().read(store) == later


def test_the_backend_reports_itself_as_a_real_external_trust_boundary():
    """`external = True` is the acceptance criterion, so it is asserted against the live
    configuration -- together with the delete refusal below, which is what earns it."""
    described = _anchor().describe()
    assert described["configured"] is True
    assert described["external_trust_boundary"] is True
    assert described["backend"] == "s3-object-store"
    assert described["location"].startswith("s3://")
    assert "limitation" not in described                    # the reference backend's field


# --------------------------------------------------- what makes the boundary a boundary
def test_the_writer_role_cannot_delete_what_it_anchored():
    """THE claim. Everything else here is bookkeeping if this fails: the identity that
    appends to the audit log must not be able to remove the external record that catches a
    consistent local truncation. Both verbs are checked -- a plain delete (which would
    hide the object behind a delete marker) and a version delete (which would destroy it)."""
    from botocore.exceptions import ClientError
    anchor = _anchor()
    store = _store_id("nodelete")
    anchor.write(_checkpoint(store, 1).as_record())
    key = anchor._seq_key(store, 1)
    client = anchor._s3()
    with pytest.raises(ClientError) as err:
        client.delete_object(Bucket=anchor.bucket, Key=key)
    assert anchor._error_code(err.value) == "AccessDenied"
    with pytest.raises(ClientError) as err:
        client.delete_object(Bucket=anchor.bucket, Key=key, VersionId="null")
    assert anchor._error_code(err.value) == "AccessDenied"
    assert _anchor().read(store) is not None                # still there afterwards


def test_the_writer_role_cannot_reach_outside_the_granted_prefix():
    """The anchor's grant is PREFIX-SCOPED, and that scoping is now independently enforced
    rather than subsumed by a broader grant on the same role.

    This is what changed with F6-R4-A finding B: the runtime role used to carry
    PowerUserAccess, which allowed Get/PutObject bucket-wide, so the prefix in the anchor's
    own statement bought nothing. PowerUserAccess now lives on a separate terraform-operator
    role that the box's instance credentials cannot assume, and the runtime role's ONLY
    reach into this bucket is the anchor prefix. Asserted from the deployed identity,
    because a policy is only worth what the live principal is actually refused."""
    from botocore.exceptions import ClientError
    anchor = _anchor()
    client = anchor._s3()
    outside = f"outside-the-anchor-prefix/{uuid.uuid4().hex}.json"
    with pytest.raises(ClientError) as err:
        client.put_object(Bucket=anchor.bucket, Key=outside, Body=b"{}")
    assert anchor._error_code(err.value) == "AccessDenied"
    with pytest.raises(ClientError) as err:
        client.get_object(Bucket=anchor.bucket, Key=outside)
    assert anchor._error_code(err.value) == "AccessDenied"


def test_the_writer_role_cannot_weaken_the_policy_that_denies_it():
    """A Deny is a boundary only if it lives outside the principal it constrains.

    The delete-Deny is an inline policy on the runtime role itself, so the question that
    decides whether it is a real boundary is whether that role can edit its own policies.
    It cannot: no IAM permission over itself, and the terraform-capable role that does hold
    them is assumable only by the account root (terraform/iam_terraform_access.tf).

    The probe is deliberately harmless in the failure case as well as the success case: it
    attempts to ADD a policy under an unused name granting an action that requires no
    permission, never to edit or delete the load-bearing one -- and if it unexpectedly
    succeeds it fails the test loudly and removes what it created."""
    import boto3
    from botocore.exceptions import ClientError
    sts = boto3.client("sts", region_name=_anchor().region or "us-east-1")
    role = sts.get_caller_identity()["Arn"].split("/")[1]   # assumed-role/<role>/<session>
    iam = boto3.client("iam", region_name=_anchor().region or "us-east-1")
    probe = f"provenance-checkpoint-tamper-probe-{uuid.uuid4().hex[:8]}"
    document = json.dumps({"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "sts:GetCallerIdentity", "Resource": "*"}]})
    try:
        with pytest.raises(ClientError) as err:
            iam.put_role_policy(RoleName=role, PolicyName=probe, PolicyDocument=document)
    finally:
        try:
            iam.delete_role_policy(RoleName=role, PolicyName=probe)
        except ClientError:
            pass                                            # the expected case: never created
    assert "AccessDenied" in str(err.value)
    # ...and it cannot even read its own policies, let alone rewrite them.
    with pytest.raises(ClientError) as err:
        iam.list_role_policies(RoleName=role)
    assert "AccessDenied" in str(err.value)


def test_an_anchored_sequence_cannot_be_overwritten_even_by_the_writer():
    """The second mechanism: the sequence objects are WRITE-ONCE at the store, so the one
    write verb this role holds cannot rewrite history either. A byte-identical re-put is
    idempotent (a retried append must not brick the store); different content at the same
    sequence is a fork attempt and is refused."""
    from claude_coder.checkpoint import AnchorRollback
    anchor = _anchor()
    store = _store_id("writeonce")
    original = _checkpoint(store, 1).as_record()
    anchor.write(original)
    anchor.write(dict(original))                            # idempotent re-put: fine
    with pytest.raises(AnchorRollback) as err:
        anchor.write(_checkpoint(store, 1, seal="d" * 64).as_record())
    assert "fork the checkpoint anchor" in str(err.value)
    with pytest.raises(AnchorRollback):
        _anchor().write(_checkpoint(store, 1, seal="e" * 64).as_record())
    assert _anchor().read(store) == original                # unchanged


def test_rewriting_the_head_pointer_does_not_lower_the_anchor():
    """The head pointer is the ONLY mutable object here, so it is the only thing an
    attacker with this role can move -- and moving it must buy them nothing. `read()`
    treats it as a floor and walks the write-once objects forward, so a rolled-back
    pointer still reports the true terminal head."""
    anchor = _anchor()
    store = _store_id("headroll")
    first = _checkpoint(store, 1).as_record()
    anchor.write(first)
    anchor.write(_checkpoint(store, 2, seal="c" * 64).as_record())
    third = _checkpoint(store, 3, seal="d" * 64).as_record()
    anchor.write(third)
    # Roll the pointer back to sequence 1 -- a plain, permitted PutObject.
    anchor._s3().put_object(Bucket=anchor.bucket, Key=anchor._head_key(store),
                            Body=json.dumps(first, sort_keys=True).encode(),
                            ContentType="application/json")
    assert _anchor().read(store) == third                   # the objects, not the pointer


def test_a_head_pointer_naming_another_store_is_refused_not_adopted():
    """Cross-namespace planting: a record whose declared store id does not match the one
    being read is malformed, not a checkpoint for this store."""
    from claude_coder.checkpoint import AnchorFormatError
    anchor = _anchor()
    store, other = _store_id("mine"), _store_id("theirs")
    anchor._s3().put_object(
        Bucket=anchor.bucket, Key=anchor._head_key(store),
        Body=json.dumps(_checkpoint(other, 1).as_record(), sort_keys=True).encode(),
        ContentType="application/json")
    with pytest.raises(AnchorFormatError):
        _anchor().read(store)


# -------------------------------------------------------- the enforcement path, for real
def test_a_tampered_checkpoint_object_holds_the_release(tmp_path):
    """The anchor store is not blindly trusted either. Anyone who can write the bucket can
    plant a record, so the checkpoint is signed with the journal's own sealing custody: a
    real signed checkpoint replaced in the REAL bucket by one whose signature does not
    verify holds the release, instead of redefining what 'complete' means."""
    repo = _repo(tmp_path, _store_id("forged"))
    repo.append("enc", "a", {"x": 1})
    anchor, store = _anchor(), repo.store_id()
    genuine = anchor.read(store)
    forged = dict(genuine, signature="0" * 64)
    anchor._s3().put_object(Bucket=anchor.bucket, Key=anchor._head_key(store),
                            Body=json.dumps(forged, sort_keys=True).encode(),
                            ContentType="application/json")
    with pytest.raises(OSError) as err:
        _repo(tmp_path, store).append("enc", "b", {"x": 2})
    assert "fails its signature" in str(err.value)
    assert len(_repo(tmp_path, store).records("enc")) == 1          # nothing committed
    assert any("fails its signature" in p
               for p in _repo(tmp_path, store).verify_chain("enc"))


def test_the_reviewers_consistent_truncation_is_refused_against_the_real_bucket(tmp_path):
    """THE reproduction, unmodified, against the real external anchor:

      1. append two records;
      2. remove the terminal witness journal line;
      3. drop the terminal audit row (dropping the append-only trigger to do it);
      4. verify_chain  -- must REPORT the truncation rather than return intact;
      5. a subsequent append -- must REFUSE rather than extend the shortened chain.

    Steps 2-3 leave a locally self-consistent store: this is the state the hash chain and
    the sealed journal structurally cannot distinguish from a shorter honest history. Only
    the external anchor can, and only because the writer could not reach into the bucket
    and shorten it too.
    """
    from tests.test_provenance import _truncate_both_tails
    store = _store_id("truncation")
    repo = _repo(tmp_path, store)
    repo.append("enc", "a", {"x": 1})
    repo.append("enc", "b", {"x": 2})
    assert _repo(tmp_path, store).verify_chain("enc") == []         # intact before

    _truncate_both_tails(repo)                                      # steps 2 and 3

    problems = _repo(tmp_path, store).verify_chain("enc")           # step 4
    assert any("SHORTER than its anchored checkpoint" in p for p in problems), problems

    with pytest.raises(OSError) as err:                             # step 5
        _repo(tmp_path, store).append("enc", "c", {"x": 3})
    assert "SHORTER than its anchored checkpoint" in str(err.value)
    assert len(_repo(tmp_path, store).records("enc")) == 1          # still refused


def test_a_healthy_run_stays_aligned_and_reports_the_external_boundary(tmp_path):
    """The other half of fail-closed: normal operation against the real bucket must be
    boringly clean -- aligned sequences, no problems, and a durable status that names the
    external boundary now in force."""
    store = _store_id("healthy")
    for kind in ("a", "b", "c"):
        _repo(tmp_path, store).append("enc", kind, {"k": kind})
    repo = _repo(tmp_path, store)
    assert repo.verify_chain("enc") == []
    status = repo.witness_status()["checkpoint"]
    assert status["problems"] == []
    assert status["journal_seq"] == status["anchored_seq"] == 3
    assert status["external_trust_boundary"] is True
    assert status["backend"] == "s3-object-store"
    assert status["store_id"] == store


def test_the_anchor_is_selected_by_configuration_alone(tmp_path, monkeypatch):
    """Production turns this on with one environment variable and no code change, so the
    real resolution path -- env var to live bucket -- is what gets driven here."""
    from claude_coder.provenance import SqliteAuditRepository
    store = _store_id("configured")
    monkeypatch.setenv("PROVENANCE_STORE_ID", store)
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_ANCHOR", _URI)
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_REQUIRED", "1")
    dbp = tmp_path / "configured.db"
    SqliteAuditRepository(dbp).append("enc", "a", {"x": 1})
    SqliteAuditRepository(dbp).append("enc", "b", {"x": 2})
    repo = SqliteAuditRepository(dbp)
    assert repo.verify_chain("enc") == []
    status = repo.witness_status()["checkpoint"]
    assert status["backend"] == "s3-object-store"
    assert status["external_trust_boundary"] is True
    assert status["required"] is True


# ------------------------------------------------- unreachable/denied must HOLD, not pass
def _unsigned_client(anchor):
    """A REAL S3 client with no credentials attached, so the real bucket answers a real 403.

    Used rather than revoking a permission because these tests must not mutate the IAM
    posture they are verifying (and, by design, this identity could not put it back).
    Anonymous access to a private bucket produces the identical `AccessDenied` the
    classification turns on."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", region_name=anchor.region or "us-east-1",
                        config=Config(signature_version=UNSIGNED, connect_timeout=5,
                                      read_timeout=10, retries={"max_attempts": 2}))


def test_an_access_denied_read_is_unverifiable_never_unanchored(tmp_path):
    """The failure mode this whole design turns on. A read the caller is not permitted to
    make returns 403, which looks EXACTLY like 'this store was never anchored'. Folding it
    into 'absent' would let one permission regression silently switch the control off
    while every release kept certifying an external anchor. It must be UNVERIFIABLE, and
    unverifiable must HOLD -- asserted with PROVENANCE_CHECKPOINT_REQUIRED set, the
    production posture."""
    from claude_coder.checkpoint import AnchorUnavailable, S3CheckpointAnchor
    from claude_coder.provenance import SqliteAuditRepository
    live = _anchor()
    denied = S3CheckpointAnchor(live.bucket, live.prefix, region=live.region,
                                client=_unsigned_client(live))
    store = _store_id("denied")
    os.environ["PROVENANCE_STORE_ID"] = store
    os.environ["PROVENANCE_CHECKPOINT_REQUIRED"] = "1"
    with pytest.raises(AnchorUnavailable) as err:
        denied.read(store)
    assert "AccessDenied" in str(err.value) and "could not read" in str(err.value)
    repo = SqliteAuditRepository(tmp_path / "denied.db", checkpoint_anchor=denied)
    with pytest.raises(OSError) as err:
        repo.append("enc", "a", {"x": 1})
    assert "unverifiable" in str(err.value)
    assert repo.records("enc") == []                        # nothing committed
    assert any("unverifiable" in p for p in repo.verify_chain("enc"))


def test_a_nonexistent_bucket_holds_the_release(tmp_path):
    """REGRESSION for a bug this file caught: `NoSuchBucket` is served as HTTP 404, so an
    absence test written against the STATUS silently read a deleted/renamed/wrong-account
    anchor store as 'never anchored' -- the anchor evaporating would have disabled the
    control instead of holding the release. Only the `NoSuchKey` error code means absent."""
    from claude_coder.checkpoint import AnchorUnavailable, S3CheckpointAnchor
    from claude_coder.provenance import SqliteAuditRepository
    live = _anchor()
    absent = S3CheckpointAnchor(f"no-such-bucket-{uuid.uuid4().hex}", "checkpoints/",
                                region=live.region)
    store = _store_id("nobucket")
    os.environ["PROVENANCE_STORE_ID"] = store
    with pytest.raises(AnchorUnavailable) as err:
        absent.read(store)
    assert "NoSuchBucket" in str(err.value)
    repo = SqliteAuditRepository(tmp_path / "nobucket.db", checkpoint_anchor=absent)
    with pytest.raises(OSError) as err:
        repo.append("enc", "a", {"x": 1})
    assert "unverifiable" in str(err.value)
    assert repo.records("enc") == []


def test_an_s3_outage_holds_the_release_instead_of_hanging_or_passing(tmp_path):
    """An outage is not a mock: the client is pointed at an unroutable endpoint, so this
    is a real connection failure through the real SDK. It must (a) be bounded by the
    backend's own timeouts rather than hanging the release path, and (b) surface as
    UNVERIFIABLE -- the same verdict as tampering, never as 'no anchor configured'."""
    import time
    import boto3
    from botocore.config import Config
    from claude_coder.checkpoint import AnchorUnavailable, S3CheckpointAnchor
    from claude_coder.provenance import SqliteAuditRepository
    live = _anchor()
    unreachable = S3CheckpointAnchor(
        live.bucket, live.prefix, region=live.region,
        client=boto3.client("s3", region_name=live.region or "us-east-1",
                            endpoint_url="https://127.0.0.1:1",
                            config=Config(connect_timeout=2, read_timeout=2,
                                          retries={"max_attempts": 1})))
    store = _store_id("outage")
    os.environ["PROVENANCE_STORE_ID"] = store
    os.environ["PROVENANCE_CHECKPOINT_REQUIRED"] = "1"
    started = time.monotonic()
    with pytest.raises(AnchorUnavailable):
        unreachable.read(store)
    assert time.monotonic() - started < 60, "the anchor must not stall the release path"
    repo = SqliteAuditRepository(tmp_path / "outage.db", checkpoint_anchor=unreachable)
    with pytest.raises(OSError) as err:
        repo.append("enc", "a", {"x": 1})
    assert "unverifiable" in str(err.value)
    assert repo.records("enc") == []
    status = repo.checkpoint_status()
    assert status["external_trust_boundary"] is True        # still what was configured...
    assert any("unverifiable" in p for p in status["problems"])   # ...and it is not OK


def test_an_unwritable_anchor_leaves_no_committed_row(tmp_path):
    """The write side of the same rule, live: the checkpoint is advanced BETWEEN the
    journal fsync and the row commit, so a denied write aborts the append rather than
    committing a row the external anchor never covered. Reads go to the real bucket with
    real credentials and succeed; only the WRITE is unauthenticated, so both halves are
    genuine S3 round trips."""
    from claude_coder.checkpoint import S3CheckpointAnchor
    from claude_coder.provenance import SqliteAuditRepository
    live = _anchor()
    store = _store_id("unwritable")
    os.environ["PROVENANCE_STORE_ID"] = store
    unsigned = _unsigned_client(live)

    class _WriteDenied(S3CheckpointAnchor):
        def _put(self, key, body, *, only_if_absent):
            saved, self._client = self._client, unsigned
            try:
                return super()._put(key, body, only_if_absent=only_if_absent)
            finally:
                self._client = saved

    anchor = _WriteDenied(live.bucket, live.prefix, region=live.region)
    assert anchor.read(store) is None                       # the READ side really works
    repo = SqliteAuditRepository(tmp_path / "unwritable.db", checkpoint_anchor=anchor)
    with pytest.raises(OSError) as err:
        repo.append("enc", "a", {"x": 1})
    # The refusal comes from the bucket, not from our own bookkeeping: the message names
    # the object the store rejected and carries the store's error code (an unauthenticated
    # conditional put is answered InvalidRequest rather than AccessDenied -- either way
    # the write did not happen, which is the property under test).
    message = str(err.value)
    assert "could not be anchored" in message and "could not write" in message
    assert anchor._seq_key(store, 1) in message
    assert repo.records("enc") == []
    assert _anchor().read(store) is None                    # and nothing reached the bucket
