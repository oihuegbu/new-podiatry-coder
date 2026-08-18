"""Output attempts are atomic and superseding — issue #6 F6-R6-B, directive §7.

THE DEFECT THESE TESTS PIN
================================================================================
`run.py` delivered each note's result with `open(path, "w")`. If a NEW attempt
for a note failed partway through — disk full, a serialization error, a crash —
the OLDER result file stayed on disk untouched. That file can carry a releasable
claim, and the claims registry and the 837P submitter read whatever sits at
`<document>_results.json` as "the current result for this note". A failed re-run
therefore left a stale success eligible for submission.

WHAT IS PROVEN HERE
================================================================================
  * the exact reproduction: an older `releasable` artifact, a write failure on
    the next attempt, and the old artifact is no longer consumable by anything;
  * a crash between publishing and committing cannot make an artifact current;
  * an incomplete (IN_PROGRESS) and a failed (FAILED) attempt are both refused,
    by name, to the registry and to the submitter;
  * a COMPLETED attempt for an OLD document version is refused as current;
  * the atomic writer never truncates a file it fails to replace;
  * a results directory with no attempt store keeps its previous behaviour, so
    the tools that materialize result files themselves are unaffected;
  * (issue #6 F8-R2) a NEWER attempt that opens in the exact window between an
    older attempt's ownership check and its pointer commit still wins — proven
    for a second thread AND for a second OS process — and a late `fail` from an
    older attempt cannot delete a newer attempt's published result;
  * the per-document transition lock is released on every failure path, and a
    holder that never releases is refused on a deadline rather than waited on
    forever;
  * (issue #6 F8-R2, REOPENED) serializing writers was not enough: a process
    SIGKILLed at EVERY durable-write boundary of every transition never leaves
    an older claim consumable once a newer attempt exists in any durable form,
    the pointer-only attempt a crash can leave is recovered without reusing its
    sequence number, and a consumer refuses a pointer the append-only history
    proves stale even when that state is written by hand.

No medical code appears in this file; every payload is a synthetic stand-in for
an artifact shape.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.release.attempt_ledger import (  # noqa: E402
    Attempt, AttemptLedger, AttemptLedgerError, AttemptLockTimeout,
    AttemptState, AttemptSuperseded, AttemptWriteError, atomic_write_json,
    consumable, document_version_of, resolve_current)

VERSION_1 = "sha256:" + "1" * 64
VERSION_2 = "sha256:" + "2" * 64


def artifact(document_id: str, version: str, *, releasable: bool = True) -> dict:
    """A stand-in for a `ClaimBundle` carrying only what the ledger reads."""
    return {
        "schema_id": "claim_bundle",
        "encounter": {"document_id": document_id, "encounter_id": document_id,
                      "source_document": {"document_version": version}},
        "release": {"holds": [] if releasable else ["held"],
                    "producer_releasable": releasable},
    }


def released(ledger: AttemptLedger, document_id: str, version: str) -> Path:
    """One COMPLETED attempt carrying a releasable artifact — the thing that
    must not survive the next attempt's failure."""
    attempt = ledger.begin(document_id, version)
    return ledger.complete(attempt, artifact(document_id, version))


# --------------------------------------------------------------------------
# the reproduction
# --------------------------------------------------------------------------

def test_a_failed_attempt_does_not_leave_the_previous_release_consumable(tmp_path,
                                                                        monkeypatch):
    """THE reproduction. Old releasable artifact + an OSError on the new attempt.

    The injection fails exactly the artifact writes (a full disk), leaving the
    ledger able to record what happened — which is the realistic case, and the
    one where the old behaviour looked healthiest: a complete, well-formed,
    releasable JSON file sitting where every consumer looks.
    """
    ledger = AttemptLedger(tmp_path)
    published = released(ledger, "NOTE_A", VERSION_1)
    assert json.loads(published.read_text())["release"]["holds"] == []
    assert consumable(tmp_path, "NOTE_A") == (True, "")

    import app.release.attempt_ledger as module
    real = module.atomic_write_json

    def full_disk(path, payload, *, indent=None):
        if isinstance(payload, dict) and payload.get("schema") == module.POINTER_SCHEMA:
            return real(path, payload, indent=indent)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(module, "atomic_write_json", full_disk)

    attempt = ledger.begin("NOTE_A", VERSION_2)
    with pytest.raises(AttemptWriteError):
        ledger.complete(attempt, artifact("NOTE_A", VERSION_2))
    ledger.fail(attempt, error="OSError: No space left on device",
                tombstone={"processing_error": "output failed"})

    ok, why = consumable(tmp_path, "NOTE_A")
    assert ok is False
    assert "FAILED" in why, why
    assert not published.exists(), (
        "the superseded releasable artifact is still addressable at the path "
        "every consumer reads")
    current = resolve_current(tmp_path)
    assert current.results == ()
    assert "NOTE_A" in current.refusals


def test_the_previous_release_is_superseded_before_the_new_attempt_can_fail(tmp_path):
    """The ordering is the property: nothing needs to go wrong for the old
    result to stop being current. Opening the attempt is enough."""
    ledger = AttemptLedger(tmp_path)
    published = released(ledger, "NOTE_B", VERSION_1)

    ledger.begin("NOTE_B", VERSION_2)

    ok, why = consumable(tmp_path, "NOTE_B")
    assert ok is False
    assert "IN_PROGRESS" in why, why
    assert published.exists(), (
        "precondition: the file is still there — it is the POINTER that makes "
        "it non-current, not its deletion")
    assert resolve_current(tmp_path).results == ()


def test_a_crash_between_publishing_and_committing_leaves_nothing_current(tmp_path):
    """The artifact is fsynced to disk before the pointer commits, so a crash in
    that window publishes bytes that are not yet anyone's current result."""
    ledger = AttemptLedger(tmp_path)
    released(ledger, "NOTE_C", VERSION_1)
    attempt = ledger.begin("NOTE_C", VERSION_2)

    # exactly what `_publish` does before its commit, and then nothing more
    atomic_write_json(ledger.published_path("NOTE_C"),
                      artifact("NOTE_C", VERSION_2), indent=2)

    ok, why = consumable(tmp_path, "NOTE_C")
    assert ok is False and "IN_PROGRESS" in why
    pointer = ledger.pointer("NOTE_C")
    assert pointer["attempt_id"] == attempt.attempt_id
    assert pointer["state"] == AttemptState.IN_PROGRESS.value


def test_a_completed_attempt_for_an_old_document_version_is_not_current(tmp_path):
    """A stale artifact for a superseded document version is refused even though
    the attempt that produced it completed successfully."""
    ledger = AttemptLedger(tmp_path)
    released(ledger, "NOTE_D", VERSION_2)
    # the previous version's artifact restored over the published path
    atomic_write_json(ledger.published_path("NOTE_D"),
                      artifact("NOTE_D", VERSION_1), indent=2)

    ok, why = consumable(tmp_path, "NOTE_D")
    assert ok is False
    assert VERSION_1 in why and VERSION_2 in why, why


def test_completing_refuses_a_document_that_changed_mid_attempt(tmp_path):
    ledger = AttemptLedger(tmp_path)
    attempt = ledger.begin("NOTE_E", VERSION_1)
    with pytest.raises(AttemptWriteError) as caught:
        ledger.complete(attempt, artifact("NOTE_E", VERSION_2))
    assert "changed while it was being processed" in str(caught.value)
    assert not ledger.published_path("NOTE_E").exists()


def test_completing_refuses_a_result_that_declares_no_source_document(tmp_path):
    """An artifact with no document version could never be served as current, so
    it is refused where the reason is still known rather than published as a
    silently unusable success."""
    ledger = AttemptLedger(tmp_path)
    attempt = ledger.begin("NOTE_E2", VERSION_1)
    with pytest.raises(AttemptWriteError) as caught:
        ledger.complete(attempt, {"schema_id": "claim_bundle", "encounter": {}})
    assert "declares none" in str(caught.value)
    assert not ledger.published_path("NOTE_E2").exists()


def test_the_tombstone_is_recorded_even_when_the_stale_artifact_cannot_be_removed(
        tmp_path, monkeypatch):
    """The failure path of the failure path. Losing the FAILED record because the
    cleanup failed would leave the encounter looking merely interrupted — the
    weaker and less actionable of the two states — so the record lands first and
    the unremovable artifact is reported afterwards."""
    ledger = AttemptLedger(tmp_path)
    released(ledger, "NOTE_P", VERSION_1)
    attempt = ledger.begin("NOTE_P", VERSION_2)

    real_unlink = Path.unlink

    def stubborn(self, *args, **kwargs):
        if self.name.endswith("_results.json"):
            raise OSError(30, "Read-only file system")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", stubborn)
    with pytest.raises(AttemptWriteError):
        ledger.fail(attempt, error="OSError: No space left on device")

    monkeypatch.setattr(Path, "unlink", real_unlink)
    assert [r["state"] for r in ledger.history("NOTE_P")][-1] == "FAILED"
    ok, why = consumable(tmp_path, "NOTE_P")
    assert ok is False and "FAILED" in why


def test_a_system_retry_tombstone_is_visible_but_never_current(tmp_path):
    """A note whose processing failed keeps an artifact (an invisible note is its
    own silent failure) and is still not served as a result."""
    ledger = AttemptLedger(tmp_path)
    released(ledger, "NOTE_F", VERSION_1)
    attempt = ledger.begin("NOTE_F", VERSION_2)
    path = ledger.system_retry(attempt, {"processing_error": "dependency down"},
                               error="RuntimeError: dependency down")

    assert path.exists()
    assert json.loads(path.read_text())["processing_error"] == "dependency down"
    ok, why = consumable(tmp_path, "NOTE_F")
    assert ok is False and "SYSTEM_RETRY" in why


# --------------------------------------------------------------------------
# the ledger itself
# --------------------------------------------------------------------------

def test_every_attempt_is_retained_and_the_history_is_append_only(tmp_path):
    ledger = AttemptLedger(tmp_path)
    first = ledger.begin("NOTE_G", VERSION_1)
    ledger.complete(first, artifact("NOTE_G", VERSION_1))
    second = ledger.begin("NOTE_G", VERSION_2)
    ledger.complete(second, artifact("NOTE_G", VERSION_2))

    states = [(r["attempt_id"], r["state"]) for r in ledger.history("NOTE_G")]
    assert states == [
        (first.attempt_id, "IN_PROGRESS"), (first.attempt_id, "COMPLETED"),
        (second.attempt_id, "IN_PROGRESS"), (second.attempt_id, "COMPLETED"),
    ]
    root = tmp_path / "attempts" / "NOTE_G"
    retained = {p.name for p in root.glob("0*.json")}
    assert retained == {f"{first.attempt_id}.json", f"{second.attempt_id}.json"}
    superseded = json.loads((root / f"{first.attempt_id}.json").read_text())
    assert superseded["encounter"]["source_document"]["document_version"] == VERSION_1
    assert second.sequence == first.sequence + 1


def test_a_damaged_history_is_fatal_not_skipped(tmp_path):
    ledger = AttemptLedger(tmp_path)
    released(ledger, "NOTE_H", VERSION_1)
    history = tmp_path / "attempts" / "NOTE_H" / "ledger.jsonl"
    history.write_text(history.read_text() + '{"schema": "attempt_rec\n')

    ok, why = consumable(tmp_path, "NOTE_H")
    assert ok is False
    assert "damaged" in why, why
    with pytest.raises(AttemptLedgerError):
        ledger.history("NOTE_H")


def test_a_pointer_that_is_not_a_pointer_is_refused_rather_than_ignored(tmp_path):
    ledger = AttemptLedger(tmp_path)
    released(ledger, "NOTE_I", VERSION_1)
    (tmp_path / "attempts" / "NOTE_I" / "current.json").write_text('{"state": "COMPLETED"}')

    ok, why = consumable(tmp_path, "NOTE_I")
    assert ok is False
    assert "schema" in why


def test_a_result_file_with_no_attempt_record_is_refused_in_a_governed_directory(tmp_path):
    """The governance marker is the DIRECTORY. A file dropped beside real
    artifacts cannot be consumed just because nothing recorded an attempt for it."""
    ledger = AttemptLedger(tmp_path)
    released(ledger, "NOTE_J", VERSION_1)
    (tmp_path / "NOTE_INTRUDER_results.json").write_text(
        json.dumps(artifact("NOTE_INTRUDER", VERSION_1)))

    current = resolve_current(tmp_path)
    assert [r.document_id for r in current.results] == ["NOTE_J"]
    assert "no processing attempt is recorded" in current.refusals["NOTE_INTRUDER"]


def test_an_ungoverned_directory_reads_exactly_as_before(tmp_path):
    """`claims_registry export-gold` and the benchmark harness materialize result
    files with no pipeline attempt behind them; they must keep working."""
    (tmp_path / "NOTE_K_results.json").write_text(json.dumps(artifact("NOTE_K", VERSION_1)))
    (tmp_path / "all_results.json").write_text("[]")

    current = resolve_current(tmp_path)
    assert current.governed is False
    assert [r.path.name for r in current.results] == ["NOTE_K_results.json"]
    assert current.refusals == {}
    assert consumable(tmp_path, "NOTE_K") == (True, "")


def test_an_older_attempt_cannot_publish_over_a_newer_one(tmp_path):
    ledger = AttemptLedger(tmp_path)
    stale = ledger.begin("NOTE_L", VERSION_1)
    fresh = ledger.begin("NOTE_L", VERSION_2)
    ledger.complete(fresh, artifact("NOTE_L", VERSION_2))

    with pytest.raises(AttemptSuperseded):
        ledger.complete(stale, artifact("NOTE_L", VERSION_1))
    # and its failure record must not drag the pointer back either
    ledger.fail(stale, error="late failure")
    pointer = ledger.pointer("NOTE_L")
    assert pointer["attempt_id"] == fresh.attempt_id
    assert pointer["state"] == "COMPLETED"


def test_begin_invalidates_the_previous_result_when_it_cannot_record(tmp_path,
                                                                    monkeypatch):
    """The failure path of the fix itself. If the supersession cannot be
    recorded, the previous result must not survive on the strength of that
    failure — removing an entry is the one action a full filesystem still
    allows."""
    ledger = AttemptLedger(tmp_path)
    published = released(ledger, "NOTE_M", VERSION_1)

    import app.release.attempt_ledger as module
    monkeypatch.setattr(module.AttemptLedger, "_append_history",
                        lambda self, doc, record: (_ for _ in ()).throw(
                            AttemptWriteError("no space")))

    with pytest.raises(AttemptWriteError):
        ledger.begin("NOTE_M", VERSION_2)

    assert not published.exists()
    assert ledger.pointer("NOTE_M") is None
    ok, why = consumable(tmp_path, "NOTE_M")
    assert ok is False
    assert "the current-attempt pointer for this encounter is gone" in why, why


# --------------------------------------------------------------------------
# the atomic writer
# --------------------------------------------------------------------------

def test_the_atomic_writer_never_truncates_what_it_fails_to_replace(tmp_path,
                                                                   monkeypatch):
    target = tmp_path / "artifact.json"
    atomic_write_json(target, {"generation": 1}, indent=2)

    def boom(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_json(target, {"generation": 2}, indent=2)

    assert json.loads(target.read_text()) == {"generation": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["artifact.json"], (
        "a failed write left a temp file a later glob could pick up")


def test_the_document_version_convention_is_the_compilers_own(tmp_path):
    """One spelling of "sha256 of the document bytes". Two would make the
    completion check silently never match."""
    from app.ingestion.source_evidence import document_digest
    pdf = tmp_path / "note.pdf"
    pdf.write_bytes(b"%PDF-1.4 synthetic")
    assert document_version_of(pdf) == document_digest(pdf.read_bytes())


# --------------------------------------------------------------------------
# the consumers
# --------------------------------------------------------------------------

def test_the_registry_refuses_a_superseded_artifact_by_name(tmp_path):
    """Discriminating: the SAME artifact is skipped for a supersession reason
    while its attempt is open, and for an eligibility reason once it completes.
    A test that only checked "not recorded" would pass on any refusal at all."""
    from tools import claims_registry as reg

    ledger = AttemptLedger(tmp_path)
    released(ledger, "NOTE_N", VERSION_1)
    registry = tmp_path / "registry.jsonl"

    ledger.begin("NOTE_N", VERSION_2)
    stats = reg.ingest(tmp_path, registry)
    assert stats["recorded"] == 0
    assert "IN_PROGRESS" in stats["skip_reasons"]["NOTE_N"]

    # the control: complete the attempt and the SAME file is judged on its merits
    fresh = AttemptLedger(tmp_path)
    attempt = fresh.begin("NOTE_N", VERSION_1)
    fresh.complete(attempt, artifact("NOTE_N", VERSION_1))
    stats = reg.ingest(tmp_path, registry)
    assert stats["recorded"] == 0
    assert "IN_PROGRESS" not in stats["skip_reasons"]["NOTE_N"], (
        "the artifact is current now; any refusal must come from the claim, not "
        "from supersession")


def test_the_submitter_blocks_a_superseded_artifact_before_it_reads_it(tmp_path,
                                                                      monkeypatch):
    """Transmission is irreversible: the gate must run before the file is opened,
    and a registry event verified from an EARLIER attempt must not authorize it.

    The published artifact is REMOVED for this test, so the gate is the only
    thing that can produce a supersession reason: without it the submitter
    reports "result file ... not found", which is a different (and much weaker)
    refusal that would still have let a surviving stale file through.
    """
    from tools import claim_submitter as cs

    ledger = AttemptLedger(tmp_path)
    published = released(ledger, "NOTE_O", VERSION_1)
    registry = tmp_path / "registry.jsonl"
    registry.write_text(json.dumps({
        "event": "finalized", "document_id": "NOTE_O", "verification": "auto",
        "claim": {}, "encounter_context_fingerprint": ""}) + "\n")
    monkeypatch.setattr(cs, "REGISTRY_PATH", registry)
    monkeypatch.setattr(cs, "LEDGER_PATH", tmp_path / "submissions.jsonl")
    monkeypatch.setattr(cs, "DRYRUN_DIR", tmp_path / "submissions")
    # The practice configuration is a separate gate with its own tests; stubbed
    # so this one fails or passes for the ledger reason only.
    monkeypatch.setattr(cs, "load_practice_config", lambda: {})
    monkeypatch.setattr(cs, "validate_config", lambda cfg: [])

    ledger.begin("NOTE_O", VERSION_2)
    published.unlink()
    stats = cs.submit_all(tmp_path, dry_run=True)

    assert stats["submitted"] == 0
    assert stats["blocked"] == 1
    assert "IN_PROGRESS" in stats["docs"]["NOTE_O"], stats["docs"]["NOTE_O"]
    assert not (tmp_path / "submissions").exists()


# --------------------------------------------------------------------------
# issue #6 F8-R2 — an older attempt must never regain "current"
# --------------------------------------------------------------------------
#
# The reviewer's reproduction, verbatim: interleave B's `begin` so it lands
# immediately AFTER A's ownership check and BEFORE A's writes complete. Before
# the per-document lock this produced a current pointer of A / COMPLETED / v1 —
# the STALE document version — while B, the newer attempt, was still running,
# with a history reading A IN_PROGRESS, B IN_PROGRESS, A COMPLETED.
#
# These tests inject at exactly that point (`_owns`, the ownership check itself)
# rather than at a convenient nearby seam, because the whole finding is about
# what can happen between that call and the pointer write.


def lock_is_free(ledger: AttemptLedger, document_id: str) -> bool:
    """Is this encounter's transition lock genuinely released?

    A RAW `flock` from a SECOND open file description in this same process. The
    kernel denies that while any description still holds the lock, so a leaked
    file descriptor is detected — which re-acquiring through the module's own
    re-entrant registry would NOT detect, because re-entrancy would simply let
    the same holder back in.
    """
    path = ledger._lock_path(document_id)
    if not path.exists():
        return True
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def interleave_at_the_ownership_check(monkeypatch, run_once):
    """Fire `run_once()` the first time any attempt checks whether it still owns
    the pointer — the exact instruction the reviewer interleaved at."""
    import app.release.attempt_ledger as module
    real_owns = module.AttemptLedger._owns
    fired = []

    def owns_then_interleave(self, attempt):
        held = real_owns(self, attempt)
        if not fired:
            fired.append(True)
            run_once()
        return held

    monkeypatch.setattr(module.AttemptLedger, "_owns", owns_then_interleave)
    return fired


def test_a_newer_attempt_opening_mid_publish_leaves_the_older_one_superseded(
        tmp_path, monkeypatch):
    """THE F8-R2 reproduction, in-process: B.begin lands inside A's publish.

    Without the lock, B's `begin` completes inside the window (the join below
    returns), the pointer moves to B, and A's commit then lands ON TOP of it —
    A / COMPLETED / v1 becomes current. With it, B blocks until A's transition
    finishes, so whatever order the lock grants, the LAST word belongs to the
    newer attempt.
    """
    ledger = AttemptLedger(tmp_path)
    older = ledger.begin("NOTE_R2T", VERSION_1)

    newer = []
    opened = threading.Event()

    def open_the_newer_attempt():
        opened.set()
        newer.append(AttemptLedger(tmp_path).begin("NOTE_R2T", VERSION_2))

    thread = threading.Thread(target=open_the_newer_attempt, daemon=True)

    def interleave():
        thread.start()
        assert opened.wait(30), "the interleaved thread never started"
        # Pre-fix this join RETURNS: B's begin runs to completion right here.
        thread.join(timeout=1.0)

    interleave_at_the_ownership_check(monkeypatch, interleave)

    try:
        with contextlib.suppress(AttemptSuperseded):
            ledger.complete(older, artifact("NOTE_R2T", VERSION_1))
    finally:
        thread.join(timeout=30)
    assert not thread.is_alive(), "the newer attempt never finished opening"
    assert newer, "the newer attempt never opened"

    pointer = ledger.pointer("NOTE_R2T")
    assert pointer["attempt_id"] == newer[0].attempt_id, (
        "the OLDER attempt regained the current pointer over a newer one")
    assert pointer["state"] == "IN_PROGRESS"
    assert pointer["document_version"] == VERSION_2, (
        "the stale document version is current again")

    ok, why = consumable(tmp_path, "NOTE_R2T")
    assert ok is False, "a stale note version regained consumable status"
    assert "IN_PROGRESS" in why, why
    assert resolve_current(tmp_path).results == ()

    # ...and the history cannot read "A IN_PROGRESS, B IN_PROGRESS, A COMPLETED"
    records = ledger.history("NOTE_R2T")
    opened_at = min(i for i, r in enumerate(records)
                    if r["attempt_id"] == newer[0].attempt_id)
    terminal = [i for i, r in enumerate(records)
                if r["state"] != AttemptState.IN_PROGRESS.value]
    assert all(i < opened_at for i in terminal), (
        f"an attempt reached a terminal state AFTER a newer attempt opened: "
        f"{[(r['attempt_id'], r['state']) for r in records]}")
    assert older.attempt_id != newer[0].attempt_id


def test_a_newer_attempt_in_a_SECOND_PROCESS_is_not_overwritten_either(tmp_path,
                                                                      monkeypatch):
    """The same interleaving across a real process boundary.

    A second thread would be excluded by any in-process lock; the acceptance
    criterion is an INTERPROCESS one, so the newer attempt here is opened by a
    separate Python process. The proof that the lock crosses the boundary is
    that the child is still alive — blocked — a full second after it reached its
    `begin`, and completes promptly once the parent's transition releases.
    """
    ledger = AttemptLedger(tmp_path)
    older = ledger.begin("NOTE_R2P", VERSION_1)
    ready = tmp_path / "child-reached-begin"
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from pathlib import Path\n"
        "from app.release.attempt_ledger import AttemptLedger\n"
        f"ledger = AttemptLedger({str(tmp_path)!r})\n"
        f"Path({str(ready)!r}).write_text('ready')\n"
        f"attempt = ledger.begin('NOTE_R2P', {VERSION_2!r})\n"
        "print(attempt.attempt_id)\n")
    child = {}
    out = err = ""

    def interleave():
        proc = subprocess.Popen([sys.executable, "-c", script],
                                cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        child["proc"] = proc
        deadline = time.monotonic() + 120
        while not ready.exists() and time.monotonic() < deadline:
            if proc.poll() is not None:
                raise AssertionError(
                    f"the child exited before reaching begin: {proc.communicate()}")
            time.sleep(0.02)
        assert ready.exists(), "the child never reached its begin"
        time.sleep(1.0)
        assert proc.poll() is None, (
            "a SECOND PROCESS opened a newer attempt inside the older attempt's "
            "publish window — the lock is not interprocess")

    interleave_at_the_ownership_check(monkeypatch, interleave)

    try:
        with contextlib.suppress(AttemptSuperseded):
            ledger.complete(older, artifact("NOTE_R2P", VERSION_1))
    finally:
        proc = child.get("proc")
        if proc is not None:
            out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err
    newer_attempt_id = out.strip()
    assert newer_attempt_id, err

    pointer = ledger.pointer("NOTE_R2P")
    assert pointer["attempt_id"] == newer_attempt_id, (
        "the older attempt regained the current pointer over a newer PROCESS")
    assert pointer["state"] == "IN_PROGRESS"
    assert pointer["document_version"] == VERSION_2
    ok, why = consumable(tmp_path, "NOTE_R2P")
    assert ok is False and "IN_PROGRESS" in why, why


def test_a_late_failure_cannot_delete_a_newer_attempts_published_result(tmp_path):
    """The other half of "an older attempt may not publish over a newer one".

    `fail` used to write (or, with no tombstone, UNLINK) the published path
    before it ever looked at ownership. A stale attempt failing late therefore
    deleted a newer attempt's COMPLETED artifact out from under a pointer that
    still promised it — a completed claim turned into "its artifact is missing".
    """
    ledger = AttemptLedger(tmp_path)
    stale = ledger.begin("NOTE_R2F", VERSION_1)
    fresh = ledger.begin("NOTE_R2F", VERSION_2)
    published = ledger.complete(fresh, artifact("NOTE_R2F", VERSION_2))
    assert consumable(tmp_path, "NOTE_R2F") == (True, "")

    ledger.fail(stale, error="late failure, no tombstone")

    assert published.exists(), (
        "an older attempt's failure deleted the newer attempt's published result")
    payload = json.loads(published.read_text())
    assert payload["encounter"]["source_document"]["document_version"] == VERSION_2
    assert consumable(tmp_path, "NOTE_R2F") == (True, "")
    pointer = ledger.pointer("NOTE_R2F")
    assert pointer["attempt_id"] == fresh.attempt_id
    assert pointer["state"] == AttemptState.COMPLETED.value

    # and with a tombstone it must not OVERWRITE the newer result either
    ledger.fail(stale, error="late failure, with tombstone",
                tombstone={"processing_error": "output failed"})
    payload = json.loads(published.read_text())
    assert payload["encounter"]["source_document"]["document_version"] == VERSION_2
    assert consumable(tmp_path, "NOTE_R2F") == (True, "")

    # the failure is still recorded — the history says what happened, the
    # pointer says what is current, and only the owner may move the pointer
    failures = [r for r in ledger.history("NOTE_R2F")
                if r["attempt_id"] == stale.attempt_id
                and r["state"] == AttemptState.FAILED.value]
    assert len(failures) == 2
    assert all(r["superseded_by"] == fresh.attempt_id for r in failures)


def test_the_ordinary_sequential_attempt_is_unaffected_by_the_lock(tmp_path):
    """The common case: nothing concurrent, and re-runs still work normally."""
    ledger = AttemptLedger(tmp_path)
    first = ledger.begin("NOTE_R2S", VERSION_1)
    ledger.complete(first, artifact("NOTE_R2S", VERSION_1))
    assert consumable(tmp_path, "NOTE_R2S") == (True, "")
    assert lock_is_free(ledger, "NOTE_R2S")

    second = ledger.begin("NOTE_R2S", VERSION_2)
    assert second.sequence == first.sequence + 1
    assert consumable(tmp_path, "NOTE_R2S")[0] is False
    published = ledger.complete(second, artifact("NOTE_R2S", VERSION_2))
    assert consumable(tmp_path, "NOTE_R2S") == (True, "")
    assert json.loads(published.read_text())["encounter"]["source_document"][
        "document_version"] == VERSION_2
    assert lock_is_free(ledger, "NOTE_R2S")

    # a second encounter in the same directory never contends with the first
    other = ledger.begin("NOTE_R2S_OTHER", VERSION_1)
    ledger.complete(other, artifact("NOTE_R2S_OTHER", VERSION_1))
    current = resolve_current(tmp_path)
    assert sorted(r.document_id for r in current.results) == [
        "NOTE_R2S", "NOTE_R2S_OTHER"]


def test_attempts_opened_at_once_get_distinct_sequences_and_the_last_one_wins(
        tmp_path):
    """Concurrent `begin`s used to race on the sequence number too: each read the
    history, then all wrote. Under the lock they serialize, so the numbers are
    distinct and the pointer belongs to the one that opened last."""
    ledger = AttemptLedger(tmp_path)
    opened = []
    barrier = threading.Barrier(4, timeout=30)

    def open_one():
        barrier.wait()
        opened.append(AttemptLedger(tmp_path).begin("NOTE_R2Q", VERSION_1))

    threads = [threading.Thread(target=open_one, daemon=True) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive()

    assert sorted(a.sequence for a in opened) == [1, 2, 3, 4], opened
    assert len({a.attempt_id for a in opened}) == 4
    last = max(opened, key=lambda a: a.sequence)
    assert ledger.pointer("NOTE_R2Q")["attempt_id"] == last.attempt_id
    assert lock_is_free(ledger, "NOTE_R2Q")


def test_the_transition_lock_is_released_on_every_failure_path(tmp_path):
    """A lock that a failing transition keeps is worse than the bug it fixes: it
    wedges the encounter for every later attempt. Each terminal transition's
    failure paths are exercised and the lock checked from a second descriptor."""
    import app.release.attempt_ledger as module

    ledger = AttemptLedger(tmp_path)
    stale = ledger.begin("NOTE_R2L", VERSION_1)
    fresh = ledger.begin("NOTE_R2L", VERSION_2)
    assert lock_is_free(ledger, "NOTE_R2L"), "begin kept the lock"

    # 1. a refused publish (the superseded path)
    with pytest.raises(AttemptSuperseded):
        ledger.complete(stale, artifact("NOTE_R2L", VERSION_1))
    assert lock_is_free(ledger, "NOTE_R2L"), "a superseded publish kept the lock"

    # 2. a completion refused for a changed document version, before any write
    with pytest.raises(AttemptWriteError):
        ledger.complete(fresh, artifact("NOTE_R2L", VERSION_1))
    assert lock_is_free(ledger, "NOTE_R2L")

    # 3. an artifact write that fails mid-transition (the round-8 fault injection)
    real = module.atomic_write_json
    with pytest.MonkeyPatch.context() as patch:
        def full_disk(path, payload, *, indent=None):
            if isinstance(payload, dict) and payload.get("schema") == module.POINTER_SCHEMA:
                return real(path, payload, indent=indent)
            raise OSError(28, "No space left on device")
        patch.setattr(module, "atomic_write_json", full_disk)
        with pytest.raises(AttemptWriteError):
            ledger.complete(fresh, artifact("NOTE_R2L", VERSION_2))
        assert lock_is_free(ledger, "NOTE_R2L"), "a failed write kept the lock"
        with pytest.raises(AttemptWriteError):
            ledger.system_retry(fresh, artifact("NOTE_R2L", VERSION_2),
                                error="dependency down")
        assert lock_is_free(ledger, "NOTE_R2L"), "a failed system_retry kept the lock"

    # 4. a `fail` whose stale artifact can neither be replaced nor removed
    ledger.complete(fresh, artifact("NOTE_R2L", VERSION_2))
    with pytest.MonkeyPatch.context() as patch:
        real_unlink = Path.unlink

        def stubborn(self, *args, **kwargs):
            if self.name.endswith("NOTE_R2L_results.json"):
                raise OSError(13, "Permission denied")
            return real_unlink(self, *args, **kwargs)
        patch.setattr(Path, "unlink", stubborn)
        with pytest.raises(AttemptWriteError):
            ledger.fail(fresh, error="output failed")
    assert lock_is_free(ledger, "NOTE_R2L"), "a raising fail() kept the lock"

    # 5. a `begin` whose supersession cannot be recorded at all
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module.AttemptLedger, "_append_history",
                      lambda self, doc, record: (_ for _ in ()).throw(
                          AttemptWriteError("no space")))
        with pytest.raises(AttemptWriteError):
            ledger.begin("NOTE_R2L", VERSION_2)
    assert lock_is_free(ledger, "NOTE_R2L"), "a failed begin kept the lock"

    # and the encounter is still usable afterwards — not wedged
    after = ledger.begin("NOTE_R2L", VERSION_2)
    ledger.complete(after, artifact("NOTE_R2L", VERSION_2))
    assert consumable(tmp_path, "NOTE_R2L") == (True, "")


def test_a_holder_that_never_releases_is_refused_on_a_deadline(tmp_path,
                                                               monkeypatch):
    """The one thing the kernel cannot do for us. An flock dies with its holder,
    so a crash cannot leave a stale lock — but a LIVE, wedged holder can keep it
    forever, and a blocking acquire would hang the whole batch with no
    diagnosis. Acquisition is bounded and refuses by name instead.
    """
    import app.release.attempt_ledger as module
    ledger = AttemptLedger(tmp_path)
    first = ledger.begin("NOTE_R2W", VERSION_1)
    ledger.complete(first, artifact("NOTE_R2W", VERSION_1))

    monkeypatch.setattr(module, "LOCK_TIMEOUT_S", 0.2)
    wedged = os.open(str(ledger._lock_path("NOTE_R2W")), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(wedged, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        with pytest.raises(AttemptLockTimeout) as raised:
            ledger.begin("NOTE_R2W", VERSION_2)
        assert time.monotonic() - started < 15, "the acquire was not bounded"
        assert "still held after" in str(raised.value), raised.value
    finally:
        fcntl.flock(wedged, fcntl.LOCK_UN)
        os.close(wedged)

    # A refused acquire must NOT destroy the live holder's state: a holder that
    # exists has, by this ledger's ordering, already superseded any earlier
    # success, so there is nothing to invalidate and a running attempt's own
    # record must survive.
    assert ledger.pointer("NOTE_R2W")["attempt_id"] == first.attempt_id
    # ...and once released, the encounter opens normally again.
    later = ledger.begin("NOTE_R2W", VERSION_2)
    assert later.sequence == first.sequence + 1


def test_a_thread_that_never_releases_is_refused_on_a_deadline_too(tmp_path,
                                                                  monkeypatch):
    """The same bound, one level up. `flock` cannot exclude two threads of one
    process (it is held by the open file description), so a thread guard does
    that — and a guard acquired without a deadline would simply relocate the
    hang the flock deadline exists to remove."""
    import app.release.attempt_ledger as module
    ledger = AttemptLedger(tmp_path)
    first = ledger.begin("NOTE_R2G", VERSION_1)
    ledger.complete(first, artifact("NOTE_R2G", VERSION_1))

    monkeypatch.setattr(module, "LOCK_TIMEOUT_S", 0.2)
    holding, release = threading.Event(), threading.Event()

    def wedge():
        with module._exclusive(ledger._lock_path("NOTE_R2G")):
            holding.set()
            release.wait(60)

    thread = threading.Thread(target=wedge, daemon=True)
    thread.start()
    try:
        assert holding.wait(30), "the wedging thread never took the lock"
        started = time.monotonic()
        with pytest.raises(AttemptLockTimeout) as raised:
            ledger.begin("NOTE_R2G", VERSION_2)
        assert time.monotonic() - started < 15, "the guard acquire was not bounded"
        assert "another thread of this process" in str(raised.value), raised.value
    finally:
        release.set()
        thread.join(timeout=30)
    assert not thread.is_alive()

    assert ledger.pointer("NOTE_R2G")["attempt_id"] == first.attempt_id
    later = ledger.begin("NOTE_R2G", VERSION_2)
    assert later.sequence == first.sequence + 1


# --------------------------------------------------------------------------
# issue #6 F8-R2 REOPENED — crash ordering, not just writer interleaving
# --------------------------------------------------------------------------
#
# The round-9 lock serializes two LIVE writers. It says nothing about a single
# process dying between the two durable writes one transition makes. `begin`
# appended its IN_PROGRESS history record BEFORE moving the pointer, so a crash
# in that window left a recorded newer attempt beside a pointer still naming the
# previous COMPLETED result — and that result stayed consumable. The reviewer's
# reproduction needed no concurrency at all:
#
#     complete v1                       pointer -> COMPLETED / v1
#     begin v2, history append lands     history -> IN_PROGRESS / v2
#     <process dies here>
#     re-open as a consumer              consumable() == (True, '')   <- the bug
#
# These tests kill a REAL process with SIGKILL at every durable-write boundary
# of every transition, and separately prove the consumer-side cross-check that
# refuses a stale pointer even when the ordering is bypassed entirely.

#: Runs one transition for v2 in a child process that SIGKILLs itself when
#: exactly `kill_at` durable writes have completed. Instruments the two
#: primitives every durable write in this module goes through — the atomic JSON
#: writer (artifacts AND the pointer) and the history append — so the boundaries
#: are discovered from the code under test rather than assumed from it.
CRASH_CHILD = r"""
import json, os, signal, sys

(results_dir, document_id, transition, kill_at, ledger_path, version_2,
 repo_root) = sys.argv[1:8]
kill_at = int(kill_at)
sys.path.insert(0, repo_root)

from app.release import attempt_ledger as module

armed = [False]
writes = []


def counted(fn, label):
    def wrapper(*args, **kwargs):
        if armed[0] and len(writes) == kill_at:
            os.kill(os.getpid(), signal.SIGKILL)
        result = fn(*args, **kwargs)
        if armed[0]:
            writes.append(label(args))
            with open(ledger_path, "w") as fh:
                fh.write(json.dumps(writes))
                fh.flush()
                os.fsync(fh.fileno())
        return result
    return wrapper


module.atomic_write_json = counted(
    module.atomic_write_json, lambda args: os.path.basename(str(args[0])))
module.AttemptLedger._append_history = counted(
    module.AttemptLedger._append_history, lambda args: "ledger.jsonl")


def payload(version):
    return {"schema_id": "claim_bundle",
            "encounter": {"document_id": document_id,
                          "encounter_id": document_id,
                          "source_document": {"document_version": version}},
            "release": {"holds": [], "producer_releasable": True}}


ledger = module.AttemptLedger(results_dir)
attempt = None
if transition != "begin":
    # setup, deliberately UNCOUNTED: the boundaries under test are the ones
    # inside the transition named on the command line.
    attempt = ledger.begin(document_id, version_2)
armed[0] = True
if transition == "begin":
    ledger.begin(document_id, version_2)
elif transition == "complete":
    ledger.complete(attempt, payload(version_2))
elif transition == "system_retry":
    ledger.system_retry(attempt, {"processing_error": "dependency down"},
                        error="dependency down")
elif transition == "fail":
    ledger.fail(attempt, error="output failed",
                tombstone={"processing_error": "output failed"})
else:
    raise SystemExit("unknown transition " + transition)
print(json.dumps(writes))
"""


def crash_child(tmp_path, transition, kill_at, document_id="NOTE_R2C"):
    """A fresh encounter with a COMPLETED, consumable v1 — then the crash."""
    root = tmp_path / f"{transition}-{kill_at}"
    root.mkdir()
    released(AttemptLedger(root), document_id, VERSION_1)
    assert consumable(root, document_id) == (True, ""), (
        "precondition: the v1 claim is consumable before the crashing attempt")
    writes_path = root / "writes.json"
    proc = subprocess.run(
        [sys.executable, "-c", CRASH_CHILD, str(root), document_id, transition,
         str(kill_at), str(writes_path), VERSION_2, str(REPO_ROOT)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    done = json.loads(writes_path.read_text()) if writes_path.exists() else []
    return root, proc, done


def durable_state(root, document_id="NOTE_R2C"):
    """What survived the kill, read exactly as a fresh consumer would."""
    ledger = AttemptLedger(root)
    try:
        history = ledger.history(document_id)
    except AttemptLedgerError:
        history = []
    try:
        pointer = ledger.pointer(document_id)
    except AttemptLedgerError:
        pointer = None
    try:
        declared = json.loads(ledger.published_path(document_id).read_text())[
            "encounter"]["source_document"]["document_version"]
    except (OSError, ValueError, KeyError, TypeError):
        declared = ""
    ok, why = consumable(root, document_id)
    return history, pointer, declared, ok, why


@pytest.mark.parametrize("transition",
                         ["begin", "complete", "system_retry", "fail"])
def test_a_crash_at_every_write_boundary_never_leaves_the_old_claim_consumable(
        tmp_path, transition):
    """SIGKILL at EVERY durable-write boundary of EVERY transition.

    The invariant asserted at each one is the module's own, stated as the
    reviewer stated it: once ANY durable record of the newer attempt exists —
    in the pointer, in the history, or in both — the previous claim is never
    consumable again. The boundary count is discovered by running the
    transition once without killing anything, so a transition that grows a
    write in future is covered without editing this test.
    """
    root, proc, done = crash_child(tmp_path, transition, -1)
    assert proc.returncode == 0, proc.stderr
    boundaries = len(done)
    assert boundaries >= 2, (transition, done)
    if transition == "begin":
        assert done == ["current.json", "ledger.jsonl"], (
            "the POINTER must be the first durable write an attempt makes; "
            "appending the history record first is the F8-R2 crash window")

    seen_recorded = []
    for kill_at in range(boundaries + 1):
        root, proc, done = crash_child(tmp_path, transition, kill_at)
        if kill_at < boundaries:
            assert proc.returncode == -9, (
                f"the child was not killed at boundary {kill_at}: "
                f"{proc.returncode} {proc.stderr}")
            assert len(done) == kill_at, (kill_at, done)
        else:
            assert proc.returncode == 0, proc.stderr

        history, pointer, declared, ok, why = durable_state(root)
        recorded = [r for r in history if r["sequence"] >= 2]
        newer_exists = bool(recorded) or bool(
            pointer and pointer["sequence"] >= 2)
        seen_recorded.append(newer_exists)

        assert not (newer_exists and ok and declared == VERSION_1), (
            f"{transition} killed at boundary {kill_at}: the v1 claim is still "
            f"consumable while a v2 attempt is durably recorded — pointer="
            f"{pointer and (pointer['state'], pointer['sequence'])}, newest "
            f"history={history[-1]['state'] if history else None}")
        if ok:
            # the ONLY shape a consumable result may have, checked rather than
            # assumed: the pointer completed, nothing newer is recorded, and the
            # published bytes are that attempt's own document version.
            assert pointer["state"] == AttemptState.COMPLETED.value, why
            assert not [r for r in history
                        if r["sequence"] > pointer["sequence"]]
            assert declared == pointer["document_version"]
        if transition == "begin":
            assert newer_exists is (kill_at >= 1), (
                f"a begin killed at boundary {kill_at} left the supersession "
                f"unrecorded; pointer={pointer}")
            if newer_exists:
                assert ok is False, why
                assert "IN_PROGRESS" in why, why
            else:
                # THE CONTROL the whole test rests on: killed before its first
                # durable write, the attempt never started and the v1 claim IS
                # still consumable. Without this the assertions above would be
                # satisfiable by a harness that simply broke the encounter.
                assert (ok, why) == (True, "")
                assert declared == VERSION_1

    assert seen_recorded[0] is (transition != "begin"), seen_recorded
    assert seen_recorded[-1] is True, (
        "the harness never produced a durable v2 record — it would pass "
        "vacuously")


def test_a_stale_pointer_is_refused_when_the_history_proves_a_newer_attempt(
        tmp_path):
    """The consumer-side cross-check — the second, independent layer.

    Codex's F8-R2 reproduction verbatim, in the state it leaves behind: pointer
    COMPLETED / v1, newest history record IN_PROGRESS / v2, the v1 artifact
    still sitting where every consumer reads it. `begin` can no longer PRODUCE
    that state, so it is written by hand here — which is the point. This is the
    layer that still refuses if a future write path, an older build, or a script
    that never took the lock reintroduces the ordering, and it is the only test
    that can reach it, which is what proves it is not dead code.
    """
    ledger = AttemptLedger(tmp_path)
    published = released(ledger, "NOTE_R2X", VERSION_1)
    assert consumable(tmp_path, "NOTE_R2X") == (True, "")
    pointer_before = ledger.pointer("NOTE_R2X")

    ledger._append_history("NOTE_R2X", {
        "schema": "attempt_record/1",
        "document_id": "NOTE_R2X",
        "document_version": VERSION_2,
        "attempt_id": "0002-c0ffee000000",
        "sequence": 2,
        "state": AttemptState.IN_PROGRESS.value,
        "opened_at": "2026-01-01T00:00:00+00:00",
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "error": "",
    })

    assert ledger.pointer("NOTE_R2X") == pointer_before, (
        "precondition: the pointer was NOT moved — that is the whole scenario")
    assert pointer_before["state"] == AttemptState.COMPLETED.value
    assert ledger.history("NOTE_R2X")[-1]["state"] == AttemptState.IN_PROGRESS.value
    assert published.exists()

    ok, why = consumable(tmp_path, "NOTE_R2X")
    assert ok is False, "a recorded newer attempt left the stale claim consumable"
    assert "0002-c0ffee000000" in why and "sequence 2" in why, why
    current = resolve_current(tmp_path)
    assert current.results == ()
    assert "NOTE_R2X" in current.refusals

    # DISCRIMINATING: the check refuses a pointer the history OUTRANKS, not any
    # encounter with more than one attempt. A real attempt opened after that
    # record allocates a number above it and is served normally.
    later = ledger.begin("NOTE_R2X", VERSION_2)
    assert later.sequence == 3, later
    ledger.complete(later, artifact("NOTE_R2X", VERSION_2))
    assert consumable(tmp_path, "NOTE_R2X") == (True, "")


def test_a_pointer_only_attempt_is_recovered_without_reusing_its_sequence(
        tmp_path):
    """The partial state pointer-first can leave, and its recovery.

    A crash between the pointer write and the history append leaves an attempt
    the pointer names and the history never recorded. Nothing is consumable
    (the pointer is IN_PROGRESS), and the NEXT attempt must not reuse that
    number: the sequence is what orders attempts, and two attempts sharing one
    would make the supersession cross-check unable to tell them apart.
    """
    ledger = AttemptLedger(tmp_path)
    published = released(ledger, "NOTE_R2Y", VERSION_1)
    orphan = Attempt(document_id="NOTE_R2Y", document_version=VERSION_2,
                     attempt_id="0002-abcdef012345", sequence=2,
                     opened_at="2026-01-01T00:00:00+00:00")
    ledger._write_pointer("NOTE_R2Y", ledger._pointer_record(
        orphan, AttemptState.IN_PROGRESS))

    assert [r["sequence"] for r in ledger.history("NOTE_R2Y")] == [1, 1], (
        "precondition: the history never recorded attempt 2")
    ok, why = consumable(tmp_path, "NOTE_R2Y")
    assert ok is False and "IN_PROGRESS" in why, why
    assert published.exists()

    recovered = ledger.begin("NOTE_R2Y", VERSION_2)
    assert recovered.sequence == 3, (
        "the next attempt reused the sequence a pointer-only attempt already "
        "published")
    ledger.complete(recovered, artifact("NOTE_R2Y", VERSION_2))
    assert consumable(tmp_path, "NOTE_R2Y") == (True, "")
    assert ledger.pointer("NOTE_R2Y")["sequence"] == 3


def test_a_record_that_cannot_be_ordered_is_refused_rather_than_read_as_zero(
        tmp_path):
    """An unreadable sequence is a damaged record, not a zero.

    Reading it as 0 would make the NEWEST attempt sort below every other record
    — inverting the exact comparison supersession depends on — so it refuses,
    the same position `history()` takes on a torn line.
    """
    ledger = AttemptLedger(tmp_path)
    released(ledger, "NOTE_R2Z", VERSION_1)
    history = tmp_path / "attempts" / "NOTE_R2Z" / "ledger.jsonl"
    history.write_text(history.read_text() + json.dumps({
        "schema": "attempt_record/1", "document_id": "NOTE_R2Z",
        "attempt_id": "0002-badsequence", "sequence": "not-a-number",
        "state": AttemptState.IN_PROGRESS.value}) + "\n")

    ok, why = consumable(tmp_path, "NOTE_R2Z")
    assert ok is False
    assert "non-numeric sequence" in why, why


def test_begin_invalidates_when_the_SUPERSESSION_ITSELF_cannot_be_written(
        tmp_path, monkeypatch):
    """The failure path of the write that now goes FIRST.

    Moving the pointer IS the supersession. When that write is the one that
    fails there is no durable record that the encounter was reopened at all, so
    — exactly as when the history append fails afterwards — the previous result
    must not survive on the strength of that failure. Its sibling (the history
    append failing AFTER the pointer moved) is covered by
    `test_begin_invalidates_the_previous_result_when_it_cannot_record`; between
    them both of `begin`'s fallible writes are exercised.
    """
    ledger = AttemptLedger(tmp_path)
    published = released(ledger, "NOTE_R2V", VERSION_1)

    import app.release.attempt_ledger as module
    real = module.atomic_write_json

    def full_disk(path, payload, *, indent=None):
        if isinstance(payload, dict) and payload.get("schema") == module.POINTER_SCHEMA:
            raise OSError(28, "No space left on device")
        return real(path, payload, indent=indent)

    monkeypatch.setattr(module, "atomic_write_json", full_disk)
    with pytest.raises(AttemptWriteError):
        ledger.begin("NOTE_R2V", VERSION_2)
    monkeypatch.undo()

    assert not published.exists()
    assert ledger.pointer("NOTE_R2V") is None
    assert [r["state"] for r in ledger.history("NOTE_R2V")] == [
        "IN_PROGRESS", "COMPLETED"], (
        "the refused attempt appended a history record even though its pointer "
        "never landed — the ordering is what keeps that state unreachable")
    ok, why = consumable(tmp_path, "NOTE_R2V")
    assert ok is False
    assert "the current-attempt pointer for this encounter is gone" in why, why
