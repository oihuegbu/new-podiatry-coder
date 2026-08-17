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
    the tools that materialize result files themselves are unaffected.

No medical code appears in this file; every payload is a synthetic stand-in for
an artifact shape.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.release.attempt_ledger import (  # noqa: E402
    AttemptLedger, AttemptLedgerError, AttemptState, AttemptSuperseded,
    AttemptWriteError, atomic_write_json, consumable, document_version_of,
    resolve_current)

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
