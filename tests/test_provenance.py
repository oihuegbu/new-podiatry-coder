"""Phase-0 provenance kernel: evidence anchoring + relation identity/merge.

Failure-path first: the point of anchoring is to REJECT a plausible-but-non-verbatim
quotation, and the point of the relation kernel is that a re-asserted edge accumulates
support (never duplicates) while any state disagreement collapses to UNCERTAIN.
Agnostic — synthetic text and synthetic event ids, no medical code."""
import json

from claude_coder import provenance as prov
from claude_coder.models import (EvidenceSpan, RelationAssertion, RelationPredicate,
                                 RelationState, ClinicalFact, FactKind)


# ------------------------------------------------------------------ anchoring
def test_exact_quote_anchors_with_verified_offsets():
    note = "line one.\nThe prominence was removed with a saw.\nline three."
    quote = "The prominence was removed with a saw."
    span = prov.anchor_span(note, EvidenceSpan(text=quote))
    assert span.anchored is True
    assert note[span.start:span.end] == quote          # the invariant
    assert span.text_sha256 and len(span.text_sha256) == 64


def test_non_verbatim_quote_does_not_anchor():
    """A quotation the model paraphrased (not present verbatim) must NOT anchor — we
    never accept a plausible-looking quote as evidence."""
    note = "The prominence was removed with a saw."
    fabricated = "The bony prominence was excised using a surgical saw."   # not verbatim
    span = prov.anchor_span(note, EvidenceSpan(text=fabricated))
    assert span.anchored is False
    assert span.start is None and span.end is None
    assert span.text_sha256                              # still hashed for the audit trail


def test_anchor_offsets_is_exact_not_fuzzy():
    note = "alpha beta gamma"
    assert prov.anchor_offsets(note, "beta") == (6, 10)
    assert prov.anchor_offsets(note, "BETA") is None     # case-exact
    assert prov.anchor_offsets(note, "beta ") == (6, 11)
    assert prov.anchor_offsets(note, "") is None


def test_anchor_facts_and_report_coverage():
    note = "aaa performed service here. ddd."
    f = ClinicalFact(FactKind.PROCEDURE, "svc",
                     evidence=[EvidenceSpan(text="performed service here"),
                               EvidenceSpan(text="not in the note at all")])
    prov.anchor_facts(note, [f])
    assert [s.anchored for s in f.evidence] == [True, False]
    rep = prov.anchoring_report([f])
    assert rep["spans_total"] == 2 and rep["spans_anchored"] == 1
    assert rep["coverage"] == 0.5 and len(rep["unanchored"]) == 1


# ------------------------------------------------------------------ relation kernel
def _rel(subj, pred, obj, state=RelationState.ASSERTED, ev=None, conf=0.5, origins=None):
    return RelationAssertion(subject_event_id=subj, predicate=pred, object_event_id=obj,
                             state=state, evidence_span_ids=list(ev or []), confidence=conf,
                             assertion_origins=list(origins or []))


def test_relation_identity_is_content_addressed():
    a = _rel("E1", RelationPredicate.PART_OF, "E2")
    b = _rel("E1", RelationPredicate.PART_OF, "E2")
    c = _rel("E1", RelationPredicate.PART_OF, "E3")
    assert a.relation_id == b.relation_id                # same edge
    assert a.relation_id != c.relation_id                # different object -> different edge
    assert prov.relation_key("E1", "part_of", "E2") == a.relation_id


def test_reassertion_accumulates_support_not_duplicates():
    rels = [_rel("E1", RelationPredicate.PART_OF, "E2", ev=["s1"]),
            _rel("E1", RelationPredicate.PART_OF, "E2", ev=["s2"], conf=0.8)]
    merged = prov.merge_relations(rels)
    assert len(merged) == 1
    m = merged[0]
    assert m.support == 2
    assert set(m.evidence_span_ids) == {"s1", "s2"}      # unioned
    assert m.confidence == 0.8
    assert m.state is RelationState.ASSERTED
    # inputs not mutated
    assert rels[0].support == 1 and rels[0].evidence_span_ids == ["s1"]


def test_support_counts_assertions_but_independence_counts_origins():
    """Codex F6-R3: raw `support` is repetition; `independent_support` is SOURCES. One origin
    that repeats itself can never look like two sources agreeing, and an edge with no recorded
    origin scores zero rather than being trusted."""
    same = [_rel("E1", RelationPredicate.PART_OF, "E2", ev=["s1"], origins=["origin-a"]),
            _rel("E1", RelationPredicate.PART_OF, "E2", ev=["s2"], origins=["origin-a"])]
    (merged,) = prov.merge_relations(same)
    assert merged.support == 2 and merged.independent_support == 1
    assert merged.assertion_origins == ["origin-a"]

    two = [_rel("E1", RelationPredicate.PART_OF, "E2", ev=["s1"], origins=["origin-a"]),
           _rel("E1", RelationPredicate.PART_OF, "E2", ev=["s2"], origins=["origin-b"])]
    (both,) = prov.merge_relations(two)
    assert both.support == 2 and both.independent_support == 2
    assert both.assertion_origins == ["origin-a", "origin-b"]
    # inputs not mutated
    assert two[0].assertion_origins == ["origin-a"]

    blank = _rel("E1", RelationPredicate.PART_OF, "E2", origins=["", "  "])
    assert blank.independent_support == 0
    assert _rel("E1", RelationPredicate.PART_OF, "E2").independent_support == 0


def test_extraction_stamps_one_origin_per_call_and_it_is_reproducible():
    """The origin is the extraction CALL's recorded identity: reproducible for the same call
    (so certificates stay stable), different when the provider/profile differs, and different
    when the caller runs a genuinely separate pass."""
    from claude_coder import extraction
    note = "Service beta was performed for condition alpha."
    axes = {a: 0.9 for a in ("occurrence", "action", "evidence", "temporal",
                             "assertion", "experiencer")}
    payload = json.dumps({
        "facts": [{"fact_id": "D", "kind": "diagnosis", "description": "condition alpha",
                   "attributes": {}, "disposition": "performed_today",
                   "evidence": ["condition alpha"], "confidence": 0.9,
                   "axis_confidence": axes}],
        "relations": []})
    profile = {"provider": "provider-one", "model": "profile-one"}
    a = extraction.extract_note(note, lambda _s, _u: payload, model_profile=profile).origin
    b = extraction.extract_note(note, lambda _s, _u: payload, model_profile=profile).origin
    assert a.origin_id == b.origin_id                      # same call -> same origin
    other = extraction.extract_note(note, lambda _s, _u: payload,
                                    model_profile={"provider": "provider-two"}).origin
    assert other.origin_id != a.origin_id                  # different provider -> independent
    run2 = extraction.extract_note(note, lambda _s, _u: payload, run_id="pass-2",
                                   model_profile=profile).origin
    assert run2.origin_id != a.origin_id                   # separate run -> independent
    assert a.as_record()["prompt_sha256"] and a.as_record()["schema_version"]
    assert "secret" not in json.dumps(a.as_record()).lower()


def test_conflicting_states_collapse_to_uncertain():
    """A relationship asserted by one pass and negated by another must NOT survive as a
    confident PART_OF — it collapses to UNCERTAIN (Phase-1 routes that to a hold, never a
    guessed demotion)."""
    rels = [_rel("E1", RelationPredicate.PART_OF, "E2", state=RelationState.ASSERTED),
            _rel("E1", RelationPredicate.PART_OF, "E2", state=RelationState.NEGATED)]
    merged = prov.merge_relations(rels)
    assert len(merged) == 1 and merged[0].state is RelationState.UNCERTAIN


def test_distinct_edges_preserved():
    rels = [_rel("E1", RelationPredicate.PART_OF, "E2"),
            _rel("E1", RelationPredicate.USED_IN, "E2"),
            _rel("E3", RelationPredicate.PART_OF, "E2")]
    assert len(prov.merge_relations(rels)) == 3


def test_null_repository_is_noop():
    prov.NullAuditRepository().append("enc", "kind", {"x": 1})   # must not raise


# ---- Phase 3: durable, hash-chained, append-only SQLite provenance store ----------------
def test_sqlite_audit_durable_append_and_chain(tmp_path):
    from claude_coder.provenance import SqliteAuditRepository
    repo = SqliteAuditRepository(tmp_path / "prov.db")
    h1 = repo.append("enc1", "kindA", {"x": 1})
    h2 = repo.append("enc1", "kindB", {"y": 2})
    assert h1 and h2 and h1 != h2
    recs = repo.records("enc1")
    assert len(recs) == 2
    assert recs[0]["record"] == {"x": 1} and recs[1]["record"] == {"y": 2}
    assert recs[1]["previous_record_sha256"] == recs[0]["record_sha256"]   # hash-chained
    assert recs[0]["control_mode"] == "ENFORCED_FAIL_CLOSED"
    # durable across a fresh handle (a separate connection sees committed rows)
    assert len(SqliteAuditRepository(tmp_path / "prov.db").records("enc1")) == 2
    # per-encounter isolation
    repo.append("enc2", "kindA", {"z": 3})
    assert len(repo.records("enc2")) == 1 and len(repo.records()) == 3


def test_sqlite_audit_is_append_only(tmp_path):
    import sqlite3
    from claude_coder.provenance import SqliteAuditRepository
    dbp = tmp_path / "prov.db"
    SqliteAuditRepository(dbp).append("e", "k", {"a": 1})
    conn = sqlite3.connect(str(dbp))
    try:
        for sql in ("UPDATE audit_log SET kind='x'", "DELETE FROM audit_log"):
            raised = False
            try:
                conn.execute(sql)
                conn.commit()
            except sqlite3.Error:
                raised = True
            assert raised, f"append-only trigger did not block: {sql}"
    finally:
        conn.close()


def test_sqlite_audit_strict_fails_closed_else_swallows(tmp_path):
    from claude_coder.provenance import SqliteAuditRepository
    bad = tmp_path                                   # a directory cannot open as a db file
    raised = False
    try:
        SqliteAuditRepository(bad, strict=True).append("e", "k", {"a": 1})
    except Exception:
        raised = True
    assert raised                                    # strict -> raise -> pipeline SYSTEM_HOLD
    assert SqliteAuditRepository(bad, strict=False).append("e", "k", {"a": 1}) == ""


def test_sqlite_audit_record_hash_verifies(tmp_path):
    import json
    from claude_coder.provenance import SqliteAuditRepository, _sha
    repo = SqliteAuditRepository(tmp_path / "prov.db")
    repo.append("e", "k", {"a": 1})
    rec = repo.records("e")[0]
    entry = {"encounter_id": rec["encounter_id"], "kind": rec["kind"],
             "recorded_at": rec["recorded_at"], "control_mode": rec["control_mode"],
             "previous_record_sha256": rec["previous_record_sha256"], "record": rec["record"]}
    assert _sha(json.dumps(entry, sort_keys=True, default=str)) == rec["record_sha256"]


# ---- Codex F6-R4: concurrent writers must not fork the per-encounter hash chain ----------
def test_sqlite_audit_concurrent_writers_do_not_fork_chain(tmp_path):
    import threading
    from claude_coder.provenance import SqliteAuditRepository
    dbp = tmp_path / "prov.db"
    SqliteAuditRepository(dbp).append("enc", "seed", {"n": 0})   # a shared predecessor
    barrier = threading.Barrier(2)
    results: dict[int, str] = {}

    def worker(i):
        repo = SqliteAuditRepository(dbp)
        barrier.wait()                                            # force interleaving
        results[i] = repo.append("enc", f"k{i}", {"n": i})

    threads = [threading.Thread(target=worker, args=(i,)) for i in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    repo = SqliteAuditRepository(dbp)
    recs = repo.records("enc")
    assert len(recs) == 3
    assert results[1] and results[2] and results[1] != results[2]
    assert repo.verify_chain("enc") == []                        # no fork
    # serialized (not forked): the two appends reference DIFFERENT predecessors
    prevs = [r["previous_record_sha256"] for r in recs[1:]]
    assert prevs[0] != prevs[1]


def test_sqlite_audit_verify_chain_detects_tampering(tmp_path):
    import sqlite3
    from claude_coder.provenance import SqliteAuditRepository
    dbp = tmp_path / "prov.db"
    repo = SqliteAuditRepository(dbp)
    repo.append("enc", "a", {"x": 1})
    assert repo.verify_chain() == []
    # INSERT (allowed by the append-only triggers) a forged row with a wrong hash
    conn = sqlite3.connect(str(dbp))
    conn.execute(
        "INSERT INTO audit_log(encounter_id, kind, recorded_at, control_mode,"
        " previous_record_sha256, record_json, record_sha256) VALUES (?,?,?,?,?,?,?)",
        ("enc", "forged", "2026-01-01T00:00:00+00:00", "ENFORCED_FAIL_CLOSED",
         "deadbeef", "{}", "not-the-real-hash"))
    conn.commit()
    conn.close()
    problems = repo.verify_chain("enc")
    assert any("hash mismatch" in p for p in problems)
    assert any("broken/forked" in p for p in problems)


# ---- Codex F6-R4-A: external witness detects tail truncation / missing terminal record -----
def test_verify_chain_detects_tail_truncation(tmp_path):
    import sqlite3
    from claude_coder.provenance import SqliteAuditRepository
    dbp = tmp_path / "prov.db"
    repo = SqliteAuditRepository(dbp)
    repo.append("enc", "a", {"x": 1})
    repo.append("enc", "b", {"x": 2})               # terminal record
    assert repo.verify_chain("enc") == []
    # an attacker drops the append-only delete guard and removes the FINAL row; the remaining
    # prefix still self-verifies, but the EXTERNAL witness still records 2 heads.
    conn = sqlite3.connect(str(dbp))
    conn.execute("DROP TRIGGER audit_no_delete")
    conn.execute("DELETE FROM audit_log WHERE seq=(SELECT MAX(seq) FROM audit_log)")
    conn.commit()
    conn.close()
    problems = repo.verify_chain("enc")
    assert any("truncation" in p or "terminal" in p for p in problems)


# ---- Codex F6-R4-A round 3: the witness itself is mandatory, sealed, chained and atomic ----
def _repo(tmp_path, name="prov.db"):
    from claude_coder.provenance import SqliteAuditRepository
    return SqliteAuditRepository(tmp_path / name)


def _seeded(tmp_path):
    repo = _repo(tmp_path)
    repo.append("enc", "a", {"x": 1})
    repo.append("enc", "b", {"x": 2})
    assert repo.verify_chain("enc") == []
    return repo


def test_deleted_witness_is_an_integrity_failure_not_success(tmp_path):
    """Deleting the sidecar used to return {} -> `verify_chain` said 'intact'. It must now be
    a loud integrity failure, because otherwise deleting the witness and truncating the log
    reopens the original blind spot."""
    repo = _seeded(tmp_path)
    repo._witness_path().unlink()
    problems = repo.verify_chain("enc")
    assert problems and any("missing" in p for p in problems)


def test_deleted_witness_plus_terminal_row_deletion_is_detected(tmp_path):
    import sqlite3
    repo = _seeded(tmp_path)
    repo._witness_path().unlink()
    conn = sqlite3.connect(str(repo.db_path))
    conn.execute("DROP TRIGGER audit_no_delete")
    conn.execute("DELETE FROM audit_log WHERE seq=(SELECT MAX(seq) FROM audit_log)")
    conn.commit()
    conn.close()
    assert repo.verify_chain("enc")                     # not silently 'intact'


def test_empty_witness_is_an_integrity_failure(tmp_path):
    repo = _seeded(tmp_path)
    repo._witness_path().write_text("")
    problems = repo.verify_chain("enc")
    assert problems and any("empty" in p for p in problems)


def test_corrupted_witness_line_is_reported_not_skipped(tmp_path):
    repo = _seeded(tmp_path)
    wp = repo._witness_path()
    lines = wp.read_text().splitlines()
    lines[0] = "{not json"
    wp.write_text("\n".join(lines) + "\n")
    problems = repo.verify_chain("enc")
    assert any("corrupt" in p for p in problems)


def test_edited_witness_entry_fails_its_seal(tmp_path):
    import json
    repo = _seeded(tmp_path)
    wp = repo._witness_path()
    lines = wp.read_text().splitlines()
    forged = json.loads(lines[-1])
    forged["record_sha256"] = "0" * 64                  # rewrite the witnessed head
    lines[-1] = json.dumps(forged, sort_keys=True)
    wp.write_text("\n".join(lines) + "\n")
    problems = repo.verify_chain("enc")
    assert any("seal mismatch" in p for p in problems)


def test_partial_witness_is_detected(tmp_path):
    """Truncating the witness tail (so it no longer covers every durable row) is detected by
    the row->witness direction of the anchoring."""
    repo = _seeded(tmp_path)
    wp = repo._witness_path()
    lines = wp.read_text().splitlines()
    wp.write_text(lines[0] + "\n")                      # drop the terminal witness entry
    problems = repo.verify_chain("enc")
    assert any("witness entry for this record is absent" in p for p in problems)


def test_duplicate_witness_entry_is_detected(tmp_path):
    repo = _seeded(tmp_path)
    wp = repo._witness_path()
    lines = wp.read_text().splitlines()
    wp.write_text("\n".join(lines + [lines[-1]]) + "\n")
    problems = repo.verify_chain("enc")
    assert problems and any("duplicate" in p or "chain link" in p for p in problems)


def test_witness_write_failure_leaves_no_committed_row(tmp_path):
    """The DB-commit vs witness-write boundary: the witness is fsynced BEFORE the commit, so
    a witness failure can never leave a committed audit row nothing vouches for."""
    import pytest
    repo = _repo(tmp_path)
    repo.append("enc", "a", {"x": 1})
    before = len(repo.records("enc"))

    def _boom(*_a, **_k):
        raise OSError("witness storage unavailable")

    repo._append_head_witness = _boom
    with pytest.raises(OSError):
        repo.append("enc", "b", {"x": 2})
    assert len(_repo(tmp_path).records("enc")) == before     # nothing committed
    assert _repo(tmp_path).verify_chain("enc") == []         # and still consistent


def test_commit_failure_after_witness_write_is_an_in_doubt_tail(tmp_path):
    """The other side of the boundary: a crash between the witness fsync and the commit is
    explicitly recoverable -- reported by name, and the writer may extend the chain once."""
    repo = _repo(tmp_path)
    repo.append("enc", "a", {"x": 1})
    repo._append_head_witness("enc", "0" * 64)               # simulate the lost commit
    problems = repo.verify_chain("enc")
    assert any("in-doubt" in p for p in problems)
    fresh = _repo(tmp_path)
    fresh.append("enc", "c", {"x": 3})                       # recovery: extension allowed once


def test_writer_refuses_to_extend_a_destroyed_witness(tmp_path):
    """Fail closed at WRITE time too: a writer must not quietly keep appending on top of a
    witness that was deleted or rewritten."""
    import pytest
    repo = _seeded(tmp_path)
    repo._witness_path().unlink()
    with pytest.raises(OSError):
        _repo(tmp_path).append("enc", "c", {"x": 3})
    # ...and the same for a witness whose tail was truncated rather than deleted
    repo2 = _repo(tmp_path, "trunc.db")
    repo2.append("enc", "a", {"x": 1})
    repo2.append("enc", "b", {"x": 2})
    wp = repo2._witness_path()
    wp.write_text(wp.read_text().splitlines()[0] + "\n")
    with pytest.raises(OSError):
        _repo(tmp_path, "trunc.db").append("enc", "d", {"x": 4})


def test_witness_seal_is_keyed_and_records_its_custody(tmp_path, monkeypatch):
    """An externally provisioned key puts custody outside the audit writer's own directory,
    so a process holding only the db + witness files cannot forge entries."""
    monkeypatch.setenv("PROVENANCE_WITNESS_KEY", "an-externally-provisioned-key")
    repo = _repo(tmp_path, "keyed.db")
    repo.append("enc", "a", {"x": 1})
    assert repo.verify_chain("enc") == []
    assert repo.witness_status()["key_custody"] == "external"
    assert not repo._witness_key_path().exists()             # no key written to disk
    monkeypatch.setenv("PROVENANCE_WITNESS_KEY", "a-different-key")
    problems = repo.verify_chain("enc")                      # cannot re-seal under a new key
    assert any("seal mismatch" in p for p in problems)


def test_legacy_store_may_adopt_the_witness_but_a_destroyed_one_may_not(tmp_path):
    """Post-fix review, DEPLOY BOUNDARY: a provenance store written before this control
    existed has unsealed rows and no journal. It must be able to adopt the witness (otherwise
    the first deploy holds every encounter forever), while a store whose witness was DESTROYED
    must still fail closed. The two are distinguished by whether anything was ever sealed."""
    import sqlite3
    import pytest
    from claude_coder.provenance import SqliteAuditRepository
    legacy = tmp_path / "legacy.db"
    repo = SqliteAuditRepository(legacy)
    repo.append("enc", "seed", {"x": 0})                  # create the schema
    repo._witness_path().unlink()                         # simulate a pre-witness store:
    conn = sqlite3.connect(str(legacy))                   # rows present, nothing sealed
    conn.execute("DROP TRIGGER audit_no_update")
    conn.execute("UPDATE audit_log SET witness_sha256=NULL")
    conn.commit()
    conn.close()
    SqliteAuditRepository(legacy).append("enc", "adopted", {"x": 1})   # adoption allowed
    problems = SqliteAuditRepository(legacy).verify_chain("enc")
    assert any("no witness seal" in p for p in problems)  # legacy row still reported honestly

    # ...but once anything IS sealed, a deleted journal is fatal
    SqliteAuditRepository(legacy)._witness_path().unlink()
    with pytest.raises(OSError):
        SqliteAuditRepository(legacy).append("enc", "c", {"x": 2})


def test_verification_never_mints_a_seal_key(tmp_path):
    """Post-fix review: `read_witness` must not CREATE a key. Minting one during a read-only
    integrity check would let a store whose key was deleted be silently re-sealed."""
    import pytest
    repo = _seeded(tmp_path)
    repo._witness_key_path().unlink()
    entries, problems = repo.read_witness()
    assert any("seal key is unavailable" in p or "cannot be verified" in p for p in problems)
    assert not repo._witness_key_path().exists()          # no key written by verification
    assert repo.verify_chain("enc")                       # and the chain is NOT declared intact
    with pytest.raises(OSError):                          # nor may a writer extend it
        _repo(tmp_path).append("enc", "c", {"x": 3})


def test_concurrent_first_writers_adopt_one_seal_key(tmp_path):
    """Post-fix review: the O_EXCL key creation must not fail (or, far worse, overwrite a key
    that already sealed entries) when writers race the very first append."""
    import threading
    from claude_coder.provenance import SqliteAuditRepository
    repo = SqliteAuditRepository(tmp_path / "race.db")
    barrier = threading.Barrier(4)
    keys: list[bytes] = []
    errors: list[Exception] = []

    def worker():
        try:
            barrier.wait()
            keys.append(repo._witness_key(create=True))
        except Exception as exc:                          # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(set(keys)) == 1                            # one key, adopted by all racers
    assert keys[0] == repo._witness_key_path().read_bytes().strip()


def test_concurrent_writers_stay_witnessed(tmp_path):
    """Post-fix review: the witness append sits inside the same BEGIN IMMEDIATE section as the
    row, so concurrent writers serialize on it — the journal must not fork or lose entries."""
    import threading
    from claude_coder.provenance import SqliteAuditRepository
    dbp = tmp_path / "concurrent.db"
    SqliteAuditRepository(dbp).append("enc", "seed", {"n": 0})   # schema + WAL established
    barrier = threading.Barrier(4)
    errors: list[Exception] = []

    def worker(i):
        repo = SqliteAuditRepository(dbp)
        try:
            barrier.wait()
            repo.append("enc", f"k{i}", {"n": i})
        except Exception as exc:                          # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    repo = SqliteAuditRepository(dbp)
    assert len(repo.records("enc")) == 5
    assert repo.verify_chain("enc") == []                 # every row sealed, journal intact


def test_in_doubt_write_is_sealed_into_the_journal(tmp_path):
    """Post-fix review: tolerating ONE in-doubt tail is required for crash recovery, so the
    anomaly must be permanently RECORDED, not merely inferable while the divergence lasts."""
    repo = _repo(tmp_path, "indoubt.db")
    repo.append("enc", "a", {"x": 1})
    lost = repo._append_head_witness("enc", "0" * 64)     # witness fsynced, commit lost
    repo.append("enc", "b", {"x": 2})                     # recovery extension
    entries, _ = repo.read_witness()
    assert entries[-1].get("in_doubt_over") == lost       # sealed into the entry
    problems = repo.verify_chain("enc")
    assert any("IN-DOUBT" in p for p in problems)
    # ...and the marker survives verification of the seal (it is part of the sealed payload)
    assert not any("seal mismatch" in p for p in problems)


def test_legacy_unwitnessed_rows_are_reported(tmp_path):
    """A row written before the witness existed is unwitnessed, and says so."""
    import sqlite3
    repo = _seeded(tmp_path)
    conn = sqlite3.connect(str(repo.db_path))
    conn.execute(
        "INSERT INTO audit_log(encounter_id, kind, recorded_at, control_mode,"
        " previous_record_sha256, record_json, record_sha256) VALUES (?,?,?,?,?,?,?)",
        ("enc2", "legacy", "2026-01-01T00:00:00+00:00", "ENFORCED_FAIL_CLOSED",
         "", "{}", "legacy-sha"))
    conn.commit()
    conn.close()
    assert any("no witness seal" in p for p in repo.verify_chain("enc2"))


# ---- Codex F6-R4-A round 4: the terminal head is anchored OUTSIDE the DB+journal pair ------
# The journal detects edits and ONE-SIDED truncation. Removing the terminal journal entry AND
# the terminal durable row TOGETHER leaves an internally consistent prefix that verifies clean
# and can be extended -- no key needed, and a better-held key does not help. Only an expected
# terminal POSITION held outside both mutable objects can catch it.
#
# NOTE ON WHAT THESE TESTS PROVE. They drive `LocalFileCheckpointAnchor`, the REFERENCE
# backend, which lives on the same filesystem as the store it anchors. They prove the
# detection logic, the fail-closed release path and the backend contract are correct --
# hermetically, with no network. They do NOT by themselves establish the production trust
# boundary: that property belongs to the STORE (an append-only bucket the writer cannot
# delete from), so it is proved against the real bucket from the deployed identity in
# `tests/test_checkpoint_s3.py`. `test_an_unanchored_store_cannot_detect_consistent_
# truncation_and_says_so` pins what remains true for any deployment that runs unanchored.
def _anchored(tmp_path, name="anchored.db"):
    """A fresh repository object bound to a persistent anchor directory. Building a new
    repository AND a new anchor object each time is what a process restart looks like."""
    from claude_coder.checkpoint import LocalFileCheckpointAnchor
    from claude_coder.provenance import SqliteAuditRepository
    return SqliteAuditRepository(
        tmp_path / name,
        checkpoint_anchor=LocalFileCheckpointAnchor(tmp_path / "anchor"))


def _truncate_both_tails(repo):
    """The reviewer's exact reproduction: drop the final witness line, then drop the SQLite
    delete trigger and the final audit row."""
    import sqlite3
    wp = repo._witness_path()
    lines = [ln for ln in wp.read_text().splitlines() if ln.strip()]
    wp.write_text("\n".join(lines[:-1]) + ("\n" if len(lines) > 1 else ""))
    conn = sqlite3.connect(str(repo.db_path))
    conn.execute("DROP TRIGGER audit_no_delete")
    conn.execute("DELETE FROM audit_log WHERE seq=(SELECT MAX(seq) FROM audit_log)")
    conn.commit()
    conn.close()


def test_consistent_db_and_journal_tail_truncation_is_detected(tmp_path):
    """THE reported reproduction. Both tails removed together; the remaining prefix is
    internally consistent, so only the external anchor can contradict it."""
    repo = _anchored(tmp_path)
    repo.append("enc", "a", {"x": 1})
    repo.append("enc", "b", {"x": 2})                     # terminal release record
    assert repo.verify_chain("enc") == []
    _truncate_both_tails(repo)
    problems = _anchored(tmp_path).verify_chain("enc")
    assert any("SHORTER than its anchored checkpoint" in p for p in problems), problems


def test_a_subsequent_append_after_consistent_truncation_is_refused(tmp_path):
    """...and the system must not carry on issuing releases on the rewritten history. This is
    the RELEASE PATH: `append` raises, which `code_encounter` turns into SYSTEM_HOLD."""
    import pytest
    repo = _anchored(tmp_path)
    repo.append("enc", "a", {"x": 1})
    repo.append("enc", "b", {"x": 2})
    _truncate_both_tails(repo)
    with pytest.raises(OSError) as err:
        _anchored(tmp_path).append("enc", "c", {"x": 3})
    assert "truncated together" in str(err.value)


def test_an_unanchored_store_cannot_detect_consistent_truncation_and_says_so(tmp_path):
    """The honest residual limit, pinned. With NO anchor configured the same attack is
    invisible -- so the status must declare that, rather than reporting a clean chain as
    though the guarantee held. Delete this test the day a real external backend is required."""
    repo = _repo(tmp_path, "unanchored.db")               # DisabledCheckpointAnchor
    repo.append("enc", "a", {"x": 1})
    repo.append("enc", "b", {"x": 2})
    _truncate_both_tails(repo)
    assert _repo(tmp_path, "unanchored.db").verify_chain("enc") == []     # undetectable
    status = _repo(tmp_path, "unanchored.db").witness_status()["checkpoint"]
    assert status["configured"] is False
    assert status["external_trust_boundary"] is False
    assert "NOT detectable" in status["limitation"]


def test_the_anchor_survives_restart_and_tracks_the_journal(tmp_path):
    """RESTART: a new process (new repository object, new anchor object) reads the same
    anchored state, extends normally, and stays aligned."""
    _anchored(tmp_path).append("enc", "a", {"x": 1})
    _anchored(tmp_path).append("enc", "b", {"x": 2})
    _anchored(tmp_path).append("enc2", "c", {"x": 3})      # anchor is per STORE, not encounter
    repo = _anchored(tmp_path)
    assert repo.verify_chain() == []
    status = repo.witness_status()["checkpoint"]
    assert status["journal_seq"] == 3 and status["anchored_seq"] == 3
    assert status["configured"] is True and status["problems"] == []


def test_an_unavailable_checkpoint_anchor_holds_the_release(tmp_path):
    """UNVERIFIABLE must fail closed. An anchor that cannot be READ stops the append (and is
    reported by verification); it can never be treated as an implicit pass."""
    import pytest
    from claude_coder.checkpoint import AnchorUnavailable, CheckpointAnchor
    from claude_coder.provenance import SqliteAuditRepository

    class _Unreachable(CheckpointAnchor):
        backend_id, configured, external = "stub-unreachable", True, True

        def read(self, store_id):
            raise AnchorUnavailable("checkpoint store unreachable")

        def write(self, record):
            raise AnchorUnavailable("checkpoint store unreachable")

    repo = SqliteAuditRepository(tmp_path / "unreachable.db",
                                 checkpoint_anchor=_Unreachable())
    with pytest.raises(OSError) as err:
        repo.append("enc", "a", {"x": 1})
    assert "unverifiable" in str(err.value)
    assert repo.records("enc") == []                       # nothing committed
    assert any("unverifiable" in p for p in repo.verify_chain("enc"))


def test_an_unwritable_checkpoint_anchor_leaves_no_committed_row(tmp_path):
    """The other side of the ordering contract: the checkpoint is advanced BETWEEN the journal
    fsync and the commit, so a write failure aborts the append rather than committing a row
    the anchor never covered."""
    import pytest
    from claude_coder.checkpoint import AnchorUnavailable, LocalFileCheckpointAnchor
    from claude_coder.provenance import SqliteAuditRepository

    class _ReadOnly(LocalFileCheckpointAnchor):
        def write(self, record):
            raise AnchorUnavailable("checkpoint store is read-only")

    repo = SqliteAuditRepository(tmp_path / "readonly.db",
                                 checkpoint_anchor=_ReadOnly(tmp_path / "anchor"))
    with pytest.raises(OSError) as err:
        repo.append("enc", "a", {"x": 1})
    assert "could not be anchored" in str(err.value)
    assert repo.records("enc") == []


def test_a_rolled_back_anchor_is_detected(tmp_path):
    """ROLLBACK: restoring an OLDER checkpoint leaves the journal further ahead than any crash
    can explain (at most one entry is recoverable), so it fails closed."""
    import json
    import pytest
    repo = _anchored(tmp_path)
    repo.append("enc", "a", {"x": 1})
    anchor = repo._anchor()
    stale = json.loads(anchor._path(repo.store_id()).read_text())      # checkpoint at seq 1
    _anchored(tmp_path).append("enc", "b", {"x": 2})
    _anchored(tmp_path).append("enc", "c", {"x": 3})
    anchor._path(repo.store_id()).write_text(json.dumps(stale, sort_keys=True))
    with pytest.raises(OSError) as err:
        _anchored(tmp_path).append("enc", "d", {"x": 4})
    assert "beyond its anchored checkpoint" in str(err.value)


def test_a_replayed_journal_at_the_same_sequence_is_detected(tmp_path):
    """REPLAY: a journal restored from an older backup can reach the anchored sequence again
    only with different seals. Same position, different content, is a rewrite."""
    import json
    import pytest
    from dataclasses import replace
    from claude_coder.checkpoint import Checkpoint
    repo = _anchored(tmp_path)
    repo.append("enc", "a", {"x": 1})
    anchor = repo._anchor()
    current = Checkpoint.from_record(anchor.read(repo.store_id()))
    other = replace(current, seal="f" * 64)
    other = replace(other, signature=repo._seal(other.payload(), create_key=True))
    # written straight to the backing store: the backend's own fork guard would refuse it,
    # which is itself the point -- reaching this state requires overwriting the anchor.
    anchor._path(repo.store_id()).write_text(json.dumps(other.as_record(), sort_keys=True))
    with pytest.raises(OSError) as err:
        _anchored(tmp_path).append("enc", "b", {"x": 2})
    assert "rewritten, rolled back or replayed" in str(err.value)


def test_the_reference_backend_refuses_to_move_backwards_or_fork(tmp_path):
    """The backend contract a real (WORM/object-locked) store would enforce structurally."""
    import pytest
    from claude_coder.checkpoint import AnchorRollback, Checkpoint, LocalFileCheckpointAnchor
    anchor = LocalFileCheckpointAnchor(tmp_path / "anchor")
    base = Checkpoint(store_id="s", seq=2, seal="a" * 64, record_sha256="b" * 64,
                      witness_version="v", written_at="2026-01-01T00:00:00+00:00",
                      signature="sig")
    anchor.write(base.as_record())
    from dataclasses import replace as _replace
    with pytest.raises(AnchorRollback):
        anchor.write(_replace(base, seq=1).as_record())            # backwards
    with pytest.raises(AnchorRollback):
        anchor.write(_replace(base, seal="c" * 64).as_record())    # fork at the same seq
    anchor.write(_replace(base, seq=3).as_record())                # forward is fine
    assert Checkpoint.from_record(anchor.read("s")).seq == 3


def test_a_forged_or_malformed_checkpoint_is_not_accepted(tmp_path):
    """The anchor store is not blindly trusted either: the checkpoint is signed with the same
    custody as the journal seal, so a third party who can write the anchor cannot fabricate a
    terminal head, and a malformed record is an error rather than 'no checkpoint'."""
    import json
    import pytest
    repo = _anchored(tmp_path)
    repo.append("enc", "a", {"x": 1})
    path = repo._anchor()._path(repo.store_id())
    forged = json.loads(path.read_text())
    forged["signature"] = "0" * 64
    path.write_text(json.dumps(forged, sort_keys=True))
    with pytest.raises(OSError) as err:
        _anchored(tmp_path).append("enc", "b", {"x": 2})
    assert "fails its signature" in str(err.value)
    path.write_text(json.dumps({"version": "terminal-head-checkpoint-v1"}))
    with pytest.raises(OSError) as err:
        _anchored(tmp_path).append("enc", "b", {"x": 2})
    assert "malformed" in str(err.value)


# ---- the control has to be ON in the DEPLOYED configuration, not merely implementable ----
# Codex F6-R4-A finding A was not that the anchor did not work; it was that a repository
# search found neither environment variable in any deployment path, so the shipped default
# was `{'backend': 'disabled', 'configured': False, 'external': False, 'required': False}`.
# These assert the wiring itself, from the checked-in source, so deleting it is a failing
# test rather than a silently disabled integrity control.
def _repo_root():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent


def test_the_deployed_runtime_requires_the_anchor_from_checked_in_source():
    """`PROVENANCE_CHECKPOINT_REQUIRED=1` is pinned in docker-compose.yml, NOT only in .env.

    .env is materialised from Secrets Manager at first boot and refreshed out of band, so it
    can be stale; docker-compose.yml is the deployed source. Keeping the requirement here
    means configuration drift can only produce "required but unanchored" (which holds the
    release) and never "silently unanchored" -- which is the exact state the reviewer found.
    """
    compose = (_repo_root() / "docker-compose.yml").read_text()
    assert "PROVENANCE_CHECKPOINT_REQUIRED=1" in compose, (
        "the deployed compose file no longer requires the terminal-head checkpoint anchor; "
        "an unanchored run would silently stop detecting consistent truncation")


def test_the_deployed_anchor_uri_is_derived_from_the_bucket_terraform_creates():
    """...and the anchor URI comes from the bucket RESOURCE, never a literal.

    The bucket name carries a random suffix, so a hand-copied URI survives exactly until the
    next clean `terraform apply` and then points at a bucket that does not exist -- which,
    because an unreachable anchor correctly fails closed, presents as a total release outage
    far from its cause. Deriving it means the runtime is always pointed at the bucket the
    same config created.
    """
    secrets = (_repo_root() / "terraform" / "secrets.tf").read_text()
    assert "PROVENANCE_CHECKPOINT_ANCHOR" in secrets, (
        "terraform no longer delivers the checkpoint anchor URI into the runtime .env")
    line = next(ln for ln in secrets.splitlines() if "PROVENANCE_CHECKPOINT_ANCHOR" in ln
                and "=" in ln.split("#")[0])
    assert "aws_s3_bucket.provenance_checkpoint.bucket" in line, line
    assert "${var.aws_region}" in line, line
    # ...and it must stay inside the prefix the app role is actually granted.
    assert "/checkpoints?" in line, line


def test_the_documented_example_env_matches_the_deployed_switches():
    """.env.example is the operator-facing description of the runtime contract. If it stops
    naming these, the next person to hand-build an .env produces a silently unanchored
    deployment -- which is how this finding happened in the first place."""
    example = (_repo_root() / ".env.example").read_text()
    for key in ("PROVENANCE_CHECKPOINT_ANCHOR", "PROVENANCE_CHECKPOINT_REQUIRED",
                "PROVENANCE_CHECKPOINT_ADOPT"):
        assert key in example, f"{key} is undocumented in .env.example"


def test_a_deleted_checkpoint_fails_closed_and_adoption_cannot_wave_it_through(
        tmp_path, monkeypatch):
    """Deleting the anchored checkpoint is the cheapest attack on the anchor, and it looks
    exactly like enabling the anchor on an existing store. Both fail closed by default; and
    the operator switch that exists for the SECOND case must not launder the FIRST.

    (Codex F6-R4-A finding A: `PROVENANCE_CHECKPOINT_ADOPT` used to return unconditionally
    whenever the anchor held nothing, so setting it turned a destroyed anchor into a clean
    adoption -- one environment variable away from disabling the whole control on a store
    that had been anchored for its entire life.)"""
    import pytest
    repo = _anchored(tmp_path)
    repo.append("enc", "a", {"x": 1})
    repo._anchor()._path(repo.store_id()).unlink()
    with pytest.raises(OSError) as err:
        _anchored(tmp_path).append("enc", "b", {"x": 2})
    assert "holds no checkpoint" in str(err.value)
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_ADOPT", "1")
    with pytest.raises(OSError) as err:
        _anchored(tmp_path).append("enc", "b", {"x": 2})
    assert "refusing to adopt" in str(err.value)
    assert "was written under a checkpoint anchor" in str(err.value)
    assert len(_anchored(tmp_path).records("enc")) == 1               # nothing committed


def test_a_tampered_checkpoint_is_not_adoptable_either(tmp_path, monkeypatch):
    """The adjacent case: rather than deleting the checkpoint, rewrite it. A record that
    exists but fails its signature never reaches the adoption branch at all -- adoption is
    only ever consulted for an EMPTY anchor -- so the switch cannot help there either."""
    import json
    import pytest
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_ADOPT", "1")
    repo = _anchored(tmp_path)
    repo.append("enc", "a", {"x": 1})
    path = repo._anchor()._path(repo.store_id())
    record = json.loads(path.read_text())
    path.write_text(json.dumps(dict(record, signature="0" * 64), sort_keys=True))
    with pytest.raises(OSError) as err:
        _anchored(tmp_path).append("enc", "b", {"x": 2})
    assert "fails its signature" in str(err.value)
    assert len(_anchored(tmp_path).records("enc")) == 1


def _legacy_unanchored_store(tmp_path, name="legacy.db"):
    """A store shaped like the one already in production BEFORE this control existed.

    Reproduced rather than described, because the migration has to work on the real thing:
    an oldest prefix of rows with no witness seal at all (written before the journal
    existed), then rows that ARE sealed into a journal (written after the journal shipped
    but before any anchor), and no checkpoint anywhere. Returns the db path.
    """
    import sqlite3
    from claude_coder.provenance import SqliteAuditRepository
    db = tmp_path / name
    repo = SqliteAuditRepository(db)
    for i in range(3):                                   # will be de-witnessed below
        repo.append("legacy-enc", f"pre-witness-{i}", {"i": i})
    repo._witness_path().unlink()
    conn = sqlite3.connect(str(db))
    conn.execute("DROP TRIGGER audit_no_update")
    conn.execute("UPDATE audit_log SET witness_sha256=NULL")
    conn.commit()
    conn.close()
    for i in range(3):                                   # witnessed, still unanchored
        SqliteAuditRepository(db).append("legacy-enc", f"witnessed-{i}", {"i": i})
    return db


def test_a_preexisting_unanchored_store_adopts_the_anchor_exactly_once(tmp_path, monkeypatch):
    """THE MIGRATION (Codex F6-R4-A finding A). Turning the anchor on in a deployment that
    already has a provenance store must be a supported, tested operation -- otherwise
    switching the control on holds every encounter forever and the deployment gets reverted.

    The reviewed sequence, end to end:
      1. the pre-existing store, unanchored, with legacy unwitnessed rows in its history;
      2. anchor + REQUIRED on, ADOPT off  -> refuses, and names the switch;
      3. one run with ADOPT on            -> adopts, commits, anchors from here;
      4. ADOPT off again                  -> keeps working, now genuinely anchored;
      5. the anchor advances with the journal from that point on.
    """
    import pytest
    from claude_coder.checkpoint import LocalFileCheckpointAnchor
    from claude_coder.provenance import SqliteAuditRepository
    db = _legacy_unanchored_store(tmp_path)
    anchor_root = tmp_path / "anchor"

    def repo():                                          # a fresh process each time
        return SqliteAuditRepository(
            db, checkpoint_anchor=LocalFileCheckpointAnchor(anchor_root))

    monkeypatch.setenv("PROVENANCE_CHECKPOINT_REQUIRED", "1")
    before = len(repo().records("legacy-enc"))
    assert repo().checkpoint_status()["anchored_seq"] is None

    with pytest.raises(OSError) as err:                                       # step 2
        repo().append("legacy-enc", "held", {"x": 1})
    assert "holds no checkpoint" in str(err.value)
    assert "PROVENANCE_CHECKPOINT_ADOPT" in str(err.value)
    assert len(repo().records("legacy-enc")) == before                        # nothing lost

    monkeypatch.setenv("PROVENANCE_CHECKPOINT_ADOPT", "1")                    # step 3
    repo().append("legacy-enc", "adopted", {"x": 1})
    assert len(repo().records("legacy-enc")) == before + 1
    status = repo().checkpoint_status()
    assert status["adoption_allowed"] is True            # the weakened posture is RECORDED
    assert status["anchored_seq"] == status["journal_seq"]

    monkeypatch.delenv("PROVENANCE_CHECKPOINT_ADOPT")                         # step 4
    repo().append("legacy-enc", "after", {"x": 2})
    after = repo().checkpoint_status()                                        # step 5
    assert after["adoption_allowed"] is False
    assert after["anchored_seq"] == after["journal_seq"]
    assert after["problems"] == []
    # The legacy prefix stays reported honestly rather than papered over by the migration:
    # the only remaining problems are the pre-witness rows, and NOTHING about the anchor.
    problems = repo().verify_chain("legacy-enc")
    assert problems and all("no witness seal" in p for p in problems), problems


def test_adoption_is_refused_once_the_store_has_been_anchored_even_once(tmp_path, monkeypatch):
    """The other half, and the reason adoption is safe: the SAME store, after its one
    adoption run, can never be re-adopted. Destroying the anchor now fails closed with
    ADOPT still set -- the sealed `anchored` mark in the journal is the evidence, and it
    lives inside the very object an attacker would have to forge the seal to edit."""
    import shutil
    import pytest
    from claude_coder.checkpoint import LocalFileCheckpointAnchor
    from claude_coder.provenance import SqliteAuditRepository
    db = _legacy_unanchored_store(tmp_path, name="adopted.db")
    anchor_root = tmp_path / "anchor2"

    def repo():
        return SqliteAuditRepository(
            db, checkpoint_anchor=LocalFileCheckpointAnchor(anchor_root))

    monkeypatch.setenv("PROVENANCE_CHECKPOINT_REQUIRED", "1")
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_ADOPT", "1")
    repo().append("legacy-enc", "adopted", {"x": 1})
    repo().append("legacy-enc", "and-another", {"x": 2})
    committed = len(repo().records("legacy-enc"))

    # Destroy the whole anchor store -- the strongest form of the attack, and exactly what
    # "enable the anchor on an existing store" looks like from inside the process.
    shutil.rmtree(anchor_root)
    with pytest.raises(OSError) as err:
        repo().append("legacy-enc", "should-not-commit", {"x": 3})
    assert "refusing to adopt" in str(err.value)
    assert len(repo().records("legacy-enc")) == committed
    assert any("refusing to adopt" in p for p in repo().verify_chain("legacy-enc"))


def test_an_uncommitted_anchored_tail_is_still_adoptable(tmp_path, monkeypatch):
    """The boundary between the two rules above, and a real availability trap if it is got
    wrong. The `anchored` mark is sealed into the journal entry BEFORE the checkpoint is
    written, so an anchor write that fails leaves a marked entry with no checkpoint and no
    committed row. If that state were read as "this store was anchored", an S3 hiccup during
    the very migration run would brick the store permanently -- the adoption it needs would
    be refused on the strength of the failed attempt itself.

    It is distinguishable and is distinguished: the append never committed, so no durable
    row carries that entry's seal. Exactly one tolerance, for the terminal entry only.
    """
    import pytest
    from claude_coder.checkpoint import AnchorUnavailable, LocalFileCheckpointAnchor
    from claude_coder.provenance import SqliteAuditRepository

    class _WriteFails(LocalFileCheckpointAnchor):
        def write(self, record):
            raise AnchorUnavailable("simulated anchor-store outage")

    db = _legacy_unanchored_store(tmp_path, name="hiccup.db")
    root = tmp_path / "anchor3"
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_REQUIRED", "1")
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_ADOPT", "1")
    committed = len(SqliteAuditRepository(db).records("legacy-enc"))

    with pytest.raises(OSError) as err:                  # the migration run, mid-outage
        SqliteAuditRepository(db, checkpoint_anchor=_WriteFails(root)).append(
            "legacy-enc", "a", {"x": 1})
    assert "could not be anchored" in str(err.value)
    healthy = SqliteAuditRepository(db, checkpoint_anchor=LocalFileCheckpointAnchor(root))
    assert len(healthy.records("legacy-enc")) == committed        # nothing committed...
    assert healthy.checkpoint_status()["journal_seq"] == 4        # ...but the journal moved

    healthy.append("legacy-enc", "a", {"x": 1})                   # retry once S3 is back
    recovered = SqliteAuditRepository(db, checkpoint_anchor=LocalFileCheckpointAnchor(root))
    assert len(recovered.records("legacy-enc")) == committed + 1
    assert recovered.checkpoint_status()["problems"] == []
    # ...and the store is genuinely anchored now, so it is no longer re-adoptable.
    import shutil
    shutil.rmtree(root)
    with pytest.raises(OSError) as err:
        SqliteAuditRepository(db, checkpoint_anchor=LocalFileCheckpointAnchor(root)).append(
            "legacy-enc", "b", {"x": 2})
    assert "refusing to adopt" in str(err.value)


def test_adoption_refuses_a_journal_it_cannot_read_cleanly(tmp_path, monkeypatch):
    """Fail-closed on the fix's own failure path. `_prior_anchoring_evidence` answers "was
    this store ever anchored?" by reading the journal; if the journal does not verify, the
    honest answer is 'unknown', and unknown must refuse rather than fall through to "no
    evidence found" -- which would be the silent-empty-success bug in a new place."""
    import json
    import pytest
    from claude_coder.checkpoint import LocalFileCheckpointAnchor
    from claude_coder.provenance import SqliteAuditRepository
    db = _legacy_unanchored_store(tmp_path, name="corrupt.db")
    root = tmp_path / "anchor4"
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_REQUIRED", "1")
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_ADOPT", "1")
    # Edit an INTERIOR entry: the tail still seals, so `_witness_tail` is happy and only the
    # whole-journal read notices. That is the case a tail-only check would have missed.
    wp = SqliteAuditRepository(db)._witness_path()
    lines = [ln for ln in wp.read_text().splitlines() if ln.strip()]
    first = json.loads(lines[0])
    first["encounter_id"] = "rewritten"
    lines[0] = json.dumps(first, sort_keys=True)
    wp.write_text("\n".join(lines) + "\n")
    with pytest.raises(OSError) as err:
        SqliteAuditRepository(db, checkpoint_anchor=LocalFileCheckpointAnchor(root)).append(
            "legacy-enc", "x", {"x": 1})
    assert "does not read cleanly" in str(err.value)


def test_required_mode_refuses_to_run_without_an_anchor(tmp_path, monkeypatch):
    """The switch that turns this control on once a real backend exists -- no code change."""
    import pytest
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_REQUIRED", "1")
    with pytest.raises(OSError) as err:
        _repo(tmp_path, "required.db").append("enc", "a", {"x": 1})
    assert "no terminal-head checkpoint anchor is configured" in str(err.value)


def test_an_unimplemented_anchor_backend_fails_closed(tmp_path, monkeypatch):
    """A backend spec this build cannot honour must STOP the writer. Reading it as 'no anchor
    configured' would let a typo (or an aspirational deployment setting) silently disable the
    control -- the exact class of silent-weakening this guard exists to prevent."""
    import pytest
    from claude_coder.checkpoint import CheckpointConfigError, resolve_checkpoint_anchor
    with pytest.raises(CheckpointConfigError):
        resolve_checkpoint_anchor("gs://some-bucket/checkpoints")   # unimplemented scheme
    with pytest.raises(CheckpointConfigError):
        resolve_checkpoint_anchor("file:")
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_ANCHOR", "gs://some-bucket/checkpoints")
    with pytest.raises(Exception):
        _repo(tmp_path, "badspec.db").append("enc", "a", {"x": 1})
    assert any("unusable" in p for p in
               _repo(tmp_path, "badspec.db").verify_chain("enc"))


def test_the_s3_anchor_spec_is_parsed_and_validated_without_touching_the_network(tmp_path):
    """The S3 backend is selected by CONFIGURATION, and every way of misconfiguring it is
    refused where the operator can see it -- at resolution, not on the release path hours
    later. Hermetic: constructing the backend resolves no credentials and makes no call."""
    import pytest
    from claude_coder.checkpoint import (CheckpointConfigError, S3CheckpointAnchor,
                                         resolve_checkpoint_anchor)
    anchor = resolve_checkpoint_anchor("s3://a-bucket/checkpoints?region=us-east-1")
    assert isinstance(anchor, S3CheckpointAnchor)
    assert (anchor.bucket, anchor.prefix, anchor.region) == \
        ("a-bucket", "checkpoints/", "us-east-1")
    described = anchor.describe()
    # The property F6-R4-A actually asked for: this backend is a real privilege boundary
    # and says so, where the reference backend must not.
    assert described["external_trust_boundary"] is True
    assert described["configured"] is True
    assert described["location"] == "s3://a-bucket/checkpoints/"
    assert "cannot erase it" in described["guarantee"]
    assert "limitation" not in described
    # A trailing-slash-less prefix is the same anchor, not a second namespace.
    assert resolve_checkpoint_anchor("s3://a-bucket/checkpoints/").prefix == "checkpoints/"
    with pytest.raises(CheckpointConfigError):
        resolve_checkpoint_anchor("s3://a-bucket")                  # no prefix: grant is
    with pytest.raises(CheckpointConfigError):
        resolve_checkpoint_anchor("s3://a-bucket/")                 # prefix-scoped
    with pytest.raises(CheckpointConfigError):
        resolve_checkpoint_anchor("s3:///checkpoints")              # no bucket
    with pytest.raises(CheckpointConfigError):
        resolve_checkpoint_anchor("s3://a-bucket/checkpoints?regoin=us-east-1")  # typo
    with pytest.raises(CheckpointConfigError):
        resolve_checkpoint_anchor("s3://a-bucket/checkpoints#frag")


def test_the_s3_anchor_uses_one_write_once_object_per_sequence(tmp_path):
    """Key layout is part of the guarantee, so it is asserted, not assumed: one immutable
    object per sequence (what a truncation cannot remove) plus an explicitly untrusted
    head pointer. Hermetic -- key derivation only."""
    from claude_coder.checkpoint import S3CheckpointAnchor
    anchor = S3CheckpointAnchor("b", "checkpoints/", region="us-east-1")
    assert anchor._head_key("prov.db") == "checkpoints/prov.db/head.json"
    assert anchor._seq_key("prov.db", 7) == "checkpoints/prov.db/seq/000000000007.json"
    assert anchor._seq_key("prov.db", 7) < anchor._seq_key("prov.db", 12)   # sorts by seq
    # A store id can never escape the granted prefix, whatever it contains.
    assert "/../" not in anchor._seq_key("../../etc/passwd", 1)
    assert anchor._seq_key("../../etc/passwd", 1).startswith("checkpoints/")


def test_the_s3_anchor_shares_one_client_per_process_and_region(monkeypatch):
    """Post-fix review, COST/RATE-LIMIT BOUNDARY. Production resolves a FRESH anchor object
    for every anchor operation, so a per-object client rebuilt a botocore session, reloaded
    the S3 model and re-resolved instance-role credentials several times per audit record.
    Instance metadata is rate-limited, so that is not just slow -- under load it returns
    throttles, and a throttle here is an UNVERIFIABLE anchor that holds a release for no
    reason. The client is shared per (pid, region): per REGION so it is reused, and per PID
    so a client's open sockets are never inherited across a fork into a worker."""
    import os
    from claude_coder import checkpoint as ck
    monkeypatch.setattr(ck, "_S3_CLIENTS", {}, raising=True)
    first = ck.resolve_checkpoint_anchor("s3://a-bucket/checkpoints?region=us-east-1")._s3()
    second = ck.resolve_checkpoint_anchor("s3://b-bucket/other/?region=us-east-1")._s3()
    assert first is second                                   # reused across anchor objects
    other_region = ck.resolve_checkpoint_anchor(
        "s3://a-bucket/checkpoints?region=us-west-2")._s3()
    assert other_region is not first                         # never across regions
    assert set(ck._S3_CLIENTS) == {(os.getpid(), "us-east-1"), (os.getpid(), "us-west-2")}
    # An explicitly injected client is always honoured over the shared one.
    sentinel = object()
    assert ck.S3CheckpointAnchor("a-bucket", "checkpoints/", region="us-east-1",
                                 client=sentinel)._s3() is sentinel


def test_the_journal_sequence_is_inside_the_seal(tmp_path):
    """The monotonic sequence is only worth anything if it cannot be renumbered to match a
    truncated journal, so it is part of the sealed payload."""
    import json
    repo = _anchored(tmp_path)
    repo.append("enc", "a", {"x": 1})
    repo.append("enc", "b", {"x": 2})
    wp = repo._witness_path()
    lines = wp.read_text().splitlines()
    assert [json.loads(ln)["seq"] for ln in lines] == [1, 2]
    edited = json.loads(lines[-1])
    edited["seq"] = 99
    wp.write_text("\n".join(lines[:-1] + [json.dumps(edited, sort_keys=True)]) + "\n")
    assert any("seal mismatch" in p for p in _anchored(tmp_path).verify_chain("enc"))


def test_an_emptied_store_with_a_live_checkpoint_is_detected(tmp_path):
    """Deleting the whole store (rather than its tail) must not read as 'nothing to verify'."""
    import sqlite3
    repo = _anchored(tmp_path)
    repo.append("enc", "a", {"x": 1})
    repo._witness_path().unlink()
    conn = sqlite3.connect(str(repo.db_path))
    conn.execute("DROP TRIGGER audit_no_delete")
    conn.execute("DELETE FROM audit_log")
    conn.commit()
    conn.close()
    problems = _anchored(tmp_path).verify_chain("enc")
    assert any("the journal is empty" in p for p in problems), problems


def test_concurrent_writers_share_one_monotonic_anchor(tmp_path):
    """Post-fix review, PROCESS BOUNDARY: the checkpoint advances inside the same
    BEGIN IMMEDIATE section as the journal entry and the row, so racing writers must produce
    one dense sequence and one anchor that ends exactly at it -- never a fork, a gap, or a
    lost update that would later read as a truncation."""
    import json
    import threading
    from claude_coder.checkpoint import Checkpoint, LocalFileCheckpointAnchor
    from claude_coder.provenance import SqliteAuditRepository
    dbp = tmp_path / "anchored-concurrent.db"

    def _repo_for():
        return SqliteAuditRepository(
            dbp, checkpoint_anchor=LocalFileCheckpointAnchor(tmp_path / "anchor"))

    _repo_for().append("enc", "seed", {"n": 0})           # schema + WAL established
    barrier = threading.Barrier(4)
    errors: list[Exception] = []

    def worker(i):
        repo = _repo_for()
        try:
            barrier.wait()
            repo.append("enc", f"k{i}", {"n": i})
        except Exception as exc:                          # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    repo = _repo_for()
    assert len(repo.records("enc")) == 5
    assert repo.verify_chain("enc") == []
    seqs = [json.loads(ln)["seq"]
            for ln in repo._witness_path().read_text().splitlines() if ln.strip()]
    assert seqs == [1, 2, 3, 4, 5]                        # dense, no fork, no gap
    anchored = Checkpoint.from_record(repo._anchor().read(repo.store_id()))
    assert anchored.seq == 5


def test_the_anchor_is_wired_from_configuration_not_only_injection(tmp_path, monkeypatch):
    """Post-fix review: every other case here INJECTS the backend, which would leave the way
    production actually turns the control on -- one environment variable -- untested. This
    drives the real resolution path end to end, including the reported reproduction."""
    from claude_coder.provenance import SqliteAuditRepository
    monkeypatch.setenv("PROVENANCE_CHECKPOINT_ANCHOR", f"file:{tmp_path / 'anchor'}")
    dbp = tmp_path / "configured.db"
    SqliteAuditRepository(dbp).append("enc", "a", {"x": 1})
    SqliteAuditRepository(dbp).append("enc", "b", {"x": 2})
    repo = SqliteAuditRepository(dbp)
    assert repo.verify_chain("enc") == []
    assert repo.witness_status()["checkpoint"]["backend"] == "local-file"
    _truncate_both_tails(repo)
    assert any("SHORTER than its anchored checkpoint" in p
               for p in SqliteAuditRepository(dbp).verify_chain("enc"))
