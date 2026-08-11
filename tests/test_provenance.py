"""Phase-0 provenance kernel: evidence anchoring + relation identity/merge.

Failure-path first: the point of anchoring is to REJECT a plausible-but-non-verbatim
quotation, and the point of the relation kernel is that a re-asserted edge accumulates
support (never duplicates) while any state disagreement collapses to UNCERTAIN.
Agnostic — synthetic text and synthetic event ids, no medical code."""
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
def _rel(subj, pred, obj, state=RelationState.ASSERTED, ev=None, conf=0.5):
    return RelationAssertion(subject_event_id=subj, predicate=pred, object_event_id=obj,
                             state=state, evidence_span_ids=list(ev or []), confidence=conf)


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
