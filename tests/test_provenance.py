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
