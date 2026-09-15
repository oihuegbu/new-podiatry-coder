"""Candidate/evidence stability contract regressions (synthetic vocabulary only)."""
from unittest.mock import patch

from app.contracts.source_evidence import (ReconciliationStatus, SourceReconciliation,
                                            SpanReconciliation)
from claude_coder import pipeline, resolution, verify
from claude_coder.data_access import MockSource
from claude_coder.models import (CandidateCode, ClinicalFact, Disposition, EvidenceSpan,
                                 FactKind, ResolutionMethod, ResolvedLine)


def _fact(fact_id: str, description: str, text: str, span_id: str,
          *, disposition=Disposition.PERFORMED) -> ClinicalFact:
    return ClinicalFact(
        kind=FactKind.PROCEDURE, description=description, fact_id=fact_id,
        disposition=disposition, confidence=0.99,
        evidence=[EvidenceSpan(text=text, span_id=span_id, anchored=True)],
    )


def _reconciliation(*span_ids: str) -> SourceReconciliation:
    return SourceReconciliation(spans=tuple(
        SpanReconciliation(span_id=sid, status=ReconciliationStatus.AGREED)
        for sid in span_ids))


def test_canonical_service_packet_is_order_independent_and_excludes_nonclaim_history():
    target = _fact("F1", "assembly repair", "assembly repair performed", "s1")
    component = _fact("F2", "assembly preparation", "assembly surface prepared", "s2")
    history = _fact("F3", "prior assembly", "prior assembly discussed", "s3",
                    disposition=Disposition.HISTORICAL)

    first = verify.build_service_evidence_packet(
        target, (history, component, target), "service-A")
    second = verify.build_service_evidence_packet(
        target, (target, component, history), "service-A")

    assert first.packet_sha256 == second.packet_sha256
    assert first.context_fact_ids == ("F1", "F2")
    assert [s.span_id for s in first.spans] == ["s1", "s2"]
    assert "F3" not in first.context_fact_ids


def test_context_citation_must_connect_target_and_candidate_not_only_a_sibling():
    target = _fact("F1", "assembly repair", "assembly repair performed", "s1")
    related = _fact("F2", "assembly preparation", "assembly preparation completed", "s2")
    unrelated = _fact("F3", "surface polishing", "surface polishing completed", "s3")
    packet = verify.build_service_evidence_packet(
        target, (target, related, unrelated), "service-A")
    candidate = CandidateCode(
        code="CODE_A", system="cpt", descriptor="secondary assembly repair",
        source="retrieval")
    reconciliation = _reconciliation("s1", "s2", "s3")

    assert verify._agreed_citable_spans(
        ("s2",), target, candidate, reconciliation, "entailed", packet) == ("s2",)
    assert verify._agreed_citable_spans(
        ("s3",), target, candidate, reconciliation, "entailed", packet) == ()


def test_singleton_verifier_uses_the_same_canonical_packet_as_audit():
    target = _fact("F1", "assembly repair", "assembly repair performed", "s1")
    component = _fact("F2", "assembly preparation", "surface prepared", "s2")
    packet = verify.build_service_evidence_packet(
        target, (target, component), "service-A")
    candidate = CandidateCode(
        code="CODE_A", system="cpt", descriptor="assembly repair",
        source="retrieval")

    prompt, citation_map = verify._shortlist_prompt(
        target, [candidate], MockSource(), evidence_packet=packet)

    assert packet.packet_sha256 in prompt
    assert "surface prepared" in prompt
    assert set(citation_map.values()) == {"s1", "s2"}


def test_packet_does_not_promote_unreconciled_raw_context_attributes():
    target = _fact("F1", "assembly repair", "assembly repair performed", "s1")
    component = _fact("F2", "assembly preparation", "surface prepared", "s2")
    component.attributes["technique"] = "observed-only"

    packet = verify.build_service_evidence_packet(
        target, (target, component), "service-A", reconciliation=_reconciliation("s1", "s2"))

    component_record = next(r for r in packet.fact_records if r["fact_id"] == "F2")
    assert component_record["attributes"] == {}


def test_resolver_candidate_universe_never_calls_model_code_proposal():
    fact = _fact("F1", "assembly repair", "assembly repair performed", "s1")
    candidate = CandidateCode(
        code="CODE_A", system="cpt", descriptor="assembly repair",
        score=0.9, source="retrieval")
    source = MockSource(
        records={("CODE_A", "cpt"): {"description": "assembly repair"}},
        semantic_class={"CODE_A": "surgical_procedure"})
    expected = ResolvedLine(
        fact=fact, chosen=candidate, method=ResolutionMethod.VERIFIED)

    with patch.object(verify, "propose_codes", side_effect=AssertionError(
            "model-authored codes must not enter the decisive universe")), patch.object(
                resolution, "_propose_then_verify_core", return_value=expected) as core:
        line = resolution._propose_then_verify(
            fact, source, [candidate], lambda *_: "{}",
            evidence_packet=verify.build_service_evidence_packet(fact))

    assert line.chosen.code == "CODE_A"
    assert core.call_args.args[3] == []


def test_candidate_set_snapshot_is_stable_and_binds_descriptor_identity():
    first = CandidateCode(
        code="CODE_B", system="cpt", descriptor="assembly removal", score=0.8,
        source="retrieval", authority={"evaluation_descriptor_snapshot": {
            "source_id": "synthetic", "sha256": "sha256:one"}})
    second = CandidateCode(
        code="CODE_A", system="cpt", descriptor="assembly repair", score=0.9,
        source="umls_recall", authority={"evaluation_descriptor_snapshot": {
            "source_id": "synthetic", "sha256": "sha256:one"}})

    a = resolution._candidate_set_snapshot(
        [second, first], (" Assembly   Repair ", "assembly repair"), "service-A")
    b = resolution._candidate_set_snapshot(
        [second, first], ("assembly repair",), "service-A")

    assert a == b
    assert a["normalized_queries"] == ["assembly repair"]
    assert [row["code"] for row in a["candidates"]] == ["CODE_A", "CODE_B"]
    assert all(len(row["descriptor_sha256"]) == 64 for row in a["candidates"])


def test_model_disagreement_preserves_recall_candidate_as_system_unresolved():
    from claude_coder.verify import CandidateDispositionEvidence, Judgement

    fact = _fact("F1", "assembly repair", "assembly repair performed", "s1")
    candidate = CandidateCode(
        code="CODE_A", system="cpt", descriptor="assembly repair", source="retrieval")
    digest = verify._descriptor_sha256(candidate)
    supported = CandidateDispositionEvidence(
        candidate_code="CODE_A", descriptor_sha256=digest, status="entailed",
        evidence_span_ids=("s1",))
    rejected = CandidateDispositionEvidence(
        candidate_code="CODE_A", descriptor_sha256=digest,
        status="different_concept", evidence_span_ids=("s1",))
    result = resolution._candidate_disposition_uniqueness(
        [candidate], candidate,
        [Judgement(candidate_dispositions=(supported,)),
         Judgement(candidate_dispositions=(rejected,))],
        _reconciliation("s1"), coverage=None, fact=fact,
        admissions={"CODE_A": resolution.CandidateAdmission(
            ("CODE_A", "cpt"), resolution.CandidateStanding.UNGROUNDED,
            (), (), (), (), {}, ("retrieval",))})

    assert result is not None
    remaining, eliminated, unresolved = result
    assert remaining == []
    assert eliminated == {}
    assert "CODE_A" in unresolved


def test_cross_run_variance_retains_candidate_and_holds_only_its_line():
    fact = _fact("F1", "assembly repair", "assembly repair performed", "s1")
    candidate = CandidateCode(
        code="CODE_A", system="cpt", descriptor="assembly repair", source="retrieval")
    snapshot = resolution._candidate_set_snapshot(
        [candidate], (fact.description,), "service-A")
    packet = verify.build_service_evidence_packet(fact, service_context_id="service-A")
    current_record = {
        "stage": "code_selection_uniqueness",
        "candidate_set": snapshot,
        "evidence_packet": packet.as_record(),
        "still_entailed": ["CODE_A"],
        "eliminated": {},
    }
    prior_record = {
        **current_record,
        "still_entailed": [],
        "eliminated": {"CODE_A": "prior independently validated rejection"},
        "released": False,
        "code": "",
    }
    line = ResolvedLine(
        fact=fact, chosen=candidate, alternatives=[],
        method=ResolutionMethod.VERIFIED, tie_record=current_record)

    guarded = pipeline._apply_cross_run_selection_guard(line, [prior_record])

    assert guarded.chosen is None
    assert [c.code for c in guarded.alternatives] == ["CODE_A"]
    assert "SYSTEM_UNRESOLVED" in guarded.rationale
    assert guarded.tie_record["cross_run_variance"] == [{
        "system": "cpt", "code": "CODE_A",
        "prior_state": "eliminated", "current_state": "supported"}]


def test_cross_run_guard_uses_latest_exact_snapshot_so_retry_can_converge():
    fact = _fact("F1", "assembly repair", "assembly repair performed", "s1")
    candidate = CandidateCode(
        code="CODE_A", system="cpt", descriptor="assembly repair", source="retrieval")
    snapshot = resolution._candidate_set_snapshot(
        [candidate], (fact.description,), "service-A")
    packet = verify.build_service_evidence_packet(fact, service_context_id="service-A")
    current_record = {
        "stage": "code_selection_uniqueness", "candidate_set": snapshot,
        "evidence_packet": packet.as_record(), "still_entailed": ["CODE_A"],
        "eliminated": {},
    }
    old_conflict = {
        **current_record, "still_entailed": [],
        "eliminated": {"CODE_A": "old rejection"}, "released": False, "code": ""}
    latest_repeat = {
        **current_record, "released": False, "code": ""}
    line = ResolvedLine(
        fact=fact, chosen=candidate, method=ResolutionMethod.VERIFIED,
        tie_record=current_record)

    guarded = pipeline._apply_cross_run_selection_guard(
        line, [old_conflict, latest_repeat])

    assert guarded.chosen is candidate
    assert "cross_run_variance" not in guarded.tie_record
