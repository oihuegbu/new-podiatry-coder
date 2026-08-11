"""Adversarial acceptance tests for the enforced evidence/service-graph phases.

All identifiers are synthetic. The tests assert architectural boundaries and failure
routing, never a particular medical code or medical term.
"""
from __future__ import annotations

import copy
import json

import pytest

from claude_coder import certificate, eligibility, extraction, provenance, resolution
from claude_coder.data_access import MockSource
from claude_coder.models import (
    CandidateCode, ClinicalFact, CodingResult, Destination, Disposition, EvidenceSpan,
    FactKind, Outcome, RelationAssertion, RelationPredicate, RelationState,
    ResolvedLine, ResolutionMethod,
)
from claude_coder.pipeline import code_encounter


class CountingSource(MockSource):
    def __init__(self):
        candidate = CandidateCode("SYNTHETIC_SERVICE", "cpt", "Synthetic service", 0.9)
        super().__init__(records={(candidate.code, candidate.system): {"active": True}},
                         retrieval={("*", "cpt"): [candidate]})
        self.retrieval_calls = 0

    def retrieve(self, *args, **kwargs):
        self.retrieval_calls += 1
        return super().retrieve(*args, **kwargs)


def _payload(*, evidence="service performed", relations=None):
    return json.dumps({
        "facts": [{
            "fact_id": "F1", "kind": "procedure", "description": "synthetic service",
            "attributes": {"performer_id": "actor-1", "billing_entity_id": "actor-1"},
            "disposition": "performed_today", "negated": False,
            "certainty": "confirmed", "experiencer": "patient",
            "evidence": [evidence], "confidence": 0.99,
            "axis_confidence": {"occurrence": 0.99, "action": 0.99,
                                "evidence": 0.99, "temporal": 0.99,
                                "performer": 0.99, "relationship": 0.99},
        }],
        "relations": relations or [],
    })


def _run(source, payload, monkeypatch=None, audit=None):
    return code_encounter(
        "enc-phase", "service performed", "2026-08-01", source=source,
        extract_llm=lambda _s, _u: payload,
        arbitrate_llm=lambda _s, _u: '{"choice":0}',
        audit_repository=audit or provenance.NullAuditRepository())


def test_auto_hold_never_calls_retrieval():
    src = CountingSource()
    result = _run(src, _payload(evidence="fabricated quotation"))
    assert src.retrieval_calls == 0
    assert result.verdict.value == "BLOCKED"


def test_eligibility_exception_routes_system_hold_without_retrieval(monkeypatch):
    src = CountingSource()
    monkeypatch.setattr(eligibility, "evaluate",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    result = _run(src, _payload())
    assert src.retrieval_calls == 0
    assert result.destination is Destination.SYSTEM_HOLD


def test_missing_intent_routes_system_hold_without_retrieval(monkeypatch):
    src = CountingSource()
    monkeypatch.setattr(eligibility, "evaluate", lambda *_a, **_k: [])
    result = _run(src, _payload())
    assert src.retrieval_calls == 0
    assert result.destination is Destination.SYSTEM_HOLD


def test_invalid_relation_endpoint_holds_before_retrieval():
    src = CountingSource()
    rel = {"subject_event_id": "F1", "predicate": "part_of",
           "object_event_id": "UNKNOWN", "state": "asserted",
           "evidence_fact_ids": ["F1"], "confidence": 0.99}
    result = _run(src, _payload(relations=[rel]))
    assert src.retrieval_calls == 0
    assert result.destination is Destination.SYSTEM_HOLD


@pytest.mark.parametrize("mutation", [
    {"confidence": True},                                    # bool -> used to become 1.0
    {"confidence": "0.99"},
    {"confidence": float("inf")},
    {"axis_confidence": {"occurrence": True, "action": 0.9, "evidence": 0.9,
                         "temporal": 0.9, "performer": 0.9, "relationship": 0.9}},
    {"axis_confidence": {"occurrence": float("nan"), "action": 0.9, "evidence": 0.9,
                         "temporal": 0.9, "performer": 0.9, "relationship": 0.9}},
])
def test_malformed_confidence_holds_before_retrieval(mutation):
    """F6-R1: a malformed confidence must produce a retryable SYSTEM_HOLD with ZERO retrieval,
    never a coerced (frequently MAXIMUM) confidence that drives autonomy thresholds."""
    payload = json.loads(_payload())
    payload["facts"][0].update(mutation)
    src = CountingSource()
    result = _run(src, json.dumps(payload))
    assert src.retrieval_calls == 0
    assert result.destination is Destination.SYSTEM_HOLD
    assert all(ln.chosen is None for ln in result.lines)


def test_malformed_relation_confidence_holds_before_retrieval():
    src = CountingSource()
    rel = {"subject_event_id": "F1", "predicate": "reason_for", "object_event_id": "F1",
           "state": "asserted", "evidence_fact_ids": ["F1"], "confidence": True}
    result = _run(src, _payload(relations=[rel]))
    assert src.retrieval_calls == 0
    assert result.destination is Destination.SYSTEM_HOLD


def test_malformed_billing_context_holds_before_retrieval():
    """F6-R2: a malformed/duplicated participant roster fails closed before retrieval."""
    src = CountingSource()
    result = code_encounter(
        "enc-phase", "service performed", "2026-08-01", source=src,
        extract_llm=lambda _s, _u: _payload(),
        arbitrate_llm=lambda _s, _u: '{"choice":0}',
        audit_repository=provenance.NullAuditRepository(),
        billing_context={"billing_entity_id": "actor-1", "participants": [
            {"id": "actor-1", "type": "person", "roles": ["performer"]},
            {"id": "actor-1", "type": "person", "roles": ["scribe"]}]})
    assert src.retrieval_calls == 0
    assert result.destination is Destination.SYSTEM_HOLD


def test_audit_failure_holds_before_retrieval():
    class BrokenAudit:
        def append(self, *_a, **_k):
            raise OSError("storage unavailable")

    src = CountingSource()
    result = _run(src, _payload(), audit=BrokenAudit())
    assert src.retrieval_calls == 0
    assert result.destination is Destination.SYSTEM_HOLD


def test_raw_fact_is_rejected_by_every_code_retriever():
    fact = ClinicalFact(FactKind.PROCEDURE, "synthetic service", fact_id="F1")
    with pytest.raises(TypeError):
        resolution.resolve(fact, CountingSource())
    from claude_coder import em
    em_fact = ClinicalFact(FactKind.EM, "synthetic visit", fact_id="F2")
    with pytest.raises(TypeError):
        em.resolve_em(em_fact, CountingSource())


def test_production_extraction_populates_axes_actor_and_valid_relation():
    payload = {
        "facts": [
            {"fact_id": "F1", "kind": "procedure", "description": "first action",
             "attributes": {"performer_id": "actor-1", "performer_function": "operator",
                            "organization_id": "org-1"},
             "disposition": "performed_today", "certainty": "confirmed",
             "experiencer": "patient", "evidence": ["first action"], "confidence": 0.9,
             "axis_confidence": {a: 0.9 for a in
                                 ("occurrence", "action", "evidence", "temporal",
                                  "performer", "relationship")}},
            {"fact_id": "F2", "kind": "procedure", "description": "second action",
             "attributes": {"performer_id": "actor-1", "performer_function": "operator",
                            "organization_id": "org-1"},
             "disposition": "performed_today", "certainty": "confirmed",
             "experiencer": "patient", "evidence": ["second action"], "confidence": 0.9,
             "axis_confidence": {a: 0.9 for a in
                                 ("occurrence", "action", "evidence", "temporal",
                                  "performer", "relationship")}},
        ],
        "relations": [{"subject_event_id": "F1", "predicate": "separate_from",
                       "object_event_id": "F2", "state": "asserted",
                       "evidence_fact_ids": ["F1", "F2"], "confidence": 0.9}],
    }
    graph = extraction.extract_note(
        "first action; second action", lambda _s, _u: json.dumps(payload),
        {"billing_entity_id": "org-1"})
    provenance.anchor_facts("first action; second action", graph.facts, "doc-v1")
    relations = provenance.validate_relations(
        provenance.bind_relation_evidence(graph.relations, graph.facts), graph.facts,
        "first action; second action")
    assert all(f.axis_confidence.get("performer") == 0.9 for f in graph.facts)
    assert all(f.attributes["billing_entity_id"] == "org-1" for f in graph.facts)
    assert len(relations) == 1 and relations[0].evidence_span_ids


def test_duplicate_quote_occurrences_have_location_specific_ids():
    note = "same quote -- same quote"
    facts = [ClinicalFact(FactKind.DIAGNOSIS, "one", fact_id="F1",
                          evidence=[EvidenceSpan("same quote")]),
             ClinicalFact(FactKind.DIAGNOSIS, "two", fact_id="F2",
                          evidence=[EvidenceSpan("same quote")])]
    provenance.anchor_facts(note, facts, "version-1")
    left, right = facts[0].evidence[0], facts[1].evidence[0]
    assert left.start != right.start
    assert left.text_sha256 == right.text_sha256
    assert left.span_id != right.span_id


def test_certificate_binds_intents_relations_spans_and_audit_hashes():
    note = "service performed"
    fact = ClinicalFact(FactKind.PROCEDURE, "synthetic service", fact_id="F1",
                        attributes={"performer_id": "actor-1",
                                    "billing_entity_id": "actor-1"},
                        disposition=Disposition.PERFORMED,
                        evidence=[EvidenceSpan(note)], confidence=0.99)
    provenance.anchor_facts(note, [fact], "doc-v1")
    (intent,) = eligibility.evaluate([fact], [], "enc", "2026-08-01")
    relation = RelationAssertion("F1", RelationPredicate.REASON_FOR, "F2",
                                 state=RelationState.ASSERTED,
                                 evidence_span_ids=[fact.evidence[0].span_id], confidence=0.9)
    line = ResolvedLine(fact, CandidateCode("SYNTHETIC", "cpt", "Synthetic service"),
                        method=ResolutionMethod.DETERMINISTIC)
    base = CodingResult("enc", "2026-08-01", lines=[line],
                        claim_line_intents=[intent], relations=[relation],
                        audit_record_hashes=["audit-a"])
    original = certificate.build_certificate(base, note)["certificate_sha256"]
    changed = copy.deepcopy(base)
    changed.audit_record_hashes = ["audit-b"]
    assert certificate.build_certificate(changed, note)["certificate_sha256"] != original
    changed = copy.deepcopy(base)
    changed.claim_line_intents[0].decisions[0].detail = "changed decision"
    assert certificate.build_certificate(changed, note)["certificate_sha256"] != original
    changed = copy.deepcopy(base)
    changed.relations[0].state = RelationState.UNCERTAIN
    assert certificate.build_certificate(changed, note)["certificate_sha256"] != original
    changed = copy.deepcopy(base)
    changed.lines[0].fact.evidence[0] = EvidenceSpan(
        note, start=1, end=len(note) + 1, anchored=True,
        text_sha256=fact.evidence[0].text_sha256,
        document_sha256=fact.evidence[0].document_sha256,
        document_version="doc-v1", span_id="changed-span")
    assert certificate.build_certificate(changed, note)["certificate_sha256"] != original


def test_em_is_an_explicit_service_intent():
    fact = ClinicalFact(
        FactKind.EM, "synthetic visit", fact_id="F1",
        attributes={"performer_id": "actor-1", "billing_entity_id": "actor-1"},
        disposition=Disposition.PERFORMED,
        evidence=[EvidenceSpan("visit", anchored=True, span_id="span")], confidence=0.9)
    (intent,) = eligibility.evaluate([fact], [], "enc", "2026-08-01")
    assert intent.component is eligibility.ClaimComponent.SERVICE
    assert intent.state is eligibility.EligibilityState.ELIGIBLE_FOR_RETRIEVAL


def test_jsonl_audit_chain_and_legacy_bridge(tmp_path):
    path = tmp_path / "enc.jsonl"
    path.write_text(json.dumps({"kind": "legacy", "record": {"x": 1}}) + "\n")
    repo = provenance.JsonlAuditRepository(tmp_path)
    first = repo.append("enc", "new", {"x": 2})
    second = repo.append("enc", "newer", {"x": 3})
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert rows[-2]["record_sha256"] == first
    assert rows[-1]["previous_record_sha256"] == first
    assert rows[-1]["record_sha256"] == second
