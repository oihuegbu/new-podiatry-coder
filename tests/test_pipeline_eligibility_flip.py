"""Retrieval consumes eligibility without treating composition as non-reportability.

Every performed service in a composed group is evaluated independently; only duplicate
mentions share one canonical retrieval. Agnostic — synthetic codes + stub LLMs, no API key.
"""
from claude_coder.data_access import MockSource
from claude_coder.pipeline import code_encounter
from claude_coder.models import CandidateCode, Outcome
from tests import shortlist_verdict as _sv

_FACTS = ('{"facts":[{"fact_id":"F1","kind":"procedure","description":"excision of lesion",'
          '"attributes":{"performer_id":"actor-1","billing_entity_id":"actor-1"},'
          '"disposition":"performed_today","negated":false,'
          '"evidence":["excision of lesion performed"],"confidence":0.99}]}')
_NOTE = "excision of lesion performed today"


def _src():
    return MockSource(records={("PROC_X", "cpt"): {"active": True}},
                      retrieval={("*", "cpt"): [CandidateCode("PROC_X", "cpt",
                                                              "Excision, lesion, each", 0.9)]})


_sel = _sv.judge(pick=1, reason="x")


def test_eligible_service_reaches_retrieval_and_resolves():
    """Baseline: with the real engine (empty relations) a performed service is ELIGIBLE
    and resolves as today."""
    r = code_encounter("e", _NOTE, "2026-03-14", source=_src(),
                       extract_llm=lambda s, u: _FACTS, verify_llm=_sel,
                       audit_repository=_audit(),
                       billing_context={"billing_entity_id": "actor-1", "participants": [{"id": "actor-1", "type": "person", "roles": ["performer"]}]})
    assert any(ln.chosen and ln.chosen.code == "PROC_X" for ln in r.lines)


def test_service_accounting_gate_rejects_a_suppressed_performed_service(monkeypatch):
    """Defense in depth: a future eligibility regression cannot silently undercode."""
    def fake_eval(fs, relations, enc, dos, source=None):
        from claude_coder.eligibility import (ClaimLineIntent, EligibilityState,
                                              ClaimComponent, EligibilityDecision)
        from claude_coder.models import Outcome
        return [ClaimLineIntent(
            intent_id="i", encounter_id=enc, component=ClaimComponent.SERVICE,
            clinical_event_ids=[f.fact_id], fact_kind=f.kind.value,
            clinical_action=f.description, attributes={}, date_of_service=dos,
            billing_entity_id=None, source_span_ids=[],
            state=EligibilityState.NON_CLAIM_EVIDENCE,
            decisions=[EligibilityDecision("synthetic_suppression", Outcome.BLOCKED,
                                           "suppressed before retrieval",
                                           "documented relationship")])
                for f in fs if f.billable]
    monkeypatch.setattr("claude_coder.eligibility.evaluate", fake_eval)

    r = code_encounter("e", _NOTE, "2026-03-14", source=_src(),
                       extract_llm=lambda s, u: _FACTS, verify_llm=_sel,
                       audit_repository=_audit())
    # Nothing was retrieved, but the omission is now a typed, fact-scoped system
    # failure rather than a successful-looking non-claim classification.
    assert not any(ln.chosen and ln.chosen.code == "PROC_X" for ln in r.lines)
    gate = next(g for g in r.gates if g.name == "performed_service_accounting")
    assert gate.outcome is Outcome.UNKNOWN
    assert gate.retryable is True
    assert gate.affected_fact_ids == ("F1",)


def test_asserted_part_of_services_each_cross_retrieval_boundary():
    note = "excision of lesion alpha performed. repair of lesion beta performed."
    facts = ('{"facts":['
             '{"fact_id":"F1","kind":"procedure","description":"excision of lesion alpha",'
             '"attributes":{"performer_id":"actor-1","billing_entity_id":"actor-1"},'
             '"disposition":"performed_today","negated":false,'
             '"evidence":["excision of lesion alpha performed"],"confidence":0.99},'
             '{"fact_id":"F2","kind":"procedure","description":"repair of lesion beta",'
             '"attributes":{"performer_id":"actor-1","billing_entity_id":"actor-1"},'
             '"disposition":"performed_today","negated":false,'
             '"evidence":["repair of lesion beta performed"],"confidence":0.99}],'
             '"relations":[{"subject_event_id":"F2","object_event_id":"F1",'
             '"predicate":"part_of","state":"asserted",'
             '"evidence_fact_ids":["F1","F2"],"confidence":0.99}]}')
    candidates = {
        ("excision of lesion alpha", "cpt"): [
            CandidateCode("PROC_A", "cpt", "Excision, lesion alpha", 0.9)],
        ("repair of lesion beta", "cpt"): [
            CandidateCode("PROC_B", "cpt", "Repair, lesion beta", 0.9)],
    }
    src = MockSource(
        records={("PROC_A", "cpt"): {"active": True},
                 ("PROC_B", "cpt"): {"active": True}},
        retrieval=candidates,
        proc_index={"excision of lesion alpha": {"PROC_A"},
                    "repair of lesion beta": {"PROC_B"}})
    r = code_encounter(
        "e", note, "2026-03-14", source=src,
        extract_llm=lambda s, u: facts, verify_llm=_sel,
        audit_repository=_audit(),
        billing_context={"billing_entity_id": "actor-1", "participants": [
            {"id": "actor-1", "type": "person", "roles": ["performer"]}]})

    services = {ln.fact.fact_id: ln for ln in r.lines
                if ln.fact.kind.value == "procedure"}
    assert services.keys() == {"F1", "F2"}
    assert all(ln.retrieval_attempted for ln in services.values())
    assert {ln.chosen.code for ln in services.values() if ln.chosen} == {"PROC_A", "PROC_B"}
    accounting = next(g for g in r.gates if g.name == "performed_service_accounting")
    assert accounting.outcome is Outcome.PASS


def _audit():
    from claude_coder.provenance import NullAuditRepository
    return NullAuditRepository()
