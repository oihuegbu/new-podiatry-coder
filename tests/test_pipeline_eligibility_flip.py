"""Phase 1c flip: retrieval consumes the eligibility decision. A billable service reaches
code retrieval only via an ELIGIBLE intent; one the engine marks NON_CLAIM (documented
integral component) is diverted BEFORE retrieval and never resolves to a code. Agnostic —
synthetic codes + stub LLMs, no API key."""
from claude_coder.data_access import MockSource
from claude_coder.pipeline import code_encounter
from claude_coder.models import CandidateCode

_FACTS = ('{"facts":[{"kind":"procedure","description":"excision of lesion",'
          '"attributes":{"performer_id":"actor-1","billing_entity_id":"actor-1"},'
          '"disposition":"performed_today","negated":false,'
          '"evidence":["excision of lesion performed"],"confidence":0.99}]}')
_NOTE = "excision of lesion performed today"


def _src():
    return MockSource(records={("PROC_X", "cpt"): {"active": True}},
                      retrieval={("*", "cpt"): [CandidateCode("PROC_X", "cpt",
                                                              "Excision, lesion, each", 0.9)]})


def _sel(system, user):
    sl = system.lower()
    if "propose" in sl:
        return '{"codes":[]}'
    if "independently" in sl:
        return '{"entailed":true,"missing_element":false,"reason":"x"}'
    return '{"choice":1,"reason":"x"}'


def test_eligible_service_reaches_retrieval_and_resolves():
    """Baseline: with the real engine (empty relations) a performed service is ELIGIBLE
    and resolves as today."""
    r = code_encounter("e", _NOTE, "2026-03-14", source=_src(),
                       extract_llm=lambda s, u: _FACTS, verify_llm=_sel,
                       corroborate_llm=_sel, audit_repository=_audit(),
                       billing_context={"billing_entity_id": "actor-1", "performer_id": "actor-1"})
    assert any(ln.chosen and ln.chosen.code == "PROC_X" for ln in r.lines)


def test_integral_component_diverted_before_retrieval(monkeypatch):
    """Flip: when the engine marks a BILLABLE service NON_CLAIM (documented integral), the
    pipeline diverts it BEFORE retrieval -- it never resolves to a code."""
    def fake_eval(fs, relations, enc, dos):
        from claude_coder.eligibility import (ClaimLineIntent, EligibilityState,
                                              ClaimComponent, EligibilityDecision)
        from claude_coder.models import Outcome
        return [ClaimLineIntent(
            intent_id="i", encounter_id=enc, component=ClaimComponent.SERVICE,
            clinical_event_ids=[f.fact_id], fact_kind=f.kind.value,
            clinical_action=f.description, attributes={}, date_of_service=dos,
            billing_entity_id=None, source_span_ids=[],
            state=EligibilityState.NON_CLAIM_EVIDENCE,
            decisions=[EligibilityDecision("part_of_demotion", Outcome.BLOCKED,
                                           "integral component of another event",
                                           "documented relationship")])
                for f in fs if f.billable]
    monkeypatch.setattr("claude_coder.eligibility.evaluate", fake_eval)

    r = code_encounter("e", _NOTE, "2026-03-14", source=_src(),
                       extract_llm=lambda s, u: _FACTS, verify_llm=_sel,
                       corroborate_llm=_sel, audit_repository=_audit())
    # nothing was retrieved/billed for the diverted service
    assert not any(ln.chosen and ln.chosen.code == "PROC_X" for ln in r.lines)
    diverted = [ln for ln in r.lines if "diverted before retrieval" in ln.rationale]
    assert diverted and "integral component" in diverted[0].rationale


def _audit():
    from claude_coder.provenance import NullAuditRepository
    return NullAuditRepository()
