"""Tri-state claim ownership (Phase-0).

Safety property: positive identity/organization evidence is required for ownership;
unknown cannot authorize retrieval or release. Decisions use actor IDs, not names.
Agnostic — synthetic ids, no code."""
from claude_coder import ownership as own
from claude_coder import gates
from claude_coder.models import (Outcome, CodingResult, ResolvedLine, ClinicalFact,
                                 CandidateCode, FactKind, ResolutionMethod, Disposition)


def _billable_line(attrs):
    fact = ClinicalFact(FactKind.PROCEDURE, "a performed service", attributes=attrs,
                        disposition=Disposition.PERFORMED)
    return ResolvedLine(fact=fact, chosen=CandidateCode("AA111", "cpt", "svc"),
                        method=ResolutionMethod.VERIFIED)


# ---------------------------------------------------------------- classifier
def test_classify_ownership_tristate_by_id():
    assert own.classify_ownership("prov-1", "prov-1") is Outcome.PASS
    assert own.classify_ownership("prov-1", "prov-2") is Outcome.BLOCKED   # different actor
    assert own.classify_ownership(None, "prov-1") is Outcome.UNKNOWN       # unstated
    assert own.classify_ownership("prov-1", None) is Outcome.UNKNOWN
    assert own.classify_ownership(None, None) is Outcome.UNKNOWN
    assert own.classify_ownership("prov-1", "org-1", "org-1", "operator") is Outcome.PASS


def test_fact_ownership_reads_ids_not_names():
    o = own.fact_ownership(ClinicalFact(
        FactKind.PROCEDURE, "svc",
        attributes={"performer_id": "p1", "billing_entity_id": "p2",
                    "performer": "Dr. Someone"}))
    assert o.performer_id == "p1" and o.billing_entity_id == "p2"
    assert o.performer_name == "Dr. Someone"                               # display only


# ---------------------------------------------------------------- gate
def test_gate_blocks_only_on_positive_contrary_evidence():
    res = CodingResult("enc", "2026-08-01",
                       lines=[_billable_line({"performer_id": "anes-9",
                                              "billing_entity_id": "surg-1"})])
    assert gates.claim_ownership_gate(res).outcome is Outcome.BLOCKED


def test_gate_passes_when_owner_matches():
    res = CodingResult("enc", "2026-08-01",
                       lines=[_billable_line({"performer_id": "surg-1",
                                              "billing_entity_id": "surg-1"})])
    assert gates.claim_ownership_gate(res).outcome is Outcome.PASS


def test_gate_unstated_ownership_is_unknown_and_retryable():
    res = CodingResult("enc", "2026-08-01", lines=[_billable_line({})])
    g = gates.claim_ownership_gate(res)
    assert g.outcome is Outcome.UNKNOWN and g.retryable is True


def test_gate_passes_practitioner_on_behalf_of_organization():
    res = CodingResult("enc", "2026-08-01", lines=[_billable_line({
        "performer_id": "prov-1", "performer_function": "operator",
        "organization_id": "org-1", "billing_entity_id": "org-1"})])
    assert gates.claim_ownership_gate(res).outcome is Outcome.PASS


def test_gate_not_applicable_without_billable_lines():
    res = CodingResult("enc", "2026-08-01", lines=[])
    assert gates.claim_ownership_gate(res).outcome is Outcome.NOT_APPLICABLE
