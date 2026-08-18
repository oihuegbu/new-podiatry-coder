"""The validation ladder (product directive section 9) and deterministic routing (section 8).

Section 9 asks for eight rungs of accuracy evidence in place of a gold corpus that does
not exist. This file holds the rungs the suite did not already prove, and says plainly
which rungs cannot be populated yet:

  rung 1  schema / property invariants ......... here (routing taxonomy totality,
                                                 hold-owner declaration coverage)
  rung 2  metamorphic, ONE axis at a time ...... tests/test_metamorphic.py covers
                                                 laterality, status, count, negation,
                                                 specificity, Excludes1; the MEASUREMENT,
                                                 PERFORMER and DATE-OF-SERVICE axes the
                                                 directive names are added here
  rung 3  plausible alternatives rejected for
          the CORRECT axis .................... tests/test_tie_policy.py, plus the
                                                 data-generated case here
  rung 4  cases generated from authoritative
          descriptors/index, not hardcoded .... here
  rungs 5-7  real historical/adjudicated data .. STRUCTURE ONLY -- app/release/
                                                 outcome_ledger.py. Asserted here to
                                                 refuse unattributable data and to
                                                 report itself as unpopulated rather
                                                 than as a measured zero.
  rung 8  calibration by decision class and
          weakest axis ........................ tools/calibration_dataset.py

Section 8 replaced generic "human review" with explicit destinations. The routing
regressions here pin each reclassification: a documented event that is simply not
claim-eligible, an unavailable authority, an unsettled relationship, an event with no
clinical action, and a barely-documented axis each go where the directive says, and
NONE of them goes to a coder by default.

No medical code appears in this file. The rung-4 cases are generated from the loaded
authoritative data at run time and skip cleanly when that data is not present.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from claude_coder import eligibility as elig
from claude_coder import gates, resolution
from claude_coder.autonomy import SHAKY_EXTRACTION, decide
from claude_coder.data_access import MockSource
from claude_coder.models import (
    CandidateCode,
    ClinicalFact,
    CodingResult,
    Destination,
    Disposition,
    EvidenceSpan,
    FactKind,
    GateResult,
    Outcome,
    ResolutionMethod,
    ResolvedLine,
    Verdict,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

_GATES_OK = [GateResult(n, Outcome.PASS, "", "") for n in
             ("date_of_service", "verbatim_evidence", "code_active_on_dos",
              "medical_necessity", "ncci_ptp", "mue", "icd_excludes1")]


def _fact(kind=FactKind.PROCEDURE, *, description="a service",
          disposition=Disposition.PERFORMED, confidence=0.99, fact_id="f1",
          attributes=None, axis_confidence=None, **kw):
    return ClinicalFact(kind, description, attributes=dict(attributes or {}),
                        evidence=[EvidenceSpan("evidence")], disposition=disposition,
                        confidence=confidence, fact_id=fact_id,
                        axis_confidence=dict(axis_confidence or {}), **kw)


def _request(fact):
    if not fact.fact_id:
        fact.fact_id = "fact"
    intent = elig.ClaimLineIntent(
        intent_id=f"t-{fact.fact_id}", encounter_id="t",
        component=(elig.ClaimComponent.DIAGNOSIS_SUPPORT
                   if fact.kind is FactKind.DIAGNOSIS else elig.ClaimComponent.SERVICE),
        clinical_event_ids=[fact.fact_id], fact_kind=fact.kind.value,
        clinical_action=fact.description, attributes=dict(fact.attributes),
        date_of_service=None, billing_entity_id=None, source_span_ids=[],
        state=elig.EligibilityState.ELIGIBLE_FOR_RETRIEVAL,
        fact_digest=elig.fact_snapshot_digest(fact))
    return elig.RetrievalRequest(intent, fact)


def _destinations(result) -> set[str]:
    return {item["destination"] for item in result.routing}


# ==========================================================================
# RUNG 1 -- schema / property invariants of the routing taxonomy
# ==========================================================================

def test_every_eligibility_gate_outcome_declares_a_routing_owner():
    """No eligibility gate may reach production relying on the coder fallback.

    `eligibility.hold_owner` falls back to OWNER_CODER for an undeclared (gate,
    outcome) -- deliberately, because over-escalation is the safe direction. That
    fallback is a safety net, not the design: an unresolved gate with no declared owner
    means a real encounter lands in a generic coder queue, which is exactly what
    directive section 8 forbids. This test reads the engine's OWN source for every
    non-PASS decision it can construct and requires each to be declared.
    """
    source = (REPO_ROOT / "claude_coder" / "eligibility.py").read_text()
    tree = ast.parse(source)
    emitted: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "EligibilityDecision"
                and len(node.args) >= 2):
            continue
        gate_node, outcome_node = node.args[0], node.args[1]
        if not isinstance(gate_node, ast.Constant) or not isinstance(gate_node.value, str):
            continue
        # Outcome.X, or a conditional expression over two of them.
        outcomes = [n for n in ast.walk(outcome_node)
                    if isinstance(n, ast.Attribute)
                    and isinstance(n.value, ast.Name) and n.value.id == "Outcome"]
        for attr in outcomes:
            if attr.attr != "PASS":
                emitted.add((gate_node.value, attr.attr))

    assert emitted, "found no EligibilityDecision literals -- the scan is broken"
    undeclared = sorted(
        (gate, outcome) for gate, outcome in emitted
        if gate not in elig.NON_BLOCKING_GATES
        and elig.hold_owner(elig.EligibilityDecision(gate, Outcome[outcome], "")) ==
        elig.OWNER_CODER)
    assert not undeclared, (
        f"these eligibility outcomes have no declared routing owner and would fall "
        f"through to a generic coder queue: {undeclared}. Add each to "
        f"`eligibility._HOLD_OWNERS` with an explicit destination, or record it in "
        f"NON_BLOCKING_GATES if it never determines the state.")


def test_declared_hold_gates_are_all_real_gates():
    """The reverse direction: a declaration for a gate the engine cannot emit is stale
    routing config that silently stops covering anything."""
    source = (REPO_ROOT / "claude_coder" / "eligibility.py").read_text()
    for gate in sorted(elig.declared_hold_gates()):
        assert f'"{gate}"' in source, (
            f"`_HOLD_OWNERS` declares an owner for gate {gate!r}, which the engine "
            f"never emits -- remove the stale declaration")


def test_every_producer_destination_maps_onto_the_directives_taxonomy():
    """Totality: every `Destination` the router can emit has ONE canonical meaning.

    A producer destination with no mapping raises rather than being guessed at, and
    exactly one canonical destination is a release. Together these are what stop a new
    route from silently inheriting another route's release semantics.
    """
    from app.contracts.claim_bundle import (
        InvalidClaimBundle, ReleaseDestination, canonical_destination)

    mapped = {}
    for destination in Destination:
        mapped[destination] = canonical_destination(destination.value)
    assert mapped[Destination.AUTO_READY] is ReleaseDestination.AUTO_READY
    assert mapped[Destination.SYSTEM_HOLD] is ReleaseDestination.SYSTEM_RETRY
    assert mapped[Destination.PROVIDER_QUERY] is ReleaseDestination.AUTO_QUERY
    assert mapped[Destination.HOLD] is ReleaseDestination.NON_BILLABLE
    assert mapped[Destination.BLOCKED] is ReleaseDestination.BLOCKED
    assert mapped[Destination.REVIEW] is ReleaseDestination.REVIEW
    # exactly one release
    assert [d for d in ReleaseDestination if d is ReleaseDestination.AUTO_READY] == [
        ReleaseDestination.AUTO_READY]
    with pytest.raises(InvalidClaimBundle):
        canonical_destination("SOMETHING_NEW")


def test_no_named_destination_is_unreachable():
    """Every destination in the directive's taxonomy is actually routed to somewhere.

    `Destination.HOLD` (the directive's NON_BILLABLE/EXCLUDED) used to appear only in
    the precedence list, so every encounter whose documented events were all excluded
    reached a coder through the "no defensible billable line" catch-all. A destination
    that nothing can emit is not a taxonomy, it is a comment.
    """
    autonomy = (REPO_ROOT / "claude_coder" / "autonomy.py").read_text()
    routed = set(re.findall(r"route\(\s*Destination\.([A-Z_]+)", autonomy))
    expected = {d.name for d in Destination} - {"AUTO_READY"}   # set, never "routed"
    assert expected <= routed, (
        f"these destinations are never routed to and are therefore unreachable: "
        f"{sorted(expected - routed)}")


# ==========================================================================
# SECTION 8 -- the reclassifications, each pinned
# ==========================================================================

def test_an_encounter_of_only_non_claim_events_is_non_billable_not_a_coder_queue():
    """Nothing performed, nothing open -> NON_BILLABLE/EXCLUDED, not generic review.

    A note documenting only planned or historical events has no judgement left in it:
    every event was disposed of by an explicit rule. Routing it to a coder is the
    generic fallback the directive forbids.
    """
    planned = ResolvedLine(fact=_fact(disposition=Disposition.PLANNED, fact_id="p1"),
                           chosen=None, method=ResolutionMethod.ABSTAINED)
    historical = ResolvedLine(
        fact=_fact(FactKind.DIAGNOSIS, description="an old condition",
                   disposition=Disposition.HISTORICAL, fact_id="p2"),
        chosen=None, method=ResolutionMethod.ABSTAINED)
    result = CodingResult("e", "2026-01-05", lines=[planned, historical],
                          gates=list(_GATES_OK))
    decide(result)
    assert result.destination is Destination.HOLD
    assert _destinations(result) == {Destination.HOLD.value}
    assert result.verdict is Verdict.REVIEW_REQUIRED       # still not a release
    assert all("not a claim-eligible event" in item["reason"] for item in result.routing)


def test_an_encounter_whose_lines_were_all_bundled_is_non_billable():
    """Same destination by the other route: authoritative claim mechanics excluded
    every line (bundled / not separately reportable), so there is nothing to decide."""
    bundled = ResolvedLine(
        fact=_fact(fact_id="b1"), chosen=CandidateCode("AAA1", "cpt", "d", 1.0),
        method=ResolutionMethod.DETERMINISTIC,
        excluded_reason="not separately reportable per authoritative data")
    result = CodingResult("e", "2026-01-05", lines=[bundled], gates=list(_GATES_OK))
    decide(result)
    assert result.destination is Destination.HOLD
    assert result.routing[0]["reason"] == (
        "not separately reportable per authoritative data")


def test_a_genuinely_undecided_empty_claim_still_reaches_a_human():
    """The catch-all must survive for what it was for.

    An unresolved billable event with no documentation gap is a real coding judgement,
    and narrowing the NON_BILLABLE case must not swallow it.
    """
    unresolved = ResolvedLine(fact=_fact(fact_id="u1"), chosen=None,
                              method=ResolutionMethod.ABSTAINED,
                              rationale="no candidate could be grounded")
    result = CodingResult("e", "2026-01-05", lines=[unresolved], gates=list(_GATES_OK))
    decide(result)
    assert result.destination is Destination.REVIEW
    assert Destination.HOLD.value not in _destinations(result)


def test_a_system_hold_encounter_does_not_also_raise_a_coder_item(tmp_path=None):
    """A cause already named must not be restated as vague coding work.

    An encounter held by an unavailable authority with no billable lines used to get
    BOTH a SYSTEM_HOLD item and a "no defensible billable line was produced" REVIEW
    item. The destination was right (precedence), but the worklist showed the same
    encounter as partly a coder's -- which is the generic fallback wearing a second
    hat. Found by this phase's post-fix review.
    """
    result = CodingResult(
        "e", "2026-01-05", lines=[],
        gates=[GateResult("mue", Outcome.UNKNOWN, "MUE table unavailable",
                          "MUE (data)", retryable=True)])
    decide(result)
    assert result.destination is Destination.SYSTEM_HOLD
    assert _destinations(result) == {Destination.SYSTEM_HOLD.value}


def test_an_unavailable_coverage_authority_is_system_work_not_coding_work():
    """A dependency that did not answer is a retry (SYSTEM_RETRY), never a coder queue.

    The necessity gate held on an unavailable coverage authority but reported the hold
    as a non-retryable UNKNOWN, which the router can only read as "needs judgement".
    The directive names this exact case: "source refresh failure ... are system work".
    """
    class Boom(MockSource):
        def qualifying_dx_for(self, code, system="cpt"):
            raise RuntimeError("coverage lookup failed")

    boom = Boom(coverage={"GG01": {"D001"}})
    proc = ResolvedLine(fact=_fact(fact_id="s1"),
                        chosen=CandidateCode("GG01", "cpt", "d", 1.0),
                        method=ResolutionMethod.VERIFIED)
    dx = ResolvedLine(fact=_fact(FactKind.DIAGNOSIS, description="a condition",
                                 fact_id="s2"),
                      chosen=CandidateCode("D001", "icd10", "d", 1.0),
                      method=ResolutionMethod.VERIFIED)
    result = CodingResult("e", "2026-01-05", lines=[proc, dx],
                          gates=[g for g in _GATES_OK if g.name != "medical_necessity"])
    gate = gates.medical_necessity_gate(result, boom)
    assert gate.outcome is Outcome.UNKNOWN and gate.retryable, (
        "an unavailable coverage authority must be a RETRYABLE hold")
    result.gates.append(gate)
    decide(result, source=boom)
    assert result.destination is Destination.SYSTEM_HOLD
    assert Destination.REVIEW.value not in _destinations(result)


def test_a_documentation_hold_in_the_same_gate_is_not_laundered_into_a_retry():
    """The failure path of the fix above: a hold the note caused must NOT become
    retryable just because another hold in the same gate was a dependency failure.
    Retrying cannot make the record say something it does not say."""
    proc = ResolvedLine(fact=_fact(fact_id="s1"),
                        chosen=CandidateCode("GG01", "cpt", "d", 1.0),
                        method=ResolutionMethod.VERIFIED)
    dx = ResolvedLine(fact=_fact(FactKind.DIAGNOSIS, description="a condition",
                                 fact_id="s2"),
                      chosen=CandidateCode("D001", "icd10", "d", 1.0),
                      method=ResolutionMethod.VERIFIED)
    result = CodingResult("e", "2026-01-05", lines=[proc, dx], gates=[])
    # No relations at all -> the ungoverned no-linkage hold, which is about the record.
    gate = gates.medical_necessity_gate(result, MockSource())
    assert gate.outcome is Outcome.UNKNOWN
    assert not gate.retryable, (
        "a hold caused by what the note documents must never be reported as retryable")


def test_an_unsettled_relationship_is_a_provider_question_not_a_coder_queue():
    """Two readings disagreed about whether an event is integral or distinct.

    That is a code-changing fact the record does not state -- the same shape as the
    `axis_consensus` hold the previous phase reclassified -- so it must become one
    precise provider question. It reached a coder before this phase.
    """
    decision = elig.EligibilityDecision(
        "conflict", Outcome.UNKNOWN, "unresolved relationship(s): part_of", "x")
    assert elig.hold_owner(decision) == elig.OWNER_PROVIDER_QUERY


def test_an_unassignable_duplicate_mention_is_a_provider_question():
    """Same class: the record does not say whether the third mention was a third
    service or a re-description of one already documented. Nobody but the provider
    can answer that."""
    for gate in ("coreference", "coreference_assignment"):
        decision = elig.EligibilityDecision(gate, Outcome.UNKNOWN, "ambiguous", "x")
        assert elig.hold_owner(decision) == elig.OWNER_PROVIDER_QUERY, gate


def test_an_event_with_no_clinical_action_is_an_integrity_block():
    """An extracted event carrying no clinical action is an unusable graph node.

    Neither a coder nor the provider has anything to act on, and every later stage
    would be reasoning about a representation that says nothing. That is the
    directive's BLOCKED ("internally inconsistent or unverifiable integrity state").
    """
    decision = elig.EligibilityDecision(
        "documentation_minimum", Outcome.UNKNOWN, "no clinical action to search on", "x")
    assert elig.hold_owner(decision) == elig.OWNER_INTEGRITY


def test_unresolved_ownership_is_a_retry_and_contradicted_ownership_is_a_block():
    """The same gate at two outcomes has two owners, and neither is a coder: an
    unbound performer identity is the context resolver's job, while a performer the
    record says did not perform the service is an integrity state."""
    unresolved = elig.EligibilityDecision("actor_ownership", Outcome.UNKNOWN, "", "")
    contradicted = elig.EligibilityDecision("actor_ownership", Outcome.BLOCKED, "", "")
    assert elig.hold_owner(unresolved) == elig.OWNER_SYSTEM
    assert elig.hold_owner(contradicted) == elig.OWNER_INTEGRITY


def test_a_barely_documented_named_axis_asks_the_provider_not_a_coder():
    """A grounded code on a fact the note barely documents is a DOCUMENTATION problem.

    When the weakest axis is named, the open item is already a precise question about a
    specified claim field -- AUTO_QUERY. With no axis recorded the concern is diffuse
    and a coder is still the honest destination. Neither releases.
    """
    named = ResolvedLine(
        fact=_fact(confidence=0.99, fact_id="a1",
                   axis_confidence={"laterality": SHAKY_EXTRACTION - 0.01}),
        chosen=CandidateCode("AAA1", "cpt", "d", 1.0),
        method=ResolutionMethod.DETERMINISTIC)
    result = CodingResult("e", "2026-01-05", lines=[named], gates=list(_GATES_OK))
    decide(result)
    assert result.destination is Destination.PROVIDER_QUERY
    assert result.verdict is Verdict.REVIEW_REQUIRED
    assert "weakest axis 'laterality'" in result.routing[0]["reason"]

    diffuse = ResolvedLine(
        fact=_fact(confidence=SHAKY_EXTRACTION - 0.01, fact_id="a2"),
        chosen=CandidateCode("AAA1", "cpt", "d", 1.0),
        method=ResolutionMethod.DETERMINISTIC)
    result = CodingResult("e", "2026-01-05", lines=[diffuse], gates=list(_GATES_OK))
    decide(result)
    assert result.destination is Destination.REVIEW


def test_every_routed_item_carries_its_own_suggested_action():
    """Self-containment, checked for the routes this phase added.

    A routing item is meant to be actionable on its own -- `pipeline._attach_recommendations`
    joins each one to a provider-facing suggested solution by stable `fact_id`. The
    barely-documented-axis route fires on a RESOLVED line, and BOTH recommendation rules
    that existed required an UNRESOLVED one, so it would have produced a provider query
    with no suggested action at all. Found by the post-fix review of this phase, not by
    a failing test.
    """
    from claude_coder import pipeline, recommendations as recs

    line = ResolvedLine(
        fact=_fact(confidence=0.99, fact_id="sc1",
                   axis_confidence={"laterality": SHAKY_EXTRACTION - 0.01}),
        chosen=CandidateCode("AAA1", "cpt", "an authoritative descriptor", 1.0),
        method=ResolutionMethod.DETERMINISTIC)
    result = CodingResult("e", "2026-01-05", lines=[line], gates=list(_GATES_OK))
    decide(result)
    result.recommendations = recs.build_recommendations(result)
    pipeline._attach_recommendations(result)

    queries = [item for item in result.routing
               if item["destination"] == Destination.PROVIDER_QUERY.value]
    assert queries and queries[0]["fact_id"] == "sc1"
    assert "recommendation" in queries[0], (
        "a PROVIDER_QUERY with no suggested action is not an actionable question")
    assert "laterality" in queries[0]["recommendation"]


def test_a_non_billable_hold_names_the_rule_that_excluded_each_event():
    """A NON_BILLABLE encounter must say WHY per event, not just decline as a whole --
    "we are not billing this" with no reason is indistinguishable from a dropped line."""
    excluded = ResolvedLine(
        fact=_fact(fact_id="nb1"), chosen=CandidateCode("AAA1", "cpt", "d", 1.0),
        method=ResolutionMethod.DETERMINISTIC,
        excluded_reason="integral to another reported service")
    result = CodingResult("e", "2026-01-05", lines=[excluded], gates=list(_GATES_OK))
    decide(result)
    assert result.destination is Destination.HOLD
    assert all(item["reason"] and item["fact_id"] for item in result.routing)


def test_model_disagreement_alone_never_names_a_coder_as_the_destination():
    """The directive's hard rule, checked against the router's own source.

    "Model disagreement, source refresh failure, index failure, or output failure are
    system work." Every REVIEW route in `autonomy.decide` must therefore be justified
    by something other than two models differing.
    """
    autonomy = (REPO_ROOT / "claude_coder" / "autonomy.py").read_text()
    review_blocks = re.findall(
        r"route\(\s*Destination\.REVIEW\s*,(.*?)\)\n", autonomy, flags=re.S)
    assert review_blocks, "no REVIEW routes found -- the scan is broken"
    for block in review_blocks:
        lowered = block.lower()
        for forbidden in ("models disagree", "model disagreement", "did not agree",
                          "consensus failed", "disagreed"):
            assert forbidden not in lowered, (
                f"a REVIEW route is justified by model disagreement: {block.strip()!r}")


# ==========================================================================
# RUNG 2 -- metamorphic: vary exactly ONE axis
# ==========================================================================

def test_metamorphic_measurement_axis():
    """Vary ONLY the documented measurement. In range -> the interval-qualified code
    resolves; out of range -> the same code is eliminated and nothing bills."""
    descriptor = "excision, area 16 sq. cm. or less"
    src = MockSource(
        records={("AREA_C", "cpt"): {"long_description": descriptor, "active": True}},
        retrieval={("*", "cpt"): [CandidateCode("AREA_C", "cpt", descriptor, 0.9)]})

    def code_for(size):
        fact = _fact(description="excision", attributes={"size_sqcm": size},
                     fact_id=f"m{size}")
        return resolution.resolve(_request(fact), src).chosen

    assert code_for(10) is not None and code_for(10).code == "AREA_C"
    assert code_for(30) is None, (
        "a measurement outside the descriptor's own interval must eliminate it")


def test_metamorphic_performer_axis():
    """Vary ONLY who performed the service. Performed by the billing entity -> the
    ownership gate passes; performed by someone else -> it BLOCKS, and the encounter
    is blocked rather than sent to a coder to adjudicate."""
    def outcome_for(performer):
        line = ResolvedLine(
            fact=_fact(fact_id="o1", attributes={
                "performer_id": performer, "billing_entity_id": "ENTITY-1",
                "organization_id": "ORG-1"}),
            chosen=CandidateCode("AAA1", "cpt", "d", 1.0),
            method=ResolutionMethod.DETERMINISTIC)
        result = CodingResult("e", "2026-01-05", lines=[line])
        return gates.claim_ownership_gate(result)

    same = outcome_for("ENTITY-1")
    other = outcome_for("SOMEONE-ELSE")
    assert same.outcome is Outcome.PASS
    assert other.outcome is Outcome.BLOCKED
    result = CodingResult("e", "2026-01-05",
                          lines=[ResolvedLine(
                              fact=_fact(fact_id="o1"),
                              chosen=CandidateCode("AAA1", "cpt", "d", 1.0),
                              method=ResolutionMethod.DETERMINISTIC)],
                          gates=[*_GATES_OK, other])
    decide(result)
    assert result.destination is Destination.BLOCKED


def test_metamorphic_date_of_service_axis():
    """Vary ONLY the date of service. A candidate inactive on the claim's DOS must be
    eliminated before selection, and the SAME candidate on a date it is active must
    survive -- so a one-character date misread cannot quietly change the code."""
    class DatedSource(MockSource):
        def active_on(self, code, system, dos):
            if code == "OLDC" and str(dos) >= "2026-01-01":
                return Outcome.BLOCKED
            return Outcome.PASS

    src = DatedSource(
        records={("OLDC", "cpt"): {"long_description": "a retired service",
                                   "active": True}},
        retrieval={("*", "cpt"): [CandidateCode("OLDC", "cpt", "a retired service", 0.9)]})

    def code_for(dos):
        fact = _fact(description="a retired service", fact_id="d1")
        return resolution.resolve(_request(fact), src, dos=dos).chosen

    assert code_for("2025-06-01") is not None
    assert code_for("2026-06-01") is None


# ==========================================================================
# RUNGS 3 + 4 -- plausible alternatives, generated from authoritative data
# ==========================================================================

def _authoritative_lateral_records(limit: int = 40) -> list[tuple[str, str, str]]:
    """Real (code, descriptor, side) triples whose OWN descriptor states one side.

    Generated from the loaded authoritative record set at run time, so the cases track
    whatever edition is installed. Nothing is enumerated by hand -- this file contains
    no medical code.
    """
    from claude_coder.ontology import parse_descriptor

    path = REPO_ROOT / "data" / "codes" / "icd10cm_codes.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    records = raw.get("codes") if isinstance(raw, dict) else raw
    if isinstance(records, dict):
        items = sorted(records.items())
    elif isinstance(records, list):
        items = [(str(r.get("code") or ""), r) for r in records if isinstance(r, dict)]
    else:
        return []

    out: list[tuple[str, str, str]] = []
    for code, record in items:
        if len(out) >= limit:
            break
        if not isinstance(record, dict):
            continue
        descriptor = str(record.get("long_description")
                         or record.get("description") or "")
        if not descriptor:
            continue
        sides = parse_descriptor(descriptor).laterality
        if len(sides) == 1 and "bilateral" not in sides:
            out.append((str(code), descriptor, next(iter(sides))))
    return out


def test_rung4_authoritative_descriptors_generate_real_laterality_cases():
    """The generator itself must find cases, or rung 4 is silently empty."""
    cases = _authoritative_lateral_records()
    if not cases:
        pytest.skip("authoritative ICD-10-CM record set is not installed")
    assert len(cases) >= 5


def test_rung3_a_plausible_alternative_is_rejected_on_the_CONTRADICTED_axis():
    """For every generated case: documenting the OTHER side eliminates the candidate,
    documenting the SAME side keeps it AND credits the laterality axis by name.

    This is the property the directive asks rung 3 to prove -- not merely that the
    wrong candidate loses, but that it loses for the correct reason. A candidate
    eliminated for the wrong reason is a coincidence, and coincidences do not
    generalise to the next descriptor.
    """
    cases = _authoritative_lateral_records()
    if not cases:
        pytest.skip("authoritative ICD-10-CM record set is not installed")

    opposite = {"left": "right", "right": "left"}
    checked = 0
    for code, descriptor, side in cases:
        other = opposite.get(side)
        if other is None:
            continue
        candidate = CandidateCode(code, "icd10", descriptor, 0.9)

        contradicted = _fact(FactKind.DIAGNOSIS, description=descriptor,
                             attributes={"laterality": other}, fact_id="r3a")
        assert resolution._evaluate(contradicted, candidate) is None, (
            f"{code}: a descriptor stating {side!r} survived documentation of "
            f"{other!r}")

        agreeing = _fact(FactKind.DIAGNOSIS, description=descriptor,
                         attributes={"laterality": side}, fact_id="r3b")
        match = resolution._evaluate(agreeing, candidate)
        assert match is not None, f"{code}: the agreeing side was eliminated"
        assert any("laterality" in reason for reason in match.rationale), (
            f"{code}: the match does not record WHICH axis it was credited for")
        checked += 1
    assert checked >= 5


# ==========================================================================
# RUNGS 5-7 -- structure only. These must refuse to look populated.
# ==========================================================================

def _identifiable_bundle(**authority):
    """A minimal ClaimBundle carrying the identity an observation must be able to name.

    Built from the contract's own models rather than a pipeline run: this rung is about
    ATTRIBUTION, and a full deployment fixture would prove the pipeline, not the ledger.
    """
    from app.contracts.claim_bundle import (
        AuthorityBinding, BundleOrigin, ClaimBundle, EncounterIdentity,
        ReleaseDestination, ReleaseStatus, SourceDocument)

    fields = {"data_fingerprint": "data-v1",
              "database_snapshot_digest": "db-v1",
              "index_build_id": "index-v1",
              "model_profiles": {"extraction": "profile-a"}}
    fields.update(authority)
    return ClaimBundle(
        produced_by=BundleOrigin.CLAUDE_CODER,
        encounter=EncounterIdentity(
            encounter_id="E1", document_id="D1", date_of_service="2026-01-05",
            source_document=SourceDocument(filename="n.pdf", document_version="doc-v1")),
        authority=AuthorityBinding(**fields),
        release=ReleaseStatus(destination=ReleaseDestination.AUTO_READY))


def test_an_observation_is_keyed_to_the_exact_claim_data_and_model_versions(tmp_path):
    """Rung 7's linkage requirement, positively.

    The same observation re-ingested is a no-op (a remittance file replayed must not
    inflate the denial count), and the SAME encounter answered from a DIFFERENT data
    snapshot files under a different key -- otherwise a denial caused by a stale code
    table would be attributed to the run that fixed it.
    """
    from app.release.outcome_ledger import OutcomeLedger, Rung

    ledger = OutcomeLedger(tmp_path)
    bundle = _identifiable_bundle()
    body = {"denied": True, "carcs": ["CO-16"]}

    first = ledger.record(Rung.OUTCOME_FEEDBACK, bundle, body,
                          observed_at="2026-02-01", source="835")
    ledger.record(Rung.OUTCOME_FEEDBACK, bundle, body,
                  observed_at="2026-02-01", source="835")
    assert len(ledger.observations(Rung.OUTCOME_FEEDBACK)) == 1, "ingest is not idempotent"
    assert ledger.status(Rung.OUTCOME_FEEDBACK)["status"] == "POPULATED"

    other_snapshot = _identifiable_bundle(database_snapshot_digest="db-v2")
    second = ledger.record(Rung.OUTCOME_FEEDBACK, other_snapshot, body,
                           observed_at="2026-02-01", source="835")
    assert second.identity.key != first.identity.key, (
        "a different compiled-database snapshot must not share an identity key")
    assert len(ledger.observations(Rung.OUTCOME_FEEDBACK)) == 2

    for field in ("encounter_id", "document_version", "claim_fingerprint",
                  "data_fingerprint", "database_snapshot_digest"):
        assert first.identity.as_dict()[field], field


def test_a_bundle_that_binds_no_data_snapshot_cannot_receive_an_outcome(tmp_path):
    """The failure path: an artifact that never bound the data that answered it cannot
    be the subject of evidence, because nothing could be re-derived from it later."""
    from app.release.outcome_ledger import (
        OutcomeLedger, Rung, UnattributableObservation)

    ledger = OutcomeLedger(tmp_path)
    unbound = _identifiable_bundle(database_snapshot_digest="")
    with pytest.raises(UnattributableObservation) as excinfo:
        ledger.record(Rung.OUTCOME_FEEDBACK, unbound, {"denied": True},
                      observed_at="2026-02-01")
    assert "database_snapshot_digest" in str(excinfo.value)


def test_an_observation_without_an_exact_claim_identity_is_refused(tmp_path):
    """Rung 7's requirement, enforced: feedback must name the exact ClaimBundle and
    data/model versions. An observation that cannot is not weak evidence, it is no
    evidence, and storing it would silently pollute every later rate."""
    from app.release.outcome_ledger import (
        OutcomeLedger, Rung, UnattributableObservation)

    ledger = OutcomeLedger(tmp_path)
    with pytest.raises(UnattributableObservation):
        ledger.record(Rung.OUTCOME_FEEDBACK, {"schema_id": "not_a_claim_bundle"},
                      {"denied": True}, observed_at="2026-01-05")


def test_unpopulated_rungs_report_as_awaiting_data_never_as_a_measured_zero(tmp_path):
    """The honest state for this deployment.

    Rungs 5-7 need real submitted and adjudicated encounters, which do not exist here.
    They must say so -- a zero denial rate over zero denials is the single most
    misleading number this reporting could produce.
    """
    from app.release.outcome_ledger import OutcomeLedger, RungStatus

    status = OutcomeLedger(tmp_path).ladder_status()
    assert {r["rung"] for r in status["observation_rungs"]} == {5, 6, 7}
    for rung in status["observation_rungs"]:
        assert rung["status"] == RungStatus.AWAITING_DEPLOYMENT_DATA.value
        assert rung["observations"] == 0
    assert set(status["test_rungs"]) == {"1", "2", "3", "4"}
    assert status["control_mode"] == "OBSERVATIONAL"


def test_metrics_report_an_unmeasured_axis_as_unknown_not_as_a_number(tmp_path):
    """Every metric the directive lists is present, and each one that cannot be
    measured yet is `None` WITH a recorded reason -- so a dashboard cannot render an
    unmeasured axis as a good score."""
    from app.release.outcome_ledger import METRIC_NAMES, OutcomeLedger, metrics

    report = metrics([], OutcomeLedger(tmp_path))
    assert set(report["metrics"]) == set(METRIC_NAMES)
    for name in ("repeat_run_claim_stability", "denial_correction_rate",
                 "candidate_recall"):
        assert report["metrics"][name] is None
        assert name in report["unavailable"]


def test_a_corrupt_outcome_ledger_is_loud_not_empty(tmp_path):
    """The fix's own failure path: an unreadable ledger must raise, never be reported
    as a rung with no observations. A silently-empty measurement is the failure mode
    this whole module exists to avoid."""
    from app.release.outcome_ledger import LedgerError, OutcomeLedger, Rung

    ledger = OutcomeLedger(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    ledger.path(Rung.OUTCOME_FEEDBACK).write_text("{ not json")
    with pytest.raises(LedgerError):
        ledger.observations(Rung.OUTCOME_FEEDBACK)
    with pytest.raises(LedgerError):
        ledger.status(Rung.OUTCOME_FEEDBACK)
