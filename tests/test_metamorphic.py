"""Metamorphic tests — low-cost validation without a gold corpus (the feedback's
enabler). Each test changes ONE structural property of the input (laterality,
disposition, negation, count, specificity, an Excludes1 relationship) and asserts
the coded output changes in the REQUIRED way. Deterministic (MockSource + stub
LLMs) — a repeatable measurement of coding-correctness properties. Synthetic codes
only. Modeled on the retrocalcaneal-exostectomy note's structure.
"""
from claude_coder.models import (ClinicalFact, CandidateCode, ResolvedLine, ResolutionMethod,
                                 FactKind, EvidenceSpan, CodingResult, Disposition, Outcome)
from claude_coder.data_access import MockSource
from claude_coder import resolution, gates, extraction, ontology


def test_laterality_flip():
    src = MockSource(records={
        ("QQ01", "icd10"): {"long_description": "Widgetopathy of right gizmo"},
        ("QQ02", "icd10"): {"long_description": "Widgetopathy of left gizmo"}},
        index={"widgetopathy of gizmo": {"QQ0"}})
    def code(side):
        f = ClinicalFact(FactKind.DIAGNOSIS, "widgetopathy of gizmo", attributes={"laterality": side},
                         evidence=[EvidenceSpan(f"widgetopathy {side} gizmo")], disposition=Disposition.PERFORMED)
        return resolution.resolve(f, src).chosen
    r, l = code("right"), code("left")
    assert r and l and "right" in r.descriptor.lower() and "left" in l.descriptor.lower() and r.code != l.code


def test_disposition_flip():
    def billable(disp):
        return ClinicalFact(FactKind.PROCEDURE, "a procedure", disposition=disp,
                            evidence=[EvidenceSpan("done")]).billable
    assert billable(Disposition.PERFORMED) and not billable(Disposition.PLANNED)


def test_negation_drops():
    aff = extraction.extract_facts("x", llm=lambda s, u: '{"facts":[{"kind":"diagnosis","description":"finding","evidence":["finding"]}]}')
    neg = extraction.extract_facts("x", llm=lambda s, u: '{"facts":[{"kind":"diagnosis","description":"finding","negated":true,"evidence":["no finding"]}]}')
    assert len(aff) == 1 and len(neg) == 0


def test_count_scales_units():
    d = "Removal of gizmo, each"
    assert ontology.billing_units(1, d) == 1 and ontology.billing_units(3, d) == 3


def test_specificity_upgrade():
    src = MockSource(records={
        ("RR10", "icd10"): {"long_description": "Widgetopathy, unspecified"},
        ("RR19", "icd10"): {"long_description": "Widgetopathy of right gizmo"}})
    f = ClinicalFact(FactKind.DIAGNOSIS, "widgetopathy right gizmo", attributes={"laterality": "right"},
                     evidence=[EvidenceSpan("widgetopathy right gizmo")], disposition=Disposition.PERFORMED)
    line = ResolvedLine(fact=f, chosen=CandidateCode("RR10", "icd10", "Widgetopathy, unspecified", 1.0),
                        method=ResolutionMethod.VERIFIED)
    out = resolution.refine_diagnosis_specificity(
        line, src, lambda s, u: '{"choice": 2, "reason":"more specific"}',
        lambda s, u: '{"entailed": true, "missing_element": false, "reason":"ok"}')
    assert out.resolved and out.chosen.code == "RR1.9"


def test_excludes1_flip():
    src = MockSource(records={("XX01", "icd10"): {}, ("YY01", "icd10"): {}}, excludes1={"XX0": {"YY0"}})
    def result(codes):
        return CodingResult("e", "2026-01-05", lines=[ResolvedLine(
            fact=ClinicalFact(FactKind.DIAGNOSIS, c, evidence=[EvidenceSpan(c)], disposition=Disposition.PERFORMED),
            chosen=CandidateCode(c, "icd10", "d", 1.0), method=ResolutionMethod.VERIFIED) for c in codes])
    assert gates.icd_excludes_gate(result(["XX01"]), src).outcome is Outcome.NOT_APPLICABLE
    assert gates.icd_excludes_gate(result(["XX01", "YY01"]), src).outcome is Outcome.UNKNOWN


def test_failure_class_routing():
    """Operational failure routes to SYSTEM_HOLD, a documentation gap to
    PROVIDER_QUERY — not one undifferentiated review queue."""
    from claude_coder.models import GateResult, Destination
    from claude_coder.autonomy import decide
    proc = ResolvedLine(
        fact=ClinicalFact(FactKind.PROCEDURE, "svc", evidence=[EvidenceSpan("svc")],
                          disposition=Disposition.PERFORMED, confidence=0.99),
        chosen=CandidateCode("AAA1", "cpt", "d", 1.0), method=ResolutionMethod.VERIFIED)
    gap = ResolvedLine(
        fact=ClinicalFact(FactKind.PROCEDURE, "unclear svc", evidence=[EvidenceSpan("unclear")],
                          disposition=Disposition.PERFORMED),
        chosen=None, method=ResolutionMethod.ABSTAINED)
    gap.documentation_gap = "needs a qualifier the note omits"
    r = CodingResult("e", "2026-01-05", lines=[proc, gap], gates=[
        GateResult("ncci_ptp", Outcome.UNKNOWN, "NCCI check unavailable", "NCCI PTP (data)", retryable=True)])
    decide(r)
    dests = {x["destination"] for x in r.routing}
    assert Destination.SYSTEM_HOLD.value in dests   # operational -> retry
    assert Destination.PROVIDER_QUERY.value in dests  # doc gap -> provider


def test_materiality_secondary_dx_non_blocking():
    """An unresolved SECONDARY diagnosis (necessity met by a resolved dx) is
    non-material: it routes to PROVIDER_QUERY, non-blocking, and the defensible
    claim still releases AUTO_READY — instead of the whole encounter blocking."""
    from claude_coder.models import GateResult, Verdict, Destination
    from claude_coder.autonomy import decide
    def line(kind, code, conf=0.99):
        return ResolvedLine(
            fact=ClinicalFact(kind, f"{kind.value} thing", evidence=[EvidenceSpan("ev")],
                              disposition=Disposition.PERFORMED, confidence=conf),
            chosen=(CandidateCode(code, "cpt" if kind is FactKind.PROCEDURE else "icd10", "d", 1.0)
                    if code else None),
            method=(ResolutionMethod.VERIFIED if code else ResolutionMethod.ABSTAINED))
    proc = line(FactKind.PROCEDURE, "AAA1")
    dx_ok = line(FactKind.DIAGNOSIS, "D001")
    dx_open = line(FactKind.DIAGNOSIS, None)
    gates_ok = [GateResult(n, Outcome.PASS, "", "") for n in
                ("date_of_service", "verbatim_evidence", "code_active_on_dos",
                 "medical_necessity", "ncci_ptp", "mue", "icd_excludes1")]
    r = CodingResult("e", "2026-01-05", lines=[proc, dx_ok, dx_open], gates=gates_ok)
    decide(r)
    assert r.verdict is Verdict.AUTO_READY   # released despite the unresolved secondary dx
    assert any(x["destination"] == Destination.PROVIDER_QUERY.value and not x["blocking"]
               for x in r.routing)


def test_provider_query_is_self_contained():
    """A PROVIDER_QUERY routing item carries its suggested-solution recommendation,
    joined by the STABLE fact_id (not the non-unique description)."""
    from claude_coder.models import GateResult
    from claude_coder.autonomy import decide
    from claude_coder import recommendations as recs, pipeline
    gap = ResolvedLine(
        fact=ClinicalFact(FactKind.PROCEDURE, "svc needing detail", evidence=[EvidenceSpan("svc")],
                          disposition=Disposition.PERFORMED, fact_id="f7"),
        chosen=None, method=ResolutionMethod.ABSTAINED)
    gap.documentation_gap = "laterality not documented"
    r = CodingResult("e", "2026-01-05", lines=[gap],
                     gates=[GateResult("date_of_service", Outcome.PASS, "", "")])
    decide(r)
    r.recommendations = recs.build_recommendations(r)
    pipeline._attach_recommendations(r)
    pq = [x for x in r.routing if x["destination"] == "PROVIDER_QUERY"]
    assert pq and pq[0]["fact_id"] == "f7"
    assert "recommendation" in pq[0]
    assert "confirm and document" in pq[0]["recommendation"]
    assert "laterality not documented" in pq[0]["recommendation"]
