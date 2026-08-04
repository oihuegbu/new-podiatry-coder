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


def test_materiality_from_authoritative_coverage():
    """Materiality of an unresolved diagnosis is decided by the AUTHORITATIVE
    dx->procedure coverage linkage (LCD qualifying_dx), not a proxy:
      (a) a procedure governed by NO policy (necessity unconfirmable) -> the
          unresolved dx BLOCKS — this is the exostectomy/Haglund case (28118 is
          ungoverned), where an unresolved principal indication must never release;
      (b) a governed procedure whose necessity is met by a RESOLVED qualifying dx ->
          an unresolved EXTRA dx is genuinely non-material -> AUTO_READY;
      (c) a governed procedure with NO resolved qualifying dx -> BLOCKS."""
    from claude_coder.models import GateResult, Verdict
    from claude_coder.autonomy import decide
    def proc(code):
        return ResolvedLine(fact=ClinicalFact(FactKind.PROCEDURE, "p", evidence=[EvidenceSpan("p")],
                            disposition=Disposition.PERFORMED, confidence=0.99),
                            chosen=CandidateCode(code, "cpt", "d", 1.0), method=ResolutionMethod.VERIFIED)
    def dx(code):
        return ResolvedLine(fact=ClinicalFact(FactKind.DIAGNOSIS, code or "unresolved dx",
                            evidence=[EvidenceSpan("dx")], disposition=Disposition.PERFORMED, confidence=0.99),
                            chosen=(CandidateCode(code, "icd10", "d", 1.0) if code else None),
                            method=(ResolutionMethod.VERIFIED if code else ResolutionMethod.ABSTAINED))
    gates_ok = [GateResult(n, Outcome.PASS, "", "") for n in
                ("date_of_service", "verbatim_evidence", "code_active_on_dos",
                 "medical_necessity", "ncci_ptp", "mue", "icd_excludes1")]
    src = MockSource(coverage={"GG01": {"DQ01"}})   # governed procedure GG01, qualifying dx DQ01
    def run(lines):
        r = CodingResult("e", "2026-01-05", lines=lines, gates=list(gates_ok))
        decide(r, source=src); return r.verdict
    assert run([proc("UU01"), dx("DQ01"), dx(None)]) is Verdict.REVIEW_REQUIRED   # (a) ungoverned -> block
    assert run([proc("GG01"), dx("DQ01"), dx(None)]) is Verdict.AUTO_READY        # (b) governed+met -> release
    assert run([proc("GG01"), dx("ZZ99"), dx(None)]) is Verdict.REVIEW_REQUIRED   # (c) governed, unmet -> block
    # (d) no source at all -> fail-closed
    r = CodingResult("e", "2026-01-05", lines=[proc("GG01"), dx("DQ01"), dx(None)], gates=list(gates_ok))
    decide(r)
    assert r.verdict is Verdict.REVIEW_REQUIRED


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


# ---- failure-path tests the mutation gate found were missing (kill the survivors) --
from claude_coder.models import GateResult, Verdict
from claude_coder.autonomy import decide, _line_confidence, _ARBITRATED_DISCOUNT, AUTONOMY_CONFIDENCE
_GATES_OK = [GateResult(n, Outcome.PASS, "", "") for n in
             ("date_of_service", "verbatim_evidence", "code_active_on_dos",
              "medical_necessity", "ncci_ptp", "mue", "icd_excludes1")]

def _dx(code, conf=0.99):
    return ResolvedLine(fact=ClinicalFact(FactKind.DIAGNOSIS, code or "unresolved dx",
                        evidence=[EvidenceSpan("dx")], disposition=Disposition.PERFORMED, confidence=conf),
                        chosen=(CandidateCode(code, "icd10", "d", 1.0) if code else None),
                        method=(ResolutionMethod.VERIFIED if code else ResolutionMethod.ABSTAINED))

def _proc(code, conf=0.99):
    return ResolvedLine(fact=ClinicalFact(FactKind.PROCEDURE, "p", evidence=[EvidenceSpan("p")],
                        disposition=Disposition.PERFORMED, confidence=conf),
                        chosen=CandidateCode(code, "cpt", "d", 1.0), method=ResolutionMethod.VERIFIED)


def test_materiality_no_procedure_blocks():
    """Fail-closed: with NO billed procedure, necessity cannot be confirmed, so an
    unresolved diagnosis BLOCKS (kills the no-procs return-True mutant)."""
    r = CodingResult("e", "2026-01-05", lines=[_dx("D001"), _dx(None)], gates=list(_GATES_OK))
    decide(r, source=MockSource(coverage={"GG01": {"D001"}}))
    assert r.verdict is Verdict.REVIEW_REQUIRED


def test_materiality_coverage_error_blocks():
    """Fail-closed: if the coverage lookup RAISES, an unresolved dx BLOCKS (kills the
    except-return-True mutant)."""
    class Boom(MockSource):
        def qualifying_dx_for(self, code, system="cpt"):
            raise RuntimeError("coverage lookup failed")
    r = CodingResult("e", "2026-01-05", lines=[_proc("GG01"), _dx("D001"), _dx(None)], gates=list(_GATES_OK))
    decide(r, source=Boom(coverage={"GG01": {"D001"}}))
    assert r.verdict is Verdict.REVIEW_REQUIRED


def test_arbitrated_confidence_is_discounted():
    """An ARBITRATED (single-model) line's confidence is discounted; a VERIFIED line's
    is not (kills the ARBITRATED is/is-not mutant)."""
    def conf(method):
        return _line_confidence(ResolvedLine(
            fact=ClinicalFact(FactKind.PROCEDURE, "p", evidence=[EvidenceSpan("p")],
                              disposition=Disposition.PERFORMED, confidence=0.9),
            chosen=CandidateCode("AAA1", "cpt", "d", 1.0), method=method))
    assert abs(conf(ResolutionMethod.ARBITRATED) - 0.9 * _ARBITRATED_DISCOUNT) < 1e-9
    assert conf(ResolutionMethod.VERIFIED) == 0.9


def test_confidence_floor_is_strict():
    """A line EXACTLY at the autonomy floor is NOT below it -> releases; the <=-mutant
    would wrongly block it (kills the floor Lt->LtE mutant)."""
    r = CodingResult("e", "2026-01-05",
                     lines=[_proc("AAA1", AUTONOMY_CONFIDENCE), _dx("D001", AUTONOMY_CONFIDENCE)],
                     gates=list(_GATES_OK))
    decide(r)
    assert r.verdict is Verdict.AUTO_READY
