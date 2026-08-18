"""Metamorphic tests — low-cost validation without a gold corpus (the feedback's
enabler). Each test changes ONE structural property of the input (laterality,
disposition, negation, count, specificity, an Excludes1 relationship) and asserts
the coded output changes in the REQUIRED way. Deterministic (MockSource + stub
LLMs) — a repeatable measurement of coding-correctness properties. Synthetic codes
only. Modeled on the retrocalcaneal-exostectomy note's structure.
"""
import json
from claude_coder.models import (ClinicalFact, CandidateCode, ResolvedLine, ResolutionMethod,
                                 FactKind, EvidenceSpan, CodingResult, Disposition, Outcome)
from claude_coder.data_access import MockSource
from claude_coder import resolution, gates, extraction, ontology


def _request(fact):
    from claude_coder.eligibility import (ClaimComponent, ClaimLineIntent,
                                          EligibilityState, RetrievalRequest,
                                          fact_snapshot_digest)
    if not fact.fact_id:
        fact.fact_id = "fact"
    intent = ClaimLineIntent(
        intent_id=f"test-{fact.fact_id}", encounter_id="test",
        component=(ClaimComponent.DIAGNOSIS_SUPPORT
                   if fact.kind is FactKind.DIAGNOSIS else ClaimComponent.SERVICE),
        clinical_event_ids=[fact.fact_id], fact_kind=fact.kind.value,
        clinical_action=fact.description, attributes=dict(fact.attributes),
        date_of_service=None, billing_entity_id=None, source_span_ids=[],
        state=EligibilityState.ELIGIBLE_FOR_RETRIEVAL,
        fact_digest=fact_snapshot_digest(fact))
    return RetrievalRequest(intent, fact)


def test_laterality_flip():
    src = MockSource(records={
        ("QQ01", "icd10"): {"long_description": "Widgetopathy of right gizmo"},
        ("QQ02", "icd10"): {"long_description": "Widgetopathy of left gizmo"}},
        index={"widgetopathy of gizmo": {"QQ0"}})
    def code(side):
        f = ClinicalFact(FactKind.DIAGNOSIS, "widgetopathy of gizmo", attributes={"laterality": side},
                         evidence=[EvidenceSpan(f"widgetopathy {side} gizmo")], disposition=Disposition.PERFORMED)
        return resolution.resolve(_request(f), src).chosen
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
        line, src, _sv.judge(pick=2, reason="more specific"),
        _sv.judge(pick=2, reason="ok"))
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
    """Materiality of an unresolved diagnosis follows the necessity gate's OWN RESOLVED
    BINDING — the claim-line diagnosis that actually justified each service — not a second,
    parallel re-derivation from coverage membership (Codex F6-R3, adjacent instance):
      (a) a procedure governed by NO policy (necessity unconfirmable) -> the
          unresolved dx BLOCKS — this is the exostectomy/Haglund case (28118 is
          ungoverned), where an unresolved principal indication must never release;
      (b) a governed procedure whose necessity the gate RESOLVED (encounter linkage AND a
          policy-qualifying diagnosis) -> an unresolved EXTRA dx is non-material -> AUTO_READY;
      (c) a governed procedure with NO resolved qualifying dx -> BLOCKS;
      (d) no source at all -> fail-closed;
      (e) a governed procedure whose covered dx is merely PRESENT on the claim, with no
          encounter linkage -> BLOCKS (coverage membership is not a justification)."""
    from claude_coder.models import GateResult, Verdict
    from claude_coder.autonomy import decide
    def proc(code, fid="P"):
        return ResolvedLine(fact=ClinicalFact(FactKind.PROCEDURE, "p", evidence=[EvidenceSpan("p")],
                            disposition=Disposition.PERFORMED, confidence=0.99, fact_id=fid),
                            chosen=CandidateCode(code, "cpt", "d", 1.0), method=ResolutionMethod.VERIFIED)
    def dx(code, fid=None):
        return ResolvedLine(fact=ClinicalFact(FactKind.DIAGNOSIS, code or "unresolved dx",
                            evidence=[EvidenceSpan("dx")], disposition=Disposition.PERFORMED,
                            confidence=0.99, fact_id=fid or f"D{code or 'x'}"),
                            chosen=(CandidateCode(code, "icd10", "d", 1.0) if code else None),
                            method=(ResolutionMethod.VERIFIED if code else ResolutionMethod.ABSTAINED))
    gates_ok = [GateResult(n, Outcome.PASS, "", "") for n in
                ("date_of_service", "verbatim_evidence", "code_active_on_dos",
                 "ncci_ptp", "mue", "icd_excludes1")]
    src = MockSource(coverage={"GG01": {"DQ01"}})   # governed procedure GG01, qualifying dx DQ01

    def _link(dx_event):
        """A record-grounded, anchored REASON_FOR edge from `dx_event` to the procedure."""
        from claude_coder.models import RelationAssertion, RelationPredicate, RelationState
        return RelationAssertion(dx_event, RelationPredicate.REASON_FOR, "P",
                                 state=RelationState.ASSERTED, confidence=0.99,
                                 evidence_span_ids=["span-1"],
                                 reconciliation_status="source_directional",
                                 reconciliation_evidence=["span-1"],
                                 assertion_origins=["origin-a"])

    def run(lines, relations=(), source=src):
        # the REAL gate writes the binding autonomy then reads -- one authority, not two
        r = CodingResult("e", "2026-01-05", lines=lines, gates=list(gates_ok),
                         relations=list(relations))
        if source is not None:
            r.gates.append(gates.medical_necessity_gate(r, source))
        decide(r, source=source)
        return r.verdict
    assert run([proc("UU01"), dx("DQ01"), dx(None)], [_link("DDQ01")]) \
        is Verdict.REVIEW_REQUIRED                                               # (a)
    assert run([proc("GG01"), dx("DQ01"), dx(None)], [_link("DDQ01")]) \
        is Verdict.AUTO_READY                                                    # (b)
    assert run([proc("GG01"), dx("ZZ99"), dx(None)], [_link("DZZ99")]) \
        is Verdict.REVIEW_REQUIRED                                               # (c)
    assert run([proc("GG01"), dx("DQ01"), dx(None)], [_link("DDQ01")], source=None) \
        is Verdict.REVIEW_REQUIRED                                               # (d)
    assert run([proc("GG01"), dx("DQ01"), dx(None)]) is Verdict.REVIEW_REQUIRED  # (e)


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
from claude_coder.autonomy import (decide, _line_confidence, _ARBITRATED_DISCOUNT,
                                    AUTONOMY_CONFIDENCE, SHAKY_EXTRACTION)
from tests import shortlist_verdict as _sv
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
    """Fail-closed: if the coverage lookup RAISES, the necessity gate cannot confirm the
    service and an unresolved dx BLOCKS (kills the except-return-True mutant)."""
    class Boom(MockSource):
        def qualifying_dx_for(self, code, system="cpt"):
            raise RuntimeError("coverage lookup failed")
    boom = Boom(coverage={"GG01": {"D001"}})
    r = CodingResult("e", "2026-01-05", lines=[_proc("GG01"), _dx("D001"), _dx(None)],
                     gates=[g for g in _GATES_OK if g.name != "medical_necessity"])
    nec = gates.medical_necessity_gate(r, boom)
    assert nec.outcome is Outcome.UNKNOWN and "unavailable" in nec.detail
    r.gates.append(nec)
    decide(r, source=boom)
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


def test_grounded_line_releases_on_closure_not_selfreport():
    """CLOSURE over self-report: a GROUNDED (VERIFIED) line whose extraction confidence
    is only MODERATE — well under the old 0.95 self-report floor but clear of the
    documentation-shakiness floor — still AUTO-RELEASES, because grounding + cleared
    gates are the release criterion, not the LLM's uncalibrated number. Kills the
    mutant that reinstates a self-report floor gate on grounded lines."""
    mid = (SHAKY_EXTRACTION + AUTONOMY_CONFIDENCE) / 2      # e.g. ~0.72: below old floor, above shaky
    assert SHAKY_EXTRACTION < mid < AUTONOMY_CONFIDENCE
    r = CodingResult("e", "2026-01-05",
                     lines=[_proc("AAA1", mid), _dx("D001", mid)], gates=list(_GATES_OK))
    decide(r)
    assert r.verdict is Verdict.AUTO_READY


def test_arbitrated_line_is_not_autonomous():
    """A single-model ARBITRATED pick is NOT grounded and never auto-releases, however
    high its confidence — it needs a coder. Kills the mutant that treats ARBITRATED as
    grounded (the `is ResolutionMethod.ARBITRATED` guard)."""
    ln = _proc("AAA1", 0.99)
    ln.method = ResolutionMethod.ARBITRATED
    r = CodingResult("e", "2026-01-05", lines=[ln, _dx("D001", 0.99)], gates=list(_GATES_OK))
    decide(r)
    assert r.verdict is Verdict.REVIEW_REQUIRED
    assert any("arbitrated" in n.lower() for n in r.notes)


def test_shaky_extraction_blocks_even_when_grounded():
    """A grounded code on a fact the note BARELY documents (extraction confidence below
    SHAKY_EXTRACTION) still gets a human — the uncertainty is in the documentation. A
    fact just AT the floor is not below it and releases. Kills the SHAKY_EXTRACTION
    boundary/removal mutants."""
    below = CodingResult("e", "2026-01-05",
                         lines=[_proc("AAA1", SHAKY_EXTRACTION - 0.01)], gates=list(_GATES_OK))
    decide(below)
    assert below.verdict is Verdict.REVIEW_REQUIRED
    at = CodingResult("e", "2026-01-05",
                      lines=[_proc("AAA1", SHAKY_EXTRACTION), _dx("D001", 0.99)], gates=list(_GATES_OK))
    decide(at)
    assert at.verdict is Verdict.AUTO_READY


def test_snomed_crosswalk_hit_is_verified_not_blindly_trusted():
    """A SNOMED CT crosswalk hit is a CANDIDATE that must be entailment-confirmed, not
    trusted deterministically — its default ICD map can be wrong for the documented
    condition (e.g. bursitis mapping to a spur code). With a verifier that REJECTS the
    mapped code, the fact must NOT resolve to it; with one that confirms, it does."""
    from claude_coder import resolution
    src = MockSource(records={("WW01", "icd10"): {"long_description": "Unrelated condition"}},
                     snomed={"documented thing": {"WW01"}})
    def fact():
        return ClinicalFact(FactKind.DIAGNOSIS, "documented thing",
                            evidence=[EvidenceSpan("documented thing")], disposition=Disposition.PERFORMED)
    def stub(entailed):
        return _sv.judge(entails=lambda d: bool(entailed), reason="x")
    reject = resolution.resolve(_request(fact()), src, llm=stub(False), corroborate=stub(False))
    assert not reject.resolved                              # crosswalk default not blindly accepted
    accept = resolution.resolve(_request(fact()), src, llm=stub(True), corroborate=stub(True))
    assert accept.resolved and accept.chosen.code == "WW01"  # confirmed -> resolves


# ---- assertion axes: an uncertain or non-patient condition is not coded (ICD-10-CM) --
def test_uncertain_or_nonpatient_fact_is_not_billable():
    """ICD-10-CM outpatient rules: a SUSPECTED/probable condition is never coded as
    confirmed, and a FAMILY-history / other-experiencer condition is not the patient's
    coded condition. Both must be non-billable regardless of disposition. Kills the
    mutant that drops either half of the assertion guard in ClinicalFact.billable."""
    confirmed = ClinicalFact(FactKind.DIAGNOSIS, "d", evidence=[EvidenceSpan("d")],
                             disposition=Disposition.PERFORMED)
    assert confirmed.billable                                 # baseline: a plain confirmed dx bills
    suspected = ClinicalFact(FactKind.DIAGNOSIS, "d", evidence=[EvidenceSpan("d")],
                             disposition=Disposition.PERFORMED, certain=False)
    assert not suspected.billable                             # suspected -> not coded as confirmed
    family = ClinicalFact(FactKind.DIAGNOSIS, "d", evidence=[EvidenceSpan("d")],
                          disposition=Disposition.PERFORMED, experiencer="family")
    assert not family.billable                                # family history -> not the patient's code


def test_extraction_sets_assertion_axes_and_drops_ruled_out():
    """Extraction must map the note's certainty/experiencer onto the fact (so the
    billable guard can act) and must DROP a ruled-out finding outright."""
    from claude_coder.extraction import extract_facts
    payload = {"facts": [
        {"kind": "diagnosis", "description": "possible stress fracture", "certainty": "suspected",
         "evidence": ["possible stress fracture"], "confidence": 0.9},
        {"kind": "diagnosis", "description": "diabetes in mother", "experiencer": "family",
         "evidence": ["mother has diabetes"], "confidence": 0.9},
        {"kind": "diagnosis", "description": "cellulitis", "certainty": "ruled_out",
         "evidence": ["cellulitis ruled out"], "confidence": 0.9},
        {"kind": "diagnosis", "description": "onychomycosis", "certainty": "confirmed",
         "evidence": ["onychomycosis"], "confidence": 0.9},
        # an UNRECOGNIZED certainty must fail closed (not coded as confirmed)
        {"kind": "diagnosis", "description": "vague finding", "certainty": "maybe-ish",
         "evidence": ["vague finding"], "confidence": 0.9},
        # an OMITTED certainty defaults to confirmed (a plainly documented condition)
        {"kind": "diagnosis", "description": "hallux valgus",
         "evidence": ["hallux valgus"], "confidence": 0.9},
    ]}
    facts = extract_facts("note", llm=lambda s, u: json.dumps(payload))
    by_desc = {f.description: f for f in facts}
    assert "cellulitis" not in by_desc                        # ruled_out dropped entirely
    assert not by_desc["possible stress fracture"].certain and not by_desc["possible stress fracture"].billable
    assert by_desc["diabetes in mother"].experiencer == "family" and not by_desc["diabetes in mother"].billable
    assert by_desc["onychomycosis"].certain and by_desc["onychomycosis"].billable
    assert not by_desc["vague finding"].certain and not by_desc["vague finding"].billable   # fail closed
    assert by_desc["hallux valgus"].certain and by_desc["hallux valgus"].billable           # omitted -> confirmed


# ---- defensibility invariants: evidence/candidate immutability + fail-closed defaults --
def test_evidence_span_is_immutable():
    """A captured evidence span is the atom of defensibility — it must be FROZEN so a
    verbatim quote can never be silently rewritten after capture (kills the
    EvidenceSpan frozen=True->False mutant)."""
    import dataclasses, pytest
    span = EvidenceSpan("verbatim quote")
    with pytest.raises(dataclasses.FrozenInstanceError):
        span.text = "tampered"


def test_candidate_code_is_immutable():
    """A candidate's code/descriptor/authority are copied from the authoritative
    source, never authored by the coder — the record must be FROZEN so it cannot be
    edited in flight (kills the CandidateCode frozen=True->False mutant)."""
    import dataclasses, pytest
    cand = CandidateCode("AAA1", "cpt", "descriptor from data")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cand.code = "BBB2"


def test_gate_result_defaults_not_retryable():
    """Fail-closed: a gate is NOT retryable unless a check EXPLICITLY marks it so —
    the default must be False, or an operational-vs-coding misclassification would
    silently route a real coding problem to a retry (kills the retryable
    False->True default mutant)."""
    g = GateResult("some_gate", Outcome.UNKNOWN)
    assert g.retryable is False


def test_extraction_rejects_malformed_facts():
    """Fail-closed input validation (Codex F6-R1): a fact with an UNRECOGNIZED kind or an
    empty description is a malformed claim-affecting assertion -- it must RAISE a typed
    error (the pipeline turns it into a retryable SYSTEM_HOLD with zero retrieval), never
    be silently dropped, because a silent drop can turn a real event into 'no findings'."""
    import pytest
    from claude_coder.extraction import extract_facts, ExtractionSchemaError
    bad_kind = {"facts": [
        {"kind": "not_a_kind", "description": "has description", "evidence": ["x"]}]}
    empty_desc = {"facts": [
        {"kind": "diagnosis", "description": "   ", "evidence": ["x"]}]}
    good = {"facts": [
        {"kind": "diagnosis", "description": "onychomycosis", "evidence": ["onychomycosis"]}]}
    with pytest.raises(ExtractionSchemaError):
        extract_facts("note", llm=lambda s, u: json.dumps(bad_kind))
    with pytest.raises(ExtractionSchemaError):
        extract_facts("note", llm=lambda s, u: json.dumps(empty_desc))
    facts = extract_facts("note", llm=lambda s, u: json.dumps(good))
    assert [f.description for f in facts] == ["onychomycosis"]   # the well-formed one resolves


def test_extraction_preserves_attributes():
    """A documented attribute set (anatomy/laterality/…) must reach the fact intact —
    it is what lets the deterministic resolver pick a specific code (kills the
    `attributes or {}` Or->And mutant, which would blank a present attribute dict)."""
    from claude_coder.extraction import extract_facts
    payload = {"facts": [{"kind": "diagnosis", "description": "bursitis",
                          "attributes": {"laterality": "left", "anatomy": "heel"},
                          "evidence": ["left heel bursitis"]}]}
    (fact,) = extract_facts("note", llm=lambda s, u: json.dumps(payload))
    assert fact.attributes == {"laterality": "left", "anatomy": "heel"}


def test_default_llm_forwards_json_mode_and_zero_temperature():
    """The production LLM wrapper must request STRICT JSON at temperature 0.0 — a fact
    extractor that let the model prose-wrap or sample would break deterministic parsing
    (kills the `json_mode=True` True->False mutant)."""
    import app.core.llm_client as client
    seen = {}
    orig = client.chat_completion
    client.chat_completion = lambda s, u, **kw: (seen.update(kw), ('{"facts": []}', {}))[1]
    try:
        from claude_coder.extraction import _default_llm
        _default_llm("sys", "user")
    finally:
        client.chat_completion = orig
    assert seen.get("json_mode") is True and seen.get("temperature") == 0.0


# ---- NCCI gate must consult the DOS-scoped data and fail CLOSED off-quarter --------
class _FakeNcciRef:
    """Stand-in for CodeReferenceDB: records the args check_ncci was called with and
    controls whether the loaded snapshot covers the DOS."""
    def __init__(self, available=True, edit=None):
        self._available = available
        self._edit = edit
        self.seen = None

    def ncci_data_available(self, dos):
        return self._available

    def check_ncci(self, c1, c2, dos=None):
        self.seen = (c1, c2, dos)
        return self._edit


def test_ncci_indicator_forwards_dos_and_fails_closed_off_quarter():
    """Bug A: the DOS is forwarded to check_ncci (not dropped). Bug B: when the loaded
    NCCI snapshot does NOT cover the DOS, the indicator is AUTHORITY_UNAVAILABLE (the
    gate routes UNKNOWN/SYSTEM_HOLD), NEVER a clean None that would silently PASS."""
    from claude_coder.data_access import AuthoritativeSource, AUTHORITY_UNAVAILABLE
    src = AuthoritativeSource()
    # covered DOS + a real edit -> returns the indicator AND forwarded the DOS
    fake = _FakeNcciRef(available=True, edit={"code1": "A", "code2": "B",
                                              "modifier_indicator": "0"})
    src._db = fake
    assert src.ncci_indicator("A", "B", "2026-08-01") == "0"
    assert fake.seen == ("A", "B", "2026-08-01")          # DOS forwarded, not dropped
    # covered DOS + no edit for this pair -> genuinely None
    src._db = _FakeNcciRef(available=True, edit=None)
    assert src.ncci_indicator("A", "B", "2026-08-01") is None
    # snapshot does NOT cover the DOS -> UNAVAILABLE (fail closed), not None
    src._db = _FakeNcciRef(available=False, edit={"code1": "A", "code2": "B",
                                                  "modifier_indicator": "0"})
    assert src.ncci_indicator("A", "B", "2026-01-05") == AUTHORITY_UNAVAILABLE


def test_ncci_edit_forwards_dos():
    """Bug A: the directional edit lookup passes the DOS so the returned payable/
    component/modifier is the row active for the claim's DOS."""
    from claude_coder.data_access import AuthoritativeSource
    src = AuthoritativeSource()
    fake = _FakeNcciRef(available=True, edit={"code1": "A", "code2": "B", "modifier": "1"})
    src._db = fake
    assert src.ncci_edit("A", "B", "2026-08-01") == {
        "payable": "A", "component": "B", "modifier": "1"}
    assert fake.seen == ("A", "B", "2026-08-01")          # DOS forwarded, not dropped


# ---- #2: an OPPS (1833(t)) facility code is not separately reportable on the pro claim --
def test_opps_1833t_code_not_separately_billable():
    """A HCPCS code paid under OPPS (Social Security Act 1833(t)) -- a facility charge
    such as a device pass-through code -- is BLOCKED from the practitioner's professional
    claim, from its authoritative statute field. A code without that statute (and no other
    non-reportable signal) is kept. Agnostic: synthetic codes, statute-driven."""
    from claude_coder.data_access import AuthoritativeSource
    from claude_coder.models import Outcome

    class _Ref:
        hcpcs = {"OPPSX": {"code": "OPPSX", "statute": "1833(T)", "description": "device"},
                 "PAYX": {"code": "PAYX", "statute": None, "description": "payable supply"}}
        mue = {}

    src = AuthoritativeSource()
    src._db = _Ref()
    assert src.separately_billable("OPPSX", "hcpcs", "2026-08-01") is Outcome.BLOCKED
    assert src.separately_billable("PAYX", "hcpcs", "2026-08-01") is not Outcome.BLOCKED


# ---- #1: a residual/catch-all diagnosis needs distinctive descriptor grounding --------
def test_residual_catchall_without_grounding_escalates():
    """A residual/catch-all diagnosis code is entailed by almost anything in its bucket,
    so it must share a DISTINCTIVE clinical term with the documented condition or escalate.
    Agnostic: synthetic descriptors, no code named."""
    from claude_coder.resolution import _residual_without_grounding
    from claude_coder.models import ClinicalFact, CandidateCode, FactKind

    def fact(d):
        return ClinicalFact(FactKind.DIAGNOSIS, d)

    def cand(desc):
        return CandidateCode("X", "icd10", desc)

    # residual code sharing NO distinctive term with the condition -> ungrounded guess
    assert _residual_without_grounding(
        fact("Haglund-type retrocalcaneal exostosis"),
        cand("Other specified disorders of bone, ankle and foot"))
    # residual code that NAMES the documented condition (bursitis) -> grounded, kept
    assert not _residual_without_grounding(
        fact("Retrocalcaneal bursitis"),
        cand("Other bursitis, not elsewhere classified, right ankle and foot"))
    # a specific (non-residual) code -> not subject to the gate
    assert not _residual_without_grounding(
        fact("Calcification at Achilles tendon insertion"),
        cand("Calcific tendinitis, right ankle and foot"))


def test_residual_catchall_escalates_through_resolve():
    """End-to-end: resolve() escalates a diagnosis that verifies to a residual/catch-all
    code with no distinctive descriptor overlap, but KEEPS a residual code that names the
    documented condition. Exercises the wrapping guard in resolve(), not just the helper."""
    from claude_coder import resolution
    from claude_coder.data_access import MockSource
    from claude_coder.models import (ClinicalFact, CandidateCode, EvidenceSpan, FactKind,
                                      ResolutionMethod)

    stub = _sv.judge(pick=1, reason="x")             # entails the sole candidate

    # catch-all descriptor sharing no distinctive term with the condition -> escalate
    src = MockSource(
        records={("Z999", "icd10"): {"long_description": "Other specified disorders of bone, ankle and foot", "active": True}},
        retrieval={("*", "icd10"): [CandidateCode("Z999", "icd10", "Other specified disorders of bone, ankle and foot", 0.9)]})
    fact = ClinicalFact(FactKind.DIAGNOSIS, "Haglund-type retrocalcaneal exostosis",
                        evidence=[EvidenceSpan("Haglund-type retrocalcaneal exostosis")])
    line = resolution.resolve(_request(fact), src, llm=stub, corroborate=stub)
    assert not line.resolved and line.method is ResolutionMethod.ABSTAINED
    # routes to coder REVIEW, not PROVIDER_QUERY: no documentation_gap, and the
    # residual candidate is surfaced as an alternative for the coder to classify.
    assert line.documentation_gap is None
    assert any(a.code == "Z999" for a in line.alternatives)
    assert "classification" in line.rationale.lower()

    # residual code that NAMES the documented condition (bursitis) -> grounded, kept
    src2 = MockSource(
        records={("Z998", "icd10"): {"long_description": "Other bursitis, not elsewhere classified, right ankle and foot", "active": True}},
        retrieval={("*", "icd10"): [CandidateCode("Z998", "icd10", "Other bursitis, not elsewhere classified, right ankle and foot", 0.9)]})
    fact2 = ClinicalFact(FactKind.DIAGNOSIS, "Retrocalcaneal bursitis",
                         evidence=[EvidenceSpan("Retrocalcaneal bursitis")])
    line2 = resolution.resolve(_request(fact2), src2, llm=stub, corroborate=stub)
    assert line2.resolved and line2.chosen.code == "Z998"


# ---- reviewer-feedback fixes: surfacing, routing, gate transparency --------------
def test_unconfirmed_candidates_surface_as_coder_review_not_doc_gap():
    """When verification could not confirm a code but candidate code(s) WERE
    retrieved, the recommendation must NAME them and frame the open question as a
    coding decision (coder review) — not 'clarify the documentation'. A residual/
    'other'/NEC candidate is flagged. Unresolved DIAGNOSIS lines are included; a
    HISTORICAL (non-billable) condition produces no recommendation. Synthetic codes."""
    from claude_coder.recommendations import build_recommendations
    from claude_coder.models import (CodingResult, ResolvedLine, ClinicalFact,
                                     CandidateCode, FactKind, ResolutionMethod, Disposition)

    # unresolved billable DIAGNOSIS with a residual/NEC candidate retrieved
    dx = ClinicalFact(FactKind.DIAGNOSIS, "documented bursitis of the ankle")
    dx_line = ResolvedLine(
        fact=dx, chosen=None, method=ResolutionMethod.ABSTAINED,
        alternatives=[CandidateCode("QQ999", "icd10",
                      "Other bursitis, not elsewhere classified, right ankle and foot")],
        rationale="no candidate's authoritative descriptor is fully entailed (verified)")
    # a HISTORICAL condition -> not billable -> must NOT generate a recommendation
    hist = ClinicalFact(FactKind.DIAGNOSIS, "old unrelated condition",
                        disposition=Disposition.HISTORICAL)
    hist_line = ResolvedLine(fact=hist, chosen=None, method=ResolutionMethod.ABSTAINED,
                             alternatives=[CandidateCode("HH000", "icd10", "whatever")])
    # an unresolved procedure with NO candidates -> genuinely thin documentation
    proc = ClinicalFact(FactKind.PROCEDURE, "an ambiguous procedure")
    proc_line = ResolvedLine(fact=proc, chosen=None, method=ResolutionMethod.ABSTAINED,
                             alternatives=[], rationale="nothing retrieved")

    res = CodingResult("enc", "2026-08-01", lines=[dx_line, hist_line, proc_line])
    recs = build_recommendations(res)

    coder = [r for r in recs if r["issue"] == "coder_review"]
    assert any("ICD10 QQ999" in r["recommendation"] for r in coder), "candidate not named"
    assert any("residual" in r["recommendation"].lower() for r in coder), "residual not flagged"
    # the residual-code recommendation must NOT tell the provider to clarify documentation
    assert all("insufficient" not in r["recommendation"].lower() for r in coder)
    # historical / non-billable condition produced no recommendation
    assert not any(r["subject"] == "old unrelated condition" for r in recs)
    # no-candidate procedure falls back to a documentation-thin recommendation
    assert any(r["issue"] == "unresolved_service" and r["subject"] == "an ambiguous procedure"
               for r in recs)


def test_ncci_gate_reports_ptp_suppression_not_not_applicable():
    """When NCCI PTP bundled a component during reconciliation, leaving a single
    released procedure, the gate must report PASS 'PTP applied ... bundled' — not the
    misleading NOT_APPLICABLE 'fewer than two procedures'. With no suppression it is
    correctly NOT_APPLICABLE. Synthetic codes; MockSource needs no edit table (only
    one released line, so no pairwise lookup)."""
    from claude_coder.gates import ncci_gate
    from claude_coder.data_access import MockSource
    from claude_coder.models import (CodingResult, ResolvedLine, ClinicalFact,
                                     CandidateCode, FactKind, ResolutionMethod, Outcome)

    proc = ClinicalFact(FactKind.PROCEDURE, "a comprehensive procedure")
    line = ResolvedLine(fact=proc, chosen=CandidateCode("AA111", "cpt", "comprehensive"),
                        method=ResolutionMethod.VERIFIED)

    supp = CodingResult("enc", "2026-08-01", lines=[line],
                        ncci_suppressed=[("BB222", "AA111")])
    g = ncci_gate(supp, MockSource())
    assert g.outcome is Outcome.PASS
    assert "PTP applied" in g.detail and "BB222 bundled into AA111" in g.detail

    clean = CodingResult("enc", "2026-08-01", lines=[line], ncci_suppressed=[])
    assert ncci_gate(clean, MockSource()).outcome is Outcome.NOT_APPLICABLE


# ---- retrieval/decision hardening (manifest, DOS filter, proposal crowding) --------
def test_source_manifest_gate_fails_closed_on_missing_required(monkeypatch):
    """A MISSING REQUIRED authoritative source BLOCKS release (fail closed); absent
    OPTIONAL aids are recorded but PASS. gates imports build_manifest lazily, so we
    patch the capability module it reaches."""
    from claude_coder import capability, gates
    from claude_coder.models import CodingResult, Outcome

    res = CodingResult("enc", "2026-08-01")
    monkeypatch.setattr(capability, "build_manifest", lambda: {
        "missing_required": ["cpt_codes"], "degraded_optional": [], "sources": [],
        "status": "BLOCKED"})
    g = gates.source_manifest_gate(res)
    assert g.outcome is Outcome.BLOCKED and "cpt_codes" in g.detail

    monkeypatch.setattr(capability, "build_manifest", lambda: {
        "missing_required": [], "degraded_optional": ["cpt_synonyms"], "sources": [],
        "status": "OK"})
    g2 = gates.source_manifest_gate(res)
    assert g2.outcome is Outcome.PASS and "cpt_synonyms" in g2.detail


def test_dos_inactive_candidates_filtered_before_shortlist():
    """Fix3: a candidate DEFINITIVELY inactive on the DOS is dropped before it can take
    a scarce shortlist slot; UNKNOWN/PASS are kept; no DOS -> no filtering."""
    from claude_coder.resolution import _active_only
    from claude_coder.models import CandidateCode, Outcome

    class S:
        def active_on(self, code, system, dos):
            return {"OLD": Outcome.BLOCKED, "MAYBE": Outcome.UNKNOWN}.get(code, Outcome.PASS)

    cands = [CandidateCode("OLD", "cpt", "x"), CandidateCode("MAYBE", "cpt", "y"),
             CandidateCode("NEW", "cpt", "z")]
    kept = [c.code for c in _active_only(cands, S(), "2026-08-01")]
    assert kept == ["MAYBE", "NEW"]                       # only definitively-inactive dropped
    assert len(_active_only(cands, S(), None)) == 3        # no DOS -> keep all


def test_llm_proposals_cannot_crowd_out_retrieval():
    """Fix4: with 6 LLM proposals and 4 retrieved candidates whose correct one is ranked
    LAST, the reserved retrieval floor keeps it in the 8-slot shortlist so verification
    can select it. Under the old proposals-first ordering it would be crowded out and the
    line would escalate. Synthetic codes."""
    import json
    from claude_coder.resolution import resolve
    from claude_coder.data_access import MockSource
    from claude_coder.models import ClinicalFact, EvidenceSpan, CandidateCode, FactKind

    recs = {(f"P{i}", "cpt"): {"long_description": f"proposed service {i}", "active": True}
            for i in range(6)}
    recs[("RCODE", "cpt")] = {"long_description": "the correct retrieved service", "active": True}
    for d in ("D1", "D2", "D3"):
        recs[(d, "cpt")] = {"long_description": f"distractor {d}", "active": True}
    retrieved = [CandidateCode("D1", "cpt", "distractor D1", 0.9),
                 CandidateCode("D2", "cpt", "distractor D2", 0.8),
                 CandidateCode("D3", "cpt", "distractor D3", 0.7),
                 CandidateCode("RCODE", "cpt", "the correct retrieved service", 0.6)]
    src = MockSource(records=recs, retrieval={("*", "cpt"): retrieved})
    fact = ClinicalFact(FactKind.PROCEDURE, "the correct retrieved service",
                        evidence=[EvidenceSpan("the correct retrieved service performed")])

    stub = _sv.judge(entails=lambda d: "retrieved service" in d.lower(),
                     propose=["P0", "P1", "P2", "P3", "P4", "P5"], reason="retrieved")

    line = resolve(_request(fact), src, llm=stub, corroborate=stub)
    assert line.resolved and line.chosen.code == "RCODE"


# ---- Codex F6-R3: necessity is per-service, requiring an explicit REASON_FOR linkage -----
def test_necessity_requires_per_service_diagnosis_linkage():
    from claude_coder.models import (ClinicalFact, CodingResult, EvidenceSpan, FactKind,
                                      ResolutionMethod, ResolvedLine, CandidateCode,
                                      RelationAssertion, RelationPredicate, RelationState,
                                      Outcome)
    from claude_coder.gates import medical_necessity_gate

    def _line(code, kind, fid):
        f = ClinicalFact(kind=kind, description="x", attributes={}, fact_id=fid,
                         evidence=[EvidenceSpan("x")])
        sysname = "cpt" if kind is FactKind.PROCEDURE else "icd10"
        return ResolvedLine(fact=f, chosen=CandidateCode(code, sysname, "d", 0.9),
                            method=ResolutionMethod.DETERMINISTIC)

    proc = _line("P1", FactKind.PROCEDURE, "pf")
    dx = _line("D1", FactKind.DIAGNOSIS, "df")
    # a procedure and an UNRELATED diagnosis -> not defensible -> UNKNOWN (hold)
    unlinked = CodingResult("e", "2026-08-01", lines=[proc, dx])
    assert medical_necessity_gate(unlinked).outcome is Outcome.UNKNOWN
    # the diagnosis explicitly justifies the procedure (REASON_FOR) AND that edge is
    # evidence-anchored and independently reconciled by the deterministic layer -> PASS
    linked = CodingResult("e", "2026-08-01", lines=[proc, dx], relations=[
        RelationAssertion(subject_event_id="df", predicate=RelationPredicate.REASON_FOR,
                          object_event_id="pf", state=RelationState.ASSERTED,
                          confidence=0.99, evidence_span_ids=["span-1"],
                          reconciliation_status="source_directional",
                          reconciliation_evidence=["span-1"])])
    assert medical_necessity_gate(linked).outcome is Outcome.PASS
    # the SAME edge without independent reconciliation cannot certify (Codex F6-R3)
    unreconciled = CodingResult("e", "2026-08-01", lines=[proc, dx], relations=[
        RelationAssertion(subject_event_id="df", predicate=RelationPredicate.REASON_FOR,
                          object_event_id="pf", state=RelationState.ASSERTED,
                          confidence=0.99, evidence_span_ids=["span-1"])])
    assert medical_necessity_gate(unreconciled).outcome is Outcome.UNKNOWN
    # no diagnosis at all -> BLOCKED
    none_dx = CodingResult("e", "2026-08-01", lines=[proc])
    assert medical_necessity_gate(none_dx).outcome is Outcome.BLOCKED


# ---- Codex F6-R3 (round 2): confidence floor + authoritative coverage policy --------------
def _nec_lines():
    from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                      ResolutionMethod, ResolvedLine, CandidateCode)

    def _line(code, kind, fid):
        f = ClinicalFact(kind=kind, description="x", attributes={}, fact_id=fid,
                         evidence=[EvidenceSpan("x")])
        sysname = "cpt" if kind is FactKind.PROCEDURE else "icd10"
        return ResolvedLine(fact=f, chosen=CandidateCode(code, sysname, "d", 0.9),
                            method=ResolutionMethod.DETERMINISTIC)
    return _line("P1", FactKind.PROCEDURE, "pf"), _line("D1", FactKind.DIAGNOSIS, "df")


def _reason(conf, *, state=None, anchored=True, reconciled="source_directional",
            proof=("span-1",), corroboration="single_origin"):
    """A REASON_FOR edge. Defaults to the ONLY shape that may satisfy necessity: asserted,
    evidence-anchored, and stamped by the deterministic provenance layer with a status that
    GROUNDS it in the record, naming the span(s) that grounded it. `corroboration` is the
    separate agreement axis, which never releases anything on its own."""
    from claude_coder.models import RelationAssertion, RelationPredicate, RelationState
    return RelationAssertion(subject_event_id="df", predicate=RelationPredicate.REASON_FOR,
                             object_event_id="pf",
                             state=state or RelationState.ASSERTED, confidence=conf,
                             evidence_span_ids=["span-1"] if anchored else [],
                             reconciliation_status=reconciled,
                             reconciliation_evidence=list(proof),
                             corroboration_status=corroboration)


def test_necessity_rejects_zero_confidence_relation():
    from claude_coder.models import CodingResult, Outcome
    from claude_coder.gates import medical_necessity_gate
    proc, dx = _nec_lines()
    r = CodingResult("e", "2026-08-01", lines=[proc, dx], relations=[_reason(0.0)])
    assert medical_necessity_gate(r).outcome is Outcome.UNKNOWN            # low-confidence edge
    r2 = CodingResult("e", "2026-08-01", lines=[proc, dx], relations=[_reason(0.99)])
    assert medical_necessity_gate(r2).outcome is Outcome.PASS     # confident, anchored, reconciled


def test_necessity_requires_coverage_policy_qualification():
    from claude_coder.models import CodingResult, Outcome
    from claude_coder.gates import medical_necessity_gate
    from claude_coder.data_access import MockSource
    proc, dx = _nec_lines()                                                # proc P1, dx D1
    r = CodingResult("e", "2026-08-01", lines=[proc, dx], relations=[_reason(0.99)])
    gov_nonqualify = MockSource(); gov_nonqualify._coverage = {"P1": {"OTHERDX"}}
    assert medical_necessity_gate(r, gov_nonqualify).outcome is Outcome.UNKNOWN   # contradiction
    gov_qualify = MockSource(); gov_qualify._coverage = {"P1": {"D1"}}
    assert medical_necessity_gate(r, gov_qualify).outcome is Outcome.PASS         # dx qualifies


# ---- Codex F6-R3 (round 3): unreconciled/unanchored model edges cannot certify necessity ---
def test_necessity_rejects_unreconciled_high_confidence_relation():
    """The reviewer's exact reproduction: an edge that passed `validate_relations` with valid
    anchored evidence but was NEVER independently reconciled. The extraction model's own
    confidence is not independent clinical support, so it must HOLD, not PASS."""
    from claude_coder.models import CodingResult, Outcome
    from claude_coder.gates import medical_necessity_gate
    proc, dx = _nec_lines()
    r = CodingResult("e", "2026-08-01", lines=[proc, dx],
                     relations=[_reason(0.99, reconciled="unreconciled")])
    assert medical_necessity_gate(r).outcome is Outcome.UNKNOWN


def test_necessity_rejects_unanchored_relation():
    """A relation with no verified evidence references is not tied to the source document."""
    from claude_coder.models import CodingResult, Outcome
    from claude_coder.gates import medical_necessity_gate
    proc, dx = _nec_lines()
    r = CodingResult("e", "2026-08-01", lines=[proc, dx], relations=[_reason(0.99, anchored=False)])
    assert medical_necessity_gate(r).outcome is Outcome.UNKNOWN


def test_necessity_rejects_conflicting_edges():
    """A conflicting/uncertain duplicate edge disqualifies the pair -- a confident edge can
    never out-vote a documented disagreement."""
    from claude_coder.models import CodingResult, Outcome, RelationState
    from claude_coder.gates import medical_necessity_gate
    proc, dx = _nec_lines()
    for bad_state in (RelationState.UNCERTAIN, RelationState.NEGATED):
        r = CodingResult("e", "2026-08-01", lines=[proc, dx],
                         relations=[_reason(0.99), _reason(0.99, state=bad_state)])
        assert medical_necessity_gate(r).outcome is Outcome.UNKNOWN


def test_necessity_rejects_agreement_only_reconciliation():
    """Round 5: 'corroborated' is no longer a reconciliation status at all, and an edge whose
    only distinction is that several origins asserted it is not grounded in the record --
    whatever the corroboration axis says, and whatever a stale value in that field claims."""
    from claude_coder.models import CodingResult, Outcome
    from claude_coder.gates import medical_necessity_gate
    from claude_coder import provenance
    proc, dx = _nec_lines()
    for status in sorted(provenance.RETIRED_RECONCILIATION_STATUSES) + [
            provenance.SOURCE_COLOCATED, provenance.UNRECONCILED]:
        r = CodingResult("e", "2026-08-01", lines=[proc, dx],
                         relations=[_reason(0.99, reconciled=status,
                                            corroboration=provenance.MULTIPLY_ASSERTED)])
        assert medical_necessity_gate(r).outcome is Outcome.UNKNOWN, status
        assert not r.necessity_support, status
    # the grounded edge still releases, and being multiply asserted does not change that
    ok = CodingResult("e", "2026-08-01", lines=[proc, dx],
                      relations=[_reason(0.99, corroboration=provenance.MULTIPLY_ASSERTED)])
    assert medical_necessity_gate(ok).outcome is Outcome.PASS
    assert ok.necessity_support[0]["supports"][0]["corroboration_status"] \
        == provenance.MULTIPLY_ASSERTED


def test_necessity_rejects_a_grounded_status_that_names_no_source_span():
    """A grounded status with an empty evidence list would certify necessity while citing no
    source text -- the exact shape Codex reported (`reconciliation_evidence: []` on a PASS).
    Unreachable from the provenance layer now, and refused by the gate regardless."""
    from claude_coder.models import CodingResult, Outcome
    from claude_coder.gates import medical_necessity_gate
    proc, dx = _nec_lines()
    r = CodingResult("e", "2026-08-01", lines=[proc, dx], relations=[_reason(0.99, proof=())])
    assert medical_necessity_gate(r).outcome is Outcome.UNKNOWN
    assert not r.necessity_support


def test_necessity_control_cannot_readmit_an_agreement_only_status(tmp_path):
    """The invariant is structural, not a property of today's config file: a control that
    lists an agreement-only or observational status fails to load, so the gate ERRORs (autonomy
    stops) instead of releasing claims on model agreement. Restoring the round-4 revision of
    this control is exactly that edit."""
    import json
    import claude_coder.gates as g
    from claude_coder.models import CodingResult, Outcome
    from claude_coder import provenance
    from app.release import source_manifest as sm
    cfg = dict(g.load_necessity_control())
    assert set(cfg["accepted_reconciliation_statuses"]) \
        <= provenance.GROUNDED_RECONCILIATION_STATUSES
    proc, dx = _nec_lines()
    saved_cache, saved_registry = g._NECESSITY_CONTROL_CACHE, dict(sm._AUTHORITATIVE)
    for bad in ("corroborated", "externally_verified", provenance.SOURCE_COLOCATED,
                provenance.MULTIPLY_ASSERTED, provenance.UNRECONCILED):
        tmp = tmp_path / f"bad_control_{bad}.json"
        tmp.write_text(json.dumps(dict(
            cfg, accepted_reconciliation_statuses=[bad, "source_directional"])))
        try:
            g._NECESSITY_CONTROL_CACHE = None
            sm._AUTHORITATIVE[g._NECESSITY_CONTROL_ID] = tmp
            out = g.medical_necessity_gate(
                CodingResult("e", "2026-08-01", lines=[proc, dx], relations=[_reason(0.99)]))
            assert out.outcome is Outcome.ERROR, bad
            assert bad in out.detail
        finally:
            g._NECESSITY_CONTROL_CACHE = saved_cache
            sm._AUTHORITATIVE.clear()
            sm._AUTHORITATIVE.update(saved_registry)
            tmp.unlink(missing_ok=True)


def test_necessity_control_is_versioned_configuration_not_an_inline_constant():
    """The control floor/accepted statuses are reviewed config, and a missing/malformed control
    file is an ERROR (autonomy stops) -- never a silent built-in default."""
    import claude_coder.gates as g
    from claude_coder.models import CodingResult, Outcome
    from app.release import source_manifest as sm
    cfg = g.load_necessity_control()
    assert cfg["control_mode"] == "ENFORCED_FAIL_CLOSED" and cfg["authority"]
    # The control is a DECLARED release source, so the file the gate loads is the file the
    # release fingerprint content-addresses -- not a path composed inside the gate.
    control_path = sm.declared_source_path(g._NECESSITY_CONTROL_ID)
    assert control_path.exists()
    assert g._NECESSITY_CONTROL_ID in sm.required_release_sources()
    assert g.medical_necessity_gate(
        CodingResult("e", "2026-08-01", lines=list(_nec_lines()),
                     relations=[_reason(0.99)])).authority.endswith(f"[{cfg['version']}]")
    proc, dx = _nec_lines()
    saved_cache = g._NECESSITY_CONTROL_CACHE
    saved_registry = dict(sm._AUTHORITATIVE)
    try:
        g._NECESSITY_CONTROL_CACHE = None
        sm._AUTHORITATIVE[g._NECESSITY_CONTROL_ID] = (control_path.parent
                                                      / "does-not-exist.json")
        out = g.medical_necessity_gate(
            CodingResult("e", "2026-08-01", lines=[proc, dx], relations=[_reason(0.99)]))
        assert out.outcome is Outcome.ERROR           # fail closed, no default floor
    finally:
        g._NECESSITY_CONTROL_CACHE = saved_cache
        sm._AUTHORITATIVE.clear()
        sm._AUTHORITATIVE.update(saved_registry)


def test_reconciliation_status_is_written_only_by_the_deterministic_layer():
    """A model-authored `reconciliation_status` cannot survive: `validate_relations` restamps
    every edge from deterministic evidence, so claiming to be reconciled achieves nothing."""
    from claude_coder import provenance
    from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind, RelationAssertion,
                                     RelationPredicate, RelationState)
    note = "shared verbatim passage"
    span = EvidenceSpan("shared verbatim passage", anchored=True, text_sha256="h",
                        span_id="shared-1")
    dxf = ClinicalFact(FactKind.DIAGNOSIS, "dx", fact_id="df", evidence=[span])
    prf = ClinicalFact(FactKind.PROCEDURE, "proc", fact_id="pf", evidence=[span])
    forged = RelationAssertion("df", RelationPredicate.REASON_FOR, "pf",
                               state=RelationState.ASSERTED, confidence=0.99,
                               evidence_span_ids=["shared-1"],
                               reconciliation_status="externally_verified")
    # co-located ONLY: one passage mentions both endpoints but states no directional claim,
    # so the deterministic layer records the observation and the control does NOT accept it.
    out = provenance.validate_relations([forged], [dxf, prf], note)
    assert out[0].reconciliation_status == provenance.SOURCE_COLOCATED
    assert provenance.SOURCE_COLOCATED not in {
        s.lower() for s in gates.load_necessity_control()["accepted_reconciliation_statuses"]}

    # no shared span and a single assertion -> unreconciled, whatever the model claimed
    other = EvidenceSpan("a different passage", anchored=True, text_sha256="h2",
                         span_id="other-1")
    prf2 = ClinicalFact(FactKind.PROCEDURE, "proc", fact_id="pf", evidence=[other])
    forged2 = RelationAssertion("df", RelationPredicate.REASON_FOR, "pf",
                                state=RelationState.ASSERTED, confidence=0.99,
                                evidence_span_ids=["shared-1", "other-1"],
                                reconciliation_status="externally_verified")
    out2 = provenance.validate_relations([forged2], [dxf, prf2], note)
    assert out2[0].reconciliation_status == provenance.UNRECONCILED


# ---- Codex F6-R3 (round 4): distinct assertion ORIGINS, directional proof, governed dual ---
# The reviewer's three independent reproductions at 917e031, plus the cross-run and
# conflicting-origin cases. All identifiers are synthetic; no medical code or term appears.

_R4_PROFILE = {"provider": "provider-one", "model": "profile-one",
               "callable": "tests.stub"}


def _r4_response(note_quotes, relations):
    """One extraction response: a diagnosis fact, a service fact, and whatever relations the
    caller wants repeated. `note_quotes` maps fact -> its verbatim evidence list."""
    axes_dx = {a: 0.9 for a in ("occurrence", "action", "evidence", "temporal",
                                "assertion", "experiencer")}
    axes_pr = {a: 0.9 for a in ("occurrence", "action", "evidence", "temporal",
                                "performer", "relationship")}
    return json.dumps({
        "facts": [
            {"fact_id": "D", "kind": "diagnosis", "description": "condition alpha",
             "attributes": {}, "disposition": "performed_today", "certainty": "confirmed",
             "experiencer": "patient", "evidence": note_quotes["D"], "confidence": 0.95,
             "axis_confidence": axes_dx},
            {"fact_id": "S", "kind": "procedure", "description": "service beta",
             "attributes": {}, "disposition": "performed_today", "certainty": "confirmed",
             "experiencer": "patient", "evidence": note_quotes["S"], "confidence": 0.95,
             "axis_confidence": axes_pr},
        ],
        "relations": relations,
    })


_R4_EDGE = {"subject_event_id": "D", "object_event_id": "S", "predicate": "reason_for",
            "state": "asserted", "evidence_fact_ids": ["D", "S"], "confidence": 0.99}

# A note whose ONE clause states the direction: the service, a linking phrase, the reason.
_R4_NOTE = "Service beta was performed for condition alpha. Nothing further."
_R4_QUOTES = {"D": ["condition alpha"], "S": ["Service beta"]}


def _r4_graph(note, response, *, extra_responses=()):
    """Anchor + bind + validate exactly as the pipeline does, over one or more responses."""
    from claude_coder import provenance
    facts, relations = [], []
    for i, raw in enumerate((response,) + tuple(extra_responses)):
        out = extraction.extract_note(note, lambda _s, _u, _r=raw: _r,
                                      run_id=f"pass-{i + 1}" if extra_responses else None,
                                      model_profile=_R4_PROFILE)
        if not facts:                       # the events are the FIRST pass's; later passes
            facts = out.facts               # only contribute their assertion of the edge
            provenance.anchor_facts(note, facts)
        relations.extend(provenance.bind_relation_evidence(out.relations, facts))
    return facts, provenance.validate_relations(relations, facts, note)


def test_duplicate_edge_in_one_response_is_not_independent_corroboration():
    """Reproduction 1: the SAME edge emitted twice in ONE response from ONE model. Raw support
    rises to 2, but both assertions share one recorded origin, so independent support stays 1
    and nothing is corroborated."""
    from claude_coder import provenance
    facts, rels = _r4_graph(_R4_NOTE, _r4_response(_R4_QUOTES, [_R4_EDGE, dict(_R4_EDGE)]))
    (edge,) = rels
    assert edge.support == 2                        # it WAS asserted twice ...
    assert edge.independent_support == 1            # ... by exactly one origin
    assert edge.corroboration_status == provenance.SINGLE_ORIGIN


def test_cross_run_same_provider_is_recorded_as_multiply_asserted_only():
    """Two separate runs of the same provider are two recorded origins, so the identical edge
    is recorded as multiply asserted -- the origin count targets repetition, not re-running.
    That count is an observation about the MODEL: it lands on the corroboration axis, never on
    the grounding axis, which here is decided (independently) by the note's own wording."""
    from claude_coder import provenance
    one = _r4_response(_R4_QUOTES, [_R4_EDGE])
    facts, rels = _r4_graph(_R4_NOTE, one, extra_responses=(one,))
    (edge,) = rels
    assert edge.independent_support == 2
    assert edge.corroboration_status == provenance.MULTIPLY_ASSERTED
    assert edge.corroboration_status not in provenance.RECONCILIATION_STATUSES
    assert edge.reconciliation_status == provenance.SOURCE_DIRECTIONAL   # from the NOTE


def test_conflicting_origins_collapse_and_cannot_support_necessity():
    """Two origins that DISAGREE about the same edge collapse to UNCERTAIN, and an uncertain
    edge disqualifies the pair outright -- two origins are not a vote."""
    from claude_coder.models import CodingResult, Outcome, RelationState
    from claude_coder.gates import medical_necessity_gate
    negated = dict(_R4_EDGE, state="negated")
    facts, rels = _r4_graph(_R4_NOTE, _r4_response(_R4_QUOTES, [_R4_EDGE]),
                            extra_responses=(_r4_response(_R4_QUOTES, [negated]),))
    (edge,) = rels
    assert edge.independent_support == 2 and edge.state is RelationState.UNCERTAIN
    from dataclasses import replace
    proc, dx = _nec_lines()                       # re-point the edge at the claim's events
    r = CodingResult("e", "2026-08-01", lines=[proc, dx],
                     relations=[replace(edge, subject_event_id="df", object_event_id="pf")])
    assert medical_necessity_gate(r).outcome is Outcome.UNKNOWN


def test_shared_span_without_directional_wording_is_only_an_observation():
    """Reproduction 2: one shared exact span naming a condition AND a service performed for an
    unrelated reason. Both facts are genuinely co-located, but the passage states no
    directional claim, so the edge is recorded as co-located and cannot certify necessity."""
    from claude_coder import provenance
    from claude_coder.models import CodingResult, Outcome
    from claude_coder.gates import medical_necessity_gate
    note = "Condition alpha is present, service beta was performed for reason delta."
    sentence = note[:-1]
    facts, rels = _r4_graph(note, _r4_response(
        {"D": ["Condition alpha", sentence], "S": ["service beta", sentence]}, [_R4_EDGE]))
    (edge,) = rels
    assert edge.reconciliation_status == provenance.SOURCE_COLOCATED
    from dataclasses import replace
    proc, dx = _nec_lines()                       # re-point the edge at the claim's events
    r = CodingResult("e", "2026-08-01", lines=[proc, dx],
                     relations=[replace(edge, subject_event_id="df", object_event_id="pf")])
    assert medical_necessity_gate(r).outcome is Outcome.UNKNOWN
    assert not r.necessity_support


def test_directional_clause_reconciles_and_records_the_proving_spans():
    """The positive case: each endpoint has its own verified mention and the source text
    between them links them in the declared orientation, so the DOCUMENT establishes the
    predicate -- and the spans that proved it are recorded for the certificate."""
    from claude_coder import provenance
    facts, rels = _r4_graph(_R4_NOTE, _r4_response(_R4_QUOTES, [_R4_EDGE]))
    (edge,) = rels
    assert edge.reconciliation_status == provenance.SOURCE_DIRECTIONAL
    assert len(edge.reconciliation_evidence) == 2
    assert set(edge.reconciliation_evidence) <= set(edge.evidence_span_ids)


def test_directional_proof_needs_the_endpoints_own_mentions():
    """Quoting ONE identical long passage for both endpoints localises neither, so the same
    documented sentence proves nothing about direction."""
    from claude_coder import provenance
    sentence = _R4_NOTE.split(".")[0]
    facts, rels = _r4_graph(_R4_NOTE, _r4_response(
        {"D": [sentence], "S": [sentence]}, [_R4_EDGE]))
    assert rels[0].reconciliation_status == provenance.SOURCE_COLOCATED


def test_directional_proof_is_voided_by_negation_and_by_a_clause_break():
    """The linking text must stay inside one clause and carry no negation marker."""
    from claude_coder import provenance
    for note in ("Service beta was not performed for condition alpha. End.",
                 "Service beta was performed. Later, for condition alpha, review. End."):
        facts, rels = _r4_graph(note, _r4_response(
            {"D": ["condition alpha"], "S": ["Service beta"]}, [_R4_EDGE]))
        assert rels[0].reconciliation_status != provenance.SOURCE_DIRECTIONAL, note


def test_governed_service_with_no_relation_at_all_holds():
    """Reproduction 3: a claim whose released diagnosis merely HAPPENS to sit in the
    procedure's coverage set, with no REASON_FOR relation anywhere. Coverage membership proves
    the pair CAN qualify; it never proves this diagnosis justified this service here."""
    from claude_coder.models import CodingResult, Outcome
    from claude_coder.gates import medical_necessity_gate
    proc, dx = _nec_lines()
    governed = MockSource(); governed._coverage = {"P1": {"D1"}}
    r = CodingResult("e", "2026-08-01", lines=[proc, dx])          # no relations at all
    out = medical_necessity_gate(r, governed)
    assert out.outcome is Outcome.UNKNOWN
    assert not r.necessity_support
    # with the encounter-specific link present as well, the same claim releases
    linked = CodingResult("e", "2026-08-01", lines=[proc, dx], relations=[_reason(0.99)])
    assert medical_necessity_gate(linked, governed).outcome is Outcome.PASS


def test_governed_service_needs_the_LINKED_diagnosis_to_be_the_qualifying_one():
    """A reconciled link to diagnosis A plus coverage that qualifies only diagnosis B is not
    two satisfied requirements -- they must meet on the SAME diagnosis."""
    from claude_coder.models import (CandidateCode, ClinicalFact, CodingResult, EvidenceSpan,
                                     FactKind, Outcome, ResolutionMethod, ResolvedLine)
    from claude_coder.gates import medical_necessity_gate
    proc, dx = _nec_lines()                       # linked dx event 'df' -> code D1
    other = ResolvedLine(
        fact=ClinicalFact(kind=FactKind.DIAGNOSIS, description="x", attributes={},
                          fact_id="df2", evidence=[EvidenceSpan("x")]),
        chosen=CandidateCode("D2", "icd10", "d", 0.9), method=ResolutionMethod.DETERMINISTIC)
    governed = MockSource(); governed._coverage = {"P1": {"D2"}}   # only the UNLINKED dx
    r = CodingResult("e", "2026-08-01", lines=[proc, dx, other], relations=[_reason(0.99)])
    assert medical_necessity_gate(r, governed).outcome is Outcome.UNKNOWN


def test_necessity_binding_is_recorded_and_bound_into_the_certificate():
    """What justified the service must be answerable FROM the certificate: the claim-line
    diagnosis pointer and the accepted relation's provenance."""
    from claude_coder import certificate
    from claude_coder.models import CodingResult, Outcome
    from claude_coder.gates import medical_necessity_gate
    proc, dx = _nec_lines()
    edge = _reason(0.99)
    edge.assertion_origins = ["origin-a", "origin-b"]
    edge.reconciliation_evidence = ["span-1"]
    r = CodingResult("e", "2026-08-01", lines=[proc, dx], relations=[edge])
    assert medical_necessity_gate(r).outcome is Outcome.PASS
    (binding,) = r.necessity_support
    (support,) = binding["supports"]
    assert binding["procedure_event_id"] == "pf" and binding["procedure_code"] == "P1"
    assert support["diagnosis_event_id"] == "df" and support["diagnosis_code"] == "D1"
    assert support["relation_id"] == edge.relation_id
    assert support["reconciliation_status"] == "source_directional"
    assert support["assertion_origins"] == ["origin-a", "origin-b"]
    assert support["independent_support"] == 2
    cert = certificate.build_certificate(r, "note text")
    assert cert["necessity_support"] == r.necessity_support
    # the binding is part of the tamper-evident hash
    r2 = CodingResult("e", "2026-08-01", lines=[proc, dx], relations=[edge])
    medical_necessity_gate(r2)
    r2.necessity_support[0]["supports"][0]["diagnosis_code"] = "D9"
    assert certificate.build_certificate(r2, "note text")["certificate_sha256"] \
        != cert["certificate_sha256"]


def test_corroboration_threshold_is_read_from_config_not_hardcoded():
    """The threshold that decides MULTIPLY_ASSERTED must be configuration the layer ACTUALLY
    reads: raising it to three must stop two origins from being recorded as agreement. (A
    threshold declared in a file nobody consumes is worse than no threshold — it reads as a
    control.) It moves only the audit axis: the grounding the note supplies is unchanged."""
    from claude_coder import provenance
    one = _r4_response(_R4_QUOTES, [_R4_EDGE])
    grammar = provenance.load_relation_grammar()
    assert grammar["min_independent_assertions"] == 2
    facts, rels = _r4_graph(_R4_NOTE, one, extra_responses=(one,))
    assert rels[0].corroboration_status == provenance.MULTIPLY_ASSERTED
    saved = grammar["min_independent_assertions"]
    try:
        grammar["min_independent_assertions"] = 3
        (edge,) = provenance.reconcile_relations(
            provenance.merge_relations([rels[0]]), facts, _R4_NOTE)
        assert edge.independent_support == 2
        assert edge.corroboration_status == provenance.SINGLE_ORIGIN
        assert edge.reconciliation_status == provenance.SOURCE_DIRECTIONAL
    finally:
        grammar["min_independent_assertions"] = saved


# ---- Codex F6-R3 (round 5): agreement between extraction runs is not source grounding -----
# The reviewer's reproduction at 6217503: two disjoint note phrases -- one diagnosis mention,
# one performed-service mention -- in a note that states NO causal/necessity relationship
# between them, with the SAME model-authored REASON_FOR assertion supplied under two run
# origins. Round 4 answered {'status': 'corroborated', 'reconciliation_evidence': [],
# 'gate': 'PASS', 'bindings': 1}: necessity certified on an inference the source never makes,
# naming zero source evidence. All identifiers are synthetic; no medical code or term appears.

# Two documented events and NOTHING linking them. Separate sentences, so no candidate linking
# text can stay inside one clause, and separate mentions, so nothing is even co-located.
_R5_NOTE = "Condition alpha noted on review. Service beta was performed. End of note."
_R5_QUOTES = {"D": ["Condition alpha"], "S": ["Service beta"]}
_R5_OTHER_PROFILE = {"provider": "provider-two", "model": "profile-two",
                     "callable": "tests.stub.other"}


def _r5_graph(profiles):
    """The identical edge asserted once per profile, each in its OWN run -- N distinct origins
    over a note that states no relationship."""
    from claude_coder import provenance
    response = _r4_response(_R5_QUOTES, [_R4_EDGE])
    facts, relations = [], []
    for i, profile in enumerate(profiles):
        out = extraction.extract_note(_R5_NOTE, lambda _s, _u, _r=response: _r,
                                      run_id=f"run-{i + 1}", model_profile=profile)
        if not facts:                      # the events are the first pass's; later passes
            facts = out.facts              # only contribute their assertion of the edge
            provenance.anchor_facts(_R5_NOTE, facts)
        relations.extend(provenance.bind_relation_evidence(out.relations, facts))
    return facts, provenance.validate_relations(relations, facts, _R5_NOTE)


def _r5_necessity(edge, source=None):
    """Run the REAL gate over `edge`, re-pointed at a claim's released events."""
    from dataclasses import replace
    from claude_coder.models import CodingResult
    from claude_coder.gates import medical_necessity_gate
    proc, dx = _nec_lines()
    r = CodingResult("e", "2026-08-01", lines=[proc, dx],
                     relations=[replace(edge, subject_event_id="df", object_event_id="pf")])
    return medical_necessity_gate(r, source), r


def test_two_same_provider_runs_agreeing_on_an_ungrounded_relation_hold():
    """The reviewer's exact reproduction. Two runs of one provider are two origins and the edge
    is honestly recorded as multiply asserted -- but the note states no relationship, so the
    grounding axis stays UNRECONCILED with no evidence, and the encounter HOLDS."""
    from claude_coder import provenance
    from claude_coder.models import Outcome
    facts, rels = _r5_graph([_R4_PROFILE, _R4_PROFILE])
    (edge,) = rels
    assert edge.independent_support == 2
    assert edge.corroboration_status == provenance.MULTIPLY_ASSERTED
    assert edge.reconciliation_status == provenance.UNRECONCILED
    assert edge.reconciliation_evidence == []
    out, r = _r5_necessity(edge)
    assert out.outcome is Outcome.UNKNOWN
    assert not r.necessity_support


def test_two_cross_provider_runs_agreeing_on_an_ungrounded_relation_still_hold():
    """Cross-provider agreement is not a magic exception: two vendors' models agreeing is still
    model inference, not something the record says, so it grounds nothing either."""
    from claude_coder import provenance
    from claude_coder.models import Outcome
    facts, rels = _r5_graph([_R4_PROFILE, _R5_OTHER_PROFILE])
    (edge,) = rels
    assert edge.independent_support == 2
    assert {o for o in edge.assertion_origins} != set()   # two genuinely different origins
    assert edge.corroboration_status == provenance.MULTIPLY_ASSERTED
    assert edge.reconciliation_status == provenance.UNRECONCILED
    out, r = _r5_necessity(edge)
    assert out.outcome is Outcome.UNKNOWN
    assert not r.necessity_support


def test_coverage_policy_does_not_rescue_an_ungrounded_agreed_relation():
    """Nor does an authoritative coverage policy that happens to qualify the linked diagnosis:
    the policy check is an ADDITIONAL requirement in both directions, never a substitute for
    grounding the linkage in this note."""
    from claude_coder.models import Outcome
    facts, rels = _r5_graph([_R4_PROFILE, _R5_OTHER_PROFILE])
    governed = MockSource(); governed._coverage = {"P1": {"D1"}}
    out, r = _r5_necessity(rels[0], source=governed)
    assert out.outcome is Outcome.UNKNOWN
    assert not r.necessity_support


def test_source_grounded_relation_still_releases_and_names_its_evidence():
    """The positive case is unchanged: when the note itself states the direction, the edge is
    grounded, the spans that state it are named, and the claim releases -- with agreement
    recorded beside the grounding rather than in place of it."""
    from claude_coder import provenance
    from claude_coder.models import Outcome
    one = _r4_response(_R4_QUOTES, [_R4_EDGE])
    facts, rels = _r4_graph(_R4_NOTE, one, extra_responses=(one,))
    (edge,) = rels
    assert edge.reconciliation_status == provenance.SOURCE_DIRECTIONAL
    assert set(edge.reconciliation_evidence) <= set(edge.evidence_span_ids)
    assert edge.reconciliation_evidence
    out, r = _r5_necessity(edge)
    assert out.outcome is Outcome.PASS
    (support,) = r.necessity_support[0]["supports"]
    assert support["reconciliation_status"] == provenance.SOURCE_DIRECTIONAL
    assert support["corroboration_status"] == provenance.MULTIPLY_ASSERTED   # recorded only
    assert support["reconciliation_evidence"]


def test_grounding_and_corroboration_are_disjoint_vocabularies():
    """The distinction is legible in the data model, not only enforced at the gate: the two
    axes are different fields with non-overlapping value sets, and the model's declared default
    tracks the provenance layer's."""
    from claude_coder import provenance
    from claude_coder.models import RelationAssertion, RelationPredicate
    assert not (provenance.RECONCILIATION_STATUSES & provenance.CORROBORATION_STATUSES)
    assert provenance.GROUNDED_RECONCILIATION_STATUSES < provenance.RECONCILIATION_STATUSES
    assert not (provenance.GROUNDED_RECONCILIATION_STATUSES
                & provenance.RETIRED_RECONCILIATION_STATUSES)
    fresh = RelationAssertion("a", RelationPredicate.REASON_FOR, "b")
    assert fresh.corroboration_status == provenance.SINGLE_ORIGIN
    assert fresh.reconciliation_status == provenance.UNRECONCILED


def test_relation_grammar_is_versioned_configuration_and_fails_closed():
    """The directional grammar is reviewed config that cites its authority; an unreadable or
    incomplete grammar stops reconciliation instead of silently reverting to co-location."""
    import pytest
    from claude_coder import provenance
    from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind, RelationAssertion,
                                     RelationPredicate)
    cfg = provenance.load_relation_grammar()
    assert cfg["control_mode"] == "ENFORCED_FAIL_CLOSED" and cfg["authority"]
    assert cfg["predicates"]["reason_for"]["object_first_cues"]
    from app.release import source_manifest as sm
    assert provenance._RELATION_GRAMMAR_ID in sm.required_release_sources()
    saved_cache = provenance._RELATION_GRAMMAR_CACHE
    saved_registry = dict(sm._AUTHORITATIVE)
    saved_path = sm.declared_source_path(provenance._RELATION_GRAMMAR_ID)
    span = EvidenceSpan("q", anchored=True, span_id="s1")
    dxf = ClinicalFact(FactKind.DIAGNOSIS, "dx", fact_id="df", evidence=[span])
    prf = ClinicalFact(FactKind.PROCEDURE, "pr", fact_id="pf", evidence=[span])
    rel = RelationAssertion("df", RelationPredicate.REASON_FOR, "pf",
                            confidence=0.9, evidence_span_ids=["s1"])
    try:
        provenance._RELATION_GRAMMAR_CACHE = None
        sm._AUTHORITATIVE[provenance._RELATION_GRAMMAR_ID] = (saved_path.parent
                                                              / "does-not-exist.json")
        with pytest.raises(provenance.RelationGrammarError):
            provenance.validate_relations([rel], [dxf, prf], "q")
    finally:
        provenance._RELATION_GRAMMAR_CACHE = saved_cache
        sm._AUTHORITATIVE.clear()
        sm._AUTHORITATIVE.update(saved_registry)
