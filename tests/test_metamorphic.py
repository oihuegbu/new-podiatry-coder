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
from claude_coder.autonomy import (decide, _line_confidence, _ARBITRATED_DISCOUNT,
                                    AUTONOMY_CONFIDENCE, SHAKY_EXTRACTION)
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
        def _s(system, user):
            sl = system.lower()
            if "propose" in sl:
                return '{"codes": []}'
            if "independently" in sl:                       # corroborate
                return '{"entailed": %s, "missing_element": false, "reason": "x"}' % (
                    "true" if entailed else "false")
            return '{"choice": %d, "reason": "x"}' % (1 if entailed else 0)   # select
        return _s
    reject = resolution.resolve(fact(), src, llm=stub(False), corroborate=stub(False))
    assert not reject.resolved                              # crosswalk default not blindly accepted
    accept = resolution.resolve(fact(), src, llm=stub(True), corroborate=stub(True))
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


def test_extraction_skips_malformed_facts():
    """Fail-closed input validation: a fact with an UNRECOGNIZED kind OR an empty
    description is dropped — never emitted as a half-formed fact (kills the
    `kind is None or not desc` Or->And mutant, which would let one half through)."""
    from claude_coder.extraction import extract_facts
    payload = {"facts": [
        {"kind": "not_a_kind", "description": "has description", "evidence": ["x"]},
        {"kind": "diagnosis", "description": "   ", "evidence": ["x"]},
        {"kind": "diagnosis", "description": "onychomycosis", "evidence": ["onychomycosis"]},
    ]}
    facts = extract_facts("note", llm=lambda s, u: json.dumps(payload))
    assert [f.description for f in facts] == ["onychomycosis"]   # only the well-formed one


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
