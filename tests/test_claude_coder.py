"""End-to-end tests for the claude-medical-coder pipeline.

Runs the WHOLE flow (extract -> resolve -> arbitrate -> gate -> autonomy ->
certificate) with a MockSource and stubbed LLMs, so it needs no API key, no RAG
index, and — deliberately — contains NO real medical code (the mock uses
synthetic identifiers). It asserts the safety properties, not just happy paths:
planned work is not billed, negated findings are dropped, unsupported evidence
blocks release, and autonomy is granted only when the chain closes.
"""
import json
import unittest

from claude_coder.data_access import MockSource
from claude_coder.models import CandidateCode, Outcome, ResolutionMethod, Verdict
from claude_coder.pipeline import code_encounter
from tests import shortlist_verdict as _sv


def _from(fn, provider):
    """Declare which model provider a stub LLM stands in for.

    Corroboration only counts as INDEPENDENT when the second judgement comes from a
    declared, DIFFERENT provider, so a test that wants to exercise the corroborated path
    has to say who its two stubs are — the same way the deployment declares it for the
    real callables. (Round 5, phase 5.)"""
    from claude_coder.verify import declare_model_profile
    return declare_model_profile(fn, provider=provider)


def _request(fact):
    from claude_coder.eligibility import (ClaimComponent, ClaimLineIntent,
                                          EligibilityState, RetrievalRequest,
                                          fact_snapshot_digest)
    from claude_coder.models import FactKind
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

# A note whose text contains, verbatim, every evidence span the extractor emits.
# Fully synthetic — the pipeline's disposition/negation logic turns on the
# LINGUISTIC markers ('denies …', 'Plan … next visit'), not any clinical term.
_LINK_SENTENCE = ("Excision of lesion alpha was performed for "
                  "condition alpha of the right side")
NOTE = (
    "Procedure: excision of lesion alpha, right site two. "
    "Assessment: condition alpha, right side. "
    + _LINK_SENTENCE + ". "
    "Patient denies finding gamma. "
    "Plan procedure beta correction next visit."
)

# What the (stubbed) CLU extractor returns: one performed procedure, one current
# diagnosis, one PLANNED procedure (must not bill), one NEGATED finding (drop).
# The procedure and the diagnosis each quote THEIR OWN phrase inside the one sentence that
# states WHY the service was done, so `provenance.reconcile_relations` can establish the
# REASON_FOR edge from the SOURCE — the document's own directional wording between the two
# verified mentions — rather than from the model's self-confidence, from repeating the edge,
# or from the two facts merely appearing near each other. (Codex F6-R3.)
_PROC_MENTION = "Excision of lesion alpha"                 # capitalised only in that sentence
_DX_MENTION = "condition alpha of the right side"


def _facts_json(*, link_evidence=True):
    """The extractor response. `link_evidence=False` drops every quote that lives in the
    linking sentence, for the case where the note never documents WHY the service was done."""
    proc_ev = ["excision of lesion alpha, right site two"]
    dx_ev = ["condition alpha, right side"]
    if link_evidence:
        proc_ev.append(_PROC_MENTION)
        dx_ev.append(_DX_MENTION)
    return json.dumps({"facts": [
        {"kind": "procedure", "description": "excision of lesion alpha",
         "attributes": {"laterality": "right", "anatomy": "site two",
                        "performer_id": "actor-1", "billing_entity_id": "actor-1"},
         "disposition": "performed_today", "negated": False, "evidence": proc_ev,
         "confidence": 0.97,
         "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                             "temporal": 0.99, "performer": 0.99, "relationship": 0.99}},
        {"kind": "diagnosis", "description": "condition alpha of the right side",
         "attributes": {"laterality": "right"}, "disposition": "performed_today",
         "negated": False, "evidence": dx_ev, "confidence": 0.98,
         "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                             "temporal": 0.99, "assertion": 0.99, "experiencer": 0.99}},
        {"kind": "procedure", "description": "procedure beta correction", "attributes": {},
         "disposition": "planned", "negated": False,
         "evidence": ["Plan procedure beta correction next visit"], "confidence": 0.9},
        {"kind": "diagnosis", "description": "finding gamma", "attributes": {},
         "disposition": "performed_today", "negated": True,
         "evidence": ["denies finding gamma"], "confidence": 0.9},
    ], "relations": [
        {"subject_event_id": "F2", "object_event_id": "F1", "predicate": "reason_for",
         "state": "asserted", "evidence_fact_ids": ["F1", "F2"], "confidence": 0.99},
    ]})


FACTS_JSON = _facts_json()

# Synthetic (non-code) identifiers — no real medical code anywhere in this test.
PROC = CandidateCode("PROC_ALPHA_EXC", "cpt",
                     "Excision, lesion alpha, single, each", 0.9, "retrieval")
DX = CandidateCode("DX_ALPHA_RIGHT", "icd10",
                   "condition alpha, right side", 0.9, "retrieval")


def _source():
    return MockSource(
        records={("PROC_ALPHA_EXC", "cpt"): {"active": True},
                 ("DX_ALPHA_RIGHT", "icd10"): {"active": True}},
        retrieval={("*", "cpt"): [PROC], ("*", "icd10"): [DX]},
    )


def _extract_stub(system, user):
    return FACTS_JSON


def _arbitrate_stub(system, user):
    return '{"choice":0,"confidence":0.0,"reason":"unused"}'


class AutonomousCoderTest(unittest.TestCase):

    def _run(self, note=NOTE, dos="2026-03-14"):
        from claude_coder.provenance import NullAuditRepository
        return code_encounter("enc-1", note, dos, source=_source(),
                              extract_llm=_extract_stub, arbitrate_llm=_arbitrate_stub,
                              audit_repository=NullAuditRepository(),
                              billing_context={"billing_entity_id": "actor-1", "participants": [{"id": "actor-1", "type": "person", "roles": ["performer"]}]})

    def test_happy_path_auto_ready(self):
        r = self._run()
        codes = {ln.chosen.code for ln in r.billable_lines}
        self.assertEqual(codes, {"PROC_ALPHA_EXC", "DX_ALPHA_RIGHT"})
        self.assertEqual(r.verdict, Verdict.AUTO_READY, r.notes)

    def test_certificate_binds_necessity_provenance_and_stays_reproducible(self):
        """F6-R3 end-to-end: the released claim's certificate answers WHY the service was
        necessary — the claim-line diagnosis pointer plus the accepted relation's provenance
        (status, the spans that proved it, the distinct assertion origins) — and, composing
        with the phase-1 content-addressed manifest work, identical inputs still reproduce
        the same certificate hash (the origin id is content-derived, not a per-call nonce)."""
        a, b = self._run(), self._run()
        self.assertEqual(a.verdict, Verdict.AUTO_READY, a.notes)
        self.assertEqual(a.certificate["certificate_sha256"],
                         b.certificate["certificate_sha256"])
        (binding,) = a.certificate["necessity_support"]
        (support,) = binding["supports"]
        self.assertEqual(binding["procedure_code"], "PROC_ALPHA_EXC")
        self.assertEqual(support["diagnosis_code"], "DX_ALPHA_RIGHT")
        self.assertEqual(support["reconciliation_status"], "source_directional")
        self.assertEqual(support["independent_support"], 1)   # one pass = one origin
        self.assertTrue(all(support["assertion_origins"]))
        self.assertTrue(set(support["reconciliation_evidence"])
                        <= set(support["evidence_span_ids"]))

    def test_unreconciled_necessity_link_loses_autonomy(self):
        """F6-R3 end-to-end: strip the one sentence that documents the diagnosis and the
        service TOGETHER. The model still asserts a 0.99-confidence REASON_FOR edge and the
        edge still anchors, but nothing independently reconciles it — so the claim loses
        autonomy instead of being certified on the extraction model's own say-so."""
        note = NOTE.replace(_LINK_SENTENCE + ". ", "")
        facts = _facts_json(link_evidence=False)
        from claude_coder.provenance import NullAuditRepository
        r = code_encounter("enc-1", note, "2026-03-14", source=_source(),
                           extract_llm=lambda s, u: facts,
                           arbitrate_llm=_arbitrate_stub,
                           audit_repository=NullAuditRepository(),
                           billing_context={"billing_entity_id": "actor-1", "participants": [
                               {"id": "actor-1", "type": "person", "roles": ["performer"]}]})
        nec = next(g for g in r.gates if g.name == "medical_necessity")
        self.assertEqual(nec.outcome, Outcome.UNKNOWN, nec.detail)
        self.assertNotEqual(r.verdict, Verdict.AUTO_READY)

    def test_agreeing_origins_cannot_certify_an_ungrounded_link_end_to_end(self):
        """F6-R3 round 5 end-to-end: the SAME ungrounded note, with the corroboration axis
        forced to its agreeing value. (Lowering the threshold stands in for a multi-pass
        caller: the deployed pipeline extracts once, so agreement is only ever reachable from
        a caller that runs more passes — which is exactly the hole being closed.) The edge is
        honestly recorded as multiply asserted, the grounding axis still says the record
        proves nothing, and the whole encounter still loses autonomy with no binding."""
        from claude_coder import provenance
        from claude_coder.provenance import NullAuditRepository
        note = NOTE.replace(_LINK_SENTENCE + ". ", "")
        facts = _facts_json(link_evidence=False)
        grammar = provenance.load_relation_grammar()
        saved = grammar["min_independent_assertions"]
        try:
            grammar["min_independent_assertions"] = 1
            r = code_encounter("enc-1", note, "2026-03-14", source=_source(),
                               extract_llm=lambda s, u: facts,
                               arbitrate_llm=_arbitrate_stub,
                               audit_repository=NullAuditRepository(),
                               billing_context={"billing_entity_id": "actor-1", "participants": [
                                   {"id": "actor-1", "type": "person",
                                    "roles": ["performer"]}]})
        finally:
            grammar["min_independent_assertions"] = saved
        from claude_coder.models import RelationPredicate
        reasons = [rel for rel in r.relations
                   if rel.predicate is RelationPredicate.REASON_FOR]
        self.assertTrue(reasons)
        for rel in reasons:
            self.assertEqual(rel.corroboration_status, provenance.MULTIPLY_ASSERTED)
            self.assertEqual(rel.reconciliation_status, provenance.UNRECONCILED)
            self.assertEqual(rel.reconciliation_evidence, [])
        nec = next(g for g in r.gates if g.name == "medical_necessity")
        self.assertEqual(nec.outcome, Outcome.UNKNOWN, nec.detail)
        self.assertFalse(r.necessity_support)
        self.assertEqual(r.verdict, Verdict.REVIEW_REQUIRED)
        # the certificate of the held encounter binds NO necessity support and shows both
        # axes, so the audit record says exactly what was and was not established
        self.assertEqual(r.certificate["necessity_support"], [])
        (cert_rel,) = [rel for rel in r.certificate["relations"]
                       if rel["predicate"] == "reason_for"]
        self.assertEqual(cert_rel["corroboration_status"], provenance.MULTIPLY_ASSERTED)
        self.assertEqual(cert_rel["reconciliation_status"], provenance.UNRECONCILED)
        self.assertEqual(cert_rel["reconciliation_evidence"], [])

    def test_planned_work_not_billed(self):
        r = self._run()
        billed = {ln.chosen.code for ln in r.billable_lines}
        self.assertNotIn("beta", " ".join(billed).lower())
        # the planned procedure produced no billable line at all
        self.assertEqual(len(r.billable_lines), 2)

    def test_negated_finding_dropped(self):
        r = self._run()
        descs = " ".join(ln.fact.description for ln in r.lines).lower()
        self.assertNotIn("finding gamma", descs)

    def test_resolution_is_deterministic(self):
        r = self._run()
        for ln in r.billable_lines:
            self.assertEqual(ln.method, ResolutionMethod.DETERMINISTIC, ln.rationale)

    def test_missing_evidence_blocks_release(self):
        # a note that does NOT contain the procedure's evidence span
        r = self._run(note="Assessment: condition alpha, right side.")
        ev = next(g for g in r.gates if g.name == "verbatim_evidence")
        hold = next(g for g in r.gates if g.name.startswith("eligibility_hold:"))
        self.assertEqual(hold.outcome, Outcome.BLOCKED)
        self.assertEqual(ev.outcome, Outcome.PASS)
        self.assertEqual(r.verdict, Verdict.BLOCKED)

    def test_missing_dos_blocks_release(self):
        r = self._run(dos=None)
        dos = next(g for g in r.gates if g.name == "date_of_service")
        self.assertEqual(dos.outcome, Outcome.BLOCKED)
        self.assertEqual(r.verdict, Verdict.BLOCKED)

    def test_certificate_is_reproducible(self):
        a = self._run().certificate["certificate_sha256"]
        b = self._run().certificate["certificate_sha256"]
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)


class OntologyResolutionTest(unittest.TestCase):
    """The deterministic resolver decides by descriptor STRUCTURE (RAG only
    supplies the pool). Synthetic codes; the ranges/laterality come from the
    descriptors — so this also proves the size-family selection needs no
    hardcoded code table."""

    def test_measurement_range_selects_leaf(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, Disposition, EvidenceSpan,
                                         FactKind, ResolutionMethod)
        from claude_coder.resolution import resolve

        small = CandidateCode("SUP_SMALL", "hcpcs",
                              "Wound dressing, sterile, size 16 sq. in. or less, each", 0.9)
        med = CandidateCode("SUP_MED", "hcpcs",
                            "Wound dressing, sterile, size more than 16 sq. in. but less "
                            "than or equal to 48 sq. in., each", 0.9)
        large = CandidateCode("SUP_LARGE", "hcpcs",
                              "Wound dressing, sterile, size more than 48 sq. in., each", 0.9)
        src = MockSource(retrieval={("*", "hcpcs"): [small, med, large]})
        fact = ClinicalFact(kind=FactKind.SUPPLY, description="wound dressing",
                            attributes={"size_sqin": 30}, disposition=Disposition.PERFORMED,
                            evidence=[EvidenceSpan("wound dressing 30 sq in applied")],
                            confidence=0.99)
        line = resolve(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "SUP_MED", line.rationale)

    def test_laterality_contradiction_eliminated(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve

        left = CandidateCode("DX_LEFT", "icd10", "some condition, left foot", 0.9)
        right = CandidateCode("DX_RIGHT", "icd10", "some condition, right foot", 0.9)
        src = MockSource(retrieval={("*", "icd10"): [left, right]})
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="some condition",
                            attributes={"laterality": "right"},
                            evidence=[EvidenceSpan("some condition, right side")],
                            confidence=0.99)
        line = resolve(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "DX_RIGHT", line.rationale)


class BundlingExclusionTest(unittest.TestCase):
    """A resolved code the source declares NOT separately reportable is dropped
    from the claim (agnostic — driven by the source's separately_billable, not a
    named code) while remaining in the audit trail."""

    def test_non_separately_billable_code_excluded(self):
        from claude_coder.data_access import MockSource
        from claude_coder.pipeline import code_encounter

        cand = CandidateCode("BUNDLED_X", "hcpcs", "bundled add-on service", 0.9)
        src = MockSource(records={("BUNDLED_X", "hcpcs"): {"active": True}},
                         retrieval={("*", "hcpcs"): [cand]},
                         nonbillable={"BUNDLED_X"})
        facts = ('{"facts":[{"kind":"supply","description":"bundled service",'
                 '"attributes":{"performer_id":"actor-1","billing_entity_id":"actor-1"},'
                 '"disposition":"performed_today","negated":false,'
                 '"evidence":["bundled service provided"],"confidence":0.99}]}')
        r = code_encounter("e", "bundled service provided during the visit",
                           "2026-03-14", source=src,
                           extract_llm=lambda s, u: facts,
                           arbitrate_llm=lambda s, u: '{"choice":0,"confidence":0}',
                           audit_repository=__import__("claude_coder.provenance",
                               fromlist=["NullAuditRepository"]).NullAuditRepository())
        self.assertNotIn("BUNDLED_X", {ln.chosen.code for ln in r.billable_lines})
        self.assertTrue(any(ln.excluded_reason for ln in r.lines))


class ModifierTest(unittest.TestCase):
    """Modifiers are discovered from data by descriptor and applied from the
    documented laterality — no modifier literal in the engine."""

    def _fact(self, laterality):
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        return ClinicalFact(kind=FactKind.PROCEDURE, description="excision",
                            attributes={"laterality": laterality},
                            evidence=[EvidenceSpan("x")])

    def test_laterality_and_bilateral_from_data(self):
        from claude_coder.modifiers import ModifierEngine
        defs = {"MR": {"description": "Right side of the body"},
                "ML": {"description": "Left side of the body"},
                "MB": {"description": "Bilateral procedure"}}
        eng = ModifierEngine(defs=defs)
        # A side/bilateral modifier is asserted ONLY for a code the fee schedule marks
        # laterality-eligible (a bilateral-surgery indicator is present); pass a real one.
        self.assertEqual(eng.assign(self._fact("right"), "Excision, lesion, each", bilat="0"), ["MR"])
        self.assertEqual(eng.assign(self._fact("left"), "Excision, lesion, each", bilat="0"), ["ML"])
        self.assertEqual(eng.assign(self._fact("bilateral"), "Excision, lesion", bilat="1"), ["MB"])
        # descriptor already encodes the side -> no modifier (no double-coding)
        self.assertEqual(eng.assign(self._fact("right"), "Excision, lesion, right side", bilat="0"), [])
        # a code with NO bilateral indicator -- a consumed supply/implant/drug billed as
        # a device/supply code (not on the PFS) -- never earns a side modifier. This is
        # the agnostic rule behind the A4570-RT / C1713-RT error class: laterality
        # requires positive fee-schedule eligibility, never a default.
        self.assertEqual(eng.assign(self._fact("right"), "Anchor/screw for bone", bilat=None), [])


class EMLevelingTest(unittest.TestCase):
    def test_mdm_two_of_three(self):
        from claude_coder.em import mdm_level
        self.assertEqual(mdm_level("moderate", "moderate", "low"), "moderate")
        self.assertEqual(mdm_level("high", "low", "low"), "low")
        self.assertEqual(mdm_level("high", "high", "low"), "high")
        self.assertIsNone(mdm_level("moderate", "", "low"))   # incomplete -> review

    def test_resolve_em_picks_level_and_setting(self):
        from claude_coder.data_access import MockSource
        from claude_coder.em import resolve_em
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        low = CandidateCode("EM_LOW", "cpt",
                            "Office visit, established patient, low level medical decision making", 0.9)
        mod = CandidateCode("EM_MOD", "cpt",
                            "Office visit, established patient, moderate level medical decision making", 0.9)
        src = MockSource(retrieval={("*", "cpt"): [low, mod]})
        fact = ClinicalFact(kind=FactKind.EM, description="office visit",
                            attributes={"problems": "moderate", "data": "moderate",
                                        "risk": "low", "new_patient": False},
                            evidence=[EvidenceSpan("office visit")], confidence=0.9)
        line = resolve_em(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "EM_MOD", line.rationale)


class UnitsTest(unittest.TestCase):
    def test_billing_units_from_descriptor(self):
        from claude_coder.ontology import billing_units
        # a "2-4 lesions" code is ONE unit for 2 lesions (the bug this fixes)
        self.assertEqual(billing_units(2, "Paring, 2 to 4 lesions"), 1)
        # an "each" code bills per item
        self.assertEqual(billing_units(3, "Excision of lesion, single, each"), 3)
        self.assertEqual(billing_units(1, "Some procedure"), 1)


class ClaimModifierTest(unittest.TestCase):
    def test_em25_and_distinct_service_from_data(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, CodingResult, EvidenceSpan,
                                         FactKind, ResolutionMethod, ResolvedLine)
        from claude_coder.modifiers import ModifierEngine
        defs = {"M25": {"description": "significant, separately identifiable evaluation and management"},
                "MXS": {"description": "Separate Structure"}}
        eng = ModifierEngine(defs=defs)

        def line(code, kind, lat=None):
            f = ClinicalFact(kind=kind, description="x",
                             attributes=({"laterality": lat} if lat else {}),
                             evidence=[EvidenceSpan("x")])
            return ResolvedLine(fact=f, chosen=CandidateCode(code, "cpt", "d", 0.9),
                                method=ResolutionMethod.DETERMINISTIC)
        p1 = line("P1", FactKind.PROCEDURE, "right")
        p2 = line("P2", FactKind.PROCEDURE, "left")
        emln = line("EMX", FactKind.EM)
        emln.fact.attributes["separately_identifiable"] = True   # 25 only when documented
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[p1, p2, emln])
        src = MockSource(ncci={("P1", "P2"): "1", ("P2", "P1"): "1"})
        eng.assign_claim(r, src)
        self.assertIn("M25", emln.modifiers)                       # E/M-25 with a procedure
        self.assertTrue("MXS" in p1.modifiers or "MXS" in p2.modifiers)  # distinct structure
        self.assertTrue(r.bypassed_ncci)                          # bypass recorded for the gate


class GlobalPackageTest(unittest.TestCase):
    """A same-day E/M is bundled into a procedure's CMS global package unless the
    note documents separately-identifiable E/M work."""

    def _lines(self, sep_ident):
        from claude_coder.models import (ClinicalFact, CodingResult, EvidenceSpan,
                                         FactKind, ResolutionMethod, ResolvedLine)

        def line(code, kind, attrs):
            f = ClinicalFact(kind=kind, description="x", attributes=attrs,
                             evidence=[EvidenceSpan("x")])
            return ResolvedLine(fact=f, chosen=CandidateCode(code, "cpt", "d", 0.9),
                                method=ResolutionMethod.DETERMINISTIC)
        proc = line("PROCG", FactKind.PROCEDURE, {})
        em = line("EMV", FactKind.EM, {"separately_identifiable": sep_ident})
        return CodingResult(encounter_id="e", date_of_service="2026-03-14",
                            lines=[proc, em]), em

    def test_em_bundled_when_not_separately_identifiable(self):
        from claude_coder.data_access import MockSource
        from claude_coder.pipeline import apply_global_package
        r, em = self._lines(sep_ident=False)
        apply_global_package(r, MockSource(gp={"PROCG": "090"}))
        self.assertTrue(em.excluded_reason)
        self.assertNotIn("EMV", {ln.chosen.code for ln in r.billable_lines})

    def test_em_kept_when_separately_identifiable(self):
        from claude_coder.data_access import MockSource
        from claude_coder.pipeline import apply_global_package
        r, em = self._lines(sep_ident=True)
        apply_global_package(r, MockSource(gp={"PROCG": "090"}))
        self.assertIsNone(em.excluded_reason)
        self.assertIn("EMV", {ln.chosen.code for ln in r.billable_lines})


class BilateralEligibilityTest(unittest.TestCase):
    """Modifier 50 / laterality is gated by the CMS bilateral indicator, so a
    per-nail code (indicator 9) gets no laterality modifier."""

    def _fact(self, lat):
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        return ClinicalFact(kind=FactKind.PROCEDURE, description="x",
                            attributes={"laterality": lat}, evidence=[EvidenceSpan("x")])

    def test_indicator_gates_modifier(self):
        from claude_coder.modifiers import ModifierEngine
        defs = {"MR": {"description": "Right side of the body"},
                "ML": {"description": "Left side of the body"},
                "MB": {"description": "Bilateral procedure"}}
        eng = ModifierEngine(defs=defs)
        # 9 = concept does not apply -> no modifier, even bilateral
        self.assertEqual(eng.assign(self._fact("bilateral"), "Debridement of nails", bilat="9"), [])
        self.assertEqual(eng.assign(self._fact("right"), "Debridement of nails", bilat="9"), [])
        # 1 = bilateral eligible -> 50; 0 = not eligible -> none
        self.assertEqual(eng.assign(self._fact("bilateral"), "Paired procedure", bilat="1"), ["MB"])
        self.assertEqual(eng.assign(self._fact("bilateral"), "Some procedure", bilat="0"), [])


class EMSettingTest(unittest.TestCase):
    def test_setting_filter_rejects_wrong_setting(self):
        from claude_coder.em import _select
        ed = CandidateCode("EMED", "cpt",
                           "Emergency department visit, moderate level medical decision making", 0.9)
        office = CandidateCode("EMOFF", "cpt",
                               "Office or other outpatient visit, established patient, "
                               "moderate level medical decision making", 0.9)
        chosen = _select("moderate", False, "office", [ed, office])  # ED ranked first
        self.assertEqual(chosen.code, "EMOFF")   # office encounter -> office code, not ED


class TerminologyIndexTest(unittest.TestCase):
    """Deterministic term->code resolution via the authoritative Alphabetic Index.
    Synthetic codes/terms — the mechanics under test are exact match, order/plural-
    independent token-set match (the Index's inverted phrasing), synonym->code, and
    the code-dotting form — none of which depend on any specific condition."""

    def test_index_term_to_code(self):
        from claude_coder.terminology import TerminologyIndex
        # AA111 has two synonyms; BB220's term is written in INVERTED order.
        idx = TerminologyIndex({"AA111": ["condition alpha", "alpha synonym"],
                                "BB220": ["gamma, entity beta", "beta variant gamma"]})
        self.assertEqual(idx.candidates("Alpha synonym"), {"AA1.11"})     # exact, dotted
        self.assertEqual(idx.candidates("entity beta gamma"), {"BB2.20"}) # inverted, token-set
        self.assertEqual(idx.candidates("no such term here"), set())      # -> caller falls back

    def test_diagnosis_resolves_via_index_first(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        # Index maps the clinician term; the record supplies the terse descriptor.
        src = MockSource(records={("C11.1", "icd10"):
                                  {"long_description": "a terse authoritative descriptor",
                                   "active": True}},
                         index={"a documented condition": {"C11.1"}})
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="a documented condition",
                            evidence=[EvidenceSpan("a documented condition")], confidence=0.99)
        line = resolve(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "C11.1")
        self.assertIn("Alphabetic Index", line.rationale)

    def test_snomed_layer_resolves_when_index_misses(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        # Index has no entry; the SNOMED map resolves the term authoritatively.
        src = MockSource(records={("C22.2", "icd10"):
                                  {"long_description": "a terse authoritative descriptor",
                                   "active": True}},
                         index={}, snomed={"a documented condition": {"C22.2"}})
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="a documented condition",
                            evidence=[EvidenceSpan("a documented condition")], confidence=0.95)
        line = resolve(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "C22.2")
        self.assertIn("SNOMED", line.rationale)

    def test_category_expands_to_leaf_by_laterality(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        # Index returns the category DX4; documented laterality selects the leaf.
        # Synthetic codes/descriptors — the mechanic is category->leaf-by-laterality,
        # not any specific condition.
        recs = {("DX40", "icd10"): {"long_description": "some condition, unspecified site", "active": True},
                ("DX41", "icd10"): {"long_description": "some condition, right site", "active": True},
                ("DX42", "icd10"): {"long_description": "some condition, left site", "active": True}}
        src = MockSource(records=recs, index={"a documented condition": {"DX4"}})
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="a documented condition",
                            attributes={"laterality": "right"},
                            evidence=[EvidenceSpan("a documented condition")], confidence=0.99)
        line = resolve(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "DX41")     # right-side leaf, not the category


class ConceptRelationIndexTest(unittest.TestCase):
    """SAME / ancestor-descendant-or-ambiguous-overlap / unresolved for two clinical
    terms, from a governed concept graph (issue #6 F7-R3-C/C3) -- never from lexical
    shape, and never a fabricated DISJOINT (an IS_A subsumption hierarchy has no basis
    to assert two concepts are opposed). Synthetic concept ids/terms — the mechanics
    under test are candidate matching (exact, despaced, token-set), ancestor-closure
    BFS, unique-vs-ambiguous resolution, and the relation verdict, none of which depend
    on any specific SNOMED concept."""

    def _index(self):
        from claude_coder.terminology import ConceptRelationIndex
        # C1 (root) -> C2 -> C3, an unrelated C9, and an AMBIGUOUS term ("shared name")
        # naming both C2 and C9 -- two different real-world concepts with one shared
        # lexical name, exactly the kind of term the licensed release audit found
        # (65 exact terms, 543 token-set keys resolve to more than one concept).
        return ConceptRelationIndex({
            "C1": {"terms": ["root structure"], "parents": []},
            "C2": {"terms": ["mid structure", "middle structure", "shared name"],
                  "parents": ["C1"]},
            "C3": {"terms": ["leaf structure", "the leaf"], "parents": ["C2"]},
            "C9": {"terms": ["other structure", "shared name"], "parents": []},
        })

    def test_two_synonyms_of_the_same_concept_are_same(self):
        idx = self._index()
        self.assertEqual(idx.relation("leaf structure", "the leaf"), "same")

    def test_an_ancestor_and_descendant_are_related_not_same(self):
        idx = self._index()
        self.assertEqual(idx.relation("leaf structure", "mid structure"),
                         "ancestor_descendant")
        self.assertEqual(idx.relation("mid structure", "root structure"),
                         "ancestor_descendant")

    def test_two_concepts_with_no_relation_are_unresolved_not_disjoint(self):
        """Codex F7-R3-C3, exact-SHA re-review: SNOMED's IS_A hierarchy carries no
        disjointness axiom (per its own OWL/NNF specification), and the licensed
        release audit found tens of thousands of non-ancestor sibling pairs that still
        share a descendant -- absent subsumption is not evidence of opposition. Two
        structurally unrelated concepts must resolve UNRESOLVED, never a fabricated
        DISJOINT that could wrongly add a billed occurrence."""
        idx = self._index()
        self.assertEqual(idx.relation("leaf structure", "other structure"),
                         "unresolved")

    def test_an_ambiguous_shared_candidate_is_related_not_a_confirmed_same(self):
        """Codex F7-R3-C3: a term resolving to MORE THAN ONE concept is itself
        ambiguous within the graph -- the candidate sets merely intersecting is not
        proof the two terms name the same real-world structure, only that they MIGHT.
        'shared name' names both C2 and C9; matched against a term unique to C2, the
        verdict must be the weaker RELATED, never a confirmed SAME."""
        idx = self._index()
        self.assertEqual(idx.relation("shared name", "middle structure"),
                         "ancestor_descendant")   # RELATED verdict string
        self.assertNotEqual(idx.relation("shared name", "middle structure"), "same")

    def test_relation_detail_carries_the_auditable_match_basis(self):
        from claude_coder.terminology import CONCEPT_SAME
        idx = self._index()
        detail = idx.relation_detail("leaf structure", "the leaf")
        self.assertEqual(detail.verdict, CONCEPT_SAME)
        self.assertEqual(detail.match_a.candidates, ("C3",))
        self.assertEqual(detail.match_a.method, "exact")
        self.assertTrue(detail.match_a.unique)
        self.assertEqual(detail.alternatives_a, ())
        self.assertEqual(detail.confidence, 1.0)

        ambiguous = idx.relation_detail("shared name", "no such term")
        self.assertEqual(ambiguous.match_a.candidates, ("C2", "C9"))
        self.assertFalse(ambiguous.match_a.unique)
        self.assertEqual(ambiguous.alternatives_a, ("C2", "C9"))

    def test_an_unknown_term_is_unresolved_not_a_guessed_relation(self):
        idx = self._index()
        self.assertEqual(idx.relation("leaf structure", "no such term"), "unresolved")
        self.assertEqual(idx.relation("no such term", "leaf structure"), "unresolved")


def _line(code, kind, descriptor="d", attrs=None, system="cpt"):
    from claude_coder.models import (ClinicalFact, EvidenceSpan, ResolutionMethod,
                                     ResolvedLine)
    f = ClinicalFact(kind=kind, description="x", attributes=(attrs or {}),
                     evidence=[EvidenceSpan("x")])
    return ResolvedLine(fact=f, chosen=CandidateCode(code, system, descriptor, 0.9),
                        method=ResolutionMethod.DETERMINISTIC)


class SectionApplicabilityTest(unittest.TestCase):
    """Mechanic 1 — an anesthesia-section code (detected from descriptor grammar,
    not a code range) is bundled into the operating provider's claim unless a
    separate anesthesia provider is documented."""

    def _result(self, anes_attrs=None):
        from claude_coder.models import CodingResult, FactKind
        surgery = _line("SURG_X", FactKind.PROCEDURE, "Ostectomy, complete excision")
        anes = _line("ANES_X", FactKind.PROCEDURE,
                     "Anesthesia for procedures on nerves of a structure", anes_attrs or {})
        return CodingResult(encounter_id="e", date_of_service="2026-03-14",
                            lines=[surgery, anes]), anes

    def test_anesthesia_excluded_on_operative_claim(self):
        from claude_coder.pipeline import apply_section_applicability
        r, anes = self._result()
        apply_section_applicability(r)
        self.assertTrue(anes.excluded_reason)
        self.assertNotIn("ANES_X", {ln.chosen.code for ln in r.billable_lines})

    def test_anesthesia_kept_when_separate_provider_documented(self):
        from claude_coder.pipeline import apply_section_applicability
        r, anes = self._result(anes_attrs={"anesthesia_provider": True})
        apply_section_applicability(r)
        self.assertIsNone(anes.excluded_reason)

    def test_escalated_anesthesia_excluded_deterministically(self):
        # an ESCALATED procedure whose candidates are anesthesia-section is excluded
        # deterministically (not left as a review item), independent of resolution.
        from claude_coder.models import (ClinicalFact, CodingResult, EvidenceSpan,
                                         FactKind, ResolutionMethod, ResolvedLine)
        from claude_coder.pipeline import apply_section_applicability
        surgery = _line("SURG", FactKind.PROCEDURE, "Ostectomy of a structure")
        af = ClinicalFact(kind=FactKind.PROCEDURE, description="regional block for anesthesia",
                          evidence=[EvidenceSpan("regional block for anesthesia")])
        anes = ResolvedLine(fact=af, chosen=None, method=ResolutionMethod.ABSTAINED,
                            alternatives=[CandidateCode("ANESX", "cpt",
                                          "Anesthesia for procedures on a structure", 0.8)])
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[surgery, anes])
        apply_section_applicability(r)
        self.assertTrue(anes.excluded_reason)
        self.assertIn("anesthesia-section", anes.excluded_reason)

    def test_surgical_line_with_incidental_anesthesia_candidate_not_excluded(self):
        # a SURGICAL procedure whose LEADING candidate is surgical must NOT be
        # excluded as anesthesia just because an anesthesia neighbour is in the pool.
        from claude_coder.models import (ClinicalFact, CodingResult, EvidenceSpan,
                                         FactKind, ResolutionMethod, ResolvedLine)
        from claude_coder.pipeline import apply_section_applicability
        surgery = _line("SURG", FactKind.PROCEDURE, "Ostectomy of a structure")
        sf = ClinicalFact(kind=FactKind.PROCEDURE, description="tendon debridement",
                          evidence=[EvidenceSpan("tendon debridement")])
        esc = ResolvedLine(fact=sf, chosen=None, method=ResolutionMethod.ABSTAINED,
                           alternatives=[CandidateCode("SURGC", "cpt", "Tenolysis of a tendon", 0.8),
                                         CandidateCode("ANESX", "cpt", "Anesthesia for procedures on a structure", 0.5)])
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[surgery, esc])
        apply_section_applicability(r)
        self.assertIsNone(esc.excluded_reason)      # leading candidate is surgical
        self.assertFalse(esc.resolved)

    def test_section_detection_is_descriptor_driven(self):
        # section is read from the descriptor's leading grammar, not any code/term.
        from claude_coder.ontology import code_section
        self.assertEqual(code_section("Anesthesia for procedures on the structure"), "anesthesia")
        self.assertIsNone(code_section("Some surgical service on a structure"))


class NcciBundlingTest(unittest.TestCase):
    """Mechanic 3 — an unmodified PTP pair DEMOTES the component (keeps the
    payable code) instead of blocking; '(separate procedure)' codes bundle."""

    def test_component_demoted_not_blocked(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import apply_ncci_bundling
        payable = _line("COMPREH", FactKind.PROCEDURE, "comprehensive procedure")
        component = _line("COMPON", FactKind.PROCEDURE, "component procedure")
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[payable, component])
        # directional edit: COMPREH is column-1 payable, COMPON is column-2, no bypass
        src = MockSource(ncci={("COMPREH", "COMPON"): "0"})
        apply_ncci_bundling(r, src)
        self.assertIsNone(payable.excluded_reason)
        self.assertTrue(component.excluded_reason)
        self.assertNotIn("COMPON", {ln.chosen.code for ln in r.billable_lines})

    def test_separate_procedure_designation_bundled(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import apply_ncci_bundling
        main = _line("MAINP", FactKind.PROCEDURE, "definitive surgical procedure")
        sep = _line("SEPP", FactKind.IMAGING, "Fluoroscopy (separate procedure), 1 hour")
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[main, sep])
        apply_ncci_bundling(r, MockSource())
        self.assertTrue(sep.excluded_reason)
        self.assertIsNone(main.excluded_reason)

    def test_separate_procedure_not_bundled_by_supply_only(self):
        # a '(separate procedure)' code must NOT bundle just because a supply/device
        # is also reported — only another actual PROCEDURE triggers the bundle.
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import apply_ncci_bundling
        sep = _line("SEPP", FactKind.PROCEDURE, "some service (separate procedure)")
        device = _line("DEVX", FactKind.SUPPLY, "implant device", system="hcpcs")
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[sep, device])
        apply_ncci_bundling(r, MockSource())
        self.assertIsNone(sep.excluded_reason)      # a device is not "another procedure"

    def test_bypassed_pair_keeps_both(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import apply_ncci_bundling
        a = _line("PA", FactKind.PROCEDURE, "procedure A")
        b = _line("PB", FactKind.PROCEDURE, "procedure B")
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[a, b])
        r.bypassed_ncci = [frozenset(("PA", "PB"))]     # a distinct-service modifier applied
        src = MockSource(ncci={("PA", "PB"): "1"})       # bypassable edit
        apply_ncci_bundling(r, src)
        self.assertIsNone(a.excluded_reason)
        self.assertIsNone(b.excluded_reason)


class DedupTest(unittest.TestCase):
    """Mechanic 4 — two facts resolving to the same code become one billable line."""

    def test_duplicate_code_collapsed(self):
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import dedup_lines
        a = _line("SAME", FactKind.PROCEDURE, "same procedure")
        b = _line("SAME", FactKind.PROCEDURE, "same procedure")
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[a, b])
        dedup_lines(r)
        billed = [ln for ln in r.billable_lines if ln.chosen.code == "SAME"]
        self.assertEqual(len(billed), 1)
        self.assertTrue(a.excluded_reason or b.excluded_reason)

    def test_a_governed_concept_source_resolves_a_true_synonym_pair_to_one_unit(self):
        """Issue #6 F7-R3-C: without a concept source, two mentions worded as genuine
        synonyms on the anatomy axis hold as UNDETERMINED (see
        AxisComparisonIsNormalized in test_evidence_graph.py) because lexical shape
        alone cannot tell a synonym pair from a real distinction. With a governed
        concept source available and confirming the SAME concept, that ambiguity
        resolves: one documented service, described twice, is one billed unit."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import dedup_lines
        from claude_coder.terminology import CONCEPT_SAME
        a = _line("SAME", FactKind.PROCEDURE, "same procedure",
                 attrs={"anatomy": "great toe"})
        b = _line("SAME", FactKind.PROCEDURE, "same procedure",
                 attrs={"anatomy": "hallux"})
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[a, b])
        src = MockSource(concept_relation={("great toe", "hallux"): CONCEPT_SAME})
        dedup_lines(r, src)
        billed = [ln for ln in r.billable_lines if ln.chosen.code == "SAME"]
        self.assertEqual(len(billed), 1)
        self.assertEqual(billed[0].units, 1)

    def test_a_reported_disjoint_relation_never_confirms_a_distinct_occurrence(self):
        """Codex F7-R3-C3, exact-SHA re-review: SNOMED's IS_A hierarchy has no basis to
        assert two concepts are opposed, so `ConceptRelationIndex` never returns
        CONCEPT_DISJOINT -- and `coreference.axis_relation` does not promote on it
        even if a source reports it anyway (defense in depth against a
        non-conforming or future `CodeSource` implementation). A pair a source claims
        is DISJOINT must still HOLD, exactly like an unresolved pair -- never silently
        confirmed as two occurrences from a relation this system does not trust."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import dedup_lines
        from claude_coder.terminology import CONCEPT_DISJOINT
        a = _line("SAME", FactKind.PROCEDURE, "same procedure",
                 attrs={"anatomy": "second toe"})
        b = _line("SAME", FactKind.PROCEDURE, "same procedure",
                 attrs={"anatomy": "great toe"})
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[a, b])
        src = MockSource(
            concept_relation={("second toe", "great toe"): CONCEPT_DISJOINT})
        dedup_lines(r, src)
        billed = [ln for ln in r.billable_lines if ln.chosen.code == "SAME"]
        self.assertEqual(billed, [],
                         "a claimed DISJOINT relation must hold, never confirm an "
                         "occurrence -- this system has no authoritative basis for it")

    def test_an_unresolved_concept_source_falls_back_to_the_existing_hold(self):
        """A concept source that cannot resolve either term (issue #6 F7-R3-C) must
        degrade to the same conservative hold as having no source at all -- never a
        wrong merge or a wrong split."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import dedup_lines
        a = _line("SAME", FactKind.PROCEDURE, "same procedure",
                 attrs={"anatomy": "great toe"})
        b = _line("SAME", FactKind.PROCEDURE, "same procedure",
                 attrs={"anatomy": "hallux"})
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[a, b])
        src = MockSource()  # no concept_relation mapping configured -> unresolved
        dedup_lines(r, src)
        billed = [ln for ln in r.billable_lines if ln.chosen.code == "SAME"]
        self.assertEqual(billed, [],
                         "an unresolved concept relation must hold, not silently bill")


class ProcedureIndexTest(unittest.TestCase):
    """Mechanic 5 — a procedure phrase resolves through the CPT/HCPCS descriptor
    index (deterministic) before embedding, the procedure-axis analog of the ICD
    Alphabetic Index."""

    def test_procedure_resolves_via_descriptor_index(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        src = MockSource(records={("PROC_OST", "cpt"):
                                  {"long_description": "some service on a named structure",
                                   "active": True}},
                         proc_index={"some service on a named structure": {"PROC_OST"}})
        fact = ClinicalFact(kind=FactKind.PROCEDURE,
                            description="some service on a named structure",
                            evidence=[EvidenceSpan("some service on a named structure")],
                            confidence=0.99)
        line = resolve(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "PROC_OST")
        self.assertIn("descriptor index", line.rationale)


class SupportRankingTest(unittest.TestCase):
    """Descriptor↔fact token support ORDERS candidates and NEVER eliminates one.

    It also never SELECTS one. Product directive section 4 allows lexical/semantic
    similarity to widen a candidate pool and forbids it from verifying a code, so the
    near-tie below is settled by what the ORIGINAL DOCUMENT was proven to say — not by
    which descriptor happens to share more words with the note.
    """

    _MATCH = CandidateCode("P_MATCH", "cpt", "excision of bursa of the foot", 0.80)
    _NEIGH = CandidateCode("P_NEIGH", "cpt", "open treatment of fracture", 0.80)

    def _source(self):
        from claude_coder.data_access import MockSource
        return MockSource(
            records={("P_MATCH", "cpt"): {"active": True},
                     ("P_NEIGH", "cpt"): {"active": True}},
            # neighbour listed first: retrieval ORDER must not decide either
            retrieval={("*", "cpt"): [self._NEIGH, self._MATCH]})

    def _fact(self, *, anchored):
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        quote = "the bursa was excised"
        return ClinicalFact(
            kind=FactKind.PROCEDURE, description="excision of bursa", confidence=0.9,
            evidence=[EvidenceSpan(quote, start=0, end=len(quote), anchored=anchored,
                                   span_id=("span-0" if anchored else None))])

    def test_a_source_anchored_word_hit_cannot_settle_an_untyped_tie(self):
        """Equal recall, and the neighbour is retrieved first. Even a source-anchored
        quotation cannot promote an untyped token overlap into clinical-role evidence;
        both candidates remain alternatives until a typed axis distinguishes them."""
        from app.contracts.source_evidence import (ReconciliationStatus,
                                                   SourceReconciliation,
                                                   SpanReconciliation)
        from claude_coder.resolution import resolve
        reconciliation = SourceReconciliation(spans=(
            SpanReconciliation(span_id="span-0", status=ReconciliationStatus.AGREED,
                               pages=(1,)),))
        line = resolve(_request(self._fact(anchored=True)), self._source(),
                       reconciliation=reconciliation)
        self.assertIsNone(line.chosen, line.rationale)
        self.assertEqual({c.code for c in line.alternatives}, {"P_MATCH", "P_NEIGH"})
        self.assertEqual(line.tie_record["winner"], "")
        self.assertFalse(line.tie_record["axes"][0]["selectable"])

    def test_token_overlap_alone_cannot_close_the_same_near_tie(self):
        """The identical pool and the identical wording, with the quotation NOT anchored
        to the source. The descriptor/note token overlap is unchanged and still favours
        the matching code — and nothing is billed, because overlap is not evidence."""
        from claude_coder.resolution import resolve
        line = resolve(_request(self._fact(anchored=False)), self._source())
        self.assertIsNone(line.chosen, line.rationale)
        self.assertEqual({c.code for c in line.alternatives}, {"P_MATCH", "P_NEIGH"})

    def test_support_never_eliminates_terse_code(self):
        # a correct but terse/generic descriptor sharing no tokens with the phrasing
        # must still resolve (support is ranking-only, not a floor).
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        terse = CandidateCode("P_TERSE", "cpt",
                              "Complete bilateral noninvasive physiologic studies", 0.82)
        src = MockSource(records={("P_TERSE", "cpt"): {"active": True}},
                         retrieval={("*", "cpt"): [terse]})
        fact = ClinicalFact(kind=FactKind.PROCEDURE,
                            description="ankle brachial index with doppler",
                            evidence=[EvidenceSpan("ABI with Doppler waveforms")],
                            confidence=0.9)
        line = resolve(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "P_TERSE")


class CptAlphabeticIndexTest(unittest.TestCase):
    """The authoritative term->code index layer (AMA CPT Alphabetic Index slot)
    resolves a documented phrase deterministically BEFORE embedding — a plain
    term->code lookup, agnostic to the term."""

    def test_cpt_index_resolves_authoritatively(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        src = MockSource(
            records={("PROC_IDX", "cpt"):
                     {"long_description": "some documented service, unspecified",
                      "active": True}},
            cpt_index={"a documented procedure phrase": {"PROC_IDX"}})
        fact = ClinicalFact(kind=FactKind.PROCEDURE,
                            description="a documented procedure phrase",
                            evidence=[EvidenceSpan("a documented procedure phrase")],
                            confidence=0.95)
        line = resolve(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "PROC_IDX")
        self.assertIn("CPT Alphabetic Index", line.rationale)


class CptIndexParserTest(unittest.TestCase):
    """tools/parse_cpt_index.py: header-driven column detection + range expansion
    to real codes only. Synthetic codes are generated at runtime (no literal code
    cluster), so the parser is exercised without any real medical code."""

    def _mod(self):
        import importlib.util
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "parse_cpt_index", root / "tools" / "parse_cpt_index.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_column_detection(self):
        m = self._mod()
        main_i, mod_i, code_i = m._match_cols(["Main Term", "Modifier", "Code/Range"])
        self.assertEqual((main_i, code_i), (0, 2))
        self.assertEqual(mod_i, [1])

    def test_range_expands_to_valid_only(self):
        m = self._mod()
        base = 90000
        valid = {str(base + i) for i in range(5)}          # generated, not literal
        self.assertEqual(m._expand(f"{base}-{base+3}", valid),
                         [str(base + i) for i in range(4)])
        self.assertEqual(m._expand(f"{base}, {base+2}", valid),
                         [str(base), str(base + 2)])
        self.assertEqual(m._expand("ZZ999", valid), [])    # not a real code -> dropped

    def test_delimiter_sniff(self):
        m = self._mod()
        self.assertEqual(m._sniff("a\tb\tc"), "\t")
        self.assertEqual(m._sniff("a|b|c"), "|")
        self.assertEqual(m._sniff("a,b,c"), ",")


class DrugTableTest(unittest.TestCase):
    """CMS Table of Drugs & Biologicals: a drug NAME resolves to its HCPCS code
    authoritatively, and billing units come from documented dose / per-unit dose."""

    def test_drug_resolves_by_name(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        src = MockSource(
            records={("DRUG_KETO", "hcpcs"):
                     {"long_description": "Injection, substance alpha, per 15 mg",
                      "active": True}},
            drug_index={"substance alpha": {"DRUG_KETO"}})
        fact = ClinicalFact(kind=FactKind.DRUG, description="substance alpha",
                            evidence=[EvidenceSpan("substance alpha 30 mg IV")],
                            confidence=0.95)
        line = resolve(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "DRUG_KETO")
        self.assertIn("Table of Drugs", line.rationale)

    def test_units_from_documented_dose(self):
        from claude_coder.ontology import drug_billing_units
        self.assertEqual(drug_billing_units("substance 30 mg IV", {"amount": 15, "unit": "mg"}), 2)
        self.assertEqual(drug_billing_units("1 g infused", {"amount": 100, "unit": "mg"}), 10)  # g->mg
        self.assertIsNone(drug_billing_units("two tablets", {"amount": 15, "unit": "mg"}))       # no dose
        self.assertIsNone(drug_billing_units("30 ml", {"amount": 15, "unit": "mg"}))             # unit clash


class DrugTableParserTest(unittest.TestCase):
    """tools/build_hcpcs_drug_table.py: a drug code is detected by descriptor
    grammar (substance-amount billing unit), never a code prefix; name + per-unit
    dose parsed out. No real code appears (synthetic descriptors only)."""

    def _mod(self):
        import importlib.util
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "build_hcpcs_drug_table", root / "tools" / "build_hcpcs_drug_table.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_dose_and_name(self):
        m = self._mod()
        self.assertEqual(m._dose_of("Injection, substance alpha, per 15 mg"),
                         (15.0, "mg"))
        self.assertEqual(m._name_of("Injection, substance alpha, per 15 mg"),
                         "substance alpha")

    def test_supply_is_not_a_drug(self):
        m = self._mod()
        # 'each' is not a substance amount -> not a dosed drug
        self.assertIsNone(m._dose_of("Needle-free injection device, each"))


class ProposeVerifyTest(unittest.TestCase):
    """Propose-then-verify: recall is a candidate GENERATOR, the authoritative
    descriptor + entailment is TRUTH. Proposals are validated against the registry;
    a code is accepted only when its official descriptor is entailed by the
    documentation. Uses ABSTRACT near-synonym acts (ALPHA vs BETA) — the mechanism
    is agnostic; it turns on descriptor↔documentation match, not any medical term."""

    # Two descriptors differing only in the ACT primitive (a near-synonym pair).
    ALPHA = "Act alpha of the structure, unspecified approach"
    BETA = "Act beta of the structure, unspecified approach"

    def _src(self):
        from claude_coder.data_access import MockSource
        return MockSource(
            records={("CODEALPHA", "cpt"): {"long_description": self.ALPHA, "active": True},
                     ("CODEBETA", "cpt"): {"long_description": self.BETA, "active": True}},
            # BETA has the HIGHER recall — verify must still reject it for ALPHA.
            retrieval={("*", "cpt"): [CandidateCode("CODEBETA", "cpt", self.BETA, 0.9),
                                      CandidateCode("CODEALPHA", "cpt", self.ALPHA, 0.8)]})

    def _fact(self):
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        return ClinicalFact(kind=FactKind.PROCEDURE,
                            description="act alpha of the structure",
                            evidence=[EvidenceSpan("act alpha performed on the structure")],
                            confidence=0.95)

    def _llm(self, propose=(), entail=True):
        # A shortlist verdict, not a bare pick: the option whose descriptor names the
        # DOCUMENTED act (alpha) is entailed, and the near-synonym (beta) is eliminated
        # WITH a reason -- which is what makes the surviving candidate provably unique.
        return _sv.judge(entails=(lambda d: "alpha" in d.lower() and "beta" not in d.lower()) if entail else (lambda d: False),
                         propose=propose, reason="documented act matches")

    def test_near_synonym_model_pick_does_not_establish_uniqueness(self):
        from claude_coder.resolution import resolve
        line = resolve(_request(self._fact()), self._src(), llm=self._llm())
        self.assertIsNone(line.chosen, line.rationale)
        self.assertEqual({c.code for c in line.alternatives},
                         {"CODEALPHA", "CODEBETA"})

    def test_proposal_widens_recall_but_does_not_select_from_untyped_terms(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import ResolutionMethod
        from claude_coder.resolution import resolve
        # retrieval only surfaces the WRONG code; the model proposes the right one,
        # which is validated against the registry and then verified.
        src = MockSource(
            records={("CODEALPHA", "cpt"): {"long_description": self.ALPHA, "active": True},
                     ("CODEBETA", "cpt"): {"long_description": self.BETA, "active": True}},
            retrieval={("*", "cpt"): [CandidateCode("CODEBETA", "cpt", self.BETA, 0.95)]})
        line = resolve(_request(self._fact()), src,
                       llm=_from(self._llm(propose=["CODEALPHA"]), "provider-a"),
                       corroborate=_from(self._corroborator(confirm=True), "provider-b"))
        self.assertEqual(line.method, ResolutionMethod.ABSTAINED)
        self.assertIsNone(line.chosen)
        self.assertEqual({c.code for c in line.alternatives},
                         {"CODEALPHA", "CODEBETA"})

    def test_escalates_when_nothing_entailed(self):
        from claude_coder.models import ResolutionMethod
        from claude_coder.resolution import resolve
        line = resolve(_request(self._fact()), self._src(), llm=self._llm(entail=False))
        self.assertFalse(line.resolved)
        self.assertEqual(line.method, ResolutionMethod.ABSTAINED)
        self.assertIn("verified", line.rationale)

    def test_fabricated_proposal_dropped(self):
        from claude_coder.verify import propose_codes
        cands = propose_codes(self._fact(), self._src(),
                              self._llm(propose=["NOTREAL", "CODEALPHA"]))
        self.assertEqual([c.code for c in cands], ["CODEALPHA"])   # nonexistent code dropped
        self.assertEqual(cands[0].descriptor, self.ALPHA)          # descriptor from the record

    def _corroborator(self, confirm, missing=False):
        """The INDEPENDENT judge, answering about the WHOLE shortlist. `confirm` means it
        finds the documented act (alpha) entailed and eliminates the near-synonym."""
        return _sv.judge(entails=(lambda d: "alpha" in d.lower() and "beta" not in d.lower()) if confirm else (lambda d: False),
                         missing_element=missing, reason="second opinion")

    def test_corroboration_cannot_make_untyped_terms_selecting(self):
        """Two declared, distinct model origins do not convert a raw descriptor-token
        difference into independent typed evidence."""
        from claude_coder.models import ResolutionMethod
        from claude_coder.resolution import resolve
        line = resolve(_request(self._fact()), self._src(),
                       llm=_from(self._llm(), "provider-a"),
                       corroborate=_from(self._corroborator(confirm=True), "provider-b"))
        self.assertEqual(line.method, ResolutionMethod.ABSTAINED)
        self.assertIsNone(line.chosen)
        self.assertEqual(line.tie_record["still_entailed"],
                         ["CODEBETA", "CODEALPHA"])

    def test_missing_element_escalates_as_provider_query(self):
        # second model says the code fits but the note omits a required element ->
        # escalate as a provider query, do NOT down-code to something that omits it.
        from claude_coder.models import ResolutionMethod
        from claude_coder.resolution import resolve
        line = resolve(_request(self._fact()), self._src(), llm=self._llm(),
                       corroborate=self._corroborator(confirm=False, missing=True))
        self.assertFalse(line.resolved)
        self.assertEqual(line.method, ResolutionMethod.ABSTAINED)
        self.assertIn("PROVIDER QUERY", line.rationale)

    def test_wrong_code_reselects_to_confirmed_alternative(self):
        # second model rejects the first pick as a WRONG code (not a doc gap) -> the
        # loop re-selects among the remaining candidates and accepts the one both
        # models agree on.
        import json
        import re
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        d1, d2 = "act alpha primary form", "act alpha secondary form"
        src = MockSource(
            records={("A1", "cpt"): {"long_description": d1, "active": True},
                     ("A2", "cpt"): {"long_description": d2, "active": True}},
            retrieval={("*", "cpt"): [CandidateCode("A1", "cpt", d1, 0.9),
                                      CandidateCode("A2", "cpt", d2, 0.8)]})

        # picks the first alpha still on the list; both forms are entailed for it
        sel = _sv.judge(entails=lambda d: "alpha" in d.lower(), reason="alpha")
        # the independent judge entails only A2, so A1 is rejected as a WRONG code
        corr = _sv.judge(entails=lambda d: "secondary" in d.lower(), reason="x")

        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="act alpha",
                            evidence=[EvidenceSpan("act alpha performed")], confidence=0.9)
        line = resolve(_request(fact), src, llm=_from(sel, "provider-a"),
                       corroborate=_from(corr, "provider-b"))
        self.assertEqual(line.method, ResolutionMethod.VERIFIED)
        self.assertEqual(line.chosen.code, "A2")     # re-selected past the rejected A1


class CorroborationIndependenceTest(unittest.TestCase):
    """Round 5, phase 5 — agreement between two calls to the SAME model provider is not
    corroboration, so it cannot buy the grounded VERIFIED method or the autonomy that rides
    on it.

    This is the milder, CONJUNCTIVE sibling of the F6-R3 necessity defect: a corroborator
    can only ever subtract (a disagreeing one abstains), never manufacture a code that was
    not already an authoritative-table candidate. What it could wrongly do is CERTIFY —
    turn one vendor's opinion, sampled twice, into the 'independently confirmed' status
    `autonomy` treats as grounded and releases without a human. Synthetic codes throughout;
    the mechanic is about assertion origins, not about any medical term."""

    ALPHA = "Act alpha of the structure, unspecified approach"

    def _src(self):
        from claude_coder.data_access import MockSource
        return MockSource(
            records={("CODEALPHA", "cpt"): {"long_description": self.ALPHA, "active": True}},
            retrieval={("*", "cpt"): [CandidateCode("CODEALPHA", "cpt", self.ALPHA, 0.9)]})

    def _fact(self):
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        return ClinicalFact(kind=FactKind.PROCEDURE,
                            description="act alpha of the structure",
                            evidence=[EvidenceSpan("act alpha performed on the structure")],
                            confidence=0.95)

    def _select(self):
        return _sv.judge(pick=1, reason="documented act matches")

    def _corroborator(self, confirm=True):
        return _sv.judge(entails=lambda d: bool(confirm), reason="second opinion")

    def _resolve(self, primary_provider, second_provider, confirm=True):
        from claude_coder.resolution import resolve
        llm = self._select()
        corr = self._corroborator(confirm)
        if primary_provider:
            llm = _from(llm, primary_provider)
        if second_provider:
            corr = _from(corr, second_provider)
        return resolve(_request(self._fact()), self._src(), llm=llm, corroborate=corr)

    # ---- the defect ---------------------------------------------------------------
    def test_same_provider_agreement_is_not_verified(self):
        """The deployed shape the finding named: a 'corroborator' that is a second profile
        of the SAME vendor. The code is still offered (it is an authoritative candidate the
        documentation entails, and dropping it would under-code), but it is ARBITRATED."""
        line = self._resolve("claude", "claude")
        self.assertTrue(line.resolved)
        self.assertEqual(line.chosen.code, "CODEALPHA")
        self.assertEqual(line.method, ResolutionMethod.ARBITRATED)
        # and it must not CLAIM independence anywhere a human or an auditor would read it
        self.assertNotIn("independently confirmed", line.rationale)
        self.assertIn("same model provider", line.rationale.lower())

    def test_same_provider_agreement_is_recorded_not_erased(self):
        """Suppressing the CREDIT must not suppress the RECORD: the audit trail still says
        a second opinion agreed, and says why that earned nothing."""
        line = self._resolve("claude", "claude")
        self.assertIn("a second opinion agreed", line.rationale)
        self.assertIn("not independently corroborated", line.rationale.lower())

    def test_undeclared_origins_fail_closed(self):
        """Two callables that declare no identity prove nothing about independence, and
        'we cannot tell' must never read as 'confirmed'."""
        line = self._resolve(None, None)
        self.assertTrue(line.resolved)
        self.assertEqual(line.method, ResolutionMethod.ARBITRATED)
        self.assertIn("declare no provider identity", line.rationale)

    def test_same_callable_for_both_roles_is_not_independent(self):
        """Passing ONE callable as both the verifier and the corroborator is the most
        literal form of self-agreement."""
        from claude_coder.resolution import resolve

        both = _from(_sv.judge(entails=lambda d: True, reason="x"), "provider-a")
        line = resolve(_request(self._fact()), self._src(), llm=both, corroborate=both)
        self.assertEqual(line.method, ResolutionMethod.ARBITRATED)

    # ---- the intended path still works --------------------------------------------
    def test_cross_provider_agreement_is_verified(self):
        line = self._resolve("provider-a", "provider-b")
        self.assertTrue(line.resolved)
        self.assertEqual(line.method, ResolutionMethod.VERIFIED)
        self.assertIn("independently confirmed", line.rationale)

    def test_declared_providers_compare_case_and_space_insensitively(self):
        """'Claude' and ' claude ' are one vendor; a formatting difference must not be
        mistaken for an independence difference."""
        line = self._resolve("Claude", "  claude ")
        self.assertEqual(line.method, ResolutionMethod.ARBITRATED)

    # ---- the direction that was already safe stays safe ----------------------------
    def test_disagreement_abstains_whatever_the_providers_are(self):
        """A corroborator can only ever SUBTRACT. Disagreement abstains — and it does so
        identically for a same-provider and a cross-provider second opinion, so nothing in
        this change made the conjunctive direction weaker."""
        for primary, second in (("claude", "claude"), ("provider-a", "provider-b"),
                                (None, None)):
            with self.subTest(primary=primary, second=second):
                line = self._resolve(primary, second, confirm=False)
                self.assertFalse(line.resolved)
                self.assertEqual(line.method, ResolutionMethod.ABSTAINED)
                self.assertIsNone(line.chosen)

    # ---- the consequence the finding is actually about: autonomy -------------------
    def test_same_provider_line_is_discounted_and_always_reviewed(self):
        """The end of the chain. A same-provider 'corroborated' line must lose BOTH things
        VERIFIED buys it: the undiscounted confidence and eligibility for auto-release."""
        from claude_coder.autonomy import _ARBITRATED_DISCOUNT, _line_confidence, decide
        from claude_coder.models import CodingResult, Destination, Verdict

        same = self._resolve("claude", "claude")
        cross = self._resolve("provider-a", "provider-b")
        self.assertAlmostEqual(_line_confidence(cross), cross.fact.confidence)
        self.assertAlmostEqual(_line_confidence(same),
                               same.fact.confidence * _ARBITRATED_DISCOUNT)
        self.assertLess(_line_confidence(same), _line_confidence(cross))

        result = CodingResult(encounter_id="e", date_of_service="2026-03-14")
        result.lines = [same]
        self.assertIs(decide(result), Verdict.REVIEW_REQUIRED)
        self.assertIs(result.destination, Destination.REVIEW)
        self.assertTrue(any(r["destination"] == Destination.REVIEW.value and r["blocking"]
                            for r in result.routing))

    # ---- the same rule on the OTHER path that mints VERIFIED -----------------------
    def test_specificity_upgrade_needs_an_independent_origin_too(self):
        """`refine_diagnosis_specificity` is the second place a VERIFIED line is minted —
        it swaps the resolved code for a more specific one a model selected. Adjacent
        instance of the same bug class, so it obeys the same rule: the sharper code is
        still adopted (billing the unspecified one when the record supports a specific one
        is the error the function exists to prevent), but without an independent origin the
        line is ARBITRATED and gets a coder."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import (CandidateCode, ClinicalFact, EvidenceSpan,
                                         FactKind, ResolvedLine)
        from claude_coder.resolution import refine_diagnosis_specificity
        broad = "Condition alpha, unspecified"
        specific = "Condition alpha of right structure"
        # same 3-character category, so the specific one is a RELATIVE of the broad one
        src = MockSource(
            records={("QQ000", "icd10"): {"long_description": broad, "active": True},
                     ("QQ011", "icd10"): {"long_description": specific, "active": True}})

        sel = _sv.judge(entails=lambda d: "right" in d.lower(),
                        reason="documented side")

        def line():
            fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="condition alpha",
                                attributes={"laterality": "right"},
                                evidence=[EvidenceSpan("condition alpha, right")],
                                confidence=0.95)
            return ResolvedLine(fact=fact,
                                chosen=CandidateCode("QQ000", "icd10", broad, 1.0),
                                method=ResolutionMethod.DETERMINISTIC, rationale="r")

        undot = lambda c: c.replace(".", "")
        same = refine_diagnosis_specificity(
            line(), src, _from(sel, "claude"), _from(self._corroborator(True), "claude"))
        self.assertEqual(undot(same.chosen.code), "QQ011")        # still sharpened
        self.assertEqual(same.method, ResolutionMethod.ARBITRATED)
        self.assertIn("not independently corroborated", same.rationale.lower())

        cross = refine_diagnosis_specificity(
            line(), src, _from(sel, "provider-a"),
            _from(self._corroborator(True), "provider-b"))
        self.assertEqual(undot(cross.chosen.code), "QQ011")
        self.assertEqual(cross.method, ResolutionMethod.VERIFIED)


class CorroborationIndependenceEndToEndTest(unittest.TestCase):
    """The unit rule has to SURVIVE the pipeline. `code_encounter` is where the two
    judgement callables are chosen, where the resolved line is post-processed (bundling,
    the learned index, global package) and where the released verdict is decided — so a
    same-provider 'corroboration' is proven non-grounding by driving the whole flow, not by
    inspecting `resolution` in isolation."""

    def _run(self, primary_provider, second_provider):
        """The suite's OWN auto-releasable encounter (`NOTE` + `_source`), driven through
        propose-then-verify instead of the deterministic path, so the corroborating call's
        ORIGIN is the only variable between the two runs below."""
        seen = []

        class _Capture:
            def append(self, encounter_id, kind, record):
                seen.append((kind, record))
                return "sha256:" + "0" * 64

        sel = _sv.judge(pick=1, reason="documented act")
        corr = _sv.judge(entails=lambda d: True, reason="second opinion")

        result = code_encounter(
            "enc-independence", NOTE, "2026-03-14", source=_source(),
            extract_llm=_extract_stub,
            verify_llm=_from(sel, primary_provider),
            corroborate_llm=_from(corr, second_provider),
            audit_repository=_Capture(),
            billing_context={"billing_entity_id": "actor-1",
                             "participants": [{"id": "actor-1", "type": "person",
                                               "roles": ["performer"]}]})
        profiles = next(rec["model_profiles"] for kind, rec in seen
                        if kind == "eligibility_enforced")
        return result, profiles

    def _arbitrated_routes(self, result):
        from claude_coder.models import Destination
        return [r for r in result.routing
                if r["destination"] == Destination.REVIEW.value
                and "arbitrated" in r["reason"]]

    def test_cross_provider_corroboration_still_releases(self):
        """Control for the test below: with two DECLARED, distinct origins this exact
        encounter resolves VERIFIED and auto-releases, as it always has."""
        from claude_coder import verify
        result, profiles = self._run("provider-a", "provider-b")
        self.assertEqual({ln.chosen.code for ln in result.billable_lines},
                         {"PROC_ALPHA_EXC", "DX_ALPHA_RIGHT"})
        self.assertTrue(all(ln.method is ResolutionMethod.VERIFIED
                            for ln in result.billable_lines))
        self.assertEqual(result.verdict, Verdict.AUTO_READY, result.notes)
        self.assertFalse(self._arbitrated_routes(result))
        self.assertEqual(profiles["corroboration_origin"], verify.DISTINCT_ORIGIN)
        self.assertIs(profiles["independent_providers"], True)
        self.assertEqual(result.certificate["source_identity"]["models"]["corroboration_origin"],
                         verify.DISTINCT_ORIGIN)

    def test_same_provider_corroboration_never_reaches_verified_or_autonomy(self):
        """The same encounter, the same agreeing second opinion — but from the SAME
        provider. The codes are unchanged (a corroborator can only ever subtract), and
        every one of them now needs a coder instead of releasing."""
        from claude_coder import verify
        from claude_coder.models import Destination
        result, profiles = self._run("claude", "claude")
        self.assertEqual({ln.chosen.code for ln in result.billable_lines},
                         {"PROC_ALPHA_EXC", "DX_ALPHA_RIGHT"})
        self.assertTrue(all(ln.method is ResolutionMethod.ARBITRATED
                            for ln in result.billable_lines))
        self.assertIsNot(result.verdict, Verdict.AUTO_READY)
        self.assertIs(result.destination, Destination.REVIEW)
        routes = self._arbitrated_routes(result)
        self.assertEqual(len(routes), len(result.billable_lines))
        self.assertTrue(all(r["blocking"] for r in routes))
        # ... and the durable audit says why, from the run's own recorded identity
        self.assertEqual(profiles["corroboration_origin"], verify.SHARED_ORIGIN)
        self.assertIs(profiles["independent_providers"], False)
        # the certificate carries the same story, so the record cannot claim more than the
        # run earned: no line is certified as a verified entailment, and the recorded model
        # identity states the origins were shared
        cert = result.certificate
        self.assertEqual(cert["verdict"], Verdict.REVIEW_REQUIRED.value)
        self.assertTrue(all(ln["method"] == ResolutionMethod.ARBITRATED.value
                            for ln in cert["lines"]))
        models = cert["source_identity"]["models"]
        self.assertEqual(models["corroboration_origin"], verify.SHARED_ORIGIN)
        self.assertIs(models["independent_providers"], False)

    def test_learned_index_is_not_fed_by_non_independent_agreement(self):
        """The learned verified-resolution index promotes a phrase->code mapping toward
        DETERMINISTIC trust, and the pipeline feeds it only from VERIFIED lines. It must
        therefore not be fed by an agreement that was not independent — otherwise the
        defect would launder itself into determinism a few encounters later."""
        import claude_coder.learned as learned
        observed = []
        real = learned.observe
        try:
            learned.observe = lambda *a, **k: observed.append(a)
            self._run("claude", "claude")
            self.assertEqual(observed, [])
            self._run("provider-a", "provider-b")
            self.assertTrue(observed)
        finally:
            learned.observe = real


class ModelProfileIdentityTest(unittest.TestCase):
    """The recorded model identity must describe the RUN, not this function's assumptions —
    it is what an auditor reads to check the independence claim."""

    def test_default_callables_are_declared_cross_provider(self):
        from claude_coder import verify
        self.assertEqual(verify.model_profile_of(verify.default_verify_llm)["provider"],
                         verify.VERIFY_PROVIDER)
        self.assertEqual(verify.model_profile_of(verify.default_corroborate_llm)["provider"],
                         verify.CORROBORATE_PROVIDER)
        self.assertEqual(
            verify.corroboration_origin(verify.default_verify_llm,
                                        verify.default_corroborate_llm),
            verify.DISTINCT_ORIGIN)

    def test_identity_reports_the_actual_callables_not_a_fixed_pair(self):
        from claude_coder import verify
        from claude_coder.pipeline import _model_profile_identity
        same = _model_profile_identity(None, _from(lambda s, u: "", "claude"),
                                       _from(lambda s, u: "", "claude"))
        self.assertIs(same["independent_providers"], False)
        self.assertEqual(same["corroboration_origin"], verify.SHARED_ORIGIN)

        cross = _model_profile_identity(None, _from(lambda s, u: "", "provider-a"),
                                        _from(lambda s, u: "", "provider-b"))
        self.assertIs(cross["independent_providers"], True)
        self.assertEqual(cross["corroboration_origin"], verify.DISTINCT_ORIGIN)

    def test_absent_corroborator_is_not_independent(self):
        from claude_coder import verify
        from claude_coder.pipeline import _model_profile_identity
        p = _model_profile_identity(None, _from(lambda s, u: "", "provider-a"), None)
        self.assertIs(p["independent_providers"], False)
        self.assertEqual(p["corroboration_origin"], verify.NO_CORROBORATION)

    def test_every_pairing_maps_to_a_known_origin_and_only_one_releases(self):
        """The status vocabulary is closed and exactly one member is creditable, so a new
        value can never be added without a reviewer deciding which side of the line it is
        on."""
        from claude_coder import verify
        undeclared = lambda s, u: ""
        pairs = [(None, None), (undeclared, None), (undeclared, undeclared),
                 (_from(lambda s, u: "", "x"), _from(lambda s, u: "", "x")),
                 (_from(lambda s, u: "", "x"), _from(lambda s, u: "", "y")),
                 (_from(lambda s, u: "", "x"), undeclared)]
        creditable = set()
        for primary, second in pairs:
            origin = verify.corroboration_origin(primary, second)
            self.assertIn(origin, verify.CORROBORATION_ORIGINS)
            if origin in verify.INDEPENDENT_CORROBORATION_ORIGINS:
                creditable.add(origin)
        self.assertEqual(creditable, {verify.DISTINCT_ORIGIN})
        self.assertEqual(verify.INDEPENDENT_CORROBORATION_ORIGINS,
                         frozenset({verify.DISTINCT_ORIGIN}))

    def test_extraction_provider_overlap_is_recorded_observationally(self):
        """Recorded so the weaker correlation is visible in the artifact, but it is NOT a
        control input — a shared extraction provider alone still leaves the corroboration
        independent."""
        from claude_coder import verify
        from claude_coder.pipeline import _model_profile_identity
        p = _model_profile_identity(None, _from(lambda s, u: "", "provider-a"),
                                    _from(lambda s, u: "", "provider-b"))
        self.assertIn("corroborator_shares_extraction_provider", p)
        self.assertEqual(p["corroboration_origin"], verify.DISTINCT_ORIGIN)
        self.assertIs(p["independent_providers"], True)

    # ---- issue #6 F7-R5: the identity must describe the call that is MADE -----------
    def test_a_supplied_extractor_is_recorded_as_itself_not_as_the_configuration(self):
        """A caller-supplied extractor used to be stamped with the configured provider,
        which made every comparison against it -- including the two-reading independence
        fact -- a statement about configuration rather than about the run."""
        from claude_coder.pipeline import _model_profile_identity
        p = _model_profile_identity(_from(lambda s, u: "", "provider-a"), None, None,
                                    _from(lambda s, u: "", "provider-b"))
        self.assertEqual(p["extraction"]["provider"], "provider-a")
        self.assertEqual(p["second_extraction"]["provider"], "provider-b")

    def test_an_undeclared_supplied_extractor_claims_no_provider(self):
        """Fail-closed: "we cannot tell who read the note" is not "the configured
        vendor read the note"."""
        from claude_coder.pipeline import _model_profile_identity
        p = _model_profile_identity(lambda s, u: "", None, None)
        self.assertEqual(p["extraction"].get("provider", ""), "")

    def test_the_pipelines_own_second_reading_is_still_identified_by_configuration(self):
        """The control case: when the PIPELINE makes the call, configuration is exactly
        what selects the model, so it remains the identity."""
        from app.core import config
        from claude_coder.pipeline import _model_profile_identity
        p = _model_profile_identity(None, None, None)
        self.assertEqual(p["extraction"]["provider"], config.LLM_PROVIDER)


class SecondReadingIndependenceTest(unittest.TestCase):
    """The two-reading control fails closed when it is not, in fact, independent.

    `independent_providers` used to be computed after both readings had been paid for
    and then only recorded -- so a deployment whose two readings resolved to ONE vendor
    produced an artifact asserting an independence the run never had (issue #6 F7-R5).
    """

    def _profiles(self, primary, second):
        return {"extraction": {"provider": primary},
                "second_extraction": {"provider": second}}

    def test_a_same_vendor_pair_is_refused_before_the_reading_is_taken(self):
        from claude_coder.extraction import SecondReadingUnavailable
        from claude_coder.pipeline import _run_graph_consensus
        calls = []
        with self.assertRaises(SecondReadingUnavailable):
            _run_graph_consensus("note", [], None,
                                 lambda system, user: calls.append(1) or "{}",
                                 self._profiles("claude", "claude"),
                                 None, None, None, enforce_independence=True)
        self.assertEqual(calls, [],
                         "a control that cannot be independent must not be paid for")

    def test_an_undeclared_pair_is_refused_too(self):
        from claude_coder.extraction import SecondReadingUnavailable
        from claude_coder.pipeline import _run_graph_consensus
        with self.assertRaises(SecondReadingUnavailable):
            _run_graph_consensus("note", [], None, lambda system, user: "{}",
                                 self._profiles("claude", ""),
                                 None, None, None, enforce_independence=True)

    def test_a_caller_supplied_second_reading_is_recorded_not_enforced(self):
        """A second extractor a CALLER supplied is a disagreement detector, whose value
        does not depend on vendor independence. It is recorded, never refused."""
        from claude_coder.pipeline import _run_graph_consensus

        class _Reached(Exception):
            pass

        def _extract(system, user):
            raise _Reached

        with self.assertRaises(_Reached):
            _run_graph_consensus("note", [], None, _extract,
                                 self._profiles("claude", "claude"),
                                 None, None, None, enforce_independence=False)


class LearnedIndexTest(unittest.TestCase):
    """The learned verified-resolution index promotes a phrase->code mapping to
    deterministic trust only when confirmed across >= PROMOTE_AT DISTINCT encounters
    and unambiguous — the automated gate (no human sign-off). Synthetic codes."""

    def _obs(self, phrase, code, enc):
        return {"phrase": phrase, "code": code, "system": "cpt",
                "descriptor": "d", "evidence": ["e"], "enc": enc}

    def test_promote_on_distinct_encounters(self):
        from claude_coder import learned
        obs = [self._obs("phrase one", "PROC_A", f"n{i}") for i in range(3)]
        entries = learned.promote(obs, promote_at=3)
        self.assertIn("phrase one", entries)
        self.assertEqual(entries["phrase one"]["code"], "PROC_A")
        self.assertEqual(entries["phrase one"]["encounters"], 3)

    def test_no_promote_under_threshold(self):
        from claude_coder import learned
        obs = [self._obs("phrase two", "PROC_A", f"n{i}") for i in range(2)]
        self.assertEqual(learned.promote(obs, promote_at=3), {})

    def test_dedup_by_encounter(self):
        from claude_coder import learned
        # same encounter observed 3x is ONE vote — cannot self-promote
        obs = [self._obs("phrase three", "PROC_A", "same") for _ in range(3)]
        self.assertEqual(learned.promote(obs, promote_at=3), {})

    def test_no_promote_when_contested(self):
        from claude_coder import learned
        # PROC_A in 3 encounters, PROC_B in 2 -> 3 < 2*2, ambiguous -> no promotion
        obs = ([self._obs("phrase four", "PROC_A", f"a{i}") for i in range(3)]
               + [self._obs("phrase four", "PROC_B", f"b{i}") for i in range(2)])
        self.assertEqual(learned.promote(obs, promote_at=3), {})

    def test_promote_when_dominant(self):
        from claude_coder import learned
        # PROC_A in 4 encounters, PROC_B in 1 -> 4 >= 2*1 -> promote PROC_A
        obs = ([self._obs("phrase five", "PROC_A", f"a{i}") for i in range(4)]
               + [self._obs("phrase five", "PROC_B", "b0")])
        entries = learned.promote(obs, promote_at=3)
        self.assertEqual(entries.get("phrase five", {}).get("code"), "PROC_A")

    def test_load_observations_roundtrip(self):
        import json
        import tempfile
        from pathlib import Path
        from claude_coder import learned
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "obs.jsonl"
            p.write_text("\n".join(json.dumps(self._obs("p", "PROC_A", f"n{i}"))
                                   for i in range(3)) + "\n")
            self.assertEqual(len(learned.load_observations(p)), 3)

    def test_entry_self_invalidates_on_descriptor_change(self):
        # entry_current is pure normalized-string equality — no domain knowledge.
        # Abstract inputs prove the generic property: same descriptor stays valid,
        # ANY change invalidates, absent text is trusted.
        from claude_coder import learned
        e = {"descriptor": "alpha beta gamma"}
        self.assertTrue(learned.entry_current(e, "Alpha  Beta,  gamma"))   # same up to normalization
        self.assertFalse(learned.entry_current(e, "alpha beta delta"))     # any change -> invalid
        self.assertTrue(learned.entry_current(e, ""))                      # current unknown -> trust
        self.assertTrue(learned.entry_current({"descriptor": ""}, "x"))    # nothing stored -> trust

    def test_learned_index_is_recall_only_not_deterministic(self):
        # Fix5: a learned phrase->code mapping is a RECALL candidate, never a privileged
        # deterministic bill (its key lacks clinical context and its freshness check
        # fails open). With an LLM verifier that does NOT confirm entailment, the learned
        # hit must ESCALATE — proving it lost deterministic trust.
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        src = MockSource(
            records={("PROC_X", "cpt"):
                     {"long_description": "some documented service, unspecified",
                      "active": True}},
            learned_index={"a documented service phrase": "PROC_X"})
        fact = ClinicalFact(kind=FactKind.PROCEDURE,
                            description="a documented service phrase",
                            evidence=[EvidenceSpan("a documented service phrase")],
                            confidence=0.95)

        reject = _sv.judge(entails=lambda d: False, reason="none entailed")

        line = resolve(_request(fact), src, llm=reject, corroborate=reject)
        self.assertFalse(line.resolved)                       # not billed on learned trust
        self.assertEqual(line.method, ResolutionMethod.ABSTAINED)
        self.assertNotIn("learned verified-resolution index", line.rationale)


class RecommendationsTest(unittest.TestCase):
    """Documentation recommendations are derived agnostically from fact kinds,
    resolution methods, and gate outcomes — no code/term/scenario. Abstract inputs."""

    def _line(self, resolved, doc_gap=None, rationale="r", confidence=0.99):
        # `confidence` is explicit because it is claim-affecting here: a resolved line
        # the note BARELY documents is not a clean line, and it now earns a
        # `documentation_clarity` recommendation (the suggested action behind the
        # PROVIDER_QUERY `autonomy.decide` routes it to). The default is a clearly
        # well-documented fact so "resolved" means resolved.
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod, ResolvedLine)
        f = ClinicalFact(kind=FactKind.PROCEDURE, description="a documented service",
                         evidence=[EvidenceSpan("a documented service performed")],
                         confidence=confidence)
        chosen = CandidateCode("PROC_X", "cpt", "d", 0.9) if resolved else None
        return ResolvedLine(fact=f, chosen=chosen,
                            method=(ResolutionMethod.VERIFIED if resolved
                                    else ResolutionMethod.ABSTAINED),
                            documentation_gap=doc_gap, rationale=rationale)

    def test_documentation_gap_becomes_query(self):
        from claude_coder.models import CodingResult
        from claude_coder.recommendations import build_recommendations
        ln = self._line(resolved=False, doc_gap="a required element was not stated")
        recs = build_recommendations(
            CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[ln]))
        self.assertEqual([r["issue"] for r in recs], ["documentation_gap"])
        self.assertIn("a required element was not stated", recs[0]["recommendation"])

    def test_unresolved_service_recommendation(self):
        from claude_coder.models import CodingResult
        from claude_coder.recommendations import build_recommendations
        ln = self._line(resolved=False)                  # abstained, no doc gap
        recs = build_recommendations(
            CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[ln]))
        self.assertEqual([r["issue"] for r in recs], ["unresolved_service"])
        self.assertIn("clarify", recs[0]["recommendation"].lower())

    def test_gate_block_becomes_remediation(self):
        from claude_coder.models import CodingResult, GateResult, Outcome
        from claude_coder.recommendations import build_recommendations
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         gates=[GateResult("verbatim_evidence", Outcome.BLOCKED, "x", "y")])
        recs = build_recommendations(r)
        self.assertEqual([x["issue"] for x in recs], ["gate_verbatim_evidence"])

    def test_resolved_line_yields_no_recommendation(self):
        from claude_coder.models import CodingResult
        from claude_coder.recommendations import build_recommendations
        recs = build_recommendations(
            CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[self._line(resolved=True)]))
        self.assertEqual(recs, [])

    def test_a_barely_documented_resolved_line_still_gets_a_suggested_action(self):
        """The companion of the case above, and the gap this phase's post-fix review
        found: `autonomy.decide` steps back from a resolved line the note barely
        documents, but BOTH existing recommendation rules required an UNRESOLVED line,
        so the routed item carried no suggested action at all."""
        from claude_coder.autonomy import SHAKY_EXTRACTION
        from claude_coder.models import CodingResult
        from claude_coder.recommendations import build_recommendations
        ln = self._line(resolved=True, confidence=SHAKY_EXTRACTION - 0.01)
        ln.fact.axis_confidence = {"laterality": SHAKY_EXTRACTION - 0.01}
        recs = build_recommendations(
            CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[ln]))
        self.assertEqual([r["issue"] for r in recs], ["documentation_clarity"])
        self.assertIn("laterality", recs[0]["recommendation"])


class IntegralBundlingTest(unittest.TestCase):
    """An escalated ancillary that is an NCCI always-bundled (indicator 0) component
    of a billed primary is decided as INTEGRAL (bundled), not escalated. A bypassable
    (indicator 1) pair stays a genuine judgement → escalated. Synthetic codes."""

    def _anc(self, cand_code):
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod, ResolvedLine)
        f = ClinicalFact(kind=FactKind.PROCEDURE, description="ancillary",
                         evidence=[EvidenceSpan("ancillary performed")])
        return ResolvedLine(fact=f, chosen=None, method=ResolutionMethod.ABSTAINED,
                            alternatives=[CandidateCode(cand_code, "cpt", "d", 0.7)])

    def _result(self, indicator):
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        primary = _line("PRIMARY", FactKind.PROCEDURE, "primary procedure")
        anc = self._anc("COMPONENT")
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[primary, anc])
        return r, anc, MockSource(ncci={("PRIMARY", "COMPONENT"): indicator})

    def test_integral_component_bundled(self):
        from claude_coder.pipeline import apply_integral_bundling
        r, anc, src = self._result("0")           # always-bundled
        apply_integral_bundling(r, src)
        self.assertEqual(anc.chosen.code, "COMPONENT")
        self.assertIn("integral", anc.excluded_reason)

    def test_bypassable_component_stays_escalated(self):
        from claude_coder.pipeline import apply_integral_bundling
        r, anc, src = self._result("1")           # separately billable with a modifier
        apply_integral_bundling(r, src)
        self.assertFalse(anc.resolved)
        self.assertIsNone(anc.excluded_reason)


class LateralityUpgradeTest(unittest.TestCase):
    """An unspecified-laterality diagnosis is upgraded to the documented-side sibling
    when the authoritative family has one (validated by descriptor, not a code)."""

    def test_unspecified_upgraded_to_documented_side(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder.resolution import resolve, upgrade_diagnosis_laterality
        recs = {("DX9", "icd10"): {"long_description": "some condition, unspecified site", "active": True},
                ("DX1", "icd10"): {"long_description": "some condition, right site", "active": True},
                ("DX2", "icd10"): {"long_description": "some condition, left site", "active": True}}
        src = MockSource(records=recs, retrieval={("*", "icd10"):
                         [CandidateCode("DX9", "icd10", "some condition, unspecified site", 1.0)]})
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="some condition",
                            attributes={"laterality": "right"},
                            evidence=[EvidenceSpan("some condition, right side")], confidence=0.98)
        line = resolve(_request(fact), src)
        self.assertEqual(line.chosen.code, "DX9")           # retrieval gives unspecified
        line = upgrade_diagnosis_laterality(line, src)
        self.assertEqual(line.chosen.code, "DX1")           # upgraded to the right sibling

    def test_no_upgrade_without_documented_side(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod, ResolvedLine)
        from claude_coder.resolution import upgrade_diagnosis_laterality
        recs = {("DX9", "icd10"): {"long_description": "some condition, unspecified site", "active": True},
                ("DX1", "icd10"): {"long_description": "some condition, right site", "active": True}}
        src = MockSource(records=recs)
        f = ClinicalFact(kind=FactKind.DIAGNOSIS, description="some condition",
                         evidence=[EvidenceSpan("x")])      # no laterality documented
        ln = ResolvedLine(fact=f, chosen=CandidateCode("DX9", "icd10", "some condition, unspecified site", 1.0),
                          method=ResolutionMethod.DETERMINISTIC)
        self.assertEqual(upgrade_diagnosis_laterality(ln, src).chosen.code, "DX9")   # unchanged


class DiagnosisModifierTest(unittest.TestCase):
    """An ICD-10 diagnosis encodes laterality IN the code and must never receive an
    RT/LT procedure modifier — even when the fact documents a side and the chosen
    code's descriptor is unspecified."""

    def test_icd10_diagnosis_gets_no_laterality_modifier(self):
        from claude_coder.data_access import MockSource
        from claude_coder.modifiers import ModifierEngine
        from claude_coder.pipeline import code_encounter
        dx = CandidateCode("DX_UNSPEC", "icd10", "some condition, unspecified site",
                           0.9, "retrieval")
        src = MockSource(records={("DX_UNSPEC", "icd10"): {"active": True}},
                         retrieval={("*", "icd10"): [dx]})
        facts = ('{"facts":[{"kind":"diagnosis","description":"some condition",'
                 '"attributes":{"laterality":"right"},"disposition":"performed_today",'
                 '"negated":false,"evidence":["some condition, right side"],'
                 '"confidence":0.98}]}')
        r = code_encounter("e", "some condition, right side documented", "2026-03-14",
                           source=src, extract_llm=lambda s, u: facts,
                           arbitrate_llm=lambda s, u: '{"choice":0,"confidence":0}',
                           modifier_engine=ModifierEngine(defs={"MR": {"description": "Right side of the body"}}),
                           audit_repository=__import__("claude_coder.provenance",
                               fromlist=["NullAuditRepository"]).NullAuditRepository())
        dxln = next(ln for ln in r.billable_lines if ln.chosen.code == "DX_UNSPEC")
        self.assertEqual(dxln.modifiers, [])           # never RT/LT on a diagnosis


class AutonomyVerifiedTest(unittest.TestCase):
    """Release rests on CLOSURE, not a self-reported confidence number. A GROUNDED
    line — deterministic authoritative match or a cross-model-confirmed (VERIFIED)
    entailment — with its gates clear auto-releases regardless of the LLM's
    (poorly calibrated) self-report; the only self-report still consulted is the
    SHAKY_EXTRACTION floor, which reviews a fact the note barely documents. A
    single-model ARBITRATED pick is not grounded and always reviews."""

    def _result(self, method, fact_conf):
        from claude_coder.models import (ClinicalFact, CodingResult, EvidenceSpan,
                                         FactKind, GateResult, Outcome, ResolvedLine)
        f = ClinicalFact(kind=FactKind.PROCEDURE, description="a service",
                         evidence=[EvidenceSpan("a service")], confidence=fact_conf)
        ln = ResolvedLine(fact=f, chosen=CandidateCode("PROC_X", "cpt", "d", 0.9),
                          method=method)
        return CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[ln],
                            gates=[GateResult("g", Outcome.PASS)])

    def test_verified_line_clears_floor(self):
        from claude_coder.autonomy import decide
        from claude_coder.models import ResolutionMethod, Verdict
        r = self._result(ResolutionMethod.VERIFIED, fact_conf=0.98)
        decide(r)
        self.assertEqual(r.verdict, Verdict.AUTO_READY, r.notes)

    def test_verified_line_moderate_confidence_releases_on_closure(self):
        # The #4 change: a VERIFIED line whose extraction self-report is only moderate
        # (0.80 — below the old 0.95 floor, well above the shaky floor) still releases,
        # because grounding + cleared gates are the release criterion, not the number.
        from claude_coder.autonomy import decide
        from claude_coder.models import ResolutionMethod, Verdict
        r = self._result(ResolutionMethod.VERIFIED, fact_conf=0.80)
        decide(r)
        self.assertEqual(r.verdict, Verdict.AUTO_READY, r.notes)

    def test_verified_line_shaky_documentation_reviews(self):
        # A fact the note BARELY documents (below SHAKY_EXTRACTION) gets a human even
        # when its code is grounded — the uncertainty is in the documentation.
        from claude_coder.autonomy import decide, SHAKY_EXTRACTION
        from claude_coder.models import ResolutionMethod, Verdict
        r = self._result(ResolutionMethod.VERIFIED, fact_conf=SHAKY_EXTRACTION - 0.1)
        decide(r)
        self.assertEqual(r.verdict, Verdict.REVIEW_REQUIRED)

    def test_arbitrated_line_reviews(self):
        # A single-model ARBITRATED pick is not grounded and never auto-releases,
        # however high its self-reported confidence.
        from claude_coder.autonomy import decide
        from claude_coder.models import ResolutionMethod, Verdict
        r = self._result(ResolutionMethod.ARBITRATED, fact_conf=0.98)
        decide(r)
        self.assertEqual(r.verdict, Verdict.REVIEW_REQUIRED)


class DiagnosisVerifyTest(unittest.TestCase):
    """Embedding recall and cross-model agreement cannot replace typed distinction
    evidence for diagnosis candidates."""

    def test_entailment_does_not_override_an_untyped_diagnosis_tie(self):
        import json
        import re
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        d_wrong = "condition beta of the structure"      # near-neighbour, higher recall
        d_right = "condition alpha of the structure"      # the entailed one, lower recall
        src = MockSource(
            records={("DXW", "icd10"): {"long_description": d_wrong, "active": True},
                     ("DXR", "icd10"): {"long_description": d_right, "active": True}},
            retrieval={("*", "icd10"): [CandidateCode("DXW", "icd10", d_wrong, 0.95),
                                        CandidateCode("DXR", "icd10", d_right, 0.80)]})

        sel = _sv.judge(entails=lambda d: "alpha" in d.lower() and "beta" not in d.lower(), reason="documented condition")
        corr = _sv.judge(entails=lambda d: "alpha" in d.lower(), reason="x")

        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="condition alpha",
                            evidence=[EvidenceSpan("condition alpha documented")],
                            confidence=0.95)
        line = resolve(_request(fact), src, llm=_from(sel, "provider-a"),
                       corroborate=_from(corr, "provider-b"))
        self.assertEqual(line.method, ResolutionMethod.ABSTAINED)
        self.assertIsNone(line.chosen)
        self.assertEqual({c.code for c in line.alternatives}, {"DXW", "DXR"})


if __name__ == "__main__":
    unittest.main()
