"""End-to-end tests for the claude-medical-coder pipeline.

Runs the WHOLE flow (extract -> resolve -> arbitrate -> gate -> autonomy ->
certificate) with a MockSource and stubbed LLMs, so it needs no API key, no RAG
index, and — deliberately — contains NO real medical code (the mock uses
synthetic identifiers). It asserts the safety properties, not just happy paths:
planned work is not billed, negated findings are dropped, unsupported evidence
blocks release, and autonomy is granted only when the chain closes.
"""
import unittest

from claude_coder.data_access import MockSource
from claude_coder.models import CandidateCode, Outcome, ResolutionMethod, Verdict
from claude_coder.pipeline import code_encounter


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
NOTE = (
    "Procedure: excision of lesion alpha, right site two. "
    "Assessment: condition alpha, right side. "
    "Patient denies finding gamma. "
    "Plan procedure beta correction next visit."
)

# What the (stubbed) CLU extractor returns: one performed procedure, one current
# diagnosis, one PLANNED procedure (must not bill), one NEGATED finding (drop).
FACTS_JSON = """{"facts":[
 {"kind":"procedure","description":"excision of lesion alpha",
  "attributes":{"laterality":"right","anatomy":"site two","performer_id":"actor-1","billing_entity_id":"actor-1"},
  "disposition":"performed_today","negated":false,
  "evidence":["excision of lesion alpha, right site two"],
  "confidence":0.97,"axis_confidence":{"occurrence":0.99,"action":0.99,"evidence":0.99,"temporal":0.99,"performer":0.99,"relationship":0.99}},
 {"kind":"diagnosis","description":"condition alpha of the right side",
  "attributes":{"laterality":"right"},"disposition":"performed_today","negated":false,
  "evidence":["condition alpha, right side"],"confidence":0.98,
  "axis_confidence":{"occurrence":0.99,"action":0.99,"evidence":0.99,"temporal":0.99,"assertion":0.99,"experiencer":0.99}},
 {"kind":"procedure","description":"procedure beta correction","attributes":{},
  "disposition":"planned","negated":false,
  "evidence":["Plan procedure beta correction next visit"],"confidence":0.9},
 {"kind":"diagnosis","description":"finding gamma","attributes":{},
  "disposition":"performed_today","negated":true,
  "evidence":["denies finding gamma"],"confidence":0.9}
],
 "relations":[
 {"subject_event_id":"F2","object_event_id":"F1","predicate":"reason_for","state":"asserted","evidence_fact_ids":["F1","F2"],"confidence":0.99}
]}"""

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
                              billing_context={"billing_entity_id": "actor-1", "performer_id": "actor-1"})

    def test_happy_path_auto_ready(self):
        r = self._run()
        codes = {ln.chosen.code for ln in r.billable_lines}
        self.assertEqual(codes, {"PROC_ALPHA_EXC", "DX_ALPHA_RIGHT"})
        self.assertEqual(r.verdict, Verdict.AUTO_READY, r.notes)

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
    """Mechanic 2 — descriptor↔fact token support breaks a near-tie in recall
    toward the concept-matching code, and NEVER eliminates a candidate."""

    def test_support_breaks_near_tie(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        # equal recall; one descriptor names the documented concept, the other is a
        # same-score neighbour. Support must pick the concept-matching one.
        match = CandidateCode("P_MATCH", "cpt", "excision of bursa of the foot", 0.80)
        neigh = CandidateCode("P_NEIGH", "cpt", "open treatment of fracture", 0.80)
        src = MockSource(records={("P_MATCH", "cpt"): {"active": True},
                                  ("P_NEIGH", "cpt"): {"active": True}},
                         retrieval={("*", "cpt"): [neigh, match]})  # neighbour listed first
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="excision of bursa",
                            evidence=[EvidenceSpan("the bursa was excised")], confidence=0.9)
        line = resolve(_request(fact), src)
        self.assertEqual(line.chosen.code, "P_MATCH", line.rationale)

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
        import json
        import re

        def stub(system, user):
            if "propose" in system.lower():
                return json.dumps({"codes": list(propose)})
            # select_entailed stub: pick the option whose descriptor names the
            # DOCUMENTED act (alpha) and not the near-synonym (beta).
            block = user.split("CANDIDATE OFFICIAL DESCRIPTORS:", 1)[-1]
            opts = re.findall(r"(?m)^(\d+)\.\s+(.*)$", block)
            if entail:
                for num, desc in opts:
                    d = desc.lower()
                    if "alpha" in d and "beta" not in d:
                        return json.dumps({"choice": int(num), "reason": "documented act matches"})
            return json.dumps({"choice": 0, "reason": "none entailed"})
        return stub

    def test_rejects_near_synonym_accepts_entailed(self):
        from claude_coder.models import ResolutionMethod
        from claude_coder.resolution import resolve
        line = resolve(_request(self._fact()), self._src(), llm=self._llm())
        self.assertEqual(line.method, ResolutionMethod.VERIFIED)
        self.assertEqual(line.chosen.code, "CODEALPHA")   # not the higher-recall near-synonym

    def test_proposal_surfaces_missed_code(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import ResolutionMethod
        from claude_coder.resolution import resolve
        # retrieval only surfaces the WRONG code; the model proposes the right one,
        # which is validated against the registry and then verified.
        src = MockSource(
            records={("CODEALPHA", "cpt"): {"long_description": self.ALPHA, "active": True},
                     ("CODEBETA", "cpt"): {"long_description": self.BETA, "active": True}},
            retrieval={("*", "cpt"): [CandidateCode("CODEBETA", "cpt", self.BETA, 0.95)]})
        line = resolve(_request(self._fact()), src, llm=self._llm(propose=["CODEALPHA"]))
        self.assertEqual(line.method, ResolutionMethod.VERIFIED)
        self.assertEqual(line.chosen.code, "CODEALPHA")

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
        import json

        def stub(system, user):
            return json.dumps({"entailed": bool(confirm),
                               "missing_element": bool(missing),
                               "reason": "second opinion"})
        return stub

    def test_corroboration_agreement_accepts(self):
        from claude_coder.models import ResolutionMethod
        from claude_coder.resolution import resolve
        line = resolve(_request(self._fact()), self._src(), llm=self._llm(),
                       corroborate=self._corroborator(confirm=True))
        self.assertEqual(line.method, ResolutionMethod.VERIFIED)
        self.assertEqual(line.chosen.code, "CODEALPHA")
        self.assertIn("independently confirmed", line.rationale)

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

        def sel(system, user):
            if "propose" in system.lower():
                return json.dumps({"codes": []})
            block = user.split("CANDIDATE OFFICIAL DESCRIPTORS:", 1)[-1]
            for num, desc in re.findall(r"(?m)^(\d+)\.\s+(.*)$", block):
                if "alpha" in desc.lower():          # picks first alpha still on the list
                    return json.dumps({"choice": int(num), "reason": "alpha"})
            return json.dumps({"choice": 0})

        def corr(system, user):
            m = re.search(r"CANDIDATE OFFICIAL DESCRIPTOR: (.+)", user)
            ok = "secondary" in (m.group(1).lower() if m else "")   # confirms only A2
            return json.dumps({"entailed": ok, "missing_element": False, "reason": "x"})

        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="act alpha",
                            evidence=[EvidenceSpan("act alpha performed")], confidence=0.9)
        line = resolve(_request(fact), src, llm=sel, corroborate=corr)
        self.assertEqual(line.method, ResolutionMethod.VERIFIED)
        self.assertEqual(line.chosen.code, "A2")     # re-selected past the rejected A1


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

        def reject(system, user):
            sl = system.lower()
            if "propose" in sl:
                return '{"codes": []}'
            if "independently" in sl:
                return '{"entailed": false, "missing_element": false, "reason": "no"}'
            return '{"choice": 0, "reason": "none entailed"}'

        line = resolve(_request(fact), src, llm=reject, corroborate=reject)
        self.assertFalse(line.resolved)                       # not billed on learned trust
        self.assertEqual(line.method, ResolutionMethod.ABSTAINED)
        self.assertNotIn("learned verified-resolution index", line.rationale)


class RecommendationsTest(unittest.TestCase):
    """Documentation recommendations are derived agnostically from fact kinds,
    resolution methods, and gate outcomes — no code/term/scenario. Abstract inputs."""

    def _line(self, resolved, doc_gap=None, rationale="r"):
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod, ResolvedLine)
        f = ClinicalFact(kind=FactKind.PROCEDURE, description="a documented service",
                         evidence=[EvidenceSpan("a documented service performed")])
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
    """Diagnoses that reach the embedding fallback are now entailment-verified +
    cross-model corroborated (same discipline as procedures), so a WRONG-concept
    code the embedding ranked highest is rejected for the entailed one. Abstract."""

    def test_entailment_overrides_wrong_embedding_top(self):
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

        def sel(system, user):
            if "propose" in system.lower():
                return json.dumps({"codes": []})
            block = user.split("CANDIDATE OFFICIAL DESCRIPTORS:", 1)[-1]
            for num, desc in re.findall(r"(?m)^(\d+)\.\s+(.*)$", block):
                if "alpha" in desc.lower() and "beta" not in desc.lower():
                    return json.dumps({"choice": int(num), "reason": "documented condition"})
            return json.dumps({"choice": 0})

        def corr(system, user):
            m = re.search(r"CANDIDATE OFFICIAL DESCRIPTOR: (.+)", user)
            ok = "alpha" in (m.group(1).lower() if m else "")
            return json.dumps({"entailed": ok, "missing_element": False, "reason": "x"})

        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="condition alpha",
                            evidence=[EvidenceSpan("condition alpha documented")],
                            confidence=0.95)
        line = resolve(_request(fact), src, llm=sel, corroborate=corr)
        self.assertEqual(line.method, ResolutionMethod.VERIFIED)
        self.assertEqual(line.chosen.code, "DXR")   # not the higher-recall wrong code


if __name__ == "__main__":
    unittest.main()
