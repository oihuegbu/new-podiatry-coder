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

# A note whose text contains, verbatim, every evidence span the extractor emits.
NOTE = (
    "Procedure: excision of the interdigital neuroma, right third interspace. "
    "Assessment: interdigital neuroma, right foot. "
    "Patient denies chest pain. "
    "Plan hammertoe correction next visit."
)

# What the (stubbed) CLU extractor returns: one performed procedure, one current
# diagnosis, one PLANNED procedure (must not bill), one NEGATED finding (drop).
FACTS_JSON = """{"facts":[
 {"kind":"procedure","description":"excision of interdigital neuroma",
  "attributes":{"laterality":"right","anatomy":"third interspace"},
  "disposition":"performed_today","negated":false,
  "evidence":["excision of the interdigital neuroma, right third interspace"],
  "confidence":0.97},
 {"kind":"diagnosis","description":"interdigital neuroma of the right foot",
  "attributes":{"laterality":"right"},"disposition":"performed_today","negated":false,
  "evidence":["interdigital neuroma, right foot"],"confidence":0.98},
 {"kind":"procedure","description":"hammertoe correction","attributes":{},
  "disposition":"planned","negated":false,
  "evidence":["Plan hammertoe correction next visit"],"confidence":0.9},
 {"kind":"diagnosis","description":"chest pain","attributes":{},
  "disposition":"performed_today","negated":true,
  "evidence":["denies chest pain"],"confidence":0.9}
]}"""

# Synthetic (non-code) identifiers — no real medical code anywhere in this test.
PROC = CandidateCode("PROC_NEUROMA_EXC", "cpt",
                     "Excision, interdigital neuroma, single, each", 0.9, "retrieval")
DX = CandidateCode("DX_NEUROMA_RIGHT", "icd10",
                   "interdigital neuroma, right foot", 0.9, "retrieval")


def _source():
    return MockSource(
        records={("PROC_NEUROMA_EXC", "cpt"): {"active": True},
                 ("DX_NEUROMA_RIGHT", "icd10"): {"active": True}},
        retrieval={("*", "cpt"): [PROC], ("*", "icd10"): [DX]},
    )


def _extract_stub(system, user):
    return FACTS_JSON


def _arbitrate_stub(system, user):
    return '{"choice":0,"confidence":0.0,"reason":"unused"}'


class AutonomousCoderTest(unittest.TestCase):

    def _run(self, note=NOTE, dos="2026-03-14"):
        return code_encounter("enc-1", note, dos, source=_source(),
                              extract_llm=_extract_stub, arbitrate_llm=_arbitrate_stub)

    def test_happy_path_auto_ready(self):
        r = self._run()
        codes = {ln.chosen.code for ln in r.billable_lines}
        self.assertEqual(codes, {"PROC_NEUROMA_EXC", "DX_NEUROMA_RIGHT"})
        self.assertEqual(r.verdict, Verdict.AUTO_READY, r.notes)

    def test_planned_work_not_billed(self):
        r = self._run()
        billed = {ln.chosen.code for ln in r.billable_lines}
        self.assertNotIn("hammertoe", " ".join(billed).lower())
        # the planned procedure produced no billable line at all
        self.assertEqual(len(r.billable_lines), 2)

    def test_negated_finding_dropped(self):
        r = self._run()
        descs = " ".join(ln.fact.description for ln in r.lines).lower()
        self.assertNotIn("chest pain", descs)

    def test_resolution_is_deterministic(self):
        r = self._run()
        for ln in r.billable_lines:
            self.assertEqual(ln.method, ResolutionMethod.DETERMINISTIC, ln.rationale)

    def test_missing_evidence_blocks_release(self):
        # a note that does NOT contain the procedure's evidence span
        r = self._run(note="Assessment: interdigital neuroma, right foot.")
        ev = next(g for g in r.gates if g.name == "verbatim_evidence")
        self.assertEqual(ev.outcome, Outcome.BLOCKED)
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
        line = resolve(fact, src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "SUP_MED", line.rationale)

    def test_laterality_contradiction_eliminated(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve

        left = CandidateCode("DX_LEFT", "icd10", "interdigital neuroma, left foot", 0.9)
        right = CandidateCode("DX_RIGHT", "icd10", "interdigital neuroma, right foot", 0.9)
        src = MockSource(retrieval={("*", "icd10"): [left, right]})
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="interdigital neuroma",
                            attributes={"laterality": "right"},
                            evidence=[EvidenceSpan("interdigital neuroma right")],
                            confidence=0.99)
        line = resolve(fact, src)
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
                 '"attributes":{},"disposition":"performed_today","negated":false,'
                 '"evidence":["bundled service provided"],"confidence":0.99}]}')
        r = code_encounter("e", "bundled service provided during the visit",
                           "2026-03-14", source=src,
                           extract_llm=lambda s, u: facts,
                           arbitrate_llm=lambda s, u: '{"choice":0,"confidence":0}')
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
        self.assertEqual(eng.assign(self._fact("right"), "Excision, neuroma, each"), ["MR"])
        self.assertEqual(eng.assign(self._fact("left"), "Excision, neuroma, each"), ["ML"])
        self.assertEqual(eng.assign(self._fact("bilateral"), "Excision, neuroma", bilat="1"), ["MB"])
        # descriptor already encodes the side -> no modifier (no double-coding)
        self.assertEqual(eng.assign(self._fact("right"), "Excision, neuroma, right foot"), [])


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
        line = resolve_em(fact, src)
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
    """Deterministic ICD-10-CM resolution via the authoritative Alphabetic Index
    — the permanent fix for the terse-descriptor / eponym gap."""

    def test_index_term_to_code(self):
        from claude_coder.terminology import TerminologyIndex
        idx = TerminologyIndex({"B351": ["tinea unguium", "onychomycosis"],
                                "G5760": ["neuroma mortons", "mortons metatarsalgia"]})
        self.assertEqual(idx.candidates("Onychomycosis"), {"B35.1"})      # exact, dotted
        self.assertEqual(idx.candidates("Morton's neuroma"), {"G57.60"})  # inverted, token-set
        self.assertEqual(idx.candidates("no such term here"), set())      # -> caller falls back

    def test_diagnosis_resolves_via_index_first(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        # Index maps the clinician term; the record supplies the terse descriptor.
        src = MockSource(records={("B35.1", "icd10"):
                                  {"long_description": "Tinea unguium", "active": True}},
                         index={"onychomycosis": {"B35.1"}})
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="onychomycosis",
                            evidence=[EvidenceSpan("onychomycosis")], confidence=0.99)
        line = resolve(fact, src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "B35.1")
        self.assertIn("Alphabetic Index", line.rationale)

    def test_snomed_layer_resolves_when_index_misses(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        # Index has no entry; the SNOMED map resolves the eponym authoritatively.
        src = MockSource(records={("G57.60", "icd10"):
                                  {"long_description": "Lesion of plantar nerve, "
                                   "unspecified lower limb", "active": True}},
                         index={}, snomed={"Morton's neuroma": {"G57.60"}})
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="Morton's neuroma",
                            evidence=[EvidenceSpan("Morton's neuroma")], confidence=0.95)
        line = resolve(fact, src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "G57.60")
        self.assertIn("SNOMED", line.rationale)


if __name__ == "__main__":
    unittest.main()
