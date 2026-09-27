"""`pipeline.apply_integral_symptom_exclusion`: a released sign/symptom diagnosis is
not additionally reported beside a released definitive diagnosis of the same or a
related site (ICD-10-CM I.B.4/I.B.5; governed parameters in the conventions pack's
`claim_controls`). Sign/symptom vs disease is the SNOMED semantic-tag profile of the
code's mapped source concepts (`icd_semantic_profile`), never a code list; sites are
governed anatomy phrases related through the body-structure graph. Synthetic codes,
descriptors and anatomy throughout.
"""
import json
import tempfile
import unittest
from pathlib import Path

from claude_coder import conventions as _conventions
from claude_coder import pipeline
from claude_coder.data_access import MockSource
from claude_coder.models import (CandidateCode, ClinicalFact, CodingResult, Disposition,
                                 EvidenceSpan, FactKind, ResolutionMethod, ResolvedLine)
from claude_coder.terminology import CONCEPT_RELATED, CONCEPT_SAME

SYMPTOM = "X79.671"      # "Pain in right alpha part" -- mapped only from (finding) concepts
DEFINITIVE = "X89.8X7"   # "Other disorders of hard tissue, right alpha heel" -- (disorder)
ANATOMY = {"alpha part": {"term": "alpha part", "candidates": ["A1"], "method": "exact",
                          "unique": True, "expansions": []},
           "alpha heel": {"term": "alpha heel", "candidates": ["A2"], "method": "exact",
                          "unique": True, "expansions": []}}


def _dx(fid, description, code, descriptor, laterality="right"):
    fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description=description,
                        attributes={"laterality": laterality} if laterality else {},
                        disposition=Disposition.PERFORMED,
                        evidence=[EvidenceSpan(text=description, anchored=True, span_id=f"sp-{fid}")],
                        confidence=0.9, fact_id=fid)
    return ResolvedLine(fact=fact, chosen=CandidateCode(code, "icd10", descriptor),
                        method=ResolutionMethod.VERIFIED)


def _source(*, profile=None, relation=CONCEPT_RELATED, anatomy=ANATOMY):
    if profile is None:
        profile = {SYMPTOM: {"finding": 3}, DEFINITIVE: {"disorder": 5, "finding": 1}}
    return MockSource(semantic_profile=profile, concept_lookup=dict(anatomy),
                     concept_relation={("alpha part", "alpha heel"): relation})


def _result(sym_desc="Pain in right alpha part", dx_desc="Other disorders of hard tissue, right alpha heel"):
    return CodingResult(encounter_id="enc", date_of_service="2026-03-14", lines=[
        _dx("S1", "pain in right alpha part", SYMPTOM, sym_desc),
        _dx("D1", "eponym disorder of the right alpha heel", DEFINITIVE, dx_desc)])


class IntegralSymptomExclusionTest(unittest.TestCase):

    def setUp(self):
        _conventions.load_claim_controls.cache_clear()

    def test_a_symptom_of_a_related_site_is_excluded_with_the_cited_authority(self):
        result = _result()
        pipeline.apply_integral_symptom_exclusion(result, _source())
        sym, dx = result.lines
        self.assertTrue(sym.excluded_reason, sym.excluded_reason)
        self.assertIn("I.B.5", sym.excluded_reason)
        self.assertIn(DEFINITIVE, sym.excluded_reason)
        self.assertEqual(sym.chosen.code, SYMPTOM)                 # kept for audit, not erased
        self.assertIsNone(dx.excluded_reason)
        self.assertEqual(sym.tie_record["claim_level_exclusion"]["definitive_fact_id"], "D1")
        self.assertEqual(sym.tie_record["claim_level_exclusion"]["relation"], CONCEPT_RELATED)

    def test_the_same_site_also_excludes(self):
        result = _result()
        pipeline.apply_integral_symptom_exclusion(result, _source(relation=CONCEPT_SAME))
        self.assertTrue(result.lines[0].excluded_reason)

    def test_an_unrelated_site_keeps_the_symptom(self):
        result = _result()
        pipeline.apply_integral_symptom_exclusion(result, _source(relation="unresolved"))
        self.assertIsNone(result.lines[0].excluded_reason)

    def test_a_contradicting_laterality_keeps_the_symptom(self):
        result = _result(dx_desc="Other disorders of hard tissue, left alpha heel")
        result.lines[1].fact.attributes["laterality"] = "left"
        pipeline.apply_integral_symptom_exclusion(result, _source())
        self.assertIsNone(result.lines[0].excluded_reason)

    def test_without_a_semantic_profile_nothing_is_classified_or_excluded(self):
        result = _result()
        pipeline.apply_integral_symptom_exclusion(result, _source(profile={}))
        self.assertIsNone(result.lines[0].excluded_reason)

    def test_two_definitive_diagnoses_never_exclude_each_other(self):
        result = _result()
        src = _source(profile={SYMPTOM: {"disorder": 2}, DEFINITIVE: {"disorder": 5}})
        pipeline.apply_integral_symptom_exclusion(result, src)
        self.assertIsNone(result.lines[0].excluded_reason)
        self.assertIsNone(result.lines[1].excluded_reason)

    def test_a_symptom_alone_stays_reported(self):
        """I.B.4: with no related definitive diagnosis established, the symptom code
        is the acceptable code."""
        result = _result()
        result.lines = result.lines[:1]
        pipeline.apply_integral_symptom_exclusion(result, _source())
        self.assertIsNone(result.lines[0].excluded_reason)

    def test_without_the_pack_entry_the_control_is_inert(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp) / "pack.json"
            pack.write_text(json.dumps({"conventions": [], "claim_controls": [
                {"id": "x", "enabled": False, "mechanic": "integral_symptom_exclusion"}]}))
            self.assertEqual(_conventions.load_claim_controls("integral_symptom_exclusion", str(pack)), ())
            self.assertEqual(_conventions.load_claim_controls("integral_symptom_exclusion",
                                                              str(Path(tmp) / "missing.json")), ())

    def test_a_mixed_profile_is_classified_by_share_and_a_balanced_one_not_at_all(self):
        """13 finding / 7 disorder concepts (65%) is a symptom code at the pack's 0.6
        threshold; 10 / 10 is unclassified and stays reported."""
        result = _result()
        pipeline.apply_integral_symptom_exclusion(
            result, _source(profile={SYMPTOM: {"finding": 13, "disorder": 7}, DEFINITIVE: {"disorder": 5}}))
        self.assertTrue(result.lines[0].excluded_reason)
        result = _result()
        pipeline.apply_integral_symptom_exclusion(
            result, _source(profile={SYMPTOM: {"finding": 10, "disorder": 10}, DEFINITIVE: {"disorder": 5}}))
        self.assertIsNone(result.lines[0].excluded_reason)

    def test_a_profile_keyed_at_the_stem_covers_its_leaves(self):
        """The extended map often targets a subcategory; a leaf inherits its most
        specific profiled ancestor stem's profile."""
        result = _result()
        pipeline.apply_integral_symptom_exclusion(
            result, _source(profile={"X79.67": {"finding": 4}, "X89.8": {"disorder": 6}}))
        self.assertTrue(result.lines[0].excluded_reason)
        src = _source(profile={"X79.67": {"finding": 4}})
        self.assertEqual(src.icd_semantic_profile("X79.671"), {"finding": 4})
        self.assertEqual(src.icd_semantic_profile("X79.9"), {})

    def test_the_shipped_pack_entry_cites_its_authority(self):
        (control,) = _conventions.load_claim_controls("integral_symptom_exclusion")
        self.assertIn("I.B.5", control["authority"])
        self.assertIn("I.B.4", control["authority"])
        self.assertEqual(control["symptom_semantic_tags"], ["finding"])
        self.assertEqual(control["definitive_semantic_tags"], ["disorder"])


if __name__ == "__main__":
    unittest.main()
