"""`pipeline.apply_section_applicability` excludes a RESOLVED anesthesia-section line
even when it is submission-HELD (unresolved performer): on the operating provider's
claim an anesthesia service is not separately reportable whoever performed it, so a
held one must not turn into a blocking administrative question (third live run).
Synthetic descriptors throughout; the section comes from descriptor grammar.
"""
import unittest

from claude_coder.models import (CandidateCode, ClaimSubmissionStatus, ClinicalFact,
                                 CodingResult, Disposition, EvidenceSpan, FactKind,
                                 ResolutionMethod, ResolvedLine)
from claude_coder.pipeline import apply_section_applicability


def _line(code, descriptor, *, held=False, attrs=None):
    fact = ClinicalFact(kind=FactKind.PROCEDURE, description=descriptor.lower(),
                        attributes=dict(attrs or {}), disposition=Disposition.PERFORMED,
                        evidence=[EvidenceSpan(text=descriptor.lower(), anchored=True,
                                               span_id=f"sp-{code}")],
                        confidence=0.9, fact_id=f"F-{code}")
    line = ResolvedLine(fact=fact, chosen=CandidateCode(code, "cpt", descriptor),
                        method=ResolutionMethod.VERIFIED)
    if held:
        line.claim_submission_status = ClaimSubmissionStatus.HELD
    return line


class HeldAnesthesiaLineTest(unittest.TestCase):

    def test_a_submission_held_anesthesia_line_is_still_excluded(self):
        surgery = _line("SURG_X", "Ostectomy, complete excision of a structure")
        anes = _line("ANES_X", "Anesthesia for open procedures on a structure", held=True)
        result = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[surgery, anes])
        self.assertIn(anes, result.submission_held_lines)          # the shape that escaped before
        apply_section_applicability(result)
        self.assertTrue(anes.excluded_reason)
        self.assertIn("anesthesia-section", anes.excluded_reason)
        self.assertNotIn(anes, result.submission_held_lines)
        self.assertIsNone(surgery.excluded_reason)

    def test_a_held_anesthesia_line_with_a_documented_separate_provider_is_kept(self):
        surgery = _line("SURG_X", "Ostectomy, complete excision of a structure")
        anes = _line("ANES_X", "Anesthesia for open procedures on a structure", held=True,
                     attrs={"anesthesia_provider": True})
        result = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[surgery, anes])
        apply_section_applicability(result)
        self.assertIsNone(anes.excluded_reason)

    def test_a_held_operative_line_still_counts_as_the_operative_claim(self):
        """The operative line itself may be held (dependency/ownership); the claim is
        still an operative claim, so the anesthesia line is excluded."""
        surgery = _line("SURG_X", "Ostectomy, complete excision of a structure", held=True)
        anes = _line("ANES_X", "Anesthesia for open procedures on a structure")
        result = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[surgery, anes])
        apply_section_applicability(result)
        self.assertTrue(anes.excluded_reason)


if __name__ == "__main__":
    unittest.main()
