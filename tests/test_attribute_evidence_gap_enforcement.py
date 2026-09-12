"""issue #6, Codex's independent re-review (F9-R14-A): `ClinicalFact.
attribute_evidence_gaps` is recorded by `extraction.finalize_attribute_evidence`
but was, before this round, otherwise inert -- not copied across an event-union
reading, not bound into the eligibility snapshot digest, and never re-checked
before a resolved line released a code. These tests pin the three enforcement
points added to close that: the shared post-retrieval guard in `resolution`,
the digest binding in `eligibility.fact_snapshot_digest`, and the independent-
copy fix in `event_union._copy_fact`. Synthetic facts/codes throughout.
"""
import unittest
from dataclasses import replace

from claude_coder import eligibility as elig
from claude_coder import event_union
from claude_coder import resolution
from claude_coder.models import (AttributeEvidenceGap, CandidateCode, ClinicalFact,
                                 FactKind, ResolutionMethod, ResolvedLine)


def _fact(**over):
    base = dict(kind=FactKind.PROCEDURE, description="assembly service",
               confidence=0.9, fact_id="F1")
    base.update(over)
    return ClinicalFact(**base)


def _cand(code="CAND_A", descriptor="assembly service"):
    return CandidateCode(code=code, system="cpt", descriptor=descriptor,
                         score=0.9, source="retrieval")


class ApplyAttributeEvidenceGapGuardTest(unittest.TestCase):

    def test_a_line_with_no_gap_passes_through_unchanged(self):
        fact = _fact()
        chosen = _cand()
        line = ResolvedLine(fact=fact, chosen=chosen, method=ResolutionMethod.VERIFIED)
        out = resolution._apply_attribute_evidence_gap_guard(line, None)
        self.assertIs(out, line)
        self.assertEqual(out.chosen, chosen)

    def test_an_already_unresolved_line_is_a_no_op(self):
        fact = _fact(attribute_evidence_gaps={
            "laterality": AttributeEvidenceGap(axis="laterality", reason="no evidence")})
        line = ResolvedLine(fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED)
        out = resolution._apply_attribute_evidence_gap_guard(line, None)
        self.assertIs(out, line)

    def test_a_gapped_fact_never_releases_its_selected_code(self):
        """The core invariant: no unresolved attribute_evidence_gap may coexist
        with a released selected code."""
        fact = _fact(attribute_evidence_gaps={
            "laterality": AttributeEvidenceGap(axis="laterality", reason="no evidence")})
        chosen = _cand()
        line = ResolvedLine(fact=fact, chosen=chosen, method=ResolutionMethod.VERIFIED)
        out = resolution._apply_attribute_evidence_gap_guard(line, None)
        self.assertIsNone(out.chosen)
        self.assertIn(chosen, out.alternatives)
        self.assertEqual(out.method, ResolutionMethod.ABSTAINED)

    def test_withdrawn_code_names_the_exact_fact_and_axis(self):
        fact = _fact(attribute_evidence_gaps={
            "laterality": AttributeEvidenceGap(axis="laterality", reason="no evidence")})
        line = ResolvedLine(fact=fact, chosen=_cand(), method=ResolutionMethod.VERIFIED)
        out = resolution._apply_attribute_evidence_gap_guard(line, None)
        gap = out.attribute_evidence_gap
        self.assertEqual(gap["fact_id"], "F1")
        self.assertEqual(gap["axes"], ["laterality"])

    def test_multiple_gapped_axes_are_all_named(self):
        fact = _fact(attribute_evidence_gaps={
            "laterality": AttributeEvidenceGap(axis="laterality", reason="no evidence"),
            "anatomy": AttributeEvidenceGap(axis="anatomy", reason="no evidence")})
        line = ResolvedLine(fact=fact, chosen=_cand(), method=ResolutionMethod.VERIFIED)
        out = resolution._apply_attribute_evidence_gap_guard(line, None)
        self.assertEqual(out.attribute_evidence_gap["axes"], ["anatomy", "laterality"])

    def test_no_coverage_never_becomes_a_provider_query(self):
        """issue #6, Codex's independent re-review, F9-R14-A item 6: without a
        complete, independently-read document search, this must never be
        labelled a provider question -- `documentation_gap` (which `autonomy.
        decide` reads as PROVIDER_QUERY) must stay unset."""
        fact = _fact(attribute_evidence_gaps={
            "laterality": AttributeEvidenceGap(axis="laterality", reason="no evidence")})
        line = ResolvedLine(fact=fact, chosen=_cand(), method=ResolutionMethod.VERIFIED)
        out = resolution._apply_attribute_evidence_gap_guard(line, None)
        self.assertIsNone(out.documentation_gap)
        self.assertFalse(out.attribute_evidence_gap["coverage_complete"])

    def test_incomplete_coverage_never_becomes_a_provider_query(self):
        class _Incomplete:
            complete = False
        fact = _fact(attribute_evidence_gaps={
            "laterality": AttributeEvidenceGap(axis="laterality", reason="no evidence")})
        line = ResolvedLine(fact=fact, chosen=_cand(), method=ResolutionMethod.VERIFIED)
        out = resolution._apply_attribute_evidence_gap_guard(line, _Incomplete())
        self.assertIsNone(out.documentation_gap)

    def test_complete_coverage_is_a_genuine_provider_question(self):
        class _Complete:
            complete = True
        fact = _fact(attribute_evidence_gaps={
            "laterality": AttributeEvidenceGap(axis="laterality", reason="no evidence")})
        line = ResolvedLine(fact=fact, chosen=_cand(), method=ResolutionMethod.VERIFIED)
        out = resolution._apply_attribute_evidence_gap_guard(line, _Complete())
        self.assertIsNotNone(out.documentation_gap)
        self.assertTrue(out.attribute_evidence_gap["coverage_complete"])

    def test_a_clean_fact_alongside_a_gapped_one_is_unaffected(self):
        """issue #6, Codex's independent re-review, required regression: a
        persistently gapped fact must hold only its own line -- a separate,
        cleanly evidenced fact's line is untouched by this guard."""
        gapped = _fact(fact_id="F1", attribute_evidence_gaps={
            "laterality": AttributeEvidenceGap(axis="laterality", reason="no evidence")})
        clean = _fact(fact_id="F2")
        gapped_line = resolution._apply_attribute_evidence_gap_guard(
            ResolvedLine(fact=gapped, chosen=_cand("CAND_A"),
                        method=ResolutionMethod.VERIFIED), None)
        clean_line = resolution._apply_attribute_evidence_gap_guard(
            ResolvedLine(fact=clean, chosen=_cand("CAND_B"),
                        method=ResolutionMethod.VERIFIED), None)
        self.assertIsNone(gapped_line.chosen)
        self.assertEqual(clean_line.chosen.code, "CAND_B")
        self.assertIsNone(clean_line.attribute_evidence_gap)


class FactSnapshotDigestGapBindingTest(unittest.TestCase):
    """issue #6, Codex's independent re-review, F9-R14-A: a gap recorded or
    cleared after eligibility approved a snapshot must trip the digest -- two
    otherwise-identical facts differing only in attribute_evidence_gaps must
    never produce the same digest."""

    def test_a_gap_changes_the_digest(self):
        clean = _fact()
        gapped = _fact(attribute_evidence_gaps={
            "laterality": AttributeEvidenceGap(axis="laterality", reason="no evidence")})
        self.assertNotEqual(elig.fact_snapshot_digest(clean),
                            elig.fact_snapshot_digest(gapped))

    def test_the_gap_reason_itself_is_bound(self):
        gap_a = _fact(attribute_evidence_gaps={
            "laterality": AttributeEvidenceGap(axis="laterality", reason="reason A")})
        gap_b = _fact(attribute_evidence_gaps={
            "laterality": AttributeEvidenceGap(axis="laterality", reason="reason B")})
        self.assertNotEqual(elig.fact_snapshot_digest(gap_a),
                            elig.fact_snapshot_digest(gap_b))

    def test_clearing_a_gap_changes_the_digest_back(self):
        gapped = _fact(attribute_evidence_gaps={
            "laterality": AttributeEvidenceGap(axis="laterality", reason="no evidence")})
        cleared = replace(gapped, attribute_evidence_gaps={})
        self.assertNotEqual(elig.fact_snapshot_digest(gapped),
                            elig.fact_snapshot_digest(cleared))
        self.assertEqual(elig.fact_snapshot_digest(cleared),
                         elig.fact_snapshot_digest(_fact()))


class CopyFactPreservesGapsIndependentlyTest(unittest.TestCase):
    """issue #6, Codex's independent re-review, F9-R14-A: `_copy_fact` must copy
    `attribute_evidence_gaps` like every other mutable field it already copies --
    otherwise two independent-reading copies of the SAME event alias the SAME
    dict, and mutating one (extraction's own in-place gap recording, or
    `graph_consensus`'s gap-clearing) leaks into the other's view of the event."""

    def test_a_second_readings_copy_carries_the_original_gap(self):
        original = _fact(attribute_evidence_gaps={
            "laterality": AttributeEvidenceGap(axis="laterality", reason="no evidence")})
        copy = event_union._copy_fact(original, "second-F1")
        self.assertEqual(copy.attribute_evidence_gaps, original.attribute_evidence_gaps)

    def test_mutating_the_copys_gaps_never_leaks_into_the_original(self):
        original = _fact(attribute_evidence_gaps={
            "laterality": AttributeEvidenceGap(axis="laterality", reason="no evidence")})
        copy = event_union._copy_fact(original, "second-F1")
        copy.attribute_evidence_gaps["anatomy"] = AttributeEvidenceGap(
            axis="anatomy", reason="a defect introduced only on the copy")
        self.assertNotIn("anatomy", original.attribute_evidence_gaps)

    def test_mutating_the_originals_gaps_never_leaks_into_the_copy(self):
        original = _fact(attribute_evidence_gaps={
            "laterality": AttributeEvidenceGap(axis="laterality", reason="no evidence")})
        copy = event_union._copy_fact(original, "second-F1")
        original.attribute_evidence_gaps["anatomy"] = AttributeEvidenceGap(
            axis="anatomy", reason="a defect introduced only on the original")
        self.assertNotIn("anatomy", copy.attribute_evidence_gaps)

    def test_no_gaps_copies_as_an_independent_empty_dict(self):
        original = _fact()
        copy = event_union._copy_fact(original, "second-F1")
        self.assertEqual(copy.attribute_evidence_gaps, {})
        self.assertIsNot(copy.attribute_evidence_gaps, original.attribute_evidence_gaps)


if __name__ == "__main__":
    unittest.main()
