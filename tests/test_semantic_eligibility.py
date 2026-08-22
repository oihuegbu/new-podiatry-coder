"""Tests for claude_coder.semantic_eligibility (issue #6 items 4/5).

Synthetic identifiers throughout, per the same discipline as `MockSource`'s own
docstring: the test suite contains no real medical code.
"""
import unittest

from claude_coder import semantic_eligibility as semelig
from claude_coder.data_access import MockSource
from claude_coder.models import CandidateCode, ClinicalFact, FactKind


def _candidate(code, system="cpt"):
    return CandidateCode(code=code, system=system, descriptor="", score=0.5)


def _fact(kind, description):
    return ClinicalFact(kind, description)


class EligibleDefaultsToTrue(unittest.TestCase):
    def test_no_compiled_record_is_eligible(self):
        source = MockSource()
        self.assertTrue(semelig.eligible(_candidate("Z1"), [_fact(FactKind.PROCEDURE, "a thing")],
                                         source, None))

    def test_no_axis_conflict_is_eligible(self):
        source = MockSource(records={("Z1", "cpt"): {"long_description": "Excision, first thing"}})
        self.assertTrue(semelig.eligible(_candidate("Z1"), [_fact(FactKind.PROCEDURE, "did a thing")],
                                         source, None))


class MeasurementRequirement(unittest.TestCase):
    def setUp(self):
        # a bounded interval in the descriptor triggers `required_attributes` == ["measurement"]
        self.source = MockSource(records={
            ("Z1", "cpt"): {"long_description": "Repair, wound 3 cm or less"}})

    def test_candidate_requiring_measurement_excluded_when_fact_has_none(self):
        fact = _fact(FactKind.PROCEDURE, "repaired a wound")
        self.assertFalse(semelig.eligible(_candidate("Z1"), [fact], self.source, None))

    def test_candidate_requiring_measurement_kept_when_fact_documents_one(self):
        fact = _fact(FactKind.PROCEDURE, "repaired a wound measuring 2 cm or less")
        self.assertTrue(semelig.eligible(_candidate("Z1"), [fact], self.source, None))

    def test_candidate_requiring_measurement_kept_when_attribute_documents_one(self):
        """A prior version of this check read ONLY the fact's prose description --
        a measurement correctly extracted into a structured attribute (e.g.
        `length_cm`) instead of left in the text always looked unmeasured. Exposed
        once `eligible_partition` stopped restoring an all-excluded pool (Codex
        F8-R2); this is the direct regression test. Dimension-matched to the "Z1"
        candidate's own "cm" (length) requirement -- see `MeasurementDimensionScoping`
        for the dimension-MISMATCH case this could otherwise mask."""
        fact = ClinicalFact(FactKind.PROCEDURE, "repaired a wound",
                            attributes={"length_cm": 2})
        self.assertTrue(semelig.eligible(_candidate("Z1"), [fact], self.source, None))


class MeasurementDimensionScoping(unittest.TestCase):
    """Codex F8-R2, round 2: a documented measurement must match the CANDIDATE's own
    required dimension, not merely exist somewhere in the intent. An earlier version
    accepted ANY measurement anywhere for ANY candidate requiring one, so an
    unrelated AREA measurement on one component could satisfy a candidate requiring
    LENGTH on a different one."""

    def setUp(self):
        self.length_candidate = MockSource(records={
            ("LEN", "cpt"): {"long_description": "excision, length 5 cm or less"}})
        self.area_candidate = MockSource(records={
            ("AREA", "cpt"): {"long_description": "excision, area 16 sq cm or less"}})

    def test_unrelated_dimension_measurement_does_not_satisfy_the_requirement(self):
        fact = ClinicalFact(FactKind.PROCEDURE, "excision", attributes={"size_sqcm": 10})
        self.assertFalse(semelig.eligible(_candidate("LEN"), [fact],
                                          self.length_candidate, None))

    def test_matching_dimension_measurement_satisfies_the_requirement(self):
        fact = ClinicalFact(FactKind.PROCEDURE, "excision", attributes={"length_cm": 3})
        self.assertTrue(semelig.eligible(_candidate("LEN"), [fact],
                                         self.length_candidate, None))

    def test_matching_dimension_from_a_different_intent_component_still_satisfies(self):
        """Dimension-scoped, not fact-scoped: a measurement documented on ANY member
        of the intent still counts, as long as its dimension matches."""
        f1 = ClinicalFact(FactKind.PROCEDURE, "first thing", fact_id="F1")
        f2 = ClinicalFact(FactKind.PROCEDURE, "second thing", fact_id="F2",
                          attributes={"length_cm": 3})
        self.assertTrue(semelig.eligible(_candidate("LEN"), [f1, f2],
                                         self.length_candidate, None))

    def test_area_requirement_rejects_a_length_only_measurement(self):
        fact = ClinicalFact(FactKind.PROCEDURE, "excision", attributes={"length_cm": 3})
        self.assertFalse(semelig.eligible(_candidate("AREA"), [fact],
                                          self.area_candidate, None))


class SemanticClassConflict(unittest.TestCase):
    def test_em_fact_excludes_a_non_em_classified_candidate(self):
        source = MockSource(records={("Z1", "cpt"): {"long_description": "some service"}},
                            semantic_class={"Z1": "surgical_procedure"})
        fact = _fact(FactKind.EM, "an office visit")
        self.assertFalse(semelig.eligible(_candidate("Z1"), [fact], source, None))

    def test_em_fact_keeps_an_em_classified_candidate(self):
        source = MockSource(records={("Z1", "cpt"): {"long_description": "some service"}},
                            semantic_class={"Z1": "evaluation_management"})
        fact = _fact(FactKind.EM, "an office visit")
        self.assertTrue(semelig.eligible(_candidate("Z1"), [fact], source, None))

    def test_non_em_fact_never_checks_semantic_class(self):
        source = MockSource(records={("Z1", "cpt"): {"long_description": "some service"}},
                            semantic_class={"Z1": "surgical_procedure"})
        fact = _fact(FactKind.PROCEDURE, "did a thing")
        self.assertTrue(semelig.eligible(_candidate("Z1"), [fact], source, None))

    def test_unclassified_candidate_never_excluded_by_semantic_class(self):
        source = MockSource(records={("Z1", "cpt"): {"long_description": "some service"}})
        fact = _fact(FactKind.EM, "an office visit")
        self.assertTrue(semelig.eligible(_candidate("Z1"), [fact], source, None))


class ActiveOnDateOfService(unittest.TestCase):
    def test_blocked_status_excludes_the_candidate(self):
        source = MockSource(records={("Z1", "cpt"): {"long_description": "a service", "active": False}})
        fact = _fact(FactKind.PROCEDURE, "did a thing")
        self.assertFalse(semelig.eligible(_candidate("Z1"), [fact], source, "2026-01-01"))

    def test_active_status_keeps_the_candidate(self):
        source = MockSource(records={("Z1", "cpt"): {"long_description": "a service", "active": True}})
        fact = _fact(FactKind.PROCEDURE, "did a thing")
        self.assertTrue(semelig.eligible(_candidate("Z1"), [fact], source, "2026-01-01"))

    def test_no_date_of_service_never_excludes(self):
        source = MockSource(records={("Z1", "cpt"): {"long_description": "a service", "active": False}})
        fact = _fact(FactKind.PROCEDURE, "did a thing")
        self.assertTrue(semelig.eligible(_candidate("Z1"), [fact], source, None))


class EligiblePartition(unittest.TestCase):
    def test_empty_candidates_returns_empty(self):
        source = MockSource()
        self.assertEqual(semelig.eligible_partition([], [], source, None), [])

    def test_narrows_to_only_eligible_candidates(self):
        source = MockSource(records={
            ("Z1", "cpt"): {"long_description": "a service", "active": True},
            ("Z2", "cpt"): {"long_description": "a service", "active": False}})
        fact = _fact(FactKind.PROCEDURE, "did a thing")
        result = semelig.eligible_partition([fact], [_candidate("Z1"), _candidate("Z2")],
                                            source, "2026-01-01")
        self.assertEqual([c.code for c in result], ["Z1"])

    def test_all_ineligible_returns_empty_not_a_restored_pool(self):
        """Codex F8-R2: an earlier version of `eligible_partition` restored the
        unfiltered pool whenever every candidate was excluded, reasoning it was
        more likely a sign the filter didn't fit than evidence nothing retrieved
        was usable. That silently let a structurally-incompatible candidate reach
        the resolver on the clearest possible eligibility signal against it, and
        made `eligibility_report` (never subject to the same fallback) disagree
        with what was actually enforced. `eligible_partition` is now a pure,
        monotonic filter -- the caller (`resolution.resolve`) treats an empty
        result as an honest abstention, not a reason to disable the filter."""
        source = MockSource(records={
            ("Z1", "cpt"): {"long_description": "a service", "active": False},
            ("Z2", "cpt"): {"long_description": "a service", "active": False}})
        fact = _fact(FactKind.PROCEDURE, "did a thing")
        candidates = [_candidate("Z1"), _candidate("Z2")]
        result = semelig.eligible_partition([fact], candidates, source, "2026-01-01")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
