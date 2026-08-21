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

    def test_never_returns_empty_when_input_was_not_empty(self):
        source = MockSource(records={
            ("Z1", "cpt"): {"long_description": "a service", "active": False},
            ("Z2", "cpt"): {"long_description": "a service", "active": False}})
        fact = _fact(FactKind.PROCEDURE, "did a thing")
        candidates = [_candidate("Z1"), _candidate("Z2")]
        result = semelig.eligible_partition([fact], candidates, source, "2026-01-01")
        self.assertEqual(result, candidates)


if __name__ == "__main__":
    unittest.main()
