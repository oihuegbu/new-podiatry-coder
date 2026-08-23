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


class ActionConceptConflict(unittest.TestCase):
    """Codex F9-R2: a candidate whose compiled action vocabulary shares nothing
    with a DIFFERENT, specific procedure the fact's own text uniquely and
    verifiably names (via the same round-trip-validated synonym scanner advisory
    recall uses) is excluded -- an unrelated service class must not survive
    eligibility purely because generic retrieval found it plausible."""

    def _source(self, candidate_descriptor, matched_descriptor):
        return MockSource(
            records={
                ("CAND1", "cpt"): {"long_description": candidate_descriptor,
                                   "active": True},
                ("MATCH1", "cpt"): {"long_description": matched_descriptor,
                                    "active": True}},
            procedure_concept_lookup={"excision of lesion": {
                "term": "excision of lesion", "candidates": ["MATCH1"],
                "method": "retrieval_consistency_validated", "unique": True,
                "expansions": ["removal of skin lesion"], "source_identity": {}}})

    def test_disjoint_action_vocabulary_is_excluded(self):
        source = self._source("Repair, tendon of foot", "Excision, lesion of skin")
        fact = _fact(FactKind.PROCEDURE,
                     "The surgeon proceeded with excision of lesion along the "
                     "plantar surface.")
        reason = semelig._ineligibility_reason(_candidate("CAND1"), [fact], source, None)
        self.assertIsNotNone(reason, "an unrelated action vocabulary must exclude")
        self.assertIn("MATCH1", reason)

    def test_overlapping_action_vocabulary_stays_eligible(self):
        """The mirror case: the candidate under test shares the SAME action as
        the uniquely-named match (both "excision") -- never excluded, even
        though the anatomy/target wording differs."""
        source = self._source("Excision, tendon of foot", "Excision, lesion of skin")
        fact = _fact(FactKind.PROCEDURE,
                     "The surgeon proceeded with excision of lesion along the "
                     "plantar surface.")
        self.assertTrue(semelig.eligible(_candidate("CAND1"), [fact], source, None))

    def test_no_scan_match_never_excludes(self):
        """Nothing in the fact's text uniquely names a different procedure --
        the mechanism must degrade to "nothing to check", not invent a
        conflict from silence."""
        source = self._source("Repair, tendon of foot", "Excision, lesion of skin")
        fact = _fact(FactKind.PROCEDURE, "A tendon repair was performed today.")
        self.assertTrue(semelig.eligible(_candidate("CAND1"), [fact], source, None))

    def test_the_matched_candidate_itself_is_never_excluded_by_its_own_match(self):
        """A candidate the scan resolves TO is not "a different procedure" from
        itself -- excluding it would be nonsensical."""
        source = self._source("Excision, lesion of skin", "Excision, lesion of skin")
        fact = _fact(FactKind.PROCEDURE,
                     "The surgeon proceeded with excision of lesion along the "
                     "plantar surface.")
        self.assertTrue(semelig.eligible(_candidate("MATCH1"), [fact], source, None))


class AnatomyDominance(unittest.TestCase):
    """Codex F9-R2, second pass: a candidate whose anatomy compatibility is UNKNOWN
    is removed only when a SIBLING candidate in the same pool is positively grounded
    to the fact's documented anatomy via the governed concept-relation index --
    never by asserting the excluded candidate's anatomy IS different (that verdict
    does not exist)."""

    def _source(self, relation_map=None):
        return MockSource(
            records={
                ("GROUNDED", "cpt"): {"long_description": "Excision, great toe",
                                      "active": True},
                ("UNGROUNDED", "cpt"): {"long_description": "Excision, unrelated site",
                                        "active": True},
                ("OTHER_GROUNDED", "cpt"): {"long_description": "Repair, great toe",
                                            "active": True}},
            concept_relation=relation_map or {})

    def test_ungrounded_sibling_excluded_when_another_is_positively_grounded(self):
        source = self._source({("great toe", "great toe"): "same"})
        fact = ClinicalFact(FactKind.PROCEDURE, "a procedure",
                            attributes={"anatomy": "great toe"})
        result = semelig.eligible_partition(
            [fact], [_candidate("GROUNDED"), _candidate("UNGROUNDED")], source, None)
        self.assertEqual([c.code for c in result], ["GROUNDED"])

    def test_hierarchical_grounding_also_dominates(self):
        source = self._source({("great toe", "great toe"): "ancestor_descendant"})
        fact = ClinicalFact(FactKind.PROCEDURE, "a procedure",
                            attributes={"anatomy": "great toe"})
        result = semelig.eligible_partition(
            [fact], [_candidate("GROUNDED"), _candidate("UNGROUNDED")], source, None)
        self.assertEqual([c.code for c in result], ["GROUNDED"])

    def test_all_unknown_excludes_nothing(self):
        """No candidate is positively grounded -- absence of grounding is not
        evidence against anyone, so nothing is removed on this basis."""
        source = self._source({})
        fact = ClinicalFact(FactKind.PROCEDURE, "a procedure",
                            attributes={"anatomy": "great toe"})
        result = semelig.eligible_partition(
            [fact], [_candidate("GROUNDED"), _candidate("UNGROUNDED")], source, None)
        self.assertEqual({c.code for c in result}, {"GROUNDED", "UNGROUNDED"})

    def test_two_grounded_candidates_both_survive(self):
        """Dominance removes the UNGROUNDED sibling, never a second grounded one --
        this is not a uniqueness/tie-break mechanism."""
        source = self._source({("great toe", "great toe"): "same"})
        fact = ClinicalFact(FactKind.PROCEDURE, "a procedure",
                            attributes={"anatomy": "great toe"})
        result = semelig.eligible_partition(
            [fact], [_candidate("GROUNDED"), _candidate("OTHER_GROUNDED"),
                    _candidate("UNGROUNDED")], source, None)
        self.assertEqual({c.code for c in result}, {"GROUNDED", "OTHER_GROUNDED"})

    def test_no_documented_anatomy_excludes_nothing(self):
        """The fact never states an anatomy value at all -- there is nothing to
        ground anyone against, so dominance cannot fire."""
        source = self._source({("great toe", "great toe"): "same"})
        fact = ClinicalFact(FactKind.PROCEDURE, "a procedure")
        result = semelig.eligible_partition(
            [fact], [_candidate("GROUNDED"), _candidate("UNGROUNDED")], source, None)
        self.assertEqual({c.code for c in result}, {"GROUNDED", "UNGROUNDED"})

    def test_explicit_laterality_contradiction_excludes_regardless_of_dominance(self):
        source = MockSource(records={
            ("LEFT", "cpt"): {"long_description": "Excision, left great toe",
                              "active": True},
            ("RIGHT", "cpt"): {"long_description": "Excision, right great toe",
                               "active": True}})
        fact = ClinicalFact(FactKind.PROCEDURE, "a procedure",
                            attributes={"laterality": "left"})
        result = semelig.eligible_partition(
            [fact], [_candidate("LEFT"), _candidate("RIGHT")], source, None)
        self.assertEqual([c.code for c in result], ["LEFT"])

    def test_eligibility_report_reason_matches_what_partition_actually_excluded(self):
        """The audit trail's stated reason must never drift from what
        `eligible_partition` actually enforced -- including the dominance pass, not
        only the single-candidate checks."""
        source = self._source({("great toe", "great toe"): "same"})
        fact = ClinicalFact(FactKind.PROCEDURE, "a procedure",
                            attributes={"anatomy": "great toe"})
        candidates = [_candidate("GROUNDED"), _candidate("UNGROUNDED")]
        kept = {c.code for c in semelig.eligible_partition([fact], candidates, source, None)}
        report = {r["code"]: r for r in semelig.eligibility_report(
            [fact], candidates, source, None)}
        for code, entry in report.items():
            self.assertEqual(entry["eligible"], code in kept, report)
        self.assertIsNotNone(report["UNGROUNDED"]["reason"])
        self.assertIsNone(report["GROUNDED"]["reason"])

    def test_dominance_fires_end_to_end_through_resolve(self):
        """The mechanism through the REAL pipeline entry point, not just the module
        in isolation: an ungrounded candidate must never be the one `resolve()`
        actually releases when a grounded sibling was retrieved for the same fact."""
        from claude_coder.resolution import resolve
        from tests.test_measurement import _request

        source = MockSource(
            records={
                ("GROUNDED", "cpt"): {"long_description": "Excision, great toe",
                                      "active": True},
                ("UNGROUNDED", "cpt"): {"long_description": "Excision, unrelated site",
                                        "active": True}},
            concept_relation={("great toe", "great toe"): "same"},
            retrieval={("*", "cpt"): [_candidate("GROUNDED"), _candidate("UNGROUNDED")]})
        fact = ClinicalFact(FactKind.PROCEDURE, "a procedure",
                            attributes={"anatomy": "great toe"}, fact_id="fx")
        line = resolve(_request(fact), source)
        self.assertIsNotNone(line.candidate_eligibility)
        report = {r["code"]: r for r in line.candidate_eligibility}
        self.assertFalse(report["UNGROUNDED"]["eligible"], report)
        self.assertTrue(report["GROUNDED"]["eligible"], report)
        if line.chosen is not None:
            self.assertEqual(line.chosen.code, "GROUNDED")


if __name__ == "__main__":
    unittest.main()
