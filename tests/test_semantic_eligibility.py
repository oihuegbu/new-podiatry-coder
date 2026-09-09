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


class AdvisoryTermsNeverExcludeACandidate(unittest.TestCase):
    """Codex F9-R2-A: a first attempt at an "action concept" eligibility check
    routed the advisory (LLM-generated, retrieval-consistency-validated) procedure-
    synonym scan into an EXCLUSION decision. That directly violated the advisory
    tier's own, already-reviewed trust-tier contract
    (`resolution._advisory_procedure_expansions`'s own docstring,
    `ResolvedLine.advisory_terminology`'s own contract): this data may WIDEN RECALL
    ONLY -- never settle identity, exclude a candidate, or authorize release.
    Reverted; these are the regressions that must never let it come back,
    including Codex's own reproduction (one service intent documenting two
    performed actions, where a naive scan-first-match approach excluded the
    SECOND action's candidate because the scanner encountered the FIRST action's
    advisory match first)."""

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

    def test_a_uniquely_named_different_procedure_never_excludes_a_candidate(self):
        """The exact shape the reverted check used to exclude on: the fact's text
        uniquely names a different, verified procedure with a disjoint compiled
        action vocabulary. Advisory data may inform recall, never eligibility."""
        source = self._source("Repair, tendon of foot", "Excision, lesion of skin")
        fact = _fact(FactKind.PROCEDURE,
                     "The surgeon proceeded with excision of lesion along the "
                     "plantar surface.")
        self.assertTrue(semelig.eligible(_candidate("CAND1"), [fact], source, None),
                        "advisory terminology must never decide eligibility")

    def test_two_documented_actions_in_one_intent_exclude_neither_candidate(self):
        """Codex's own reproduction: a multi-fact service intent documents TWO
        performed actions. Neither candidate -- for either action -- may be
        excluded because the advisory scanner happened to match a DIFFERENT
        action first."""
        source = self._source("Repair, tendon of foot", "Excision, lesion of skin")
        first_action = _fact(FactKind.PROCEDURE,
                             "excision of lesion performed on the plantar surface")
        second_action = _fact(FactKind.PROCEDURE, "tendon repair also performed")
        facts = [first_action, second_action]
        self.assertTrue(semelig.eligible(_candidate("CAND1"), facts, source, None))
        self.assertTrue(semelig.eligible(_candidate("MATCH1"), facts, source, None))


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
        from claude_coder.models import AttributeEvidence, EvidenceSpan, RelationState
        span = EvidenceSpan("a procedure on the left", anchored=True, span_id="s1")
        fact = ClinicalFact(FactKind.PROCEDURE, "a procedure",
                            attributes={"laterality": "left"}, evidence=[span],
                            attribute_evidence={"laterality": (
                                AttributeEvidence(span=span, assertion_state=RelationState.ASSERTED,
                                                  value="left"),)})
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


class AnatomyPhraseDecomposition(unittest.TestCase):
    """Codex F9-R2-C: the anatomy check went UNKNOWN almost everywhere on the live
    note because it only ever tried one raw, possibly-composite attribute string
    whole. It must decompose a composite mention on generic list/conjunction
    punctuation, and also consume the ALREADY governed-normalized synonym
    expansions (`fact.governed_terms["anatomy"]`), not just the raw phrase."""

    def test_a_composite_mention_grounds_via_its_second_structure(self):
        """"tendon and calcaneus" names TWO structures -- only "calcaneus" grounds
        the candidate here, and a single whole-string comparison would have missed
        it entirely."""
        source = MockSource(
            records={
                ("GROUNDED", "cpt"): {"long_description": "Excision, calcaneus",
                                      "active": True},
                ("UNGROUNDED", "cpt"): {"long_description": "Excision, unrelated site",
                                        "active": True}},
            concept_relation={("calcaneus", "calcaneus"): "same"})
        fact = ClinicalFact(FactKind.PROCEDURE, "a procedure",
                            attributes={"anatomy": "tendon and calcaneus"})
        result = semelig.eligible_partition(
            [fact], [_candidate("GROUNDED"), _candidate("UNGROUNDED")], source, None)
        self.assertEqual([c.code for c in result], ["GROUNDED"])

    def test_a_governed_synonym_expansion_grounds_when_the_raw_phrase_does_not(self):
        """The raw attribute phrase itself doesn't match anything, but a governed
        synonym expansion already resolved for it (the SAME normalization
        `coreference.normalize_fact_terminology` writes to `fact.governed_terms`)
        does."""
        source = MockSource(
            records={
                ("GROUNDED", "cpt"): {"long_description": "Excision, heel bone",
                                      "active": True},
                ("UNGROUNDED", "cpt"): {"long_description": "Excision, unrelated site",
                                        "active": True}},
            concept_relation={("heel prominence", "heel bone"): "same"})
        fact = ClinicalFact(
            FactKind.PROCEDURE, "a procedure",
            attributes={"anatomy": "raw note phrasing with no direct match"},
            governed_terms={"anatomy": ("heel prominence",)})
        result = semelig.eligible_partition(
            [fact], [_candidate("GROUNDED"), _candidate("UNGROUNDED")], source, None)
        self.assertEqual([c.code for c in result], ["GROUNDED"])

    def test_a_slash_separated_composite_mention_grounds_via_its_second_structure(self):
        """Codex F9-R2-C, third pass: real extraction output commonly uses
        "structure alpha / structure beta" for a composite mention, not "and" --
        the slash form must decompose exactly like the "and" form does."""
        source = MockSource(
            records={
                ("GROUNDED", "cpt"): {"long_description": "Excision, calcaneus",
                                      "active": True},
                ("UNGROUNDED", "cpt"): {"long_description": "Excision, unrelated site",
                                        "active": True}},
            concept_relation={("calcaneus", "calcaneus"): "same"})
        fact = ClinicalFact(FactKind.PROCEDURE, "a procedure",
                            attributes={"anatomy": "tendon / calcaneus"})
        result = semelig.eligible_partition(
            [fact], [_candidate("GROUNDED"), _candidate("UNGROUNDED")], source, None)
        self.assertEqual([c.code for c in result], ["GROUNDED"])

    def test_a_candidate_with_alternative_targets_grounds_via_either_one(self):
        """Codex F9-R2-C, third pass: a candidate descriptor naming "structure alpha
        or structure beta" legitimately applies to EITHER target. A fact documenting
        only beta must still ground it -- not leave it UNKNOWN (and so exposed to
        dominance exclusion) merely because alpha, the OTHER alternative, was not
        documented."""
        source = MockSource(
            records={
                ("EITHER", "cpt"): {
                    "long_description": "Excision, structure alpha or structure beta",
                    "active": True},
                ("UNGROUNDED", "cpt"): {"long_description": "Excision, unrelated site",
                                        "active": True}},
            concept_relation={("structure beta", "structure beta"): "same"})
        fact = ClinicalFact(FactKind.PROCEDURE, "a procedure",
                            attributes={"anatomy": "structure beta"})
        result = semelig.eligible_partition(
            [fact], [_candidate("EITHER"), _candidate("UNGROUNDED")], source, None)
        self.assertEqual([c.code for c in result], ["EITHER"])

    def test_with_or_without_idiom_is_not_mis_split_into_alternative_targets(self):
        """Codex F9-R2-C, fourth pass, exact reproduction: "Action, structure
        gamma, with or without qualifier delta" must yield ONLY "structure gamma"
        as a target -- "qualifier delta" is a qualifier CLAUSE, dropped wholesale,
        never promoted into a second, false target."""
        feats = semelig._ontology.parse_descriptor(
            "Excision, structure gamma, with or without qualifier delta")
        targets = semelig._candidate_anatomy_targets(feats)
        self.assertEqual(targets, ("structure gamma",), targets)

    def test_qualifier_clause_cannot_ground_a_candidate_or_dominate_a_sibling(self):
        """Codex F9-R2-C, fourth pass, claim-level reproduction: even when a
        governed relation exists for the QUALIFIER phrase itself, it must never
        be read as anatomy -- it must not ground the candidate that carries it,
        and must not let dominance remove an otherwise-unknown sibling."""
        source = MockSource(
            records={
                ("QUALIFIED", "cpt"): {
                    "long_description":
                        "Excision, structure alpha, with or without qualifier beta",
                    "active": True},
                ("UNGROUNDED", "cpt"): {"long_description": "Excision, unrelated site",
                                        "active": True}},
            # A governed relation exists for the QUALIFIER phrase, not the real
            # target -- if the qualifier were wrongly read as anatomy, this would
            # ground QUALIFIED and let dominance remove UNGROUNDED.
            concept_relation={("qualifier beta", "qualifier beta"): "same"})
        fact = ClinicalFact(FactKind.PROCEDURE, "a procedure",
                            attributes={"anatomy": "qualifier beta"})
        result = semelig.eligible_partition(
            [fact], [_candidate("QUALIFIED"), _candidate("UNGROUNDED")], source, None)
        # Neither candidate is dominance-excluded: QUALIFIED's real target
        # ("structure alpha") was never documented, and the qualifier clause must
        # not be read as if it grounded QUALIFIED's anatomy.
        self.assertEqual({c.code for c in result}, {"QUALIFIED", "UNGROUNDED"})


def _service_role_fact(role: str | None, description="a procedure") -> ClinicalFact:
    """A PROCEDURE fact with a claim-authorized "service_role" attribute, exactly
    the shape `_fact_attribute_value`/`claim_authorized_value` requires (scope-
    valid, source-reconciled, ASSERTED) -- same construction the existing
    laterality-contradiction test above uses. `role=None` builds a fact that never
    states one at all."""
    if role is None:
        return ClinicalFact(FactKind.PROCEDURE, description)
    from claude_coder.models import AttributeEvidence, EvidenceSpan, RelationState
    span = EvidenceSpan(f"performed under {role}", anchored=True, span_id="s1")
    return ClinicalFact(
        FactKind.PROCEDURE, description, attributes={"service_role": role},
        evidence=[span],
        attribute_evidence={"service_role": (
            AttributeEvidence(span=span, assertion_state=RelationState.ASSERTED,
                              value=role),)})


class ServiceRoleExclusion(unittest.TestCase):
    """issue #6 F9-R11-H: a documented procedure service_role (operative vs.
    anesthesia -- the one FactKind.PROCEDURE sub-distinction with no other
    structural signal) must positively exclude an authoritative candidate
    classified as the OTHER role, BEFORE verification ever sees it -- broad
    recall must not itself define the clinically eligible set."""

    def _source(self):
        # Codex F9-R11-H-B: candidate role now comes ONLY from semantic_class()
        # (never a descriptor-phrase proxy), so both roles are stubbed via
        # MockSource's semantic_class map, exactly like every other
        # semantic_class-driven test in this file.
        return MockSource(
            records={
                ("OP", "cpt"): {"long_description": "Ostectomy, calcaneus",
                                "active": True},
                ("ANES", "cpt"): {"long_description": "Anesthesia for procedures "
                                                       "on the calcaneus",
                                  "active": True},
                ("PLAIN", "cpt"): {"long_description": "Excision, unrelated site",
                                   "active": True},
            },
            semantic_class={"OP": "surgical_procedure", "ANES": "anesthesia"})

    def test_operative_fact_excludes_the_anesthesia_classified_candidate(self):
        fact = _service_role_fact("operative")
        result = semelig.eligible_partition(
            [fact], [_candidate("OP"), _candidate("ANES")], self._source(), "2026-01-01")
        self.assertEqual([c.code for c in result], ["OP"])

    def test_anesthesia_fact_excludes_the_operative_classified_candidate(self):
        fact = _service_role_fact("anesthesia")
        result = semelig.eligible_partition(
            [fact], [_candidate("OP"), _candidate("ANES")], self._source(), "2026-01-01")
        self.assertEqual([c.code for c in result], ["ANES"])

    def test_fact_with_no_service_role_excludes_nothing_on_this_basis(self):
        """Absence of the axis is a data gap, never evidence against either
        candidate -- both survive the role check (though not necessarily every
        other eligibility check)."""
        fact = _service_role_fact(None)
        result = semelig.eligible_partition(
            [fact], [_candidate("OP"), _candidate("ANES")], self._source(), "2026-01-01")
        self.assertEqual({c.code for c in result}, {"OP", "ANES"})

    def test_candidate_with_unclassifiable_role_is_not_excluded(self):
        """An operative fact does not exclude a candidate whose OWN role cannot be
        classified from already-authoritative structure -- absence of grounding
        on the candidate side is not evidence against it either."""
        fact = _service_role_fact("operative")
        result = semelig.eligible_partition(
            [fact], [_candidate("OP"), _candidate("PLAIN")], self._source(), "2026-01-01")
        self.assertEqual({c.code for c in result}, {"OP", "PLAIN"})

    def test_mixed_fact_kinds_never_check_service_role(self):
        """A multi-member intent (e.g. a procedure plus its diagnosis) has no
        single procedure role to check candidates against -- the check does not
        fire at all rather than picking one fact's role arbitrarily."""
        proc = _service_role_fact("operative")
        dx = ClinicalFact(FactKind.DIAGNOSIS, "a condition", fact_id="D1")
        result = semelig.eligible_partition(
            [proc, dx], [_candidate("OP"), _candidate("ANES")], self._source(),
            "2026-01-01")
        self.assertEqual({c.code for c in result}, {"OP", "ANES"})

    def test_eligibility_report_reflects_the_service_role_exclusion(self):
        fact = _service_role_fact("operative")
        candidates = [_candidate("OP"), _candidate("ANES")]
        kept = {c.code for c in
                semelig.eligible_partition([fact], candidates, self._source(), "2026-01-01")}
        report = {r["code"]: r for r in
                 semelig.eligibility_report([fact], candidates, self._source(), "2026-01-01")}
        for code, entry in report.items():
            self.assertEqual(entry["eligible"], code in kept, report)

    # ---------------------------------------------------- Codex F9-R11-H-A
    def test_conflicting_roles_in_a_composed_intent_exclude_neither_candidate(self):
        """A composed intent (e.g. a PART_OF-linked operative component and
        anesthesia component) with genuinely CONFLICTING documented roles must
        not pick one arbitrarily -- neither candidate family is excluded on
        this basis; a real conflict needs upstream composition/provider-query
        resolution, not a guess."""
        op_fact = _service_role_fact("operative", description="operative component")
        anes_fact = _service_role_fact("anesthesia", description="anesthesia component")
        result = semelig.eligible_partition(
            [op_fact, anes_fact], [_candidate("OP"), _candidate("ANES")],
            self._source(), "2026-01-01")
        self.assertEqual({c.code for c in result}, {"OP", "ANES"})

    def test_conflicting_roles_are_order_invariant(self):
        """The exact defect Codex's independent reproduction found: fact ORDER
        must never change which candidate family survives."""
        op_fact = _service_role_fact("operative", description="operative component")
        anes_fact = _service_role_fact("anesthesia", description="anesthesia component")
        forward = semelig.eligible_partition(
            [op_fact, anes_fact], [_candidate("OP"), _candidate("ANES")],
            self._source(), "2026-01-01")
        backward = semelig.eligible_partition(
            [anes_fact, op_fact], [_candidate("OP"), _candidate("ANES")],
            self._source(), "2026-01-01")
        self.assertEqual({c.code for c in forward}, {c.code for c in backward})

    def test_one_known_role_plus_one_missing_role_still_enforces_the_known_one(self):
        """A composed intent where only ONE member documents a role (the other
        is silent, not conflicting) still has exactly one AUTHORIZED role --
        the known one enforces normally. Missing is not the same as
        conflicting."""
        op_fact = _service_role_fact("operative", description="operative component")
        silent = ClinicalFact(FactKind.PROCEDURE, "an unrelated component",
                              fact_id="F-silent")
        result = semelig.eligible_partition(
            [op_fact, silent], [_candidate("OP"), _candidate("ANES")],
            self._source(), "2026-01-01")
        self.assertEqual([c.code for c in result], ["OP"])

    def test_three_component_intent_with_one_conflicting_role_excludes_neither(self):
        """A third, agreeing component does not let two-out-of-three roles
        "win" -- ANY genuine conflict in the set excludes nothing, not just a
        strict pairwise tie."""
        op1 = _service_role_fact("operative", description="component one")
        op2 = _service_role_fact("operative", description="component two")
        anes = _service_role_fact("anesthesia", description="component three")
        result = semelig.eligible_partition(
            [op1, op2, anes], [_candidate("OP"), _candidate("ANES")],
            self._source(), "2026-01-01")
        self.assertEqual({c.code for c in result}, {"OP", "ANES"})

    # ---------------------------------------------------- Codex F9-R11-H-D
    def test_role_control_is_never_silently_absent_from_the_report(self):
        """The exact defect: `eligibility_report` must carry a typed,
        non-None role_control decision for EVERY candidate, in every one of
        the control's five distinct non-run/run states -- never a bare
        omission a reader could confuse with "forgot to check". This is the
        end-to-end F5-shaped regression Codex required: it must FAIL if the
        control silently skips (i.e. if role_control were ever missing or a
        placeholder null)."""
        cases = [
            # (facts, expected top-level status for the OP candidate)
            ([_service_role_fact("operative")], "compatible"),
            ([_service_role_fact("anesthesia")], "excluded"),
            ([_service_role_fact(None)], "fact_role_missing"),
            ([_service_role_fact("operative", description="c1"),
             _service_role_fact("anesthesia", description="c2")], "fact_role_conflict"),
            ([_service_role_fact("operative"),
             ClinicalFact(FactKind.DIAGNOSIS, "a condition", fact_id="D1")],
             "mixed_kind_intent"),
        ]
        for facts, expected_status in cases:
            report = {r["code"]: r for r in semelig.eligibility_report(
                facts, [_candidate("OP")], self._source(), "2026-01-01")}
            rc = report["OP"]["role_control"]
            self.assertIsNotNone(rc, f"role_control silently absent for {expected_status}")
            self.assertEqual(rc["status"], expected_status)
            self.assertTrue(rc["source_id"] or expected_status.startswith(
                ("fact_role", "mixed_kind")), rc)

    def test_role_control_reports_an_unclassifiable_candidate_explicitly(self):
        fact = _service_role_fact("operative")
        report = {r["code"]: r for r in semelig.eligibility_report(
            [fact], [_candidate("PLAIN")], self._source(), "2026-01-01")}
        rc = report["PLAIN"]["role_control"]
        self.assertEqual(rc["status"], "candidate_unclassified")
        self.assertIsNone(rc["candidate_role"])
        self.assertEqual(rc["fact_roles"], ["operative"])


if __name__ == "__main__":
    unittest.main()
