"""Tests for claude_coder.composition (issue #6 items 2/3).

Deliberately agnostic per the plan's own test philosophy: every note fixture uses
generic, non-clinical placeholder text ("SECTION A", "did something to the first
thing") -- the module under test derives structure from document FORMATTING alone,
so nothing here needs a real clinical term to exercise it honestly.
"""
import unittest

from claude_coder import composition
from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                 RelationAssertion, RelationPredicate, RelationState)


def _fact(fact_id, text, start, end, reading_channel_id=None):
    return ClinicalFact(
        FactKind.PROCEDURE, text, fact_id=fact_id,
        evidence=[EvidenceSpan(text=text, start=start, end=end, anchored=True,
                               span_id=f"span-{fact_id}",
                               reading_channel_id=reading_channel_id)])


class SectionParsing(unittest.TestCase):
    def test_standalone_uppercase_line_is_a_header(self):
        note = "SECTION A\nsome body text here.\n\nSECTION B\nmore body text.\n"
        sections = composition._sections(note)
        headers = [h for h, _, _ in sections]
        self.assertEqual(headers, ["SECTION A", "SECTION B"])

    def test_inline_label_value_line_is_not_a_header(self):
        note = "DATE OF SERVICE: 1/1/2026\nSECTION A\nbody text.\n"
        sections = composition._sections(note)
        headers = [h for h, _, _ in sections]
        self.assertNotIn("DATE OF SERVICE", headers)
        self.assertIn("SECTION A", headers)

    def test_text_before_any_header_is_unheaded(self):
        note = "Facility name on its own line, mixed Case.\nSECTION A\nbody.\n"
        sections = composition._sections(note)
        self.assertIsNone(sections[0][0])

    def test_repeated_header_text_stays_two_distinct_segments(self):
        note = "SECTION A\nfirst body.\nSECTION A\nsecond body.\n"
        sections = composition._sections(note)
        headers = [h for h, _, _ in sections]
        self.assertEqual(headers, ["SECTION A", "SECTION A"])
        # different offset ranges even though the header text repeats
        self.assertNotEqual(sections[0][1:], sections[1][1:])


class ComposeEmitsSameEpisodeAs(unittest.TestCase):
    def setUp(self):
        self.note = "SECTION A\ndid something to the first thing and the second thing.\nSECTION B\ndid something unrelated to a third thing.\n"

    def test_two_facts_in_the_same_section_get_same_episode_as(self):
        header_a_start = self.note.index("SECTION A")
        body_a = self.note.index("did something to")
        facts = [_fact("F1", "first thing", body_a, body_a + 5),
                 _fact("F2", "second thing", body_a + 10, body_a + 15)]
        rels = composition.compose(facts, self.note)
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].predicate, RelationPredicate.SAME_EPISODE_AS)
        self.assertEqual({rels[0].subject_event_id, rels[0].object_event_id}, {"F1", "F2"})
        self.assertEqual(rels[0].evidence_span_ids, ["event:F1", "event:F2"])
        self.assertEqual(rels[0].extraction_source, "structural_composition_v1")

    def test_facts_in_different_sections_get_no_relation(self):
        body_a = self.note.index("did something to")
        body_b = self.note.index("did something unrelated")
        facts = [_fact("F1", "first thing", body_a, body_a + 5),
                 _fact("F2", "third thing", body_b, body_b + 5)]
        rels = composition.compose(facts, self.note)
        self.assertEqual(rels, [])

    def test_unanchored_fact_contributes_no_relation(self):
        body_a = self.note.index("did something to")
        facts = [_fact("F1", "first thing", body_a, body_a + 5),
                 ClinicalFact(FactKind.PROCEDURE, "second thing", fact_id="F2",
                             evidence=[EvidenceSpan(text="second thing", anchored=False)])]
        rels = composition.compose(facts, self.note)
        self.assertEqual(rels, [])

    def test_fact_anchored_only_in_a_second_reading_channel_is_excluded(self):
        body_a = self.note.index("did something to")
        facts = [_fact("F1", "first thing", body_a, body_a + 5),
                 _fact("F2", "second thing", body_a + 10, body_a + 15,
                      reading_channel_id="recall-1")]
        rels = composition.compose(facts, self.note)
        self.assertEqual(rels, [])

    def test_unheaded_preamble_text_is_never_grouped(self):
        note = "Facility name line.\nplain text here with no heading at all.\n"
        offset = note.index("plain text")
        facts = [_fact("F1", "a", offset, offset + 1),
                 _fact("F2", "b", offset + 2, offset + 3)]
        rels = composition.compose(facts, note)
        self.assertEqual(rels, [])

    def test_three_facts_in_one_section_get_all_pairwise_edges(self):
        note = "SECTION A\none two three all in one place.\n"
        base = note.index("one two three")
        facts = [_fact("F1", "one", base, base + 3),
                 _fact("F2", "two", base + 4, base + 7),
                 _fact("F3", "three", base + 8, base + 13)]
        rels = composition.compose(facts, note)
        self.assertEqual(len(rels), 3)
        pairs = {frozenset((r.subject_event_id, r.object_event_id)) for r in rels}
        self.assertEqual(pairs, {frozenset(("F1", "F2")), frozenset(("F1", "F3")),
                                 frozenset(("F2", "F3"))})


class ServiceIntentsReachability(unittest.TestCase):
    def test_two_facts_joined_by_part_of_form_one_intent(self):
        facts = [ClinicalFact(FactKind.PROCEDURE, "a", fact_id="F1"),
                 ClinicalFact(FactKind.PROCEDURE, "b", fact_id="F2")]
        rels = [RelationAssertion(subject_event_id="F1", predicate=RelationPredicate.PART_OF,
                                  object_event_id="F2", state=RelationState.ASSERTED,
                                  evidence_span_ids=["s1"])]
        intents = composition.service_intents(facts, rels)
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].component_event_ids, ["F1", "F2"])

    def test_same_episode_as_alone_never_composes_one_intent(self):
        """Codex F8-R4: SAME_EPISODE_AS is session/episode membership, not service
        composition -- two independently reportable events documented in the same
        note section must stay two separate intents."""
        facts = [ClinicalFact(FactKind.PROCEDURE, "a", fact_id="F1"),
                 ClinicalFact(FactKind.PROCEDURE, "b", fact_id="F2")]
        rels = [RelationAssertion(subject_event_id="F1", predicate=RelationPredicate.SAME_EPISODE_AS,
                                  object_event_id="F2", state=RelationState.ASSERTED,
                                  evidence_span_ids=["s1"])]
        intents = composition.service_intents(facts, rels)
        self.assertEqual(len(intents), 2)
        self.assertEqual(sorted(i.component_event_ids[0] for i in intents), ["F1", "F2"])

    def test_unlinked_fact_is_its_own_singleton_intent(self):
        facts = [ClinicalFact(FactKind.PROCEDURE, "a", fact_id="F1"),
                 ClinicalFact(FactKind.PROCEDURE, "b", fact_id="F2")]
        intents = composition.service_intents(facts, [])
        self.assertEqual(len(intents), 2)
        self.assertEqual(sorted(i.component_event_ids[0] for i in intents), ["F1", "F2"])

    def test_negated_relation_never_joins_two_events(self):
        facts = [ClinicalFact(FactKind.PROCEDURE, "a", fact_id="F1"),
                 ClinicalFact(FactKind.PROCEDURE, "b", fact_id="F2")]
        rels = [RelationAssertion(subject_event_id="F1", predicate=RelationPredicate.PART_OF,
                                  object_event_id="F2", state=RelationState.NEGATED,
                                  evidence_span_ids=["s1"])]
        intents = composition.service_intents(facts, rels)
        self.assertEqual(len(intents), 2)

    def test_uncertain_relation_never_joins_two_events(self):
        facts = [ClinicalFact(FactKind.PROCEDURE, "a", fact_id="F1"),
                 ClinicalFact(FactKind.PROCEDURE, "b", fact_id="F2")]
        rels = [RelationAssertion(subject_event_id="F1", predicate=RelationPredicate.PART_OF,
                                  object_event_id="F2", state=RelationState.UNCERTAIN,
                                  evidence_span_ids=["s1"])]
        intents = composition.service_intents(facts, rels)
        self.assertEqual(len(intents), 2)

    def test_part_of_chain_transitively_joins_three_events(self):
        facts = [ClinicalFact(FactKind.PROCEDURE, "a", fact_id="F1"),
                 ClinicalFact(FactKind.PROCEDURE, "b", fact_id="F2"),
                 ClinicalFact(FactKind.PROCEDURE, "c", fact_id="F3")]
        rels = [RelationAssertion(subject_event_id="F1", predicate=RelationPredicate.PART_OF,
                                  object_event_id="F2", state=RelationState.ASSERTED,
                                  evidence_span_ids=["s1"]),
                RelationAssertion(subject_event_id="F2", predicate=RelationPredicate.PART_OF,
                                  object_event_id="F3", state=RelationState.ASSERTED,
                                  evidence_span_ids=["s2"])]
        intents = composition.service_intents(facts, rels)
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].component_event_ids, ["F1", "F2", "F3"])

    def test_a_predicate_outside_the_composing_set_never_joins_events(self):
        facts = [ClinicalFact(FactKind.PROCEDURE, "a", fact_id="F1"),
                 ClinicalFact(FactKind.PROCEDURE, "b", fact_id="F2")]
        rels = [RelationAssertion(subject_event_id="F1", predicate=RelationPredicate.SEPARATE_FROM,
                                  object_event_id="F2", state=RelationState.ASSERTED,
                                  evidence_span_ids=["s1"])]
        intents = composition.service_intents(facts, rels)
        self.assertEqual(len(intents), 2)

    def test_intent_id_is_stable_for_the_same_membership(self):
        facts = [ClinicalFact(FactKind.PROCEDURE, "a", fact_id="F1"),
                 ClinicalFact(FactKind.PROCEDURE, "b", fact_id="F2")]
        rels = [RelationAssertion(subject_event_id="F1", predicate=RelationPredicate.PART_OF,
                                  object_event_id="F2", state=RelationState.ASSERTED,
                                  evidence_span_ids=["s1"])]
        first = composition.service_intents(facts, rels)[0].intent_id
        second = composition.service_intents(list(reversed(facts)), rels)[0].intent_id
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
