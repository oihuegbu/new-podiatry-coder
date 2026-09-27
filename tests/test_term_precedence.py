"""Governed term specificity precedence (product-owner decision 2026-09-25).

Two still-entailed candidates, EACH bound to the record's own wording by a
governed source, differ in WHAT their source terms name: one term names the
documented condition, the other states only a generic word and a site (or a
site and a side alone). The comparison reads the governed terms themselves,
with anatomy identified by the governed body-structure graph -- never a term
list in Python, never a model preference. Everything here is synthetic.
"""
import unittest

from claude_coder import terminology as term
from claude_coder.data_access import MockSource, governed_condition_content
from claude_coder.models import (AttributeEvidence, CandidateCode, ClinicalFact,
                                 EvidenceSpan, FactKind, RelationState)
from claude_coder.resolution import resolve
from tests import shortlist_verdict as _sv
from tests.test_claude_coder import _request

NAMED = "X20.7"        # its governed term names the condition ("eponym shape disorder")
GENERIC = "X10.1"      # its governed term states a generic word and a site ("shape disorder alpha part")
DOCUMENTED = "eponym shape disorder of the right alpha part"
ANATOMY = {"alpha part": {"term": "alpha part", "candidates": ["C1"], "method": "exact",
                          "unique": True, "expansions": []}}


def _identity(source_id):
    return {"source_id": source_id, "sha256": "sha256:" + "0" * 64, "size": 10}


def _index_match(code, *terms):
    return {"method": "contained_source_phrase", "normalized_query": DOCUMENTED,
            "source_terms": list(terms), "mapped_code": code,
            "source_identity": _identity("synthetic-index")}


def _fact():
    span = EvidenceSpan(DOCUMENTED, anchored=True, span_id="s1")
    return ClinicalFact(kind=FactKind.DIAGNOSIS, description=DOCUMENTED,
                        attributes={"laterality": "right"}, evidence=[span], fact_id="F1",
                        confidence=0.95,
                        attribute_evidence={"laterality": (AttributeEvidence(
                            span=span, scope="local", assertion_state=RelationState.ASSERTED,
                            value="right"),)})


def _source(*, generic_terms=("shape disorder alpha part",), anatomy=ANATOMY):
    return MockSource(
        records={(NAMED, "icd10"): {"long_description": "Other specified disorders of hard tissue, right alpha part",
                                    "active": True},
                 (GENERIC, "icd10"): {"long_description": "Other acquired shape disorders of right alpha part",
                                      "active": True}},
        index_recall={DOCUMENTED: {GENERIC: _index_match(GENERIC, *generic_terms)}},
        snomed={DOCUMENTED: {NAMED}},          # MockSource binds the whole wording as the term
        concept_lookup=dict(anatomy))


def _both_entailed():
    from claude_coder import verify as _verify
    judge = _sv.judge(entails=lambda _: True, reason="both entailed")
    return _verify.declare_model_profile(judge, provider="provider-a")


class GovernedTermPrecedenceTest(unittest.TestCase):

    def test_the_term_naming_the_condition_governs_over_a_generic_word_plus_site(self):
        line = resolve(_request(_fact()), _source(), llm=_both_entailed())
        self.assertIsNotNone(line.chosen, line.rationale)
        self.assertEqual(line.chosen.code, NAMED)
        rec = line.tie_record["governed_term_precedence"]
        self.assertEqual(rec["selected"], NAMED)
        self.assertIn(GENERIC, rec["eliminated"])
        self.assertEqual(rec["condition_content"][GENERIC], ["disorder", "shape"])
        self.assertEqual(rec["condition_content"][NAMED], ["disorder", "eponym", "shape"])
        self.assertIn("Section I.B.1", rec["authority"])
        self.assertIn("governed term precedence", line.tie_record["eliminated"][GENERIC])

    def test_two_terms_each_naming_something_the_other_does_not_stay_a_tie(self):
        """The record states "chronic"; one governed term names it and the other names
        the eponym -- each states a condition word the other lacks, so neither governs."""
        documented = "eponym chronic shape disorder of the right alpha part"
        src = MockSource(
            records={(NAMED, "icd10"): {"long_description": "Other specified disorders of hard tissue, right alpha part",
                                        "active": True},
                     (GENERIC, "icd10"): {"long_description": "Other acquired shape disorders of right alpha part",
                                          "active": True}},
            index_recall={documented: {GENERIC: {**_index_match(GENERIC, "chronic shape disorder alpha part"),
                                                 "normalized_query": documented}}},
            concept_lookup=dict(ANATOMY))
        src.snomed_code_matches = lambda description, system, *, laterality=None: {
            NAMED: {"method": "contained_source_phrase", "normalized_query": documented,
                    "source_terms": ["eponym shape disorder"], "mapped_code": NAMED,
                    "source_identity": _identity("synthetic-snomed-map")}}
        span = EvidenceSpan(documented, anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description=documented,
                            attributes={"laterality": "right"}, evidence=[span], fact_id="F1",
                            confidence=0.95,
                            attribute_evidence={"laterality": (AttributeEvidence(
                                span=span, scope="local", assertion_state=RelationState.ASSERTED,
                                value="right"),)})
        line = resolve(_request(fact), src, llm=_both_entailed())
        self.assertIsNone(line.chosen, line.rationale)
        self.assertEqual(sorted(line.tie_record["still_entailed"]), sorted([GENERIC, NAMED]))
        self.assertNotIn("governed_term_precedence", line.tie_record)

    def _two_index_terms(self, anatomy):
        """The condition-naming term bound through the crosswalk (the governed
        identity standing the winner must hold), the generic one through the Index."""
        src = MockSource(
            records={(NAMED, "icd10"): {"long_description": "Other specified disorders of hard tissue, right alpha part",
                                        "active": True},
                     (GENERIC, "icd10"): {"long_description": "Other acquired shape disorders of right alpha part",
                                          "active": True}},
            index_recall={DOCUMENTED: {GENERIC: _index_match(GENERIC, "shape disorder alpha part")}},
            concept_lookup=dict(anatomy))
        src.snomed_code_matches = lambda description, system, *, laterality=None: {
            NAMED: {"method": "contained_source_phrase", "normalized_query": DOCUMENTED,
                    "source_terms": ["eponym shape disorder"], "mapped_code": NAMED,
                    "source_identity": _identity("synthetic-snomed-map")}}
        return src

    def test_without_an_anatomy_graph_the_site_words_stay_and_the_tie_stays(self):
        """Fail-closed: with no governed anatomy scan the generic term's site words
        look like condition words, so its content is no longer contained in the
        condition-naming term's -- the same two terms that settle WITH the graph."""
        settled = resolve(_request(_fact()), self._two_index_terms(ANATOMY), llm=_both_entailed())
        self.assertEqual(settled.chosen.code, NAMED, settled.rationale)
        held = resolve(_request(_fact()), self._two_index_terms({}), llm=_both_entailed())
        self.assertIsNone(held.chosen, held.rationale)
        self.assertNotIn("governed_term_precedence", held.tie_record)
        self.assertEqual(sorted(held.tie_record["still_entailed"]), sorted([GENERIC, NAMED]))

    def test_a_site_and_side_only_term_loses_to_any_condition_naming_term(self):
        src = _source(generic_terms=("alpha part right",))
        line = resolve(_request(_fact()), src, llm=_both_entailed())
        self.assertEqual(line.chosen.code, NAMED, line.rationale)
        self.assertEqual(line.tie_record["governed_term_precedence"]["condition_content"][GENERIC], [])

    def test_only_the_words_the_record_states_count(self):
        """A term matched on one distinctive token ("eponym" out of "eponym malady or
        osteosis") counts for that token alone; the term the wording contains whole
        ("eponym shape disorder") names more of what the record states and governs."""
        src = self._two_index_terms(ANATOMY)
        src._index_recall = {DOCUMENTED: {GENERIC: {
            "method": "distinctive_source_token", "normalized_query": DOCUMENTED,
            "matched_tokens": ["eponym"], "source_terms": ["eponym malady or osteosis"],
            "mapped_code": GENERIC, "source_identity": _identity("synthetic-index")}}}
        line = resolve(_request(_fact()), src, llm=_both_entailed())
        self.assertEqual(line.chosen.code, NAMED, line.rationale)
        rec = line.tie_record["governed_term_precedence"]
        self.assertEqual(rec["condition_content"][GENERIC], ["eponym"])
        self.assertEqual(rec["condition_content"][NAMED], ["disorder", "eponym", "shape"])

    def test_a_retrieval_only_rival_is_not_this_steps_business(self):
        """The baseline floor (Codex F8-R1 scoping, deliberately unchanged) owns a
        governed-vs-retrieval tie; precedence compares governed terms only."""
        src = MockSource(
            records={(NAMED, "icd10"): {"long_description": "Other specified disorders of hard tissue, right alpha part",
                                        "active": True},
                     (GENERIC, "icd10"): {"long_description": "Other acquired shape disorders of right alpha part",
                                          "active": True}},
            retrieval={("*", "icd10"): [CandidateCode(GENERIC, "icd10",
                                                       "Other acquired shape disorders of right alpha part", 0.9)]},
            snomed={DOCUMENTED: {NAMED}},
            concept_lookup=dict(ANATOMY))
        line = resolve(_request(_fact()), src, llm=_both_entailed())
        self.assertNotIn("governed_term_precedence", line.tie_record or {})


class ConditionContentTest(unittest.TestCase):

    def test_anatomy_laterality_and_scaffold_words_are_not_condition_content(self):
        scan = lambda text: ("alpha part",) if "alpha part" in text else ()   # noqa: E731
        self.assertEqual(term.condition_content(["alpha part right"], scan), set())
        self.assertEqual(term.condition_content(["shape disorders of the left alpha part"], scan),
                         {"shape", "disorder"})
        self.assertEqual(term.condition_content(["structure of alpha part"], scan), set())

    def test_the_source_helper_uses_the_sources_anatomy_scan(self):
        src = MockSource(concept_lookup=dict(ANATOMY))
        self.assertEqual(governed_condition_content(src, ["eponym disorder alpha part"]),
                         {"eponym", "disorder"})
        self.assertEqual(governed_condition_content(object(), ["eponym disorder alpha part"]),
                         {"eponym", "disorder", "alpha"})     # no scan: site words stay ("part" is scaffold)


class IndexRecallGuardTest(unittest.TestCase):
    """`AuthoritativeSource.index_code_matches` drops a match whose source term
    names no condition (a site and a side alone)."""

    class _Stub:
        _idx_identity = {"source_id": "index_terms"}

        def __init__(self):
            self._index = term.TerminologyIndex({"X30.1": ["alpha part right"],
                                                 "X30.2": ["lesion alpha part right"]})

        def _terminology_index(self):
            return self._index

        def leaf_codes(self, code, system):
            return {code}

        def concept_scan(self, axis, text):
            return ("alpha part",) if axis == "anatomy" and "alpha part" in text else ()

    def test_a_site_and_side_term_is_not_recall_evidence(self):
        from claude_coder.data_access import AuthoritativeSource
        out = AuthoritativeSource.index_code_matches(
            self._Stub(), "lesion of the right alpha part documented", "icd10")
        self.assertNotIn("X30.1", out)
        self.assertIn("X30.2", out)
        self.assertEqual(out["X30.2"]["source_terms"], ["lesion alpha part right"])


class AnatomyScanTest(unittest.TestCase):

    def test_scan_phrases_reports_the_governed_windows(self):
        idx = term.ConceptRelationIndex({
            "A": {"terms": ["alpha part"], "parents": []},
            "B": {"terms": ["beta region"], "parents": []}})
        self.assertEqual(idx.scan_phrases("lesion of the alpha part and beta region"),
                         ("alpha part", "beta region"))
        self.assertEqual(idx.scan_phrases("alpha part"), ("alpha part",))
        self.assertEqual(idx.scan_phrases("nothing governed here"), ())


if __name__ == "__main__":
    unittest.main()
