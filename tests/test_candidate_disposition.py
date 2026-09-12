"""The candidate-level SEMANTIC disposition record (issue #6, Codex's independent
re-review, F9-R15-B) -- the replacement for the reverted `requirement.
_descriptor_term_requirements`, which promoted raw descriptor tokens into
literal-text MUST_SUPPORT requirements and was proven unsafe (a synonym/
paraphrase could make a literal token correctly absent while the actual
requirement was fully documented, grounding a false elimination). Covers both
halves: `verify._candidate_dispositions` (fail-closed parsing, never trusting a
clause the model didn't reproduce verbatim) and `resolution.
_candidate_disposition_uniqueness` (the elimination bar itself). Synthetic
descriptors/codes throughout.
"""
import unittest

from claude_coder import resolution as res
from claude_coder import verify as _verify
from claude_coder.models import CandidateCode, ClinicalFact, FactKind


def _cand(code, descriptor):
    return CandidateCode(code=code, system="cpt", descriptor=descriptor, score=0.9,
                         source="retrieval")


ALPHA = _cand("CAND_ALPHA", "assembly service, alpha technique")
BETA = _cand("CAND_BETA", "assembly service, beta technique")
GAMMA = _cand("CAND_GAMMA", "assembly service, gamma technique")

_DESCRIPTORS = {"CAND_ALPHA": ALPHA.descriptor, "CAND_BETA": BETA.descriptor,
               "CAND_GAMMA": GAMMA.descriptor}


def _reconciliation(statuses):
    from app.contracts.source_evidence import (ReconciliationStatus,
                                                SourceReconciliation,
                                                SpanReconciliation)
    return SourceReconciliation(spans=tuple(
        SpanReconciliation(span_id=sid, status=ReconciliationStatus[status])
        for sid, status in statuses.items()))


class _Coverage:
    def __init__(self, complete):
        self.complete = complete


class _SourceStub:
    def descriptions(self, code, system):
        return []


def _judgement(dispositions):
    return _verify.Judgement(candidate_dispositions=tuple(dispositions))


def _disp(code, status, clause, span_ids=(), offset=None):
    """Offset computed from the real descriptor by default -- never hand-
    counted, which is exactly the kind of transcription error this
    mechanism's own verbatim-reproduction check exists to catch. Pass an
    explicit `offset` only to construct a deliberately mismatched entry."""
    if offset is None:
        descriptor = _DESCRIPTORS[code]
        idx = descriptor.find(clause)
        assert idx >= 0, f"{clause!r} not found in {descriptor!r}"
        offset = (idx, idx + len(clause))
    return _verify.CandidateDispositionEvidence(
        candidate_code=code, status=status, authority_clause=clause,
        authority_offset=offset, evidence_span_ids=span_ids)


class ParsingTest(unittest.TestCase):
    """`verify._candidate_dispositions` -- fail-closed at every branch."""

    def test_a_well_formed_entry_parses(self):
        ans = {"candidate_dispositions": [
            {"option": 1, "status": "entailed", "authority_clause": "alpha technique",
             "span_ids": ["e1"]}]}
        out = _verify._candidate_dispositions(ans, [ALPHA, BETA], {"e1": "s0"}, {})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].candidate_code, "CAND_ALPHA")
        self.assertEqual(out[0].status, "entailed")
        self.assertEqual(out[0].evidence_span_ids, ("s0",))

    def test_a_clause_not_verbatim_in_the_real_descriptor_is_dropped(self):
        """The model cannot author the clause it is judged against."""
        ans = {"candidate_dispositions": [
            {"option": 1, "status": "entailed", "authority_clause": "gamma technique"}]}
        out = _verify._candidate_dispositions(ans, [ALPHA, BETA], {}, {})
        self.assertEqual(out, ())

    def test_an_invalid_status_is_dropped(self):
        ans = {"candidate_dispositions": [
            {"option": 1, "status": "maybe", "authority_clause": "alpha technique"}]}
        self.assertEqual(_verify._candidate_dispositions(ans, [ALPHA, BETA], {}, {}), ())

    def test_an_out_of_range_option_is_dropped(self):
        ans = {"candidate_dispositions": [
            {"option": 99, "status": "entailed", "authority_clause": "alpha technique"}]}
        self.assertEqual(_verify._candidate_dispositions(ans, [ALPHA, BETA], {}, {}), ())

    def test_a_duplicate_option_keeps_only_the_first(self):
        ans = {"candidate_dispositions": [
            {"option": 1, "status": "entailed", "authority_clause": "alpha technique"},
            {"option": 1, "status": "not_documented", "authority_clause": "alpha technique"}]}
        out = _verify._candidate_dispositions(ans, [ALPHA, BETA], {}, {})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].status, "entailed")

    def test_a_cited_id_the_shortlist_never_showed_is_dropped_not_invented(self):
        ans = {"candidate_dispositions": [
            {"option": 1, "status": "entailed", "authority_clause": "alpha technique",
             "span_ids": ["made-up"]}]}
        out = _verify._candidate_dispositions(ans, [ALPHA, BETA], {"e1": "s0"}, {})
        self.assertEqual(out[0].evidence_span_ids, ())

    def test_no_candidate_dispositions_field_parses_to_empty(self):
        self.assertEqual(_verify._candidate_dispositions({}, [ALPHA, BETA], {}, {}), ())

    def test_contract_and_field_are_gated_on_two_or_more_candidates(self):
        """A single-candidate shortlist has nothing to disambiguate."""
        self.assertNotIn("candidate_dispositions", _verify._SELECT_SYSTEM)
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="x")
        prompt, _ = _verify._shortlist_prompt(fact, [ALPHA], _SourceStub(), ())
        self.assertNotIn("[e1]", prompt)


class EliminationTest(unittest.TestCase):
    """`resolution._candidate_disposition_uniqueness` -- the elimination bar."""

    def test_disagreement_between_evaluators_leaves_the_candidate_standing(self):
        j0 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "not_documented", "beta technique")])
        j1 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "entailed", "beta technique")])
        remaining, eliminated = res._candidate_disposition_uniqueness(
            [ALPHA, BETA], ALPHA, [j0, j1], None, _Coverage(True))
        self.assertIn(BETA, remaining)
        self.assertEqual(eliminated, {})

    def test_not_documented_without_complete_coverage_leaves_it_standing(self):
        j0 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "not_documented", "beta technique")])
        j1 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "not_documented", "beta technique")])
        remaining, eliminated = res._candidate_disposition_uniqueness(
            [ALPHA, BETA], ALPHA, [j0, j1], None, _Coverage(False))
        self.assertIn(BETA, remaining)
        self.assertEqual(eliminated, {})

    def test_not_documented_with_complete_coverage_and_both_evaluators_agreeing_eliminates(self):
        j0 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "not_documented", "beta technique")])
        j1 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "not_documented", "beta technique")])
        remaining, eliminated = res._candidate_disposition_uniqueness(
            [ALPHA, BETA], ALPHA, [j0, j1], None, _Coverage(True))
        self.assertNotIn(BETA, remaining)
        self.assertIn("CAND_BETA", eliminated)

    def test_contradicted_without_validated_spans_leaves_it_standing(self):
        """A bare claim of contradiction, with no cited evidence, is not enough."""
        j0 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "contradicted", "beta technique")])
        j1 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "contradicted", "beta technique")])
        remaining, eliminated = res._candidate_disposition_uniqueness(
            [ALPHA, BETA], ALPHA, [j0, j1], None, _Coverage(True))
        self.assertIn(BETA, remaining)

    def test_contradicted_with_validated_spans_on_both_sides_eliminates(self):
        recon = _reconciliation({"s1": "AGREED", "s2": "AGREED"})
        j0 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "contradicted", "beta technique",
                             span_ids=("s1",))])
        j1 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "contradicted", "beta technique",
                             span_ids=("s2",))])
        remaining, eliminated = res._candidate_disposition_uniqueness(
            [ALPHA, BETA], ALPHA, [j0, j1], recon, _Coverage(True))
        self.assertNotIn(BETA, remaining)
        self.assertIn("CAND_BETA", eliminated)

    def test_an_unreconciled_span_never_validates_a_contradiction(self):
        recon = _reconciliation({"s1": "DISAGREED", "s2": "AGREED"})
        j0 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "contradicted", "beta technique",
                             span_ids=("s1",))])
        j1 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "contradicted", "beta technique",
                             span_ids=("s2",))])
        remaining, _ = res._candidate_disposition_uniqueness(
            [ALPHA, BETA], ALPHA, [j0, j1], recon, _Coverage(True))
        self.assertIn(BETA, remaining)

    def test_the_chosen_candidate_is_never_eliminated_even_if_both_evaluators_call_it_out(self):
        """A structural safety guard: this mechanism exists to rule OTHER
        candidates out for an already-entailed `chosen`, never to re-litigate
        `chosen` itself -- eliminating it here (while a different candidate
        stayed standing) would let the caller's naive `len(remaining) == 1`
        check release `chosen` anyway."""
        recon = _reconciliation({"s1": "AGREED", "s2": "AGREED"})
        j0 = _judgement([_disp("CAND_ALPHA", "contradicted", "alpha technique",
                              span_ids=("s1",)),
                        _disp("CAND_BETA", "entailed", "beta technique")])
        j1 = _judgement([_disp("CAND_ALPHA", "contradicted", "alpha technique",
                              span_ids=("s2",)),
                        _disp("CAND_BETA", "entailed", "beta technique")])
        remaining, eliminated = res._candidate_disposition_uniqueness(
            [ALPHA, BETA], ALPHA, [j0, j1], recon, _Coverage(True))
        self.assertIn(ALPHA, remaining)
        self.assertNotIn("CAND_ALPHA", eliminated)

    def test_a_clause_that_does_not_reproduce_from_the_real_descriptor_leaves_it_standing(self):
        """Defense in depth against a stale/mismatched entry, independent of
        `verify._candidate_dispositions`'s own parse-time check."""
        tampered = _disp("CAND_BETA", "not_documented", "WRONG CLAUSE", offset=(0, 12))
        j0 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"), tampered])
        j1 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "not_documented", "beta technique")])
        remaining, _ = res._candidate_disposition_uniqueness(
            [ALPHA, BETA], ALPHA, [j0, j1], None, _Coverage(True))
        self.assertIn(BETA, remaining)

    def test_fewer_than_two_judgements_defers_entirely(self):
        j0 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique")])
        self.assertIsNone(res._candidate_disposition_uniqueness(
            [ALPHA], ALPHA, [j0], None, _Coverage(True)))

    def test_a_judgement_missing_an_entry_for_any_candidate_defers_entirely(self):
        j0 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique")])
        j1 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "not_documented", "beta technique")])
        self.assertIsNone(res._candidate_disposition_uniqueness(
            [ALPHA, BETA], ALPHA, [j0, j1], None, _Coverage(True)))

    def test_three_candidates_are_all_accounted_for(self):
        j0 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "not_documented", "beta technique"),
                        _disp("CAND_GAMMA", "not_documented", "gamma technique")])
        j1 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "not_documented", "beta technique"),
                        _disp("CAND_GAMMA", "not_documented", "gamma technique")])
        remaining, eliminated = res._candidate_disposition_uniqueness(
            [ALPHA, BETA, GAMMA], ALPHA, [j0, j1], None, _Coverage(True))
        self.assertEqual(remaining, [ALPHA])
        self.assertEqual(set(eliminated), {"CAND_BETA", "CAND_GAMMA"})

    def test_candidate_order_permutation_produces_the_same_result(self):
        j0 = _judgement([_disp("CAND_BETA", "not_documented", "beta technique"),
                        _disp("CAND_ALPHA", "entailed", "alpha technique")])
        j1 = _judgement([_disp("CAND_ALPHA", "entailed", "alpha technique"),
                        _disp("CAND_BETA", "not_documented", "beta technique")])
        remaining, eliminated = res._candidate_disposition_uniqueness(
            [BETA, ALPHA], ALPHA, [j0, j1], None, _Coverage(True))
        self.assertEqual({c.code for c in remaining}, {"CAND_ALPHA"})
        self.assertEqual(set(eliminated), {"CAND_BETA"})


if __name__ == "__main__":
    unittest.main()
