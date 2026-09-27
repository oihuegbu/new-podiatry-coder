"""A judging model's verdict is read as ONE statement: a candidate its structured
`candidate_dispositions` mark "entailed" is entailed, so a same-verdict entry in the
eliminated list is a self-contradiction that eliminates nothing (fourth live run: the
crosswalk's code was disposition-entailed AND list-eliminated; the list alone was
believed and an evaluator-adjudicated elimination removed the governed candidate).
Also: a silent-record adjudication may never drop a candidate whose identity a
governed source established from the record's own wording in favor of a proposal
with no such identity. Synthetic throughout.
"""
import unittest

from claude_coder import resolution
from claude_coder.models import CandidateCode, ClinicalFact, EvidenceSpan, FactKind
from claude_coder.verify import CandidateDispositionEvidence, Judgement, _descriptor_sha256

GENERIC = CandidateCode("X10.1", "icd10", "Other acquired contour disorders, right alpha part", 0.9,
                        source="retrieval")
NAMED = CandidateCode("X20.7", "icd10", "Other specified disorders of hard tissue, right alpha part", 1.0,
                      source="snomed-crosswalk",
                      authority={"term_to_code_match": {
                          "method": "distinctive_source_token", "normalized_query": "eponym disorder",
                          "source_terms": ["eponym disorder"], "mapped_code": "X20.7",
                          "source_identity": {"source_id": "snomed_crosswalk",
                                              "sha256": "sha256:" + "0" * 64, "size": 10}}})
PLAIN = CandidateCode("X20.8", "icd10", "Other specified disorders of hard tissue, right alpha part", 0.8,
                      source="retrieval")


def _fact():
    span = EvidenceSpan("eponym disorder of the right alpha part", anchored=True, span_id="s1")
    return ClinicalFact(kind=FactKind.DIAGNOSIS, description="eponym disorder of the right alpha part",
                        attributes={"laterality": "right"}, evidence=[span], fact_id="F1", confidence=0.9)


def _agreed(*span_ids):
    from app.contracts.source_evidence import (ReconciliationStatus, SourceReconciliation,
                                               SpanReconciliation)
    return SourceReconciliation(spans=tuple(
        SpanReconciliation(span_id=s, status=ReconciliationStatus.AGREED, pages=(1,))
        for s in span_ids))


def _disposition(cand, status):
    return CandidateDispositionEvidence(candidate_code=cand.code,
                                        descriptor_sha256=_descriptor_sha256(cand), status=status,
                                        evidence_span_ids=("s1",), missing_fact="",
                                        evaluator_origin={"provider": "p"})


class VerdictSelfConsistencyTest(unittest.TestCase):

    def test_a_disposition_entailed_candidate_is_entailed_and_never_eliminated(self):
        j = Judgement(chosen=GENERIC, entailed=(GENERIC.code,),
                      eliminated={NAMED.code: "the hard tissue descriptor under-represents the condition"},
                      declared=True,
                      candidate_dispositions=(_disposition(GENERIC, "entailed"),
                                              _disposition(NAMED, "entailed")))
        self.assertTrue(j.entails(NAMED.code))
        self.assertEqual(j.elimination_of(NAMED.code), "")

    def test_a_disposition_that_does_not_entail_leaves_the_list_verdict_alone(self):
        j = Judgement(chosen=GENERIC, entailed=(GENERIC.code,),
                      eliminated={NAMED.code: "the hard tissue descriptor under-represents the condition"},
                      declared=True,
                      candidate_dispositions=(_disposition(NAMED, "different_concept"),))
        self.assertFalse(j.entails(NAMED.code))
        self.assertTrue(j.elimination_of(NAMED.code))


class GovernedIdentityAdjudicationGuardTest(unittest.TestCase):

    def _view(self, loser, winner=GENERIC):
        reason = "the specified hard tissue descriptor under-represents the documented condition"
        j = Judgement(chosen=winner, entailed=(winner.code,), eliminated={loser.code: reason},
                      declared=True)
        remaining, eliminated = resolution._uniqueness_view(
            _fact(), [winner, loser], winner, [j], {}, _agreed("s1"))
        return [c.code for c in remaining], eliminated

    def test_a_silent_record_adjudication_cannot_drop_a_governed_identity_for_retrieval(self):
        remaining, eliminated = self._view(NAMED)
        self.assertIn(NAMED.code, remaining, eliminated)
        self.assertNotIn(NAMED.code, eliminated)

    def test_the_same_adjudication_still_removes_a_retrieval_only_rival(self):
        """Unchanged 2026-09-22 behavior when neither side carries governed identity."""
        remaining, eliminated = self._view(PLAIN)
        self.assertNotIn(PLAIN.code, remaining)
        self.assertIn("evaluator-adjudicated", eliminated[PLAIN.code])


if __name__ == "__main__":
    unittest.main()
