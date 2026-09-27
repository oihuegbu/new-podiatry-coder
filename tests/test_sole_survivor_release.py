"""Single-evaluator settlement of a sole survivor (`resolution._settle_uniqueness`).

With one evaluator the two-matrix disposition pass never runs, which had left two
release branches dead: (a) the evaluator proposed a retrieval-only candidate the
baseline floor then removed, leaving the SUPPORTED rival it had ALSO entailed as the
sole survivor -- held as a one-candidate "tie" (third live run); (b) no proposal at
all, sole entailed SUPPORTED survivor. Both now release; a bare (undeclared) verdict
or an ungrounded survivor still does not. Synthetic throughout.
"""
import json
import unittest

from claude_coder import verify as _verify
from claude_coder.data_access import MockSource
from claude_coder.models import (AttributeEvidence, CandidateCode, ClinicalFact,
                                 EvidenceSpan, FactKind, RelationState)
from claude_coder.resolution import resolve
from tests import shortlist_verdict as _sv
from tests.test_claude_coder import _request

SUPPORTED = "X20.7"     # governed (crosswalk) identity; its distinctive words are in the record
RETRIEVAL = "X10.1"     # retrieval only; its distinctive words are absent from the record
DOCUMENTED = "eponym disorder of hard tissue of the right alpha part"
DESC = {SUPPORTED: "Other specified disorders of hard tissue, right alpha part",
        RETRIEVAL: "Other acquired contour disorders, right alpha part"}   # identity clause = the condition


def _fact():
    span = EvidenceSpan(DOCUMENTED, anchored=True, span_id="s1")
    return ClinicalFact(kind=FactKind.DIAGNOSIS, description=DOCUMENTED,
                        attributes={"laterality": "right"}, evidence=[span], fact_id="F1",
                        confidence=0.95,
                        attribute_evidence={"laterality": (AttributeEvidence(
                            span=span, scope="local", assertion_state=RelationState.ASSERTED,
                            value="right"),)})


def _source():
    return MockSource(
        records={(c, "icd10"): {"long_description": d, "active": True} for c, d in DESC.items()},
        retrieval={("*", "icd10"): [CandidateCode(RETRIEVAL, "icd10", DESC[RETRIEVAL], 0.9)]},
        snomed={DOCUMENTED: {SUPPORTED}})


def _pinned(fn):
    return _verify.declare_model_profile(fn, provider="provider-a")


def _agreed(*span_ids):
    """The original page confirmed these quotations -- the baseline floor reads only
    the fact's OWN reconciled evidence."""
    from app.contracts.source_evidence import (ReconciliationStatus, SourceReconciliation,
                                               SpanReconciliation)
    return SourceReconciliation(spans=tuple(
        SpanReconciliation(span_id=s, status=ReconciliationStatus.AGREED, pages=(1,))
        for s in span_ids))


def _no_proposal_judge():
    """Declared verdict: BOTH entailed, no choice."""
    def stub(system, user):
        if "propose" in system.lower():
            return json.dumps({"codes": []})
        opts = _sv.options(user)
        return json.dumps({"choice": 0, "entailed": [n for n, _ in opts],
                           "eliminated": [], "reason": "both entailed, no pick"})
    return stub


class SoleSurvivorReleaseTest(unittest.TestCase):

    def test_a_floor_eliminated_proposal_yields_to_the_sole_entailed_supported_survivor(self):
        judge = _sv.judge(entails=lambda d: True, prefer=lambda d: "contour" in d,
                          reason="both entailed")
        line = resolve(_request(_fact()), _source(), llm=_pinned(judge), reconciliation=_agreed("s1"))
        self.assertIsNotNone(line.chosen, line.rationale)
        self.assertEqual(line.chosen.code, SUPPORTED)
        self.assertTrue(line.tie_record.get("reselected_from_verified_survivor"), line.tie_record)
        self.assertEqual(line.tie_record.get("proposed"), RETRIEVAL)
        self.assertIn(RETRIEVAL, line.tie_record["eliminated"])

    def test_no_proposal_with_a_sole_entailed_supported_survivor_releases(self):
        line = resolve(_request(_fact()), _source(), llm=_pinned(_no_proposal_judge()), reconciliation=_agreed("s1"))
        self.assertIsNotNone(line.chosen, line.rationale)
        self.assertEqual(line.chosen.code, SUPPORTED)
        self.assertTrue(line.tie_record.get("selected_sole_entailed_survivor"), line.tie_record)

    def test_an_undeclared_bare_pick_still_does_not_release(self):
        judge = _sv.judge(entails=lambda d: True, prefer=lambda d: "contour" in d,
                          declare=False, reason="bare pick")
        line = resolve(_request(_fact()), _source(), llm=_pinned(judge), reconciliation=_agreed("s1"))
        self.assertIsNone(line.chosen, line.rationale)

    def test_a_survivor_the_evaluator_did_not_entail_does_not_release(self):
        judge = _sv.judge(entails=lambda d: "contour" in d, reason="only the retrieval one")
        line = resolve(_request(_fact()), _source(), llm=_pinned(judge), reconciliation=_agreed("s1"))
        self.assertNotEqual(getattr(line.chosen, "code", None), SUPPORTED, line.rationale)


if __name__ == "__main__":
    unittest.main()
