"""issue #6, Codex's independent re-review (F9-R15-C): a fact-scoped BLOCKED/
ERROR gate must exclude only that fact (and anything genuinely entangled with
it), never the whole encounter. Before this fix, `autonomy.decide()`'s hard-
stop branch treated EVERY BLOCKED/ERROR gate as encounter-wide before ever
consulting `affected_fact_ids`, so a single fact-local integrity failure (a
quote that does not anchor to the source, say) erased every other independently
defensible line in the same encounter. Synthetic facts/codes throughout.
"""
import unittest

from claude_coder import autonomy
from claude_coder.models import (CandidateCode, ClinicalFact, CodingResult, Destination,
                                 FactKind, GateResult, Outcome, ResolutionMethod,
                                 ResolvedLine, Verdict)


def _fact(fact_id):
    return ClinicalFact(kind=FactKind.PROCEDURE, description=f"service {fact_id}",
                        confidence=0.9, fact_id=fact_id)


def _cand(code):
    return CandidateCode(code=code, system="cpt", descriptor="assembly service",
                         score=0.9, source="retrieval")


def _line(fact_id, code):
    return ResolvedLine(fact=_fact(fact_id), chosen=_cand(code),
                        method=ResolutionMethod.VERIFIED)


def _result(gates, lines):
    return CodingResult(encounter_id="enc-1", date_of_service="2026-01-01",
                        lines=lines, gates=gates)


class ScopedHardStopTest(unittest.TestCase):

    def test_a_scoped_blocked_gate_excludes_only_its_own_fact(self):
        clean = _line("F2", "CAND_B")
        result = _result(
            gates=[GateResult("evidence_required", Outcome.BLOCKED,
                              "quoted evidence does not anchor to the source",
                              "eligibility-before-retrieval", affected_fact_ids=("F1",))],
            lines=[_line("F1", "CAND_A"), clean])
        verdict = autonomy.decide(result)
        self.assertNotEqual(result.destination, Destination.BLOCKED)
        self.assertIn(clean, [ln for ln in result.billable_lines])
        self.assertEqual({ln.fact.fact_id for ln in result.billable_lines}, {"F2"})

    def test_the_scoped_gate_itself_still_appears_in_routing_as_blocked(self):
        """Never silently downgraded to a provider question or dropped."""
        result = _result(
            gates=[GateResult("evidence_required", Outcome.BLOCKED,
                              "quoted evidence does not anchor to the source",
                              "eligibility-before-retrieval", affected_fact_ids=("F1",))],
            lines=[_line("F1", "CAND_A"), _line("F2", "CAND_B")])
        autonomy.decide(result)
        item = next(r for r in result.routing if r["subject"] == "evidence_required")
        self.assertEqual(item["destination"], Destination.BLOCKED.value)
        self.assertFalse(item["blocking"])

    def test_an_unscoped_blocked_gate_still_hard_stops_the_whole_encounter(self):
        """A genuine encounter-wide structural/integrity failure (no named
        fact_ids) is UNCHANGED -- it still blocks everything immediately."""
        result = _result(
            gates=[GateResult("source_manifest", Outcome.BLOCKED,
                              "document version mismatch", "integrity")],
            lines=[_line("F1", "CAND_A"), _line("F2", "CAND_B")])
        verdict = autonomy.decide(result)
        self.assertEqual(verdict, Verdict.BLOCKED)
        self.assertEqual(result.destination, Destination.BLOCKED)
        # An unscoped hard stop returns immediately, before any excluded_reason
        # is stamped -- unchanged from before this round, since a genuine
        # encounter-wide integrity failure means nothing in the encounter can
        # be trusted regardless of what `billable_lines` would otherwise show;
        # `destination`/`verdict` (asserted above) are what release actually
        # gates on, not this property.
        self.assertEqual(len(result.billable_lines), 2)

    def test_an_unnamed_second_blocked_gate_stays_unscoped_even_alongside_a_scoped_one(self):
        """A mix of one scoped and one unscoped BLOCKED gate must still hard-
        stop the whole encounter -- the unscoped one alone is sufficient, and
        the scoped one does not somehow make it partial."""
        result = _result(
            gates=[GateResult("evidence_required", Outcome.BLOCKED, "detail",
                              "eligibility-before-retrieval", affected_fact_ids=("F1",)),
                  GateResult("source_manifest", Outcome.BLOCKED,
                            "document version mismatch", "integrity")],
            lines=[_line("F1", "CAND_A"), _line("F2", "CAND_B")])
        verdict = autonomy.decide(result)
        self.assertEqual(verdict, Verdict.BLOCKED)
        self.assertEqual(result.destination, Destination.BLOCKED)


if __name__ == "__main__":
    unittest.main()
