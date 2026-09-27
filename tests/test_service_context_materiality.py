"""An unresolved performed procedure that shares a SERVICE CONTEXT (the same
operative episode, as `composition.service_intents` grouped it) with a billed
procedure is material to that billed line: the claim must not release
AUTO_READY around it (designated note, third live run: 28118 released alone
while the same episode's tendon reattachment sat unresolved as a non-blocking
coder note). An unresolved procedure sharing NO service context stays the
non-blocking "isolated ambiguity" F9-R9-A established. Synthetic throughout.
"""
import unittest

from claude_coder.autonomy import decide
from claude_coder.models import (CandidateCode, ClinicalFact, CodingResult, Disposition,
                                 EvidenceSpan, FactKind, ResolutionMethod, ResolvedLine)


def _fact(fid, description):
    return ClinicalFact(kind=FactKind.PROCEDURE, description=description,
                        attributes={"performer_id": "person-1", "billing_entity_id": "person-1"},
                        disposition=Disposition.PERFORMED,
                        evidence=[EvidenceSpan(text=f"{description} performed", anchored=True,
                                               span_id=f"sp-{fid}")],
                        confidence=0.9, fact_id=fid)


def _result(*, shared_context: bool):
    billed = _fact("PA", "procedure alpha")
    open_ = _fact("PB", "procedure beta")
    lines = [
        ResolvedLine(fact=billed, chosen=CandidateCode("CPT_A", "cpt", "procedure alpha"),
                     method=ResolutionMethod.DETERMINISTIC),
        ResolvedLine(fact=open_, chosen=None, method=ResolutionMethod.ABSTAINED,
                     alternatives=[CandidateCode("CPT_B", "cpt", "procedure beta"),
                                   CandidateCode("CPT_B2", "cpt", "procedure beta, variant")],
                     rationale="2 shortlisted candidates are still entailed"),
    ]
    intents = ([{"intent_id": "si-1", "component_event_ids": ["PA", "PB"]}]
               if shared_context else
               [{"intent_id": "si-1", "component_event_ids": ["PA"]},
                {"intent_id": "si-2", "component_event_ids": ["PB"]}])
    return CodingResult(encounter_id="enc", date_of_service="2026-03-14", lines=lines,
                        service_intents=intents)


class ServiceContextMaterialityTest(unittest.TestCase):

    def test_an_unresolved_same_episode_procedure_holds_the_billed_line(self):
        result = _result(shared_context=True)
        decide(result, source=None)
        self.assertNotEqual(result.destination.value, "AUTO_READY", result.notes)
        blocking = [r for r in result.routing if r["blocking"]]
        self.assertTrue(any(r["fact_id"] == "PB" for r in blocking), result.routing)
        self.assertIn("PA", result.dependency_excluded_fact_ids)
        reasons = result.dependency_hold_reasons["PA"]
        self.assertTrue(any(r["basis"] == "service_intent" and r["source_fact_id"] == "PB"
                            for r in reasons), reasons)
        # the resolved code is HELD, visibly, never dropped or released around the gap
        self.assertIn("PA", [ln.fact.fact_id for ln in result.submission_held_lines])
        self.assertNotIn("CPT_A", [ln.chosen.code for ln in result.billable_lines])

    def test_an_unresolved_procedure_in_its_own_context_stays_non_blocking(self):
        """F9-R9-A unchanged: no shared episode, no relation -- an isolated open
        question does not hold an independently defensible line hostage."""
        result = _result(shared_context=False)
        decide(result, source=None)
        self.assertEqual(result.destination.value, "AUTO_READY", result.notes)
        self.assertEqual([ln.chosen.code for ln in result.billable_lines], ["CPT_A"])
        non_blocking = [r for r in result.routing if not r["blocking"]]
        self.assertTrue(any(r["fact_id"] == "PB" for r in non_blocking))


if __name__ == "__main__":
    unittest.main()
