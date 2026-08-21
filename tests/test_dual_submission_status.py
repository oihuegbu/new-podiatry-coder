"""Dual coding/submission status (issue #6 item 7): unresolved (not contradicted)
actor ownership now reaches retrieval and resolves to a code, but the resulting
line is stamped HELD rather than READY. An affirmative ownership CONTRADICTION
still never reaches retrieval at all -- that boundary is unchanged.

Agnostic -- synthetic codes + stub LLMs, no API key, no real medical code."""
import unittest

from claude_coder.data_access import MockSource
from claude_coder.eligibility import (ClaimComponent, EligibilityDecision,
                                      EligibilityState, RetrievalRequest, _classify,
                                      evaluate)
from claude_coder.models import (CandidateCode, ClaimSubmissionStatus, ClinicalFact,
                                 Disposition, EvidenceSpan, FactKind, Outcome)
from claude_coder.pipeline import code_encounter
from claude_coder.provenance import NullAuditRepository
from tests import shortlist_verdict as _sv


def _decision(gate, outcome):
    return EligibilityDecision(gate, outcome, "test", "test")


_ALL_PASS_SERVICE_GATES = ["evidence_required", "occurrence", "actor_ownership",
                          "part_of_demotion", "conflict", "documentation_minimum",
                          "axis_consensus"]


def _decisions(overrides: dict[str, Outcome]) -> list[EligibilityDecision]:
    return [_decision(g, overrides.get(g, Outcome.PASS)) for g in _ALL_PASS_SERVICE_GATES]


class ClassifyActorOwnershipUnknown(unittest.TestCase):
    def test_ownership_unknown_alone_is_eligible_not_auto_hold(self):
        state = _classify(_decisions({"actor_ownership": Outcome.UNKNOWN}))
        self.assertEqual(state, EligibilityState.ELIGIBLE_FOR_RETRIEVAL)

    def test_ownership_blocked_still_auto_holds(self):
        state = _classify(_decisions({"actor_ownership": Outcome.BLOCKED}))
        self.assertEqual(state, EligibilityState.AUTO_HOLD)

    def test_ownership_unknown_plus_conflict_unknown_still_auto_holds(self):
        state = _classify(_decisions({"actor_ownership": Outcome.UNKNOWN,
                                      "conflict": Outcome.UNKNOWN}))
        self.assertEqual(state, EligibilityState.AUTO_HOLD)

    def test_ownership_unknown_plus_documentation_minimum_unknown_still_auto_holds(self):
        state = _classify(_decisions({"actor_ownership": Outcome.UNKNOWN,
                                      "documentation_minimum": Outcome.UNKNOWN}))
        self.assertEqual(state, EligibilityState.AUTO_HOLD)

    def test_ownership_unknown_plus_axis_consensus_unknown_still_auto_holds(self):
        state = _classify(_decisions({"actor_ownership": Outcome.UNKNOWN,
                                      "axis_consensus": Outcome.UNKNOWN}))
        self.assertEqual(state, EligibilityState.AUTO_HOLD)

    def test_all_pass_is_eligible(self):
        state = _classify(_decisions({}))
        self.assertEqual(state, EligibilityState.ELIGIBLE_FOR_RETRIEVAL)


class EvaluateStampsSubmissionStatus(unittest.TestCase):
    def _service_fact(self, performer_id=None, billing_entity_id=None):
        attrs = {}
        if performer_id is not None:
            attrs["performer_id"] = performer_id
        if billing_entity_id is not None:
            attrs["billing_entity_id"] = billing_entity_id
        return ClinicalFact(
            FactKind.PROCEDURE, "did a thing", attributes=attrs,
            disposition=Disposition.PERFORMED, fact_id="F1",
            evidence=[EvidenceSpan(text="did a thing", anchored=True, span_id="s1")])

    def test_unresolved_ownership_intent_is_eligible_and_held(self):
        fact = self._service_fact(performer_id="actor-1")  # no billing_entity_id -> UNKNOWN
        intents = evaluate([fact], [], "enc", "2026-01-01")
        intent = next(i for i in intents if i.component is ClaimComponent.SERVICE)
        self.assertEqual(intent.state, EligibilityState.ELIGIBLE_FOR_RETRIEVAL)
        self.assertEqual(intent.claim_submission_status, ClaimSubmissionStatus.HELD)

    def test_resolved_ownership_intent_is_eligible_and_ready(self):
        fact = self._service_fact(performer_id="actor-1", billing_entity_id="actor-1")
        intents = evaluate([fact], [], "enc", "2026-01-01")
        intent = next(i for i in intents if i.component is ClaimComponent.SERVICE)
        self.assertEqual(intent.state, EligibilityState.ELIGIBLE_FOR_RETRIEVAL)
        self.assertEqual(intent.claim_submission_status, ClaimSubmissionStatus.READY)

    def test_contradicted_ownership_intent_is_auto_hold(self):
        fact = self._service_fact(performer_id="actor-1", billing_entity_id="actor-2")
        intents = evaluate([fact], [], "enc", "2026-01-01")
        intent = next(i for i in intents if i.component is ClaimComponent.SERVICE)
        self.assertEqual(intent.state, EligibilityState.AUTO_HOLD)

    def test_held_intent_still_constructs_a_retrieval_request(self):
        fact = self._service_fact(performer_id="actor-1")
        intents = evaluate([fact], [], "enc", "2026-01-01")
        intent = next(i for i in intents if i.component is ClaimComponent.SERVICE)
        # No exception: an ELIGIBLE_FOR_RETRIEVAL intent constructs a RetrievalRequest
        # regardless of its claim_submission_status -- retrieval only ever gates on
        # `state`, never on submission status.
        RetrievalRequest(intent, fact)


_FACTS_UNRESOLVED = ('{"facts":[{"kind":"procedure","description":"excision of lesion",'
                    '"attributes":{"performer_id":"actor-1"},'
                    '"disposition":"performed_today","negated":false,'
                    '"evidence":["excision of lesion performed"],"confidence":0.99}]}')
# `performer_id` is only trusted when the CONTEXT'S OWN participant roster designates
# it a performer (extraction.py: "resolve actor identity ONLY from the typed
# participant graph"), so a genuine ownership CONTRADICTION (not merely an unverifiable
# claim, which strips to UNKNOWN) requires a `billing_context` naming "actor-1" as a
# real performer participant while the encounter's own billing entity is someone else.
_FACTS_CONTRADICTED = _FACTS_UNRESOLVED
_CONTRADICTING_CONTEXT = {"billing_entity_id": "actor-2",
                         "participants": [{"id": "actor-1", "type": "person",
                                           "roles": ["performer"]}]}
_NOTE = "excision of lesion performed today"
_sel = _sv.judge(pick=1, reason="x")


def _src():
    return MockSource(records={("PROC_X", "cpt"): {"active": True}},
                      retrieval={("*", "cpt"): [CandidateCode("PROC_X", "cpt",
                                                              "Excision, lesion, each", 0.9)]})


class PipelineEndToEnd(unittest.TestCase):
    def test_unresolved_ownership_reaches_retrieval_and_bills_held(self):
        # No `billing_context` supplied -- extraction validates `performer_id`
        # only against the context's own typed participant roster, so with no
        # roster it strips the model-claimed id entirely; ownership resolves
        # UNKNOWN (nothing asserted, not a contradiction), the genuinely common
        # real-world "no billing context available yet" case.
        r = code_encounter("e", _NOTE, "2026-03-14", source=_src(),
                           extract_llm=lambda s, u: _FACTS_UNRESOLVED, verify_llm=_sel,
                           corroborate_llm=_sel, audit_repository=NullAuditRepository())
        billed = [ln for ln in r.lines if ln.chosen and ln.chosen.code == "PROC_X"]
        self.assertTrue(billed, "unresolved (not contradicted) ownership must still reach retrieval")
        self.assertEqual(billed[0].claim_submission_status, ClaimSubmissionStatus.HELD)

    def test_contradicted_ownership_never_reaches_retrieval(self):
        r = code_encounter("e", _NOTE, "2026-03-14", source=_src(),
                           extract_llm=lambda s, u: _FACTS_CONTRADICTED, verify_llm=_sel,
                           corroborate_llm=_sel, audit_repository=NullAuditRepository(),
                           billing_context=_CONTRADICTING_CONTEXT)
        billed = [ln for ln in r.lines if ln.chosen and ln.chosen.code == "PROC_X"]
        self.assertFalse(billed, "an affirmative ownership contradiction must still block retrieval")


if __name__ == "__main__":
    unittest.main()
