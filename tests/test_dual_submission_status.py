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
                          "composition_context", "relationship_context",
                          "documentation_minimum",
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

    def test_ownership_unknown_plus_relationship_context_does_not_auto_hold(self):
        state = _classify(_decisions({"actor_ownership": Outcome.UNKNOWN,
                                      "relationship_context": Outcome.PASS}))
        self.assertEqual(state, EligibilityState.ELIGIBLE_FOR_RETRIEVAL)

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


_FACTS_UNRESOLVED = ('{"facts":[{"fact_id":"F1","kind":"procedure",'
                    '"description":"excision of lesion",'
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


class HeldSubmissionIsEnforcedNotOnlyStamped(unittest.TestCase):
    """Codex F8-R3: `claim_submission_status=HELD` must actually block autonomous
    release, not merely be readable in the audit trail. Regression across the
    full producer -> autonomy -> bundle -> release-authorizer chain."""

    def test_held_line_is_excluded_from_billable_lines(self):
        r = code_encounter("e", _NOTE, "2026-03-14", source=_src(),
                           extract_llm=lambda s, u: _FACTS_UNRESOLVED, verify_llm=_sel,
                           corroborate_llm=_sel, audit_repository=NullAuditRepository())
        self.assertTrue(any(ln.chosen and ln.chosen.code == "PROC_X" for ln in r.lines))
        self.assertFalse(any(ln.chosen and ln.chosen.code == "PROC_X"
                             for ln in r.billable_lines),
                         "a HELD line must never be submission-ready")
        self.assertTrue(any(ln.chosen and ln.chosen.code == "PROC_X"
                            for ln in r.submission_held_lines),
                        "the coded line must still be visible for administrative routing")

    def test_autonomy_never_returns_auto_ready_with_a_held_line(self):
        from claude_coder.models import Destination, Verdict
        r = code_encounter("e", _NOTE, "2026-03-14", source=_src(),
                           extract_llm=lambda s, u: _FACTS_UNRESOLVED, verify_llm=_sel,
                           corroborate_llm=_sel, audit_repository=NullAuditRepository())
        self.assertNotEqual(r.destination, Destination.AUTO_READY)
        self.assertNotEqual(r.verdict, Verdict.AUTO_READY)
        self.assertTrue(any(item["blocking"] for item in r.routing))

    def test_bundle_carries_the_held_line_as_held_not_erased_but_no_ready_claim_still_blocks(self):
        """issue #6 F9-R7 item 4 supersedes this test's ORIGINAL assertion (that a
        HELD line was excluded from `service_lines` entirely, visible only as an
        untyped audit dict): that shape was itself the erasure bug the product-
        priority reset named -- a discovered, recommended code disappearing from
        the artifact instead of being carried with an explicit status. A bundle
        containing ONLY a held service still cannot release because it
        has no submission-ready claim.  A held line is not itself encounter-wide:
        when an unrelated RECOMMENDED service exists, the canonical submission
        projection may release that service without transmitting this one."""
        from app.contracts.claim_bundle import (AuthorityBinding, EncounterContext,
                                                 LineStatus, SourceDocument,
                                                 bundle_from_coding_result)
        r = code_encounter("e", _NOTE, "2026-03-14", source=_src(),
                           extract_llm=lambda s, u: _FACTS_UNRESOLVED, verify_llm=_sel,
                           corroborate_llm=_sel, audit_repository=NullAuditRepository())
        bundle = bundle_from_coding_result(
            r, source_document=SourceDocument(), context=EncounterContext(),
            authority=AuthorityBinding())
        held = [sl for sl in bundle.service_lines if sl.code == "PROC_X"]
        self.assertTrue(held, "the coded line must survive into the bundle, not disappear")
        self.assertEqual(held[0].status, LineStatus.HELD_POLICY_OR_DATA)
        held_audit = [e for e in bundle.audit.excluded_lines
                     if e.get("code") == "PROC_X"]
        self.assertTrue(held_audit, "the coded line must still be visible in the audit trail")
        self.assertEqual(held_audit[0]["claim_submission_status"], "held")
        blockers = bundle.release_blockers()
        self.assertTrue(any("no submission-ready service" in b for b in blockers),
                        blockers)

    def test_held_line_is_visible_but_does_not_block_a_ready_submission_projection(self):
        """One uncertain selected line neither transmits nor erases a ready line."""
        from app.contracts.claim_bundle import (
            BundleOrigin, ClaimBundle, DiagnosisLine, EncounterIdentity,
            ExternalDisposition, LineStatus, ReleaseDestination, ReleaseStatus,
            ServiceLine,
        )

        diagnosis = DiagnosisLine(
            sequence=1, system="test-dx", code="DX_TEST", primary=True,
            clinical_event_id="D1")
        ready = ServiceLine(
            sequence=1, system="test-service", code="SERVICE_READY", units=1,
            diagnosis_pointers=(1,), clinical_event_id="S1")
        held = ServiceLine(
            sequence=2, system="test-service", code="SERVICE_HELD", units=1,
            diagnosis_pointers=(), clinical_event_id="S2",
            status=LineStatus.HELD_POLICY_OR_DATA,
            external_disposition=ExternalDisposition.EXCLUDED,
            blocking_stage="submission", reason_code="held_policy_or_data")
        bundle = ClaimBundle(
            produced_by=BundleOrigin.CLAUDE_CODER,
            encounter=EncounterIdentity(encounter_id="e", document_id="e",
                                        date_of_service="2026-01-01"),
            diagnoses=(diagnosis,), service_lines=(ready, held),
            release=ReleaseStatus(destination=ReleaseDestination.AUTO_READY),
        )

        self.assertEqual(bundle.submission_diagnoses, (diagnosis,))
        self.assertEqual(bundle.submission_service_lines, (ready,))
        self.assertEqual(bundle.submission_diagnosis_pointers(ready), (1,))
        self.assertEqual([line["code"] for line in bundle.claim_content()["service_lines"]],
                         ["SERVICE_READY"])
        blockers = bundle.release_blockers()
        self.assertFalse(any("coded line's submission is HELD" in b for b in blockers),
                         blockers)
        self.assertFalse(any("SERVICE_HELD has no diagnosis" in b for b in blockers),
                         blockers)

    def test_release_blockers_refuses_auto_ready_when_only_service_is_held(self):
        """A forged AUTO_READY destination cannot turn a held-only artifact into a claim."""
        from app.contracts.claim_bundle import (AuthorityBinding, EncounterContext,
                                                 ReleaseDestination, SourceDocument,
                                                 bundle_from_coding_result)
        r = code_encounter("e", _NOTE, "2026-03-14", source=_src(),
                           extract_llm=lambda s, u: _FACTS_UNRESOLVED, verify_llm=_sel,
                           corroborate_llm=_sel, audit_repository=NullAuditRepository())
        bundle = bundle_from_coding_result(
            r, source_document=SourceDocument(), context=EncounterContext(),
            authority=AuthorityBinding())
        # Simulate a producer defect: force AUTO_READY even though the only
        # selected service is held and the submission projection is empty.
        tampered = bundle.model_copy(update={
            "release": bundle.release.model_copy(
                update={"destination": ReleaseDestination.AUTO_READY,
                       "producer_releasable": True})})
        blockers = tampered.release_blockers()
        self.assertTrue(any("no submission-ready service" in b for b in blockers),
                        blockers)

    def test_held_primary_diagnosis_is_removed_and_ready_pointer_is_compacted(self):
        """The canonical payload and pointer map agree after a held diagnosis drops."""
        from app.contracts.claim_bundle import (
            BundleOrigin, ClaimBundle, DiagnosisLine, EncounterIdentity,
            ExternalDisposition, LineStatus, ReleaseDestination, ReleaseStatus,
            ServiceLine,
        )

        held_primary = DiagnosisLine(
            sequence=1, system="test-dx", code="DX_HELD", primary=True,
            clinical_event_id="D1", status=LineStatus.HELD_POLICY_OR_DATA,
            external_disposition=ExternalDisposition.EXCLUDED,
            blocking_stage="submission", reason_code="held_policy_or_data")
        ready_diagnosis = DiagnosisLine(
            sequence=2, system="test-dx", code="DX_READY", primary=False,
            clinical_event_id="D2")
        ready_service = ServiceLine(
            sequence=1, system="test-service", code="SERVICE_READY", units=1,
            diagnosis_pointers=(2,), clinical_event_id="S1")
        bundle = ClaimBundle(
            produced_by=BundleOrigin.CLAUDE_CODER,
            encounter=EncounterIdentity(encounter_id="e", document_id="e",
                                        date_of_service="2026-01-01"),
            diagnoses=(held_primary, ready_diagnosis),
            service_lines=(ready_service,),
            release=ReleaseStatus(destination=ReleaseDestination.AUTO_READY),
        )

        content = bundle.claim_content()
        self.assertEqual(content["diagnoses"], [{
            "sequence": 1, "system": "test-dx", "code": "DX_READY",
            "primary": True, "clinical_event_id": "D2",
        }])
        self.assertEqual(content["service_lines"][0]["diagnosis_pointers"], [1])
        self.assertEqual(bundle.submission_diagnosis_pointers(ready_service), (1,))


if __name__ == "__main__":
    unittest.main()
