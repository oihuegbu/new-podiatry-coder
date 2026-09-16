"""issue #6, Codex's independent re-review, F9-R20-A + its clarification
("downstream controls classify; they do not erase"): a resolved, evidence-
backed line entangled with an unresolved or gate-held dependency used to be
ERASED from the ClaimBundle entirely -- `autonomy.decide()` set
`excluded_reason`, which `bundle_from_coding_result` places in neither
`diagnosis_lines`/`billable_lines` NOR `submission_held_lines`, so a
genuinely valid selection simply vanished with only an untyped audit trace.
Entanglement is a SUBMISSION problem (the claim cannot be certified
independently of the still-open dependency), never proof the selection
itself is invalid -- `claim_submission_status = HELD` (the SAME typed
signal issue #6 item 7 already built for an unresolved actor-ownership
fact) is now used instead, so the code/evidence/provenance stay visible
while the claim correctly stays non-releasable.

issue #6, Codex's independent re-review (F9-R21-D, hardened by F9-R22-B):
entanglement used to be computed from `ClinicalGraph.binding_for`'s AUDIT-
scope closure, which pulls in every one-hop relation regardless of predicate
-- conflating "co-occurred in the same clinical episode" with "can change
this line's own billing correctness" (reproduced live: an unresolved
anesthesia event held an independently valid, resolved procedure line via
mere `SAME_EPISODE_AS` proximity). `autonomy._entangled` is now built ONLY
from joint claim-line-intent membership plus a GROUNDED, ASSERTED,
directional `REASON_FOR` edge (diagnosis -> service, never automatically in
reverse) -- so these tests construct real, grounded `RelationAssertion`s
(and real claim-line-intent stand-ins) instead of a duck-typed graph-closure
stub. A bare `PART_OF` edge is deliberately NOT its own propagation path any
more (F9-R22-B: a generic standalone `PART_OF` assertion is not proof that a
separately resolved component changes its parent's code/modifier/units --
joint claim-line-intent membership is what represents that), and a negated/
unreconciled edge of ANY predicate must never propagate a hold either.

Synthetic facts/codes throughout.
"""
import json
import unittest
from unittest.mock import patch

from claude_coder import autonomy
from claude_coder.models import (CandidateCode, ClaimSubmissionStatus, ClinicalFact,
                                 CodingResult, Destination, FactKind, GateResult,
                                 Outcome, RelationAssertion, RelationPredicate,
                                 RelationState, ResolutionMethod, ResolvedLine, Verdict)


def _fact(fact_id, kind=FactKind.PROCEDURE):
    return ClinicalFact(kind=kind, description=f"service {fact_id}",
                        confidence=0.9, fact_id=fact_id)


def _cand(code):
    return CandidateCode(code=code, system="cpt", descriptor="assembly service",
                         score=0.9, source="retrieval")


def _resolved_line(fact_id, code, kind=FactKind.PROCEDURE):
    return ResolvedLine(fact=_fact(fact_id, kind), chosen=_cand(code),
                        method=ResolutionMethod.VERIFIED, rationale="entailed")


def _unresolved_line(fact_id, kind=FactKind.PROCEDURE):
    return ResolvedLine(fact=_fact(fact_id, kind), chosen=None,
                        method=ResolutionMethod.ABSTAINED, rationale="held")


def _reason_for(diagnosis_id, service_id, *, grounded=True):
    """A documented diagnosis-justifies-service edge -- GROUNDED by default
    (`ASSERTED`, a `GROUNDED_RECONCILIATION_STATUSES` member, non-empty
    `reconciliation_evidence`), since that is the only shape F9-R22-B lets
    propagate a hold. Pass `grounded=False` to build the negative case (an
    edge the record never actually established)."""
    if grounded:
        return RelationAssertion(subject_event_id=diagnosis_id,
                                 predicate=RelationPredicate.REASON_FOR,
                                 object_event_id=service_id,
                                 state=RelationState.ASSERTED,
                                 reconciliation_status="source_directional",
                                 reconciliation_evidence=[f"s-{diagnosis_id}"])
    return RelationAssertion(subject_event_id=diagnosis_id,
                             predicate=RelationPredicate.REASON_FOR,
                             object_event_id=service_id)


def _intent(*clinical_event_ids):
    """A real `eligibility.ClaimLineIntent` -- `certificate.build_certificate`
    reads its full field set unconditionally whenever `result.claim_line_
    intents` is set, so a bare duck-typed stand-in (only `.clinical_event_
    ids`, what `_entangled` itself reads) is not enough for the bundle/
    certificate end-to-end tests below."""
    from claude_coder import eligibility as _elig
    return _elig.ClaimLineIntent(
        intent_id=f"intent-{'-'.join(clinical_event_ids)}",
        encounter_id="enc-1",
        component=_elig.ClaimComponent.SERVICE,
        clinical_event_ids=list(clinical_event_ids),
        fact_kind="procedure",
        clinical_action="",
        attributes={},
        date_of_service="2026-01-01",
        billing_entity_id=None,
        source_span_ids=[],
        state=_elig.EligibilityState.ELIGIBLE_FOR_RETRIEVAL)


def _result(lines, relations=(), gates=(), intents=()):
    return CodingResult(encounter_id="enc-1", date_of_service="2026-01-01",
                        lines=list(lines), gates=list(gates),
                        relations=list(relations), claim_line_intents=list(intents))


class DependencyHoldTest(unittest.TestCase):
    """`autonomy.decide()`'s dependency-entanglement stamping."""

    def test_resolved_line_entangled_with_an_unresolved_one_is_held_not_excluded(self):
        """The required regression: unresolved diagnosis REASON_FOR a resolved
        service -> the service is held with its code/evidence intact, zero
        submission-ready lines, and a release blocker. The service must not
        disappear. (F9-R21-D: REASON_FOR propagates directionally,
        diagnosis -> service -- the diagnosis is the open item here, not the
        thing held.)"""
        diagnosis = _unresolved_line("D1", kind=FactKind.DIAGNOSIS)
        service = _resolved_line("P1", "SERVICE_A")
        relation = _reason_for("D1", "P1")
        result = _result([diagnosis, service], relations=[relation])

        autonomy.decide(result)

        self.assertIsNone(service.excluded_reason,
                          "entanglement is a submission problem, never an invalid "
                          "selection -- excluded_reason must stay unset")
        self.assertIsNotNone(service.chosen,
                             "the resolved code must survive -- never cleared")
        self.assertEqual(service.chosen.code, "SERVICE_A")
        self.assertEqual(service.claim_submission_status, ClaimSubmissionStatus.HELD)
        self.assertEqual(result.dependency_hold_reasons["P1"][0]["basis"],
                         "grounded_reason_for")
        self.assertEqual(result.dependency_hold_reasons["P1"][0]["source_fact_id"],
                         "D1")
        self.assertIn(service, result.submission_held_lines)
        self.assertNotIn(service, result.billable_lines)
        # Zero submission-ready lines, and release is blocked.
        self.assertEqual(result.billable_lines, [])
        self.assertNotEqual(result.destination, Destination.AUTO_READY)
        self.assertNotEqual(result.verdict, Verdict.AUTO_READY)

    def test_the_held_line_routes_as_a_coder_review_item_not_a_provider_query(self):
        """Distinguished from the pre-existing unresolved-actor-ownership HELD
        shape (which IS a provider-answerable question) -- an entangled
        dependency is a coding-judgement item, not something a provider can
        answer."""
        diagnosis = _unresolved_line("D1", kind=FactKind.DIAGNOSIS)
        service = _resolved_line("P1", "SERVICE_A")
        relation = _reason_for("D1", "P1")
        result = _result([diagnosis, service], relations=[relation])

        autonomy.decide(result)

        item = next(r for r in result.routing if r["subject"] == "service P1")
        self.assertEqual(item["destination"], Destination.REVIEW.value)

    def test_a_gate_scoped_hold_names_the_exact_material_control(self):
        service = _resolved_line("P1", "SERVICE_A")
        gate = GateResult(
            "medical_necessity", Outcome.UNKNOWN,
            "no grounded diagnosis-to-service support", "synthetic authority",
            affected_fact_ids=("P1",))
        result = _result([service], gates=[gate])

        autonomy.decide(result)

        self.assertEqual(service.claim_submission_status, ClaimSubmissionStatus.HELD)
        self.assertEqual(result.dependency_hold_reasons["P1"][0]["basis"],
                         "gate_scope")
        self.assertEqual(result.dependency_hold_reasons["P1"][0]["gate"],
                         "medical_necessity")
        self.assertIn("no grounded diagnosis-to-service support", service.rationale)
        self.assertNotIn("clinical episode", service.rationale)

    def test_an_unrelated_resolved_line_is_untouched(self):
        """Only genuinely entangled facts are held -- an independent,
        unrelated line stays fully billable."""
        diagnosis = _unresolved_line("D1", kind=FactKind.DIAGNOSIS)
        service = _resolved_line("P1", "SERVICE_A")
        unrelated = _resolved_line("D2", "DX_B", kind=FactKind.DIAGNOSIS)
        relation = _reason_for("D1", "P1")
        result = _result([diagnosis, service, unrelated], relations=[relation])

        autonomy.decide(result)

        self.assertEqual(unrelated.claim_submission_status, ClaimSubmissionStatus.READY)
        self.assertIn(unrelated, result.billable_lines)

    def test_repeated_decide_calls_do_not_stack_the_hold_reason(self):
        """`decide()` may run more than once per encounter (claim-set
        reconciliation rounds) -- the guard against re-stamping an
        already-HELD line must actually prevent the rationale from growing
        unbounded across rounds."""
        diagnosis = _unresolved_line("D1", kind=FactKind.DIAGNOSIS)
        service = _resolved_line("P1", "SERVICE_A")
        relation = _reason_for("D1", "P1")
        result = _result([diagnosis, service], relations=[relation])

        autonomy.decide(result)
        first_rationale = service.rationale
        autonomy.decide(result)
        self.assertEqual(service.rationale, first_rationale)


class SameEpisodeProximityNeverHoldsTest(unittest.TestCase):
    """issue #6, Codex's independent re-review (F9-R21-D): the required
    negative regression -- an unresolved parallel service sharing an episode
    with a supported, resolved service must leave the resolved service alone
    unless a typed claim dependency (`PART_OF`/`REASON_FOR`) actually exists.
    `SAME_EPISODE_AS` is deliberately NOT a claim-impact predicate."""

    def test_same_episode_as_alone_does_not_hold_the_resolved_line(self):
        anesthesia = _unresolved_line("A1")
        procedure = _resolved_line("P1", "SERVICE_A")
        relation = RelationAssertion(subject_event_id="A1",
                                     predicate=RelationPredicate.SAME_EPISODE_AS,
                                     object_event_id="P1")
        result = _result([anesthesia, procedure], relations=[relation])

        autonomy.decide(result)

        self.assertEqual(procedure.claim_submission_status, ClaimSubmissionStatus.READY)
        self.assertIn(procedure, result.billable_lines)
        self.assertNotIn("P1", result.dependency_hold_reasons)

    def test_a_negated_unreconciled_reason_for_does_not_hold_the_resolved_line(self):
        """issue #6, Codex's independent re-review (F9-R22-B): a relation
        must itself be genuinely established by the record before it may
        propagate anything -- an explicitly `NEGATED`, unreconciled
        `REASON_FOR` assertion (evidence the source never actually
        established) must not hold the independently resolved service."""
        diagnosis = _unresolved_line("D1", kind=FactKind.DIAGNOSIS)
        service = _resolved_line("P1", "SERVICE_A")
        relation = RelationAssertion(subject_event_id="D1",
                                     predicate=RelationPredicate.REASON_FOR,
                                     object_event_id="P1",
                                     state=RelationState.NEGATED)
        result = _result([diagnosis, service], relations=[relation])

        autonomy.decide(result)

        self.assertEqual(service.claim_submission_status, ClaimSubmissionStatus.READY)
        self.assertIn(service, result.billable_lines)

    def test_an_ungrounded_reason_for_does_not_hold_the_resolved_line(self):
        """An `ASSERTED` `REASON_FOR` edge that never reconciled against the
        source (still `unreconciled`, no `reconciliation_evidence`) is just
        as ungrounded as a negated one -- state alone is not enough."""
        diagnosis = _unresolved_line("D1", kind=FactKind.DIAGNOSIS)
        service = _resolved_line("P1", "SERVICE_A")
        relation = _reason_for("D1", "P1", grounded=False)
        result = _result([diagnosis, service], relations=[relation])

        autonomy.decide(result)

        self.assertEqual(service.claim_submission_status, ClaimSubmissionStatus.READY)
        self.assertIn(service, result.billable_lines)

    def test_an_unresolved_separate_component_part_of_does_not_hold_the_primary(self):
        """issue #6, Codex's independent re-review (F9-R22-B): a bare,
        grounded `PART_OF` edge is no longer, by itself, proof that a
        separately resolved component changes its parent's own billing
        correctness -- only joint claim-line-intent membership (the
        eligibility engine's own "these facts are ONE code-determining
        line" decision) does. An unresolved separate supply/component that
        the engine did NOT compose into the primary service's own claim
        line must not hold it, even with a fully grounded `PART_OF` edge
        between them."""
        component = _unresolved_line("U1")
        primary = _resolved_line("P1", "SERVICE_A")
        relation = RelationAssertion(subject_event_id="U1",
                                     predicate=RelationPredicate.PART_OF,
                                     object_event_id="P1",
                                     state=RelationState.ASSERTED,
                                     reconciliation_status="source_directional",
                                     reconciliation_evidence=["s-U1"])
        result = _result([component, primary], relations=[relation])

        autonomy.decide(result)

        self.assertEqual(primary.claim_submission_status, ClaimSubmissionStatus.READY)
        self.assertIn(primary, result.billable_lines)

    def test_an_additional_unresolved_indication_does_not_hold_an_already_supported_service(self):
        """An open additional diagnosis stays visible on its own line but is
        not material to a service whose exact necessity binding already closed
        on another grounded diagnosis."""
        weak_diagnosis = _unresolved_line("D1", kind=FactKind.DIAGNOSIS)
        strong_diagnosis = _resolved_line("D2", "DX_B", kind=FactKind.DIAGNOSIS)
        service = _resolved_line("P1", "SERVICE_A")
        relations = [_reason_for("D1", "P1"), _reason_for("D2", "P1")]
        result = _result([weak_diagnosis, strong_diagnosis, service], relations=relations)
        result.necessity_support = [{
            "procedure_event_id": "P1",
            "supports": [{"diagnosis_event_id": "D2"}],
        }]

        autonomy.decide(result)

        self.assertEqual(service.claim_submission_status, ClaimSubmissionStatus.READY)
        self.assertIn(service, result.billable_lines)
        self.assertNotIn(service, result.submission_held_lines)
        self.assertNotIn("P1", result.dependency_hold_reasons)


class DependencyHoldBundleTest(unittest.TestCase):
    """End-to-end into the real `ClaimBundle` projection."""

    def test_the_held_diagnosis_is_visible_in_the_bundle_as_held_policy_or_data(self):
        from app.contracts.claim_bundle import (AuthorityBinding, EncounterContext,
                                                LineStatus, SourceDocument,
                                                bundle_from_coding_result)
        diagnosis = _resolved_line("D1", "DX_A", kind=FactKind.DIAGNOSIS)
        procedure = _unresolved_line("P1")
        # issue #6, Codex's independent re-review (F9-R22-B): a bare
        # `PART_OF` edge is no longer its own propagation path -- joint
        # claim-line-intent membership (the eligibility engine's own "these
        # facts are ONE code-determining line" decision) is what represents
        # this now. D1 composed into the same claim line as the still-
        # unresolved P1 leaves D1 with nothing resolved to attach to yet, so
        # D1 (not P1) is the one held here.
        result = _result([diagnosis, procedure], intents=[_intent("D1", "P1")])
        autonomy.decide(result)

        bundle = bundle_from_coding_result(
            result, source_document=SourceDocument(), context=EncounterContext(),
            authority=AuthorityBinding())
        held_diagnoses = [d for d in bundle.diagnoses if d.code == "DX_A"]
        self.assertTrue(held_diagnoses,
                        "the resolved diagnosis must survive into the bundle, not "
                        "disappear")
        self.assertEqual(held_diagnoses[0].status, LineStatus.HELD_POLICY_OR_DATA)
        self.assertTrue(bundle.release_blockers())

    def test_the_certificate_attests_the_held_diagnosis_too(self):
        """issue #6, Codex's independent re-review (F9-R19-A consolidated
        live-run remediation, Finding 5): `certificate.build_certificate`
        used to attest ONLY `result.billable_lines`, while the ClaimBundle
        (since F9-R20-A) also projects `result.submission_held_lines` --
        so a genuinely resolved, HELD diagnosis appeared in the bundle but
        the certificate reported it unattested. Both artifacts must now
        certify the SAME coded-output line set, and agree on submission
        status too."""
        from app.contracts.claim_bundle import (AuthorityBinding, EncounterContext,
                                                SourceDocument, bundle_from_coding_result)
        from claude_coder import certificate
        diagnosis = _resolved_line("D1", "DX_A", kind=FactKind.DIAGNOSIS)
        procedure = _unresolved_line("P1")
        result = _result([diagnosis, procedure], intents=[_intent("D1", "P1")])
        autonomy.decide(result)

        cert = certificate.build_certificate(result, "note text")
        cert_line = next(ln for ln in cert["lines"] if ln["code"] == "DX_A")
        self.assertEqual(cert_line["submission_status"], "held")

        bundle = bundle_from_coding_result(
            result, source_document=SourceDocument(), context=EncounterContext(),
            authority=AuthorityBinding())
        problems = bundle._attested_line_problems(cert)
        self.assertEqual(problems, (),
                         "the certificate must attest the held diagnosis exactly "
                         "like the bundle does, including its submission status")


class EncounterWideFailurePreservesLinesTest(unittest.TestCase):
    """issue #6, Codex's independent re-review, F9-R20-A clarification:
    "Only a genuinely encounter-wide source/integrity failure may prevent
    encounter release. Even then, extracted events, selected codes,
    candidates, evidence, and failure details must remain visible in the
    non-releasable ClaimBundle." -- `pipeline._system_hold_result` used to
    build a brand-new, EMPTY `CodingResult` for a mid-loop operational
    failure, discarding every already-resolved line from earlier facts in
    the SAME encounter."""

    def test_system_hold_result_carries_forward_already_resolved_lines(self):
        from claude_coder.pipeline import _system_hold_result
        already_resolved = _resolved_line("F1", "CAND_A")
        result = _system_hold_result(
            "enc-1", "2026-01-01", "retrieval_execution:F2",
            RuntimeError("simulated failure"), source=None, lines=[already_resolved])
        self.assertIn(already_resolved, result.lines)
        self.assertIsNotNone(already_resolved.chosen)
        self.assertEqual(result.destination, Destination.SYSTEM_HOLD)

    def test_system_hold_result_with_no_lines_is_unchanged(self):
        """A genuinely pre-retrieval failure (nothing extracted yet) still
        produces an empty result exactly as before -- `lines` is optional."""
        from claude_coder.pipeline import _system_hold_result
        result = _system_hold_result(
            "enc-1", "2026-01-01", "source_evidence_integrity",
            RuntimeError("simulated failure"), source=None)
        self.assertEqual(result.lines, [])


class PerEventResolutionFailureDoesNotAbortEncounterTest(unittest.TestCase):
    """A provider/retrieval failure for one event must classify that event only.

    This is intentionally end-to-end through ``code_encounter`` rather than a
    helper-only test: the former regression returned from the middle of the
    per-fact loop, meaning subsequent, independently documented services never
    entered the ClaimBundle at all.
    """

    @staticmethod
    def _facts():
        facts = []
        for fact_id, wording in (("F1", "service alpha performed"),
                                 ("F2", "service beta performed"),
                                 ("F3", "service gamma performed")):
            facts.append({
                "fact_id": fact_id,
                "kind": "procedure",
                "description": wording,
                "attributes": {"performer_id": "actor-1",
                               "billing_entity_id": "actor-1"},
                "disposition": "performed_today",
                "negated": False,
                "evidence": [wording],
                "confidence": 0.99,
            })
        return json.dumps({"facts": facts})

    def test_one_resolution_failure_holds_only_its_event_and_continues(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import CandidateCode
        from claude_coder.pipeline import code_encounter
        from claude_coder.provenance import NullAuditRepository

        source = MockSource()

        def resolve_one(request, _source, **_kwargs):
            if request.fact.fact_id == "F2":
                raise RuntimeError("simulated provider outage")
            return ResolvedLine(
                fact=request.fact,
                chosen=CandidateCode(
                    code=f"SYNTHETIC_{request.fact.fact_id}", system="cpt",
                    descriptor="synthetic service", score=1.0,
                    source="test-authority"),
                method=ResolutionMethod.VERIFIED,
                rationale="synthetically supported")

        note = ("service alpha performed. service beta performed. "
                "service gamma performed.")
        with patch("claude_coder.pipeline.resolution.resolve", side_effect=resolve_one):
            result = code_encounter(
                "enc", note, "2026-01-01", source=source,
                extract_llm=lambda _system, _user: self._facts(),
                # Supplying a verifier prevents the production default provider
                # client from being created; the patched resolver never invokes it.
                verify_llm=lambda _system, _user: "{}",
                audit_repository=NullAuditRepository(),
                billing_context={
                    "billing_entity_id": "actor-1",
                    "participants": [{"id": "actor-1", "type": "person",
                                      "roles": ["performer"]}],
                })

        by_id = {line.fact.fact_id: line for line in result.lines}
        self.assertEqual(set(by_id), {"F1", "F2", "F3"})
        self.assertTrue(by_id["F1"].resolved)
        self.assertTrue(by_id["F3"].resolved)
        self.assertEqual(
            {line.fact.fact_id for line in result.billable_lines}, {"F1", "F3"},
            "a fact-scoped provider failure must not hold independently defensible "
            "sibling lines at final ClaimBundle routing")
        self.assertFalse(by_id["F2"].resolved)
        self.assertTrue(by_id["F2"].retrieval_attempted)
        self.assertIn("retry required", by_id["F2"].rationale)
        self.assertIsNone(by_id["F2"].documentation_gap,
                          "an external execution failure is never a provider query")

        failures = [gate for gate in result.gates
                    if gate.name == "retrieval_execution:F2"]
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0].retryable)
        self.assertEqual(failures[0].affected_fact_ids, ("F2",))
        self.assertFalse(any(gate.name.startswith("retrieval_execution:")
                             and not gate.affected_fact_ids
                             for gate in result.gates))


class DecidePreservingDependencyHoldsTest(unittest.TestCase):
    """`pipeline._decide_preserving_dependency_holds` (issue #6, independent
    review of Codex's 514e0c1/610f7fb round): three late-stage call sites in
    `code_encounter` (a missing data fingerprint, a certificate-build
    exception, an audit-persist exception) run AFTER
    `_reconcile_claim_after_pruning` has already returned its merged,
    correct `dependency_hold_reasons`. `autonomy.decide()` itself
    deliberately sets that field fresh on every call ("never accumulated
    here; the caller is the one with a reason to accumulate") -- so a bare
    `decide()` call at those three sites would silently overwrite the fuller
    history with whatever THIS one call happens to see, exactly the "fix
    works round 1, loses state one call later" class this codebase has hit
    before. Nothing external reads this field today, so nothing user-facing
    breaks yet, but the field exists specifically so a held line stays
    auditable without reverse-engineering a generic prose message."""

    def test_a_prior_hold_reason_survives_a_call_that_finds_nothing_new(self):
        from claude_coder import pipeline

        service = _resolved_line("P1", "SERVICE_A")
        result = _result([service])
        result.dependency_hold_reasons = {
            "D1": [{"basis": "grounded_reason_for", "source_fact_id": "X1"}]}

        pipeline._decide_preserving_dependency_holds(result, source=None)

        self.assertIn("D1", result.dependency_hold_reasons,
                      "a hold reason recorded by an earlier reconciliation round "
                      "must survive a later, unrelated decide() call that finds "
                      "nothing new to block")
        self.assertEqual(result.dependency_hold_reasons["D1"][0]["source_fact_id"],
                         "X1")

    def test_a_new_hold_reason_from_this_call_is_added_not_dropped(self):
        from claude_coder import pipeline

        diagnosis = _unresolved_line("D2", kind=FactKind.DIAGNOSIS)
        service = _resolved_line("P1", "SERVICE_A")
        relation = _reason_for("D2", "P1")
        result = _result([diagnosis, service], relations=[relation])
        result.dependency_hold_reasons = {
            "D1": [{"basis": "grounded_reason_for", "source_fact_id": "X1"}]}

        pipeline._decide_preserving_dependency_holds(result, source=None)

        self.assertIn("D1", result.dependency_hold_reasons,
                      "the prior round's reason must still be present")
        self.assertIn("P1", result.dependency_hold_reasons,
                      "a genuinely new dependency this call discovers must still "
                      "be recorded, not suppressed by the preservation logic")
        self.assertEqual(result.dependency_hold_reasons["P1"][0]["basis"],
                         "grounded_reason_for")

    def test_no_duplicate_entries_when_the_same_reason_recurs(self):
        from claude_coder import pipeline

        diagnosis = _unresolved_line("D1", kind=FactKind.DIAGNOSIS)
        service = _resolved_line("P1", "SERVICE_A")
        relation = _reason_for("D1", "P1")
        result = _result([diagnosis, service], relations=[relation])
        autonomy.decide(result)
        prior_reasons = {k: list(v) for k, v in result.dependency_hold_reasons.items()}

        pipeline._decide_preserving_dependency_holds(result, source=None)

        self.assertEqual(result.dependency_hold_reasons.keys(), prior_reasons.keys())
        for fact_id, reasons in prior_reasons.items():
            self.assertEqual(result.dependency_hold_reasons[fact_id], reasons,
                             "re-finding the SAME reason must not duplicate it")


if __name__ == "__main__":
    unittest.main()
