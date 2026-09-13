"""issue #6, Codex's independent re-review, F9-R20-A + its clarification
("downstream controls classify; they do not erase"): a resolved, evidence-
backed line entangled (via the clinical graph) with an unresolved or
gate-held dependency used to be ERASED from the ClaimBundle entirely --
`autonomy.decide()` set `excluded_reason`, which `bundle_from_coding_result`
places in neither `diagnosis_lines`/`billable_lines` NOR
`submission_held_lines`, so a genuinely valid selection simply vanished with
only an untyped audit trace. Entanglement is a SUBMISSION problem (the claim
cannot be certified independently of the still-open dependency), never proof
the selection itself is invalid -- `claim_submission_status = HELD` (the
SAME typed signal issue #6 item 7 already built for an unresolved actor-
ownership fact) is now used instead, so the code/evidence/provenance stay
visible while the claim correctly stays non-releasable.

Synthetic facts/codes throughout.
"""
import unittest

from claude_coder import autonomy
from claude_coder.models import (CandidateCode, ClaimSubmissionStatus, ClinicalFact,
                                 CodingResult, Destination, FactKind, GateResult,
                                 Outcome, ResolutionMethod, ResolvedLine, Verdict)


def _fact(fact_id, kind=FactKind.PROCEDURE):
    return ClinicalFact(kind=kind, description=f"service {fact_id}",
                        confidence=0.9, fact_id=fact_id)


def _cand(code):
    return CandidateCode(code=code, system="cpt", descriptor="assembly service",
                         score=0.9, source="retrieval")


def _resolved_line(fact_id, code, kind=FactKind.PROCEDURE):
    return ResolvedLine(fact=_fact(fact_id, kind), chosen=_cand(code),
                        method=ResolutionMethod.VERIFIED, rationale="entailed")


def _unresolved_line(fact_id):
    return ResolvedLine(fact=_fact(fact_id), chosen=None,
                        method=ResolutionMethod.ABSTAINED, rationale="held")


class _Binding:
    def __init__(self, clinical_event_ids):
        self.clinical_event_ids = clinical_event_ids


class _EntangledGraph:
    """Minimal duck-typed stand-in for `ClinicalGraph.binding_for` -- returns
    a fixed closure for any queried fact_id, without needing a real graph."""

    def __init__(self, closures: dict[str, set[str]]):
        self._closures = closures

    def binding_for(self, node_ids):
        closure = set(node_ids or [])
        for n in list(closure):
            closure |= self._closures.get(n, set())
        return _Binding(clinical_event_ids=closure)


def _result(lines, graph=None, gates=()):
    result = CodingResult(encounter_id="enc-1", date_of_service="2026-01-01",
                          lines=list(lines), gates=list(gates))
    if graph is not None:
        result.graph = graph
    return result


class DependencyHoldTest(unittest.TestCase):
    """`autonomy.decide()`'s dependency-entanglement stamping."""

    def test_resolved_line_entangled_with_an_unresolved_one_is_held_not_excluded(self):
        """The required regression: resolved diagnosis REASON_FOR unresolved
        procedure -> one held diagnosis with code/evidence intact, zero
        submission-ready lines, and a release blocker. The diagnosis must not
        disappear."""
        diagnosis = _resolved_line("D1", "DX_A", kind=FactKind.DIAGNOSIS)
        procedure = _unresolved_line("P1")
        graph = _EntangledGraph({"P1": {"D1"}})
        result = _result([diagnosis, procedure], graph=graph)

        autonomy.decide(result)

        self.assertIsNone(diagnosis.excluded_reason,
                          "entanglement is a submission problem, never an invalid "
                          "selection -- excluded_reason must stay unset")
        self.assertIsNotNone(diagnosis.chosen,
                             "the resolved code must survive -- never cleared")
        self.assertEqual(diagnosis.chosen.code, "DX_A")
        self.assertEqual(diagnosis.claim_submission_status, ClaimSubmissionStatus.HELD)
        self.assertIn(diagnosis, result.submission_held_lines)
        self.assertNotIn(diagnosis, result.billable_lines)
        # Zero submission-ready lines, and release is blocked.
        self.assertEqual(result.billable_lines, [])
        self.assertNotEqual(result.destination, Destination.AUTO_READY)
        self.assertNotEqual(result.verdict, Verdict.AUTO_READY)

    def test_the_held_line_routes_as_a_coder_review_item_not_a_provider_query(self):
        """Distinguished from the pre-existing unresolved-actor-ownership HELD
        shape (which IS a provider-answerable question) -- an entangled
        dependency is a coding-judgement item, not something a provider can
        answer."""
        diagnosis = _resolved_line("D1", "DX_A", kind=FactKind.DIAGNOSIS)
        procedure = _unresolved_line("P1")
        graph = _EntangledGraph({"P1": {"D1"}})
        result = _result([diagnosis, procedure], graph=graph)

        autonomy.decide(result)

        item = next(r for r in result.routing if r["subject"] == "service D1")
        self.assertEqual(item["destination"], Destination.REVIEW.value)

    def test_an_unrelated_resolved_line_is_untouched(self):
        """Only genuinely entangled facts are held -- an independent,
        unrelated line stays fully billable."""
        diagnosis = _resolved_line("D1", "DX_A", kind=FactKind.DIAGNOSIS)
        procedure = _unresolved_line("P1")
        unrelated = _resolved_line("D2", "DX_B", kind=FactKind.DIAGNOSIS)
        graph = _EntangledGraph({"P1": {"D1"}})
        result = _result([diagnosis, procedure, unrelated], graph=graph)

        autonomy.decide(result)

        self.assertEqual(unrelated.claim_submission_status, ClaimSubmissionStatus.READY)
        self.assertIn(unrelated, result.billable_lines)

    def test_repeated_decide_calls_do_not_stack_the_hold_reason(self):
        """`decide()` may run more than once per encounter (claim-set
        reconciliation rounds) -- the guard against re-stamping an
        already-HELD line must actually prevent the rationale from growing
        unbounded across rounds."""
        diagnosis = _resolved_line("D1", "DX_A", kind=FactKind.DIAGNOSIS)
        procedure = _unresolved_line("P1")
        graph = _EntangledGraph({"P1": {"D1"}})
        result = _result([diagnosis, procedure], graph=graph)

        autonomy.decide(result)
        first_rationale = diagnosis.rationale
        autonomy.decide(result)
        self.assertEqual(diagnosis.rationale, first_rationale)


class DependencyHoldBundleTest(unittest.TestCase):
    """End-to-end into the real `ClaimBundle` projection."""

    def test_the_held_diagnosis_is_visible_in_the_bundle_as_held_policy_or_data(self):
        from app.contracts.claim_bundle import (AuthorityBinding, EncounterContext,
                                                LineStatus, SourceDocument,
                                                bundle_from_coding_result)
        diagnosis = _resolved_line("D1", "DX_A", kind=FactKind.DIAGNOSIS)
        procedure = _unresolved_line("P1")
        graph = _EntangledGraph({"P1": {"D1"}})
        result = _result([diagnosis, procedure], graph=graph)
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


if __name__ == "__main__":
    unittest.main()
