"""issue #6, Codex's independent re-review, F9-R18-A reopened P1 --
`resolution._apply_attribute_axis_conflict_guard`, the shared post-resolution
finalizer for CLINICAL attribute axis conflicts (`models.AttributeAxisConflict`
/ `ClinicalFact.attribute_axis_conflicts`).

The original F9-R18-A fix checked materiality only inside
`_propose_then_verify_core`, via `tiebreak.discriminating_axes` (which needs
2+ candidates to compare). A direct authoritative/deterministic SINGLETON hit,
whose own descriptor still carries an absolute, unauthorized requirement on
the conflicted axis, closed through `_decide` without ever being checked --
an evidence-unsupported, side-specific release. This file exercises the
finalizer directly against Codex's exact-SHA reproduction shapes, and the
"required regressions" list from the reopened review. Synthetic
codes/descriptors throughout -- no real medical terminology.
"""
import json
import unittest

from claude_coder import models
from claude_coder import resolution as res
from claude_coder.models import (CandidateCode, ClinicalFact, EvidenceSpan, FactKind,
                                 ResolutionMethod, ResolvedLine)


def _cand(code, descriptor):
    return CandidateCode(code=code, system="cpt", descriptor=descriptor, score=0.9,
                         source="retrieval")


def _conflict(axis, value_primary, value_second, question="which is it?"):
    return models.AttributeAxisConflict(
        axis=axis, provider_question=question,
        value_primary=value_primary, value_second=value_second)


def _fact(attributes=None, conflicts=None, evidence=()):
    fact = ClinicalFact(kind=FactKind.PROCEDURE, description="procedure alpha performed",
                        attributes=attributes or {}, evidence=list(evidence),
                        fact_id="F1")
    fact.attribute_axis_conflicts = dict(conflicts or {})
    return fact


def _reconciliation(statuses):
    from app.contracts.source_evidence import (ReconciliationStatus,
                                                SourceReconciliation,
                                                SpanReconciliation)
    return SourceReconciliation(spans=tuple(
        SpanReconciliation(span_id=sid, status=ReconciliationStatus[status])
        for sid, status in statuses.items()))


class _SourceStub:
    def descriptions(self, code, system):
        return []


def _disposition_llm(status, clause, span_tags=(), missing_fact="", raises=False):
    """A minimal LLM callable answering ONLY the structured
    candidate-disposition contract for a single-candidate (option 1) call --
    the shape `_apply_attribute_axis_conflict_guard` actually reads."""
    def stub(system, user):
        if raises:
            raise RuntimeError("stub verifier failure")
        return json.dumps({
            "choice": 1 if status == "entailed" else 0,
            "entailed": [1] if status == "entailed" else [],
            "eliminated": [] if status == "entailed" else
                          [{"option": 1, "reason": "stub"}],
            "reason": "stub",
            "candidate_dispositions": [{
                "option": 1, "status": status, "authority_clause": clause,
                "span_ids": list(span_tags), "missing_fact": missing_fact}]})
    return stub


class MaterialityTest(unittest.TestCase):
    """`resolution._material_axis_conflicts_for` -- singleton-safe materiality,
    never requiring 2+ candidates to compare."""

    def test_a_clause_the_chosen_candidate_states_verbatim_is_material(self):
        cand = _cand("PROC_RIGHT", "Procedure alpha, right side")
        conflicts = {"laterality": _conflict("laterality", "right", "left")}
        out = res._material_axis_conflicts_for(cand, conflicts)
        self.assertEqual([axis for axis, _ in out], ["laterality"])

    def test_a_conflict_absent_from_the_chosen_candidates_own_descriptor_is_not_material(self):
        cand = _cand("PROC_X", "Procedure alpha, each")
        conflicts = {"laterality": _conflict("laterality", "right", "left")}
        self.assertEqual(res._material_axis_conflicts_for(cand, conflicts), [])

    def test_no_conflicts_at_all_is_never_material(self):
        cand = _cand("PROC_RIGHT", "Procedure alpha, right side")
        self.assertEqual(res._material_axis_conflicts_for(cand, {}), [])


class EventScopedRegionTest(unittest.TestCase):
    """`resolution._event_page_region_text` -- bounded to THIS fact's own
    anchored pages, never the whole document (issue #6, Codex's independent
    re-review, F9-R18-A reopened P1: a term belonging to a DIFFERENT event's
    page must never settle THIS event's conflict)."""

    def test_only_this_facts_own_anchored_pages_are_returned(self):
        fact = _fact(evidence=[EvidenceSpan(text="x", span_id="s1", page=2)])
        page_text = {1: "Event one used method alpha.", 2: "Event two was performed."}
        self.assertEqual(res._event_page_region_text(fact, page_text),
                         "Event two was performed.")
        self.assertNotIn("method alpha", res._event_page_region_text(fact, page_text))

    def test_no_page_text_supplied_returns_empty(self):
        fact = _fact(evidence=[EvidenceSpan(text="x", span_id="s1", page=2)])
        self.assertEqual(res._event_page_region_text(fact, None), "")

    def test_no_anchored_page_on_the_fact_returns_empty(self):
        fact = _fact(evidence=[EvidenceSpan(text="x", span_id="s1")])
        self.assertEqual(res._event_page_region_text(fact, {1: "text"}), "")


class GuardTest(unittest.TestCase):
    """`resolution._apply_attribute_axis_conflict_guard` -- the shared
    finalizer itself, applied exactly like `_apply_attribute_evidence_gap_guard`."""

    def _line(self, cand, fact):
        return ResolvedLine(fact=fact, chosen=cand, alternatives=[],
                            method=ResolutionMethod.DETERMINISTIC, rationale="x")

    def test_a_non_material_conflict_never_touches_the_line(self):
        """Required regression: a non-material arbitrary conflict still
        permits release."""
        cand = _cand("PROC_X", "Procedure alpha, each")
        fact = _fact(attributes={"laterality": "right"},
                     conflicts={"laterality": _conflict("laterality", "right", "left")})
        line = self._line(cand, fact)
        out = res._apply_attribute_axis_conflict_guard(
            line, _SourceStub(), None, None, None, None, None)
        self.assertIs(out.chosen, cand)

    def test_direct_singleton_with_no_verifier_is_withdrawn_not_billed(self):
        """Codex's exact-SHA reproduction: a lone "Procedure alpha, right
        side" candidate must not release `method=deterministic` merely
        because there was nothing to tie it against -- and with no verifier
        configured, it stays a visible candidate, never billed."""
        cand = _cand("PROC_RIGHT", "Procedure alpha, right side")
        fact = _fact(attributes={"laterality": "right"},
                     conflicts={"laterality": _conflict("laterality", "right", "left")})
        line = self._line(cand, fact)
        out = res._apply_attribute_axis_conflict_guard(
            line, _SourceStub(), None, None, None, None, None)
        self.assertIsNone(out.chosen)
        self.assertIn(cand, out.alternatives)
        self.assertEqual(out.method, ResolutionMethod.ABSTAINED)

    def test_only_one_evaluator_configured_is_withdrawn_not_billed(self):
        cand = _cand("PROC_RIGHT", "Procedure alpha, right side")
        fact = _fact(attributes={"laterality": "right"},
                     conflicts={"laterality": _conflict("laterality", "right", "left")},
                     evidence=[EvidenceSpan(text="x", span_id="s1")])
        line = self._line(cand, fact)
        llm = _disposition_llm("entailed", "right side", ["e1"])
        out = res._apply_attribute_axis_conflict_guard(
            line, _SourceStub(), llm, None, _reconciliation({"s1": "AGREED"}), None, None)
        self.assertIsNone(out.chosen)
        self.assertIn(cand, out.alternatives)

    def test_verified_singleton_releases_with_cited_evidence_from_both_evaluators(self):
        """Required regression: a verified singleton releases only with
        source-reconciled cited evidence."""
        cand = _cand("PROC_RIGHT", "Procedure alpha, right side")
        fact = _fact(attributes={"laterality": "right"},
                     conflicts={"laterality": _conflict("laterality", "right", "left")},
                     evidence=[EvidenceSpan(text="x", span_id="s1")])
        line = self._line(cand, fact)
        llm = _disposition_llm("entailed", "right side", ["e1"])
        corroborate = _disposition_llm("entailed", "right side", ["e1"])
        out = res._apply_attribute_axis_conflict_guard(
            line, _SourceStub(), llm, corroborate, _reconciliation({"s1": "AGREED"}),
            None, None)
        self.assertIs(out.chosen, cand)

    def test_entailed_without_a_validated_cited_span_does_not_release(self):
        cand = _cand("PROC_RIGHT", "Procedure alpha, right side")
        fact = _fact(attributes={"laterality": "right"},
                     conflicts={"laterality": _conflict("laterality", "right", "left")},
                     evidence=[EvidenceSpan(text="x", span_id="s1")])
        line = self._line(cand, fact)
        # No span_ids cited at all -- a bare "entailed" status, never sufficient.
        llm = _disposition_llm("entailed", "right side", [])
        corroborate = _disposition_llm("entailed", "right side", [])
        out = res._apply_attribute_axis_conflict_guard(
            line, _SourceStub(), llm, corroborate, _reconciliation({"s1": "AGREED"}),
            None, None)
        self.assertIsNone(out.chosen)

    def test_both_evaluators_contradict_the_singleton_withdraws_it(self):
        cand = _cand("PROC_RIGHT", "Procedure alpha, right side")
        fact = _fact(attributes={"laterality": "right"},
                     conflicts={"laterality": _conflict("laterality", "right", "left")},
                     evidence=[EvidenceSpan(text="x", span_id="s1")])
        line = self._line(cand, fact)
        llm = _disposition_llm("contradicted", "right side", ["e1"])
        corroborate = _disposition_llm("contradicted", "right side", ["e1"])
        out = res._apply_attribute_axis_conflict_guard(
            line, _SourceStub(), llm, corroborate, _reconciliation({"s1": "AGREED"}),
            None, None)
        self.assertIsNone(out.chosen)

    def test_both_not_documented_and_page_region_silent_yields_candidates_needing_fact(self):
        cand = _cand("PROC_RIGHT", "Procedure alpha, right side")
        fact = _fact(attributes={"laterality": "right"},
                     conflicts={"laterality": _conflict(
                         "laterality", "right", "left", question="which side?")},
                     evidence=[EvidenceSpan(text="x", span_id="s1", page=1)])
        line = self._line(cand, fact)
        llm = _disposition_llm("not_documented", "right side", [],
                               missing_fact="the operative side")
        corroborate = _disposition_llm("not_documented", "right side", [],
                                       missing_fact="the operative side")

        class _Coverage:
            complete = True

        page_text = {1: "The procedure was performed without stating a side."}
        out = res._apply_attribute_axis_conflict_guard(
            line, _SourceStub(), llm, corroborate, _reconciliation({}), _Coverage(),
            page_text)
        self.assertIsNone(out.chosen)
        self.assertEqual(out.documentation_gap, "which side?")

    def test_both_not_documented_but_the_page_region_confirms_it_authorizes(self):
        """The document itself asserts the value somewhere on this event's own
        anchored page, but extraction/reconciliation never bound it as a
        cited quotation -- the bounded, event-scoped reconciliation step
        finds it and authorizes, rather than treating it as silence."""
        cand = _cand("PROC_RIGHT", "Procedure alpha, right side")
        fact = _fact(attributes={"laterality": "right"},
                     conflicts={"laterality": _conflict("laterality", "right", "left")},
                     evidence=[EvidenceSpan(text="x", span_id="s1", page=1)])
        line = self._line(cand, fact)
        llm = _disposition_llm("not_documented", "right side", [],
                               missing_fact="the operative side")
        corroborate = _disposition_llm("not_documented", "right side", [],
                                       missing_fact="the operative side")

        class _Coverage:
            complete = True

        page_text = {1: "The procedure was performed on the right side today."}
        out = res._apply_attribute_axis_conflict_guard(
            line, _SourceStub(), llm, corroborate, _reconciliation({}), _Coverage(),
            page_text)
        self.assertIs(out.chosen, cand)

    def test_a_term_from_another_events_page_never_settles_this_conflict(self):
        """Required regression: a term belonging to another event in the
        full corpus cannot classify this event as resolved or as a system
        defect -- only THIS event's own anchored page(s) may."""
        cand = _cand("PROC_RIGHT", "Procedure alpha, right side")
        fact = _fact(attributes={"laterality": "right"},
                     conflicts={"laterality": _conflict("laterality", "right", "left")},
                     evidence=[EvidenceSpan(text="x", span_id="s1", page=2)])
        line = self._line(cand, fact)
        llm = _disposition_llm("not_documented", "right side", [],
                               missing_fact="the operative side")
        corroborate = _disposition_llm("not_documented", "right side", [],
                                       missing_fact="the operative side")

        class _Coverage:
            complete = True

        # The "right side" wording lives on a DIFFERENT event's page (1), not
        # this fact's own anchored page (2).
        page_text = {1: "A different procedure was performed on the right side.",
                    2: "This procedure was performed."}
        out = res._apply_attribute_axis_conflict_guard(
            line, _SourceStub(), llm, corroborate, _reconciliation({}), _Coverage(),
            page_text)
        self.assertIsNone(out.chosen)
        self.assertIsNotNone(out.documentation_gap)

    def test_a_verifier_call_that_raises_yields_a_scoped_system_hold(self):
        cand = _cand("PROC_RIGHT", "Procedure alpha, right side")
        fact = _fact(attributes={"laterality": "right"},
                     conflicts={"laterality": _conflict("laterality", "right", "left")},
                     evidence=[EvidenceSpan(text="x", span_id="s1")])
        line = self._line(cand, fact)
        llm = _disposition_llm("entailed", "right side", ["e1"], raises=True)
        out = res._apply_attribute_axis_conflict_guard(
            line, _SourceStub(), llm, None, _reconciliation({"s1": "AGREED"}), None, None)
        self.assertIsNone(out.chosen)
        self.assertIn("SYSTEM ERROR, retryable", out.rationale)

    def test_already_authorized_via_claim_authorized_value_skips_verification(self):
        """An axis a genuinely ASSERTED, reconciled `attribute_evidence` entry
        already authorizes (`graph_consensus.claim_authorized_value`) never
        needs re-verification through this guard at all -- release stands,
        with no verifier even configured."""
        from claude_coder.models import AttributeEvidence, RelationState
        span = EvidenceSpan(text="right side stated", span_id="s1", anchored=True)
        fact = _fact(attributes={"laterality": "right"},
                     conflicts={"laterality": _conflict("laterality", "right", "left")},
                     evidence=[span])
        fact.attribute_evidence = {"laterality": (
            AttributeEvidence(span=span, scope="local",
                              assertion_state=RelationState.ASSERTED, value="right"),)}
        cand = _cand("PROC_RIGHT", "Procedure alpha, right side")
        line = self._line(cand, fact)
        out = res._apply_attribute_axis_conflict_guard(
            line, _SourceStub(), None, None, _reconciliation({"s1": "AGREED"}), None, None)
        self.assertIs(out.chosen, cand)

    def test_no_chosen_candidate_is_a_no_op(self):
        fact = _fact(conflicts={"laterality": _conflict("laterality", "right", "left")})
        line = ResolvedLine(fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
                            rationale="held")
        out = res._apply_attribute_axis_conflict_guard(
            line, _SourceStub(), None, None, None, None, None)
        self.assertIs(out, line)


if __name__ == "__main__":
    unittest.main()
