"""issue #6, Codex's independent re-review (F9-R22-A, hardened by the
F9-R23 clarification "Gate A"): the candidate-disposition citation contract
is now schema-enforced (`SHORTLIST_JUDGEMENT_SCHEMA`) and, when an answer's
`candidate_dispositions` still fail the citation bar (a missing entry, an
uncited entailed/contradicted/different_concept disposition, an unnamed
not_documented `missing_fact`, or a cited span with no genuine content
relationship to the candidate), `verify.select_entailed`/`corroborate` make
ONE bounded same-evaluator repair call before falling back to dropping just
the defective candidate(s). `_agreed_citable_spans` is the shared bar
itself: a cited span counts only when it is genuine target-event evidence
for the fact, reconciled AGREED -- never VACUOUS (punctuation/whitespace
cannot substantively support a disposition), never a span reconciled
elsewhere in the document but irrelevant to this fact -- AND genuinely
content-related to the candidate it is cited for (never a real, anchored,
AGREED span that simply talks about something else).

Synthetic facts/codes throughout.
"""
import json
import unittest

from app.contracts.source_evidence import (ReconciliationStatus, SourceReconciliation,
                                            SpanReconciliation)
from claude_coder import verify
from claude_coder.models import CandidateCode, ClinicalFact, EvidenceSpan, FactKind


def _fact(fact_id="F1", description="assembly service",
         span_text="assembly performed", span_id="s1"):
    return ClinicalFact(kind=FactKind.PROCEDURE, description=description,
                        confidence=0.9, fact_id=fact_id,
                        evidence=[EvidenceSpan(span_text, anchored=True, span_id=span_id)])


def _cand(code, descriptor=None):
    return CandidateCode(code=code, system="cpt",
                         descriptor=descriptor or f"{code} assembly descriptor",
                         score=0.9, source="retrieval")


def _reconciliation(statuses: dict) -> SourceReconciliation:
    return SourceReconciliation(spans=tuple(
        SpanReconciliation(span_id=sid, status=ReconciliationStatus[status])
        for sid, status in statuses.items()))


def _disposition(option, status, digest, span_tags=(), missing_fact=""):
    return {"option": option, "status": status, "descriptor_sha256": digest,
           "span_ids": list(span_tags), "missing_fact": missing_fact}


def _answer(choice, dispositions, eliminated=()):
    return {"choice": choice, "entailed": [choice] if choice else [], "reason": "",
           "eliminated": list(eliminated), "candidate_dispositions": dispositions,
           "requirements": []}


class ScriptedLLM:
    """Returns successive JSON answers for successive calls -- scripts the
    FIRST (possibly defective) answer and the bounded REPAIR answer
    separately, and records every (system, user) pair it was called with."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return json.dumps(self._answers[len(self.calls) - 1])


class AgreedCitableSpansTest(unittest.TestCase):
    """`verify._agreed_citable_spans` -- the shared citation bar."""

    def test_a_genuinely_agreed_citable_span_counts(self):
        fact = _fact()
        cand = _cand("ALPHA")
        reconciliation = _reconciliation({"s1": "AGREED"})
        self.assertEqual(
            verify._agreed_citable_spans(("s1",), fact, cand, reconciliation), ("s1",))

    def test_an_irrelevant_agreed_span_does_not_count(self):
        """A span reconciled AGREED elsewhere in the document, but that is
        not this fact's own target-event evidence, must not count."""
        fact = _fact()
        cand = _cand("ALPHA")
        reconciliation = _reconciliation({"s-unrelated": "AGREED"})
        self.assertEqual(
            verify._agreed_citable_spans(("s-unrelated",), fact, cand, reconciliation), ())

    def test_a_vacuous_span_does_not_count(self):
        """Punctuation/whitespace cannot substantively support a disposition
        -- VACUOUS is never enough, even for the fact's own real span."""
        fact = _fact()
        cand = _cand("ALPHA")
        reconciliation = _reconciliation({"s1": "VACUOUS"})
        self.assertEqual(verify._agreed_citable_spans(("s1",), fact, cand, reconciliation), ())

    def test_no_reconciliation_means_nothing_counts(self):
        fact = _fact()
        cand = _cand("ALPHA")
        self.assertEqual(verify._agreed_citable_spans(("s1",), fact, cand, None), ())

    def test_an_attached_but_semantically_unrelated_span_does_not_count(self):
        """issue #6, Codex's independent re-review (F9-R23 clarification,
        "Gate A"): the mandatory regression. A span that IS the fact's own,
        genuinely anchored, reconciled AGREED evidence -- so it clears
        every location/membership/status check -- must still fail to
        validate a disposition when its own TEXT shares nothing with
        either the candidate's descriptor or the fact's description.
        Location/reconciliation proof is not content proof."""
        fact = _fact(description="assembly service",
                     span_text="patient reports no known drug allergies")
        cand = _cand("ALPHA", descriptor="ALPHA assembly descriptor")
        reconciliation = _reconciliation({"s1": "AGREED"})
        self.assertEqual(verify._agreed_citable_spans(("s1",), fact, cand, reconciliation), ())

    def test_a_negated_agreed_span_does_not_count(self):
        """Location and vocabulary cannot turn an explicitly negated event
        into support for a candidate."""
        fact = _fact(description="assembly service",
                     span_text="No assembly service was performed")
        cand = _cand("ALPHA", descriptor="assembly service")
        reconciliation = _reconciliation({"s1": "AGREED"})
        self.assertEqual(
            verify._agreed_citable_spans(("s1",), fact, cand, reconciliation), ())

    def test_positive_core_word_cannot_hide_a_negated_candidate_qualifier(self):
        """A citation preserves polarity for candidate-specific content,
        rather than passing because some generic shared token is positive."""
        fact = _fact(
            description="assembly service",
            span_text="Assembly service was performed, but not in mode alpha")
        cand = _cand("ALPHA", descriptor="assembly service, mode alpha")
        reconciliation = _reconciliation({"s1": "AGREED"})

        self.assertEqual(
            verify._agreed_citable_spans(("s1",), fact, cand, reconciliation), ())

    def test_descriptor_declared_exception_preserves_valid_negative_support(self):
        """Negating a descriptor's own exception is compatible with the
        candidate and must not be mistaken for negating the service."""
        fact = _fact(
            description="assembly service",
            span_text="Assembly service was performed; branch gamma was not used")
        cand = _cand("ALPHA", descriptor="assembly service, except branch gamma")
        reconciliation = _reconciliation({"s1": "AGREED"})

        self.assertEqual(
            verify._agreed_citable_spans(("s1",), fact, cand, reconciliation),
            ("s1",))


class ValidateJudgementContractTest(unittest.TestCase):
    """`verify.validate_judgement_contract` -- structural defect detection."""

    def test_a_fully_cited_entailed_disposition_has_no_defects(self):
        cand = _cand("ALPHA")
        fact = _fact()
        reconciliation = _reconciliation({"s1": "AGREED"})
        digest = verify._descriptor_sha256(cand)
        ans = _answer(1, [_disposition(1, "entailed", digest, ["e1"])])
        llm = ScriptedLLM([ans])
        j = verify.select_entailed(fact, [cand], None, llm, force_disposition=True,
                                   reconciliation=reconciliation)
        self.assertEqual(verify.validate_judgement_contract(j, [cand], fact, reconciliation),
                         {})
        self.assertEqual(len(llm.calls), 1, "a structurally valid answer needs no repair")

    def test_a_missing_disposition_entry_is_a_defect(self):
        cand = _cand("ALPHA")
        fact = _fact()
        j = verify._judgement({"candidate_dispositions": []}, [cand])
        defects = verify.validate_judgement_contract(j, [cand], fact, _reconciliation({}))
        self.assertIn("ALPHA", defects)

    def test_not_documented_without_missing_fact_is_a_defect(self):
        cand = _cand("ALPHA")
        fact = _fact()
        digest = verify._descriptor_sha256(cand)
        j = verify._judgement(_answer(0, [_disposition(1, "not_documented", digest)]), [cand])
        defects = verify.validate_judgement_contract(j, [cand], fact, _reconciliation({}))
        self.assertIn("ALPHA", defects)

    def test_not_documented_with_a_missing_fact_has_no_defect(self):
        cand = _cand("ALPHA")
        fact = _fact()
        digest = verify._descriptor_sha256(cand)
        j = verify._judgement(
            _answer(0, [_disposition(1, "not_documented", digest,
                                     missing_fact="laterality")]), [cand])
        self.assertEqual(
            verify.validate_judgement_contract(j, [cand], fact, _reconciliation({})), {})


class RepairLoopTest(unittest.TestCase):
    """`select_entailed`/`corroborate`'s bounded same-evaluator repair call."""

    def test_omitted_citation_triggers_one_bounded_repair_and_releases_if_repaired(self):
        cand = _cand("ALPHA")
        fact = _fact()
        reconciliation = _reconciliation({"s1": "AGREED"})
        digest = verify._descriptor_sha256(cand)
        defective = _answer(1, [_disposition(1, "entailed", digest, [])])
        fixed = _answer(1, [_disposition(1, "entailed", digest, ["e1"])])
        llm = ScriptedLLM([defective, fixed])

        j = verify.select_entailed(fact, [cand], None, llm, force_disposition=True,
                                   reconciliation=reconciliation)

        self.assertEqual(len(llm.calls), 2, "exactly one bounded repair call")
        self.assertEqual(j.candidate_dispositions[0].evidence_span_ids, ("s1",))
        self.assertEqual(
            verify.validate_judgement_contract(j, [cand], fact, reconciliation), {})

    def test_a_valid_disposition_never_triggers_a_repair_call(self):
        cand = _cand("ALPHA")
        fact = _fact()
        reconciliation = _reconciliation({"s1": "AGREED"})
        digest = verify._descriptor_sha256(cand)
        ans = _answer(1, [_disposition(1, "entailed", digest, ["e1"])])
        llm = ScriptedLLM([ans])
        verify.select_entailed(fact, [cand], None, llm, force_disposition=True,
                               reconciliation=reconciliation)
        self.assertEqual(len(llm.calls), 1)

    def test_failed_repair_leaves_only_the_affected_candidate_unvalidated(self):
        """The required regression: a failed repair holds only the one
        candidate that never cleared the citation bar -- a sibling
        candidate's own, already-valid disposition is untouched. The
        still-defective disposition is NOT silently dropped (that would
        make this judgement look like it never addressed ALPHA at all,
        deferring `_candidate_disposition_uniqueness` for the WHOLE
        shortlist back to the weaker legacy path) -- it stays present, but
        genuinely fails to validate."""
        alpha, beta = _cand("ALPHA"), _cand("BETA")
        fact = _fact()
        reconciliation = _reconciliation({"s1": "AGREED"})
        d_alpha, d_beta = verify._descriptor_sha256(alpha), verify._descriptor_sha256(beta)
        elim = [{"option": 2, "reason": "different concept", "missing_element": False}]
        defective = _answer(1, [
            _disposition(1, "entailed", d_alpha, []),           # never cites anything
            _disposition(2, "different_concept", d_beta, ["e1"])], eliminated=elim)
        # the repair call still omits ALPHA's citation -- only ALPHA stays defective
        still_defective = _answer(1, [
            _disposition(1, "entailed", d_alpha, []),
            _disposition(2, "different_concept", d_beta, ["e1"])], eliminated=elim)
        llm = ScriptedLLM([defective, still_defective])

        j = verify.select_entailed(fact, [alpha, beta], None, llm,
                                   reconciliation=reconciliation)

        by_code = {d.candidate_code: d for d in j.candidate_dispositions}
        self.assertIn("ALPHA", by_code, "still answered for, just not validly")
        self.assertEqual(
            verify._agreed_citable_spans(by_code["ALPHA"].evidence_span_ids, fact, alpha,
                                         reconciliation), (),
            "ALPHA's citation never cleared the bar, even after repair")
        self.assertEqual(
            verify._agreed_citable_spans(by_code["BETA"].evidence_span_ids, fact, beta,
                                         reconciliation), ("s1",),
            "BETA's own already-valid disposition is untouched")

    def test_evaluator_origin_survives_repair(self):
        cand = _cand("ALPHA")
        fact = _fact()
        reconciliation = _reconciliation({"s1": "AGREED"})
        digest = verify._descriptor_sha256(cand)
        defective = _answer(1, [_disposition(1, "entailed", digest, [])])
        fixed = _answer(1, [_disposition(1, "entailed", digest, ["e1"])])
        llm = ScriptedLLM([defective, fixed])

        j = verify.corroborate(fact, [cand], None, llm, force_disposition=True,
                               reconciliation=reconciliation)

        self.assertEqual(j.candidate_dispositions[0].evaluator_origin,
                         {"provider": verify.CORROBORATE_PROVIDER})

    def test_no_reconciliation_skips_the_repair_call_entirely(self):
        """A repair call could never fix an unfixably-absent reconciliation
        -- skipped outright rather than wasting a call that can only fail
        again the same way."""
        cand = _cand("ALPHA")
        fact = _fact()
        digest = verify._descriptor_sha256(cand)
        ans = _answer(1, [_disposition(1, "entailed", digest, [])])
        llm = ScriptedLLM([ans])
        verify.select_entailed(fact, [cand], None, llm, force_disposition=True,
                               reconciliation=None)
        self.assertEqual(len(llm.calls), 1)


class GateAEndToEndTest(unittest.TestCase):
    """issue #6, Codex's independent re-review (F9-R23 clarification, "Gate
    A"): the full mandatory regression, end to end through
    `resolution._candidate_disposition_uniqueness` -- both evaluators
    return STRUCTURALLY PERFECT, independently corroborated JSON (correct
    schema, correct descriptor hash, a real, anchored, reconciled AGREED
    span), and that span is still semantically unrelated to the candidate.
    Expected: no SUPPORTED standing, no selected code -- and this is a
    SYSTEM gap (`system_unresolved`), never a positive elimination and
    never grounds for a provider question, since the record was never
    actually read as saying anything against the candidate either."""

    def test_a_structurally_perfect_but_semantically_irrelevant_citation_never_releases(self):
        from claude_coder import resolution as res

        cand = _cand("ALPHA", descriptor="ALPHA assembly descriptor")
        fact = _fact(description="assembly service",
                     span_text="patient reports no known drug allergies")
        reconciliation = _reconciliation({"s1": "AGREED"})
        digest = verify._descriptor_sha256(cand)
        # Both evaluators independently cite the SAME real, anchored, AGREED
        # span for this fact -- structurally flawless, just semantically
        # unrelated to the candidate.
        ans = _answer(1, [_disposition(1, "entailed", digest, ["e1"])])
        # Two answers each: the repair attempt reasserts the same (still
        # unrelated) citation -- repair cannot manufacture a relationship
        # the source text does not have.
        llm0 = ScriptedLLM([ans, ans])
        llm1 = ScriptedLLM([ans, ans])

        j0 = verify.select_entailed(fact, [cand], None, llm0, force_disposition=True,
                                    reconciliation=reconciliation)
        j1 = verify.corroborate(fact, [cand], None, llm1, force_disposition=True,
                                reconciliation=reconciliation)

        # Neither evaluator's own repair loop found a fix -- the span itself
        # never relates to the candidate, so repair cannot manufacture one.
        self.assertEqual(len(llm0.calls), 2, "the unrelated citation triggers one bounded "
                                             "repair attempt, which cannot fix it")

        remaining, eliminated, system_unresolved = res._candidate_disposition_uniqueness(
            [cand], cand, [j0, j1], reconciliation, None, fact=fact)
        self.assertNotIn(cand, remaining, "no code releases on an irrelevant citation")
        self.assertEqual(eliminated, {}, "an unrelated citation disproves nothing -- it is "
                                         "a system evidence gap, not a validated rejection")
        self.assertIn("ALPHA", system_unresolved)


if __name__ == "__main__":
    unittest.main()
