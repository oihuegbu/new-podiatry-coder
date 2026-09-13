"""issue #6, Codex's independent re-review (F9-R19-A consolidated live-run
remediation, Findings 1 and 2): "not successfully eliminated != positively
supported" -- `resolution._candidate_disposition_uniqueness` now returns a
genuine THREE-way `(remaining, eliminated, system_unresolved)` split instead
of silently folding evaluator disagreement, an unreproduced descriptor
identity, an uncited contradiction, or an unvalidated not_documented verdict
into the SAME bucket as a candidate both evaluators genuinely, validly call
"entailed". A qualified-child family's shared stem is now ALSO checked as a
required, unconditional `family_viability` precondition (Finding 2) before
its differential (suffix) clause is ever asked about.

Synthetic codes/descriptors/facts throughout.
"""
import hashlib
import unittest

from app.contracts.source_evidence import (ReconciliationStatus, SourceReconciliation,
                                            SpanReconciliation)
from claude_coder import requirement as req
from claude_coder import resolution
from claude_coder import verify as _verify
from claude_coder.models import CandidateCode, ClinicalFact, EvidenceSpan, FactKind
from claude_coder.models import ClaimSubmissionStatus


def _cand(code, descriptor):
    return CandidateCode(code=code, system="cpt", descriptor=descriptor, score=0.9,
                         source="retrieval")


def _fact(description="assembly service performed"):
    span = EvidenceSpan(text=description, anchored=True, span_id="s1")
    return ClinicalFact(kind=FactKind.PROCEDURE, description=description,
                        evidence=[span], confidence=0.9, fact_id="F1")


def _disp(cand, status, evidence_span_ids=(), missing_fact=""):
    return _verify.CandidateDispositionEvidence(
        candidate_code=cand.code, descriptor_sha256=_verify._descriptor_sha256(cand),
        status=status, evidence_span_ids=tuple(evidence_span_ids), missing_fact=missing_fact)


def _judgement(dispositions, requirement_judgements=()):
    return _verify.Judgement(candidate_dispositions=tuple(dispositions),
                             requirement_judgements=tuple(requirement_judgements),
                             declared=True)


def _reconciliation(statuses):
    return SourceReconciliation(spans=tuple(
        SpanReconciliation(span_id=sid, status=ReconciliationStatus[status])
        for sid, status in statuses.items()))


def _coverage(text):
    return req.CoverageCorpus(channel_id="test-channel", text=text,
                              text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                              covered_pages=(1,), page_image_sha256=("stub-hash",))


class DispositionClassificationTest(unittest.TestCase):
    """`_candidate_disposition_uniqueness`'s core three-way split, tested
    directly against two synthetic, unrelated candidates (no family
    structure) -- Finding 1's invariant in isolation."""

    A = _cand("CAND_A", "assembly service, type A")
    B = _cand("CAND_B", "assembly service, type B")

    def test_evaluator_disagreement_is_system_unresolved_never_entailed(self):
        j0 = _judgement([_disp(self.A, "entailed"), _disp(self.B, "not_documented",
                                                          missing_fact="type B stated")])
        j1 = _judgement([_disp(self.A, "contradicted", evidence_span_ids=("s1",)),
                         _disp(self.B, "not_documented", missing_fact="type B stated")])
        remaining, eliminated, system_unresolved = resolution._candidate_disposition_uniqueness(
            [self.A, self.B], self.A, [j0, j1], reconciliation=None, coverage=None)
        self.assertIn("CAND_A", system_unresolved)
        self.assertNotIn("CAND_A", remaining_codes := [c.code for c in remaining])
        self.assertNotIn("CAND_A", eliminated)

    def test_an_unreproduced_identity_is_system_unresolved(self):
        bad = _verify.CandidateDispositionEvidence(
            candidate_code="CAND_A", descriptor_sha256="0" * 64, status="entailed")
        j0 = _judgement([bad, _disp(self.B, "entailed")])
        j1 = _judgement([_disp(self.A, "entailed"), _disp(self.B, "entailed")])
        remaining, eliminated, system_unresolved = resolution._candidate_disposition_uniqueness(
            [self.A, self.B], self.A, [j0, j1], reconciliation=None, coverage=None)
        self.assertIn("CAND_A", system_unresolved)
        self.assertNotIn("CAND_A", eliminated)

    def test_an_uncited_contradiction_is_system_unresolved_not_eliminated(self):
        j0 = _judgement([_disp(self.A, "contradicted"), _disp(self.B, "entailed")])
        j1 = _judgement([_disp(self.A, "contradicted"), _disp(self.B, "entailed")])
        remaining, eliminated, system_unresolved = resolution._candidate_disposition_uniqueness(
            [self.A, self.B], self.A, [j0, j1], reconciliation=None, coverage=None)
        self.assertIn("CAND_A", system_unresolved)
        self.assertNotIn("CAND_A", eliminated,
                         "an uncited contradiction is not validated proof -- it must "
                         "not be treated as a grounded elimination")

    def test_a_validated_contradiction_is_cleanly_eliminated(self):
        j0 = _judgement([_disp(self.A, "contradicted", evidence_span_ids=("s1",)),
                         _disp(self.B, "entailed")])
        j1 = _judgement([_disp(self.A, "contradicted", evidence_span_ids=("s1",)),
                         _disp(self.B, "entailed")])
        remaining, eliminated, system_unresolved = resolution._candidate_disposition_uniqueness(
            [self.A, self.B], self.A, [j0, j1],
            reconciliation=_reconciliation({"s1": "AGREED"}), coverage=None)
        self.assertIn("CAND_A", eliminated)
        self.assertNotIn("CAND_A", system_unresolved)
        self.assertEqual([c.code for c in remaining], ["CAND_B"])

    def test_not_documented_without_complete_coverage_is_system_unresolved(self):
        j0 = _judgement([_disp(self.A, "not_documented", missing_fact="type A stated"),
                         _disp(self.B, "entailed")])
        j1 = _judgement([_disp(self.A, "not_documented", missing_fact="type A stated"),
                         _disp(self.B, "entailed")])
        remaining, eliminated, system_unresolved = resolution._candidate_disposition_uniqueness(
            [self.A, self.B], self.A, [j0, j1], reconciliation=None, coverage=None)
        self.assertIn("CAND_A", system_unresolved)
        self.assertNotIn("CAND_A", eliminated)

    def test_both_genuinely_entailed_with_no_contract_stays_remaining(self):
        j0 = _judgement([_disp(self.A, "entailed"), _disp(self.B, "entailed")])
        j1 = _judgement([_disp(self.A, "entailed"), _disp(self.B, "entailed")])
        remaining, eliminated, system_unresolved = resolution._candidate_disposition_uniqueness(
            [self.A, self.B], self.A, [j0, j1], reconciliation=None, coverage=None)
        self.assertEqual(sorted(c.code for c in remaining), ["CAND_A", "CAND_B"])
        self.assertEqual(eliminated, {})
        self.assertEqual(system_unresolved, {})


class FamilyViabilityContractTest(unittest.TestCase):
    """Finding 2: a qualified-child family's shared stem is a REQUIRED
    precondition, checked UNCONDITIONALLY (never gated behind a model's own
    free-text elimination reason) via the same requirement-grounding
    machinery every other MUST_SUPPORT axis uses."""

    # Non-numeric differentiators deliberately: a numeric clause ("less than
    # 3 cm") would ALSO parse as a separate `measurement` axis (`provable=
    # False` by design -- a numeric interval needs a typed, unit-converted
    # comparison, never a word match), which permanently stays `unsettled`
    # and would block `narrow()`'s winner decision regardless of this test's
    # own concern. These mirror the REAL scenario the designated operative
    # note exercised (a bone-graft family differentiated by graft source).
    SMALL = _cand("CAND_SMALL", "excision of lesion, subcutaneous; with allograft")
    LARGE = _cand("CAND_LARGE", "excision of lesion, subcutaneous; with autograft")

    def test_an_undocumented_shared_family_eliminates_both_siblings(self):
        """Both children entailed at the disposition layer, but the shared
        stem ("excision of lesion, subcutaneous") is validated NOT_DOCUMENTED
        by both evaluators -- neither sibling may survive, and no
        differential (graft/size) question is ever built from them."""
        fact = _fact("an unrelated procedure was performed")
        reqs = req.compile_requirements([self.SMALL, self.LARGE])
        viability_reqs = [r for r in reqs if r.axis == "family_viability"]
        self.assertEqual(len(viability_reqs), 2)   # one per sibling, same clause
        rjs = tuple(req.RequirementJudgement(requirement_id=r.requirement_id,
                                             status=req.RequirementStatus.NOT_DOCUMENTED)
                   for r in reqs)
        j0 = _judgement([_disp(self.SMALL, "entailed"), _disp(self.LARGE, "entailed")], rjs)
        j1 = _judgement([_disp(self.SMALL, "entailed"), _disp(self.LARGE, "entailed")], rjs)
        coverage = _coverage("an unrelated procedure was performed")
        remaining, eliminated, system_unresolved = resolution._candidate_disposition_uniqueness(
            [self.SMALL, self.LARGE], self.SMALL, [j0, j1],
            reconciliation=None, coverage=coverage, fact=fact, requirements=reqs)
        self.assertEqual(remaining, [])
        self.assertIn("CAND_SMALL", eliminated)
        self.assertIn("CAND_LARGE", eliminated)

    def test_documented_family_with_undocumented_differential_leaves_both_standing(self):
        """The shared stem IS documented, but neither sibling's own suffix
        (the differential) is -- both remain, which is exactly what must
        turn into ONE precise provider question ("less than 3cm vs 3cm or
        greater"), never a silent pick."""
        text = "excision of lesion, subcutaneous, performed today"
        fact = _fact(text)
        reqs = req.compile_requirements([self.SMALL, self.LARGE])
        rjs = []
        for r in reqs:
            status = (req.RequirementStatus.SUPPORTED if r.axis == "family_viability"
                     else req.RequirementStatus.NOT_DOCUMENTED)
            rjs.append(req.RequirementJudgement(
                requirement_id=r.requirement_id, status=status,
                evidence_span_ids=(("s1",) if status is req.RequirementStatus.SUPPORTED
                                   else ())))
        j0 = _judgement([_disp(self.SMALL, "entailed"), _disp(self.LARGE, "entailed")], rjs)
        j1 = _judgement([_disp(self.SMALL, "entailed"), _disp(self.LARGE, "entailed")], rjs)
        coverage = _coverage(text)
        reconciliation = _reconciliation({"s1": "AGREED"})
        span = EvidenceSpan(text=text, anchored=True, span_id="s1")
        fact.evidence = [span]
        remaining, eliminated, system_unresolved = resolution._candidate_disposition_uniqueness(
            [self.SMALL, self.LARGE], self.SMALL, [j0, j1],
            reconciliation=reconciliation, coverage=coverage, fact=fact, requirements=reqs)
        self.assertEqual(sorted(c.code for c in remaining), ["CAND_LARGE", "CAND_SMALL"])
        self.assertEqual(eliminated, {})

    def test_a_documented_differential_selects_through_narrows_existing_logic(self):
        """Once `_candidate_disposition_uniqueness` clears the family's
        VIABILITY (this test's own concern is the layer above it -- see
        `test_documented_family_with_undocumented_differential_leaves_both_
        standing`), the DIFFERENTIAL itself resolves entirely through
        `tiebreak.narrow`'s EXISTING, already-proven positive-presence
        winner logic -- deliberately never duplicated in the new contract
        check (see that check's own comment: eliminating on an undocumented
        differential for EVERY sibling would convert "please specify which"
        into "none of these apply"). Only the sibling whose OWN clause is
        source-confirmed wins; the other is correctly left out."""
        from claude_coder.models import ClinicalFact as _CF
        from claude_coder import tiebreak
        text = "excision of lesion, subcutaneous, with allograft, performed today"
        span = EvidenceSpan(text=text, anchored=True, span_id="span-0")
        fact = _CF(kind=FactKind.PROCEDURE, description=text, evidence=[span],
                   confidence=0.99, fact_id="F1")
        reconciliation = _reconciliation({"span-0": "AGREED"})
        outcome = tiebreak.narrow(fact, [self.SMALL, self.LARGE], reconciliation)
        self.assertEqual(outcome.winner.code if outcome.winner else None, "CAND_SMALL")


class SettleUniquenessSystemHoldTest(unittest.TestCase):
    """`_settle_uniqueness` integration: a system-unresolved sibling must
    never block a genuinely clean release, and a shortlist that cannot
    resolve at all because of system-unresolved candidates must route to a
    retryable system hold, never a provider query."""

    def test_a_system_unresolved_rival_does_not_block_a_clean_release(self):
        chosen = _cand("CAND_CHOSEN", "assembly service, clean")
        rival = _cand("CAND_RIVAL", "assembly service, rival")
        fact = _fact("assembly service, clean, performed today")
        # Legacy judgement layer: only `chosen` is named entailed by both;
        # `rival`'s disposition will disagree between evaluators, so it must
        # land in system_unresolved, not silently count as a live rival.
        j0 = _verify.Judgement(
            chosen=chosen, entailed=("CAND_CHOSEN",), declared=True,
            candidate_dispositions=(_disp(chosen, "entailed"),
                                    _disp(rival, "entailed")))
        j1 = _verify.Judgement(
            chosen=chosen, entailed=("CAND_CHOSEN",), declared=True,
            candidate_dispositions=(_disp(chosen, "entailed"),
                                    _disp(rival, "contradicted")))
        line = resolution._settle_uniqueness(
            fact, chosen, [chosen, rival], [j0, j1], {}, "entailed", "", None)
        self.assertEqual(line.chosen.code, "CAND_CHOSEN")

    def test_system_unresolved_candidates_route_to_a_retryable_hold_not_a_provider_query(self):
        from claude_coder.models import SYSTEM_UNRESOLVED_MARKER
        a = _cand("CAND_A", "assembly service, type A")
        b = _cand("CAND_B", "assembly service, type B")
        fact = _fact("assembly service performed today")
        j0 = _verify.Judgement(
            chosen=a, entailed=("CAND_A", "CAND_B"), declared=True,
            candidate_dispositions=(_disp(a, "entailed"), _disp(b, "entailed")))
        j1 = _verify.Judgement(
            chosen=a, entailed=("CAND_A", "CAND_B"), declared=True,
            candidate_dispositions=(_disp(a, "contradicted"), _disp(b, "entailed")))
        line = resolution._settle_uniqueness(
            fact, a, [a, b], [j0, j1], {}, "entailed", "", None)
        self.assertIsNone(line.chosen)
        self.assertIsNone(line.documentation_gap,
                          "a system verification gap must never become a provider "
                          "question")
        self.assertIn(SYSTEM_UNRESOLVED_MARKER, line.rationale)


if __name__ == "__main__":
    unittest.main()
