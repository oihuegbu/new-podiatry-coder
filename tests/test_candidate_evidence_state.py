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
        # issue #6, Codex's independent re-review (F9-R21-A): B's "entailed"
        # disposition now also needs its own validated, reconciled evidence
        # span to count as SUPPORTED (never a bare claim) -- cited here as
        # s2, alongside A's own contradiction span s1.
        j0 = _judgement([_disp(self.A, "contradicted", evidence_span_ids=("s1",)),
                         _disp(self.B, "entailed", evidence_span_ids=("s2",))])
        j1 = _judgement([_disp(self.A, "contradicted", evidence_span_ids=("s1",)),
                         _disp(self.B, "entailed", evidence_span_ids=("s2",))])
        remaining, eliminated, system_unresolved = resolution._candidate_disposition_uniqueness(
            [self.A, self.B], self.A, [j0, j1],
            reconciliation=_reconciliation({"s1": "AGREED", "s2": "AGREED"}), coverage=None)
        self.assertIn("CAND_A", eliminated)
        self.assertNotIn("CAND_A", system_unresolved)
        self.assertEqual([c.code for c in remaining], ["CAND_B"])

    def test_not_documented_without_complete_coverage_is_system_unresolved(self):
        j0 = _judgement([_disp(self.A, "not_documented", missing_fact="type A stated"),
                         _disp(self.B, "entailed", evidence_span_ids=("s2",))])
        j1 = _judgement([_disp(self.A, "not_documented", missing_fact="type A stated"),
                         _disp(self.B, "entailed", evidence_span_ids=("s2",))])
        remaining, eliminated, system_unresolved = resolution._candidate_disposition_uniqueness(
            [self.A, self.B], self.A, [j0, j1],
            reconciliation=_reconciliation({"s2": "AGREED"}), coverage=None)
        self.assertIn("CAND_A", system_unresolved)
        self.assertNotIn("CAND_A", eliminated)

    def test_both_genuinely_entailed_with_no_contract_stays_remaining(self):
        # issue #6, Codex's independent re-review (F9-R21-A): "genuinely
        # entailed" now requires validated, reconciled evidence on both
        # sides -- both candidates cite real spans here.
        j0 = _judgement([_disp(self.A, "entailed", evidence_span_ids=("s1",)),
                         _disp(self.B, "entailed", evidence_span_ids=("s2",))])
        j1 = _judgement([_disp(self.A, "entailed", evidence_span_ids=("s1",)),
                         _disp(self.B, "entailed", evidence_span_ids=("s2",))])
        remaining, eliminated, system_unresolved = resolution._candidate_disposition_uniqueness(
            [self.A, self.B], self.A, [j0, j1],
            reconciliation=_reconciliation({"s1": "AGREED", "s2": "AGREED"}), coverage=None)
        self.assertEqual(sorted(c.code for c in remaining), ["CAND_A", "CAND_B"])
        self.assertEqual(eliminated, {})
        self.assertEqual(system_unresolved, {})


class FamilyViabilityContractTest(unittest.TestCase):
    """issue #6, Codex's independent re-review (F9-R21-B): `family_viability`
    is now pure AUDIT CONTEXT (`provable=False, selectable=False,
    queryable=False`) -- never a literal-prefix-text MUST_SUPPORT proxy
    (the same unsafe shape `AXIS_DESCRIPTOR_TERM`'s own history already
    rejected: a paraphrase can support the family while never repeating the
    descriptor's exact wording), and never askable of a provider. Family
    viability is established entirely by the disposition layer itself
    (F9-R21-A): an "entailed" verdict counts as SUPPORTED only once BOTH
    evaluators cite validated, reconciled evidence for the candidate's OWN
    COMPLETE descriptor -- stem and differential together."""

    # Non-numeric differentiators deliberately: a numeric clause ("less than
    # 3 cm") would ALSO parse as a separate `measurement` axis (`provable=
    # False` by design -- a numeric interval needs a typed, unit-converted
    # comparison, never a word match), which permanently stays `unsettled`
    # and would block `narrow()`'s winner decision regardless of this test's
    # own concern. These mirror the REAL scenario the designated operative
    # note exercised (a bone-graft family differentiated by graft source).
    SMALL = _cand("CAND_SMALL", "excision of lesion, subcutaneous; with allograft")
    LARGE = _cand("CAND_LARGE", "excision of lesion, subcutaneous; with autograft")

    def test_family_viability_never_compiles_a_requirement(self):
        """The axis is audit-only -- `compile_requirements` filters on
        `provable`, so a raw descriptor prefix can never ground an
        elimination through this axis at all."""
        reqs = req.compile_requirements([self.SMALL, self.LARGE])
        self.assertEqual([r for r in reqs if r.axis == "family_viability"], [])

    def test_an_unvalidated_entailed_claim_for_an_unrelated_note_is_system_unresolved(self):
        """Both evaluators bare-assert "entailed" with no cited evidence for
        a note that describes something else entirely -- F9-R21-A's fix
        (not a lexical viability check) is what correctly refuses to treat
        either sibling as supported."""
        fact = _fact("an unrelated procedure was performed")
        reqs = req.compile_requirements([self.SMALL, self.LARGE])
        j0 = _judgement([_disp(self.SMALL, "entailed"), _disp(self.LARGE, "entailed")])
        j1 = _judgement([_disp(self.SMALL, "entailed"), _disp(self.LARGE, "entailed")])
        remaining, eliminated, system_unresolved = resolution._candidate_disposition_uniqueness(
            [self.SMALL, self.LARGE], self.SMALL, [j0, j1],
            reconciliation=None, coverage=None, fact=fact, requirements=reqs)
        self.assertEqual(remaining, [])
        self.assertEqual(eliminated, {},
                         "never falsely ELIMINATE on absent lexical family text")
        self.assertIn("CAND_SMALL", system_unresolved)
        self.assertIn("CAND_LARGE", system_unresolved)

    def test_a_paraphrased_family_is_not_falsely_eliminated(self):
        """issue #6, Codex's independent re-review (F9-R21-B), the exact
        paraphrase-safety property: the note never repeats the descriptor's
        own "excision of lesion, subcutaneous" wording verbatim, but both
        evaluators -- reading for CLINICAL MEANING, not literal text --
        validly cite real, reconciled evidence that SMALL's complete
        descriptor is entailed. The old lexical `family_viability`
        MUST_SUPPORT check would have wrongly eliminated this (the prefix
        text is nowhere in the note); the corrected design does not."""
        text = "the surgeon removed the lesion beneath the skin and used donor graft"
        fact = _fact(text)
        span = EvidenceSpan(text=text, anchored=True, span_id="s1")
        fact.evidence = [span]
        recon = _reconciliation({"s1": "AGREED"})
        reqs = req.compile_requirements([self.SMALL])
        j0 = _judgement([_disp(self.SMALL, "entailed", evidence_span_ids=("s1",))])
        j1 = _judgement([_disp(self.SMALL, "entailed", evidence_span_ids=("s1",))])
        remaining, eliminated, system_unresolved = resolution._candidate_disposition_uniqueness(
            [self.SMALL], self.SMALL, [j0, j1],
            reconciliation=recon, coverage=None, fact=fact, requirements=reqs)
        self.assertEqual([c.code for c in remaining], ["CAND_SMALL"])
        self.assertEqual(eliminated, {})
        self.assertEqual(system_unresolved, {})

    def test_family_viability_never_becomes_a_provider_question(self):
        """issue #6, Codex's independent re-review (F9-R21-B): even when the
        shared stem IS textually present and positively documented,
        `family_viability` must never be named in a provider question --
        only the genuine clinical differential may be asked about."""
        from claude_coder import tiebreak
        text = "excision of lesion, subcutaneous, performed today"
        span = EvidenceSpan(text=text, anchored=True, span_id="s1")
        fact = _fact(text)
        fact.evidence = [span]
        reconciliation = _reconciliation({"s1": "AGREED"})
        outcome = tiebreak.narrow(fact, [self.SMALL, self.LARGE], reconciliation)
        self.assertNotIn("family_viability", outcome.provider_question)
        self.assertEqual(
            outcome.provider_question,
            "The record does not state the fact that distinguishes the candidate "
            "codes for 'excision of lesion, subcutaneous, performed today'. Please "
            "document: qualified_child (with allograft vs with autograft).",
            "only the genuine clinical differential may be asked about -- never "
            "the shared family stem")

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
    """`_settle_uniqueness` integration (issue #6, Codex's independent
    re-review, F9-R21-A): a clinically eligible, system-unresolved
    candidate BLOCKS release even when a different candidate cleanly
    narrows to a single, positively-supported winner -- an unresolved
    rival's applicability is genuinely unknown, and "unrelated" is not
    something the mechanism may assume without proof. Release requires
    EVERY remaining candidate to have validated, reconciled positive
    evidence, exactly like every other axis's elimination bar."""

    def _reconciliation(self, statuses):
        return _reconciliation(statuses)

    def test_an_unvalidated_entailed_disposition_is_system_unresolved_never_released(self):
        """Codex's independent exact-SHA reproduction: two bare "entailed"
        claims, neither citing a reconciled span, must never release as
        `verified_entailment`."""
        chosen = _cand("CAND_CHOSEN", "assembly service, clean")
        fact = _fact("assembly service, clean, performed today")
        j0 = _verify.Judgement(
            chosen=chosen, entailed=("CAND_CHOSEN",), declared=True,
            candidate_dispositions=(_disp(chosen, "entailed"),))
        j1 = _verify.Judgement(
            chosen=chosen, entailed=("CAND_CHOSEN",), declared=True,
            candidate_dispositions=(_disp(chosen, "entailed"),))
        line = resolution._settle_uniqueness(
            fact, chosen, [chosen], [j0, j1], {}, "entailed", "", None)
        self.assertIsNone(line.chosen)
        self.assertNotEqual(getattr(line, "method", None), None)

    def test_one_supported_candidate_plus_one_unresolved_eligible_candidate_is_a_system_hold(self):
        """A rival whose own disposition could not be validated (evaluators
        disagree) is genuinely unknown, not proven wrong -- it must block
        release, per Codex's required ordering."""
        chosen = _cand("CAND_CHOSEN", "assembly service, clean")
        rival = _cand("CAND_RIVAL", "assembly service, rival")
        fact = _fact("assembly service, clean, performed today")
        span = EvidenceSpan(text="assembly service, clean, performed today",
                            anchored=True, span_id="s1")
        fact.evidence = [span]
        recon = self._reconciliation({"s1": "AGREED"})
        j0 = _verify.Judgement(
            chosen=chosen, entailed=("CAND_CHOSEN",), declared=True,
            candidate_dispositions=(_disp(chosen, "entailed", evidence_span_ids=("s1",)),
                                    _disp(rival, "entailed")))
        j1 = _verify.Judgement(
            chosen=chosen, entailed=("CAND_CHOSEN",), declared=True,
            candidate_dispositions=(_disp(chosen, "entailed", evidence_span_ids=("s1",)),
                                    _disp(rival, "contradicted")))
        line = resolution._settle_uniqueness(
            fact, chosen, [chosen, rival], [j0, j1], {}, "entailed", "", recon)
        self.assertIsNone(line.chosen,
                         "an unresolved eligible rival must block release, never be "
                         "silently ignored")

    def test_one_supported_candidate_plus_one_positively_excluded_candidate_releases(self):
        """A rival BOTH evaluators validly, cleanly eliminate (agreement +
        reconciled evidence) is proven wrong -- it must not block release."""
        chosen = _cand("CAND_CHOSEN", "assembly service, clean")
        excluded = _cand("CAND_EXCLUDED", "assembly service, excluded")
        fact = _fact("assembly service, clean, performed today")
        span = EvidenceSpan(text="assembly service, clean, performed today",
                            anchored=True, span_id="s1")
        excl_span = EvidenceSpan(text="not the excluded variant",
                                 anchored=True, span_id="s2")
        fact.evidence = [span, excl_span]
        recon = self._reconciliation({"s1": "AGREED", "s2": "AGREED"})
        j0 = _verify.Judgement(
            chosen=chosen, entailed=("CAND_CHOSEN",), declared=True,
            candidate_dispositions=(
                _disp(chosen, "entailed", evidence_span_ids=("s1",)),
                _disp(excluded, "contradicted", evidence_span_ids=("s2",))))
        j1 = _verify.Judgement(
            chosen=chosen, entailed=("CAND_CHOSEN",), declared=True,
            candidate_dispositions=(
                _disp(chosen, "entailed", evidence_span_ids=("s1",)),
                _disp(excluded, "contradicted", evidence_span_ids=("s2",))))
        line = resolution._settle_uniqueness(
            fact, chosen, [chosen, excluded], [j0, j1], {}, "entailed", "", recon)
        self.assertIsNotNone(line.chosen, line.rationale)
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
