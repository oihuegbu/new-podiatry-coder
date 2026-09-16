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
from unittest.mock import patch

from app.contracts.source_evidence import (ReconciliationStatus, SourceReconciliation,
                                            SpanReconciliation)
from claude_coder import requirement as req
from claude_coder import resolution
from claude_coder import tiebreak
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

    def test_all_validly_rejected_candidates_become_a_typed_recall_gap(self):
        """A complete rejection proves that none of the generated candidates
        fits. It does not prove the performed event is non-reportable and must
        not become a provider question or generic coder tie."""
        from claude_coder.models import CANDIDATE_RECALL_GAP_MARKER
        first = _cand("CAND_FIRST", "assembly service, variant one")
        second = _cand("CAND_SECOND", "assembly service, variant two")
        fact = _fact("assembly service performed today")
        dispositions = (
            _disp(first, "not_documented", missing_fact="variant one documented"),
            _disp(second, "not_documented", missing_fact="variant two documented"),
        )
        judges = [_judgement(dispositions), _judgement(dispositions)]
        line = resolution._settle_uniqueness(
            fact, first, [first, second], judges, {}, "no supported candidate", "",
            reconciliation=None, coverage=_coverage("assembly service performed today"))
        self.assertIsNone(line.chosen)
        self.assertTrue(line.candidate_recall_gap)
        self.assertIsNone(line.documentation_gap)
        self.assertIn(CANDIDATE_RECALL_GAP_MARKER, line.rationale)
        self.assertEqual({c.code for c in line.alternatives},
                         {"CAND_FIRST", "CAND_SECOND"})

    def test_page_local_rejection_defers_provider_question_until_later_candidates_run(self):
        """A question exposed by an exhausted *page* cannot terminate recall.

        A lower-ranked authoritative candidate may be fully documented, so the
        page executor must advance before presenting a provider question.  The
        synthetic tie outcome isolates that lifecycle policy from terminology.
        """
        first = _cand("CAND_FIRST", "assembly service, variant one")
        second = _cand("CAND_SECOND", "assembly service, variant two")
        fact = _fact("assembly service performed today")
        dispositions = (
            _disp(first, "not_documented", missing_fact="variant one documented"),
            _disp(second, "not_documented", missing_fact="variant two documented"),
        )
        judges = [_judgement(dispositions), _judgement(dispositions)]
        tie = tiebreak.TieOutcome(
            provider_question="document the synthetic qualifying fact",
            detail="synthetic differentiator is not documented")

        with patch.object(resolution._tiebreak, "narrow", return_value=tie):
            deferred = resolution._settle_uniqueness(
                fact, first, [first, second], judges, {}, "no supported candidate", "",
                reconciliation=None, coverage=_coverage("assembly service performed today"),
                defer_page_local_exhaustion=True)
            final = resolution._settle_uniqueness(
                fact, first, [first, second], judges, {}, "no supported candidate", "",
                reconciliation=None, coverage=_coverage("assembly service performed today"))

        self.assertTrue(deferred.candidate_recall_gap)
        self.assertIsNone(deferred.documentation_gap)
        self.assertFalse(final.candidate_recall_gap)
        self.assertEqual(final.documentation_gap, "document the synthetic qualifying fact")

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

    def test_one_supported_candidate_plus_one_gate_a_failing_candidate_is_a_system_hold(self):
        """issue #6, Codex's independent re-review (F9-R23 clarification,
        "Gate B"): the safety property Gate B asks for -- an UNGROUNDED
        recall hit must never manufacture a tie or a provider question --
        is already delivered by Gate A working together with this SAME
        pre-existing rival-blocks-release mechanism (F9-R21-A), proven
        above for "evaluators disagree". Here the rival's own citation is
        structurally perfect (a real, anchored, `AGREED` span both
        evaluators independently cite) but fails Gate A's content check
        (the span's text is unrelated to both the rival's own descriptor
        and the fact's description) -- `_settle_uniqueness` must still
        hold, never release CAND_CHOSEN through a false single-survivor
        narrowing, and never escalate to a provider question either."""
        chosen = _cand("CAND_CHOSEN", "assembly service, clean")
        rival = _cand("CAND_RIVAL", "gadget housing, unrelated device")
        fact = _fact("assembly service, clean, performed today")
        span = EvidenceSpan(text="assembly service, clean, performed today",
                            anchored=True, span_id="s1")
        unrelated_span = EvidenceSpan(text="patient reports no known drug allergies",
                                      anchored=True, span_id="s2")
        fact.evidence = [span, unrelated_span]
        recon = self._reconciliation({"s1": "AGREED", "s2": "AGREED"})
        j0 = _verify.Judgement(
            chosen=chosen, entailed=("CAND_CHOSEN", "CAND_RIVAL"), declared=True,
            candidate_dispositions=(_disp(chosen, "entailed", evidence_span_ids=("s1",)),
                                    _disp(rival, "entailed", evidence_span_ids=("s2",))))
        j1 = _verify.Judgement(
            chosen=chosen, entailed=("CAND_CHOSEN", "CAND_RIVAL"), declared=True,
            candidate_dispositions=(_disp(chosen, "entailed", evidence_span_ids=("s1",)),
                                    _disp(rival, "entailed", evidence_span_ids=("s2",))))
        line = resolution._settle_uniqueness(
            fact, chosen, [chosen, rival], [j0, j1], {}, "entailed", "", recon)
        self.assertIsNone(line.chosen,
                         "a Gate-A-failing rival must block release, never be silently "
                         "treated as proven wrong or ignored")
        self.assertIsNone(line.documentation_gap,
                          "a Gate-A citation failure is a system evidence gap, never a "
                          "provider-answerable question")

    def test_a_sole_uncontested_survivor_releases_even_with_ungrounded_standing(self):
        """issue #6, Codex's independent re-review (F9-R24-B, second
        re-review -- REVERTED after regression testing found it wrong): a
        version of `_settle_uniqueness` once required
        `CandidateStanding.SUPPORTED` on this ordinary "the proposal is
        the only candidate and nothing contests it" shape too, matching
        the reselection branch below. That broke 25 pre-existing tests
        representing ordinary, legitimate single-candidate releases,
        because `candidate_admission`'s positive-identity bar (a compiled
        requirement, a direct authoritative term, or a governed crosswalk
        mapping) is calibrated for DISCRIMINATING BETWEEN RIVALS, not for
        gating an uncontested single retrieval both evaluators
        independently entail against a real, content-relevant citation
        (Gate A). This is the same regression independently discovered
        and reverted earlier in `_propose_then_verify_core`'s own history
        (a post-verification veto re-litigating an already-Gate-A-
        validated release). This test pins the CORRECT behavior: a sole
        survivor with UNGROUNDED admission standing still releases when
        Gate A and both evaluators independently support it, because
        nothing here ever contested its identity."""
        chosen = _cand("CAND_CHOSEN", "assembly service, clean")
        fact = _fact("assembly service, clean, performed today")
        span = EvidenceSpan(text="assembly service, clean, performed today",
                            anchored=True, span_id="s1")
        fact.evidence = [span]
        recon = self._reconciliation({"s1": "AGREED"})
        j0 = _verify.Judgement(
            chosen=chosen, entailed=("CAND_CHOSEN",), declared=True,
            candidate_dispositions=(_disp(chosen, "entailed", evidence_span_ids=("s1",)),))
        j1 = _verify.Judgement(
            chosen=chosen, entailed=("CAND_CHOSEN",), declared=True,
            candidate_dispositions=(_disp(chosen, "entailed", evidence_span_ids=("s1",)),))
        admissions = {
            chosen.code: resolution.CandidateAdmission(
                (chosen.code, chosen.system), resolution.CandidateStanding.UNGROUNDED,
                (), (), (), ("s1",),
                {"code": chosen.code, "descriptor": chosen.descriptor},
                (chosen.source,), "synthetic recall-only candidate, no positive identity signal"),
        }
        line = resolution._settle_uniqueness(
            fact, chosen, [chosen], [j0, j1], {}, "entailed", "", recon,
            admissions=admissions)
        self.assertEqual(line.chosen.code if line.chosen else None, "CAND_CHOSEN",
                         "an uncontested sole survivor must still release on Gate A + "
                         "dual-evaluator entailment alone -- admission standing is for "
                         "discriminating between rivals, not for re-gating an uncontested "
                         "single retrieval")

    def test_complete_independent_matrices_can_select_without_a_model_proposal(self):
        """Selection belongs to deterministic reconciliation, not to whichever
        evaluator happened to nominate a code.  Two complete descriptor-bound
        matrices may leave one unique survivor even when ``chosen`` is absent."""
        supported = _cand("CAND_SUPPORTED", "assembly service, supported form")
        excluded = _cand("CAND_EXCLUDED", "assembly service, different form")
        fact = _fact("assembly service, supported form, performed")
        fact.evidence = [
            EvidenceSpan(text="assembly service, supported form, performed",
                         anchored=True, span_id="s1"),
            EvidenceSpan(text="different form was not performed",
                         anchored=True, span_id="s2"),
        ]
        recon = self._reconciliation({"s1": "AGREED", "s2": "AGREED"})
        judgements = [
            _verify.Judgement(
                chosen=None, entailed=(), declared=True,
                candidate_dispositions=(
                    _disp(supported, "entailed", evidence_span_ids=("s1",)),
                    _disp(excluded, "contradicted", evidence_span_ids=("s2",))))
            for _ in range(2)
        ]

        line = resolution._settle_uniqueness(
            fact, None, [supported, excluded], judgements, {},
            "no proposal", "distinct_origin", recon)

        self.assertEqual(line.chosen.code if line.chosen else None,
                         "CAND_SUPPORTED")
        self.assertTrue(line.tie_record["selected_from_complete_disposition_matrix"])
        self.assertIn("CAND_EXCLUDED", line.tie_record["eliminated"])

    def test_no_proposal_with_two_supported_survivors_remains_a_tie(self):
        first = _cand("CAND_FIRST", "assembly service, first form")
        second = _cand("CAND_SECOND", "assembly service, second form")
        fact = _fact("assembly service performed")
        fact.evidence = [EvidenceSpan(text="assembly service performed",
                                      anchored=True, span_id="s1")]
        recon = self._reconciliation({"s1": "AGREED"})
        judgements = [
            _verify.Judgement(
                chosen=None, entailed=(), declared=True,
                candidate_dispositions=(
                    _disp(first, "entailed", evidence_span_ids=("s1",)),
                    _disp(second, "entailed", evidence_span_ids=("s1",))))
            for _ in range(2)
        ]

        line = resolution._settle_uniqueness(
            fact, None, [first, second], judgements, {},
            "no proposal", "distinct_origin", recon)

        self.assertIsNone(line.chosen)
        self.assertEqual(sorted(c.code for c in line.alternatives),
                         ["CAND_FIRST", "CAND_SECOND"])

    def test_ungrounded_recall_rival_cannot_block_a_supported_candidate(self):
        """Standing is load-bearing: a recall-only rival that evaluators do
        not independently support is audited out and cannot create a tie,
        system hold, or provider question."""
        chosen = _cand("CAND_CHOSEN", "assembly service, clean")
        rival = _cand("CAND_RIVAL", "unrelated device")
        fact = _fact("assembly service, clean, performed today")
        span = EvidenceSpan(text="assembly service, clean, performed today",
                            anchored=True, span_id="s1")
        fact.evidence = [span]
        recon = self._reconciliation({"s1": "AGREED"})
        j0 = _verify.Judgement(
            chosen=chosen, entailed=("CAND_CHOSEN",), declared=True,
            candidate_dispositions=(
                _disp(chosen, "entailed", evidence_span_ids=("s1",)),
                _disp(rival, "entailed")))
        j1 = _verify.Judgement(
            chosen=chosen, entailed=("CAND_CHOSEN",), declared=True,
            candidate_dispositions=(
                _disp(chosen, "entailed", evidence_span_ids=("s1",)),
                _disp(rival, "different_concept")))
        admissions = {
            chosen.code: resolution.CandidateAdmission(
                (chosen.code, chosen.system), resolution.CandidateStanding.SUPPORTED,
                ("descriptor_term",), (), (), ("s1",),
                {"code": chosen.code, "descriptor": chosen.descriptor},
                (chosen.source,), "synthetic supported identity"),
            rival.code: resolution.CandidateAdmission(
                (rival.code, rival.system), resolution.CandidateStanding.UNGROUNDED,
                (), (), (), (),
                {"code": rival.code, "descriptor": rival.descriptor},
                (rival.source,), "synthetic recall-only candidate"),
        }
        line = resolution._settle_uniqueness(
            fact, chosen, [chosen, rival], [j0, j1], {}, "entailed",
            "distinct_origin", recon, admissions=admissions)
        self.assertEqual(line.chosen.code if line.chosen else None, "CAND_CHOSEN")
        self.assertIsNone(line.documentation_gap)

    def test_unique_governed_identity_outranks_only_recall_only_topical_rival(self):
        """Both descriptors may be semantically compatible, but only the candidate
        with governed identity lineage identifies the documented concept.  The
        recall-only rival cannot manufacture a tie after both descriptors have
        independently passed their own verification."""
        chosen = _cand("CAND_CHOSEN", "assembly service, broad category")
        rival = _cand("CAND_RIVAL", "assembly service, alternate category")
        fact = _fact("assembly service documented")
        s1 = EvidenceSpan(text="assembly service documented", anchored=True, span_id="s1")
        s2 = EvidenceSpan(text="assembly service documented", anchored=True, span_id="s2")
        fact.evidence = [s1, s2]
        recon = self._reconciliation({"s1": "AGREED", "s2": "AGREED"})
        j0 = _verify.Judgement(candidate_dispositions=(
            _disp(chosen, "entailed", evidence_span_ids=("s1",)),
            _disp(rival, "entailed", evidence_span_ids=("s2",))), declared=True)
        j1 = _verify.Judgement(candidate_dispositions=(
            _disp(chosen, "entailed", evidence_span_ids=("s1",)),
            _disp(rival, "entailed", evidence_span_ids=("s2",))), declared=True)
        admissions = {
            chosen.code: resolution.CandidateAdmission(
                (chosen.code, chosen.system), resolution.CandidateStanding.SUPPORTED,
                ("governed_term_mapping",), (), (), ("s1",),
                {"code": chosen.code, "descriptor": chosen.descriptor},
                ("governed-map",), "source-bound identity"),
            rival.code: resolution.CandidateAdmission(
                (rival.code, rival.system), resolution.CandidateStanding.UNGROUNDED,
                (), (), (), (),
                {"code": rival.code, "descriptor": rival.descriptor},
                ("retrieval",), "topical recall only"),
        }

        remaining, eliminated, unresolved = resolution._candidate_disposition_uniqueness(
            [chosen, rival], chosen, [j0, j1], recon, coverage=None,
            fact=fact, admissions=admissions)

        self.assertEqual([candidate.code for candidate in remaining], ["CAND_CHOSEN"])
        self.assertIn("CAND_RIVAL", eliminated)
        self.assertEqual(unresolved, {})

    def test_unique_governed_survivor_replaces_a_wrong_initial_proposal(self):
        governed = _cand("CAND_GOVERNED", "assembly service, broad category")
        proposed = _cand("CAND_PROPOSED", "assembly service, alternate category")
        fact = _fact("assembly service documented")
        s1 = EvidenceSpan(text="assembly service documented", anchored=True, span_id="s1")
        s2 = EvidenceSpan(text="assembly service documented", anchored=True, span_id="s2")
        fact.evidence = [s1, s2]
        recon = self._reconciliation({"s1": "AGREED", "s2": "AGREED"})
        judgements = [_verify.Judgement(
            chosen=proposed, entailed=(governed.code, proposed.code), declared=True,
            candidate_dispositions=(
                _disp(governed, "entailed", evidence_span_ids=("s1",)),
                _disp(proposed, "entailed", evidence_span_ids=("s2",))))
            for _ in range(2)]
        admissions = {
            governed.code: resolution.CandidateAdmission(
                (governed.code, governed.system), resolution.CandidateStanding.SUPPORTED,
                ("governed_term_mapping",), (), (), ("s1",),
                {"code": governed.code, "descriptor": governed.descriptor},
                ("governed-map",), "source-bound identity"),
            proposed.code: resolution.CandidateAdmission(
                (proposed.code, proposed.system), resolution.CandidateStanding.UNGROUNDED,
                (), (), (), (),
                {"code": proposed.code, "descriptor": proposed.descriptor},
                ("retrieval",), "topical recall only"),
        }

        line = resolution._settle_uniqueness(
            fact, proposed, [governed, proposed], judgements, {}, "verified", "",
            recon, admissions=admissions)

        self.assertEqual(line.chosen.code if line.chosen else None, "CAND_GOVERNED")
        self.assertTrue(line.tie_record["reselected_from_verified_survivor"])
        self.assertEqual(line.tie_record["proposed"], "CAND_PROPOSED")

    def test_two_governed_identity_candidates_remain_a_real_tie(self):
        chosen = _cand("CAND_CHOSEN", "assembly service, broad category")
        rival = _cand("CAND_RIVAL", "assembly service, alternate category")
        fact = _fact("assembly service documented")
        s1 = EvidenceSpan(text="assembly service documented", anchored=True, span_id="s1")
        fact.evidence = [s1]
        recon = self._reconciliation({"s1": "AGREED"})
        judgements = [_verify.Judgement(candidate_dispositions=(
            _disp(chosen, "entailed", evidence_span_ids=("s1",)),
            _disp(rival, "entailed", evidence_span_ids=("s1",))), declared=True)
            for _ in range(2)]
        admissions = {
            candidate.code: resolution.CandidateAdmission(
                (candidate.code, candidate.system), resolution.CandidateStanding.SUPPORTED,
                ("governed_term_mapping",), (), (), ("s1",),
                {"code": candidate.code, "descriptor": candidate.descriptor},
                ("governed-map",), "source-bound identity")
            for candidate in (chosen, rival)
        }

        remaining, eliminated, unresolved = resolution._candidate_disposition_uniqueness(
            [chosen, rival], chosen, judgements, recon, coverage=None,
            fact=fact, admissions=admissions)

        self.assertEqual(sorted(candidate.code for candidate in remaining),
                         ["CAND_CHOSEN", "CAND_RIVAL"])
        self.assertEqual(eliminated, {})
        self.assertEqual(unresolved, {})

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

    def test_validated_cross_evaluator_disagreement_uses_shared_evidence_not_a_vote(self):
        """A valid semantic split is an adjudication signal, not a terminal hold.

        The candidate descriptors define a structural qualified-child axis.  Both
        evaluators positively support ``chosen``; they disagree about its sibling,
        but every cited disposition is descriptor-bound and source-confirmed.  The
        resolver must settle the pair from the original document's same
        authoritative axis, never by preferring either evaluator.
        """
        chosen = _cand("CAND_ALPHA", "assembly service; variant alpha")
        rival = _cand("CAND_BETA", "assembly service; variant beta")
        fact = _fact("assembly service; variant alpha performed")
        span = EvidenceSpan(text="assembly service; variant alpha performed",
                            anchored=True, span_id="s1")
        fact.evidence = [span]
        recon = self._reconciliation({"s1": "AGREED"})
        primary = _verify.Judgement(
            chosen=chosen, entailed=(chosen.code, rival.code), declared=True,
            candidate_dispositions=(
                _disp(chosen, "entailed", evidence_span_ids=("s1",)),
                _disp(rival, "entailed", evidence_span_ids=("s1",))))
        corroborator = _verify.Judgement(
            chosen=chosen, entailed=(chosen.code,), declared=True,
            candidate_dispositions=(
                _disp(chosen, "entailed", evidence_span_ids=("s1",)),
                _disp(rival, "different_concept", evidence_span_ids=("s1",))))

        line = resolution._settle_uniqueness(
            fact, chosen, [chosen, rival], [primary, corroborator], {},
            "independent verification", "cross-vendor corroboration", recon)

        self.assertEqual(line.chosen.code if line.chosen else None, chosen.code,
                         line.rationale)
        audit = line.tie_record["evidence_constrained_disagreement"]
        self.assertEqual(audit["validated_disagreements"], [rival.code])
        self.assertEqual(audit["winner"], chosen.code)
        self.assertIn("no evaluator vote", audit["decision_rule"])

    def test_single_disputed_candidate_releases_only_on_complete_source_contract(self):
        """A single semantic split can be settled by non-model evidence.

        There is no rival for ordinary tie narrowing here.  Release is allowed
        only because the source admission proves normalized identity, every
        compiled required axis is complete, and the event's source quotation is
        reconciled.  This is intentionally not a one-model-wins fallback.
        """
        candidate = _cand("CAND_ALPHA", "assembly service; variant alpha")
        fact = _fact("assembly service; variant alpha performed")
        span = EvidenceSpan(text="assembly service; variant alpha performed",
                            anchored=True, span_id="s1")
        fact.evidence = [span]
        recon = self._reconciliation({"s1": "AGREED"})
        primary = _verify.Judgement(
            chosen=candidate, entailed=(candidate.code,), declared=True,
            candidate_dispositions=(
                _disp(candidate, "entailed", evidence_span_ids=("s1",)),))
        corroborator = _verify.Judgement(
            chosen=None, entailed=(), declared=True,
            candidate_dispositions=(
                _disp(candidate, "different_concept", evidence_span_ids=("s1",)),))
        admissions = {
            candidate.code: resolution.CandidateAdmission(
                (candidate.code, candidate.system),
                resolution.CandidateStanding.SUPPORTED,
                ("governed_term_mapping",), (), (), ("s1",),
                {"code": candidate.code, "descriptor": candidate.descriptor},
                ("authoritative-index",), "source-governed identity"),
        }

        line = resolution._settle_uniqueness(
            fact, candidate, [candidate], [primary, corroborator], {},
            "independent verification", "cross-vendor corroboration", recon,
            admissions=admissions)

        self.assertEqual(line.chosen.code if line.chosen else None, candidate.code,
                         line.rationale)
        audit = line.tie_record["evidence_constrained_disagreement"]
        self.assertEqual(audit["deterministically_supported"], [candidate.code])

    def test_single_disputed_candidate_with_incomplete_source_contract_stays_held(self):
        """An unresolved requirement prevents the source-contract release path."""
        from claude_coder.models import SYSTEM_UNRESOLVED_MARKER
        candidate = _cand("CAND_ALPHA", "assembly service; variant alpha")
        fact = _fact("assembly service; variant alpha performed")
        span = EvidenceSpan(text="assembly service; variant alpha performed",
                            anchored=True, span_id="s1")
        fact.evidence = [span]
        recon = self._reconciliation({"s1": "AGREED"})
        primary = _verify.Judgement(
            chosen=candidate, entailed=(candidate.code,), declared=True,
            candidate_dispositions=(
                _disp(candidate, "entailed", evidence_span_ids=("s1",)),))
        corroborator = _verify.Judgement(
            chosen=None, entailed=(), declared=True,
            candidate_dispositions=(
                _disp(candidate, "different_concept", evidence_span_ids=("s1",)),))
        admissions = {
            candidate.code: resolution.CandidateAdmission(
                (candidate.code, candidate.system),
                resolution.CandidateStanding.SUPPORTED,
                ("governed_term_mapping",), (), ("typed_attribute",), ("s1",),
                {"code": candidate.code, "descriptor": candidate.descriptor},
                ("authoritative-index",), "source identity but incomplete contract"),
        }

        line = resolution._settle_uniqueness(
            fact, candidate, [candidate], [primary, corroborator], {},
            "independent verification", "cross-vendor corroboration", recon,
            admissions=admissions)

        self.assertIsNone(line.chosen)
        self.assertIsNone(line.documentation_gap)
        self.assertIn(SYSTEM_UNRESOLVED_MARKER, line.rationale)

    def test_a_disputed_candidates_own_undocumented_indication_clause_becomes_a_provider_question(self):
        """issue #6, independent root-cause investigation (real-data replay
        against the designated note): before `tiebreak.AXIS_INDICATION_CLAUSE`
        existed, two evaluators disagreeing specifically about a candidate
        whose OWN descriptor states a positive "(eg, ...)" indication clause
        had NO governed axis to adjudicate through -- `_tiebreak.narrow` had
        nothing to test (no winner, and no provider question either, since
        `discriminating_axes` never surfaced the clause at all), so the
        disagreement fell to a permanent, unrescuable SYSTEM_UNRESOLVED hold
        that reproduced identically on every retry. Reproduced live on the
        real note (CPT 28118 vs 28120, "...eg, osteomyelitis or bossing...");
        pinned here with synthetic vocabulary. Both evaluators agree on
        CAND_ALPHA (states no indication clause); they disagree specifically
        about CAND_BETA, whose own descriptor requires documenting "variant
        condition" -- a fact the evidence never states either way. This must
        now become a genuine, specific, answerable provider question naming
        that exact fact, never a silent hold with no path to resolution."""
        from claude_coder.models import SYSTEM_UNRESOLVED_MARKER
        chosen = _cand("CAND_ALPHA", "assembly service, broad category")
        rival = _cand("CAND_BETA",
                      "assembly service, broad category (eg, variant condition)")
        fact = _fact("assembly service performed today")
        span = EvidenceSpan(text="assembly service performed today", anchored=True,
                           span_id="s1")
        fact.evidence = [span]
        recon = self._reconciliation({"s1": "AGREED"})
        primary = _verify.Judgement(
            chosen=chosen, entailed=(chosen.code, rival.code), declared=True,
            candidate_dispositions=(
                _disp(chosen, "entailed", evidence_span_ids=("s1",)),
                _disp(rival, "entailed", evidence_span_ids=("s1",))))
        corroborator = _verify.Judgement(
            chosen=chosen, entailed=(chosen.code,), declared=True,
            candidate_dispositions=(
                _disp(chosen, "entailed", evidence_span_ids=("s1",)),
                _disp(rival, "different_concept", evidence_span_ids=("s1",))))

        line = resolution._settle_uniqueness(
            fact, chosen, [chosen, rival], [primary, corroborator], {},
            "independent verification", "cross-vendor corroboration", recon)

        self.assertIsNone(line.chosen)
        self.assertNotIn(SYSTEM_UNRESOLVED_MARKER, line.rationale or "")
        self.assertIsNotNone(line.documentation_gap)
        self.assertIn("variant condition", line.documentation_gap)
        self.assertIn("indication_clause", line.documentation_gap)

    def test_malformed_disagreement_remains_a_system_hold_not_a_provider_question(self):
        """The new resolver must not launder a citation failure into autonomy.

        This superficially resembles the resolvable case above, but the rival's
        negative disposition is uncited.  It is therefore a system-integrity
        failure, not a fact a provider could repair and not a basis to release
        the independently supported candidate.
        """
        from claude_coder.models import SYSTEM_UNRESOLVED_MARKER
        chosen = _cand("CAND_ALPHA", "assembly service; variant alpha")
        rival = _cand("CAND_BETA", "assembly service; variant beta")
        fact = _fact("assembly service; variant alpha performed")
        span = EvidenceSpan(text="assembly service; variant alpha performed",
                            anchored=True, span_id="s1")
        fact.evidence = [span]
        recon = self._reconciliation({"s1": "AGREED"})
        primary = _verify.Judgement(
            chosen=chosen, entailed=(chosen.code, rival.code), declared=True,
            candidate_dispositions=(
                _disp(chosen, "entailed", evidence_span_ids=("s1",)),
                _disp(rival, "entailed", evidence_span_ids=("s1",))))
        corroborator = _verify.Judgement(
            chosen=chosen, entailed=(chosen.code,), declared=True,
            candidate_dispositions=(
                _disp(chosen, "entailed", evidence_span_ids=("s1",)),
                _disp(rival, "different_concept")))

        line = resolution._settle_uniqueness(
            fact, chosen, [chosen, rival], [primary, corroborator], {},
            "independent verification", "cross-vendor corroboration", recon)

        self.assertIsNone(line.chosen)
        self.assertIsNone(line.documentation_gap)
        self.assertIn(SYSTEM_UNRESOLVED_MARKER, line.rationale)

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
