"""Descriptor/instructional-note requirement compilation and validation (issue #6
F9-R6, Phase 0). `compile_requirements` is a MECHANICAL PROJECTION of `tiebreak.
discriminating_axes` — never an independently invented axis compiler — so these
tests pin that projection rule (`required == probe.selectable`, silent axes produce
nothing, only `provable` axes compile) and `validated_requirement`'s independent
re-check (clause must reproduce, cited spans must be reconciled). Synthetic
descriptors/ids throughout — the mechanism reads descriptor grammar and a real ICD
Tabular data shape, never a term list.
"""
import unittest

from claude_coder import requirement as req
from claude_coder.data_access import MockSource
from claude_coder.models import CandidateCode


def _cand(code, descriptor, system="cpt"):
    return CandidateCode(code=code, system=system, descriptor=descriptor, score=0.9,
                         source="retrieval")


# Same synthetic convention as tests/test_tie_policy.py: two candidates that satisfy
# every axis identically except one descriptor word.
POWERED = _cand("CAND_POWERED", "assembly service, powered technique")
MANUAL = _cand("CAND_MANUAL", "assembly service, manual technique")
LEFT = _cand("CAND_LEFT", "assembly service performed on the left")
RIGHT = _cand("CAND_RIGHT", "assembly service performed on the right")


class CompileRequirementsTest(unittest.TestCase):

    def test_a_selectable_axis_produces_a_required_requirement(self):
        reqs = req.compile_requirements([LEFT, RIGHT])
        laterality = [r for r in reqs if r.axis == "laterality"]
        self.assertEqual(len(laterality), 2)   # one per candidate
        self.assertTrue(all(r.required for r in laterality))
        self.assertTrue(all(r.role is req.RequirementRole.MUST_SUPPORT for r in laterality))
        by_code = {r.candidate_code: r for r in laterality}
        self.assertEqual(by_code["CAND_LEFT"].expected, ("left",))
        self.assertEqual(by_code["CAND_RIGHT"].expected, ("right",))

    def test_a_non_selectable_axis_produces_an_optional_requirement(self):
        reqs = req.compile_requirements([POWERED, MANUAL])
        term_reqs = [r for r in reqs if r.axis == "descriptor_term"]
        self.assertEqual(len(term_reqs), 2)
        self.assertTrue(all(not r.required for r in term_reqs))
        self.assertTrue(all(r.role is req.RequirementRole.POSITIVE_ALIAS for r in term_reqs))

    def test_measurement_axis_never_compiles_a_requirement(self):
        """Measurement is not `provable` by words at all (`tiebreak.AxisProbe`'s own
        docstring: a numeric interval needs a typed, unit-converted comparison) --
        `compile_requirements` deliberately skips non-provable axes rather than
        fabricating a text clause from a synthetic interval key."""
        five = _cand("CAND_5", "assembly service, 5 cm")
        ten = _cand("CAND_10", "assembly service, 10 cm")
        reqs = req.compile_requirements([five, ten])
        self.assertEqual([r for r in reqs if r.axis == "measurement"], [])

    def test_identical_descriptors_produce_no_requirements(self):
        same_a = _cand("CAND_A", "assembly service performed")
        same_b = _cand("CAND_B", "assembly service performed")
        self.assertEqual(req.compile_requirements([same_a, same_b]), ())

    def test_a_single_candidate_produces_no_requirements(self):
        """`discriminating_axes` itself requires at least two candidates to derive a
        DIFFERENCE from -- mirrored here, not re-implemented."""
        self.assertEqual(req.compile_requirements([LEFT]), ())

    def test_authority_clause_reproduces_verbatim_from_the_real_descriptor(self):
        reqs = req.compile_requirements([LEFT, RIGHT])
        left_req = next(r for r in reqs if r.candidate_code == "CAND_LEFT")
        start, end = left_req.authority_offset
        self.assertEqual(left_req.authority_source_text[start:end],
                         left_req.authority_clause)
        self.assertEqual(left_req.authority_clause, "left")

    def test_requirement_ids_are_unique_within_one_compile_call(self):
        reqs = req.compile_requirements([LEFT, RIGHT])
        ids = [r.requirement_id for r in reqs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_as_record_includes_authority_source_text(self):
        """issue #6 F9-R6-R5: without the full source text, an auditor cannot
        reproduce the clause-offset check from the record alone."""
        reqs = req.compile_requirements([LEFT, RIGHT])
        left_req = next(r for r in reqs if r.candidate_code == "CAND_LEFT")
        record = left_req.as_record()
        self.assertEqual(record["authority_source_text"], left_req.authority_source_text)
        start, end = record["authority_offset"]
        self.assertEqual(record["authority_source_text"][start:end],
                         record["authority_clause"])

    def test_source_identity_carries_the_candidate_authority(self):
        """issue #6 F9-R6-R5: the candidate's own real provenance, not just the
        axis's kind/system, so an auditor can tell which edition of the
        descriptor a requirement was compiled against."""
        provenanced = CandidateCode(code="CAND_LEFT2", system="cpt",
                                    descriptor="assembly service performed on the left",
                                    score=0.9, source="retrieval",
                                    authority={"index": "rag-hybrid", "system": "cpt"})
        other = CandidateCode(code="CAND_RIGHT2", system="cpt",
                              descriptor="assembly service performed on the right",
                              score=0.9, source="retrieval")
        reqs = req.compile_requirements([provenanced, other])
        target = next(r for r in reqs if r.candidate_code == "CAND_LEFT2")
        self.assertEqual(target.source_identity["authority"],
                         {"index": "rag-hybrid", "system": "cpt"})

    def test_descriptor_requirement_carries_the_bound_source_snapshot_identity(self):
        """issue #6 F9-R6-R5, fourth re-review: when `source` supplies
        `record_snapshot_identity`, a compiled descriptor requirement's
        `source_identity["snapshot"]` carries the exact bound source
        identity -- distinct from `authority` (the candidate's own retrieval
        provenance), this is the WHOLE-FILE identity the descriptor table was
        actually parsed from."""
        source = MockSource(snapshot={"source_id": "cpt_codes", "sha256": "abc123",
                                      "size": 4096})
        reqs = req.compile_requirements([LEFT, RIGHT], source=source)
        target = next(r for r in reqs if r.candidate_code == "CAND_LEFT")
        self.assertEqual(target.source_identity["snapshot"],
                         {"source_id": "cpt_codes", "sha256": "abc123", "size": 4096})

    def test_descriptor_requirement_snapshot_degrades_to_empty_without_the_method(self):
        """A source that doesn't implement `record_snapshot_identity` at all (an
        older test double) must never raise -- the requirement compiles exactly
        as before, just with an empty snapshot."""
        reqs = req.compile_requirements([LEFT, RIGHT], source=None)
        target = next(r for r in reqs if r.candidate_code == "CAND_LEFT")
        self.assertEqual(target.source_identity["snapshot"], {})

    def test_descriptor_requirement_snapshot_degrades_when_the_source_raises(self):
        class _Raises:
            def record_snapshot_identity(self, code, system):
                raise RuntimeError("simulated unavailable manifest identity")
        reqs = req.compile_requirements([LEFT, RIGHT], source=_Raises())
        target = next(r for r in reqs if r.candidate_code == "CAND_LEFT")
        self.assertEqual(target.source_identity["snapshot"], {})

    def test_inclusion_term_requirement_carries_the_bound_source_snapshot_identity(self):
        source = MockSource(instructional_terms={"A000": {"classical cholera"}},
                            snapshot={"source_id": "instructional_notes",
                                     "sha256": "def456", "size": 2048})
        icd_a = _cand("A00.0", "cholera, unspecified", system="icd10")
        icd_b = _cand("A00.1", "cholera, another type", system="icd10")
        reqs = req.compile_requirements([icd_a, icd_b], source=source)
        incl = next(r for r in reqs if r.axis == "inclusion_term")
        self.assertEqual(incl.source_identity["snapshot"],
                         {"source_id": "instructional_notes", "sha256": "def456",
                          "size": 2048})

    def test_inclusion_term_requirement_snapshot_degrades_to_empty_without_the_method(self):
        icd_a = _cand("A00.0", "cholera, unspecified", system="icd10")
        icd_b = _cand("A00.1", "cholera, another type", system="icd10")
        source = MockSource(instructional_terms={"A000": {"classical cholera"}})
        reqs = req.compile_requirements([icd_a, icd_b], source=source)
        incl = next(r for r in reqs if r.axis == "inclusion_term")
        self.assertEqual(incl.source_identity["snapshot"], {})

    def test_inclusion_term_requirements_only_for_icd10_when_source_supplies_them(self):
        icd_a = _cand("A00.0", "cholera, unspecified", system="icd10")
        icd_b = _cand("A00.1", "cholera, another type", system="icd10")
        source = MockSource(instructional_terms={
            "A000": {"classical cholera"},
        })
        reqs = req.compile_requirements([icd_a, icd_b], source=source)
        incl = [r for r in reqs if r.axis == "inclusion_term"]
        self.assertEqual(len(incl), 1)
        self.assertEqual(incl[0].candidate_code, "A00.0")
        self.assertEqual(incl[0].expected, ("classical cholera",))
        # issue #6 F9-R6-R3 re-review: inclusion terms are non-exhaustive
        # EXAMPLES, never required/selectable -- POSITIVE_ALIAS only, so
        # absence of even every listed example can never disqualify a
        # candidate on its own.
        self.assertFalse(incl[0].required)
        self.assertFalse(incl[0].selectable)
        self.assertIs(incl[0].role, req.RequirementRole.POSITIVE_ALIAS)

    def test_no_inclusion_term_requirements_without_a_source(self):
        icd_a = _cand("A00.0", "cholera, unspecified", system="icd10")
        icd_b = _cand("A00.1", "cholera, another type", system="icd10")
        reqs = req.compile_requirements([icd_a, icd_b], source=None)
        self.assertEqual([r for r in reqs if r.axis == "inclusion_term"], [])

    def test_a_source_without_instructional_terms_method_degrades_silently(self):
        """A source that doesn't implement `instructional_terms` at all (an older
        test double, or a genuinely unavailable optional source) must never raise --
        it just contributes no inclusion-term requirements."""
        class _Bare:
            pass
        icd_a = _cand("A00.0", "cholera, unspecified", system="icd10")
        icd_b = _cand("A00.1", "cholera, another type", system="icd10")
        reqs = req.compile_requirements([icd_a, icd_b], source=_Bare())
        self.assertEqual([r for r in reqs if r.axis == "inclusion_term"], [])

    def test_a_raising_instructional_terms_source_degrades_that_candidate_only(self):
        icd_a = _cand("A00.0", "cholera, unspecified", system="icd10")
        icd_b = _cand("A00.1", "cholera, another type", system="icd10")

        class _Raises:
            def instructional_terms(self, code, system):
                raise RuntimeError("simulated unavailable instructional notes")
        reqs = req.compile_requirements([icd_a, icd_b], source=_Raises())
        self.assertEqual([r for r in reqs if r.axis == "inclusion_term"], [])


class CoverageCorpusValidationTest(unittest.TestCase):
    """issue #6 F9-R6-R4, fourth re-review: `CoverageCorpus` self-validates at
    construction. The old `.complete` property never checked `channel_id`/
    `covered_pages` were populated at all, or that `text_sha256` genuinely
    matches `text` -- a blank-channel, zero-page, or wrong-hash corpus could
    report `complete=True`. A frozen dataclass may still raise from
    `__post_init__` (it only forbids ASSIGNING fields there), so construction
    now fails fast instead of silently validating a corpus that never proves
    what it claims to."""

    TEXT = "assembly service performed on the left"

    def _hash(self, text):
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()

    def test_wrong_text_sha256_raises(self):
        with self.assertRaises(ValueError):
            req.CoverageCorpus(channel_id="ch1", text=self.TEXT, text_sha256="wrong",
                               covered_pages=(1,), page_image_sha256=("h1",))

    def test_empty_channel_id_raises_even_with_real_text_and_hash(self):
        with self.assertRaises(ValueError):
            req.CoverageCorpus(channel_id="", text=self.TEXT,
                               text_sha256=self._hash(self.TEXT),
                               covered_pages=(1,), page_image_sha256=("h1",))

    def test_empty_covered_pages_raises_even_with_real_text_and_hash(self):
        with self.assertRaises(ValueError):
            req.CoverageCorpus(channel_id="ch1", text=self.TEXT,
                               text_sha256=self._hash(self.TEXT),
                               covered_pages=(), page_image_sha256=())

    def test_mismatched_page_image_sha256_cardinality_raises(self):
        with self.assertRaises(ValueError):
            req.CoverageCorpus(channel_id="ch1", text=self.TEXT,
                               text_sha256=self._hash(self.TEXT),
                               covered_pages=(1, 2), page_image_sha256=("h1",))

    def test_a_correctly_constructed_fully_covered_corpus_is_complete(self):
        corpus = req.CoverageCorpus(channel_id="ch1", text=self.TEXT,
                                    text_sha256=self._hash(self.TEXT),
                                    covered_pages=(1,), page_image_sha256=("h1",))
        self.assertTrue(corpus.complete)

    def test_a_corpus_with_uncovered_pages_is_not_complete(self):
        corpus = req.CoverageCorpus(channel_id="ch1", text=self.TEXT,
                                    text_sha256=self._hash(self.TEXT),
                                    covered_pages=(1,), page_image_sha256=("h1",),
                                    uncovered_pages=(2,))
        self.assertFalse(corpus.complete)


class ValidatedRequirementTest(unittest.TestCase):
    """issue #6 F9-R6-R2, TWO rounds of re-review: `validated_requirement` no
    longer trusts a cited span's mere existence/reconciliation status. SUPPORTED
    requires the SPECIFIC cited span's OWN text to genuinely, un-negatedly
    contain the term (`evidence_by_span_id`) -- never a whole-document search
    standing in for a particular citation's content. NOT_DOCUMENTED requires
    the WHOLE `coverage` corpus to show genuine, negation-aware absence.
    CONTRADICTED never validates as a judgement status, for any axis, for any
    citation (see `requirement.validated_requirement`'s docstring)."""

    DOCUMENTED = "assembly service performed on the left"
    UNDOCUMENTED = "assembly service performed, no laterality stated"
    NEGATED = "assembly service performed, not on the left"

    def _req(self):
        return req.DescriptorRequirement(
            requirement_id="laterality:CAND_LEFT:0", axis="laterality",
            candidate_code="CAND_LEFT", required=True,
            role=req.RequirementRole.MUST_SUPPORT, expected=("left",),
            authority_clause="left",
            authority_offset=(len("assembly service performed on the "),
                              len("assembly service performed on the left")),
            authority_source_text="assembly service performed on the left",
            selectable=True, queryable=True)

    def _reconciliation(self, statuses):
        from app.contracts.source_evidence import (ReconciliationStatus,
                                                    SourceReconciliation,
                                                    SpanReconciliation)
        return SourceReconciliation(spans=tuple(
            SpanReconciliation(span_id=sid, status=ReconciliationStatus[status])
            for sid, status in statuses.items()))

    def _coverage(self, text):
        import hashlib
        return req.CoverageCorpus(channel_id="test-channel", text=text,
                                  text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                  covered_pages=(1,), page_image_sha256=("stub-hash",))

    def test_wrong_requirement_id_never_validates(self):
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id="not-the-same-id", status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("s1",))
        self.assertFalse(req.validated_requirement(
            r, judgement, evidence_by_span_id={"s1": self.DOCUMENTED},
            reconciliation=self._reconciliation({"s1": "AGREED"})))

    def test_a_clause_that_does_not_reproduce_never_validates(self):
        """The clause claims to come from the candidate's own descriptor but the
        offset no longer points at it (a hallucinated or stale requirement)."""
        from dataclasses import replace
        r = self._req()
        tampered = replace(r, authority_clause="right")
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("s1",))
        self.assertFalse(req.validated_requirement(
            tampered, judgement, evidence_by_span_id={"s1": self.DOCUMENTED},
            reconciliation=self._reconciliation({"s1": "AGREED"})))

    def test_contradicted_never_validates_regardless_of_citation(self):
        """issue #6 F9-R6-R2: the direct regression pin -- a well-formed,
        reconciled, correctly-cited CONTRADICTED judgement must never validate,
        for any axis. Retired permanently, not merely tightened."""
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.CONTRADICTED,
            evidence_span_ids=("s1",))
        self.assertFalse(req.validated_requirement(
            r, judgement, evidence_by_span_id={"s1": self.DOCUMENTED},
            reconciliation=self._reconciliation({"s1": "AGREED"})))

    def test_supported_with_an_agreed_span_validates(self):
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("s1",))
        self.assertTrue(req.validated_requirement(
            r, judgement, evidence_by_span_id={"s1": self.DOCUMENTED},
            reconciliation=self._reconciliation({"s1": "AGREED"})))

    def test_supported_with_a_vacuous_span_validates(self):
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("s1",))
        self.assertTrue(req.validated_requirement(
            r, judgement, evidence_by_span_id={"s1": self.DOCUMENTED},
            reconciliation=self._reconciliation({"s1": "VACUOUS"})))

    def test_supported_requires_the_cited_spans_own_text_to_contain_the_phrase(self):
        """issue #6 F9-R6-R2: the core content-check pin. A SUPPORTED judgement
        whose cited span is properly reconciled but whose OWN text genuinely
        lacks the expected phrase must never validate -- the citation's mere
        reality is not enough."""
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("s1",))
        self.assertFalse(req.validated_requirement(
            r, judgement, evidence_by_span_id={"s1": self.UNDOCUMENTED},
            reconciliation=self._reconciliation({"s1": "AGREED"})))

    def test_supported_never_borrows_content_from_an_uncited_span(self):
        """issue #6 F9-R6-R2, second re-review: the deeper fix -- even when a
        DIFFERENT, real, reconciled span genuinely contains the phrase, only
        the span the judgement actually CITED counts. A whole-document search
        standing in for a specific citation's content is exactly the gap the
        first-round fix left open."""
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("s1",))
        evidence = {"s1": self.UNDOCUMENTED, "s2": self.DOCUMENTED}
        self.assertFalse(req.validated_requirement(
            r, judgement, evidence_by_span_id=evidence,
            reconciliation=self._reconciliation({"s1": "AGREED", "s2": "AGREED"})))

    def test_supported_negated_span_never_validates(self):
        """issue #6 F9-R6-R6: a NEGATED mention in the cited span's own text
        ("not on the left") must never validate SUPPORTED -- a bare contiguous
        match is not enough, even scoped to the correct span."""
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("s1",))
        self.assertFalse(req.validated_requirement(
            r, judgement, evidence_by_span_id={"s1": self.NEGATED},
            reconciliation=self._reconciliation({"s1": "AGREED"})))

    def test_a_disagreed_span_never_validates(self):
        """SUPPORTED, with the phrase genuinely present in the cited span's own
        text (content check passes), but the cited span DISAGREED (reconciliation
        check must be what fails)."""
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("s1",))
        self.assertFalse(req.validated_requirement(
            r, judgement, evidence_by_span_id={"s1": self.DOCUMENTED},
            reconciliation=self._reconciliation({"s1": "DISAGREED"})))

    def test_an_unlisted_span_never_validates(self):
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("never-reconciled",))
        self.assertFalse(req.validated_requirement(
            r, judgement, evidence_by_span_id={"s1": self.DOCUMENTED},
            reconciliation=self._reconciliation({"s1": "AGREED"})))

    def test_supported_with_no_span_never_validates(self):
        """An evaluator claiming SUPPORTED must cite something -- unlike
        NOT_DOCUMENTED, absence-of-citation is not itself evidence here."""
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=())
        self.assertFalse(req.validated_requirement(
            r, judgement, evidence_by_span_id={}, reconciliation=self._reconciliation({})))

    def test_supported_never_validates_without_an_evidence_index(self):
        """`evidence_by_span_id=None` (no lookup supplied at all) must fail
        closed, exactly like an empty `coverage` does for NOT_DOCUMENTED."""
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("s1",))
        self.assertFalse(req.validated_requirement(
            r, judgement, evidence_by_span_id=None,
            reconciliation=self._reconciliation({"s1": "AGREED"})))

    def test_not_documented_with_no_span_validates(self):
        """Absence has nothing to cite by definition -- validating this is NOT, on
        its own, sufficient to eliminate anything (see resolution._grounded_
        elimination's deliberate additional gating)."""
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.NOT_DOCUMENTED,
            evidence_span_ids=())
        self.assertTrue(req.validated_requirement(
            r, judgement, coverage=self._coverage(self.UNDOCUMENTED)))

    def test_not_documented_requires_the_phrase_to_be_genuinely_absent(self):
        """issue #6 F9-R6-R4: a NOT_DOCUMENTED verdict for a phrase that IS
        actually present in the coverage corpus must never validate -- the
        judgement's claim contradicts the deterministic truth."""
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.NOT_DOCUMENTED,
            evidence_span_ids=())
        self.assertFalse(req.validated_requirement(
            r, judgement, coverage=self._coverage(self.DOCUMENTED)))

    def test_not_documented_never_validates_against_a_negated_mention(self):
        """issue #6 F9-R6-R6: a phrase that appears ONLY negated ("not on the
        left") in the coverage corpus is NOT the same claim as genuine silence
        -- NOT_DOCUMENTED must refuse to validate against it too, not just
        SUPPORTED."""
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.NOT_DOCUMENTED,
            evidence_span_ids=())
        self.assertFalse(req.validated_requirement(
            r, judgement, coverage=self._coverage(self.NEGATED)))

    def test_supported_never_validates_with_an_empty_span_text(self):
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("s1",))
        self.assertFalse(req.validated_requirement(
            r, judgement, evidence_by_span_id={"s1": ""},
            reconciliation=self._reconciliation({"s1": "AGREED"})))

    def test_not_documented_never_validates_with_no_coverage_supplied(self):
        """issue #6 F9-R6-R4: no real corpus was supplied at all, so no claim --
        positive OR negative -- can be made. Must fail closed, matching
        `CoverageCorpus.complete`'s own posture."""
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.NOT_DOCUMENTED,
            evidence_span_ids=())
        self.assertFalse(req.validated_requirement(r, judgement, coverage=None))

    def test_not_documented_never_validates_with_an_incomplete_coverage_corpus(self):
        """A coverage corpus with uncovered pages is not "complete" even if it
        has real text -- the empty-document/partial-coverage AND-gate."""
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.NOT_DOCUMENTED,
            evidence_span_ids=())
        import hashlib
        incomplete = req.CoverageCorpus(
            channel_id="test-channel", text=self.UNDOCUMENTED,
            text_sha256=hashlib.sha256(self.UNDOCUMENTED.encode()).hexdigest(),
            covered_pages=(1, 2), page_image_sha256=("h1", "h2"),
            uncovered_pages=(3,))
        self.assertFalse(req.validated_requirement(r, judgement, coverage=incomplete))

    def test_no_reconciliation_at_all_never_validates_a_cited_span(self):
        r = self._req()
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("s1",))
        self.assertFalse(req.validated_requirement(
            r, judgement, evidence_by_span_id={"s1": self.DOCUMENTED}, reconciliation=None))


class DeterministicStatusTest(unittest.TestCase):
    """Direct unit coverage of `requirement.deterministic_status` -- the actual
    text search `validated_requirement` defers to, never a verifier's claim."""

    def _req(self, expected=("left",)):
        return req.DescriptorRequirement(
            requirement_id="laterality:CAND_LEFT:0", axis="laterality",
            candidate_code="CAND_LEFT", required=True,
            role=req.RequirementRole.MUST_SUPPORT, expected=expected,
            authority_clause=expected[0], authority_offset=(0, len(expected[0])),
            authority_source_text=expected[0], selectable=True, queryable=True)

    def _coverage(self, text, uncovered_pages=()):
        import hashlib
        return req.CoverageCorpus(channel_id="test-channel", text=text,
                                  text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                  covered_pages=(1,), page_image_sha256=("stub-hash",),
                                  uncovered_pages=uncovered_pages)

    def test_present_phrase_is_supported(self):
        self.assertEqual(
            req.deterministic_status(self._req(),
                                     self._coverage("assembly service performed on the left")),
            req.RequirementStatus.SUPPORTED)

    def test_absent_phrase_is_not_documented(self):
        self.assertEqual(
            req.deterministic_status(self._req(), self._coverage("assembly service performed")),
            req.RequirementStatus.NOT_DOCUMENTED)

    def test_negated_phrase_is_contradicted(self):
        """issue #6 F9-R6-R6: a phrase occurring only negated maps to
        CONTRADICTED -- a genuinely different, more specific claim than either
        SUPPORTED or NOT_DOCUMENTED, so neither can validate against it."""
        self.assertEqual(
            req.deterministic_status(
                self._req(), self._coverage("assembly service, not on the left")),
            req.RequirementStatus.CONTRADICTED)

    def test_no_coverage_is_none(self):
        self.assertIsNone(req.deterministic_status(self._req(), None))

    def test_incomplete_coverage_is_none(self):
        self.assertIsNone(req.deterministic_status(
            self._req(), self._coverage("assembly service performed on the left",
                                        uncovered_pages=(2,))))

    def test_any_one_of_several_expected_terms_present_is_supported(self):
        """A candidate's inclusion-term requirement may list more than one
        alternative phrase (each compiled as its own DescriptorRequirement in
        production, but the underlying search itself must recognize any of a
        requirement's `expected` alternatives, not just the first)."""
        r = self._req(expected=("classic presentation", "atypical presentation"))
        self.assertEqual(
            req.deterministic_status(r, self._coverage("documented as atypical presentation today")),
            req.RequirementStatus.SUPPORTED)


class InstructionalTermsMockSourceTest(unittest.TestCase):
    """`MockSource.instructional_terms` mirrors `AuthoritativeSource`'s real
    ancestor-category rollup and ICD-10-only scoping exactly."""

    def test_rolls_up_from_ancestor_category(self):
        source = MockSource(instructional_terms={"A00": {"category-level term"}})
        self.assertEqual(source.instructional_terms("A00.0", "icd10"),
                         ("category-level term",))

    def test_non_icd10_system_returns_empty(self):
        source = MockSource(instructional_terms={"A000": {"classical cholera"}})
        self.assertEqual(source.instructional_terms("A00.0", "cpt"), ())

    def test_unconfigured_code_returns_empty_not_an_error(self):
        source = MockSource(instructional_terms={"A000": {"classical cholera"}})
        self.assertEqual(source.instructional_terms("Z99.9", "icd10"), ())


if __name__ == "__main__":
    unittest.main()
