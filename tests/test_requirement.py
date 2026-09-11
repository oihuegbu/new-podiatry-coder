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


class SemanticConceptRequirementTest(unittest.TestCase):
    """issue #6, Codex's independent re-review, root cause 3: `compile_requirements`
    now also projects each candidate's compiled semantic record (`semantics.
    compiled_record`'s `action_concepts`/`anatomy_concepts`) into MUST_SUPPORT
    requirements -- a governed source beyond the untyped `AXIS_DESCRIPTOR_TERM`
    bucket, but still only for the DIFFERENCE across the tied set, exactly the
    principle every other axis here already follows."""

    def _source(self, *candidates):
        return MockSource(records={(c.code, c.system):
                                   {"active": True, "long_description": c.descriptor}
                                   for c in candidates})

    def test_differing_action_concept_produces_a_must_support_requirement(self):
        exc = _cand("CAND_EXC", "Excision, lesion alpha")
        rep = _cand("CAND_REP", "Repair, lesion alpha")
        reqs = req.compile_requirements([exc, rep], source=self._source(exc, rep))
        action = {r.candidate_code: r for r in reqs if r.axis == "semantic_action"}
        self.assertEqual(set(action), {"CAND_EXC", "CAND_REP"})
        self.assertEqual(action["CAND_EXC"].expected, ("excision",))
        self.assertEqual(action["CAND_REP"].expected, ("repair",))
        self.assertTrue(all(r.role is req.RequirementRole.MUST_SUPPORT
                            for r in action.values()))
        # the SHARED anatomy target must never become a requirement -- it says
        # nothing about which candidate the record means.
        self.assertEqual([r for r in reqs if r.axis == "semantic_anatomy"], [])

    def test_differing_anatomy_concept_produces_a_must_support_requirement(self):
        alpha = _cand("CAND_ALPHA", "Excision, lesion alpha")
        beta = _cand("CAND_BETA", "Excision, lesion beta")
        reqs = req.compile_requirements([alpha, beta], source=self._source(alpha, beta))
        anatomy = {r.candidate_code: r for r in reqs if r.axis == "semantic_anatomy"}
        self.assertEqual(set(anatomy), {"CAND_ALPHA", "CAND_BETA"})
        self.assertEqual(anatomy["CAND_ALPHA"].expected, ("alpha",))
        self.assertEqual(anatomy["CAND_BETA"].expected, ("beta",))
        # the SHARED action must never become a requirement either.
        self.assertEqual([r for r in reqs if r.axis == "semantic_action"], [])

    def test_identical_semantic_concepts_produce_no_semantic_requirement(self):
        a = _cand("CAND_A2", "Excision, lesion alpha")
        b = _cand("CAND_B2", "Excision, lesion alpha")
        reqs = req.compile_requirements([a, b], source=self._source(a, b))
        self.assertEqual([r for r in reqs if r.axis in ("semantic_action", "semantic_anatomy")], [])

    def test_no_source_produces_no_semantic_requirement(self):
        """Mirrors every other source-dependent projection here (inclusion terms,
        snapshots): absent `source` degrades to nothing, never a guess."""
        alpha = _cand("CAND_ALPHA3", "Excision, lesion alpha")
        beta = _cand("CAND_BETA3", "Excision, lesion beta")
        reqs = req.compile_requirements([alpha, beta], source=None)
        self.assertEqual([r for r in reqs if r.axis in ("semantic_action", "semantic_anatomy")], [])

    def test_a_code_not_in_the_authoritative_source_is_silently_excluded(self):
        """`semantics.compiled_record` returns None for a code the source cannot
        look up (issue #6, item 1's own contract) -- that candidate contributes no
        semantic-concept requirement, but does not raise or block the OTHER
        candidate's requirement either."""
        known = _cand("CAND_KNOWN", "Excision, lesion alpha")
        unknown = _cand("CAND_UNKNOWN", "Excision, lesion beta")
        source = MockSource(records={("CAND_KNOWN", "cpt"):
                                     {"active": True, "long_description": known.descriptor}})
        reqs = req.compile_requirements([known, unknown], source=source)
        self.assertEqual([r for r in reqs if r.axis in ("semantic_action", "semantic_anatomy")], [])


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


class SemanticConceptEndToEndEliminationTest(unittest.TestCase):
    """issue #6, Codex's independent re-review, root cause 3, end to end: a
    documented action entailing several authoritative candidates that differ
    only in WHICH tissue/structure they name previously had no governed basis
    to eliminate any of them (Codex's exact reproduction: eight current CPT
    candidates, all "entailed", nothing distinguished them). The new
    semantic_anatomy requirement gives `_grounded_elimination` real material to
    confirm a model-named elimination against the document -- reproduced here
    with two candidates differing only in anatomy target, resolving uniquely
    to the one the note actually documents."""

    def _resolve(self, note_text):
        import hashlib
        from claude_coder.models import ClinicalFact, Disposition, EvidenceSpan, FactKind
        from claude_coder.resolution import resolve
        from claude_coder.eligibility import (ClaimComponent, ClaimLineIntent,
                                              EligibilityState, RetrievalRequest,
                                              fact_snapshot_digest)
        from app.contracts.source_evidence import (ReconciliationStatus,
                                                    SourceReconciliation,
                                                    SpanReconciliation)
        from tests import shortlist_verdict as _sv

        alpha = _cand("CAND_ALPHA_E2E", "Excision, lesion alpha", 0.90)
        beta = _cand("CAND_BETA_E2E", "Excision, lesion beta", 0.90)
        source = MockSource(
            records={(c.code, c.system): {"active": True, "long_description": c.descriptor}
                     for c in (alpha, beta)},
            retrieval={("*", "cpt"): [alpha, beta]})

        span = EvidenceSpan(text=note_text, anchored=True, start=0, end=len(note_text),
                            span_id="span-0")
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="excision",
                            disposition=Disposition.PERFORMED, evidence=[span],
                            confidence=0.99)
        intent = ClaimLineIntent(
            intent_id="t", encounter_id="test", component=ClaimComponent.SERVICE,
            clinical_event_ids=[fact.fact_id], fact_kind=fact.kind.value,
            clinical_action=fact.description, attributes=dict(fact.attributes),
            date_of_service=None, billing_entity_id=None, source_span_ids=[],
            state=EligibilityState.ELIGIBLE_FOR_RETRIEVAL,
            fact_digest=fact_snapshot_digest(fact))
        request = RetrievalRequest(intent, fact)

        reconciliation = SourceReconciliation(spans=(
            SpanReconciliation(span_id="span-0", status=ReconciliationStatus.AGREED,
                              pages=(1,)),))
        coverage = req.CoverageCorpus(
            channel_id="primary", text=note_text,
            text_sha256=hashlib.sha256(note_text.encode()).hexdigest(),
            covered_pages=(1,), uncovered_pages=(), page_image_sha256=("imgsha",))

        # The model judges beta NOT entailed (its own real judgement call --
        # unaffected by this fix) and, asked about the compiled semantic_anatomy
        # requirement, reports beta's own "beta" concept as not_documented --
        # `_grounded_elimination` then CONFIRMS that named elimination against
        # the document via the new requirement (never invents one the model
        # itself never made).
        llm = _sv.judge(entails=lambda d: "alpha" in d, reason="entailed",
                        requirement_status={"semantic_anatomy:CAND_BETA_E2E":
                                            "not_documented"})
        return resolve(request, source, llm=llm, corroborate=llm,
                       reconciliation=reconciliation, coverage=coverage)

    def test_a_model_named_elimination_is_confirmed_by_the_semantic_requirement(self):
        """The model itself already judged beta eliminated (a real judgement call,
        not manufactured by this fix); `_grounded_elimination` uses the new
        semantic_anatomy requirement to CONFIRM that elimination against the
        document, releasing alpha uniquely. This is the mechanism's actual,
        safe role: strengthening a model-named elimination with document proof,
        never inventing one independently -- see the companion safety test
        below for the case where the model names nothing."""
        line = self._resolve("excision of lesion alpha performed")
        self.assertIsNotNone(line.chosen, line.rationale)
        self.assertEqual(line.chosen.code, "CAND_ALPHA_E2E")

    def test_technique_only_difference_never_eliminates_on_its_own(self):
        """Companion safety check: `_grounded_elimination` only ever CONFIRMS an
        elimination a judging model itself already named
        (`_uniqueness_view`'s `named = [j.elimination_of(...)]` gate) -- it can
        never manufacture one purely from the compiled requirement. Two
        candidates a model finds BOTH entailed, with NEITHER named eliminated,
        must stay a tie regardless of what semantic_anatomy/semantic_action
        requirements exist, exactly like the untyped descriptor-term axis
        already never selects on its own."""
        from tests import shortlist_verdict as _sv
        from claude_coder.models import ResolutionMethod
        alpha = _cand("CAND_ALPHA_TIE", "Excision, lesion alpha", 0.90)
        beta = _cand("CAND_BETA_TIE", "Excision, lesion beta", 0.90)
        source = MockSource(
            records={(c.code, c.system): {"active": True, "long_description": c.descriptor}
                     for c in (alpha, beta)},
            retrieval={("*", "cpt"): [alpha, beta]})
        import hashlib
        from claude_coder.models import ClinicalFact, Disposition, EvidenceSpan, FactKind
        from claude_coder.resolution import resolve
        from claude_coder.eligibility import (ClaimComponent, ClaimLineIntent,
                                              EligibilityState, RetrievalRequest,
                                              fact_snapshot_digest)
        note_text = "excision performed"
        span = EvidenceSpan(text=note_text, anchored=True, start=0, end=len(note_text),
                            span_id="span-0")
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="excision",
                            disposition=Disposition.PERFORMED, evidence=[span],
                            confidence=0.99)
        intent = ClaimLineIntent(
            intent_id="t2", encounter_id="test", component=ClaimComponent.SERVICE,
            clinical_event_ids=[fact.fact_id], fact_kind=fact.kind.value,
            clinical_action=fact.description, attributes=dict(fact.attributes),
            date_of_service=None, billing_entity_id=None, source_span_ids=[],
            state=EligibilityState.ELIGIBLE_FOR_RETRIEVAL,
            fact_digest=fact_snapshot_digest(fact))
        request = RetrievalRequest(intent, fact)
        # entails both, names neither eliminated (declare=True with both entailed
        # and no distinguishing preference -> no elimination reason offered).
        llm = _sv.judge(entails=lambda d: True, reason="entailed")
        line = resolve(request, source, llm=llm, corroborate=llm)
        self.assertIsNone(line.chosen, line.rationale)
        self.assertIsNot(line.method, ResolutionMethod.DETERMINISTIC)


if __name__ == "__main__":
    unittest.main()
