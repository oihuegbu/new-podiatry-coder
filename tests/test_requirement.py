"""Descriptor/instructional-note requirement compilation and validation (issue #6
F9-R6, Phase 0). `compile_requirements` is a MECHANICAL PROJECTION of `tiebreak.
discriminating_axes` — never an independently invented axis compiler — so these
tests pin that projection rule (`required == probe.selectable`, silent axes produce
nothing, only `provable` axes compile) and `validated_requirement`'s independent
re-check (clause must reproduce, cited spans must be reconciled). Synthetic
descriptors/ids throughout — the mechanism reads descriptor grammar and a real ICD
Tabular data shape, never a term list.
"""
import hashlib
import unittest

from claude_coder import requirement as req
from claude_coder import resolution
from claude_coder.data_access import MockSource
from claude_coder.models import CandidateCode, ClinicalFact, EvidenceSpan, FactKind
from claude_coder import verify
from app.contracts.source_evidence import (ReconciliationStatus, SourceReconciliation,
                                           SpanReconciliation)


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


class SemanticConceptRequirementRemovedTest(unittest.TestCase):
    """issue #6, Codex's independent re-review, root cause 3: a prior round's
    `_semantic_concept_requirements` (projecting `semantics.compiled_record`'s
    raw `action_concepts`/`anatomy_concepts` tokens into MUST_SUPPORT
    requirements) was REMOVED, not merely disabled -- Codex found it compiled
    an ungoverned structural parse as if it were a validated concept (a
    technique/approach word indistinguishable from real anatomy), AND a
    multi-token `expected` tuple that `asserted_status` treats as ANY-term-
    supported, letting an unrelated documented word "confirm" a requirement
    whose real, unstated term was absent.

    Round 3: `_semantic_anatomy_requirements` (see `SemanticAnatomyRequirementTest`
    below) is the governed replacement -- a `semantic_anatomy` axis CAN now be
    compiled, but only from a target phrase resolved against a real SNOMED
    concept graph. This test's OWN fixture (`MockSource` with no
    `concept_lookup` configured for "lesion alpha") never resolves, so no
    `semantic_anatomy` requirement is produced for it either -- pinned here so
    an ungoverned structural token specifically can never silently reappear as
    a requirement, without over-claiming that the axis itself stays forever
    unpopulated in general."""

    def test_an_ungoverned_structural_token_never_compiles_into_either_axis(self):
        source = MockSource(records={
            ("CAND_EXC", "cpt"): {"active": True, "long_description": "Excision, lesion alpha"},
            ("CAND_REP", "cpt"): {"active": True, "long_description": "Repair, lesion alpha"}})
        exc = _cand("CAND_EXC", "Excision, lesion alpha")
        rep = _cand("CAND_REP", "Repair, lesion alpha")
        reqs = req.compile_requirements([exc, rep], source=source)
        self.assertEqual(
            [r for r in reqs if r.axis in ("semantic_action", "semantic_anatomy")], [])


class SemanticAnatomyRequirementTest(unittest.TestCase):
    """issue #6, Codex's independent re-review (round 3): the governed
    replacement for the anatomy half of the removed `_semantic_concept_
    requirements` -- `requirement._semantic_anatomy_requirements`, wired into
    `compile_requirements` so its output flows through the EXISTING
    `validated_requirement`/`resolution._grounded_elimination` path. Reuses
    the SAME governed SNOMED Body Structure concept index (`concept_lookup`)
    and the SAME atomic target decomposition
    `semantic_eligibility._anatomy_compatibility` already trusts -- never a
    new capability invented here. Synthetic, agnostic vocabulary throughout
    ("structure alpha/beta/gamma"), matching this file's existing convention."""

    ALPHA = _cand("CAND_ALPHA", "assembly service, structure alpha")
    BETA = _cand("CAND_BETA", "assembly service, structure beta")

    def _records(self, *cands):
        return {(c.code, c.system): {"active": True, "long_description": c.descriptor}
               for c in cands}

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

    def test_provenance_carries_the_terminology_source_and_descriptor_snapshot(self):
        """issue #6, Codex's independent re-review (round 4): independent
        reproduction with a terminology version in the lookup returned a
        requirement carrying the concept id but NO terminology source/version
        or descriptor snapshot -- not yet defensible lineage. Both must now
        survive into `source_identity`, exactly like the descriptor axis above
        already carries `record_snapshot_identity`."""
        source = MockSource(
            records=self._records(self.ALPHA, self.BETA),
            snapshot={"source_id": "cpt_codes", "sha256": "deadbeef", "size": 123},
            concept_lookup={
                "structure alpha": {"candidates": ["C_ALPHA"], "unique": True,
                                    "expansions": [],
                                    "source_identity": {"source_id": "snomed_concept_terms",
                                                        "sha256": "abc123", "size": 456}},
                "structure beta": {"candidates": ["C_BETA"], "unique": True, "expansions": []}})
        reqs = [r for r in req.compile_requirements([self.ALPHA, self.BETA], source=source)
               if r.axis == "semantic_anatomy" and r.candidate_code == "CAND_ALPHA"]
        self.assertEqual(len(reqs), 1)
        identity = reqs[0].source_identity
        self.assertEqual(identity["terminology_identity"],
                         {"source_id": "snomed_concept_terms", "sha256": "abc123", "size": 456})
        self.assertEqual(identity["descriptor_snapshot"],
                         {"source_id": "cpt_codes", "sha256": "deadbeef", "size": 123})

    def test_an_unresolved_technique_word_never_compiles_a_semantic_anatomy_requirement(self):
        """The SAME defect-1 fixture as the removed-code pin above, through the
        NEW mechanism this time: "powered"/"manual" are technique words this
        repo's SNOMED Body Structure graph simply does not contain, so an
        unconfigured `concept_lookup` (real behavior for an unrecognized term)
        must still drop them -- never mistaken for anatomy just because a real
        governed axis now exists."""
        source = MockSource(records=self._records(POWERED, MANUAL))
        reqs = req.compile_requirements([POWERED, MANUAL], source=source)
        self.assertEqual([r for r in reqs if r.axis == "semantic_anatomy"], [])

    def test_a_governed_distinguishing_anatomy_target_compiles_one_atomic_requirement_each(self):
        source = MockSource(
            records=self._records(self.ALPHA, self.BETA),
            concept_lookup={
                "structure alpha": {"candidates": ["C_ALPHA"], "unique": True, "expansions": []},
                "structure beta": {"candidates": ["C_BETA"], "unique": True, "expansions": []}})
        reqs = [r for r in req.compile_requirements([self.ALPHA, self.BETA], source=source)
               if r.axis == "semantic_anatomy"]
        by_code = {r.candidate_code: r for r in reqs}
        self.assertEqual(set(by_code), {"CAND_ALPHA", "CAND_BETA"})
        self.assertEqual(by_code["CAND_ALPHA"].expected, ("structure alpha",))
        self.assertEqual(by_code["CAND_BETA"].expected, ("structure beta",))
        self.assertEqual(by_code["CAND_ALPHA"].role, req.RequirementRole.MUST_SUPPORT)
        self.assertTrue(by_code["CAND_ALPHA"].selectable)

    def test_two_distinct_governed_targets_never_bundle_into_one_expected(self):
        """issue #6, Codex's independent re-review, root cause 3, defect 2: the
        direct regression pin. A candidate naming TWO alternative, DIFFERENT
        governed concepts ("structure alpha OR structure beta") must compile
        TWO separate requirements -- bundling both into one `expected` tuple is
        exactly the shape that let `asserted_status`'s ANY-of-tuple semantics
        "confirm" a requirement whose real, unstated term was absent."""
        both = _cand("CAND_BOTH", "assembly service, structure alpha or structure beta")
        other = _cand("CAND_OTHER", "assembly service, structure gamma")
        source = MockSource(
            records=self._records(both, other),
            concept_lookup={
                "structure alpha": {"candidates": ["C_ALPHA"], "unique": True, "expansions": []},
                "structure beta": {"candidates": ["C_BETA"], "unique": True, "expansions": []},
                "structure gamma": {"candidates": ["C_GAMMA"], "unique": True, "expansions": []}})
        reqs = [r for r in req.compile_requirements([both, other], source=source)
               if r.axis == "semantic_anatomy" and r.candidate_code == "CAND_BOTH"]
        self.assertEqual(len(reqs), 2, "two distinct concepts must be TWO requirements")
        all_expected = [set(r.expected) for r in reqs]
        self.assertIn({"structure alpha"}, all_expected)
        self.assertIn({"structure beta"}, all_expected)

    def test_a_concept_shared_by_every_tied_candidate_is_never_a_requirement(self):
        """A concept every tied candidate's OWN descriptor requires says nothing
        about which one the record means -- validated NOT_DOCUMENTED on a
        shared concept would otherwise let `_grounded_elimination` "confirm"
        eliminating either one on a defect the other equally has."""
        alpha2 = _cand("CAND_ALPHA2", "assembly service, structure alpha")
        source = MockSource(
            records=self._records(self.ALPHA, alpha2),
            concept_lookup={
                "structure alpha": {"candidates": ["C_ALPHA"], "unique": True, "expansions": []}})
        reqs = [r for r in req.compile_requirements([self.ALPHA, alpha2], source=source)
               if r.axis == "semantic_anatomy"]
        self.assertEqual(reqs, [],
                         "a concept every tied candidate shares says nothing about "
                         "which one the record means")

    def test_a_candidates_own_alternative_sharing_the_tied_concept_compiles_no_requirement(self):
        """issue #6, independent review: reproduces the exact real-note failure
        mode found against CPT 28118/28119/28120 -- a candidate whose own
        descriptor names TWO alternative anatomy targets ("structure alpha or
        structure beta") where ONE of those alternatives (beta) is a concept
        every OTHER tied candidate also accepts. `structure beta` alone is
        correctly excluded as non-discriminating (the prior test above), but
        the bug was promoting the candidate's OTHER own alternative
        (`structure alpha`, unique to this one candidate) into a MUST_SUPPORT
        requirement anyway -- demanding the record document `alpha`
        specifically, when the candidate's own descriptor already treats
        alpha/beta as interchangeable (an OR, not an AND) and `beta` alone,
        which the tied set already shares, is fully sufficient. That
        manufactured a requirement no candidate's own descriptor actually
        imposes, producing an unresolvable split between independent
        evaluators: one correctly reads the descriptor's real OR semantics as
        satisfied by the documented `beta`; the other correctly finds the
        wrongly-compiled `alpha`-only requirement unmet. Live symptom: CPT
        28120 ("...talus or calcaneus") tied against 28118/28119 (calcaneus
        only) over a documented calcaneus procedure -- `calcaneus` is shared
        and correctly excluded, but `talus` (28120's OTHER own alternative)
        was wrongly compiled into a forced requirement, and the designated
        operative note held on exactly this pattern (F9-R23-round-N,
        independent-anatomy-alternative regression)."""
        either = _cand("CAND_EITHER", "assembly service, structure alpha or structure beta")
        rival = _cand("CAND_RIVAL", "assembly service, structure beta")
        source = MockSource(
            records=self._records(either, rival),
            concept_lookup={
                "structure alpha": {"candidates": ["C_ALPHA"], "unique": True, "expansions": []},
                "structure beta": {"candidates": ["C_BETA"], "unique": True, "expansions": []}})
        reqs = [r for r in req.compile_requirements([either, rival], source=source)
               if r.axis == "semantic_anatomy"]
        self.assertEqual(
            reqs, [],
            "a candidate already satisfiable through an anatomy concept the whole "
            "tied set shares must not ALSO be held to its own other alternative -- "
            "its alternatives are an OR, not an AND")

    def test_an_ambiguous_match_never_compiles_a_requirement(self):
        """Fail-closed, the same discipline `_anatomy_compatibility` already
        applies: a term resolving to MORE than one concept id is not governed
        enough to eliminate anything on."""
        source = MockSource(
            records=self._records(self.ALPHA, self.BETA),
            concept_lookup={
                "structure alpha": {"candidates": ["C_ALPHA", "C_ALPHA2"], "unique": False,
                                    "expansions": []},
                "structure beta": {"candidates": ["C_BETA"], "unique": True, "expansions": []}})
        reqs = [r for r in req.compile_requirements([self.ALPHA, self.BETA], source=source)
               if r.axis == "semantic_anatomy" and r.candidate_code == "CAND_ALPHA"]
        self.assertEqual(reqs, [], "an ambiguous concept match must never ground anything")

    def test_a_governed_synonym_is_included_in_expected_and_validates_supported(self):
        """`expected` carries the target phrase PLUS its own known governed
        synonym for the SAME concept -- never a second, distinct concept's
        words (that would be defect 2 again). A note using only the synonym
        must still validate SUPPORTED, proving the tolerance is real, not
        merely declared."""
        source = MockSource(
            records=self._records(self.ALPHA, self.BETA),
            concept_lookup={
                "structure alpha": {"candidates": ["C_ALPHA"], "unique": True,
                                    "expansions": ["alpha structure synonym"]},
                "structure beta": {"candidates": ["C_BETA"], "unique": True, "expansions": []}})
        reqs = [r for r in req.compile_requirements([self.ALPHA, self.BETA], source=source)
               if r.axis == "semantic_anatomy" and r.candidate_code == "CAND_ALPHA"]
        self.assertEqual(len(reqs), 1)
        r = reqs[0]
        self.assertEqual(set(r.expected), {"structure alpha", "alpha structure synonym"})

        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("s1",))
        span_text = "note documents the alpha structure synonym today"
        self.assertTrue(req.validated_requirement(
            r, judgement, evidence_by_span_id={"s1": span_text},
            reconciliation=self._reconciliation({"s1": "AGREED"})))

    def test_a_validated_not_documented_verdict_grounds_exactly_like_any_other_must_support_axis(
            self):
        """End-to-end proof this axis is wired through the EXISTING plumbing,
        not a parallel one: its compiled requirement, run through the SAME
        `validated_requirement` every other MUST_SUPPORT axis uses, correctly
        validates NOT_DOCUMENTED against a corpus that truly never mentions
        the target (nor its own synonyms) -- exactly the shape `resolution.
        _grounded_elimination` consumes to eliminate a candidate."""
        source = MockSource(
            records=self._records(self.ALPHA, self.BETA),
            concept_lookup={
                "structure alpha": {"candidates": ["C_ALPHA"], "unique": True, "expansions": []},
                "structure beta": {"candidates": ["C_BETA"], "unique": True, "expansions": []}})
        reqs = [r for r in req.compile_requirements([self.ALPHA, self.BETA], source=source)
               if r.axis == "semantic_anatomy" and r.candidate_code == "CAND_ALPHA"]
        self.assertEqual(len(reqs), 1)
        r = reqs[0]
        self.assertEqual(r.role, req.RequirementRole.MUST_SUPPORT)

        coverage = self._coverage("assembly service performed today, no anatomy stated")
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.NOT_DOCUMENTED)
        self.assertTrue(req.validated_requirement(r, judgement, coverage=coverage))


class SemanticActionAndQualifierRequirementTest(unittest.TestCase):
    """issue #6, Codex's independent re-review (F9-R13-C): the action and
    qualifier halves of the governed semantic-requirement system, wired
    through `compile_requirements()` (compile time) and the new
    `requirement.semantic_axis_status()` (judgement time) -- never a raw
    literal-text or blind-selectable path. `selectable=False` on both keeps
    them out of `tiebreak._axes_from_requirements`'s generic narrowing;
    selection happens only via `resolution._select_by_semantic_axes`,
    exercised separately in tests/test_tie_policy.py."""

    ACTION_A = _cand("CAND_ACTION_A", "assembly procedure, structure alpha")
    ACTION_B = _cand("CAND_ACTION_B", "installation procedure, structure alpha")

    def _records(self, *cands):
        return {(c.code, c.system): {"active": True, "long_description": c.descriptor}
               for c in cands}

    def test_an_unmapped_action_phrase_never_compiles_a_requirement(self):
        """Fail-closed, the same discipline the anatomy axis already applies:
        a phrase this graph does not recognize even against ITSELF (no
        `procedure_relation` entry configured at all) is not governed enough
        to require anything on."""
        source = MockSource(records=self._records(self.ACTION_A, self.ACTION_B))
        reqs = req.compile_requirements([self.ACTION_A, self.ACTION_B], source=source)
        self.assertEqual([r for r in reqs if r.axis == "semantic_action"], [])

    def test_a_governed_action_compiles_one_atomic_selectable_false_requirement(self):
        """issue #6, Codex's independent re-review (F9-R13-C, round 2/P1-B):
        the governed self-check (and the judgement-time comparison) now run
        against each candidate's FULL official descriptor, never a
        structural comma-split prefix -- the exact-SHA reproduction proved a
        prefix-only phrase frequently omits the target that makes the
        concept specific."""
        source = MockSource(
            records=self._records(self.ACTION_A, self.ACTION_B),
            procedure_relation={
                (self.ACTION_A.descriptor, self.ACTION_A.descriptor): {
                    "verdict": "same",
                    "term_a": {"candidates": ["C_ACTION_A"], "unique": True},
                    "source_identity": {"source_id": "snomed_procedure_terms",
                                        "sha256": "abc"}},
                (self.ACTION_B.descriptor, self.ACTION_B.descriptor): {
                    "verdict": "same",
                    "term_a": {"candidates": ["C_ACTION_B"], "unique": True}}})
        reqs = [r for r in req.compile_requirements([self.ACTION_A, self.ACTION_B], source=source)
               if r.axis == "semantic_action"]
        by_code = {r.candidate_code: r for r in reqs}
        self.assertEqual(set(by_code), {"CAND_ACTION_A", "CAND_ACTION_B"})
        self.assertEqual(by_code["CAND_ACTION_A"].expected, (self.ACTION_A.descriptor,))
        self.assertEqual(by_code["CAND_ACTION_A"].role, req.RequirementRole.MUST_SUPPORT)
        self.assertFalse(by_code["CAND_ACTION_A"].selectable,
                         "must never enter tiebreak's generic literal narrowing")
        self.assertEqual(by_code["CAND_ACTION_A"].source_identity["concept_id"], "C_ACTION_A")
        self.assertEqual(by_code["CAND_ACTION_A"].source_identity["terminology_identity"],
                         {"source_id": "snomed_procedure_terms", "sha256": "abc"})

    def test_an_ambiguous_action_match_never_compiles_a_requirement(self):
        """Fail-closed, the same discipline the anatomy axis already applies:
        a descriptor resolving to MORE than one concept id is not governed
        enough to require anything on."""
        source = MockSource(
            records=self._records(self.ACTION_A, self.ACTION_B),
            procedure_relation={
                (self.ACTION_A.descriptor, self.ACTION_A.descriptor): {
                    "verdict": "same",
                    "term_a": {"candidates": ["C1", "C2"], "unique": False}}})
        reqs = [r for r in req.compile_requirements([self.ACTION_A, self.ACTION_B], source=source)
               if r.axis == "semantic_action" and r.candidate_code == "CAND_ACTION_A"]
        self.assertEqual(reqs, [])

    def test_qualifier_requirements_never_compile_from_the_descriptor_tail(self):
        """issue #6, Codex's independent re-review (F9-R13-C, round 2/P1-B):
        `_semantic_qualifier_requirements` must never manufacture an
        `approach` requirement from `ontology.parse_descriptor`'s
        `anatomy_phrase` -- the source itself calls it anatomy, and
        re-labeling it a qualifier was type confusion, reproduced exactly on
        differing tails that are plainly anatomy, not approach."""
        differing_a = _cand("CAND_DIFF_A", "assembly procedure, structure alpha")
        differing_b = _cand("CAND_DIFF_B", "assembly procedure, structure beta")
        source = MockSource(records=self._records(differing_a, differing_b))
        reqs = req.compile_requirements([differing_a, differing_b], source=source)
        self.assertEqual([r for r in reqs if r.axis == "semantic_qualifier"], [])
        self.assertEqual(req._semantic_qualifier_requirements(
            [differing_a, differing_b], source), [])


class SemanticAxisStatusTest(unittest.TestCase):
    """`requirement.semantic_axis_status()` -- the judgement-time truth test
    for `semantic_action`/`semantic_qualifier` requirements (issue #6,
    Codex's independent re-review, F9-R13-C)."""

    def _action_req(self, phrase, code="CAND_A"):
        return req.DescriptorRequirement(
            requirement_id=f"semantic_action:{code}:0", axis="semantic_action",
            candidate_code=code, required=True, role=req.RequirementRole.MUST_SUPPORT,
            expected=(phrase,), authority_clause=phrase, authority_offset=(0, len(phrase)),
            authority_source_text=phrase, selectable=False, queryable=False)

    def _qualifier_req(self, phrase, code="CAND_A"):
        return req.DescriptorRequirement(
            requirement_id=f"semantic_qualifier:{code}:0", axis="semantic_qualifier",
            candidate_code=code, required=True, role=req.RequirementRole.MUST_SUPPORT,
            expected=(phrase,), authority_clause=phrase, authority_offset=(0, len(phrase)),
            authority_source_text=phrase, selectable=False, queryable=False)

    def _fact(self, description, *, attributes=None, attribute_evidence=None):
        from claude_coder.models import ClinicalFact, Disposition, EvidenceSpan, FactKind
        span = EvidenceSpan(text=description, start=0, end=len(description),
                           anchored=True, span_id="s1")
        return ClinicalFact(kind=FactKind.PROCEDURE, description=description,
                            attributes=dict(attributes or {}),
                            disposition=Disposition.PERFORMED, evidence=[span],
                            confidence=0.9, fact_id="F1",
                            attribute_evidence=dict(attribute_evidence or {}))

    def _approach_evidence(self, value, text="documented approach text"):
        """A claim-AUTHORIZED `approach` value -- `graph_consensus.
        claim_authorized_value` requires a reconciled, ASSERTED, value-bound
        `attribute_evidence` entry, never a bare `attributes` read (the same
        bar laterality/service_role are already held to)."""
        from claude_coder.models import AttributeEvidence, EvidenceSpan, RelationState
        span = EvidenceSpan(text=text, start=0, end=len(text), anchored=True, span_id="s1")
        return {"approach": (AttributeEvidence(span=span, scope="local",
                                               assertion_state=RelationState.ASSERTED,
                                               value=value),)}

    def test_action_synonym_with_unique_governed_identity_is_supported(self):
        """The fact's OWN description uses a governed SYNONYM of the
        candidate's descriptor action phrase -- never a literal match --
        proving equivalence is established via the concept graph, not text
        presence."""
        source = MockSource(procedure_relation={
            ("fitting procedure performed", "assembly procedure"): {"verdict": "same"}})
        r = self._action_req("assembly procedure")
        fact = self._fact("fitting procedure performed")
        self.assertEqual(req.semantic_axis_status(r, fact, None, source),
                         req.RequirementStatus.SUPPORTED)

    def test_ambiguous_or_unmapped_action_abstains(self):
        """No configured relation at all -- CONCEPT_UNRESOLVED -- must read as
        UNRESOLVED, never SUPPORTED and never CONTRADICTED (no typed opposite
        signal exists for a documented action)."""
        source = MockSource()   # nothing configured -> CONCEPT_UNRESOLVED
        r = self._action_req("assembly procedure")
        fact = self._fact("an unrelated procedure was performed")
        self.assertEqual(req.semantic_axis_status(r, fact, None, source),
                         req.RequirementStatus.UNRESOLVED)

    def test_qualifier_is_supported_only_by_its_own_typed_approach_attribute(self):
        r = self._qualifier_req("open approach")
        fact = self._fact("procedure performed", attributes={"approach": "open approach"},
                          attribute_evidence=self._approach_evidence("open approach"))
        self.assertEqual(req.semantic_axis_status(r, fact, None, MockSource()),
                         req.RequirementStatus.SUPPORTED)

    def test_qualifier_cannot_borrow_anatomy_or_other_attribute_evidence(self):
        """issue #6, Codex's independent re-review (F9-R13-C): qualifier
        support must come ONLY from this fact's own typed `approach` value --
        never from a DIFFERENT attribute (here, `anatomy`) that happens to
        carry the identical text, even with equally real, reconciled
        evidence behind IT. Proves no silent fallback/borrowing across
        attribute keys."""
        from claude_coder.models import AttributeEvidence, EvidenceSpan, RelationState
        span = EvidenceSpan(text="open approach documented", start=0, end=10,
                           anchored=True, span_id="s1")
        r = self._qualifier_req("open approach")
        fact = self._fact(
            "procedure performed", attributes={"anatomy": "open approach"},
            attribute_evidence={"anatomy": (AttributeEvidence(
                span=span, scope="local", assertion_state=RelationState.ASSERTED,
                value="open approach"),)})
        self.assertEqual(req.semantic_axis_status(r, fact, None, MockSource()),
                         req.RequirementStatus.UNRESOLVED)

    def test_qualifier_with_no_typed_evidence_at_all_is_unresolved(self):
        r = self._qualifier_req("open approach")
        fact = self._fact("procedure performed")
        self.assertEqual(req.semantic_axis_status(r, fact, None, MockSource()),
                         req.RequirementStatus.UNRESOLVED)

    def test_qualifier_with_a_different_documented_approach_is_contradicted(self):
        """A genuinely different, positively documented approach is an
        explicit typed CONTRADICTED -- not silent UNRESOLVED -- since the
        fact affirmatively states something else."""
        r = self._qualifier_req("open approach")
        fact = self._fact(
            "procedure performed", attributes={"approach": "percutaneous approach"},
            attribute_evidence=self._approach_evidence("percutaneous approach"))
        self.assertEqual(req.semantic_axis_status(r, fact, None, MockSource()),
                         req.RequirementStatus.CONTRADICTED)


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


class ExclusionClauseRequirementTest(unittest.TestCase):
    """issue #6, Codex's independent re-review (F9-R19-A): a candidate's own
    descriptor stating "except"/"excluding"/"other than" compiles into a
    `RequirementRole.EXCLUSION` requirement -- the opposite polarity of every
    other elimination-eligible axis: genuinely DOCUMENTING the excluded
    condition (validated SUPPORTED) is what eliminates the candidate that
    carries it, never its absence. Synthetic descriptors throughout."""

    WITH_EXCLUSION = _cand("CAND_EXC", "assembly service, except powered variant")
    PLAIN = _cand("CAND_PLAIN", "assembly service performed")

    def _coverage(self, text):
        import hashlib
        return req.CoverageCorpus(channel_id="test-channel", text=text,
                                  text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                  covered_pages=(1,), page_image_sha256=("stub-hash",))

    def _reconciliation(self, statuses):
        from app.contracts.source_evidence import (ReconciliationStatus,
                                                    SourceReconciliation,
                                                    SpanReconciliation)
        return SourceReconciliation(spans=tuple(
            SpanReconciliation(span_id=sid, status=ReconciliationStatus[status])
            for sid, status in statuses.items()))

    def test_an_exclusion_clause_compiles_as_a_required_exclusion_role_requirement(self):
        reqs = [r for r in req.compile_requirements([self.WITH_EXCLUSION, self.PLAIN])
               if r.axis == "exclusion_clause"]
        self.assertEqual(len(reqs), 1)     # only CAND_EXC states one
        r = reqs[0]
        self.assertEqual(r.candidate_code, "CAND_EXC")
        self.assertTrue(r.required)
        self.assertEqual(r.role, req.RequirementRole.EXCLUSION)
        self.assertEqual(r.expected, ("powered variant",))
        self.assertEqual(r.authority_source_text, self.WITH_EXCLUSION.descriptor)

    def test_a_descriptor_stating_no_exclusion_compiles_nothing_for_that_candidate(self):
        reqs = [r for r in req.compile_requirements([self.WITH_EXCLUSION, self.PLAIN])
               if r.axis == "exclusion_clause"]
        self.assertEqual([r for r in reqs if r.candidate_code == "CAND_PLAIN"], [])

    def test_two_candidates_with_no_exclusion_clause_compile_nothing(self):
        a = _cand("CAND_A", "assembly service performed")
        b = _cand("CAND_B", "assembly service performed differently")
        reqs = [r for r in req.compile_requirements([a, b]) if r.axis == "exclusion_clause"]
        self.assertEqual(reqs, [])

    def test_exclusion_clause_validates_supported_through_the_existing_plumbing(self):
        """Wired through the SAME `validated_requirement` every other axis
        uses -- proves this is not a parallel, unvetted mechanism."""
        reqs = [r for r in req.compile_requirements([self.WITH_EXCLUSION, self.PLAIN])
               if r.axis == "exclusion_clause"]
        r = reqs[0]
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("s1",))
        span_text = "documentation confirms the powered variant was used today"
        self.assertTrue(req.validated_requirement(
            r, judgement, evidence_by_span_id={"s1": span_text},
            reconciliation=self._reconciliation({"s1": "AGREED"})))

    def test_exclusion_clause_not_documented_validates_against_full_coverage(self):
        reqs = [r for r in req.compile_requirements([self.WITH_EXCLUSION, self.PLAIN])
               if r.axis == "exclusion_clause"]
        r = reqs[0]
        coverage = self._coverage("assembly service performed today, nothing else stated")
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.NOT_DOCUMENTED)
        self.assertTrue(req.validated_requirement(r, judgement, coverage=coverage))

    def _judgements(self, requirement_ids, statuses, evidence_span_ids=("s1",)):
        """Both judging models unanimously reporting the SAME status for each
        of `requirement_ids` -- the duck-typed shape `resolution.
        _grounded_elimination` reads (`j.requirement_judgements`)."""
        rjs = tuple(req.RequirementJudgement(
            requirement_id=rid, status=statuses,
            evidence_span_ids=(evidence_span_ids if statuses == req.RequirementStatus.SUPPORTED
                               else ()))
            for rid in requirement_ids)
        model_a = type("J", (), {"requirement_judgements": rjs})()
        model_b = type("J", (), {"requirement_judgements": rjs})()
        return [model_a, model_b]

    def test_grounded_elimination_eliminates_on_a_documented_exclusion_condition(self):
        """End-to-end through the REAL `resolution._grounded_elimination` --
        the actual function `_uniqueness_view` calls, not merely
        `validated_requirement` in isolation. Both models named CAND_EXC
        eliminated (a real precondition of this function ever being
        consulted at all -- see `_uniqueness_view`); this proves the
        DOCUMENT-GROUNDING check for an EXCLUSION-role axis fires on
        SUPPORTED, the opposite polarity from every MUST_SUPPORT axis."""
        from claude_coder import resolution
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from app.contracts.source_evidence import (ReconciliationStatus,
                                                    SourceReconciliation,
                                                    SpanReconciliation)

        span = EvidenceSpan(text="documentation confirms the powered variant was used",
                            anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="assembly service",
                            evidence=[span], confidence=0.9, fact_id="F1")
        reqs = req.compile_requirements([self.WITH_EXCLUSION, self.PLAIN])
        excl_req = next(r for r in reqs if r.axis == "exclusion_clause")
        reconciliation = SourceReconciliation(spans=(
            SpanReconciliation(span_id="s1", status=ReconciliationStatus.AGREED),))
        judgements = self._judgements((excl_req.requirement_id,),
                                      req.RequirementStatus.SUPPORTED)

        grounded, detail = resolution._grounded_elimination(
            fact, self.WITH_EXCLUSION, self.PLAIN, reconciliation,
            requirements=reqs, judgements=judgements, coverage=None)
        self.assertTrue(grounded, detail)
        self.assertIn("SUPPORTED", detail)

    def test_grounded_elimination_never_eliminates_on_an_undocumented_exclusion(self):
        """The mirror case: the SAME exclusion requirement, but every
        evaluator reports NOT_DOCUMENTED for it -- the excluded condition is
        genuinely absent, so CAND_EXC's own descriptor's exception does NOT
        apply, and nothing grounds an elimination through this axis. Falls
        through to the pairwise/word-overlap fallback, which also finds
        nothing distinguishing these two synthetic descriptors."""
        from claude_coder import resolution
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind

        span = EvidenceSpan(text="assembly service performed, nothing else stated",
                            anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="assembly service",
                            evidence=[span], confidence=0.9, fact_id="F1")
        reqs = req.compile_requirements([self.WITH_EXCLUSION, self.PLAIN])
        excl_req = next(r for r in reqs if r.axis == "exclusion_clause")
        coverage = self._coverage("assembly service performed, nothing else stated")
        judgements = self._judgements((excl_req.requirement_id,),
                                      req.RequirementStatus.NOT_DOCUMENTED)

        grounded, _detail = resolution._grounded_elimination(
            fact, self.WITH_EXCLUSION, self.PLAIN, reconciliation=None,
            requirements=reqs, judgements=judgements, coverage=coverage)
        self.assertFalse(grounded)


class IndicationClauseRequirementTest(unittest.TestCase):
    """issue #6, independent root-cause investigation (real-data replay
    against the designated note, F1/CPT 28118 vs 28120): a candidate's own
    descriptor stating a positive "(eg, ...)"/"(e.g., ...)"/"(for example,
    ...)" indication clause.

    Codex's independent re-review (P1-2 correction) found the original
    version of this axis unsafe: it compiled into a `RequirementRole.
    MUST_SUPPORT` requirement, letting an undocumented, purely-illustrative
    example (AMA/CPT convention: "(eg, ...)" is non-exhaustive, never a
    checklist) alone ground a candidate's elimination or a provider
    question, with no typed, authoritative requirement field or
    source-governed rule behind it. It now compiles as
    `RequirementRole.POSITIVE_ALIAS` (`required=False`) -- recorded for
    recall/audit/explanation only, exactly like an ICD-10-CM inclusion
    term, never elimination-eligible. Synthetic descriptors throughout."""

    WITH_INDICATION = _cand("CAND_IND", "assembly service (eg, variant condition)")
    PLAIN = _cand("CAND_PLAIN", "assembly service performed")

    def _coverage(self, text):
        import hashlib
        return req.CoverageCorpus(channel_id="test-channel", text=text,
                                  text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                  covered_pages=(1,), page_image_sha256=("stub-hash",))

    def _reconciliation(self, statuses):
        from app.contracts.source_evidence import (ReconciliationStatus,
                                                    SourceReconciliation,
                                                    SpanReconciliation)
        return SourceReconciliation(spans=tuple(
            SpanReconciliation(span_id=sid, status=ReconciliationStatus[status])
            for sid, status in statuses.items()))

    def test_an_indication_clause_compiles_as_a_non_required_positive_alias(self):
        """P1-2 correction: an illustrative "(eg, ...)" clause compiles for
        recall/audit only -- never `required`, never `MUST_SUPPORT` -- so it
        can never ground an elimination or a provider question on its own."""
        reqs = [r for r in req.compile_requirements([self.WITH_INDICATION, self.PLAIN])
               if r.axis == "indication_clause"]
        self.assertEqual(len(reqs), 1)     # only CAND_IND states one
        r = reqs[0]
        self.assertEqual(r.candidate_code, "CAND_IND")
        self.assertFalse(r.required)
        self.assertEqual(r.role, req.RequirementRole.POSITIVE_ALIAS)
        self.assertEqual(r.expected, ("variant condition",))
        self.assertEqual(r.authority_source_text, self.WITH_INDICATION.descriptor)

    def test_a_descriptor_stating_no_indication_compiles_nothing_for_that_candidate(self):
        reqs = [r for r in req.compile_requirements([self.WITH_INDICATION, self.PLAIN])
               if r.axis == "indication_clause"]
        self.assertEqual([r for r in reqs if r.candidate_code == "CAND_PLAIN"], [])

    def test_two_candidates_with_no_indication_clause_compile_nothing(self):
        a = _cand("CAND_A", "assembly service performed")
        b = _cand("CAND_B", "assembly service performed differently")
        reqs = [r for r in req.compile_requirements([a, b]) if r.axis == "indication_clause"]
        self.assertEqual(reqs, [])

    def test_indication_clause_validates_supported_through_the_existing_plumbing(self):
        """Wired through the SAME `validated_requirement` every other axis
        uses -- proves this is not a parallel, unvetted mechanism."""
        reqs = [r for r in req.compile_requirements([self.WITH_INDICATION, self.PLAIN])
               if r.axis == "indication_clause"]
        r = reqs[0]
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("s1",))
        span_text = "documentation confirms the variant condition was addressed"
        self.assertTrue(req.validated_requirement(
            r, judgement, evidence_by_span_id={"s1": span_text},
            reconciliation=self._reconciliation({"s1": "AGREED"})))

    def test_indication_clause_not_documented_validates_against_full_coverage(self):
        reqs = [r for r in req.compile_requirements([self.WITH_INDICATION, self.PLAIN])
               if r.axis == "indication_clause"]
        r = reqs[0]
        coverage = self._coverage("assembly service performed today, nothing else stated")
        judgement = req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.NOT_DOCUMENTED)
        self.assertTrue(req.validated_requirement(r, judgement, coverage=coverage))

    def _judgements(self, requirement_ids, statuses, evidence_span_ids=("s1",)):
        rjs = tuple(req.RequirementJudgement(
            requirement_id=rid, status=statuses,
            evidence_span_ids=(evidence_span_ids if statuses == req.RequirementStatus.SUPPORTED
                               else ()))
            for rid in requirement_ids)
        model_a = type("J", (), {"requirement_judgements": rjs})()
        model_b = type("J", (), {"requirement_judgements": rjs})()
        return [model_a, model_b]

    def test_grounded_elimination_never_eliminates_on_an_undocumented_indication_clause(self):
        """P1-2 correction, end-to-end through the REAL
        `resolution._grounded_elimination` -- the actual function
        `_uniqueness_view` calls, not merely `validated_requirement` in
        isolation. Both models unanimously report NOT_DOCUMENTED for
        CAND_IND's own indication clause, over a fully-covered corpus that
        never states it -- unlike before this correction, an illustrative
        example alone must NEVER ground an elimination (it compiles as
        `POSITIVE_ALIAS`, elimination-ineligible), so this falls through to
        whatever else `_grounded_elimination` uses to decide, which finds
        no other governed, provider-answerable axis distinguishing these
        two candidates either."""
        from claude_coder import resolution
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind

        span = EvidenceSpan(text="assembly service performed, nothing else stated",
                            anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="assembly service",
                            evidence=[span], confidence=0.9, fact_id="F1")
        reqs = req.compile_requirements([self.WITH_INDICATION, self.PLAIN])
        ind_req = next(r for r in reqs if r.axis == "indication_clause")
        coverage = self._coverage("assembly service performed, nothing else stated")
        judgements = self._judgements((ind_req.requirement_id,),
                                      req.RequirementStatus.NOT_DOCUMENTED)

        grounded, detail = resolution._grounded_elimination(
            fact, self.WITH_INDICATION, self.PLAIN, reconciliation=None,
            requirements=reqs, judgements=judgements, coverage=coverage)
        self.assertFalse(grounded, detail)

    def test_grounded_elimination_never_eliminates_on_a_documented_indication_clause(self):
        """The mirror case: the SAME indication clause, but every evaluator
        reports SUPPORTED for it (a real cited span backs it) -- documented
        or not, this axis is elimination-ineligible (`POSITIVE_ALIAS`), so
        nothing grounds an elimination through it either way."""
        from claude_coder import resolution
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from app.contracts.source_evidence import (ReconciliationStatus,
                                                    SourceReconciliation,
                                                    SpanReconciliation)

        span = EvidenceSpan(text="documentation confirms the variant condition was addressed",
                            anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="assembly service",
                            evidence=[span], confidence=0.9, fact_id="F1")
        reqs = req.compile_requirements([self.WITH_INDICATION, self.PLAIN])
        ind_req = next(r for r in reqs if r.axis == "indication_clause")
        reconciliation = SourceReconciliation(spans=(
            SpanReconciliation(span_id="s1", status=ReconciliationStatus.AGREED),))
        judgements = self._judgements((ind_req.requirement_id,),
                                      req.RequirementStatus.SUPPORTED)

        grounded, _detail = resolution._grounded_elimination(
            fact, self.WITH_INDICATION, self.PLAIN, reconciliation, requirements=reqs,
            judgements=judgements, coverage=None)
        self.assertFalse(grounded)


class QualifiedChildRequirementTest(unittest.TestCase):
    """issue #6, Codex's independent re-review (F9-R19-A): CPT's own family-
    indentation convention -- candidates sharing an identical stem up to a
    semicolon each compile their OWN post-semicolon qualifying clause as a
    real, `selectable` MUST_SUPPORT requirement (structurally derived, never
    an arbitrary leftover-word difference like `descriptor_term`). Synthetic
    descriptors throughout."""

    SMALL = _cand("CAND_SMALL", "excision of lesion, subcutaneous; less than 3 cm")
    LARGE = _cand("CAND_LARGE", "excision of lesion, subcutaneous; 3 cm or greater")
    UNRELATED = _cand("CAND_UNRELATED", "assembly service, unrelated family")

    def _coverage(self, text):
        import hashlib
        return req.CoverageCorpus(channel_id="test-channel", text=text,
                                  text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                  covered_pages=(1,), page_image_sha256=("stub-hash",))

    def test_sibling_members_each_compile_their_own_qualifying_clause(self):
        reqs = [r for r in req.compile_requirements([self.SMALL, self.LARGE])
               if r.axis == "qualified_child"]
        self.assertEqual(len(reqs), 2)
        self.assertTrue(all(r.required for r in reqs))
        self.assertTrue(all(r.role is req.RequirementRole.MUST_SUPPORT for r in reqs))
        by_code = {r.candidate_code: r for r in reqs}
        self.assertEqual(by_code["CAND_SMALL"].expected, ("less than 3 cm",))
        self.assertEqual(by_code["CAND_LARGE"].expected, ("3 cm or greater",))

    def test_a_candidate_from_an_unrelated_family_compiles_nothing(self):
        reqs = [r for r in req.compile_requirements(
                    [self.SMALL, self.LARGE, self.UNRELATED])
               if r.axis == "qualified_child"]
        self.assertEqual([r for r in reqs if r.candidate_code == "CAND_UNRELATED"], [])

    def test_a_lone_semicolon_descriptor_with_no_sibling_compiles_nothing(self):
        lone = _cand("CAND_LONE", "excision of lesion, subcutaneous; unusual variant")
        reqs = [r for r in req.compile_requirements([lone, self.UNRELATED])
               if r.axis == "qualified_child"]
        self.assertEqual(reqs, [])

    def test_a_child_value_on_only_one_remaining_candidate_is_not_a_tie_axis(self):
        """One candidate's qualifier plus another candidate's silence is not a
        differential and must not generate a provider question."""
        from claude_coder import tiebreak
        qualified = _cand("CAND_QUALIFIED", "assembly service; residual variant")
        sibling = _cand("CAND_SIBLING", "assembly service; specific variant")
        silent = _cand("CAND_SILENT", "different service without a child clause")
        compiled = req.compile_requirements([qualified, sibling, silent])
        axes = tiebreak._axes_from_requirements(
            compiled, {"CAND_QUALIFIED", "CAND_SILENT"}, frozenset())
        self.assertFalse(any(a.axis == "qualified_child" for a in axes))

    def test_qualified_child_probe_requires_values_for_every_tied_candidate(self):
        """The direct descriptor-derived path obeys the same no-silence rule
        as the compiled-requirement path."""
        from claude_coder import tiebreak
        qualified = _cand("CAND_QUALIFIED", "assembly service; branch alpha")
        sibling = _cand("CAND_SIBLING", "assembly service; branch beta")
        silent = _cand("CAND_SILENT", "unrelated service")
        axes = tiebreak.discriminating_axes([qualified, sibling, silent])
        self.assertFalse(any(a.axis == "qualified_child" for a in axes))

    def test_one_sided_child_requirement_never_becomes_a_provider_question(self):
        from claude_coder import tiebreak
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from app.contracts.source_evidence import (
            ReconciliationStatus, SourceReconciliation, SpanReconciliation)
        qualified = _cand("CAND_QUALIFIED", "assembly service; residual variant")
        sibling = _cand("CAND_SIBLING", "assembly service; specific variant")
        silent = _cand("CAND_SILENT", "different service without a child clause")
        compiled = req.compile_requirements([qualified, sibling, silent])
        span = EvidenceSpan("service performed today", anchored=True, span_id="s1")
        fact = ClinicalFact(FactKind.PROCEDURE, "service performed", fact_id="F1",
                            evidence=[span], confidence=0.9)
        reconciliation = SourceReconciliation(spans=(SpanReconciliation(
            span_id="s1", status=ReconciliationStatus.AGREED),))
        outcome = tiebreak.narrow(
            fact, [qualified, silent], reconciliation, requirements=compiled)
        self.assertEqual(outcome.provider_question, "")
        self.assertNotIn("qualified_child", outcome.unsettled)

    def test_qualified_child_axis_is_selectable_like_laterality(self):
        """`tiebreak.discriminating_axes` marks this axis `selectable`,
        exactly like laterality -- so `narrow`'s EXISTING literal-presence
        winner logic (unchanged) may settle a tie through it, never a new
        selection mechanism. (Full page-proof narrowing is exercised at the
        `tiebreak`-level tests, not here.)"""
        from claude_coder import tiebreak
        axes = tiebreak.discriminating_axes([self.SMALL, self.LARGE])
        qualified = [a for a in axes if a.axis == "qualified_child"]
        self.assertEqual(len(qualified), 1)
        self.assertTrue(qualified[0].selectable)
        self.assertTrue(qualified[0].provable)

    def test_compound_exception_is_not_part_of_the_positive_child_axis(self):
        """A descriptor's exception has inverse polarity and therefore belongs
        only to the exclusion requirement.  Its coordinated alternatives are
        preserved as independent source-derived values; they are never copied
        into the positive family-child requirement."""
        broad = _cand(
            "CAND_BROAD",
            "assembly service; branch alpha or branch beta, except branch gamma or branch delta",
        )
        narrow = _cand(
            "CAND_NARROW",
            "assembly service; branch gamma or branch delta",
        )
        compiled = req.compile_requirements([broad, narrow])
        by_axis_code = {(r.axis, r.candidate_code): r for r in compiled}

        self.assertEqual(
            by_axis_code[("qualified_child", "CAND_BROAD")].expected,
            ("branch alpha", "branch beta"),
        )
        self.assertEqual(
            by_axis_code[("exclusion_clause", "CAND_BROAD")].expected,
            ("branch gamma", "branch delta"),
        )

    def test_threshold_and_optional_or_grammar_remain_indivisible(self):
        """The word 'or' is not always an alternative-value coordinator."""
        threshold_a = _cand("CAND_A", "assembly service; 3 units or greater")
        threshold_b = _cand("CAND_B", "assembly service; less than 3 units")
        optional_a = _cand("CAND_C", "component service; with or without attachment")
        optional_b = _cand("CAND_D", "component service; separate attachment")

        threshold = {r.candidate_code: r.expected for r in req.compile_requirements(
            [threshold_a, threshold_b]) if r.axis == "qualified_child"}
        optional = {r.candidate_code: r.expected for r in req.compile_requirements(
            [optional_a, optional_b]) if r.axis == "qualified_child"}

        self.assertEqual(threshold["CAND_A"], ("3 units or greater",))
        self.assertEqual(optional["CAND_C"], ("with or without attachment",))

    def test_real_alternative_still_splits_beside_protected_or_grammar(self):
        """Protected coordination is local to its own 'or', not a reason to
        flatten every alternative elsewhere in the same source clause."""
        mixed_a = _cand(
            "CAND_A", "assembly service; mode alpha or mode beta with or without attachment")
        mixed_b = _cand("CAND_B", "assembly service; mode gamma")
        comparison_a = _cand(
            "CAND_C", "component service; tier alpha or tier beta, 3 units or greater")
        comparison_b = _cand("CAND_D", "component service; tier gamma")

        mixed = {r.candidate_code: r.expected for r in req.compile_requirements(
            [mixed_a, mixed_b]) if r.axis == "qualified_child"}
        compared = {r.candidate_code: r.expected for r in req.compile_requirements(
            [comparison_a, comparison_b]) if r.axis == "qualified_child"}

        self.assertEqual(
            mixed["CAND_A"],
            ("mode alpha", "mode beta with or without attachment"),
        )
        self.assertEqual(
            compared["CAND_C"],
            ("tier alpha", "tier beta, 3 units or greater"),
        )


class ModelCitedDescriptorTermGroundingTest(unittest.TestCase):
    """issue #6, real-note investigation (designated note, F1: a wide,
    retrieval-broad candidate family sharing no compiled, typed axis with the
    winner -- e.g. a tumor-resection code beside a plain excision code, where
    nothing about "tumor" is a laterality/indication/exclusion/qualified-child
    clause). `_requirement_grounded_status` (typed axes only) and the untyped
    pairwise fallback (gated behind `selectable`, which the one axis carrying
    this vocabulary, `AXIS_DESCRIPTOR_TERM`, deliberately is not) both decline
    to ground these -- reproduced live: a real model correctly, citably
    eliminated six such candidates by name, and every one of those correct
    eliminations vanished before the final ClaimBundle. This is the bounded
    fix: ground ONLY when every judging model's own reason for eliminating
    the loser engages a real word from the loser's OWN official descriptor
    that the winner's descriptor does not share, excluding any word the fact's
    own evidence already establishes (a shared anatomy word is never grounds
    to rule out a SIBLING's own additional concept), and that word is
    independently confirmed absent from the complete document. Synthetic
    descriptors throughout."""

    WINNER = _cand("CAND_WIN", "assembly service, structure alpha")
    LOSER = _cand("CAND_LOSE", "assembly service, structure alpha, technique gamma")

    def _fact(self, evidence_text="Assembly service performed on structure alpha"):
        return ClinicalFact(
            kind=FactKind.PROCEDURE, description="assembly service",
            evidence=[EvidenceSpan(text=evidence_text, anchored=True, span_id="s1")],
            confidence=0.9, fact_id="F1")

    def _coverage(self, text):
        return req.CoverageCorpus(channel_id="test-channel", text=text,
                                  text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                  covered_pages=(1,), page_image_sha256=("stub-hash",))

    def _judgement(self, reason):
        return verify.Judgement(chosen=self.WINNER, entailed=(self.WINNER.code,),
                                eliminated={self.LOSER.code: reason}, declared=True)

    def test_grounds_when_the_evaluators_own_reason_cites_a_real_absent_descriptor_word(self):
        fact = self._fact()
        coverage = self._coverage("Assembly service performed on structure alpha.")
        judgement = self._judgement("technique gamma is not documented anywhere")
        grounded, detail = resolution._grounded_elimination(
            fact, self.LOSER, self.WINNER, reconciliation=None, requirements=(),
            judgements=[judgement], coverage=coverage)
        self.assertTrue(grounded, detail)
        self.assertIn("gamma", detail)

    def test_never_grounds_when_the_cited_word_is_actually_present(self):
        fact = self._fact()
        coverage = self._coverage(
            "Assembly service performed on structure alpha using technique gamma.")
        judgement = self._judgement("technique gamma is not documented anywhere")
        grounded, _detail = resolution._grounded_elimination(
            fact, self.LOSER, self.WINNER, reconciliation=None, requirements=(),
            judgements=[judgement], coverage=coverage)
        self.assertFalse(grounded)

    def test_never_grounds_on_a_reason_that_names_none_of_the_losers_own_words(self):
        """A free-floating reason that never engages the loser's OWN
        descriptor vocabulary must never ground anything -- the whole point
        is refusing exactly the untethered prose Codex's F8-R1 re-review
        found could convert a false elimination into a release."""
        fact = self._fact()
        coverage = self._coverage("Assembly service performed on structure alpha.")
        judgement = self._judgement("this is simply a different procedure")
        grounded = resolution._model_cited_descriptor_term_grounded(
            fact, self.LOSER, self.WINNER, [judgement], coverage)
        self.assertIsNone(grounded)

    def test_never_grounds_on_a_word_the_facts_own_evidence_already_establishes(self):
        """A word shared with the fact's own evidence (here, "structure
        alpha") can never ground an elimination through this path -- it is
        already true of the event itself, not evidence one sibling's own
        additional concept is absent. `technique`/`gamma` remain the only
        engageable words; without them cited, nothing grounds."""
        fact = self._fact()
        coverage = self._coverage("Assembly service performed on structure alpha.")
        judgement = self._judgement("structure alpha is documented but nothing else is")
        grounded = resolution._model_cited_descriptor_term_grounded(
            fact, self.LOSER, self.WINNER, [judgement], coverage)
        self.assertIsNone(grounded)

    def test_never_grounds_without_a_complete_coverage_corpus(self):
        fact = self._fact()
        judgement = self._judgement("technique gamma is not documented anywhere")
        grounded = resolution._model_cited_descriptor_term_grounded(
            fact, self.LOSER, self.WINNER, [judgement], None)
        self.assertIsNone(grounded)

    def test_tolerates_a_duck_typed_judgement_lacking_elimination_of(self):
        """`_requirement_grounded_status`'s own callers sometimes pass a bare
        test double exposing only `requirement_judgements` -- this path must
        decline gracefully (None), never raise."""
        fact = self._fact()
        coverage = self._coverage("Assembly service performed on structure alpha.")
        bare = type("J", (), {"requirement_judgements": ()})()
        grounded = resolution._model_cited_descriptor_term_grounded(
            fact, self.LOSER, self.WINNER, [bare], coverage)
        self.assertIsNone(grounded)

    def test_checks_the_whole_distinguishing_phrase_not_a_decomposed_word(self):
        """issue #6, real-note investigation (designated note, F4: a
        candidate descriptor naming "a Dwyer or Chambers type PROCEDURE" --
        the note separately uses the bare word "procedure" elsewhere, as a
        section header and a generic reference to the whole encounter,
        unrelated to any specific named technique). Decomposing the
        descriptor's own phrase into independent words and checking each
        one's absence separately would find "procedure" trivially present
        (in that unrelated sense) and wrongly refuse a genuine, correct
        elimination. Checking the WHOLE contiguous phrase ("structure gamma
        type technique") as one unit -- never decomposed -- is the fix: that
        exact phrase recurs nowhere, even though one of its own words does,
        alone, elsewhere."""
        winner = _cand("CAND_WIN", "assembly service, structure alpha")
        loser = _cand("CAND_LOSE",
                      "assembly service, structure alpha, structure gamma type technique")
        fact = ClinicalFact(
            kind=FactKind.PROCEDURE, description="assembly service",
            evidence=[EvidenceSpan(text="Assembly service performed on structure alpha",
                                   anchored=True, span_id="s1")],
            confidence=0.9, fact_id="F1")
        coverage = self._coverage(
            "TECHNIQUE\n\n"
            "Assembly service performed on structure alpha using a standard technique. "
            "No complications occurred during the technique."
        )
        judgement = verify.Judgement(
            chosen=winner, entailed=(winner.code,),
            eliminated={loser.code: ("this was not a structure gamma type technique, "
                                    "which requires a distinct technique")},
            declared=True)
        grounded, detail = resolution._grounded_elimination(
            fact, loser, winner, reconciliation=None, requirements=(),
            judgements=[judgement], coverage=coverage)
        self.assertTrue(grounded, detail)
        self.assertIn("gamma type technique", detail)

    def test_never_engages_laterality_even_when_a_reason_names_a_side(self):
        """A real prior defect in this codebase showed lexical laterality
        matching can silently resolve the WRONG side once a negation falls
        outside a naive window -- laterality is settled EXCLUSIVELY by the
        fact's own typed attribute everywhere else in this module
        (`tiebreak._typed_laterality_support`), and this path must never
        quietly reopen that exact risk just because an evaluator's free-form
        reason happens to mention a side. "left"/"right"/"bilateral" must
        never enter a checked phrase here at all -- refuses (None), never
        grounds either way, regardless of what the document says elsewhere."""
        winner = _cand("CAND_RIGHT", "assembly service, structure alpha, right side")
        loser = _cand("CAND_LEFT", "assembly service, structure alpha, left side")
        fact = ClinicalFact(
            kind=FactKind.PROCEDURE, description="assembly service",
            attributes={"laterality": "right"},
            evidence=[EvidenceSpan(text="Assembly service performed on structure alpha",
                                   anchored=True, span_id="s1")],
            confidence=0.9, fact_id="F1")
        # A DIFFERENT, unrelated mention of "left" elsewhere in a long
        # document -- exactly the construction that broke naive lexical
        # laterality matching before ("right ... ultimately ruled out"
        # wrongly counting "right" as stated).
        coverage = self._coverage(
            "Patient presents for evaluation. Left knee was examined and "
            "found unremarkable. Assembly service performed on structure alpha.")
        judgement = verify.Judgement(
            chosen=winner, entailed=(winner.code,),
            eliminated={loser.code: "the documentation does not support the left side"},
            declared=True)
        grounded = resolution._model_cited_descriptor_term_grounded(
            fact, loser, winner, [judgement], coverage)
        self.assertIsNone(grounded)


class ExclusionClauseDeterministicGroundingTest(unittest.TestCase):
    """issue #6, real-note investigation: an evaluator can reasonably label an
    exclusion clause's inverted polarity CONTRADICTED instead of SUPPORTED --
    `validated_requirement` correctly never trusts either label for what a
    model merely claims (Codex's own documented policy), which left an
    exclusion clause with NO way to ground at all when a model chose the
    "wrong" label for a genuinely correct conclusion. This is the fix: the
    SAME judgement-independent, closed, literal, negation-aware check
    `AXIS_LATERALITY` already uses (never trusting any evaluator's status
    label) applied to an exclusion clause's own excluded term -- first
    lexically against the fact's own evidence, then (when a `source` capable
    of `concept_relation` is supplied) through the SAME governed SNOMED
    Body Structure concept index `semantic_eligibility._anatomy_compatibility`
    already trusts, for EXACT identity only -- never the weaker ancestor/
    descendant tier, which the terminology module's own docstring calls
    "not a confirmed verdict either way". Synthetic descriptors throughout."""

    WINNER = _cand("CAND_WIN", "assembly service")
    LOSER = _cand("CAND_EXC", "assembly service, except structure gamma")

    def _fact(self, evidence_text, anatomy=""):
        attrs = {"anatomy": anatomy} if anatomy else {}
        return ClinicalFact(
            kind=FactKind.PROCEDURE, description="assembly service", attributes=attrs,
            evidence=[EvidenceSpan(text=evidence_text, anchored=True, span_id="s1")],
            confidence=0.9, fact_id="F1")

    def _requirements(self):
        return req.compile_requirements([self.WINNER, self.LOSER])

    def test_grounds_deterministically_on_a_literal_mention_with_no_judgements_at_all(self):
        fact = self._fact("Assembly service performed on structure gamma.")
        grounded, detail = resolution._requirement_grounded_status(
            fact, self.LOSER, self._requirements(), [], None, None)
        self.assertTrue(grounded, detail)

    def test_never_grounds_when_the_excluded_term_is_genuinely_absent(self):
        fact = self._fact("Assembly service performed on structure alpha.")
        grounded = resolution._requirement_grounded_status(
            fact, self.LOSER, self._requirements(), [], None, None)
        self.assertIsNone(grounded)

    def test_grounds_via_a_governed_exact_concept_synonym_when_no_literal_match_exists(self):
        """The fact documents "structure zeta" in lay/alternate phrasing; a
        governed source confirms it is the SAME concept as the excluded
        "structure gamma" -- bridging exactly the vocabulary gap a pure
        lexical check cannot."""
        fact = self._fact("Assembly service performed on structure zeta.",
                          anatomy="structure zeta")
        source = MockSource(concept_relation={("structure zeta", "structure gamma"): "same"})
        grounded, detail = resolution._requirement_grounded_status(
            fact, self.LOSER, self._requirements(), [], None, None, source=source)
        self.assertTrue(grounded, detail)

    def test_never_grounds_on_a_merely_related_not_confirmed_concept_tier(self):
        """`CONCEPT_RELATED` ("ancestor_descendant") is documented as "not a
        confirmed verdict either way" -- must never, by itself, ground a hard
        elimination, only the exact `CONCEPT_SAME` tier may."""
        fact = self._fact("Assembly service performed on structure zeta.",
                          anatomy="structure zeta")
        source = MockSource(concept_relation={
            ("structure zeta", "structure gamma"): "ancestor_descendant"})
        grounded = resolution._requirement_grounded_status(
            fact, self.LOSER, self._requirements(), [], None, None, source=source)
        self.assertIsNone(grounded)

    def test_never_grounds_without_a_source_when_no_literal_match_exists(self):
        fact = self._fact("Assembly service performed on structure zeta.",
                          anatomy="structure zeta")
        grounded = resolution._requirement_grounded_status(
            fact, self.LOSER, self._requirements(), [], None, None, source=None)
        self.assertIsNone(grounded)


class DefinitionalClauseRequirementTest(unittest.TestCase):
    """issue #6, real-note investigation (designated note, F1: a candidate
    descriptor naming the several specific alternative forms a general act
    word covers, told apart from `AXIS_INDICATION_CLAUSE`'s illustrative
    "(eg, ...)" example purely by grammar -- the ABSENCE of an "eg"/"for
    example" marker is itself the signal that this parenthetical DEFINES
    what the adjacent term means (CPT's own drafting convention for an
    exhaustive enumeration), never an open-ended example. Same polarity as
    `AXIS_QUALIFIED_CHILD`'s own required precondition: documenting ONE
    alternative supports the candidate that carries it; documenting NONE of
    them, in a fully-searched record, may ground its elimination.
    Synthetic descriptors throughout."""

    WITH_DEFINITION = _cand(
        "CAND_DEF", "assembly service (form alpha, form beta, or form gamma)")
    PLAIN = _cand("CAND_PLAIN", "assembly service performed")

    def _coverage(self, text):
        return req.CoverageCorpus(channel_id="test-channel", text=text,
                                  text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                  covered_pages=(1,), page_image_sha256=("stub-hash",))

    def test_told_apart_from_an_illustrative_clause_by_grammar_alone(self):
        """The SAME parenthetical shape as an "(eg, ...)" clause, minus the
        marker, must compile as `definitional_clause`, never
        `indication_clause` -- and vice versa."""
        illustrative = _cand("CAND_ILL", "assembly service (eg, form alpha)")
        reqs_def = [r for r in req.compile_requirements([self.WITH_DEFINITION, self.PLAIN])
                   if r.axis in ("definitional_clause", "indication_clause")]
        reqs_ill = [r for r in req.compile_requirements([illustrative, self.PLAIN])
                   if r.axis in ("definitional_clause", "indication_clause")]
        self.assertEqual([r.axis for r in reqs_def], ["definitional_clause"])
        self.assertEqual([r.axis for r in reqs_ill], ["indication_clause"])

    def test_splits_every_comma_and_or_joined_alternative_separately(self):
        """"A, B, or C" must compile as THREE separate alternatives, never one
        unsplit, near-unmatchable joined phrase."""
        reqs = [r for r in req.compile_requirements([self.WITH_DEFINITION, self.PLAIN])
               if r.axis == "definitional_clause"]
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0].expected, ("form alpha", "form beta", "form gamma"))

    def test_compiles_as_a_required_must_support_requirement(self):
        reqs = [r for r in req.compile_requirements([self.WITH_DEFINITION, self.PLAIN])
               if r.axis == "definitional_clause"]
        r = reqs[0]
        self.assertEqual(r.candidate_code, "CAND_DEF")
        self.assertTrue(r.required)
        self.assertEqual(r.role, req.RequirementRole.MUST_SUPPORT)

    def test_a_single_unenumerated_parenthetical_compiles_nothing(self):
        """A bare remark or a protected "with or without" phrase is not a
        genuine multi-alternative enumeration -- left to the ordinary
        bag-of-words axis, never promoted here."""
        single = _cand("CAND_SINGLE", "assembly service (with or without attachment)")
        reqs = [r for r in req.compile_requirements([single, self.PLAIN])
               if r.axis == "definitional_clause"]
        self.assertEqual(reqs, [])

    def test_an_except_marked_parenthetical_is_left_to_exclusion_clause(self):
        excluded = _cand("CAND_EXC", "assembly service (except form alpha or form beta)")
        reqs = [r for r in req.compile_requirements([excluded, self.PLAIN])
               if r.axis in ("definitional_clause", "exclusion_clause")]
        self.assertEqual([r.axis for r in reqs], ["exclusion_clause"])

    def test_grounded_elimination_eliminates_on_no_documented_alternative(self):
        """End-to-end through the REAL `resolution._grounded_elimination` --
        every alternative genuinely NOT_DOCUMENTED in a fully-covered record
        grounds an elimination, the SAME polarity `AXIS_QUALIFIED_CHILD`
        already trusts."""
        from claude_coder import resolution
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind

        span = EvidenceSpan(text="assembly service performed, nothing else stated",
                            anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="assembly service",
                            evidence=[span], confidence=0.9, fact_id="F1")
        reqs = req.compile_requirements([self.WITH_DEFINITION, self.PLAIN])
        coverage = self._coverage("assembly service performed, nothing else stated")
        rjs = tuple(req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.NOT_DOCUMENTED)
            for r in reqs if r.axis == "definitional_clause")
        judgements = [type("J", (), {"requirement_judgements": rjs})()]

        grounded, detail = resolution._grounded_elimination(
            fact, self.WITH_DEFINITION, self.PLAIN, reconciliation=None,
            requirements=reqs, judgements=judgements, coverage=coverage)
        self.assertTrue(grounded, detail)

    def test_grounded_elimination_never_eliminates_on_one_documented_alternative(self):
        """Documenting ONE alternative satisfies the whole requirement --
        the mirror case, never eliminated."""
        from claude_coder import resolution
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind

        span = EvidenceSpan(text="assembly service performed via form beta",
                            anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="assembly service",
                            evidence=[span], confidence=0.9, fact_id="F1")
        reqs = req.compile_requirements([self.WITH_DEFINITION, self.PLAIN])
        coverage = self._coverage("assembly service performed via form beta")
        rjs = tuple(req.RequirementJudgement(
            requirement_id=r.requirement_id, status=req.RequirementStatus.SUPPORTED,
            evidence_span_ids=("s1",))
            for r in reqs if r.axis == "definitional_clause")
        judgements = [type("J", (), {"requirement_judgements": rjs})()]

        grounded, _detail = resolution._grounded_elimination(
            fact, self.WITH_DEFINITION, self.PLAIN, reconciliation=None,
            requirements=reqs, judgements=judgements, coverage=coverage)
        self.assertFalse(grounded)


class SequenceQualifierRequirementTest(unittest.TestCase):
    """issue #6, real-note investigation (designated note, F4: a candidate
    descriptor stating a real, cross-specialty CPT drafting convention --
    "Repair, primary, ..." vs "Repair, secondary, ..." -- released as
    VERIFIED with nothing independently confirming which one applied).
    "primary"/"secondary"/"initial"/"subsequent" are generic English
    sequence/order words, told apart purely by POSITION (immediately after
    the descriptor's own first comma) -- confirmed against the real,
    complete CPT data set: 32 codes across orthopedics, ENT, plastic surgery,
    and pathology, never scenario-specific. Synthetic descriptors
    throughout."""

    PRIMARY = _cand("CAND_PRIMARY", "assembly, primary, structure alpha")
    SECONDARY = _cand("CAND_SECONDARY", "assembly, secondary, structure alpha")
    PLAIN = _cand("CAND_PLAIN", "assembly service, structure alpha")

    def test_compiles_only_when_the_qualifier_is_the_first_comma_delimited_word(self):
        reqs = [r for r in req.compile_requirements([self.PRIMARY, self.PLAIN])
               if r.axis == "sequence_qualifier"]
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0].candidate_code, "CAND_PRIMARY")
        self.assertEqual(reqs[0].expected, ("primary",))
        self.assertTrue(reqs[0].required)
        self.assertEqual(reqs[0].role, req.RequirementRole.MUST_SUPPORT)

    def test_a_plain_descriptor_with_no_qualifier_compiles_nothing(self):
        reqs = [r for r in req.compile_requirements([self.PRIMARY, self.PLAIN])
               if r.axis == "sequence_qualifier" and r.candidate_code == "CAND_PLAIN"]
        self.assertEqual(reqs, [])

    def test_a_word_outside_the_closed_set_never_compiles(self):
        """Purely structural + a closed, generic vocabulary -- an arbitrary
        word in the same position must never be mistaken for one of these
        four sequence markers."""
        other = _cand("CAND_OTHER", "assembly, revised, structure alpha")
        reqs = [r for r in req.compile_requirements([other, self.PLAIN])
               if r.axis == "sequence_qualifier"]
        self.assertEqual(reqs, [])

    def test_primary_and_secondary_compile_as_distinct_required_alternatives(self):
        reqs = {r.candidate_code: r.expected for r in
               req.compile_requirements([self.PRIMARY, self.SECONDARY])
               if r.axis == "sequence_qualifier"}
        self.assertEqual(reqs["CAND_PRIMARY"], ("primary",))
        self.assertEqual(reqs["CAND_SECONDARY"], ("secondary",))


class ChosenCandidateOwnRequirementsConfirmedTest(unittest.TestCase):
    """issue #6, real-note investigation (designated note, F4: "Repair,
    secondary, Achilles tendon" released as VERIFIED with nothing
    independently confirming "secondary" at all): `_uniqueness_view`/
    `_settle_uniqueness` only ever spend a compiled requirement ELIMINATING a
    rival -- the sole surviving, CHOSEN candidate's own required axes were
    never symmetrically checked. `_chosen_own_requirements_confirmed` closes
    that gap; `_entailed_line` now calls it before minting VERIFIED whenever
    `requirements` is supplied. Synthetic descriptors throughout."""

    WINNER = _cand("CAND_WIN", "assembly, primary, structure alpha")
    LOSER = _cand("CAND_LOSE", "assembly, secondary, structure alpha")

    def _fact(self, text):
        return ClinicalFact(
            kind=FactKind.PROCEDURE, description="assembly service",
            evidence=[EvidenceSpan(text=text, anchored=True, span_id="s1")],
            confidence=0.9, fact_id="F1")

    def _coverage(self, text):
        return req.CoverageCorpus(channel_id="test-channel", text=text,
                                  text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                  covered_pages=(1,), page_image_sha256=("stub-hash",))

    def test_never_releases_verified_when_the_winners_own_premise_is_unconfirmed(self):
        fact = self._fact("Assembly service performed on structure alpha.")
        candidates = [self.WINNER, self.LOSER]
        requirements = req.compile_requirements(candidates)
        judgement = verify.Judgement(
            chosen=self.WINNER, entailed=(self.WINNER.code,),
            eliminated={self.LOSER.code: "no secondary indication is documented"},
            declared=True)
        line = resolution._entailed_line(
            fact, self.WINNER, candidates, "single-evaluator entailment",
            requirements=requirements, judgements=[judgement], coverage=None)
        self.assertIsNone(line.chosen)
        self.assertEqual(line.method, resolution.ResolutionMethod.ABSTAINED)
        self.assertIn("sequence_qualifier", line.documentation_gap)

    def test_releases_verified_when_the_winners_own_premise_is_validated(self):
        fact = self._fact("This is the primary repair on structure alpha.")
        candidates = [self.WINNER, self.LOSER]
        requirements = req.compile_requirements(candidates)
        win_req = next(r for r in requirements
                       if r.axis == "sequence_qualifier" and r.candidate_code == "CAND_WIN")
        judgement = verify.Judgement(
            chosen=self.WINNER, entailed=(self.WINNER.code,),
            eliminated={self.LOSER.code: "no secondary indication is documented"},
            declared=True,
            requirement_judgements=(
                req.RequirementJudgement(requirement_id=win_req.requirement_id,
                                         status=req.RequirementStatus.SUPPORTED,
                                         evidence_span_ids=("s1",)),))
        reconciliation = SourceReconciliation(spans=(
            SpanReconciliation(span_id="s1", status=ReconciliationStatus.AGREED),))
        coverage = self._coverage("This is the primary repair on structure alpha.")
        line = resolution._entailed_line(
            fact, self.WINNER, candidates, "single-evaluator entailment",
            requirements=requirements, judgements=[judgement],
            reconciliation=reconciliation, coverage=coverage)
        self.assertEqual(line.chosen.code if line.chosen else None, "CAND_WIN")
        self.assertEqual(line.method, resolution.ResolutionMethod.VERIFIED)

    def test_skips_the_check_entirely_when_no_requirements_are_supplied(self):
        """A caller that omits `requirements` (the default) gets exactly
        prior behavior -- this new check is additive, never mandatory for
        callers that haven't been updated to supply the extra context."""
        fact = self._fact("Assembly service performed on structure alpha.")
        line = resolution._entailed_line(
            fact, self.WINNER, [self.WINNER], "single-evaluator entailment")
        self.assertEqual(line.chosen.code, "CAND_WIN")
        self.assertEqual(line.method, resolution.ResolutionMethod.VERIFIED)

    def test_never_demands_confirmation_for_laterality(self):
        """AXIS_LATERALITY is settled EXCLUSIVELY by the fact's own typed
        attribute -- this check must never demand a redundant
        `requirement_judgements` entry for it."""
        winner = _cand("CAND_R", "assembly service, right structure")
        loser = _cand("CAND_L", "assembly service, left structure")
        fact = ClinicalFact(
            kind=FactKind.PROCEDURE, description="assembly service",
            attributes={"laterality": "right"},
            evidence=[EvidenceSpan(text="assembly service performed on the right structure",
                                   anchored=True, span_id="s1")],
            confidence=0.9, fact_id="F1")
        candidates = [winner, loser]
        requirements = req.compile_requirements(candidates)
        judgement = verify.Judgement(
            chosen=winner, entailed=(winner.code,),
            eliminated={loser.code: "documented as the right side, not the left"},
            declared=True)
        line = resolution._entailed_line(
            fact, winner, candidates, "single-evaluator entailment",
            requirements=requirements, judgements=[judgement], coverage=None)
        self.assertEqual(line.chosen.code if line.chosen else None, "CAND_R")
        self.assertEqual(line.method, resolution.ResolutionMethod.VERIFIED)


def _admissions(candidates, supported):
    """Synthetic `candidate_admission` records: SUPPORTED governed identity
    standing for `supported` codes, recall-only (UNGROUNDED) for the rest."""
    return {
        c.code: resolution.CandidateAdmission(
            (c.code, c.system),
            (resolution.CandidateStanding.SUPPORTED if c.code in supported
             else resolution.CandidateStanding.UNGROUNDED),
            (("governed_term_mapping",) if c.code in supported else ()), (), (), (),
            {"code": c.code, "descriptor": c.descriptor}, (c.source,))
        for c in candidates}


class BaselineDescriptorGroundingTest(unittest.TestCase):
    """issue #6, real-note investigation (designated note F11/F12): a candidate
    surviving into a tie purely because no evaluator named a reason to rule
    it out is not the same as being entailed. `_baseline_descriptor_grounded`
    (used by `_settle_uniqueness`) checks a candidate's own distinguishing
    vocabulary (`AXIS_DESCRIPTOR_TERM`) against the fact's own evidence
    before letting it count as "still entailed" -- but only for a candidate
    that has not already cleared a curated, authoritative term-to-code
    match (`authority["term_to_code_match"]`), never by its `source` label
    alone."""

    AUTHORITATIVE = CandidateCode(
        code="CAND_AUTH", system="icd10",
        descriptor="Other specified condition alpha, structure gamma",
        score=1.0, source="crosswalk-match",
        authority={"source": "curated crosswalk",
                   "term_to_code_match": {"method": "contained_source_phrase"}})
    RETRIEVAL_ONE = CandidateCode(
        code="CAND_R1", system="icd10",
        descriptor="Other calcification of structure beta, structure gamma",
        score=0.05, source="retrieval")
    RETRIEVAL_TWO = CandidateCode(
        code="CAND_R2", system="icd10",
        descriptor="Other specific arthropathy of structure delta, structure gamma",
        score=0.04, source="retrieval")

    def _fact(self, text):
        return ClinicalFact(
            kind=FactKind.DIAGNOSIS, description="condition alpha",
            evidence=[EvidenceSpan(text=text, anchored=True, span_id="s1")],
            confidence=0.9, fact_id="F1")

    def test_eliminates_retrieval_candidates_whose_own_term_is_absent(self):
        fact = self._fact("Painful condition alpha of structure gamma.")
        shortlist = [self.AUTHORITATIVE, self.RETRIEVAL_ONE, self.RETRIEVAL_TWO]
        reconciliation = SourceReconciliation(spans=(
            SpanReconciliation(span_id="s1", status=ReconciliationStatus.AGREED),))
        judgement = verify.Judgement(chosen=self.AUTHORITATIVE,
                                     entailed=(self.AUTHORITATIVE.code,),
                                     eliminated={}, declared=True)
        line = resolution._settle_uniqueness(
            fact, self.AUTHORITATIVE, shortlist, [judgement], {}, "",
            reconciliation, requirements=(), coverage=None,
            admissions=_admissions(shortlist, supported={"CAND_AUTH"}))
        self.assertEqual(line.chosen.code if line.chosen else None, "CAND_AUTH")
        self.assertEqual(line.method, resolution.ResolutionMethod.VERIFIED)

    def test_a_page_word_miss_never_settles_a_tie_without_governed_standing(self):
        """Codex F8-R1 / F9-R19-A: with NO candidate holding governed identity
        standing, an ungrounded rival stays standing exactly as before this
        floor existed -- a raw descriptor-word miss is not typed evidence."""
        fact = self._fact("Painful condition alpha of structure gamma.")
        shortlist = [self.AUTHORITATIVE, self.RETRIEVAL_ONE, self.RETRIEVAL_TWO]
        reconciliation = SourceReconciliation(spans=(
            SpanReconciliation(span_id="s1", status=ReconciliationStatus.AGREED),))
        judgement = verify.Judgement(chosen=self.AUTHORITATIVE,
                                     entailed=(self.AUTHORITATIVE.code,),
                                     eliminated={}, declared=True)
        for admissions in (None, _admissions(shortlist, supported=set())):
            line = resolution._settle_uniqueness(
                fact, self.AUTHORITATIVE, shortlist, [judgement], {}, "",
                reconciliation, requirements=(), coverage=None, admissions=admissions)
            self.assertIsNone(line.chosen, line.rationale)
            self.assertEqual(set(line.tie_record["still_entailed"]),
                             {"CAND_AUTH", "CAND_R1", "CAND_R2"})

    def test_never_eliminates_the_authoritative_term_matched_candidate(self):
        """The authoritative candidate's own distinguishing word ("alpha") is
        genuinely absent from this terse fact's evidence too -- it must stay
        anyway, because `term_to_code_match` exempts it from this floor."""
        fact = self._fact("Painful prominence of structure gamma.")
        shortlist = [self.AUTHORITATIVE, self.RETRIEVAL_ONE, self.RETRIEVAL_TWO]
        reconciliation = SourceReconciliation(spans=(
            SpanReconciliation(span_id="s1", status=ReconciliationStatus.AGREED),))
        judgement = verify.Judgement(chosen=self.AUTHORITATIVE,
                                     entailed=(self.AUTHORITATIVE.code,),
                                     eliminated={}, declared=True)
        line = resolution._settle_uniqueness(
            fact, self.AUTHORITATIVE, shortlist, [judgement], {}, "",
            reconciliation, requirements=(), coverage=None,
            admissions=_admissions(shortlist, supported={"CAND_AUTH"}))
        self.assertEqual(line.chosen.code if line.chosen else None, "CAND_AUTH")

    def test_never_eliminates_every_remaining_candidate_at_once(self):
        """All three candidates here are `source="retrieval"` and none of
        their distinguishing words appear in evidence -- this must NOT
        collapse `remaining` to zero and manufacture a false "nothing
        fits" state; it must leave the tie exactly as it found it."""
        r3 = CandidateCode(code="CAND_R3", system="icd10",
                           descriptor="Unrelated condition epsilon, structure gamma",
                           score=0.03, source="retrieval")
        shortlist = [self.RETRIEVAL_ONE, self.RETRIEVAL_TWO, r3]
        fact = self._fact("Painful prominence of structure gamma.")
        reconciliation = SourceReconciliation(spans=(
            SpanReconciliation(span_id="s1", status=ReconciliationStatus.AGREED),))
        judgement = verify.Judgement(chosen=None, entailed=(), eliminated={}, declared=True)
        line = resolution._settle_uniqueness(
            fact, None, shortlist, [judgement], {}, "",
            reconciliation, requirements=(), coverage=None,
            admissions=_admissions(shortlist, supported={"CAND_R1"}))
        self.assertIsNone(line.chosen)
        self.assertIn("CAND_R1", line.rationale)
        self.assertIn("CAND_R2", line.rationale)
        self.assertIn("CAND_R3", line.rationale)

    def test_keeps_a_retrieval_candidate_whose_own_term_is_present(self):
        """A `source="retrieval"` candidate whose distinguishing word DOES
        appear in the fact's own evidence is never eliminated by this
        floor -- only genuine, confirmed absence counts (and only against a
        rival with governed standing)."""
        fact = self._fact("Structure beta calcification was noted near structure gamma.")
        shortlist = [self.RETRIEVAL_ONE, self.RETRIEVAL_TWO]
        reconciliation = SourceReconciliation(spans=(
            SpanReconciliation(span_id="s1", status=ReconciliationStatus.AGREED),))
        judgement = verify.Judgement(chosen=None, entailed=(), eliminated={}, declared=True)
        line = resolution._settle_uniqueness(
            fact, None, shortlist, [judgement], {}, "",
            reconciliation, requirements=(), coverage=None,
            admissions=_admissions(shortlist, supported={"CAND_R1"}))
        self.assertIn("CAND_R1", line.rationale)
        self.assertNotIn("CAND_R2", line.rationale)

    def test_a_non_retrieval_source_without_term_match_is_not_exempt(self):
        """issue #6, F12 re-review: a mechanically-derived sibling-expansion
        candidate (e.g. `icd10-specificity-family`) carries no
        `term_to_code_match` -- exemption must track that field, never the
        `source` label alone, or a genuinely ungrounded sibling escapes this
        floor merely because it wasn't literally sourced from "retrieval"."""
        sibling = CandidateCode(
            code="CAND_SIB", system="icd10",
            descriptor="Other specified condition alpha, structure gamma",
            score=0.17, source="icd10-specificity-family",
            authority={"source": "ICD-10-CM authoritative specificity family",
                      "base_candidate": "CAND_R1"})
        fact = self._fact("Structure beta calcification was noted near structure gamma.")
        shortlist = [self.RETRIEVAL_ONE, sibling]
        reconciliation = SourceReconciliation(spans=(
            SpanReconciliation(span_id="s1", status=ReconciliationStatus.AGREED),))
        judgement = verify.Judgement(chosen=None, entailed=(), eliminated={}, declared=True)
        line = resolution._settle_uniqueness(
            fact, None, shortlist, [judgement], {}, "",
            reconciliation, requirements=(), coverage=None,
            admissions=_admissions(shortlist, supported={"CAND_R1"}))
        self.assertIn("CAND_R1", line.rationale)
        self.assertNotIn("CAND_SIB", line.rationale)


class UngroundedTieNeverAsksTheProviderTest(unittest.TestCase):
    """issue #6, real-note investigation (designated note F2/F3/F5/F7): a tie in
    which NO candidate's own distinguishing vocabulary appears in the fact's
    evidence must not manufacture a provider question out of those candidates'
    differences -- the question would be about the wrong codes entirely."""

    RIGHT_POWERED = CandidateCode(
        code="CAND_RP", system="cpt",
        descriptor="Assembly of structure alpha, right, powered technique",
        score=0.5, source="retrieval")
    LEFT_MANUAL = CandidateCode(
        code="CAND_LM", system="cpt",
        descriptor="Assembly of structure beta, left, manual technique",
        score=0.4, source="retrieval")

    def _settle(self, text):
        fact = ClinicalFact(
            kind=FactKind.PROCEDURE, description="widget service",
            evidence=[EvidenceSpan(text=text, anchored=True, span_id="s1")],
            confidence=0.9, fact_id="F1")
        reconciliation = SourceReconciliation(spans=(
            SpanReconciliation(span_id="s1", status=ReconciliationStatus.AGREED),))
        judgement = verify.Judgement(chosen=None, entailed=(), eliminated={}, declared=True)
        return resolution._settle_uniqueness(
            fact, None, [self.RIGHT_POWERED, self.LEFT_MANUAL], [judgement], {}, "",
            reconciliation, requirements=(), coverage=None)

    def test_an_all_ungrounded_tie_withholds_the_provider_question(self):
        line = self._settle("A widget service was performed.")
        self.assertIsNone(line.chosen)
        self.assertIsNone(line.documentation_gap)
        self.assertIn("no provider question", line.rationale)
        self.assertEqual(set(line.tie_record["baseline_ungrounded"]), {"CAND_RP", "CAND_LM"})

    def test_a_tie_with_a_grounded_candidate_still_asks(self):
        """The record states one candidate's own word ("powered"), so the pool is
        not all-ungrounded: the question is NOT withheld, and (no candidate
        holding governed standing) nothing is eliminated either -- Codex F8-R1."""
        line = self._settle("A powered widget service was performed.")
        self.assertIsNone(line.chosen)
        self.assertNotIn("baseline_ungrounded", line.tie_record)
        self.assertIsNotNone(line.documentation_gap)
        self.assertEqual(set(line.tie_record["still_entailed"]), {"CAND_RP", "CAND_LM"})

    def test_ungrounded_pool_is_all_or_nothing(self):
        fact = ClinicalFact(
            kind=FactKind.PROCEDURE, description="widget service",
            evidence=[EvidenceSpan(text="A powered widget service was performed.",
                                   anchored=True, span_id="s1")],
            confidence=0.9, fact_id="F1")
        self.assertEqual(resolution._baseline_ungrounded_pool(
            fact, [self.RIGHT_POWERED, self.LEFT_MANUAL], None), {})
        fact2 = ClinicalFact(
            kind=FactKind.PROCEDURE, description="widget service",
            evidence=[EvidenceSpan(text="A widget service was performed.",
                                   anchored=True, span_id="s1")],
            confidence=0.9, fact_id="F1")
        pool = resolution._baseline_ungrounded_pool(
            fact2, [self.RIGHT_POWERED, self.LEFT_MANUAL], None)
        self.assertEqual(set(pool), {"CAND_RP", "CAND_LM"})

    def test_a_lone_candidate_is_judged_on_its_whole_vocabulary(self):
        fact = ClinicalFact(
            kind=FactKind.IMAGING, description="intra-operative check",
            evidence=[EvidenceSpan(text="Imaging confirmed the contour of structure gamma.",
                                   anchored=True, span_id="s1")],
            confidence=0.9, fact_id="F1")
        lone = CandidateCode(code="CAND_X", system="cpt",
                             descriptor="Radiologic examination; site delta, 2 views",
                             score=0.3, source="retrieval")
        self.assertEqual(set(resolution._baseline_ungrounded_pool(fact, [lone], None)),
                         {"CAND_X"})
        # Fail-safe: a generic word that forms a whole clause on its own
        # ("structure") still grounds a lone candidate -- the line stays open...
        shared = CandidateCode(code="CAND_Z", system="cpt",
                               descriptor="Radiologic examination; structure, 2 views",
                               score=0.3, source="retrieval")
        self.assertEqual(resolution._baseline_ungrounded_pool(fact, [shared], None), {})
        # ...but a longer run that merely CONTAINS that word ("structure delta")
        # is checked as the whole phrase, and is absent.
        contained = CandidateCode(code="CAND_W", system="cpt",
                                  descriptor="Radiologic examination; structure delta, 2 views",
                                  score=0.3, source="retrieval")
        self.assertEqual(set(resolution._baseline_ungrounded_pool(fact, [contained], None)),
                         {"CAND_W"})
        stated = CandidateCode(code="CAND_Y", system="cpt",
                               descriptor="Imaging of structure gamma", score=0.3,
                               source="retrieval")
        self.assertEqual(resolution._baseline_ungrounded_pool(fact, [stated], None), {})


class GovernedFamilyBaselineGroundingTest(unittest.TestCase):
    """issue #6, real-note investigation (designated note F3): two siblings that
    differ only by a qualified-child qualifier have no residual descriptor_term
    vocabulary at all -- the family stem and both qualifiers are governed words.
    Whether that STEM is documented is what decides the family's viability, so
    the baseline floor falls back to the candidate's own vocabulary minus grammar,
    laterality, and the words every pool member shares."""

    LOCAL = CandidateCode(code="CAND_LOC", system="cpt", score=0.4, source="retrieval",
                          descriptor="Division, percutaneous, structure alpha (separate procedure); local technique")
    GENERAL = CandidateCode(code="CAND_GEN", system="cpt", score=0.3, source="retrieval",
                            descriptor="Division, percutaneous, structure alpha (separate procedure); general technique")
    REPAIR = CandidateCode(code="CAND_REP", system="cpt", score=0.3, source="retrieval",
                           descriptor="Repair, secondary, structure alpha, with or without graft")

    def _fact(self, text):
        return ClinicalFact(
            kind=FactKind.PROCEDURE, description="deposits removed",
            evidence=[EvidenceSpan(text=text, anchored=True, span_id="s1")],
            confidence=0.9, fact_id="F1")

    def test_an_undocumented_family_stem_leaves_the_siblings_ungrounded(self):
        pool = [self.LOCAL, self.GENERAL, self.REPAIR]
        fact = self._fact("Deposits near the structure alpha attachment were removed.")
        self.assertEqual(set(resolution._baseline_ungrounded_pool(fact, pool, None)),
                         {"CAND_LOC", "CAND_GEN", "CAND_REP"})

    def test_a_documented_family_stem_grounds_the_siblings(self):
        pool = [self.LOCAL, self.GENERAL, self.REPAIR]
        fact = self._fact("A percutaneous division of structure alpha was performed.")
        self.assertEqual(resolution._baseline_ungrounded_pool(fact, pool, None), {})
        self.assertIsNone(resolution._baseline_descriptor_grounded(fact, self.LOCAL, pool, None))

    def test_siblings_differing_only_by_laterality_still_fail_open(self):
        pool = [_cand("CAND_R", "assembly service, right structure alpha"),
                _cand("CAND_L", "assembly service, left structure alpha")]
        fact = self._fact("Something else entirely was documented.")
        self.assertEqual(resolution._baseline_ungrounded_pool(fact, pool, None), {})
