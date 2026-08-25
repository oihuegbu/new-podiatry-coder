"""claude_coder.verify's per-requirement judging extension (issue #6 F9-R6, Phase 1)
— additive to the existing whole-shortlist entailment contract, never a replacement.
The one invariant every test here ultimately protects: when a shortlist compiles NO
requirements, the prompt and the parsed `Judgement` are indistinguishable from
before this field existed — zero format-regression risk for the vast majority of
shortlists this phase doesn't touch. Synthetic descriptors/facts throughout.
"""
import json
import unittest

from claude_coder import requirement as req
from claude_coder import verify as _verify
from claude_coder.models import (AttributeEvidence, CandidateCode, ClinicalFact,
                                 EvidenceSpan, FactKind)


def _fact(*evidence_texts):
    spans = [EvidenceSpan(text=t, start=0, end=len(t), anchored=True, span_id=f"s{i}")
            for i, t in enumerate(evidence_texts)]
    return ClinicalFact(kind=FactKind.PROCEDURE, description="assembly service",
                        evidence=spans, confidence=0.99, fact_id="F1")


LEFT = CandidateCode("CAND_LEFT", "cpt", "assembly service performed on the left",
                     score=0.9, source="retrieval")
RIGHT = CandidateCode("CAND_RIGHT", "cpt", "assembly service performed on the right",
                      score=0.9, source="retrieval")


class _Source:
    def descriptions(self, code, system):
        return []   # forces _best_descriptor's fallback to cand.descriptor


def _requirements():
    return req.compile_requirements([LEFT, RIGHT])


class ShortlistPromptTest(unittest.TestCase):

    def test_empty_requirements_render_byte_identical_to_before(self):
        fact = _fact("performed on the left")
        with_empty, id_map = _verify._shortlist_prompt(fact, [LEFT, RIGHT], _Source(), ())
        self.assertNotIn("REQUIREMENTS:", with_empty)
        self.assertNotIn("[e1]", with_empty)          # plain evidence text, no bracketed ids
        self.assertIn("performed on the left", with_empty)
        self.assertEqual(id_map, {})

    def test_nonempty_requirements_render_a_requirements_section(self):
        fact = _fact("performed on the left")
        prompt, id_map = _verify._shortlist_prompt(fact, [LEFT, RIGHT], _Source(),
                                                    _requirements())
        self.assertIn("REQUIREMENTS:", prompt)
        self.assertIn("laterality", prompt)
        self.assertIn("[e1]", prompt)                 # bracketed evidence id now shown
        self.assertEqual(id_map, {"e1": "s0"})

    def test_requirements_never_expose_offset_or_source_identity(self):
        """The model must never see validation-only fields it could echo back to
        fake a match."""
        prompt, _ = _verify._shortlist_prompt(_fact("x"), [LEFT, RIGHT], _Source(),
                                              _requirements())
        self.assertNotIn("authority_offset", prompt)
        self.assertNotIn("source_identity", prompt)

    def test_system_prompt_gains_the_requirements_contract_only_when_nonempty(self):
        self.assertNotIn("REQUIREMENT", _verify._SELECT_SYSTEM)
        with_contract = _verify._SELECT_SYSTEM + _verify._REQUIREMENTS_CONTRACT
        self.assertIn("REQUIREMENT", with_contract)


class AttributeEvidenceCitabilityTest(unittest.TestCase):
    """issue #6 F9-R6-R4: `_evidence_options` must also surface `fact.
    attribute_evidence`'s spans -- otherwise a correctly inherited, independently
    validated attribute (e.g. a laterality/site value stated once on a parent
    fact) can settle graph consensus but can never be cited by the requirement
    verifier at all."""

    def test_local_attribute_evidence_span_is_rendered_as_an_additional_citable_id(self):
        from dataclasses import replace
        fact = _fact("performed on the left")
        extra = EvidenceSpan(text="left side confirmed", start=0, end=20,
                             anchored=True, span_id="attr-span")
        fact = replace(fact, attribute_evidence={
            "laterality": (AttributeEvidence(span=extra, scope="local"),)})
        prompt, id_map = _verify._evidence_options(fact)
        self.assertIn("[e2]", prompt)
        self.assertIn("left side confirmed", prompt)
        self.assertEqual(id_map["e2"], "attr-span")

    def test_unvalidated_inherited_attribute_evidence_is_not_citable(self):
        from dataclasses import replace
        fact = _fact("performed on the left")
        extra = EvidenceSpan(text="inherited elsewhere", start=0, end=19,
                             anchored=True, span_id="inherited-span")
        fact = replace(fact, attribute_evidence={
            "laterality": (AttributeEvidence(span=extra, scope="inherited",
                                             scope_validated=False),)})
        prompt, id_map = _verify._evidence_options(fact)
        self.assertNotIn("inherited elsewhere", prompt)
        self.assertNotIn("inherited-span", id_map.values())

    def test_validated_inherited_attribute_evidence_is_citable(self):
        from dataclasses import replace
        fact = _fact("performed on the left")
        extra = EvidenceSpan(text="inherited and validated", start=0, end=24,
                             anchored=True, span_id="validated-span")
        fact = replace(fact, attribute_evidence={
            "laterality": (AttributeEvidence(span=extra, scope="inherited",
                                             scope_validated=True),)})
        prompt, id_map = _verify._evidence_options(fact)
        self.assertIn("inherited and validated", prompt)
        self.assertIn("validated-span", id_map.values())

    def test_attribute_evidence_span_already_in_fact_evidence_is_not_duplicated(self):
        from dataclasses import replace
        fact = _fact("performed on the left")
        same_span = fact.evidence[0]
        fact = replace(fact, attribute_evidence={
            "laterality": (AttributeEvidence(span=same_span, scope="local"),)})
        prompt, id_map = _verify._evidence_options(fact)
        self.assertEqual(prompt.count("[e"), 1)
        self.assertEqual(len(id_map), 1)


class EvidenceTextBySpanIdTest(unittest.TestCase):
    """issue #6 F9-R6-R2, second re-review: `evidence_text_by_span_id` is the
    span_id -> text lookup `requirement.validated_requirement` uses to check a
    SUPPORTED citation's own content -- built from the SAME `_citable_evidence`
    source `_evidence_options` renders, so the two can never define "citable"
    differently."""

    def test_matches_the_same_spans_evidence_options_renders(self):
        from dataclasses import replace
        fact = _fact("performed on the left")
        extra = EvidenceSpan(text="left side confirmed", start=0, end=20,
                             anchored=True, span_id="attr-span")
        fact = replace(fact, attribute_evidence={
            "laterality": (AttributeEvidence(span=extra, scope="local"),)})
        _, id_to_span = _verify._evidence_options(fact)
        text_by_id = _verify.evidence_text_by_span_id(fact)
        self.assertEqual(set(text_by_id), set(id_to_span.values()) - {""})
        self.assertEqual(text_by_id["attr-span"], "left side confirmed")

    def test_fact_evidence_span_text_is_looked_up_correctly(self):
        fact = _fact("performed on the left")
        text_by_id = _verify.evidence_text_by_span_id(fact)
        self.assertEqual(text_by_id["s0"], "performed on the left")

    def test_unvalidated_inherited_span_is_not_in_the_lookup(self):
        from dataclasses import replace
        fact = _fact("performed on the left")
        extra = EvidenceSpan(text="inherited elsewhere", start=0, end=19,
                             anchored=True, span_id="inherited-span")
        fact = replace(fact, attribute_evidence={
            "laterality": (AttributeEvidence(span=extra, scope="inherited",
                                             scope_validated=False),)})
        self.assertNotIn("inherited-span", _verify.evidence_text_by_span_id(fact))


class RequirementJudgementParsingTest(unittest.TestCase):

    def test_no_requirements_compiled_never_parses_a_requirements_field_even_if_present(self):
        """Defense in depth: even if a model somehow emitted a "requirements" key
        for a shortlist that compiled none, it must be ignored -- there is nothing
        to validate it against."""
        ans = {"requirements": [{"requirement_id": "bogus", "status": "supported"}]}
        out = _verify._requirement_judgements(ans, (), {}, {})
        self.assertEqual(out, ())

    def test_well_formed_judgement_parses(self):
        reqs = _requirements()
        target = reqs[0]
        ans = {"requirements": [
            {"requirement_id": target.requirement_id, "status": "supported",
             "span_ids": ["s0"], "quote": "left"}]}
        out = _verify._requirement_judgements(ans, reqs, {"s0": "s0"}, {"provider": "test"})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].requirement_id, target.requirement_id)
        self.assertEqual(out[0].status, req.RequirementStatus.SUPPORTED)
        self.assertEqual(out[0].evidence_span_ids, ("s0",))

    def test_unknown_requirement_id_is_dropped(self):
        reqs = _requirements()
        ans = {"requirements": [
            {"requirement_id": "never-compiled", "status": "supported",
             "span_ids": [], "quote": ""}]}
        self.assertEqual(_verify._requirement_judgements(ans, reqs, {}, {}), ())

    def test_invalid_status_is_dropped(self):
        reqs = _requirements()
        target = reqs[0]
        ans = {"requirements": [
            {"requirement_id": target.requirement_id, "status": "maybe-ish"}]}
        self.assertEqual(_verify._requirement_judgements(ans, reqs, {}, {}), ())

    def test_a_cited_id_the_shortlist_never_showed_is_dropped_not_invented(self):
        reqs = _requirements()
        target = reqs[0]
        ans = {"requirements": [
            {"requirement_id": target.requirement_id, "status": "supported",
             "span_ids": ["made-up-id"]}]}
        out = _verify._requirement_judgements(ans, reqs, {"s0": "s0"}, {})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].evidence_span_ids, ())   # the fake id never made it through

    def test_malformed_requirements_field_is_ignored_not_a_crash(self):
        reqs = _requirements()
        self.assertEqual(_verify._requirement_judgements(
            {"requirements": "not-a-list"}, reqs, {}, {}), ())
        self.assertEqual(_verify._requirement_judgements(
            {"requirements": [42, "nope", None]}, reqs, {}, {}), ())

    def test_duplicate_requirement_id_keeps_only_the_first(self):
        reqs = _requirements()
        target = reqs[0]
        ans = {"requirements": [
            {"requirement_id": target.requirement_id, "status": "supported",
             "span_ids": ["s0"]},
            {"requirement_id": target.requirement_id, "status": "contradicted",
             "span_ids": ["s0"]}]}
        out = _verify._requirement_judgements(ans, reqs, {"s0": "s0"}, {})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].status, req.RequirementStatus.SUPPORTED)


class JudgementBackwardCompatibilityTest(unittest.TestCase):

    def test_judgement_defaults_to_no_requirement_judgements(self):
        j = _verify.Judgement(entailed=("X",))
        self.assertEqual(j.requirement_judgements, ())
        self.assertEqual(j.as_record()["requirements"], [])

    def test_select_entailed_and_corroborate_default_requirements_to_empty(self):
        """Every existing call site (including every fixture in
        tests/shortlist_verdict.py) that doesn't pass `requirements` at all must
        keep working exactly as before."""
        fact = _fact("performed on the left")
        stub = lambda system, user: json.dumps(
            {"choice": 1, "entailed": [1], "reason": "stub",
             "eliminated": [{"option": 2, "reason": "stub: not entailed",
                            "missing_element": False}]})
        judgement = _verify.select_entailed(fact, [LEFT, RIGHT], _Source(), stub)
        self.assertEqual(judgement.requirement_judgements, ())
        self.assertTrue(judgement.entails("CAND_LEFT"))


if __name__ == "__main__":
    unittest.main()
