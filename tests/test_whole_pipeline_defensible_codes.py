"""THE single, reusable, whole-pipeline verification harness (issue #6): any
fix to retrieval, eligibility, tie-resolution, or verification can be run
through `run_pipeline()` below to confirm a DEFENSIBLE CODE actually reaches
the released ClaimBundle -- not merely that some isolated function returns
the value a unit test expected of it.

Per the free-verification-before-real-note policy: a unit-level test of one
function in isolation is not sufficient proof a fix works end to end. This
module runs the REAL `pipeline.code_encounter()` entrypoint -- the SAME one
a real note goes through: extract -> resolve -> arbitrate -> gate ->
autonomy -> certificate -- with every model-facing callable a deterministic,
scripted stub. It spends NO real API credits and makes NO network calls
(safe to run with `--network none`).

To verify a NEW fix with this harness: build a `MockSource` shaped like the
fix's real bug (real candidate descriptors reproduced with SYNTHETIC
vocabulary, never a hardcoded real code/scenario), a minimal extraction
`facts_json` (see `_facts_json` below for the wire shape), and one or two
scripted `shortlist_verdict.judge(...)` callables; call `run_pipeline(...)`;
assert on `defensible_codes(result)` (the set of codes that actually reached
`result.billable_lines`) and/or `result.verdict`/`result.notes` for a
scenario whose correct outcome is a held/reviewed line rather than a release
-- both are "defensible": a wrongly-released code and a wrongly-silent hold
are the SAME category of defect this harness exists to catch.
"""
import json
import unittest

from claude_coder.data_access import MockSource
from claude_coder.models import CandidateCode, Verdict
from claude_coder.pipeline import code_encounter
from claude_coder.provenance import NullAuditRepository
from claude_coder.verify import declare_model_profile
from tests import shortlist_verdict as _sv


def run_pipeline(note_text, facts_json, source, *, verify_llm=None,
                 corroborate_llm=None, dos="2026-01-01", encounter_id="enc-verify"):
    """Runs the REAL `code_encounter()` entrypoint end to end. Every model
    callable is a scripted stub (`extract_llm` returns `facts_json` verbatim;
    `verify_llm`/`corroborate_llm` are `shortlist_verdict.judge(...)`
    callables, or `declare_model_profile`-wrapped ones for a genuinely
    independent two-model scenario) -- zero LLM credits spent, ever."""
    return code_encounter(
        encounter_id, note_text, dos, source=source,
        extract_llm=lambda s, u: facts_json,
        verify_llm=verify_llm, corroborate_llm=corroborate_llm,
        audit_repository=NullAuditRepository(),
        billing_context={
            "billing_entity_id": "actor-1",
            "participants": [{"id": "actor-1", "type": "person", "roles": ["performer"]}]})


def defensible_codes(result) -> set[str]:
    """The codes that actually reached the released ClaimBundle's billable
    lines -- the one assertion every fix's whole-pipeline scenario shares."""
    return {ln.chosen.code for ln in result.billable_lines if ln.chosen}


def _declare(*, entails, provider, reason="stub", **judge_kwargs):
    """A FRESH `shortlist_verdict.judge(...)` callable, declared under
    `provider` in one step. `declare_model_profile` STAMPS the provider
    identity directly onto the callable object it is given
    (`fn.model_profile = ...`) -- passing the SAME judge object to this
    twice for two different providers silently overwrites the first
    declaration with the second (both roles then carry the LAST provider
    stamped), which reads to the resolver as "one vendor's opinion sampled
    twice" and downgrades a genuine two-model VERIFIED release to a bare,
    coder-routed ARBITRATED one -- discovered live while building this
    harness. Always builds a NEW judge object per call so two declared
    roles can never alias the same underlying callable by accident."""
    judge = _sv.judge(entails=entails, reason=reason, **judge_kwargs)
    return declare_model_profile(judge, provider=provider)


def _facts_json(facts, relations=()):
    return json.dumps({"facts": facts, "relations": list(relations)})


class WholePipelineDefensibleCodes(unittest.TestCase):
    """One reusable harness, exercised by every fix's own scenario below.
    Synthetic vocabulary throughout -- no real medical code anywhere in this
    file, matching this suite's own established convention."""

    def test_a_grounded_candidate_reaches_the_claim_bundle_past_a_poisoned_sibling(self):
        """issue #6, independent root-cause investigation (real note: CPT
        24305 "Tendon lengthening, upper arm or elbow, EACH TENDON" wrongly
        survived eligibility against a fact documenting "Achilles tendon").
        Whole-pipeline proof: a candidate whose only anatomy signal is a
        cardinality-led phrase ("each structure") must never reach the
        released ClaimBundle when a genuinely grounded, independently
        verified sibling was retrieved for the same fact -- and the grounded
        sibling itself must actually arrive there, AUTO_READY, with a
        genuinely independent two-model verification behind it."""
        link = "Repair of structure alpha was performed for condition alpha of the right side"
        note = ("Procedure: repair of structure alpha. "
                "Assessment: condition alpha, right side. " + link + ".")
        facts_json = _facts_json(
            facts=[
                {"fact_id": "F1", "kind": "procedure",
                 "description": "repair of structure alpha",
                 "attributes": {"anatomy": "structure alpha", "laterality": "right",
                                "performer_id": "actor-1", "billing_entity_id": "actor-1"},
                 "attribute_evidence": {
                     "anatomy": [{"text": "Repair of structure alpha", "scope": "local",
                                 "assertion_state": "asserted", "value": "structure alpha"}],
                     "laterality": [{"text": "condition alpha, right side", "scope": "local",
                                    "assertion_state": "asserted", "value": "right"}]},
                 "disposition": "performed_today", "negated": False,
                 "evidence": ["Repair of structure alpha",
                             "Repair of structure alpha was performed"],
                 "confidence": 0.97,
                 "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                                    "temporal": 0.99, "performer": 0.99,
                                    "relationship": 0.99}},
                {"fact_id": "F2", "kind": "diagnosis",
                 "description": "condition alpha of the right side",
                 "attributes": {"laterality": "right"},
                 "attribute_evidence": {
                     "laterality": [{"text": "condition alpha, right side", "scope": "local",
                                    "assertion_state": "asserted", "value": "right"}]},
                 "disposition": "performed_today", "negated": False,
                 "evidence": ["condition alpha, right side",
                             "condition alpha of the right side"],
                 "confidence": 0.98,
                 "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                                    "temporal": 0.99, "assertion": 0.99,
                                    "experiencer": 0.99}},
            ],
            relations=[
                {"subject_event_id": "F2", "object_event_id": "F1", "predicate": "reason_for",
                 "state": "asserted", "evidence_fact_ids": ["F1", "F2"], "confidence": 0.99},
            ])

        grounded_desc = "Repair, structure alpha"
        cardinality_desc = "Lengthening, other site, each structure"
        source = MockSource(
            records={
                ("GROUNDED", "cpt"): {"long_description": grounded_desc, "active": True},
                ("CARDINALITY_ONLY", "cpt"): {"long_description": cardinality_desc,
                                             "active": True},
                ("DX_ALPHA_RIGHT", "icd10"): {"long_description": "condition alpha, right side",
                                             "active": True}},
            # A governed relation exists for the cardinality-led phrase ITSELF
            # (its generic head noun ancestor-relates to the fact's own
            # documented structure, exactly like "tendon" does to "Achilles
            # tendon" in the real bug) -- without the fix, this is exactly
            # what would wrongly ground CARDINALITY_ONLY.
            concept_relation={("structure alpha", "structure alpha"): "same",
                             ("structure alpha", "each structure"): "ancestor_descendant"},
            retrieval={("*", "cpt"): [CandidateCode("GROUNDED", "cpt", grounded_desc, 0.8),
                                     CandidateCode("CARDINALITY_ONLY", "cpt",
                                                  cardinality_desc, 0.7)],
                      ("*", "icd10"): [CandidateCode("DX_ALPHA_RIGHT", "icd10",
                                                    "condition alpha, right side", 0.9)]},
            index={"condition alpha of the right side": {"DX_ALPHA_RIGHT"}})

        def entails(d):
            return "repair" in d.lower() or "condition alpha" in d.lower()

        result = run_pipeline(
            note, facts_json, source,
            verify_llm=_declare(entails=entails, provider="provider-a",
                               reason="documented act matches"),
            corroborate_llm=_declare(entails=entails, provider="provider-b",
                                    reason="documented act matches"))

        self.assertEqual(result.verdict, Verdict.AUTO_READY, result.notes)
        codes = defensible_codes(result)
        self.assertEqual(codes, {"GROUNDED", "DX_ALPHA_RIGHT"}, result.notes)
        self.assertNotIn("CARDINALITY_ONLY", codes)

    def test_an_undocumented_indication_clause_becomes_a_provider_question_not_a_silent_hold(self):
        """issue #6, independent root-cause investigation (real note: F1,
        CPT 28118 vs 28120, "(eg, osteomyelitis or bossing)"). Whole-pipeline
        proof of the OTHER defensible outcome this harness must also catch:
        when the record genuinely does not settle a tie, the pipeline must
        never silently force a code into the ClaimBundle NOR fall back to an
        unanswerable, unrescuable hold -- it must route to ONE specific,
        answerable provider question naming the exact undocumented fact.
        Before this fix, this exact scripted disagreement (a looser primary
        judge, a corroborator correctly flagging the indication-requiring
        candidate's own precondition as undocumented) had no governed axis
        to resolve through and fell to a permanent SYSTEM_UNRESOLVED hold --
        a defect this harness would have caught just as decisively as the
        positive-release case above."""
        note = "Assembly service performed today for structure alpha."
        facts_json = _facts_json(facts=[
            {"fact_id": "F1", "kind": "procedure",
             "description": "assembly service performed today",
             "attributes": {"performer_id": "actor-1", "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Assembly service performed today"], "confidence": 0.97,
             "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                                "temporal": 0.99, "performer": 0.99, "relationship": 0.99}},
        ])

        alpha_desc = "assembly service, broad category"
        beta_desc = "assembly service, broad category (eg, variant condition)"
        source = MockSource(
            records={("CAND_ALPHA", "cpt"): {"long_description": alpha_desc, "active": True},
                    ("CAND_BETA", "cpt"): {"long_description": beta_desc, "active": True}},
            retrieval={("*", "cpt"): [CandidateCode("CAND_ALPHA", "cpt", alpha_desc, 0.8),
                                     CandidateCode("CAND_BETA", "cpt", beta_desc, 0.7)]})

        # Primary: looser, entails both (mirrors the real note's real
        # evaluator that wrongly accepted the indication-clause candidate).
        primary = _declare(entails=lambda d: True, provider="provider-a",
                          reason="documented act matches")
        # Corroborator: the indication clause's own fact ("variant condition")
        # is genuinely undocumented -- correctly flags it as missing (mirrors
        # the real note's other evaluator, which correctly rejected it).
        corroborator = _declare(
            entails=lambda d: "variant condition" not in d.lower(), provider="provider-b",
            missing_element=True, reason="variant condition is not documented")

        result = run_pipeline(note, facts_json, source,
                              verify_llm=primary, corroborate_llm=corroborator)

        self.assertEqual(defensible_codes(result), set(), result.notes)
        self.assertIn(result.verdict, (Verdict.REVIEW_REQUIRED,), result.notes)
        joined_notes = " ".join(result.notes)
        self.assertIn("PROVIDER_QUERY", joined_notes)
        self.assertIn("variant condition", joined_notes)
        self.assertNotIn("SYSTEM_UNRESOLVED", joined_notes)
