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

issue #6, independent verification: the RESULT this harness inspects is the
same canonical `ClaimBundle` payload production actually writes to
`output/results/*_results.json` -- `app.contracts.claim_bundle.
bundle_from_coding_result(result, ...).to_payload()`, the EXACT call
`run.py`'s own `build_bundle()` makes -- never the raw internal
`CodingResult`'s ad-hoc `.billable_lines`/`.notes` attributes (what an
earlier draft of this file inspected, before this was checked against the
real production shape and found NOT to mirror it: `.billable_lines` is a
pipeline-internal convenience property, not what a real note's artifact
ever contains). "Defensible", for a code in this payload, is production's
own vocabulary: `external_disposition == "BILLABLE_AND_DEFENSIBLE"`
(`app.contracts.claim_bundle.ExternalDisposition`).

To verify a NEW fix with this harness: build a `MockSource` shaped like the
fix's real bug (real candidate descriptors reproduced with SYNTHETIC
vocabulary, never a hardcoded real code/scenario), a minimal extraction
`facts_json` (see the wire shape in the scenarios below), and one or two
scripted `shortlist_verdict.judge(...)` callables (via `_declare`, never the
same judge object declared twice -- see its own docstring); call
`run_pipeline(...)`, then `build_claim_bundle_payload(result)`; assert on
`defensible_codes(payload)` and/or `payload["release"]["producer_verdict"]`/
`payload["candidate_lines"]` for a scenario whose correct outcome is a
held/reviewed line rather than a release -- both are "defensible": a
wrongly-released code and a wrongly-silent hold are the SAME category of
defect this harness exists to catch. A MINIMAL synthetic scenario's
`payload["release"]["holds"]` will still legitimately list every missing
real-world claim field this harness never supplies (patient demographics,
payer, coverage, authoritative-data fingerprints) -- that is expected and
orthogonal to what a fix under test changes; `producer_verdict`/
`external_disposition` are the fields that answer "did resolution pick (or
correctly withhold) a defensible code," which is what this harness verifies.
"""
import json
import unittest

from app.contracts.claim_bundle import (AuthorityBinding, SourceDocument,
                                        bundle_from_coding_result)
from app.contracts.encounter_context import EncounterContext
from claude_coder.data_access import MockSource
from claude_coder.models import CandidateCode
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


def build_claim_bundle_payload(result) -> dict:
    """The SAME canonical ClaimBundle JSON payload production writes to
    `output/results/*_results.json` -- `run.py`'s `build_bundle()` makes
    this EXACT call (`bundle_from_coding_result(...).to_payload()`), just
    with a real PDF's `SourceDocument`/resolved `EncounterContext` in place
    of the minimal defaults a synthetic scenario has no need to supply."""
    bundle = bundle_from_coding_result(
        result, source_document=SourceDocument(), context=EncounterContext(),
        authority=AuthorityBinding())
    return bundle.to_payload()


def defensible_codes(payload: dict) -> set[str]:
    """The codes production itself calls defensible: every `service_lines`/
    `diagnoses` entry in the REAL ClaimBundle payload whose
    `external_disposition` is `BILLABLE_AND_DEFENSIBLE` -- the one assertion
    every fix's whole-pipeline scenario shares."""
    lines = (*payload.get("service_lines", ()), *payload.get("diagnoses", ()))
    return {ln["code"] for ln in lines
           if ln.get("external_disposition") == "BILLABLE_AND_DEFENSIBLE"}


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

        payload = build_claim_bundle_payload(result)
        self.assertEqual(payload["release"]["producer_verdict"], "AUTO_READY", payload)
        codes = defensible_codes(payload)
        self.assertEqual(codes, {"GROUNDED", "DX_ALPHA_RIGHT"}, payload)
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

        payload = build_claim_bundle_payload(result)
        self.assertEqual(defensible_codes(payload), set(), payload)
        self.assertEqual(payload["release"]["producer_verdict"], "REVIEW_REQUIRED", payload)
        self.assertEqual(payload["release"]["producer_destination"], "PROVIDER_QUERY", payload)
        (line,) = payload["candidate_lines"]
        self.assertEqual(line["external_disposition"], "CANDIDATE_REQUIRING_FACT", line)
        self.assertIn("variant condition", line["blocking_reason"])
        self.assertIn("indication_clause", line["blocking_reason"])
        self.assertNotIn("SYSTEM_UNRESOLVED", line["blocking_reason"])
