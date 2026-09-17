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

`build_claim_bundle_payload(result)` binds a FULLY POPULATED, RESOLVED
patient/subscriber/payer/rendering-provider/billing-entity/affiliation/
coverage context (`fully_resolved_context()`) and a populated authority
binding (`synthetic_authority_binding()`) by default -- synthetic
identifiers, but a real, valid instance of the same contract object a
genuine roster-backed deployment produces, not the empty `EncounterContext()`
`problems()` correctly refuses to call releasable. A scenario whose own fix
is the only thing that could still block release therefore reaches a TRUE,
zero-`holds` AUTO_READY -- `payload["release"]["holds"] == []` -- exactly
mirroring what a genuinely complete real claim looks like, not merely a
producer-level verdict with a page of unrelated envelope holds still listed.
A scenario that wants to isolate the resolution-layer verdict from the
claim-envelope layer instead may pass `context=EncounterContext()` explicitly.

To verify a NEW fix with this harness: build a `MockSource` shaped like the
fix's real bug (real candidate descriptors reproduced with SYNTHETIC
vocabulary, never a hardcoded real code/scenario), a minimal extraction
`facts_json` (see the wire shape in the scenarios below), and one or two
scripted `shortlist_verdict.judge(...)` callables (via `_declare`, never the
same judge object declared twice -- see its own docstring); call
`run_pipeline(...)`, then `build_claim_bundle_payload(result)`; assert on
`defensible_codes(payload)` and/or `payload["release"]["producer_verdict"]`/
`holds`/`payload["candidate_lines"]` for a scenario whose correct outcome is
a held/reviewed line rather than a release -- both are "defensible": a
wrongly-released code and a wrongly-silent hold are the SAME category of
defect this harness exists to catch.
"""
import json
import unittest

from app.contracts.claim_bundle import (
    AffiliationBinding, AuthorityBinding, BillingEntityIdentity,
    CONTEXT_SERVICE_DATE_SOURCE, ContextResolution, CoverageBinding,
    EncounterContext, PatientIdentity, PayerIdentity, ProviderIdentity,
    REQUIRED_ENCOUNTER_CONTEXT, ServiceDateBinding, SourceDocument,
    SubscriberIdentity, AUTHORITATIVE_FIELD_SOURCE, bundle_from_coding_result)
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


def fully_resolved_context(dos: str = "2026-01-01") -> EncounterContext:
    """A completely populated, RESOLVED `EncounterContext` -- every field
    `REQUIRED_ENCOUNTER_CONTEXT` names, each declared `AUTHORITATIVE_FIELD_
    SOURCE` (never note-derived), plus a reproducing fingerprint. Synthetic
    identifiers throughout (this suite's own no-real-PII/no-real-code
    convention), but a REAL, valid instance of the same contract object a
    genuine roster-backed deployment produces -- not the empty
    `EncounterContext()` default, which `problems()` correctly refuses to
    call releasable (a claim cannot be built from note-extracted context
    alone). Confirms a fix's scenario can reach a TRUE, zero-`holds`
    AUTO_READY -- not merely a producer-level verdict -- when nothing else
    about the claim blocks it."""
    patient = PatientIdentity(patient_id="PAT-1", first_name="Alexis", last_name="Quintero",
                              date_of_birth="1982-09-02", gender="F")
    subscriber = SubscriberIdentity(member_id="MEM-1", group_number="GRP-1")
    payer = PayerIdentity(name="Synthetic Payer", payer_id="PAYER-1")
    provider = ProviderIdentity(npi="1888888888", first_name="Robin", last_name="Vasquez")
    billing_entity = BillingEntityIdentity(entity_id="ENTITY-1",
                                           name="Synthetic Practice PLLC",
                                           npi="1999999998")
    affiliation = AffiliationBinding(affiliation_id="AFF-1", provider_npi=provider.npi,
                                     billing_entity_id=billing_entity.entity_id,
                                     effective_start="2020-01-01")
    coverage = CoverageBinding(coverage_id="COV-1", patient_id=patient.patient_id,
                               payer_id=payer.payer_id, effective_start="2020-01-01")
    service_date = ServiceDateBinding(date_of_service=dos, source=CONTEXT_SERVICE_DATE_SOURCE,
                                      declared_date=dos)
    field_sources = {path: AUTHORITATIVE_FIELD_SOURCE for path in REQUIRED_ENCOUNTER_CONTEXT}
    context = EncounterContext(
        resolution=ContextResolution.RESOLVED, provider_id="synthetic-test-roster",
        context_version="v1", service_date=service_date,
        patient=patient, subscriber=subscriber, payer=payer, rendering_provider=provider,
        billing_entity=billing_entity, affiliation=affiliation, coverage=coverage,
        place_of_service="11", jurisdiction="FL", field_sources=field_sources)
    return context.model_copy(update={"fingerprint": context.compute_fingerprint()})


def synthetic_authority_binding() -> AuthorityBinding:
    """A populated `AuthorityBinding` -- synthetic-but-well-formed digests
    standing in for the real compliance.db/source-manifest bytes a live
    `AuthoritativeSource` binds (this harness's `MockSource` has no real
    bytes to hash). Orthogonal to the patient/payer/billing context above:
    this is what closes the remaining "no authoritative-data fingerprint"/
    "no compiled-database snapshot" holds so a scenario can reach TRUE
    zero-`holds` AUTO_READY end to end."""
    return AuthorityBinding(
        data_fingerprint="sha256:" + "a" * 64,
        source_manifest_fingerprint="sha256:" + "b" * 64,
        database_snapshot_digest="sha256:" + "c" * 64)


def build_claim_bundle_payload(result, *, context=None, authority=None,
                              source_document=None) -> dict:
    """The SAME canonical ClaimBundle JSON payload production writes to
    `output/results/*_results.json` -- `run.py`'s `build_bundle()` makes
    this EXACT call (`bundle_from_coding_result(...).to_payload()`).
    Defaults to a fully populated, RESOLVED context/authority/source
    document (see `fully_resolved_context`/`synthetic_authority_binding`
    above) so a scenario proves it can reach a TRUE, zero-`holds`
    AUTO_READY when its own fix's mechanism is the only thing that could
    still block it -- pass `context=EncounterContext()` explicitly for a
    scenario that wants to isolate the resolution-layer verdict from the
    claim-envelope layer instead."""
    bundle = bundle_from_coding_result(
        result,
        source_document=source_document or SourceDocument(
            document_version="sha256:" + "d" * 64,
            extracted_text_sha256="sha256:" + "e" * 64, page_count=1),
        context=context if context is not None else fully_resolved_context(),
        authority=authority if authority is not None else synthetic_authority_binding())
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
        self.assertEqual(payload["release"]["destination"], "AUTO_READY", payload)
        self.assertEqual(payload["release"]["holds"], [],
                         "a fully populated patient/payer/billing context and "
                         "authority binding must reach a TRUE, zero-hold release")
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

    def test_crossed_governed_equivalent_candidates_reach_the_claim_bundle(self):
        """issue #6, independent investigation ("same code, different
        wording"): evaluator A independently entails an OLDER code entry for
        a documented procedure; evaluator B independently entails a
        DIFFERENT, NEWER code entry for the SAME real-world procedure. Naively
        this is indistinguishable from the F8-R1 "two shortlisted candidates
        both still entailed" tie -- but the governed SNOMED Procedure concept
        graph resolves the two code entries to one procedure, so this must
        release, AUTO_READY, exactly like a genuinely unique pick -- not fall
        to a permanent, unrescuable hold over two evaluators who actually
        agree. Whole-pipeline proof of `resolution.
        _corroborated_via_equivalent_concept`/the matching elimination path in
        `_uniqueness_view`, both of which call `coreference.
        governed_procedure_relation` -- the STRICT, source-backed-only half
        of the F9-R4 governed concept-graph mechanism, deliberately never
        `action_relation_detail`'s free wording shortcut (unsafe for
        comparing two candidates' own descriptors to each other -- see that
        function's docstring) -- applied here to two candidates' own
        official descriptors."""
        note = ("Excision of structure alpha performed today for condition "
                "alpha of the right side.")
        facts_json = _facts_json(
            facts=[
                {"fact_id": "F1", "kind": "procedure",
                 "description": "excision of structure alpha",
                 "attributes": {"performer_id": "actor-1", "billing_entity_id": "actor-1"},
                 "disposition": "performed_today", "negated": False,
                 "evidence": ["Excision of structure alpha performed today"],
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
                 "evidence": ["condition alpha of the right side"],
                 "confidence": 0.98,
                 "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                                    "temporal": 0.99, "assertion": 0.99,
                                    "experiencer": 0.99}},
            ],
            relations=[
                {"subject_event_id": "F2", "object_event_id": "F1", "predicate": "reason_for",
                 "state": "asserted", "evidence_fact_ids": ["F1", "F2"], "confidence": 0.99},
            ])

        old_desc = "Excision, structure alpha, older code entry"
        new_desc = "Excision, structure alpha, newer code entry"
        dx_desc = "condition alpha, right side"
        source = MockSource(
            records={("CODE_OLD", "cpt"): {"long_description": old_desc, "active": True},
                    ("CODE_NEW", "cpt"): {"long_description": new_desc, "active": True},
                    ("DX_ALPHA_RIGHT", "icd10"): {"long_description": dx_desc,
                                                 "active": True}},
            retrieval={("*", "cpt"): [CandidateCode("CODE_OLD", "cpt", old_desc, 0.9),
                                     CandidateCode("CODE_NEW", "cpt", new_desc, 0.8)],
                      ("*", "icd10"): [CandidateCode("DX_ALPHA_RIGHT", "icd10", dx_desc, 0.9)]},
            index={"condition alpha of the right side": {"DX_ALPHA_RIGHT"}},
            # The governed concept graph confirms the two code entries name
            # the SAME real-world procedure -- the crossed disagreement below
            # must resolve through this, never through an invented heuristic.
            procedure_relation={(old_desc, new_desc): {
                "verdict": "same",
                "term_a": {"term": "a", "candidates": ["C1"], "method": "exact",
                          "unique": True},
                "term_b": {"term": "b", "candidates": ["C1"], "method": "exact",
                          "unique": True}}})

        # Primary independently entails only the OLDER code entry (plus the
        # undisputed diagnosis); the corroborator independently entails only
        # the NEWER one (plus the same diagnosis) -- a genuine crossed pick
        # on the procedure, not a scripted agreement.
        primary = _declare(entails=lambda d: "older" in d or "condition alpha" in d.lower(),
                          provider="provider-a", reason="matches the older entry")
        corroborator = _declare(
            entails=lambda d: "newer" in d or "condition alpha" in d.lower(),
            provider="provider-b", reason="matches the newer entry")

        result = run_pipeline(note, facts_json, source,
                              verify_llm=primary, corroborate_llm=corroborator)

        payload = build_claim_bundle_payload(result)
        self.assertEqual(payload["release"]["producer_verdict"], "AUTO_READY", payload)
        self.assertEqual(payload["release"]["destination"], "AUTO_READY", payload)
        self.assertEqual(payload["release"]["holds"], [], payload)
        codes = defensible_codes(payload)
        self.assertEqual(codes, {"CODE_OLD", "DX_ALPHA_RIGHT"}, payload)
        self.assertNotIn("CODE_NEW", codes)
