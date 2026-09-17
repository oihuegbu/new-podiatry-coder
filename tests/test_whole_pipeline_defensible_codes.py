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
`facts_json` (see the wire shape in the scenarios below), and a scripted
`shortlist_verdict.judge(...)` callable (via `_declare`); call
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


def complete_capability_manifest() -> dict:
    """A synthetic, COMPLETE capability manifest -- `gates.source_manifest_gate`
    reads only `missing_required`/`integrity_errors`/`degraded_optional`, so
    this is a minimal, valid instance of exactly what that gate inspects.

    issue #6, independent review (P1-3 correction): `source_manifest_gate`
    is the ONE gate in this pipeline that does not take a `source: CodeSource`
    parameter -- unlike `code_active_gate`/`medical_necessity_gate`/`ncci_gate`
    /etc., which this harness's `MockSource` already correctly substitutes
    for the real authoritative data, `source_manifest_gate` always called
    `capability.build_manifest()`, which probes the REAL, configured
    filesystem paths regardless of what `source` the rest of the pipeline
    used. Confirmed live: this harness's own scenarios passed inside this
    repo's docker test image (which bind-mounts the real, complete
    `data/codes/` directory from the host) and FAILED in an isolated
    checkout lacking it -- the AUTO_READY result was never actually
    reproducible from the test's own fixtures, only from ambient host state.
    Injected via `code_encounter`'s `capability_manifest=` parameter (which
    every REAL caller omits, so production gates against the real,
    configured sources exactly as before), this makes the harness's result
    reproduce identically in ANY environment."""
    return {
        "manifest_version": "test-fixture-v1",
        "required_sources_schema": "test-fixture-v1",
        "sources": [],
        "missing_required": [],
        "degraded_optional": [],
        "integrity_errors": [],
        "status": "OK",
    }


def blocked_capability_manifest(missing_source_id: str = "synthetic_required_source") -> dict:
    """A synthetic manifest with ONE required source missing -- the other
    outcome `source_manifest_gate` must prove, per the same P1-3 correction:
    an absent/corrupt required source BLOCKS release, it does not merely
    happen to pass because the harness never truly exercised the failure
    path either."""
    return {
        "manifest_version": "test-fixture-v1",
        "required_sources_schema": "test-fixture-v1",
        "sources": [],
        "missing_required": [missing_source_id],
        "degraded_optional": [],
        "integrity_errors": [],
        "status": "BLOCKED",
    }


def run_pipeline(note_text, facts_json, source, *, verify_llm=None,
                 dos="2026-01-01", encounter_id="enc-verify",
                 capability_manifest=None):
    """Runs the REAL `code_encounter()` entrypoint end to end. Every model
    callable is a scripted stub (`extract_llm` returns `facts_json` verbatim;
    `verify_llm` is a `shortlist_verdict.judge(...)` callable) -- zero LLM
    credits spent, ever. Code selection runs ONE verifying evaluator; it no
    longer reconciles a second, independently-provider'd corroborator
    against it (see `resolution.resolve`'s own docstring).

    `capability_manifest` defaults to `complete_capability_manifest()` (not
    `None`) so this harness's AUTO_READY scenarios never depend on ambient,
    real on-disk data files -- pass `blocked_capability_manifest()`
    explicitly for a scenario that wants to prove the BLOCKED outcome
    instead (see P1-3 correction above)."""
    return code_encounter(
        encounter_id, note_text, dos, source=source,
        extract_llm=lambda s, u: facts_json,
        verify_llm=verify_llm,
        audit_repository=NullAuditRepository(),
        capability_manifest=(complete_capability_manifest()
                             if capability_manifest is None else capability_manifest),
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
    `provider` in one step (`declare_model_profile` stamps the provider
    identity directly onto the callable object it is given, for the
    ClaimBundle's audit-facing model-profile record). Always builds a NEW
    judge object per call so no two scenarios can ever alias the same
    underlying callable by accident."""
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
        released ClaimBundle when a genuinely grounded sibling was retrieved
        for the same fact -- and the grounded sibling itself must actually
        arrive there, AUTO_READY, with a positively verified entailment
        behind it."""
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

    def test_an_undocumented_indication_clause_never_becomes_a_fabricated_provider_question(self):
        """issue #6, independent root-cause investigation (real note: F1,
        CPT 28118 vs 28120, "(eg, osteomyelitis or bossing)"), then Codex's
        independent re-review (P1-2 correction). Whole-pipeline proof of the
        OTHER defensible outcome this harness must also catch: when the
        record genuinely does not settle a tie, the pipeline must never
        silently force a code into the ClaimBundle -- but it must also never
        fabricate a SPECIFIC provider question out of a purely illustrative
        "(eg, ...)" example with no typed, authoritative requirement field or
        source-governed rule behind it (an earlier version of
        `tiebreak.AXIS_INDICATION_CLAUSE` did exactly that, which Codex
        found unsafe: AMA/CPT convention makes such a clause a
        non-exhaustive example, never a checklist). The evaluator honestly
        finds BOTH the broad and the indication-qualified candidate
        plausible (neither is eliminated by its own say-so); with no
        governed, provider-answerable axis distinguishing them, this must
        route to an honest, generic "candidates still tied" review -- never
        a confident-looking but unearned question naming "variant
        condition" specifically."""
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

        # Looser: entails both candidates plausible (mirrors the real note's
        # real evaluator that could not rule out the indication-clause
        # candidate on its own). Neither candidate is eliminated by this
        # evaluator's own say-so, so the tie reaches the original-document
        # axis check unchanged.
        primary = _declare(entails=lambda d: True, provider="provider-a",
                          reason="documented act matches")

        result = run_pipeline(note, facts_json, source, verify_llm=primary)

        payload = build_claim_bundle_payload(result)
        self.assertEqual(defensible_codes(payload), set(), payload)
        self.assertEqual(payload["release"]["producer_verdict"], "REVIEW_REQUIRED", payload)
        # P1-2 correction: with the illustrative clause no longer selectable/
        # queryable, no governed axis distinguishes these two candidates, so
        # this is an honest, generic coder-review tie -- never a fabricated
        # PROVIDER_QUERY naming an unearned specific fact.
        self.assertEqual(payload["release"]["producer_destination"], "REVIEW", payload)
        (line,) = payload["candidate_lines"]
        self.assertEqual(line["external_disposition"], "EXCLUDED", line)
        self.assertNotIn("variant condition", line["blocking_reason"])
        self.assertNotIn("indication_clause", line["blocking_reason"])
        self.assertIn("still entailed by the documentation", line["blocking_reason"])
        self.assertNotIn("SYSTEM_UNRESOLVED", line["blocking_reason"])


class HarnessHermeticityTest(unittest.TestCase):
    """issue #6, independent review (P1-3 correction): proves this harness's
    `AUTO_READY` result is reproducible from the test's OWN injected fixture,
    never from whatever real data files happen to be mounted wherever the
    test runs -- and that `source_manifest_gate` genuinely BLOCKS on an
    absent required source rather than having no live failure path at all.
    Uses the SAME grounded scenario as
    `test_a_grounded_candidate_reaches_the_claim_bundle_past_a_poisoned_sibling`,
    varying only the injected `capability_manifest`."""

    def _grounded_scenario(self, capability_manifest):
        link = "Repair of structure alpha was performed for condition alpha of the right side"
        note = ("Assessment: condition alpha, right side. " + link + ".")
        facts_json = _facts_json(
            facts=[
                {"fact_id": "F1", "kind": "procedure",
                 "description": "repair of structure alpha",
                 "attributes": {"performer_id": "actor-1", "billing_entity_id": "actor-1"},
                 "disposition": "performed_today", "negated": False,
                 "evidence": ["Repair of structure alpha was performed"],
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
        source = MockSource(
            records={("GROUNDED", "cpt"): {"long_description": grounded_desc, "active": True},
                    ("DX_ALPHA_RIGHT", "icd10"): {"long_description": "condition alpha, right side",
                                                 "active": True}},
            retrieval={("*", "cpt"): [CandidateCode("GROUNDED", "cpt", grounded_desc, 0.8)],
                      ("*", "icd10"): [CandidateCode("DX_ALPHA_RIGHT", "icd10",
                                                    "condition alpha, right side", 0.9)]},
            index={"condition alpha of the right side": {"DX_ALPHA_RIGHT"}})
        primary = _declare(
            entails=lambda d: "repair" in d.lower() or "condition alpha" in d.lower(),
            provider="provider-a", reason="documented act matches")
        return run_pipeline(note, facts_json, source, verify_llm=primary,
                            capability_manifest=capability_manifest)

    def test_a_complete_injected_manifest_reaches_true_zero_hold_auto_ready(self):
        """The fixture this harness's OTHER scenarios rely on by default:
        proves `complete_capability_manifest()` alone -- no ambient real data
        files -- is sufficient for `source_manifest_gate` to PASS and the
        whole claim to reach a genuine, zero-`holds` AUTO_READY."""
        result = self._grounded_scenario(complete_capability_manifest())
        payload = build_claim_bundle_payload(result)
        self.assertEqual(payload["release"]["producer_verdict"], "AUTO_READY", payload)
        self.assertEqual(payload["release"]["holds"], [], payload)
        self.assertEqual(defensible_codes(payload), {"GROUNDED", "DX_ALPHA_RIGHT"}, payload)

    def test_a_missing_required_source_blocks_release_not_a_silent_pass(self):
        """The other outcome `source_manifest_gate` must prove: an
        absent/corrupt required source BLOCKS release outright, even though
        every OTHER fact about this encounter is identical to the clean,
        AUTO_READY scenario above -- confirms the gate has a genuine,
        reachable failure path through this harness, not just a fixture that
        happens to always pass. This is a release-level block, not a
        per-code one: `defensible_codes` reflects each line's own
        evidence-grounded disposition, a concern this gate deliberately does
        not touch, so a missing required source blocking the release is not
        expected to empty it out."""
        result = self._grounded_scenario(blocked_capability_manifest())
        payload = build_claim_bundle_payload(result)
        self.assertEqual(payload["release"]["producer_verdict"], "BLOCKED", payload)
        self.assertIn("source_manifest", payload["release"]["reason_codes"], payload)
        self.assertNotEqual(payload["release"]["holds"], [], payload)
