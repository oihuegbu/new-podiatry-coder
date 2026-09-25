"""End-to-end tests for the claude-medical-coder pipeline.

Runs the WHOLE flow (extract -> resolve -> arbitrate -> gate -> autonomy ->
certificate) with a MockSource and stubbed LLMs, so it needs no API key, no RAG
index, and — deliberately — contains NO real medical code (the mock uses
synthetic identifiers). It asserts the safety properties, not just happy paths:
planned work is not billed, negated findings are dropped, unsupported evidence
blocks release, and autonomy is granted only when the chain closes.
"""
import json
import unittest

from claude_coder.data_access import MockSource
from claude_coder.models import CandidateCode, Outcome, ResolutionMethod, Verdict
from claude_coder.pipeline import code_encounter
from tests import shortlist_verdict as _sv


def _from(fn, provider):
    """Declare which model provider a stub LLM stands in for.

    Corroboration only counts as INDEPENDENT when the second judgement comes from a
    declared, DIFFERENT provider, so a test that wants to exercise the corroborated path
    has to say who its two stubs are — the same way the deployment declares it for the
    real callables. (Round 5, phase 5.)"""
    from claude_coder.verify import declare_model_profile
    return declare_model_profile(fn, provider=provider)


def _request(fact):
    from claude_coder.eligibility import (ClaimComponent, ClaimLineIntent,
                                          EligibilityState, RetrievalRequest,
                                          fact_snapshot_digest)
    from claude_coder.models import FactKind
    if not fact.fact_id:
        fact.fact_id = "fact"
    intent = ClaimLineIntent(
        intent_id=f"test-{fact.fact_id}", encounter_id="test",
        component=(ClaimComponent.DIAGNOSIS_SUPPORT
                   if fact.kind is FactKind.DIAGNOSIS else ClaimComponent.SERVICE),
        clinical_event_ids=[fact.fact_id], fact_kind=fact.kind.value,
        clinical_action=fact.description, attributes=dict(fact.attributes),
        date_of_service=None, billing_entity_id=None, source_span_ids=[],
        state=EligibilityState.ELIGIBLE_FOR_RETRIEVAL,
        fact_digest=fact_snapshot_digest(fact))
    return RetrievalRequest(intent, fact)

# A note whose text contains, verbatim, every evidence span the extractor emits.
# Fully synthetic — the pipeline's disposition/negation logic turns on the
# LINGUISTIC markers ('denies …', 'Plan … next visit'), not any clinical term.
_LINK_SENTENCE = ("Excision of lesion alpha was performed for "
                  "condition alpha of the right side")
NOTE = (
    "Procedure: excision of lesion alpha, right site two. "
    "Assessment: condition alpha, right side. "
    + _LINK_SENTENCE + ". "
    "Patient denies finding gamma. "
    "Plan procedure beta correction next visit."
)

# What the (stubbed) CLU extractor returns: one performed procedure, one current
# diagnosis, one PLANNED procedure (must not bill), one NEGATED finding (drop).
# The procedure and the diagnosis each quote THEIR OWN phrase inside the one sentence that
# states WHY the service was done, so `provenance.reconcile_relations` can establish the
# REASON_FOR edge from the SOURCE — the document's own directional wording between the two
# verified mentions — rather than from the model's self-confidence, from repeating the edge,
# or from the two facts merely appearing near each other. (Codex F6-R3.)
_PROC_MENTION = "Excision of lesion alpha"                 # capitalised only in that sentence
_DX_MENTION = "condition alpha of the right side"


def _facts_json(*, link_evidence=True):
    """The extractor response. `link_evidence=False` drops every quote that lives in the
    linking sentence, for the case where the note never documents WHY the service was done."""
    proc_ev = ["excision of lesion alpha, right site two"]
    dx_ev = ["condition alpha, right side"]
    if link_evidence:
        proc_ev.append(_PROC_MENTION)
        dx_ev.append(_DX_MENTION)
    return json.dumps({"facts": [
        {"fact_id": "F1", "kind": "procedure", "description": "excision of lesion alpha",
         "attributes": {"laterality": "right", "anatomy": "site two",
                        "performer_id": "actor-1", "billing_entity_id": "actor-1"},
         "attribute_evidence": {
             "laterality": [{"text": proc_ev[0], "scope": "local",
                            "assertion_state": "asserted", "value": "right"}],
             "anatomy": [{"text": proc_ev[0], "scope": "local",
                         "assertion_state": "asserted", "value": "site two"}]},
         "disposition": "performed_today", "negated": False, "evidence": proc_ev,
         "confidence": 0.97,
         "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                             "temporal": 0.99, "performer": 0.99, "relationship": 0.99}},
        {"fact_id": "F2", "kind": "diagnosis",
         "description": "condition alpha of the right side",
         "attributes": {"laterality": "right"},
         "attribute_evidence": {
             "laterality": [{"text": dx_ev[0], "scope": "local",
                            "assertion_state": "asserted", "value": "right"}]},
         "disposition": "performed_today",
         "negated": False, "evidence": dx_ev, "confidence": 0.98,
         "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                             "temporal": 0.99, "assertion": 0.99, "experiencer": 0.99}},
        {"fact_id": "F3", "kind": "procedure", "description": "procedure beta correction",
         "attributes": {},
         "disposition": "planned", "negated": False,
         "evidence": ["Plan procedure beta correction next visit"], "confidence": 0.9},
        {"fact_id": "F4", "kind": "diagnosis", "description": "finding gamma",
         "attributes": {},
         "disposition": "performed_today", "negated": True,
         "evidence": ["denies finding gamma"], "confidence": 0.9},
    ], "relations": [
        {"subject_event_id": "F2", "object_event_id": "F1", "predicate": "reason_for",
         "state": "asserted", "evidence_fact_ids": ["F1", "F2"], "confidence": 0.99},
    ]})


FACTS_JSON = _facts_json()

# Synthetic (non-code) identifiers — no real medical code anywhere in this test.
# issue #6 F9-R12-E, third re-review: descriptor deliberately carries no
# cardinality word ("each"/"single"/...) -- `_needs_verification` treats
# those as a qualifier requiring an authorized documented count/quantity,
# forcing this candidate to defer to a verifier this end-to-end suite's
# own no-verifier convention doesn't supply. Not what this fixture is
# testing (evidence linking, certificates, gates), so kept out of it.
PROC = CandidateCode("PROC_ALPHA_EXC", "cpt",
                     "Excision, lesion alpha", 0.9, "retrieval")
DX = CandidateCode("DX_ALPHA_RIGHT", "icd10",
                   "condition alpha, right side", 0.9, "retrieval")


def _source():
    # issue #6 F9-R12-E, third re-review: PROC/DX must resolve through a
    # genuine DIRECT authoritative route (not bare retrieval, which is
    # verification-required with no LLM as of this round) so this whole
    # end-to-end suite keeps its deliberate "stub LLMs -> deterministic
    # path, no verifier" testing convention safely -- an unqualified exact
    # CPT/ICD Index hit is exactly the route `_take()` still trusts with no
    # LLM present.
    return MockSource(
        records={("PROC_ALPHA_EXC", "cpt"):
                 {"active": True, "long_description": "Excision, lesion alpha"},
                 ("DX_ALPHA_RIGHT", "icd10"):
                 {"active": True, "long_description": "condition alpha, right side"}},
        retrieval={("*", "cpt"): [PROC], ("*", "icd10"): [DX]},
        cpt_index={"excision of lesion alpha": {"PROC_ALPHA_EXC"}},
        index={"condition alpha of the right side": {"DX_ALPHA_RIGHT"}},
    )


def _extract_stub(system, user):
    return FACTS_JSON


def _arbitrate_stub(system, user):
    return '{"choice":0,"confidence":0.0,"reason":"unused"}'


class AutonomousCoderTest(unittest.TestCase):

    def _run(self, note=NOTE, dos="2026-03-14"):
        from claude_coder.provenance import NullAuditRepository
        return code_encounter("enc-1", note, dos, source=_source(),
                              extract_llm=_extract_stub, arbitrate_llm=_arbitrate_stub,
                              audit_repository=NullAuditRepository(),
                              billing_context={"billing_entity_id": "actor-1", "participants": [{"id": "actor-1", "type": "person", "roles": ["performer"]}]})

    def test_happy_path_auto_ready(self):
        r = self._run()
        codes = {ln.chosen.code for ln in r.billable_lines}
        self.assertEqual(codes, {"PROC_ALPHA_EXC", "DX_ALPHA_RIGHT"})
        self.assertEqual(r.verdict, Verdict.AUTO_READY, r.notes)

    def test_certificate_binds_necessity_provenance_and_stays_reproducible(self):
        """F6-R3 end-to-end: the released claim's certificate answers WHY the service was
        necessary — the claim-line diagnosis pointer plus the accepted relation's provenance
        (status, the spans that proved it, the distinct assertion origins) — and, composing
        with the phase-1 content-addressed manifest work, identical inputs still reproduce
        the same certificate hash (the origin id is content-derived, not a per-call nonce)."""
        a, b = self._run(), self._run()
        self.assertEqual(a.verdict, Verdict.AUTO_READY, a.notes)
        self.assertEqual(a.certificate["certificate_sha256"],
                         b.certificate["certificate_sha256"])
        (binding,) = a.certificate["necessity_support"]
        (support,) = binding["supports"]
        self.assertEqual(binding["procedure_code"], "PROC_ALPHA_EXC")
        self.assertEqual(support["diagnosis_code"], "DX_ALPHA_RIGHT")
        self.assertEqual(support["reconciliation_status"], "source_directional")
        self.assertEqual(support["independent_support"], 1)   # one pass = one origin
        self.assertTrue(all(support["assertion_origins"]))
        self.assertTrue(set(support["reconciliation_evidence"])
                        <= set(support["evidence_span_ids"]))

    def test_unreconciled_necessity_link_loses_autonomy(self):
        """F6-R3 end-to-end: strip the one sentence that documents the diagnosis and the
        service TOGETHER. The model still asserts a 0.99-confidence REASON_FOR edge and the
        edge still anchors, but nothing independently reconciles it — so the claim loses
        autonomy instead of being certified on the extraction model's own say-so."""
        note = NOTE.replace(_LINK_SENTENCE + ". ", "")
        facts = _facts_json(link_evidence=False)
        from claude_coder.provenance import NullAuditRepository
        r = code_encounter("enc-1", note, "2026-03-14", source=_source(),
                           extract_llm=lambda s, u: facts,
                           arbitrate_llm=_arbitrate_stub,
                           audit_repository=NullAuditRepository(),
                           billing_context={"billing_entity_id": "actor-1", "participants": [
                               {"id": "actor-1", "type": "person", "roles": ["performer"]}]})
        # issue #6 F9-R11-A, Codex's independent re-review of aff9da6: `r.
        # gates` now reflects the FINAL, post-reconciliation claim, not the
        # round that first discovered the ungrounded link -- once the
        # procedure is excluded, a FRESH `medical_necessity` gate correctly
        # reports NOT_APPLICABLE ("no procedures to justify") rather than
        # the original UNKNOWN ("no record-grounded link"), because by then
        # there genuinely are no procedures left to check. Re-running gates
        # every round (required so a gate can never certify a service whose
        # support was removed by a LATER round) means this exact wording is
        # no longer stable; what must hold is the real invariant -- the
        # necessity gate never PASSES for an ungrounded link, and the
        # procedure itself is never billable.
        nec = next(g for g in r.gates if g.name == "medical_necessity")
        self.assertNotEqual(nec.outcome, Outcome.PASS, nec.detail)
        self.assertFalse(any(ln.chosen and ln.chosen.code == "PROC_ALPHA_EXC"
                             for ln in r.billable_lines))
        self.assertNotEqual(r.verdict, Verdict.AUTO_READY)

    def test_agreeing_origins_cannot_certify_an_ungrounded_link_end_to_end(self):
        """F6-R3 round 5 end-to-end: the SAME ungrounded note, with the corroboration axis
        forced to its agreeing value. (Lowering the threshold stands in for a multi-pass
        caller: the deployed pipeline extracts once, so agreement is only ever reachable from
        a caller that runs more passes — which is exactly the hole being closed.) The edge is
        honestly recorded as multiply asserted, the grounding axis still says the record
        proves nothing, and the whole encounter still loses autonomy with no binding."""
        from claude_coder import provenance
        from claude_coder.provenance import NullAuditRepository
        note = NOTE.replace(_LINK_SENTENCE + ". ", "")
        facts = _facts_json(link_evidence=False)
        grammar = provenance.load_relation_grammar()
        saved = grammar["min_independent_assertions"]
        try:
            grammar["min_independent_assertions"] = 1
            r = code_encounter("enc-1", note, "2026-03-14", source=_source(),
                               extract_llm=lambda s, u: facts,
                               arbitrate_llm=_arbitrate_stub,
                               audit_repository=NullAuditRepository(),
                               billing_context={"billing_entity_id": "actor-1", "participants": [
                                   {"id": "actor-1", "type": "person",
                                    "roles": ["performer"]}]})
        finally:
            grammar["min_independent_assertions"] = saved
        from claude_coder.models import RelationPredicate
        reasons = [rel for rel in r.relations
                   if rel.predicate is RelationPredicate.REASON_FOR]
        self.assertTrue(reasons)
        for rel in reasons:
            self.assertEqual(rel.corroboration_status, provenance.MULTIPLY_ASSERTED)
            self.assertEqual(rel.reconciliation_status, provenance.UNRECONCILED)
            self.assertEqual(rel.reconciliation_evidence, [])
        # issue #6 F9-R11-A: see the sibling test above -- `r.gates` is the
        # FINAL, post-reconciliation state, so the exact wording is no
        # longer stable once the procedure itself is excluded.
        nec = next(g for g in r.gates if g.name == "medical_necessity")
        self.assertNotEqual(nec.outcome, Outcome.PASS, nec.detail)
        self.assertFalse(r.necessity_support)
        self.assertNotEqual(r.verdict, Verdict.AUTO_READY)
        # the certificate of the held encounter binds NO necessity support and shows both
        # axes, so the audit record says exactly what was and was not established
        self.assertEqual(r.certificate["necessity_support"], [])
        (cert_rel,) = [rel for rel in r.certificate["relations"]
                       if rel["predicate"] == "reason_for"]
        self.assertEqual(cert_rel["corroboration_status"], provenance.MULTIPLY_ASSERTED)
        self.assertEqual(cert_rel["reconciliation_status"], provenance.UNRECONCILED)
        self.assertEqual(cert_rel["reconciliation_evidence"], [])

    def test_planned_work_not_billed(self):
        r = self._run()
        billed = {ln.chosen.code for ln in r.billable_lines}
        self.assertNotIn("beta", " ".join(billed).lower())
        # the planned procedure produced no billable line at all
        self.assertEqual(len(r.billable_lines), 2)

    def test_negated_finding_dropped(self):
        r = self._run()
        descs = " ".join(ln.fact.description for ln in r.lines).lower()
        self.assertNotIn("finding gamma", descs)

    def test_resolution_is_deterministic(self):
        r = self._run()
        for ln in r.billable_lines:
            self.assertEqual(ln.method, ResolutionMethod.DETERMINISTIC, ln.rationale)

    def test_missing_evidence_blocks_release(self):
        # a note that does NOT contain the procedure's evidence span
        r = self._run(note="Assessment: condition alpha, right side.")
        ev = next(g for g in r.gates if g.name == "verbatim_evidence")
        hold = next(g for g in r.gates if g.name.startswith("eligibility_hold:"))
        self.assertEqual(hold.outcome, Outcome.BLOCKED)
        # issue #6, Codex's independent re-review (F9-R15-C): the blocked
        # procedure's gate is scoped to its own fact_id, not the whole
        # encounter.
        #
        # issue #6, Codex's independent re-review (F9-R22-B): the extractor's
        # REASON_FOR edge (diagnosis -> this now-unbillable procedure) is
        # asserted regardless of note content, but a note that never even
        # documents the procedure at all can never GROUND that edge (no
        # source text establishes the directional wording between the two)
        # -- and a relation the record never actually established must not
        # propagate a hold (previously this test asserted the OPPOSITE:
        # that the diagnosis was dependency-held onto the procedure's own
        # missing-evidence failure, which was exactly the "any REASON_FOR
        # edge regardless of grounding" over-propagation this finding fixed).
        # The diagnosis's own selection has real, independent evidence in
        # the note, so it correctly releases on its own.
        self.assertEqual(ev.outcome, Outcome.PASS)
        from claude_coder.models import Destination
        self.assertEqual(r.destination, Destination.AUTO_READY)
        self.assertEqual(r.verdict, Verdict.AUTO_READY)
        self.assertEqual([ln.chosen.code for ln in r.billable_lines], ["DX_ALPHA_RIGHT"])

    def test_a_missing_eligibility_intent_holds_only_its_own_fact(self):
        """issue #6, Codex's independent re-review (F9-R23 system-hold audit
        addendum): the `eligibility_intent:<fact_id>` gate omitted
        `affected_fact_ids`, and `autonomy.decide` explicitly reads an empty
        scope as encounter-wide -- a single fact with no produced
        eligibility intent held the WHOLE encounter, even when a different,
        independently-supported fact had nothing to do with it. F1
        (procedure) is made to have NO eligibility intent at all; F2
        (diagnosis) must still resolve and release, and F1's own hold must
        be a scoped, retryable SYSTEM_HOLD -- never a provider question or
        a coder review item."""
        from unittest.mock import patch
        from claude_coder import eligibility as _elig
        from claude_coder.models import Destination

        real_evaluate = _elig.evaluate

        def _drop_f1_intent(facts, relations, encounter_id, date_of_service, source=None):
            intents = real_evaluate(facts, relations, encounter_id, date_of_service,
                                    source=source)
            return [it for it in intents if "F1" not in it.clinical_event_ids]

        with patch("claude_coder.eligibility.evaluate", side_effect=_drop_f1_intent):
            r = self._run()

        self.assertEqual([ln.chosen.code for ln in r.billable_lines], ["DX_ALPHA_RIGHT"],
                         "F2's own resolution is independent of F1's missing intent")
        gate = next(g for g in r.gates if g.name == "eligibility_intent:F1")
        self.assertEqual(gate.affected_fact_ids, ("F1",))
        item = next(rt for rt in r.routing if rt["subject"] == "eligibility_intent:F1")
        self.assertEqual(item["destination"], Destination.SYSTEM_HOLD.value)
        self.assertNotEqual(r.destination, Destination.BLOCKED,
                            "a scoped, fact-local integrity gap must never hard-stop the "
                            "whole encounter")

    def test_missing_dos_blocks_release(self):
        r = self._run(dos=None)
        dos = next(g for g in r.gates if g.name == "date_of_service")
        self.assertEqual(dos.outcome, Outcome.BLOCKED)
        self.assertEqual(r.verdict, Verdict.BLOCKED)

    def test_certificate_is_reproducible(self):
        a = self._run().certificate["certificate_sha256"]
        b = self._run().certificate["certificate_sha256"]
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)


class OntologyResolutionTest(unittest.TestCase):
    """The deterministic resolver decides by descriptor STRUCTURE (RAG only
    supplies the pool). Synthetic codes; the ranges/laterality come from the
    descriptors — so this also proves the size-family selection needs no
    hardcoded code table."""

    def test_measurement_range_selects_leaf(self):
        """issue #6 F9-R12-E, third re-review: `SYSTEM_FOR_KIND` fixes
        SUPPLY to the "hcpcs" system these candidates need, and SUPPLY is
        (by design) never eligible for propose-then-verify -- so there is
        no verifier path available at all for this fact kind, and plain
        retrieval is now verification-required by default (Codex). The
        elimination mechanic under test (two candidates structurally
        eliminated by an out-of-range measurement) still runs -- `_decide`
        is still called over the full pool, exactly as before -- but its
        survivor cannot auto-bill with no verifier available to confirm
        it, so it correctly surfaces as the sole remaining ALTERNATIVE on
        an honest abstention rather than closing DETERMINISTIC."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, Disposition, EvidenceSpan,
                                         FactKind, ResolutionMethod)
        from claude_coder.resolution import resolve

        small = CandidateCode("SUP_SMALL", "hcpcs",
                              "Wound dressing, sterile, size 16 sq. in. or less, each", 0.9)
        med = CandidateCode("SUP_MED", "hcpcs",
                            "Wound dressing, sterile, size more than 16 sq. in. but less "
                            "than or equal to 48 sq. in., each", 0.9)
        large = CandidateCode("SUP_LARGE", "hcpcs",
                              "Wound dressing, sterile, size more than 48 sq. in., each", 0.9)
        src = MockSource(retrieval={("*", "hcpcs"): [small, med, large]})
        fact = ClinicalFact(kind=FactKind.SUPPLY, description="wound dressing",
                            attributes={"size_sqin": 30}, disposition=Disposition.PERFORMED,
                            evidence=[EvidenceSpan("wound dressing 30 sq in applied")],
                            confidence=0.99)
        line = resolve(_request(fact), src)
        self.assertNotEqual(line.method, ResolutionMethod.DETERMINISTIC, line.rationale)
        self.assertIsNone(line.chosen, line.rationale)
        alt_codes = {c.code for c in (line.alternatives or [])}
        self.assertEqual(alt_codes, {"SUP_MED"}, line.rationale)   # eliminated to the one leaf

    def test_laterality_contradiction_eliminated(self):
        """issue #6 F9-R12-E, third re-review: see
        test_measurement_range_selects_leaf above -- same reasoning, a stub
        verifier replaces the removed no-verifier retrieval shortcut."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import (AttributeEvidence, ClinicalFact, EvidenceSpan,
                                         FactKind, RelationState)
        from claude_coder.resolution import resolve

        left = CandidateCode("DX_LEFT", "icd10", "some condition, left foot", 0.9)
        right = CandidateCode("DX_RIGHT", "icd10", "some condition, right foot", 0.9)
        src = MockSource(retrieval={("*", "icd10"): [left, right]})
        span = EvidenceSpan("some condition, right side", anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="some condition",
                            attributes={"laterality": "right"},
                            evidence=[span],
                            attribute_evidence={"laterality": (
                                AttributeEvidence(span=span, assertion_state=RelationState.ASSERTED,
                                                  value="right"),)},
                            confidence=0.99)
        llm = _sv.judge(entails=lambda d: "right" in d.lower(), reason="entailed")
        line = resolve(_request(fact), src, llm=_from(llm, "provider-a"))
        self.assertEqual(line.chosen.code, "DX_RIGHT", line.rationale)

    def test_unbound_lexical_text_never_authorizes_a_deterministic_selection(self):
        """issue #6 F9-R7-A, Codex's independent re-review of 92f4596: the exact
        reported reproduction, through the ORDINARY deterministic path (not a
        direct `claim_authorized_value` unit test) -- no `attribute_evidence` at
        all, and evidence text whose far-distance negation a lexical heuristic
        cannot correctly parse. Must never resolve deterministically to either
        side; the claim is unauthorized, not merely ambiguous between two
        equally-plausible readings."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder.resolution import resolve

        left = CandidateCode("DX_LEFT", "icd10", "some condition, left foot", 0.9)
        right = CandidateCode("DX_RIGHT", "icd10", "some condition, right foot", 0.9)
        src = MockSource(retrieval={("*", "icd10"): [left, right]})
        fact = ClinicalFact(
            kind=FactKind.DIAGNOSIS, description="some condition",
            attributes={"laterality": "right"},
            evidence=[EvidenceSpan(
                "right involvement was considered but was ultimately ruled out",
                anchored=True, span_id="s1")],
            confidence=0.99)
        line = resolve(_request(fact), src)
        self.assertNotEqual(line.chosen.code if line.chosen else None, "DX_RIGHT",
                            "an unauthorized laterality claim must never select "
                            "the very side its own confirmed text rules out")


class BundlingExclusionTest(unittest.TestCase):
    """A resolved code the source declares NOT separately reportable is dropped
    from the claim (agnostic — driven by the source's separately_billable, not a
    named code) while remaining in the audit trail."""

    def test_non_separately_billable_code_excluded(self):
        """issue #6 F9-R12-E, third re-review: this test exercises the
        PIPELINE'S bundling-exclusion stage, which needs the candidate to
        actually resolve first -- a genuine, unqualified descriptor-index
        hit (safe with no verifier, per Codex's regression #4) replaces
        the bare retrieval candidate, which is now verification-required
        and would abstain before bundling exclusion ever got a chance to
        run."""
        from claude_coder.data_access import MockSource
        from claude_coder.pipeline import code_encounter

        cand = CandidateCode("BUNDLED_X", "hcpcs", "bundled add-on service", 0.9)
        src = MockSource(records={("BUNDLED_X", "hcpcs"): {"active": True,
                                                           "long_description": "bundled add-on service"}},
                         retrieval={("*", "hcpcs"): [cand]},
                         proc_index={"bundled service": {"BUNDLED_X"}},
                         nonbillable={"BUNDLED_X"})
        facts = ('{"facts":[{"fact_id":"F1","kind":"supply",'
                 '"description":"bundled service",'
                 '"attributes":{"performer_id":"actor-1","billing_entity_id":"actor-1"},'
                 '"disposition":"performed_today","negated":false,'
                 '"evidence":["bundled service provided"],"confidence":0.99}]}')
        r = code_encounter("e", "bundled service provided during the visit",
                           "2026-03-14", source=src,
                           extract_llm=lambda s, u: facts,
                           arbitrate_llm=lambda s, u: '{"choice":0,"confidence":0}',
                           audit_repository=__import__("claude_coder.provenance",
                               fromlist=["NullAuditRepository"]).NullAuditRepository())
        self.assertNotIn("BUNDLED_X", {ln.chosen.code for ln in r.billable_lines})
        self.assertTrue(any(ln.excluded_reason for ln in r.lines))


class ModifierTest(unittest.TestCase):
    """Modifiers are discovered from data by descriptor and applied from the
    documented laterality — no modifier literal in the engine."""

    def _fact(self, laterality):
        from claude_coder.models import (AttributeEvidence, ClinicalFact, EvidenceSpan,
                                         FactKind, RelationState)
        span = EvidenceSpan("x", anchored=True, span_id="s1")
        return ClinicalFact(kind=FactKind.PROCEDURE, description="excision",
                            attributes={"laterality": laterality},
                            evidence=[span],
                            attribute_evidence={"laterality": (
                                AttributeEvidence(span=span, assertion_state=RelationState.ASSERTED,
                                                  value=laterality),)})

    def test_laterality_and_bilateral_from_data(self):
        from claude_coder.modifiers import ModifierEngine
        defs = {"MR": {"description": "Right side of the body"},
                "ML": {"description": "Left side of the body"},
                "MB": {"description": "Bilateral procedure"}}
        eng = ModifierEngine(defs=defs)
        # A side/bilateral modifier is asserted ONLY for a code the fee schedule marks
        # laterality-eligible (a bilateral-surgery indicator is present); pass a real one.
        self.assertEqual(eng.assign(self._fact("right"), "Excision, lesion, each", bilat="0"), ["MR"])
        self.assertEqual(eng.assign(self._fact("left"), "Excision, lesion, each", bilat="0"), ["ML"])
        self.assertEqual(eng.assign(self._fact("bilateral"), "Excision, lesion", bilat="1"), ["MB"])
        # descriptor already encodes the side -> no modifier (no double-coding)
        self.assertEqual(eng.assign(self._fact("right"), "Excision, lesion, right side", bilat="0"), [])
        # a code with NO bilateral indicator -- a consumed supply/implant/drug billed as
        # a device/supply code (not on the PFS) -- never earns a side modifier. This is
        # the agnostic rule behind the A4570-RT / C1713-RT error class: laterality
        # requires positive fee-schedule eligibility, never a default.
        self.assertEqual(eng.assign(self._fact("right"), "Anchor/screw for bone", bilat=None), [])

    def test_unbound_lexical_text_never_authorizes_a_modifier(self):
        """issue #6 F9-R7-A: the exact reported far-distance-negation
        reproduction, through modifier assignment -- no `attribute_evidence`,
        evidence text a lexical heuristic could misread as confirming
        laterality. Must never fabricate RT/LT from an unauthorized claim."""
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder.modifiers import ModifierEngine
        fact = ClinicalFact(
            kind=FactKind.PROCEDURE, description="excision", attributes={"laterality": "right"},
            evidence=[EvidenceSpan(
                "right involvement was considered but was ultimately ruled out",
                anchored=True, span_id="s1")])
        defs = {"MR": {"description": "Right side of the body"},
                "ML": {"description": "Left side of the body"}}
        eng = ModifierEngine(defs=defs)
        self.assertEqual(eng.assign(fact, "Excision, lesion, each", bilat="0"), [])


class EMLevelingTest(unittest.TestCase):
    def test_mdm_two_of_three(self):
        from claude_coder.em import mdm_level
        self.assertEqual(mdm_level("moderate", "moderate", "low"), "moderate")
        self.assertEqual(mdm_level("high", "low", "low"), "low")
        self.assertEqual(mdm_level("high", "high", "low"), "high")
        self.assertIsNone(mdm_level("moderate", "", "low"))   # incomplete -> review

    def test_resolve_em_picks_level_and_setting(self):
        from claude_coder.data_access import MockSource
        from claude_coder.em import resolve_em
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        low = CandidateCode("EM_LOW", "cpt",
                            "Office visit, established patient, low level medical decision making", 0.9)
        mod = CandidateCode("EM_MOD", "cpt",
                            "Office visit, established patient, moderate level medical decision making", 0.9)
        src = MockSource(retrieval={("*", "cpt"): [low, mod]})
        fact = ClinicalFact(kind=FactKind.EM, description="office visit",
                            attributes={"problems": "moderate", "data": "moderate",
                                        "risk": "low", "new_patient": False},
                            evidence=[EvidenceSpan("office visit")], confidence=0.9)
        line = resolve_em(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "EM_MOD", line.rationale)


class UnitsTest(unittest.TestCase):
    def test_billing_units_from_descriptor(self):
        from claude_coder.ontology import billing_units
        # a "2-4 lesions" code is ONE unit for 2 lesions (the bug this fixes)
        self.assertEqual(billing_units(2, "Paring, 2 to 4 lesions"), 1)
        # an "each" code bills per item
        self.assertEqual(billing_units(3, "Excision of lesion, single, each"), 3)
        self.assertEqual(billing_units(1, "Some procedure"), 1)


class ClaimModifierTest(unittest.TestCase):
    def test_em25_and_distinct_service_from_data(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (AttributeEvidence, ClinicalFact, CodingResult,
                                         EvidenceSpan, FactKind, RelationState,
                                         ResolutionMethod, ResolvedLine)
        from claude_coder.modifiers import ModifierEngine
        defs = {"M25": {"description": "significant, separately identifiable evaluation and management"},
                "MXS": {"description": "Separate Structure"}}
        eng = ModifierEngine(defs=defs)

        def line(code, kind, lat=None):
            span = EvidenceSpan("x", anchored=True, span_id=f"s-{code}")
            attr_ev = ({"laterality": (AttributeEvidence(
                            span=span, assertion_state=RelationState.ASSERTED, value=lat),)}
                       if lat else {})
            f = ClinicalFact(kind=kind, description="x",
                             attributes=({"laterality": lat} if lat else {}),
                             evidence=[span], attribute_evidence=attr_ev)
            return ResolvedLine(fact=f, chosen=CandidateCode(code, "cpt", "d", 0.9),
                                method=ResolutionMethod.DETERMINISTIC)
        p1 = line("P1", FactKind.PROCEDURE, "right")
        p2 = line("P2", FactKind.PROCEDURE, "left")
        emln = line("EMX", FactKind.EM)
        emln.fact.attributes["separately_identifiable"] = True   # 25 only when documented
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[p1, p2, emln])
        src = MockSource(ncci={("P1", "P2"): "1", ("P2", "P1"): "1"})
        eng.assign_claim(r, src)
        self.assertIn("M25", emln.modifiers)                       # E/M-25 with a procedure
        self.assertTrue("MXS" in p1.modifiers or "MXS" in p2.modifiers)  # distinct structure
        self.assertTrue(r.bypassed_ncci)                          # bypass recorded for the gate

    def test_assign_claim_is_idempotent_it_clears_its_own_prior_output(self):
        """issue #6 F9-R9-B, Codex's independent re-review of 6ff2761:
        `assign_claim` must be safely re-runnable after the claim set changes
        -- it used to only ever APPEND, so a second call on a narrower
        surviving set left a stale distinct-service modifier and a stale
        `bypassed_ncci` entry for a partner that is no longer even on the
        claim. It must now clear its own prior output (the exact modifier
        VALUES it owns, and `bypassed_ncci` entirely) before recomputing."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import (AttributeEvidence, ClinicalFact, CodingResult,
                                         EvidenceSpan, FactKind, RelationState,
                                         ResolutionMethod, ResolvedLine)
        from claude_coder.modifiers import ModifierEngine
        defs = {"MXS": {"description": "Separate Structure"}}
        eng = ModifierEngine(defs=defs)

        def line(code, lat):
            span = EvidenceSpan("x", anchored=True, span_id=f"s-{code}")
            attr_ev = {"laterality": (AttributeEvidence(
                span=span, assertion_state=RelationState.ASSERTED, value=lat),)}
            f = ClinicalFact(kind=FactKind.PROCEDURE, description="x",
                             attributes={"laterality": lat},
                             evidence=[span], attribute_evidence=attr_ev, fact_id=code)
            return ResolvedLine(fact=f, chosen=CandidateCode(code, "cpt", "d", 0.9),
                                method=ResolutionMethod.DETERMINISTIC)
        p1 = line("P1", "right")
        p2 = line("P2", "left")
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[p1, p2])
        src = MockSource(ncci={("P1", "P2"): "1", ("P2", "P1"): "1"})
        eng.assign_claim(r, src)
        self.assertIn("MXS", p2.modifiers)
        self.assertTrue(r.bypassed_ncci)

        # P1 is no longer on the claim (simulating exclusion by autonomy.decide's
        # entanglement pruning, which runs AFTER this method the first time).
        p1.excluded_reason = "excluded for an unrelated reason"
        eng.assign_claim(r, src)
        self.assertNotIn("MXS", p2.modifiers, "the stale modifier must be cleared, "
                         "not left over from a partner that is no longer billed")
        self.assertEqual(r.bypassed_ncci, [], "a bypass for a code no longer on "
                         "the claim must not still be recorded")


def _combined_intent(*clinical_event_ids, encounter_id="e", dos="2026-03-14"):
    """issue #6, Codex's independent re-review (F9-R22-B): a real
    `eligibility.ClaimLineIntent` composing `clinical_event_ids` into ONE
    code-determining claim line -- the only thing (besides a grounded,
    directional `REASON_FOR` edge) that may now let `autonomy._entangled`
    treat a documented compositional relationship (e.g. "ancillary
    component of procedure one") as claim-affecting. A bare `PART_OF` edge
    is no longer its own propagation path."""
    from claude_coder import eligibility as _elig
    return _elig.ClaimLineIntent(
        intent_id=f"intent-{'-'.join(clinical_event_ids)}", encounter_id=encounter_id,
        component=_elig.ClaimComponent.SERVICE, clinical_event_ids=list(clinical_event_ids),
        fact_kind="procedure", clinical_action="", attributes={}, date_of_service=dos,
        billing_entity_id=None, source_span_ids=[],
        state=_elig.EligibilityState.ELIGIBLE_FOR_RETRIEVAL)


class ClaimAfterPruningReconciliationTest(unittest.TestCase):
    """issue #6 F9-R9-B, Codex's independent re-review of 6ff2761:
    `pipeline._reconcile_claim_after_pruning` re-derives modifiers/NCCI/
    integral/global-package/gates from the surviving line set, repeating
    until `autonomy.decide`'s graph/necessity pruning stops changing it.
    Reproduced directly before the fix: two procedures earned a distinct-
    service modifier from their NCCI pair; a third, unresolved fact PART_OF
    the first procedure caused `decide` to exclude it -- the SECOND
    procedure kept its now-unjustified modifier and `bypassed_ncci` still
    named a code that was no longer even on the claim."""

    def test_a_stale_distinct_service_modifier_is_removed_after_pruning(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (AttributeEvidence, ClaimSubmissionStatus,
                                         ClinicalFact, CodingResult,
                                         EvidenceSpan, FactKind, RelationAssertion,
                                         RelationPredicate, RelationState,
                                         ResolutionMethod, ResolvedLine)
        from claude_coder.modifiers import ModifierEngine
        from claude_coder import eligibility, graph as graph_mod
        from claude_coder.pipeline import _reconcile_claim_after_pruning

        defs = {"MXS": {"description": "Separate Structure"}}
        eng = ModifierEngine(defs=defs)

        def line(fid, code, kind, text, lat=None, resolved=True):
            span = EvidenceSpan(text, anchored=True, span_id=f"s-{fid}")
            attr_ev = ({"laterality": (AttributeEvidence(
                            span=span, assertion_state=RelationState.ASSERTED, value=lat),)}
                       if lat else {})
            base_attrs = {"laterality": lat} if lat else {}
            if kind is not FactKind.DIAGNOSIS:
                base_attrs.update({"performer_id": "person-1",
                                  "billing_entity_id": "person-1"})
            f = ClinicalFact(kind=kind, description=text, attributes=base_attrs,
                             evidence=[span], attribute_evidence=attr_ev, fact_id=fid)
            system = "icd10" if kind is FactKind.DIAGNOSIS else "cpt"
            if resolved:
                return ResolvedLine(fact=f, chosen=CandidateCode(code, system, "d", 0.9),
                                    method=ResolutionMethod.DETERMINISTIC)
            return ResolvedLine(fact=f, chosen=None, method=ResolutionMethod.ABSTAINED,
                                alternatives=[CandidateCode("ALT1", "cpt", "alt", 0.5)],
                                documentation_gap="which variant?")

        p1 = line("P1", "P1CODE", FactKind.PROCEDURE, "procedure one right", "right")
        p2 = line("P2", "P2CODE", FactKind.PROCEDURE, "procedure two left", "left")
        dx = line("DX", "DXCODE", FactKind.DIAGNOSIS, "diagnosis for both")
        f3 = line("F3", None, FactKind.PROCEDURE,
                 "ancillary component of procedure one", resolved=False)

        rel_part = RelationAssertion(
            subject_event_id="F3", predicate=RelationPredicate.PART_OF,
            object_event_id="P1", state=RelationState.ASSERTED,
            evidence_span_ids=["s-P1", "s-F3"])
        rel_dx1 = RelationAssertion(
            subject_event_id="DX", predicate=RelationPredicate.REASON_FOR,
            object_event_id="P1", state=RelationState.ASSERTED,
            evidence_span_ids=["s-DX", "s-P1"], confidence=0.95,
            reconciliation_status="source_directional", reconciliation_evidence=["s-P1"])
        rel_dx2 = RelationAssertion(
            subject_event_id="DX", predicate=RelationPredicate.REASON_FOR,
            object_event_id="P2", state=RelationState.ASSERTED,
            evidence_span_ids=["s-DX", "s-P2"], confidence=0.95,
            reconciliation_status="source_directional", reconciliation_evidence=["s-P2"])
        relations = [rel_part, rel_dx1, rel_dx2]
        facts = [p1.fact, p2.fact, dx.fact, f3.fact]
        intents = eligibility.evaluate(facts, relations, "e", "2026-03-14")
        episodes, _ = eligibility.build_episodes(facts, relations, "e", "2026-03-14")
        compiled = graph_mod.build_graph(
            facts, relations, intents, encounter_id="e", date_of_service="2026-03-14",
            episodes=episodes, extraction_schema_version="v1", relation_grammar_version="v1")

        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[p1, p2, dx, f3], relations=relations, graph=compiled,
                         claim_line_intents=list(intents) + [_combined_intent("F3", "P1")])
        src = MockSource(
            records={("P1CODE", "cpt"): {"active": True}, ("P2CODE", "cpt"): {"active": True},
                    ("DXCODE", "icd10"): {"active": True}},
            ncci={("P1CODE", "P2CODE"): "1", ("P2CODE", "P1CODE"): "1"})
        note_text = ("procedure one right procedure two left diagnosis for both "
                    "ancillary component of procedure one")

        eng.assign_claim(r, src)
        self.assertIn("MXS", p2.modifiers, "sanity: the pair earns the modifier "
                      "before any pruning")

        _reconcile_claim_after_pruning(r, src, note_text, eng, None, [], None)

        # issue #6, Codex's independent re-review (F9-R20-A clarification):
        # "downstream controls classify; they do not erase" -- entanglement
        # with unresolved F3 is a SUBMISSION problem, never proof P1's own
        # selection is invalid, so P1 is HELD (code/evidence intact, visible
        # via `submission_held_lines`), not erased via `excluded_reason`.
        self.assertIsNone(p1.excluded_reason,
                          "P1's own selection is not invalid -- entanglement "
                          "holds submission, it does not erase the selection")
        self.assertEqual(p1.claim_submission_status, ClaimSubmissionStatus.HELD,
                         "P1 must be held -- entangled with unresolved F3")
        self.assertIsNone(p2.excluded_reason,
                          "P2 has its own independent diagnosis link and must survive")
        self.assertNotIn("MXS", p2.modifiers,
                         "the modifier justified only by the now-excluded P1 must "
                         "not survive the recompute")
        self.assertEqual(r.bypassed_ncci, [],
                         "no NCCI pair remains -- P1 is gone -- so nothing should "
                         "still be recorded as bypassed")
        self.assertEqual(sorted(ln.chosen.code for ln in r.billable_lines),
                         ["DXCODE", "P2CODE"])
        ncci_gate = next(g for g in r.gates if g.name == "ncci_ptp")
        self.assertIn(ncci_gate.outcome.value, ("NOT_APPLICABLE", "PASS"),
                     "with only one surviving procedure there is no PTP pair left "
                     "to leave unresolved")

    def test_non_convergence_fails_loud_instead_of_returning_a_moving_claim(self):
        """The monotonic-GROWTH invariant the loop's termination proof
        depends on -- `decide`'s own `dependency_excluded_fact_ids` can only
        ever name a fact already in the accumulated set, once truly
        converged -- is enforced defensively: if it is ever violated (decide
        keeps finding a fact outside the accumulated set, which should be
        impossible given there are only finitely many facts), the loop must
        fail LOUD rather than silently return a claim that might still be
        changing."""
        from unittest.mock import patch
        from claude_coder.data_access import MockSource
        from claude_coder.models import (CandidateCode, ClinicalFact, CodingResult,
                                         EvidenceSpan, FactKind, ResolutionMethod,
                                         ResolvedLine)
        from claude_coder.modifiers import ModifierEngine
        from claude_coder.pipeline import _reconcile_claim_after_pruning

        fact = ClinicalFact(FactKind.PROCEDURE, "x",
                            evidence=[EvidenceSpan("x", anchored=True, span_id="s1")],
                            confidence=0.9, fact_id="F1")
        line = ResolvedLine(fact=fact, chosen=CandidateCode("C1", "cpt", "d", 0.9),
                            method=ResolutionMethod.DETERMINISTIC)
        result = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[line])
        eng = ModifierEngine(defs={})
        src = MockSource(records={("C1", "cpt"): {"active": True}})

        # A fake `decide` that reports a BRAND-NEW fact_id every single call
        # -- deliberately violating the invariant no real `decide` call
        # could (there are only finitely many facts, so it must eventually
        # report only ids already accumulated), to prove the safety net
        # actually fires.
        calls = {"n": 0}
        def never_converging_decide(res, source=None):
            calls["n"] += 1
            res.dependency_excluded_fact_ids = frozenset({f"NEVER_SEEN_{calls['n']}"})

        with patch("claude_coder.pipeline.decide", side_effect=never_converging_decide):
            with self.assertRaises(RuntimeError):
                _reconcile_claim_after_pruning(result, src, "x", eng, None, [], None)

    def test_a_stale_gate_does_not_survive_removal_of_its_own_support(self):
        """issue #6 F9-R11-A, Codex's independent re-review of aff9da6: an
        earlier version of this fix computed gates exactly ONCE, before the
        loop -- so `medical_necessity`'s own PASS (and the `necessity_
        support` binding it writes) survived as a stale record even after a
        LATER round's dependency pruning removed the diagnosis that earned
        it. Reproduced directly before this fix: a resolved procedure P with
        its own resolved, grounded diagnosis DX; DX is itself `PART_OF` an
        unresolved diagnosis U (same episode) -- pruning excludes DX, but
        the gate computed before that pruning still said PASS, and P stayed
        billable with no real support. Gates must be recomputed every round
        so the FINAL state is what's attested, never a frozen earlier one."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClaimSubmissionStatus, ClinicalFact,
                                         CodingResult, Destination,
                                         EvidenceSpan, FactKind, RelationAssertion,
                                         RelationPredicate, RelationState,
                                         ResolutionMethod, ResolvedLine, CandidateCode)
        from claude_coder import eligibility, graph as graph_mod
        from claude_coder.pipeline import _reconcile_claim_after_pruning
        from claude_coder.modifiers import ModifierEngine

        def line(fid, code, kind, text, resolved=True):
            span = EvidenceSpan(text, anchored=True, span_id=f"s-{fid}")
            f = ClinicalFact(kind=kind, description=text, evidence=[span], fact_id=fid)
            if resolved:
                system = "icd10" if kind is FactKind.DIAGNOSIS else "cpt"
                return ResolvedLine(fact=f, chosen=CandidateCode(code, system, "d", 0.9),
                                    method=ResolutionMethod.DETERMINISTIC)
            return ResolvedLine(fact=f, chosen=None, method=ResolutionMethod.ABSTAINED,
                                alternatives=[CandidateCode("ALT1", "icd10", "alt", 0.5)],
                                documentation_gap="which variant?")

        p = line("P", "P_CODE", FactKind.PROCEDURE, "procedure performed")
        dx = line("DX", "DX_CODE", FactKind.DIAGNOSIS, "diagnosis documented")
        u = line("U", None, FactKind.DIAGNOSIS, "unresolved related diagnosis",
                resolved=False)
        rel_necessity = RelationAssertion(
            subject_event_id="DX", predicate=RelationPredicate.REASON_FOR,
            object_event_id="P", state=RelationState.ASSERTED,
            evidence_span_ids=["s-DX", "s-P"], confidence=0.95,
            reconciliation_status="source_directional", reconciliation_evidence=["s-P"])
        rel_episode = RelationAssertion(
            subject_event_id="U", predicate=RelationPredicate.PART_OF,
            object_event_id="DX", state=RelationState.ASSERTED,
            evidence_span_ids=["s-U", "s-DX"])
        relations = [rel_necessity, rel_episode]
        facts = [p.fact, dx.fact, u.fact]
        intents = eligibility.evaluate(facts, relations, "e", "2026-03-14")
        episodes, _ = eligibility.build_episodes(facts, relations, "e", "2026-03-14")
        compiled = graph_mod.build_graph(
            facts, relations, intents, encounter_id="e", date_of_service="2026-03-14",
            episodes=episodes, extraction_schema_version="v1", relation_grammar_version="v1")

        result = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                              lines=[p, dx, u], relations=relations, graph=compiled,
                              claim_line_intents=list(intents) + [_combined_intent("DX", "U")])
        src = MockSource(records={("P_CODE", "cpt"): {"active": True},
                                  ("DX_CODE", "icd10"): {"active": True}})
        eng = ModifierEngine(defs={})
        note_text = "procedure performed diagnosis documented unresolved related diagnosis"

        _reconcile_claim_after_pruning(result, src, note_text, eng, None, [], None)

        # issue #6, Codex's independent re-review (F9-R20-A clarification):
        # DX's own selection is unaffected by U's ambiguity -- it is HELD,
        # not erased via `excluded_reason`. It still drops out of
        # `billable_lines`/`diagnosis_lines` exactly as an excluded line
        # would, so P's necessity support is removed all the same.
        self.assertIsNone(dx.excluded_reason,
                          "DX's own selection is not invalid -- entanglement "
                          "holds submission, it does not erase the selection")
        self.assertEqual(dx.claim_submission_status, ClaimSubmissionStatus.HELD,
                         "DX must be held -- entangled with unresolved U")
        # `billable_lines` alone (resolved + not excluded) does not decide
        # submission -- the necessity gate going hard-BLOCKED (no documented
        # diagnosis remains for P) is what must, and does, force the WHOLE
        # encounter's destination/verdict away from anything that could ever
        # reach AUTO_READY, regardless of what else `billable_lines` lists.
        nec = next(g for g in result.gates if g.name == "medical_necessity")
        self.assertNotEqual(nec.outcome, Outcome.PASS,
                            "the FINAL gate must reflect P's support having been "
                            "removed, never a frozen earlier PASS")
        self.assertFalse(result.necessity_support,
                         "no binding should remain for a procedure with no support")
        self.assertNotEqual(result.verdict, Verdict.AUTO_READY)
        self.assertNotEqual(result.destination, Destination.AUTO_READY)

    def test_an_ncci_component_returns_when_its_primary_is_pruned(self):
        """issue #6 F9-R11-B, Codex's independent re-review of aff9da6: a
        line excluded by a claim-set mechanic (here, NCCI bundling) was
        treated as PERMANENTLY excluded, even after the ONLY reason for that
        exclusion (its primary/payable partner) was later removed by
        dependency pruning. Reproduced directly before this fix: A and B
        have an unresolved (indicator '0') NCCI conflict, so B is demoted
        into A; an unresolved fact U is `PART_OF` A, so dependency pruning
        excludes A -- but B stayed excluded forever, even though no
        conflicting pair remained. The claim-set-mechanic state must be
        restored to a pristine baseline and re-derived fresh every round,
        not assumed monotonic."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClaimSubmissionStatus, ClinicalFact,
                                         CodingResult, EvidenceSpan,
                                         FactKind, RelationAssertion, RelationPredicate,
                                         RelationState, ResolutionMethod, ResolvedLine,
                                         CandidateCode)
        from claude_coder import eligibility, graph as graph_mod
        from claude_coder.pipeline import (_reconcile_claim_after_pruning,
                                           _snapshot_pre_claim_set_state,
                                           apply_ncci_bundling, apply_integral_bundling,
                                           apply_global_package)
        from claude_coder.modifiers import ModifierEngine

        def line(fid, code, kind, text, attrs=None, resolved=True):
            span = EvidenceSpan(text, anchored=True, span_id=f"s-{fid}")
            f = ClinicalFact(kind=kind, description=text, evidence=[span], fact_id=fid,
                             attributes=(attrs or {}))
            if resolved:
                system = "icd10" if kind is FactKind.DIAGNOSIS else "cpt"
                return ResolvedLine(fact=f, chosen=CandidateCode(code, system, "d", 0.9),
                                    method=ResolutionMethod.DETERMINISTIC)
            return ResolvedLine(fact=f, chosen=None, method=ResolutionMethod.ABSTAINED,
                                alternatives=[CandidateCode("ALT1", "cpt", "alt", 0.5)],
                                documentation_gap="which variant?")

        a = line("A", "A_CODE", FactKind.PROCEDURE, "procedure A performed",
                {"anatomy": "site-a", "performer_id": "p1", "billing_entity_id": "p1"})
        b = line("B", "B_CODE", FactKind.PROCEDURE, "procedure B performed",
                {"anatomy": "site-b", "performer_id": "p1", "billing_entity_id": "p1"})
        dxa = line("DXA", "DXA_CODE", FactKind.DIAGNOSIS, "diagnosis for A")
        dxb = line("DXB", "DXB_CODE", FactKind.DIAGNOSIS, "diagnosis for B")
        u = line("U", None, FactKind.PROCEDURE, "ancillary component of procedure A",
                resolved=False)

        rel_part = RelationAssertion(
            subject_event_id="U", predicate=RelationPredicate.PART_OF,
            object_event_id="A", state=RelationState.ASSERTED,
            evidence_span_ids=["s-U", "s-A"])
        rel_dxa = RelationAssertion(
            subject_event_id="DXA", predicate=RelationPredicate.REASON_FOR,
            object_event_id="A", state=RelationState.ASSERTED,
            evidence_span_ids=["s-DXA", "s-A"], confidence=0.95,
            reconciliation_status="source_directional", reconciliation_evidence=["s-A"])
        rel_dxb = RelationAssertion(
            subject_event_id="DXB", predicate=RelationPredicate.REASON_FOR,
            object_event_id="B", state=RelationState.ASSERTED,
            evidence_span_ids=["s-DXB", "s-B"], confidence=0.95,
            reconciliation_status="source_directional", reconciliation_evidence=["s-B"])
        relations = [rel_part, rel_dxa, rel_dxb]
        facts = [a.fact, b.fact, dxa.fact, dxb.fact, u.fact]
        intents = eligibility.evaluate(facts, relations, "e", "2026-03-14")
        episodes, _ = eligibility.build_episodes(facts, relations, "e", "2026-03-14")
        compiled = graph_mod.build_graph(
            facts, relations, intents, encounter_id="e", date_of_service="2026-03-14",
            episodes=episodes, extraction_schema_version="v1", relation_grammar_version="v1")

        result = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                              lines=[a, b, dxa, dxb, u], relations=relations,
                              graph=compiled,
                              claim_line_intents=list(intents) + [_combined_intent("U", "A")])
        src = MockSource(
            records={("A_CODE", "cpt"): {"active": True}, ("B_CODE", "cpt"): {"active": True},
                    ("DXA_CODE", "icd10"): {"active": True},
                    ("DXB_CODE", "icd10"): {"active": True}},
            ncci={("A_CODE", "B_CODE"): "0"})
        eng = ModifierEngine(defs={})
        note_text = ("procedure A performed procedure B performed diagnosis for A "
                    "diagnosis for B ancillary component of procedure A")

        # Baseline MUST be taken before the mechanics' one-time pre-escalation
        # pass (issue #6 F9-R11-B) -- exactly as `code_encounter` now does --
        # or it would capture B already demoted and never let it return.
        baseline = _snapshot_pre_claim_set_state(result)
        eng.assign_claim(result, src)
        apply_ncci_bundling(result, src)
        apply_integral_bundling(result, src)
        apply_global_package(result, src)
        self.assertIsNotNone(b.excluded_reason,
                             "sanity: B is demoted by the unresolved NCCI conflict "
                             "before any pruning")
        # issue #6 F9-R11-D, Codex's independent re-review of 42f2b45: this is
        # the exact set `code_encounter` used to compute `billed_span_ids`
        # from, BEFORE reconciliation ran -- B's evidence span is missing,
        # so an independent reader would never be asked to verify B's page
        # even though B ends up billable. `code_encounter` now runs
        # reconciliation BEFORE computing this set.
        pre_reconciliation_span_ids = {s.span_id for ln in result.billable_lines
                                       for s in (ln.fact.evidence or []) if s.span_id}
        self.assertNotIn("s-B", pre_reconciliation_span_ids,
                         "sanity: B's span is not yet targetable before reconciliation")

        _reconcile_claim_after_pruning(result, src, note_text, eng, None, [], None,
                                       baseline=baseline)

        # issue #6, Codex's independent re-review (F9-R20-A clarification):
        # A's own selection is unaffected by U's ambiguity -- it is HELD,
        # not erased via `excluded_reason`. It still drops out of
        # `billable_lines` exactly as an excluded line would, so B's NCCI
        # partner is gone all the same and B correctly returns.
        self.assertIsNone(a.excluded_reason,
                          "A's own selection is not invalid -- entanglement "
                          "holds submission, it does not erase the selection")
        self.assertEqual(a.claim_submission_status, ClaimSubmissionStatus.HELD,
                         "A must be held -- entangled with unresolved U")
        self.assertIsNone(b.excluded_reason,
                          "B's only NCCI partner is gone -- it must return")
        self.assertEqual(sorted(ln.chosen.code for ln in result.billable_lines),
                         ["B_CODE", "DXA_CODE", "DXB_CODE"])
        post_reconciliation_span_ids = {s.span_id for ln in result.billable_lines
                                        for s in (ln.fact.evidence or []) if s.span_id}
        self.assertIn("s-B", post_reconciliation_span_ids,
                      "B's span must be targetable for independent verification "
                      "once reconciliation has run -- this is exactly what "
                      "escalation page-targeting must see")


class GlobalPackageTest(unittest.TestCase):
    """A same-day E/M is bundled into a procedure's CMS global package unless the
    note documents separately-identifiable E/M work."""

    def _lines(self, sep_ident):
        from claude_coder.models import (ClinicalFact, CodingResult, EvidenceSpan,
                                         FactKind, ResolutionMethod, ResolvedLine)

        def line(code, kind, attrs):
            f = ClinicalFact(kind=kind, description="x", attributes=attrs,
                             evidence=[EvidenceSpan("x")])
            return ResolvedLine(fact=f, chosen=CandidateCode(code, "cpt", "d", 0.9),
                                method=ResolutionMethod.DETERMINISTIC)
        proc = line("PROCG", FactKind.PROCEDURE, {})
        em = line("EMV", FactKind.EM, {"separately_identifiable": sep_ident})
        return CodingResult(encounter_id="e", date_of_service="2026-03-14",
                            lines=[proc, em]), em

    def test_em_bundled_when_not_separately_identifiable(self):
        from claude_coder.data_access import MockSource
        from claude_coder.pipeline import apply_global_package
        r, em = self._lines(sep_ident=False)
        apply_global_package(r, MockSource(gp={"PROCG": "090"}))
        self.assertTrue(em.excluded_reason)
        self.assertNotIn("EMV", {ln.chosen.code for ln in r.billable_lines})

    def test_em_kept_when_separately_identifiable(self):
        from claude_coder.data_access import MockSource
        from claude_coder.pipeline import apply_global_package
        r, em = self._lines(sep_ident=True)
        apply_global_package(r, MockSource(gp={"PROCG": "090"}))
        self.assertIsNone(em.excluded_reason)
        self.assertIn("EMV", {ln.chosen.code for ln in r.billable_lines})


class BilateralEligibilityTest(unittest.TestCase):
    """Modifier 50 / laterality is gated by the CMS bilateral indicator, so a
    per-nail code (indicator 9) gets no laterality modifier."""

    def _fact(self, lat):
        from claude_coder.models import (AttributeEvidence, ClinicalFact, EvidenceSpan,
                                         FactKind, RelationState)
        span = EvidenceSpan("x", anchored=True, span_id="s1")
        return ClinicalFact(kind=FactKind.PROCEDURE, description="x",
                            attributes={"laterality": lat}, evidence=[span],
                            attribute_evidence={"laterality": (
                                AttributeEvidence(span=span, assertion_state=RelationState.ASSERTED,
                                                  value=lat),)})

    def test_indicator_gates_modifier(self):
        from claude_coder.modifiers import ModifierEngine
        defs = {"MR": {"description": "Right side of the body"},
                "ML": {"description": "Left side of the body"},
                "MB": {"description": "Bilateral procedure"}}
        eng = ModifierEngine(defs=defs)
        # 9 = concept does not apply -> no modifier, even bilateral
        self.assertEqual(eng.assign(self._fact("bilateral"), "Debridement of nails", bilat="9"), [])
        self.assertEqual(eng.assign(self._fact("right"), "Debridement of nails", bilat="9"), [])
        # 1 = bilateral eligible -> 50; 0 = not eligible -> none
        self.assertEqual(eng.assign(self._fact("bilateral"), "Paired procedure", bilat="1"), ["MB"])
        self.assertEqual(eng.assign(self._fact("bilateral"), "Some procedure", bilat="0"), [])


class EMSettingTest(unittest.TestCase):
    def test_setting_filter_rejects_wrong_setting(self):
        from claude_coder.em import _select
        ed = CandidateCode("EMED", "cpt",
                           "Emergency department visit, moderate level medical decision making", 0.9)
        office = CandidateCode("EMOFF", "cpt",
                               "Office or other outpatient visit, established patient, "
                               "moderate level medical decision making", 0.9)
        chosen = _select("moderate", False, "office", [ed, office])  # ED ranked first
        self.assertEqual(chosen.code, "EMOFF")   # office encounter -> office code, not ED


class TerminologyIndexTest(unittest.TestCase):
    """Deterministic term->code resolution via the authoritative Alphabetic Index.
    Synthetic codes/terms — the mechanics under test are exact match, order/plural-
    independent token-set match (the Index's inverted phrasing), synonym->code, and
    the code-dotting form — none of which depend on any specific condition."""

    def test_index_term_to_code(self):
        from claude_coder.terminology import TerminologyIndex
        # AA111 has two synonyms; BB220's term is written in INVERTED order.
        idx = TerminologyIndex({"AA111": ["condition alpha", "alpha synonym"],
                                "BB220": ["gamma, entity beta", "beta variant gamma"]})
        self.assertEqual(idx.candidates("Alpha synonym"), {"AA1.11"})     # exact, dotted
        self.assertEqual(idx.candidates("entity beta gamma"), {"BB2.20"}) # inverted, token-set
        self.assertEqual(idx.candidates("no such term here"), set())      # -> caller falls back

    def test_cross_reference_terms_are_merged_into_lookup(self):
        # issue #6 F9-R12-A: `cross_reference_terms_by_code` is a SECOND,
        # distinct term source (kept separate in the source JSON specifically
        # so the embedding loader never sees it) -- but exact-Index LOOKUP
        # must still find it, folded into the same candidates() answer as a
        # direct term.
        from claude_coder.terminology import TerminologyIndex
        idx = TerminologyIndex({"AA111": ["condition alpha"]},
                               {"AA111": ["a redirect alias"], "BB220": ["a redirect alias"]})
        self.assertEqual(idx.candidates("condition alpha"), {"AA1.11"})
        self.assertEqual(idx.candidates("a redirect alias"), {"AA1.11", "BB2.20"})

    def test_direct_candidates_excludes_cross_reference_only_hits(self):
        # issue #6 F9-R12-A, REOPENED: `direct_candidates()` answers the SAME
        # lookup as `candidates()` but ONLY through `terms_by_code` -- a code
        # reachable solely through a cross-reference redirect must be absent.
        from claude_coder.terminology import TerminologyIndex
        idx = TerminologyIndex({"AA111": ["condition alpha", "shared term"]},
                               {"BB220": ["a redirect alias", "shared term"]})
        self.assertEqual(idx.direct_candidates("condition alpha"), {"AA1.11"})
        self.assertEqual(idx.direct_candidates("a redirect alias"), set())
        self.assertEqual(idx.candidates("a redirect alias"), {"BB2.20"})
        # a term BOTH sides carry: candidates() unions both codes, but
        # direct_candidates() answers ONLY the direct one.
        self.assertEqual(idx.candidates("shared term"), {"AA1.11", "BB2.20"})
        self.assertEqual(idx.direct_candidates("shared term"), {"AA1.11"})

    def test_diagnosis_resolves_via_index_first(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        # Index maps the clinician term; the record supplies the terse descriptor.
        src = MockSource(records={("C11.1", "icd10"):
                                  {"long_description": "a terse authoritative descriptor",
                                   "active": True}},
                         index={"a documented condition": {"C11.1"}})
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="a documented condition",
                            evidence=[EvidenceSpan("a documented condition")], confidence=0.99)
        line = resolve(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "C11.1")
        self.assertIn("Alphabetic Index", line.rationale)

    def test_contained_index_phrase_is_verified_candidate_not_a_lost_retrieval_gap(self):
        """A longer documented diagnosis can contain a versioned Index phrase.

        Whole-term Index lookup correctly does not treat this as an exact direct
        entry, but candidate generation must preserve the governed phrase as a
        verification-required candidate.  The test is entirely synthetic: its
        invariant is source recall -> fixed candidate universe -> descriptor
        verification, never a clinical term or a medical code.
        """
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind, ResolutionMethod
        from claude_coder.resolution import resolve
        source_phrase = "synthetic condition alpha"
        documented = "detailed synthetic condition alpha with extra context"
        src = MockSource(
            records={
                ("C33.3", "icd10"): {
                    "long_description": "synthetic condition alpha", "active": True,
                },
            },
            index_recall={
                documented: {
                    "C33.3": {
                        "method": "contained_source_phrase",
                        "normalized_query": documented,
                        "source_terms": [source_phrase],
                        "mapped_code": "C33.3",
                        "source_identity": {"source_id": "synthetic-index"},
                    },
                },
            },
        )
        fact = ClinicalFact(
            kind=FactKind.DIAGNOSIS, description=documented,
            evidence=[EvidenceSpan(documented)], confidence=0.95,
        )
        # No verifier: a contained phrase is real recall, but never the direct
        # Index shortcut that could bill without descriptor verification.
        held = resolve(_request(fact), src)
        self.assertIsNone(held.chosen, held.rationale)
        self.assertNotEqual(held.method, ResolutionMethod.DETERMINISTIC, held.rationale)
        self.assertIn("C33.3", {c.code for c in (held.alternatives or [])})

        verifier = _sv.judge(entails=lambda _: True, reason="entailed")
        released = resolve(_request(fact), src, llm=_from(verifier, "provider-a"))
        self.assertEqual(released.chosen.code, "C33.3", released.rationale)
        self.assertNotEqual(released.method, ResolutionMethod.DETERMINISTIC,
                            released.rationale)

    def test_snomed_layer_contributes_a_candidate_but_never_closes_without_verification(self):
        """issue #6 F9-R12-E (Codex): a SNOMED CT -> ICD-10-CM crosswalk hit
        maps a concept to a best-fit DEFAULT code that can be less specific
        than, or wrong for, the documented condition -- this module's own
        long-standing docstring already said it must be ALWAYS entailment-
        confirmed, never trusted deterministically, but with no LLM
        available to actually perform that confirmation, it used to fall
        straight through to `_decide` and close DETERMINISTIC anyway. Fixed
        via `requires_verification`: with no verifier, it must ABSTAIN
        (the code retained as a candidate requiring confirmation), never
        auto-bill on the crosswalk alone."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        src = MockSource(records={("C22.2", "icd10"):
                                  {"long_description": "a terse authoritative descriptor",
                                   "active": True}},
                         index={}, snomed={"a documented condition": {"C22.2"}})
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="a documented condition",
                            evidence=[EvidenceSpan("a documented condition")], confidence=0.95)
        line = resolve(_request(fact), src)
        self.assertNotEqual(line.method, ResolutionMethod.DETERMINISTIC, line.rationale)
        self.assertIsNone(line.chosen, line.rationale)
        alt_codes = {c.code for c in (line.alternatives or [])}
        self.assertIn("C22.2", alt_codes, line.rationale)

    def test_snomed_layer_resolves_once_a_verifier_confirms_entailment(self):
        """The positive counterpart: with an LLM available to actually
        perform the entailment confirmation the SNOMED crosswalk always
        requires, a SNOMED-sourced hit still resolves -- the fix routes it
        through real verification, it does not simply disable SNOMED
        recall."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        src = MockSource(records={("C22.2", "icd10"):
                                  {"long_description": "a terse authoritative descriptor",
                                   "active": True}},
                         index={}, snomed={"a documented condition": {"C22.2"}})
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="a documented condition",
                            evidence=[EvidenceSpan("a documented condition")], confidence=0.95)
        llm = _sv.judge(entails=lambda d: True, reason="entailed")
        line = resolve(_request(fact), src, llm=_from(llm, "provider-a"))
        self.assertEqual(line.chosen.code, "C22.2", line.rationale)

    def test_governed_diagnosis_sources_are_unioned_before_direct_index_closure(self):
        """A direct Index hit cannot hide a different governed-map candidate.

        Codes and descriptors are synthetic.  This covers only the generic
        invariant: collect every governed source candidate before allowing a
        deterministic selection.
        """
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        src = MockSource(
            records={
                ("C11.1", "icd10"): {"long_description": "synthetic direct condition",
                                       "active": True},
                ("C22.2", "icd10"): {"long_description": "synthetic governed condition",
                                       "active": True},
            },
            index={"synthetic documented condition": {"C11.1"}},
            snomed={"synthetic documented condition": {"C22.2"}},
        )
        fact = ClinicalFact(
            kind=FactKind.DIAGNOSIS, description="synthetic documented condition",
            evidence=[EvidenceSpan("synthetic documented condition")], confidence=0.95,
        )
        line = resolve(_request(fact), src)
        self.assertNotEqual(line.method, ResolutionMethod.DETERMINISTIC)
        observed = {(record["code"], record["system"])
                    for record in (line.candidate_eligibility or [])}
        self.assertEqual(observed, {("C11.1", "icd10"), ("C22.2", "icd10")})

    def test_multi_map_governed_diagnosis_source_keeps_every_candidate(self):
        """Source-map cardinality must not silently remove diagnosis recall."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder.resolution import resolve
        src = MockSource(
            records={
                ("C11.1", "icd10"): {"long_description": "synthetic first condition",
                                       "active": True},
                ("C22.2", "icd10"): {"long_description": "synthetic second condition",
                                       "active": True},
            },
            index={},
            snomed={"synthetic documented condition": {"C11.1", "C22.2"}},
        )
        fact = ClinicalFact(
            kind=FactKind.DIAGNOSIS, description="synthetic documented condition",
            evidence=[EvidenceSpan("synthetic documented condition")], confidence=0.95,
        )
        line = resolve(_request(fact), src)
        observed = {(record["code"], record["system"])
                    for record in (line.candidate_eligibility or [])}
        self.assertEqual(observed, {("C11.1", "icd10"), ("C22.2", "icd10")})

    def test_same_fact_and_source_snapshot_produce_the_same_candidate_universe(self):
        """Candidate generation is repeatable before any model decision.

        This deliberately compares the complete code-level eligibility record,
        not merely the winner.  A later refactor may not change a previously
        verified universe by stopping after a different source or control-flow
        branch for the same fact and authoritative snapshot.
        """
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder.resolution import resolve
        src = MockSource(
            records={
                ("C11.1", "icd10"): {"long_description": "synthetic direct condition",
                                       "active": True},
                ("C22.2", "icd10"): {"long_description": "synthetic governed condition",
                                       "active": True},
            },
            index={"synthetic documented condition": {"C11.1"}},
            snomed={"synthetic documented condition": {"C22.2"}},
        )
        fact = ClinicalFact(
            kind=FactKind.DIAGNOSIS, description="synthetic documented condition",
            evidence=[EvidenceSpan("synthetic documented condition")], confidence=0.95,
        )

        def universe():
            line = resolve(_request(fact), src)
            return [(record["code"], record["system"], record["eligible"],
                     record.get("reason"))
                    for record in (line.candidate_eligibility or [])]

        self.assertEqual(universe(), universe())

    def test_category_expands_to_leaf_by_laterality(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (AttributeEvidence, ClinicalFact, EvidenceSpan,
                                         FactKind, RelationState, ResolutionMethod)
        from claude_coder.resolution import resolve
        # Index returns the category DX4; documented laterality selects the leaf.
        # Synthetic codes/descriptors — the mechanic is category->leaf-by-laterality,
        # not any specific condition.
        recs = {("DX40", "icd10"): {"long_description": "some condition, unspecified site", "active": True},
                ("DX41", "icd10"): {"long_description": "some condition, right site", "active": True},
                ("DX42", "icd10"): {"long_description": "some condition, left site", "active": True}}
        src = MockSource(records=recs, index={"a documented condition": {"DX4"}})
        span = EvidenceSpan("a documented condition", anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="a documented condition",
                            attributes={"laterality": "right"},
                            evidence=[span],
                            attribute_evidence={"laterality": (
                                AttributeEvidence(span=span, assertion_state=RelationState.ASSERTED,
                                                  value="right"),)},
                            confidence=0.99)
        line = resolve(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "DX41")     # right-side leaf, not the category


class MultiCodeIndexCrossReferenceCandidateTest(unittest.TestCase):
    """issue #6 F9-R12-A: a cross-reference (<see>/<seeAlso>) redirect can span
    MULTIPLE, structurally distinct code families -- e.g. the real 'paronychia'
    alias resolving to BOTH the toe and finger cellulitis families. Unlike the
    single-stem category above, `index_codes` here returns more than one STEM,
    so the len(idx)==1 deterministic-trust gate does not apply; each stem's
    billable leaves must be UNIONED into the candidate pool as a normal
    candidate source (never auto-selected), narrowed by whatever the
    documentation actually supports -- the same tie policy any other
    multi-candidate pool goes through. Synthetic codes/descriptors throughout."""

    def test_a_multi_code_redirect_narrowed_by_a_documented_attribute_still_needs_verification(self):
        """issue #6 F9-R12-E (Codex): a multi-code Index redirect is
        "supplementary navigation, not proof" even AFTER `_decide`'s own
        laterality-narrowing settles it down to a single candidate -- the
        underlying signal is still a `see`/`seeAlso` hit, not a direct
        entry, so it stays `requires_verification=True` and must not
        auto-close with no LLM available to confirm it, matching every
        other redirect-only single-code case."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import (AttributeEvidence, ClinicalFact, EvidenceSpan,
                                         FactKind, RelationState, ResolutionMethod)
        from claude_coder.resolution import resolve
        # Two DISTINCT stems (not siblings under one category) -- the shape of a
        # redirect spanning two separate families, not a single category expanding
        # to its own children.
        recs = {("DX50", "icd10"): {"long_description": "some condition, right site", "active": True},
                ("DX60", "icd10"): {"long_description": "some condition, left site", "active": True}}
        src = MockSource(records=recs, index={"a documented condition": {"DX50", "DX60"}})
        span = EvidenceSpan("a documented condition", anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="a documented condition",
                            attributes={"laterality": "right"},
                            evidence=[span],
                            attribute_evidence={"laterality": (
                                AttributeEvidence(span=span, assertion_state=RelationState.ASSERTED,
                                                  value="right"),)},
                            confidence=0.99)
        line = resolve(_request(fact), src)
        self.assertNotEqual(line.method, ResolutionMethod.DETERMINISTIC, line.rationale)
        self.assertIsNone(line.chosen, line.rationale)
        alt_codes = {c.code for c in (line.alternatives or [])}
        self.assertIn("DX50", alt_codes, line.rationale)

    def test_a_multi_code_redirect_resolves_once_a_verifier_confirms_the_narrowed_candidate(self):
        """The positive counterpart: with an LLM available, the same
        laterality-narrowed redirect candidate is entailment-confirmed and
        released -- the fix routes it through real verification, it does
        not simply disable index-redirect recall."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import (AttributeEvidence, ClinicalFact, EvidenceSpan,
                                         FactKind, RelationState)
        from claude_coder.resolution import resolve
        recs = {("DX50", "icd10"): {"long_description": "some condition, right site", "active": True},
                ("DX60", "icd10"): {"long_description": "some condition, left site", "active": True}}
        src = MockSource(records=recs, index={"a documented condition": {"DX50", "DX60"}})
        span = EvidenceSpan("a documented condition", anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="a documented condition",
                            attributes={"laterality": "right"},
                            evidence=[span],
                            attribute_evidence={"laterality": (
                                AttributeEvidence(span=span, assertion_state=RelationState.ASSERTED,
                                                  value="right"),)},
                            confidence=0.99)
        llm = _sv.judge(entails=lambda d: "right" in d.lower(), reason="entailed")
        line = resolve(_request(fact), src, llm=_from(llm, "provider-a"))
        self.assertEqual(line.chosen.code, "DX50", line.rationale)

    def test_an_unnarrowed_multi_code_redirect_is_never_auto_billed(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind, ResolutionMethod
        from claude_coder.resolution import resolve
        recs = {("DX50", "icd10"): {"long_description": "some condition, right site", "active": True},
                ("DX60", "icd10"): {"long_description": "some condition, left site", "active": True}}
        src = MockSource(records=recs, index={"a documented condition": {"DX50", "DX60"}})
        # No laterality (or any other distinguishing) evidence documented at all --
        # the redirect is real candidate signal, but nothing narrows it, so it must
        # NOT auto-bill either family.
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="a documented condition",
                            evidence=[EvidenceSpan("a documented condition")],
                            confidence=0.99)
        line = resolve(_request(fact), src)
        self.assertIsNone(line.chosen, line.rationale)
        self.assertNotEqual(line.method, ResolutionMethod.DETERMINISTIC)
        alt_codes = {c.code for c in (line.alternatives or [])}
        self.assertTrue({"DX50", "DX60"} & alt_codes, line.rationale)


class SingleCodeCrossReferenceOnlyCandidateTest(unittest.TestCase):
    """issue #6 F9-R12-A, REOPENED (Codex's re-review): even a SINGLE-code
    Index hit must not close deterministically when that code is reachable
    ONLY through a see/seeAlso cross-reference redirect -- a redirect is
    supplementary navigation, not proof the note supports that code (the
    real source has 1,142 single-code-resolved directives, 85 of them
    seeAlso). `MockSource(index=..., index_direct=...)` lets a test
    distinguish the two: `index` is the full (direct+cross-reference) hit
    set `resolve()` reads for candidate recall, `index_direct` is the
    subset reachable through a direct entry -- the gate for the old
    immediate-trust shortcut. Synthetic codes throughout."""

    def test_a_direct_single_code_hit_still_closes_deterministically(self):
        # Unaffected control: MockSource defaults `index_direct` to the SAME
        # dict as `index` when not overridden, so every pre-existing index=
        # test (never mentioning redirects) keeps its prior behavior exactly.
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind, ResolutionMethod
        from claude_coder.resolution import resolve
        src = MockSource(
            records={("DIRECT1", "icd10"): {"long_description": "a documented condition",
                                            "active": True}},
            index={"a documented condition": {"DIRECT1"}})
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="a documented condition",
                            evidence=[EvidenceSpan("a documented condition")], confidence=0.9)
        line = resolve(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC, line.rationale)
        self.assertEqual(line.chosen.code, "DIRECT1", line.rationale)

    def test_a_cross_reference_only_single_code_hit_never_unilaterally_wins(self):
        """The real safety property: a redirect-only hit must not shortcut
        straight to DETERMINISTIC before a plausible competing candidate
        (here, from ordinary retrieval) ever gets a chance to be weighed --
        the OLD bug bypassed retrieval/recall entirely for a bare single-hit
        Index result, auto-billing it unilaterally regardless of what else
        was retrievable for the same query."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind, ResolutionMethod
        from claude_coder.resolution import resolve
        recs = {("CROSSONLY", "icd10"): {"long_description": "some condition, unspecified",
                                         "active": True},
               ("ALSOPLAUS", "icd10"): {"long_description": "some other condition, unspecified",
                                        "active": True}}
        src = MockSource(
            records=recs,
            index={"a documented condition": {"CROSSONLY"}},
            index_direct={},   # CROSSONLY is reachable ONLY via cross-reference
            # Score kept ABOVE the recall floor so this candidate actually
            # competes rather than being screened out on relevance alone --
            # the property under test is the TIE POLICY seeing both, not a
            # relevance contest.
            retrieval={("*", "icd10"): [
                CandidateCode("ALSOPLAUS", "icd10", "some other condition, unspecified", 0.9)]})
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="a documented condition",
                            evidence=[EvidenceSpan("a documented condition")], confidence=0.9)
        line = resolve(_request(fact), src)
        unilaterally_billed = (line.method == ResolutionMethod.DETERMINISTIC
                               and line.chosen is not None
                               and line.chosen.code == "CROSSONLY")
        self.assertFalse(unilaterally_billed, line.rationale)

    def test_a_redirect_only_hit_with_no_competing_candidate_still_abstains(self):
        """issue #6 F9-R12-A, second re-review (Codex's exact required
        regression): a redirect-only single-code hit must not close
        deterministically even as the pool's SOLE survivor -- with no
        retrieval candidates at all and no LLM to independently verify it,
        it must ABSTAIN and retain the code as a candidate, never auto-bill
        on the redirect alone. (The earlier test above only proved the pool
        no longer shortcuts PAST a competing candidate; this proves the
        narrower, harder case Codex's own reproduction targeted directly.)"""
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind, ResolutionMethod
        from claude_coder.resolution import resolve
        src = MockSource(
            records={("CROSSONLY", "icd10"): {"long_description": "some condition, unspecified",
                                              "active": True}},
            index={"a documented condition": {"CROSSONLY"}},
            index_direct={})   # reachable ONLY via cross-reference; no retrieval configured at all
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="a documented condition",
                            evidence=[EvidenceSpan("a documented condition")], confidence=0.9)
        line = resolve(_request(fact), src)
        self.assertNotEqual(line.method, ResolutionMethod.DETERMINISTIC, line.rationale)
        self.assertIsNone(line.chosen, line.rationale)
        alt_codes = {c.code for c in (line.alternatives or [])}
        self.assertIn("CROSSONLY", alt_codes, line.rationale)


class ConceptRelationIndexTest(unittest.TestCase):
    """SAME / ancestor-descendant-or-ambiguous-overlap / unresolved for two clinical
    terms, from a governed concept graph (issue #6 F7-R3-C/C3) -- never from lexical
    shape, and never a fabricated DISJOINT (an IS_A subsumption hierarchy has no basis
    to assert two concepts are opposed). Synthetic concept ids/terms — the mechanics
    under test are candidate matching (exact, despaced, token-set), ancestor-closure
    BFS, unique-vs-ambiguous resolution, and the relation verdict, none of which depend
    on any specific SNOMED concept."""

    def _index(self):
        from claude_coder.terminology import ConceptRelationIndex
        # C1 (root) -> C2 -> C3, an unrelated C9, and an AMBIGUOUS term ("shared name")
        # naming both C2 and C9 -- two different real-world concepts with one shared
        # lexical name, exactly the kind of term the licensed release audit found
        # (65 exact terms, 543 token-set keys resolve to more than one concept).
        return ConceptRelationIndex({
            "C1": {"terms": ["root structure"], "parents": []},
            "C2": {"terms": ["mid structure", "middle structure", "shared name"],
                  "parents": ["C1"]},
            "C3": {"terms": ["leaf structure", "the leaf"], "parents": ["C2"]},
            "C9": {"terms": ["other structure", "shared name"], "parents": []},
            # issue #6 F9-R4-R2: two DIFFERENT-width, unrelated governed terms,
            # needed to reproduce/pin the cross-width masking bug -- every term
            # above is 2 tokens, which cannot expose it (a single-width table
            # never masks itself).
            "C10": {"terms": ["repair"], "parents": []},
            # A 1-token term that is also the FIRST token of "leaf structure"
            # (C3) -- a genuinely different concept, so preferring the longer
            # phrase at this shared start position is a real, checkable choice,
            # not a vacuous one (nothing else names "leaf" alone).
            "C11": {"terms": ["leaf"], "parents": []},
        })

    def test_two_synonyms_of_the_same_concept_are_same(self):
        idx = self._index()
        self.assertEqual(idx.relation("leaf structure", "the leaf"), "same")

    def test_an_ancestor_and_descendant_are_related_not_same(self):
        idx = self._index()
        self.assertEqual(idx.relation("leaf structure", "mid structure"),
                         "ancestor_descendant")
        self.assertEqual(idx.relation("mid structure", "root structure"),
                         "ancestor_descendant")

    def test_two_concepts_with_no_relation_are_unresolved_not_disjoint(self):
        """Codex F7-R3-C3, exact-SHA re-review: SNOMED's IS_A hierarchy carries no
        disjointness axiom (per its own OWL/NNF specification), and the licensed
        release audit found tens of thousands of non-ancestor sibling pairs that still
        share a descendant -- absent subsumption is not evidence of opposition. Two
        structurally unrelated concepts must resolve UNRESOLVED, never a fabricated
        DISJOINT that could wrongly add a billed occurrence."""
        idx = self._index()
        self.assertEqual(idx.relation("leaf structure", "other structure"),
                         "unresolved")

    def test_an_ambiguous_shared_candidate_is_related_not_a_confirmed_same(self):
        """Codex F7-R3-C3: a term resolving to MORE THAN ONE concept is itself
        ambiguous within the graph -- the candidate sets merely intersecting is not
        proof the two terms name the same real-world structure, only that they MIGHT.
        'shared name' names both C2 and C9; matched against a term unique to C2, the
        verdict must be the weaker RELATED, never a confirmed SAME."""
        idx = self._index()
        self.assertEqual(idx.relation("shared name", "middle structure"),
                         "ancestor_descendant")   # RELATED verdict string
        self.assertNotEqual(idx.relation("shared name", "middle structure"), "same")

    def test_relation_detail_carries_the_auditable_match_basis(self):
        from claude_coder.terminology import CONCEPT_SAME
        idx = self._index()
        detail = idx.relation_detail("leaf structure", "the leaf")
        self.assertEqual(detail.verdict, CONCEPT_SAME)
        self.assertEqual(detail.match_a.candidates, ("C3",))
        self.assertEqual(detail.match_a.method, "exact")
        self.assertTrue(detail.match_a.unique)
        self.assertEqual(detail.alternatives_a, ())
        self.assertEqual(detail.confidence, 1.0)

        ambiguous = idx.relation_detail("shared name", "no such term")
        self.assertEqual(ambiguous.match_a.candidates, ("C2", "C9"))
        self.assertFalse(ambiguous.match_a.unique)
        self.assertEqual(ambiguous.alternatives_a, ("C2", "C9"))

    def test_an_unknown_term_is_unresolved_not_a_guessed_relation(self):
        idx = self._index()
        self.assertEqual(idx.relation("leaf structure", "no such term"), "unresolved")
        self.assertEqual(idx.relation("no such term", "leaf structure"), "unresolved")

    # ---- match_longest / embedded=True (issue #6 F9-R4-R1) --------------------------
    def test_match_longest_finds_a_governed_term_embedded_in_a_longer_sentence(self):
        """Codex's exact reopened gap: `match` requires the WHOLE string to equal a
        term, so a real action description ("performed leaf structure today") never
        matched anything. `match_longest` scans for the phrase anywhere inside it."""
        idx = self._index()
        m = idx.match_longest("the surgeon performed leaf structure today without incident")
        self.assertEqual(m.candidates, ("C3",))
        self.assertTrue(m.method.startswith("token_scan"))

    def test_match_longest_prefers_the_longest_matching_phrase(self):
        """issue #6 F9-R4-R2: "longest wins" applies where it actually means
        something -- two governed phrases that could BOTH start at the SAME
        position. "leaf" (C11) is a real, different governed concept that is
        also the first token of "leaf structure" (C3); at that shared start
        position, the longer phrase must win and C11 must never appear."""
        idx = self._index()
        m = idx.match_longest("today we addressed the leaf structure fully")
        self.assertEqual(m.candidates, ("C3",))
        self.assertNotIn("C11", m.candidates)

    def test_match_longest_is_insensitive_to_surrounding_punctuation(self):
        idx = self._index()
        m = idx.match_longest("Procedure: leaf structure, performed today.")
        self.assertEqual(m.candidates, ("C3",))

    def test_match_longest_finds_two_different_width_concepts_in_one_text(self):
        """issue #6 F9-R4-R2, Codex's exact reproduction: a compound action naming
        TWO separate governed procedures of DIFFERENT phrase lengths must find
        BOTH, never let the wider phrase's match suppress the scan for the other
        ("removal of bone spur and tenotomy were performed" -> both procedures)."""
        idx = self._index()
        m = idx.match_longest("today we performed leaf structure and repair together")
        self.assertEqual(set(m.candidates), {"C3", "C10"})
        self.assertFalse(m.unique)

    def test_match_longest_non_overlapping_matches_do_not_double_count(self):
        """Two DIFFERENT 2-token governed phrases, directly adjacent with no
        separator token between them, must each be found once -- the cursor
        advances past a match's own consumed tokens, never re-scanning inside it
        or skipping the very next token that starts a second, real match."""
        idx = self._index()
        m = idx.match_longest("leaf structure other structure")
        self.assertEqual(set(m.candidates), {"C3", "C9"})

    def test_match_longest_two_different_widths_survives_punctuation(self):
        idx = self._index()
        m = idx.match_longest("Procedure: leaf structure. Repair performed.")
        self.assertEqual(set(m.candidates), {"C3", "C10"})

    def test_match_longest_two_different_embedded_concepts_is_ambiguous(self):
        """Codex's acceptance criterion: two different governed concepts both occurring
        in one action description must stay ambiguous/unresolved, never arbitrarily
        pick one."""
        idx = self._index()
        m = idx.match_longest("mentions both leaf structure and other structure today")
        self.assertEqual(set(m.candidates), {"C3", "C9"})
        self.assertFalse(m.unique)

    def test_match_longest_falls_back_to_whole_string_match_first(self):
        """When the WHOLE string already equals a governed term, `match_longest`
        returns that match directly without falling through to the scan (same
        candidates/method as plain `match`)."""
        idx = self._index()
        direct = idx.match("leaf structure")
        via_longest = idx.match_longest("leaf structure")
        self.assertEqual(via_longest.candidates, direct.candidates)
        self.assertEqual(via_longest.method, direct.method)

    def test_match_longest_no_embedded_phrase_is_none(self):
        idx = self._index()
        m = idx.match_longest("an entirely unrelated sentence about something else")
        self.assertEqual(m.candidates, ())
        self.assertEqual(m.method, "none")

    def test_embedded_relation_detail_resolves_two_real_action_sentences(self):
        """The end-to-end reproduction: two full action descriptions, worded
        differently, each containing the SAME governed phrase, resolve SAME -- exactly
        the summary-line-vs-narrative-sentence case F9-R4 exists for."""
        idx = self._index()
        detail = idx.relation_detail(
            "the surgeon performed leaf structure today",
            "leaf structure was addressed without complication",
            embedded=True)
        from claude_coder.terminology import CONCEPT_SAME
        self.assertEqual(detail.verdict, CONCEPT_SAME)

    def test_embedded_relation_detail_ancestor_descendant_never_promotes_to_same(self):
        idx = self._index()
        detail = idx.relation_detail(
            "performed leaf structure today", "performed mid structure today",
            embedded=True)
        self.assertEqual(detail.verdict, "ancestor_descendant")

    def test_non_embedded_relation_detail_is_unaffected_by_this_change(self):
        """The default (embedded=False) path -- used by anatomy's `concept_relation_
        detail` -- must behave exactly as before: a full sentence does NOT match a
        bare governed term."""
        idx = self._index()
        detail = idx.relation_detail(
            "the surgeon performed leaf structure today", "leaf structure")
        from claude_coder.terminology import CONCEPT_UNRESOLVED
        self.assertEqual(detail.verdict, CONCEPT_UNRESOLVED)


def _line(code, kind, descriptor="d", attrs=None, system="cpt"):
    from claude_coder.models import (ClinicalFact, EvidenceSpan, ResolutionMethod,
                                     ResolvedLine)
    f = ClinicalFact(kind=kind, description="x", attributes=(attrs or {}),
                     evidence=[EvidenceSpan("x")])
    return ResolvedLine(fact=f, chosen=CandidateCode(code, system, descriptor, 0.9),
                        method=ResolutionMethod.DETERMINISTIC)


class SectionApplicabilityTest(unittest.TestCase):
    """Mechanic 1 — an anesthesia-section code (detected from descriptor grammar,
    not a code range) is bundled into the operating provider's claim unless a
    separate anesthesia provider is documented."""

    def _result(self, anes_attrs=None):
        from claude_coder.models import CodingResult, FactKind
        surgery = _line("SURG_X", FactKind.PROCEDURE, "Ostectomy, complete excision")
        anes = _line("ANES_X", FactKind.PROCEDURE,
                     "Anesthesia for procedures on nerves of a structure", anes_attrs or {})
        return CodingResult(encounter_id="e", date_of_service="2026-03-14",
                            lines=[surgery, anes]), anes

    def test_anesthesia_excluded_on_operative_claim(self):
        from claude_coder.pipeline import apply_section_applicability
        r, anes = self._result()
        apply_section_applicability(r)
        self.assertTrue(anes.excluded_reason)
        self.assertNotIn("ANES_X", {ln.chosen.code for ln in r.billable_lines})

    def test_anesthesia_kept_when_separate_provider_documented(self):
        from claude_coder.pipeline import apply_section_applicability
        r, anes = self._result(anes_attrs={"anesthesia_provider": True})
        apply_section_applicability(r)
        self.assertIsNone(anes.excluded_reason)

    def test_escalated_anesthesia_excluded_deterministically(self):
        # an ESCALATED procedure whose candidates are anesthesia-section is excluded
        # deterministically (not left as a review item), independent of resolution.
        from claude_coder.models import (ClinicalFact, CodingResult, EvidenceSpan,
                                         FactKind, ResolutionMethod, ResolvedLine)
        from claude_coder.pipeline import apply_section_applicability
        surgery = _line("SURG", FactKind.PROCEDURE, "Ostectomy of a structure")
        af = ClinicalFact(kind=FactKind.PROCEDURE, description="regional block for anesthesia",
                          evidence=[EvidenceSpan("regional block for anesthesia")])
        anes = ResolvedLine(fact=af, chosen=None, method=ResolutionMethod.ABSTAINED,
                            alternatives=[CandidateCode("ANESX", "cpt",
                                          "Anesthesia for procedures on a structure", 0.8)])
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[surgery, anes])
        apply_section_applicability(r)
        self.assertTrue(anes.excluded_reason)
        self.assertIn("anesthesia-section", anes.excluded_reason)

    def test_surgical_line_with_incidental_anesthesia_candidate_not_excluded(self):
        # a SURGICAL procedure whose LEADING candidate is surgical must NOT be
        # excluded as anesthesia just because an anesthesia neighbour is in the pool.
        from claude_coder.models import (ClinicalFact, CodingResult, EvidenceSpan,
                                         FactKind, ResolutionMethod, ResolvedLine)
        from claude_coder.pipeline import apply_section_applicability
        surgery = _line("SURG", FactKind.PROCEDURE, "Ostectomy of a structure")
        sf = ClinicalFact(kind=FactKind.PROCEDURE, description="tendon debridement",
                          evidence=[EvidenceSpan("tendon debridement")])
        esc = ResolvedLine(fact=sf, chosen=None, method=ResolutionMethod.ABSTAINED,
                           alternatives=[CandidateCode("SURGC", "cpt", "Tenolysis of a tendon", 0.8),
                                         CandidateCode("ANESX", "cpt", "Anesthesia for procedures on a structure", 0.5)])
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[surgery, esc])
        apply_section_applicability(r)
        self.assertIsNone(esc.excluded_reason)      # leading candidate is surgical
        self.assertFalse(esc.resolved)

    def test_section_detection_is_descriptor_driven(self):
        # section is read from the descriptor's leading grammar, not any code/term.
        from claude_coder.ontology import code_section
        self.assertEqual(code_section("Anesthesia for procedures on the structure"), "anesthesia")
        self.assertIsNone(code_section("Some surgical service on a structure"))


class NcciBundlingTest(unittest.TestCase):
    """Mechanic 3 — an unmodified PTP pair DEMOTES the component (keeps the
    payable code) instead of blocking; '(separate procedure)' codes bundle."""

    def test_component_demoted_not_blocked(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import apply_ncci_bundling
        payable = _line("COMPREH", FactKind.PROCEDURE, "comprehensive procedure")
        component = _line("COMPON", FactKind.PROCEDURE, "component procedure")
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[payable, component])
        # directional edit: COMPREH is column-1 payable, COMPON is column-2, no bypass
        src = MockSource(ncci={("COMPREH", "COMPON"): "0"})
        apply_ncci_bundling(r, src)
        self.assertIsNone(payable.excluded_reason)
        self.assertTrue(component.excluded_reason)
        self.assertNotIn("COMPON", {ln.chosen.code for ln in r.billable_lines})

    def test_separate_procedure_designation_bundled(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import apply_ncci_bundling
        main = _line("MAINP", FactKind.PROCEDURE, "definitive surgical procedure")
        sep = _line("SEPP", FactKind.IMAGING, "Fluoroscopy (separate procedure), 1 hour")
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[main, sep])
        apply_ncci_bundling(r, MockSource())
        self.assertTrue(sep.excluded_reason)
        self.assertIsNone(main.excluded_reason)

    def test_separate_procedure_not_bundled_by_supply_only(self):
        # a '(separate procedure)' code must NOT bundle just because a supply/device
        # is also reported — only another actual PROCEDURE triggers the bundle.
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import apply_ncci_bundling
        sep = _line("SEPP", FactKind.PROCEDURE, "some service (separate procedure)")
        device = _line("DEVX", FactKind.SUPPLY, "implant device", system="hcpcs")
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[sep, device])
        apply_ncci_bundling(r, MockSource())
        self.assertIsNone(sep.excluded_reason)      # a device is not "another procedure"

    def test_bypassed_pair_keeps_both(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import apply_ncci_bundling
        a = _line("PA", FactKind.PROCEDURE, "procedure A")
        b = _line("PB", FactKind.PROCEDURE, "procedure B")
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[a, b])
        r.bypassed_ncci = [frozenset(("PA", "PB"))]     # a distinct-service modifier applied
        src = MockSource(ncci={("PA", "PB"): "1"})       # bypassable edit
        apply_ncci_bundling(r, src)
        self.assertIsNone(a.excluded_reason)
        self.assertIsNone(b.excluded_reason)


class DedupTest(unittest.TestCase):
    """Mechanic 4 — two facts resolving to the same code become one billable line."""

    def test_duplicate_code_collapsed(self):
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import dedup_lines
        a = _line("SAME", FactKind.PROCEDURE, "same procedure")
        b = _line("SAME", FactKind.PROCEDURE, "same procedure")
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[a, b])
        dedup_lines(r)
        billed = [ln for ln in r.billable_lines if ln.chosen.code == "SAME"]
        self.assertEqual(len(billed), 1)
        self.assertTrue(a.excluded_reason or b.excluded_reason)

    def test_a_governed_concept_source_resolves_a_true_synonym_pair_to_one_unit(self):
        """Issue #6 F7-R3-C: without a concept source, two mentions worded as genuine
        synonyms on the anatomy axis hold as UNDETERMINED (see
        AxisComparisonIsNormalized in test_evidence_graph.py) because lexical shape
        alone cannot tell a synonym pair from a real distinction. With a governed
        concept source available and confirming the SAME concept, that ambiguity
        resolves: one documented service, described twice, is one billed unit."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import dedup_lines
        from claude_coder.terminology import CONCEPT_SAME
        a = _line("SAME", FactKind.PROCEDURE, "same procedure",
                 attrs={"anatomy": "great toe"})
        b = _line("SAME", FactKind.PROCEDURE, "same procedure",
                 attrs={"anatomy": "hallux"})
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[a, b])
        src = MockSource(concept_relation={("great toe", "hallux"): CONCEPT_SAME})
        dedup_lines(r, src)
        billed = [ln for ln in r.billable_lines if ln.chosen.code == "SAME"]
        self.assertEqual(len(billed), 1)
        self.assertEqual(billed[0].units, 1)

    def test_a_reported_disjoint_relation_never_confirms_a_distinct_occurrence(self):
        """Codex F7-R3-C3, exact-SHA re-review: SNOMED's IS_A hierarchy has no basis to
        assert two concepts are opposed, so `ConceptRelationIndex` never returns
        CONCEPT_DISJOINT -- and `coreference.axis_relation` does not promote on it
        even if a source reports it anyway (defense in depth against a
        non-conforming or future `CodeSource` implementation). A pair a source claims
        is DISJOINT must still HOLD, exactly like an unresolved pair -- never silently
        confirmed as two occurrences from a relation this system does not trust."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import dedup_lines
        from claude_coder.terminology import CONCEPT_DISJOINT
        a = _line("SAME", FactKind.PROCEDURE, "same procedure",
                 attrs={"anatomy": "second toe"})
        b = _line("SAME", FactKind.PROCEDURE, "same procedure",
                 attrs={"anatomy": "great toe"})
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[a, b])
        src = MockSource(
            concept_relation={("second toe", "great toe"): CONCEPT_DISJOINT})
        dedup_lines(r, src)
        billed = [ln for ln in r.billable_lines if ln.chosen.code == "SAME"]
        self.assertEqual(billed, [],
                         "a claimed DISJOINT relation must hold, never confirm an "
                         "occurrence -- this system has no authoritative basis for it")

    def test_an_unresolved_concept_source_falls_back_to_the_existing_hold(self):
        """A concept source that cannot resolve either term (issue #6 F7-R3-C) must
        degrade to the same conservative hold as having no source at all -- never a
        wrong merge or a wrong split."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        from claude_coder.pipeline import dedup_lines
        a = _line("SAME", FactKind.PROCEDURE, "same procedure",
                 attrs={"anatomy": "great toe"})
        b = _line("SAME", FactKind.PROCEDURE, "same procedure",
                 attrs={"anatomy": "hallux"})
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[a, b])
        src = MockSource()  # no concept_relation mapping configured -> unresolved
        dedup_lines(r, src)
        billed = [ln for ln in r.billable_lines if ln.chosen.code == "SAME"]
        self.assertEqual(billed, [],
                         "an unresolved concept relation must hold, not silently bill")


class ProcedureIndexTest(unittest.TestCase):
    """Mechanic 5 — a procedure phrase resolves through the CPT/HCPCS descriptor
    index (deterministic) before embedding, the procedure-axis analog of the ICD
    Alphabetic Index."""

    def test_procedure_resolves_via_descriptor_index(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        src = MockSource(records={("PROC_OST", "cpt"):
                                  {"long_description": "some service on a named structure",
                                   "active": True}},
                         proc_index={"some service on a named structure": {"PROC_OST"}})
        fact = ClinicalFact(kind=FactKind.PROCEDURE,
                            description="some service on a named structure",
                            evidence=[EvidenceSpan("some service on a named structure")],
                            confidence=0.99)
        line = resolve(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "PROC_OST")
        self.assertIn("descriptor index", line.rationale)


class SupportRankingTest(unittest.TestCase):
    """Descriptor↔fact token support ORDERS candidates and NEVER eliminates one.

    It also never SELECTS one. Product directive section 4 allows lexical/semantic
    similarity to widen a candidate pool and forbids it from verifying a code, so the
    near-tie below is settled by what the ORIGINAL DOCUMENT was proven to say — not by
    which descriptor happens to share more words with the note.
    """

    _MATCH = CandidateCode("P_MATCH", "cpt", "excision of bursa of the foot", 0.80)
    _NEIGH = CandidateCode("P_NEIGH", "cpt", "open treatment of fracture", 0.80)

    def _source(self):
        from claude_coder.data_access import MockSource
        return MockSource(
            records={("P_MATCH", "cpt"): {"active": True},
                     ("P_NEIGH", "cpt"): {"active": True}},
            # neighbour listed first: retrieval ORDER must not decide either
            retrieval={("*", "cpt"): [self._NEIGH, self._MATCH]})

    def _fact(self, *, anchored):
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        quote = "the bursa was excised"
        return ClinicalFact(
            kind=FactKind.PROCEDURE, description="excision of bursa", confidence=0.9,
            evidence=[EvidenceSpan(quote, start=0, end=len(quote), anchored=anchored,
                                   span_id=("span-0" if anchored else None))])

    def test_a_source_anchored_word_hit_cannot_settle_an_untyped_tie(self):
        """Equal recall, and the neighbour is retrieved first. Even a source-anchored
        quotation cannot promote an untyped token overlap into clinical-role evidence;
        both candidates remain alternatives until a typed axis distinguishes them."""
        from app.contracts.source_evidence import (ReconciliationStatus,
                                                   SourceReconciliation,
                                                   SpanReconciliation)
        from claude_coder.resolution import resolve
        reconciliation = SourceReconciliation(spans=(
            SpanReconciliation(span_id="span-0", status=ReconciliationStatus.AGREED,
                               pages=(1,)),))
        line = resolve(_request(self._fact(anchored=True)), self._source(),
                       reconciliation=reconciliation)
        self.assertIsNone(line.chosen, line.rationale)
        self.assertEqual({c.code for c in line.alternatives}, {"P_MATCH", "P_NEIGH"})
        self.assertEqual(line.tie_record["winner"], "")
        self.assertFalse(line.tie_record["axes"][0]["selectable"])

    def test_token_overlap_alone_cannot_close_the_same_near_tie(self):
        """The identical pool and the identical wording, with the quotation NOT anchored
        to the source. The descriptor/note token overlap is unchanged and still favours
        the matching code — and nothing is billed, because overlap is not evidence."""
        from claude_coder.resolution import resolve
        line = resolve(_request(self._fact(anchored=False)), self._source())
        self.assertIsNone(line.chosen, line.rationale)
        self.assertEqual({c.code for c in line.alternatives}, {"P_MATCH", "P_NEIGH"})

    def test_support_never_eliminates_terse_code(self):
        # a correct but terse/generic descriptor sharing no tokens with the phrasing
        # must still resolve (support is ranking-only, not a floor).
        # issue #6 F9-R12-E, third re-review: plain retrieval is now
        # verification-required by default -- a stub verifier stands in
        # for the removed no-verifier shortcut.
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder.resolution import resolve
        terse = CandidateCode("P_TERSE", "cpt",
                              "Complete bilateral noninvasive physiologic studies", 0.82)
        src = MockSource(records={("P_TERSE", "cpt"): {"active": True}},
                         retrieval={("*", "cpt"): [terse]})
        fact = ClinicalFact(kind=FactKind.PROCEDURE,
                            description="ankle brachial index with doppler",
                            evidence=[EvidenceSpan("ABI with Doppler waveforms")],
                            confidence=0.9)
        llm = _sv.judge(entails=lambda d: True, reason="entailed")
        line = resolve(_request(fact), src, llm=_from(llm, "provider-a"))
        self.assertEqual(line.chosen.code, "P_TERSE")


class CptAlphabeticIndexTest(unittest.TestCase):
    """The authoritative term->code index layer (AMA CPT Alphabetic Index slot)
    resolves a documented phrase deterministically BEFORE embedding — a plain
    term->code lookup, agnostic to the term."""

    def test_cpt_index_resolves_authoritatively(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        src = MockSource(
            records={("PROC_IDX", "cpt"):
                     {"long_description": "some documented service, unspecified",
                      "active": True}},
            cpt_index={"a documented procedure phrase": {"PROC_IDX"}})
        fact = ClinicalFact(kind=FactKind.PROCEDURE,
                            description="a documented procedure phrase",
                            evidence=[EvidenceSpan("a documented procedure phrase")],
                            confidence=0.95)
        line = resolve(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "PROC_IDX")
        self.assertIn("CPT Alphabetic Index", line.rationale)


class CptIndexParserTest(unittest.TestCase):
    """tools/parse_cpt_index.py: header-driven column detection + range expansion
    to real codes only. Synthetic codes are generated at runtime (no literal code
    cluster), so the parser is exercised without any real medical code."""

    def _mod(self):
        import importlib.util
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "parse_cpt_index", root / "tools" / "parse_cpt_index.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_column_detection(self):
        m = self._mod()
        main_i, mod_i, code_i = m._match_cols(["Main Term", "Modifier", "Code/Range"])
        self.assertEqual((main_i, code_i), (0, 2))
        self.assertEqual(mod_i, [1])

    def test_range_expands_to_valid_only(self):
        m = self._mod()
        base = 90000
        valid = {str(base + i) for i in range(5)}          # generated, not literal
        self.assertEqual(m._expand(f"{base}-{base+3}", valid),
                         [str(base + i) for i in range(4)])
        self.assertEqual(m._expand(f"{base}, {base+2}", valid),
                         [str(base), str(base + 2)])
        self.assertEqual(m._expand("ZZ999", valid), [])    # not a real code -> dropped

    def test_delimiter_sniff(self):
        m = self._mod()
        self.assertEqual(m._sniff("a\tb\tc"), "\t")
        self.assertEqual(m._sniff("a|b|c"), "|")
        self.assertEqual(m._sniff("a,b,c"), ",")


class DrugTableTest(unittest.TestCase):
    """CMS Table of Drugs & Biologicals: a drug NAME resolves to its HCPCS code
    authoritatively, and billing units come from documented dose / per-unit dose."""

    def test_drug_resolves_by_name(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        src = MockSource(
            records={("DRUG_KETO", "hcpcs"):
                     {"long_description": "Injection, substance alpha, per 15 mg",
                      "active": True}},
            drug_index={"substance alpha": {"DRUG_KETO"}})
        fact = ClinicalFact(kind=FactKind.DRUG, description="substance alpha",
                            evidence=[EvidenceSpan("substance alpha 30 mg IV")],
                            confidence=0.95)
        line = resolve(_request(fact), src)
        self.assertEqual(line.method, ResolutionMethod.DETERMINISTIC)
        self.assertEqual(line.chosen.code, "DRUG_KETO")
        self.assertIn("Table of Drugs", line.rationale)

    def test_units_from_documented_dose(self):
        from claude_coder.ontology import drug_billing_units
        self.assertEqual(drug_billing_units("substance 30 mg IV", {"amount": 15, "unit": "mg"}), 2)
        self.assertEqual(drug_billing_units("1 g infused", {"amount": 100, "unit": "mg"}), 10)  # g->mg
        self.assertIsNone(drug_billing_units("two tablets", {"amount": 15, "unit": "mg"}))       # no dose
        self.assertIsNone(drug_billing_units("30 ml", {"amount": 15, "unit": "mg"}))             # unit clash


class DrugEntailmentGateTest(unittest.TestCase):
    """issue #6 F9-R12-E, fourth re-review: Codex's required `_ENTAILMENT_KINDS`
    regressions for DRUG. Before the fix, a Table-of-Drugs hit bypassed
    `_needs_verification` entirely for ANY drug candidate (qualified or not),
    since DRUG was never in the entailment-eligible kinds -- a qualified hit
    (here, a descriptor carrying a bundled-component 'with' clause) closed
    DETERMINISTIC with zero confirmation, no verifier gate at all. See
    `DrugTableTest.test_drug_resolves_by_name` for the companion unqualified
    case, unaffected by this fix (it never needed verification in the first
    place)."""

    def _source(self):
        from claude_coder.data_access import MockSource
        return MockSource(
            records={("DRUG_BETA", "hcpcs"):
                     {"long_description":
                      "Injection, substance beta, with preservative, per 10 mg",
                      "active": True}},
            drug_index={"substance beta": {"DRUG_BETA"}})

    def _fact(self):
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        return ClinicalFact(kind=FactKind.DRUG, description="substance beta",
                            evidence=[EvidenceSpan("substance beta 20 mg IV")],
                            confidence=0.95)

    def test_qualified_drug_hit_without_verifier_stays_a_candidate(self):
        from claude_coder.resolution import resolve
        line = resolve(_request(self._fact()), self._source())
        self.assertIsNone(line.chosen)
        self.assertIn("DRUG_BETA", {c.code for c in (line.alternatives or [])})

    def test_qualified_drug_hit_with_verifier_resolves(self):
        from claude_coder.resolution import resolve
        llm = _sv.judge(entails=lambda d: True, reason="entailed")
        line = resolve(_request(self._fact()), self._source(), llm=llm)
        self.assertIsNotNone(line.chosen)
        self.assertEqual(line.chosen.code, "DRUG_BETA")


class DrugUnitsAvailabilityTest(unittest.TestCase):
    """issue #6 F9-R12-F (Codex): a KNOWN per-unit dose denominator with no
    computable documented dose must never silently bill as 1 unit (the
    count-based default) -- the drug's identity may be right but the billable
    units are not. Enforced as a LINE-level disposition in pipeline.py (never
    an encounter-wide erase: one held drug line must not remove a separate
    defensible line), with `gates.drug_units_gate` as a fail-closed backstop
    for any line that reaches `billable_lines` with `chosen` re-populated some
    other way than that per-line check."""

    GAMMA_DESC = "Injection, substance gamma, per 15 mg"

    def _source(self):
        return MockSource(
            records={("DRUG_GAMMA", "hcpcs"):
                     {"long_description": self.GAMMA_DESC, "active": True}},
            drug_index={"substance gamma": {"DRUG_GAMMA"}})

    def _facts(self, evidence_text):
        import json
        return json.dumps({"facts": [{
            "fact_id": "F1", "kind": "drug", "description": "substance gamma",
            "attributes": {"performer_id": "actor-1", "billing_entity_id": "actor-1"},
            "disposition": "performed_today", "negated": False,
            "evidence": [evidence_text], "confidence": 0.99,
            "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                                "temporal": 0.99, "performer": 0.99,
                                "relationship": 0.99}}]})

    def _run(self, evidence_text, source=None):
        from claude_coder.provenance import NullAuditRepository
        # the note must literally contain the evidence quote verbatim (the
        # source-evidence anchoring check requires this; unrelated to what
        # this test exercises), so the note IS the exact evidence text.
        note = evidence_text + "."
        return code_encounter(
            "enc-drug", note, "2026-03-14",
            source=(source or self._source()),
            extract_llm=lambda s, u: self._facts(evidence_text),
            arbitrate_llm=lambda s, u: '{"choice":0,"confidence":0.0,"reason":"unused"}',
            audit_repository=NullAuditRepository(),
            billing_context={"billing_entity_id": "actor-1", "participants": [
                {"id": "actor-1", "type": "person", "roles": ["performer"]}]})

    def test_documented_compatible_dose_resolves_and_computes_units(self):
        """Codex regression: direct drug hit + documented compatible dose
        resolves and computes units (30 mg documented / 'per 15 mg' = 2)."""
        from claude_coder.models import FactKind
        result = self._run("substance gamma 30 mg administered")
        (line,) = [ln for ln in result.lines if ln.fact.kind is FactKind.DRUG]
        self.assertIsNotNone(line.chosen)
        self.assertEqual(line.chosen.code, "DRUG_GAMMA")
        self.assertEqual(line.units, 2)

    def test_missing_dose_becomes_a_line_level_candidate_not_billed(self):
        """Codex regression: direct drug hit + missing dose becomes a
        line-level candidate with the one missing fact named -- never
        silently billed at 1 unit."""
        from claude_coder.models import FactKind
        result = self._run("substance gamma administered")
        (line,) = [ln for ln in result.lines if ln.fact.kind is FactKind.DRUG]
        self.assertIsNone(line.chosen)
        self.assertIn("DRUG_GAMMA", {c.code for c in (line.alternatives or [])})
        self.assertIn("DRUG_GAMMA", line.documentation_gap or "")
        self.assertNotIn(line, result.billable_lines)

    def test_incompatible_dose_unit_becomes_a_line_level_candidate_not_billed(self):
        """Codex regression: direct drug hit + a documented dose whose unit is
        dimensionally incompatible with the code's denominator (mL vs. mg)
        does the same -- held, not billed on an unconvertible guess."""
        from claude_coder.models import FactKind
        result = self._run("substance gamma 30 mL administered")
        (line,) = [ln for ln in result.lines if ln.fact.kind is FactKind.DRUG]
        self.assertIsNone(line.chosen)
        self.assertIn("DRUG_GAMMA", {c.code for c in (line.alternatives or [])})
        self.assertTrue(line.documentation_gap)

    def test_one_held_drug_line_does_not_erase_a_separate_defensible_line(self):
        """Codex regression: the hold is LINE-level -- a separate, independently
        resolvable procedure in the SAME encounter is untouched and still
        billable."""
        import json
        from claude_coder.models import FactKind
        source = self._source()
        source._records[("PROC_DELTA", "cpt")] = {
            "active": True, "long_description": "Excision, lesion delta"}
        source._cpt_index["excision of lesion delta"] = {"PROC_DELTA"}
        facts = json.dumps({"facts": [
            {"fact_id": "F1", "kind": "drug", "description": "substance gamma",
             "attributes": {"performer_id": "actor-1", "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["substance gamma administered"], "confidence": 0.99,
             "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                                 "temporal": 0.99, "performer": 0.99,
                                 "relationship": 0.99}},
            {"fact_id": "F2", "kind": "procedure",
             "description": "excision of lesion delta",
             "attributes": {"performer_id": "actor-1", "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Excision of lesion delta performed"], "confidence": 0.99,
             "axis_confidence": {"occurrence": 0.99, "action": 0.99, "evidence": 0.99,
                                 "temporal": 0.99, "performer": 0.99,
                                 "relationship": 0.99}},
        ]})
        from claude_coder.provenance import NullAuditRepository
        result = code_encounter(
            "enc-drug2",
            "substance gamma administered. Excision of lesion delta performed.",
            "2026-03-14", source=source, extract_llm=lambda s, u: facts,
            arbitrate_llm=lambda s, u: '{"choice":0,"confidence":0.0,"reason":"unused"}',
            audit_repository=NullAuditRepository(),
            billing_context={"billing_entity_id": "actor-1", "participants": [
                {"id": "actor-1", "type": "person", "roles": ["performer"]}]})
        drug_line = next(ln for ln in result.lines if ln.fact.kind is FactKind.DRUG)
        proc_line = next(ln for ln in result.lines if ln.fact.kind is FactKind.PROCEDURE)
        self.assertIsNone(drug_line.chosen)
        self.assertIsNotNone(proc_line.chosen)
        self.assertEqual(proc_line.chosen.code, "PROC_DELTA")
        self.assertIn(proc_line, result.billable_lines)
        self.assertNotIn(drug_line, result.billable_lines)

    def test_gate_catches_a_manually_constructed_chosen_drug_line(self):
        """Codex regression: the gate is a fail-closed BACKSTOP -- a `chosen`
        drug line built some other way than pipeline.py's own per-line check
        (here, constructed directly) with a known per-unit denominator and no
        computable dose must still BLOCK, never NOT_APPLICABLE."""
        from claude_coder.gates import drug_units_gate
        from claude_coder.models import (ClinicalFact, CodingResult, Disposition,
                                         EvidenceSpan, FactKind, Outcome,
                                         ResolutionMethod, ResolvedLine)
        source = MockSource(
            records={("DRUG_DELTA", "hcpcs"):
                     {"long_description": "Injection, substance delta", "active": True}},
            drug_units={"DRUG_DELTA": {"amount": 15, "unit": "mg"}})
        fact = ClinicalFact(kind=FactKind.DRUG, description="substance delta",
                            disposition=Disposition.PERFORMED,
                            evidence=[EvidenceSpan("substance delta administered")],
                            confidence=0.95)
        line = ResolvedLine(
            fact=fact,
            chosen=CandidateCode("DRUG_DELTA", "hcpcs",
                                 "Injection, substance delta", 1.0,
                                 "cms-table-of-drugs"),
            method=ResolutionMethod.DETERMINISTIC)
        result = CodingResult(encounter_id="enc", date_of_service="2026-03-14",
                              lines=[line])
        gate = drug_units_gate(result, source)
        self.assertEqual(gate.outcome, Outcome.BLOCKED, gate.detail)
        self.assertIn("DRUG_DELTA", gate.detail)


class DrugTableParserTest(unittest.TestCase):
    """tools/build_hcpcs_drug_table.py: a drug code is detected by descriptor
    grammar (substance-amount billing unit), never a code prefix; name + per-unit
    dose parsed out. No real code appears (synthetic descriptors only)."""

    def _mod(self):
        import importlib.util
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "build_hcpcs_drug_table", root / "tools" / "build_hcpcs_drug_table.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_dose_and_name(self):
        m = self._mod()
        self.assertEqual(m._dose_of("Injection, substance alpha, per 15 mg"),
                         (15.0, "mg"))
        self.assertEqual(m._name_of("Injection, substance alpha, per 15 mg"),
                         "substance alpha")

    def test_supply_is_not_a_drug(self):
        m = self._mod()
        # 'each' is not a substance amount -> not a dosed drug
        self.assertIsNone(m._dose_of("Needle-free injection device, each"))


class ProposeVerifyTest(unittest.TestCase):
    """Propose-then-verify: recall is a candidate GENERATOR, the authoritative
    descriptor + entailment is TRUTH. Proposals are validated against the registry;
    a code is accepted only when its official descriptor is entailed by the
    documentation. Uses ABSTRACT near-synonym acts (ALPHA vs BETA) — the mechanism
    is agnostic; it turns on descriptor↔documentation match, not any medical term."""

    # Two descriptors differing only in the ACT primitive (a near-synonym pair).
    ALPHA = "Act alpha of the structure, unspecified approach"
    BETA = "Act beta of the structure, unspecified approach"

    def _src(self):
        from claude_coder.data_access import MockSource
        return MockSource(
            records={("CODEALPHA", "cpt"): {"long_description": self.ALPHA, "active": True},
                     ("CODEBETA", "cpt"): {"long_description": self.BETA, "active": True}},
            # BETA has the HIGHER recall — verify must still reject it for ALPHA.
            retrieval={("*", "cpt"): [CandidateCode("CODEBETA", "cpt", self.BETA, 0.9),
                                      CandidateCode("CODEALPHA", "cpt", self.ALPHA, 0.8)]})

    def _fact(self):
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        return ClinicalFact(kind=FactKind.PROCEDURE,
                            description="act alpha of the structure",
                            evidence=[EvidenceSpan("act alpha performed on the structure")],
                            confidence=0.95)

    def _llm(self, propose=(), entail=True):
        # A shortlist verdict, not a bare pick: the option whose descriptor names the
        # DOCUMENTED act (alpha) is entailed, and the near-synonym (beta) is eliminated
        # WITH a reason -- which is what makes the surviving candidate provably unique.
        return _sv.judge(entails=(lambda d: "alpha" in d.lower() and "beta" not in d.lower()) if entail else (lambda d: False),
                         propose=propose, reason="documented act matches")

    def test_near_synonym_model_pick_does_not_establish_uniqueness(self):
        from claude_coder.resolution import resolve
        line = resolve(_request(self._fact()), self._src(), llm=self._llm())
        self.assertIsNone(line.chosen, line.rationale)
        self.assertEqual({c.code for c in line.alternatives},
                         {"CODEALPHA", "CODEBETA"})

    def test_model_code_proposal_cannot_change_deterministic_candidate_set(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import ResolutionMethod
        from claude_coder.resolution import resolve
        # Retrieval surfaces one code and the model emits another code number.
        # Candidate identity is source-derived now: the model response cannot add
        # a registry record to the decisive universe.
        src = MockSource(
            records={("CODEALPHA", "cpt"): {"long_description": self.ALPHA, "active": True},
                     ("CODEBETA", "cpt"): {"long_description": self.BETA, "active": True}},
            retrieval={("*", "cpt"): [CandidateCode("CODEBETA", "cpt", self.BETA, 0.95)]})
        line = resolve(_request(self._fact()), src,
                       llm=self._llm(propose=["CODEALPHA"]))
        self.assertEqual(line.method, ResolutionMethod.ABSTAINED)
        self.assertIsNone(line.chosen)
        self.assertEqual({c.code for c in line.alternatives}, {"CODEBETA"})
        self.assertNotIn("CODEALPHA", {
            row["code"] for row in (line.candidate_eligibility or [])})

    def test_escalates_when_nothing_entailed(self):
        from claude_coder.models import ResolutionMethod
        from claude_coder.resolution import resolve
        line = resolve(_request(self._fact()), self._src(), llm=self._llm(entail=False))
        self.assertFalse(line.resolved)
        self.assertEqual(line.method, ResolutionMethod.ABSTAINED)
        self.assertIn("verified", line.rationale)

    def test_fabricated_proposal_dropped(self):
        from claude_coder.verify import propose_codes
        cands = propose_codes(self._fact(), self._src(),
                              self._llm(propose=["NOTREAL", "CODEALPHA"]))
        self.assertEqual([c.code for c in cands], ["CODEALPHA"])   # nonexistent code dropped
        self.assertEqual(cands[0].descriptor, self.ALPHA)          # descriptor from the record

    def test_untyped_descriptor_terms_cannot_ground_elimination(self):
        """A raw descriptor-token difference (untyped, no governed axis behind
        it) does not ground an elimination even when the single evaluator
        itself names a reason -- `_grounded_elimination` independently
        confirms every named reason, and a bare wording difference does not
        clear that bar."""
        from claude_coder.models import ResolutionMethod
        from claude_coder.resolution import resolve
        line = resolve(_request(self._fact()), self._src(), llm=self._llm())
        self.assertEqual(line.method, ResolutionMethod.ABSTAINED)
        self.assertIsNone(line.chosen)
        self.assertEqual(line.tie_record["still_entailed"],
                         ["CODEBETA", "CODEALPHA"])

    def test_missing_element_escalates_as_provider_query(self):
        # the evaluator says the code fits but the note omits a required element ->
        # escalate as a provider query, do NOT down-code to something that omits it.
        from claude_coder.models import ResolutionMethod
        from claude_coder.resolution import resolve
        line = resolve(_request(self._fact()), self._src(),
                       llm=_sv.judge(entails=lambda d: False, missing_element=True,
                                    reason="documented act matches"))
        self.assertFalse(line.resolved)
        self.assertEqual(line.method, ResolutionMethod.ABSTAINED)
        self.assertIn("PROVIDER QUERY", line.rationale)


class ProposedCandidateServiceRoleTest(unittest.TestCase):
    """issue #6 F9-R11-H-D: `_service_role_control`'s `blocks_line` backstop
    only ever saw the RETRIEVAL-time candidate universe -- a candidate
    `verify.propose_codes` widens the pool with (from the model's memory,
    not retrieval) never passed through it at all.

    Fourth re-review's fix (a post-selection re-check on `line.chosen`
    alone) was itself incomplete, per the fifth re-review's own adversarial
    reproduction: an incompatible proposal that LOST a verifier tie against
    a compatible retrieved candidate never became `line.chosen` at all, so
    the post-selection check never saw it -- and the tie escalation
    (`chosen=None`) swallowed the compatible candidate right along with it,
    losing an otherwise-defensible first-pass code. Fifth re-review moves
    the check to BEFORE the verification shortlist is built: `resolve()`
    now generates proposals, merges them with the already-eligible
    retrieval pool, and runs role control over that COMPLETE universe
    inside `_propose_then_verify` itself, excluding incompatible candidates
    (or aborting on a genuine multi-role ambiguity) before any verifier
    call is spent. Synthetic codes throughout."""

    OP_DESC = "Operative act alpha on the structure"
    ANES_DESC = "Anesthesia for act alpha on the structure"

    def _src(self, *, op_eligible_at_retrieval: bool, anes_also_retrieved: bool = False):
        from claude_coder.data_access import MockSource
        records = {("OP", "cpt"): {"long_description": self.OP_DESC, "active": True},
                  ("ANES", "cpt"): {"long_description": self.ANES_DESC, "active": True}}
        hits = []
        if op_eligible_at_retrieval:
            hits.append(CandidateCode("OP", "cpt", self.OP_DESC, 0.9))
        if anes_also_retrieved:
            hits.append(CandidateCode("ANES", "cpt", self.ANES_DESC, 0.85))
        retrieval = {("*", "cpt"): hits}
        return MockSource(records=records, retrieval=retrieval,
                          semantic_class={"OP": "surgical_procedure",
                                          "ANES": "anesthesia"})

    def _fact(self):
        # A claim-authorized service_role needs the fully evidenced shape
        # `graph_consensus.claim_authorized_value` requires (scope-valid,
        # source-reconciled, ASSERTED) -- a raw `fact.attributes[...]` write
        # is deliberately NOT trusted on its own; see
        # `test_semantic_eligibility._service_role_fact` for the same
        # construction.
        from claude_coder.models import (AttributeEvidence, ClinicalFact, EvidenceSpan,
                                         FactKind, RelationState)
        span = EvidenceSpan("operative act alpha performed", anchored=True, span_id="s1")
        return ClinicalFact(
            kind=FactKind.PROCEDURE, description="operative act alpha on the structure",
            evidence=[span], confidence=0.95,
            attributes={"service_role": "operative"},
            attribute_evidence={"service_role": (
                AttributeEvidence(span=span, assertion_state=RelationState.ASSERTED,
                                  value="operative"),)})

    def test_a_role_incompatible_model_proposal_is_excluded_before_verification(self):
        """Retrieval surfaces nothing useful; the model proposes the
        anesthesia-role candidate against an operative-role fact. Excluded
        before the verifier ever sees it -- never released."""
        from claude_coder.models import ResolutionMethod
        from claude_coder.resolution import resolve
        src = self._src(op_eligible_at_retrieval=False)
        llm = _sv.judge(entails=lambda d: True, propose=["ANES"], reason="proposed")
        line = resolve(_request(self._fact()), src, llm=_from(llm, "provider-a"))
        self.assertIsNone(line.chosen, line.rationale)
        self.assertNotEqual(line.method, ResolutionMethod.VERIFIED)

    def test_a_compatible_retrieved_candidate_survives_excluding_an_incompatible_proposal(self):
        """Codex's required regression #3, and the exact defect the fourth
        re-review's post-selection-only check missed: excluding the
        incompatible proposal must happen BEFORE the verification
        shortlist is built, so it can never contest (and win) a tie that
        would otherwise swallow an equally-entailed, but role-compatible,
        retrieved candidate along with it. `entails` accepts BOTH
        descriptors -- proving OP wins because ANES was excluded
        pre-verification, not because the verifier happened to prefer it."""
        from claude_coder.resolution import resolve
        src = self._src(op_eligible_at_retrieval=True)
        llm = _sv.judge(entails=lambda d: True, propose=["ANES"], reason="proposed")
        line = resolve(_request(self._fact()), src, llm=_from(llm, "provider-a"))
        self.assertIsNotNone(line.chosen, line.rationale)
        self.assertEqual(line.chosen.code, "OP", line.rationale)

    def test_model_code_proposal_never_enters_candidate_eligibility(self):
        """The model judges a fixed source-derived universe; it cannot write it."""
        from claude_coder.resolution import resolve
        src = self._src(op_eligible_at_retrieval=True)
        llm = _sv.judge(entails=lambda d: "operative" in d.lower(),
                        propose=["ANES"], reason="proposed")
        line = resolve(_request(self._fact()), src, llm=_from(llm, "provider-a"))
        report = {r["code"]: r for r in (line.candidate_eligibility or [])}
        self.assertIn("OP", report, line.candidate_eligibility)
        self.assertNotIn("ANES", report, line.candidate_eligibility)
        self.assertTrue(report["OP"]["eligible"])

    def test_a_first_pass_excluded_retrieval_candidate_remains_audited(self):
        """Codex's required regression #1, retrieval side: ANES is
        RETRIEVED directly (not proposed) and excluded by the FIRST
        eligibility pass in `resolve()`, before `_propose_then_verify` is
        ever called. Passing the already-narrowed pool into
        `_propose_then_verify` silently dropped exactly this candidate from
        what was supposed to be the complete report -- it must still
        appear, with its exact reason, and the compatible candidate must
        still resolve."""
        from claude_coder.resolution import resolve
        src = self._src(op_eligible_at_retrieval=True, anes_also_retrieved=True)
        llm = _sv.judge(entails=lambda d: "operative" in d.lower(), reason="entailed")
        line = resolve(_request(self._fact()), src, llm=_from(llm, "provider-a"))
        report = {r["code"]: r for r in (line.candidate_eligibility or [])}
        self.assertIn("ANES", report, line.candidate_eligibility)
        self.assertFalse(report["ANES"]["eligible"])
        self.assertTrue(report["ANES"].get("reason"), report["ANES"])
        self.assertIsNotNone(line.chosen, line.rationale)
        self.assertEqual(line.chosen.code, "OP", line.rationale)


class UnclassifiedFactRoleServiceConflictTest(unittest.TestCase):
    """issue #6, Codex's independent re-review (F9-R19-A consolidated live-run
    remediation, Finding 3): `blocks_line` used to fire ONLY for
    FACT_ROLE_CONFLICT/MIXED_KIND_INTENT, never for FACT_ROLE_MISSING (no
    fact documents a `service_role` attribute at all) -- so a fact whose OWN
    role could not be determined let a candidate pool spanning genuinely
    incompatible procedure roles (anesthesia and operative-surgery codes
    both surviving) reach the clinical tie-breaker untouched. Reproduced
    directly on the designated operative note: an anesthesia-care fact with
    no classified `service_role` held with candidates including
    gastrocnemius recession, osteotomy, radical resection and total ankle
    replacement -- all operative, not anesthesia, procedures. Synthetic
    descriptors/codes throughout."""

    OP_DESC = "Operative act alpha on the structure"
    ANES_DESC = "Anesthesia for act alpha on the structure"

    def _src(self):
        from claude_coder.data_access import MockSource
        records = {("OP", "cpt"): {"long_description": self.OP_DESC, "active": True},
                  ("ANES", "cpt"): {"long_description": self.ANES_DESC, "active": True}}
        retrieval = {("*", "cpt"): [CandidateCode("OP", "cpt", self.OP_DESC, 0.9),
                                    CandidateCode("ANES", "cpt", self.ANES_DESC, 0.85)]}
        return MockSource(records=records, retrieval=retrieval,
                          semantic_class={"OP": "surgical_procedure",
                                          "ANES": "anesthesia"})

    def _fact_with_no_service_role(self):
        """No `service_role` attribute at all -- `_authorized_roles` finds
        nothing, so `_service_role_control`'s base_status is
        FACT_ROLE_MISSING, never FACT_ROLE_CONFLICT."""
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        span = EvidenceSpan("anesthesia care performed", anchored=True, span_id="s1")
        return ClinicalFact(kind=FactKind.PROCEDURE, description="anesthesia care performed",
                            evidence=[span], confidence=0.95)

    def test_an_unclassified_fact_role_with_multi_role_candidates_reaches_verification(self):
        from claude_coder.resolution import resolve
        src = self._src()
        llm = _sv.judge(entails=lambda d: True, reason="entailed")
        line = resolve(_request(self._fact_with_no_service_role()), src,
                       llm=_from(llm, "provider-a"))
        self.assertIsNotNone(line.candidate_eligibility)
        self.assertTrue(all(not r["role_control"]["blocks_line"]
                            for r in line.candidate_eligibility))
        self.assertNotEqual(line.documentation_gap,
                            "classification_data_gap:service_role_conflict")

    def test_an_unclassified_fact_role_with_only_one_classified_role_still_releases(self):
        """The control must not become OVER-eager: a fact with no
        documented role, but a candidate pool that classifies into only
        ONE distinct procedure role, is not the ambiguity Finding 3
        targets -- it must still release normally."""
        from claude_coder.data_access import MockSource
        from claude_coder.resolution import resolve
        src = MockSource(
            records={("OP", "cpt"): {"long_description": self.OP_DESC, "active": True}},
            retrieval={("*", "cpt"): [CandidateCode("OP", "cpt", self.OP_DESC, 0.9)]},
            semantic_class={"OP": "surgical_procedure"})
        llm = _sv.judge(entails=lambda d: True, reason="entailed")
        line = resolve(_request(self._fact_with_no_service_role()), src,
                       llm=_from(llm, "provider-a"))
        self.assertIsNotNone(line.chosen, line.rationale)
        self.assertEqual(line.chosen.code, "OP")


class SingletonNonProcedureRoleControlTest(unittest.TestCase):
    """The procedure-role backstop is inapplicable to a single non-procedure
    event. This drives the complete resolution entry point so the former live
    ``classification_data_gap:service_role_conflict`` failure cannot recur
    through a caller-boundary mismatch."""

    def test_single_imaging_event_never_gets_a_mixed_kind_service_role_hold(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import CandidateCode, ClinicalFact, EvidenceSpan, FactKind
        from claude_coder.resolution import resolve
        records = {
            ("CAND_OPERATION", "cpt"): {
                "long_description": "operative service", "active": True},
            ("CAND_ANESTHESIA", "cpt"): {
                "long_description": "anesthesia service", "active": True},
        }
        source = MockSource(
            records=records,
            retrieval={("*", "cpt"): [
                CandidateCode("CAND_OPERATION", "cpt", "operative service", 0.9),
                CandidateCode("CAND_ANESTHESIA", "cpt", "anesthesia service", 0.8),
            ]},
            semantic_class={"CAND_OPERATION": "surgical_procedure",
                            "CAND_ANESTHESIA": "anesthesia"})
        fact = ClinicalFact(
            FactKind.IMAGING, "imaging service performed", fact_id="IMG1",
            evidence=[EvidenceSpan("imaging service performed", anchored=True,
                                   span_id="s1")], confidence=0.95)
        line = resolve(_request(fact), source)
        self.assertNotEqual(line.documentation_gap,
                            "classification_data_gap:service_role_conflict")
        self.assertTrue(line.candidate_eligibility)
        self.assertEqual({r["role_control"]["status"]
                          for r in line.candidate_eligibility}, {"not_applicable"})
        self.assertFalse(any(r["role_control"]["blocks_line"]
                             for r in line.candidate_eligibility))


class NonSeparatelyBillableCandidatePreFilterTest(unittest.TestCase):
    """issue #6, Codex's independent re-review (F9-R19-A consolidated
    live-run remediation, Finding 3): `AuthoritativeSource.
    separately_billable` already existed and was already applied to a
    resolved line's FINAL chosen code (`pipeline.py`, post-resolution) --
    but a candidate that never won a tie (both sides stayed "entailed")
    never reached that check, so a non-separately-reportable candidate
    (e.g. an informational quality/performance-measure code) could still
    contest a tie undetected. Reproduced directly on the designated
    operative note: a real implant/supply code (an anchor/screw) tied
    against an unrelated CMS quality-measure code with no shared clinical
    meaning at all. Now applied as a candidate-pool pre-filter, at the same
    stage as the service-role control, before any verifier call. Synthetic
    codes throughout."""

    SUPPLY_DESC = "Anchor/screw for soft tissue-to-bone fixation (implantable)"
    MEASURE_DESC = "Patient had a quality measure assessment performed"

    def test_a_non_separately_billable_candidate_is_excluded_before_verification(self):
        """Direct call to `_propose_then_verify` with an explicit pool --
        the same level `ProposedCandidateServiceRoleTest` above tests the
        sibling service-role pre-filter at -- isolates this specific
        candidate-pool partition from `resolve()`'s own upstream
        deterministic-match dispatch, which is a separate concern."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder import resolution
        src = MockSource(
            records={("SUPPLY", "hcpcs"): {"long_description": self.SUPPLY_DESC,
                                           "active": True},
                    ("MEASURE", "hcpcs"): {"long_description": self.MEASURE_DESC,
                                          "active": True}},
            nonbillable={"MEASURE"})
        span = EvidenceSpan("suture anchors used", anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="suture anchors used",
                            evidence=[span], confidence=0.95, fact_id="f1")
        pool = [CandidateCode("SUPPLY", "hcpcs", self.SUPPLY_DESC, 0.9, "retrieval"),
               CandidateCode("MEASURE", "hcpcs", self.MEASURE_DESC, 0.85, "retrieval")]
        llm = _sv.judge(entails=lambda d: True, reason="entailed")
        line = resolution._propose_then_verify(fact, src, pool, _from(llm, "provider-a"))
        report = {r["code"]: r for r in (line.candidate_eligibility or [])}
        self.assertIn("MEASURE", report)
        self.assertFalse(report["MEASURE"]["eligible"])
        self.assertEqual(report["MEASURE"]["reason"],
                         "not separately reportable per authoritative data")
        self.assertIsNotNone(line.chosen, line.rationale)
        self.assertEqual(line.chosen.code, "SUPPLY", line.rationale)


class CandidateKindControlTest(unittest.TestCase):
    """issue #6, Codex's independent re-review (F9-R21-C): a candidate's own
    authoritative `semantic_class()` categorically incompatible with a
    documented non-procedure fact (quality-measure, E&M, anesthesia-status)
    is excluded from the shortlist before verification -- reproduced live: a
    suture-anchor SUPPLY fact's candidate pool retained an unrelated
    quality-measure candidate as "entailed". Synthetic codes throughout."""

    SUPPLY_DESC = "Anchor/screw for soft tissue-to-bone fixation (implantable)"
    MEASURE_DESC = "Patient had a quality measure assessment performed"

    def test_a_quality_measure_candidate_is_excluded_from_a_supply_facts_pool(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder import resolution
        src = MockSource(
            records={("SUPPLY", "hcpcs"): {"long_description": self.SUPPLY_DESC,
                                           "active": True},
                    ("MEASURE", "hcpcs"): {"long_description": self.MEASURE_DESC,
                                          "active": True}},
            semantic_class={"MEASURE": "performance_measure_tracking"})
        span = EvidenceSpan("suture anchors used", anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.SUPPLY, description="suture anchors used",
                            evidence=[span], confidence=0.95, fact_id="f1")
        pool = [CandidateCode("SUPPLY", "hcpcs", self.SUPPLY_DESC, 0.9, "retrieval"),
               CandidateCode("MEASURE", "hcpcs", self.MEASURE_DESC, 0.85, "retrieval")]
        llm = _sv.judge(entails=lambda d: True, reason="entailed")
        line = resolution._propose_then_verify(fact, src, pool, _from(llm, "provider-a"))
        report = {r["code"]: r for r in (line.candidate_eligibility or [])}
        self.assertIn("MEASURE", report)
        self.assertFalse(report["MEASURE"]["eligible"])
        self.assertIn("performance_measure_tracking", report["MEASURE"]["reason"])
        self.assertIsNotNone(line.chosen, line.rationale)
        self.assertEqual(line.chosen.code, "SUPPLY", line.rationale)

    def test_a_surgical_candidate_is_excluded_from_an_imaging_facts_pool(self):
        """Related procedure context may support a descriptor element, but it must
        never change the target event's typed kind.  A surgery-classified candidate
        therefore cannot be selected for an imaging event merely because its text
        mentions the operation being imaged."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder import semantic_eligibility as semelig
        src = MockSource(semantic_class={"OPERATION": "surgical_procedure"})
        fact = ClinicalFact(
            kind=FactKind.IMAGING,
            description="imaging used to confirm completion of the operation",
            evidence=[EvidenceSpan(
                "imaging confirmed completion of the operation", anchored=True,
                span_id="s1")], confidence=0.95, fact_id="IMG1")
        candidates = [CandidateCode(
            "OPERATION", "cpt", "operative removal service", 0.9, "retrieval")]

        excluded = semelig._candidate_kind_control([fact], candidates, src, None)

        self.assertIn(("OPERATION", "cpt"), excluded)
        self.assertIn("surgical_procedure", excluded[("OPERATION", "cpt")])

    def test_exact_descriptor_hit_cannot_bypass_kind_control(self):
        """The authoritative-index shortcut and broad recall must enforce one
        target-kind boundary; an exact hit is a candidate lead, not permission to
        change an imaging event into an operation."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder.resolution import resolve
        source = MockSource(
            records={("OPERATION", "cpt"): {
                "long_description": "operative removal service", "active": True}},
            proc_index={"imaging used to confirm operation": {"OPERATION"}},
            semantic_class={"OPERATION": "surgical_procedure"})
        fact = ClinicalFact(
            kind=FactKind.IMAGING,
            description="imaging used to confirm operation", fact_id="IMG1",
            evidence=[EvidenceSpan(
                "imaging used to confirm operation", anchored=True, span_id="s1")],
            confidence=0.95)

        line = resolve(_request(fact), source)

        self.assertIsNone(line.chosen)
        report = {r["code"]: r for r in line.candidate_eligibility}
        self.assertFalse(report["OPERATION"]["eligible"])
        self.assertIn("surgical_procedure", report["OPERATION"]["reason"])

    def test_a_procedure_facts_pool_is_untouched_by_the_kind_control(self):
        """`_service_role_control` (operative vs. anesthesia) already owns a
        procedure fact's own role distinction -- this supplementary control
        must not ALSO fire for a procedure fact, even one whose pool happens
        to contain a candidate classified into one of these classes."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder import semantic_eligibility as semelig
        src = MockSource(semantic_class={"E_M_CODE": "evaluation_management"})
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="procedure performed",
                            evidence=[EvidenceSpan("x", anchored=True, span_id="s1")],
                            confidence=0.9, fact_id="f1")
        candidates = [CandidateCode("E_M_CODE", "cpt", "d", 0.9)]
        self.assertEqual(
            semelig._candidate_kind_control([fact], candidates, src, None), {})

    def test_no_semantic_class_support_is_a_silent_no_op(self):
        """Unlike `_service_role_control`, an unavailable classifier here is
        not itself grounds to fail closed -- this is a supplementary safety
        net layered on top of that control, not the primary one."""
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder import semantic_eligibility as semelig

        class _NoSemanticClassSource:
            pass

        fact = ClinicalFact(kind=FactKind.SUPPLY, description="supply used",
                            evidence=[EvidenceSpan("x", anchored=True, span_id="s1")],
                            confidence=0.9, fact_id="f1")
        candidates = [CandidateCode("X", "hcpcs", "d", 0.9)]
        self.assertEqual(semelig._candidate_kind_control(
            [fact], candidates, _NoSemanticClassSource(), None), {})


class CandidateAdmissionTest(unittest.TestCase):
    """issue #6, Codex's independent re-review (F9-R23 Root Finding 1):
    `resolution.candidate_admission` -- the pre-verification admission
    standing "broad retrieval supplies recall; positive evidence supplies
    standing; complete candidate requirements authorize selection."
    Synthetic codes throughout."""

    def test_a_role_incompatible_candidate_is_contradicted(self):
        # A claim-authorized service_role needs the fully evidenced shape
        # `graph_consensus.claim_authorized_value` requires (scope-valid,
        # source-reconciled, ASSERTED) -- mirrors
        # RoleControlBackstopTest._fact above.
        from claude_coder.data_access import MockSource
        from claude_coder.models import (AttributeEvidence, ClinicalFact, EvidenceSpan,
                                         FactKind, RelationState)
        from claude_coder import resolution
        src = MockSource(semantic_class={"ANESTH": "anesthesia",
                                        "OP": "surgical_procedure"})
        span = EvidenceSpan("operative act alpha performed", anchored=True, span_id="s1")
        fact = ClinicalFact(
            kind=FactKind.PROCEDURE, description="operative act alpha on the structure",
            evidence=[span], confidence=0.95, fact_id="f1",
            attributes={"service_role": "operative"},
            attribute_evidence={"service_role": (
                AttributeEvidence(span=span, assertion_state=RelationState.ASSERTED,
                                  value="operative"),)})
        candidate = CandidateCode("ANESTH", "cpt", "anesthesia service", 0.9, "retrieval")
        admission = resolution.candidate_admission(fact, candidate, (), src, None, None, None)
        self.assertEqual(admission.standing, resolution.CandidateStanding.CONTRADICTED)
        self.assertIn("service_role", admission.contradicted_axes)

    def test_a_topically_unrelated_recall_hit_is_ungrounded(self):
        """The named Root Finding 1 reproduction: a candidate whose own
        descriptor shares NOTHING with the documented fact -- no compiled
        requirement, no direct-term/authoritative-index/UMLS lineage, no
        topical overlap -- has no positive standing of its own."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder import resolution
        src = MockSource()
        fact = ClinicalFact(kind=FactKind.SUPPLY, description="suture anchor implanted",
                            evidence=[EvidenceSpan("suture anchor implanted",
                                                   anchored=True, span_id="s1")],
                            confidence=0.95, fact_id="f1")
        candidate = CandidateCode("Q999", "hcpcs",
                                  "telehealth originating site facility fee", 0.4,
                                  "retrieval")
        admission = resolution.candidate_admission(fact, candidate, (), src, None, None, None)
        self.assertEqual(admission.standing, resolution.CandidateStanding.UNGROUNDED)
        self.assertEqual(admission.positive_axes, ())

    def test_topical_overlap_is_recall_only_not_candidate_standing(self):
        """Shared words are recall, not proof that a candidate identifies
        the documented service. Entailment may promote it later."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder import resolution
        src = MockSource()
        fact = ClinicalFact(kind=FactKind.SUPPLY, description="manual repair structure alpha",
                            evidence=[EvidenceSpan("manual repair structure alpha",
                                                   anchored=True, span_id="s1")],
                            confidence=0.95, fact_id="f1")
        candidate = CandidateCode("ANCHOR1", "hcpcs",
                                  "powered repair structure beta", 0.9, "retrieval")
        admission = resolution.candidate_admission(fact, candidate, (), src, None, None, None)
        self.assertEqual(admission.standing, resolution.CandidateStanding.UNGROUNDED)
        self.assertEqual(admission.positive_axes, ())

    def test_a_direct_term_hit_is_positive_identity(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder import resolution

        class _DirectTermSource(MockSource):
            def exact_direct_code_term(self, term):
                return [{"code": "ANCHOR1", "term": term}] if term == "suture anchor" else []

        src = _DirectTermSource()
        fact = ClinicalFact(kind=FactKind.SUPPLY, description="suture anchor",
                            evidence=[EvidenceSpan("suture anchor", anchored=True,
                                                   span_id="s1")],
                            confidence=0.95, fact_id="f1")
        candidate = CandidateCode("ANCHOR1", "hcpcs", "suture anchor, implantable", 0.9,
                                  "retrieval")
        admission = resolution.candidate_admission(fact, candidate, (), src, None, None, None)
        self.assertEqual(admission.standing, resolution.CandidateStanding.SUPPORTED)
        self.assertIn("direct_term", admission.positive_axes)

    def test_registry_validated_and_umls_recall_sources_do_not_create_standing(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder import resolution
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="service alpha",
                            evidence=[EvidenceSpan("service alpha", anchored=True,
                                                   span_id="s1")], fact_id="f1")
        for source_name, authority in (
                ("llm-proposed-validated", {}),
                ("retrieval", {"sources": ["umls_recall"]})):
            with self.subTest(source=source_name):
                candidate = CandidateCode("SYN", "cpt", "service alpha", 0.9,
                                          source_name, authority=authority)
                admission = resolution.candidate_admission(
                    fact, candidate, (), MockSource(), None, None, None)
                self.assertEqual(admission.standing,
                                 resolution.CandidateStanding.UNGROUNDED)

    def test_raw_descriptor_term_support_is_recall_only_not_identity(self):
        """An exact leftover descriptor word is not a governed concept and
        therefore cannot independently promote a candidate."""
        import hashlib
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder import requirement, resolution
        text = "assembly service in mode alpha"
        fact = ClinicalFact(
            kind=FactKind.PROCEDURE, description=text,
            evidence=[EvidenceSpan(text, anchored=True, span_id="s1")], fact_id="f1")
        candidate = CandidateCode(
            "CAND_ALPHA", "cpt", "assembly service, mode alpha", 0.9, "retrieval")
        sibling = CandidateCode(
            "CAND_BETA", "cpt", "assembly service, mode beta", 0.9, "retrieval")
        requirements = requirement.compile_requirements([candidate, sibling])
        coverage = requirement.CoverageCorpus(
            channel_id="test-channel", text=text,
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            covered_pages=(1,), page_image_sha256=("stub-hash",))

        admission = resolution.candidate_admission(
            fact, candidate, requirements, MockSource(), None, coverage, None)

        self.assertEqual(admission.standing, resolution.CandidateStanding.UNGROUNDED)
        self.assertIn("descriptor_term", admission.positive_axes)


class ProposedCandidateDeterministicExclusionTest(unittest.TestCase):
    """issue #6 F9-R11-H-D, seventh re-review: `_evaluate` returns a bare
    `None` (no reason) for two DETERMINISTIC eliminations -- an explicit
    laterality contradiction, and a documented measurement outside the
    descriptor's bounded interval. A registry-valid model PROPOSAL eliminated
    this way used to vanish before `_propose_then_verify`'s own "complete"
    candidate universe was even built, so the ClaimBundle audit trail could
    say neither which candidate was excluded nor why. `_evaluate_reason`
    (a reason-carrying sibling of `_evaluate`, which stays unchanged for its
    other five callers) closes this. Synthetic codes throughout."""

    def test_a_laterality_contradictory_model_code_is_not_a_candidate(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (AttributeEvidence, ClinicalFact, EvidenceSpan,
                                         FactKind, RelationState)
        from claude_coder.resolution import resolve
        compatible = CandidateCode("RIGHT", "cpt", "act alpha, right side", 0.9)
        src = MockSource(
            records={("RIGHT", "cpt"): {"long_description": "act alpha, right side",
                                        "active": True},
                    ("LEFT", "cpt"): {"long_description": "act alpha, left side",
                                      "active": True}},
            retrieval={("*", "cpt"): [compatible]})
        # laterality needs the fully evidenced, claim-authorized shape
        # `_fact_laterality`/`claim_authorized_value` requires -- a raw
        # `attributes[...]` write alone is not trusted (same construction as
        # `OntologyResolutionTest.test_laterality_contradiction_eliminated`).
        span = EvidenceSpan("act alpha performed on the right side",
                            anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="act alpha, right side",
                            attributes={"laterality": "right"}, evidence=[span],
                            attribute_evidence={"laterality": (
                                AttributeEvidence(span=span, assertion_state=RelationState.ASSERTED,
                                                  value="right"),)},
                            confidence=0.95)
        llm = _sv.judge(entails=lambda d: "right" in d.lower(),
                        propose=["LEFT"], reason="proposed")
        line = resolve(_request(fact), src, llm=_from(llm, "provider-a"))
        report = {r["code"]: r for r in (line.candidate_eligibility or [])}
        self.assertNotIn("LEFT", report, line.candidate_eligibility)
        self.assertIsNotNone(line.chosen, line.rationale)
        self.assertEqual(line.chosen.code, "RIGHT", line.rationale)

    def test_an_out_of_range_model_code_is_not_a_candidate(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder.resolution import resolve
        compatible = CandidateCode(
            "UNBOUNDED", "cpt", "wound dressing, sterile, each", 0.9)
        src = MockSource(
            records={("UNBOUNDED", "cpt"): {"long_description":
                                            "wound dressing, sterile, each",
                                            "active": True},
                    ("SMALL", "cpt"): {"long_description": "wound dressing, sterile, "
                                                           "size 16 sq. in. or less, each",
                                      "active": True}},
            retrieval={("*", "cpt"): [compatible]})
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="wound dressing",
                            attributes={"size_sqin": 60},
                            evidence=[EvidenceSpan("wound dressing 60 sq in applied")],
                            confidence=0.95)
        llm = _sv.judge(entails=lambda d: True, propose=["SMALL"], reason="proposed")
        line = resolve(_request(fact), src, llm=_from(llm, "provider-a"))
        report = {r["code"]: r for r in (line.candidate_eligibility or [])}
        self.assertNotIn("SMALL", report, line.candidate_eligibility)
        self.assertIsNotNone(line.chosen, line.rationale)
        self.assertEqual(line.chosen.code, "UNBOUNDED", line.rationale)


class ProposedAndRetrievedSameCodeExclusionTest(unittest.TestCase):
    """issue #6 F9-R11-H-D, eighth re-review: the SAME (code, system) can arrive
    through BOTH retrieval and the model proposal at once. `_evaluate_reason`
    deterministically excludes it either way, but the merge used to SKIP a
    proposal's exclusion whenever `candidate_eligibility` already carried a
    record for that identity (from retrieval, typically eligible=True) --
    reproduced directly: 'SMALL' retrieved as bounded/unbounded alongside
    'UNBOUNDED', also proposed, selection lands correctly on 'UNBOUNDED' via
    `_ranked` dropping 'SMALL' independently, but the ClaimBundle audit kept
    SMALL's stale `eligible: true, reason: null` retrieval record. Fixed by
    reconciling the exclusion against candidate IDENTITY, overriding any
    existing record for that (code, system) rather than deferring to it.
    Synthetic codes throughout."""

    def test_a_laterality_contradictory_code_present_in_both_retrieval_and_proposal_is_excluded(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (AttributeEvidence, ClinicalFact, EvidenceSpan,
                                         FactKind, RelationState)
        from claude_coder.resolution import resolve
        compatible = CandidateCode("RIGHT", "cpt", "act alpha, right side", 0.9)
        # LEFT is RETRIEVED too, not just proposed -- eligibility_report already
        # produces an (eligible=True) record for it before the proposal's own
        # deterministic exclusion is merged in.
        contradicted_retrieved = CandidateCode("LEFT", "cpt", "act alpha, left side", 0.5)
        src = MockSource(
            records={("RIGHT", "cpt"): {"long_description": "act alpha, right side",
                                        "active": True},
                    ("LEFT", "cpt"): {"long_description": "act alpha, left side",
                                      "active": True}},
            retrieval={("*", "cpt"): [compatible, contradicted_retrieved]})
        span = EvidenceSpan("act alpha performed on the right side",
                            anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="act alpha, right side",
                            attributes={"laterality": "right"}, evidence=[span],
                            attribute_evidence={"laterality": (
                                AttributeEvidence(span=span, assertion_state=RelationState.ASSERTED,
                                                  value="right"),)},
                            confidence=0.95)
        # LEFT is ALSO the model's proposal -- same identity from both sources.
        llm = _sv.judge(entails=lambda d: "right" in d.lower(),
                        propose=["LEFT"], reason="proposed")
        line = resolve(_request(fact), src, llm=_from(llm, "provider-a"))
        report = {r["code"]: r for r in (line.candidate_eligibility or [])}
        self.assertIn("LEFT", report, line.candidate_eligibility)
        self.assertFalse(report["LEFT"]["eligible"], report["LEFT"])
        self.assertIsNotNone(report["LEFT"].get("reason"), report["LEFT"])
        self.assertIn("laterality", report["LEFT"]["reason"].lower(), report["LEFT"])
        self.assertIsNotNone(line.chosen, line.rationale)
        self.assertEqual(line.chosen.code, "RIGHT", line.rationale)

    def test_an_out_of_range_retrieval_candidate_is_not_promoted_by_model_proposal(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        from claude_coder.resolution import resolve
        compatible = CandidateCode("UNBOUNDED", "cpt", "wound dressing, sterile, each", 0.9)
        contradicted_retrieved = CandidateCode(
            "SMALL", "cpt", "wound dressing, sterile, size 16 sq. in. or less, each", 0.5)
        src = MockSource(
            records={("UNBOUNDED", "cpt"): {"long_description":
                                            "wound dressing, sterile, each",
                                            "active": True},
                    ("SMALL", "cpt"): {"long_description": "wound dressing, sterile, "
                                                           "size 16 sq. in. or less, each",
                                      "active": True}},
            retrieval={("*", "cpt"): [compatible, contradicted_retrieved]})
        fact = ClinicalFact(kind=FactKind.PROCEDURE, description="wound dressing",
                            attributes={"size_sqin": 60},
                            evidence=[EvidenceSpan("wound dressing 60 sq in applied")],
                            confidence=0.95)
        llm = _sv.judge(entails=lambda d: True, propose=["SMALL"], reason="proposed")
        line = resolve(_request(fact), src, llm=_from(llm, "provider-a"))
        report = {r["code"]: r for r in (line.candidate_eligibility or [])}
        self.assertIn("SMALL", report, line.candidate_eligibility)
        # Candidate eligibility is role/kind compatibility; the later
        # descriptor-interval contract performs the measurement exclusion.
        self.assertTrue(report["SMALL"]["eligible"], report["SMALL"])
        snapshot = (line.tie_record or {}).get("candidate_set") or {}
        self.assertEqual(
            [row["code"] for row in snapshot.get("candidates", [])],
            ["UNBOUNDED", "SMALL"])
        self.assertIsNotNone(line.chosen, line.rationale)
        self.assertEqual(line.chosen.code, "UNBOUNDED", line.rationale)


class ModelProfileIdentityTest(unittest.TestCase):
    """The recorded model identity must describe the RUN, not this function's assumptions —
    it is what an auditor reads to check the independence claim."""

    def test_default_callables_are_declared_cross_provider(self):
        from claude_coder import verify
        self.assertEqual(verify.model_profile_of(verify.default_verify_llm)["provider"],
                         verify.VERIFY_PROVIDER)
        self.assertEqual(verify.model_profile_of(verify.default_corroborate_llm)["provider"],
                         verify.CORROBORATE_PROVIDER)
        self.assertEqual(
            verify.corroboration_origin(verify.default_verify_llm,
                                        verify.default_corroborate_llm),
            verify.DISTINCT_ORIGIN)

    def test_identity_reports_the_actual_callables_not_a_fixed_pair(self):
        from claude_coder import verify
        from claude_coder.pipeline import _model_profile_identity
        same = _model_profile_identity(None, _from(lambda s, u: "", "claude"),
                                       _from(lambda s, u: "", "claude"))
        self.assertIs(same["independent_providers"], False)
        self.assertEqual(same["corroboration_origin"], verify.SHARED_ORIGIN)

        cross = _model_profile_identity(None, _from(lambda s, u: "", "provider-a"),
                                        _from(lambda s, u: "", "provider-b"))
        self.assertIs(cross["independent_providers"], True)
        self.assertEqual(cross["corroboration_origin"], verify.DISTINCT_ORIGIN)

    def test_absent_corroborator_is_not_independent(self):
        from claude_coder import verify
        from claude_coder.pipeline import _model_profile_identity
        p = _model_profile_identity(None, _from(lambda s, u: "", "provider-a"), None)
        self.assertIs(p["independent_providers"], False)
        self.assertEqual(p["corroboration_origin"], verify.NO_CORROBORATION)

    def test_every_pairing_maps_to_a_known_origin_and_only_one_releases(self):
        """The status vocabulary is closed and exactly one member is creditable, so a new
        value can never be added without a reviewer deciding which side of the line it is
        on."""
        from claude_coder import verify
        undeclared = lambda s, u: ""
        pairs = [(None, None), (undeclared, None), (undeclared, undeclared),
                 (_from(lambda s, u: "", "x"), _from(lambda s, u: "", "x")),
                 (_from(lambda s, u: "", "x"), _from(lambda s, u: "", "y")),
                 (_from(lambda s, u: "", "x"), undeclared)]
        creditable = set()
        for primary, second in pairs:
            origin = verify.corroboration_origin(primary, second)
            self.assertIn(origin, verify.CORROBORATION_ORIGINS)
            if origin in verify.INDEPENDENT_CORROBORATION_ORIGINS:
                creditable.add(origin)
        self.assertEqual(creditable, {verify.DISTINCT_ORIGIN})
        self.assertEqual(verify.INDEPENDENT_CORROBORATION_ORIGINS,
                         frozenset({verify.DISTINCT_ORIGIN}))

    def test_extraction_provider_overlap_is_recorded_observationally(self):
        """Recorded so the weaker correlation is visible in the artifact, but it is NOT a
        control input — a shared extraction provider alone still leaves the corroboration
        independent."""
        from claude_coder import verify
        from claude_coder.pipeline import _model_profile_identity
        p = _model_profile_identity(None, _from(lambda s, u: "", "provider-a"),
                                    _from(lambda s, u: "", "provider-b"))
        self.assertIn("corroborator_shares_extraction_provider", p)
        self.assertEqual(p["corroboration_origin"], verify.DISTINCT_ORIGIN)
        self.assertIs(p["independent_providers"], True)

    # ---- issue #6 F7-R5: the identity must describe the call that is MADE -----------
    def test_a_supplied_extractor_is_recorded_as_itself_not_as_the_configuration(self):
        """A caller-supplied extractor used to be stamped with the configured provider,
        which made every comparison against it -- including the two-reading independence
        fact -- a statement about configuration rather than about the run."""
        from claude_coder.pipeline import _model_profile_identity
        p = _model_profile_identity(_from(lambda s, u: "", "provider-a"), None, None,
                                    _from(lambda s, u: "", "provider-b"))
        self.assertEqual(p["extraction"]["provider"], "provider-a")
        self.assertEqual(p["second_extraction"]["provider"], "provider-b")

    def test_an_undeclared_supplied_extractor_claims_no_provider(self):
        """Fail-closed: "we cannot tell who read the note" is not "the configured
        vendor read the note"."""
        from claude_coder.pipeline import _model_profile_identity
        p = _model_profile_identity(lambda s, u: "", None, None)
        self.assertEqual(p["extraction"].get("provider", ""), "")

    def test_the_pipelines_own_second_reading_is_still_identified_by_configuration(self):
        """The control case: when the PIPELINE makes the call, configuration is exactly
        what selects the model, so it remains the identity."""
        from app.core import config
        from claude_coder.pipeline import _model_profile_identity
        p = _model_profile_identity(None, None, None)
        self.assertEqual(p["extraction"]["provider"], config.LLM_PROVIDER)


class SecondReadingIndependenceTest(unittest.TestCase):
    """The two-reading control fails closed when it is not, in fact, independent.

    `independent_providers` used to be computed after both readings had been paid for
    and then only recorded -- so a deployment whose two readings resolved to ONE vendor
    produced an artifact asserting an independence the run never had (issue #6 F7-R5).
    """

    def _profiles(self, primary, second):
        return {"extraction": {"provider": primary},
                "second_extraction": {"provider": second}}

    def test_a_same_vendor_pair_is_refused_before_the_reading_is_taken(self):
        from claude_coder.extraction import SecondReadingUnavailable
        from claude_coder.pipeline import _run_graph_consensus
        calls = []
        with self.assertRaises(SecondReadingUnavailable):
            _run_graph_consensus("note", [], None,
                                 lambda system, user: calls.append(1) or "{}",
                                 self._profiles("claude", "claude"),
                                 None, None, None, enforce_independence=True)
        self.assertEqual(calls, [],
                         "a control that cannot be independent must not be paid for")

    def test_an_undeclared_pair_is_refused_too(self):
        from claude_coder.extraction import SecondReadingUnavailable
        from claude_coder.pipeline import _run_graph_consensus
        with self.assertRaises(SecondReadingUnavailable):
            _run_graph_consensus("note", [], None, lambda system, user: "{}",
                                 self._profiles("claude", ""),
                                 None, None, None, enforce_independence=True)

    def test_a_caller_supplied_second_reading_is_recorded_not_enforced(self):
        """A second extractor a CALLER supplied is a disagreement detector, whose value
        does not depend on vendor independence. It is recorded, never refused."""
        from claude_coder.pipeline import _run_graph_consensus

        class _Reached(Exception):
            pass

        def _extract(system, user):
            raise _Reached

        with self.assertRaises(_Reached):
            _run_graph_consensus("note", [], None, _extract,
                                 self._profiles("claude", "claude"),
                                 None, None, None, enforce_independence=False)


class LearnedIndexTest(unittest.TestCase):
    """The learned verified-resolution index promotes a phrase->code mapping to
    deterministic trust only when confirmed across >= PROMOTE_AT DISTINCT encounters
    and unambiguous — the automated gate (no human sign-off). Synthetic codes."""

    def _obs(self, phrase, code, enc):
        return {"phrase": phrase, "code": code, "system": "cpt",
                "descriptor": "d", "evidence": ["e"], "enc": enc}

    def test_promote_on_distinct_encounters(self):
        from claude_coder import learned
        obs = [self._obs("phrase one", "PROC_A", f"n{i}") for i in range(3)]
        entries = learned.promote(obs, promote_at=3)
        self.assertIn("phrase one", entries)
        self.assertEqual(entries["phrase one"]["code"], "PROC_A")
        self.assertEqual(entries["phrase one"]["encounters"], 3)

    def test_no_promote_under_threshold(self):
        from claude_coder import learned
        obs = [self._obs("phrase two", "PROC_A", f"n{i}") for i in range(2)]
        self.assertEqual(learned.promote(obs, promote_at=3), {})

    def test_dedup_by_encounter(self):
        from claude_coder import learned
        # same encounter observed 3x is ONE vote — cannot self-promote
        obs = [self._obs("phrase three", "PROC_A", "same") for _ in range(3)]
        self.assertEqual(learned.promote(obs, promote_at=3), {})

    def test_no_promote_when_contested(self):
        from claude_coder import learned
        # PROC_A in 3 encounters, PROC_B in 2 -> 3 < 2*2, ambiguous -> no promotion
        obs = ([self._obs("phrase four", "PROC_A", f"a{i}") for i in range(3)]
               + [self._obs("phrase four", "PROC_B", f"b{i}") for i in range(2)])
        self.assertEqual(learned.promote(obs, promote_at=3), {})

    def test_promote_when_dominant(self):
        from claude_coder import learned
        # PROC_A in 4 encounters, PROC_B in 1 -> 4 >= 2*1 -> promote PROC_A
        obs = ([self._obs("phrase five", "PROC_A", f"a{i}") for i in range(4)]
               + [self._obs("phrase five", "PROC_B", "b0")])
        entries = learned.promote(obs, promote_at=3)
        self.assertEqual(entries.get("phrase five", {}).get("code"), "PROC_A")

    def test_load_observations_roundtrip(self):
        import json
        import tempfile
        from pathlib import Path
        from claude_coder import learned
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "obs.jsonl"
            p.write_text("\n".join(json.dumps(self._obs("p", "PROC_A", f"n{i}"))
                                   for i in range(3)) + "\n")
            self.assertEqual(len(learned.load_observations(p)), 3)

    def test_entry_self_invalidates_on_descriptor_change(self):
        # entry_current is pure normalized-string equality — no domain knowledge.
        # Abstract inputs prove the generic property: same descriptor stays valid,
        # ANY change invalidates, absent text is trusted.
        from claude_coder import learned
        e = {"descriptor": "alpha beta gamma"}
        self.assertTrue(learned.entry_current(e, "Alpha  Beta,  gamma"))   # same up to normalization
        self.assertFalse(learned.entry_current(e, "alpha beta delta"))     # any change -> invalid
        self.assertTrue(learned.entry_current(e, ""))                      # current unknown -> trust
        self.assertTrue(learned.entry_current({"descriptor": ""}, "x"))    # nothing stored -> trust

    def test_learned_index_is_recall_only_not_deterministic(self):
        # Fix5: a learned phrase->code mapping is a RECALL candidate, never a privileged
        # deterministic bill (its key lacks clinical context and its freshness check
        # fails open). With an LLM verifier that does NOT confirm entailment, the learned
        # hit must ESCALATE — proving it lost deterministic trust.
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        src = MockSource(
            records={("PROC_X", "cpt"):
                     {"long_description": "some documented service, unspecified",
                      "active": True}},
            learned_index={"a documented service phrase": "PROC_X"})
        fact = ClinicalFact(kind=FactKind.PROCEDURE,
                            description="a documented service phrase",
                            evidence=[EvidenceSpan("a documented service phrase")],
                            confidence=0.95)

        reject = _sv.judge(entails=lambda d: False, reason="none entailed")

        line = resolve(_request(fact), src, llm=reject)
        self.assertFalse(line.resolved)                       # not billed on learned trust
        self.assertEqual(line.method, ResolutionMethod.ABSTAINED)
        self.assertNotIn("learned verified-resolution index", line.rationale)


class RecommendationsTest(unittest.TestCase):
    """Documentation recommendations are derived agnostically from fact kinds,
    resolution methods, and gate outcomes — no code/term/scenario. Abstract inputs."""

    def _line(self, resolved, doc_gap=None, rationale="r", confidence=0.99):
        # `confidence` is explicit because it is claim-affecting here: a resolved line
        # the note BARELY documents is not a clean line, and it now earns a
        # `documentation_clarity` recommendation (the suggested action behind the
        # PROVIDER_QUERY `autonomy.decide` routes it to). The default is a clearly
        # well-documented fact so "resolved" means resolved.
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod, ResolvedLine)
        f = ClinicalFact(kind=FactKind.PROCEDURE, description="a documented service",
                         evidence=[EvidenceSpan("a documented service performed")],
                         confidence=confidence)
        chosen = CandidateCode("PROC_X", "cpt", "d", 0.9) if resolved else None
        return ResolvedLine(fact=f, chosen=chosen,
                            method=(ResolutionMethod.VERIFIED if resolved
                                    else ResolutionMethod.ABSTAINED),
                            documentation_gap=doc_gap, rationale=rationale)

    def test_documentation_gap_becomes_query(self):
        from claude_coder.models import CodingResult
        from claude_coder.recommendations import build_recommendations
        ln = self._line(resolved=False, doc_gap="a required element was not stated")
        recs = build_recommendations(
            CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[ln]))
        self.assertEqual([r["issue"] for r in recs], ["documentation_gap"])
        self.assertIn("a required element was not stated", recs[0]["recommendation"])

    def test_unresolved_service_recommendation(self):
        from claude_coder.models import CodingResult
        from claude_coder.recommendations import build_recommendations
        ln = self._line(resolved=False)                  # abstained, no doc gap
        recs = build_recommendations(
            CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[ln]))
        self.assertEqual([r["issue"] for r in recs], ["unresolved_service"])
        self.assertIn("clarify", recs[0]["recommendation"].lower())

    def test_gate_block_becomes_remediation(self):
        from claude_coder.models import CodingResult, GateResult, Outcome
        from claude_coder.recommendations import build_recommendations
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         gates=[GateResult("verbatim_evidence", Outcome.BLOCKED, "x", "y")])
        recs = build_recommendations(r)
        self.assertEqual([x["issue"] for x in recs], ["gate_verbatim_evidence"])

    def test_resolved_line_yields_no_recommendation(self):
        from claude_coder.models import CodingResult
        from claude_coder.recommendations import build_recommendations
        recs = build_recommendations(
            CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[self._line(resolved=True)]))
        self.assertEqual(recs, [])

    def test_a_barely_documented_resolved_line_still_gets_a_suggested_action(self):
        """The companion of the case above, and the gap this phase's post-fix review
        found: `autonomy.decide` steps back from a resolved line the note barely
        documents, but BOTH existing recommendation rules required an UNRESOLVED line,
        so the routed item carried no suggested action at all."""
        from claude_coder.autonomy import SHAKY_EXTRACTION
        from claude_coder.models import CodingResult
        from claude_coder.recommendations import build_recommendations
        ln = self._line(resolved=True, confidence=SHAKY_EXTRACTION - 0.01)
        ln.fact.axis_confidence = {"laterality": SHAKY_EXTRACTION - 0.01}
        recs = build_recommendations(
            CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[ln]))
        self.assertEqual([r["issue"] for r in recs], ["documentation_clarity"])
        self.assertIn("laterality", recs[0]["recommendation"])


class IntegralBundlingTest(unittest.TestCase):
    """An escalated ancillary that is an NCCI always-bundled (indicator 0) component
    of a billed primary is decided as INTEGRAL (bundled), not escalated. A bypassable
    (indicator 1) pair stays a genuine judgement → escalated. Synthetic codes."""

    def _anc(self, cand_code):
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod, ResolvedLine)
        f = ClinicalFact(kind=FactKind.PROCEDURE, description="ancillary",
                         evidence=[EvidenceSpan("ancillary performed")])
        return ResolvedLine(fact=f, chosen=None, method=ResolutionMethod.ABSTAINED,
                            alternatives=[CandidateCode(cand_code, "cpt", "d", 0.7)])

    def _result(self, indicator):
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, FactKind
        primary = _line("PRIMARY", FactKind.PROCEDURE, "primary procedure")
        anc = self._anc("COMPONENT")
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[primary, anc])
        return r, anc, MockSource(ncci={("PRIMARY", "COMPONENT"): indicator})

    def test_integral_component_bundled(self):
        from claude_coder.pipeline import apply_integral_bundling
        r, anc, src = self._result("0")           # always-bundled
        apply_integral_bundling(r, src)
        self.assertEqual(anc.chosen.code, "COMPONENT")
        self.assertIn("integral", anc.excluded_reason)

    def test_bypassable_component_stays_escalated(self):
        from claude_coder.pipeline import apply_integral_bundling
        r, anc, src = self._result("1")           # separately billable with a modifier
        apply_integral_bundling(r, src)
        self.assertFalse(anc.resolved)
        self.assertIsNone(anc.excluded_reason)


class LateralityUpgradeTest(unittest.TestCase):
    """An unspecified-laterality diagnosis is upgraded to the documented-side sibling
    when the authoritative family has one (validated by descriptor, not a code)."""

    def test_unspecified_upgraded_to_documented_side(self):
        """issue #6 F9-R12-E, third re-review: plain retrieval is now
        verification-required by default -- a stub verifier stands in for
        the removed no-verifier shortcut so `resolve()` still produces a
        `chosen` line for `upgrade_diagnosis_laterality` to operate on."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import (AttributeEvidence, ClinicalFact, EvidenceSpan,
                                         FactKind, RelationState)
        from claude_coder.resolution import resolve, upgrade_diagnosis_laterality
        recs = {("DX9", "icd10"): {"long_description": "some condition, unspecified site", "active": True},
                ("DX1", "icd10"): {"long_description": "some condition, right site", "active": True},
                ("DX2", "icd10"): {"long_description": "some condition, left site", "active": True}}
        src = MockSource(records=recs, retrieval={("*", "icd10"):
                         [CandidateCode("DX9", "icd10", "some condition, unspecified site", 1.0)]})
        span = EvidenceSpan("some condition, right side", anchored=True, span_id="s1")
        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="some condition",
                            attributes={"laterality": "right"},
                            evidence=[span],
                            attribute_evidence={"laterality": (
                                AttributeEvidence(span=span, assertion_state=RelationState.ASSERTED,
                                                  value="right"),)}, confidence=0.98)
        llm = _sv.judge(entails=lambda d: True, reason="entailed")
        line = resolve(_request(fact), src, llm=_from(llm, "provider-a"))
        # Specificity relatives now enter the original verification pool, so
        # the supported leaf is selected before the legacy post-pass.
        self.assertEqual(line.chosen.code, "DX1")
        line = upgrade_diagnosis_laterality(line, src)
        self.assertEqual(line.chosen.code, "DX1")           # idempotent

    def test_unspecified_retrieval_leaf_expands_before_initial_verification(self):
        """The unspecified leaf must not be the verifier's only option when
        anchored laterality and an authoritative, descriptor-compatible sibling
        exist.  The sibling is still selected only by independent descriptor
        entailment; family expansion itself never approves a code."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import (AttributeEvidence, CandidateCode,
                                         ClinicalFact, EvidenceSpan, FactKind,
                                         RelationState, ResolutionMethod)
        from claude_coder.resolution import resolve
        recs = {
            ("DX9", "icd10"): {
                "long_description": "some condition, unspecified site", "active": True},
            ("DX1", "icd10"): {
                "long_description": "some condition, right site", "active": True},
            ("DX2", "icd10"): {
                "long_description": "some condition, left site", "active": True},
        }
        src = MockSource(records=recs, retrieval={
            ("*", "icd10"): [CandidateCode(
                "DX9", "icd10", "retrieval alias", 1.0)]})
        span = EvidenceSpan("some condition, right side", anchored=True, span_id="s1")
        fact = ClinicalFact(
            kind=FactKind.DIAGNOSIS, description="some condition",
            attributes={"laterality": "right"}, evidence=[span],
            attribute_evidence={"laterality": (
                AttributeEvidence(span=span, assertion_state=RelationState.ASSERTED,
                                  value="right"),)}, confidence=0.98)
        primary = _sv.judge(entails=lambda descriptor: "right" in descriptor.lower(),
                            reason="documented specificity")

        line = resolve(_request(fact), src, llm=primary)

        self.assertEqual(line.method, ResolutionMethod.VERIFIED, line.rationale)
        self.assertEqual(line.chosen.code, "DX1", line.rationale)
        self.assertNotEqual(line.chosen.code, "DX9")
        self.assertEqual(line.chosen.descriptor, "some condition, right site")

    def test_no_upgrade_without_documented_side(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod, ResolvedLine)
        from claude_coder.resolution import upgrade_diagnosis_laterality
        recs = {("DX9", "icd10"): {"long_description": "some condition, unspecified site", "active": True},
                ("DX1", "icd10"): {"long_description": "some condition, right site", "active": True}}
        src = MockSource(records=recs)
        f = ClinicalFact(kind=FactKind.DIAGNOSIS, description="some condition",
                         evidence=[EvidenceSpan("x")])      # no laterality documented
        ln = ResolvedLine(fact=f, chosen=CandidateCode("DX9", "icd10", "some condition, unspecified site", 1.0),
                          method=ResolutionMethod.DETERMINISTIC)
        self.assertEqual(upgrade_diagnosis_laterality(ln, src).chosen.code, "DX9")   # unchanged

    def test_unbound_lexical_text_never_authorizes_a_specificity_upgrade(self):
        """issue #6 F9-R7-A: the exact reported far-distance-negation
        reproduction, through diagnosis-specificity upgrade -- no
        `attribute_evidence`, evidence text a lexical heuristic could misread
        as confirming laterality. Must never upgrade to the ruled-out side."""
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod, ResolvedLine)
        from claude_coder.resolution import upgrade_diagnosis_laterality
        recs = {("DX9", "icd10"): {"long_description": "some condition, unspecified site", "active": True},
                ("DX1", "icd10"): {"long_description": "some condition, right site", "active": True}}
        src = MockSource(records=recs)
        f = ClinicalFact(
            kind=FactKind.DIAGNOSIS, description="some condition",
            attributes={"laterality": "right"},
            evidence=[EvidenceSpan(
                "right involvement was considered but was ultimately ruled out",
                anchored=True, span_id="s1")])
        ln = ResolvedLine(fact=f, chosen=CandidateCode("DX9", "icd10", "some condition, unspecified site", 1.0),
                          method=ResolutionMethod.DETERMINISTIC)
        self.assertEqual(upgrade_diagnosis_laterality(ln, src).chosen.code, "DX9")   # unchanged


class DiagnosisModifierTest(unittest.TestCase):
    """An ICD-10 diagnosis encodes laterality IN the code and must never receive an
    RT/LT procedure modifier — even when the fact documents a side and the chosen
    code's descriptor is unspecified."""

    def test_icd10_diagnosis_gets_no_laterality_modifier(self):
        """issue #6 F9-R12-E, third re-review: a genuine direct Index hit
        (safe with no verifier, per Codex's regression #4) replaces bare
        retrieval, which is now verification-required by default and would
        abstain with no verifier configured on this pipeline run."""
        from claude_coder.data_access import MockSource
        from claude_coder.modifiers import ModifierEngine
        from claude_coder.pipeline import code_encounter
        dx = CandidateCode("DX_UNSPEC", "icd10", "some condition, unspecified site",
                           0.9, "retrieval")
        src = MockSource(records={("DX_UNSPEC", "icd10"):
                                  {"active": True, "long_description":
                                   "some condition, unspecified site"}},
                         retrieval={("*", "icd10"): [dx]},
                         index={"some condition": {"DX_UNSPEC"}})
        facts = ('{"facts":[{"fact_id":"F1","kind":"diagnosis",'
                 '"description":"some condition",'
                 '"attributes":{"laterality":"right"},'
                 '"attribute_evidence":{"laterality":[{"text":'
                 '"some condition, right side","scope":"local",'
                 '"assertion_state":"asserted","value":"right"}]},'
                 '"disposition":"performed_today",'
                 '"negated":false,"evidence":["some condition, right side"],'
                 '"confidence":0.98}]}')
        r = code_encounter("e", "some condition, right side documented", "2026-03-14",
                           source=src, extract_llm=lambda s, u: facts,
                           arbitrate_llm=lambda s, u: '{"choice":0,"confidence":0}',
                           modifier_engine=ModifierEngine(defs={"MR": {"description": "Right side of the body"}}),
                           audit_repository=__import__("claude_coder.provenance",
                               fromlist=["NullAuditRepository"]).NullAuditRepository())
        dxln = next(ln for ln in r.billable_lines if ln.chosen.code == "DX_UNSPEC")
        self.assertEqual(dxln.modifiers, [])           # never RT/LT on a diagnosis


class AutonomyVerifiedTest(unittest.TestCase):
    """Release rests on CLOSURE, not a self-reported confidence number. A GROUNDED
    line — deterministic authoritative match or a VERIFIED entailment against
    the candidate's own descriptor/requirement contract — with its gates clear
    auto-releases regardless of the LLM's (poorly calibrated) self-report; the
    only self-report still consulted is the SHAKY_EXTRACTION floor, which
    reviews a fact the note barely documents. `arbitration.arbitrate`'s tie-break
    pick (ARBITRATED) is not grounded and always reviews."""

    def _result(self, method, fact_conf):
        from claude_coder.models import (ClinicalFact, CodingResult, EvidenceSpan,
                                         FactKind, GateResult, Outcome, ResolvedLine)
        f = ClinicalFact(kind=FactKind.PROCEDURE, description="a service",
                         evidence=[EvidenceSpan("a service")], confidence=fact_conf)
        ln = ResolvedLine(fact=f, chosen=CandidateCode("PROC_X", "cpt", "d", 0.9),
                          method=method)
        return CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[ln],
                            gates=[GateResult("g", Outcome.PASS)])

    def test_verified_line_clears_floor(self):
        from claude_coder.autonomy import decide
        from claude_coder.models import ResolutionMethod, Verdict
        r = self._result(ResolutionMethod.VERIFIED, fact_conf=0.98)
        decide(r)
        self.assertEqual(r.verdict, Verdict.AUTO_READY, r.notes)

    def test_verified_line_moderate_confidence_releases_on_closure(self):
        # The #4 change: a VERIFIED line whose extraction self-report is only moderate
        # (0.80 — below the old 0.95 floor, well above the shaky floor) still releases,
        # because grounding + cleared gates are the release criterion, not the number.
        from claude_coder.autonomy import decide
        from claude_coder.models import ResolutionMethod, Verdict
        r = self._result(ResolutionMethod.VERIFIED, fact_conf=0.80)
        decide(r)
        self.assertEqual(r.verdict, Verdict.AUTO_READY, r.notes)

    def test_verified_line_shaky_documentation_reviews(self):
        # A fact the note BARELY documents (below SHAKY_EXTRACTION) gets a human even
        # when its code is grounded — the uncertainty is in the documentation.
        from claude_coder.autonomy import decide, SHAKY_EXTRACTION
        from claude_coder.models import ResolutionMethod, Verdict
        r = self._result(ResolutionMethod.VERIFIED, fact_conf=SHAKY_EXTRACTION - 0.1)
        decide(r)
        self.assertEqual(r.verdict, Verdict.REVIEW_REQUIRED)

    def test_arbitrated_line_reviews(self):
        # arbitration.arbitrate's single-model tie-break pick is not grounded
        # and never auto-releases, however high its self-reported confidence.
        from claude_coder.autonomy import decide
        from claude_coder.models import ResolutionMethod, Verdict
        r = self._result(ResolutionMethod.ARBITRATED, fact_conf=0.98)
        decide(r)
        self.assertEqual(r.verdict, Verdict.REVIEW_REQUIRED)


class DiagnosisVerifyTest(unittest.TestCase):
    """Embedding recall and cross-model agreement cannot replace typed distinction
    evidence for diagnosis candidates."""

    def test_entailment_does_not_override_an_untyped_diagnosis_tie(self):
        import json
        import re
        from claude_coder.data_access import MockSource
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod)
        from claude_coder.resolution import resolve
        d_wrong = "condition beta of the structure"      # near-neighbour, higher recall
        d_right = "condition alpha of the structure"      # the entailed one, lower recall
        src = MockSource(
            records={("DXW", "icd10"): {"long_description": d_wrong, "active": True},
                     ("DXR", "icd10"): {"long_description": d_right, "active": True}},
            retrieval={("*", "icd10"): [CandidateCode("DXW", "icd10", d_wrong, 0.95),
                                        CandidateCode("DXR", "icd10", d_right, 0.80)]})

        sel = _sv.judge(entails=lambda d: "alpha" in d.lower() and "beta" not in d.lower(), reason="documented condition")

        fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="condition alpha",
                            evidence=[EvidenceSpan("condition alpha documented")],
                            confidence=0.95)
        line = resolve(_request(fact), src, llm=sel)
        self.assertEqual(line.method, ResolutionMethod.ABSTAINED)
        self.assertIsNone(line.chosen)
        self.assertEqual({c.code for c in line.alternatives}, {"DXW", "DXR"})


if __name__ == "__main__":
    unittest.main()


class SurgicalPackageComponentTest(unittest.TestCase):
    """issue #6, real-note investigation: a documented intra-operative action with
    NO code of its own, documented as part of a billed global-period procedure's
    operative episode, is INCLUDED in that procedure's global surgical package
    (CMS MCPM Ch.12 §40.1 / NCCI Policy Manual Ch.I) -- never left as an open
    hold. Synthetic codes/structures throughout."""

    def _component(self, evidence_text, descriptors, fact_id="C", kind=None,
                   performer=None):
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod, ResolvedLine)
        kind = kind or FactKind.PROCEDURE
        attrs = {"performer_id": performer} if performer else {}
        f = ClinicalFact(kind=kind, description="component action", attributes=attrs,
                         evidence=[EvidenceSpan(evidence_text, anchored=True, span_id="c1")],
                         fact_id=fact_id)
        alts = [CandidateCode(f"ALT{i}", "cpt", d, 0.4, source="retrieval")
                for i, d in enumerate(descriptors)]
        return ResolvedLine(fact=f, chosen=None, method=ResolutionMethod.ABSTAINED,
                            alternatives=alts)

    def _primary(self, performer=None):
        from claude_coder.models import FactKind
        ln = _line("PRIMARY", FactKind.PROCEDURE, "Assembly of structure alpha",
                   attrs=({"performer_id": performer} if performer else {}))
        ln.fact.fact_id = "P"
        return ln

    def _relation(self, predicate, subject="C", obj="P", status=None, state=None):
        from claude_coder.models import RelationAssertion, RelationPredicate, RelationState
        from claude_coder.provenance import SOURCE_STRUCTURED_PRIMARY
        return RelationAssertion(
            subject_event_id=subject, predicate=RelationPredicate(predicate),
            object_event_id=obj, state=(state or RelationState.ASSERTED),
            reconciliation_status=(status or SOURCE_STRUCTURED_PRIMARY))

    def _result(self, component, relations, gp="090", primary=None):
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult
        primary = primary or self._primary()
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14",
                         lines=[primary, component], relations=relations)
        return r, MockSource(gp={"PRIMARY": gp})

    UNRELATED = ("Excision of lesion of structure epsilon", "Drainage of structure zeta")

    def test_component_with_no_code_of_its_own_is_included_in_the_package(self):
        from claude_coder.pipeline import apply_surgical_package_components
        comp = self._component("Tissue of structure alpha was tidied in the field.",
                               self.UNRELATED)
        r, src = self._result(comp, [self._relation("same_episode_as")])
        apply_surgical_package_components(r, src)
        self.assertIsNotNone(comp.excluded_reason)
        self.assertIn("included in the global surgical package of PRIMARY", comp.excluded_reason)
        self.assertIsNone(comp.chosen)
        self.assertEqual([ln.chosen.code for ln in r.billable_lines], ["PRIMARY"])

    def test_part_of_edge_also_qualifies(self):
        from claude_coder.pipeline import apply_surgical_package_components
        comp = self._component("Tissue was tidied.", self.UNRELATED)
        r, src = self._result(comp, [self._relation("part_of")])
        apply_surgical_package_components(r, src)
        self.assertIn("PRIMARY", comp.excluded_reason or "")

    def test_a_lone_ungrounded_candidate_line_is_included(self):
        from claude_coder.models import FactKind
        from claude_coder.pipeline import apply_surgical_package_components
        comp = self._component("Imaging confirmed the contour of structure alpha.",
                               ("Radiologic examination; site delta, 2 views",),
                               kind=FactKind.IMAGING)
        r, src = self._result(comp, [self._relation("same_episode_as")])
        apply_surgical_package_components(r, src)
        self.assertIn("PRIMARY", comp.excluded_reason or "")

    def test_a_candidate_the_record_names_keeps_the_line_open(self):
        """Designated-note F20 shape: the record states a candidate's own word,
        so that candidate could still be documented into a code -- never absorbed."""
        from claude_coder.pipeline import apply_surgical_package_components
        comp = self._component("A brace was applied at the end of the case.",
                               ("Brace, prefabricated, structure alpha",) + self.UNRELATED)
        r, src = self._result(comp, [self._relation("same_episode_as")])
        apply_surgical_package_components(r, src)
        self.assertIsNone(comp.excluded_reason)

    def test_no_grounded_relation_means_no_package_membership(self):
        from claude_coder.pipeline import apply_surgical_package_components
        comp = self._component("Tissue was tidied.", self.UNRELATED)
        r, src = self._result(comp, [self._relation("same_episode_as", status="unreconciled")])
        apply_surgical_package_components(r, src)
        self.assertIsNone(comp.excluded_reason)
        r, src = self._result(comp, [])
        apply_surgical_package_components(r, src)
        self.assertIsNone(comp.excluded_reason)

    def test_a_primary_without_a_global_period_has_no_package(self):
        from claude_coder.pipeline import apply_surgical_package_components
        comp = self._component("Tissue was tidied.", self.UNRELATED)
        r, src = self._result(comp, [self._relation("same_episode_as")], gp="XXX")
        apply_surgical_package_components(r, src)
        self.assertIsNone(comp.excluded_reason)

    def test_a_different_performers_procedure_is_not_this_lines_package(self):
        from claude_coder.pipeline import apply_surgical_package_components
        comp = self._component("Tissue was tidied.", self.UNRELATED, performer="DR_A")
        r, src = self._result(comp, [self._relation("same_episode_as")],
                              primary=self._primary(performer="DR_B"))
        apply_surgical_package_components(r, src)
        self.assertIsNone(comp.excluded_reason)

    def test_a_line_with_no_candidates_is_left_for_the_recall_path(self):
        from claude_coder.pipeline import apply_surgical_package_components
        comp = self._component("Tissue was tidied.", ())
        r, src = self._result(comp, [self._relation("same_episode_as")])
        apply_surgical_package_components(r, src)
        self.assertIsNone(comp.excluded_reason)


class ExcludedLineRecommendationTest(unittest.TestCase):
    """issue #6, real-note investigation: a line a reporting control has already
    excluded from the claim ("included in the global surgical package of ...")
    must not ALSO ask the provider to document it (its stale `documentation_gap`)
    or ask a coder whether it is "separately reportable vs integral"."""

    def _excluded(self, doc_gap):
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod, ResolvedLine)
        f = ClinicalFact(kind=FactKind.IMAGING, description="an intra-operative check",
                         evidence=[EvidenceSpan("an intra-operative check was done")])
        return ResolvedLine(
            fact=f, chosen=None, method=ResolutionMethod.ABSTAINED,
            alternatives=[CandidateCode("ALT_X", "cpt", "d", 0.4)],
            documentation_gap=doc_gap, rationale="r",
            excluded_reason="included in the global surgical package of PRIMARY")

    def test_an_excluded_line_emits_no_provider_or_coder_recommendation(self):
        from claude_coder.models import CodingResult
        from claude_coder.recommendations import build_recommendations
        for gap in ("the descriptor requires two views", None):
            recs = build_recommendations(CodingResult(
                encounter_id="e", date_of_service="2026-03-14", lines=[self._excluded(gap)]))
            self.assertEqual([r for r in recs if r["issue"] in
                              ("documentation_gap", "coder_review", "unresolved_service")],
                             [], recs)

    def test_an_open_line_still_gets_its_recommendation(self):
        from claude_coder.models import CodingResult
        from claude_coder.recommendations import build_recommendations
        ln = self._excluded("the descriptor requires two views")
        ln.excluded_reason = None
        recs = build_recommendations(CodingResult(
            encounter_id="e", date_of_service="2026-03-14", lines=[ln]))
        self.assertEqual([r["issue"] for r in recs], ["documentation_gap"])


class CrossLineCodeOwnershipTest(unittest.TestCase):
    """Product-owner release-policy decision (2026-09-22): a code already released
    for a DISTINCT documented event on this claim is not also this event's code,
    so a two-way tie between that code and one other entailed candidate resolves
    to the other. Synthetic codes."""

    def _lines(self, standing, alternatives, owner_fact="F_A"):
        from claude_coder.models import (ClinicalFact, EvidenceSpan, FactKind,
                                         ResolutionMethod, ResolvedLine)
        released_fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="condition beta",
                                     evidence=[EvidenceSpan("condition beta", anchored=True,
                                                            span_id="a1")], fact_id=owner_fact)
        released = ResolvedLine(fact=released_fact,
                                chosen=CandidateCode("CODE_X", "icd10", "Condition beta", 0.9),
                                method=ResolutionMethod.VERIFIED)
        held_fact = ClinicalFact(kind=FactKind.DIAGNOSIS, description="condition alpha",
                                 evidence=[EvidenceSpan("condition alpha with condition beta",
                                                        anchored=True, span_id="b1")],
                                 fact_id="F_B")
        held = ResolvedLine(fact=held_fact, chosen=None, method=ResolutionMethod.ABSTAINED,
                            alternatives=[CandidateCode(c, "icd10", d, 0.5)
                                          for c, d in alternatives],
                            tie_record={"still_entailed": list(standing)},
                            rationale="2 shortlisted candidates are still entailed")
        return released, held

    def test_the_sibling_owned_code_leaves_the_tie_and_the_survivor_releases(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult, ResolutionMethod
        from claude_coder.pipeline import apply_cross_line_code_ownership
        released, held = self._lines(["CODE_X", "CODE_Y"],
                                     [("CODE_X", "Condition beta"), ("CODE_Y", "Condition alpha")])
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[released, held])
        apply_cross_line_code_ownership(r, MockSource())
        self.assertEqual(held.chosen.code if held.chosen else None, "CODE_Y")
        self.assertEqual(held.method, ResolutionMethod.VERIFIED)
        self.assertIn("cross-line code ownership", held.rationale)
        self.assertIn("released for F_A", held.rationale)

    def test_more_than_one_survivor_stays_a_tie(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult
        from claude_coder.pipeline import apply_cross_line_code_ownership
        released, held = self._lines(
            ["CODE_X", "CODE_Y", "CODE_Z"],
            [("CODE_X", "Condition beta"), ("CODE_Y", "Condition alpha"), ("CODE_Z", "Condition gamma")])
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[released, held])
        apply_cross_line_code_ownership(r, MockSource())
        self.assertIsNone(held.chosen)

    def test_a_code_released_for_the_same_event_is_not_ownership(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult
        from claude_coder.pipeline import apply_cross_line_code_ownership
        released, held = self._lines(["CODE_X", "CODE_Y"],
                                     [("CODE_X", "Condition beta"), ("CODE_Y", "Condition alpha")],
                                     owner_fact="F_B")
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[released, held])
        apply_cross_line_code_ownership(r, MockSource())
        self.assertIsNone(held.chosen)

    def test_a_standing_candidate_missing_from_alternatives_blocks_the_mechanic(self):
        from claude_coder.data_access import MockSource
        from claude_coder.models import CodingResult
        from claude_coder.pipeline import apply_cross_line_code_ownership
        released, held = self._lines(["CODE_X", "CODE_Y"], [("CODE_X", "Condition beta")])
        r = CodingResult(encounter_id="e", date_of_service="2026-03-14", lines=[released, held])
        apply_cross_line_code_ownership(r, MockSource())
        self.assertIsNone(held.chosen)

class MissingElementPromotionTest(unittest.TestCase):
    """Product-owner release-policy decision (2026-09-22, option 2): the
    evaluator's OWN verdict can flag exactly one candidate as "the right
    kind, missing one required element" (`missing_element`) while the
    document is otherwise silent on that axis. Promoted through the SAME
    `_entailed_line`/`_chosen_own_requirements_confirmed` confirmation every
    other release path already uses -- a governed convention may authorize
    it there; nothing here duplicates that match logic. Exercises
    `resolution._propose_then_verify_core` directly (the function that owns
    this branch) rather than the full `resolve()` orchestration, whose
    other gates (semantic eligibility, candidate dominance, baseline
    grounding) are exercised by their own dedicated test files. Synthetic
    vocabulary throughout."""

    SEC = CandidateCode("CAND_SEC", "cpt", "Repair, secondary, structure alpha", 0.9,
                        source="retrieval")
    PRI = CandidateCode("CAND_PRI", "cpt", "Repair, primary, structure alpha", 0.5,
                        source="retrieval")
    UNRELATED = CandidateCode("CAND_UNRELATED", "cpt", "Excision, unrelated structure gamma",
                              0.3, source="retrieval")

    def _src(self):
        return MockSource(records={
            ("CAND_SEC", "cpt"): {"long_description": self.SEC.descriptor, "active": True},
            ("CAND_PRI", "cpt"): {"long_description": self.PRI.descriptor, "active": True},
            ("CAND_UNRELATED", "cpt"): {"long_description": self.UNRELATED.descriptor,
                                        "active": True}})

    def _fact(self, text):
        from claude_coder.models import ClinicalFact, EvidenceSpan, FactKind
        return ClinicalFact(kind=FactKind.PROCEDURE, description="structure alpha repaired",
                            evidence=[EvidenceSpan(text, anchored=True, span_id="s1")],
                            confidence=0.9, fact_id="F1")

    def _stub(self, missing_by_code):
        """A raw judge: nothing entailed, `missing_by_code` controls
        `missing_element` per eliminated option -- the real shape
        `shortlist_verdict.verdict`'s single, uniform flag cannot produce."""
        by_desc = {self.SEC.descriptor: "CAND_SEC", self.PRI.descriptor: "CAND_PRI",
                  self.UNRELATED.descriptor: "CAND_UNRELATED"}

        def _code_for(desc):
            # `_sv.options` returns each descriptor prefixed with a bracketed
            # identity hash ("[<hash>] Repair, secondary, ...") -- match by
            # substring, never exact equality.
            for text, code in by_desc.items():
                if text in desc:
                    return code
            return ""

        def stub(system, user):
            if "propose" in system.lower():
                return json.dumps({"codes": []})
            opts = _sv.options(user)
            eliminated = [{"option": n, "reason": f"stub: not {_code_for(d)}",
                          "missing_element": missing_by_code.get(_code_for(d), False)}
                         for n, d in opts]
            return json.dumps({"choice": 0, "entailed": [], "eliminated": eliminated,
                               "reason": "stub"})
        return stub

    def _pack(self, tmp):
        import pathlib
        rule = {
            "id": "structure-alpha-reattachment-is-secondary", "enabled": True,
            "axis": "sequence_qualifier", "value": "secondary",
            "applies_when": {"fact_kinds": ["procedure"],
                             "candidate_descriptor_regex": r"\bstructure alpha\b",
                             "evidence_regex": r"\breattach(?:ed|ment)?\b"},
            "authority": "synthetic authority citation", "verification_status": "test",
        }
        path = pathlib.Path(tmp) / "pack.json"
        path.write_text(json.dumps({"version": "test", "conventions": [rule]}))
        return path

    def _run(self, evidence_text, missing_by_code, pack_path=None):
        import tempfile
        from unittest.mock import patch
        from claude_coder import conventions, resolution as res
        fact = self._fact(evidence_text)
        pool = [self.SEC, self.PRI, self.UNRELATED]
        llm = self._stub(missing_by_code)
        conventions.load_pack.cache_clear()
        try:
            if pack_path is None:
                with patch.object(conventions, "PACK_PATH", __import__("pathlib").Path(
                        "/nonexistent/pack.json")):
                    return res._propose_then_verify_core(
                        fact, self._src(), pool, [], [], llm, reconciliation=_agreed_local("s1"))
            with patch.object(conventions, "PACK_PATH", pack_path):
                return res._propose_then_verify_core(
                    fact, self._src(), pool, [], [], llm, reconciliation=_agreed_local("s1"))
        finally:
            conventions.load_pack.cache_clear()

    def test_promotes_and_a_convention_authorizes_it(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pack = self._pack(tmp)
            line = self._run("structure alpha was reattached under tension",
                             {"CAND_SEC": True, "CAND_PRI": False, "CAND_UNRELATED": False},
                             pack_path=pack)
        self.assertEqual(line.chosen.code if line.chosen else None, "CAND_SEC", line.rationale)
        self.assertIn("structure-alpha-reattachment-is-secondary", line.rationale)

    def test_no_convention_still_produces_the_original_provider_query(self):
        line = self._run("structure alpha was reattached under tension",
                         {"CAND_SEC": True, "CAND_PRI": False, "CAND_UNRELATED": False})
        self.assertIsNone(line.chosen)
        self.assertIn("PROVIDER QUERY", line.rationale)

    def test_two_flagged_candidates_never_promotes(self):
        """Ambiguous which candidate to promote -- falls through unchanged."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pack = self._pack(tmp)
            line = self._run("structure alpha was reattached under tension",
                             {"CAND_SEC": True, "CAND_PRI": True, "CAND_UNRELATED": False},
                             pack_path=pack)
        self.assertIsNone(line.chosen)

    def test_a_flagged_candidate_with_no_compiled_requirement_never_auto_releases(self):
        """Safety guard: a candidate flagged missing_element with NO compiled
        MUST_SUPPORT/EXCLUSION requirement of its own must never be promoted
        -- there is nothing typed for `_chosen_own_requirements_confirmed` to
        check, and an empty requirement set there returns unconditional
        confirmation. Two candidates differing only on an UNGOVERNED word
        (no selectable axis compiles at all) reproduce that shape."""
        from claude_coder import resolution as res
        a = CandidateCode("CAND_A", "cpt", "Assembly service including component one", 0.9,
                          source="retrieval")
        b = CandidateCode("CAND_B", "cpt", "Assembly service including component two", 0.8,
                          source="retrieval")
        src = MockSource(records={
            ("CAND_A", "cpt"): {"long_description": a.descriptor, "active": True},
            ("CAND_B", "cpt"): {"long_description": b.descriptor, "active": True}})
        fact = self._fact("assembly service performed")

        def stub(system, user):
            if "propose" in system.lower():
                return json.dumps({"codes": []})
            opts = _sv.options(user)
            eliminated = [{"option": n, "reason": "stub: not entailed",
                          "missing_element": (n == opts[0][0])} for n, d in opts]
            return json.dumps({"choice": 0, "entailed": [], "eliminated": eliminated,
                               "reason": "stub"})

        line = res._propose_then_verify_core(fact, src, [a, b], [], [], stub,
                                             reconciliation=_agreed_local("s1"))
        self.assertIsNone(line.chosen, line.rationale)
        self.assertIn("PROVIDER QUERY", line.rationale)


def _agreed_local(*span_ids):
    from app.contracts.source_evidence import (ReconciliationStatus, SourceReconciliation,
                                                SpanReconciliation)
    return SourceReconciliation(spans=tuple(
        SpanReconciliation(span_id=sid, status=ReconciliationStatus.AGREED)
        for sid in span_ids))
