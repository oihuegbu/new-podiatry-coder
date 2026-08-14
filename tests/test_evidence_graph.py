"""Acceptance coverage for the Clinical Evidence and Service Graph (directive §3).

WHAT THIS PROVES

  1. Two independent readings that disagree on a normalized fact axis are settled by the
     ORIGINAL PAGE — and when the page cannot settle it, the encounter goes to
     PROVIDER_QUERY with the exact axis named. It is NEVER routed to generic coder
     review, which the directive explicitly forbids for model disagreement.
  2. Prose that differs while the axes agree produces no disagreement, no hold and no
     routing at all.
  3. A documented condition attached to an eligible service can be retrieved as a
     supporting clinical condition but can never become a billable service line: the
     graph's role boundary comes from the eligibility component, decided before
     retrieval.
  4. An explicit duplicate/cannot-link constraint is first-class on the graph, is
     enforced by the merge, and a violation is a graph-integrity BLOCK.
  5. `ClaimBundle.GraphReference` is bound to the nodes/edges the RELEASED lines rest
     on, and an unbound native claim line is a release blocker.

Everything runs through the real modules. No medical code appears anywhere in this file.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from claude_coder import eligibility  # noqa: E402
from claude_coder import eligibility as elig_module  # noqa: E402
from claude_coder import graph, graph_consensus  # noqa: E402
from claude_coder.models import (ClinicalFact, Disposition, EvidenceSpan,  # noqa: E402
                                 FactKind, Outcome, RelationAssertion,
                                 RelationPredicate, RelationState)

NOTE = (
    "Procedure performed today on the left side. "
    "The documented condition addressed today is the reason for it. "
    "A second, separately documented procedure was also performed today."
)


def _span(text, *, anchored=True, span_id=None):
    start = NOTE.find(text)
    return EvidenceSpan(text=text, start=(start if start >= 0 else None),
                        end=((start + len(text)) if start >= 0 else None),
                        anchored=anchored,
                        span_id=(span_id or f"span-{abs(hash(text)) % 100000}"))


def _fact(fact_id, kind, description, *, spans, attributes=None,
          disposition=Disposition.PERFORMED, certain=True, experiencer="patient"):
    """A service fact carries resolved actor identity by default.

    Not incidental: `_gate_actor_ownership` is tri-state and holds an event whose
    performer/billing identity is unresolved, so a service fixture without ids would be
    held for OWNERSHIP and would silently stop proving anything about the gate under
    test. Diagnoses do not run that gate.
    """
    resolved = {} if kind is FactKind.DIAGNOSIS else {
        "performer_id": "person-1", "billing_entity_id": "person-1"}
    resolved.update(attributes or {})
    return ClinicalFact(
        kind=kind, description=description, attributes=resolved,
        disposition=disposition, certain=certain, experiencer=experiencer,
        evidence=list(spans), confidence=0.9, fact_id=fact_id)


def _intents(facts, relations, encounter="enc", dos="2026-03-14"):
    return eligibility.evaluate(facts, relations, encounter, dos)


def _graph(facts, relations, intents=None, encounter="enc", dos="2026-03-14"):
    intents = intents if intents is not None else _intents(facts, relations, encounter, dos)
    episodes, _ = eligibility.build_episodes(facts, relations, encounter, dos)
    return graph.build_graph(facts, relations, intents, encounter_id=encounter,
                             date_of_service=dos, episodes=episodes,
                             extraction_schema_version="clinical-graph-v1",
                             relation_grammar_version="grammar-v1")


# ---------------------------------------------------------------------------
class TwoReadingAxisConsensus(unittest.TestCase):
    """Directive §3: compare graph axes, settle from the source, query when it cannot."""

    def _readings(self, primary_laterality, second_laterality, *,
                  primary_quote, second_quote):
        primary = [_fact("F1", FactKind.PROCEDURE, "procedure performed",
                         spans=[_span(primary_quote, span_id="p1")],
                         attributes={"laterality": primary_laterality})]
        second = [_fact("S1", FactKind.PROCEDURE, "performed procedure",
                        spans=[_span(second_quote, span_id="s1")],
                        attributes={"laterality": second_laterality})]
        return primary, second

    def test_differing_prose_on_agreeing_axes_is_not_a_disagreement(self):
        """The directive's hard rule: model prose difference routes NOTHING."""
        primary, second = self._readings("left", "left",
                                         primary_quote="Procedure performed today",
                                         second_quote="Procedure performed today")
        report, primary_by_id, second_by_node = graph_consensus.compare(primary, second)
        self.assertEqual(report.matched_events, 1,
                         "differently worded readings of one event must align")
        self.assertEqual(report.disagreements, (),
                         f"prose difference must not produce a disagreement: "
                         f"{[d.as_record() for d in report.disagreements]}")
        graph_consensus.apply_resolutions(primary_by_id, second_by_node, [])
        self.assertEqual(primary[0].axis_conflicts, [])

    def test_axis_disagreement_the_source_settles_is_accepted_not_escalated(self):
        """One reading's value is verbatim in its anchored quotation; the other's is not."""
        primary, second = self._readings(
            "right", "left",
            primary_quote="Procedure performed today",          # says nothing about side
            second_quote="performed today on the left side")    # states the axis
        report, primary_by_id, second_by_node = graph_consensus.compare(primary, second)
        axes = {d.axis for d in report.disagreements}
        self.assertIn("laterality", axes)
        resolutions = graph_consensus.resolve(list(report.disagreements), primary_by_id,
                                              second_by_node, None)
        laterality = next(r for r in resolutions if r.axis == "laterality")
        self.assertIs(laterality.verdict, graph_consensus.AxisVerdict.RESOLVED_FROM_SOURCE)
        self.assertEqual(laterality.accepted_from, "second")
        self.assertEqual(laterality.accepted_value, "left")
        graph_consensus.apply_resolutions(primary_by_id, second_by_node, resolutions)
        # The document's reading CORRECTS the primary graph, and the quotation that
        # proved it travels with the corrected fact.
        self.assertEqual(primary[0].attributes["laterality"], "left")
        self.assertIn("s1", [s.span_id for s in primary[0].evidence])
        self.assertEqual(primary[0].axis_conflicts, [])

    def test_axis_the_source_cannot_settle_becomes_a_precise_provider_query(self):
        """Both readings rest on confirmed quotations, neither is uniquely stated."""
        primary, second = self._readings(
            "right", "left",
            primary_quote="Procedure performed today",
            second_quote="Procedure performed today")
        report, primary_by_id, second_by_node = graph_consensus.compare(primary, second)
        resolutions = graph_consensus.resolve(list(report.disagreements), primary_by_id,
                                              second_by_node, None)
        laterality = next(r for r in resolutions if r.axis == "laterality")
        self.assertIs(laterality.verdict, graph_consensus.AxisVerdict.UNRESOLVED)
        self.assertIn("laterality", laterality.provider_question)
        graph_consensus.apply_resolutions(primary_by_id, second_by_node, resolutions)
        self.assertTrue(primary[0].axis_conflicts)

        # ...and eligibility HOLDS it before retrieval, on that gate specifically.
        intents = _intents(primary, [])
        intent = intents[0]
        self.assertIs(intent.state, eligibility.EligibilityState.AUTO_HOLD)
        non_pass = [d for d in intent.decisions if d.outcome is not Outcome.PASS]
        self.assertEqual([d.gate for d in non_pass], ["axis_consensus"],
                         "an unsettled axis must be the ONLY reason it is held")

    def test_unresolved_axis_routes_to_provider_query_never_to_a_coder(self):
        """End to end through the real pipeline entrypoint and the real router."""
        from claude_coder.autonomy import decide
        from claude_coder.models import CodingResult, Destination, ResolutionMethod, \
            ResolvedLine

        held = _fact("F1", FactKind.PROCEDURE, "procedure performed",
                     spans=[_span("Procedure performed today", span_id="p1")],
                     attributes={"laterality": "right"})
        held.axis_conflicts = ["The record does not settle 'laterality' for this event."]
        intents = _intents([held], [])
        intent = intents[0]
        non_pass = [d for d in intent.decisions if d.outcome is not Outcome.PASS]
        axis_only = bool(non_pass) and all(d.gate == "axis_consensus" for d in non_pass)
        self.assertTrue(axis_only)

        result = CodingResult(encounter_id="enc", date_of_service="2026-03-14")
        result.lines = [ResolvedLine(
            fact=held, chosen=None, method=ResolutionMethod.ABSTAINED,
            rationale="held for a targeted provider query",
            excluded_reason=None,
            documentation_gap="; ".join(d.detail for d in non_pass))]
        decide(result)
        self.assertIs(result.destination, Destination.PROVIDER_QUERY)
        self.assertFalse(
            any(r["destination"] == Destination.REVIEW.value for r in result.routing),
            f"model disagreement must never reach a coder queue: {result.routing}")


# ---------------------------------------------------------------------------
class EligibilityBeforeRetrievalRole(unittest.TestCase):
    """Directive §3: a condition may be retrieved as a condition, never as a service."""

    def _service_with_supporting_condition(self):
        service = _fact("F1", FactKind.PROCEDURE, "procedure performed",
                        spans=[_span("Procedure performed today", span_id="p1")])
        condition = _fact("F2", FactKind.DIAGNOSIS, "documented condition",
                          spans=[_span("The documented condition addressed today",
                                       span_id="p2")])
        reason = RelationAssertion(
            subject_event_id="F2", predicate=RelationPredicate.REASON_FOR,
            object_event_id="F1", state=RelationState.ASSERTED,
            evidence_span_ids=["p1", "p2"])
        return [service, condition], [reason]

    def test_supporting_condition_never_carries_a_service_role(self):
        facts, relations = self._service_with_supporting_condition()
        compiled = _graph(facts, relations)
        self.assertIs(compiled.role_of("F1"), graph.NodeRole.SERVICE)
        self.assertIs(compiled.role_of("F2"), graph.NodeRole.CLINICAL_CONDITION)
        self.assertEqual(compiled.integrity_problems(), ())
        # Both are eligible for retrieval — the condition IS retrievable, as a condition.
        eligible = set(compiled.eligible_node_ids())
        self.assertEqual(eligible, {"F1", "F2"})
        condition_intent = compiled.intent_for("F2")
        self.assertIs(condition_intent.component,
                      eligibility.ClaimComponent.DIAGNOSIS_SUPPORT)

    def test_a_service_role_asserted_over_a_condition_is_a_graph_integrity_block(self):
        """The boundary is ENFORCED, not merely produced correctly by construction."""
        facts, relations = self._service_with_supporting_condition()
        intents = _intents(facts, relations)
        condition_intent = next(i for i in intents if "F2" in i.clinical_event_ids)
        # Something downstream tries to promote the condition into a service line.
        condition_intent.component = eligibility.ClaimComponent.SERVICE
        compiled = _graph(facts, relations, intents)
        problems = compiled.integrity_problems()
        self.assertTrue(any("F2" in p and "role" in p for p in problems), problems)

    def test_a_condition_cannot_be_retrieved_through_a_service_retrieval_request(self):
        """The hard capability boundary still refuses a substituted event."""
        facts, relations = self._service_with_supporting_condition()
        intents = _intents(facts, relations)
        service_intent = next(i for i in intents if "F1" in i.clinical_event_ids)
        with self.assertRaises(ValueError):
            eligibility.RetrievalRequest(service_intent, facts[1])


# ---------------------------------------------------------------------------
class CannotLinkConstraints(unittest.TestCase):
    """Directive §3: explicit duplicate / cannot-link constraints."""

    def _two_distinct_plus_duplicate(self):
        a = _fact("F1", FactKind.PROCEDURE, "documented procedure action",
                  spans=[_span("Procedure performed today", span_id="p1")])
        b = _fact("F2", FactKind.PROCEDURE, "documented procedure action",
                  spans=[_span("A second, separately documented procedure",
                               span_id="p2")])
        separate = RelationAssertion(
            subject_event_id="F1", predicate=RelationPredicate.SEPARATE_FROM,
            object_event_id="F2", state=RelationState.ASSERTED,
            evidence_span_ids=["p1", "p2"])
        return [a, b], [separate]

    def test_cannot_link_is_first_class_on_the_graph(self):
        facts, relations = self._two_distinct_plus_duplicate()
        compiled = _graph(facts, relations)
        pairs = {c.pair() for c in compiled.cannot_links}
        self.assertIn(frozenset({"F1", "F2"}), pairs)
        self.assertEqual(compiled.integrity_problems(), ())

    def test_documented_distinctness_is_not_collapsed_into_one_line(self):
        facts, relations = self._two_distinct_plus_duplicate()
        intents = _intents(facts, relations)
        merged = [i for i in intents
                  if {"F1", "F2"}.issubset(set(i.clinical_event_ids))]
        self.assertEqual(merged, [], "a cannot-link must never be merged away")

    def test_a_merge_across_a_cannot_link_is_a_graph_integrity_block(self):
        facts, relations = self._two_distinct_plus_duplicate()
        intents = _intents(facts, relations)
        first = next(i for i in intents if "F1" in i.clinical_event_ids)
        second = next(i for i in intents if "F2" in i.clinical_event_ids)
        # Simulate a downstream merge that ignores the constraint.
        first.clinical_event_ids = ["F1", "F2"]
        intents = [i for i in intents if i is not second]
        compiled = _graph(facts, relations, intents)
        problems = compiled.integrity_problems()
        self.assertTrue(any("cannot-link" in p for p in problems), problems)

    def test_known_known_attribute_conflict_is_recorded_as_a_constraint(self):
        a = _fact("F1", FactKind.PROCEDURE, "documented procedure action",
                  spans=[_span("Procedure performed today", span_id="p1")],
                  attributes={"approach": "first approach"})
        b = _fact("F2", FactKind.PROCEDURE, "documented procedure action",
                  spans=[_span("A second, separately documented procedure",
                               span_id="p2")],
                  attributes={"approach": "other approach"})
        compiled = _graph([a, b], [])
        bases = [c.basis for c in compiled.cannot_links]
        self.assertTrue(any("approach" in basis for basis in bases), bases)


# ---------------------------------------------------------------------------
class GraphBinding(unittest.TestCase):
    """Directive §3/§5: the bundle names the graph its released lines rest on."""

    def _encounter(self):
        service = _fact("F1", FactKind.PROCEDURE, "procedure performed",
                        spans=[_span("Procedure performed today", span_id="p1")])
        condition = _fact("F2", FactKind.DIAGNOSIS, "documented condition",
                          spans=[_span("The documented condition addressed today",
                                       span_id="p2")])
        unrelated = _fact("F3", FactKind.PROCEDURE, "unrelated planned action",
                          spans=[_span("A second, separately documented procedure",
                                       span_id="p3")],
                          disposition=Disposition.PLANNED)
        reason = RelationAssertion(
            subject_event_id="F2", predicate=RelationPredicate.REASON_FOR,
            object_event_id="F1", state=RelationState.ASSERTED,
            evidence_span_ids=["p1", "p2"])
        return [service, condition, unrelated], [reason]

    def test_binding_is_narrowed_to_the_released_lines_and_one_documented_hop(self):
        facts, relations = self._encounter()
        compiled = _graph(facts, relations)
        binding = compiled.binding_for(["F1"])
        self.assertIn("F1", binding.clinical_event_ids)
        self.assertIn("F2", binding.clinical_event_ids,
                      "the condition the record gave as the reason must be bound")
        self.assertNotIn("F3", binding.clinical_event_ids,
                         "an unrelated event must not be bound to this line")
        self.assertEqual(len(binding.relation_ids), 1)
        self.assertIn("p1", binding.evidence_span_ids)
        self.assertIn("p2", binding.evidence_span_ids)

    def test_reference_payload_carries_the_graph_versions(self):
        facts, relations = self._encounter()
        payload = _graph(facts, relations).reference_payload(["F1"])
        self.assertEqual(payload["extraction_schema_version"], "clinical-graph-v1")
        self.assertEqual(payload["relation_grammar_version"], "grammar-v1")

    def test_an_unbound_native_claim_line_is_a_release_blocker(self):
        from app.contracts.claim_bundle import (
            BundleOrigin, ClaimBundle, DiagnosisLine, EncounterIdentity, GraphReference,
            ReleaseDestination, ReleaseStatus, ServiceLine, finalize)

        def _bundle(graph_reference):
            return finalize(ClaimBundle(
                produced_by=BundleOrigin.CLAUDE_CODER,
                encounter=EncounterIdentity(encounter_id="n", document_id="n",
                                            date_of_service="2026-03-14"),
                graph=graph_reference,
                diagnoses=(DiagnosisLine(sequence=1, system="icd10", code="D",
                                         clinical_event_id="F2", primary=True),),
                service_lines=(ServiceLine(sequence=1, system="cpt", code="S", units=1,
                                           clinical_event_id="F1",
                                           diagnosis_pointers=(1,)),),
                release=ReleaseStatus(destination=ReleaseDestination.AUTO_READY)))

        unbound = _bundle(GraphReference())
        self.assertTrue(
            any("not bound to the clinical graph" in b
                for b in unbound.release_blockers()), unbound.release_blockers())

        bound = _bundle(GraphReference(
            extraction_schema_version="clinical-graph-v1",
            relation_grammar_version="grammar-v1",
            clinical_event_ids=("F1", "F2")))
        self.assertFalse(
            any("clinical graph" in b for b in bound.release_blockers()),
            bound.release_blockers())

    def test_a_bundle_built_from_a_result_binds_only_the_released_events(self):
        from app.contracts.claim_bundle import (
            AuthorityBinding, EncounterContext, SourceDocument,
            bundle_from_coding_result)
        from claude_coder.models import (CandidateCode, CodingResult, ResolutionMethod,
                                         ResolvedLine)

        facts, relations = self._encounter()
        compiled = _graph(facts, relations)
        service, condition, unrelated = facts
        result = CodingResult(encounter_id="enc", date_of_service="2026-03-14")
        result.graph = compiled
        result.claim_line_intents = list(compiled.intents)
        result.relations = list(relations)
        result.lines = [
            ResolvedLine(fact=condition,
                         chosen=CandidateCode(code="D", system="icd10",
                                              descriptor="condition"),
                         method=ResolutionMethod.DETERMINISTIC),
            ResolvedLine(fact=service,
                         chosen=CandidateCode(code="S", system="cpt",
                                              descriptor="service"),
                         method=ResolutionMethod.DETERMINISTIC),
            ResolvedLine(fact=unrelated, chosen=None,
                         method=ResolutionMethod.ABSTAINED),
        ]
        bundle = bundle_from_coding_result(
            result, source_document=SourceDocument(filename="n"),
            context=EncounterContext(), authority=AuthorityBinding())
        self.assertEqual(bundle.graph.extraction_schema_version, "clinical-graph-v1")
        self.assertIn("F1", bundle.graph.clinical_event_ids)
        self.assertIn("F2", bundle.graph.clinical_event_ids)
        self.assertNotIn("F3", bundle.graph.clinical_event_ids,
                         "an event no released line rests on must not be bound")
        self.assertFalse(any("not bound to the clinical graph" in b
                             for b in bundle.release_blockers()),
                         bundle.release_blockers())


# ---------------------------------------------------------------------------
class GraphIsTheSingleRepresentation(unittest.TestCase):
    """The graph carries every axis the directive names, from the real primitives."""

    def test_nodes_carry_the_typed_axes_and_participants(self):
        fact = _fact("F1", FactKind.PROCEDURE, "procedure performed",
                     spans=[_span("Procedure performed today", span_id="p1")],
                     attributes={"anatomy": "site", "laterality": "left", "count": 2,
                                 "performer_id": "person-1",
                                 "billing_entity_id": "person-1"})
        compiled = _graph([fact], [])
        node = compiled.nodes["F1"]
        self.assertEqual(node.kind, FactKind.PROCEDURE.value)
        self.assertEqual(node.status, Disposition.PERFORMED.value)
        self.assertEqual(node.experiencer, "patient")
        self.assertEqual(node.performer_id, "person-1")
        self.assertEqual(node.billing_entity_id, "person-1")
        self.assertEqual(node.attributes["count"], 2)
        self.assertTrue(node.anchored)
        self.assertIsNotNone(node.service_episode_id)

    def test_an_edge_naming_an_unknown_event_is_an_integrity_problem(self):
        fact = _fact("F1", FactKind.PROCEDURE, "procedure performed",
                     spans=[_span("Procedure performed today", span_id="p1")])
        dangling = RelationAssertion(
            subject_event_id="F1", predicate=RelationPredicate.PART_OF,
            object_event_id="MISSING", state=RelationState.ASSERTED)
        compiled = graph.build_graph([fact], [dangling], _intents([fact], []),
                                     encounter_id="enc")
        self.assertTrue(any("MISSING" in p for p in compiled.integrity_problems()))

    def test_graph_record_is_stable_and_self_describing(self):
        fact = _fact("F1", FactKind.PROCEDURE, "procedure performed",
                     spans=[_span("Procedure performed today", span_id="p1")])
        compiled = _graph([fact], [])
        record = compiled.as_record()
        self.assertEqual(record["schema_version"], graph.GRAPH_SCHEMA_VERSION)
        self.assertEqual(compiled.graph_sha256(), _graph([fact], []).graph_sha256())


# ---------------------------------------------------------------------------
# END TO END, through the real entrypoint
# ---------------------------------------------------------------------------
# The classes above prove the mechanics. These prove the WIRING: that a real
# `code_encounter` call with two independent readings settles a disagreeing axis
# against the document, corrects the graph, and — when the document cannot settle it —
# holds the event before retrieval and routes it to PROVIDER_QUERY. A unit-level proof
# would not catch a mechanism that is correct but never called.

NOTE_E2E = ("Procedure alpha performed today on the left side. "
            "Condition alpha addressed today.")

_BILLING = {"billing_entity_id": "actor-1",
            "participants": [{"id": "actor-1", "type": "person",
                              "roles": ["performer"]}]}


def _reading(description, laterality, quote):
    import json
    return json.dumps({"facts": [{
        "fact_id": "F1", "kind": "procedure", "description": description,
        "attributes": {"laterality": laterality, "performer_id": "actor-1",
                       "billing_entity_id": "actor-1"},
        "disposition": "performed_today", "negated": False,
        "evidence": [quote], "confidence": 0.99}]})


def _mock_source():
    from claude_coder.data_access import MockSource
    from claude_coder.models import CandidateCode
    return MockSource(
        records={("PROC_X", "cpt"): {"active": True}},
        retrieval={("*", "cpt"): [CandidateCode("PROC_X", "cpt",
                                                "Procedure alpha, each", 0.9)]})


def _stub_llm(system, user):
    lowered = system.lower()
    if "propose" in lowered:
        return '{"codes":[]}'
    if "independently" in lowered:
        return '{"entailed":true,"missing_element":false,"reason":"x"}'
    return '{"choice":1,"reason":"x"}'


def _null_audit():
    from claude_coder.provenance import NullAuditRepository
    return NullAuditRepository()


def _run(primary_reading, second_reading, **kwargs):
    from claude_coder.pipeline import code_encounter
    return code_encounter(
        "enc", kwargs.pop("note_text", NOTE_E2E), "2026-03-14",
        source=_mock_source(),
        extract_llm=lambda s, u: primary_reading,
        extract_llm_b=lambda s, u: second_reading,
        verify_llm=_stub_llm, corroborate_llm=_stub_llm,
        billing_context=_BILLING, audit_repository=_null_audit(), **kwargs)


class EndToEndTwoReadingConsensus(unittest.TestCase):

    def test_the_document_corrects_the_graph_and_the_line_still_bills(self):
        """The second reading quotes text that STATES the axis; the primary's does not."""
        result = _run(
            _reading("excision procedure alpha performed", "right",
                     "Procedure alpha performed today"),
            _reading("procedure alpha performed excision", "left",
                     "performed today on the left side"))
        self.assertIsNotNone(result.consensus, "a second reading must be recorded")
        resolutions = result.consensus["resolutions"]
        laterality = next(r for r in resolutions if r["axis"] == "laterality")
        self.assertEqual(laterality["verdict"], "resolved_from_source")
        self.assertEqual(laterality["accepted_value"], "left")
        # The graph carries the corrected axis, and the line still reached retrieval.
        node = result.graph.nodes["F1"]
        self.assertEqual(node.attributes["laterality"], "left")
        self.assertEqual(node.axis_conflicts, ())
        self.assertTrue(any(ln.chosen and ln.chosen.code == "PROC_X"
                            for ln in result.lines),
                        "a settled axis must not stop the line from being coded")

    def test_an_unsettleable_axis_holds_before_retrieval_and_asks_the_provider(self):
        """Neither reading's quotation states the axis: the record simply lacks it."""
        from claude_coder.models import Destination

        result = _run(
            _reading("excision procedure alpha performed", "right",
                     "Procedure alpha performed today"),
            _reading("procedure alpha performed excision", "left",
                     "Procedure alpha performed today"))
        laterality = next(r for r in result.consensus["resolutions"]
                          if r["axis"] == "laterality")
        self.assertEqual(laterality["verdict"], "unresolved")
        self.assertIn("laterality", laterality["provider_question"])

        # Held BEFORE retrieval — no code was ever fetched for it.
        self.assertFalse(any(ln.chosen for ln in result.lines),
                         "an unsettled code-changing axis must stop retrieval")
        self.assertTrue(result.graph.nodes["F1"].axis_conflicts)

        # ...and routed to the PROVIDER, never to a coder.
        self.assertIs(result.destination, Destination.PROVIDER_QUERY)
        self.assertFalse(
            any(r["destination"] == Destination.REVIEW.value for r in result.routing),
            f"model disagreement must never reach a coder queue: {result.routing}")
        query = next(r for r in result.routing
                     if r["destination"] == Destination.PROVIDER_QUERY.value)
        self.assertIn("laterality", query["reason"])

    def test_the_original_page_is_what_settles_it_not_the_two_models(self):
        """The page verifier from directive §1 decides, and says so in the record.

        The ORIGINAL document contradicts the primary reading's quotation and confirms
        the second reading's. Neither reading's value is verbatim in its own quotation
        here, so the ONLY thing that can settle this is the page — which is the point.
        """
        import tempfile
        from pathlib import Path as _Path

        from app.contracts.source_evidence import PAGE_SEPARATOR
        from app.ingestion.source_evidence import compile_source_evidence
        from tests.source_pdf import build_pdf, vision_extraction

        vision_text = ("Procedure alpha performed today, side one. "
                       "Condition alpha addressed today.")
        # The ORIGINAL page says something different where the primary reading quoted.
        pdf_lines = ["Procedure beta performed today, side one.",
                     "Condition alpha addressed today."]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pdf_path = _Path(tmp.name) / "note.pdf"
        pdf_path.write_bytes(build_pdf([pdf_lines]))
        document = compile_source_evidence(
            pdf_path,
            vision_extraction([vision_text],
                              metadata={"date_of_service": "2026-03-14"},
                              page_separator=PAGE_SEPARATOR))

        result = _run(
            _reading("excision procedure alpha performed", "right",
                     "Procedure alpha performed today"),          # contradicted by the page
            _reading("procedure alpha performed excision", "left",
                     "Condition alpha addressed today"),          # confirmed by the page
            note_text=document.primary_text(),
            source_evidence=document)
        laterality = next(r for r in result.consensus["resolutions"]
                          if r["axis"] == "laterality")
        self.assertEqual(laterality["verdict"], "resolved_from_source")
        self.assertEqual(laterality["proof"], "original_page_reconciliation",
                         "the ORIGINAL PAGE must be what settled it")
        self.assertEqual(laterality["accepted_from"], "second")
        self.assertEqual(result.graph.nodes["F1"].attributes["laterality"], "left")
        # ...and carrying the winning reading's confirmed quotation onto the fact does
        # NOT launder the primary reading's contradicted one: the source-evidence gate
        # takes the WORST outcome across every quotation a line rests on, so the
        # misreading still stops the claim.
        from claude_coder.models import Outcome as _Outcome
        gate = next(g for g in result.gates
                    if g.name == "source_evidence_reconciliation")
        self.assertIn(gate.outcome, (_Outcome.BLOCKED, _Outcome.UNKNOWN), gate.detail)

    def test_a_second_reading_that_fails_holds_the_encounter_with_zero_retrieval(self):
        """A control that could not run must never look like two readings agreeing."""
        from claude_coder.models import Destination

        def _broken(system, user):
            raise RuntimeError("second reading unavailable")

        from claude_coder.pipeline import code_encounter
        held = code_encounter(
            "enc", NOTE_E2E, "2026-03-14", source=_mock_source(),
            extract_llm=lambda s, u: _reading("excision procedure alpha performed",
                                              "right", "Procedure alpha performed today"),
            extract_llm_b=_broken, verify_llm=_stub_llm, corroborate_llm=_stub_llm,
            billing_context=_BILLING, audit_repository=_null_audit())
        self.assertFalse(any(ln.chosen for ln in held.lines),
                         "a failed second reading must yield zero retrieval")
        self.assertIs(held.destination, Destination.SYSTEM_HOLD)
        self.assertTrue(any(g.name == "pre_retrieval_integrity" for g in held.gates))

    def test_a_held_condition_is_not_made_multi_cause_by_a_recorded_note(self):
        """Post-fix regression (second-pass review).

        A condition always carries a `diagnosis_linkage` decision that is UNKNOWN when
        the note documents no explicit reason edge — and that decision is deliberately
        NON-blocking. Counting it made a condition held ONLY by an unsettled fact axis
        look held for two reasons, which routed it to a coder: precisely the outcome the
        directive forbids for a model disagreement.
        """
        import json

        from claude_coder import eligibility as elig
        from claude_coder.models import Destination

        def _condition(description, laterality):
            return json.dumps({"facts": [{
                "fact_id": "F1", "kind": "diagnosis", "description": description,
                "attributes": {"laterality": laterality},
                "disposition": "performed_today", "negated": False,
                "certainty": "confirmed", "experiencer": "patient",
                "evidence": ["Condition alpha addressed today"], "confidence": 0.99}]})

        result = _run(_condition("condition alpha documented", "right"),
                      _condition("documented condition alpha", "left"))
        intent = result.claim_line_intents[0]
        self.assertIs(intent.state, elig.EligibilityState.AUTO_HOLD)
        # Recorded-but-non-blocking notes are present...
        self.assertIn("diagnosis_linkage", [d.gate for d in intent.decisions])
        # ...and are NOT counted as reasons the item is held.
        self.assertEqual([d.gate for d in elig.blocking_decisions(intent)],
                         ["axis_consensus"])
        self.assertIs(result.destination, Destination.PROVIDER_QUERY)
        self.assertFalse(
            any(r["destination"] == Destination.REVIEW.value for r in result.routing),
            f"an unsettled axis on a condition must not reach a coder: {result.routing}")

    def test_a_condition_promoted_to_a_service_blocks_the_live_encounter(self):
        """Eligibility-before-retrieval, proven through the REAL entrypoint.

        The directive: "diagnoses may be retrieved as supported clinical conditions but
        cannot manufacture a service." Proving that by construction is not enough — the
        boundary has to STOP a live encounter. Here the eligibility engine is made to
        hand back a SERVICE component for a documented condition (the only way a
        condition could reach a service line) and the encounter must stop hard.
        """
        import json

        from claude_coder.models import Verdict

        def _condition_reading():
            return json.dumps({"facts": [{
                "fact_id": "F1", "kind": "diagnosis",
                "description": "condition alpha documented",
                "attributes": {}, "disposition": "performed_today", "negated": False,
                "certainty": "confirmed", "experiencer": "patient",
                "evidence": ["Condition alpha addressed today"], "confidence": 0.99}]})

        real_evaluate = elig_module.evaluate

        def promoted(facts, relations, encounter_id, dos):
            intents = real_evaluate(facts, relations, encounter_id, dos)
            for intent in intents:
                intent.component = elig_module.ClaimComponent.SERVICE
            return intents

        from claude_coder import pipeline as pipeline_module
        original = elig_module.evaluate
        elig_module.evaluate = promoted
        try:
            result = pipeline_module.code_encounter(
                "enc", NOTE_E2E, "2026-03-14", source=_mock_source(),
                extract_llm=lambda s, u: _condition_reading(),
                verify_llm=_stub_llm, corroborate_llm=_stub_llm,
                billing_context=_BILLING, audit_repository=_null_audit())
        finally:
            elig_module.evaluate = original

        self.assertIs(result.verdict, Verdict.BLOCKED,
                      "a condition carrying a service role must stop the encounter")
        blocked = [g for g in result.gates if g.name == "clinical_graph_integrity"]
        self.assertTrue(blocked, [g.name for g in result.gates])
        self.assertIn("service", blocked[0].detail)
        self.assertFalse(any(ln.chosen for ln in result.lines),
                         "nothing may be retrieved once the graph is incoherent")

    def test_a_misread_page_stays_an_integrity_block_not_a_provider_query(self):
        """Precedence regression (second-pass review), found by the phase-1 e2e suite.

        When NEITHER reading rests on a quotation the original page confirms, the event
        does not have a documentation gap — its evidence is contradicted by the
        document. Holding it for a provider question would take the misread quotation
        out of the reach of the source-evidence control that must BLOCK it, silently
        downgrading an integrity stop to a question. The axis comparison must therefore
        stand down and let that control see the event.
        """
        import tempfile
        from pathlib import Path as _Path

        from app.contracts.source_evidence import PAGE_SEPARATOR
        from app.ingestion.source_evidence import compile_source_evidence
        from claude_coder.models import Outcome
        from tests.source_pdf import build_pdf, vision_extraction

        vision_text = ("Procedure alpha performed today, side one. "
                       "Condition alpha addressed today.")
        # The ORIGINAL page contradicts BOTH readings' quotations.
        pdf_lines = ["Procedure beta performed today, side two.",
                     "Condition beta addressed today."]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pdf_path = _Path(tmp.name) / "note.pdf"
        pdf_path.write_bytes(build_pdf([pdf_lines]))
        document = compile_source_evidence(
            pdf_path,
            vision_extraction([vision_text],
                              metadata={"date_of_service": "2026-03-14"},
                              page_separator=PAGE_SEPARATOR))

        result = _run(
            _reading("excision procedure alpha performed", "right",
                     "Procedure alpha performed today"),
            _reading("procedure alpha performed excision", "left",
                     "Condition alpha addressed today"),
            note_text=document.primary_text(),
            source_evidence=document)

        laterality = next(r for r in result.consensus["resolutions"]
                          if r["axis"] == "laterality")
        self.assertEqual(laterality["verdict"], "unresolved")
        self.assertEqual(laterality["provider_question"], "",
                         "a contradicted quotation must not become a provider question")
        self.assertEqual(result.graph.nodes["F1"].axis_conflicts, (),
                         "the event must not be held before retrieval by the axis gate")
        # The source-evidence control keeps ownership of the failure.
        reconciliation = [g for g in result.gates
                          if g.name == "source_evidence_reconciliation"]
        self.assertTrue(reconciliation, [g.name for g in result.gates])
        self.assertIs(reconciliation[0].outcome, Outcome.BLOCKED)
        self.assertFalse(reconciliation[0].retryable,
                         "a misread page is an integrity stop, not a retryable failure")

    def test_the_second_reading_can_be_disabled_without_a_code_change(self):
        """Rollback path. The control is a configuration switch, and turning it off
        leaves ONE reading, an honest record that says so, and a graph that still binds
        the claim — never a run that quietly reports two readings agreeing."""
        from unittest import mock

        from app.core import config
        from claude_coder import extraction as extraction_module
        from claude_coder.pipeline import code_encounter

        primary = _reading("excision procedure alpha performed", "left",
                           "Procedure alpha performed today")

        def _must_not_run(system, user):
            raise AssertionError("the second reading ran while it was disabled")

        with mock.patch.object(config, "GRAPH_CONSENSUS", False), \
                mock.patch.object(extraction_module, "_default_llm",
                                  lambda s, u: primary), \
                mock.patch.object(extraction_module, "default_second_extract_llm",
                                  _must_not_run):
            result = code_encounter(
                "enc", NOTE_E2E, "2026-03-14", source=_mock_source(),
                verify_llm=_stub_llm, corroborate_llm=_stub_llm,
                billing_context=_BILLING, audit_repository=_null_audit())

        self.assertIsNone(result.consensus,
                          "a disabled control must record that no second reading ran")
        self.assertIsNotNone(result.graph)
        self.assertIn("F1", result.graph.nodes)
        self.assertEqual(result.graph.nodes["F1"].axis_conflicts, ())
        self.assertTrue(any(ln.chosen and ln.chosen.code == "PROC_X"
                            for ln in result.lines),
                        "one reading still codes the encounter")



if __name__ == "__main__":
    unittest.main()
