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
  6. The second reading provides RECALL redundancy, not only axis agreement: a performed
     service only IT found is proven against the original page and then run through the
     same ownership, occurrence, dedup, eligibility, retrieval and certificate path as
     every primary event -- while a reworded duplicate never becomes a second claim line
     and an event the page contradicts never enters the graph (issue #6 F7-R3).

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
from tests import shortlist_verdict as _sv

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


#: Both judging roles: the sole shortlisted candidate is entailed and anything else is
#: eliminated with a named reason, so the selection is provably unique.
_stub_llm = _sv.judge(pick=1, reason="x")


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
            vision_extraction([vision_text], pdf_path=pdf_path,
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

    def test_a_second_reading_that_fails_says_why_it_failed(self):
        """Fail-closed is not the whole requirement: the hold must also be DIAGNOSABLE.

        The test above proves the encounter stops with zero retrieval, which is the
        claim-safety half. This is the operability half, and its absence is what made a
        real incident expensive: the ClaimBundle carries no gate detail, so a pipeline
        that stopped here surfaced to an operator only as `SYSTEM_RETRY | 0 diagnosis
        line(s) | 0 service line(s)`. A missing second-reading credential, an
        unreachable vendor and a malformed extractor response were indistinguishable,
        and every note in the batch failed the same way with the cause recorded
        nowhere. The cause, the stage and the traceback must reach the log.
        """
        def _broken(system, user):
            raise RuntimeError("second reading unavailable")

        from claude_coder.pipeline import code_encounter
        with self.assertLogs("claude_coder.pipeline", level="ERROR") as captured:
            code_encounter(
                "enc", NOTE_E2E, "2026-03-14", source=_mock_source(),
                extract_llm=lambda s, u: _reading(
                    "excision procedure alpha performed", "right",
                    "Procedure alpha performed today"),
                extract_llm_b=_broken, verify_llm=_stub_llm,
                corroborate_llm=_stub_llm, billing_context=_BILLING,
                audit_repository=_null_audit())
        logged = "\n".join(captured.output)
        # WHICH encounter, WHICH boundary, WHICH failure, and where it came from.
        self.assertIn("enc", logged)
        self.assertIn("pre_retrieval_integrity", logged, logged)
        self.assertIn("RuntimeError", logged, logged)
        self.assertIn("second reading unavailable", logged, logged)
        self.assertIn("Traceback", logged, logged)

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
            vision_extraction([vision_text], pdf_path=pdf_path,
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

    def test_a_disagreement_pays_for_a_targeted_read_of_only_the_pages_it_needs(self):
        """The escalation branch itself, end to end.

        The page carrying both quotations has NO independent reading at all (an
        image-only page), so nothing can settle the disagreement until one is obtained.
        The consensus step must therefore aim the PAID reader at exactly that page —
        not at the whole document — and then decide from what it reads.
        """
        import tempfile
        from pathlib import Path as _Path

        from app.contracts.source_evidence import (
            ChannelKind, PAGE_SEPARATOR, ReadChannel, build_page_read)
        from app.ingestion.source_evidence import (
            SECONDARY_VISION_CHANNEL_ID, compile_source_evidence)
        from tests.source_pdf import build_pdf, vision_extraction

        vision_text = ("Procedure alpha performed today, side one. "
                       "Condition alpha addressed today.")
        # An image-only page: the PDF carries no embedded text, so no channel in the
        # compiled document can confirm or refute anything on it.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pdf_path = _Path(tmp.name) / "note.pdf"
        pdf_path.write_bytes(build_pdf([[]]))
        document = compile_source_evidence(
            pdf_path,
            vision_extraction([vision_text], pdf_path=pdf_path,
                              metadata={"date_of_service": "2026-03-14"},
                              page_separator=PAGE_SEPARATOR))

        read_calls = []

        class _PaidReader:
            def channel(self):
                return ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                                   kind=ChannelKind.VISION, provider="openai")

            def read_pages(self, page_numbers):
                read_calls.append(tuple(page_numbers))
                # What the ORIGINAL page actually says: it contradicts the primary
                # reading's quotation and confirms the second reading's.
                return {number: build_page_read(
                    SECONDARY_VISION_CHANNEL_ID, number,
                    "Procedure beta performed today, side one. "
                    "Condition alpha addressed today.")
                    for number in page_numbers}

        result = _run(
            _reading("excision procedure alpha performed", "right",
                     "Procedure alpha performed today"),
            _reading("procedure alpha performed excision", "left",
                     "Condition alpha addressed today"),
            note_text=document.primary_text(),
            source_evidence=document,
            source_reader=_PaidReader())

        self.assertEqual(read_calls, [(1,)],
                         "the paid read must be aimed at exactly the page the "
                         "disagreeing quotations sit on")
        # This page has NO channel at all covering it, so it is read PROACTIVELY as
        # part of recall (issue #6 F7-R3, defect A) rather than by the later,
        # disagreement-driven escalation -- one read either way, attributed to
        # whichever mechanism actually obtained it.
        self.assertEqual(result.consensus["recall_page_read_pages"], [1])
        self.assertEqual(result.consensus["escalated_pages"], [])
        laterality = next(r for r in result.consensus["resolutions"]
                          if r["axis"] == "laterality")
        self.assertEqual(laterality["verdict"], "resolved_from_source")
        self.assertEqual(laterality["proof"], "original_page_reconciliation")
        self.assertEqual(laterality["accepted_from"], "second")
        self.assertEqual(result.graph.nodes["F1"].attributes["laterality"], "left")

    def test_a_failed_targeted_read_is_a_dependency_failure_not_a_coder_queue(self):
        """The failure path of the escalation: a read that could not happen proves
        nothing, and must not be allowed to look like a resolution."""
        import tempfile
        from pathlib import Path as _Path

        from app.contracts.source_evidence import PAGE_SEPARATOR
        from app.ingestion.source_evidence import compile_source_evidence
        from claude_coder.models import Destination, Outcome
        from tests.source_pdf import build_pdf, vision_extraction

        vision_text = ("Procedure alpha performed today, side one. "
                       "Condition alpha addressed today.")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pdf_path = _Path(tmp.name) / "note.pdf"
        pdf_path.write_bytes(build_pdf([[]]))
        document = compile_source_evidence(
            pdf_path,
            vision_extraction([vision_text], pdf_path=pdf_path,
                              metadata={"date_of_service": "2026-03-14"},
                              page_separator=PAGE_SEPARATOR))

        class _BrokenReader:
            def channel(self):
                raise RuntimeError("independent reader unavailable")

            def read_pages(self, page_numbers):
                raise RuntimeError("independent reader unavailable")

        result = _run(
            _reading("excision procedure alpha performed", "right",
                     "Procedure alpha performed today"),
            _reading("procedure alpha performed excision", "left",
                     "Condition alpha addressed today"),
            note_text=document.primary_text(),
            source_evidence=document,
            source_reader=_BrokenReader())

        self.assertEqual(result.consensus["escalated_pages"], [])
        self.assertIn("unavailable", result.consensus["escalation_detail"])
        laterality = next(r for r in result.consensus["resolutions"]
                          if r["axis"] == "laterality")
        self.assertEqual(laterality["verdict"], "unresolved")
        self.assertEqual(laterality["provider_question"], "",
                         "with nothing confirmed, this is not a documentation gap")

        # The axis comparison deliberately does NOT convert an unreadable page into a
        # provider query (see the precedence test above), so the source-evidence control
        # keeps ownership and reports it as what it is: a dependency that could not be
        # reached — retry it, do not pay a coder to look at it.
        gate = next(g for g in result.gates
                    if g.name == "source_evidence_reconciliation")
        self.assertIs(gate.outcome, Outcome.UNKNOWN, gate.detail)
        self.assertTrue(gate.retryable,
                        "an unreadable page is a dependency failure, not judgement")
        self.assertFalse(
            any(r["destination"] == Destination.REVIEW.value for r in result.routing),
            f"a read that could not happen must not send the claim to a coder: "
            f"{result.routing}")

    def test_a_paid_reader_that_is_not_independent_blocks_instead_of_being_credited(self):
        """Issue #6 F7-R5: the enforcement point, end to end.

        A reader whose channel declares the SAME vendor that produced the primary
        reading is not a second opinion -- it is the transcription answering its own
        question. It used to be admitted to the document, contribute a reading nothing
        could credit, and leave a record naming a channel that proved nothing. It now
        stops the encounter at the pre-retrieval boundary, and never pays for a page.
        """
        import tempfile
        from pathlib import Path as _Path

        from app.contracts.source_evidence import (
            ChannelKind, PAGE_SEPARATOR, ReadChannel, build_page_read)
        from app.ingestion.source_evidence import (
            SECONDARY_VISION_CHANNEL_ID, compile_source_evidence)
        from claude_coder.models import Outcome, Verdict
        from tests.source_pdf import build_pdf, vision_extraction

        vision_text = ("Procedure alpha performed today, side one. "
                       "Condition alpha addressed today.")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pdf_path = _Path(tmp.name) / "note.pdf"
        pdf_path.write_bytes(build_pdf([[]]))
        document = compile_source_evidence(
            pdf_path,
            vision_extraction([vision_text], pdf_path=pdf_path,
                              metadata={"date_of_service": "2026-03-14"},
                              page_separator=PAGE_SEPARATOR))

        read_calls = []

        class _SameVendorReader:
            """Declares the vendor that actually produced the primary reading."""

            def channel(self):
                return ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                                   kind=ChannelKind.VISION,
                                   provider=document.primary_channel.provider)

            def read_pages(self, page_numbers):
                read_calls.append(tuple(page_numbers))
                return {number: build_page_read(
                    SECONDARY_VISION_CHANNEL_ID, number, vision_text)
                    for number in page_numbers}

        result = _run(
            _reading("excision procedure alpha performed", "right",
                     "Procedure alpha performed today"),
            _reading("procedure alpha performed excision", "left",
                     "Condition alpha addressed today"),
            note_text=document.primary_text(),
            source_evidence=document,
            source_reader=_SameVendorReader())

        self.assertEqual(read_calls, [],
                         "a channel that cannot be independent must not be paid for")
        self.assertIsNot(result.verdict, Verdict.AUTO_READY)
        self.assertFalse(result.billable_lines,
                         "the encounter must hold with zero retrieval, not release")
        hold = next(g for g in result.gates if g.outcome is Outcome.UNKNOWN)
        self.assertTrue(hold.retryable, hold.detail)



# ---------------------------------------------------------------------------
# THE EVENT-CANDIDATE UNION (issue #6 F7-R3).
#
# The class above proves the second reading detects a disagreement about an event BOTH
# readings found. These prove the other half of what a second independent reading is
# for: RECALL. A performed service the primary extractor missed entirely used to be
# recorded in `unmatched_second` metadata and dropped -- canonical graph, eligible
# nodes, retrieval and `integrity_problems()` all continued as if the note documented
# one service, and the claim silently under-coded.
#
# Every case here runs through the real `code_encounter`, because the defect was
# precisely a mechanism that existed and was never wired to anything.

NOTE_UNION = ("Procedure alpha performed today on the left side. "
              "Procedure beta performed today on the left side. "
              "Condition alpha addressed today.")


def _multi_reading(*events):
    """One reading of the note, as raw extractor output. `events` are
    (fact_id, description, verbatim quote) triples."""
    import json
    return json.dumps({"facts": [
        {"fact_id": fact_id, "kind": "procedure", "description": description,
         "attributes": {"laterality": "left", "performer_id": "actor-1",
                        "billing_entity_id": "actor-1"},
         "disposition": "performed_today", "negated": False,
         "evidence": [quote], "confidence": 0.99}
        for fact_id, description, quote in events]})


def _union_source():
    """Two DISTINCT synthetic services.

    Deliberately distinct: if a recovered event were wrongly admitted it must show up as
    a visibly extra claim line, not as one the duplicate-code collapse could quietly
    absorb. No real medical code appears here.
    """
    from claude_coder.data_access import MockSource
    from claude_coder.models import CandidateCode

    class _ByDescription(MockSource):
        def retrieve(self, description, system, top_k=20):
            if system != "cpt":
                return []
            if "beta" in (description or "").lower():
                return [CandidateCode("PROC_Y", "cpt", "Procedure beta, each", 0.9)]
            return [CandidateCode("PROC_X", "cpt", "Procedure alpha, each", 0.9)]

    return _ByDescription(records={("PROC_X", "cpt"): {"active": True},
                                   ("PROC_Y", "cpt"): {"active": True}})


def _run_union(primary_reading, second_reading, **kwargs):
    from claude_coder.pipeline import code_encounter
    return code_encounter(
        "enc", kwargs.pop("note_text", NOTE_UNION), "2026-03-14",
        source=kwargs.pop("source", None) or _union_source(),
        extract_llm=lambda s, u: primary_reading,
        extract_llm_b=lambda s, u: second_reading,
        verify_llm=_stub_llm, corroborate_llm=_stub_llm,
        billing_context=_BILLING, audit_repository=_null_audit(), **kwargs)


_PRIMARY_ONE_SERVICE = ("F1", "excision procedure alpha performed",
                        "Procedure alpha performed today on the left side")
_SECOND_SAME_SERVICE = ("F1", "procedure alpha performed excision",
                        "Procedure alpha performed today on the left side")
_SECOND_EXTRA_SERVICE = ("F2", "procedure beta performed excision",
                         "Procedure beta performed today on the left side")


def _document(vision_text, pdf_lines):
    """A compiled `SourceEvidenceDocument` whose ORIGINAL page says `pdf_lines` and whose
    transcription says `vision_text`."""
    import tempfile
    from pathlib import Path as _Path

    from app.contracts.source_evidence import PAGE_SEPARATOR
    from app.ingestion.source_evidence import compile_source_evidence
    from tests.source_pdf import build_pdf, vision_extraction

    tmp = tempfile.TemporaryDirectory()
    pdf_path = _Path(tmp.name) / "note.pdf"
    pdf_path.write_bytes(build_pdf([pdf_lines]))
    document = compile_source_evidence(
        pdf_path,
        vision_extraction([vision_text], pdf_path=pdf_path,
                          metadata={"date_of_service": "2026-03-14"},
                          page_separator=PAGE_SEPARATOR))
    return document, tmp


class EndToEndEventCandidateUnion(unittest.TestCase):

    def _recovered(self, result):
        self.assertIsNotNone(result.consensus, "a second reading must be recorded")
        return list(result.consensus["recovered_events"])

    def _codes(self, result):
        return sorted(ln.chosen.code for ln in result.lines if ln.chosen)

    # ---- the omission: the exact reproduction the finding was raised on -------------
    def test_a_service_only_the_second_reading_found_is_recovered_and_billed(self):
        """Codex F7-R3, reproduced: the primary graph carries ONE eligible performed
        service; the second reading carries that service plus a second, independently
        anchored, billable one. The second service must not disappear merely because the
        primary extractor omitted it."""
        result = _run_union(
            _multi_reading(_PRIMARY_ONE_SERVICE),
            _multi_reading(_SECOND_SAME_SERVICE, _SECOND_EXTRA_SERVICE))

        recovered = self._recovered(result)
        self.assertEqual([r["verdict"] for r in recovered], ["admitted"], recovered)
        node_id = recovered[0]["node_id"]
        self.assertTrue(node_id, "an admitted event must carry a canonical graph id")

        # It is a NODE of the canonical graph, not a metadata footnote...
        self.assertIn(node_id, result.graph.nodes)
        self.assertIn("F1", result.graph.nodes)
        # ...it went through ELIGIBILITY like every primary event...
        self.assertIn(node_id, result.graph.eligible_node_ids(),
                      "a recovered event must be decided by the same eligibility gate")
        # ...it was RETRIEVED and CODED...
        self.assertEqual(self._codes(result), ["PROC_X", "PROC_Y"],
                         "the service the primary extractor missed must reach the claim")
        # ...and the graph is coherent, which the old silent omission also claimed.
        self.assertEqual(result.graph.integrity_problems(), ())

    def test_the_old_silent_omission_would_now_be_visible(self):
        """The control's own value, stated as a comparison: with the union in place the
        one-service reading and the two-service reading no longer produce the same
        claim."""
        one = _run_union(_multi_reading(_PRIMARY_ONE_SERVICE),
                         _multi_reading(_SECOND_SAME_SERVICE))
        two = _run_union(_multi_reading(_PRIMARY_ONE_SERVICE),
                         _multi_reading(_SECOND_SAME_SERVICE, _SECOND_EXTRA_SERVICE))
        self.assertEqual(self._codes(one), ["PROC_X"])
        self.assertEqual(self._codes(two), ["PROC_X", "PROC_Y"])
        self.assertEqual(one.consensus["recovered_events"], [])

    def test_a_recovered_service_is_proven_against_the_original_page(self):
        """Admission is gated on the SAME original-page reconciliation every released
        fact must pass -- not on the second model having asserted it."""
        vision_text = NOTE_UNION
        document, tmp = _document(
            vision_text,
            ["Procedure alpha performed today on the left side.",
             "Procedure beta performed today on the left side.",
             "Condition alpha addressed today."])
        self.addCleanup(tmp.cleanup)

        result = _run_union(
            _multi_reading(_PRIMARY_ONE_SERVICE),
            _multi_reading(_SECOND_SAME_SERVICE, _SECOND_EXTRA_SERVICE),
            note_text=document.primary_text(), source_evidence=document)

        recovered = self._recovered(result)
        self.assertEqual([r["verdict"] for r in recovered], ["admitted"], recovered)
        self.assertIn("original_page_reconciliation", recovered[0]["reason"],
                      "the ORIGINAL PAGE must be what admitted it")
        self.assertIn(recovered[0]["node_id"], result.graph.nodes)
        self.assertEqual(self._codes(result), ["PROC_X", "PROC_Y"])

    # ---- the duplicate: recall must not become double-billing -----------------------
    def test_a_reworded_duplicate_does_not_create_a_second_claim_line(self):
        """The second reading re-finds the SAME event in words too different to align.

        It quotes the same passage of the document, which is what makes it the same
        event -- and an unverified duplicate must never manufacture an extra line.
        """
        result = _run_union(
            _multi_reading(_PRIMARY_ONE_SERVICE),
            _multi_reading(("F1", "procedure alpha performed excision",
                            "Procedure alpha performed today"),
                           ("F2", "operative removal alpha lesion resection",
                            "alpha performed today on the left side")))

        recovered = self._recovered(result)
        self.assertEqual([r["verdict"] for r in recovered],
                         ["duplicate_of_primary"], recovered)
        self.assertEqual(recovered[0]["merged_into"], "F1")
        self.assertEqual(recovered[0]["node_id"], "",
                         "a duplicate must not be given a graph identity")
        # Proven on the GRAPH, not only on the line count: the duplicate-code collapse
        # downstream could hide an extra line, but it cannot hide an extra node.
        self.assertEqual(sorted(result.graph.nodes), ["F1"])
        self.assertEqual(self._codes(result), ["PROC_X"])
        self.assertEqual(len(result.billable_lines), 1, result.billable_lines)

    # ---- the conflict: the page, not the model, decides ------------------------------
    def test_a_recovered_event_the_page_contradicts_is_excluded_not_added(self):
        """The second reading finds an event the ORIGINAL PAGE does not support. It must
        not silently enter the graph -- and the encounter's real service must still
        bill."""
        vision_text = ("Procedure alpha performed today, side one. "
                       "Condition alpha addressed today.")
        # The recall extraction reads the document's own text layer, so its quotations
        # are verbatim there by construction and a contradiction can only come from a
        # READING OF THE PAGE. Here the text layer carries a line no reader of the page
        # image sees -- the transcription does not have it, and the paid independent
        # read does not find it either.
        document, tmp = _document(
            vision_text,
            ["Procedure alpha performed today, side one.",
             "Procedure beta performed today, side one.",
             "Condition alpha addressed today."])
        self.addCleanup(tmp.cleanup)

        result = _run_union(
            _multi_reading(("F1", "excision procedure alpha performed",
                            "Procedure alpha performed today, side one")),
            _multi_reading(("F1", "procedure alpha performed excision",
                            "Procedure alpha performed today, side one"),
                           ("F2", "procedure beta performed excision",
                            "Procedure beta performed today, side one")),
            note_text=document.primary_text(), source_evidence=document,
            source_reader=_independent_reader(vision_text))

        recovered = self._recovered(result)
        self.assertEqual([r["verdict"] for r in recovered],
                         ["rejected_source_contradicted"], recovered)
        self.assertEqual(recovered[0]["node_id"], "")
        self.assertEqual(sorted(result.graph.nodes), ["F1"],
                         "an event the page contradicts must not become a node")
        self.assertEqual(self._codes(result), ["PROC_X"],
                         "the documented service must still be coded")
        self.assertNotIn("PROC_Y", self._codes(result))

    # ---- the unverifiable: neither silently dropped nor silently included ------------
    def test_an_unverifiable_recovered_event_holds_instead_of_vanishing(self):
        """No channel can read the page the candidate is quoted from. "We could not
        check" is never "confirmed" -- and it is never "forget it" either."""
        from claude_coder.models import Destination, Outcome

        vision_text = NOTE_UNION
        # An image-only page: nothing in the compiled document can confirm or refute it.
        document, tmp = _document(vision_text, [])
        self.addCleanup(tmp.cleanup)

        result = _run_union(
            _multi_reading(_PRIMARY_ONE_SERVICE),
            _multi_reading(_SECOND_SAME_SERVICE, _SECOND_EXTRA_SERVICE),
            note_text=document.primary_text(), source_evidence=document)

        recovered = self._recovered(result)
        self.assertEqual([r["verdict"] for r in recovered],
                         ["held_unverified"], recovered)
        self.assertEqual(recovered[0]["node_id"], "",
                         "an unproven event must not silently enter the graph")
        self.assertEqual(sorted(result.graph.nodes), ["F1"])

        # ...and it did not silently vanish either: it is a gate on the encounter.
        gate = next(g for g in result.gates
                    if g.name.startswith("second_reading_event_unverified:"))
        self.assertIs(gate.outcome, Outcome.UNKNOWN, gate.detail)
        self.assertTrue(gate.retryable,
                        "an unread page is a dependency failure, not a coder's judgement")
        self.assertIsNot(result.destination, Destination.AUTO_READY)
        self.assertFalse(
            any(r["destination"] == Destination.REVIEW.value for r in result.routing),
            f"a page that could not be read must not reach a coder queue: "
            f"{result.routing}")

    # ---- the binding: a recovered event cannot bypass the certificate ----------------
    def test_a_recovered_event_is_bound_by_the_certificate_and_the_bundle(self):
        """It participates in the graph digest and the claim binding exactly like a
        primary event -- there is no path by which a recovered line is released without
        the graph that justifies it being attested."""
        from app.contracts.claim_bundle import (AuthorityBinding, EncounterContext,
                                                SourceDocument, bundle_from_coding_result)

        result = _run_union(
            _multi_reading(_PRIMARY_ONE_SERVICE),
            _multi_reading(_SECOND_SAME_SERVICE, _SECOND_EXTRA_SERVICE))
        node_id = self._recovered(result)[0]["node_id"]

        record = result.graph.certificate_record()
        self.assertIn(node_id, record["node_ids"])
        self.assertEqual(record["graph_sha256"], result.graph.graph_sha256())

        bundle = bundle_from_coding_result(
            result, source_document=SourceDocument(filename="n"),
            context=EncounterContext(), authority=AuthorityBinding())
        self.assertIn(node_id, bundle.graph.clinical_event_ids,
                      "a released recovered line must name the graph it rests on")
        self.assertEqual(bundle.graph.graph_sha256, result.graph.graph_sha256())
        self.assertFalse(any("not bound to the clinical graph" in b
                             for b in bundle.release_blockers()),
                         bundle.release_blockers())

    # ---- the relational context travels with the event -------------------------------
    def test_a_recovered_component_the_record_calls_part_of_another_is_not_billed_twice(self):
        """The second reading's own edges naming a recovered event are carried into the
        graph and validated like primary edges, so an event the record calls PART OF
        another service is demoted rather than becoming an extra line."""
        import json

        def _reading_with_part_of():
            return json.dumps({
                "facts": [
                    {"fact_id": "F1", "kind": "procedure",
                     "description": "procedure alpha performed excision",
                     "attributes": {"laterality": "left", "performer_id": "actor-1",
                                    "billing_entity_id": "actor-1"},
                     "disposition": "performed_today", "negated": False,
                     "evidence": ["Procedure alpha performed today on the left side"],
                     "confidence": 0.99},
                    {"fact_id": "F2", "kind": "procedure",
                     "description": "procedure beta performed excision",
                     "attributes": {"laterality": "left", "performer_id": "actor-1",
                                    "billing_entity_id": "actor-1"},
                     "disposition": "performed_today", "negated": False,
                     "evidence": ["Procedure beta performed today on the left side"],
                     "confidence": 0.99}],
                "relations": [
                    {"subject_event_id": "F2", "predicate": "part_of",
                     "object_event_id": "F1", "state": "asserted",
                     "evidence_fact_ids": ["F1", "F2"], "confidence": 0.9}]})

        result = _run_union(_multi_reading(_PRIMARY_ONE_SERVICE),
                            _reading_with_part_of())
        recovered = self._recovered(result)
        self.assertEqual([r["verdict"] for r in recovered], ["admitted"], recovered)
        node_id = recovered[0]["node_id"]
        # The EDGE came across with it, expressed in canonical graph ids.
        carried = [e for e in result.graph.edges
                   if node_id in e.endpoints() and "F1" in e.endpoints()]
        self.assertTrue(carried, [e.as_record() for e in result.graph.edges])
        self.assertEqual(result.graph.integrity_problems(), ())

    # ---- failure paths OF THE FIX ITSELF ---------------------------------------------
    def test_stranding_one_recovered_event_strands_what_depended_on_it(self):
        """The fix's own cascade. A recovered event held back because one of its edges
        cannot be carried is no longer in the graph -- so an edge naming it cannot be
        carried either, and whatever that edge DEMOTED must be held too. Stopping after
        one pass would admit a component with exactly the PART_OF that suppressed it
        deleted, which is the extra billable line this control exists to prevent."""
        import json

        note = ("Procedure alpha performed today on the left side. "
                "Procedure beta performed today on the left side. "
                "Procedure delta performed today on the left side. "
                "Condition alpha addressed today.")

        def _event(fact_id, description, quote):
            return {"fact_id": fact_id, "kind": "procedure", "description": description,
                    "attributes": {"laterality": "left", "performer_id": "actor-1",
                                   "billing_entity_id": "actor-1"},
                    "disposition": "performed_today", "negated": False,
                    "evidence": [quote], "confidence": 0.99}

        second = json.dumps({
            "facts": [
                _event("F1", "procedure alpha performed excision",
                       "Procedure alpha performed today on the left side"),
                _event("F2", "procedure beta performed excision",
                       "Procedure beta performed today on the left side"),
                _event("F3", "procedure delta performed excision",
                       "Procedure delta performed today on the left side"),
                # Quotes text that is not in the document at all: nothing can place it,
                # so it never enters the graph and never enters the id mapping.
                _event("F4", "procedure omega performed excision",
                       "Procedure omega performed today")],
            "relations": [
                {"subject_event_id": "F2", "predicate": "part_of",
                 "object_event_id": "F3", "state": "asserted",
                 "evidence_fact_ids": ["F2", "F3"], "confidence": 0.9},
                {"subject_event_id": "F3", "predicate": "part_of",
                 "object_event_id": "F4", "state": "asserted",
                 "evidence_fact_ids": ["F3"], "confidence": 0.9}]})

        result = _run_union(_multi_reading(_PRIMARY_ONE_SERVICE), second,
                            note_text=note)

        verdicts = {r["second_event_id"]: r["verdict"]
                    for r in self._recovered(result)}
        self.assertEqual(verdicts["F4"], "rejected_unanchored", verdicts)
        self.assertEqual(verdicts["F3"], "held_unverified", verdicts)
        self.assertEqual(verdicts["F2"], "held_unverified",
                         f"the cascade must reach the event whose demotion was "
                         f"deleted: {verdicts}")
        self.assertEqual(sorted(result.graph.nodes), ["F1"])
        self.assertEqual(self._codes(result), ["PROC_X"])

    def test_a_recovered_edge_the_relation_kernel_rejects_holds_only_the_recovery(self):
        """The other failure path of the fix. `validate_relations` fails closed BY
        RAISING, and at the pipeline boundary a raise is a whole-encounter system hold.
        A malformed edge from the SECOND reading must take the recovered events out, not
        the encounter down: the recovered set is validated on a trial copy first."""
        import json
        from claude_coder.models import Destination

        # The PRIMARY reading's quotation is not verbatim in the note, so it anchors
        # nothing -- which makes an edge whose only evidence reference is that event
        # unbindable, and therefore rejected by the relation kernel.
        primary = json.dumps({"facts": [
            {"fact_id": "F1", "kind": "procedure",
             "description": "excision procedure alpha performed",
             "attributes": {"laterality": "left", "performer_id": "actor-1",
                            "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Procedure omega performed today"], "confidence": 0.99}]})
        second = json.dumps({
            "facts": [
                {"fact_id": "F1", "kind": "procedure",
                 "description": "procedure alpha performed excision",
                 "attributes": {"laterality": "left", "performer_id": "actor-1",
                                "billing_entity_id": "actor-1"},
                 "disposition": "performed_today", "negated": False,
                 "evidence": ["Procedure alpha performed today on the left side"],
                 "confidence": 0.99},
                {"fact_id": "F2", "kind": "procedure",
                 "description": "procedure beta performed excision",
                 "attributes": {"laterality": "left", "performer_id": "actor-1",
                                "billing_entity_id": "actor-1"},
                 "disposition": "performed_today", "negated": False,
                 "evidence": ["Procedure beta performed today on the left side"],
                 "confidence": 0.99}],
            "relations": [
                {"subject_event_id": "F2", "predicate": "part_of",
                 "object_event_id": "F1", "state": "asserted",
                 "evidence_fact_ids": ["F1"], "confidence": 0.9}]})

        result = _run_union(primary, second)

        recovered = self._recovered(result)
        self.assertEqual([r["verdict"] for r in recovered], ["held_unverified"],
                         recovered)
        self.assertEqual(recovered[0]["node_id"], "")
        self.assertNotIn("second-reading-F2", result.graph.nodes)
        # ...and the ENCOUNTER is not taken down by it.
        self.assertIsNot(result.destination, Destination.SYSTEM_HOLD)
        self.assertFalse(any(g.name == "pre_retrieval_integrity" for g in result.gates),
                         [g.name for g in result.gates])


# ---------------------------------------------------------------------------
# INDEPENDENT DOCUMENT RECALL, AND OCCURRENCE CARDINALITY (issue #6 F7-R3, reopened)
#
# The class above proves recall across two extractions. Both of those extractions used
# to read ONE string -- the primary vision transcription -- so a service the
# TRANSCRIPTION omitted was invisible to both readings and no amount of second-model
# recall could reach it. And a mention the two readings worded differently was admitted
# as a NEW EVENT purely because it was quoted from a document region no primary event
# rested on, which claim assembly then turned into an extra billable UNIT on the
# reasoning that a maximum-units edit would catch anything excessive.
#
# These cases prove both halves: the recall extraction now reads an INDEPENDENT reading
# of the original document, and a repeated mention adds a billable occurrence only when
# the record documents one.

_RECALL_TRANSCRIPT = ("Procedure alpha performed today on the left side. "
                      "Condition alpha addressed today.")
#: What the ORIGINAL document actually says -- a service more than the transcription.
_RECALL_PAGE = ["Procedure alpha performed today on the left side.",
                "Procedure beta performed today on the left side.",
                "Condition alpha addressed today."]
#: The same document, whose extra line RE-DESCRIBES the service already transcribed.
_REDESCRIBED_PAGE = ["Procedure alpha performed today on the left side.",
                     "Alpha procedure completed on the left side.",
                     "Condition alpha addressed today."]


def _independent_reader(page_text, calls=None):
    """A paid page reader from a vendor that is genuinely independent of the primary."""
    from app.contracts.source_evidence import ChannelKind, ReadChannel, build_page_read
    from app.ingestion.source_evidence import SECONDARY_VISION_CHANNEL_ID

    class _Reader:
        def channel(self):
            return ReadChannel(channel_id=SECONDARY_VISION_CHANNEL_ID,
                               kind=ChannelKind.VISION, provider="openai")

        def read_pages(self, page_numbers):
            if calls is not None:
                calls.append(tuple(page_numbers))
            return {number: build_page_read(SECONDARY_VISION_CHANNEL_ID, number,
                                            page_text)
                    for number in page_numbers}

    return _Reader()


def _run_recall(primary_reading, second_llm, **kwargs):
    """`_run_union`, but with a CALLABLE second extractor so the test can see the text
    it was actually given."""
    from claude_coder.pipeline import code_encounter
    return code_encounter(
        "enc", kwargs.pop("note_text", NOTE_UNION), "2026-03-14",
        source=kwargs.pop("source", None) or _union_source(),
        extract_llm=lambda s, u: primary_reading,
        extract_llm_b=second_llm,
        verify_llm=_stub_llm, corroborate_llm=_stub_llm,
        billing_context=_BILLING, audit_repository=_null_audit(), **kwargs)


class IndependentDocumentRecall(unittest.TestCase):
    """Defect A: the recall extraction must read the DOCUMENT, not the first reading."""

    def _recovered(self, result):
        self.assertIsNotNone(result.consensus, "a second reading must be recorded")
        return list(result.consensus["recovered_events"])

    def _codes(self, result):
        return sorted(ln.chosen.code for ln in result.lines if ln.chosen)

    def test_a_service_the_transcription_itself_omitted_is_recovered(self):
        """Codex F7-R3 (reopened), reproduced exactly.

        The PRIMARY TRANSCRIPTION -- not merely the primary extraction -- omits a
        performed service entirely. An independent reading of the original document
        contains it. Both extractors used to be handed the transcription, so both missed
        it identically and the claim was silently short a line.
        """
        calls = []
        captured = []
        document, tmp = _document(_RECALL_TRANSCRIPT, _RECALL_PAGE)
        self.addCleanup(tmp.cleanup)
        self.assertNotIn("Procedure beta", document.primary_text(),
                         "the transcription must genuinely omit the second service")

        second = _multi_reading(_SECOND_SAME_SERVICE, _SECOND_EXTRA_SERVICE)

        def _second_llm(system, user):
            captured.append(user)
            return second

        result = _run_recall(
            _multi_reading(_PRIMARY_ONE_SERVICE), _second_llm,
            note_text=document.primary_text(), source_evidence=document,
            source_reader=_independent_reader(" ".join(_RECALL_PAGE), calls))

        # The recall extractor was handed a DIFFERENT reading of the document, and that
        # reading contains the omitted service.
        self.assertTrue(captured, "the second extractor must have been called")
        self.assertIn("Procedure beta performed today on the left side", captured[0],
                      "the recall extraction must read an independent reading of the "
                      "document, not the transcription that omitted the service")

        recovered = self._recovered(result)
        self.assertEqual([r["verdict"] for r in recovered], ["admitted"], recovered)
        self.assertIn(recovered[0]["node_id"], result.graph.nodes)
        # A passage the transcription does not contain is settled by an INDEPENDENT
        # read of the page image, not by the transcription's silence.
        self.assertEqual(calls, [(1,)],
                         "the paid read must be aimed at exactly the page carrying the "
                         "passage the two readings disagree about")
        self.assertEqual(self._codes(result), ["PROC_X", "PROC_Y"],
                         "the service the TRANSCRIPTION omitted must reach the claim")
        self.assertEqual(result.graph.integrity_problems(), ())

    def test_an_unconfirmed_transcription_omission_holds_and_is_never_dropped(self):
        """The failure path of the recovery: no third reading of the page exists.

        The independent reading carries a passage the transcription does not. One of the
        two is wrong and nothing available can say which, so the candidate is neither
        admitted nor discarded -- it holds, loudly, as system work."""
        from claude_coder.models import Destination, Outcome

        document, tmp = _document(_RECALL_TRANSCRIPT, _RECALL_PAGE)
        self.addCleanup(tmp.cleanup)

        result = _run_recall(
            _multi_reading(_PRIMARY_ONE_SERVICE),
            lambda s, u: _multi_reading(_SECOND_SAME_SERVICE, _SECOND_EXTRA_SERVICE),
            note_text=document.primary_text(), source_evidence=document)

        recovered = self._recovered(result)
        self.assertEqual([r["verdict"] for r in recovered], ["held_unverified"],
                         recovered)
        self.assertEqual(recovered[0]["node_id"], "")
        self.assertEqual(self._codes(result), ["PROC_X"])
        gate = next(g for g in result.gates
                    if g.name.startswith("second_reading_event_unverified:"))
        self.assertIs(gate.outcome, Outcome.UNKNOWN, gate.detail)
        self.assertTrue(gate.retryable)
        self.assertIsNot(result.destination, Destination.AUTO_READY)

    def test_a_passage_only_the_text_layer_carries_is_refused_when_the_page_denies_it(self):
        """The other failure path: the independent reading is the one that is wrong.

        A text layer can carry text no reader of the page would ever see. Two readings
        of the page image -- the transcription and an independent paid read -- both fail
        to find the passage, so the candidate is refused rather than admitted, and the
        encounter's real service still bills."""
        document, tmp = _document(_RECALL_TRANSCRIPT, _RECALL_PAGE)
        self.addCleanup(tmp.cleanup)

        result = _run_recall(
            _multi_reading(_PRIMARY_ONE_SERVICE),
            lambda s, u: _multi_reading(_SECOND_SAME_SERVICE, _SECOND_EXTRA_SERVICE),
            note_text=document.primary_text(), source_evidence=document,
            # The page image says what the transcription said: no second service.
            source_reader=_independent_reader(_RECALL_TRANSCRIPT))

        recovered = self._recovered(result)
        self.assertEqual([r["verdict"] for r in recovered],
                         ["rejected_source_contradicted"], recovered)
        self.assertEqual(recovered[0]["node_id"], "")
        self.assertEqual(sorted(result.graph.nodes), ["F1"])
        self.assertEqual(self._codes(result), ["PROC_X"])

    def test_a_recovered_event_that_re_describes_an_existing_one_does_not_double_it(self):
        """Defect A composed with Defect B, which is where an overbill would come from.

        The independent reading carries a passage the transcription omitted -- but it
        RE-DESCRIBES the service already on the claim in different words. It is recovered
        (nothing may assume it is a duplicate), it resolves to the same authoritative
        code, and it therefore becomes ONE line with ONE unit."""
        document, tmp = _document(_RECALL_TRANSCRIPT, _REDESCRIBED_PAGE)
        self.addCleanup(tmp.cleanup)

        result = _run_recall(
            _multi_reading(_PRIMARY_ONE_SERVICE),
            lambda s, u: _multi_reading(
                _SECOND_SAME_SERVICE,
                ("F2", "procedure alpha removal completed",
                 "Alpha procedure completed on the left side")),
            note_text=document.primary_text(), source_evidence=document,
            source_reader=_independent_reader(" ".join(_REDESCRIBED_PAGE)))

        recovered = self._recovered(result)
        self.assertEqual([r["verdict"] for r in recovered], ["admitted"], recovered)
        billable = result.billable_lines
        self.assertEqual([ln.chosen.code for ln in billable], ["PROC_X"], billable)
        self.assertEqual(billable[0].units, 1,
                         f"one service, described twice, is one unit: "
                         f"{billable[0].rationale}")

    def test_the_recall_reading_and_its_blind_spot_are_in_the_durable_record(self):
        """The control has to say what it actually read.

        "Nothing extra was found on this page" and "no reading other than the
        transcription ever covered this page" are different facts and only one of them
        is evidence, so both the reading used and the pages it could not cover are
        recorded rather than inferred from an empty recovery list."""
        from app.ingestion.source_evidence import EMBEDDED_TEXT_CHANNEL_ID

        covered, tmp = _document(_RECALL_TRANSCRIPT, _RECALL_PAGE)
        self.addCleanup(tmp.cleanup)
        result = _run_recall(
            _multi_reading(_PRIMARY_ONE_SERVICE),
            lambda s, u: _multi_reading(_SECOND_SAME_SERVICE),
            note_text=covered.primary_text(), source_evidence=covered)
        self.assertEqual(result.consensus["recall_reading_channel_id"],
                         EMBEDDED_TEXT_CHANNEL_ID)
        self.assertEqual(result.consensus["recall_uncovered_pages"], [])

        # An image-only page: no reading but the transcription covers it, so a service
        # the transcription omitted there is NOT recoverable — and the record says so.
        blind, tmp2 = _document(_RECALL_TRANSCRIPT, [])
        self.addCleanup(tmp2.cleanup)
        result = _run_recall(
            _multi_reading(_PRIMARY_ONE_SERVICE),
            lambda s, u: _multi_reading(_SECOND_SAME_SERVICE),
            note_text=blind.primary_text(), source_evidence=blind)
        self.assertEqual(result.consensus["recall_reading_channel_id"], "",
                         "no independent reading existed to run recall over")
        self.assertEqual(result.consensus["recall_uncovered_pages"], [1])

    def test_the_transcription_is_never_recorded_as_its_own_independent_checker(self):
        """A recall quotation may be CONFIRMED by the transcription, but the merged
        reconciliation must never list the primary channel among the channels that
        independently checked the claim — that would attest an independence the run
        never had."""
        document, tmp = _document(_RECALL_TRANSCRIPT, _RECALL_PAGE)
        self.addCleanup(tmp.cleanup)

        result = _run_recall(
            _multi_reading(_PRIMARY_ONE_SERVICE),
            lambda s, u: _multi_reading(_SECOND_SAME_SERVICE, _SECOND_EXTRA_SERVICE),
            note_text=document.primary_text(), source_evidence=document,
            source_reader=_independent_reader(" ".join(_RECALL_PAGE)))

        record = result.source_reconciliation
        self.assertIsNotNone(record)
        self.assertNotIn(document.primary_channel_id,
                         record.independent_channel_ids,
                         f"the transcription cannot be its own authority: "
                         f"{record.independent_channel_ids}")



class AxisComparisonIsNormalized(unittest.TestCase):
    """Codex F7-R3, round-9 re-review, defect D: `known_known_differences` used to
    compare axis values as raw, lowercased strings, so two spellings of the SAME
    documented value manufactured a distinguishing axis -- and therefore a false
    DISTINCT_EVENT verdict, and an extra billed occurrence -- purely from wording."""

    def test_equivalent_free_text_wording_is_not_a_documented_difference(self):
        from claude_coder import coreference as cr
        self.assertEqual(
            cr.known_known_differences({"laterality": "left side"},
                                       {"laterality": "Left"}), ())
        self.assertEqual(
            cr.known_known_differences({"approach": "open approach"},
                                       {"approach": "Open"}), ())

    def test_genuinely_different_free_text_values_still_differ(self):
        from claude_coder import coreference as cr
        self.assertEqual(
            cr.known_known_differences({"laterality": "left"},
                                       {"laterality": "right"}), ("laterality",))
        # A shared, non-distinguishing word ('approach') must not paper over two
        # genuinely different values -- plain token OVERLAP is not enough; one value's
        # tokens must be a subset of the other's, or the difference stands.
        self.assertEqual(
            cr.known_known_differences({"approach": "first approach"},
                                       {"approach": "other approach"}), ("approach",))

    def test_identifier_axes_are_never_normalized(self):
        """An identifier is either the same identifier or it is not -- stemming it
        could accidentally equate two different performers or split one apart."""
        from claude_coder import coreference as cr
        self.assertEqual(
            cr.known_known_differences({"performer_id": "actor-1"},
                                       {"performer_id": "actor-11"}),
            ("performer_id",))
        self.assertEqual(
            cr.known_known_differences({"performer_id": "Actor-1"},
                                       {"performer_id": "actor-1"}), ())

    def test_the_wording_variant_no_longer_overbills_end_to_end(self):
        """The claim-level reproduction: one event, mentioned twice with LATERALITY
        worded differently but meaning the same side. Before this fix the raw-string
        axis compare would have called this a documented difference and billed two
        units for one performed service."""
        import json

        note = ("Procedure alpha performed today on the left side. "
                "Alpha procedure completed on the Left. "
                "Condition alpha addressed today.")
        primary = json.dumps({"facts": [
            {"fact_id": "F1", "kind": "procedure",
             "description": "excision procedure alpha performed",
             "attributes": {"laterality": "left side", "performer_id": "actor-1",
                            "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Procedure alpha performed today on the left side"],
             "confidence": 0.99}]})
        second = json.dumps({"facts": [
            {"fact_id": "F1", "kind": "procedure",
             "description": "excision procedure alpha performed",
             "attributes": {"laterality": "left side", "performer_id": "actor-1",
                            "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Procedure alpha performed today on the left side"],
             "confidence": 0.99},
            {"fact_id": "F2", "kind": "procedure",
             "description": "procedure alpha removal completed",
             # SAME side, worded differently -- "Left" vs "left side".
             "attributes": {"laterality": "Left", "performer_id": "actor-1",
                            "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Alpha procedure completed on the Left"],
             "confidence": 0.99}]})
        result = _run_union(primary, second, note_text=note)
        billable = result.billable_lines
        self.assertEqual([ln.chosen.code for ln in billable], ["PROC_X"], billable)
        self.assertEqual(billable[0].units, 1,
                         f"one service, restated with equivalent laterality wording, "
                         f"is one unit: {billable[0].rationale}")


class OccurrenceCardinality(unittest.TestCase):
    """Defect B: a repeated MENTION is not a repeated SERVICE."""

    def _codes(self, result):
        return sorted(ln.chosen.code for ln in result.lines if ln.chosen)

    def test_one_event_mentioned_twice_in_different_words_is_one_unit(self):
        """Codex F7-R3 (reopened), reproduced exactly: one performed event, mentioned
        twice in different document regions with different wording, same anatomy,
        laterality, performer and episode, and no stated count, repeat or distinction
        anywhere in the note.

        The union used to admit the second mention as a new event purely because it was
        quoted from a region no primary event rested on; both resolved to one code; and
        claim assembly read "different wording" as a separately documented repeat and
        billed TWO units, justified by the maximum-units edit permitting two. A maximum
        is not evidence."""
        note = ("Procedure alpha performed today on the left side. "
                "Alpha procedure completed on the left side. "
                "Condition alpha addressed today.")

        result = _run_union(
            _multi_reading(_PRIMARY_ONE_SERVICE),
            _multi_reading(_SECOND_SAME_SERVICE,
                           ("F2", "procedure alpha removal completed",
                            "Alpha procedure completed on the left side")),
            note_text=note)

        eligible = [i for i in result.claim_line_intents
                    if i.state is eligibility.EligibilityState.ELIGIBLE_FOR_RETRIEVAL
                    and i.component is eligibility.ClaimComponent.SERVICE]
        self.assertEqual(len(eligible), 2,
                         "both mentions still reach retrieval -- nothing may assume "
                         "they are one service before the authoritative data says so")
        billable = result.billable_lines
        self.assertEqual([ln.chosen.code for ln in billable], ["PROC_X"], billable)
        self.assertEqual(billable[0].units, 1,
                         f"one documented service must bill one unit: "
                         f"{billable[0].rationale}")
        merged = next(ln for ln in result.lines
                      if ln.excluded_reason and ln.chosen
                      and ln.chosen.code == "PROC_X")
        self.assertIn("no second occurrence", merged.excluded_reason)

    def test_a_second_mention_in_a_new_region_still_takes_the_coreference_test(self):
        """Region novelty is recall evidence that a MENTION exists, never proof that an
        OCCURRENCE happened. A mention in a region no primary event rests on, whose
        documented action and axes are the primary event's, corefers to it."""
        note = ("Procedure alpha performed today on the left side. "
                "Alpha procedure completed on the left side. "
                "Condition alpha addressed today.")

        result = _run_union(
            _multi_reading(("F1", "alpha excision performed",
                            "Procedure alpha performed today on the left side")),
            _multi_reading(("F2", "alpha excised today",
                            "Alpha procedure completed on the left side")),
            note_text=note)

        recovered = list(result.consensus["recovered_events"])
        self.assertEqual([r["verdict"] for r in recovered],
                         ["duplicate_of_primary"], recovered)
        self.assertEqual(recovered[0]["merged_into"], "F1")
        self.assertEqual(recovered[0]["node_id"], "")
        self.assertEqual(sorted(result.graph.nodes), ["F1"])
        billable = result.billable_lines
        self.assertEqual([ln.chosen.code for ln in billable], ["PROC_X"])
        self.assertEqual(billable[0].units, 1)

    def test_two_services_the_record_distinguishes_remain_two_units(self):
        """The other direction, which the fix must not break: the record states two
        different lateralities, so it documents two occurrences and both are billed."""
        import json

        note = ("Procedure alpha performed today on the left side. "
                "Alpha procedure completed on the right side. "
                "Condition alpha addressed today.")
        primary = json.dumps({"facts": [
            {"fact_id": "F1", "kind": "procedure",
             "description": "excision procedure alpha performed",
             "attributes": {"laterality": "left", "performer_id": "actor-1",
                            "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Procedure alpha performed today on the left side"],
             "confidence": 0.99},
            {"fact_id": "F2", "kind": "procedure",
             "description": "procedure alpha removal completed",
             "attributes": {"laterality": "right", "performer_id": "actor-1",
                            "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Alpha procedure completed on the right side"],
             "confidence": 0.99}]})

        result = _run_union(primary, primary, note_text=note)

        billable = result.billable_lines
        self.assertEqual([ln.chosen.code for ln in billable], ["PROC_X"], billable)
        self.assertEqual(billable[0].units, 2,
                         f"two documented occurrences must bill two units: "
                         f"{billable[0].rationale}")
        self.assertIn("laterality", billable[0].rationale)

    def test_a_stated_count_is_the_only_thing_that_multiplies_one_mention(self):
        """A count the RECORD states is source-anchored cardinality and does bill as
        such -- it is the only thing that can multiply a single documented mention."""
        import json

        note = "Procedure alpha performed twice today on the left side."
        reading = json.dumps({"facts": [
            {"fact_id": "F1", "kind": "procedure",
             "description": "excision procedure alpha performed",
             "attributes": {"laterality": "left", "count": 2,
                            "performer_id": "actor-1",
                            "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Procedure alpha performed twice today on the left side"],
             "confidence": 0.99}]})

        result = _run_union(reading, reading, note_text=note)

        billable = result.billable_lines
        self.assertEqual([ln.chosen.code for ln in billable], ["PROC_X"], billable)
        self.assertEqual(billable[0].units, 2,
                         "a count the record states is source-anchored cardinality")

    def test_left_right_and_a_repeat_of_right_bill_two_not_three(self):
        """Codex F7-R3, round-9 re-review, defect B: comparing every new mention only
        against the FIRST line this code was ever seen on overcounted a genuine
        multi-occurrence cluster -- left, right, then a re-description of right used to
        compare the third mention against LEFT (still the first-seen representative)
        and add a third, undocumented unit. It must bill the two occurrences the
        record actually documents, not three."""
        import json

        note = ("Procedure alpha performed today on the left side. "
                "Alpha procedure completed on the right side. "
                "Alpha procedure completed on the right, redone. "
                "Condition alpha addressed today.")
        primary = json.dumps({"facts": [
            {"fact_id": "F1", "kind": "procedure",
             "description": "excision procedure alpha performed",
             "attributes": {"laterality": "left", "performer_id": "actor-1",
                            "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Procedure alpha performed today on the left side"],
             "confidence": 0.99},
            {"fact_id": "F2", "kind": "procedure",
             "description": "procedure alpha removal completed",
             "attributes": {"laterality": "right", "performer_id": "actor-1",
                            "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Alpha procedure completed on the right side"],
             "confidence": 0.99},
            {"fact_id": "F3", "kind": "procedure",
             "description": "alpha procedure redone",
             "attributes": {"laterality": "right", "performer_id": "actor-1",
                            "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Alpha procedure completed on the right, redone"],
             "confidence": 0.99}]})

        result = _run_union(primary, primary, note_text=note)

        billable = result.billable_lines
        self.assertEqual([ln.chosen.code for ln in billable], ["PROC_X"], billable)
        self.assertEqual(billable[0].units, 2,
                         f"two documented occurrences (left, right) must bill two "
                         f"units even with a third re-description of the right one: "
                         f"{billable[0].rationale}")

    def test_a_count_stated_only_on_a_later_mention_is_not_discarded(self):
        """Codex F7-R3, round-9 re-review, defect C: only the FIRST-SEEN line's units
        ever survived dedup, so a count stated on a LATER mention of the same
        occurrence was silently thrown away -- 'documented once, then again as twice'
        billed one instead of two. The record's stated count must be honored
        regardless of which mention it arrives on."""
        import json

        note = ("Procedure alpha performed today on the left side. "
                "Alpha procedure completed twice on the left side.")
        primary = json.dumps({"facts": [
            {"fact_id": "F1", "kind": "procedure",
             "description": "excision procedure alpha performed",
             "attributes": {"laterality": "left", "performer_id": "actor-1",
                            "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Procedure alpha performed today on the left side"],
             "confidence": 0.99},
            {"fact_id": "F2", "kind": "procedure",
             "description": "procedure alpha removal completed",
             "attributes": {"laterality": "left", "count": 2,
                            "performer_id": "actor-1",
                            "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Alpha procedure completed twice on the left side"],
             "confidence": 0.99}]})

        result = _run_union(primary, primary, note_text=note)

        billable = result.billable_lines
        self.assertEqual([ln.chosen.code for ln in billable], ["PROC_X"], billable)
        self.assertEqual(billable[0].units, 2,
                         f"a count stated on the second mention must still be "
                         f"honored: {billable[0].rationale}")

    def test_conflicting_documented_counts_hold_instead_of_guessing(self):
        """The other direction of defect C: two mentions of the same occurrence each
        state an EXPLICIT count, and the counts disagree. The record does not agree
        with itself, and picking either count would be a guess, so the line holds."""
        import json

        note = ("Procedure alpha performed twice today on the left side. "
                "Alpha procedure completed three times on the left side.")
        primary = json.dumps({"facts": [
            {"fact_id": "F1", "kind": "procedure",
             "description": "excision procedure alpha performed",
             "attributes": {"laterality": "left", "count": 2,
                            "performer_id": "actor-1",
                            "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Procedure alpha performed twice today on the left side"],
             "confidence": 0.99},
            {"fact_id": "F2", "kind": "procedure",
             "description": "procedure alpha removal completed",
             "attributes": {"laterality": "left", "count": 3,
                            "performer_id": "actor-1",
                            "billing_entity_id": "actor-1"},
             "disposition": "performed_today", "negated": False,
             "evidence": ["Alpha procedure completed three times on the left side"],
             "confidence": 0.99}]})

        result = _run_union(primary, primary, note_text=note)

        billable = result.billable_lines
        self.assertEqual(billable, [], "conflicting documented counts must hold, "
                                       "never guess either one")
        held = next(ln for ln in result.lines
                    if ln.documentation_gap and "different counts" in
                    ln.documentation_gap)
        self.assertIn("2", held.documentation_gap)
        self.assertIn("3", held.documentation_gap)


if __name__ == "__main__":
    unittest.main()
