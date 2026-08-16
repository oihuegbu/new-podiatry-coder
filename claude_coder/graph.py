"""The Clinical Evidence and Service Graph - ONE typed, source-anchored clinical
representation, shared by extraction, retrieval, validation, certification and claim
assembly (product directive section 3).

WHY THIS MODULE EXISTS

The primitives were already here and already enforced: anchored `EvidenceSpan`s, the
`RelationAssertion` kernel, measurement/ownership resolution, service episodes, and an
eligibility engine that gates retrieval. What was missing was ONE addressable object
binding them, so every stage argues about the same nodes and edges and a released claim
line can name the exact graph that justified it. Without it each stage carried its own
partial view -- `lines` here, `relations` there, `claim_line_intents` somewhere else --
and `ClaimBundle.GraphReference` could only be filled with an unfiltered dump of every
id the run happened to produce, which binds nothing.

WHAT IT IS

  * NODES  - one per extracted clinical event, carrying the typed axes that can change a
             claim: kind, code-free action, occurrence status, assertion certainty,
             beneficiary, every documented attribute axis (anatomy, laterality, count,
             depth/area/length, dose and units -- whatever the record stated), the
             resolved performer/organization/billing entity, the service episode, and the
             ORIGINAL-DOCUMENT location of every quotation behind it.
  * EDGES  - one per reconciled `RelationAssertion`, carrying predicate, state, the spans
             that grounded it and its independent-assertion count.
  * ROLE   - what claim role a node may take, derived from the ELIGIBILITY component and
             from nothing else. This is the boundary that stops a documented condition
             from manufacturing a service line: a node whose role is CLINICAL_CONDITION
             may be retrieved as a supported condition, and can never be released as a
             service.
  * CANNOT-LINK - the duplicate/distinctness constraints made explicit and durable
             instead of being computed inside a merge and discarded.

NO MEDICAL CODE APPEARS HERE, and none may. Kinds, predicates and axis names are the
extraction vocabulary; codes are resolved downstream from authoritative data and are
never an input to anything in this module.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .eligibility import (ClaimComponent, ClaimLineIntent, EligibilityState,
                          ServiceEpisode)
from .models import RelationPredicate, RelationState

#: Identity of THIS graph shape. A consumer that finds another value is reading a
#: different contract and must say so rather than guess which fields exist.
GRAPH_SCHEMA_VERSION = "clinical-evidence-graph-v1"


class NodeRole(str, Enum):
    """What CLAIM ROLE a node may take.

    Derived from the eligibility component and from nothing else -- not from the code
    system, not from a kind list -- so "may this become a billable service line?" has
    exactly one answer in the system, decided BEFORE retrieval.
    """

    SERVICE = "service"                        # may become a professional service line
    CLINICAL_CONDITION = "clinical_condition"  # may be coded as a supported condition only


_ROLE_FOR_COMPONENT: dict[ClaimComponent, NodeRole] = {
    ClaimComponent.SERVICE: NodeRole.SERVICE,
    ClaimComponent.DIAGNOSIS_SUPPORT: NodeRole.CLINICAL_CONDITION,
}


def _service_kind_values() -> frozenset[str]:
    """Which event kinds the ELIGIBILITY ENGINE treats as potential service lines.

    Read from that engine rather than restated here: two independent notions of "is this
    kind a service?" is exactly how a documented condition would end up carrying a
    service component with nothing noticing.
    """
    from .eligibility import _SERVICE_KINDS
    return frozenset(k.value for k in _SERVICE_KINDS)

#: Predicates whose two endpoints are interchangeable. Kept in step with the eligibility
#: engine deliberately: two different notions of "which way does this edge point" would
#: let a closure include an endpoint the gate excluded.
_SYMMETRIC = frozenset({RelationPredicate.SEPARATE_FROM, RelationPredicate.SAME_EPISODE_AS})


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


@dataclass(frozen=True)
class GraphNode:
    """One typed, source-anchored clinical event."""

    node_id: str
    kind: str
    role: NodeRole
    action: str                       # code-free clinical action (never a code)
    status: str                       # occurrence status (performed/ordered/planned/...)
    certain: bool
    experiencer: str                  # beneficiary axis: whose condition/event this is
    billable: bool
    attributes: dict[str, Any] = field(default_factory=dict)
    axis_confidence: dict[str, float] = field(default_factory=dict)
    evidence_span_ids: tuple[str, ...] = ()
    anchored: bool = False
    #: Where in the ORIGINAL document the quotations behind this node sit, and what an
    #: independent reading of those pages said about them (directive section 1 machinery).
    source_pages: tuple[int, ...] = ()
    source_page_image_sha256: tuple[str, ...] = ()
    source_reconciliation: tuple[str, ...] = ()
    performer_id: str = ""
    performer_function: str = ""
    organization_id: str = ""
    billing_entity_id: str = ""
    encounter_id: str = ""
    date_of_service: str | None = None
    service_episode_id: str | None = None
    #: Axes two independent readings disagreed on that the ORIGINAL PAGE could not
    #: settle. Non-empty means a targeted provider query, never a coder queue.
    axis_conflicts: tuple[str, ...] = ()

    def as_record(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id, "kind": self.kind, "role": self.role.value,
            "action": self.action, "status": self.status, "certain": self.certain,
            "experiencer": self.experiencer, "billable": self.billable,
            "attributes": dict(self.attributes),
            "axis_confidence": dict(self.axis_confidence),
            "evidence_span_ids": list(self.evidence_span_ids),
            "anchored": self.anchored,
            "source_pages": list(self.source_pages),
            "source_page_image_sha256": list(self.source_page_image_sha256),
            "source_reconciliation": list(self.source_reconciliation),
            "performer_id": self.performer_id,
            "performer_function": self.performer_function,
            "organization_id": self.organization_id,
            "billing_entity_id": self.billing_entity_id,
            "encounter_id": self.encounter_id,
            "date_of_service": self.date_of_service,
            "service_episode_id": self.service_episode_id,
            "axis_conflicts": list(self.axis_conflicts),
        }


@dataclass(frozen=True)
class GraphEdge:
    """One documented relationship between two events."""

    edge_id: str
    subject_id: str
    predicate: str
    object_id: str
    state: str
    symmetric: bool = False
    evidence_span_ids: tuple[str, ...] = ()
    reconciliation_status: str = ""
    reconciliation_evidence: tuple[str, ...] = ()
    corroboration_status: str = ""
    independent_support: int = 0

    def endpoints(self) -> tuple[str, str]:
        return (self.subject_id, self.object_id)

    def as_record(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id, "subject_id": self.subject_id,
            "predicate": self.predicate, "object_id": self.object_id,
            "state": self.state, "symmetric": self.symmetric,
            "evidence_span_ids": list(self.evidence_span_ids),
            "reconciliation_status": self.reconciliation_status,
            "reconciliation_evidence": list(self.reconciliation_evidence),
            "corroboration_status": self.corroboration_status,
            "independent_support": self.independent_support,
        }


@dataclass(frozen=True)
class CannotLink:
    """An explicit constraint that two events are NOT the same event.

    First-class on the graph rather than a local variable inside the merge, because the
    constraint is what stops a distinct service from being collapsed into a duplicate --
    and a claim reviewer is entitled to see it.
    """

    left_node_id: str
    right_node_id: str
    basis: str            # documented distinctness, or a known-known attribute conflict

    def pair(self) -> frozenset[str]:
        return frozenset((self.left_node_id, self.right_node_id))

    def as_record(self) -> dict[str, Any]:
        return {"left_node_id": self.left_node_id, "right_node_id": self.right_node_id,
                "basis": self.basis}


@dataclass(frozen=True)
class GraphBinding:
    """Exactly which graph a set of released lines rests on -- the payload that fills
    `ClaimBundle.GraphReference`."""

    clinical_event_ids: tuple[str, ...] = ()
    claim_line_intent_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    evidence_span_ids: tuple[str, ...] = ()

    def as_record(self) -> dict[str, Any]:
        return {"clinical_event_ids": list(self.clinical_event_ids),
                "claim_line_intent_ids": list(self.claim_line_intent_ids),
                "relation_ids": list(self.relation_ids),
                "evidence_span_ids": list(self.evidence_span_ids)}


class GraphIntegrityError(ValueError):
    """The graph contradicts itself. Raised only by callers that choose to; the graph
    itself REPORTS problems (`integrity_problems`) so the pipeline can route them to a
    typed BLOCKED outcome rather than losing them to an exception."""


@dataclass
class ClinicalGraph:
    """The single clinical representation for one encounter."""

    encounter_id: str
    date_of_service: str | None = None
    schema_version: str = GRAPH_SCHEMA_VERSION
    extraction_schema_version: str = ""
    relation_grammar_version: str = ""
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: tuple[GraphEdge, ...] = ()
    intents: tuple[ClaimLineIntent, ...] = ()
    episodes: tuple[ServiceEpisode, ...] = ()
    cannot_links: tuple[CannotLink, ...] = ()
    #: How a two-reading axis disagreement was settled, when a second reading ran.
    axis_resolutions: tuple[Any, ...] = ()
    #: Events a second independent reading reported that the primary reading did not.
    #: Recorded, never silently merged -- see `graph_consensus`.
    unmatched_second_reading: tuple[dict[str, Any], ...] = ()

    # ------------------------------------------------------------------ lookups
    def intent_for(self, node_id: str) -> ClaimLineIntent | None:
        for intent in self.intents:
            if node_id in (intent.clinical_event_ids or []):
                return intent
        return None

    def role_of(self, node_id: str) -> NodeRole | None:
        node = self.nodes.get(node_id)
        return node.role if node is not None else None

    def eligible_node_ids(self) -> tuple[str, ...]:
        out: list[str] = []
        for intent in self.intents:
            if intent.state is EligibilityState.ELIGIBLE_FOR_RETRIEVAL:
                out.extend(e for e in (intent.clinical_event_ids or []) if e in self.nodes)
        return tuple(dict.fromkeys(out))

    def edges_touching(self, node_ids: set[str]) -> tuple[GraphEdge, ...]:
        return tuple(e for e in self.edges
                     if e.subject_id in node_ids or e.object_id in node_ids)

    # -------------------------------------------------------------- integrity
    def integrity_problems(self) -> tuple[str, ...]:
        """Contradictions INSIDE the graph.

        These are coherence questions ("does this still describe one encounter?"), not
        policy questions. Any one of them means a downstream consumer would be reasoning
        about a representation that disagrees with itself, so the pipeline turns them
        into a hard stop rather than a retry.
        """
        out: list[str] = []
        for edge in self.edges:
            for endpoint in edge.endpoints():
                if endpoint not in self.nodes:
                    out.append(f"edge {edge.edge_id} names event {endpoint!r}, "
                               f"which is not a node of this graph")
        claimed: dict[str, str] = {}
        for intent in self.intents:
            for event_id in (intent.clinical_event_ids or []):
                if event_id not in self.nodes:
                    out.append(f"claim-line intent {intent.intent_id} names event "
                               f"{event_id!r}, which is not a node of this graph")
                    continue
                previous = claimed.get(event_id)
                if previous is not None and previous != intent.intent_id:
                    out.append(f"event {event_id!r} is claimed by two intents "
                               f"({previous} and {intent.intent_id})")
                claimed[event_id] = intent.intent_id
                expected = _ROLE_FOR_COMPONENT.get(intent.component)
                actual = self.nodes[event_id].role
                if expected is not None and actual is not expected:
                    out.append(
                        f"event {event_id!r} carries role {actual.value!r} but its "
                        f"intent {intent.intent_id} is a {intent.component.value} "
                        f"component")
                # THE eligibility-before-retrieval role boundary, enforced: only an
                # event of a kind the eligibility engine treats as a service may carry
                # a service component. A documented condition promoted into a service
                # component -- the one way a diagnosis could manufacture a billable
                # service line -- is an integrity failure, not a silent acceptance.
                kind = self.nodes[event_id].kind
                is_service_kind = kind in _service_kind_values()
                if (intent.component is ClaimComponent.SERVICE) is not is_service_kind:
                    out.append(
                        f"event {event_id!r} is a {kind!r} event but its intent "
                        f"{intent.intent_id} claims the "
                        f"{intent.component.value} role")
        # A cannot-link whose endpoints ended up inside ONE intent means a documented
        # distinctness was overridden by a merge -- the exact failure the constraint
        # exists to prevent.
        for constraint in self.cannot_links:
            left, right = constraint.left_node_id, constraint.right_node_id
            for intent in self.intents:
                members = set(intent.clinical_event_ids or [])
                if left in members and right in members:
                    out.append(
                        f"intent {intent.intent_id} merges events {left!r} and {right!r}, "
                        f"which a cannot-link constraint holds apart ({constraint.basis})")
        # An eligible event with no anchored quotation would be a billable line with no
        # source. The eligibility gate already refuses it; this proves the two agree.
        for node_id in self.eligible_node_ids():
            node = self.nodes[node_id]
            if not node.evidence_span_ids or not node.anchored:
                out.append(f"event {node_id!r} is eligible for retrieval with no "
                           f"anchored source quotation")
        return tuple(dict.fromkeys(out))

    # --------------------------------------------------------------- bindings
    def binding_for(self, node_ids) -> GraphBinding:
        """The graph a set of released lines actually rests on.

        The closure is one documented hop: the released events themselves, every other
        mention the same claim-line intent merged into them, and -- across every edge
        that touches them -- the events on the other end. That is what makes the binding
        MEANINGFUL rather than a dump: a released service names the conditions the record
        gave as its reason, the components the record called integral to it, and the
        services the record called distinct from it.
        """
        seeds = {str(n) for n in (node_ids or []) if str(n) in self.nodes}
        closure = set(seeds)
        for intent in self.intents:
            if seeds.intersection(intent.clinical_event_ids or []):
                closure.update(e for e in (intent.clinical_event_ids or [])
                               if e in self.nodes)
        edges = [e for e in self.edges
                 if e.subject_id in closure or e.object_id in closure]
        for edge in edges:
            closure.update(e for e in edge.endpoints() if e in self.nodes)
        # Intents are re-derived over the FULL closure so a supporting condition pulled
        # in by an edge brings its own intent with it.
        bound_intents = [i for i in self.intents
                         if closure.intersection(i.clinical_event_ids or [])]
        spans: set[str] = set()
        for node_id in closure:
            spans.update(self.nodes[node_id].evidence_span_ids)
        for edge in edges:
            spans.update(edge.evidence_span_ids)
            spans.update(edge.reconciliation_evidence)
        return GraphBinding(
            clinical_event_ids=tuple(sorted(closure)),
            claim_line_intent_ids=tuple(sorted({i.intent_id for i in bound_intents})),
            relation_ids=tuple(sorted({e.edge_id for e in edges})),
            evidence_span_ids=tuple(sorted(s for s in spans if s)),
        )

    def reference_payload(self, node_ids) -> dict[str, Any]:
        """`ClaimBundle.GraphReference` for a set of released lines, versions included."""
        binding = self.binding_for(node_ids)
        return {"extraction_schema_version": self.extraction_schema_version,
                "relation_grammar_version": self.relation_grammar_version,
                # The graph these ids point INTO, by content address -- the same
                # value `certificate_record()` binds. Ids are reusable; the digest
                # is what makes "the same graph" checkable. (Issue #6 F7-R1.)
                "graph_sha256": self.graph_sha256(),
                **binding.as_record()}

    # ------------------------------------------------------------------ audit
    def graph_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_record(), sort_keys=True, default=str).encode()).hexdigest()

    def certificate_record(self) -> dict[str, Any]:
        """What the RELEASE CERTIFICATE binds: the graph's identity, not its bulk.

        The full record is already durable in the audit repository, and the certificate
        binds that chain through `audit_record_hashes` — so what belongs here is the
        identity that makes the two inseparable: which graph, built to which schema and
        grammar, containing which nodes and edges, under which cannot-link constraints.
        Copying every node's attributes as well would double the artifact without making
        one more thing provable.
        """
        return {
            "schema_version": self.schema_version,
            "graph_sha256": self.graph_sha256(),
            "extraction_schema_version": self.extraction_schema_version,
            "relation_grammar_version": self.relation_grammar_version,
            "node_ids": sorted(self.nodes),
            "node_roles": {k: self.nodes[k].role.value for k in sorted(self.nodes)},
            "edge_ids": sorted({e.edge_id for e in self.edges}),
            "cannot_links": [c.as_record() for c in self.cannot_links],
            "integrity_problems": list(self.integrity_problems()),
        }

    def as_record(self) -> dict[str, Any]:
        """The durable audit record of the graph every later stage reasoned about."""
        return {
            "schema_version": self.schema_version,
            "extraction_schema_version": self.extraction_schema_version,
            "relation_grammar_version": self.relation_grammar_version,
            "encounter_id": self.encounter_id,
            "date_of_service": self.date_of_service,
            "nodes": [self.nodes[k].as_record() for k in sorted(self.nodes)],
            "edges": [e.as_record() for e in self.edges],
            "episodes": [{"episode_id": ep.episode_id, "encounter_id": ep.encounter_id,
                          "date_of_service": ep.date_of_service,
                          "event_ids": list(ep.event_ids),
                          "grouping_signals": list(ep.grouping_signals)}
                         for ep in self.episodes],
            "intents": [{"intent_id": i.intent_id, "component": i.component.value,
                         "state": i.state.value,
                         "clinical_event_ids": list(i.clinical_event_ids or []),
                         "mention_count": i.mention_count,
                         "distinctness_facts": list(i.distinctness_facts or [])}
                        for i in self.intents],
            "cannot_links": [c.as_record() for c in self.cannot_links],
            "axis_resolutions": [r.as_record() if hasattr(r, "as_record") else dict(r)
                                 for r in self.axis_resolutions],
            "unmatched_second_reading": [dict(u) for u in self.unmatched_second_reading],
            "integrity_problems": list(self.integrity_problems()),
        }


# ------------------------------------------------------------------ construction
def _node_from_fact(fact, intent: ClaimLineIntent | None, encounter_id: str,
                    date_of_service: str | None) -> GraphNode:
    attributes = dict(getattr(fact, "attributes", None) or {})
    spans = list(getattr(fact, "evidence", None) or [])
    pages: list[int] = []
    images: list[str] = []
    statuses: list[str] = []
    for span in spans:
        page = getattr(span, "page", None)
        if isinstance(page, int) and page not in pages:
            pages.append(page)
        image = _clean(getattr(span, "page_image_sha256", ""))
        if image and image not in images:
            images.append(image)
        status = _clean(getattr(span, "source_reconciliation", ""))
        if status and status not in statuses:
            statuses.append(status)
    role = (_ROLE_FOR_COMPONENT.get(intent.component) if intent is not None else None)
    if role is None:
        # No intent means eligibility produced no decision for this event; it can never
        # be released (the pipeline refuses retrieval for it), and the SAFE role is the
        # one that cannot become a service line.
        role = NodeRole.CLINICAL_CONDITION
    return GraphNode(
        node_id=_clean(getattr(fact, "fact_id", "")),
        kind=_clean(getattr(getattr(fact, "kind", None), "value", "")),
        role=role,
        action=_clean(getattr(fact, "description", "")),
        status=_clean(getattr(getattr(fact, "disposition", None), "value", "")),
        certain=bool(getattr(fact, "certain", True)),
        experiencer=_clean(getattr(fact, "experiencer", "")),
        billable=bool(getattr(fact, "billable", False)),
        attributes=attributes,
        axis_confidence=dict(getattr(fact, "axis_confidence", None) or {}),
        evidence_span_ids=tuple(_clean(getattr(s, "span_id", "")) for s in spans
                                if _clean(getattr(s, "span_id", ""))),
        anchored=any(getattr(s, "anchored", False) for s in spans),
        source_pages=tuple(pages),
        source_page_image_sha256=tuple(images),
        source_reconciliation=tuple(statuses),
        performer_id=_clean(attributes.get("performer_id")),
        performer_function=_clean(attributes.get("performer_function")),
        organization_id=_clean(attributes.get("organization_id")),
        billing_entity_id=_clean(attributes.get("billing_entity_id")),
        encounter_id=encounter_id,
        date_of_service=date_of_service,
        service_episode_id=(intent.service_episode_id if intent is not None else None),
        axis_conflicts=tuple(_clean(c) for c in
                             (getattr(fact, "axis_conflicts", None) or []) if _clean(c)),
    )


def _edge_from_relation(relation) -> GraphEdge:
    predicate = getattr(relation, "predicate", None)
    predicate_value = getattr(predicate, "value", None) or _clean(predicate)
    return GraphEdge(
        edge_id=_clean(getattr(relation, "relation_id", "")),
        subject_id=_clean(getattr(relation, "subject_event_id", "")),
        predicate=predicate_value,
        object_id=_clean(getattr(relation, "object_event_id", "")),
        state=_clean(getattr(getattr(relation, "state", None), "value", "")),
        symmetric=predicate in _SYMMETRIC,
        evidence_span_ids=tuple(_clean(s) for s in
                                (getattr(relation, "evidence_span_ids", None) or [])
                                if _clean(s)),
        reconciliation_status=_clean(getattr(relation, "reconciliation_status", "")),
        reconciliation_evidence=tuple(
            _clean(s) for s in (getattr(relation, "reconciliation_evidence", None) or [])
            if _clean(s)),
        corroboration_status=_clean(getattr(relation, "corroboration_status", "")),
        independent_support=int(getattr(relation, "independent_support", 0) or 0),
    )


def documented_cannot_links(relations, intents) -> tuple[CannotLink, ...]:
    """Every explicit "these are not the same event" constraint, from both sources the
    eligibility engine already uses: a documented distinctness assertion, and a
    known-known conflict on a distinguishing attribute axis.

    Reads the eligibility engine OWN conflict predicate rather than restating it, so the
    constraint recorded on the graph and the constraint the merge enforced cannot drift
    apart.
    """
    from .eligibility import _DISTINCT_ATTR_AXES, _known_known_conflict

    out: list[CannotLink] = []
    seen: set[tuple[frozenset[str], str]] = set()

    def add(left: str, right: str, basis: str) -> None:
        if not left or not right or left == right:
            return
        key = (frozenset((left, right)), basis)
        if key in seen:
            return
        seen.add(key)
        out.append(CannotLink(left_node_id=left, right_node_id=right, basis=basis))

    for relation in (relations or []):
        if (getattr(relation, "predicate", None) is RelationPredicate.SEPARATE_FROM
                and getattr(relation, "state", None) is RelationState.ASSERTED):
            add(_clean(getattr(relation, "subject_event_id", "")),
                _clean(getattr(relation, "object_event_id", "")),
                "documented distinctness asserted between these events")

    intent_list = list(intents or [])
    for i in range(len(intent_list)):
        for j in range(i + 1, len(intent_list)):
            left, right = intent_list[i], intent_list[j]
            if not _known_known_conflict(left, right):
                continue
            axes = [axis for axis in _DISTINCT_ATTR_AXES
                    if _clean((left.attributes or {}).get(axis)).lower()
                    and _clean((right.attributes or {}).get(axis)).lower()
                    and _clean((left.attributes or {}).get(axis)).lower()
                    != _clean((right.attributes or {}).get(axis)).lower()]
            basis = ("known values differ on distinguishing axis: "
                     + ", ".join(axes)) if axes else "known distinguishing-axis conflict"
            for a in (left.clinical_event_ids or []):
                for b in (right.clinical_event_ids or []):
                    add(_clean(a), _clean(b), basis)
    return tuple(out)


def build_graph(facts, relations, intents, *, encounter_id: str,
                date_of_service: str | None = None, episodes=None,
                extraction_schema_version: str = "",
                relation_grammar_version: str = "",
                axis_resolutions=None,
                unmatched_second_reading=None) -> ClinicalGraph:
    """Compile one encounter into the single clinical representation.

    Everything here is a projection of what earlier stages already decided -- facts from
    extraction + anchoring + source reconciliation, edges from the validated relation
    kernel, roles and episodes from the eligibility engine. Nothing is inferred: this
    module never decides billability, never invents a relationship, and never touches a
    code.
    """
    intent_list = tuple(intents or [])
    by_event: dict[str, ClaimLineIntent] = {}
    for intent in intent_list:
        for event_id in (intent.clinical_event_ids or []):
            by_event.setdefault(_clean(event_id), intent)
    nodes: dict[str, GraphNode] = {}
    for fact in (facts or []):
        node = _node_from_fact(fact, by_event.get(_clean(getattr(fact, "fact_id", ""))),
                               encounter_id, date_of_service)
        if node.node_id:
            nodes[node.node_id] = node
    return ClinicalGraph(
        encounter_id=encounter_id,
        date_of_service=date_of_service,
        extraction_schema_version=_clean(extraction_schema_version),
        relation_grammar_version=_clean(relation_grammar_version),
        nodes=nodes,
        edges=tuple(_edge_from_relation(r) for r in (relations or [])),
        intents=intent_list,
        episodes=tuple(episodes or []),
        cannot_links=documented_cannot_links(relations, intent_list),
        axis_resolutions=tuple(axis_resolutions or []),
        unmatched_second_reading=tuple(unmatched_second_reading or []),
    )
