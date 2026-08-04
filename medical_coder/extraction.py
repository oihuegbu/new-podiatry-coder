from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from typing import Any

from .compiler import canonical_json
from .models import EvidenceFact, EvidenceGraph


class ExtractionContractError(ValueError):
    pass


@dataclass(frozen=True)
class ConsensusOutcome:
    evidence: EvidenceGraph | None
    conflicts: tuple[str, ...]


def reject_identifier_generation(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = "".join(character for character in key.casefold() if character.isalnum())
            if normalized == "code" or normalized.endswith("codes"):
                raise ExtractionContractError(f"extraction response contains forbidden identifier field at {path}.{key}")
            reject_identifier_generation(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            reject_identifier_generation(nested, f"{path}[{index}]")


def validate_extraction_payload(payload: dict[str, Any]) -> None:
    reject_identifier_generation(payload)
    allowed = {"document_id", "document_hash", "facts", "relationships"}
    unknown = set(payload) - allowed
    if unknown:
        raise ExtractionContractError(f"unknown extraction fields: {sorted(unknown)}")
    if not isinstance(payload.get("facts"), list):
        raise ExtractionContractError("extraction response requires a fact list")


def canonical_fact(fact: EvidenceFact) -> bytes:
    value = asdict(fact)
    value.pop("fact_id", None)
    return canonical_json(value)


def consensus(graphs: list[EvidenceGraph]) -> EvidenceGraph:
    if len(graphs) < 2:
        raise ExtractionContractError("independent extraction consensus requires at least two runs")
    document_ids = {graph.document_id for graph in graphs}
    document_hashes = {graph.document_hash for graph in graphs}
    if len(document_ids) != 1 or len(document_hashes) != 1:
        raise ExtractionContractError("extraction runs do not reference the same document")
    baseline = {canonical_fact(fact): fact for fact in graphs[0].facts}
    for graph in graphs[1:]:
        current = {canonical_fact(fact): fact for fact in graph.facts}
        if current.keys() != baseline.keys():
            raise ExtractionContractError("independent evidence graphs disagree materially")
    relationships = {json.dumps(item, sort_keys=True) for item in graphs[0].relationships}
    if any({json.dumps(item, sort_keys=True) for item in graph.relationships} != relationships for graph in graphs[1:]):
        raise ExtractionContractError("independent relationship graphs disagree materially")
    return EvidenceGraph(
        document_id=graphs[0].document_id,
        document_hash=graphs[0].document_hash,
        facts=tuple(baseline.values()),
        relationships=graphs[0].relationships,
    )


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))


def _span_overlap(left: EvidenceFact, right: EvidenceFact) -> bool:
    return any(
        l.document_id == r.document_id and l.start_offset < r.end_offset and r.start_offset < l.end_offset
        for l in left.source_spans for r in right.source_spans
    )


def _same_material_fact(left: EvidenceFact, right: EvidenceFact) -> bool:
    if left.status != right.status:
        return False
    left_identity = left.normalization.concept_id if left.normalization and not left.normalization.alternatives else ""
    right_identity = right.normalization.concept_id if right.normalization and not right.normalization.alternatives else ""
    if left_identity and right_identity:
        return left_identity == right_identity
    return _normalized_text(left.text) == _normalized_text(right.text) and _span_overlap(left, right)


def normalized_consensus(graphs: list[EvidenceGraph]) -> ConsensusOutcome:
    """Reconcile independent, source-normalized graphs without asking an LLM to arbitrate."""
    if len(graphs) < 2:
        raise ExtractionContractError("normalized consensus requires at least two provider graphs")
    if len({graph.document_id for graph in graphs}) != 1 or len({graph.document_hash for graph in graphs}) != 1:
        raise ExtractionContractError("provider graphs do not describe the same immutable document")
    baseline = list(graphs[0].facts)
    selected: list[EvidenceFact] = []
    conflicts: list[str] = []
    matched_by_graph: list[set[int]] = [set() for _ in graphs]
    identity_maps: list[dict[str, str]] = [dict() for _ in graphs]
    for baseline_index, fact in enumerate(baseline):
        matched_by_graph[0].add(baseline_index)
        identity_maps[0][fact.fact_id] = fact.fact_id
        matches = [fact]
        for graph_index, graph in enumerate(graphs[1:], 1):
            indexes = [index for index, candidate in enumerate(graph.facts) if _same_material_fact(fact, candidate)]
            if len(indexes) != 1:
                if fact.status.value in {"performed", "present", "ordered", "planned"}:
                    conflicts.append(f"material fact lacks unique cross-provider agreement at offset {fact.source_spans[0].start_offset}")
                break
            matched_by_graph[graph_index].add(indexes[0])
            matched = graph.facts[indexes[0]]
            identity_maps[graph_index][matched.fact_id] = fact.fact_id
            matches.append(matched)
        else:
            shared_attributes = {
                key: value
                for key, value in fact.attributes.items()
                if all(candidate.attributes.get(key) == value for candidate in matches[1:])
            }
            conflicting_keys = {
                key for candidate in matches for key in candidate.attributes
                if len({item.attributes.get(key) for item in matches}) > 1
            }
            if conflicting_keys:
                conflicts.append(
                    f"providers disagree on attributes {','.join(sorted(conflicting_keys))} for offset {fact.source_spans[0].start_offset}"
                )
                continue
            selected.append(replace(fact, attributes=shared_attributes))
    for graph_index, graph in enumerate(graphs[1:], 1):
        for index, fact in enumerate(graph.facts):
            if index not in matched_by_graph[graph_index] and fact.status.value in {"performed", "present", "ordered", "planned"}:
                conflicts.append(f"provider {graph_index + 1} found an unmatched material fact at offset {fact.source_spans[0].start_offset}")
    if conflicts:
        return ConsensusOutcome(None, tuple(sorted(set(conflicts))))
    selected_ids = {fact.fact_id for fact in selected}
    relationship_sets = []
    for graph_index, graph in enumerate(graphs):
        normalized_relationships = set()
        for relationship in graph.relationships:
            source = identity_maps[graph_index].get(str(relationship.get("source_fact_id")))
            target = identity_maps[graph_index].get(str(relationship.get("target_fact_id")))
            if source in selected_ids and target in selected_ids:
                normalized_relationships.add((source, target, str(relationship.get("relation"))))
        relationship_sets.append(normalized_relationships)
    if any(items != relationship_sets[0] for items in relationship_sets[1:]):
        return ConsensusOutcome(None, ("independent providers disagree on evidence relationships",))
    relationships = tuple(
        {"source_fact_id": source, "target_fact_id": target, "relation": relation}
        for source, target, relation in sorted(relationship_sets[0])
    )
    return ConsensusOutcome(
        EvidenceGraph(graphs[0].document_id, graphs[0].document_hash, tuple(selected), relationships),
        (),
    )

