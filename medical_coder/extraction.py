from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .compiler import canonical_json
from .models import EvidenceFact, EvidenceGraph


class ExtractionContractError(ValueError):
    pass


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

