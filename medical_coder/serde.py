from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .models import (
    Candidate,
    ClaimContext,
    EvidenceFact,
    EvidenceGraph,
    FactStatus,
    Normalization,
    SourceSpan,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_evidence(path: Path) -> EvidenceGraph:
    value = read_json(path)
    facts = []
    for item in value["facts"]:
        normalization = item.get("normalization")
        facts.append(
            EvidenceFact(
                fact_id=item["fact_id"],
                fact_type=item["fact_type"],
                text=item["text"],
                status=FactStatus(item["status"]),
                source_spans=tuple(SourceSpan(**span) for span in item["source_spans"]),
                attributes=item.get("attributes", {}),
                normalization=Normalization(
                    concept_id=normalization["concept_id"],
                    system=normalization["system"],
                    source=normalization["source"],
                    confidence=float(normalization["confidence"]),
                    alternatives=tuple(normalization.get("alternatives", [])),
                )
                if normalization
                else None,
                certainty=item.get("certainty", "asserted"),
            )
        )
    return EvidenceGraph(
        document_id=value["document_id"],
        document_hash=value["document_hash"],
        facts=tuple(facts),
        relationships=tuple(value.get("relationships", [])),
    )


def load_context(path: Path) -> ClaimContext:
    value = read_json(path)
    return ClaimContext(
        date_of_service=date.fromisoformat(value["date_of_service"]),
        payer_identifier=value["payer_identifier"],
        payer_type=value["payer_type"],
        claim_type=value["claim_type"],
        billing_entity_role=value["billing_entity_role"],
        performing_entity_role=value["performing_entity_role"],
        place_of_service=value["place_of_service"],
        jurisdiction=value["jurisdiction"],
        contract_profile=value["contract_profile"],
        authorization_status=value["authorization_status"],
        organizational_profile=value.get("organizational_profile", ""),
        facility_type=value.get("facility_type", ""),
        practice_profile=value.get("practice_profile", ""),
    )


def load_candidates(path: Path) -> list[Candidate]:
    return [
        Candidate(
            artifact_id=item["artifact_id"],
            system=item["system"],
            version=item["version"],
            display=item["display"],
            score=float(item["score"]),
            supporting_fact_ids=tuple(item["supporting_fact_ids"]),
        )
        for item in read_json(path)
    ]

