from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .models import (
    Candidate,
    CandidateDecision,
    ClaimContext,
    ClaimLine,
    CodingResult,
    DecisionCertificate,
    DecisionState,
    EvidenceFact,
    EvidenceGraph,
    FactStatus,
    GateResult,
    GateStatus,
    Normalization,
    RoutingDecision,
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
                    raw_phrase=normalization.get("raw_phrase", ""),
                    candidate_expansions=tuple(normalization.get("candidate_expansions", [])),
                    resolution_factors=tuple(normalization.get("resolution_factors", [])),
                    target_artifact_id=normalization.get("target_artifact_id", ""),
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
    return context_from_dict(read_json(path))


def context_from_dict(value: dict[str, Any]) -> ClaimContext:
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


def coding_result_from_dict(value: dict[str, Any]) -> CodingResult:
    certificate_value = value["certificate"]
    decisions = tuple(
        CandidateDecision(
            artifact_id=item["artifact_id"],
            state=DecisionState(item["state"]),
            gates=tuple(
                GateResult(
                    name=gate["name"],
                    status=GateStatus(gate["status"]),
                    reason=gate["reason"],
                    source_refs=tuple(gate.get("source_refs", [])),
                )
                for gate in item["gates"]
            ),
            autonomous_release=bool(item["autonomous_release"]),
        )
        for item in certificate_value["decisions"]
    )
    certificate = DecisionCertificate(
        encounter_id=certificate_value["encounter_id"],
        snapshot_id=certificate_value["snapshot_id"],
        evidence_hash=certificate_value["evidence_hash"],
        context_hash=certificate_value["context_hash"],
        decisions=decisions,
        decision_hash=certificate_value["decision_hash"],
    )
    routing_value = value["routing"]
    return CodingResult(
        encounter_id=value["encounter_id"],
        state=value["state"],
        claim_lines=tuple(
            ClaimLine(
                artifact_id=item["artifact_id"],
                diagnosis_artifact_ids=tuple(item["diagnosis_artifact_ids"]),
                units=float(item["units"]),
                modifier_artifact_ids=tuple(item.get("modifier_artifact_ids", [])),
                disposition=DecisionState(item["disposition"]),
                source_fact_ids=tuple(item.get("source_fact_ids", [])),
            )
            for item in value["claim_lines"]
        ),
        routing=RoutingDecision(
            route=routing_value["route"],
            reason=routing_value["reason"],
            provider_query=routing_value.get("provider_query"),
            affected_fact_ids=tuple(routing_value.get("affected_fact_ids", [])),
        ),
        certificate=certificate,
        warnings=tuple(value.get("warnings", [])),
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

