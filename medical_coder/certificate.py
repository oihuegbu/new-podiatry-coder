from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from .compiler import canonical_json, sha256_bytes
from .models import CandidateDecision, ClaimContext, DecisionCertificate, EvidenceGraph


def stable_hash(value: object) -> str:
    return sha256_bytes(canonical_json(value))


def build_certificate(
    encounter_id: str,
    snapshot_id: str,
    evidence: EvidenceGraph,
    context: ClaimContext,
    decisions: Iterable[CandidateDecision],
) -> DecisionCertificate:
    decision_tuple = tuple(decisions)
    evidence_hash = sha256_bytes(canonical_json(asdict(evidence)))
    context_hash = sha256_bytes(canonical_json(asdict(context)))
    payload = {
        "encounter_id": encounter_id,
        "snapshot_id": snapshot_id,
        "evidence_hash": evidence_hash,
        "context_hash": context_hash,
        "decisions": [asdict(decision) for decision in decision_tuple],
    }
    return DecisionCertificate(
        encounter_id=encounter_id,
        snapshot_id=snapshot_id,
        evidence_hash=evidence_hash,
        context_hash=context_hash,
        decisions=decision_tuple,
        decision_hash=sha256_bytes(canonical_json(payload)),
    )

