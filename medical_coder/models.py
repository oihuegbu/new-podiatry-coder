from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from typing import Any


class FactStatus(str, Enum):
    PERFORMED = "performed"
    PLANNED = "planned"
    ORDERED = "ordered"
    HISTORICAL = "historical"
    CONSIDERED = "considered"
    NEGATED = "negated"
    UNKNOWN = "unknown"


class DecisionState(str, Enum):
    SUPPORTED_REPORTABLE = "SUPPORTED_REPORTABLE"
    SUPPORTED_DIFFERENT_CLAIM = "SUPPORTED_DIFFERENT_CLAIM"
    SUPPORTED_PACKAGED = "SUPPORTED_PACKAGED"
    PERFORMED_INTEGRAL = "PERFORMED_INTEGRAL"
    DOCUMENTED_NOT_PERFORMED = "DOCUMENTED_NOT_PERFORMED"
    HISTORICAL = "HISTORICAL"
    ORDERED_ONLY = "ORDERED_ONLY"
    CANDIDATE_NOT_ENTAILED = "CANDIDATE_NOT_ENTAILED"
    CANDIDATE_INACTIVE = "CANDIDATE_INACTIVE"
    MORE_SPECIFIC_CANDIDATE_AVAILABLE = "MORE_SPECIFIC_CANDIDATE_AVAILABLE"
    SOURCE_CONFLICT = "SOURCE_CONFLICT"
    POLICY_REVIEW = "POLICY_REVIEW"
    PROVIDER_QUERY = "PROVIDER_QUERY"
    CLAIM_CONTEXT_REQUIRED = "CLAIM_CONTEXT_REQUIRED"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    COMPILER_ERROR = "COMPILER_ERROR"


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class SourceSpan:
    document_id: str
    start_offset: int
    end_offset: int
    verbatim_text: str
    page: int | None = None

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("source span requires document_id")
        if self.start_offset < 0 or self.end_offset <= self.start_offset:
            raise ValueError("invalid source offsets")
        if not self.verbatim_text:
            raise ValueError("source span requires verbatim text")

    def verify_against(self, document_text: str) -> None:
        if document_text[self.start_offset : self.end_offset] != self.verbatim_text:
            raise ValueError("source span does not match document bytes")


@dataclass(frozen=True)
class Normalization:
    concept_id: str
    system: str
    source: str
    confidence: float
    alternatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("normalization confidence must be within [0, 1]")


@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    fact_type: str
    text: str
    status: FactStatus
    source_spans: tuple[SourceSpan, ...]
    attributes: dict[str, Any] = field(default_factory=dict)
    normalization: Normalization | None = None
    certainty: str = "asserted"

    def __post_init__(self) -> None:
        if not self.fact_id or not self.fact_type or not self.text.strip():
            raise ValueError("fact identity, type, and text are required")
        if not self.source_spans:
            raise ValueError("every fact requires at least one source span")


@dataclass(frozen=True)
class EvidenceGraph:
    document_id: str
    document_hash: str
    facts: tuple[EvidenceFact, ...]
    relationships: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        ids = [fact.fact_id for fact in self.facts]
        if len(ids) != len(set(ids)):
            raise ValueError("fact identifiers must be unique")
        if any(span.document_id != self.document_id for fact in self.facts for span in fact.source_spans):
            raise ValueError("fact span belongs to another document")


@dataclass(frozen=True)
class ClaimContext:
    date_of_service: date
    payer_identifier: str
    payer_type: str
    claim_type: str
    billing_entity_role: str
    performing_entity_role: str
    place_of_service: str
    jurisdiction: str
    contract_profile: str
    authorization_status: str
    organizational_profile: str = ""
    facility_type: str = ""
    practice_profile: str = ""

    def missing_fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, value in asdict(self).items()
            if name != "date_of_service" and not str(value).strip()
        )


@dataclass(frozen=True)
class Candidate:
    artifact_id: str
    system: str
    version: str
    display: str
    score: float
    supporting_fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class GateResult:
    name: str
    status: GateStatus
    reason: str
    source_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateDecision:
    artifact_id: str
    state: DecisionState
    gates: tuple[GateResult, ...]
    autonomous_release: bool


@dataclass(frozen=True)
class DecisionCertificate:
    encounter_id: str
    snapshot_id: str
    evidence_hash: str
    context_hash: str
    decisions: tuple[CandidateDecision, ...]
    decision_hash: str

