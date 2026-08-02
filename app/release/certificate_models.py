"""Immutable, serializable models for claim release decisions."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ReadinessDisposition(str, Enum):
    AUTO_READY = "AUTO_READY"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    BLOCKED = "BLOCKED"


class ControlOutcome(str, Enum):
    PASS = "PASS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    BLOCKED = "BLOCKED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_CHECKED = "NOT_CHECKED"
    ERROR = "ERROR"


class ControlResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    control_id: str
    outcome: ControlOutcome
    reason: str = ""
    evidence: tuple[str, ...] = ()


class ClaimReadinessCertificate(BaseModel):
    """A frozen decision over one exact note/context/claim/source snapshot."""

    model_config = ConfigDict(frozen=True)

    certificate_version: int = 1
    document_id: str
    created_at: str
    disposition: ReadinessDisposition
    certificate_fingerprint: str
    note_fingerprint: str
    source_document_fingerprint: str
    context_fingerprint: str
    claim_fingerprint: str
    source_manifest_fingerprint: str
    rule_pack_fingerprint: str
    autonomous_scope_id: str = ""
    autonomous_scope_fingerprint: str = ""
    system_versions: dict[str, str] = Field(default_factory=dict)
    claim_payload: dict[str, Any] = Field(default_factory=dict)
    controls: tuple[ControlResult, ...]
    assumptions: tuple[str, ...] = ()
