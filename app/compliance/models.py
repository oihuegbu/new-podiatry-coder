"""Canonical claim model + compliance finding contract.

The scrubber operates on these normalized objects, NOT on the raw LLM coding
output. The engine builds a `Claim` from a `CodingResult`-shaped dict once, so
every agent is decoupled from the coding format and unit-testable in isolation.
"""
from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Finding contract — what every agent emits. Powers the review queue.
# --------------------------------------------------------------------------- #
class Status(str, Enum):
    PASS = "PASS"      # filter satisfied
    WARN = "WARN"      # advisory; does not block a clean claim on its own
    FAIL = "FAIL"      # blocks the claim → routes to review


class DenialRisk(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Disposition(str, Enum):
    CLEAN = "CLEAN"    # zero FAIL findings — billable
    REVIEW = "REVIEW"  # one or more FAIL findings — human review required


class Finding(BaseModel):
    """One result from one compliance agent about one rule on one claim."""
    filter_id: str = Field(description="e.g. MUE_MAI, NCCI_PTP, GLOBAL_PERIOD")
    filter_name: str = ""
    status: Status = Status.PASS
    codes: list[str] = Field(default_factory=list, description="codes the finding is about")
    reason: str = ""
    recommendation: str = ""
    denial_risk: DenialRisk = DenialRisk.NONE
    source_rule: str = Field(
        default="",
        description="traceability: which authoritative rule + effective date triggered this",
    )
    auto_fixable: bool = False
    suggested_fix: dict = Field(default_factory=dict)

    @property
    def is_blocking(self) -> bool:
        return self.status == Status.FAIL


# --------------------------------------------------------------------------- #
# Canonical claim model
# --------------------------------------------------------------------------- #
class Diagnosis(BaseModel):
    code: str
    description: str = ""
    is_primary: bool = False
    laterality: str | None = None
    supporting_text: str = ""
    source_section: str = ""


class ClaimLine(BaseModel):
    """One billable procedure/supply line (CPT or HCPCS)."""
    code: str
    code_system: str = "CPT"          # CPT | HCPCS
    description: str = ""
    units: int = 1
    modifiers: list[str] = Field(default_factory=list)
    place_of_service: str | None = None
    linked_diagnoses: list[str] = Field(default_factory=list)
    is_addon: bool = False
    supporting_text: str = ""
    evidence_spans: list[str] = Field(default_factory=list)
    # populated by agents as they validate (kept for output/audit)
    findings: list[Finding] = Field(default_factory=list)


class Provider(BaseModel):
    npi: str | None = None
    specialty: str | None = None
    organization_name: str | None = None


class Payer(BaseModel):
    # "Unknown"/False, not "Medicare"/True — a bare Payer() (e.g. a future
    # call site or test that omits explicit args) must never silently claim
    # Medicare for a claim whose payer was never actually identified. This
    # exact bug was already found and fixed once at the one real call site
    # (engine.py's build_claim, which always passes explicit values derived
    # from payer_registry.parse_insurance_text) — these defaults matched it
    # until now, which meant they'd resurrect the same bug the moment
    # anything constructed a bare Payer().
    name: str = "Unknown"
    payer_id: str | None = None       # internal key — payers.json / prior_auth_required
    stedi_trading_partner_id: str | None = None  # Stedi's own payer ID — eligibility (270/271) calls only
    plan: str | None = None
    is_medicare: bool = False         # ORIGINAL (FFS) Medicare only — never Medicare Advantage
    kind: str = "unknown"             # medicare_ffs | medicare_advantage | commercial | medicaid | tricare | workers_comp | unknown
    # Bound to Original Medicare's coverage floor (FFS, and MA per 42 CFR
    # 422.101). Coverage-derived checks key off this instead of is_medicare.
    follows_medicare_coverage: bool = False


class Subscriber(BaseModel):
    """Patient/insured identifiers needed for eligibility (270) checks."""
    member_id: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    date_of_birth: str | None = None   # YYYYMMDD for X12
    authorization_number: str | None = None  # prior-auth on file, if any


class Claim(BaseModel):
    """Normalized representation of a single encounter ready for scrubbing."""
    document_id: str = ""
    date_of_service: date | None = None
    place_of_service: str | None = None
    # USPS state of the encounter (from insurance text / letterhead), used to
    # scope MAC-local coverage policies (LCDs/Articles) to their jurisdiction.
    # None = unknown → jurisdiction scoping stays conservative (no filtering).
    state: str | None = None
    provider: Provider = Field(default_factory=Provider)
    payer: Payer = Field(default_factory=Payer)
    subscriber: Subscriber = Field(default_factory=Subscriber)
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    lines: list[ClaimLine] = Field(default_factory=list)
    note_text: str = ""
    note_sections: dict = Field(default_factory=dict)
    # context carried from extraction (post-op status, etc.)
    prior_surgery_info: dict = Field(default_factory=dict)

    @property
    def primary_diagnosis(self) -> Diagnosis | None:
        for d in self.diagnoses:
            if d.is_primary:
                return d
        return self.diagnoses[0] if self.diagnoses else None


class ScrubResult(BaseModel):
    """Final output of the scrubber for one claim."""
    document_id: str = ""
    disposition: Disposition = Disposition.REVIEW
    findings: list[Finding] = Field(default_factory=list)
    # Per-filter execution trail: one entry per agent that ran, including
    # silent passes ({filter_id, filter_name, status, findings}). Without
    # this, a filter that stopped firing after a rule/data fix was
    # indistinguishable from one that was skipped or crashed — reviewers
    # reading two runs of the same note couldn't tell "checked and passed"
    # from "not checked" (raised on note 031 when a fixed false FAIL simply
    # vanished between runs).
    filter_results: list[dict] = Field(default_factory=list)
    clean: bool = False
    summary: str = ""

    @property
    def blocking_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.is_blocking]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.status == Status.WARN]

    def finalize(self, filter_count: int | None = None) -> "ScrubResult":
        blocking = self.blocking_findings
        self.clean = len(blocking) == 0
        self.disposition = Disposition.CLEAN if self.clean else Disposition.REVIEW
        if self.clean:
            n = len(self.warnings)
            count_phrase = f"all {filter_count}" if filter_count is not None else "all"
            self.summary = (
                f"Clean claim — passed {count_phrase} compliance filters."
                if n == 0
                else f"Clean claim — passed all filters with {n} advisory note(s)."
            )
        else:
            self.summary = (
                f"Routed to review — {len(blocking)} blocking issue(s) across "
                f"{len({f.filter_id for f in blocking})} filter(s)."
            )
        return self
