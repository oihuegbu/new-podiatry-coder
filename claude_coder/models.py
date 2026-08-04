"""Core data model for claude-medical-coder.

Provenance is built in BY CONSTRUCTION: every code that reaches a claim carries
the evidence span it came from, the clinical fact it resolved, the authoritative
record that defines it, and how it was chosen. There is not a single medical
code literal in this file — codes only ever arrive as data pulled from the
authoritative source at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FactKind(str, Enum):
    PROCEDURE = "procedure"
    DIAGNOSIS = "diagnosis"
    SUPPLY = "supply"                 # DME / dressing / therapeutic shoe / insert
    DRUG = "drug"
    IMAGING = "imaging"
    EM = "evaluation_management"


# Which authoritative code system a fact of this kind resolves against. This is
# a structural mapping (fact category -> code system), NOT a code mapping.
SYSTEM_FOR_KIND: dict[FactKind, str] = {
    FactKind.PROCEDURE: "cpt",
    FactKind.DIAGNOSIS: "icd10",
    FactKind.SUPPLY: "hcpcs",
    FactKind.DRUG: "hcpcs",
    FactKind.IMAGING: "cpt",
    FactKind.EM: "cpt",
}


class Disposition(str, Enum):
    """Was the service actually rendered? Only PERFORMED/dispensed work is
    billable; everything else must be excluded or reviewed, never coded."""
    PERFORMED = "performed_today"
    ORDERED = "ordered"
    PLANNED = "planned"
    DISCUSSED = "discussed"
    HISTORICAL = "historical"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class EvidenceSpan:
    """Verbatim text copied from the note — the atom of defensibility."""
    text: str
    section: str | None = None


@dataclass
class ClinicalFact:
    """A billable clinical event in plain clinical language — never a code.

    `attributes` carries the axes that DETERMINE a code (anatomy, laterality,
    count, depth, area, product, dose…) so a deterministic resolver can map the
    fact to a code from the authoritative data instead of the model guessing a
    code from memory.
    """
    kind: FactKind
    description: str
    attributes: dict[str, Any] = field(default_factory=dict)
    # Direct construction states a known intent (the callers here are trusted code
    # asserting a performed event). The fail-closed guard for UNTRUSTED input lives
    # at the trust boundary — extraction._coerce_disposition maps a missing/malformed
    # disposition from model output to UNCLEAR, so a real note never bills an event
    # whose disposition was not explicitly documented.
    disposition: Disposition = Disposition.PERFORMED
    evidence: list[EvidenceSpan] = field(default_factory=list)
    confidence: float = 0.0
    fact_id: str = ""

    @property
    def system(self) -> str:
        return SYSTEM_FOR_KIND[self.kind]

    @property
    def billable(self) -> bool:
        # A documented diagnosis is codeable (it establishes necessity) unless it
        # is purely historical. A procedure/supply/drug/imaging is codeable only
        # if it was actually PERFORMED / dispensed today — never if merely
        # ordered, planned or discussed.
        if self.kind is FactKind.DIAGNOSIS:
            return self.disposition is not Disposition.HISTORICAL
        return self.disposition is Disposition.PERFORMED


@dataclass(frozen=True)
class CandidateCode:
    """A code offered by the authoritative source for a fact. The descriptor,
    activity window and any policy attributes are copied straight from the data
    — the coder never authors them."""
    code: str
    system: str
    descriptor: str
    score: float = 0.0                 # recall relevance (similarity), for ranking
    source: str = ""
    authority: dict[str, Any] = field(default_factory=dict)   # data provenance


class ResolutionMethod(str, Enum):
    DETERMINISTIC = "deterministic"   # one candidate whose descriptor entails the fact
    ARBITRATED = "llm_arbitrated"     # model picked among retrieved candidates
    VERIFIED = "verified_entailment"  # candidate whose authoritative descriptor the
                                      # documentation entails (propose-then-verify)
    ABSTAINED = "abstained"           # genuine ambiguity / no candidate -> review


@dataclass
class ResolvedLine:
    fact: ClinicalFact
    chosen: CandidateCode | None
    alternatives: list[CandidateCode] = field(default_factory=list)
    method: ResolutionMethod = ResolutionMethod.ABSTAINED
    rationale: str = ""
    modifiers: list[str] = field(default_factory=list)   # data-driven, e.g. RT/LT/50
    units: int = 1                                        # billing units (descriptor-driven)
    # set when a resolved code is NOT a separately reportable line (bundled /
    # non-covered per data): it is kept for the audit trail but not billed.
    excluded_reason: str | None = None
    # set when the line escalated because the best-matching code needs an element
    # the documentation does not state — carries the specific gap for a provider query.
    documentation_gap: str | None = None

    @property
    def resolved(self) -> bool:
        return self.chosen is not None


class Outcome(str, Enum):
    """Gate outcomes are POSITIVE assertions. Release requires PASS or a proven
    NOT_APPLICABLE; UNKNOWN / ERROR / BLOCKED all stop autonomy. There is no
    "clean = absence of failure" — a check that did not run is not a pass."""
    PASS = "PASS"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


@dataclass
class GateResult:
    name: str
    outcome: Outcome
    detail: str = ""
    authority: str = ""               # what source/rule decided this
    # True when this gate could not clear because an AUTHORITY was UNAVAILABLE (data
    # not loaded / a lookup error) — an OPERATIONAL problem a retry can fix, NOT a
    # coding decision. The router sends these to SYSTEM_HOLD, never to a coder.
    retryable: bool = False

    @property
    def clears(self) -> bool:
        return self.outcome in (Outcome.PASS, Outcome.NOT_APPLICABLE)


class Verdict(str, Enum):
    AUTO_READY = "AUTO_READY"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    BLOCKED = "BLOCKED"


class Destination(str, Enum):
    """Where a non-auto-released encounter should ACTUALLY go — so operational
    failures, documentation gaps, and genuine coding judgement don't collapse into
    one human queue. A coder only ever sees REVIEW."""
    AUTO_READY = "AUTO_READY"          # release to billing, no human
    SYSTEM_HOLD = "SYSTEM_HOLD"        # operational/data failure -> retry + ops alert (not a coder)
    PROVIDER_QUERY = "PROVIDER_QUERY"  # documentation gap -> one structured question to the provider
    REVIEW = "REVIEW"                  # genuine coding/clinical judgement -> coder
    HOLD = "HOLD"                      # documentation cannot support a claim -> do not bill
    BLOCKED = "BLOCKED"                # a hard release gate failed


@dataclass
class CodingResult:
    encounter_id: str
    date_of_service: str | None
    lines: list[ResolvedLine] = field(default_factory=list)
    gates: list[GateResult] = field(default_factory=list)
    verdict: Verdict = Verdict.REVIEW_REQUIRED
    notes: list[str] = field(default_factory=list)
    certificate: dict[str, Any] | None = None   # tamper-evident evidence packet
    bypassed_ncci: list = field(default_factory=list)   # code pairs cleared by a modifier
    # actionable documentation recommendations (what to document/clarify to code it)
    recommendations: list[dict] = field(default_factory=list)
    # the actionable next-step destination (set by autonomy.decide) and the per-item
    # routing breakdown, so an encounter that isn't AUTO_READY is dispatched to the
    # RIGHT place (retry / provider / coder / hold) instead of one review queue.
    destination: "Destination | None" = None
    routing: list[dict] = field(default_factory=list)

    @property
    def billable_lines(self) -> list[ResolvedLine]:
        return [ln for ln in self.lines
                if ln.resolved and ln.fact.billable and not ln.excluded_reason]

    @property
    def procedure_lines(self) -> list[ResolvedLine]:
        return [ln for ln in self.billable_lines
                if ln.fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING, FactKind.EM)]

    @property
    def diagnosis_lines(self) -> list[ResolvedLine]:
        return [ln for ln in self.billable_lines
                if ln.fact.kind is FactKind.DIAGNOSIS]
