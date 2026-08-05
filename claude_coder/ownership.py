"""Claim ownership — is the billing entity the one that actually performed the service?

Tri-state and fail-safe: a billed service is BLOCKED only when there is POSITIVE evidence
that a DIFFERENT actor performed it (performer id != billing-entity id). When ownership is
simply unstated — the common case, since most notes imply the biller from the signing
provider — the result is UNRESOLVED, which the gate treats as non-blocking (assumed
billing provider). This avoids self-DoS'ing every note that does not spell out its billing
entity, while still blocking a service the documentation attributes to someone else.

Decisions rest on resolved ACTOR IDENTIFIERS, never on name-string equality (which would
false-block the same person written two ways). Phase 1 populates actor ids via entity
resolution; until then ownership is UNRESOLVED and nothing is blocked on this axis.

Agnostic: no role vocabulary, no medical code — pure identity comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import Outcome


@dataclass(frozen=True)
class Ownership:
    performer_id: str | None = None
    performer_function: str | None = None
    organization_id: str | None = None
    billing_entity_id: str | None = None
    performer_name: str | None = None          # display only, never used for the decision


def classify_ownership(performer_id: str | None,
                       billing_entity_id: str | None) -> Outcome:
    """PASS when the billing entity performed the service; BLOCKED when a DIFFERENT actor
    did (positive contrary evidence); UNKNOWN when either identity is unstated (never
    blocks — the gate assumes the billing provider)."""
    if not performer_id or not billing_entity_id:
        return Outcome.UNKNOWN
    return Outcome.PASS if str(performer_id) == str(billing_entity_id) else Outcome.BLOCKED


def fact_ownership(fact) -> Ownership:
    """Read structured actor participation off a fact's attributes. Absent until Phase-1
    extraction populates it -> an all-None Ownership -> UNKNOWN (non-blocking)."""
    a = getattr(fact, "attributes", None) or {}
    return Ownership(
        performer_id=a.get("performer_id"),
        performer_function=a.get("performer_function"),
        organization_id=a.get("organization_id"),
        billing_entity_id=a.get("billing_entity_id"),
        performer_name=a.get("performer"),
    )
