"""Claim ownership — is the billing entity the one that actually performed the service?

Tri-state and fail-closed: unknown identity is UNKNOWN and cannot authorize retrieval.
Person-to-organization claims pass only with an explicit organization association and
performer function; a person id is never compared as though it were an organization id.

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


def classify_ownership(performer_id: str | None, billing_entity_id: str | None,
                       organization_id: str | None = None,
                       performer_function: str | None = None) -> Outcome:
    """Resolve claim ownership without name matching or implicit identity assumptions.

    A self-billing individual passes by id equality. A practitioner billing on behalf of
    an organization passes when the structured actor record explicitly binds the
    performer to that organization and records the function performed. Missing identity
    remains UNKNOWN; positive mismatch is BLOCKED.
    """
    if not performer_id or not billing_entity_id:
        return Outcome.UNKNOWN
    if str(performer_id) == str(billing_entity_id):
        return Outcome.PASS
    if organization_id and performer_function and str(organization_id) == str(billing_entity_id):
        return Outcome.PASS
    return Outcome.BLOCKED


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
