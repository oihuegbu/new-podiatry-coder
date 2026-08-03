"""Stage 2 — Ontological linking (deterministic fact -> code).

A performed fact is mapped to a code by RETRIEVING candidates from the
authoritative source and keeping only those whose data descriptor is consistent
with, and entailed by, the documented attributes. When exactly one candidate
clearly entails the fact it is chosen deterministically; genuine ambiguity is
handed to bounded arbitration; nothing plausible means abstain-for-review.

All discrimination is against the DATA descriptor (e.g. the candidate's own
"...right foot" text vs the fact's documented laterality). No medical code, and
no code-specific rule, appears here — only generic clinical words (left/right,
token overlap). Swapping the data changes the outcome with no code change.
"""
from __future__ import annotations

import re

from .data_access import CodeSource
from .models import ClinicalFact, ResolutionMethod, ResolvedLine

_LATERALITY = {"left", "right", "bilateral"}
_DET_MIN_OVERLAP = 0.5      # the chosen descriptor must cover half the fact's tokens
_DET_MARGIN = 0.15          # …and clearly beat the runner-up


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower())


def _tokens(s: str) -> set[str]:
    return {t for t in _norm(s).split() if len(t) > 2}


def _fact_laterality(fact: ClinicalFact) -> str:
    lat = str(fact.attributes.get("laterality", "")).lower().strip()
    return lat if lat in _LATERALITY else ""


def _descriptor_sides(descriptor: str) -> set[str]:
    d = _norm(descriptor)
    return {w for w in _LATERALITY if re.search(rf"\b{w}\b", d)}


def _consistent(fact: ClinicalFact, descriptor: str) -> bool:
    """Reject a candidate whose descriptor states a DIFFERENT side than the note.
    A descriptor that is silent on laterality is not a contradiction."""
    fl = _fact_laterality(fact)
    if not fl or fl == "bilateral":
        return True
    sides = _descriptor_sides(descriptor)
    return (not sides) or (fl in sides)


def _overlap(fact: ClinicalFact, descriptor: str) -> float:
    ft = _tokens(fact.description)
    ft |= _tokens(" ".join(str(v) for v in fact.attributes.values()))
    if not ft:
        return 0.0
    return len(ft & _tokens(descriptor)) / len(ft)


def resolve(fact: ClinicalFact, source: CodeSource, top_k: int = 20) -> ResolvedLine:
    if not fact.billable:
        return ResolvedLine(
            fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
            rationale=f"not performed today (disposition={fact.disposition.value}) — not billed")

    query = fact.description + " " + " ".join(
        f"{k} {v}" for k, v in fact.attributes.items())
    candidates = source.retrieve(query.strip(), fact.system, top_k=top_k)

    consistent = [c for c in candidates if _consistent(fact, c.descriptor)]
    if not consistent:
        return ResolvedLine(
            fact=fact, chosen=None, alternatives=candidates[:5],
            method=ResolutionMethod.ABSTAINED,
            rationale="no retrieved code is consistent with the documented attributes")

    scored = sorted(consistent, key=lambda c: (_overlap(fact, c.descriptor), c.score),
                    reverse=True)
    top_ov = _overlap(fact, scored[0].descriptor)
    second_ov = _overlap(fact, scored[1].descriptor) if len(scored) > 1 else 0.0

    if top_ov >= _DET_MIN_OVERLAP and (len(scored) == 1 or top_ov - second_ov >= _DET_MARGIN):
        return ResolvedLine(
            fact=fact, chosen=scored[0], alternatives=scored[1:4],
            method=ResolutionMethod.DETERMINISTIC,
            rationale=(f"descriptor entails fact — token overlap {top_ov:.2f}, "
                       f"margin over next {top_ov - second_ov:.2f}"))

    # Real ambiguity: several descriptors fit comparably. Hand the shortlist to
    # bounded arbitration rather than guess.
    return ResolvedLine(
        fact=fact, chosen=None, alternatives=scored[:5],
        method=ResolutionMethod.ABSTAINED,
        rationale=f"{len(scored)} candidates within margin — needs arbitration")
