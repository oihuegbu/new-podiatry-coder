"""Stage 2 — Deterministic ontological resolution.

The DECISION is made by structured rules over descriptor features, not by vector
rank. Retrieval is demoted to what it is good at — RECALL: it narrows ~10^5 codes
to a small candidate pool. The resolver then evaluates each candidate field by
field against the documented fact and applies coding-guideline MECHANICS:

  • laterality contradiction   → eliminate  (a "left" descriptor for a right foot)
  • measurement out of range   → eliminate  (size 30 vs a "≤16 sq in" descriptor)
  • concept entailment floor    → eliminate  (the core action/site must match)
  • specificity preference      → rank       (a descriptor that positively matches
                                              more documented attributes wins)

The surviving code is chosen deterministically when it is the unique survivor or
dominates on specificity; genuine ties go to bounded arbitration. Every decision
carries a per-field rationale (the audit trail). None of the mechanics reference
a code — they operate on features parsed from the authoritative descriptors, so
the size-family selection the old pipeline HARDCODED is here derived from data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .data_access import CodeSource
from .models import CandidateCode, ClinicalFact, ResolutionMethod, ResolvedLine
from .ontology import DescriptorFeatures, measurement_of, parse_descriptor

_LATERALITY = {"left", "right", "bilateral"}
_STOP = _LATERALITY | {
    "of", "the", "and", "or", "with", "without", "to", "for", "a", "an", "in",
    "on", "by", "per", "each", "single", "size", "sterile", "unspecified",
}
_MIN_CONCEPT = 0.34        # the core concept must overlap at least this much
_DET_MARGIN = 0.15         # …and clearly beat the runner-up when specificity ties
_RECALL_POOL = 40


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(s).lower())
            if len(t) > 2 and t not in _STOP}


def _fact_laterality(fact: ClinicalFact) -> str:
    lat = str(fact.attributes.get("laterality", "")).lower().strip()
    return lat if lat in _LATERALITY else ""


def _fact_concept(fact: ClinicalFact) -> set[str]:
    toks = _tokens(fact.description)
    for k, v in fact.attributes.items():
        if str(k).lower() in ("laterality", "count", "quantity"):
            continue
        toks |= _tokens(str(v))
    return toks


@dataclass
class _Match:
    candidate: CandidateCode
    features: DescriptorFeatures
    concept: float
    specificity: int
    rationale: list[str] = field(default_factory=list)


def _evaluate(fact: ClinicalFact, cand: CandidateCode) -> _Match | None:
    """Apply the guideline mechanics. Return None if the candidate is
    eliminated, else a scored match with its per-field reasons."""
    feats = parse_descriptor(cand.descriptor)
    reasons: list[str] = []

    # RULE — laterality contradiction eliminates.
    fl = _fact_laterality(fact)
    if fl and fl != "bilateral" and feats.laterality and fl not in feats.laterality:
        return None

    # RULE — a documented measurement must fall in the descriptor's interval.
    measure = measurement_of(fact.attributes)
    if measure is not None and feats.interval and feats.interval.bounded():
        if not feats.interval.contains(measure):
            return None
        reasons.append(f"measure {measure:g} in descriptor range")

    # RULE — concept entailment floor.
    fc = _fact_concept(fact)
    concept = (len(fc & feats.core_tokens) / len(fc)) if fc else 0.0
    if concept < _MIN_CONCEPT:
        return None
    reasons.append(f"concept {concept:.2f}")

    # specificity — count the constraining attributes the descriptor POSITIVELY
    # accounts for (a more specific code wins per ICD-10-CM specificity guidance).
    spec = 0
    if fl and fl in feats.laterality:
        spec += 1
        reasons.append(f"laterality {fl}")
    if measure is not None and feats.interval and feats.interval.bounded():
        spec += 1

    return _Match(cand, feats, concept, spec, reasons)


def resolve(fact: ClinicalFact, source: CodeSource, top_k: int = _RECALL_POOL) -> ResolvedLine:
    if not fact.billable:
        return ResolvedLine(
            fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
            rationale=f"not performed today (disposition={fact.disposition.value}) — not billed")

    # Retrieval is RECALL ONLY — generate a candidate pool for the concept.
    query = fact.description + " " + " ".join(
        str(v) for k, v in fact.attributes.items() if str(k).lower() != "count")
    pool = source.retrieve(query.strip(), fact.system, top_k=top_k)
    if not pool:
        return ResolvedLine(fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
                            rationale="no candidate retrieved for the concept")

    # STRUCTURED decision over the pool.
    matches = [m for m in (_evaluate(fact, c) for c in pool) if m is not None]
    if not matches:
        return ResolvedLine(
            fact=fact, chosen=None, alternatives=pool[:5],
            method=ResolutionMethod.ABSTAINED,
            rationale="no retrieved code satisfies the documented attributes")

    matches.sort(key=lambda m: (m.specificity, m.concept), reverse=True)
    top = matches[0]
    if len(matches) == 1:
        deterministic = True
    else:
        nxt = matches[1]
        deterministic = (top.specificity > nxt.specificity
                         or top.concept - nxt.concept >= _DET_MARGIN)

    if deterministic:
        return ResolvedLine(
            fact=fact, chosen=top.candidate,
            alternatives=[m.candidate for m in matches[1:4]],
            method=ResolutionMethod.DETERMINISTIC,
            rationale="; ".join(top.rationale))

    return ResolvedLine(
        fact=fact, chosen=None, alternatives=[m.candidate for m in matches[:5]],
        method=ResolutionMethod.ABSTAINED,
        rationale=f"{len(matches)} candidates tie on structure — needs arbitration")
