"""Stage 2 — Deterministic ontological resolution.

Division of labour, each component doing what it is good at:

  • RECALL (embedding retrieval) supplies the concept signal. Semantic
    similarity over enriched, synonym-bearing text is exactly what handles terse
    descriptors and clinician vocabulary ("Morton's neuroma" ≈ "Lesion of plantar
    nerve"). The pool is already cosine-thresholded, so relevance is the RAG's
    job — not a brittle token-overlap floor re-derived here (that floor wrongly
    eliminated correct-but-terse codes; it is gone).

  • STRUCTURED RULES make the decision. They are agnostic MECHANICS over features
    parsed from the authoritative descriptors — no code is named:
      – laterality contradiction    → ELIMINATE  (a "left" descriptor, right foot)
      – measurement out of range     → ELIMINATE  (size 30 vs a "≤16 sq in" code)
      – specificity                  → RANK       (a code that positively matches
                                                   more documented attributes wins,
                                                   per ICD-10-CM specificity rules)

A survivor is chosen deterministically when it is unique, dominates on
specificity, or clearly leads on recall; otherwise the ambiguity goes to bounded
arbitration. Every decision carries a per-field rationale (the audit trail).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .data_access import CodeSource
from .models import CandidateCode, ClinicalFact, ResolutionMethod, ResolvedLine
from .ontology import DescriptorFeatures, measurement_of, parse_descriptor

_LATERALITY = {"left", "right", "bilateral"}
_SCORE_MARGIN = 0.05       # recall lead that settles a pick among relevant candidates
_RELEVANCE_FLOOR = 0.6     # policy dial: min recall similarity for a deterministic pick
_RECALL_POOL = 40


def _fact_laterality(fact: ClinicalFact) -> str:
    lat = str(fact.attributes.get("laterality", "")).lower().strip()
    return lat if lat in _LATERALITY else ""


@dataclass
class _Match:
    candidate: CandidateCode
    features: DescriptorFeatures
    recall: float
    specificity: int
    rationale: list[str] = field(default_factory=list)


def _evaluate(fact: ClinicalFact, cand: CandidateCode) -> _Match | None:
    """Apply the agnostic elimination rules and score specificity. Return None if
    the candidate CONTRADICTS the documented facts, else a scored match. Concept
    relevance is not judged here — retrieval already guaranteed it."""
    feats = parse_descriptor(cand.descriptor)
    reasons: list[str] = []

    # ELIMINATION — laterality contradiction.
    fl = _fact_laterality(fact)
    if fl and fl != "bilateral" and feats.laterality and fl not in feats.laterality:
        return None

    # ELIMINATION — a documented measurement must fall in the descriptor's range.
    measure = measurement_of(fact.attributes)
    if measure is not None and feats.interval and feats.interval.bounded():
        if not feats.interval.contains(measure):
            return None

    # SPECIFICITY — count the constraining attributes the descriptor POSITIVELY
    # accounts for; a more specific code wins ties.
    spec = 0
    if fl and fl in feats.laterality:
        spec += 1
        reasons.append(f"laterality {fl}")
    if measure is not None and feats.interval and feats.interval.bounded():
        spec += 1
        reasons.append(f"measure {measure:g} in range")
    reasons.append(f"recall {cand.score:.2f}")

    return _Match(cand, feats, cand.score, spec, reasons)


def resolve(fact: ClinicalFact, source: CodeSource, top_k: int = _RECALL_POOL) -> ResolvedLine:
    if not fact.billable:
        return ResolvedLine(
            fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
            rationale=f"not performed today (disposition={fact.disposition.value}) — not billed")

    # Multi-query RECALL: search the structured query AND the verbatim evidence
    # (which often carries the eponym / clinician term the descriptor lacks),
    # then union the pools keeping each code's best relevance. Agnostic recall
    # boost; the eventual fix for eponyms is index enrichment at the data layer.
    query = fact.description + " " + " ".join(
        str(v) for k, v in fact.attributes.items() if str(k).lower() != "count")
    queries = [query.strip()] + [s.text for s in fact.evidence[:1]]
    best: dict[str, CandidateCode] = {}
    for q in queries:
        if not q.strip():
            continue
        for c in source.retrieve(q, fact.system, top_k=top_k):
            if c.code not in best or c.score > best[c.code].score:
                best[c.code] = c
    pool = sorted(best.values(), key=lambda c: c.score, reverse=True)
    if not pool:
        return ResolvedLine(fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
                            rationale="no candidate retrieved for the concept")

    survivors = [m for m in (_evaluate(fact, c) for c in pool) if m is not None]
    if not survivors:
        return ResolvedLine(
            fact=fact, chosen=None, alternatives=pool[:5],
            method=ResolutionMethod.ABSTAINED,
            rationale="every retrieved code contradicts a documented attribute")

    # Rank by concept relevance (recall) FIRST; specificity only breaks near-ties
    # among comparably-relevant codes — it must never promote a lower-recall code
    # over a more-relevant one (doing so once picked a laterality-matching but
    # semantically-wrong burn code for a neuroma). A deterministic pick also
    # requires the leader to clear the relevance floor; anything softer goes to
    # arbitration, which judges the descriptor against the fact and escalates on
    # a concept mismatch.
    survivors.sort(key=lambda m: (m.recall, m.specificity), reverse=True)
    top = survivors[0]
    if top.recall < _RELEVANCE_FLOOR:
        deterministic = False
    elif len(survivors) == 1:
        deterministic = True
    else:
        nxt = survivors[1]
        close = abs(top.recall - nxt.recall) < _SCORE_MARGIN
        deterministic = (top.recall - nxt.recall >= _SCORE_MARGIN
                         or (close and top.specificity > nxt.specificity))

    if deterministic:
        return ResolvedLine(
            fact=fact, chosen=top.candidate,
            alternatives=[m.candidate for m in survivors[1:4]],
            method=ResolutionMethod.DETERMINISTIC,
            rationale="; ".join(top.rationale))

    return ResolvedLine(
        fact=fact, chosen=None, alternatives=[m.candidate for m in survivors[:5]],
        method=ResolutionMethod.ABSTAINED,
        rationale=f"{len(survivors)} candidates comparable on structure & recall — arbitration")
