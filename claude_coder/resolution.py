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
from .models import CandidateCode, ClinicalFact, FactKind, ResolutionMethod, ResolvedLine
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

    # AUTHORITATIVE FIRST: for a diagnosis, resolve through the ICD-10-CM
    # Alphabetic Index (clinician term -> code) before any embedding. This is the
    # permanent fix for the eponym / terse-descriptor gap — deterministic and
    # provenance-clean wherever the Index carries the term. The embedding is only
    # reached when the Index has no entry for the phrasing.
    if fact.kind is FactKind.DIAGNOSIS:
        idx = source.index_codes(fact.description, fact.system)
        # Trust the Index only for an UNAMBIGUOUS single-code mapping (the clean
        # authoritative wins, e.g. onychomycosis -> B35.1). A multi-code result is
        # a laterality family OR Index noise; either way defer to the embedding +
        # structured path, which disambiguates by documented evidence. This makes
        # the deterministic Index path safe against parse noise.
        if len(idx) == 1:
            pool = _authoritative_pool(next(iter(idx)), source)
            if pool:
                line = _decide(fact, pool, authority="ICD-10-CM Alphabetic Index")
                if line.resolved:
                    return line
        # SECOND authoritative layer: the SNOMED CT -> ICD-10-CM map (long-tail
        # eponyms/synonyms the ICD Index lacks, e.g. "Morton's neuroma"). Same
        # single-code trust rule; no-op until the map is ingested (needs UMLS).
        snomed = source.snomed_codes(fact.description, fact.system)
        if len(snomed) == 1:
            pool = _authoritative_pool(next(iter(snomed)), source)
            if pool:
                line = _decide(fact, pool, authority="SNOMED CT -> ICD-10-CM map")
                if line.resolved:
                    return line

    # Multi-query RECALL: search the structured query AND the verbatim evidence
    # (which often carries the eponym / clinician term the descriptor lacks),
    # then union the pools keeping each code's best relevance. Fallback for
    # phrasings the authoritative Index does not carry.
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

    return _decide(fact, pool)


def _candidate_from_code(code: str, source: CodeSource) -> CandidateCode:
    """Wrap an authoritative-Index code as a top-relevance candidate, descriptor
    from the authoritative record."""
    rec = source.lookup(code, "icd10") or {}
    desc = (rec.get("long_description") or rec.get("description")
            or rec.get("short_description") or "")
    return CandidateCode(code=code, system="icd10", descriptor=str(desc), score=1.0,
                         source="icd10-index",
                         authority={"source": "ICD-10-CM Alphabetic Index"})


def _authoritative_pool(code: str, source: CodeSource) -> list[CandidateCode]:
    """Expand an authoritative code to its billable LEAVES — a leaf stays itself,
    a category (e.g. M20.4-) becomes its children (M20.40/41/42) — so the
    structured decision can pick the specific code by documented laterality."""
    return [_candidate_from_code(c, source) for c in source.leaf_codes(code, "icd10")]


def _decide(fact: ClinicalFact, pool: list[CandidateCode],
            authority: str | None = None) -> ResolvedLine:
    """Structured decision over a candidate pool: eliminate contradictions,
    rank by relevance (recall) then specificity, pick deterministically when the
    leader is clear, else hand the shortlist to arbitration."""
    survivors = [m for m in (_evaluate(fact, c) for c in pool) if m is not None]
    if not survivors:
        return ResolvedLine(
            fact=fact, chosen=None, alternatives=pool[:5],
            method=ResolutionMethod.ABSTAINED,
            rationale="every candidate contradicts a documented attribute")

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

    tag = f" ({authority})" if authority else ""
    if deterministic:
        return ResolvedLine(
            fact=fact, chosen=top.candidate,
            alternatives=[m.candidate for m in survivors[1:4]],
            method=ResolutionMethod.DETERMINISTIC,
            rationale=f"{'; '.join(top.rationale)}{tag}")
    return ResolvedLine(
        fact=fact, chosen=None, alternatives=[m.candidate for m in survivors[:5]],
        method=ResolutionMethod.ABSTAINED,
        rationale=f"{len(survivors)} candidates comparable — arbitration{tag}")
