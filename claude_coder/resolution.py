"""Stage 2 — Deterministic ontological resolution.

Division of labour, each component doing what it is good at:

  • RECALL (embedding retrieval) supplies the concept signal. Semantic
    similarity over enriched, synonym-bearing text is exactly what handles terse
    descriptors and clinician vocabulary (a clinician eponym ≈ a terse anatomic
    descriptor). The pool is already cosine-thresholded, so relevance is the RAG's
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
from .ontology import (DescriptorFeatures, measurement_of, parse_descriptor,
                       support_score)

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
    support: int = 0
    rationale: list[str] = field(default_factory=list)


def _fact_text(fact: ClinicalFact) -> str:
    """The documented words for this fact: its description plus verbatim evidence
    — the text a candidate descriptor's concept tokens are checked against."""
    return " ".join([fact.description] + [s.text for s in fact.evidence])


def _evaluate(fact: ClinicalFact, cand: CandidateCode,
              source: CodeSource | None = None) -> _Match | None:
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

    # SUPPORT (mechanic 2) — how many concept tokens the note shares with the
    # code's AUTHORITATIVE descriptors. Scored over ALL description tiers (long /
    # medium / plain-language consumer), so plain wording that distinguishes near-
    # homographs (similar wording, different act) participates. A RANK signal ONLY (breaks
    # near-ties in recall); never an elimination, so a correct-but-terse code is
    # never dropped by it.
    desc_text = cand.descriptor
    if source is not None:
        try:
            tiers = source.descriptions(cand.code, cand.system)
            if tiers:
                desc_text = " ".join([cand.descriptor, *tiers])
        except Exception:
            pass
    support = support_score(desc_text, _fact_text(fact))
    reasons.append(f"recall {cand.score:.2f}")

    return _Match(cand, feats, cand.score, spec, support, reasons)


def resolve(fact: ClinicalFact, source: CodeSource, top_k: int = _RECALL_POOL,
            llm=None, corroborate=None) -> ResolvedLine:
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
        # authoritative wins, an unambiguous single code). A multi-code result is
        # a laterality family OR Index noise; either way defer to the embedding +
        # structured path, which disambiguates by documented evidence. This makes
        # the deterministic Index path safe against parse noise.
        if len(idx) == 1:
            pool = _authoritative_pool(next(iter(idx)), source)
            if pool:
                line = _decide(fact, pool, authority="ICD-10-CM Alphabetic Index", source=source)
                if line.resolved:
                    return line
        # SECOND authoritative layer: the SNOMED CT -> ICD-10-CM map (long-tail
        # eponyms/synonyms the ICD Index lacks). Same
        # single-code trust rule; no-op until the map is ingested (needs UMLS).
        snomed = source.snomed_codes(fact.description, fact.system)
        if len(snomed) == 1:
            pool = _authoritative_pool(next(iter(snomed)), source)
            if pool:
                line = _decide(fact, pool, authority="SNOMED CT -> ICD-10-CM map", source=source)
                if line.resolved:
                    return line

    # AUTHORITATIVE FIRST (procedure axis, mechanic 5): resolve a procedure/supply/
    # imaging phrase through the CPT/HCPCS descriptor index before any embedding —
    # the deterministic analog of the ICD Index. Same single-code trust rule: a
    # unique descriptor match is taken deterministically; anything else defers to
    # recall (which handles the many-competitor / terse cases).
    elif fact.kind in (FactKind.PROCEDURE, FactKind.SUPPLY, FactKind.IMAGING,
                       FactKind.DRUG):
        # AUTHORITATIVE FIRST (drug axis): the CMS Table of Drugs & Biologicals
        # (drug name -> HCPCS code). A dosed drug resolves by name deterministically
        # here before any embedding; empty until the table is prepared, so it
        # degrades to the descriptor index + recall below.
        if fact.kind is FactKind.DRUG:
            didx = source.drug_index_codes(fact.description, fact.system)
            if len(didx) == 1:
                code = next(iter(didx))
                rec = source.lookup(code, fact.system) or {}
                desc = (rec.get("long_description") or rec.get("description")
                        or rec.get("short_description") or "")
                cand = CandidateCode(code=code, system=fact.system, descriptor=str(desc),
                                     score=1.0, source="cms-table-of-drugs",
                                     authority={"source": "CMS Table of Drugs & Biologicals"})
                line = _decide(fact, [cand],
                               authority="CMS Table of Drugs & Biologicals", source=source)
                if line.resolved:
                    return line

        # AUTHORITATIVE FIRST: the AMA CPT Alphabetic Index (term -> code), the true
        # analog of the ICD Index. This is what resolves a documented procedure
        # phrase where descriptor/embedding cannot (a note's specific value vs a
        # descriptor's 'other than <a different value>'). Empty until the licensed
        # Index file is ingested (see
        # data_access.cpt_index_codes / tools/parse_cpt_index.py), so it is a no-op
        # that degrades gracefully to the descriptor index + embedding below.
        cidx = source.cpt_index_codes(fact.description, fact.system)
        if len(cidx) == 1:
            code = next(iter(cidx))
            rec = source.lookup(code, fact.system) or {}
            desc = (rec.get("long_description") or rec.get("description")
                    or rec.get("short_description") or "")
            cand = CandidateCode(code=code, system=fact.system, descriptor=str(desc),
                                 score=1.0, source="cpt-alphabetic-index",
                                 authority={"source": "AMA CPT Alphabetic Index"})
            line = _decide(fact, [cand], authority="AMA CPT Alphabetic Index", source=source)
            if line.resolved:
                return line

        # LEARNED verified-resolution index: a phrase this coder has resolved and had
        # confirmed across enough distinct encounters resolves DETERMINISTICALLY here
        # (no LLM), with provenance — the buildable, license-clean path toward the
        # Index's determinism. Self-invalidating in data_access; empty until promoted.
        lidx = source.learned_index_codes(fact.description, fact.system)
        if len(lidx) == 1:
            code = next(iter(lidx))
            rec = source.lookup(code, fact.system) or {}
            desc = (rec.get("long_description") or rec.get("description")
                    or rec.get("short_description") or "")
            cand = CandidateCode(code=code, system=fact.system, descriptor=str(desc),
                                 score=1.0, source="learned-verified-index",
                                 authority={"source": "learned verified-resolution index"})
            line = _decide(fact, [cand],
                           authority="learned verified-resolution index", source=source)
            if line.resolved:
                return line

        pidx = source.procedure_index_codes(fact.description, fact.system)
        if len(pidx) == 1:
            code = next(iter(pidx))
            rec = source.lookup(code, fact.system) or {}
            desc = (rec.get("long_description") or rec.get("description")
                    or rec.get("short_description") or "")
            cand = CandidateCode(code=code, system=fact.system, descriptor=str(desc),
                                 score=1.0, source="cpt-descriptor-index",
                                 authority={"source": "CPT/HCPCS descriptor index"})
            line = _decide(fact, [cand], authority="CPT/HCPCS descriptor index", source=source)
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

    # PROPOSE-THEN-VERIFY (when an LLM is available): widen the pool with
    # authoritative-validated LLM proposals, then accept the first candidate whose
    # OFFICIAL descriptor the documentation entails; escalate otherwise. Applies to
    # procedures/imaging AND to DIAGNOSES that reached the embedding fallback — an
    # ICD Index / SNOMED hit already returned deterministically above, so this only
    # verifies the UNGROUNDED embedding picks (the ones that were confidently wrong,
    # e.g. a code asserting a qualifier the documentation does not support). Runs even on
    # an empty recall pool, since a validated proposal can rescue a missed concept.
    if llm is not None and fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING,
                                         FactKind.DIAGNOSIS):
        return _propose_then_verify(fact, source, pool, llm, corroborate)

    if not pool:
        return ResolvedLine(fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
                            rationale="no candidate retrieved for the concept")

    return _decide(fact, pool, source=source)


def _strip_laterality(text: str) -> str:
    """A descriptor with the laterality word removed — so laterality variants of the
    same concept compare equal ('… right foot' ~ '… unspecified foot')."""
    return re.sub(r"\s+", " ",
                  re.sub(r"\b(right|left|bilateral|unspecified)\b", " ",
                         str(text).lower())).strip(" ,;")


def upgrade_diagnosis_laterality(line: ResolvedLine, source: CodeSource) -> ResolvedLine:
    """ICD-10-CM specificity: when a diagnosis resolved to an UNSPECIFIED-laterality
    code but the note documents a side AND a laterality-specific SIBLING exists in
    the authoritative data, upgrade to it. Agnostic — it uses descriptor grammar
    ('unspecified' vs 'right'/'left') and the code's OWN sibling family (validated by
    descriptor: a sibling must be the same concept with only laterality changed),
    never a hardcoded code or family."""
    fact = line.fact
    if not (line.resolved and fact.kind is FactKind.DIAGNOSIS and line.chosen):
        return line
    lat = str(fact.attributes.get("laterality", "")).lower().strip()
    if lat not in ("right", "left"):
        return line
    desc = line.chosen.descriptor.lower()
    if lat in desc:                      # already specific to the documented side
        return line
    if "unspecified" not in desc:        # not an unspecified-laterality code — leave it
        return line
    from .terminology import _dot
    undot = line.chosen.code.replace(".", "").upper()
    stem = undot[:-1]                    # the presumed laterality position in the family
    if not stem:
        return line
    family = _strip_laterality(desc)
    target = None
    for sib in source.leaf_codes(stem, "icd10"):
        su = sib.replace(".", "").upper()
        if su == undot:
            continue
        sdesc = (source.descriptions(su, "icd10") or [""])[0]
        # a genuine laterality sibling: names the documented side AND is otherwise
        # the identical concept (self-validates the structural guess above).
        if sdesc and lat in sdesc.lower() and _strip_laterality(sdesc) == family:
            if target is not None:
                return line              # ambiguous family — keep the original
            target = (su, sdesc)
    if target is None:
        return line
    code, sdesc = target
    line.chosen = CandidateCode(code=_dot(code), system="icd10", descriptor=sdesc,
                                score=1.0, source="laterality-specificity",
                                authority={"source": "ICD-10-CM laterality specificity"})
    line.rationale = f"{line.rationale}; upgraded to documented laterality ({lat})"
    return line


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
    a category becomes its more-specific billable children — so the
    structured decision can pick the specific code by documented laterality."""
    return [_candidate_from_code(c, source) for c in source.leaf_codes(code, "icd10")]


VERIFY_K = 8           # shortlist size sent to the entailment-selection call
MAX_RESELECT = 2       # re-selection attempts after a WRONG-CODE (not documentation-gap) rejection


def _ranked(fact: ClinicalFact, pool: list[CandidateCode],
            source: CodeSource | None) -> list[_Match]:
    """Survivors (candidates that contradict no documented attribute) ranked by
    recall, then (specificity, support) — the latter two only separate candidates
    of comparable recall. Shared by the deterministic path and propose-then-verify."""
    survivors = [m for m in (_evaluate(fact, c, source) for c in pool) if m is not None]
    survivors.sort(key=lambda m: (m.recall, m.specificity, m.support), reverse=True)
    return survivors


def _propose_then_verify(fact: ClinicalFact, source: CodeSource,
                         pool: list[CandidateCode], llm, corroborate=None) -> ResolvedLine:
    """Recall as candidate GENERATOR, authoritative descriptor + entailment as TRUTH.
    Widen the pool with validated LLM proposals, select the candidate whose OFFICIAL
    descriptor the documentation entails, then (when a corroborator is supplied)
    require an INDEPENDENT second model to agree before accepting. Escalate if the
    selection finds nothing OR the second model disagrees. Nothing bills on recall
    alone, and nothing bills on a single model's say-so."""
    from . import verify as _verify
    proposals = [c for c in _verify.propose_codes(fact, source, llm)
                 if _evaluate(fact, c, source) is not None]
    order: list[CandidateCode] = []
    seen: set[str] = set()
    for c in proposals + [m.candidate for m in _ranked(fact, pool, source)]:
        if c.code not in seen:
            seen.add(c.code)
            order.append(c)
    shortlist = order[:VERIFY_K]
    tried: set[str] = set()
    last_reason = ""
    for _ in range(1 + MAX_RESELECT):
        cands = [c for c in shortlist if c.code not in tried]
        if not cands:
            break
        chosen, why = _verify.select_entailed(fact, cands, source, llm)
        if chosen is None:
            return ResolvedLine(
                fact=fact, chosen=None, alternatives=shortlist,
                method=ResolutionMethod.ABSTAINED,
                rationale="no candidate's authoritative descriptor is fully entailed by "
                          "the documentation (verified) — escalate")
        if corroborate is None:                      # no second model configured
            return _verified_line(fact, chosen, shortlist, why)
        ok, why2, missing = _verify.corroborate(fact, chosen, source, corroborate)
        if ok:
            why = f"{why}; independently confirmed" if why else "independently confirmed"
            return _verified_line(fact, chosen, shortlist, why)
        last_reason = why2
        if missing:
            # The code is the right KIND of service but its descriptor requires an
            # element the note does not state. Re-selecting a code that omits the
            # element would UNDER-code, so escalate as a provider query instead.
            return ResolvedLine(
                fact=fact, chosen=None, alternatives=[chosen] + shortlist[:4],
                method=ResolutionMethod.ABSTAINED,
                documentation_gap=why2,
                rationale=f"PROVIDER QUERY — the best-matching code ({chosen.code}) "
                          f"requires an element the documentation does not state "
                          f"({why2}); confirm it was performed / amend the note, "
                          f"else a less-specific code applies")
        tried.add(chosen.code)                       # wrong code -> try another candidate
    return ResolvedLine(
        fact=fact, chosen=None, alternatives=shortlist,
        method=ResolutionMethod.ABSTAINED,
        rationale=f"no candidate confirmed by independent second-model verification "
                  f"after re-selection ({last_reason}) — escalate")


def _verified_line(fact: ClinicalFact, chosen: CandidateCode,
                   shortlist: list[CandidateCode], why: str) -> ResolvedLine:
    return ResolvedLine(
        fact=fact, chosen=chosen,
        alternatives=[c for c in shortlist if c.code != chosen.code][:4],
        method=ResolutionMethod.VERIFIED,
        rationale=f"authoritative descriptor entailed by documentation: {why}"
                  if why else "authoritative descriptor entailed by documentation")


def _decide(fact: ClinicalFact, pool: list[CandidateCode],
            authority: str | None = None,
            source: CodeSource | None = None) -> ResolvedLine:
    """Structured decision over a candidate pool: eliminate contradictions,
    rank by relevance (recall) then specificity, pick deterministically when the
    leader is clear, else hand the shortlist to arbitration."""
    survivors = _ranked(fact, pool, source)
    if not survivors:
        return ResolvedLine(
            fact=fact, chosen=None, alternatives=pool[:5],
            method=ResolutionMethod.ABSTAINED,
            rationale="every candidate contradicts a documented attribute")

    top = survivors[0]
    if top.recall < _RELEVANCE_FLOOR:
        deterministic = False
    elif len(survivors) == 1:
        deterministic = True
    else:
        nxt = survivors[1]
        close = abs(top.recall - nxt.recall) < _SCORE_MARGIN
        deterministic = (top.recall - nxt.recall >= _SCORE_MARGIN
                         or (close and (top.specificity, top.support)
                             > (nxt.specificity, nxt.support)))

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
