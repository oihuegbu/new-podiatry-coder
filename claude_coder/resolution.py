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
from .models import CandidateCode, ClinicalFact, FactKind, Outcome, ResolutionMethod, ResolvedLine
from .ontology import (DescriptorFeatures, measurement_of, parse_descriptor,
                       support_score)

_LATERALITY = {"left", "right", "bilateral"}
_SCORE_MARGIN = 0.05       # recall lead that settles a pick among relevant candidates
_RELEVANCE_FLOOR = 0.6     # policy dial: min recall similarity for a deterministic pick
_RECALL_POOL = 40


# Generic coding-grammar qualifiers (like left/right/unspecified) that denote a
# DISTINCT billable variant — a descriptor carrying one must have it supported by the
# documentation, so a hit whose descriptor asserts one is verified rather than closed
# deterministically. Not medical codes or conditions; generic status vocabulary.
_STATUS_QUALIFIERS = ("secondary", "revision", "delayed", "reconstruction", "sequela")

# A residual/catch-all diagnosis descriptor ("other specified" / "not elsewhere
# classified" / "unspecified" bucket) is entailed by almost any case in its broad
# category, so entailment alone is not grounding -- it must ALSO share a DISTINCTIVE
# clinical term with the documentation. These are the residual markers and the generic
# ICD-grammar words excluded from the distinctive-token overlap. No conditions/codes.
_RESIDUAL_MARKERS = ("other specified", "not elsewhere classified", "unspecified",
                     "other disorders of", "other and unspecified")
_GENERIC_TOKENS = set(_LATERALITY) | {
    "other", "specified", "unspecified", "elsewhere", "classified", "disorders",
    "disorder", "site", "sites", "region", "part", "parts", "certain"}


def _distinctive_tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z]+", text.lower())
            if len(t) > 4 and t not in _GENERIC_TOKENS}


def _residual_without_grounding(fact: ClinicalFact, chosen: CandidateCode) -> bool:
    """A DIAGNOSIS resolved to a RESIDUAL/catch-all code whose descriptor shares NO
    distinctive clinical term with the documented condition -- an ungrounded guess."""
    desc = chosen.descriptor.lower()
    if not any(m in desc for m in _RESIDUAL_MARKERS):
        return False
    return not (_distinctive_tokens(fact.description) & _distinctive_tokens(desc))


def _needs_verification(fact: ClinicalFact, cand: CandidateCode) -> bool:
    """Does this authoritative hit carry a distinguishing qualifier the documentation
    may not support — so it must be entailment-CONFIRMED rather than closed
    deterministically? True when the descriptor states a bundled-component clause
    ('with'/'without …'), a status qualifier (secondary/revision/…), a measurement it
    needs but the note lacks, or a count it needs but the note lacks. A plain concept
    match with none of these closes deterministically (recovers autonomy + cost)."""
    d = cand.descriptor.lower()
    if re.search(r"\bwith(?:out)?\b", d):
        return True
    if any(q in d for q in _STATUS_QUALIFIERS):
        return True
    feats = parse_descriptor(cand.descriptor)
    if _interval_unsupported(fact, feats):
        return True
    if feats.cardinality and not (fact.attributes.get("count")
                                  or fact.attributes.get("quantity")):
        return True
    return False


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
    interval_unsupported: bool = False


def _measure_in_range(fact: ClinicalFact, feats) -> "bool | None":
    """True/False/None: is the UNIQUE dimension-compatible documented measurement within a
    descriptor's bounded interval? None when there is no bounded interval, or no unique
    compatible convertible measurement (incomparable)."""
    if not (feats.interval and feats.interval.bounded() and feats.interval.unit):
        return None
    from . import measurement as _meas
    idim, _ = _meas.unit_dimension(feats.interval.unit)
    if idim is None:
        return None
    dm = _meas.measurement_for_constraint(
        fact.attributes, idim, feats.interval.semantic_role)
    if dm is None:
        return None
    dv = _meas.convert(dm.value, dm.dimension, dm.unit, feats.interval.unit)
    if dv is None and dm.unit == feats.interval.unit:
        dv = dm.value
    if dv is None:
        return None
    return feats.interval.contains(dv)


def _interval_unsupported(fact: ClinicalFact, feats) -> bool:
    """A descriptor REQUIRES a bounded measurement interval the documentation does NOT
    support: a bounded interval is present but there is no unique dimension-compatible
    convertible in-range measurement (missing / incompatible / unitless / ambiguous /
    unconvertible). Such a code must never close deterministically nor skip verification.
    (Codex review F4-R1.)"""
    if not (feats.interval and feats.interval.bounded()):
        return False
    return _measure_in_range(fact, feats) is not True


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

    # TYPED, DIMENSION-GUARDED, role-safe measurement comparison, used for BOTH
    # elimination and specificity. A documented value is compared to a descriptor interval
    # ONLY when they share a dimension UNAMBIGUOUSLY (exactly one documented measurement of
    # that dimension) and the unit converts. An incompatible, unitless, or ambiguous
    # measurement neither ELIMINATES a candidate nor earns SPECIFICITY credit -- fail-closed
    # on the comparison. (Codex review F4: the old specificity path used a bare, unit-blind
    # number and could deterministically prefer a dimensionally-incompatible candidate.)
    _in_range = _measure_in_range(fact, feats)
    if _in_range is False:
        return None                     # documented measurement out of the code's range
    _iu = bool(feats.interval and feats.interval.bounded()) and _in_range is not True

    # SPECIFICITY — count the constraining attributes the descriptor POSITIVELY
    # accounts for; a more specific code wins ties.
    spec = 0
    if fl and fl in feats.laterality:
        spec += 1
        reasons.append(f"laterality {fl}")
    if _in_range is True:
        spec += 1
        reasons.append("documented measurement within the code's range")

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

    return _Match(cand, feats, cand.score, spec, support, reasons, interval_unsupported=_iu)


def resolve(request, source: CodeSource, top_k: int = _RECALL_POOL,
            llm=None, corroborate=None, dos: str | None = None) -> ResolvedLine:
    """Resolve an eligible retrieval request, never a raw clinical fact."""
    from .eligibility import RetrievalRequest
    if not isinstance(request, RetrievalRequest):
        raise TypeError("code retrieval requires an eligible RetrievalRequest")
    fact = request.fact
    if not fact.billable:
        return ResolvedLine(
            fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
            rationale=f"not performed today (disposition={fact.disposition.value}) — not billed")

    # An authoritative single index/descriptor hit is a strong CANDIDATE, not a
    # finished verdict. The Alphabetic Index (or a descriptor index) gives a
    # term->code lead that the CPT/ICD workflow requires confirming against the
    # Tabular + the documented facts. So: in deterministic/test mode (no verifier) a
    # clean single hit still resolves deterministically; but in real mode, for the
    # verifiable kinds, the hit is SEEDED into propose-then-verify and billed only
    # after descriptor ENTAILMENT + independent corroboration — never on the hit
    # alone (which could carry an undocumented qualifier).
    seeds: list[CandidateCode] = []
    _pv_kind = fact.kind in (FactKind.PROCEDURE, FactKind.IMAGING, FactKind.DIAGNOSIS)

    def _take(cands, authority, always_verify=False):
        cands = [c for c in cands if c]
        if not cands:
            return None
        # Narrowed: a hit whose descriptor carries a distinguishing qualifier the note
        # may not support is routed through entailment confirmation; a clean, plain
        # index/descriptor match closes deterministically (most simple encounters
        # should close after authoritative resolution, no LLM call). EXCEPTION: a
        # SNOMED CT crosswalk hit (always_verify) maps a concept to a best-fit DEFAULT
        # ICD code that can be less specific than — or wrong for — the documented
        # condition, so it is ALWAYS entailment-confirmed, never trusted deterministically.
        if (llm is not None and _pv_kind
                and (always_verify or any(_needs_verification(fact, c) for c in cands))):
            seeds.extend(cands)                  # qualified / crosswalk -> confirm downstream
            return None
        line = _decide(fact, cands, authority=authority, source=source)
        return line if line.resolved else None

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
                r = _take(pool, "ICD-10-CM Alphabetic Index")
                if r is not None:
                    return r
        # SECOND authoritative layer: the SNOMED CT -> ICD-10-CM crosswalk (the long-
        # tail eponyms/synonyms the ICD Index lacks — e.g. an eponymous condition).
        # A single crosswalk hit is a strong CANDIDATE, not a verdict: the concept's
        # default ICD map can be less specific than, or wrong for, the documented
        # condition, so it is ALWAYS entailment-confirmed (never trusted blindly).
        snomed = source.snomed_codes(fact.description, fact.system)
        if len(snomed) == 1:
            pool = _authoritative_pool(next(iter(snomed)), source)
            if pool:
                r = _take(pool, "SNOMED CT -> ICD-10-CM map", always_verify=True)
                if r is not None:
                    return r

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
                r = _take([cand], "CMS Table of Drugs & Biologicals")
                if r is not None:
                    return r

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
            r = _take([cand], "AMA CPT Alphabetic Index")
            if r is not None:
                return r

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
            # Fix5: the learned index is keyed on phrase->code WITHOUT clinical context
            # (system already partitions, but anatomy/laterality/approach/measurement do
            # not) and its freshness check fails open, so a learned hit must NOT bill
            # deterministically. It contributes a CANDIDATE that still passes verification.
            if _LEARNED_DETERMINISTIC:
                r = _take([cand], "learned verified-resolution index")
                if r is not None:
                    return r
            else:
                seeds.append(cand)

        pidx = source.procedure_index_codes(fact.description, fact.system)
        if len(pidx) == 1:
            code = next(iter(pidx))
            rec = source.lookup(code, fact.system) or {}
            desc = (rec.get("long_description") or rec.get("description")
                    or rec.get("short_description") or "")
            cand = CandidateCode(code=code, system=fact.system, descriptor=str(desc),
                                 score=1.0, source="cpt-descriptor-index",
                                 authority={"source": "CPT/HCPCS descriptor index"})
            r = _take([cand], "CPT/HCPCS descriptor index")
            if r is not None:
                return r

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

    # The authoritative index hits (if any) LEAD the shortlist as high-confidence
    # candidates — but they are billed only if propose-then-verify below confirms
    # entailment + corroboration, so a unique Index hit no longer auto-bills.
    if seeds:
        seen_codes = {c.code for c in seeds}
        pool = list(seeds) + [c for c in pool if c.code not in seen_codes]

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
        line = _propose_then_verify(fact, source, pool, llm, corroborate, dos=dos)
        # #1 grounding: a DIAGNOSIS that verified only to a residual/catch-all category
        # with no distinctive descriptor overlap is an ungrounded guess (entailment
        # against a catch-all is near-tautological) -- escalate, never bill it verified.
        if (line.resolved and fact.kind is FactKind.DIAGNOSIS
                and _residual_without_grounding(fact, line.chosen)):
            return ResolvedLine(
                fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
                alternatives=[line.chosen],
                rationale=("the documented condition mapped only to a residual/catch-all "
                    f"code ({line.chosen.code}) whose descriptor shares no distinctive "
                    "clinical term with the documentation -- a coder CLASSIFICATION/mapping "
                    "decision (identify the specific code, or confirm the residual bucket); "
                    "not a provider documentation gap, and not billed on a non-specific code"))
        return line

    if not pool:
        return ResolvedLine(fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
                            rationale="no candidate retrieved for the concept")

    return _decide(fact, pool, source=source, dos=dos)


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


def refine_diagnosis_specificity(line: ResolvedLine, source: CodeSource,
                                 llm=None, corroborate=None) -> ResolvedLine:
    """ICD-10-CM 'code to the highest documented specificity'. Entailment is
    NECESSARY BUT NOT SUFFICIENT: an 'unspecified'/NOS descriptor is entailed by
    every case in its concept, so a specific, equally-entailed sibling must win —
    and it can only win if it is actually offered for comparison. Two steps, most
    conservative first:

      1. STRUCTURAL laterality upgrade (no LLM) — an unspecified-LATERALITY code to
         the documented-side sibling in the SAME descriptor family. Handles the
         narrow case only ('… unspecified <site>' -> '… right <site>').

      2. VERIFIED specificity upgrade (when an entailment LLM is available) — for a
         BROADER unspecified/NOS code that step 1 cannot bridge because its specific
         counterpart lives in a different descriptor family (a fully-unspecified
         '<concept>, unspecified' catch-all vs a site-and-side-specific
         '<concept> of right <site>'):
         gather the code's MORE-specific, on-concept, documented-side relatives from
         its OWN authoritative category, offer {chosen + relatives} to the SAME
         entailment verifier, and adopt a strictly-more-specific relative it selects
         AND an independent model confirms. If a specific relative is selected but
         the independent check REJECTS it, the choice is genuinely ambiguous —
         escalate rather than silently bill the unspecified code when the record
         supports a specific one (fail-closed).

    Agnostic: 'unspecified' is descriptor text; relatives come from the code's own
    authoritative category leaves; the judgement is the existing entailment
    machinery. No code or family is named here."""
    line = upgrade_diagnosis_laterality(line, source)          # step 1 (cheap)
    fact = line.fact
    if not (line.resolved and fact.kind is FactKind.DIAGNOSIS and line.chosen):
        return line
    if llm is None:
        return line
    lat = str(fact.attributes.get("laterality", "")).lower().strip()
    if lat not in ("right", "left"):
        return line                                # no documented side to sharpen to
    desc = line.chosen.descriptor.lower()
    if lat in desc or "unspecified" not in desc:
        return line                                # already side-specific, or not unspecified
    from .terminology import _dot
    from . import verify as _verify
    undot = line.chosen.code.replace(".", "").upper()
    root = undot[:3]                               # the code's ICD-10-CM category
    tok = lambda s: {t for t in re.split(r"[^a-z]+", s.lower()) if len(t) > 3}
    concept = tok(_strip_laterality(desc))         # distinctive concept words of the chosen code
    relatives: list[CandidateCode] = []
    for sib in source.leaf_codes(root, "icd10"):
        su = sib.replace(".", "").upper()
        if su == undot:
            continue
        sdesc = (source.descriptions(su, "icd10") or [""])[0]
        sl = sdesc.lower()
        if not sdesc or lat not in sl or "unspecified" in _strip_laterality(sl):
            continue                               # not strictly more specific / wrong side
        if concept and not (concept & tok(_strip_laterality(sl))):
            continue                               # off-concept sibling in the same category
        cand = CandidateCode(code=_dot(su), system="icd10", descriptor=sdesc, score=1.0,
                             source="specificity-relative",
                             authority={"source": "ICD-10-CM specificity"})
        if _evaluate(fact, cand, source) is not None:   # contradicts no documented attribute
            relatives.append(cand)
    if not relatives:
        return line                                # no more-specific code exists -> keep it
    # rank by concept overlap, keep the shortlist bounded and cheap
    relatives.sort(key=lambda c: len(concept & tok(_strip_laterality(c.descriptor))),
                   reverse=True)
    shortlist = [line.chosen] + relatives[:6]
    picked, why = _verify.select_entailed(fact, shortlist, source, llm)
    if picked is None or picked.code == line.chosen.code:
        return line                                # verifier keeps the unspecified code -> respect it
    if corroborate is not None:
        ok, why2, _missing = _verify.corroborate(fact, picked, source, corroborate)
        if not ok:
            prior = line.chosen
            line.chosen = None
            line.method = ResolutionMethod.ABSTAINED
            line.alternatives = [prior] + relatives[:4]
            line.documentation_gap = (
                f"the record documents a {lat} side but resolution is split between the "
                f"unspecified '{prior.descriptor}' and a more-specific code")
            line.rationale = (
                f"specificity ambiguous — documentation supports a more specific {lat} "
                f"code than the unspecified '{prior.code}', but independent verification "
                f"did not confirm the specific candidate — escalate")
            return line
    line.chosen = picked
    line.method = ResolutionMethod.VERIFIED
    line.rationale = (f"{line.rationale}; upgraded to the most specific entailed code "
                      f"({why})" if why else f"{line.rationale}; upgraded to most specific entailed code")
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
_MIN_RETRIEVED_SLOTS = 4   # Fix4: authoritative-retrieval slots reserved in the shortlist
                           # so LLM memory proposals can never fully displace recall
_LEARNED_DETERMINISTIC = False  # Fix5: learned index is recall-only until re-keyed with
                                # full clinical context (system/anatomy/laterality/...)


def _active_only(cands: list[CandidateCode], source: CodeSource,
                 dos: str | None) -> list[CandidateCode]:
    """Fix3: drop candidates DEFINITIVELY inactive on the DOS before they can occupy
    scarce shortlist slots. Conservative: only an explicit BLOCKED (missing/terminated
    on DOS) is removed; UNKNOWN/PASS are kept (the code_active_on_dos gate is the
    backstop). No-op when the DOS is unknown."""
    if not dos:
        return cands
    keep = []
    for c in cands:
        try:
            if source.active_on(c.code, c.system, dos) is Outcome.BLOCKED:
                continue
        except Exception:
            pass
        keep.append(c)
    return keep
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
                         pool: list[CandidateCode], llm, corroborate=None,
                         dos: str | None = None) -> ResolvedLine:
    """Recall as candidate GENERATOR, authoritative descriptor + entailment as TRUTH.
    Widen the pool with validated LLM proposals, select the candidate whose OFFICIAL
    descriptor the documentation entails, then (when a corroborator is supplied)
    require an INDEPENDENT second model to agree before accepting. Escalate if the
    selection finds nothing OR the second model disagrees. Nothing bills on recall
    alone, and nothing bills on a single model's say-so."""
    from . import verify as _verify
    proposed_matches = [_evaluate(fact, c, source)
                        for c in _verify.propose_codes(fact, source, llm)]
    retrieved_matches = _ranked(fact, pool, source)
    unsupported = [m.candidate for m in [*proposed_matches, *retrieved_matches]
                   if m is not None and m.interval_unsupported]
    proposals = [m.candidate for m in proposed_matches
                 if m is not None and not m.interval_unsupported]
    retrieved = [m.candidate for m in retrieved_matches if not m.interval_unsupported]
    # Fix4: reserve a floor of shortlist slots for authoritative RETRIEVAL so LLM
    # memory proposals cannot crowd it out. Keep up to (VERIFY_K - floor) proposals
    # first, then retrieved, then any leftover proposals fill remaining room.
    prop_cap = max(0, VERIFY_K - _MIN_RETRIEVED_SLOTS)
    order: list[CandidateCode] = []
    seen: set[str] = set()
    for c in proposals[:prop_cap] + retrieved + proposals[prop_cap:]:
        if c.code not in seen:
            seen.add(c.code)
            order.append(c)
    # Fix3: drop DOS-inactive candidates before capping to the shortlist.
    shortlist = _active_only(order, source, dos)[:VERIFY_K]
    if not shortlist and unsupported:
        return _bounded_interval_hold(fact, unsupported)
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
        chosen_match = _evaluate(fact, chosen, source)
        if chosen_match is None:
            return ResolvedLine(
                fact=fact, chosen=None, alternatives=shortlist,
                method=ResolutionMethod.ABSTAINED,
                rationale="verifier selected a candidate that contradicts documented axes")
        if chosen_match.interval_unsupported:
            return _bounded_interval_hold(fact, [chosen] + unsupported)
        # Codex F4-R1 re-review: the unsupported-required-constraint gate applies to the
        # VERIFIED path too -- a bounded-interval code whose measurement the documentation
        # does not support must abstain regardless of selection OR corroborator agreement.
        if _interval_unsupported(fact, parse_descriptor(chosen.descriptor)):
            return ResolvedLine(
                fact=fact, chosen=None, alternatives=shortlist,
                method=ResolutionMethod.ABSTAINED,
                documentation_gap=("the code's descriptor requires a measurement within a "
                    "specific range and the documentation provides no compatible measurement "
                    "of that dimension -- document the measurement or use a less-specific code"),
                rationale="selected code requires a bounded measurement the documentation "
                          "does not support -- not billed regardless of model agreement")
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


def _bounded_interval_hold(fact: ClinicalFact,
                           candidates: list[CandidateCode]) -> ResolvedLine:
    """A model agreement cannot override an unsupported descriptor constraint."""
    unique = list({c.code: c for c in candidates}.values())
    return ResolvedLine(
        fact=fact, chosen=None, alternatives=unique[:5],
        method=ResolutionMethod.ABSTAINED,
        documentation_gap=("the code descriptor requires a bounded measurement and the "
                           "documentation has no unique dimension-compatible, convertible "
                           "measurement supporting that interval"),
        rationale="bounded descriptor interval is unsupported; deterministic and model-"
                  "verified paths must abstain")


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
            source: CodeSource | None = None,
            dos: str | None = None) -> ResolvedLine:
    """Structured decision over a candidate pool: eliminate contradictions,
    rank by relevance (recall) then specificity, pick deterministically when the
    leader is clear, else hand the shortlist to arbitration."""
    if source is not None and dos:
        active = _active_only(pool, source, dos)   # Fix3: no DOS-inactive deterministic pick
        if pool and not active:                    # all candidates inactive on the DOS
            return ResolvedLine(
                fact=fact, chosen=None, alternatives=pool[:5],
                method=ResolutionMethod.ABSTAINED,
                rationale="every candidate is inactive/terminated on the date of service")
        pool = active
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
    if deterministic and getattr(top, "interval_unsupported", False):
        # Codex F4-R1: the leader's descriptor requires a bounded measurement its
        # documentation does not support (dimension-compatible + convertible + in range).
        return ResolvedLine(
            fact=fact, chosen=None, alternatives=[m.candidate for m in survivors[:5]],
            method=ResolutionMethod.ABSTAINED,
            documentation_gap=("the code's descriptor requires a measurement within a "
                "specific range and the documentation provides no compatible measurement "
                "of that dimension -- document the measurement or use a less-specific code"),
            rationale="bounded-interval code not supported by a dimension-compatible "
                      "documented measurement -- not billed deterministically")
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
