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
      – laterality contradiction    → ELIMINATE  (a "left" descriptor, a right-side fact)
      – measurement out of range     → ELIMINATE  (a documented value outside the
                                                   descriptor's own interval)
      – specificity                  → SELECT     (a code that positively matches
                                                   more documented attributes wins,
                                                   per ICD-10-CM specificity rules)

A survivor is chosen deterministically ONLY when it is the sole admitted candidate
or the sole candidate that satisfies a documented axis the others do not. When
several remain, the directive's tie policy takes over: the axes that actually
DISCRIMINATE between them are re-inspected against the ORIGINAL DOCUMENT
(`tiebreak.narrow`), and if the page still does not single one out the line becomes
ONE targeted provider query — never a similarity tiebreak and never a generic coder
queue. Every decision carries a per-field rationale (the audit trail).

The MODEL-VERIFIED path (`_propose_then_verify`) runs the SAME policy, with the two
judging models doing the eliminating instead of the descriptor features: both answer
about the WHOLE shortlist, a code is released only when exactly ONE candidate is still
entailed and every other one carries a NAMED elimination, and several standing candidates
go to the same `tiebreak.narrow` re-inspection and then to ONE targeted provider query.
Two models agreeing about one candidate is not a proof that it is the only defensible one
(Codex F8-R1), so agreement alone can no longer release a code.

Recall similarity and lexical token overlap ADMIT and ORDER candidates. Neither
selects one: the directive allows fuzzy/lexical/semantic retrieval to widen a
candidate pool and forbids it from verifying a code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace as _dc_replace
from enum import Enum

from .data_access import CodeSource
from .models import CandidateCode, ClinicalFact, FactKind, Outcome, ResolutionMethod, ResolvedLine
from . import graph_consensus as _gc
from . import tiebreak as _tiebreak
from .ontology import (DescriptorFeatures, measurement_of, parse_descriptor,
                       support_score)

_LATERALITY = {"left", "right", "bilateral"}
# NOTE: a `_SCORE_MARGIN` recall lead used to settle a pick among comparable
# candidates. It is deliberately gone -- a retrieval-score margin is similarity, and
# similarity may widen the candidate pool but never close a tie (directive section 4).
_RELEVANCE_FLOOR = 0.6     # policy dial: min recall similarity to ADMIT a candidate
_RECALL_POOL = 40

# issue #6 F9-R12-E, third re-review (Codex): the fact kinds eligible for
# entailment confirmation (propose-then-verify with an LLM, or -- with none
# -- deferred to `seeds` and excluded from the no-LLM fallback's own
# selection rather than trusted blind) -- defined ONCE and used at BOTH
# gates below (`_pv_kind` and the propose-then-verify dispatch), so they can
# never drift apart. Previously omitted SUPPLY and DRUG entirely, which cut
# both ways unsafely: a supply found through vector retrieval could never
# enter propose-then-verify even with a verifier supplied (permanently
# stuck abstaining), while a drug found through the authoritative drug
# index bypassed `_needs_verification` altogether (a qualified hit closing
# deterministic with no confirmation, no verifier gate at all). The
# verification prompt itself already names "procedure, service, supply,
# drug, or diagnosis" -- this only wires existing plumbing to the fact
# kinds it already describes.
_ENTAILMENT_KINDS = frozenset({
    FactKind.DIAGNOSIS, FactKind.PROCEDURE, FactKind.IMAGING,
    FactKind.SUPPLY, FactKind.DRUG,
})


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


def _governed_term_mapping_grounded(chosen: CandidateCode) -> bool:
    """Did a versioned term-to-code source provide an auditable semantic bridge?

    A residual descriptor intentionally does not repeat every synonym/eponym it
    classifies.  Literal descriptor overlap is therefore not the only possible
    grounding: a governed mapping may supply the missing term identity, but only
    when the candidate preserves the match method, matching source term(s), mapped
    code, and versioned source identity.  Merely labelling a candidate with a source
    name is insufficient.
    """
    authority = dict(chosen.authority or {})
    payloads = []
    if chosen.source == "snomed-crosswalk":
        payloads.append(authority)
    nested = authority.get("snomed-crosswalk")
    if isinstance(nested, dict):
        payloads.append(nested)
    for payload in payloads:
        match = dict(payload.get("term_to_code_match") or {})
        identity = dict(match.get("source_identity") or {})
        if (match.get("method")
                and match.get("normalized_query")
                and match.get("source_terms")
                and str(match.get("mapped_code") or "").replace(".", "").upper()
                    == str(chosen.code or "").replace(".", "").upper()
                and identity.get("source_id")
                and identity.get("sha256")
                and identity.get("size")):
            return True
    return False


def _residual_without_grounding(fact: ClinicalFact, chosen: CandidateCode) -> bool:
    """A DIAGNOSIS resolved to a RESIDUAL/catch-all code whose descriptor shares NO
    distinctive clinical term with the documented condition AND carries no governed,
    source-bound term-to-code match -- an ungrounded guess."""
    desc = chosen.descriptor.lower()
    if not any(m in desc for m in _RESIDUAL_MARKERS):
        return False
    if _distinctive_tokens(fact.description) & _distinctive_tokens(desc):
        return False
    return not _governed_term_mapping_grounded(chosen)


def _needs_verification(fact: ClinicalFact, cand: CandidateCode, reconciliation=None) -> bool:
    """Does this authoritative hit carry a distinguishing qualifier the documentation
    may not support — so it must be entailment-CONFIRMED rather than closed
    deterministically? True when the descriptor states a bundled-component clause
    ('with'/'without …'), a status qualifier (secondary/revision/…), a measurement it
    needs but the note lacks, or a count it needs but the note lacks. A plain concept
    match with none of these closes deterministically (recovers autonomy + cost).

    issue #6 F9-R6-R2, sixth re-review: the count/quantity check now reads through
    `graph_consensus.claim_authorized_value` -- a raw, unauthorized count merely
    BEING PRESENT must not skip forcing verification the way a genuinely documented
    one does; the fail-safe direction here is "missing forces MORE verification,"
    so treating an unauthorized value as absent only ever makes this MORE
    conservative, never less."""
    d = cand.descriptor.lower()
    if re.search(r"\bwith(?:out)?\b", d):
        return True
    if any(q in d for q in _STATUS_QUALIFIERS):
        return True
    feats = parse_descriptor(cand.descriptor)
    if _interval_unsupported(fact, feats):
        return True
    if feats.cardinality and not (
            _gc.claim_authorized_value(fact, "count", reconciliation)
            or _gc.claim_authorized_value(fact, "quantity", reconciliation)):
        return True
    return False


def _fact_laterality(fact: ClinicalFact, reconciliation=None) -> str:
    """The CLAIM-AUTHORIZED laterality value (issue #6 F9-R6-R2, sixth re-review) --
    NEVER the raw `fact.attributes["laterality"]` directly. Proven exploitable: a
    fact with a source-confirmed NEGATED laterality still eliminated/scored
    candidates deterministically via the raw value, completely bypassing every
    negation-awareness this codebase had already built into `tiebreak.narrow`/
    `graph_consensus.resolve` -- those only ever protected the genuine-TIE path;
    this is the ORDINARY elimination/specificity path every candidate goes through
    first, and it was never touched by any prior round's fix."""
    lat = str(_gc.claim_authorized_value(fact, "laterality", reconciliation) or "").lower().strip()
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
              source: CodeSource | None = None, reconciliation=None) -> _Match | None:
    """Apply the agnostic elimination rules and score specificity. Return None if
    the candidate CONTRADICTS the documented facts, else a scored match. Concept
    relevance is not judged here — retrieval already guaranteed it.

    Thin wrapper over `_evaluate_reason` (issue #6 F9-R11-H-D, seventh
    re-review) -- kept so this function's five other, reason-blind callers
    are completely unaffected by that addition."""
    return _evaluate_reason(fact, cand, source, reconciliation)[0]


def _evaluate_reason(fact: ClinicalFact, cand: CandidateCode,
                     source: CodeSource | None = None, reconciliation=None
                     ) -> tuple["_Match | None", str | None]:
    """Same elimination/scoring `_evaluate` performs, but also NAMES why a
    candidate was eliminated (issue #6 F9-R11-H-D, seventh re-review):
    `_evaluate`'s five other call sites only ever checked `is not None`, so
    a deterministically-eliminated candidate (laterality contradiction, or a
    documented measurement outside the descriptor's bounded interval) simply
    vanished -- for an authoritative-validated model PROPOSAL specifically,
    that meant it disappeared before `_propose_then_verify`'s own candidate
    universe was even built, with no record or reason surviving into the
    audit trail at all. `_evaluate` itself stays untouched in contract; only
    `_propose_then_verify` calls this richer variant."""
    feats = parse_descriptor(cand.descriptor)
    reasons: list[str] = []

    # ELIMINATION — laterality contradiction.
    fl = _fact_laterality(fact, reconciliation)
    if fl and fl != "bilateral" and feats.laterality and fl not in feats.laterality:
        return None, (f"candidate's descriptor states laterality "
                      f"{sorted(feats.laterality)}, contradicting the fact's "
                      f"documented laterality {fl!r}")

    # TYPED, DIMENSION-GUARDED, role-safe measurement comparison, used for BOTH
    # elimination and specificity. A documented value is compared to a descriptor interval
    # ONLY when they share a dimension UNAMBIGUOUSLY (exactly one documented measurement of
    # that dimension) and the unit converts. An incompatible, unitless, or ambiguous
    # measurement neither ELIMINATES a candidate nor earns SPECIFICITY credit -- fail-closed
    # on the comparison. (Codex review F4: the old specificity path used a bare, unit-blind
    # number and could deterministically prefer a dimensionally-incompatible candidate.)
    _in_range = _measure_in_range(fact, feats)
    if _in_range is False:
        # documented measurement out of the code's range
        return None, ("documented measurement is outside the candidate descriptor's "
                      "bounded interval")
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

    return (_Match(cand, feats, cand.score, spec, support, reasons,
                   interval_unsupported=_iu), None)


def _advisory_procedure_expansions(fact, source) -> list[dict]:
    """Advisory (LLM-generated, round-trip-validated) procedure-synonym RECALL
    expansion (issue #6 item 3/F8-R2, the product owner's own narrowed
    acceptance, escalated re-review: whole-string equality never found a term
    written mid-sentence in real prose). `data_access.concept_scan("procedure",
    text)` finds every known synonym phrase that occurs as a token-bounded
    substring of `text` -- normalization and word boundaries only, no fuzzy
    matching, over exactly the table `cpt_verified_synonyms.json` already
    backs. Each matched phrase is then resolved through
    `concept_lookup("procedure", term)`, unchanged: candidate terms KEPT only
    when they independently round-trip to their own code through the same
    authoritative retrieval index every other lookup uses. This function only
    WIDENS the RECALL query set with a unique match's alternate phrasings; it
    never normalizes `fact.governed_terms` (that stays anatomy's alone,
    `coreference._CONCEPT_GOVERNED_AXES`), never asserts clinical identity,
    never excludes a candidate, and never authorizes release -- every candidate
    an expansion query surfaces still passes the exact same eligibility/
    entailment/verification path as any other. Scanned against the fact's own
    documented description and its first verbatim quotation, the same two texts
    `resolve()`'s own multi-query recall already searches -- an advisory match is
    recorded only when it is UNIQUE (an ambiguous match resolves nothing, exactly
    like every other governed/advisory lookup in this codebase)."""
    lookup = getattr(source, "concept_lookup", None)
    scan = getattr(source, "concept_scan", None)
    if not callable(lookup) or not callable(scan) or fact.system not in ("cpt", "hcpcs"):
        return []
    candidates_text = [("description", fact.description)]
    candidates_text += [("evidence", s.text) for s in fact.evidence[:1]]
    out: list[dict] = []
    seen_terms: set[str] = set()
    for field, text in candidates_text:
        text = str(text or "").strip()
        if not text:
            continue
        try:
            matched_terms = scan("procedure", text)
        except Exception:
            continue
        for term in matched_terms:
            if term in seen_terms:
                continue
            seen_terms.add(term)
            try:
                result = lookup("procedure", term)
            except Exception:
                continue
            expansions = [str(e) for e in (result.get("expansions") or []) if str(e).strip()]
            if result.get("unique") and expansions:
                out.append({"term": term, "method": str(result.get("method") or ""),
                           "match_kind": "token_boundary_scan", "matched_in": field,
                           "expansions": expansions,
                           "source_identity": dict(result.get("source_identity") or {})})
    return out


def _merge_candidate(existing: CandidateCode, incoming: CandidateCode) -> CandidateCode:
    """Two candidates for the SAME `(code, system)` from DIFFERENT recall
    sources -- keep the higher-scored one's identity/descriptor/score (raw
    score is still what orders the pool; this never changes WHICH candidate
    ranks where), but fold BOTH sources' `authority` into one record,
    namespaced by source, rather than discarding the loser's lineage entirely
    (issue #6 F9-R8-C, Codex's independent re-review of 5ab4a13). A caller
    that only ever reads `authority["matches"]`/whatever shape one specific
    source wrote still finds it, under that source's own key."""
    winner, loser = ((existing, incoming) if existing.score >= incoming.score
                    else (incoming, existing))
    if winner.source == loser.source:
        return winner           # same source, e.g. two UMLS calls for one code -- no merge needed
    sources = tuple(sorted({existing.source, incoming.source}))
    merged_authority = {"sources": sources,
                        existing.source: dict(existing.authority or {}),
                        incoming.source: dict(incoming.authority or {})}
    # issue #6 F9-R12-E: a genuinely trusted route (requires_verification=
    # False) may supply closure even when the OTHER source that also found
    # this same code was itself untrusted -- the direct route already
    # independently confirms the code; the weaker signal's own lineage
    # still survives in `merged_authority` above, it just doesn't downgrade
    # the trust an independent direct hit already earned. Verification-
    # required only when BOTH sides were.
    return CandidateCode(code=winner.code, system=winner.system,
                         descriptor=winner.descriptor, score=winner.score,
                         source=winner.source, authority=merged_authority,
                         requires_verification=(existing.requires_verification
                                                and incoming.requires_verification))


def resolve(request, source: CodeSource, top_k: int = _RECALL_POOL,
            llm=None, corroborate=None, dos: str | None = None,
            reconciliation=None,
            coverage: "_requirement.CoverageCorpus | None" = None,
            page_text: dict | None = None) -> ResolvedLine:
    """Public entry point -- resolves via `_resolve_core`, then applies the
    shared post-retrieval guards every internal path (direct authoritative
    index, descriptor index, learned index, deterministic, semantic-axis
    selection, tie-narrowed, propose-then-verify) funnels through before a
    caller ever sees a selected code (issue #6, Codex's independent
    re-review, F9-R14-A / F9-R18-A). See `_resolve_core` for the actual
    resolution logic, `_apply_attribute_evidence_gap_guard` for the first
    guard, and `_apply_attribute_axis_conflict_guard` for the second.

    `page_text` (page_number -> that page's own primary-channel text, issue
    #6, Codex's independent re-review, F9-R18-A reopened P1): the ONE bounded
    original-document region `_apply_attribute_axis_conflict_guard` may
    consult for a material, unresolved clinical-attribute conflict neither
    evaluator's own cited evidence settles. `None` (the default) simply means
    that guard's bounded-reconciliation step has nothing to check -- every
    other guarantee still applies."""
    line = _resolve_core(request, source, top_k=top_k, llm=llm, corroborate=corroborate,
                         dos=dos, reconciliation=reconciliation, coverage=coverage)
    line = _apply_attribute_evidence_gap_guard(
        line, coverage, source=source, dos=dos)
    return _apply_attribute_axis_conflict_guard(
        line, source, llm, corroborate, reconciliation, coverage, page_text, dos)


def _apply_attribute_evidence_gap_guard(line: ResolvedLine, coverage,
                                        source: CodeSource | None = None,
                                        dos: str | None = None) -> ResolvedLine:
    """Record every fact-local attribute evidence gap, but withdraw a selected
    candidate only when that candidate's own authoritative contract consumes the
    gapped axis.

    An extraction attribute is not automatically a claim input.  A note can carry
    attributes such as laterality, anatomy, approach, or a device property even
    when a particular candidate's descriptor, modifiers, and units do not depend
    on that attribute.  Treating every gap as code-changing made one unsupported
    component attribute erase otherwise valid selections across a multi-service
    encounter.  Materiality is therefore derived from the same compiled candidate
    requirements and ``ClaimInputContract`` already used by the axis-conflict
    guard below -- never from a medical-term list or a code-family heuristic.

    The typed gap is still stamped for audit even when it is immaterial to the
    selected candidate.  If no candidate has been selected there is nothing to
    withdraw, and the gap remains visible for whichever candidate is considered
    later.

    issue #6, Codex's independent re-review (F9-R15-A), two corrections to the
    first version of this guard:

    1. The typed disposition is now stamped on EVERY gapped fact's line,
       whether or not `_resolve_core` selected a candidate -- an already-
       abstained line (e.g. held on a genuine tie) used to return early and
       lose the gap entirely, so the line's OWN record never named this as one
       of its reasons. `chosen` is withdrawn only when one was actually set.
    2. `documentation_gap` (which `autonomy.decide` reads as a PROVIDER_QUERY)
       is NEVER set here anymore. `coverage.complete` proves only that every
       PAGE was read -- it does not prove the SPECIFIC AXIS is absent from
       what was read, and treating it as if it did relabels a possible
       EXTRACTION failure (evidence that does not bind to the claimed value)
       as a documentation question, exactly the category error the guard was
       built to avoid one level up. The only signal allowed to promote a gap
       into a provider question is a validated `NOT_DOCUMENTED` result from
       the EXISTING candidate requirement machinery for that SAME axis (e.g. a
       real `laterality` MUST_SUPPORT requirement `_grounded_elimination`
       already validated) -- a genuinely different, already-reviewed
       mechanism this guard does not reach into and must not approximate.
       Until that specific integration exists, an extraction-origin gap stays
       a line-local, technical hold -- never a provider or coder question on
       its own.

    The withdrawn code (when one existed) moves to `alternatives` (never
    silently dropped -- an auditor must still see what would have released);
    the typed disposition on `ResolvedLine.attribute_evidence_gap` names the
    exact fact, axes, and (per axis) the rejected value/evidence/relation
    `AttributeEvidenceGap` now carries, so this cannot be lost between here
    and the certificate/ClaimBundle.
    """
    fact = line.fact
    gaps = getattr(fact, "attribute_evidence_gaps", None) or {}
    if not gaps:
        return line
    axes = sorted(gaps)
    reason = (f"axis {axes[0]!r} has no relation-valid, value-bound evidence"
             if len(axes) == 1 else
             f"axes {axes} have no relation-valid, value-bound evidence")
    per_axis = {axis: {"reason": str(getattr(gap, "reason", "") or ""),
                       "rejected_value": str(getattr(gap, "rejected_value", "") or ""),
                       "rejected_evidence_span_ids": sorted(
                           getattr(gap, "rejected_evidence_span_ids", None) or ()),
                       "rejected_relation_id": str(
                           getattr(gap, "rejected_relation_id", "") or "")}
               for axis, gap in gaps.items()}
    material_axes = _material_evidence_gap_axes(line.chosen, gaps, source, dos)
    disposition = {"fact_id": fact.fact_id, "axes": axes,
                   "material_axes": material_axes, "reason": reason,
                   "per_axis": per_axis}
    if line.chosen is None:
        return _dc_replace(line, attribute_evidence_gap=disposition)
    if not material_axes:
        return _dc_replace(line, attribute_evidence_gap=disposition)
    withdrawn = [line.chosen] + [c for c in line.alternatives if c.code != line.chosen.code]
    material_reason = (f"axis {material_axes[0]!r} is required by the selected "
                       f"candidate but has no relation-valid, value-bound evidence"
                       if len(material_axes) == 1 else
                       f"axes {material_axes} are required by the selected candidate "
                       f"but have no relation-valid, value-bound evidence")
    return _dc_replace(
        line, chosen=None, alternatives=withdrawn[:5], method=ResolutionMethod.ABSTAINED,
        rationale=f"selected code withdrawn for {fact.fact_id}: {material_reason}",
        attribute_evidence_gap=disposition)


@dataclass(frozen=True)
class ClaimInputContract:
    """The axes downstream modifier/unit generation will actually consult
    for ONE selected candidate (issue #6, Codex's independent re-review,
    F9-R18-A reopened P1, second correction) -- derived from the SAME
    authoritative records and descriptor parsers those consumers already
    use, so the axis-conflict guard's own notion of "material" can never
    drift from what those consumers actually read. The original
    `_required_claim_axes` this replaces mismatched both real consumers:

      - `laterality`: True whenever `modifiers.ModifierEngine.assign` would
        actually consult it -- ANY non-empty bilateral indicator except "9"
        (not applicable), matching `modifiers.py`'s own governed-indicator
        check exactly. The prior version only flagged indicator "1",
        silently letting an indicator "0" (bilateral NOT allowed, so the
        SIDE itself still determines which unilateral code line applies)
        line's still-unresolved laterality bypass the guard.
      - `quantity_axes`: BOTH alias keys claim assembly actually reads
        (`claim_authorized_value(fact, "count", ...) or
        claim_authorized_value(fact, "quantity", ...)` -- `pipeline.py`'s own
        unit computation, verbatim) whenever the descriptor's own parsed
        cardinality means units are computed from either. The prior version
        only ever added "count", so a conflict recorded under "quantity"
        specifically was invisible even though it controls the SAME units.
      - `dose`: True when `source.drug_unit` OR the descriptor's own parsed
        dose denominator (`ontology.parse_dose_denominator`, the SAME
        fail-closed fallback `pipeline.py`/`gates.py` already use when the
        authoritative table has no entry) means units are computed from a
        documented dose.
    """
    laterality: bool = False
    quantity_axes: tuple[str, ...] = ()
    dose: bool = False


def claim_input_contract(chosen: CandidateCode, source: CodeSource,
                         dos: str | None) -> ClaimInputContract:
    feats = parse_descriptor(chosen.descriptor)
    indicator = None
    bilat_indicator = getattr(source, "bilat_indicator", None)
    if callable(bilat_indicator):
        try:
            indicator = bilat_indicator(chosen.code, dos)
        except Exception:
            indicator = None
    laterality = bool(indicator and str(indicator) != "9" and not feats.laterality)
    quantity_axes = ("count", "quantity") if feats.cardinality else ()
    dose = False
    drug_unit = getattr(source, "drug_unit", None)
    if callable(drug_unit):
        try:
            dose = bool(drug_unit(chosen.code))
        except Exception:
            dose = False
    if not dose:
        from .ontology import parse_dose_denominator
        dose = bool(parse_dose_denominator(chosen.descriptor))
    return ClaimInputContract(laterality=laterality, quantity_axes=quantity_axes, dose=dose)


def _material_evidence_gap_axes(chosen: CandidateCode | None, gaps: dict,
                                source: CodeSource | None,
                                dos: str | None) -> list[str]:
    """Return only gap axes that can change ``chosen``'s code/modifiers/units.

    This is intentionally candidate-local and data-driven.  The union is:
    compiled MUST_SUPPORT requirements from the authoritative descriptors, axes
    consumed by downstream claim assembly, and a rejected value literally present
    in the bound descriptor.  The last signal covers a descriptor-specific value
    even when a source has no richer compiled metadata.  Absence from all three is
    not proof the extracted attribute was correct; it means only that this selected
    line does not consume it, so it cannot justify suppressing that line.
    """
    if chosen is None or not gaps:
        return []
    from . import requirement as _requirement
    required: set[str] = set()
    if source is not None:
        requirements = _requirement.compile_requirements([chosen], source)
        required.update(req.axis for req in requirements
                        if req.candidate_code == chosen.code
                        and req.role == _requirement.RequirementRole.MUST_SUPPORT)
    contract = claim_input_contract(chosen, source, dos)
    if contract.laterality:
        required.add("laterality")
    required.update(contract.quantity_axes)
    if contract.dose:
        required.add("dose")
    for axis, gap in gaps.items():
        rejected = str(getattr(gap, "rejected_value", "") or "").strip()
        if rejected and _requirement._find_clause(chosen.descriptor, rejected) is not None:
            required.add(axis)
    return sorted(set(gaps) & required)


def _material_axis_conflicts_for(chosen: CandidateCode, conflicts: dict,
                                 requirements: tuple = (),
                                 contract: "ClaimInputContract | None" = None
                                 ) -> list[tuple[str, Any, str | None]]:
    """Every unresolved clinical-attribute conflict that is MATERIAL to
    `chosen` specifically (issue #6, Codex's independent re-review, F9-R18-A;
    corrected reopened P1). Returns `(axis, conflict, required_value)`
    triples: `required_value` is the SPECIFIC disputed value `chosen`'s own
    descriptor literally states (via a reproduced clause), or `None` when
    materiality comes only from `contract`/compiled requirements -- the
    descriptor itself is silent on which value applies, so there is no
    specific value to check compatibility against, only that SOME value
    gets authorized.

    A literal descriptor clause (`requirement._find_clause`) is a fast
    POSITIVE signal -- never the complete rule on its own, since a candidate
    can require an axis without literally spelling out the disputed value
    (e.g. a code whose own applicable bilateral indicator makes laterality
    claim-relevant even though its descriptor is silent on side). Materiality
    is therefore the union of three signals, none of them a new term list or
    specialty vocabulary:
      - a literal clause match against `chosen`'s own descriptor,
      - a compiled `DescriptorRequirement` for `chosen` with
        `role == MUST_SUPPORT` (the same typed, governed axes
        `compile_requirements` already derives -- empty for a true singleton,
        since `tiebreak.discriminating_axes` needs 2+ candidates to compare;
        meaningful once this fact reaches a real tie),
      - `contract` (`claim_input_contract`): axes downstream modifier/unit
        generation will consult for THIS specific candidate, singleton or
        not.
    """
    if not conflicts:
        return []
    from . import requirement as _requirement
    contract = contract or ClaimInputContract()
    typed_axes = {req.axis for req in requirements
                 if req.candidate_code == chosen.code
                 and req.role == _requirement.RequirementRole.MUST_SUPPORT}
    if contract.laterality:
        typed_axes.add("laterality")
    typed_axes.update(contract.quantity_axes)
    if contract.dose:
        typed_axes.add("dose")
    out: list[tuple[str, Any, str | None]] = []
    for axis, conflict in sorted(conflicts.items()):
        required_value = None
        for value in (getattr(conflict, "value_primary", "") or "",
                     getattr(conflict, "value_second", "") or ""):
            if value and _requirement._find_clause(chosen.descriptor, value) is not None:
                required_value = value
                break
        if required_value is not None or axis in typed_axes:
            out.append((axis, conflict, required_value))
    return out


def _event_page_region_text(fact: ClinicalFact, page_text: dict | None) -> str:
    """The text of exactly the original-document page(s) this fact's OWN
    evidence is anchored to (issue #6, Codex's independent re-review,
    F9-R18-A reopened P1) -- never the whole document.

    `page_text` (page_number -> that page's own primary-channel text) is
    supplied by the one caller that has a real source document
    (`pipeline.code_encounter`, mirroring how `coverage`/`reconciliation`
    already reach this module); every other caller gets `None` and this
    returns "" (no bounded region to check -- callers must treat that as
    "nothing to authorize from," never silently search something wider
    instead). Bounded to this fact's own anchored pages specifically so a
    value documented for a DIFFERENT event's page can never be mistaken for
    this event's own source support -- the whole-corpus search this replaces
    could not tell those apart (Codex's second reproduction: a term
    belonging to another event in the full corpus wrongly settling this
    event's conflict).
    """
    if not page_text:
        return ""
    pages = sorted({p for p in (getattr(s, "page", None) for s in
                               (getattr(fact, "evidence", None) or ())) if p is not None})
    return "\n\n".join(page_text[p] for p in pages if p in page_text)


def _resolve_material_axis_conflict(
        fact: ClinicalFact, chosen: CandidateCode, axis: str, conflict,
        source: CodeSource, llm, corroborate, reconciliation,
        coverage: "_requirement.CoverageCorpus | None",
        page_text: dict | None) -> tuple[str, str]:
    """Settle ONE material, unresolved clinical-attribute conflict against
    `chosen` specifically. Returns `(outcome, detail)`:

      "authorized"   -- `graph_consensus.claim_authorized_value` now
                        authorizes this axis (either it already did, or the
                        bounded autonomous-adjudication step below settled it
                        and the axis now reproduces through that SAME,
                        pre-existing, fail-closed accessor); release may stand.
      "contradicted" -- both independent evaluators, with validated evidence,
                        call `chosen`'s own descriptor identity CONTRADICTED
                        or a DIFFERENT CONCEPT; `chosen` is wrong.
      "silent"       -- both independent evaluators agree the note genuinely
                        never documents this axis (validated per
                        `_candidate_disposition_uniqueness`'s own
                        `not_documented` bar, itself gated on the COMPLETE
                        NOTE having actually been rendered to both), and a
                        bounded, page-scoped re-check of this event's own
                        source region confirms it is not there either -- a
                        real, provider-answerable gap.
      "system_error" -- the bounded, page-scoped reconciliation check itself
                        raised (never "the value was not found" -- that is
                        "silent"), OR the page-scoped check DID find one of
                        the disputed values there but `claim_authorized_value`
                        still does not reproduce it -- the document may say
                        it, but no event-local AttributeEvidence/
                        AxisAdjudication binds it; retryable, never a
                        documentation gap or a manufactured authorization.
      "unverified"   -- no verifier is configured, only one evaluator is
                        configured, the evaluators disagree, or the bounded
                        autonomous adjudication could not settle it either --
                        the conservative default: `chosen` is withdrawn, not
                        eliminated, so it stays visible as a candidate rather
                        than either billing or falsely ruling it out (issue
                        #6, Codex's independent re-review, F9-R18-A reopened
                        P1, required correction item 2).

    A candidate disposition may ELIMINATE (contradicted/different_concept) or
    defer -- it may NEVER itself manufacture clinical-axis proof (issue #6,
    Codex's independent re-review, F9-R18-A reopened P1 correction). Positive
    authorization runs EXCLUSIVELY through `claim_authorized_value`, the same
    fail-closed accessor every other claim-affecting consumer in this
    codebase already uses; a "both evaluators say entailed" or "the page
    lexically contains the word" verdict is deliberately never treated as
    substitute proof, since either was shown exploitable (a reconciled but
    UNRELATED cited span; a term stated for a DIFFERENT event on a shared
    page).
    """
    label = f"{axis!r} ({conflict.value_primary!r} vs {conflict.value_second!r})"

    def _try_authorize() -> str | None:
        return _gc.claim_authorized_value(fact, axis, reconciliation)

    if llm is None:
        return "unverified", (
            f"no verifier is configured to confirm {label} for {chosen.code}, whose own "
            f"descriptor requires it -- retained as a candidate, not billed")
    from . import requirement as _requirement
    from . import verify as _verify
    requirements = _requirement.compile_requirements([chosen], source)
    try:
        j0 = _verify.select_entailed(fact, [chosen], source, llm, requirements,
                                     force_disposition=True,
                                     reconciliation=reconciliation, coverage=coverage)
        judgements = [j0]
        if corroborate is not None:
            j1 = _verify.corroborate(fact, [chosen], source, corroborate, requirements,
                                     force_disposition=True,
                                     reconciliation=reconciliation, coverage=coverage)
            judgements.append(j1)
    except Exception as exc:
        return "system_error", (
            f"verifying {label} for {chosen.code} failed "
            f"({type(exc).__name__}: {exc})")
    if len(judgements) < 2:
        return "unverified", (
            f"only one independent evaluator is configured to confirm {label} for "
            f"{chosen.code} -- retained as a candidate, not billed")
    settled = _candidate_disposition_uniqueness(
        [chosen], chosen, judgements, reconciliation, coverage, fact=fact)
    entries = [{d.candidate_code: d for d in getattr(j, "candidate_dispositions", ())}
              for j in judgements]
    d0 = entries[0].get(chosen.code)
    d1 = entries[1].get(chosen.code)
    if settled is not None:
        remaining, eliminated, _system_unresolved = settled
        if chosen.code in eliminated:
            status = getattr(d0, "status", "")
            if status == "not_documented":
                region = _event_page_region_text(fact, page_text)
                try:
                    values = tuple(v for v in
                                   (conflict.value_primary, conflict.value_second) if v)
                    found = bool(region) and values and (
                        _tiebreak.asserted_status(values, region) == "supported")
                except Exception as exc:
                    return "system_error", (
                        f"bounded reconciliation of {label} for {chosen.code} failed "
                        f"({type(exc).__name__}: {exc})")
                if not found:
                    return "silent", (conflict.provider_question or (
                        f"the record does not settle {axis!r} for {fact.description!r}"))
                # issue #6, Codex's independent re-review (F9-R18-A reopened
                # P1 correction): the bounded region DOES lexically contain a
                # disputed value, but that is evidence of a binding gap, not
                # proof of authorization on its own -- only
                # `claim_authorized_value` may authorize.
                if _try_authorize() is not None:
                    return "authorized", ""
                return "system_error", (
                    f"source text may state {label} for {fact.fact_id}, but no "
                    f"event-local AttributeEvidence/AxisAdjudication binds it; "
                    f"retry targeted reconciliation")
            return "contradicted", eliminated[chosen.code]
    # Neither entailed-and-authorized nor validly eliminated. Last resort,
    # bounded and autonomous (issue #6, Codex's independent re-review,
    # F9-R18-A reopened P1 correction): an independent, cross-vendor verifier
    # pair may settle the axis itself from ONLY the reconciled attribute
    # spans already attached to THIS target fact -- never a full page, never
    # a preference between two readings' say-so. `second=None`: this module
    # only ever has the canonical, already-merged fact, never the original
    # second reading's own object.
    disagreement = _gc.AxisDisagreement(
        node_id=fact.fact_id, axis=axis, value_primary=conflict.value_primary,
        value_second=conflict.value_second, basis="unresolved cross-reading conflict")
    try:
        support = _gc.adjudicate_axis(disagreement, fact, None, reconciliation,
                                      llm, corroborate)
    except Exception as exc:
        return "system_error", (
            f"autonomous adjudication of {label} for {fact.fact_id} failed "
            f"({type(exc).__name__}: {exc})")
    if support is not None:
        _gc._record_axis_adjudication(fact, axis, support.value,
                                      _gc.PROOF_CROSS_VENDOR, support.span_ids)
        if _try_authorize() is not None:
            return "authorized", ""
    return "unverified", (
        f"the independent evaluators did not both confirm {label} for {chosen.code} "
        f"with source-cited evidence -- retained as a candidate, not billed")


def _apply_attribute_axis_conflict_guard(
        line: ResolvedLine, source: CodeSource, llm, corroborate, reconciliation,
        coverage: "_requirement.CoverageCorpus | None",
        page_text: dict | None, dos: str | None = None) -> ResolvedLine:
    """The ONE shared post-resolution finalizer for clinical-attribute axis
    conflicts (issue #6, Codex's independent re-review, F9-R18-A, reopened
    P1) -- applied, like `_apply_attribute_evidence_gap_guard`, AFTER every
    internal `_resolve_core` path (direct authoritative index, descriptor
    index, learned index, deterministic `_take`/`_decide`, tie-narrowed, and
    propose-then-verify) has already decided its own candidate, so none of
    them can bypass it.

    The original F9-R18-A fix checked materiality only inside
    `_propose_then_verify_core`, using `tiebreak.discriminating_axes` (which
    needs 2+ candidates) -- so a direct authoritative singleton hit, whose OWN
    descriptor still carried an absolute, unauthorized requirement on the
    conflicted axis, closed deterministically through `_decide` without ever
    being checked at all. Codex's exact-SHA reproduction: a lone "Procedure
    alpha, right side" candidate released `method=deterministic` although the
    fact's own laterality was an unresolved cross-reading conflict and
    `claim_authorized_value` returned `None` for it -- an evidence-unsupported,
    side-specific release.

    Never released for free: a raw, conflicted `fact.attributes[axis]` value
    is NOT authorizing on its own (`claim_authorized_value` already enforces
    this everywhere it is the accessor, including modifier/unit generation in
    `modifiers.py`); this guard additionally requires, for any conflict
    MATERIAL to `line.chosen`'s own descriptor
    (`_material_axis_conflicts_for`), either an existing independent
    authorization or a fresh, source-cited disposition from BOTH configured
    evaluators (`_resolve_material_axis_conflict`) before letting the
    selection stand. Applied before modifier/unit generation and ClaimBundle
    projection (both live downstream of `resolve()`'s return), so an
    unauthorized release can never reach either.
    """
    fact = line.fact
    conflicts = getattr(fact, "attribute_axis_conflicts", None) or {}
    if not conflicts or line.chosen is None:
        return line
    chosen = line.chosen
    from . import requirement as _requirement
    requirements = _requirement.compile_requirements([chosen], source)
    contract = claim_input_contract(chosen, source, dos)
    material = _material_axis_conflicts_for(chosen, conflicts, requirements, contract)
    if not material:
        return line
    withdrawn = [chosen] + [c for c in line.alternatives if c.code != chosen.code]
    outstanding: list[tuple[str, Any]] = []
    for axis, conflict, required_value in material:
        value = _gc.claim_authorized_value(fact, axis, reconciliation)
        if value is None:
            outstanding.append((axis, conflict))
            continue
        # issue #6, Codex's independent re-review (F9-R18-A reopened P1,
        # third correction): an authorized value on the SAME axis is not
        # enough on its own -- it must be the value `chosen`'s own
        # descriptor actually requires. A fact could authorize one value
        # while the selected candidate's descriptor requires a DIFFERENT
        # one; releasing on "some value was authorized" would bill the
        # wrong candidate. Only checked when the descriptor names a
        # SPECIFIC required value (a literal clause match) -- a
        # contract-only conflict (descriptor silent on which value applies)
        # has nothing specific to compare against, so any authorized value
        # is compatible.
        if required_value is not None and _gc._norm(value) != _gc._norm(required_value):
            return _dc_replace(
                line, chosen=None, alternatives=withdrawn[:5],
                method=ResolutionMethod.ABSTAINED,
                rationale=(
                    f"selected code withdrawn for {fact.fact_id}: its own descriptor "
                    f"requires {axis}={required_value!r}, but the record authorizes "
                    f"{axis}={value!r}"))
    if not outstanding:
        return line
    for axis, conflict in outstanding:
        outcome, detail = _resolve_material_axis_conflict(
            fact, chosen, axis, conflict, source, llm, corroborate, reconciliation,
            coverage, page_text)
        if outcome == "authorized":
            continue
        if outcome == "silent":
            return _dc_replace(
                line, chosen=None, alternatives=withdrawn[:5],
                method=ResolutionMethod.ABSTAINED, documentation_gap=detail,
                rationale=f"candidate needs one specific fact for {fact.fact_id}: {detail}")
        if outcome == "system_error":
            return _dc_replace(
                line, chosen=None, alternatives=withdrawn[:5],
                method=ResolutionMethod.ABSTAINED,
                rationale=f"SYSTEM ERROR, retryable -- not a documentation gap: {detail}")
        # "contradicted" or "unverified": either way `chosen` does not stand.
        return _dc_replace(
            line, chosen=None, alternatives=withdrawn[:5],
            method=ResolutionMethod.ABSTAINED,
            rationale=f"selected code withdrawn for {fact.fact_id}: {detail}")
    return line


def _resolve_core(request, source: CodeSource, top_k: int = _RECALL_POOL,
                  llm=None, corroborate=None, dos: str | None = None,
                  reconciliation=None,
                  coverage: "_requirement.CoverageCorpus | None" = None) -> ResolvedLine:
    """Resolve an eligible retrieval request, never a raw clinical fact.

    `reconciliation` is the encounter's `SourceReconciliation` (directive section 1).
    It is what the TIE POLICY re-inspects a tie's discriminating axes against, so the
    document -- not a retrieval score, not a model vote -- settles which of several
    surviving candidates is released. None means no original document accompanied the
    encounter; the tie policy then falls back to the anchored transcription, exactly as
    `graph_consensus.source_support` already does for fact axes.

    `coverage` (issue #6 F9-R6 Phase 3, `CoverageCorpus`-typed since the
    F9-R6-R4/R5 re-review): identifies the ONE independently-read document text
    a NOT_DOCUMENTED verdict may be deterministically checked against -- NEVER
    a model's self-reported claim of having read everything (the exact mistake
    this session's F8-R1 finding already closed for a different mechanism).
    `coverage.complete` requires BOTH real text AND no page the independent
    channel failed to cover -- neither alone is sufficient. The one caller
    that supplies a real `CoverageCorpus` is `pipeline.code_encounter`, built
    from `recall: ChannelReading` (the same whole-document reading
    `graph_consensus.ConsensusReport.recall_uncovered_pages` is computed
    from), never `fact.evidence` alone -- without this, "not documented" could
    only ever mean "not in the narrow excerpt the verifier happened to be
    shown," not "not documented anywhere." Defaults to `None` (never assume a
    complete, searched document) for every other caller."""
    from .eligibility import RetrievalRequest
    if not isinstance(request, RetrievalRequest):
        raise TypeError("code retrieval requires an eligible RetrievalRequest")
    fact = request.fact
    # issue #6 item 5/F8-R2: semantic eligibility reads what the whole documented
    # EVENT states -- every fact the canonical `ClaimLineIntent` this fact belongs
    # to also names (duplicate mentions of the SAME documented event), not just
    # this one isolated fact -- falls back to the fact alone when the caller
    # supplied no grouping (or the fact belongs to no multi-member intent), which
    # is exactly today's behavior.
    #
    # issue #6, Codex's independent re-review (F9-R13-B): this used to be every
    # fact `composition.service_intents` grouped with this one -- the BROAD
    # PART_OF/USES_DEVICE-connected service episode (procedure + anesthesia +
    # supply + imaging, etc.), not just duplicate mentions of ONE event. A
    # related-but-different-KIND component sharing that episode mixed its own
    # fact kind and `service_role` into this line's semantic eligibility,
    # contaminating it -- the broad composition grouping still exists and is
    # exactly right for bundling/necessity/code-relationship controls elsewhere
    # in this module; it is simply never the semantic input to ONE claim line's
    # own candidate eligibility again.
    elig_facts = list(request.intent_facts) or [fact]
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
    _pv_kind = fact.kind in _ENTAILMENT_KINDS

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
        #
        # issue #6 F9-R12-E (Codex): this gate previously ALSO required
        # `llm is not None` -- meaning a candidate needing verification, with
        # no verifier available to actually perform it, fell straight through
        # to the immediate `_decide` call below and could still close
        # DETERMINISTIC on zero confirmation (reproduced directly via this
        # module's own `test_snomed_layer_resolves_when_index_misses`: a sole
        # SNOMED hit with no LLM closed deterministic, despite this exact
        # comment saying "ALWAYS entailment-confirmed"). Trust is now tracked
        # ON THE CANDIDATE (`requires_verification`, via `dataclasses.replace`
        # since CandidateCode is frozen) rather than gated on llm presence:
        # a must-verify candidate ALWAYS defers to `seeds` -- confirmed
        # downstream via propose-then-verify when an LLM exists, or excluded
        # from the no-LLM fallback's own tie policy when one doesn't (see the
        # final `else:` branch below), never auto-approved either way.
        if _pv_kind and (always_verify or any(_needs_verification(fact, c) for c in cands)):
            seeds.extend(_dc_replace(c, requires_verification=True) for c in cands)
            return None
        trusted = [_dc_replace(c, requires_verification=False) for c in cands]
        line = _decide(fact, trusted, authority=authority, source=source,
                       reconciliation=reconciliation)
        if not line.resolved:
            return None
        # F8-R2 (P2): a deterministic authoritative-index hit skips the RECALL
        # pool's eligibility filter entirely (by design -- these are exact
        # term->code hits, not a semantic guess, so they are never excluded by
        # it) but must still carry an eligibility AUDIT record, exactly like
        # every other candidate path, so a reader of `candidate_eligibility`
        # never sees an unexplained None for a line that resolved this way.
        from . import semantic_eligibility as _semelig
        line.candidate_eligibility = _semelig.eligibility_report(
            elig_facts, trusted, source, dos, reconciliation)
        return line

    # AUTHORITATIVE FIRST: for a diagnosis, resolve through the ICD-10-CM
    # Alphabetic Index (clinician term -> code) before any embedding. This is the
    # permanent fix for the eponym / terse-descriptor gap — deterministic and
    # provenance-clean wherever the Index carries the term. The embedding is only
    # reached when the Index has no entry for the phrasing.
    if fact.kind is FactKind.DIAGNOSIS:
        idx = source.index_codes(fact.description, fact.system)
        # issue #6 F9-R12-A, REOPENED: `idx` mixes direct Index entries with
        # cross-reference (see/seeAlso) redirect aliases -- a redirect is
        # supplementary navigation, not proof the note supports that code,
        # so a single-code hit may close deterministically ONLY when that
        # exact code is ALSO reachable through a DIRECT entry, never when
        # the sole route to it is a redirect.
        idx_direct = source.index_codes_direct(fact.description, fact.system)
        # Trust the Index only for an UNAMBIGUOUS single-code, DIRECT mapping
        # (the clean authoritative wins, an unambiguous single code). A
        # multi-code result is a laterality family OR Index noise; either
        # way defer to the embedding + structured path, which disambiguates
        # by documented evidence. This makes the deterministic Index path
        # safe against parse noise.
        if len(idx) == 1 and next(iter(idx)) in idx_direct:
            pool = _authoritative_pool(next(iter(idx)), source)
            if pool:
                r = _take(pool, "ICD-10-CM Alphabetic Index")
                if r is not None:
                    return r
        elif idx:
            # issue #6 F9-R12-A: a multi-code Index hit (a cross-reference
            # redirect spanning a laterality/site family -- e.g. "paronychia"
            # covering both the toe and finger cellulitis leaves), OR a
            # single-code hit reachable ONLY through a redirect (never a
            # direct entry), is real candidate signal, not parse noise to
            # discard -- but never trusted deterministically (the whole
            # reason for the direct-single-hit gate above). Every stem's
            # billable leaves are seeded into propose-then-verify so the
            # documented facts (laterality, anatomy, descriptor entailment)
            # narrow it exactly like any other candidate -- it is a
            # candidate SOURCE only, never an independent approval.
            seen_codes = {c.code for c in seeds}
            for stem in idx:
                for c in _authoritative_pool(stem, source):
                    if c.code not in seen_codes:
                        seen_codes.add(c.code)
                        # issue #6 F9-R12-E: every candidate reaching this
                        # branch (a multi-code hit, OR a single code reached
                        # ONLY via a redirect) is verification-required --
                        # `requires_verification` defaults to True already,
                        # tracked via the shared field (not a one-off
                        # authority tag) so the SAME no-LLM-fallback
                        # exclusion protects every verification-required
                        # source uniformly, not just redirects.
                        seeds.append(c)
        # SECOND authoritative layer: the SNOMED CT -> ICD-10-CM crosswalk (the long-
        # tail eponyms/synonyms the ICD Index lacks — e.g. an eponymous condition).
        # A single crosswalk hit is a strong CANDIDATE, not a verdict: the concept's
        # default ICD map can be less specific than, or wrong for, the documented
        # condition, so it is ALWAYS entailment-confirmed (never trusted blindly).
        match_fn = getattr(source, "snomed_code_matches", None)
        if callable(match_fn):
            snomed_matches = dict(match_fn(fact.description, fact.system) or {})
        else:
            # Backward-compatible protocol fallback.  It remains ungrounded for a
            # residual descriptor because it carries no match/source identity.
            snomed_matches = {code: {} for code in
                              source.snomed_codes(fact.description, fact.system)}
        if len(snomed_matches) == 1:
            mapped_code, match = next(iter(snomed_matches.items()))
            pool = _authoritative_pool(
                mapped_code, source,
                candidate_source="snomed-crosswalk",
                authority={
                    "source": "SNOMED CT -> ICD-10-CM map",
                    "term_to_code_match": dict(match or {}),
                },
            )
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
    # GOVERNED ALTERNATE WORDING (issue #6 F7-R3-C4): the two independent readings may
    # have worded a code-changing axis differently (e.g. anatomy) and been recognized
    # as the SAME concept rather than a disagreement -- `governed_terms` is that
    # recognition's own record (`graph_consensus.compare`). This fact's own attribute
    # keeps whichever wording the PRIMARY reading used, so a code indexed only under
    # the SECOND reading's synonym would otherwise never be queried at all, even
    # though the encounter no longer holds on the axis. One extra query per confirmed
    # alternate, substituted for that one axis, gives it a real chance to be found --
    # a verified expansion must IMPROVE recall, not merely remove a hold.
    for axis, alternates in (fact.governed_terms or {}).items():
        for alt in alternates:
            attrs = dict(fact.attributes, **{axis: alt})
            alt_query = fact.description + " " + " ".join(
                str(v) for k, v in attrs.items() if str(k).lower() != "count")
            if alt_query.strip():
                queries.append(alt_query.strip())
    # ADVISORY PROCEDURE-SYNONYM RECALL (issue #6 item 3/F8-R2): widens the query
    # set only -- see `_advisory_procedure_expansions`'s own docstring for the
    # trust-tier discipline. Recorded for audit regardless of whether any
    # resulting candidate ends up billed.
    _advisory = _advisory_procedure_expansions(fact, source)
    for entry in _advisory:
        for alt in entry["expansions"]:
            if alt.strip():
                queries.append(alt.strip())
    best: dict[str, CandidateCode] = {}
    for q in queries:
        if not q.strip():
            continue
        for c in source.retrieve(q, fact.system, top_k=top_k):
            # issue #6 F9-R12-E, second re-review (Codex): vector retrieval
            # is RECALL, not verification -- `_decide`'s own axis-matching
            # only eliminates a candidate on a known laterality/measurement
            # CONTRADICTION and ranks survivors by documented-axis coverage;
            # it never independently confirms that a similarity-ranked hit
            # is the medically correct code for an otherwise-unqualified
            # descriptor. `CandidateCode`'s safe default
            # (`requires_verification=True`) is left untouched here -- a
            # retrieval-only candidate needs entailment confirmation (an
            # LLM) or an independent direct authoritative route
            # (`_take()`'s own unqualified-hit path) before it may close,
            # exactly like every other non-authoritative source.
            if c.code not in best or c.score > best[c.code].score:
                best[c.code] = c
    # UMLS RECALL SEED (issue #6 F9-R7 item 2): the SAME normalized-phrase set
    # already assembled above (structured query + verbatim evidence + governed
    # alternates + advisory synonyms), unioned by the SAME max-score-per-code
    # rule. Additive only -- a code this source proposes competes for pool
    # membership on equal footing but can never outrank a descriptor-grounded
    # hit for the same code, and the descriptor-entailment/typed-facet-
    # uniqueness/DOS-activity/CMS-validation path below is completely unaware
    # this candidate came from a different source. `umls_candidates` itself
    # degrades to [] when the term index is absent or a term is unmatched.
    #
    # issue #6 F9-R8-C, Codex's independent re-review of 5ab4a13: the
    # max-score-per-code rule above REPLACES, never merges -- when a code is
    # already in `best` under a different source, the higher-scored one wins
    # outright and the loser's lineage (UMLS's own `authority["matches"]`, in
    # the common case where a RAG/descriptor hit outscores it) is discarded
    # entirely before it ever reaches the bundle. Reproduced directly: a code
    # proposed by both retrieval (0.9) and UMLS (0.3) survived with only
    # `source="retrieval"` and its own authority -- the UMLS match was gone.
    # `_merge_candidate` keeps the winner's identity/score but folds BOTH
    # sources' authority into one namespaced record.
    for c in source.umls_candidates([q for q in queries if q.strip()],
                                    fact.system, dos):
        prior = best.get(c.code)
        if prior is None:
            best[c.code] = c
        else:
            best[c.code] = _merge_candidate(prior, c)
    pool = sorted(best.values(), key=lambda c: c.score, reverse=True)

    # ---- Semantic eligibility-before-retrieval (issue #6 items 4/5, F8-R2) -------
    # Narrows EVERY candidate path -- the broad RECALL pool AND the authoritative
    # index `seeds` alike -- to candidates whose compiled semantic record does not
    # positively conflict with what THIS fact documents. Seeds are exact term->code
    # hits, not a semantic guess, but "exact index match" and "semantically
    # compatible with this documentation" are different questions; an index hit
    # for the wrong date-of-service or the wrong semantic class is still wrong
    # (Codex F8-R2: seeds previously bypassed this filter entirely). Absence of a
    # compiled signal on either side never excludes a candidate; only an actual,
    # documented conflict does.
    from . import semantic_eligibility as _semelig
    # issue #6 item 8: recorded over the FULL candidate universe -- seeds and pool
    # together, before either is filtered -- so the audit trail shows every
    # candidate this fact's retrieval ever considered, kept or excluded, and can
    # never disagree with what was actually enforced below.
    #
    # F9-R2's anatomy-dominance check is GROUP-level (a candidate is excluded only
    # when a SIBLING in the SAME pool is positively grounded) -- calling
    # `eligible_partition` separately on `pool` and `seeds` would run that
    # comparison over two different, smaller groups than the one the audit report
    # above was computed over, and could keep a candidate in one partition that the
    # report -- correctly, seeing the whole union -- already marked excluded because
    # a grounded sibling existed in the OTHER partition. `pool`/`seeds` are therefore
    # derived directly FROM the report below, not by a second, independently-scoped
    # filter call, so enforcement can never drift from what the audit trail states.
    _all_candidates = list(seeds) + [c for c in pool if c.code not in
                                     {s.code for s in seeds}]
    _candidate_eligibility = _semelig.eligibility_report(elig_facts, _all_candidates,
                                                          source, dos, reconciliation)
    _eligible_ids = {(r["code"], r["system"]) for r in _candidate_eligibility
                     if r["eligible"]}
    pool = [c for c in pool if (c.code, c.system) in _eligible_ids]
    seeds = [c for c in seeds if (c.code, c.system) in _eligible_ids]

    # Fail-closed backstop for a genuine, OBSERVED service-role ambiguity
    # (issue #6 F9-R11-H-D, third re-review): `blocks_line` is only ever True
    # when this intent's OWN retrieved candidates actually classify into more
    # than one distinct role while the documented facts are conflicting or of
    # mixed kind -- not a hypothetical conflict with no practical consequence
    # for this pool. Such a candidate is still `eligible` above (role
    # incompatibility alone did not exclude it -- there is no single
    # authorized role to compare it against), so without this check
    # resolution would proceed normally over an ambiguous pool. This is the
    # BACKSTOP, not the fix: it stops a bad release; it does not correct the
    # upstream intent composition that produced the ambiguity (the real fix
    # needs the event-id/performer-ownership split this module's own
    # docstring already documents as deliberately not implemented here).
    if any(r.get("role_control", {}).get("blocks_line") for r in _candidate_eligibility):
        line = ResolvedLine(
            fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
            alternatives=[c for c in _all_candidates
                         if (c.code, c.system) in _eligible_ids],
            rationale=("this claim-line intent's own retrieved candidates classify into "
                      "more than one service role (operative vs. anesthesia) while the "
                      "documented facts are conflicting or of mixed kind -- a "
                      "composition/coder decision, never auto-resolved from an "
                      "ambiguous candidate pool"),
            documentation_gap="classification_data_gap:service_role_conflict")
        line.candidate_eligibility = _candidate_eligibility
        return line

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
    # F8-R2: which reasons semantic eligibility excluded candidates for, over the
    # FULL candidate universe -- used below both to phrase an empty-pool abstain
    # honestly and to route it. A measurement gap is answerable by the provider
    # ("please document the size"), exactly like the interval-constraint failures
    # `_decide`'s own entailment check already routes as a `documentation_gap`;
    # this eligibility check can now catch the identical defect EARLIER (before a
    # candidate ever reaches `_decide` OR `_propose_then_verify`), so it must
    # route it the same way, regardless of which of those two this fact's kind
    # sends it through. A semantic-class/date-of-service mismatch is not
    # something documenting more would fix -- the retrieved candidate is the
    # wrong SERVICE entirely, a coder classification decision.
    excluded_reasons = sorted({str(c.get("reason") or "")
                               for c in _candidate_eligibility
                               if not c.get("eligible") and c.get("reason")})
    measurement_gap = bool(excluded_reasons) and all(
        "measurement" in r for r in excluded_reasons)
    gap_summary = "; ".join(excluded_reasons)

    if llm is not None and fact.kind in _ENTAILMENT_KINDS:
        # issue #6 F9-R11-H-D, sixth re-review: the UNFILTERED retrieval/index
        # universe, not the already-eligibility-narrowed `pool` -- passing the
        # narrowed pool silently dropped every candidate the FIRST eligibility
        # pass had already excluded from `_propose_then_verify`'s own
        # "complete" report. `_propose_then_verify` recomputes eligibility over
        # this same set itself (cheap -- no LLM call), so nothing here is
        # trusted twice; it is simply given everything to report on.
        line = _propose_then_verify(fact, source, _all_candidates, llm, corroborate,
                                    dos=dos, reconciliation=reconciliation,
                                    coverage=coverage, elig_facts=elig_facts)
        # #1 grounding: a DIAGNOSIS that verified only to a residual/catch-all category
        # with no distinctive descriptor overlap is an ungrounded guess (entailment
        # against a catch-all is near-tautological) -- escalate, never bill it verified.
        if (line.resolved and fact.kind is FactKind.DIAGNOSIS
                and _residual_without_grounding(fact, line.chosen)):
            _prior_eligibility = line.candidate_eligibility
            line = ResolvedLine(
                fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
                alternatives=[line.chosen],
                rationale=("the documented condition mapped only to a residual/catch-all "
                    f"code ({line.chosen.code}) whose descriptor shares no distinctive "
                    "clinical term with the documentation and carries no versioned, "
                    "source-bound term-to-code match -- a coder CLASSIFICATION/mapping "
                    "decision (identify the specific code, or confirm the residual bucket); "
                    "not a provider documentation gap, and not billed on a non-specific code"))
            line.candidate_eligibility = _prior_eligibility
        # `_propose_then_verify` ran against an EMPTY pool (eligibility excluded
        # every retrieved candidate before it ever got a chance to check anything)
        # and, finding no candidate to propose against either, abstained with no
        # documentation_gap of its own -- stamp the eligibility-derived one so this
        # still routes as a provider question when the gap is a measurement one,
        # never silently falling back to a generic coder queue.
        if not pool and not line.resolved and not line.documentation_gap and measurement_gap:
            line.documentation_gap = gap_summary
    elif not pool:
        # F8-R2: distinguish "retrieval found nothing at all" from "retrieval found
        # candidates, but semantic eligibility excluded every one of them" -- the
        # honest, typed abstention Codex's acceptance criterion asks for, rather
        # than silently falling back to an unfiltered pool (removed above).
        if not _all_candidates:
            rationale = "no candidate retrieved for the concept"
        elif measurement_gap:
            rationale = (f"every retrieved candidate requires a documented "
                        f"measurement this fact does not state ({gap_summary})")
        else:
            rationale = (f"every retrieved candidate conflicted with this fact's "
                        f"own documented semantics ({gap_summary or 'see candidate_eligibility'}) "
                        f"-- a coder classification decision, never auto-selected "
                        f"from a structurally incompatible pool")
        line = ResolvedLine(fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
                            rationale=rationale,
                            documentation_gap=(gap_summary if measurement_gap else None))
        line.candidate_eligibility = _candidate_eligibility
    else:
        # `_decide` runs on the FULL pool, exactly as before F9-R12-E -- its
        # own elimination/tie-policy diagnostics (`tie_record`, the complete
        # `alternatives` list on a genuine tie) must still be produced and
        # reported even when no candidate is trustworthy; a TIE is already
        # safe (nothing bills), so there is nothing to gate there.
        line = _decide(fact, pool, source=source, dos=dos, reconciliation=reconciliation)
        # issue #6 F9-R12-E (Codex): a candidate this resolver itself marked
        # verification-required (SNOMED/redirect/learned/UMLS/embedding/
        # model-proposal seeds, and now plain RETRIEVAL too -- everything
        # except a plain, unqualified direct Index/descriptor hit `_take`
        # already confirmed needs no further check) must not close
        # deterministically just because `_decide` picked it as the sole
        # survivor or tie-breaker. With an LLM available, `_propose_then_
        # verify` above already re-verifies every candidate through
        # entailment regardless of provenance; this NO-LLM fallback is the
        # one path with no such check, so an unconfirmed SELECTION here
        # (not a tie -- `_decide` itself already declines those) is
        # downgraded to an honest abstention, keeping `_decide`'s own
        # diagnostics (tie_record, rationale) and adding the unconfirmed
        # code back as a candidate rather than silently vanishing it.
        if line.resolved and line.chosen.requires_verification:
            unconfirmed = line.chosen
            line.chosen = None
            line.method = ResolutionMethod.ABSTAINED
            line.rationale = (f"{line.rationale} -- but {unconfirmed.code} still needs "
                              f"independent entailment/verification and no LLM is "
                              f"available to perform it; retained as a candidate, not billed")
            existing = {c.code for c in (line.alternatives or [])}
            if unconfirmed.code not in existing:
                line.alternatives = list(line.alternatives or []) + [unconfirmed]
        line.candidate_eligibility = _candidate_eligibility
    # NOTE: the `if llm is not None and fact.kind in (...)` branch above sets
    # `line.candidate_eligibility` itself, INSIDE `_propose_then_verify` --
    # the COMPLETE report there also covers model-proposed candidates
    # (issue #6 F9-R11-H-D, fifth re-review), which `_candidate_eligibility`
    # here (computed before proposals exist) does not. Overwriting it here
    # unconditionally would silently narrow the audit trail back down.
    if _advisory:
        line.advisory_terminology = _advisory
    return line


def _strip_laterality(text: str) -> str:
    """A descriptor with the laterality word removed — so laterality variants of the
    same concept compare equal ('… right foot' ~ '… unspecified foot')."""
    return re.sub(r"\s+", " ",
                  re.sub(r"\b(right|left|bilateral|unspecified)\b", " ",
                         str(text).lower())).strip(" ,;")


def upgrade_diagnosis_laterality(line: ResolvedLine, source: CodeSource,
                                 reconciliation=None) -> ResolvedLine:
    """ICD-10-CM specificity: when a diagnosis resolved to an UNSPECIFIED-laterality
    code but the note documents a side AND a laterality-specific SIBLING exists in
    the authoritative data, upgrade to it. Agnostic — it uses descriptor grammar
    ('unspecified' vs 'right'/'left') and the code's OWN sibling family (validated by
    descriptor: a sibling must be the same concept with only laterality changed),
    never a hardcoded code or family.

    issue #6 F9-R6-R2, sixth re-review: previously read `fact.attributes["laterality"]`
    raw, with NO verification of any kind -- the cheapest, least-checked path in the
    whole pipeline for an upgrade that changes the billed code. Now requires
    `graph_consensus.claim_authorized_value`; an unauthorized (e.g. source-negated)
    value declines the upgrade instead of applying it."""
    fact = line.fact
    if not (line.resolved and fact.kind is FactKind.DIAGNOSIS and line.chosen):
        return line
    lat = str(_gc.claim_authorized_value(fact, "laterality", reconciliation) or "").lower().strip()
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
                                 llm=None, corroborate=None,
                                 reconciliation=None,
                                 coverage: "_requirement.CoverageCorpus | None" = None
                                 ) -> ResolvedLine:
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
    machinery. No code or family is named here.

    `reconciliation` threads the encounter's source-evidence reconciliation into
    `_uniqueness_view`'s grounded-elimination check (Codex F8-R1, exact-SHA re-review):
    without it, this sibling selection could not enforce the same original-page proof
    invariant the primary code-selection path enforces, and a model-named elimination
    here would fall back to unverified evidence text even when a reconciliation existed
    and had already rejected it."""
    line = upgrade_diagnosis_laterality(line, source, reconciliation)   # step 1 (cheap)
    fact = line.fact
    if not (line.resolved and fact.kind is FactKind.DIAGNOSIS and line.chosen):
        return line
    if llm is None:
        return line
    # issue #6 F9-R6-R2, sixth re-review: claim-authorized, not the raw attribute --
    # this gates which relatives even get OFFERED to the verifier below (step 2), so
    # an unauthorized value must never narrow candidacy toward the wrong side either.
    lat = str(_gc.claim_authorized_value(fact, "laterality", reconciliation) or "").lower().strip()
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
        if _evaluate(fact, cand, source, reconciliation) is not None:   # contradicts no documented attribute
            relatives.append(cand)
    if not relatives:
        return line                                # no more-specific code exists -> keep it
    # rank by concept overlap, keep the shortlist bounded and cheap
    relatives.sort(key=lambda c: len(concept & tok(_strip_laterality(c.descriptor))),
                   reverse=True)
    shortlist = [line.chosen] + relatives[:6]
    judgement = _verify.select_entailed(fact, shortlist, source, llm,
                                        reconciliation=reconciliation, coverage=coverage)
    picked, why = judgement.chosen, judgement.reason
    if picked is None or picked.code == line.chosen.code:
        return line                                # verifier keeps the unspecified code -> respect it
    corroboration = _verify.corroboration_origin(llm, corroborate)
    judgements = [judgement]
    if corroborate is not None:
        second = _verify.corroborate(fact, shortlist, source, corroborate,
                                     reconciliation=reconciliation, coverage=coverage)
        ok = second.entails(picked.code)
        if ok:
            judgements.append(second)
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
    # F8-R1, the adjacent instance: this function also used to adopt whichever ONE relative
    # the selector named. ICD-10-CM's specificity rule authoritatively eliminates the
    # ORIGINAL unspecified code (that is this function's entire premise), but it says
    # nothing about SEVERAL more-specific relatives being equally documented -- and picking
    # between those is exactly the choice a model may not make alone.
    offered = [c for c in shortlist if c.code != line.chosen.code]
    still_entailed, _elim = _uniqueness_view(fact, offered, picked, judgements, {},
                                             reconciliation)
    if len(still_entailed) > 1:
        prior = line.chosen
        line.chosen = None
        line.method = ResolutionMethod.ABSTAINED
        line.alternatives = still_entailed[:5]
        line.documentation_gap = (
            f"the record documents a {lat} side, and {len(still_entailed)} more-specific "
            f"codes are each entailed by it — document the distinguishing detail")
        line.rationale = (
            f"specificity ambiguous — {len(still_entailed)} more-specific codes than the "
            f"unspecified '{prior.code}' are each still entailed by the documentation "
            f"({', '.join(c.code for c in still_entailed)}); a model preferring one of "
            f"them is not evidence the others are wrong — escalate")
        return line
    line.chosen = picked
    # The SAME independence rule as the propose-then-verify path: this upgrade replaced the
    # resolved code with one a model selected, so it can only carry the grounded VERIFIED
    # method when an INDEPENDENT origin confirmed it. Otherwise the sharper code is adopted
    # (billing the unspecified one when the record supports a specific one is the error this
    # function exists to prevent) but the line is ARBITRATED, so a coder confirms it instead
    # of it auto-releasing on one vendor's say-so. (Round 5, phase 5.)
    _independent = _independently_corroborated(corroboration)
    line.method = (ResolutionMethod.VERIFIED if _independent
                   else ResolutionMethod.ARBITRATED)
    line.rationale = (f"{line.rationale}; upgraded to the most specific entailed code "
                      f"({why})" if why else f"{line.rationale}; upgraded to most specific entailed code")
    if not _independent:
        line.rationale = (f"{line.rationale}; NOT independently corroborated — "
                          f"{_origin_caveat(corroboration)} — needs a coder")
    return line


def _candidate_from_code(code: str, source: CodeSource, *,
                         candidate_source: str = "icd10-index",
                         authority: dict | None = None) -> CandidateCode:
    """Wrap an authoritative-Index code as a top-relevance candidate, descriptor
    from the authoritative record."""
    rec = source.lookup(code, "icd10") or {}
    desc = (rec.get("long_description") or rec.get("description")
            or rec.get("short_description") or "")
    return CandidateCode(code=code, system="icd10", descriptor=str(desc), score=1.0,
                         source=candidate_source,
                         authority=(dict(authority) if authority is not None else
                                    {"source": "ICD-10-CM Alphabetic Index"}))


def _authoritative_pool(code: str, source: CodeSource, *,
                        candidate_source: str = "icd10-index",
                        authority: dict | None = None) -> list[CandidateCode]:
    """Expand an authoritative code to its billable LEAVES — a leaf stays itself,
    a category becomes its more-specific billable children — so the
    structured decision can pick the specific code by documented laterality."""
    return [_candidate_from_code(c, source, candidate_source=candidate_source,
                                 authority=authority)
            for c in source.leaf_codes(code, "icd10")]


VERIFY_K = 8           # shortlist size sent to the entailment-selection call
_MIN_RETRIEVED_SLOTS = 4   # Fix4: authoritative-retrieval slots reserved in the shortlist
# issue #6 F9-R7-C, Codex's independent re-review of 92f4596: a UMLS-sourced
# candidate's fixed recall score (0.3, `data_access.py`) is not commensurate
# with a RAG cosine-similarity score, so it needs its OWN reserved shortlist
# lane rather than competing on raw score -- otherwise 8+ higher-scored RAG
# candidates crowd it out of VERIFY_K entirely before entailment verification
# ever sees it, reproduced directly. Deliberately small: UMLS still only ever
# PROPOSES a candidate for verification, never selects or releases one.
_MIN_UMLS_SLOTS = 1
                           # so LLM memory proposals can never fully displace recall
_LEARNED_DETERMINISTIC = False  # Fix5: learned index is recall-only until re-keyed with
                                # full clinical context (system/anatomy/laterality/...)


def _bind_evaluation_descriptors(candidates: list[CandidateCode],
                                 source: CodeSource) -> list[CandidateCode]:
    """issue #6, Codex's independent re-review (F9-R17-A): ONE authoritative
    descriptor per candidate, bound once and used everywhere downstream --
    requirement compilation, both verifier prompts, response parsing, tie
    comparison, audit, and the released line.

    Before this fix, `verify._shortlist_prompt` displayed `_best_descriptor
    (source, c)` (the richer long/medium/consumer record `source.
    descriptions()` returns) while `verify._candidate_dispositions` validated
    the model's verbatim `authority_clause` against `candidate.descriptor` (a
    DIFFERENT, often shorter registry/retrieval descriptor) -- two different
    strings for the same candidate, one shown, one validated against. A model
    that correctly quoted a phrase from the descriptor it was ACTUALLY shown
    had its answer silently dropped because that phrase never appears in the
    shorter string the parser checked instead. A missing disposition entry
    makes `_candidate_disposition_uniqueness` return `None` (defer entirely,
    per its own fail-closed contract for incomplete coverage) -- so this bug
    could make EVERY candidate in a shortlist look permanently undecidable,
    with nothing about it visible as a parse failure rather than a genuine
    model disagreement.

    Replaces `candidate.descriptor` with the same authoritative text
    `_best_descriptor` would select (falling back to the candidate's own
    descriptor exactly as `_best_descriptor` does when `source.descriptions`
    has nothing), and records the original retrieval descriptor plus the
    governing descriptor's own source snapshot identity in `authority` for
    lineage/audit -- never used as a second validation authority.
    """
    bound: list[CandidateCode] = []
    for cand in candidates:
        try:
            tiers = source.descriptions(cand.code, cand.system) or []
        except Exception:
            tiers = []
        authoritative = tiers[0] if tiers else cand.descriptor
        snapshot_fn = getattr(source, "record_snapshot_identity", None)
        snapshot: dict[str, Any] = {}
        if callable(snapshot_fn):
            try:
                snapshot = snapshot_fn(cand.code, cand.system) or {}
            except Exception:
                snapshot = {}
        bound.append(_dc_replace(
            cand, descriptor=authoritative,
            authority={**dict(cand.authority or {}),
                      "recall_descriptor": cand.descriptor,
                      "evaluation_descriptor_snapshot": snapshot}))
    return bound


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


def _ranked(fact: ClinicalFact, pool: list[CandidateCode],
            source: CodeSource | None, reconciliation=None) -> list[_Match]:
    """Survivors (candidates that contradict no documented attribute), ORDERED by
    recall, then (specificity, support). This is an ordering, not a decision: it fixes
    the shortlist offered to entailment verification and the `alternatives` an audit
    record shows. `_decide` selects only on a documented axis or on the original
    document, so no candidate is ever billed because it sorted first."""
    survivors = [m for m in (_evaluate(fact, c, source, reconciliation) for c in pool)
                if m is not None]
    survivors.sort(key=lambda m: (m.recall, m.specificity, m.support), reverse=True)
    return survivors


def _tie_escalation(fact: ClinicalFact, candidates: list[CandidateCode],
                    reconciliation, reason: str, tie=None,
                    record: dict | None = None,
                    requirements: tuple = ()) -> ResolvedLine:
    """Tie policy step 5 -- turn an unresolved tie into ONE targeted provider query.

    The directive forbids exactly one outcome here: routing the line to generic human
    coder review because candidates tied or two models disagreed. So the candidates are
    re-inspected against the ORIGINAL DOCUMENT for the axes that actually distinguish
    them, and whatever the page could not settle becomes a specific, answerable
    question about the record -- or, when the descriptors differ on nothing a provider
    could document, an explicit hold that says exactly that.

    This path never RELEASES a candidate. It is reached only after an entailment
    verifier or an independent corroborator declined one, or after the shortlist was
    found to hold MORE THAN ONE still-entailed candidate, and a document-side narrowing
    must not be able to overturn a safety control that already said no. Its job is to
    make the escalation SPECIFIC, not to re-open the decision.

    `tie` lets a caller that has ALREADY narrowed exactly these candidates hand the
    outcome in instead of paying for a second, identical re-inspection; `record` carries
    whatever evaluation led here (the uniqueness verdicts) into the SAME audit record, so
    "why not the other candidate?" is answerable from one place.
    """
    if tie is None:
        tie = _tiebreak.narrow(fact, candidates, reconciliation, requirements)
    # A tie the page DID settle, on a candidate the judgements did not both entail, still
    # leaves a real, answerable question -- narrow only fills `provider_question` when it
    # gives up, so it is rebuilt here rather than degrading into a bare hold with no owner.
    question = tie.provider_question
    if not question and tie.axes and not tie.source_integrity:
        # issue #6 F9-R2-B: never rebuild a question over an axis `narrow` already
        # found the record documents (`tie.documented`) -- a shared/ambiguous but
        # confirmed value is a coder's candidate-mapping question, not a provider
        # documentation gap.
        askable = tuple(a for a in tie.axes if a.axis not in tie.documented)
        question = _tiebreak.provider_query(fact, askable)
    return ResolvedLine(
        fact=fact, chosen=None, alternatives=candidates[:5],
        method=ResolutionMethod.ABSTAINED,
        documentation_gap=(question or None),
        tie_record={**(record or {}), **tie.as_record()},
        rationale=f"{reason} -- {tie.detail}")


def _requirement_grounded_status(fact: ClinicalFact, cand: CandidateCode,
                                 requirements: tuple, judgements: list,
                                 reconciliation, coverage
                                 ) -> tuple[bool, str] | None:
    """Whether `cand`'s OWN compiled MUST_SUPPORT/EXCLUSION requirements
    ground a validated elimination -- UNCONDITIONALLY, never gated behind
    whether either judging model's free-text, whole-shortlist verdict
    happened to NAME `cand` as eliminated (issue #6, Codex's independent
    re-review, F9-R19-A Finding 1: a validly, unanimously-judged
    NOT_DOCUMENTED/CONTRADICTED required fact must be able to eliminate a
    candidate on its own -- a model's holistic "entailed" call is a
    different, coarser question than its own per-requirement answer, and
    gating the finer signal behind the coarser one let a genuinely
    unsupported family member (e.g. a qualified-child family whose shared
    stem was never documented) survive as "entailed" merely because no
    model's free-text reasoning happened to single it out).

    Extracted from `_grounded_elimination`'s own by-axis requirement loop
    (unchanged logic, unchanged message text) so BOTH that function's
    existing `named`-gated pairwise path AND the new unconditional
    candidate classifier (`_classify_candidates`) ground eliminations from
    the exact same, single-sourced requirement evaluation -- never two
    independently-drifting copies of this logic.

    Returns `(True, detail)` when grounded, `None` when this candidate's
    compiled requirements do not (or cannot yet) ground an elimination --
    the caller falls through to whatever ELSE it uses to decide (a
    pairwise/word-overlap fallback in `_grounded_elimination`; a
    disposition-only verdict in `_classify_candidates`)."""
    from . import requirement as _requirement
    from . import verify as _verify
    cand_reqs = [r for r in requirements
                if r.candidate_code == cand.code
                and r.role in (_requirement.RequirementRole.MUST_SUPPORT,
                              _requirement.RequirementRole.EXCLUSION)]
    by_axis: dict[str, list] = {}
    for r in cand_reqs:
        by_axis.setdefault(r.axis, []).append(r)
    evidence_by_span_id = _verify.evidence_text_by_span_id(fact) if by_axis else {}
    for axis, axis_reqs in by_axis.items():
        # issue #6, Codex's independent re-review (F9-R19-A): an EXCLUSION-role
        # axis group grounds on the OPPOSITE judgement status from every other
        # elimination-eligible axis -- the candidate's own descriptor names a
        # condition it does NOT apply under, so genuinely DOCUMENTING that
        # condition (validated SUPPORTED) is what eliminates it, never its
        # absence. `validated_requirement`'s SUPPORTED path grounds on the
        # cited span's own reconciled content, not on `coverage.complete` (that
        # gate exists only for the NOT_DOCUMENTED path's whole-corpus search),
        # so an EXCLUSION group is checked regardless of `coverage` state.
        is_exclusion = axis_reqs[0].role is _requirement.RequirementRole.EXCLUSION
        target_status = ({_requirement.RequirementStatus.SUPPORTED} if is_exclusion
                         else {_requirement.RequirementStatus.NOT_DOCUMENTED})
        if not is_exclusion and (coverage is None or not coverage.complete):
            continue          # NOT_DOCUMENTED needs a fully-covered, real corpus to search
        grounded_reqs: list | None = []
        for req in axis_reqs:
            outcomes = [rj for j in judgements for rj in j.requirement_judgements
                       if rj.requirement_id == req.requirement_id]
            if not outcomes or len(outcomes) < len(judgements):
                grounded_reqs = None
                break            # not every evaluator answered -- whole axis standing
            if not all(_requirement.validated_requirement(
                    req, rj, evidence_by_span_id=evidence_by_span_id,
                    reconciliation=reconciliation, coverage=coverage) for rj in outcomes):
                grounded_reqs = None
                break            # an uncited, unreproduced, or content-mismatched
                                 # verdict -- whole axis standing
            if {rj.status for rj in outcomes} != target_status:
                grounded_reqs = None
                break            # disagreement, or the wrong-polarity status among
                                 # them -- this candidate is not groundedly eliminated
            grounded_reqs.append(req)
        if grounded_reqs:
            names = ", ".join(sorted(r.requirement_id for r in grounded_reqs))
            if is_exclusion:
                detail = (f"every requirement on axis {axis!r} for {cand.code} "
                          f"({names}) is validated SUPPORTED by every evaluator -- "
                          f"{cand.code}'s own descriptor names a condition it does "
                          f"not apply under, and that condition is documented")
            else:
                detail = (f"every alternative requirement on axis {axis!r} for "
                          f"{cand.code} ({names}) is validated NOT_DOCUMENTED by "
                          f"every evaluator, in a fully-covered, searched source")
            return True, detail
    return None


def _grounded_elimination(fact: ClinicalFact, loser: CandidateCode, winner: CandidateCode,
                          reconciliation, requirements: tuple = (),
                          judgements: list = (),
                          coverage: "_requirement.CoverageCorpus | None" = None
                          ) -> tuple[bool, str]:
    """issue #6 F9-R6: when `requirements` compiled a MANDATORY (`required=True`)
    requirement for `loser`'s own descriptor, elimination is decided from that first
    -- intrinsic to the loser, no comparison to `winner` needed -- rather than from
    the pairwise tiebreak/word-overlap check below.

    issue #6 F9-R6-R2/R3/R4/R5/R6, permanent design changes across two rounds of
    re-review from the original Phase 2/3 version:

    1. CONTRADICTED is retired as elimination grounds entirely (see `requirement.
       validated_requirement`'s docstring for the full reasoning) -- only a
       unanimous, validated NOT_DOCUMENTED verdict, with `coverage.complete`
       True, may ground an elimination now.
    2. Requirements are grouped BY AXIS, and elimination requires EVERY
       requirement in that axis group to be independently validated
       NOT_DOCUMENTED -- not just one. Only requirements with an elimination-
       eligible `role` (`MUST_SUPPORT`/`EXCLUSION`, never `POSITIVE_ALIAS`)
       enter the loop at all -- ICD-10-CM inclusion terms are non-exhaustive
       EXAMPLES per the coding guidelines, not a checklist, and structurally
       cannot ground an elimination anymore regardless of how many are
       unanimously reported absent. Laterality (and every other MUST_SUPPORT
       axis) only ever compiles exactly one requirement per candidate, so
       grouping is a no-op for it.

    Any requirement in a group failing its own checks (incomplete judging, an
    unvalidated citation, non-unanimous or non-NOT_DOCUMENTED statuses)
    disqualifies the WHOLE axis group for that loser -- fail-closed, and
    deliberately conservative: a candidate with several must-support
    requirements is harder to eliminate than a single-requirement candidate,
    not an accident.

    Anything short of full-group validated NOT_DOCUMENTED agreement (no
    elimination-eligible requirement for this loser, not every evaluator
    answered, an unvalidated citation, disagreement among evaluators, or an
    incomplete `coverage`) falls through to the pairwise check below UNCHANGED
    -- this is a STRUCTURAL fallback, not a flag: a candidate this phase doesn't
    yet have a typed answer for is judged exactly as it always was.

    Codex F8-R1 (round-9 re-review): a model's NAMED reason for ruling out `loser` is
    not, by itself, grounds to remove it from the standing set -- both judging models
    returning SOME non-empty reason string is exactly the "richer JSON shape" of model
    agreement Codex's reproduction showed converting a false elimination into a release.

    Preferentially grounded against the SAME document-proof the tie policy already uses to
    settle a genuine tie (`tiebreak.narrow`, built on `graph_consensus.source_support`) --
    never the model's own prose, and never a second, parallel definition of proof. When
    that proof settles the two candidates' discriminating axis (in either direction), its
    answer is final: a document that confirms the winner's term grounds the elimination; a
    document that is checked and states the LOSER's term too, or settles nothing, refuses
    it.

    `narrow` gives up for a THIRD reason that is not a checked-and-inconclusive document: no
    reconciled page reading could be checked at all. That reason has TWO different causes:

      * genuinely no verification channel exists for this call -- `reconciliation is None`.
        Most of `resolve`'s callers judge from evidence text with no page-anchoring
        infrastructure behind them at all, so treating THIS case as "unverifiable, therefore
        never grounded" would block the ordinary near-synonym rejection this module has
        always done. Falling back to the RAW evidence text the judging models themselves were
        shown is never WEAKER proof than what already grounded their verdict here, because
        nothing stronger was ever obtainable for this fact.
      * a reconciliation WAS supplied, but it did not confirm this fact's quotations --
        disagreed, unverifiable, or (Codex's exact-SHA re-review, second pass) the quotation
        could not even be LOCATED at all (unanchored, no span id, so it was never anchored
        into a reading `reconciliation` could speak to in the first place). Both are the
        reconciliation channel failing to stand behind the evidence -- an unlocatable
        quotation is not a weaker case than a located-and-disagreed one, it is the SAME
        failure with one less fact established. That text is exactly what the STRONGER proof
        mechanism was supplied to check and could not stand behind -- falling back to it
        anyway launders rejected (or unverifiable) evidence into a confirmed elimination,
        the same unsafe direction as the original defect, one layer deeper each time. Neither
        sub-case may fall back.

    The two are told apart ONLY by whether a reconciliation object exists for this call at
    all -- never by whether this particular fact's evidence happened to anchor successfully,
    which is exactly the distinction the first fix drew and Codex's re-review found unsafe:
    an anchored-but-disagreed span and an unanchored, un-locatable span are both "the
    reconciliation channel that was supplied could not confirm this," and must both refuse.
    """
    grounded = _requirement_grounded_status(fact, loser, requirements, judgements,
                                            reconciliation, coverage)
    if grounded is not None:
        return grounded
    # issue #6 F9-R6 Phase 4 NOTE: `requirements` is deliberately NOT threaded into
    # this fallback narrow call. This call is the escape hatch for a loser the
    # requirement mechanism above did not (or could not) ground -- letting it see
    # the SAME compiled requirements would let the untyped, evaluator-independent
    # pairwise/word-overlap fallback ground exactly the elimination the strict,
    # unanimous-AND-validated requirement gate above just declined to ground,
    # collapsing the two mechanisms' different bars into one. Genuine tie-narrowing
    # against these axes still happens -- in `_settle_uniqueness`'s own `narrow`
    # call, reached when no elimination could be established for EITHER candidate
    # at all, which is Phase 4's actual scope.
    tie = _tiebreak.narrow(fact, [winner, loser], reconciliation)
    if tie.winner is not None and tie.winner.code == winner.code:
        return True, tie.detail
    if not tie.source_integrity:
        return False, tie.detail
    if reconciliation is not None:
        # A reconciliation channel was supplied for this call but could not confirm this
        # fact's quotations -- disagreed, unverifiable, or never locatable at all -- so
        # refuse rather than fall back to the same evidence the channel declined to
        # stand behind. Never conditioned on whether THIS fact happened to anchor.
        return False, tie.detail
    text = " ".join(str(getattr(s, "text", "") or "")
                    for s in (getattr(fact, "evidence", None) or []))
    # issue #6 F9-R6-R2, fourth re-review: AXIS_LATERALITY is excluded here.
    # Laterality is the only axis that is ever both `provable` and `selectable`
    # today, so leaving it in this loop would silently re-derive it by the same
    # fixed-token-window lexical matching `tiebreak.narrow` was just changed to
    # never use for this axis -- reintroducing the exact wrong-side-selection
    # danger that fix closed, one call site later, the moment `tie.winner` comes
    # back None. Laterality is now exclusively the typed-attribute path's
    # responsibility (`tiebreak._typed_laterality_support`, consulted inside the
    # `tie = _tiebreak.narrow(...)` call just above). This filter is a no-op for
    # every axis kind that isn't laterality and stays ready for a future
    # genuinely-new selectable axis that isn't laterality.
    loser_terms = {t for probe in tie.axes
                   if probe.provable and probe.selectable and probe.axis != _tiebreak.AXIS_LATERALITY
                   for t in probe.terms_by_code.get(loser.code, ())}
    winner_terms = {t for probe in tie.axes
                    if probe.provable and probe.selectable and probe.axis != _tiebreak.AXIS_LATERALITY
                    for t in probe.terms_by_code.get(winner.code, ())}
    # issue #6 F9-R6-R2/R6-R6 re-review: `_tiebreak.asserted_status`, not a bare
    # contiguous-phrase check -- a requirement-derived term can be a whole
    # multi-word phrase ("classic presentation"), and a NEGATED mention ("not
    # classic presentation") must never count as the document stating it, the
    # same reasoning `narrow()` itself now applies.
    loser_hits = {t for t in loser_terms if _tiebreak.asserted_status((t,), text) == "supported"}
    winner_hits = {t for t in winner_terms if _tiebreak.asserted_status((t,), text) == "supported"}
    if loser_hits:
        return False, (f"the documentation states {sorted(loser_hits)}, "
                       f"{loser.code}'s own distinguishing term")
    if winner_hits:
        return True, (f"the documentation states {sorted(winner_hits)}, "
                      f"{winner.code}'s own distinguishing term, and not {loser.code}'s")
    return False, "neither candidate's distinguishing term is stated in the documentation"


def _uniqueness_view(fact: ClinicalFact, shortlist: list[CandidateCode],
                     chosen: CandidateCode, judgements: list,
                     eliminated_earlier: dict[str, str], reconciliation=None,
                     requirements: tuple = (),
                     coverage: "_requirement.CoverageCorpus | None" = None,
                     ) -> tuple[list[CandidateCode], dict[str, str]]:
    """Which shortlisted candidates are STILL ENTAILED once every judging model has
    answered about every one of them, and the NAMED reason each of the others is out.

    Codex F8-R1: two models agreeing about ONE candidate eliminates nothing else. A
    candidate is out only when EVERY judging model NAMED a reason for ruling it out AND
    that reason is independently confirmed against the original document (or an earlier
    re-selection round already had it rejected outright by deterministic constraints, which
    needs no further grounding -- it was never a model's prose to begin with). Silence about
    a candidate, an undeclared verdict, the two models disagreeing about it, and a NAMED
    elimination the document does not independently confirm all leave it STANDING -- the
    fail-closed direction, because a standing alternative BLOCKS the release rather than
    permitting one.

    `coverage` (issue #6 F9-R6 Phases 3-4, `CoverageCorpus`-typed since the
    F9-R6-R4/R5 re-review) is passed straight through to `_grounded_elimination`,
    the only place it is actually read -- see that function for what it gates.
    """
    remaining: list[CandidateCode] = []
    eliminated: dict[str, str] = {}
    for cand in shortlist:
        if cand.code == chosen.code:
            remaining.append(cand)
            continue
        prior = eliminated_earlier.get(cand.code, "")
        if prior:
            eliminated[cand.code] = prior
            continue
        named = [j.elimination_of(cand.code) for j in judgements]
        if named and all(named):
            grounded, ground_detail = _grounded_elimination(fact, cand, chosen,
                                                            reconciliation,
                                                            requirements, judgements,
                                                            coverage)
            if grounded:
                eliminated[cand.code] = (f"{'; '.join(dict.fromkeys(named))} "
                                         f"(document-confirmed: {ground_detail})")
                continue
        remaining.append(cand)
    return remaining, eliminated


def _reconciled_span_lookup(reconciliation):
    """(settled, permitted) -- the same span-id -> `ReconciliationStatus` lookup
    and the same permitted-status set `_candidate_disposition_uniqueness` and
    `_apply_attribute_axis_conflict_guard` both need to validate a
    `CandidateDispositionEvidence`'s cited spans against. Factored out
    (issue #6, Codex's independent re-review, F9-R18-A reopened P1) so the
    axis-conflict guard can apply the EXACT SAME validation bar rather than a
    re-derived approximation of it."""
    from app.contracts.source_evidence import ReconciliationStatus
    settled = reconciliation.by_span_id() if reconciliation is not None else {}
    permitted = {ReconciliationStatus.AGREED, ReconciliationStatus.VACUOUS}
    return settled, permitted


def _disposition_identity_matches(cand: CandidateCode, d) -> bool:
    """Whether a `CandidateDispositionEvidence`'s `descriptor_sha256` still
    matches the SERVER's own current hash of this candidate's official
    descriptor (issue #6, Codex's independent re-review, F9-R18-A reopened
    P1 correction) -- replaces the earlier model-authored-clause-reproduces
    check. `verify._candidate_dispositions` already validated this at parse
    time; re-checked here against the actual shortlist candidate so a
    stale/mismatched entry can never slip through a caller that reused a
    judgement across shortlists."""
    from .verify import _descriptor_sha256
    return bool(d.descriptor_sha256) and d.descriptor_sha256 == _descriptor_sha256(cand)


def _disposition_spans_validated(d, settled, permitted) -> bool:
    """Whether a `CandidateDispositionEvidence`'s cited evidence span ids are all
    genuinely reconciled (AGREED/VACUOUS), never a bare, unvalidated citation."""
    if not d.evidence_span_ids:
        return False
    return all(sid in settled and settled[sid].status in permitted
              for sid in d.evidence_span_ids)


def _candidate_disposition_uniqueness(shortlist: list[CandidateCode],
                                      chosen: CandidateCode | None,
                                      judgements: list, reconciliation, coverage,
                                      fact: "ClinicalFact | None" = None,
                                      requirements: tuple = (),
                                      admissions: dict[str, "CandidateAdmission"] | None = None,
                                      ) -> tuple[list[CandidateCode], dict[str, str],
                                                dict[str, str]] | None:
    """The candidate-level SEMANTIC entailment record (issue #6, Codex's
    independent re-review, F9-R15-B), replacing the reverted `requirement.
    _descriptor_term_requirements` (which promoted raw descriptor TOKENS into
    literal-text MUST_SUPPORT requirements -- proven unsafe: a synonym/
    paraphrase could make a literal token correctly absent while the actual
    requirement was fully documented, grounding a false elimination).

    Tried in ADDITION to (never instead of) `_uniqueness_view`'s existing
    axis/requirement-based elimination -- this only ever narrows `shortlist`
    further, on its OWN, independent, stricter bar; it is never a parallel
    selector and never lowers what `_uniqueness_view` already required.

    Returns `None` when this shortlist cannot be settled this way at all
    (fewer than two judgements, or either judgement did not answer EVERY
    candidate) -- the caller falls back to the existing tie-narrowing/
    escalation path completely unchanged. Otherwise returns a THREE-way
    `(remaining, eliminated, system_unresolved)` (issue #6, Codex's
    independent re-review, F9-R19-A Finding 1: "not successfully eliminated
    != positively supported" -- the original two-way split silently folded
    every candidate this mechanism could not cleanly PROVE eliminated
    (evaluator disagreement, an identity mismatch, an uncited contradiction,
    an incomplete-coverage not_documented verdict) into the SAME `remaining`
    bucket as a candidate BOTH evaluators genuinely, validly called
    "entailed" -- so `_settle_uniqueness` reported a candidate the system
    could simply not verify as if it were positively supported evidence,
    exactly the invariant violation Finding 1 names). `remaining` is now
    ONLY candidates both evaluators validly call "entailed" (identity-
    matched) whose OWN compiled MUST_SUPPORT/EXCLUSION contract (viability,
    differential, exclusion requirements -- Finding 2) also does not ground
    an elimination; `eliminated` maps every candidate this mechanism could
    positively dispose of (via disposition OR contract) to why;
    `system_unresolved` maps every candidate this mechanism could neither
    confirm NOR eliminate -- a system verification gap, never treated as
    supporting evidence for a tie question.

    `fact`/`requirements` (optional, default empty): when supplied, every
    "entailed" candidate is ALSO checked against its own compiled
    MUST_SUPPORT/EXCLUSION requirement contract via
    `_requirement_grounded_status`, UNCONDITIONALLY -- never gated behind
    whether either judgement's free-text elimination happened to name this
    candidate (see that function's own docstring). Omitted (the default),
    this behaves exactly as it always has for the single-candidate,
    contract-free callers that predate F9-R19-A (e.g.
    `_resolve_material_axis_conflict`'s attribute-axis-conflict check).

    issue #6, Codex's independent re-review (F9-R16-B): `chosen` is deliberately
    NOT special-cased -- it flows through the exact same per-candidate bar as
    every other candidate below. The first version of this function skipped
    validating `chosen`'s own disposition entirely (to protect against a
    different bug: eliminating `chosen` while a different candidate survived,
    which `_settle_uniqueness`'s old count-only check would then misrelease as
    `chosen` anyway). Codex's reproduction showed that "protection" let a
    self-contradicting model answer through: a judgement's LEGACY `choice`/
    `entailed` field (which `_uniqueness_view` trusts) can pick `chosen` while
    that SAME judgement's structured `candidate_dispositions` calls `chosen`
    "contradicted" -- and skipping `chosen` here let it release anyway with
    `verified_entailment`. The real fix is structural, not a skip:
    `_settle_uniqueness` now checks `remaining[0].code == chosen.code` (never
    just `len(remaining) == 1`) before releasing, which makes it SAFE to
    validate -- and properly eliminate -- `chosen` through this same bar: if
    both evaluators structurally, validly contradict it, it is eliminated like
    any other candidate, and the caller's membership check then correctly
    falls through to the tie/hold path instead of either releasing `chosen`
    or silently swapping in whichever different candidate happens to survive.

    A candidate is validly disposed only when BOTH judgements' own
    `CandidateDispositionEvidence` for it:
      - has a `descriptor_sha256` that matches THIS candidate's own current
        official descriptor identity (issue #6, Codex's independent
        re-review, F9-R18-A reopened P1 correction -- replaces the earlier
        model-authored-clause-reproduces check: a verbatim substring is still
        an arbitrary choice the model makes, and two evaluators quoting two
        different-but-both-verbatim clauses used to compare unequal as
        strings without ever proving they disagreed about the SAME thing.
        The server-computed hash is not authored by either evaluator, so
        "both cite the same identity" is a structural fact, not a
        string-equality coincidence),
      - agrees with the OTHER judgement on the exact same status (issue #6,
        Codex's independent re-review, F9-R16-B's original concern --
        agreeing on the SAME descriptor identity now makes "equal status"
        sufficient; there is no second, separate clause to disagree about),
        and that status is:
          * "contradicted"/"different_concept": ONLY with validated,
            reconciled evidence spans on BOTH sides -- a genuine semantic
            judgement backed by real source text, never a bare claim.
          * "not_documented": ONLY when `coverage is not None and
            coverage.complete` -- a complete, independently-read whole-
            document search -- ON TOP OF both evaluators' own independent
            semantic reading AND a non-empty `missing_fact` on both sides
            (issue #6, Codex's independent re-review, F9-R16-B: an empty
            `missing_fact` cannot be turned into a precise provider question,
            so this must never eliminate on a vaguer basis than it could also
            route a question from). Exact token absence alone (what the
            reverted mechanism relied on) is never treated as semantic proof
            here; this requires the SAME evaluator judgement layer Codex's
            contract asks for, not a re-derivation of it from raw text.
    Anything short of that (a missing entry, an unreproduced clause, an
    uncited contradiction/different_concept, disagreement between the two
    evaluators on status or clause, incomplete coverage, or an empty
    `missing_fact` for not_documented) leaves the candidate standing.
    """
    if len(judgements) < 2:
        return None
    per_judgement: list[dict] = []
    for j in judgements:
        entries = {d.candidate_code: d for d in
                  getattr(j, "candidate_dispositions", ())}
        if not all(c.code in entries for c in shortlist):
            return None      # this evaluator did not answer every candidate
        per_judgement.append(entries)
    j0, j1 = per_judgement[0], per_judgement[1]

    settled, permitted = _reconciled_span_lookup(reconciliation)

    def _identity_matches(cand: CandidateCode, d) -> bool:
        return _disposition_identity_matches(cand, d)

    def _spans_validated(d, cand: CandidateCode) -> bool:
        # issue #6, Codex's independent re-review (F9-R22-A, hardened by the
        # F9-R23 clarification "Gate A"): when `fact` is supplied (every
        # real production caller), reuse the SAME shared, stricter bar
        # `verify.validate_judgement_contract`'s repair loop already
        # checked the disposition against -- target-event span MEMBERSHIP
        # (never a span reconciled elsewhere in the document but irrelevant
        # to this fact), AGREED status ONLY (never VACUOUS, since
        # punctuation/whitespace cannot substantively support a
        # disposition), AND genuine CONTENT relatedness to `cand`'s own
        # descriptor or the fact's description (never a real, anchored,
        # AGREED span that is simply about something else). Falls back to
        # the older, bare status-only check (AGREED or VACUOUS, no
        # membership or content check) only for the contract-free callers
        # that predate `fact` being threaded through at all.
        if fact is not None:
            from .verify import _agreed_citable_spans
            return bool(_agreed_citable_spans(
                d.evidence_span_ids, fact, cand, reconciliation, d.status))
        return _disposition_spans_validated(d, settled, permitted)

    remaining: list[CandidateCode] = []
    eliminated: dict[str, str] = {}
    system_unresolved: dict[str, str] = {}
    for cand in shortlist:
        admission = (admissions or {}).get(cand.code)
        recall_only = bool(
            admission is not None
            and admission.standing is CandidateStanding.UNGROUNDED
        )
        # issue #6, Codex's independent re-review (F9-R16-B): `chosen` is NO
        # LONGER special-cased here. The prior version skipped validating
        # `chosen`'s own disposition entirely, so a judgement whose LEGACY
        # `choice`/`entailed` field picked `chosen` (which `_uniqueness_view`
        # trusts) could still carry a STRUCTURED disposition of "contradicted"
        # for that same candidate -- a self-contradicting model answer that
        # released a code both evaluators' own structured judgement called
        # out. `chosen` now flows through the exact same per-candidate bar as
        # any other candidate: it survives (stays in `remaining`) when both
        # evaluators call it "entailed" (the ordinary case), and is properly
        # eliminated when both structurally, validly contradict it -- the
        # caller (`_settle_uniqueness`) now checks `remaining[0].code ==
        # chosen.code` before releasing, so `chosen` being eliminated here
        # correctly routes to the tie/hold path instead of a false release,
        # and never lets a DIFFERENT surviving candidate release in its place.
        d0, d1 = j0[cand.code], j1[cand.code]
        if not (_identity_matches(cand, d0) and _identity_matches(cand, d1)):
            if recall_only:
                eliminated[cand.code] = (
                    "recall-only candidate did not earn evidence standing: its "
                    "evaluator disposition was not bound to the current authoritative "
                    "descriptor")
                continue
            system_unresolved[cand.code] = (
                f"a disposition for {cand.code} did not reproduce this candidate's "
                f"own current official descriptor identity")
            continue
        # issue #6, Codex's independent re-review (F9-R23 root finding 2):
        # each evaluator's raw status is normalized to its OWN validated
        # billing-disposition CLASS -- supported, rejected, or unresolved --
        # before the two evaluators are compared, rather than comparing raw
        # status strings for exact equality. Before this, `contradicted` on
        # one side and `different_concept` on the other -- both a validated
        # REJECTION of the candidate, just naming a different sub-reason --
        # was read as "the evaluators disagreed" and fell to
        # `system_unresolved`, even though both evaluators, independently
        # and with source-confirmed evidence, rejected the same candidate.
        # Reproduced repeatedly on the designated note. The finer sub-reason
        # (which raw status each evaluator actually gave) is retained in the
        # audit message; only the SELECTION-relevant class is compared.
        d0_class = _validated_disposition_class(d0, cand, coverage, _spans_validated)
        d1_class = _validated_disposition_class(d1, cand, coverage, _spans_validated)
        if d0_class == "rejected" and d1_class == "rejected":
            eliminated[cand.code] = (
                f"both independent evaluators, on {cand.code}'s own official "
                f"descriptor (identity {d0.descriptor_sha256[:12]}...), independently "
                f"rejected it with source-confirmed evidence (raw verdicts: "
                f"{d0.status!r}; {d1.status!r})")
            continue
        if d0_class == "supported" and d1_class == "supported":
            # issue #6, Codex's independent re-review (F9-R19-A Finding 1/2):
            # positive disposition support is necessary but not sufficient --
            # this candidate's OWN VIABILITY/EXCLUSION contract must also not
            # ground an elimination, checked UNCONDITIONALLY (never gated
            # behind a model's free-text elimination reason). `fact`/
            # `requirements` default to `None`/`()` for callers that predate
            # this contract layer, in which case `_requirement_grounded_
            # status` always returns `None` (nothing compiled) and behavior
            # is unchanged.
            #
            # Deliberately EXCLUDES `qualified_child` (the differential, as
            # opposed to `family_viability`, the precondition): unlike
            # viability/exclusion, a differential's absence is not, by
            # itself, proof against a candidate -- if NEITHER sibling's own
            # clause is documented, eliminating both here would silently
            # convert "please specify which" into "none of these apply",
            # exactly the false-elimination shape Finding 2 exists to
            # prevent. The differential instead resolves ENTIRELY through
            # `tiebreak.narrow`'s existing positive-presence winner logic at
            # `_settle_uniqueness`'s STEP 3/4 (selects when exactly one
            # sibling's clause is documented; otherwise correctly leaves the
            # tie open for a precise provider question) -- never duplicated
            # or pre-empted here.
            _contract_requirements = tuple(
                r for r in requirements if r.axis != "qualified_child")
            grounded = (_requirement_grounded_status(
                            fact, cand, _contract_requirements, judgements,
                            reconciliation, coverage)
                       if fact is not None else None)
            if grounded is not None:
                eliminated[cand.code] = grounded[1]
            else:
                remaining.append(cand)
            continue
        if d0_class == "unresolved" and d1_class == "unresolved" and d0.status == d1.status:
            reason = (f"both evaluators judged {cand.code} {d0.status!r}, but the "
                     f"verdict lacks source-confirmed, reconciled evidence on both "
                     f"sides" if d0.status != "not_documented" else
                     f"both evaluators judged {cand.code} not_documented, but this "
                     f"could not be validated against a complete, independently-read "
                     f"whole-document search")
            if recall_only:
                eliminated[cand.code] = (
                    "recall-only candidate did not earn evidence standing: " + reason)
            else:
                system_unresolved[cand.code] = reason
            continue
        if recall_only:
            eliminated[cand.code] = (
                "recall-only candidate did not earn evidence standing: independent "
                f"evaluators did not agree it was supported ({d0.status!r} vs "
                f"{d1.status!r})")
            continue
        system_unresolved[cand.code] = (
            f"independent evaluators disagreed on {cand.code}'s disposition "
            f"({d0.status!r} vs {d1.status!r})")
    # Evaluator entailment can confirm that a broad topical descriptor is
    # compatible with the record; it cannot by itself make that recall hit the
    # documented concept's identity.  If exactly one surviving candidate has
    # governed identity standing and every rival is explicitly recall-only,
    # retain the identity-grounded candidate and audit the topical rivals.  A
    # second grounded candidate, a missing admission, or any unresolved
    # evaluator contract still fails closed as a tie/system hold.
    identity_grounded = [
        cand for cand in remaining
        if (admissions or {}).get(cand.code) is not None
        and (admissions or {})[cand.code].standing is CandidateStanding.SUPPORTED
    ]
    if len(identity_grounded) == 1:
        winner = identity_grounded[0]
        rivals = [cand for cand in remaining if cand.code != winner.code]
        if rivals and all(
                (admissions or {}).get(cand.code) is not None
                and (admissions or {})[cand.code].standing is CandidateStanding.UNGROUNDED
                for cand in rivals):
            for cand in rivals:
                eliminated[cand.code] = (
                    "recall-only candidate was semantically plausible but did not "
                    "identify the documented concept; another surviving candidate "
                    "has unique governed identity evidence and independently passed "
                    "descriptor/requirement verification")
            remaining = [winner]
    return remaining, eliminated, system_unresolved


def _validated_disposition_class(d, cand: CandidateCode, coverage, spans_validated) -> str:
    """This ONE evaluator's disposition, normalized to the SELECTION-
    relevant class it validly establishes -- "supported", "rejected", or
    "unresolved" -- independent of the raw status string (issue #6, Codex's
    independent re-review, F9-R23 root finding 2). `spans_validated` is the
    caller's own closure (already bound to `fact`/`reconciliation`), so
    this function stays free of the identity/evidence-lookup plumbing --
    the identity check itself is the caller's job: it must fire before
    either side's status is even read, so it stays a single shared check,
    not duplicated per side.

    "supported": an "entailed" verdict citing at least one target-event
    span reconciled AGREED and genuinely content-related to `cand` (F9-R23
    clarification "Gate A").
    "rejected": a "contradicted"/"different_concept" verdict citing at
    least one such span, OR a "not_documented" verdict backed by complete
    whole-document coverage and a named `missing_fact`.
    "unresolved": anything short of that, including an unrecognized status
    string."""
    if d.status == "entailed":
        return "supported" if spans_validated(d, cand) else "unresolved"
    if d.status in ("contradicted", "different_concept"):
        return "rejected" if spans_validated(d, cand) else "unresolved"
    if d.status == "not_documented":
        complete = coverage is not None and getattr(coverage, "complete", False)
        return "rejected" if complete and d.missing_fact else "unresolved"
    return "unresolved"


#: issue #6, Codex's independent re-review (F9-R13-C): these two axes are
#: `role=MUST_SUPPORT` like any other compiled requirement, but they select
#: only -- through `_select_by_semantic_axes`/`requirement.semantic_axis_
#: status`'s own concept-equivalence and typed-attribute tests -- and must
#: never reach either pre-existing generic path that would otherwise treat
#: them like ordinary literal-text axes: `resolution._grounded_elimination`
#: (groups by ROLE alone, not axis name) and `tiebreak._axes_from_
#: requirements`/`tiebreak.narrow` (derives a reportable, blocking "unsettled"
#: axis for ANY requirement whose axis isn't already governed, regardless of
#: `selectable`).
_SEMANTIC_AXES = frozenset({"semantic_action", "semantic_qualifier"})


def _select_by_semantic_axes(fact: ClinicalFact, remaining: list[CandidateCode],
                             requirements: tuple, reconciliation, source
                             ) -> CandidateCode | None:
    """One additional candidate-selection condition (issue #6, Codex's
    independent re-review, F9-R13-C): select the ONE remaining candidate for
    which EVERY one of its own compiled `semantic_action` requirements reads
    SUPPORTED via `requirement.semantic_axis_status` -- never eliminates a
    candidate, never guesses.

    Round 2 (P1-C): requires COMPLETE `semantic_action` coverage across the
    WHOLE remaining/tied set before this may select anything at all. Codex's
    exact-SHA reproduction proved the earlier "a candidate with no compiled
    requirement is simply skipped" rule unsafe: a governed, fully-supported
    candidate could win outright merely because a surviving rival's action
    phrase never resolved (no authority coverage for IT), even though missing
    coverage for that rival is not evidence it is wrong -- it is an absence
    of information, and absence is never evidence either way. If even ONE
    remaining candidate has no `semantic_action` requirement of its own, this
    returns None immediately and defers completely to the existing tie policy
    (`tiebreak.narrow`, then a provider/coder tie question) -- exactly as
    before this axis existed.
    """
    from . import requirement as _requirement
    semantic_reqs = [r for r in requirements if r.axis == "semantic_action"]
    if not semantic_reqs:
        return None
    remaining_codes = {c.code for c in remaining}
    action_by_code = {
        code: [r for r in semantic_reqs if r.candidate_code == code]
        for code in remaining_codes
    }
    if any(not action_by_code[code] for code in remaining_codes):
        return None    # incomplete coverage -- missing authority is not evidence
    supported = [
        cand for cand in remaining
        if all(_requirement.semantic_axis_status(r, fact, reconciliation, source)
              == _requirement.RequirementStatus.SUPPORTED
              for r in action_by_code[cand.code])
    ]
    return supported[0] if len(supported) == 1 else None


def _exact_direct_code_term_signal(fact: ClinicalFact, candidates: list[CandidateCode],
                                   source: Any) -> dict[str, list[dict]]:
    """issue #6, Codex's independent re-review (F9-R19-A): {candidate_code ->
    [matched direct atoms]} wherever this fact's own description literally IS
    a current CPT/HCPCS atom's own wording, restricted to atoms whose code is
    among `candidates`. AUDIT SIGNAL ONLY -- see `_settle_uniqueness`'s own
    comment at the call site and `AuthoritativeSource.exact_direct_code_term`'s
    docstring; this function never eliminates or selects anything, and its
    caller must not either. Degrades to {} when `source` does not implement
    the lookup, or the fact carries no description."""
    fn = getattr(source, "exact_direct_code_term", None)
    description = str(getattr(fact, "description", "") or "").strip()
    if not callable(fn) or not description:
        return {}
    try:
        hits = fn(description) or []
    except Exception:
        return {}
    codes = {c.code for c in candidates}
    out: dict[str, list[dict]] = {}
    for atom in hits:
        code = str((atom or {}).get("code") or "")
        if code in codes:
            out.setdefault(code, []).append(dict(atom))
    return out


# ---- pre-verification candidate admission (issue #6, Codex's independent re-review, ------
# F9-R23 Root Finding 1 / the Gate A-B clarification): "broad retrieval supplies recall;
# positive evidence supplies standing; complete candidate requirements authorize selection."
# A candidate reaching `select_entailed`/`corroborate` at all used to require nothing more
# than surviving the pool-construction/role/kind/separately-billable filters above -- a bare
# RAG or UMLS recall hit, or an LLM's own unconfirmed proposal, could still contest
# uniqueness on equal footing with a candidate the record's own words directly named. This
# decides EVERY candidate's admission standing before either evaluator is ever asked,
# reusing only EXISTING, already-authoritative signals -- never a new taxonomy, never a
# code/term list.
class CandidateStanding(str, Enum):
    SUPPORTED = "supported"       #: may enter the decisive shortlist and contest uniqueness
    CONTRADICTED = "contradicted"  #: audited as excluded; can never contest
    UNGROUNDED = "ungrounded"     #: recall-only until verified evidence earns standing


@dataclass(frozen=True)
class CandidateAdmission:
    """One candidate's full pre-verification admission record."""
    candidate_key: tuple[str, str]
    standing: CandidateStanding
    positive_axes: tuple[str, ...]
    contradicted_axes: tuple[str, ...]
    unresolved_axes: tuple[str, ...]
    evidence_span_ids: tuple[str, ...]
    descriptor_snapshot: dict
    source_lineage: tuple[str, ...]
    reason: str = ""

    def as_record(self) -> dict:
        return {"candidate_key": list(self.candidate_key), "standing": self.standing.value,
                "positive_axes": list(self.positive_axes),
                "contradicted_axes": list(self.contradicted_axes),
                "unresolved_axes": list(self.unresolved_axes),
                "evidence_span_ids": list(self.evidence_span_ids),
                "descriptor_snapshot": dict(self.descriptor_snapshot),
                "source_lineage": list(self.source_lineage), "reason": self.reason}


# Schema-level governed identity axes, not clinical vocabulary. Raw
# descriptor-token differences are explicitly audit/recall-only in
# `tiebreak`; treating that axis as identity here would promote the exact
# weak lexical hit this lifecycle exists to contain.
_IDENTITY_REQUIREMENT_AXES = frozenset({"semantic_action"})


def candidate_admission(fact: ClinicalFact, candidate: CandidateCode,
                        requirements: tuple, source: CodeSource, reconciliation,
                        coverage: "_requirement.CoverageCorpus | None" = None,
                        dos: str | None = None) -> CandidateAdmission:
    """One candidate's pre-verification admission standing.

    CONTRADICTED first, unconditionally: `semantic_eligibility.
    _service_role_control`/`_candidate_kind_control` -- the SAME
    authoritative role/kind compatibility controls already applied as a
    candidate-pool pre-filter elsewhere -- run again here, per-candidate,
    so a role/kind conflict disqualifies before any requirement is even
    read.

    Otherwise: every compiled `DescriptorRequirement` targeting this
    candidate is judged by the SAME judgement-independent status function
    the disposition/elimination layers already trust for it --
    `requirement.semantic_axis_status` (concept-equivalence, for
    `semantic_action`/`semantic_qualifier`) or `requirement.
    deterministic_status` (a real, independently-read search of the
    document) for every other axis -- never re-derived here, never an
    evaluator's own self-report. A CONTRADICTED axis disqualifies
    outright. Otherwise, standing is SUPPORTED only when a governed
    identity axis is independently supported or the record's normalized
    term exactly matches a current authoritative atom
    (`AuthoritativeSource.exact_direct_code_term`). Registry validity,
    UMLS lineage, retrieval rank, and raw descriptor overlap supply recall,
    never standing. A recall-only candidate may earn standing later only
    when both independent evaluators support its authoritative descriptor
    with validated, reconciled evidence. An unresolved `MUST_SUPPORT` axis
    is recorded (`unresolved_axes`) but does not itself withhold standing;
    the downstream disposition/requirement machinery resolves it."""
    from . import requirement as _requirement
    from . import semantic_eligibility as _semelig
    key = (candidate.code, candidate.system)
    role = _semelig._service_role_control([fact], [candidate], source, reconciliation, dos)
    role_decision = role.get(key)
    if role_decision is not None and role_decision.excluded:
        return CandidateAdmission(
            key, CandidateStanding.CONTRADICTED, (), ("service_role",), (), (),
            {"code": candidate.code, "descriptor": candidate.descriptor},
            (candidate.source,),
            reason=(f"candidate's authoritative classification "
                   f"{role_decision.candidate_role!r} is incompatible with the fact's "
                   f"own documented service_role {role_decision.fact_roles[0]!r}"
                   if role_decision.fact_roles else
                   f"candidate's authoritative classification "
                   f"{role_decision.candidate_role!r} is incompatible with the fact's "
                   f"own documented service_role"))
    kind_blocked = _semelig._candidate_kind_control([fact], [candidate], source, dos)
    if key in kind_blocked:
        return CandidateAdmission(
            key, CandidateStanding.CONTRADICTED, (), ("candidate_kind",), (), (),
            {"code": candidate.code, "descriptor": candidate.descriptor},
            (candidate.source,), reason=kind_blocked[key])

    positive: set[str] = set()
    identity_positive: set[str] = set()
    contradicted: set[str] = set()
    unresolved: set[str] = set()
    own_requirements = tuple(r for r in requirements if r.candidate_code == candidate.code)
    for req in own_requirements:
        status = (_requirement.semantic_axis_status(req, fact, reconciliation, source)
                 if req.axis in _SEMANTIC_AXES
                 else _requirement.deterministic_status(req, coverage))
        if status is _requirement.RequirementStatus.SUPPORTED:
            positive.add(req.axis)
            if req.axis in _IDENTITY_REQUIREMENT_AXES:
                identity_positive.add(req.axis)
        elif status is _requirement.RequirementStatus.CONTRADICTED:
            contradicted.add(req.axis)
        elif req.role is _requirement.RequirementRole.MUST_SUPPORT:
            unresolved.add(req.axis)

    direct_hits = _exact_direct_code_term_signal(fact, [candidate], source)
    if direct_hits.get(candidate.code):
        positive.add("direct_term")
        identity_positive.add("direct_term")
    # A source-bound term mapping establishes concept identity, not code
    # approval.  The candidate still has to survive its authoritative
    # descriptor, compiled requirements, two independent evaluations, and all
    # downstream claim controls.  Bare UMLS/RAG lineage remains recall-only.
    if _governed_term_mapping_grounded(candidate):
        positive.add("governed_term_mapping")
        identity_positive.add("governed_term_mapping")
    # issue #6, Codex's independent re-review (F9-R23 Root Finding 1): an
    # unresolved MUST_SUPPORT axis is recorded (`unresolved_axes`, for
    # audit) but deliberately does NOT, by itself, withhold standing here.
    # That axis's own resolution -- including the case where BOTH
    # evaluators independently confirm genuine, complete-coverage
    # NOT_DOCUMENTED and it validly grounds an elimination -- is exactly
    # what the EXISTING, already-correct `resolution._grounded_elimination`/
    # disposition machinery downstream exists to decide, via the
    # evaluators' own STRUCTURED judgement (never a bare deterministic
    # text search standing in for it). Gating pre-verification standing on
    # it here would short-circuit that mechanism before the verifier ever
    # gets a chance to confirm or contradict it -- an axis genuinely
    # undocumented on BOTH sides of a laterality-style split is a real,
    # already-tested elimination path, not a reason to withhold every
    # candidate from verification entirely.
    if contradicted:
        standing = CandidateStanding.CONTRADICTED
        reason = (f"the documentation positively contradicts this candidate's own "
                 f"required axis/axes: {', '.join(sorted(contradicted))}")
    elif identity_positive:
        standing = CandidateStanding.SUPPORTED
        reason = ("positive identity evidence: "
                  f"{', '.join(sorted(identity_positive))}; supported axes: "
                  f"{', '.join(sorted(positive))}")
    else:
        standing = CandidateStanding.UNGROUNDED
        reason = ("candidate recall/code-existence/topical similarity did not establish "
                  "identity; supported non-identity axes: "
                  f"{', '.join(sorted(positive)) or 'none'}")
    evidence_span_ids: tuple[str, ...] = ()
    if standing is CandidateStanding.SUPPORTED and reconciliation is not None:
        from app.contracts.source_evidence import ReconciliationStatus
        settled = reconciliation.by_span_id()
        evidence_span_ids = tuple(dict.fromkeys(
            s.span_id for s in (fact.evidence or [])
            if getattr(s, "anchored", False) and getattr(s, "span_id", None)
            and settled.get(s.span_id) is not None
            and settled[s.span_id].status is ReconciliationStatus.AGREED
        ))
    sources = {str(candidate.source)} if candidate.source else set()
    sources.update(str(s) for s in (candidate.authority or {}).get("sources", ()) if s)
    descriptor_snapshot = dict(
        (candidate.authority or {}).get("evaluation_descriptor_snapshot") or {})
    descriptor_snapshot.update({"code": candidate.code,
                                "descriptor": candidate.descriptor})
    return CandidateAdmission(
        key, standing, tuple(sorted(positive)), tuple(sorted(contradicted)),
        tuple(sorted(unresolved)), evidence_span_ids,
        descriptor_snapshot, tuple(sorted(sources)),
        reason=reason)


def _system_unresolved_line(fact: ClinicalFact, shortlist: list[CandidateCode],
                            system_unresolved: dict[str, str],
                            eliminated: dict[str, str], record: dict) -> ResolvedLine:
    """issue #6, Codex's independent re-review (F9-R19-A Finding 1): one or
    more candidates' evidence state could be neither confirmed nor
    eliminated (evaluator disagreement, an unreproduced descriptor
    identity, an uncited contradiction, an unvalidated not_documented
    verdict) -- a SYSTEM verification gap, never a documentation gap a
    provider could answer and never a coding judgement a coder owns.
    `documentation_gap` is deliberately left unset: `autonomy.decide`'s
    routing only ever builds a PROVIDER_QUERY from a real documentation
    gap, and this is explicitly not one. `rationale` carries
    `models.SYSTEM_UNRESOLVED_MARKER` so `pipeline.py`'s per-fact loop can
    synthesize the SAME retryable, `Destination.SYSTEM_HOLD`-routed gate
    shape already used for every other system-integrity hold in this
    codebase (e.g. `second_reading_relation_unplaced`) -- never a new,
    parallel destination."""
    from .models import SYSTEM_UNRESOLVED_MARKER
    named = ", ".join(f"{code} ({reason})" for code, reason in sorted(system_unresolved.items()))
    return ResolvedLine(
        fact=fact, chosen=None, alternatives=shortlist[:5],
        method=ResolutionMethod.ABSTAINED,
        documentation_gap=None,
        tie_record={**record, "system_unresolved": dict(sorted(system_unresolved.items())),
                   "eliminated": dict(sorted(eliminated.items()))},
        rationale=(f"{SYSTEM_UNRESOLVED_MARKER} candidate evidence could not be "
                  f"independently verified for: {named} -- system retry needed, "
                  f"never a provider question or a coder's judgement call"))


def _settle_uniqueness(fact: ClinicalFact, chosen: CandidateCode | None,
                       shortlist: list[CandidateCode], judgements: list,
                       eliminated_earlier: dict[str, str], why: str,
                       corroboration: str, reconciliation,
                       requirements: tuple = (),
                       coverage: "_requirement.CoverageCorpus | None" = None,
                       source: Any = None,
                       admissions: dict[str, CandidateAdmission] | None = None,
                       ) -> ResolvedLine:
    """Release ONLY when exactly one shortlisted candidate is still entailed; otherwise
    hand the survivors to the tie policy the deterministic path already uses.

    This is the SAME "eliminate -> unique -> narrow against the page -> release, else one
    targeted provider query" algorithm as `_decide`, over the same `tiebreak` module. The
    only thing that differs is who did the eliminating: there it is the descriptor
    features, here it is two models' named eliminations. Nothing about "the first model
    picked this one" is allowed to stand in for uniqueness.
    """
    # issue #6, Codex's independent re-review (F9-R13-C): `semantic_action`/
    # `semantic_qualifier` requirements are `role=MUST_SUPPORT` like any other
    # -- `_grounded_elimination` groups purely by ROLE, never by axis name, so
    # passing them in unfiltered would let a judge's generic NOT_DOCUMENTED
    # verdict (validated only by `requirement.validated_requirement`'s
    # LITERAL-TEXT search, never by the concept-equivalence/typed-attribute
    # test `semantic_axis_status` actually applies) eliminate a candidate
    # through a path neither this round's design nor Codex's spec intended:
    # these two axes select, they do not eliminate. Excluded here from both
    # elimination and `tiebreak.narrow`'s literal-text axis reporting
    # (`_axes_from_requirements` derives an axis for ANY requirement whose
    # name isn't already governed, regardless of `selectable`) -- the full,
    # unfiltered `requirements` (below and in `_select_by_semantic_axes`)
    # still carries them for audit completeness and for the one path that IS
    # meant to read them.
    _elimination_requirements = tuple(r for r in requirements if r.axis not in _SEMANTIC_AXES)
    # A verifier is allowed to return a complete disposition matrix without
    # nominating a code.  That is not a failed verification: it is a safer
    # representation of "classify every candidate, then let the deterministic
    # settlement layer decide."  The old early-return path discarded that
    # matrix and never called the independent evaluator, leaving every
    # multi-procedure fact whose first evaluator declined to choose in a
    # permanent tie.  With no proposal there is no candidate to special-case in
    # the legacy elimination view, so start from the whole shortlist and let the
    # independently validated disposition matrix account for every member.
    if chosen is None:
        remaining, eliminated = list(shortlist), {}
    else:
        remaining, eliminated = _uniqueness_view(
            fact, shortlist, chosen, judgements, eliminated_earlier,
            reconciliation, _elimination_requirements, coverage)
    # issue #6, Codex's independent re-review (F9-R15-B): tried in ADDITION to
    # (never instead of) the axis/requirement-based elimination just above --
    # narrows `remaining` further only when both independent evaluators'
    # structured, semantic per-candidate dispositions agree a candidate is
    # validly disposed. Operates on the ALREADY-narrowed `remaining` set (never
    # re-admits anything `_uniqueness_view` already eliminated), so this can
    # only shrink the standing set further, never widen or replace it.
    _disposition_verdict = _candidate_disposition_uniqueness(
        remaining, chosen, judgements, reconciliation, coverage,
        fact=fact, requirements=_elimination_requirements,
        admissions=admissions)
    _system_unresolved: dict[str, str] = {}
    if _disposition_verdict is not None:
        remaining, _further_eliminated, _system_unresolved = _disposition_verdict
        eliminated.update(_further_eliminated)
    # Candidates eliminated BEFORE the shortlist existed (a failed deterministic
    # constraint) belong in the same accounting: the record has to show the whole
    # retrieved pool being disposed of, not only the part the models were shown.
    for code, ruled_out in eliminated_earlier.items():
        eliminated.setdefault(code, ruled_out)
    record = {
        "stage": "code_selection_uniqueness",
        "shortlist": [c.code for c in shortlist],
        "selected": chosen.code if chosen is not None else "",
        "still_entailed": [c.code for c in remaining],
        "eliminated": dict(sorted(eliminated.items())),
        "judgements": [j.as_record() for j in judgements],
        # issue #6 F9-R6 Phase 5: the COMPILED requirement matrix itself (axis,
        # candidate, required/optional, role, expected terms, authority clause +
        # offset, source identity) alongside the per-evaluator verdicts already
        # carried in `judgements` above -- together they let an auditor
        # independently reproduce `requirement.validated_requirement`'s
        # clause-reproduction and span checks from this one durable record,
        # rather than trusting the elimination prose in `eliminated` on its own
        # say-so.
        "requirements": [r.as_record() for r in requirements],
        # issue #6 F9-R6-R4/R5 re-review: a single, self-describing, typed
        # coverage record (channel, content hash, page coverage) rather than
        # two loose primitives -- binds WHAT was searched, not just whether.
        "coverage": (coverage.as_record() if coverage is not None else None),
        # issue #6, Codex's independent re-review (F9-R19-A): whether this
        # fact's own description literally IS a current CPT/HCPCS atom's own
        # wording (`data_access.AuthoritativeSource.exact_direct_code_term`),
        # per still-standing candidate. AUDIT ONLY, exactly as that method's
        # own docstring requires -- recorded here for visibility, never read
        # by any elimination/selection branch above or below in this
        # function; a hit is positive identity evidence, not authorization.
        "exact_direct_code_term": _exact_direct_code_term_signal(fact, remaining, source),
    }
    # issue #6, Codex's independent re-review (F9-R21-A): a clinically
    # eligible candidate this mechanism could neither confirm NOR eliminate
    # must be checked BEFORE any release, not after -- its applicability is
    # genuinely UNKNOWN, and a rival whose status is unknown is not proof
    # the winner is right. Codex's independent exact-SHA reproduction: a
    # clean single-remaining winner released as `verified_entailment` while
    # an unresolved rival sat unaddressed in the same shortlist, because the
    # prior version checked this only AFTER a clean release already fired.
    # It is a SYSTEM verification gap (evaluator disagreement, an
    # unreproduced identity, an uncited verdict), never a documentation gap
    # or a coding judgement -- so it routes to a retryable system hold, not
    # a provider query or a tie question, regardless of how cleanly
    # `remaining` itself narrowed.
    if _system_unresolved:
        return _system_unresolved_line(fact, shortlist, _system_unresolved,
                                       eliminated, record)

    if not remaining:
        return ResolvedLine(
            fact=fact, chosen=None, alternatives=shortlist[:5],
            method=ResolutionMethod.ABSTAINED, tie_record=record,
            rationale=("both independent evaluators accounted for every candidate, "
                       "and no candidate remained supported by the documentation"))

    # The initial `chosen` value is a proposal, not an authority.  Ordinarily
    # the unique survivor is the proposal itself.  If the complete structured
    # disposition pass instead eliminates that proposal and leaves a different
    # sole survivor, reselection is safe only when that pass actually ran AND
    # the survivor has governed identity standing.  In that case both
    # evaluators already validated the survivor's current descriptor and every
    # rival has been accounted for; holding merely because the first proposal
    # was wrong would invert propose-then-verify into propose-as-truth.
    #
    # issue #6, Codex's independent re-review (F9-R24-B, second re-review;
    # reverted after regression testing): a version of this branch once
    # required `CandidateStanding.SUPPORTED` on the ordinary "the proposal
    # survives, nothing contests it" shape too, matching the reselection
    # branch three lines down. That is WRONG for this branch specifically:
    # `candidate_admission`'s positive-identity bar (a compiled requirement,
    # a direct authoritative term, or a governed crosswalk mapping) is
    # calibrated for DISCRIMINATING BETWEEN RIVALS, not for gating an
    # uncontested single retrieval both evaluators independently entail
    # against a real, content-relevant citation (Gate A). Enforcing it here
    # too reproduced, concretely (25 failing tests), the exact regression
    # independently discovered and reverted earlier in this file's own
    # history: a blanket post-hoc standing veto re-litigates a release Gate
    # A + independent dual-evaluator entailment already validated, using a
    # cruder, unrelated proxy. The reselection branch below is different in
    # kind -- it is CHOOSING a candidate the original proposal did not name,
    # which is exactly the rival-discrimination case `CandidateStanding`
    # exists for -- so its own standing check stays as Codex wrote it.
    if len(remaining) == 1:
        survivor = remaining[0]
        if chosen is None and _disposition_verdict is not None:
            matrix_record = {
                **record,
                "selected": survivor.code,
                "selected_from_complete_disposition_matrix": True,
            }
            note = ("the first evaluator made no proposal; two independent, "
                    "descriptor-bound disposition matrices left exactly one "
                    "source-cited candidate")
            return _entailed_line(
                fact, survivor, shortlist, f"{why}; {note}" if why else note,
                corroboration, uniqueness=matrix_record)
        if chosen is not None and survivor.code == chosen.code:
            return _entailed_line(fact, survivor, shortlist, why, corroboration,
                                  uniqueness=record)
        survivor_admission = (admissions or {}).get(survivor.code)
        if (chosen is not None
                and _disposition_verdict is not None
                and survivor_admission is not None
                and survivor_admission.standing is CandidateStanding.SUPPORTED):
            reselection_record = {
                **record,
                "proposed": chosen.code,
                "selected": survivor.code,
                "reselected_from_verified_survivor": True,
            }
            note = (f"the initial proposal {chosen.code} was eliminated; "
                    f"{survivor.code} is the sole independently verified candidate "
                    "with governed identity standing")
            return _entailed_line(
                fact, survivor, shortlist, f"{why}; {note}" if why else note,
                corroboration, uniqueness=reselection_record)

    # issue #6, Codex's independent re-review (F9-R13-C): one additional
    # selection condition, tried BEFORE the original-document tie policy --
    # never in place of its own entailment/DOS/measurement controls. A
    # semantic-axis winner must still clear every one of them, exactly like a
    # `tiebreak.narrow` winner does two lines below; this never lowers that
    # bar, only adds one more way to reach it.
    semantic_winner = _select_by_semantic_axes(fact, remaining, requirements,
                                               reconciliation, source)
    if (semantic_winner is not None
            and (_disposition_verdict is not None
                 or all(j.entails(semantic_winner.code) for j in judgements))
            and _evaluate(fact, semantic_winner, reconciliation=reconciliation) is not None
            and not _interval_unsupported(fact, parse_descriptor(semantic_winner.descriptor))):
        note = (f"{len(remaining)} candidates remained entailed; {semantic_winner.code}'s "
                f"own action/qualifier requirements are the only ones fully supported by "
                f"this fact's own reconciled evidence")
        return _entailed_line(fact, semantic_winner, shortlist,
                              (f"{why}; {note}" if why else note), corroboration,
                              uniqueness={**record, "semantic_axis_selection": semantic_winner.code})

    # STEPS 3 and 4 -- several candidates are independently entailed, so the ORIGINAL
    # DOCUMENT decides, exactly as it does for a deterministic tie.
    tie = _tiebreak.narrow(fact, remaining, reconciliation, _elimination_requirements)
    winner = tie.winner
    if (winner is not None
            and (_disposition_verdict is not None
                 or all(j.entails(winner.code) for j in judgements))
            and _evaluate(fact, winner, reconciliation=reconciliation) is not None
            and not _interval_unsupported(fact, parse_descriptor(winner.descriptor))):
        note = (f"{len(remaining)} candidates remained entailed, and the tie was narrowed "
                f"against the original document ({tie.proof}): {tie.detail}")
        return _entailed_line(fact, winner, shortlist,
                              (f"{why}; {note}" if why else note), corroboration,
                              uniqueness={**record, **tie.as_record()})

    # STEP 5 -- ONE targeted provider query, never a generic coder queue.
    return _tie_escalation(
        fact, remaining, reconciliation,
        f"{len(remaining)} shortlisted candidates are still entailed by the documentation "
        f"({', '.join(c.code for c in remaining)}) -- agreement on one of them is not "
        f"evidence that the others are wrong",
        tie=tie, record=record, requirements=requirements)


def _propose_then_verify(fact: ClinicalFact, source: CodeSource,
                         pool: list[CandidateCode], llm, corroborate=None,
                         dos: str | None = None,
                         reconciliation=None,
                         coverage: "_requirement.CoverageCorpus | None" = None,
                         elig_facts: list[ClinicalFact] | None = None
                         ) -> ResolvedLine:
    """Recall as candidate GENERATOR, authoritative descriptor + entailment as TRUTH.
    Widen the pool with validated LLM proposals, select the candidate whose OFFICIAL
    descriptor the documentation entails, then (when a corroborator is supplied)
    require an INDEPENDENT second model to agree before accepting. Escalate if the
    selection finds nothing OR the second model disagrees. Nothing bills on recall
    alone, and nothing bills on a single model's say-so.

    Agreement is necessary and NOT sufficient. Both judgements answer about the WHOLE
    shortlist, and the selected candidate is released only when every OTHER shortlisted
    candidate carries a NAMED elimination. When more than one candidate is still entailed
    the line is a TIE, and it goes to the same document-first tie policy the deterministic
    path uses -- narrowed against the original page, else ONE targeted provider query.
    (Codex F8-R1: two models agreeing on one candidate never eliminated the rest.)

    issue #6 F9-R11-H-D, fifth re-review: model-PROPOSED candidates are role-
    controlled over the FULL candidate universe (the already-eligible retrieval
    `pool` plus every validated proposal) BEFORE any shortlist is built or any
    verification call runs -- not checked against `line.chosen` after selection
    (the fourth re-review's own defect: an incompatible proposal that lost a
    verifier TIE against a compatible retrieved candidate never reached
    `line.chosen` at all, so a post-selection check could not see it, and the
    tie escalation swallowed the compatible candidate along with it). A
    role-incompatible proposal is excluded here, before it can ever contest a
    tie; a genuine multi-role ambiguity across the combined universe aborts
    with the typed `classification_data_gap` hold immediately, before any
    verifier call is spent.

    `pool` here MUST be the caller's UNFILTERED retrieval/index universe
    (sixth re-review: passing the caller's own already-eligibility-filtered
    pool silently dropped every candidate the FIRST pass had already
    excluded from this function's own "complete" report -- an excluded
    retrieval candidate's exact reason disappeared from the audit trail
    exactly because it had been correctly excluded). Every validated
    proposal is included too, including ones `_evaluate` rejects for an
    unsupported measurement interval -- a different axis than role
    eligibility, but still a real exclusion this function's own audit must
    not silently drop. The returned line's `candidate_eligibility` is the
    COMPLETE report -- every retrieval/index candidate AND every proposal,
    chosen, excluded, or neither -- so the audit trail (and the ClaimBundle
    that projects it) shows every candidate this fact's resolution actually
    considered, with its exact reason."""
    from . import verify as _verify
    from . import semantic_eligibility as _semelig
    proposed_evals = [(c, *_evaluate_reason(fact, c, source, reconciliation))
                      for c in _verify.propose_codes(fact, source, llm)]
    proposals_raw = [c for c, m, _r in proposed_evals
                     if m is not None and not m.interval_unsupported]
    proposals_unsupported = [c for c, m, _r in proposed_evals
                             if m is not None and m.interval_unsupported]
    # Registry-valid proposals `_evaluate` eliminated OUTRIGHT (laterality
    # contradiction, or a documented measurement outside the descriptor's
    # bounded interval) -- issue #6 F9-R11-H-D, seventh re-review: these
    # never reached `full_universe` at all before, so the audit trail could
    # not say a proposal was even considered, let alone why it was excluded.
    proposals_deterministically_excluded = [(c, r) for c, m, r in proposed_evals
                                            if m is None]

    # Candidate admission is per clinical event.  Sibling procedures/supplies
    # in the same composed intent remain available to downstream relationship
    # and claim-edit logic, but must not change this event's candidate kind or
    # role before its own authoritative descriptors are verified.
    facts_for_role_check = [fact]
    pool_ids = {(c.code, c.system) for c in pool}
    extra: list[CandidateCode] = []
    seen_extra: set[tuple[str, str]] = set()
    for c in proposals_raw + proposals_unsupported:
        key = (c.code, c.system)
        if key not in pool_ids and key not in seen_extra:
            seen_extra.add(key)
            extra.append(c)
    full_universe = list(pool) + extra
    candidate_eligibility = _semelig.eligibility_report(
        facts_for_role_check, full_universe, source, dos, reconciliation)
    # Merge the deterministically-excluded proposals into the SAME report.
    # issue #6 F9-R11-H-D, eighth re-review: when the SAME (code, system) also
    # arrived through retrieval, `eligibility_report` already emitted a record
    # for it (typically eligible=True) -- a skip-if-already-reported merge left
    # that stale, incorrect record standing (`eligible: true, reason: null`)
    # even though this exact candidate is deterministically excluded (laterality
    # contradiction / out-of-range measurement). The exclusion must OVERRIDE the
    # existing record for that identity, not be dropped by it.
    deterministic_exclusions = {(c.code, c.system): reason
                                for c, reason in proposals_deterministically_excluded}
    for record in candidate_eligibility:
        reason = deterministic_exclusions.pop((record["code"], record["system"]), None)
        if reason:
            record["eligible"] = False
            record["reason"] = reason
            record["role_control"]["blocks_line"] = False
    for (code, system), reason in deterministic_exclusions.items():
        candidate_eligibility.append({
            "code": code, "system": system, "eligible": False, "reason": reason,
            "role_control": {"status": "not_evaluated", "fact_roles": [],
                             "candidate_role": None, "blocks_line": False,
                             "authority_source_id": None, "authority_version": None}})
    # issue #6, Codex's independent re-review (F9-R19-A Finding 3): a
    # candidate the source's own authoritative data declares NOT separately
    # reportable (bundled/non-covered/MUE 0 -- e.g. an informational quality/
    # performance-measure code) must not compete with payable clinical
    # candidates at all. `source.separately_billable` already existed and was
    # already applied to a resolved LINE's final chosen code (`pipeline.py`,
    # post-resolution) -- but a candidate that never WON a tie (both sides
    # stayed "entailed") never reached that check, so it could still
    # contest the tie undetected. Applied here, at the SAME candidate-pool
    # partition stage as the service-role control, before any verifier call.
    for record in candidate_eligibility:
        if not record["eligible"]:
            continue
        try:
            blocked = source.separately_billable(
                record["code"], record["system"], dos) is Outcome.BLOCKED
        except Exception:
            blocked = False
        if blocked:
            record["eligible"] = False
            record["reason"] = "not separately reportable per authoritative data"
    # issue #6, Codex's independent re-review (F9-R21-C): a candidate whose
    # OWN authoritative classification is categorically the WRONG KIND of
    # service for a non-procedure fact (a quality-measure/E&M/anesthesia-
    # status code surviving for a documented SUPPLY/IMAGING/DRUG/DIAGNOSIS
    # event) must not compete in that fact's shortlist at all -- reproduced
    # live: a suture-anchor supply fact's pool retained unrelated quality-
    # measure/imaging/other candidates as "entailed" with nothing checking
    # whether they were even the right KIND of code.
    for (code, system), reason in _semelig._candidate_kind_control(
            facts_for_role_check, full_universe, source, dos).items():
        for record in candidate_eligibility:
            if (record["code"], record["system"]) == (code, system) and record["eligible"]:
                record["eligible"] = False
                record["reason"] = reason
                break
    eligible_ids = {(r["code"], r["system"]) for r in candidate_eligibility
                    if r["eligible"]}

    if any(r.get("role_control", {}).get("blocks_line") for r in candidate_eligibility):
        line = ResolvedLine(
            fact=fact, chosen=None, method=ResolutionMethod.ABSTAINED,
            alternatives=[c for c in full_universe
                         if (c.code, c.system) in eligible_ids],
            rationale=("this claim-line intent's candidate universe (retrieval plus "
                      "model-proposed candidates) classifies into more than one "
                      "service role (operative vs. anesthesia) while the documented "
                      "facts are conflicting or of mixed kind -- a composition/coder "
                      "decision, never auto-resolved from an ambiguous pool"),
            documentation_gap="classification_data_gap:service_role_conflict")
        line.candidate_eligibility = candidate_eligibility
        return line

    proposals = [c for c in proposals_raw if (c.code, c.system) in eligible_ids]
    pool = [c for c in pool if (c.code, c.system) in eligible_ids]
    line = _propose_then_verify_core(
        fact, source, pool, proposals, proposals_unsupported, llm, corroborate,
        dos=dos, reconciliation=reconciliation, coverage=coverage)
    line.candidate_eligibility = candidate_eligibility
    return line


def _propose_then_verify_core(fact: ClinicalFact, source: CodeSource,
                              pool: list[CandidateCode],
                              proposals: list[CandidateCode],
                              proposals_unsupported: list[CandidateCode],
                              llm, corroborate=None, dos: str | None = None,
                              reconciliation=None,
                              coverage: "_requirement.CoverageCorpus | None" = None
                              ) -> ResolvedLine:
    """The shortlist-build + verification loop, over an ALREADY role-
    eligibility-filtered `pool`/`proposals` (issue #6 F9-R11-H-D, fifth
    re-review split this out of `_propose_then_verify` so eligibility runs
    BEFORE any candidate reaches a verifier, not after one is selected)."""
    from . import verify as _verify
    # WHOSE second opinion this run has, decided once from the two callables' declared
    # identities: it governs whether an agreement may be credited as independent
    # confirmation below (and it must be computed even when `corroborate` is None, since
    # "no second opinion" is itself one of the non-independent origins).
    corroboration = _verify.corroboration_origin(llm, corroborate)
    retrieved_matches = _ranked(fact, pool, source, reconciliation)
    unsupported = [m.candidate for m in retrieved_matches
                   if m is not None and m.interval_unsupported] + proposals_unsupported
    retrieved_all = [m.candidate for m in retrieved_matches if not m.interval_unsupported]
    # issue #6 F9-R7-C: split off UMLS-sourced candidates for their OWN reserved
    # lane (see `_MIN_UMLS_SLOTS`) -- they must not compete on raw score against
    # embedding-similarity-ranked retrieval.
    #
    # issue #6 F9-R9-C, Codex's independent re-review of 6ff2761: `_merge_candidate`
    # (F9-R8-C) keeps the HIGHER-scored candidate's scalar `.source` when a code is
    # proposed by both retrieval and UMLS -- checking `.source == "umls_recall"`
    # alone therefore misses a merged candidate whenever retrieval outscored UMLS
    # for it, even though its `authority["sources"]` still names `umls_recall`.
    # Reproduced directly: 8 higher-scored rivals plus a target proposed by both
    # retrieval (lower score than the rivals) and UMLS -- the merged target's
    # `.source` became "retrieval", so it got no reserved-lane protection and was
    # crowded out before verification, the exact defect `_MIN_UMLS_SLOTS` exists
    # to prevent. `_candidate_sources` reads the REAL merged provenance
    # (`_merge_candidate`'s own `authority["sources"]` field) instead of the
    # single scalar, so admission can never disagree with what the lineage
    # itself already states.
    def _candidate_sources(c: CandidateCode) -> set[str]:
        sources = {str(getattr(c, "source", "") or "")} if getattr(c, "source", "") else set()
        sources.update(str(s) for s in (getattr(c, "authority", None) or {}).get("sources", ()) if s)
        return sources

    umls_seeds = [c for c in retrieved_all if "umls_recall" in _candidate_sources(c)]
    retrieved = [c for c in retrieved_all if "umls_recall" not in _candidate_sources(c)]
    umls_cap = min(len(umls_seeds), _MIN_UMLS_SLOTS)
    # Fix4: reserve a floor of shortlist slots for authoritative RETRIEVAL so LLM
    # memory proposals cannot crowd it out. Keep up to (VERIFY_K - floor) proposals
    # first, then the reserved UMLS lane, then retrieved, then any leftover
    # proposals and UMLS seeds fill remaining room.
    prop_cap = max(0, VERIFY_K - _MIN_RETRIEVED_SLOTS - umls_cap)
    order: list[CandidateCode] = []
    seen: set[str] = set()
    for c in (proposals[:prop_cap] + umls_seeds[:umls_cap] + retrieved
             + proposals[prop_cap:] + umls_seeds[umls_cap:]):
        if c.code not in seen:
            seen.add(c.code)
            order.append(c)
    # Fix3: drop DOS-inactive candidates before capping to the shortlist.
    shortlist = _active_only(order, source, dos)[:VERIFY_K]
    if not shortlist and unsupported:
        return _bounded_interval_hold(fact, unsupported)
    # issue #6, Codex's independent re-review (F9-R17-A): bind ONE authoritative
    # descriptor per candidate now, before anything downstream reads
    # `candidate.descriptor` -- requirement compilation, both verifier prompts,
    # response parsing, tie comparison, audit, and the released line all read
    # this SAME bound field from here on, so what a model is shown and what its
    # answer is validated against can never diverge again.
    shortlist = _bind_evaluation_descriptors(shortlist, source)
    # issue #6 F9-R6: compiled ONCE against the whole shortlist and passed
    # identically to every verifier call below (both models judge the SAME
    # requirement_ids -- structural, not coincidental) and into uniqueness
    # settlement. A reselection round's narrower `cands` subset still renders
    # correctly against this same list (`verify._requirement_options` only shows
    # entries whose candidate is in the option list actually passed).
    from . import requirement as _requirement
    requirements = _requirement.compile_requirements(shortlist, source)
    # issue #6, Codex's independent re-review (F9-R23 Root Finding 1 / Gate
    # A-B clarification): decide EVERY candidate's pre-verification
    # admission standing before either evaluator is ever asked -- "broad
    # retrieval supplies recall; positive evidence supplies standing;
    # complete candidate requirements authorize selection." A CONTRADICTED
    # candidate (role/kind incompatible, or a compiled requirement the
    # record positively contradicts) is excluded outright here, never
    # shown to a verifier -- the same, already-established pre-filter
    # pattern `_service_role_control`/`_candidate_kind_control`/
    # `separately_billable` already apply earlier in this same function.
    #
    # An UNGROUNDED candidate (recall/proposal alone, no positive identity
    # signal) is deliberately NOT ALSO removed from the shortlist shown to
    # the verifiers here -- doing so changes the shortlist's own SIZE,
    # which changes whether the verifiers and `tiebreak.narrow` see a
    # genuine TIE to force axis discovery against, and independently
    # reproduced a regression: removing an ungrounded sibling let the one
    # remaining candidate release on an axis that was STILL genuinely
    # unresolved for it too, because a single-candidate shortlist no
    # longer forced that axis to be discovered and checked. Standing is
    # instead enforced only on the RESELECTION branch of
    # `_settle_uniqueness` (issue #6, F9-R24-B second re-review, then
    # reverted for the ORDINARY branch after regression testing): choosing
    # a different candidate than the original proposal requires that
    # survivor to have `CandidateStanding.SUPPORTED`, converting a
    # reselection onto an UNGROUNDED survivor into a system-hold via
    # `_system_unresolved_line`. The ordinary "the proposal survives,
    # nothing contests it" shape deliberately does NOT add this check --
    # see the comment at that branch for why a blanket veto there
    # reproduces the exact regression this pre-verification pool filter's
    # own comment already warns about. There is no separate wrapper
    # function; the one real check lives inline in `_settle_uniqueness`
    # itself, on the reselection branch only.
    admissions = {c.code: candidate_admission(fact, c, requirements, source,
                                              reconciliation, coverage, dos)
                 for c in shortlist}
    # issue #6, Codex's independent re-review (F9-R23 Root Finding 1): the
    # pre-verification pool filter is deliberately NARROWER than
    # `CandidateStanding.CONTRADICTED` itself -- only a role/kind
    # incompatibility (the SAME two categorical exclusions
    # `_service_role_control`/`_candidate_kind_control` already apply
    # elsewhere in this function, unrelated to this round's change and
    # already proven safe) removes a candidate from the shortlist shown to
    # the verifiers. A CONTRADICTED compiled-REQUIREMENT axis (e.g.
    # laterality) stays visible in `admissions` for audit, but is left for
    # the EXISTING, already-tested `_grounded_elimination`/disposition
    # machinery to act on post-verification -- independently reproduced:
    # removing a requirement-contradicted sibling pre-verification changed
    # the remaining candidate's shortlist from a genuine TIE to a solo
    # entry, which stopped `tiebreak.narrow`/the verifiers from ever being
    # forced to discover and check that SAME axis for the survivor too.
    _categorical_axes = frozenset({"service_role", "candidate_kind"})
    verifiable = [c for c in shortlist
                 if not (admissions[c.code].standing is CandidateStanding.CONTRADICTED
                        and set(admissions[c.code].contradicted_axes) & _categorical_axes)]

    def _with_admissions(line: ResolvedLine) -> ResolvedLine:
        if line.tie_record is None:
            return line
        return _dc_replace(line, tie_record={
            **line.tie_record,
            "candidate_admissions": {code: a.as_record() for code, a in admissions.items()}})

    if not verifiable:
        contradicted_admissions = {code: a.reason for code, a in admissions.items()
                                   if a.standing is CandidateStanding.CONTRADICTED}
        record = {"stage": "candidate_admission",
                 "shortlist": [c.code for c in shortlist],
                 "admissions": {code: a.as_record() for code, a in admissions.items()}}
        # Every candidate was positively, validly disqualified -- nothing
        # is left standing to ask a provider or a coder about either; this
        # documented event has no defensible candidate in this pool.
        return ResolvedLine(
            fact=fact, chosen=None, alternatives=shortlist[:5],
            method=ResolutionMethod.ABSTAINED, tie_record=record,
            rationale=(f"no candidate in this pool has positive standing -- every "
                      f"candidate was positively disqualified: "
                      f"{'; '.join(f'{c} ({r})' for c, r in sorted(contradicted_admissions.items()))}"))
    # issue #6, Codex's independent re-review (F9-R18-A, reopened P1): clinical-
    # attribute axis-conflict enforcement is no longer branch-local here -- a
    # check confined to this one propose-then-verify path could never see the
    # deterministic `_take`/`_decide` shortcuts in `_resolve_core`, which is
    # exactly how an unauthorized side-specific release slipped through. It is
    # now the SHARED `_apply_attribute_axis_conflict_guard`, applied once in
    # public `resolve()` after `_resolve_core` returns, regardless of which
    # internal path produced the line -- see that function's docstring.
    # code -> the NAMED reason an earlier round's judgement ruled that candidate out.
    # Keyed by code (not a bare set) because a release now has to be able to say WHY every
    # alternative is gone, not merely that it was skipped.
    tried: dict[str, str] = {}
    missing_tried: dict[str, str] = {}
    # Deterministically eliminated before any model saw them: a required bounded interval
    # the documentation does not support. Named in the uniqueness record so the audit trail
    # shows the whole pool being accounted for, not just the part the models judged.
    constraint_eliminated = {c.code: ("the descriptor requires a bounded measurement the "
                                      "documentation does not support")
                             for c in unsupported}
    last_reason = ""
    # Every bounded shortlist candidate gets one chance.  The former fixed
    # re-selection count could stop before a lower-ranked but fully supported
    # candidate was ever evaluated.  This bound is still finite (the shortlist
    # is already capped above) and ``tried`` removes at least one candidate on
    # every continuing iteration, so it cannot loop indefinitely.
    for _ in range(len(verifiable)):
        cands = [c for c in verifiable if c.code not in tried]
        if not cands:
            break
        primary = _verify.select_entailed(fact, cands, source, llm, requirements,
                                          reconciliation=reconciliation, coverage=coverage)
        chosen, why = primary.chosen, primary.reason
        if chosen is None:
            # A complete per-candidate answer is useful even when the evaluator
            # deliberately declines to nominate a billing code.  Obtain the
            # independent matrix over the exact same shortlist and let the shared
            # deterministic settlement function select only a unique, source-cited
            # survivor.  Without a second evaluator we retain the existing tie path.
            if corroborate is not None:
                second = _verify.corroborate(
                    fact, cands, source, corroborate, requirements,
                    reconciliation=reconciliation, coverage=coverage)
                note = ("independent disposition reconciliation"
                        if _independently_corroborated(corroboration)
                        else "a second disposition was obtained, but not from an "
                             "independent origin")
                # A prior candidate may have failed solely for a missing
                # requirement.  If both evaluators now eliminate every remaining
                # candidate, the prior gap is the only live path and must retain
                # its provider-query classification.  Do this only after the
                # remaining pool has actually been evaluated; if either evaluator
                # still entails an alternative, normal uniqueness settlement below
                # remains authoritative.
                if (missing_tried
                        and all(not primary.entails(c.code)
                                and not second.entails(c.code) for c in cands)):
                    details = list(dict.fromkeys(missing_tried.values()))
                    question = "; ".join(details)
                    return _with_admissions(ResolvedLine(
                        fact=fact, chosen=None, alternatives=verifiable[:5],
                        method=ResolutionMethod.ABSTAINED,
                        documentation_gap=question,
                        rationale=("PROVIDER QUERY — every otherwise-plausible "
                                   "remaining candidate requires an element the "
                                   "documentation does not establish "
                                   f"({question})")))
                return _with_admissions(_settle_uniqueness(
                    fact, None, cands, [primary, second],
                    {**constraint_eliminated, **tried},
                    f"{why}; {note}" if why else note,
                    corroboration, reconciliation, requirements,
                    coverage, source, admissions=admissions))
            # Tie policy step 5: one evaluator supplied no unique selection and
            # there is no independent disposition matrix, so name the missing
            # discriminating fact rather than guessing.
            return _with_admissions(_tie_escalation(
                fact, verifiable, reconciliation,
                "no candidate's authoritative descriptor is fully entailed by the "
                "documentation (verified)", requirements=requirements))
        chosen_match = _evaluate(fact, chosen, source, reconciliation)
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
        judgements = [primary]
        missing, why2 = False, ""
        if corroborate is not None:
            # The second opinion re-judges the WHOLE shortlist under the same contract and
            # is not told which candidate was picked -- a corroborator asked only "is THIS
            # one entailed?" cannot notice that another candidate is entailed too, which is
            # precisely how a non-unique code used to auto-release (Codex F8-R1).
            second = _verify.corroborate(fact, cands, source, corroborate, requirements,
                                         reconciliation=reconciliation, coverage=coverage)
            if not second.entails(chosen.code):
                why2 = (second.elimination_of(chosen.code) or second.reason
                        or "the independent second judgement does not find this "
                           "descriptor entailed by the documentation")
                last_reason = why2
                missing = second.missing_element.get(chosen.code, False)
                tried[chosen.code] = why2
                if missing:
                    missing_tried[chosen.code] = why2
            else:
                judgements.append(second)
                note = ("independently confirmed"
                        if _independently_corroborated(corroboration)
                        else "a second opinion agreed, but not from an independent origin")
                why = f"{why}; {note}" if why else note
        if chosen.code not in tried:
            # Both judgements (or the only one there is) accept the chosen candidate. That
            # makes it DEFENSIBLE, not UNIQUE -- which is the whole finding -- so release
            # only if nothing else on the shortlist survived, and otherwise settle it
            # against the original document through the same tie policy.
            return _with_admissions(_settle_uniqueness(
                fact, chosen, verifiable, judgements,
                {**constraint_eliminated, **tried}, why,
                corroboration, reconciliation, requirements,
                coverage, source, admissions=admissions))
        # A missing element disqualifies THIS candidate; it does not prove every
        # other authoritative candidate is an under-code.  Continue through the
        # remaining pool exactly as for any other named elimination.  If nothing
        # else survives, `_tie_escalation` below still converts the compiled missing
        # requirement into one targeted provider question.  This avoids the generic
        # failure where a first-ranked overqualified candidate prevented a later,
        # fully supported candidate from ever being evaluated.
        #
        # Otherwise this is a WRONG code; `tried` already carries the named reason,
        # so the next round likewise re-selects from the candidates that remain.
    # If every candidate has now been eliminated and at least one survivor-in-
    # principle failed only because a required element is not documented, the
    # unresolved work is a provider question.  Reaching this point proves the
    # whole pool was tried; this is deliberately later than reselection so one
    # overqualified first pick cannot hide a fully supported alternative.
    if missing_tried and all(c.code in tried for c in verifiable):
        details = list(dict.fromkeys(missing_tried.values()))
        question = "; ".join(details)
        return _with_admissions(ResolvedLine(
            fact=fact, chosen=None, alternatives=verifiable[:5],
            method=ResolutionMethod.ABSTAINED,
            documentation_gap=question,
            rationale=("PROVIDER QUERY — every otherwise-plausible remaining candidate "
                       "requires an element the documentation does not establish "
                       f"({question})")))

    # Tie policy step 5, and the case the directive names explicitly: THE MODELS
    # DISAGREED. That is never a reason to send an otherwise-resolved line to a generic
    # coder queue. The candidates the two models argued over are re-inspected against
    # the ORIGINAL DOCUMENT for the axes that actually distinguish them, and what the
    # page cannot settle becomes one specific question about the record.
    #
    # The question is asked about the candidates that were actually SELECTED AND
    # REJECTED, not about the whole shortlist: the directive asks for ONE targeted
    # query, and a question naming every code retrieval happened to return is not one.
    contested = [c for c in verifiable if c.code in tried] or verifiable
    return _with_admissions(_tie_escalation(
        fact, contested, reconciliation,
        f"independent second-model verification confirmed no candidate after "
        f"re-selection ({last_reason})", requirements=requirements))


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


def _origin_caveat(corroboration: str) -> str:
    """Why this entailment did NOT earn independent status — the audit sentence that has to
    accompany a code the system is keeping but cannot call independently confirmed."""
    from . import verify as _verify
    return {
        _verify.NO_CORROBORATION:
            "no independent second opinion was configured, so the entailment rests on a "
            "single model judgement",
        _verify.SHARED_ORIGIN:
            "the second opinion came from the SAME model provider as the primary "
            "verification — one vendor's judgement sampled twice is model self-confidence, "
            "not independent confirmation",
        _verify.UNDECLARED_ORIGIN:
            "the verification and corroboration calls declare no provider identity, so "
            "their independence could not be established",
    }.get(corroboration, "independent corroboration was not established")


def _independently_corroborated(corroboration: str) -> bool:
    from . import verify as _verify
    return corroboration in _verify.INDEPENDENT_CORROBORATION_ORIGINS


def _entailed_line(fact: ClinicalFact, chosen: CandidateCode,
                   shortlist: list[CandidateCode], why: str,
                   corroboration: str,
                   uniqueness: dict | None = None) -> ResolvedLine:
    """The line for a candidate whose AUTHORITATIVE descriptor the verifier found entailed.

    VERIFIED is the GROUNDED, autonomy-eligible method, and `autonomy` reads it as "the
    documentation entails this descriptor and an INDEPENDENT second model confirmed it".
    That second half is a claim about the corroborating assertion's ORIGIN, so it is
    checked here rather than assumed from the fact that a corroborator was configured:
    unless the corroborating judgement came from a genuinely distinct origin, the two
    agreeing calls are one vendor's opinion sampled twice.

    When independence is not established the code is still KEPT and offered — it is a
    candidate from the authoritative tables whose official descriptor the documentation
    entails, and dropping it would under-code — but the method is ARBITRATED, which is
    exactly "a model picked among candidates": `autonomy` discounts its confidence and
    always routes it to a coder. The agreement itself stays visible in the rationale, so
    the audit trail records that it happened and why it earned nothing. (Round 5, phase 5;
    the milder, conjunctive sibling of the F6-R3 necessity defect.)"""
    independent = _independently_corroborated(corroboration)
    base = (f"authoritative descriptor entailed by documentation: {why}"
            if why else "authoritative descriptor entailed by documentation")
    if not independent:
        base = (f"{base}; NOT independently corroborated — {_origin_caveat(corroboration)}"
                f" — needs a coder")
    return ResolvedLine(
        fact=fact, chosen=chosen,
        alternatives=[c for c in shortlist if c.code != chosen.code][:4],
        method=(ResolutionMethod.VERIFIED if independent
                else ResolutionMethod.ARBITRATED),
        tie_record=({**uniqueness, "released_code": chosen.code} if uniqueness else None),
        rationale=base)


def _decide(fact: ClinicalFact, pool: list[CandidateCode],
            authority: str | None = None,
            source: CodeSource | None = None,
            dos: str | None = None,
            reconciliation=None) -> ResolvedLine:
    """The directive's TIE POLICY over a candidate pool, in its stated order.

      1. ELIMINATE candidates that fail a required constraint (`_evaluate` drops a
         contradicted axis; `_active_only` drops a code inactive on the DOS).
      2. SELECT automatically only when ONE candidate uniquely satisfies every
         documented axis.
      3. Otherwise RE-INSPECT only the survivors' discriminating axes against the
         original document (`tiebreak.narrow`).
      4. RELEASE the candidate the page uniquely entails.
      5. Otherwise issue ONE targeted provider query -- never a generic coder review
         because the candidates tied."""
    if source is not None and dos:
        active = _active_only(pool, source, dos)   # Fix3: no DOS-inactive deterministic pick
        if pool and not active:                    # all candidates inactive on the DOS
            return ResolvedLine(
                fact=fact, chosen=None, alternatives=pool[:5],
                method=ResolutionMethod.ABSTAINED,
                rationale="every candidate is inactive/terminated on the date of service")
        pool = active
    survivors = _ranked(fact, pool, source, reconciliation)
    if not survivors:
        return ResolvedLine(
            fact=fact, chosen=None, alternatives=pool[:5],
            method=ResolutionMethod.ABSTAINED,
            rationale="every candidate contradicts a documented attribute")

    tag = f" ({authority})" if authority else ""

    # STEP 2 -- automatic selection ONLY when one candidate uniquely satisfies every
    # documented axis. `specificity` counts the documented axes a descriptor POSITIVELY
    # accounts for, so a unique maximum is exactly that condition.
    #
    # The recall MARGIN and the lexical SUPPORT score that used to decide this are
    # deliberately gone. Both are SIMILARITY, and the directive is explicit that
    # lexical/semantic/fuzzy similarity may only WIDEN the candidate pool -- it can
    # never verify a code, and must never deterministically close a tie between
    # clinically confusable candidates. Recall still ADMITS a candidate (the pool is a
    # retrieval product and the floor is where that pool came from); it no longer picks
    # between admitted ones.
    admitted = [m for m in survivors if m.recall >= _RELEVANCE_FLOOR]
    top = None
    if len(admitted) == 1:
        top = admitted[0]
    elif len(admitted) > 1:
        best = max(m.specificity for m in admitted)
        leaders = [m for m in admitted if m.specificity == best]
        if best > 0 and len(leaders) == 1:
            top = leaders[0]

    # STEPS 3 and 4 -- several candidates still satisfy every documented axis, so
    # re-inspect ONLY the axes that distinguish them against the ORIGINAL DOCUMENT and
    # release the one the page uniquely entails.
    tie = None
    semantic_note = ""
    if top is None and len(admitted) > 1:
        # issue #6 F9-R6 Phase 4: this deterministic path never calls a verifier, so
        # there are no requirement JUDGEMENTS to validate -- but the compiled
        # requirements (governed axes projected from the candidates' own descriptors
        # and, ICD-only, their instructional notes) are exactly the same axes
        # `discriminating_axes` already contributes, and narrowing against the
        # original page is a pure document lookup either way. Compiled fresh here
        # (never threaded in) since this path has no shortlist-wide compile step of
        # its own the way the LLM path does.
        from . import requirement as _requirement
        tie_requirements = _requirement.compile_requirements(
            [m.candidate for m in admitted], source)
        # issue #6, Codex's independent re-review (F9-R13-C): the same
        # additional selection condition `_settle_uniqueness` tries on the
        # judged path, tried here too before `tiebreak.narrow` -- never in
        # place of the interval/DOS controls immediately below, which still
        # run unconditionally against whichever candidate `top` ends up.
        semantic_winner = _select_by_semantic_axes(
            fact, [m.candidate for m in admitted], tie_requirements,
            reconciliation, source)
        if semantic_winner is not None:
            top = next(m for m in admitted if m.candidate.code == semantic_winner.code)
            semantic_note = (f"{semantic_winner.code}'s own action/qualifier "
                             f"requirements are the only ones fully supported by "
                             f"this fact's own reconciled evidence")
        else:
            # `semantic_action`/`semantic_qualifier` must never reach
            # `tiebreak.narrow`'s literal-text axis reporting either -- see
            # the matching comment in `_settle_uniqueness`.
            _narrow_requirements = tuple(r for r in tie_requirements
                                         if r.axis not in _SEMANTIC_AXES)
            tie = _tiebreak.narrow(fact, [m.candidate for m in admitted], reconciliation,
                                   _narrow_requirements)
            if tie.winner is not None:
                top = next(m for m in admitted if m.candidate.code == tie.winner.code)

    if top is not None and top.interval_unsupported:
        # Codex F4-R1: the leader's descriptor requires a bounded measurement its
        # documentation does not support (dimension-compatible + convertible + in
        # range). Checked for every way a leader can be reached -- unique survivor,
        # unique specificity, or narrowed against the page. A required constraint that
        # failed at step 1 may not be resurrected by a later step.
        return ResolvedLine(
            fact=fact, chosen=None, alternatives=[m.candidate for m in survivors[:5]],
            method=ResolutionMethod.ABSTAINED,
            documentation_gap=("the code's descriptor requires a measurement within a "
                "specific range and the documentation provides no compatible measurement "
                "of that dimension -- document the measurement or use a less-specific code"),
            rationale="bounded-interval code not supported by a dimension-compatible "
                      "documented measurement -- not billed deterministically")
    if top is not None:
        if semantic_note:
            why = f"{'; '.join(top.rationale)}; {semantic_note}"
        elif tie is not None:
            why = (f"{'; '.join(top.rationale)}; tie narrowed against the original "
                   f"document ({tie.proof}): {tie.detail}")
        else:
            why = "; ".join(top.rationale)
        return ResolvedLine(
            fact=fact, chosen=top.candidate,
            alternatives=[m.candidate for m in survivors if m is not top][:3],
            method=ResolutionMethod.DETERMINISTIC,
            tie_record=(tie.as_record() if tie is not None else None),
            rationale=f"{why}{tag}")

    # STEP 5 -- neither the documented axes nor the page single one out.
    if tie is None:
        return ResolvedLine(
            fact=fact, chosen=None, alternatives=[m.candidate for m in survivors[:5]],
            method=ResolutionMethod.ABSTAINED,
            rationale=f"no retrieved candidate clears the relevance floor for a "
                      f"deterministic decision{tag}")
    return ResolvedLine(
        fact=fact, chosen=None, alternatives=[m.candidate for m in survivors[:5]],
        method=ResolutionMethod.ABSTAINED,
        documentation_gap=(tie.provider_question or None),
        tie_record=tie.as_record(),
        rationale=(f"{len(admitted)} candidates satisfy every documented axis; "
                   f"{tie.detail}{tag}"))
