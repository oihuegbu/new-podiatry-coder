"""Semantic eligibility-before-retrieval (issue #6 items 4/5): narrow retrieval's
broad candidate pool to codes whose COMPILED SEMANTIC RECORD (`claude_coder.semantics`,
built entirely from authoritative data) positively CONFLICTS with what the fact's own
documentation states -- never a hardcoded code family, never a full-table scan, and
never a disqualification for an axis neither side documents. Absence of a constraint
is not a violation of one (the same principle `semantics.compiled_record` itself
follows): a candidate is excluded only when a compiled, deterministic field says
something that contradicts the fact, never on a lexical hunch.

Scope, honestly narrower than the plan's five-axis list, and deliberately so.
`action_concepts`/`anatomy_concepts` (derived from a CODE's own comma-structured
"Action, target" descriptor grammar, `ontology.parse_descriptor`) have no equally
reliable counterpart on the FACT side -- a fact's `description` is natural prose
extracted from the note, not a formal descriptor, so it essentially never carries
the same punctuation. Comparing the fact's whole-text vocabulary against a
candidate's action/anatomy tokens directly would be a bare LEXICAL overlap check,
and lexical overlap cannot tell a genuine mismatch from a clinical synonym
("exostosis" vs. "spur", "excision" vs. "resection") -- on a live billing pipeline,
a false exclusion there means a real service silently drops out of the candidate
pool.

ACTION now ships (issue #6 F9-R2), routed through the governed procedure-synonym
axis instead of raw prose: `_fact_named_different_action()` uses the SAME
round-trip-validated scan advisory recall uses (`data_access.concept_scan`/
`concept_lookup`) to find whether the fact's own text UNIQUELY names a DIFFERENT,
specific, verified procedure -- and if so, compares THAT verified match's own
compiled `action_concepts` against the CANDIDATE's, both derived identically from
authoritative CPT descriptor grammar (never fact prose against candidate tokens
directly). A candidate is excluded only when the two action vocabularies share
NOTHING. This still degrades honestly to "nothing to check" whenever the fact's
text does not uniquely match anything in the synonym table -- which is common; it
narrows real cases, it does not replace retrieval or tie-break.

ANATOMY now ships too (issue #6 F9-R2, second pass), but as POSITIVE DOMINANCE, not
inferred disjointness -- the governed concept-relation index
(`AuthoritativeSource.concept_relation`) only ever returns SAME/RELATED/UNRESOLVED
for two terms, never a DISJOINT verdict (issue #6 F7-R3-C3: "an IS_A hierarchy has
no basis to assert opposition"), so there is still no non-lexical signal that two
anatomy phrases are *different* structures. There does not need to be one:
`_anatomy_dominance_exclusions()` (a GROUP-level check, run once per candidate pool
by `eligible_partition`/`eligibility_report`, never by the single-candidate
`eligible()`) removes a candidate whose anatomy compatibility is UNKNOWN only when a
SIBLING candidate in the SAME pool is positively grounded (SAME or a governed
ancestor/descendant relation) to the fact's own documented anatomy. This never
claims the removed candidate's anatomy IS wrong -- only that a better-grounded
alternative already exists, which needs no disjoint verdict at all. When every
candidate is UNKNOWN, nothing is removed on this basis: absence of grounding is not
evidence against anyone. A CLOSED axis (laterality: left/right/bilateral) still
supports a real, structural CONTRADICTED_EXPLICIT verdict when both sides state one
and they disagree -- that comparison was always safe; it was never a concept-graph
guess.

Measurement/interval requirement, semantic-class conflict, and code activity on the
date of service round out what ships deterministically at the single-candidate
level.

Scope note on grouping (issue #6 item 5, Codex F8-R2): this module has always
accepted whatever fact list its caller supplies -- it never assumed exactly one.
`resolution.resolve()` now supplies every fact `composition.service_intents`
grouped with the one under retrieval (`eligibility.RetrievalRequest.intent_facts`),
so this filters against what the whole documented SERVICE states, not one isolated
fact considered alone, whenever that fact belongs to a multi-member intent.

Scope note on candidate paths (Codex F8-R2): every candidate path is filtered --
the broad RECALL pool AND the authoritative-index `seeds` in `resolution.py` alike
-- and `eligible_partition` is a PURE filter with no fallback that restores an
excluded candidate; see its own docstring for why an earlier version's fallback was
itself a defect, not a safety net.
"""
from __future__ import annotations

from . import ontology as _ontology
from . import semantics as _semantics
from .models import ClinicalFact, FactKind, Outcome

#: A candidate positively classified as evaluation/management is a different kind of
#: service from a candidate that is not, and vice versa -- the one FactKind/semantic-
#: class correspondence narrow and certain enough to check without guessing (issue #6
#: item 4). Other classes (`surgical_procedure`, `anesthesia`, ...) do not have an
#: equally reliable FactKind correspondence -- FactKind.PROCEDURE alone does not
#: distinguish a surgical from a non-surgical procedure -- so they are deliberately
#: left unchecked here rather than approximated.
_FACT_KIND_SEMANTIC_CLASS = {FactKind.EM: "evaluation_management"}


def _candidate_measurement_dimension(candidate, source) -> str | None:
    """The physical DIMENSION (area/length/mass) the candidate's own descriptor's
    bounded interval is stated in, or None when it cannot be determined (no unit
    named, or the descriptor carries no bounded interval at all). Read directly
    from the candidate's own descriptor -- `compiled_record` computes the same
    interval internally but does not expose the dimension itself, so this reuses
    `ontology.parse_descriptor`/`measurement.unit_dimension` rather than a second
    parser."""
    from . import measurement as _measurement
    rec = getattr(source, "lookup", None)
    rec = rec(candidate.code, candidate.system) if callable(rec) else None
    descriptor = str((rec or {}).get("long_description")
                     or (rec or {}).get("description")
                     or (rec or {}).get("short_description") or "")
    feats = _ontology.parse_descriptor(descriptor)
    if not (feats.interval and feats.interval.bounded()):
        return None
    dimension, _ = _measurement.unit_dimension(feats.interval.unit)
    return dimension


def _has_documented_measurement(facts: list[ClinicalFact], dimension: str | None) -> bool:
    """Whether a documented measurement exists for this intent that could satisfy
    a candidate requiring `dimension` -- either in a fact's own prose description
    (a bounded interval `ontology.parse_descriptor` detects, e.g. "16 sq cm or
    less") OR in its structured attributes (e.g. `size_sqcm`, `depth_mm` --
    `measurement.measurements_of`, the SAME typed extractor
    `resolution._decide`/`tiebreak` already use).

    Scoped to `dimension` when it's known (Codex F8-R2: an earlier version
    accepted ANY measurement anywhere in the intent for ANY candidate requiring
    one, so an unrelated AREA measurement on one component could satisfy a
    candidate requiring LENGTH on a different one) -- an attribute measurement
    only counts when its own dimension matches; a description-detected interval
    has no independently-typed dimension to check, so it counts on its own terms
    exactly as it always could (this module never invented dimension-tagging for
    prose text, only for the already-typed attribute path). `dimension is None`
    (the candidate's own requirement could not be determined) falls back to the
    original, honestly looser "any measurement" check -- absence of information
    is never a reason to invent a MORE specific requirement than the candidate's
    own descriptor actually states.

    A prior version of this check read ONLY the prose description, so a fact
    whose measurement was extracted into a structured attribute (the common,
    correctly-extracted case) always looked unmeasured here -- a real gap this
    eligibility check's own review pass exposed once `eligible_partition`
    stopped silently restoring an all-excluded pool (Codex F8-R2, round 1): the
    restore-all fallback had been masking this defect, not compensating for a
    deliberately narrow check."""
    from . import measurement as _measurement
    for f in facts:
        feats = _ontology.parse_descriptor(f.description or "")
        if feats.interval and feats.interval.bounded():
            return True
        for m in _measurement.measurements_of(f.attributes or {}):
            if dimension is None or m.dimension == dimension:
                return True
    return False


def _fact_named_different_action(facts: list[ClinicalFact], candidate,
                                 source) -> tuple[str, set[str]] | None:
    """`(matched_code, action_concepts)` for a DIFFERENT, specific procedure the
    facts' own text UNIQUELY names, via the same round-trip-validated synonym scan
    advisory recall uses (`concept_scan`/`concept_lookup`) -- or None when nothing
    scans, the scan names only THIS candidate, or the match's own action vocabulary
    could not be compiled. The comparison this feeds is between two CANDIDATES' own
    compiled `action_concepts`, both derived identically from authoritative CPT
    descriptor grammar (`ontology.parse_descriptor`) -- never fact prose compared
    lexically against a candidate's tokens (issue #6 F9-R2)."""
    scan = getattr(source, "concept_scan", None)
    lookup = getattr(source, "concept_lookup", None)
    if not callable(scan) or not callable(lookup):
        return None
    for f in facts:
        if f.system not in ("cpt", "hcpcs"):
            continue
        texts = [f.description] + [s.text for s in (f.evidence or [])[:1]]
        for text in texts:
            text = str(text or "").strip()
            if not text:
                continue
            try:
                matched_terms = scan("procedure", text)
            except Exception:
                continue
            for term in matched_terms:
                try:
                    result = lookup("procedure", term)
                except Exception:
                    continue
                codes = result.get("candidates") or []
                if not result.get("unique") or len(codes) != 1:
                    continue
                matched_code = codes[0]
                if matched_code == candidate.code:
                    continue          # names THIS candidate -- not a conflict signal
                matched_record = _semantics.compiled_record(matched_code, "cpt", source)
                action_concepts = set((matched_record or {}).get("action_concepts") or [])
                if action_concepts:
                    return matched_code, action_concepts
    return None


#: Verdicts for `_anatomy_compatibility` -- issue #6 F9-R2, second pass. Named to
#: match the product owner's own vocabulary rather than reusing `coreference`'s
#: SAME_EVENT/DISTINCT_EVENT/UNDETERMINED, which are about EVENT identity, not
#: candidate/fact axis compatibility -- a different question with a different owner.
_SUPPORTED_EXACT = "SUPPORTED_EXACT"
_SUPPORTED_HIERARCHICAL = "SUPPORTED_HIERARCHICAL"
_UNKNOWN = "UNKNOWN"
_CONTRADICTED_EXPLICIT = "CONTRADICTED_EXPLICIT"


def _fact_attribute_value(facts: list[ClinicalFact], key: str) -> str:
    """The first non-empty `key` attribute across `facts` -- the same field
    `graph_consensus`/`coreference` already read as that axis's documented value.
    Empty when nothing states one."""
    for f in facts:
        val = str((f.attributes or {}).get(key) or "").strip()
        if val:
            return val
    return ""


def _candidate_descriptor_features(candidate, source):
    rec = getattr(source, "lookup", None)
    rec = rec(candidate.code, candidate.system) if callable(rec) else None
    descriptor = str((rec or {}).get("long_description") or (rec or {}).get("description")
                     or (rec or {}).get("short_description") or "")
    return _ontology.parse_descriptor(descriptor)


def _anatomy_compatibility(candidate, facts: list[ClinicalFact], source) -> str:
    """How this ONE candidate's anatomy relates to what `facts` document -- never a
    comparison between candidates; that comparative step is
    `_anatomy_dominance_exclusions`, below.

    CONTRADICTED_EXPLICIT only from the CLOSED laterality vocabulary
    (left/right/bilateral, `ontology._LATERALITY`) when both sides state one and they
    disagree -- a structural, never a lexical, comparison. SUPPORTED_EXACT/
    SUPPORTED_HIERARCHICAL come only from the governed concept-relation index
    (SAME / ancestor-descendant); anything else -- nothing documented, no candidate
    anatomy phrase to compare, no concept-relation capability, or the index itself
    UNRESOLVED -- is honestly UNKNOWN, never a guess in either direction."""
    feats = _candidate_descriptor_features(candidate, source)

    fact_laterality = _fact_attribute_value(facts, "laterality").lower()
    if fact_laterality and feats.laterality and fact_laterality not in feats.laterality:
        return _CONTRADICTED_EXPLICIT

    fact_anatomy = _fact_attribute_value(facts, "anatomy")
    if not fact_anatomy or not feats.anatomy_phrase:
        return _UNKNOWN
    relate = getattr(source, "concept_relation", None)
    if not callable(relate):
        return _UNKNOWN
    try:
        verdict = relate(fact_anatomy, feats.anatomy_phrase)
    except Exception:
        return _UNKNOWN
    from . import terminology as _term
    if verdict == _term.CONCEPT_SAME:
        return _SUPPORTED_EXACT
    if verdict == _term.CONCEPT_RELATED:
        return _SUPPORTED_HIERARCHICAL
    return _UNKNOWN


def _anatomy_dominance_exclusions(facts: list[ClinicalFact], candidates: list,
                                  source) -> dict[tuple[str, str], str]:
    """`{(code, system) -> reason}` for candidates the anatomy-dominance rule removes
    from THIS pool (issue #6 F9-R2, second pass): an explicit laterality
    contradiction always excludes; separately, a candidate whose anatomy
    compatibility is UNKNOWN is excluded only when at least one OTHER, non-
    contradicted candidate in the SAME pool is positively grounded (SUPPORTED_EXACT
    or SUPPORTED_HIERARCHICAL) to the facts' own documented anatomy. Comparative, not
    absolute -- this never asserts the excluded candidate's anatomy IS wrong, only
    that a better-grounded sibling already exists in this specific pool. When every
    candidate is UNKNOWN, nothing is removed on that basis: absence of grounding is
    not evidence against anyone. A pool of fewer than two candidates has nothing to
    compare, so nothing is excluded."""
    if len(candidates) < 2:
        return {}
    verdicts = {(c.code, c.system): _anatomy_compatibility(c, facts, source)
               for c in candidates}
    out: dict[tuple[str, str], str] = {}
    for c in candidates:
        key = (c.code, c.system)
        if verdicts[key] == _CONTRADICTED_EXPLICIT:
            out[key] = ("candidate's stated laterality contradicts the fact's own "
                       "documented laterality")
    supported = any(verdicts[(c.code, c.system)] in (_SUPPORTED_EXACT, _SUPPORTED_HIERARCHICAL)
                    for c in candidates if (c.code, c.system) not in out)
    if supported:
        for c in candidates:
            key = (c.code, c.system)
            if key in out:
                continue
            if verdicts[key] == _UNKNOWN:
                out[key] = ("another candidate in the same pool is positively "
                           "grounded (concept-graph SAME or ancestor/descendant) to "
                           "the fact's documented anatomy, while this one is not")
    return out


def _ineligibility_reason(candidate, facts: list[ClinicalFact], source,
                          date_of_service: str | None) -> str | None:
    """None when eligible; otherwise the one compiled-record axis that positively
    conflicted. The single place this module's actual decision logic lives --
    `eligible()` and `eligibility_report()` both read it, so the audit trail's
    stated reason can never drift from the reason a candidate was actually kept
    or dropped."""
    record = _semantics.compiled_record(candidate.code, candidate.system, source)
    if record is None:
        return None

    if "measurement" in (record.get("required_attributes") or []):
        dimension = _candidate_measurement_dimension(candidate, source)
        if not _has_documented_measurement(facts, dimension):
            return ("candidate's descriptor requires a documented measurement/interval "
                   "the fact's text or attributes do not state" +
                   (f" (dimension: {dimension})" if dimension else ""))

    fact_kinds = {f.kind for f in facts}
    expected_classes = {_FACT_KIND_SEMANTIC_CLASS[k] for k in fact_kinds
                        if k in _FACT_KIND_SEMANTIC_CLASS}
    candidate_class = record.get("semantic_class")
    if expected_classes and candidate_class and candidate_class not in expected_classes:
        return (f"candidate is classified {candidate_class!r}, incompatible with the "
               f"documented fact kind's expected class {sorted(expected_classes)}")

    active = getattr(source, "active_on", None)
    if callable(active) and date_of_service:
        try:
            status = active(candidate.code, candidate.system, date_of_service)
        except Exception:
            status = None
        if status is Outcome.BLOCKED:
            return "candidate is not active on the encounter's date of service"

    if candidate.system in ("cpt", "hcpcs"):
        candidate_actions = set(record.get("action_concepts") or [])
        if candidate_actions:
            matched = _fact_named_different_action(facts, candidate, source)
            if matched:
                matched_code, matched_actions = matched
                if candidate_actions.isdisjoint(matched_actions):
                    return (f"the fact's text uniquely names a different verified "
                           f"procedure ({matched_code}), whose action vocabulary "
                           f"{sorted(matched_actions)} shares nothing with this "
                           f"candidate's {sorted(candidate_actions)}")

    return None


def eligible(candidate, facts: list[ClinicalFact], source,
            date_of_service: str | None) -> bool:
    """Whether ONE already-retrieved candidate is semantically eligible for what
    `facts` document. Defaults to True (eligible) whenever there is nothing
    compiled to check against, or nothing compiled conflicts -- never a reason to
    exclude a candidate the vector search already found relevant."""
    return _ineligibility_reason(candidate, facts, source, date_of_service) is None


def eligible_partition(facts: list[ClinicalFact], candidates: list, source,
                       date_of_service: str | None) -> list:
    """`candidates`, narrowed to the ones `eligible()` accepts for `facts`, then
    narrowed once more by `_anatomy_dominance_exclusions` (issue #6 F9-R2, second
    pass) -- a GROUP-level pass over the survivors, run here because this is the one
    place that sees the whole pool together; the single-candidate `eligible()` has no
    visibility into siblings and cannot make this comparison.

    A PURE, monotonic filter -- never restores an excluded candidate for any
    reason. Codex F8-R2: an earlier version of this function fell back to the
    unfiltered list whenever every candidate was excluded, reasoning that an
    all-excluded result was more likely a sign the filter didn't fit this case
    than evidence nothing retrieved was usable. That reasoning was wrong twice
    over: it silently let a structurally-incompatible candidate reach the
    resolver exactly when eligibility had the clearest possible signal against
    every one of them, AND it made `eligibility_report` (which never applied
    the same fallback) claim a candidate was excluded when it was actually
    still processed -- an audit/enforcement mismatch, not merely an
    over-cautious safety net. The caller (`resolution.resolve`) already treats
    an empty pool as an honest abstention; that is the correct outcome here
    too, not a reason to disable the filter."""
    pool = [c for c in candidates if eligible(c, facts, source, date_of_service)]
    excluded = _anatomy_dominance_exclusions(facts, pool, source)
    return [c for c in pool if (c.code, c.system) not in excluded]


def eligibility_report(facts: list[ClinicalFact], candidates: list, source,
                       date_of_service: str | None) -> list[dict]:
    """A full per-candidate audit record over `candidates` -- which `eligible_partition`
    would keep or exclude, and why -- preserved for the audit trail EVEN on a
    held/blocked outcome (issue #6 item 8), never only for a released line. Computed
    over the SAME `_ineligibility_reason` AND the same `_anatomy_dominance_exclusions`
    (run over the identical eligible-survivor pool) `eligible_partition` itself reads,
    so this record can never claim a different reason than the one that actually
    decided it."""
    pool = [c for c in candidates if eligible(c, facts, source, date_of_service)]
    dominance = _anatomy_dominance_exclusions(facts, pool, source)
    report = []
    for c in candidates:
        reason = _ineligibility_reason(c, facts, source, date_of_service)
        if reason is None:
            reason = dominance.get((c.code, c.system))
        report.append({"code": c.code, "system": c.system, "eligible": reason is None,
                       "reason": reason})
    return report
