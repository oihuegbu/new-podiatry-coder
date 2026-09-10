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

ACTION stays unimplemented, and an attempt to ship it (issue #6 F9-R2, second pass)
was REVERTED after Codex's re-review (F9-R2-A): it routed the advisory
procedure-synonym scan (`data_access.concept_scan`/`concept_lookup`,
`cpt_verified_synonyms.json`) into an ELIGIBILITY EXCLUSION -- excluding a candidate
whenever the fact's text uniquely named a different verified procedure with a
disjoint action vocabulary. That directly violated the advisory tier's own,
already-reviewed trust-tier contract (`resolution._advisory_procedure_expansions`'s
own docstring, `ResolvedLine.advisory_terminology`'s own contract): LLM-generated,
retrieval-consistency-validated terms may WIDEN RECALL ONLY -- never settle
identity, exclude a candidate, or authorize release. A weaker-tier source deciding
eligibility is exactly the thing that contract exists to forbid, regardless of how
narrowly it was scoped. `concept_scan`/`concept_lookup` remain wired into
`resolution._advisory_procedure_expansions` for recall-widening only, unchanged.

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

`_fact_anatomy_phrases()` (issue #6 F9-R2-C) widens what gets COMPARED, not what
counts as a match: the live note showed this check going UNKNOWN almost everywhere
because it only ever tried one raw, possibly-composite `fact.attributes["anatomy"]`
string whole against one candidate phrase. It now also (a) decomposes a composite
mention on generic list/conjunction punctuation ("structure A and structure B"
names TWO structures a single comparison could never resolve) and (b) tries every
governed synonym `coreference.normalize_fact_terminology` already resolved for it
(`fact.governed_terms["anatomy"]`) -- a stable, concept-normalized phrasing, not
prose re-parsed here a second time. Still purely structural: no clinical term is
named or enumerated by this module.

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

SERVICE ROLE (issue #6 F9-R11-H) narrows one specific case the "ACTION stays
unimplemented" note above does NOT cover, and does so without repeating that
attempt's defect: FactKind.PROCEDURE alone cannot distinguish the operative act
from the anesthesia administered FOR it, so both land in the same undifferentiated
recall pool on a real note (confirmed live: an anesthesia-block CPT family showing
up inside a surgical procedure's candidate shortlist). Unlike the reverted ACTION
attempt, this does not route an ADVISORY source into eligibility -- the fact side
is a real, evidence-backed EXTRACTION AXIS (`service_role`, the same
attributes/attribute_evidence mechanism every other axis already uses, so it is
independently verifiable against the note like laterality or anatomy), and the
candidate side reads ONLY `semantic_class()` -- the same already-authoritative,
already-reviewed structural classifier the compiled-semantics module already
trusts. `umls_crosswalk_entry` is deliberately NOT used for this (its own
docstring: "must NEVER be used to select or expand a code" -- a real, separate
follow-up item, not folded in here under time pressure). An earlier revision of
this module also fell back to `ontology.code_section`'s literal descriptor-phrase
table when `semantic_class()` could not yet classify "anesthesia" -- Codex
F9-R11-H-B correctly identified that as the SAME hardcoded-proxy pattern
CLAUDE.md forbids for a claim-affecting decision, notwithstanding that it read an
authoritative descriptor's own text: the MATCHING RULE itself was a hand-authored,
unversioned Python literal, not sourced from `coding_semantics.json`. Removed;
`semantic_class()`'s own `anesthesia` rule (`pfs_status_any`) is now wired to a
real CMS PFS status-indicator column instead (`data_access.
AuthoritativeSource.pfs_status`, backed by `ComplianceDataStore.billing_status`
since issue #6 F9-R11-H-C, second re-review).

Not implemented here, and deliberately left for separate, more careful design:
Codex's broader "positive action/anatomy support from UMLS CUI lineage" proposal
(F9-R11-H items 3-4) for candidates OTHER than the procedure/anesthesia split --
`umls_crosswalk_entry`'s docstring warns three times over against exactly this
class of use, and the ACTION revert above is a direct precedent for what goes
wrong when that boundary is crossed under time pressure rather than reviewed
carefully on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from . import graph_consensus as _gc
from . import ontology as _ontology
from . import semantics as _semantics
from .models import ClinicalFact, FactKind, Outcome

#: A candidate positively classified as evaluation/management is a different kind of
#: service from a candidate that is not, and vice versa -- the one FactKind/semantic-
#: class correspondence narrow and certain enough to check without guessing (issue #6
#: item 4).
_FACT_KIND_SEMANTIC_CLASS = {FactKind.EM: "evaluation_management"}

#: Closed FACT-side vocabulary for the ONE FactKind.PROCEDURE sub-distinction that
#: matters for eligibility and has no other structural signal (Codex F9-R11-H):
#: FactKind.PROCEDURE alone does not distinguish the operative/procedural act from
#: the anesthesia administered FOR it -- both are extracted as "procedure" facts,
#: so without this the anesthesia candidate pool and the operative candidate pool
#: are the same undifferentiated recall set. The extractor now documents this via
#: the SAME evidence-backed "attributes"/"attribute_evidence" mechanism as every
#: other axis (`extraction._SYSTEM`) -- no wire-schema change, since attribute
#: NAMES were already open-vocabulary. A fact that never states it (the overwhelming
#: majority of non-procedure facts, and any procedure fact the note does not make
#: this distinction for) stays UNKNOWN and is never guessed either way.
_PROCEDURE_ROLES = ("operative", "anesthesia")

#: CANDIDATE-side procedure role, from already-loaded authoritative structure only
#: -- never a code range/prefix, and never a hand-authored descriptor PROXY either
#: (Codex F9-R11-H-B): an earlier version of this mapping fell back to
#: `ontology.code_section`'s literal `("anesthesia for", "anesthesia,")` phrase
#: table when `semantic_class()` could not classify "anesthesia" -- CLAUDE.md's
#: no-hardcoding rule forbids exactly this pattern for a claim-affecting decision
#: (approximating a real, queryable field with a fixed proxy), even though the
#: proxy read an authoritative descriptor's own text. `semantic_class()`'s
#: `surgical_procedure` rule (`global_days_kind: numeric`, a real CMS PFS field)
#: covers "operative"; its `anesthesia` rule (`pfs_status_any: ["J"]`) now reads
#: the CMS PFS STATUS CODE column via `data_access.AuthoritativeSource.pfs_status`
#: (backed by `ComplianceDataStore.billing_status`, DOS-aware since issue #6
#: F9-R11-H-C, second re-review) -- ONLY
#: `semantic_class()` is consulted here. A candidate this classifier cannot
#: resolve to a role is an honest data gap, never guessed from its descriptor
#: text. This mapping names no medical code -- it maps an already-authoritative
#: CLASSIFICATION NAME onto this module's own closed role vocabulary, the same
#: pattern `_FACT_KIND_SEMANTIC_CLASS` above already uses.
_SEMANTIC_CLASS_TO_PROCEDURE_ROLE = {"surgical_procedure": "operative",
                                     "anesthesia": "anesthesia"}


def _candidate_procedure_role(candidate, source, *, dos: str | None = None) -> str | None:
    """This candidate's procedure role (`_PROCEDURE_ROLES`), or None when
    `semantic_class()` runs and returns an honest "unclassified" -- never a
    guess and never a descriptor-phrase proxy (Codex F9-R11-H-B).

    Does NOT catch a real authority failure into None (Codex F9-R11-H-D,
    second re-review): a prior version caught every exception `semantic_class`
    raised -- including `SemanticClassUnavailable`, a genuine "the
    authoritative classifier could not be read" failure -- and reported it
    identically to "read the classifier fine, this code just isn't
    classified", silently WEAKENING a claim-affecting control exactly when
    its own authority was broken. `source` missing the method entirely is
    the one case treated the same way: no `semantic_class` support is itself
    an authority failure, not an honest per-code "unclassified" answer.
    A real failure propagates to the pipeline's existing `retrieval_execution`
    system-hold boundary (`resolution.resolve`'s caller already wraps every
    resolution call in a catch-all that converts any raised exception into a
    typed hold) -- this module raises, it does not catch.

    Never even asks for a non-CPT/HCPCS candidate (issue #6 F9-R11-H-D, second
    re-review): the operative/anesthesia axis is a PROCEDURE-code distinction
    -- an ICD-10 diagnosis candidate has no such role, honestly None, and
    `semantic_class()`'s OWN rule branches already gate every procedure-role-
    relevant rule the same way. `_service_role_control` now classifies every
    candidate even for a MIXED_KIND_INTENT (a composed intent's candidate pool
    can be for a non-procedure member fact), so without this guard it would
    call the procedure-role classifier on diagnosis candidates for no reason
    other than audit-trail completeness -- needlessly exercising an unrelated
    ICD-chapter rule branch this axis has no business touching."""
    if candidate.system not in ("cpt", "hcpcs"):
        return None
    classifier = getattr(source, "semantic_class", None)
    if not callable(classifier):
        from .data_access import SemanticClassUnavailable
        raise SemanticClassUnavailable(
            "source does not implement semantic_class -- the service-role "
            "authority is unavailable")
    cls = classifier(candidate.code, candidate.system, dos=dos)
    return _SEMANTIC_CLASS_TO_PROCEDURE_ROLE.get(cls)


def _authorized_roles(facts: list[ClinicalFact], reconciliation) -> set[str]:
    """The set of DISTINCT claim-authorized `service_role` values documented
    anywhere across `facts` (Codex F9-R11-H-A). NEVER "first nonempty": a
    composed multi-fact intent (e.g. an operative component and an anesthesia
    component grouped by `PART_OF`) can genuinely carry more than one role, and
    picking whichever fact happened to sort first made the excluded candidate
    family depend on fact ORDER, not on what was documented -- the same
    intent produced opposite exclusions depending on list order, independently
    reproduced. Returns every distinct value stated, so the caller can tell
    "exactly one, consistently documented" apart from "conflicting" or "none"."""
    return {v for v in (str(_gc.claim_authorized_value(f, "service_role", reconciliation)
                            or "").strip().lower()
                        for f in facts) if v}


class RoleControlStatus(str, Enum):
    """Why the service-role control did or did not exclude ONE candidate
    (Codex F9-R11-H-D). Before this, `_service_role_exclusions` returned a bare
    `{}` for five materially different situations -- no documented fact role,
    conflicting fact roles, a mixed-kind intent, an unclassifiable candidate,
    and "ran, found nothing incompatible" -- so the audit trail could not
    distinguish "this control never had a chance to run" from "it ran and
    found this candidate compatible", which is exactly why a real-note
    discrepancy (an operative-classified candidate surviving in an
    anesthesia-role fact's pool) could not be traced to a cause. Every
    candidate now gets ONE of these, always, whether or not it survives."""
    EXCLUDED = "excluded"                        #: ran; this candidate's role conflicted
    COMPATIBLE = "compatible"                     #: ran; this candidate's role matched
    CANDIDATE_UNCLASSIFIED = "candidate_unclassified"  #: ran; this candidate's own role
                                                        #: could not be determined
    FACT_ROLE_MISSING = "fact_role_missing"       #: did not run; no fact documented a role
    FACT_ROLE_CONFLICT = "fact_role_conflict"     #: did not run; facts disagree on role
    MIXED_KIND_INTENT = "mixed_kind_intent"       #: did not run; not every fact is a PROCEDURE


@dataclass(frozen=True)
class RoleControlDecision:
    """One candidate's service-role control outcome, always populated -- never
    a silent skip (Codex F9-R11-H-D). `fact_roles` is the full set
    `_authorized_roles` found (empty when none, >1 element when conflicting,
    exactly one element when the control actually ran); `candidate_role` is
    this candidate's own classification, or None when unclassifiable.

    `blocks_line` (Codex F9-R11-H-D, second re-review): True only for a
    FACT_ROLE_CONFLICT/MIXED_KIND_INTENT decision where the candidates in
    THIS pool actually classify into more than one distinct procedure role --
    a genuine, observable ambiguity that could change which candidate family
    wins, as opposed to a conflict/mixed state whose candidates all agree (or
    none classify), where the ambiguity is moot for this specific pool. This
    field names the condition; it is not yet wired into an upstream
    hold/split -- the report is honest that the condition exists, without
    claiming a line was actually stopped from releasing on it."""
    status: RoleControlStatus
    fact_roles: tuple[str, ...]
    candidate_role: str | None
    excluded: bool
    blocks_line: bool = False
    authority_source_id: str = "semantic_class"
    authority_version: str | None = None


def _service_role_control(facts: list[ClinicalFact], candidates: list,
                          source, reconciliation=None, dos: str | None = None
                          ) -> dict[tuple[str, str], RoleControlDecision]:
    """`{(code, system) -> RoleControlDecision}` for EVERY candidate, always --
    the typed replacement for the old bare-`{}` `_service_role_exclusions`
    (Codex F9-R11-H-D). Runs BEFORE verification (via `eligible_partition`,
    below) so an incompatible-role candidate never reaches the model at all.

    The control only EXCLUDES candidates when EVERY fact in this intent is a
    FactKind.PROCEDURE (a mixed intent has no single procedure role to check
    against -- `MIXED_KIND_INTENT`) AND the composed intent documents EXACTLY
    ONE distinct role across every member fact (`_authorized_roles`; zero is
    `FACT_ROLE_MISSING`, more than one is `FACT_ROLE_CONFLICT` -- Codex
    F9-R11-H-A: a genuine role conflict across a composed intent needs
    upstream composition/provider-query resolution, not an arbitrary pick
    between two documented alternatives). A candidate is excluded only when
    its OWN role is positively classified AND differs from the fact's; an
    unclassifiable candidate role is kept (`CANDIDATE_UNCLASSIFIED`) --
    absence of grounding is not evidence against anyone, the same principle
    `_anatomy_dominance_exclusions` already follows.

    Every candidate's role is classified regardless of which of the above
    applies (Codex F9-R11-H-D, second re-review) -- a prior version skipped
    classification entirely for FACT_ROLE_MISSING/CONFLICT/MIXED_KIND_INTENT,
    so the audit trail could show a real-note survivor's role controversy was
    never resolved but not WHAT that candidate's own classification actually
    was, which is exactly the fact a reviewer needs to judge whether the
    ambiguity is real."""
    mixed_kind = {f.kind for f in facts} != {FactKind.PROCEDURE}
    roles = () if mixed_kind else tuple(sorted(_authorized_roles(facts, reconciliation)))
    if mixed_kind:
        base_status = RoleControlStatus.MIXED_KIND_INTENT
    elif not roles:
        base_status = RoleControlStatus.FACT_ROLE_MISSING
    elif len(roles) > 1:
        base_status = RoleControlStatus.FACT_ROLE_CONFLICT
    elif roles[0] not in _PROCEDURE_ROLES:
        base_status = RoleControlStatus.FACT_ROLE_MISSING
    else:
        base_status = None                    # the control actually runs, below

    if base_status is not None:
        classified = {(c.code, c.system): _candidate_procedure_role(c, source, dos=dos)
                     for c in candidates}
        distinct_roles = {r for r in classified.values() if r is not None}
        blocks = (base_status in (RoleControlStatus.FACT_ROLE_CONFLICT,
                                  RoleControlStatus.MIXED_KIND_INTENT)
                 and len(distinct_roles) > 1)
        return {key: RoleControlDecision(base_status, roles, role, False, blocks)
               for key, role in classified.items()}

    fact_role = roles[0]
    out: dict[tuple[str, str], RoleControlDecision] = {}
    for c in candidates:
        role = _candidate_procedure_role(c, source, dos=dos)
        if role is None:
            status, excluded = RoleControlStatus.CANDIDATE_UNCLASSIFIED, False
        elif role != fact_role:
            status, excluded = RoleControlStatus.EXCLUDED, True
        else:
            status, excluded = RoleControlStatus.COMPATIBLE, False
        out[(c.code, c.system)] = RoleControlDecision(status, (fact_role,), role, excluded)
    return out


def _service_role_exclusions(facts: list[ClinicalFact], candidates: list,
                             source, reconciliation=None,
                             dos: str | None = None) -> dict[tuple[str, str], str]:
    """`{(code, system) -> reason}` -- the exclusion-reason VIEW of
    `_service_role_control`, for callers that only need the filter, not the
    full audit decision."""
    control = _service_role_control(facts, candidates, source, reconciliation, dos)
    return {key: (f"candidate's authoritative classification is "
                 f"{decision.candidate_role!r}, incompatible with the fact's "
                 f"documented service_role {decision.fact_roles[0]!r}")
           for key, decision in control.items() if decision.excluded}


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


#: Verdicts for `_anatomy_compatibility` -- issue #6 F9-R2, second pass. Named to
#: match the product owner's own vocabulary rather than reusing `coreference`'s
#: SAME_EVENT/DISTINCT_EVENT/UNDETERMINED, which are about EVENT identity, not
#: candidate/fact axis compatibility -- a different question with a different owner.
_SUPPORTED_EXACT = "SUPPORTED_EXACT"
_SUPPORTED_HIERARCHICAL = "SUPPORTED_HIERARCHICAL"
_UNKNOWN = "UNKNOWN"
_CONTRADICTED_EXPLICIT = "CONTRADICTED_EXPLICIT"


def _fact_attribute_value(facts: list[ClinicalFact], key: str, reconciliation=None) -> str:
    """The first CLAIM-AUTHORIZED `key` attribute across `facts` (issue #6
    F9-R6-R2, sixth re-review) -- NEVER the raw attribute directly. Proven
    exploitable: a `NEGATED` laterality still eliminated candidates here via the
    raw value, before `resolution._evaluate` even runs. Empty when nothing
    states one, OR when the only value present is not genuinely, provably
    asserted."""
    for f in facts:
        val = str(_gc.claim_authorized_value(f, key, reconciliation) or "").strip()
        if val:
            return val
    return ""


#: Generic list/conjunction punctuation -- structural, not clinical vocabulary.
#: Slash included (issue #6 F9-R2-C, third pass: real extraction output commonly
#: uses "structure alpha / structure beta" for a composite mention).
_LIST_SPLIT = re.compile(r"\s*(?:,|;|/|\band\b|&)\s*", re.IGNORECASE)


def _fact_anatomy_phrases(facts: list[ClinicalFact]) -> tuple[str, ...]:
    """Every distinct anatomy phrase `facts` document (issue #6 F9-R2-C) -- the raw
    `anatomy` attribute, DECOMPOSED on generic list/conjunction punctuation (a
    composite mention like "structure A and structure B" names TWO structures, not
    one a single `concept_relation` call could ever resolve), plus every governed
    synonym already resolved for it (`fact.governed_terms["anatomy"]`,
    `coreference.normalize_fact_terminology`'s own output -- a stable,
    concept-normalized alternate phrasing, not raw prose re-parsed here). Purely
    structural: no clinical term is named or enumerated, only generic punctuation."""
    phrases: list[str] = []
    seen: set[str] = set()
    for f in facts:
        raw = str((f.attributes or {}).get("anatomy") or "").strip()
        parts = [p.strip() for p in _LIST_SPLIT.split(raw) if p.strip()] if raw else []
        governed = tuple(str(t).strip() for t in
                         (getattr(f, "governed_terms", None) or {}).get("anatomy", ())
                         if str(t).strip())
        for phrase in (*parts, *governed):
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                phrases.append(phrase)
    return tuple(phrases)


def _candidate_descriptor_features(candidate, source):
    rec = getattr(source, "lookup", None)
    rec = rec(candidate.code, candidate.system) if callable(rec) else None
    descriptor = str((rec or {}).get("long_description") or (rec or {}).get("description")
                     or (rec or {}).get("short_description") or "")
    return _ontology.parse_descriptor(descriptor)


#: A fixed English idiom, not clinical vocabulary -- "structure alpha, with or
#: without graft" qualifies ONE target; it does not name "graft" as a second one.
#: Issue #6 F9-R2-C, fourth pass: an earlier version only deleted the idiom's own
#: words and still split the surrounding comma, which promoted the qualifier text
#: itself ("graft") into a false second target. This TRUNCATES the phrase at the
#: idiom instead -- everything from "with or without" onward (and the list
#: separator immediately before it, if any) is a qualifier CLAUSE, dropped
#: wholesale, never treated as a candidate anatomical target.
_WITH_OR_WITHOUT = re.compile(r"[,;]?\s*\bwith\s+or\s+without\b.*$", re.IGNORECASE)
#: Generic alternation punctuation for a CANDIDATE's own target list -- "structure
#: alpha or structure beta" names two alternative targets a single comparison could
#: never resolve. Deliberately narrower than `_LIST_SPLIT` (no bare "and"/"&"): a
#: descriptor's target list is far more likely to use "or" for genuine alternatives
#: and "and"/"&" for a single compound target ("skin and subcutaneous tissue").
_TARGET_SPLIT = re.compile(r"\s*(?:,|;|/|\bor\b)\s*", re.IGNORECASE)


def _candidate_anatomy_targets(feats) -> tuple[str, ...]:
    """The candidate's own anatomy phrase, decomposed into separate TARGET
    components (issue #6 F9-R2-C): "structure alpha or structure beta" names two
    alternative targets a single `concept_relation` call could never resolve, and
    comparing only the whole tail wrongly left a candidate that legitimately covers
    a documented target UNKNOWN (and so exposed to dominance exclusion) whenever
    that target was only ONE of several the descriptor names. A trailing "with or
    without ..." qualifier CLAUSE is truncated first -- dropped wholesale, never
    split into a false additional target -- because it qualifies the target(s)
    already named, it does not name a new one."""
    text = feats.anatomy_phrase
    if not text:
        return ()
    truncated = _WITH_OR_WITHOUT.sub("", text).strip()
    if not truncated:
        return ()
    parts = tuple(dict.fromkeys(
        p.strip() for p in _TARGET_SPLIT.split(truncated) if p.strip()))
    return parts or (truncated,)


def _anatomy_compatibility(candidate, facts: list[ClinicalFact], source,
                           reconciliation=None) -> str:
    """How this ONE candidate's anatomy relates to what `facts` document -- never a
    comparison between candidates; that comparative step is
    `_anatomy_dominance_exclusions`, below.

    CONTRADICTED_EXPLICIT only from the CLOSED laterality vocabulary
    (left/right/bilateral, `ontology._LATERALITY`) when both sides state one and they
    disagree -- a structural, never a lexical, comparison. SUPPORTED_EXACT/
    SUPPORTED_HIERARCHICAL come only from the governed concept-relation index
    (SAME / ancestor-descendant), tried across every documented fact PHRASE against
    every candidate TARGET component (issue #6 F9-R2-C, third pass: both sides are
    now decomposed, not just the fact side, so a fact documenting one of a
    candidate's several alternative targets still grounds it). Anything else --
    nothing documented, no candidate target to compare, no concept-relation
    capability, or the index itself UNRESOLVED for every pair -- is honestly
    UNKNOWN, never a guess in either direction."""
    feats = _candidate_descriptor_features(candidate, source)

    fact_laterality = _fact_attribute_value(facts, "laterality", reconciliation).lower()
    if fact_laterality and feats.laterality and fact_laterality not in feats.laterality:
        return _CONTRADICTED_EXPLICIT

    anatomy_phrases = _fact_anatomy_phrases(facts)
    candidate_targets = _candidate_anatomy_targets(feats)
    if not anatomy_phrases or not candidate_targets:
        return _UNKNOWN
    relate = getattr(source, "concept_relation", None)
    if not callable(relate):
        return _UNKNOWN
    from . import terminology as _term
    best = _UNKNOWN
    for phrase in anatomy_phrases:
        for target in candidate_targets:
            try:
                verdict = relate(phrase, target)
            except Exception:
                continue
            if verdict == _term.CONCEPT_SAME:
                return _SUPPORTED_EXACT
            if verdict == _term.CONCEPT_RELATED:
                best = _SUPPORTED_HIERARCHICAL
    return best


def _anatomy_dominance_exclusions(facts: list[ClinicalFact], candidates: list,
                                  source, reconciliation=None) -> dict[tuple[str, str], str]:
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
    verdicts = {(c.code, c.system): _anatomy_compatibility(c, facts, source, reconciliation)
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

    return None


def eligible(candidate, facts: list[ClinicalFact], source,
            date_of_service: str | None) -> bool:
    """Whether ONE already-retrieved candidate is semantically eligible for what
    `facts` document. Defaults to True (eligible) whenever there is nothing
    compiled to check against, or nothing compiled conflicts -- never a reason to
    exclude a candidate the vector search already found relevant."""
    return _ineligibility_reason(candidate, facts, source, date_of_service) is None


def eligible_partition(facts: list[ClinicalFact], candidates: list, source,
                       date_of_service: str | None, reconciliation=None) -> list:
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
    role_excluded = _service_role_exclusions(facts, pool, source, reconciliation,
                                             dos=date_of_service)
    pool = [c for c in pool if (c.code, c.system) not in role_excluded]
    excluded = _anatomy_dominance_exclusions(facts, pool, source, reconciliation)
    return [c for c in pool if (c.code, c.system) not in excluded]


def eligibility_report(facts: list[ClinicalFact], candidates: list, source,
                       date_of_service: str | None, reconciliation=None) -> list[dict]:
    """A full per-candidate audit record over `candidates` -- which `eligible_partition`
    would keep or exclude, and why -- preserved for the audit trail EVEN on a
    held/blocked outcome (issue #6 item 8), never only for a released line. Computed
    over the SAME `_ineligibility_reason` AND the same `_anatomy_dominance_exclusions`
    (run over the identical eligible-survivor pool) `eligible_partition` itself reads,
    so this record can never claim a different reason than the one that actually
    decided it."""
    pool = [c for c in candidates if eligible(c, facts, source, date_of_service)]
    role_control = _service_role_control(facts, pool, source, reconciliation,
                                         dos=date_of_service)
    surviving = [c for c in pool
                if not role_control[(c.code, c.system)].excluded]
    dominance = _anatomy_dominance_exclusions(facts, surviving, source, reconciliation)
    report = []
    for c in candidates:
        key = (c.code, c.system)
        reason = _ineligibility_reason(c, facts, source, date_of_service)
        role_decision = role_control.get(key)
        if reason is None and role_decision is not None and role_decision.excluded:
            reason = (f"candidate's authoritative classification is "
                     f"{role_decision.candidate_role!r}, incompatible with the "
                     f"fact's documented service_role {role_decision.fact_roles[0]!r}")
        if reason is None:
            reason = dominance.get(key)
        report.append({
            "code": c.code, "system": c.system, "eligible": reason is None,
            "reason": reason,
            # Codex F9-R11-H-D: ALWAYS present, even when this candidate was never
            # evaluated for role compatibility at all (key absent from
            # `role_control` -- excluded earlier by `eligible()` before the role
            # control ever ran) -- an audit trail that omits the field on a skip
            # is indistinguishable from one that forgot to check, which is the
            # exact defect this typed decision exists to close.
            "role_control": (
                {"status": role_decision.status.value,
                 "fact_roles": list(role_decision.fact_roles),
                 "candidate_role": role_decision.candidate_role,
                 "blocks_line": role_decision.blocks_line,
                 "authority_source_id": role_decision.authority_source_id,
                 "authority_version": role_decision.authority_version}
                if role_decision is not None else
                {"status": "not_evaluated", "fact_roles": [], "candidate_role": None,
                 "blocks_line": False, "authority_source_id": None,
                 "authority_version": None}),
        })
    return report
