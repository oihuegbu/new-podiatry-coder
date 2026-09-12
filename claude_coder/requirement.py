"""Descriptor/instructional-note requirement compilation and validation (issue #6
F9-R6) — the typed, source-grounded channel a candidate's own authoritative record
exposes for elimination, widening WHICH AXIS KINDS may eliminate a candidate without
ever loosening WHAT COUNTS as reaching that bar.

This module sits strictly downstream of `tiebreak.discriminating_axes` — it is a
mechanical PROJECTION of each `AxisProbe` into one or more `DescriptorRequirement`
records per candidate, never an independent axis compiler. For axis-derived
requirements, `required = probe.selectable`: an axis that could never eliminate a
candidate before this module existed still cannot after it. Only `provable` axes
(a quotation's WORDS can settle them) compile into text-clause requirements at
all — `AXIS_MEASUREMENT` (a typed, unit-converted interval comparison, never
provable by words) is deliberately not compiled here; `resolution.
_grounded_elimination` reasons about it directly against `fact.attributes` via
the existing `_measure_in_range`/`_interval_unsupported`, never through a
fabricated text clause.

`RequirementRole` (issue #6 F9-R6-R3/R6-R6 re-review) is the finer-grained,
authoritative discriminator for what a requirement's outcome may actually DO:
`MUST_SUPPORT` (validated absence may disqualify), `POSITIVE_ALIAS` (a
non-exhaustive example that may help narrow when present, but whose absence
proves nothing — ICD inclusion terms are always this), or `EXCLUSION` (reserved,
unpopulated). `required`/`selectable` stay as fields, kept consistent WITH role
at every construction site rather than independently meaningful.

Never medical vocabulary in Python: every requirement's `expected` value and
`authority_clause` come from DATA — a candidate's own AMA/CMS descriptor text
(`CandidateCode.descriptor`), or the CDC/NCHS ICD-10-CM Tabular's own
`inclusionTerm` field (`AuthoritativeSource.instructional_terms`) — never a
hardcoded term/code/family/specialty list.

Why `instructional_terms` is ICD-10-CM-only, researched and not just assumed
(issue #6, post-F9-R6 follow-up): neither `data/codes/cpt_codes.json` nor
`hcpcs_codes.json` carries any field resembling ICD-10-CM's `inclusionTerm` —
CPT records hold only description tiers + `concept_id` + `effective_date`;
HCPCS's non-description fields (`coverage_code`, `betos`, `statute`, ...) are
payment/coverage metadata, already consumed elsewhere in `app/compliance/`, not
clinical-disambiguation text. The real analog — AMA CPT's parenthetical/
cross-reference notes — is not public domain (AMA copyright) and is not present
in this repo's CPT extract at all. The closest available public-domain
alternative, the CMS NCCI Policy Manual (`data/policy/ncci_policy_manual.txt`,
already fetched and used by `tools/policy_corpus.py` for a DIFFERENT purpose —
verifying a model-supplied quote isn't invented), is organized by billing-rule
topic/chapter, not per individual CPT/HCPCS code, so there is no reliable way to
extract a candidate-code's own disambiguating term from it without either
brittle text-search heuristics or a model doing the extraction — the latter
being exactly the untrusted-self-report pattern this module exists to avoid.
Closing this gap for real needs a genuine structured, per-code, public-domain
(or licensed) data source that does not currently exist in this repo; it is not
a wiring gap the way ICD-10-CM's `inclusionTerm` was.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from . import tiebreak as _tiebreak
from .models import CandidateCode


class RequirementStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_DOCUMENTED = "not_documented"
    UNRESOLVED = "unresolved"


class RequirementRole(str, Enum):
    """What a requirement's outcome is actually allowed to DO (issue #6
    F9-R6-R3/R6-R6 re-review) -- a strictly finer discriminator than the
    boolean `required`/`selectable` pair, which conflated "this axis kind can
    ever eliminate/select" with "this SPECIFIC requirement's absence disproves
    the candidate", a distinction the ICD-10-CM inclusion-term finding proved
    matters.
    """
    #: This requirement's own absence, once validated, may disqualify the
    #: candidate. Only axes with a real closed/typed shape (currently:
    #: laterality) ever earn this.
    MUST_SUPPORT = "must_support"
    #: A non-exhaustive EXAMPLE that may help narrow/select when genuinely,
    #: assertedly present, but whose absence proves nothing -- the record may
    #: simply describe the same condition a different, unlisted way. ICD
    #: inclusion terms are always this. Never enters elimination.
    POSITIVE_ALIAS = "positive_alias"
    #: Reserved for a genuinely exclusionary source fact (not populated by
    #: any compiler yet) -- kept distinct from MUST_SUPPORT so a future
    #: exclusion-shaped source never has to overload "this axis is required".
    EXCLUSION = "exclusion"


@dataclass(frozen=True)
class CoverageCorpus:
    """The identity of the ONE independently-read document text a
    NOT_DOCUMENTED or SUPPORTED verdict may be deterministically checked
    against (issue #6 F9-R6-R4/R5 re-review) -- never a bare string. Binds
    WHICH channel was searched, a content hash (so the audit record can prove
    the same corpus was used without embedding the whole document text), and
    page coverage, so "not documented" can never quietly mean "in an
    unidentified or partial excerpt".
    """
    channel_id: str
    text: str
    text_sha256: str
    covered_pages: tuple[int, ...] = ()
    uncovered_pages: tuple[int, ...] = ()
    page_image_sha256: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Self-validate identity at construction, not on first read (issue #6
        F9-R6-R4, fourth re-review): the old `.complete` check never verified
        `channel_id`/`covered_pages` were populated at all, or that `text_sha256`
        genuinely matches `text` -- a blank-channel, zero-page, or wrong-hash
        corpus could report `complete=True` today. A frozen dataclass may still
        raise from `__post_init__` (it only forbids ASSIGNING fields here),
        so this fails fast and loud instead of silently validating a corpus
        that never proves what it claims to."""
        import hashlib
        if not self.channel_id or not self.covered_pages:
            raise ValueError(
                "CoverageCorpus must identify a channel and at least one covered page")
        if self.text_sha256 != hashlib.sha256(self.text.encode("utf-8")).hexdigest():
            raise ValueError("CoverageCorpus.text_sha256 does not match its own text")
        if len(self.page_image_sha256) != len(self.covered_pages):
            raise ValueError(
                "CoverageCorpus needs exactly one page_image_sha256 per covered page")

    @property
    def complete(self) -> bool:
        """Both gates a validated absence needs, together: real text AND no
        page this channel failed to cover. Neither alone is sufficient -- a
        zero-page or fully-uncovered-but-nonempty-flag document must never
        let NOT_DOCUMENTED validate. `channel_id`/`covered_pages`/hash identity
        are now guaranteed by `__post_init__`, so this property only needs to
        check what can still vary AFTER construction succeeds."""
        return bool(self.text.strip()) and bool(self.covered_pages) and not self.uncovered_pages

    def as_record(self) -> dict[str, Any]:
        return {"channel_id": self.channel_id, "text_sha256": self.text_sha256,
                "covered_pages": list(self.covered_pages),
                "uncovered_pages": list(self.uncovered_pages),
                "page_image_sha256": list(self.page_image_sha256),
                "complete": self.complete}


@dataclass(frozen=True)
class DescriptorRequirement:
    """One typed, source-anchored fact a candidate's own authoritative record
    states — compiled from an `AxisProbe` `tiebreak.discriminating_axes` already
    derived, never a separately invented axis. `required` mirrors the originating
    probe's `selectable`: only an axis that could already eliminate/select a
    candidate before this module existed can be `required` here. `role`
    (issue #6 F9-R6-R3/R6-R6 re-review) is the authoritative elimination-
    eligibility discriminator now used at the call site -- `required` stays
    for backward-compatible audit/prompt display, kept consistent with `role`
    at every construction site rather than independently meaningful.
    """
    requirement_id: str
    axis: str
    candidate_code: str
    required: bool
    role: RequirementRole
    expected: tuple[str, ...]
    authority_clause: str
    authority_offset: tuple[int, int]
    authority_source_text: str
    source_identity: dict[str, Any] = field(default_factory=dict)
    selectable: bool = False
    queryable: bool = False

    def as_record(self) -> dict[str, Any]:
        return {"requirement_id": self.requirement_id, "axis": self.axis,
                "candidate_code": self.candidate_code, "required": self.required,
                "role": self.role.value,
                "expected": list(self.expected),
                "authority_clause": self.authority_clause,
                "authority_offset": list(self.authority_offset),
                # issue #6 F9-R6-R5: without the full source text, an auditor
                # cannot reproduce the clause-offset check this record claims to
                # support -- `authority_clause`/`authority_offset` alone are not
                # self-contained.
                "authority_source_text": self.authority_source_text,
                "selectable": self.selectable, "queryable": self.queryable,
                "source_identity": dict(self.source_identity)}


@dataclass(frozen=True)
class RequirementJudgement:
    """One evaluator's verdict on one `DescriptorRequirement`, exactly as returned
    by the verifier — never trusted on its own; see `validated_requirement`."""
    requirement_id: str
    status: RequirementStatus
    evidence_span_ids: tuple[str, ...] = ()
    quoted_text: str = ""
    evaluator_origin: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def as_record(self) -> dict[str, Any]:
        return {"requirement_id": self.requirement_id, "status": self.status.value,
                "evidence_span_ids": list(self.evidence_span_ids),
                "quoted_text": self.quoted_text,
                "evaluator_origin": dict(self.evaluator_origin), "reason": self.reason}


def _find_clause(source_text: str, term: str) -> tuple[str, tuple[int, int]] | None:
    """The verbatim clause + offset for `term` inside `source_text`, or None when
    `term` is not literally present — a requirement can never be compiled from a
    clause that does not actually reproduce from the record it claims to come from.
    """
    if not source_text or not term:
        return None
    idx = source_text.lower().find(term.lower())
    if idx < 0:
        return None
    end = idx + len(term)
    return source_text[idx:end], (idx, end)


def _semantic_anatomy_requirements(candidates: list[CandidateCode], source: Any
                                   ) -> list[DescriptorRequirement]:
    """MUST_SUPPORT requirements from each candidate's own GOVERNED anatomy target,
    where "governed" means resolved to exactly one concept in the real SNOMED CT
    Body Structure graph -- the same index and the same atomic decomposition
    `semantic_eligibility._anatomy_compatibility` already uses for an analogous
    purpose, never a new capability invented here.

    Fixes both defects Codex's independent re-review found in the REMOVED
    `_semantic_concept_requirements` (see the historical comment at this
    function's call site):

    1. Real anatomy vs. a technique/approach word. `ontology.parse_descriptor`'s
       `anatomy_phrase` is still a raw structural comma-split (unchanged by this
       fix -- see its own docstring), so this never trusts it directly. Each
       decomposed target phrase (`semantic_eligibility._candidate_anatomy_targets`,
       the SAME atomic list-split that already truncates "with or without"
       qualifier clauses and splits alternatives) must resolve, via
       `source.concept_lookup("anatomy", phrase)`, to EXACTLY ONE concept id
       (`unique=True`) before it may compile into anything. A technique/approach
       word is simply not in this concept graph at all -- it resolves to zero
       candidates and is silently dropped, never mistaken for anatomy. An
       AMBIGUOUS match (more than one candidate concept) is dropped too, the
       same fail-closed discipline `_anatomy_compatibility` already applies.

    2. One semantic assertion per requirement. `expected` is the target phrase
       PLUS ONLY its own known governed synonyms for the SAME concept id
       (`concept_lookup`'s own `expansions` field -- alternate official
       phrasings, never populated for an ambiguous match). That is not the
       bundling defect Codex found: every term in `expected` here names the
       IDENTICAL concept, so `tiebreak.asserted_status`'s ANY-of-`expected`
       semantics is the CORRECT reading (any accepted phrasing of one concept
       counts as the one assertion being made), never a second, distinct claim
       smuggled in under the same tuple.

    A concept EVERY tied candidate's own descriptor requires is excluded from
    every one of their requirement sets (identity by governed CONCEPT ID, never
    by phrase string, so two different wordings of the same concept are
    correctly treated as one shared requirement, not two distinct ones) -- the
    same "only the DIFFERENCE across the tied set becomes a requirement"
    principle `discriminating_axes` already applies to laterality and the
    descriptor-term bucket. Without this, a concept literally every candidate
    shares being validated NOT_DOCUMENTED would let `resolution.
    _grounded_elimination` "confirm" eliminating ANY one of them on a defect
    every other tied candidate -- including whichever one a model separately
    named as the winner -- equally has, which proves nothing about which one
    the record actually means.

    issue #6, Codex's independent re-review (round 4, RC3 provenance gap):
    `source_identity` now carries `concept_lookup`'s OWN `source_identity` (the
    terminology index's snapshot identity the concept id was actually matched
    against) and this candidate's `record_snapshot_identity` (the descriptor
    table snapshot the target phrase was read from) -- the same two bindings
    the descriptor axis above already attaches. Independent reproduction
    without these caught it: a requirement naming a real concept id with no
    terminology source/version behind it is not yet defensible lineage.
    """
    if source is None or len(candidates) < 2:
        return []
    lookup = getattr(source, "concept_lookup", None)
    if not callable(lookup):
        return []
    from . import ontology as _ontology
    from . import semantic_eligibility as _semeli
    snap_fn = getattr(source, "record_snapshot_identity", None)

    # code -> {target phrase: {"concept_id", "expansions", "terminology_identity"}}
    resolved: dict[str, dict[str, dict[str, Any]]] = {}
    for c in candidates:
        feats = _ontology.parse_descriptor(c.descriptor)
        entries: dict[str, dict[str, Any]] = {}
        for target in _semeli._candidate_anatomy_targets(feats):
            try:
                match = lookup("anatomy", target)
            except Exception:
                continue
            if not isinstance(match, dict):
                continue
            ids = tuple(match.get("candidates") or ())
            if not match.get("unique") or len(ids) != 1:
                continue    # unresolved or ambiguous -- never governed enough
            expansions = tuple(str(e).strip() for e in (match.get("expansions") or ())
                               if str(e).strip())
            entries[target] = {
                "concept_id": ids[0], "expansions": expansions,
                "terminology_identity": dict(match.get("source_identity") or {})}
        if entries:
            resolved[c.code] = entries
    if len(resolved) < 2:
        return []

    concept_sets = {code: {e["concept_id"] for e in entries.values()}
                    for code, entries in resolved.items()}
    shared = set.intersection(*concept_sets.values())

    out: list[DescriptorRequirement] = []
    by_code = {c.code: c for c in candidates}
    for code in sorted(resolved):
        candidate = by_code[code]
        descriptor_snapshot: dict[str, Any] = {}
        if callable(snap_fn):
            try:
                descriptor_snapshot = snap_fn(code, candidate.system) or {}
            except Exception:
                descriptor_snapshot = {}
        for target, entry in sorted(resolved[code].items()):
            concept_id = entry["concept_id"]
            if concept_id in shared:
                continue
            found = _find_clause(candidate.descriptor, target)
            if found is None:
                continue
            clause, offset = found
            out.append(DescriptorRequirement(
                requirement_id=f"semantic_anatomy:{code}:{len(out)}",
                axis="semantic_anatomy", candidate_code=code, required=True,
                role=RequirementRole.MUST_SUPPORT,
                expected=(target,) + entry["expansions"],
                authority_clause=clause, authority_offset=offset,
                authority_source_text=candidate.descriptor,
                source_identity={"kind": "semantic_concept", "system": candidate.system,
                                 "authority": dict(candidate.authority or {}),
                                 "concept_axis": "anatomy", "concept_id": concept_id,
                                 "terminology_identity": entry["terminology_identity"],
                                 "descriptor_snapshot": descriptor_snapshot},
                selectable=True, queryable=False))
    return out


def _semantic_action_requirements(candidates: list[CandidateCode], source: Any
                                  ) -> list[DescriptorRequirement]:
    """MUST_SUPPORT requirements from each candidate's own COHERENT GOVERNED
    action identity (issue #6, Codex's independent re-review, F9-R13-C, round
    2/P1-B): resolved from the candidate's FULL official descriptor, never a
    structural comma-split prefix. Codex's exact-SHA reproduction proved the
    prefix approach ('assembly procedure', 'installation procedure' -- the
    action_phrase alone) frequently omits the target that makes the SNOMED
    Procedure concept specific, leaving both sides UNRESOLVED even when the
    fact-side action was genuinely governed. `procedure_relation_detail`'s own
    `embedded=True` matcher already resolves a governed phrase CONTAINED
    within a longer description -- feeding it the whole descriptor lets it
    find that phrase itself, rather than requiring this module to have
    pre-isolated it structurally first.

    Governed at compile time by a self-check against the candidate's own full
    descriptor (compared to itself): a real concept graph resolving it to
    EXACTLY ONE concept id is a faithful proxy for "this descriptor names a
    real, unambiguous action concept" -- the same discipline `concept_lookup`'s
    `unique` gate applies elsewhere. An ungoverned/technique descriptor this
    graph cannot resolve uniquely returns UNRESOLVED/ambiguous and is dropped,
    never mistaken for a real action.

    `selectable=False` deliberately: this requirement is never handed to
    `tiebreak._axes_from_requirements`'s generic literal-text narrowing --
    selection happens ONLY through `semantic_axis_status`, called by
    `resolution._select_by_semantic_axes`, which reads THIS fact's own atomic
    description against the candidate's descriptor through the SAME governed
    graph, never blind text presence. `procedure_relation_detail`'s "never
    selects or expands a code" boundary is honored literally: it is consulted
    only inside that one axis-support test, one input among several (official
    descriptor + independent entailment + DOS/CMS controls) a selection still
    requires ALL of.
    """
    if source is None or len(candidates) < 2:
        return []
    relate = getattr(source, "procedure_relation_detail", None)
    if not callable(relate):
        return []
    from . import terminology as _term
    snap_fn = getattr(source, "record_snapshot_identity", None)

    out: list[DescriptorRequirement] = []
    for candidate in sorted(candidates, key=lambda c: c.code):
        descriptor = candidate.descriptor.strip()
        if not descriptor:
            continue
        try:
            self_check = relate(descriptor, descriptor)
        except Exception:
            continue
        if not isinstance(self_check, dict) or self_check.get("verdict") != _term.CONCEPT_SAME:
            continue    # not a real, uniquely recognized governed action concept
        concept_ids = tuple((self_check.get("term_a") or {}).get("candidates") or ())
        if not (self_check.get("term_a") or {}).get("unique") or len(concept_ids) != 1:
            continue    # ambiguous -- never governed enough to require anything on
        found = _find_clause(candidate.descriptor, descriptor)
        if found is None:
            continue
        clause, offset = found
        descriptor_snapshot: dict[str, Any] = {}
        if callable(snap_fn):
            try:
                descriptor_snapshot = snap_fn(candidate.code, candidate.system) or {}
            except Exception:
                descriptor_snapshot = {}
        out.append(DescriptorRequirement(
            requirement_id=f"semantic_action:{candidate.code}:{len(out)}",
            axis="semantic_action", candidate_code=candidate.code, required=True,
            role=RequirementRole.MUST_SUPPORT,
            expected=(descriptor,),
            authority_clause=clause, authority_offset=offset,
            authority_source_text=candidate.descriptor,
            source_identity={"kind": "semantic_concept", "system": candidate.system,
                             "authority": dict(candidate.authority or {}),
                             "concept_axis": "action", "concept_id": concept_ids[0],
                             "terminology_identity": dict(
                                 self_check.get("source_identity") or {}),
                             "descriptor_snapshot": descriptor_snapshot},
            selectable=False, queryable=False))
    return out


def _semantic_qualifier_requirements(candidates: list[CandidateCode], source: Any
                                     ) -> list[DescriptorRequirement]:
    """DO NOT manufacture approach/qualifier requirements from
    `ontology.parse_descriptor`'s `anatomy_phrase` (issue #6, Codex's
    independent re-review, F9-R13-C, round 2/P1-B). `anatomy_phrase` is the
    ENTIRE descriptor tail after the first comma/semicolon -- it can be
    anatomy, technique, extent, or several clauses at once; the source itself
    calls it `anatomy_phrase`, and re-labeling it a "qualifier"/"approach" for
    THIS axis was type confusion, not a design nuance. Codex's exact-SHA
    reproduction proved this concretely: a tail like "structure alpha" or
    "structure beta" compiled as a `semantic_qualifier` requirement and was
    compared against the fact's typed `approach` attribute, which it is not
    and never was -- always UNRESOLVED, keeping the selector inert regardless
    of the missing-snapshot fix.

    Returns nothing until a genuinely, independently typed descriptor
    qualifier facet exists in this repo (a real per-code field this codebase
    does not yet have) -- not a stand-in, not a heuristic. An untyped
    residual descriptor difference remains exactly what it already is
    elsewhere: `tiebreak.AXIS_DESCRIPTOR_TERM`, audited but never selecting,
    or (when no other axis settles it) one precise candidate-missing-fact
    line -- never silently promoted to a requirement here.
    """
    return []


def _norm_qualifier(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def semantic_axis_status(req: "DescriptorRequirement", fact: Any, reconciliation: Any,
                         source: Any) -> "RequirementStatus":
    """The concept-equivalence-based status of ONE `semantic_action`/
    `semantic_qualifier` requirement (issue #6, Codex's independent re-review,
    F9-R13-C) -- reading ONLY this fact's own atomic evidence, never borrowed
    from a sibling fact and never from raw text presence anywhere else in the
    document.

    `semantic_action`: `source.procedure_relation_detail(fact_description,
    candidate_action_phrase)` -- a unique governed SAME verdict is SUPPORTED;
    anything else (unresolved, ambiguous, no capability) is UNRESOLVED. There
    is no typed "opposite" signal for a documented action, so this axis never
    reports CONTRADICTED.

    `semantic_qualifier`: this fact's own claim-AUTHORIZED `approach` value
    (`graph_consensus.claim_authorized_value` -- the same reconciled,
    ASSERTED, value-bound `attribute_evidence` gate laterality/service_role
    already use, never a raw `attributes` read) compared to the requirement's
    own tail text. A genuinely different claim-authorized value is an
    explicit, typed CONTRADICTED -- not silently UNRESOLVED, since the fact
    positively states a different approach than this candidate needs.

    Returns UNRESOLVED for any other axis or on any capability/data gap --
    fails closed, never guesses.
    """
    if req.axis == "semantic_action":
        relate = getattr(source, "procedure_relation_detail", None)
        if not callable(relate):
            return RequirementStatus.UNRESOLVED
        description = str(getattr(fact, "description", "") or "").strip()
        if not description or not req.expected:
            return RequirementStatus.UNRESOLVED
        from . import terminology as _term
        try:
            detail = relate(description, req.expected[0])
        except Exception:
            return RequirementStatus.UNRESOLVED
        if isinstance(detail, dict) and detail.get("verdict") == _term.CONCEPT_SAME:
            return RequirementStatus.SUPPORTED
        return RequirementStatus.UNRESOLVED
    if req.axis == "semantic_qualifier":
        from . import graph_consensus as _gc
        documented = _gc.claim_authorized_value(fact, "approach", reconciliation)
        if not documented or not req.expected:
            return RequirementStatus.UNRESOLVED
        norm_documented = _norm_qualifier(documented)
        if any(norm_documented == _norm_qualifier(t) for t in req.expected):
            return RequirementStatus.SUPPORTED
        return RequirementStatus.CONTRADICTED
    return RequirementStatus.UNRESOLVED


def _descriptor_term_requirements(candidates: list[CandidateCode]
                                  ) -> list[DescriptorRequirement]:
    """MUST_SUPPORT requirements from each candidate's own raw, distinguishing
    descriptor wording (issue #6, Codex's independent re-review, F9-R14-B) --
    the SAME terms `tiebreak.discriminating_axes`'s `AXIS_DESCRIPTOR_TERM` probe
    already isolates mechanically (each candidate's descriptor tokens minus every
    token every candidate shares -- never a medical keyword list), now compiled
    into a real requirement instead of only an audited, non-eliminating probe.

    This does NOT resurrect the reverted `_APPROACH_WORDS` mistake
    (`discriminating_axes`'s own docstring) or the removed `anatomy_phrase`-as-
    qualifier heuristic (`_semantic_qualifier_requirements`'s docstring): both of
    those eliminated directly from bare text presence/absence, trusting no model
    and no full-document search, and were found unsafe for exactly that reason.
    THIS requirement can only ever ground an elimination through the existing,
    unrelated-to-this-round bar in `resolution._grounded_elimination`: TWO
    INDEPENDENT models must EACH affirmatively judge the term NOT_DOCUMENTED
    after being shown this fact's own evidence, AND a deterministic, complete
    whole-document search (`requirement.deterministic_status`, gated on
    `CoverageCorpus.complete`) must independently agree -- never bare silence
    alone, and never one model's self-report. A missing judgement, a
    disagreement, an unvalidated citation, or incomplete coverage all leave the
    candidate standing exactly as `AXIS_DESCRIPTOR_TERM` always has -- the
    fail-closed posture is unchanged; only the set of candidate requirements
    available to the EXISTING gate grows.

    One requirement PER INDIVIDUAL term (never one requirement whose `expected`
    bundles several distinct words together): a prior, since-removed mechanism
    in this same module bundled several distinct tokens into one `expected`
    tuple and `tiebreak.asserted_status` treats a multi-term `expected` as
    ANY-term-supported, letting an unstated word ride along with a stated one
    to a false SUPPORTED verdict. Splitting one-term-per-requirement, grouped
    under the shared `descriptor_term` axis (so `_grounded_elimination`'s
    axis-group check already requires EVERY one of a candidate's own
    distinguishing terms, not just one, to be independently, unanimously
    absent) avoids that exact defect shape.

    `selectable=False, queryable=False`: this requirement only ever ELIMINATES
    through the validated NOT_DOCUMENTED path above -- it is never handed to
    `tiebreak._axes_from_requirements`'s generic literal-text narrowing (that
    remains `AXIS_DESCRIPTOR_TERM`'s own, unrelated, audit-only reporting path),
    and it never becomes a provider question on its own (an open-ended
    descriptor-token bag is not provider-answerable, same reasoning
    `AxisProbe.queryable` already documents).
    """
    if len(candidates) < 2:
        return []
    by_code = {c.code: c for c in candidates}
    out: list[DescriptorRequirement] = []
    for probe in _tiebreak.discriminating_axes(candidates):
        if probe.axis != _tiebreak.AXIS_DESCRIPTOR_TERM:
            continue
        for code, terms in sorted(probe.terms_by_code.items()):
            candidate = by_code.get(code)
            if candidate is None:
                continue
            for term in terms:
                found = _find_clause(candidate.descriptor, term)
                if found is None:
                    continue
                clause, offset = found
                out.append(DescriptorRequirement(
                    requirement_id=f"descriptor_term:{code}:{len(out)}",
                    axis="descriptor_term", candidate_code=code, required=True,
                    role=RequirementRole.MUST_SUPPORT,
                    expected=(term,), authority_clause=clause,
                    authority_offset=offset, authority_source_text=candidate.descriptor,
                    source_identity={"kind": "descriptor", "system": candidate.system,
                                     "authority": dict(candidate.authority or {})},
                    selectable=False, queryable=False))
    return out


def compile_requirements(candidates: list[CandidateCode], source: Any = None
                         ) -> tuple[DescriptorRequirement, ...]:
    """Every typed requirement the tied candidates' own authoritative records
    state — a mechanical projection of `tiebreak.discriminating_axes(candidates)`,
    plus (ICD-10-CM only, when `source` supplies it) each candidate's own governed
    `inclusionTerm` phrases. Never re-derives axis logic independently.

    A candidate with an empty `terms_by_code` entry for a probe is silent on that
    axis — correctly produces NO requirement for it (nothing to require; the same
    reason `AxisProbe`'s own docstring gives for never treating silence as proof).
    Only `provable` axes compile at all (see module docstring re: measurement).
    """
    axes = [p for p in _tiebreak.discriminating_axes(candidates) if p.provable]
    by_code = {c.code: c for c in candidates}
    out: list[DescriptorRequirement] = []
    for probe in axes:
        for code, terms in sorted(probe.terms_by_code.items()):
            if not terms:
                continue
            candidate = by_code.get(code)
            if candidate is None:
                continue
            found = _find_clause(candidate.descriptor, terms[0])
            if found is None:
                continue
            clause, offset = found
            # issue #6 F9-R6-R5, fourth re-review: the bound WHOLE-FILE snapshot
            # identity (sha256/size/edition) the descriptor table was actually
            # parsed from, keyed to this specific code -- duck-typed and
            # fail-closed to {} exactly like `instructional_terms` below, since
            # not every `source` implementation supplies it (e.g. a bare test
            # double built before this method existed).
            snap_fn = getattr(source, "record_snapshot_identity", None)
            snapshot: dict[str, Any] = {}
            if callable(snap_fn):
                try:
                    snapshot = snap_fn(code, candidate.system) or {}
                except Exception:
                    snapshot = {}
            out.append(DescriptorRequirement(
                requirement_id=f"{probe.axis}:{code}:{len(out)}",
                axis=probe.axis, candidate_code=code, required=probe.selectable,
                role=(RequirementRole.MUST_SUPPORT if probe.selectable
                     else RequirementRole.POSITIVE_ALIAS),
                expected=tuple(terms), authority_clause=clause,
                authority_offset=offset, authority_source_text=candidate.descriptor,
                # issue #6 F9-R6-R5: the candidate's own real, already-populated
                # provenance dict, not just the axis's kind/system -- lets an
                # auditor tell WHICH edition of the descriptor this requirement
                # was compiled against.
                source_identity={"kind": "descriptor", "system": candidate.system,
                                 "authority": dict(candidate.authority or {}),
                                 "snapshot": snapshot},
                selectable=probe.selectable, queryable=probe.queryable))

    # issue #6, Codex's independent re-review, root cause 3: a prior round added
    # `_semantic_concept_requirements`, projecting `semantics.compiled_record`'s
    # `action_concepts`/`anatomy_concepts` into MUST_SUPPORT requirements.
    # REMOVED, not merely disabled: Codex found two real defects, not a design
    # nuance. (1) `action_concepts`/`anatomy_concepts` are `ontology.
    # parse_descriptor`'s raw comma-split tokens -- a STRUCTURAL parse, not a
    # semantically-validated one, so a technique/approach word ("assembly
    # service, POWERED technique") parses into `anatomy_concepts` exactly like
    # real anatomy would, with no governed source in this repo to tell them
    # apart. (2) a candidate's several distinct tokens were compiled into ONE
    # `expected` tuple, but `tiebreak.asserted_status` treats a multi-term
    # `expected` as ANY-term-supported -- reproduced exactly: `expected =
    # ("alpha", "powered")`, document states only "powered", and the WHOLE
    # requirement (including the unstated "alpha") validated SUPPORTED. A
    # governed replacement needs requirements compiled one-semantic-assertion-
    # per-requirement (never mixing anatomy and technique tokens in one
    # `expected`), classified only against a versioned SNOMED/UMLS concept
    # identity, and a composite proof (descriptor + atomic fact's own
    # reconciled span + governed equivalence + independent entailment).
    #
    # Round 3 (Codex's independent re-review): that governed replacement, for
    # the ANATOMY half. Codex rejected treating the full replacement as
    # deferred/future work -- `_semantic_anatomy_requirements` below is that
    # replacement, not a stand-in: it fixes both named defects directly (see
    # its own docstring), reuses the SAME governed SNOMED Body Structure
    # concept index `semantic_eligibility._anatomy_compatibility` already
    # trusts for an analogous purpose (never a new, unreviewed capability),
    # and is wired through this exact function so its output flows through the
    # EXISTING `validated_requirement`/`resolution._grounded_elimination`
    # path -- no parallel selector.
    #
    # Round 4 (Codex's independent re-review, F9-R13-C): the action and
    # qualifier halves. Codex refined the boundary rather than accepting
    # deferral: `procedure_relation_detail`'s "never selects or expands a
    # code" warning is honored literally -- it never independently
    # eliminates/selects here either. It is consulted ONLY inside
    # `_semantic_axis_status` (below), which `resolution._settle_uniqueness`
    # calls to test ONE additional candidate-selection condition on top of
    # every existing DOS/entailment/CMS control, never in place of them. Both
    # `_semantic_action_requirements`/`_semantic_qualifier_requirements`
    # compile with `selectable=False` -- deliberately kept OUT of
    # `tiebreak._axes_from_requirements`'s generic literal-text narrowing
    # (`tiebreak.narrow`), which is exactly the "_APPROACH_WORDS" mistake
    # `discriminating_axes`'s own docstring documents reverting: raw
    # descriptor wording must never independently select a code through blind
    # literal presence. Selection here happens only through the new,
    # dedicated, deterministic `_semantic_axis_status` path, gated on real
    # governed identity (action) or this fact's own typed, reconciled
    # attribute evidence (qualifier) -- never on text merely appearing
    # somewhere in the document.
    out.extend(_semantic_anatomy_requirements(candidates, source))
    out.extend(_semantic_action_requirements(candidates, source))
    out.extend(_semantic_qualifier_requirements(candidates, source))
    out.extend(_descriptor_term_requirements(candidates))

    if source is not None:
        resolver = getattr(source, "instructional_terms", None)
        snapshot_fn = getattr(source, "instructional_terms_snapshot", None)
        if callable(resolver):
            # issue #6 F9-R6-R5, fourth re-review: bound once per compile call
            # (the instructional_notes table is one file shared by every
            # candidate/term below, unlike descriptors which are per-system) --
            # duck-typed and fail-closed to {} the same way `resolver` itself is.
            inclusion_snapshot: dict[str, Any] = {}
            if callable(snapshot_fn):
                try:
                    inclusion_snapshot = snapshot_fn() or {}
                except Exception:
                    inclusion_snapshot = {}
            for candidate in sorted(candidates, key=lambda c: c.code):
                if candidate.system != "icd10":
                    continue
                try:
                    terms = resolver(candidate.code, candidate.system)
                except Exception:
                    continue
                for term in terms:
                    found = _find_clause(term, term)   # the term IS its own source text
                    if found is None:
                        continue
                    clause, offset = found
                    out.append(DescriptorRequirement(
                        requirement_id=f"inclusion_term:{candidate.code}:{len(out)}",
                        axis="inclusion_term", candidate_code=candidate.code,
                        # issue #6 F9-R6-R3 re-review: inclusion terms are
                        # non-exhaustive EXAMPLES per the ICD-10-CM guidelines
                        # -- absence of even every listed example never
                        # disproves the diagnosis, since an unlisted synonym
                        # can map to the same code. POSITIVE_ALIAS, never
                        # required, never selectable: may widen retrieval
                        # (unaffected, upstream of this compiler) and may only
                        # narrow/select once a real asserted-span control
                        # exists for this axis kind -- not wired here.
                        required=False, role=RequirementRole.POSITIVE_ALIAS,
                        expected=(term,), authority_clause=clause,
                        authority_offset=offset, authority_source_text=term,
                        source_identity={"kind": "instructional_notes",
                                         "system": candidate.system,
                                         "authority": dict(candidate.authority or {}),
                                         "snapshot": inclusion_snapshot},
                        selectable=False, queryable=False))
    return tuple(out)


def deterministic_status(req: DescriptorRequirement, coverage: CoverageCorpus | None
                         ) -> RequirementStatus | None:
    """The TRUE relation between `req.expected` and the ONE independently-read
    `coverage` corpus, found by ACTUALLY SEARCHING the text -- never a
    verifier's self-report standing in for it (issue #6 F9-R6-R2 re-review).
    Uses `tiebreak.asserted_status`, the SAME clause-scoped, negation-aware
    primitive `tiebreak.narrow` proves its own axes with, so "the document
    states this" means the same thing wherever it's checked in this codebase.

    Returns `None` when `coverage` is missing or `not coverage.complete`: no
    claim, positive OR negative, can be made about a document nobody fully
    supplied -- the caller (`validated_requirement`) must fail closed on
    `None`.

    Maps `asserted_status`'s three-way answer onto `RequirementStatus`:
    "supported" -> SUPPORTED, "absent" -> NOT_DOCUMENTED, and (issue #6
    F9-R6-R6 re-review) "negated" -> CONTRADICTED -- a phrase that appears
    ONLY negated ("no classic presentation") is a genuinely different, more
    specific claim than silence, and must validate NEITHER a SUPPORTED NOR a
    NOT_DOCUMENTED verdict (`validated_requirement` rejects CONTRADICTED
    outright on the judgement side regardless, so this never reopens a
    judgement-driven elimination path for it -- it only makes the OTHER two
    statuses correctly refuse to validate against a negated occurrence).
    """
    if coverage is None or not coverage.complete:
        return None
    status = _tiebreak.asserted_status(req.expected, coverage.text)
    return {"supported": RequirementStatus.SUPPORTED,
           "negated": RequirementStatus.CONTRADICTED,
           "absent": RequirementStatus.NOT_DOCUMENTED}[status]


def validated_requirement(req: DescriptorRequirement, judgement: RequirementJudgement,
                          *, evidence_by_span_id: dict[str, str] | None = None,
                          reconciliation: Any = None,
                          coverage: CoverageCorpus | None = None) -> bool:
    """Does an evaluator's judgement of `req` deserve to affect selection?

    issue #6 F9-R6-R2 (Codex re-review, two rounds): the ORIGINAL version of
    this function checked only that a cited span was REAL and page-reconciled
    -- never that the span's actual CONTENT related to `req.expected` at all.
    The FIRST fix added a deterministic whole-document search, but still
    validated SUPPORTED against a span that was merely reconciled, not
    against what that SPECIFIC cited span's own text says -- a model could
    cite a real but unrelated span, as long as the phrase happened to appear
    ANYWHERE ELSE in the document. This version fixes both, permanently:

    1. CONTRADICTED never validates as a JUDGEMENT status, for any axis,
       unconditionally -- see the `if judgement.status not in (...)` check
       below. Confirming a phrase-type requirement is genuinely contradicted
       (the note actively states the opposite, not merely silent) needs real
       negation-detection -- which this module now HAS
       (`tiebreak.asserted_status`), but the judgement-driven CONTRADICTED
       path stays retired regardless: a model's own CONTRADICTED claim is
       still never trusted, deterministic negation detection is used instead,
       purely to make SUPPORTED/NOT_DOCUMENTED correctly refuse a negated
       occurrence rather than to resurrect CONTRADICTED as a judgement-driven
       elimination path. Laterality's real closed-enumeration contradiction
       remains fully handled by the pre-existing, judgement-INDEPENDENT
       `tiebreak.narrow` document-proof fallback in
       `resolution._grounded_elimination`.
    2. SUPPORTED requires EVERY cited span to, INDIVIDUALLY, genuinely and
       un-negatedly support the term -- checked against that span's own text
       (`evidence_by_span_id`), never the whole document standing in for a
       specific citation's content. NOT_DOCUMENTED requires the WHOLE
       `coverage` corpus to show genuine absence -- "negated" (found, but
       explicitly negated) is a different, more specific claim than silence
       and must not validate NOT_DOCUMENTED either. `coverage` missing or
       incomplete means NOT_DOCUMENTED can never validate -- fails closed,
       matching `CoverageCorpus.complete`'s own posture.

    On top of the content checks: the clause must still actually reproduce
    from the record it claims to come from (a hallucinated/paraphrased clause
    is never grounds to eliminate anything), and every cited span must be
    reconciled to a member of `{AGREED, VACUOUS}` — the SAME bar
    `graph_consensus._spans_support` already holds fact-level evidence to. A
    NOT_DOCUMENTED verdict citing no span is permitted — absence has nothing
    to quote by definition — but validating it here is not, on its own,
    sufficient to eliminate anything: `resolution._grounded_elimination`
    additionally requires `req.role` to actually be elimination-eligible
    (`MUST_SUPPORT`/`EXCLUSION`, never `POSITIVE_ALIAS`) before a validated,
    unanimous NOT_DOCUMENTED verdict may eliminate.
    """
    if judgement.requirement_id != req.requirement_id:
        return False
    start, end = req.authority_offset
    if req.authority_source_text[start:end] != req.authority_clause:
        return False
    if judgement.status not in (RequirementStatus.SUPPORTED,
                                RequirementStatus.NOT_DOCUMENTED):
        return False   # CONTRADICTED never validates as a judgement -- see docstring
    if judgement.status is RequirementStatus.NOT_DOCUMENTED:
        if judgement.evidence_span_ids:
            return False
        return deterministic_status(req, coverage) is RequirementStatus.NOT_DOCUMENTED
    # SUPPORTED: every cited span must itself, individually, genuinely and
    # un-negatedly support the term.
    if not judgement.evidence_span_ids or reconciliation is None or evidence_by_span_id is None:
        return False
    from app.contracts.source_evidence import ReconciliationStatus
    settled = reconciliation.by_span_id()
    permitted = {ReconciliationStatus.AGREED, ReconciliationStatus.VACUOUS}
    for sid in judgement.evidence_span_ids:
        if sid not in settled or settled[sid].status not in permitted:
            return False
        span_text = evidence_by_span_id.get(sid)
        if span_text is None:
            return False
        if _tiebreak.asserted_status(req.expected, span_text) != "supported":
            return False
    return True
