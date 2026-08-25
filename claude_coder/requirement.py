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

    @property
    def complete(self) -> bool:
        """Both gates a validated absence needs, together: real text AND no
        page this channel failed to cover. Neither alone is sufficient -- a
        zero-page or fully-uncovered-but-nonempty-flag document must never
        let NOT_DOCUMENTED validate."""
        return bool(self.text) and not self.uncovered_pages

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
                                 "authority": dict(candidate.authority or {})},
                selectable=probe.selectable, queryable=probe.queryable))

    if source is not None:
        resolver = getattr(source, "instructional_terms", None)
        if callable(resolver):
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
                                         "authority": dict(candidate.authority or {})},
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
