"""Descriptor/instructional-note requirement compilation and validation (issue #6
F9-R6) — the typed, source-grounded channel a candidate's own authoritative record
exposes for elimination, widening WHICH AXIS KINDS may eliminate a candidate without
ever loosening WHAT COUNTS as reaching that bar.

This module sits strictly downstream of `tiebreak.discriminating_axes` — it is a
mechanical PROJECTION of each `AxisProbe` into one or more `DescriptorRequirement`
records per candidate, never an independent axis compiler. `required = probe.
selectable`, so mandatory maps exactly onto today's selectable/non-selectable line:
an axis that could never eliminate a candidate before this module existed still
cannot after it. Only `provable` axes (a quotation's WORDS can settle them) compile
into text-clause requirements at all — `AXIS_MEASUREMENT` (a typed, unit-converted
interval comparison, never provable by words) is deliberately not compiled here;
`resolution._grounded_elimination` reasons about it directly against `fact.
attributes` via the existing `_measure_in_range`/`_interval_unsupported`, never
through a fabricated text clause.

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


@dataclass(frozen=True)
class DescriptorRequirement:
    """One typed, source-anchored fact a candidate's own authoritative record
    states — compiled from an `AxisProbe` `tiebreak.discriminating_axes` already
    derived, never a separately invented axis. `required` mirrors the originating
    probe's `selectable`: only an axis that could already eliminate/select a
    candidate before this module existed can be `required` here.
    """
    requirement_id: str
    axis: str
    candidate_code: str
    required: bool
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
                        required=True, expected=(term,), authority_clause=clause,
                        authority_offset=offset, authority_source_text=term,
                        source_identity={"kind": "instructional_notes",
                                         "system": candidate.system,
                                         "authority": dict(candidate.authority or {})},
                        selectable=True, queryable=False))
    return tuple(out)


def deterministic_status(req: DescriptorRequirement, searchable_text: str
                         ) -> RequirementStatus | None:
    """The TRUE relation between `req.expected` and `searchable_text`, found by
    ACTUALLY SEARCHING the text -- never a verifier's self-report standing in for
    it (issue #6 F9-R6-R2 re-review). Uses the SAME contiguous, ordered phrase
    matcher `tiebreak.narrow` proves its own axes with (`tiebreak._token_sequence`/
    `_phrase_present`), so "the document states this" means the same thing
    wherever it's checked in this codebase.

    Returns `None` when `searchable_text` is empty: no claim, positive OR
    negative, can be made about a document nobody supplied -- the caller
    (`validated_requirement`) must fail closed on `None`, exactly as
    `document_fully_covered=False` already fails closed on no coverage proof.

    Only ever returns SUPPORTED or NOT_DOCUMENTED. CONTRADICTED is not
    derivable here (see `validated_requirement`'s docstring for why) and
    UNRESOLVED is not a claim this function is positioned to make -- absence
    from `searchable_text` is exactly what NOT_DOCUMENTED means.
    """
    if not searchable_text:
        return None
    tokens = _tiebreak._token_sequence(searchable_text)
    present = any(_tiebreak._phrase_present(term, tokens) for term in req.expected)
    return RequirementStatus.SUPPORTED if present else RequirementStatus.NOT_DOCUMENTED


def validated_requirement(req: DescriptorRequirement, judgement: RequirementJudgement,
                          reconciliation: Any = None, searchable_text: str = "") -> bool:
    """Does an evaluator's judgement of `req` deserve to affect selection?

    issue #6 F9-R6-R2 (Codex re-review): the ORIGINAL version of this function
    checked only that a cited span was REAL and page-reconciled -- never that the
    span's actual CONTENT related to `req.expected` at all. A model could cite any
    real, unrelated, agreed span and claim CONTRADICTED, and this function
    returned True. The shipped Phase-2 positive-path test proved this empirically:
    it cited deliberately neutral text as "proof" of a contradiction, and
    validation passed.

    The fix is two-fold, and both parts are PERMANENT design decisions, not
    interim patches:

    1. CONTRADICTED is retired as elimination grounds, for every axis,
       unconditionally -- see the `if judgement.status not in (...)` check below.
       Confirming a phrase-type requirement is genuinely CONTRADICTED (the note
       actively states the opposite, not merely that it's silent) needs real
       negation-detection this codebase does not have and building it would be
       exactly the kind of unsound guess this module exists to prevent. Laterality
       is the one axis with a real closed-enumeration contradiction ("left" stated
       when the candidate needs "right") -- that is already, and remains, fully
       handled by the pre-existing, judgement-INDEPENDENT `tiebreak.narrow`
       document-proof fallback in `resolution._grounded_elimination`, so nothing
       is lost by retiring the judgement path for it too.
    2. SUPPORTED and NOT_DOCUMENTED must now be independently, deterministically
       confirmed by `deterministic_status` actually searching `searchable_text` --
       the verifier's own status becomes a mere PROPOSAL that this function either
       confirms or refuses; it is never trusted on its own. `searchable_text`
       empty (no real corpus supplied) means neither status can ever validate --
       fails closed, matching `document_fully_covered=False`'s existing default-
       refuse posture. This closes a related gap (issue #6 F9-R6-R4): a NOT_
       DOCUMENTED verdict from a verifier shown only `fact.evidence` (this one
       fact's own narrow excerpt) is a claim about that excerpt, not about the
       whole document -- `searchable_text` must be the full, independently-read
       document text (see `pipeline.code_encounter`'s call site) for a NOT_
       DOCUMENTED verdict to mean what it claims.

    On top of the content check: the clause must still actually reproduce from
    the record it claims to come from (a hallucinated/paraphrased clause is never
    grounds to eliminate anything), and every cited span must be reconciled to a
    member of `{AGREED, VACUOUS}` — the SAME bar `graph_consensus._spans_support`
    already holds fact-level evidence to. A NOT_DOCUMENTED verdict citing no span
    is permitted — absence has nothing to quote by definition — but validating it
    here is not, on its own, sufficient to eliminate anything:
    `resolution._grounded_elimination` additionally requires
    `document_fully_covered=True` AND a non-empty `searchable_text` before a
    validated, unanimous NOT_DOCUMENTED verdict may eliminate.
    """
    if judgement.requirement_id != req.requirement_id:
        return False
    start, end = req.authority_offset
    if req.authority_source_text[start:end] != req.authority_clause:
        return False
    if judgement.status not in (RequirementStatus.SUPPORTED,
                                RequirementStatus.NOT_DOCUMENTED):
        return False   # CONTRADICTED retired -- see docstring above
    truth = deterministic_status(req, searchable_text)
    if truth is None or truth is not judgement.status:
        return False
    if judgement.status is RequirementStatus.NOT_DOCUMENTED:
        return not judgement.evidence_span_ids
    if not judgement.evidence_span_ids:
        return False
    if reconciliation is None:
        return False
    from app.contracts.source_evidence import ReconciliationStatus
    settled = reconciliation.by_span_id()
    permitted = {ReconciliationStatus.AGREED, ReconciliationStatus.VACUOUS}
    return all(sid in settled and settled[sid].status in permitted
              for sid in judgement.evidence_span_ids)
