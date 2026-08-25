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
                source_identity={"kind": "descriptor", "system": candidate.system},
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
                                         "system": candidate.system},
                        selectable=True, queryable=False))
    return tuple(out)


def validated_requirement(req: DescriptorRequirement, judgement: RequirementJudgement,
                          reconciliation: Any = None) -> bool:
    """Does an evaluator's judgement of `req` deserve to affect selection?

    The clause must actually reproduce from the record it claims to come from (a
    hallucinated/paraphrased clause is never grounds to eliminate anything), and
    every cited span must be reconciled to a member of `{AGREED, VACUOUS}` — the
    SAME bar `graph_consensus._spans_support` already holds fact-level evidence to.
    A NOT_DOCUMENTED verdict citing no span is permitted — absence has nothing to
    quote by definition — but validating it here is not, on its own, sufficient to
    eliminate anything: `resolution._grounded_elimination` (issue #6 F9-R6 Phase 2)
    never wires NOT_DOCUMENTED into elimination without a "complete page coverage"
    signal this module deliberately does not invent (explicitly deferred to a
    future round — trusting a model's self-reported coverage would repeat the exact
    mistake issue #6 F8-R1 already found and closed for a different mechanism).
    """
    if judgement.requirement_id != req.requirement_id:
        return False
    start, end = req.authority_offset
    if req.authority_source_text[start:end] != req.authority_clause:
        return False
    if judgement.status is RequirementStatus.NOT_DOCUMENTED and not judgement.evidence_span_ids:
        return True
    if not judgement.evidence_span_ids:
        return False
    if reconciliation is None:
        return False
    from app.contracts.source_evidence import ReconciliationStatus
    settled = reconciliation.by_span_id()
    permitted = {ReconciliationStatus.AGREED, ReconciliationStatus.VACUOUS}
    return all(sid in settled and settled[sid].status in permitted
              for sid in judgement.evidence_span_ids)
