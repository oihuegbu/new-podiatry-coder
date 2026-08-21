"""Semantic eligibility-before-retrieval (issue #6 items 4/5): narrow retrieval's
broad candidate pool to codes whose COMPILED SEMANTIC RECORD (`claude_coder.semantics`,
built entirely from authoritative data) positively CONFLICTS with what the fact's own
documentation states -- never a hardcoded code family, never a full-table scan, and
never a disqualification for an axis neither side documents. Absence of a constraint
is not a violation of one (the same principle `semantics.compiled_record` itself
follows): a candidate is excluded only when a compiled, deterministic field says
something that contradicts the fact, never on a lexical hunch.

Scope, honestly narrower than the plan's five-axis list, and deliberately so:
`action_concepts`/`anatomy_concepts` (derived from a CODE's own comma-structured
"Action, target" descriptor grammar, `ontology.parse_descriptor`) have no equally
reliable counterpart on the FACT side -- a fact's `description` is natural prose
extracted from the note, not a formal descriptor, so it essentially never carries
the same punctuation. Comparing the fact's whole-text vocabulary against a
candidate's action/anatomy tokens instead would be a bare LEXICAL overlap check,
and lexical overlap cannot tell a genuine mismatch from a clinical synonym
("exostosis" vs. "spur", "excision" vs. "resection") -- on a live billing pipeline,
a false exclusion there means a real service silently drops out of the candidate
pool. That axis is left for a follow-up that first routes the fact's text through
the governed procedure-synonym axis (`data_access.concept_lookup("procedure", ...)`)
so synonyms normalize before comparison, rather than shipped as a half-safe lexical
guess now. What DOES ship here is deterministic and synonym-proof: a documented
measurement/interval requirement, a semantic-class conflict, and code activity on
the date of service.

Scope note on grouping: this filters retrieval's already-returned candidate list
(`resolution.py`'s broad `source.retrieve` pool), one fact at a time. Grouping
multiple facts of one `composition.ServiceIntent` into a single eligibility check
is a natural next step once retrieval requests are built per-intent rather than
per-fact -- not yet true of `eligibility.RetrievalRequest`, so this module accepts
whatever fact list its caller has (today, always one).
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


def _documented_interval(facts: list[ClinicalFact]):
    """The bounded measurement/interval this documentation states, or None -- reuses
    `ontology`'s own interval detector (unconditional on descriptor punctuation, so
    it applies equally well to a fact's natural-prose description) rather than a
    second implementation."""
    text = " ".join(f.description for f in facts if f.description)
    feats = _ontology.parse_descriptor(text)
    return feats.interval if feats.interval and feats.interval.bounded() else None


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

    if "measurement" in (record.get("required_attributes") or []) and \
            _documented_interval(facts) is None:
        return ("candidate's descriptor requires a documented measurement/interval "
               "the fact's text does not state")

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
                       date_of_service: str | None) -> list:
    """`candidates`, narrowed to the ones `eligible()` accepts for `facts`.

    Narrowing ONLY: if every candidate happens to fail (a sign the filter itself
    does not fit this case, not evidence that nothing retrieved is usable), the
    original, unfiltered list is returned rather than an empty one -- this
    mechanism may never single-handedly turn a fact retrieval FOUND candidates for
    into one that finds none at all."""
    if not candidates:
        return candidates
    kept = [c for c in candidates if eligible(c, facts, source, date_of_service)]
    return kept if kept else candidates


def eligibility_report(facts: list[ClinicalFact], candidates: list, source,
                       date_of_service: str | None) -> list[dict]:
    """A full per-candidate audit record over `candidates` -- which `eligible_partition`
    would keep or exclude, and why -- preserved for the audit trail EVEN on a
    held/blocked outcome (issue #6 item 8), never only for a released line. Computed
    over the SAME `_ineligibility_reason` `eligible_partition` itself reads, so this
    record can never claim a different reason than the one that actually decided it."""
    report = []
    for c in candidates:
        reason = _ineligibility_reason(c, facts, source, date_of_service)
        report.append({"code": c.code, "system": c.system, "eligible": reason is None,
                       "reason": reason})
    return report
