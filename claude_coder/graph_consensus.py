"""Two independent readings, compared on GRAPH AXES - and settled by the document
(product directive section 3).

THE RULE THIS MODULE EXISTS TO ENFORCE

    "Two independent models should extract normalized fact graphs, not final codes.
     Compare graph axes: automatically accept a uniquely resolved, high-confidence axis
     supported by source evidence; send a disagreement to a targeted original-page
     verifier; if the note genuinely lacks a code-changing fact, generate a precise
     provider query or exclude the affected line; NEVER route an otherwise resolved
     claim to generic coder review merely because model prose differs."

The second reading is therefore NOT A VOTER. Agreement between two models is never the
deciding rule here (it cannot be: two models share failure modes, and the collaboration
contract already forbids treating agreement as proof). The second reading is a
DISAGREEMENT DETECTOR. What decides is the original document:

  1. ALIGNMENT - two readings of the same event are matched on their normalized token
     content, never on identical wording. Differently-phrased descriptions of the same
     event align, so prose difference produces no disagreement at all and can never
     route anything anywhere. Description text is used ONLY to align; it is never an
     axis, so it can never be a disagreement.
  2. AXIS COMPARISON - only the typed, code-changing axes are compared: occurrence
     status, assertion certainty, beneficiary, and every documented attribute axis the
     two readings between them recorded (anatomy, laterality, count, depth/area/length,
     dose, units, approach, performer - whatever the record stated). The axis set is
     derived from the data, so a new axis is compared the day extraction starts emitting
     it, with no list to update.
  3. ESCALATION - a disagreeing axis goes to the page verifier from directive section 1
     (`app.contracts.source_evidence.reconcile_spans`), aimed at exactly the pages the
     two readings quoted. The reading whose quotations the ORIGINAL PAGE confirms, and
     whose value is literally present in that confirmed quotation, wins.
  4. FALLBACK - when the page cannot settle it, the record genuinely does not state the
     fact, and the outcome is a precise provider query naming the exact axis (or, for a
     non-billable event, exclusion). Never a coder queue.

NO MEDICAL CODE AND NO DOMAIN VOCABULARY APPEARS HERE. Axis names come from the
extraction schema and from the record itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .models import Disposition

#: Identity of this comparison contract.
CONSENSUS_SCHEMA_VERSION = "graph-consensus-v1"

#: Two readings describe the same event when their distinctive-token sets overlap at
#: least this much (Jaccard) AND their kinds match. A TUNING knob for alignment only:
#: it can never release anything. Aligning too eagerly merges two events into one
#: comparison (which produces conflicting-value disagreements, i.e. a query - safe);
#: aligning too reluctantly leaves an event unmatched (recorded, never billed from the
#: second reading - also safe).
ALIGNMENT_THRESHOLD = 0.5

#: Structural axes every fact carries, read off the fact rather than its attributes.
_STRUCTURAL_AXES = ("occurrence_status", "assertion_certainty", "beneficiary")

#: Attribute keys that are not a reading of the document at all: they are resolved from
#: the authoritative encounter context and are therefore identical in both readings by
#: construction. Comparing them would compare the roster to itself.
_CONTEXT_RESOLVED_AXES = ("billing_entity_id",)


class AxisVerdict(str, Enum):
    RESOLVED_FROM_SOURCE = "resolved_from_source"   # the document settled it
    UNRESOLVED = "unresolved"                       # the document does not state it


#: How the winning reading was proved. Recorded so an artifact can never imply the
#: ORIGINAL PAGE settled something when only the transcription was available.
PROOF_ORIGINAL_PAGE = "original_page_reconciliation"
PROOF_ANCHORED_TEXT = "anchored_source_text"


def _norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) > 3}


def _value_tokens(value: str) -> set[str]:
    """Tokens that must be present in a quotation for it to literally SAY this value.

    Deliberately keeps short tokens (a laterality or a count is short) while dropping
    empties - the alignment tokenizer above drops short tokens on purpose, and reusing
    it here would make every short value unprovable.
    """
    return {t for t in re.split(r"[^a-z0-9]+", (value or "").lower()) if t}


def _similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _kind(fact) -> str:
    return str(getattr(getattr(fact, "kind", None), "value", "") or "")


def _axis_values(fact) -> dict[str, str]:
    """Every comparable axis of one reading of one event, as normalized strings.

    Derived from the fact itself, never from a list of axis names - an axis extraction
    starts emitting tomorrow is compared tomorrow.
    """
    out: dict[str, str] = {
        "occurrence_status": _norm(getattr(getattr(fact, "disposition", None), "value", "")),
        "assertion_certainty": _norm(bool(getattr(fact, "certain", True))),
        "beneficiary": _norm(getattr(fact, "experiencer", "")),
    }
    for key, value in (getattr(fact, "attributes", None) or {}).items():
        axis = str(key)
        if axis in _CONTEXT_RESOLVED_AXES or axis in _STRUCTURAL_AXES:
            continue
        out[axis] = _norm(value)
    return out


@dataclass(frozen=True)
class AxisDisagreement:
    """One code-changing axis the two readings did not read the same way."""

    node_id: str                  # the PRIMARY reading event id this concerns
    axis: str
    value_primary: str
    value_second: str
    basis: str                    # conflicting values, or present in one reading only
    action: str = ""              # the primary reading code-free action, for the query text

    def as_record(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "axis": self.axis,
                "value_primary": self.value_primary, "value_second": self.value_second,
                "basis": self.basis}


@dataclass(frozen=True)
class AxisResolution:
    """What the DOCUMENT said about a disagreeing axis."""

    node_id: str
    axis: str
    verdict: AxisVerdict
    accepted_value: str = ""
    accepted_from: str = ""        # primary | second | (empty when unresolved)
    proof: str = ""                # PROOF_ORIGINAL_PAGE | PROOF_ANCHORED_TEXT
    detail: str = ""
    evidence_span_ids: tuple[str, ...] = ()
    #: The precise, self-contained question to send when the document cannot settle it.
    provider_question: str = ""

    @property
    def unresolved(self) -> bool:
        return self.verdict is AxisVerdict.UNRESOLVED

    def as_record(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "axis": self.axis,
                "verdict": self.verdict.value, "accepted_value": self.accepted_value,
                "accepted_from": self.accepted_from, "proof": self.proof,
                "detail": self.detail,
                "evidence_span_ids": list(self.evidence_span_ids),
                "provider_question": self.provider_question}


@dataclass
class ConsensusReport:
    """The full, auditable record of one two-reading comparison."""

    schema_version: str = CONSENSUS_SCHEMA_VERSION
    control_mode: str = "ENFORCED_FAIL_CLOSED"
    second_reading_origin: dict[str, Any] = field(default_factory=dict)
    #: Did the second reading come from a DIFFERENT declared provider? Recorded, never
    #: relied upon: nothing here is decided by the two readings agreeing, so a
    #: same-provider second reading is still a legitimate disagreement detector - but an
    #: artifact must never imply an independence the run did not have.
    independent_providers: bool = False
    matched_events: int = 0
    axes_compared: int = 0
    disagreements: tuple[AxisDisagreement, ...] = ()
    resolutions: tuple[AxisResolution, ...] = ()
    #: Events one reading recorded and the other did not. Recorded, never merged: a
    #: second reading may not add a billable line the primary graph never carried
    #: through eligibility, ownership and anchoring.
    unmatched_primary: tuple[dict[str, Any], ...] = ()
    unmatched_second: tuple[dict[str, Any], ...] = ()
    escalated_pages: tuple[int, ...] = ()
    escalation_detail: str = ""

    @property
    def unresolved(self) -> tuple[AxisResolution, ...]:
        return tuple(r for r in self.resolutions if r.unresolved)

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "control_mode": self.control_mode,
            "second_reading_origin": dict(self.second_reading_origin),
            "independent_providers": self.independent_providers,
            "matched_events": self.matched_events,
            "axes_compared": self.axes_compared,
            "disagreements": [d.as_record() for d in self.disagreements],
            "resolutions": [r.as_record() for r in self.resolutions],
            "unmatched_primary": [dict(u) for u in self.unmatched_primary],
            "unmatched_second": [dict(u) for u in self.unmatched_second],
            "escalated_pages": list(self.escalated_pages),
            "escalation_detail": self.escalation_detail,
        }


# ------------------------------------------------------------------- alignment
def align(primary: list, second: list) -> tuple[list[tuple[Any, Any]], list, list]:
    """Match each second-reading event to the primary event it describes.

    Greedy best-first over same-kind pairs, so the strongest correspondence is taken
    first and nothing is matched twice. Wording is never compared for equality - only
    distinctive-token overlap - which is precisely what stops differing prose from
    becoming a disagreement.
    """
    scored: list[tuple[float, int, int]] = []
    primary_tokens = [_tokens(getattr(f, "description", "")) for f in primary]
    second_tokens = [_tokens(getattr(f, "description", "")) for f in second]
    for i, left in enumerate(primary):
        for j, right in enumerate(second):
            if _kind(left) != _kind(right):
                continue
            score = _similarity(primary_tokens[i], second_tokens[j])
            if score >= ALIGNMENT_THRESHOLD:
                scored.append((score, i, j))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_primary: set[int] = set()
    used_second: set[int] = set()
    pairs: list[tuple[Any, Any]] = []
    for _score, i, j in scored:
        if i in used_primary or j in used_second:
            continue
        used_primary.add(i)
        used_second.add(j)
        pairs.append((primary[i], second[j]))
    unmatched_primary = [f for i, f in enumerate(primary) if i not in used_primary]
    unmatched_second = [f for j, f in enumerate(second) if j not in used_second]
    return pairs, unmatched_primary, unmatched_second


def compare_axes(pairs: list[tuple[Any, Any]]) -> tuple[list[AxisDisagreement], int]:
    """Every code-changing axis the two readings did not read the same way."""
    out: list[AxisDisagreement] = []
    compared = 0
    for left, right in pairs:
        left_axes = _axis_values(left)
        right_axes = _axis_values(right)
        for axis in sorted(set(left_axes) | set(right_axes)):
            a = left_axes.get(axis, "")
            b = right_axes.get(axis, "")
            compared += 1
            if a == b:
                continue
            if a and b:
                basis = "the two readings recorded different values"
            else:
                basis = "only one reading recorded a value"
            out.append(AxisDisagreement(
                node_id=str(getattr(left, "fact_id", "") or ""), axis=axis,
                value_primary=a, value_second=b, basis=basis,
                action=str(getattr(left, "description", "") or "")))
    return out, compared


# ------------------------------------------------------------------ resolution
def _span_support(fact, reconciliation) -> tuple[bool, str, str, tuple[str, ...]]:
    """Is this reading of the event backed by the SOURCE, and what did the source say?

    Returns (supported, proof, verified_text, span_ids). Fail-closed: an event with no
    anchored quotation, or any quotation the original page did not confirm, is NOT
    supported - "we could not check" is never "confirmed".
    """
    spans = [s for s in (getattr(fact, "evidence", None) or [])
             if getattr(s, "anchored", False) and str(getattr(s, "span_id", "") or "")]
    if not spans:
        return False, "", "", ()
    span_ids = tuple(str(s.span_id) for s in spans)
    text = " ".join(str(getattr(s, "text", "") or "") for s in spans)
    if reconciliation is None:
        # No original document accompanied this encounter, so the strongest available
        # proof is the anchored transcription. Recorded AS SUCH; the source-evidence
        # gate separately holds any encounter that DID come from a document.
        return True, PROOF_ANCHORED_TEXT, text, span_ids
    from app.contracts.source_evidence import ReconciliationStatus

    settled = reconciliation.by_span_id()
    permitted = {ReconciliationStatus.AGREED, ReconciliationStatus.VACUOUS}
    for span_id in span_ids:
        outcome = settled.get(span_id)
        if outcome is None or outcome.status not in permitted:
            return False, "", "", span_ids
    return True, PROOF_ORIGINAL_PAGE, text, span_ids


def _question(disagreement: AxisDisagreement) -> str:
    values = [v for v in (disagreement.value_primary, disagreement.value_second) if v]
    read_as = " and ".join(repr(v) for v in values) if values else "nothing"
    return (f"The record does not settle {disagreement.axis!r} for "
            f"{disagreement.action!r}: two independent readings of the original "
            f"document read it as {read_as}, and the page itself does not confirm "
            f"either. Please document {disagreement.axis!r} explicitly for this event.")


def resolve(disagreements: list[AxisDisagreement], primary_by_id: dict,
            second_by_node: dict, reconciliation) -> list[AxisResolution]:
    """Settle each disagreeing axis against the ORIGINAL DOCUMENT, never by vote.

    A reading wins only when the source confirms its quotations AND the other reading is
    not equally confirmed on that axis. Where both readings rest on confirmed quotations,
    the tiebreak is whether the value is LITERALLY PRESENT in the confirmed quotation -
    the document stating it, rather than a model asserting it. Anything else is
    unresolved, which is a provider question, never a coder queue.
    """
    out: list[AxisResolution] = []
    for item in disagreements:
        primary = primary_by_id.get(item.node_id)
        second = second_by_node.get(item.node_id)
        p_ok, p_proof, p_text, p_spans = (
            _span_support(primary, reconciliation) if primary is not None
            else (False, "", "", ()))
        s_ok, s_proof, s_text, s_spans = (
            _span_support(second, reconciliation) if second is not None
            else (False, "", "", ()))

        winner = ""
        proof = ""
        spans: tuple[str, ...] = ()
        detail = ""
        if p_ok and not s_ok:
            winner, proof, spans = "primary", p_proof, p_spans
            detail = ("only the primary reading rests on quotations the source "
                      "confirms")
        elif s_ok and not p_ok:
            winner, proof, spans = "second", s_proof, s_spans
            detail = ("only the second reading rests on quotations the source "
                      "confirms")
        elif p_ok and s_ok:
            p_says = bool(item.value_primary) and _value_tokens(
                item.value_primary).issubset(_value_tokens(p_text))
            s_says = bool(item.value_second) and _value_tokens(
                item.value_second).issubset(_value_tokens(s_text))
            if p_says and not s_says:
                winner, proof, spans = "primary", p_proof, p_spans
                detail = "the confirmed quotation states this value verbatim"
            elif s_says and not p_says:
                winner, proof, spans = "second", s_proof, s_spans
                detail = "the confirmed quotation states this value verbatim"
            else:
                detail = ("both readings rest on confirmed quotations and neither "
                          "value is uniquely stated by them")
        else:
            detail = ("neither reading rests on quotations the source confirms, so "
                      "this event has a source-integrity problem, not a documentation "
                      "gap")

        if winner:
            out.append(AxisResolution(
                node_id=item.node_id, axis=item.axis,
                verdict=AxisVerdict.RESOLVED_FROM_SOURCE,
                accepted_value=(item.value_primary if winner == "primary"
                                else item.value_second),
                accepted_from=winner, proof=proof, detail=detail,
                evidence_span_ids=spans))
        else:
            # PRECEDENCE, and the reason it matters (post-fix review finding). An
            # unsettled axis becomes a provider question ONLY when the readings are
            # genuinely arguing about a fact the record does not state. When NEITHER
            # reading rests on a quotation the original document confirms, the event
            # does not have a documentation gap -- its evidence is contradicted by the
            # page, which is an INTEGRITY STOP owned by the source-evidence control
            # (directive section 1). Asking the provider there would be wrong twice
            # over: it invites a documentation answer to a misreading, and -- because a
            # held event never reaches a claim line -- it would take the misread
            # quotation out of the reach of the very gate that must block it, silently
            # downgrading a BLOCK to a query. So no question is raised, the fact is left
            # untouched, and the control that owns the failure sees it.
            settleable = p_ok or s_ok
            out.append(AxisResolution(
                node_id=item.node_id, axis=item.axis,
                verdict=AxisVerdict.UNRESOLVED, detail=detail,
                evidence_span_ids=tuple(dict.fromkeys(p_spans + s_spans)),
                provider_question=(_question(item) if settleable else "")))
    return out


# --------------------------------------------------------------- applying back
def apply_resolutions(primary_by_id: dict, second_by_node: dict,
                      resolutions: list[AxisResolution]) -> None:
    """Write the DOCUMENT decision back onto the primary graph, in place.

    A value the source confirmed on the second reading CORRECTS the primary fact - and
    the quotation that proved it is carried onto that fact, so the corrected axis stays
    as defensible as every other. An unresolved axis is recorded on the fact as a
    conflict, which the eligibility engine turns into a hold before retrieval and the
    router turns into a targeted provider query.
    """
    for resolution in resolutions:
        fact = primary_by_id.get(resolution.node_id)
        if fact is None:
            continue
        if resolution.unresolved:
            # An unresolved axis with no question is one the SOURCE-EVIDENCE control
            # owns (see `resolve`): recording a conflict here would hold the event
            # before retrieval and hide the misreading from that control.
            if not resolution.provider_question:
                continue
            conflicts = list(getattr(fact, "axis_conflicts", None) or [])
            if resolution.provider_question not in conflicts:
                conflicts.append(resolution.provider_question)
            fact.axis_conflicts = conflicts
            continue
        if resolution.accepted_from != "second":
            continue                      # the primary reading already holds this value
        _write_axis(fact, resolution.axis, resolution.accepted_value)
        source_fact = second_by_node.get(resolution.node_id)
        if source_fact is not None:
            _carry_evidence(fact, source_fact, resolution.evidence_span_ids)


def _write_axis(fact, axis: str, value: str) -> None:
    """Set one axis from its normalized string form, or refuse.

    Fail-closed on the typed axes: a value the type does not recognize is NOT written and
    NOT silently coerced - the axis simply keeps the primary reading, and (because the
    axis only reached here after disagreeing) the record still carries the disagreement.
    """
    if axis == "occurrence_status":
        try:
            fact.disposition = Disposition(value)
        except ValueError:
            return
        return
    if axis == "assertion_certainty":
        if value in ("true", "false"):
            fact.certain = (value == "true")
        return
    if axis == "beneficiary":
        if value:
            fact.experiencer = value
        return
    attributes = dict(getattr(fact, "attributes", None) or {})
    if value:
        attributes[axis] = value
    else:
        attributes.pop(axis, None)
    fact.attributes = attributes


def _carry_evidence(fact, source_fact, span_ids: tuple[str, ...]) -> None:
    have = {str(getattr(s, "span_id", "") or "") for s in (fact.evidence or [])}
    wanted = set(span_ids)
    for span in (getattr(source_fact, "evidence", None) or []):
        span_id = str(getattr(span, "span_id", "") or "")
        if span_id and span_id in wanted and span_id not in have:
            fact.evidence.append(span)
            have.add(span_id)


def _event_record(fact) -> dict[str, Any]:
    return {"node_id": str(getattr(fact, "fact_id", "") or ""),
            "kind": _kind(fact),
            "action": str(getattr(fact, "description", "") or ""),
            "billable": bool(getattr(fact, "billable", False)),
            "anchored": any(getattr(s, "anchored", False)
                            for s in (getattr(fact, "evidence", None) or []))}


def disagreement_span_ids(disagreements: list[AxisDisagreement], primary_by_id: dict,
                          second_by_node: dict) -> set[str]:
    """Exactly the quotations a targeted original-page read has to cover.

    This is what makes the escalation TARGETED rather than a second read of the whole
    document: only the pages carrying the quotations behind a disagreeing axis.
    """
    wanted: set[str] = set()
    for item in disagreements:
        for source in (primary_by_id.get(item.node_id), second_by_node.get(item.node_id)):
            for span in (getattr(source, "evidence", None) or []) if source else ():
                span_id = str(getattr(span, "span_id", "") or "")
                if span_id and getattr(span, "anchored", False):
                    wanted.add(span_id)
    return wanted


def compare(primary_facts: list, second_facts: list, *,
            second_origin: Any = None) -> tuple[ConsensusReport, dict, dict]:
    """Align two readings and list every code-changing axis they disagree on.

    Returns the report plus the two id-keyed views the resolver needs, so the caller can
    escalate to the page verifier between comparison and resolution.
    """
    pairs, unmatched_primary, unmatched_second = align(primary_facts, second_facts)
    disagreements, compared = compare_axes(pairs)
    primary_by_id = {str(getattr(f, "fact_id", "") or ""): f for f in primary_facts}
    second_by_node = {str(getattr(left, "fact_id", "") or ""): right
                      for left, right in pairs}
    report = ConsensusReport(
        second_reading_origin=(second_origin.as_record()
                               if hasattr(second_origin, "as_record") else {}),
        matched_events=len(pairs),
        axes_compared=compared,
        disagreements=tuple(disagreements),
        unmatched_primary=tuple(_event_record(f) for f in unmatched_primary),
        unmatched_second=tuple(_event_record(f) for f in unmatched_second),
    )
    return report, primary_by_id, second_by_node
