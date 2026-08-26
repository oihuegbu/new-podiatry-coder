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

from .models import Disposition, RelationState

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


def _axis_value_signature(fact, axis: str, value: Any) -> str:
    """The axis value AS CLAIMED BY THIS READING, for cross-reader comparison
    (issue #6 F9-R6-R2, sixth re-review) -- downgraded to "" when this reading's
    OWN attribute_evidence, value-bound to this exact value, is entirely
    NEGATED/UNCERTAIN (never ASSERTED). Without this, two readings writing the
    identical raw string but disagreeing on whether their OWN evidence actually
    asserts it ("right"/ASSERTED vs "right"/NEGATED) compared as EQUAL, so no
    `AxisDisagreement` was ever raised and the disagreement-resolution machinery
    this whole effort built never ran for exactly the case it exists for.

    Deliberately does NOT check source reconciliation -- unavailable this early
    (before relations/spans are reconciled against the original document), and
    a stronger, LATER check (`claim_authorized_value`) every real consumer goes
    through regardless of whether a disagreement was ever raised here. This is
    only a reading's own INTERNAL consistency between what it claims and what
    its own evidence says about that claim.

    Used ONLY to decide whether two readings' surface-identical raw values
    should still count as a genuine disagreement (`compare_axes` below) -- NEVER
    used as the value stored on `AxisDisagreement` itself, which must stay the
    real raw value for `resolve()`'s own per-reading assertion check to
    evaluate. Conflating the two was tried and reverted during this same round:
    it left `resolve()` receiving `""` instead of the real claimed value for a
    self-negated reading, which broke its own value-acceptance logic instead of
    feeding it a genuine, evaluable disagreement."""
    norm_value = _norm(value)
    if not norm_value:
        return ""
    entries = (getattr(fact, "attribute_evidence", None) or {}).get(axis)
    if not entries:
        return norm_value
    bound = [e for e in entries if _norm(e.value) == norm_value]
    if not bound:
        return norm_value
    if any(e.assertion_state == RelationState.ASSERTED for e in bound):
        return norm_value
    return ""


def _axis_values(fact) -> dict[str, str]:
    """Every comparable axis of one reading of one event, as normalized strings.

    Derived from the fact itself, never from a list of axis names - an axis extraction
    starts emitting tomorrow is compared tomorrow. The RAW value, unmodified by
    assertion state -- see `_axis_value_signature` for the separate, signature-
    aware check `compare_axes` additionally applies before treating two equal
    raw values as a genuine agreement.
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
    #: Did the second reading come from a DIFFERENT provider, as declared by the
    #: callable that made it? Nothing here is decided by the two readings AGREEING, so a
    #: same-provider second reading is still a legitimate disagreement detector and is
    #: allowed - but an artifact must never imply an independence the run did not have.
    #: When the pipeline enables the second reading as its OWN independence control
    #: (`independence_enforced`), a pair that is not positively different is refused
    #: before either reading is taken and no report is produced at all (issue #6 F7-R5).
    independent_providers: bool = False
    #: Was independence a PRECONDITION of this run (the pipeline's control) or an
    #: observation about a caller-supplied second extractor? The two are different
    #: claims and a reader of the record must be able to tell them apart.
    independence_enforced: bool = False
    matched_events: int = 0
    axes_compared: int = 0
    disagreements: tuple[AxisDisagreement, ...] = ()
    resolutions: tuple[AxisResolution, ...] = ()
    #: Events one reading recorded and the other did not, as the ALIGNMENT saw them.
    #: `unmatched_second` is the raw input to the event-candidate union below, not a
    #: verdict: what became of each of those events is `recovered_events`.
    unmatched_primary: tuple[dict[str, Any], ...] = ()
    unmatched_second: tuple[dict[str, Any], ...] = ()
    #: The EVENT-CANDIDATE UNION's verdict on every event only the second reading found
    #: (`claude_coder.event_union`): admitted into the canonical graph, recognised as a
    #: reworded duplicate of a primary event, excluded because the original page
    #: contradicts it, or held because nothing could confirm it. Before this existed, an
    #: event the primary extractor missed was recorded here and silently dropped, and the
    #: encounter under-coded with no integrity complaint (issue #6 F7-R3).
    recovered_events: tuple[dict[str, Any], ...] = ()
    #: WHICH READING the recall extraction was actually run over. Empty means it was run
    #: over the primary transcription -- the pre-F7-R3 behaviour, which can only recover
    #: what the primary EXTRACTION missed, never what the TRANSCRIPTION missed. A reader
    #: of this record must be able to tell those two situations apart.
    recall_reading_channel_id: str = ""
    #: Pages no independent reading covered, so the recall extraction never saw them.
    #: This is the control's own blind spot, stated rather than left to be inferred from
    #: an empty `recovered_events`: on these pages "nothing extra was found" and "nothing
    #: was looked at" are indistinguishable, and only one of them is evidence.
    recall_uncovered_pages: tuple[int, ...] = ()
    #: Pages an independent vision channel was proactively read for, BEFORE extraction,
    #: because no other channel covered them (issue #6 F7-R3, round-9 re-review: reading
    #: them only later, to verify a quotation a candidate event already rested on, cannot
    #: recover a service the primary transcription omitted on such a page in the first
    #: place). Disjoint from `recall_uncovered_pages` by construction -- a page here was
    #: successfully covered and so is no longer counted as uncovered.
    recall_page_read_pages: tuple[int, ...] = ()
    recall_page_read_detail: str = ""
    #: Durable, per-page provenance for every uncovered page EXEMPTED from the
    #: recall-coverage gate as genuinely blank: the exempting channel and its own
    #: validated `detail`/`text_sha256` (Codex F7-R3-A, exact-SHA re-review, fourth
    #: pass -- BLANK, MISSING and UNREADABLE must never be byte-equivalent in the
    #: record; naming the specific read that justified an exemption, not just a
    #: boolean gate outcome, is what makes that distinction durable).
    recall_blank_pages: tuple[dict[str, Any], ...] = ()
    escalated_pages: tuple[int, ...] = ()
    escalation_detail: str = ""
    #: Every axis pair a governed concept source (issue #6 F7-R3-C4) confirmed SAME
    #: across the two readings -- the raw wording on each side, the axis, and (when the
    #: source can supply it) the concept identity/method/confidence behind that
    #: confirmation. Recorded even though no `AxisDisagreement` is raised for these
    #: pairs: an autonomy outcome a governed match changed must be traceable from the
    #: audit trail, not merely absent from the disagreement list (Codex F7-R3-C4,
    #: exact-SHA re-review: "the decisive terminology action is not defensible from
    #: the final audit").
    governed_matches: tuple[dict[str, Any], ...] = ()

    @property
    def unresolved(self) -> tuple[AxisResolution, ...]:
        return tuple(r for r in self.resolutions if r.unresolved)

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "control_mode": self.control_mode,
            "second_reading_origin": dict(self.second_reading_origin),
            "independent_providers": self.independent_providers,
            "independence_enforced": self.independence_enforced,
            "matched_events": self.matched_events,
            "axes_compared": self.axes_compared,
            "disagreements": [d.as_record() for d in self.disagreements],
            "resolutions": [r.as_record() for r in self.resolutions],
            "unmatched_primary": [dict(u) for u in self.unmatched_primary],
            "unmatched_second": [dict(u) for u in self.unmatched_second],
            "recovered_events": [dict(u) for u in self.recovered_events],
            "recall_reading_channel_id": self.recall_reading_channel_id,
            "recall_uncovered_pages": list(self.recall_uncovered_pages),
            "recall_page_read_pages": list(self.recall_page_read_pages),
            "recall_page_read_detail": self.recall_page_read_detail,
            "recall_blank_pages": [dict(p) for p in self.recall_blank_pages],
            "escalated_pages": list(self.escalated_pages),
            "escalation_detail": self.escalation_detail,
            "governed_matches": [dict(m) for m in self.governed_matches],
        }


# ------------------------------------------------------------------- alignment
def align(primary: list, second: list, source: Any = None
         ) -> tuple[list[tuple[Any, Any]], list, list]:
    """Match each second-reading event to the primary event it describes.

    TWO tiers, in priority order:

    1. GOVERNED (issue #6 F9-R4): when `source` supplies a SNOMED Procedure concept
       graph, a pair whose actions uniquely resolve to the SAME procedure concept
       (`coreference.action_relation_detail`) is a source-anchored, highest-priority
       match -- this is what lets a note's own summary line align with its detailed
       procedure narrative for the same step even though the two share little
       distinctive vocabulary. Only forces a match when the governed verdict names
       EXACTLY ONE same-kind counterpart on each side; an action that governed-
       matches more than one still-unmatched candidate is left unresolved by this
       tier rather than guessed at (never both, never an arbitrary pick).
    2. LEXICAL (unchanged): greedy best-first Jaccard overlap over same-kind pairs
       neither side of which tier 1 already matched, so the strongest remaining
       correspondence is taken first and nothing is matched twice. Wording is never
       compared for equality - only distinctive-token overlap - which is precisely
       what stops differing prose from becoming a disagreement. This tier is now an
       OBSERVATIONAL tiebreak among pairs tier 1 did not resolve, never semantic
       proof on its own (Codex F9-R4).
    """
    used_primary: set[int] = set()
    used_second: set[int] = set()
    pairs: list[tuple[Any, Any]] = []

    if source is not None:
        from . import coreference as _coref
        governed: set[tuple[int, int]] = set()
        for i, left in enumerate(primary):
            for j, right in enumerate(second):
                if _kind(left) != _kind(right):
                    continue
                verdict, _detail = _coref.action_relation_detail(
                    getattr(left, "description", ""),
                    getattr(right, "description", ""), source)
                if verdict == _coref.SAME_EVENT:
                    governed.add((i, j))
        by_primary: dict[int, int] = {}
        by_second: dict[int, int] = {}
        for i, j in governed:
            by_primary[i] = by_primary.get(i, 0) + 1
            by_second[j] = by_second.get(j, 0) + 1
        for i, j in sorted(governed):
            if i in used_primary or j in used_second:
                continue
            if by_primary[i] == 1 and by_second[j] == 1:
                used_primary.add(i)
                used_second.add(j)
                pairs.append((primary[i], second[j]))

    scored: list[tuple[float, int, int]] = []
    primary_tokens = [_tokens(getattr(f, "description", "")) for f in primary]
    second_tokens = [_tokens(getattr(f, "description", "")) for f in second]
    for i, left in enumerate(primary):
        if i in used_primary:
            continue
        for j, right in enumerate(second):
            if j in used_second:
                continue
            if _kind(left) != _kind(right):
                continue
            score = _similarity(primary_tokens[i], second_tokens[j])
            if score >= ALIGNMENT_THRESHOLD:
                scored.append((score, i, j))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _score, i, j in scored:
        if i in used_primary or j in used_second:
            continue
        used_primary.add(i)
        used_second.add(j)
        pairs.append((primary[i], second[j]))
    unmatched_primary = [f for i, f in enumerate(primary) if i not in used_primary]
    unmatched_second = [f for j, f in enumerate(second) if j not in used_second]
    return pairs, unmatched_primary, unmatched_second


def compare_axes(pairs: list[tuple[Any, Any]],
                 source: Any = None
                 ) -> tuple[list[AxisDisagreement], int, list[dict[str, Any]]]:
    """Every code-changing axis the two readings did not read the same way, plus
    every axis a governed concept source POSITIVELY confirmed as the same value
    despite differing wording.

    `source` (issue #6 F7-R3-C4) is put to the SAME governed axis-relation mechanic
    claim assembly uses (`coreference.axis_relation`) before a raw string mismatch is
    recorded as a disagreement: two readings worded as a confirmed-SAME concept on a
    governed axis (anatomy) are not a disagreement at all, exactly as two readings
    worded identically never were. This is the choke point where a genuine synonym
    pair used to become an unsettleable cross-reading conflict -- routed to
    `eligibility._gate_axis_consensus` as AUTO_HOLD -- before claim assembly's own
    concept-aware comparison ever ran, because this comparison is upstream of it and
    used only raw string equality. A relation the graph cannot confirm (ancestor/
    descendant, ambiguous overlap, or unresolved) still raises a disagreement,
    exactly as before -- this only removes disagreements the graph POSITIVELY
    confirms are not real, never adds new tolerance beyond that.

    Every axis suppressed this way is recorded in the third return value (issue #6
    F7-R3-C4, exact-SHA re-review: a verified expansion must not "remove a hold
    without improving candidate recall", and "the decisive terminology action" must
    be "defensible from the final audit"). The caller applies these onto the
    surviving fact (`ClinicalFact.governed_terms`) so retrieval can query under the
    confirmed alternate wording too, and binds the raw record into the consensus
    report for audit visibility.
    """
    from . import coreference as _coref
    out: list[AxisDisagreement] = []
    governed: list[dict[str, Any]] = []
    compared = 0
    for left, right in pairs:
        left_axes = _axis_values(left)
        right_axes = _axis_values(right)
        for axis in sorted(set(left_axes) | set(right_axes)):
            a = left_axes.get(axis, "")
            b = right_axes.get(axis, "")
            compared += 1
            if a == b:
                # issue #6 F9-R6-R2, sixth re-review: equal RAW strings are not
                # automatically an agreement -- "right"/ASSERTED and "right"/
                # NEGATED must not silently pass as the same claim, or the
                # disagreement-resolution machinery this whole effort built
                # never runs for exactly the case it exists for. `a`/`b`
                # themselves stay the real raw values below regardless (never
                # replaced by the signature), so `resolve()` still evaluates
                # the genuine claimed value, not a placeholder.
                if _axis_value_signature(left, axis, a) == _axis_value_signature(right, axis, b):
                    continue
                # Equal raw strings, but the two readings' OWN evidence disagrees on
                # whether the value is actually asserted -- raised directly, NEVER
                # routed through `axis_relation_detail` below: that function exists
                # to ask "are two DIFFERENT-LOOKING strings the same concept", and
                # for an enumerated axis with IDENTICAL canonical values it always
                # answers SAME_EVENT trivially (line ~307 above), which would
                # silently swallow this exact case into `governed` and never reach
                # `resolve()` at all -- exactly the gap this check exists to close.
                out.append(AxisDisagreement(
                    node_id=str(getattr(left, "fact_id", "") or ""), axis=axis,
                    value_primary=a, value_second=b,
                    basis=("the two readings recorded the same value but disagree "
                          "on whether their own evidence actually asserts it"),
                    action=str(getattr(left, "description", "") or "")))
                continue
            # ONE atomic call for both the claim decision and the audit detail (Codex
            # F7-R3-C4, exact-SHA re-review, eighth pass): the verdict this promotes on
            # and the detail the consensus record names must come from the SAME source
            # response, never two independent calls that could disagree.
            verdict, detail = (_coref.axis_relation_detail(axis, a, b, source)
                               if a and b else (_coref.UNDETERMINED, {}))
            if verdict == _coref.SAME_EVENT:
                span_ids = tuple(dict.fromkeys(
                    str(getattr(s, "span_id", "") or "")
                    for f in (left, right)
                    for s in (getattr(f, "evidence", None) or [])
                    if getattr(s, "span_id", None)))
                governed.append({
                    "node_id": str(getattr(left, "fact_id", "") or ""),
                    "axis": axis, "value_primary": a, "value_second": b,
                    "evidence_span_ids": list(span_ids), **detail})
                continue
            if a and b:
                basis = "the two readings recorded different values"
            else:
                basis = "only one reading recorded a value"
            out.append(AxisDisagreement(
                node_id=str(getattr(left, "fact_id", "") or ""), axis=axis,
                value_primary=a, value_second=b, basis=basis,
                action=str(getattr(left, "description", "") or "")))
    return out, compared, governed


# ------------------------------------------------------------------ resolution
def _spans_support(spans: list, reconciliation) -> tuple[bool, str, str, tuple[str, ...]]:
    """The shared mechanic behind `_span_support`/`_attribute_span_support`: is THIS
    list of spans backed by the SOURCE, and what did the source say?

    Returns (supported, proof, verified_text, span_ids). Fail-closed: no anchored
    span, or any quotation the original page did not confirm, is NOT supported --
    "we could not check" is never "confirmed".
    """
    spans = [s for s in spans
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


def _span_support(fact, reconciliation) -> tuple[bool, str, str, tuple[str, ...]]:
    """Is this reading of the EVENT backed by the SOURCE, and what did the source say?
    See `_spans_support` for the mechanic; this scopes it to the fact's whole
    (undifferentiated) evidence pool."""
    return _spans_support(getattr(fact, "evidence", None) or [], reconciliation)


def _attribute_span_support(fact, attr_name: str, value: str, reconciliation
                            ) -> tuple[bool, str, str, tuple[str, ...]] | None:
    """Per-attribute proof (issue #6 F9-R5): the SAME shape as `_span_support`, scoped
    to evidence anchored specifically to `attr_name` rather than the fact's whole
    evidence pool -- so a value stated once in a scoped heading/parent event can be
    proven without one of its tokens happening to also appear elsewhere in the fact's
    unrelated evidence text. Returns `None` (not a tuple) when the fact has no USABLE
    `attribute_evidence` for this attribute at all, so the caller falls back to
    `_span_support`'s whole-fact-text check -- a strictly ADDITIVE signal, never a
    narrowing of what already worked.

    A "local" entry needs no further check -- its evidence is the fact's own sentence.
    An "inherited" entry is IGNORED here unless `scope_validated` is True (issue #6
    F9-R5-A): the extraction-time candidate relation match is provisional, and only
    `provenance.validate_attribute_evidence` -- run after the relations graph is fully
    reconciled -- may promote it. An unvalidated inherited entry is treated as if it
    were never emitted, not as a weaker signal.

    issue #6 F9-R6-R2, sixth re-review: `value` scopes this to entries VALUE-BOUND
    to it (an entry whose own `.value` canonically matches `value`) when any exist,
    so the reconciliation check below can never be satisfied by evidence that
    proves an entirely different, unrelated value -- falls back to every usable
    entry regardless of value only when NONE of them are value-bound to `value` at
    all (an axis with no entries even talking about this value, which is the same
    "no proof either way" shape as no evidence at all, not a narrowing).

    `ok` reflects SOURCE RECONCILIATION of the best-available usable quotation(s),
    regardless of assertion state (issue #6 F9-R6-R2, fifth re-review) -- deliberately
    NOT gated on whether the value is actually asserted, so a quotation the model
    itself marked NEGATED but which the source genuinely confirms is reported as
    CONFIRMED (a real documentation state -- the note explicitly rules the value
    out), never as an unconfirmed SOURCE-INTEGRITY problem, which is a different,
    more specific claim reserved for a quotation the source does not back at all.
    Prefers ASSERTED entries' spans when any exist (the strongest, most directly
    citable proof); falls back to every usable (value-bound-if-any) entry's spans
    otherwise, purely for this reconciliation check -- never as a route to accepting
    the value itself. `resolve()` gates ACCEPTANCE separately via
    `asserted_attribute_support` below, which is the only function that may say a
    value is genuinely proven.
    """
    entries = (getattr(fact, "attribute_evidence", None) or {}).get(attr_name)
    if not entries:
        return None
    usable = [e for e in entries if e.scope == "local" or e.scope_validated]
    if not usable:
        return None
    norm_value = _norm(value)
    bound = [e for e in usable if _norm(e.value) == norm_value] if norm_value else []
    pool = bound or usable
    asserted = [e for e in pool if e.assertion_state == RelationState.ASSERTED]
    return _spans_support([e.span for e in (asserted or pool)], reconciliation)


def asserted_attribute_support(fact, attr_name: str, value: str, reconciliation) -> bool:
    """Whether `fact`'s OWN `attribute_evidence` for `attr_name` contains at least one
    genuinely ASSERTED, scope-validated, source-RECONCILED entry BOUND TO `value` --
    the no-fallback, VALUE-level (not just axis-level) safety-critical check a
    MUST_SUPPORT-governed axis (laterality) needs (issue #6 F9-R6-R2, fifth AND
    sixth re-review).

    issue #6 F9-R6-R2, sixth re-review: `value` is now REQUIRED, not merely
    axis-level -- proven exploitable without it: a reconciled quote genuinely
    stating "left", marked ASSERTED, was accepted as proof of an unrelated
    `attributes["laterality"]="right"`, since the old axis-only check only asked
    "does SOME asserted entry exist for this axis," never "does THIS entry
    assert the value actually being checked." An entry whose own `.value` does
    not canonically match `value` (including an older/unbound entry whose
    `.value` is still `""`) never counts, regardless of its assertion_state.

    Unlike `_attribute_span_support`, this never falls back to whole-fact-text
    lexical matching when no per-attribute evidence exists for the axis --
    fails closed instead, since that lexical fallback is exactly the mechanism
    proven unsound against arbitrary-distance negation. Also unlike
    `_attribute_span_support`, this is gated STRICTLY on the value-bound
    ASSERTED subset's own reconciliation -- a confirmed but NEGATED quotation
    must never satisfy this, even though `_attribute_span_support` itself
    reports such a quotation as `ok=True` (confirmed, just not asserting)."""
    norm_value = _norm(value)
    if not norm_value:
        return False           # nothing to assert -- never matches an unbound ("") entry
    entries = (getattr(fact, "attribute_evidence", None) or {}).get(attr_name)
    if not entries:
        return False
    usable = [e for e in entries if e.scope == "local" or e.scope_validated]
    bound = [e for e in usable if _norm(e.value) == norm_value]
    asserted = [e for e in bound if e.assertion_state == RelationState.ASSERTED]
    if not asserted:
        return False
    ok, _proof, _text, _spans = _spans_support([e.span for e in asserted], reconciliation)
    return ok


def claim_authorized_value(fact, axis: str, reconciliation) -> str | None:
    """THE one accessor every claim-affecting consumer -- candidate elimination/
    specificity scoring, diagnosis specificity upgrades, modifier assignment,
    distinct-service determination, billing units, anatomy eligibility exclusion
    -- must use instead of reading `fact.attributes[axis]` directly (issue #6
    F9-R6-R2, sixth re-review). Round 3 through 5 protected `tiebreak.narrow`'s
    tie-break path; the review that prompted this function proved the actual,
    ordinary release path never went through it at all: `resolution._evaluate`'s
    deterministic elimination/specificity scoring reads `fact.attributes
    ["laterality"]` raw, so a fact with a source-confirmed NEGATED attribute
    value still resolved deterministically to the wrong-side candidate every
    time -- every prior round's fix was real but scoped to the wrong (rarer)
    path.

    Returns the value only when a scope-valid, source-RECONCILED
    `AttributeEvidence` entry exists whose own `.value` canonically matches it
    AND whose `assertion_state` is ASSERTED (`asserted_attribute_support`).
    When no `attribute_evidence` exists for the axis at all, OR none of it is
    value-bound to the current value, falls back to the SAME whole-fact-text
    `_span_support` + `tiebreak.asserted_status` check `resolve()`'s own
    fallback already uses -- no claim-affecting consumer this round is left
    with WEAKER verification than `resolve()` itself already provides for the
    "no per-attribute evidence yet" case, but none is left with NO verification
    at all either, which is the state every site this function replaces was in
    before this round.

    Returns `None` (fail closed) when the value cannot be authorized -- the
    caller must treat that exactly as "not documented" for its own decision,
    never as "documented, use the raw value anyway"."""
    value = str((getattr(fact, "attributes", None) or {}).get(axis) or "").strip()
    if not value:
        return None
    entries = (getattr(fact, "attribute_evidence", None) or {}).get(axis)
    usable = [e for e in entries if e.scope == "local" or e.scope_validated] if entries else []
    norm_value = _norm(value)
    bound = [e for e in usable if _norm(e.value) == norm_value]
    if bound:
        return value if asserted_attribute_support(fact, axis, value, reconciliation) else None
    from . import tiebreak as _tiebreak
    ok, _proof, text, _spans = _span_support(fact, reconciliation)
    if ok and _tiebreak.asserted_status((value,), text) == "supported":
        return value
    return None


def source_support(fact, reconciliation) -> tuple[bool, str, str, tuple[str, ...]]:
    """THE definition of "the original document backs this reading of this event".

    Public because the EVENT-CANDIDATE UNION (`claude_coder.event_union`) admits a
    recovered event on exactly this test. Two implementations of "is this reading
    source-supported?" is precisely how an event the primary extractor missed would be
    admitted on a weaker bar than the one a corrected axis has to clear.
    """
    return _span_support(fact, reconciliation)


def _question(disagreement: AxisDisagreement) -> str:
    values = [v for v in (disagreement.value_primary, disagreement.value_second) if v]
    # Honest about WHAT happened, not just what was said: a value present on only one
    # side is not "two readings disagreed" -- that phrasing was confusing when this
    # path was reachable for an asymmetric axis (`resolve` now settles most of those
    # before ever reaching here; this remains correct for whatever residual case
    # still arrives with a single recorded value and no source-confirmed win).
    if len(values) <= 1:
        read_as = repr(values[0]) if values else "nothing"
        return (f"The record does not settle {disagreement.axis!r} for "
                f"{disagreement.action!r}: one reading of the original document read "
                f"it as {read_as}, and the page itself does not confirm it. Please "
                f"document {disagreement.axis!r} explicitly for this event.")
    read_as = " and ".join(repr(v) for v in values)
    return (f"The record does not settle {disagreement.axis!r} for "
            f"{disagreement.action!r}: two independent readings of the original "
            f"document read it as {read_as}, and the page itself does not confirm "
            f"either. Please document {disagreement.axis!r} explicitly for this event.")


def resolve(disagreements: list[AxisDisagreement], primary_by_id: dict,
            second_by_node: dict, reconciliation) -> list[AxisResolution]:
    """Settle each disagreeing axis against the ORIGINAL DOCUMENT, never by vote.

    A reading wins only when its OWN value is LITERALLY PRESENT in its own
    source-confirmed quotation -- the document stating the value, never a model
    asserting it, and never merely the surrounding EVENT being source-confirmed
    (Codex F8-R1, round 2: a reading whose event quotation reconciled but whose
    specific attribute value was never checked against that quotation was
    previously accepted whenever the OTHER reading's event simply wasn't
    confirmed -- event-level confirmation is not value-level confirmation,
    regardless of how many readings had a confirmed event). Anything else is
    unresolved, which is a provider question, never a coder queue.

    Proof is now PER-ATTRIBUTE first (issue #6 F9-R5): when a reading carries
    `attribute_evidence` for this specific axis, that -- not the fact's whole,
    undifferentiated evidence text -- is what the value is checked against, so a
    value stated once in a scoped heading/parent event can be proven without one of
    its tokens merely happening to co-occur elsewhere in the fact's evidence pool.
    Falls back to `_span_support`'s whole-fact-text check when no per-attribute
    evidence was recorded for this axis -- strictly additive, never a narrowing of
    what already resolved before this field existed.
    """
    from . import tiebreak as _tiebreak

    def _entailment(fact, axis, value) -> tuple[bool, str, str, tuple[str, ...], bool]:
        """(ok, proof, text, spans, says) for one reading. VALUE-level entailment,
        gated on the reading's own EVENT actually being source-confirmed (ok) --
        applied uniformly, so no branch can accept a value its own confirmed
        quotation never states, regardless of what the OTHER reading did or did not
        confirm.

        issue #6 F9-R6-R2, fifth re-review, ROOT CAUSE FIX: when `fact` carries
        per-attribute evidence for `axis` at all, whether the value may be ACCEPTED
        (`says`) is gated through `asserted_attribute_support` -- genuine,
        extraction-time ASSERTED judgement, never re-derived by re-scanning raw
        text with a token-window heuristic afterward (`tiebreak.asserted_status`),
        which cannot correctly resolve negation scope at arbitrary distance
        (proven: the FOURTH re-review's fix used exactly that heuristic here and
        still accepted a value the source text explicitly ruled out many tokens
        away). `ok` (whether this reading's EVENT is source-confirmed, for the
        detail-message branching below) stays keyed to `_attribute_span_support`'s
        OWN, separate reconciliation check -- deliberately NOT collapsed into
        `says`, so a quotation the model correctly marked NEGATED but which the
        source genuinely confirms is reported as a real documentation state (falls
        to the "no reading's own confirmed quotation states this value verbatim"
        branch), never mis-filed as an unconfirmed SOURCE-INTEGRITY problem, which
        is a different, more specific claim. The lexical check is used ONLY as the
        pre-existing, strictly-additive fallback when NO per-attribute evidence
        exists for this axis at all -- unchanged from before, never the axis this
        vulnerability was found on.
        """
        if fact is None:
            return False, "", "", (), False
        attr = _attribute_span_support(fact, axis, value, reconciliation)
        if attr is not None:
            ok, proof, text, spans = attr
            says = bool(value) and asserted_attribute_support(fact, axis, value, reconciliation)
            return ok, proof, text, spans, says
        ok, proof, text, spans = _span_support(fact, reconciliation)
        says = ok and bool(value) and _tiebreak.asserted_status(
            (value,), text) == "supported"
        return ok, proof, text, spans, says

    out: list[AxisResolution] = []
    for item in disagreements:
        primary = primary_by_id.get(item.node_id)
        second = second_by_node.get(item.node_id)
        p_ok, p_proof, p_text, p_spans, p_says = _entailment(
            primary, item.axis, item.value_primary)
        s_ok, s_proof, s_text, s_spans, s_says = _entailment(
            second, item.axis, item.value_second)

        winner = ""
        proof = ""
        spans: tuple[str, ...] = ()
        detail = ""
        if p_says and not s_says:
            winner, proof, spans = "primary", p_proof, p_spans
            detail = "the confirmed quotation states this value verbatim"
        elif s_says and not p_says:
            winner, proof, spans = "second", s_proof, s_spans
            detail = "the confirmed quotation states this value verbatim"
        elif not p_ok and not s_ok:
            detail = ("neither reading rests on quotations the source confirms, so "
                      "this event has a source-integrity problem, not a documentation "
                      "gap")
        else:
            # p_ok or s_ok (or both) -- some reading's EVENT is source-confirmed --
            # but no reading's own recorded value is literally supported by its own
            # confirmed quotation. Covers every remaining shape uniformly: only one
            # reading's event confirmed (with or without a recorded value on that
            # axis), both events confirmed but neither value stated, and a value
            # recorded by only one reading with nothing to corroborate it. None of
            # these may auto-accept; `_question()` already phrases the one-reading
            # vs two-reading cases correctly on its own (`len(values) <= 1`), so this
            # detail only needs to be honest, not case-specific.
            detail = ("no reading's own confirmed quotation states this value "
                      "verbatim -- a reading's EVENT being source-confirmed is not "
                      "proof of this specific attribute")

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
            _carry_attribute_evidence(fact, source_fact, resolution.axis)


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


def _carry_attribute_evidence(fact, source_fact, axis: str) -> None:
    """When the SECOND reading's value wins an axis, carry ITS `attribute_evidence`
    for that axis onto the winning fact too (issue #6 F9-R6-R2, sixth re-review) --
    `_write_axis` already writes the winning raw string, and `_carry_evidence`
    already copies the whole-fact evidence spans, but neither ever touched
    `attribute_evidence`. Without this, the winning fact's `attribute_evidence[axis]`
    stayed exactly what the LOSING primary reading originally populated (describing
    a value that is no longer what `attributes[axis]` holds, or empty if the primary
    never emitted any) -- `claim_authorized_value`/`asserted_attribute_support`
    would then find no value-bound evidence for the value that just won, and
    wrongly fail closed on a resolution that should have succeeded.

    Appends (never replaces) -- the losing primary's own entries stay for audit,
    and the value-binding filter in `claim_authorized_value` already ignores any
    entry whose `.value` does not match the CURRENT `attributes[axis]`, so a stale
    primary-reading entry for the old value can never be mistaken for proof of the
    new one. Deduplicated by span_id, matching `_carry_evidence`'s own discipline.
    """
    source_entries = (getattr(source_fact, "attribute_evidence", None) or {}).get(axis)
    if not source_entries:
        return
    existing = (getattr(fact, "attribute_evidence", None) or {}).get(axis) or ()
    have = {str(getattr(e.span, "span_id", "") or "") for e in existing}
    merged = list(existing)
    for entry in source_entries:
        span_id = str(getattr(entry.span, "span_id", "") or "")
        if span_id and span_id not in have:
            merged.append(entry)
            have.add(span_id)
    fact.attribute_evidence = {
        **(getattr(fact, "attribute_evidence", None) or {}),
        axis: tuple(merged),
    }


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
            second_origin: Any = None,
            alignment: tuple | None = None,
            source: Any = None) -> tuple[ConsensusReport, dict, dict]:
    """Align two readings and list every code-changing axis they disagree on.

    Returns the report plus the two id-keyed views the resolver needs, so the caller can
    escalate to the page verifier between comparison and resolution.

    `alignment` accepts an `align()` result the caller already has. The event-candidate
    union needs the SAME correspondence this comparison used -- an event counted as
    matched here must not be proposed as a new event there -- so the alignment is
    computed once and shared rather than recomputed and trusted to agree.

    `source` (issue #6 F7-R3-C4) is passed straight through to `compare_axes`.
    """
    pairs, unmatched_primary, unmatched_second = (
        alignment if alignment is not None else align(primary_facts, second_facts))
    disagreements, compared, governed_matches = compare_axes(pairs, source)
    primary_by_id = {str(getattr(f, "fact_id", "") or ""): f for f in primary_facts}
    second_by_node = {str(getattr(left, "fact_id", "") or ""): right
                      for left, right in pairs}
    # Apply every governed match onto the surviving PRIMARY fact (issue #6 F7-R3-C4):
    # retrieval reads `ClinicalFact.governed_terms` to query under the confirmed
    # alternate wording too, so a code indexed only under the SECOND reading's
    # synonym is not silently unreachable once the two readings merge with no
    # disagreement raised.
    for match in governed_matches:
        fact = primary_by_id.get(match["node_id"])
        alt = str(match.get("value_second") or "").strip()
        if fact is None or not alt:
            continue
        axis = match["axis"]
        current = dict(getattr(fact, "governed_terms", None) or {})
        if alt not in current.get(axis, ()):
            current[axis] = current.get(axis, ()) + (alt,)
        fact.governed_terms = current
    report = ConsensusReport(
        second_reading_origin=(second_origin.as_record()
                               if hasattr(second_origin, "as_record") else {}),
        matched_events=len(pairs),
        axes_compared=compared,
        disagreements=tuple(disagreements),
        unmatched_primary=tuple(_event_record(f) for f in unmatched_primary),
        unmatched_second=tuple(_event_record(f) for f in unmatched_second),
        governed_matches=tuple(governed_matches),
    )
    return report, primary_by_id, second_by_node
