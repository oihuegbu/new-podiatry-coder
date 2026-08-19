"""ONE definition of "are these two mentions the same performed event?" (issue #6 F7-R3).

WHY THIS MODULE EXISTS

Three places in this pipeline had to answer that question and each answered it its own
way, so the three answers disagreed and the disagreement was billable:

  * `event_union` treated any quotation sitting in a document region no primary event
    rested on as a NEW EVENT. Region novelty is real evidence -- but it is evidence that
    a MENTION exists somewhere the first reading did not look, not evidence that a
    second occurrence of the service happened.
  * `eligibility` merged two mentions only when their action text tokenized IDENTICALLY,
    so the same event written twice in different words stayed two intents.
  * claim assembly then read "two intents, same resolved code, different quoted wording"
    as a separately-documented repeat and ADDED UNITS, on the reasoning that the
    medically-unlikely-edit ceiling would catch anything excessive.

That last step is the defect the others feed. A maximum is a limit on what MAY be
billed; it is never evidence that anything happened twice. One service, documented once
and described twice, could therefore be billed twice while every control reported clean.

WHAT IS AND IS NOT EVIDENCE OF A SECOND OCCURRENCE

Distinctness must be ESTABLISHED, never assumed, and the only things that establish it
are things the record actually says:

  * the events differ on a distinguishing axis where BOTH values are known and they
    differ -- anatomy, laterality, performer, approach, an explicitly distinct site,
    session, objective or encounter;
  * the record explicitly separates them (an asserted SEPARATE_FROM edge);
  * they belong to different episodes (different encounter/date);
  * the record states a count.

Different WORDING establishes nothing. Neither does a new document region, and neither
does what an edit ceiling happens to permit. Where the record does not settle it, the
answer is UNDETERMINED -- and an undetermined pair is never counted twice.

WHAT UNDETERMINED DELIBERATELY DOES *NOT* DO

It does not merge the two mentions into one event before retrieval. Two mentions with
compatible axes and unrelated actions ("something was done" and "something else was
done" on one side, in one session, by one performer) are also undetermined by this
test, and collapsing them would DELETE a legitimate second service -- the undercoding
this pipeline's recall machinery exists to prevent. So both survive to retrieval, are
resolved against the authoritative data independently, and meet again at claim
assembly: two mentions that resolve to the SAME authoritative code, in the same
episode, with no documented distinctness, are one service described twice and become
one line with the units the record itself states. Two mentions that resolve to
DIFFERENT codes were two services and stay two lines. Procedure identity is therefore
settled by the authoritative descriptor set rather than by anyone's prose -- which is
the only place in this system where it can be settled at all.

NO MEDICAL CODE AND NO CLINICAL VOCABULARY APPEARS HERE. The axes are axis NAMES, the
normalizer is ordinary morphology, and procedure identity is deferred to the
authoritative data.
"""
from __future__ import annotations

import re
from typing import Any

#: Identity of this contract, recorded wherever a verdict is written into the audit.
COREFERENCE_CONTRACT_VERSION = "event-coreference-v1"

# ---- verdicts ----------------------------------------------------------------------
#: The record establishes that these two mentions are ONE event.
SAME_EVENT = "same_event"
#: The record establishes that these two mentions are TWO events.
DISTINCT_EVENT = "distinct_event"
#: The record establishes neither. Never counted as two.
UNDETERMINED = "undetermined"

#: The axes on which a documented difference makes two mentions two EVENTS. Axis NAMES
#: only -- the values are whatever the record stated, and nothing here knows or cares
#: what any particular value means.
DISTINGUISHING_AXES: tuple[str, ...] = (
    "anatomy", "laterality", "performer_id", "performer", "approach",
    "distinct_site", "distinct_session", "distinct_objective", "distinct_encounter",
)

#: Attributes that carry a STATED count of how many times the service was performed.
CARDINALITY_ATTRIBUTES: tuple[str, ...] = ("count", "quantity")

#: Tokens that describe the DOCUMENTATION rather than the action: dispositional,
#: temporal and administrative filler that two writers use interchangeably for one
#: event. Ordinary English, deliberately not a clinical lexicon -- removing them makes
#: "X was done" and "did X" compare equal without teaching this module any domain.
_NON_DISTINCTIVE = frozenset({
    "performed", "performing", "perform", "completed", "complete", "done", "carried",
    "underwent", "undergone", "today", "todays", "session", "visit", "encounter",
    "procedure", "procedural", "service", "services", "status", "documented",
    "documentation", "noted", "note", "record", "recorded", "patient", "patients",
    "with", "without", "were", "was", "been", "being", "have", "having", "that",
    "this", "then", "than", "from", "into", "onto", "over", "under", "after",
    "before", "during", "also", "same", "both", "each", "well", "using", "used",
})

#: Ordinary English suffixes, longest first. Stripping them is morphology, not meaning:
#: it makes one writer's noun and another's verb the same token, and it can only ever
#: merge two spellings of ONE root -- it can never make two different roots equal.
_SUFFIXES = ("ations", "ation", "ements", "ement", "ments", "ment", "ings", "ing",
             "ions", "ion", "ies", "ied", "ers", "er", "ed", "es", "s")

_WORD = re.compile(r"[^a-z0-9]+")

#: Below this length a token carries no distinguishing signal, and a stem shorter than
#: this is over-stemmed: it would start making unrelated roots compare equal, which
#: merges two real services into one line. Short is safe in only one direction here, so
#: the floor is deliberately the same one the previous exact-token rule used.
_MIN_STEM = 4


def _stem(token: str) -> str:
    """One token reduced to its root spelling. Never shortened below `_MIN_STEM`."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            return token[: -len(suffix)]
    return token


def action_form(text: Any) -> frozenset[str]:
    """The distinctive root tokens of one clinical action phrase.

    Word ORDER, inflection, and documentation filler are presentation rather than
    action, so they are removed: two writers describing one action with the same words
    in a different arrangement produce the same form. Nothing here can turn two
    different roots into one -- establishing that two unrelated spellings mean the same
    procedure is exactly what the authoritative descriptor set is for.
    """
    out = set()
    for raw in _WORD.split(str(text or "").lower()):
        if not raw or len(raw) < _MIN_STEM or raw in _NON_DISTINCTIVE:
            continue
        stem = _stem(raw)
        if len(stem) >= _MIN_STEM and stem not in _NON_DISTINCTIVE:
            out.add(stem)
    return frozenset(out)


def action_identity(left: Any, right: Any) -> str:
    """SAME_EVENT when two action phrases are the same action, else UNDETERMINED.

    DISTINCT_EVENT is deliberately unreachable from wording. Two phrases with no root in
    common may be two services or one service described two ways, and this test cannot
    tell -- so it never claims to. Distinctness comes from the record's own axes, or
    from the authoritative data once both mentions have resolved.
    """
    a, b = action_form(left), action_form(right)
    if a and b and a == b:
        return SAME_EVENT
    return UNDETERMINED


#: Axes carrying a STABLE IDENTIFIER (a system key, not free clinical text): compared
#: literally, never normalized. An identifier is either the same identifier or it is
#: not; stemming one could accidentally equate two different ids or split one in half.
_IDENTIFIER_AXES = frozenset({"performer_id"})

#: Axes whose value space is a CLOSED, small clinical enumeration the record states
#: directly -- not open vocabulary. Compared literally (case/whitespace folding only):
#: there is no synonym/abbreviation/eponym risk in 'left' vs 'right' the way there is
#: in open anatomy, approach, or site text (a lay term vs its clinical synonym), so a genuine
#: inequality here IS a confirmed, unambiguous difference. (Codex F7-R3, exact-SHA
#: re-review, third pass.)
_ENUMERATED_AXES = frozenset({"laterality"})

#: The CLOSED value set `laterality` actually takes, in canonical form. The extraction
#: boundary does not enforce this enumeration -- a fact's `attributes["laterality"]` is
#: whatever string the model wrote, verbatim -- so a value OUTSIDE this set (e.g. a full
#: phrase like 'left side' rather than the bare enum value) is not a value the literal-
#: comparison guarantee covers. Codex F7-R3-C1, exact-SHA re-review: trusting an
#: arbitrary, non-canonical string as if it were already the closed enum is exactly how
#: a noncanonical value slipped a confirmed difference (and a false extra unit) past the
#: enumerated-axis literal compare. A non-canonical value degrades to the SAME
#: exact-match-or-ambiguous treatment open vocabulary gets, rather than being trusted at
#: face value.
_CANONICAL_LATERALITY = frozenset({"left", "right", "bilateral", "unspecified"})

#: The only CANONICAL laterality pair that is genuinely, unambiguously OPPOSED.
#: Codex F7-R3-C1, exact-SHA re-review, third pass: a flat "unequal canonical values
#: are different" rule is wrong occurrence logic for this specific enum, because its
#: own values overlap in meaning rather than partitioning cleanly. 'unspecified'
#: asserts nothing (unknown, not a third side); 'bilateral' covers BOTH sides, so it
#: does not contradict either 'left' or 'right' alone -- it is compatible evidence
#: for a bilateral procedure being (correctly or incompletely) described as
#: unilateral in one mention, not proof of two distinct sides. Only 'left' and
#: 'right' actually exclude each other.
_LATERALITY_DISJOINT_PAIR = frozenset({"left", "right"})

#: Open-vocabulary axes wired to a governed concept source (issue #6 F7-R3-C) when a
#: `source` is supplied. Only 'anatomy' has an authoritative concept graph in this
#: codebase today (SNOMED CT Body Structure, tools/build_snomed_concept_terms.py);
#: the other open axes (approach, distinct_site, distinct_objective,
#: distinct_encounter) keep the identity-or-ambiguous fallback below until a
#: comparable governed source exists for them -- never approximated by a lexical
#: heuristic in the meantime.
_CONCEPT_GOVERNED_AXES = frozenset({"anatomy"})


def _axis_relation(axis: str, va: str, vb: str, source: Any = None) -> str:
    """SAME_EVENT, DISTINCT_EVENT, or UNDETERMINED for one axis's two stated values.

    Identifier axes (performer_id): an id is either the same id or a different one --
    no overlap semantics, so any inequality is a confirmed difference.

    Enumerated axes (laterality) where BOTH values are already canonical members of
    the enum: real RELATION semantics, not flat inequality (`_LATERALITY_DISJOINT_PAIR`)
    -- equal values are the same; 'left' vs 'right' is the one genuinely disjoint pair;
    everything else ('unspecified' paired with anything, 'bilateral' paired with a
    single side) is UNDETERMINED, because the record does not actually state two
    excluding facts.

    Everything else -- OPEN vocabulary, or an enumerated axis where either value is
    NOT canonical (the extraction boundary does not enforce the enumeration, so a
    stored value may be an arbitrary string) -- gets the conservative open-vocabulary
    treatment: an exact match (case/whitespace folded) is the same value. For an axis
    in `_CONCEPT_GOVERNED_AXES`, an inexact match is then put to the governed concept
    graph (`source.concept_relation`) when a `source` was supplied: a confirmed SAME
    concept is SAME_EVENT, a confirmed DISJOINT concept is DISTINCT_EVENT, and an
    ancestor/descendant or unresolved relation -- like every other open axis, and
    like this axis when no `source` is available at all -- is UNDETERMINED. Lexical
    shape alone never establishes a confirmed DISTINCT_EVENT for open text (Codex
    F7-R3-C, exact-SHA re-review, third pass) or for a non-canonical enumerated value
    it cannot be told apart from; the concept graph is consulted precisely because it
    is authoritative data, not lexical shape.
    """
    if axis in _IDENTIFIER_AXES:
        return SAME_EVENT if va.lower() == vb.lower() else DISTINCT_EVENT
    if (axis in _ENUMERATED_AXES and va.lower() in _CANONICAL_LATERALITY
            and vb.lower() in _CANONICAL_LATERALITY):
        a, b = va.lower(), vb.lower()
        if a == b:
            return SAME_EVENT
        if {a, b} == _LATERALITY_DISJOINT_PAIR:
            return DISTINCT_EVENT
        return UNDETERMINED
    if va.lower() == vb.lower():
        return SAME_EVENT
    if axis in _CONCEPT_GOVERNED_AXES and source is not None:
        concept_relation = getattr(source, "concept_relation", None)
        if callable(concept_relation):
            from .terminology import CONCEPT_DISJOINT, CONCEPT_SAME
            try:
                verdict = concept_relation(va, vb)
            except Exception:
                verdict = None
            if verdict == CONCEPT_SAME:
                return SAME_EVENT
            if verdict == CONCEPT_DISJOINT:
                return DISTINCT_EVENT
    return UNDETERMINED


def known_known_differences(left: dict | None, right: dict | None,
                            axes: tuple[str, ...] = DISTINGUISHING_AXES,
                            source: Any = None) -> tuple[str, ...]:
    """The axes on which BOTH mentions state a value and the values are GENUINELY,
    UNAMBIGUOUSLY opposed.

    Known-plus-missing is NOT a difference: one reading recording an axis the other left
    blank is an incomplete reading, not a second event.

    Codex F7-R3, exact-SHA re-review, third pass: raw lexical SHAPE -- token overlap,
    subset containment, or disjointness -- is not a stable terminology identity in
    EITHER direction for OPEN clinical vocabulary (anatomy, approach, site, session,
    objective, encounter). The prior fix treated disjoint normalized tokens as a
    confirmed difference; the reviewer's counterexample showed that is unsafe too --
    a lay term and its clinical synonym for the same structure can share no root at
    all, a pair no stemmer can be expected to unify. Without a versioned terminology-
    normalization service, the only claim wording alone can safely make about OPEN
    vocabulary is IDENTITY: an exact match (after case/whitespace folding) is the
    same documented value. Where a governed concept source IS available (`source`,
    issue #6 F7-R3-C -- currently `anatomy` only, see `_CONCEPT_GOVERNED_AXES`), an
    inexact match is put to that authoritative graph rather than left an unresolved
    guess; anything the graph does not confirm DISJOINT is `known_known_ambiguous`,
    never promoted to a confirmed difference by disjointness, containment, or any
    other lexical heuristic.

    A CLOSED, small clinical enumeration the record states directly (laterality) has
    an explicit RELATION (`_axis_relation`), not a flat equality test: 'left' vs
    'right' is the one genuinely disjoint canonical pair, but 'unspecified' or
    'bilateral' paired with a single side is deliberately NOT a confirmed difference
    either (Codex F7-R3-C1, exact-SHA re-review, third pass -- a flat inequality rule
    made 'unspecified' vs 'left' and 'bilateral' vs 'left' both overbill). The
    extraction boundary does not itself enforce the enumeration, so a NON-CANONICAL
    value (a full phrase like 'left side' rather than the bare enum value) is not
    something this function may trust at face value either -- it degrades to the same
    identity-or-ambiguous treatment open vocabulary gets.

    Genuine distinctness from OPEN vocabulary (and from a non-canonical enumerated
    value) is established elsewhere in this system -- the governed concept graph
    above, an explicit SEPARATE_FROM relation (`explicitly_separated`, checked before
    this function even runs), or a different service episode -- never by comparing
    two raw strings.
    """
    a, b = dict(left or {}), dict(right or {})
    out: list[str] = []
    for axis in axes:
        va = str(a.get(axis, "") or "").strip()
        vb = str(b.get(axis, "") or "").strip()
        if va and vb and _axis_relation(axis, va, vb, source) == DISTINCT_EVENT:
            out.append(axis)
    return tuple(out)


def known_known_ambiguous(left: dict | None, right: dict | None,
                          axes: tuple[str, ...] = DISTINGUISHING_AXES,
                          source: Any = None) -> tuple[str, ...]:
    """The axes on which both mentions state a value whose RELATION
    (`_axis_relation`) is UNDETERMINED -- neither confirmed same nor confirmed
    different.

    For open vocabulary that is an inexact lexical match and has no governed concept
    source (or the source cannot resolve either value), this covers the same ground
    it always has (Codex F7-R3, exact-SHA re-review, third pass): raw shape cannot
    tell a synonym/abbreviation/eponym apart from a real distinction. It also covers
    an ancestor/descendant concept relation on a governed axis -- related but not
    proof of either sameness or difference. For the laterality enumeration, it also
    covers 'unspecified' or 'bilateral' paired with a single side (Codex F7-R3-C1) --
    the record states something, but not something that excludes the other value. A
    caller deciding whether two mentions are the SAME event must treat any ambiguous
    axis as blocking that verdict, never as a match.
    """
    a, b = dict(left or {}), dict(right or {})
    out: list[str] = []
    for axis in axes:
        va = str(a.get(axis, "") or "").strip()
        vb = str(b.get(axis, "") or "").strip()
        if va and vb and _axis_relation(axis, va, vb, source) == UNDETERMINED:
            out.append(axis)
    return tuple(out)


def documented_cardinality(attributes: dict | None) -> int | None:
    """The count the RECORD states for this event, or None when it states none.

    None means "the record did not say", which is one occurrence for billing purposes
    and can never become more than one by inference.
    """
    values = dict(attributes or {})
    for axis in CARDINALITY_ATTRIBUTES:
        raw = values.get(axis)
        if raw is None or isinstance(raw, bool):
            continue
        try:
            count = int(raw)
        except (TypeError, ValueError):
            continue
        if count >= 1:
            return count
    return None


def event_verdict(*, left_kind: Any, right_kind: Any,
                  left_action: Any, right_action: Any,
                  left_attributes: dict | None, right_attributes: dict | None,
                  left_episode: Any = None, right_episode: Any = None,
                  explicitly_separated: bool = False,
                  source: Any = None) -> tuple[str, str]:
    """Are these two mentions one event or two? Returns `(verdict, reason)`.

    Every DISTINCT_EVENT answer names the documented thing that established it, so an
    extra billed occurrence can always be traced to a statement in the record rather
    than to a heuristic.

    `source` is optional and, when supplied, is consulted for axes in
    `_CONCEPT_GOVERNED_AXES` (issue #6 F7-R3-C) to tell an inexact lexical match on
    those axes apart as a governed-SAME, governed-DISJOINT, or still-ambiguous
    relation instead of defaulting every inexact match straight to `UNDETERMINED`.
    """
    lk = str(getattr(left_kind, "value", left_kind) or "")
    rk = str(getattr(right_kind, "value", right_kind) or "")
    if lk and rk and lk != rk:
        return DISTINCT_EVENT, f"different documented event kind ({lk} vs {rk})"
    if explicitly_separated:
        return DISTINCT_EVENT, "the record explicitly separates these two events"
    axes = known_known_differences(left_attributes, right_attributes, source=source)
    if axes:
        return DISTINCT_EVENT, ("the record states different values for "
                                + ", ".join(axes))
    # Codex F7-R3, exact-SHA re-review: a RELATED-but-not-identical free-text value
    # ('structure' vs 'fifth structure') must never let this reach SAME_EVENT below --
    # the extra wording may be exactly the qualifier that distinguishes two real
    # events, and wording alone cannot settle which. Checked BEFORE the action/episode
    # tests below, because neither of those may override an axis this ambiguous.
    ambiguous = known_known_ambiguous(left_attributes, right_attributes, source=source)
    if ambiguous:
        return UNDETERMINED, (
            "the record states related but not identical values for "
            + ", ".join(ambiguous) + " -- wording alone does not establish whether "
            "this is the same documented value or a distinguishing qualifier")
    le, re_ = str(left_episode or ""), str(right_episode or "")
    if le and re_ and le != re_:
        return DISTINCT_EVENT, "documented in different service episodes"
    if action_identity(left_action, right_action) == SAME_EVENT:
        return SAME_EVENT, ("the same documented action on the same documented axes "
                            "in the same episode")
    return UNDETERMINED, (
        "the two mentions describe compatible events in different words and the record "
        "states nothing that distinguishes them; whether they are one event or two is "
        "not established by the document")


def is_additional_occurrence(verdict: str) -> bool:
    """May a second mention with this verdict add BILLABLE OCCURRENCES?

    Only an established DISTINCT_EVENT may. UNDETERMINED may not -- that is the whole
    correction: an unproven second occurrence is not a billable one, and the absence of
    a stated distinction is not a statement that two services occurred.
    """
    return verdict == DISTINCT_EVENT
