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


def known_known_differences(left: dict | None, right: dict | None,
                            axes: tuple[str, ...] = DISTINGUISHING_AXES
                            ) -> tuple[str, ...]:
    """The axes on which BOTH mentions state a value and the values GENUINELY differ.

    Known-plus-missing is NOT a difference: one reading recording an axis the other left
    blank is an incomplete reading, not a second event.

    Nor is a lexical restatement of the SAME value (Codex F7-R3, round-9 re-review,
    defect D): raw string inequality treated 'left side' as different from 'Left',
    manufacturing a documented difference -- and therefore a billable second occurrence
    -- out of two writers' wording rather than out of anything the record actually
    distinguishes. An identifier axis is still compared literally (normalizing an id is
    never correct). A free-text axis is compared on the SAME normalized root tokens
    `action_form` reduces the clinical action to, and two values are the SAME documented
    value when one's tokens are a SUBSET of the other's -- one writer simply adding a
    qualifier word ('side', 'approach') the other omitted is not a second, contradicting
    value. Two values that each carry a token the OTHER does not (neither is a subset of
    the other) are a genuine documented difference: plain overlap is not enough, or
    'first approach' and 'other approach' would compare equal on the shared word
    'approach' alone. The authoritative descriptor set, never this axis check, is what
    ultimately settles whether restated wording means one service or two.
    """
    a, b = dict(left or {}), dict(right or {})
    out: list[str] = []
    for axis in axes:
        va = str(a.get(axis, "") or "").strip()
        vb = str(b.get(axis, "") or "").strip()
        if not (va and vb):
            continue
        if axis in _IDENTIFIER_AXES:
            if va.lower() != vb.lower():
                out.append(axis)
            continue
        fa, fb = action_form(va), action_form(vb)
        if fa and fb:
            if not (fa <= fb or fb <= fa):
                out.append(axis)
        elif va.lower() != vb.lower():
            # Either value normalized away to nothing (e.g. below the stem floor) --
            # fall back to a literal compare rather than reading "no signal either way"
            # as agreement, and rather than letting an empty set's vacuous subset
            # relationship manufacture an agreement neither value actually states.
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
                  explicitly_separated: bool = False) -> tuple[str, str]:
    """Are these two mentions one event or two? Returns `(verdict, reason)`.

    Every DISTINCT_EVENT answer names the documented thing that established it, so an
    extra billed occurrence can always be traced to a statement in the record rather
    than to a heuristic.
    """
    lk = str(getattr(left_kind, "value", left_kind) or "")
    rk = str(getattr(right_kind, "value", right_kind) or "")
    if lk and rk and lk != rk:
        return DISTINCT_EVENT, f"different documented event kind ({lk} vs {rk})"
    if explicitly_separated:
        return DISTINCT_EVENT, "the record explicitly separates these two events"
    axes = known_known_differences(left_attributes, right_attributes)
    if axes:
        return DISTINCT_EVENT, ("the record states different values for "
                                + ", ".join(axes))
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
