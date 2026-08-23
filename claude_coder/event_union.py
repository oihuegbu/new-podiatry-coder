"""The high-recall EVENT-CANDIDATE UNION across independent readings (directive §3/§4).

WHY THIS MODULE EXISTS

`graph_consensus` compares two readings on their code-changing AXES. That makes the
second reading a disagreement detector for events BOTH readings found -- and nothing at
all for an event only ONE of them found. An event the primary extractor missed entirely
was recorded in `unmatched_second` metadata and then dropped: never proven against the
page, never given ownership, never deduplicated, never run through eligibility, never
retrievable. A note documenting two performed services whose primary reading caught one
of them therefore produced a complete-looking, internally consistent, silently
UNDER-CODED claim (issue #6, finding F7-R3).

The redundancy the directive asks for is RECALL redundancy as well as axis agreement, so
the two readings must meet as a UNION OF EVENT CANDIDATES, not as an axis comparison of
already-aligned events. This module is that union's identity-and-admission stage.

WHAT IT DECIDES, AND WHAT IT DELIBERATELY DOES NOT

It decides exactly two things about an event only the second reading reported:

  1. IS IT A NEW EVENT AT ALL?  Two tests, and a candidate must clear BOTH.

     INDEPENDENT ANCHORING first: two readings of one document that describe the same
     event quote the same passage of it, so a candidate whose every quotation lands
     inside text the primary graph already reasons about is the same documented event in
     different words -- the exact "reworded duplicate" that must never become a second
     claim line -- and is recorded as such, never admitted.

     A quotation in a region no primary event rests on then clears the FIRST test only.
     It is RECALL EVIDENCE that a mention exists somewhere the primary extraction did
     not look; it is NOT proof that a second occurrence happened, and treating it as
     proof is what let one event, written twice in two places in different words, become
     two events (issue #6 F7-R3, reopened). So the candidate is put through the SAME
     coreference test every other pair of mentions in this pipeline goes through
     (`claude_coder.coreference`) -- documented action form plus anatomy, laterality,
     performer and the rest of the record's own distinguishing axes. A candidate that
     corefers with a primary event is that event, wherever it was quoted from.

     A SECOND, independent identity signal runs once the original document has actually
     been read (`admit`, below, is the only stage that HAS a reconciliation -- `propose`
     runs before any page read, by design, so it can target the escalation at only the
     candidates that could change the claim): a candidate whose quotation reconciles to
     the SAME page/region of the real document as an already-admitted primary event's
     quotation is that event, regardless of what the text coreference test decided.
     Character offsets are only comparable inside one reading (see `_regions`, below);
     a RECONCILED page/region is anchored to the physical document itself, so it is the
     one coordinate system both readings genuinely share. This is what closes the case a
     paraphrase-sensitive text test cannot: two readings quoting the identical passage in
     different words (issue #6 F9-R1 -- confirmed on a real note: two differently-worded
     anatomical phrasings of one documented structure's location read as one documented
     passage, not a disagreement over two occurrences).

     Where coreference is UNDETERMINED the candidate is admitted, deliberately, exactly
     as before this fix: it may be the service the primary reading missed, and refusing
     it would reinstate the silent undercoding this module exists to prevent. Gating
     admission on a POSITIVELY DISTINCT text verdict was considered and rejected --
     `coreference.action_identity`'s own docstring records that DISTINCT_EVENT is
     deliberately unreachable from wording alone (it takes an explicit record
     separation or a contradicting axis value), so almost every genuinely new,
     sparsely-worded service would fail that bar and be wrongly held. What it may never
     do is multiply the claim on TEXT alone -- if it turns out to be a re-description,
     both mentions resolve to one authoritative code and claim assembly makes them one
     line with one unit (unchanged downstream safety net); the reconciled-location test
     above is what catches the case that safety net cannot, a duplicate that resolves to
     a DIFFERENT candidate code because the two readings described it at different
     abstraction levels.

  2. DOES THE ORIGINAL DOCUMENT BACK IT?  Admission is gated on the SAME source-evidence
     reconciliation every other released fact must pass, through the same one definition
     of "the source backs this reading" (`graph_consensus.source_support`). A reading the
     page contradicts is excluded with its reason; a reading nothing could check is HELD,
     because "we could not check" is never "confirmed" -- and a held candidate is never
     silently dropped either, since the whole defect being fixed here is a silent drop.

It decides NOTHING ELSE. It does not judge billability, ownership, occurrence, duplicate
cardinality, eligibility, or codes. An ADMITTED candidate becomes an ordinary
`ClinicalFact` in the primary fact list BEFORE reconciliation, eligibility, graph
construction and retrieval run, so it goes through the identical ownership, occurrence,
dedup/cannot-link, eligibility, retrieval, deterministic-verification and certificate
path as every primary event -- there is no shortcut path, and no second code path that
could drift from the first.

NO MEDICAL CODE AND NO DOMAIN VOCABULARY APPEARS HERE. The identity test is document
geometry; the admission test is the source-evidence contract.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

#: Identity of this union contract. A consumer that finds another value is reading a
#: different contract and must say so rather than guess which fields exist.
UNION_SCHEMA_VERSION = "event-candidate-union-v1"

# ---- verdicts ---------------------------------------------------------------------
#: The document backs it and it is a genuinely new event: it enters the canonical graph
#: and is decided by the ordinary pipeline, exactly like a primary event.
ADMITTED = "admitted"
#: Every quotation behind it sits in text a primary event already rests on: the same
#: documented event, differently worded. Recorded; never a second claim line.
DUPLICATE_OF_PRIMARY = "duplicate_of_primary"
#: The second reading quoted text that is not verbatim in the document at all, so there
#: is no region to prove and nothing to reconcile.
REJECTED_UNANCHORED = "rejected_unanchored"
#: An independent reading of the original page contradicts its quotations.
REJECTED_SOURCE_CONTRADICTED = "rejected_source_contradicted"
#: Nothing could confirm or refute it -- an unread page, or relational context that
#: could not be carried into the graph faithfully. Never admitted, never dropped.
HELD_UNVERIFIED = "held_unverified"

#: The verdicts that place the event INTO the canonical graph.
ADMITTING_VERDICTS = frozenset({ADMITTED})
#: The verdicts that must stop the encounter from presenting as a complete claim. A
#: candidate here is one an independent reading of the document reported and this run
#: could neither confirm nor refute; releasing as if the graph were complete would
#: reinstate exactly the silent omission this module exists to prevent.
HOLDING_VERDICTS = frozenset({HELD_UNVERIFIED})


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _regions(fact) -> tuple[tuple[str, int, int], ...]:
    """The document regions one reading of one event rests on.

    Each region carries the READING its offsets belong to. Character offsets are only
    comparable inside one reading of the document: two readings of one page produce two
    unrelated coordinate systems, so treating an offset from one as an offset in the
    other would call two unrelated passages the same place -- confidently, and wrongly
    (issue #6 F7-R3). Regions in different readings simply never overlap, which sends
    the candidate to the coreference test rather than to a false identity.

    Only ANCHORED quotations count: an unanchored quote has no offsets, so it locates
    nothing and can neither establish identity nor be reconciled against a page.
    """
    out: list[tuple[str, int, int]] = []
    for span in (getattr(fact, "evidence", None) or []):
        if not getattr(span, "anchored", False):
            continue
        start, end = getattr(span, "start", None), getattr(span, "end", None)
        if isinstance(start, int) and isinstance(end, int) and end > start:
            out.append((_clean(getattr(span, "reading_channel_id", "")), start, end))
    return tuple(dict.fromkeys(out))


def _overlap(left: tuple[str, int, int], right: tuple[str, int, int]) -> bool:
    return left[0] == right[0] and left[1] < right[2] and right[1] < left[2]


def _span_ids(fact) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        _clean(getattr(s, "span_id", "")) for s in (getattr(fact, "evidence", None) or [])
        if getattr(s, "anchored", False) and _clean(getattr(s, "span_id", ""))))


@dataclass
class EventCandidate:
    """One event only the second reading reported, and what became of it."""

    fact: Any                          # the SECOND reading's fact, never mutated here
    second_event_id: str = ""
    verdict: str = ""                  # empty until decided
    reason: str = ""
    #: The primary event this turned out to be another wording of, when duplicate.
    merged_into: str = ""
    #: AUDIT ONLY -- the text-coreference signal `propose` observed against primary
    #: events, when it did not itself settle identity (a SAME_EVENT match already
    #: becomes DUPLICATE_OF_PRIMARY above). Never gates admission: `DISTINCT_EVENT` is
    #: deliberately unreachable from wording alone
    #: (`coreference.action_identity`'s own docstring), so requiring it before
    #: admission would wrongly hold almost every genuinely new, sparsely-worded
    #: service. Recorded so a reviewer can see what the text test found even when a
    #: different signal (reconciled location, or source confirmation alone) decided
    #: the candidate. Empty means untested -- no primary events existed to compare
    #: against.
    coreference: str = ""
    #: The id this event carries IN THE CANONICAL GRAPH, when admitted.
    node_id: str = ""
    span_ids: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return _clean(getattr(getattr(self.fact, "kind", None), "value", ""))

    @property
    def decided(self) -> bool:
        return bool(self.verdict)

    def as_record(self) -> dict[str, Any]:
        return {
            "second_event_id": self.second_event_id,
            "kind": self.kind,
            "action": _clean(getattr(self.fact, "description", "")),
            "billable": bool(getattr(self.fact, "billable", False)),
            "anchored": bool(self.span_ids),
            "verdict": self.verdict,
            "reason": self.reason,
            "merged_into": self.merged_into,
            "coreference": self.coreference,
            "node_id": self.node_id,
            "evidence_span_ids": list(self.span_ids),
        }


@dataclass
class Recovery:
    """What the union added to the encounter, and what it refused to add."""

    schema_version: str = UNION_SCHEMA_VERSION
    control_mode: str = "ENFORCED_FAIL_CLOSED"
    candidates: tuple[EventCandidate, ...] = ()
    #: New `ClinicalFact`s to append to the PRIMARY fact list, before reconciliation,
    #: eligibility, graph construction and retrieval run.
    facts: tuple[Any, ...] = ()
    #: Second-reading edges naming an admitted event, with both endpoints remapped onto
    #: canonical graph ids. Still unbound and unvalidated: the caller runs them through
    #: the SAME `bind_relation_evidence` / `validate_relations` path as primary edges.
    relations: tuple[Any, ...] = ()

    @property
    def admitted(self) -> tuple[EventCandidate, ...]:
        return tuple(c for c in self.candidates if c.verdict in ADMITTING_VERDICTS)

    @property
    def holds(self) -> tuple[EventCandidate, ...]:
        return tuple(c for c in self.candidates if c.verdict in HOLDING_VERDICTS)

    def withdraw(self, reason: str) -> None:
        """Take back every admission, because the recovered set could not be carried
        into the graph as a whole.

        The caller validates the recovered facts and edges on a TRIAL copy of the graph
        inputs before appending them, so a second reading that produced an edge the
        relation kernel rejects takes the RECOVERED EVENTS out -- recorded and held --
        instead of failing the whole encounter closed. The primary encounter is
        unaffected either way; what is refused is the addition, not the note.
        """
        for candidate in self.candidates:
            if candidate.verdict not in ADMITTING_VERDICTS:
                continue
            candidate.verdict = HELD_UNVERIFIED
            candidate.node_id = ""
            candidate.reason = reason
        self.facts = ()
        self.relations = ()

    def as_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(c.as_record() for c in self.candidates)


# ------------------------------------------------------------------- identity stage
def propose(primary_facts, second_only_facts, source=None) -> list[EventCandidate]:
    """Which second-reading-only events are NEW events, decided by document geometry.

    Runs before any page read, so the reconciliation that follows is aimed only at
    candidates that could actually change the claim.

    `source` (issue #6 F7-R3-C4) is passed to the coreference test so a second-reading
    event worded as a governed-SAME concept of a primary event (e.g. a synonym anatomy
    phrase) is recognized as the SAME documented event rather than proposed as a new
    one purely because the wording did not match.
    """
    owners: list[tuple[tuple[int, int], str]] = []
    for fact in (primary_facts or []):
        fact_id = _clean(getattr(fact, "fact_id", ""))
        for region in _regions(fact):
            owners.append((region, fact_id))

    primary = list(primary_facts or [])
    out: list[EventCandidate] = []
    for fact in (second_only_facts or []):
        candidate = EventCandidate(
            fact=fact,
            second_event_id=_clean(getattr(fact, "fact_id", "")),
            span_ids=_span_ids(fact))
        regions = _regions(fact)
        if not regions:
            candidate.verdict = REJECTED_UNANCHORED
            candidate.reason = (
                "the second reading quoted no text that is verbatim in the document, "
                "so this event rests on nothing that can be located on a page or "
                "proven against it")
            out.append(candidate)
            continue
        covering = [fact_id for region in regions
                    for other, fact_id in owners if _overlap(region, other)]
        fresh = [region for region in regions
                 if not any(_overlap(region, other) for other, _ in owners)]
        if not fresh:
            candidate.verdict = DUPLICATE_OF_PRIMARY
            candidate.merged_into = covering[0] if covering else ""
            candidate.reason = (
                "every quotation behind this reading sits in document text a primary "
                "event already rests on, so it is the same documented event in "
                "different words, not an additional one")
            out.append(candidate)
            continue
        # A NEW REGION IS RECALL EVIDENCE OF A MENTION, NOT OF AN OCCURRENCE. It buys
        # the candidate the same coreference test everything else takes, never a bypass
        # of it: an event the record already carries, quoted somewhere else in the same
        # document, is still that one event.
        merged_into, signal = _identity_against_primary(fact, primary, source)
        if merged_into:
            candidate.verdict = DUPLICATE_OF_PRIMARY
            candidate.merged_into = merged_into
            candidate.reason = (
                f"quoted from a region no primary event rests on, but it corefers with "
                f"primary event {merged_into}: the same documented action on the same "
                f"documented axes, in the same episode. A second place in the record "
                f"where one event is written about is evidence that the event is "
                f"documented, never that it happened twice")
        else:
            # Not settled by text alone. Recorded for `admit` to weigh alongside the
            # reconciled-location test, which `propose` cannot run (no page has been
            # read yet at this stage).
            candidate.coreference = signal
        out.append(candidate)
    return out


def _identity_against_primary(fact, primary_facts, source=None) -> tuple[str, str]:
    """`(merged_into, signal)` for this candidate against every primary event's
    coreference verdict. `merged_into` is set only on a proven SAME_EVENT match --
    unchanged behavior, still delegated wholly to `claude_coder.coreference`, so a
    mention cannot be one event here and two events one stage later. `signal` is
    `_coref.UNDETERMINED` when at least one primary event's verdict was ambiguous and
    none was SAME_EVENT -- the record neither proves this is a re-description nor
    proves it is a distinct occurrence -- otherwise `_coref.DISTINCT_EVENT` (every
    primary event compared was proven distinct, or there was nothing to compare
    against at all: the case this module exists for, a service the primary reading
    missed entirely).

    Episodes are not compared: this runs inside ONE encounter on ONE date, so every
    event here is in the same episode by construction -- and passing an episode this
    module does not have would weaken the test, not strengthen it.

    `source` (issue #6 F7-R3-C4) is passed straight through to `event_verdict`.
    """
    from . import coreference as _coref

    undetermined = False
    for other in (primary_facts or []):
        verdict, _reason = _coref.event_verdict(
            left_kind=getattr(fact, "kind", None),
            right_kind=getattr(other, "kind", None),
            left_action=getattr(fact, "description", ""),
            right_action=getattr(other, "description", ""),
            left_attributes=getattr(fact, "attributes", None),
            right_attributes=getattr(other, "attributes", None),
            source=source)
        if verdict == _coref.SAME_EVENT:
            return _clean(getattr(other, "fact_id", "")), _coref.SAME_EVENT
        if verdict == _coref.UNDETERMINED:
            undetermined = True
    return "", (_coref.UNDETERMINED if undetermined else _coref.DISTINCT_EVENT)


def pending(candidates) -> list[EventCandidate]:
    """Candidates that are genuinely new events and now need the document to back them."""
    return [c for c in (candidates or []) if not c.decided]


def pending_span_ids(candidates) -> set[str]:
    """Exactly the quotations an original-page read has to cover for the union.

    Keeps the escalation TARGETED: only the pages a candidate new event is quoted from.
    """
    wanted: set[str] = set()
    for candidate in pending(candidates):
        wanted.update(candidate.span_ids)
    return wanted


# ------------------------------------------------------------------ admission stage
def _unique_id(prefix: str, base: str, taken: set[str]) -> str:
    node_id = f"{prefix}{base}" if prefix else base
    suffix = 2
    while node_id in taken:
        node_id = f"{prefix}{base}-{suffix}"
        suffix += 1
    taken.add(node_id)
    return node_id


def _copy_fact(fact, node_id: str):
    """An independent copy carrying its canonical graph id.

    Copied rather than renamed in place, and with its mutable members copied too, so the
    second reading's own objects stay exactly as the axis comparison recorded them and
    nothing downstream can mutate one view of an event through the other.
    """
    return replace(
        fact,
        fact_id=node_id,
        attributes=dict(getattr(fact, "attributes", None) or {}),
        evidence=list(getattr(fact, "evidence", None) or []),
        axis_confidence=dict(getattr(fact, "axis_confidence", None) or {}),
        axis_conflicts=list(getattr(fact, "axis_conflicts", None) or []))


def _remap(relation, mapping: dict[str, str]):
    """One second-reading edge, expressed in canonical graph ids, or None.

    None means the edge cannot be carried FAITHFULLY -- an endpoint that is not in the
    graph, a self-reference, or no evidence reference left after remapping. It is never
    silently dropped by the caller: a candidate an uncarryable edge names is held
    instead of admitted, because the edge the graph would lose is exactly the kind that
    DEMOTES an event (a component asserted PART_OF another service), and losing it is
    how an integral component becomes an extra billable line.
    """
    subject = mapping.get(_clean(getattr(relation, "subject_event_id", "")))
    obj = mapping.get(_clean(getattr(relation, "object_event_id", "")))
    if not subject or not obj or subject == obj:
        return None
    refs: list[str] = []
    for ref in (getattr(relation, "evidence_span_ids", None) or []):
        text = _clean(ref)
        if not text:
            continue
        if text.startswith("event:"):
            target = mapping.get(text.split(":", 1)[1])
            if target:
                refs.append(f"event:{target}")
            continue
        refs.append(text)
    if not refs:
        return None
    return replace(relation, subject_event_id=subject, object_event_id=obj,
                   evidence_span_ids=list(dict.fromkeys(refs)),
                   assertion_origins=list(getattr(relation, "assertion_origins", None) or []))


def _region_overlap(a, b) -> bool:
    """Two reconciled PDF bounding boxes over the same physical page, genuinely
    overlapping -- not merely both nonempty. Both come from `SpanReconciliation.region`
    (`app.contracts.source_evidence.PageRegion`), so `x0`/`top`/`x1`/`bottom` are
    already in the SAME coordinate system regardless of which reading's channel
    proved which span."""
    return (a.page_number == b.page_number
           and a.x0 < b.x1 and b.x0 < a.x1 and a.top < b.bottom and b.top < a.bottom)


def _reconciled_location(fact, settled: dict) -> tuple[set[int], list]:
    """This fact's reconciled pages and regions, from `SourceReconciliation.by_span_id()`
    -- the ONE coordinate system anchored to the physical document itself, the same for
    every reading (unlike `_regions`'s reading-scoped character offsets, above). A span
    with no reconciliation entry (never checked, or checked before this fact's own
    id existed) contributes nothing; it is not evidence of absence."""
    pages: set[int] = set()
    regions: list = []
    for span in (getattr(fact, "evidence", None) or []):
        span_id = _clean(getattr(span, "span_id", ""))
        rec = settled.get(span_id) if span_id else None
        if rec is None:
            continue
        pages.update(rec.pages)
        if rec.region is not None:
            regions.append(rec.region)
    return pages, regions


def _physical_duplicates(candidates, primary_facts, reconciliation) -> dict[str, str]:
    """`{second_event_id -> primary fact_id}` for every candidate whose RECONCILED
    document location overlaps an already-admitted primary event's reconciled
    location (issue #6 F9-R1). Same page plus overlapping region when both sides have
    region granularity; same page alone when either side lacks it -- withholding a
    reconciled region is not evidence the quotations are in different places on that
    page, so it must not sharpen the test past what was actually established.

    With no reconciliation, physical location was never established for anything and
    this check contributes nothing -- identity still rests on `propose`'s text
    coreference test alone, exactly as before this fix.
    """
    if reconciliation is None:
        return {}
    settled = reconciliation.by_span_id()
    primary_locations = [
        (fid, _reconciled_location(f, settled))
        for f in (primary_facts or [])
        for fid in [_clean(getattr(f, "fact_id", ""))] if fid]

    out: dict[str, str] = {}
    for candidate in candidates:
        cand_pages, cand_regions = _reconciled_location(candidate.fact, settled)
        if not cand_pages:
            continue
        for primary_id, (p_pages, p_regions) in primary_locations:
            if not (cand_pages & p_pages):
                continue
            if cand_regions and p_regions and not any(
                    _region_overlap(a, b) for a in cand_regions for b in p_regions):
                continue          # same page, but demonstrably different regions
            out[candidate.second_event_id] = primary_id
            break
    return out


def _source_verdict(candidate: EventCandidate, reconciliation) -> tuple[str, str]:
    """What the ORIGINAL DOCUMENT says about this candidate's quotations.

    Uses the ONE definition of "the source backs this reading" that settles axis
    disagreements, so a recovered event is held to exactly the standard a corrected axis
    is -- there is no weaker bar for an event the primary reading happened to miss.
    """
    from . import graph_consensus as _gc

    supported, proof, _text, _spans = _gc.source_support(candidate.fact, reconciliation)
    if supported:
        return ADMITTED, (
            f"a genuinely new event whose quotations the document backs ({proof})")
    if reconciliation is None:
        # `source_support` cannot fail with no reconciliation; defensive only.
        return HELD_UNVERIFIED, ("no reading of the original document was available to "
                                 "prove this event")
    from app.contracts.source_evidence import BLOCKING_STATUSES

    settled = reconciliation.by_span_id()
    outcomes = [settled.get(span_id) for span_id in candidate.span_ids]
    if any(o is not None and o.status in BLOCKING_STATUSES for o in outcomes):
        return REJECTED_SOURCE_CONTRADICTED, (
            "an independent reading of the original page does not say what this event "
            "rests on, so the document does not support adding it")
    return HELD_UNVERIFIED, (
        "no independent reading covers the page this event is quoted from, so nothing "
        "confirmed or refuted it -- obtain the page reading and decide it, never "
        "release as if the document had been read")


def admit(candidates, *, reconciliation, alignment, second_relations,
          taken_ids, id_prefix: str = "", primary_facts=None) -> Recovery:
    """Decide every undecided candidate against the document, and express the survivors
    in canonical graph terms.

    `alignment` maps second-reading event ids onto the primary event ids they describe,
    so an edge between a recovered event and an event both readings found survives the
    move into the canonical graph.

    `primary_facts` (issue #6 F9-R1) is the primary reading's own fact list, consulted
    ONLY to test a candidate's RECONCILED document location against each primary
    event's own reconciled location -- `propose` cannot run this test itself; no page
    has been read yet at that stage. None/empty degrades it to a no-op, never to a
    hold: identity still rests on `propose`'s text coreference test alone.

    Returns facts/relations for the caller to APPEND to the primary graph inputs. This
    module deliberately does not build a graph, evaluate eligibility or retrieve: doing
    any of that here would be the second code path the finding warns about.
    """
    decided = list(candidates or [])

    # Physical-location identity runs FIRST and can settle an undecided candidate
    # regardless of what `propose`'s text coreference test found: a reconciled
    # page/region match is the one identity signal both readings genuinely share, and
    # it closes the exact case a paraphrase-sensitive text test cannot (two readings
    # quoting one passage in different words).
    physical_dup = _physical_duplicates(decided, primary_facts, reconciliation)
    for candidate in decided:
        primary_id = physical_dup.get(candidate.second_event_id)
        if primary_id and not candidate.decided:
            candidate.verdict = DUPLICATE_OF_PRIMARY
            candidate.merged_into = primary_id
            candidate.reason = (
                f"an independent reading of the original document reconciles this "
                f"event's quotation to the same page/region as primary event "
                f"{primary_id}'s quotation -- the same documented event, regardless "
                f"of how differently the two readings worded it")

    for candidate in decided:
        if candidate.decided:
            continue
        candidate.verdict, candidate.reason = _source_verdict(candidate, reconciliation)

    # Canonical ids first, so an edge can be remapped onto them below.
    taken = set(taken_ids or set())
    mapping: dict[str, str] = {}
    for second_id, primary_id in (alignment or {}).items():
        if _clean(second_id) and _clean(primary_id):
            mapping[_clean(second_id)] = _clean(primary_id)
    for candidate in decided:
        # A reworded duplicate maps onto the primary event it duplicates, so an edge
        # naming it stays carryable instead of stranding the event it points at.
        if candidate.verdict == DUPLICATE_OF_PRIMARY and candidate.merged_into:
            mapping.setdefault(candidate.second_event_id, candidate.merged_into)
        if candidate.verdict in ADMITTING_VERDICTS:
            candidate.node_id = _unique_id(id_prefix, candidate.second_event_id, taken)
            mapping[candidate.second_event_id] = candidate.node_id

    # An event whose documented relational context cannot be reproduced is held, not
    # admitted: an unplaced event is exactly the one a lost PART_OF would have demoted.
    #
    # TO A FIXPOINT, and that is not defensive padding: stranding one event removes it
    # from the graph, which makes every edge naming it uncarryable in turn, which strands
    # the events on the other end of those edges. Stopping after one pass would leave a
    # recovered event admitted with exactly the demotion it depended on deleted -- the
    # extra billable line this whole control exists to avoid. `stranded` only grows and
    # is bounded by the admitted set, so the loop terminates.
    admitted_second_ids = {c.second_event_id for c in decided
                           if c.verdict in ADMITTING_VERDICTS}
    stranded: set[str] = set()
    relations: list[Any] = []
    while True:
        live = admitted_second_ids - stranded
        placed = {second_id: node_id for second_id, node_id in mapping.items()
                  if second_id not in stranded}
        relations = []
        newly: set[str] = set()
        for relation in (second_relations or []):
            endpoints = {_clean(getattr(relation, "subject_event_id", "")),
                         _clean(getattr(relation, "object_event_id", ""))}
            if not endpoints & live:
                continue                  # says nothing about a live recovered event
            carried = _remap(relation, placed)
            if carried is None:
                newly |= endpoints & live
                continue
            relations.append(carried)
        if not newly:
            break
        stranded |= newly

    for candidate in decided:
        if candidate.second_event_id not in stranded:
            continue
        candidate.verdict = HELD_UNVERIFIED
        candidate.node_id = ""
        candidate.reason = (
            "the second reading places this event in a relationship this graph "
            "cannot reproduce faithfully, so admitting it could report as separate "
            "an event the record calls part of another")

    facts = tuple(_copy_fact(c.fact, c.node_id) for c in decided
                  if c.verdict in ADMITTING_VERDICTS and c.node_id)
    return Recovery(candidates=tuple(decided), facts=facts, relations=tuple(relations))
