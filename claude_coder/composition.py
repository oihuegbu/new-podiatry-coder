"""Structural composition (issue #6 items 2/3): derive relations, and a read-time
service-intent grouping over them, from the note's own DOCUMENT STRUCTURE alone --
never from a judgement about which clinical actions are usually integral to which.
`extraction.py` already reserves, and never makes, that judgement ("emit only what
the note documents"); this module does not make it either. What it adds is a
DIFFERENT, purely structural signal extraction cannot see on its own: which events
were documented together, under the same heading, in the same section of the note.

Agnostic: `_HEADER_LINE` describes a generic clinical-note FORMATTING convention (a
short, standalone, all-caps line introducing a prose body -- the same shape whether
the heading says "PROCEDURE", "HISTORY OF PRESENT ILLNESS", or anything else), never
a specific header vocabulary, section name, or clinical term.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .models import (AttributeEvidence, ClinicalFact, RelationAssertion,
                     RelationPredicate, RelationState)

#: A section header, as a document-FORMATTING convention: the entire line, nothing
#: else on it (no same-line label:value pair -- that is what separates a real heading
#: like "PROCEDURE" from an inline field like "DOB: 1/1/1980", which always fails the
#: end-of-line anchor right after its colon), mostly uppercase, short.
_HEADER_LINE = re.compile(r"^[ \t]*([A-Z][A-Z0-9 /&()\-]{1,60})[ \t]*:?[ \t]*$")

#: Relations grouped over for `service_intents`' reachability closure -- ONLY
#: `PART_OF`, never `SAME_EPISODE_AS` (Codex F8-R4). `SAME_EPISODE_AS` asserts
#: session/episode membership -- two events were documented in the same visit, the
#: same note section -- which is exactly the CONTEXT-ONLY, non-suppressing scope
#: `eligibility.ServiceEpisode`'s own docstring already claims for it ("an episode
#: never suppresses a claim line; it just scopes ... analysis"). It does not assert
#: those events are components of ONE composed service: two independently
#: reportable procedures documented under the same heading are still two separate
#: services, and unioning them into one `ServiceIntent` would incorrectly narrow
#: retrieval/eligibility to whichever service's vocabulary happens to dominate the
#: pair. Only `PART_OF` -- an explicit, documented composition assertion -- groups
#: events for downstream relationship controls. It does not itself decide that a
#: performed component is non-reportable; reportability is decided only after each
#: service has candidates, using authoritative descriptor and claim-edit data.
_COMPOSING_PREDICATES = frozenset({RelationPredicate.PART_OF})

# Attributes whose meaning is explicitly scoped by the document section rather
# than by clinical ontology.  This is intentionally a tiny schema-level set:
# laterality is a closed claim-context axis and can be stated once in a procedure
# heading for every service documented beneath it.  Open clinical vocabularies
# (anatomy, approach, product, objective, and so on) are never propagated.
_SECTION_SCOPED_AXES = frozenset({"laterality"})


def _sections(note_text: str) -> list[tuple[str | None, int, int]]:
    """[(header text or None, start offset, end offset)], a deterministic scan of the
    verbatim note text. `header` is None for any text before the first detected
    heading (e.g. a facility/demographics block) -- that text carries no structural
    grouping signal and is deliberately never assigned to a group by guesswork."""
    out: list[tuple[str | None, int, int]] = []
    header: str | None = None
    seg_start = 0
    pos = 0
    for line in note_text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        match = _HEADER_LINE.match(stripped)
        if match:
            if pos > seg_start:
                out.append((header, seg_start, pos))
            header = match.group(1).strip()
            seg_start = pos + len(line)
        pos += len(line)
    if pos > seg_start:
        out.append((header, seg_start, pos))
    return out


def _segment_for(offset: int, sections: list[tuple[str | None, int, int]]) -> int | None:
    """The INDEX of the section this offset falls in, never the header text alone --
    two physically distinct occurrences of the identically-worded heading (e.g. a
    medications list stated both before and after the procedure) are different
    segments and must never be merged just because their headings read the same."""
    for index, (header, start, end) in enumerate(sections):
        if header is not None and start <= offset < end:
            return index
    return None


def _primary_offset(fact: ClinicalFact) -> int | None:
    """Where this fact sits in the PRIMARY transcription -- the only reading
    `note_text`'s offsets are into. A fact anchored only in an independent reading
    channel, or not anchored at all, contributes no position here (never guessed;
    see `EvidenceSpan.reading_channel_id`)."""
    for span in fact.evidence:
        if span.anchored and not span.reading_channel_id and isinstance(span.start, int):
            return span.start
    return None


def compose(facts: list[ClinicalFact], note_text: str) -> list[RelationAssertion]:
    """SAME_EPISODE_AS between every pair of facts whose primary-reading evidence
    falls in the SAME deterministically-detected section of the note.

    Symmetric and non-suppressing on its own -- matches `eligibility.ServiceEpisode`'s
    own stated philosophy ("an episode never suppresses a claim line; it just scopes
    ... analysis"). Composition never demotes anything by itself; it only supplies a
    grouping signal a later eligibility/retrieval pass (issue #6 items 4/5) can use.
    Every edge cites its two endpoint EVENTS as evidence (`bind_relation_evidence`
    resolves each `event:<fact_id>` reference into that fact's own anchored spans),
    so `validate_relations` grounds it in the same verified offsets used everywhere
    else in this graph -- never a span this module invents itself.
    """
    sections = _sections(note_text)
    by_segment: dict[int, list[str]] = {}
    for fact in facts:
        if not fact.fact_id:
            continue
        offset = _primary_offset(fact)
        if offset is None:
            continue
        segment = _segment_for(offset, sections)
        if segment is None:
            continue
        by_segment.setdefault(segment, []).append(fact.fact_id)

    relations: list[RelationAssertion] = []
    for group in by_segment.values():
        ordered = sorted(dict.fromkeys(group))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                relations.append(RelationAssertion(
                    subject_event_id=ordered[i],
                    predicate=RelationPredicate.SAME_EPISODE_AS,
                    object_event_id=ordered[j],
                    state=RelationState.ASSERTED,
                    evidence_span_ids=[f"event:{ordered[i]}", f"event:{ordered[j]}"],
                    extraction_source="structural_composition_v1",
                    confidence=1.0,
                ))
    return relations


def reconcile_section_context(facts: list[ClinicalFact], note_text: str,
                              reconciliation=None) -> list[dict]:
    """Propagate one unambiguous, source-authorized section context value.

    Clinical notes commonly state a closed context attribute once in a section
    heading or lead service and then list several component/independent services
    beneath it.  Requiring every sentence to repeat that context loses valid
    modifiers; globally copying it across an encounter is unsafe when a note has
    multiple sites or sides.  This function uses the document's physical section
    boundaries as the middle ground:

    * only facts anchored in the same deterministic section participate;
    * at least one fact must carry local, value-bound, ASSERTED evidence that the
      source reconciliation authorizes;
    * every authorized local value in the section must agree;
    * an explicit different value is never overwritten;
    * only a one-sided cross-reading omission for that same value is cleared.

    The copied evidence remains the original anchored span and is marked
    ``scope='section'`` with a stable section-context id and source fact id.  It
    therefore remains auditable and cannot be confused with sentence-local proof.
    Returns compact audit records for every propagation.
    """
    from . import graph_consensus as _gc

    sections = _sections(note_text)
    by_segment: dict[int, list[ClinicalFact]] = {}
    for fact in facts:
        offset = _primary_offset(fact)
        if offset is None:
            continue
        segment = _segment_for(offset, sections)
        if segment is not None:
            by_segment.setdefault(segment, []).append(fact)

    audit: list[dict] = []
    for segment, members in by_segment.items():
        _header, start, end = sections[segment]
        context_id = "section-context:" + hashlib.sha256(
            f"{start}:{end}:{note_text[start:end]}".encode("utf-8")
        ).hexdigest()[:16]
        for axis in _SECTION_SCOPED_AXES:
            sources: list[tuple[ClinicalFact, AttributeEvidence, str]] = []
            for fact in members:
                value = _gc.claim_authorized_value(fact, axis, reconciliation)
                if value is None:
                    continue
                for entry in (fact.attribute_evidence or {}).get(axis, ()):
                    if (entry.scope == "local"
                            and entry.assertion_state is RelationState.ASSERTED
                            and _gc._norm(entry.value) == _gc._norm(value)
                            and _gc._spans_support(
                                [entry.span], reconciliation)[0]):
                        sources.append((fact, entry, value))
            values = {_gc._norm(value) for _fact, _entry, value in sources if value}
            if len(values) != 1:
                continue
            source_fact, source_entry, value = min(
                sources,
                key=lambda item: (
                    getattr(item[1].span, "start", None)
                    if isinstance(getattr(item[1].span, "start", None), int)
                    else 10**18,
                    item[0].fact_id))

            for fact in members:
                current = str((fact.attributes or {}).get(axis) or "").strip()
                if current and _gc._norm(current) != _gc._norm(value):
                    continue
                if _gc.claim_authorized_value(fact, axis, reconciliation) is not None:
                    continue
                conflict = (fact.attribute_axis_conflicts or {}).get(axis)
                if conflict is not None:
                    stated = {_gc._norm(v) for v in
                              (conflict.value_primary, conflict.value_second) if v}
                    if stated - {_gc._norm(value)}:
                        continue
                fact.attributes[axis] = value
                entries = list((fact.attribute_evidence or {}).get(axis, ()))
                entries.append(AttributeEvidence(
                    span=source_entry.span, scope="section",
                    parent_fact_id=source_fact.fact_id,
                    source_relation_id=context_id, scope_validated=True,
                    assertion_state=RelationState.ASSERTED, value=value))
                fact.attribute_evidence[axis] = tuple(entries)
                fact.attribute_evidence_gaps.pop(axis, None)
                if conflict is not None:
                    fact.attribute_axis_conflicts.pop(axis, None)
                audit.append({
                    "fact_id": fact.fact_id,
                    "axis": axis,
                    "value": value,
                    "source_fact_id": source_fact.fact_id,
                    "source_span_id": source_entry.span.span_id,
                    "context_id": context_id,
                })
    return audit


@dataclass
class ServiceIntent:
    """One connected group of clinical events under PART_OF
    reachability -- a read-time PROJECTION over the existing graph, never a new
    persisted object (issue #6 item 2's own design constraint). A fact linked to
    nothing is still a valid, one-member intent: absence of a relation is not absence
    of an intent, it is simply an intent of one."""
    intent_id: str
    component_event_ids: list[str]


def service_intents(facts: list[ClinicalFact], relations: list[RelationAssertion]
                    ) -> list[ServiceIntent]:
    """Partition `facts` into `ServiceIntent`s by connected-component reachability
    over every ASSERTED `PART_OF` edge in `relations` (never `SAME_EPISODE_AS`, see
    `_COMPOSING_PREDICATES`) -- extracted (`extraction.py`'s) edges only today, since
    this module's own `compose()` never emits `PART_OF` itself. A `NEGATED` or
    `UNCERTAIN` edge never joins two events into one intent: only a settled,
    asserted relationship does."""
    ids = [f.fact_id for f in facts if f.fact_id]
    parent = {fid: fid for fid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rel in relations or []:
        if (rel.predicate in _COMPOSING_PREDICATES and rel.state is RelationState.ASSERTED
                and rel.subject_event_id in parent and rel.object_event_id in parent):
            ra, rb = find(rel.subject_event_id), find(rel.object_event_id)
            if ra != rb:
                parent[ra] = rb

    groups: dict[str, list[str]] = {}
    for fid in ids:
        groups.setdefault(find(fid), []).append(fid)

    intents = []
    for members in groups.values():
        ordered = sorted(members)
        intent_id = hashlib.sha256("|".join(ordered).encode("utf-8")).hexdigest()[:16]
        intents.append(ServiceIntent(intent_id=intent_id, component_event_ids=ordered))
    return intents
