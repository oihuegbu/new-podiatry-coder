"""Phase-0 evidence provenance + relation kernel (SHADOW — no release impact yet).

Two graph-compatible foundations, both behind an append-only repository seam so the
Phase-3 durable store swaps in without touching callers:

  1. EVIDENCE ANCHORING. Every quoted evidence span is anchored to an EXACT offset in
     the source and the slice is re-verified (note_text[start:end] == exact_text) — a
     plausible-looking quotation that is not verbatim in the source does NOT anchor.
     This never fabricates offsets; an unanchorable span is reported, never guessed.

  2. RELATION KERNEL. `RelationAssertion` is a first-class, content-addressed edge
     (subject, predicate, object) with identity + merge: re-assertions of the same edge
     ACCUMULATE evidentiary support instead of creating parallel edges, and any state
     disagreement collapses to UNCERTAIN (weak/conflicting relationships never become a
     confident PART_OF). No eligibility or billability decision lives here — this only
     records what the documentation asserts.

Agnostic: pure text/graph mechanics; no medical code, term, or scenario.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import EvidenceSpan, RelationAssertion, RelationState


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------- evidence anchoring
def anchor_offsets(note_text: str, exact_text: str) -> tuple[int, int] | None:
    """The [start, end) of the FIRST exact occurrence of `exact_text`, re-verified so
    the returned slice equals the quote. None when the quote is not verbatim present —
    never a fuzzy or approximate match (defensibility requires the exact span)."""
    if not exact_text:
        return None
    i = note_text.find(exact_text)
    if i < 0:
        return None
    j = i + len(exact_text)
    if note_text[i:j] != exact_text:          # invariant guard (should never trip)
        return None
    return i, j


def anchor_span(note_text: str, span: EvidenceSpan) -> EvidenceSpan:
    """Return a copy of `span` carrying its verified offsets + content hash, or an
    UNANCHORED copy (anchored=False) when the quote is not verbatim in the source."""
    pos = anchor_offsets(note_text, span.text)
    if pos is None:
        return replace(span, start=None, end=None,
                       text_sha256=_sha(span.text), anchored=False)
    i, j = pos
    return replace(span, start=i, end=j, text_sha256=_sha(span.text), anchored=True)


def anchor_facts(note_text: str, facts: list) -> list:
    """Anchor every fact's evidence spans IN PLACE (facts are mutable; spans are frozen,
    so each span is replaced with an anchored copy). Transparent to billing — only adds
    offsets/hash; the span text is unchanged."""
    for f in facts:
        f.evidence = [anchor_span(note_text, s) for s in (f.evidence or [])]
    return facts


def anchoring_report(facts: list) -> dict[str, Any]:
    """Shadow coverage report: how many evidence spans anchored, and which did not."""
    total = anchored = 0
    unanchored: list[dict] = []
    for f in facts:
        for s in (f.evidence or []):
            total += 1
            if s.anchored:
                anchored += 1
            else:
                unanchored.append({"fact_id": f.fact_id, "text": s.text[:80]})
    return {
        "spans_total": total,
        "spans_anchored": anchored,
        "coverage": (anchored / total) if total else 1.0,
        "unanchored": unanchored,
    }


# ------------------------------------------------------------------ relation kernel
def relation_key(subject_event_id: str, predicate: str, object_event_id: str) -> str:
    """Content-addressed identity for an edge — so the same (subject, predicate, object)
    asserted by different passes is ONE edge with accumulated support, not duplicates."""
    raw = f"{subject_event_id}|{predicate}|{object_event_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def merge_relations(relations: list[RelationAssertion]) -> list[RelationAssertion]:
    """Deduplicate edges by content identity. Re-assertions union their evidence and
    raise support; ANY state disagreement collapses to UNCERTAIN so a conflicting or
    weakly-asserted relationship never survives as a confident edge. Inputs are not
    mutated."""
    by_id: dict[str, RelationAssertion] = {}
    for r in relations:
        rid = r.relation_id
        if rid not in by_id:
            by_id[rid] = replace(r, evidence_span_ids=list(r.evidence_span_ids))
            continue
        m = by_id[rid]
        m.evidence_span_ids = list(dict.fromkeys(
            list(m.evidence_span_ids) + list(r.evidence_span_ids)))
        m.support += r.support
        m.confidence = max(m.confidence, r.confidence)
        if m.state is not r.state:
            m.state = RelationState.UNCERTAIN
    return list(by_id.values())


# ---------------------------------------------------- append-only repository seam
class AuditRepository:
    """Append-only shadow audit sink. Phase 3 replaces the implementation with a durable
    store (PostgreSQL) behind THIS interface, so Phase-0 callers never change."""

    def append(self, encounter_id: str, kind: str, record: dict) -> None:  # pragma: no cover
        raise NotImplementedError


class NullAuditRepository(AuditRepository):
    """Default no-op sink (tests / environments without a writable output dir)."""

    def append(self, encounter_id: str, kind: str, record: dict) -> None:
        return None


class JsonlAuditRepository(AuditRepository):
    """Per-encounter append-only JSONL artifact. Never raises into the caller — a
    shadow write failure must not affect release."""

    def __init__(self, root) -> None:
        self.root = Path(root)

    def append(self, encounter_id: str, kind: str, record: dict) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(encounter_id))
            line = json.dumps({
                "encounter_id": encounter_id,
                "kind": kind,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "record": record,
            }, default=str)
            with open(self.root / f"{safe}.jsonl", "a") as fh:
                fh.write(line + "\n")
        except Exception:
            return None
