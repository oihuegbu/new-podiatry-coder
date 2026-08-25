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
import os
import re
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoint import (ADOPT_ENV, REQUIRED_ENV, AnchorFormatError, Checkpoint,
                         CheckpointError, checkpoint_adoption_allowed, checkpoint_required,
                         resolve_checkpoint_anchor)
from .models import (EvidenceSpan, RelationAssertion, RelationPredicate,
                     RelationState)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _witness_payload(entry: dict) -> dict:
    """The sealed portion of a witness entry — everything except the seal itself."""
    return {k: v for k, v in entry.items() if k != "seal"}


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


def _span_id(document_sha256: str, document_version: str, span: EvidenceSpan,
             start: int, end: int) -> str:
    raw = "|".join((document_sha256, document_version, str(start), str(end),
                    _sha(span.text), str(span.section or ""), str(span.page or "")))
    return _sha(raw)


def anchor_span(note_text: str, span: EvidenceSpan, *, start_hint: int | None = None,
                document_version: str | None = None,
                reading_channel_id: str | None = None) -> EvidenceSpan:
    """Return a copy of `span` carrying its verified offsets + content hash, or an
    UNANCHORED copy (anchored=False) when the quote is not verbatim in the source.

    `reading_channel_id` names WHICH reading of the document `note_text` is, and is
    carried on the span so a later consumer slices the same string these offsets were
    verified against. The span id needs no extra salt for it: the id already includes
    `_sha(note_text)`, so two readings of one document produce different ids for the
    same quotation by construction."""
    doc_hash = _sha(note_text)
    version = str(document_version or doc_hash)
    hint = span.start if start_hint is None else start_hint
    pos = None
    if isinstance(hint, int) and hint >= 0:
        end = hint + len(span.text)
        if note_text[hint:end] == span.text:
            pos = (hint, end)
    if pos is None:
        pos = anchor_offsets(note_text, span.text)
    reading = str(reading_channel_id or "") or None
    if pos is None:
        return replace(span, start=None, end=None,
                       text_sha256=_sha(span.text), anchored=False,
                       document_sha256=doc_hash, document_version=version, span_id=None,
                       reading_channel_id=reading)
    i, j = pos
    return replace(span, start=i, end=j, text_sha256=_sha(span.text), anchored=True,
                   document_sha256=doc_hash, document_version=version,
                   span_id=_span_id(doc_hash, version, span, i, j),
                   reading_channel_id=reading)


def reading_of(span) -> str:
    """The reading a span's offsets belong to; "" is the primary transcription.

    ONE accessor, because every consumer that slices a document by a span's offsets has
    to agree about which string it is slicing, and a second spelling of "unset" is how
    one of them would end up slicing the wrong one."""
    return str(getattr(span, "reading_channel_id", "") or "")


def readings_map(note_text: str, extra: dict | None = None) -> dict[str, str]:
    """Every reading of the document currently in play, keyed as `reading_of` keys it."""
    out = {"": str(note_text or "")}
    for channel_id, text in (extra or {}).items():
        key = str(channel_id or "")
        if key:
            out[key] = str(text or "")
    return out


def anchor_facts(note_text: str, facts: list, document_version: str | None = None,
                 reading_channel_id: str | None = None) -> list:
    """Anchor every fact's evidence spans IN PLACE (facts are mutable; spans are frozen,
    so each span is replaced with an anchored copy). Transparent to billing — only adds
    offsets/hash; the span text is unchanged. Repeated quotations are assigned
    successive exact occurrences when available, while an extractor-supplied offset
    is accepted only after exact slice verification.

    Also anchors every `attribute_evidence` span (issue #6 F9-R5), the SAME way and
    against the SAME occurrence bookkeeping as fact-level `evidence` -- a per-attribute
    quote is proof only once it carries the same verified offsets/span_id/anchored
    status a fact-level quote does; without this, `attribute_evidence` would be inert
    data no reconciliation or axis-consensus check could ever confirm.
    """
    occurrences: dict[str, list[int]] = {}
    used: dict[str, int] = {}

    def _anchor_one(s):
        hint = s.start
        if hint is None and s.text:
            if s.text not in occurrences:
                positions, start = [], 0
                while True:
                    pos = note_text.find(s.text, start)
                    if pos < 0:
                        break
                    positions.append(pos)
                    start = pos + max(1, len(s.text))
                occurrences[s.text] = positions
            idx = used.get(s.text, 0)
            positions = occurrences[s.text]
            if idx < len(positions):
                hint = positions[idx]
                used[s.text] = idx + 1
        return anchor_span(note_text, s, start_hint=hint,
                           document_version=document_version,
                           reading_channel_id=reading_channel_id)

    for f in facts:
        f.evidence = [_anchor_one(s) for s in (f.evidence or [])]
        if getattr(f, "attribute_evidence", None):
            f.attribute_evidence = {
                attr: tuple(replace(entry, span=_anchor_one(entry.span))
                           for entry in entries)
                for attr, entries in f.attribute_evidence.items()}
    return facts


def _fact_spans(fact):
    """Every span this fact carries that provenance must anchor/reconcile against the
    source -- its own `evidence` pool AND every `attribute_evidence` quote (issue #6
    F9-R5). One shared iterator so `span_targets`/`span_targets_by_reading`/
    `apply_reconciliation` can never drift on which spans they cover."""
    yield from (fact.evidence or [])
    for entries in (getattr(fact, "attribute_evidence", None) or {}).values():
        for entry in entries:
            yield entry.span


def span_targets(facts: list) -> list:
    """Every ANCHORED quotation, as something the source-evidence layer can prove.

    Unanchored spans are excluded deliberately: a quotation that is not verbatim in
    the transcription has no character offsets, so it cannot be attributed to a page
    of the original document. It is already held by `gates.evidence_gate`, and
    inventing a page for it here would turn one honest failure into a fabricated
    location.
    """
    from app.contracts.source_evidence import SpanTarget
    targets = []
    for fact in facts:
        for span in _fact_spans(fact):
            if not span.anchored or not span.span_id:
                continue
            targets.append(SpanTarget(span_id=span.span_id, text=span.text,
                                      start=span.start, end=span.end,
                                      fact_id=fact.fact_id))
    return targets


def span_targets_by_reading(facts: list) -> dict[str, list]:
    """The same quotations, GROUPED BY the reading their offsets belong to.

    A quotation an independent reading proposed carries offsets into that reading, so
    proving it against a page requires the page arithmetic of that reading. Reconciling
    the two groups with one set of offsets is not a smaller mistake than not reconciling
    them at all -- it names a page confidently and wrongly (issue #6 F7-R3).
    """
    from app.contracts.source_evidence import SpanTarget
    out: dict[str, list] = {}
    for fact in facts:
        for span in _fact_spans(fact):
            if not span.anchored or not span.span_id:
                continue
            out.setdefault(reading_of(span), []).append(
                SpanTarget(span_id=span.span_id, text=span.text,
                           start=span.start, end=span.end, fact_id=fact.fact_id))
    return out


def _reconciled_copy(span, index):
    record = index.get(span.span_id or "")
    if record is None:
        return span
    region = record.region
    return replace(
        span,
        page=(record.pages[0] if record.pages else span.page),
        page_image_sha256=(record.page_image_sha256[0]
                           if record.page_image_sha256 else None),
        region=((region.x0, region.top, region.x1, region.bottom)
                if region is not None else None),
        source_reconciliation=record.status.value,
        verified_by_channel_id=record.verified_by_channel_id or None)


def apply_reconciliation(facts: list, reconciliation) -> list:
    """Write each quotation's ORIGINAL-DOCUMENT location and proof back onto its span.

    This is what carries page coordinates and reconciliation evidence into every
    `EvidenceSpan` — and therefore into the certificate and the ClaimBundle — rather
    than leaving them in a side report only this module reads (directive §1).

    Also writes back onto every `attribute_evidence` span (issue #6 F9-R5), the SAME
    way, so a per-attribute quote's page/region/reconciliation status is as durable and
    auditable as a fact-level quote's.
    """
    index = reconciliation.by_span_id() if reconciliation is not None else {}
    for fact in facts:
        fact.evidence = [_reconciled_copy(s, index) for s in (fact.evidence or [])]
        if getattr(fact, "attribute_evidence", None):
            fact.attribute_evidence = {
                attr: tuple(replace(entry, span=_reconciled_copy(entry.span, index))
                           for entry in entries)
                for attr, entries in fact.attribute_evidence.items()}
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
    """Deduplicate edges by content identity. Re-assertions union their evidence, their
    ASSERTION ORIGINS and their raw support; ANY state disagreement collapses to UNCERTAIN
    so a conflicting or weakly-asserted relationship never survives as a confident edge.
    Inputs are not mutated.

    `support` counts assertions; `assertion_origins` counts SOURCES. They are deliberately
    separate: one extraction response that emits the identical edge twice raises support to
    2 while contributing exactly ONE origin, so nothing downstream can read that repetition
    as two sources agreeing. Independence is a property of where an assertion came from,
    never of how many times it was written down. (Codex F6-R3.)

    Neither number is grounding. Both feed the corroboration axis, which `reconcile_relations`
    records for audit and confidence and which no claim-affecting control may accept."""
    by_id: dict[str, RelationAssertion] = {}
    for r in relations:
        rid = r.relation_id
        if rid not in by_id:
            by_id[rid] = replace(r, evidence_span_ids=list(r.evidence_span_ids),
                                 assertion_origins=list(r.assertion_origins or []),
                                 reconciliation_evidence=list(r.reconciliation_evidence or []))
            continue
        m = by_id[rid]
        m.evidence_span_ids = list(dict.fromkeys(
            list(m.evidence_span_ids) + list(r.evidence_span_ids)))
        m.assertion_origins = list(dict.fromkeys(
            [str(o).strip() for o in (m.assertion_origins or []) if str(o).strip()]
            + [str(o).strip() for o in (r.assertion_origins or []) if str(o).strip()]))
        m.support += r.support
        m.confidence = max(m.confidence, r.confidence)
        if m.state is not r.state:
            m.state = RelationState.UNCERTAIN
    return list(by_id.values())


# ------------------------------------------- directional relation-evidence grammar (config)
# Path from the RELEASE-SOURCE DECLARATION (see gates._NECESSITY_CONTROL_ID): the grammar's
# bytes decide which relations release a claim, so they are content-addressed by the same
# manifest that identifies the authoritative data. (Codex F6-R5.)
_RELATION_GRAMMAR_ID = "relation_evidence_grammar"
#: `(validated grammar, content identity of the exact bytes it was parsed from)` -- see
#: `gates._NECESSITY_CONTROL_CACHE` for why the identity lives inside the cache.
_RELATION_GRAMMAR_CACHE: tuple | None = None
# Compiled matchers, keyed by grammar version. Deliberately NOT stored inside the config
# object: a config dict must stay JSON-serialisable, or any audit record that records it
# would raise and turn a healthy encounter into a SYSTEM_HOLD.
_RELATION_PATTERN_CACHE: dict[str, dict] = {}
_REQUIRED_GRAMMAR_KEYS = ("version", "control_mode", "authority", "max_linking_chars",
                          "clause_terminators", "negation_markers",
                          "min_independent_assertions", "predicates")


class RelationGrammarError(RuntimeError):
    """The directional relation-evidence grammar is missing or malformed. Raised (never
    defaulted) so reconciliation stops instead of silently reverting to bare co-location."""


def load_relation_grammar() -> dict:
    """The reviewed directional relation-evidence grammar, fully validated. Cached per
    process. Fail-closed: a missing or malformed grammar raises rather than degrading to
    'both facts appeared nearby', which is exactly the defect this control exists to fix."""
    global _RELATION_GRAMMAR_CACHE
    if _RELATION_GRAMMAR_CACHE is not None:
        return _RELATION_GRAMMAR_CACHE[0]
    try:
        from app.release.source_manifest import declared_json_snapshot
        cfg, identity = declared_json_snapshot(_RELATION_GRAMMAR_ID, RelationGrammarError)
    except Exception as exc:
        raise RelationGrammarError(
            f"relation evidence grammar unreadable ({_RELATION_GRAMMAR_ID}): {exc}") from exc
    if not isinstance(cfg, dict):
        raise RelationGrammarError("relation evidence grammar must be a JSON object")
    missing = [k for k in _REQUIRED_GRAMMAR_KEYS if k not in cfg]
    if missing:
        raise RelationGrammarError(
            f"relation evidence grammar is missing required key(s): {missing}")
    if not cfg.get("authority"):
        raise RelationGrammarError("a control must cite its authority")
    window = cfg["max_linking_chars"]
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise RelationGrammarError("max_linking_chars must be a positive integer")
    threshold = cfg["min_independent_assertions"]
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 2:
        raise RelationGrammarError(
            "min_independent_assertions must be an integer >= 2 (one origin is not agreement)")
    for key in ("clause_terminators", "negation_markers"):
        if not isinstance(cfg[key], list) or not cfg[key] or \
                not all(isinstance(x, str) and x for x in cfg[key]):
            raise RelationGrammarError(f"{key} must be a non-empty array of strings")
    preds = cfg["predicates"]
    if not isinstance(preds, dict) or not preds:
        raise RelationGrammarError("predicates must be a non-empty object")
    for name, entry in preds.items():
        if not isinstance(entry, dict):
            raise RelationGrammarError(f"predicate {name!r} must be an object")
        cues = [c for key in ("subject_first_cues", "object_first_cues")
                for c in (entry.get(key) or [])]
        for key in ("subject_first_cues", "object_first_cues"):
            value = entry.get(key)
            if value is not None and (not isinstance(value, list)
                                      or not all(isinstance(c, str) and c.strip() for c in value)):
                raise RelationGrammarError(
                    f"predicate {name!r} {key} must be an array of non-empty strings")
        if not cues:
            raise RelationGrammarError(f"predicate {name!r} declares no directional cue")
    _RELATION_GRAMMAR_CACHE = (cfg, identity)
    return cfg


def relation_grammar_snapshot() -> dict | None:
    """The content identity of the grammar bytes THIS process parsed, or None when the
    grammar was never consulted. (Codex F6-R5-B.)"""
    return dict(_RELATION_GRAMMAR_CACHE[1]) if _RELATION_GRAMMAR_CACHE else None


def _cue_pattern(cues: list) -> "re.Pattern | None":
    """Whole-word, case-insensitive alternation over the configured phrases. Phrases may be
    multi-word; whitespace inside a phrase matches any run of whitespace."""
    parts = [r"\s+".join(re.escape(tok) for tok in str(c).split())
             for c in (cues or []) if str(c).strip()]
    if not parts:
        return None
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(parts) + r")(?![A-Za-z0-9])",
                      re.IGNORECASE)


def _grammar_patterns(grammar: dict) -> dict:
    """Compiled matchers for one loaded grammar, memoised by version beside the config."""
    key = str(grammar.get("version") or "")
    cached = _RELATION_PATTERN_CACHE.get(key)
    if cached is not None:
        return cached
    compiled = {
        "negation": _cue_pattern(grammar.get("negation_markers")),
        "terminators": tuple(grammar.get("clause_terminators") or ()),
        "window": int(grammar.get("max_linking_chars") or 0),
        "predicates": {
            str(name).strip().lower(): {
                "subject_first": _cue_pattern(entry.get("subject_first_cues")),
                "object_first": _cue_pattern(entry.get("object_first_cues")),
            } for name, entry in (grammar.get("predicates") or {}).items()
        },
    }
    _RELATION_PATTERN_CACHE[key] = compiled
    return compiled


def _links_directionally(linking_text: str, cue: "re.Pattern | None", compiled: dict) -> bool:
    """Does `linking_text` — the source text BETWEEN the two endpoint mentions — actually
    state the relationship? It must stay inside one clause, inside the configured window,
    carry no negation/enumeration marker, and contain a cue for the required orientation."""
    if cue is None or linking_text is None:
        return False
    if len(linking_text) > compiled["window"]:
        return False
    if any(t in linking_text for t in compiled["terminators"]):
        return False
    negation = compiled["negation"]
    if negation is not None and negation.search(linking_text):
        return False
    return bool(cue.search(linking_text))


class RelationIntegrityError(ValueError):
    """A relation graph cannot safely drive eligibility."""


_SYMMETRIC = {"separate_from", "same_episode_as"}


def bind_relation_evidence(relations: list[RelationAssertion], facts: list) -> list[RelationAssertion]:
    """Resolve extractor event references to verified location-specific span ids."""
    by_event = {f.fact_id: [s.span_id for s in (f.evidence or []) if s.anchored and s.span_id]
                for f in facts}
    bound = []
    for rel in relations or []:
        span_ids: list[str] = []
        for ref in rel.evidence_span_ids or []:
            if str(ref).startswith("event:"):
                span_ids.extend(by_event.get(str(ref).split(":", 1)[1], []))
            else:
                span_ids.append(str(ref))
        bound.append(replace(rel, evidence_span_ids=list(dict.fromkeys(span_ids))))
    return bound


# An edge carries TWO INDEPENDENT, SEPARATELY RECORDED provenance facts. Conflating them into
# one enum is what let model agreement stand in for documentation, so they are different
# fields with different vocabularies and only one of them can release a claim. (Codex F6-R3,
# round 5.)
#
#   1. `reconciliation_status` -- GROUNDING: what THE RECORD says about this edge. Established
#      by re-reading the source document at verified offsets. Only a grounded status may be
#      accepted by a claim-affecting control.
#   2. `corroboration_status`  -- AGREEMENT: how many distinct assertion origins said it.
#      Legitimate for the audit trail and confidence display; never grounding, because N
#      samples of one model's belief is still that model's belief.
#
# Both are written by `reconcile_relations` only; nothing an extraction model emits can set
# either, because an edge asserted by the same model that authored the events is not
# independent support for that edge.

# ---- grounding axis (what the source document establishes) --------------------------------
UNRECONCILED = "unreconciled"
# One clause of the source document STATES the directional relationship: the two endpoints'
# own verified verbatim mentions sit either side of a linking phrase from the reviewed
# grammar, in the orientation that phrase declares.
SOURCE_DIRECTIONAL = "source_directional"
# OBSERVATIONAL ONLY -- both endpoints are documented in one verified passage, but nothing in
# that passage states the DIRECTIONAL claim. Co-occurrence is not a clinical proposition, so
# this status exists to record what was seen, not to satisfy a release control.
SOURCE_COLOCATED = "source_colocated"

# The statuses that constitute INDEPENDENT GROUNDING IN THE RECORD -- the only values a
# claim-affecting control may accept. This set is the single place that answers "does this
# status mean the document itself supports the edge?", so a control config is validated
# against it (see `gates.load_necessity_control`) rather than being trusted to list only
# safe values. A status is a member because a deterministic re-read of the source proved it,
# never because assertions agreed.
GROUNDED_RECONCILIATION_STATUSES = frozenset({SOURCE_DIRECTIONAL})
# Every status this layer can stamp. UNRECONCILED and SOURCE_COLOCATED are deliberately NOT
# grounded: they record, respectively, that nothing was proved and that co-occurrence was
# observed.
RECONCILIATION_STATUSES = frozenset({UNRECONCILED, SOURCE_DIRECTIONAL, SOURCE_COLOCATED})
# Values that WERE reconciliation statuses and no longer are. Named so that a control config
# (or a persisted record) still carrying one fails loudly with the reason, instead of quietly
# matching nothing -- or, worse, being re-added by a config edit and silently reinstating the
# defect. "corroborated" moved to the corroboration axis below in round 5.
RETIRED_RECONCILIATION_STATUSES = frozenset({"corroborated", "externally_verified"})

# ---- corroboration axis (how many distinct origins asserted it) ---------------------------
# Exactly one recorded assertion origin -- or none, which scores as one for display and as
# zero support everywhere it matters.
SINGLE_ORIGIN = "single_origin"
# The same edge came from at least `min_independent_assertions` DISTINCT recorded assertion
# origins (run + provider/profile + prompt + schema). Repetition inside one response is one
# origin and therefore does not corroborate anything. This is AUDIT AND CONFIDENCE
# INFORMATION ONLY: multiple agreeing model runs -- same provider or cross-provider -- are
# still model self-confidence sampled more than once, so this value can never, on its own,
# make a relation claim-affecting. (Codex F6-R3, round 5: cross-provider agreement is not a
# magic exception; it is still not source grounding.)
MULTIPLY_ASSERTED = "multiply_asserted"
CORROBORATION_STATUSES = frozenset({SINGLE_ORIGIN, MULTIPLY_ASSERTED})


def _usable_span(span, readings: dict[str, str], shas: dict[str, str]):
    """A span may localise an endpoint only when it is anchored at exact offsets INTO A
    READING THIS CALL WAS GIVEN, and its slice still re-verifies against that reading.
    Anything else contributes no position.

    The reading is looked up rather than assumed: a quotation an independent reading of
    the document proposed carries that reading's offsets, and re-verifying it against
    the transcription would either mislocate it or -- for a passage the transcription
    never contained -- silently drop the only endpoint that could place a recovered
    event in a relationship (issue #6 F7-R3)."""
    if not getattr(span, "anchored", False) or not getattr(span, "span_id", None):
        return False
    reading = reading_of(span)
    text = readings.get(reading)
    if text is None:
        return False
    start, end = getattr(span, "start", None), getattr(span, "end", None)
    if not isinstance(start, int) or not isinstance(end, int) or isinstance(start, bool):
        return False
    if start < 0 or end > len(text) or start >= end:
        return False
    if getattr(span, "document_sha256", None) not in (None, "", shas.get(reading)):
        return False
    return text[start:end] == span.text


def _directional_proof(rel, subject_spans: list, object_spans: list,
                       readings: dict[str, str], compiled: dict) -> list[str] | None:
    """The span ids that prove `rel`'s DIRECTION, or None when the document does not state it.

    The two endpoints must have DISJOINT verified mentions: one identical passage quoted for
    both endpoints localises neither, so it can never establish which of them is the reason
    and which is the service. The text between the two mentions must then link them in the
    orientation the grammar declares for this predicate.
    """
    pred = rel.predicate.value if hasattr(rel.predicate, "value") else str(rel.predicate)
    cues = compiled["predicates"].get(str(pred).strip().lower())
    if not cues:
        return None
    cited = set(rel.evidence_span_ids or [])
    subj = sorted((s for s in subject_spans if s.span_id in cited),
                  key=lambda s: (s.start, s.end, s.span_id))
    objs = sorted((s for s in object_spans if s.span_id in cited),
                  key=lambda s: (s.start, s.end, s.span_id))
    for a in subj:
        for b in objs:
            if a.span_id == b.span_id:
                continue
            # The two mentions must sit in the SAME reading of the document. There is no
            # text "between" a passage in one reading and a passage in another, so a
            # cross-reading pair localises nothing and must never be measured as if it
            # did (issue #6 F7-R3).
            if reading_of(a) != reading_of(b):
                continue
            note_text = readings.get(reading_of(a)) or ""
            if a.end <= b.start:                       # subject ... link ... object
                orientation, linking = "subject_first", note_text[a.end:b.start]
            elif b.end <= a.start:                     # object ... link ... subject
                orientation, linking = "object_first", note_text[b.end:a.start]
            else:
                continue                               # overlapping: no separable link text
            if _links_directionally(linking, cues[orientation], compiled):
                return [a.span_id, b.span_id]
    return None


def reconcile_relations(relations: list[RelationAssertion], facts: list, note_text: str,
                        *, min_independent_assertions: int | None = None,
                        readings: dict[str, str] | None = None
                        ) -> list[RelationAssertion]:
    """Stamp each edge with the two provenance facts a control may read about it.

    GROUNDING (`reconciliation_status`) -- what the SOURCE DOCUMENT establishes, by
    deterministic, re-checkable re-reading of the note at verified offsets:

      1. SOURCE_DIRECTIONAL -- the source document itself STATES the directional relationship:
         each endpoint has its own verified verbatim mention, the two mentions are disjoint,
         and the text between them links them, within one clause, by a phrase the reviewed
         relation-evidence grammar declares for this predicate in that orientation. This is
         the only grounded status (`GROUNDED_RECONCILIATION_STATUSES`).
      2. SOURCE_COLOCATED -- observational only: one verified passage documents both endpoints
         but states no directional claim. Recorded so the audit shows what WAS seen; it is not
         an accepted justification.
      3. UNRECONCILED -- the record establishes nothing about this edge.

    A non-UNRECONCILED status ALWAYS names the verified spans that established it, so a
    certificate can never report a grounded relation while citing no source text.

    CORROBORATION (`corroboration_status`) -- how many DISTINCT recorded assertion origins
    asserted the identical (subject, predicate, object) edge. An origin is one extraction call
    (run + provider/profile + prompt + schema), so the same edge emitted twice in one response
    is ONE origin, while the same edge from two separate runs carries two. At or above
    `min_independent_assertions` (default: the reviewed grammar's, so the threshold is
    configuration that is actually READ) the edge is MULTIPLY_ASSERTED.

    The two axes are deliberately independent and are never combined here. Agreement between
    model runs -- however many, whatever provider -- is not evidence about the record, so it
    cannot raise, substitute for, or short-circuit the grounding axis. Round 4 ranked
    corroboration ABOVE directional proof in one enum, which let two agreeing runs certify an
    edge the note never states, with an empty evidence list; that ranking is gone. (Codex
    F6-R3, round 5.)

    This never raises confidence and never changes state; it only records verifiable
    provenance facts that release controls may then require.
    """
    grammar = load_relation_grammar()
    compiled = _grammar_patterns(grammar)
    threshold = int(grammar["min_independent_assertions"]
                    if min_independent_assertions is None else min_independent_assertions)
    # Every reading of the document a span may be anchored in. `note_text` is always the
    # primary transcription; an independent reading is supplied by the caller that ran
    # an extraction over it, and is never inferred here.
    readings = readings_map(note_text, readings)
    shas = {key: _sha(text) for key, text in readings.items()}
    span_ids_by_event: dict[str, set] = {}
    located_by_event: dict[str, list] = {}
    for f in facts:
        fid = getattr(f, "fact_id", None)
        if not fid:
            continue
        span_ids_by_event[fid] = {s.span_id for s in (f.evidence or [])
                                  if getattr(s, "anchored", False) and s.span_id}
        located_by_event[fid] = [s for s in (f.evidence or [])
                                 if _usable_span(s, readings, shas)]
    out: list[RelationAssertion] = []
    for rel in relations or []:
        status, proof = UNRECONCILED, []
        cited = set(rel.evidence_span_ids or [])
        shared = (span_ids_by_event.get(rel.subject_event_id, set())
                  & span_ids_by_event.get(rel.object_event_id, set()) & cited)
        directional = _directional_proof(
            rel, located_by_event.get(rel.subject_event_id, []),
            located_by_event.get(rel.object_event_id, []), readings, compiled)
        # GROUNDING: only what the document proves. Corroboration is NOT consulted here --
        # that is the whole point of the two axes being separate.
        if directional:
            status, proof = SOURCE_DIRECTIONAL, list(directional)
        elif shared:
            status, proof = SOURCE_COLOCATED, sorted(shared)
        # AGREEMENT: recorded beside the grounding, never folded into it.
        corroboration = (MULTIPLY_ASSERTED
                         if int(getattr(rel, "independent_support", 0) or 0) >= threshold
                         else SINGLE_ORIGIN)
        out.append(replace(rel, reconciliation_status=status,
                           reconciliation_evidence=list(proof),
                           corroboration_status=corroboration))
    return out


def validate_relations(relations: list[RelationAssertion], facts: list,
                       note_text: str, *, readings: dict[str, str] | None = None
                       ) -> list[RelationAssertion]:
    """Validate graph identity before a relation can suppress or separate an event.

    Invalid input fails closed rather than being silently dropped, because dropping a
    malformed edge can turn an unknown relationship into an apparently eligible line.

    `note_text` is the anchoring document: directional reconciliation re-reads the source
    between the endpoints' verified offsets, so the same text the spans were anchored to must
    be supplied. Spans that do not re-verify against it simply cannot localise an endpoint.
    """
    event_ids = {f.fact_id for f in facts if f.fact_id}
    span_ids = {s.span_id for f in facts for s in (f.evidence or []) if s.span_id}
    normalized: list[RelationAssertion] = []
    for rel in relations or []:
        if rel.subject_event_id not in event_ids or rel.object_event_id not in event_ids:
            raise RelationIntegrityError(
                f"relation {rel.relation_id} references an unknown clinical event")
        pred = rel.predicate.value if hasattr(rel.predicate, "value") else str(rel.predicate)
        if rel.subject_event_id == rel.object_event_id:
            raise RelationIntegrityError(f"relation {rel.relation_id} is self-referential")
        if not 0.0 <= float(rel.confidence) <= 1.0:
            raise RelationIntegrityError(f"relation {rel.relation_id} has invalid confidence")
        if not rel.evidence_span_ids:
            raise RelationIntegrityError(f"relation {rel.relation_id} has no anchored evidence")
        if set(rel.evidence_span_ids or []) - span_ids:
            raise RelationIntegrityError(
                f"relation {rel.relation_id} references unverified evidence spans")
        current = rel
        if pred in _SYMMETRIC and rel.subject_event_id > rel.object_event_id:
            current = replace(rel, subject_event_id=rel.object_event_id,
                              object_event_id=rel.subject_event_id)
        normalized.append(current)
    # Merge first (so re-assertions accumulate support and conflicting states collapse to
    # UNCERTAIN), THEN record how each surviving edge was reconciled. A release control that
    # requires reconciliation therefore reads a value this deterministic layer wrote, never
    # one the extraction model supplied. (Codex F6-R3.)
    return reconcile_relations(merge_relations(normalized), facts, note_text,
                               readings=readings)


def validate_attribute_evidence(facts: list, relations: list) -> list:
    """Stamp `scope_validated=True` on every `attribute_evidence` entry that has
    actually earned trust, IN PLACE (issue #6 F9-R5-A) -- must run AFTER
    `validate_relations` (which also reconciles), on the SAME `relations` list it
    returned, never on the raw pre-reconciliation list: only a RECONCILED relation
    carries a trustworthy `reconciliation_status`, and a merge can also change a
    relation's `state` (conflicting states collapse to UNCERTAIN).

    A `"local"` entry needs no relation and is always validated -- its evidence is the
    fact's own sentence. An `"inherited"` entry is validated only when ALL of:

      * a relation with that exact `source_relation_id` exists in `relations`;
      * its predicate is `PART_OF` -- never `SAME_EPISODE_AS`, which establishes only
        that two events belong to one encounter, not that they share any specific
        code-changing attribute (laterality, anatomy, product, count, approach, ...);
      * its state is `ASSERTED` -- never negated, uncertain, or (after merge) collapsed
        to uncertain by a conflicting assertion;
      * its direction is EXACTLY `subject_event_id == fact.fact_id` and
        `object_event_id == entry.parent_fact_id` -- this fact IS PART_OF the named
        parent; the reverse direction never authorizes inheritance, matching PART_OF's
        own directional semantics (a parent is not part of its own component);
      * its `reconciliation_status` is a member of `GROUNDED_RECONCILIATION_STATUSES`
        -- the source document itself states the relationship, not merely an
        unreconciled assertion or bare co-location.

    Anything short of that is DROPPED from the fact's `attribute_evidence` entirely,
    never kept unvalidated -- an axis-consensus check that later sees an entry at all
    may trust it was validated; nothing downstream re-derives this.
    """
    by_relation = {r.relation_id: r for r in (relations or [])}
    for fact in facts:
        raw = getattr(fact, "attribute_evidence", None) or {}
        if not raw:
            continue
        cleaned: dict[str, tuple] = {}
        for axis, entries in raw.items():
            kept = []
            for entry in entries:
                if entry.scope == "local":
                    kept.append(replace(entry, scope_validated=True))
                    continue
                rel = by_relation.get(entry.source_relation_id)
                valid = bool(
                    rel is not None
                    and rel.state is RelationState.ASSERTED
                    and rel.predicate is RelationPredicate.PART_OF
                    and rel.subject_event_id == fact.fact_id
                    and rel.object_event_id == entry.parent_fact_id
                    and rel.reconciliation_status in GROUNDED_RECONCILIATION_STATUSES)
                if valid:
                    kept.append(replace(entry, scope_validated=True))
            if kept:
                cleaned[axis] = tuple(kept)
        fact.attribute_evidence = cleaned
    return facts


# ---------------------------------------------------- append-only repository seam
class AuditRepository:
    """Append-only shadow audit sink. Phase 3 replaces the implementation with a durable
    store (PostgreSQL) behind THIS interface, so Phase-0 callers never change."""

    def append(self, encounter_id: str, kind: str, record: dict) -> str:  # pragma: no cover
        raise NotImplementedError


class NullAuditRepository(AuditRepository):
    """Default no-op sink (tests / environments without a writable output dir)."""

    def append(self, encounter_id: str, kind: str, record: dict) -> str:
        payload = {"encounter_id": encounter_id, "kind": kind, "record": record}
        return _sha(json.dumps(payload, sort_keys=True, default=str))


class SqliteAuditRepository(AuditRepository):
    """Phase 3: durable, hash-chained, append-only provenance store in a DEDICATED SQLite
    database (WAL, synchronous=FULL, INSERT-only enforced by triggers), kept separate from
    the authoritative compliance.db (different lifecycle/retention). Same contract as
    JsonlAuditRepository: append() returns the entry's record_sha256, entries are per-
    encounter hash-chained, and in strict mode a write failure RAISES so the pipeline routes
    SYSTEM_HOLD (release provenance is fail-closed, never an empty-success fallback).

    Swap PostgreSQL in behind AuditRepository only if a scale trigger appears (multi-host
    concurrent writers, high write concurrency, or HA-replication/PITR compliance) -- the
    seam and the hash-chained record shape are unchanged, so it is not a rewrite.
    """

    _DDL = (
        "CREATE TABLE IF NOT EXISTS audit_log ("
        " seq INTEGER PRIMARY KEY AUTOINCREMENT,"
        " encounter_id TEXT NOT NULL,"
        " kind TEXT NOT NULL,"
        " recorded_at TEXT NOT NULL,"
        " control_mode TEXT NOT NULL,"
        " previous_record_sha256 TEXT NOT NULL,"
        " record_json TEXT NOT NULL,"
        " record_sha256 TEXT NOT NULL,"
        # The witness record that sealed this row, so the anchoring is BIDIRECTIONAL: the
        # witness proves the row exists, and the row proves the witness line exists. Erasing
        # the terminal record now requires consistently truncating BOTH sides. (Codex F6-R4-A.)
        " witness_sha256 TEXT)",
        "CREATE INDEX IF NOT EXISTS idx_audit_encounter ON audit_log(encounter_id, seq)",
        "CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log"
        " BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END",
        "CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log"
        " BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END",
    )

    def __init__(self, db_path, *, strict: bool = True, checkpoint_anchor=None) -> None:
        self.db_path = Path(db_path)
        self.strict = strict
        # Injected only by tests/callers that supply their own backend; production
        # resolves the configured one per call so a configuration change takes effect
        # without a restart and so a broken spec fails closed at the point of use.
        self._injected_anchor = checkpoint_anchor

    def _connect(self):
        import sqlite3
        if self.db_path.parent and str(self.db_path.parent):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None -> manual transaction control, so append() can take the
        # write lock with BEGIN IMMEDIATE before reading the previous hash (Codex F6-R4).
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        # busy_timeout FIRST: switching journal mode and running the DDL below both take
        # locks, and until the busy handler is installed a concurrent opener gets an immediate
        # SQLITE_BUSY ("database is locked") instead of waiting. Ordering this after the
        # journal-mode switch made simultaneous first-opens fail. (Post-fix review.)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        for stmt in self._DDL:
            conn.execute(stmt)
        # Forward migration for stores created before the witness column existed. ADD COLUMN
        # is DDL, so the append-only UPDATE/DELETE triggers do not (and must not) block it;
        # pre-existing rows keep a NULL witness and verify_chain reports them as unwitnessed
        # rather than silently accepting them.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(audit_log)").fetchall()}
        if "witness_sha256" not in cols:
            try:
                conn.execute("ALTER TABLE audit_log ADD COLUMN witness_sha256 TEXT")
            except sqlite3.OperationalError as exc:
                # A concurrent opener may have migrated between the probe and here. Adopt its
                # result; anything else is a real schema failure and must surface.
                if "duplicate column name" not in str(exc).lower():
                    raise
        return conn

    def append(self, encounter_id: str, kind: str, record: dict) -> str:
        try:
            conn = self._connect()
            try:
                # Serialize the read-previous + append so two concurrent writers cannot both
                # observe the same predecessor and fork the per-encounter hash chain. The
                # RESERVED lock is taken at BEGIN IMMEDIATE (before the SELECT), and other
                # writers block up to busy_timeout instead of racing. (Codex F6-R4.) The same
                # lock serializes the witness append, so the witness chain cannot fork either.
                conn.execute("BEGIN IMMEDIATE")
                # Fail CLOSED before extending a chain whose external witness was destroyed
                # or rewritten: a writer must never quietly continue on top of a broken
                # anchor. It compares the DB's terminal row against the witness tail AND
                # the witness tail against the externally anchored checkpoint, so a
                # CONSISTENT truncation of both mutable sides is refused here, on the
                # release path, rather than reported after the fact. (Codex F6-R4-A.)
                # ONE anchor object for the whole append. Production resolves a fresh one
                # per call so a configuration change takes effect without a restart -- but
                # WITHIN an append the three uses (verify, mark, advance) must be the same
                # anchor, or a configuration change landing mid-append could seal
                # `anchored: true` into the journal and then advance nothing, leaving a
                # committed row whose entry claims an anchor that never recorded it. That
                # entry would be permanent, unforgeable evidence of a lie, and it would make
                # the store un-adoptable forever. (Post-fix review.)
                anchor = self._anchor()
                in_doubt_over, witness_seq = self._assert_witness_extendable(
                    conn, anchor=anchor)
                row = conn.execute(
                    "SELECT record_sha256 FROM audit_log WHERE encounter_id=?"
                    " ORDER BY seq DESC LIMIT 1", (str(encounter_id),)).fetchone()
                previous = row[0] if row else ""
                entry = {
                    "encounter_id": encounter_id,
                    "kind": kind,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "control_mode": "ENFORCED_FAIL_CLOSED",
                    "previous_record_sha256": previous,
                    "record": record,
                }
                entry["record_sha256"] = _sha(json.dumps(entry, sort_keys=True, default=str))
                # ORDERING IS THE ATOMICITY CONTRACT (Codex F6-R4-A): the witness record is
                # written and fsynced BEFORE the row is committed. Therefore
                #   - witness write fails  -> rollback; NO committed row and no witness: clean;
                #   - commit fails/crashes -> the witness is ahead by exactly one record, an
                #     explicitly recoverable IN-DOUBT tail that `_assert_witness_extendable`
                #     tolerates once and `verify_chain` reports by name.
                # The previous order (commit, then witness) could leave a committed row with
                # no witness at all -- indistinguishable from a truncated witness.
                new_seq = witness_seq + 1
                # Whether this entry is being written UNDER a configured anchor is sealed
                # into the entry itself, so the journal carries permanent, unforgeable
                # evidence of whether this store was ever anchored. That is what makes the
                # legacy-adoption path safe rather than a bypass: see
                # `_prior_anchoring_evidence`.
                witness_sha = self._append_head_witness(
                    encounter_id, entry["record_sha256"], in_doubt_over=in_doubt_over,
                    expected_seq=new_seq, anchored=bool(anchor.configured))
                # ...and the externally anchored checkpoint advances between the journal
                # fsync and the commit, so the anchor is never BEHIND by more than the one
                # entry a crash can lose, and is never AHEAD of the journal in healthy
                # operation. "Anchor ahead of journal" therefore has exactly one meaning:
                # journal entries were removed. A failure here aborts the append (no row is
                # committed), which is the required posture -- an unverifiable anchor must
                # hold the release, not be logged and ignored.
                self._advance_checkpoint(new_seq, witness_sha, entry["record_sha256"],
                                         anchor=anchor)
                conn.execute(
                    "INSERT INTO audit_log(encounter_id, kind, recorded_at, control_mode,"
                    " previous_record_sha256, record_json, record_sha256, witness_sha256)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (encounter_id, kind, entry["recorded_at"], entry["control_mode"],
                     previous, json.dumps(record, sort_keys=True, default=str),
                     entry["record_sha256"], witness_sha))
                conn.commit()
                return entry["record_sha256"]
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise
            finally:
                conn.close()
        except Exception:
            if self.strict:
                raise
            return ""

    def records(self, encounter_id: str | None = None) -> list[dict]:
        """Read the chained records back (verification/audit; not release-gating)."""
        conn = self._connect()
        try:
            q = ("SELECT encounter_id, kind, recorded_at, control_mode,"
                 " previous_record_sha256, record_json, record_sha256 FROM audit_log")
            args: tuple = ()
            if encounter_id is not None:
                q += " WHERE encounter_id=?"
                args = (str(encounter_id),)
            q += " ORDER BY seq"
            return [{"encounter_id": r[0], "kind": r[1], "recorded_at": r[2],
                     "control_mode": r[3], "previous_record_sha256": r[4],
                     "record": json.loads(r[5]), "record_sha256": r[6]}
                    for r in conn.execute(q, args).fetchall()]
        finally:
            conn.close()


    # ------------------------------------------------------- external terminal-head witness
    # The witness is a SEALED, hash-chained, append-only journal kept outside the audit log.
    # Every entry is an HMAC-SHA256 over its canonical payload AND its predecessor's seal, so
    # a line cannot be edited, reordered, removed from the middle, or forged without the key.
    # The audit row stores the seal of the witness entry that sealed it, so the anchoring runs
    # in BOTH directions. Missing / empty / corrupt / duplicated / partial witness state is an
    # INTEGRITY FAILURE whenever durable rows exist -- never an empty-success `{}`.
    #
    # TRUST BOUNDARY: the HMAC key is read from the PROVENANCE_WITNESS_KEY environment
    # variable when set, which puts key custody OUTSIDE the filesystem the audit writer owns
    # -- a process that can reach the database and the witness file still cannot forge or
    # shorten the journal without the key. When no key is provisioned, one is generated once
    # into a 0600 sidecar so the seal is still tamper-evident against every writer that does
    # not hold it; that weaker default is recorded, not silently assumed, by `witness_status`.
    #
    # WHAT THE KEY CANNOT DO, AND WHY THERE IS A CHECKPOINT ANCHOR (Codex F6-R4-A):
    # removing the LAST journal entry and the LAST durable row TOGETHER leaves a prefix that
    # is internally consistent under any key -- the seals that remain are genuine, the chain
    # links still join, and the legitimate writer (which holds the key) extends the shortened
    # journal without noticing. Detecting that requires an expected terminal POSITION recorded
    # outside both mutable objects. So every journal entry now carries a monotonic `seq`
    # inside its sealed payload, and `checkpoint.CheckpointAnchor` records (seq, seal) in a
    # separate store. "Anchor ahead of journal" is then unambiguous evidence of removal.
    # The anchor is only as strong as its backing store, which is why the backend declares
    # `external` and why an unconfigured anchor is REPORTED rather than assumed benign.
    _WITNESS_VERSION = "audit-head-witness-v3"

    def _witness_path(self):
        return Path(str(self.db_path) + ".heads")

    def _witness_key_path(self):
        return Path(str(self.db_path) + ".witnesskey")

    def _witness_key(self, *, create: bool = False) -> bytes:
        """The sealing key: externally provisioned when available, else a 0600 sidecar.

        `create` is False on every READ path. Verification must never mint a key: doing so
        would let a store whose key was deleted be re-sealed under a fresh key, and would
        make a read-only integrity check write to disk. A missing key with entries present
        is an integrity failure, reported by `read_witness`.
        """
        env = os.getenv("PROVENANCE_WITNESS_KEY")
        if env and env.strip():
            return env.strip().encode("utf-8")
        kp = self._witness_key_path()
        if kp.exists():
            data = kp.read_bytes().strip()
            if not data:
                raise OSError(f"audit witness key at {kp} is empty")
            return data
        if not create:
            raise OSError(
                f"audit witness seal key is unavailable ({kp} is absent and "
                f"PROVENANCE_WITNESS_KEY is unset); witness entries cannot be verified")
        import secrets
        kp.parent.mkdir(parents=True, exist_ok=True)
        # PUBLISH ATOMICALLY. Creating the key file with O_EXCL and *then* writing it leaves a
        # window in which a racing reader sees a real but EMPTY key file and fails. So the key
        # is written to a private temp file, fsynced, and linked into place: `os.link` is
        # atomic and refuses to clobber, so the key file only ever appears complete, and a
        # racer that loses simply adopts the winner's key rather than overwriting a key that
        # may already have sealed entries. (Post-fix review.)
        key = secrets.token_hex(32).encode("ascii")
        tmp = kp.parent / f"{kp.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, key)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(str(tmp), str(kp))
        except FileExistsError:
            pass                       # another writer published first: adopt theirs
        finally:
            try:
                os.unlink(str(tmp))
            except OSError:
                pass
        data = kp.read_bytes().strip()
        if not data:
            raise OSError(f"audit witness key at {kp} is empty")
        return data

    def key_custody(self) -> str:
        """`external` when the seal key is provisioned outside this store's own directory."""
        env = os.getenv("PROVENANCE_WITNESS_KEY")
        return "external" if env and env.strip() else "adjacent-keyfile"

    def _seal(self, payload: dict, *, create_key: bool = False) -> str:
        import hmac
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hmac.new(self._witness_key(create=create_key), canonical.encode("utf-8"),
                        hashlib.sha256).hexdigest()

    def _witness_tail(self) -> tuple[dict | None, int]:
        """`(last entry, its sequence number)`; `(None, 0)` when the journal does not exist.

        The sequence is read from the entry's SEALED payload, so it cannot be changed without
        the key. Entries written before sequencing existed carry none; for those it falls back
        to the entry's 1-based position, which is what the sequence would have been. That
        fallback is derived from the journal itself and so is worth nothing on its own -- its
        value is that the anchor recorded it, and the anchor is what a truncation cannot reach.

        Raises when the journal exists but its tail is unreadable/unsealed -- a corrupt tail
        must stop the writer, not be skipped."""
        wp = self._witness_path()
        if not wp.exists():
            return None, 0
        lines = [ln for ln in wp.read_text().splitlines() if ln.strip()]
        if not lines:
            raise OSError(f"audit head witness {wp} exists but is empty")
        try:
            tail = json.loads(lines[-1])
        except Exception as exc:
            raise OSError(f"audit head witness {wp} has a corrupt final entry: {exc}") from exc
        if not isinstance(tail, dict) or self._seal(_witness_payload(tail)) != tail.get("seal"):
            raise OSError(f"audit head witness {wp} final entry fails its seal")
        seq = tail.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            if seq is not None:
                raise OSError(f"audit head witness {wp} final entry has an invalid sequence")
            seq = len(lines)                       # pre-sequencing entry: positional fallback
        return tail, int(seq)

    # -------------------------------------------------- externally anchored terminal head
    def store_id(self) -> str:
        """Stable identity of THIS store in the checkpoint anchor. Overridable so several
        deployments can share one anchor store without colliding."""
        override = str(os.getenv("PROVENANCE_STORE_ID") or "").strip()
        return override or self.db_path.name

    def _anchor(self):
        """The configured checkpoint anchor. Resolution failures propagate: a misspelled or
        unimplemented backend must stop the writer, not be read as 'no anchor configured'."""
        if self._injected_anchor is not None:
            return self._injected_anchor
        return resolve_checkpoint_anchor()

    def _tail_append_committed(self, tail: dict | None, conn=None) -> bool:
        """Did the append that wrote the journal's TERMINAL entry commit its durable row?

        The ordering contract is: seal + fsync the journal entry -> advance the checkpoint
        anchor -> commit the row. So a committed row bearing the tail's seal is proof that
        the anchor advance for that sequence SUCCEEDED. Conversely a tail with no committed
        row is the one legitimate state in which an entry can be marked anchored while the
        anchor holds nothing for it (the anchor write failed, aborting that append).

        With no connection (the reporting path) the answer is unknown, and unknown resolves
        to True -- the conservative direction, since it withholds the one tolerance the
        adoption path grants rather than widening it.
        """
        if tail is None:
            return False
        if conn is None:
            return True
        row = conn.execute(
            "SELECT witness_sha256 FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
        return bool(row and row[0] and str(row[0]) == str(tail.get("seal")))

    def _prior_anchoring_evidence(self, tail: dict | None, tail_seq: int,
                                  conn=None) -> int | None:
        """The sequence of a journal entry that PROVES this store was already anchored, or
        None when nothing in the journal proves it.

        Read on the adoption path only. Adoption exists for exactly one shape -- a journal
        that PREDATES the anchor, for which "the anchor holds no checkpoint" is simply the
        truth. For a store that WAS anchored, that same emptiness means the anchored
        checkpoint was destroyed or the anchor was repointed, which is precisely the attack
        the anchor exists to catch, and no operator switch may wave it through.

        The evidence is the sealed `anchored` mark. It cannot be stripped without the
        sealing key (the seal covers it) and it cannot be reached by deleting the anchor
        store, so it survives exactly the manipulation that produces the adoption prompt.

        Exactly one tolerance: the TERMINAL entry may carry the mark with no checkpoint
        behind it, because the mark is sealed before the anchor is advanced -- and only
        while that append is still uncommitted (see `_tail_append_committed`). Every entry
        below the tail is proof: the entry after it exists, so its own append ran to
        completion, and completion requires the anchor advance to have succeeded.

        A journal that cannot be read cleanly cannot answer the question at all, so it
        REFUSES rather than defaulting to "no evidence found" -- the empty-success failure
        mode this control exists to avoid.
        """
        entries, problems = self.read_witness()
        if problems:
            raise OSError(
                f"refusing to adopt store {self.store_id()!r}: its witness journal does not "
                f"read cleanly, so whether it was already anchored cannot be established "
                f"({'; '.join(problems[:3])})")
        tolerated = None if self._tail_append_committed(tail, conn) else int(tail_seq)
        for entry in entries:
            if not entry.get("anchored"):
                continue
            raw = entry.get("seq")
            seq = int(raw) if isinstance(raw, int) and not isinstance(raw, bool) else -1
            if seq != tolerated:
                return seq
        return None

    def _assert_checkpoint_extendable(self, tail: dict | None, tail_seq: int,
                                      conn=None, anchor=None) -> None:
        """Fail closed unless the journal's terminal head still matches its EXTERNAL anchor.

        This is the check the self-anchored witness structurally could not make. The anchor is
        advanced after a journal entry is fsynced and before the row is committed, so in
        healthy operation `anchor.seq == journal.seq`, and the only reachable divergence is
        the anchor lagging by ONE (its write failed, which aborted that append). Therefore:

          * anchor ahead of the journal            -> journal entries were REMOVED  -> refuse
          * same sequence, different seal          -> journal rewritten/rolled back -> refuse
          * journal more than one entry ahead      -> unanchored extension          -> refuse
          * anchor unreachable / malformed / unsigned -> UNVERIFIABLE                -> refuse

        Refusing means `append()` raises, which the pipeline turns into SYSTEM_HOLD. An
        integrity control that cannot verify itself must hold the release, not warn.

        `conn` is the write path's open connection, used only to tell an uncommitted
        terminal append from a completed one on the adoption path. It is optional because
        the reporting path (`checkpoint_status`) has none, and its absence only ever
        narrows what adoption tolerates.
        """
        anchor = self._anchor() if anchor is None else anchor
        if not anchor.configured:
            if checkpoint_required():
                raise OSError(
                    f"{REQUIRED_ENV} is set but no terminal-head checkpoint anchor is "
                    f"configured; refusing to extend a self-anchored journal")
            return          # reported by `checkpoint_status()`, never silently assumed benign
        try:
            stored = anchor.read(self.store_id())
        except AnchorFormatError as exc:
            # A record that exists but is not well-formed is a rewritten checkpoint, not an
            # absent one -- reported precisely so the operator knows which failure they have.
            raise OSError(f"anchored terminal-head checkpoint is malformed: {exc}") from exc
        except CheckpointError as exc:
            raise OSError(f"terminal-head checkpoint anchor is unverifiable: {exc}") from exc
        if stored is None:
            if tail is None:
                return                       # first ever append: this establishes the anchor
            if checkpoint_adoption_allowed():
                # ADOPTION IS NOT A BYPASS (Codex F6-R4-A finding A). Operator consent
                # covers the deploy boundary -- a journal written before the anchor
                # existed -- and nothing else. A store that carries sealed evidence of
                # having been anchored, whose anchor now holds nothing, is a DESTROYED
                # anchor, and it fails closed no matter what this switch says.
                proven_at = self._prior_anchoring_evidence(tail, tail_seq, conn)
                if proven_at is None:
                    return                   # explicit, reviewed one-time adoption
                raise OSError(
                    f"refusing to adopt store {self.store_id()!r}: its witness journal "
                    f"records that entry {proven_at} was written under a checkpoint anchor, "
                    f"so a checkpoint for it WAS anchored and is now absent -- the anchored "
                    f"checkpoint was destroyed, or the anchor was repointed at a store that "
                    f"does not hold this journal's history. {ADOPT_ENV} adopts a journal "
                    f"that PREDATES the anchor; it cannot re-adopt one whose anchor was "
                    f"removed")
            raise OSError(
                f"the terminal-head checkpoint anchor holds no checkpoint for store "
                f"{self.store_id()!r} although its witness journal has {tail_seq} entries -- "
                f"the anchored checkpoint was removed, or the anchor was enabled on an "
                f"existing store; refusing to extend (set {ADOPT_ENV}=1 for a one-time, "
                f"reviewed adoption)")
        try:
            cp = Checkpoint.from_record(stored)
        except CheckpointError as exc:
            raise OSError(f"anchored terminal-head checkpoint is malformed: {exc}") from exc
        import hmac
        # Verified with the SEALING key on a read path, so `create=False`: verification must
        # never mint a key (the same rule the journal seal follows), and a checkpoint whose
        # signature cannot be computed is unverifiable, not acceptable.
        if not hmac.compare_digest(self._seal(cp.payload()), str(cp.signature)):
            raise OSError(
                "anchored terminal-head checkpoint fails its signature (forged or rewritten "
                "in the anchor store)")
        if cp.store_id != self.store_id():
            raise OSError(
                f"anchored checkpoint belongs to store {cp.store_id!r}, not "
                f"{self.store_id()!r}")
        if tail is None:
            raise OSError(
                f"the anchor records witness journal entry {cp.seq} but the journal is empty "
                f"-- the journal was destroyed or replaced")
        if cp.seq > tail_seq:
            raise OSError(
                f"the witness journal is SHORTER than its anchored checkpoint (anchor seq "
                f"{cp.seq}, journal seq {tail_seq}) -- the durable tail and the journal tail "
                f"were truncated together")
        if cp.seq == tail_seq and str(cp.seal) != str(tail.get("seal")):
            raise OSError(
                f"witness journal entry {tail_seq} does not match the anchored checkpoint "
                f"-- the journal was rewritten, rolled back or replayed")
        if cp.seq < tail_seq - 1:
            raise OSError(
                f"the witness journal has {tail_seq - cp.seq} entries beyond its anchored "
                f"checkpoint (anchor seq {cp.seq}, journal seq {tail_seq}); at most one "
                f"unanchored entry is recoverable")

    def _advance_checkpoint(self, seq: int, seal: str, record_sha256: str,
                            anchor=None) -> dict | None:
        """Record the new terminal head in the external anchor, signed with the sealing key.

        Called between the journal fsync and the row commit, so the anchor can only ever lag
        by the single entry a crash can lose. Any failure raises, aborting the append.

        `anchor` is passed by `append()` so the object that VERIFIED the head, the one whose
        `configured` flag was sealed into the journal entry, and the one advanced here are
        all the same object."""
        anchor = self._anchor() if anchor is None else anchor
        if not anchor.configured:
            return None
        cp = Checkpoint(
            store_id=self.store_id(), seq=int(seq), seal=str(seal),
            record_sha256=str(record_sha256), witness_version=self._WITNESS_VERSION,
            written_at=datetime.now(timezone.utc).isoformat())
        signed = replace(cp, signature=self._seal(cp.payload(), create_key=True))
        try:
            anchor.write(signed.as_record())
        except CheckpointError as exc:
            raise OSError(f"terminal-head checkpoint could not be anchored: {exc}") from exc
        return signed.as_record()

    def checkpoint_status(self) -> dict:
        """Observability for the anchor: which backend, whether it is a real trust boundary,
        the anchored vs journal sequence, and any problem that would hold a release. Total --
        it reports failures instead of raising, because it is also called from `verify_chain`.
        """
        # `adoption_allowed` is reported because it is a WEAKENED posture: while it is set, a
        # store whose journal predates the anchor may start anchoring mid-history. It is a
        # one-run migration switch, and an artifact produced while it was left on must say so
        # rather than imply the anchor covered the whole journal. It is deliberately NOT a
        # `problem`: the migration run itself must be able to complete.
        status: dict[str, Any] = {"store_id": self.store_id(),
                                  "required": checkpoint_required(),
                                  "adoption_allowed": checkpoint_adoption_allowed()}
        problems: list[str] = []
        try:
            anchor = self._anchor()
            status.update(anchor.describe())
        except Exception as exc:
            status.update({"backend": "unresolved", "configured": False,
                           "external_trust_boundary": False})
            problems.append(f"terminal-head checkpoint anchor is unusable: {exc}")
            status["problems"] = problems
            return status
        tail, tail_seq, tail_readable = None, 0, True
        try:
            tail, tail_seq = self._witness_tail()
        except Exception as exc:
            tail_readable = False
            problems.append(f"witness journal tail is unreadable: {exc}")
        status["journal_seq"] = tail_seq
        if status.get("configured"):
            try:
                stored = anchor.read(self.store_id())
                status["anchored_seq"] = (
                    Checkpoint.from_record(stored).seq if stored else None)
            except Exception as exc:
                status["anchored_seq"] = None
                problems.append(f"anchored checkpoint could not be read: {exc}")
        # Only compare against the anchor when the journal's own tail was READABLE. An
        # unreadable tail is already reported above; feeding the comparison a placeholder
        # would add a second, WRONG diagnosis ("the journal is empty") for a journal that is
        # merely corrupt. The write path is unaffected -- there the same unreadable tail
        # raises before any comparison happens, so nothing is skipped. (Post-fix review.)
        if tail_readable:
            try:
                self._assert_checkpoint_extendable(tail, tail_seq, anchor=anchor)
            except Exception as exc:
                problems.append(str(exc))
        status["problems"] = problems
        return status

    def _assert_witness_extendable(self, conn, anchor=None) -> tuple[str | None, int]:
        """Refuse to extend a chain whose external witness no longer matches the log.

        Returns `(seal of an observed IN-DOUBT tail or None, the journal's terminal sequence)`
        so the next entry can record that it was written over one, and so the caller knows
        which sequence it is extending. Compares the log's GLOBAL terminal row against the
        witness tail, and the witness tail against the EXTERNAL anchor. Exactly two states of
        the row-vs-journal comparison are healthy:
          - aligned: the terminal row's witness seal IS the witness tail (normal), or
          - in-doubt by one: the tail's predecessor is the terminal row's seal, i.e. a write
            crashed between the witness fsync and the commit (documented, recoverable).
        Anything else means the witness was truncated, replaced, or rewritten, or that rows
        exist with no witness at all -- both fail closed.

        KNOWN, DELIBERATE LIMIT: at write time a crashed commit and a deleted terminal ROW are
        indistinguishable from O(1) local state, so exactly ONE in-doubt tail is tolerated (a
        crash must be recoverable). The anomaly is not lost: the divergence is permanent, the
        next entry records `in_doubt_over`, and `verify_chain` reports it by name for as long
        as the store exists.
        """
        row = conn.execute(
            "SELECT record_sha256, witness_sha256 FROM audit_log ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        tail, tail_seq = self._witness_tail()
        # The EXTERNAL check runs first and unconditionally: the whole point is that it is the
        # only one a consistent truncation of both local sides cannot satisfy, so it must not
        # sit behind a branch that a truncated store can take.
        self._assert_checkpoint_extendable(tail, tail_seq, conn, anchor=anchor)
        if row is None:
            if tail is not None:
                raise OSError(
                    "audit head witness records entries but the audit log is empty -- the "
                    "durable log was truncated or replaced")
            return None, tail_seq
        record_sha, witness_sha = row[0], row[1]
        if not witness_sha:
            # DEPLOY BOUNDARY. A store written before this control existed has rows with no
            # seal and no journal. That is a legacy prefix, not an attack, and bricking every
            # future append on it would be a self-inflicted outage -- so such a store may
            # ADOPT the witness starting now. But if a journal exists, or any row IS sealed,
            # then an unsealed terminal row means someone appended around the witness (or
            # destroyed it), which fails closed.
            sealed_exists = conn.execute(
                "SELECT 1 FROM audit_log WHERE witness_sha256 IS NOT NULL"
                " AND witness_sha256 <> '' LIMIT 1").fetchone()
            if tail is not None or sealed_exists:
                raise OSError(
                    "the terminal audit row carries no external witness seal although this "
                    "store is witnessed; refusing to extend a chain that was appended to "
                    "around the witness")
            # legacy pre-witness store: the journal starts with this append
            return None, tail_seq
        if tail is None:
            raise OSError("audit rows exist but the external head witness is missing")
        if tail.get("seal") == witness_sha:
            return None, tail_seq
        if tail.get("previous_seal") == witness_sha and tail.get("record_sha256") != record_sha:
            # in-doubt tail: witness ahead by exactly one entry
            return str(tail.get("seal")), tail_seq
        raise OSError(
            "external head witness does not match the audit log's terminal record "
            "(witness truncated, rewritten, or out of order)")

    def _append_head_witness(self, encounter_id: str, record_sha256: str,
                             in_doubt_over: str | None = None,
                             expected_seq: int | None = None,
                             anchored: bool = False) -> str:
        """Seal and durably append the new terminal head, returning its seal.

        Written BEFORE the row is committed and fsynced, so a witness failure aborts the
        whole append instead of leaving a committed row nothing vouches for. When it is
        written over an in-doubt tail, that fact is SEALED INTO the entry, so the anomaly is
        permanently recorded in the journal instead of only inferable from a divergence.

        `expected_seq` is the sequence the caller's extendability check authorised. It is
        asserted rather than assumed: if the journal moved between that check and this append
        the two disagree, and the append aborts instead of anchoring a sequence that was never
        verified.

        `anchored` records, INSIDE THE SEAL, that an external checkpoint anchor was
        configured when this entry was written. It is the store's own durable, unforgeable
        answer to "was this journal ever anchored?" -- the question the legacy-adoption path
        must answer to tell a journal that PREDATES the anchor (adoptable) from one whose
        anchored checkpoint was DESTROYED (never adoptable). It is written as a sealed field
        rather than kept in a sidecar because a sidecar is exactly as deletable as the
        checkpoint it is standing in for. Absent on entries predating this field, which is
        the correct reading for them: they were written with no anchor."""
        tail, tail_seq = self._witness_tail()
        seq = tail_seq + 1
        if expected_seq is not None and int(expected_seq) != seq:
            raise OSError(
                f"witness journal changed between the extendability check and the append "
                f"(expected to write entry {int(expected_seq)}, would write {seq})")
        entry = {
            "version": self._WITNESS_VERSION,
            "seq": seq,
            "encounter_id": str(encounter_id),
            "record_sha256": str(record_sha256),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "previous_seal": str(tail.get("seal")) if tail else "",
        }
        if in_doubt_over:
            entry["in_doubt_over"] = str(in_doubt_over)
        if anchored:
            entry["anchored"] = True
        entry["seal"] = self._seal(entry, create_key=True)
        wp = self._witness_path()
        wp.parent.mkdir(parents=True, exist_ok=True)
        with open(wp, "a") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return entry["seal"]

    def read_witness(self) -> tuple[list[dict], list[str]]:
        """(entries, problems) for the whole journal. A corrupt, unsealed, mis-chained or
        duplicated entry is REPORTED, never skipped."""
        problems: list[str] = []
        wp = self._witness_path()
        if not wp.exists():
            return [], ["external head witness file is missing"]
        try:
            raw = wp.read_text()
        except Exception as exc:
            return [], [f"external head witness is unreadable: {exc}"]
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return [], ["external head witness is empty"]
        entries: list[dict] = []
        previous_seal = ""
        previous_seq: int | None = None
        seen_seals: set[str] = set()
        for i, line in enumerate(lines):
            try:
                w = json.loads(line)
                if not isinstance(w, dict):
                    raise ValueError("not an object")
            except Exception as exc:
                problems.append(f"witness entry {i}: corrupt/unparseable ({exc})")
                previous_seal = None          # the chain can no longer be followed
                continue
            try:
                expected = self._seal(_witness_payload(w))
            except Exception as exc:
                problems.append(f"witness entry {i}: seal cannot be verified ({exc})")
                entries.append(w)
                previous_seal = w.get("seal")
                continue
            if expected != w.get("seal"):
                problems.append(f"witness entry {i}: seal mismatch (forged or edited)")
            if previous_seal is not None and w.get("previous_seal", "") != previous_seal:
                problems.append(f"witness entry {i}: broken witness chain link")
            if w.get("seal") in seen_seals:
                problems.append(f"witness entry {i}: duplicate witness seal")
            # The sealed sequence must be dense and increasing. Entries predating sequencing
            # carry none; once one appears, every later entry must continue it, so a
            # sequenced journal cannot silently lose an interior entry or regress.
            seq = w.get("seq")
            if seq is None:
                if previous_seq is not None:
                    problems.append(
                        f"witness entry {i}: sequence disappeared after entry {previous_seq}")
            elif isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
                problems.append(f"witness entry {i}: invalid sequence {seq!r}")
            else:
                if previous_seq is not None and seq != previous_seq + 1:
                    problems.append(
                        f"witness entry {i}: sequence {seq} does not follow {previous_seq}")
                previous_seq = seq
            seen_seals.add(w.get("seal"))
            previous_seal = w.get("seal")
            entries.append(w)
        return entries, problems

    def witness_status(self) -> dict:
        """Observability for the witness control: version, custody, entry count, problems, and
        the state of the EXTERNAL terminal-head anchor -- including, when no anchor is
        configured, the fact that consistent truncation is undetectable. An artifact that
        records this can never imply a guarantee that is not actually in force."""
        entries, problems = self.read_witness()
        return {"version": self._WITNESS_VERSION, "key_custody": self.key_custody(),
                "entries": len(entries), "problems": problems,
                "path": str(self._witness_path()),
                "checkpoint": self.checkpoint_status()}

    def verify_chain(self, encounter_id: str | None = None) -> list[str]:
        """Return a list of integrity problems (empty => intact). Detects hash tampering (a
        stored record_sha256 that does not equal the recomputed hash of its fields), broken
        or forked links (previous_record_sha256 != the prior record\'s hash within an
        encounter), and reordering/gaps (seq not strictly increasing). (Codex F6-R4.)"""
        conn = self._connect()
        try:
            q = ("SELECT seq, encounter_id, kind, recorded_at, control_mode,"
                 " previous_record_sha256, record_json, record_sha256, witness_sha256"
                 " FROM audit_log")
            args: tuple = ()
            if encounter_id is not None:
                q += " WHERE encounter_id=?"
                args = (str(encounter_id),)
            q += " ORDER BY seq"
            rows = conn.execute(q, args).fetchall()
            any_rows = bool(conn.execute(
                "SELECT 1 FROM audit_log LIMIT 1").fetchone())
        finally:
            conn.close()
        problems: list[str] = []
        last_sha: dict[str, str] = {}
        present: dict[str, list] = {}
        row_seals: dict[str, tuple] = {}
        last_seq = -1
        for (seq, enc, kind, at, mode, prev, rj, sha, wsha) in rows:
            if seq <= last_seq:
                problems.append(f"seq {seq}: not strictly increasing")
            last_seq = seq
            entry = {"encounter_id": enc, "kind": kind, "recorded_at": at,
                     "control_mode": mode, "previous_record_sha256": prev,
                     "record": json.loads(rj)}
            if _sha(json.dumps(entry, sort_keys=True, default=str)) != sha:
                problems.append(f"seq {seq} enc {enc}: record hash mismatch (tampered)")
            expected_prev = last_sha.get(enc, "")
            if prev != expected_prev:
                problems.append(f"seq {seq} enc {enc}: broken/forked chain link")
            last_sha[enc] = sha
            present.setdefault(enc, []).append((seq, sha))
            row_seals[sha] = (seq, enc, wsha)
        # ------------------------------------------------- external-witness cross-check
        # Bidirectional and FAIL-CLOSED (Codex F6-R4-A). A missing, empty, corrupt, unsealed,
        # duplicated or partial witness is an integrity failure whenever durable rows exist —
        # the previous implementation returned {} for an absent witness, so deleting the
        # sidecar reopened the very tail-truncation blind spot it was added to close.
        # ------------------------------------------ external terminal-head checkpoint anchor
        # Checked BEFORE the "nothing durable yet" shortcut: an emptied store with a live
        # anchored checkpoint is precisely the erasure this control exists to catch, and it
        # would otherwise return "intact". `checkpoint_status()` is total, so a broken anchor
        # is reported here rather than escaping a verification call.
        checkpoint = self.checkpoint_status()
        problems.extend(f"terminal-head checkpoint: {p}"
                        for p in checkpoint.get("problems") or [])
        entries, witness_problems = self.read_witness()
        if not any_rows and not entries:
            return problems                     # nothing durable yet: nothing to witness
        problems.extend(witness_problems)
        if any_rows and not entries:
            problems.append("durable audit rows exist but the external head witness does not")
            return problems
        scoped = [w for w in entries
                  if encounter_id is None or str(w.get("encounter_id")) == str(encounter_id)]
        witnessed = {str(w.get("record_sha256")): w for w in scoped}
        if len(witnessed) != len(scoped):
            problems.append("external head witness contains duplicate record entries")
        # every durable row must be sealed, and its seal must still be IN the witness
        for sha, (seq, enc, wsha) in row_seals.items():
            if not wsha:
                problems.append(f"seq {seq} enc {enc}: audit row carries no witness seal")
                continue
            w = witnessed.get(sha)
            if w is None:
                problems.append(
                    f"seq {seq} enc {enc}: witness entry for this record is absent "
                    f"(witness truncated or rewritten)")
            elif w.get("seal") != wsha:
                problems.append(
                    f"seq {seq} enc {enc}: witness seal does not match the audit row")
        # every witnessed record must still exist in the log; the single trailing in-doubt
        # entry (witness fsynced, commit lost) is named as such rather than hidden
        tail_sha = str(scoped[-1].get("record_sha256")) if scoped else None
        for w in scoped:
            if w.get("in_doubt_over"):
                problems.append(
                    f"enc {w.get('encounter_id')}: this entry was written over an IN-DOUBT "
                    f"witness tail ({str(w['in_doubt_over'])[:16]}…) -- a commit was lost or "
                    f"a terminal record was removed at that point")
        for w in scoped:
            sha = str(w.get("record_sha256"))
            if sha in row_seals:
                continue
            enc = w.get("encounter_id")
            if sha == tail_sha:
                problems.append(
                    f"enc {enc}: witnessed terminal record is absent from the audit log "
                    f"(in-doubt write, or terminal-record truncation)")
            else:
                problems.append(
                    f"enc {enc}: witnessed audit record is missing from the log "
                    f"(tail/interior truncation)")
        return problems


class JsonlAuditRepository(AuditRepository):
    """Per-encounter hash-chained JSONL artifact.

    Enforced callers require a successful durable append before release. A failure
    raises and the pipeline routes SYSTEM_HOLD; there is no empty-success fallback.
    """

    def __init__(self, root, *, strict: bool = True) -> None:
        self.root = Path(root)
        self.strict = strict

    def append(self, encounter_id: str, kind: str, record: dict) -> str:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(encounter_id))
            path = self.root / f"{safe}.jsonl"
            previous = ""
            if path.exists() and path.stat().st_size:
                with open(path, encoding="utf-8") as prior:
                    tail = [line for line in prior.read().splitlines() if line.strip()]
                if not tail:
                    raise OSError("audit chain has no readable final record")
                last = json.loads(tail[-1])
                # One-time deterministic bridge from pre-chain shadow records.
                previous = str(last.get("record_sha256") or
                               _sha(json.dumps(last, sort_keys=True, default=str)))
            entry = {
                "encounter_id": encounter_id,
                "kind": kind,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "control_mode": "ENFORCED_FAIL_CLOSED",
                "previous_record_sha256": previous,
                "record": record,
            }
            entry["record_sha256"] = _sha(json.dumps(entry, sort_keys=True, default=str))
            line = json.dumps(entry, sort_keys=True, default=str)
            with open(path, "a") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            return entry["record_sha256"]
        except Exception:
            if self.strict:
                raise
            return ""
