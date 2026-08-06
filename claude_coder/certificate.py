"""The release certificate — the defensibility artifact.

Binds the exact released claim to everything it depends on: the source note
(by hash), the date of service, each billed line with its verbatim evidence and
the authoritative record that defines it, the outcome of every gate, and the
verdict — then content-addresses the whole packet with a SHA-256. Re-running the
same inputs reproduces the same certificate hash; changing the note, a code, an
evidence span, a gate outcome, or the source edition invalidates it.

This is what makes an autonomous claim answerable after the fact: for any billed
line you can show the note text that supported it, the descriptor and edition it
came from, how it was chosen, and that it passed every control — and prove the
submitted claim is byte-for-byte the one that passed. (An HMAC signature with a
private key would add non-repudiation on top of this integrity hash.)
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import CodingResult


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def build_certificate(result: CodingResult, note_text: str,
                      source_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    # Bind EVERY field that changes the bill or its justification: not just the code
    # and evidence but the UNITS billed, the documented axes the code rests on, the
    # line kind, and each gate's detail — so altering a unit count, an attribute, or
    # a gate result invalidates the hash. (Omitting units let a 1 -> 9 unit change
    # produce an identical certificate.)
    lines = [{
        "system": ln.chosen.system,
        "code": ln.chosen.code,
        "kind": ln.fact.kind.value,
        "modifiers": sorted(ln.modifiers),
        "units": ln.units,
        "descriptor": ln.chosen.descriptor,
        "method": ln.method.value,
        "attributes": ln.fact.attributes,
        "evidence": [{
            "text": s.text,
            "section": s.section,
            "page": s.page,
            "start": s.start,
            "end": s.end,
            "text_sha256": s.text_sha256,
            "document_sha256": s.document_sha256,
            "document_version": s.document_version,
            "span_id": s.span_id,
            "anchored": s.anchored,
        } for s in ln.fact.evidence],
        "authority": ln.chosen.authority,
    } for ln in result.billable_lines]

    intents = [{
        "intent_id": it.intent_id,
        "component": it.component.value,
        "clinical_event_ids": list(it.clinical_event_ids),
        "fact_kind": it.fact_kind,
        "clinical_action": it.clinical_action,
        "attributes": it.attributes,
        "date_of_service": it.date_of_service,
        "billing_entity_id": it.billing_entity_id,
        "source_span_ids": list(it.source_span_ids),
        "state": it.state.value,
        "service_episode_id": it.service_episode_id,
        "mention_count": it.mention_count,
        "distinctness_facts": list(it.distinctness_facts),
        "decisions": [{"gate": d.gate, "outcome": d.outcome.value,
                       "detail": d.detail, "authority": d.authority}
                      for d in it.decisions],
    } for it in result.claim_line_intents]
    relations = [{
        "relation_id": r.relation_id,
        "subject_event_id": r.subject_event_id,
        "predicate": r.predicate.value,
        "object_event_id": r.object_event_id,
        "state": r.state.value,
        "evidence_span_ids": list(r.evidence_span_ids),
        "extraction_source": r.extraction_source,
        "confidence": r.confidence,
        "reconciliation_status": r.reconciliation_status,
        "support": r.support,
    } for r in result.relations]

    payload: dict[str, Any] = {
        "encounter_id": result.encounter_id,
        "date_of_service": result.date_of_service,
        "note_sha256": _sha(note_text),
        "lines": lines,
        "claim_line_intents": intents,
        "relations": relations,
        "audit_record_hashes": list(result.audit_record_hashes),
        "control_mode": result.control_mode,
        "gates": [{"name": g.name, "outcome": g.outcome.value, "detail": g.detail}
                  for g in result.gates],
        "verdict": result.verdict.value,
        "source_identity": source_identity or {},
    }
    payload["certificate_sha256"] = _sha(_canonical(payload))
    return payload
