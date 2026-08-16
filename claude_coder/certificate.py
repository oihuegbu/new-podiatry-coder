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
from typing import Any

# ONE definition of "the same bytes", shared with the claim contract. The
# certificate's content address is re-derived downstream -- by
# `ClaimBundle.certificate.problems()`, which is how a bundle edited on disk is
# detected -- so producer and verifier must canonicalize identically. Two
# identical implementations in two files is the drift class this codebase keeps
# finding; importing the shared one removes it by construction.
from app.contracts.claim_bundle import canonical_json as _canonical
from app.contracts.claim_bundle import content_digest, evidence_records

from .models import CodingResult


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        # WHICH clinical-graph event this line bills. The join key that lets the
        # claim contract compare its lines with these EXACTLY rather than by a
        # sorted (system, code) multiset -- the comparison that let nine units,
        # an extra modifier and a different authority record through. (F7-R1.)
        "clinical_event_id": ln.fact.fact_id,
        "kind": ln.fact.kind.value,
        # IN THE ORDER THE PRODUCER EMITTED THEM. Sorting was a lossy summary of
        # a claim-affecting fact: on a professional line the first modifier is
        # not interchangeable with the second, and a certificate that sorts them
        # cannot attest which order was billed. (F7-R1.)
        "modifiers": list(ln.modifiers),
        "units": ln.units,
        "descriptor": ln.chosen.descriptor,
        "method": ln.method.value,
        "attributes": ln.fact.attributes,
        # Projected through the CLAIM CONTRACT's one evidence-record function --
        # WHERE in the original document each quotation is and which independent
        # reading proved it (F6-R6-A) -- so the certificate's evidence and the
        # bundle's evidence are the same bytes and can be compared exactly
        # instead of approximately. (F7-R1.)
        "evidence": evidence_records(ln.fact.evidence),
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
        # GROUNDING: what the RECORD establishes, and WHICH source text established it.
        "reconciliation_status": r.reconciliation_status,
        "reconciliation_evidence": list(getattr(r, "reconciliation_evidence", []) or []),
        # AGREEMENT: from how many DISTINCT assertion origins the edge arrived — so a reader
        # can tell repetition from re-assertion, and both of those from source grounding.
        # Recorded, never accepted as justification. (Codex F6-R3, round 5.)
        "corroboration_status": str(getattr(r, "corroboration_status", "") or ""),
        "assertion_origins": sorted(str(o) for o in (r.assertion_origins or [])),
        "independent_support": r.independent_support,
        "support": r.support,
    } for r in result.relations]

    payload: dict[str, Any] = {
        "encounter_id": result.encounter_id,
        "date_of_service": result.date_of_service,
        # WHERE that date came from and how it was proven. Bound into the content
        # address so a certificate can never be separated from the evidence that
        # the claim's date of service -- which selects the coverage, the billing
        # affiliation, the authorization window and the effective code edition --
        # was established rather than transcribed. (Issue #6 F7-R4.)
        "service_date_binding": (dict(result.service_date_binding)
                                 if getattr(result, "service_date_binding", None)
                                 else None),
        "note_sha256": _sha(note_text),
        "lines": lines,
        "claim_line_intents": intents,
        "relations": relations,
        # WHY each released service was medically necessary: the claim-line diagnosis pointer
        # and the accepted relation's provenance, as the necessity gate resolved it. Bound
        # here so the justification is answerable FROM THE CERTIFICATE, not only internally
        # consistent inside the run. (Codex F6-R3.)
        "necessity_support": list(result.necessity_support or []),
        "audit_record_hashes": list(result.audit_record_hashes),
        "control_mode": result.control_mode,
        "gates": [{"name": g.name, "outcome": g.outcome.value, "detail": g.detail}
                  for g in result.gates],
        "verdict": result.verdict.value,
        # The multi-channel reading of the ORIGINAL document and every quotation's
        # proof against it. Bound into the content address, so a certificate cannot be
        # separated from the evidence that the transcription behind it was checked.
        "source_evidence": (result.source_reconciliation.certificate_record()
                            if result.source_reconciliation is not None else None),
        "source_identity": source_identity or {},
        # THE single clinical representation this claim was decided from, bound by its
        # own content digest, plus the two-reading axis comparison that settled its
        # code-changing axes. Without these the certificate can show the released lines
        # and the edges, but not that they came from ONE coherent graph, nor what a
        # second independent reading disagreed about and how the document settled it.
        # (Directive section 3.)
        "clinical_graph": (result.graph.certificate_record()
                           if getattr(result, "graph", None) is not None else None),
        "graph_consensus": (dict(result.consensus)
                            if getattr(result, "consensus", None) else None),
    }
    payload["certificate_sha256"] = content_digest(payload)
    return payload
