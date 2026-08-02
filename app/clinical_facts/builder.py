"""Build a stable clinical-fact envelope before code selection.

The envelope does not diagnose or select codes. It reconciles exact NER spans
and extraction-declared encounter events into one fingerprinted representation
so retrieval, coding, consistency, and release reason about the same facts.
"""

from __future__ import annotations

import hashlib
import json
import re

from app.models.schemas import ClinicalEntity


def _fingerprint(value) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str).encode()).hexdigest()


def _lines(text: str) -> list[str]:
    return [value.strip() for value in re.split(r"[\r\n]+|(?<=[.!?])\s+", text)
            if value.strip()]


def _evidence(anchor: str, note: str) -> str:
    if not anchor or not note:
        return ""
    direct = note.casefold().find(anchor.casefold())
    if direct >= 0:
        return note[direct:direct + len(anchor)]
    # Conservative normalized containment handles punctuation/spacing only;
    # it never manufactures a paraphrase or performs semantic similarity.
    wanted = re.sub(r"[^a-z0-9]+", " ", anchor.casefold()).strip()
    matches = [line for line in _lines(note)
               if wanted and wanted in re.sub(
                   r"[^a-z0-9]+", " ", line.casefold()).strip()]
    return matches[0] if len(matches) == 1 else ""


def build_clinical_fact_report(*, entities: list[ClinicalEntity], sections: dict,
                               procedures: list, imaging: list,
                               supplies: list, prior_surgery: dict) -> dict:
    note = str(sections.get("full_text") or "")
    facts, unresolved = [], []
    for entity in entities:
        span = entity.source_span or {}
        start, end = span.get("document_start"), span.get("document_end")
        span_verified = bool(
            span.get("verified") and isinstance(start, int)
            and isinstance(end, int) and 0 <= start < end <= len(note)
            and note[start:end] == entity.text)
        span = {**span, "verified": span_verified}
        fact = {
            "kind": "entity",
            "category": entity.category,
            "raw_text": entity.text,
            "normalized_text": entity.normalized_text or entity.clinical_term,
            "section": entity.source_section,
            "laterality": entity.laterality,
            "negated": bool(entity.negated),
            "temporality": ("historical" if str(entity.source_section).upper()
                            in {"PMH", "HISTORY"} else "current"),
            "source_span": span,
            "normalization_status": entity.normalization_status,
        }
        facts.append(fact)
        if not span_verified:
            unresolved.append({"kind": "entity", "label": entity.text,
                               "reason": "exact source span is unverified"})

    for kind, values in (("performed_procedure", procedures),
                         ("performed_imaging", imaging),
                         ("dispensed_supply", supplies)):
        for value in values or []:
            label = str(value.get("description") if isinstance(value, dict)
                        else value).strip()
            evidence = _evidence(label, note)
            fact = {"kind": kind, "label": label,
                    "status": "completed", "evidence_span": evidence,
                    "evidence_verified": bool(evidence)}
            facts.append(fact)
            if not evidence:
                unresolved.append({"kind": kind, "label": label,
                                   "reason": "extracted event lacks verbatim note evidence"})

    if prior_surgery and prior_surgery.get("is_post_op_visit"):
        label = str(prior_surgery.get("prior_surgery_description") or "").strip()
        evidence = _evidence(label, note)
        facts.append({"kind": "prior_surgery", "label": label,
                      "status": "historical", "evidence_span": evidence,
                      "evidence_verified": bool(evidence),
                      "days_post_op": prior_surgery.get("days_post_op")})
        if not evidence:
            unresolved.append({
                "kind": "prior_surgery", "label": label,
                "reason": "post-operative context lacks verbatim note evidence"})
    body = {"schema_version": 1, "facts": facts,
            "unresolved_material_facts": unresolved,
            "note_sha256": "sha256:" + hashlib.sha256(note.encode()).hexdigest()}
    body["facts_fingerprint"] = _fingerprint(facts)
    body["status"] = "REVIEW_REQUIRED" if unresolved else "PASS"
    body["report_fingerprint"] = _fingerprint(body)
    return body
