"""Build and verify fail-closed claim-readiness certificates.

This is the deterministic artifact intended for the autonomous release
boundary. Existing CLEAN/REVIEW values remain internal validator outcomes.
The artifact is emitted inertly until an explicitly approved integration
connects it to an external release action.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from app.release.certificate_models import (
    ClaimReadinessCertificate, ControlOutcome, ControlResult,
    ReadinessDisposition,
)
from app.release.mutation_ledger import normalize_claim
from app.release.scope_registry import approved_scope, scope_fingerprint


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=str).encode()


def _fingerprint(value) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def claim_payload(result: dict) -> dict:
    """Canonical billable content, without non-billable model rationale."""
    fields = {
        "icd_codes": ("code", "type", "description"),
        "cpt_codes": ("code", "modifiers", "units", "dx_pointers",
                      "diagnosis_pointers", "linked_diagnoses", "description"),
        "hcpcs_codes": ("code", "modifiers", "units", "dx_pointers",
                        "diagnosis_pointers", "linked_diagnoses", "description"),
    }
    payload = {}
    for array, keep in fields.items():
        payload[array] = [
            {key: row[key] for key in keep
             if row.get(key) not in (None, "", [])}
            for row in result.get(array) or []
            if isinstance(row, dict) and row.get("code")
        ]
    payload["final_disposition"] = result.get("final_disposition") or ""
    return payload


def _context(result: dict) -> dict:
    meta = result.get("patient_metadata") or {}
    payer_kind = ""
    try:
        from app.compliance.payer_registry import parse_insurance_text
        payer_kind = parse_insurance_text(
            str(meta.get("insurance") or "")).kind
    except Exception:
        pass
    return {
        "date_of_service": meta.get("date_of_service") or "",
        "payer_kind": payer_kind,
        "payer_identity": meta.get("insurance") or "",
        "provider_specialty": meta.get("provider_specialty") or "",
        "place_of_service": meta.get("place_of_service") or "",
        "note_category": ((result.get("rag_context") or {})
                          .get("vision_context") or {}).get(
                              "note_category") or "",
        "claim_family": result.get("claim_family") or "professional",
    }


def _control(control_id: str, outcome: ControlOutcome, reason: str = "",
             evidence=()) -> ControlResult:
    return ControlResult(control_id=control_id, outcome=outcome,
                         reason=reason, evidence=tuple(str(v) for v in evidence))


def _note_control(result: dict) -> ControlResult:
    integrity = result.get("note_integrity") or {}
    note = ((result.get("rag_context") or {}).get("note_full_text") or "")
    if not integrity:
        return _control("note_integrity", ControlOutcome.NOT_CHECKED,
                        "complete-document extraction proof is absent")
    expected = "sha256:" + hashlib.sha256(note.encode()).hexdigest()
    if not integrity.get("complete"):
        return _control("note_integrity", ControlOutcome.BLOCKED,
                        "document extraction is incomplete")
    if not note or integrity.get("extracted_text_sha256") != expected:
        return _control("note_integrity", ControlOutcome.BLOCKED,
                        "extracted note is absent or its checksum changed")
    if not integrity.get("source_pdf_sha256"):
        return _control("note_integrity", ControlOutcome.NOT_CHECKED,
                        "source PDF checksum is absent")
    if integrity.get("page_count") != integrity.get("extracted_page_count"):
        return _control("note_integrity", ControlOutcome.BLOCKED,
                        "not every source page was extracted")
    return _control("note_integrity", ControlOutcome.PASS)


def _source_control(result: dict) -> ControlResult:
    manifest = result.get("authoritative_source_manifest") or {}
    records = manifest.get("records") or []
    if manifest.get("errors"):
        return _control("authoritative_sources", ControlOutcome.ERROR,
                        "; ".join(manifest["errors"]))
    if not records or not manifest.get("fingerprint"):
        return _control("authoritative_sources", ControlOutcome.NOT_CHECKED,
                        "authoritative source manifest is absent")
    from app.release.source_manifest import manifest_fingerprint
    if manifest.get("fingerprint") != manifest_fingerprint(manifest):
        return _control("authoritative_sources", ControlOutcome.ERROR,
                        "authoritative source manifest fingerprint is invalid")
    if any(not r.get("source_id") or not str(r.get("sha256", "")).startswith(
            "sha256:") for r in records):
        return _control("authoritative_sources", ControlOutcome.ERROR,
                        "source manifest has an unchecksummed record")
    present = {str(r.get("source_id")) for r in records}
    mandatory = {"icd10_codes", "cpt_codes", "hcpcs_codes", "ncci_edits",
                 "mue_limits", "coverage_policy", "validator_rules"}
    missing = sorted(mandatory - present)
    if missing:
        return _control("authoritative_sources", ControlOutcome.NOT_CHECKED,
                        "missing authoritative sources: " + ", ".join(missing))
    return _control("authoritative_sources", ControlOutcome.PASS)


def _line_controls(result: dict) -> tuple[ControlResult, ControlResult]:
    note = ((result.get("rag_context") or {}).get("note_full_text") or "")
    meta = result.get("patient_metadata") or {}
    try:
        from app.compliance.engine import _parse_dos
        dos = _parse_dos(meta)
    except Exception:
        dos = None
    dos_text = dos.isoformat() if hasattr(dos, "isoformat") else str(dos or "")
    evidence_errors, temporal_errors = [], []
    diagnoses = {str(row.get("code") or "").upper()
                 for row in result.get("icd_codes") or []}
    for array in ("icd_codes", "cpt_codes", "hcpcs_codes"):
        for row in result.get(array) or []:
            code = str(row.get("code") or "").upper()
            spans = row.get("evidence_spans") or []
            if not spans or any(str(span) not in note for span in spans):
                evidence_errors.append(f"{array}:{code} lacks verbatim evidence")
            if not row.get("source_record_ids"):
                evidence_errors.append(f"{array}:{code} lacks source records")
            start, end = row.get("source_effective_from"), row.get(
                "source_effective_to")
            if not dos_text or not start or not end or not (
                    str(start) <= dos_text <= str(end)):
                temporal_errors.append(
                    f"{array}:{code} not proven active for date of service")
            if array != "icd_codes":
                try:
                    units_valid = int(row.get("units") or 0) >= 1
                except (TypeError, ValueError):
                    units_valid = False
                if not units_valid:
                    evidence_errors.append(f"{array}:{code} has invalid units")
                linked = row.get("linked_diagnoses") or []
                pointers = (row.get("dx_pointers") or
                            row.get("diagnosis_pointers") or [])
                if not linked and not pointers:
                    evidence_errors.append(f"{array}:{code} lacks diagnosis linkage")
                elif linked and any(str(v).upper() not in diagnoses
                                    for v in linked):
                    evidence_errors.append(f"{array}:{code} links an absent diagnosis")
                if pointers:
                    try:
                        pointer_values = [int(v) for v in pointers]
                    except (TypeError, ValueError):
                        pointer_values = []
                    if not pointer_values or any(
                            v < 1 or v > len(diagnoses) for v in pointer_values):
                        evidence_errors.append(
                            f"{array}:{code} has invalid diagnosis pointers")
                claims = {str(v.get("modifier") or "").upper(): v
                          for v in row.get("modifier_reasoning") or []
                          if isinstance(v, dict)}
                for modifier in row.get("modifiers") or []:
                    if claims.get(str(modifier).upper(), {}).get("status") != "applied":
                        evidence_errors.append(
                            f"{array}:{code} modifier rationale is incomplete")
    evidence = (_control("line_evidence_and_linkage", ControlOutcome.BLOCKED,
                         "; ".join(evidence_errors)) if evidence_errors else
                _control("line_evidence_and_linkage", ControlOutcome.PASS))
    temporal = (_control("code_temporal_validity", ControlOutcome.BLOCKED,
                         "; ".join(temporal_errors)) if temporal_errors else
                _control("code_temporal_validity", ControlOutcome.PASS))
    return evidence, temporal


def _mutation_control(result: dict) -> ControlResult:
    candidate = result.get("candidate_claim") or {}
    if not candidate:
        return _control("mutation_resolution", ControlOutcome.NOT_CHECKED,
                        "candidate claim snapshot is absent")
    from app.release.mutation_ledger import claim_diff
    diffs = claim_diff(candidate, normalize_claim(result))
    ledger = result.get("mutation_ledger") or []
    if not diffs:
        return _control("mutation_resolution", ControlOutcome.PASS)
    if len(ledger) != len(diffs) or any(
            row.get("state") != "applied" for row in ledger):
        return _control("mutation_resolution", ControlOutcome.BLOCKED,
                        "one or more claim mutations are unresolved")
    return _control("mutation_resolution", ControlOutcome.PASS)


def _legacy_controls(result: dict) -> list[ControlResult]:
    controls = []
    controls.append(_control(
        "pipeline_execution",
        ControlOutcome.PASS if result.get("success") else ControlOutcome.ERROR,
        "" if result.get("success") else "pipeline did not succeed"))
    cons = result.get("consistency") or {}
    repeatable = (cons.get("runs") or 0) >= 2 and bool(cons.get("unanimous"))
    controls.append(_control(
        "repeatability", ControlOutcome.PASS if repeatable else
        ControlOutcome.REVIEW_REQUIRED,
        "" if repeatable else "claim is not unanimous across independent runs"))
    clean = str(result.get("final_disposition") or "").upper() == "CLEAN"
    controls.append(_control(
        "compliance_scrub", ControlOutcome.PASS if clean else
        ControlOutcome.BLOCKED,
        "" if clean else "compliance scrub did not return CLEAN"))
    scrub = result.get("claim_scrub") or {}
    filter_results = scrub.get("filter_results")
    expected_count = int(scrub.get("expected_filter_count") or 0)
    if not filter_results or not expected_count or \
            len(filter_results) != expected_count:
        controls.append(_control(
            "mandatory_filter_execution", ControlOutcome.NOT_CHECKED,
            "per-filter execution trail is absent or incomplete"))
    else:
        bad = [str(row.get("filter_id") or "unknown") for row in filter_results
               if not isinstance(row, dict) or str(row.get("status") or "").upper()
               in {"ERROR", "UNKNOWN", "NOT_CHECKED", ""}]
        controls.append(_control(
            "mandatory_filter_execution",
            ControlOutcome.ERROR if bad else ControlOutcome.PASS,
            "unverified filters: " + ", ".join(bad) if bad else ""))
    has_dx = bool(result.get("icd_codes"))
    has_service = bool(result.get("cpt_codes") or result.get("hcpcs_codes"))
    controls.append(_control(
        "claim_shape", ControlOutcome.PASS if has_dx and has_service else
        ControlOutcome.BLOCKED,
        "" if has_dx and has_service else
        "claim requires at least one diagnosis and one service line"))
    audit = result.get("clinical_audit") or {}
    audit_ok = audit.get("verdict") == "upheld"
    if audit_ok:
        try:
            from tools.clinical_auditor import corrections_fingerprint
            audit_ok = audit.get("fingerprint") == corrections_fingerprint(result)
        except Exception:
            audit_ok = False
    controls.append(_control(
        "independent_clinical_audit", ControlOutcome.PASS if audit_ok else
        ControlOutcome.REVIEW_REQUIRED,
        "" if audit_ok else "clinical audit is absent, disputed, or stale"))
    try:
        from tools.record_coherence import coherence_violations
        violations = coherence_violations(result)
        controls.append(_control(
            "record_coherence", ControlOutcome.PASS if not violations else
            ControlOutcome.BLOCKED, "; ".join(violations)))
    except Exception as exc:
        controls.append(_control("record_coherence", ControlOutcome.ERROR,
                                 f"coherence could not be verified ({exc})"))
    return controls


def _certificate_fingerprint(payload: dict) -> str:
    return _fingerprint({k: v for k, v in payload.items()
                         if k != "certificate_fingerprint"})


def build_readiness_certificate(result: dict) -> ClaimReadinessCertificate:
    context = _context(result)
    controls = _legacy_controls(result)
    controls += [_note_control(result), _source_control(result)]
    controls.extend(_line_controls(result))
    controls.append(_mutation_control(result))
    scope, scope_reason = approved_scope(context)
    controls.append(_control(
        "autonomous_scope", ControlOutcome.PASS if scope else
        ControlOutcome.REVIEW_REQUIRED, scope_reason))
    outcomes = {c.outcome for c in controls}
    if outcomes & {ControlOutcome.BLOCKED, ControlOutcome.ERROR,
                   ControlOutcome.NOT_CHECKED}:
        disposition = ReadinessDisposition.BLOCKED
    elif ControlOutcome.REVIEW_REQUIRED in outcomes:
        disposition = ReadinessDisposition.REVIEW_REQUIRED
    else:
        disposition = ReadinessDisposition.AUTO_READY

    manifest = result.get("authoritative_source_manifest") or {}
    integrity = result.get("note_integrity") or {}
    claim = claim_payload(result)
    rules = next((r for r in manifest.get("records") or []
                  if r.get("source_id") == "validator_rules"), {})
    payload = {
        "certificate_version": 1,
        "document_id": str(result.get("document_id") or ""),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disposition": disposition.value,
        "certificate_fingerprint": "",
        "note_fingerprint": str(integrity.get("extracted_text_sha256") or ""),
        "source_document_fingerprint": str(
            integrity.get("source_pdf_sha256") or ""),
        "context_fingerprint": _fingerprint(context),
        "claim_fingerprint": _fingerprint(claim),
        "source_manifest_fingerprint": str(manifest.get("fingerprint") or ""),
        "rule_pack_fingerprint": str(rules.get("sha256") or ""),
        "autonomous_scope_id": str((scope or {}).get("id") or ""),
        "autonomous_scope_fingerprint": scope_fingerprint(scope) if scope else "",
        "system_versions": {"release_gate": "1"},
        "claim_payload": claim,
        "controls": [c.model_dump(mode="json") for c in controls],
        "assumptions": [],
    }
    payload["certificate_fingerprint"] = _certificate_fingerprint(payload)
    return ClaimReadinessCertificate.model_validate(payload)


def verify_readiness_certificate(result: dict, certificate: dict | None = None
                                 ) -> tuple[bool, str]:
    try:
        cert = ClaimReadinessCertificate.model_validate(
            certificate or result.get("claim_readiness_certificate") or {})
    except Exception as exc:
        return False, f"claim-readiness certificate is absent or invalid ({exc})"
    data = cert.model_dump(mode="json")
    if cert.disposition != ReadinessDisposition.AUTO_READY:
        return False, f"readiness disposition is {cert.disposition.value}"
    if cert.claim_payload != claim_payload(result) or \
            cert.claim_fingerprint != _fingerprint(claim_payload(result)):
        return False, "claim changed after readiness certification"
    integrity = result.get("note_integrity") or {}
    if cert.note_fingerprint != integrity.get("extracted_text_sha256"):
        return False, "note changed after readiness certification"
    if cert.source_document_fingerprint != integrity.get("source_pdf_sha256"):
        return False, "source document changed after readiness certification"
    manifest = result.get("authoritative_source_manifest") or {}
    from app.release.source_manifest import manifest_fingerprint
    if manifest.get("fingerprint") != manifest_fingerprint(manifest):
        return False, "authoritative source manifest fingerprint is invalid"
    if cert.source_manifest_fingerprint != manifest.get("fingerprint"):
        return False, "authoritative sources changed after certification"
    if cert.context_fingerprint != _fingerprint(_context(result)):
        return False, "claim context changed after certification"
    if cert.certificate_fingerprint != _certificate_fingerprint(data):
        return False, "certificate fingerprint is invalid"
    current_scope, why = approved_scope(_context(result))
    if not current_scope or cert.autonomous_scope_fingerprint != \
            scope_fingerprint(current_scope):
        return False, why or "autonomous scope changed after certification"
    return True, ""
