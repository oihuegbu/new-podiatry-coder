"""Deterministic, signed authorization for one exact autonomous claim.

The certificate is the release boundary, not an informational annotation.
It binds the billable claim, encounter context, full note/source identity,
every control input, the source snapshot, and the authenticated operating
scope. Registry ingest and claim submission both verify this same artifact.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from datetime import date, datetime, timezone

from app.release.certificate_models import (
    ClaimReadinessCertificate, ControlOutcome, ControlResult,
    ReadinessDisposition,
)
from app.release.mutation_ledger import normalize_claim
from app.release.scope_registry import approved_scope, scope_fingerprint


_EVIDENCE_MIN_LEN = 14
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CERTIFICATE_KEY_ENV = "CLAIM_READINESS_SIGNING_KEY"
_MIN_SIGNING_KEY_BYTES = 32


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=str).encode()


def _fingerprint(value) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def claim_payload(result: dict) -> dict:
    """Canonical content that can affect a professional claim transaction."""
    fields = {
        "icd_codes": ("code", "type", "description", "laterality"),
        "cpt_codes": ("code", "modifiers", "units", "dx_pointers",
                      "diagnosis_pointers", "linked_diagnoses", "description",
                      "laterality", "place_of_service", "ndc"),
        "hcpcs_codes": ("code", "modifiers", "units", "dx_pointers",
                        "diagnosis_pointers", "linked_diagnoses", "description",
                        "laterality", "place_of_service", "ndc"),
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


def _derived_specialty(meta: dict) -> str:
    explicit = str(meta.get("provider_specialty") or "").strip().lower()
    if explicit:
        return explicit
    provider = " ".join(str(meta.get(k) or "") for k in
                        ("provider", "signature_block")).upper()
    return "podiatry" if re.search(r"\bD\.?P\.?M\.?\b", provider) else ""


def _context(result: dict) -> dict:
    meta = result.get("patient_metadata") or {}
    parsed = None
    try:
        from app.compliance.payer_registry import parse_insurance_text
        parsed = parse_insurance_text(str(meta.get("insurance") or ""))
    except Exception:
        pass
    facility = meta.get("service_facility") or {}
    if not isinstance(facility, dict):
        facility = {}
    billing_npi = meta.get("billing_npi") or ""
    if not billing_npi:
        try:
            from tools.claim_submitter import load_practice_config
            billing_npi = ((load_practice_config().get("billing_provider") or {})
                           .get("npi") or "")
        except Exception:
            billing_npi = ""
    practice_config_fingerprint = ""
    try:
        from tools.claim_submitter import load_practice_config
        practice_config_fingerprint = _fingerprint(load_practice_config())
    except Exception:
        pass
    return {
        "date_of_service": meta.get("date_of_service") or "",
        "payer_kind": getattr(parsed, "kind", "") or "",
        "payer_id": getattr(parsed, "payer_id", "") or "",
        "payer_identity": meta.get("insurance") or "",
        "plan": meta.get("plan") or meta.get("insurance_plan") or "",
        "member_id": meta.get("member_id") or meta.get("insurance_id") or
                     getattr(parsed, "member_id", "") or "",
        "provider_specialty": _derived_specialty(meta),
        "provider": meta.get("provider") or "",
        "rendering_npi": meta.get("provider_npi") or meta.get("npi") or "",
        "billing_npi": billing_npi,
        "submission_configuration_fingerprint": practice_config_fingerprint,
        "place_of_service": meta.get("place_of_service") or "",
        "jurisdiction": facility.get("state") or meta.get("state") or "",
        "service_facility": facility,
        "note_category": ((result.get("rag_context") or {})
                          .get("vision_context") or {}).get(
                              "note_category") or "",
        "claim_family": result.get("claim_family") or "professional",
    }


def encounter_context_payload(result: dict) -> dict:
    """Submission-relevant encounter identity, excluding the note text itself."""
    integrity = result.get("note_integrity") or {}
    return {
        "document_id": str(result.get("document_id") or ""),
        "patient_metadata": result.get("patient_metadata") or {},
        "context": _context(result),
        "source_pdf_sha256": integrity.get("source_pdf_sha256") or "",
        "extracted_text_sha256": integrity.get("extracted_text_sha256") or "",
    }


def encounter_context_fingerprint(result: dict) -> str:
    return _fingerprint(encounter_context_payload(result))


def readiness_input_payload(result: dict) -> dict:
    """Every mutable input capable of changing a readiness control outcome."""
    rag = result.get("rag_context") or {}
    return {
        "document_id": result.get("document_id") or "",
        "success": bool(result.get("success")),
        "encounter": encounter_context_payload(result),
        "note_full_text": rag.get("note_full_text") or "",
        "vision_context": rag.get("vision_context") or {},
        "claim": claim_payload(result),
        "candidate_claim": result.get("candidate_claim") or {},
        "mutation_ledger": result.get("mutation_ledger") or [],
        "material_corrections": result.get("material_corrections") or [],
        "ner_entities": result.get("ner_entities") or [],
        "terminology_normalization": (
            result.get("terminology_normalization") or {}),
        "clinical_facts": result.get("clinical_facts") or {},
        "consistency": result.get("consistency") or {},
        "claim_scrub": result.get("claim_scrub") or {},
        "clinical_audit": result.get("clinical_audit") or {},
        "note_integrity": result.get("note_integrity") or {},
        "authoritative_source_manifest": (
            result.get("authoritative_source_manifest") or {}),
        "procedures_performed_today": (
            result.get("procedures_performed_today") or []),
        "validation_issues": result.get("validation_issues") or [],
        "review_routing": result.get("review_routing") or "",
        "adjudication": result.get("adjudication") or {},
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
    if not _SHA256_RE.fullmatch(str(integrity.get("source_pdf_sha256") or "")):
        return _control("note_integrity", ControlOutcome.NOT_CHECKED,
                        "source PDF checksum is absent or malformed")
    pages = integrity.get("page_coverage") or []
    expected_pages = int(integrity.get("page_count") or 0)
    covered = sorted(int(p.get("page_number")) for p in pages
                     if isinstance(p, dict) and
                     str(p.get("page_number") or "").isdigit() and
                     p.get("status") in {"extracted", "blank"} and
                     (p.get("text_sha256") or p.get("status") == "blank"))
    if expected_pages < 1 or covered != list(range(1, expected_pages + 1)):
        return _control("note_integrity", ControlOutcome.BLOCKED,
                        "per-page extraction coverage is absent or incomplete")
    if integrity.get("extracted_page_count") != expected_pages:
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
    from app.release.source_manifest import (
        build_source_manifest, manifest_fingerprint, valid_record,
    )
    if manifest.get("fingerprint") != manifest_fingerprint(manifest):
        return _control("authoritative_sources", ControlOutcome.ERROR,
                        "authoritative source manifest fingerprint is invalid")
    if any(not valid_record(r) for r in records):
        return _control("authoritative_sources", ControlOutcome.ERROR,
                        "source manifest contains an invalid record identity")
    current = build_source_manifest()
    if current.get("fingerprint") != manifest.get("fingerprint"):
        return _control("authoritative_sources", ControlOutcome.BLOCKED,
                        "authoritative sources changed or are not the live snapshot")
    present = {str(r.get("source_id")) for r in records}
    mandatory = {"icd10_codes", "cpt_codes", "hcpcs_codes", "ncci_edits",
                 "mue_limits", "coverage_policy", "validator_rules",
                 "pfs_indicators",
                 "compliance_database", "validator_implementation",
                 "consistency_implementation",
                 "scrubber_implementation", "release_gate_implementation",
                 "compliance_datastore_implementation",
                 "payer_registry_implementation",
                 "mutation_ledger_implementation",
                 "submission_configuration", "terminology_registry",
                 "terminology_implementation", "terminology_source_catalog",
                 "source_requirements",
                 "mcd_coverage_cache", "autonomous_scope_registry",
                 "scope_bootstrap_implementation",
                 "scope_authorization_implementation",
                 "identifier_validation_implementation",
                 "model_execution_implementation",
                 "terminology_builder_implementation",
                 "source_preflight_implementation",
                 "clinical_facts_implementation",
                 "clinical_audit_implementation",
                 "record_coherence_implementation"}
    missing = sorted(mandatory - present)
    if missing:
        return _control("authoritative_sources", ControlOutcome.NOT_CHECKED,
                        "missing authoritative sources: " + ", ".join(missing))
    try:
        from app.compliance.engine import _parse_dos
        dos = _parse_dos(result.get("patient_metadata") or {})
    except Exception:
        dos = None
    by_id = {str(record.get("source_id")): record for record in records}
    contract_errors = _source_contract_errors(result, by_id, dos)
    if contract_errors:
        return _control(
            "authoritative_sources", ControlOutcome.BLOCKED,
            "; ".join(contract_errors))
    return _control("authoritative_sources", ControlOutcome.PASS)


def _source_contract_errors(result: dict, by_id: dict[str, dict],
                            dos: date | None) -> list[str]:
    """Evaluate versioned applicability/freshness requirements as data."""
    from app.core.config import SOURCE_REQUIREMENTS_FILE
    try:
        pack = json.loads(SOURCE_REQUIREMENTS_FILE.read_text())
    except Exception as exc:
        return [f"source-requirement pack unavailable ({exc})"]
    if pack.get("schema_version") != 1 or not isinstance(
            pack.get("requirements"), list):
        return ["source-requirement pack is malformed"]
    context = _context(result)
    try:
        from app.compliance.payer_registry import parse_insurance_text
        follows_medicare = bool(parse_insurance_text(
            str((result.get("patient_metadata") or {}).get(
                "insurance") or "")).follows_medicare_coverage)
    except Exception:
        follows_medicare = False
    service_count = len(result.get("cpt_codes") or []) + len(
        result.get("hcpcs_codes") or [])
    predicates = {
        "always": True,
        "has_cpt": bool(result.get("cpt_codes")),
        "has_hcpcs": bool(result.get("hcpcs_codes")),
        "service": service_count > 0,
        "multiple_services": service_count > 1,
        "medicare_coverage_service": follows_medicare and service_count > 0,
    }
    errors = []
    for requirement in pack["requirements"]:
        if not isinstance(requirement, dict):
            errors.append("source-requirement entry is malformed")
            continue
        applies = str(requirement.get("applies_when") or "")
        if applies not in predicates:
            errors.append(
                f"source requirement {requirement.get('id') or '?'} has an "
                "unsupported applicability predicate")
            continue
        if not predicates[applies]:
            continue
        source_id = str(requirement.get("source_id") or "")
        record = by_id.get(source_id)
        if not record:
            errors.append(f"required source is absent: {source_id}")
            continue
        if requirement.get("dos_release_window"):
            windows = record.get("release_windows") or [{
                "effective_from": record.get("release_effective_from"),
                "effective_to": record.get("release_effective_to"),
            }]
            covered = bool(dos) and any(
                (start := _parse_iso(window.get("effective_from")))
                and (end := _parse_iso(window.get("effective_to")))
                and start <= dos <= end
                for window in windows if isinstance(window, dict))
            if not covered:
                errors.append(
                    f"claim date is outside the loaded source release window: "
                    f"{source_id}")
        if requirement.get("max_age_days") is not None:
            try:
                limit = int(requirement["max_age_days"])
                fetched = datetime.fromisoformat(
                    str(record.get("fetched_at") or "").replace("Z", "+00:00"))
                if fetched.tzinfo is None:
                    fetched = fetched.replace(tzinfo=timezone.utc)
                age = datetime.now(timezone.utc) - fetched.astimezone(timezone.utc)
                if limit < 0 or age.days > limit:
                    errors.append(
                        f"required source exceeds its freshness contract: {source_id} "
                        f"({age.days} days > {limit})")
            except (TypeError, ValueError):
                errors.append(
                    f"required source has no valid freshness timestamp: {source_id}")
    return errors


def _parse_iso(value) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _line_controls(result: dict) -> tuple[ControlResult, ControlResult]:
    note = ((result.get("rag_context") or {}).get("note_full_text") or "")
    try:
        from app.compliance.engine import _parse_dos
        dos = _parse_dos(result.get("patient_metadata") or {})
    except Exception:
        dos = None
    evidence_errors, temporal_errors = [], []
    diagnoses = [str(row.get("code") or "").upper()
                 for row in result.get("icd_codes") or []]
    diagnosis_set = set(diagnoses)
    manifest_ids = {str(r.get("source_id")) for r in
                    (result.get("authoritative_source_manifest") or {})
                    .get("records") or []}
    expected_source = {"icd_codes": "icd10_codes", "cpt_codes": "cpt_codes",
                       "hcpcs_codes": "hcpcs_codes"}
    for array in ("icd_codes", "cpt_codes", "hcpcs_codes"):
        for row in result.get(array) or []:
            code = str(row.get("code") or "").upper()
            spans = row.get("evidence_spans") or []
            normalized = [str(span).strip() for span in spans]
            if (not normalized or any(len(span) < _EVIDENCE_MIN_LEN or
                                      span not in note for span in normalized)):
                evidence_errors.append(f"{array}:{code} lacks substantive verbatim evidence")
            source_ids = row.get("source_record_ids") or []
            prefix = expected_source[array] + ":"
            expected_id = prefix + code
            if ({str(v).upper() for v in source_ids} != {expected_id.upper()}
                    or expected_source[array] not in manifest_ids):
                evidence_errors.append(f"{array}:{code} lacks a bound source record")
            start = _parse_iso(row.get("source_effective_from"))
            end = _parse_iso(row.get("source_effective_to"))
            if (not row.get("source_temporal_authority") or not dos or
                    not start or not end or not start <= dos <= end):
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
                if linked and any(str(v).upper() not in diagnosis_set for v in linked):
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
    normalized_ledger = [
        {k: row.get(k) for k in
         ("array", "code", "occurrence", "before", "after")}
        for row in ledger if isinstance(row, dict)
    ]
    if (len(normalized_ledger) != len(ledger) or normalized_ledger != diffs or any(
            row.get("state") != "applied" or not row.get("reason") or
            not row.get("rule_id") or not row.get("effective_on") or
            not row.get("evidence_spans") or not row.get("source_record_ids")
            for row in ledger if isinstance(row, dict))):
        return _control("mutation_resolution", ControlOutcome.BLOCKED,
                        "mutation ledger does not exactly account for every claim diff")
    return _control("mutation_resolution", ControlOutcome.PASS)


def _mandatory_filter_control(result: dict) -> ControlResult:
    scrub = result.get("claim_scrub") or {}
    rows = scrub.get("filter_results") or []
    try:
        from app.compliance.agents import build_default_agents
        required = [a.filter_id for a in build_default_agents(None)]
    except Exception as exc:
        return _control("mandatory_filter_execution", ControlOutcome.ERROR,
                        f"required filter registry is unavailable ({exc})")
    observed = [str(row.get("filter_id") or "") for row in rows
                if isinstance(row, dict)]
    if observed != required or scrub.get("expected_filter_count") != len(required):
        return _control("mandatory_filter_execution", ControlOutcome.NOT_CHECKED,
                        "filter execution trail does not exactly match the active registry")
    unverified = [observed[i] for i, row in enumerate(rows)
                  if str(row.get("status") or "").upper() in
                  {"ERROR", "UNKNOWN", "NOT_CHECKED", ""}]
    failed = [observed[i] for i, row in enumerate(rows)
              if str(row.get("status") or "").upper() == "FAIL"]
    if unverified:
        return _control("mandatory_filter_execution", ControlOutcome.ERROR,
                        "unverified filters: " + ", ".join(unverified))
    if failed:
        return _control("mandatory_filter_execution", ControlOutcome.BLOCKED,
                        "blocking filters: " + ", ".join(failed))
    return _control("mandatory_filter_execution", ControlOutcome.PASS)


def _validation_control(result: dict) -> ControlResult:
    """Preserve the deterministic validator's release decision.

    The compliance scrub evaluates a different set of controls and therefore
    cannot clear a validator defect merely because every scrub agent passed.
    Errors/critical findings are hard blockers; warnings retain the validator's
    review requirement.  Informational, already-resolved corrections remain
    eligible for the normal mutation/audit controls below.
    """
    if "validation_issues" not in result:
        return _control("deterministic_validation",
                        ControlOutcome.NOT_CHECKED,
                        "validator execution output is absent")
    raw_issues = result.get("validation_issues")
    if not isinstance(raw_issues, list) or any(
            not isinstance(row, dict) for row in raw_issues):
        return _control("deterministic_validation", ControlOutcome.ERROR,
                        "validator issue payload is malformed")
    issues = list(raw_issues)
    blocking = [row for row in issues if str(row.get("severity") or "").upper()
                in {"ERROR", "CRITICAL"}]
    warnings = [row for row in issues if str(row.get("severity") or "").upper()
                == "WARNING"]
    unknown = [row for row in issues if str(row.get("severity") or "").upper()
               not in {"INFO", "WARNING", "ERROR", "CRITICAL"}]

    def _labels(rows: list[dict]) -> str:
        return ", ".join(sorted({
            str(row.get("category") or row.get("code") or "validator_issue")
            for row in rows
        }))

    if blocking:
        return _control("deterministic_validation", ControlOutcome.BLOCKED,
                        "unresolved validator errors: " + _labels(blocking))
    if unknown:
        return _control("deterministic_validation", ControlOutcome.ERROR,
                        "validator issues have unknown severity: "
                        + _labels(unknown))
    if warnings:
        return _control("deterministic_validation",
                        ControlOutcome.REVIEW_REQUIRED,
                        "unresolved validator warnings: " + _labels(warnings))
    return _control("deterministic_validation", ControlOutcome.PASS)


def _terminology_control(result: dict) -> ControlResult:
    """Require traceable, stable terminology interpretation for autonomy."""
    report = result.get("terminology_normalization")
    if not isinstance(report, dict) or not report:
        return _control(
            "terminology_normalization", ControlOutcome.NOT_CHECKED,
            "terminology normalization evidence is absent")
    required = {
        "schema_version", "registry_version", "registry_sha256",
        "entities", "entity_fingerprint", "note_occurrences",
        "unresolved_billing_relevant", "status", "report_fingerprint",
        "registry_files", "authority_role",
    }
    missing = sorted(required - set(report))
    if missing:
        return _control(
            "terminology_normalization", ControlOutcome.ERROR,
            "terminology report is incomplete: " + ", ".join(missing))
    if report.get("schema_version") != 1:
        return _control(
            "terminology_normalization", ControlOutcome.ERROR,
            "terminology report schema is unsupported")
    if not all(isinstance(report.get(name), list) for name in
               ("entities", "note_occurrences", "unresolved_billing_relevant")):
        return _control(
            "terminology_normalization", ControlOutcome.ERROR,
            "terminology report collections are malformed")
    if not _SHA256_RE.fullmatch(str(report.get("registry_sha256") or "")):
        return _control(
            "terminology_normalization", ControlOutcome.ERROR,
            "terminology registry identity is malformed")
    try:
        from app.terminology import (TerminologyNormalizer,
                                     terminology_entity_fingerprint)
        live_registry = TerminologyNormalizer().registry_sha256
        entity_fingerprint = terminology_entity_fingerprint(
            result.get("ner_entities") or [])
    except Exception as exc:
        return _control(
            "terminology_normalization", ControlOutcome.ERROR,
            f"terminology evidence could not be verified ({exc})")
    if report.get("registry_sha256") != live_registry:
        return _control(
            "terminology_normalization", ControlOutcome.BLOCKED,
            "terminology registry changed after normalization")
    if report.get("authority_role") != "retrieval_only":
        return _control(
            "terminology_normalization", ControlOutcome.ERROR,
            "terminology normalization attempted to act as coding authority")
    if report.get("entity_fingerprint") != entity_fingerprint:
        return _control(
            "terminology_normalization", ControlOutcome.BLOCKED,
            "normalized entity evidence does not match persisted NER entities")
    if report.get("entity_fingerprint") != _fingerprint(report.get("entities")):
        return _control(
            "terminology_normalization", ControlOutcome.ERROR,
            "terminology entity fingerprint is invalid")
    body = {key: value for key, value in report.items()
            if key != "report_fingerprint"}
    if report.get("report_fingerprint") != _fingerprint(body):
        return _control(
            "terminology_normalization", ControlOutcome.ERROR,
            "terminology report fingerprint is invalid")
    unresolved = report.get("unresolved_billing_relevant") or []
    if any(not isinstance(row, dict) or not row.get("raw_text")
           or row.get("status") not in {"ambiguous", "unresolved"}
           for row in unresolved):
        return _control(
            "terminology_normalization", ControlOutcome.ERROR,
            "unresolved terminology evidence is malformed")
    unverified_spans = [
        str(row.get("text") or "") for row in report.get("entities") or []
        if not bool((row.get("source_span") or {}).get("verified"))
    ]
    expected_status = "REVIEW_REQUIRED" if unresolved else "PASS"
    if report.get("status") != expected_status:
        return _control(
            "terminology_normalization", ControlOutcome.ERROR,
            "terminology report status contradicts its unresolved terms")
    if unverified_spans:
        return _control(
            "terminology_normalization", ControlOutcome.REVIEW_REQUIRED,
            "entity text lacks an exact source span: "
            + ", ".join(sorted(set(unverified_spans))))
    if unresolved:
        labels = sorted({
            f"{row.get('section')}:{row.get('raw_text')}"
            for row in unresolved
        })
        return _control(
            "terminology_normalization", ControlOutcome.REVIEW_REQUIRED,
            "billing-relevant terminology is unresolved: " + ", ".join(labels))
    return _control("terminology_normalization", ControlOutcome.PASS)


def _legacy_controls(result: dict) -> list[ControlResult]:
    controls = [_control(
        "pipeline_execution",
        ControlOutcome.PASS if result.get("success") else ControlOutcome.ERROR,
        "" if result.get("success") else "pipeline did not succeed")]
    cons = result.get("consistency") or {}
    repeatable = ((cons.get("runs") or 0) >= 2
                  and bool(cons.get("unanimous"))
                  and cons.get("input_consistent") is True)
    controls.append(_control(
        "repeatability", ControlOutcome.PASS if repeatable else
        ControlOutcome.REVIEW_REQUIRED,
        "" if repeatable else
        "claim outputs and critical extracted inputs are not independently unanimous"))
    independence = cons.get("model_independence") or {}
    domains = independence.get("observed_domains") or []
    profiles = independence.get("observed_profiles") or []
    try:
        required_domains = int(independence.get("required_domains"))
    except (TypeError, ValueError):
        required_domains = 0
    valid_profiles = bool(profiles) and all(
        isinstance(profile, dict)
        and str(profile.get("provider") or "").strip().lower()
        and str(profile.get("model") or "").strip()
        and str(profile.get("independence_domain") or "").strip().lower()
        == str(profile.get("provider") or "").strip().lower()
        and isinstance(profile.get("models_used"), list)
        and bool(profile.get("models_used"))
        and str(profile.get("model") or "").strip()
        in {str(value or "").strip()
            for value in profile.get("models_used") or []}
        for profile in profiles)
    derived_domains = sorted({
        str(profile.get("provider") or "").strip().lower()
        for profile in profiles if isinstance(profile, dict)
    })
    from app.core.config import MIN_INDEPENDENT_MODEL_DOMAINS
    coder_profile = result.get("model_execution") or {}
    coder_identity = (
        str(coder_profile.get("provider") or "").strip().lower(),
        str(coder_profile.get("model") or "").strip())
    observed_identities = {
        (str(profile.get("provider") or "").strip().lower(),
         str(profile.get("model") or "").strip())
        for profile in profiles if isinstance(profile, dict)
    }
    independently_corroborated = bool(
        independence.get("satisfied")
        and not independence.get("invalid_run_profiles")
        and valid_profiles
        and len(profiles) == int(cons.get("runs") or 0)
        and required_domains == MIN_INDEPENDENT_MODEL_DOMAINS
        and domains == derived_domains
        and len(derived_domains) >= required_domains
        and coder_identity in observed_identities)
    controls.append(_control(
        "independent_model_corroboration",
        ControlOutcome.PASS if independently_corroborated else
        ControlOutcome.REVIEW_REQUIRED,
        "" if independently_corroborated else
        "autonomous coding requires agreeing runs from independently operated "
        "model-provider domains; observed: " + (", ".join(domains) or "none")))
    adjudication = result.get("adjudication") or {}
    adjudication_profiles = adjudication.get("execution_profiles") or []
    if adjudication:
        try:
            adjudication_passes = int(adjudication.get("passes") or 0)
        except (TypeError, ValueError):
            adjudication_passes = 0
        valid_adjudication_profiles = (
            adjudication_passes >= 2
            and len(adjudication_profiles) == adjudication_passes
            and all(
                isinstance(profile, dict)
                and str(profile.get("provider") or "").strip().lower()
                and str(profile.get("model") or "").strip()
                and str(profile.get("independence_domain") or "").strip().lower()
                == str(profile.get("provider") or "").strip().lower()
                for profile in adjudication_profiles)
            and len({
                (str(profile.get("provider") or "").strip().lower(),
                 str(profile.get("model") or "").strip())
                for profile in adjudication_profiles
            }) >= 2)
        controls.append(_control(
            "adjudication_model_separation",
            ControlOutcome.PASS if valid_adjudication_profiles else
            ControlOutcome.REVIEW_REQUIRED,
            "" if valid_adjudication_profiles else
            "adjudication influenced the claim without two persisted, "
            "distinct model identities"))
    clean = str(result.get("final_disposition") or "").upper() == "CLEAN"
    scrub = result.get("claim_scrub") or {}
    scrub_clean = bool(scrub.get("clean")) or str(
        scrub.get("disposition") or "").upper() == "CLEAN"
    controls.append(_control(
        "compliance_scrub", ControlOutcome.PASS if clean and scrub_clean else
        ControlOutcome.BLOCKED,
        "" if clean and scrub_clean else "compliance scrub did not return CLEAN"))
    controls.append(_mandatory_filter_control(result))
    controls.append(_validation_control(result))
    controls.append(_terminology_control(result))
    controls.append(_clinical_fact_control(result))
    has_dx = bool(result.get("icd_codes"))
    has_service = bool(result.get("cpt_codes") or result.get("hcpcs_codes"))
    controls.append(_control(
        "claim_shape", ControlOutcome.PASS if has_dx and has_service else
        ControlOutcome.BLOCKED,
        "" if has_dx and has_service else
        "claim requires at least one diagnosis and one service line"))
    audit = result.get("clinical_audit") or {}
    audit_profile = audit.get("execution_profile") or {}
    coder_profile = result.get("model_execution") or {}
    coder_models = coder_profile.get("models_used") or [coder_profile.get("model")]
    coder_identities = {
        (str(coder_profile.get("provider") or "").strip().lower(),
        str(model or "").strip()) for model in coder_models if str(model or "").strip()
    }
    adjudication_identities = {
        (str(profile.get("provider") or "").strip().lower(),
         str(profile.get("model") or "").strip())
        for profile in adjudication_profiles if isinstance(profile, dict)
        and str(profile.get("provider") or "").strip()
        and str(profile.get("model") or "").strip()
    }
    audit_identity = (
        str(audit_profile.get("provider") or "").strip().lower(),
        str(audit_profile.get("model") or "").strip())
    audit_separate = bool(
        audit_profile.get("provider") and audit_profile.get("model")
        and coder_profile.get("provider") and coder_profile.get("model")
        and str(audit_profile.get("independence_domain") or "").strip().lower()
        == str(audit_profile.get("provider") or "").strip().lower()
        and audit_identity not in coder_identities
        and audit_identity not in adjudication_identities)
    audit_ok = audit.get("verdict") == "upheld" and audit_separate
    if audit_ok:
        try:
            from tools.clinical_auditor import corrections_fingerprint
            audit_ok = audit.get("fingerprint") == corrections_fingerprint(result)
        except Exception:
            audit_ok = False
    controls.append(_control(
        "independent_clinical_audit", ControlOutcome.PASS if audit_ok else
        ControlOutcome.REVIEW_REQUIRED,
        "" if audit_ok else
        "clinical audit is absent, disputed, stale, or not model-separated"))
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


def _clinical_fact_control(result: dict) -> ControlResult:
    report = result.get("clinical_facts")
    if not isinstance(report, dict) or not report:
        return _control("clinical_facts", ControlOutcome.NOT_CHECKED,
                        "normalized clinical-fact evidence is absent")
    required = {"schema_version", "facts", "unresolved_material_facts",
                "note_sha256", "facts_fingerprint", "status",
                "report_fingerprint"}
    if required - set(report) or report.get("schema_version") != 1:
        return _control("clinical_facts", ControlOutcome.ERROR,
                        "clinical-fact report is incomplete or unsupported")
    if not isinstance(report.get("facts"), list) or not isinstance(
            report.get("unresolved_material_facts"), list):
        return _control("clinical_facts", ControlOutcome.ERROR,
                        "clinical-fact report collections are malformed")
    note = str((result.get("rag_context") or {}).get("note_full_text") or "")
    if report.get("note_sha256") != "sha256:" + hashlib.sha256(
            note.encode()).hexdigest():
        return _control("clinical_facts", ControlOutcome.BLOCKED,
                        "clinical facts do not bind to the persisted note")
    if report.get("facts_fingerprint") != _fingerprint(report["facts"]):
        return _control("clinical_facts", ControlOutcome.ERROR,
                        "clinical-fact fingerprint is invalid")
    body = {key: value for key, value in report.items()
            if key != "report_fingerprint"}
    if report.get("report_fingerprint") != _fingerprint(body):
        return _control("clinical_facts", ControlOutcome.ERROR,
                        "clinical-fact report fingerprint is invalid")
    facts = report["facts"]
    evidence_errors = []
    for index, fact in enumerate(facts):
        if not isinstance(fact, dict) or not fact.get("kind"):
            evidence_errors.append(f"fact {index + 1} is malformed")
            continue
        if fact.get("kind") == "entity":
            span = fact.get("source_span") or {}
            start, end = span.get("document_start"), span.get("document_end")
            if (not span.get("verified") or not isinstance(start, int)
                    or not isinstance(end, int) or not 0 <= start < end <= len(note)
                    or note[start:end] != fact.get("raw_text")):
                evidence_errors.append(
                    f"entity fact {index + 1} lacks an exact note span")
        else:
            evidence = str(fact.get("evidence_span") or "")
            if bool(fact.get("evidence_verified")) != bool(
                    evidence and evidence in note):
                evidence_errors.append(
                    f"event fact {index + 1} has invalid evidence")
    if evidence_errors:
        return _control("clinical_facts", ControlOutcome.BLOCKED,
                        "; ".join(evidence_errors))
    unresolved = report["unresolved_material_facts"]
    if any(not isinstance(row, dict) or not row.get("kind")
           or not row.get("reason") for row in unresolved):
        return _control("clinical_facts", ControlOutcome.ERROR,
                        "unresolved clinical-fact evidence is malformed")
    expected = "REVIEW_REQUIRED" if unresolved else "PASS"
    if report.get("status") != expected:
        return _control("clinical_facts", ControlOutcome.ERROR,
                        "clinical-fact status contradicts unresolved facts")
    if unresolved:
        labels = sorted({str(row.get("label") or row.get("kind") or "fact")
                         for row in unresolved if isinstance(row, dict)})
        return _control("clinical_facts", ControlOutcome.REVIEW_REQUIRED,
                        "material clinical facts lack exact evidence: " +
                        ", ".join(labels[:20]))
    return _control("clinical_facts", ControlOutcome.PASS)


def _certificate_fingerprint(payload: dict) -> str:
    return _fingerprint({k: v for k, v in payload.items()
                         if k not in {"certificate_fingerprint",
                                      "certificate_signature"}})


def _certificate_signature(payload: dict) -> str:
    key = os.getenv(_CERTIFICATE_KEY_ENV, "")
    if len(key.encode()) < _MIN_SIGNING_KEY_BYTES:
        return ""
    return "hmac-sha256:" + hmac.new(
        key.encode(), _certificate_fingerprint(payload).encode(),
        hashlib.sha256).hexdigest()


def build_readiness_certificate(
        result: dict, *, created_at: str | None = None
) -> ClaimReadinessCertificate:
    context = _context(result)
    controls = _legacy_controls(result)
    controls += [_note_control(result), _source_control(result)]
    controls.extend(_line_controls(result))
    controls.append(_mutation_control(result))
    signing_key = os.getenv(_CERTIFICATE_KEY_ENV, "")
    strong_signing_key = len(signing_key.encode()) >= _MIN_SIGNING_KEY_BYTES
    controls.append(_control(
        "certificate_signing", ControlOutcome.PASS if strong_signing_key else
        ControlOutcome.NOT_CHECKED,
        "" if strong_signing_key else
        "claim readiness signing key is absent or shorter than 32 bytes"))
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
        "certificate_version": 2,
        "document_id": str(result.get("document_id") or ""),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "disposition": disposition.value,
        "certificate_fingerprint": "",
        "certificate_signature": "",
        "readiness_input_fingerprint": _fingerprint(
            readiness_input_payload(result)),
        "encounter_context_fingerprint": encounter_context_fingerprint(result),
        "note_fingerprint": str(integrity.get("extracted_text_sha256") or ""),
        "source_document_fingerprint": str(
            integrity.get("source_pdf_sha256") or ""),
        "context_fingerprint": _fingerprint(context),
        "claim_fingerprint": _fingerprint(claim),
        "source_manifest_fingerprint": str(manifest.get("fingerprint") or ""),
        "rule_pack_fingerprint": str(rules.get("sha256") or ""),
        "autonomous_scope_id": str((scope or {}).get("id") or ""),
        "autonomous_scope_fingerprint": scope_fingerprint(scope) if scope else "",
        "system_versions": {"release_gate": "2"},
        "claim_payload": claim,
        "controls": [c.model_dump(mode="json") for c in controls],
        "assumptions": [],
    }
    payload["certificate_fingerprint"] = _certificate_fingerprint(payload)
    payload["certificate_signature"] = _certificate_signature(payload)
    return ClaimReadinessCertificate.model_validate(payload)


def refresh_release_artifacts(result: dict) -> ClaimReadinessCertificate:
    """Rebuild provenance, mutation accounting, and the final certificate."""
    from app.release.mutation_ledger import reconcile_mutation_ledger
    from app.release.source_manifest import build_source_manifest
    prior = result.get("claim_readiness_certificate") or {}
    result["mutation_ledger"] = reconcile_mutation_ledger(
        result.get("candidate_claim") or {}, result,
        result.get("material_corrections") or [])
    result["authoritative_source_manifest"] = build_source_manifest()
    current_input = _fingerprint(readiness_input_payload(result))
    created_at = (prior.get("created_at")
                  if prior.get("readiness_input_fingerprint") == current_input
                  else None)
    cert = build_readiness_certificate(result, created_at=created_at)
    result["claim_readiness_certificate"] = cert.model_dump(mode="json")
    return cert


def verify_readiness_certificate(
        result: dict, certificate: dict | None = None
) -> tuple[bool, str]:
    try:
        cert = ClaimReadinessCertificate.model_validate(
            certificate or result.get("claim_readiness_certificate") or {})
    except Exception as exc:
        return False, f"claim-readiness certificate is absent or invalid ({exc})"
    data = cert.model_dump(mode="json")
    if cert.disposition != ReadinessDisposition.AUTO_READY:
        return False, f"readiness disposition is {cert.disposition.value}"
    if len(os.getenv(_CERTIFICATE_KEY_ENV, "").encode()) < _MIN_SIGNING_KEY_BYTES:
        return False, "claim readiness signing key is absent or too short"
    if not hmac.compare_digest(cert.certificate_signature,
                               _certificate_signature(data)):
        return False, "certificate signature is invalid"
    if cert.certificate_fingerprint != _certificate_fingerprint(data):
        return False, "certificate fingerprint is invalid"
    integrity = result.get("note_integrity") or {}
    if cert.source_document_fingerprint != str(
            integrity.get("source_pdf_sha256") or ""):
        return False, "source document changed after certification"
    if cert.note_fingerprint != str(
            integrity.get("extracted_text_sha256") or ""):
        return False, "extracted note changed after certification"
    if cert.claim_fingerprint != _fingerprint(claim_payload(result)):
        return False, "claim changed after certification"
    if cert.encounter_context_fingerprint != encounter_context_fingerprint(result):
        return False, "encounter context changed after certification"
    if cert.source_manifest_fingerprint != str(
            (result.get("authoritative_source_manifest") or {}).get(
                "fingerprint") or ""):
        return False, "authoritative source manifest changed after certification"
    if cert.readiness_input_fingerprint != _fingerprint(
            readiness_input_payload(result)):
        return False, "one or more readiness control inputs changed"
    rebuilt = build_readiness_certificate(result, created_at=cert.created_at)
    if rebuilt.model_dump(mode="json") != data:
        return False, "readiness controls no longer reproduce the certificate"
    return True, ""
