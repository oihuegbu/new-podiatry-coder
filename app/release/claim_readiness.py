"""Deterministic, signed authorization for one exact autonomous claim.

The certificate is the release boundary, not an informational annotation.
It binds the billable claim, encounter context, full note/source identity,
every control input, the source snapshot, and the authenticated operating
scope. Registry ingest and claim submission both verify this same artifact.

TWO ARTIFACT SHAPES, TWO BATTERIES, ONE BOUNDARY (issue #6, F6-R4-A1)
--------------------------------------------------------------------
This module now authorizes both:

  * a canonical `ClaimBundle` (`app/contracts/claim_bundle.py`) — the shape
    every NEW artifact has — via `verify_bundle_readiness()`;
  * a retired `app.pipeline` result — the shape already on disk — via
    `verify_readiness_certificate()`, unchanged.

They are separate functions rather than one polymorphic one on purpose. The
legacy battery asserts invariants that only the retired pipeline established
(N-run consistency, the mutation ledger, the compliance scrub, the clinical
audit); the bundle battery asserts invariants only the canonical contract
establishes (a reproducing claim fingerprint, a resolved encounter context, a
self-addressing certificate, bound authority). Running either battery over the
other shape would return a confident answer about controls that never ran —
which is the exact failure mode this finding was raised for. Callers dispatch
on the artifact's declared schema, never on duck-typing.
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
from app.release.source_manifest import DeclaredSourceUnavailable


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
    except DeclaredSourceUnavailable:
        # NOT swallowed: an unreadable payer registry is the one failure here that
        # silently REWRITES the certificate's context rather than leaving a field blank
        # -- payer_kind/payer_id would be recorded as "" and the certificate would be
        # signed as though the note named no payer we recognize. The certificate cannot
        # be built without the authority it is supposed to bind. (Codex F6-R5-A.)
        raise
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
    # Runtime/implementation identities this gate additionally binds (they are not
    # release-bearing DATA, so they are not part of the required-source declaration).
    mandatory = {"coverage_policy", "validator_rules",
                 "compliance_database", "validator_implementation",
                 "scrubber_implementation", "release_gate_implementation",
                 "submission_configuration", "terminology_registry",
                 "terminology_implementation"}
    # The claim-affecting DATA identities come from the SAME versioned declaration the
    # coder's release fingerprint validates against, so a newly required source cannot
    # be enforced in one release path and silently absent from the other. A declaration
    # that cannot be resolved is an ERROR, never an empty (vacuously satisfied) set.
    # (Codex F6-R5, post-fix review: adjacent instance of the parallel-list drift class.)
    try:
        from app.release.source_manifest import required_release_sources
        mandatory |= set(required_release_sources())
    except Exception as exc:
        return _control("authoritative_sources", ControlOutcome.ERROR,
                        f"required-source declaration is unavailable ({exc})")
    missing = sorted(mandatory - present)
    if missing:
        return _control("authoritative_sources", ControlOutcome.NOT_CHECKED,
                        "missing authoritative sources: " + ", ".join(missing))
    try:
        from app.compliance.engine import _parse_dos
        dos = _parse_dos(result.get("patient_metadata") or {})
    except Exception:
        dos = None
    needed = {"icd10_codes"}
    if result.get("cpt_codes"):
        needed.add("cpt_codes")
    if result.get("hcpcs_codes"):
        needed.add("hcpcs_codes")
    by_id = {str(record.get("source_id")): record for record in records}
    stale = []
    for source_id in sorted(needed):
        record = by_id.get(source_id) or {}
        start = _parse_iso(record.get("release_effective_from"))
        end = _parse_iso(record.get("release_effective_to"))
        if not dos or not start or not end or not start <= dos <= end:
            stale.append(source_id)
    if stale:
        return _control(
            "authoritative_sources", ControlOutcome.BLOCKED,
            "claim date is outside the loaded source release window: "
            + ", ".join(stale))
    return _control("authoritative_sources", ControlOutcome.PASS)


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
        from app.core.config import TERMINOLOGY_REGISTRY_FILE
        from app.release.source_manifest import sha256_file
        from app.terminology import terminology_entity_fingerprint
        live_registry = sha256_file(TERMINOLOGY_REGISTRY_FILE)
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


# --------------------------------------------------------------------------
# canonical ClaimBundle authorization
# --------------------------------------------------------------------------

def bundle_encounter_context_fingerprint(bundle) -> str:
    """The encounter-context fingerprint a bundle is bound to, RECOMPUTED.

    Recomputed rather than read: the stored value is what a tamperer would
    edit, and the registry uses this to detect a context changed after
    verification. `EncounterContext.problems()` separately reports when the
    stored and recomputed values disagree, so neither check can be the only
    one that notices.
    """
    return bundle.context.compute_fingerprint()


def verify_bundle_readiness(bundle) -> tuple[bool, str]:
    """Authorize (or refuse) one canonical `ClaimBundle` for autonomous release.

    Fail-closed and total: it returns the FIRST reason the bundle cannot be
    released, and it can only return True when every one of these holds —

      * the artifact is a `ClaimBundle` this build implements (an adapted
        legacy artifact is refused here; its authorization is
        `verify_readiness_certificate`, which runs the controls its shape
        actually has);
      * the bundle is internally coherent — claim fingerprint reproduces,
        diagnosis order and pointers resolve, the certificate's own content
        address reproduces over the certificate it carries;
      * the encounter context is RESOLVED, complete, conflict-free, and its
        fingerprint reproduces;
      * an authoritative-data and source-manifest identity is bound;
      * the release destination is AUTO_READY with a certificate, and the
        certificate attests the same encounter and verdict as the bundle.

    The last one is the cross-check that the certificate and the claim have not
    been recombined: a valid certificate from a DIFFERENT encounter, pasted
    into this bundle, still self-addresses correctly and would otherwise pass.
    """
    from app.contracts.claim_bundle import BundleOrigin, ClaimBundle

    if not isinstance(bundle, ClaimBundle):
        return False, (f"not a ClaimBundle "
                       f"({type(bundle).__name__}); refusing to authorize")
    if bundle.produced_by is not BundleOrigin.CLAUDE_CODER:
        return False, (
            f"bundle origin {bundle.produced_by.value} is an adapted legacy "
            f"artifact; bundle-native controls did not run for it")

    blockers = bundle.release_blockers()
    if blockers:
        return False, blockers[0]

    certificate = bundle.certificate
    if certificate is None:                       # unreachable via blockers
        return False, "no release certificate"    # pragma: no cover - defensive
    payload = certificate.certificate
    attested_encounter = str(payload.get("encounter_id") or "")
    if attested_encounter != bundle.encounter.encounter_id:
        return False, (f"certificate attests encounter "
                       f"{attested_encounter!r}, not "
                       f"{bundle.encounter.encounter_id!r}")
    attested_verdict = str(payload.get("verdict") or "")
    if attested_verdict != bundle.release.producer_verdict:
        return False, (f"certificate attests verdict {attested_verdict!r}, "
                       f"the bundle records {bundle.release.producer_verdict!r}")
    attested_dos = payload.get("date_of_service")
    if attested_dos != bundle.encounter.date_of_service:
        return False, ("certificate and bundle disagree about the date of "
                       "service")
    attested_codes = sorted(
        (str(line.get("system") or ""), str(line.get("code") or ""))
        for line in (payload.get("lines") or []) if isinstance(line, dict))
    bundle_codes = sorted(
        (line.system, line.code)
        for line in (*bundle.diagnoses, *bundle.service_lines))
    if attested_codes != bundle_codes:
        return False, ("certificate does not attest the same billed codes as "
                       "the bundle")
    return True, ""


def verify_bundle_artifact(payload: dict) -> tuple[bool, str]:
    """Load an on-disk payload as a bundle and authorize it, in one step.

    Load failures are refusals with their typed reason, never an empty bundle:
    "this file is not a claim bundle" and "this claim is not releasable" must
    never both surface as False-with-no-reason.
    """
    from app.contracts.claim_bundle import ClaimBundleError, load_bundle
    try:
        bundle = load_bundle(payload)
    except ClaimBundleError as exc:
        return False, str(exc)
    return verify_bundle_readiness(bundle)


# --------------------------------------------------------------------------
# legacy app.pipeline authorization
# --------------------------------------------------------------------------

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
