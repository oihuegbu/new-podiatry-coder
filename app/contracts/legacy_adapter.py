"""TEMPORARY readers for pre-ClaimBundle artifacts.

================================================================================
SCOPE — read the second paragraph before using anything here
================================================================================
The directive permits exactly one concession while the canonical contract lands:
"A temporary legacy adapter may read old artifacts, but new artifacts must have
one path." This module is that concession and nothing more. It converts two
retired result shapes into a `ClaimBundle` VIEW so that downstream construction
(the 837P builder) has one implementation instead of two:

    app.pipeline result        -> BundleOrigin.LEGACY_APP_PIPELINE
    claude_coder.run/1 artifact -> BundleOrigin.LEGACY_CLAUDE_CODER_RUN_1

**A bundle produced here is not a natively produced bundle and must never be
treated as one.** Every bundle it returns carries a `LEGACY_*` origin, and the
release-decision boundaries dispatch on that origin: a legacy artifact is
authorized (or refused) by the legacy control battery that was designed for its
shape — `app/release/claim_readiness.verify_readiness_certificate` and the
legacy half of `claims_registry.eligible_for_auto` — never by the bundle-native
checks, which assume invariants the old producers never established. The
converse matters more: because `load_bundle()` refuses anything that does not
declare `schema_id == "claim_bundle"`, a legacy artifact can never arrive at a
native reader by accident, and a corrupt NATIVE artifact can never fall through
to this module and be re-read under weaker rules (see `is_claim_bundle`).

Two legacy semantics are reproduced here deliberately, and belong nowhere else:

  * DIAGNOSIS-POINTER BACKFILL. The old 837P builder, given a service line with
    no pointers, pointed it at every documented diagnosis (capped at four).
    That is a fabricated linkage — the record never made it — and the native
    path does not do it: a natively produced line with no necessity binding
    gets no pointers and is held. It is reproduced HERE so that already-verified
    legacy claims keep building exactly as they did, and so the fabrication is
    visible in one clearly-labelled place instead of living on in the builder.
  * `final_disposition`. A legacy-only scrub verdict with no native equivalent.
    It is carried in the audit surface, not promoted to a release field.

No medical codes appear here; codes are copied opaquely from the artifacts.
"""

from __future__ import annotations

from typing import Any

from app.contracts.claim_bundle import (
    AuditSurface, AuthorityBinding, BundleOrigin, CertificateReference,
    ClaimBundle, CodeAuthority, DiagnosisLine, EncounterIdentity,
    EvidenceReference, InvalidClaimBundle, LineMethod, ReleaseDestination,
    ReleaseStatus, ServiceLine, SourceDocument, finalize, is_claim_bundle,
)
from app.contracts.encounter_context import context_from_note_metadata

#: The round-6 interim artifact this module still reads.
CLAUDE_CODER_RUN_1 = "claude_coder.run/1"

#: 837P allows at most four diagnosis pointers per service line.  A transaction
#: cardinality from the X12 professional-claim implementation guide, not a
#: medical-code fact.
_MAX_POINTERS = 4


def is_legacy_artifact(payload: Any) -> bool:
    """Is this a result shape this module knows how to read?

    Explicitly EXCLUDES anything declaring the canonical schema id, whether or
    not it is valid: a native artifact that fails validation must surface as a
    broken native artifact, never be silently downgraded to a legacy read.
    """
    if not isinstance(payload, dict) or is_claim_bundle(payload):
        return False
    if payload.get("schema") == CLAUDE_CODER_RUN_1:
        return True
    return any(key in payload for key in
               ("icd_codes", "cpt_codes", "hcpcs_codes"))


def legacy_shape(payload: Any) -> str:
    """Name the retired shape, or raise. Used for logging and for the origin."""
    if isinstance(payload, dict) and payload.get("schema") == CLAUDE_CODER_RUN_1:
        return CLAUDE_CODER_RUN_1
    if is_legacy_artifact(payload):
        return "app.pipeline"
    raise InvalidClaimBundle(
        "payload is neither a ClaimBundle nor a recognised legacy result shape")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _units(value: Any) -> int:
    parsed = _int_or_none(value)
    return parsed if parsed and parsed >= 1 else 1


def _evidence(row: dict[str, Any]) -> tuple[EvidenceReference, ...]:
    spans = row.get("evidence_spans") or row.get("evidence") or []
    if not isinstance(spans, list):
        return ()
    return tuple(EvidenceReference(text=str(span))
                 for span in spans if str(span or "").strip())


def _legacy_pointers(row: dict[str, Any], diagnoses: list[DiagnosisLine]
                     ) -> tuple[int, ...]:
    """The retired builder's pointer resolution, in its original fidelity order.

    1. numeric pointers the old pipeline produced;
    2. else the line's `linked_diagnoses` code strings, translated to positions;
    3. else EVERY documented diagnosis (primary first) — the fabrication noted
       in the module docstring, preserved only for already-verified legacy
       claims.
    """
    order = [d.code.replace(".", "").upper() for d in diagnoses]
    raw = row.get("dx_pointers") or row.get("diagnosis_pointers")
    if isinstance(raw, list) and raw:
        pointers: list[int] = []
        for value in raw:
            parsed = _int_or_none(value)
            if parsed and 1 <= parsed <= len(order) and parsed not in pointers:
                pointers.append(parsed)
        if pointers:
            return tuple(pointers[:_MAX_POINTERS])
    linked = row.get("linked_diagnoses")
    if isinstance(linked, list) and linked and order:
        pointers = []
        for code in linked:
            normalized = str(code or "").replace(".", "").upper()
            if normalized in order:
                position = order.index(normalized) + 1
                if position not in pointers:
                    pointers.append(position)
        if pointers:
            return tuple(pointers[:_MAX_POINTERS])
    return tuple(range(1, min(len(order), _MAX_POINTERS) + 1))


def _authority(row: dict[str, Any]) -> CodeAuthority:
    ids = row.get("source_record_ids") or []
    return CodeAuthority(
        source_record_id=str(ids[0]) if isinstance(ids, list) and ids else "",
        effective_from=str(row.get("source_effective_from") or ""),
        effective_to=str(row.get("source_effective_to") or ""),
        detail={k: row[k] for k in
                ("source_record_ids", "source_temporal_authority",
                 "source_effective_from", "source_effective_to")
                if k in row},
    )


# --------------------------------------------------------------------------
# app.pipeline
# --------------------------------------------------------------------------

def bundle_from_app_pipeline_result(result: dict[str, Any],
                                    claim: dict[str, Any] | None = None
                                    ) -> ClaimBundle:
    """View an `app.pipeline` result (or a registry event's slimmed claim) as a bundle.

    `claim` lets the claims registry pass the exact arrays it verified, so the
    837P is built from the VERIFIED claim rather than from whatever the result
    file says today — the property the retired submitter's principle 1 already
    guaranteed and which must survive this refactor.
    """
    arrays = claim if isinstance(claim, dict) and claim else result
    metadata = result.get("patient_metadata") or {}
    document_id = str(result.get("document_id") or "")

    diagnoses: list[DiagnosisLine] = []
    icd_rows = [row for row in (arrays.get("icd_codes") or [])
                if isinstance(row, dict) and row.get("code")]
    primaries = [row for row in icd_rows if row.get("type") == "primary"]
    ordered = primaries + [row for row in icd_rows if row.get("type") != "primary"]
    for row in ordered:
        sequence = len(diagnoses) + 1
        diagnoses.append(DiagnosisLine(
            sequence=sequence,
            system="icd10",
            code=str(row["code"]),
            descriptor=str(row.get("description") or ""),
            method=LineMethod.UNKNOWN,
            evidence=_evidence(row),
            authority=_authority(row),
            primary=(sequence == 1),
        ))

    service_lines: list[ServiceLine] = []
    for key, system in (("cpt_codes", "cpt"), ("hcpcs_codes", "hcpcs")):
        for row in (arrays.get(key) or []):
            if not isinstance(row, dict) or not row.get("code"):
                continue
            service_lines.append(ServiceLine(
                sequence=len(service_lines) + 1,
                system=system,
                code=str(row["code"]),
                descriptor=str(row.get("description") or ""),
                method=LineMethod.UNKNOWN,
                evidence=_evidence(row),
                authority=_authority(row),
                units=_units(row.get("units")),
                modifiers=tuple(str(m) for m in (row.get("modifiers") or [])),
                diagnosis_pointers=_legacy_pointers(row, diagnoses),
                place_of_service=str(row.get("place_of_service")
                                     or metadata.get("place_of_service") or ""),
                ndc=str(row.get("ndc") or ""),
            ))

    certificate_payload = result.get("claim_readiness_certificate") or {}
    certificate = None
    if certificate_payload:
        # The legacy certificate is an HMAC-signed pydantic model with its own
        # verifier; it does NOT self-address the way the native certificate
        # does, so no content digest is asserted here. Its authorization stays
        # with `verify_readiness_certificate`.
        certificate = CertificateReference(
            certificate_sha256="",
            certificate=dict(certificate_payload),
            control_mode="legacy",
        )

    integrity = result.get("note_integrity") or {}
    bundle = ClaimBundle(
        produced_by=BundleOrigin.LEGACY_APP_PIPELINE,
        encounter=EncounterIdentity(
            encounter_id=document_id,
            document_id=document_id,
            date_of_service=str(metadata.get("date_of_service") or "") or None,
            source_document=SourceDocument(
                document_version=str(integrity.get("source_pdf_sha256") or ""),
                extracted_text_sha256=str(
                    integrity.get("extracted_text_sha256") or ""),
                page_count=_int_or_none(integrity.get("page_count")),
            ),
        ),
        diagnoses=tuple(diagnoses),
        service_lines=tuple(service_lines),
        context=context_from_note_metadata(metadata),
        authority=AuthorityBinding(
            source_manifest_fingerprint=str(
                (result.get("authoritative_source_manifest") or {})
                .get("fingerprint") or ""),
            source_manifest=result.get("authoritative_source_manifest") or {},
            # The app-side manifest publishes `records`; the coder's capability manifest
            # publishes `sources`. Two shapes, one identity -- read the one this artifact
            # actually carries rather than inventing a third.
            database_snapshot_digest=str(next(
                (r.get("sha256") for r in
                 ((result.get("authoritative_source_manifest") or {})
                  .get("records") or [])
                 if isinstance(r, dict)
                 and r.get("source_id") == "compliance_database"), "") or ""),
        ),
        certificate=certificate,
        release=ReleaseStatus(
            # A legacy artifact is never routed by this module. Its release
            # decision belongs to the legacy control battery; presenting it as
            # AUTO_READY here would let a bundle-native reader release it
            # without those controls ever running.
            destination=ReleaseDestination.REVIEW,
            producer_releasable=False,
            producer_verdict="legacy",
            producer_destination="legacy",
            reason_codes=("legacy_artifact",),
        ),
        audit=AuditSurface(
            notes=(f"final_disposition={result.get('final_disposition') or ''}",),
        ),
    )
    return finalize(bundle)


# --------------------------------------------------------------------------
# claude_coder.run/1
# --------------------------------------------------------------------------

def bundle_from_claude_coder_run_1(payload: dict[str, Any]) -> ClaimBundle:
    """View round 6's interim `claude_coder.run/1` artifact as a bundle.

    That artifact is the one the finding was raised against: it carries
    `claim_lines` and a `certificate` but no encounter context and no diagnosis
    pointers, which is precisely why the registry lost its lines. Adapting it
    cannot restore information it never held — the resulting bundle is
    UNRESOLVED context with backfilled pointers, and it exists so that an
    operator can still inspect and re-run old output, not so that it can be
    submitted.
    """
    diagnoses: list[DiagnosisLine] = []
    services: list[dict[str, Any]] = []
    for row in (payload.get("claim_lines") or []):
        if not isinstance(row, dict) or not row.get("code"):
            continue
        if str(row.get("system") or "").lower() == "icd10":
            sequence = len(diagnoses) + 1
            diagnoses.append(DiagnosisLine(
                sequence=sequence,
                system=str(row.get("system") or ""),
                code=str(row["code"]),
                descriptor=str(row.get("descriptor") or ""),
                rationale=str(row.get("rationale") or ""),
                evidence=tuple(EvidenceReference(text=str(t))
                               for t in (row.get("evidence") or [])),
                authority=CodeAuthority(detail=dict(row.get("authority") or {})),
                primary=(sequence == 1),
            ))
        else:
            services.append(row)

    service_lines = [
        ServiceLine(
            sequence=index,
            system=str(row.get("system") or ""),
            code=str(row["code"]),
            descriptor=str(row.get("descriptor") or ""),
            rationale=str(row.get("rationale") or ""),
            evidence=tuple(EvidenceReference(text=str(t))
                           for t in (row.get("evidence") or [])),
            authority=CodeAuthority(detail=dict(row.get("authority") or {})),
            units=_units(row.get("units")),
            modifiers=tuple(str(m) for m in (row.get("modifiers") or [])),
            diagnosis_pointers=_legacy_pointers(row, diagnoses),
            kind=str(row.get("kind") or ""),
        )
        for index, row in enumerate(services, start=1)
    ]

    certificate_payload = payload.get("certificate") or {}
    certificate = None
    if certificate_payload:
        certificate = CertificateReference(
            certificate_sha256=str(
                certificate_payload.get("certificate_sha256") or ""),
            certificate=dict(certificate_payload),
            control_mode=str(payload.get("control_mode") or ""),
        )

    document_id = str(payload.get("document_id") or "")
    bundle = ClaimBundle(
        produced_by=BundleOrigin.LEGACY_CLAUDE_CODER_RUN_1,
        encounter=EncounterIdentity(
            encounter_id=document_id,
            document_id=document_id,
            date_of_service=payload.get("date_of_service"),
            source_document=SourceDocument(
                filename=str(payload.get("source_pdf") or ""),
                document_version=str(payload.get("document_version") or ""),
            ),
        ),
        diagnoses=tuple(diagnoses),
        service_lines=tuple(service_lines),
        context=context_from_note_metadata(None),
        certificate=certificate,
        release=ReleaseStatus(
            destination=ReleaseDestination.REVIEW,
            producer_releasable=False,
            producer_verdict=str(payload.get("verdict") or ""),
            producer_destination=str(payload.get("destination") or ""),
            reason_codes=("legacy_artifact",),
        ),
        audit=AuditSurface(
            audit_trail=str(payload.get("audit_trail") or ""),
            audit_record_hashes=tuple(
                str(h) for h in (payload.get("audit_record_hashes") or [])),
            notes=tuple(str(n) for n in (payload.get("notes") or [])),
        ),
        processing_error=str(payload.get("error") or ""),
    )
    return finalize(bundle)


def bundle_from_legacy(payload: dict[str, Any],
                       claim: dict[str, Any] | None = None) -> ClaimBundle:
    """Dispatch to the right legacy reader, or refuse."""
    shape = legacy_shape(payload)
    if shape == CLAUDE_CODER_RUN_1:
        return bundle_from_claude_coder_run_1(payload)
    return bundle_from_app_pipeline_result(payload, claim)
