"""The canonical `ClaimBundle` — one versioned claim contract, owned by neither pipeline.

================================================================================
WHY THIS MODULE EXISTS — issue #6, finding F6-R4-A1 (P1), product directive §5
================================================================================
`run.py` produced one result shape; `tools/claims_registry.py`,
`app/release/claim_readiness.py` and `tools/claim_submitter.py` read a
*different* one. Nothing failed — the registry's `extract_claim()` simply
returned empty code arrays for a perfectly good `AUTO_READY` result and
`eligible_for_auto()` reported "pipeline did not succeed". Diagnosis and
service lines disappeared silently at the producer/consumer boundary, which
means the system could never turn a certified coding result into a claim.

The fix is not a translation shim on one side of that boundary. It is a single
strict, versioned contract that BOTH sides speak, living in neither producer's
package so neither can quietly redefine it:

    original document
        -> claude_coder.pipeline.code_encounter        (the note->code decision)
        -> ClaimBundle                                  (THIS module)
        -> tools/claims_registry.py                     (durable verified ledger)
        -> app/release/claim_readiness.py               (release authorization)
        -> tools/claim_submitter.py                     (837P construction)

Per the directive the bundle carries, in one artifact:

    encounter and source-document identities .... EncounterIdentity
    clinical/service graph references ........... GraphReference
    ordered diagnoses ........................... ClaimBundle.diagnoses
    professional service lines .................. ClaimBundle.service_lines
    diagnosis pointers .......................... ServiceLine.diagnosis_pointers
    modifiers and units ......................... ServiceLine.modifiers/.units
    patient/subscriber/payer/provider/
      facility/POS .............................. EncounterContext
    eligibility and validation outcomes ......... ClaimBundle.outcomes
    authoritative-source and index-build ids .... AuthorityBinding
    certificate and audit references ............ CertificateReference
    release status and reason codes ............. ReleaseStatus

RULES THIS MODULE ENFORCES (directive §5, verbatim: "Reject unknown schema
versions; do not infer missing fields"):

  * `load_bundle()` refuses an unrecognised `schema_id`/`schema_version` with a
    typed error. It never guesses which producer wrote a payload.
  * Models are `extra="forbid"`: a field this version does not know about is a
    rejection, not something to ignore. A future field is a version bump.
  * Absent context is NEVER defaulted to a plausible value. An encounter whose
    billing context could not be resolved is `ContextResolution.UNRESOLVED`
    with the specific missing field paths recorded, and `release_blockers()`
    is non-empty for it — so "we do not know the payer" can never present as
    "the payer field happens to be empty".
  * NO field answers "may this be billed?". `release.producer_releasable` is
    the producer's assertion; the answer is `release_blockers()` being empty,
    re-derived from the bundle's own content by whichever consumer is asking.
    A bundle asserting release while missing its certificate, while carrying a
    claim fingerprint that does not reproduce, while its diagnosis pointers
    dangle, or while its encounter context is UNRESOLVED, is stopped at every
    consumer boundary independently.

NO MEDICAL CODES APPEAR HERE, and none may. Codes, modifiers, units and
place-of-service values are carried as opaque data resolved upstream from the
authoritative source; this module never enumerates, classifies or range-checks
one. `REQUIRED_ENCOUNTER_CONTEXT` names *claim-transaction envelope fields*
(who the patient is, which payer), never code sets.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# --------------------------------------------------------------------------
# schema identity
# --------------------------------------------------------------------------

#: Producer-independent identity of this artifact shape. A reader that finds
#: anything else on disk is looking at a different contract and must say so.
SCHEMA_ID = "claim_bundle"
#: 2 adds the ORIGINAL-DOCUMENT location and reconciliation proof to
#: `EvidenceReference` (issue #6 F6-R6-A, directive §1). It is a version bump rather
#: than a silent field addition because the ABSENCE of those fields means two
#: different things: in a v1 artifact the question was never askable, in a v2 artifact
#: it was asked and the answer is recorded. A reader that could not tell those apart
#: would read "no proof recorded" as "no proof needed".
#: 3 binds the certificate to ONE COMPLETE claim rather than to a summary of it:
#: `GraphReference.graph_sha256` and the certificate's `certified_claim` seal
#: (issue #6 F7-R1). A version bump for the same reason version 2 was one --
#: the ABSENCE of the seal means two different things. In a v2 artifact the
#: certificate and the claim were only ever checked to name the same set of
#: codes; in a v3 artifact they are checked to be the SAME CLAIM -- line order,
#: units, modifiers, pointers, context, evidence, authority and graph included.
#: A reader that could not tell those apart would read an unbound certificate
#: as a bound one, which is precisely the finding.
SCHEMA_VERSION = 3
#: Every version this build can read. Adding a version means adding a reader,
#: never silently accepting a shape whose semantics are unknown.
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2, 3})


class ClaimBundleError(Exception):
    """Base class: any refusal to interpret a payload as a ClaimBundle."""


class UnknownClaimBundleSchema(ClaimBundleError):
    """The payload declares a schema id/version this build does not implement."""


class InvalidClaimBundle(ClaimBundleError):
    """The payload declares a supported schema but does not satisfy it."""


# --------------------------------------------------------------------------
# canonicalization — the ONE definition of "the same bytes"
# --------------------------------------------------------------------------

def canonical_json(value: Any) -> str:
    """Deterministic serialization used for every fingerprint in the claim path.

    Producers (`claude_coder.certificate`) and consumers (registry, readiness,
    submitter) must agree byte-for-byte or a certificate that verifies in one
    place fails in another; that agreement is this function, imported, not
    re-implemented. `sort_keys` + tight separators + no `default=` coercion:
    a value that is not JSON-native must be normalized by its owner rather
    than silently stringified into a fingerprint nobody can reproduce.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def content_digest(value: Any) -> str:
    """Bare lowercase hex sha256 over `canonical_json(value)`."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def prefixed_digest(value: Any) -> str:
    """`sha256:<hex>` — the prefixed form used by the release/manifest layer."""
    return "sha256:" + content_digest(value)


#: Key under which a release certificate carries its binding to ONE complete
#: claim, and the identity of that binding's shape. The certificate is built by
#: the producer BEFORE the encounter context, the source-document identity and
#: the authoritative-data snapshot are known; the binding is SEALED onto it by
#: `seal_claim_certificate()` at the one place that assembles the whole claim,
#: and the producer's own content address is preserved inside the seal so the
#: durable audit record that bound it still identifies the same attestation.
CERTIFIED_CLAIM_KEY = "certified_claim"
CERTIFIED_CLAIM_SCHEMA = "certified_claim/1"

#: The sections of the certified claim, keyed by their payload field, mapped to
#: how a refusal names them. The seal carries a digest PER SECTION as well as
#: over the whole payload: one aggregate digest proves the claim changed, and
#: proves nothing about WHAT changed, which turns every downstream refusal into
#: "this artifact is not coherent" — the same undiagnosable outcome that made a
#: lossy summary comparison attractive in the first place. The aggregate is
#: still the control; the sections are how it explains itself.
CERTIFIED_CLAIM_SECTIONS: dict[str, str] = {
    "encounter": "the encounter and source-document identity",
    "diagnoses": "the ordered diagnoses",
    "service_lines": "the service lines (units, modifiers, pointers, POS/NDC)",
    "context_fingerprint": "the encounter context",
    "graph": "the clinical-graph binding",
    "authority": "the authoritative data snapshot",
    "release": "the release routing",
}


def evidence_records(spans: Any) -> list[dict[str, Any]]:
    """ONE canonical record shape for a line's evidence, in documented order.

    The release certificate and this contract each carry the same spans in
    their own shape. A comparison between them is only EXACT if both are
    projected through one function: two independently written projections
    differ over an int-vs-float bounding box or a missing key long before they
    differ over a fact, and that difference reads as tampering -- or, worse,
    gets the comparison dropped as unreliable, which is how the certificate
    came to attest evidence nothing checked the claim against. Duck-typed so it
    takes a producer `EvidenceSpan` and this module's `EvidenceReference` alike.
    """
    def _int(value: Any) -> int | None:
        return None if value is None else int(value)

    out: list[dict[str, Any]] = []
    for span in (spans or []):
        region = getattr(span, "region", None)
        out.append({
            "text": str(getattr(span, "text", "") or ""),
            "span_id": str(getattr(span, "span_id", "") or ""),
            "section": str(getattr(span, "section", "") or ""),
            "page": _int(getattr(span, "page", None)),
            "start": _int(getattr(span, "start", None)),
            "end": _int(getattr(span, "end", None)),
            "text_sha256": str(getattr(span, "text_sha256", "") or ""),
            "document_sha256": str(getattr(span, "document_sha256", "") or ""),
            "document_version": str(getattr(span, "document_version", "") or ""),
            "anchored": bool(getattr(span, "anchored", False)),
            "page_image_sha256": str(getattr(span, "page_image_sha256", "") or ""),
            "region": ([float(v) for v in region] if region else None),
            "source_reconciliation": str(
                getattr(span, "source_reconciliation", "") or ""),
            "verified_by_channel_id": str(
                getattr(span, "verified_by_channel_id", "") or ""),
        })
    return out


# --------------------------------------------------------------------------
# enums
# --------------------------------------------------------------------------

class BundleOrigin(str, Enum):
    """Which code path produced this bundle. Recorded so a consumer can never
    mistake an adapted legacy artifact for a natively produced one — the whole
    class of bug this contract exists to close."""

    CLAUDE_CODER = "claude_coder.pipeline.code_encounter"
    #: Produced by `app.contracts.legacy_adapter` from a pre-cutover
    #: `app.pipeline` result. Transitional; see that module.
    LEGACY_APP_PIPELINE = "legacy_adapter:app.pipeline"
    #: Produced by `app.contracts.legacy_adapter` from round 6's interim
    #: `claude_coder.run/1` artifact.
    LEGACY_CLAUDE_CODER_RUN_1 = "legacy_adapter:claude_coder.run/1"


class ReleaseDestination(str, Enum):
    """The directive's deterministic routing destinations (§8).

    Exactly one of these is a release. Everything else names WHO must act and
    why, so operational failure, missing documentation and genuine coding
    judgement cannot collapse into one undifferentiated human queue.
    """

    AUTO_READY = "AUTO_READY"        # complete bundle, every hard invariant satisfied
    AUTO_QUERY = "AUTO_QUERY"        # the record lacks a specific code-changing fact
    SYSTEM_RETRY = "SYSTEM_RETRY"    # a dependency failed; system work, not coding work
    NON_BILLABLE = "NON_BILLABLE"    # the documented event is not claim-eligible
    BLOCKED = "BLOCKED"              # internally inconsistent / unverifiable integrity
    #: TRANSITIONAL. The directive's target routing set has no generic coder
    #: queue; today's producer still emits one, and mapping it onto any of the
    #: five above would assert something the producer did not decide. It is a
    #: distinct member so the residual volume is *measurable* while directive
    #: phase 8 (deterministic routing) removes it. Never a release.
    REVIEW = "REVIEW"


#: Producer routing vocabulary -> canonical destination. A producer value that
#: is not listed is refused rather than mapped to a guess (see
#: `canonical_destination`).
_PRODUCER_DESTINATIONS: dict[str, ReleaseDestination] = {
    "AUTO_READY": ReleaseDestination.AUTO_READY,
    "SYSTEM_HOLD": ReleaseDestination.SYSTEM_RETRY,
    "PROVIDER_QUERY": ReleaseDestination.AUTO_QUERY,
    "HOLD": ReleaseDestination.NON_BILLABLE,
    "REVIEW": ReleaseDestination.REVIEW,
    "BLOCKED": ReleaseDestination.BLOCKED,
}


def canonical_destination(producer_destination: str | None) -> ReleaseDestination:
    """Map a producer's routing value onto the canonical set, or refuse.

    An ABSENT destination is `SYSTEM_RETRY`: a result that never reached a
    routing decision is an incomplete run — system work — and must not be
    reported as a coding conclusion. An UNRECOGNISED destination raises: a
    producer that grew a new route has changed the release semantics, and
    guessing which of the six it resembles is precisely the silent
    reinterpretation this contract forbids.
    """
    key = str(producer_destination or "").strip().upper()
    if not key:
        return ReleaseDestination.SYSTEM_RETRY
    destination = _PRODUCER_DESTINATIONS.get(key)
    if destination is None:
        raise InvalidClaimBundle(
            f"producer destination {producer_destination!r} has no canonical "
            f"ClaimBundle routing; add it to _PRODUCER_DESTINATIONS with an "
            f"explicit release meaning rather than inferring one")
    return destination


class ContextResolution(str, Enum):
    """How the encounter's billing context was established (directive §2).

    `UNRESOLVED` is the honest state for a deployment with no
    `EncounterContextProvider`: the note's own extracted metadata may be
    carried as corroboration, but note text is not an authority for who the
    payer or the rendering provider is, and a bundle in this state can never
    auto-release.
    """

    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    CONFLICT = "CONFLICT"


class LineMethod(str, Enum):
    """How a code was chosen. Carried verbatim for audit; not a control input
    here (the producer's gates already acted on it)."""

    DETERMINISTIC = "deterministic"
    VERIFIED = "verified_entailment"
    ARBITRATED = "llm_arbitrated"
    ABSTAINED = "abstained"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------
# leaf models
# --------------------------------------------------------------------------

class _Strict(BaseModel):
    """Frozen + `extra="forbid"`: an unexpected key is a contract violation, and
    a bundle cannot be mutated after it is read (a consumer that "fixes up" a
    field it did not like is the drift this module prevents)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceDocument(_Strict):
    """Identity of the ORIGINAL physician document, not of any transcription."""

    filename: str = ""
    #: sha256 of the source document bytes (the PDF itself).
    document_version: str = ""
    #: sha256 of the extracted text the coder actually read.
    extracted_text_sha256: str = ""
    page_count: int | None = None


class EncounterIdentity(_Strict):
    encounter_id: str
    document_id: str
    #: ISO date. Absent (None) is preserved as absent — never today's date.
    date_of_service: str | None = None
    source_document: SourceDocument = Field(default_factory=SourceDocument)


class GraphReference(_Strict):
    """References INTO the clinical/service graph that produced these lines.

    Ids only, deliberately: the graph itself is the producer's durable audit
    record (directive §3 will make it the single clinical representation). The
    bundle binds WHICH graph nodes/edges justified the claim so the two cannot
    drift apart, without copying the graph into every claim artifact.
    """

    extraction_schema_version: str = ""
    relation_grammar_version: str = ""
    #: Content address of the WHOLE graph these ids point into. Ids alone are
    #: reusable: swapping which relation or evidence-span id a claim names, or
    #: rewriting the graph those ids resolve in, leaves every id-shaped field
    #: plausible. The digest is what makes "the same graph" checkable, and it is
    #: the same value the certificate records for its `clinical_graph`, so the
    #: two cannot come apart. (Issue #6 F7-R1.)
    graph_sha256: str = ""
    clinical_event_ids: tuple[str, ...] = ()
    claim_line_intent_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    evidence_span_ids: tuple[str, ...] = ()


class EvidenceReference(_Strict):
    """One verbatim span, with the source location that proves it.

    `anchored` is the producer's assertion that the quote was found at
    [start, end) in the source text. Directive §1 will strengthen this to a
    page region in the ORIGINAL document; the fields for that
    (`page`, `document_sha256`) already exist so the Source Evidence Compiler
    fills them rather than changing this shape.
    """

    text: str
    span_id: str = ""
    section: str = ""
    page: int | None = None
    start: int | None = None
    end: int | None = None
    text_sha256: str = ""
    document_sha256: str = ""
    document_version: str = ""
    anchored: bool = False
    #: sha256 of the rendered image of the ORIGINAL page this quotation sits on.
    page_image_sha256: str = ""
    #: (x0, top, x1, bottom) in PDF user space — the exact region on that page. Absent
    #: when the confirming channel reports no geometry (a model reading an image
    #: returns text, not boxes); an approximated box would be worse than none.
    region: tuple[float, float, float, float] | None = None
    #: A `contracts.source_evidence.ReconciliationStatus` value. Empty means no source
    #: document accompanied the encounter, which is NOT the same as unproven.
    source_reconciliation: str = ""
    #: Which independent reading confirmed (or refuted) it.
    verified_by_channel_id: str = ""


class CodeAuthority(_Strict):
    """WHICH authoritative record defines this code, and for WHEN.

    `detail` is the producer's authority dict copied verbatim (edition, table,
    row identity). It is opaque here on purpose: this contract must not grow an
    opinion about any particular code system's metadata.
    """

    source_id: str = ""
    source_record_id: str = ""
    edition: str = ""
    effective_from: str = ""
    effective_to: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class _CodedLine(_Strict):
    """Fields every coded line carries, diagnosis or service alike."""

    #: 1-based position. Explicit rather than list-index-derived so a
    #: re-ordered list is a detectable corruption, not a silent re-sequencing
    #: of the claim (diagnosis order is claim-affecting).
    sequence: int = Field(ge=1)
    system: str
    code: str
    descriptor: str = ""
    #: The clinical-graph event this line bills. The join key back to the
    #: evidence graph and the necessity binding.
    clinical_event_id: str = ""
    method: LineMethod = LineMethod.UNKNOWN
    rationale: str = ""
    evidence: tuple[EvidenceReference, ...] = ()
    authority: CodeAuthority = Field(default_factory=CodeAuthority)


class DiagnosisLine(_CodedLine):
    """One ordered diagnosis. `sequence == 1` is the first-listed diagnosis;
    `primary` records the producer's own assertion so a disagreement between
    ordering and assertion is visible instead of resolved by convention."""

    primary: bool = False


class ServiceLine(_CodedLine):
    """One professional service line: what was done, how many, with which
    modifiers, justified by which diagnoses."""

    units: int = Field(ge=1)
    modifiers: tuple[str, ...] = ()
    #: 1-based pointers into `ClaimBundle.diagnoses`. Empty means the producer
    #: established NO diagnosis linkage for this line — a release blocker, not
    #: an invitation for a downstream component to point at every diagnosis.
    diagnosis_pointers: tuple[int, ...] = ()
    #: Kept per line (a claim may mix settings) as well as at claim level.
    place_of_service: str = ""
    ndc: str = ""
    #: Fact category the line came from (procedure / imaging / supply / drug /
    #: E-M). Opaque string: a category vocabulary, never a code family.
    kind: str = ""


class PatientIdentity(_Strict):
    #: The identifier the context source knows this patient by -- the KEY the
    #: identity was resolved under, never a value copied out of the record body.
    #: It exists so every other identity the encounter resolves (above all the
    #: coverage, which is resolved in its OWN branch) can be checked to belong to
    #: THIS patient rather than merely to have resolved successfully.
    patient_id: str = ""
    first_name: str = ""
    last_name: str = ""
    date_of_birth: str = ""
    gender: str = ""
    #: The practice's own record number. Not an identity the claim asserts.
    record_number: str = ""


class SubscriberIdentity(_Strict):
    member_id: str = ""
    group_number: str = ""
    relationship_to_patient: str = ""
    authorization_number: str = ""


class PayerIdentity(_Strict):
    name: str = ""
    payer_id: str = ""
    kind: str = ""
    plan: str = ""


class ProviderIdentity(_Strict):
    npi: str = ""
    first_name: str = ""
    last_name: str = ""
    display_name: str = ""
    taxonomy_code: str = ""
    specialty: str = ""


class FacilityIdentity(_Strict):
    name: str = ""
    npi: str = ""
    address1: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""


class BillingEntityIdentity(_Strict):
    """The organization this encounter's claim is billed under.

    Resolved from the rendering provider's AFFILIATION ON THE DATE OF SERVICE,
    never from "the practice that owns the deployment". A provider who changed
    group mid-year bills the group they were affiliated with on the DOS; a
    claim that names the current group is wrong for every earlier encounter,
    and wrong in a way no downstream control can detect.
    """

    entity_id: str = ""
    name: str = ""
    npi: str = ""
    tax_id: str = ""
    taxonomy_code: str = ""


class AffiliationBinding(_Strict):
    """WHICH provider->billing-entity affiliation record was in force on the DOS.

    The window is carried, not only its verdict, so an auditor can see which
    affiliation record authorized this claim without re-reading the context
    edition that produced it — and so a re-resolution against a corrected
    roster produces a visibly different binding rather than the same answer.

    `effective_end` empty means open-ended. Both bounds are INCLUSIVE calendar
    dates; no wall clock and no timezone participates in the comparison.
    """

    affiliation_id: str = ""
    provider_npi: str = ""
    billing_entity_id: str = ""
    effective_start: str = ""
    effective_end: str = ""


class CoverageBinding(_Strict):
    """The coverage — and any prior authorization — in force on the DOS.

    `authorization_number` is populated ONLY from an authorization whose own
    declared coverage, rendering provider and facility match the ones resolved
    for THIS encounter and whose window contains the DOS. An authorization
    obtained under a different provider, facility or payer is not authorization
    for this claim, and carrying it over would be the exact silent-staleness
    the context fingerprint exists to prevent.
    """

    coverage_id: str = ""
    #: WHOSE coverage this is, and WHICH payer it names -- carried from the
    #: coverage record itself so the binding can be checked against the patient
    #: this encounter resolved independently. A claim that carries one patient's
    #: demographics with another patient's member id is fully populated, fully
    #: fingerprinted and wrong; nothing downstream can detect it from the member
    #: id alone. (Issue #6 F7-R2.)
    patient_id: str = ""
    payer_id: str = ""
    effective_start: str = ""
    effective_end: str = ""
    authorization_id: str = ""
    authorization_number: str = ""
    authorization_effective_start: str = ""
    authorization_effective_end: str = ""


#: Where a bound date of service came from. Only the first two may support a
#: release: one is an authoritative encounter-context source, the other is the
#: original document proven against an independent reading of its own page.
CONTEXT_SERVICE_DATE_SOURCE = "encounter_context"
DOCUMENT_SERVICE_DATE_SOURCE = "source_document_reconciled"
#: A date the caller simply asserted. Recorded as what it is so it can never be
#: mistaken for either of the above.
CALLER_SERVICE_DATE_SOURCE = "caller_unverified"


class ServiceDateBinding(_Strict):
    """THE date of service -- one value, one origin, one proof.

    WHY THIS IS A CONTRACT OBJECT AND NOT A STRING (issue #6 F7-R4)
    ---------------------------------------------------------------
    The DOS decides which coverage is in force, which billing affiliation the
    claim is filed under, whether an authorization applies, which edition of the
    code set is effective, and what the claim itself says the service date was.
    It was previously taken from the primary vision model's structured metadata
    and never checked against anything: a one-character misread selected a
    different coverage, a different affiliation and a different effective code
    edition, and the resulting context was still fully populated and still
    fingerprinted -- wrong, but indistinguishable from right.

    So the date is BOUND, once, with its origin recorded next to it, and the
    same bound value is what every consumer reads. `problems()` refuses the
    binding a claim may not rest on, rather than leaving the distinction to
    whichever caller happens to look.
    """

    #: The bound ISO date. Empty means NO date could be established -- which
    #: holds the encounter; it never falls back to the caller's assertion.
    date_of_service: str = ""
    #: One of the three constants above.
    source: str = ""
    #: What the encounter-context source declared for this encounter, if it
    #: declared one. Authoritative when present: it is an identifier-resolved
    #: fact, not a reading of a page.
    declared_date: str = ""
    #: What the ORIGINAL DOCUMENT's own reading proposed, and how that proposal
    #: was proven against an INDEPENDENT reading of the page it is written on
    #: (`app.contracts.source_evidence.reconcile_service_date`). Recorded even
    #: when the context source is the authority, because a document that states
    #: a different date than the roster is a conflict a human must settle.
    documented_date: str = ""
    document_status: str = ""
    document_detail: str = ""
    document_span_id: str = ""
    document_pages: tuple[int, ...] = ()
    page_image_sha256: tuple[str, ...] = ()
    verified_by_channel_id: str = ""

    def problems(self) -> tuple[str, ...]:
        out: list[str] = []
        if not self.date_of_service:
            out.append("no date of service is bound to this encounter, so no "
                       "coverage, affiliation, authorization or code-activity "
                       "decision on this claim was made against a known date")
        elif self.source not in (CONTEXT_SERVICE_DATE_SOURCE,
                                 DOCUMENT_SERVICE_DATE_SOURCE):
            out.append(
                f"the claim's date of service {self.date_of_service!r} was "
                f"supplied by the caller ({self.source or 'origin unrecorded'}) "
                f"and never established from the encounter context or "
                f"reconciled against the original document")
        return tuple(out)


class ResolutionStep(_Strict):
    """One link of the identifier chain that produced this context.

    Machine-shaped deliberately: it is fingerprinted with the rest of the
    context, so it must change when the RESOLUTION changes and must not change
    when its prose is reworded.
    """

    step: str
    identifier: str = ""
    resolved_to: str = ""
    outcome: str = ""


#: `field_sources` labels. A RESOLVED context whose required fields are not all
#: `AUTHORITATIVE` is a contract violation, not a preference — see
#: `EncounterContext.problems()`.
AUTHORITATIVE_FIELD_SOURCE = "authoritative"
CORROBORATION_FIELD_SOURCE = "note_corroboration"


#: Encounter-level context a professional claim cannot be assembled without.
#: These are transaction-envelope identities, NOT code sets — the bundle is the
#: authority for who the encounter was about; practice-level values (billing
#: provider, fee schedule, submitter) remain the submitter's configuration and
#: are deliberately absent here.
#:
#: One definition, consumed by `EncounterContext.missing_required()`, by
#: `ClaimBundle.release_blockers()`, by readiness verification and by the 837P
#: builder — so "which fields are mandatory" cannot drift between the component
#: that holds a claim and the component that builds one.
REQUIRED_ENCOUNTER_CONTEXT: tuple[str, ...] = (
    "patient.first_name",
    "patient.last_name",
    "patient.date_of_birth",
    "patient.gender",
    "subscriber.member_id",
    "coverage.coverage_id",
    "payer.name",
    "rendering_provider.npi",
    "billing_entity.entity_id",
    "billing_entity.name",
    "affiliation.affiliation_id",
    "affiliation.billing_entity_id",
    "affiliation.effective_start",
    "place_of_service",
    "jurisdiction",
)


class EncounterContext(_Strict):
    """Who the encounter was about — resolved, never inferred from the note.

    `resolution` is the load-bearing field. `UNRESOLVED` means no authoritative
    `EncounterContextProvider` established this context; any populated field is
    then CORROBORATION carried for a human/next phase, and the bundle cannot
    auto-release. That distinction is the difference between a fail-safe hold
    and a claim built from whatever a vision model read off a letterhead.
    """

    resolution: ContextResolution = ContextResolution.UNRESOLVED
    #: Identity of the adapter that resolved this (directive §2). Free-form so
    #: an EHR/FHIR, practice-management or versioned-roster adapter can each
    #: name itself.
    provider_id: str = ""
    #: Version/edition of the context source, so a changed roster changes the
    #: fingerprint and invalidates a stale authorization.
    context_version: str = ""
    #: THE date of service, bound once with its origin and its proof, and read
    #: by every consumer instead of each subsystem carrying its own copy.
    #: (Issue #6 F7-R4.)
    service_date: ServiceDateBinding = Field(default_factory=ServiceDateBinding)
    patient: PatientIdentity = Field(default_factory=PatientIdentity)
    subscriber: SubscriberIdentity = Field(default_factory=SubscriberIdentity)
    payer: PayerIdentity = Field(default_factory=PayerIdentity)
    rendering_provider: ProviderIdentity = Field(default_factory=ProviderIdentity)
    service_facility: FacilityIdentity = Field(default_factory=FacilityIdentity)
    place_of_service: str = ""
    jurisdiction: str = ""
    #: Ownership/billing context (directive §2): who the claim is billed under
    #: FOR THIS DATE OF SERVICE, which affiliation record establishes that, and
    #: which coverage/authorization was in force.
    billing_entity: BillingEntityIdentity = Field(
        default_factory=BillingEntityIdentity)
    affiliation: AffiliationBinding = Field(default_factory=AffiliationBinding)
    coverage: CoverageBinding = Field(default_factory=CoverageBinding)
    #: The identifier chain, in order, that produced the fields above. This is
    #: the auditable answer to "how was this provider selected?" — and the
    #: reason the answer can never be "the model read it off the letterhead".
    resolution_steps: tuple[ResolutionStep, ...] = ()
    #: For every `REQUIRED_ENCOUNTER_CONTEXT` path, WHICH SOURCE supplied it.
    #: A RESOLVED context in which any required field is note-derived is
    #: refused by `problems()`: extracted text corroborates, it never decides.
    field_sources: dict[str, str] = Field(default_factory=dict)
    #: Field paths the provider could not resolve, and identity disagreements
    #: between sources. Both are recorded rather than blanked so a hold names
    #: what is missing instead of "context incomplete".
    unresolved: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    #: `sha256:<hex>` over everything above. Stored (not only computed) so a
    #: consumer can detect a context edited after certification.
    fingerprint: str = ""

    @property
    def date_of_service(self) -> str:
        """The ONE bound date of service. Empty when none could be established.

        A property rather than a second field: two places to read a date from is
        exactly how different subsystems came to be deciding against different
        dates in the first place.
        """
        return self.service_date.date_of_service

    def _identity_payload(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data.pop("fingerprint", None)
        return data

    def compute_fingerprint(self) -> str:
        return prefixed_digest(self._identity_payload())

    def field_value(self, path: str) -> str:
        """Read a dotted required-field path. Unknown paths raise rather than
        return "" — a typo in `REQUIRED_ENCOUNTER_CONTEXT` must not silently
        report a field as present."""
        node: Any = self
        for part in path.split("."):
            if not hasattr(node, part):
                raise InvalidClaimBundle(
                    f"required context path {path!r} does not exist on "
                    f"EncounterContext")
            node = getattr(node, part)
        return str(node or "").strip()

    def missing_required(self) -> tuple[str, ...]:
        """Required professional-claim context fields with no value."""
        return tuple(path for path in REQUIRED_ENCOUNTER_CONTEXT
                     if not self.field_value(path))

    def problems(self) -> tuple[str, ...]:
        """Every reason this context cannot support an autonomous release."""
        out: list[str] = []
        if self.resolution is not ContextResolution.RESOLVED:
            out.append(
                f"encounter context is {self.resolution.value} "
                f"(provider={self.provider_id or 'none configured'}): a claim "
                f"cannot be built from note-extracted context alone")
        for path in self.missing_required():
            out.append(f"required encounter context is absent: {path}")
        for path in self.unresolved:
            out.append(f"context provider could not resolve: {path}")
        for conflict in self.conflicts:
            out.append(f"context sources disagree: {conflict}")
        out.extend(self._resolved_context_invariants())
        if self.fingerprint and self.fingerprint != self.compute_fingerprint():
            out.append("encounter context fingerprint does not reproduce "
                       "(context changed after it was recorded)")
        if not self.fingerprint:
            out.append("encounter context carries no fingerprint")
        return tuple(out)

    def _resolved_context_invariants(self) -> tuple[str, ...]:
        """Invariants that only a RESOLVED context can violate.

        An UNRESOLVED context is already blocked by `problems()`; running these
        against it would report the same hold twice under two different names.
        What they catch is a resolver BUG — a context that claims authority it
        does not have, or that binds a billing entity to the wrong affiliation
        or the wrong provider. Those cannot be caught downstream: every field
        would be populated and every fingerprint would reproduce.
        """
        if self.resolution is not ContextResolution.RESOLVED:
            return ()
        out: list[str] = []
        for path in REQUIRED_ENCOUNTER_CONTEXT:
            if not self.field_value(path):
                continue          # already reported by missing_required()
            origin = self.field_sources.get(path, "")
            if origin != AUTHORITATIVE_FIELD_SOURCE:
                out.append(
                    f"required encounter context {path} was not supplied by an "
                    f"authoritative context source (recorded source: "
                    f"{origin or 'none'}); note-extracted text corroborates a "
                    f"claim, it never decides one")
        if self.affiliation.billing_entity_id and self.billing_entity.entity_id \
                and self.affiliation.billing_entity_id != \
                self.billing_entity.entity_id:
            out.append(
                f"the affiliation in force binds billing entity "
                f"{self.affiliation.billing_entity_id!r} but the context names "
                f"{self.billing_entity.entity_id!r}")
        if self.affiliation.provider_npi and self.rendering_provider.npi and \
                self.affiliation.provider_npi != self.rendering_provider.npi:
            out.append(
                f"the affiliation in force belongs to provider "
                f"{self.affiliation.provider_npi!r}, not to this encounter's "
                f"rendering provider {self.rendering_provider.npi!r}")
        # The SAME class of defect for the identity pair that decides who the
        # payer is billed for: patient identity and subscriber coverage are
        # resolved in separate branches and then combined, so nothing else in the
        # system can notice that they describe two different people. A claim
        # carrying one patient's demographics with another patient's member id is
        # AUTO_READY-shaped and wrong. (Issue #6 F7-R2.)
        if self.coverage.patient_id and self.patient.patient_id and \
                self.coverage.patient_id != self.patient.patient_id:
            out.append(
                f"the coverage in force belongs to patient "
                f"{self.coverage.patient_id!r}, not to this encounter's patient "
                f"{self.patient.patient_id!r}")
        # A resolved context whose date of service was never established from an
        # authority or proven against the original document cannot support the
        # date-versioned decisions the rest of the claim rests on. (F7-R4.)
        out.extend(self.service_date.problems())
        return tuple(out)


class DecisionOutcome(_Strict):
    """One eligibility / validation / release decision, verbatim.

    `retryable` distinguishes "an authority was unavailable" (system work) from
    "the documentation does not support this" (coding work) — the distinction
    the directive's routing depends on.
    """

    stage: str
    name: str
    outcome: str
    detail: str = ""
    authority: str = ""
    retryable: bool = False


class AuthorityBinding(_Strict):
    """Exactly which authoritative data and retrieval index produced these codes.

    Directive §6: `database_snapshot_digest` binds the COMPILED DATABASE the claim was
    actually answered from. `source_manifest` is carried verbatim (opaque) — this
    contract must not restate the manifest's schema.
    """

    data_fingerprint: str = ""
    source_manifest_fingerprint: str = ""
    source_manifest: dict[str, Any] = Field(default_factory=dict)
    #: SHA-256 of the exact `compliance.db` bytes the NCCI / coverage / code-set queries
    #: ran against. Separate from `source_manifest_fingerprint` deliberately: that
    #: identifies the whole declared source SET, while the live edit decision is answered
    #: by ONE derived database compiled from it. Binding only the raw JSON digests is what
    #: let a claim be attested to bytes no query ever opened. Absent, the claim holds.
    database_snapshot_digest: str = ""
    index_build_id: str = ""
    index_checksum: str = ""
    code_counts: dict[str, int] = Field(default_factory=dict)
    model_profiles: dict[str, Any] = Field(default_factory=dict)

    def problems(self) -> tuple[str, ...]:
        out: list[str] = []
        if not self.data_fingerprint:
            out.append("no authoritative-data fingerprint is bound to this claim")
        if not self.source_manifest_fingerprint:
            out.append("no authoritative-source manifest identity is bound to "
                       "this claim")
        if not self.database_snapshot_digest:
            out.append("no compiled-database snapshot is bound to this claim; the "
                       "edit/coverage decisions cannot be traced to the bytes that "
                       "answered them")
        return tuple(out)


class CertificateReference(_Strict):
    """The producer's release certificate, carried whole plus its digest.

    Carried WHOLE rather than by reference: a downstream verifier that can only
    see a hash can prove nothing about what was certified, and a certificate
    stored somewhere else is a second artifact that can go missing.
    """

    certificate_sha256: str
    certificate: dict[str, Any] = Field(default_factory=dict)
    control_mode: str = ""

    def producer_body(self) -> dict[str, Any]:
        """The producer's own attestation: everything it certified, seal aside.

        Its content digest is the address the PIPELINE bound into the durable
        terminal release record before any claim existed. Preserving it through
        the seal is what keeps that audit record and this artifact the same
        attestation — a seal that could not be checked back to the producer's
        address would be a second, unanchored certificate.
        """
        return {k: v for k, v in self.certificate.items()
                if k not in (CERTIFIED_CLAIM_KEY, "certificate_sha256")}

    def certified_claim(self) -> dict[str, Any]:
        """This certificate's binding to ONE complete claim, or `{}`."""
        block = self.certificate.get(CERTIFIED_CLAIM_KEY)
        return dict(block) if isinstance(block, dict) else {}

    def problems(self) -> tuple[str, ...]:
        """Recompute the producer's own content address over the certificate.

        The certificate self-addresses by hashing every field except
        `certificate_sha256`; reproducing that here is what makes tampering
        with a stored bundle detectable without re-running the pipeline.
        """
        out: list[str] = []
        if not self.certificate:
            out.append("certificate reference carries no certificate")
            return tuple(out)
        if not self.certificate_sha256:
            out.append("certificate carries no content address")
            return tuple(out)
        body = {k: v for k, v in self.certificate.items()
                if k != "certificate_sha256"}
        if content_digest(body) != self.certificate_sha256:
            out.append("certificate content address does not reproduce "
                       "(the certificate or the claim it binds was altered)")
        embedded = str(self.certificate.get("certificate_sha256") or "")
        if embedded and embedded != self.certificate_sha256:
            out.append("certificate reference and certificate disagree about "
                       "the certificate's own content address")
        # The seal, if this certificate carries one, must still contain the
        # PRODUCER's intact attestation at the address the pipeline's terminal
        # audit record bound. Re-derived here rather than trusted, so a body
        # rewritten under a recomputed outer address is visible from the
        # artifact alone. (Issue #6 F7-R1.)
        seal = self.certified_claim()
        if seal:
            if str(seal.get("schema") or "") != CERTIFIED_CLAIM_SCHEMA:
                out.append(
                    f"certificate claim binding declares schema "
                    f"{seal.get('schema')!r}, not {CERTIFIED_CLAIM_SCHEMA!r}")
            if not str(seal.get("certified_claim_sha256") or ""):
                out.append("certificate claim binding names no certified claim")
            if not isinstance(seal.get("section_sha256"), dict):
                out.append("certificate claim binding carries no per-section "
                           "digests, so a refusal could not say what changed")
            producer = str(seal.get("producer_certificate_sha256") or "")
            if not producer:
                out.append("certificate claim binding does not preserve the "
                           "producer certificate's own content address")
            elif content_digest(self.producer_body()) != producer:
                out.append("the producer certificate inside this claim binding "
                           "does not reproduce its own content address (what "
                           "the producer attested was altered)")
        return tuple(out)


class ReleaseStatus(_Strict):
    """Where this encounter goes, and why.

    `producer_releasable` is deliberately NOT called `releasable`: it is the
    PRODUCER'S assertion (its verdict was AUTO_READY and it built a
    certificate), and it is an INPUT to the release decision, never the
    decision. Authorization is `ClaimBundle.release_blockers()` being empty,
    which re-derives the answer from the bundle's own content and additionally
    requires a resolved encounter context, a reproducing claim fingerprint and
    a bound authority. A field named `releasable` would be read as the answer
    by the next component that skims this model — which is exactly how the
    producer/consumer disagreement in F6-R4-A1 happened.
    """

    destination: ReleaseDestination
    producer_releasable: bool = False
    #: Verbatim producer routing, kept alongside the canonical destination so a
    #: mapping change is auditable against the run that produced it.
    producer_verdict: str = ""
    producer_destination: str = ""
    #: Stable, machine-readable reasons (gate names, control ids, hold codes).
    reason_codes: tuple[str, ...] = ()
    #: Human-readable holds, one per unmet condition.
    holds: tuple[str, ...] = ()


class AuditSurface(_Strict):
    """Everything the producer wants a human to be able to read WITHOUT re-running.

    This is the round-6 `claude_coder.run/1` artifact's independent value —
    rendered audit trail, non-billable lines, routing breakdown, documentation
    recommendations, necessity bindings. It is a section of the bundle rather
    than a second file on disk: two artifacts per note is two shapes per note,
    which is the defect this contract closes.
    """

    audit_trail: str = ""
    #: Content addresses of the durable provenance rows this encounter
    #: committed. Deliberately here and NOT on `CertificateReference`: an
    #: encounter that HELD still committed audit rows, and a field that only
    #: exists when a certificate does would report "no durable audit" for every
    #: held encounter — the observability the round-6 artifact had and which
    #: the checkpoint regression tests read.
    audit_record_hashes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    routing: tuple[dict[str, Any], ...] = ()
    recommendations: tuple[dict[str, Any], ...] = ()
    necessity_support: tuple[dict[str, Any], ...] = ()
    #: Resolved-but-not-billed lines (bundled, non-covered, excluded, held).
    #: Kept so a dropped line is visible as a decision, never as an absence.
    excluded_lines: tuple[dict[str, Any], ...] = ()


# --------------------------------------------------------------------------
# the bundle
# --------------------------------------------------------------------------

class ClaimBundle(_Strict):
    """One versioned, self-verifying claim artifact."""

    schema_id: str = SCHEMA_ID
    schema_version: int = SCHEMA_VERSION
    produced_by: BundleOrigin
    produced_at: str = ""

    encounter: EncounterIdentity
    graph: GraphReference = Field(default_factory=GraphReference)
    diagnoses: tuple[DiagnosisLine, ...] = ()
    service_lines: tuple[ServiceLine, ...] = ()
    context: EncounterContext = Field(default_factory=EncounterContext)
    outcomes: tuple[DecisionOutcome, ...] = ()
    authority: AuthorityBinding = Field(default_factory=AuthorityBinding)
    certificate: CertificateReference | None = None
    release: ReleaseStatus
    audit: AuditSurface = Field(default_factory=AuditSurface)

    #: `sha256:<hex>` over `claim_content()`. Stored so a claim edited on disk
    #: after certification is detectable by any reader, including one that
    #: cannot re-run the coder.
    claim_fingerprint: str = ""
    #: Set only when the encounter could not be processed at all. A bundle with
    #: an error is still a bundle — writing nothing would leave an earlier
    #: success on disk as the newest word on this note.
    processing_error: str = ""

    # ---------------------------------------------------------------- content

    def claim_content(self) -> dict[str, Any]:
        """The canonical billable payload — everything that can change the claim.

        Deliberately excludes rationale/audit prose and the certificate itself:
        the fingerprint must change when a code, unit, modifier, pointer,
        diagnosis order or context identity changes, and must NOT change when a
        human-readable explanation is reworded.
        """
        return {
            "encounter_id": self.encounter.encounter_id,
            "document_id": self.encounter.document_id,
            "date_of_service": self.encounter.date_of_service,
            "source_document": self.encounter.source_document.model_dump(mode="json"),
            "diagnoses": [
                {"sequence": d.sequence, "system": d.system, "code": d.code,
                 "primary": d.primary, "clinical_event_id": d.clinical_event_id}
                for d in self.diagnoses
            ],
            "service_lines": [
                {"sequence": s.sequence, "system": s.system, "code": s.code,
                 "units": s.units, "modifiers": list(s.modifiers),
                 "diagnosis_pointers": list(s.diagnosis_pointers),
                 "place_of_service": s.place_of_service, "ndc": s.ndc,
                 "clinical_event_id": s.clinical_event_id}
                for s in self.service_lines
            ],
            "context_fingerprint": self.context.fingerprint,
        }

    def compute_claim_fingerprint(self) -> str:
        return prefixed_digest(self.claim_content())

    def certified_claim_content(self) -> dict[str, Any]:
        """THE claim a certificate attests — complete, ordered, nothing summarized.

        WHY THIS EXISTS (issue #6 F7-R1)
        --------------------------------
        `claim_content()` is the change-detection fingerprint of the BILLABLE
        payload. It was never the thing a certificate was compared against; the
        certificate was compared against a sorted `(system, code)` multiset. So
        an artifact could carry nine units where one was certified, an extra
        modifier, a different patient, a different authority record and a
        different graph, recompute its own two fingerprints, and still verify —
        internally consistent, and no longer the claim anything attested.

        This payload is the fix, and its rules are:

        * ORDER IS CONTENT. Diagnoses and service lines appear in claim order
          with their recorded `sequence`; modifiers and diagnosis pointers keep
          the order the producer emitted them in (on a professional claim the
          first modifier is not interchangeable with the second, and pointer
          order is the necessity ranking).
        * NOTHING IS SUMMARIZED. Units, POS, NDC, kind, method, primary status
          and the clinical-event id are carried per line; evidence and the
          authoritative record are carried by digest over their FULL canonical
          projection, so any change to any field of any span or authority row
          changes this payload.
        * THE ENVELOPE IS PART OF THE CLAIM. Who the encounter was about (the
          recomputed context fingerprint), which document it came from, which
          authoritative data and index answered it, and which graph justified
          it are all bound here — they decide the claim as surely as the codes.
        * DERIVED STATE IS EXCLUDED. `release.holds` is `release_blockers()`,
          which is computed FROM this digest; including it would make the
          digest depend on its own verification result.
        * PROSE IS EXCLUDED. Rationale and audit text can be reworded without
          changing the claim, and a digest that moved when they did would be
          re-derived away by the first consumer it inconvenienced.
        """
        def _line(line) -> dict[str, Any]:
            return {
                "sequence": line.sequence,
                "system": line.system,
                "code": line.code,
                "descriptor": line.descriptor,
                "clinical_event_id": line.clinical_event_id,
                "method": line.method.value,
                "evidence_sha256": content_digest(evidence_records(line.evidence)),
                "authority_sha256": content_digest(
                    line.authority.model_dump(mode="json")),
            }

        return {
            "schema": CERTIFIED_CLAIM_SCHEMA,
            "encounter": {
                "encounter_id": self.encounter.encounter_id,
                "document_id": self.encounter.document_id,
                "date_of_service": self.encounter.date_of_service,
                "source_document":
                    self.encounter.source_document.model_dump(mode="json"),
            },
            "diagnoses": [dict(_line(d), primary=d.primary)
                          for d in self.diagnoses],
            "service_lines": [
                dict(_line(s),
                     units=s.units,
                     modifiers=list(s.modifiers),
                     diagnosis_pointers=list(s.diagnosis_pointers),
                     place_of_service=s.place_of_service,
                     ndc=s.ndc,
                     kind=s.kind)
                for s in self.service_lines
            ],
            # RECOMPUTED, never the stored value: the stored fingerprint is
            # what a tamperer edits, and `EncounterContext.problems()` is a
            # separate check, not this one's dependency.
            "context_fingerprint": self.context.compute_fingerprint(),
            "graph": self.graph.model_dump(mode="json"),
            "authority": self.authority.model_dump(mode="json"),
            "release": {
                "destination": self.release.destination.value,
                "producer_releasable": self.release.producer_releasable,
                "producer_verdict": self.release.producer_verdict,
                "producer_destination": self.release.producer_destination,
            },
        }

    def compute_certified_claim_digest(self) -> str:
        """`sha256:<hex>` over `certified_claim_content()` — the one exact digest
        every consumer compares the certificate against."""
        return prefixed_digest(self.certified_claim_content())

    def certified_claim_sections(self) -> dict[str, str]:
        """Per-section digests of the same payload, for DIAGNOSIS not for control.

        A refusal has to be able to say "the authoritative data snapshot changed"
        rather than "two 64-character strings differ", or the operator's only
        move is to re-run everything and hope. These are derived from exactly the
        same content as the aggregate digest, so they can never disagree with it
        about whether the claim changed — only about how to describe it.
        """
        content = self.certified_claim_content()
        return {name: content_digest(content[name])
                for name in CERTIFIED_CLAIM_SECTIONS}

    # ------------------------------------------------------------- integrity

    def integrity_problems(self) -> tuple[str, ...]:
        """Contradictions INSIDE the bundle — checked by every consumer.

        These are not policy questions ("may this claim be submitted?") but
        coherence questions ("does this artifact still describe one claim?").
        A single problem here stops the claim at whichever boundary noticed,
        because an artifact that contradicts itself cannot be reasoned about
        further.
        """
        out: list[str] = []
        if self.schema_id != SCHEMA_ID:
            out.append(f"schema id {self.schema_id!r} is not {SCHEMA_ID!r}")
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            out.append(f"schema version {self.schema_version} is not supported")

        expected = self.compute_claim_fingerprint()
        if not self.claim_fingerprint:
            out.append("claim carries no fingerprint")
        elif self.claim_fingerprint != expected:
            out.append("claim fingerprint does not reproduce "
                       "(the claim changed after it was recorded)")

        for index, line in enumerate(self.diagnoses, start=1):
            if line.sequence != index:
                out.append(f"diagnosis {line.code or '?'} is out of sequence "
                           f"(recorded {line.sequence}, listed {index})")
        primaries = [d for d in self.diagnoses if d.primary]
        if len(primaries) > 1:
            out.append("more than one diagnosis is marked primary")
        if primaries and primaries[0].sequence != 1:
            out.append("the primary diagnosis is not first-listed")

        count = len(self.diagnoses)
        for index, line in enumerate(self.service_lines, start=1):
            if line.sequence != index:
                out.append(f"service line {line.code or '?'} is out of sequence "
                           f"(recorded {line.sequence}, listed {index})")
            for pointer in line.diagnosis_pointers:
                if pointer < 1 or pointer > count:
                    out.append(
                        f"service line {line.code or '?'} points at diagnosis "
                        f"{pointer}, which does not exist (claim has {count})")
            if len(set(line.diagnosis_pointers)) != len(line.diagnosis_pointers):
                out.append(f"service line {line.code or '?'} repeats a "
                           f"diagnosis pointer")

        if self.certificate is not None:
            out.extend(self.certificate.problems())
        out.extend(self.certificate_binding_problems())
        if self.release.producer_releasable and self.certificate is None:
            out.append("bundle asserts release without a certificate")
        if self.release.producer_releasable and \
                self.release.destination is not ReleaseDestination.AUTO_READY:
            out.append(
                f"bundle asserts release while routed to "
                f"{self.release.destination.value}")
        if self.processing_error and self.release.producer_releasable:
            out.append("bundle asserts release after a processing failure")
        return tuple(out)

    def certificate_binding_problems(self) -> tuple[str, ...]:
        """Is the certificate bound to THIS EXACT claim? (Issue #6 F7-R1.)

        Part of `integrity_problems()` deliberately, not of readiness alone:
        "the certificate attests a different claim" is a coherence question, and
        putting it anywhere a consumer could skip is how the codes-only
        comparison came to be the only thing standing between a recombined
        artifact and an 837P. Every consumer that re-derives coherence — the
        registry, release authorization, and the submitter's live-artifact
        check — gets it without asking for it.

        A bundle with NO certificate is not checked here (it is already refused
        for having none). A LEGACY-origin bundle carries a certificate this
        contract did not produce and cannot re-derive; its authorization is
        `verify_readiness_certificate`, and demanding a seal of it would report
        an unfixable defect on every adapted artifact.
        """
        certificate = self.certificate
        if certificate is None:
            return ()
        native = self.produced_by is BundleOrigin.CLAUDE_CODER
        seal = certificate.certified_claim()
        if not seal:
            if not native:
                return ()
            return ("the release certificate is not bound to this claim: it "
                    "carries no certified-claim binding, so what it attests "
                    "cannot be compared with what this bundle bills",)
        out: list[str] = []
        attested = str(seal.get("certified_claim_sha256") or "")
        derived = self.compute_certified_claim_digest()
        if attested != derived:
            attested_sections = seal.get("section_sha256")
            attested_sections = (attested_sections
                                 if isinstance(attested_sections, dict) else {})
            mine = self.certified_claim_sections()
            differing = [label for name, label in CERTIFIED_CLAIM_SECTIONS.items()
                         if str(attested_sections.get(name) or "") != mine[name]]
            detail = ("it differs in " + "; ".join(differing) if differing else
                      "its certified-claim digest does not reproduce")
            out.append(
                f"the release certificate does not attest this exact claim: "
                f"{detail} (certified {attested or '<none>'}, this bundle "
                f"{derived})")
        out.extend(self._attested_line_problems(certificate.certificate))
        graph_record = certificate.certificate.get("clinical_graph")
        attested_graph = str((graph_record or {}).get("graph_sha256") or "") \
            if isinstance(graph_record, dict) else ""
        if self.graph.graph_sha256 and attested_graph and \
                self.graph.graph_sha256 != attested_graph:
            out.append(
                "the clinical graph this claim binds is not the graph the "
                "certificate attests")
        return tuple(out)

    def _attested_line_problems(self, payload: dict[str, Any]) -> tuple[str, ...]:
        """Compare the certificate's own billed lines with the bundle's, EXACTLY.

        The certificate records the producer's billable lines; this contract
        records the same lines as the claim arranges them. The two shapes are
        joined on (clinical event, system, code) — an identity, not a summary —
        and then every field BOTH shapes carry is compared: units, ordered
        modifiers, the full evidence projection and the full authority record.
        A certificate line with no counterpart is reported too: dropping a
        certified line from the claim is the tamper the old sorted-code
        multiset did catch, and widening the comparison must not lose it.

        Units and modifiers are compared only for SERVICE lines because only
        they exist on a professional diagnosis line at all; the diagnosis's own
        order and primary status are bound by the certified-claim digest.
        """
        index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for line in (payload.get("lines") or []):
            if not isinstance(line, dict):
                continue
            key = (str(line.get("clinical_event_id") or ""),
                   str(line.get("system") or ""), str(line.get("code") or ""))
            index.setdefault(key, []).append(line)

        out: list[str] = []
        for line in (*self.diagnoses, *self.service_lines):
            key = (line.clinical_event_id, line.system, line.code)
            matches = index.get(key)
            if not matches:
                out.append(
                    f"the certificate does not attest claim line "
                    f"{line.code or '?'} (event "
                    f"{line.clinical_event_id or '<none>'})")
                continue
            attested = matches.pop(0)
            if not matches:
                index.pop(key, None)
            if content_digest(evidence_records(line.evidence)) != \
                    content_digest(attested.get("evidence") or []):
                out.append(f"the certificate attests different evidence for "
                           f"claim line {line.code or '?'}")
            if content_digest(line.authority.detail) != \
                    content_digest(attested.get("authority") or {}):
                out.append(f"the certificate attests a different authoritative "
                           f"record for claim line {line.code or '?'}")
            if isinstance(line, ServiceLine):
                if attested.get("units") != line.units:
                    out.append(
                        f"the certificate attests {attested.get('units')!r} "
                        f"units for service line {line.code or '?'}, the claim "
                        f"bills {line.units}")
                if [str(m) for m in (attested.get("modifiers") or [])] != \
                        list(line.modifiers):
                    out.append(
                        f"the certificate attests different modifiers for "
                        f"service line {line.code or '?'}")
        for key, leftover in index.items():
            for _ in leftover:
                out.append(f"the certificate attests billed line {key[2] or '?'} "
                           f"(event {key[0] or '<none>'}), which this claim does "
                           f"not carry")
        return tuple(out)

    def release_blockers(self) -> tuple[str, ...]:
        """Every reason this bundle must NOT become a claim, independently derived.

        The producer's `release.releasable` flag is deliberately NOT trusted as
        the answer: it is one input, re-derived here from the bundle's own
        content so that a consumer reaching a different conclusion is a visible
        failure rather than an unnoticed one.
        """
        out: list[str] = list(self.integrity_problems())
        if self.processing_error:
            out.append(f"encounter did not process: {self.processing_error}")
        if self.release.destination is not ReleaseDestination.AUTO_READY:
            out.append(f"release destination is "
                       f"{self.release.destination.value}, not AUTO_READY")
        if not self.release.producer_releasable:
            out.append("producer did not assert an autonomous release "
                       "(no AUTO_READY verdict with a certificate)")
        out.extend(self.release.holds)
        if self.certificate is None:
            out.append("no release certificate")
        out.extend(self.context.problems())
        out.extend(self.authority.problems())
        if not self.diagnoses:
            out.append("claim has no diagnosis lines")
        if not self.service_lines:
            out.append("claim has no service lines")
        for line in self.service_lines:
            if not line.diagnosis_pointers:
                out.append(f"service line {line.code or '?'} has no diagnosis "
                           f"linkage")
        if not self.encounter.date_of_service:
            out.append("encounter has no date of service")
        # Every released line must name the clinical-graph event it bills, and that
        # event must be inside the graph binding this bundle carries. Otherwise the
        # line cannot be traced back to the evidence, relations and eligibility
        # decision that justified it — which is the whole point of the graph being
        # the single clinical representation. Scoped to natively produced bundles:
        # a legacy artifact predates the graph, is never routed by this contract and
        # is already held for that reason.
        if self.produced_by is BundleOrigin.CLAUDE_CODER:
            bound = set(self.graph.clinical_event_ids)
            for line in (*self.diagnoses, *self.service_lines):
                if line.clinical_event_id not in bound:
                    out.append(
                        f"claim line {line.code or '?'} is not bound to the clinical "
                        f"graph (event {line.clinical_event_id or '<none>'})")
            if ((self.diagnoses or self.service_lines)
                    and not self.graph.extraction_schema_version):
                out.append("claim lines carry no clinical-graph schema identity")
        # De-duplicate while preserving order: the same condition can be
        # reported by two checks (e.g. a missing certificate), and a caller
        # printing the list should not read that as two separate defects.
        seen: set[str] = set()
        ordered: list[str] = []
        for reason in out:
            if reason not in seen:
                seen.add(reason)
                ordered.append(reason)
        return tuple(ordered)

    @property
    def is_releasable(self) -> bool:
        return not self.release_blockers()

    # ----------------------------------------------------------- serialization

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------
# strict readers
# --------------------------------------------------------------------------

def is_claim_bundle(payload: Any) -> bool:
    """Does this payload CLAIM to be a ClaimBundle?

    Identity only — never validity. A caller uses this to choose a reader; it
    then still has to `load_bundle()`, which is what decides whether the
    payload actually satisfies the contract. Splitting the two is deliberate:
    "this is a bundle but a broken one" must not be reachable as "this is not a
    bundle, try the legacy adapter", which would let a corrupt new artifact be
    silently reinterpreted under old rules.
    """
    return isinstance(payload, dict) and payload.get("schema_id") == SCHEMA_ID


def load_bundle(payload: Any) -> ClaimBundle:
    """Parse a payload as a ClaimBundle, or refuse with a typed error.

    Refuses — rather than coerces — an unknown schema id, an unsupported
    version, an unknown field, or a missing required field. There is no
    "best effort" mode: this is the boundary the finding was about.
    """
    if not isinstance(payload, dict):
        raise InvalidClaimBundle(
            f"a ClaimBundle payload must be an object, got "
            f"{type(payload).__name__}")
    schema_id = payload.get("schema_id")
    if schema_id != SCHEMA_ID:
        raise UnknownClaimBundleSchema(
            f"payload declares schema_id {schema_id!r}, not {SCHEMA_ID!r}; "
            f"this build cannot interpret it and will not guess")
    version = payload.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnknownClaimBundleSchema(
            f"payload declares schema_version {version!r}; this build reads "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}")
    try:
        return ClaimBundle.model_validate(payload)
    except ValidationError as exc:
        raise InvalidClaimBundle(
            f"payload declares {SCHEMA_ID}/{version} but does not satisfy it: "
            f"{exc}") from None


# --------------------------------------------------------------------------
# producer-side construction
# --------------------------------------------------------------------------

def _evidence_of(fact) -> tuple[EvidenceReference, ...]:
    return tuple(
        EvidenceReference(
            text=str(getattr(span, "text", "") or ""),
            span_id=str(getattr(span, "span_id", "") or ""),
            section=str(getattr(span, "section", "") or ""),
            page=getattr(span, "page", None),
            start=getattr(span, "start", None),
            end=getattr(span, "end", None),
            text_sha256=str(getattr(span, "text_sha256", "") or ""),
            document_sha256=str(getattr(span, "document_sha256", "") or ""),
            document_version=str(getattr(span, "document_version", "") or ""),
            anchored=bool(getattr(span, "anchored", False)),
            page_image_sha256=str(getattr(span, "page_image_sha256", "") or ""),
            region=(tuple(float(v) for v in getattr(span, "region", None))
                    if getattr(span, "region", None) else None),
            source_reconciliation=str(
                getattr(span, "source_reconciliation", "") or ""),
            verified_by_channel_id=str(
                getattr(span, "verified_by_channel_id", "") or ""),
        )
        for span in (getattr(fact, "evidence", None) or [])
    )


def _authority_of(chosen) -> CodeAuthority:
    detail = dict(getattr(chosen, "authority", None) or {})
    return CodeAuthority(
        source_id=str(detail.get("source_id") or detail.get("source") or ""),
        source_record_id=str(detail.get("source_record_id")
                             or detail.get("record_id") or ""),
        edition=str(detail.get("edition") or detail.get("version") or ""),
        effective_from=str(detail.get("effective_from") or ""),
        effective_to=str(detail.get("effective_to") or ""),
        detail=detail,
    )


def _units_of(line) -> int:
    """The producer's STATED unit count, or a refusal. (Issue #6 F7-R1.)

    This used to be `max(1, int(getattr(line, "units", 1) or 1))`. That is a
    silent repair of a number nobody decided, and it is worse than it looks:
    the release certificate was built over the producer's ORIGINAL units, so a
    zero or negative count became a 1-unit claim attested by a certificate that
    said something else — and the codes-only comparison in force at the time
    could not see the divergence at all. A unit count is billable quantity; a
    producer that could not establish one has not produced a claim.

    An ABSENT count still means one unit. That is this contract's documented
    default for a line whose code carries no quantity dimension, not the
    rewriting of a value the producer stated.
    """
    raw = getattr(line, "units", None)
    if raw is None:
        return 1
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise InvalidClaimBundle(
            f"service line units must be a whole number, got "
            f"{type(raw).__name__} {raw!r}; a claim cannot bill a quantity "
            f"this contract had to convert")
    if raw < 1:
        raise InvalidClaimBundle(
            f"service line units must be at least 1, got {raw}; coercing an "
            f"unusable quantity to 1 would bill a number the producer never "
            f"decided and the certificate never attested")
    return raw


def _method_of(line) -> LineMethod:
    raw = getattr(getattr(line, "method", None), "value", None) or \
        str(getattr(line, "method", "") or "")
    try:
        return LineMethod(raw)
    except ValueError:
        return LineMethod.UNKNOWN


def _line_snapshot(line) -> dict[str, Any]:
    """Audit-surface record for a line that is NOT billed."""
    chosen = getattr(line, "chosen", None)
    fact = getattr(line, "fact", None)
    return {
        "system": getattr(chosen, "system", None),
        "code": getattr(chosen, "code", None),
        "descriptor": getattr(chosen, "descriptor", None),
        "kind": getattr(getattr(fact, "kind", None), "value", ""),
        "subject": getattr(fact, "description", ""),
        "method": _method_of(line).value,
        "modifiers": list(getattr(line, "modifiers", None) or []),
        "units": getattr(line, "units", None),
        "rationale": getattr(line, "rationale", ""),
        "excluded_reason": getattr(line, "excluded_reason", None),
        "documentation_gap": getattr(line, "documentation_gap", None),
        "evidence": [str(getattr(s, "text", "") or "")
                     for s in (getattr(fact, "evidence", None) or [])],
    }


def seal_claim_certificate(reference: CertificateReference,
                           bundle: ClaimBundle) -> CertificateReference:
    """Bind a producer certificate to ONE complete claim. (Issue #6 F7-R1.)

    The producer builds its certificate inside the coding pipeline, where the
    encounter context, the source-document identity and the authoritative-data
    snapshot are not yet known — so the certificate cannot, on its own, attest
    the claim that is finally assembled. Sealing is the join: the producer's
    attestation is carried through UNCHANGED, its own content address is
    preserved inside the seal (that address is what the pipeline's terminal
    durable audit record bound, and it stays checkable from the artifact), and
    the digest of the complete claim is added next to it. The whole packet is
    then re-addressed, so the certificate a consumer verifies is the one that
    attests both halves.

    Idempotent: an already-sealed certificate is re-sealed from its producer
    body, never sealed twice over its own seal.

    Refuses a producer certificate that does not reproduce its own content
    address. Sealing an already-broken attestation would launder it: the outer
    address would reproduce perfectly over a body nobody attested.
    """
    body = reference.producer_body()
    producer_sha = content_digest(body)
    declared = str(reference.certificate.get("certificate_sha256") or "")
    prior = reference.certified_claim()
    expected = str(prior.get("producer_certificate_sha256") or "") or declared
    if expected and expected != producer_sha:
        raise InvalidClaimBundle(
            "the producer certificate does not reproduce its own content "
            "address; refusing to seal a claim to an attestation that was "
            "already altered")
    body[CERTIFIED_CLAIM_KEY] = {
        "schema": CERTIFIED_CLAIM_SCHEMA,
        "producer_certificate_sha256": producer_sha,
        "certified_claim_sha256": bundle.compute_certified_claim_digest(),
        "section_sha256": bundle.certified_claim_sections(),
    }
    sealed_sha = content_digest(body)
    body["certificate_sha256"] = sealed_sha
    return reference.model_copy(update={"certificate": body,
                                        "certificate_sha256": sealed_sha})


def bundle_from_coding_result(
    result,
    *,
    source_document: SourceDocument,
    context: EncounterContext,
    authority: AuthorityBinding,
    audit_trail: str = "",
    produced_at: str = "",
    produced_by: BundleOrigin = BundleOrigin.CLAUDE_CODER,
) -> ClaimBundle:
    """Translate one coding result into the canonical bundle.

    Duck-typed on purpose: this module may not import a pipeline
    implementation (see the package docstring), so it reads the attributes a
    coding result exposes rather than naming its class. Everything that could
    change the claim is copied; nothing is invented.

    DIAGNOSIS POINTERS come from the producer's own medical-necessity binding
    (`result.necessity_support`), which records, per procedure event, the
    claim-line diagnoses that justified it and the evidence that established
    the link. A service line whose necessity binding is absent gets NO
    pointers — and `release_blockers()` then holds it. The alternative (point
    at every diagnosis, as the legacy 837P builder did when pointers were
    missing) manufactures a linkage the record never made.
    """
    diagnoses: list[DiagnosisLine] = []
    event_to_sequence: dict[str, int] = {}
    for line in getattr(result, "diagnosis_lines", None) or []:
        chosen = line.chosen
        fact = line.fact
        sequence = len(diagnoses) + 1
        event_id = str(getattr(fact, "fact_id", "") or "")
        diagnoses.append(DiagnosisLine(
            sequence=sequence,
            system=str(chosen.system),
            code=str(chosen.code),
            descriptor=str(chosen.descriptor or ""),
            clinical_event_id=event_id,
            method=_method_of(line),
            rationale=str(getattr(line, "rationale", "") or ""),
            evidence=_evidence_of(fact),
            authority=_authority_of(chosen),
            primary=(sequence == 1),
        ))
        if event_id:
            event_to_sequence[event_id] = sequence

    # procedure_event_id -> the diagnosis events the necessity gate accepted
    necessity: dict[str, list[str]] = {}
    for binding in (getattr(result, "necessity_support", None) or []):
        if not isinstance(binding, dict):
            continue
        procedure_event = str(binding.get("procedure_event_id") or "")
        if not procedure_event:
            continue
        supports = [str((s or {}).get("diagnosis_event_id") or "")
                    for s in (binding.get("supports") or [])
                    if isinstance(s, dict)]
        necessity[procedure_event] = [s for s in supports if s]

    billable = list(getattr(result, "billable_lines", None) or [])
    diagnosis_ids = {id(line) for line in
                     (getattr(result, "diagnosis_lines", None) or [])}
    service_lines: list[ServiceLine] = []
    for line in billable:
        if id(line) in diagnosis_ids:
            continue
        chosen = line.chosen
        fact = line.fact
        event_id = str(getattr(fact, "fact_id", "") or "")
        pointers: list[int] = []
        for diagnosis_event in necessity.get(event_id, []):
            sequence = event_to_sequence.get(diagnosis_event)
            if sequence and sequence not in pointers:
                pointers.append(sequence)
        service_lines.append(ServiceLine(
            sequence=len(service_lines) + 1,
            system=str(chosen.system),
            code=str(chosen.code),
            descriptor=str(chosen.descriptor or ""),
            clinical_event_id=event_id,
            method=_method_of(line),
            rationale=str(getattr(line, "rationale", "") or ""),
            evidence=_evidence_of(fact),
            authority=_authority_of(chosen),
            units=_units_of(line),
            modifiers=tuple(str(m) for m in (getattr(line, "modifiers", None) or [])),
            diagnosis_pointers=tuple(sorted(pointers)),
            place_of_service=context.place_of_service,
            kind=str(getattr(getattr(fact, "kind", None), "value", "") or ""),
        ))

    gates = list(getattr(result, "gates", None) or [])
    outcomes = tuple(
        DecisionOutcome(
            stage="release_gate",
            name=str(getattr(gate, "name", "") or ""),
            outcome=str(getattr(getattr(gate, "outcome", None), "value", "")
                        or ""),
            detail=str(getattr(gate, "detail", "") or ""),
            authority=str(getattr(gate, "authority", "") or ""),
            retryable=bool(getattr(gate, "retryable", False)),
        ) for gate in gates
    )
    eligibility_outcomes: list[DecisionOutcome] = []
    for intent in (getattr(result, "claim_line_intents", None) or []):
        for decision in (getattr(intent, "decisions", None) or []):
            eligibility_outcomes.append(DecisionOutcome(
                stage="eligibility",
                name=str(getattr(decision, "gate", "") or ""),
                outcome=str(getattr(getattr(decision, "outcome", None), "value", "")
                            or ""),
                detail=str(getattr(decision, "detail", "") or ""),
                authority=str(getattr(decision, "authority", "") or ""),
            ))

    certificate_payload = getattr(result, "certificate", None) or None
    certificate = None
    if certificate_payload:
        certificate = CertificateReference(
            certificate_sha256=str(
                certificate_payload.get("certificate_sha256") or ""),
            certificate=dict(certificate_payload),
            control_mode=str(getattr(result, "control_mode", "") or ""),
        )

    verdict = str(getattr(getattr(result, "verdict", None), "value", "") or "")
    producer_destination = str(
        getattr(getattr(result, "destination", None), "value", "") or "")
    destination = canonical_destination(producer_destination)
    # The producer's own release assertion: AUTO_READY *and* a certificate. A
    # certificate is never built when the data fingerprint, the release
    # evidence or the terminal durable write failed, so an AUTO_READY verdict
    # without one is not a release.
    producer_releasable = bool(destination is ReleaseDestination.AUTO_READY
                               and verdict == "AUTO_READY"
                               and certificate is not None)
    reason_codes = tuple(sorted({
        outcome.name for outcome in outcomes
        if outcome.outcome not in ("PASS", "NOT_APPLICABLE") and outcome.name
    }))

    excluded = tuple(
        _line_snapshot(line) for line in (getattr(result, "lines", None) or [])
        if id(line) not in {id(b) for b in billable}
    )

    encounter = EncounterIdentity(
        encounter_id=str(getattr(result, "encounter_id", "") or ""),
        document_id=str(getattr(result, "encounter_id", "") or ""),
        date_of_service=getattr(result, "date_of_service", None),
        source_document=source_document,
    )
    # ---- Clinical/service graph binding ---------------------------------
    # Bound to THE graph the producer decided from
    # (`claude_coder.graph.ClinicalGraph`), narrowed to the nodes and edges the
    # RELEASED lines actually rest on: those events, every duplicate mention
    # their claim-line intent merged in, and one documented hop outward — the
    # conditions the record gave as the reason, the components it called
    # integral, the services it called distinct. That is a binding; the previous
    # unfiltered dump of every id the run produced was not, because it stayed
    # identical whichever lines were released.
    #
    # A producer with no graph yields an EMPTY reference and `release_blockers()`
    # then refuses the claim (below) rather than letting an unbound line through
    # — the graph is not optional for a natively produced bundle.
    released_event_ids = [line.clinical_event_id
                          for line in (*diagnoses, *service_lines)
                          if line.clinical_event_id]
    reference_payload = getattr(getattr(result, "graph", None),
                                "reference_payload", None)
    graph = (GraphReference(**reference_payload(released_event_ids))
             if callable(reference_payload) else GraphReference())

    bundle = ClaimBundle(
        produced_by=produced_by,
        produced_at=produced_at,
        encounter=encounter,
        graph=graph,
        diagnoses=tuple(diagnoses),
        service_lines=tuple(service_lines),
        context=context,
        outcomes=outcomes + tuple(eligibility_outcomes),
        authority=authority,
        certificate=certificate,
        release=ReleaseStatus(
            destination=destination,
            producer_releasable=producer_releasable,
            producer_verdict=verdict,
            producer_destination=producer_destination,
            reason_codes=reason_codes,
            holds=(),
        ),
        audit=AuditSurface(
            audit_trail=audit_trail,
            audit_record_hashes=tuple(
                str(h) for h in (getattr(result, "audit_record_hashes", None) or [])),
            notes=tuple(str(n) for n in (getattr(result, "notes", None) or [])),
            routing=tuple(dict(r) for r in (getattr(result, "routing", None) or [])
                          if isinstance(r, dict)),
            recommendations=tuple(
                dict(r) for r in (getattr(result, "recommendations", None) or [])
                if isinstance(r, dict)),
            necessity_support=tuple(
                dict(r) for r in (getattr(result, "necessity_support", None) or [])
                if isinstance(r, dict)),
            excluded_lines=excluded,
        ),
    )
    # ---- Bind the certificate to THIS EXACT claim (issue #6 F7-R1) --------
    # Done here, before `finalize()`, because this is the ONE place that has the
    # whole claim: the producer's lines, the resolved encounter context, the
    # source-document identity, the authoritative snapshot and the graph. The
    # seal is what `integrity_problems()` then re-derives at every consumer.
    if bundle.certificate is not None:
        bundle = bundle.model_copy(update={
            "certificate": seal_claim_certificate(bundle.certificate, bundle)})
    return finalize(bundle)


def failure_bundle(*, document_id: str, filename: str, error: str,
                   produced_at: str = "",
                   produced_by: BundleOrigin = BundleOrigin.CLAUDE_CODER,
                   ) -> ClaimBundle:
    """A bundle for an encounter that could not be processed at all.

    SYSTEM_RETRY, not a coding destination: a note that failed to process has
    produced no coding judgement, and routing it to a human would ask a person
    to review a decision nobody made.
    """
    bundle = ClaimBundle(
        produced_by=produced_by,
        produced_at=produced_at,
        encounter=EncounterIdentity(
            encounter_id=document_id, document_id=document_id,
            source_document=SourceDocument(filename=filename)),
        release=ReleaseStatus(
            destination=ReleaseDestination.SYSTEM_RETRY,
            producer_releasable=False,
            reason_codes=("processing_failure",),
            holds=(error,)),
        processing_error=error,
    )
    return finalize(bundle)


def finalize(bundle: ClaimBundle) -> ClaimBundle:
    """Stamp the derived fingerprints and the independently derived holds.

    Called by every constructor so no producer can emit a bundle whose
    fingerprints were never computed. `release.holds` is filled from
    `release_blockers()` — with `holds` cleared first so re-finalizing is
    idempotent rather than accumulating — which means a written bundle always
    states, in the artifact, why it is not releasable.
    """
    context = bundle.context
    if context.fingerprint != context.compute_fingerprint():
        context = context.model_copy(
            update={"fingerprint": context.compute_fingerprint()})
    staged = bundle.model_copy(update={
        "context": context,
        "release": bundle.release.model_copy(update={"holds": ()}),
    })
    staged = staged.model_copy(update={
        "claim_fingerprint": staged.compute_claim_fingerprint()})
    blockers = staged.release_blockers()
    return staged.model_copy(update={
        "release": staged.release.model_copy(update={"holds": blockers}),
    })
