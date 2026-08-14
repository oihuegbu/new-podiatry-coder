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
SCHEMA_VERSION = 1
#: Every version this build can read. Adding a version means adding a reader,
#: never silently accepting a shape whose semantics are unknown.
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})


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
    effective_start: str = ""
    effective_end: str = ""
    authorization_id: str = ""
    authorization_number: str = ""
    authorization_effective_start: str = ""
    authorization_effective_end: str = ""


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

    Directive §6 will bind the promoted database/index snapshot itself here;
    the fields exist now so that phase fills them in rather than inventing a
    parallel attestation. `source_manifest` is carried verbatim (opaque) — this
    contract must not restate the manifest's schema.
    """

    data_fingerprint: str = ""
    source_manifest_fingerprint: str = ""
    source_manifest: dict[str, Any] = Field(default_factory=dict)
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
            units=max(1, int(getattr(line, "units", 1) or 1)),
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
    graph = GraphReference(
        clinical_event_ids=tuple(
            str(getattr(getattr(line, "fact", None), "fact_id", "") or "")
            for line in (getattr(result, "lines", None) or [])
            if getattr(getattr(line, "fact", None), "fact_id", "")),
        claim_line_intent_ids=tuple(
            str(getattr(intent, "intent_id", "") or "")
            for intent in (getattr(result, "claim_line_intents", None) or [])),
        relation_ids=tuple(
            str(getattr(relation, "relation_id", "") or "")
            for relation in (getattr(result, "relations", None) or [])),
        evidence_span_ids=tuple(sorted({
            reference.span_id
            for line in (*diagnoses, *service_lines)
            for reference in line.evidence if reference.span_id
        })),
    )

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
