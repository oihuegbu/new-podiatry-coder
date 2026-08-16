"""`EncounterContextProvider` — where a claim's billing context comes from.

================================================================================
WHY THIS MODULE EXISTS — issue #6 F6-R4-A1, product directive §2
================================================================================
A professional claim needs facts the clinical note is not the authority for:
who the subscriber is, which payer covers them, which rendering provider's NPI
signs the claim, which organization the claim is billed under on that date,
which place of service applies. The deployed system previously took whichever
of those a vision model happened to read off a letterhead, or took nothing at
all — and "nothing at all" arrived downstream as empty strings that looked
exactly like "the note names no payer".

Phase 1 seeded the seam: a typed provider interface plus a flat per-encounter
context file. That file resolved nothing BY IDENTIFIER — it was one inline dump
per encounter, so "which billing entity did this provider bill under on this
date of service?" had no representation at all and could only ever be answered
"whatever the operator pasted into this encounter's block". Phase 2 makes the
resolution chain real.

THE CHAIN (directive §2, resolved by stable identifier, never by inference)
--------------------------------------------------------------------------
    encounter / document id  ->  encounter record
    encounter record         ->  patient id      ->  patient identity
    encounter record         ->  coverage        ->  subscriber + payer id
    coverage                 ->  payer id        ->  payer identity
    encounter record         ->  rendering NPI   ->  participant identity
    rendering NPI + DOS      ->  affiliation     ->  billing entity
    encounter record         ->  facility id     ->  POS + jurisdiction
    encounter record         ->  authorization   ->  coverage/provider/facility
                                                     checked, or NOT carried

Every link is an exact identifier lookup. There is no fuzzy match, no "closest"
entry and no "the practice's usual provider": selecting a provider identity by
similarity, or defaulting a billing entity to whoever owns the deployment, is
exactly what directive §2 forbids. Note extraction may CORROBORATE — and a
disagreement between the note and the resolved context is a `CONFLICT` that
holds the encounter — but it never supplies a required field. That invariant is
enforced structurally by `EncounterContext.field_sources` (see
`claim_bundle.EncounterContext.problems()`), not by convention.

TIME IS PART OF THE ANSWER
--------------------------
Affiliations, coverages and authorizations are all time-bound. A provider who
moved group mid-year bills the group they were affiliated with ON THE DATE OF
SERVICE; a claim naming their current group is wrong for every earlier
encounter, and no downstream control can detect it because every field is
populated and every fingerprint reproduces. So:

  * every window is compared as INCLUSIVE calendar dates parsed from ISO text;
  * no wall clock, `now()`, locale or timezone participates anywhere in this
    module — a resolution run today and the same run next year over the same
    edition and the same DOS produce byte-identical contexts;
  * a missing date of service does not fall back to "current": it holds.

FAILURE BOUNDARIES — what stops one encounter vs. the whole batch
-----------------------------------------------------------------
  FILE-level integrity   (unreadable, not an object, wrong/absent schema, no
                          version, a required section missing or of the wrong
                          type)  ->  `EncounterContextUnavailable`, which the
                          entrypoint turns into a refused batch. A corrupt
                          source and an unconfigured deployment must never look
                          alike, or nobody would notice the corruption.
  ROW/ENCOUNTER-level    (unknown identifier, duplicate identifier, expired or
                          conflicting affiliation/coverage, unparseable date in
                          a row, stale authorization)  ->  a named hold on THAT
                          ENCOUNTER ONLY. One operator typo in one affiliation
                          row must not stop every other encounter in the file.

No medical codes appear here. Place of service and jurisdiction are opaque
values declared by the context source and validated downstream against the
authoritative CMS sets; this module never enumerates one.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from app.contracts.claim_bundle import (
    AUTHORITATIVE_FIELD_SOURCE, CALLER_SERVICE_DATE_SOURCE,
    CONTEXT_SERVICE_DATE_SOURCE, CORROBORATION_FIELD_SOURCE,
    DOCUMENT_SERVICE_DATE_SOURCE, REQUIRED_ENCOUNTER_CONTEXT,
    AffiliationBinding, BillingEntityIdentity, ContextResolution,
    CoverageBinding, EncounterContext, FacilityIdentity, PatientIdentity,
    PayerIdentity, ProviderIdentity, ResolutionStep, ServiceDateBinding,
    SubscriberIdentity,
)
from app.contracts.source_evidence import ServiceDateEvidence


class EncounterContextUnavailable(Exception):
    """The configured context source could not be read, or is malformed.

    Typed, and never degraded into an empty context: an unreadable roster and a
    roster that says nothing about this encounter are different facts, and only
    one of them is a data-integrity incident.
    """


class _Hold(Exception):
    """A per-encounter resolution failure. NEVER batch-fatal.

    Raised inside one resolution branch and caught by that branch, so an
    unresolvable coverage does not mask an unresolvable affiliation and neither
    affects any other encounter in the same source.
    """


@runtime_checkable
class EncounterContextProvider(Protocol):
    """Resolve one encounter's billing context by stable identifiers.

    THE CONTRACT EVERY ADAPTER OWES (an EHR/FHIR, practice-management or
    scheduling adapter added later implements exactly this and nothing more):

      * `provider_id` identifies the adapter and its contract version. It is
        fingerprinted into the claim, so two adapters must never share one id.
      * `preflight()` proves the source is reachable and well-formed BEFORE any
        encounter is processed, and raises `EncounterContextUnavailable` if it
        is not. It returns an opaque descriptor for the run log. It must not
        resolve an encounter — the entrypoint previously probed the source by
        calling `resolve()` with empty identifiers, which is a real resolution
        request that happened to be harmless for one implementation.
      * `resolve()` resolves ONE encounter and reports its own failures in the
        returned `EncounterContext` (`UNRESOLVED`/`CONFLICT` with named
        reasons). It raises only for SOURCE-level failure, which is fatal to
        the batch.
      * Resolution is by identifier. An adapter that selects an identity by
        similarity, recency or "the only one we have" violates this interface
        even if it never raises.
    """

    provider_id: str

    def preflight(self) -> dict[str, Any]:
        ...

    def resolve(self, *, encounter_id: str, document_id: str,
                date_of_service: str | None,
                note_metadata: dict[str, Any] | None = None,
                document_service_date: ServiceDateEvidence | None = None
                ) -> EncounterContext:
        ...


# --------------------------------------------------------------------------
# note-metadata corroboration (shared by both providers)
# --------------------------------------------------------------------------

def _split_name(full: str) -> tuple[str, str]:
    """'First [Middle] Last' or 'Last, First' -> (first, last); ('','') if the
    value cannot be split. Never a partial guess: a single token is not a name
    a claim can carry."""
    value = str(full or "").strip()
    if not value:
        return "", ""
    if "," in value:
        last, _, first = value.partition(",")
        first = first.strip().split()[0] if first.strip() else ""
        return (first, last.strip()) if first and last.strip() else ("", "")
    parts = value.split()
    return (parts[0], parts[-1]) if len(parts) >= 2 else ("", "")


def _stamp(context: EncounterContext, *, source_label: str) -> EncounterContext:
    """Record which source supplied each populated required field, then seal.

    Called once, at the end of every construction path, so no context can be
    emitted whose `field_sources` were never recorded — which
    `EncounterContext.problems()` treats as "not authoritative" and refuses.
    """
    sources = {path: source_label for path in REQUIRED_ENCOUNTER_CONTEXT
               if context.field_value(path)}
    context = context.model_copy(update={"field_sources": sources})
    return context.model_copy(update={"fingerprint": context.compute_fingerprint()})


def context_from_note_metadata(metadata: dict[str, Any] | None,
                               service_date: ServiceDateBinding | None = None
                               ) -> EncounterContext:
    """Everything the note itself stated, as CORROBORATION only.

    Always `UNRESOLVED`, and every populated required field is labelled
    `note_corroboration`, so this context is structurally incapable of
    authorizing a release even if a future caller forgets why. It exists so
    note-derived context is carried into the bundle for a human and for the
    next phase — the finding was partly that `run.py` obtained patient metadata
    and discarded it.
    """
    meta = dict(metadata or {})
    first, last = _split_name(meta.get("patient_name") or "")
    facility = meta.get("service_facility") or {}
    if not isinstance(facility, dict):
        facility = {}
    provider_first, provider_last = _split_name(meta.get("provider") or "")
    context = EncounterContext(
        resolution=ContextResolution.UNRESOLVED,
        provider_id=NoteMetadataContextProvider.provider_id,
        context_version="",
        service_date=service_date or ServiceDateBinding(),
        patient=PatientIdentity(
            first_name=first, last_name=last,
            date_of_birth=str(meta.get("date_of_birth") or ""),
            gender=str(meta.get("gender") or meta.get("sex") or ""),
            record_number=str(meta.get("mrn") or ""),
        ),
        subscriber=SubscriberIdentity(
            member_id=str(meta.get("member_id")
                          or meta.get("insurance_id") or ""),
            group_number=str(meta.get("group_number") or ""),
            authorization_number=str(meta.get("authorization_number") or ""),
        ),
        payer=PayerIdentity(
            name=str(meta.get("insurance") or ""),
            plan=str(meta.get("insurance_plan") or meta.get("plan") or ""),
        ),
        rendering_provider=ProviderIdentity(
            npi=str(meta.get("provider_npi") or meta.get("npi") or ""),
            first_name=provider_first, last_name=provider_last,
            display_name=str(meta.get("provider") or ""),
            specialty=str(meta.get("provider_specialty") or ""),
        ),
        service_facility=FacilityIdentity(
            name=str(facility.get("name") or ""),
            address1=str(facility.get("address") or ""),
            city=str(facility.get("city") or ""),
            state=str(facility.get("state") or ""),
            postal_code=str(facility.get("zip") or ""),
        ),
        place_of_service=str(meta.get("place_of_service") or ""),
        jurisdiction=str(facility.get("state") or meta.get("state") or ""),
    )
    # The hold reason is the ABSENCE OF AN AUTHORITY, stated once, not the list
    # of fields the note happened not to mention: a note that names every field
    # is still not an authority for any of them.
    context = context.model_copy(update={
        "unresolved": ("encounter_context_provider",),
    })
    return _stamp(context, source_label=CORROBORATION_FIELD_SOURCE)


class NoteMetadataContextProvider:
    """The default when no authoritative context source is configured.

    Returns `UNRESOLVED` for every encounter, by construction. This is a
    deliberate zero-autonomy state, not a degraded one: the deployment holds
    every note until a real adapter is provisioned.
    """

    provider_id = "note_metadata_corroboration/1"

    def preflight(self) -> dict[str, Any]:
        return {"provider_id": self.provider_id, "authoritative": False,
                "detail": "no encounter context source is configured; every "
                          "encounter will hold"}

    def resolve(self, *, encounter_id: str, document_id: str,
                date_of_service: str | None,
                note_metadata: dict[str, Any] | None = None,
                document_service_date: ServiceDateEvidence | None = None
                ) -> EncounterContext:
        # The context is UNRESOLVED whatever happens here, so nothing this
        # provider returns can release a claim. The date is still BOUND rather
        # than dropped: it is the value the rest of the run is about to make
        # every date-versioned decision against, and an artifact that does not
        # record where it came from cannot be audited afterwards.
        _bound, binding = bind_service_date(
            declared=None, caller=date_of_service, evidence=document_service_date,
            holds=[], conflicts=[], steps=[])
        return context_from_note_metadata(note_metadata, service_date=binding)


# --------------------------------------------------------------------------
# versioned roster adapter
# --------------------------------------------------------------------------

#: Note-metadata keys that CORROBORATE a resolved identity, mapped to the
#: resolved context path they must agree with. Identity fields only: a
#: disagreement about the patient's birth date, member id or the NPI that
#: signed the note means the context entry and the document are not describing
#: the same encounter. The note NEVER supplies these — it only agrees or
#: disagrees with them.
_CORROBORATED: tuple[tuple[tuple[str, ...], str], ...] = (
    (("date_of_birth",), "patient.date_of_birth"),
    (("member_id", "insurance_id"), "subscriber.member_id"),
    (("provider_npi", "npi"), "rendering_provider.npi"),
)

#: Sections keyed by a stable identifier. A duplicated key here means the
#: identifier is not stable, which is precisely the property the whole design
#: rests on — so any encounter resolving THROUGH a duplicated key holds.
_MAPPING_SECTIONS: tuple[str, ...] = (
    "patients", "payers", "providers", "billing_entities", "facilities",
    "encounters",
)
#: Sections that are lists of rows, each carrying its own identifier field.
_LIST_SECTIONS: dict[str, str] = {
    "coverages": "coverage_id",
    "affiliations": "affiliation_id",
    "authorizations": "authorization_id",
}
#: List sections a source may omit entirely (an authorization is optional; a
#: coverage or an affiliation is not — omitting them would mean no encounter in
#: the file could ever resolve, which is a file-level mistake worth naming).
_OPTIONAL_SECTIONS: frozenset[str] = frozenset({"authorizations"})


def _normalized(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def _as_date(value: Any, label: str) -> date | None:
    """An ISO calendar date, or None when the value is absent.

    An UNPARSEABLE value raises `_Hold` rather than being treated as absent:
    silently reading a typo'd bound as "open-ended" would widen an affiliation
    window instead of holding, which is the failure direction that bills a
    claim under the wrong entity.
    """
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise _Hold(f"{label} is not an ISO calendar date: {text!r}") from None


def _window(row: dict[str, Any], label: str) -> tuple[date, date | None]:
    """(start, end) for a time-bound row. `start` is REQUIRED.

    An affiliation with no declared start is indistinguishable from "the
    affiliation this provider has always had", which is the shortcut the whole
    time-bound design exists to avoid — so it is refused rather than treated as
    open-ended in the past.
    """
    start = _as_date(row.get("effective_start"), f"{label} effective_start")
    if start is None:
        raise _Hold(f"{label} declares no effective_start; a time-bound record "
                    f"with no start cannot be evaluated against a date of service")
    end = _as_date(row.get("effective_end"), f"{label} effective_end")
    if end is not None and end < start:
        raise _Hold(f"{label} ends ({end.isoformat()}) before it starts "
                    f"({start.isoformat()})")
    return start, end


def _covers(dos: date, start: date, end: date | None) -> bool:
    """Is `dos` inside [start, end]? BOTH BOUNDS INCLUSIVE.

    Stated explicitly because the off-by-one has a direction that matters: a
    service performed on the last day of an affiliation is covered by it, and an
    exclusive end would silently reassign that encounter's billing entity.
    Pure calendar-date comparison — no clock, no timezone, no locale.
    """
    return start <= dos and (end is None or dos <= end)


def bind_service_date(*, declared: Any, caller: str | None,
                      evidence: ServiceDateEvidence | None,
                      holds: list[str], conflicts: list[str],
                      steps: list[ResolutionStep]
                      ) -> tuple[date | None, ServiceDateBinding]:
    """Establish THE date of service, once, and say where it came from.

    ISSUE #6 F7-R4 -- WHY THIS IS NOT "parse whatever the caller passed"
    --------------------------------------------------------------------
    The caller's date of service used to be the primary vision model's structured
    metadata field, accepted unchecked. Every time-bound decision on the claim --
    which coverage is in force, which billing affiliation applies, whether an
    authorization covers the service, which edition of the code set is effective,
    and the service date the claim itself carries -- was then made against a value
    nothing had ever compared to the original document. A one-character misread
    produced a fully populated, fully fingerprinted context for the wrong date.

    There are exactly two things that may establish a claim's date of service:

      1. THE ENCOUNTER CONTEXT SOURCE, when it declares one for this encounter.
         That is an identifier-resolved fact from an authority, not a reading of a
         page, so it wins.
      2. THE ORIGINAL DOCUMENT, when the date it states has been located on a page
         and PROVEN there against an independent reading of that page
         (`source_evidence.reconcile_service_date`). A document date that could not
         be located, or that an independent channel reads differently, establishes
         nothing -- and is refused rather than used, because a date nobody could
         confirm looks exactly like a confirmed one downstream.

    A caller assertion binds only when NO source-evidence document accompanied the
    encounter at all (note text supplied directly, nothing to reconcile against).
    It is labelled `CALLER_SERVICE_DATE_SOURCE`, which
    `EncounterContext.problems()` refuses for a RESOLVED context -- so it can carry
    a date through an audit trail, and can never carry one onto a claim.

    Disagreements between any two of the three are CONFLICTS, not corrections: the
    document and the roster naming different service dates is a question for a
    human, and silently preferring either is how the wrong one gets billed.
    """
    documented: date | None = None
    document_fields: dict[str, Any] = {}
    if evidence is not None:
        document_fields = {
            "documented_date": evidence.candidate,
            "document_status": evidence.status.value,
            "document_detail": evidence.detail,
            "document_span_id": evidence.span_id,
            "document_pages": evidence.pages,
            "page_image_sha256": evidence.page_image_sha256,
            "verified_by_channel_id": evidence.verified_by_channel_id,
        }
        try:
            documented = _as_date(evidence.candidate, "documented date of service")
        except _Hold as hold:                     # pragma: no cover - always ISO
            holds.append(str(hold))

    declared_date: date | None = None
    declared_raw = str(declared or "").strip()
    if declared_raw:
        try:
            declared_date = _as_date(declared_raw, "context source date_of_service")
        except _Hold as hold:
            holds.append(str(hold))

    caller_date: date | None = None
    if str(caller or "").strip():
        try:
            caller_date = _as_date(caller, "encounter date of service")
        except _Hold as hold:
            holds.append(str(hold))

    if declared_date is not None and documented is not None and \
            declared_date != documented:
        conflicts.append(
            f"date of service: the encounter context source says "
            f"{declared_date.isoformat()!r}, the document says "
            f"{documented.isoformat()!r}")
    if documented is not None and caller_date is not None and \
            documented != caller_date:
        conflicts.append(
            f"date of service: the caller supplied {caller_date.isoformat()!r}, "
            f"the document's own reading states {documented.isoformat()!r}")
    if declared_date is not None and caller_date is not None and \
            documented is None and declared_date != caller_date:
        conflicts.append(
            f"date of service: the encounter context source says "
            f"{declared_date.isoformat()!r}, the caller supplied "
            f"{caller_date.isoformat()!r}")

    bound: date | None = None
    source = ""
    if declared_date is not None:
        bound, source = declared_date, CONTEXT_SERVICE_DATE_SOURCE
    elif evidence is not None:
        if documented is not None and evidence.reconciled:
            bound, source = documented, DOCUMENT_SERVICE_DATE_SOURCE
        else:
            holds.append(
                f"the encounter's date of service is not established: the "
                f"encounter context source declares none, and the date the "
                f"document reports ({evidence.candidate or 'none readable'}) is "
                f"{evidence.status.value} against an independent reading of the "
                f"original page ({evidence.detail})")
    elif caller_date is not None:
        bound, source = caller_date, CALLER_SERVICE_DATE_SOURCE
    else:
        holds.append(
            "encounter has no usable date of service, so no time-bound "
            "affiliation, coverage or authorization can be resolved")

    steps.append(ResolutionStep(
        step="date_of_service",
        identifier=(bound.isoformat() if bound is not None else ""),
        resolved_to=source,
        outcome=("resolved" if bound is not None else "absent")))
    return bound, ServiceDateBinding(
        date_of_service=(bound.isoformat() if bound is not None else ""),
        source=source,
        declared_date=(declared_date.isoformat() if declared_date is not None else ""),
        **document_fields)


def _describe(row: dict[str, Any]) -> str:
    start = str(row.get("effective_start") or "?")
    end = str(row.get("effective_end") or "open-ended")
    return f"{start}..{end}"


def _section(entry: dict[str, Any], model) -> dict[str, Any]:
    """One identity record, restricted to the fields the model declares.

    An unknown key is DROPPED rather than passed through: the source is
    operator-maintained data, and a typo there must not raise a validation
    error that holds every encounter in the file. The corresponding required
    field then simply stays empty and is reported by `missing_required()` —
    which names the field the operator meant to set.
    """
    if not isinstance(entry, dict):
        return {}
    allowed = set(model.model_fields)
    return {k: str(v or "") for k, v in entry.items() if k in allowed}


def _duplicate_mapping_keys(text: str) -> dict[str, tuple[str, ...]]:
    """Identifier keys that appear TWICE in a keyed section.

    `json.loads` keeps the last of a duplicated key and says nothing, so a
    source with two `"ENC-1"` entries resolves silently to whichever the
    operator wrote second. The whole design rests on identifiers being stable,
    so the file is parsed a second time with the raw key/value pairs preserved
    and the collisions are recorded. An encounter resolving through a collided
    identifier holds; the rest of the file is unaffected.
    """
    try:
        top = json.loads(text, object_pairs_hook=list)
    except ValueError:                       # already reported by the real load
        return {}
    if not isinstance(top, list):
        return {}
    found: dict[str, tuple[str, ...]] = {}
    for pair in top:
        # `object_pairs_hook` is handed a list of TUPLES, not of lists.
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        section, value = pair
        if section not in _MAPPING_SECTIONS or not isinstance(value, list):
            continue
        seen: set[str] = set()
        duplicates: set[str] = set()
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            key = item[0]
            if key in seen:
                duplicates.add(key)
            seen.add(key)
        if duplicates:
            found[section] = tuple(sorted(duplicates))
    return found


class VersionedRosterContextProvider:
    """Encounter context from a versioned, checked-in context source.

    "A versioned local roster import" (directive §2) — the source a medium
    private surgical practice actually controls without an integration project.
    It is normalized, not a per-encounter dump: patients, payers, coverages,
    providers, billing entities, affiliations and facilities are declared ONCE
    and referenced by identifier, so the same provider cannot be affiliated
    with two different entities in two different encounters' inline blocks
    without the source itself being self-contradictory.

    File shape (`encounter_context/2`; a malformed file is
    `EncounterContextUnavailable`, never an empty roster):

        {
          "schema": "encounter_context/2",
          "version": "<edition identifier>",
          "patients":   {"<patient_id>": {first_name, last_name,
                                          date_of_birth, gender,
                                          record_number}},
          "payers":     {"<payer_id>": {name, payer_id, kind, plan}},
          "providers":  {"<npi>": {npi, first_name, last_name, display_name,
                                   taxonomy_code, specialty}},
          "billing_entities": {"<entity_id>": {name, npi, tax_id,
                                               taxonomy_code}},
          "facilities": {"<facility_id>": {name, npi, address1, city, state,
                                           postal_code, place_of_service,
                                           jurisdiction}},
          "coverages":  [{coverage_id, patient_id, payer_id, member_id,
                          group_number, relationship_to_patient,
                          effective_start, effective_end}],
          "affiliations": [{affiliation_id, provider_npi, billing_entity_id,
                            effective_start, effective_end}],
          "authorizations": [{authorization_id, authorization_number,
                              coverage_id, rendering_provider_npi, facility_id,
                              effective_start, effective_end}],
          "encounters": {"<encounter or document id>": {
                            patient_id, rendering_provider_npi, facility_id,
                            coverage_id?, authorization_id?, date_of_service?}}
        }

    `encounter_context/1` — phase 1's flat per-encounter dump — is REFUSED with
    a message naming this schema. It could not express an affiliation, a
    coverage window or an authorization, so accepting it would mean either
    silently releasing claims whose billing entity was never resolved for the
    date of service, or carrying a second half-correct code path forever.
    """

    provider_id = "versioned_roster/2"
    schema = "encounter_context/2"
    superseded_schemas = ("encounter_context/1",)

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._loaded: dict[str, Any] | None = None
        self._duplicate_keys: dict[str, tuple[str, ...]] = {}
        self._duplicate_rows: dict[str, tuple[str, ...]] = {}

    # ------------------------------------------------------------------ load
    def _load(self) -> dict[str, Any]:
        if self._loaded is not None:
            return self._loaded
        try:
            text = self.path.read_text()
        except OSError as exc:
            raise EncounterContextUnavailable(
                f"encounter context file {self.path} could not be read: "
                f"{exc}") from None
        try:
            raw = json.loads(text)
        except ValueError as exc:
            raise EncounterContextUnavailable(
                f"encounter context file {self.path} is not valid JSON: "
                f"{exc}") from None
        if not isinstance(raw, dict):
            raise EncounterContextUnavailable(
                f"encounter context file {self.path} must contain an object")
        declared = raw.get("schema")
        if declared in self.superseded_schemas:
            raise EncounterContextUnavailable(
                f"encounter context file {self.path} declares the superseded "
                f"schema {declared!r}, which cannot express a coverage window, "
                f"a provider affiliation for a date of service, or a prior "
                f"authorization. Migrate it to {self.schema!r}; reading it "
                f"would mean releasing claims whose billing entity was never "
                f"resolved for the date of service.")
        if declared != self.schema:
            raise EncounterContextUnavailable(
                f"encounter context file {self.path} declares schema "
                f"{declared!r}, not {self.schema!r}")
        if not str(raw.get("version") or "").strip():
            raise EncounterContextUnavailable(
                f"encounter context file {self.path} declares no version; a "
                f"context edition must be identifiable to be fingerprinted")
        for name in _MAPPING_SECTIONS:
            if not isinstance(raw.get(name), dict):
                raise EncounterContextUnavailable(
                    f"encounter context file {self.path} has no {name!r} "
                    f"object; every identity a claim needs must be declared "
                    f"once and referenced by identifier")
        for name in _LIST_SECTIONS:
            value = raw.get(name)
            if value is None and name in _OPTIONAL_SECTIONS:
                continue
            if not isinstance(value, list) or \
                    any(not isinstance(row, dict) for row in value):
                raise EncounterContextUnavailable(
                    f"encounter context file {self.path} has no {name!r} list "
                    f"of records")
        self._duplicate_keys = _duplicate_mapping_keys(text)
        self._duplicate_rows = {
            name: _duplicate_row_ids(raw.get(name) or [], id_field)
            for name, id_field in _LIST_SECTIONS.items()
        }
        self._loaded = raw
        return raw

    def preflight(self) -> dict[str, Any]:
        """Prove the source is readable and well-formed before any note runs.

        Deliberately NOT `resolve()` with empty identifiers: that is a real
        resolution request whose harmlessness is an implementation accident,
        and an adapter that hit a remote system would have issued a live query
        for an encounter that does not exist.
        """
        raw = self._load()
        return {
            "provider_id": self.provider_id,
            "authoritative": True,
            "path": str(self.path),
            "version": str(raw.get("version") or ""),
            "encounters": len(raw.get("encounters") or {}),
            "duplicate_identifiers": {**self._duplicate_keys,
                                      **{k: v for k, v in
                                         self._duplicate_rows.items() if v}},
        }

    def source_version(self) -> str:
        return str(self._load().get("version") or "")

    # --------------------------------------------------------------- resolve
    def resolve(self, *, encounter_id: str, document_id: str,
                date_of_service: str | None,
                note_metadata: dict[str, Any] | None = None,
                document_service_date: ServiceDateEvidence | None = None
                ) -> EncounterContext:
        raw = self._load()
        version = str(raw.get("version") or "")
        steps: list[ResolutionStep] = []
        holds: list[str] = []
        conflicts: list[str] = []

        try:
            entry, key = self._encounter_entry(raw, encounter_id, document_id)
        except _Hold as hold:
            steps.append(ResolutionStep(
                step="encounter", identifier=encounter_id or document_id,
                outcome=str(hold)))
            return self._unresolved(note_metadata, version, steps, [str(hold)])
        steps.append(ResolutionStep(step="encounter",
                                    identifier=encounter_id or document_id,
                                    resolved_to=key, outcome="resolved"))

        dos, service_date = bind_service_date(
            declared=entry.get("date_of_service"), caller=date_of_service,
            evidence=document_service_date, holds=holds, conflicts=conflicts,
            steps=steps)

        # THE CHAIN, NOT SEVEN INDEPENDENT LOOKUPS (issue #6 F7-R2)
        # --------------------------------------------------------
        # Each branch below is independent in its FAILURE (one unresolvable
        # coverage must not hide an expired affiliation), but NOT in what it is
        # allowed to resolve against. Every downstream branch is handed the
        # identity the previous branch actually resolved -- the patient the
        # coverage must belong to, the provider the affiliation must belong to,
        # the coverage/provider/facility the authorization must have been issued
        # for -- so two records that each resolve perfectly well but describe two
        # different people can never be combined into one claim.
        branch = dict(steps=steps, holds=holds)
        patient, patient_id = self._branch(
            self._resolve_patient, raw, entry,
            default=(PatientIdentity(), ""), **branch)
        subscriber, payer, coverage = self._branch(
            self._resolve_coverage, raw, entry, patient_id, dos,
            default=(SubscriberIdentity(), PayerIdentity(), CoverageBinding()),
            **branch)
        provider = self._branch(self._resolve_participant, raw, entry,
                                default=ProviderIdentity(), **branch)
        affiliation, billing_entity = self._branch(
            self._resolve_affiliation, raw, provider, dos,
            default=(AffiliationBinding(), BillingEntityIdentity()), **branch)
        facility, place_of_service, jurisdiction, facility_id = self._branch(
            self._resolve_facility, raw, entry,
            default=(FacilityIdentity(), "", "", ""), **branch)
        coverage = self._branch(
            self._resolve_authorization, raw, entry, dos, coverage, provider,
            facility_id, default=coverage, **branch)
        subscriber = subscriber.model_copy(update={
            "authorization_number": coverage.authorization_number})

        context = EncounterContext(
            resolution=ContextResolution.RESOLVED,
            provider_id=self.provider_id,
            context_version=version,
            service_date=service_date,
            patient=patient,
            subscriber=subscriber,
            payer=payer,
            rendering_provider=provider,
            service_facility=facility,
            place_of_service=place_of_service,
            jurisdiction=jurisdiction,
            billing_entity=billing_entity,
            affiliation=affiliation,
            coverage=coverage,
            resolution_steps=tuple(steps),
        )
        conflicts.extend(self._conflicts(context, note_metadata))
        resolution = ContextResolution.RESOLVED
        if conflicts:
            resolution = ContextResolution.CONFLICT
        elif holds or context.missing_required():
            resolution = ContextResolution.UNRESOLVED
        context = context.model_copy(update={
            "resolution": resolution,
            "unresolved": tuple(dict.fromkeys(holds)),
            "conflicts": tuple(dict.fromkeys(conflicts)),
        })
        return _stamp(context, source_label=AUTHORITATIVE_FIELD_SOURCE)

    # ------------------------------------------------------------- branches
    @staticmethod
    def _branch(fn, *call_args, steps: list[ResolutionStep],
                holds: list[str], default):
        """Run one resolution branch; a `_Hold` becomes a named hold, not a raise.

        Branches are independent on purpose: an unresolvable coverage must not
        hide an expired affiliation, because an operator who fixes the first
        would otherwise re-run the batch only to discover the second.

        `steps`/`holds` are keyword-only so a new branch cannot silently bind
        them to a positional argument that happens to be in the right place.
        """
        try:
            return fn(*call_args, steps)
        except _Hold as hold:
            holds.append(str(hold))
            return default

    def _encounter_entry(self, raw: dict[str, Any], encounter_id: str,
                         document_id: str) -> tuple[dict[str, Any], str]:
        """The encounter record, by exact identifier. Encounter id, then document id.

        When both identifiers are supplied AND both match DIFFERENT records,
        the request is ambiguous and holds: preferring one silently would let a
        mis-keyed source bill this encounter under another encounter's context.
        """
        encounters = raw.get("encounters") or {}
        matches = []
        for key in (encounter_id, document_id):
            if key and isinstance(encounters.get(key), dict) and \
                    key not in {k for k, _ in matches}:
                matches.append((key, encounters[key]))
        if not matches:
            raise _Hold(f"encounter {encounter_id or document_id!r} is not in "
                        f"the encounter context source")
        for key, _ in matches:
            if key in self._duplicate_keys.get("encounters", ()):
                raise _Hold(
                    f"encounter identifier {key!r} is declared more than once "
                    f"in the context source; it does not identify one encounter")
        if len(matches) > 1 and matches[0][1] != matches[1][1]:
            raise _Hold(
                f"encounter id {encounter_id!r} and document id {document_id!r} "
                f"resolve to two different encounter records")
        return matches[0][1], matches[0][0]

    def _resolve_patient(self, raw: dict[str, Any], entry: dict[str, Any],
                         steps: list[ResolutionStep]
                         ) -> tuple[PatientIdentity, str]:
        """encounter -> patient id -> patient identity, and the id it resolved under.

        The identifier is returned alongside the identity, and stamped ONTO it,
        for the same reason `_resolve_participant` overwrites the provider's own
        `npi` with its key: it is what every later branch checks its own record
        against. A patient identity that cannot say which patient it is cannot be
        used to prove that this encounter's coverage belongs to this encounter's
        patient. (Issue #6 F7-R2.)
        """
        patient_id = str(entry.get("patient_id") or "").strip()
        record = self._by_key(raw, "patients", patient_id, "patient")
        steps.append(ResolutionStep(step="patient", identifier=patient_id,
                                    resolved_to=patient_id, outcome="resolved"))
        return (PatientIdentity(**{**_section(record, PatientIdentity),
                                   "patient_id": patient_id}),
                patient_id)

    def _resolve_coverage(self, raw: dict[str, Any], entry: dict[str, Any],
                          patient_id: str, dos: date | None,
                          steps: list[ResolutionStep]
                          ) -> tuple[SubscriberIdentity, PayerIdentity,
                                     CoverageBinding]:
        """encounter -> coverage -> subscriber + payer, FOR THIS PATIENT, on the DOS.

        An explicit `coverage_id` selects the record. Without one, the patient's
        coverages are selected by identifier and narrowed BY DATE — and the
        result must be UNIQUE. Two coverages active on the same date is a real
        situation (primary/secondary) that this contract cannot express, so it
        holds rather than picking the first row.

        EVERY path ends at the same check (issue #6 F7-R2): the coverage that
        binds to this claim must be the RESOLVED PATIENT'S coverage. An explicit
        `coverage_id` used to be matched on the identifier and its effective
        window alone — so an encounter naming patient P1 and an active coverage
        belonging to P2 resolved cleanly, with P1's demographics and P2's member
        id on one professional claim, no holds, and nothing downstream able to
        tell. Identity and coverage are resolved by two branches; that is exactly
        why the join between them has to be asserted here rather than assumed.
        """
        if dos is None:
            raise _Hold("coverage cannot be resolved without a date of service")
        if not patient_id:
            raise _Hold("this encounter's patient did not resolve, so no "
                        "coverage can be bound to it; a claim must not carry a "
                        "coverage that belongs to nobody it has identified")
        rows = self._rows(raw, "coverages")
        coverage_id = str(entry.get("coverage_id") or "").strip()
        if coverage_id:
            candidates = [r for r in rows
                          if str(r.get("coverage_id") or "") == coverage_id]
            if not candidates:
                raise _Hold(f"coverage {coverage_id!r} is not in the encounter "
                            f"context source")
            self._reject_duplicate_row(coverage_id, "coverages", "coverage")
            self._reject_foreign_coverage(candidates, patient_id)
            active = [r for r in candidates if _covers(dos, *_window(
                r, f"coverage {coverage_id!r}"))]
            if not active:
                raise _Hold(
                    f"coverage {coverage_id!r} is not in force on the date of "
                    f"service {dos.isoformat()} "
                    f"({'; '.join(_describe(r) for r in candidates)})")
        else:
            candidates = [r for r in rows
                          if str(r.get("patient_id") or "") == patient_id]
            if not candidates:
                raise _Hold(f"no coverage record names patient {patient_id!r}; "
                            f"the encounter declares no coverage_id either")
            for row in candidates:
                self._reject_duplicate_row(str(row.get("coverage_id") or ""),
                                           "coverages", "coverage")
            active = [r for r in candidates if _covers(dos, *_window(
                r, f"coverage {str(r.get('coverage_id') or '?')!r}"))]
            if not active:
                raise _Hold(
                    f"no coverage for patient {patient_id!r} is in force on the "
                    f"date of service {dos.isoformat()} "
                    f"({'; '.join(_describe(r) for r in candidates)})")
            if len(active) > 1:
                raise _Hold(
                    f"patient {patient_id!r} has "
                    f"{len(active)} coverages in force on {dos.isoformat()} "
                    f"({', '.join(sorted(str(r.get('coverage_id') or '?') for r in active))}); "
                    f"the encounter must declare which one it is billed under")
        row = active[0]
        resolved_id = str(row.get("coverage_id") or "")
        self._reject_duplicate_row(resolved_id, "coverages", "coverage")
        # Re-asserted on the row that actually binds, not only on the candidate
        # set: the two selection paths above are the kind of code a later change
        # adds a third branch to, and this is the invariant all of them owe.
        self._reject_foreign_coverage([row], patient_id)
        payer_id = str(row.get("payer_id") or "").strip()
        payer_record = self._by_key(raw, "payers", payer_id, "payer")
        steps.append(ResolutionStep(step="coverage", identifier=resolved_id,
                                    resolved_to=payer_id, outcome="resolved"))
        subscriber = SubscriberIdentity(
            member_id=str(row.get("member_id") or ""),
            group_number=str(row.get("group_number") or ""),
            relationship_to_patient=str(row.get("relationship_to_patient") or ""),
        )
        payer = PayerIdentity(**_section(payer_record, PayerIdentity))
        binding = CoverageBinding(
            coverage_id=resolved_id,
            # Carried onto the claim so the referential check above is
            # REPRODUCIBLE from the artifact by a consumer that never saw the
            # roster (`EncounterContext.problems()` re-derives it).
            patient_id=str(row.get("patient_id") or ""),
            payer_id=payer_id,
            effective_start=str(row.get("effective_start") or ""),
            effective_end=str(row.get("effective_end") or ""),
        )
        return subscriber, payer, binding

    @staticmethod
    def _reject_foreign_coverage(rows: list[dict[str, Any]],
                                 patient_id: str) -> None:
        """Refuse any coverage record that is not THIS patient's. (Issue #6 F7-R2.)

        A coverage naming NO patient is refused for the same reason as one naming
        a different patient: the claim would assert a subscriber relationship the
        source never states. The hold is raised inside the coverage branch, so it
        is this ENCOUNTER's hold — every other encounter in the same batch is
        unaffected, which is the whole failure-boundary contract of this module.
        """
        for row in rows:
            owner = str(row.get("patient_id") or "").strip()
            if owner == patient_id:
                continue
            identifier = str(row.get("coverage_id") or "?")
            raise _Hold(
                f"coverage {identifier!r} belongs to patient "
                f"{owner or '<none declared>'}, not to this encounter's patient "
                f"{patient_id!r}; a claim must not carry one patient's "
                f"demographics with another patient's coverage and member id")

    def _resolve_participant(self, raw: dict[str, Any], entry: dict[str, Any],
                             steps: list[ResolutionStep]) -> ProviderIdentity:
        """signed/rendering provider NPI -> participant.

        The NPI is the identifier and the key: the participant's own `npi`
        field is overwritten with the key it was found under, so a record whose
        body disagrees with its key can never contribute an NPI to a claim.
        """
        npi = str(entry.get("rendering_provider_npi") or "").strip()
        record = self._by_key(raw, "providers", npi, "rendering provider")
        steps.append(ResolutionStep(step="rendering_provider", identifier=npi,
                                    resolved_to=npi, outcome="resolved"))
        return ProviderIdentity(**{**_section(record, ProviderIdentity),
                                   "npi": npi})

    def _resolve_affiliation(self, raw: dict[str, Any],
                             provider: ProviderIdentity, dos: date | None,
                             steps: list[ResolutionStep]
                             ) -> tuple[AffiliationBinding,
                                        BillingEntityIdentity]:
        """participant -> billing entity and affiliation FOR THE DATE OF SERVICE.

        The affiliation in force must be UNIQUE. Two active rows naming two
        entities is genuinely ambiguous; two active rows naming the SAME entity
        is also refused, because the bound `affiliation_id` is fingerprinted
        into the claim and picking either would make the fingerprint depend on
        row order rather than on the facts.
        """
        npi = provider.npi
        if not npi:
            raise _Hold("the rendering provider did not resolve, so no billing "
                        "affiliation can be established for this encounter")
        if dos is None:
            raise _Hold("the provider's billing affiliation cannot be resolved "
                        "without a date of service")
        rows = [r for r in self._rows(raw, "affiliations")
                if str(r.get("provider_npi") or "") == npi]
        if not rows:
            raise _Hold(f"no affiliation record ties rendering provider {npi!r} "
                        f"to a billing entity")
        for row in rows:
            self._reject_duplicate_row(str(row.get("affiliation_id") or ""),
                                       "affiliations", "affiliation")
        active = [r for r in rows
                  if _covers(dos, *_window(r, f"affiliation "
                                              f"{str(r.get('affiliation_id') or '?')!r}"))]
        if not active:
            raise _Hold(
                f"rendering provider {npi!r} has no billing affiliation in "
                f"force on the date of service {dos.isoformat()} "
                f"({'; '.join(_describe(r) for r in rows)})")
        entities = sorted({str(r.get("billing_entity_id") or "") for r in active})
        if len(entities) > 1:
            raise _Hold(
                f"rendering provider {npi!r} is affiliated with "
                f"{len(entities)} billing entities on {dos.isoformat()} "
                f"({', '.join(entities)}); the encounter cannot be billed "
                f"under one of them by preference")
        if len(active) > 1:
            raise _Hold(
                f"rendering provider {npi!r} has "
                f"{len(active)} overlapping affiliation records in force on "
                f"{dos.isoformat()}; which record authorizes this claim is not "
                f"determined by the source")
        row = active[0]
        affiliation_id = str(row.get("affiliation_id") or "")
        self._reject_duplicate_row(affiliation_id, "affiliations", "affiliation")
        entity_id = entities[0]
        entity_record = self._by_key(raw, "billing_entities", entity_id,
                                     "billing entity")
        steps.append(ResolutionStep(step="affiliation", identifier=affiliation_id,
                                    resolved_to=entity_id,
                                    outcome=f"in force {_describe(row)}"))
        entity = BillingEntityIdentity(
            **{**_section(entity_record, BillingEntityIdentity),
               "entity_id": entity_id})
        affiliation = AffiliationBinding(
            affiliation_id=affiliation_id, provider_npi=npi,
            billing_entity_id=entity_id,
            effective_start=str(row.get("effective_start") or ""),
            effective_end=str(row.get("effective_end") or ""),
        )
        return affiliation, entity

    def _resolve_facility(self, raw: dict[str, Any], entry: dict[str, Any],
                          steps: list[ResolutionStep]
                          ) -> tuple[FacilityIdentity, str, str, str]:
        """facility id -> facility identity, place of service and jurisdiction.

        Both are DECLARED by the facility record, not derived from its address.
        Deriving the coverage jurisdiction from a state would be a proxy for a
        real lookup (`data/codes/mac_jurisdictions.json`); until that lookup is
        wired, the source states it explicitly and is auditable for it.
        """
        facility_id = str(entry.get("facility_id") or "").strip()
        record = self._by_key(raw, "facilities", facility_id, "service facility")
        facility = FacilityIdentity(**_section(record, FacilityIdentity))
        place_of_service = str(record.get("place_of_service") or "").strip()
        jurisdiction = str(record.get("jurisdiction") or "").strip()
        if not place_of_service:
            raise _Hold(f"service facility {facility_id!r} declares no place of "
                        f"service")
        if not jurisdiction:
            raise _Hold(f"service facility {facility_id!r} declares no coverage "
                        f"jurisdiction")
        steps.append(ResolutionStep(step="facility", identifier=facility_id,
                                    resolved_to=place_of_service,
                                    outcome="resolved"))
        # The RESOLVED identifier travels with the identity for the same reason
        # the patient's does: the authorization branch must check itself against
        # the facility this encounter actually resolved, not against whatever the
        # encounter record claimed before anyone looked it up. (Issue #6 F7-R2.)
        return facility, place_of_service, jurisdiction, facility_id

    def _resolve_authorization(self, raw: dict[str, Any], entry: dict[str, Any],
                               dos: date | None, coverage: CoverageBinding,
                               provider: ProviderIdentity,
                               facility_id: Any, steps: list[ResolutionStep]
                               ) -> CoverageBinding:
        """The prior authorization, ONLY if it authorizes THIS encounter.

        An authorization is issued for a specific coverage, rendering provider
        and facility, for a specific window. If any of those changed, it does
        not authorize this claim — and carrying its number onto the claim
        anyway is exactly the silent staleness directive §2 names. A mismatch
        HOLDS rather than quietly dropping the number: an operator who recorded
        an authorization believed one was required.

        `facility_id` is the identifier the FACILITY BRANCH RESOLVED, not the one
        the encounter record declared. Checking against the declared value would
        let an authorization "match" a facility that is not in the source at all
        — the same combine-two-independent-resolutions-without-a-cross-check
        defect as F7-R2, one branch over.
        """
        authorization_id = str(entry.get("authorization_id") or "").strip()
        if not authorization_id:
            return coverage
        rows = [r for r in self._rows(raw, "authorizations")
                if str(r.get("authorization_id") or "") == authorization_id]
        if not rows:
            raise _Hold(f"authorization {authorization_id!r} is not in the "
                        f"encounter context source")
        self._reject_duplicate_row(authorization_id, "authorizations",
                                   "authorization")
        row = rows[0]
        label = f"authorization {authorization_id!r}"
        mismatches = []
        for field, expected, described in (
            ("coverage_id", coverage.coverage_id, "coverage"),
            ("rendering_provider_npi", provider.npi, "rendering provider"),
            ("facility_id", str(facility_id or ""), "service facility"),
        ):
            declared = str(row.get(field) or "")
            if declared != expected:
                mismatches.append(
                    f"{described} {declared!r} (this encounter's is "
                    f"{expected!r})")
        if mismatches:
            raise _Hold(f"{label} was issued for {'; '.join(mismatches)}; it "
                        f"does not authorize this encounter")
        if dos is None:
            raise _Hold(f"{label} cannot be checked without a date of service")
        start, end = _window(row, label)
        if not _covers(dos, start, end):
            raise _Hold(f"{label} is not in force on the date of service "
                        f"{dos.isoformat()} ({_describe(row)})")
        steps.append(ResolutionStep(step="authorization",
                                    identifier=authorization_id,
                                    resolved_to=coverage.coverage_id,
                                    outcome=f"in force {_describe(row)}"))
        return coverage.model_copy(update={
            "authorization_id": authorization_id,
            "authorization_number": str(row.get("authorization_number") or ""),
            "authorization_effective_start": str(row.get("effective_start") or ""),
            "authorization_effective_end": str(row.get("effective_end") or ""),
        })

    # ---------------------------------------------------------------- lookup
    def _by_key(self, raw: dict[str, Any], section: str, key: str,
                described: str) -> dict[str, Any]:
        if not key:
            raise _Hold(f"the encounter record declares no {described} identifier")
        if key in self._duplicate_keys.get(section, ()):
            raise _Hold(f"{described} identifier {key!r} is declared more than "
                        f"once in the context source; it does not identify one "
                        f"{described}")
        record = (raw.get(section) or {}).get(key)
        if not isinstance(record, dict):
            raise _Hold(f"{described} {key!r} is not in the encounter context "
                        f"source")
        return record

    @staticmethod
    def _rows(raw: dict[str, Any], section: str) -> list[dict[str, Any]]:
        return [r for r in (raw.get(section) or []) if isinstance(r, dict)]

    def _reject_duplicate_row(self, identifier: str, section: str,
                              described: str) -> None:
        if identifier and identifier in self._duplicate_rows.get(section, ()):
            raise _Hold(f"{described} identifier {identifier!r} appears on more "
                        f"than one record in the context source; it does not "
                        f"identify one {described}")

    # ----------------------------------------------------------- corroboration
    def _unresolved(self, note_metadata: dict[str, Any] | None, version: str,
                    steps: list[ResolutionStep], holds: list[str]
                    ) -> EncounterContext:
        """Hold THIS encounter, name why, and keep the note's own metadata.

        The corroborating metadata still travels with the bundle (it is what a
        human needs to fix the source), and it is still labelled
        `note_corroboration`, so it cannot be mistaken for resolved context by
        anything downstream.
        """
        base = context_from_note_metadata(note_metadata)
        context = base.model_copy(update={
            "resolution": ContextResolution.UNRESOLVED,
            "provider_id": self.provider_id,
            "context_version": version,
            "resolution_steps": tuple(steps),
            "unresolved": tuple(dict.fromkeys(holds)),
            # No encounter record resolved, so no date of service is bound to
            # this encounter -- deliberately NOT the caller's assertion, which is
            # what an artifact would otherwise carry as if it had been checked.
            "service_date": ServiceDateBinding(),
        })
        return _stamp(context, source_label=CORROBORATION_FIELD_SOURCE)

    def _conflicts(self, context: EncounterContext,
                   note_metadata: dict[str, Any] | None) -> tuple[str, ...]:
        """Identity disagreements between the resolved context and the note.

        Only compared when BOTH sides state a value: the note not mentioning a
        member id is not a disagreement, and treating it as one would hold
        every encounter whose document is simply less detailed than the source.
        The note's value is never adopted — it can only agree or object.
        """
        meta = dict(note_metadata or {})
        out: list[str] = []
        for meta_keys, path in _CORROBORATED:
            raw_value = next((meta.get(k) for k in meta_keys if meta.get(k)), None)
            documented = _normalized(raw_value)
            resolved = _normalized(context.field_value(path))
            if documented and resolved and documented != resolved:
                out.append(
                    f"{path}: encounter context says "
                    f"{context.field_value(path)!r}, the document says "
                    f"{str(raw_value)!r}")
        note_first, note_last = _split_name(meta.get("patient_name") or "")
        if note_last and context.patient.last_name and \
                _normalized(note_last) != _normalized(context.patient.last_name):
            out.append(
                f"patient.last_name: encounter context says "
                f"{context.patient.last_name!r}, the document says {note_last!r}")
        return tuple(out)


def _duplicate_row_ids(rows: Any, id_field: str) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        identifier = str(row.get(id_field) or "")
        if not identifier:
            continue
        if identifier in seen:
            duplicates.add(identifier)
        seen.add(identifier)
    return tuple(sorted(duplicates))


# --------------------------------------------------------------------------
# adapter selection — configuration, not code
# --------------------------------------------------------------------------

#: `name -> factory(locator)`. An EHR/FHIR, HL7-export, practice-management or
#: scheduling adapter is added by implementing `EncounterContextProvider` and
#: registering it here; NOTHING in `build_provider`, `run.py` or the deployment
#: configuration changes to accept it. That is the whole point of the registry:
#: this project has no live EHR to integrate against or test against, so
#: building a speculative one now would ship an unverifiable code path — but a
#: deployment that acquires one must not need a rewrite to use it.
_ADAPTERS: dict[str, Callable[[str], Any]] = {}

#: A bare path in the configuration means the versioned local roster — the one
#: source a practice controls without an integration project.
DEFAULT_ADAPTER = "versioned_roster"


def register_adapter(name: str, factory: Callable[[str], Any]) -> None:
    if name in _ADAPTERS:
        raise ValueError(f"encounter context adapter {name!r} is already "
                         f"registered")
    _ADAPTERS[name] = factory


def registered_adapters() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


register_adapter(DEFAULT_ADAPTER, VersionedRosterContextProvider)


def build_provider(source_spec: str | None) -> EncounterContextProvider:
    """The provider a deployment gets for its configuration.

    `source_spec` is either a bare path (the versioned local roster) or
    `<adapter>:<locator>` for any registered adapter. Nothing configured ->
    the note-metadata provider, i.e. every encounter holds. That is the current
    deployed reality, stated as a typed decision instead of as an absence.

    An unknown adapter NAME is refused rather than being retried as a file
    path: `fhir:https://...` typed with no fhir adapter registered would
    otherwise become "file not found", which reads as a missing roster and
    sends an operator to fix the wrong thing.
    """
    spec = str(source_spec or "").strip()
    if not spec:
        return NoteMetadataContextProvider()
    name, separator, locator = spec.partition(":")
    if separator and name.isidentifier():
        if name not in _ADAPTERS:
            raise EncounterContextUnavailable(
                f"no encounter context adapter named {name!r} is registered "
                f"(registered: {', '.join(registered_adapters())})")
        if not locator.strip():
            raise EncounterContextUnavailable(
                f"encounter context adapter {name!r} was given no source "
                f"locator")
        return _ADAPTERS[name](locator.strip())
    return _ADAPTERS[DEFAULT_ADAPTER](spec)
