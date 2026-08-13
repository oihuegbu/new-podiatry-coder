"""`EncounterContextProvider` — where a claim's billing context comes from.

================================================================================
WHY THIS MODULE EXISTS — issue #6 F6-R4-A1 (second part), product directive §2
================================================================================
A professional claim needs facts the clinical note is not the authority for:
who the subscriber is, which payer covers them, which rendering provider's NPI
signs the claim, which place of service applies. The deployed system previously
took whichever of those a vision model happened to read off a letterhead, or
took nothing at all — and "nothing at all" arrived downstream as empty strings
that looked exactly like "the note names no payer".

This module makes the distinction structural:

  * `EncounterContextProvider` is the interface directive §2 will grow
    EHR/FHIR, practice-management and scheduling adapters behind. Resolution is
    BY STABLE IDENTIFIER (encounter/document id, provider NPI), never by model
    inference over a broad roster.
  * `NoteMetadataContextProvider` is the honest no-authority default. It
    carries what the note said as CORROBORATION and returns
    `ContextResolution.UNRESOLVED`, so every encounter holds — fail-safe, and
    visibly so, rather than silently.
  * `VersionedRosterContextProvider` reads a versioned, checked-in context file
    ("a versioned local roster import", directive §2) keyed by encounter or
    document id. It resolves only when the entry exists AND every required
    field is present; otherwise it names what is missing.

Corroboration is not authority, but a DISAGREEMENT is still a signal: when a
roster resolves an encounter and the note's own metadata contradicts it on an
identity field, the context is `CONFLICT` and the encounter holds. Two sources
that disagree about who the patient is must never be reconciled by preferring
one silently.

No medical codes appear here. Place of service is carried as an opaque value
resolved by the context source and validated downstream against the
authoritative CMS set; this module never enumerates one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.contracts.claim_bundle import (
    ContextResolution, EncounterContext, FacilityIdentity, PatientIdentity,
    PayerIdentity, ProviderIdentity, SubscriberIdentity,
)


class EncounterContextUnavailable(Exception):
    """The configured context source could not be read or is malformed.

    Typed, and never degraded into an empty context: an unreadable roster and a
    roster that says nothing about this encounter are different facts, and only
    one of them is a data-integrity incident.
    """


@runtime_checkable
class EncounterContextProvider(Protocol):
    """Resolve one encounter's billing context by stable identifiers."""

    provider_id: str

    def resolve(self, *, encounter_id: str, document_id: str,
                date_of_service: str | None,
                note_metadata: dict[str, Any] | None = None
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


def context_from_note_metadata(metadata: dict[str, Any] | None
                               ) -> EncounterContext:
    """Everything the note itself stated, as CORROBORATION only.

    Always `UNRESOLVED`. This function exists so that note-derived context is
    carried into the bundle for the next phase and for a human — the finding
    was partly that `run.py` obtained patient metadata and discarded it — while
    remaining structurally incapable of authorizing a release.
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
    return context.model_copy(update={"fingerprint": context.compute_fingerprint()})


class NoteMetadataContextProvider:
    """The default when no authoritative context source is configured.

    Returns `UNRESOLVED` for every encounter, by construction. This is a
    deliberate zero-autonomy state, not a degraded one: the deployment holds
    every note until directive §2 provisions a real adapter.
    """

    provider_id = "note_metadata_corroboration/1"

    def resolve(self, *, encounter_id: str, document_id: str,
                date_of_service: str | None,
                note_metadata: dict[str, Any] | None = None
                ) -> EncounterContext:
        return context_from_note_metadata(note_metadata)


# --------------------------------------------------------------------------
# versioned roster adapter
# --------------------------------------------------------------------------

#: Note-metadata fields that CORROBORATE a resolved identity, mapped to the
#: resolved context path they must agree with. Identity fields only: a
#: disagreement about the patient's name, birth date or member id means the
#: roster entry and the document are not describing the same encounter.
_CORROBORATED: tuple[tuple[str, str], ...] = (
    ("date_of_birth", "patient.date_of_birth"),
    ("member_id", "subscriber.member_id"),
)


def _normalized(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


class VersionedRosterContextProvider:
    """Encounter context from a versioned, checked-in context file.

    File shape (validated strictly — a malformed file is
    `EncounterContextUnavailable`, never an empty roster):

        {
          "schema": "encounter_context/1",
          "version": "<edition identifier>",
          "encounters": {
            "<encounter or document id>": {
              "patient": {...}, "subscriber": {...}, "payer": {...},
              "rendering_provider": {...}, "service_facility": {...},
              "place_of_service": "...", "jurisdiction": "..."
            }
          }
        }

    Resolution is by exact identifier lookup — encounter id first, then
    document id. There is no fuzzy match and no "closest" entry: selecting a
    provider identity by similarity is exactly what directive §2 forbids.
    """

    provider_id = "versioned_roster/1"
    schema = "encounter_context/1"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._loaded: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._loaded is not None:
            return self._loaded
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            raise EncounterContextUnavailable(
                f"encounter context file {self.path} could not be read: "
                f"{exc}") from None
        if not isinstance(raw, dict):
            raise EncounterContextUnavailable(
                f"encounter context file {self.path} must contain an object")
        if raw.get("schema") != self.schema:
            raise EncounterContextUnavailable(
                f"encounter context file {self.path} declares schema "
                f"{raw.get('schema')!r}, not {self.schema!r}")
        if not isinstance(raw.get("encounters"), dict):
            raise EncounterContextUnavailable(
                f"encounter context file {self.path} has no 'encounters' object")
        if not str(raw.get("version") or "").strip():
            raise EncounterContextUnavailable(
                f"encounter context file {self.path} declares no version; a "
                f"context edition must be identifiable to be fingerprinted")
        self._loaded = raw
        return raw

    def resolve(self, *, encounter_id: str, document_id: str,
                date_of_service: str | None,
                note_metadata: dict[str, Any] | None = None
                ) -> EncounterContext:
        roster = self._load()
        entry = None
        for key in (encounter_id, document_id):
            if key and isinstance(roster["encounters"].get(key), dict):
                entry = roster["encounters"][key]
                break
        version = str(roster.get("version") or "")
        if entry is None:
            # Unknown encounter: hold THIS encounter only, and say why. The
            # corroborating note metadata still travels with the bundle.
            base = context_from_note_metadata(note_metadata)
            context = base.model_copy(update={
                "resolution": ContextResolution.UNRESOLVED,
                "provider_id": self.provider_id,
                "context_version": version,
                "unresolved": (f"encounter '{encounter_id or document_id}' is "
                               f"not in the encounter context source",),
            })
            return context.model_copy(
                update={"fingerprint": context.compute_fingerprint()})

        context = EncounterContext(
            resolution=ContextResolution.RESOLVED,
            provider_id=self.provider_id,
            context_version=version,
            patient=PatientIdentity(**_section(entry, "patient",
                                               PatientIdentity)),
            subscriber=SubscriberIdentity(**_section(entry, "subscriber",
                                                     SubscriberIdentity)),
            payer=PayerIdentity(**_section(entry, "payer", PayerIdentity)),
            rendering_provider=ProviderIdentity(
                **_section(entry, "rendering_provider", ProviderIdentity)),
            service_facility=FacilityIdentity(
                **_section(entry, "service_facility", FacilityIdentity)),
            place_of_service=str(entry.get("place_of_service") or ""),
            jurisdiction=str(entry.get("jurisdiction") or ""),
        )
        missing = context.missing_required()
        conflicts = self._conflicts(context, note_metadata)
        resolution = ContextResolution.RESOLVED
        if conflicts:
            resolution = ContextResolution.CONFLICT
        elif missing:
            resolution = ContextResolution.UNRESOLVED
        context = context.model_copy(update={
            "resolution": resolution,
            "unresolved": missing,
            "conflicts": conflicts,
        })
        return context.model_copy(
            update={"fingerprint": context.compute_fingerprint()})

    def _conflicts(self, context: EncounterContext,
                   note_metadata: dict[str, Any] | None) -> tuple[str, ...]:
        """Identity disagreements between the resolved context and the note.

        Only compared when BOTH sides state a value: the note not mentioning a
        member id is not a disagreement, and treating it as one would hold
        every encounter whose document is simply less detailed than the roster.
        """
        meta = dict(note_metadata or {})
        out: list[str] = []
        for meta_key, path in _CORROBORATED:
            documented = _normalized(meta.get(meta_key))
            resolved = _normalized(context.field_value(path))
            if documented and resolved and documented != resolved:
                out.append(
                    f"{path}: encounter context says {context.field_value(path)!r}, "
                    f"the document says {str(meta.get(meta_key))!r}")
        note_first, note_last = _split_name(meta.get("patient_name") or "")
        if note_last and context.patient.last_name and \
                _normalized(note_last) != _normalized(context.patient.last_name):
            out.append(
                f"patient.last_name: encounter context says "
                f"{context.patient.last_name!r}, the document says {note_last!r}")
        return tuple(out)


def _section(entry: dict[str, Any], key: str, model) -> dict[str, Any]:
    """One roster section, restricted to the fields the model declares.

    An unknown key is DROPPED rather than passed through: the roster is
    operator-maintained data, and a typo there must not raise a validation
    error that holds every encounter in the file. The corresponding required
    field then simply stays empty and is reported by `missing_required()` —
    which names the field the operator meant to set.
    """
    section = entry.get(key) or {}
    if not isinstance(section, dict):
        return {}
    allowed = set(model.model_fields)
    return {k: str(v or "") for k, v in section.items() if k in allowed}


def build_provider(context_file: str | None) -> EncounterContextProvider:
    """The provider a deployment gets for its configuration.

    No file configured -> the note-metadata provider, i.e. every encounter
    holds. That is the current deployed reality, stated as a typed decision
    instead of as an absence.
    """
    if context_file:
        return VersionedRosterContextProvider(context_file)
    return NoteMetadataContextProvider()
