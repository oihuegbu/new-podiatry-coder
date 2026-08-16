"""Encounter-context resolution, proved through the DEPLOYED entrypoint.

================================================================================
WHAT THIS PROVES — issue #6, product directive §2
================================================================================
The shortfall the directive names: "One optional global JSON roster is passed to
every note. The deployed environment supplies none, so every encounter holds
before retrieval. That is fail-safe but has zero autonomous throughput."

Phase 1 seeded the seam. What it could not do was RESOLVE anything: its context
file was one inline dump per encounter, so "which billing entity did this
provider bill under on this date of service?" had no representation and could
only be answered by whatever an operator pasted into that encounter's block.
This module drives the four acceptance tests the directive states, through
`run.main()` — the exact Python target of `docker compose run app python run.py`
— against the real contract, the real registry and the real 837P builder:

  1. a fully documented synthetic encounter reaches AUTO_READY with no human
     input, resolved along the identifier chain;
  2. unknown, duplicate, expired or conflicting affiliations hold ONLY the
     affected encounter — proved with a TWO-note batch in which the other note
     must still reach AUTO_READY from the same source file;
  3. a different provider/facility/payer changes the context fingerprint and
     invalidates a stale authorization;
  4. a clean deployment provisions the configured adapter automatically —
     configuration only, no developer wiring, and the container can actually
     see the file.

Everything real/substituted is exactly as documented in
`tests/test_claim_bundle_e2e.py`, whose deployment fixture this module reuses so
the positive and negative cases cannot drift apart. NO medical code, payer, POS
value or NPI is hardcoded: all are selected at runtime from the authoritative
data, and every NPI is computed with a valid CMS check digit.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run as entrypoint  # noqa: E402  — the module the deployment executes
from app.contracts.claim_bundle import load_bundle  # noqa: E402
from tools import claim_submitter as cs  # noqa: E402
from tests.source_pdf import build_pdf, vision_extraction  # noqa: E402
from tools import claims_registry as reg  # noqa: E402

# The deployment fixture and its roster identifiers, imported (not copied) so a
# change to the canonical encounter changes both suites at once. `deployment`
# does not start with `test_`, so pytest treats it as the fixture it is.
from tests.test_claim_bundle_e2e import (  # noqa: E402
    AFFILIATION_ID, COVERAGE_ID, DOCUMENT_VERSION, DOS_ISO, ENTITY_ID,
    EXTRACTED_TEXT_SHA, FACILITY_ID, NOTE_TEXT, PATIENT_ID, PAYER_ID,
    ROSTER_START, STEM, _valid_npi, _verified_once, deployment,
)

#: A second note in the SAME batch, reading the same source file. Its only job
#: is to prove that one broken encounter is one held encounter.
OTHER_STEM = "NOTE_BUNDLE_E2E_002"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _run(deployment, context_file=None) -> int:
    return entrypoint.main([
        "--billing-context", str(deployment.billing_context_file),
        "--encounter-context", str(context_file or deployment.context_file)])


def _bundle(deployment, stem):
    return load_bundle(json.loads(
        (deployment.output_dir / f"{stem}_results.json").read_text()))


def _add_second_note(deployment, *, npi: str) -> None:
    """A second note, with its OWN provider and affiliation.

    Its own provider deliberately: an encounter that shared the first note's
    provider would break BOTH notes when its affiliation broke, which proves
    nothing about isolation. Here the two encounters are independent, so a
    failure that reaches the wrong one is visible.
    """
    (deployment.tmp_path / "attachments" / f"{OTHER_STEM}.pdf").write_bytes(
        build_pdf([[NOTE_TEXT]]))
    roster = deployment.roster
    roster["providers"][npi] = {"first_name": "Sam", "last_name": "Okonkwo",
                                "display_name": "Sam Okonkwo"}
    roster["affiliations"].append({
        "affiliation_id": "AFF-E2E-2", "provider_npi": npi,
        "billing_entity_id": ENTITY_ID,
        "effective_start": ROSTER_START, "effective_end": ""})
    roster["encounters"][OTHER_STEM] = {
        "patient_id": PATIENT_ID, "coverage_id": COVERAGE_ID,
        "rendering_provider_npi": npi, "facility_id": FACILITY_ID}


def _other_payer_alias(current: str) -> str:
    from app.compliance.payer_registry import _load_payers
    for payer in _load_payers():
        if not (payer.get("stedi_trading_partner_id") and payer.get("aliases")):
            continue
        alias = max(payer["aliases"], key=len)
        if alias != current:
            return alias
    raise AssertionError("the payer registry declares only one routable payer")


# ==========================================================================
# ACCEPTANCE 1 — a fully documented encounter reaches AUTO_READY, no human
# ==========================================================================

def test_a_fully_documented_encounter_reaches_auto_ready_with_no_human_input(
        deployment):
    """The directive's first acceptance test, with the CHAIN asserted.

    Reaching AUTO_READY is not enough on its own: it would also be reached by a
    resolver that inlined whatever an operator pasted per encounter. What makes
    the claim defensible is that every identity was reached by an exact
    identifier lookup and that the billing entity is the one the affiliation IN
    FORCE ON THE DATE OF SERVICE names — both of which the bundle now records.
    """
    assert _run(deployment) == 0
    bundle = _bundle(deployment, STEM)

    assert bundle.release_blockers() == (), bundle.release_blockers()
    assert bundle.release.destination.value == "AUTO_READY"
    assert bundle.context.resolution.value == "RESOLVED"

    steps = {step.step: step for step in bundle.context.resolution_steps}
    assert set(steps) == {"encounter", "date_of_service", "patient", "coverage",
                          "rendering_provider", "affiliation", "facility"}, steps
    assert steps["encounter"].resolved_to == STEM
    assert steps["date_of_service"].identifier == DOS_ISO
    assert steps["patient"].resolved_to == PATIENT_ID
    assert steps["coverage"].identifier == COVERAGE_ID
    assert steps["coverage"].resolved_to == PAYER_ID
    assert steps["rendering_provider"].resolved_to == deployment.rendering_npi
    assert steps["affiliation"].identifier == AFFILIATION_ID
    assert steps["affiliation"].resolved_to == ENTITY_ID
    assert steps["facility"].identifier == FACILITY_ID

    # participant -> billing entity FOR the DOS, with the window that says so
    assert bundle.context.billing_entity.entity_id == ENTITY_ID
    assert bundle.context.billing_entity.npi == deployment.billing_npi
    assert bundle.context.affiliation.provider_npi == deployment.rendering_npi
    assert bundle.context.affiliation.effective_start == ROSTER_START

    # every required field came from the AUTHORITATIVE source, none from the note
    from app.contracts.claim_bundle import (
        AUTHORITATIVE_FIELD_SOURCE, REQUIRED_ENCOUNTER_CONTEXT)
    assert set(bundle.context.field_sources) == set(REQUIRED_ENCOUNTER_CONTEXT)
    assert set(bundle.context.field_sources.values()) == {AUTHORITATIVE_FIELD_SOURCE}

    # ...and it is a submittable claim end to end, still with no human input
    assert deployment.ingest()["recorded"] == 1
    assert deployment.dry_run()["submitted"] == 1


def test_the_note_cannot_supply_or_override_a_resolved_identity(deployment,
                                                                monkeypatch):
    """Extraction corroborates. It never decides — and objecting is not deciding.

    The note is made to state a DIFFERENT rendering-provider NPI than the source
    resolved. The resolved NPI must survive unchanged (the note cannot select a
    provider identity), and the disagreement must HOLD the encounter (two
    sources that disagree about who signed the claim are not reconciled by
    preferring one silently).
    """
    impostor = _valid_npi("177777777")
    assert impostor != deployment.rendering_npi
    monkeypatch.setattr(entrypoint, "extract_from_pdf", lambda pdf_path:
                        vision_extraction(
                            [NOTE_TEXT],
                            metadata={"date_of_service": DOS_ISO,
                                      "provider_npi": impostor},
                            document_version=DOCUMENT_VERSION,
                            extracted_text_sha256=EXTRACTED_TEXT_SHA))
    assert _run(deployment) == 0
    bundle = _bundle(deployment, STEM)

    assert bundle.context.rendering_provider.npi == deployment.rendering_npi
    assert bundle.context.resolution.value == "CONFLICT"
    assert any("rendering_provider.npi" in c for c in bundle.context.conflicts), \
        bundle.context.conflicts
    assert bundle.release_blockers()
    assert deployment.ingest()["recorded"] == 0


def test_a_resolved_context_may_not_carry_a_note_derived_required_field():
    """The invariant behind the test above, at the boundary that enforces it.

    A resolver bug that merged note metadata into a RESOLVED context would
    produce a bundle in which every field is populated and every fingerprint
    reproduces — nothing downstream could tell. `field_sources` makes it a
    contract violation instead of an invisible one.
    """
    from app.contracts.claim_bundle import (
        CORROBORATION_FIELD_SOURCE, ContextResolution, EncounterContext,
        ProviderIdentity)

    context = EncounterContext(
        resolution=ContextResolution.RESOLVED,
        rendering_provider=ProviderIdentity(npi=_valid_npi("166666666")),
        field_sources={"rendering_provider.npi": CORROBORATION_FIELD_SOURCE},
    )
    problems = " ".join(context.problems())
    assert "rendering_provider.npi was not supplied by an authoritative" in \
        problems, problems


# ==========================================================================
# ACCEPTANCE 2 — a broken encounter holds ONLY itself
# ==========================================================================

def _break_unknown_encounter(roster, npi):
    del roster["encounters"][OTHER_STEM]


def _break_expired_affiliation(roster, npi):
    for row in roster["affiliations"]:
        if row["provider_npi"] == npi:
            # ends the DAY BEFORE the date of service
            row["effective_end"] = "2026-03-13"


def _break_conflicting_affiliations(roster, npi):
    roster["billing_entities"]["ENT-E2E-2"] = {
        "name": "Second Entity PLLC", "npi": _valid_npi("155555555")}
    roster["affiliations"].append({
        "affiliation_id": "AFF-E2E-3", "provider_npi": npi,
        "billing_entity_id": "ENT-E2E-2",
        "effective_start": ROSTER_START, "effective_end": ""})


def _break_duplicate_affiliation_identifier(roster, npi):
    duplicate = dict(next(r for r in roster["affiliations"]
                          if r["provider_npi"] == npi))
    roster["affiliations"].append(duplicate)


_ISOLATION_CASES = {
    "unknown": (_break_unknown_encounter,
                "is not in the encounter context source"),
    "expired": (_break_expired_affiliation,
                "no billing affiliation in force on the date of service"),
    "conflicting": (_break_conflicting_affiliations,
                    "is affiliated with 2 billing entities"),
    "duplicate": (_break_duplicate_affiliation_identifier,
                  "appears on more than one record"),
}


@pytest.mark.parametrize("case", sorted(_ISOLATION_CASES))
def test_a_broken_encounter_holds_only_itself(deployment, case):
    """The directive's second acceptance test, in a TWO-note batch.

    A single-note batch cannot distinguish "this encounter held" from "the
    source stopped working", which is the whole point of the requirement. Both
    notes read the SAME context file; only the second one's chain is broken.
    """
    break_it, expected = _ISOLATION_CASES[case]
    other_npi = _valid_npi("144444444")
    _add_second_note(deployment, npi=other_npi)
    deployment.write_context(version="context-edition-1",
                             mutate=lambda roster: break_it(roster, other_npi))

    assert _run(deployment) == 0, "one broken encounter must not fail the batch"

    healthy = _bundle(deployment, STEM)
    assert healthy.release_blockers() == (), healthy.release_blockers()
    assert healthy.release.destination.value == "AUTO_READY"
    assert healthy.context.resolution.value == "RESOLVED"

    broken = _bundle(deployment, OTHER_STEM)
    assert broken.context.resolution.value == "UNRESOLVED"
    assert any(expected in reason for reason in broken.context.unresolved), \
        broken.context.unresolved
    assert broken.release_blockers()

    # and the batch boundary agrees: one claim recorded, one refused by name
    stats = deployment.ingest()
    assert stats["recorded"] == 1, stats["skip_reasons"]
    assert OTHER_STEM in stats["skip_reasons"]
    assert deployment.dry_run()["submitted"] == 1


def test_a_duplicated_encounter_identifier_holds_only_that_encounter(deployment):
    """`json.loads` keeps the LAST of a duplicated key and says nothing.

    An identifier that does not identify one record is the failure the whole
    design rests on not happening, and it cannot be produced by `json.dumps` —
    so the source is written as raw text here, exactly as a bad export would.
    """
    other_npi = _valid_npi("144444444")
    _add_second_note(deployment, npi=other_npi)
    roster = copy.deepcopy(deployment.roster)
    entry = json.dumps(roster["encounters"][OTHER_STEM])
    text = json.dumps(roster, indent=2)
    marker = '"encounters": {'
    assert text.count(marker) == 1
    deployment.context_file.write_text(text.replace(
        marker, f'{marker}\n    {json.dumps(OTHER_STEM)}: {entry},', 1))

    assert _run(deployment) == 0
    assert _bundle(deployment, STEM).release_blockers() == ()
    broken = _bundle(deployment, OTHER_STEM)
    assert any("declared more than once" in reason
               for reason in broken.context.unresolved), broken.context.unresolved


@pytest.mark.parametrize("effective_end,releasable", [
    ("2026-03-14", True),    # ends ON the date of service — still in force
    ("2026-03-13", False),   # ends the day before — not in force
])
def test_the_affiliation_window_is_inclusive_at_both_bounds(deployment,
                                                            effective_end,
                                                            releasable):
    """The off-by-one has a direction, and it reassigns the billing entity.

    A service performed on the last day of an affiliation is covered by it. An
    exclusive end bound would silently move that encounter to whichever entity
    the NEXT affiliation names — a wrong claim that every downstream control
    would accept. Pure calendar dates: no clock, no timezone, no locale takes
    part, so this result cannot depend on when or where the suite runs.
    """
    def _bound(roster):
        roster["affiliations"][0]["effective_end"] = effective_end

    deployment.write_context(version="context-edition-1", mutate=_bound)
    assert _run(deployment) == 0
    bundle = _bundle(deployment, STEM)
    assert (bundle.release_blockers() == ()) is releasable, \
        bundle.release_blockers()


def test_a_malformed_row_holds_its_encounter_and_not_the_batch(deployment):
    """An operator typo in one row is not a corrupt file.

    A file that cannot be parsed fails the batch (proved in
    `test_claim_bundle_e2e.py`). A single unparseable DATE inside an otherwise
    valid file must not: it would stop every unrelated encounter over one typo.
    It also must not be read as "no bound", which would WIDEN the window.
    """
    other_npi = _valid_npi("144444444")
    _add_second_note(deployment, npi=other_npi)

    def _typo(roster):
        for row in roster["affiliations"]:
            if row["provider_npi"] == other_npi:
                row["effective_start"] = "03/14/2026"

    deployment.write_context(version="context-edition-1", mutate=_typo)
    assert _run(deployment) == 0
    assert _bundle(deployment, STEM).release_blockers() == ()
    broken = _bundle(deployment, OTHER_STEM)
    assert any("is not an ISO calendar date" in reason
               for reason in broken.context.unresolved), broken.context.unresolved


# ==========================================================================
# ACCEPTANCE 3 — a changed party changes the fingerprint, and stale auth dies
# ==========================================================================

def _change_provider(deployment, roster):
    npi = _valid_npi("133333333")
    roster["providers"][npi] = {"first_name": "Lee", "last_name": "Sandoval",
                                "display_name": "Lee Sandoval"}
    roster["affiliations"].append({
        "affiliation_id": "AFF-E2E-9", "provider_npi": npi,
        "billing_entity_id": ENTITY_ID,
        "effective_start": ROSTER_START, "effective_end": ""})
    roster["encounters"][STEM]["rendering_provider_npi"] = npi


def _change_facility(deployment, roster):
    roster["facilities"][FACILITY_ID]["address1"] = "2 Other Way"
    roster["facilities"][FACILITY_ID]["npi"] = _valid_npi("122222222")


def _change_payer(deployment, roster):
    roster["payers"][PAYER_ID]["name"] = _other_payer_alias(
        roster["payers"][PAYER_ID]["name"])


@pytest.mark.parametrize("change", ["provider", "facility", "payer"])
def test_a_changed_party_changes_the_fingerprint_and_stops_the_dry_run(
        deployment, change):
    """The directive's third acceptance test, for each party it names.

    Phase 1 proved it for place of service. A different PROVIDER, FACILITY or
    PAYER is just as much a different claim: the earlier verification was of a
    claim that no longer exists, and re-using it would submit a claim nobody
    ever verified.
    """
    _verified_once(deployment)
    verified = reg.bundle_of_event(
        reg.current_view(reg.load_events(deployment.registry_path))[STEM])

    mutate = {"provider": _change_provider, "facility": _change_facility,
              "payer": _change_payer}[change]
    deployment.write_context(version="context-edition-2",
                             mutate=lambda roster: mutate(deployment, roster))
    assert _run(deployment) == 0
    fresh = _bundle(deployment, STEM)
    assert fresh.context.fingerprint != verified.context.fingerprint
    assert fresh.claim_fingerprint != verified.claim_fingerprint, (
        "the context fingerprint is part of the claim, so the claim identity "
        "must move with it")

    stats = deployment.dry_run()
    assert stats["submitted"] == 0
    assert stats["blocked"] == 1
    assert "context changed" in stats["docs"][STEM], stats["docs"][STEM]


def _with_authorization(deployment, roster, *, number="AUTH-E2E-1"):
    roster["authorizations"].append({
        "authorization_id": "AUTH-E2E-ID", "authorization_number": number,
        "coverage_id": COVERAGE_ID,
        "rendering_provider_npi": deployment.rendering_npi,
        "facility_id": FACILITY_ID,
        "effective_start": ROSTER_START, "effective_end": "2026-12-31"})
    roster["encounters"][STEM]["authorization_id"] = "AUTH-E2E-ID"


def test_an_authorization_is_carried_only_while_its_parties_still_match(
        deployment):
    """A prior authorization is not a property of the patient.

    It was issued for one coverage, one rendering provider and one facility.
    When any of them changes, the encounter HOLDS rather than reusing the
    number: silently carrying it over would put an approval on a claim the
    payer never approved — and the claim would look complete.
    """
    deployment.write_context(
        version="context-edition-1",
        mutate=lambda roster: _with_authorization(deployment, roster))
    assert _run(deployment) == 0
    authorized = _bundle(deployment, STEM)
    assert authorized.release_blockers() == (), authorized.release_blockers()
    assert authorized.context.coverage.authorization_number == "AUTH-E2E-1"
    assert authorized.context.subscriber.authorization_number == "AUTH-E2E-1"

    def _new_provider_same_authorization(roster):
        _with_authorization(deployment, roster)
        _change_provider(deployment, roster)

    deployment.write_context(version="context-edition-2",
                             mutate=_new_provider_same_authorization)
    assert _run(deployment) == 0
    stale = _bundle(deployment, STEM)
    assert stale.context.coverage.authorization_number == "", (
        "an authorization issued for another provider was carried onto the claim")
    assert stale.context.resolution.value == "UNRESOLVED"
    assert any("does not authorize this encounter" in reason
               for reason in stale.context.unresolved), stale.context.unresolved
    assert stale.release_blockers()


def test_an_authorization_outside_its_window_is_not_carried(deployment):
    """Expiry is the other way an authorization goes stale, and it is silent."""
    def _expired(roster):
        _with_authorization(deployment, roster)
        roster["authorizations"][-1]["effective_end"] = "2026-03-13"

    deployment.write_context(version="context-edition-1", mutate=_expired)
    assert _run(deployment) == 0
    bundle = _bundle(deployment, STEM)
    assert bundle.context.coverage.authorization_number == ""
    assert any("is not in force on the date of service" in reason
               for reason in bundle.context.unresolved), bundle.context.unresolved


# ==========================================================================
# ACCEPTANCE 4 — a clean deployment provisions the adapter automatically
# ==========================================================================

CONTEXT_DIR = REPO_ROOT / "data" / "context"
TEMPLATE = CONTEXT_DIR / "encounter_context.example.json"


def test_configuration_alone_provisions_the_adapter(deployment, monkeypatch):
    """No flag, no code, no fixture wiring — only the deployed env variable.

    `docker compose run app python run.py` passes NO arguments, so an adapter
    that only works when a developer types `--encounter-context` is an adapter
    the deployment does not have.
    """
    monkeypatch.setenv("ENCOUNTER_CONTEXT_FILE", str(deployment.context_file))
    assert entrypoint.main(
        ["--billing-context", str(deployment.billing_context_file)]) == 0
    bundle = _bundle(deployment, STEM)
    assert bundle.context.resolution.value == "RESOLVED"
    assert bundle.release.destination.value == "AUTO_READY"
    assert bundle.release_blockers() == ()


def test_the_container_can_actually_see_the_configured_source():
    """A named volume at /app/data hides anything the host writes under ./data.

    Without a bind mount MORE SPECIFIC than `app_data:/app/data`, configuring
    `ENCOUNTER_CONTEXT_FILE` to a host path fails the batch on every boot with
    "file could not be read" and nothing in the deployment explains why. This
    is the difference between a provisioned adapter and one that only works in
    a test.
    """
    compose = (REPO_ROOT / "docker-compose.yml").read_text()
    assert "- ./data/context:/app/data/context:ro" in compose, (
        "docker-compose.yml no longer bind-mounts the encounter context "
        "directory; a host-side roster would be invisible in the container")
    assert "- app_data:/app/data\n" in compose, (
        "the /app/data named volume this mount must be more specific than is "
        "gone; re-check that the context mount still takes precedence")

    env_example = (REPO_ROOT / ".env.example").read_text()
    assert "ENCOUNTER_CONTEXT_FILE=data/context/encounter_context.json" in \
        env_example, ("the documented configuration no longer points inside the "
                      "bind-mounted directory")


def test_the_shipped_template_loads_and_holds_until_it_is_filled_in():
    """The template must be readable by the adapter and useless until edited.

    Both halves matter. A template the adapter refuses is a deployment that
    cannot start; a template that RESOLVES would let a copy-paste bill a claim
    under placeholder identities.
    """
    from app.contracts.encounter_context import VersionedRosterContextProvider

    provider = VersionedRosterContextProvider(TEMPLATE)
    preflight = provider.preflight()
    assert preflight["authoritative"] is True
    assert preflight["encounters"] == 1
    assert not preflight["duplicate_identifiers"]

    (encounter_id,) = json.loads(TEMPLATE.read_text())["encounters"]
    context = provider.resolve(encounter_id=encounter_id, document_id="",
                               date_of_service=DOS_ISO)
    assert context.resolution.value != "RESOLVED"
    assert context.problems()


def test_a_superseded_context_schema_is_refused_by_name(deployment):
    """Phase 1's flat schema cannot express an affiliation for a date of service.

    Reading it anyway would mean releasing claims whose billing entity was never
    resolved for the DOS. It is refused with the migration named, and the batch
    stops — a silent downgrade to "every encounter holds" would be indis-
    tinguishable from an unconfigured deployment.
    """
    legacy = deployment.tmp_path / "legacy_context.json"
    legacy.write_text(json.dumps({
        "schema": "encounter_context/1", "version": "context-edition-0",
        "encounters": {STEM: {}}}))
    assert _run(deployment, context_file=legacy) == 1
    assert not (deployment.output_dir / f"{STEM}_results.json").exists()


def test_an_unknown_adapter_is_refused_by_name_not_retried_as_a_path():
    """A future EHR/FHIR adapter is a REGISTRATION, not a rewrite.

    `build_provider` selects from a registry, so adding one changes no
    configuration, no entrypoint and no container. Until one is registered, a
    spec naming it must say so — retrying `fhir:https://...` as a filename
    would report "file not found" and send an operator to fix the wrong thing.
    """
    from app.contracts.encounter_context import (
        EncounterContextUnavailable, NoteMetadataContextProvider,
        VersionedRosterContextProvider, build_provider, registered_adapters)

    assert "versioned_roster" in registered_adapters()
    assert isinstance(build_provider(None), NoteMetadataContextProvider)
    assert isinstance(build_provider("data/context/x.json"),
                      VersionedRosterContextProvider)
    assert isinstance(build_provider("versioned_roster:data/context/x.json"),
                      VersionedRosterContextProvider)
    with pytest.raises(EncounterContextUnavailable) as excinfo:
        build_provider("fhir:https://example.invalid/Encounter")
    assert "no encounter context adapter named 'fhir'" in str(excinfo.value)


def test_the_practice_config_cannot_substitute_a_resolved_provider(deployment):
    """The sibling defect this phase's review found, on the real submission path.

    Directive §2 forbids selecting a provider identity from a broad roster.
    The encounter context resolves the rendering provider BY NPI — and the
    claim submitter then matched the practice config's own roster against a
    SUBSTRING of that provider's display name and, failing that, fell back to a
    configured default. Either one signs the claim as a different human, after
    the authoritative resolution, with every downstream check still passing.

    Here the practice config is armed with both traps: an entry whose pattern
    matches the resolved provider's name and a default provider, each carrying
    a valid but DIFFERENT NPI. The submitted 837P must carry the resolved one.
    """
    config = json.loads(deployment.practice_config_file.read_text())
    impostor = _valid_npi("111111111")
    fallback = _valid_npi("199999997")
    assert impostor != deployment.rendering_npi != fallback
    config["rendering_providers"] = {
        "providers": [{"match": ["vasquez"], "first_name": "Someone",
                       "last_name": "Else", "npi": impostor,
                       "taxonomy_code": "213E00000X"}],
        "trust_note_npi": True,
        "default": {"first_name": "Default", "last_name": "Provider",
                    "npi": fallback, "taxonomy_code": "213E00000X"},
    }
    deployment.practice_config_file.write_text(json.dumps(config))

    assert _run(deployment) == 0
    bundle = _bundle(deployment, STEM)
    assert bundle.context.rendering_provider.npi == deployment.rendering_npi
    assert deployment.ingest()["recorded"] == 1
    assert deployment.dry_run()["submitted"] == 1

    payload = json.loads(
        (deployment.dryrun_dir / f"{STEM}_837p.json").read_text())
    assert payload["rendering"]["npi"] == deployment.rendering_npi, (
        "the practice config substituted a different provider for the one the "
        "encounter context resolved by identifier")


def test_an_invalid_resolved_npi_blocks_instead_of_falling_back(deployment):
    """A bad resolved identifier must not become a clean claim under another NPI.

    Falling back to the configured default here is the same defect wearing a
    different hat: the claim would be submitted, and it would name a provider
    who did not render the service.
    """
    config = json.loads(deployment.practice_config_file.read_text())
    config["rendering_providers"] = {
        "providers": [], "trust_note_npi": True,
        "default": {"first_name": "Default", "last_name": "Provider",
                    "npi": _valid_npi("199999997"),
                    "taxonomy_code": "213E00000X"}}
    deployment.practice_config_file.write_text(json.dumps(config))

    bad_npi = deployment.rendering_npi[:-1] + \
        str((int(deployment.rendering_npi[-1]) + 1) % 10)

    def _bad_check_digit(roster):
        roster["providers"][bad_npi] = roster["providers"].pop(
            deployment.rendering_npi)
        for row in roster["affiliations"]:
            if row["provider_npi"] == deployment.rendering_npi:
                row["provider_npi"] = bad_npi
        roster["encounters"][STEM]["rendering_provider_npi"] = bad_npi

    deployment.write_context(version="context-edition-1",
                             mutate=_bad_check_digit)
    assert _run(deployment) == 0
    assert deployment.ingest()["recorded"] == 1
    stats = deployment.dry_run()
    assert stats["submitted"] == 0
    assert "may not substitute another provider" in stats["docs"][STEM], \
        stats["docs"][STEM]


# ==========================================================================
# guarding the guards — checks that could silently never fire
# ==========================================================================
# The duplicate-key detector in this module shipped, in review, with a bug that
# made it return "no duplicates" for EVERY file: `object_pairs_hook` hands the
# hook a list of TUPLES and the code tested for lists. It passed every test that
# did not deliberately construct a duplicate. So each check below that a normal
# run cannot reach now has a case that reaches it.

def test_a_context_source_date_of_service_that_disagrees_holds(deployment):
    """The source may cross-check the DOS. It may not silently redefine it.

    Time-bound resolution is evaluated against the CLAIM's own date of service,
    so a source that believes the encounter happened on another day is stating
    a disagreement about which encounter this is — not correcting the document.
    """
    def _other_day(roster):
        roster["encounters"][STEM]["date_of_service"] = "2026-03-15"

    deployment.write_context(version="context-edition-1", mutate=_other_day)
    assert _run(deployment) == 0
    bundle = _bundle(deployment, STEM)
    assert bundle.context.resolution.value == "CONFLICT"
    assert any("date of service" in c for c in bundle.context.conflicts), \
        bundle.context.conflicts
    assert bundle.release_blockers()


def test_a_window_that_ends_before_it_starts_holds(deployment):
    """An inverted window is not an empty window — it is an unreadable record.

    Left unchecked it simply never matches, which reads downstream as "this
    provider has no affiliation" and sends an operator hunting for a missing
    row that is right there.
    """
    def _inverted(roster):
        roster["affiliations"][0]["effective_end"] = "2019-01-01"

    deployment.write_context(version="context-edition-1", mutate=_inverted)
    assert _run(deployment) == 0
    bundle = _bundle(deployment, STEM)
    assert any("before it starts" in reason
               for reason in bundle.context.unresolved), \
        bundle.context.unresolved


def test_two_identifiers_naming_two_different_records_are_ambiguous(deployment):
    """Driven at the adapter, because the entrypoint cannot reach this case.

    `run.py` passes the note's file stem as BOTH the encounter id and the
    document id, so today they always agree. The interface accepts two
    identifiers and a future adapter will supply two genuinely different ones —
    at which point preferring one silently would bill this encounter under
    another encounter's context. Testing it only through the entrypoint would
    mean not testing it at all.
    """
    from app.contracts.encounter_context import VersionedRosterContextProvider

    def _second_record(roster):
        roster["encounters"]["DOC-OTHER"] = dict(
            roster["encounters"][STEM],
            facility_id=FACILITY_ID, patient_id=PATIENT_ID)
        roster["encounters"]["DOC-OTHER"]["coverage_id"] = COVERAGE_ID
        roster["encounters"]["DOC-OTHER"]["rendering_provider_npi"] = \
            _valid_npi("188888887")

    path = deployment.write_context(
        version="context-edition-1", mutate=_second_record,
        path=deployment.tmp_path / "two_records.json")
    context = VersionedRosterContextProvider(path).resolve(
        encounter_id=STEM, document_id="DOC-OTHER", date_of_service=DOS_ISO)
    assert context.resolution.value == "UNRESOLVED"
    assert any("resolve to two different encounter records" in reason
               for reason in context.unresolved), context.unresolved


def test_the_contract_cross_checks_the_affiliation_it_was_handed():
    """A resolver bug that bound the wrong affiliation is otherwise invisible.

    Every field would be populated, `missing_required()` would be empty and
    every fingerprint would reproduce. These two invariants are the only thing
    between "the resolver has a bug" and "the claim names the wrong billing
    entity" — so they are asserted directly rather than trusted to fire.
    """
    from app.contracts.claim_bundle import (
        AUTHORITATIVE_FIELD_SOURCE, REQUIRED_ENCOUNTER_CONTEXT,
        AffiliationBinding, BillingEntityIdentity, ContextResolution,
        EncounterContext, ProviderIdentity)

    sources = {path: AUTHORITATIVE_FIELD_SOURCE
               for path in REQUIRED_ENCOUNTER_CONTEXT}
    wrong_entity = EncounterContext(
        resolution=ContextResolution.RESOLVED, field_sources=sources,
        billing_entity=BillingEntityIdentity(entity_id="ENT-A"),
        affiliation=AffiliationBinding(billing_entity_id="ENT-B"))
    assert any("but the context names" in problem
               for problem in wrong_entity.problems()), wrong_entity.problems()

    wrong_provider = EncounterContext(
        resolution=ContextResolution.RESOLVED, field_sources=sources,
        rendering_provider=ProviderIdentity(npi=_valid_npi("188888888")),
        affiliation=AffiliationBinding(provider_npi=_valid_npi("199999998")))
    assert any("not to this encounter's rendering provider" in problem
               for problem in wrong_provider.problems()), \
        wrong_provider.problems()


# ==========================================================================
# ISSUE #6, CODEX F7-R2 (P1) — a coverage must belong to THIS encounter's patient
# ==========================================================================
#
# Reproduction as the reviewer ran it: encounter E1 references patient P1 and an
# explicit, ACTIVE coverage C2 that belongs to a DIFFERENT patient P2. It resolved
# `RESOLVED`, with P1's demographics, C2's coverage and P2's member id, no holds,
# and `problems() == ()`. Patient identity and subscriber coverage are resolved by
# two independent branches; nothing joined them, and nothing downstream could,
# because every field was populated and every fingerprint reproduced.

SECOND_PATIENT_ID = "PAT-E2E-2"
SECOND_COVERAGE_ID = "COV-E2E-2"
SECOND_MEMBER_ID = "MEMBER-P2"


def _another_patients_active_coverage(roster):
    """A second patient whose coverage is in force over the SAME window as the
    first's — so the encounter's coverage differs from a correct one in exactly
    one thing: whose it is."""
    roster["patients"][SECOND_PATIENT_ID] = {
        "first_name": "Devon", "last_name": "Marsh",
        "date_of_birth": "01/03/1975", "gender": "M",
        "record_number": "MRN-E2E-2"}
    roster["coverages"].append({
        "coverage_id": SECOND_COVERAGE_ID, "patient_id": SECOND_PATIENT_ID,
        "payer_id": PAYER_ID, "member_id": SECOND_MEMBER_ID,
        "group_number": "GRP-E2E-2", "relationship_to_patient": "",
        "effective_start": ROSTER_START, "effective_end": ""})
    roster["encounters"][STEM]["coverage_id"] = SECOND_COVERAGE_ID


def test_an_encounter_cannot_resolve_with_another_patients_coverage(deployment):
    """The adapter boundary, on the reviewer's exact reproduction."""
    from app.contracts.encounter_context import VersionedRosterContextProvider

    path = deployment.write_context(
        version="ctx-foreign-coverage", mutate=_another_patients_active_coverage,
        path=deployment.tmp_path / "foreign_coverage.json")
    context = VersionedRosterContextProvider(path).resolve(
        encounter_id=STEM, document_id=STEM, date_of_service=DOS_ISO)

    assert context.resolution.value != "RESOLVED", context.resolution
    assert any("not to this encounter's patient" in reason
               for reason in context.unresolved), context.unresolved
    # The other patient's identity reached NOTHING: not the member id, not the
    # coverage binding, not the payer.
    assert context.subscriber.member_id != SECOND_MEMBER_ID
    assert context.coverage.coverage_id == ""
    assert context.problems()


def test_a_coverage_with_no_declared_patient_is_refused_too(deployment):
    """The same defect's quiet twin: a coverage row that names nobody would
    otherwise pass a `!=` check written the obvious way."""
    from app.contracts.encounter_context import VersionedRosterContextProvider

    def _orphan(roster):
        roster["coverages"][0]["patient_id"] = ""

    path = deployment.write_context(
        version="ctx-orphan-coverage", mutate=_orphan,
        path=deployment.tmp_path / "orphan_coverage.json")
    context = VersionedRosterContextProvider(path).resolve(
        encounter_id=STEM, document_id=STEM, date_of_service=DOS_ISO)

    assert context.resolution.value != "RESOLVED"
    assert any("<none declared>" in reason for reason in context.unresolved), \
        context.unresolved


def test_another_patients_coverage_holds_only_that_encounter(deployment):
    """The DEPLOYED entrypoint, and the failure boundary that matters.

    A referential-integrity break in one encounter's coverage is that
    encounter's hold. The second note in the same batch reads the same source
    file and must still reach AUTO_READY, or the fix would have traded a wrong
    claim for a stopped practice.
    """
    _add_second_note(deployment, npi=_valid_npi("122222222"))
    deployment.write_context(version="ctx-foreign-coverage-batch",
                             mutate=_another_patients_active_coverage)

    assert _run(deployment) == 0
    broken = _bundle(deployment, STEM)
    healthy = _bundle(deployment, OTHER_STEM)

    assert broken.context.resolution.value != "RESOLVED"
    assert any("not to this encounter's patient" in reason
               for reason in broken.context.unresolved), broken.context.unresolved
    assert broken.release_blockers()
    assert broken.context.subscriber.member_id != SECOND_MEMBER_ID
    assert broken.context.coverage.coverage_id == ""

    assert healthy.release_blockers() == (), healthy.release_blockers()
    assert healthy.release.destination.value == "AUTO_READY"
    # Exactly one claim recorded, and it is the healthy one.
    assert deployment.ingest()["recorded"] == 1


def test_the_contract_cross_checks_the_coverage_it_was_handed():
    """The invariant behind the resolver, at the boundary that enforces it.

    Stated on the CONTRACT rather than only inside `VersionedRoster…` for the
    same reason the affiliation invariants are: the next adapter (an EHR/FHIR or
    practice-management one) resolves patient and coverage in two branches too,
    and this is what stops it from shipping the same defect.
    """
    from app.contracts.claim_bundle import (
        AUTHORITATIVE_FIELD_SOURCE, REQUIRED_ENCOUNTER_CONTEXT, ContextResolution,
        CoverageBinding, EncounterContext, PatientIdentity)

    sources = {path: AUTHORITATIVE_FIELD_SOURCE
               for path in REQUIRED_ENCOUNTER_CONTEXT}
    wrong_patient = EncounterContext(
        resolution=ContextResolution.RESOLVED, field_sources=sources,
        patient=PatientIdentity(patient_id="PAT-A"),
        coverage=CoverageBinding(coverage_id="COV-1", patient_id="PAT-B"))
    assert any("not to this encounter's patient" in problem
               for problem in wrong_patient.problems()), wrong_patient.problems()


def test_an_authorization_is_checked_against_the_RESOLVED_facility(deployment):
    """The same bug class, one branch over (found by the F7-R2 second pass).

    The authorization used to be checked against the facility id the ENCOUNTER
    RECORD declared, not the one the facility branch resolved — so an
    authorization could 'match' a facility that is not in the source at all.
    """
    from app.contracts.encounter_context import VersionedRosterContextProvider

    def _unknown_facility(roster):
        roster["encounters"][STEM]["facility_id"] = "FAC-NOT-IN-SOURCE"
        roster["encounters"][STEM]["authorization_id"] = "AUTH-1"
        roster["authorizations"].append({
            "authorization_id": "AUTH-1", "authorization_number": "AUTH-NUMBER-1",
            "coverage_id": COVERAGE_ID, "facility_id": "FAC-NOT-IN-SOURCE",
            "rendering_provider_npi": deployment.rendering_npi,
            "effective_start": ROSTER_START, "effective_end": ""})

    path = deployment.write_context(
        version="ctx-unknown-facility", mutate=_unknown_facility,
        path=deployment.tmp_path / "unknown_facility.json")
    context = VersionedRosterContextProvider(path).resolve(
        encounter_id=STEM, document_id=STEM, date_of_service=DOS_ISO)

    assert context.resolution.value != "RESOLVED"
    assert any("service facility" in reason for reason in context.unresolved), \
        context.unresolved
    # The authorization number never reached the claim.
    assert context.subscriber.authorization_number == ""
    assert context.coverage.authorization_id == ""


# ==========================================================================
# ISSUE #6, CODEX F7-R4 (P1) — ONE date of service, established not transcribed
# ==========================================================================
#
# `read_note()` took the DOS from the primary vision model's structured metadata
# and nothing ever compared it to the document. The roster's encounter DOS is
# optional, so when it was absent the resolver accepted any parseable caller value
# and used it for coverage, affiliation, authorization, code activity and the claim
# itself. A one-character misread selected a different coverage, a different
# affiliation and a different effective code edition — and still produced a fully
# populated, fully fingerprinted context.

#: The three axes a date can be misread on.
PERTURBED_DATES = ["2026-04-14", "2026-03-15", "2027-03-14"]


def _transcription(monkeypatch, *, page_text=NOTE_TEXT, dos=DOS_ISO):
    """Substitute the vision channel. The PDF ON DISK is never touched, so its
    embedded text layer remains an independent reading of the TRUE document."""
    monkeypatch.setattr(entrypoint, "extract_from_pdf", lambda pdf_path:
                        vision_extraction(
                            [page_text], metadata={"date_of_service": dos},
                            document_version=DOCUMENT_VERSION,
                            extracted_text_sha256=EXTRACTED_TEXT_SHA))


def _declare_roster_dos(deployment, date_of_service=DOS_ISO, version="ctx-dos"):
    return deployment.write_context(
        version=version,
        mutate=lambda roster: roster["encounters"][STEM].update(
            {"date_of_service": date_of_service}))


def test_the_date_of_service_binds_from_the_reconciled_document(deployment):
    """No roster DOS: the document must PROVE its own date before it can bind."""
    from app.contracts.claim_bundle import DOCUMENT_SERVICE_DATE_SOURCE
    from app.ingestion.source_evidence import EMBEDDED_TEXT_CHANNEL_ID

    assert _run(deployment) == 0
    bundle = _bundle(deployment, STEM)
    assert bundle.release_blockers() == (), bundle.release_blockers()

    binding = bundle.context.service_date
    assert binding.date_of_service == DOS_ISO
    assert binding.source == DOCUMENT_SERVICE_DATE_SOURCE
    assert binding.documented_date == DOS_ISO
    assert binding.document_status == "AGREED"
    assert binding.document_pages == (1,)
    assert binding.verified_by_channel_id == EMBEDDED_TEXT_CHANNEL_ID
    assert binding.page_image_sha256 and all(binding.page_image_sha256)

    # ...and it is the SAME value every consumer saw.
    assert bundle.context.date_of_service == DOS_ISO
    assert bundle.encounter.date_of_service == DOS_ISO
    assert f"DOS={DOS_ISO}" in bundle.audit.audit_trail
    certificate = bundle.certificate.certificate
    assert certificate["date_of_service"] == DOS_ISO
    assert certificate["service_date_binding"]["source"] == \
        DOCUMENT_SERVICE_DATE_SOURCE
    assert certificate["service_date_binding"]["document_status"] == "AGREED"


def test_a_roster_declared_date_of_service_is_the_authority(deployment):
    """When the context source declares one, it wins: it is an identifier-resolved
    fact from an authority, not a reading of a page."""
    from app.contracts.claim_bundle import CONTEXT_SERVICE_DATE_SOURCE

    _declare_roster_dos(deployment)
    assert _run(deployment) == 0
    bundle = _bundle(deployment, STEM)

    assert bundle.release_blockers() == (), bundle.release_blockers()
    binding = bundle.context.service_date
    assert binding.source == CONTEXT_SERVICE_DATE_SOURCE
    assert binding.declared_date == DOS_ISO
    assert binding.date_of_service == DOS_ISO
    # The document was still read and still recorded — it just was not the authority.
    assert binding.documented_date == DOS_ISO
    assert bundle.encounter.date_of_service == DOS_ISO


@pytest.mark.parametrize("perturbed", PERTURBED_DATES)
def test_a_metadata_only_date_misread_cannot_bind(deployment, monkeypatch,
                                                  perturbed):
    """Month, day and year perturbation with NO roster-declared DOS.

    Only the structured metadata field is perturbed; the pages still say the true
    date. The claim's date can therefore be pointed at no page of the original —
    which is exactly what a claim date must never be.
    """
    _transcription(monkeypatch, dos=perturbed)
    assert _run(deployment) == 0
    bundle = _bundle(deployment, STEM)

    binding = bundle.context.service_date
    assert binding.date_of_service == ""
    assert binding.documented_date == perturbed
    assert binding.document_status == "NOT_LOCATED"
    # No page of the original was ever named for it — the date the claim would
    # have carried exists only in the metadata field.
    assert binding.document_pages == ()
    assert binding.document_span_id == ""
    assert "written nowhere" in binding.document_detail
    assert bundle.context.resolution.value != "RESOLVED"
    assert not bundle.encounter.date_of_service
    assert bundle.release_blockers()
    assert deployment.ingest()["recorded"] == 0


@pytest.mark.parametrize("perturbed", PERTURBED_DATES)
def test_a_transcription_wide_date_misread_cannot_bind(deployment, monkeypatch,
                                                       perturbed):
    """The harder case: the WHOLE transcription reads the date wrong, metadata and
    page text alike, so it is self-consistent. Only a reading that is not the
    transcription's own can detect it — which is what the compiler provides."""
    _transcription(monkeypatch, page_text=NOTE_TEXT.replace(DOS_ISO, perturbed),
                   dos=perturbed)
    assert _run(deployment) == 0
    bundle = _bundle(deployment, STEM)

    binding = bundle.context.service_date
    assert binding.date_of_service == ""
    # BLOCKING either way; see `ServiceDateReconciliationTest` for why an ISO date
    # (one token) reports as not appearing in the independent reading while a
    # written date reports as read differently.
    assert binding.document_status in {"DISAGREED", "NOT_LOCATED"}
    # Unlike the metadata-only misread, this one WAS anchored to a page of the
    # original and an independent channel read that page and refused it.
    assert binding.document_pages == (1,)
    assert binding.document_span_id
    assert binding.verified_by_channel_id
    assert bundle.context.resolution.value != "RESOLVED"
    assert not bundle.encounter.date_of_service
    assert bundle.release_blockers()
    assert deployment.ingest()["recorded"] == 0


@pytest.mark.parametrize("perturbed", PERTURBED_DATES)
def test_a_document_date_that_contradicts_the_roster_is_a_conflict(
        deployment, monkeypatch, perturbed):
    """The same three perturbations WITH a roster-declared DOS.

    The roster stays authoritative — so the claim never silently adopts the
    misread — but two sources naming two service dates is a question for a human,
    not something to be resolved by preferring either one silently.
    """
    _transcription(monkeypatch, page_text=NOTE_TEXT.replace(DOS_ISO, perturbed),
                   dos=perturbed)
    _declare_roster_dos(deployment, version="ctx-dos-conflict")
    assert _run(deployment) == 0
    bundle = _bundle(deployment, STEM)

    assert bundle.context.resolution.value == "CONFLICT"
    assert any("date of service" in conflict
               for conflict in bundle.context.conflicts), bundle.context.conflicts
    assert bundle.context.service_date.date_of_service == DOS_ISO
    assert bundle.context.service_date.documented_date == perturbed
    assert bundle.release_blockers()
    assert deployment.ingest()["recorded"] == 0


def test_the_bound_date_of_service_changes_the_context_fingerprint(deployment):
    """A different bound date is a different context. If the binding were outside
    the fingerprint, a claim could be re-dated without invalidating anything."""
    assert _run(deployment) == 0
    first = _bundle(deployment, STEM).context

    changed = first.model_copy(update={
        "service_date": first.service_date.model_copy(
            update={"date_of_service": "2026-03-21"})})
    assert changed.compute_fingerprint() != first.fingerprint
    assert any("fingerprint does not reproduce" in problem
               for problem in changed.problems()), changed.problems()


def test_a_caller_asserted_date_of_service_can_never_reach_a_claim():
    """The hole the finding names, closed at the contract.

    A date nobody established is recorded as what it is and refused for a
    RESOLVED context — so it can travel through an audit trail and can never
    travel onto a claim.
    """
    from app.contracts.claim_bundle import (
        AUTHORITATIVE_FIELD_SOURCE, CALLER_SERVICE_DATE_SOURCE,
        REQUIRED_ENCOUNTER_CONTEXT, ContextResolution, EncounterContext,
        ServiceDateBinding)

    sources = {path: AUTHORITATIVE_FIELD_SOURCE
               for path in REQUIRED_ENCOUNTER_CONTEXT}
    asserted = EncounterContext(
        resolution=ContextResolution.RESOLVED, field_sources=sources,
        service_date=ServiceDateBinding(date_of_service=DOS_ISO,
                                        source=CALLER_SERVICE_DATE_SOURCE))
    assert any("never established from the encounter context" in problem
               for problem in asserted.problems()), asserted.problems()

    undated = EncounterContext(resolution=ContextResolution.RESOLVED,
                               field_sources=sources)
    assert any("no date of service is bound" in problem
               for problem in undated.problems()), undated.problems()


def test_the_bound_date_of_service_is_the_only_date_any_consumer_sees():
    """Eligibility, the service episodes, the graph, the claim result and the
    certificate must all read ONE value — the point of binding it at all.

    Driven through `code_encounter` directly so the assertion is about the
    producer's own plumbing rather than about what an artifact happens to expose.
    """
    from claude_coder.certificate import build_certificate
    from tests.test_evidence_graph import _reading, _run as _code_encounter

    bound = "2026-03-14"
    reading = _reading("excision procedure alpha performed", "right",
                       "Procedure alpha performed today")
    result = _code_encounter(
        reading, reading,
        service_date_binding={"date_of_service": bound,
                              "source": "encounter_context"})

    assert result.date_of_service == bound
    assert result.graph is not None
    assert result.graph.date_of_service == bound
    assert {episode.date_of_service for episode in result.graph.episodes} == {bound}
    assert {intent.date_of_service for intent in result.claim_line_intents} == {bound}
    certificate = build_certificate(result, "note text")
    assert certificate["date_of_service"] == bound
    assert certificate["service_date_binding"]["source"] == "encounter_context"
