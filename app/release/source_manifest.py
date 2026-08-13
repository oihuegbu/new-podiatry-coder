"""Checksummed provenance for authoritative coding inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import calendar
from datetime import date
from functools import lru_cache
from pathlib import Path

from app.core import config

_AUTHORITATIVE = {
    "icd10_codes": config.ICD10_FILE,
    "cpt_codes": config.CPT_FILE,
    "hcpcs_codes": config.HCPCS_FILE,
    "ncci_edits": config.NCCI_FILE,
    "mue_limits": config.MUE_FILE,
    "coverage_policy": config.LCD_FILE,
    # Global-period / fee-schedule policy decides post-operative bundling, so it is
    # release-bearing and must be registered here (not only in the coder's capability
    # probe) -- the required-source declaration below resolves its path from THIS
    # registry, so the two can never drift apart.  (Codex F6-R5.)
    "global_periods": config.GLOBAL_PERIODS_FILE,
    # The SECOND PFS extract: the one the coder itself reads for global-period and
    # bilateral indicators.  It is a different file from `global_periods` above (a
    # different quarterly RVU release, parsed for different columns), and it was read at
    # decision time while being certified by nobody.  (Codex F6-R5, round 5.)
    "pfs_indicators": config.PFS_INDICATOR_FILE,
    # Modifier definitions: the coder's modifier engine resolves every modifier it emits
    # out of these bytes, so they decide what appears on the claim.
    "modifier_definitions": config.MODIFIER_FILE,
    # ICD-10-CM Tabular instructional notes: the Excludes1 conflict gate is evaluated
    # from them and degrades to NOT_APPLICABLE without them, so their absence RELAXES a
    # validation gate.
    "instructional_notes": config.INSTRUCTIONAL_NOTES_FILE,
    "validator_rules": config.VALIDATOR_RULES_FILE,
    # SNOMED root concepts + the confidence CAP applied to a root-level concept. Absence
    # leaves the root set empty, so the cap is never applied and a root-level match keeps
    # its full confidence -- absence RELAXES a validation restriction.
    "snomed_root_concepts": config.SNOMED_ROOTS_FILE,
    # Governed terminology is not code authority, but it is a release-bearing
    # interpretation source and must be bound into the same immutable manifest.
    "terminology_registry": config.TERMINOLOGY_REGISTRY_FILE,
    # Reviewed claim-affecting CONTROL CONFIGURATION.  These are not medical-code data,
    # but their bytes decide which diagnosis->service relations may release a claim, so
    # their content identity belongs in the same certifiable manifest as the data.  A
    # version string in an audit record is not an identity: the file can change without
    # it changing.  (Codex F6-R5, round 5.)
    "necessity_relation_control": config.NECESSITY_RELATION_CONTROL_FILE,
    "relation_evidence_grammar": config.RELATION_EVIDENCE_GRAMMAR_FILE,
    # --- round 6: sources the APP-side claim path reads ---------------------------
    # Codex F6-R5-A: round 5 derived the required set from `claude_coder`'s runtime
    # graph and expanded the structural filename-literal guard over `claude_coder/*.py`
    # only -- but `app/**` is in the deployed image and still owns claim-affecting
    # reads (the human-run 837P submission step resolves the payer through
    # `app.compliance.payer_registry`; `app.compliance.datastore.store` owns the
    # modifier-role and semantic-class vocabulary; `app.release.scope_registry` owns
    # what may be released autonomously).  Every one of these was composed as a
    # filename literal in its own reader and therefore reached the manifest only
    # through the incidental `data/codes/*.json` sweep below: a file that went MISSING
    # simply dropped out of the manifest, so the release could not tell an intentional
    # absence from a silently relaxed claim path.  They are identities now.
    "coding_semantics": config.CODING_SEMANTICS_FILE,
    "payer_registry": config.PAYERS_FILE,
    "pos_codes": config.POS_CODES_FILE,
    "modifier_exempt": config.MODIFIER_EXEMPT_FILE,
    "ncci_aoc_edits": config.NCCI_AOC_FILE,
    "mce_edits": config.MCE_EDITS_FILE,
    "icd10_chronic": config.ICD10_CHRONIC_FILE,
    "cpt_categories": config.CPT_CATEGORIES_FILE,
    "icd10_chapters": config.ICD10_CHAPTERS_FILE,
    "icd10_extensions": config.ICD10_EXTENSIONS_FILE,
    "mac_jurisdictions": config.MAC_JURISDICTIONS_FILE,
    "mcd_coverage_cache": config.MCD_COVERAGE_CACHE_FILE,
    "descriptor_qualifiers": config.DESCRIPTOR_QUALIFIERS_FILE,
    "autonomous_scopes": config.SCOPE_REGISTRY_FILE,
}


class DeclaredSourceUnavailable(RuntimeError):
    """A DECLARED release source could not be obtained as usable authoritative data.

    The base of every "this authority is unreadable" error in the repository, raised or
    subclassed on BOTH sides of the production graph (`claude_coder.data_access`'s
    `AuthoritativeDataUnavailable` family, and the `app/**` readers) so one `except`
    catches "the authority is unavailable" regardless of which tree read it.

    ABSENCE of a required source is caught upstream -- the capability manifest reports
    `missing_required` and the source-manifest gate blocks before a certificate exists.
    PRESENCE-but-unusable is what this exception exists for: a truncated, malformed or
    schema-drifted file is hashed happily by the manifest (it is there, it has bytes) and
    every one of these read paths used to swallow the parse failure into an EMPTY table --
    which for each of these sources is the PERMISSIVE answer, so corruption RELAXED the
    claim while absence blocked it.  (Codex F6-R5-A, round 6.)
    """

# Sources the claim-affecting path reads whose ABSENCE is reviewed and accepted, each with
# the justification for why absence cannot make an ineligible/unsupported claim releasable.
# This is the other half of the disposition: every registered identity is either REQUIRED
# below or exempted HERE, with a stated reason -- "optional" is never an omission nobody
# looked at.  (Codex F6-R5, round 5.)
#
# The shared bar for every entry: absence may only SHRINK what the coder can propose, and
# every proposed code still has to clear the same deterministic eligibility, entailment,
# necessity, NCCI, MUE and Excludes1 gates.  Absence is additionally recorded in the
# capability manifest (`degraded_optional`) and bound into the release fingerprint, so a
# release produced without an aid is identifiable rather than indistinguishable.
_OPTIONAL_SOURCES: dict[str, dict] = {
    "index_terms": {
        "path": config.CODES_DIR / "icd10cm_index_terms.json",
        "role": "official alphabetic index terms",
        "absence_justification":
            "term->code recall aid; absence can only remove candidates from retrieval, "
            "never admit a code that failed eligibility, entailment or validation",
    },
    "cpt_synonyms": {
        "path": config.CODES_DIR / "cpt_synonyms.json",
        "role": "synonym recall aid",
        "absence_justification":
            "retrieval-recall aid only; every candidate it can surface still has to be "
            "entailed by the note and clear every deterministic gate",
    },
    "hcpcs_synonyms": {
        "path": config.CODES_DIR / "hcpcs_synonyms.json",
        "role": "synonym recall aid",
        "absence_justification":
            "retrieval-recall aid only; every candidate it can surface still has to be "
            "entailed by the note and clear every deterministic gate",
    },
    "icd10_synonyms": {
        "path": config.CODES_DIR / "icd10_synonyms.json",
        "role": "synonym recall aid",
        "absence_justification":
            "retrieval-recall aid only; every candidate it can surface still has to be "
            "entailed by the note and clear every deterministic gate",
    },
    "snomed_crosswalk": {
        "path": config.CODES_DIR / "snomed_icd10_map.json",
        "role": "concept crosswalk",
        "absence_justification":
            "long-tail synonym/eponym recall aid requiring a UMLS licence; absence "
            "removes candidates only, and each surviving candidate is still validated",
    },
    "cpt_index_terms": {
        "path": config.CODES_DIR / "cpt_index_terms.json",
        "role": "procedure descriptor index",
        "absence_justification":
            "AMA-licensed recall aid that cannot be redistributed; absence removes "
            "candidates only and the coder falls back to descriptor/embedding retrieval",
    },
    "learned_cpt_index": {
        "path": config.CODES_DIR / "learned_cpt_index.json",
        "role": "learned resolution index",
        "absence_justification":
            "cache of previously VERIFIED resolutions; absence only forces the same "
            "resolution to be re-verified from authoritative data",
    },
    "hcpcs_drug_table": {
        "path": config.CODES_DIR / "hcpcs_drug_table.json",
        "role": "drug dosing table",
        "absence_justification":
            "drug-name recall aid and per-unit dose table; absence cannot change billed "
            "units because a documented dose with no authoritative per-unit dose HOLDS "
            "the claim (gates.drug_units_gate) instead of falling back to a count",
    },
    # --- round 6 (Codex F6-R5-A): app-side sources whose absence is reviewed ----------
    "em_mdm_grid": {
        "path": config.EM_MDM_GRID_FILE,
        "role": "AMA-licensed E/M medical-decision-making grid",
        "absence_justification":
            "AMA-licensed content a deployment may not hold; `store.mdm_grid` returns "
            "None without it and the E/M leveller then makes NO MDM-based level claim "
            "at all, so absence removes a proposal rather than admitting a higher level",
    },
    "rule_exercise": {
        "path": config.RULE_EXERCISE_FILE,
        "role": "rule-exercise telemetry for the coding memorandum",
        "absence_justification":
            "GENERATED telemetry, not an upstream authority: it records which reviewed "
            "rules a prior run exercised, and the memorandum narrates it. Absence "
            "shortens an explanatory narrative; no code, modifier, unit or gate outcome "
            "reads it",
    },
}


def _authoritative_paths() -> dict[str, Path]:
    # REVIEWED-OPTIONAL sources are deliberately NOT added here. This registry is the
    # release manifest's "every one of these files must be hashable" set -- an absent path
    # is recorded as an ERROR -- whereas a reviewed-optional aid is allowed to be absent.
    # Their presence, bytes and absence are carried by the CAPABILITY manifest (which the
    # release certificate binds) as `absent-optional` / `degraded_optional`, and when they
    # are present the codes/ sweep below still content-addresses them here.
    paths = dict(_AUTHORITATIVE)
    # Bulk discovery is a BACKSTOP for files nobody declared: skip any path an explicit
    # identity above already owns, so one file is never hashed twice under two identities
    # (which would make "which record is this file" ambiguous in the manifest).
    declared = {str(p) for p in paths.values()}
    for path in sorted(config.CODES_DIR.glob("*.json")):
        if str(path) not in declared:
            paths.setdefault(f"codes/{path.name}", path)
    for path in sorted((config.DATA_DIR / "rules").glob("*.json")):
        if str(path) not in declared:
            paths.setdefault(f"rules/{path.name}", path)
    runtime = {
        "compliance_database": config.DATA_DIR / "compliance.db",
        "validator_implementation": config.BASE_DIR / "app" / "validation" /
                                    "validator.py",
        "scrubber_implementation": config.BASE_DIR / "app" / "compliance" /
                                   "engine.py",
        "release_gate_implementation": config.BASE_DIR / "app" / "release" /
                                       "claim_readiness.py",
        "terminology_implementation": config.BASE_DIR / "app" /
                                      "terminology" / "normalizer.py",
        "submission_configuration": Path(os.getenv(
            "PRACTICE_CONFIG_PATH",
            str(config.DATA_DIR / "practice_config.json"))),
    }
    paths.update(runtime)
    return paths


def authoritative_paths() -> dict[str, Path]:
    """Public view of the authoritative source registry ({source_id: path}).

    Exposed so other release-bearing components (the coder's capability manifest) bind the
    SAME source identities and release metadata instead of maintaining a parallel list that
    can silently drift. (Codex F6-R5.)"""
    return _authoritative_paths()


def release_metadata(source_id: str) -> dict:
    """Public view of a source's release/edition/effective window."""
    return _release_metadata(source_id)


# The sources for which this module can report a REAL upstream release/edition/effective
# window (they are ingested into versioned tables that carry one).  This set is the
# authority on "does the authority publish release metadata for this source?" -- the
# required-source declaration below consults it rather than restating it.
RELEASE_METADATA_SOURCES = frozenset({
    "ncci_edits", "mue_limits", "coverage_policy",
    "icd10_codes", "cpt_codes", "hcpcs_codes",
})

# Version of the required-source SCHEMA below.  A release attestation records it, so a
# certificate built against an older/other required-source definition is identifiable
# rather than silently comparable.  Bump it whenever the required set or a role changes.
REQUIRED_SOURCE_SCHEMA_VERSION = "release-required-sources-v2"

# The COMPLETE set of release-bearing source identities a certifiable release must
# account for, each with the ROLE it plays.  Absence of any one of these means a claim
# was coded without an input that can change the claim, so a manifest that simply omits
# one (with `missing_required == []` and a self-consistent digest) must NOT certify.
#
# Identities are the registry keys of `_AUTHORITATIVE` above -- resolved through it, so
# this is a declaration of WHICH registered sources are required, never a second,
# parallel list of paths that can drift.  No medical code, code family, or descriptor
# appears here.  (Codex F6-R5.)
_REQUIRED_RELEASE_SOURCES: dict[str, dict[str, str]] = {
    "icd10_codes": {"role": "diagnosis code table"},
    "cpt_codes": {"role": "procedure code table"},
    "hcpcs_codes": {"role": "supply/device code table"},
    "ncci_edits": {"role": "procedure-pair edit policy"},
    "mue_limits": {"role": "unit-limit policy"},
    "global_periods": {
        "role": "global-period / fee-schedule policy",
        # REVIEWED EXCEPTION, stated explicitly rather than inferred from a blank field:
        # the global-period extract is not ingested into a versioned effective-window /
        # data_source_version table, so no upstream release window exists to require.
        # Its identity therefore rests entirely on the content digest, which IS required.
        "release_metadata_exemption":
            "not ingested into a versioned effective-window table; no upstream release "
            "window is published for this extract, so identity rests on the content "
            "digest alone",
    },
    # --- added in round 5 after deriving the set from the RUNTIME dependency graph ---
    "coverage_policy": {
        # Read at decision time by `data_access._coverage_map` to decide whether a service
        # is GOVERNED and which diagnoses qualify.  Absent/invalid, it silently became an
        # empty map, moving every service onto the less restrictive ungoverned path.
        "role": "coverage policy (medical-necessity linkage)",
    },
    "pfs_indicators": {
        "role": "PFS global-period / bilateral indicators",
        "release_metadata_exemption":
            "parsed extract of a CMS quarterly RVU file; it is not ingested into a "
            "versioned effective-window table, so no upstream window is queryable and "
            "identity rests on the content digest alone",
    },
    "modifier_definitions": {
        "role": "modifier definitions",
        "release_metadata_exemption":
            "not ingested into a versioned effective-window table; the file carries no "
            "queryable upstream release window, so identity rests on the content digest",
    },
    "instructional_notes": {
        "role": "tabular exclusion notes",
        "release_metadata_exemption":
            "ICD-10-CM Tabular notes ship with the code edition rather than a separate "
            "release window of their own; identity rests on the content digest",
    },
    "validator_rules": {
        "role": "deterministic validation rule pack",
        "release_metadata_exemption":
            "reviewed in-repo rule pack, not an ingested upstream publication; it carries "
            "its own pack version and its identity rests on the content digest",
    },
    "snomed_root_concepts": {
        "role": "SNOMED root concepts / confidence cap",
        "release_metadata_exemption":
            "reviewed in-repo control table, not an ingested upstream publication with an "
            "effective window; identity rests on the content digest",
    },
    "terminology_registry": {
        "role": "governed clinical terminology registry",
        "release_metadata_exemption":
            "reviewed in-repo interpretation source, not an ingested upstream publication "
            "with an effective window; identity rests on the content digest",
    },
    "necessity_relation_control": {
        "role": "necessity relation control configuration",
        "release_metadata_exemption":
            "reviewed in-repo control configuration; no external authority publishes an "
            "effective window for it, so identity rests on the content digest (its own "
            "declared control version is recorded in the audit trail, not as provenance)",
    },
    "relation_evidence_grammar": {
        "role": "directional relation-evidence grammar",
        "release_metadata_exemption":
            "reviewed in-repo control configuration; no external authority publishes an "
            "effective window for it, so identity rests on the content digest (its own "
            "declared control version is recorded in the audit trail, not as provenance)",
    },
    # --- round 6, Codex F6-R5-A: the app-side claim-affecting sources ----------------
    # Each of these was read at decision time out of a filename literal in its own
    # module.  The role recorded here is the ANSWER the source supplies; the shared
    # reason all of them are REQUIRED rather than reviewed-optional is that for each,
    # an empty/absent table is the PERMISSIVE answer -- it removes a restriction or
    # silently substitutes a different one -- so absence must fail the release rather
    # than quietly change what may be billed.
    "coding_semantics": {
        "role": "modifier-role and code-semantic-class vocabulary",
        "release_metadata_exemption":
            "reviewed in-repo control table (the code-free role vocabulary the generic "
            "coding mechanics resolve against); no external authority publishes an "
            "effective window for it, so identity rests on the content digest",
    },
    "payer_registry": {
        # Read by app.compliance.payer_registry, which the human-run 837P submission
        # step and the compliance claim builder both resolve the payer through.  Absent
        # or unreadable, EVERY note's insurance text failed to match a payer: payer_id
        # None, is_medicare False, follows_medicare_coverage False -- i.e. the note was
        # silently treated as an unrecognized commercial payer, changing which coverage
        # floor and prior-authorization policy apply.
        "role": "payer identity / coverage-floor and prior-auth routing",
        "release_metadata_exemption":
            "reviewed in-repo payer alias registry, not an ingested upstream publication "
            "with an effective window; identity rests on the content digest",
    },
    "pos_codes": {
        "role": "place-of-service codes and facility indicator",
        "release_metadata_exemption":
            "CMS place-of-service list ingested without a published effective window; "
            "identity rests on the content digest",
    },
    "modifier_exempt": {
        "role": "modifier-51 / modifier-63 exemption lists",
        "release_metadata_exemption":
            "extract of the CPT/CMS exemption appendices, not ingested into a versioned "
            "effective-window table; identity rests on the content digest",
    },
    "ncci_aoc_edits": {
        "role": "NCCI add-on-code edit pairs",
        "release_metadata_exemption":
            "the add-on-code edit file is published alongside the quarterly NCCI PTP "
            "release but is not ingested into the versioned effective-window table the "
            "PTP edits are; identity rests on the content digest",
    },
    "mce_edits": {
        "role": "Medicare Code Editor age / unacceptable-principal-diagnosis edits",
        "release_metadata_exemption":
            "parsed MCE extract, not ingested into a versioned effective-window table; "
            "identity rests on the content digest",
    },
    "icd10_chronic": {
        "role": "AHRQ chronic-condition indicator",
        "release_metadata_exemption":
            "AHRQ HCUP CCIR extract, not ingested into a versioned effective-window "
            "table; identity rests on the content digest",
    },
    "cpt_categories": {
        "role": "licensed CPT category membership",
        "release_metadata_exemption":
            "the file carries its own licensed CPT edition year, which the reader binds "
            "to the date of service; it is not ingested into the versioned "
            "effective-window table, so identity rests on the content digest",
    },
    "icd10_chapters": {
        "role": "ICD-10-CM chapter classification",
        "release_metadata_exemption":
            "ships with the ICD-10-CM code edition rather than a release window of its "
            "own; identity rests on the content digest",
    },
    "icd10_extensions": {
        "role": "ICD-10-CM 7th-character extension roles",
        "release_metadata_exemption":
            "ships with the ICD-10-CM code edition rather than a release window of its "
            "own; identity rests on the content digest",
    },
    "mac_jurisdictions": {
        "role": "MAC jurisdiction map (which contractor's coverage policy applies)",
        "release_metadata_exemption":
            "reviewed in-repo jurisdiction map, not an ingested upstream publication "
            "with an effective window; identity rests on the content digest",
    },
    "mcd_coverage_cache": {
        # Overlaid over the flat LCD seed by `_ingest_lcd`; it carries the covered-ICD
        # GROUP roles the seed lacks, i.e. the claim-composition grammar of coverage.
        "role": "parsed CMS MCD article cache (covered-ICD group roles)",
        "release_metadata_exemption":
            "a locally written parse cache of the CMS MCD bulk export; the coverage "
            "window it feeds is published by `coverage_policy`, so this identity rests "
            "on the content digest",
    },
    "descriptor_qualifiers": {
        "role": "descriptor-qualifier ontology for family arbitration",
        "release_metadata_exemption":
            "reviewed in-repo rule configuration, not an ingested upstream publication; "
            "identity rests on the content digest",
    },
    "autonomous_scopes": {
        # Not medical data: the HMAC-authenticated registry of what may be released
        # WITHOUT a human.  A substituted or truncated registry changes autonomy, so its
        # bytes belong in the same certifiable manifest as the code tables.
        "role": "authenticated autonomous operating-scope registry",
        "release_metadata_exemption":
            "human-approved in-repo operating configuration; no external authority "
            "publishes an effective window for it, so identity rests on the content "
            "digest (each scope carries its own approval dates and signature)",
    },
}


def _assert_registry_dispositioned() -> None:
    """Every EXPLICITLY registered source must be dispositioned: required, or optional
    with a written justification for why its absence cannot change a released claim.

    This is the structural half of the fix.  Round 4's required set was under-declared
    because a source could be registered and then simply not mentioned anywhere -- silence
    read as "optional".  Silence is now an error: adding a source to `_AUTHORITATIVE`
    without deciding what it is fails loudly at the first call, in every consumer.
    """
    dispositioned = set(_REQUIRED_RELEASE_SOURCES) | set(_OPTIONAL_SOURCES)
    undeclared = sorted(set(_AUTHORITATIVE) - dispositioned)
    if undeclared:
        raise RuntimeError(
            "registered authoritative source(s) with no reviewed disposition (neither "
            f"required nor exempted with a justification): {undeclared}")
    both = sorted(set(_REQUIRED_RELEASE_SOURCES) & set(_OPTIONAL_SOURCES))
    if both:
        raise RuntimeError(
            f"source(s) declared both required and optional: {both}")
    for source_id, entry in _OPTIONAL_SOURCES.items():
        if not str(entry.get("role") or "").strip():
            raise RuntimeError(f"optional source {source_id!r} declares no role")
        if not str(entry.get("absence_justification") or "").strip():
            raise RuntimeError(
                f"optional source {source_id!r} records no justification for why its "
                f"absence cannot change a released claim")
        if not entry.get("path"):
            raise RuntimeError(f"optional source {source_id!r} declares no path")


def optional_release_sources() -> dict[str, dict]:
    """{source_id -> {source_id, role, path, absence_justification}} for every source whose
    absence has been reviewed and accepted.  Raises (never returns a partial set) when the
    registry is not fully dispositioned."""
    _assert_registry_dispositioned()
    return {source_id: {"source_id": source_id,
                        "role": str(entry["role"]),
                        "path": entry["path"],
                        "absence_justification": str(entry["absence_justification"])}
            for source_id, entry in _OPTIONAL_SOURCES.items()}


def declared_source_path(source_id: str) -> Path:
    """The path of a DECLARED source, for the decision-time code that reads it.

    Readers resolve their file through this function instead of composing a filename
    literal, so a source that a claim-affecting decision depends on cannot exist outside
    the manifest: an undeclared identity raises here the first time it is read, rather
    than quietly producing a file nobody certifies.  (Codex F6-R5, round 5.)
    """
    _assert_registry_dispositioned()
    path = _AUTHORITATIVE.get(source_id)
    if path is None:
        entry = _OPTIONAL_SOURCES.get(source_id)
        path = entry["path"] if entry else None
    if path is None:
        raise RuntimeError(
            f"{source_id!r} is not a declared release source; a file read at decision "
            f"time must be registered and dispositioned in app.release.source_manifest")
    return Path(path)


def declared_document(source_id: str,
                      error: type[DeclaredSourceUnavailable]) -> dict:
    """The parsed JSON document a DECLARED source publishes, read FAIL-CLOSED.

    Unreadable, unparseable, truncated, or not-a-JSON-object all raise `error` -- the
    caller's own typed subclass, so the hold names the authority that was lost rather than
    surfacing as a bare parse error from a module no caller expects to raise.

    This is the ONE mechanic every declared decision-time read shares, and it lives HERE,
    next to the declaration, rather than in either consuming tree: round 5 put it in
    `claude_coder.data_access`, which meant `app/**`'s readers -- the other half of the
    deployed image -- could not reach it without inverting the dependency, and so kept
    re-deriving (and re-losing) fail-closed behavior in their own `try: ... except:
    return {}`.  `claude_coder.data_access.declared_document` now delegates here, so both
    trees share one implementation.  (Codex F6-R5-A, round 6.)
    """
    try:
        path = declared_source_path(source_id)
    except Exception as exc:
        # The DECLARATION itself is unresolvable -- an unregistered identity, or a
        # registry that is not fully dispositioned.  The reader cannot obtain the
        # authority, which is the same conclusion as unreadable bytes, so it travels the
        # same typed path and holds.
        raise error(f"authoritative {source_id} is not resolvable from the release-source "
                    f"declaration: {exc}") from exc
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise error(f"authoritative {source_id} unreadable at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise error(f"authoritative {source_id} at {path} is not a JSON object "
                    f"(got {type(payload).__name__})")
    return payload


def declared_table(source_id: str, key: str,
                   error: type[DeclaredSourceUnavailable]) -> dict:
    """The NON-EMPTY mapping a declared source publishes under `key`.

    Non-empty is part of the contract, not a nicety: a document that parses but carries no
    table (wrong schema, truncated write, an extract whose builder failed) yields exactly
    the same `{}` the swallowed-exception path used to yield, and `{}` is the permissive
    answer for every one of these sources.
    """
    payload = declared_document(source_id, error)
    table = payload.get(key)
    if not isinstance(table, dict) or not table:
        raise error(f"authoritative {source_id} at {declared_source_path(source_id)} "
                    f"publishes no non-empty {key!r} table")
    return table


def required_release_sources() -> dict[str, dict]:
    """{source_id -> {source_id, role, path, release_metadata_required,
    release_metadata_exemption}} for every source a certifiable release must account for.

    Fails LOUDLY (raises) rather than degrading to a partial set when the declaration and
    the authority disagree:
      - a required identity that is not registered in `_AUTHORITATIVE`;
      - a source the authority publishes release metadata for that nonetheless carries a
        (now stale) reviewed exemption;
      - a source the authority publishes NO release metadata for that carries no reviewed
        exemption -- silence is not an exemption;
      - a registered source that is neither required nor exempted with a justification.
    Callers treat a raise as "not certifiable" / manifest unavailable, never as "empty".
    """
    _assert_registry_dispositioned()
    spec: dict[str, dict] = {}
    for source_id, declared in _REQUIRED_RELEASE_SOURCES.items():
        path = _AUTHORITATIVE.get(source_id)
        if path is None:
            raise RuntimeError(
                f"required release source {source_id!r} is not registered in the "
                f"authoritative-source registry")
        exemption = str(declared.get("release_metadata_exemption") or "").strip()
        provides = source_id in RELEASE_METADATA_SOURCES
        if provides and exemption:
            raise RuntimeError(
                f"required release source {source_id!r} carries a release-metadata "
                f"exemption but the authority publishes release metadata for it")
        if not provides and not exemption:
            raise RuntimeError(
                f"required release source {source_id!r} has no published release "
                f"metadata and no reviewed exemption recorded")
        spec[source_id] = {"source_id": source_id,
                           "role": str(declared["role"]),
                           "path": path,
                           "release_metadata_required": provides,
                           "release_metadata_exemption": exemption}
    return spec


def release_window_populated(release) -> bool:
    """True when a manifest record carries a REAL upstream effective/edition window.

    An ingest timestamp (`version`) is not an authority edition, so it does not satisfy
    this on its own: the window is what makes a claim date checkable against the loaded
    release.  A blank/absent release block on a source the authority publishes metadata
    for means the manifest lost provenance it was supposed to carry.
    """
    if not isinstance(release, dict):
        return False
    return any(str(release.get(key) or "").strip()
               for key in ("effective_from", "release_effective_from"))


def sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=256)
def _sha256_cached(path_text: str, mtime_ns: int, size: int) -> str:
    """Hash once per immutable filesystem identity; edits invalidate it."""
    path = Path(path_text)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_source_manifest() -> dict:
    records = []
    errors = _checkpoint_database()
    for source_id, path in _authoritative_paths().items():
        try:
            stat = path.stat()
            try:
                display_path = str(path.relative_to(config.BASE_DIR))
            except ValueError:
                display_path = f"external/{path.name}"
            records.append({"source_id": source_id,
                            "path": display_path,
                            "sha256": sha256_file(path),
                            "size": stat.st_size,
                            **_release_metadata(source_id)})
        except Exception as exc:
            errors.append(f"{source_id}: {exc}")
    errors.extend(_database_source_errors())
    body = {"records": records, "errors": errors}
    body["fingerprint"] = manifest_fingerprint(body)
    return body


def _checkpoint_database() -> list[str]:
    """Canonicalize committed WAL frames before hashing compliance.db.

    This is a storage checkpoint, not a data change.  If another writer owns
    the WAL, the manifest fails closed instead of hashing a stale main file.
    """
    db_path = config.DATA_DIR / "compliance.db"
    if not db_path.exists():
        return ["compliance_database: absent"]
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        busy, frames, checkpointed = conn.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        conn.close()
    except sqlite3.Error as exc:
        return [f"compliance_database checkpoint failed: {exc}"]
    if busy or frames != checkpointed:
        return ["compliance_database checkpoint incomplete; concurrent writer active"]
    return []


def _database_source_errors() -> list[str]:
    """Detect a database built from different source-file identities."""
    db_path = config.DATA_DIR / "compliance.db"
    if not db_path.exists():
        return ["compliance_database: absent"]
    sources = {
        "icd10_codes": [config.ICD10_FILE],
        "cpt_codes": [config.CPT_FILE],
        "hcpcs_codes": [config.HCPCS_FILE],
        "ncci": [config.NCCI_FILE],
        "mue": [config.MUE_FILE],
        "lcd": [config.LCD_FILE, config.MCD_COVERAGE_CACHE_FILE],
    }

    def identity(paths: list[Path]) -> str:
        parts = []
        for path in paths:
            try:
                stat = path.stat()
                parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
            except OSError:
                parts.append(f"{path.name}:missing")
        return "|".join(parts)

    try:
        conn = sqlite3.connect(db_path, timeout=30)
        rows = conn.execute(
            "SELECT source_id, fingerprint FROM data_file_fingerprint"
        ).fetchall()
        busy, frames, checkpointed = conn.execute(
            "PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        conn.close()
    except sqlite3.Error as exc:
        return [f"compliance_database provenance unavailable: {exc}"]
    recorded = dict(rows)
    errors = [
        f"compliance_database source mismatch: {source_id}"
        for source_id, paths in sources.items()
        if recorded.get(source_id) != identity(paths)
    ]
    if busy or frames != checkpointed or frames:
        errors.append("compliance_database changed while manifest was built")
    return errors


def manifest_fingerprint(manifest: dict) -> str:
    body = {"records": manifest.get("records") or [],
            "errors": manifest.get("errors") or []}
    return "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def valid_record(record: dict) -> bool:
    """Validate the minimum identity carried by every source record."""
    return bool(record.get("source_id") and record.get("path") and
                isinstance(record.get("size"), int) and record["size"] >= 0 and
                _SHA256_RE.fullmatch(str(record.get("sha256") or "")))


def _release_metadata(source_id: str) -> dict:
    """Effective/version metadata from the database built from the source.

    Whole-file checksums identify the exact bytes. These fields explain which
    date window/version those bytes represent without loading very large JSON
    sources into memory a second time.
    """
    if source_id not in RELEASE_METADATA_SOURCES:
        return {"effective_from": "", "effective_to": "", "version": ""}
    db_path = config.DATA_DIR / "compliance.db"
    if not db_path.exists():
        return {"effective_from": "", "effective_to": "", "version": ""}
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        row = None
        version_row = None
        if source_id == "ncci_edits":
            row = conn.execute(
                "SELECT MIN(effective_from), MAX(effective_to) FROM ncci_ptp"
            ).fetchone()
        elif source_id == "mue_limits":
            row = conn.execute(
                "SELECT MIN(effective_from), MAX(effective_to) FROM mue"
            ).fetchone()
        elif source_id == "coverage_policy":
            # The LCD/Article coverage window, from the policies whose dates came from the
            # authority itself (temporal_authority=1) -- policies with no published dates
            # carry placeholders and would report a vacuous window.
            row = conn.execute(
                "SELECT MIN(effective_from), MAX(effective_to) FROM coverage_policy "
                "WHERE temporal_authority=1 AND effective_from != ''").fetchone()
        elif source_id in {"icd10_codes", "cpt_codes", "hcpcs_codes"}:
            system = {"icd10_codes": "ICD10", "cpt_codes": "CPT",
                      "hcpcs_codes": "HCPCS"}[source_id]
            row = conn.execute(
                "SELECT MIN(effective_from), MAX(effective_to) FROM code_set "
                "WHERE code_system=?", (system,)).fetchone()
        else:
            row = None
        try:
            seed_id = {
                "ncci_edits": "seed:ncci", "mue_limits": "seed:mue",
                "coverage_policy": "seed:lcd",
                "icd10_codes": "seed:icd10_codes",
                "cpt_codes": "seed:cpt_codes",
                "hcpcs_codes": "seed:hcpcs_codes",
            }.get(source_id, "")
            version_row = conn.execute(
                "SELECT effective_from, ingested_at FROM data_source_version "
                "WHERE source_id IN (?, ?, ?) ORDER BY ingested_at DESC LIMIT 1",
                (source_id, source_id.removesuffix("_edits"), seed_id),
            ).fetchone()
        except sqlite3.Error:
            # Older stores may predate the refresh-version registry.  That
            # must not discard effective bounds successfully read above.
            version_row = None
        metadata = {
            "effective_from": str((row or ("", ""))[0] or ""),
            "effective_to": str((row or ("", ""))[1] or ""),
            "version": ("/".join(str(v or "") for v in version_row)
                        if version_row else ""),
        }
        metadata.update(_edition_release_window(source_id))
        return metadata
    except (OSError, sqlite3.Error):
        return {"effective_from": "", "effective_to": "", "version": ""}
    finally:
        if conn is not None:
            conn.close()


def _quarter_window(year: int, month: int) -> tuple[str, str]:
    quarter_month = ((month - 1) // 3) * 3 + 1
    end_month = quarter_month + 2
    return (date(year, quarter_month, 1).isoformat(),
            date(year, end_month, calendar.monthrange(year, end_month)[1]).isoformat())


def _edition_release_window(source_id: str) -> dict:
    """Claim-date coverage of the exact licensed/published snapshot.

    Code lifecycle dates and release freshness are separate concepts.  A
    decades-old HCPCS code may remain active, while an April quarterly file
    is still insufficient authority for an August claim.  The manifest
    records both so the release certificate can enforce the latter.
    """
    try:
        if source_id == "cpt_codes":
            data = json.loads(config.CPT_FILE.read_text())
            year = int((data.get("metadata") or {}).get("year"))
            return {"release_effective_from": date(year, 1, 1).isoformat(),
                    "release_effective_to": date(year, 12, 31).isoformat(),
                    "release_basis": "licensed CPT edition"}
        if source_id == "icd10_codes":
            rows = json.loads(config.ICD10_FILE.read_text())
            fy = int(next(str(row.get("fy")) for row in rows if row.get("fy")))
            return {"release_effective_from": date(fy - 1, 10, 1).isoformat(),
                    "release_effective_to": date(fy, 9, 30).isoformat(),
                    "release_basis": "ICD-10-CM fiscal-year edition"}
        if source_id == "hcpcs_codes":
            rows = json.loads(config.HCPCS_FILE.read_text())
            source_name = str(((rows[0].get("metadata") or {}).get("source_file")))
            match = re.search(r"(20\d{2})[_-]?(JAN|APR|JUL|OCT)", source_name,
                              re.IGNORECASE)
            if not match:
                match = re.search(r"(JAN|APR|JUL|OCT)[_-]?(20\d{2})", source_name,
                                  re.IGNORECASE)
                if match:
                    month_name, year_text = match.groups()
                else:
                    return {}
            else:
                year_text, month_name = match.groups()
            month = {"JAN": 1, "APR": 4, "JUL": 7, "OCT": 10}[month_name.upper()]
            start, end = _quarter_window(int(year_text), month)
            return {"release_effective_from": start,
                    "release_effective_to": end,
                    "release_basis": "CMS quarterly HCPCS release"}
    except (OSError, ValueError, TypeError, StopIteration, IndexError):
        return {}
    return {}
