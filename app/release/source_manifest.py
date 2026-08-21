"""Checksummed provenance for authoritative coding inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import calendar
from datetime import date
import time
from pathlib import Path
from typing import Any, Callable

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


# --- round 7 (directive section 6): the RUNTIME / DERIVED decision-time inputs --------
# Every entry here is read at DECISION time exactly like the JSON tables above -- most
# importantly `compliance.db`, the compiled database `CodeReferenceDB.check_ncci()` answers
# every live NCCI question from.  All six were NAMED by `_authoritative_paths()` (so
# `build_source_manifest` hashed them) but never DISPOSITIONED, so `required_release_sources`
# / `optional_release_sources` omitted them -- and `claude_coder.capability.build_manifest()`,
# the manifest the RELEASE CERTIFICATE is built over, therefore never probed them at all.
# The certificate attested to raw JSON bytes while the decision was answered by a compiled
# database nobody hashed: certified bytes != read bytes, the same class round 5/6 closed for
# the JSON sources.  (Codex F6-R5-A remainder.)
#
# Resolved through CALLABLES rather than a literal dict for a reason that is not stylistic:
# `submission_configuration` is env-overridable (`PRACTICE_CONFIG_PATH`) and must stay
# late-bound, and no path may freeze whatever `config.DATA_DIR` happened to be at import.
_RUNTIME_SOURCES: dict[str, Callable[[], Path]] = {
    "compliance_database": lambda: config.DATA_DIR / "compliance.db",
    "validator_implementation":
        lambda: config.BASE_DIR / "app" / "validation" / "validator.py",
    "scrubber_implementation":
        lambda: config.BASE_DIR / "app" / "compliance" / "engine.py",
    "release_gate_implementation":
        lambda: config.BASE_DIR / "app" / "release" / "claim_readiness.py",
    "terminology_implementation":
        lambda: config.BASE_DIR / "app" / "terminology" / "normalizer.py",
    "submission_configuration": lambda: Path(os.getenv(
        "PRACTICE_CONFIG_PATH", str(config.DATA_DIR / "practice_config.json"))),
}


def _runtime_paths() -> dict[str, Path]:
    """The runtime/derived source identities, resolved NOW (never frozen at import)."""
    return {source_id: Path(resolve())
            for source_id, resolve in _RUNTIME_SOURCES.items()}


def _declared_registry() -> dict[str, Path]:
    """Every EXPLICITLY declared source identity: the reviewed data tables AND the
    runtime/derived inputs.

    This -- not `_AUTHORITATIVE` alone -- is the set the disposition invariant covers, the
    set `declared_source_path` resolves against, and the set `required_release_sources`
    binds its paths from.  Keeping the runtime block outside that invariant is exactly how
    `compliance.db` came to be hashed by one manifest and absent from the other.
    """
    registry = dict(_AUTHORITATIVE)
    registry.update(_runtime_paths())
    return registry


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
    "snomed_concept_terms": {
        "path": config.CODES_DIR / "snomed_concept_terms.json",
        "role": "SNOMED CT Body Structure concept identity and hierarchy",
        "absence_justification":
            "UMLS-licensed concept graph used only to resolve whether two OPEN-"
            "vocabulary anatomy values are the same, an ancestor/descendant pair, or "
            "genuinely disjoint concept; absence degrades the axis comparison to its "
            "existing conservative identity-or-ambiguous behavior (no confirmed "
            "difference or match is ever asserted from lexical shape alone), never to "
            "a wrong or fabricated relation",
    },
    "learned_cpt_index": {
        "path": config.CODES_DIR / "learned_cpt_index.json",
        "role": "learned resolution index",
        "absence_justification":
            "cache of previously VERIFIED resolutions; absence only forces the same "
            "resolution to be re-verified from authoritative data",
    },
    "cpt_verified_synonyms": {
        "path": config.CODES_DIR / "cpt_verified_synonyms.json",
        "role": "round-trip-validated CPT procedure synonym terms (issue #6, "
               "compiled-semantic-layer plan item 3)",
        "absence_justification":
            "each candidate term is llm-generated (cpt_synonyms.json's own "
            "provenance: 'RETRIEVAL AID ONLY -- NOT an authoritative source'), then "
            "VALIDATED by tools/verify_cpt_synonyms.py -- kept only when it "
            "independently retrieves its own originating code through the same "
            "authoritative retrieval index every other candidate lookup uses; "
            "absence only means fewer verified alternate phrasings are offered for "
            "the procedure governed-terminology axis, never a wrong or fabricated "
            "match",
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
    # --- round 7, directive section 6: the RETRIEVAL INDEX identity -----------------
    # The last production filename literal that reached the release fingerprint without a
    # declaration.  `data_fingerprint` copies these bytes into the certificate as
    # `codes_checksum`, and the ClaimBundle carries them as `AuthorityBinding.index_checksum`
    # -- an attested value composed from a path two modules spelled out by hand.
    "retrieval_index_checksum": {
        "path": config.DATA_DIR / "qdrant_store" / "codes_checksum.txt",
        "role": "content checksum of the built semantic retrieval index",
        "absence_justification":
            "the semantic index is a CANDIDATE-GENERATION aid: absence removes candidates "
            "from retrieval and can never admit one, because every candidate it surfaces "
            "still has to be entailed by the note and clear every deterministic gate. It "
            "is also a deployment-built artifact rather than a repository input, so a "
            "clean build legitimately has none until the index is built",
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
    # The runtime/derived inputs belong to the DECLARED registry now rather than being
    # appended here afterwards: appending them in this one function (and nowhere else) is
    # what let them be hashed by THIS manifest while staying invisible both to the
    # disposition invariant and to the certificate's own manifest.  (Directive section 6.)
    paths = _declared_registry()
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
REQUIRED_SOURCE_SCHEMA_VERSION = "release-required-sources-v3"

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
    # --- round 7, directive section 6: the RUNTIME / DERIVED inputs -------------------
    # `authoritative_paths()` already NAMED all six; not one was dispositioned, so the
    # capability manifest -- and therefore the release certificate -- carried none of
    # them.  All six are REQUIRED on the same bar as the tables above: for each, absence
    # or substitution changes what may be billed while changing nothing the certificate
    # could see.  None of them is an upstream publication, so each records why its
    # identity rests on its content digest rather than a release window.
    "compliance_database": {
        # THE finding.  `CodeReferenceDB.check_ncci()` answers every live NCCI PTP
        # question out of this compiled database (2.6M rows, deliberately never held in
        # memory), and `ComplianceDataStore` answers coverage, global-period, MUE and
        # code-existence questions out of it too.  Absent or empty, `check_ncci` returned
        # None -- which every caller reads as "no edit", the PERMISSIVE answer -- while
        # the certificate went on hashing the raw NCCI JSON the decision never opened.
        "role": "compiled code/edit/coverage database queried at decision time",
        "release_metadata_exemption":
            "a DERIVED build product, not an upstream publication: the effective windows "
            "it serves belong to the JSON sources it was compiled from, which publish "
            "them and are separately required here, so its own identity rests on the "
            "content digest of the exact database bytes that answered the query",
    },
    "validator_implementation": {
        "role": "deterministic claim-validation implementation",
        "release_metadata_exemption":
            "release-bearing executable code, not an ingested publication; identity "
            "rests on the content digest of the module the deployment actually ran",
    },
    "scrubber_implementation": {
        "role": "compliance-scrub implementation",
        "release_metadata_exemption":
            "release-bearing executable code, not an ingested publication; identity "
            "rests on the content digest of the module the deployment actually ran",
    },
    "release_gate_implementation": {
        "role": "claim-readiness release-gate implementation",
        "release_metadata_exemption":
            "release-bearing executable code, not an ingested publication; identity "
            "rests on the content digest of the module the deployment actually ran",
    },
    "terminology_implementation": {
        "role": "clinical-terminology normalization implementation",
        "release_metadata_exemption":
            "release-bearing executable code, not an ingested publication; identity "
            "rests on the content digest of the module the deployment actually ran",
    },
    "submission_configuration": {
        # The billing entity, NPIs and payer routing the 837P is built from.  It is
        # env-overridable (`PRACTICE_CONFIG_PATH`), so the identity that matters is the
        # bytes at the path THIS deployment resolved, not a repo default.
        "role": "billing-entity / submission configuration for the professional claim",
        "release_metadata_exemption":
            "deployment-owned configuration resolved per environment, not an ingested "
            "upstream publication with an effective window; identity rests on the "
            "content digest of the file the deployment resolved",
    },
}


#: Declared sources whose bytes are PARSED INTO MEMORY once, eagerly, and then answered
#: from memory for the rest of the batch (`CodeReferenceDB.load_all`).  For these the file
#: on disk at certification time is NOT what answered anything -- the parsed copy is -- so a
#: certifiable release must carry the identity captured AT THAT PARSE.  Absence of a binding
#: is itself an integrity error: "nobody identified the bytes that became the in-memory
#: table" is not a clean result.  (Codex F6-R5-B.)
#:
#: Lazily-read sources (coverage policy, tabular notes, PFS indicators, ...) are bound the
#: same way WHEN they are read, but cannot be required here: an encounter that never had to
#: consult one legitimately has no binding for it.
#:
#: These are DECLARED SOURCE IDENTITIES, not medical codes: no code, code family, prefix
#: range or descriptor appears here.
SNAPSHOT_BOUND_SOURCES = (
    "icd10_codes", "cpt_codes", "hcpcs_codes", "mue_limits", "snomed_root_concepts")


def _assert_registry_dispositioned() -> None:
    """Every EXPLICITLY registered source must be dispositioned: required, or optional
    with a written justification for why its absence cannot change a released claim.

    This is the structural half of the fix.  Round 4's required set was under-declared
    because a source could be registered and then simply not mentioned anywhere -- silence
    read as "optional".  Silence is now an error: adding a source to `_AUTHORITATIVE`
    without deciding what it is fails loudly at the first call, in every consumer.
    """
    dispositioned = set(_REQUIRED_RELEASE_SOURCES) | set(_OPTIONAL_SOURCES)
    # `_declared_registry()`, not `_AUTHORITATIVE`: the runtime/derived inputs are
    # declared identities too, and leaving them outside this check is precisely how
    # `compliance_database` stayed undispositioned while being named by the registry.
    undeclared = sorted(set(_declared_registry()) - dispositioned)
    if undeclared:
        raise RuntimeError(
            "registered authoritative source(s) with no reviewed disposition (neither "
            f"required nor exempted with a justification): {undeclared}")
    both = sorted(set(_REQUIRED_RELEASE_SOURCES) & set(_OPTIONAL_SOURCES))
    if both:
        raise RuntimeError(
            f"source(s) declared both required and optional: {both}")
    # A source whose in-memory parse must be BOUND has to be a required release source:
    # the certificate compares the captured identity against the manifest RECORD for that
    # source, and only a required source is guaranteed to have one. (Codex F6-R5-B.)
    unbindable = sorted(set(SNAPSHOT_BOUND_SOURCES) - set(_REQUIRED_RELEASE_SOURCES))
    if unbindable:
        raise RuntimeError(
            "source(s) whose in-memory snapshot must be bound are not declared required "
            f"release sources, so no manifest record can be compared to them: {unbindable}")
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
    path = _declared_registry().get(source_id)
    if path is None:
        entry = _OPTIONAL_SOURCES.get(source_id)
        path = entry["path"] if entry else None
    if path is None:
        raise RuntimeError(
            f"{source_id!r} is not a declared release source; a file read at decision "
            f"time must be registered and dispositioned in app.release.source_manifest")
    return Path(path)


def read_declared_snapshot(source_id: str,
                           error: type[DeclaredSourceUnavailable]) -> tuple[bytes, dict]:
    """(exact bytes, content identity) of a DECLARED source, read through ONE open handle.

    The generalisation of `open_database_snapshot` to every source whose content is parsed
    into memory (Codex F6-R5-B).  The compiled database is queried through a descriptor the
    caller keeps open, so pinning the inode is what makes its identity durable.  A JSON
    source is different in a way that makes the binding STRONGER: it is read exactly once,
    in full, and every later answer comes from the parsed copy -- so the identity is taken
    from the very buffer that is about to be parsed.  The bytes that produced the in-memory
    table and the bytes the identity describes are then not merely "the same file", they are
    the same object; no reopen, restat or re-hash sits between them for a replacement to
    slip through.

    A file rewritten WHILE it is being read raises rather than returning a digest that
    describes no single state of it: `read()` can straddle a non-atomic writer, and half of
    one edition plus half of another is not a snapshot anything may be certified against.
    (An atomic `os.replace` refresh -- what every builder under `tools/` already does -- can
    never produce that: the reader keeps the unlinked inode it opened.)

    Raises the caller's typed error for an unresolvable declaration, unreadable bytes, or a
    concurrent rewrite -- never a partial identity, and never an empty table, which is the
    permissive answer for every one of these sources.
    """
    try:
        path = declared_source_path(source_id)
    except Exception as exc:
        raise error(f"authoritative {source_id} is not resolvable from the release-source "
                    f"declaration: {exc}") from exc
    started = time.time_ns()
    try:
        with open(path, "rb") as handle:
            opened = os.fstat(handle.fileno())
            payload = handle.read()
            settled = os.fstat(handle.fileno())
    except OSError as exc:
        raise error(f"authoritative {source_id} unreadable at {path}: {exc}") from exc
    if _file_key(settled) != _file_key(opened):
        raise error(
            f"authoritative {source_id} at {path} was being rewritten while it was read; "
            f"no single set of bytes can be bound to this encounter")
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    if settled.st_mtime_ns + _RACY_MTIME_WINDOW_NS < started:
        # The same cache, and the same racily-clean rule, `sha256_file` uses -- so the
        # manifest's own re-hash of this path is served from the digest of the bytes that
        # were actually parsed instead of costing a second full read per source per process.
        _DIGEST_CACHE[str(path)] = (_file_key(settled), digest)
    return payload, {"source_id": source_id, "path": str(path),
                     "sha256": digest, "size": int(settled.st_size)}


def declared_json_snapshot(source_id: str,
                           error: type[DeclaredSourceUnavailable]) -> tuple[Any, dict]:
    """(parsed JSON of ANY shape, identity of the exact bytes parsed).

    Any shape because the authoritative code tables are JSON ARRAYS, not objects, and they
    need the identical binding the object-shaped sources get.
    """
    payload, identity = read_declared_snapshot(source_id, error)
    try:
        return json.loads(payload), identity
    except Exception as exc:
        raise error(f"authoritative {source_id} unreadable at {identity['path']}: "
                    f"{exc}") from exc


def declared_document_snapshot(source_id: str,
                               error: type[DeclaredSourceUnavailable]) -> tuple[dict, dict]:
    """(parsed JSON object, identity of the exact bytes parsed) — see `declared_document`."""
    document, identity = declared_json_snapshot(source_id, error)
    if not isinstance(document, dict):
        raise error(f"authoritative {source_id} at {identity['path']} is not a JSON object "
                    f"(got {type(document).__name__})")
    return document, identity


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

    Callers whose result is CACHED and answers later decisions from memory must use
    `declared_document_snapshot` instead and bind the identity it returns: the certificate
    otherwise attests to whatever is at the path when it is built, which is a different
    fact about a different moment. (Codex F6-R5-B.)
    """
    return declared_document_snapshot(source_id, error)[0]


def declared_table_snapshot(source_id: str, key: str,
                            error: type[DeclaredSourceUnavailable]) -> tuple[dict, dict]:
    """(the non-empty mapping under `key`, identity of the exact bytes parsed)."""
    payload, identity = declared_document_snapshot(source_id, error)
    table = payload.get(key)
    if not isinstance(table, dict) or not table:
        raise error(f"authoritative {source_id} at {identity['path']} "
                    f"publishes no non-empty {key!r} table")
    return table, identity


def declared_table(source_id: str, key: str,
                   error: type[DeclaredSourceUnavailable]) -> dict:
    """The NON-EMPTY mapping a declared source publishes under `key`.

    Non-empty is part of the contract, not a nicety: a document that parses but carries no
    table (wrong schema, truncated write, an extract whose builder failed) yields exactly
    the same `{}` the swallowed-exception path used to yield, and `{}` is the permissive
    answer for every one of these sources.
    """
    return declared_table_snapshot(source_id, key, error)[0]


class SourceSnapshotSet:
    """The content identities the in-memory authoritative data was actually parsed from.

    One of these travels with each object that PARSES a declared source and then answers
    decisions from the parsed copy (`CodeReferenceDB` for the code/edit tables,
    `AuthoritativeSource` for the lazily-read policy and rule documents).  They are merged
    at certification time and every one of them must equal the manifest's own record, which
    is what turns "a source was replaced after it was loaded" from a silent pass into a
    hold. (Codex F6-R5-B.)

    Deliberately NOT a process-global registry: the binding must belong to the objects that
    hold the parsed data, so its lifetime is exactly the data's lifetime -- a new load
    legitimately begins a new snapshot, and nothing outside those objects can accumulate
    stale identities that later encounters would then have to explain.

    Two different sets of bytes bound for the SAME source is recorded as a conflict rather
    than silently last-one-wins: the decisions may have come from either, so the claim is
    bound to no single snapshot and must hold.
    """

    def __init__(self) -> None:
        self._bound: dict[str, dict] = {}
        self._conflicts: list[str] = []

    def bind(self, identity: dict) -> dict:
        """Record the identity captured at a parse; returns it unchanged for chaining."""
        record = identity if isinstance(identity, dict) else {}
        source_id = str(record.get("source_id") or "").strip()
        if not source_id or not is_content_digest(record.get("sha256")):
            self._conflicts.append(
                "a declared source was parsed into memory without a usable content "
                f"identity ({record!r}); the certificate cannot name the bytes it used")
            return identity
        prior = self._bound.get(source_id)
        if prior is None:
            self._bound[source_id] = dict(record)
        elif prior.get("sha256") != record.get("sha256"):
            self._conflicts.append(
                f"{source_id}: two different sets of bytes were parsed into memory while "
                f"this claim was being coded ({prior.get('sha256')} then "
                f"{record.get('sha256')}); no single snapshot can be certified for it")
        return identity

    def merge(self, other: "SourceSnapshotSet | None") -> "SourceSnapshotSet":
        if other is None:
            return self
        for identity in other.identities.values():
            self.bind(identity)
        self._conflicts.extend(other.conflicts)
        return self

    @property
    def identities(self) -> dict[str, dict]:
        return {source_id: dict(identity)
                for source_id, identity in self._bound.items()}

    @property
    def conflicts(self) -> list[str]:
        return list(self._conflicts)


def source_snapshot_drift(bound: dict) -> list[str]:
    """Every reason a declared source on disk is no longer the snapshot `bound` names.

    The certify-time half of `read_declared_snapshot`, and the exact analogue of
    `database_snapshot_drift` for the sources that are held in memory instead of queried.
    `bound` is `{source_id -> identity captured when the bytes were parsed}`; this
    re-derives each identity from the file the certificate is about to attest to.  Any
    difference means the decisions and the attestation describe two different editions,
    which is a HOLD -- and it must be detectable HERE, because nothing obliges a second
    read to happen and notice.  (Codex F6-R5-B.)

    Returns a list rather than raising: its caller composes a manifest's `integrity_errors`
    and must report every problem at once.
    """
    if not isinstance(bound, dict):
        return ["source snapshots: no content identities were bound when the authoritative "
                "data was parsed, so the certificate cannot be shown to describe the data "
                "the decisions were made against"]
    errors: list[str] = []
    for source_id in sorted(bound):
        identity = bound[source_id]
        if not isinstance(identity, dict) or not is_content_digest(identity.get("sha256")):
            errors.append(
                f"{source_id}: no content identity was bound when it was parsed into "
                f"memory, so the certificate cannot be shown to describe the data the "
                f"decisions were made against")
            continue
        try:
            path = declared_source_path(str(source_id))
            current = sha256_file(path)
            size = int(path.stat().st_size)
        except Exception as exc:
            errors.append(f"{source_id}: the bound snapshot can no longer be "
                          f"re-identified ({exc})")
            continue
        changed = [label for field, label, value in
                   (("sha256", "content digest", current), ("size", "size in bytes", size))
                   if identity.get(field) != value]
        if changed:
            errors.append(
                f"{source_id}: replaced or rewritten after it was parsed into memory for "
                f"this encounter ({', '.join(changed)} differ from the bound snapshot); the "
                f"certificate would attest to bytes no decision was made from")
    return errors


class CompiledDatabaseUnavailable(DeclaredSourceUnavailable):
    """The compiled compliance database cannot serve as the certified authority.

    Missing, unopenable, truncated/malformed, schema-drifted, EMPTY in a table a decision
    is answered from, compiled from source bytes other than the ones on disk now, or
    swapped underneath a process that already bound it -- every one of those raises this
    instead of resolving to the shape each reader treats as "nothing to report".

    For every decision table below, EMPTY is the PERMISSIVE answer -- "no NCCI edit", "no
    unit limit", "no coverage policy governs this", "this code does not exist" -- so an
    empty required table is an integrity failure, never a lookup miss.  This is the same
    class of hole `declared_document` / `declared_table` closed for the JSON sources; it
    stayed open for the compiled database because no reader resolved through the
    declaration at all.  (Directive section 6 / Codex F6-R5-A remainder.)
    """


#: Tables a claim decision is ANSWERED from, and which must therefore be present AND
#: non-empty for the database to be usable as the certified authority.  These are SCHEMA
#: names -- no medical code, code family, prefix range or descriptor appears here, so the
#: no-hardcoded-codes rule is satisfied by construction rather than by exemption.  Kept in
#: ONE place so the certificate's integrity check and the readers' fail-closed check can
#: never disagree about what "usable" means.
REQUIRED_DATABASE_TABLES = ("code_set", "ncci_ptp", "mue", "coverage_policy")

#: {`data_file_fingerprint` row id -> the DECLARED sources that row must describe}.  The
#: row ids are the database's own schema; the paths are resolved through the declaration
#: so the staleness check and the manifest can never compare different files.
_DATABASE_SOURCE_ROWS: dict[str, tuple[str, ...]] = {
    "icd10_codes": ("icd10_codes",),
    "cpt_codes": ("cpt_codes",),
    "hcpcs_codes": ("hcpcs_codes",),
    "ncci": ("ncci_edits",),
    "mue": ("mue_limits",),
    "lcd": ("coverage_policy", "mcd_coverage_cache"),
}


def compliance_database_path() -> Path:
    """The compiled database's path, resolved through the DECLARATION.

    Every decision-time reader calls this instead of composing `DATA_DIR / "compliance.db"`
    itself, for the same reason the JSON readers call `declared_source_path`: a file a
    claim depends on cannot exist outside the manifest.  (Directive section 6.)
    """
    return declared_source_path("compliance_database")


def compliance_database_identity() -> dict:
    """{source_id, path, sha256, size} -- the exact BYTES of the compiled database.

    This is what the ClaimBundle binds.  Row counts and ingest timestamps cannot tell two
    materially different databases apart, and the raw-JSON digests the certificate already
    carried describe files the NCCI lookup never opened.  Raises rather than returning a
    partial identity: a release must never be certified against a database nobody can name.
    """
    path = compliance_database_path()
    try:
        stat = path.stat()
        digest = sha256_file(path)
    except OSError as exc:
        raise CompiledDatabaseUnavailable(
            f"compiled compliance database unreadable at {path}: {exc}") from exc
    wal_digest, wal_size = _wal_identity(path)
    return {"source_id": COMPLIANCE_DATABASE_SOURCE_ID, "path": str(path),
            "sha256": digest, "size": int(stat.st_size),
            "wal_sha256": wal_digest, "wal_size": wal_size}


def open_database_snapshot() -> tuple[Any, dict]:
    """(open handle, content identity) for the compiled database, identified THROUGH the
    handle returned -- the snapshot a claim's decisions are answered from.

    A filesystem stat tuple identifies a file only for as long as somebody keeps LOOKING at
    it.  A database replaced after an encounter's last query and before its certificate is
    built is never re-stat'ed by anyone, so the certificate ends up attesting to the digest
    of a file that answered nothing -- and a stat tuple is not a content identity in the
    first place: a same-size in-place write can leave (device, inode, size, mtime) untouched
    (see `_RACY_MTIME_WINDOW_NS`, observed on this deployment's own filesystem).

    The identity is therefore taken from the BYTES, read through a descriptor the caller
    holds open: the open handle pins the inode, so no later file can inherit its
    (device, inode) and impersonate it, and the digest describes exactly what the
    connections opened alongside it are reading.  Raises rather than returning a partial
    identity -- a release must never be certified against a database nobody can name.
    (Codex F6-R5-A.)
    """
    path = compliance_database_path()
    try:
        handle = open(path, "rb")
    except OSError as exc:
        raise CompiledDatabaseUnavailable(
            f"compiled compliance database unreadable at {path}: {exc}") from exc
    started = time.time_ns()
    try:
        opened = os.fstat(handle.fileno())
        digest = _sha256_handle(handle)
        settled = os.fstat(handle.fileno())
        wal_digest, wal_size = _wal_identity(path)
    except OSError as exc:
        handle.close()
        raise CompiledDatabaseUnavailable(
            f"compiled compliance database unreadable at {path}: {exc}") from exc
    if _file_key(settled) != _file_key(opened):
        # Rewritten WHILE it was being identified: this digest describes no single state of
        # it, so there is no snapshot to bind the encounter to.
        handle.close()
        raise CompiledDatabaseUnavailable(
            f"the compiled compliance database at {path} was being rewritten while it was "
            f"identified; no single set of bytes can be bound to this encounter")
    if settled.st_mtime_ns + _RACY_MTIME_WINDOW_NS < started:
        # Same cache and same racily-clean rule as `sha256_file` -- the key carries the
        # (device, inode) these bytes were read from, so it can only ever be served for
        # THIS file.  Without it, binding the snapshot costs one extra full re-hash of a
        # multi-hundred-megabyte database per process.
        _DIGEST_CACHE[str(path)] = (_file_key(settled), digest)
    return handle, {"source_id": COMPLIANCE_DATABASE_SOURCE_ID, "path": str(path),
                    "sha256": digest, "size": int(opened.st_size),
                    "wal_sha256": wal_digest, "wal_size": wal_size}


#: The identity fields a bound snapshot and the file being certified must agree on.
_SNAPSHOT_IDENTITY_FIELDS = (
    ("sha256", "content digest"), ("size", "size in bytes"),
    ("wal_sha256", "write-ahead log digest"), ("wal_size", "write-ahead log size"))


def database_snapshot_drift(bound: dict) -> list[str]:
    """Every reason the database on disk is no longer the snapshot `bound` names.

    The certify-time half of `open_database_snapshot`.  `bound` is the identity captured
    from the handle that ANSWERED this encounter's queries; this recomputes the same
    identity from the file the certificate is about to attest to.  Any difference means the
    decisions and the attestation describe two different databases, which is a HOLD -- and
    it has to be detectable HERE, with no later query happening to notice it, because
    nothing requires one to happen at all. (Codex F6-R5-A.)

    Returns a list rather than raising: its callers compose a manifest's `integrity_errors`
    and must report every problem at once.  An UNBOUND identity is itself an error -- "the
    database was never identified when it answered" is not a clean result.
    """
    if not isinstance(bound, dict) or not is_content_digest(bound.get("sha256")):
        return [f"{COMPLIANCE_DATABASE_SOURCE_ID}: no content identity was bound when the "
                f"database answered this encounter, so the certificate cannot be shown to "
                f"describe the database the decisions were made against"]
    try:
        current = compliance_database_identity()
    except Exception as exc:
        return [f"{COMPLIANCE_DATABASE_SOURCE_ID}: the bound snapshot can no longer be "
                f"re-identified ({exc})"]
    changed = [label for field, label in _SNAPSHOT_IDENTITY_FIELDS
               if bound.get(field) != current.get(field)]
    if not changed:
        return []
    return [f"{COMPLIANCE_DATABASE_SOURCE_ID}: replaced or rewritten after it answered this "
            f"encounter's queries ({', '.join(changed)} differ from the bound snapshot); "
            f"the certificate would attest to bytes no decision was made from"]


def _file_identity(paths: list[Path]) -> str:
    """The identity string `ComplianceDataStore` records when it ingests a source file."""
    parts = []
    for path in paths:
        try:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            parts.append(f"{path.name}:missing")
    return "|".join(parts)


def _source_fingerprint_mismatches(conn) -> list[str]:
    """Sources whose bytes on disk differ from the ones the database was compiled from.

    This is the STALENESS half: a database that opens cleanly and answers every query can
    still be the wrong database -- built from a previous quarter's edit file while the
    manifest hashes this quarter's JSON.  The certificate would then attest to inputs the
    decision did not use, which is the finding.
    """
    try:
        recorded = dict(conn.execute(
            "SELECT source_id, fingerprint FROM data_file_fingerprint").fetchall())
    except sqlite3.Error as exc:
        return [f"compliance_database provenance unavailable: {exc}"]
    errors = []
    for row_id, source_ids in _DATABASE_SOURCE_ROWS.items():
        try:
            paths = [declared_source_path(sid) for sid in source_ids]
        except Exception as exc:
            errors.append(f"compliance_database source {row_id}: "
                          f"not resolvable from the declaration ({exc})")
            continue
        if recorded.get(row_id) != _file_identity(paths):
            errors.append(f"compliance_database source mismatch: {row_id}")
    return errors


def compliance_database_errors() -> list[str]:
    """Every reason the compiled database may not be certified as the queried authority.

    Deliberately CHEAP -- no `PRAGMA integrity_check`, which would re-read in full the
    multi-hundred-megabyte file the manifest is already SHA-256ing.  The digest is what
    binds IDENTITY; these probes are what catch a database that is present but UNUSABLE:
    truncated or malformed (SQLite raises on the first read), schema-drifted (a decision
    table gone), empty where empty is the permissive answer, or compiled from source bytes
    other than the ones on disk now.

    Returns a LIST rather than raising so `build_manifest` can report every problem at once
    and set `status = BLOCKED`; the READ path raises `CompiledDatabaseUnavailable` instead.
    """
    try:
        db_path = compliance_database_path()
    except Exception as exc:
        return [f"compliance_database: not resolvable from the declaration ({exc})"]
    if not db_path.exists():
        return ["compliance_database: absent"]
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        errors: list[str] = []
        for table in REQUIRED_DATABASE_TABLES:
            try:
                # EXISTS stops at the first row, so this is O(1) rather than a count over
                # 2.6M NCCI rows -- cheap enough to run on every manifest build.
                populated = conn.execute(
                    f'SELECT EXISTS(SELECT 1 FROM "{table}")').fetchone()[0]
            except sqlite3.Error as exc:
                errors.append(
                    f"compliance_database: decision table {table!r} unreadable ({exc})")
                continue
            if not populated:
                errors.append(
                    f"compliance_database: decision table {table!r} is empty; empty is "
                    f"the permissive answer for it, not a lookup miss")
        errors.extend(_source_fingerprint_mismatches(conn))
        return errors
    except sqlite3.Error as exc:
        return [f"compliance_database: unusable ({exc})"]
    finally:
        if conn is not None:
            conn.close()


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
    registry = _declared_registry()
    for source_id, declared in _REQUIRED_RELEASE_SOURCES.items():
        path = registry.get(source_id)
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


#: How far in the past a file's mtime must lie before its digest may be CACHED.
#:
#: Filesystem timestamp granularity is coarser than the nanosecond field suggests: two
#: writes inside one tick produce byte-DIFFERENT files carrying an IDENTICAL
#: (device, inode, size, mtime) key -- observed directly on this deployment's own
#: filesystem, where a same-size in-place SQLite `UPDATE` changed the file's bytes and
#: left `st_mtime_ns` untouched to the nanosecond.  A cache keyed on that tuple then
#: answers with the FIRST file's digest, and the certificate attests to bytes that are no
#: longer there -- which is the exact "certified bytes != read bytes" failure this whole
#: finding is about, reintroduced by the optimisation meant to serve it.
#:
#: This is the "racily clean" problem git solves in its index, and this is git's fix: a
#: file modified within this window of the moment it was hashed is never cached, so the
#: next attestation re-reads it.  One second covers every filesystem this runs on, and
#: costs nothing in practice -- the authoritative sources and the compiled database are
#: written at ingest/build time, long before any claim is certified against them.
_RACY_MTIME_WINDOW_NS = 1_000_000_000

#: {path -> ((device, inode, size, mtime_ns), digest)} for files proven non-racy.
_DIGEST_CACHE: dict[str, tuple[tuple, str]] = {}


def _file_key(stat) -> tuple:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def sha256_file(path: Path) -> str:
    """The SHA-256 of a file's exact bytes; cached only where caching is PROVABLY safe."""
    stat = path.stat()
    key, identity = str(path), _file_key(stat)
    cached = _DIGEST_CACHE.get(key)
    if cached is not None and cached[0] == identity:
        return cached[1]
    started = time.time_ns()
    digest = _sha256_bytes(path)
    after = path.stat()
    if _file_key(after) != identity:
        # The file changed WHILE it was being hashed, so this digest describes no single
        # state of it.  Report the bytes actually read and cache NOTHING; the manifest's
        # own concurrent-writer checks are what escalate this into an error.
        return digest
    if after.st_mtime_ns + _RACY_MTIME_WINDOW_NS < started:
        _DIGEST_CACHE[key] = (identity, digest)
    return digest


def _sha256_handle(handle) -> str:
    """The SHA-256 of everything left to read on an ALREADY-OPEN handle.

    The single hashing primitive here: `sha256_file` reaches a file by PATH,
    `open_database_snapshot` reaches one by an open DESCRIPTOR it then keeps, and the two
    must produce the same digest for the same bytes -- otherwise the identity bound when
    the database answered a query could never be compared against the identity the
    certificate records, which is the whole point of binding it.
    """
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_bytes(path: Path) -> str:
    with open(path, "rb") as handle:
        return _sha256_handle(handle)


#: The digest of no bytes, so an ABSENT write-ahead log and a zero-length one are ONE
#: identity -- SQLite creates an empty `-wal` merely because a read-write connection
#: opened, and that is not a change in what any reader sees.
_EMPTY_SHA256 = "sha256:" + hashlib.sha256(b"").hexdigest()

#: SQLite's write-ahead log.  This deployment runs the compiled database in WAL mode (the
#: refresh timer and the pipeline are concurrent openers -- see `ComplianceDataStore.conn`),
#: and in WAL mode the bytes a reader SEES are the main database file PLUS whatever
#: committed frames the log still holds.  The main file's digest alone is therefore not the
#: content that answered a query.  The `-shm` sidecar is deliberately NOT part of the
#: identity: it is a transient shared-memory index that READERS themselves write to, so
#: binding it would report drift on every clean run.
_WAL_SUFFIX = "-wal"


def _wal_identity(path: Path) -> tuple[str, int]:
    """(digest, size) of the write-ahead log beside `path`; absent or empty is one state."""
    wal = path.with_name(path.name + _WAL_SUFFIX)
    try:
        size = wal.stat().st_size
    except OSError:
        return _EMPTY_SHA256, 0
    if not size:
        return _EMPTY_SHA256, 0
    try:
        return sha256_file(wal), int(size)
    except OSError:
        # Truncated away between the stat and the read -- a checkpoint, which MOVED those
        # frames into the main file and is therefore already visible in its digest.
        return _EMPTY_SHA256, 0


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
    for error in _database_source_errors():
        # `_checkpoint_database` above and the source/usability probes below can both
        # report the same absent database; one identity, one error.
        if error not in errors:
            errors.append(error)
    body = {"records": records, "errors": errors}
    body["fingerprint"] = manifest_fingerprint(body)
    return body


def _checkpoint_database() -> list[str]:
    """Canonicalize committed WAL frames before hashing compliance.db.

    This is a storage checkpoint, not a data change.  If another writer owns
    the WAL, the manifest fails closed instead of hashing a stale main file.

    TOTAL, like every other contributor to `build_source_manifest`: an unresolvable
    declaration is an ERROR STRING here, never a raise.  The release gate reads the
    manifest's `errors` and converts them into a structured ERROR outcome; an exception
    escaping this helper would leave the gate with no outcome to report at all.
    """
    try:
        db_path = compliance_database_path()
    except Exception as exc:
        return [f"compliance_database: not resolvable from the declaration ({exc})"]
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
    """Everything that disqualifies the compiled database, plus the write-race check.

    The USABILITY and STALENESS probes are shared with `compliance_database_errors()` --
    which the coder's capability manifest also calls -- so the app-side release gate and
    the certificate can never reach different conclusions about the same database.  The
    only part that stays here is the concurrent-writer check, because it needs a
    read-WRITE connection and this manifest has already checkpointed the WAL.
    """
    errors = compliance_database_errors()
    try:
        db_path = compliance_database_path()
    except Exception:
        # Already reported by `compliance_database_errors()` above; see
        # `_checkpoint_database` for why this is an error string and not a raise.
        return errors
    if not db_path.exists():
        return errors
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        busy, frames, checkpointed = conn.execute(
            "PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        conn.close()
    except sqlite3.Error as exc:
        return errors + [f"compliance_database provenance unavailable: {exc}"]
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

#: The declared identity of the compiled database, named ONCE so the query-time binding,
#: the manifest record and the certificate validator cannot drift onto different spellings.
COMPLIANCE_DATABASE_SOURCE_ID = "compliance_database"


def is_content_digest(value) -> bool:
    """True for a well-formed `sha256:<64 lowercase hex>` content identity."""
    return bool(_SHA256_RE.fullmatch(str(value or "")))


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
    try:
        db_path = compliance_database_path()
    except Exception:
        # Release metadata is DESCRIPTIVE; the identity of the data is the digest.  An
        # unresolvable declaration is reported as a hard error by the disposition
        # invariant and by `compliance_database_errors()`, so it must not also escape
        # from this descriptive helper into `build_source_manifest`'s per-record loop.
        return {"effective_from": "", "effective_to": "", "version": ""}
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
