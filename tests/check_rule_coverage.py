"""Architectural guard: every rule-bearing field in every reference data file
must be either (a) consumed by a named store method / check, or (b) explicitly
waived here with a reason.

Why this exists: the recurring production-bug pattern in this project was
never "the rule was wrong" — it was "the rule existed in already-ingested
reference data and nothing consumed it" (billing status X, bilat_surg,
excludes1, includes-subsumption, useAdditionalCode, coverage_code, PC/TC,
MCE age lists — every one found by a human review after shipping). This
guard turns that class of gap into a CI failure:

  * a data refresh that introduces a NEW field fails the guard until the
    field is wired into enforcement or waived with a written reason;
  * renaming/removing a consumer symbol fails the guard until the manifest
    is updated — coverage claims can't silently go stale.

Consumers are verified by symbol presence in app/ source (the deepest layer
that reads the field: a store method, agent, or validator check).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CODES = DATA / "codes"
APP = ROOT / "app"

# --------------------------------------------------------------------------
# The manifest. field -> {"consumer": "<symbol in app/ source>"} when wired,
# or {"waived": "<reason>"} when deliberately unenforced. Bookkeeping fields
# (provenance, display text) are listed as waived too — the point is that
# EVERY observed field has a written disposition.
# --------------------------------------------------------------------------
MANIFEST: dict[str, dict[str, dict]] = {
    "global_periods.json codes.*": {
        "status":          {"consumer": "not_separately_billable_reason"},
        "global_days":     {"consumer": "global_period"},
        "global_days_int": {"waived": "derived convenience duplicate of global_days"},
        "pctc_ind":        {"consumer": "pfs_indicators"},
        "mult_proc":       {"waived": "payment-side multiple-procedure fee reduction; "
                                      "claim-side modifier-51 exemption is governed by CPT "
                                      "Appendix E (modifier_exempt.json), not this indicator"},
        "bilat_surg":      {"consumer": "pfs_indicators"},
        "asst_surg":       {"consumer": "pfs_indicators"},
        "co_surg":         {"consumer": "pfs_indicators"},
        "team_surg":       {"consumer": "pfs_indicators"},
        "endo_base":       {"waived": "endoscopy-family payment reduction base — no "
                                      "endoscopic procedures in podiatry claim scope, and the "
                                      "rule is a fee calculation, not a claim validity edit"},
    },
    "codes/hcpcs_codes.json []": {
        "code":              {"consumer": "_ingest_hcpcs"},
        "short_description": {"consumer": "_ingest_hcpcs"},
        "long_description":  {"waived": "display text; short_description is ingested"},
        "effective_from":    {"waived": "quarterly pricing-cycle marker, NOT a lifecycle "
                                        "date (verified: shifts for decades-old codes); "
                                        "add_date is the real introduction date"},
        "effective_to":      {"consumer": "_ingest_hcpcs"},
        "add_date":          {"consumer": "_ingest_hcpcs"},
        "modifiers":         {"waived": "DMEPOS pricing-file modifier column (NU/RR/UE "
                                        "rental-purchase context), a fee-schedule key — "
                                        "modifier validity is governed by modifiers.json"},
        "coverage_code":     {"consumer": "hcpcs_noncoverage_reason"},
        "betos":             {"waived": "BETOS/RBCS analytics classification — no billing rule"},
        "action_code":       {"waived": "lifecycle change marker; discontinuation is already "
                                        "captured via effective_to ingestion"},
        "metadata":          {"waived": "provenance"},
    },
    "codes/icd10cm_codes.json []": {
        "code":           {"consumer": "_ingest_code_set"},
        "description":    {"consumer": "_ingest_code_set"},
        "effective_from": {"consumer": "_ingest_code_set"},
        "effective_to":   {"consumer": "_ingest_code_set"},
        "fy":             {"waived": "fiscal-year label duplicating effective dates"},
        "status":         {"consumer": "_ingest_code_set"},
    },
    "codes/icd10cm_instructional_notes.json codes.*": {
        "code":            {"consumer": "_ingest_icd10_tabular_desc"},
        "description":     {"consumer": "_ingest_icd10_tabular_desc"},
        "inclusionTerm":   {"consumer": "_ingest_icd10_inclusion_terms"},
        "inclusionTerm_code_refs": {"waived": "code refs inside synonym phrases — the "
                                              "phrases themselves are the evidence "
                                              "vocabulary (icd10_inclusion_term)"},
        "includes":        {"consumer": "_ingest_icd10_includes"},
        "includes_code_refs": {"consumer": "_ingest_icd10_includes"},
        "excludes1":       {"consumer": "_ingest_icd10_excludes1"},
        "excludes1_code_refs": {"consumer": "_ingest_icd10_excludes1"},
        "excludes2":       {"waived": "Type 2 Excludes = 'not included here' — both codes "
                                      "MAY be reported together when the patient has both "
                                      "conditions (ICD-10-CM guideline I.A.12.b); permissive, "
                                      "nothing to enforce"},
        "excludes2_code_refs": {"waived": "see excludes2"},
        "codeFirst":       {"consumer": "_ingest_icd10_code_first"},
        "codeFirst_code_refs": {"consumer": "_ingest_icd10_code_first"},
        "useAdditionalCode": {"consumer": "_ingest_icd10_use_additional_code"},
        "useAdditionalCode_code_refs": {"consumer": "_ingest_icd10_use_additional_code"},
        "codeAlso":        {"consumer": "_ingest_icd10_code_also"},
        "codeAlso_code_refs": {"consumer": "_ingest_icd10_code_also"},
    },
    "codes/mce_edits.json codes": {
        "age_newborn":     {"consumer": "MCEAgent"},
        "age_pediatric":   {"consumer": "MCEAgent"},
        "age_maternity":   {"consumer": "MCEAgent"},
        "age_adult":       {"consumer": "MCEAgent"},
        "manifestation_not_pdx": {"consumer": "MCEAgent"},
        "unacceptable_pdx": {"consumer": "MCEAgent"},
        "unacceptable_pdx_unless_secondary": {"consumer": "MCEAgent"},
    },
    "codes/mue_practitioner.json []": {
        "code":           {"consumer": "_ingest_mue"},
        "mue_value":      {"consumer": "_ingest_mue"},
        "effective_date": {"consumer": "_ingest_mue"},
        "description":    {"consumer": "_ingest_mue"},  # carries the MAI digit
        "source_file":    {"consumer": "_published_effective_date"},
    },
    "codes/ncci_data.json []": {
        "code1":          {"consumer": "_ingest_ncci"},
        "code2":          {"consumer": "_ingest_ncci"},
        "edit_type":      {"waived": "constant 'ptp' tag for this file"},
        "effective_date": {"consumer": "_ingest_ncci"},
        "end_date":       {"consumer": "_ingest_ncci"},
        "modifier":       {"consumer": "_ingest_ncci"},
        "description":    {"waived": "human-readable restatement of the modifier indicator"},
        "metadata":       {"waived": "provenance"},
        "source_file":    {"waived": "provenance"},
    },
    "codes/ncci_aoc_edits.json []": {
        "code1":          {"consumer": "_ingest_ncci_aoc"},
        "code2":          {"consumer": "_ingest_ncci_aoc"},
        "edit_type":      {"waived": "constant 'aoc' tag for this file"},
        "effective_date": {"consumer": "_ingest_ncci_aoc"},
        "end_date":       {"consumer": "_ingest_ncci_aoc"},
        "modifier":       {"consumer": "_ingest_ncci_aoc"},
        "description":    {"waived": "human-readable restatement"},
        "metadata":       {"waived": "provenance"},
    },
    "codes/pos_codes.json codes.*": {
        "name":     {"consumer": "_ingest_pos"},
        "facility": {"consumer": "_ingest_pos"},
    },
    "codes/modifier_exempt.json codes[]": {
        "code":               {"consumer": "_ingest_modifier_exempt"},
        "modifier_51_exempt": {"consumer": "_ingest_modifier_exempt"},
        "modifier_63_exempt": {"consumer": "_ingest_modifier_exempt"},
        "description_51":     {"waived": "display text"},
        "description_63":     {"waived": "display text"},
        "metadata":           {"waived": "provenance"},
    },
    "codes/podiatry_lcd.json lcd[]": {
        "lcd_id":         {"consumer": "_ingest_lcd"},
        "title":          {"consumer": "policies_titled"},
        "contractor":     {"consumer": "coverage_policy_states"},
        "status":         {"consumer": "_ingest_lcd"},
        "governed_cpts":  {"consumer": "coverage_policies_for_cpt"},
        "qualifying_dx":  {"consumer": "coverage_icd_covered"},
        "companion_article_id":      {"consumer": "_ingest_lcd"},
        "icd10_in_companion_article": {"consumer": "_ingest_lcd"},
        "description":    {"waived": "policy narrative HTML; the machine-readable rule "
                                     "content is governed_cpts/qualifying_dx"},
        "effective_date": {"waived": "only status='A' (currently active) policies are "
                                     "ingested, and the weekly MCD refresh replaces them "
                                     "wholesale — activity, not date math, governs applicability"},
        "version":        {"waived": "provenance (policy revision counter)"},
    },
    "codes/podiatry_lcd.json article[]": {
        "article_id":     {"consumer": "_ingest_lcd"},
        "title":          {"consumer": "policies_titled"},
        "contractor":     {"consumer": "coverage_policy_states"},
        "status":         {"consumer": "_ingest_lcd"},
        "governed_cpts":  {"consumer": "coverage_policies_for_cpt"},
        "qualifying_dx":  {"consumer": "coverage_icd_covered"},
        "description":    {"waived": "policy narrative HTML"},
        "effective_date": {"waived": "see lcd[] effective_date"},
        "version":        {"waived": "provenance"},
    },
    "codes/mac_jurisdictions.json ab_mac_jurisdictions[]": {
        "jurisdiction": {"waived": "human-readable jurisdiction label (JE/JF/...); "
                                   "resolution keys off contractor aliases + states"},
        "contractor":   {"waived": "canonical display name; matching uses aliases"},
        "aliases":      {"consumer": "contractor_states"},
        "states":       {"consumer": "contractor_states"},
    },
    "codes/mac_jurisdictions.json dme_mac_jurisdictions[]": {
        "jurisdiction": {"waived": "human-readable label; see ab_mac_jurisdictions"},
        "contractor":   {"waived": "canonical display name; matching uses aliases"},
        "aliases":      {"consumer": "contractor_states"},
        "states":       {"consumer": "contractor_states"},
    },
    "codes/modifiers.json modifiers.*": {
        "name":           {"consumer": "_ingest_modifiers"},
        "description":    {"consumer": "_ingest_modifiers"},
        "level":          {"waived": "duplicates system"},
        "system":         {"consumer": "_ingest_modifiers"},
        "section":        {"consumer": "_ingest_modifiers"},
        "effective_date": {"waived": "sparsely populated in source; modifiers are stable"},
        "metadata":       {"waived": "provenance"},
    },
    # AHRQ HCUP CCIR chronicity flags: {"metadata": {...}, "codes": {"E119": 1, ...}}.
    # codes is a flat code->indicator map (no per-entry dicts), so the guard
    # checks the top-level shape and the single consumer of the values.
    "codes/icd10cm_chronic.json .": {
        "metadata": {"waived": "provenance"},
        "codes":    {"consumer": "icd10_is_chronic"},
    },
    # Declarative validator rule pack: config-authored rules executed by the
    # generic template executors in app/validation/rule_engine.py. The guard
    # pins the loader as consumer; per-rule enforcement is covered by the
    # validator regression suite (tests/test_validator_checks.py), which
    # exercises every rule through its delegating _check_* method.
    "rules/validator_rules.json .": {
        "version":  {"waived": "pack version label (provenance)"},
        "_comment": {"waived": "authoring documentation"},
        "rules":    {"consumer": "load_rule_pack"},
    },
    "terminology/clinical_abbreviations.json .": {
        "schema_version": {"consumer": "TerminologyNormalizer"},
        "version":        {"consumer": "normalization_registry_version"},
        "governance":     {"waived": "human-readable safety contract"},
        "sources":        {"consumer": "source_version"},
        "acceptance":     {"consumer": "min_confidence"},
        "unknown_detection": {"consumer": "unknown_pattern"},
        "entries":        {"consumer": "self.entries"},
    },
    "terminology/clinical_abbreviations.json sources.*": {
        "name":          {"waived": "human-readable source name"},
        "version":       {"consumer": "source_version"},
        "review_status": {"waived": "governance metadata surfaced through the bound registry"},
        "authority_role": {"consumer": "authority_role"},
        "provenance_kind": {"consumer": "provenance_kind"},
        "scope":         {"waived": "human-readable source scope"},
    },
    "terminology/clinical_abbreviations.json acceptance": {
        "min_confidence":       {"consumer": "min_confidence"},
        "min_margin":           {"consumer": "min_margin"},
        "context_window_chars": {"consumer": "context_window"},
        "evidence_weights":     {"consumer": "evidence_weights"},
    },
    "terminology/clinical_abbreviations.json acceptance.evidence_weights": {
        "required_section":     {"consumer": "required_section"},
        "required_context_any": {"consumer": "required_context_any"},
        "required_context_all": {"consumer": "required_context_all"},
        "anatomy":              {"consumer": "anatomy"},
        "laterality":           {"consumer": "laterality"},
    },
    "terminology/clinical_abbreviations.json unknown_detection": {
        "pattern":                   {"consumer": "unknown_pattern"},
        "negation_patterns":         {"consumer": "negation_patterns"},
        "ignored_terms":             {"consumer": "ignored_terms"},
        "billing_relevant_sections": {"consumer": "billing_sections"},
    },
    "terminology/clinical_abbreviations.json entries[]": {
        "id":            {"consumer": "entry_id"},
        "patterns":      {"consumer": "regexes"},
        "coding_impact": {"consumer": "coding_impact"},
        "candidates":    {"consumer": "_candidate_scores"},
    },
    "terminology/clinical_abbreviations.json entries[].candidates[]": {
        "expansion":            {"consumer": "expansion"},
        "confidence":           {"consumer": "confidence"},
        "source_id":            {"consumer": "source_id"},
        "required_context_any": {"consumer": "required_any"},
        "anatomy_any":          {"consumer": "anatomy"},
        "required_sections":    {"consumer": "required_sections"},
        "allowed_laterality":   {"consumer": "allowed_laterality"},
    },
}


def _observed_fields(spec: str) -> set[str] | None:
    """Fields actually present in the data file addressed by `spec`, which is
    '<relpath> <accessor>' where accessor is: '[]' (list of dicts — sample),
    'key' (the dict at that key — its keys ARE the fields), 'key.*' (values
    of that dict are dicts — union of their keys), 'key[]' (list under
    key), or '.' (the root dict — its keys ARE the fields). Large list
    files are sampled by streaming the head of the file."""
    rel, accessor = spec.rsplit(" ", 1)
    path = DATA / rel
    if not path.exists():
        return None
    if accessor == "[]" and path.stat().st_size > 20_000_000:
        head = path.open(encoding="utf-8", errors="replace").read(60_000)
        return set(re.findall(r'"(\w+)"\s*:', head))
    data = json.loads(path.read_text())
    if accessor == ".":
        return set(data.keys())
    if accessor == "[]":
        items = data
    elif accessor == "entries[].candidates[]":
        items = [candidate for entry in data.get("entries", [])
                 if isinstance(entry, dict)
                 for candidate in entry.get("candidates", [])
                 if isinstance(candidate, dict)]
    elif accessor.endswith("[]"):
        items = data[accessor[:-2]]
    elif accessor.endswith(".*"):
        node = data
        for part in accessor[:-2].split("."):
            node = node[part]
        items = list(node.values())
    else:
        node = data
        for part in accessor.split("."):
            node = node[part]
        return set(node.keys())
    fields: set[str] = set()
    for item in items[:5000] if isinstance(items, list) else items:
        if isinstance(item, dict):
            fields.update(item.keys())
    return fields


def main() -> int:
    src = "\n".join(p.read_text() for p in APP.rglob("*.py"))
    failures: list[str] = []
    checked_files = 0

    for spec, fields in MANIFEST.items():
        observed = _observed_fields(spec)
        if observed is None:
            failures.append(f"{spec}: data file missing")
            continue
        checked_files += 1
        unmapped = observed - set(fields)
        for f in sorted(unmapped):
            failures.append(
                f"{spec}: field '{f}' exists in the data but has no disposition — "
                f"wire it into a check or waive it with a reason")
        for f, disp in fields.items():
            if f not in observed:
                failures.append(
                    f"{spec}: manifest lists '{f}' but the data no longer has it — "
                    f"remove the stale entry")
            consumer = disp.get("consumer")
            if consumer and consumer not in src:
                failures.append(
                    f"{spec}: declared consumer '{consumer}' for '{f}' not found in "
                    f"app/ source — the coverage claim is stale")

    if failures:
        print("❌ RULE COVERAGE GUARD FAILED:")
        for f in failures:
            print(f"   - {f}")
        return 1
    n_fields = sum(len(v) for v in MANIFEST.values())
    n_wired = sum(1 for v in MANIFEST.values() for d in v.values() if "consumer" in d)
    print(f"✅ Rule coverage: {n_fields} fields across {checked_files} data files — "
          f"{n_wired} wired to enforcement, {n_fields - n_wired} waived with reasons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
