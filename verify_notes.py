"""Independent, authoritative-source-backed re-verification of pipeline output.

Re-checks every processed note's result JSON against CodeReferenceDB /
ComplianceDataStore directly — does not trust the pipeline's own
self-reported validation_issues/claim_scrub, re-derives code existence,
modifier validity, modifier_reasoning/modifiers-array consistency, MUE
limits, NCCI PTP conflicts, and billability (is this the *right* billable
code, not just a code that exists) from the real data independently.

"Exists" vs "billable": ICD-10-CM's authoritative code set (icd10cm_codes.json)
contains only terminal/billable codes by construction (verified: zero codes
in it are a proper prefix of another code in it, i.e. no non-billable
category headers are present) — existence there already implies billability.
CPT and HCPCS have no such guarantee, so billability is checked separately
via global_periods.json's own self-documented status field (not separately
payable/noncovered/statutory exclusion — this file covers both CPT and
HCPCS-shaped codes, e.g. A4570) and add-on-code structure.

  python verify_notes.py                       # check output/results/*.json
  python verify_notes.py --dir path/to/results  # check a different directory
"""
from __future__ import annotations

import argparse
import glob
import json

from app.rag.code_reference import CodeReferenceDB
from app.compliance.datastore.store import ComplianceDataStore
from app.compliance.payer_registry import parse_insurance_text
from app.compliance.engine import _parse_dos

# E/M section of CPT — same structural-range pattern already used elsewhere
# in this codebase (medical_necessity.py's _is_em) rather than a hardcoded
# code list, so it stays correct as CPT adds/removes codes in this range.
_EM_SECTION = range(99202, 99500)


def _is_em(code: str) -> bool:
    return code.isdigit() and int(code) in _EM_SECTION


_PROCEDURE_SEPARATION_MODIFIERS = {"59", "XE", "XS", "XP", "XU"}
_EM_SEPARATION_MODIFIERS = {"25", "57"}


def check_note(path: str, ref_db: CodeReferenceDB, store: ComplianceDataStore) -> tuple[str, list[str]]:
    d = json.load(open(path))
    doc_id = d.get("document_id", path)
    problems: list[str] = []

    icd = d.get("icd_codes", [])
    cpt = d.get("cpt_codes", [])
    hcpcs = d.get("hcpcs_codes", [])
    dos = _parse_dos(d.get("patient_metadata", {}))

    # 1. Code existence in authoritative FY2026 code sets, and — for codes
    #    that do exist — whether they were actually effective on this
    #    claim's real date of service (independent re-check of
    #    validator.py's _check_code_existence; CodeReferenceDB now stores
    #    real effective_from/effective_to per code instead of treating
    #    every code as always active).
    for system, codes, label in (("icd10", icd, "ICD10"), ("cpt", cpt, "CPT"), ("hcpcs", hcpcs, "HCPCS")):
        for c in codes:
            code = c.get("code", "")
            validator = {"icd10": ref_db.validate_icd10, "cpt": ref_db.validate_cpt, "hcpcs": ref_db.validate_hcpcs}[system]
            if not validator(code):
                problems.append(f"{label} {code} not found in authoritative code set")
            elif dos and not ref_db.is_active_for_dos(system, code, dos):
                problems.append(f"{label} {code} exists but is not effective on this claim's date of service ({dos})")

    # 1b. Billability — existence alone doesn't mean separately payable.
    #     Covers CPT AND HCPCS: global_periods.json's own self-documented
    #     status field carries HCPCS-shaped codes too (e.g. A4570, status
    #     'I' = not valid for Medicare since 2001) — a code-existing-but-
    #     not-billable HCPCS line would otherwise pass silently.
    # follows_medicare_coverage (FFS + Medicare Advantage), not is_medicare —
    # MA plans inherit Original Medicare's coverage floor (42 CFR 422.101).
    is_medicare = parse_insurance_text(
        d.get("patient_metadata", {}).get("insurance", "")
    ).follows_medicare_coverage
    for c in cpt + hcpcs:
        code = c.get("code", "")
        reason = store.not_separately_billable_reason(code)
        if reason:
            problems.append(f"{code}: {reason} — not separately billable as a standalone line item")
        elif is_medicare and store.billing_status(code) == "I":
            problems.append(f"{code}: status 'I' (not valid for Medicare) but claim payer is Medicare")
        else:
            # PFS status 'X' — excluded from the Physician Fee Schedule but
            # payable under another schedule (CLFS/DMEPOS); review-level, not
            # a hard failure. See store.pfs_exclusion_advisory.
            advisory = store.pfs_exclusion_advisory(code)
            if advisory:
                problems.append(f"{code}: {advisory} [review]")

    # 1c. Add-on codes (global_period=ZZZ, AMA Appendix D descriptor phrasing)
    #     cannot be billed standalone — they must accompany their specific
    #     required primary procedure per the real NCCI Add-On Code (AOC) edit
    #     table, not just any other non-addon code on the claim.
    cpt_codes_only = [c.get("code", "") for c in cpt]
    for c in cpt:
        code = c.get("code", "")
        if not (store.is_addon(code) or store.global_period(code) == "ZZZ"):
            continue
        aoc_edits = store.ncci_aoc_edits(code)
        valid_primaries = {e.get("code2") for e in aoc_edits}
        if valid_primaries and "CCCCC" not in valid_primaries:
            has_primary = any(oc in valid_primaries for oc in cpt_codes_only if oc != code)
            if not has_primary:
                problems.append(
                    f"CPT {code}: add-on code billed without its required primary procedure "
                    f"({', '.join(sorted(valid_primaries))}) in the same claim"
                )
        else:
            # no specific AOC pairing data (or CMS wildcard) — fall back to
            # the looser "some primary procedure is present" check
            has_primary = any(
                oc != code and not store.is_addon(oc) and store.global_period(oc) != "ZZZ"
                for oc in cpt_codes_only
            )
            if not has_primary:
                problems.append(f"CPT {code}: add-on code billed without a primary procedure code in the same claim")

    # 1d. CPT laterality — independent re-check of validator.py's
    #     _check_cpt_laterality. CMS's bilateral-surgery indicator
    #     (bilat_surg='1' — the 150% bilateral-adjustment rule applies)
    #     is the real, code-specific signal that a code is billed as
    #     either unilateral (RT/LT) or bilateral (50); the coder's own
    #     structured `laterality` field is a separate, already-populated
    #     source of truth for which side. A mismatch means the pipeline's
    #     auto-correction either didn't run or regressed.
    for c in cpt:
        code = c.get("code", "")
        if not code or store.bilat_surg(code) != "1":
            continue
        mods = c.get("modifiers", []) or []
        if "RT" in mods or "LT" in mods or "50" in mods:
            continue
        laterality = str(c.get("laterality") or "").strip().upper()
        if laterality in ("RIGHT", "LEFT", "BILATERAL"):
            problems.append(
                f"CPT {code}: laterality field says '{laterality}' but modifiers array has no "
                f"RT/LT/50 — CMS bilateral-surgery indicator requires one"
            )
        else:
            problems.append(
                f"CPT {code}: requires a laterality modifier (RT/LT/50) per CMS bilateral-surgery "
                f"indicator, but neither the modifiers array nor the laterality field states a side"
            )

    # 2. Modifier validity (real modifiers.json data via compliance.db)
    for c in cpt + hcpcs:
        code = c.get("code", "")
        for mod in c.get("modifiers", []) or []:
            if not store.modifier_valid(mod):
                problems.append(f"{code}: modifier '{mod}' not a recognized CPT/HCPCS modifier")

    # 3. modifier_reasoning vs modifiers array consistency (independent re-check
    #    of validator.py's _check_modifier_reasoning_consistency). modifier_reasoning
    #    is now a structured list of {modifier, status, reason} claims (see
    #    schemas.py's ModifierClaim) instead of free text a regex had to guess
    #    the polarity of — status is checked directly, nothing to parse.
    for c in cpt + hcpcs:
        code = c.get("code", "")
        mods = c.get("modifiers", []) or []
        for claim in c.get("modifier_reasoning", []) or []:
            if not isinstance(claim, dict):
                continue
            modifier = str(claim.get("modifier", "")).strip().upper()
            status = str(claim.get("status", "")).strip().lower()
            reason = claim.get("reason", "")
            if not modifier:
                continue
            if status == "applied" and modifier not in mods:
                problems.append(
                    f"{code}: reasoning claims modifier -{modifier} applied but missing "
                    f"from modifiers array: \"{str(reason)[:80]}\""
                )
            elif status == "not_applicable" and modifier in mods:
                problems.append(
                    f"{code}: reasoning claims modifier -{modifier} is not applicable but it "
                    f"is still present in the modifiers array: \"{str(reason)[:80]}\""
                )

    # 3b. -25/-57 mutual exclusivity — always invalid together on the same
    #     E/M line regardless of global period (found live: two independent
    #     auto-correction rules each added one without checking the other's
    #     work, leaving a self-contradicting pair on the same claim).
    for c in cpt:
        code = c.get("code", "")
        if not _is_em(code):
            continue
        mods = c.get("modifiers", []) or []
        if "25" in mods and "57" in mods:
            problems.append(f"{code}: modifiers -25 and -57 both present — mutually exclusive on the same E/M line")

    # 3c. ICD-10-CM Type 1 Excludes — independent re-check of validator.py's
    #     _check_icd_excludes1. Two codes on the claim that are Excludes1 of
    #     each other per the real Tabular List data are structurally
    #     mutually exclusive ("not coded here"), not a stylistic choice.
    icd_codes_only = [c.get("code", "") for c in icd]
    for i in range(len(icd_codes_only)):
        for j in range(i + 1, len(icd_codes_only)):
            if store.excludes1_conflict(icd_codes_only[i], icd_codes_only[j]):
                problems.append(
                    f"{icd_codes_only[i]} and {icd_codes_only[j]}: ICD-10-CM Type 1 Excludes of "
                    f"each other ('not coded here') — structurally mutually exclusive per the "
                    f"Tabular List, not codeable together"
                )

    # 4. MUE — units within limit
    for c in cpt:
        code = c.get("code", "")
        units = c.get("units", 1) or 1
        mue_row = store.mue(code)
        if mue_row and mue_row.get("mue"):
            try:
                limit = int(mue_row["mue"])
                if units > limit:
                    problems.append(f"CPT {code}: units={units} exceeds MUE limit={limit}")
            except (ValueError, TypeError):
                pass

    # 5. NCCI PTP — every CPT pair, checked against the real edit table.
    #    59/XE/XS/XP/XU separate two procedures from each other; 25/57
    #    separate an E/M from a procedure — different purposes, so only
    #    credit 25/57 when one side of the pair is actually an E/M code.
    codes_with_mods = [(c.get("code", ""), set(c.get("modifiers", []) or [])) for c in cpt]
    for i in range(len(codes_with_mods)):
        for j in range(i + 1, len(codes_with_mods)):
            c1, m1 = codes_with_mods[i]
            c2, m2 = codes_with_mods[j]
            edit = store.ncci_pair(c1, c2)
            if not edit:
                continue
            indicator = str(edit.get("modifier_indicator", ""))
            pair_is_em = _is_em(c1) or _is_em(c2)
            sep_set = _EM_SEPARATION_MODIFIERS if pair_is_em else _PROCEDURE_SEPARATION_MODIFIERS
            has_sep_mod = bool((m1 | m2) & sep_set)
            if indicator == "0":
                problems.append(
                    f"NCCI PTP {c1}/{c2}: indicator=0 (never unbundle) — column2 code should "
                    f"not be billed with column1 regardless of modifier"
                )
            elif indicator == "1" and not has_sep_mod:
                suggestion = "25/57" if pair_is_em else "59/XE/XS/XP/XU"
                problems.append(
                    f"NCCI PTP {c1}/{c2}: indicator=1 (modifier allowed) but neither code "
                    f"carries a separation modifier ({suggestion})"
                )

    return doc_id, problems


def main() -> int:
    ap = argparse.ArgumentParser(description="Authoritative-source re-verification of coded notes")
    ap.add_argument("--dir", default="output/results", help="directory of *_results.json files")
    args = ap.parse_args()

    ref_db = CodeReferenceDB()
    ref_db.load_all()
    store = ComplianceDataStore()
    store.build_or_load()  # ensures schema migrations (e.g. billing_status) have run

    files = sorted(glob.glob(f"{args.dir}/*_results.json"))
    files = [f for f in files if not f.endswith("all_results.json")]

    all_problems: dict[str, list[str]] = {}
    for f in files:
        doc_id, problems = check_note(f, ref_db, store)
        all_problems[doc_id] = problems

    print(f"Checked {len(files)} notes\n")
    for doc_id, problems in all_problems.items():
        status = "CLEAN" if not problems else f"{len(problems)} issue(s)"
        print(f"=== {doc_id}: {status} ===")
        for p in problems:
            print(f"  - {p}")

    total = sum(len(p) for p in all_problems.values())
    print(f"\nTOTAL: {total} issue(s) across {len(files)} notes")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
