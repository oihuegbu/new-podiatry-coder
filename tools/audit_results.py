"""Independent audit of pipeline results: code accuracy + billability.

Re-validates every code in every *_results.json against the authoritative
reference data (CodeReferenceDB + ComplianceDataStore). Does NOT reuse the
pipeline's own validation output — this is a second, independent pass so
that pipeline bugs cannot mask themselves.

Checks per note:
  ICD-10:  exists | billable leaf | active on DOS | description drift
           | exactly one primary | no duplicates | Excludes1 conflicts
  CPT:     exists | active on DOS | modifiers valid | MUE vs units
           | NCCI PTP among billed pairs | PFS status billable
           | dx pointers present (1-4)
  HCPCS:   exists | active on DOS | noncoverage reason | PFS status
Usage: python tools/audit_results.py [results_dir]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from app.rag.code_reference import CodeReferenceDB
from app.compliance.datastore.store import ComplianceDataStore

RESULTS_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "output/results")

# PFS status indicators that mean "not separately payable under the PFS"
PFS_NONPAYABLE = {"I", "N", "E"}   # I=invalid, N=non-covered, E=excluded by regulation
PFS_ADVISORY = {"X", "B", "T", "P"}  # payable elsewhere / bundled / conditional


def _parse_dos(raw) -> str | None:
    if not raw:
        return None
    raw = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _codes(entry_list):
    out = []
    for c in entry_list or []:
        if isinstance(c, dict) and c.get("code"):
            out.append(c)
    return out


def audit_note(path: Path, db: CodeReferenceDB, store: ComplianceDataStore):
    d = json.loads(path.read_text())
    findings = []  # (severity, category, message)
    add = lambda sev, cat, msg: findings.append((sev, cat, msg))

    meta = d.get("patient_metadata") or {}
    dos = _parse_dos(meta.get("date_of_service") or meta.get("dos"))

    icd = _codes(d.get("icd_codes"))
    cpt = _codes(d.get("cpt_codes"))
    hcpcs = _codes(d.get("hcpcs_codes"))

    # ---------------- ICD-10 ----------------
    seen = set()
    primaries = [c["code"] for c in icd
                 if c.get("is_primary") or str(c.get("type", "")).lower() == "primary"]
    for c in icd:
        code = c["code"].strip().upper()
        norm = code.replace(".", "")
        if code in seen:
            add("ERROR", "icd_duplicate", f"{code} appears more than once")
        seen.add(code)
        ref = db.validate_icd10(code)
        if not ref:
            # The curated podiatry reference set is a subset; the store's
            # tabular table holds the ENTIRE CDC/NCHS code set. A code in the
            # tabular but outside the curated set is real (e.g. D48.1,
            # soft-tissue neoplasm) — warn, don't error.
            tab = store.icd10_tabular_description(code)
            if tab:
                add("WARN", "icd_outside_subset",
                    f"{code} ('{tab[:60]}') is a real ICD-10-CM code outside the "
                    f"curated podiatry subset — verify specialty relevance")
            else:
                add("ERROR", "icd_not_found", f"{code} not in reference DB")
            continue
        # billable leaf: strictly longer billable codes under it make it a header
        children = [x for x in store.icd10_billable_under(code)
                    if x[0].replace(".", "") != norm]
        if children:
            add("ERROR", "icd_not_billable",
                f"{code} is a non-billable header ({len(children)} billable children, e.g. {children[0][0]})")
        if dos and not db.is_active_for_dos("icd10", code, dos):
            add("ERROR", "icd_inactive", f"{code} not active on DOS {dos}")
        official = store.icd10_tabular_description(code) or ref.get("description", "")
        given = (c.get("description") or "").strip().lower()
        if official and given and given[:40] not in official.lower() and official.lower()[:40] not in given:
            add("WARN", "icd_desc_drift",
                f"{code}: result says '{c.get('description')}' vs official '{official[:70]}'")
    if icd and len(primaries) == 0:
        add("ERROR", "icd_no_primary", "no primary diagnosis flagged")
    if len(primaries) > 1:
        add("ERROR", "icd_multi_primary", f"multiple primaries: {primaries}")
    # Excludes1 conflicts among billed dx
    codes_only = sorted(seen)
    for i, a in enumerate(codes_only):
        for b in codes_only[i + 1:]:
            if store.excludes1_conflict(a, b):
                add("ERROR", "icd_excludes1", f"{a} and {b} are mutually exclusive (Excludes1)")

    # ---------------- CPT ----------------
    n_dx = len(icd)
    cpt_codes = []
    for c in cpt:
        code = c["code"].strip().upper()
        cpt_codes.append((code, c))
        ref = db.validate_cpt(code)
        if not ref:
            add("ERROR", "cpt_not_found", f"{code} not in reference DB")
            continue
        if dos and not db.is_active_for_dos("cpt", code, dos):
            add("ERROR", "cpt_inactive", f"{code} not active on DOS {dos}")
        for m in c.get("modifiers") or []:
            if not store.modifier_valid(str(m)):
                add("ERROR", "modifier_invalid", f"{code}: modifier {m} not a valid modifier")
        units = c.get("units") or 1
        mue = db.get_mue(code)
        if mue is not None and units > mue:
            add("ERROR", "mue_exceeded", f"{code}: units {units} > MUE {mue}")
        ptrs = c.get("linked_diagnoses") or []
        if n_dx and not ptrs:
            add("ERROR", "dx_pointer_missing", f"{code}: no diagnosis pointers")
        if len(ptrs) > 4:
            add("ERROR", "dx_pointer_overflow", f"{code}: {len(ptrs)} pointers (max 4)")
        dangling = [p for p in ptrs if str(p).strip().upper() not in seen]
        if dangling:
            add("ERROR", "dx_pointer_dangling", f"{code}: pointers not on claim: {dangling}")
        status = (store.billing_status(code, dos) or "").strip()
        if status in PFS_NONPAYABLE:
            add("ERROR", "pfs_nonpayable", f"{code}: PFS status '{status}' — not payable under PFS")
        elif status in PFS_ADVISORY:
            add("WARN", "pfs_advisory", f"{code}: PFS status '{status}' — check billing path")
    # NCCI PTP among all billed CPT/HCPCS pairs (edit direction: col2 is bundled)
    anatomic = store.anatomic_modifiers()
    procedure_codes = cpt_codes + [
        (str(entry.get("code") or "").strip().upper(), entry)
        for entry in hcpcs if entry.get("code")
    ]
    by_code = {code: entry for code, entry in procedure_codes}
    if len(procedure_codes) >= 2 and not db.ncci_data_available(dos):
        add("ERROR", "ncci_data_unavailable",
            f"no loaded NCCI release covers DOS {dos or 'unknown'}")
    checked = set()
    for i in range(len(procedure_codes)):
        for j in range(i + 1, len(procedure_codes)):
            a, b = procedure_codes[i][0], procedure_codes[j][0]
            if (a, b) in checked:
                continue
            checked.add((a, b))
            edit = db.check_ncci(a, b, dos)
            if not edit:
                continue
            col1, col2 = edit["code1"], edit["code2"]
            if col2 not in by_code or col1 not in by_code:
                continue
            ind = str(edit.get("modifier") or "")
            if ind == "9":
                continue
            mods_c1 = set(str(m) for m in (by_code[col1].get("modifiers") or []))
            mods_c2 = set(str(m) for m in (by_code[col2].get("modifiers") or []))
            # NCCI PTP-associated modifiers (CMS: valid on either column code)
            ptp_assoc = {"24", "25", "27", "57", "58", "59", "78", "79", "91",
                         "XE", "XP", "XS", "XU"} | anatomic
            bypass = ptp_assoc & (mods_c1 | mods_c2)
            anat_split = (mods_c1 & anatomic) and (mods_c2 & anatomic) and \
                         (mods_c1 & anatomic) != (mods_c2 & anatomic)
            if ind == "0":
                add("ERROR", "ncci_never",
                    f"{col2} bundled into {col1}: PTP edit, modifier NOT allowed (indicator 0)")
            elif ind == "1" and not bypass and not anat_split:
                add("ERROR", "ncci_unbypassed",
                    f"{col2} bundled into {col1}: PTP edit (indicator 1), no PTP-associated modifier on either line")

    # ---------------- HCPCS ----------------
    for c in hcpcs:
        code = c["code"].strip().upper()
        ref = db.validate_hcpcs(code)
        if not ref:
            add("ERROR", "hcpcs_not_found", f"{code} not in reference DB")
            continue
        if dos and not db.is_active_for_dos("hcpcs", code, dos):
            add("ERROR", "hcpcs_inactive", f"{code} not active on DOS {dos}")
        reason = store.hcpcs_noncoverage_reason(code)
        if reason:
            add("WARN", "hcpcs_noncovered", f"{code}: {reason}")
        for m in c.get("modifiers") or []:
            if not store.modifier_valid(str(m)):
                add("ERROR", "modifier_invalid", f"{code}: modifier {m} not valid")

    return findings, {
        "icd": len(icd), "cpt": len(cpt), "hcpcs": len(hcpcs),
        "disposition": d.get("final_disposition"),
    }


def main():
    db = CodeReferenceDB()
    db.load_all()
    store = ComplianceDataStore()
    files = sorted(RESULTS_DIR.glob("*_results.json"))
    files = [f for f in files if f.name != "all_results.json"]
    total_err = total_warn = 0
    for f in files:
        try:
            findings, stats = audit_note(f, db, store)
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {f.name}: AUDIT CRASH: {e}")
            continue
        errs = [x for x in findings if x[0] == "ERROR"]
        warns = [x for x in findings if x[0] == "WARN"]
        total_err += len(errs)
        total_warn += len(warns)
        flag = "OK " if not errs else "ERR"
        print(f"\n=== [{flag}] {f.name}  (icd={stats['icd']} cpt={stats['cpt']} "
              f"hcpcs={stats['hcpcs']} disp={stats['disposition']})")
        for sev, cat, msg in findings:
            print(f"    {sev:5s} {cat:22s} {msg}")
    print(f"\n==== TOTAL: {len(files)} notes | {total_err} errors | {total_warn} warnings ====")


if __name__ == "__main__":
    main()
