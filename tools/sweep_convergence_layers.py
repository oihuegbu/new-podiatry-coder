"""Dry sweep for the four convergence layers (E/M risk floor, guidance
laterality strip, digit-supply modifier, use-additional promotion): replay
each processed note's saved billing arrays through ONLY those checks, no LLM,
and print every mutation/flag. Measures live firing rate and surfaces false
positives before the layers gate a real batch.

Usage: PYTHONPATH=/app python tools/sweep_convergence_layers.py [results_dir] [notes_dir]
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pdfplumber

from app.compliance.engine import _parse_date
from app.rag.code_reference import CodeReferenceDB
from app.compliance.datastore.store import ComplianceDataStore
from app.validation.validator import CodingValidator

RESULTS_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "output/results")
NOTES_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "doctors_notes")

CATS = ("em_mdm_risk_floor", "em_level_corrected", "guidance_laterality_removed",
        "digit_modifier_applied", "digit_modifier_required",
        "digit_modifier_side_conflict", "missing_use_additional_code")


def note_text(stem: str) -> str | None:
    pdf = NOTES_DIR / f"{stem}.pdf"
    if not pdf.exists():
        return None
    with pdfplumber.open(pdf) as doc:
        return re.sub(r"\s+", " ", " ".join((p.extract_text() or "") for p in doc.pages))


def main():
    db = CodeReferenceDB()
    db.load_all()
    store = ComplianceDataStore()
    store.build_or_load()
    total = 0
    for f in sorted(RESULTS_DIR.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        stem = f.name.replace("_results.json", "")
        txt = note_text(stem)
        if not txt:
            continue
        d = json.loads(f.read_text())
        icd = [dict(c) for c in d.get("icd_codes") or [] if isinstance(c, dict)]
        cpt = [copy.deepcopy(c) for c in d.get("cpt_codes") or [] if isinstance(c, dict)]
        hcpcs = [copy.deepcopy(c) for c in d.get("hcpcs_codes") or [] if isinstance(c, dict)]
        pm = d.get("patient_metadata") or {}
        dos = _parse_date(str(pm.get("date_of_service") or ""))
        dob = str(pm.get("date_of_birth") or "")

        before = {
            "em": [(c.get("code"), (c.get("mdm_details") or {}).get("risk_score"))
                   for c in cpt if str(c.get("code", "")).startswith("99")],
            "mods": {c.get("code"): sorted(c.get("modifiers") or []) for c in cpt + hcpcs},
            "icd_n": len(icd),
        }

        v = CodingValidator(db, store)
        v.issues = []
        v._check_em_mdm_risk_floor(cpt, icd, dos, dob)
        v._check_em_level_consistency(cpt)
        v._check_guidance_laterality(cpt)
        v._check_digit_supply_modifier(cpt, hcpcs, txt)
        cr = {"icd10_codes": icd, "supporting_conditions":
              [dict(c) for c in d.get("supporting_conditions") or [] if isinstance(c, dict)]}
        v._check_missing_use_additional_code(icd, cr, txt)

        found = [i for i in v.issues if i.category in CATS]
        if found:
            print(f"\n=== {stem}")
            for i in found:
                total += 1
                print(f"  {i.severity:7s} {i.category:32s} {i.code:8s} {i.message[:150]}")
            after_mods = {c.get("code"): sorted(c.get("modifiers") or []) for c in cpt + hcpcs}
            for code, m in after_mods.items():
                if before["mods"].get(code, m) != m:
                    print(f"    mods {code}: {before['mods'].get(code)} -> {m}")
            if len(icd) != before["icd_n"]:
                print(f"    icd count: {before['icd_n']} -> {len(icd)} "
                      f"(+{[e['code'] for e in icd[before['icd_n']:]]})")
    print(f"\n==== {total} findings across corpus ====")


if __name__ == "__main__":
    main()
