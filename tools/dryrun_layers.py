"""Ad-hoc staging probe: run the newest deterministic layers against the
latest saved results (arrays + note sections) WITHOUT any LLM call, printing
every firing so over-fires are visible before deployment.

Run inside the container:  PYTHONPATH=/app python tools/dryrun_layers.py
"""
import copy
import glob
import json

from app.rag.code_reference import CodeReferenceDB
from app.compliance.datastore.store import ComplianceDataStore
from app.validation.validator import CodingValidator


def main():
    db = CodeReferenceDB()
    db.load_all()
    store = ComplianceDataStore()
    store.build_or_load()
    v = CodingValidator(db, store)

    for f in sorted(glob.glob("output/results/0*_results.json"))[:10]:
        r = json.load(open(f))
        stem = f.split("/")[-1].replace("_results.json", "")
        secs = r.get("note_sections") or {}
        note = " \n".join(str(x) for x in secs.values())
        anchor = " \n".join(
            str(secs.get(k, "")) for k in
            ("assessment_diagnoses", "imaging_diagnostics", "chief_complaint"))
        icd = copy.deepcopy(r.get("icd_codes") or [])
        cpt = copy.deepcopy(r.get("cpt_codes") or [])
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_debridement_depth(cpt, note)
        v._check_imaging_context(cpt, note)
        v._check_radiograph_view_count(cpt, note)
        v._check_digit_modifier_scope(cpt)
        v._check_same_site_ptp_bundling(cpt)
        v._check_diabetes_ulcer_combination(icd)
        v._check_marginal_secondary_demotion(
            icd, {"supporting_conditions": []}, anchor)
        for i in v.issues:
            print(f"{stem} | {i.category} {i.code} | {i.message[:110]}")
        if v._non_billable_codes_to_suppress:
            print(f"{stem} | suppress: {sorted(v._non_billable_codes_to_suppress)}")


if __name__ == "__main__":
    main()
