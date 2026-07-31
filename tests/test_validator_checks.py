"""Regression tests for the deterministic validator checks and payer registry.

Everything here is data-driven the same way the checks themselves are: codes
used in fixtures are looked up against the live reference data first, and the
assertions verify behavior (mismatch flagged / agreement silent), never
hardcoded rule outcomes.

Run:  PYTHONPATH=. python tests/test_validator_checks.py
"""
from datetime import date, timedelta

from app.rag.code_reference import CodeReferenceDB
from app.compliance.datastore.store import ComplianceDataStore
from app.validation.validator import CodingValidator

PASSED = FAILED = 0


def check(label: str, cond: bool):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ✅ {label}")
    else:
        FAILED += 1
        print(f"  ❌ {label}")


def cats(v):
    return {i.category for i in v.issues}


def main():
    db = CodeReferenceDB()
    db.load_all()
    store = ComplianceDataStore()
    store.build_or_load()
    v = CodingValidator(db, store)

    print("\n[E/M level vs descriptor MDM]")
    v.issues = []
    v._check_em_level_consistency([{"code": "99213", "mdm_details": {"mdm_level": "moderate"}}])
    check("99213 claimed moderate MDM → flagged (descriptor says low)",
          "em_level_mismatch" in cats(v))
    v.issues = []
    v._check_em_level_consistency([{"code": "99213", "mdm_details": {"mdm_level": "low"}}])
    check("99213 claimed low MDM → silent", not v.issues)
    v.issues = []
    v._check_em_level_consistency([{"code": "99211", "mdm_details": {"mdm_level": "low"}}])
    check("99211 (no MDM level in descriptor) → silent", not v.issues)

    print("\n[J-code units vs descriptor denomination]")
    v.issues = []
    v._check_drug_units([{"code": "J3301", "units": 1,
                          "evidence_spans": ["triamcinolone 40 mg injected"]}])
    check("J3301 (per 10 mg) x1 with 40 mg documented → flagged",
          "drug_units_mismatch" in cats(v))
    v.issues = []
    v._check_drug_units([{"code": "J3301", "units": 4,
                          "evidence_spans": ["triamcinolone 40 mg injected"]}])
    check("J3301 x4 with 40 mg documented → silent", not v.issues)
    v.issues = []
    v._check_drug_units([{"code": "J3301", "units": 1, "evidence_spans": ["injected"]}])
    check("no dose documented, no note → silent (cannot verify)", not v.issues)
    # Undocumented-dose suppression (live 009): a drug line whose units
    # cannot be derived from ANY documentation is unbillable as written.
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_drug_units(
        [{"code": "J3370", "units": 1, "evidence_spans": []}],
        "IV vancomycin initiated empirically per ID recommendation. "
        "Wound culture pending.")
    check("J-code with no dose anywhere in note → suppressed (live 009)",
          "drug_dose_undocumented" in cats(v)
          and "J3370" in v._non_billable_codes_to_suppress)
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_drug_units(
        [{"code": "J3370", "units": 2, "evidence_spans": []}],
        "Vancomycin 1000 mg administered IV over 60 minutes in office.")
    check("dose in a note sentence naming the drug → kept, unit math runs",
          "J3370" not in v._non_billable_codes_to_suppress
          and not any(i.category == "drug_dose_undocumented" for i in v.issues))
    # Gram-dosed agents ('vancomycin 1 g') must count as documented dose for
    # an mg-denominated descriptor — same fact, x1000.
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_drug_units(
        [{"code": "J3370", "units": 2, "evidence_spans": []}],
        "Vancomycin 1 g administered IV in office; tolerated well.")
    check("gram-documented dose → kept (mg-denominated descriptor, x1000)",
          "J3370" not in v._non_billable_codes_to_suppress
          and not any(i.category == "drug_dose_undocumented" for i in v.issues)
          and not any(i.category == "drug_units_mismatch" for i in v.issues))
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_drug_units(
        [{"code": "J3301", "units": 4,
          "evidence_spans": ["triamcinolone 40 mg injected"]}],
        "Kenalog injected into the joint. No dose in this sentence.")
    check("dose already in line evidence → note arm not consulted, silent",
          not v.issues and not v._non_billable_codes_to_suppress)
    v._non_billable_codes_to_suppress = set()
    # Fractional denominations ('per 0.5 mg' class) crashed live (note 013):
    # int(per_unit) truncated 0.x to 0 → integer modulo by zero. Find one in
    # the store's own data so no code is hardcoded; the math must be
    # float-safe regardless.
    frac = None
    for c, info in getattr(db, "hcpcs_codes", {}).items():
        d = (info.get("long_description") or "").lower().strip()
        m = re.search(r"(?:,|per)\s*(0\.\d+)\s*(mg|mcg|ml)\s*$", d)
        if m and d.startswith("injection"):
            frac = (c, float(m.group(1)), m.group(2))
            break
    if frac:
        code_f, per_f, uom_f = frac
        v.issues = []
        v._check_drug_units([{"code": code_f, "units": 1,
                              "evidence_spans": [f"administered 2 {uom_f} dose"]}])
        check(f"fractional denomination ({code_f}, per {per_f:g} {uom_f}) "
              f"→ no crash, ceil math correct",
              any(i.category == "drug_units_mismatch" for i in v.issues))
    else:
        check("SKIP: no fractional-denomination injection code in dataset", True)

    print("\n[timed infusion administration documentation]")
    # Live 009: 96365 flapped present-in-1-of-3 on 'IV vancomycin initiated
    # empirically per ID recommendation' — no duration, no start/stop.
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_timed_infusion_documentation(
        [{"code": "96365"}],
        "Deep wound debridement performed. IV vancomycin initiated "
        "empirically per ID recommendation. Follow up in 5 days.")
    check("96365 with no documented infusion time → suppressed (live 009)",
          "infusion_time_undocumented" in cats(v)
          and "96365" in v._non_billable_codes_to_suppress)
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_timed_infusion_documentation(
        [{"code": "96365"}],
        "IV vancomycin 1 g infused over 60 minutes in office; tolerated well.")
    check("documented duration in an infusion sentence → kept",
          not v.issues and not v._non_billable_codes_to_suppress)
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_timed_infusion_documentation(
        [{"code": "97597"}],
        "Debridement of wound performed. No infusion today.")
    check("non-time-denominated code → untouched",
          not v.issues and not v._non_billable_codes_to_suppress)
    v._non_billable_codes_to_suppress = set()

    print("\n[count-ranged code families]")
    v.issues = []
    v._check_count_based_selection([{"code": "11720", "evidence_spans": ["debrided 8 nails"]}])
    check("11720 (1-5) with 8 nails → flagged", "count_range_mismatch" in cats(v))
    v.issues = []
    v._check_count_based_selection([{"code": "11721", "evidence_spans": ["debrided 8 nails"]}])
    check("11721 (6+) with 8 nails → silent", not v.issues)
    v.issues = []
    v._check_count_based_selection([{"code": "99213", "evidence_spans": ["8 nails noted"]}])
    check("non-count code → silent", not v.issues)

    print("\n[ICD↔CPT laterality agreement]")
    v.issues = []
    v._check_icd_cpt_laterality_agreement(
        [{"code": "64455", "modifiers": ["RT"], "linked_diagnoses": ["G57.62"]}])
    check("RT procedure + left-side dx → flagged",
          "icd_cpt_laterality_conflict" in cats(v))
    v.issues = []
    v._check_icd_cpt_laterality_agreement(
        [{"code": "64455", "modifiers": ["RT"], "linked_diagnoses": ["G57.61"]}])
    check("RT procedure + right-side dx → silent", not v.issues)
    v.issues = []
    v._check_icd_cpt_laterality_agreement(
        [{"code": "64455", "modifiers": ["RT", "LT"], "linked_diagnoses": ["G57.62"]}])
    check("RT+LT line → skipped", not v.issues)

    print("\n[diagnosis pointers]")
    v.issues = []
    icd = [{"code": "G57.61", "type": "primary"}]
    cpt = [{"code": "99213",
            "linked_diagnoses": ["E11.40", "G57.61", "M79.671", "M79.672", "L84"]}]
    v._check_dx_pointers(icd, cpt, [])
    check(">4 pointers → flagged", "dx_pointer_overflow" in cats(v))
    check("primary reordered to pointer 1", cpt[0]["linked_diagnoses"][0] == "G57.61")

    print("\n[ICD Includes subsumption (Tabular List hierarchy)]")
    v.issues = []
    icd = [{"code": "I70.221", "type": "primary"},
           {"code": "I70.235", "type": "secondary"},
           {"code": "L97.519", "type": "secondary"}]
    cpt = [{"code": "93923", "linked_diagnoses": ["I70.221", "I70.235"]}]
    hcpcs = [{"code": "A6212", "linked_diagnoses": ["L97.519", "I70.221"]}]
    cr = {"icd10_codes": icd}
    v._check_icd_includes_subsumption(icd, cpt, hcpcs, cr)
    codes_left = [c["code"] for c in icd]
    check("rest pain (I70.221) removed — subsumed by ulceration (I70.235)",
          "I70.221" not in codes_left and "I70.235" in codes_left)
    check("subsumption flagged", "icd_includes_subsumption" in cats(v))
    check("keeper inherits primary designation",
          next(c for c in icd if c["code"] == "I70.235")["type"] == "primary")
    check("companion L97.519 untouched", "L97.519" in codes_left)
    check("CPT linked dx relinked to keeper (deduped)",
          cpt[0]["linked_diagnoses"] == ["I70.235"])
    check("HCPCS linked dx relinked",
          hcpcs[0]["linked_diagnoses"] == ["L97.519", "I70.235"])
    v.issues = []
    icd = [{"code": "I70.235", "type": "primary"}, {"code": "I70.222", "type": "secondary"}]
    v._check_icd_includes_subsumption(icd, [], [], {"icd10_codes": icd})
    check("opposite-leg rest pain NOT subsumed (laterality respected)",
          len(icd) == 2 and not v.issues)
    v.issues = []
    icd = [{"code": "I11.0", "type": "primary"}, {"code": "I50.9", "type": "secondary"}]
    v._check_icd_includes_subsumption(icd, [], [], {"icd10_codes": icd})
    check("I11.0 + I50.9 kept — useAdditionalCode mandates the companion",
          len(icd) == 2 and not v.issues)

    print("\n[missing use-additional-code companion (Tabular List mirror)]")
    # Faithful to the real 031 note's weak phrasing: 'dialysis' is the ONLY
    # distinctive Z99.2 term present (no 'renal', no 'dependence') — passes
    # only via the Tabular note line's own wording ('dialysis status').
    note_031 = ("68-year-old male with T2DM, ESRD on dialysis, active Charcot right foot. "
                "Medications: insulin, erythropoietin, lisinopril, aspirin. "
                "Pamidronate pending nephrology clearance.")
    v.issues = []
    cr = {"supporting_conditions": [{"code": "Z79.4"}]}
    icd = [{"code": "N18.6", "type": "secondary"}]
    v._check_missing_use_additional_code(icd, cr, note_031)
    added = [c["code"] for c in icd]
    check("N18.6 + documented dialysis, no Z99.2 → flagged (live 031 regression)",
          "missing_use_additional_code" in cats(v))
    check("Z99.2 auto-added to the billed diagnoses (secondary)", "Z99.2" in added)
    check("auto-added entry flagged for review",
          next(c for c in icd if c["code"] == "Z99.2")["needs_review"]
          and next(c for c in icd if c["code"] == "Z99.2")["type"] == "secondary")
    check("companion NOT parked in supporting_conditions",
          [c["code"] for c in cr["supporting_conditions"]] == ["Z79.4"])
    v.issues = []
    cr = {"supporting_conditions": []}
    v._check_missing_use_additional_code(
        [{"code": "N18.6", "type": "primary"}, {"code": "Z99.2", "type": "secondary"}],
        cr, "ESRD on renal dialysis three times weekly.")
    check("companion already on claim → silent", not v.issues and not cr["supporting_conditions"])
    v.issues = []
    cr = {"supporting_conditions": []}
    v._check_missing_use_additional_code(
        [{"code": "N18.6", "type": "primary"}], cr,
        "Chronic kidney disease, no renal replacement documented. Kidney function declining.")
    check("condition not documented (single generic word) → silent",
          not v.issues and not cr["supporting_conditions"])
    # Official-acronym arm (live 010): M86's Tabular mandates B95-B97 for
    # the infectious agent; the note speaks only the acronym 'MRSA', which
    # the Tabular's own inclusion term spells out for B95.62.
    v.issues = []
    cr = {"supporting_conditions": []}
    icd = [{"code": "M86.9", "type": "primary"}]
    v._check_missing_use_additional_code(
        icd, cr,
        "Chronic osteomyelitis of the distal phalanx, right hallux. Deep "
        "bone culture grew MRSA; MRSA as cause of disease. Six-week "
        "antibiotic course planned.")
    check("M86.9 + 'MRSA' documented → B95.62 auto-added via Tabular acronym",
          "B95.62" in [c["code"] for c in icd]
          and "missing_use_additional_code" in cats(v))
    v.issues = []
    cr = {"supporting_conditions": []}
    icd = [{"code": "M86.9", "type": "primary"},
           {"code": "B95.62", "type": "secondary"}]
    v._check_missing_use_additional_code(
        icd, cr, "Chronic osteomyelitis. Deep bone culture grew MRSA.")
    check("agent code already on claim → silent, nothing added",
          not v.issues and len(icd) == 2)
    v.issues = []
    cr = {"supporting_conditions": []}
    icd = [{"code": "M86.9", "type": "primary"}]
    v._check_missing_use_additional_code(
        icd, cr, "Chronic osteomyelitis of the hallux. Cultures pending, "
                 "no organism identified to date.")
    check("no organism documented → silent (conditional instruction)",
          not any(c["code"].startswith("B9") for c in icd))

    print("\n[measurement companion (Tabular acronym + descriptor range, no code literals)]")
    # E66.9's own Tabular note: 'use additional code to identify body mass
    # index (BMI), if known (Z68.-)'. Note documents BMI 36.2 → Z68.36's own
    # descriptor range 36.0-36.9 selects the exact code.
    v.issues = []
    v._check_measurement_companion(
        [{"code": "E66.9", "type": "secondary"}],
        "Class II obesity. BMI 36.2 recorded at intake. Counseled on weight management.")
    hits = [i for i in v.issues if i.category == "measurement_companion_missing"]
    check("obesity + documented BMI 36.2, no Z68 → flagged", bool(hits))
    check("exact range code Z68.36 selected from descriptor range",
          hits and hits[0].code == "Z68.36")
    v.issues = []
    v._check_measurement_companion(
        [{"code": "E66.9", "type": "secondary"}, {"code": "Z68.36", "type": "secondary"}],
        "Obesity. BMI 36.2.")
    check("companion already on claim → silent",
          not [i for i in v.issues if i.category == "measurement_companion_missing"])
    v.issues = []
    v._check_measurement_companion(
        [{"code": "E66.9", "type": "secondary"}],
        "Obesity documented. Patient declined weight measurement today.")
    check("value not documented ('if known') → silent",
          not [i for i in v.issues if i.category == "measurement_companion_missing"])

    print("\n[image guidance codes derived from CPT descriptors]")
    gmap = v._guidance_cpts()
    check("fluoroscopic + ultrasonic guidance codes discovered from descriptors",
          gmap.get("fluoroscopic") == "77002" and gmap.get("ultrasonic") == "76942")
    check("CT and MRI guidance modalities also discovered",
          "computed tomography" in gmap and "magnetic resonance" in gmap)

    print("\n[missing code-first etiology (Tabular List mirror)]")
    v.issues = []
    v._check_missing_code_first_etiology(
        [{"code": "L97.519", "type": "primary"}],
        "Non-healing right heel ulcer. Known atherosclerosis of native arteries of the right leg.")
    check("L97 ulcer + documented atherosclerosis, no etiology code → flagged",
          "missing_code_first_etiology" in cats(v))
    v.issues = []
    v._check_missing_code_first_etiology(
        [{"code": "I70.235", "type": "primary"}, {"code": "L97.519", "type": "secondary"}],
        "Atherosclerosis with ulceration right heel.")
    check("etiology present on claim → silent", not v.issues)
    v.issues = []
    v._check_missing_code_first_etiology(
        [{"code": "L97.519", "type": "primary"}], "Non-healing right heel ulcer, cause unclear.")
    check("etiology not documented → silent", not v.issues)

    print("\n[missing code-also companion (Tabular List mirror)]")
    # E84 (cystic fibrosis) carries a real 'code also' note referencing
    # K86.81 (exocrine pancreatic insufficiency) — verified against the store
    assert any(r == "K8681" for _, refs in store.code_also_groups("E84.0")
               for r, _ in refs)
    v.issues = []
    v._check_missing_code_also(
        [{"code": "E84.0", "type": "primary"}],
        "Cystic fibrosis with documented exocrine pancreatic insufficiency on enzyme replacement.")
    check("code-also companion documented but absent → flagged",
          "missing_code_also" in cats(v))
    v.issues = []
    v._check_missing_code_also(
        [{"code": "E84.0", "type": "primary"}, {"code": "K86.81", "type": "secondary"}],
        "Cystic fibrosis with exocrine pancreatic insufficiency.")
    check("companion on claim → silent", not v.issues)
    v.issues = []
    v._check_missing_code_also(
        [{"code": "E84.0", "type": "primary"}],
        "Cystic fibrosis, pulmonary exacerbation. GI review negative.")
    check("companion not documented → silent", not v.issues)

    print("\n[CHANGED-correction enforcement]")
    from app.coding.code_assigner import _enforce_changed_corrections
    r = {"corrections_made": [{"type": "CHANGED", "code": "Z88.5", "to_code": "Z88.6",
                               "reason": "aspirin is an analgesic"}],
         "supporting_conditions": [{"code": "Z88.5", "rationale": "x"}],
         "icd10_codes": [], "cpt_codes": [], "hcpcs_codes": []}
    _enforce_changed_corrections(r, db)
    check("structured to_code applied to the code field",
          r["supporting_conditions"][0]["code"] == "Z88.6")
    r = {"corrections_made": [],
         "supporting_conditions": [{"code": "Z88.5",
             "rationale": "Aspirin is an analgesic, not a narcotic — aspirin allergy maps to "
                          "the analgesic-agent sibling (Z88.6), not narcotic-agent (Z88.5). "
                          "Corrected per category family disambiguation data.",
             "review_reason": "corrected to analgesic-agent category"}],
         "icd10_codes": [], "cpt_codes": [], "hcpcs_codes": []}
    _enforce_changed_corrections(r, db)
    check("narrative-only sibling correction applied (live Z88.5 regression)",
          r["supporting_conditions"][0]["code"] == "Z88.6")
    r = {"corrections_made": [],
         "icd10_codes": [{"code": "Z88.5",
             "rationale": "Retained Z88.5 rather than switching to Z88.6 — opioid documented."}],
         "supporting_conditions": [], "cpt_codes": [], "hcpcs_codes": []}
    _enforce_changed_corrections(r, db)
    check("refusal phrasing does not trigger a switch",
          r["icd10_codes"][0]["code"] == "Z88.5")
    r = {"corrections_made": [{"type": "CHANGED", "code": "Z88.5", "to_code": "Z88.99",
                               "reason": "x"}],
         "supporting_conditions": [{"code": "Z88.5", "rationale": "x"}],
         "icd10_codes": [], "cpt_codes": [], "hcpcs_codes": []}
    _enforce_changed_corrections(r, db)
    check("nonexistent replacement code not applied",
          r["supporting_conditions"][0]["code"] == "Z88.5")
    # Direction awareness (live flip-back regression): the LLM already applied
    # the correction (entry carries the NEW code) and the narrative describes
    # it as 'corrected from OLD to NEW' — the backstop must not revert to OLD.
    r = {"corrections_made": [{"type": "CHANGED", "code": "Z88.5",
                               "reason": "aspirin is a non-narcotic analgesic"}],
         "supporting_conditions": [{"code": "Z88.6",
             "rationale": "Aspirin is a non-narcotic analgesic, not a narcotic agent; corrected "
                          "from Z88.5 (narcotic agent) to Z88.6 (analgesic agent) per category "
                          "family disambiguation.",
             "review_reason": "Corrected sibling code — aspirin is an analgesic, not a narcotic"}],
         "icd10_codes": [], "cpt_codes": [], "hcpcs_codes": []}
    _enforce_changed_corrections(r, db)
    check("already-applied correction never flipped back to the old code",
          r["supporting_conditions"][0]["code"] == "Z88.6")

    print("\n[modifier -57 requires a same-day major procedure]")
    # Decision-for-surgery language + only minor (000-global) codes on today's
    # claim: no bundling risk today, so no ERROR (live false positive: plan
    # said 'surgical correction discussed', claim carried 000-globals only).
    v.issues = []
    v._check_modifier57([{"code": "99214", "modifiers": ["25"]}, {"code": "11055", "modifiers": []}],
                        "Discussed surgical correction of the bunion; will schedule.")
    check("surgery decision + no 090-global today → INFO advisory, not ERROR",
          "modifier_57_missing" not in cats(v) and "modifier_57_future_surgery" in cats(v))
    # 090-global procedure actually on today's claim → the ERROR stands
    assert (store.global_period("28297") or "") == "090"
    v.issues = []
    v._check_modifier57([{"code": "99214", "modifiers": []}, {"code": "28297", "modifiers": ["RT"]}],
                        "Patient elects surgical correction; will proceed with bunionectomy today.")
    check("surgery decision + 090-global on claim, no -57 → ERROR",
          "modifier_57_missing" in cats(v))

    print("\n[NCCI anatomic-modifier separation (validator mirror)]")
    pair = db.check_ncci("28297", "28285")
    if pair and str(pair.get("modifier", "")).strip() == "1":
        v.issues = []
        v._check_ncci([{"code": "28297", "modifiers": ["TA"]},
                       {"code": "28285", "modifiers": ["T6"]}])
        check("differing anatomic modifiers → exception applied (INFO)",
              any(i.category == "ncci_edit" and i.severity == "INFO" for i in v.issues)
              and not any(i.severity in ("WARNING", "ERROR") for i in v.issues))
        v.issues = []
        v._check_ncci([{"code": "28297", "modifiers": ["RT"]},
                       {"code": "28285", "modifiers": ["RT"]}])
        check("same anatomic modifier both lines → still flagged",
              any(i.category == "ncci_edit" and i.severity == "WARNING" for i in v.issues))
        # generic side vs same-side digit is NOT site separation (note 010):
        # RT names the whole side and T6 lies within it
        v.issues = []
        v._check_ncci([{"code": "28297", "modifiers": ["RT"]},
                       {"code": "28285", "modifiers": ["RT", "T6"]}])
        check("generic RT vs same-side digit → still flagged (no bypass)",
              any(i.category == "ncci_edit" and i.severity == "WARNING" for i in v.issues))
    else:
        print("  (skipped — 28297/28285 no longer an indicator-1 edit)")

    print("\n[billability suppression is an auto-correction (INFO)]")
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_billability([{"code": "4260F"}], [])
    check("Category II code suppressed as INFO, not ERROR",
          any(i.category == "billability" and i.severity == "INFO"
              and "AUTO-CORRECTED" in i.message for i in v.issues)
          and not any(i.severity == "ERROR" for i in v.issues))
    check("code queued for deterministic removal",
          "4260F" in v._non_billable_codes_to_suppress)

    print("\n[code-array normalization (malformed LLM output)]")
    # Observed live (note 053): the verify pass compacted icd10_codes to bare
    # strings, crashing the first downstream .get(). Strings become minimal
    # entries; non-list values are dropped so the pre-verification fallback
    # kicks in; missing keys stay missing (fallback semantics preserved).
    from app.coding.code_assigner import _normalize_code_arrays
    r = {"icd10_codes": ["E11.42", {"code": "L84", "rationale": "x"}, None],
         "cpt_codes": "not a list"}
    _normalize_code_arrays(r)
    check("bare-string code wrapped, dict kept, garbage dropped",
          r["icd10_codes"] == [{"code": "E11.42"}, {"code": "L84", "rationale": "x"}])
    check("non-list array removed so pre-verification fallback applies",
          "cpt_codes" not in r)
    check("missing keys stay missing", "hcpcs_codes" not in r)

    print("\n[verify-pass additions are reference-gated]")
    # Live batch: D48.1 (non-billable header) re-added as a bare string and
    # CPT 20926 (deleted years ago) invented as a new line — Pass 4's output
    # was the only ungated path onto the claim.
    from app.coding.code_assigner import _gate_verify_additions
    fr = {"icd10_codes": [{"code": "D48.1"}, {"code": "E11.621"}],
          "supporting_conditions": [], "hcpcs_codes": [{"code": "A9270"}],
          "cpt_codes": [{"code": "20926"}, {"code": "99213"}, {"code": "Q9999"}]}
    _gate_verify_additions(fr, {"icd10_codes": [], "supporting_conditions": [],
                                "cpt_codes": [], "hcpcs_codes": []}, db, store)
    check("valid additions kept (E11.621, 99213)",
          any(e["code"] == "E11.621" for e in fr["icd10_codes"])
          and any(e["code"] == "99213" for e in fr["cpt_codes"]))
    check("header with billable children kept for the specificity filter (D48.1)",
          any(e["code"] == "D48.1" for e in fr["icd10_codes"]))
    check("codes known nowhere removed (deleted 20926, fabricated Q9999)",
          not any(e["code"] in ("20926", "Q9999") for e in fr["cpt_codes"]))
    check("HCPCS unlisted-but-valid policy preserved",
          any(e["code"] == "A9270" for e in fr["hcpcs_codes"]))
    # a code that was already on the claim pre-verification is never touched
    fr2 = {"icd10_codes": [{"code": "D48.1"}], "supporting_conditions": [],
           "cpt_codes": [], "hcpcs_codes": []}
    _gate_verify_additions(fr2, {"icd10_codes": [{"code": "D48.1"}],
                                 "supporting_conditions": [], "cpt_codes": [],
                                 "hcpcs_codes": []}, db, store)
    check("pre-verification codes always pass through",
          fr2["icd10_codes"] and fr2["icd10_codes"][0]["code"] == "D48.1")

    print("\n[ICD sibling-descriptor differential (best-code-vs-note)]")
    # Live batch: left-hallux subungual hematoma (nail damage) billed as
    # S90.122A — left LESSER toe, WITHOUT nail damage. The note documents
    # the sibling's attributes ('hallux' → great toe), never 'lesser'.
    note6 = ("Left hallux nail pain after marathon. Black nail with subungual "
             "hematoma of the left hallux. Trephination performed x2 on the hallux.")
    icd_sib = [{"code": "S90.122A", "type": "primary"}]
    v.issues = []
    v._check_icd_sibling_descriptor(icd_sib, note6)
    check("S90.122A flagged, great-toe sibling suggested",
          any(i.category == "sibling_matches_note_better" and "great" in i.message
              for i in v.issues))
    # correct sibling selection stays silent: right lesser toe documented as such
    icd_ok2 = [{"code": "S90.221A", "type": "primary"}]
    v.issues = []
    v._check_icd_sibling_descriptor(
        icd_ok2, "Acute pain right second toe after dropping weight plate. "
                 "Subungual hematoma right second toe, nail damage present.")
    check("correct lesser-toe code with matching note → silent",
          not any(i.category == "sibling_matches_note_better" for i in v.issues))
    # Tabular inclusion terms count as the code's own evidence: B35.1
    # ('Tinea unguium') is supported by the note's 'onychomycosis'
    icd_b35 = [{"code": "B35.1", "type": "primary"}]
    v.issues = []
    v._check_icd_sibling_descriptor(
        icd_b35, "Onychomycosis of the right hallux nail confirmed by PAS stain.")
    check("B35.1 supported via its own inclusion term (onychomycosis) → not flagged",
          not any("B35.1" in str(i.code) and i.category in
                  ("sibling_matches_note_better", "billed_attribute_undocumented")
                  for i in v.issues))
    # billed-attribute-undocumented branch: documented paronychia billed as
    # L03.041 (acute LYMPHANGITIS — a rare entity the note never mentions)
    icd_l03 = [{"code": "L03.041", "type": "secondary"}]
    v.issues = []
    v._check_icd_sibling_descriptor(
        icd_l03, "Acute paronychia right hallux with purulence at nail fold. "
                 "Incision and drainage performed.")
    check("L03.041's lymphangitis never documented → flagged for review",
          any(i.category in ("billed_attribute_undocumented",
                             "sibling_matches_note_better") for i in v.issues))
    # ...and the Alphabetic Index rescues the CORRECT sibling: 'paronychia'
    # resolves to the L03.03x cellulitis family via its Index see-also, so
    # billing L03.031 on the same note is supported, not flagged.
    icd_l03c = [{"code": "L03.031", "type": "secondary"}]
    v.issues = []
    v._check_icd_sibling_descriptor(
        icd_l03c, "Acute paronychia right hallux with purulence at nail fold. "
                  "Incision and drainage performed.")
    check("L03.031 (cellulitis) supported via Index 'paronychia' see-also → silent",
          not any(i.category in ("billed_attribute_undocumented",
                                 "sibling_matches_note_better")
                  and str(i.code) == "L03.031" for i in v.issues))

    print("\n[negated findings are not documentation]")
    # Live batch: 'No lymphangitis.' in the exam made the sibling check claim
    # the note documents lymphangitis and suggest L03.021 over cellulitis.
    note2 = ("Bilateral hallux ingrown nails with infection. Erythema extending "
             "1 cm proximal to nail fold. No lymphangitis. Cellulitis of right "
             "toe present at the nail border.")
    icd_neg = [{"code": "L03.031", "type": "secondary"}]
    v.issues = []
    v._check_icd_sibling_descriptor(icd_neg, note2)
    check("'No lymphangitis' → cellulitis code NOT flagged toward lymphangitis sibling",
          not any(i.category == "sibling_matches_note_better"
                  and "lymphangitis" in i.message for i in v.issues))

    print("\n[CPT descriptor-condition evidence]")
    # Live batch: 27766 (medial malleolus FRACTURE) billed for an intentional
    # osteotomy approach during an allograft OCD transplant — no fracture
    # documented; 'microfracture' must not satisfy 'fracture'.
    note38 = ("Osteochondral allograft transplant right talar OCD via medial "
              "malleolar osteotomy approach. Prior microfracture failed. "
              "Allograft press-fit into defect. Osteotomy fixed with screws.")
    cpt38 = [{"code": "27766", "modifiers": ["RT"], "linked_diagnoses": ["M93.271"]}]
    v.issues = []
    v._check_cpt_descriptor_evidence(cpt38, note38)
    check("27766 flagged: fracture never documented (microfracture doesn't count)",
          any(i.category == "descriptor_condition_undocumented" for i in v.issues))
    # genuine fracture care stays silent
    cpt41 = [{"code": "28615", "modifiers": ["RT"], "linked_diagnoses": ["S93.324A"]}]
    v.issues = []
    v._check_cpt_descriptor_evidence(
        cpt41, "ORIF Lisfranc fracture-dislocation right foot. Anatomic reduction "
               "of the tarsometatarsal dislocation achieved.")
    check("28615 with documented dislocation → silent",
          not any(i.category == "descriptor_condition_undocumented" for i in v.issues))

    print("\n[qualifier contradiction (autograft vs allograft class)]")
    entry46 = {"code": "28446", "modifiers": ["RT"]}
    v.issues = []
    v._check_qualifier_contradiction(entry46, note38.lower())
    check("28446 (autograft) contradicted by documented allograft",
          any(i.category == "descriptor_qualifier_contradicted" and "allograft" in i.message
              for i in v.issues))
    v.issues = []
    v._check_qualifier_contradiction(
        {"code": "28446", "modifiers": ["RT"]},
        "open osteochondral autograft harvested and transferred; autograft press-fit.")
    check("documented autograft → silent", not v.issues)

    print("\n[dedicated unbilled code surfaced (10140-vs-11740 class)]")
    note5 = ("Electrocautery trephine device applied over hematoma. Immediate "
             "decompression of the subungual hematoma with evacuation of blood. "
             "Subungual hematoma right second toe.")
    cpt5 = [{"code": "10140", "modifiers": ["T6"], "linked_diagnoses": ["S90.221A"]}]
    v.issues = []
    v._check_unbilled_descriptor_match(cpt5, [], note5)
    # 10140 is an NCCI component of 11740, so the dedicated-code finding now
    # lands as the stronger comprehensive UPGRADE (line converges on 11740)
    # rather than a warning left for the coder.
    check("11740 (evacuation of subungual hematoma) surfaced as the dedicated code",
          any("11740" in str(i.code) and i.category in
              ("dedicated_code_unbilled", "component_upgraded_to_comprehensive")
              for i in v.issues)
          or cpt5[0]["code"] == "11740")

    print("\n[new-layer noise guards (live batch false positives)]")
    # E/M descriptors say '...moderate LEVEL of medical decision making' —
    # 'level' comes from S/T heading TAILS ('at lower leg level'), not heads,
    # and must not be an injury entity.
    v.issues = []
    v._check_cpt_descriptor_evidence(
        [{"code": "99213", "modifiers": ["25"], "linked_diagnoses": []}],
        "Established patient visit for nail trephination. No injuries.")
    check("E/M code not flagged for 'level'",
          not any(i.category == "descriptor_condition_undocumented" for i in v.issues))
    # selection/infection share the suffix 'ection' — a rhyme, not a variant
    # axis ('ection' is not a word); autograft/allograft stay ('graft' is).
    axes = v._qualifier_axes()
    check("selection/infection rhyme not an axis",
          ("infection", "selection") not in axes and ("selection", "infection") not in axes)
    check("autograft/allograft still an axis",
          ("allograft", "autograft") in axes)
    # Category II ($0.00 tracking) codes never surface as dedicated alternatives
    note19 = ("Offloading pressure relief prescribed. Pharmacologic therapy "
              "reviewed for osteoporosis prevention. Callus debrided.")
    v.issues = []
    v._check_unbilled_descriptor_match(
        [{"code": "11055", "modifiers": ["T9"], "linked_diagnoses": ["L84"]}], [], note19)
    check("Category II codes (4269F class) never surfaced",
          not any(i.category == "dedicated_code_unbilled" and str(i.code).endswith("F")
                  for i in v.issues))
    # one undocumented rare token disqualifies: 21510 is thorax-specific
    note10 = ("Incision and deep debridement with opening of bone cortex for "
              "osteomyelitis of the distal phalanx. Bone curetted.")
    v.issues = []
    v._check_unbilled_descriptor_match(
        [{"code": "28124", "modifiers": ["TA"], "linked_diagnoses": ["M86.672"]}], [], note10)
    check("21510 (thorax) not surfaced for foot osteomyelitis",
          not any("21510" in str(i.code) for i in v.issues))

    print("\n[futurity/allergy guards (live batch false positives)]")
    # 'Endoscopic plantar fasciotomy DISCUSSED as surgical option' — intent,
    # not a rendered service; 29893 must not surface as unbilled.
    v.issues = []
    v._check_unbilled_descriptor_match(
        [{"code": "0232T", "modifiers": [], "linked_diagnoses": ["M72.2"]}], [],
        "PRP injection performed to right plantar fascia under ultrasound. "
        "Endoscopic plantar fasciotomy discussed as surgical option for both "
        "feet if PRP unsuccessful.")
    check("29893 not surfaced when fasciotomy merely 'discussed as option'",
          not any("29893" in str(i.code) for i in v.issues))
    # 'Formal avulsion deferred pending vascular clearance' → 11730 not rendered
    v.issues = []
    v._check_unbilled_descriptor_match(
        [{"code": "97597", "modifiers": [], "linked_diagnoses": ["L60.2"]}], [],
        "Nail plate decompressed with elevator. Formal avulsion of the nail "
        "plate deferred pending vascular clearance; partial simple avulsion "
        "planned at follow-up.")
    check("11730 not surfaced when avulsion 'deferred pending clearance'",
          not any("11730" in str(i.code) for i in v.issues))
    # ALLERGIES: Aspirin — an avoided drug is not therapy; Z79.02 (billed for
    # clopidogrel) must not be flagged toward the aspirin sibling Z79.82.
    icd_z79 = [{"code": "Z79.02", "type": "secondary"}]
    v.issues = []
    v._check_icd_sibling_descriptor(
        icd_z79, "MEDICATIONS: Metoprolol, Atorvastatin, Clopidogrel. "
                 "ALLERGIES: Aspirin (GI bleed - avoided). Long-term "
                 "antiplatelet therapy with clopidogrel continues.")
    check("Z79.02 not flagged toward Z79.82 when aspirin is an allergy",
          not any(str(i.code) == "Z79.02" for i in v.issues))

    # 'compartment pressures 28 mmHg — below threshold for fasciotomy' means
    # NO fasciotomy was performed; 29893 must not surface (live 041 FP).
    v.issues = []
    v._check_unbilled_descriptor_match(
        [{"code": "28615", "modifiers": ["RT"], "linked_diagnoses": ["S93.324A"]}], [],
        "ORIF of Lisfranc fracture-dislocation. Plantar ecchymosis noted. "
        "Compartment pressures measured, 28 mmHg, below threshold for "
        "fasciotomy. Fascial watch-dog sutures placed for serial checks.")
    check("29893 not surfaced when pressures 'below threshold for fasciotomy'",
          not any("29893" in str(i.code) for i in v.issues))

    print("\n[dx pointer integrity (dangling pointer remap/drop)]")
    # Live class: claim ICD corrected to a sibling but the service line kept
    # pointing at the pre-correction code (S90.122A→S90.112A, Q69.2→Q69.9).
    v.issues = []
    icd_pi = [{"code": "S90.112A", "type": "primary"}, {"code": "L60.1", "type": "secondary"}]
    cpt_pi = [{"code": "99213", "linked_diagnoses": ["S90.122A", "L60.1"]},
              {"code": "11740", "linked_diagnoses": ["S90.122A"]}]
    v._check_dx_pointer_integrity(icd_pi, cpt_pi, [])
    check("dangling S90.122A remapped to claim sibling S90.112A on both lines",
          cpt_pi[0]["linked_diagnoses"] == ["S90.112A", "L60.1"]
          and cpt_pi[1]["linked_diagnoses"] == ["S90.112A"])
    check("remap logged as INFO", any(i.category == "dx_pointer_integrity"
                                      and i.severity == "INFO" for i in v.issues))
    v.issues = []
    icd_pi = [{"code": "M79.671", "type": "primary"}]
    cpt_pi = [{"code": "99213", "linked_diagnoses": ["E11.42"]}]
    v._check_dx_pointer_integrity(icd_pi, cpt_pi, [])
    check("pointer with no family match dropped (backfill restocks later)",
          cpt_pi[0]["linked_diagnoses"] == [])
    v.issues = []
    cpt_pi = [{"code": "99213", "linked_diagnoses": ["M79.671"]}]
    v._check_dx_pointer_integrity(icd_pi, cpt_pi, [])
    check("valid pointers untouched, no finding",
          cpt_pi[0]["linked_diagnoses"] == ["M79.671"] and not v.issues)

    print("\n[MUE of 0 — zero units payable, payer-gated enforcement]")
    # 90389 (tetanus immune globulin) carries MUE 0: CMS pays no units ever.
    if db.get_mue("90389") == 0:
        # Medicare-coverage payer → line auto-suppressed (deterministic
        # enactment of the MUE agent's own recommendation; measured live as
        # the biggest run-to-run presence-flap class: A4570/A6545/L1940).
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._payer_follows_medicare = True
        entry_mue = {"code": "90389", "units": 1}
        v._check_mue([entry_mue])
        check("Medicare-coverage payer: MUE-0 line auto-suppressed with INFO",
              "90389" in v._non_billable_codes_to_suppress
              and any(i.category == "mue_limit" and i.severity == "INFO"
                      for i in v.issues))
        # Commercial/unrecognized payer → flag-and-review, never removed
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._payer_follows_medicare = False
        entry_mue2 = {"code": "90389", "units": 1}
        v._check_mue([entry_mue2])
        check("commercial payer: MUE-0 flagged ERROR, not suppressed",
              "90389" not in v._non_billable_codes_to_suppress
              and any(i.category == "mue_limit" and i.severity == "ERROR"
                      and i.code == "90389" for i in v.issues))
        check("commercial payer: MUE-0 line routed to review",
              entry_mue2.get("needs_review") is True)
        v._payer_follows_medicare = False
    else:
        check("SKIP: 90389 MUE is not 0 in this dataset", True)

    print("\n[injury 7th character D→S with post-traumatic condition]")
    icd42 = [{"code": "M19.171", "type": "primary"},
             {"code": "S93.321D", "type": "secondary"}]
    v.issues = []
    v._check_injury_seventh_char(icd42)
    check("S93.321D corrected to sequela S93.321S",
          icd42[1]["code"] == "S93.321S"
          and any(i.category == "seventh_char_sequela" for i in v.issues))
    icd_active = [{"code": "S93.321D", "type": "primary"}]
    v.issues = []
    v._check_injury_seventh_char(icd_active)
    check("no post-traumatic companion → D untouched",
          icd_active[0]["code"] == "S93.321D" and not v.issues)
    icd_acute = [{"code": "M19.171", "type": "primary"},
                 {"code": "S93.321A", "type": "secondary"}]
    v.issues = []
    v._check_injury_seventh_char(icd_acute)
    check("'A' (possible new same-site injury) never touched",
          icd_acute[1]["code"] == "S93.321A")

    print("\n[verify-pass omitted fields inherited from pre-verification entries]")
    # Live batch: Pass 4 re-emitted every ICD entry without "type", so the
    # schema default ("secondary") erased the primary designation on 3 notes.
    from app.coding.code_assigner import _inherit_dropped_fields
    fr3 = {"icd10_codes": [{"code": "E11.610", "rationale": "new reasoning"},
                           {"code": "Z79.4", "type": "secondary"}],
           "cpt_codes": [{"code": "99215", "modifiers": ["25"]}]}
    combined3 = {"icd10_codes": [{"code": "E11.610", "type": "primary", "confidence": 0.94},
                                 {"code": "Z79.4", "type": "primary"}],
                 "cpt_codes": [{"code": "99215", "modifiers": ["25"], "units": 1}]}
    _inherit_dropped_fields(fr3, combined3)
    check("omitted 'type' restored from pre-verification entry",
          fr3["icd10_codes"][0]["type"] == "primary"
          and fr3["icd10_codes"][0]["confidence"] == 0.94)
    check("explicitly-returned field respected over the prior value",
          fr3["icd10_codes"][1]["type"] == "secondary")
    check("field the model DID return is never overwritten",
          fr3["icd10_codes"][0]["rationale"] == "new reasoning")
    check("omitted units restored on CPT entry",
          fr3["cpt_codes"][0]["units"] == 1)

    print("\n[corrections_made normalization (malformed LLM output)]")
    # Observed live: the verify pass emitted corrections_made as bare strings,
    # crashing every downstream .get() and aborting the whole note.
    from app.coding.code_assigner import _normalize_corrections, _enforce_added_corrections
    r = {"corrections_made": ["Removed 11719 — routine nail trim not documented",
                              {"type": "ADDED", "code": "Z99.2", "reason": "dialysis"}],
         "icd10_codes": [], "supporting_conditions": [], "cpt_codes": [], "hcpcs_codes": []}
    _normalize_corrections(r)
    check("string entries coerced to dicts, dict entries untouched",
          all(isinstance(c, dict) for c in r["corrections_made"])
          and r["corrections_made"][1]["code"] == "Z99.2")
    check("coerced entry recovers type and code from the text",
          r["corrections_made"][0]["type"] == "REMOVED"
          and r["corrections_made"][0]["code"] == "11719")
    r2 = {"corrections_made": "not even a list", "icd10_codes": [],
          "supporting_conditions": [], "cpt_codes": [], "hcpcs_codes": []}
    _normalize_corrections(r2)
    check("non-list corrections_made replaced with empty list",
          r2["corrections_made"] == [])
    # end-to-end: normalized output must survive the enforcement passes
    r3 = {"corrections_made": ["Corrected E11.9 to E11.65 per documented hyperglycemia"],
          "icd10_codes": [{"code": "E11.9", "rationale": "x"}],
          "supporting_conditions": [], "cpt_codes": [], "hcpcs_codes": []}
    _normalize_corrections(r3)
    _enforce_added_corrections(r3, db)
    check("normalized strings flow through enforcement without crashing", True)

    print("\n[LLM transient-error retry classification]")
    from app.core.llm_client import _is_retryable
    check("Anthropic overloaded_error → retryable",
          _is_retryable(Exception("{'type': 'error', 'error': {'type': 'overloaded_error', "
                                  "'message': 'Overloaded'}}")))
    check("rate limit → retryable", _is_retryable(Exception("429 rate_limit_error")))
    check("gateway 529 → retryable", _is_retryable(Exception("HTTP 529")))
    check("auth failure → NOT retryable",
          not _is_retryable(Exception("401 invalid x-api-key")))
    check("bad request → NOT retryable",
          not _is_retryable(Exception("400 max_tokens exceeds model maximum")))

    print("\n[payer registry]")
    from app.compliance.payer_registry import parse_insurance_text
    r = parse_insurance_text("Humana Medicare Advantage")
    check("MA is not FFS medicare", r.payer_id == "medicare_advantage" and not r.is_medicare)
    check("MA follows medicare coverage", r.follows_medicare_coverage)
    check("bare Medicare is FFS", parse_insurance_text("Medicare Part B").is_medicare)
    check("Humana PPO is commercial", parse_insurance_text("Humana PPO").kind == "commercial")

    print("\n[telehealth POS ⇄ modifier]")
    from app.compliance.agents.pos_eligibility import POSEligibilityAgent
    from app.compliance.models import Claim, ClaimLine, Payer
    agent = POSEligibilityAgent(store)
    f = agent.check(Claim(place_of_service="10", payer=Payer(),
                          lines=[ClaimLine(code="99213", code_system="CPT", modifiers=[])]))
    check("telehealth POS without 95/93 → WARN",
          any("telehealth" in x.reason for x in f))
    f = agent.check(Claim(place_of_service="11", payer=Payer(),
                          lines=[ClaimLine(code="99213", code_system="CPT", modifiers=["95"])]))
    check("office POS with 95 → WARN (contradiction)",
          any("not a telehealth POS" in x.reason for x in f))
    f = agent.check(Claim(place_of_service="10", payer=Payer(),
                          lines=[ClaimLine(code="99213", code_system="CPT", modifiers=["95"])]))
    check("telehealth POS with 95 → silent", not f)

    print("\n[DOS sanity]")
    from app.compliance.agents.specificity import SpecificityAgent
    sp = SpecificityAgent(store)
    f = sp.check(Claim(payer=Payer(), date_of_service=None, lines=[], diagnoses=[]))
    check("missing DOS → WARN", any("missing or unparseable" in x.reason for x in f))
    f = sp.check(Claim(payer=Payer(), date_of_service=date.today() + timedelta(days=30),
                       lines=[], diagnoses=[]))
    check("future DOS → WARN", any("in the future" in x.reason for x in f))

    print("\n[dx-pointer overflow auto-trim]")
    # 5 linked dxs on one line: keep primary + clinical conditions, drop the
    # Z-chapter status code (deterministic, explainable ranking)
    icd5 = [{"code": "E11.610", "type": "primary"}, {"code": "M14.671", "type": "secondary"},
            {"code": "E11.42", "type": "secondary"}, {"code": "N18.6", "type": "secondary"},
            {"code": "Z79.4", "type": "secondary"}]
    line = [{"code": "99215", "linked_diagnoses": ["E11.610", "M14.671", "E11.42", "N18.6", "Z79.4"]}]
    v.issues = []
    v._check_dx_pointers(icd5, line, [])
    check("5 pointers → auto-trimmed to 4", len(line[0]["linked_diagnoses"]) == 4)
    check("Z-chapter status code dropped, clinical dxs kept",
          line[0]["linked_diagnoses"] == ["E11.610", "M14.671", "E11.42", "N18.6"])
    check("overflow WARNING names the dropped code",
          any(i.category == "dx_pointer_overflow" and "Z79.4" in i.message for i in v.issues))
    v.issues = []
    line4 = [{"code": "99215", "linked_diagnoses": ["E11.610", "M14.671", "E11.42", "N18.6"]}]
    v._check_dx_pointers(icd5, line4, [])
    check("4 pointers → untouched, no overflow issue",
          line4[0]["linked_diagnoses"] == ["E11.610", "M14.671", "E11.42", "N18.6"]
          and "dx_pointer_overflow" not in cats(v))
    # determinism: same input always trims identically
    trims = set()
    for _ in range(5):
        l = [{"code": "99215", "linked_diagnoses": ["E11.610", "M14.671", "E11.42", "N18.6", "Z79.4"]}]
        v.issues = []
        v._check_dx_pointers(icd5, l, [])
        trims.add(tuple(l[0]["linked_diagnoses"]))
    check("trim is deterministic across runs", len(trims) == 1)

    print("\n[descriptor-variant evidence (Q4038 material assumption)]")
    # guard: the variant relationship must come from the live descriptors
    idx = v._hcpcs_variant_index()
    check("Q4038 pairs with a plaster sibling from descriptors alone",
          any(own == "fiberglass" and sib == "plaster" for own, sib, _ in idx.get("Q4038", [])))
    v.issues = []
    entry = {"code": "Q4038", "description": "Cast sup shrt leg fiberglass"}
    v._check_descriptor_variant_evidence(
        [entry], "Total contact cast applied with well-padded cast materials.")
    check("material never documented → variant WARNING + needs_review",
          "descriptor_variant_unverified" in cats(v) and entry.get("needs_review") is True)
    v.issues = []
    v._check_descriptor_variant_evidence(
        [{"code": "Q4038"}],
        "Short leg fiberglass total contact cast applied to this adult patient.")
    check("all distinguishing attributes documented → silent", not v.issues)
    v.issues = []
    v._check_descriptor_variant_evidence(
        [{"code": "Q4038"}], "Short leg plaster cast applied, adult.")
    check("note documents the SIBLING's material → HIGH-risk contradiction",
          any(i.category == "descriptor_variant_unverified" and i.denial_risk == "HIGH"
              for i in v.issues))

    print("\n[HCPCS descriptor age ranges (DOB-driven, not prose-driven)]")
    # guard: ranges parsed from the live CMS descriptors, not curated
    check("pediatric descriptor parses to (0, 10)", v._descriptor_age_range("Q4039") == (0, 10.0))
    r = v._descriptor_age_range("Q4038")
    check("adult descriptor parses to (11, ∞)", r is not None and r[0] == 11 and r[1] == float("inf"))
    entry = {"code": "Q4039", "description": "Cast sup shrt leg ped plster"}
    v.issues = []
    v._check_hcpcs_age_range([entry], date(2026, 5, 29), "08/03/1955")
    check("pediatric supply code on a 70-year-old → auto-corrected to adult sibling",
          entry["code"] == "Q4037"
          and any(i.category == "hcpcs_age_range_mismatch" for i in v.issues))
    entry = {"code": "Q4038"}
    v.issues = []
    v._check_hcpcs_age_range([entry], date(2026, 5, 29), "08/03/1955")
    check("age-appropriate code → untouched", entry["code"] == "Q4038" and not v.issues)
    entry = {"code": "Q4038"}
    v.issues = []
    v._check_hcpcs_age_range([entry], date(2026, 5, 29), "08/03/2020")
    check("adult supply code on a child → corrected to pediatric sibling",
          entry["code"] == "Q4040")
    entry = {"code": "Q4038"}
    v.issues = []
    v._check_hcpcs_age_range([entry], date(2026, 5, 29), "")
    check("no DOB → conservative, no change", entry["code"] == "Q4038" and not v.issues)
    # the variant-evidence check must NOT re-flag the age attribute (it is
    # settled by DOB above, not by whether the prose says "adult")
    v.issues = []
    v._check_descriptor_variant_evidence(
        [{"code": "Q4037"}], "Short leg plaster cast applied.")
    check("age attribute excluded from prose-evidence flags",
          not any("pediatric" in i.message for i in v.issues))

    print("\n[useAdditionalCode refs aggregate the whole ancestor chain]")
    # E11.621 carries its own note (ulcer site L97.4-/L97.5-) AND inherits
    # E11's (Z79.4/Z79.84/Z79.85). First-hit-wins returned only the child's,
    # so Z79.4 was flagged 'unjustified' on diabetic-ulcer claims (live x3).
    refs = store.use_additional_code_refs("E11.621")
    check("E11.621 refs include BOTH its own (L97x) and inherited (Z79x) notes",
          any(r.startswith("L97") for r in refs) and any(r.startswith("Z794") for r in refs))
    v.issues = []
    v._check_unjustified_zcodes([{"code": "E11.621", "type": "primary"},
                                 {"code": "Z79.4", "type": "secondary"}])
    check("Z79.4 with E11.621 on claim → justified, no warning",
          "unjustified_zcode" not in cats(v))
    v.issues = []
    v._check_unjustified_zcodes([{"code": "L84", "type": "primary"},
                                 {"code": "Z79.4", "type": "secondary"}])
    check("Z79.4 with no recommending condition → still flagged",
          "unjustified_zcode" in cats(v))

    print("\n[advisory suppression propagates to validator issues (F8)]")
    # The scrubber advisory and the validator's 'unjustified_zcode' WARNING
    # are two emissions of ONE adjudicated claim about one code — a
    # suppression carrying validator_categories must retire both, or the
    # record ships the advisory suppressed in the scrub yet still active in
    # validation_issues/warnings (observed live on routine_00003, Z79.01).
    from app.models.schemas import ValidationIssue
    v.issues = [
        ValidationIssue(severity="WARNING", code="Z79.01",
                        category="unjustified_zcode", message="advisory"),
        ValidationIssue(severity="WARNING", code="Z79.4",
                        category="unjustified_zcode", message="other code"),
        ValidationIssue(severity="WARNING", code="Z79.01",
                        category="icd_specificity", message="other category"),
        ValidationIssue(severity="ERROR", code="Z79.01",
                        category="unjustified_zcode", message="never suppressible"),
    ]
    v._scrub_advisory_suppressions = []
    v._advisory_suppression_corrections = []
    # UNSCOPED directive (no clause): retains the pre-migration blunt reach
    # over every issue on (category, code), tagged or not. Every live rule
    # in the pack is this shape, which is why adding `clause` to
    # ValidationIssue is a no-op on today's behavior.
    v.suppress_scrub_advisory(
        "DOCUMENTATION", "Z79.01", rule_id="test-rule",
        authority="ICD-10-CM I.C.21.c.3", note="test",
        validator_categories=["unjustified_zcode"])
    v._apply_validator_advisory_suppressions()
    remaining = {(i.severity, i.code, i.category) for i in v.issues}
    check("matching WARNING (category+code) removed at source",
          ("WARNING", "Z79.01", "unjustified_zcode") not in remaining)
    check("same category, different code → survives",
          ("WARNING", "Z79.4", "unjustified_zcode") in remaining)
    check("same code, different category → survives",
          ("WARNING", "Z79.01", "icd_specificity") in remaining)
    check("ERROR severity never config-suppressible",
          ("ERROR", "Z79.01", "unjustified_zcode") in remaining)
    check("removal recorded as a correction (audit trail, not a vanishing)",
          any(c.get("category") == "validator_advisory_suppressed"
              and c.get("code") == "Z79.01"
              for c in v._advisory_suppression_corrections))
    check("suppression directive carries validator_categories",
          v._scrub_advisory_suppressions[0].get("validator_categories")
          == ["unjustified_zcode"])
    v.issues = []
    v._scrub_advisory_suppressions = []
    v._advisory_suppression_corrections = []

    print("\n[clause-scoped suppression cannot retire sibling assertions]")
    # The routine_00003 inversion: one filter emits several distinct
    # assertions about one code, and a rule verified against ONE of them
    # must not retire the others. The engine half enforces this at
    # engine._apply_advisory_suppressions; these cases pin the validator
    # half, whose fallback direction is the thing that must never invert.
    def _suppress(directive_clause, issue_clause):
        v.issues = [ValidationIssue(
            severity="WARNING", code="Z79.01",
            category="unjustified_zcode", message="advisory",
            clause=issue_clause)]
        v._scrub_advisory_suppressions = []
        v._advisory_suppression_corrections = []
        v.suppress_scrub_advisory(
            "DOCUMENTATION", "Z79.01", rule_id="test-rule",
            authority="ICD-10-CM I.C.21.c.3", note="test",
            clause=directive_clause,
            validator_categories=["unjustified_zcode"])
        v._apply_validator_advisory_suppressions()
        return not v.issues  # True == the advisory was suppressed

    check("clause-scoped directive retires the SAME clause",
          _suppress("documentation_prerequisite",
                    "documentation_prerequisite"))
    check("clause-scoped directive does NOT retire a sibling clause",
          not _suppress("documentation_prerequisite",
                        "coverage_composition"))
    # THE TRAP. If this ever passes, the fallback has inverted: untagged
    # issues would become suppressible by directives verified against an
    # assertion those issues may not even make — strictly worse than the
    # blunt (category, code) matching this migration exists to retire.
    check("clause-scoped directive does NOT retire an UNTAGGED issue",
          not _suppress("documentation_prerequisite", ""))
    # The migration ramp, in both directions.
    check("unscoped directive still retires an untagged issue",
          _suppress("", ""))
    check("unscoped directive still retires a tagged issue",
          _suppress("", "documentation_prerequisite"))
    check("correction records HOW the directive matched",
          _suppress("documentation_prerequisite",
                    "documentation_prerequisite")
          and v._advisory_suppression_corrections[0].get("match_scope")
          == "clause")
    check("category-wide match is recorded as such, not as clause-exact",
          _suppress("", "")
          and v._advisory_suppression_corrections[0].get("match_scope")
          == "category")
    v.issues = []
    v._scrub_advisory_suppressions = []
    v._advisory_suppression_corrections = []

    print("\n[duplicate diagnosis entries auto-removed]")
    icd_dup = [{"code": "Q66.89", "type": "secondary"}, {"code": "Q66.89", "type": "primary"},
               {"code": "M62.471", "type": "secondary"}]
    cr = {"icd10_codes": icd_dup}
    v.issues = []
    v._check_duplicate_diagnoses(icd_dup, cr)
    check("duplicate removed, primary designation wins",
          [(e["code"], e["type"]) for e in icd_dup] ==
          [("Q66.89", "primary"), ("M62.471", "secondary")]
          and any(i.category == "duplicate_diagnosis_removed" for i in v.issues))
    icd_nodup = [{"code": "Q66.89", "type": "primary"}, {"code": "M62.471", "type": "secondary"}]
    v.issues = []
    v._check_duplicate_diagnoses(icd_nodup, {"icd10_codes": icd_nodup})
    check("no duplicates → untouched, silent", len(icd_nodup) == 2 and not v.issues)

    print("\n[multiple primaries auto-demoted]")
    icd_mp = [{"code": "L60.0", "type": "primary"}, {"code": "L03.031", "type": "primary"},
              {"code": "E11.9", "type": "secondary"}]
    v.issues = []
    v._check_sequencing(icd_mp)
    check("first-listed primary kept, rest demoted",
          icd_mp[0]["type"] == "primary" and icd_mp[1]["type"] == "secondary"
          and any(i.category == "sequencing" and i.severity == "INFO"
                  and "AUTO-CORRECTED" in i.message for i in v.issues))

    print("\n[no primary at all auto-promoted (mirror of multi-primary)]")
    icd_np = [{"code": "E11.610", "type": "secondary"}, {"code": "M14.671", "type": "secondary"}]
    v.issues = []
    v._check_sequencing(icd_np)
    check("first-listed promoted to primary with INFO trail",
          icd_np[0]["type"] == "primary" and icd_np[1]["type"] == "secondary"
          and any(i.category == "sequencing" and i.severity == "INFO"
                  and "promoted" in i.message for i in v.issues))
    v.issues = []
    v._check_sequencing([])
    check("empty ICD list → silent, nothing invented", not v.issues)
    icd_ok = [{"code": "E11.610", "type": "primary"}, {"code": "Z79.4", "type": "secondary"}]
    v.issues = []
    v._check_sequencing(icd_ok)
    check("exactly one primary → untouched, silent",
          icd_ok[0]["type"] == "primary" and not v.issues)

    print("\n[empty box 24E backfilled deterministically]")
    icd_l = [{"code": "M19.171", "type": "primary"}, {"code": "S93.324S", "type": "secondary"}]
    cpt_l = [{"code": "28730", "linked_diagnoses": []}]
    v.issues = []
    v._check_cpt_dx_linkage(cpt_l, icd_l)
    check("unlinked CPT auto-linked from the claim's diagnoses (primary first)",
          cpt_l[0]["linked_diagnoses"][:1] == ["M19.171"]
          and any(i.category == "cpt_icd_linkage" and i.severity == "INFO" for i in v.issues))
    v.issues = []
    cpt_l2 = [{"code": "28730", "linked_diagnoses": []}]
    v._check_cpt_dx_linkage(cpt_l2, [])
    check("no diagnoses on the claim at all → WARNING stands, nothing invented",
          cpt_l2[0]["linked_diagnoses"] == []
          and any(i.category == "cpt_icd_linkage" and i.severity == "WARNING" for i in v.issues))

    print("\n[ruled-out physician codes are correct omissions]")
    v.issues = []
    v._check_physician_code_preservation(
        [], {"missing_physician_codes": [
            {"code": "L84", "description": "Corns and callosities", "section": "ASSESSMENT",
             "raw_text": "L84 — Corns and callosities (differential ruled out by black dot visualization)"}]},
        [], "")
    check("ruled-out differential → INFO trail, not a HIGH-risk omission",
          "physician_code_ruled_out" in cats(v) and "physician_code_missing" not in cats(v))
    v.issues = []
    v._check_physician_code_preservation(
        [], {"missing_physician_codes": [
            {"code": "L84", "description": "Corns and callosities", "section": "ASSESSMENT",
             "raw_text": "L84 — Corns and callosities, bilateral heels"}]},
        [], "")
    check("genuinely dropped physician code → WARNING stands",
          "physician_code_missing" in cats(v))

    print("\n[laterality defensibility is note-evidence-driven]")
    v.issues = []
    v._check_physician_code_preservation(
        [{"code": "11055", "modifiers": ["RT"], "code_source": "ai_inferred"}], {}, [],
        "Paring of callus, right foot, plantar aspect.")
    check("side word in the note → no warning (INFO trail instead)",
          "laterality_not_in_note" not in cats(v)
          and "laterality_confirmed_in_note" in cats(v))
    v.issues = []
    v._check_physician_code_preservation(
        [{"code": "11055", "modifiers": ["RT"], "code_source": "ai_inferred"}], {}, [],
        "Paring of callus, plantar aspect. No side documented anywhere.")
    check("side word absent from the note → warning fires",
          "laterality_not_in_note" in cats(v))

    print("\n[real ICD outside curated subset → review, not existence error]")
    # Find a code present in the store's full CDC tabular but absent from the
    # curated podiatry reference set (D48.1 class) — data-driven, no literal.
    outside = None
    row_iter = store.conn.execute(
        "SELECT code, description FROM icd10_tabular_desc "
        "WHERE length(code) >= 4 LIMIT 5000")
    for r in row_iter:
        dotted = r[0][:3] + "." + r[0][3:] if len(r[0]) > 3 else r[0]
        if not db.validate_icd10(dotted):
            outside = dotted
            break
    if outside:
        v.issues = []
        entry_os = {"code": outside, "type": "secondary"}
        v._check_code_existence([entry_os], [], [])
        check(f"tabular-valid code outside subset ({outside}) → WARNING not ERROR",
              any(i.category == "icd_outside_subset" and i.severity == "WARNING"
                  for i in v.issues)
              and not any(i.category == "code_existence" for i in v.issues))
        check("entry routed to review", entry_os.get("needs_review") is True)
    else:
        check("SKIP: no outside-subset code found in tabular sample", True)
    v.issues = []
    v._check_code_existence([{"code": "ZZ9.99", "type": "secondary"}], [], [])
    check("nonexistent code still ERRORs",
          any(i.category == "code_existence" and i.severity == "ERROR"
              for i in v.issues))

    print("\n[redundant RT/LT next to a digit modifier]")
    # Find a digit modifier whose own reference name states each side —
    # data-driven, no hand-typed T-modifier map (that map was once inverted).
    right_digit = left_digit = None
    for r in store.conn.execute("SELECT code FROM modifier"):
        mod_code = r[0]
        if mod_code in ("RT", "LT"):
            continue
        side = store.modifier_laterality(mod_code)
        if side == "RT" and right_digit is None:
            right_digit = mod_code
        elif side == "LT" and left_digit is None:
            left_digit = mod_code
        if right_digit and left_digit:
            break
    if right_digit:
        line = {"code": "11750", "modifiers": ["RT", right_digit]}
        v.issues = []
        v._check_redundant_laterality([line], [])
        check(f"RT stripped when {right_digit} (a right-side digit modifier) present",
              line["modifiers"] == [right_digit]
              and "redundant_laterality_removed" in cats(v))
        line2 = {"code": "11750", "modifiers": ["LT", right_digit]}
        v.issues = []
        v._check_redundant_laterality([line2], [])
        check(f"LT + {right_digit} (right-side digit) → contradiction ERROR, nothing stripped",
              "LT" in line2["modifiers"] and "laterality_contradiction" in cats(v))
        line3 = {"code": "11750", "modifiers": [right_digit]}
        v.issues = []
        v._check_redundant_laterality([line3], [])
        check("digit modifier alone → silent", not v.issues)
        if left_digit:
            line4 = {"code": "11750", "modifiers": ["RT", right_digit, left_digit]}
            v.issues = []
            v._check_redundant_laterality([line4], [])
            check("digit modifiers spanning both sides → ambiguous, untouched",
                  "RT" in line4["modifiers"] and not v.issues)
    else:
        check("SKIP: no sided digit modifier in reference data", True)

    print("\n[59/X stripped when the PTP table doesn't need it]")
    # Real indicator-1 PTP pair and a no-edit pair, both from the live table.
    pair_row = store.conn.execute(
        "SELECT col1, col2 FROM ncci_ptp WHERE modifier_indicator='1' "
        "AND col1 NOT LIKE '99%' AND col2 NOT LIKE '99%' LIMIT 1").fetchone()
    if pair_row:
        c1, c2 = pair_row[0], pair_row[1]
        col2_line = {"code": c2, "modifiers": ["59"]}
        v.issues = []
        v._check_unnecessary_separation_modifier([{"code": c1, "modifiers": []}, col2_line])
        check(f"59 kept on {c2} — indicator-1 edit with {c1}, no anatomic separation",
              "59" in col2_line["modifiers"]
              and "unnecessary_separation_modifier_removed" not in cats(v))
        if right_digit and left_digit:
            col2_sep = {"code": c2, "modifiers": ["59", right_digit]}
            v.issues = []
            v._check_unnecessary_separation_modifier(
                [{"code": c1, "modifiers": [left_digit]}, col2_sep])
            check("59 stripped — pair already separated by differing anatomic modifiers",
                  "59" not in col2_sep["modifiers"]
                  and "unnecessary_separation_modifier_removed" in cats(v))
    else:
        check("SKIP: no indicator-1 PTP pair in table", True)
    lone = {"code": "11720", "modifiers": ["59"]}
    v.issues = []
    v._check_unnecessary_separation_modifier([lone])
    check("59 on a line with no PTP partner at all → stripped",
          lone["modifiers"] == []
          and "unnecessary_separation_modifier_removed" in cats(v))

    print("\n[E/M level follows internally consistent MDM axes]")
    em_line = {"code": "99215",
               "mdm_details": {"mdm_level": "moderate", "problems_score": 3,
                               "data_score": 2, "risk_score": 3}}
    v.issues = []
    v._check_em_level_consistency([em_line])
    swapped_info = db.validate_cpt(em_line["code"]) or {}
    swapped_desc = (swapped_info.get("long_description")
                    or swapped_info.get("short_description") or "").lower()
    check("99215 with consistent moderate MDM (3,2,3) → swapped to the "
          "same-family moderate-level code",
          em_line["code"] != "99215" and "moderate" in swapped_desc
          and "em_level_corrected" in cats(v))
    em_line2 = {"code": "99215",
                "mdm_details": {"mdm_level": "moderate", "problems_score": 4,
                                "data_score": 4, "risk_score": 4}}
    v.issues = []
    v._check_em_level_consistency([em_line2])
    check("axes (4,4,4=high) contradict claimed 'moderate' → ERROR stands, no swap",
          em_line2["code"] == "99215" and "em_level_mismatch" in cats(v))
    em_line3 = {"code": "99214",
                "mdm_details": {"mdm_level": "moderate", "problems_score": 3,
                                "data_score": 3, "risk_score": 2}}
    v.issues = []
    v._check_em_level_consistency([em_line3])
    check("code already matches claimed level → silent, untouched",
          em_line3["code"] == "99214" and not v.issues)

    print("\n[PFS status-P supplies suppressed from the professional claim]")
    p_code_row = store.conn.execute(
        "SELECT code FROM global_period WHERE billing_status='P' "
        "AND code GLOB '[A-Z]*' LIMIT 1").fetchone()
    if p_code_row:
        p_code = p_code_row[0]
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_billability([], [{"code": p_code, "units": 1}])
        check(f"status-P code ({p_code}) → auto-suppressed with INFO",
              p_code in v._non_billable_codes_to_suppress
              and any(i.category == "billability" and i.severity == "INFO"
                      for i in v.issues))
    else:
        check("SKIP: no status-P code in PFS data", True)

    print("\n[E/M MDM risk-axis floor (AMA minor-surgery-with-risk-factors row)]")
    # minor-surgery CPT straight from the store's own global-period data
    minor_row = store.conn.execute(
        "SELECT code FROM global_period WHERE glob_days IN ('000','010') "
        "AND code GLOB '[0-9]*' LIMIT 1").fetchone()
    dm_code = next((c for c, info in db.icd10.items()
                    if "diabetes mellitus" in (info.get("description") or "").lower()), None)
    if minor_row and dm_code:
        minor_cpt = minor_row[0]
        em = {"code": "99213",
              "mdm_details": {"mdm_level": "low", "problems_score": 3,
                              "data_score": 1, "risk_score": 2}}
        lines = [em, {"code": minor_cpt, "modifiers": []}]
        v.issues = []
        v._check_em_mdm_risk_floor(lines, [{"code": dm_code}], date(2026, 1, 5), "")
        check("minor surgery + DM dx on claim → risk floored to 3, level recomputed",
              em["mdm_details"]["risk_score"] == 3
              and em["mdm_details"]["mdm_level"] == "moderate"
              and "em_mdm_risk_floor" in cats(v))
        v._check_em_level_consistency(lines)
        check("floored level flows into the descriptor-driven sibling swap (→ 99214)",
              em["code"] == "99214")
        # age >= 65 as the risk factor, no DM
        em2 = {"code": "99213",
               "mdm_details": {"mdm_level": "low", "problems_score": 3,
                               "data_score": 1, "risk_score": 2}}
        v.issues = []
        v._check_em_mdm_risk_floor([em2, {"code": minor_cpt}], [{"code": "L84"}],
                                   date(2026, 1, 5), "1941-03-02")
        check("minor surgery + age 84 → risk floored to 3",
              em2["mdm_details"]["risk_score"] == 3)
        # no risk factor → untouched
        em3 = {"code": "99213",
               "mdm_details": {"mdm_level": "low", "problems_score": 3,
                               "data_score": 1, "risk_score": 2}}
        v.issues = []
        v._check_em_mdm_risk_floor([em3, {"code": minor_cpt}], [{"code": "L84"}],
                                   date(2026, 1, 5), "1990-03-02")
        check("minor surgery, no risk factor → untouched",
              em3["mdm_details"]["risk_score"] == 2 and not v.issues)
        # no minor surgery → untouched even with DM
        em4 = {"code": "99213",
               "mdm_details": {"mdm_level": "low", "problems_score": 3,
                               "data_score": 1, "risk_score": 2}}
        v.issues = []
        v._check_em_mdm_risk_floor([em4], [{"code": dm_code}], date(2026, 1, 5), "")
        check("no minor surgery on claim → untouched", em4["mdm_details"]["risk_score"] == 2)
        # internally inconsistent level/axes → left for em_level_consistency
        em5 = {"code": "99213",
               "mdm_details": {"mdm_level": "high", "problems_score": 3,
                               "data_score": 1, "risk_score": 2}}
        v.issues = []
        v._check_em_mdm_risk_floor([em5, {"code": minor_cpt}], [{"code": dm_code}],
                                   date(2026, 1, 5), "")
        check("claimed level contradicts axes → no bump (ERROR path owns it)",
              em5["mdm_details"]["risk_score"] == 2)
    else:
        check("SKIP: no minor-surgery/DM fixture in data", True)

    print("\n[E/M MDM problems-axis floor (AMA 2+ chronic illnesses row)]")
    chronic_rows = store.conn.execute(
        "SELECT code FROM icd10_chronic WHERE chronic=1 LIMIT 2").fetchall()
    nonchronic_row = store.conn.execute(
        "SELECT code FROM icd10_chronic WHERE chronic=0 LIMIT 1").fetchone()
    if len(chronic_rows) == 2 and nonchronic_row:
        cc1, cc2 = chronic_rows[0][0], chronic_rows[1][0]
        nc = nonchronic_row[0]
        pf = {"code": "99213",
              "mdm_details": {"mdm_level": "low", "problems_score": 2,
                              "data_score": 3, "risk_score": 2}}
        v.issues = []
        v._check_em_mdm_problems_floor([pf], [{"code": cc1}, {"code": cc2}])
        check("2 CCIR-chronic dx on claim → problems floored to 3, level recomputed",
              pf["mdm_details"]["problems_score"] == 3
              and pf["mdm_details"]["mdm_level"] == "moderate"
              and "em_mdm_problems_floor" in cats(v))
        v._check_em_level_consistency([pf])
        check("floored level flows into the sibling swap (→ 99214)",
              pf["code"] == "99214")
        # only one chronic dx → untouched
        pf2 = {"code": "99213",
               "mdm_details": {"mdm_level": "low", "problems_score": 2,
                               "data_score": 3, "risk_score": 2}}
        v.issues = []
        v._check_em_mdm_problems_floor([pf2], [{"code": cc1}, {"code": nc}])
        check("1 chronic + 1 non-chronic dx → untouched",
              pf2["mdm_details"]["problems_score"] == 2 and not v.issues)
        # already at/above moderate → silent
        pf3 = {"code": "99214",
               "mdm_details": {"mdm_level": "moderate", "problems_score": 4,
                               "data_score": 3, "risk_score": 3}}
        v.issues = []
        v._check_em_mdm_problems_floor([pf3], [{"code": cc1}, {"code": cc2}])
        check("problems already >= 3 → untouched",
              pf3["mdm_details"]["problems_score"] == 4 and not v.issues)
        # internally inconsistent level/axes → left for em_level_consistency
        pf4 = {"code": "99213",
               "mdm_details": {"mdm_level": "high", "problems_score": 2,
                               "data_score": 3, "risk_score": 2}}
        v.issues = []
        v._check_em_mdm_problems_floor([pf4], [{"code": cc1}, {"code": cc2}])
        check("claimed level contradicts axes → no bump (ERROR path owns it)",
              pf4["mdm_details"]["problems_score"] == 2)
    else:
        check("SKIP: icd10_chronic not populated (CCIR file missing)", True)

    print("\n[guidance-code laterality strip]")
    guide = {"code": "76942", "modifiers": ["RT"]}
    v.issues = []
    v._check_guidance_laterality([guide, {"code": "20612", "modifiers": ["RT"]}])
    check("RT stripped from 76942 (guidance descriptor, bilat_surg 0)",
          "RT" not in guide["modifiers"] and "guidance_laterality_removed" in cats(v))
    sided = {"code": "73660", "modifiers": ["RT"]}  # X-ray toe: bilat_surg 3, not guidance
    v.issues = []
    v._check_guidance_laterality([sided])
    check("73660 (diagnostic radiograph, bilateral concept applies) → untouched",
          sided["modifiers"] == ["RT"] and not v.issues)
    # descriptor merely MENTIONS guidance ('...including image guidance') —
    # a sided injection, not a guidance service. Live false positive.
    if db.validate_cpt("0232T"):
        prp = {"code": "0232T", "modifiers": ["RT"]}
        v.issues = []
        v._check_guidance_laterality([prp])
        check("0232T (injection 'including image guidance') → untouched",
              prp["modifiers"] == ["RT"] and not v.issues)
    else:
        check("SKIP: 0232T not in CPT data", True)

    print("\n[digit-specify supply modifier]")
    sup_row = None
    for c, info in db.hcpcs.items():
        if "specify digit" in ((info.get("long_description") or info.get("description") or "")).lower():
            sup_row = c
            break
    if sup_row:
        # 1) digit derived from the claim's own procedure modifiers
        sup = {"code": sup_row, "modifiers": ["RT"]}
        v.issues = []
        v._check_digit_supply_modifier(
            [{"code": "11055", "modifiers": ["T5"]}], [sup], "")
        check(f"{sup_row} RT → replaced by the claim's unique digit modifier T5",
              sup["modifiers"] == ["T5"] and "digit_modifier_applied" in cats(v))
        # 2) digit derived from the note's own words (right hallux → T5)
        sup2 = {"code": sup_row, "modifiers": ["RT"]}
        v.issues = []
        v._check_digit_supply_modifier(
            [{"code": "97597", "modifiers": ["RT"]}], [sup2],
            "Dressing applied to the right hallux ulcer.")
        check(f"{sup_row} RT + note 'right hallux' → T5 from the modifier's own name",
              sup2["modifiers"] == ["T5"] and "digit_modifier_applied" in cats(v))
        # 2b) numeric ordinal spelling ('right 5th toe', live note 019) → T9
        sup2b = {"code": sup_row, "modifiers": ["RT"]}
        v.issues = []
        v._check_digit_supply_modifier(
            [{"code": "11055", "modifiers": ["RT"]}], [sup2b],
            "Painful corn right 5th toe dorsum, debrided and padded.")
        check(f"{sup_row} + note 'right 5th toe' → T9 from the modifier's own name",
              sup2b["modifiers"] == ["T9"] and "digit_modifier_applied" in cats(v))
        # 3) side conflict → ERROR, no silent rewrite
        sup3 = {"code": sup_row, "modifiers": ["LT"]}
        v.issues = []
        v._check_digit_supply_modifier(
            [{"code": "11055", "modifiers": ["T5"]}], [sup3], "")
        check("LT supply vs right-sided digit → conflict flagged, untouched",
              sup3["modifiers"] == ["LT"] and "digit_modifier_side_conflict" in cats(v))
        # 4) nothing derivable → WARNING only
        sup4 = {"code": sup_row, "modifiers": ["RT"]}
        v.issues = []
        v._check_digit_supply_modifier([{"code": "99213", "modifiers": []}], [sup4], "")
        check("no digit derivable → flagged for coder, untouched",
              sup4["modifiers"] == ["RT"] and "digit_modifier_required" in cats(v))
    else:
        check("SKIP: no 'specify digit' supply in HCPCS data", True)

    print("\n[use-additional companion promoted to billed diagnoses]")
    esrd = {"code": "N18.6", "type": "primary"}
    icd_arr = [esrd]
    result = {"icd10_codes": icd_arr}
    v.issues = []
    v._check_missing_use_additional_code(
        icd_arr, result, "ESRD patient, on chronic renal dialysis three times weekly.")
    added = [e for e in icd_arr if e.get("source_section") == "validator:use_additional_code"]
    check("N18.6 + documented dialysis → companion added to icd_codes as secondary",
          len(added) == 1 and added[0].get("type") == "secondary"
          and added[0].get("needs_review") is True)
    check("companion NOT parked in supporting_conditions",
          not result.get("supporting_conditions"))

    print("\n[separation-modifier placement (NCCI column-2)]")
    pair = store.conn.execute(
        "SELECT col1, col2 FROM ncci_ptp WHERE modifier_indicator='1' "
        "AND col1 NOT LIKE '99%' AND col2 NOT LIKE '99%' LIMIT 1").fetchone()
    if pair:
        c1, c2 = pair["col1"], pair["col2"]
        l1, l2 = {"code": c1, "modifiers": ["59"]}, {"code": c2, "modifiers": []}
        v.issues = []
        v._check_separation_modifier_placement([l1, l2])
        check(f"59 on column-1 {c1} → moved to column-2 {c2}",
              "59" not in l1["modifiers"] and "59" in l2["modifiers"]
              and "separation_modifier_moved" in cats(v))
        # already correctly placed → untouched
        l1b, l2b = {"code": c1, "modifiers": []}, {"code": c2, "modifiers": ["59"]}
        v.issues = []
        v._check_separation_modifier_placement([l1b, l2b])
        check("59 already on column-2 → untouched",
              l2b["modifiers"] == ["59"] and not v.issues)
        # on both lines → column-1 copy dropped, column-2 kept
        l1c, l2c = {"code": c1, "modifiers": ["59"]}, {"code": c2, "modifiers": ["59"]}
        v.issues = []
        v._check_separation_modifier_placement([l1c, l2c])
        check("59 on both lines → column-1 copy dropped",
              "59" not in l1c["modifiers"] and l2c["modifiers"] == ["59"])
    else:
        check("SKIP: no indicator-1 PTP pair in store", True)
    pair0 = store.conn.execute(
        "SELECT col1, col2 FROM ncci_ptp WHERE modifier_indicator='0' "
        "AND col1 NOT LIKE '99%' AND col2 NOT LIKE '99%' LIMIT 1").fetchone()
    if pair0:
        l1d = {"code": pair0["col1"], "modifiers": ["59"]}
        l2d = {"code": pair0["col2"], "modifiers": []}
        v.issues = []
        v._check_separation_modifier_placement([l1d, l2d])
        check("indicator-0 pair → placement untouched (strip check owns it)",
              l1d["modifiers"] == ["59"] and not v.issues)
    else:
        check("SKIP: no indicator-0 PTP pair in store", True)

    print("\n[CPT digit modifier from linked diagnosis]")
    dx_full = None   # dx naming side + WHICH toe outright
    dx_side = None   # dx naming side + toe-scope only ('Cellulitis of right toe')
    for code_c, info in db.icd10.items():
        d = (info.get("description") or "").lower()
        if dx_full is None and "right great toe" in d:
            dx_full = code_c
        if dx_side is None and d.endswith("right toe"):
            dx_side = code_c
        if dx_full and dx_side:
            break
    if dx_full:
        line = {"code": "10060", "modifiers": ["RT"], "linked_diagnoses": [dx_full]}
        v.issues = []
        v._check_cpt_digit_laterality([line], [])
        check(f"10060 RT + linked '{dx_full}' (right great toe) → upgraded to T5",
              line["modifiers"] == ["T5"] and "digit_modifier_applied" in cats(v))
        conflict_line = {"code": "10060", "modifiers": ["LT"],
                         "linked_diagnoses": [dx_full]}
        v.issues = []
        v._check_cpt_digit_laterality([conflict_line], [])
        check("LT line vs right-great-toe dx → conflict flagged, untouched",
              conflict_line["modifiers"] == ["LT"]
              and "digit_modifier_side_conflict" in cats(v))
    else:
        check("SKIP: no right-great-toe dx in dataset", True)
    if dx_side:
        # dx pins side + toe-scope; the note's unique digit mention resolves it
        line2 = {"code": "10060", "modifiers": ["RT"], "linked_diagnoses": [dx_side]}
        v.issues = []
        v._check_cpt_digit_laterality(
            [line2], [], "Paronychia of the right great toe with abscess, I&D performed.")
        check(f"10060 RT + linked '{dx_side}' (right toe) + note 'right great toe' → T5",
              line2["modifiers"] == ["T5"] and "digit_modifier_applied" in cats(v))
        # ambiguous note (two digits named) → untouched
        line3 = {"code": "10060", "modifiers": ["RT"], "linked_diagnoses": [dx_side]}
        v.issues = []
        v._check_cpt_digit_laterality(
            [line3], [], "Right great toe and right second toe both involved.")
        check("two same-side digits in note → ambiguous, untouched",
              line3["modifiers"] == ["RT"] and not v.issues)
    else:
        check("SKIP: no side+toe-scope dx in dataset", True)
    plain = {"code": "97597", "modifiers": ["RT"], "linked_diagnoses": ["L97.511"]}
    v.issues = []
    v._check_cpt_digit_laterality(
        [plain], [], "Debridement of the right heel ulcer.")
    check("linked dx names no toe → untouched",
          plain["modifiers"] == ["RT"] and not v.issues)

    print("\n[with/without axis arbitration]")
    ww_own = db.validate_icd10("S90.122A")
    ww_sib = db.validate_icd10("S90.222A")
    if ww_own and ww_sib:
        e1 = {"code": "S90.122A", "type": "secondary"}
        v.issues = []
        v._check_with_without_axis(
            [e1], "Contusion of the toe with damage to the nail plate observed.")
        check("'without damage to nail' billed, damage documented → swapped to 'with'",
              e1["code"] == "S90.222A" and "with_without_axis_corrected" in cats(v)
              and e1.get("needs_review") is True)
        e2 = {"code": "S90.222A", "type": "secondary"}
        v.issues = []
        v._check_with_without_axis(
            [e2], "Contusion of the left lesser toe. Nail intact, no drainage.")
        check("'with damage to nail' billed, damage never documented → swapped to 'without'",
              e2["code"] == "S90.122A" and "with_without_axis_corrected" in cats(v))
        e3 = {"code": "S90.122A", "type": "secondary"}
        v.issues = []
        v._check_with_without_axis(
            [e3], "Contusion of the left lesser toe. No nail damage seen.")
        check("'without' billed, damage only in a negated span → kept",
              e3["code"] == "S90.122A" and not v.issues)
        # counterpart already billed → no swap (duplicate guard)
        e4 = {"code": "S90.122A", "type": "secondary"}
        e5 = {"code": "S90.222A", "type": "secondary"}
        v.issues = []
        v._check_with_without_axis(
            [e4, e5], "Contusion with damage to the nail plate.")
        check("counterpart already on claim → no swap",
              e4["code"] == "S90.122A" and e5["code"] == "S90.222A")
    else:
        check("SKIP: S90.122A/S90.222A pair not in dataset", True)

    print("\n[E/M minor-procedure -25 exposure (observe-only)]")
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_em_minor_procedure_bundling(
        [{"code": "99214", "modifiers": ["25"], "linked_diagnoses": ["B35.1", "L60.3"]},
         {"code": "11750", "modifiers": ["T5"], "linked_diagnoses": ["B35.1", "L60.3"]}],
        [])
    check("established E/M, all dxs procedure-addressed → exposure flagged, NOT suppressed",
          "em_minor_procedure_bundled" in cats(v)
          and not v._non_billable_codes_to_suppress)
    v.issues = []
    v._check_em_minor_procedure_bundling(
        [{"code": "99215", "modifiers": ["25"],
          "linked_diagnoses": ["E11.621", "Z79.4"]},
         {"code": "11042", "modifiers": ["TA"], "linked_diagnoses": ["E11.621"]}],
        [])
    check("E/M with separately managed dx (Z79.4) → silent", not v.issues)
    v.issues = []
    v._check_em_minor_procedure_bundling(
        [{"code": "99204", "modifiers": ["25"], "linked_diagnoses": ["B35.1"]},
         {"code": "11750", "modifiers": ["T5"], "linked_diagnoses": ["B35.1"]}],
        [])
    check("new-patient E/M → silent", not v.issues)

    print("\n[imaging note-evidence gate]")
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_imaging_note_evidence(
        [{"code": "73630", "modifiers": ["RT"]}],
        "Debridement of mycotic nails performed. No imaging obtained today.")
    check("73630 with no x-ray/radiograph anywhere in note → suppressed",
          "73630" in v._non_billable_codes_to_suppress
          and "imaging_not_documented" in cats(v))
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_imaging_note_evidence(
        [{"code": "73630", "modifiers": ["RT"]}],
        "X-ray of the right foot, 3 views, obtained today showing no fracture.")
    check("73630 with documented x-ray → kept",
          not v._non_billable_codes_to_suppress and not v.issues)
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_imaging_note_evidence(
        [{"code": "76000", "modifiers": []}],
        "Intraoperative fluoroscopy confirmed alignment.")
    check("76000 with documented fluoroscopy → kept",
          not v._non_billable_codes_to_suppress)
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_imaging_note_evidence(
        [{"code": "11750", "modifiers": ["T5"]}], "No imaging words at all.")
    check("non-radiology code → untouched", not v.issues)

    print("\n[A-code supply RT/LT strip]")
    sup_a = {"code": "A4570", "modifiers": ["RT"]}
    sup_l = {"code": "L3260", "modifiers": ["RT"]}
    v.issues = []
    v._check_supply_laterality_strip([sup_a, sup_l])
    check("A4570 RT stripped (materials line)",
          sup_a["modifiers"] == [] and "supply_laterality_removed" in cats(v))
    check("L3260 (L-code fitted device) untouched", sup_l["modifiers"] == ["RT"])

    print("\n[ICD laterality corrected to the claim's own side]")
    if db.validate_icd10("S90.111A") and db.validate_icd10("S90.112A"):
        dx_wrong = {"code": "S90.111A", "type": "primary"}
        v.issues = []
        v._check_icd_cpt_laterality_agreement(
            [{"code": "11730", "modifiers": ["TA"], "linked_diagnoses": ["S90.111A"]}],
            [dx_wrong])
        check("right-toe dx on all-TA (left) claim → swapped to left sibling",
              dx_wrong["code"] == "S90.112A"
              and "icd_laterality_corrected" in cats(v))
        dx_mixed = {"code": "S90.111A", "type": "primary"}
        v.issues = []
        v._check_icd_cpt_laterality_agreement(
            [{"code": "11730", "modifiers": ["TA"], "linked_diagnoses": ["S90.111A"]},
             {"code": "11740", "modifiers": ["T5"], "linked_diagnoses": ["S90.111A"]}],
            [dx_mixed])
        check("mixed-side claim → no swap (warning only)",
              dx_mixed["code"] == "S90.111A"
              and "icd_laterality_corrected" not in cats(v))
        dx_dup = {"code": "S90.111A", "type": "secondary"}
        dx_present = {"code": "S90.112A", "type": "primary"}
        v.issues = []
        v._check_icd_cpt_laterality_agreement(
            [{"code": "11730", "modifiers": ["TA"],
              "linked_diagnoses": ["S90.112A"]}],
            [dx_dup, dx_present])
        check("opposite sibling already billed → no swap",
              dx_dup["code"] == "S90.111A")
    else:
        check("SKIP: S90.111A/112A pair not in dataset", True)

    print("\n[CPT digit from the code's own toe descriptor]")
    if db.validate_cpt("28126"):
        line_toe = {"code": "28126", "modifiers": ["LT"], "units": 1,
                    "linked_diagnoses": ["M86.672"]}
        v.issues = []
        v._check_cpt_digit_laterality(
            [line_toe], [], "Resection of the phalangeal base of the left hallux.")
        check("28126 LT ('each toe' descriptor) + note 'left hallux' → TA",
              line_toe["modifiers"] == ["TA"]
              and "digit_modifier_applied" in cats(v))
        line_multi = {"code": "28126", "modifiers": ["LT"], "units": 3,
                      "linked_diagnoses": ["M86.672"]}
        v.issues = []
        v._check_cpt_digit_laterality(
            [line_multi], [], "Resection of the phalangeal base of the left hallux.")
        check("units>1 on an 'each toe' code → skipped (spans digits)",
              line_multi["modifiers"] == ["LT"])
    else:
        check("SKIP: 28126 not in CPT data", True)

    print("\n[onset-qualifier axis (acute/chronic/congenital)]")
    if db.validate_icd10("M86.672") and db.validate_icd10("M86.9"):
        # neither acute nor chronic documented → Index bare-term default M86.9
        e_chr = {"code": "M86.672", "type": "primary"}
        v.issues = []
        v._check_onset_qualifier_axis(
            [e_chr],
            "Osteomyelitis of the left hallux distal phalanx confirmed on MRI. "
            "Bone debrided.")
        check("chronic osteomyelitis billed, no qualifier documented → M86.9 default",
              e_chr["code"] == "M86.9"
              and "onset_qualifier_undocumented" in cats(v)
              and e_chr.get("needs_review") is True)
        # billed qualifier documented in the same sentence → untouched
        e_doc = {"code": "M86.672", "type": "primary"}
        v.issues = []
        v._check_onset_qualifier_axis(
            [e_doc],
            "Chronic osteomyelitis of the left ankle and foot, longstanding.")
        check("'chronic osteomyelitis' documented → kept",
              e_doc["code"] == "M86.672" and not v.issues)
        # counterpart qualifier documented → transplant within the axis
        e_swap = {"code": "M86.672", "type": "primary"}
        v.issues = []
        v._check_onset_qualifier_axis(
            [e_swap],
            "Acute osteomyelitis of the left ankle and foot, one week of symptoms.")
        check("chronic billed, 'acute osteomyelitis' documented → acute sibling M86.172",
              e_swap["code"] == "M86.172"
              and "onset_qualifier_undocumented" in cats(v))
    else:
        check("SKIP: M86.672/M86.9 not in dataset", True)
    if db.validate_icd10("Q84.5") and db.validate_icd10("L60.2"):
        e_cong = {"code": "Q84.5", "type": "secondary"}
        v.issues = []
        v._check_onset_qualifier_axis(
            [e_cong],
            "Hypertrophic nail of the right hallux, thickened for two years.")
        check("congenital Q84.5 billed, congenital never documented → acquired L60.2",
              e_cong["code"] == "L60.2"
              and "onset_qualifier_undocumented" in cats(v))
        # acquired counterpart already billed → congenital line removed as duplicate
        icd_dup = [{"code": "Q84.5", "type": "secondary"},
                   {"code": "L60.2", "type": "primary"}]
        v.issues = []
        v._check_onset_qualifier_axis(
            icd_dup, "Hypertrophic nail of the right hallux, thickened.")
        check("acquired spelling already on claim → congenital line removed",
              [e["code"] for e in icd_dup] == ["L60.2"])
    else:
        check("SKIP: Q84.5/L60.2 not in dataset", True)
    # category-definitional qualifier (every L97 code is a 'chronic ulcer') → never touched
    if db.validate_icd10("L97.511"):
        e_l97 = {"code": "L97.511", "type": "primary"}
        v.issues = []
        v._check_onset_qualifier_axis(
            [e_l97],
            "Non-pressure ulcer of the right foot limited to breakdown of skin.")
        check("L97 'chronic' is category-definitional → untouched",
              e_l97["code"] == "L97.511"
              and "onset_qualifier_undocumented" not in cats(v))
    else:
        check("SKIP: L97.511 not in dataset", True)

    print("\n[radiograph view-count arbitration]")
    if db.validate_cpt("73620") and db.validate_cpt("73630"):
        # no views documented → complete-study code downgraded to fewest-views sibling
        rx = {"code": "73630", "modifiers": ["RT"]}
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_radiograph_view_count(
            [rx], "Post-op X-ray confirms adequate bony resection.")
        check("73630 (min 3 views) with no views documented → swapped to 73620",
              rx["code"] == "73620" and "radiograph_view_count" in cats(v)
              and rx.get("needs_review") is True)
        # three projections named → complete study supported
        rx2 = {"code": "73630", "modifiers": ["RT"]}
        v.issues = []
        v._check_radiograph_view_count(
            [rx2], "X-ray right foot: AP, lateral and oblique views obtained.")
        check("73630 with AP/lateral/oblique named → kept",
              rx2["code"] == "73630" and not v.issues)
        # explicit numeric count
        rx3 = {"code": "73630", "modifiers": ["RT"]}
        v.issues = []
        v._check_radiograph_view_count(
            [rx3], "Radiographs of the right foot, 3 views, unremarkable.")
        check("73630 with '3 views' stated → kept",
              rx3["code"] == "73630" and not v.issues)
        # two views documented → 2-view sibling is the supportable member
        rx4 = {"code": "73630", "modifiers": ["RT"]}
        v.issues = []
        v._check_radiograph_view_count(
            [rx4], "AP and lateral views of the right foot show no fracture.")
        check("73630 with only AP/lateral documented → swapped to 73620",
              rx4["code"] == "73620")
        # both family members billed with no views → higher one suppressed
        rx5 = {"code": "73630", "modifiers": ["RT"]}
        rx6 = {"code": "73620", "modifiers": ["RT"]}
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_radiograph_view_count(
            [rx5, rx6], "X-ray of the right foot obtained.")
        check("73630+73620 both billed, no views → 73630 suppressed as duplicate",
              "73630" in v._non_billable_codes_to_suppress)
        v._non_billable_codes_to_suppress = set()
    else:
        check("SKIP: 73620/73630 not in dataset", True)

    print("\n[E/M MDM problems-axis ceiling]")
    em_hi = {"code": "99215", "mdm_details": {
        "mdm_level": "high", "problems_score": 4, "data_score": 2,
        "risk_score": 4}}
    v.issues = []
    v._check_em_mdm_problems_ceiling(
        [em_hi], [{"code": "L60.2"}, {"code": "E11.40"}],
        "Pincer nail deformity with pressure ulceration. Wound debrided. "
        "Vascular consult requested before elective procedure.")
    check("problems=4 with no severe-exacerbation/threat documentation → capped to 3",
          em_hi["mdm_details"]["problems_score"] == 3
          and "em_mdm_problems_ceiling" in cats(v))
    check("2-of-3 recomputed to moderate",
          em_hi["mdm_details"]["mdm_level"] == "moderate")
    # threat term inside a billed dx's own descriptor → high stands
    gangrene_dx = None
    for c, info in (getattr(db, "icd10", {}) or {}).items():
        if "gangrene" in (info.get("description") or "").lower() and c.startswith("I70"):
            gangrene_dx = c if "." in c else c[:3] + "." + c[3:]
            break
    if gangrene_dx:
        em_ok = {"code": "99215", "mdm_details": {
            "mdm_level": "high", "problems_score": 4, "data_score": 2,
            "risk_score": 4}}
        v.issues = []
        v._check_em_mdm_problems_ceiling(
            [em_ok], [{"code": gangrene_dx}],
            "Critical limb ischemia with gangrene developing distally.")
        check(f"problems=4 with {gangrene_dx} (descriptor names gangrene) → kept high",
              em_ok["mdm_details"]["problems_score"] == 4 and not v.issues)
    else:
        check("SKIP: no I70 gangrene dx in dataset", True)
    # sentence-level threat term tied to a billed dx → high stands
    em_sent = {"code": "99215", "mdm_details": {
        "mdm_level": "high", "problems_score": 4, "data_score": 2,
        "risk_score": 4}}
    v.issues = []
    v._check_em_mdm_problems_ceiling(
        [em_sent], [{"code": "E11.40"}],
        "Severe exacerbation of diabetic neuropathy with new ulceration.")
    check("'severe exacerbation' adjacent to billed dx terms → kept high",
          em_sent["mdm_details"]["problems_score"] == 4 and not v.issues)
    # internally inconsistent structure → untouched (consistency check owns it)
    em_inc = {"code": "99215", "mdm_details": {
        "mdm_level": "low", "problems_score": 4, "data_score": 2,
        "risk_score": 4}}
    v.issues = []
    v._check_em_mdm_problems_ceiling(
        [em_inc], [{"code": "L60.2"}], "Routine nail care visit.")
    check("claimed level contradicts axes → ceiling skipped",
          em_inc["mdm_details"]["problems_score"] == 4)

    print("\n[same-site PTP bundling]")
    if db.check_ncci("11740", "29550"):
        c1 = [{"code": "11740", "modifiers": ["T6"], "units": 1},
              {"code": "29550", "modifiers": ["T6", "59"], "units": 1}]
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_same_site_ptp_bundling(c1)
        check("29550 (col-2, same T6 as 11740) suppressed despite 59",
              "29550" in v._non_billable_codes_to_suppress
              and "same_site_ptp_bundled" in cats(v))
        # different digits → distinct sites → edit legitimately bypassed
        c2 = [{"code": "11740", "modifiers": ["T6"], "units": 1},
              {"code": "29550", "modifiers": ["T7", "59"], "units": 1}]
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_same_site_ptp_bundling(c2)
        check("different digit modifiers → kept",
              not v._non_billable_codes_to_suppress)
        # plain RT/LT is not exact-site proof (a foot has many sites)
        c3 = [{"code": "11740", "modifiers": ["RT"], "units": 1},
              {"code": "29550", "modifiers": ["RT", "59"], "units": 1}]
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_same_site_ptp_bundling(c3)
        check("shared RT only (not digit-exact) → kept",
              not v._non_billable_codes_to_suppress)
        # multi-unit line spans digits → skipped
        c4 = [{"code": "11740", "modifiers": ["T6"], "units": 1},
              {"code": "29550", "modifiers": ["T6", "59"], "units": 2}]
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_same_site_ptp_bundling(c4)
        check("multi-unit column-2 line → skipped",
              not v._non_billable_codes_to_suppress)
    else:
        check("SKIP: 11740/29550 PTP edit not in dataset", True)
    if db.check_ncci("11730", "11740"):
        c5 = [{"code": "11730", "modifiers": ["TA"], "units": 1},
              {"code": "11740", "modifiers": ["TA", "59"], "units": 1}]
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_same_site_ptp_bundling(c5)
        check("11740 (col-2 of 11730 pair, same TA) suppressed",
              "11740" in v._non_billable_codes_to_suppress)
        v._non_billable_codes_to_suppress = set()
    else:
        check("SKIP: 11730/11740 PTP edit not in dataset", True)

    print("\n[digit modifier de-scope (mirror of the upgrade)]")
    d97 = {"code": "97597", "modifiers": ["T5"], "units": 1,
           "linked_diagnoses": ["L97.511", "E11.40"]}
    v.issues = []
    v._check_digit_modifier_scope([d97])
    check("97597 T5 with foot-level dx linkage → normalized to RT",
          d97["modifiers"] == ["RT"]
          and "digit_modifier_descoped" in cats(v))
    d117 = {"code": "11750", "modifiers": ["T5"], "units": 1,
            "linked_diagnoses": ["B35.1"]}
    v.issues = []
    v._check_digit_modifier_scope([d117])
    check("11750 (nail procedure by descriptor) keeps T5",
          d117["modifiers"] == ["T5"] and not v.issues)
    d114 = {"code": "11740", "modifiers": ["T6"], "units": 1,
            "linked_diagnoses": []}
    v.issues = []
    v._check_digit_modifier_scope([d114])
    check("11740 (subungual by descriptor) keeps T6",
          d114["modifiers"] == ["T6"] and not v.issues)
    d90 = {"code": "97597", "modifiers": ["T5"], "units": 1,
           "linked_diagnoses": ["S90.121A"]}
    v.issues = []
    v._check_digit_modifier_scope([d90])
    check("digit-level linked dx (great toe contusion) keeps T5",
          d90["modifiers"] == ["T5"] and not v.issues)

    print("\n[view-count family anatomy fix]")
    if db.validate_cpt("73660") and db.validate_cpt("71120"):
        t = {"code": "73660", "modifiers": ["T5"]}
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_radiograph_view_count([t], "X-ray of the toe obtained.")
        check("73660 never swaps cross-anatomy (71120 sternum bug)",
              t["code"] != "71120")
        # suppressed lines must not be mutated afterwards
        t2 = {"code": "73630", "modifiers": ["RT"]}
        v.issues = []
        v._non_billable_codes_to_suppress = {"73630"}
        v._check_radiograph_view_count([t2], "X-ray of the foot obtained.")
        check("already-suppressed line is not mutated by view count",
              t2["code"] == "73630")
        v._non_billable_codes_to_suppress = set()
    else:
        check("SKIP: 73660/71120 not in dataset", True)

    print("\n[debridement tissue-token ownership]")
    d9 = {"code": "11042", "modifiers": ["LT"]}
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_debridement_depth(
        [d9], "Deep wound debridement performed with bone biopsy obtained; "
              "specimens sent for culture and histopathology.")
    check("'bone biopsy' in a debridement sentence does NOT set bone depth",
          d9["code"] == "97597")
    d10 = {"code": "97597", "modifiers": ["LT"]}
    v.issues = []
    v._check_debridement_depth(
        [d10], "Debridement carried down to necrotic bone at the ulcer base.")
    check("real bone debridement still upgrades",
          d10["code"] == "11044")
    v._non_billable_codes_to_suppress = set()

    print("\n[imaging context gate (prior/future/intraop)]")
    note_008 = ("X-ray at prior visit reveals subungual exostosis. "
                "Intraoperative fluoroscopy used to confirm complete resection. "
                "Post-op X-ray confirms adequate bony resection. "
                "Post-op X-ray at 6 weeks.")
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_imaging_context(
        [{"code": "73620", "modifiers": ["RT"]},
         {"code": "76000", "modifiers": []},
         {"code": "28124", "modifiers": ["T5"]}], note_008)
    check("x-ray only in prior/confirmation/future contexts → 73620 suppressed",
          "73620" in v._non_billable_codes_to_suppress
          and "imaging_not_separately_billable" in cats(v))
    check("intraop-confirmation fluoroscopy alongside surgery → 76000 suppressed",
          "76000" in v._non_billable_codes_to_suppress)
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_imaging_context(
        [{"code": "73620", "modifiers": ["RT"]}],
        "X-ray of the right foot obtained today: no fracture identified "
        "on AP and lateral views.")
    check("diagnostic study rendered today (negated finding) → kept",
          not v._non_billable_codes_to_suppress and not v.issues)
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_imaging_context(
        [{"code": "76000", "modifiers": []}],
        "Fluoroscopy used to confirm needle placement during evaluation.")
    check("confirmation wording with NO surgical CPT on claim → kept",
          not v._non_billable_codes_to_suppress)

    print("\n[debridement depth arbitration]")
    fam = v._debridement_family()
    if "97597" in fam and "11042" in fam and "11044" in fam:
        check("family derived from descriptors (97597<11042<11044)",
              fam["97597"][0] < fam["11042"][0] < fam["11044"][0])
        # no tissue level documented → shallowest member
        d1 = {"code": "11042", "modifiers": ["LT"]}
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_debridement_depth(
            [d1], "Deep wound debridement performed. Bone biopsy obtained.")
        check("11042 with no tissue level in a debridement sentence → 97597",
              d1["code"] == "97597"
              and "debridement_depth_mismatch" in cats(v)
              and d1.get("needs_review") is True)
        # documented subcutaneous depth → 11042 stands
        d2 = {"code": "11042", "modifiers": ["LT"]}
        v.issues = []
        v._check_debridement_depth(
            [d2], "Wound debrided sharply down to subcutaneous tissue.")
        check("11042 with subcutaneous documented → kept",
              d2["code"] == "11042" and not v.issues)
        # documented bone debridement → 97597 upgraded
        d3 = {"code": "97597", "modifiers": ["LT"]}
        v.issues = []
        v._check_debridement_depth(
            [d3], "Debridement of necrotic bone at the ulcer base performed.")
        check("97597 with bone documented in debridement sentence → 11044",
              d3["code"] == "11044")
        # both family members billed → duplicate suppressed
        d4 = {"code": "11042", "modifiers": ["LT"]}
        d5 = {"code": "97597", "modifiers": ["LT"]}
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_debridement_depth(
            [d4, d5], "Wound debridement performed with curette.")
        check("11042+97597 both billed, no depth → 11042 suppressed as dup",
              "11042" in v._non_billable_codes_to_suppress)
        v._non_billable_codes_to_suppress = set()
    else:
        check("SKIP: debridement family not derivable from CPT data", True)

    print("\n[diabetes-ulcer with-convention completion]")
    if db.validate_icd10("E11.40") and db.validate_icd10("L97.511") \
            and db.validate_icd10("E11.621"):
        icd_dm = [{"code": "E11.40", "type": "secondary"},
                  {"code": "L97.511", "type": "primary"}]
        v.issues = []
        v._check_diabetes_ulcer_combination(icd_dm)
        codes_dm = [e["code"] for e in icd_dm]
        check("E11.40 + L97.511 → E11.621 auto-added per with-convention",
              "E11.621" in codes_dm
              and "with_convention_completion" in cats(v))
        # already satisfied → silent
        icd_ok = [{"code": "E11.621", "type": "secondary"},
                  {"code": "L97.511", "type": "primary"}]
        v.issues = []
        v._check_diabetes_ulcer_combination(icd_ok)
        check("combination already on claim → silent",
              len(icd_ok) == 2 and not v.issues)
        # no ulcer on claim → silent
        icd_no = [{"code": "E11.40", "type": "primary"}]
        v.issues = []
        v._check_diabetes_ulcer_combination(icd_no)
        check("diabetes without ulcer → silent", len(icd_no) == 1 and not v.issues)
    else:
        check("SKIP: E11.40/L97.511/E11.621 not in dataset", True)

    print("\n[marginal-secondary demotion]")
    if db.validate_icd10("I73.9"):
        cr = {"supporting_conditions": []}
        icd_m = [{"code": "M86.9", "type": "primary"},
                 {"code": "I73.9", "type": "secondary"}]
        v.issues = []
        v._check_marginal_secondary_demotion(
            icd_m, cr, "Osteomyelitis distal phalanx left hallux; "
                       "non-pressure ulcer with exposure of bone")
        check("I73.9 (unspecified, not in assessment) → demoted to supporting",
              [e["code"] for e in icd_m] == ["M86.9"]
              and any(e.get("code") == "I73.9"
                      for e in cr["supporting_conditions"])
              and "marginal_secondary_demoted" in cats(v))
        # documented in the anchor → kept
        icd_k = [{"code": "M86.9", "type": "primary"},
                 {"code": "I73.9", "type": "secondary"}]
        v.issues = []
        v._check_marginal_secondary_demotion(
            icd_k, {"supporting_conditions": []},
            "Peripheral vascular disease with rest pain; osteomyelitis")
        check("I73.9 documented in assessment → kept",
              len(icd_k) == 2 and not v.issues)
        # primary type exempt even if generic
        icd_p = [{"code": "I73.9", "type": "primary"}]
        v.issues = []
        v._check_marginal_secondary_demotion(
            icd_p, {"supporting_conditions": []}, "Foot pain evaluation")
        check("primary diagnosis exempt from demotion", len(icd_p) == 1)
        # validator-added companion exempt
        icd_v = [{"code": "M86.9", "type": "primary"},
                 {"code": "I73.9", "type": "secondary",
                  "source_section": "validator:use_additional_code"}]
        v.issues = []
        v._check_marginal_secondary_demotion(
            icd_v, {"supporting_conditions": []}, "Osteomyelitis")
        check("validator-added companion exempt", len(icd_v) == 2)
        # specific-entity code (no 'unspecified'/'other specified') untouched
        icd_s = [{"code": "M86.9", "type": "primary"},
                 {"code": "L60.2", "type": "secondary"}]
        v.issues = []
        v._check_marginal_secondary_demotion(
            icd_s, {"supporting_conditions": []}, "Osteomyelitis")
        check("specific-entity secondary (L60.2) untouched", len(icd_s) == 2)
    else:
        check("SKIP: I73.9 not in dataset", True)
    # abbreviation rescue: 'T2DM' in the assessment documents E11.x
    if db.validate_icd10("E11.40"):
        icd_dm2 = [{"code": "L60.2", "type": "primary"},
                   {"code": "E11.40", "type": "secondary"}]
        v.issues = []
        v._check_marginal_secondary_demotion(
            icd_dm2, {"supporting_conditions": []},
            "Onychogryphosis / Pincer nail; T2DM with diabetic neuropathy")
        check("E11.40 rescued by 'T2DM' abbreviation in assessment",
              len(icd_dm2) == 2 and not v.issues)
    else:
        check("SKIP: E11.40 not in dataset", True)
    # index-synonym rescue: M89.8X7 kept when 'exostosis' is in the anchor
    if db.validate_icd10("M89.8X7"):
        icd_x = [{"code": "L60.3", "type": "primary"},
                 {"code": "M89.8X7", "type": "secondary"}]
        v.issues = []
        v._check_marginal_secondary_demotion(
            icd_x, {"supporting_conditions": []},
            "Subungual exostosis, right hallux; nail dystrophy")
        check("M89.8X7 rescued by Index synonym 'exostosis' in assessment",
              len(icd_x) == 2 and not v.issues)
    else:
        check("SKIP: M89.8X7 not in dataset", True)

    print("\n[icd sibling arbitration: site and side invariants]")
    # Site invariance (live, note 008): M25.771 'Osteophyte, right ankle'
    # was swapped to M25.511 'Pain in right SHOULDER' — the df ubiquity cut
    # dropped both site words, so a cross-joint pair looked like a pure
    # attribute axis. The undocumented site must block the swap.
    note_exos = ("53-year-old male with painful right hallux. Subungual "
                 "exostosis, right hallux; nail dystrophy. Pain with pressure. "
                 "Palpable subungual mass on X-ray (exostosis of distal phalanx).")
    if db.validate_icd10("M25.771"):
        icd_site = [{"code": "M89.8X7", "type": "primary"},
                    {"code": "M25.771", "type": "secondary"}]
        v.issues = []
        v._check_icd_sibling_descriptor(icd_site, note_exos)
        check("M25.771 (ankle) NOT swapped to a different joint's sibling",
              icd_site[1]["code"] == "M25.771")
    else:
        check("SKIP: M25.771 not in dataset", True)
    # Side invariance (live, note 006): S90.122A 'left lesser toe' swapped
    # to S90.111A 'RIGHT great toe' on a left-hallux note — left/right are
    # stopwords, so the opposite-side sibling tied and sorted first. The
    # swap must land on the same-side sibling (S90.112A, left great toe).
    if db.validate_icd10("S90.122A") and db.validate_icd10("S90.112A"):
        # live 006 wording: the note names the hallux (great toe) but never
        # the word 'contusion' — writing 'contusion' in the fixture would
        # document the billed code's own Index chain and correctly block
        # the swap as ambiguous
        note_hallux = ("Left hallux nail pain after marathon — black nail. "
                       "Left hallux: subungual hematoma 70% of nail plate. "
                       "Subungual hematoma; onycholysis, left hallux.")
        icd_side = [{"code": "S90.122A", "type": "primary"}]
        v.issues = []
        v._check_icd_sibling_descriptor(icd_side, note_hallux)
        check("S90.122A (left lesser) swaps to LEFT great-toe sibling, "
              "never the right-side one",
              icd_side[0]["code"] == "S90.112A")
    else:
        check("SKIP: S90.12x/S90.11x not in dataset", True)

    print("\n[primary designation from procedure linkage]")
    # Live (note 004): type=primary flipped between two diagnoses across
    # runs of an identical note. The non-E/M procedure line's first pointer
    # is the deterministic anchor.
    if db.validate_icd10("E11.621") and db.validate_icd10("L60.2"):
        icd_pd = [{"code": "L60.2", "type": "primary"},
                  {"code": "E11.621", "type": "secondary"}]
        cpt_pd = [{"code": "99214", "modifiers": ["25"],
                   "linked_diagnoses": ["L60.2"]},
                  {"code": "97597", "modifiers": ["RT"],
                   "linked_diagnoses": ["E11.621"]}]
        v.issues = []
        v._check_primary_designation(icd_pd, cpt_pd)
        check("procedure-anchored dx promoted to primary",
              icd_pd[1]["type"] == "primary" and icd_pd[0]["type"] == "secondary"
              and "primary_designation" in cats(v))
        # already primary → silent no-op
        v.issues = []
        v._check_primary_designation(icd_pd, cpt_pd)
        check("stable when anchor already primary", not v.issues)
        # current primary is itself first-pointed by a procedure → keep
        icd_keep = [{"code": "L60.2", "type": "primary"},
                    {"code": "E11.621", "type": "secondary"}]
        cpt_keep = [{"code": "97597", "modifiers": [],
                     "linked_diagnoses": ["E11.621"]},
                    {"code": "11750", "modifiers": ["TA"],
                     "linked_diagnoses": ["L60.2"]}]
        v.issues = []
        v._check_primary_designation(icd_keep, cpt_keep)
        check("ambiguous (procedures disagree) → no re-designation",
              icd_keep[0]["type"] == "primary" and not v.issues)
    else:
        check("SKIP: E11.621/L60.2 not in dataset", True)
    # codeFirst etiology outranks the pointer: manifestation first-pointed,
    # billed etiology takes the designation.
    if db.validate_icd10("L97.511") and db.validate_icd10("E11.621"):
        refs = [r.replace(".", "").upper()
                for r in store.code_first_etiology_refs("L97.511")]
        if any("E11621".startswith(r) or r.startswith("E11") for r in refs):
            icd_cf = [{"code": "L97.511", "type": "primary"},
                      {"code": "E11.621", "type": "secondary"}]
            cpt_cf = [{"code": "97597", "modifiers": [],
                       "linked_diagnoses": ["L97.511"]}]
            v.issues = []
            v._check_primary_designation(icd_cf, cpt_cf)
            check("codeFirst etiology promoted over first-pointed manifestation",
                  icd_cf[1]["type"] == "primary")
        else:
            check("SKIP: L97.511 carries no E11-family codeFirst ref", True)
    else:
        check("SKIP: L97.511/E11.621 not in dataset", True)

    print("\n[NCCI comprehensive upgrade]")
    # Live (notes 001/006): the note documents the comprehensive service
    # (phenol matrixectomy / partial avulsion) but a run bills only its
    # NCCI column-2 component. The line converges on the comprehensive code.
    edit_1150 = db.check_ncci("11750", "11730")
    if edit_1150 and edit_1150.get("code1") == "11750":
        note_matrix = (
            "PROCEDURE(S) PERFORMED: Digital block. Medial one-fifth of nail "
            "plate removed using nail splitter and elevator. Phenol 88% applied "
            "to exposed matrix for three 30-second applications. "
            "PLAN/TREATMENT: Partial nail avulsion with chemical matrixectomy "
            "(phenol 88%) for permanent removal, left hallux medial border.")
        cpt_up = [{"code": "11730", "modifiers": ["TA"], "units": 1}]
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_unbilled_descriptor_match(cpt_up, [], note_matrix)
        check("component upgraded to documented comprehensive code",
              cpt_up[0]["code"] == "11750"
              and "component_upgraded_to_comprehensive" in cats(v))
        # comprehensive already billed → warning only, no double line
        cpt_both = [{"code": "11750", "modifiers": ["TA"], "units": 1},
                    {"code": "11730", "modifiers": ["TA"], "units": 1}]
        v.issues = []
        v._check_unbilled_descriptor_match(cpt_both, [], note_matrix)
        check("no upgrade when comprehensive already on the claim",
              cpt_both[1]["code"] == "11730"
              and "component_upgraded_to_comprehensive" not in cats(v))
    else:
        check("SKIP: 11750/11730 not an NCCI pair in dataset", True)

    print("\n[primary designation: unconditional codeFirst arm]")
    # Live (note 004, 1/3 runs): L97.5- ulcer typed primary over its billed
    # E11.621 etiology; the codeFirst convention is mandatory sequencing
    # regardless of the procedure-pointer shape.
    if db.validate_icd10("L97.511") and db.validate_icd10("E11.621"):
        icd_cf = [
            {"code": "L97.511", "type": "primary"},
            {"code": "E11.621", "type": "secondary"},
        ]
        v.issues = []
        v._check_primary_designation(icd_cf, [])
        check("codeFirst etiology promoted to primary without procedure lines",
              icd_cf[1]["type"] == "primary" and icd_cf[0]["type"] == "secondary")
    else:
        check("SKIP: L97.511/E11.621 not in dataset", True)

    print("\n[CPT family pathology-axis arbitration (eg-parenthetical)]")
    # Live (note 008): a documented subungual EXOSTOSIS resection flapped
    # between the 'bone cyst or benign tumor' code (28108 — no cyst and no
    # tumor anywhere in the note) and the partial-excision code whose own
    # eg-parenthetical names bossing (28124), the descriptor register for
    # exostosis. CPT's reference note under 28108 directs bossing/exostosis
    # of a phalanx to 28124.
    if db.validate_cpt("28108") and db.validate_cpt("28124"):
        note_ex = ("Palpable subungual mass confirmed on X-ray (exostosis of "
                   "distal phalanx). No signs of malignancy. Total nail plate "
                   "avulsed. Exostosis visualized and resected with rongeur "
                   "until smooth cortical bone remaining. Zadek matrixectomy "
                   "completed.")
        cpt_pa = [{"code": "28108", "modifiers": ["T5"], "units": 1,
                   "linked_diagnoses": ["M89.8X7"]}]
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_cpt_family_pathology_axis(cpt_pa, note_ex)
        check("exostosis note: cyst/tumor code swapped to the bossing code",
              cpt_pa[0]["code"] == "28124"
              and any(i.category == "cpt_pathology_axis" for i in v.issues))
        check("swap keeps modifiers/pointers and flags review",
              cpt_pa[0]["modifiers"] == ["T5"]
              and cpt_pa[0].get("needs_review") is True)
        # billed code's OWN eg-pathology documented → correct member, no swap
        note_cyst = ("X-ray demonstrates a well-circumscribed bone cyst of "
                     "the distal phalanx. Cyst excised with curette; cavity "
                     "curetted to healthy bone.")
        cpt_cy = [{"code": "28108", "modifiers": ["T5"], "units": 1,
                   "linked_diagnoses": []}]
        v.issues = []
        v._check_cpt_family_pathology_axis(cpt_cy, note_cyst)
        check("bone-cyst note: cyst/tumor code stays (own terms documented)",
              cpt_cy[0]["code"] == "28108")
        # idempotent: the correct code with its eg-pathology documented
        cpt_ok = [{"code": "28124", "modifiers": ["T5"], "units": 1,
                   "linked_diagnoses": []}]
        v.issues = []
        v._check_cpt_family_pathology_axis(cpt_ok, note_ex)
        check("correct bossing code untouched on the exostosis note",
              cpt_ok[0]["code"] == "28124" and not v.issues)
    else:
        check("SKIP: 28108/28124 not in dataset", True)

    print("\n[comprehensive upgrade: noun/verb register bridge (-al stem)]")
    # Live (note 001): 'nail plate removed ... phenol applied to exposed
    # matrix' failed to upgrade 11730 → 11750 because the descriptor's
    # 'removal' never matched the note's 'removed'.
    if db.validate_cpt("11730") and db.validate_cpt("11750"):
        note_001 = ("Medial one-fifth of nail plate removed using nail "
                    "splitter and elevator. Phenol 88% applied to exposed "
                    "matrix for three 30-second applications. Partial nail "
                    "avulsion with chemical matrixectomy, left hallux.")
        cpt_up = [{"code": "11730", "modifiers": ["TA"], "units": 1,
                   "linked_diagnoses": ["L60.0"]}]
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_unbilled_descriptor_match(cpt_up, [], note_001)
        check("'removed'/'removal' register bridged → 11730 upgraded to 11750",
              cpt_up[0]["code"] == "11750")
    else:
        check("SKIP: 11730/11750 not in dataset", True)

    print("\n[comprehensive completion: non-kindred documented col-1 act]")
    # Live (note 006): trephination + partial avulsion both documented; the
    # run billing only the evacuation (11740) under-coded the avulsion. The
    # completion arm adds the documented comprehensive act; the same-site
    # PTP check then bundles the component exactly as in the runs that
    # billed both.
    if db.validate_cpt("11740") and db.validate_cpt("11730"):
        note_006 = ("Electrocautery trephination performed x2. Separated "
                    "distal 5mm of nail plate trimmed and removed. Nail "
                    "trephination; partial nail avulsion of separated "
                    "portion, left hallux.")
        cpt_cc = [{"code": "11740", "modifiers": ["TA"], "units": 1,
                   "linked_diagnoses": ["S60.121A"]}]
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_unbilled_descriptor_match(cpt_cc, [], note_006)
        added = [e for e in cpt_cc if e.get("code") == "11730"]
        check("documented avulsion added alongside billed evacuation",
              len(added) == 1
              and any(i.category == "comprehensive_completion_added"
                      for i in v.issues))
        check("added line inherits the component's site modifiers/pointers",
              added and added[0]["modifiers"] == ["TA"]
              and added[0]["linked_diagnoses"] == ["S60.121A"])
        # same-site bundling then removes the column-2 evacuation
        v._check_same_site_ptp_bundling(cpt_cc)
        check("PTP verdict applied: evacuation (column-2, same site) bundles",
              "11740" in v._non_billable_codes_to_suppress)
        v._non_billable_codes_to_suppress = set()
        # scattered token grazes must NOT add a line: same tokens spread
        # across distant sentences
        note_scatter = ("Nail plate intact. Discussed future avulsion as an "
                        "option if symptoms persist. Partial improvement "
                        "noted. Separated hematoma evacuated by trephination.")
        cpt_sc = [{"code": "11740", "modifiers": ["TA"], "units": 1,
                   "linked_diagnoses": []}]
        v.issues = []
        v._check_unbilled_descriptor_match(cpt_sc, [], note_scatter)
        check("no single-sentence dedicated act → no line added",
              not any(e.get("code") == "11730" for e in cpt_sc))
        # deferral override (live, note 004): the plan names the act without
        # a futurity cue, but the body defers it — never add a deferred act
        note_deferred = ("Wound debridement performed with curette and "
                         "scissors. Formal avulsion deferred pending "
                         "vascular clearance. Bilateral partial nail "
                         "avulsion with chemical matrixectomy, right hallux; "
                         "wound debridement.")
        cpt_df = [{"code": "97597", "modifiers": ["RT"], "units": 1,
                   "linked_diagnoses": []}]
        v.issues = []
        v._check_unbilled_descriptor_match(cpt_df, [], note_deferred)
        check("act named in plan but deferred in body → no line added",
              not any(e.get("code") == "11730" for e in cpt_df))
    else:
        check("SKIP: 11740/11730 not in dataset", True)

    print("\n[ulcer severity tier (ICD final-character axis)]")
    # Live (note 004): 'wound depth 2mm, Wagner grade 1' ulcer flapped
    # L97.511/L97.512 — no fat-layer documentation, so only the
    # skin-breakdown tier is supportable.
    if db.validate_icd10("L97.511") and db.validate_icd10("L97.512"):
        note_004 = ("Bilateral border pressure ulcerations (0.3 cm x 0.4 cm "
                    "each, wound depth 2mm, no tunneling, Wagner grade 1). "
                    "Non-pressure ulcer right hallux.")
        icd_ut = [{"code": "L97.512", "type": "secondary"}]
        v.issues = []
        v._check_ulcer_severity_tier(icd_ut, note_004)
        check("no deeper-tissue evidence → fat-layer code tiered down to skin",
              icd_ut[0]["code"] == "L97.511")
        icd_ok = [{"code": "L97.512", "type": "secondary"}]
        v.issues = []
        v._check_ulcer_severity_tier(
            icd_ok, "Plantar ulcer with fat layer exposed, no necrosis.")
        check("documented fat-layer exposure → fat-tier code stands",
              icd_ok[0]["code"] == "L97.512")
        icd_dup = [{"code": "L97.512", "type": "secondary"},
                   {"code": "L97.511", "type": "secondary"}]
        v.issues = []
        v._check_ulcer_severity_tier(icd_dup, note_004)
        check("supportable tier already billed → no duplicate swap",
              icd_dup[0]["code"] == "L97.512")
    else:
        check("SKIP: L97.511/L97.512 not in dataset", True)

    print("\n[operative-field debridement gate (NCCI Ch.1)]")
    # Live (note 010): 'wound margins debrided to bleeding tissue' inside a
    # phalangectomy op note produced a standalone 97597 in 1 of 3 runs.
    if db.validate_cpt("97597") and db.validate_cpt("28124"):
        note_010 = ("Distal phalangectomy left hallux — bone transected at "
                    "metaphysis. Wound margins debrided to bleeding tissue. "
                    "Wound left open for secondary intention.")
        cpt_of = [{"code": "28124", "modifiers": ["TA"], "units": 1},
                  {"code": "97597", "modifiers": ["LT", "59"], "units": 1}]
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_operative_field_debridement(cpt_of, note_010)
        check("margin-only debridement of a same-claim surgery suppressed",
              "97597" in v._non_billable_codes_to_suppress)
        v._non_billable_codes_to_suppress = set()
        # a distinct wound debridement (no margin wording) stays billable
        note_004d = ("Wound debridement performed with curette and scissors. "
                     "Silver-impregnated dressing applied.")
        cpt_kd = [{"code": "97597", "modifiers": ["RT"], "units": 1}]
        v.issues = []
        v._check_operative_field_debridement(cpt_kd, note_004d)
        check("distinct wound debridement (no margin context) kept",
              "97597" not in v._non_billable_codes_to_suppress)
        # margin wording WITHOUT a surgical procedure on the claim → kept
        cpt_ns = [{"code": "97597", "modifiers": ["RT"], "units": 1}]
        v.issues = []
        v._check_operative_field_debridement(cpt_ns, note_010)
        check("margin wording but no surgery on claim → kept",
              "97597" not in v._non_billable_codes_to_suppress)
    else:
        check("SKIP: 97597/28124 not in dataset", True)

    print("\n[primary designation: suppressed lines don't vote]")
    # Live (note 002): a paronychia I&D the same-site PTP check bundled away
    # still voted with its first pointer, blocking the anchor on the
    # matrixectomy's ingrown-nail code — the final claim is identical with
    # or without the removed line, so the designation must be too.
    if db.validate_icd10("L60.0") and db.validate_icd10("L03.031"):
        icd_sv = [
            {"code": "L03.031", "type": "primary"},
            {"code": "L60.0", "type": "secondary"},
        ]
        cpt_sv = [
            {"code": "11750", "modifiers": ["T5"], "units": 1,
             "linked_diagnoses": ["L60.0"]},
            {"code": "10060", "modifiers": ["T5"], "units": 1,
             "linked_diagnoses": ["L03.031"]},
        ]
        v.issues = []
        v._non_billable_codes_to_suppress = {"10060"}
        v._bundled_codes_to_suppress = set()
        v._check_primary_designation(icd_sv, cpt_sv)
        check("bundled-away I&D's pointer ignored; matrixectomy anchors L60.0",
              icd_sv[1]["type"] == "primary" and icd_sv[0]["type"] == "secondary")
        # same claim, nothing suppressed → both lines vote → ambiguous, no touch
        icd_sv2 = [
            {"code": "L03.031", "type": "primary"},
            {"code": "L60.0", "type": "secondary"},
        ]
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_primary_designation(icd_sv2, cpt_sv)
        check("with both lines live the anchor is ambiguous — designation kept",
              icd_sv2[0]["type"] == "primary")
        v._non_billable_codes_to_suppress = set()
    else:
        check("SKIP: L60.0/L03.031 not in dataset", True)

    print("\n[assessment diagnosis completion (guideline IV.J)]")
    if db.validate_icd10("L60.1"):
        icd_ac = [{"code": "S90.112A", "type": "primary"}]
        v.issues = []
        v._check_assessment_dx_completion(
            icd_ac, "Subungual hematoma; Onycholysis, left hallux")
        check("assessment-listed 'onycholysis' added as L60.1",
              any(e.get("code") == "L60.1" for e in icd_ac)
              and "assessment_dx_omitted" in cats(v))
        # negated term is not documentation
        icd_neg = [{"code": "S90.112A", "type": "primary"}]
        v.issues = []
        v._check_assessment_dx_completion(icd_neg, "No onycholysis observed")
        check("negated assessment term adds nothing",
              len(icd_neg) == 1 and not v.issues)
        # family already billed → skip
        icd_fam = [{"code": "L60.2", "type": "primary"}]
        v.issues = []
        v._check_assessment_dx_completion(icd_fam, "Onycholysis, left hallux")
        check("same-category code already billed → no duplicate add",
              len(icd_fam) == 1 and not v.issues)
    else:
        check("SKIP: L60.1 not in dataset", True)

    print("\n[NCCI comprehensive upgrade: kinship guard]")
    # Live regression (note 004): a performed wound debridement (97597) was
    # rewritten into a nail avulsion (11730) the note explicitly DEFERRED —
    # the NCCI edit alone (97597 is column-2 of 11730) is bundling policy,
    # not identity of work. Zero descriptor-token overlap must block it.
    note_004 = (
        "Involuted pincer nail, right hallux. Non-pressure ulcer right hallux. "
        "PROCEDURE(S) PERFORMED: Wound debridement performed with curette and "
        "scissors. Bilateral nail borders gently decompressed with nail "
        "elevator. Formal avulsion deferred pending vascular clearance. "
        "PLAN/TREATMENT: Bilateral partial nail avulsion with chemical "
        "matrixectomy, right hallux; wound debridement.")
    cpt_kin = [{"code": "97597", "modifiers": ["RT"], "units": 1}]
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_unbilled_descriptor_match(cpt_kin, [], note_004)
    check("97597 (debridement) NOT rewritten into an unrelated nail code",
          cpt_kin[0]["code"] == "97597")

    print("\n[digit modifier: add-missing arm]")
    # Live (note 004): 11730 flapped between ['RT'] and [] across runs —
    # both shapes must converge on the digit the note pins ('right hallux',
    # linked dx names the right side).
    if db.validate_icd10("L97.511"):
        note_dig = ("Pincer nail causing pain — right hallux. Partial nail "
                    "avulsion performed on the right hallux.")
        for start_mods in (["RT"], []):
            entry = {"code": "11730", "modifiers": list(start_mods), "units": 1,
                     "linked_diagnoses": ["L97.511"]}
            v.issues = []
            v._check_cpt_digit_laterality([entry], [], note_dig)
            check(f"11730 {start_mods or '(bare)'} → T5",
                  entry["modifiers"] == ["T5"])
    else:
        check("SKIP: L97.511 not in dataset", True)

    print("\n[HCPCS coverage-code suppression (payer-gated)]")
    # Live (note 004, Humana MA): S8450 (coverage code 'I' — not payable by
    # Medicare) flapped present-in-2-of-3 runs; on a Medicare-bound payer
    # the line denies in any circumstance, so it must be suppressed
    # deterministically. Other payers keep the review flag.
    if store.hcpcs_noncoverage_reason("S8450"):
        for follows, expect_suppressed in ((True, True), (False, False)):
            hc = [{"code": "S8450", "modifiers": ["T5"], "units": 1}]
            v.issues = []
            v._non_billable_codes_to_suppress = set()
            v._payer_follows_medicare = follows
            v._check_billability([], hc)
            check(f"S8450 payer_follows_medicare={follows} → "
                  f"{'suppressed' if expect_suppressed else 'review-flagged'}",
                  ("S8450" in v._non_billable_codes_to_suppress) == expect_suppressed)
        v._payer_follows_medicare = False
    else:
        check("SKIP: S8450 has no coverage row", True)

    print("\n[hematoma-release term equivalence (trephination = evacuation)]")
    # Live (note 005): 'electrocautery trephination ... decompression' is the
    # same intervention 11740's descriptor spells 'Evacuation of subungual
    # hematoma'; the run billing generic 10140 must upgrade like the others.
    words, low = v._note_evidence(
        "Nail trephination performed; immediate decompression of subungual hematoma.")
    check("'evacuation' documented via trephination/decompression equivalence",
          v._desc_documented("evacuation", words, low))

    print("\n[digit scope: 'phalanx' descriptors + same-side site identity]")
    # Live (note 010): 20240 'Biopsy, bone ... phalanx' kept a generic LT
    # beside the same toe's phalangectomy (28124 TA) — 'phalang\\w*' missed
    # the singular 'phalanx', and LT-vs-TA was treated as different sites,
    # so the indicator-1 PTP edit was bypassed on the same great toe.
    note_010 = ("Distal phalangectomy left hallux — bone transected at the "
                "metaphysis. Three intraoperative bone cultures taken.")
    cpt_010 = [
        {"code": "28124", "modifiers": ["TA"], "units": 1,
         "linked_diagnoses": ["M86.9"]},
        {"code": "20240", "modifiers": ["LT"], "units": 1,
         "linked_diagnoses": ["M86.9"]},
    ]
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_cpt_digit_laterality(cpt_010, [], note_010)
    check("20240's LT upgraded to TA (descriptor 'phalanx', note names "
          "the left hallux)", cpt_010[1]["modifiers"] == ["TA"])
    v._check_same_site_ptp_bundling(cpt_010)
    check("same-toe bone biopsy bundled into the phalangectomy",
          "20240" in v._non_billable_codes_to_suppress)
    # _sites_distinct semantics
    check("TA vs TA: same site", not v._sites_distinct({"TA"}, {"TA"}))
    check("LT vs TA: generic side does not separate from a same-side digit",
          not v._sites_distinct({"LT"}, {"TA"}))
    check("RT vs TA: opposite sides separate", v._sites_distinct({"RT"}, {"TA"}))
    check("RT vs LT: opposite generic sides separate",
          v._sites_distinct({"RT"}, {"LT"}))
    check("T5 vs TA: different digits separate",
          v._sites_distinct({"T5"}, {"TA"}))

    print("\n[E/M MDM data-axis floor (AMA Extensive-data row)]")
    note_data = (
        "Non-healing wound left hallux, diabetic patient. ESR 78, CRP 42, "
        "WBC 13.2. HbA1c 10.8. ABI 0.72 left. Infectious disease and "
        "vascular surgery consulted. IV vancomycin initiated empirically "
        "per ID recommendation.")
    em = {"code": "99214", "mdm_details": {
        "mdm_level": "moderate", "problems_score": 3,
        "data_score": 3, "risk_score": 4}}
    v.issues = []
    v._check_em_mdm_data_floor([em], [], note_data)
    check("data axis floored to 4 (3+ results + external discussion), "
          "level recomputes to high",
          em["mdm_details"]["data_score"] == 4
          and em["mdm_details"]["mdm_level"] == "high"
          and "em_mdm_data_floor" in cats(v))
    # without the consult/recommendation evidence, no floor
    em2 = {"code": "99214", "mdm_details": {
        "mdm_level": "moderate", "problems_score": 3,
        "data_score": 3, "risk_score": 4}}
    v.issues = []
    v._check_em_mdm_data_floor(
        [em2], [], "ESR 78, CRP 42, WBC 13.2. Wound debrided today.")
    check("no floor without external-discussion evidence",
          em2["mdm_details"]["data_score"] == 3 and not v.issues)
    # medication doses are not test results
    em3 = {"code": "99213", "mdm_details": {
        "mdm_level": "low", "problems_score": 2,
        "data_score": 2, "risk_score": 2}}
    v.issues = []
    v._check_em_mdm_data_floor(
        [em3], [],
        "Insulin NPH 40 units BID. Metformin XR 500 mg. Gabapentin TID 300 mg. "
        "Podiatry consulted; plan per PCP recommendation.")
    check("medication doses don't count as test results",
          em3["mdm_details"]["data_score"] == 2 and not v.issues)

    print("\n[MDM claimed-level normalization]")
    # live shape (note 009): the whole 2-of-3 derivation emitted as the level
    mdm_free = {"mdm_level": ("high (problems) / moderate (data, risk) → "
                              "overall MDM moderate by 2-of-3 rule")}
    check("derivation sentence → the 'overall' conclusion wins",
          v._mdm_claimed_level(mdm_free) == "moderate"
          and mdm_free["mdm_level"] == "moderate")
    check("exact bare level → unchanged",
          v._mdm_claimed_level({"mdm_level": "High"}) == "high")
    check("single unambiguous level word in prose → extracted",
          v._mdm_claimed_level({"mdm_level": "MDM is moderate overall"}) == "moderate")
    mdm_amb = {"mdm_level": "somewhere between low and moderate"}
    check("ambiguous multi-level text without a conclusion → '' (skip)",
          v._mdm_claimed_level(mdm_amb) == ""
          and mdm_amb["mdm_level"] == "somewhere between low and moderate")
    check("empty → ''", v._mdm_claimed_level({"mdm_level": ""}) == "")

    print("\n[E/M MDM risk-axis HIGH floor (toxicity-monitored parenteral therapy)]")
    note_vanc = ("Deep wound debridement performed. Infectious disease and "
                 "vascular surgery consulted. IV vancomycin initiated "
                 "empirically per ID recommendation.")
    rh = {"code": "99214", "mdm_details": {
        "mdm_level": "moderate", "problems_score": 4,
        "data_score": 3, "risk_score": 3}}
    v.issues = []
    v._check_em_mdm_risk_high_floor([rh], note_vanc)
    check("IV vancomycin initiation → risk floored to 4, level recomputes to high",
          rh["mdm_details"]["risk_score"] == 4
          and rh["mdm_details"]["mdm_level"] == "high"
          and "em_mdm_risk_high_floor" in cats(v))
    # oral antibiotics: no parenteral route in the sentence → no floor
    rh2 = {"code": "99213", "mdm_details": {
        "mdm_level": "low", "problems_score": 2,
        "data_score": 2, "risk_score": 2}}
    v.issues = []
    v._check_em_mdm_risk_high_floor(
        [rh2], "Cephalexin 500 mg PO QID started for cellulitis.")
    check("oral antibiotic start → untouched",
          rh2["mdm_details"]["risk_score"] == 2 and not v.issues)
    # route+agent present but only CONSIDERED for the future → no floor
    rh3 = {"code": "99214", "mdm_details": {
        "mdm_level": "moderate", "problems_score": 3,
        "data_score": 3, "risk_score": 3}}
    v.issues = []
    v._check_em_mdm_risk_high_floor(
        [rh3], "Will consider IV vancomycin if cultures return positive.")
    check("future/conditional IV therapy → untouched",
          rh3["mdm_details"]["risk_score"] == 3 and not v.issues)
    # level/axes contradict → no bump (ERROR path owns it)
    rh4 = {"code": "99214", "mdm_details": {
        "mdm_level": "low", "problems_score": 4,
        "data_score": 3, "risk_score": 3}}
    v.issues = []
    v._check_em_mdm_risk_high_floor([rh4], note_vanc)
    check("claimed level contradicts axes → no bump",
          rh4["mdm_details"]["risk_score"] == 3)

    print("\n[009 full E/M convergence: free-text level + axis flips → 99215]")
    # run 2's exact live shape: axes (4,3,3), derivation sentence as level
    conv = {"code": "99214", "mdm_details": {
        "mdm_level": ("high (problems) / moderate (data, risk) → overall "
                      "MDM moderate by 2-of-3 rule"),
        "problems_score": 4, "data_score": 3, "risk_score": 3}}
    note_009 = (
        "74-year-old female with poorly controlled T2DM (HbA1c 10.8%). "
        "ESR 78, CRP 42, WBC 13.2. ABI 0.72 left. Deep wound debridement "
        "performed. Bone biopsy of distal phalanx cortex obtained. "
        "Infectious disease and vascular surgery consulted. IV vancomycin "
        "initiated empirically per ID recommendation.")
    lines = [conv]
    v.issues = []
    v._check_em_mdm_risk_high_floor(lines, note_009)
    v._check_em_mdm_data_floor(lines, [], note_009)
    v._check_em_mdm_problems_ceiling(lines, [], note_009)
    v._check_em_level_consistency(lines)
    check("minority run converges to the majority's 99215",
          conv["code"] == "99215"
          and conv["mdm_details"]["mdm_level"] == "high")

    print("\n[ICD Includes-chain severity upgrade]")
    note_gangrene = (
        "Left hallux: Necrotic wound margins, purulent discharge. Gangrene "
        "developing distally. Left ABI 0.48 — critical limb ischemia. "
        "Atherosclerosis with critical limb ischemia. Vascular IR performing "
        "left lower extremity angiogram tomorrow.")
    up_icd = [{"code": "M86.9", "type": "primary"},
              {"code": "I70.242", "type": "secondary"}]
    up_cpt = [{"code": "99215", "linked_diagnoses": ["I70.242"]}]
    v.issues = []
    v._check_icd_includes_severity_upgrade(up_icd, up_cpt, [], note_gangrene)
    check("documented gangrene upgrades ulceration code to the "
          "Includes-chain member (I70.242 → I70.262)",
          up_icd[1]["code"] == "I70.262"
          and "icd_includes_severity_upgrade" in cats(v))
    check("service-line dx links follow the upgrade",
          up_cpt[0]["linked_diagnoses"] == ["I70.262"])
    up2 = [{"code": "I70.262", "type": "secondary"}]
    v.issues = []
    v._check_icd_includes_severity_upgrade(up2, [], [], note_gangrene)
    check("already the ranked member → untouched",
          up2[0]["code"] == "I70.262" and not v.issues)
    up3 = [{"code": "I70.242", "type": "secondary"}]
    v.issues = []
    v._check_icd_includes_severity_upgrade(
        up3, [], [],
        "Atherosclerosis of left lower extremity with calf ulceration. "
        "No gangrene. Distal pulses diminished.")
    check("negated gangrene → untouched",
          up3[0]["code"] == "I70.242" and not v.issues)
    up4 = [{"code": "I70.242", "type": "secondary"}]
    v.issues = []
    v._check_icd_includes_severity_upgrade(
        up4, [], [], "Atherosclerosis with calf ulceration, left leg.")
    check("higher condition not documented → untouched",
          up4[0]["code"] == "I70.242" and not v.issues)

    print("\n[CPT excision extent axis (partial vs complete)]")
    note_transect = (
        "OR case. Distal phalangectomy left hallux\n— bone transected at "
        "metaphysis proximal to infected cortex. Wound margins debrided "
        "to bleeding tissue.")
    ex = [{"code": "28150", "modifiers": ["TA"]}]
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_excision_extent_axis(ex, note_transect)
    check("transection wording swaps complete member to partial "
          "(28150 → 28124), across a PDF hard-wrap",
          ex[0]["code"] == "28124" and "excision_extent_axis" in cats(v))
    ex2 = [{"code": "28124", "modifiers": ["TA"]}]
    v.issues = []
    v._check_excision_extent_axis(ex2, note_transect)
    check("already the partial member → untouched",
          ex2[0]["code"] == "28124" and not v.issues)
    ex3 = [{"code": "28150", "modifiers": ["TA"]}]
    v.issues = []
    v._check_excision_extent_axis(
        ex3, "Distal phalangectomy left hallux — entire distal phalanx "
             "excised and disarticulated at the IP joint.")
    check("complete-removal wording → untouched",
          ex3[0]["code"] == "28150" and not v.issues)
    ex4 = [{"code": "28150", "modifiers": ["TA"]}]
    v.issues = []
    v._check_excision_extent_axis(
        ex4, "Distal phalangectomy left hallux performed without complication.")
    check("no extent wording either way → untouched",
          ex4[0]["code"] == "28150" and not v.issues)

    print("\n[same-site PTP bundling: specific digit within same-side generic]")
    ss_cpt = [{"code": "97597", "modifiers": ["RT"], "units": 1},
              {"code": "29550", "modifiers": ["59", "T5"], "units": 1}]
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_same_site_ptp_bundling(ss_cpt)
    check("T5 lies within RT (same side) → column-2 29550 bundled",
          "29550" in v._non_billable_codes_to_suppress
          and "same_site_ptp_bundled" in cats(v))
    ss_cpt2 = [{"code": "97597", "modifiers": ["LT"], "units": 1},
               {"code": "29550", "modifiers": ["59", "T5"], "units": 1}]
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._check_same_site_ptp_bundling(ss_cpt2)
    check("T5 (right) vs LT → distinct sides, both lines stand",
          "29550" not in v._non_billable_codes_to_suppress)

    print("\n[primary designation: coverage-qualifying anchor (payer-gated)]")
    v._payer_follows_medicare = True
    pd_icd = [{"code": "S90.112A", "type": "primary"},
              {"code": "L60.1", "type": "secondary"}]
    pd_cpt = [{"code": "99213", "modifiers": ["25"],
               "linked_diagnoses": ["S90.112A", "L60.1"]},
              {"code": "11730", "modifiers": ["TA"], "units": 1,
               "linked_diagnoses": ["S90.112A"]}]
    v.issues = []
    v._non_billable_codes_to_suppress = set()
    v._bundled_codes_to_suppress = set()
    v._check_primary_designation(pd_icd, pd_cpt)
    check("nail-surgery LCDs qualify only L60.1 → it takes the designation",
          pd_icd[1]["type"] == "primary" and pd_icd[0]["type"] == "secondary"
          and "primary_designation" in cats(v))
    pd_icd2 = [{"code": "L60.1", "type": "primary"},
               {"code": "S90.112A", "type": "secondary"}]
    v.issues = []
    v._check_primary_designation(pd_icd2, pd_cpt)
    check("already the qualifying primary → untouched",
          pd_icd2[0]["type"] == "primary" and not v.issues)
    v._payer_follows_medicare = False
    pd_icd3 = [{"code": "S90.112A", "type": "primary"},
               {"code": "L60.1", "type": "secondary"}]
    v.issues = []
    v._check_primary_designation(pd_icd3, pd_cpt)
    check("commercial payer → LCD lists don't drive the designation",
          pd_icd3[0]["type"] == "primary")
    v._payer_follows_medicare = False

    print("\n[undocumented specific sibling → unspecified member]")
    note_pn = ("Type 2 DM with peripheral neuropathy. Peripheral neuropathy "
               "present. Poorly controlled diabetic on insulin.")
    us_icd = [{"code": "E11.42", "type": "secondary"}]
    us_cpt = [{"code": "97597", "linked_diagnoses": ["E11.42"]}]
    v.issues = []
    v._check_undocumented_specific_sibling(us_icd, us_cpt, [], note_pn)
    check("undocumented 'polyneuropathy' → E11.42 downgraded to E11.40",
          us_icd[0]["code"] == "E11.40"
          and "undocumented_specific_sibling" in cats(v))
    check("service-line links follow the downgrade",
          us_cpt[0]["linked_diagnoses"] == ["E11.40"])
    us2 = [{"code": "E11.42", "type": "secondary"}]
    v.issues = []
    v._check_undocumented_specific_sibling(
        us2, [], [], "Diabetic polyneuropathy confirmed on EMG. Type 2 "
        "diabetes mellitus, poorly controlled.")
    check("documented polyneuropathy → specific code kept",
          us2[0]["code"] == "E11.42" and not v.issues)
    us3 = [{"code": "L97.511", "type": "secondary"}]
    v.issues = []
    v._check_undocumented_specific_sibling(
        us3, [], [], "Pressure ulceration of the hyponychium, wound depth "
        "2mm. Wound debridement performed.")
    check("attribute-level residual (L97 severity axis) → tier rule's turf, "
          "untouched", us3[0]["code"] == "L97.511" and not v.issues)

    print("\n[undocumented specific sibling: Alphabetic Index phrase "
          "evidence]")
    # The Index routes 'brittle nails' to L60.3 (Nail dystrophy) and NOT to
    # L60.9 — a note documenting the Index phrase supports the specific
    # member even though the word 'dystrophy' never appears (guideline
    # I.B.1: the Index is step one of code assignment). Measured live
    # (routine_00003): the expert reviewer's specificity ruling was
    # overridden on every replay because this check could not see
    # Index-phrase evidence.
    ix1 = [{"code": "L60.3", "type": "secondary"}]
    v.issues = []
    v._check_undocumented_specific_sibling(
        ix1, [], [], "Nails brittle and splitting at the free edges "
        "bilaterally. Trimming performed.")
    check("Index phrase ('brittle nails' → L60.3) documented → specific "
          "code kept without the word 'dystrophy'",
          ix1[0]["code"] == "L60.3" and not v.issues)
    # noun/adjective register bridge (the live routine_00003 wording):
    # 'dystrophic' IS documentation of 'dystrophy' — both stem to
    # 'dystroph', the same way removal/removed both stem to 'remov'
    ix1b = [{"code": "L60.3", "type": "secondary"}]
    v.issues = []
    v._check_undocumented_specific_sibling(
        ix1b, [], [], "Periodic debridement of dystrophic, thickened "
        "toenails. 4 dystrophic nails mechanically debrided.")
    check("adjectival form ('dystrophic') documents the descriptor noun "
          "('dystrophy') → L60.3 kept",
          ix1b[0]["code"] == "L60.3" and not v.issues)
    ix2 = [{"code": "L60.3", "type": "secondary"}]
    v.issues = []
    v._check_undocumented_specific_sibling(
        ix2, [], [], "Thickened yellow mycotic nails with subungual "
        "debris. Debridement of nails performed.")
    check("no L60.3-specific Index phrase documented ('thickening nail' "
          "routes to L60.2, not L60.3) → still downgraded to L60.9",
          ix2[0]["code"] == "L60.9"
          and "undocumented_specific_sibling" in cats(v))
    # Phrases the Index routes to BOTH siblings (Z74.1 and Z74.9 share
    # cross-reference aliases like 'smoker') prove neither member and
    # must cancel in the set difference — only member-specific phrases
    # ('need for assistance with personal care' → Z74.1 alone) rescue.
    ix3 = [{"code": "Z74.1", "type": "secondary"}]
    v.issues = []
    v._check_undocumented_specific_sibling(
        ix3, [], [], "Patient is a smoker. Nail care provided.")
    if ix3[0]["code"] == "Z74.9":
        check("sibling-shared Index phrase ('smoker' on both Z74.1 and "
              "Z74.9) cancels → downgrade still fires", True)
    else:
        # if Z74.1's distinguishing axis isn't lexicon-eligible the check
        # never fires at all — shared phrases must still not rescue
        check("sibling-shared Index phrase gives no rescue",
              not any(i.category == "undocumented_specific_sibling"
                      and "Index" in i.message for i in v.issues))
    ix4 = [{"code": "Z74.1", "type": "secondary"}]
    v.issues = []
    v._check_undocumented_specific_sibling(
        ix4, [], [], "Dependence on a care provider is documented: the "
        "patient needs assistance with personal care.")
    check("documented need-for-assistance evidence → Z74.1 kept",
          ix4[0]["code"] == "Z74.1" and not v.issues)

    print("\n[dispensed footwear completion]")
    note_shoe = ("Electrocautery trephine applied over hematoma. "
                 "Hard-soled shoe provided. Return if pain increases.")
    fw_hcpcs = []
    fw_icd = [{"code": "S90.121A", "type": "primary"}]
    v.issues = []
    v._check_dispensed_footwear_completion(
        fw_hcpcs, [], fw_icd, note_shoe, date(2026, 5, 29), "2001-08-17")
    check("documented 'hard-soled shoe provided' → adult surgical boot/shoe "
          "line added, linked to primary",
          len(fw_hcpcs) == 1
          and "surgical boot" in fw_hcpcs[0]["description"].lower()
          and fw_hcpcs[0]["linked_diagnoses"] == ["S90.121A"]
          and "dispensed_supply_completion" in cats(v))
    fw2 = list(fw_hcpcs)
    v.issues = []
    v._check_dispensed_footwear_completion(
        fw2, [], fw_icd, note_shoe, date(2026, 5, 29), "2001-08-17")
    check("footwear already billed → no duplicate", len(fw2) == 1)
    fw3 = []
    v.issues = []
    v._check_dispensed_footwear_completion(
        fw3, [], fw_icd, "Offloading shoe prescribed. Return in 1 week.",
        date(2026, 5, 29), "1948-06-05")
    check("'prescribed' is not dispensing → no add", not fw3)
    fw4 = []
    v.issues = []
    v._check_dispensed_footwear_completion(
        fw4, [], fw_icd, note_shoe, date(2026, 5, 29), "2020-01-01")
    check("pediatric patient → bracket ambiguity, no add", not fw4)

    print("\n[supply digit-modifier strip (A-codes)]")
    sup = [{"code": "A4570", "modifiers": ["T6"], "units": 1}]
    v.issues = []
    v._check_supply_laterality_strip(sup)
    check("digit modifier stripped from plain supply (A4570 'Splint')",
          sup[0]["modifiers"] == []
          and "supply_laterality_removed" in cats(v))

    print("\n[consistency: satisfied instructional groups]")
    from app.validation.consistency import _icd_flip_is_claim_inert
    # Z79.84 flip (live, note 004): the E11.x use-additional group names it,
    # but Z79.4 satisfies the group in every run → the flip is claim-inert.
    z_groups = store.use_additional_code_groups("E11.621")
    z_refs = {r for _c, refs in z_groups for r, _n in refs}
    if any(r.replace(".", "").upper().startswith("Z79") for r in z_refs):
        runs_sat = [
            {"icd_codes": [{"code": "E11.621", "type": "primary"},
                           {"code": "Z79.4", "type": "secondary"},
                           {"code": "Z79.84", "type": "secondary"}],
             "cpt_codes": [], "hcpcs_codes": []},
            {"icd_codes": [{"code": "E11.621", "type": "primary"},
                           {"code": "Z79.4", "type": "secondary"}],
             "cpt_codes": [], "hcpcs_codes": []},
        ]
        check("alternative-ref flip is inert when the group is satisfied "
              "in every run",
              _icd_flip_is_claim_inert("Z79.84", runs_sat, store))
        runs_unsat = [
            {"icd_codes": [{"code": "E11.621", "type": "primary"},
                           {"code": "Z79.84", "type": "secondary"}],
             "cpt_codes": [], "hcpcs_codes": []},
            {"icd_codes": [{"code": "E11.621", "type": "primary"}],
             "cpt_codes": [], "hcpcs_codes": []},
        ]
        check("flip still gates when a run leaves the group unsatisfied",
              not _icd_flip_is_claim_inert("Z79.84", runs_unsat, store))
    else:
        check("SKIP: E11.621 carries no Z79-family use-additional ref", True)

    print("\n[negation backstop]")
    from app.ner.entity_extractor import _NEGATION_CUE
    check("leading cues match",
          all(_NEGATION_CUE.match(t) for t in
              ("no erythema", "Denies fever", "negative for drainage", "without edema")))
    check("mid-span negatives don't match",
          not any(_NEGATION_CUE.match(t) for t in
                  ("not improving with rest", "wound noted", "nocturnal pain")))

    print("\n" + "=" * 50)
    print(f"RESULT: {PASSED} passed, {FAILED} failed")
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
