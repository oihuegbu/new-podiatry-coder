#!/usr/bin/env python3
"""Tests for the denial feedback loop and self-consistency layers.

Run: PYTHONPATH=. python tests/test_feedback_consistency.py
"""

import json
import sys
from pathlib import Path

if __name__ != "__main__":
    import pytest
    pytest.skip("script harness; run with python tests/test_feedback_consistency.py",
                allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.validation.consistency import (
    compare_runs, select_canonical, annotate_result)
from tools.denial_feedback import (
    parse_835, parse_csv, classify_denial, _load_carc_map, _merge,
    _family_matches)

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def _run(icd=None, cpt=None, disposition="CLEAN", tier="AUTO"):
    return {
        "icd_codes": icd or [],
        "supporting_conditions": [],
        "cpt_codes": cpt or [],
        "hcpcs_codes": [],
        "snomed_codes": [],
        "final_disposition": disposition,
        "auto_coding_tier": tier,
    }


print("\n[self-consistency: unanimous runs]")
a = _run(icd=[{"code": "M20.11", "type": "primary"}],
         cpt=[{"code": "99213", "modifiers": ["25"], "units": 1}])
b = _run(icd=[{"code": "M20.11", "type": "primary"}],
         cpt=[{"code": "99213", "modifiers": ["25"], "units": 1}])
rep = compare_runs([a, b])
check("identical runs → unanimous", rep["unanimous"])
check("no disagreements listed", not rep["disagreements"])
out = annotate_result(dict(a), rep)
check("unanimous → tier untouched", out["auto_coding_tier"] == "AUTO")

print("\n[self-consistency: presence disagreement]")
c = _run(icd=[{"code": "M20.11", "type": "primary"},
              {"code": "E11.9", "type": "secondary"}],
         cpt=[{"code": "99213", "modifiers": ["25"], "units": 1}])
rep = compare_runs([a, c])
check("code present in 1/2 runs → not unanimous", not rep["unanimous"])
pres = [d for d in rep["disagreements"] if d["kind"] == "presence"]
check("E11.9 reported as presence disagreement",
      any(d["code"] == "E11.9" for d in pres))
ann = annotate_result(json.loads(json.dumps(c)), rep)
check("disagreement routes note to REVIEW", ann["auto_coding_tier"] == "REVIEW")
check("disposition upgraded from CLEAN", ann["final_disposition"] == "REVIEW")
flagged = [e for e in ann["icd_codes"] if e["code"] == "E11.9"]
check("disagreeing entry marked needs_review",
      flagged and flagged[0].get("needs_review") is True
      and "self-consistency" in flagged[0].get("review_reason", ""))
stable = [e for e in ann["icd_codes"] if e["code"] == "M20.11"]
check("unanimous entry untouched", not stable[0].get("needs_review"))

print("\n[self-consistency: attribute disagreement]")
d1 = _run(cpt=[{"code": "11720", "modifiers": ["59"], "units": 1}])
d2 = _run(cpt=[{"code": "11720", "modifiers": [], "units": 1}])
rep = compare_runs([d1, d2])
attr = [x for x in rep["disagreements"] if x["kind"] == "attributes"]
check("modifier flip across runs detected",
      attr and attr[0]["code"] == "11720" and "modifiers" in attr[0]["fields"])
d3 = _run(cpt=[{"code": "11720", "modifiers": ["59"], "units": 2}])
rep = compare_runs([d1, d3])
attr = [x for x in rep["disagreements"] if x["kind"] == "attributes"]
check("unit flip detected", attr and "units" in attr[0]["fields"])

print("\n[self-consistency: SNOMED variance is advisory-only]")
s1 = _run(cpt=[{"code": "99213", "modifiers": [], "units": 1}])
s1["snomed_codes"] = [{"concept_id": "22253000"}, {"concept_id": "3092008"}]
s2 = _run(cpt=[{"code": "99213", "modifiers": [], "units": 1}])
s2["snomed_codes"] = [{"concept_id": "22253000"}]
rep = compare_runs([s1, s2])
check("SNOMED-only variance stays unanimous (no REVIEW routing)",
      rep["unanimous"])
check("but the variance is still recorded in the report",
      any(d["array"] == "snomed_codes" and d.get("advisory")
          for d in rep["disagreements"]))
ann = annotate_result(json.loads(json.dumps(s1)), rep)
check("tier untouched by advisory-only variance", ann["auto_coding_tier"] == "AUTO")

print("\n[self-consistency: supporting_conditions variance is advisory-only]")
c1 = _run(cpt=[{"code": "99213", "modifiers": [], "units": 1}])
c1["supporting_conditions"] = [{"code": "Z79.4", "type": "supporting"},
                               {"code": "M81.0", "type": "supporting"}]
c2 = _run(cpt=[{"code": "99213", "modifiers": [], "units": 1}])
c2["supporting_conditions"] = [{"code": "Z79.4", "type": "supporting"}]
rep = compare_runs([c1, c2])
check("supporting_conditions-only variance stays unanimous (never on a CMS-1500)",
      rep["unanimous"])
check("variance still recorded as advisory",
      any(d["array"] == "supporting_conditions" and d.get("advisory")
          for d in rep["disagreements"]))
ann = annotate_result(json.loads(json.dumps(c1)), rep)
check("tier untouched by supporting-only variance", ann["auto_coding_tier"] == "AUTO")

print("\n[self-consistency: optional external-cause ICD variance is advisory-only]")


class _ChapterStore:
    @staticmethod
    def is_external_cause(code):
        return code == "W22.8XXA"


x1 = _run(icd=[{"code": "S90.122A", "type": "primary"},
               {"code": "W22.8XXA", "type": "secondary"}],
          cpt=[{"code": "99213", "modifiers": [], "units": 1}])
x2 = _run(icd=[{"code": "S90.122A", "type": "primary"}],
          cpt=[{"code": "99213", "modifiers": [], "units": 1}])
rep = compare_runs([x1, x2], store=_ChapterStore())
check("external-cause (Chapter 20) presence flip stays unanimous "
      "(ICD-10-CM guidelines: reporting is optional)", rep["unanimous"])
check("flip still recorded as advisory",
      any(d["code"] == "W22.8XXA" and d.get("advisory")
          for d in rep["disagreements"]))
x3 = _run(icd=[{"code": "S90.122A", "type": "primary"}],
          cpt=[{"code": "99213", "modifiers": [], "units": 1}])
x4 = _run(icd=[], cpt=[{"code": "99213", "modifiers": [], "units": 1}])
rep = compare_runs([x3, x4])
check("injury-code (Chapter 19) flip still gates — not advisory",
      not rep["unanimous"])

print("\n[self-consistency: claim-inert secondary ICD flips are advisory (store-aware)]")


class _FakeStore:
    """Duck-typed stand-in for ComplianceDataStore: one E/M family
    (99213/14/15), one useAdditionalCode rule (E11 → Z79.4), one coverage
    policy for CPT 11055 whose dx list holds E11.40."""

    _EM_PREFIX = ("Office or other outpatient visit for the evaluation and "
                  "management of an established patient")

    def em_family_prefix(self, code):
        return self._EM_PREFIX if code in ("99213", "99214", "99215") else None

    def use_additional_code_groups(self, code):
        if code.upper().replace(".", "").startswith("E11"):
            return [("E11", [("Z79.4", "Use additional code to identify insulin use (Z79.4)")])]
        return []

    def code_first_groups(self, code):
        return []

    def code_also_groups(self, code):
        return []

    def coverage_policies_for_cpt(self, cpt):
        return ["L99999"] if cpt == "11055" else []

    def coverage_policy_has_dx_rules(self, policy_id):
        return True

    def coverage_icd_covered(self, policy_id, icd):
        return icd.upper().replace(".", "") in ("E1140", "B351")


fs = _FakeStore()
base = _run(icd=[{"code": "M20.11", "type": "primary"}],
            cpt=[{"code": "99213", "modifiers": [], "units": 1,
                  "linked_diagnoses": ["M20.11"]}])
extra = _run(icd=[{"code": "M20.11", "type": "primary"},
                  {"code": "I73.9", "type": "secondary"}],
             cpt=[{"code": "99213", "modifiers": [], "units": 1,
                   "linked_diagnoses": ["M20.11"]}])
rep = compare_runs([base, extra], store=fs)
check("unpointed, unrequired secondary flip → advisory, stays unanimous",
      rep["unanimous"]
      and any(d["code"] == "I73.9" and d.get("advisory")
              for d in rep["disagreements"]))
# same flip WITHOUT a store → conservative, gates as before
rep = compare_runs([base, extra])
check("no store → same flip stays billing-gating (conservative fallback)",
      not rep["unanimous"])
# pointed at a service line → billing-gating
pointed = json.loads(json.dumps(extra))
pointed["cpt_codes"][0]["linked_diagnoses"] = ["M20.11", "I73.9"]
rep = compare_runs([base, pointed], store=fs)
check("same flip pointed by a service line → gates", not rep["unanimous"])
# named by another dx's useAdditionalCode note → billing-gating
ua_base = _run(icd=[{"code": "E11.9", "type": "primary"}],
               cpt=[{"code": "99213", "modifiers": [], "units": 1}])
ua_extra = _run(icd=[{"code": "E11.9", "type": "primary"},
                     {"code": "Z79.4", "type": "secondary"}],
                cpt=[{"code": "99213", "modifiers": [], "units": 1}])
rep = compare_runs([ua_base, ua_extra], store=fs)
check("flip on a Tabular-mandated companion (E11 → Z79.4) → gates",
      not rep["unanimous"])
# on a coverage policy's dx list for a claim CPT → billing-gating
pol_base = _run(icd=[{"code": "M20.11", "type": "primary"}],
                cpt=[{"code": "11055", "modifiers": [], "units": 1}])
pol_extra = _run(icd=[{"code": "M20.11", "type": "primary"},
                      {"code": "E11.40", "type": "secondary"}],
                 cpt=[{"code": "11055", "modifiers": [], "units": 1}])
rep = compare_runs([pol_base, pol_extra], store=fs)
check("flip on a policy-listed dx for a claim CPT → gates", not rep["unanimous"])
# ...but when EVERY run already satisfies that policy with another covered
# dx, the flipping code is a redundant coverage alternative — coverage needs
# one covered diagnosis, not all of them (measured live, note 003: L60.2
# flipped 1/3 on a claim B35.1 covers in every run) → advisory
sat_base = _run(icd=[{"code": "B35.1", "type": "primary"}],
                cpt=[{"code": "11055", "modifiers": [], "units": 1,
                      "linked_diagnoses": ["B35.1"]}])
sat_extra = _run(icd=[{"code": "B35.1", "type": "primary"},
                      {"code": "E11.40", "type": "secondary"}],
                 cpt=[{"code": "11055", "modifiers": [], "units": 1,
                       "linked_diagnoses": ["B35.1"]}])
rep = compare_runs([sat_base, sat_extra], store=fs)
check("policy-listed flip with the policy satisfied in every run → advisory",
      rep["unanimous"]
      and any(d["code"] == "E11.40" and d.get("advisory")
              for d in rep["disagreements"]))
# flip code is PRIMARY in the run where present → billing-gating
pri_base = _run(icd=[{"code": "M20.11", "type": "primary"}],
                cpt=[{"code": "99213", "modifiers": [], "units": 1}])
pri_extra = _run(icd=[{"code": "M20.11", "type": "secondary"},
                      {"code": "I73.9", "type": "primary"}],
                 cpt=[{"code": "99213", "modifiers": [], "units": 1}])
rep = compare_runs([pri_base, pri_extra], store=fs)
check("flip on a PRIMARY dx → gates", not rep["unanimous"])

print("\n[self-consistency: E/M sibling flips merge into one em_level disagreement]")
em_a = _run(icd=[{"code": "M20.11", "type": "primary"}],
            cpt=[{"code": "99213", "modifiers": [], "units": 1,
                  "mdm_details": {"mdm_level": "low", "problems_score": 2,
                                  "data_score": 2, "risk_score": 2}}])
em_b = _run(icd=[{"code": "M20.11", "type": "primary"}],
            cpt=[{"code": "99214", "modifiers": [], "units": 1,
                  "mdm_details": {"mdm_level": "moderate", "problems_score": 3,
                                  "data_score": 2, "risk_score": 3}}])
rep = compare_runs([em_a, em_b], store=fs)
em_dis = [d for d in rep["disagreements"] if d["kind"] == "em_level"]
pres = [d for d in rep["disagreements"]
        if d["kind"] == "presence" and d["code"] in ("99213", "99214")]
check("one merged em_level disagreement, no separate presence rows",
      len(em_dis) == 1 and not pres)
check("merged row still gates unanimity", not rep["unanimous"])
check("per-run MDM axes captured for the next layer's design",
      em_dis and [r["code"] for r in em_dis[0]["by_run"]] == ["99213", "99214"]
      and em_dis[0]["by_run"][0]["risk_score"] == 2
      and em_dis[0]["by_run"][1]["risk_score"] == 3)
ann = annotate_result(json.loads(json.dumps(em_b)), rep)
em_entry = ann["cpt_codes"][0]
check("canonical E/M entry flagged with the level-flip reason",
      em_entry.get("needs_review") is True
      and "E/M level flipped" in em_entry.get("review_reason", ""))
# messy case: one run carries BOTH siblings → no merge, plain presence rows
messy = _run(icd=[{"code": "M20.11", "type": "primary"}],
             cpt=[{"code": "99213", "modifiers": [], "units": 1},
                  {"code": "99214", "modifiers": [], "units": 1}])
rep = compare_runs([em_a, em_b, messy], store=fs)
check("a run carrying both siblings → no merge, stays as plain presence flips",
      not any(d["kind"] == "em_level" for d in rep["disagreements"])
      and sum(1 for d in rep["disagreements"]
              if d["kind"] == "presence" and d["code"] in ("99213", "99214")) == 2)

print("\n[self-consistency: disposition-only disagreement]")
e1 = _run(cpt=[{"code": "99213", "modifiers": [], "units": 1}], disposition="CLEAN")
e2 = _run(cpt=[{"code": "99213", "modifiers": [], "units": 1}], disposition="REVIEW")
rep = compare_runs([e1, e2])
check("same codes, differing disposition → not unanimous", not rep["unanimous"])
ann = annotate_result(dict(e1), rep)
check("disposition disagreement noted in review reasons",
      any("disposition varied" in r for r in ann["auto_coding_review_reasons"]))

print("\n[self-consistency: canonical selection]")
maj = _run(icd=[{"code": "M20.11", "type": "primary"}])
odd = _run(icd=[{"code": "M20.11", "type": "primary"},
                {"code": "E11.9", "type": "secondary"}])
idx = select_canonical([maj, odd, json.loads(json.dumps(maj))])
check("run agreeing with the 2/3 majority chosen as canonical", idx in (0, 2))
# SNOMED bulk must not outvote billing content: r2 carries 5 majority SNOMEDs
# but MISSES the majority billing code — it must never be canonical.
snomeds = [{"concept_id": str(1000 + i)} for i in range(5)]
r0 = _run(cpt=[{"code": "99213", "modifiers": [], "units": 1}])
r1 = _run(cpt=[{"code": "99213", "modifiers": [], "units": 1}])
r1["snomed_codes"] = list(snomeds)
r2 = _run(cpt=[])
r2["snomed_codes"] = list(snomeds)
check("informational SNOMED agreement cannot outvote billing majority",
      select_canonical([r0, r1, r2]) in (0, 1))

print("\n[denial feedback: CSV ingest]")
csv_text = (
    "document_id,code,carc,rarc,payer,dos,amount\n"
    "042_grace_tillman_note2,11721,50,N115,Medicare FL,2026-05-01,84.20\n"
    "042_grace_tillman_note2,99213,97,,Medicare FL,2026-05-01,42.00\n")
recs = parse_csv(csv_text)
check("two CSV denials parsed", len(recs) == 2)
check("fields normalized", recs[0]["carc"] == "50" and recs[0]["code"] == "11721")
merged, added = _merge([], recs)
merged, added2 = _merge(merged, recs)
check("re-ingest is idempotent", added == 2 and added2 == 0)

print("\n[denial feedback: 835 ingest]")
edi = ("ISA*00*          *00*          *ZZ*PAYER          *ZZ*PRACTICE       "
       "*260701*1200*^*00501*000000001*0*P*:~"
       "N1*PR*MEDICARE FLORIDA~"
       "CLP*042_grace_tillman_note2*1*126.20*42.00**MC*ICN123~"
       "SVC*HC:11721:Q8*84.20*0~"
       "CAS*CO*50*84.20~"
       "SVC*HC:99213:25*42.00*42.00~"
       "CAS*PR*3*10.00~")
recs = parse_835(edi)
check("one payer-side (CO) denial extracted, PR ignored",
      len(recs) == 1 and recs[0]["carc"] == "50")
check("service code + claim id carried through",
      recs[0]["code"] == "11721"
      and recs[0]["document_id"] == "042_grace_tillman_note2")
check("payer name captured", "MEDICARE" in recs[0]["payer"].upper())

print("\n[denial feedback: classification against a stored result]")
carc_map = _load_carc_map()
result = {
    "claim_scrub": {"findings": [
        {"filter_id": "MEDICAL_NECESSITY", "status": "FAIL",
         "codes": ["11721"], "reason": "no covered dx on line"}]},
    "validation_issues": [],
    "icd_codes": [], "cpt_codes": [{"code": "99213"}], "hcpcs_codes": [],
}
den = {"document_id": "x", "code": "11721", "carc": "50"}
check("denial matching a fired check family → CAUGHT",
      classify_denial(den, result, carc_map)["status"] == "CAUGHT")
den = {"document_id": "x", "code": "99213", "carc": "97"}
check("denied code the pipeline passed clean → MISSED",
      classify_denial(den, result, carc_map)["status"] == "MISSED")
den = {"document_id": "x", "code": "11721", "carc": "97"}
c = classify_denial(den, result, carc_map)
check("flagged under unrelated family → FLAGGED_OTHER",
      c["status"] == "FLAGGED_OTHER")
den = {"document_id": "x", "code": "11721", "carc": "999"}
check("unknown CARC → UNMAPPED",
      classify_denial(den, result, carc_map)["status"] == "UNMAPPED")
check("missing result → NO_RESULT",
      classify_denial(den, None, carc_map)["status"] == "NO_RESULT")

print("\n[denial feedback: family matching is token-based, not substring]")
check("'age' matches hcpcs_age_range", _family_matches("age", "hcpcs_age_range"))
check("'age' does NOT match cpt_dx_linkage (substring trap)",
      not _family_matches("age", "cpt_dx_linkage"))
check("'modifier' matches MODIFIERS filter id (plural leveling)",
      _family_matches("modifier", "modifiers"))
check("'ncci' matches ncci_edit", _family_matches("ncci", "ncci_edit"))
check("'mue' matches mue_limit", _family_matches("mue", "mue_limit"))
check("multiword family matches its category",
      _family_matches("dx_pointer", "dx_pointer_overflow"))
check("'pos' does not match dx_pointer", not _family_matches("pos", "dx_pointer"))
# The exact false-CAUGHT this fix prevents: an age denial (CARC 6) on a code
# whose only flag is a dx-linkage finding must NOT classify as CAUGHT.
result_linkage = {
    "claim_scrub": {"findings": [
        {"filter_id": "CPT_DX_LINKAGE", "status": "WARN",
         "codes": ["99213"], "reason": "line repointed"}]},
    "validation_issues": [], "icd_codes": [], "cpt_codes": [], "hcpcs_codes": [],
}
c = classify_denial({"document_id": "x", "code": "99213", "carc": "6"},
                    result_linkage, _load_carc_map())
check("age denial vs dx-linkage flag → FLAGGED_OTHER, not CAUGHT",
      c["status"] == "FLAGGED_OTHER")

print("\n[denial feedback: claim-level denial (no service-line code)]")
result_claim = {
    "claim_scrub": {"findings": [
        {"filter_id": "PRIOR_AUTH", "status": "FAIL",
         "codes": ["29445"], "reason": "auth required, none on file"}]},
    "validation_issues": [], "icd_codes": [], "cpt_codes": [], "hcpcs_codes": [],
}
c = classify_denial({"document_id": "x", "code": "", "carc": "197"},
                    result_claim, _load_carc_map())
check("claim-level CARC 197 matches any prior-auth flag on the claim → CAUGHT",
      c["status"] == "CAUGHT")

print("\n[denial feedback: CARC map integrity]")
known_families = {
    "modifier", "pos_eligibility", "pos", "age", "sex", "mce", "dx",
    "cpt_dx_linkage", "medical_necessity", "dx_pointer", "documentation",
    "specificity", "duplicate", "benefits", "coverage", "billability",
    "ncci", "global_period", "addon", "frequency", "mue", "icd",
    "code_validity", "prior_auth"}
bad = {f for e in carc_map.values() for f in e["check_families"]} - known_families
check("every mapped family is a recognized check token", not bad)
check("map covers the core denial classes (bundling, necessity, units, auth)",
      all(k in carc_map for k in ("97", "50", "151", "197")))

print(f"\n{'=' * 50}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
