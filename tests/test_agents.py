"""Adversarial agent tests — craft claims with KNOWN violations and assert each
agent fires correctly (and stays silent on clean input). Run:  python -m tests.test_agents
"""
from __future__ import annotations

# This is intentionally a script-style adversarial harness.  During pytest
# collection its top-level execution used to terminate the entire suite via
# SystemExit; keep the executable entry point while making collection explicit.
if __name__ != "__main__":
    import pytest
    pytest.skip("script harness; run with python tests/test_agents.py",
                allow_module_level=True)

from app.compliance.datastore.store import ComplianceDataStore
from app.compliance.engine import ClaimScrubber, build_claim
from app.compliance.agents.specificity import SpecificityAgent
from app.compliance.agents.mue_mai import MUEAgent
from app.compliance.agents.global_period import GlobalPeriodAgent
from app.compliance.agents.ncci_ptp import NCCIPTPAgent
from app.compliance.agents.modifiers import ModifierAgent
from app.compliance.agents.frequency import FrequencyAgent
from app.compliance.agents.addon import AddOnAgent
from app.compliance.agents.pos_eligibility import POSEligibilityAgent
from app.compliance.agents.medical_necessity import MedicalNecessityAgent
from app.compliance.models import DenialRisk, Status

_STORE = ComplianceDataStore()
_STORE.build_or_load()

PASS, FAIL = 0, 0


def check(name: str, cond: bool):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}")


def run(agent, result: dict):
    return agent.check(build_claim(result))


def base(cpt=None, icd=None, dos="2026-07-20", prior=None, insurance=None, dob=None):
    meta = {"date_of_service": dos}
    if insurance:
        meta["insurance"] = insurance
    if dob:
        meta["date_of_birth"] = dob
    return {
        "document_id": "TEST",
        "patient_metadata": meta,
        "rag_context": {"prior_surgery_info": prior or {}},
        "icd_codes": icd or [],
        "cpt_codes": cpt or [],
        "hcpcs_codes": [],
    }


# --------------------------------------------------------------------- #
print("\n[Specificity #1]")
a = SpecificityAgent(_STORE)
f = run(a, base(icd=[{"code": "E11.4", "type": "primary"}]))   # header, has children
check("unspecified E11.4 → FAIL", any(x.status == Status.FAIL and "specific" in x.reason.lower() for x in f))

f = run(a, base(icd=[{"code": "E11.40", "type": "primary"}]))  # billable leaf
check("billable leaf E11.40 → no specificity FAIL", not any("specific" in x.reason.lower() for x in f))

f = run(a, base(icd=[{"code": "Z99.999", "type": "primary"}]))  # nonexistent
check("nonexistent Z99.999 → FAIL existence", any(x.status == Status.FAIL and "does not exist" in x.reason for x in f))

f = run(a, base(cpt=[{"code": "99213"}], icd=[{"code": "M20.11", "type": "primary"}]))  # all valid leaves
check("valid claim → no findings", len(f) == 0)

# --------------------------------------------------------------------- #
print("\n[MUE/MAI #3]")
a = MUEAgent(_STORE)
_mue_bounds = _STORE._mue_release_bounds()
assert _mue_bounds is not None
_mue_dos = _mue_bounds[0].isoformat()
f = run(a, base(cpt=[{"code": "0001U", "units": 5}], dos=_mue_dos))  # MAI 2, cap 1
check("MAI-2 over cap → FAIL (hard wall)", any(x.status == Status.FAIL and "absolute" in x.reason.lower() for x in f))

f = run(a, base(cpt=[{"code": "0001U", "units": 1}], dos=_mue_dos))  # within cap
check("MAI-2 within cap → no finding", len(f) == 0)

# find an MAI-1 code with a known cap for the split-line test
row = _STORE.conn.execute("SELECT code, mue_value FROM mue WHERE mai='1' AND mue_value=1 LIMIT 1").fetchone()
if row:
    code, cap = row["code"], row["mue_value"]
    f = run(a, base(cpt=[{"code": code, "units": cap + 2}], dos=_mue_dos))
    check(f"MAI-1 over cap no modifier ({code}) → FAIL", any(x.status == Status.FAIL for x in f))
    f = run(a, base(cpt=[{"code": code, "units": cap + 2, "modifiers": ["59"]}], dos=_mue_dos))
    check(f"MAI-1 over cap WITH 59 ({code}) → WARN not FAIL", any(x.status == Status.WARN for x in f) and not any(x.status == Status.FAIL for x in f))

# MUE of 0: CMS pays zero units on this claim type, at ANY quantity — the
# old `cap <= 0: continue` skipped these, letting the scrub verdict come
# back CLEAN and override the validator's ERROR (observed live: A4570).
row = _STORE.conn.execute("SELECT code FROM mue WHERE mue_value=0 LIMIT 1").fetchone()
if row:
    code0 = row["code"]
    f = run(a, base(cpt=[{"code": code0, "units": 1}], dos=_mue_dos))
    check(f"MUE-0 code ({code0}) x1 → FAIL, never payable",
          any(x.status == Status.FAIL and "zero" in x.reason.lower() for x in f))
else:
    check("SKIP: no MUE-0 code in dataset", True)

# --------------------------------------------------------------------- #
print("\n[Global Period #6]")
a = GlobalPeriodAgent(_STORE)
prior = {"is_post_op_visit": True, "days_post_op": 14,
         "prior_surgery_cpt": "28285", "prior_surgery_description": "hammertoe correction"}
gd = _STORE.global_period("28285")
print(f"    (28285 global days in store = {gd})")
f = run(a, base(cpt=[{"code": "99213"}], prior=prior))
check("E/M in global window, no modifier → FAIL", any(x.status == Status.FAIL for x in f))

f = run(a, base(cpt=[{"code": "99213", "modifiers": ["24"]}], prior=prior))
check("E/M in global window WITH mod 24 → WARN not FAIL", any(x.status == Status.WARN for x in f) and not any(x.status == Status.FAIL for x in f))

f = run(a, base(cpt=[{"code": "99024"}], prior=prior))
check("99024 (no-charge) in window → no finding", len(f) == 0)

f = run(a, base(cpt=[{"code": "99213"}]))  # not a post-op visit
check("E/M with no prior surgery → no finding", len(f) == 0)

# PROCEDURES in the postop window are part of the surgical package too —
# 58/78/79 (not 24) are the qualifying bypass modifiers for procedure lines.
f = run(a, base(cpt=[{"code": "11750"}], prior=prior))
check("procedure in global window, no 58/78/79 → FAIL",
      any(x.status == Status.FAIL and "Procedure" in x.reason for x in f))
f = run(a, base(cpt=[{"code": "11750", "modifiers": ["79"]}], prior=prior))
check("procedure in global window WITH 79 → WARN not FAIL",
      any(x.status == Status.WARN for x in f) and not any(x.status == Status.FAIL for x in f))
f = run(a, base(cpt=[{"code": "73630"}], prior=prior))  # XXX-global diagnostic
check("diagnostic (XXX global) in window → no finding", len(f) == 0)

# --------------------------------------------------------------------- #
print("\n[NCCI PTP #2]")
a = NCCIPTPAgent(_STORE)
# Select test pairs from the live NCCI table rather than hardcoding them —
# fixture pairs rot with every quarterly CMS refresh (the previous hardcoded
# pairs, e.g. 15271/C5271, no longer exist in current data, and a pair once
# assumed "non-conflicting" had since become a real edit).
_ncci_release = _STORE.conn.execute(
    "SELECT MAX(effective_from) AS release_start FROM ncci_ptp"
).fetchone()
_NCCI_DOS = _ncci_release["release_start"]


def _live_pair(indicator: str):
    return _STORE.conn.execute(
        "SELECT col1, col2 FROM ncci_ptp WHERE modifier_indicator=? "
        "AND effective_from<=? AND effective_to>=? "
        "AND col1 NOT GLOB '099*' AND col2 NOT GLOB '099*' LIMIT 1",
        (indicator, _NCCI_DOS, _NCCI_DOS),
    ).fetchone()


row0, row1 = _live_pair("0"), _live_pair("1")
if row0:
    f = run(a, base(cpt=[{"code": row0["col1"]}, {"code": row0["col2"]}], dos=_NCCI_DOS))
    check(f"indicator-0 pair ({row0['col1']}/{row0['col2']}) → FAIL (hard edit)",
          any(x.status == Status.FAIL for x in f))
if row1:
    f = run(a, base(cpt=[{"code": row1["col1"]}, {"code": row1["col2"]}], dos=_NCCI_DOS))
    check(f"indicator-1 pair ({row1['col1']}/{row1['col2']}) no modifier → FAIL",
          any(x.status == Status.FAIL for x in f))
    f = run(a, base(cpt=[{"code": row1["col1"]}, {"code": row1["col2"], "modifiers": ["59"]}], dos=_NCCI_DOS))
    check(f"indicator-1 pair ({row1['col1']}/{row1['col2']}) WITH 59 → WARN not FAIL",
          any(x.status == Status.WARN for x in f) and not any(x.status == Status.FAIL for x in f))
# Anatomic NCCI-associated modifiers: DIFFERENT anatomic modifiers on the two
# lines document distinct sites and bypass an indicator-1 edit like 59/X{EPSU}
# (live regression: 28297-RT vs 28285-RT,T6 — right bunion vs right 2nd toe —
# was FAILed as 'no separation modifier'). Same-side-on-both separates nothing.
# Guards data-driven: the anatomic set must come from the modifier reference
# data, and the test pair must still be a live indicator-1 edit.
_anat = _STORE.anatomic_modifiers()
check("anatomic modifier set derived from reference data (T6/RT/FA in, 59/50 out)",
      {"T6", "RT", "FA", "LT", "TA"} <= _anat and not ({"59", "50", "25"} & _anat))
_edit_28297 = _STORE.ncci_pair("28297", "28285", _NCCI_DOS)
if _edit_28297 and _edit_28297.get("modifier_indicator") == "1":
    f = run(a, base(cpt=[{"code": "28297", "modifiers": ["RT"]},
                         {"code": "28285", "modifiers": ["RT", "T6"]}], dos=_NCCI_DOS))
    check("indicator-1 pair, differing anatomic modifiers (RT vs RT+T6) → WARN not FAIL",
          any(x.status == Status.WARN for x in f) and not any(x.status == Status.FAIL for x in f))
    f = run(a, base(cpt=[{"code": "28297", "modifiers": ["RT"]},
                         {"code": "28285", "modifiers": ["RT"]}], dos=_NCCI_DOS))
    check("indicator-1 pair, SAME anatomic modifier both lines (RT/RT) → still FAIL",
          any(x.status == Status.FAIL for x in f))
    f = run(a, base(cpt=[{"code": "28297", "modifiers": ["RT"]},
                         {"code": "28285", "modifiers": []}], dos=_NCCI_DOS))
    check("indicator-1 pair, anatomic modifier on ONE line only → still FAIL",
          any(x.status == Status.FAIL for x in f))
else:
    print("    (skipped anatomic-separation cases — 28297/28285 no longer an indicator-1 edit)")
# A pair verified absent from the table in BOTH directions on this DOS
_clean = ("99024", "97597")
_has_edit = any(
    _STORE.ncci_pair(x, y, _NCCI_DOS) for x, y in (_clean, _clean[::-1])
)
if not _has_edit:
    f = run(a, base(cpt=[{"code": _clean[0]}, {"code": _clean[1]}], dos=_NCCI_DOS))
    check("non-conflicting pair → no NCCI finding", len(f) == 0)
else:
    print("    (skipped non-conflicting-pair case — chosen pair now has a live edit)")

print("\n[Modifiers #4]")
a = ModifierAgent(_STORE)
f = run(a, base(cpt=[{"code": "11721", "modifiers": ["ZZ"]}]))
check("unrecognized modifier ZZ → FAIL", any(x.status == Status.FAIL for x in f))
f = run(a, base(cpt=[{"code": "11721", "modifiers": ["RT", "LT"]}]))
check("RT+LT same line → WARN", any(x.status == Status.WARN for x in f))
f = run(a, base(cpt=[{"code": "11721", "modifiers": ["59"]}]))
check("general separation modifier → advisory WARN (prefer specific role)",
      any(x.status == Status.WARN and "specific" in x.recommendation.lower()
          for x in f))
f = run(a, base(cpt=[{"code": "11721", "modifiers": ["79"]}]))
check("postop modifier 79 with no post-op context → WARN",
      any("postoperative global period" in x.reason for x in f))
f = run(a, base(cpt=[{"code": "11721", "modifiers": ["79"]}],
                prior={"is_post_op_visit": True, "prior_surgery_cpt": "28296", "days_post_op": 14}))
check("postop modifier 79 inside real post-op window → silent (global-period agent owns it)",
      not any("no prior surgery" in x.reason for x in f))
f = run(a, base(cpt=[{"code": "11721", "modifiers": ["RT"]}]))
check("valid single modifier RT → no finding", len(f) == 0)

# E/M + same-day procedure: modifier-25 bundling
f = run(a, base(cpt=[{"code": "99213"}, {"code": "11721"}]))  # E/M + surgery, no 25
check("E/M + procedure, no separation modifier → FAIL",
      any(x.status == Status.FAIL and "same-day procedure bundling" in x.source_rule
          for x in f))
f = run(a, base(cpt=[{"code": "99213", "modifiers": ["25"]}, {"code": "11721"}]))  # has 25
check("E/M + procedure WITH mod 25 → no 25-bundling finding", not any("same day" in x.reason for x in f))
f = run(a, base(cpt=[{"code": "99213"}, {"code": "73630"}]))  # E/M + radiology (not surgery)
check("E/M + radiology (not surgery) → no 25 finding", not any("same day" in x.reason for x in f))

print("\n[Frequency / duplicate #7]")
a = FrequencyAgent(_STORE)
f = run(a, base(cpt=[{"code": "11721"}, {"code": "11721"}]))  # identical dup
check("duplicate identical lines → FAIL", any(x.status == Status.FAIL for x in f))
f = run(a, base(cpt=[{"code": "11721", "modifiers": ["RT"]}, {"code": "11721", "modifiers": ["LT"]}]))
check("same code RT + LT → no finding (distinct sites)", len(f) == 0)

print("\n[Add-on #8]")
a = AddOnAgent(_STORE)
f = run(a, base(cpt=[{"code": "77002"}]))  # add-on alone
check("add-on 77002 billed alone → FAIL", any(x.status == Status.FAIL for x in f))
f = run(a, base(cpt=[{"code": "64455"}, {"code": "77002"}]))  # add-on with primary
check("add-on 77002 WITH primary 64455 → no finding", len(f) == 0)

print("\n[POS & eligibility #9]")
a = POSEligibilityAgent(_STORE)
f = run(a, base(cpt=[{"code": "11721", "place_of_service": "88"}]))  # invalid POS
check("invalid POS 88 → FAIL", any(x.status == Status.FAIL for x in f))
f = run(a, base(cpt=[{"code": "11721", "place_of_service": "11"}]))  # office
check("valid POS 11 → no finding", len(f) == 0)

print("\n[Medical Necessity #5]")
a = MedicalNecessityAgent(_STORE)
# 11055 = routine foot care, governed by every MAC's routine-foot-care policy;
# non-qualifying dx (I10 hypertension) → FAIL. Guard data-driven (policy IDs
# rot as MACs retire/renumber LCDs): confirm I10 is truly non-qualifying
# under every restrictive policy governing 11055.
_p11055 = [p for p in _STORE.coverage_policies_for_cpt("11055")
           if _STORE.coverage_policy_has_dx_rules(p)]
assert _p11055 and not any(_STORE.coverage_icd_covered(p, "I10") for p in _p11055)
f = run(a, base(cpt=[{"code": "11055", "linked_diagnoses": ["I10"]}],
                icd=[{"code": "I10", "type": "primary"}], insurance="Medicare Part B"))
check("Medicare + routine foot care + non-qualifying dx (I10) → FAIL", any(x.status == Status.FAIL for x in f))
# Unknown claim state + every gating policy scoped to a known MAC area:
# still FAILs (routes to review), but the reason must say coverage is
# UNVERIFIABLE without the state — not assert "will deny" as a fact.
check("unknown state + all-scoped policies → FAIL reads 'cannot be verified', not 'will deny'",
      any(x.status == Status.FAIL and "could not be determined" in x.reason for x in f)
      and not any("will deny" in x.reason for x in f))
# Known state (KY letterhead) + in-jurisdiction restrictive policy → the
# definite "will deny" FAIL applies. Guard data-driven: some restrictive
# policy governing 11055 must actually serve KY.
if any(_STORE.policy_applies_in_state(p, "KY") for p in _p11055):
    f = run(a, base(cpt=[{"code": "11055", "linked_diagnoses": ["I10"]}],
                    icd=[{"code": "I10", "type": "primary"}],
                    insurance="Medicare Part B - Louisville, KY 40202"))
    check("known state, in-jurisdiction mismatch → definite 'will deny' FAIL",
          any(x.status == Status.FAIL and "will deny" in x.reason for x in f))
# LCDs are MEDICARE policies — the same mismatch on a non-Medicare payer is
# an advisory, never a denial prediction (they don't bind Medicaid/commercial)
f = run(a, base(cpt=[{"code": "11055", "linked_diagnoses": ["I10"]}],
                icd=[{"code": "I10", "type": "primary"}], insurance="Aetna"))
check("same mismatch, commercial payer → UNKNOWN without payer policy",
      any(x.status == Status.UNKNOWN for x in f))
# qualifying systemic dx (E11.42 diabetic neuropathy) → covered
f = run(a, base(cpt=[{"code": "11055", "linked_diagnoses": ["E11.42"]}],
                icd=[{"code": "E11.42", "type": "primary"}],
                insurance="Medicare Part B"))
check("routine foot care + qualifying dx E11.42 → no coverage FAIL",
      not any(x.status == Status.FAIL for x in f))
# MAC jurisdiction: 29445 (note-031 regression) — every restrictive policy
# governing it is issued by a non-Florida MAC, so a Florida claim must not
# be gated by any of them (previously all 8 fired and FAILed the claim).
# The exclusion must be VISIBLE as a PASS-trail finding, not a silent skip.
f = run(a, base(cpt=[{"code": "29445", "linked_diagnoses": ["M14.671"]}],
                icd=[{"code": "M14.671", "type": "primary"}],
                insurance="Medicare Part B - Miami, FL 33101"))
check("29445 on Florida claim → no WARN/FAIL from out-of-jurisdiction policies",
      not any(x.status in (Status.WARN, Status.FAIL) for x in f))
check("jurisdiction exclusion leaves an explicit PASS trail",
      any(x.status == Status.PASS and "out-of-jurisdiction" in x.reason for x in f))
# zero-ICD policies are documentation policies, not diagnosis gates: in
# CGS territory (KY) 29445 is governed only by L34049/A57067, which publish
# NO covered-ICD list — must not deny for "no qualifying diagnosis"
assert not _STORE.coverage_policy_has_dx_rules("L34049")  # guard: still zero-ICD
f = run(a, base(cpt=[{"code": "29445", "linked_diagnoses": ["M14.671"]}],
                icd=[{"code": "M14.671", "type": "primary"}],
                insurance="Medicare Part B - Louisville, KY 40202"))
check("zero-ICD policy in-jurisdiction → non-restrictive, no WARN/FAIL",
      not any(x.status in (Status.WARN, Status.FAIL) for x in f))
check("zero-ICD pass leaves an explicit PASS trail",
      any(x.status == Status.PASS and "no covered-diagnosis list" in x.reason for x in f))
# CMS-1500 pointer scope: covered dx present on the CLAIM but not pointed at
# the line → repoint WARN, not a medical-necessity FAIL
f = run(a, base(cpt=[{"code": "11055", "linked_diagnoses": ["I10"]}],
                icd=[{"code": "I10", "type": "primary"}, {"code": "E11.42", "type": "secondary"}],
                insurance="Medicare Part B"))
check("covered-but-unlinked dx on claim → repoint WARN not FAIL",
      any(x.status == Status.WARN and "Point 11055 at E11.42" in x.reason for x in f)
      and not any(x.status == Status.FAIL for x in f))
# MCD "Not Applicable" placeholder (XX000): must never survive ingestion as a
# fake diagnosis gate (it made 55 real policies unsatisfiable)
check("XX000 placeholder purged from coverage tables",
      _STORE.conn.execute("SELECT COUNT(*) FROM coverage_icd WHERE icd_code='XX000'").fetchone()[0] == 0)
# Group-N explicit noncovered list (mirror of the covered list): seed a
# noncovered-only policy for a CPT with no other coverage policy
_STORE.conn.execute("DELETE FROM coverage_cpt WHERE policy_id='ATEST1'")
_STORE.conn.execute("DELETE FROM coverage_icd_noncovered WHERE policy_id='ATEST1'")
_STORE.conn.execute("DELETE FROM coverage_policy WHERE policy_id='ATEST1'")
_STORE.conn.execute("INSERT INTO coverage_cpt VALUES ('ATEST1','97810')")  # acupuncture; no seed policy governs it
_STORE.conn.execute("INSERT INTO coverage_icd_noncovered VALUES ('ATEST1','I10')")
_STORE.conn.execute(
    "INSERT INTO coverage_policy "
    "(policy_id,title,contractor,states,effective_from,effective_to,temporal_authority) "
    "VALUES ('ATEST1','Test noncovered-only policy','','','2026-01-01','2026-12-31',1)"
)
f = run(a, base(cpt=[{"code": "97810", "linked_diagnoses": ["I10"]}],
                icd=[{"code": "I10", "type": "primary"}], insurance="Medicare Part B"))
check("dx on explicit Group-N noncovered list → FAIL",
      any(x.status == Status.FAIL and "explicitly listed as NOT" in x.reason for x in f))
f = run(a, base(cpt=[{"code": "97810", "linked_diagnoses": ["M54.50"]}],
                icd=[{"code": "M54.50", "type": "primary"}], insurance="Medicare Part B"))
check("dx not on the noncovered list, no covered list → no WARN/FAIL",
      not any(x.status in (Status.WARN, Status.FAIL) for x in f))
_STORE.conn.execute("DELETE FROM coverage_cpt WHERE policy_id='ATEST1'")
_STORE.conn.execute("DELETE FROM coverage_icd_noncovered WHERE policy_id='ATEST1'")
_STORE.conn.execute("DELETE FROM coverage_policy WHERE policy_id='ATEST1'")

print("\n[MAC jurisdiction resolution]")
from app.compliance import geo
check("Coverage-API spelling (WPS Insurance Corporation) resolves",
      geo.contractor_states("WPS Insurance Corporation (MAC - Part A, MAC - Part B)") == {"IA", "KS", "MO", "NE", "IN", "MI"})
check("MCD-export truncated spelling (Wisconsin Physicians Service Insurance C) resolves",
      geo.contractor_states("Wisconsin Physicians Service Insurance C (MAC - Part B)") == {"IA", "KS", "MO", "NE", "IN", "MI"})
check("joint DME MAC policy resolves to union of DME areas",
      (lambda s: s and {"FL", "NY", "CA", "OH"} <= s)(
          geo.contractor_states("CGS Administrators, LLC (DME MAC) Noridian Healthcare Solutions, LLC (DME MAC)")))
check("unknown contractor → None (applies everywhere)",
      geo.contractor_states("Some Future Contractor, Inc.") is None)
check("state from insurance name", geo.state_from_text("Medicaid Florida") == "FL")
check("state from letterhead ZIP", geo.state_from_text("Foot & Ankle Clinic, Louisville, KY 40202") == "KY")
check("West Virginia not mistaken for Virginia",
      geo.state_from_text("somewhere in West Virginia today") == "WV")
check("no state signal → None", geo.state_from_text("Aetna PPO, Member ID 123") is None)
# a CPT with no coverage policy → never flagged
f = run(a, base(cpt=[{"code": "99213", "linked_diagnoses": ["M20.11"]}], icd=[{"code": "M20.11", "type": "primary"}]))
check("CPT with no coverage policy → no finding", len(f) == 0)
# claim bills procedures but has NO diagnosis at all → FAIL (guards dropped-dx bug)
f = run(a, base(cpt=[{"code": "99204"}, {"code": "73630"}], icd=[],
                insurance="Medicare Part B"))
check("procedures with zero diagnoses → FAIL", any(x.status == Status.FAIL and "NO diagnosis" in x.reason for x in f))
# One governing source record lacks an authoritative effective date.  The
# agent must stop before applying even a familiar class-findings rule; another
# active policy cannot prove that the undated version was in force on the DOS.
f = run(a, base(cpt=[{"code": "11721", "linked_diagnoses": ["E11.42"]}],
                icd=[{"code": "E11.42", "type": "primary"}], insurance="Medicare Part B"))
check("undated governing coverage article → UNKNOWN, not a guessed rule result",
      any(x.status == Status.UNKNOWN
          and x.clause == "coverage_policy_temporal_authority" for x in f)
      and not any(x.status in (Status.WARN, Status.FAIL) for x in f))
f = run(a, base(cpt=[{"code": "11721", "linked_diagnoses": ["E11.42"], "modifiers": ["Q8"]}],
                icd=[{"code": "E11.42", "type": "primary"}], insurance="Medicare Part B"))
check("supplying a modifier cannot bypass missing temporal authority",
      any(x.status == Status.UNKNOWN
          and x.clause == "coverage_policy_temporal_authority" for x in f))

print("\n[Medical Necessity — claim-composition (conjunction) gate]")
# Synthetic grammar policy on 97810 (no seed policy governs it): group 1 is
# primary-eligible (I10), group 2 is the required secondary (R51.9) — the
# A57193/A56232 mycotic-nail shape ("primary must be accompanied by the
# symptom secondary") without depending on live MCD data.
for _t in ("coverage_cpt", "coverage_icd", "coverage_group", "coverage_policy"):
    _STORE.conn.execute(f"DELETE FROM {_t} WHERE policy_id='ATEST2'")
_STORE.conn.execute("INSERT INTO coverage_cpt VALUES ('ATEST2','97810')")
_STORE.conn.execute("INSERT INTO coverage_icd VALUES ('ATEST2','I10',1)")
_STORE.conn.execute("INSERT INTO coverage_icd VALUES ('ATEST2','R519',2)")
_STORE.conn.executemany(
    "INSERT INTO coverage_group VALUES ('ATEST2',?,?,?,?)",
    [(1, "primary_eligible", "", "test primary group"),
     (2, "required_secondary", "", "test secondary group")])
_STORE.conn.execute(
    "INSERT INTO coverage_policy "
    "(policy_id,title,contractor,states,effective_from,effective_to,temporal_authority) "
    "VALUES ('ATEST2','Test composition policy','','','2026-01-01','2026-12-31',1)")
a = MedicalNecessityAgent(_STORE)
f = run(a, base(cpt=[{"code": "97810", "linked_diagnoses": ["I10"]}],
                icd=[{"code": "I10", "type": "primary"}], insurance="Medicare Part B"))
check("primary-eligible dx alone → composition FAIL (Medicare)",
      any(x.status == Status.FAIL and x.clause == "coverage_composition"
          and "accompanied" in x.reason for x in f))
check("composition FAIL carries a repair fix, and never says drop the primary",
      any(x.suggested_fix.get("action") == "add_required_secondary_dx"
          and "Do not remove" in x.recommendation for x in f))
f = run(a, base(cpt=[{"code": "97810", "linked_diagnoses": ["I10", "R51.9"]}],
                icd=[{"code": "I10", "type": "primary"}, {"code": "R51.9", "type": "secondary"}],
                insurance="Medicare Part B"))
check("primary + required secondary linked → no WARN/FAIL",
      not any(x.status in (Status.WARN, Status.FAIL) for x in f))
f = run(a, base(cpt=[{"code": "97810", "linked_diagnoses": ["I10"]}],
                icd=[{"code": "I10", "type": "primary"}, {"code": "R51.9", "type": "secondary"}],
                insurance="Medicare Part B"))
check("required secondary on claim but unlinked → repoint WARN, not FAIL",
      any(x.status == Status.WARN and x.clause == "diagnosis_pointer" for x in f)
      and not any(x.status == Status.FAIL for x in f))
f = run(a, base(cpt=[{"code": "97810", "linked_diagnoses": ["I10"]}],
                icd=[{"code": "I10", "type": "primary"}], insurance="Aetna"))
check("same composition gap, commercial payer → UNKNOWN without payer policy",
      any(x.status == Status.UNKNOWN for x in f))
# a group scoped to OTHER procedure codes must not participate for this line:
# with the primary group scoped away, I10 no longer reaches the policy at all
_STORE.conn.execute("UPDATE coverage_group SET cpt_scope='11111' "
                    "WHERE policy_id='ATEST2' AND group_id=1")
f = run(a, base(cpt=[{"code": "97810", "linked_diagnoses": ["I10"]}],
                icd=[{"code": "I10", "type": "primary"}], insurance="Medicare Part B"))
check("primary group scoped to another CPT → plain coverage FAIL, not composition",
      any(x.status == Status.FAIL and x.clause == "diagnosis_coverage" for x in f)
      and not any(x.clause == "coverage_composition" for x in f))
_STORE.conn.execute("UPDATE coverage_group SET cpt_scope='' "
                    "WHERE policy_id='ATEST2' AND group_id=1")
# authoritative per-policy states (coverage_policy.states) scope jurisdiction:
# a policy whose MAC serves only FL must not gate a KY claim
_STORE.conn.execute("UPDATE coverage_policy SET states='FL' WHERE policy_id='ATEST2'")
f = run(a, base(cpt=[{"code": "97810", "linked_diagnoses": ["I10"]}],
                icd=[{"code": "I10", "type": "primary"}],
                insurance="Medicare Part B - Louisville, KY 40202"))
check("states-column jurisdiction: FL-only policy silent on KY claim",
      not any(x.status in (Status.WARN, Status.FAIL) for x in f))
for _t in ("coverage_cpt", "coverage_icd", "coverage_group", "coverage_policy"):
    _STORE.conn.execute(f"DELETE FROM {_t} WHERE policy_id='ATEST2'")
_STORE.conn.commit()

print("\n[claim state from letterhead service_facility (C2)]")
# The vision extractor captures the letterhead/footer band as
# patient_metadata.service_facility — the band note_full_text never
# contains (observed live: routine_00003's 'Columbus, OH 43215' letterhead
# was invisible to jurisdiction routing). The structured state wins when
# it is a real USPS abbreviation; junk falls back to free-text inference.
r = base(insurance="Aetna")
r["patient_metadata"]["service_facility"] = {
    "name": "Buckeye Foot & Ankle", "city": "Columbus",
    "state": "OH", "zip": "43215"}
c = build_claim(r)
check("service_facility.state (valid USPS) wins", c.state == "OH")
check("service_facility.name becomes provider organization",
      c.provider.organization_name == "Buckeye Foot & Ankle")
r["patient_metadata"]["service_facility"] = {
    "name": "X", "state": "Ohio", "zip": "43215", "city": "Columbus"}
c = build_claim(r)
check("non-USPS state string falls back to text inference (ZIP+city)",
      c.state == "OH")
r["patient_metadata"]["service_facility"] = {}
r["patient_metadata"]["insurance"] = "Aetna PPO, Member 123"
c = build_claim(r)
check("no facility, no state signal anywhere → None (conservative)",
      c.state is None)

print("\n[scrubber advisory suppression — clause scoping]")
# A suppression retires exactly the (filter, code, CLAUSE) it was verified
# against. Blunt (filter, code) matching let a rule verified against the
# class-findings-modifier clause silently retire the sibling composition
# gate on the same code (observed live, routine_00003).
from app.compliance.engine import _apply_advisory_suppressions
from app.compliance.models import Finding as _F

_warn = lambda clause, code="11720": _F(
    filter_id="MEDICAL_NECESSITY", filter_name="Medical necessity",
    status=Status.WARN, codes=[code], denial_risk=DenialRisk.HIGH,
    reason="advisory", recommendation="r", source_rule="s", clause=clause)
_supp = [{"filter_id": "MEDICAL_NECESSITY", "code": "11720",
          "clause": "class_findings_modifier", "rule_id": "test-rule",
          "authority": "LCD", "note": "documented pathway"}]
out = _apply_advisory_suppressions(
    [_warn("class_findings_modifier"), _warn("coverage_composition")],
    _supp, "MEDICAL_NECESSITY")
check("matching clause WARN → replaced with PASS audit-trail finding",
      any(x.status == Status.PASS and x.clause == "class_findings_modifier"
          for x in out))
check("sibling clause on the same code survives untouched",
      any(x.status == Status.WARN and x.clause == "coverage_composition"
          for x in out))
_fail = _F(filter_id="MEDICAL_NECESSITY", filter_name="Medical necessity",
           status=Status.FAIL, codes=["11720"], denial_risk=DenialRisk.HIGH,
           reason="gate", recommendation="r", source_rule="s",
           clause="class_findings_modifier")
out = _apply_advisory_suppressions([_fail], _supp, "MEDICAL_NECESSITY")
check("FAIL is never config-suppressible even on exact clause match",
      out[0].status == Status.FAIL)
out = _apply_advisory_suppressions(
    [_warn("")], _supp, "MEDICAL_NECESSITY")
check("clause-carrying directive never matches a clauseless finding",
      out[0].status == Status.WARN)

print("\n[Prior Auth #10]")
from app.compliance.agents.prior_auth import PriorAuthAgent
# seed a PA-required code for the test (data-driven; table has whatever
# data/codes/prior_auth_*.json loaded — this adds one more row for a payer_id
# that matches Medicare, consistent with the 6-column schema: payer, code,
# category, hcpcs_prefix, note, source)
_STORE.conn.execute("DELETE FROM prior_auth_required WHERE code='J9999'")
_STORE.conn.execute(
    "INSERT INTO prior_auth_required VALUES ('medicare','J9999',NULL,NULL,'test drug','test seed')"
)
a = PriorAuthAgent(_STORE)
f = run(a, base(cpt=[{"code": "J9999"}], insurance="Medicare Part B"))
check("partial payer corpus → UNKNOWN before individual rules are trusted",
      any(x.status == Status.UNKNOWN and "complete" in x.reason.lower() for x in f))
# Exercise rule matching only after the fixture explicitly declares a complete,
# effective corpus. Production seed files remain incomplete and therefore
# cannot turn an absent row into an autonomous pass.
_STORE.conn.execute(
    "UPDATE prior_auth_policy SET complete=1,effective_from='2026-01-01',"
    "effective_to='2026-12-31' WHERE payer='medicare' AND plan=''"
)
f = run(a, base(cpt=[{"code": "J9999"}], insurance="Medicare Part B"))
check("PA-required code, no auth on file → FAIL",
      any(x.status == Status.FAIL for x in f))
r = base(cpt=[{"code": "J9999"}], insurance="Medicare Part B"); r["patient_metadata"]["prior_auth_number"] = "AUTH123"
f = run(a, r)
check("bare auth number without code/unit/DOS verification → UNKNOWN",
      any(x.status == Status.UNKNOWN for x in f)
      and not any(x.status in (Status.WARN, Status.FAIL) for x in f))
f = run(a, base(cpt=[{"code": "11721"}], insurance="Medicare Part B"))
check("code not requiring PA → no finding", len(f) == 0)
f = run(a, base(cpt=[{"code": "J9999"}]))  # no insurance text -> unrecognized payer
check("unrecognized payer → UNKNOWN (not defaulted to Medicare)",
      any(x.status == Status.UNKNOWN for x in f))

print("\n[Eligibility #11]")
from app.compliance.agents.benefits import BenefitsAgent
from app.compliance.adapters.stedi import ClearinghouseAdapter, EligibilityResult
class FakeAdapter(ClearinghouseAdapter):
    def __init__(self, active, service_covered=None):
        self._active = active
        self._service_covered = service_covered
        self.calls = []
    def is_configured(self): return True
    def check_eligibility(self, **kw):
        self.calls.append(kw)
        return EligibilityResult(
            configured=True, checked=True, active=self._active,
            service_coverage_confirmed=self._service_covered)
def elig_claim(active_member=True):
    r = base(cpt=[{"code": "11721"}], insurance="Medicare Part B")
    r["patient_metadata"].update({
        "member_id": "M123", "patient_last_name": "Doe",
        "provider_npi": "1234567893",
    })
    return r
f = BenefitsAgent(_STORE, FakeAdapter(active=False, service_covered=False)).check(build_claim(elig_claim()))
check("inactive coverage → FAIL", any(x.status == Status.FAIL for x in f))
adapter = FakeAdapter(active=True, service_covered=True)
f = BenefitsAgent(_STORE, adapter).check(build_claim(elig_claim()))
check("active coverage → no finding", len(f) == 0)
check("eligibility request is tied to DOS and exact service",
      len(adapter.calls) == 1
      and str(adapter.calls[0]["date_of_service"]) == "2026-07-20"
      and adapter.calls[0]["procedure_code"] == "11721"
      and adapter.calls[0]["product_or_service_id_qualifier"] == "CJ")
f = BenefitsAgent(_STORE, FakeAdapter(active=True)).check(build_claim(elig_claim()))
check("active plan without service confirmation → UNKNOWN",
      any(x.status == Status.UNKNOWN for x in f))
f = BenefitsAgent(_STORE, FakeAdapter(active=True, service_covered=True)).check(build_claim(base(cpt=[{"code": "11721"}])))
check("no member id → UNKNOWN (cannot check at coding stage)",
      any(x.status == Status.UNKNOWN for x in f))

print("\n[Documentation #12]")
from app.compliance.agents.documentation import DocumentationAgent
a = DocumentationAgent(_STORE)
f = run(a, base(cpt=[{"code": "11721"}]))  # no supporting text/evidence
check("code with no documentation → WARN", any(x.status == Status.WARN for x in f))
r = base(cpt=[{"code": "11721", "evidence_spans": ["nail debridement performed on 5 nails"]}],
         icd=[{"code": "M20.11", "type": "primary", "supporting_text": "hallux valgus right"}])
f = run(a, r)
check("documented code + dx → no finding", len(f) == 0)

print("\n[Billability #13 — HCPCS coverage code]")
from app.compliance.agents.billability import BillabilityAgent
a = BillabilityAgent(_STORE)
# A4570 carries HCPCS coverage code 'I' (not payable by Medicare) — verified live
assert _STORE.hcpcs_noncoverage_reason("A4570")
r = base(insurance="Medicare Part B")
r["hcpcs_codes"] = [{"code": "A4570", "units": 1}]
f = run(a, r)
check("HCPCS coverage 'I' code + Medicare payer → FAIL",
      any(x.status == Status.FAIL and "coverage code" in x.reason for x in f))
r = base(insurance="Aetna PPO")
r["hcpcs_codes"] = [{"code": "A4570", "units": 1}]
f = run(a, r)
check("HCPCS coverage 'I' code + commercial payer → no coverage finding",
      not any("coverage code" in x.reason for x in f))
assert _STORE.hcpcs_noncoverage_reason("A6212") is None  # covered supply
r = base(insurance="Medicare Part B")
r["hcpcs_codes"] = [{"code": "A6212", "units": 1}]
f = run(a, r)
check("covered HCPCS (A6212) + Medicare → no coverage finding",
      not any("coverage code" in x.reason for x in f))

print("\n[Modifiers #4 — PFS payment-policy indicators]")
a = ModifierAgent(_STORE)
# indicators verified live from the store so the tests don't rot with data updates
assert _STORE.pfs_indicators("29445").get("pctc_ind") == "0"
f = run(a, base(cpt=[{"code": "29445", "modifiers": ["26"]}]))
check("modifier 26 on no-PC/TC-split code → FAIL",
      any(x.status == Status.FAIL and "PC/TC" in x.reason for x in f))
assert _STORE.pfs_indicators("93923").get("pctc_ind") == "1"
f = run(a, base(cpt=[{"code": "93923", "modifiers": ["26"]}]))
check("modifier 26 on PC/TC-split code (93923) → no PC/TC finding",
      not any("PC/TC" in x.reason for x in f))
assert _STORE.pfs_indicators("93923").get("bilat_surg") == "2"
f = run(a, base(cpt=[{"code": "93923", "modifiers": ["50"]}]))
check("modifier 50 on inherently-bilateral code → FAIL",
      any(x.status == Status.FAIL and "bilateral" in x.reason for x in f))
assert _STORE.pfs_indicators("29445").get("asst_surg") == "1"
f = run(a, base(cpt=[{"code": "29445", "modifiers": ["80"]}]))
check("assistant-surgeon 80 on restricted code → FAIL",
      any(x.status == Status.FAIL and "assistant" in x.reason for x in f))
assert _STORE.pfs_indicators("29445").get("co_surg") == "0"
f = run(a, base(cpt=[{"code": "29445", "modifiers": ["62"]}]))
check("co-surgeon 62 on not-permitted code → FAIL",
      any(x.status == Status.FAIL and "co-surgeon" in x.reason for x in f))
f = run(a, base(cpt=[{"code": "29445", "modifiers": ["RT"]}]))
check("plain RT on 29445 → no PFS-indicator finding",
      not any("PFS" in x.source_rule for x in f))

print("\n[MCE diagnosis edits #14]")
from app.compliance.agents.mce import MCEAgent
a = MCEAgent(_STORE)


def mce_claim(icd, dob=None, dos="2026-02-05"):
    r = base(icd=icd, dos=dos)
    if dob:
        r["patient_metadata"]["date_of_birth"] = dob
    return r


# age conflict: newborn-list code (Z38.00, verified in store) on a 67-year-old
assert "age_newborn" in _STORE.mce_families("Z38.00")
f = run(a, mce_claim([{"code": "Z38.00", "type": "primary"}], dob="03/14/1958"))
check("newborn-list dx on adult patient → FAIL age conflict",
      any(x.status == Status.FAIL and "age conflict" in x.reason for x in f))
f = run(a, mce_claim([{"code": "Z38.00", "type": "primary"}]))
check("newborn-list dx, DOB unknown → no age finding (can't compute age)",
      not any("age conflict" in x.reason for x in f))
# manifestation as principal (D63.1, anemia in CKD — verified in store)
assert "manifestation_not_pdx" in _STORE.mce_families("D63.1")
f = run(a, mce_claim([{"code": "D63.1", "type": "primary"}, {"code": "N18.6", "type": "secondary"}]))
check("manifestation code as principal → FAIL",
      any(x.status == Status.FAIL and "manifestation" in x.reason for x in f))
f = run(a, mce_claim([{"code": "N18.6", "type": "primary"}, {"code": "D63.1", "type": "secondary"}]))
check("same manifestation code as SECONDARY → no finding", len(f) == 0)
# unacceptable principal (Z99.2 — verified in store)
assert "unacceptable_pdx" in _STORE.mce_families("Z99.2")
f = run(a, mce_claim([{"code": "Z99.2", "type": "primary"}]))
check("unacceptable-principal dx as principal → FAIL",
      any(x.status == Status.FAIL and "unacceptable" in x.reason for x in f))
# the MCE's own carve-out: Z51.89 acceptable as principal WITH a secondary
if "unacceptable_pdx_unless_secondary" in _STORE.mce_families("Z51.89"):
    f = run(a, mce_claim([{"code": "Z51.89", "type": "primary"}, {"code": "N18.6", "type": "secondary"}]))
    check("carve-out code as principal WITH secondary → no finding", len(f) == 0)
    f = run(a, mce_claim([{"code": "Z51.89", "type": "primary"}]))
    check("carve-out code as principal, NO secondary → FAIL",
          any(x.status == Status.FAIL for x in f))
# external cause as principal (chapter 20 structural boundary)
f = run(a, mce_claim([{"code": "W01.0XXA", "type": "primary"}, {"code": "S93.401A", "type": "secondary"}]))
check("external-cause code as principal → FAIL",
      any(x.status == Status.FAIL and "external-cause" in x.reason for x in f))
# duplicate of principal
f = run(a, mce_claim([{"code": "E11.9", "type": "primary"}, {"code": "E11.9", "type": "secondary"}]))
check("secondary duplicates principal → WARN",
      any(x.status == Status.WARN and "duplicate" in x.reason.lower() for x in f))
# clean claim
f = run(a, mce_claim([{"code": "E11.9", "type": "primary"}, {"code": "I10", "type": "secondary"}], dob="03/14/1958"))
check("age-appropriate, well-sequenced claim → no findings", len(f) == 0)

# --------------------------------------------------------------------- #
print("\n[Modifiers #4 — surgical modifier grammar (private-practice professional claims)]")
a = ModifierAgent(_STORE)

# E/M-only modifier on a procedure line
f = run(a, base(cpt=[{"code": "29445", "modifiers": ["25", "RT"]}]))
check("modifier 25 on a procedure line → FAIL (E/M-only)",
      any(x.status == Status.FAIL and "E/M services only" in x.reason for x in f))
# procedure-only modifier on an E/M line
f = run(a, base(cpt=[{"code": "99215", "modifiers": ["58"]}]))
check("modifier 58 on an E/M line → FAIL (procedure-only)",
      any(x.status == Status.FAIL and "procedure modifier" in x.reason for x in f))

# split-care 54/55/56
assert (_STORE.global_period("28296") or "").strip() == "090"   # guard: bunionectomy 090 global
assert (_STORE.global_period("29445") or "").strip() == "000"   # guard: TCC 000 global
f = run(a, base(cpt=[{"code": "28296", "modifiers": ["54", "55"]}]))
check("54+55 on one line → FAIL (exclusive split-care parts)",
      any(x.status == Status.FAIL and "exclusive part" in x.reason for x in f))
f = run(a, base(cpt=[{"code": "29445", "modifiers": ["54", "RT"]}]))
check("54 on a 000-global code → FAIL (no package to split)",
      any(x.status == Status.FAIL and "positive numeric global period" in x.reason
          for x in f))
f = run(a, base(cpt=[{"code": "28296", "modifiers": ["54", "RT"]}]))
check("54 on a 090-global code → no split-care finding",
      not any("split" in x.reason.lower() or "010/090" in x.reason for x in f))

# facility-only discontinued modifiers on a professional claim
f = run(a, base(cpt=[{"code": "28296", "modifiers": ["73"]}]))
check("modifier 73 on professional claim → FAIL (institutional-only)",
      any(x.status == Status.FAIL and "institutional" in x.reason.lower() for x in f))
f = run(a, base(cpt=[{"code": "28296", "modifiers": ["52", "53"]}]))
check("52+53 combined → FAIL (mutually exclusive outcomes)",
      any(x.status == Status.FAIL and "reduced vs discontinued" in x.source_rule
          for x in f))

# modifier 63: Appendix F exemption + patient-age impossibility
row = _STORE.conn.execute(
    "SELECT code FROM modifier_exempt WHERE modifier_63_exempt=1 LIMIT 1").fetchone()
if row:
    f = run(a, base(cpt=[{"code": row["code"], "modifiers": ["63"]}], dob="01/01/2026"))
    check(f"63 on Appendix-F exempt code {row['code']} → FAIL",
          any(x.status == Status.FAIL and "Appendix F" in x.source_rule for x in f))
f = run(a, base(cpt=[{"code": "28296", "modifiers": ["63"]}], dob="03/14/1958"))
check("63 on an adult patient → FAIL (cannot be an infant under 4 kg)",
      any(x.status == Status.FAIL and "4 kg" in x.reason for x in f))
f = run(a, base(cpt=[{"code": "28296", "modifiers": ["63"]}]))
check("63 with unknown DOB and non-exempt code → no age finding (conservative)",
      not any("4 kg" in x.reason for x in f))

# modifier 51 vs PFS multiple-procedure indicator
row = _STORE.conn.execute(
    "SELECT code FROM global_period WHERE mult_proc IN ('0','9') "
    "AND code GLOB '[0-9]*' AND CAST(code AS INT) BETWEEN 10000 AND 69999 LIMIT 20").fetchall()
mp_code = next((r["code"] for r in row if not _STORE.is_modifier_51_exempt(r["code"])), None)
if mp_code:
    f = run(a, base(cpt=[{"code": mp_code, "modifiers": ["51"]}]))
    check(f"51 on mult_proc=0/9 code {mp_code} → WARN (no reduction concept)",
          any(x.status == Status.WARN and "multiple-" in x.reason for x in f))

# bilateral units double-billing
brow = _STORE.conn.execute(
    "SELECT code FROM global_period WHERE bilat_surg='1' AND code GLOB '[0-9]*' LIMIT 1").fetchone()
if brow:
    f = run(a, base(cpt=[{"code": brow["code"], "modifiers": ["50"], "units": 2}]))
    check("50 with 2 units → WARN (150% adjustment already pays both sides)",
          any(x.status == Status.WARN and "150%" in x.reason for x in f))
    f = run(a, base(cpt=[{"code": brow["code"], "modifiers": ["50"], "units": 1}]))
    check("50 with 1 unit → no units finding",
          not any("150%" in x.reason for x in f))

# repeat modifiers 76/77
f = run(a, base(cpt=[{"code": "28296", "modifiers": ["76"]}]))
check("76 with no initial line → WARN (nothing to repeat)",
      any(x.status == Status.WARN and "repeat" in x.reason.lower() for x in f))
f = run(a, base(cpt=[{"code": "28296", "modifiers": []},
                     {"code": "28296", "modifiers": ["76"]}]))
check("76 with an initial unmodified line → no repeat finding",
      not any("no initial line" in x.reason for x in f))

# teaching-setting assistant modifier & anesthesia-by-surgeon
f = run(a, base(cpt=[{"code": "28296", "modifiers": ["82"]}]))
check("82 on private-practice claim → WARN (resident-unavailability certification)",
      any(x.status == Status.WARN and "RESIDENT" in x.reason for x in f))
f = run(a, base(cpt=[{"code": "28296", "modifiers": ["47"]}], insurance="Medicare Part B"))
check("47 (anesthesia by surgeon) on Medicare → WARN, HIGH denial risk",
      any(x.status == Status.WARN and "anesthesia" in x.reason.lower()
          and x.denial_risk == DenialRisk.HIGH for x in f))

# --------------------------------------------------------------------- #
print("\n[Surgical package #15 — NCCI Ch.1 descriptor designations]")
from app.compliance.agents.surgical_package import SurgicalPackageAgent
a = SurgicalPackageAgent(_STORE)

# find a real '(separate procedure)' surgical CPT and a surgical companion
# with NO published NCCI edit between them (the exact gap §J exists to cover)
sep_rows = _STORE.conn.execute(
    "SELECT code FROM code_set WHERE code_system='CPT' AND description LIKE "
    "'%(separate procedure)%' AND code GLOB '[0-9]*' "
    "AND CAST(code AS INT) BETWEEN 10000 AND 69999 LIMIT 25").fetchall()
comp_rows = _STORE.conn.execute(
    "SELECT code FROM code_set WHERE code_system='CPT' AND description NOT LIKE "
    "'%(separate procedure)%' AND code GLOB '[0-9]*' "
    "AND CAST(code AS INT) BETWEEN 27600 AND 28899 LIMIT 40").fetchall()
sep_pair = None
for sr in sep_rows:
    for cr in comp_rows:
        if (sr["code"] != cr["code"]
                and not _STORE.ncci_pair(sr["code"], cr["code"], _NCCI_DOS)):
            sep_pair = (sr["code"], cr["code"])
            break
    if sep_pair:
        break
if sep_pair:
    sp, comp = sep_pair
    f = run(a, base(cpt=[{"code": sp}, {"code": comp}]))
    check(f"'(separate procedure)' {sp} + companion {comp}, no modifier → WARN §J",
          any("separate procedure" in x.reason for x in f))
    f = run(a, base(cpt=[{"code": sp, "modifiers": ["XS"]}, {"code": comp}]))
    check("same pair with XS (distinct site) → §J silent",
          not any("separate procedure" in x.reason for x in f))
    f = run(a, base(cpt=[{"code": sp}]))
    check(f"'(separate procedure)' {sp} billed alone → silent (performed independently)",
          not any("separate procedure" in x.reason for x in f))

# anesthesia billed on the surgeon's own claim
f = run(a, base(cpt=[{"code": "00400"}, {"code": "28296"}], insurance="Medicare Part B"))
check("anesthesia + surgery on one professional claim, Medicare → FAIL §G",
      any(x.status == Status.FAIL and "anesthesia" in x.reason.lower() for x in f))
f = run(a, base(cpt=[{"code": "00400"}, {"code": "28296"}], insurance="Cigna PPO"))
check("same claim, commercial payer → WARN (advisory)",
      any(x.status == Status.WARN and "anesthesia" in x.reason.lower() for x in f))

# unlisted procedure codes
assert _STORE.is_unlisted_procedure("28899")  # guard: from the descriptor itself
f = run(a, base(cpt=[{"code": "28899"}]))
check("unlisted 28899 → WARN §T (manual pricing + op report)",
      any(x.status == Status.WARN and "unlisted" in x.reason.lower() for x in f))
f = run(a, base(cpt=[{"code": "28296"}]))
check("listed code → §T silent", not any("unlisted" in x.reason.lower() for x in f))

# --------------------------------------------------------------------- #
print("\n[Scrubber per-filter execution trail]")
from app.compliance.agents import build_default_agents
_scrubber = ClaimScrubber(_STORE, agents=build_default_agents(_STORE))
_sr = _scrubber.scrub(base(cpt=[{"code": "29445", "linked_diagnoses": ["M14.671"]}],
                           icd=[{"code": "M14.671", "type": "primary"}],
                           insurance="Medicare Part B - Miami, FL 33101"))
check("every agent appears in filter_results (checked-vs-skipped is auditable)",
      len(_sr.filter_results) == len(_scrubber.agents))
_mn = next((r for r in _sr.filter_results if r["filter_id"] == "MEDICAL_NECESSITY"), None)
check("MEDICAL_NECESSITY recorded even when it emits no WARN/FAIL",
      _mn is not None and _mn["status"] == "PASS")
check("statuses restricted to PASS/WARN/FAIL/UNKNOWN/ERROR",
      all(r["status"] in ("PASS", "WARN", "FAIL", "UNKNOWN", "ERROR")
          for r in _sr.filter_results))

# --------------------------------------------------------------------- #
# --- cleanup: remove the PA test row so the shared DB stays pristine ---
_STORE.conn.execute("DELETE FROM prior_auth_required WHERE code='J9999'")
_STORE.conn.execute(
    "UPDATE prior_auth_policy SET complete=0 WHERE payer='medicare' AND plan=''"
)
_STORE.conn.commit()

print(f"\n{'='*50}\nRESULT: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
