"""Adversarial agent tests — craft claims with KNOWN violations and assert each
agent fires correctly (and stays silent on clean input). Run:  python -m tests.test_agents
"""
from __future__ import annotations

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
from app.compliance.models import Status

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


def base(cpt=None, icd=None, dos="2026-02-05", prior=None):
    return {
        "document_id": "TEST",
        "patient_metadata": {"date_of_service": dos},
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
f = run(a, base(cpt=[{"code": "0001U", "units": 5}]))  # MAI 2, cap 1
check("MAI-2 over cap → FAIL (hard wall)", any(x.status == Status.FAIL and "absolute" in x.reason.lower() for x in f))

f = run(a, base(cpt=[{"code": "0001U", "units": 1}]))  # within cap
check("MAI-2 within cap → no finding", len(f) == 0)

# find an MAI-1 code with a known cap for the split-line test
row = _STORE.conn.execute("SELECT code, mue_value FROM mue WHERE mai='1' AND mue_value=1 LIMIT 1").fetchone()
if row:
    code, cap = row["code"], row["mue_value"]
    f = run(a, base(cpt=[{"code": code, "units": cap + 2}]))
    check(f"MAI-1 over cap no modifier ({code}) → FAIL", any(x.status == Status.FAIL for x in f))
    f = run(a, base(cpt=[{"code": code, "units": cap + 2, "modifiers": ["59"]}]))
    check(f"MAI-1 over cap WITH 59 ({code}) → WARN not FAIL", any(x.status == Status.WARN for x in f) and not any(x.status == Status.FAIL for x in f))

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

# --------------------------------------------------------------------- #
print("\n[NCCI PTP #2]")
a = NCCIPTPAgent(_STORE)
f = run(a, base(cpt=[{"code": "15271"}, {"code": "C5271"}], dos="2026-05-01"))  # indicator 0
check("indicator-0 pair → FAIL (hard edit)", any(x.status == Status.FAIL for x in f))
f = run(a, base(cpt=[{"code": "26121"}, {"code": "C5271"}], dos="2026-05-01"))  # indicator 1, no mod
check("indicator-1 pair no modifier → FAIL", any(x.status == Status.FAIL for x in f))
f = run(a, base(cpt=[{"code": "26121"}, {"code": "C5271", "modifiers": ["59"]}], dos="2026-05-01"))
check("indicator-1 pair WITH 59 → WARN not FAIL", any(x.status == Status.WARN for x in f) and not any(x.status == Status.FAIL for x in f))
f = run(a, base(cpt=[{"code": "11721"}, {"code": "99213"}], dos="2026-05-01"))  # unrelated
check("non-conflicting pair → no NCCI finding", len(f) == 0)

print("\n[Modifiers #4]")
a = ModifierAgent(_STORE)
f = run(a, base(cpt=[{"code": "11721", "modifiers": ["ZZ"]}]))
check("unrecognized modifier ZZ → FAIL", any(x.status == Status.FAIL for x in f))
f = run(a, base(cpt=[{"code": "11721", "modifiers": ["RT", "LT"]}]))
check("RT+LT same line → WARN", any(x.status == Status.WARN for x in f))
f = run(a, base(cpt=[{"code": "11721", "modifiers": ["59"]}]))
check("modifier 59 → advisory WARN (prefer X{EPSU})", any("X{EPSU}" in x.reason or "XE/XS" in x.recommendation for x in f))
f = run(a, base(cpt=[{"code": "11721", "modifiers": ["RT"]}]))
check("valid single modifier RT → no finding", len(f) == 0)

# E/M + same-day procedure: modifier-25 bundling
f = run(a, base(cpt=[{"code": "99213"}, {"code": "11721"}]))  # E/M + surgery, no 25
check("E/M + procedure, no mod 25 → FAIL", any(x.status == Status.FAIL and "25" in x.reason for x in f))
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
# 11055 = routine foot care governed by L36199; non-qualifying dx (I10 hypertension) → FAIL
assert not _STORE.coverage_icd_covered("L36199", "I10")  # guard: confirm truly non-qualifying
f = run(a, base(cpt=[{"code": "11055", "linked_diagnoses": ["I10"]}], icd=[{"code": "I10", "type": "primary"}]))
check("routine foot care + non-qualifying dx (I10) → FAIL", any(x.status == Status.FAIL for x in f))
# qualifying systemic dx (E11.42 diabetic neuropathy) → covered
f = run(a, base(cpt=[{"code": "11055", "linked_diagnoses": ["E11.42"]}], icd=[{"code": "E11.42", "type": "primary"}]))
check("routine foot care + qualifying dx E11.42 → no finding", len(f) == 0)
# a CPT with no coverage policy → never flagged
f = run(a, base(cpt=[{"code": "99213", "linked_diagnoses": ["M20.11"]}], icd=[{"code": "M20.11", "type": "primary"}]))
check("CPT with no coverage policy → no finding", len(f) == 0)
# claim bills procedures but has NO diagnosis at all → FAIL (guards dropped-dx bug)
f = run(a, base(cpt=[{"code": "99204"}, {"code": "73630"}], icd=[]))
check("procedures with zero diagnoses → FAIL", any(x.status == Status.FAIL and "NO diagnosis" in x.reason for x in f))

print("\n[Prior Auth #10]")
from app.compliance.agents.prior_auth import PriorAuthAgent
# seed a PA-required code for the test (data-driven; table is empty by default)
_STORE.conn.execute("DELETE FROM prior_auth_required WHERE code='J9999'")
_STORE.conn.execute("INSERT INTO prior_auth_required VALUES ('Medicare','J9999','test drug')")
a = PriorAuthAgent(_STORE)
f = run(a, base(cpt=[{"code": "J9999"}]))
check("PA-required code, no auth on file → FAIL", any(x.status == Status.FAIL for x in f))
r = base(cpt=[{"code": "J9999"}]); r["patient_metadata"]["prior_auth_number"] = "AUTH123"
f = run(a, r)
check("PA-required code WITH auth number → WARN not FAIL", any(x.status == Status.WARN for x in f) and not any(x.status == Status.FAIL for x in f))
f = run(a, base(cpt=[{"code": "11721"}]))
check("code not requiring PA → no finding", len(f) == 0)

print("\n[Eligibility #11]")
from app.compliance.agents.benefits import BenefitsAgent
from app.compliance.adapters.stedi import ClearinghouseAdapter, EligibilityResult
class FakeAdapter(ClearinghouseAdapter):
    def __init__(self, active): self._active = active
    def is_configured(self): return True
    def check_eligibility(self, **kw): return EligibilityResult(configured=True, checked=True, active=self._active)
def elig_claim(active_member=True):
    r = base(cpt=[{"code": "11721"}])
    r["patient_metadata"].update({"member_id": "M123", "payer_id": "MEDICARE", "patient_last_name": "Doe"})
    return r
f = BenefitsAgent(_STORE, FakeAdapter(active=False)).check(build_claim(elig_claim()))
check("inactive coverage → FAIL", any(x.status == Status.FAIL for x in f))
f = BenefitsAgent(_STORE, FakeAdapter(active=True)).check(build_claim(elig_claim()))
check("active coverage → no finding", len(f) == 0)
f = BenefitsAgent(_STORE, FakeAdapter(active=True)).check(build_claim(base(cpt=[{"code": "11721"}])))
check("no member id → no finding (cannot check at coding stage)", len(f) == 0)

print("\n[Documentation #12]")
from app.compliance.agents.documentation import DocumentationAgent
a = DocumentationAgent(_STORE)
f = run(a, base(cpt=[{"code": "11721"}]))  # no supporting text/evidence
check("code with no documentation → WARN", any(x.status == Status.WARN for x in f))
r = base(cpt=[{"code": "11721", "evidence_spans": ["nail debridement performed on 5 nails"]}],
         icd=[{"code": "M20.11", "type": "primary", "supporting_text": "hallux valgus right"}])
f = run(a, r)
check("documented code + dx → no finding", len(f) == 0)

# --------------------------------------------------------------------- #
# --- cleanup: remove the PA test row so the shared DB stays pristine ---
_STORE.conn.execute("DELETE FROM prior_auth_required WHERE code='J9999'")
_STORE.conn.commit()

print(f"\n{'='*50}\nRESULT: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
