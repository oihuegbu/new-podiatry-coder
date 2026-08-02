"""Filter #14 — CMS Medicare Code Editor (MCE) diagnosis edits.

The MCE is CMS's own claim-level diagnosis editor. Its edit lists (parsed
from CMS's "Definitions of Medicare Code Edits" text into mce_edits.json)
define rule families no other filter covers:

  * Age conflict — a diagnosis "clinically and virtually impossible in a
    patient of the stated age" (MCE's wording): newborn-only (age 0),
    pediatric (0-17), maternity (9-64), adult (15-124) code lists, with the
    ranges parsed from the MCE text itself.
  * Manifestation code as principal diagnosis — manifestation codes
    "describe the manifestation of an underlying disease, not the disease
    itself" and cannot be first-listed.
  * Unacceptable principal diagnosis — status/circumstance codes (many
    Z codes, B95-B97 organism codes) not permitted as principal. The MCE's
    own carve-out (codes acceptable when a secondary diagnosis is present)
    is honored from the data.
  * External cause codes as principal — ICD-10-CM Chapter 20
    codes "describe the circumstance causing an injury, not the nature of
    the injury". The chapter boundary is classification structure (like the
    E/M section range), not a hand-picked code list.
  * Duplicate of principal diagnosis — a secondary diagnosis identical to
    the principal.

Sex conflict is deliberately not implemented: CMS deactivated MCE sex
editing for all ICD-10 codes as of 10/01/2024.
"""
from __future__ import annotations

from datetime import date, datetime

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status

_AGE_FAMILIES = ("age_newborn", "age_pediatric", "age_maternity", "age_adult")


def _age_at(dos: date, dob_yyyymmdd: str) -> int | None:
    try:
        dob = datetime.strptime(dob_yyyymmdd, "%Y%m%d").date()
    except (TypeError, ValueError):
        return None
    if dob > dos:
        return None
    years = dos.year - dob.year - ((dos.month, dos.day) < (dob.month, dob.day))
    return years


class MCEAgent(ComplianceAgent):
    filter_id = "MCE_DX_EDITS"
    filter_name = "Medicare Code Editor diagnosis edits"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        primary = claim.primary_diagnosis
        secondaries = [d for d in claim.diagnoses if primary and d is not primary]

        # --- age conflict (needs both DOB and DOS to compute age) ---
        age = None
        if claim.date_of_service and claim.subscriber.date_of_birth:
            age = _age_at(claim.date_of_service, claim.subscriber.date_of_birth)
        if age is not None:
            for dx in claim.diagnoses:
                fams = self.store.mce_families(dx.code) & set(_AGE_FAMILIES)
                for fam in sorted(fams):
                    rng = self.store.mce_age_range(fam)
                    if rng and not (rng[0] <= age <= rng[1]):
                        cat = fam.replace("age_", "")
                        findings.append(self.finding(
                            status=Status.FAIL, codes=[dx.code], denial_risk=DenialRisk.HIGH,
                            reason=f"{dx.code} is on the MCE {cat} diagnosis list "
                                   f"(age {rng[0]}-{rng[1]}), but the patient is {age} "
                                   f"at the date of service — an age conflict the MCE "
                                   f"calls 'clinically and virtually impossible'.",
                            recommendation="Verify the diagnosis and the patient's date of "
                                           "birth — one of them is wrong.",
                            source_rule=f"MCE age conflict — {cat} list",
                        ))

        if primary is None:
            return findings
        pfam = self.store.mce_families(primary.code)

        # --- manifestation code as principal ---
        if "manifestation_not_pdx" in pfam:
            findings.append(self.finding(
                status=Status.FAIL, codes=[primary.code], denial_risk=DenialRisk.HIGH,
                reason=f"{primary.code} is an MCE manifestation code — it describes the "
                       f"manifestation of an underlying disease, not the disease itself, "
                       f"and cannot be the principal diagnosis.",
                recommendation="Sequence the underlying etiology first and this code as "
                               "secondary.",
                source_rule="MCE manifestation code as principal diagnosis",
            ))

        # --- unacceptable principal diagnosis (with the MCE's own carve-out) ---
        if "unacceptable_pdx_unless_secondary" in pfam and secondaries:
            pass  # the MCE itself accepts this code as principal when a secondary exists
        elif pfam & {"unacceptable_pdx", "unacceptable_pdx_unless_secondary"}:
            findings.append(self.finding(
                status=Status.FAIL, codes=[primary.code], denial_risk=DenialRisk.HIGH,
                reason=f"{primary.code} is on the MCE unacceptable-principal-diagnosis "
                       f"list — it describes a circumstance/status influencing health, "
                       f"not a current illness or injury, and cannot be first-listed.",
                recommendation="Re-sequence: bill the treated condition as principal and "
                               "this code as secondary.",
                source_rule="MCE unacceptable principal diagnosis",
            ))

        # --- external cause code as principal (ICD-10-CM Chapter 20) ---
        if self.store.is_external_cause(primary.code):
            findings.append(self.finding(
                status=Status.FAIL, codes=[primary.code], denial_risk=DenialRisk.HIGH,
                reason=f"{primary.code} is an external-cause code (ICD-10-CM Chapter 20) "
                       f"— it describes the circumstance causing an injury, not "
                       f"the injury itself, and cannot be the principal diagnosis.",
                recommendation="Code the nature of the injury/condition as principal; keep "
                               "the external-cause code as secondary.",
                source_rule="MCE external causes of morbidity as principal diagnosis",
            ))

        # --- duplicate of principal diagnosis ---
        pnorm = primary.code.replace(".", "").upper()
        for dx in secondaries:
            if dx.code.replace(".", "").upper() == pnorm:
                findings.append(self.finding(
                    status=Status.WARN, codes=[dx.code], denial_risk=DenialRisk.LOW,
                    reason=f"{dx.code} appears as both principal and secondary diagnosis "
                           f"— the secondary is a duplicate.",
                    recommendation="Remove the duplicate secondary diagnosis.",
                    source_rule="MCE duplicate of principal diagnosis",
                ))
        return findings
