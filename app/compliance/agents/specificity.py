"""Filter #1 — Code validity & specificity.

  * Every code must exist and be active for the date of service.
  * ICD-10 must be coded to the highest specificity: if a billed code has more
    specific children in the code set, an unspecified parent is a denial risk.
    (ICD-10 has a "code also / code to highest level" rule — payers deny an
    unspecified code when a more specific one exists.)
"""
from __future__ import annotations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status

# ICD-10 codes are billable only at a "leaf" level. A code that still has more
# specific children AND whose own length is short is almost certainly unspecified.
# We rely purely on the code set hierarchy (no hardcoded code lists).


class SpecificityAgent(ComplianceAgent):
    filter_id = "CODE_SPECIFICITY"
    filter_name = "Code validity & specificity"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        dos = claim.date_of_service

        # --- existence + active-for-DOS for every code on the claim ---
        for dx in claim.diagnoses:
            self._check_existence(findings, "ICD10", dx.code, "ICD-10-CM diagnosis", dos)
        for line in claim.lines:
            label = "CPT procedure" if line.code_system == "CPT" else "HCPCS"
            self._check_existence(findings, line.code_system, line.code, label, dos)

        # --- ICD specificity: unspecified-when-more-specific-exists ---
        for dx in claim.diagnoses:
            if not self.store.code_active_any_date("ICD10", dx.code):
                continue  # existence already reported above
            if self.store.children_exist("ICD10", dx.code):
                findings.append(self.finding(
                    status=Status.FAIL, codes=[dx.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"ICD-10 {dx.code} is not at the highest level of specificity — "
                           f"more specific codes exist under it.",
                    recommendation="Assign the most specific child code supported by the documentation "
                                   "(laterality, episode, severity, site).",
                    source_rule="ICD-10-CM tabular hierarchy",
                ))
        return findings

    def _check_existence(self, findings, system, code, label, dos):
        if self.store.code_exists(system, code, dos):
            return
        if self.store.code_active_any_date(system, code):
            findings.append(self.finding(
                status=Status.FAIL, codes=[code], denial_risk=DenialRisk.HIGH,
                reason=f"{label} {code} is not active for the date of service "
                       f"({dos.isoformat() if dos else 'unknown'}).",
                recommendation="Use a code valid on the DOS, or verify the date of service.",
                source_rule=f"{system} effective-date range",
            ))
        elif system == "ICD10" and self.store.children_exist(system, code):
            # Not a billable code itself, but more specific children exist →
            # this is a non-billable category header (unspecified-level coding).
            findings.append(self.finding(
                status=Status.FAIL, codes=[code], denial_risk=DenialRisk.HIGH,
                reason=f"{label} {code} is a non-billable category header — more specific "
                       f"codes exist beneath it; it cannot be billed at this level.",
                recommendation="Assign the most specific billable child code (laterality, "
                               "episode, severity, site) supported by the documentation.",
                source_rule="ICD-10-CM tabular hierarchy (highest-specificity rule)",
            ))
        else:
            findings.append(self.finding(
                status=Status.FAIL, codes=[code], denial_risk=DenialRisk.HIGH,
                reason=f"{label} {code} does not exist in the {system} code set.",
                recommendation="Remove or correct this code — it is not a valid code.",
                source_rule=f"{system} code set (FY2026)",
            ))
