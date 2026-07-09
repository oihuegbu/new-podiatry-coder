"""Filter #5 — Medical necessity (LCD/NCD ICD↔CPT linkage).

A procedure can be perfectly coded and still deny as "not medically necessary"
if the linked diagnosis isn't on the policy's covered list. CMS publishes NCDs
(national) and MACs publish LCDs/Billing & Coding Articles listing which ICD-10
codes justify a given CPT.

This agent is fully data-driven off the `coverage_cpt` / `coverage_icd` tables.
Today they hold the routine-foot-care policy (L36199, generalized out of the old
hardcoded check); the same tables are populated from MCD Articles via
`store.load_coverage_articles(...)` with no change to this logic.
"""
from __future__ import annotations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status


class MedicalNecessityAgent(ComplianceAgent):
    filter_id = "MEDICAL_NECESSITY"
    filter_name = "Medical necessity (LCD/NCD coverage)"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        # all diagnoses available on the claim (CMS-1500: any dx supports any line)
        claim_dx = [d.code for d in claim.diagnoses]

        # A claim that bills procedures/supplies but carries NO diagnosis at all
        # cannot establish medical necessity — every CMS-1500 service line must
        # point to a diagnosis. This also guards against upstream coding failures
        # that drop the diagnoses (e.g. an LLM pass that fails to parse).
        billable = [ln for ln in claim.lines if ln.code]
        if billable and not claim_dx:
            findings.append(self.finding(
                status=Status.FAIL, codes=[ln.code for ln in billable],
                denial_risk=DenialRisk.HIGH,
                reason="Claim bills procedures/supplies but has NO diagnosis code — medical "
                       "necessity cannot be established and every service line requires a "
                       "diagnosis pointer.",
                recommendation="Add the supporting ICD-10-CM diagnosis(es); if none exist, the "
                               "services are not billable.",
                source_rule="CMS-1500 diagnosis-pointer requirement / ICD↔CPT linkage",
            ))

        for line in claim.lines:
            policies = self.store.coverage_policies_for_cpt(line.code)
            if not policies:
                continue  # no coverage policy governs this code

            for policy_id in policies:
                # the dx that can justify this line = its linked dx, else all claim dx
                candidate_dx = line.linked_diagnoses or claim_dx
                covered = any(
                    self.store.coverage_icd_covered(policy_id, dx) for dx in candidate_dx
                )
                if covered:
                    continue
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                    reason=f"{line.code} is governed by coverage policy {policy_id}, but none of the "
                           f"claim's diagnoses are on its covered list — will deny as not medically "
                           f"necessary.",
                    recommendation=f"Add a qualifying diagnosis covered under {policy_id} (and document "
                                   f"it), or do not bill {line.code}.",
                    source_rule=f"Coverage policy {policy_id} (LCD/NCD/Article ICD↔CPT list)",
                ))
        return findings
