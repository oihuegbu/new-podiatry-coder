"""Filter #9 — Place of Service & provider eligibility.

Phase-2 scope: validate the Place of Service code against the CMS POS set and
expose its facility/non-facility designation (drives the site-of-service
differential). An invalid POS code is a hard denial.

Per-code "not payable in this POS" edits and provider specialty/credentialing
checks require PFS payability data + a provider-enrollment source, which are not
yet ingested — those checks are added when those sources land (no logic rewrite,
the POS facility flag is already available here).
"""
from __future__ import annotations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status


class POSEligibilityAgent(ComplianceAgent):
    filter_id = "POS_ELIGIBILITY"
    filter_name = "Place of Service & provider eligibility"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        # POS may live at claim level or per line; check whatever is populated.
        pos_values = {claim.place_of_service} | {ln.place_of_service for ln in claim.lines}
        pos_values = {p for p in pos_values if p}

        for pos in pos_values:
            info = self.store.pos_info(pos)
            if info is None:
                findings.append(self.finding(
                    status=Status.FAIL, codes=[], denial_risk=DenialRisk.HIGH,
                    reason=f"Place of Service '{pos}' is not a valid POS code.",
                    recommendation="Use a valid 2-digit POS code from the CMS POS set.",
                    source_rule="CMS Place of Service code set",
                ))
        # Missing POS is intentionally not flagged here: when the extraction does
        # not capture POS it is a capture gap, not a coding error. POS-payability
        # enforcement is added with the PFS payability source.
        return findings
