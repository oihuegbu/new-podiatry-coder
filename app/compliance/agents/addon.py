"""Filter #8 — Add-on code rules.

Add-on codes (CPT Appendix D — designated by the "list separately in addition"
/ "each additional" descriptor phrasing) may only be billed alongside a valid
primary procedure on the same claim. Billing one alone denies.

Add-on status is derived from the code descriptors (data-driven). The exact
add-on→valid-primary mapping comes from the CMS Add-On Code Edit file; until
that source is ingested we enforce the high-yield rule — an add-on billed with
NO primary procedure present — and note that precise primary matching is pending.
"""
from __future__ import annotations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status


class AddOnAgent(ComplianceAgent):
    filter_id = "ADDON"
    filter_name = "Add-on code rules"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        cpt_lines = [ln for ln in claim.lines if ln.code_system == "CPT"]
        addon_codes = {ln.code for ln in cpt_lines if self.store.is_addon(ln.code)}
        primary_lines = [ln for ln in cpt_lines if ln.code not in addon_codes]

        for ln in cpt_lines:
            if ln.code not in addon_codes:
                continue
            if not primary_lines:
                findings.append(self.finding(
                    status=Status.FAIL, codes=[ln.code], denial_risk=DenialRisk.HIGH,
                    reason=f"Add-on code {ln.code} is billed with no primary procedure on the claim.",
                    recommendation="Add the appropriate primary procedure code, or remove the add-on.",
                    source_rule="CPT Appendix D add-on designation",
                    auto_fixable=False,
                ))
            # else: a primary procedure exists. Precise primary-code matching is
            # validated once the CMS Add-On Code Edit file is ingested.
        return findings
