"""Filter #8 — Add-on code rules.

Add-on codes (CPT Appendix D — designated by the "list separately in addition"
/ "each additional" descriptor phrasing) may only be billed alongside a valid
primary procedure on the same claim. Billing one alone denies.

Add-on status is derived from the code descriptors (data-driven). Where the
CMS Add-On Code (AOC) Edit file specifies which primary code(s) a given
add-on requires (code2 != 'CCCCC' wildcard), the claim's other lines must
include one of them — not just any non-addon code — since an add-on billed
alongside an unrelated primary procedure denies just as surely as one billed
with no primary at all.
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
        other_codes = {ln.code for ln in cpt_lines}

        for ln in cpt_lines:
            if ln.code not in addon_codes:
                continue
            others = other_codes - {ln.code}
            aoc_edits = self.store.ncci_aoc_edits(ln.code)
            valid_primaries = {e.get("code2") for e in aoc_edits}

            if valid_primaries and "CCCCC" not in valid_primaries:
                if not (others & valid_primaries):
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[ln.code], denial_risk=DenialRisk.HIGH,
                        reason=f"Add-on code {ln.code} is billed without its required primary "
                               f"procedure ({', '.join(sorted(valid_primaries))}) on the claim.",
                        recommendation="Add the specific required primary procedure code, or remove the add-on.",
                        source_rule="CMS NCCI Add-On Code (AOC) edit table",
                        auto_fixable=False,
                    ))
            elif not others:
                # No specific AOC pairing data for this code (or CMS wildcard) —
                # fall back to the looser "some other procedure is present" check.
                findings.append(self.finding(
                    status=Status.FAIL, codes=[ln.code], denial_risk=DenialRisk.HIGH,
                    reason=f"Add-on code {ln.code} is billed with no primary procedure on the claim.",
                    recommendation="Add the appropriate primary procedure code, or remove the add-on.",
                    source_rule="CPT Appendix D add-on designation",
                    auto_fixable=False,
                ))
        return findings
