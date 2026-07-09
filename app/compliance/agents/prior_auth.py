"""Filter #10 — Prior authorization / precertification.

Even a perfectly compliant code denies if the payer required prior auth and none
is on file. This is payer/plan/code specific. The agent reads required-PA codes
from the `prior_auth_required` table (data-driven, payer-aware) and checks
whether an authorization number is present on the claim.

The PA-required code list comes from the payer's published Required Prior
Authorization list (Medicare FFS publishes one; commercial payers via their
companion guides). Until that list is loaded the table is empty → no false
positives. Structured FHIR-first (CMS is phasing out X12 278 by 2027).
"""
from __future__ import annotations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status


class PriorAuthAgent(ComplianceAgent):
    filter_id = "PRIOR_AUTH"
    filter_name = "Prior authorization"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        payer = claim.payer.name or "Medicare"
        auth_on_file = bool(claim.subscriber.authorization_number)

        for line in claim.lines:
            rule = self.store.prior_auth_required(line.code, payer)
            if not rule:
                continue  # this code does not require PA for this payer
            if auth_on_file:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.LOW,
                    reason=f"{line.code} requires prior authorization ({payer}); an auth number is on "
                           f"file — confirm it covers this code and date.",
                    recommendation="Verify the authorization number matches the code/units/date.",
                    source_rule=f"{payer} prior-authorization list",
                ))
            else:
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                    reason=f"{line.code} requires prior authorization for {payer} and no authorization "
                           f"number is on file — it will deny.",
                    recommendation="Obtain prior authorization (X12 278 / payer FHIR PA API) before "
                                   "submission, or do not bill this code.",
                    source_rule=f"{payer} prior-authorization list" + (f" — {rule.get('note')}" if rule.get("note") else ""),
                ))
        return findings
