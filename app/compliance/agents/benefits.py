"""Filter #11 — Eligibility & benefit coverage (270/271).

The patient must be eligible on the DOS and the service a covered benefit. This
agent calls the clearinghouse adapter (Stedi today) when the claim carries the
subscriber identifiers needed for a 270 inquiry.

Fail-closed behavior (unverified eligibility never becomes a pass):
  * no member id / payer        → UNKNOWN (blocks autonomous release)
  * clearinghouse not configured → UNKNOWN
  * coverage returned INACTIVE   → FAIL (patient responsibility, not payer-billable)
  * lookup errored               → UNKNOWN
"""
from __future__ import annotations

from app.compliance.adapters.stedi import ClearinghouseAdapter, default_adapter
from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status


class BenefitsAgent(ComplianceAgent):
    filter_id = "ELIGIBILITY"
    filter_name = "Eligibility & benefit coverage"

    def __init__(self, store, adapter: ClearinghouseAdapter | None = None):
        super().__init__(store)
        self.adapter = adapter or default_adapter()

    def check(self, claim: Claim) -> list[Finding]:
        sub = claim.subscriber
        # stedi_trading_partner_id, not payer_id — Stedi's eligibility API needs
        # its own payer ID namespace (e.g. "BCBSF"), not this system's internal
        # payer_id ("bcbs_fl"); they're resolved together in payers.json but are
        # different ID systems. See payer_registry.py.
        if not (sub.member_id and claim.payer.stedi_trading_partner_id):
            return [self.finding(
                status=Status.UNKNOWN, denial_risk=DenialRisk.HIGH,
                reason="Eligibility could not be checked because the member or "
                       "clearinghouse payer identifier is absent.",
                recommendation="Resolve payer identity and subscriber identifiers, then "
                               "run a 270/271 eligibility inquiry.",
                source_rule="eligibility (270/271)",
            )]

        if not self.adapter.is_configured():
            return [self.finding(
                status=Status.UNKNOWN, denial_risk=DenialRisk.HIGH,
                reason="Clearinghouse not configured — eligibility could not be verified.",
                recommendation="Verify patient eligibility/benefits before submission.",
                source_rule="eligibility (270/271)",
            )]

        if not claim.provider.npi:
            return [self.finding(
                status=Status.UNKNOWN, denial_risk=DenialRisk.HIGH,
                reason="Eligibility could not be checked because the billing/rendering "
                       "provider NPI is absent.",
                recommendation="Resolve the claim provider NPI before sending the 270 inquiry.",
                source_rule="eligibility (270/271) provider identity",
            )]

        if claim.date_of_service is None:
            return [self.finding(
                status=Status.UNKNOWN, denial_risk=DenialRisk.HIGH,
                reason="Eligibility cannot be tied to the claim because the date of "
                       "service is absent.",
                recommendation="Resolve the DOS and rerun procedure-specific eligibility.",
                source_rule="eligibility (270/271) date-of-service requirement",
            )]

        dob = (sub.date_of_birth or "")
        findings: list[Finding] = []
        if not claim.lines:
            return [self.finding(
                status=Status.UNKNOWN, denial_risk=DenialRisk.HIGH,
                reason="No billable service line is available for a service-specific "
                       "benefit inquiry.",
                recommendation="Build the service lines before checking benefits.",
                source_rule="eligibility (270/271) service-specific inquiry",
            )]

        for line in claim.lines:
            qualifier = "HC" if str(line.code_system).upper() == "HCPCS" else "CJ"
            res = self.adapter.check_eligibility(
                payer_id=claim.payer.stedi_trading_partner_id,
                member_id=sub.member_id, first_name=sub.first_name,
                last_name=sub.last_name, date_of_birth=dob,
                npi=claim.provider.npi, date_of_service=claim.date_of_service,
                procedure_code=line.code,
                product_or_service_id_qualifier=qualifier,
            )
            if res.active is False or res.service_coverage_confirmed is False:
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code],
                    denial_risk=DenialRisk.HIGH,
                    reason=f"Eligibility/benefits response did not confirm active, "
                           f"covered benefits for {line.code} on the date of service.",
                    recommendation="Confirm active coverage and the service benefit with "
                                   "the payer before submission.",
                    source_rule="procedure-specific 271 eligibility response",
                ))
                continue
            if res.active is not True or res.service_coverage_confirmed is not True:
                msg = "; ".join(res.errors) or "procedure benefit not confirmed"
                findings.append(self.finding(
                    status=Status.UNKNOWN, codes=[line.code],
                    denial_risk=DenialRisk.HIGH,
                    reason=f"Eligibility for {line.code} on the DOS could not be "
                           f"confirmed ({msg}).",
                    recommendation="Obtain a procedure-specific 271 response before submission.",
                    source_rule="procedure-specific 271 eligibility response",
                ))
        return findings
