"""Filter #11 — Eligibility & benefit coverage (270/271).

The patient must be eligible on the DOS and the service a covered benefit. This
agent calls the clearinghouse adapter (Stedi today) when the claim carries the
subscriber identifiers needed for a 270 inquiry.

Graceful degradation (so the gate is never blocked by infrastructure):
  * no member id / payer        → cannot check at coding stage → no finding
  * clearinghouse not configured → advisory WARN (verify eligibility manually)
  * coverage returned INACTIVE   → FAIL (patient responsibility, not payer-billable)
  * lookup errored               → advisory WARN
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
        if not (sub.member_id and claim.payer.payer_id):
            return []  # not enough identifiers to run a 270 at this stage

        if not self.adapter.is_configured():
            return [self.finding(
                status=Status.WARN, denial_risk=DenialRisk.MEDIUM,
                reason="Clearinghouse not configured — eligibility could not be verified.",
                recommendation="Verify patient eligibility/benefits before submission.",
                source_rule="eligibility (270/271)",
            )]

        dob = (sub.date_of_birth or "")
        res = self.adapter.check_eligibility(
            payer_id=claim.payer.payer_id, member_id=sub.member_id,
            first_name=sub.first_name, last_name=sub.last_name,
            date_of_birth=dob, npi=claim.provider.npi,
        )
        if res.active is False:
            return [self.finding(
                status=Status.FAIL, denial_risk=DenialRisk.HIGH,
                reason="Eligibility check returned INACTIVE coverage for the date of service.",
                recommendation="Confirm active coverage / correct payer; may be patient responsibility.",
                source_rule="271 eligibility response",
            )]
        if res.active is None:
            msg = "; ".join(res.errors) or "indeterminate response"
            return [self.finding(
                status=Status.WARN, denial_risk=DenialRisk.MEDIUM,
                reason=f"Eligibility could not be confirmed ({msg}).",
                recommendation="Manually verify eligibility before submission.",
                source_rule="271 eligibility response",
            )]
        return []  # active coverage confirmed
