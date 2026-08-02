"""Filter #10 — Prior authorization / precertification.

Even a perfectly compliant code denies if the payer required prior auth and none
is on file. This is payer/plan/code specific. The agent reads required-PA codes
from the `prior_auth_required` table (data-driven, payer-aware) and checks
whether an authorization number is present on the claim.

The PA-required code/category list comes from each payer's published Required
Prior Authorization list — see data/codes/prior_auth_<payer>.json (Medicare
DMEPOS is exact-code, Tricare is category-based via HCPCS prefix). Payers with
no file loaded produce UNKNOWN and block autonomous release.

Payer identity comes from Claim.payer.payer_id (canonical key from
payers.json, resolved from the note's free-text insurance field by
payer_registry.parse_insurance_text — see engine.build_claim). Unrecognized
payers have payer_id=None and fail closed, never defaulting to Medicare.
"""
from __future__ import annotations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status


class PriorAuthAgent(ComplianceAgent):
    filter_id = "PRIOR_AUTH"
    filter_name = "Prior authorization"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        payer_id = claim.payer.payer_id
        payer_label = claim.payer.name or "the payer"
        if not payer_id:
            return [self.finding(
                status=Status.UNKNOWN, denial_risk=DenialRisk.HIGH,
                reason="Payer identity is unresolved, so prior-authorization policy "
                       "cannot be selected.",
                recommendation="Resolve the canonical payer and plan before release.",
                source_rule="payer-specific prior-authorization policy",
            )]
        if not self.store.prior_auth_policy_available(payer_id):
            return [self.finding(
                status=Status.UNKNOWN, denial_risk=DenialRisk.HIGH,
                reason=f"No authoritative prior-authorization corpus is loaded for "
                       f"{payer_label}.",
                recommendation="Load and version the payer's current PA policy or route "
                               "the claim for human review.",
                source_rule=f"{payer_label} prior-authorization policy availability",
            )]
        auth_on_file = bool(claim.subscriber.authorization_number)

        for line in claim.lines:
            rule = self.store.prior_auth_required(line.code, payer_id)
            if not rule:
                continue  # this code does not require PA for this payer
            basis = f"code {rule['code']}" if rule.get("code") else f"category '{rule.get('category')}'"
            if auth_on_file:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.LOW,
                    reason=f"{line.code} requires prior authorization ({payer_label}, matched via "
                           f"{basis}); an auth number is on file — confirm it covers this code and date.",
                    recommendation="Verify the authorization number matches the code/units/date.",
                    source_rule=f"{payer_label} prior-authorization list",
                ))
            else:
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                    reason=f"{line.code} requires prior authorization for {payer_label} (matched via "
                           f"{basis}) and no authorization number is on file — it will deny.",
                    recommendation="Obtain prior authorization (X12 278 / payer FHIR PA API) before "
                                   "submission, or do not bill this code.",
                    source_rule=f"{payer_label} prior-authorization list" + (f" — {rule.get('note')}" if rule.get("note") else ""),
                ))
        return findings
