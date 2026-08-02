"""Filter #10 — Prior authorization / precertification.

Even a perfectly compliant code denies if the payer required prior auth and none
is on file. This is payer/plan/code specific. The agent reads required-PA codes
from the `prior_auth_required` table (data-driven, payer-aware). A bare
authorization number is only a lead: autonomous release still requires a
verified authorization whose plan, code, units, and DOS scope match the line.

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
        policy = self.store.prior_auth_policy_status(
            payer_id, plan=claim.payer.plan, dos=claim.date_of_service)
        if not policy.get("available"):
            reason = policy.get("reason") or "corpus_unavailable"
            return [self.finding(
                status=Status.UNKNOWN, denial_risk=DenialRisk.HIGH,
                reason=f"A complete, effective-dated prior-authorization corpus is not "
                       f"available for {payer_label} ({reason}).",
                recommendation="Load a complete payer/plan PA corpus covering the DOS or "
                               "route the claim for human review.",
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
                    status=Status.UNKNOWN, codes=[line.code], denial_risk=DenialRisk.HIGH,
                    reason=f"{line.code} requires prior authorization ({payer_label}, matched via "
                           f"{basis}); an authorization number is present but has not been "
                           f"verified against this code, units, plan, and date of service.",
                    recommendation="Verify the authorization through an X12 278 or payer API "
                                   "and persist its code/unit/date scope before release.",
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
