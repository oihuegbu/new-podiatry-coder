"""Filter #13 — Billability (is this code separately payable at all).

Distinct from every other filter here: NCCI/MUE/global-period/add-on all
assume the code itself is a valid, separately-billable service and check
*how* it interacts with other codes or units. This filter checks the code's
own Medicare billing status — global_periods.json's self-documented status
field (A/B/C/I/N/R/T/X) — before any of that matters. A code with status
B (bundled/not separately payable), N (noncovered), or X (statutory
exclusion) denies regardless of modifiers, units, or NCCI pairing; status I
(not valid for Medicare) only applies when the claim's payer actually is
Medicare, since Medicaid/commercial payers often cover items Medicare
doesn't.

Found via `verify_notes.py` audit: A6550 and Q4051 (both status X) shipped
as clean claims because no filter checked this dimension at all.
"""
from __future__ import annotations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status


class BillabilityAgent(ComplianceAgent):
    filter_id = "BILLABILITY"
    filter_name = "Billability (Medicare status indicator)"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        for line in claim.lines:
            reason = self.store.not_separately_billable_reason(line.code)
            pfs_advisory = self.store.pfs_exclusion_advisory(line.code)
            if reason:
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                    reason=f"{line.code}: {reason} — not separately billable as a standalone line item.",
                    recommendation="Remove this code, or verify it's correctly bundled into another billed service.",
                    source_rule="global_periods.json billing status (CMS PFS)",
                ))
            elif pfs_advisory:
                # status 'X' = PFS statutory exclusion — payable under another
                # fee schedule (CLFS/DMEPOS), so route for review rather than
                # denying: the question is who bills it, not whether it's real.
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: {pfs_advisory}.",
                    recommendation="Confirm the performing entity bills this under its own fee schedule; "
                                   "remove it from this claim if billed by a reference lab/DME supplier.",
                    source_rule="global_periods.json billing status (CMS PFS)",
                ))
            # HCPCS coverage code (I/M/S) — the HCPCS file's own Medicare
            # non-coverage signal. Most HCPCS II supplies/orthotics never
            # appear on the PFS, so the billing-status checks above can't
            # see them; this is the only per-code coverage answer for them.
            # Payer-scoped like status 'I' below: it's a *Medicare* coverage
            # verdict, not a statement about commercial/Medicaid payers.
            elif (claim.payer.follows_medicare_coverage
                  and line.code_system == "HCPCS"
                  and (cov_reason := self.store.hcpcs_noncoverage_reason(line.code))):
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                    reason=f"{line.code}: HCPCS {cov_reason}, and this claim's payer follows "
                           f"Medicare coverage rules ({claim.payer.kind}).",
                    recommendation="Remove the line, bill the patient with a signed ABN, or "
                                   "verify a benefit category exists for this item.",
                    source_rule="HCPCS coverage code (CMS alpha-numeric HCPCS file)",
                ))
            # follows_medicare_coverage, not is_medicare: a code CMS marks
            # "not valid for Medicare" is equally invalid under Medicare
            # Advantage, whose coverage floor is Original Medicare's
            # (42 CFR 422.101) — only commercial/Medicaid payers may cover it.
            elif claim.payer.follows_medicare_coverage and self.store.billing_status(line.code) == "I":
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                    reason=f"{line.code}: status 'I' (not valid for Medicare) but claim payer "
                           f"follows Medicare coverage rules ({claim.payer.kind}).",
                    recommendation="Verify payer/coverage — this code is not valid for Medicare billing.",
                    source_rule="global_periods.json billing status (CMS PFS)",
                ))
        return findings
