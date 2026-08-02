"""Filter #6 — Global surgical period.

Each surgical service carries a CMS PFS global indicator.
Care related to the surgery billed within that window is bundled into the
surgical package unless an appropriate modifier applies:
Scope: within a PRIOR surgery's global window, detect (a) a separately-billed
E/M with no qualifying modifier, and (b) a separately-billed procedure with
no applicable postoperative modifier.  Both service and modifier roles are
resolved from the authoritative CPT/HCPCS/PFS data.
"""
from __future__ import annotations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status

def _global_days_int(glob: str | None) -> int | None:
    """Map a GLOB DAYS indicator to a number of postoperative days, or None."""
    if glob is None:
        return None
    g = str(glob).strip().upper()
    if g in ("000", "0"):
        return 0
    if g in ("010", "10"):
        return 10
    if g in ("090", "90"):
        return 90
    if g.isdigit():
        return int(g)
    return None  # XXX/MMM/YYY/ZZZ — concept doesn't apply / contractor-priced


class GlobalPeriodAgent(ComplianceAgent):
    filter_id = "GLOBAL_PERIOD"
    filter_name = "Global surgical period"

    def check(self, claim: Claim) -> list[Finding]:
        info = claim.prior_surgery_info or {}
        if not info.get("is_post_op_visit"):
            return []

        prior_cpt = str(info.get("prior_surgery_cpt", "") or "").strip()
        days_post_op = info.get("days_post_op")
        prior_desc = info.get("prior_surgery_description", "prior surgery")

        if not prior_cpt or days_post_op is None:
            return [self.finding(
                status=Status.WARN, denial_risk=DenialRisk.MEDIUM,
                reason="Post-operative visit detected but prior CPT or days-post-op is unknown — "
                       "cannot confirm global-period status.",
                recommendation="Confirm the prior surgery CPT and date before submission.",
                source_rule="encounter context",
            )]

        glob = self.store.global_period(prior_cpt, claim.date_of_service)
        gdays = _global_days_int(glob)
        if not gdays or days_post_op > gdays:
            return []  # outside any global window — separately billable

        findings: list[Finding] = []
        em_bypass_modifiers = self.store.modifier_codes_for_role("postoperative_em")
        procedure_bypass_modifiers = self.store.modifier_codes_for_role(
            "postoperative_procedure")
        no_charge_followup = self.store.postoperative_followup_code(claim.date_of_service)
        for line in claim.lines:
            if no_charge_followup and line.code == no_charge_followup:
                continue
            if not self.store.is_em_code(line.code, claim.date_of_service):
                # PROCEDURES during the postop window are bundled too — the
                # global surgical package covers related return procedures,
                # not just visits. A procedure line needs 58 (staged/planned),
                # 78 (return to OR for complication), or 79 (unrelated) to be
                # separately payable; 24 is E/M-only. Only lines the PFS data
                # itself marks as procedures (000/010/090 global) are held to
                # this — XXX/ZZZ diagnostic/add-on codes aren't part of the
                # surgical-package concept.
                line_glob = self.store.global_period(line.code, claim.date_of_service)
                if _global_days_int(line_glob) is None:
                    continue
                proc_bypass = set(line.modifiers) & procedure_bypass_modifiers
                if proc_bypass:
                    findings.append(self.finding(
                        status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"Procedure {line.code} billed on post-op day {days_post_op} of "
                               f"{prior_desc} (CPT {prior_cpt}, {gdays}-day global) with modifier "
                               f"{'/'.join(sorted(proc_bypass))} — ensure the note documents it as "
                               "supported by its authoritative postoperative meaning.",
                        recommendation="Verify the modifier is supported by the documentation.",
                        source_rule=f"PFS GLOB DAYS={glob} for {prior_cpt}",
                    ))
                else:
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                        reason=f"Procedure {line.code} billed on post-op day {days_post_op} of "
                               f"{prior_desc} (CPT {prior_cpt}, {gdays}-day global period) with no "
                               f"qualifying modifier — related procedures within the global window "
                               f"are included in the surgical package.",
                        recommendation=(
                            f"Use an applicable postoperative modifier "
                            f"({ '/'.join(sorted(procedure_bypass_modifiers)) }) only when its "
                            "authoritative meaning is documented; otherwise do not bill separately."
                            if procedure_bypass_modifiers else
                            "Authoritative postoperative modifier roles are unavailable; do not "
                            "bill separately until the reference data is restored."
                        ),
                        source_rule=f"PFS GLOB DAYS={glob} for {prior_cpt}",
                    ))
                continue
            if set(line.modifiers) & em_bypass_modifiers:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"E/M {line.code} billed on post-op day {days_post_op} of {prior_desc} "
                           f"(CPT {prior_cpt}, {gdays}-day global) with a global-period modifier — "
                           f"ensure it is documented as unrelated/staged.",
                    recommendation="Verify the modifier's authoritative meaning is supported by the note.",
                    source_rule=f"PFS GLOB DAYS={glob} for {prior_cpt}",
                ))
            else:
                recommendation = (
                    f"Use {no_charge_followup} when the documented service matches its CPT "
                    "descriptor, or use an authoritative unrelated-postoperative E/M modifier "
                    f"({ '/'.join(sorted(em_bypass_modifiers)) }) when supported."
                    if no_charge_followup and em_bypass_modifiers else
                    "Do not bill separately until the authoritative postoperative service and "
                    "modifier references are available and documentation supports them."
                )
                finding_kwargs = {}
                if no_charge_followup:
                    finding_kwargs = {
                        "auto_fixable": True,
                        "suggested_fix": {"replace_code": line.code,
                                          "with_code": no_charge_followup},
                    }
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                    reason=f"E/M {line.code} billed on post-op day {days_post_op} of {prior_desc} "
                           f"(CPT {prior_cpt}, {gdays}-day global period) with no qualifying modifier — "
                           f"this visit is included in the surgical package.",
                    recommendation=recommendation,
                    source_rule=f"PFS GLOB DAYS={glob} for {prior_cpt}",
                    **finding_kwargs,
                ))
        return findings
