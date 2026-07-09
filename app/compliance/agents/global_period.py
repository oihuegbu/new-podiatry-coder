"""Filter #6 — Global surgical period.

Each surgical CPT carries a global indicator (000/010/090/MMM/XXX/YYY/ZZZ).
Care related to the surgery billed within that window is bundled into the
surgical package unless an appropriate modifier applies:
  * 24 — unrelated E/M during a postoperative period
  * 58 — staged/planned related procedure
  * 78 — return to OR for a complication (related)
  * 79 — unrelated procedure during the postoperative period

Phase-1 scope: detect a separately-billed E/M during a PRIOR surgery's global
window with no qualifying modifier (it should be 99024, no-charge).

NOTE on "no hardcoding": the E/M section boundary (99202–99499) and the global-
period modifier meanings are *structural facts of the CPT/HCPCS grammar*, not
medical-rule code lists. The actual global-day values come 100% from the data
store (PFS GLOB DAYS).
"""
from __future__ import annotations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status

# CPT structural section boundaries (AMA code-system grammar, not a medical list)
_EM_SECTION = range(99202, 99500)
_POSTOP_NOCHARGE = "99024"
# Modifiers that legitimately allow billing within a global period
_GLOBAL_BYPASS_MODIFIERS = {"24", "25", "57", "58", "78", "79"}


def _is_em(code: str) -> bool:
    return code.isdigit() and int(code) in _EM_SECTION


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
        for line in claim.lines:
            if not _is_em(line.code):
                continue
            if set(line.modifiers) & _GLOBAL_BYPASS_MODIFIERS:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"E/M {line.code} billed on post-op day {days_post_op} of {prior_desc} "
                           f"(CPT {prior_cpt}, {gdays}-day global) with a global-period modifier — "
                           f"ensure it is documented as unrelated/staged.",
                    recommendation="Verify the modifier (24/58/78/79) is supported by the note.",
                    source_rule=f"PFS GLOB DAYS={glob} for {prior_cpt}",
                ))
            else:
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                    reason=f"E/M {line.code} billed on post-op day {days_post_op} of {prior_desc} "
                           f"(CPT {prior_cpt}, {gdays}-day global period) with no qualifying modifier — "
                           f"this visit is included in the surgical package.",
                    recommendation=f"Use {_POSTOP_NOCHARGE} (post-op follow-up, no charge), or add "
                                   f"modifier 24 if the E/M is unrelated and documented.",
                    source_rule=f"PFS GLOB DAYS={glob} for {prior_cpt}",
                    auto_fixable=True,
                    suggested_fix={"replace_code": line.code, "with_code": _POSTOP_NOCHARGE},
                ))
        return findings
