"""Filter #9 — Place of Service & provider eligibility.

Phase-2 scope: validate the Place of Service code against the CMS POS set and
expose its facility/non-facility designation (drives the site-of-service
differential). An invalid POS code is a hard denial.

Per-code "not payable in this POS" edits and provider specialty/credentialing
checks require PFS payability data + a provider-enrollment source, which are not
yet ingested — those checks are added when those sources land (no logic rewrite,
the POS facility flag is already available here).
"""
from __future__ import annotations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status


class POSEligibilityAgent(ComplianceAgent):
    filter_id = "POS_ELIGIBILITY"
    filter_name = "Place of Service & provider eligibility"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        # POS may live at claim level or per line; check whatever is populated.
        pos_values = {claim.place_of_service} | {ln.place_of_service for ln in claim.lines}
        pos_values = {p for p in pos_values if p}

        for pos in pos_values:
            info = self.store.pos_info(pos)
            if info is None:
                findings.append(self.finding(
                    status=Status.FAIL, codes=[], denial_risk=DenialRisk.HIGH,
                    reason=f"Place of Service '{pos}' is not a valid POS code.",
                    recommendation="Use a valid 2-digit POS code from the CMS POS set.",
                    source_rule="CMS Place of Service code set",
                ))

        # Telehealth POS ⇄ telehealth modifier consistency. Both sides come
        # from the reference data itself: POS codes whose CMS name says
        # "telehealth", and modifiers whose AMA name says "telemedicine"
        # (95/93/GT/...). A telehealth-POS claim with no telehealth modifier
        # on any professional service line denies or misprices with most
        # payers; the reverse (modifier without telehealth POS) signals a
        # POS extraction error.
        tele_mods = self.store.telehealth_modifiers()
        if tele_mods:
            claim_pos_tele = any(self.store.pos_is_telehealth(p) for p in pos_values)
            for line in claim.lines:
                line_pos = line.place_of_service or claim.place_of_service
                line_tele_pos = claim_pos_tele if not line.place_of_service else \
                    self.store.pos_is_telehealth(line.place_of_service)
                line_tele_mods = {m.strip().upper() for m in line.modifiers} & tele_mods
                if line_tele_pos and not line_tele_mods:
                    findings.append(self.finding(
                        status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code} billed with telehealth POS {line_pos} but carries no "
                               f"telehealth modifier ({'/'.join(sorted(tele_mods))}) — most payers "
                               f"require the modifier to adjudicate telehealth services.",
                        recommendation="Append 95 (audio+video) or 93 (audio-only) per how the "
                                       "encounter was actually conducted, if documented.",
                        source_rule="CMS POS set (telehealth POS) + AMA telemedicine modifiers",
                    ))
                elif line_tele_mods and pos_values and not line_tele_pos:
                    findings.append(self.finding(
                        status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code} carries telehealth modifier "
                               f"{'/'.join(sorted(line_tele_mods))} but the POS "
                               f"({line_pos}) is not a telehealth POS — the modifier and POS "
                               f"contradict each other.",
                        recommendation="Verify how the encounter was conducted; fix the POS "
                                       "(02/10) or remove the telehealth modifier.",
                        source_rule="CMS POS set (telehealth POS) + AMA telemedicine modifiers",
                    ))
        # Missing POS is intentionally not flagged here: when the extraction does
        # not capture POS it is a capture gap, not a coding error. POS-payability
        # enforcement is added with the PFS payability source.
        return findings
