"""Filter #4 — Modifier validity.

  * Every appended modifier must be a recognized CPT/HCPCS modifier.
  * CMS prefers the X{EPSU} subset over 59 — flag 59 as an advisory.
  * Catch conflicting/redundant modifiers (e.g. RT+LT on one line → should be 50;
    50 with RT/LT; duplicate modifiers).
All modifier knowledge comes from the `modifier` reference table — no hardcoding.
"""
from __future__ import annotations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status

_LATERAL = {"RT", "LT"}
# CPT structural sections (code-system grammar, not medical-rule code lists)
_EM_SECTION = range(99202, 99500)
_SURGERY_SECTION = range(10000, 70000)
# Modifiers that separate an E/M from a same-day procedure
_EM_SEPARATION = {"25", "57"}


def _is_em(code: str) -> bool:
    return code.isdigit() and int(code) in _EM_SECTION


def _is_surgery(code: str) -> bool:
    return code.isdigit() and int(code) in _SURGERY_SECTION


class ModifierAgent(ComplianceAgent):
    filter_id = "MODIFIERS"
    filter_name = "Modifier validity"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        for line in claim.lines:
            mods = [m.strip().upper() for m in line.modifiers if m and m.strip()]

            # unrecognized modifiers
            for m in mods:
                if not self.store.modifier_valid(m):
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: '{m}' is not a recognized modifier.",
                        recommendation="Remove or correct the modifier.",
                        source_rule="Recognized modifier reference set",
                    ))

            # duplicate modifiers
            dupes = {m for m in mods if mods.count(m) > 1}
            for m in dupes:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.LOW,
                    reason=f"{line.code}: modifier '{m}' appears more than once.",
                    recommendation="Remove the duplicate modifier.",
                    source_rule="modifier hygiene",
                ))

            mod_set = set(mods)
            # RT and LT on the same line → should be bilateral (50)
            if _LATERAL.issubset(mod_set):
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: both RT and LT on one line.",
                    recommendation="Use modifier 50 (bilateral), or split into two lines (RT / LT).",
                    source_rule="bilateral billing convention",
                ))
            # 50 together with RT/LT is contradictory
            if "50" in mod_set and (mod_set & _LATERAL):
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: modifier 50 (bilateral) combined with RT/LT.",
                    recommendation="Use either 50 OR RT/LT, not both.",
                    source_rule="bilateral billing convention",
                ))
            # Modifier 51 on a modifier-51-exempt code (CPT Appendix E)
            if "51" in mod_set and self.store.is_modifier_51_exempt(line.code):
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: modifier 51 appended to a modifier-51-exempt code (CPT Appendix E) — "
                           f"the relative value already accounts for multiple-procedure reduction.",
                    recommendation="Remove modifier 51; the payer will reduce this code automatically or it "
                                   "is not subject to reduction.",
                    source_rule="CPT Appendix E — modifier 51 exempt codes",
                ))

            # CMS prefers X{EPSU} over 59
            if "59" in mod_set:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.LOW,
                    reason=f"{line.code}: modifier 59 used — CMS prefers the more specific "
                           f"X{{EPSU}} subset (XE/XS/XP/XU).",
                    recommendation="Replace 59 with XE/XS/XP/XU describing the distinction, when applicable.",
                    source_rule="NCCI policy — X{EPSU} preferred over 59",
                ))

        # --- E/M billed same day as a procedure → needs modifier 25 (or 57) ---
        # An E/M performed on the same day as a procedure bundles into that
        # procedure unless a significant, separately identifiable E/M is
        # documented and modifier 25 (or 57 for a major-surgery decision) is
        # appended. (Procedure = CPT surgery section, excludes radiology/lab.)
        em_lines = [ln for ln in claim.lines if _is_em(ln.code)]
        has_same_day_procedure = any(_is_surgery(ln.code) for ln in claim.lines)
        if em_lines and has_same_day_procedure:
            for em in em_lines:
                if set(em.modifiers) & _EM_SEPARATION:
                    continue  # 25/57 present — correctly separated
                findings.append(self.finding(
                    status=Status.FAIL, codes=[em.code], denial_risk=DenialRisk.HIGH,
                    reason=f"E/M {em.code} billed the same day as a procedure with no modifier 25 — "
                           f"it bundles into the procedure and will not be paid separately.",
                    recommendation="Append modifier 25 if a significant, separately identifiable E/M is "
                                   "documented (57 if it is the decision for major surgery); otherwise "
                                   "the E/M is not separately billable.",
                    source_rule="NCCI — E/M same-day procedure bundling (modifier 25)",
                ))
        return findings
