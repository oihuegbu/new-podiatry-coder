"""Filter #3 — Medically Unlikely Edits (MUE) with MAI awareness.

The key nuance Kachi called out: MUEs are NOT all the same.
  * MAI 1 — line-level edit. Units over the cap on a single line may be bypassed
            by splitting onto separate lines with appropriate modifiers.
  * MAI 2 — date-of-service edit, ABSOLUTE. Never payable above the cap. Hard wall.
  * MAI 3 — date-of-service edit, clinical benchmark. Can exceed with strong
            documentation, but heavily scrutinized → review, not auto-pass.
"""
from __future__ import annotations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status

class MUEAgent(ComplianceAgent):
    filter_id = "MUE_MAI"
    filter_name = "Medically Unlikely Edits (MAI-aware)"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        dos = claim.date_of_service
        if not dos or not self.store.mue_data_available(dos):
            return [self.finding(
                status=Status.UNKNOWN, denial_risk=DenialRisk.HIGH,
                reason="The loaded MUE release does not authoritatively cover the "
                       "claim date of service.",
                recommendation="Load the CMS Practitioner MUE release covering the DOS "
                               "before autonomous release.",
                source_rule="CMS NCCI Practitioner MUE quarterly release",
            )]
        line_bypass_modifiers = (
            self.store.modifier_codes_for_role("ncci_procedure_separation")
            | self.store.modifier_codes_for_role("repeat_service")
            | self.store.modifier_codes_for_role("laboratory_repeat")
            | self.store.anatomic_modifiers()
        )

        for line in claim.lines:
            mue = self.store.mue(line.code, dos)
            if not mue:
                findings.append(self.finding(
                    status=Status.UNKNOWN, codes=[line.code],
                    denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: no authoritative MUE row proves the "
                           "applicable unit edit for this claim type and DOS.",
                    recommendation="Verify the code against the complete CMS MUE release "
                                   "or route the claim for human review.",
                    source_rule="CMS NCCI Practitioner MUE quarterly release",
                ))
                continue
            cap = mue["mue_value"]
            mai = mue.get("mai") or ""
            if cap == 0:
                # MUE of 0 = CMS pays ZERO units of this code on this claim
                # type, ever — the practitioner MUE file assigns 0 to items
                # payable only through other channels (e.g. DMEPOS supplies
                # on a professional claim). The old `cap <= 0: continue`
                # skipped these entirely, so the scrub verdict came back
                # CLEAN and overrode the validator's own MUE-0 ERROR
                # (observed live: A4570 lines shipping AUTO).
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                    reason=f"{line.code}: MUE is 0 on this claim type — CMS pays zero "
                           f"units of this code on a professional claim, at any quantity.",
                    recommendation="Remove the line. If the supply/service is real, bill it "
                                   "through its payable channel (e.g. DMEPOS/DME MAC) or the "
                                   "payable equivalent code the documentation supports.",
                    source_rule=f"NCCI MUE {line.code_system} cap=0 MAI={mai or '?'} (eff on DOS)",
                ))
                continue
            if line.units <= cap:
                continue  # within the cap — fine

            src = f"NCCI MUE {line.code_system} cap={cap} MAI={mai or '?'} (eff on DOS)"
            over = f"{line.units} units billed, MUE cap is {cap}"

            if mai == "2":
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                    reason=f"{line.code}: {over}. MAI 2 is an absolute date-of-service edit — "
                           f"units above the cap are never payable.",
                    recommendation=f"Reduce units to {cap}. This cannot be bypassed with modifiers.",
                    source_rule=src, auto_fixable=True,
                    suggested_fix={"code": line.code, "set_units": cap},
                ))
            elif mai == "1":
                has_bypass = bool(set(line.modifiers) & line_bypass_modifiers)
                if has_bypass:
                    findings.append(self.finding(
                        status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: {over}. MAI 1 (line-level) with a separation modifier "
                               f"present — ensure each line is distinctly documented.",
                        recommendation="Confirm separate anatomic sites / sessions support the split lines.",
                        source_rule=src,
                    ))
                else:
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: {over}. MAI 1 is a line-level edit; the excess will deny "
                               f"unless split onto separate lines with an appropriate modifier.",
                        recommendation=(
                            "Split into separate lines at or below the cap with an applicable "
                            "authoritative separation/repeat/anatomic modifier "
                            f"({ '/'.join(sorted(line_bypass_modifiers)) }) only if the units are "
                            "genuinely distinct, or reduce units."
                        ),
                        source_rule=src,
                    ))
            elif mai == "3":
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: {over}. MAI 3 (clinical benchmark) may be exceeded only with "
                           f"strong supporting documentation — requires human review.",
                    recommendation="Route for review; attach documentation justifying medical necessity "
                                   "for the units above the benchmark.",
                    source_rule=src,
                ))
            else:
                # MUE present but MAI unknown — be conservative, flag for review
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: {over}. MUE exceeded (MAI unspecified).",
                    recommendation=f"Reduce to {cap} or document medical necessity.",
                    source_rule=src,
                ))
        return findings
