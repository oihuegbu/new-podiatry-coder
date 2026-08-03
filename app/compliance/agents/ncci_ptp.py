"""Filter #2 — NCCI Procedure-to-Procedure (PTP) edits.

Column1/Column2 pairs that shouldn't be billed together. The modifier indicator
governs whether the edit can be overridden:
  * 0 — never unbundle (no modifier can override) → hard edit, suppress Column 2
  * 1 — a modifier may bypass IF the services are genuinely distinct
  * 9 — edit deleted / not applicable
The engine must know the difference, not just that a conflict exists.
"""
from __future__ import annotations

from itertools import combinations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status


class NCCIPTPAgent(ComplianceAgent):
    filter_id = "NCCI_PTP"
    filter_name = "NCCI procedure-to-procedure edits"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        dos = claim.date_of_service
        lines = [ln for ln in claim.lines if ln.code_system in {"CPT", "HCPCS"}]
        if len(lines) < 2:
            return findings

        if dos is None:
            return [self.finding(
                status=Status.UNKNOWN,
                denial_risk=DenialRisk.HIGH,
                reason="NCCI PTP edits cannot be evaluated without a date of service.",
                recommendation="Verify the date of service and re-run the complete scrub.",
                source_rule="CMS NCCI PTP effective-date control",
                clause="data_availability",
            )]
        if not self.store.ncci_data_available(dos):
            return [self.finding(
                status=Status.UNKNOWN,
                denial_risk=DenialRisk.HIGH,
                reason=(f"No local NCCI PTP release covers date of service "
                        f"{dos.isoformat()}; absence of a pair cannot be treated as no edit."),
                recommendation="Load and validate the CMS NCCI release covering the DOS, then re-scrub.",
                source_rule="CMS NCCI PTP effective-date control",
                clause="data_availability",
            )]

        for a, b in combinations(lines, 2):
            edit = self.store.ncci_pair(a.code, b.code, dos)
            if not edit:
                continue
            ind = (edit.get("modifier_indicator") or "").strip()
            if ind == "9":
                continue  # edit not applicable

            # Determine which line is Column 2 (the bundled/component code)
            col2_code = edit["col2"]
            col2_line = a if a.code == col2_code else b
            col1_line = b if col2_line is a else a
            pair = f"{col1_line.code}/{col2_line.code}"
            mods_present = set(col1_line.modifiers) | set(col2_line.modifiers)
            # Resolve CMS/AMA modifier roles from the ingested reference
            # names.  No code values or code-family boundaries live here.
            is_em_pair = (self.store.is_em_code(col1_line.code, dos)
                          or self.store.is_em_code(col2_line.code, dos))
            sep_modifiers = self.store.modifier_codes_for_role(
                "ncci_procedure_separation")
            if is_em_pair:
                sep_modifiers |= self.store.modifier_codes_for_role(
                    "ncci_em_separation")
            has_sep = bool(mods_present & sep_modifiers)
            # CMS's NCCI-associated modifiers also include the ANATOMIC set
            # (RT/LT, FA/F1–F9, TA/T1–T9, E1–E4, coronary) — two lines whose
            # anatomic modifiers DIFFER document distinct sites and bypass an
            # indicator-1 edit exactly like 59/X{EPSU}. Both-sides-required
            # and sets-must-differ: RT on both lines asserts the same side
            # and separates nothing (observed live: 28297-RT vs 28285-RT,T6
            # — right bunion vs right 2nd toe — was FAILed as unseparated
            # even though T6 is precisely the CMS-sanctioned way to say
            # 'different toe').
            anatomic = self.store.anatomic_modifiers()
            sites1 = {m.strip().upper() for m in col1_line.modifiers} & anatomic
            sites2 = {m.strip().upper() for m in col2_line.modifiers} & anatomic
            if sites1 and sites2 and sites1 != sites2:
                has_sep = True
            src = f"NCCI PTP {pair} indicator={ind or '?'} (eff on DOS)"

            if ind == "0":
                findings.append(self.finding(
                    status=Status.FAIL, codes=[col1_line.code, col2_line.code],
                    denial_risk=DenialRisk.HIGH,
                    reason=f"NCCI hard edit: {col2_line.code} is a component of {col1_line.code} "
                           f"(indicator 0 — cannot be unbundled by any modifier).",
                    recommendation=f"Remove {col2_line.code}; it is included in {col1_line.code}.",
                    source_rule=src, auto_fixable=True,
                    suggested_fix={"remove_code": col2_line.code},
                ))
            elif ind == "1":
                if has_sep:
                    findings.append(self.finding(
                        status=Status.WARN, codes=[col1_line.code, col2_line.code],
                        denial_risk=DenialRisk.MEDIUM,
                        reason=f"NCCI pair {pair} (indicator 1) — a separation modifier is present; "
                               f"the two services must be documented as distinct.",
                        recommendation="Confirm distinct session/site/encounter justifies the modifier.",
                        source_rule=src,
                    ))
                else:
                    role_codes = sorted(sep_modifiers)
                    role_text = "/".join(role_codes) if role_codes else "the applicable modifier"
                    fix = (f"Add {role_text} to the E/M if its authoritative meaning is supported"
                           if is_em_pair else
                           f"Add {role_text} to the component service only if the services are "
                           "genuinely distinct")
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[col1_line.code, col2_line.code],
                        denial_risk=DenialRisk.MEDIUM,
                        reason=f"NCCI pair {pair} (indicator 1) — billed together with no separation "
                               f"modifier; {col2_line.code} will bundle into {col1_line.code}.",
                        recommendation=f"{fix}; otherwise remove the component code.",
                        source_rule=src,
                    ))
            else:
                findings.append(self.finding(
                    status=Status.WARN, codes=[col1_line.code, col2_line.code],
                    denial_risk=DenialRisk.MEDIUM,
                    reason=f"NCCI pair {pair} present but modifier indicator is unknown.",
                    recommendation="Manually confirm whether this pair may be billed together.",
                    source_rule=src,
                ))
        return findings
