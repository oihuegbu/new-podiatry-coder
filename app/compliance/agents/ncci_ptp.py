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

# NCCI-associated separation modifiers (CMS prefers the X{EPSU} subset over 59)
_SEPARATION_MODIFIERS = {"59", "XE", "XS", "XP", "XU"}
# E/M ↔ procedure pairs are unbundled with 25 (or 57), not 59
_EM_SEPARATION_MODIFIERS = {"25", "57"}
_EM_SECTION = range(99202, 99500)


def _is_em(code: str) -> bool:
    return code.isdigit() and int(code) in _EM_SECTION


class NCCIPTPAgent(ComplianceAgent):
    filter_id = "NCCI_PTP"
    filter_name = "NCCI procedure-to-procedure edits"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        dos = claim.date_of_service
        cpt_lines = [ln for ln in claim.lines if ln.code_system == "CPT"]

        for a, b in combinations(cpt_lines, 2):
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
            # An E/M ↔ procedure pair is separated by 25/57; all others by 59/X{EPSU}.
            sep_modifiers = set(_SEPARATION_MODIFIERS)
            if _is_em(col1_line.code) or _is_em(col2_line.code):
                sep_modifiers |= _EM_SEPARATION_MODIFIERS
            has_sep = bool(mods_present & sep_modifiers)
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
                    is_em_pair = _is_em(col1_line.code) or _is_em(col2_line.code)
                    fix = ("Add modifier 25 (or 57) to the E/M if it is significant and separately "
                           "identifiable") if is_em_pair else \
                          ("Add 59 or (preferred) XE/XS/XP/XU to the component code if the services "
                           "are genuinely distinct")
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
