"""Filter #12 — Documentation support.

The note must substantiate everything billed: each code needs supporting text,
and distinctness modifiers (59 / X{EPSU} / 25) need explicit justification —
appending 59 with no documented distinct service is a top audit trigger.

This layer determines whether a claim survives a post-payment audit even after
it pays, so findings are advisory (WARN) rather than hard denials.
"""
from __future__ import annotations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status

_DISTINCT_LANGUAGE = (
    "separate", "distinct", "different site", "different session", "different encounter",
    "independent", "unrelated",
)


class DocumentationAgent(ComplianceAgent):
    filter_id = "DOCUMENTATION"
    filter_name = "Documentation support"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        note = (claim.note_text or "").lower()

        for line in claim.lines:
            has_support = bool((line.supporting_text or "").strip() or line.evidence_spans)
            if not has_support:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code} has no supporting documentation captured from the note.",
                    recommendation="Confirm the procedure/service is documented; attach the supporting "
                                   "note text before submission.",
                    source_rule="documentation audit",
                ))
            # distinctness modifiers must be justified
            distinct_mods = set(line.modifiers) & self.store.modifier_codes_for_role(
                "ncci_procedure_separation")
            if distinct_mods and note and not any(k in note for k in _DISTINCT_LANGUAGE):
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.HIGH,
                    reason=f"{line.code} carries {sorted(distinct_mods)} but the note has no language "
                           f"establishing a distinct/separate service — a top audit trigger.",
                    recommendation="Document the distinct site/session/encounter that justifies the "
                                   "modifier, or remove it.",
                    source_rule="documentation audit — modifier justification",
                ))

        for dx in claim.diagnoses:
            if not (dx.supporting_text or "").strip():
                findings.append(self.finding(
                    status=Status.WARN, codes=[dx.code], denial_risk=DenialRisk.LOW,
                    reason=f"Diagnosis {dx.code} has no supporting text captured from the note.",
                    recommendation="Confirm the diagnosis is documented in the encounter.",
                    source_rule="documentation audit",
                ))
        return findings
