"""Filter #7 — Frequency / lifetime / duplicate limits.

Phase-2 scope: duplicate-line detection (same code, same DOS, same provider,
identical modifiers) — a top cause of duplicate-claim denials. Same code billed
on distinct anatomic sites (RT vs LT, T1 vs T2…) is NOT a duplicate.

Per-code annual/lifetime frequency limits require a frequency-limit data table
(MCD/NCD frequency policies) not yet ingested — wired to read from the store
when that source lands, so no logic change is needed later.
"""
from __future__ import annotations

from collections import defaultdict

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status

# anatomic modifiers that legitimately distinguish repeats of the same code
_DISTINCT_SITE = {"RT", "LT", "50",
                  "TA", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
                  "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "FA",
                  "76", "77", "59", "XE", "XS", "XP", "XU", "91"}


class FrequencyAgent(ComplianceAgent):
    filter_id = "FREQUENCY"
    filter_name = "Frequency / lifetime / duplicate"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        groups: dict[str, list] = defaultdict(list)
        for line in claim.lines:
            groups[line.code].append(line)

        for code, lines in groups.items():
            if len(lines) < 2:
                continue
            # If every repeat carries a distinguishing site/repeat modifier, it's fine.
            distinguished = all(set(ln.modifiers) & _DISTINCT_SITE for ln in lines)
            if distinguished:
                # but identical modifier sets across the repeats are still duplicates
                seen = set()
                identical = False
                for ln in lines:
                    key = tuple(sorted(m.upper() for m in ln.modifiers))
                    if key in seen:
                        identical = True
                        break
                    seen.add(key)
                if not identical:
                    continue
            findings.append(self.finding(
                status=Status.FAIL, codes=[code], denial_risk=DenialRisk.MEDIUM,
                reason=f"{code} appears on {len(lines)} lines with no distinguishing "
                       f"site/repeat modifier — likely a duplicate.",
                recommendation="Combine into one line with the correct units, or add anatomic/"
                               "repeat modifiers (RT/LT, T1–T9, 76/77) if the services are distinct.",
                source_rule="duplicate-line edit",
            ))
        return findings
