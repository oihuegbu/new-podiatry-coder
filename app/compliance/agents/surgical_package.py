"""Filter #15 — Surgical package rules (NCCI Policy Manual Chapter 1).

Three Chapter-1 rule families that live in the CODE SET itself (descriptor
designations and CPT section grammar), not in the PTP edit tables — which is
exactly why they were previously unenforced: no edit-table row ever fires for
them.

  * §J "Separate procedure" designation — a code whose own descriptor carries
    "(separate procedure)" is not separately reportable when performed with
    another procedure in an anatomically related area through the same
    approach. The manual is explicit that the PTP tables contain "many, but
    not all, possible edits based on these principles", so absence of a PTP
    row is NOT clearance for these codes.
  * §G Anesthesia included in the surgical procedure — Medicare allows no
    separate payment for anesthesia furnished by the practitioner who also
    performs the procedure. An anesthesia-section line alongside a surgical
    line on the same single-provider claim is package-bundled.
  * §T Unlisted procedure codes — no fee-schedule value exists; they price
    manually and require the operative report. Flag for review so one never
    ships silently as if routine.

All designations are read from each code's own descriptor / section range —
no curated code lists.
"""
from __future__ import annotations

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status

# CPT structural section ranges (code-system grammar, not medical-rule lists)
_SURGERY_SECTION = range(10000, 70000)
_ANESTHESIA_SECTION = range(100, 2000)   # 00100–01999
# NCCI-associated separation modifiers + laterality (a separate-procedure code
# performed at a DIFFERENT site/session is legitimately reportable)
_SEPARATION_MODIFIERS = {"59", "XE", "XS", "XP", "XU", "RT", "LT",
                         "TA", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
                         "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "FA"}


def _in_section(code: str, section: range) -> bool:
    return code.isdigit() and int(code) in section


class SurgicalPackageAgent(ComplianceAgent):
    filter_id = "SURGICAL_PACKAGE"
    filter_name = "Surgical package (NCCI Ch.1 designations)"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        cpt_lines = [ln for ln in claim.lines if ln.code_system == "CPT" and ln.code]
        surgical = [ln for ln in cpt_lines if _in_section(ln.code, _SURGERY_SECTION)]

        # --- §J: "(separate procedure)" designation ---
        for line in surgical:
            if not self.store.is_separate_procedure(line.code):
                continue
            companions = [ln for ln in surgical if ln is not line]
            if not companions:
                continue  # performed independently — the designation permits it
            mods = {m.strip().upper() for m in line.modifiers}
            if mods & _SEPARATION_MODIFIERS:
                continue  # distinct site/session asserted — PTP agent audits the pair
            # If NCCI publishes a PTP row for the pair, the PTP agent already
            # adjudicates it with the modifier indicator; this check exists
            # for the pairs the tables DON'T cover.
            if any(self.store.ncci_pair(line.code, c.code, claim.date_of_service)
                   for c in companions):
                continue
            others = ", ".join(sorted({c.code for c in companions}))
            findings.append(self.finding(
                status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                reason=f"{line.code}'s CPT descriptor carries the '(separate procedure)' "
                       f"designation and it is billed alongside {others} with no "
                       f"distinct-site/session modifier — per NCCI Ch.1 §J it is not "
                       f"separately reportable when performed with another procedure in an "
                       f"anatomically related area through the same approach (and NCCI's "
                       f"PTP tables deliberately do not list every such pair).",
                recommendation=f"If {line.code} was performed at a separate site/session, "
                               f"append the appropriate anatomic or X{{EPSU}} modifier; if it "
                               f"was integral to the other procedure, remove it.",
                source_rule="NCCI Policy Manual Ch.1 §J — CPT 'separate procedure' designation",
            ))

        # --- §G: anesthesia billed with the surgery it serves ---
        anesthesia = [ln for ln in cpt_lines if _in_section(ln.code, _ANESTHESIA_SECTION)]
        if anesthesia and surgical:
            surg_codes = ", ".join(sorted({ln.code for ln in surgical}))
            for line in anesthesia:
                status = (Status.FAIL if claim.payer.follows_medicare_coverage
                          else Status.WARN)
                findings.append(self.finding(
                    status=status, codes=[line.code], denial_risk=DenialRisk.HIGH,
                    reason=f"Anesthesia code {line.code} billed on the same claim as "
                           f"procedure(s) {surg_codes} — a single-provider claim means the "
                           f"operating practitioner furnished the anesthesia, and Medicare "
                           f"makes no separate payment for it (included in the surgical "
                           f"package)."
                           + ("" if claim.payer.follows_medicare_coverage else
                              " Advisory for this non-Medicare payer — verify its policy."),
                    recommendation="Remove the anesthesia line (local/regional anesthesia by "
                                   "the surgeon is part of the procedure's package); a "
                                   "separate anesthesia practitioner bills on their own claim.",
                    source_rule="NCCI Policy Manual Ch.1 §G / CMS Anesthesia Rules",
                ))

        # --- §T: unlisted procedure codes require review + documentation ---
        for line in cpt_lines:
            if self.store.is_unlisted_procedure(line.code):
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code} is an unlisted-procedure code — it has no "
                           f"fee-schedule value, prices manually, and the claim must carry "
                           f"the operative report / description of the service.",
                    recommendation="Confirm no specific CPT code describes the service; attach "
                                   "the operative report and a comparable-code pricing basis.",
                    source_rule="NCCI Policy Manual Ch.1 §T — unlisted procedure codes",
                ))
        return findings
