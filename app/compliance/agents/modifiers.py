"""Filter #4 — Modifier validity.

  * Every appended modifier must be a recognized CPT/HCPCS modifier.
  * CMS prefers the X{EPSU} subset over 59 — flag 59 as an advisory.
  * Catch conflicting/redundant modifiers (e.g. RT+LT on one line → should be 50;
    50 with RT/LT; duplicate modifiers).
All modifier knowledge comes from the `modifier` reference table — no hardcoding.
"""
from __future__ import annotations

from datetime import datetime

from app.compliance.agents.base import ComplianceAgent
from app.compliance.models import Claim, DenialRisk, Finding, Status

_LATERAL = {"RT", "LT"}
# CPT structural sections (code-system grammar, not medical-rule code lists)
_EM_SECTION = range(99202, 99500)
_SURGERY_SECTION = range(10000, 70000)
# Modifiers that separate an E/M from a same-day procedure
_EM_SEPARATION = {"25", "57"}
# Modifiers whose entire meaning is "this service is inside a PRIOR surgery's
# postoperative global period" (24 unrelated E/M, 58 staged, 78 complication
# return, 79 unrelated procedure) — modifier-system semantics, same set the
# global-period agent uses as its bypass list. Mirror check: that agent asks
# "post-op window without a bypass modifier?"; this one asks "bypass modifier
# without a post-op window?" — a stray 79 on a fresh encounter misstates the
# encounter context to the payer and previously passed every filter, because
# the global-period agent returns early when no prior surgery is detected.
_POSTOP_CONTEXT_MODIFIERS = {"24", "58", "78", "79"}
# Modifier-system grammar (CPT Appendix A semantics, not medical-rule lists):
# 24/25/57 attach ONLY to E/M codes; 58/78/79 (and the split-care and
# repeat-procedure sets below) attach only to procedures.
_EM_ONLY_MODIFIERS = {"24", "25", "57"}
_PROCEDURE_ONLY_MODIFIERS = {"58", "78", "79", "54", "55", "56", "76", "77",
                             "50", "51", "52", "53", "62", "66", "63"}
# Split-care set: one provider performing only part of the global package.
# Only meaningful on codes that HAVE a post-op package (010/090 global days).
_SPLIT_CARE = {"54", "55", "56"}
# Discontinued-procedure family: 53 is the physician form; 73/74 are defined
# exclusively for the FACILITY side (hospital-outpatient/ASC institutional
# claims). This system scrubs a private practice's professional (CMS-1500)
# claims, so 73/74 are categorically wrong here — even when the procedure
# happened at an ASC, the surgeon's professional claim uses 53 and the
# facility reports 73/74 on its own institutional claim.
_FACILITY_ONLY_DISCONTINUED = {"73", "74"}
_REPEAT_MODIFIERS = {"76", "77"}


def _is_em(code: str) -> bool:
    return code.isdigit() and int(code) in _EM_SECTION


def _is_surgery(code: str) -> bool:
    return code.isdigit() and int(code) in _SURGERY_SECTION


def _age_years(claim: Claim) -> float | None:
    """Patient age at DOS from the claim's own DOB (YYYYMMDD) — same
    derivation the MCE agent uses. None when either date is missing."""
    dob_raw = (claim.subscriber.date_of_birth or "").strip()
    if not dob_raw or not claim.date_of_service:
        return None
    try:
        dob = datetime.strptime(dob_raw, "%Y%m%d").date()
    except ValueError:
        return None
    days = (claim.date_of_service - dob).days
    return days / 365.25 if days >= 0 else None


class ModifierAgent(ComplianceAgent):
    filter_id = "MODIFIERS"
    filter_name = "Modifier validity"

    def check(self, claim: Claim) -> list[Finding]:
        findings: list[Finding] = []
        in_postop_window = bool((claim.prior_surgery_info or {}).get("is_post_op_visit"))
        for line in claim.lines:
            mods = [m.strip().upper() for m in line.modifiers if m and m.strip()]

            # post-op-context modifier without any post-op context
            if not in_postop_window:
                for m in sorted(set(mods) & _POSTOP_CONTEXT_MODIFIERS):
                    findings.append(self.finding(
                        status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: modifier {m} asserts this service falls within a prior "
                               f"surgery's postoperative global period, but the encounter documents "
                               f"no prior surgery / post-op context.",
                        recommendation=f"Remove modifier {m}, or document the prior surgery (CPT and "
                                       f"date) that places this encounter in its global period.",
                        source_rule="global-period modifier semantics (24/58/78/79)",
                    ))

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

            # --- PFS payment-policy indicators: CMS publishes per-code
            # modifier-validity signals (global_periods.json, PPRRVU-derived)
            # that were ingested but never consumed. Each indicator below IS
            # the authoritative answer to "may this modifier appear on this
            # code" — no code lists involved.
            pfs = self.store.pfs_indicators(line.code, claim.date_of_service)
            if pfs:
                # PC/TC split: 26 (professional) / TC (technical) only exist
                # for indicator '1' (diagnostic test with a PC/TC split).
                # '2' = code already IS the professional component; '3' =
                # already technical-only; 0/4/5/8/9 = no split concept.
                pctc = pfs.get("pctc_ind")
                for m in ("26", "TC"):
                    if m in mod_set and pctc is not None and pctc != "1":
                        findings.append(self.finding(
                            status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                            reason=f"{line.code}: modifier {m} appended, but its PFS PC/TC "
                                   f"indicator is '{pctc}' — this code has no professional/"
                                   f"technical split to bill separately.",
                            recommendation=f"Remove modifier {m}; bill the code globally.",
                            source_rule=f"PFS PC/TC indicator={pctc}",
                        ))
                # Bilateral: '1' is the only value where the 150% bilateral
                # adjustment (modifier 50) applies. '2' = code is already
                # inherently bilateral; '0'/'9' = concept doesn't apply.
                bilat = pfs.get("bilat_surg")
                if "50" in mod_set and bilat is not None and bilat != "1":
                    detail = ("the code is already inherently bilateral"
                              if bilat == "2" else "the bilateral payment concept does not apply")
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: modifier 50 appended, but its PFS bilateral-"
                               f"surgery indicator is '{bilat}' — {detail}.",
                        recommendation="Remove modifier 50; bill a single line (or RT/LT "
                                       "lines if sides must be distinguished).",
                        source_rule=f"PFS bilateral-surgery indicator={bilat}",
                    ))
                # Assistant surgeon (80/81/82/AS): '0'/'1' = payment
                # restriction applies (assistant rarely/never paid).
                asst = pfs.get("asst_surg")
                asst_mods = mod_set & {"80", "81", "82", "AS"}
                if asst_mods and asst in ("0", "1"):
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                        reason=f"{line.code}: assistant-surgeon modifier "
                               f"{'/'.join(sorted(asst_mods))} appended, but its PFS "
                               f"assistant-surgery indicator is '{asst}' — assistants are "
                               f"restricted from payment for this procedure.",
                        recommendation="Remove the assistant-surgeon line/modifier, or provide "
                                       "medical-necessity documentation if the payer allows appeal.",
                        source_rule=f"PFS assistant-surgery indicator={asst}",
                    ))
                # Co-surgeon (62) / team surgeon (66): '0' = not permitted.
                for m, ind_key, label in (("62", "co_surg", "co-surgeon"),
                                          ("66", "team_surg", "team-surgery")):
                    ind = pfs.get(ind_key)
                    if m in mod_set and ind == "0":
                        findings.append(self.finding(
                            status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                            reason=f"{line.code}: modifier {m} appended, but its PFS "
                                   f"{label} indicator is '0' — {label} billing is not "
                                   f"permitted for this procedure.",
                            recommendation=f"Remove modifier {m}.",
                            source_rule=f"PFS {label} indicator=0",
                        ))

            # --- Modifier↔code-category grammar: E/M-only modifiers on a
            # procedure line (or procedure-only modifiers on an E/M line)
            # assert a service category the code isn't — a hard form error.
            if not _is_em(line.code):
                for m in sorted(mod_set & _EM_ONLY_MODIFIERS):
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: modifier {m} is defined for E/M services only, "
                               f"but {line.code} is not an E/M code.",
                        recommendation=f"Remove modifier {m} from this line (append it to the "
                                       f"E/M line if one exists and qualifies).",
                        source_rule="CPT Appendix A — E/M-only modifier",
                    ))
            else:
                for m in sorted(mod_set & _PROCEDURE_ONLY_MODIFIERS):
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: modifier {m} is a procedure modifier and is not "
                               f"valid on an E/M service.",
                        recommendation=f"Remove modifier {m} from the E/M line.",
                        source_rule="CPT Appendix A — procedure-only modifier",
                    ))

            # --- Split-care 54/55/56: bill parts of ONE global package. Only
            # codes carrying a 010/090 post-op package can be split; a
            # 000-global or XXX code has no package to divide. And one
            # provider line cannot simultaneously claim two different
            # exclusive parts ("surgical care only" + "post-op only").
            split_present = sorted(mod_set & _SPLIT_CARE)
            if split_present:
                if len(split_present) > 1:
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: modifiers {'/'.join(split_present)} combined — each "
                               f"asserts this line is ONLY one exclusive part of the global "
                               f"package (surgical / post-op / pre-op care only); they cannot "
                               f"coexist on one line.",
                        recommendation="Keep the one modifier matching the care actually "
                                       "furnished; a provider furnishing the whole package "
                                       "bills with none of them.",
                        source_rule="CMS global surgery — split-care modifiers 54/55/56",
                    ))
                glob = (self.store.global_period(line.code, claim.date_of_service) or "").strip()
                if glob and glob not in ("010", "090"):
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: split-care modifier {'/'.join(split_present)} "
                               f"appended, but the code's PFS global indicator is '{glob}' — "
                               f"only 010/090-day global procedures have a pre/post-operative "
                               f"package that can be split between providers.",
                        recommendation=f"Remove modifier {'/'.join(split_present)}; there is no "
                                       f"global package to divide on this code.",
                        source_rule=f"CMS global surgery (MLN907166) — PFS GLOB DAYS={glob}",
                    ))

            # --- 73/74 belong to the FACILITY'S institutional claim, never a
            # physician's professional claim — and this system scrubs a
            # private practice's professional claims exclusively. No POS can
            # legitimize them here: at an ASC the surgeon still reports 53 on
            # the professional claim while the facility reports 73/74 on its
            # own claim form.
            fac_disc = sorted(mod_set & _FACILITY_ONLY_DISCONTINUED)
            if fac_disc:
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: modifier {'/'.join(fac_disc)} reports a discontinued "
                           f"procedure on the FACILITY's institutional claim (hospital "
                           f"outpatient/ASC) — it is never valid on a physician's "
                           f"professional claim, which this is.",
                    recommendation="Use modifier 53 (discontinued procedure) on the "
                                   "professional claim; the facility reports 73/74 on its own "
                                   "claim if applicable.",
                    source_rule="CMS — modifiers 73/74 are institutional (facility) claim "
                                "modifiers; professional claims use 53",
                ))
            # 52 (reduced) and 53 (discontinued) describe mutually exclusive
            # outcomes of the same service — partially completed as planned
            # vs terminated for patient risk.
            if {"52", "53"}.issubset(mod_set):
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: modifiers 52 (reduced services) and 53 (discontinued "
                           f"procedure) combined — a service was either electively reduced or "
                           f"terminated for patient wellbeing, not both.",
                    recommendation="Keep the one modifier matching what the note documents.",
                    source_rule="CPT Appendix A — 52 vs 53 semantics",
                ))

            # --- Modifier 63 (procedure on infant < 4 kg): CPT Appendix F
            # lists codes whose value already includes the infant work
            # (63-exempt, ingested from modifier_exempt.json but previously
            # never consumed); and the patient's own DOB rules 63 out for
            # anyone who cannot weigh under 4 kg.
            if "63" in mod_set:
                if self.store.is_modifier_63_exempt(line.code):
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: modifier 63 appended to a modifier-63-exempt code "
                               f"(CPT Appendix F) — the code's value already reflects the "
                               f"increased complexity of infant patients.",
                        recommendation="Remove modifier 63.",
                        source_rule="CPT Appendix F — modifier 63 exempt codes",
                    ))
                age = _age_years(claim)
                if age is not None and age >= 1:
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                        reason=f"{line.code}: modifier 63 asserts a procedure on an infant "
                               f"presently under 4 kg, but the patient is "
                               f"{int(age)} year(s) old at the date of service.",
                        recommendation="Remove modifier 63.",
                        source_rule="CPT modifier 63 definition (infants less than 4 kg)",
                    ))

            # --- Modifier 51 vs the code's own PFS multiple-procedure
            # indicator (complements the Appendix E exemption check above —
            # CMS's own per-code signal that no reduction concept applies).
            if "51" in mod_set and not self.store.is_modifier_51_exempt(line.code):
                mult = (pfs or {}).get("mult_proc")
                if mult in ("0", "9"):
                    findings.append(self.finding(
                        status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.LOW,
                        reason=f"{line.code}: modifier 51 appended, but its PFS multiple-"
                               f"procedure indicator is '{mult}' — no multiple-procedure "
                               f"payment adjustment applies to this code.",
                        recommendation="Remove modifier 51 (most payers rank and reduce "
                                       "automatically; 51 adds no information here).",
                        source_rule=f"PFS multiple-procedure indicator={mult}",
                    ))

            # --- Modifier 50 with units > 1 double-bills the bilateral
            # adjustment: Medicare bilateral billing is ONE line, ONE unit,
            # modifier 50 (payment made at 150%).
            if "50" in mod_set and line.units > 1:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: modifier 50 with {line.units} units — bilateral "
                           f"procedures are billed as one line with one unit (the 150% "
                           f"bilateral adjustment already pays both sides); extra units "
                           f"double-bill the second side.",
                    recommendation="Reduce to 1 unit, or drop modifier 50 and bill RT/LT lines "
                                   "if the payer requires split lines.",
                    source_rule="CMS bilateral billing convention (modifier 50, 1 unit)",
                ))

            # --- Repeat modifiers 76/77 assert THIS line repeats an earlier
            # service of the same code today — meaningless when no initial
            # (unmodified) line of that code exists on the claim.
            rep = sorted(mod_set & _REPEAT_MODIFIERS)
            if rep:
                initial_exists = any(
                    ln is not line and ln.code == line.code
                    and not (set(m.upper() for m in ln.modifiers) & _REPEAT_MODIFIERS)
                    for ln in claim.lines
                )
                if not initial_exists:
                    findings.append(self.finding(
                        status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: repeat-procedure modifier {'/'.join(rep)} "
                               f"appended, but no initial line of {line.code} exists on this "
                               f"claim for it to repeat.",
                        recommendation="Bill the initial service as its own unmodified line, "
                                       "or remove the repeat modifier if the service was "
                                       "performed once.",
                        source_rule="CPT Appendix A — repeat procedure modifiers 76/77",
                    ))

            # --- Modifier 82 (assistant surgeon — qualified resident NOT
            # available) exists for teaching-hospital settings where a
            # resident would normally assist. A private surgical practice has
            # no residency program to certify unavailability for; its
            # assistants report 80/81 (physician) or AS (PA/NP/CNS).
            if "82" in mod_set:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: modifier 82 asserts a qualified RESIDENT was "
                           f"unavailable — a teaching-hospital certification that a private "
                           f"practice's professional claim cannot normally support.",
                    recommendation="Use modifier 80/81 for a physician assistant-at-surgery "
                                   "or AS for a PA/NP/CNS; keep 82 only if the case was "
                                   "genuinely performed at a teaching facility with the "
                                   "unavailability documented.",
                    source_rule="CMS assistant-at-surgery modifiers — 82 requires resident "
                                "unavailability (teaching settings)",
                ))

            # --- Modifier 47 (anesthesia by surgeon): Medicare makes no
            # separate payment for anesthesia furnished by the operating
            # practitioner — it is part of the surgical package (NCCI Ch.1 §G).
            if "47" in mod_set:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code],
                    denial_risk=DenialRisk.HIGH if claim.payer.follows_medicare_coverage
                    else DenialRisk.MEDIUM,
                    reason=f"{line.code}: modifier 47 (anesthesia by surgeon) — Medicare "
                           f"allows no separate payment for anesthesia furnished by the "
                           f"practitioner performing the procedure; it is included in the "
                           f"surgical package."
                           + ("" if claim.payer.follows_medicare_coverage
                              else " Non-Medicare payers vary — verify this payer recognizes 47."),
                    recommendation="Remove modifier 47 for Medicare-bound claims; verify payer "
                                   "policy otherwise.",
                    source_rule="NCCI Policy Manual Ch. 1 §G — anesthesia included in the "
                                "surgical procedure",
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
