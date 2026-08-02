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
        roles = {
            name: self.store.modifier_codes_for_role(name)
            for name in (
                "laterality", "bilateral", "multiple_procedure",
                "professional_component", "technical_component", "assistant_surgery",
                "co_surgeon", "team_surgeon", "postoperative_context",
                "postoperative_em", "postoperative_procedure", "ncci_em_separation",
                "split_care", "repeat_service", "facility_discontinued",
                "reduced_service", "professional_discontinued", "infant_procedure",
                "teaching_assistant", "anesthesia_by_surgeon",
                "ncci_procedure_separation", "ncci_legacy_separation",
                "ncci_specific_separation",
            )
        }
        em_only = roles["postoperative_em"] | roles["ncci_em_separation"]
        procedure_only = set().union(*(
            roles[name] for name in (
                "postoperative_procedure", "split_care", "repeat_service",
                "bilateral", "multiple_procedure", "reduced_service",
                "professional_discontinued", "co_surgeon", "team_surgeon",
                "infant_procedure",
            )
        ))
        for line in claim.lines:
            mods = [m.strip().upper() for m in line.modifiers if m and m.strip()]

            # post-op-context modifier without any post-op context
            if not in_postop_window:
                for m in sorted(set(mods) & roles["postoperative_context"]):
                    findings.append(self.finding(
                        status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: modifier {m} asserts this service falls within a prior "
                               f"surgery's postoperative global period, but the encounter documents "
                               f"no prior surgery / post-op context.",
                        recommendation=f"Remove modifier {m}, or document the prior surgery (CPT and "
                                       f"date) that places this encounter in its global period.",
                        source_rule="AMA/CMS postoperative modifier semantics",
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
            lateral_present = mod_set & roles["laterality"]
            lateral_sides = {
                self.store.modifier_laterality(modifier)
                for modifier in lateral_present
                if self.store.modifier_laterality(modifier)
            }
            bilateral_present = mod_set & roles["bilateral"]
            # Opposing side modifiers on one line should use the authoritative
            # bilateral modifier when that billing convention applies.
            if len(lateral_sides) > 1:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: opposing laterality modifiers appear on one line.",
                    recommendation=(
                        f"Use the authoritative bilateral modifier "
                        f"({ '/'.join(sorted(roles['bilateral'])) }), or split into side-specific "
                        "lines according to payer policy."
                    ),
                    source_rule="bilateral billing convention",
                ))
            if bilateral_present and lateral_present:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: bilateral and side-specific modifiers are combined.",
                    recommendation="Use either the bilateral convention or side-specific lines, not both.",
                    source_rule="bilateral billing convention",
                ))
            multiple_present = mod_set & roles["multiple_procedure"]
            if multiple_present and self.store.is_modifier_51_exempt(line.code):
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: multiple-procedure modifier "
                           f"{'/'.join(sorted(multiple_present))} appended to an exempt code "
                           f"(CPT Appendix E) — "
                           f"the relative value already accounts for multiple-procedure reduction.",
                    recommendation="Remove the modifier; the payer will reduce this code automatically or it "
                                   "is not subject to reduction.",
                    source_rule="CPT Appendix E — multiple-procedure modifier exempt codes",
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
                for m in sorted(
                    roles["professional_component"] | roles["technical_component"]
                ):
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
                if bilateral_present and bilat is not None and bilat != "1":
                    detail = ("the code is already inherently bilateral"
                              if bilat == "2" else "the bilateral payment concept does not apply")
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: bilateral modifier "
                               f"{'/'.join(sorted(bilateral_present))} appended, but its PFS bilateral-"
                               f"surgery indicator is '{bilat}' — {detail}.",
                        recommendation="Remove the bilateral modifier; bill a single line or "
                                       "side-specific lines when the payer requires them.",
                        source_rule=f"PFS bilateral-surgery indicator={bilat}",
                    ))
                # Assistant surgeon (80/81/82/AS): '0'/'1' = payment
                # restriction applies (assistant rarely/never paid).
                asst = pfs.get("asst_surg")
                asst_mods = mod_set & roles["assistant_surgery"]
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
                for role, ind_key, label in (
                    ("co_surgeon", "co_surg", "co-surgeon"),
                    ("team_surgeon", "team_surg", "team-surgery"),
                ):
                    ind = pfs.get(ind_key)
                    present = mod_set & roles[role]
                    if present and ind == "0":
                        findings.append(self.finding(
                            status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                            reason=f"{line.code}: modifier {'/'.join(sorted(present))} appended, but its PFS "
                                   f"{label} indicator is '0' — {label} billing is not "
                                   f"permitted for this procedure.",
                            recommendation=f"Remove modifier {'/'.join(sorted(present))}.",
                            source_rule=f"PFS {label} indicator=0",
                        ))

            # --- Modifier↔code-category grammar: E/M-only modifiers on a
            # procedure line (or procedure-only modifiers on an E/M line)
            # assert a service category the code isn't — a hard form error.
            if not self.store.is_em_code(line.code, claim.date_of_service):
                for m in sorted(mod_set & em_only):
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: modifier {m} is defined for E/M services only, "
                               f"but {line.code} is not an E/M code.",
                        recommendation=f"Remove modifier {m} from this line (append it to the "
                                       f"E/M line if one exists and qualifies).",
                        source_rule="CPT Appendix A — E/M-only modifier",
                    ))
            else:
                for m in sorted(mod_set & procedure_only):
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
            split_present = sorted(mod_set & roles["split_care"])
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
                if glob and (not glob.isdigit() or int(glob) <= 0):
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: split-care modifier {'/'.join(split_present)} "
                               f"appended, but the code's PFS global indicator is '{glob}' — "
                               "only procedures with a positive numeric global period have a "
                               "pre/post-operative package that can be split between providers.",
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
            fac_disc = sorted(mod_set & roles["facility_discontinued"])
            if fac_disc:
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: modifier {'/'.join(fac_disc)} reports a discontinued "
                           f"procedure on the FACILITY's institutional claim (hospital "
                           f"outpatient/ASC) — it is never valid on a physician's "
                           f"professional claim, which this is.",
                    recommendation=(
                        "Use the professional discontinued-procedure modifier "
                        f"({ '/'.join(sorted(roles['professional_discontinued'])) }) on the "
                        "professional claim; the facility reports its own applicable modifier."
                    ),
                    source_rule="CMS — facility and professional discontinued-procedure semantics",
                ))
            # 52 (reduced) and 53 (discontinued) describe mutually exclusive
            # outcomes of the same service — partially completed as planned
            # vs terminated for patient risk.
            reduced_present = mod_set & roles["reduced_service"]
            discontinued_present = mod_set & roles["professional_discontinued"]
            if reduced_present and discontinued_present:
                findings.append(self.finding(
                    status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: reduced-service and discontinued-procedure modifiers "
                           f"are combined — a service was either electively reduced or "
                           f"terminated for patient wellbeing, not both.",
                    recommendation="Keep the one modifier matching what the note documents.",
                    source_rule="CPT Appendix A — reduced vs discontinued service semantics",
                ))

            # --- Modifier 63 (procedure on infant < 4 kg): CPT Appendix F
            # lists codes whose value already includes the infant work
            # (63-exempt, ingested from modifier_exempt.json but previously
            # never consumed); and the patient's own DOB rules 63 out for
            # anyone who cannot weigh under 4 kg.
            infant_present = mod_set & roles["infant_procedure"]
            if infant_present:
                if self.store.is_modifier_63_exempt(line.code):
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                        reason=f"{line.code}: infant-procedure modifier appended to an exempt code "
                               f"(CPT Appendix F) — the code's value already reflects the "
                               f"increased complexity of infant patients.",
                        recommendation=f"Remove modifier {'/'.join(sorted(infant_present))}.",
                        source_rule="CPT Appendix F — infant-procedure modifier exempt codes",
                    ))
                age = _age_years(claim)
                if age is not None and age >= 1:
                    findings.append(self.finding(
                        status=Status.FAIL, codes=[line.code], denial_risk=DenialRisk.HIGH,
                        reason=f"{line.code}: modifier {'/'.join(sorted(infant_present))} asserts a procedure on an infant "
                               f"presently under 4 kg, but the patient is "
                               f"{int(age)} year(s) old at the date of service.",
                        recommendation=f"Remove modifier {'/'.join(sorted(infant_present))}.",
                        source_rule="CPT infant-procedure modifier definition",
                    ))

            # --- Modifier 51 vs the code's own PFS multiple-procedure
            # indicator (complements the Appendix E exemption check above —
            # CMS's own per-code signal that no reduction concept applies).
            if multiple_present and not self.store.is_modifier_51_exempt(line.code):
                mult = (pfs or {}).get("mult_proc")
                if mult in ("0", "9"):
                    findings.append(self.finding(
                        status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.LOW,
                        reason=f"{line.code}: multiple-procedure modifier appended, but its PFS multiple-"
                               f"procedure indicator is '{mult}' — no multiple-procedure "
                               f"payment adjustment applies to this code.",
                        recommendation="Remove the modifier (most payers rank and reduce "
                                       "automatically; 51 adds no information here).",
                        source_rule=f"PFS multiple-procedure indicator={mult}",
                    ))

            # --- Modifier 50 with units > 1 double-bills the bilateral
            # adjustment: Medicare bilateral billing is ONE line, ONE unit,
            # modifier 50 (payment made at 150%).
            if bilateral_present and line.units > 1:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: bilateral modifier with {line.units} units — bilateral "
                           f"procedures are billed as one line with one unit (the 150% "
                           f"bilateral adjustment already pays both sides); extra units "
                           f"double-bill the second side.",
                    recommendation="Reduce to one unit, or drop the bilateral modifier and bill side-specific lines "
                                   "if the payer requires split lines.",
                    source_rule="CMS bilateral billing convention",
                ))

            # --- Repeat modifiers 76/77 assert THIS line repeats an earlier
            # service of the same code today — meaningless when no initial
            # (unmodified) line of that code exists on the claim.
            rep = sorted(mod_set & roles["repeat_service"])
            if rep:
                initial_exists = any(
                    ln is not line and ln.code == line.code
                    and not (set(m.upper() for m in ln.modifiers) & roles["repeat_service"])
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
            teaching_present = mod_set & roles["teaching_assistant"]
            if teaching_present:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.MEDIUM,
                    reason=f"{line.code}: modifier {'/'.join(sorted(teaching_present))} asserts a qualified RESIDENT was "
                           f"unavailable — a teaching-hospital certification that a private "
                           f"practice's professional claim cannot normally support.",
                    recommendation="Use the applicable assistant-at-surgery modifier from the "
                                   "authoritative reference; keep this teaching-specific modifier only if the case was "
                                   "genuinely performed at a teaching facility with the "
                                   "unavailability documented.",
                    source_rule="CMS assistant-at-surgery modifiers — teaching modifier requires resident "
                                "unavailability (teaching settings)",
                ))

            # --- Modifier 47 (anesthesia by surgeon): Medicare makes no
            # separate payment for anesthesia furnished by the operating
            # practitioner — it is part of the surgical package (NCCI Ch.1 §G).
            anesthesia_by_surgeon = mod_set & roles["anesthesia_by_surgeon"]
            if anesthesia_by_surgeon:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code],
                    denial_risk=DenialRisk.HIGH if claim.payer.follows_medicare_coverage
                    else DenialRisk.MEDIUM,
                    reason=f"{line.code}: anesthesia-by-surgeon modifier "
                           f"{'/'.join(sorted(anesthesia_by_surgeon))} — Medicare "
                           f"allows no separate payment for anesthesia furnished by the "
                           f"practitioner performing the procedure; it is included in the "
                           f"surgical package."
                           + ("" if claim.payer.follows_medicare_coverage
                              else " Non-Medicare payers vary — verify this payer recognizes it."),
                    recommendation="Remove the modifier for Medicare-bound claims; verify payer "
                                   "policy otherwise.",
                    source_rule="NCCI Policy Manual Ch. 1 §G — anesthesia included in the "
                                "surgical procedure",
                ))

            legacy_separation = mod_set & roles["ncci_legacy_separation"]
            if legacy_separation:
                findings.append(self.finding(
                    status=Status.WARN, codes=[line.code], denial_risk=DenialRisk.LOW,
                    reason=f"{line.code}: general distinct-service modifier used — CMS prefers "
                           "the more specific separation role when applicable.",
                    recommendation=(
                        "Use the applicable specific modifier "
                        f"({ '/'.join(sorted(roles['ncci_specific_separation'])) }) describing "
                        "the distinction when supported."
                    ),
                    source_rule="NCCI policy — specific separation modifiers preferred",
                ))

        # --- E/M billed same day as a procedure → needs modifier 25 (or 57) ---
        # An E/M performed on the same day as a procedure bundles into that
        # procedure unless a significant, separately identifiable E/M is
        # documented and modifier 25 (or 57 for a major-surgery decision) is
        # appended. (Procedure = CPT surgery section, excludes radiology/lab.)
        em_lines = [
            ln for ln in claim.lines
            if self.store.is_em_code(ln.code, claim.date_of_service)
        ]
        has_same_day_procedure = any(
            self.store.is_surgical_procedure(ln.code, claim.date_of_service)
            for ln in claim.lines
        )
        if em_lines and has_same_day_procedure:
            for em in em_lines:
                if set(em.modifiers) & roles["ncci_em_separation"]:
                    continue
                findings.append(self.finding(
                    status=Status.FAIL, codes=[em.code], denial_risk=DenialRisk.HIGH,
                    reason=f"E/M {em.code} billed the same day as a procedure with no applicable "
                           "separation modifier — "
                           f"it bundles into the procedure and will not be paid separately.",
                    recommendation=(
                        "Append the applicable E/M separation modifier "
                        f"({ '/'.join(sorted(roles['ncci_em_separation'])) }) only when its "
                        "authoritative definition is documented; otherwise the E/M is not "
                        "separately billable."
                    ),
                    source_rule="NCCI — E/M same-day procedure bundling",
                ))
        return findings
