from app.rag.code_reference import CodeReferenceDB
from app.models.schemas import ValidationIssue, DocumentationAudit, EncounterIntegrity
from app.core.logger import get_logger

logger = get_logger(__name__)

EM_CODES = {"99202", "99203", "99204", "99205", "99211", "99212", "99213", "99214", "99215"}
POST_OP_EM = {"99024"}
ROUTINE_FOOT_CARE_CPTS = {"11719", "11720", "11721", "11055", "11056", "11057"}
IMAGING_PREFIXES = ("70", "71", "72", "73", "74", "75", "76", "77", "78", "79")
VALID_MODIFIERS = {
    "25", "59", "XE", "XS", "XP", "XU", "50", "51", "76", "77",
    "LT", "RT", "TA", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
    "26", "TC", "47", "80", "81", "82", "AS", "QW", "QX", "QY", "QZ",
}
MANIFESTATION_PREFIXES = {"H36", "G63", "N08", "M14"}
ETIOLOGY_PREFIXES = {"E10", "E11", "E13", "I70", "I73"}
# DM combination code prefixes — when any of these is coded, E11.9/E10.9 is redundant
DM_COMBINATION_PREFIXES = (
    "E10.1", "E10.2", "E10.3", "E10.4", "E10.5", "E10.6", "E10.7", "E10.8",
    "E11.1", "E11.2", "E11.3", "E11.4", "E11.5", "E11.6", "E11.7", "E11.8",
    "E13.1", "E13.2", "E13.3", "E13.4", "E13.5", "E13.6", "E13.7", "E13.8",
)
# Long-term drug Z-codes that should not appear on outpatient podiatry claims
INAPPROPRIATE_LONGTERM_ZCODES = {"Z79.84", "Z79.899", "Z79.84", "Z79.01", "Z79.02"}


class CodingValidator:
    def __init__(self, ref_db: CodeReferenceDB):
        self.db = ref_db
        self.issues: list[ValidationIssue] = []
        self._bundled_codes_to_suppress: set[str] = set()

    def validate(
        self,
        coding_result: dict,
        prior_surgery_info: dict | None = None,
    ) -> dict:
        self.issues = []
        self._bundled_codes_to_suppress = set()

        icd = coding_result.get("icd10_codes", [])
        cpt = coding_result.get("cpt_codes", [])
        hcpcs = coding_result.get("hcpcs_codes", [])
        snomed = coding_result.get("snomed_codes", [])

        self._check_code_existence(icd, cpt, hcpcs)
        self._check_ncci(cpt)
        self._check_mue(cpt)
        self._check_lcd(icd, cpt)
        self._check_sequencing(icd)
        self._check_cpt_dx_linkage(cpt)
        self._check_orphan_dx(icd, cpt, hcpcs)
        self._check_modifiers(cpt)
        self._check_em_modifier25(cpt)
        self._check_global_period(cpt, prior_surgery_info)
        self._check_redundant_dm_codes(icd)
        self._check_inappropriate_zcodes(icd)
        self._check_snomed_consistency(snomed)

        # Remove bundled codes from CPT list (NCCI suppression)
        if self._bundled_codes_to_suppress:
            original_count = len(cpt)
            cpt[:] = [c for c in cpt if c.get("code", "") not in self._bundled_codes_to_suppress]
            removed = original_count - len(cpt)
            if removed:
                logger.info(f"  Suppressed {removed} NCCI-bundled CPT code(s): {self._bundled_codes_to_suppress}")
                coding_result["cpt_codes"] = cpt

        enc = self._encounter_integrity(icd)
        audit = self._documentation_audit(coding_result)
        tier, confidence, reasons = self._compute_tier(icd, cpt, hcpcs)

        # pre_submission_audit_findings = only ERROR and WARNING items (not INFO)
        critical_issues = [
            i for i in self.issues if i.severity in ("ERROR", "WARNING", "CRITICAL")
        ]

        return {
            "validation_issues": [i.model_dump() for i in self.issues],
            "encounter_integrity": enc.model_dump(),
            "documentation_audit": audit.model_dump(),
            "pre_submission_audit_findings": [i.model_dump() for i in critical_issues],
            "pre_submission_audit_score": self._audit_score(),
            "auto_coding_tier": tier,
            "auto_coding_confidence": confidence,
            "auto_coding_review_reasons": reasons,
            "auto_coding_summary": self._summary(tier, reasons),
        }

    # --- Individual checks ---

    def _check_code_existence(self, icd, cpt, hcpcs):
        for entry in icd:
            code = entry.get("code", "")
            if self.db.validate_icd10(code):
                entry["s3_validated"] = True
            else:
                self._add(
                    "ERROR", code, "code_existence",
                    f"ICD-10-CM {code} not found in FY2026 code set",
                    "Verify code or use a valid alternative",
                    denial_risk="HIGH",
                )

        for entry in cpt:
            code = entry.get("code", "")
            if self.db.validate_cpt(code):
                entry["ama_validated"] = True
            else:
                self._add(
                    "ERROR", code, "code_existence",
                    f"CPT {code} not found in code set",
                    "Verify code or use a valid alternative",
                    denial_risk="HIGH",
                )

        for entry in hcpcs:
            code = entry.get("code", "")
            if code and not self.db.validate_hcpcs(code):
                self._add(
                    "INFO", code, "code_existence",
                    f"HCPCS {code} not found in database (may still be valid — verify with payer)",
                    "Verify HCPCS code validity with payer",
                    denial_risk="MEDIUM",
                )

    def _check_ncci(self, cpt):
        codes = [c.get("code", "") for c in cpt if c.get("code")]
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                conflict = self.db.check_ncci(codes[i], codes[j])
                if not conflict:
                    continue

                # Determine modifier indicator:
                # '0' = no modifier exception allowed (hard edit)
                # '1' = modifier exception allowed
                # '9' = concept does not apply
                # '' or other = treat as hard edit
                mod_indicator = str(conflict.get("modifier", "")).strip()
                modifier_allowed = mod_indicator in ("1", "9")

                # Check if a valid separator modifier is present on either code
                sep_modifiers = {"59", "XE", "XS", "XP", "XU"}
                code_i_entry = next((c for c in cpt if c.get("code") == codes[i]), {})
                code_j_entry = next((c for c in cpt if c.get("code") == codes[j]), {})
                has_separator = (
                    bool(set(code_i_entry.get("modifiers", [])) & sep_modifiers)
                    or bool(set(code_j_entry.get("modifiers", [])) & sep_modifiers)
                )

                if modifier_allowed and has_separator:
                    self._add(
                        "INFO", f"{codes[i]}|{codes[j]}", "ncci_edit",
                        f"NCCI pair {codes[i]}/{codes[j]} — modifier exception applied",
                        "Verify distinct service is documented",
                        denial_risk="LOW",
                    )
                elif modifier_allowed and not has_separator:
                    self._add(
                        "WARNING", f"{codes[i]}|{codes[j]}", "ncci_edit",
                        f"NCCI conflict: {codes[i]} and {codes[j]} — modifier exception allowed but not applied",
                        "Add modifier 59/XE/XS/XP/XU if services are distinct and documented",
                        denial_risk="MEDIUM",
                    )
                else:
                    # Hard edit — no modifier exception possible
                    # column2 code is the one to suppress (codes[j] when entry is c1|c2)
                    key_fwd = f"{codes[i]}|{codes[j]}"
                    key_rev = f"{codes[j]}|{codes[i]}"
                    if self.db.ncci.get(key_fwd):
                        bundled = codes[j]   # j is the column-2 (component) code
                        primary = codes[i]
                    else:
                        bundled = codes[i]
                        primary = codes[j]
                    self._bundled_codes_to_suppress.add(bundled)
                    self._add(
                        "ERROR", f"{codes[i]}|{codes[j]}", "ncci_edit",
                        f"NCCI hard edit: {primary} and {bundled} are mutually exclusive. "
                        f"{bundled} is bundled into {primary} — suppressing {bundled}.",
                        f"Remove {bundled} from claim — it is included in {primary}",
                        denial_risk="HIGH",
                    )

    def _check_mue(self, cpt):
        for entry in cpt:
            code = entry.get("code", "")
            units = entry.get("units", 1)
            mue = self.db.get_mue(code)
            if mue is not None and mue > 0:
                if units > mue:
                    self._add(
                        "ERROR", code, "mue_limit",
                        f"CPT {code} units ({units}) exceeds MUE limit ({mue})",
                        f"Reduce to {mue} or document medical necessity for exception",
                        denial_risk="HIGH",
                    )
                else:
                    entry["mue_validated"] = True
                    entry["mue_limit"] = mue

    def _check_lcd(self, icd, cpt):
        routine = [c for c in cpt if c.get("code", "") in ROUTINE_FOOT_CARE_CPTS]
        if not routine:
            return
        has_qual = any(self.db.is_lcd_qualifying(c.get("code", "")) for c in icd)
        if not has_qual:
            for c in routine:
                self._add(
                    "WARNING", c.get("code", ""), "lcd_coverage",
                    f"Routine foot care CPT {c.get('code')} requires a qualifying systemic DX per LCD {self.db.lcd_id}",
                    "Add qualifying DX: DM with neuropathy, PVD, etc.",
                    denial_risk="HIGH",
                )

    def _check_sequencing(self, icd):
        primaries = [c for c in icd if c.get("type") == "primary"]
        if len(primaries) > 1:
            self._add(
                "WARNING", "", "sequencing",
                "Multiple ICD-10-CM codes marked as primary — only one should be first-listed",
                "Review sequencing guidelines and designate a single primary",
                denial_risk="MEDIUM",
            )

    def _check_cpt_dx_linkage(self, cpt):
        for entry in cpt:
            if not entry.get("linked_diagnoses"):
                self._add(
                    "WARNING", entry.get("code", ""), "cpt_icd_linkage",
                    f"CPT {entry.get('code')} has no linked diagnosis code",
                    "Link at least one ICD-10-CM code for medical necessity",
                    denial_risk="MEDIUM",
                )

    def _check_orphan_dx(self, icd, cpt, hcpcs):
        """Flag ICD codes not linked to any procedure — WARNING level (affects tier)."""
        linked = set()
        for c in cpt + hcpcs:
            for dx in c.get("linked_diagnoses", []):
                d = dx if isinstance(dx, str) else dx.get("code", "")
                linked.add(d.replace(".", "").upper())

        for entry in icd:
            code = entry.get("code", "").replace(".", "").upper()
            if code not in linked:
                self._add(
                    "WARNING", entry.get("code", ""), "ORPHAN_DIAGNOSIS",
                    f"{entry.get('code')} is not linked to any CPT or HCPCS code — may appear unsupported",
                    "Link to a CPT/HCPCS code or remove if not addressed today",
                    denial_risk="MEDIUM",
                )

    def _check_modifiers(self, cpt):
        for entry in cpt:
            for mod in entry.get("modifiers", []):
                if mod not in VALID_MODIFIERS:
                    self._add(
                        "WARNING", entry.get("code", ""), "modifier_validity",
                        f"Modifier '{mod}' is not a recognized modifier for CPT {entry.get('code')}",
                        "Verify modifier is appropriate for this service",
                        denial_risk="MEDIUM",
                    )

    def _check_em_modifier25(self, cpt):
        """
        Two-directional modifier -25 check:
        1. MISSING: E/M + billable procedure but no -25
        2. INCORRECT: E/M has -25 but only imaging (no billable procedure) was performed
        """
        em_entry = None
        has_billable_procedure = False
        has_imaging_only = False

        for c in cpt:
            code = c.get("code", "")
            if code in EM_CODES:
                em_entry = c
            elif code.startswith(IMAGING_PREFIXES):
                has_imaging_only = True
            elif code not in POST_OP_EM:
                # Any non-E/M, non-imaging, non-post-op code = billable procedure
                has_billable_procedure = True

        if em_entry is None:
            return

        em_code = em_entry.get("code", "")
        has_mod25 = "25" in em_entry.get("modifiers", [])

        # Case 1: Modifier -25 present but NO billable procedure (only imaging)
        if has_mod25 and not has_billable_procedure:
            self._add(
                "WARNING", em_code, "modifier_25_incorrect",
                f"Modifier -25 on {em_code} without a same-day billable procedure. "
                f"Diagnostic imaging (73xxx) does NOT trigger modifier -25.",
                "Remove modifier -25 if no same-day billable procedure was performed",
                denial_risk="MEDIUM",
            )

        # Case 2: Billable procedure present but modifier -25 MISSING
        if has_billable_procedure and not has_mod25:
            self._add(
                "WARNING", em_code, "em_procedure_same_day",
                f"E/M {em_code} billed with a same-day procedure but modifier -25 is missing",
                "Add modifier -25 if E/M was separately identifiable beyond the procedure decision",
                denial_risk="MEDIUM",
            )

    def _check_global_period(self, cpt, prior_surgery_info: dict | None):
        """
        Detect when a billable E/M is billed during a prior surgery's global period.
        The correct code during global period is 99024 (post-op follow-up, no charge).
        """
        if not prior_surgery_info or not prior_surgery_info.get("is_post_op_visit"):
            return

        days_post_op = prior_surgery_info.get("days_post_op")
        prior_cpt = prior_surgery_info.get("prior_surgery_cpt", "")
        prior_desc = prior_surgery_info.get("prior_surgery_description", "prior surgery")

        if days_post_op is None or not prior_cpt:
            # Can't determine global period without days and prior CPT
            # Flag as INFO for human review
            self._add(
                "INFO", "", "global_period",
                f"Post-operative visit detected but could not determine days post-op or prior CPT. "
                f"Verify this visit does not fall within a global surgical period.",
                "Manually confirm global period status before submission",
                denial_risk="MEDIUM",
            )
            return

        global_days = self.db.get_global_period(prior_cpt)

        if global_days == 0:
            # No global period concern
            return

        if days_post_op <= global_days:
            # Visit falls within global period — billable E/M should be 99024
            billable_em = [c for c in cpt if c.get("code", "") in EM_CODES]
            if billable_em:
                for em in billable_em:
                    self._add(
                        "ERROR", em.get("code", ""), "global_period",
                        f"E/M {em.get('code')} billed on post-op day {days_post_op} for {prior_desc} "
                        f"(CPT {prior_cpt}, {global_days}-day global period). "
                        f"This visit is included in the surgical package — use 99024 instead.",
                        "Replace with CPT 99024 (post-operative follow-up visit, no charge to patient)",
                        denial_risk="HIGH",
                    )

    def _check_redundant_dm_codes(self, icd):
        """
        Flag when both a DM generic code (E11.9/E10.9) AND a more specific DM
        combination code are present — they cannot both be coded for the same encounter.
        """
        codes = [c.get("code", "") for c in icd]
        has_dm_generic = any(c in ("E11.9", "E10.9", "E13.9") for c in codes)
        has_dm_specific = any(c.startswith(pfx) for c in codes for pfx in DM_COMBINATION_PREFIXES)

        if has_dm_generic and has_dm_specific:
            generic = next((c for c in codes if c in ("E11.9", "E10.9", "E13.9")), "")
            self._add(
                "ERROR", generic, "redundant_dm_code",
                f"{generic} (unspecified DM) coded alongside a more specific DM combination code. "
                f"Per ICD-10-CM guidelines, do not code both — the combination code captures the DM.",
                f"Remove {generic} — the specific DM combination code already captures the diabetes",
                denial_risk="MEDIUM",
            )

    def _check_inappropriate_zcodes(self, icd):
        """Flag long-term drug Z-codes that are inappropriate for outpatient podiatry encounters."""
        for entry in icd:
            code = entry.get("code", "").replace(".", "")
            # Rebuild with dot for comparison
            dotted = entry.get("code", "")
            if dotted in INAPPROPRIATE_LONGTERM_ZCODES:
                self._add(
                    "WARNING", dotted, "inappropriate_zcode",
                    f"{dotted} (long-term drug therapy code) is not appropriate for outpatient podiatry E/M. "
                    f"The underlying condition codes (E11.x, I10, E78.5) are sufficient.",
                    f"Remove {dotted} — not required for outpatient podiatry billing",
                    denial_risk="LOW",
                )

    def _check_snomed_consistency(self, snomed):
        """
        Two checks:
        1. Duplicate concept_id assigned to different entity texts (label drift)
        2. Root/parent concepts used instead of specific descendants (low confidence)
        """
        seen_ids: dict[str, str] = {}  # concept_id → first entity_text seen

        for entry in snomed:
            concept_id = str(entry.get("concept_id", "")).strip()
            entity_text = entry.get("entity_text", "")
            description = entry.get("description", "")

            # Check for duplicate concept_id with different label
            if concept_id in seen_ids:
                if seen_ids[concept_id] != entity_text:
                    self._add(
                        "WARNING", concept_id, "snomed_label_drift",
                        f"SNOMED concept {concept_id} ({description}) assigned to two different terms: "
                        f'"{seen_ids[concept_id]}" and "{entity_text}". One mapping is incorrect.',
                        "Review both SNOMED mappings and correct the wrong one",
                        denial_risk="LOW",
                    )
            else:
                seen_ids[concept_id] = entity_text

            # Check for root/generic concepts
            if self.db.is_snomed_root(concept_id):
                root_label = self.db.get_snomed_root_label(concept_id)
                current_conf = entry.get("confidence", 1.0)
                if current_conf > self.db.snomed_root_confidence_cap:
                    entry["confidence"] = self.db.snomed_root_confidence_cap
                    entry["is_root_concept"] = True
                self._add(
                    "WARNING", concept_id, "snomed_root_concept",
                    f"SNOMED {concept_id} is a top-level parent concept ({root_label}) — "
                    f"too generic for clinical coding. A specific descendant should be used.",
                    f"Find a more specific SNOMED concept for '{entity_text}'",
                    denial_risk="LOW",
                )

    # --- Encounter integrity ---

    def _encounter_integrity(self, icd) -> EncounterIntegrity:
        issues = []
        man_idx = etio_idx = None
        for idx, entry in enumerate(icd):
            prefix = entry.get("code", "").replace(".", "")[:3]
            if prefix in MANIFESTATION_PREFIXES and man_idx is None:
                man_idx = idx
            if prefix in ETIOLOGY_PREFIXES and etio_idx is None:
                etio_idx = idx
        if man_idx is not None and etio_idx is not None and man_idx < etio_idx:
            issues.append({
                "type": "SEQUENCING_ERROR",
                "severity": "WARNING",
                "message": (
                    f"ICD {icd[man_idx].get('code')} (manifestation) is sequenced before "
                    f"etiology ({icd[etio_idx].get('code')})"
                ),
            })
        errors = sum(1 for i in issues if i.get("severity") == "ERROR")
        warns = sum(1 for i in issues if i.get("severity") == "WARNING")
        return EncounterIntegrity(encounter_issues=issues, error_count=errors, warning_count=warns)

    # --- Documentation audit ---

    def _documentation_audit(self, coding_result: dict) -> DocumentationAudit:
        entries = []
        for key in ["icd10_codes", "cpt_codes", "hcpcs_codes"]:
            for entry in coding_result.get(key, []):
                support = []
                rationale = entry.get("rationale", "")
                if rationale:
                    support.append(f"Rationale: {rationale}")

                src = entry.get("source_section", "") or entry.get("source", "")
                if src:
                    support.append(f"Section: {src}")

                mdm = entry.get("mdm_details", {})
                if mdm and mdm.get("mdm_level"):
                    p = mdm.get("problems_score", mdm.get("problem_score", "?"))
                    d = mdm.get("data_score", "?")
                    r = mdm.get("risk_score", "?")
                    support.append(f"MDM: {mdm.get('mdm_level')} (problems_score: {p} / data_score: {d} / risk_score: {r})")
                    linked = entry.get("linked_diagnoses", [])
                    if linked:
                        support.append(f"Linked DX: {', '.join(linked)}")

                supporting_text = entry.get("supporting_text", "")
                if supporting_text:
                    support.append(f"Supporting text: {supporting_text}")

                gaps = []
                if not support:
                    gaps.append("No documentation evidence found for this code")

                entries.append({
                    "code": entry.get("code", ""),
                    "description": entry.get("description", ""),
                    "documentation_support": support,
                    "documentation_gaps": gaps,
                    "supported": len(support) > 0,
                })

        total = len(entries)
        fully_supported = sum(1 for e in entries if e["supported"] and not e["documentation_gaps"])
        partially_supported = sum(1 for e in entries if e["supported"] and e["documentation_gaps"])
        unsupported = sum(1 for e in entries if not e["supported"])
        with_gaps = sum(1 for e in entries if e["documentation_gaps"])

        return DocumentationAudit(
            audit_entries=entries,
            total_codes=total,
            fully_supported=fully_supported,
            partially_supported=partially_supported,
            unsupported=unsupported,
            codes_with_gaps=with_gaps,
            documentation_score=round((fully_supported + 0.5 * partially_supported) / total, 2) if total else 1.0,
        )

    # --- Tier scoring ---

    def _audit_score(self) -> float:
        errors = sum(1 for i in self.issues if i.severity == "ERROR")
        warns = sum(1 for i in self.issues if i.severity == "WARNING")
        infos = sum(1 for i in self.issues if i.severity == "INFO")
        return max(0.0, round(1.0 - errors * 0.15 - warns * 0.05 - infos * 0.01, 2))

    def _compute_tier(self, icd, cpt, hcpcs) -> tuple[str, float, list[str]]:
        """
        Tier rules (enforced strictly):
        - Any ERROR → REJECT
        - Any WARNING → REVIEW
        - INFO only → AUTO (with high confidence)
        - Zero issues → AUTO (highest confidence)
        """
        errors = [i for i in self.issues if i.severity == "ERROR"]
        warnings = [i for i in self.issues if i.severity == "WARNING"]
        infos = [i for i in self.issues if i.severity == "INFO"]

        if errors:
            tier, base = "REJECT", 0.3
        elif warnings:
            tier, base = "REVIEW", 0.75
        elif infos:
            tier, base = "AUTO", 0.90
        else:
            tier, base = "AUTO", 0.97

        confs = [c.get("confidence", 0.5) for lst in [icd, cpt, hcpcs] for c in lst]
        avg = sum(confs) / len(confs) if confs else 0.5
        conf = round(min(base, avg), 2)

        # Build human-readable review reasons
        reasons = []
        for i in errors:
            reasons.append(f"[ERROR] {i.code}: {i.message}")
        for i in warnings:
            reasons.append(f"[WARNING] {i.code}: {i.message}")

        return tier, conf, reasons

    def _summary(self, tier: str, reasons: list[str]) -> str:
        if tier == "AUTO" and not reasons:
            return "All validation checks passed. Ready for submission."
        if tier == "AUTO":
            return f"Auto-codeable with {len(reasons)} informational note(s). Review before submission."
        if tier == "REVIEW":
            return f"Coder review required — {len(reasons)} issue(s) need resolution before submission."
        return f"Claim rejected — {len(reasons)} critical error(s) must be corrected before submission."

    # --- Helper ---

    def _add(self, severity, code, category, message, recommendation, denial_risk=None):
        self.issues.append(ValidationIssue(
            severity=severity,
            code=str(code),
            category=category,
            message=message,
            recommendation=recommendation,
            denial_risk=denial_risk,
        ))
