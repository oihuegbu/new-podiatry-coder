from app.rag.code_reference import CodeReferenceDB
from app.models.schemas import ValidationIssue, DocumentationAudit, EncounterIntegrity
from app.core.logger import get_logger

logger = get_logger(__name__)

EM_CODES = {"99202", "99203", "99204", "99205", "99211", "99212", "99213", "99214", "99215"}
ROUTINE_FOOT_CARE_CPTS = {"11719", "11720", "11721", "11055", "11056", "11057"}
VALID_MODIFIERS = {
    "25", "59", "XE", "XS", "XP", "XU", "50", "51", "76", "77",
    "LT", "RT", "TA", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
    "26", "TC", "47", "80", "81", "82", "AS", "QW", "QX", "QY", "QZ",
}
MANIFESTATION_PREFIXES = {"H36", "G63", "N08", "M14"}
ETIOLOGY_PREFIXES = {"E10", "E11", "E13", "I70", "I73"}


class CodingValidator:
    def __init__(self, ref_db: CodeReferenceDB):
        self.db = ref_db
        self.issues: list[ValidationIssue] = []

    def validate(self, coding_result: dict) -> dict:
        self.issues = []
        icd = coding_result.get("icd10_codes", [])
        cpt = coding_result.get("cpt_codes", [])
        hcpcs = coding_result.get("hcpcs_codes", [])

        self._check_code_existence(icd, cpt, hcpcs)
        self._check_ncci(cpt)
        self._check_mue(cpt)
        self._check_lcd(icd, cpt)
        self._check_sequencing(icd)
        self._check_cpt_dx_linkage(cpt)
        self._check_orphan_dx(icd, cpt, hcpcs)
        self._check_modifiers(cpt)
        self._check_em_same_day(cpt)

        enc = self._encounter_integrity(icd)
        audit = self._documentation_audit(coding_result)
        tier, confidence, reasons = self._compute_tier(icd, cpt, hcpcs)

        return {
            "validation_issues": [i.model_dump() for i in self.issues],
            "encounter_integrity": enc.model_dump(),
            "documentation_audit": audit.model_dump(),
            "pre_submission_audit_findings": [i.model_dump() for i in self.issues],
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
                self._add("ERROR", code, "code_existence",
                          f"ICD-10-CM {code} not found in FY2026 code set",
                          "Verify code or use a valid alternative")

        for entry in cpt:
            code = entry.get("code", "")
            if self.db.validate_cpt(code):
                entry["ama_validated"] = True
            else:
                self._add("ERROR", code, "code_existence",
                          f"CPT {code} not found in code set",
                          "Verify code or use a valid alternative")

        for entry in hcpcs:
            code = entry.get("code", "")
            if code and not self.db.validate_hcpcs(code):
                self._add("INFO", code, "code_existence",
                          f"HCPCS {code} not found in database (may still be valid)",
                          "Verify HCPCS code validity")

    def _check_ncci(self, cpt):
        codes = [c.get("code", "") for c in cpt if c.get("code")]
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                conflict = self.db.check_ncci(codes[i], codes[j])
                if conflict:
                    mod_ok = conflict.get("modifier", "") in ("1", "9")
                    has_sep = any(
                        any(m in ("59", "XE", "XS", "XP", "XU") for m in c.get("modifiers", []))
                        for c in cpt if c.get("code") in (codes[i], codes[j])
                    )
                    if mod_ok and has_sep:
                        self._add("INFO", f"{codes[i]}|{codes[j]}", "ncci_edit",
                                  f"NCCI pair {codes[i]}/{codes[j]} resolved via modifier", "")
                    else:
                        sev = "WARNING" if mod_ok else "ERROR"
                        self._add(sev, f"{codes[i]}|{codes[j]}", "ncci_edit",
                                  f"NCCI conflict: {codes[i]} and {codes[j]}",
                                  "Add modifier 59/X if distinct, or remove one code" if mod_ok
                                  else "Mutually exclusive — remove one")

    def _check_mue(self, cpt):
        for entry in cpt:
            code = entry.get("code", "")
            units = entry.get("units", 1)
            mue = self.db.get_mue(code)
            if mue is not None and mue > 0:
                if units > mue:
                    self._add("ERROR", code, "mue_limit",
                              f"CPT {code} units ({units}) exceeds MUE ({mue})",
                              f"Reduce to {mue} or document exception")
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
                self._add("WARNING", c.get("code", ""), "lcd_coverage",
                          f"Routine foot care CPT {c.get('code')} needs qualifying DX per LCD {self.db.lcd_id}",
                          "Add qualifying systemic condition")

    def _check_sequencing(self, icd):
        primaries = [c for c in icd if c.get("type") == "primary"]
        if len(primaries) > 1:
            self._add("WARNING", "", "sequencing",
                      "Multiple codes marked as primary diagnosis",
                      "Only one should be primary/first-listed")

    def _check_cpt_dx_linkage(self, cpt):
        for entry in cpt:
            if not entry.get("linked_diagnoses"):
                self._add("WARNING", entry.get("code", ""), "cpt_icd_linkage",
                          f"CPT {entry.get('code')} has no linked diagnosis",
                          "Link at least one ICD-10-CM for medical necessity")

    def _check_orphan_dx(self, icd, cpt, hcpcs):
        linked = set()
        for c in cpt + hcpcs:
            for dx in c.get("linked_diagnoses", []):
                d = dx if isinstance(dx, str) else dx.get("code", "")
                linked.add(d.replace(".", ""))
        for entry in icd:
            code = entry.get("code", "").replace(".", "")
            if code not in linked:
                self._add("INFO", entry.get("code", ""), "ORPHAN_DIAGNOSIS",
                          f"{entry.get('code')} not linked to any procedure",
                          "Link to CPT/HCPCS or remove if not addressed today",
                          denial_risk="LOW")

    def _check_modifiers(self, cpt):
        for entry in cpt:
            for mod in entry.get("modifiers", []):
                if mod not in VALID_MODIFIERS:
                    self._add("WARNING", entry.get("code", ""), "modifier_validity",
                              f"Modifier '{mod}' may not be valid on CPT {entry.get('code')}",
                              "Verify modifier is appropriate")

    def _check_em_same_day(self, cpt):
        em = None
        has_proc = False
        for c in cpt:
            code = c.get("code", "")
            if code in EM_CODES:
                em = c
            elif not code.startswith("7"):
                has_proc = True
        if em and has_proc and "25" not in em.get("modifiers", []):
            self._add("WARNING", em.get("code", ""), "em_procedure_same_day",
                      f"E/M {em.get('code')} with procedure on same day needs modifier 25",
                      "Add modifier 25")

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
                "message": f"ICD {icd[man_idx].get('code')} (manifestation) before etiology ({icd[etio_idx].get('code')})",
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
                for span in entry.get("evidence_spans", []):
                    support.append(f'Entity: "{span}"')
                if entry.get("rationale"):
                    support.append(f"Rationale: {entry['rationale']}")
                if entry.get("source_section"):
                    support.append(f"Section: {entry['source_section']}")
                mdm = entry.get("mdm_details", {})
                if mdm and mdm.get("mdm_level"):
                    support.append(f"MDM: {mdm.get('mdm_level')} (P:{mdm.get('problem_score')}/D:{mdm.get('data_score')}/R:{mdm.get('risk_score')})")
                for dx in entry.get("linked_diagnoses", []):
                    support.append(f"Linked DX: {dx}")
                entries.append({
                    "code": entry.get("code", ""),
                    "description": entry.get("description", ""),
                    "documentation_support": support,
                    "documentation_gaps": [] if support else ["No documentation evidence"],
                    "supported": len(support) > 0,
                })
        total = len(entries)
        supported = sum(1 for e in entries if e["supported"])
        return DocumentationAudit(
            audit_entries=entries,
            total_codes=total,
            fully_supported=supported,
            unsupported=total - supported,
            codes_with_gaps=sum(1 for e in entries if e["documentation_gaps"]),
            documentation_score=round(supported / total, 2) if total else 1.0,
        )

    # --- Scoring ---

    def _audit_score(self) -> float:
        errors = sum(1 for i in self.issues if i.severity == "ERROR")
        warns = sum(1 for i in self.issues if i.severity == "WARNING")
        infos = sum(1 for i in self.issues if i.severity == "INFO")
        return max(0.0, round(1.0 - errors * 0.2 - warns * 0.05 - infos * 0.01, 2))

    def _compute_tier(self, icd, cpt, hcpcs) -> tuple[str, float, list[str]]:
        errors = sum(1 for i in self.issues if i.severity == "ERROR")
        warns = sum(1 for i in self.issues if i.severity == "WARNING")

        if errors > 0:
            tier, base = "REJECT", 0.3
        elif warns > 2:
            tier, base = "REVIEW", 0.6
        elif warns > 0:
            tier, base = "REVIEW", 0.8
        else:
            tier, base = "AUTO", 0.95

        confs = [c.get("confidence", 0.5) for lst in [icd, cpt, hcpcs] for c in lst]
        avg = sum(confs) / len(confs) if confs else 0.5
        conf = round(min(base, avg), 2)

        reasons = [
            f"{i.code}: {i.message}" for i in self.issues if i.severity in ("ERROR", "WARNING")
        ]
        return tier, conf, reasons

    def _summary(self, tier: str, reasons: list[str]) -> str:
        if tier == "AUTO":
            return "Auto-codeable. Ready for submission."
        return f"Coder review recommended ({len(reasons)} items). Address review reasons before submission."

    # --- Helper ---

    def _add(self, severity, code, category, message, recommendation, denial_risk=None):
        self.issues.append(ValidationIssue(
            severity=severity, code=code, category=category,
            message=message, recommendation=recommendation,
            denial_risk=denial_risk,
        ))
