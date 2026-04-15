import json
from app.core.config import (
    ICD10_FILE, CPT_FILE, HCPCS_FILE, NCCI_FILE, MUE_FILE, LCD_FILE,
    GLOBAL_PERIODS_FILE, SNOMED_ROOTS_FILE,
)
from app.core.logger import get_logger

logger = get_logger(__name__)


class CodeReferenceDB:
    """In-memory lookup tables for code validation (existence, NCCI, MUE, LCD, global periods, SNOMED roots)."""

    def __init__(self):
        self.icd10: dict[str, dict] = {}
        self.cpt: dict[str, dict] = {}
        self.hcpcs: dict[str, dict] = {}
        self.ncci: dict[str, dict] = {}
        self.mue: dict[str, dict] = {}
        self.lcd_qualifying_dx: list[str] = []
        self.lcd_id: str = ""
        self.global_periods: dict[str, int] = {}
        self.global_period_defaults: dict[str, int] = {}
        self.snomed_roots: dict[str, str] = {}
        self.snomed_root_confidence_cap: float = 0.4

    def load_all(self) -> None:
        self._load_icd10()
        self._load_cpt()
        self._load_hcpcs()
        self._load_ncci()
        self._load_mue()
        self._load_lcd()
        self._load_global_periods()
        self._load_snomed_roots()

    def _load_icd10(self) -> None:
        with open(ICD10_FILE) as f:
            data = json.load(f)
        for entry in data:
            code = entry.get("code", "").strip()
            if code:
                self.icd10[code] = {
                    "code": code,
                    "description": entry.get("description", ""),
                    "status": entry.get("status", "active"),
                }
        logger.info(f"Loaded {len(self.icd10)} ICD-10-CM codes")

    def _load_cpt(self) -> None:
        with open(CPT_FILE) as f:
            data = json.load(f)
        codes_list = data.get("codes", data) if isinstance(data, dict) else data
        for entry in codes_list:
            code = entry.get("code", "").strip()
            if code:
                self.cpt[code] = {
                    "code": code,
                    "short_description": entry.get("short_description", ""),
                    "long_description": entry.get("long_description", ""),
                }
        logger.info(f"Loaded {len(self.cpt)} CPT codes")

    def _load_hcpcs(self) -> None:
        with open(HCPCS_FILE) as f:
            data = json.load(f)
        for entry in data:
            raw_code = entry.get("code", "").strip()
            if len(raw_code) >= 5:
                code = raw_code[:5]
                if code[0].isalpha() and code[1:].isdigit():
                    self.hcpcs[code] = {
                        "code": code,
                        "description": raw_code[5:].strip() or entry.get("short_description", ""),
                    }
        logger.info(f"Loaded {len(self.hcpcs)} HCPCS codes")

    def _load_ncci(self) -> None:
        with open(NCCI_FILE) as f:
            data = json.load(f)
        for entry in data:
            c1 = entry.get("code1", "").strip()
            c2 = entry.get("code2", "").strip()
            if not c1 or not c2 or len(c1) > 7 or len(c2) > 7:
                continue
            if not any(ch.isdigit() for ch in c1):
                continue
            # The modifier indicator may be in 'modifier' or 'description' field
            # depending on the source file format. '0'=no modifier allowed,
            # '1'=modifier allowed, '9'=concept does not apply.
            mod_raw = entry.get("modifier", "") or entry.get("description", "")
            mod_indicator = str(mod_raw).strip()
            self.ncci[f"{c1}|{c2}"] = {
                "code1": c1,
                "code2": c2,
                "edit_type": entry.get("edit_type", "PTP"),
                "modifier": mod_indicator,
            }
        logger.info(f"Loaded {len(self.ncci)} NCCI edit pairs")

    def _load_mue(self) -> None:
        with open(MUE_FILE) as f:
            data = json.load(f)
        for entry in data:
            code = entry.get("code", "").strip()
            if code:
                self.mue[code] = {"mue_value": entry.get("mue_value", 0)}
        logger.info(f"Loaded {len(self.mue)} MUE entries")

    def _load_lcd(self) -> None:
        with open(LCD_FILE) as f:
            data = json.load(f)
        self.lcd_qualifying_dx = data.get("qualifying_dx", [])
        self.lcd_id = data.get("lcd_id", "L36199")
        logger.info(f"Loaded {len(self.lcd_qualifying_dx)} LCD qualifying DX codes")

    def _load_global_periods(self) -> None:
        try:
            with open(GLOBAL_PERIODS_FILE) as f:
                data = json.load(f)
            self.global_periods = {k: int(v) for k, v in data.get("codes", {}).items()}
            # Load prefix-based defaults (skip the 'note' key)
            raw_defaults = data.get("default_by_prefix", {})
            self.global_period_defaults = {
                k: int(v) for k, v in raw_defaults.items()
                if k != "note" and str(v).isdigit()
            }
            logger.info(f"Loaded {len(self.global_periods)} global period entries")
        except Exception as e:
            logger.warning(f"Could not load global periods file: {e}")

    def _load_snomed_roots(self) -> None:
        try:
            with open(SNOMED_ROOTS_FILE) as f:
                data = json.load(f)
            self.snomed_roots = data.get("root_concepts", {})
            self.snomed_root_confidence_cap = float(data.get("confidence_cap", 0.4))
            logger.info(f"Loaded {len(self.snomed_roots)} SNOMED root concept IDs")
        except Exception as e:
            logger.warning(f"Could not load SNOMED roots file: {e}")

    # --- Lookup helpers ---

    def validate_icd10(self, code: str) -> dict | None:
        return self.icd10.get(code.replace(".", "").strip())

    def validate_cpt(self, code: str) -> dict | None:
        return self.cpt.get(code.strip())

    def validate_hcpcs(self, code: str) -> dict | None:
        return self.hcpcs.get(code.strip())

    def check_ncci(self, code1: str, code2: str) -> dict | None:
        return self.ncci.get(f"{code1}|{code2}") or self.ncci.get(f"{code2}|{code1}")

    def get_mue(self, code: str) -> int | None:
        entry = self.mue.get(code.strip())
        return entry["mue_value"] if entry else None

    def is_lcd_qualifying(self, code: str) -> bool:
        clean = code.replace(".", "").strip()
        return clean in self.lcd_qualifying_dx or code in self.lcd_qualifying_dx

    def get_global_period(self, cpt_code: str) -> int:
        """Return the global period (days) for a CPT code. Returns 0 if unknown."""
        code = cpt_code.strip()
        if code in self.global_periods:
            return self.global_periods[code]
        # Fallback: prefix-based default
        for prefix, days in self.global_period_defaults.items():
            if code.startswith(prefix):
                return days
        return 0

    def is_snomed_root(self, concept_id: str) -> bool:
        """Return True if the SNOMED concept ID is a generic root/parent concept."""
        return str(concept_id).strip() in self.snomed_roots

    def get_snomed_root_label(self, concept_id: str) -> str | None:
        return self.snomed_roots.get(str(concept_id).strip())
