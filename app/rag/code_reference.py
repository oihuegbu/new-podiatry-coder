import json
from app.core.config import ICD10_FILE, CPT_FILE, HCPCS_FILE, NCCI_FILE, MUE_FILE, LCD_FILE
from app.core.logger import get_logger

logger = get_logger(__name__)


class CodeReferenceDB:
    """In-memory lookup tables for code validation (existence, NCCI, MUE, LCD)."""

    def __init__(self):
        self.icd10: dict[str, dict] = {}
        self.cpt: dict[str, dict] = {}
        self.hcpcs: dict[str, dict] = {}
        self.ncci: dict[str, dict] = {}
        self.mue: dict[str, dict] = {}
        self.lcd_qualifying_dx: list[str] = []
        self.lcd_id: str = ""

    def load_all(self) -> None:
        self._load_icd10()
        self._load_cpt()
        self._load_hcpcs()
        self._load_ncci()
        self._load_mue()
        self._load_lcd()

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
            self.ncci[f"{c1}|{c2}"] = {
                "code1": c1,
                "code2": c2,
                "edit_type": entry.get("edit_type", ""),
                "modifier": entry.get("modifier", ""),
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
