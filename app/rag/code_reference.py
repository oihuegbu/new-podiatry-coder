import json
import sqlite3
import calendar
from datetime import date
from app.core.config import (
    ICD10_FILE, CPT_FILE, HCPCS_FILE, MUE_FILE,
    SNOMED_ROOTS_FILE, DATA_DIR,
)
from app.core.logger import get_logger

logger = get_logger(__name__)

_COMPLIANCE_DB_PATH = DATA_DIR / "compliance.db"
_OPEN = "9999-12-31"


def _norm(code: str) -> str:
    return (code or "").replace(".", "").strip().upper()


def _clean_date(val) -> str:
    """Normalize a date-ish value to 'YYYY-MM-DD'; empty/None -> wide-open
    start. Also accepts CMS's compact 'YYYYMMDD' form (cpt_codes.json's own
    effective_date field, e.g. '20240101')."""
    import re
    if not val:
        return "1900-01-01"
    s = str(val).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    if re.match(r"^\d{8}$", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return "1900-01-01"


class CodeReferenceDB:
    """In-memory lookup tables for code validation (existence, MUE, SNOMED roots).

    LCD/Article medical-necessity coverage lives entirely in ComplianceDataStore's
    coverage_cpt/coverage_icd tables (MedicalNecessityAgent) — not duplicated here.

    NCCI pairs are the one exception: at 1.7M+ rows they're queried from the
    SQLite compliance.db (built by ComplianceDataStore) instead of being held
    in memory here — see check_ncci().

    Global period data (glob_days, billing_status, bilat_surg) lives entirely
    in ComplianceDataStore's global_period table (store.global_period() /
    store.billing_status() / store.bilat_surg()) — not duplicated here. A
    prior in-memory copy (get_global_period()) collapsed XXX/YYY/ZZZ/MMM and
    "code not found" to the same 0 value, making "no global period" and
    "unknown code" indistinguishable; removed in favor of the single
    real-data source already used everywhere else in the codebase.
    """

    def __init__(self):
        self.icd10: dict[str, dict] = {}
        self.cpt: dict[str, dict] = {}
        self.hcpcs: dict[str, dict] = {}
        self.mue: dict[str, dict] = {}
        self.snomed_roots: dict[str, str] = {}
        self.snomed_root_confidence_cap: float = 0.4
        self._ncci_release_window_loaded = False
        self._ncci_release_window: tuple[date, date] | None = None

    def load_all(self) -> None:
        # Treat one load as one authoritative data snapshot. A subsequent
        # load may follow a compliance.db refresh and must rediscover its
        # NCCI release window.
        self._ncci_release_window_loaded = False
        self._ncci_release_window = None
        self._load_icd10()
        self._load_cpt()
        self._load_hcpcs()
        self._load_ncci()
        self._load_mue()
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
                    "effective_from": _clean_date(entry.get("effective_from")),
                    "effective_to": _clean_date(entry.get("effective_to")) if entry.get("effective_to") else _OPEN,
                    "temporal_authority": True,
                }
        logger.info(f"Loaded {len(self.icd10)} ICD-10-CM codes")

    def _load_cpt(self) -> None:
        with open(CPT_FILE) as f:
            data = json.load(f)
        codes_list = data.get("codes", data) if isinstance(data, dict) else data
        for entry in codes_list:
            code = entry.get("code", "").strip()
            if code:
                # CPT's effective_date is NOT a code-introduction date —
                # verified empirically: 99202/99213/99214 (decades-old,
                # unquestionably not new codes) all carry effective_date
                # 2024-01-01, and the full distribution of populated values
                # clusters on Jan 1/Jul 1 across 2020-2026 — a periodic
                # descriptor-revision cycle marker, not a lifecycle date.
                # Using it as an activation gate would flag huge numbers of
                # long-standing, merely-reworded codes (e.g. any E/M code
                # touched by the 2021 guideline overhaul) as "not active"
                # for any older, perfectly valid date of service. No
                # reliable introduction/retirement signal exists in this
                # source for CPT, so — unlike ICD-10 and HCPCS, both
                # verified against real add/discontinue signals — CPT stays
                # always-open rather than approximating with the wrong field.
                self.cpt[code] = {
                    "code": code,
                    "short_description": entry.get("short_description", ""),
                    "long_description": entry.get("long_description", ""),
                    "effective_from": "1900-01-01",
                    "effective_to": _OPEN,
                    # Presence in this snapshot proves identity, not that the
                    # code was active on an arbitrary historical DOS.  The
                    # release gate must not turn the wide-open compatibility
                    # window above into a temporal-authority assertion.
                    "temporal_authority": False,
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
                    # add_date, not effective_from — see store.py's
                    # _ingest_hcpcs for the same fix and why (effective_from
                    # shifts on every quarterly revision cycle even for
                    # decades-old codes; add_date stays stable at the
                    # code's true origin). effective_to stays reliable for
                    # discontinuation (action_code='D' entries).
                    eff_to = entry.get("effective_to")
                    self.hcpcs[code] = {
                        "code": code,
                        "description": raw_code[5:].strip() or entry.get("short_description", ""),
                        # The full CMS descriptor carries the billing-unit
                        # denomination for drugs ("..., 10 mg" = one unit per
                        # 10 mg) that the truncated short description loses —
                        # needed by the validator's J-code units cross-check.
                        "long_description": entry.get("long_description", ""),
                        "effective_from": _clean_date(entry.get("add_date")),
                        "effective_to": _clean_date(eff_to) if eff_to else _OPEN,
                        "temporal_authority": True,
                    }
        logger.info(f"Loaded {len(self.hcpcs)} HCPCS codes")

    def _load_ncci(self) -> None:
        # NCCI pairs (1.7M+ rows) are queried from the SQLite compliance.db
        # built by ComplianceDataStore instead of being duplicated here as an
        # in-memory dict (~700MB) — see check_ncci().
        logger.info("NCCI edit pairs: served from compliance.db (no in-memory duplicate)")

    def _load_mue(self) -> None:
        with open(MUE_FILE) as f:
            data = json.load(f)
        for entry in data:
            code = entry.get("code", "").strip()
            if code:
                self.mue[code] = {"mue_value": entry.get("mue_value", 0)}
        logger.info(f"Loaded {len(self.mue)} MUE entries")

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
        """Existence at any point in time (no date dimension) — unchanged
        behavior, still the right check for callers asking "is this a real
        code at all" (e.g. hallucination detection) rather than "was it
        valid on this specific date." See is_active_for_dos() for the
        date-gated check, mirroring ComplianceDataStore's own
        code_exists(dos) vs code_active_any_date() split."""
        return self.icd10.get(code.replace(".", "").strip())

    def validate_cpt(self, code: str) -> dict | None:
        return self.cpt.get(code.strip())

    def validate_hcpcs(self, code: str) -> dict | None:
        return self.hcpcs.get(code.strip())

    def is_active_for_dos(self, system: str, code: str, dos=None) -> bool:
        """True if `code` was effective on `dos` (defaults to today).
        Previously validate_icd10/cpt/hcpcs had no date dimension at all —
        a not-yet-effective or already-discontinued code validated as
        existing regardless of the claim's actual date of service. Real
        effective_from/effective_to are now stored per entry (see
        _load_icd10/_load_cpt/_load_hcpcs); this is the date-gated
        companion to validate_*()'s existence-only check."""
        table = {"icd10": self.icd10, "cpt": self.cpt, "hcpcs": self.hcpcs}.get(system)
        if table is None:
            return False
        norm = code.replace(".", "").strip() if system == "icd10" else code.strip()
        entry = table.get(norm)
        if entry is None:
            return False
        d = dos if isinstance(dos, str) else (dos.isoformat() if dos else date.today().isoformat())
        return entry["effective_from"] <= d <= entry["effective_to"]

    def _ncci_release_bounds(self) -> tuple[date, date] | None:
        """Return the loaded NCCI snapshot quarter, querying SQLite once."""
        if self._ncci_release_window_loaded:
            return self._ncci_release_window
        conn = sqlite3.connect(f"file:{_COMPLIANCE_DB_PATH}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT MAX(effective_from) FROM ncci_ptp",
            ).fetchone()
        finally:
            conn.close()
        bounds = None
        if row and row[0]:
            try:
                release_date = date.fromisoformat(row[0])
            except (TypeError, ValueError):
                pass
            else:
                quarter_index = (release_date.month - 1) // 3
                start = date(release_date.year, quarter_index * 3 + 1, 1)
                end_month = (quarter_index + 1) * 3
                bounds = (
                    start,
                    date(start.year, end_month,
                         calendar.monthrange(start.year, end_month)[1]),
                )
        self._ncci_release_window = bounds
        self._ncci_release_window_loaded = True
        return bounds

    def ncci_data_available(self, dos=None) -> bool:
        """Return whether the loaded quarterly NCCI snapshot covers DOS."""
        if dos is None:
            return False
        try:
            d = date.fromisoformat(dos) if isinstance(dos, str) else dos
        except (TypeError, ValueError):
            return False
        bounds = self._ncci_release_bounds()
        if bounds is None:
            return False
        start, end = bounds
        return start <= d <= end

    def check_ncci(self, code1: str, code2: str, dos=None) -> dict | None:
        """Look up an NCCI PTP pair from the release covering the claim DOS."""
        if not self.ncci_data_available(dos):
            return None
        d = dos if isinstance(dos, str) else dos.isoformat()
        c1, c2 = _norm(code1), _norm(code2)
        conn = sqlite3.connect(f"file:{_COMPLIANCE_DB_PATH}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT col1, col2, modifier_indicator FROM ncci_ptp "
                "WHERE ((col1=? AND col2=?) OR (col1=? AND col2=?)) "
                "AND effective_from<=? AND effective_to>=? "
                "ORDER BY effective_from DESC LIMIT 1",
                (c1, c2, c2, c1, d, d),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        row_c1, row_c2, modifier = row
        return {"code1": row_c1, "code2": row_c2, "edit_type": "PTP", "modifier": modifier}

    def get_mue(self, code: str) -> int | None:
        entry = self.mue.get(code.strip())
        return entry["mue_value"] if entry else None

    def icd10_siblings(self, prefix: str) -> list[tuple[str, str]]:
        """Every real ICD-10-CM code sharing the given category prefix
        (e.g. "Z88" -> all 10 Z88.x drug-allergy-category codes), with
        descriptions — the ICD-10 equivalent of a CPT code family. Unlike
        CPT (which shares a literal descriptor stem before a semicolon),
        ICD-10 siblings share only a numeric/alpha category prefix; each
        sibling's full description differs entirely (e.g. Z88.5 "Allergy
        status to narcotic agent" vs Z88.6 "...analgesic agent")."""
        norm_prefix = prefix.replace(".", "").strip().upper()
        return sorted(
            (code, entry["description"])
            for code, entry in self.icd10.items()
            if code.startswith(norm_prefix)
        )

    def is_snomed_root(self, concept_id: str) -> bool:
        """Return True if the SNOMED concept ID is a generic root/parent concept."""
        return str(concept_id).strip() in self.snomed_roots

    def get_snomed_root_label(self, concept_id: str) -> str | None:
        return self.snomed_roots.get(str(concept_id).strip())
