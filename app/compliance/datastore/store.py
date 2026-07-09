"""ComplianceDataStore — authoritative, effective-dated rules in SQLite.

Why SQLite (not the loose JSON dicts the old validator used):
  * 74K ICD + 11K CPT + thousands of NCCI/MUE rows need indexed lookups.
  * Every rule is keyed by an effective-date *range* so a claim is scrubbed
    against the rules in force on its date of service (filter #1: active-for-DOS).
  * CMS purges old quarters; we retain history as date ranges.

All ingestion parses to a canonical schema and FILTERS garbage rows loudly,
rather than silently storing them (the bug in the old loader).
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import date
from pathlib import Path

from app.core.config import (
    ICD10_FILE, CPT_FILE, HCPCS_FILE, NCCI_FILE, MUE_FILE, LCD_FILE,
    GLOBAL_PERIODS_FILE, DATA_DIR, CODES_DIR,
)

POS_FILE = CODES_DIR / "pos_codes.json"
MODIFIERS_FILE = CODES_DIR / "modifiers.json"
MODIFIER_EXEMPT_FILE = CODES_DIR / "modifier_exempt.json"
NCCI_AOC_FILE = CODES_DIR / "ncci_aoc_edits.json"

# Official CPT phrasing that designates an add-on code (Appendix D)
_ADDON_PHRASES = ("list separately in addition", "each additional", "add-on code")
from app.core.logger import get_logger

logger = get_logger(__name__)

DB_PATH = DATA_DIR / "compliance.db"

# A valid CPT/HCPCS code: 5 chars — 5 digits, 4 digits + letter (CPT III),
# or letter + 4 digits (HCPCS II). Used to reject copyright/header junk rows.
_CODE_RE = re.compile(r"^[A-Z0-9]{4}[A-Z0-9]$")
_HCPCS_RE = re.compile(r"^[A-Z]\d{4}$")
_OPEN = "9999-12-31"  # sentinel for "no end date"


def _norm(code: str) -> str:
    return (code or "").replace(".", "").strip().upper()


def _clean_date(val) -> str:
    """Normalize a date-ish value to 'YYYY-MM-DD'; empty/None → wide-open start."""
    if not val:
        return "1900-01-01"
    s = str(val).strip()
    return s if re.match(r"^\d{4}-\d{2}-\d{2}$", s) else "1900-01-01"


class ComplianceDataStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    # ----------------------------------------------------------------- build
    def build_or_load(self, force_rebuild: bool = False) -> None:
        exists = self.db_path.exists()
        if exists and not force_rebuild and self._is_populated():
            self._ensure_migrations()
            logger.info(f"ComplianceDataStore loaded ({self.db_path.name})")
            return
        logger.info("Building ComplianceDataStore (SQLite)...")
        if exists and force_rebuild:
            self.db_path.unlink()
            self._conn = None
        self._create_schema()
        self._ingest_code_set("ICD10", ICD10_FILE)
        self._ingest_cpt()
        self._ingest_hcpcs()
        self._ingest_ncci()
        self._ingest_mue()
        self._ingest_global_periods()
        self._ingest_lcd()
        self._ingest_pos()
        self._ingest_modifiers()
        self._ingest_modifier_exempt()
        self._ingest_ncci_aoc()
        self.conn.commit()
        logger.info("ComplianceDataStore built successfully")

    def _ensure_migrations(self) -> None:
        """Idempotent additive migrations for DBs built by an earlier version."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS data_source_version (
                source_id TEXT NOT NULL, effective_from TEXT, ingested_at TEXT,
                row_count INTEGER, file_name TEXT);
            CREATE TABLE IF NOT EXISTS modifier_exempt (
                code TEXT NOT NULL PRIMARY KEY,
                modifier_51_exempt INTEGER NOT NULL DEFAULT 0,
                modifier_63_exempt INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS ncci_aoc (
                code1 TEXT NOT NULL,
                code2 TEXT NOT NULL,
                modifier_indicator TEXT,
                effective_from TEXT NOT NULL DEFAULT '1900-01-01',
                effective_to   TEXT NOT NULL DEFAULT '9999-12-31');
            CREATE INDEX IF NOT EXISTS ix_ncci_aoc ON ncci_aoc(code1);
            """
        )
        self.conn.commit()

    def _is_populated(self) -> bool:
        try:
            cur = self.conn.execute("SELECT COUNT(*) c FROM code_set")
            return cur.fetchone()["c"] > 0
        except sqlite3.Error:
            return False

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            DROP TABLE IF EXISTS code_set;
            DROP TABLE IF EXISTS ncci_ptp;
            DROP TABLE IF EXISTS ncci_aoc;
            DROP TABLE IF EXISTS mue;
            DROP TABLE IF EXISTS global_period;
            DROP TABLE IF EXISTS lcd_qualifying;
            DROP TABLE IF EXISTS modifier_exempt;

            CREATE TABLE code_set (
                code_system   TEXT NOT NULL,
                code          TEXT NOT NULL,
                description   TEXT,
                effective_from TEXT NOT NULL DEFAULT '1900-01-01',
                effective_to   TEXT NOT NULL DEFAULT '9999-12-31',
                status        TEXT DEFAULT 'active'
            );
            CREATE INDEX ix_code_set ON code_set(code_system, code);

            CREATE TABLE ncci_ptp (
                col1 TEXT NOT NULL,
                col2 TEXT NOT NULL,
                modifier_indicator TEXT,         -- 0 = never unbundle, 1 = modifier allowed, 9 = n/a
                effective_from TEXT NOT NULL DEFAULT '1900-01-01',
                effective_to   TEXT NOT NULL DEFAULT '9999-12-31'
            );
            CREATE INDEX ix_ncci ON ncci_ptp(col1, col2);

            CREATE TABLE mue (
                code TEXT NOT NULL,
                mue_value INTEGER NOT NULL,
                mai TEXT,                        -- 1 = line, 2 = DOS absolute, 3 = DOS clinical
                rationale TEXT,
                effective_from TEXT NOT NULL DEFAULT '1900-01-01',
                effective_to   TEXT NOT NULL DEFAULT '9999-12-31'
            );
            CREATE INDEX ix_mue ON mue(code);

            CREATE TABLE global_period (
                code TEXT NOT NULL,
                glob_days TEXT,                  -- 000/010/090/MMM/XXX/YYY/ZZZ (stored as given)
                effective_from TEXT NOT NULL DEFAULT '1900-01-01',
                effective_to   TEXT NOT NULL DEFAULT '9999-12-31'
            );
            CREATE INDEX ix_glob ON global_period(code);

            CREATE TABLE lcd_qualifying (
                lcd_id TEXT NOT NULL,
                dx_code TEXT NOT NULL
            );
            CREATE INDEX ix_lcd ON lcd_qualifying(dx_code);

            CREATE TABLE addon (
                code TEXT NOT NULL PRIMARY KEY
            );

            CREATE TABLE pos (
                code TEXT NOT NULL PRIMARY KEY,
                name TEXT,
                facility TEXT          -- 'F' facility, 'N' non-facility
            );

            CREATE TABLE modifier (
                code TEXT NOT NULL PRIMARY KEY,
                category TEXT
            );

            -- CPT Appendix E/F: codes exempt from modifier 51 or 63.
            CREATE TABLE modifier_exempt (
                code TEXT NOT NULL PRIMARY KEY,
                modifier_51_exempt INTEGER NOT NULL DEFAULT 0,
                modifier_63_exempt INTEGER NOT NULL DEFAULT 0
            );

            -- NCCI Add-On Code (AOC) edits: paired code restrictions analogous
            -- to PTP edits, sourced from the CMS AOC table. code2='CCCCC' is a
            -- contractor wildcard meaning the edit applies to all primary codes.
            CREATE TABLE ncci_aoc (
                code1 TEXT NOT NULL,
                code2 TEXT NOT NULL,
                modifier_indicator TEXT,
                effective_from TEXT NOT NULL DEFAULT '1900-01-01',
                effective_to   TEXT NOT NULL DEFAULT '9999-12-31'
            );
            CREATE INDEX ix_ncci_aoc ON ncci_aoc(code1);

            -- Medical-necessity coverage (LCD/NCD/Article): which CPTs a policy
            -- governs and which ICDs it covers. Generalizes the old L36199 check;
            -- MCD Billing & Coding Articles ingest into the same two tables.
            CREATE TABLE coverage_cpt (
                policy_id TEXT NOT NULL,
                cpt_code  TEXT NOT NULL
            );
            CREATE INDEX ix_cov_cpt ON coverage_cpt(cpt_code);

            CREATE TABLE coverage_icd (
                policy_id TEXT NOT NULL,
                icd_code  TEXT NOT NULL
            );
            CREATE INDEX ix_cov_icd ON coverage_icd(policy_id, icd_code);

            -- Prior-authorization requirements (payer/plan/code specific).
            CREATE TABLE prior_auth_required (
                payer    TEXT NOT NULL DEFAULT 'Medicare',
                code     TEXT NOT NULL,
                note     TEXT
            );
            CREATE INDEX ix_pa ON prior_auth_required(payer, code);

            -- Provenance / history: one row per ingested source snapshot. CMS
            -- purges old quarters; we keep every snapshot we ever loaded.
            CREATE TABLE IF NOT EXISTS data_source_version (
                source_id     TEXT NOT NULL,
                effective_from TEXT,
                ingested_at   TEXT,
                row_count     INTEGER,
                file_name     TEXT
            );
            """
        )

    # --------------------------------------------------------------- ingest
    def _ingest_code_set(self, system: str, path) -> None:
        with open(path) as f:
            data = json.load(f)
        rows = []
        for e in data:
            code = _norm(e.get("code", ""))
            if not code:
                continue
            rows.append((
                system, code, e.get("description", ""),
                _clean_date(e.get("effective_from")),
                str(e.get("effective_to") or "").strip() or _OPEN,
                e.get("status", "active"),
            ))
        self.conn.executemany(
            "INSERT INTO code_set VALUES (?,?,?,?,?,?)", rows
        )
        logger.info(f"  code_set[{system}]: {len(rows)} codes")

    def _ingest_cpt(self) -> None:
        with open(CPT_FILE) as f:
            data = json.load(f)
        codes = data.get("codes", data) if isinstance(data, dict) else data
        rows, addon_rows = [], []
        for e in codes:
            code = _norm(e.get("code", ""))
            if not code:
                continue
            desc = e.get("long_description") or e.get("short_description") or e.get("description", "")
            rows.append(("CPT", code, desc, "1900-01-01", _OPEN, "active"))
            # Add-on status is derived from the official CPT descriptor phrasing
            # (Appendix D) — fully data-driven, no hardcoded code list.
            if any(p in (desc or "").lower() for p in _ADDON_PHRASES):
                addon_rows.append((code,))
        self.conn.executemany("INSERT INTO code_set VALUES (?,?,?,?,?,?)", rows)
        self.conn.executemany("INSERT OR IGNORE INTO addon VALUES (?)", addon_rows)
        logger.info(f"  code_set[CPT]: {len(rows)} codes ({len(addon_rows)} add-on codes derived)")

    def _ingest_hcpcs(self) -> None:
        with open(HCPCS_FILE) as f:
            data = json.load(f)
        rows, skipped, seen = [], 0, set()
        for e in data:
            raw = (e.get("code", "") or "").strip().upper()
            # HCPCS file is dirty (fixed-width artifacts) — extract the clean leading token
            m = re.match(r"^([A-Z]\d{4})", raw)
            if not m:
                skipped += 1
                continue
            code = m.group(1)
            if code in seen:   # source has many rows per code (modifiers/pricing) — dedup
                continue
            seen.add(code)
            desc = (e.get("short_description") or "").strip()
            rows.append(("HCPCS", code, desc, "1900-01-01", _OPEN, "active"))
        self.conn.executemany("INSERT INTO code_set VALUES (?,?,?,?,?,?)", rows)
        logger.info(f"  code_set[HCPCS]: {len(rows)} unique codes ({skipped} dirty rows skipped)")

    def _ingest_ncci(self) -> None:
        with open(NCCI_FILE) as f:
            data = json.load(f)
        rows, skipped = [], 0
        for e in data:
            c1, c2 = _norm(e.get("code1", "")), _norm(e.get("code2", ""))
            # Reject copyright/header/prose rows — keep only real code pairs
            if not (_CODE_RE.match(c1) and _CODE_RE.match(c2)):
                skipped += 1
                continue
            mod = str(e.get("modifier", "") or e.get("description", "")).strip()
            mod = mod[0] if mod[:1] in ("0", "1", "9") else ""
            rows.append((
                c1, c2, mod,
                _clean_date(e.get("effective_date")),
                str(e.get("end_date") or "").strip() or _OPEN,
            ))
        self.conn.executemany("INSERT INTO ncci_ptp VALUES (?,?,?,?,?)", rows)
        logger.info(f"  ncci_ptp: {len(rows)} pairs ({skipped} junk rows filtered)")

    def _ingest_mue(self) -> None:
        with open(MUE_FILE) as f:
            data = json.load(f)
        rows = []
        for e in data:
            code = _norm(e.get("code", ""))
            if not code:
                continue
            desc = str(e.get("description", "")).strip()
            # MAI is the leading character of the descriptor ("2 Date of Service Edit...")
            mai = desc[0] if desc[:1] in ("1", "2", "3") else ""
            rationale = desc[1:].strip(" :|") if mai else desc
            rows.append((
                code, int(e.get("mue_value", 0) or 0), mai, rationale,
                _clean_date(e.get("effective_date")),
                str(e.get("end_date") or "").strip() or _OPEN,
            ))
        self.conn.executemany("INSERT INTO mue VALUES (?,?,?,?,?,?)", rows)
        n_mai = sum(1 for r in rows if r[2])
        logger.info(f"  mue: {len(rows)} entries ({n_mai} with parsed MAI)")

    def _ingest_global_periods(self) -> None:
        try:
            with open(GLOBAL_PERIODS_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  global_period: could not load ({exc})")
            return
        rows = []
        for code, days in data.get("codes", {}).items():
            # New format: dict with {global_days, status, pctc_ind, ...}
            # Old format: bare integer/string
            if isinstance(days, dict):
                glob_days = str(days.get("global_days", "") or "").strip()
            else:
                glob_days = str(days).strip() if days is not None else ""
            if not glob_days:
                continue
            rows.append((_norm(code), glob_days, "1900-01-01", _OPEN))
        self.conn.executemany("INSERT INTO global_period VALUES (?,?,?,?)", rows)
        logger.info(f"  global_period: {len(rows)} codes")

    def _ingest_lcd(self) -> None:
        try:
            with open(LCD_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  lcd_qualifying: could not load ({exc})")
            return
        lcd_id = data.get("lcd_id", "L36199")
        dx_rows = [(lcd_id, _norm(dx)) for dx in data.get("qualifying_dx", [])]
        self.conn.executemany("INSERT INTO lcd_qualifying VALUES (?,?)", dx_rows)
        # Also feed the generalized coverage engine (filter #5)
        self.conn.executemany(
            "INSERT INTO coverage_icd VALUES (?,?)", dx_rows,
        )
        cpt_rows = [(lcd_id, _norm(c)) for c in data.get("governed_cpts", [])]
        self.conn.executemany("INSERT INTO coverage_cpt VALUES (?,?)", cpt_rows)
        logger.info(f"  coverage[{lcd_id}]: {len(cpt_rows)} governed CPTs, {len(dx_rows)} covered dx")

    def load_coverage_articles(self, articles: list[dict]) -> None:
        """Ingest MCD Billing & Coding Articles into the coverage tables.

        Each article: {policy_id, cpt_codes: [...], covered_icd: [...]}.
        Same shape as the LCD record, so the agent needs no change when the
        MCD bulk download is wired into the refresh layer.
        """
        cpt_rows, icd_rows = [], []
        pids = set()
        for art in articles:
            pid = art.get("policy_id") or art.get("article_id", "")
            pids.add(pid)
            for c in art.get("cpt_codes", []):
                cpt_rows.append((pid, _norm(c)))
            for dx in art.get("covered_icd", []):
                icd_rows.append((pid, _norm(dx)))
        # Idempotent: replace the policy's rows so repeated refreshes don't duplicate
        for pid in pids:
            self.conn.execute("DELETE FROM coverage_cpt WHERE policy_id=?", (pid,))
            self.conn.execute("DELETE FROM coverage_icd WHERE policy_id=?", (pid,))
        self.conn.executemany("INSERT INTO coverage_cpt VALUES (?,?)", cpt_rows)
        self.conn.executemany("INSERT INTO coverage_icd VALUES (?,?)", icd_rows)
        self.conn.commit()
        logger.info(f"  coverage articles: +{len(cpt_rows)} CPT, +{len(icd_rows)} ICD rows")

    def _ingest_pos(self) -> None:
        try:
            with open(POS_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  pos: could not load ({exc})")
            return
        rows = [(c, v.get("name", ""), v.get("facility", "N"))
                for c, v in data.get("codes", {}).items()]
        self.conn.executemany("INSERT INTO pos VALUES (?,?,?)", rows)
        logger.info(f"  pos: {len(rows)} codes")

    def _ingest_modifier_exempt(self) -> None:
        try:
            with open(MODIFIER_EXEMPT_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  modifier_exempt: could not load ({exc})")
            return
        rows = [
            (
                _norm(e["code"]),
                int(bool(e.get("modifier_51_exempt", False))),
                int(bool(e.get("modifier_63_exempt", False))),
            )
            for e in data.get("codes", [])
            if e.get("code")
        ]
        self.conn.executemany("INSERT OR IGNORE INTO modifier_exempt VALUES (?,?,?)", rows)
        n51 = sum(1 for r in rows if r[1])
        n63 = sum(1 for r in rows if r[2])
        logger.info(f"  modifier_exempt: {len(rows)} codes ({n51} mod-51-exempt, {n63} mod-63-exempt)")

    def _ingest_ncci_aoc(self) -> None:
        try:
            with open(NCCI_AOC_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  ncci_aoc: could not load ({exc})")
            return
        rows, skipped = [], 0
        for e in data:
            c1 = _norm(e.get("code1", ""))
            c2 = _norm(e.get("code2", ""))
            if not c1 or not c2:
                skipped += 1
                continue
            mod = str(e.get("modifier", "") or "").strip()
            mod = mod[0] if mod[:1] in ("0", "1", "2", "9") else ""
            rows.append((
                c1, c2, mod,
                _clean_date(e.get("effective_date")),
                str(e.get("end_date") or "").strip() or _OPEN,
            ))
        self.conn.executemany("INSERT INTO ncci_aoc VALUES (?,?,?,?,?)", rows)
        logger.info(f"  ncci_aoc: {len(rows)} AOC edit pairs ({skipped} skipped)")

    def _ingest_modifiers(self) -> None:
        try:
            with open(MODIFIERS_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  modifier: could not load ({exc})")
            return
        rows = [(code.upper(), cat) for code, cat in data.get("modifiers", {}).items()]
        self.conn.executemany("INSERT INTO modifier VALUES (?,?)", rows)
        logger.info(f"  modifier: {len(rows)} recognized modifiers")

    # ---------------------------------------------------------------- lookup
    def is_addon(self, code: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM addon WHERE code=? LIMIT 1", (_norm(code),)
        ).fetchone() is not None

    def pos_info(self, code: str) -> dict | None:
        if not code:
            return None
        c = str(code).strip().zfill(2)
        row = self.conn.execute(
            "SELECT code, name, facility FROM pos WHERE code=? LIMIT 1", (c,)
        ).fetchone()
        return dict(row) if row else None

    def modifier_valid(self, mod: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM modifier WHERE code=? LIMIT 1", (str(mod).strip().upper(),)
        ).fetchone() is not None

    def is_modifier_51_exempt(self, code: str) -> bool:
        row = self.conn.execute(
            "SELECT modifier_51_exempt FROM modifier_exempt WHERE code=? LIMIT 1", (_norm(code),)
        ).fetchone()
        return row is not None and bool(row[0])

    def is_modifier_63_exempt(self, code: str) -> bool:
        row = self.conn.execute(
            "SELECT modifier_63_exempt FROM modifier_exempt WHERE code=? LIMIT 1", (_norm(code),)
        ).fetchone()
        return row is not None and bool(row[0])

    def ncci_aoc_edits(self, code: str, dos=None) -> list[dict]:
        """AOC edits where *code* is the add-on code (code1).
        code2='CCCCC' is the CMS wildcard meaning the edit applies to all primary codes.
        """
        d = self._dos(dos)
        rows = self.conn.execute(
            "SELECT code1, code2, modifier_indicator FROM ncci_aoc "
            "WHERE code1=? AND effective_from<=? AND effective_to>=?",
            (_norm(code), d, d),
        ).fetchall()
        return [dict(r) for r in rows]

    def coverage_policies_for_cpt(self, cpt_code: str) -> list[str]:
        """Policy IDs whose medical-necessity rules govern this CPT."""
        rows = self.conn.execute(
            "SELECT DISTINCT policy_id FROM coverage_cpt WHERE cpt_code=?", (_norm(cpt_code),)
        ).fetchall()
        return [r["policy_id"] for r in rows]

    def coverage_icd_covered(self, policy_id: str, icd_code: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM coverage_icd WHERE policy_id=? AND icd_code=? LIMIT 1",
            (policy_id, _norm(icd_code)),
        ).fetchone() is not None

    def prior_auth_required(self, code: str, payer: str = "Medicare") -> dict | None:
        row = self.conn.execute(
            "SELECT code, note FROM prior_auth_required WHERE code=? AND payer=? LIMIT 1",
            (_norm(code), payer),
        ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _dos(dos: date | str | None) -> str:
        if dos is None:
            return date.today().isoformat()
        return dos if isinstance(dos, str) else dos.isoformat()

    def code_exists(self, system: str, code: str, dos=None) -> bool:
        d = self._dos(dos)
        row = self.conn.execute(
            "SELECT 1 FROM code_set WHERE code_system=? AND code=? "
            "AND effective_from<=? AND effective_to>=? LIMIT 1",
            (system, _norm(code), d, d),
        ).fetchone()
        return row is not None

    def code_active_any_date(self, system: str, code: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM code_set WHERE code_system=? AND code=? LIMIT 1",
            (system, _norm(code)),
        ).fetchone()
        return row is not None

    def children_exist(self, system: str, code: str) -> bool:
        """True if a more specific code exists under this one (specificity check)."""
        c = _norm(code)
        row = self.conn.execute(
            "SELECT 1 FROM code_set WHERE code_system=? AND code LIKE ? AND code<>? LIMIT 1",
            (system, c + "%", c),
        ).fetchone()
        return row is not None

    def _asof(self, table: str, cols: str, where: str, params: tuple, dos: str):
        """Effective-dated lookup with graceful fallback for single-snapshot data.

        1. a rule whose [effective_from, effective_to] range contains the DOS;
        2. else the most recent rule effective on/before the DOS;
        3. else the earliest available rule (snapshot postdates the claim — best
           available; caps/edits are stable quarter-to-quarter).
        Once multiple quarters are retained, step 1 makes lookups exact.
        """
        base = f"SELECT {cols} FROM {table} WHERE {where}"
        row = self.conn.execute(
            f"{base} AND effective_from<=? AND effective_to>=? "
            "ORDER BY effective_from DESC LIMIT 1", (*params, dos, dos),
        ).fetchone()
        if row:
            return row
        row = self.conn.execute(
            f"{base} AND effective_from<=? ORDER BY effective_from DESC LIMIT 1",
            (*params, dos),
        ).fetchone()
        if row:
            return row
        return self.conn.execute(
            f"{base} ORDER BY effective_from ASC LIMIT 1", params,
        ).fetchone()

    def ncci_pair(self, c1: str, c2: str, dos=None) -> dict | None:
        d = self._dos(dos)
        for a, b in ((c1, c2), (c2, c1)):
            row = self._asof(
                "ncci_ptp", "col1, col2, modifier_indicator",
                "col1=? AND col2=?", (_norm(a), _norm(b)), d,
            )
            if row:
                return {"col1": row["col1"], "col2": row["col2"],
                        "modifier_indicator": row["modifier_indicator"]}
        return None

    def mue(self, code: str, dos=None) -> dict | None:
        row = self._asof(
            "mue", "mue_value, mai, rationale", "code=?", (_norm(code),), self._dos(dos),
        )
        return dict(row) if row else None

    def global_period(self, code: str, dos=None) -> str | None:
        row = self.conn.execute(
            "SELECT glob_days FROM global_period WHERE code=? LIMIT 1", (_norm(code),)
        ).fetchone()
        return row["glob_days"] if row else None

    def is_lcd_qualifying(self, dx_code: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM lcd_qualifying WHERE dx_code=? LIMIT 1", (_norm(dx_code),)
        ).fetchone()
        return row is not None

    # --------------------------------------------------------- refresh / history
    def ingest_snapshot(self, table: str, columns: list[str], rows: list[tuple],
                        source_id: str, effective_from: str, file_name: str = "") -> int:
        """Additively ingest a new source snapshot WITHOUT dropping prior data.

        History is retained as effective-dated rows; `_asof()` picks the right
        snapshot per DOS. Re-ingesting the same (source_id, effective_from) is a
        no-op so refresh jobs are idempotent.
        """
        from datetime import datetime
        already = self.conn.execute(
            "SELECT 1 FROM data_source_version WHERE source_id=? AND effective_from=? LIMIT 1",
            (source_id, effective_from),
        ).fetchone()
        if already:
            logger.info(f"  refresh[{source_id}]: snapshot {effective_from} already present — skip")
            return 0
        placeholders = ",".join("?" * len(columns))
        self.conn.executemany(
            f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})", rows
        )
        self.conn.execute(
            "INSERT INTO data_source_version VALUES (?,?,?,?,?)",
            (source_id, effective_from, datetime.now().isoformat(timespec="seconds"),
             len(rows), file_name),
        )
        self.conn.commit()
        logger.info(f"  refresh[{source_id}]: +{len(rows)} rows (eff {effective_from})")
        return len(rows)

    def version_history(self, source_id: str | None = None) -> list[dict]:
        q = "SELECT * FROM data_source_version"
        params: tuple = ()
        if source_id:
            q += " WHERE source_id=?"
            params = (source_id,)
        q += " ORDER BY ingested_at DESC"
        return [dict(r) for r in self.conn.execute(q, params).fetchall()]

    def stats(self) -> dict:
        out = {}
        for tbl in ("code_set", "ncci_ptp", "mue", "global_period", "lcd_qualifying"):
            out[tbl] = self.conn.execute(f"SELECT COUNT(*) c FROM {tbl}").fetchone()["c"]
        out["mue_with_mai"] = self.conn.execute(
            "SELECT COUNT(*) c FROM mue WHERE mai<>''"
        ).fetchone()["c"]
        return out
