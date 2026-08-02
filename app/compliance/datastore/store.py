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
import calendar
from datetime import date
from pathlib import Path

from app.core.config import (
    ICD10_FILE, CPT_FILE, HCPCS_FILE, NCCI_FILE, MUE_FILE, LCD_FILE,
    MCD_COVERAGE_CACHE_FILE, GLOBAL_PERIODS_FILE, DATA_DIR, CODES_DIR,
)

POS_FILE = CODES_DIR / "pos_codes.json"
MODIFIERS_FILE = CODES_DIR / "modifiers.json"
MODIFIER_EXEMPT_FILE = CODES_DIR / "modifier_exempt.json"
NCCI_AOC_FILE = CODES_DIR / "ncci_aoc_edits.json"
ICD10_INSTRUCTIONAL_NOTES_FILE = CODES_DIR / "icd10cm_instructional_notes.json"
ICD10_INDEX_TERMS_FILE = CODES_DIR / "icd10cm_index_terms.json"
MCE_EDITS_FILE = CODES_DIR / "mce_edits.json"
ICD10_CHRONIC_FILE = CODES_DIR / "icd10cm_chronic.json"
EM_MDM_GRID_FILE = CODES_DIR / "em_mdm_grid.json"

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


_CODE_TOKEN_RE = re.compile(r"[A-TV-Z][0-9][0-9A-Z]{0,5}(?:\.[0-9A-Z-]{1,4})?")


def _ref_note_line(ref: str, lines) -> str:
    """The instructional-note line whose parenthetical cites `ref` — e.g.
    ref Z99.2 in N18.6's notes returns 'code to identify dialysis status
    (Z99.2)'. That line is the Tabular List's own clinical wording for the
    referenced condition, which is far closer to how notes are written than
    the ref code's formal description ('dialysis status' vs 'Dependence on
    renal dialysis') — so it's stored alongside the ref for documentation-
    evidence matching. Empty when no line cites the ref."""
    target = _norm(ref).rstrip("-")
    for ln in lines or []:
        for tok in _CODE_TOKEN_RE.findall(str(ln).upper()):
            t = tok.replace(".", "").rstrip("-")
            if t.startswith(target) or target.startswith(t):
                return str(ln)
    return ""


def _norm(code: str) -> str:
    return (code or "").replace(".", "").strip().upper()


def _clean_date(val) -> str:
    """Normalize a date-ish value to 'YYYY-MM-DD'; empty/None → wide-open start.
    Also accepts CMS's compact 'YYYYMMDD' form (e.g. cpt_codes.json's own
    effective_date field, '20240101') — without this, every CPT
    effective_date silently fell through to the "unknown, always active"
    default despite being real, present data."""
    if not val:
        return "1900-01-01"
    s = str(val).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    if re.match(r"^\d{8}$", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return "1900-01-01"


class ComplianceDataStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._ncci_release_window_loaded = False
        self._ncci_release_window: tuple[date, date] | None = None
        self._mue_release_window_loaded = False
        self._mue_release_window: tuple[date, date] | None = None
        self._mue_release_windows: tuple[tuple[date, date], ...] = ()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            # Concurrent openers are a fact of this deployment (the systemd
            # refresh timer and the pipeline can hit the DB at the same
            # moment). Default journaling + 5s lock timeout turned that into
            # 'database is locked' crashes that left half-cleared tables
            # behind. WAL lets readers coexist with one writer; the busy
            # timeout makes a second writer wait instead of dying mid-ingest.
            self._conn.execute("PRAGMA busy_timeout=60000")
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass  # e.g. read-only or network filesystem — timeout still applies
        return self._conn

    # ----------------------------------------------------------------- build
    def build_or_load(self, force_rebuild: bool = False) -> None:
        exists = self.db_path.exists()
        if exists and not force_rebuild and self._is_populated():
            self._ensure_migrations()
            self._refresh_stale_sources()
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
        self._ingest_prior_auth()
        self._ingest_icd10_code_first()
        self._ingest_icd10_use_additional_code()
        self._ingest_icd10_code_also()
        self._ingest_icd10_excludes1()
        self._ingest_icd10_includes()
        self._ingest_icd10_tabular_desc()
        self._ingest_icd10_inclusion_terms()
        self._ingest_icd10_index_terms()
        self._ingest_mce_edits()
        self._ingest_icd10_chronic()
        self._record_source_fingerprints()
        self.conn.commit()
        logger.info("ComplianceDataStore built successfully")

    # ------------------------------------------------- source-file freshness
    # Maps every ingested source file to the table rows it owns and the
    # ingest method(s) that rebuild them. This is the single registry the
    # staleness check walks — previously, editing/refreshing any of these
    # JSON files did NOTHING to an already-built compliance.db (build_or_load
    # returned as soon as code_set had rows), so the scrubber/validator ran
    # on frozen rules while CodeReferenceDB re-read the same JSON fresh on
    # every start — a silent split-brain between the two authorities.
    def _data_sources(self) -> list[dict]:
        return [
            {"id": "icd10_codes", "paths": [ICD10_FILE],
             "clear": [("code_set", "code_system='ICD10'")],
             "ingest": [lambda: self._ingest_code_set("ICD10", ICD10_FILE)]},
            {"id": "cpt_codes", "paths": [CPT_FILE],
             "clear": [("code_set", "code_system='CPT'"), ("addon", None)],
             "ingest": [self._ingest_cpt]},
            {"id": "hcpcs_codes", "paths": [HCPCS_FILE],
             "clear": [("code_set", "code_system='HCPCS'"), ("hcpcs_coverage", None)],
             "ingest": [self._ingest_hcpcs]},
            {"id": "ncci", "paths": [NCCI_FILE],
             "clear": [("ncci_ptp", None)], "ingest": [self._ingest_ncci]},
            {"id": "mue", "paths": [MUE_FILE],
             "clear": [("mue", None)], "ingest": [self._ingest_mue]},
            {"id": "global_periods", "paths": [GLOBAL_PERIODS_FILE],
             "clear": [("global_period", None)], "ingest": [self._ingest_global_periods]},
            # The MCD-export cache is fingerprinted WITH the seed file: a
            # weekly refresh that rewrites the cache must re-ingest coverage
            # (the cache carries the covered-ICD group roles), and a rebuild
            # must clear the group/noncovered tables too or stale rows from
            # a prior ingest would survive the seed re-load.
            {"id": "lcd", "paths": [LCD_FILE, MCD_COVERAGE_CACHE_FILE],
             "clear": [("coverage_cpt", None), ("coverage_icd", None),
                       ("coverage_policy", None), ("lcd_qualifying", None),
                       ("coverage_group", None),
                       ("coverage_icd_noncovered", None)],
             "ingest": [self._ingest_lcd]},
            {"id": "pos", "paths": [POS_FILE],
             "clear": [("pos", None)], "ingest": [self._ingest_pos]},
            {"id": "modifiers", "paths": [MODIFIERS_FILE],
             "clear": [("modifier", None)], "ingest": [self._ingest_modifiers]},
            {"id": "modifier_exempt", "paths": [MODIFIER_EXEMPT_FILE],
             "clear": [("modifier_exempt", None)], "ingest": [self._ingest_modifier_exempt]},
            {"id": "ncci_aoc", "paths": [NCCI_AOC_FILE],
             "clear": [("ncci_aoc", None)], "ingest": [self._ingest_ncci_aoc]},
            {"id": "prior_auth", "paths": sorted(CODES_DIR.glob("prior_auth_*.json")),
             "clear": [("prior_auth_required", None)], "ingest": [self._ingest_prior_auth]},
            {"id": "mce_edits", "paths": [MCE_EDITS_FILE],
             "clear": [("mce_edit", None), ("mce_age_range", None)],
             "ingest": [self._ingest_mce_edits]},
            {"id": "icd10_instructional_notes", "paths": [ICD10_INSTRUCTIONAL_NOTES_FILE],
             "clear": [("icd10_code_first", None), ("icd10_use_additional_code", None),
                       ("icd10_code_also", None),
                       ("icd10_excludes1", None), ("icd10_includes", None),
                       ("icd10_tabular_desc", None), ("icd10_inclusion_term", None)],
             "ingest": [self._ingest_icd10_code_first,
                        self._ingest_icd10_use_additional_code,
                        self._ingest_icd10_code_also,
                        self._ingest_icd10_excludes1,
                        self._ingest_icd10_includes,
                        self._ingest_icd10_tabular_desc,
                        self._ingest_icd10_inclusion_terms]},
            {"id": "icd10_index_terms", "paths": [ICD10_INDEX_TERMS_FILE],
             "clear": [("icd10_index_term", None)],
             "ingest": [self._ingest_icd10_index_terms]},
            {"id": "icd10_chronic", "paths": [ICD10_CHRONIC_FILE],
             "clear": [("icd10_chronic", None)],
             "ingest": [self._ingest_icd10_chronic]},
        ]

    @staticmethod
    def _fingerprint(paths) -> str:
        """size:mtime per file — cheap enough to run on every load even for
        the 400MB+ NCCI file (no content hashing), and reliable because these
        files are only ever replaced wholesale (manual update or refresh
        layer), never appended to in place."""
        parts = []
        for p in paths:
            try:
                st = Path(p).stat()
                parts.append(f"{Path(p).name}:{st.st_size}:{st.st_mtime_ns}")
            except OSError:
                parts.append(f"{Path(p).name}:missing")
        return "|".join(parts)

    def _ensure_fingerprint_table(self) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS data_file_fingerprint ("
            "source_id TEXT NOT NULL PRIMARY KEY, fingerprint TEXT NOT NULL)"
        )

    def _record_source_fingerprints(self) -> None:
        self._ensure_fingerprint_table()
        for src in self._data_sources():
            self.conn.execute(
                "INSERT OR REPLACE INTO data_file_fingerprint VALUES (?,?)",
                (src["id"], self._fingerprint(src["paths"])),
            )
            self._record_seed_provenance(src)

    def _record_seed_provenance(self, src: dict) -> None:
        """Log a seed-file ingest into data_source_version so version_history()
        answers 'what data is this DB running on' for ALL data, not only
        refresh-layer snapshots — seed loads previously left no provenance at
        all, making the provenance table look empty on a fully working DB."""
        from datetime import datetime
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS data_source_version ("
            "source_id TEXT NOT NULL, effective_from TEXT, ingested_at TEXT, "
            "row_count INTEGER, file_name TEXT)"
        )
        source_id = f"seed:{src['id']}"
        today = date.today().isoformat()
        already = self.conn.execute(
            "SELECT 1 FROM data_source_version WHERE source_id=? AND effective_from=? LIMIT 1",
            (source_id, today),
        ).fetchone()
        if already:
            return
        self.conn.execute(
            "INSERT INTO data_source_version VALUES (?,?,?,?,?)",
            (source_id, today, datetime.now().isoformat(timespec="seconds"), -1,
             ", ".join(Path(p).name for p in src["paths"])),
        )

    def _refresh_stale_sources(self) -> None:
        """Re-ingest exactly the sources whose files changed since the DB last
        saw them. On the first run after this feature ships, there are no
        stored fingerprints — we record current ones WITHOUT re-ingesting
        (same as-of-now assumption the DB already operated under; a full
        forced re-ingest here would cost a multi-minute NCCI rebuild on every
        upgraded install for no known change)."""
        self._ensure_fingerprint_table()
        stored = {
            row["source_id"]: row["fingerprint"]
            for row in self.conn.execute("SELECT * FROM data_file_fingerprint")
        }
        if not stored:
            self._record_source_fingerprints()
            self.conn.commit()
            return
        for src in self._data_sources():
            current = self._fingerprint(src["paths"])
            if stored.get(src["id"]) == current:
                continue
            logger.info(f"  data source '{src['id']}' changed on disk — re-ingesting")
            for table, where in src["clear"]:
                sql = f"DELETE FROM {table}" + (f" WHERE {where}" if where else "")
                try:
                    self.conn.execute(sql)
                except sqlite3.Error as exc:
                    logger.warning(f"  could not clear {table} ({exc}) — skipping re-ingest")
                    break
            else:
                for ingest in src["ingest"]:
                    ingest()
                self.conn.execute(
                    "INSERT OR REPLACE INTO data_file_fingerprint VALUES (?,?)",
                    (src["id"], current),
                )
                self._record_seed_provenance(src)
        self.conn.commit()

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
            CREATE TABLE IF NOT EXISTS icd10_code_first (
                code TEXT NOT NULL,          -- manifestation code (e.g. H36.811)
                etiology_ref TEXT NOT NULL,  -- referenced etiology code/prefix (e.g. E75)
                note_text TEXT DEFAULT ''    -- the note line naming this ref
            );
            CREATE INDEX IF NOT EXISTS ix_icd10_code_first ON icd10_code_first(code);
            CREATE TABLE IF NOT EXISTS icd10_use_additional_code (
                code TEXT NOT NULL,          -- condition code (e.g. E11)
                ref TEXT NOT NULL,           -- recommended companion code/prefix (e.g. Z79.84)
                note_text TEXT DEFAULT ''    -- the note line naming this ref
            );
            CREATE INDEX IF NOT EXISTS ix_icd10_use_additional_code ON icd10_use_additional_code(code);
            CREATE INDEX IF NOT EXISTS ix_icd10_use_additional_code_ref ON icd10_use_additional_code(ref);
            CREATE TABLE IF NOT EXISTS icd10_code_also (
                code TEXT NOT NULL,          -- code carrying the codeAlso note
                ref TEXT NOT NULL,           -- companion code/prefix the note cites
                note_text TEXT DEFAULT ''    -- the note line naming this ref
            );
            CREATE INDEX IF NOT EXISTS ix_icd10_code_also ON icd10_code_also(code);
            CREATE TABLE IF NOT EXISTS icd10_excludes1 (
                code TEXT NOT NULL,          -- code carrying the excludes1 note (e.g. M12.5)
                excluded_ref TEXT NOT NULL   -- excluded code/prefix (e.g. M19.1)
            );
            CREATE INDEX IF NOT EXISTS ix_icd10_excludes1 ON icd10_excludes1(code);
            CREATE INDEX IF NOT EXISTS ix_icd10_excludes1_ref ON icd10_excludes1(excluded_ref);
            CREATE TABLE IF NOT EXISTS icd10_includes (
                code TEXT NOT NULL,          -- code carrying the includes note (e.g. I70.23)
                included_ref TEXT NOT NULL   -- subsumed code/prefix (e.g. I70.221)
            );
            CREATE INDEX IF NOT EXISTS ix_icd10_includes ON icd10_includes(code);
            CREATE INDEX IF NOT EXISTS ix_icd10_includes_ref ON icd10_includes(included_ref);
            CREATE TABLE IF NOT EXISTS icd10_tabular_desc (
                code TEXT NOT NULL PRIMARY KEY,  -- normalized (dotless) Tabular entry code
                description TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS icd10_inclusion_term (
                code TEXT NOT NULL,   -- normalized (dotless) Tabular entry code
                term TEXT NOT NULL    -- one official synonym/example phrase
            );
            CREATE INDEX IF NOT EXISTS ix_icd10_inclusion_term ON icd10_inclusion_term(code);
            CREATE TABLE IF NOT EXISTS icd10_index_term (
                code TEXT NOT NULL,   -- normalized (dotless) code or code stem
                term TEXT NOT NULL    -- Alphabetic Index phrase leading to it
            );
            CREATE INDEX IF NOT EXISTS ix_icd10_index_term ON icd10_index_term(code);
            CREATE TABLE IF NOT EXISTS mce_edit (
                family TEXT NOT NULL,
                code   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_mce_edit ON mce_edit(code);
            CREATE TABLE IF NOT EXISTS mce_age_range (
                category TEXT NOT NULL PRIMARY KEY,
                min_age  INTEGER NOT NULL,
                max_age  INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hcpcs_coverage (
                code TEXT NOT NULL PRIMARY KEY,
                coverage_code TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS icd10_chronic (
                code    TEXT NOT NULL PRIMARY KEY,
                chronic INTEGER NOT NULL
            );
            """
        )
        self.conn.commit()

        # modifier.systems: added when CPT-vs-HCPCS applicability was sourced
        # from AMA/CMS data instead of being unavailable. Re-ingest on upgrade
        # since old rows predate both the column and the richer source data.
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(modifier)")}
        if "systems" not in cols:
            self.conn.execute("ALTER TABLE modifier ADD COLUMN systems TEXT")
            self.conn.execute("DELETE FROM modifier")
            self._ingest_modifiers()
            self.conn.commit()
        # Re-check VALUES, not just column presence (same pattern as bilat_surg
        # below): a DB built while _ingest_modifiers read a "systems" (plural)
        # key that never existed in modifiers.json has the column but every
        # row empty — which made modifier_valid_for_cpt() return False for all
        # 99 recognized modifiers and stripped every modifier from every claim.
        n_sys = self.conn.execute(
            "SELECT COUNT(*) FROM modifier WHERE systems IS NOT NULL AND systems != ''"
        ).fetchone()[0]
        if n_sys == 0:
            self.conn.execute("DELETE FROM modifier")
            self._ingest_modifiers()
            self.conn.commit()

        # global_period.billing_status: added when the source's own status
        # field (A/B/C/I/N/R/T/X — see global_periods.json's indicator_meanings)
        # was wired up for billability checks instead of being discarded.
        # Re-ingest since old rows predate the column.
        gp_cols = {row[1] for row in self.conn.execute("PRAGMA table_info(global_period)")}
        if "billing_status" not in gp_cols:
            self.conn.execute("ALTER TABLE global_period ADD COLUMN billing_status TEXT")
            self.conn.execute("DELETE FROM global_period")
            self._ingest_global_periods()
            self.conn.commit()

        # global_period.bilat_surg: CMS bilateral-surgery indicator, added
        # when CPT laterality (RT/LT/50) enforcement was wired to real data
        # instead of a CPT-section prefix guess. Re-check on every load
        # (not just column-presence) — a DB built between the column's
        # introduction and this being backfilled would have the column but
        # every row NULL, since ALTER TABLE ADD COLUMN doesn't retroactively
        # populate existing rows.
        gp_cols = {row[1] for row in self.conn.execute("PRAGMA table_info(global_period)")}
        if "bilat_surg" not in gp_cols:
            self.conn.execute("ALTER TABLE global_period ADD COLUMN bilat_surg TEXT")
            self.conn.commit()
        # Remaining PFS payment-policy indicator columns (PC/TC, multiple-
        # procedure, assistant/co-/team-surgeon) — added when the modifier
        # checks for 26/TC, 80/81/82/AS, 62 and 66 were wired to the real
        # per-code indicators the source file always carried. Same
        # values-not-just-columns re-check pattern as bilat_surg above.
        gp_cols = {row[1] for row in self.conn.execute("PRAGMA table_info(global_period)")}
        for col in ("pctc_ind", "mult_proc", "asst_surg", "co_surg", "team_surg"):
            if col not in gp_cols:
                self.conn.execute(f"ALTER TABLE global_period ADD COLUMN {col} TEXT")
        self.conn.commit()
        n_pctc = self.conn.execute(
            "SELECT COUNT(*) FROM global_period WHERE pctc_ind IS NOT NULL"
        ).fetchone()[0]
        n_bilat = self.conn.execute(
            "SELECT COUNT(*) FROM global_period WHERE bilat_surg IS NOT NULL"
        ).fetchone()[0]
        if n_bilat == 0 or n_pctc == 0:
            self.conn.execute("DELETE FROM global_period")
            self._ingest_global_periods()
            self.conn.commit()

        n_mce = self.conn.execute("SELECT COUNT(*) FROM mce_edit").fetchone()[0]
        if n_mce == 0:
            self._ingest_mce_edits()
            self.conn.commit()

        # hcpcs_coverage: the coverage_code field (I/M/S = Medicare
        # non-coverage) always existed in hcpcs_codes.json but was never
        # ingested. Re-run the HCPCS ingest once on DBs built before the
        # table was populated.
        n_cov = self.conn.execute("SELECT COUNT(*) FROM hcpcs_coverage").fetchone()[0]
        if n_cov == 0:
            self.conn.execute("DELETE FROM code_set WHERE code_system='HCPCS'")
            self._ingest_hcpcs()
            self.conn.commit()

        # code_set[CPT]/code_set[HCPCS] effective dates: _ingest_cpt/
        # _ingest_hcpcs used to hardcode every row to "1900-01-01"/open
        # regardless of the code's own real effective_date (CPT) or
        # effective_from/effective_to (HCPCS) — a schema-unchanged, values-
        # only fix, so there's no new/missing column to detect the way the
        # migrations above do. A DB built before this fix has EVERY CPT/
        # HCPCS row still on the literal default; check for that (not a
        # count, which stays the same either way) and re-ingest if so.
        n_dated = self.conn.execute(
            "SELECT COUNT(*) FROM code_set WHERE code_system IN ('CPT','HCPCS') "
            "AND effective_from != '1900-01-01'"
        ).fetchone()[0]
        if n_dated == 0:
            self.conn.execute("DELETE FROM code_set WHERE code_system IN ('CPT','HCPCS')")
            self.conn.execute("DELETE FROM addon")
            self._ingest_cpt()
            self._ingest_hcpcs()
            self.conn.commit()

        # coverage_cpt/coverage_icd: _ingest_lcd() used to expect a single
        # top-level lcd_id/qualifying_dx shape that podiatry_lcd.json never
        # actually had (it's a list of hundreds of LCDs/Articles) — so these
        # tables were silently empty on every DB built before this fix,
        # meaning MedicalNecessityAgent (filter #5) never fired. Re-ingest
        # whenever they're empty, not just on fresh builds.
        n_coverage = self.conn.execute("SELECT COUNT(*) FROM coverage_cpt").fetchone()[0]
        if n_coverage == 0:
            self._ingest_lcd()

        # coverage_policy (policy titles): added when policy-kind checks (e.g.
        # routine-foot-care class-findings modifiers) started keying off the
        # policy's own real CMS title instead of a hardcoded CPT list. A DB
        # built before this has coverage rows but no title table — re-ingest
        # LCDs to populate it.
        self._ensure_coverage_policy_table()
        n_titles = self.conn.execute("SELECT COUNT(*) FROM coverage_policy").fetchone()[0]
        if n_titles == 0:
            self._ingest_lcd()

        # coverage_policy.contractor: added when MedicalNecessityAgent gained
        # MAC-jurisdiction scoping (an LCD from CGS (KY/OH) must not gate a
        # Florida claim). A DB built before the column has titles but every
        # contractor blank — re-ingest LCDs to populate it. Values-not-column
        # check: _ensure_coverage_policy_table ALTERs the column in, so its
        # mere presence proves nothing about the data.
        n_contractor = self.conn.execute(
            "SELECT COUNT(*) FROM coverage_policy WHERE contractor != ''"
        ).fetchone()[0]
        if n_contractor == 0 and n_titles > 0:
            self._ingest_lcd()

        # coverage_group (covered-ICD group roles): added when
        # MedicalNecessityAgent gained the claim-composition gate. A DB
        # built before this has coverage rows but no group grammar — if the
        # parsed MCD-export cache is on disk (weekly refresh), re-ingest so
        # the composition gate has roles to evaluate. Without the cache
        # there is nothing to gain from re-ingesting the flat seed.
        n_groups = self.conn.execute(
            "SELECT COUNT(*) FROM coverage_group").fetchone()[0]
        if n_groups == 0 and MCD_COVERAGE_CACHE_FILE.exists():
            self._ingest_lcd()

        # coverage_policy.states (authoritative MAC service areas from the
        # export's contractor_jurisdiction tables) + related-LCD supersession
        # both arrived with the same parser revision — a DB whose policies
        # all have blank states was ingested from a pre-revision cache (or
        # none). Re-ingest only helps if the cache on disk actually carries
        # the new fields, so gate on that too.
        n_states = self.conn.execute(
            "SELECT COUNT(*) FROM coverage_policy WHERE states != ''"
        ).fetchone()[0]
        if n_states == 0 and MCD_COVERAGE_CACHE_FILE.exists():
            try:
                with open(MCD_COVERAGE_CACHE_FILE) as f:
                    _cache_head = json.load(f)
                if any(a.get("states") for a in _cache_head.get("articles", [])):
                    self._ingest_lcd()
            except Exception:
                pass  # unreadable cache already logged by _ingest_lcd

        # MCD "Not Applicable" placeholder (see _MCD_NA_PLACEHOLDER): purge it
        # from DBs ingested before load_coverage_articles started filtering it
        # — 55 policies' only "diagnosis" was this sentinel, which made them
        # unsatisfiable diagnosis gates.
        self.conn.execute("DELETE FROM coverage_icd WHERE icd_code=?",
                          (self._MCD_NA_PLACEHOLDER,))
        self.conn.execute("DELETE FROM lcd_qualifying WHERE dx_code=?",
                          (self._MCD_NA_PLACEHOLDER,))
        self.conn.commit()

        # note_text: the Tabular note line citing each ref, added when the
        # missing-companion/missing-etiology mirror checks started matching
        # documentation against the note's own clinical wording ('dialysis
        # status') instead of only the ref code's formal description. Tables
        # built before the column need a re-ingest to populate it.
        for tbl, ingest in (("icd10_code_first", self._ingest_icd10_code_first),
                            ("icd10_use_additional_code", self._ingest_icd10_use_additional_code)):
            cols = {row[1] for row in self.conn.execute(f"PRAGMA table_info({tbl})")}
            if "note_text" not in cols:
                self.conn.execute(f"ALTER TABLE {tbl} ADD COLUMN note_text TEXT DEFAULT ''")
                self.conn.execute(f"DELETE FROM {tbl}")
                ingest()
                self.conn.commit()

        n_code_first = self.conn.execute("SELECT COUNT(*) FROM icd10_code_first").fetchone()[0]
        if n_code_first == 0:
            self._ingest_icd10_code_first()

        n_use_additional = self.conn.execute("SELECT COUNT(*) FROM icd10_use_additional_code").fetchone()[0]
        if n_use_additional == 0:
            self._ingest_icd10_use_additional_code()

        n_code_also = self.conn.execute("SELECT COUNT(*) FROM icd10_code_also").fetchone()[0]
        if n_code_also == 0:
            self._ingest_icd10_code_also()

        n_excludes1 = self.conn.execute("SELECT COUNT(*) FROM icd10_excludes1").fetchone()[0]
        if n_excludes1 == 0:
            self._ingest_icd10_excludes1()

        n_includes = self.conn.execute("SELECT COUNT(*) FROM icd10_includes").fetchone()[0]
        if n_includes == 0:
            self._ingest_icd10_includes()

        n_tab_desc = self.conn.execute("SELECT COUNT(*) FROM icd10_tabular_desc").fetchone()[0]
        if n_tab_desc == 0:
            self._ingest_icd10_tabular_desc()

        n_incl_terms = self.conn.execute(
            "SELECT COUNT(*) FROM icd10_inclusion_term").fetchone()[0]
        if n_incl_terms == 0:
            self._ingest_icd10_inclusion_terms()

        n_index_terms = self.conn.execute(
            "SELECT COUNT(*) FROM icd10_index_term").fetchone()[0]
        if n_index_terms == 0:
            self._ingest_icd10_index_terms()

        # prior_auth_required: was (payer, code, note) — no way to represent
        # category-based payer policies (e.g. Tricare, which publishes broad
        # categories like "Durable Medical Equipment" rather than enumerated
        # codes). Recreate with the richer schema and re-ingest on upgrade.
        pa_cols = {row[1] for row in self.conn.execute("PRAGMA table_info(prior_auth_required)")}
        if "category" not in pa_cols:
            self.conn.executescript(
                """
                DROP TABLE IF EXISTS prior_auth_required;
                CREATE TABLE prior_auth_required (
                    payer         TEXT NOT NULL,
                    code          TEXT,
                    category      TEXT,
                    hcpcs_prefix  TEXT,
                    note          TEXT,
                    source        TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_pa_code ON prior_auth_required(payer, code);
                CREATE INDEX IF NOT EXISTS ix_pa_prefix ON prior_auth_required(payer, hcpcs_prefix);
                """
            )
            self._ingest_prior_auth()

    def _is_populated(self) -> bool:
        try:
            cur = self.conn.execute("SELECT COUNT(*) c FROM code_set")
            return cur.fetchone()["c"] > 0
        except sqlite3.OperationalError as exc:
            # 'no such table' genuinely means unbuilt. Anything else (locked,
            # busy, disk I/O) is a transient fault — swallowing it here once
            # sent build_or_load() down the DESTRUCTIVE rebuild path against
            # a live DB another process was writing. Fail loudly instead.
            if "no such table" in str(exc):
                return False
            raise

    def _create_schema(self) -> None:
        # Drop EVERY table this script creates, not a subset — a build that
        # crashed midway (e.g. 'database is locked' during a boot-time timer
        # collision) leaves a partial schema, and the next build_or_load()
        # lands here again; any table created below but not dropped here
        # wedges that recovery with 'table X already exists'.
        self.conn.executescript(
            """
            DROP TABLE IF EXISTS code_set;
            DROP TABLE IF EXISTS ncci_ptp;
            DROP TABLE IF EXISTS ncci_aoc;
            DROP TABLE IF EXISTS mue;
            DROP TABLE IF EXISTS global_period;
            DROP TABLE IF EXISTS lcd_qualifying;
            DROP TABLE IF EXISTS modifier_exempt;
            DROP TABLE IF EXISTS addon;
            DROP TABLE IF EXISTS pos;
            DROP TABLE IF EXISTS modifier;
            DROP TABLE IF EXISTS coverage_cpt;
            DROP TABLE IF EXISTS coverage_icd;
            DROP TABLE IF EXISTS prior_auth_required;
            DROP TABLE IF EXISTS icd10_code_first;
            DROP TABLE IF EXISTS icd10_use_additional_code;
            DROP TABLE IF EXISTS icd10_code_also;
            DROP TABLE IF EXISTS icd10_excludes1;
            DROP TABLE IF EXISTS icd10_includes;
            DROP TABLE IF EXISTS icd10_tabular_desc;
            DROP TABLE IF EXISTS icd10_inclusion_term;
            DROP TABLE IF EXISTS icd10_index_term;
            DROP TABLE IF EXISTS mce_edit;
            DROP TABLE IF EXISTS mce_age_range;
            DROP TABLE IF EXISTS hcpcs_coverage;

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
                billing_status TEXT,             -- A/B/C/I/N/R/T/X (see global_periods.json's
                                                  -- own indicator_meanings.status) — the real
                                                  -- Medicare payability indicator, e.g. B=bundled/
                                                  -- not separately payable, N=noncovered, X=statutory
                                                  -- exclusion. Values outside that documented set
                                                  -- (E/M/J/P also appear in the source) are stored
                                                  -- as-is but not interpreted — undocumented in the
                                                  -- source, so not safe to assume a meaning for.
                bilat_surg TEXT,                 -- CMS bilateral-surgery indicator (see
                                                  -- global_periods.json's own
                                                  -- indicator_meanings.bilat_surg): '1' means
                                                  -- the 150% bilateral payment adjustment
                                                  -- applies to this code, i.e. it's billed as
                                                  -- either a unilateral (RT/LT) or bilateral
                                                  -- (50) line — the real, code-specific signal
                                                  -- for whether a laterality modifier is
                                                  -- expected, vs. inferring it from a CPT
                                                  -- section/prefix guess.
                -- Remaining PFS payment-policy indicators, each with its
                -- meaning documented in global_periods.json's own
                -- indicator_meanings block. Each one is a modifier-validity
                -- rule CMS publishes per code:
                pctc_ind  TEXT,                  -- PC/TC split: 26/TC only meaningful when '1'
                mult_proc TEXT,                  -- multiple-procedure reduction family
                asst_surg TEXT,                  -- 80/81/82/AS restriction ('0'/'1' = restricted)
                co_surg   TEXT,                  -- 62 permitted? ('0' = not permitted)
                team_surg TEXT,                  -- 66 permitted? ('0' = not permitted)
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
                category TEXT,
                systems TEXT   -- comma-joined, e.g. 'cpt,hcpcs' or 'hcpcs' — from
                                -- modifiers.json systems (AMA/CMS source membership)
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

            -- group_id preserves the source's own Group structure (MCD
            -- covered-ICD Group N). NULL = ungrouped/flat ingest (seed
            -- file) — evaluated exactly like the pre-group behavior.
            CREATE TABLE coverage_icd (
                policy_id TEXT NOT NULL,
                icd_code  TEXT NOT NULL,
                group_id  INTEGER
            );
            CREATE INDEX ix_cov_icd ON coverage_icd(policy_id, icd_code);

            -- Covered-ICD group grammar (role in claim composition), parsed
            -- from each group's own paragraph in the MCD export — the
            -- structure whose loss made claim-composition edits (e.g.
            -- "B35.1 must be reported as primary WITH a symptom secondary")
            -- inexpressible. role: primary_eligible | required_secondary |
            -- unspecified (standalone). cpt_scope: comma-joined procedure
            -- codes the group is scoped to ('' = all governed codes).
            -- paragraph: provenance excerpt backing the parsed role.
            CREATE TABLE coverage_group (
                policy_id TEXT NOT NULL,
                group_id  INTEGER NOT NULL,
                role      TEXT NOT NULL DEFAULT 'unspecified',
                cpt_scope TEXT NOT NULL DEFAULT '',
                paragraph TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX ix_cov_group ON coverage_group(policy_id);

            -- Prior-authorization requirements (payer/plan/code specific).
            -- Two rule shapes share this table: exact-code (Medicare DMEPOS —
            -- code set, hcpcs_prefix NULL) and category-based (Tricare —
            -- category set, hcpcs_prefix set when a standard HCPCS Level II
            -- prefix maps to it, e.g. "Durable Medical Equipment" -> "E").
            CREATE TABLE prior_auth_required (
                payer         TEXT NOT NULL,
                code          TEXT,
                category      TEXT,
                hcpcs_prefix  TEXT,
                note          TEXT,
                source        TEXT
            );
            CREATE INDEX ix_pa_code ON prior_auth_required(payer, code);
            CREATE INDEX ix_pa_prefix ON prior_auth_required(payer, hcpcs_prefix);

            -- ICD-10-CM instructional notes: "code first" (manifestation
            -- must be sequenced after its etiology) and "use additional
            -- code" (companion code recommended alongside this condition).
            -- Also created via CREATE TABLE IF NOT EXISTS in
            -- _ensure_migrations() for DBs built before this table existed
            -- — declared here too so a genuinely fresh build (no prior db
            -- file, so _ensure_migrations() never runs) doesn't crash in
            -- _ingest_icd10_code_first() with "no such table".
            CREATE TABLE icd10_code_first (
                code TEXT NOT NULL,          -- manifestation code (e.g. H36.811)
                etiology_ref TEXT NOT NULL,  -- referenced etiology code/prefix (e.g. E75)
                note_text TEXT DEFAULT ''    -- the note line naming this ref, e.g.
                                             -- 'atherosclerosis of the lower extremities (I70.23-...)'
                                             -- — the Tabular's own clinical vocabulary for the
                                             -- condition, used for documentation-evidence matching
            );
            CREATE INDEX ix_icd10_code_first ON icd10_code_first(code);
            CREATE TABLE icd10_use_additional_code (
                code TEXT NOT NULL,          -- condition code (e.g. E11)
                ref TEXT NOT NULL,           -- recommended companion code/prefix (e.g. Z79.84)
                note_text TEXT DEFAULT ''    -- the note line naming this ref, e.g.
                                             -- 'code to identify dialysis status (Z99.2)'
            );
            CREATE INDEX ix_icd10_use_additional_code ON icd10_use_additional_code(code);
            CREATE INDEX ix_icd10_use_additional_code_ref ON icd10_use_additional_code(ref);

            -- ICD-10-CM "Code also" notes: two codes may be required to
            -- fully describe the condition, sequencing discretionary
            -- (unlike codeFirst's mandated order or useAdditionalCode's
            -- mandated secondary position) — e.g. C25 'code also if
            -- applicable exocrine pancreatic insufficiency (K86.81)'.
            CREATE TABLE icd10_code_also (
                code TEXT NOT NULL,          -- code carrying the codeAlso note
                ref TEXT NOT NULL,           -- companion code/prefix the note cites
                note_text TEXT DEFAULT ''    -- the note line naming this ref
            );
            CREATE INDEX ix_icd10_code_also ON icd10_code_also(code);

            -- ICD-10-CM Type 1 Excludes notes: "not coded here" — the two
            -- referenced code categories describe clinically distinct
            -- conditions CMS's own Tabular List says cannot be coded
            -- together on the same claim (e.g. M12.5 Traumatic arthropathy
            -- excludes1 M19.1 Post-traumatic osteoarthritis — structurally
            -- mutually exclusive, not a stylistic choice between similar
            -- codes). Not necessarily symmetric in the source data (the
            -- note may only be annotated on one code's own Tabular entry
            -- even though the exclusion holds both ways), so callers must
            -- check both directions of a pair.
            CREATE TABLE icd10_excludes1 (
                code TEXT NOT NULL,          -- code carrying the excludes1 note (e.g. M12.5)
                excluded_ref TEXT NOT NULL   -- excluded code/prefix (e.g. M19.1)
            );
            CREATE INDEX ix_icd10_excludes1 ON icd10_excludes1(code);
            CREATE INDEX ix_icd10_excludes1_ref ON icd10_excludes1(excluded_ref);

            -- ICD-10-CM "Includes" hierarchy: the carrying code's Tabular
            -- entry names other codes/categories it subsumes (e.g. I70.23-
            -- 'Includes: ... rest pain (I70.22-)') — the basis for the
            -- redundant-pair subsumption check.
            CREATE TABLE icd10_includes (
                code TEXT NOT NULL,          -- code carrying the includes note (e.g. I70.23)
                included_ref TEXT NOT NULL   -- subsumed code/prefix (e.g. I70.221)
            );
            CREATE INDEX ix_icd10_includes ON icd10_includes(code);
            CREATE INDEX ix_icd10_includes_ref ON icd10_includes(included_ref);

            -- Every Tabular List entry's own description, INCLUDING category-
            -- level (non-billable) entries like Z79 or I70.2 that the billable
            -- code file (icd10cm_codes.json) doesn't carry. Instructional-note
            -- refs frequently point at categories, so resolving a ref to
            -- human-readable text needs this table, not code_set.
            CREATE TABLE icd10_tabular_desc (
                code TEXT NOT NULL PRIMARY KEY,  -- normalized (dotless) Tabular entry code
                description TEXT NOT NULL
            );

            -- Tabular List inclusion terms: the official synonym/example
            -- phrases printed under an entry (e.g. L03.0 lists 'Paronychia';
            -- B35.1 lists 'Onychomycosis'). These are the code's own
            -- alternate names — the evidence vocabulary for matching a
            -- note's wording to a code beyond its one-line description.
            CREATE TABLE icd10_inclusion_term (
                code TEXT NOT NULL,
                term TEXT NOT NULL
            );
            CREATE INDEX ix_icd10_inclusion_term ON icd10_inclusion_term(code);

            -- ICD-10-CM Alphabetic Index: every lookup phrase that leads to
            -- a code ('cellulitis toe' -> L03.03-, 'paronychia' via its
            -- see-also to the Cellulitis family). THE authoritative
            -- term-to-code mapping; the Tabular alone cannot map synonyms
            -- like paronychia or onychomycosis to the right sibling.
            CREATE TABLE icd10_index_term (
                code TEXT NOT NULL,
                term TEXT NOT NULL
            );
            CREATE INDEX ix_icd10_index_term ON icd10_index_term(code);

            -- CMS Medicare Code Editor (MCE) diagnosis edit lists: which edit
            -- families a diagnosis belongs to (age_newborn/age_pediatric/
            -- age_maternity/age_adult, manifestation_not_pdx,
            -- unacceptable_pdx, unacceptable_pdx_unless_secondary), plus the
            -- age ranges the MCE text itself defines per age category.
            CREATE TABLE mce_edit (
                family TEXT NOT NULL,
                code   TEXT NOT NULL
            );
            CREATE INDEX ix_mce_edit ON mce_edit(code);
            CREATE TABLE mce_age_range (
                category TEXT NOT NULL PRIMARY KEY,
                min_age  INTEGER NOT NULL,
                max_age  INTEGER NOT NULL
            );

            -- HCPCS coverage code (hcpcs_codes.json's coverage_code field,
            -- from CMS's own alpha-numeric HCPCS file): C=carrier judgment,
            -- D=special coverage instructions, I=not payable by Medicare,
            -- M=non-covered by Medicare, S=non-covered by Medicare statute.
            -- I/M/S are per-code Medicare coverage denials that PFS billing
            -- status can't see (most HCPCS II codes aren't on the PFS).
            CREATE TABLE hcpcs_coverage (
                code TEXT NOT NULL PRIMARY KEY,
                coverage_code TEXT NOT NULL
            );

            -- AHRQ HCUP Chronic Condition Indicator Refined (CCIR): per
            -- ICD-10-CM code, 1=chronic / 0=not chronic / 9=no
            -- determination. THE classification-level answer to 'is this
            -- diagnosis a chronic illness' used by the E/M problems-axis
            -- floor ('2 or more stable chronic illnesses' = moderate per
            -- the 2021 AMA MDM table).
            CREATE TABLE icd10_chronic (
                code    TEXT NOT NULL PRIMARY KEY,
                chronic INTEGER NOT NULL
            );

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
            # CPT's effective_date is NOT a code-introduction date — verified
            # empirically: 99202/99213/99214 (decades-old, unquestionably
            # not new codes) all carry effective_date 2024-01-01, and the
            # full distribution of populated values clusters on Jan 1/Jul 1
            # across 2020-2026 — a periodic descriptor-revision cycle
            # marker, not a lifecycle date. Using it as an activation gate
            # would flag huge numbers of long-standing, merely-reworded
            # codes (e.g. any E/M code touched by the 2021 guideline
            # overhaul) as "not active" for any older, perfectly valid date
            # of service. No reliable introduction/retirement signal exists
            # in this source for CPT, so — unlike ICD-10 and HCPCS, both
            # verified against real add/discontinue signals — CPT stays
            # always-open rather than approximating with the wrong field.
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
        rows, cov_rows, skipped, seen = [], [], 0, set()
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
            cov = str(e.get("coverage_code") or "").strip().upper()
            if cov:
                cov_rows.append((code, cov))
            desc = (e.get("short_description") or "").strip()
            # add_date (not effective_from) is the real introduction date —
            # verified: effective_from shifts on every quarterly pricing/
            # descriptor cycle even for decades-old codes (e.g. A4570:
            # add_date=1982-01-01 but effective_from=2001-07-01 — the same
            # "revision cycle, not lifecycle" problem confirmed for CPT's
            # effective_date), while add_date is populated on all 9068
            # codes and stays stable at the code's true origin (e.g. a
            # cluster of 525 codes at add_date=1986-01-01, HCPCS's own
            # system-migration baseline). effective_to is still reliable
            # for discontinuation specifically — action_code='D' entries
            # all carry a real, terminal effective_to (e.g. C9145,
            # discontinued 2026-03-31) — so that half of the original fix
            # stands unchanged.
            eff_to = str(e.get("effective_to") or "").strip()
            rows.append((
                "HCPCS", code, desc,
                _clean_date(e.get("add_date")),
                _clean_date(eff_to) if eff_to else _OPEN,
                "active",
            ))
        self.conn.executemany("INSERT INTO code_set VALUES (?,?,?,?,?,?)", rows)
        self.conn.executemany(
            "INSERT OR REPLACE INTO hcpcs_coverage VALUES (?,?)", cov_rows
        )
        logger.info(f"  code_set[HCPCS]: {len(rows)} unique codes ({skipped} dirty rows "
                    f"skipped, {len(cov_rows)} coverage codes)")

    def _ingest_ncci(self) -> None:
        # A refresh can move the authoritative release quarter. Invalidate
        # the per-store lookup cache before replacing its rows so already-
        # constructed store instances cannot retain the prior window.
        self._ncci_release_window_loaded = False
        self._ncci_release_window = None
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
        self._mue_release_window_loaded = False
        self._mue_release_window = None
        self._mue_release_windows = ()
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
                status = str(days.get("status", "") or "").strip() or None
                def _ind(key):
                    return str(days.get(key, "") or "").strip() or None
                bilat_surg = _ind("bilat_surg")
                pctc, mult = _ind("pctc_ind"), _ind("mult_proc")
                asst, co, team = _ind("asst_surg"), _ind("co_surg"), _ind("team_surg")
            else:
                glob_days = str(days).strip() if days is not None else ""
                status = bilat_surg = pctc = mult = asst = co = team = None
            if not glob_days:
                continue
            rows.append((_norm(code), glob_days, status, bilat_surg,
                         pctc, mult, asst, co, team, "1900-01-01", _OPEN))
        # Named columns, not positional VALUES — ALTER TABLE ADD COLUMN (the
        # migration path for DBs built before billing_status existed) always
        # appends the new column at the end of the table regardless of where
        # it's declared in CREATE TABLE, so positional inserts would silently
        # write billing_status into the wrong column on a migrated DB.
        self.conn.executemany(
            "INSERT INTO global_period (code, glob_days, billing_status, bilat_surg, "
            "pctc_ind, mult_proc, asst_surg, co_surg, team_surg, effective_from, effective_to) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        logger.info(f"  global_period: {len(rows)} codes")

    def _ingest_lcd(self) -> None:
        """podiatry_lcd.json is the full CMS Coverage API dataset: hundreds of
        LCDs and Billing/Coding Articles across every specialty and MAC
        jurisdiction — not one single policy. Ingest every active one
        generically via load_coverage_articles(); coverage_cpt/coverage_icd
        already support arbitrary policy_id counts, and MedicalNecessityAgent
        (filter #5) already queries them per-code, not by a hardcoded policy —
        it needs no change to pick up whatever's loaded here.
        """
        try:
            with open(LCD_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  lcd_qualifying: could not load ({exc})")
            return

        # Many LCDs carry their governed CPTs directly but leave qualifying_dx
        # empty, with icd10_in_companion_article=true pointing at a separate
        # Billing/Coding Article (via companion_article_id) for the actual
        # diagnosis list — a real, resolvable cross-reference this dataset
        # provides but that was never followed. Confirmed live: 405 of 964
        # LCDs (42%) have empty qualifying_dx + icd10_in_companion_article,
        # 255 of them resolve to a real article in this same file, covering
        # 839 unique CPT codes. Without resolving this, MedicalNecessityAgent
        # sees the CPT as governed by a policy but can never find a
        # diagnosis that satisfies it — a real documented, covered diagnosis
        # would still FAIL as "not medically necessary" on every claim
        # billing one of those 839 codes, regardless of what's documented.
        article_by_id = {
            a.get("article_id"): a for a in data.get("article", []) if a.get("article_id")
        }

        articles = []
        for entry in data.get("lcd", []):
            if entry.get("status") != "A":
                continue
            qualifying_dx = entry.get("qualifying_dx", [])
            if not qualifying_dx and entry.get("icd10_in_companion_article"):
                companion = article_by_id.get(entry.get("companion_article_id"))
                if companion:
                    qualifying_dx = companion.get("qualifying_dx", [])
            articles.append({
                "policy_id": entry.get("lcd_id", ""),
                "title": entry.get("title", ""),
                "contractor": entry.get("contractor", ""),
                "cpt_codes": entry.get("governed_cpts", []),
                "covered_icd": qualifying_dx,
            })
        for entry in data.get("article", []):
            if entry.get("status") != "A":
                continue
            articles.append({
                "policy_id": entry.get("article_id", ""),
                "title": entry.get("title", ""),
                "contractor": entry.get("contractor", ""),
                "cpt_codes": entry.get("governed_cpts", []),
                "covered_icd": entry.get("qualifying_dx", []),
                # seed entries may carry group grammar under this key too
                "covered_icd_groups": entry.get("qualifying_dx_groups", []),
            })

        # Overlay the parsed MCD-export cache (weekly refresh, see
        # runner._write_coverage_cache): it carries the covered-ICD GROUP
        # roles the flat seed lacks, plus noncovered lists and fresher
        # covered lists. Later entries win per policy_id inside
        # load_coverage_articles (per-policy delete + insert), so the
        # cache — newer and grammar-complete — is appended AFTER the seed.
        # A rebuild without the cache degrades to the flat seed exactly as
        # before (composition gate silently inert, never wrong).
        try:
            if MCD_COVERAGE_CACHE_FILE.exists():
                with open(MCD_COVERAGE_CACHE_FILE) as f:
                    cache = json.load(f)
                cached = [a for a in cache.get("articles", [])
                          if isinstance(a, dict) and a.get("policy_id")]
                if cached:
                    articles.extend(cached)
                    logger.info(
                        f"  lcd_qualifying: MCD-export cache overlaid "
                        f"({len(cached)} articles, fetched "
                        f"{cache.get('fetched_at', '?')})")
        except Exception as exc:
            logger.warning(f"  lcd_qualifying: MCD cache unreadable ({exc}) "
                           f"— continuing with seed only")

        self.load_coverage_articles(articles)

        # Flat qualifying-dx lookup (is_lcd_qualifying()/stats()) — union across
        # all active policies; no per-claim MAC-jurisdiction routing exists to
        # pick a single one.
        dx_rows = {
            (a["policy_id"], _norm(dx)) for a in articles for dx in a["covered_icd"]
            if _norm(dx) != self._MCD_NA_PLACEHOLDER
        }
        self.conn.executemany("INSERT INTO lcd_qualifying VALUES (?,?)", dx_rows)
        self.conn.commit()

        n_cpt = sum(len(a["cpt_codes"]) for a in articles)
        logger.info(f"  lcd_qualifying: {len(articles)} active LCD/Article policies "
                    f"({n_cpt} CPT refs, {len(dx_rows)} DX refs)")

    def _ingest_prior_auth(self) -> None:
        """Loads every data/codes/prior_auth_<payer>.json file generically —
        drop in a new payer's file and it's picked up with no code change.
        Two shapes: {"codes": [...]} for exact-code payers (Medicare DMEPOS)
        and {"categories": [...]} for category-based payers (Tricare) — see
        prior_auth_required() for how each is matched against a claim line.
        """
        rows = []
        files = sorted(CODES_DIR.glob("prior_auth_*.json"))
        for path in files:
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception as exc:
                logger.warning(f"  prior_auth_required: could not load {path.name} ({exc})")
                continue
            payer_id = data.get("payer_id", "")
            source = data.get("source", "")
            for c in data.get("codes", []):
                code = _norm(c.get("code", ""))
                if not code:
                    continue
                rows.append((payer_id, code, None, None, c.get("description", ""), source))
            for cat in data.get("categories", []):
                rows.append((
                    payer_id, None, cat.get("category", ""), cat.get("hcpcs_prefix"),
                    cat.get("note", ""), source,
                ))
        if rows:
            self.conn.executemany(
                "INSERT INTO prior_auth_required VALUES (?,?,?,?,?,?)", rows
            )
            self.conn.commit()
        logger.info(f"  prior_auth_required: {len(rows)} rule(s) from {len(files)} payer file(s)")

    # MCD's "Not Applicable" placeholder row (description literally "Not
    # Applicable") — present when an article's diagnosis section states that
    # diagnosis does not determine coverage. It is a FORMAT sentinel, not a
    # diagnosis: ingesting it made 55 real policies (e.g. A53001 Wound Care)
    # look diagnosis-restricted with a list nothing could ever match, turning
    # each into an unconditional denial gate for every governed CPT.
    _MCD_NA_PLACEHOLDER = "XX000"

    def load_coverage_articles(self, articles: list[dict]) -> None:
        """Ingest MCD Billing & Coding Articles into the coverage tables.

        Each article: {policy_id, title, contractor, cpt_codes: [...],
        covered_icd: [...], noncovered_icd: [...], covered_icd_groups:
        [{group, role, cpt_scope, paragraph, codes}, ...]}. Same shape as
        the LCD record, so the agent needs no change when the MCD bulk
        download is wired into the refresh layer.

        When the SAME policy_id appears more than once, the LAST entry
        wins wholesale — _ingest_lcd deliberately appends the (newer,
        group-carrying) MCD-export cache after the flat seed. A union
        would be worse than either source alone: the seed's ungrouped
        rows read as standalone-role coverage and would silently satisfy
        any composition rule the cache's grouped rows expressed.
        """
        self._ensure_coverage_policy_table()
        by_pid: dict[str, dict] = {}
        for art in articles:
            pid = art.get("policy_id") or art.get("article_id", "")
            by_pid[pid] = art   # later entries replace earlier ones

        # Family supersession: post-Cures-Act, an LCD publishes indications
        # PROSE while its related Billing & Coding article publishes the
        # code lists (with group grammar). The seed data predates that split
        # and flattened the article's codes into the LCD entry too — a flat
        # duplicate that satisfies coverage on any covered dx and thereby
        # masks the article's own composition rules (observed live: L34246's
        # flat 641-code list satisfied B35.1 alone while its article A57193
        # demanded a symptom secondary alongside it). When an article
        # carries group grammar AND names its parent LCD (the export's
        # article_related_documents.csv), the LCD's covered-ICD list is
        # retired — the article's grammar is the family's diagnosis gate.
        # The LCD keeps its CPT rows and title (it still governs the codes;
        # with no dx rules it reads as a documentation policy, exactly how
        # post-Cures LCDs behave).
        superseded_lcds: set[str] = set()
        for art in by_pid.values():
            if art.get("covered_icd_groups") or art.get("qualifying_dx_groups"):
                superseded_lcds.update(art.get("related_lcds") or [])
        superseded_lcds -= {pid for pid, art in by_pid.items()
                            if art.get("covered_icd_groups")
                            or art.get("qualifying_dx_groups")}

        cpt_rows, icd_rows, lcd_rows = [], [], []
        noncov_rows, title_rows, group_rows = [], [], []
        for pid, art in by_pid.items():
            if pid in superseded_lcds:
                art = dict(art, covered_icd=[], covered_icd_groups=[],
                           qualifying_dx_groups=[])
            title_rows.append((pid, art.get("title", "") or "",
                               " ".join(str(art.get("contractor", "") or "").split()),
                               ",".join(s.strip().upper()
                                        for s in (art.get("states") or []) if s)))
            for c in art.get("cpt_codes", []):
                cpt_rows.append((pid, _norm(c)))

            # group grammar first: codes carried by a group get that
            # group's id; only codes in NO group fall back to a flat
            # (NULL-group) row, so a policy's grouped and ungrouped rows
            # can never shadow each other.
            grouped: dict[str, set[int]] = {}
            for g in (art.get("covered_icd_groups")
                      or art.get("qualifying_dx_groups") or []):
                if not isinstance(g, dict):
                    continue
                try:
                    gid = int(g.get("group"))
                except (TypeError, ValueError):
                    continue
                role = str(g.get("role") or "unspecified")
                scope = ",".join(
                    _norm(c) for c in (g.get("cpt_scope") or []) if c)
                group_rows.append((pid, gid, role, scope,
                                   str(g.get("paragraph") or "")[:400]))
                for dx in g.get("codes", []):
                    ndx = _norm(dx)
                    if ndx != self._MCD_NA_PLACEHOLDER:
                        grouped.setdefault(ndx, set()).add(gid)
            for ndx, gids in grouped.items():
                for gid in sorted(gids):
                    icd_rows.append((pid, ndx, gid))
                lcd_rows.append((pid, ndx))
            for dx in art.get("covered_icd", []):
                ndx = _norm(dx)
                if ndx == self._MCD_NA_PLACEHOLDER or ndx in grouped:
                    continue
                icd_rows.append((pid, ndx, None))
                lcd_rows.append((pid, ndx))
            for dx in art.get("noncovered_icd", []):
                if _norm(dx) != self._MCD_NA_PLACEHOLDER:
                    noncov_rows.append((pid, _norm(dx)))
        # Idempotent: replace the policy's rows so repeated refreshes don't duplicate
        for pid in by_pid:
            self.conn.execute("DELETE FROM coverage_cpt WHERE policy_id=?", (pid,))
            self.conn.execute("DELETE FROM coverage_icd WHERE policy_id=?", (pid,))
            self.conn.execute("DELETE FROM coverage_group WHERE policy_id=?", (pid,))
            self.conn.execute("DELETE FROM coverage_icd_noncovered WHERE policy_id=?", (pid,))
            self.conn.execute("DELETE FROM coverage_policy WHERE policy_id=?", (pid,))
            # lcd_qualifying (the flat qualifying-dx union behind
            # is_lcd_qualifying()) was NOT updated here before — article
            # refreshes silently diverged from the coverage tables, so
            # qualifying-dx checks kept answering from seed-era data.
            self.conn.execute("DELETE FROM lcd_qualifying WHERE lcd_id=?", (pid,))
        # Superseded LCDs that are NOT in this batch (e.g. an article-only
        # refresh over a DB whose LCD rows came from an earlier seed ingest)
        # still need their stale covered lists retired.
        for pid in superseded_lcds - set(by_pid):
            self.conn.execute("DELETE FROM coverage_icd WHERE policy_id=?", (pid,))
            self.conn.execute("DELETE FROM lcd_qualifying WHERE lcd_id=?", (pid,))
        self.conn.executemany("INSERT INTO coverage_cpt VALUES (?,?)", cpt_rows)
        self.conn.executemany("INSERT INTO coverage_icd VALUES (?,?,?)", icd_rows)
        self.conn.executemany("INSERT INTO coverage_group VALUES (?,?,?,?,?)", group_rows)
        self.conn.executemany("INSERT INTO coverage_icd_noncovered VALUES (?,?)", noncov_rows)
        self.conn.executemany(
            "INSERT INTO coverage_policy (policy_id, title, contractor, states) "
            "VALUES (?,?,?,?)",
            title_rows,
        )
        self.conn.executemany("INSERT INTO lcd_qualifying VALUES (?,?)", sorted(set(lcd_rows)))
        self.conn.commit()
        logger.info(f"  coverage articles: +{len(cpt_rows)} CPT, +{len(icd_rows)} ICD rows, "
                    f"+{len(group_rows)} group-grammar rows, "
                    f"+{len(noncov_rows)} noncovered-ICD rows, +{len(title_rows)} policy titles"
                    + (f"; {len(superseded_lcds)} LCD covered-list(s) retired in favor of "
                       f"their grammar-carrying articles" if superseded_lcds else ""))

    def _ensure_coverage_policy_table(self) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS coverage_policy ("
            "policy_id TEXT NOT NULL PRIMARY KEY, title TEXT NOT NULL DEFAULT '', "
            # issuing MAC — LCDs/Articles are LOCAL policies that only govern
            # claims in the states their contractor adjudicates (resolved via
            # app.compliance.geo + mac_jurisdictions.json)
            "contractor TEXT NOT NULL DEFAULT '')"
        )
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(coverage_policy)")}
        if "contractor" not in cols:
            self.conn.execute(
                "ALTER TABLE coverage_policy ADD COLUMN contractor TEXT NOT NULL DEFAULT ''"
            )
        # states: the policy's authoritative MAC service area, comma-joined
        # USPS abbreviations from the MCD export's own contractor_jurisdiction
        # ⋈ state_lookup tables (parsers.parse_mcd_export). '' = unknown —
        # coverage_policy_states then falls back to resolving the contractor
        # NAME via mac_jurisdictions.json, the pre-column behavior.
        if "states" not in cols:
            self.conn.execute(
                "ALTER TABLE coverage_policy ADD COLUMN states TEXT NOT NULL DEFAULT ''"
            )
        # Group-N "ICD-10 codes that DO NOT support medical necessity" — the
        # mirror image of coverage_icd. Populated by the MCD bulk-export
        # refresh (the Coverage API seed file exposes only covered lists).
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS coverage_icd_noncovered ("
            "policy_id TEXT NOT NULL, icd_code TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_cov_icd_noncov "
            "ON coverage_icd_noncovered(policy_id, icd_code)"
        )
        # Covered-ICD group grammar (see the schema comment in
        # _create_schema): added when MedicalNecessityAgent gained the
        # claim-composition gate. A DB built before this has coverage_icd
        # without group_id and no coverage_group table.
        icd_cols = {row[1] for row in
                    self.conn.execute("PRAGMA table_info(coverage_icd)")}
        if "group_id" not in icd_cols:
            self.conn.execute(
                "ALTER TABLE coverage_icd ADD COLUMN group_id INTEGER")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS coverage_group ("
            "policy_id TEXT NOT NULL, group_id INTEGER NOT NULL, "
            "role TEXT NOT NULL DEFAULT 'unspecified', "
            "cpt_scope TEXT NOT NULL DEFAULT '', "
            "paragraph TEXT NOT NULL DEFAULT '')"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_cov_group "
            "ON coverage_group(policy_id)"
        )

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

    def _ingest_icd10_code_first(self) -> None:
        """codeFirst_code_refs from icd10cm_instructional_notes.json (parsed
        from CDC/NCHS's own icd10cm-tabular-2026.xml) — real etiology/
        manifestation sequencing pairs, replacing the hardcoded prefix-set
        approximation that was, for at least one pairing checked against
        this real data (H36 paired with E10/E11/E13 diabetes codes), simply
        wrong: H36's actual codeFirst references are lipid storage disorders
        (E75) and sickle-cell disorders (D57) — diabetic retinopathy is a
        self-contained combination code (E11.3x), not an H36+E11 pair at all.
        """
        try:
            with open(ICD10_INSTRUCTIONAL_NOTES_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  icd10_code_first: could not load ({exc})")
            return
        rows = []
        for code, entry in data.get("codes", {}).items():
            lines = entry.get("codeFirst", [])
            for ref in entry.get("codeFirst_code_refs", []):
                rows.append((_norm(code), _norm(ref), _ref_note_line(ref, lines)))
        self.conn.executemany("INSERT INTO icd10_code_first VALUES (?,?,?)", rows)
        logger.info(f"  icd10_code_first: {len(rows)} manifestation->etiology pairs")

    def _ingest_icd10_use_additional_code(self) -> None:
        """useAdditionalCode_code_refs from icd10cm_instructional_notes.json
        — real per-condition companion-code recommendations (e.g. E11
        recommends Z79.4/Z79.84/Z79.85 to identify diabetes medication
        control status). Powers a justified-vs-orphaned Z-code check that
        replaces a prior hardcoded 4-code 'inappropriate for podiatry' list
        whose premise this same real data actually contradicts: Z79.84 is
        the CDC/CMS-recommended companion for E11.x, not something to flag.
        """
        try:
            with open(ICD10_INSTRUCTIONAL_NOTES_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  icd10_use_additional_code: could not load ({exc})")
            return
        rows = []
        for code, entry in data.get("codes", {}).items():
            lines = entry.get("useAdditionalCode", [])
            for ref in entry.get("useAdditionalCode_code_refs", []):
                rows.append((_norm(code), _norm(ref), _ref_note_line(ref, lines)))
        self.conn.executemany("INSERT INTO icd10_use_additional_code VALUES (?,?,?)", rows)
        logger.info(f"  icd10_use_additional_code: {len(rows)} condition->companion-code pairs")

    def _ingest_icd10_code_also(self) -> None:
        """codeAlso_code_refs from icd10cm_instructional_notes.json — the
        Tabular's 'Code also' notes: a second code may be required to fully
        describe the condition, with sequencing left to the encounter
        circumstances (vs codeFirst/useAdditionalCode which mandate order).
        Same shape and evidence-matching note_text as the other two note
        families."""
        try:
            with open(ICD10_INSTRUCTIONAL_NOTES_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  icd10_code_also: could not load ({exc})")
            return
        rows = []
        for code, entry in data.get("codes", {}).items():
            lines = entry.get("codeAlso", [])
            for ref in entry.get("codeAlso_code_refs", []):
                rows.append((_norm(code), _norm(ref), _ref_note_line(ref, lines)))
        self.conn.executemany("INSERT INTO icd10_code_also VALUES (?,?,?)", rows)
        logger.info(f"  icd10_code_also: {len(rows)} condition->companion-code pairs")

    def _ingest_icd10_excludes1(self) -> None:
        """excludes1_code_refs from icd10cm_instructional_notes.json — real
        Type 1 Excludes notes ("not coded here"): the two referenced code
        categories are structurally mutually exclusive per CMS's own
        Tabular List, not just similar codes a coder might choose between.
        Found live: M12.5 (Traumatic arthropathy) carries an explicit
        excludes1 note referencing M19.1 (Post-traumatic osteoarthritis) —
        real data was already loaded (this same source file powers
        code_first/use_additional_code) but this field was never queried.
        """
        try:
            with open(ICD10_INSTRUCTIONAL_NOTES_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  icd10_excludes1: could not load ({exc})")
            return
        rows = []
        for code, entry in data.get("codes", {}).items():
            for ref in entry.get("excludes1_code_refs", []):
                rows.append((_norm(code), _norm(ref)))
        self.conn.executemany("INSERT INTO icd10_excludes1 VALUES (?,?)", rows)
        logger.info(f"  icd10_excludes1: {len(rows)} Type 1 Excludes pairs")

    def _ingest_icd10_includes(self) -> None:
        """includes_code_refs from icd10cm_instructional_notes.json — the
        Tabular List's subsumption notes: a code whose Includes note reads
        "any condition classifiable to X" CAPTURES X's condition within
        itself when both apply, so billing both is redundant per the Tabular
        List's own hierarchy. Found live: I70.23- (atherosclerosis with
        ulceration) includes "any condition classifiable to I70.211 and
        I70.221" — rest pain is subsumed by ulceration of the same limb,
        only the ulceration code is reported. Same source file/field family
        as excludes1; this field was parsed but never queried."""
        try:
            with open(ICD10_INSTRUCTIONAL_NOTES_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  icd10_includes: could not load ({exc})")
            return
        rows = []
        for code, entry in data.get("codes", {}).items():
            for ref in entry.get("includes_code_refs", []):
                rows.append((_norm(code), _norm(ref)))
        self.conn.executemany("INSERT INTO icd10_includes VALUES (?,?)", rows)
        logger.info(f"  icd10_includes: {len(rows)} subsumption (Includes) refs")

    def _ingest_icd10_tabular_desc(self) -> None:
        """Every Tabular List entry's description from
        icd10cm_instructional_notes.json — including category-level entries
        (Z79, I70.2, ...) absent from the billable-codes file. Needed to
        resolve instructional-note refs (which often point at categories) to
        the clinical terms used for documentation-evidence matching."""
        try:
            with open(ICD10_INSTRUCTIONAL_NOTES_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  icd10_tabular_desc: could not load ({exc})")
            return
        rows = [
            (_norm(code), entry.get("description", ""))
            for code, entry in data.get("codes", {}).items()
            if entry.get("description")
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO icd10_tabular_desc VALUES (?,?)", rows
        )
        logger.info(f"  icd10_tabular_desc: {len(rows)} Tabular entry descriptions")

    def _ingest_icd10_inclusion_terms(self) -> None:
        """Tabular List inclusionTerm phrases — the official synonyms an
        entry answers to (L03.0 lists 'Paronychia', B35.1 lists
        'Onychomycosis'). A note documenting a synonym IS documenting the
        code; evidence matching that only sees the one-line description
        calls such codes unsupported (false positive) or fails to see that
        a sibling matches the documentation (false negative)."""
        try:
            with open(ICD10_INSTRUCTIONAL_NOTES_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  icd10_inclusion_term: could not load ({exc})")
            return
        rows = []
        for code, entry in data.get("codes", {}).items():
            terms = entry.get("inclusionTerm") or []
            if isinstance(terms, str):
                terms = [terms]
            for t in terms:
                if str(t).strip():
                    rows.append((_norm(code), str(t).strip()))
        self.conn.executemany(
            "INSERT INTO icd10_inclusion_term VALUES (?,?)", rows
        )
        logger.info(f"  icd10_inclusion_term: {len(rows)} Tabular inclusion terms")

    def _ingest_icd10_index_terms(self) -> None:
        """ICD-10-CM Alphabetic Index phrases (icd10cm_index_terms.json,
        parsed from the official CDC Index XML): every lookup path leading
        to a code, plus one-hop see/see-also aliases. This is the code
        set's own answer to 'what may a clinician call this condition' —
        e.g. 'paronychia' resolves to the L03.0x cellulitis family and NOT
        to L03.04x lymphangitis."""
        try:
            with open(ICD10_INDEX_TERMS_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  icd10_index_term: could not load ({exc})")
            return
        rows = [
            (_norm(code), term)
            for code, terms in data.get("terms", {}).items()
            for term in terms if term
        ]
        self.conn.executemany("INSERT INTO icd10_index_term VALUES (?,?)", rows)
        logger.info(f"  icd10_index_term: {len(rows)} Alphabetic Index phrases")

    def _ingest_icd10_chronic(self) -> None:
        """AHRQ HCUP Chronic Condition Indicator Refined (icd10cm_chronic.json,
        parsed from CCIR_v<fy>.csv): chronicity flag for every ICD-10-CM code
        (1=chronic, 0=not chronic, 9=no determination)."""
        try:
            with open(ICD10_CHRONIC_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  icd10_chronic: could not load ({exc})")
            return
        rows = [
            (_norm(code), int(ind))
            for code, ind in data.get("codes", {}).items()
            if code and ind in (0, 1, 9)
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO icd10_chronic VALUES (?,?)", rows)
        logger.info(f"  icd10_chronic: {len(rows)} CCIR chronicity flags")

    def _ingest_mce_edits(self) -> None:
        """CMS Medicare Code Editor diagnosis edit lists (mce_edits.json,
        parsed from CMS's own 'Definitions of Medicare Code Edits' text) —
        age-conflict categories with the MCE's own age ranges, manifestation-
        as-principal, and unacceptable-principal-diagnosis lists. Sex
        conflict is deliberately absent: CMS deactivated it for all ICD-10
        codes as of 10/01/2024 (documented in the file's excluded_edits)."""
        try:
            with open(MCE_EDITS_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  mce_edit: could not load ({exc})")
            return
        rows = []
        for family, entries in data.get("codes", {}).items():
            for e in entries:
                code = _norm(e.get("code", ""))
                if code:
                    rows.append((family, code))
        self.conn.executemany("INSERT INTO mce_edit VALUES (?,?)", rows)
        ranges = [
            (cat, int(r.get("min_age", 0)), int(r.get("max_age", 0)))
            for cat, r in data.get("age_categories", {}).items()
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO mce_age_range VALUES (?,?,?)", ranges
        )
        logger.info(f"  mce_edit: {len(rows)} codes across "
                    f"{len(data.get('codes', {}))} MCE edit families")

    def _ingest_modifiers(self) -> None:
        try:
            with open(MODIFIERS_FILE) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(f"  modifier: could not load ({exc})")
            return
        # Code-system provenance comes from the file's own real fields — each
        # entry self-reports its "system" (CPT/HCPCS), and the file-level
        # sources[].covers lists say which authoritative source (AMA CPT
        # Appendix A vs CMS HCPCS Level II file) contributed each modifier; a
        # modifier can be cross-listed by both. A prior version read a
        # "systems" (plural) key that never existed in the file, so every row
        # was stored with an empty systems value and any consumer filtering on
        # it matched nothing.
        source_tags: dict[str, set[str]] = {}
        for src in data.get("sources", []) or []:
            src_name = (src.get("name") or "").lower()
            tag = "cpt" if "cpt" in src_name else ("hcpcs" if "hcpcs" in src_name else None)
            if not tag:
                continue
            for covered in src.get("covers", []) or []:
                source_tags.setdefault(str(covered).upper(), set()).add(tag)
        rows = []
        for code, entry in data.get("modifiers", {}).items():
            name = entry.get("name", "") if isinstance(entry, dict) else str(entry)
            systems = set(source_tags.get(code.upper(), set()))
            own_system = (entry.get("system") or "").strip().lower() if isinstance(entry, dict) else ""
            if own_system:
                systems.add(own_system)
            rows.append((code.upper(), name, ",".join(sorted(systems))))
        self.conn.executemany("INSERT INTO modifier VALUES (?,?,?)", rows)
        logger.info(f"  modifier: {len(rows)} recognized modifiers")

    # ---------------------------------------------------------------- lookup
    def is_addon(self, code: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM addon WHERE code=? LIMIT 1", (_norm(code),)
        ).fetchone() is not None

    def code_first_etiology_refs(self, code: str) -> list[str]:
        """Etiology code(s)/prefix(es) that must be sequenced before `code`
        per its real ICD-10-CM Tabular List codeFirst notes — empty if `code`
        isn't a manifestation code under this convention. Aggregated across
        the WHOLE ancestor chain (e.g. H36811 + H3681 + H368 + H36): every
        level's note is a real, independent Tabular instruction. A prior
        version stopped at the first level with rows, which let a child's
        own note SHADOW its category's note entirely."""
        return self._all_note_refs("icd10_code_first", "etiology_ref", code)

    def use_additional_code_refs(self, code: str) -> list[str]:
        """Companion code(s)/prefix(es) recommended by this code's real
        ICD-10-CM Tabular List useAdditionalCode notes (e.g. E11 -> Z79.4/
        Z79.84/Z79.85) — empty if no level of `code`'s chain carries one.
        Aggregated across ALL ancestor levels: E11.621 carries its own note
        (ulcer site, L97.4-/L97.5-) AND inherits E11's (insulin/antidiabetic
        Z79.x). The prior first-hit-wins lookup returned only the child's
        note, so Z79.4 on an E11.621 claim was flagged as 'unjustified'
        even though E11's own note is precisely what justifies it (observed
        live on three claims)."""
        return self._all_note_refs("icd10_use_additional_code", "ref", code)

    def _all_note_refs(self, table: str, col: str, code: str) -> list[str]:
        norm = _norm(code)
        out: list[str] = []
        seen = set()
        for length in range(len(norm), 2, -1):
            rows = self.conn.execute(
                f"SELECT DISTINCT {col} FROM {table} WHERE code=?", (norm[:length],)
            ).fetchall()
            for r in rows:
                if r[0] not in seen:
                    seen.add(r[0])
                    out.append(r[0])
        return out

    def _note_ref_groups(self, table: str, col: str, code: str) -> list[tuple[str, list[tuple[str, str]]]]:
        """(carrier, [(ref, note_line), ...]) for every instructional-note-
        carrying ancestor of `code` — one group per note carrier, ALL levels
        of the chain (a billed E11.621 can inherit distinct notes from E11.62
        AND E11). Grouping matters because a single note's refs are
        ALTERNATIVES ("...to identify control: Z79.4, Z79.84, Z79.85") —
        satisfying any one ref satisfies that carrier's note, so callers must
        not treat each ref as an independent requirement. note_line is the
        Tabular's own wording for the ref ('dialysis status (Z99.2)') — see
        _ref_note_line."""
        norm = _norm(code)
        groups: list[tuple[str, list[tuple[str, str]]]] = []
        for length in range(len(norm), 2, -1):
            prefix = norm[:length]
            rows = self.conn.execute(
                f"SELECT DISTINCT {col}, note_text FROM {table} WHERE code=?", (prefix,)
            ).fetchall()
            if rows:
                groups.append((prefix, [(r[0], r[1] or "") for r in rows]))
        return groups

    def use_additional_code_groups(self, code: str) -> list[tuple[str, list[tuple[str, str]]]]:
        """(carrier, [(companion ref, note line), ...]) per useAdditionalCode
        note on `code`'s ancestor chain."""
        return self._note_ref_groups("icd10_use_additional_code", "ref", code)

    def code_first_groups(self, code: str) -> list[tuple[str, list[tuple[str, str]]]]:
        """(carrier, [(etiology ref, note line), ...]) per codeFirst note on
        `code`'s ancestor chain."""
        return self._note_ref_groups("icd10_code_first", "etiology_ref", code)

    def code_also_groups(self, code: str) -> list[tuple[str, list[tuple[str, str]]]]:
        """(carrier, [(companion ref, note line), ...]) per codeAlso note on
        `code`'s ancestor chain — 'Code also' means a second code may be
        required for the full picture, sequencing discretionary."""
        return self._note_ref_groups("icd10_code_also", "ref", code)

    def icd10_is_chronic(self, code: str) -> bool | None:
        """AHRQ CCIR chronicity for an ICD-10-CM code: True=chronic,
        False=not chronic, None=no determination (CCIR value 9) or code not
        in the CCIR file. Callers needing certainty must treat None as
        'unknown', never as either definite answer."""
        row = self.conn.execute(
            "SELECT chronic FROM icd10_chronic WHERE code=? LIMIT 1",
            (_norm(code),),
        ).fetchone()
        if row is None or row["chronic"] == 9:
            return None
        return row["chronic"] == 1

    def icd10_inclusion_terms(self, code: str, min_level: int = 3) -> list[str]:
        """The Tabular List's official synonym phrases for a code, including
        ancestor levels' — a parent's inclusion terms (L03.0 'Paronychia')
        describe all its children (L03.031). min_level restricts to levels
        at/below a given prefix length: when comparing two siblings, terms
        from their COMMON ancestor describe both codes and must not count
        as evidence for either one specifically."""
        norm = _norm(code)
        terms: list[str] = []
        for ln in range(max(3, min_level), len(norm) + 1):
            rows = self.conn.execute(
                "SELECT term FROM icd10_inclusion_term WHERE code=?", (norm[:ln],)
            ).fetchall()
            terms.extend(r[0] for r in rows)
        return terms

    def icd10_category_descriptions(self, chapter_prefixes: tuple = ()) -> list[tuple[str, str]]:
        """(code, description) for every 3-character ICD-10-CM category
        ('S92' → 'Fracture of foot and toe, except ankle'), optionally
        restricted to chapters by first letter. Category headings name the
        disease entity first ('Fracture of...', 'Dislocation and sprain
        of...') — the authoritative source for condition-word vocabularies."""
        rows = self.conn.execute(
            "SELECT code, description FROM icd10_tabular_desc WHERE LENGTH(code)=3"
        ).fetchall()
        out = [(r[0], r[1]) for r in rows]
        if chapter_prefixes:
            out = [(c, d) for c, d in out if c.startswith(tuple(chapter_prefixes))]
        return out

    def icd10_index_terms(self, code: str, min_level: int = 3) -> list[str]:
        """Alphabetic Index phrases that resolve to this code, including
        phrases attached to its ancestor stems (the Index often points at a
        stem like L03.03- that covers all its children). min_level restricts
        to stems at/below a given prefix length, mirroring
        icd10_inclusion_terms: when the question is whether a SPECIFIC
        family member is supported (vs. its unspecified sibling), phrases
        the Index attaches to the 3-char category describe the whole family
        and prove neither member — min_level=4 excludes them."""
        norm = _norm(code)
        terms: list[str] = []
        for ln in range(max(3, min_level), len(norm) + 1):
            rows = self.conn.execute(
                "SELECT term FROM icd10_index_term WHERE code=?", (norm[:ln],)
            ).fetchall()
            terms.extend(r[0] for r in rows)
        return terms

    def icd10_tabular_description(self, code: str) -> str:
        """Tabular List description for any entry — billable OR category-level
        (e.g. Z79, I70.2) — from the CDC/NCHS Tabular XML. Empty string if
        the entry doesn't exist."""
        row = self.conn.execute(
            "SELECT description FROM icd10_tabular_desc WHERE code=? LIMIT 1",
            (_norm(code),),
        ).fetchone()
        return row[0] if row else ""

    def icd10_billable_under(self, ref: str) -> list[tuple[str, str]]:
        """(code, description) for every billable ICD-10-CM code in code_set
        that the given ref/prefix covers — a leaf ref (Z99.2) returns itself,
        a category ref (Z79) returns all its billable children. Ordered so an
        exact match comes first."""
        norm = _norm(ref)
        rows = self.conn.execute(
            "SELECT code, description FROM code_set "
            "WHERE code_system='ICD10' AND code LIKE ? || '%' "
            "ORDER BY LENGTH(code), code",
            (norm,),
        ).fetchall()
        return [(r[0], r[1] or "") for r in rows]

    def excludes1_refs(self, code: str) -> list[str]:
        """Code(s)/prefix(es) this code's real ICD-10-CM Tabular List Type 1
        Excludes note says cannot be coded together with it — empty if
        `code` carries no such note. Same prefix-matching fallback as
        code_first_etiology_refs/use_additional_code_refs."""
        norm = _norm(code)
        for length in range(len(norm), 2, -1):
            prefix = norm[:length]
            rows = self.conn.execute(
                "SELECT DISTINCT excluded_ref FROM icd10_excludes1 WHERE code=?", (prefix,)
            ).fetchall()
            if rows:
                return [r[0] for r in rows]
        return []

    def excludes1_conflict(self, code1: str, code2: str) -> bool:
        """True if code1/code2 are Type 1 Excludes of each other. Checks
        both directions — the source Tabular List note isn't always
        annotated symmetrically (e.g. M12.5 lists M19.1 in its own
        excludes1, but M19.1's entry doesn't reciprocally list M12.5), even
        though the exclusion holds both ways in practice."""
        n1, n2 = _norm(code1), _norm(code2)
        refs1 = self.excludes1_refs(code1)
        if any(n2.startswith(r) or r.startswith(n2) for r in refs1):
            return True
        refs2 = self.excludes1_refs(code2)
        return any(n1.startswith(r) or r.startswith(n1) for r in refs2)

    def includes_subsumption(self, keeper: str, other: str) -> str | None:
        """The Tabular List code whose Includes note subsumes `other` within
        `keeper` — None when no subsumption applies.

        `keeper` subsumes `other` when a note-carrying ancestor of keeper
        (e.g. I70.23 for keeper I70.235) lists an includes ref covering
        `other` (e.g. I70.221) — the Tabular List's own statement that the
        other condition is classifiable INTO keeper's code, so billing both
        is redundant. Two data-driven guards:
        - the note carrier must NOT also be an ancestor of `other` (a
          category-level note like I05's covers all its own children — that
          says nothing about redundancy between two siblings);
        - no useAdditionalCode note on keeper's chain may reference `other`
          (combination categories like I11/I12 subsume the causal condition
          but explicitly REQUIRE the companion code — I11.0 says "use
          additional code ... (I50.-)", so I50.x alongside I11.0 is
          mandated, not redundant).
        """
        nk, no = _norm(keeper), _norm(other)
        if not nk or not no or nk == no:
            return None
        carriers = self.conn.execute(
            "SELECT DISTINCT code, included_ref FROM icd10_includes WHERE ? LIKE code || '%'",
            (nk,),
        ).fetchall()
        hit = None
        for carrier, ref in carriers:
            if no.startswith(carrier):
                continue  # carrier covers both codes — sibling note, not subsumption
            if no.startswith(ref):
                hit = carrier
                break
        if hit is None:
            return None
        companions = self.conn.execute(
            "SELECT DISTINCT ref FROM icd10_use_additional_code WHERE ? LIKE code || '%'",
            (nk,),
        ).fetchall()
        if any(no.startswith(r[0]) for r in companions):
            return None
        return hit

    def mce_families(self, icd_code: str) -> set[str]:
        """MCE edit families this diagnosis belongs to (age_newborn,
        age_pediatric, age_maternity, age_adult, manifestation_not_pdx,
        unacceptable_pdx, unacceptable_pdx_unless_secondary) — empty set if
        none. Exact-code membership: the MCE lists enumerate billable codes
        individually, not by category prefix."""
        rows = self.conn.execute(
            "SELECT DISTINCT family FROM mce_edit WHERE code=?", (_norm(icd_code),)
        ).fetchall()
        return {r[0] for r in rows}

    def mce_age_range(self, category: str) -> tuple[int, int] | None:
        """(min_age, max_age) inclusive for an MCE age-conflict category —
        the range comes from the MCE definitions text itself."""
        row = self.conn.execute(
            "SELECT min_age, max_age FROM mce_age_range WHERE category=?",
            (category.replace("age_", ""),),
        ).fetchone()
        return (row[0], row[1]) if row else None

    def pfs_indicators(self, code: str, dos=None) -> dict:
        """Per-code CMS PFS payment-policy indicators (PC/TC split,
        multiple-procedure, bilateral, assistant/co-/team-surgeon) — the
        code-specific modifier-validity signals CMS itself publishes.
        Empty dict when the code isn't on the fee schedule."""
        row = self._asof(
            "global_period",
            "pctc_ind, mult_proc, bilat_surg, asst_surg, co_surg, team_surg",
            "code=?", (_norm(code),), self._dos(dos),
        )
        if not row:
            return {}
        return {k: row[k] for k in row.keys() if row[k] is not None}

    def hcpcs_noncoverage_reason(self, code: str) -> str | None:
        """Medicare non-coverage reason from the HCPCS file's own coverage
        code — 'I' (not payable by Medicare), 'M' (non-covered), 'S'
        (non-covered by statute). None for covered/carrier-judgment codes
        (C/D) or codes not in the HCPCS file."""
        row = self.conn.execute(
            "SELECT coverage_code FROM hcpcs_coverage WHERE code=?", (_norm(code),)
        ).fetchone()
        if not row:
            return None
        meanings = {
            "I": "coverage code 'I' — not payable by Medicare",
            "M": "coverage code 'M' — non-covered by Medicare",
            "S": "coverage code 'S' — non-covered by Medicare statute",
        }
        return meanings.get(row[0])

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

    def modifier_count(self) -> int:
        """Number of recognized modifiers ingested. Consumers that REMOVE
        modifiers based on this table (e.g. the coding-time strip) must treat
        0 as 'data source absent — do not strip anything' rather than 'no
        modifier is valid': with an empty table every modifier looks
        unrecognized and a strip would delete all of them from every claim."""
        return self.conn.execute("SELECT COUNT(*) FROM modifier").fetchone()[0]

    def modifier_laterality(self, mod: str) -> str | None:
        """'RT' / 'LT' if this modifier denotes a body side, else None —
        derived from the modifier's own name in the AMA/CMS reference data
        ('Left foot, great toe' → LT; 'Right side …' → RT), never from a
        hardcoded toe-modifier→side mapping. A prior hand-written mapping in
        validator.py had TA/T1–T4 and T5–T9 INVERTED (TA is the LEFT great
        toe per this very table, but was mapped to RT) — deriving from the
        reference data makes that class of transcription error impossible."""
        row = self.conn.execute(
            "SELECT category FROM modifier WHERE code=? LIMIT 1", (str(mod).strip().upper(),)
        ).fetchone()
        if row is None:
            return None
        name = (row["category"] or "").lower()
        has_left, has_right = "left" in name, "right" in name
        if has_left and not has_right:
            return "LT"
        if has_right and not has_left:
            return "RT"
        return None

    def modifier_name(self, mod: str) -> str | None:
        """The modifier's own AMA/CMS name ('Right foot, great toe' for T5),
        or None if the modifier isn't in the reference data. Generic
        accessor for checks that need the name's wording itself (digit
        designators, side words) rather than a pre-digested judgment like
        modifier_laterality."""
        row = self.conn.execute(
            "SELECT category FROM modifier WHERE code=? LIMIT 1", (str(mod).strip().upper(),)
        ).fetchone()
        return row["category"] if row else None

    def anatomic_modifiers(self) -> set[str]:
        """Modifiers whose OWN AMA/CMS name designates an anatomic site —
        laterality (RT/LT: 'Right side…'), digits (FA/F1–F9, TA/T1–T9:
        'Left foot, great toe'), eyelids (E1–E4: 'Upper left, eyelid') and
        coronary arteries (LC/LD/LM/RC/RI). These are CMS's NCCI-associated
        anatomic modifiers: two procedures carrying DIFFERENT anatomic
        modifiers document distinct sites, which bypasses a PTP edit with
        indicator 1 exactly like 59/X{EPSU}. Derived from each modifier's
        own name in the reference data, never a curated list (same rationale
        as modifier_laterality above — a hand-typed T-modifier table was
        once transcribed inverted)."""
        rows = self.conn.execute("SELECT code, category FROM modifier").fetchall()
        out = set()
        for r in rows:
            name = (r["category"] or "").lower()
            if re.search(r"\b(left|right)\b", name) or "coronary artery" in name:
                out.add(r["code"])
        return out

    def telehealth_modifiers(self) -> set[str]:
        """Modifiers whose OWN AMA/CMS name designates a telemedicine/
        telehealth service (95, 93, GT, GQ, FQ — whichever the reference
        data actually contains) — derived from the modifier table's names,
        never a hardcoded list, so newly added telehealth modifiers are
        picked up automatically on re-ingest."""
        rows = self.conn.execute(
            "SELECT code FROM modifier WHERE lower(category) LIKE '%telemedicine%' "
            "OR lower(category) LIKE '%telehealth%'"
        ).fetchall()
        return {r["code"] for r in rows}

    def pos_is_telehealth(self, pos: str) -> bool:
        """True if the POS code's own CMS name designates telehealth —
        data-driven from the ingested POS set, not a hardcoded {02, 10}."""
        info = self.pos_info(pos)
        return bool(info and "telehealth" in (info.get("name") or "").lower())

    def modifier_valid_for_cpt(self, mod: str) -> bool:
        """True if this modifier is cross-listed in AMA's own CPT modifier data
        (systems includes 'cpt') — narrower than modifier_valid(), which only
        checks whether the modifier is recognized at all (CPT or HCPCS).

        NOTE: this is AMA CPT-book scope, NOT claim-form billability — CMS
        Level II modifiers absent from AMA's cross-listing (Q7–Q9 routine-
        foot-care class findings, KX, GA/GX/GY/GZ, QW…) are still legitimately
        appended to CPT lines on CMS-1500 claims per CMS billing rules. Use
        this as advisory/annotation data only; never as a removal gate."""
        row = self.conn.execute(
            "SELECT systems FROM modifier WHERE code=? LIMIT 1", (str(mod).strip().upper(),)
        ).fetchone()
        return row is not None and "cpt" in (row[0] or "").split(",")

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

    def coverage_groups(self, policy_id: str) -> list[dict]:
        """Covered-ICD group grammar for one policy: [{group_id, role,
        cpt_scope (set, empty = all governed codes), paragraph}, ...].
        Empty list = policy ingested without group structure (flat seed) —
        callers must then evaluate coverage exactly as before groups
        existed (any covered dx satisfies), never stricter."""
        rows = self.conn.execute(
            "SELECT group_id, role, cpt_scope, paragraph FROM coverage_group "
            "WHERE policy_id=? ORDER BY group_id", (policy_id,)
        ).fetchall()
        return [{
            "group_id": r["group_id"],
            "role": r["role"] or "unspecified",
            "cpt_scope": {c for c in (r["cpt_scope"] or "").split(",") if c},
            "paragraph": r["paragraph"] or "",
        } for r in rows]

    def coverage_dx_groups(self, policy_id: str, icd_code: str) -> list:
        """Group ids the diagnosis appears in under this policy. [] = not
        on the policy's covered list at all; [None] = covered by a flat
        (ungrouped) ingest — standalone by definition."""
        rows = self.conn.execute(
            "SELECT group_id FROM coverage_icd WHERE policy_id=? AND icd_code=?",
            (policy_id, _norm(icd_code)),
        ).fetchall()
        return [r["group_id"] for r in rows]

    def coverage_icd_explicitly_noncovered(self, policy_id: str, icd_code: str) -> bool:
        """Whether the policy's Group-N list explicitly names this diagnosis
        as NOT supporting medical necessity — the mirror image of the covered
        list, and the only diagnosis signal for policies that publish a
        noncovered list without a covered one."""
        return self.conn.execute(
            "SELECT 1 FROM coverage_icd_noncovered WHERE policy_id=? AND icd_code=? LIMIT 1",
            (policy_id, _norm(icd_code)),
        ).fetchone() is not None

    def coverage_policy_has_dx_rules(self, policy_id: str) -> bool:
        """Whether the policy publishes ANY covered-diagnosis list. Policies
        without one (e.g. broad PT/OT billing articles) impose documentation
        rules, not an ICD gate — treating their empty list as "no diagnosis
        can ever satisfy this" turned every such policy into an unconditional
        denial for every claim billing a governed code."""
        return self.conn.execute(
            "SELECT 1 FROM coverage_icd WHERE policy_id=? LIMIT 1", (policy_id,)
        ).fetchone() is not None

    def coverage_policy_states(self, policy_id: str) -> set[str] | None:
        """States where the policy's issuing MAC adjudicates claims.
        Preferred source: the policy's own states column, populated from
        the MCD export's contractor_jurisdiction ⋈ state_lookup tables (the
        contractor's authoritative service area). Fallback: resolving the
        contractor NAME via mac_jurisdictions.json — the only signal for
        seed-era rows ingested before the column existed. None = can't be
        narrowed — caller must treat the policy as potentially applicable
        everywhere."""
        from app.compliance import geo
        row = self.conn.execute(
            "SELECT contractor, states FROM coverage_policy "
            "WHERE policy_id=? LIMIT 1",
            (policy_id,),
        ).fetchone()
        if row is None:
            return None
        states = {s for s in (row["states"] or "").split(",") if s}
        if states:
            return states
        return geo.contractor_states(row["contractor"])

    def policy_applies_in_state(self, policy_id: str, state: str | None) -> bool:
        """False only when BOTH the claim's state and the policy's MAC service
        area are known and disjoint — unknowns stay conservative (apply)."""
        if not state:
            return True
        states = self.coverage_policy_states(policy_id)
        return states is None or state.upper() in states

    def prior_auth_required(self, code: str, payer_id: str) -> dict | None:
        """payer_id must be the canonical id from payers.json (e.g. "medicare",
        "tricare"), not the human-readable payer name — that's how
        prior_auth_required rows are keyed. Checks exact-code rules first
        (e.g. Medicare DMEPOS), then category rules by HCPCS letter prefix
        (e.g. Tricare's "Durable Medical Equipment" -> E-codes)."""
        if not payer_id:
            return None
        code = _norm(code)
        row = self.conn.execute(
            "SELECT code, category, note FROM prior_auth_required "
            "WHERE payer=? AND code=? LIMIT 1",
            (payer_id, code),
        ).fetchone()
        if row:
            return dict(row)
        prefix = code[0] if code and code[0].isalpha() else None
        if prefix:
            row = self.conn.execute(
                "SELECT code, category, note FROM prior_auth_required "
                "WHERE payer=? AND hcpcs_prefix=? LIMIT 1",
                (payer_id, prefix),
            ).fetchone()
            if row:
                return dict(row)
        return None

    def prior_auth_policy_available(self, payer_id: str | None) -> bool:
        """Whether this payer has an authoritative PA policy loaded.

        A missing code row only means "PA not required" after the payer's
        policy corpus is known to be present.  Without this distinction the
        absence of an entire payer feed silently became a pass.
        """
        if not payer_id:
            return False
        row = self.conn.execute(
            "SELECT 1 FROM prior_auth_required WHERE payer=? LIMIT 1",
            (payer_id,),
        ).fetchone()
        return row is not None

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

    def _ncci_release_bounds(self) -> tuple[date, date] | None:
        """Return the loaded snapshot's quarter, querying SQLite once.

        Pair-heavy validator paths can perform thousands of NCCI lookups for
        one claim. Re-running ``MAX(effective_from)`` for every pair made the
        fail-closed availability guard dominate runtime. The release only
        changes when this store ingests NCCI data, which invalidates the cache.
        ``getattr`` keeps lightweight ``__new__`` test stores compatible.
        """
        if getattr(self, "_ncci_release_window_loaded", False):
            return getattr(self, "_ncci_release_window", None)
        row = self.conn.execute(
            "SELECT MAX(effective_from) AS release_start FROM ncci_ptp",
        ).fetchone()
        bounds = None
        if row and row["release_start"]:
            try:
                release_date = date.fromisoformat(row["release_start"])
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
        """Whether an NCCI release in the local store actually covers DOS.

        The imported CMS file is a quarterly snapshot, not a complete edit
        history: active rows commonly have an open-ended effective_to. The
        newest effective_from in the snapshot identifies its release quarter;
        open-ended rows must not make that one snapshot appear valid forever.
        """
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

    def ncci_pair(self, c1: str, c2: str, dos=None) -> dict | None:
        """Return an NCCI edit only from the release covering the claim DOS."""
        if not self.ncci_data_available(dos):
            return None
        d = dos if isinstance(dos, str) else dos.isoformat()
        for a, b in ((c1, c2), (c2, c1)):
            row = self.conn.execute(
                "SELECT col1, col2, modifier_indicator FROM ncci_ptp "
                "WHERE col1=? AND col2=? AND effective_from<=? AND effective_to>=? "
                "ORDER BY effective_from DESC LIMIT 1",
                (_norm(a), _norm(b), d, d),
            ).fetchone()
            if row:
                return {"col1": row["col1"], "col2": row["col2"],
                        "modifier_indicator": row["modifier_indicator"]}
        return None

    def _mue_release_bounds(self) -> tuple[date, date] | None:
        if getattr(self, "_mue_release_window_loaded", False):
            return getattr(self, "_mue_release_window", None)
        rows = self.conn.execute(
            "SELECT DISTINCT effective_from AS release_start FROM mue "
            "WHERE effective_from IS NOT NULL AND effective_from<>''",
        ).fetchall()
        windows = []
        for row in rows:
            try:
                release_date = date.fromisoformat(row["release_start"])
            except (TypeError, ValueError):
                continue
            quarter_index = (release_date.month - 1) // 3
            quarter_end_month = (quarter_index + 1) * 3
            quarter_end = date(
                release_date.year, quarter_end_month,
                calendar.monthrange(release_date.year, quarter_end_month)[1])
            # CMS MUE exports can label a release with the prior quarter's
            # closing date (03-31 for the release governing 04-01 onward).
            if release_date == quarter_end:
                start = (date(release_date.year + 1, 1, 1)
                         if quarter_end_month == 12 else
                         date(release_date.year, quarter_end_month + 1, 1))
            else:
                start = date(release_date.year, quarter_index * 3 + 1, 1)
            end_month = (((start.month - 1) // 3) + 1) * 3
            windows.append((
                start,
                date(start.year, end_month,
                     calendar.monthrange(start.year, end_month)[1]),
            ))
        windows = sorted(set(windows))
        bounds = windows[-1] if windows else None
        self._mue_release_windows = tuple(windows)
        self._mue_release_window = bounds
        self._mue_release_window_loaded = True
        return bounds

    def mue_data_available(self, dos=None) -> bool:
        """True only when the loaded quarterly MUE release covers DOS."""
        if dos is None:
            return False
        try:
            d = date.fromisoformat(dos) if isinstance(dos, str) else dos
        except (TypeError, ValueError):
            return False
        self._mue_release_bounds()
        return any(start <= d <= end for start, end in
                   getattr(self, "_mue_release_windows", ()))

    def mue(self, code: str, dos=None) -> dict | None:
        if not self.mue_data_available(dos):
            return None
        d = self._dos(dos)
        row = self.conn.execute(
            "SELECT mue_value, mai, rationale FROM mue WHERE code=? "
            "AND effective_from<=? AND effective_to>=? "
            "ORDER BY effective_from DESC LIMIT 1",
            (_norm(code), d, d),
        ).fetchone()
        return dict(row) if row else None

    def global_period(self, code: str, dos=None) -> str | None:
        row = self.conn.execute(
            "SELECT glob_days FROM global_period WHERE code=? LIMIT 1", (_norm(code),)
        ).fetchone()
        return row["glob_days"] if row else None

    def billing_status(self, code: str, dos=None) -> str | None:
        """Raw status letter from global_periods.json (A/B/C/I/N/R/T/X, plus
        E/M/J/P which appear in the source but aren't documented by its own
        indicator_meanings — returned as-is; interpretation lives in
        not_separately_billable_reason / pfs_exclusion_advisory)."""
        row = self.conn.execute(
            "SELECT billing_status FROM global_period WHERE code=? LIMIT 1", (_norm(code),)
        ).fetchone()
        return row["billing_status"] if row else None

    def em_mdm_level(self, code: str) -> str | None:
        """The MDM level an E/M code requires, read from that code's OWN
        AMA descriptor ('... low level of medical decision making ...') —
        never a code table. None for codes not leveled by MDM (99211)."""
        row = self.conn.execute(
            "SELECT description FROM code_set WHERE code_system='CPT' "
            "AND code=? LIMIT 1", (_norm(code),)
        ).fetchone()
        if not row or not row["description"]:
            return None
        m = re.search(
            r"(straightforward|low|moderate|high)\s+(?:level\s+of\s+)?"
            r"medical decision making",
            row["description"], re.IGNORECASE,
        )
        return m.group(1).lower() if m else None

    def mdm_grid(self, dos=None) -> dict | None:
        """The AMA MDM level-selection table (licensed CPT content,
        data/codes/em_mdm_grid.json) in the revision effective on `dos` —
        the structured authority for E/M leveling: per level, the
        problems-addressed definitions, the data categories with their
        combination thresholds, and the risk examples, plus the CPT
        2-of-3-elements selection rule. None when the file is absent
        (deployments without the licensed content degrade gracefully)."""
        grid = getattr(self, "_mdm_grid_cache", None)
        if grid is None:
            try:
                grid = json.loads(EM_MDM_GRID_FILE.read_text())
            except (OSError, ValueError):
                grid = {}
            self._mdm_grid_cache = grid
        revisions = grid.get("revisions") or []
        if not revisions:
            return None
        d = self._dos(dos)
        for rev in revisions:
            if (str(rev.get("effective_from") or "1900-01-01") <= d
                    <= str(rev.get("effective_to") or _OPEN)):
                return {"selection_rule": grid.get("selection_rule", ""),
                        "time_rule": grid.get("time_rule", ""),
                        "source": grid.get("source", ""),
                        "effective_from": rev.get("effective_from"),
                        "levels": rev.get("levels") or {}}
        return None

    def mdm_requirements(self, code: str, dos=None) -> dict | None:
        """The MDM grid row a specific E/M code must satisfy: level from
        the code's own descriptor, row from the grid revision in force on
        the DOS. The data-driven answer to 'what documentation supports
        billing this code' — every element requirement citeable to the
        licensed AMA table instead of model memory. None when the code
        is not MDM-leveled or the grid is unavailable."""
        level = self.em_mdm_level(code)
        if not level:
            return None
        grid = self.mdm_grid(dos)
        if not grid or level not in (grid.get("levels") or {}):
            return None
        return {"code": _norm(code), "level": level,
                "selection_rule": grid["selection_rule"],
                "time_rule": grid["time_rule"],
                "source": grid["source"],
                "requirements": grid["levels"][level]}

    def em_family_prefix(self, code: str) -> str | None:
        """The E/M family key for an MDM-leveled E/M code: its AMA
        descriptor's prefix before the '<level> medical decision making'
        phrase. Two codes with the same prefix are level siblings of one
        family (99213/99214/99215). None when the code has no MDM phrase
        (99211) or the prefix is too short to identify a family."""
        row = self.conn.execute(
            "SELECT description FROM code_set WHERE code_system='CPT' AND code=? LIMIT 1",
            (_norm(code),),
        ).fetchone()
        if not row or not row["description"]:
            return None
        m = re.search(
            r"(straightforward|low|moderate|high)\s+(?:level\s+of\s+)?medical decision making",
            row["description"], re.IGNORECASE,
        )
        if not m:
            return None
        prefix = row["description"][:m.start()]
        return prefix if len(prefix) >= 20 else None

    def em_level_sibling(self, code: str, target_level: str) -> str | None:
        """The CPT code in the SAME E/M family as `code` whose own AMA
        descriptor states `target_level` medical decision making — found by
        descriptor structure, never a hardcoded code table.

        E/M descriptors within a family are identical up to the MDM-level
        phrase ('Office or other outpatient visit … established patient,
        which requires … examination and LOW level of medical decision
        making …'), so the family key is the descriptor prefix before that
        phrase and the sibling is the unique family member whose descriptor
        carries the target level. Returns None when the descriptor has no
        MDM phrase (99211), the prefix is ambiguous, or no unique sibling
        exists."""
        prefix = self.em_family_prefix(code)
        if prefix is None:
            return None
        level_re = re.compile(
            r"(straightforward|low|moderate|high)\s+(?:level\s+of\s+)?medical decision making",
            re.IGNORECASE,
        )
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        matches = []
        for r in self.conn.execute(
            "SELECT DISTINCT code, description FROM code_set "
            "WHERE code_system='CPT' AND description LIKE ? ESCAPE '\\'",
            (escaped + "%",),
        ):
            mm = level_re.search(r["description"] or "")
            if mm and mm.group(1).lower() == target_level.strip().lower():
                matches.append(r["code"])
        return matches[0] if len(set(matches)) == 1 else None

    def is_separate_procedure(self, code: str) -> bool:
        """True when the code's own CPT descriptor carries the '(separate
        procedure)' designation — NCCI Policy Manual Ch. 1 §J: such a code is
        not separately reportable when performed with another procedure in an
        anatomically related area through the same approach. Derived from the
        descriptor text itself, never a curated list."""
        row = self.conn.execute(
            "SELECT 1 FROM code_set WHERE code_system='CPT' AND code=? "
            "AND description LIKE '%(separate procedure)%' LIMIT 1", (_norm(code),)
        ).fetchone()
        return row is not None

    def is_unlisted_procedure(self, code: str) -> bool:
        """True when the CPT descriptor designates an unlisted code ('Unlisted
        procedure, …') — NCCI Policy Manual Ch. 1 §T: unlisted codes have no
        fee-schedule value, price manually, and require documentation."""
        row = self.conn.execute(
            "SELECT 1 FROM code_set WHERE code_system='CPT' AND code=? "
            "AND description LIKE 'Unlisted %' LIMIT 1", (_norm(code),)
        ).fetchone()
        return row is not None

    def pos_is_facility(self, pos_code: str) -> bool | None:
        """True/False from the CMS POS set's own facility designation
        ('F'/'N'), None when the POS code is unknown."""
        row = self.conn.execute(
            "SELECT facility FROM pos WHERE code=? LIMIT 1", (str(pos_code or "").strip(),)
        ).fetchone()
        if row is None:
            return None
        return str(row["facility"]).strip().upper() == "F"

    def bilat_surg(self, code: str, dos=None) -> str | None:
        """CMS bilateral-surgery indicator (0/1/2/3/9 — see global_periods.json's
        indicator_meanings.bilat_surg). '1' is the real, code-specific signal
        that a laterality modifier (RT/LT/50) is expected on this code — used
        instead of guessing from a CPT section/prefix."""
        row = self.conn.execute(
            "SELECT bilat_surg FROM global_period WHERE code=? LIMIT 1", (_norm(code),)
        ).fetchone()
        return row["bilat_surg"] if row else None

    def not_separately_billable_reason(self, code: str, dos=None) -> str | None:
        """None if the code's billing status doesn't rule out standalone
        billing; otherwise a human-readable reason, using only the subset of
        global_periods.json's own indicator_meanings that genuinely means
        "not separately payable anywhere" (B/N). Status 'I' (not valid for
        Medicare) is intentionally excluded — it's payer-conditional, not an
        absolute non-billability signal, so callers that know the claim's
        payer should check billing_status() == 'I' themselves rather than
        rely on this.

        Status 'X' (statutory exclusion) is also intentionally excluded —
        the source data is the PHYSICIAN Fee Schedule, so X only means "not
        a physician service under the PFS statute", NOT "not billable at
        all": lab codes (8xxxx, paid under the Clinical Lab Fee Schedule)
        and supply A-codes (paid under DMEPOS) dominate the real X rows
        (2,574 in the current data — 87070 bacterial culture, A6550 NPWT
        wound set, etc.), and all are legitimately payable under their own
        fee schedules. Treating X as a hard non-billability signal was
        observed suppressing real DMEPOS revenue. X is surfaced separately
        via pfs_exclusion_advisory() as a review-level signal.

        CPT Category II codes (4 digits + 'F' suffix) are the one universal
        exception to that payer-conditional rule: they're AMA performance-
        measure tracking codes with zero RVU value, carrying no payment under
        ANY payer by design — not a coverage decision like a normal code's
        status='I'. Confirmed structurally against real data: all 565
        Category II codes in global_periods.json carry status I/M, never A
        (active/normally billable) — a 100% pattern across the whole code
        category, not a per-code judgment call.
        """
        norm = _norm(code)
        if len(norm) == 5 and norm[:4].isdigit() and norm[4] == "F":
            return "CPT Category II code (performance-measure tracking, zero RVU by AMA design)"
        status = self.billing_status(code, dos)
        meanings = {
            "B": "bundled/not separately payable",
            "N": "noncovered",
            # CMS PFS RVU record layout: P = "Bundled/excluded codes — no
            # RVUs and no payment amounts; no separate payment is made"
            # (payment is packaged into the practice expense of the service
            # they are incident to). On a PROFESSIONAL claim this is as
            # absolute as B. Distinct from X (payable under another schedule
            # by design): a status-P supply consumed in the office is never
            # separately payable; a supply DISPENSED for home use is a
            # different benefit billed to the DME MAC on its own claim, not
            # a line on this one. Measured live: office-applied dressing
            # codes (all status P) flapped on/off across independent runs
            # of the same note — the LLM re-deciding, per run, a question
            # CMS's own status indicator answers deterministically.
            "P": "bundled/excluded — packaged into the practice expense of "
                 "the service it is incident to; if the supply was dispensed "
                 "for home use, bill the DME MAC on a separate DMEPOS claim",
        }
        return f"status '{status}' ({meanings[status]})" if status in meanings else None

    def coverage_policy_title(self, policy_id: str) -> str:
        row = self.conn.execute(
            "SELECT title FROM coverage_policy WHERE policy_id=? LIMIT 1", (policy_id,)
        ).fetchone()
        return (row["title"] or "") if row else ""

    def policies_titled(self, policy_ids: list[str], phrase: str) -> list[str]:
        """Subset of `policy_ids` whose real CMS policy title contains
        `phrase` (case-insensitive) — lets checks key off what kind of policy
        governs a code (e.g. 'routine foot care') using CMS's own catalog
        titles rather than a hardcoded CPT list."""
        phrase_l = phrase.lower()
        return [
            pid for pid in policy_ids
            if phrase_l in self.coverage_policy_title(pid).lower()
        ]

    def pfs_exclusion_advisory(self, code: str, dos=None) -> str | None:
        """Advisory (review, never auto-remove) for PFS status 'X': the code
        is statutorily excluded from the Physician Fee Schedule but typically
        payable under another fee schedule (CLFS for labs, DMEPOS for
        supplies) — the claim question is WHO bills it and under which
        schedule, not whether the service is real. See
        not_separately_billable_reason() for why X must not hard-suppress."""
        if self.billing_status(code, dos) == "X":
            return (
                "status 'X' (statutorily excluded from the Physician Fee Schedule — "
                "verify it is billed under the correct fee schedule (e.g. CLFS/DMEPOS) "
                "and by the entity that performed it)"
            )
        return None

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
        if not rows:
            # Never record provenance for an empty snapshot: a 0-row "ingest"
            # (e.g. a landing page fetched instead of the data file) would
            # register (source_id, effective_from) as done and turn every
            # future REAL ingest for that date into a skipped no-op.
            logger.warning(f"  refresh[{source_id}]: 0 rows — snapshot NOT recorded")
            return 0
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
        if table == "mue":
            self._mue_release_window_loaded = False
            self._mue_release_window = None
            self._mue_release_windows = ()
        elif table == "ncci_ptp":
            self._ncci_release_window_loaded = False
            self._ncci_release_window = None
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
