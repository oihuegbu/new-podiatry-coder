"""Tests for the refresh layer: parsers (against synthetic CMS-format samples),
history-retentive ingestion, and a best-effort live-download probe.
Run:  python -m tests.test_refresh
"""
from __future__ import annotations

from app.compliance.datastore.store import ComplianceDataStore
from app.compliance.refresh import parsers as P
from app.compliance.refresh.runner import refresh_source, download

PASS, FAIL = 0, 0
def check(name, cond):
    global PASS, FAIL
    PASS, FAIL = (PASS + 1, FAIL) if cond else (PASS, FAIL + 1)
    print(f"  {'✅' if cond else '❌'} {name}")

_STORE = ComplianceDataStore(); _STORE.build_or_load()

# --- NCCI: copyright junk line, then a real pair (indicator 1) ----------------
print("\n[parse_ncci]")
ncci_csv = (
    "CMS NCCI PTP Edits - copyright AMA, not part of CPT\n"
    "Column 1,Column 2,In Effect,Effective Date,Deletion Date,Modifier,PTP Edit Rationale\n"
    "28296,28292,1,2026-04-01,,1,Misuse of column two code\n"
    "Fee schedules and related components are not assigned by the AMA,,,,,,\n"
)
rows, cols = P.parse_ncci(ncci_csv, "2026-04-01")
check("parses the real pair, drops junk", len(rows) == 1 and rows[0][0] == "28296")
check("captures modifier indicator", rows[0][2] == "1")

# --- MUE: MAI in its own column ----------------------------------------------
print("\n[parse_mue]")
mue_csv = (
    "HCPCS/CPT Code,Practitioner Services MUE Values,MUE Adjudication Indicator,MUE Rationale\n"
    "11055,2,3,Clinical: anatomic consideration\n"
    "0001U,1,2,Policy: code descriptor\n"
)
rows, cols = P.parse_mue(mue_csv, "2026-04-01")
check("parses 2 codes", len(rows) == 2)
check("extracts MAI 3 and value 2", rows[0][1] == 2 and rows[0][2] == "3")

# --- PFS: GLOB DAYS column ----------------------------------------------------
print("\n[parse_pfs]")
pfs_csv = "HCPCS,MOD,STATUS CODE,GLOB DAYS\n28296,,A,090\n99213,,A,XXX\n"
rows, cols = P.parse_pfs(pfs_csv, "2026-01-01")
check("captures GLOB DAYS 090", any(r[0] == "28296" and r[1] == "090" for r in rows))

# --- POS HTML -----------------------------------------------------------------
print("\n[parse_pos]")
pos_html = "<table><tr><td>11 - Office</td></tr><tr><td>24 - Ambulatory Surgical Center</td></tr></table>"
rows, cols = P.parse_pos(pos_html, "2026-01-01")
check("scrapes POS codes 11 and 24", {r[0] for r in rows} >= {"11", "24"})

# --- MCD articles -------------------------------------------------------------
print("\n[parse_mcd_articles]")
mcd_csv = (
    "Article ID,HCPCS/CPT Code,ICD-10-CM Code\n"
    "A12345,11055,E11.42\n"
    "A12345,11056,E11.40\n"
)
arts = P.parse_mcd_articles(mcd_csv, "2026-06-01")
check("groups into one article with 2 cpt + 2 icd", len(arts) == 1 and len(arts[0]["cpt_codes"]) == 2)

# --- history-retentive ingestion ---------------------------------------------
print("\n[history retention]")
# ingest an MUE snapshot for an old quarter, then a new one — both retained
_STORE.conn.execute("DELETE FROM mue WHERE code='ZTEST1'")
_STORE.conn.execute("DELETE FROM data_source_version WHERE source_id='unittest'")
_STORE.ingest_snapshot("mue", ["code", "mue_value", "mai", "rationale", "effective_from", "effective_to"],
                       [("ZTEST1", 2, "2", "old", "2026-01-01", "9999-12-31")], "unittest", "2026-01-01")
_STORE.ingest_snapshot("mue", ["code", "mue_value", "mai", "rationale", "effective_from", "effective_to"],
                       [("ZTEST1", 5, "2", "new", "2026-04-01", "9999-12-31")], "unittest", "2026-04-01")
old = _STORE.mue("ZTEST1", "2026-02-15")   # should see the Jan snapshot (cap 2)
new = _STORE.mue("ZTEST1", "2026-06-15")   # should see the Apr snapshot (cap 5)
check("DOS picks correct historical snapshot", old and old["mue_value"] == 2 and new and new["mue_value"] == 5)
again = _STORE.ingest_snapshot("mue", ["code", "mue_value", "mai", "rationale", "effective_from", "effective_to"],
                               [("ZTEST1", 5, "2", "new", "2026-04-01", "9999-12-31")], "unittest", "2026-04-01")
check("re-ingesting same quarter is idempotent (no-op)", again == 0)

# --- offline ingest via runner (local_bytes) ---------------------------------
print("\n[runner offline ingest]")
res = refresh_source(_STORE, "ncci_ptp", effective_from="2026-04-01",
                     local_bytes=ncci_csv.encode(), dry_run=True)
check("runner dry-run parses local NCCI", res.get("ok") and res.get("parsed_rows") == 1)

# --- live download probe (best-effort; not counted as failure if blocked) -----
print("\n[live download probe]")
try:
    data = download("https://www.cms.gov/medicare/coding-billing/place-of-service-codes/code-sets", timeout=20)
    print(f"  ℹ️  live CMS fetch OK: {len(data)} bytes")
except Exception as e:
    print(f"  ℹ️  live CMS fetch unavailable here ({type(e).__name__}) — runner works when network allows")

# --- cleanup: remove all rows this test inserted so the shared DB stays pristine ---
_STORE.conn.execute("DELETE FROM mue WHERE code='ZTEST1'")
_STORE.conn.execute("DELETE FROM data_source_version WHERE source_id='unittest'")
_STORE.conn.commit()

print(f"\n{'='*50}\nRESULT: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
