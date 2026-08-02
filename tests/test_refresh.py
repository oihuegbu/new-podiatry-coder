"""Tests for the refresh layer: parsers (against synthetic CMS-format samples),
history-retentive ingestion, and a best-effort live-download probe.
Run:  python -m tests.test_refresh
"""
from __future__ import annotations

if __name__ != "__main__":
    import pytest
    pytest.skip("script harness; run with python tests/test_refresh.py",
                allow_module_level=True)

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

check("active MCD row uses version-effective date and open end",
      P._mcd_article_window({
          "status": "A", "article_eff_date": "2026-04-01 00:00:00",
      }) == ("2026-04-01", "9999-12-31"))
check("retired MCD row preserves its closed authoritative window",
      P._mcd_article_window({
          "status": "R", "article_eff_date": "2025-10-01 00:00:00",
          "article_end_date": "2026-03-05 00:00:00",
      }) == ("2025-10-01", "2026-03-05"))
check("proposed or undated MCD row has no policy authority",
      P._mcd_article_window({
          "status": "P", "article_eff_date": "2026-04-01 00:00:00",
      }) is None
      and P._mcd_article_window({"status": "A"}) is None)

# --- covered-ICD group grammar (roles + scope) --------------------------------
print("\n[group role grammar]")
# Form 1 — self-describing trailing label (CGS A57193): the role is the
# paragraph's own tail label, not keyword presence (both groups share body text)
_body = ("For treatment of mycotic nails, ICD-10 CM code B35.1 or ICD-10-CM "
         "L60.1-L60.5 respectively, must be reported as primary, with the "
         "diagnosis representing the patient's symptom reported as the "
         "secondary ICD-10-CM code. ")
role, scope = P.group_role_from_paragraph(_body + "Primary Diagnosis :")
check("tail label 'Primary Diagnosis:' → primary_eligible", role == "primary_eligible")
check("ICD mentions (B35.1, L60.1-L60.5) never leak into CPT scope", scope == [])
role, _ = P.group_role_from_paragraph(_body + "Secondary Diagnosis:")
check("tail label 'Secondary Diagnosis:' → required_secondary", role == "required_secondary")
# Form 2 — cross-reference (NGS A57759): the referring group holds the primary
# codes; the referred-to group holds the named (secondary) role
_xref = {2: "Refer to Group 3 for the secondary ICD-10-CM codes required for "
            "coverage for codes 11719, 11720, 11721 and G0127.",
         3: "Treatment of mycotic nails may be covered when the patient has a "
            "qualifying systemic condition."}
_r = P.resolve_group_roles(_xref)
check("xref: referring group → primary_eligible", _r[2][0] == "primary_eligible")
check("xref: referred group → required_secondary", _r[3][0] == "required_secondary")
check("xref: inline 'codes 11719, ... and G0127' → CPT scope",
      _r[2][1] == ["11719", "11720", "11721", "G0127"])
# Form 3 — conjunction (WPS A56232): both named groups required together
_conj = {2: "Codes 11720 and 11721 billed without a Q modifier require a code "
            "from group 2 (clinical evidence of mycosis of the nail) and a "
            "code from group 3 (pain or secondary infection).",
         3: "Codes 11720 and 11721 billed without a Q modifier require a code "
            "from group 2 (clinical evidence of mycosis of the nail) and a "
            "code from group 3 (pain or secondary infection)."}
_r = P.resolve_group_roles(_conj)
check("conjunction: first named group → primary_eligible", _r[2][0] == "primary_eligible")
check("conjunction: second named group → required_secondary", _r[3][0] == "required_secondary")
# Unreadable grammar stays standalone — the conservative pre-group behavior
_r = P.resolve_group_roles({1: "The ICD-10-CM codes below represent covered diagnoses."})
check("unreadable paragraph → unspecified (standalone)", _r[1][0] == "unspecified")
# Self-described labels always win over a cross-reference pointing elsewhere
_r = P.resolve_group_roles({
    1: _body + "Primary Diagnosis :",
    2: "Refer to Group 1 for the secondary ICD-10-CM codes required for coverage.",
})
check("self-described label wins over conflicting xref", _r[1][0] == "primary_eligible")

# --- coverage ingest: states column + related-LCD supersession ----------------
print("\n[coverage states + LCD supersession]")
for _pid in ("ATESTR1", "LTESTR1"):
    _STORE.conn.execute("DELETE FROM coverage_cpt WHERE policy_id=?", (_pid,))
    _STORE.conn.execute("DELETE FROM coverage_icd WHERE policy_id=?", (_pid,))
    _STORE.conn.execute("DELETE FROM coverage_group WHERE policy_id=?", (_pid,))
    _STORE.conn.execute("DELETE FROM coverage_policy WHERE policy_id=?", (_pid,))
    _STORE.conn.execute("DELETE FROM lcd_qualifying WHERE lcd_id=?", (_pid,))
_STORE.load_coverage_articles([
    # flat seed-era LCD entry (the pre-Cures shape: article codes flattened in)
    {"policy_id": "LTESTR1", "title": "Test LCD", "contractor": "",
     "cpt_codes": ["97810"], "covered_icd": ["I10", "R51.9"]},
    # its grammar-carrying billing article, naming the LCD as its parent
    {"policy_id": "ATESTR1", "title": "Billing and Coding: Test", "contractor": "",
     "states": ["FL", "GA"], "related_lcds": ["LTESTR1"],
     "cpt_codes": ["97810"], "covered_icd": ["I10", "R51.9"],
     "covered_icd_groups": [
         {"group": 1, "role": "primary_eligible", "cpt_scope": [],
          "paragraph": "test", "codes": ["I10"]},
         {"group": 2, "role": "required_secondary", "cpt_scope": [],
          "paragraph": "test", "codes": ["R51.9"]},
     ]},
])
check("article's states column round-trips through coverage_policy_states",
      _STORE.coverage_policy_states("ATESTR1") == {"FL", "GA"})
check("superseded LCD's flat covered list retired (dx gate moves to the article)",
      not _STORE.coverage_policy_has_dx_rules("LTESTR1"))
check("superseded LCD keeps its CPT rows (still governs the code)",
      "LTESTR1" in _STORE.coverage_policies_for_cpt("97810"))
check("article's group grammar ingested with roles",
      {g["role"] for g in _STORE.coverage_groups("ATESTR1")}
      == {"primary_eligible", "required_secondary"})
for _pid in ("ATESTR1", "LTESTR1"):
    _STORE.conn.execute("DELETE FROM coverage_cpt WHERE policy_id=?", (_pid,))
    _STORE.conn.execute("DELETE FROM coverage_icd WHERE policy_id=?", (_pid,))
    _STORE.conn.execute("DELETE FROM coverage_group WHERE policy_id=?", (_pid,))
    _STORE.conn.execute("DELETE FROM coverage_policy WHERE policy_id=?", (_pid,))
    _STORE.conn.execute("DELETE FROM lcd_qualifying WHERE lcd_id=?", (_pid,))
_STORE.conn.commit()

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
