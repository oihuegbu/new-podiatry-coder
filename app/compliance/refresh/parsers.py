"""Parsers for the real CMS source-file formats → canonical row dicts.

All parsers are header-NAME driven (not column-position) so they survive CMS
reordering columns between quarters. Each returns a list of dicts plus the
detected effective date. Garbage/copyright header rows are filtered.
"""
from __future__ import annotations

import csv
import io
import re
import zipfile

_CODE_RE = re.compile(r"^[A-Z0-9]{4}[A-Z0-9]$")


def _rows(text: str) -> list[dict]:
    """CSV text → list of dicts, tolerant of leading junk lines before the header."""
    lines = text.splitlines()
    # find the header row: first comma-bearing line with a strong header token.
    # (Strong tokens avoid matching copyright lines that mention "CPT"/"code".)
    strong = ("column 1", "column1", "adjudication", "mue value", "glob days",
              "icd-10", "icd10", "article id", "hcpcs/cpt", "procedure to procedure")
    start = 0
    for i, ln in enumerate(lines[:80]):
        low = ln.lower()
        if "," in ln and any(k in low for k in strong):
            start = i
            break
    reader = csv.DictReader(lines[start:])
    out = []
    for row in reader:
        clean = {}
        for k, v in row.items():
            if k is None:  # extra columns beyond the header (csv restkey)
                continue
            if isinstance(v, list):
                v = ",".join(x for x in v if x)
            clean[k.strip()] = (v or "").strip()
        out.append(clean)
    return out


def _get(row: dict, *names: str) -> str:
    """Fetch a value by any of several possible header names (case-insensitive)."""
    low = {k.lower(): v for k, v in row.items()}
    for n in names:
        for k, v in low.items():
            if n in k:
                return v
    return ""


def unzip_first(data: bytes, want_ext=(".csv", ".txt")) -> str:
    """Return the text of the first matching member of a ZIP (or the raw text)."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return data.decode("latin-1", errors="replace")
    for name in zf.namelist():
        if name.lower().endswith(want_ext):
            return zf.read(name).decode("latin-1", errors="replace")
    # fall back to the first member
    return zf.read(zf.namelist()[0]).decode("latin-1", errors="replace") if zf.namelist() else ""


# --------------------------------------------------------------------------- #
def parse_ncci(text: str, effective_date: str) -> tuple[list[tuple], list[str]]:
    """→ rows for ncci_ptp(col1, col2, modifier_indicator, effective_from, effective_to)."""
    rows = []
    for r in _rows(text):
        c1 = _get(r, "column 1", "column1").replace(".", "").strip().upper()
        c2 = _get(r, "column 2", "column2").replace(".", "").strip().upper()
        if not (_CODE_RE.match(c1) and _CODE_RE.match(c2)):
            continue
        mod = _get(r, "modifier").strip()
        mod = mod[0] if mod[:1] in ("0", "1", "9") else ""
        eff = _get(r, "effective") or effective_date
        end = _get(r, "deletion", "end")
        rows.append((c1, c2, mod, eff or effective_date, end or "9999-12-31"))
    cols = ["col1", "col2", "modifier_indicator", "effective_from", "effective_to"]
    return rows, cols


def parse_mue(text: str, effective_date: str) -> tuple[list[tuple], list[str]]:
    """→ rows for mue(code, mue_value, mai, rationale, effective_from, effective_to)."""
    rows = []
    for r in _rows(text):
        code = _get(r, "hcpcs", "cpt", "code").replace(".", "").strip().upper()
        if not _CODE_RE.match(code):
            continue
        try:
            val = int(float(_get(r, "mue value", "mue values", "value") or 0))
        except ValueError:
            val = 0
        mai = _get(r, "adjudication", "mai").strip()
        mai = mai[0] if mai[:1] in ("1", "2", "3") else ""
        rationale = _get(r, "rationale")
        rows.append((code, val, mai, rationale, effective_date, "9999-12-31"))
    cols = ["code", "mue_value", "mai", "rationale", "effective_from", "effective_to"]
    return rows, cols


def parse_pfs(text: str, effective_date: str) -> tuple[list[tuple], list[str]]:
    """→ rows for global_period(code, glob_days, effective_from, effective_to)."""
    rows = []
    for r in _rows(text):
        code = _get(r, "hcpcs", "cpt", "code").strip().upper()
        if not _CODE_RE.match(code):
            continue
        glob = _get(r, "glob days", "global").strip()
        if not glob:
            continue
        rows.append((code, glob, effective_date, "9999-12-31"))
    return rows, ["code", "glob_days", "effective_from", "effective_to"]


def parse_pos(html: str, effective_date: str) -> tuple[list[tuple], list[str]]:
    """Best-effort scrape of the CMS POS HTML table → pos(code, name, facility).

    Facility designation isn't in the HTML; defaults to 'N' and is corrected from
    the maintained reference file. Primarily refreshes names/new codes.
    """
    rows = []
    for m in re.finditer(r"\b(\d{2})\b\s*[-–:]\s*([A-Z][A-Za-z0-9 ,/&'\-]{3,60})", html):
        code, name = m.group(1), m.group(2).strip()
        rows.append((code, name, "N"))
    # dedup by code
    seen, uniq = set(), []
    for c, n, f in rows:
        if c not in seen:
            seen.add(c); uniq.append((c, n, f))
    return uniq, ["code", "name", "facility"]


def parse_mcd_articles(text: str, effective_date: str) -> list[dict]:
    """MCD Billing & Coding Article export → coverage article dicts.

    Expected columns: article_id/policy, HCPCS/CPT code, ICD-10 code (covered).
    Groups rows into {policy_id, cpt_codes, covered_icd} for
    `store.load_coverage_articles()`.
    """
    articles: dict[str, dict] = {}
    for r in _rows(text):
        pid = _get(r, "article", "policy", "lcd").strip()
        if not pid:
            continue
        art = articles.setdefault(pid, {"policy_id": pid, "cpt_codes": set(), "covered_icd": set()})
        cpt = _get(r, "hcpcs", "cpt", "procedure").replace(".", "").strip().upper()
        icd = _get(r, "icd-10", "icd10", "diagnosis").replace(".", "").strip().upper()
        if _CODE_RE.match(cpt):
            art["cpt_codes"].add(cpt)
        if icd:
            art["covered_icd"].add(icd)
    return [{"policy_id": p, "cpt_codes": sorted(a["cpt_codes"]),
             "covered_icd": sorted(a["covered_icd"])} for p, a in articles.items()]


PARSERS = {
    "parse_ncci": parse_ncci,
    "parse_mue": parse_mue,
    "parse_pfs": parse_pfs,
    "parse_pos": parse_pos,
    "parse_mcd_articles": parse_mcd_articles,
}
