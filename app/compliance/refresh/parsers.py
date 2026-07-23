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


def _sniff_delimiter(text: str) -> str:
    """Tab if the payload is tab-structured (the real CMS PTP files are
    tab-delimited .TXT, verified against the live 2026-Q3 file), else comma."""
    sample = "\n".join(text.splitlines()[:60])
    return "\t" if sample.count("\t") > sample.count(",") else ","


def _table(text: str) -> list[list[str]]:
    """Full-text CSV/TSV parse → list of row lists. csv.reader over the whole
    text (NOT splitlines + DictReader): the real MUE file's preamble is a
    QUOTED multi-line copyright field and its header cell is 'HCPCS/\\nCPT
    Code' with an embedded newline — line-based parsing mangles both."""
    return list(csv.reader(io.StringIO(text), delimiter=_sniff_delimiter(text)))


def _rows(text: str) -> list[dict]:
    """CMS table text → list of dicts, tolerant of junk lines before the
    header and of the PFS two-row header (group label row above the name
    row, e.g. 'GLOB'/'DAYS' → 'GLOB DAYS')."""
    table = _table(text)
    # find the header row: first row with a strong header token.
    # (Strong tokens avoid matching copyright lines that mention "CPT"/"code".)
    strong = ("column 1", "column1", "adjudication", "mue value", "glob days",
              "icd-10", "icd10", "article id", "hcpcs/cpt", "procedure to procedure",
              "hcpcs")
    hdr_i = None
    for i, row in enumerate(table[:80]):
        low = " ".join(c.lower().replace("\n", " ") for c in row)
        # >1 populated cell: banner rows like the PTP file's
        # 'Column1/Column2 Edits' line carry a strong token in a single cell
        # (padded with empty tab cells) and must not be taken as the header.
        if sum(1 for c in row if c.strip()) > 1 and any(k in low for k in strong):
            hdr_i = i
            break
    if hdr_i is None:
        return []
    header = [c.replace("\n", " ").strip() for c in table[hdr_i]]
    # Two-row header (PFS RVU files): the row ABOVE the name row carries group
    # labels for some columns ('STATUS'+'CODE', 'GLOB'+'DAYS', 'BILAT'+'SURG').
    # Merge group+name per column when the row above is clearly a header
    # fragment: several populated cells (single-cell banner/title rows are
    # not group labels), none of which is a procedure code.
    if hdr_i > 0:
        above = table[hdr_i - 1]
        if (sum(1 for c in above if c.strip()) >= 2
                and not _CODE_RE.match((above[0] or "").strip().upper())):
            merged = []
            for j, name in enumerate(header):
                grp = above[j].replace("\n", " ").strip() if j < len(above) else ""
                merged.append(f"{grp} {name}".strip() if grp else name)
            header = merged
    out = []
    for row in table[hdr_i + 1:]:
        clean = {}
        for j, name in enumerate(header):
            if not name:
                continue
            v = row[j] if j < len(row) else ""
            clean[name] = (v or "").strip()
        if clean:
            out.append(clean)
    return out


def _norm_date(val: str, default: str = "") -> str:
    """Normalize CMS date forms → YYYY-MM-DD. The live PTP file uses compact
    YYYYMMDD; storing that raw would corrupt the effective-range string
    comparisons the store does against ISO dates."""
    s = (val or "").strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    if re.match(r"^\d{8}$", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return default


def _get(row: dict, *names: str) -> str:
    """Fetch a value by any of several possible header names (case-insensitive)."""
    low = {k.lower(): v for k, v in row.items()}
    for n in names:
        for k, v in low.items():
            if n in k:
                return v
    return ""


def unzip_first(data: bytes, want_ext=(".csv", ".txt"), prefer: tuple[str, ...] = ()) -> str:
    """Return the text of the best-matching member of a ZIP (or the raw text).

    `prefer` is an ordered tuple of regexes tried against member names first —
    needed for archives that bundle several datasets (the PFS RVU zip ships
    GPCI/OPPSCAP/ANES/PPRRVU together; "first CSV" is the wrong file)."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return data.decode("latin-1", errors="replace")
    names = zf.namelist()
    for pat in prefer:
        rx = re.compile(pat, re.I)
        for name in names:
            if rx.search(name):
                return zf.read(name).decode("latin-1", errors="replace")
    for name in names:
        if name.lower().endswith(want_ext):
            return zf.read(name).decode("latin-1", errors="replace")
    # fall back to the first member
    return zf.read(names[0]).decode("latin-1", errors="replace") if names else ""


# --------------------------------------------------------------------------- #
def parse_ncci(text: str, effective_date: str) -> tuple[list[tuple], list[str]]:
    """→ rows for ncci_ptp(col1, col2, modifier_indicator, effective_from, effective_to).

    Live format (verified 2026-Q3 ccipra TXT): tab-delimited, compact
    YYYYMMDD dates, '*' = no deletion data."""
    rows = []
    for r in _rows(text):
        c1 = _get(r, "column 1", "column1").replace(".", "").strip().upper()
        c2 = _get(r, "column 2", "column2").replace(".", "").strip().upper()
        if not (_CODE_RE.match(c1) and _CODE_RE.match(c2)):
            continue
        mod = _get(r, "modifier").strip()
        mod = mod[0] if mod[:1] in ("0", "1", "9") else ""
        eff = _norm_date(_get(r, "effective"), effective_date)
        end = _norm_date(_get(r, "deletion", "end"), "")
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
    """→ rows for global_period(code, glob_days, billing_status, bilat_surg,
    pctc_ind, mult_proc, asst_surg, co_surg, team_surg,
    effective_from, effective_to).

    Live format (verified 2026 PPRRVU July release): two-row header where
    'GLOB DAYS' / 'STATUS CODE' / 'BILAT SURG' / 'PCTC IND' etc. each split
    across the group and name rows — merged by _rows(). EVERY column that
    global_period queries serve (billability status, bilateral, and the five
    PFS payment-policy indicators behind the 26/TC, 51, 80/81/82/AS, 62 and
    66 modifier checks) must be captured here: pfs_indicators() reads the
    most-recent effective row, so a refresh snapshot that omits any of them
    silently blanks that indicator for every code on the fee schedule from
    that quarter forward — observed live when pctc_ind came back NULL after
    the first quarterly refresh."""
    rows, seen = [], set()
    for r in _rows(text):
        code = _get(r, "hcpcs", "cpt", "code").strip().upper()
        if not _CODE_RE.match(code):
            continue
        # PPRRVU repeats codes per pricing modifier (26/TC/53) — the bare
        # (no-modifier) row is the code's own global/status entry.
        if _get(r, "mod").strip():
            continue
        if code in seen:
            continue
        seen.add(code)
        glob = _get(r, "glob days", "global").strip()
        if not glob:
            continue
        def ind(*names: str) -> str | None:
            return _get(r, *names).strip() or None
        rows.append((
            code, glob,
            ind("status"),
            ind("bilat"),
            ind("pctc"),
            ind("mult proc", "mult surg", "multiple proc"),
            ind("asst surg", "assistant"),
            # live 2026 PPRRVU spells it 'CO- SURG' (space after the hyphen)
            ind("co- surg", "co-surg", "co surg"),
            ind("team surg"),
            effective_date, "9999-12-31",
        ))
    return rows, ["code", "glob_days", "billing_status", "bilat_surg",
                  "pctc_ind", "mult_proc", "asst_surg", "co_surg", "team_surg",
                  "effective_from", "effective_to"]


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


def parse_mcd_export(raw: bytes) -> list[dict]:
    """CMS MCD bulk 'current_article.zip' export → coverage article dicts.

    Verified live format (downloads.cms.gov/medicare-coverage-database/
    downloads/exports/current_article.zip): an outer zip holding
    current_article_csv.zip, which holds a RELATIONAL set of CSVs — not one
    flat file.     Joined here on article_id:
      article.csv                → article_id, title
      article_x_hcpc_code.csv    → article_id, hcpc_code_id (the CPT/HCPCS)
      article_x_icd10_covered.csv→ article_id, icd10_code_id (covered dx)
      contractor.csv + article_x_contractor.csv → issuing MAC name(s), so
        refreshed articles keep the MAC-jurisdiction scoping the seed data
        has (an article without contractor info applies everywhere, which
        would silently widen every refreshed policy back to nationwide).
    Output shape matches store.load_coverage_articles() exactly (policy_id,
    title, contractor, cpt_codes, covered_icd), so titles keep feeding the
    policy-kind checks (e.g. routine foot care) on every weekly refresh."""
    outer = zipfile.ZipFile(io.BytesIO(raw))
    inner_name = next((n for n in outer.namelist() if n.lower().endswith("_csv.zip")), None)
    zf = zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) if inner_name else outer

    # article.csv embeds each policy's full HTML description in one field,
    # which exceeds csv's default 128 KB field cap.
    csv.field_size_limit(64 * 1024 * 1024)

    def rows_of(member: str) -> list[dict]:
        name = next((n for n in zf.namelist() if n.lower() == member), None)
        if name is None:
            return []
        text = zf.read(name).decode("latin-1", errors="replace")
        return list(csv.DictReader(io.StringIO(text)))

    # Only ACTIVE articles (status 'A') — the export also carries other
    # statuses (88 'P' rows in the live 2026-07 export); ingesting them
    # would gate claims on policies that aren't in force. Same filter the
    # seed-file ingest (_ingest_lcd) applies.
    titles = {r["article_id"]: (r.get("title") or "").strip()
              for r in rows_of("article.csv")
              if r.get("article_id") and (r.get("status") or "").strip() == "A"}

    # contractor.csv keys contractor_id → business name; article_x_contractor
    # links articles to their issuing MAC(s). Header names vary slightly
    # across export revisions, so match by fragment. Missing members leave
    # contractor blank — the jurisdiction check then treats the policy as
    # applicable everywhere (conservative) rather than dropping it.
    type_lookup = {}
    for r in rows_of("contractor_type_lookup.csv"):
        tid = (r.get("contractor_type_id") or "").strip()
        desc = next((v for k, v in r.items()
                     if k and "descr" in k.lower() and v), "")
        if tid and desc:
            type_lookup[tid] = desc.strip()
    contractor_names: dict[str, str] = {}
    for r in rows_of("contractor.csv"):
        cid = (r.get("contractor_id") or "").strip()
        name = next((v for k, v in r.items()
                     if k and ("bus_name" in k.lower() or "name" in k.lower()) and v), "")
        if not (cid and name):
            continue
        # append the contractor TYPE ("DME MAC" vs "MAC - Part B") when the
        # export provides it — the jurisdiction resolver keys DME policies to
        # the DME state map, which differs from the same contractor's A/B area
        ctype = type_lookup.get((r.get("contractor_type_id") or "").strip(), "")
        contractor_names[cid] = f"{name.strip()} ({ctype})" if ctype else name.strip()
    article_contractors: dict[str, set[str]] = {}
    for r in rows_of("article_x_contractor.csv"):
        pid, cid = (r.get("article_id") or "").strip(), (r.get("contractor_id") or "").strip()
        if pid and cid in contractor_names:
            article_contractors.setdefault(pid, set()).add(contractor_names[cid])

    articles: dict[str, dict] = {}

    def art(pid: str) -> dict:
        return articles.setdefault(pid, {
            "policy_id": f"A{pid}",  # MCD exports bare numeric ids; the seed
            "title": titles.get(pid, ""),  # data and agents use the A-prefixed form
            "contractor": " ".join(sorted(article_contractors.get(pid, set()))),
            "cpt_codes": set(), "covered_icd": set(), "noncovered_icd": set(),
        })

    for r in rows_of("article_x_hcpc_code.csv"):
        pid, code = r.get("article_id", ""), (r.get("hcpc_code_id") or "").strip().upper()
        if pid in titles and _CODE_RE.match(code):
            art(pid)["cpt_codes"].add(code)
    for r in rows_of("article_x_icd10_covered.csv"):
        pid, dx = r.get("article_id", ""), (r.get("icd10_code_id") or "").replace(".", "").strip().upper()
        if pid in titles and dx:
            art(pid)["covered_icd"].add(dx)
    # Group-N mirror list — diagnoses the policy explicitly says do NOT
    # support medical necessity (45k rows in the live export, previously
    # dropped on the floor)
    for r in rows_of("article_x_icd10_noncovered.csv"):
        pid, dx = r.get("article_id", ""), (r.get("icd10_code_id") or "").replace(".", "").strip().upper()
        if pid in titles and dx:
            art(pid)["noncovered_icd"].add(dx)

    return [{"policy_id": a["policy_id"], "title": a["title"],
             "contractor": a["contractor"],
             "cpt_codes": sorted(a["cpt_codes"]),
             "covered_icd": sorted(a["covered_icd"]),
             "noncovered_icd": sorted(a["noncovered_icd"])}
            for a in articles.values()
            if a["cpt_codes"] or a["covered_icd"] or a["noncovered_icd"]]


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
