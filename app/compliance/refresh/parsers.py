"""Parsers for the real CMS source-file formats → canonical row dicts.

All parsers are header-NAME driven (not column-position) so they survive CMS
reordering columns between quarters. Each returns a list of dicts plus the
detected effective date. Garbage/copyright header rows are filtered.
"""
from __future__ import annotations

import csv
import html
import io
import re
import zipfile

_CODE_RE = re.compile(r"^[A-Z0-9]{4}[A-Z0-9]$")

# ---------------------------------------------------------------- coverage
# Group-role grammar for MCD covered-ICD groups. The export publishes each
# article's covered diagnoses in numbered Groups, and each Group carries its
# own paragraph stating how its codes participate in coverage — e.g. A57193
# Group 2 ends "... must be reported as primary ... Primary Diagnosis :" and
# Group 3 ends "... Secondary Diagnosis:". Flattening the groups (the old
# behavior) discarded exactly the grammar that makes a claim-COMPOSITION
# check expressible: "B35.1 is covered" was truthfully answered on a claim
# missing the required symptom secondary that the same article says must
# accompany it (the enforcement is automatic denial — e.g. CMS A56640:
# "if a covered secondary diagnosis is not on the claim, the edit will
# automatically deny the service as not medically necessary").
_TAG_RE = re.compile(r"<[^>]+>")
# The role is declared by the paragraph's own trailing label ("Primary
# Diagnosis :", "Secondary Diagnosis Codes:"), not by keyword presence —
# A57193's Groups 2 and 3 share the SAME body text and differ only in that
# label, so a contains-check would misread one of them.
_ROLE_TAIL_RE = re.compile(
    r"\b(primary|secondary)\s+diagnos\w*(?:\s+codes?)?\s*:?\s*$", re.I)
# Cross-reference role grammar: MACs also declare composition by pointing
# one group at another — e.g. NGS A57759 Group 2 (the primary mycotic
# codes): "Refer to Group 3 for the secondary ICD-10-CM codes required for
# coverage for codes 11719, 11720, 11721 and G0127." The group DOING the
# referring holds the other role of the pair; the group REFERRED TO holds
# the named one. Resolved per article in resolve_group_roles().
_ROLE_XREF_RE = re.compile(
    r"\brefer\s+to\s+group\s+(\d+)\s+for\s+the\s+(primary|secondary)\b"
    r"[^.]{0,80}?\brequired\b", re.I)
# Conjunction grammar — a third observed spelling (WPS A56232): "Codes
# 11720 and 11721 billed without a Q modifier require a code from group 2
# (clinical evidence of mycosis of the nail) and a code from group 3 (pain
# or secondary infection)." Both named groups are required together; the
# first named (the condition) maps to primary_eligible and the second (the
# symptom) to required_secondary, which makes coverage evaluate as full
# only when both are on the claim — the conjunction's exact semantics.
_ROLE_CONJ_RE = re.compile(
    r"\brequire[sd]?\s+a\s+code\s+from\s+group\s+(\d+)\b[^.]{0,120}?"
    r"\band\s+a\s+code\s+from\s+group\s+(\d+)\b", re.I)
# Groups scope themselves to specific procedure codes, in two observed
# spellings: a "For Codes: 11055, 11056, 11057" preamble (CGS A57193) or an
# inline "codes 11719, 11720, 11721 and G0127" mention (NGS A57759). Both
# reduce to: CPT/HCPCS-shaped tokens following the word "code(s)". The token
# shape (5 digits, or letter + 4 digits) cannot match ICD-10-CM codes as
# MCD paragraphs write them (letter + 2 digits + dot, e.g. B35.1, L60.2),
# so diagnosis mentions never leak into a procedure scope.
_SCOPE_LIST_RE = re.compile(
    r"\bcodes?\s*:?\s*\(?\s*"
    r"((?:(?:[A-Z]\d{4}|\d{5})\s*(?:,\s*|and\s+|\s+)?)+)", re.I)
_SCOPE_CODE_RE = re.compile(r"\b(?:[A-Z]\d{4}|\d{5})\b")


def paragraph_text(raw: str) -> str:
    """MCD group paragraphs arrive double-HTML-encoded ('&amp;lt;p&amp;gt;'
    → '&lt;p&gt;' → '<p>') with '&sol;' slashes; unescape twice, drop tags,
    collapse whitespace."""
    txt = html.unescape(html.unescape(raw or "")).replace("&sol;", "/")
    return " ".join(_TAG_RE.sub(" ", txt).split())


def group_role_from_paragraph(raw_paragraph: str) -> tuple[str, list[str]]:
    """(role, cpt_scope) parsed from a covered-ICD group's own paragraph.

    role: 'primary_eligible'  — codes establish coverage only when a
              required-secondary code accompanies them on the claim;
          'required_secondary' — codes support coverage but cannot
              establish it alone;
          'unspecified'        — standalone (a code alone establishes
              coverage), the conservative default for any paragraph whose
              grammar this parser cannot read — which reproduces the flat
              pre-group behavior exactly, never a stricter gate.
    cpt_scope: procedure codes the group is scoped to ([] = all governed
    codes). Deterministic text grammar over the authoritative paragraph —
    never a hardcoded policy or code list. Reads only the SELF-describing
    grammar; cross-group references are resolved article-wide by
    resolve_group_roles()."""
    txt = paragraph_text(raw_paragraph)
    scope = sorted({tok for m in _SCOPE_LIST_RE.finditer(txt)
                    for tok in _SCOPE_CODE_RE.findall(m.group(1).upper())})
    tail = _ROLE_TAIL_RE.search(txt)
    if tail:
        return ("primary_eligible" if tail.group(1).lower() == "primary"
                else "required_secondary"), scope
    return "unspecified", scope


def resolve_group_roles(paragraphs: dict[int, str]) -> dict[int, tuple[str, list[str]]]:
    """Article-wide role resolution over {group_number: raw_paragraph}.

    Pass 1: each group's self-describing grammar (group_role_from_paragraph).
    Pass 2: cross-references — a group whose paragraph says "Refer to
    Group N for the secondary ... codes required ..." holds the PRIMARY
    codes itself and names group N as the required secondary (and
    symmetrically if it refers for the primary codes). Self-described
    labels always win over a cross-reference; unreadable grammar stays
    'unspecified' (standalone — the conservative, pre-group behavior)."""
    resolved = {gid: group_role_from_paragraph(p)
                for gid, p in paragraphs.items()}
    self_described = {gid for gid, (role, _) in resolved.items()
                      if role != "unspecified"}

    def assign(gid: int, role: str) -> None:
        if gid in paragraphs and gid not in self_described:
            resolved[gid] = (role, resolved[gid][1])

    for gid, raw in paragraphs.items():
        txt = paragraph_text(raw)
        m = _ROLE_XREF_RE.search(txt)
        if m:
            target, named_role = int(m.group(1)), m.group(2).lower()
            if named_role == "secondary":
                assign(target, "required_secondary")
                assign(gid, "primary_eligible")
            else:
                assign(target, "primary_eligible")
                assign(gid, "required_secondary")
            continue
        m = _ROLE_CONJ_RE.search(txt)
        if m:
            assign(int(m.group(1)), "primary_eligible")
            assign(int(m.group(2)), "required_secondary")
    return resolved


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
def parse_hcpcs_fixed_width(text: str, *, source_file: str,
                            source_url: str) -> list[dict]:
    """Official CMS alpha-numeric contractor record -> source JSON records.

    Field positions come from ``HCPC20YY_recordlayout.txt`` shipped in the
    same quarterly CMS archive.  Record IDs 3/4 are procedure first/
    continuation rows and 7/8 are modifier first/continuation rows.  Keeping
    modifiers in the source file preserves the full CMS release even though
    ``ComplianceDataStore`` deliberately ingests only five-character Level II
    service codes into ``code_set``.

    The parser is intentionally structural and contains no medical code
    values, families, or prefixes.  A continuation without a matching first
    row, a duplicate first row, or an invalid first-row identity makes the
    entire refresh invalid rather than silently producing a partial code set.
    """
    records: list[dict] = []
    current: dict | None = None
    seen: set[tuple[str, str]] = set()

    def field(line: str, begin: int, end: int) -> str:
        # CMS layout positions are one-based and inclusive.
        return line[begin - 1:end].strip()

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        current["long_description"] = " ".join(current.pop("_long_parts")).strip()
        records.append(current)
        current = None

    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if len(line) < 11:
            raise ValueError(
                f"HCPCS contractor row {line_number} is shorter than its record header")
        record_id = field(line, 11, 11)
        if record_id in {"3", "7"}:
            if len(line) < 293:
                raise ValueError(
                    f"HCPCS contractor row {line_number} is missing required detail fields")
            finish()
            kind = "procedure" if record_id == "3" else "modifier"
            code = field(line, 1, 5) if kind == "procedure" else field(line, 4, 5)
            expected_length = 5 if kind == "procedure" else 2
            if len(code) != expected_length or not code.isalnum():
                raise ValueError(
                    f"HCPCS contractor row {line_number} has an invalid {kind} identity")
            identity = (kind, code.upper())
            if identity in seen:
                raise ValueError(
                    f"HCPCS contractor row {line_number} duplicates {kind} {code}")
            seen.add(identity)
            current = {
                "code": code.upper(),
                "short_description": field(line, 92, 119),
                "_long_parts": [field(line, 12, 91)],
                "effective_from": _norm_date(field(line, 277, 284), "") or None,
                "effective_to": _norm_date(field(line, 285, 292), "") or None,
                "modifiers": [],
                "coverage_code": field(line, 230, 230) or None,
                "betos": field(line, 257, 259) or None,
                "action_code": field(line, 293, 293) or None,
                "add_date": _norm_date(field(line, 269, 276), "") or None,
                "metadata": {
                    "source_file": source_file,
                    "source_url": source_url,
                    "record_type": kind,
                },
            }
            if not all((current["short_description"], current["_long_parts"][0],
                        current["effective_from"], current["add_date"],
                        current["action_code"])):
                raise ValueError(
                    f"HCPCS contractor row {line_number} has incomplete required fields")
        elif record_id in {"4", "8"}:
            expected_kind = "procedure" if record_id == "4" else "modifier"
            if current is None or current["metadata"]["record_type"] != expected_kind:
                raise ValueError(
                    f"HCPCS contractor row {line_number} is an orphaned continuation")
            continuation_code = (
                field(line, 1, 5) if expected_kind == "procedure"
                else field(line, 4, 5))
            if continuation_code.upper() != current["code"]:
                raise ValueError(
                    f"HCPCS contractor row {line_number} changes identity mid-description")
            current["_long_parts"].append(field(line, 12, 91))
        else:
            raise ValueError(
                f"HCPCS contractor row {line_number} has unknown record id {record_id!r}")
    finish()
    if not records:
        raise ValueError("HCPCS contractor file contains no records")
    return records


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


def parse_pos(markup: str, effective_date: str) -> tuple[list[tuple], list[str]]:
    """CMS POS HTML table -> ``pos(code, name, facility)`` candidates.

    The current CMS web table publishes code/name/description but not the
    Medicare PFS facility designation. ``facility`` is therefore deliberately
    ``None``: the runner may preserve an installed authoritative designation,
    but must never invent non-facility status for a new or changed code.
    """
    rows = []
    for table_row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", markup,
                                flags=re.I | re.S):
        cells = [paragraph_text(cell) for cell in re.findall(
            r"<t[dh]\b[^>]*>(.*?)</t[dh]>", table_row,
            flags=re.I | re.S)]
        if len(cells) >= 2 and re.fullmatch(r"\d{2}", cells[0]):
            rows.append((cells[0], cells[1], None))
    # Small synthetic/legacy pages sometimes render code and name in one
    # cell. Keep a bounded fallback without interpreting code ranges.
    if not rows:
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", markup,
                           flags=re.I | re.S) or [markup]
        for cell in cells:
            match = re.match(
                r"^\s*(\d{2})\s*[-–:]\s*"
                r"([A-Z][A-Za-z0-9 ,/&'\-]{3,60}?)\s*$",
                paragraph_text(cell))
            if match:
                rows.append((match.group(1), match.group(2).strip(), None))
    # dedup by code
    seen, uniq = set(), []
    for c, n, facility in rows:
        if c not in seen:
            seen.add(c); uniq.append((c, n, facility))
    return uniq, ["code", "name", "facility"]


def _mcd_article_window(row: dict) -> tuple[str, str] | None:
    """Effective window for one latest-version MCD Article export row.

    The official CSV uses article_eff_date/article_end_date, not the display
    labels previously searched by the parser. Active and retired versions are
    authoritative for their stated window; proposed rows are not policy.
    """
    status = str(row.get("status") or "").strip().upper()
    if status not in {"A", "R"}:
        return None
    start = _norm_date(str(row.get("article_eff_date") or "")[:10], "")
    if not start:
        return None
    if status == "R":
        end_raw = (row.get("article_end_date")
                   or row.get("article_rev_end_date") or "")
        end = _norm_date(str(end_raw)[:10], "")
        if not end:
            return None
    else:
        end = "9999-12-31"
    return start, end


def parse_mcd_export(raw: bytes) -> list[dict]:
    """CMS MCD bulk 'all_article.zip' export → coverage article dicts.

    Verified live format (downloads.cms.gov/medicare-coverage-database/
    downloads/exports/all_article.zip): an outer zip holding an inner
    relational CSV archive — not one
    flat file.     Joined here on article_id:
      article.csv                → article_id, title
      article_x_hcpc_code.csv    → article_id, hcpc_code_id (the CPT/HCPCS)
      article_x_icd10_covered.csv→ article_id, icd10_code_id (covered dx)
      contractor.csv + article_x_contractor.csv → issuing MAC name(s), so
        refreshed articles keep the MAC-jurisdiction scoping the seed data
        has (an article without contractor info applies everywhere, which
        would silently widen every refreshed policy back to nationwide).
      contractor_jurisdiction.csv + state_lookup.csv → the authoritative
        per-article STATE set (see the comment at the join below).
      article_related_documents.csv → parent LCD ids, so the store can
        retire a superseded LCD's flat seed-era code list.
    Output shape matches store.load_coverage_articles() exactly (policy_id,
    title, contractor, states, related_lcds, cpt_codes, covered_icd), so
    titles keep feeding the policy-kind checks (e.g. routine foot care) on
    every weekly refresh."""
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

    # The all-Article dataset exposes the latest active and retired version of
    # each policy. Proposed rows remain excluded. Older superseded versions
    # are not present in this CMS download, so DOS windows outside these
    # exact rows remain UNKNOWN rather than borrowing current policy content.
    article_rows = [r for r in rows_of("article.csv")
                    if r.get("article_id") and _mcd_article_window(r)]
    titles = {r["article_id"]: (r.get("title") or "").strip()
              for r in article_rows}
    effective_windows = {
        r["article_id"]: _mcd_article_window(r) for r in article_rows
    }

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
    # Authoritative MAC service areas: the export's OWN relational tables
    # (contractor_jurisdiction.csv ⋈ state_lookup.csv, keyed by contractor
    # id+type+version) say exactly which states each contractor entry
    # adjudicates. This replaces guessing states from the contractor's
    # business name via mac_jurisdictions.json — which returned "unknown"
    # for any name variant it didn't recognize, and an unknown service area
    # conservatively applies the policy EVERYWHERE (observed live: NGS
    # article A57759 (NY/CT/IL/MN/WI...) gating Ohio claims because
    # "National Government Services, Inc." didn't resolve).
    state_abbrev = {
        (r.get("state_id") or "").strip(): (r.get("state_abbrev") or "").strip().upper()
        for r in rows_of("state_lookup.csv")
    }
    contractor_states: dict[tuple[str, str, str], set[str]] = {}
    for r in rows_of("contractor_jurisdiction.csv"):
        if (r.get("term_date") or "").strip():
            continue  # terminated jurisdiction entry
        key = ((r.get("contractor_id") or "").strip(),
               (r.get("contractor_type_id") or "").strip(),
               (r.get("contractor_version") or "").strip())
        abbr = state_abbrev.get((r.get("state_id") or "").strip())
        if abbr:
            contractor_states.setdefault(key, set()).add(abbr)

    article_contractors: dict[str, set[str]] = {}
    article_states: dict[str, set[str]] = {}
    for r in rows_of("article_x_contractor.csv"):
        pid, cid = (r.get("article_id") or "").strip(), (r.get("contractor_id") or "").strip()
        if not pid:
            continue
        if cid in contractor_names:
            article_contractors.setdefault(pid, set()).add(contractor_names[cid])
        key = (cid, (r.get("contractor_type_id") or "").strip(),
               (r.get("contractor_version") or "").strip())
        article_states.setdefault(pid, set()).update(
            contractor_states.get(key, set()))

    # Article → parent LCD (article_related_documents.csv). Post-Cures-Act,
    # LCDs no longer publish code lists — the related Billing & Coding
    # article carries them (with group grammar). The store uses this link
    # to retire a superseded LCD's flat seed-era code list so it cannot
    # satisfy coverage past its own article's composition rules.
    article_related_lcds: dict[str, set[str]] = {}
    for r in rows_of("article_related_documents.csv"):
        pid = (r.get("article_id") or "").strip()
        lcd = (r.get("r_lcd_id") or "").strip()
        if pid in titles and lcd:
            article_related_lcds.setdefault(pid, set()).add(f"L{lcd}")

    articles: dict[str, dict] = {}

    def art(pid: str) -> dict:
        return articles.setdefault(pid, {
            "policy_id": f"A{pid}",  # MCD exports bare numeric ids; the seed
            "title": titles.get(pid, ""),  # data and agents use the A-prefixed form
            "contractor": " ".join(sorted(article_contractors.get(pid, set()))),
            "states": sorted(article_states.get(pid, set())),
            "related_lcds": sorted(article_related_lcds.get(pid, set())),
            "effective_from": (effective_windows.get(pid) or ("", ""))[0],
            "effective_to": (effective_windows.get(pid) or ("", ""))[1],
            "temporal_authority": bool(effective_windows.get(pid)),
            "cpt_codes": set(), "covered_icd": set(), "noncovered_icd": set(),
        })

    for r in rows_of("article_x_hcpc_code.csv"):
        pid, code = r.get("article_id", ""), (r.get("hcpc_code_id") or "").strip().upper()
        if pid in titles and _CODE_RE.match(code):
            art(pid)["cpt_codes"].add(code)

    # Covered-ICD GROUP metadata (article_x_icd10_covered_group.csv): each
    # group's paragraph declares its role in claim composition (primary-
    # eligible / required-secondary / standalone) and any procedure-code
    # scope — grammar the flat covered_icd set cannot carry. Roles resolve
    # ARTICLE-WIDE (resolve_group_roles): self-describing labels first,
    # then cross-references between groups ("Refer to Group 3 for the
    # secondary ... required"). Keyed (article_id, group_number).
    group_paragraphs: dict[str, dict[int, str]] = {}
    for r in rows_of("article_x_icd10_covered_group.csv"):
        pid = (r.get("article_id") or "").strip()
        gid = (r.get("icd10_covered_group") or "").strip()
        if pid in titles and gid.isdigit():
            group_paragraphs.setdefault(pid, {})[int(gid)] = r.get("paragraph") or ""
    group_meta: dict[tuple[str, str], dict] = {}
    for pid, paragraphs in group_paragraphs.items():
        for gid, (role, scope) in resolve_group_roles(paragraphs).items():
            group_meta[(pid, str(gid))] = {
                "group": gid, "role": role, "cpt_scope": scope,
                # provenance: the paragraph is the authority for the role
                "paragraph": paragraph_text(paragraphs[gid])[:400],
            }

    group_codes: dict[tuple[str, str], set] = {}
    for r in rows_of("article_x_icd10_covered.csv"):
        pid, dx = r.get("article_id", ""), (r.get("icd10_code_id") or "").replace(".", "").strip().upper()
        if pid in titles and dx:
            art(pid)["covered_icd"].add(dx)
            gid = (r.get("icd10_covered_group") or "").strip()
            if gid.isdigit():
                group_codes.setdefault((pid, gid), set()).add(dx)
    for (pid, gid), codes in sorted(group_codes.items()):
        meta = group_meta.get((pid, gid)) or {
            "group": int(gid), "role": "unspecified", "cpt_scope": [],
            "paragraph": "",
        }
        art(pid).setdefault("covered_icd_groups", []).append(
            dict(meta, codes=sorted(codes)))
    # Group-N mirror list — diagnoses the policy explicitly says do NOT
    # support medical necessity (45k rows in the live export, previously
    # dropped on the floor)
    for r in rows_of("article_x_icd10_noncovered.csv"):
        pid, dx = r.get("article_id", ""), (r.get("icd10_code_id") or "").replace(".", "").strip().upper()
        if pid in titles and dx:
            art(pid)["noncovered_icd"].add(dx)

    return [{"policy_id": a["policy_id"], "title": a["title"],
             "contractor": a["contractor"],
             "states": a["states"],
             "related_lcds": a["related_lcds"],
             "effective_from": a.get("effective_from", ""),
             "effective_to": a.get("effective_to", "9999-12-31"),
             "temporal_authority": bool(a.get("temporal_authority")),
             "cpt_codes": sorted(a["cpt_codes"]),
             "covered_icd": sorted(a["covered_icd"]),
             "noncovered_icd": sorted(a["noncovered_icd"]),
             "covered_icd_groups": sorted(a.get("covered_icd_groups", []),
                                          key=lambda g: g["group"])}
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
        art = articles.setdefault(pid, {
            "policy_id": pid, "cpt_codes": set(), "covered_icd": set(),
            "effective_from": effective_date, "effective_to": "9999-12-31",
            "temporal_authority": bool(effective_date),
        })
        cpt = _get(r, "hcpcs", "cpt", "procedure").replace(".", "").strip().upper()
        icd = _get(r, "icd-10", "icd10", "diagnosis").replace(".", "").strip().upper()
        if _CODE_RE.match(cpt):
            art["cpt_codes"].add(cpt)
        if icd:
            art["covered_icd"].add(icd)
    return [{"policy_id": p, "cpt_codes": sorted(a["cpt_codes"]),
             "covered_icd": sorted(a["covered_icd"]),
             "effective_from": a["effective_from"],
             "effective_to": a["effective_to"],
             "temporal_authority": a["temporal_authority"]}
            for p, a in articles.items()]


PARSERS = {
    "parse_ncci": parse_ncci,
    "parse_mue": parse_mue,
    "parse_pfs": parse_pfs,
    "parse_pos": parse_pos,
    "parse_mcd_articles": parse_mcd_articles,
}
