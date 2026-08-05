#!/usr/bin/env python3
"""Build data/codes/hcpcs_codes.json (HCPCS Level II) reproducibly from the current CMS
quarterly Alpha-Numeric file — same package pattern as the NCCI builders.

CMS publishes '{quarter}-{year}-alpha-numeric-hcpcs-file.zip' containing the ANWEB
workbook (…_ANWEB_*.xlsx). We parse the XLSX with the STDLIB (zipfile + ElementTree —
no openpyxl dependency), mapping columns by HEADER name (robust to column reordering):
  HCPC->code, LONG/SHORT DESCRIPTION, COV->coverage_code, BETOS, ACTION CD->action_code,
  ADD DT->add_date, ACT EFF DT->effective_from, TERM DT->effective_to.
store._ingest_hcpcs / code_reference read exactly these. No hardcoded codes; fail-closed.

Usage:
  python tools/build_hcpcs.py                    # current quarter from CMS
  python tools/build_hcpcs.py --quarter july --year 2026
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.core.config import HCPCS_FILE

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_QUARTERS = ["january", "april", "july", "october"]   # HCPCS releases


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "coder-refresh/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def _iso(v: str) -> str | None:
    v = (v or "").strip()
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", v)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _read_xlsx_rows(blob: bytes):
    """Yield each sheet row as {COLUMN_LETTER: value}, resolving shared strings. Stdlib."""
    xz = zipfile.ZipFile(io.BytesIO(blob))
    shared = []
    if "xl/sharedStrings.xml" in xz.namelist():
        for si in ET.fromstring(xz.read("xl/sharedStrings.xml")).findall(f"{_NS}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{_NS}t")))
    root = ET.fromstring(xz.read("xl/worksheets/sheet1.xml"))
    for r in root.find(f"{_NS}sheetData").findall(f"{_NS}row"):
        cells = {}
        for c in r.findall(f"{_NS}c"):
            v = c.find(f"{_NS}v")
            if v is None or v.text is None:
                continue
            val = shared[int(v.text)] if c.get("t") == "s" else v.text
            cells[re.match(r"[A-Z]+", c.get("r")).group()] = val
        yield cells


def _resolve_zip(quarter: str | None, year: int | None) -> tuple[bytes, str]:
    today = date.today()
    year = year or today.year
    order = ([quarter] if quarter else
             [_QUARTERS[min(3, (today.month - 1) // 3)]] + list(reversed(_QUARTERS)))
    tried = []
    for q in dict.fromkeys(order):
        url = f"https://www.cms.gov/files/zip/{q}-{year}-alpha-numeric-hcpcs-file.zip"
        try:
            return _download(url), f"{q}-{year}"
        except Exception as e:
            tried.append(f"{q}:{str(e)[:30]}")
    raise SystemExit(f"ABORT: no HCPCS zip resolved ({tried}). Pass --quarter/--year.")


def build(args) -> int:
    blob, tag = _resolve_zip(args.quarter, args.year)
    z = zipfile.ZipFile(io.BytesIO(blob))
    xlsx = next((n for n in z.namelist() if "ANWEB" in n and n.endswith(".xlsx")
                 and "Transaction" not in n and "Correction" not in n), None)
    if not xlsx:
        raise SystemExit(f"ABORT: no ANWEB .xlsx in the {tag} HCPCS zip: {z.namelist()[:6]}")

    rows = _read_xlsx_rows(z.read(xlsx))
    header = next(rows)                              # row 1 = column headers
    col = {str(v).strip().upper(): k for k, v in header.items()}

    def g(row, name):
        return str(row.get(col.get(name, ""), "") or "").strip()

    by_code: dict[str, dict] = {}
    for row in rows:
        code = g(row, "HCPC").upper()
        if not code:
            continue
        term = _iso(g(row, "TERM DT"))
        rec = {
            "code": code,
            "short_description": g(row, "SHORT DESCRIPTION"),
            "long_description": g(row, "LONG DESCRIPTION"),
            "effective_from": _iso(g(row, "ACT EFF DT")) or _iso(g(row, "ADD DT")),
            "effective_to": term,
            "modifiers": [],
            "coverage_code": g(row, "COV") or None,
            "betos": g(row, "BETOS") or None,
            "action_code": g(row, "ACTION CD") or None,
            "add_date": _iso(g(row, "ADD DT")),
            "metadata": {"source_file": xlsx, "source_zip": f"{tag}-alpha-numeric-hcpcs-file"},
        }
        by_code[code] = rec                          # one row per code (last wins = latest action)

    out = list(by_code.values())
    if not out:
        raise SystemExit("ABORT: 0 HCPCS codes parsed — workbook format unrecognized; "
                         "refusing to overwrite the snapshot.")
    tmp = HCPCS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, HCPCS_FILE)
    print(f"wrote {HCPCS_FILE} — {len(out)} HCPCS codes ({tag}, source {xlsx})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quarter", choices=_QUARTERS, help="release quarter (default: current)")
    ap.add_argument("--year", type=int, help="release year (default: current)")
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
