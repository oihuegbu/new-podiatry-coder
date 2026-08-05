#!/usr/bin/env python3
"""Build data/codes/icd10cm_codes.json (ICD-10-CM code set) reproducibly from the
current CMS/NCHS fiscal-year release — same package pattern as the NCCI builders.

CMS publishes '{FY}-code-descriptions-tabular-order.zip' containing the fixed-width
'icd10cm_order_{FY}.txt' (the "order file"):
    order#(5)  code(cols 6-12)  flag(col 14: 0=header/non-billable, 1=billable)
    short-desc(cols 16-75)  long-desc(cols 77+)
Only billable (flag=1) leaf codes are coded, matching the store's code_set.
store._ingest_code_set / code_reference read {code, description, effective_from,
effective_to, fy, status}. No hardcoded codes; fail-closed; atomic write.

Usage:
  python tools/build_icd10cm.py                 # current fiscal year from CMS
  python tools/build_icd10cm.py --fy 2026
  python tools/build_icd10cm.py --file icd10cm_order_2026.txt --fy 2026   # offline
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.core.config import ICD10_FILE


def _current_fy(today: date | None = None) -> int:
    """ICD-10-CM fiscal year: FY{Y+1} begins Oct 1 of year Y."""
    d = today or date.today()
    return d.year + 1 if d.month >= 10 else d.year


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "coder-refresh/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def _parse_order(text: str, fy: int) -> list[dict]:
    eff = f"{fy - 1}-10-01"
    out = []
    for line in text.splitlines():
        if len(line) < 16:
            continue
        if line[14] != "1":                 # keep only billable leaf codes (flag 1)
            continue
        code = line[6:13].strip().upper()
        if not code:
            continue
        long_desc = line[77:].strip() if len(line) > 77 else line[16:76].strip()
        out.append({"code": code, "description": long_desc,
                    "effective_from": eff, "effective_to": "",
                    "fy": str(fy), "status": "active"})
    return out


def build(args) -> int:
    fy = args.fy or _current_fy()
    if args.file:
        text = Path(args.file[0]).read_text(errors="replace")
    else:
        url = f"https://www.cms.gov/files/zip/{fy}-code-descriptions-tabular-order.zip"
        try:
            zf = zipfile.ZipFile(io.BytesIO(_download(url)))
        except Exception as exc:
            raise SystemExit(f"ABORT: could not fetch/open {url} ({exc}). If CMS renamed "
                             f"the FY{fy} zip, pass --file the icd10cm_order_{fy}.txt.")
        member = next((n for n in zf.namelist()
                       if "order" in n.lower() and "addenda" not in n.lower()
                       and n.lower().endswith(".txt")), None)
        if not member:
            raise SystemExit(f"ABORT: no icd10cm_order .txt in {url}: {zf.namelist()[:6]}")
        text = zf.read(member).decode("latin-1", errors="replace")

    out = _parse_order(text, fy)
    if not out:
        raise SystemExit("ABORT: 0 billable ICD-10-CM codes parsed — format unrecognized; "
                         "refusing to overwrite the snapshot.")
    tmp = ICD10_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, ICD10_FILE)
    print(f"wrote {ICD10_FILE} — {len(out)} billable ICD-10-CM codes (FY{fy})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fy", type=int, help="fiscal year (default: current)")
    ap.add_argument("--file", action="append", default=[],
                    help="local icd10cm_order .txt to parse instead of fetching")
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
