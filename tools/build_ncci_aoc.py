#!/usr/bin/env python3
"""Build data/codes/ncci_aoc_edits.json (NCCI Add-On Code edits) reproducibly from the
current CMS quarterly release — same package pattern as build_ncci_ptp.py / build_mue.py.

The CMS "Add-On Code Edits" landing page publishes a quarterly zip
(add-code-edits-medicare-effective-MMDDYYYY.zip) containing a fixed-width TXT whose
data lines are:
    {modifier}{code1}   code2   YYYYDDD(julian eff date)   description
e.g. '20054T   CCCCC   2013091   Contractor Defined Primary Codes'
  -> modifier=2, add-on code1=0054T, primary code2=CCCCC, eff=2013-04-01.
store._ingest_ncci_aoc reads {code1, code2, modifier, effective_date, end_date}.
No hardcoded codes; fail-closed on no-file/zero-rows; atomic write.

Usage:
  python tools/build_ncci_aoc.py                    # newest quarter from CMS
  python tools/build_ncci_aoc.py --file AOC.txt --effective-from 2026-07-01   # offline
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.core.config import CODES_DIR

AOC_FILE = CODES_DIR / "ncci_aoc_edits.json"
LANDING = ("https://www.cms.gov/medicare/coding-billing/"
           "national-correct-coding-initiative-ncci-edits/medicare-ncci-add-code-edits")
_OPEN = "9999-12-31"
_CODE_RE = re.compile(r"^[A-Z0-9]{4,5}$")


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "coder-refresh/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def _resolve_newest(html: str) -> tuple[str | None, str | None]:
    """Newest add-on-code zip URL + its effective date (from the MMDDYYYY filename)."""
    best, best_eff = None, None
    for h in re.findall(r'href="([^"]*add-code-edits[^"]*\.zip)"', html, re.I):
        m = re.search(r"effective-(\d{2})(\d{2})(\d{4})", h, re.I)
        if not m:
            continue
        mm, dd, yyyy = m.groups()
        eff = f"{yyyy}-{mm}-{dd}"
        if best_eff is None or eff > best_eff:
            best, best_eff = h, eff
    if best and best.startswith("/"):
        best = "https://www.cms.gov" + best
    return best, best_eff


def _julian_to_iso(tok: str) -> str:
    """YYYYDDD (year + day-of-year) -> YYYY-MM-DD; '' if not parseable."""
    tok = tok.strip()
    if not re.fullmatch(r"\d{7}", tok):
        return ""
    year, doy = int(tok[:4]), int(tok[4:])
    if not (1 <= doy <= 366):
        return ""
    return (date(year, 1, 1) + timedelta(days=doy - 1)).isoformat()


def _parse(text: str, snapshot_eff: str, source_file: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        f1 = parts[0].strip().upper()
        # field 1 is {modifier}{code1}: leading 0/1/2/9 is the edit indicator.
        if f1[:1] not in ("0", "1", "2", "9"):
            continue
        modifier, code1 = f1[0], f1[1:]
        code2 = parts[1].strip().upper()
        if not (_CODE_RE.match(code1) and _CODE_RE.match(code2)):
            continue
        eff = _julian_to_iso(parts[2]) or snapshot_eff
        desc = " ".join(parts[3:]).strip()
        out.append({"code1": code1, "code2": code2, "edit_type": "AOC",
                    "modifier": modifier, "effective_date": eff, "end_date": "",
                    "description": desc, "metadata": {"source_file": source_file}})
    return out


def build(args) -> int:
    if args.file:
        raw_text = Path(args.file[0]).read_text(errors="replace")
        name = Path(args.file[0]).name
        eff = args.effective_from or date.today().isoformat()
        out = _parse(raw_text, eff, name)
    else:
        url, eff = _resolve_newest(_download(LANDING).decode("utf-8", errors="replace"))
        if not url:
            raise SystemExit("ABORT: no add-on-code zip resolved from the CMS landing page "
                             "(page format changed, or offline). Re-run with --file.")
        eff = args.effective_from or eff or date.today().isoformat()
        zf = zipfile.ZipFile(io.BytesIO(_download(url)))
        member = next((n for n in zf.namelist() if n.lower().endswith(".txt")), None)
        if not member:
            raise SystemExit(f"ABORT: no .txt member in {url}: {zf.namelist()[:6]}")
        out = _parse(zf.read(member).decode("latin-1", errors="replace"), eff, member)

    if not out:
        raise SystemExit("ABORT: 0 AOC edits parsed — payload format unrecognized; "
                         "refusing to overwrite the snapshot.")
    tmp = AOC_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, AOC_FILE)
    print(f"wrote {AOC_FILE} — {len(out)} NCCI AOC edits (effective {eff})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", action="append", default=[],
                    help="local AOC .txt file to parse instead of fetching from CMS")
    ap.add_argument("--effective-from", dest="effective_from",
                    help="override the snapshot effective date (YYYY-MM-DD)")
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
