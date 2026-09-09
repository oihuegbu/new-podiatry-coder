#!/usr/bin/env python3
"""Build data/codes/global_period.json from the authoritative CMS source.

Global-surgical-package days are published by CMS in the Physician Fee Schedule
Relative Value File (the PPRRVU file), column "GLOB DAYS". This tool downloads
the current quarterly RVU zip from cms.gov, parses that column, and writes a
provenance-tagged JSON the coder loads — so the global periods are real CMS data
that self-updates each quarter, never hand-entered.

Values: 000 (0-day minor), 010 (10-day minor), 090 (90-day major), XXX (concept
does not apply), YYY (carrier-priced), ZZZ (add-on, global tied to the primary),
MMM (maternity).

Also extracts the SAME file's STATUS CODE column (issue #6 F9-R11-H-B): the CMS
payment-status indicator that `data/codes/coding_semantics.json`'s `anesthesia`
class rule (`pfs_status_any: ["J"]`) is configured to read but, until this, had
no wired source column -- `claude_coder.semantic_eligibility` was falling back to
a hand-authored descriptor-phrase proxy for exactly this classification, which
CLAUDE.md's no-hardcoding rule forbids for a claim-affecting decision. One real
CMS field, the one the rule already named.

Usage (needs internet; run on the box):
  python tools/build_global_period.py [--url https://www.cms.gov/files/zip/rvu26b.zip]
"""
import argparse
import csv
import io
import json
import sys
import urllib.request
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.release.source_manifest import declared_source_path

DEFAULT_URL = "https://www.cms.gov/files/zip/rvu26b.zip"
GLOBAL_VALUES = {"000", "010", "090", "XXX", "YYY", "ZZZ", "MMM"}
#: CMS PFS payment-status indicators (PPRRVU file's own published legend) -- a
#: payment-POLICY classification value, structurally the same kind of thing
#: GLOBAL_VALUES already is in this file (used here only to LOCATE the column by
#: its value shape, never to select or exclude a medical code). Single uppercase
#: letters; distinct enough from the surrounding numeric RVU/dollar columns that
#: an incomplete list still detects the column reliably.
STATUS_VALUES = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "L", "M",
                 "N", "P", "Q", "R", "T", "X"}


def _pprrvu_rows(zip_bytes: bytes) -> list[list[str]]:
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    name = next((n for n in zf.namelist()
                 if n.lower().startswith("pprrvu") and n.lower().endswith(".csv")
                 and "nonqpp" in n.lower()), None)
    name = name or next(n for n in zf.namelist()
                        if n.lower().startswith("pprrvu") and n.lower().endswith(".csv"))
    text = zf.read(name).decode("latin-1")
    return list(csv.reader(io.StringIO(text))), name


def _glob_column(data_rows: list[list[str]]) -> int:
    """Locate the GLOB DAYS column by the column whose values are consistently
    global-period codes — robust to layout shifts, no hardcoded index."""
    counts: Counter = Counter()
    for r in data_rows[:800]:
        for idx, v in enumerate(r):
            if v.strip().upper() in GLOBAL_VALUES:
                counts[idx] += 1
    if not counts:
        raise SystemExit("no GLOB DAYS column found — file format changed")
    return counts.most_common(1)[0][0]


def _status_column(data_rows: list[list[str]]) -> int:
    """Locate the STATUS CODE column the SAME robust way as `_glob_column` --
    the column whose values are consistently drawn from the CMS payment-status
    indicator set, not a fixed index (issue #6 F9-R11-H-B)."""
    counts: Counter = Counter()
    for r in data_rows[:800]:
        for idx, v in enumerate(r):
            if v.strip().upper() in STATUS_VALUES:
                counts[idx] += 1
    if not counts:
        raise SystemExit("no STATUS CODE column found — file format changed")
    return counts.most_common(1)[0][0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args()

    print(f"Downloading {args.url} …")
    req = urllib.request.Request(args.url, headers={"User-Agent": "Mozilla/5.0"})
    zip_bytes = urllib.request.urlopen(req, timeout=120).read()
    rows, name = _pprrvu_rows(zip_bytes)
    print(f"Parsing {name} ({len(rows)} rows)")

    hdr = next((i for i, r in enumerate(rows)
                if r and r[0].strip().upper() == "HCPCS"), None)
    if hdr is None:
        raise SystemExit("HCPCS header row not found")
    data = rows[hdr + 1:]
    gcol = _glob_column(data)
    scol = _status_column(data)

    # BILAT SURG sits a fixed 5 columns after GLOB DAYS in the PPRRVU layout
    # (GLOB, PRE OP, INTRA OP, POST OP, MULT PROC, BILAT SURG). Values 0/1/2/3/9.
    bcol = gcol + 5
    bilat_values = {"0", "1", "2", "3", "9"}

    codes: dict[str, dict] = {}
    for r in data:
        if len(r) <= gcol:
            continue
        code = r[0].strip().upper()
        mod = r[1].strip()
        gval = r[gcol].strip().upper()
        if not code or mod or gval not in GLOBAL_VALUES:
            continue                              # base code rows only
        bval = r[bcol].strip() if len(r) > bcol else ""
        sval = r[scol].strip().upper() if len(r) > scol else ""
        codes[code] = {"global": gval,
                       "bilat": bval if bval in bilat_values else "9",
                       "status": sval if sval in STATUS_VALUES else ""}

    # The PRODUCER writes exactly where the declaration says the coder reads and the
    # release manifest content-addresses -- one path, three consumers. (Codex F6-R5.)
    out = declared_source_path("pfs_indicators")
    payload = {
        "source": "CMS Medicare PFS Relative Value File (PPRRVU): GLOB DAYS + "
                  "BILAT SURG + STATUS CODE columns",
        "url": args.url,
        "file": name,
        "provenance": "authoritative CMS data, parsed verbatim from the RVU file",
        "generated": date.today().isoformat(),
        "count": len(codes),
        "codes": dict(sorted(codes.items())),
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"Wrote {len(codes)} codes -> {out}")
    print(f"Global: {dict(sorted(Counter(v['global'] for v in codes.values()).items()))}")
    print(f"Bilat:  {dict(sorted(Counter(v['bilat'] for v in codes.values()).items()))}")
    print(f"Status: {dict(sorted(Counter(v['status'] for v in codes.values()).items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
