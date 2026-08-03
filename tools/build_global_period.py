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
from app.core.config import DATA_DIR

DEFAULT_URL = "https://www.cms.gov/files/zip/rvu26b.zip"
GLOBAL_VALUES = {"000", "010", "090", "XXX", "YYY", "ZZZ", "MMM"}


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

    periods: dict[str, str] = {}
    for r in data:
        if len(r) <= gcol:
            continue
        code = r[0].strip().upper()
        mod = r[1].strip()
        val = r[gcol].strip().upper()
        if not code or mod or val not in GLOBAL_VALUES:
            continue                              # base code rows only
        periods[code] = val

    out = DATA_DIR / "codes" / "global_period.json"
    payload = {
        "source": "CMS Medicare PFS Relative Value File (PPRRVU), GLOB DAYS column",
        "url": args.url,
        "file": name,
        "provenance": "authoritative CMS data, parsed verbatim from the RVU file",
        "generated": date.today().isoformat(),
        "count": len(periods),
        "global_period": dict(sorted(periods.items())),
    }
    out.write_text(json.dumps(payload, indent=1))
    dist = Counter(periods.values())
    print(f"Wrote {len(periods)} codes -> {out}")
    print(f"Distribution: {dict(sorted(dist.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
