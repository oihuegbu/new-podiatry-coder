#!/usr/bin/env python3
"""Parse AHRQ HCUP's Chronic Condition Indicator Refined (CCIR) for
ICD-10-CM into data/codes/icd10cm_chronic.json.

Source: https://hcup-us.ahrq.gov/toolssoftware/chronic_icd10/chronic_icd10.jsp
(ZIP containing CCIR_v<fy>-<r>.csv). The CSV maps EVERY ICD-10-CM diagnosis
code to a chronicity indicator: 1=chronic, 0=not chronic, 9=no determination.

This is the authoritative chronicity source behind the E/M problems-axis
floor (validator._check_em_mdm_problems_floor): the 2021 AMA MDM table's
moderate row includes '2 or more stable chronic illnesses', and 'chronic'
must come from a real classification, never a keyword guess against
descriptions.

Usage:
    python tools/parse_ccir.py CCIR_v2026-1.csv [output.json]
"""

import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

OUT_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "codes" / "icd10cm_chronic.json"


def parse(csv_path: Path) -> dict:
    codes: dict[str, int] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            code = row[0].strip().strip("'\"").upper()
            ind = row[-1].strip().strip("'\"")
            # Skip banner/header rows — real rows have an ICD-shaped code
            # (letter + digits) and a 0/1/9 indicator.
            if not re.fullmatch(r"[A-Z][0-9][0-9A-Z]{1,5}", code):
                continue
            if ind not in ("0", "1", "9"):
                continue
            codes[code] = int(ind)
    if len(codes) < 50000:
        raise SystemExit(
            f"Parsed only {len(codes)} codes — CCIR carries the full ICD-10-CM "
            f"code set (~74k); this looks like the wrong file or a format change.")
    return codes


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    csv_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_DEFAULT

    codes = parse(csv_path)
    n_chronic = sum(1 for v in codes.values() if v == 1)
    payload = {
        "metadata": {
            "source": "AHRQ HCUP Chronic Condition Indicator Refined (CCIR) for ICD-10-CM",
            "source_url": "https://hcup-us.ahrq.gov/toolssoftware/chronic_icd10/chronic_icd10.jsp",
            "source_file": csv_path.name,
            "indicator_values": {"0": "not chronic", "1": "chronic", "9": "no determination"},
            "generated": date.today().isoformat(),
            "total_codes": len(codes),
            "chronic_codes": n_chronic,
        },
        "codes": codes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Wrote {out_path}: {len(codes)} codes ({n_chronic} chronic)")


if __name__ == "__main__":
    main()
