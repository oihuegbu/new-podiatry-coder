#!/usr/bin/env python3
"""Build data/codes/mue_practitioner.json (NCCI Practitioner MUE table) reproducibly
from the CURRENT CMS quarterly release — same pattern as build_ncci_ptp.py.

The store reads a JSON list of {code, mue_value, description, effective_date, end_date}
(store._ingest_mue derives the MUE Adjudication Indicator from the leading character of
`description`). This builder fetches the newest CMS MUE quarter via the live refresh
machinery (app.compliance.refresh), parses it, and writes that shape. No hardcoded codes.

Usage
-----
  python tools/build_mue.py                         # newest quarter from CMS
  python tools/build_mue.py --file MUE.csv --effective-from 2026-07-01   # offline

Fail-closed: aborts (writing nothing) if no file resolves or zero rows parse, so a bad
run never overwrites a good snapshot with an empty one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import MUE_FILE
from app.compliance.refresh import runner, parsers as P
from app.compliance.refresh.sources import SOURCES_BY_ID


def build(args) -> int:
    src = SOURCES_BY_ID["mue"]
    if args.file:
        payloads, eff = [(Path(f).read_bytes(), Path(f).name) for f in args.file], args.effective_from
    else:
        urls, eff = runner._resolve_urls(src)
        if not urls:
            raise SystemExit("ABORT: no MUE file resolved from the CMS landing page "
                             "(page format changed, or offline). Re-run with --file.")
        payloads = [(runner.download(u, timeout=300), u.rsplit("/", 1)[-1]) for u in urls]
    eff = args.effective_from or eff or date.today().isoformat()

    out = []
    for raw, name in payloads:
        text = runner._payload_text(src, raw)
        rows, _cols = P.parse_mue(text, eff)   # (code, mue_value, mai, rationale, eff_from, eff_to)
        for code, val, mai, rationale, eff_from, eff_to in rows:
            # store._ingest_mue reads the MAI from description[0]; reconstruct that shape.
            description = f"{mai} {rationale}".strip() if mai else rationale
            out.append({"code": code, "mue_value": val, "description": description,
                        "effective_date": eff_from, "end_date": eff_to,
                        "source_file": name})

    if not out:
        raise SystemExit("ABORT: 0 MUE rows parsed — payload was a landing page or an "
                         "unrecognized format; refusing to overwrite the snapshot.")
    tmp = MUE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, MUE_FILE)   # atomic
    print(f"wrote {MUE_FILE} — {len(out)} MUE entries (effective {eff}, {payloads[0][1]})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", action="append", default=[],
                    help="local MUE file(s) to parse instead of fetching from CMS")
    ap.add_argument("--effective-from", dest="effective_from",
                    help="override the effective date (YYYY-MM-DD)")
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
