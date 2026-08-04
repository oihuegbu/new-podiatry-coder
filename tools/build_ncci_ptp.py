#!/usr/bin/env python3
"""Build data/codes/ncci_data.json (NCCI Procedure-to-Procedure edits) reproducibly
from the CURRENT CMS quarterly release.

Why this exists
---------------
This file is the offline snapshot the compliance store reads to build its NCCI PTP
table (`store._ingest_ncci` → `json.load(NCCI_FILE)`), and it is what claude_coder's
`check_ncci` ultimately resolves against. It used to be a ~497 MB git-LFS blob with
NO in-repo producer, so on any checkout where git-lfs was not present it silently
degraded to a 134-byte pointer — the NCCI table never populated and every PTP check
fell through to fail-closed. That is exactly the "fix the source, reproducible from a
clean build" failure this repo's conventions forbid.

Now it is a build product, like snomed_icd10_map.json / global_period.json: this tool
uses the SAME fetch + parse machinery the live refresh runner uses
(`app.compliance.refresh`) — it scrapes the CMS landing page for the newest quarter's
practitioner PTP files, downloads and parses them, and writes the JSON the store
reads. It hardcodes NO medical codes; every edit pair comes from CMS.

Usage
-----
  python tools/build_ncci_ptp.py            # newest quarter from CMS
  python tools/build_ncci_ptp.py --file f1.zip [--file f2.zip ...] --effective-from 2026-07-01
      # offline/air-gapped: parse local licensed/downloaded PTP files instead

Fail-closed: aborts non-zero (writing nothing) if no file resolves or zero pairs
parse, so a bad run never overwrites a good snapshot with an empty one.
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

from app.core.config import NCCI_FILE
from app.compliance.refresh import runner, parsers as P
from app.compliance.refresh.sources import SOURCES_BY_ID


def _payloads(args, src) -> tuple[list[tuple[bytes, str]], str | None]:
    """(raw_bytes, name) for each PTP file + the resolved effective date."""
    if args.file:
        return [(Path(f).read_bytes(), Path(f).name) for f in args.file], args.effective_from
    urls, eff = runner._resolve_urls(src)
    if not urls:
        raise SystemExit("ABORT: no NCCI PTP file resolved from the CMS landing page "
                         "(page format changed, or offline). Re-run with --file.")
    return [(runner.download(u, timeout=300), u.rsplit("/", 1)[-1]) for u in urls], eff


def build(args) -> int:
    src = SOURCES_BY_ID["ncci_ptp"]
    payloads, eff = _payloads(args, src)
    eff = args.effective_from or eff or date.today().isoformat()

    tmp = NCCI_FILE.with_suffix(".json.tmp")
    total = 0
    # Stream the (large) output so a ~2M-pair snapshot never materializes fully in
    # memory. Parse one PTP file at a time and dump its rows immediately.
    with open(tmp, "w") as out:
        out.write("[")
        first = True
        for raw, name in payloads:
            text = runner._payload_text(src, raw)
            rows, _cols = P.parse_ncci(text, eff)   # (col1, col2, mod_ind, eff_from, eff_to)
            for c1, c2, mod, eff_from, eff_to in rows:
                if not first:
                    out.write(",")
                first = False
                # keys the store reads verbatim: code1/code2/modifier/effective_date/end_date
                json.dump({"code1": c1, "code2": c2, "modifier": mod,
                           "effective_date": eff_from, "end_date": eff_to}, out)
                total += 1
            print(f"  parsed {name}: running total {total} pairs")
        out.write("]")

    if total == 0:
        tmp.unlink(missing_ok=True)
        raise SystemExit("ABORT: 0 PTP pairs parsed — payload was a landing page or an "
                         "unrecognized format; refusing to overwrite the snapshot.")
    os.replace(tmp, NCCI_FILE)   # atomic: a reader never sees a partial file
    print(f"wrote {NCCI_FILE} — {total} NCCI PTP pairs (effective {eff})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", action="append", default=[],
                    help="local PTP file(s) to parse instead of fetching from CMS")
    ap.add_argument("--effective-from", dest="effective_from",
                    help="override the effective date (YYYY-MM-DD)")
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
