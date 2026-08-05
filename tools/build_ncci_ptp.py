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
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import NCCI_FILE
from app.compliance.refresh import runner, parsers as P
from app.compliance.refresh.sources import SOURCES_BY_ID

_QSTART = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}


def _all_quarter_files(src) -> list[tuple[str, str]]:
    """Every PRACTITIONER PTP edit zip URL on the CMS page (ALL quarters), as
    (url, quarter_effective_date), oldest quarter first. CMS keeps only current+prior
    quarter, but each file is CUMULATIVE (retains deleted edits) — unioning all
    available quarters maximizes historical completeness.

    Mirrors runner._resolve_ncci_ptp's URL extraction: the hrefs sit behind a
    '/license/ama?file=' wrapper that must be STRIPPED to reach the direct /files/zip/
    URL (downloading the wrapper returns an HTML license page, not the zip — the cause
    of the earlier 0-rows failure). Unlike the runner, this keeps ALL quarters, not
    just the newest."""
    html = runner.download(src.url, timeout=120).decode("utf-8", errors="replace")
    hits = re.findall(
        r'href="([^"]*?(\d{4})q([1-4])-practitioner-ptp-edits[^"]*?-f\d[^"]*?\.zip)"',
        html, re.I)
    out = {}
    for href, yr, q in hits:
        url = runner._abs(src.url, re.sub(r"^/license/ama\?file=", "", href))
        out[url] = f"{yr}-{_QSTART[int(q)]}"
    return sorted(out.items(), key=lambda kv: kv[1])   # oldest quarter first


def _payloads(args, src) -> list[tuple[bytes, str, str | None]]:
    """(raw_bytes, name, quarter_effective_date) for every PTP file, oldest quarter first."""
    if args.file:
        return [(Path(f).read_bytes(), Path(f).name, args.effective_from) for f in args.file]
    quarters = _all_quarter_files(src)
    if not quarters:
        raise SystemExit("ABORT: no practitioner PTP zip resolved from the CMS landing page "
                         "(page format changed, or offline). Re-run with --file.")
    return [(runner.download(u, timeout=300), u.rsplit("/", 1)[-1], eff) for u, eff in quarters]


def build(args) -> int:
    src = SOURCES_BY_ID["ncci_ptp"]
    payloads = _payloads(args, src)                 # oldest quarter first
    newest_eff = args.effective_from or (payloads[-1][2] if payloads else None) or date.today().isoformat()

    # UNION all quarters: key each edit by (code1, code2, effective_from); the NEWEST
    # quarter processed last wins, so an edit's end_date reflects the latest release
    # (an edit deleted this quarter carries its real deletion date, not an open one).
    merged: dict[tuple[str, str, str], tuple[str, str]] = {}
    for raw, name, file_eff in payloads:
        text = runner._payload_text(src, raw)
        rows, _cols = P.parse_ncci(text, file_eff or newest_eff)   # (c1,c2,mod,eff_from,eff_to)
        for c1, c2, mod, eff_from, eff_to in rows:
            merged[(c1, c2, eff_from)] = (mod, eff_to)
        print(f"  parsed {name}: {len(rows)} rows -> merged total {len(merged)}")

    if not merged:
        raise SystemExit("ABORT: 0 PTP pairs parsed — payload was a landing page or an "
                         "unrecognized format; refusing to overwrite the snapshot.")
    tmp = NCCI_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as out:                     # stream the ~2M-pair snapshot
        out.write("[")
        first = True
        for (c1, c2, eff_from), (mod, eff_to) in merged.items():
            if not first:
                out.write(",")
            first = False
            json.dump({"code1": c1, "code2": c2, "modifier": mod,
                       "effective_date": eff_from, "end_date": eff_to}, out)
        out.write("]")
    os.replace(tmp, NCCI_FILE)   # atomic: a reader never sees a partial file
    print(f"wrote {NCCI_FILE} — {len(merged)} NCCI PTP pairs "
          f"(unioned {len(payloads)} quarterly file(s), newest effective {newest_eff})")
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
