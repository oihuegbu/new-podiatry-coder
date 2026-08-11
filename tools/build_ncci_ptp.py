#!/usr/bin/env python3
"""Build data/codes/ncci_data.json (NCCI Procedure-to-Procedure edits) reproducibly
from PINNED, CHECKSUM-VERIFIED CMS release files.

Why this exists
---------------
This file is the offline snapshot the compliance store reads to build its NCCI PTP
table (`store._ingest_ncci` → `json.load(NCCI_FILE)`), and it is what claude_coder's
`check_ncci` ultimately resolves against. It used to be a ~497 MB git-LFS blob with
NO in-repo producer, so on any checkout where git-lfs was not present it silently
degraded to a 134-byte pointer — the NCCI table never populated and every PTP check
fell through to fail-closed. That is exactly the "fix the source, reproducible from a
clean build" failure this repo's conventions forbid.

Reproducible build inputs, not just a reproducible output (Codex F6-R8)
----------------------------------------------------------------------
The builder used to SCRAPE the CMS landing page for whichever "current"/"prior"
quarters CMS happened to expose at build time. The output lock detected drift, but the
INPUTS were mutable: CMS keeps only the current + prior quarter, so a clean rebuild of
an already-reviewed commit could later fail — or resolve different files — purely
because CMS rotated a quarter. That breaks disaster recovery and reproducible builds
even though the source code and the lock are unchanged.

So the default mode is now PINNED:

  * `data/sources/ncci_ptp.lock.json` records each release input's exact URL, file name,
    byte size and SHA-256, plus the resulting output SHA-256 and pair count;
  * the builder fetches EXACTLY those inputs and verifies every input checksum before
    parsing anything — a changed/rotated/substituted upstream file aborts the build
    loudly instead of silently producing different bytes;
  * `NCCI_INPUT_DIR` may point at a controlled immutable copy of those same files
    (licensed artifact storage, an S3 mirror, a DR bundle). Cached files are checksum-
    verified identically, so the recovery path is as trustworthy as the network path
    and needs no CMS availability at all.

Upgrading to a new CMS quarter is a SEPARATE, EXPLICIT workflow (`--refresh`) that
resolves the current quarters, records their identities/checksums, and writes a
PROPOSED lock for review. It never redefines the current image build in place.

Usage
-----
  python tools/build_ncci_ptp.py                      # pinned build from the reviewed lock
  python tools/build_ncci_ptp.py --refresh            # propose a NEW lock from CMS (review it)
  python tools/build_ncci_ptp.py --file f1.zip [--file f2.zip ...] --effective-from 2026-07-01
      # offline/air-gapped: parse local licensed/downloaded PTP files instead

Fail-closed: aborts non-zero (writing nothing) if an input is unavailable, an input
checksum does not match, no file resolves, or zero pairs parse — so a bad run never
overwrites a good snapshot with an empty or unverified one.
"""
from __future__ import annotations

import argparse
import hashlib
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

LOCK = ROOT / "data" / "sources" / "ncci_ptp.lock.json"
PROPOSED_LOCK = ROOT / "data" / "sources" / "ncci_ptp.lock.proposed.json"
INPUT_DIR_ENV = "NCCI_INPUT_DIR"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _all_quarter_files(src) -> list[tuple[str, str]]:
    """Every PRACTITIONER PTP edit zip URL on the CMS page (ALL quarters), as
    (url, quarter_effective_date), oldest quarter first. CMS keeps only current+prior
    quarter, but each file is CUMULATIVE (retains deleted edits) — unioning all
    available quarters maximizes historical completeness.

    Mirrors runner._resolve_ncci_ptp's URL extraction: the hrefs sit behind a
    '/license/ama?file=' wrapper that must be STRIPPED to reach the direct /files/zip/
    URL (downloading the wrapper returns an HTML license page, not the zip — the cause
    of the earlier 0-rows failure). Unlike the runner, this keeps ALL quarters, not
    just the newest.

    Only the explicit `--refresh` workflow calls this: a normal build must never depend
    on what CMS happens to be publishing today.
    """
    html = runner.download(src.url, timeout=120).decode("utf-8", errors="replace")
    hits = re.findall(
        r'href="([^"]*?(\d{4})q([1-4])-practitioner-ptp-edits[^"]*?-f\d[^"]*?\.zip)"',
        html, re.I)
    out = {}
    for href, yr, q in hits:
        url = runner._abs(src.url, re.sub(r"^/license/ama\?file=", "", href))
        out[url] = f"{yr}-{_QSTART[int(q)]}"
    return sorted(out.items(), key=lambda kv: kv[1])   # oldest quarter first


def _read_lock() -> dict:
    if not LOCK.exists():
        raise SystemExit(f"ABORT: pinned input lock {LOCK} is missing; a build must not "
                         f"resolve mutable upstream listings. Run --refresh to propose one.")
    lock = json.loads(LOCK.read_text())
    inputs = lock.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise SystemExit(f"ABORT: {LOCK} declares no pinned release inputs; the build inputs "
                         f"are unidentified and the build is not reproducible.")
    for i, spec in enumerate(inputs):
        missing = [k for k in ("name", "url", "sha256", "effective_from")
                   if not str(spec.get(k) or "").strip()]
        if missing:
            raise SystemExit(f"ABORT: {LOCK} input #{i} is missing {missing}")
    return lock


def _fetch_pinned(spec: dict) -> bytes:
    """One pinned input, from controlled storage when provided, else its pinned URL.

    The checksum is verified in BOTH paths, so a cached/mirrored copy carries exactly the
    same guarantee as a fresh download and neither can substitute different bytes."""
    name, want = spec["name"], str(spec["sha256"]).lower()
    cache_dir = os.getenv(INPUT_DIR_ENV)
    raw = None
    origin = spec["url"]
    if cache_dir:
        cached = Path(cache_dir) / name
        if cached.exists():
            raw = cached.read_bytes()
            origin = str(cached)
    if raw is None:
        raw = runner.download(spec["url"], timeout=300)
    got = _sha256(raw)
    if got != want:
        raise SystemExit(
            f"ABORT: pinned NCCI input {name} does not match its recorded checksum\n"
            f"  origin : {origin}\n  expected: {want}\n  actual  : {got}\n"
            f"The upstream release was rotated, replaced, or corrupted. Do NOT silently "
            f"rebuild from it: run `tools/build_ncci_ptp.py --refresh`, review the proposed "
            f"lock, and promote it deliberately.")
    if spec.get("bytes") and int(spec["bytes"]) != len(raw):
        raise SystemExit(f"ABORT: pinned NCCI input {name} size mismatch "
                         f"({len(raw)} != {spec['bytes']})")
    print(f"  verified pinned input {name} ({len(raw)} bytes, sha256 {got[:16]}…) "
          f"from {origin}")
    return raw


def _payloads(args, src) -> tuple[list[tuple[bytes, str, str | None]], list[dict]]:
    """((raw_bytes, name, quarter_effective_date) …, input provenance records).

    Oldest quarter first — the merge below is order-dependent (the newest quarter wins),
    so the input ORDER is part of the pinned identity, not an accident of scraping.
    """
    if args.file:
        payloads, records = [], []
        for f in args.file:
            raw = Path(f).read_bytes()
            payloads.append((raw, Path(f).name, args.effective_from))
            records.append({"name": Path(f).name, "url": f"file://{Path(f).resolve()}",
                            "sha256": _sha256(raw), "bytes": len(raw),
                            "effective_from": args.effective_from or ""})
        return payloads, records
    if args.refresh:
        quarters = _all_quarter_files(src)
        if not quarters:
            raise SystemExit("ABORT: no practitioner PTP zip resolved from the CMS landing "
                             "page (page format changed, or offline). Re-run with --file.")
        payloads, records = [], []
        for url, eff in quarters:
            raw = runner.download(url, timeout=300)
            name = url.rsplit("/", 1)[-1]
            payloads.append((raw, name, eff))
            records.append({"name": name, "url": url, "sha256": _sha256(raw),
                            "bytes": len(raw), "effective_from": eff})
            print(f"  resolved {name} ({eff}, {len(raw)} bytes)")
        return payloads, records
    lock = _read_lock()
    payloads, records = [], []
    for spec in lock["inputs"]:
        raw = _fetch_pinned(spec)
        payloads.append((raw, spec["name"], spec["effective_from"]))
        records.append(dict(spec))
    return payloads, records


def build(args) -> int:
    src = SOURCES_BY_ID["ncci_ptp"]
    payloads, input_records = _payloads(args, src)   # oldest quarter first
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
    if args.refresh:
        _write_proposed_lock(input_records, len(merged), newest_eff)
    return 0


def _write_proposed_lock(inputs: list[dict], pairs: int, effective_from: str) -> None:
    """Record the resolved inputs + the resulting output as a PROPOSED lock.

    Deliberately a separate file: an intentional version upgrade must be reviewed and
    promoted, never applied by a build that silently redefines what the image contains.
    """
    digest = hashlib.sha256()
    with open(NCCI_FILE, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    proposed = {
        "source": "CMS NCCI practitioner PTP edits",
        "builder": "tools/build_ncci_ptp.py",
        "output": str(NCCI_FILE.relative_to(ROOT)),
        "effective_from": effective_from,
        "pairs": pairs,
        "output_sha256": digest.hexdigest(),
        "inputs": inputs,
        "input_cache_env": INPUT_DIR_ENV,
        "note": ("PROPOSED lock produced by tools/build_ncci_ptp.py --refresh. Review the "
                 "input identities/checksums and the resulting output checksum, then promote "
                 "this file over ncci_ptp.lock.json in a reviewed commit."),
    }
    PROPOSED_LOCK.parent.mkdir(parents=True, exist_ok=True)
    PROPOSED_LOCK.write_text(json.dumps(proposed, indent=2) + "\n")
    print(f"wrote proposed lock {PROPOSED_LOCK.relative_to(ROOT)} — review and promote it "
          f"deliberately; the current build still uses {LOCK.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", action="append", default=[],
                    help="local PTP file(s) to parse instead of the pinned inputs")
    ap.add_argument("--refresh", action="store_true",
                    help="resolve the CURRENT CMS quarters and write a PROPOSED new lock "
                         "(explicit version upgrade; never used by an image build)")
    ap.add_argument("--effective-from", dest="effective_from",
                    help="override the effective date (YYYY-MM-DD)")
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
