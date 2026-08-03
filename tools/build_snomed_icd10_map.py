#!/usr/bin/env python3
"""Build data/codes/snomed_icd10_map.json — the SNOMED CT -> ICD-10-CM term map
that backs the coder's second authoritative resolution layer.

The ICD-10-CM Alphabetic Index covers many clinician terms; SNOMED CT is the
comprehensive clinical terminology (millions of synonyms/eponyms), and NLM
publishes the authoritative SNOMED CT US Edition -> ICD-10-CM map. This tool
ingests that map — via the public, no-login copy the open-source Tuva Project
mirrors on S3 (a verbatim redistribution of the NLM SNOMED CT US Edition map) —
and inverts it to term -> ICD-10-CM code(s). Each row already carries the SNOMED
term AND the ICD-10-CM mapTarget, so no separate SNOMED description file is
needed.

Only the UNCONDITIONAL default maps (mapRule TRUE / OTHERWISE TRUE) are kept —
age/context-conditional rules need patient data the coder resolves elsewhere.

LICENSING: the underlying content is SNOMED CT (US Edition), free for use within
the United States under the NLM UMLS license; ICD-10-CM is public domain. Confirm
the SNOMED CT license applies to your use before deploying.

Usage (needs internet; run on the box):
  python tools/build_snomed_icd10_map.py            # downloads the public map
  python tools/build_snomed_icd10_map.py --file <local snomed_icd_10_map.csv[.gz]>
"""
import argparse
import csv
import gzip
import io
import json
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.config import DATA_DIR

DEFAULT_URL = ("https://tuva-public-resources.s3.amazonaws.com/versioned_terminology/"
               "latest/snomed_icd_10_map.csv_0_0_0.csv.gz")
# NLM ExtendedMap column order (headerless): ... active(2) ... term(6) ...
# mapRule(9) mapAdvice(10) mapTarget(11) ...
_ACTIVE, _TERM, _RULE, _TARGET = 2, 6, 9, 11
_ICD = re.compile(r"^[A-Z]\d{2}[A-Z0-9]*$")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ",
                  str(s).lower().replace("'", ""))).strip()


def _dot(code: str) -> str:
    c = code.upper()
    return c if len(c) <= 3 else f"{c[:3]}.{c[3:]}"


def _open(args):
    if args.file:
        raw = Path(args.file).read_bytes()
    else:
        print(f"Downloading {args.url} …")
        req = urllib.request.Request(args.url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=180).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return io.StringIO(raw.decode("utf-8", "replace"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--file", help="local SNOMED->ICD map CSV (.gz ok) instead of download")
    args = ap.parse_args()

    terms: dict[str, set[str]] = defaultdict(set)
    rows = kept = 0
    for row in csv.reader(_open(args)):
        rows += 1
        if len(row) <= _TARGET or row[_ACTIVE] != "1":
            continue
        rule = row[_RULE].strip().upper()
        if rule not in ("TRUE", "OTHERWISE TRUE"):   # unconditional default maps only
            continue
        target = row[_TARGET].strip().upper()
        if not _ICD.match(target):                   # skip 'NC'/blank/non-ICD targets
            continue
        term = _norm(row[_TERM])
        if term:
            terms[term].add(_dot(target))
            kept += 1

    out = DATA_DIR / "codes" / "snomed_icd10_map.json"
    payload = {
        "source": "NLM SNOMED CT US Edition -> ICD-10-CM map (public Tuva Project mirror)",
        "url": args.url,
        "license": "SNOMED CT US Edition (NLM UMLS, free for US use); ICD-10-CM public domain",
        "provenance": "authoritative NLM map, unconditional default rules, inverted to term->code",
        "generated": date.today().isoformat(),
        "count": len(terms),
        "terms": {t: sorted(c) for t, c in sorted(terms.items())},
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"Parsed {rows} rows, kept {kept} term-maps -> {len(terms)} distinct terms -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
