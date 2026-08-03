#!/usr/bin/env python3
"""Build data/codes/snomed_icd10_map.json from the authoritative SNOMED CT ->
ICD-10-CM map (NLM/UMLS).

The ICD-10-CM Alphabetic Index covers many clinician terms, but SNOMED CT is the
comprehensive clinical terminology — millions of synonyms/eponyms — and NLM
publishes the authoritative SNOMED CT -> ICD-10-CM map. Joining SNOMED
DESCRIPTIONS (term -> concept) with that MAP (concept -> ICD-10-CM code) yields a
term -> code lookup for the long tail the Index lacks (e.g. "Morton's neuroma").

Source (free, but requires a UMLS license — https://uts.nlm.nih.gov/):
  SNOMED CT US Edition RF2 release. Two files:
    - Descriptions:  sct2_Description_*-en_US*.txt   (conceptId, term)
    - ICD-10-CM map: der2_iisssccRefset_ExtendedMapFull*_US*.txt
                     (referencedComponentId = concept, mapTarget = ICD-10-CM)

This tool does not download SNOMED (it is license-gated); point it at the RF2
files you extracted from your UMLS release:

  python tools/build_snomed_icd10_map.py --desc <sct2_Description...txt> \\
                                         --map  <der2_...ExtendedMapFull...txt>

Output is provenance-tagged JSON; the coder loads it as the second authoritative
layer (after the ICD Index, before embedding). Absent the file, that layer is a
no-op and the coder behaves exactly as today.
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.config import DATA_DIR

_ICD = re.compile(r"^[A-Z]\d{2}(\.[A-Z0-9]{1,4})?$")     # ICD-10-CM shape (dotted)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ",
                  str(s).lower().replace("'", ""))).strip()


def _rows(path: str):
    with open(path, encoding="utf-8") as fh:
        r = csv.reader(fh, delimiter="\t")
        header = next(r)
        idx = {name: i for i, name in enumerate(header)}
        for row in r:
            yield row, idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", required=True, help="SNOMED sct2_Description RF2 file")
    ap.add_argument("--map", required=True, help="SNOMED ExtendedMapFull (ICD-10-CM) RF2 file")
    args = ap.parse_args()

    # concept -> {ICD-10-CM codes}
    concept_codes: dict[str, set[str]] = defaultdict(set)
    for row, idx in _rows(args.map):
        if row[idx["active"]] != "1":
            continue
        target = row[idx.get("mapTarget", -1)].strip().upper() if "mapTarget" in idx else ""
        if _ICD.match(target):
            concept_codes[row[idx["referencedComponentId"]]].add(target)

    # term -> {ICD-10-CM codes}, via each concept's descriptions
    terms: dict[str, set[str]] = defaultdict(set)
    for row, idx in _rows(args.desc):
        if row[idx["active"]] != "1":
            continue
        concept = row[idx["conceptId"]]
        codes = concept_codes.get(concept)
        if not codes:
            continue
        n = _norm(row[idx["term"]])
        if n:
            terms[n].update(codes)

    out = DATA_DIR / "codes" / "snomed_icd10_map.json"
    payload = {
        "source": "NLM SNOMED CT US Edition -> ICD-10-CM map (ExtendedMapFull) + descriptions",
        "provenance": "authoritative UMLS/SNOMED data, joined verbatim",
        "generated": date.today().isoformat(),
        "count": len(terms),
        "terms": {t: sorted(c) for t, c in sorted(terms.items())},
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"Wrote {len(terms)} terms -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
