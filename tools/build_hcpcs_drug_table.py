#!/usr/bin/env python3
"""Prepare the HCPCS drug/biological index -> data/codes/hcpcs_drug_table.json.

The authoritative CMS "Table of Drugs and Biologicals" maps a drug NAME to its
HCPCS code and per-unit DOSAGE (e.g. a drug name with a 'per N mg' dose -> the
code, unit=15 mg). CMS HCPCS Level II is PUBLIC-DOMAIN / free (unlike AMA CPT), so
this is fully automatable.

Two inputs, both automated:
  • PRIMARY (always): the authoritative HCPCS Level II descriptors already
    ingested in data/codes/hcpcs_codes.json. A drug/biological code is detected by
    DESCRIPTOR GRAMMAR — its billing unit is a SUBSTANCE AMOUNT ('per 15 mg',
    ', 1 mg', 'per 100 units') — never by a J/A/Q code prefix (no hardcoded codes).
    The generic name and per-unit dose are parsed out of the descriptor.
  • OPTIONAL enrichment: an external CMS Table-of-Drugs / ASP file (--table-file or
    --table-url), a delimited drug-name -> code table that adds BRAND/trade-name
    synonyms the generic descriptor lacks. Auto-fetched when a URL is configured;
    simply skipped otherwise (the primary output still stands).

Output mirrors the other index files ({version, source, provenance, terms:{code:
[names]}}) plus a units map {code:{amount, unit}} for correct unit derivation
(documented dose / per-unit dose). No medical code is authored here.

Usage:
  python tools/build_hcpcs_drug_table.py
  python tools/build_hcpcs_drug_table.py --table-file cms_table_of_drugs.csv
  python tools/build_hcpcs_drug_table.py --table-url https://…  [--out path.json]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Substance billing units (mass / volume / activity). Generic dosing vocabulary —
# the same kind of non-code lexicon as left/right, NOT a medical code list. A code
# whose billing unit is one of these is a drug/biological; 'each'/'week'/'kit' are
# supplies and are deliberately excluded.
_UNIT = (r"mg|milligram|mcg|microgram|ug|µg|g|gram|ml|milliliter|cc|units?|iu|"
         r"meq|mmol|mci|gbq|billion units|million units")
_DOSE = re.compile(rf"(?:,?\s*per\s+|,\s+)(\d+(?:\.\d+)?)\s*({_UNIT})\b", re.I)
# leading administration/route boilerplate to strip from the generic name
_ROUTE = re.compile(r"^(injection|infusion|inhalation solution|inhalation|"
                    r"oral|intravenous|intrathecal|implant|supply of)\b[,:]?\s*", re.I)


def _load_hcpcs() -> dict:
    from app.core.config import DATA_DIR
    data = json.loads((DATA_DIR / "codes" / "hcpcs_codes.json").read_text())
    rows = (data if isinstance(data, list)
            else data.get("codes") or next((v for v in data.values()
                                            if isinstance(v, list)), []))
    return {str(r["code"]): r for r in rows if isinstance(r, dict) and r.get("code")}


def _dose_of(descriptor: str):
    """(amount, unit) for the LAST substance-dose clause in a descriptor, else None."""
    matches = list(_DOSE.finditer(descriptor))
    if not matches:
        return None
    m = matches[-1]
    return float(m.group(1)), m.group(2).lower()


def _name_of(descriptor: str) -> str:
    """The drug name: descriptor minus the dose clause and leading route words."""
    name = _DOSE.sub("", descriptor).strip(" ,;")
    name = _ROUTE.sub("", name).strip(" ,;")
    return re.sub(r"\s+", " ", name)


def _fetch_table(url: str | None, path: str | None) -> list[tuple[str, str]]:
    """(name, code) rows from an external Table-of-Drugs/ASP file, if provided."""
    blob = None
    if path:
        blob = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    elif url:
        req = urllib.request.Request(url, headers={"User-Agent": "coder/1.0"})
        blob = urllib.request.urlopen(req, timeout=180).read().decode(
            "utf-8-sig", errors="replace")
    if not blob:
        return []
    delim = "\t" if "\t" in blob.splitlines()[0] else ("|" if "|" in blob.splitlines()[0] else ",")
    rows = list(csv.reader(io.StringIO(blob), delimiter=delim))
    if not rows:
        return []
    header = [h.strip().lower() for h in rows[0]]
    name_i = next((i for i, h in enumerate(header) if "name" in h or "drug" in h), 0)
    code_i = next((i for i, h in enumerate(header)
                   if "code" in h or "hcpcs" in h), len(header) - 1)
    out = []
    for r in rows[1:]:
        if len(r) > max(name_i, code_i) and r[name_i].strip() and r[code_i].strip():
            out.append((r[name_i].strip(), r[code_i].strip().upper()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-file", help="local external Table-of-Drugs/ASP file (brand names)")
    ap.add_argument("--table-url", help="URL of an external Table-of-Drugs/ASP file")
    ap.add_argument("--out", help="output path")
    args = ap.parse_args()
    from app.core.config import DATA_DIR

    hcpcs = _load_hcpcs()
    terms: dict[str, set] = defaultdict(set)
    units: dict[str, dict] = {}
    drug_codes = 0
    for code, rec in hcpcs.items():
        desc = str(rec.get("long_description") or rec.get("description") or "")
        dose = _dose_of(desc)
        if not dose:
            continue                      # billing unit is not a substance amount -> not a drug
        name = _name_of(desc)
        if not name:
            continue
        terms[code].add(name.lower())
        units[code] = {"amount": dose[0], "unit": dose[1]}
        drug_codes += 1

    # optional brand/trade-name enrichment from an external CMS table
    enriched = 0
    valid = set(hcpcs)
    for name, code in _fetch_table(args.table_url, args.table_file):
        if code in valid:
            terms[code].add(name.lower())
            enriched += 1

    out = Path(args.out) if args.out else DATA_DIR / "codes" / "hcpcs_drug_table.json"
    payload = {
        "version": "",
        "source": "hcpcs_codes.json (authoritative CMS HCPCS Level II descriptors)"
                  + (" + external table" if enriched else ""),
        "provenance": ("CMS HCPCS Level II drug/biological codes (public domain); "
                       "generic name + per-unit dose parsed from the authoritative "
                       "descriptor; brand synonyms from the external table when supplied"),
        "terms": {c: sorted(v) for c, v in sorted(terms.items())},
        "units": units,
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"{drug_codes} drug/biological codes ({enriched} external brand names) "
          f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
