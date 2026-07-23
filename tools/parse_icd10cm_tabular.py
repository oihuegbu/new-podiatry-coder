"""Generate data/codes/icd10cm_instructional_notes.json from CDC/NCHS's
official ICD-10-CM Tabular List XML (icd10cm-tabular-YYYY.xml, from
https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/).

Extracts EVERY per-code instructional note family the Tabular defines:
  inclusionTerm, includes, excludes1, excludes2, codeFirst,
  useAdditionalCode, codeAlso — each with a parallel <family>_code_refs
  list of the ICD-10 codes cited in the note text's parentheticals.

This file is the single upstream for the compliance store's ICD-10
relationship tables (excludes1, includes-subsumption, code-first,
use-additional-code, code-also, tabular descriptions). Keeping the
generator in-repo means a new fiscal year's XML regenerates everything
with one command — the original extraction was a one-off script that
didn't survive, which is how codeAlso stayed unextracted for months.

Usage: python tools/parse_icd10cm_tabular.py <icd10cm-tabular-YYYY.xml> [out.json]
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_FAMILIES = ("inclusionTerm", "includes", "excludes1", "excludes2",
             "codeFirst", "useAdditionalCode", "codeAlso")

# ICD-10-CM code token as cited in note parentheticals, e.g. (Z99.2),
# (I70.2-), (E08-E13). Ranges are expanded to their two endpoints.
# First char is any A-Z: U codes are real ICD-10-CM (U07.1 COVID-19).
_REF_RE = re.compile(r"[A-Z][0-9][0-9A-Z]{0,5}(?:\.[0-9A-Z]{1,4})?-?")


def _code_refs(lines: list[str], valid_names: set[str]) -> list[str]:
    """Codes cited by a note. Parenthesized citations — the Tabular's usual
    form, '(Z99.2)' — are always taken. Codes are ALSO cited bare in running
    text (Includes notes especially: 'any condition in I50.- or I51.4-I51.7
    due to hypertension'), so bare tokens are taken too, but only when they
    are unambiguous code references: dotted (E72.4), family-dashed (I50.-),
    or a bare category that actually exists in this Tabular — which rejects
    free-text lookalikes such as 'vitamin B12' (no B12 category exists)."""
    refs: list[str] = []

    def _add(tok: str) -> None:
        tok = tok.rstrip("-")   # "T81.44-" cites the T81.44 family
        if tok and tok not in refs:
            refs.append(tok)

    for ln in lines:
        parens = re.findall(r"\(([^()]*)\)", ln)
        for paren in parens:
            for tok in _REF_RE.findall(paren):
                _add(tok)
        bare = re.sub(r"\([^()]*\)", " ", ln)   # strip paren text already handled
        for tok in _REF_RE.findall(bare):
            if "." in tok or tok.endswith("-") or tok in valid_names:
                _add(tok)
    return refs


def _diag_entries(elem: ET.Element, out: dict, valid_names: set[str]) -> None:
    for diag in elem.findall("diag"):
        name = (diag.findtext("name") or "").strip()
        desc = (diag.findtext("desc") or "").strip()
        if name:
            entry: dict = {"code": name, "description": desc}
            for family in _FAMILIES:
                fam_el = diag.find(family)
                if fam_el is None:
                    continue
                lines = [(n.text or "").strip() for n in fam_el.findall("note")]
                lines = [ln for ln in lines if ln]
                if not lines:
                    continue
                entry[family] = lines
                refs = _code_refs(lines, valid_names)
                if refs:
                    entry[f"{family}_code_refs"] = refs
            out[name] = entry
        _diag_entries(diag, out, valid_names)  # nested subcodes


def parse(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # first pass: every code/category name in this Tabular, used to validate
    # bare (non-parenthesized) code citations in note text
    valid_names = {(d.findtext("name") or "").strip()
                   for d in root.iter("diag")}
    valid_names.discard("")
    codes: dict = {}
    for chapter in root.iter("chapter"):
        for section in chapter.findall("section"):
            _diag_entries(section, codes, valid_names)
        _diag_entries(chapter, codes, valid_names)
    version = (root.findtext("version") or "").strip()
    return {
        "source": f"CDC/NCHS ICD-10-CM Tabular List FY{version or 'unknown'}",
        "source_url": ("https://ftp.cdc.gov/pub/health_statistics/nchs/"
                       "publications/ICD10CM/"),
        "description": ("ICD-10-CM Tabular List instructional notes "
                        "(Excludes1/Excludes2/Includes/codeFirst/"
                        "useAdditionalCode/codeAlso), parsed from CDC/NCHS's "
                        "official icd10cm-tabular XML by "
                        "tools/parse_icd10cm_tabular.py."),
        "codes": codes,
    }


def main() -> None:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).resolve().parent.parent / "data" / "codes"
        / "icd10cm_instructional_notes.json")
    result = parse(src)
    codes = result["codes"]
    counts = {f: sum(1 for e in codes.values() if f in e) for f in _FAMILIES}
    print(f"  {len(codes)} codes with notes; per-family: {counts}")
    if not codes or not all(counts[f] for f in ("excludes1", "codeFirst",
                                                "useAdditionalCode", "codeAlso")):
        raise SystemExit("tabular parse incomplete — refusing to write")
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
