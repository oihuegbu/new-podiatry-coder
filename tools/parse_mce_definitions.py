"""Parse CMS's "Definitions of Medicare Code Edits" text file into
data/codes/mce_edits.json.

Source: CMS MS-DRG Classifications and Software page — the "Definition of
Medicare Code Edits vXX" zip contains a .txt of every MCE edit list
(https://www.cms.gov/medicare/payment/prospective-payment-systems/
acute-inpatient-pps/ms-drg-classifications-and-software).

Extracted rule families (each a plain ICD-10-CM code list in the source):
  * Age conflict — newborn (age 0), pediatric (0-17), maternity (9-64),
    adult (15-124) diagnosis lists; a billed diagnosis outside its
    category's age range is "clinically and virtually impossible in a
    patient of the stated age" (MCE's own wording).
  * Manifestation code as principal diagnosis — codes that "describe the
    manifestation of an underlying disease, not the disease itself" and
    are not allowed as principal.
  * Unacceptable principal diagnosis — codes describing "a circumstance
    which influences an individual's health status but not a current
    illness or injury" (many Z codes, B95-B97 organism codes, etc.).

Deliberately NOT extracted (documented so the rule-coverage guard can
assert the omission is a decision, not an oversight):
  * Sex conflict — deactivated by CMS for all ICD-10 codes as of 10/01/2024.
  * Questionable admission, procedure-vs-LOS, invalid discharge status,
    non-covered procedures (ICD-10-PCS) — inpatient/UB-04 admission
    concepts with no outpatient professional-claim (CMS-1500) meaning.

Age ranges are parsed from the MCE text's own category definitions, not
hardcoded here.

Usage: python tools/parse_mce_definitions.py <MCE .txt path> [out.json]
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

# An MCE code-list line is "<ICD10 code>\t<short description>"
_CODE_LINE = re.compile(r"^([A-TV-Z][0-9][0-9A-Z]{1,5})\t(.+)$")

# Category age ranges as stated in the MCE text itself, e.g.
# "* Pediatric. Age range is 0-17 years inclusive" / "* Perinatal/Newborn.
# Age 0 years only". The en-dash arrives as \x96 in the latin-1 source.
_RANGE_LINE = re.compile(
    r"\*\s*(?P<name>[A-Za-z/]+)\.\s*Age(?:\s+range\s+is)?\s*"
    r"(?P<lo>\d+)(?:\s*[-\u0096\u2013]\s*(?P<hi>\d+))?\s*years?",
)

# Section header -> (edit_id, category key). Matched on the numbered
# heading lines of the definitions file.
_SECTIONS = {
    "4. Age conflict": "age_conflict",
    "6. Manifestation code as principal diagnosis": "manifestation_not_pdx",
    "9. Unacceptable principal diagnosis": "unacceptable_pdx",
}

# Within the age-conflict section, lettered sub-headings name the category.
_AGE_SUBSECTIONS = {
    "perinatal/newborn diagnoses": "newborn",
    "pediatric diagnoses": "pediatric",
    "maternity diagnoses": "maternity",
    "adult diagnoses": "adult",
}


def parse(text: str) -> dict:
    lines = text.splitlines()
    categories: dict[str, dict] = {}
    codes: dict[str, list] = {"manifestation_not_pdx": [], "unacceptable_pdx": []}
    for k in _AGE_SUBSECTIONS.values():
        codes[f"age_{k}"] = []

    # The MCE text carves out codes that are "acceptable when a secondary
    # diagnosis is also coded" (currently Z51.89) — parsed from the source's
    # own preamble structure, kept separate so the consuming check can apply
    # the carve-out instead of hard-flagging.
    codes["unacceptable_pdx_unless_secondary"] = []
    in_carveout = False

    section = None      # current numbered MCE section
    age_cat = None      # current lettered age sub-list
    numbered = re.compile(r"^\d{1,2}\.\s")

    for raw in lines:
        line = raw.strip()

        # EVERY numbered heading is a section boundary. Tracked sections are
        # entered; untracked ones (questionable admission, non-covered
        # procedure, ...) EXIT the current section — without this, code
        # lists of untracked sections bled into whichever tracked section
        # preceded them (found live: section 8's delivery-outcome codes
        # landed in the manifestation list).
        if numbered.match(line):
            section = next((v for k, v in _SECTIONS.items()
                            if line == k or line.startswith(k + "\t")), None)
            age_cat = None
            continue

        if section == "unacceptable_pdx":
            if 'considered "acceptable" when a secondary diagnosis' in line:
                in_carveout = True
            elif in_carveout and line.startswith("The following pages"):
                in_carveout = False

        if section == "age_conflict":
            m = _RANGE_LINE.search(line)
            if m:
                name = m.group("name").lower()
                key = next((v for k, v in _AGE_SUBSECTIONS.items()
                            if name in k), None)
                if key:
                    lo = int(m.group("lo"))
                    hi = int(m.group("hi")) if m.group("hi") else lo
                    categories[key] = {"min_age": lo, "max_age": hi}
                continue
            # "B. Pediatric diagnoses (age 0 through 17)" -> pediatric
            if re.match(r"^[A-D]\.\s", line):
                bare = re.sub(r"^[A-D]\.\s*", "", line).lower()
                sub = next((v for k, v in _AGE_SUBSECTIONS.items()
                            if bare.startswith(k)), None)
                if sub:
                    age_cat = sub
                    continue

        m = _CODE_LINE.match(raw)
        if not m or section is None:
            continue
        code, desc = m.group(1), m.group(2).strip()
        if section == "age_conflict":
            if age_cat:
                codes[f"age_{age_cat}"].append({"code": code, "description": desc})
        elif section == "unacceptable_pdx" and in_carveout:
            codes["unacceptable_pdx_unless_secondary"].append({"code": code, "description": desc})
        else:
            codes[section].append({"code": code, "description": desc})

    # The definitions file repeats the unacceptable-PDX list in an appendix —
    # dedupe every family while preserving order.
    for family, entries in codes.items():
        seen: set[str] = set()
        deduped = []
        for e in entries:
            if e["code"] not in seen:
                seen.add(e["code"])
                deduped.append(e)
        codes[family] = deduped

    return {
        "description": "CMS Medicare Code Editor (MCE) diagnosis edit lists",
        "source": "CMS Definitions of Medicare Code Edits",
        "source_url": ("https://www.cms.gov/medicare/payment/prospective-payment-systems/"
                       "acute-inpatient-pps/ms-drg-classifications-and-software"),
        "retrieved": date.today().isoformat(),
        "excluded_edits": {
            "sex_conflict": "deactivated by CMS for all ICD-10 codes as of 10/01/2024",
            "questionable_admission": "inpatient admission-justification concept — no CMS-1500 meaning",
            "invalid_discharge_status": "UB-04 field — no CMS-1500 meaning",
            "non_covered_procedure": "ICD-10-PCS (inpatient) — professional claims bill CPT/HCPCS",
            "procedure_inconsistent_with_los": "inpatient length-of-stay concept",
        },
        "age_categories": categories,
        "codes": codes,
        "counts": {k: len(v) for k, v in codes.items()},
    }


def main() -> None:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).resolve().parent.parent / "data" / "codes" / "mce_edits.json")
    text = src.read_text(encoding="latin-1")
    result = parse(text)
    for family, n in result["counts"].items():
        print(f"  {family}: {n} codes")
    print(f"  age categories: {result['age_categories']}")
    required = {k: v for k, v in result["counts"].items()
                if k != "unacceptable_pdx_unless_secondary"}
    if not all(required.values()) or len(result["age_categories"]) < 4:
        raise SystemExit("MCE parse incomplete — refusing to write partial data")
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
