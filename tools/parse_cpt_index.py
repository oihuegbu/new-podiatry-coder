#!/usr/bin/env python3
"""Parse the AMA CPT® Link 'Index file' -> data/codes/cpt_index_terms.json.

The CPT Alphabetic Index is the AUTHORITATIVE procedure term->code map — the CPT
analog of the NCHS ICD-10-CM Alphabetic Index. It is LICENSED AMA content shipped
inside the CPT Link / CPT Standard Data File distribution (the delimited 'Index'
table of index entries). It is NOT publicly downloadable; this tool ingests YOUR
licensed copy — the same license the descriptor data already in cpt_codes.json
came from.

Output is the SAME shape as parse_icd10cm_index.py
({version, source, provenance, terms:{code:[phrases]}}) so
claude_coder.data_access.cpt_index_codes consumes it identically. No medical code
is authored here: every code comes from the AMA file, and a code RANGE is
expanded only to members that actually exist in the authoritative cpt_codes.json.

Usage:
  python tools/parse_cpt_index.py <cpt_link_index_file> [out.json]

The input is delimited (CSV/TSV/pipe auto-detected) with a header row. Columns
are matched BY NAME (case-insensitive) so the tool is robust to the exact CPT
Link column order:
  - MAIN-TERM column      : header contains 'main' | 'term' | 'index' | 'entry'
  - MODIFIER/SUBTERM cols  : header contains 'modif' | 'subterm' | 'sub' | 'level'
  - CODE column           : header contains 'code' | 'range' | 'cpt'
Each row's phrase = main term + its modifiers; the code cell may be a single
code, a range ('28306-28309'), or a list ('28306, 28308') — all handled.

NOTE: validate the detected columns against your actual CPT Link Index file
(print-summary at the end reports the mapping and counts); adjust the header
matchers below if your distribution labels them differently.
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MAIN = ("main", "term", "index", "entry")
_MOD = ("modif", "subterm", "sub", "level", "qualifier")
_CODE = ("code", "range", "cpt")


def _valid_codes() -> set[str]:
    """The authoritative CPT code set (so ranges expand only to real codes)."""
    try:
        from app.core.config import DATA_DIR
        data = json.loads((DATA_DIR / "codes" / "cpt_codes.json").read_text())
    except Exception:
        return set()
    rows = (data if isinstance(data, list)
            else data.get("codes") or next((v for v in data.values()
                                            if isinstance(v, list)), []))
    return {str(r["code"]) for r in rows if isinstance(r, dict) and r.get("code")}


def _sniff(text: str) -> str:
    head = text.splitlines()[0] if text else ""
    for d in ("\t", "|", ","):
        if d in head:
            return d
    return ","


def _match_cols(header: list[str]) -> tuple[int | None, list[int], int | None]:
    main = code = None
    mods: list[int] = []
    for i, h in enumerate(header):
        low = h.strip().lower()
        if code is None and any(k in low for k in _CODE):
            code = i
        elif main is None and any(k in low for k in _MAIN):
            main = i
        elif any(k in low for k in _MOD):
            mods.append(i)
    return main, mods, code


def _expand(cell: str, valid: set[str]) -> list[str]:
    """A code cell -> the real CPT codes it names: a single code, a comma list, or
    a numeric range (endpoints inclusive), each kept only if it exists in the
    authoritative set."""
    out: list[str] = []
    for part in re.split(r"[;,]", cell):
        part = part.strip().upper()
        if not part:
            continue
        m = re.match(r"^(\w{4,5})\s*[-–]\s*(\w{4,5})$", part)
        if m and m.group(1).isdigit() and m.group(2).isdigit():
            lo, hi = int(m.group(1)), int(m.group(2))
            width = len(m.group(1))
            out.extend(str(n).zfill(width) for n in range(lo, hi + 1))
        else:
            out.append(part)
    return [c for c in out if c in valid]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    from app.core.config import DATA_DIR
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        DATA_DIR / "codes" / "cpt_index_terms.json"
    valid = _valid_codes()

    text = src.read_text(encoding="utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text), delimiter=_sniff(text))
    rows = list(reader)
    if not rows:
        print("empty index file"); return 1
    header = rows[0]
    main_i, mod_i, code_i = _match_cols(header)
    if main_i is None or code_i is None:
        print(f"could not locate main-term/code columns in header: {header}")
        return 1

    out: dict[str, set] = defaultdict(set)
    rowcount = 0
    for r in rows[1:]:
        if len(r) <= max(main_i, code_i):
            continue
        main_term = (r[main_i] or "").strip()
        if not main_term:
            continue
        mods = [r[i].strip() for i in mod_i if i < len(r) and r[i].strip()]
        phrase = re.sub(r"\s+", " ", " ".join([main_term, *mods])).strip()
        if not phrase:
            continue
        codes = _expand(r[code_i] or "", valid)
        for c in codes:
            out[c].add(phrase.lower())
        rowcount += 1 if codes else 0

    payload = {
        "version": "",
        "source": src.name,
        "provenance": ("authoritative AMA CPT Alphabetic Index (CPT Link 'Index' "
                       "file), licensed; parsed verbatim, ranges expanded to real "
                       "cpt_codes.json members"),
        "terms": {c: sorted(ps) for c, ps in sorted(out.items())},
    }
    dst.write_text(json.dumps(payload, indent=1))
    phrases = sum(len(v) for v in out.values())
    print(f"columns -> main={header[main_i]!r} code={header[code_i]!r} "
          f"modifiers={[header[i] for i in mod_i]}")
    print(f"{rowcount} index rows -> {len(out)} codes, {phrases} phrases -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
