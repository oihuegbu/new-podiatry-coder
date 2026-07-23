"""Dry sweep: run ONLY the best-code-vs-note layers against every processed
note (PDF text + the codes its result actually billed) without invoking the
LLM pipeline. Measures the new checks' live precision across the corpus:
every finding printed here is either a real coding error or a false positive
to tune away.

Usage: PYTHONPATH=/app python tools/sweep_new_layers.py [results_dir] [notes_dir]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pdfplumber

from app.rag.code_reference import CodeReferenceDB
from app.compliance.datastore.store import ComplianceDataStore
from app.validation.validator import CodingValidator

RESULTS_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "output/results")
NOTES_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else "doctors_notes")

CATS = ("sibling_matches_note_better", "billed_attribute_undocumented",
        "descriptor_condition_undocumented", "descriptor_qualifier_contradicted",
        "dedicated_code_unbilled", "seventh_char_sequela")


def note_text(stem: str) -> str | None:
    pdf = NOTES_DIR / f"{stem}.pdf"
    if not pdf.exists():
        return None
    with pdfplumber.open(pdf) as doc:
        return re.sub(r"\s+", " ", " ".join((p.extract_text() or "") for p in doc.pages))


def main():
    db = CodeReferenceDB()
    db.load_all()
    store = ComplianceDataStore()
    store.build_or_load()
    total = 0
    for f in sorted(RESULTS_DIR.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        stem = f.name.replace("_results.json", "")
        txt = note_text(stem)
        if not txt:
            continue
        d = json.loads(f.read_text())
        icd = [dict(c) for c in d.get("icd_codes") or [] if isinstance(c, dict)]
        cpt = [dict(c) for c in d.get("cpt_codes") or [] if isinstance(c, dict)]
        hcpcs = [dict(c) for c in d.get("hcpcs_codes") or [] if isinstance(c, dict)]
        v = CodingValidator(db, store)
        v.issues = []
        v._check_cpt_descriptor_evidence(cpt, txt)
        v._check_unbilled_descriptor_match(cpt, hcpcs, txt)
        v._check_icd_sibling_descriptor(icd, txt)
        v._check_injury_seventh_char(icd)
        found = [i for i in v.issues if i.category in CATS]
        if found:
            print(f"\n=== {stem}")
            for i in found:
                total += 1
                print(f"  {i.severity:7s} {i.category:32s} {i.code:8s} {i.message[:130]}")
    print(f"\n==== {total} findings across corpus ====")


if __name__ == "__main__":
    main()
