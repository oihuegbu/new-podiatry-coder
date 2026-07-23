"""Dry sweep: run ONLY the new deterministic normalization layers against
every processed note's billed arrays, without invoking the LLM pipeline —
redundant RT/LT stripping, unnecessary 59/X removal, E/M level alignment to
consistent MDM axes, and PFS status-P supply suppression. Every line printed
is a change the next full pass would make; audit each for false positives
before rerunning the corpus.

Usage: PYTHONPATH=/app python tools/sweep_normalizations.py [results_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.rag.code_reference import CodeReferenceDB
from app.compliance.datastore.store import ComplianceDataStore
from app.validation.validator import CodingValidator

RESULTS_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "output/results")

CATS = ("redundant_laterality_removed", "laterality_contradiction",
        "unnecessary_separation_modifier_removed", "em_level_corrected",
        "em_level_mismatch", "billability")


def main():
    db = CodeReferenceDB()
    db.load_all()
    store = ComplianceDataStore()
    store.build_or_load()
    total = 0
    for f in sorted(RESULTS_DIR.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        d = json.loads(f.read_text())
        cpt = [dict(c) for c in d.get("cpt_codes") or [] if isinstance(c, dict)]
        hcpcs = [dict(c) for c in d.get("hcpcs_codes") or [] if isinstance(c, dict)]
        v = CodingValidator(db, store)
        v.issues = []
        v._non_billable_codes_to_suppress = set()
        v._check_redundant_laterality(cpt, hcpcs)
        v._check_unnecessary_separation_modifier(cpt)
        v._check_em_level_consistency(cpt)
        # only report NEW suppressions (status P); B/N were already live
        v._check_billability(cpt, hcpcs)
        found = [i for i in v.issues if i.category in CATS
                 and (i.category != "billability" or "'P'" in i.message)]
        if found:
            print(f"\n=== {f.name.replace('_results.json', '')}")
            for i in found:
                total += 1
                print(f"  {i.severity:7s} {i.category:40s} {i.code:8s} {i.message[:120]}")
    print(f"\n==== {total} normalization actions across corpus ====")


if __name__ == "__main__":
    main()
