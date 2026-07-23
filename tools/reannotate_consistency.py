#!/usr/bin/env python3
"""Re-annotate saved results under the advisory-aware consistency rules.

The first --consistency 3 corpus pass ran with a comparator that let SNOMED
variance gate routing; mid-batch the comparator was fixed so advisory arrays
(SNOMED — never on a CMS-1500) are reported but do not gate unanimity. The
in-flight batch kept the old code in memory, so its saved results carry:

  * tier/disposition forced to REVIEW even when the only variance was SNOMED
  * a "Self-consistency check — ..." review reason listing SNOMED concepts
  * needs_review marks on entries flagged by the old rules

Every result embeds its full consistency report, so this is deterministically
reversible: recompute unanimity from the embedded report under the current
rules (app.validation.consistency), and where the note is now unanimous,
restore the ORIGINAL tier/disposition (recorded in the report at compare
time), drop the consistency review reason, and clear consistency-added
needs_review marks. Notes with genuine billing-array disagreements are left
routed to REVIEW, with their review reasons rebuilt to name only billing
codes.

Usage: python tools/reannotate_consistency.py [results_dir]
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.validation.consistency import _is_advisory, annotate_result

DEFAULT_RESULTS = Path("output/results")

_REASON_PREFIX = "Self-consistency check — "


def reannotate(result: dict) -> tuple[dict, str]:
    """Returns (updated result, action) where action is one of
    'no_report', 'unchanged', 'downgraded', 'rebuilt'."""
    report = result.get("consistency")
    if not isinstance(report, dict) or "disagreements" not in report:
        return result, "no_report"

    # Recompute under current rules: advisory arrays never gate. Widen only,
    # never revert: results saved by an older comparator carry advisory:False
    # for entries (supporting_conditions, external-cause ICDs) that are
    # advisory under current PURE rules — those get upgraded. But a stored
    # advisory:True may come from compare_runs' STORE-BASED analysis
    # (claim-inert secondary-ICD flips), which this offline tool cannot
    # recompute — overwriting it with the pure function would silently
    # revert a correct verdict.
    for d in report["disagreements"]:
        d["advisory"] = bool(d.get("advisory")) or _is_advisory(
            d.get("array", ""), str(d.get("code", "")))
    billing = [d for d in report["disagreements"] if not d["advisory"]]
    dispositions = report.get("dispositions") or []
    tiers = report.get("tiers") or []
    unanimous = (not billing and len(set(dispositions)) <= 1
                 and len(set(tiers)) <= 1)
    report["unanimous"] = unanimous

    # Strip everything the old annotation added; rebuild from scratch.
    reasons = [r for r in (result.get("auto_coding_review_reasons") or [])
               if not str(r).startswith(_REASON_PREFIX)]
    result["auto_coding_review_reasons"] = reasons
    for array in ("icd_codes", "supporting_conditions", "cpt_codes",
                  "hcpcs_codes", "snomed_codes"):
        for e in result.get(array) or []:
            if not isinstance(e, dict):
                continue
            rr = str(e.get("review_reason") or "")
            if "self-consistency" in rr:
                parts = [p for p in rr.split(" | ")
                         if "self-consistency" not in p]
                e["review_reason"] = " | ".join(parts) or None
                if not parts:
                    e["needs_review"] = False

    if unanimous:
        # The runs agreed on everything billable: restore the tier and
        # disposition the runs themselves produced (recorded pre-annotation).
        if tiers and len(set(tiers)) == 1:
            result["auto_coding_tier"] = tiers[0]
        if dispositions and len(set(dispositions)) == 1:
            result["final_disposition"] = dispositions[0]
        return result, "downgraded"

    # Genuine billing disagreement: re-apply annotation under current rules.
    result.pop("consistency", None)
    result = annotate_result(result, report)
    return result, "rebuilt"


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    counts: dict[str, int] = {}
    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        original = json.loads(f.read_text())
        updated, action = reannotate(copy.deepcopy(original))
        if action != "no_report" and updated == original:
            action = "unchanged"  # already annotated under current rules
        counts[action] = counts.get(action, 0) + 1
        if action in ("downgraded", "rebuilt"):
            f.write_text(json.dumps(updated, indent=2, default=str))
            tier = updated.get("auto_coding_tier")
            n_billing = len([d for d in updated["consistency"]["disagreements"]
                             if not d.get("advisory")])
            print(f"  [{action:10s}] {f.name:55s} tier={tier:6s} "
                  f"billing-disagreements={n_billing}")
    print(f"\nTotals: {counts}")


if __name__ == "__main__":
    main()
