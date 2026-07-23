"""Tests for the finalized-claims registry (tools/claims_registry.py)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.claims_registry import (  # noqa: E402
    append_events, current_view, eligible_for_auto, export_gold,
    extract_claim, ingest, load_events, make_finalized_event,
)

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def _result(doc: str, *, success=True, runs=3, unanimous=True,
            disposition="CLEAN", cpt_mods=("TA",), audited=True) -> dict:
    r = {
        "document_id": doc,
        "success": success,
        "final_disposition": disposition,
        "consistency": {"runs": runs, "unanimous": unanimous},
        "patient_metadata": {"insurance": "Medicare Part B"},
        "icd_codes": [
            {"code": "L60.0", "type": "primary", "description": "Ingrowing nail",
             "confidence": 0.99, "rationale": "should be stripped"},
        ],
        "cpt_codes": [
            {"code": "11750", "modifiers": list(cpt_mods), "units": 1,
             "dx_pointers": [1], "evidence": "should be stripped"},
        ],
        "hcpcs_codes": [],
    }
    if audited:
        # The clinical review is the universal CLEAN gate: eligibility
        # requires an upheld verdict whose fingerprint matches the claim.
        from tools.clinical_auditor import corrections_fingerprint
        r["clinical_audit"] = {"verdict": "upheld",
                               "fingerprint": corrections_fingerprint(r)}
    return r


def main():
    tmp = Path(tempfile.mkdtemp())
    reg = tmp / "claims_registry.jsonl"
    results_dir = tmp / "results"
    results_dir.mkdir()

    print("\n== extract_claim slims to billable fields ==")
    claim = extract_claim(_result("n1"))
    check("rationale stripped", "rationale" not in claim["icd_codes"][0])
    check("evidence stripped", "evidence" not in claim["cpt_codes"][0])
    check("modifiers kept", claim["cpt_codes"][0]["modifiers"] == ["TA"])
    check("disposition kept", claim["final_disposition"] == "CLEAN")

    print("\n== eligibility gate ==")
    check("unanimous CLEAN eligible", eligible_for_auto(_result("n1"))[0])
    check("failed run ineligible",
          not eligible_for_auto(_result("n1", success=False))[0])
    check("single run ineligible",
          not eligible_for_auto(_result("n1", runs=1))[0])
    check("no consistency ineligible",
          not eligible_for_auto({"success": True, "final_disposition": "CLEAN"})[0])
    check("disagreement ineligible",
          not eligible_for_auto(_result("n1", unanimous=False))[0])
    check("REVIEW disposition ineligible",
          not eligible_for_auto(_result("n1", disposition="REVIEW"))[0])
    check("unreviewed claim ineligible (universal audit gate)",
          not eligible_for_auto(_result("n1", audited=False))[0])
    stale = _result("n1")
    stale["clinical_audit"]["fingerprint"] = "0000000000000000"
    check("stale audit fingerprint ineligible",
          not eligible_for_auto(stale)[0])

    print("\n== ingest: idempotence and change detection ==")
    (results_dir / "n1_results.json").write_text(json.dumps(_result("n1")))
    (results_dir / "n2_results.json").write_text(
        json.dumps(_result("n2", unanimous=False)))
    s1 = ingest(results_dir, reg)
    check("first ingest records 1", s1["recorded"] == 1, str(s1))
    check("disagreeing note skipped", s1["skipped"] == 1, str(s1))
    s2 = ingest(results_dir, reg)
    check("re-ingest unchanged", s2["recorded"] == 0 and s2["unchanged"] == 1,
          str(s2))
    # Claim changes (modifier flip) -> a NEW event, history preserved
    (results_dir / "n1_results.json").write_text(
        json.dumps(_result("n1", cpt_mods=("T5",))))
    s3 = ingest(results_dir, reg)
    check("changed claim re-recorded", s3["recorded"] == 1, str(s3))
    events = load_events(reg)
    check("append-only history (2 finalized for n1)",
          sum(1 for e in events if e["document_id"] == "n1") == 2)
    view = current_view(events)
    check("current view has latest claim",
          view["n1"]["claim"]["cpt_codes"][0]["modifiers"] == ["T5"])

    print("\n== human record outranks auto ==")
    human = make_finalized_event(
        "n1", _result("n1", cpt_mods=("TA",)), verification="human",
        verified_by="coder-JD", source="n1_results.json")
    append_events([human], reg)
    view = current_view(load_events(reg))
    check("human is current", view["n1"]["verification"] == "human")
    s4 = ingest(results_dir, reg)  # auto re-ingest must not displace
    check("auto never displaces human", s4["human_protected"] == 1, str(s4))
    view = current_view(load_events(reg))
    check("human claim retained",
          view["n1"]["claim"]["cpt_codes"][0]["modifiers"] == ["TA"])

    print("\n== outcome attaches to current view ==")
    append_events([{"registry_version": 1, "event": "outcome",
                    "document_id": "n1", "recorded_at": "2026-07-18T00:00:00",
                    "status": "paid", "carcs": []}], reg)
    view = current_view(load_events(reg))
    check("outcome attached", view["n1"].get("outcome", {}).get("status") == "paid")

    print("\n== export-gold round-trips through benchmark_ab scoring ==")
    gold_dir = tmp / "gold"
    n = export_gold(gold_dir, reg)
    check("export writes current view", n == 1, f"n={n}")
    exported = json.loads((gold_dir / "n1_results.json").read_text())
    check("gold carries provenance",
          exported["gold_provenance"]["verified_by"] == "coder-JD")
    from tools.benchmark_ab import score_note
    s = score_note(exported, _result("n1", cpt_mods=("TA",)))
    check("benchmark scores exported gold perfectly",
          all(v == 1.0 for v in s.values()), str(s))
    s_bad = score_note(exported, _result("n1", cpt_mods=("T5",)))
    check("benchmark detects modifier drift", s_bad["cpt_modifiers"] < 1.0)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
