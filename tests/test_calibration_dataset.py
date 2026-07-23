"""Tests for the calibration dataset exporter (tools/calibration_dataset.py).

Run:  PYTHONPATH=. python tests/test_calibration_dataset.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.calibration_dataset import (  # noqa: E402
    export, extract_features, extract_labels)
from tools.claims_registry import make_finalized_event  # noqa: E402
import tools.claims_registry as reg_mod  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def _result(doc: str, *, unanimous=True, disposition="CLEAN",
            n_billing_disagreements=0) -> dict:
    disagreements = ([{"code": "97597", "advisory": False}]
                     * n_billing_disagreements
                     + [{"code": "12345", "advisory": True}])
    return {
        "document_id": doc,
        "success": True,
        "final_disposition": disposition,
        "processing_time": 300.5,
        "consistency": {"runs": 3, "unanimous": unanimous,
                        "disagreements": disagreements},
        "patient_metadata": {"insurance": "Medicare Part B"},
        "note_sections": {"chief_complaint": "Ingrown toenail"},
        "rag_context": {
            "vision_context": {"note_category": "established_patient_visit",
                               "procedures_performed_today": []},
            "exemplars": {"mode": "shadow", "registry_size": 9,
                          "matches": [{"document_id": "x",
                                       "similarity": 0.42}]},
        },
        "icd_codes": [{"code": "L60.0", "type": "primary",
                       "description": "Ingrowing nail"}],
        "cpt_codes": [
            {"code": "99213", "modifiers": ["25"], "units": 1,
             "description": "Office visit"},
            {"code": "11750", "modifiers": ["TA"], "units": 1,
             "description": "Excision of nail and nail matrix"}],
        "hcpcs_codes": [],
        "validation_issues": [
            {"severity": "WARNING", "category": "x",
             "message": "AUTO-CORRECTED: something"},
            {"severity": "ERROR", "category": "y", "message": "bad"}],
        "auto_coding_confidence": 0.93,
        "api_usage": {"total_tokens": 21000},
    }


def main():
    tmp = Path(tempfile.mkdtemp())
    reg = tmp / "claims_registry.jsonl"
    results_dir = tmp / "results"
    results_dir.mkdir()
    out = tmp / "calibration_dataset.jsonl"
    # point the registry module at the temp ledger
    reg_mod.REGISTRY_PATH = reg

    print("\n== feature extraction ==")
    f = extract_features(_result("n1"))
    check("counts", f["n_icd"] == 1 and f["n_cpt"] == 2 and f["n_hcpcs"] == 0)
    check("modifier count", f["n_modifiers"] == 2)
    check("E/M code found", f["em_code"] == "99213")
    check("validation split",
          f["n_validation_errors"] == 1 and f["n_validation_warnings"] == 1
          and f["n_auto_corrections"] == 1)
    check("exemplar coverage", f["n_exemplar_neighbors"] == 1
          and f["max_exemplar_similarity"] == 0.42)
    check("router verdict not leaked into features",
          "needs_review" not in f and "auto_coding_tier" not in f)

    print("\n== label extraction ==")
    lab = extract_labels(_result("n1"), {})
    check("unanimous+CLEAN → needs_review False", lab["needs_review"] is False)
    lab = extract_labels(_result("n2", unanimous=False), {})
    check("disagreement → needs_review True", lab["needs_review"] is True)
    lab = extract_labels(_result("n3", disposition="REVIEW"), {})
    check("REVIEW disposition → needs_review True", lab["needs_review"] is True)
    check("no human verdict → human_corrected unknown (None)",
          lab["human_corrected"] is None)
    # human recorded the SAME claim → not corrected
    r4 = _result("n4")
    ev = make_finalized_event("n4", r4, verification="human",
                              verified_by="coder", source="t")
    lab = extract_labels(r4, {"n4": ev})
    check("human verified identical claim → human_corrected False",
          lab["human_corrected"] is False)
    # human recorded a DIFFERENT claim → corrected
    r5 = _result("n5")
    changed = json.loads(json.dumps(r5))
    changed["cpt_codes"][0]["modifiers"] = []
    ev = make_finalized_event("n5", changed, verification="human",
                              verified_by="coder", source="t")
    lab = extract_labels(r5, {"n5": ev})
    check("human changed the claim → human_corrected True",
          lab["human_corrected"] is True)

    print("\n== export idempotence ==")
    for doc in ("a1", "a2"):
        (results_dir / f"{doc}_results.json").write_text(
            json.dumps(_result(doc), default=str))
    stats = export(results_dir, out)
    check("first export: 2 new", stats["new"] == 2 and stats["total"] == 2)
    stats = export(results_dir, out)
    check("re-export unchanged", stats["unchanged"] == 2 and stats["new"] == 0)
    (results_dir / "a2_results.json").write_text(
        json.dumps(_result("a2", unanimous=False), default=str))
    stats = export(results_dir, out)
    check("changed result updates its row in place",
          stats["updated"] == 1 and stats["total"] == 2)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    check("one row per document", len(rows) == 2)
    check("updated label visible",
          next(r for r in rows if r["document_id"] == "a2")
          ["labels"]["needs_review"] is True)

    print(f"\n{'=' * 40}\nRESULT: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
