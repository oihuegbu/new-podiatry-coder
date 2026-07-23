"""Tests for verified-claim exemplar retrieval (app/coding/exemplars.py).

Pure-python: fabricated registry files in a temp dir, no LLM calls, no
reference data. Run:  PYTHONPATH=. python tests/test_exemplars.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.coding import exemplars  # noqa: E402
from tools.claims_registry import make_finalized_event, make_fingerprint  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def _result(doc: str, chief: str, assessment: str, category: str,
            cpt: list[dict], icd: list[dict]) -> dict:
    return {
        "document_id": doc,
        "success": True,
        "final_disposition": "CLEAN",
        "consistency": {"runs": 3, "unanimous": True},
        "patient_metadata": {"insurance": "Medicare Part B"},
        "note_sections": {"chief_complaint": chief,
                          "assessment_diagnoses": assessment},
        "rag_context": {"vision_context": {
            "note_category": category,
            "procedures_performed_today": []}},
        "icd_codes": icd,
        "cpt_codes": cpt,
        "hcpcs_codes": [],
    }


def _write_registry(path: Path, results: list[dict]):
    with open(path, "w") as f:
        for r in results:
            ev = make_finalized_event(r["document_id"], r, verification="auto",
                                      verified_by="test", source="test")
            f.write(json.dumps(ev, default=str) + "\n")


ULCER = _result(
    "100_ulcer_note", "Non-healing diabetic ulcer left hallux",
    "Osteomyelitis distal phalanx; diabetic foot ulcer with exposed bone",
    "established_patient_visit",
    cpt=[{"code": "99215", "modifiers": ["25"], "units": 1,
          "description": "Office/outpatient visit, established, high MDM"},
         {"code": "97597", "modifiers": ["LT"], "units": 1,
          "description": "Debridement open wound"}],
    icd=[{"code": "E11.621", "type": "primary",
          "description": "Type 2 diabetes mellitus with foot ulcer"}])

NAIL = _result(
    "200_nail_note", "Painful ingrown toenail right hallux",
    "Onychocryptosis right hallux, incurvated nail border",
    "established_patient_visit",
    cpt=[{"code": "11750", "modifiers": ["TA"], "units": 1,
          "description": "Excision of nail and nail matrix"}],
    icd=[{"code": "L60.0", "type": "primary",
          "description": "Ingrowing nail"}])

QUERY_SECTIONS = {
    "chief_complaint": "Non-healing wound left hallux in diabetic patient",
    "assessment_diagnoses": "Osteomyelitis distal phalanx left hallux; "
                            "diabetic foot ulcer with exposure of bone",
    "plan": "Deep wound debridement performed; bone biopsy obtained",
}


def main():
    tmp = Path(tempfile.mkdtemp())
    reg = tmp / "claims_registry.jsonl"
    _write_registry(reg, [ULCER, NAIL])

    print("\n== fingerprint ==")
    fp = make_fingerprint(ULCER)
    check("category captured", fp["note_category"] == "established_patient_visit")
    check("assessment captured", "Osteomyelitis" in fp["assessment"])

    print("\n== mode resolution ==")
    exemplars.EXEMPLAR_MODE = "auto"
    check("auto below threshold → shadow", exemplars.resolve_mode(2) == "shadow")
    check("auto at threshold → shadow",
          exemplars.resolve_mode(exemplars.EXEMPLAR_LIVE_THRESHOLD) == "shadow")
    check("auto above threshold → live",
          exemplars.resolve_mode(exemplars.EXEMPLAR_LIVE_THRESHOLD + 1) == "live")
    exemplars.EXEMPLAR_MODE = "off"
    check("explicit off wins", exemplars.resolve_mode(10_000) == "off")
    exemplars.EXEMPLAR_MODE = "live"
    check("explicit live wins", exemplars.resolve_mode(1) == "live")
    exemplars.EXEMPLAR_MODE = "auto"

    print("\n== shadow retrieval (default below threshold) ==")
    block, info = exemplars.for_note(
        "300_new_note", "established_patient_visit", QUERY_SECTIONS,
        registry_path=reg)
    check("mode is shadow", info["mode"] == "shadow")
    check("prompt block empty in shadow", block == "")
    check("similar ulcer note matched",
          any(m["document_id"] == "100_ulcer_note" for m in info["matches"]))
    check("dissimilar nail note not matched",
          not any(m["document_id"] == "200_nail_note" for m in info["matches"]))
    check("similarity recorded",
          all(0 < m["similarity"] <= 1 for m in info["matches"]))

    print("\n== self-exclusion ==")
    block, info = exemplars.for_note(
        "100_ulcer_note", "established_patient_visit", QUERY_SECTIONS,
        registry_path=reg)
    check("a note is never its own exemplar",
          not any(m["document_id"] == "100_ulcer_note" for m in info["matches"]))

    print("\n== live injection ==")
    exemplars.EXEMPLAR_MODE = "live"
    block, info = exemplars.for_note(
        "300_new_note", "established_patient_visit", QUERY_SECTIONS,
        registry_path=reg)
    check("mode is live", info["mode"] == "live")
    check("block rendered", "VERIFIED SIMILAR ENCOUNTERS" in block)
    check("verified codes present", "99215" in block and "E11.621" in block)
    check("modifiers rendered", "-25" in block)
    check("framed as example not lookup", "NOT lookups" in block)
    check("dissimilar claim's codes absent", "11750" not in block)
    exemplars.EXEMPLAR_MODE = "auto"

    print("\n== degenerate inputs ==")
    block, info = exemplars.for_note(
        "400_note", "", {"chief_complaint": ""}, registry_path=reg)
    check("empty note terms → no matches, no crash",
          block == "" and info["matches"] == [])
    block, info = exemplars.for_note(
        "500_note", "established_patient_visit", QUERY_SECTIONS,
        registry_path=tmp / "missing.jsonl")
    check("missing registry → shadow with empty matches",
          info["mode"] == "shadow" and info["matches"] == [])

    print("\n== pre-fingerprint events fall back to claim descriptions ==")
    old_ev = make_finalized_event("600_old", ULCER, verification="auto",
                                  verified_by="test", source="test")
    del old_ev["fingerprint"]
    reg2 = tmp / "reg2.jsonl"
    reg2.write_text(json.dumps(old_ev, default=str) + "\n")
    # Descriptions alone carry less signal than a full fingerprint, so test
    # the fallback mechanism below the production similarity bar.
    saved_sim = exemplars.EXEMPLAR_MIN_SIM
    exemplars.EXEMPLAR_MIN_SIM = 0.05
    block, info = exemplars.for_note(
        "700_new", "established_patient_visit", QUERY_SECTIONS,
        registry_path=reg2)
    exemplars.EXEMPLAR_MIN_SIM = saved_sim
    check("old event still retrievable via code descriptions",
          any(m["document_id"] == "600_old" for m in info["matches"]))

    print(f"\n{'=' * 40}\nRESULT: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
