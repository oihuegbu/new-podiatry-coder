#!/usr/bin/env python3
"""Calibration dataset: labeled training data for a learned confidence model.

The end state is a model that predicts, per claim, "does this need a human?"
better than the current rule-based routing (consistency unanimity + scrub
disposition). Training it needs labeled examples at scale; this tool is the
wiring that accumulates them from day one, so the data exists the moment the
volume does.

One row per processed note, joining three sources:
  features  from the note's result file — claim shape (code/modifier counts,
            E/M level), pipeline signals (validation error/warning/
            auto-correction counts, consistency disagreements, auto-coding
            tier and confidence), exemplar neighbor coverage, cost/latency;
  labels    from the routing decision and the finalized-claims registry —
            `needs_review` (the rule-based router's verdict: not unanimous
            or not CLEAN), `human_corrected` (a coder's recorded claim
            actually differs from the pipeline's — the ground-truth signal
            the confidence model should learn to anticipate), and payer
            adjudication (`outcome_status`, CARCs) when attached.

Rows are keyed by document and appended idempotently: re-exporting after a
new batch updates existing rows in place (labels sharpen as human verdicts
and payer outcomes arrive) and appends new ones. Output is JSONL next to
the registry (data/registry/calibration_dataset.jsonl) — same durability
story, same bind mount.

LEAKAGE WARNING for model training: `needs_review` is DEFINED by the
consistency verdict, so the consistency feature group (consistency_runs,
n_billing_disagreements, n_advisory_disagreements) determines it by
construction — a model trained on all features against `needs_review`
just relearns the router. Legitimate targets:
  * needs_review WITHOUT the consistency features — "predict routing from
    a single run" (the model that could replace the 3x consistency cost);
  * human_corrected / outcome_status with ALL features — "predict whether
    a human or payer will find a problem" (the model that could beat the
    router). Consistency signals are honest predictors for these.

Usage:
  python tools/calibration_dataset.py export [results_dir] [--out FILE]
  python tools/calibration_dataset.py stats  [--out FILE]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "output" / "results"
DEFAULT_OUT = ROOT / "data" / "registry" / "calibration_dataset.jsonl"

DATASET_VERSION = 1


def _n(x) -> int:
    return len(x) if isinstance(x, list) else 0


def extract_features(result: dict) -> dict:
    """Model inputs only — nothing here may leak the label (the router's
    verdict fields are recorded as labels, not features)."""
    cpt = result.get("cpt_codes") or []
    icd = result.get("icd_codes") or []
    hcpcs = result.get("hcpcs_codes") or []
    issues = result.get("validation_issues") or []
    cons = result.get("consistency") or {}
    disagreements = cons.get("disagreements") or []
    rc = result.get("rag_context") or {}
    vc = rc.get("vision_context") or {}
    ex = rc.get("exemplars") or {}
    em = next((e for e in cpt
               if str(e.get("code", "")).startswith(("992", "993"))), None)
    sims = [m.get("similarity", 0) for m in ex.get("matches") or []]
    return {
        "note_category": vc.get("note_category") or "",
        "payer": (result.get("patient_metadata") or {}).get("insurance") or "",
        "n_icd": _n(icd), "n_cpt": _n(cpt), "n_hcpcs": _n(hcpcs),
        "n_modifiers": sum(_n(e.get("modifiers")) for e in cpt + hcpcs),
        "em_code": (em or {}).get("code") or "",
        "n_validation_errors": sum(
            1 for i in issues if i.get("severity") == "ERROR"),
        "n_validation_warnings": sum(
            1 for i in issues if i.get("severity") == "WARNING"),
        "n_auto_corrections": sum(
            1 for i in issues if "AUTO-CORRECTED" in (i.get("message") or "")),
        "consistency_runs": cons.get("runs") or 0,
        "n_billing_disagreements": sum(
            1 for d in disagreements if not d.get("advisory")),
        "n_advisory_disagreements": sum(
            1 for d in disagreements if d.get("advisory")),
        "auto_coding_confidence": result.get("auto_coding_confidence"),
        "n_exemplar_neighbors": len(sims),
        "max_exemplar_similarity": max(sims) if sims else 0.0,
        "processing_time_s": result.get("processing_time"),
        "total_tokens": (result.get("api_usage") or {}).get("total_tokens"),
    }


def extract_labels(result: dict, registry_view: dict) -> dict:
    from tools.claims_registry import extract_claim, _claim_key
    doc = str(result.get("document_id", ""))
    cons = result.get("consistency") or {}
    reg = registry_view.get(doc)
    human_corrected = None  # unknown until a human verdict exists
    if reg and reg.get("verification") == "human":
        human_corrected = (_claim_key(reg["claim"])
                           != _claim_key(extract_claim(result)))
    outcome = (reg or {}).get("outcome") or {}
    return {
        # the rule-based router's verdict — what the learned model must beat
        "needs_review": (not cons.get("unanimous", False)
                         or (result.get("final_disposition") or "").upper()
                         != "CLEAN"),
        # the note needed judgment beyond the deterministic stack: the
        # expert-coder adjudicator settled it (a distinct outcome class
        # between clean-first-pass and human-corrected)
        "adjudicated": bool(result.get("adjudication")),
        "human_corrected": human_corrected,
        "outcome_status": outcome.get("status"),
        "outcome_carcs": outcome.get("carcs") or [],
    }


def export(results_dir: Path, out_path: Path) -> dict:
    from tools.claims_registry import load_events, current_view
    view = current_view(load_events())
    rows: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["document_id"]] = r

    stats = {"new": 0, "updated": 0, "unchanged": 0, "skipped": 0}
    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        result = json.loads(f.read_text())
        doc = str(result.get("document_id")
                  or f.stem.removesuffix("_results"))
        if not result.get("success"):
            stats["skipped"] += 1
            continue
        row = {
            "dataset_version": DATASET_VERSION,
            "document_id": doc,
            "features": extract_features(result),
            "labels": extract_labels(result, view),
        }
        old = rows.get(doc)
        if old is None:
            stats["new"] += 1
        elif json.dumps(old, sort_keys=True) == json.dumps(
                row, sort_keys=True, default=str):
            stats["unchanged"] += 1
            continue
        else:
            stats["updated"] += 1
        rows[doc] = row

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for doc in sorted(rows):
            fh.write(json.dumps(rows[doc], sort_keys=True, default=str) + "\n")
    stats["total"] = len(rows)
    return stats


def print_stats(out_path: Path) -> None:
    if not out_path.exists():
        print("No dataset yet — run 'export' after a batch.")
        return
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    n = len(rows)
    review = sum(1 for r in rows if r["labels"]["needs_review"])
    human_known = [r for r in rows if r["labels"]["human_corrected"] is not None]
    corrected = sum(1 for r in human_known if r["labels"]["human_corrected"])
    adjudicated = sum(1 for r in rows if r["labels"]["outcome_status"])
    print(f"{n} labeled note(s)")
    print(f"  routed to review: {review} ({review / n:.0%})" if n else "")
    print(f"  human verdict known: {len(human_known)} "
          f"(corrected: {corrected})")
    print(f"  payer-adjudicated: {adjudicated}")
    print("A learned confidence model becomes worth training once "
          "human-verdict rows reach the hundreds; until then this dataset "
          "just accumulates.")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("export", help="(re)build rows from a results dir")
    s.add_argument("results_dir", nargs="?", default=str(DEFAULT_RESULTS))
    s.add_argument("--out", default=str(DEFAULT_OUT))
    s = sub.add_parser("stats", help="label coverage of the dataset")
    s.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    if args.cmd == "export":
        stats = export(Path(args.results_dir), Path(args.out))
        print(f"Calibration dataset: {stats['total']} row(s) "
              f"({stats['new']} new, {stats['updated']} updated, "
              f"{stats['unchanged']} unchanged, {stats['skipped']} skipped) "
              f"-> {args.out}")
    else:
        print_stats(Path(args.out))


if __name__ == "__main__":
    main()
