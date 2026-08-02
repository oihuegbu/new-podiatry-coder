#!/usr/bin/env python3
"""Standalone loop finalization — resume the unanimity loop's ending when
the loop process itself was killed mid-finalization.

The unanimity loop defers routing, adjudication, the clinical audit and
the registry ingest to its finalization block. If the loop's process dies
before that block completes (measured live: the SSH session driving the
`docker compose run --rm` client dropped, killing the container while the
last holdout's adjudication was polling a message batch), the batch is
left in a half-finalized state: some holdouts adjudicated and recorded,
others stranded "deferred", and the final audit/ingest never run.

This tool re-runs exactly the loop's finalization steps, idempotently
(each step skips what is already settled — fingerprints for the audit,
tier/claim comparison for the registry, class keys for the triage scan):

  1. adjudicate every still-non-unanimous note in scope
  2. route whatever remains non-unanimous to human review
  3. clinical-correctness review of the whole scope (the universal
     CLEAN gate — promotes held claims, routes disputes)
  4. registry ingest (records what the audit just promoted)
  5. triage scan (enqueues audit disputes/findings for actuation)
  6. rebuild all_results.json to match disk

Usage (inside the app container):
  python tools/finalize_scope.py [results_dir] [--start-at S] [--end-at E]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_RESULTS = ROOT / "output" / "results"


def _scope(results_dir: Path, start_at: str, end_at: str) -> list[str]:
    docs = []
    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        doc = f.stem.removesuffix("_results")
        if start_at and doc < start_at:
            continue
        if end_at and not doc.startswith(end_at) and doc > end_at:
            continue
        docs.append(doc)
    return docs


def _non_unanimous(results_dir: Path, docs: list[str]) -> list[str]:
    out = []
    for doc in docs:
        f = results_dir / f"{doc}_results.json"
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        if not (r.get("consistency") or {}).get("unanimous"):
            out.append(doc)
    return out


def _audit_only(results_dir: Path, scope: list[str]) -> None:
    try:
        from tools.clinical_auditor import audit_batch
        caud = audit_batch(results_dir, docs=scope)
        print(f"\nClinical audit (final): {caud['audited']} audited "
              f"({caud['upheld']} upheld, {caud['disputed']} "
              f"disputed), {caud['skipped']} unchanged/skipped",
              flush=True)
        for doc, msg in sorted(caud["docs"].items()):
            if "disputed" in msg:
                print(f"  [clinical-audit] {doc}: {msg}", flush=True)
    except Exception as exc:
        print(f"Clinical audit (final) failed ({exc})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", nargs="?", default=str(DEFAULT_RESULTS))
    p.add_argument("--start-at", default="")
    p.add_argument("--end-at", default="")
    args = p.parse_args()
    results_dir = Path(args.results_dir)

    scope = _scope(results_dir, args.start_at, args.end_at)
    print(f"Scope: {len(scope)} note(s)", flush=True)

    holdouts = _non_unanimous(results_dir, scope)
    if holdouts and os.getenv("CODER_ADJUDICATION", "1") == "1":
        try:
            from tools.coder_adjudicator import adjudicate
            print(f"\nCoder adjudication: {len(holdouts)} "
                  f"judgment-shaped holdout(s)", flush=True)
            astats = adjudicate(results_dir, docs=holdouts)
            print(json.dumps(astats, indent=2, default=str), flush=True)
            holdouts = _non_unanimous(results_dir, holdouts)
        except Exception as exc:
            print(f"Coder adjudication failed ({exc}) — routing all "
                  f"holdouts to review", flush=True)

    if holdouts:
        from tools.replay_reconcile import finalize_review_routing
        print(f"\nFinalizing: routing {len(holdouts)} remaining "
              f"holdout(s) to human review", flush=True)
        finalize_review_routing(results_dir, holdouts)

    if os.getenv("CLINICAL_AUDIT", "1") == "1":
        # AUDIT_CONVERGENCE=1 (default): instead of one review pass that
        # leaves every dispute with a human, run the audit-convergence
        # loop — adjudicate each grounded dispute against the authorities,
        # actuate the verified fix into deterministic structure, replay,
        # and re-review until upheld or stalled.
        if os.getenv("AUDIT_CONVERGENCE", "1") == "1":
            try:
                from tools.audit_convergence_loop import converge
                summary = converge(results_dir, docs=scope)
                print(f"\nAudit convergence: {summary['status']} after "
                      f"{len(summary['iterations'])} iteration(s); "
                      f"remaining dispute(s): "
                      f"{summary.get('final_disputed', [])}", flush=True)
            except Exception as exc:
                print(f"Audit convergence failed ({exc}) — falling back "
                      f"to the single review pass", flush=True)
                _audit_only(results_dir, scope)
        else:
            _audit_only(results_dir, scope)

    # Pack consolidation — same wiring as tools/unanimity_loop.py: scan
    # cached by pack+corpus hash, merges gated on byte-identical corpus
    # replay, never a blocker for finalization.
    if os.getenv("PACK_CONSOLIDATION", "0") == "1":
        try:
            from tools.pack_consolidation import consolidate
            csum = consolidate(
                results_dir,
                merge=os.getenv("PACK_CONSOLIDATION_MERGE", "0") == "1")
            print(f"\nPack consolidation: "
                  f"{len(csum['dormancy'].get('tagged', []))} newly "
                  f"dormant, {len(csum['merges'])} merge(s) accepted, "
                  f"{len(csum['declined']) + len(csum['rejected'])} "
                  f"declined/rejected (accepted merges remain draft proposals)",
                  flush=True)
        except Exception as exc:
            print(f"Pack consolidation failed ({exc}) — pack left "
                  f"unconsolidated", flush=True)

    try:
        from tools.claims_registry import ingest as registry_ingest
        rstats = registry_ingest(results_dir)
        print(f"Claims registry (final): {rstats['recorded']} recorded, "
              f"{rstats['unchanged']} unchanged, "
              f"{rstats['skipped']} awaiting review/ineligible", flush=True)
    except Exception as exc:
        print(f"Claims registry ingest (final) failed ({exc})", flush=True)

    try:
        from tools.flip_triage import scan
        tstats = scan(results_dir)
        print(f"Triage scan: {tstats.get('new_classes', 0)} new class(es), "
              f"{tstats.get('audit_disputes_seen', 0)} audit "
              f"dispute(s)/finding(s) seen", flush=True)
    except Exception as exc:
        print(f"Triage scan failed ({exc})", flush=True)

    combined = []
    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        try:
            combined.append(json.loads(f.read_text()))
        except Exception:
            pass
    (results_dir / "all_results.json").write_text(
        json.dumps(combined, indent=2, default=str))
    print("\nFINALIZE_DONE", flush=True)


if __name__ == "__main__":
    main()
