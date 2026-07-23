#!/usr/bin/env python3
"""Audit-convergence loop — grounded review findings become deterministic
structure, then the notes replay until the review has nothing left to
dispute.

The unanimity loop converges REPEATABILITY failures (runs disagree). This
loop converges CORRECTNESS failures — defects the clinical review catches
that runs can never expose, because a wrong deterministic decision is
unanimous by construction. It automates the same discipline the manual
process used: judge against the authorities, fix structurally, re-run,
repeat.

Each iteration:
  1. clinical review of the scope (idempotent by fingerprint — only
     changed claims spend LLM)
  2. audit adjudication (tools/coder_adjudicator.adjudicate_audit): the
     expert coder decides every mechanizable disputed finding from the
     authoritative sources, applies it mechanically, replays it through
     the full deterministic stack, re-reviews it, and records the upheld
     corrected claim at the adjudicated tier — the verified realignment
     target the audit_dispute flip classes wait for
  3. triage scan + actuation: the now-verified classes open, and the
     proposal machinery turns each into a declarative rule, synthesized
     template, or graduated gate, accepted only when replay lands
     byte-identical on the verified claim (structural, never a hardcoded
     medical code)
  4. deterministic replay of the whole scope under the grown pack — any
     note whose claim changes loses its stale review block and is
     re-reviewed on the next iteration

The loop exits when the scope has no disputed reviews (converged), or
when an iteration produces no adjudications, no accepted rules, and no
claim changes (stalled — exactly then the remaining disputes stay held at
REVIEW for a human, never before). Every hold is fail-closed throughout:
nothing becomes CLEAN or registry-recorded mid-loop without an upheld
review of its current content.

Usage (inside the app container):
  python tools/audit_convergence_loop.py [results_dir] \
      [--docs stem1,stem2] [--max-iterations N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

DEFAULT_RESULTS = ROOT / "output" / "results"
MAX_ITERATIONS = int(os.getenv("AUDIT_CONVERGENCE_ITERATIONS", "3"))
ACTUATION_LIMIT = int(os.getenv("AUDIT_CONVERGENCE_ACTUATION_LIMIT", "10"))


def _scope_docs(results_dir: Path, docs: list[str] | None) -> list[str]:
    out = []
    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        doc = f.stem.removesuffix("_results")
        if docs is None or doc in docs:
            out.append(doc)
    return out


def _disputed_docs(results_dir: Path, docs: list[str]) -> dict[str, str]:
    """{doc: review fingerprint} for every scoped note whose clinical
    review currently disputes the claim."""
    out = {}
    for doc in docs:
        f = results_dir / f"{doc}_results.json"
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        audit = (r.get("clinical_audit") or {}) if isinstance(r, dict) \
            else {}
        if audit.get("verdict") == "disputed":
            out[doc] = str(audit.get("fingerprint") or "")
    return out


def _observable_sigs(result: dict) -> dict:
    """{observable: emission signature} of a saved result — every
    measurement observable's full firing set (tools/observables.py;
    advisory emission plus whatever the vocabulary has grown to). A rule
    that changes a measured phenomenon changes THIS and nothing in the
    billing signature, so replay rewrites must watch both."""
    try:
        from tools.observables import record_signatures
        return record_signatures(result)
    except Exception as exc:
        logger.warning(f"observable signatures unavailable ({exc})")
        return {}


def replay_scope(results_dir: Path, docs: list[str], rep) -> int:
    """Deterministically replay every scoped note's stored runs through
    the CURRENT rule pack — including notes that are already unanimous,
    which reconcile() skips by design (an audit dispute is unanimous by
    construction, so the note a new rule just fixed is exactly the one
    reconcile would never touch). A note is rewritten when the replayed
    claim's billing signature differs from the saved one, OR — billing
    lines byte-identical — when any measurement observable's emission
    signature differs (advisory emission or any synthesized observable),
    because an adjudicated emission suppression is claim-invisible by
    design; the rewrite drops the stale review block, so the next review
    pass judges the new record. Returns the number of notes rewritten."""
    from app.compliance.agents import build_default_agents
    from app.compliance.engine import ClaimScrubber
    from app.validation.consistency import (annotate_result, compare_runs,
                                            select_canonical)
    from tools.auto_actuate import (Replayer, _load_runs, _note_text_for)
    from tools.replay_reconcile import _rebuild_run

    scrubber = ClaimScrubber(rep.store,
                             agents=build_default_agents(rep.store))
    changed = 0
    for doc in docs:
        f = results_dir / f"{doc}_results.json"
        try:
            main = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(main, dict) or not main.get("success"):
            continue
        runs = _load_runs(doc, results_dir)
        note = _note_text_for(doc, results_dir, runs or [main], main)
        if not note:
            continue
        source = runs if len(runs) >= 2 else [main]
        try:
            rebuilt = []
            for run in source:
                arrays, report = rep.replay_arrays(run, note)
                rebuilt.append(_rebuild_run(run, arrays, report,
                                            scrubber, note))
            new_report = (compare_runs(rebuilt, store=rep.store)
                          if len(rebuilt) >= 2 else None)
        except Exception as exc:
            logger.warning(f"Replay {doc} failed ({exc}) — left as-is")
            continue
        if new_report and not new_report.get("unanimous"):
            # the pack change split previously-agreeing runs — leave the
            # saved result alone; the unanimity machinery owns this case
            logger.warning(f"Replay {doc}: pack change split the runs — "
                           f"left as-is for the consistency loop")
            continue
        idx = select_canonical(rebuilt) if len(rebuilt) >= 2 else 0
        old_sig = Replayer.signature(main.get("icd_codes"),
                                     main.get("cpt_codes"),
                                     main.get("hcpcs_codes"))
        new_sig = Replayer.signature(rebuilt[idx].get("icd_codes"),
                                     rebuilt[idx].get("cpt_codes"),
                                     rebuilt[idx].get("hcpcs_codes"))
        advisory_only = (new_sig == old_sig and
                         _observable_sigs(rebuilt[idx])
                         != _observable_sigs(main))
        if new_sig == old_sig and not advisory_only:
            continue
        if new_report:
            payload = annotate_result(rebuilt[idx], new_report)
            payload["consistency"] = new_report
        else:
            payload = rebuilt[idx]
            if main.get("consistency"):
                payload["consistency"] = main["consistency"]
        payload["replayed_by_audit_convergence"] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "reason": ("observable emission changed under a rule pack "
                       "grown from verified clinical-review findings "
                       "(claim lines byte-identical)" if advisory_only else
                       "claim changed under a rule pack grown from "
                       "verified clinical-review findings"),
        }
        f.write_text(json.dumps(payload, indent=2, default=str))
        changed += 1
        logger.info(f"Replayed {doc}: "
                    + ("observable emission" if advisory_only else "claim")
                    + " changed under the grown pack — review block reset "
                      "for a fresh verdict")
    return changed


def converge(results_dir: Path, docs: list[str] | None = None,
             max_iterations: int = MAX_ITERATIONS,
             rep=None) -> dict:
    from tools import flip_triage
    from tools.auto_actuate import Replayer, actuate
    from tools.clinical_auditor import audit_batch
    from tools.coder_adjudicator import adjudicate_audit

    scope = _scope_docs(results_dir, docs)
    summary = {"status": "converged", "iterations": [], "scope": len(scope)}
    if not scope:
        return summary
    rep = rep or Replayer()
    act_scope = tuple(sorted(docs)) if docs else ()

    for it in range(1, max_iterations + 1):
        caud = audit_batch(results_dir, docs=scope)
        disputed = _disputed_docs(results_dir, scope)
        logger.info(f"[audit-convergence {it}/{max_iterations}] "
                    f"{len(disputed)} disputed review(s) in scope "
                    f"({caud['audited']} freshly audited)")
        if not disputed:
            summary["status"] = "converged"
            summary["iterations"].append({"iteration": it, "disputed": 0})
            break

        astats = adjudicate_audit(results_dir, docs=sorted(disputed),
                                  rep=rep)
        flip_triage.scan(results_dir)
        try:
            actstats = actuate(results_dir, limit=ACTUATION_LIMIT,
                               dry_run=False, scope=act_scope)
        except Exception as exc:
            logger.warning(f"Actuation failed this iteration ({exc})")
            actstats = {"actuated": 0}
        changed = replay_scope(results_dir, scope, rep)

        progress = (astats.get("adjudicated", 0)
                    + astats.get("partial", 0)
                    + actstats.get("actuated", 0)
                    + actstats.get("resolved_baseline", 0)
                    + changed)
        summary["iterations"].append({
            "iteration": it,
            "disputed": len(disputed),
            "adjudicated": astats.get("adjudicated", 0),
            "partially_adjudicated": astats.get("partial", 0),
            "still_disputed_after_adjudication":
                astats.get("still_disputed", 0),
            "rules_actuated": actstats.get("actuated", 0),
            "resolved_baseline": actstats.get("resolved_baseline", 0),
            "claims_changed_on_replay": changed,
        })
        if progress == 0:
            # MEASUREMENT-GAP GROWTH: before declaring a stall, check
            # whether the loop is blocked on a phenomenon the gates
            # cannot MEASURE — a routing-grade finding no observable's
            # vocabulary resolves (advisory emission was exactly this
            # gap once, hand-built; tools/observable_synthesis.py is
            # that growth automated). An installed observable changes
            # what the next iteration can adjudicate, so the loop
            # continues instead of stalling — the note re-runs against
            # the grown measurement system, the same way it re-runs
            # against a grown rule pack.
            grown = 0
            try:
                from tools.observable_synthesis import grow_observables
                grown = grow_observables(results_dir, sorted(disputed))
            except Exception as exc:
                logger.warning(f"Observable synthesis unavailable/failed "
                               f"({exc})")
            if grown:
                summary["observables_installed"] = \
                    summary.get("observables_installed", 0) + grown
                summary["iterations"][-1]["observables_installed"] = grown
                logger.info(
                    f"[audit-convergence] iteration {it}: {grown} new "
                    f"measurement observable(s) installed — the gates' "
                    f"vocabulary grew, continuing instead of stalling")
                continue
            summary["status"] = "stalled"
            logger.info(
                f"[audit-convergence] iteration {it} produced no "
                f"adjudications, no accepted rules, no claim changes, "
                f"and no measurement-vocabulary growth — "
                f"{len(disputed)} dispute(s) stay held at REVIEW for a "
                f"human coder")
            break
    else:
        summary["status"] = "iterations_exhausted"

    # settle the books: review anything the last replay changed, then let
    # the registry record what is now upheld
    final = audit_batch(results_dir, docs=scope)
    summary["final_disputed"] = sorted(_disputed_docs(results_dir, scope))
    if summary["final_disputed"]:
        summary["status"] = summary["status"] if summary["status"] != \
            "converged" else "stalled"
    try:
        from tools.claims_registry import ingest as registry_ingest
        summary["registry"] = registry_ingest(results_dir)
    except Exception as exc:
        logger.warning(f"Registry ingest failed: {exc}")
    flip_triage.scan(results_dir)

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
    logger.info(f"[audit-convergence] {summary['status'].upper()}: "
                f"{len(summary['final_disputed'])} dispute(s) remain "
                f"(final review pass audited {final['audited']})")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", nargs="?", default=str(DEFAULT_RESULTS))
    p.add_argument("--docs", default="",
                   help="comma-separated note stems to restrict to")
    p.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = p.parse_args()
    docs = [s.strip() for s in args.docs.split(",") if s.strip()] or None
    summary = converge(Path(args.results_dir), docs=docs,
                       max_iterations=args.max_iterations)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
