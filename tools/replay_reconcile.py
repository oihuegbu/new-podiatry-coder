#!/usr/bin/env python3
"""Replay reconciliation: realize newly accepted rules on the batch that
produced them — no new LLM runs, no manual rerun.

The actuation pipeline accepts a rule only after replay proves it converges
the flip on the stored runs. This module completes that thought: after
actuation, every still-non-unanimous note's stored consistency runs are
replayed through the CURRENT rule pack (validator + 13-filter scrubber —
both deterministic), and if the replayed claims now agree, the note's saved
result is rebuilt from the replayed canonical run and marked unanimous.
The claims-registry ingest that follows then auto-records it — the full
capture -> rule -> verified-claim loop closes inside a single batch.

Notes whose replays still disagree stay deferred; routing them to a human
is the batch driver's finalization decision, not this module's.

CLI:
  docker compose run --rm app python tools/replay_reconcile.py [results_dir]
      [--docs stem1,stem2]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from tools.auto_actuate import (  # noqa: E402
    Replayer, _load_runs, _load_main, _note_text_for)

DEFAULT_RESULTS = ROOT / "output" / "results"

_ARRAY_KEYS = (("icd_codes", "icd10_codes"),
               ("cpt_codes", "cpt_codes"),
               ("hcpcs_codes", "hcpcs_codes"),
               ("supporting_conditions", "supporting_conditions"))


def _rebuild_run(run: dict, arrays: dict, report: dict,
                 scrubber, note_text: str) -> dict:
    """A stored run dump, its billing content re-derived under the current
    rule pack: replayed claim arrays, fresh validation report, fresh scrub
    verdict. Mirrors pipeline.py's own assembly (validation spread + \
_apply_scrub_verdict) so the rebuilt dump is shaped exactly like a live one."""
    out = json.loads(json.dumps(run, default=str))  # deep copy, JSON-safe
    for dump_key, cr_key in _ARRAY_KEYS:
        out[dump_key] = arrays.get(cr_key, out.get(dump_key) or [])

    for k, v in report.items():
        if k in ("auto_coding_review_reasons", "auto_coding_summary"):
            continue
        out[k] = v
    out["auto_coding_review_reasons"] = list(
        report.get("auto_coding_review_reasons") or [])
    out["auto_coding_summary"] = report.get("auto_coding_summary", "")

    # CORRECTION HISTORY SURVIVES REPLAYS. The fresh validation report can
    # only describe what THIS replay did — but the stored run's arrays are
    # already post-validation, so a correction made in the original pass
    # (measured live on routine_00008: a 99213 E/M removed with an explicit
    # medical-necessity flag) is invisible to the replay's own diff and
    # would vanish from the audited record. Carry every prior-pass
    # correction forward (deduplicated, tagged) so the clinical review
    # always verdicts the claim's FULL correction history, and the audit
    # fingerprint covers it.
    def _mkey(m: dict) -> tuple:
        return (m.get("category"), str(m.get("code") or "").upper(),
                m.get("action"), m.get("message"))
    fresh = [m for m in (out.get("material_corrections") or [])
             if isinstance(m, dict)]
    seen = {_mkey(m) for m in fresh}
    carried = [dict(m, carried_from_prior_pass=True)
               for m in (run.get("material_corrections") or [])
               if isinstance(m, dict) and _mkey(m) not in seen]
    out["material_corrections"] = fresh + carried

    scrub_payload = dict(out, note_text=note_text)
    scrub = scrubber.scrub(scrub_payload)
    # mode="json": the rebuilt record must be byte-shaped like a SAVED one.
    # A bare model_dump() keeps Status/DenialRisk as enum members, and every
    # downstream reader of the record contract (observables' signature(),
    # record_coherence, the emission-aware replay gates) compares against
    # the saved-file string form — measured live on routine_00003, where
    # in-memory replays read `str(Status.WARN)` ("Status.WARN") and the
    # advisory_emission observable silently measured NOTHING: the replay
    # gate saw every must-not-fire goal as already satisfied at baseline,
    # and replay_scope saw a perpetual emission diff against the saved file.
    out["claim_scrub"] = scrub.model_dump(mode="json")
    out["final_disposition"] = scrub.disposition.value
    out["final_summary"] = scrub.summary
    conf = float(out.get("auto_coding_confidence") or 0.0)

    # Same audit-pending gate as pipeline._apply_scrub_verdict: NO rebuilt
    # claim is CLEAN until the clinical-correctness review upholds it —
    # whole-claim review plus every interpretive correction (a replay
    # changes the claim, so any prior audit block is stale by fingerprint
    # and the post-batch audit re-judges it). Fail closed: no audit ->
    # REVIEW.
    interpretive = [m for m in (report.get("material_corrections") or [])
                    if isinstance(m, dict) and m.get("interpretive")]
    if scrub.clean:
        out.pop("clinical_audit", None)  # stale — the claim just changed
        out["final_disposition"] = "REVIEW"
        out["auto_coding_tier"] = "REVIEW"
        out["final_summary"] = (f"Scrub CLEAN, held for clinical audit: "
                                f"{scrub.summary}")
        out["auto_coding_summary"] = out["final_summary"]
        out["auto_coding_confidence"] = min(conf, 0.84)
        marker = "[clinical_audit/pending]"
        rr = [r for r in (out.get("auto_coding_review_reasons") or [])
              if marker not in str(r)]
        out["auto_coding_review_reasons"] = rr + [
            f"{marker} claim awaits the clinical-correctness review "
            f"({len(interpretive)} interpretive layer correction(s) to "
            f"verdict + whole-claim review) — scrub verdict is CLEAN and "
            f"will be restored on an upheld audit"]
        return out

    if scrub.clean:
        out["auto_coding_tier"] = "AUTO"
        out["auto_coding_summary"] = scrub.summary
        out["auto_coding_confidence"] = max(conf, 0.85)
    else:
        out["auto_coding_tier"] = "REVIEW"
        out["auto_coding_summary"] = scrub.summary
        out["auto_coding_confidence"] = min(conf, 0.84)
        out["auto_coding_review_reasons"] = (
            list(out.get("auto_coding_review_reasons") or [])
            + [f"[{f.filter_id}/{f.denial_risk.value}] "
               f"{', '.join(f.codes)}: {f.reason}"
               for f in scrub.blocking_findings])
    return out


def reconcile(results_dir: Path, docs: list[str] | None = None,
              rep: Replayer | None = None) -> dict:
    """Replay every (scoped) non-unanimous note's stored runs through the
    current pack; rewrite the saved result for any note that now agrees.
    Returns {"checked", "reconciled", "still_split", "docs": {...}}."""
    from app.validation.consistency import (
        compare_runs, select_canonical, annotate_result)
    from app.compliance.engine import ClaimScrubber
    from app.compliance.agents import build_default_agents

    stats = {"checked": 0, "reconciled": 0, "still_split": 0, "docs": {}}
    candidates = []
    for f in sorted(results_dir.glob("*_results.json")):
        if f.name == "all_results.json":
            continue
        doc = f.stem.removesuffix("_results")
        if docs is not None and doc not in docs:
            continue
        try:
            main = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(main, dict):
            continue
        cons = main.get("consistency") or {}
        if not cons or cons.get("unanimous"):
            continue
        candidates.append((doc, f, main))
    if not candidates:
        return stats

    rep = rep or Replayer()
    scrubber = ClaimScrubber(rep.store,
                             agents=build_default_agents(rep.store))
    for doc, f, main in candidates:
        runs = _load_runs(doc, results_dir)
        note = _note_text_for(doc, results_dir, runs, main)
        if len(runs) < 2 or not note:
            continue
        stats["checked"] += 1
        try:
            rebuilt = []
            for run in runs:
                arrays, report = rep.replay_arrays(run, note)
                rebuilt.append(_rebuild_run(run, arrays, report,
                                            scrubber, note))
            from app.validation.run_store import inherit_run_metadata
            new_report = inherit_run_metadata(
                compare_runs(rebuilt, store=rep.store), cons)
        except Exception as exc:
            logger.warning(f"Reconcile {doc}: replay failed ({exc}) — "
                           f"left as-is")
            continue
        if not new_report["unanimous"]:
            stats["still_split"] += 1
            stats["docs"][doc] = "still split after replay"
            continue
        idx = select_canonical(rebuilt)
        payload = annotate_result(rebuilt[idx], new_report)
        payload["reconciled_by_replay"] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "runs": len(rebuilt),
            "reason": "stored runs replayed through the updated rule pack "
                      "now produce identical claims",
        }
        from app.validation.run_store import atomic_write_json
        atomic_write_json(f, payload)
        stats["reconciled"] += 1
        stats["docs"][doc] = "reconciled — unanimous under current pack"
        logger.info(f"Reconciled {doc}: replay under the current pack is "
                    f"unanimous ({len(rebuilt)} runs)")
    return stats


def finalize_review_routing(results_dir: Path, docs: list[str]) -> int:
    """Automation is done with these notes — apply the deferred consistency
    verdict so their saved results carry the needs_review flags and REVIEW
    tier/disposition (annotate_result with route=True is a pure replay of
    the embedded report). Returns the number of notes routed."""
    from app.validation.consistency import annotate_result
    routed = 0
    for doc in docs:
        f = results_dir / f"{doc}_results.json"
        try:
            result = json.loads(f.read_text())
            report = result.get("consistency")
            if not report or report.get("unanimous"):
                continue
            annotate_result(result, report, route=True)
            from app.validation.run_store import atomic_write_json
            atomic_write_json(f, result)
            routed += 1
            logger.info(f"Routed to REVIEW: {doc}")
        except Exception as exc:
            logger.warning(f"Finalize failed for {doc}: {exc}")
    return routed


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", nargs="?", default=str(DEFAULT_RESULTS))
    p.add_argument("--docs", default="",
                   help="comma-separated note stems to restrict to")
    args = p.parse_args()
    docs = [s.strip() for s in args.docs.split(",") if s.strip()] or None
    stats = reconcile(Path(args.results_dir), docs=docs)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
