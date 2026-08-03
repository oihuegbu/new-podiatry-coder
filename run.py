#!/usr/bin/env python3
"""
Podiatry Medical Coding System
NER → RAG (Qdrant hybrid) → LLM (Claude Opus 5.0) → Validation Pipeline

Usage:
    python run.py                     # Process all notes
    python run.py --note NOTE_01...   # Process single note
    python run.py --rebuild-index     # Force rebuild Qdrant collections
    python run.py --setup-only        # Load/build all dependencies (models,
                                       # Qdrant collections, compliance.db)
                                       # and exit — no notes processed. Run
                                       # this once; the built state persists
                                       # in Docker volumes, so later
                                       # `python run.py` invocations skip
                                       # straight to fast-path processing.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import NOTES_DIR, OUTPUT_DIR
from app.core.logger import get_logger
from app.pipeline import MedicalCodingPipeline

logger = get_logger("main")

# ---------------------------------------------------------------- parallel
# consistency workers. Process-based (spawn), NOT threads: the pipeline's
# SQLite connections and mutable validator state are not thread-safe, and a
# spawned process builds its own full pipeline instance so nothing is shared.
# The N runs of one note are fully independent by design (use_cache=False),
# so running them concurrently changes wall time only — per-run behavior,
# inputs and outputs are byte-identical to the sequential path.
_WORKER_PIPELINE = None


def _consistency_worker_init():
    global _WORKER_PIPELINE
    from app.pipeline import MedicalCodingPipeline
    _WORKER_PIPELINE = MedicalCodingPipeline()
    _WORKER_PIPELINE.initialize()
    logger.info(f"  [CONSISTENCY] worker pid {os.getpid()} ready")


def _consistency_worker_run(pdf_path_str: str, run_idx: int, total: int) -> dict:
    logger.info(f"  [CONSISTENCY] {Path(pdf_path_str).stem}: run {run_idx}/{total} "
                f"(worker pid {os.getpid()})")
    try:
        result = _WORKER_PIPELINE.process_note(Path(pdf_path_str), use_cache=False)
        return result.model_dump()
    except Exception as exc:
        # Log the full traceback HERE, in the worker — it is the only place
        # the original stack exists. Then re-raise as a plain, picklable
        # exception: SDK exceptions (e.g. anthropic APIStatusError
        # subclasses) require keyword-only args that pickle's default
        # reconstruction can't supply — sending one across the process
        # boundary crashes the Pool's result-handler thread, after which
        # every p.get() in the parent hangs forever (observed live: batch
        # froze 3h mid-corpus on note 029).
        import traceback
        logger.error(f"  [CONSISTENCY] run {run_idx}/{total} failed for "
                     f"{Path(pdf_path_str).name}:\n{traceback.format_exc()}")
        raise RuntimeError(
            f"consistency run {run_idx}/{total} failed for "
            f"{Path(pdf_path_str).name}: {type(exc).__name__}: {exc}"
        ) from None


def main():
    parser = argparse.ArgumentParser(description="Podiatry Medical Coding System")
    parser.add_argument("--note", type=str, help="Process a single PDF note (filename or path)")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild Qdrant vector collections")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache lookup and force fresh processing")
    parser.add_argument("--setup-only", action="store_true", help="Load/build all dependencies and exit — process no notes")
    parser.add_argument("--consistency", type=int,
                        default=int(os.getenv("CONSISTENCY_RUNS", "1")),
                        help="Run each note N times and flag non-unanimous codes "
                             "for review (self-consistency; forces --no-cache)")
    parser.add_argument("--consistency-workers", type=int,
                        default=int(os.getenv("CONSISTENCY_WORKERS", "1")),
                        help="Run a note's N consistency runs concurrently in this "
                             "many worker processes (1 = sequential). Each worker "
                             "builds its own pipeline instance — the runs are "
                             "independent, so this only changes wall time. Mind "
                             "the LLM provider's rate limits before raising it.")
    parser.add_argument("--start-at", type=str, default="",
                        help="Skip notes sorting before this stem/filename prefix — "
                             "resume a batch without redoing completed notes")
    parser.add_argument("--end-at", type=str, default="",
                        help="Skip notes sorting after this stem/filename prefix — "
                             "bound a batch to a subset (e.g. --end-at 011 with "
                             "--start-at 001 runs notes 001-010)")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated note stems: process exactly these "
                             "notes (a non-contiguous subset --start/--end can't "
                             "express, e.g. the unanimity loop's holdouts)")
    args = parser.parse_args()

    from app.core.config import LLM_PROVIDER, CLAUDE_MODEL, OPENAI_MODEL
    active_model = CLAUDE_MODEL if LLM_PROVIDER == "claude" else OPENAI_MODEL
    logger.info("=" * 70)
    logger.info("PODIATRY MEDICAL CODING SYSTEM")
    logger.info(f"Pipeline: NER → RAG (Qdrant hybrid) → LLM ({LLM_PROVIDER}:{active_model}) → Validation")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    # Initialize pipeline
    pipeline = MedicalCodingPipeline()
    pipeline.initialize(force_rebuild_index=args.rebuild_index)

    if args.setup_only:
        logger.info("\n--setup-only: dependencies loaded, no notes processed. Exiting.")
        return

    # Find notes to process
    if args.note:
        note_path = Path(args.note)
        if not note_path.exists():
            note_path = NOTES_DIR / args.note
        if not note_path.exists():
            logger.error(f"Note not found: {args.note}")
            sys.exit(1)
        note_files = [note_path]
    else:
        note_files = sorted(NOTES_DIR.glob("*.pdf"))

    if not note_files:
        logger.error(f"No clinical notes found in {NOTES_DIR}")
        sys.exit(1)

    if args.start_at:
        before = len(note_files)
        note_files = [p for p in note_files if p.stem >= args.start_at]
        logger.info(f"--start-at {args.start_at}: skipping {before - len(note_files)} "
                    f"already-completed note(s)")
    if args.end_at:
        before = len(note_files)
        # Inclusive: '--end-at 010' must include 010's own note file —
        # '010_samuel...' compares GREATER than the bare prefix '010', so a
        # strict < silently dropped the last requested note (measured live:
        # two batches in a row skipped their final note with no message).
        note_files = [p for p in note_files
                      if p.stem[:len(args.end_at)] <= args.end_at]
        logger.info(f"--end-at {args.end_at}: excluding {before - len(note_files)} "
                    f"later note(s)")
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        note_files = [p for p in note_files if p.stem in wanted]
        missing = wanted - {p.stem for p in note_files}
        if missing:
            logger.error(f"--only: no PDF found for {sorted(missing)}")
            sys.exit(1)
        logger.info(f"--only: processing exactly {len(note_files)} note(s)")

    logger.info(f"\nProcessing {len(note_files)} clinical note(s)\n")

    # Worker pool for parallel consistency runs — created once for the whole
    # batch (each worker's pipeline init costs ~1 min; per-note pools would
    # pay it 54 times). 'spawn' start method: forked children would inherit
    # the parent's live SQLite connections and Qdrant client, which must not
    # be shared across processes; spawned children import fresh and build
    # their own in _consistency_worker_init.
    pool = None
    if args.consistency > 1 and args.consistency_workers > 1:
        import multiprocessing as mp
        # cap at the batch's total run count — extra workers would only idle
        n_workers = min(args.consistency_workers,
                        args.consistency * len(note_files))
        pool = mp.get_context("spawn").Pool(
            processes=n_workers, initializer=_consistency_worker_init)
        logger.info(f"[CONSISTENCY] {n_workers} parallel run workers starting "
                    f"(pipeline init in each)")

    # Process notes
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    deferred_docs: list[str] = []  # non-unanimous, awaiting finalization

    # Cross-note parallelism: every note's N runs are submitted to the pool
    # up front, so the workers drain one batch-wide queue instead of
    # parallelizing only within a single note (which capped useful workers
    # at N and left the batch note-sequential). With W workers, ~W/N notes
    # are in flight at once; wall time approaches max(single-run time) as W
    # grows. Results are still collected, compared and saved in note order.
    # The runs stay fully independent (own worker pipeline, use_cache=False),
    # so per-run behavior is byte-identical to the sequential path.
    jobs: dict = {}
    if pool is not None and args.consistency > 1:
        for pdf_path in note_files:
            jobs[pdf_path] = [
                pool.apply_async(_consistency_worker_run,
                                 (str(pdf_path), i + 1, args.consistency))
                for i in range(args.consistency)
            ]

    for pdf_path in note_files:
        try:
            if args.consistency > 1:
                # Self-consistency: N independent uncached runs; any code that
                # is not unanimous across them is routed to human review.
                from app.validation.consistency import (
                    compare_runs, select_canonical, annotate_result)
                if pool is not None:
                    dumps = [p.get() for p in jobs[pdf_path]]
                else:
                    dumps = []
                    for i in range(args.consistency):
                        logger.info(f"  [CONSISTENCY] run {i + 1}/{args.consistency}")
                        r = pipeline.process_note(pdf_path, use_cache=False)
                        dumps.append(r.model_dump())
                # Persist every independent run — flip forensics need the
                # losing runs' validation traces, and only the canonical
                # payload survives otherwise.
                runs_dir = OUTPUT_DIR / "consistency_runs"
                runs_dir.mkdir(parents=True, exist_ok=True)
                for i, dump in enumerate(dumps, 1):
                    with open(runs_dir / f"{pdf_path.stem}_run{i}.json", "w") as rf:
                        json.dump(dump, rf, indent=2, default=str)
                report = compare_runs(dumps, store=pipeline.compliance_store)
                idx = select_canonical(dumps)
                # A disagreement at save time is not yet a human's problem:
                # the post-batch actuation may mint a rule and the replay
                # reconciliation realize it minutes from now. Routing to
                # REVIEW is a FINALIZATION decision, made after this batch's
                # actuation+reconcile pass (below) — or by the unanimity
                # loop's own end-of-loop finalization when it is driving.
                payload = annotate_result(dumps[idx], report, route=False)
                if not report["unanimous"]:
                    deferred_docs.append(pdf_path.stem)
                n_billing = sum(1 for d in report["disagreements"]
                                if not d.get("advisory"))
                n_advisory = len(report["disagreements"]) - n_billing
                if not report["unanimous"]:
                    logger.warning(
                        f"  [CONSISTENCY] {pdf_path.stem}: {n_billing} billing "
                        f"disagreement(s) ({n_advisory} advisory) across "
                        f"{report['runs']} runs — review deferred pending "
                        f"actuation + replay reconciliation")
                else:
                    logger.info(
                        f"  [CONSISTENCY] {pdf_path.stem}: billing arrays "
                        f"unanimous across {report['runs']} runs"
                        + (f" ({n_advisory} advisory-only variance)" if n_advisory else ""))
            else:
                result = pipeline.process_note(pdf_path, use_cache=not args.no_cache)
                payload = result.model_dump()
            results.append(payload)

            output_file = OUTPUT_DIR / f"{pdf_path.stem}_results.json"
            with open(output_file, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            logger.info(f"  Saved → {output_file.name}")

        except Exception as e:
            logger.error(f"FAILED: {pdf_path.name} — {e}")
            import traceback
            traceback.print_exc()

    if pool is not None:
        pool.close()
        pool.join()

    def _rebuild_all_results():
        # Combined output — rebuilt from the per-note files on disk, not
        # this batch's in-memory list, so a resumed batch (--start-at) or
        # partial failure never overwrites the corpus aggregate with a
        # subset. Called again after the post-batch growth loop, whose
        # reconciliation/finalization rewrites per-note files.
        combined_payloads = []
        for f_path in sorted(OUTPUT_DIR.glob("*_results.json")):
            if f_path.name == "all_results.json":
                continue
            try:
                combined_payloads.append(json.loads(f_path.read_text()))
            except Exception as exc:
                logger.warning(f"  all_results: skipped unreadable {f_path.name} ({exc})")
        with open(OUTPUT_DIR / "all_results.json", "w") as f:
            json.dump(combined_payloads, f, indent=2, default=str)

    if results:
        _rebuild_all_results()

    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("BATCH COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total: {len(results)} | Success: {sum(1 for r in results if r.get('success'))} | Failed: {sum(1 for r in results if not r.get('success'))}")

    tiers = {"AUTO": 0, "REVIEW": 0, "REJECT": 0}
    for r in results:
        tiers[r.get("auto_coding_tier")] = tiers.get(r.get("auto_coding_tier"), 0) + 1

    logger.info(f"Auto: {tiers['AUTO']} | Review: {tiers['REVIEW']} | Reject: {tiers['REJECT']}")
    logger.info(f"Output → {OUTPUT_DIR}")

    # Denial feedback gate — part of the pipeline, not a side tool: when the
    # registry holds payer denials (ingested from 835s/CSV), every batch
    # re-checks that each denial's error class is now caught pre-submission.
    # A MISSED denial means a deterministic layer is still absent for a
    # failure mode a payer has already demonstrated — loudest possible flag.
    try:
        from tools.denial_feedback import _load_registry, build_report
        if _load_registry():
            report = build_report(OUTPUT_DIR)
            counts = report["counts"]
            missed = [c for c in report["denials"]
                      if c["status"] == "MISSED" and not c.get("waived")]
            logger.info(f"Denial feedback: {counts}")
            for c in missed:
                logger.warning(
                    f"  [DENIAL MISSED] {c.get('document_id')} {c.get('code')} "
                    f"CARC {c.get('carc')} ({c.get('carc_label', '')[:50]}) — "
                    f"pipeline still passes this clean; a layer is missing")
    except Exception as e:  # never let the gate break batch output
        logger.warning(f"Denial feedback gate skipped: {e}")

    # Post-batch growth loop, in dependency order:
    #   1. flip triage + actuation  — disagreements become declarative rules
    #   2. replay reconciliation    — accepted rules are realized on THIS
    #      batch's stored runs (no new LLM spend); notes whose replays now
    #      agree become unanimous on disk
    #   3. finalization             — whatever is STILL split has survived
    #      an actuation pass that minted no fix for it: only now does it
    #      become a human's problem (unless the unanimity loop is driving —
    #      it iterates further and owns the finalization decision)
    #   4. registry ingest + calibration — record every unanimous+CLEAN
    #      claim, including ones reconciliation just converged
    if args.consistency > 1:
        actuated_rules = 0
        try:
            from tools import flip_triage
            tstats = flip_triage.scan(OUTPUT_DIR)
            logger.info(
                f"Flip queue: {tstats['total_classes']} class(es), "
                f"{tstats['open']} open")
            # total_classes, not open: actuate() also reopens escalations
            # whose "no template fits" verdict predates the current template
            # vocabulary — a queue with zero open classes can still yield.
            if tstats["total_classes"] and os.getenv("AUTO_ACTUATE", "1") == "1":
                from tools.auto_actuate import actuate
                limit = int(os.getenv("AUTO_ACTUATE_LIMIT", "5"))
                # Scope to THIS batch's documents: evidence, convergence,
                # and the inertness control set all stay inside the corpus
                # just processed — stale results from older corpora in the
                # same directory can neither trigger nor veto a rule.
                # AUTO_ACTUATE_SCOPE (comma-separated stems/prefixes)
                # overrides for callers that process a SUBSET but need the
                # gates verified against the whole corpus — the unanimity
                # loop reruns only holdouts, but a rule must still be inert
                # on the notes that are already unanimous.
                scope_env = os.getenv("AUTO_ACTUATE_SCOPE", "")
                batch_scope = (
                    tuple(s.strip() for s in scope_env.split(",")
                          if s.strip())
                    if scope_env else tuple(p.stem for p in note_files))
                astats = actuate(OUTPUT_DIR, limit=limit, dry_run=False,
                                 scope=batch_scope)
                actuated_rules = astats["actuated"]
                logger.info(
                    f"Auto-actuation: {astats['actuated']} rule(s) accepted, "
                    f"{astats['escalated']} class(es) escalated to human "
                    f"review (of {astats['considered']} considered)")
        except Exception as e:
            logger.warning(f"Flip actuation skipped: {e}")

        # Unconditional when anything is split: rules accepted THIS pass are
        # the obvious trigger, but actuation also proves classes "resolved
        # at baseline" (the current pack already converges their replays —
        # measured live: 6 such classes with zero acceptances), and the pack
        # can have moved through any parallel driver. Replay is cheap and
        # deterministic; a skipped reconcile strands convergeable notes.
        if deferred_docs:
            try:
                from tools.replay_reconcile import reconcile
                rstats = reconcile(OUTPUT_DIR, docs=deferred_docs)
                logger.info(
                    f"Replay reconciliation: {rstats['reconciled']} note(s) "
                    f"now unanimous under the new rules, "
                    f"{rstats['still_split']} still split "
                    f"(of {rstats['checked']} replayed)")
                deferred_docs = [
                    d for d in deferred_docs
                    if not rstats["docs"].get(d, "").startswith("reconciled")]
            except Exception as e:
                logger.warning(f"Replay reconciliation skipped: {e}")

        # Expert-coder adjudication: what survives actuation + reconcile is
        # judgment-shaped (modifier-25 significance, MDM leveling,
        # documentation sufficiency) — no generic deterministic rule decides
        # it. The automated coder (Fable 5, bound to the authoritative
        # sources in the case file) adjudicates each holdout; independent
        # passes must agree, verdicts touch only the disputed items, and
        # the realigned runs must replay unanimous. Only abstentions and
        # split verdicts continue on to the human queue. When the unanimity
        # loop is driving it adjudicates at ITS end instead — mid-loop
        # holdouts may still converge under the next accepted rule.
        if deferred_docs and os.getenv("DEFER_REVIEW_ROUTING", "0") != "1" \
                and os.getenv("CODER_ADJUDICATION", "1") == "1":
            try:
                from tools.coder_adjudicator import adjudicate
                astats = adjudicate(OUTPUT_DIR, docs=deferred_docs)
                logger.info(
                    f"Coder adjudication: {astats['adjudicated']} note(s) "
                    f"settled, {astats['abstained']} abstained, "
                    f"{astats['split_verdicts']} split, "
                    f"{astats['failed_replay']} failed replay "
                    f"(of {astats['considered']} considered)")
                deferred_docs = [
                    d for d in deferred_docs
                    if not astats["docs"].get(d, "").startswith("adjudicated")]
            except Exception as e:
                logger.warning(f"Coder adjudication skipped: {e}")

        # Finalization: the unanimity loop (DEFER_REVIEW_ROUTING=1) keeps
        # iterating with fresh LLM runs and finalizes at ITS end; every
        # other driver finalizes here — this batch's actuation+reconcile
        # was its one automated shot.
        if deferred_docs and os.getenv("DEFER_REVIEW_ROUTING", "0") != "1":
            try:
                from tools.replay_reconcile import finalize_review_routing
                routed = finalize_review_routing(OUTPUT_DIR, deferred_docs)
                logger.info(f"Finalized: {routed} note(s) routed to REVIEW")
            except Exception as e:
                logger.warning(f"Review finalization skipped: {e}")

        # Clinical-correctness review — the universal CLEAN gate: EVERY
        # scrub-CLEAN claim was HELD at REVIEW by the pipeline under the
        # [clinical_audit/pending] marker, and only this review's upheld
        # verdict promotes it to CLEAN. The review is a whole-claim expert
        # pass (code selection, primary designation, missing documented
        # codes, coverage logic, wrong system advisories) PLUS a verdict on
        # every interpretive layer correction — self-reported and
        # diff-derived, so no unreported mutation escapes it. A dispute
        # replaces the hold with the named item (human queue), and every
        # disputed correction/finding is enqueued as an audit_dispute flip
        # class on the next triage scan — the growth loop that turns
        # confirmed review findings into deterministic rules/templates once
        # a human verifies the correct claim. Fail closed: no review ->
        # never CLEAN.
        # DEFER_CLINICAL_AUDIT=1 (set by the unanimity loop) skips the audit
        # here: mid-loop, a holdout's interpretive corrections still change
        # every iteration, so auditing them now re-spends the LLM on
        # intermediate states and lets a dispute be overwritten by the next
        # iteration before the growth queue captures it. The loop runs the
        # audit ONCE at its finalization on the settled claims instead.
        # Registry auto-recording independently blocks un-audited
        # interpretive claims (eligible_for_auto fails closed), so nothing
        # ships CLEAN while the audit is deferred.
        if (os.getenv("CLINICAL_AUDIT", "1") == "1"
                and os.getenv("DEFER_CLINICAL_AUDIT", "0") != "1"):
            try:
                from tools.clinical_auditor import audit_batch
                batch_docs = [str(r.get("document_id", ""))
                              for r in results
                              if isinstance(r, dict) and r.get("document_id")]
                caud = audit_batch(OUTPUT_DIR, docs=batch_docs or None)
                logger.info(
                    f"Clinical audit: {caud['audited']} audited "
                    f"({caud['upheld']} upheld, {caud['disputed']} "
                    f"disputed), {caud['skipped']} unchanged/skipped")
                for doc, msg in sorted(caud["docs"].items()):
                    if "disputed" in msg:
                        logger.warning(f"  [clinical-audit] {doc}: {msg}")
                # Audit convergence — a disputed review does not just sit
                # in the human queue: AUDIT_CONVERGENCE=1 (default) runs
                # the convergence loop, where the expert-coder adjudicator
                # settles each grounded finding against the authoritative
                # sources, actuation turns the verified fix into a
                # deterministic rule/template/gate, and the note replays
                # and is re-reviewed until upheld. Only a STALLED loop
                # leaves disputes with a human.
                if os.getenv("AUDIT_CONVERGENCE", "1") == "1":
                    from tools.audit_convergence_loop import (
                        _disputed_docs, _scope_docs, converge)
                    # judge from the saved files, not this pass's counters:
                    # a note still disputed from an EARLIER batch is skipped
                    # by fingerprint (caud counts it as unchanged), and it
                    # deserves convergence just the same
                    in_scope = _scope_docs(OUTPUT_DIR, batch_docs or None)
                    if _disputed_docs(OUTPUT_DIR, in_scope):
                        summary = converge(OUTPUT_DIR,
                                           docs=batch_docs or None)
                        logger.info(
                            f"Audit convergence: {summary['status']} after "
                            f"{len(summary['iterations'])} iteration(s); "
                            f"remaining dispute(s): "
                            f"{summary.get('final_disputed', [])}")
            except Exception as e:
                logger.warning(f"Clinical audit skipped: {e}")

        try:
            from tools.claims_registry import ingest as registry_ingest
            stats = registry_ingest(OUTPUT_DIR)
            logger.info(
                f"Claims registry: {stats['recorded']} recorded, "
                f"{stats['unchanged']} unchanged, "
                f"{stats['human_protected']} human-protected, "
                f"{stats['skipped']} awaiting review/ineligible")
        except Exception as e:  # never let the ledger break batch output
            logger.warning(f"Claims registry ingest skipped: {e}")

        # Pack consolidation — the growth loop's maintenance counterpart
        # (tools/pack_consolidation.py): exercise scan (cached by
        # pack+corpus hash — free when nothing changed), dormancy tags,
        # and behavior-preserving merges gated on byte-identical corpus
        # replay. When the unanimity loop is driving
        # (DEFER_REVIEW_ROUTING=1) it consolidates at ITS finalization
        # instead — mid-loop the pack is still growing.
        if os.getenv("DEFER_REVIEW_ROUTING", "0") != "1" \
                and os.getenv("PACK_CONSOLIDATION", "1") == "1":
            try:
                from tools.pack_consolidation import consolidate
                csum = consolidate(
                    OUTPUT_DIR,
                    merge=os.getenv("PACK_CONSOLIDATION_MERGE",
                                    "1") == "1")
                logger.info(
                    f"Pack consolidation: "
                    f"{len(csum['dormancy'].get('tagged', []))} newly "
                    f"dormant, {len(csum['merges'])} merge(s) accepted, "
                    f"{len(csum['declined']) + len(csum['rejected'])} "
                    f"declined/rejected")
            except Exception as e:
                logger.warning(f"Pack consolidation skipped: {e}")

        # Claim submission — registry-verified CLEAN claims -> 837P via the
        # clearinghouse adapter. Opt-in (AUTO_SUBMIT_CLAIMS=1): transmission
        # is irreversible, so the default posture is build-on-demand via
        # `python tools/claim_submitter.py --dry-run`. Every envelope value
        # (fee schedule, NPIs/TIN, payer trading-partner IDs, facility) is
        # read from data/practice_config.json + data/codes/payers.json at
        # submission time; a missing value blocks that one claim with a
        # recorded reason and never crashes the batch.
        if os.getenv("AUTO_SUBMIT_CLAIMS", "0") == "1":
            try:
                from tools.claim_submitter import submit_all
                sstats = submit_all(OUTPUT_DIR)
                logger.info(
                    f"Claim submission: {sstats['submitted']} submitted, "
                    f"{sstats['already_submitted']} already submitted, "
                    f"{sstats['blocked']} blocked")
                for doc, msg in sorted(sstats["docs"].items()):
                    if not msg.startswith("submitted"):
                        logger.info(f"  [submit] {doc}: {msg}")
            except Exception as e:
                logger.warning(f"Claim submission skipped: {e}")

        # Calibration dataset — labeled rows (features from each result,
        # labels from the routing verdict + registry human/payer outcomes)
        # accumulate from day one so a learned confidence model can be
        # trained the moment human-verdict volume supports it.
        try:
            from tools.calibration_dataset import (
                export as calib_export, DEFAULT_OUT as CALIB_OUT)
            cstats = calib_export(OUTPUT_DIR, CALIB_OUT)
            logger.info(
                f"Calibration dataset: {cstats['total']} row(s) "
                f"({cstats['new']} new, {cstats['updated']} updated)")
        except Exception as e:
            logger.warning(f"Calibration dataset export skipped: {e}")

        # Template graduation proposal — maturity is evaluated, but runtime
        # code never writes executable application modules.
        if os.getenv("AUTO_GRADUATE", "0") == "1":
            try:
                from tools.graduate_templates import graduate
                gstats = graduate(OUTPUT_DIR)
                if gstats["considered"]:
                    logger.info(
                        f"Template graduation: "
                        f"{len(gstats['promoted'])} proposed, "
                        f"{len(gstats['not_yet'])} not yet eligible, "
                        f"{len(gstats['failed'])} rolled back "
                        f"(of {gstats['considered']} sandboxed)")
            except Exception as e:
                logger.warning(f"Template graduation skipped: {e}")

        # Reconciliation/finalization rewrote per-note files after the
        # first aggregate build — refresh so all_results.json matches disk.
        if results:
            _rebuild_all_results()


if __name__ == "__main__":
    main()
