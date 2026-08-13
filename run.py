#!/usr/bin/env python3
"""Podiatry Medical Coding System — the deployed batch entrypoint.

    PDF ─► verbatim note text (app.ingestion.pdf_parser — a text-extraction utility)
        ─► claude_coder.pipeline.code_encounter   ◄── THE note→code decision engine
        ─► {stem}_results.json + all_results.json in OUTPUT_DIR

Usage:
    python run.py                     # process every PDF in NOTES_DIR
    python run.py --note NOTE_01.pdf  # process a single note
    python run.py --rebuild-index     # force-rebuild the Qdrant collections
    python run.py --setup-only        # build/load all dependencies and exit —
                                      # no notes processed. Run once; the built
                                      # state persists in the Docker volumes, so
                                      # later `python run.py` invocations skip
                                      # straight to processing.

================================================================================
WHY THIS FILE CHANGED — issue #6, Codex finding F6-R4-A1 (P1)
================================================================================
Until this cutover the deployed entrypoint — `docker-compose.yml`'s
`command: ["python", "run.py"]`, and the `process-notes.sh` helper plus
`note-watcher.service` that `terraform/templates/user_data.sh.tftpl` installs —
constructed `app.pipeline.MedicalCodingPipeline`.

That is a *different, non-communicating* implementation from
`claude_coder.pipeline.code_encounter`, which is where the provenance
repository, the source gates, eligibility-before-retrieval, certificate
creation, and the external terminal-head checkpoint all live. `app/` and
`claude_coder/` do not import each other, so pinning
`PROVENANCE_CHECKPOINT_REQUIRED=1` in Compose was inert for the real note
processor: no deployed claim ever passed through any of those controls.

`app.pipeline` is a previous build that did not translate doctors' notes to
codes accurately; `claude_coder` is its replacement. This entrypoint now makes
the note→code decision with `claude_coder.pipeline.code_encounter` and keeps
only the operational shell around it — note discovery, note selection, output
files, logging and the batch summary.

Reachability of the checkpoint chain from THIS file is regression-tested end to
end by `tests/test_deployment_entrypoint.py`, which drives `main()` exactly as
`docker compose run app python run.py` does, with a required-but-unavailable
checkpoint anchor, and proves no releasable claim or certificate can emerge.

DELIBERATELY NOT CALLED FROM HERE ANY MORE — not an oversight, not a TODO
--------------------------------------------------------------------------------
The post-batch "growth loop" this file used to drive is paradigm-specific to the
retired `app.pipeline` self-consistency model: it re-ran each note N times,
compared the N runs' billing arrays, and minted/replayed declarative rules out
of the disagreements. `claude_coder` was explicitly designed to replace that
approach with built-in propose-then-verify plus independent cross-model
corroboration (see `claude_coder/README.md`, "Running it"), so there are no N
runs to compare and nothing for that machinery to consume. Porting it to the new
paradigm is out of scope for this cutover.

The tool modules themselves are INTENTIONALLY LEFT IN PLACE and untouched — they
remain independently runnable and may be revisited — they are simply no longer
invoked from the deployed entrypoint:

    app/validation/consistency.py     N-run comparison / canonical selection
    tools/flip_triage.py              flip triage queue
    tools/auto_actuate.py             declarative rule auto-actuation
    tools/replay_reconcile.py         replay reconciliation + review finalization
    tools/coder_adjudicator.py        expert-coder adjudication
    tools/clinical_auditor.py         clinical-correctness review
    tools/audit_convergence_loop.py   audit-dispute convergence
    tools/claims_registry.py          finalized-claims registry ingest
    tools/pack_consolidation.py       rule-pack consolidation
    tools/calibration_dataset.py      calibration row export
    tools/graduate_templates.py       template graduation
    tools/denial_feedback.py          denial-feedback gate
    tools/claim_submitter.py          837P claim submission — see below

Automatic claim submission deserves its own note. `AUTO_SUBMIT_CLAIMS=1` used to
make this entrypoint transmit real 837P claims to the clearinghouse at the end of
every batch. Transmission is irreversible, and the submitter reads an
`app.pipeline`-shaped result that this entrypoint no longer produces, so the
deployed entrypoint now submits nothing under any configuration. Submission is an
explicit, separate, human-run step: `python tools/claim_submitter.py --dry-run`.

`tools/unanimity_loop.py` *drives* this file (`subprocess … run.py --consistency
N --consistency-workers W`). It is left in place too, and the retired flags below
are still parsed here for exactly one reason: so that driver fails LOUDLY, naming
this finding, instead of silently receiving one non-consistency run when it asked
for N independent ones.

Environment variables that only the removed call sites read (`CONSISTENCY_RUNS`,
`CONSISTENCY_WORKERS`, `AUTO_ACTUATE*`, `DEFER_REVIEW_ROUTING`,
`CODER_ADJUDICATION`, `CLINICAL_AUDIT`, `DEFER_CLINICAL_AUDIT`,
`AUDIT_CONVERGENCE`, `PACK_CONSOLIDATION*`, `AUTO_SUBMIT_CLAIMS`,
`AUTO_GRADUATE`) are now vestigial *for this entrypoint*; they still configure the
tools above when those are run by hand. They are annotated as such in
`.env.example` and `terraform/secrets.tf`.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import NOTES_DIR, OUTPUT_DIR
from app.core.dates import parse_date_of_service
from app.core.logger import get_logger
from app.ingestion.pdf_parser import extract_from_pdf
from claude_coder.data_access import AuthoritativeSource
from claude_coder.models import Verdict
from claude_coder.pipeline import code_encounter, render

logger = get_logger("main")

#: Identity of the per-note JSON shape written below. It is deliberately NOT
#: `app.pipeline.CodingResult.model_dump()`'s shape — that model belongs to the
#: retired pipeline. Bump this when the payload changes so a reader can tell which
#: producer wrote a file.
RESULTS_SCHEMA = "claude_coder.run/1"

#: Retired-flag exit code: distinct from argparse's own 2 so a driver can tell
#: "you asked for something this entrypoint no longer does" from "bad usage".
EXIT_RETIRED_FLAG = 3


# ----------------------------------------------------------------- note inputs
def read_note(pdf_path: Path) -> dict:
    """The three inputs `code_encounter` needs, out of one PDF.

    `extract_from_pdf` is a TEXT-EXTRACTION utility (per-page verbatim
    transcription + patient metadata), not a coding-decision component — reusing
    it is not reusing the retired pipeline. Everything downstream of this
    function is `claude_coder`.

      note_text         the complete verbatim transcription. This exact string is
                        what every evidence span is anchored INTO, so it must be
                        the full text, never a reconstruction from selected
                        sections.
      date_of_service   parsed from the extracted metadata. `claude_coder` does
                        NOT extract a DOS itself — its extraction step is
                        code-free clinical facts — so the DOS is genuinely an
                        external, pre-parsed argument. Unparseable/absent stays
                        None: the release gates fail closed on it rather than
                        letting anything guess a service date.
      document_version  the immutable identity of the SOURCE document (the PDF's
                        own sha256), which is what evidence-span ids are salted
                        with. Falls back to None -> the pipeline salts with the
                        text hash instead.
    """
    extraction = extract_from_pdf(pdf_path)
    sections = extraction.get("sections") or {}
    note_text = str(sections.get("full_text") or "")
    if not note_text.strip():
        raise ValueError(
            f"PDF text extraction produced no text for {pdf_path.name}; refusing to "
            f"code an empty note")
    metadata = extraction.get("metadata") or {}
    dos = parse_date_of_service(metadata)
    integrity = extraction.get("note_integrity") or {}
    document_version = str(integrity.get("source_pdf_sha256") or "").strip() or None
    return {
        "note_text": note_text,
        "date_of_service": dos.isoformat() if dos else None,
        "document_version": document_version,
        "patient_metadata": metadata,
    }


def load_billing_context(path: str | None) -> dict | None:
    """Structured encounter context (billing entity + participant roster).

    Same file and same schema as `python -m claude_coder.cli --billing-context`;
    the pipeline's extractor validates it strictly and fails closed on a
    malformed roster. None is the correct default when a deployment has no
    reviewed roster: actor ownership then resolves UNKNOWN, which holds every
    claim line rather than assuming the billing entity performed the service.
    That hold is intentional and visible — see the warning logged by `main`.
    """
    if not path:
        return None
    with open(path) as fh:
        context = json.load(fh)
    if not isinstance(context, dict):
        raise ValueError(f"--billing-context {path}: file must contain a JSON object")
    return context


# ---------------------------------------------------------------- note outputs
def _line_payload(line) -> dict:
    chosen = line.chosen
    return {
        "system": chosen.system if chosen else None,
        "code": chosen.code if chosen else None,
        "descriptor": chosen.descriptor if chosen else None,
        "modifiers": list(line.modifiers),
        "units": line.units,
        "kind": line.fact.kind.value,
        "subject": line.fact.description,
        "method": line.method.value,
        "rationale": line.rationale,
        "excluded_reason": line.excluded_reason,
        "authority": chosen.authority if chosen else {},
        "evidence": [s.text for s in line.fact.evidence],
    }


def result_payload(result, *, pdf_path: Path, document_version: str | None) -> dict:
    """The per-note artifact.

    Deliberately explicit about the two things a reader most needs and most
    easily gets wrong:

      `releasable` is the ONLY field that says a claim may be billed without a
      human. It requires BOTH an AUTO_READY verdict AND a certificate — a
      certificate is never built when the data fingerprint, the release
      evidence, or the terminal durable write failed, so an AUTO_READY verdict
      with no certificate is not a release.

      `claim_lines` holds only lines that are actually billable (resolved,
      billable, not excluded); everything else stays under `other_lines` for the
      audit trail instead of being dropped.
    """
    # Identity, not equality: ResolvedLine is a dataclass, so two genuinely
    # distinct lines that happen to carry equal field values would compare equal
    # and silently disappear from `other_lines` under an `in` test.
    billable_ids = {id(ln) for ln in result.billable_lines}
    return {
        "schema": RESULTS_SCHEMA,
        "produced_by": "claude_coder.pipeline.code_encounter",
        "document_id": result.encounter_id,
        "source_pdf": pdf_path.name,
        "document_version": document_version,
        "date_of_service": result.date_of_service,
        "processed": True,
        "error": None,
        "verdict": result.verdict.value,
        "destination": result.destination.value if result.destination else None,
        "control_mode": result.control_mode,
        "releasable": bool(result.verdict is Verdict.AUTO_READY and result.certificate),
        "claim_lines": [_line_payload(ln) for ln in result.billable_lines],
        "other_lines": [_line_payload(ln) for ln in result.lines
                        if id(ln) not in billable_ids],
        "gates": [{"name": g.name, "outcome": g.outcome.value, "detail": g.detail,
                   "authority": g.authority, "retryable": g.retryable}
                  for g in result.gates],
        "notes": list(result.notes),
        "routing": list(result.routing),
        "recommendations": list(result.recommendations),
        "necessity_support": list(result.necessity_support),
        "audit_record_hashes": list(result.audit_record_hashes),
        "certificate": result.certificate,
        "certificate_sha256": (result.certificate or {}).get("certificate_sha256"),
        # The explainability surface, verbatim, so the JSON artifact is readable
        # without re-running anything.
        "audit_trail": render(result),
    }


def failure_payload(pdf_path: Path, exc: Exception) -> dict:
    """A note that could not be processed at all still gets an artifact.

    Writing nothing would leave a stale success from an earlier run in place and
    make a failed note invisible in OUTPUT_DIR — an empty success by omission.
    """
    return {
        "schema": RESULTS_SCHEMA,
        "produced_by": "claude_coder.pipeline.code_encounter",
        "document_id": pdf_path.stem,
        "source_pdf": pdf_path.name,
        "processed": False,
        "error": f"{type(exc).__name__}: {exc}",
        "verdict": None,
        "destination": None,
        "releasable": False,
        "claim_lines": [],
    }


def write_result(payload: dict, pdf_path: Path) -> Path:
    output_file = OUTPUT_DIR / f"{pdf_path.stem}_results.json"
    with open(output_file, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return output_file


def rebuild_all_results() -> None:
    """Combined output — rebuilt from the per-note files on disk, not from this
    batch's in-memory list, so a resumed batch (`--start-at`) or a partial
    failure never overwrites the corpus aggregate with a subset."""
    combined = []
    for path in sorted(OUTPUT_DIR.glob("*_results.json")):
        if path.name == "all_results.json":
            continue
        try:
            combined.append(json.loads(path.read_text()))
        except Exception as exc:
            logger.warning(f"  all_results: skipped unreadable {path.name} ({exc})")
    with open(OUTPUT_DIR / "all_results.json", "w") as fh:
        json.dump(combined, fh, indent=2, default=str)


# ------------------------------------------------------------- note selection
def select_notes(args) -> list[Path]:
    """The notes this invocation should process, or [] with the SPECIFIC reason
    already logged — a caller that reported a generic "no notes found" over the
    top of "--only: no PDF found for [...]" would bury the real cause."""
    if args.note:
        note_path = Path(args.note)
        if not note_path.exists():
            note_path = NOTES_DIR / args.note
        if not note_path.exists():
            logger.error(f"Note not found: {args.note}")
            return []
        return [note_path]

    note_files = sorted(NOTES_DIR.glob("*.pdf"))
    if not note_files:
        logger.error(f"No clinical notes found in {NOTES_DIR}")
        return []
    if args.start_at:
        before = len(note_files)
        note_files = [p for p in note_files if p.stem >= args.start_at]
        logger.info(f"--start-at {args.start_at}: skipping {before - len(note_files)} "
                    f"already-completed note(s)")
    if args.end_at:
        before = len(note_files)
        # Inclusive: '--end-at 010' must include 010's own file — '010_samuel…'
        # compares GREATER than the bare prefix '010', so a strict < silently
        # dropped the last requested note (measured live: two batches in a row).
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
            return []
        logger.info(f"--only: processing exactly {len(note_files)} note(s)")
    return note_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Podiatry Medical Coding System")
    parser.add_argument("--note", type=str,
                        help="Process a single PDF note (filename or path)")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Force rebuild the Qdrant vector collections")
    parser.add_argument("--setup-only", action="store_true",
                        help="Build/load all dependencies and exit — process no notes")
    # Defaults from the environment because the deployed path takes no flags:
    # `docker compose run app python run.py` (and the note-watcher's
    # process-notes.sh) would otherwise have no way to supply a roster without
    # editing a generated helper script. Documented in .env.example.
    parser.add_argument("--billing-context", type=str,
                        default=os.getenv("BILLING_CONTEXT_FILE", ""),
                        help="Path to a JSON file declaring billing_entity_id and the "
                             "encounter's participant roster (env: BILLING_CONTEXT_FILE). "
                             "Without it, actor ownership is UNKNOWN and every claim "
                             "line holds.")
    parser.add_argument("--start-at", type=str, default="",
                        help="Skip notes sorting before this stem/filename prefix — "
                             "resume a batch without redoing completed notes")
    parser.add_argument("--end-at", type=str, default="",
                        help="Skip notes sorting after this stem/filename prefix "
                             "(inclusive) — bound a batch to a subset")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated note stems: process exactly these notes")
    # --- retired flags: parsed only so their callers fail loudly (F6-R4-A1) ---
    parser.add_argument("--no-cache", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--consistency", type=int, default=1,
                        help=argparse.SUPPRESS)
    parser.add_argument("--consistency-workers", type=int, default=1,
                        help=argparse.SUPPRESS)
    return parser


def reject_retired_flags(args) -> int | None:
    """Retired-flag handling, stated out loud rather than silently ignored.

    `--no-cache` was a switch on `app.pipeline`'s result cache. `claude_coder`
    has no result cache — every encounter is coded fresh — so the flag is a
    no-op that still means what the caller wanted. It warns and continues.

    `--consistency`/`--consistency-workers` asked for N independent runs whose
    disagreements drive the retired growth loop. Honouring the flag by running
    once would silently give a caller a fraction of the assurance it requested,
    so it is refused instead.
    """
    if args.no_cache:
        logger.warning(
            "--no-cache is a no-op: the result cache belonged to the retired "
            "app.pipeline; claude_coder codes every encounter fresh already.")
    if args.consistency > 1 or args.consistency_workers > 1:
        logger.error(
            f"--consistency={args.consistency} "
            f"--consistency-workers={args.consistency_workers} is retired "
            f"(issue #6, F6-R4-A1). The deployed entrypoint now runs "
            f"claude_coder.pipeline.code_encounter, which replaces the N-run "
            f"self-consistency comparison with built-in propose-then-verify plus "
            f"independent cross-model corroboration. Running once while you asked for "
            f"multiple independent runs would silently give you less assurance than "
            f"you requested, so this run is refused. Drop the flags to process the "
            f"batch.")
        return EXIT_RETIRED_FLAG
    return None


# ------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    retired = reject_retired_flags(args)
    if retired is not None:
        return retired

    from app.core.config import CLAUDE_MODEL, LLM_PROVIDER, OPENAI_MODEL
    active_model = CLAUDE_MODEL if LLM_PROVIDER == "claude" else OPENAI_MODEL
    logger.info("=" * 70)
    logger.info("PODIATRY MEDICAL CODING SYSTEM")
    logger.info(f"Pipeline: PDF text → claude_coder (extract → eligibility → "
                f"resolve → gates → certificate) [{LLM_PROVIDER}:{active_model}]")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    try:
        billing_context = load_billing_context(args.billing_context)
    except Exception as exc:
        logger.error(f"--billing-context could not be read: {exc}")
        return 1
    if billing_context is None:
        logger.warning(
            "No --billing-context supplied: actor ownership resolves UNKNOWN, so "
            "every claim line will HOLD before retrieval. This is fail-closed by "
            "design — supply the reviewed participant roster to release claims.")

    if args.setup_only:
        AuthoritativeSource().prepare(force_rebuild_index=args.rebuild_index)
        logger.info("\n--setup-only: dependencies loaded, no notes processed. Exiting.")
        return 0

    # Note selection BEFORE dependency loading, deliberately: `prepare()` can cost
    # an hour on a cold index, and a typo in --note/--only must not be reported
    # only after paying for it.
    note_files = select_notes(args)
    if not note_files:
        return 1

    # ONE authoritative source for the whole batch: it caches the vector store,
    # the reference tables and every authoritative file, so per-note construction
    # would repay a multi-minute load on every note.
    source = AuthoritativeSource()
    source.prepare(force_rebuild_index=args.rebuild_index)

    logger.info(f"\nProcessing {len(note_files)} clinical note(s)\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    payloads = []
    failures = 0
    for pdf_path in note_files:
        logger.info("=" * 70)
        logger.info(f"PROCESSING: {pdf_path.name}")
        logger.info("=" * 70)
        try:
            note = read_note(pdf_path)
            if not note["date_of_service"]:
                logger.warning(
                    f"  {pdf_path.name}: no parseable date of service in the note; "
                    f"date-dependent gates will hold this encounter")
            logger.info(f"  DOS: {note['date_of_service']} | "
                        f"document_version: {note['document_version']}")
            result = code_encounter(
                pdf_path.stem,
                note["note_text"],
                note["date_of_service"],
                source=source,
                billing_context=billing_context,
                document_version=note["document_version"],
            )
            payload = result_payload(result, pdf_path=pdf_path,
                                     document_version=note["document_version"])
            logger.info(f"  VERDICT: {payload['verdict']} → {payload['destination']} "
                        f"| {len(payload['claim_lines'])} claim line(s) "
                        f"| releasable={payload['releasable']}")
        except Exception as exc:
            failures += 1
            logger.error(f"FAILED: {pdf_path.name} — {type(exc).__name__}: {exc}")
            import traceback
            logger.error(traceback.format_exc())
            payload = failure_payload(pdf_path, exc)
        payloads.append(payload)
        # Its own boundary: a note that coded fine but whose artifact cannot be
        # written has NOT been delivered, and must be counted as a failure rather
        # than aborting the remaining notes on an I/O problem.
        try:
            logger.info(f"  Saved → {write_result(payload, pdf_path).name}")
        except Exception as exc:
            if payload.get("processed"):
                failures += 1
            logger.error(f"FAILED to write results for {pdf_path.name} — "
                         f"{type(exc).__name__}: {exc}")

    try:
        rebuild_all_results()
    except Exception as exc:   # the aggregate is derived; never lose the summary to it
        logger.error(f"all_results.json could not be rebuilt — "
                     f"{type(exc).__name__}: {exc}")

    logger.info(f"\n{'=' * 70}")
    logger.info("BATCH COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"Total: {len(payloads)} | Processed: {len(payloads) - failures} "
                f"| Failed: {failures}")
    destinations: dict[str, int] = {}
    for payload in payloads:
        key = payload.get("destination") or ("PROCESSING_FAILURE"
                                             if not payload.get("processed") else "UNKNOWN")
        destinations[key] = destinations.get(key, 0) + 1
    logger.info("Destinations: " + " | ".join(
        f"{name}: {count}" for name, count in sorted(destinations.items())))
    logger.info(f"Releasable (AUTO_READY + certificate): "
                f"{sum(1 for p in payloads if p.get('releasable'))}")
    logger.info(f"Output → {OUTPUT_DIR}")

    # A batch in which NOTHING could be processed is an operational failure, not
    # an empty success — exit non-zero so a caller/`set -e` script sees it. A
    # partial failure keeps exit 0 (each failure is logged and has its own
    # artifact), matching the prior entrypoint's contract for resumable batches.
    if failures and failures == len(payloads):
        logger.error("Every note in this batch failed to process.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
