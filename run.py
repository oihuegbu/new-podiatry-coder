#!/usr/bin/env python3
"""Podiatry Medical Coding System — the deployed batch entrypoint.

    PDF ─► verbatim note text (app.ingestion.pdf_parser — a text-extraction utility)
        ─► claude_coder.pipeline.code_encounter   ◄── THE note→code decision engine
        ─► ClaimBundle (app.contracts.claim_bundle) ◄── THE claim contract
        ─► {stem}_results.json + all_results.json in OUTPUT_DIR

================================================================================
THE ARTIFACT THIS WRITES — issue #6 F6-R4-A1, product directive §5
================================================================================
Every per-note file is one canonical `ClaimBundle`: a strict, versioned schema
owned by NEITHER pipeline implementation (`app/contracts/`), read by the claims
registry, by release-readiness verification and by the 837P builder. Round 6's
`claude_coder.run/1` shape is no longer written — this entrypoint emitted it
while the retained claim path read the retired `app.pipeline` shape, so an
AUTO_READY encounter arrived at the registry with no diagnosis and no service
lines and was refused as "pipeline did not succeed". One producer, one contract,
one consumer vocabulary is the fix; a translation shim on either side is not.

The bundle also carries the encounter's billing CONTEXT, which this file used to
obtain (`read_note` extracts patient metadata) and then discard. It is resolved
through an `EncounterContextProvider` (`--encounter-context`), never inferred
from the note: without a configured provider the context is UNRESOLVED and every
bundle holds, visibly, in the artifact itself.

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
invoked from the deployed entrypoint.

CAVEAT, stated rather than left to be rediscovered: most of them still read a
result file in the retired `app.pipeline` claim shape, so run by hand against a
`ClaimBundle` artifact they will see an EMPTY claim rather than an error. That is
the same defect class as F6-R4-A1 and it is not fixed here — none of them is on
the path from a certified result to an 837P. The exact set is frozen by
`tests/test_claim_bundle_e2e.test_no_new_module_reads_a_result_in_the_retired_claim_shape`
so it can only shrink. `tools/claims_registry.py` and `tools/claim_submitter.py`
— the retained claim path — ARE migrated.

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
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.contracts.claim_bundle import (
    SCHEMA_ID, SCHEMA_VERSION, AuthorityBinding, SourceDocument,
    bundle_from_coding_result, failure_bundle,
)
from app.contracts.encounter_context import build_provider
from app.core.config import NOTES_DIR, OUTPUT_DIR
from app.core.dates import parse_date_of_service
from app.core.logger import get_logger
from app.ingestion.pdf_parser import extract_from_pdf
from app.contracts.source_evidence import reconcile_service_date
from app.ingestion.source_evidence import (
    IndependentVisionReader, compile_source_evidence)
from claude_coder.data_access import AuthoritativeSource
from claude_coder.pipeline import code_encounter, render

logger = get_logger("main")

#: Identity of the per-note JSON shape written below — the CANONICAL claim
#: contract (`app/contracts/claim_bundle.py`), shared with the claims registry,
#: readiness verification and the 837P builder.
#:
#: It replaces round 6's `claude_coder.run/1`, which this entrypoint no longer
#: writes. That artifact's independent value was its audit surface (rendered
#: trail, non-billed lines, routing, recommendations); all of it is now a
#: section of the bundle (`ClaimBundle.audit`). Emitting BOTH would put two
#: shapes of the same note on disk — which is the defect finding F6-R4-A1 was
#: about, reintroduced one directory later. Old `claude_coder.run/1` and
#: `app.pipeline` files already on disk stay readable through
#: `app/contracts/legacy_adapter.py`.
RESULTS_SCHEMA = f"{SCHEMA_ID}/{SCHEMA_VERSION}"

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
      date_of_service   parsed from the extracted metadata. This is a CANDIDATE
                        reading, never the claim's date: it is one model's read of
                        one metadata field. `service_date_evidence` below is what
                        proves it, and `EncounterContextProvider.resolve()` is what
                        BINDS the claim's actual date of service (issue #6 F7-R4).
      service_date_evidence
                        that candidate located on a page of the ORIGINAL document
                        and reconciled against an independent reading of that page,
                        by the same compiler that reconciles every clinical
                        quotation. A misread date now either fails to appear on any
                        page (NOT_LOCATED) or is read differently by the independent
                        channel (DISAGREED) — and either way it cannot bind.
      document_version  the immutable identity of the SOURCE document (the PDF's
                        own sha256), which is what evidence-span ids are salted
                        with. Falls back to None -> the pipeline salts with the
                        text hash instead.
    """
    extraction = extract_from_pdf(pdf_path)
    # THE ORIGINAL DOCUMENT, read by more than one channel (issue #6 F6-R6-A). The
    # vision transcription above is one candidate reading of the PDF; on its own it is
    # the authority against which its own correctness would be "proven", which proves
    # nothing. The compiler binds the PDF's digest, each rendered page image's digest,
    # and the document's own embedded text layer with word boxes as an independent,
    # deterministic second reading — and refuses (raises) rather than returning a
    # single-channel document that would look checked and is not.
    source_evidence = compile_source_evidence(pdf_path, extraction)
    sections = extraction.get("sections") or {}
    note_text = str(sections.get("full_text") or "")
    if not note_text.strip():
        raise ValueError(
            f"PDF text extraction produced no text for {pdf_path.name}; refusing to "
            f"code an empty note")
    metadata = extraction.get("metadata") or {}
    dos = parse_date_of_service(metadata)
    # THE DATE OF SERVICE IS A CODE-CHANGING FACT (issue #6 F7-R4). It selects the
    # coverage in force, the billing affiliation, the authorization window and the
    # effective code edition, and it is the claim's own service date — yet it
    # arrived as a structured metadata field of the SAME transcription every other
    # fact is checked against, and nothing ever checked it. It is now proven exactly
    # like every quotation: located on a page of the original document, then
    # reconciled against an independent reading of that page.
    # Reconciled against the channels the compiler already built (the document's
    # own text layer). It is deliberately NOT escalated to the paid
    # `IndependentVisionReader` here: that reader writes ONE fixed channel id into
    # the document, and the pipeline escalates the same channel later for the
    # quotations behind released lines — a second writer at this point would make
    # the later `with_channel()` refuse and silently degrade the clinical
    # reconciliation. An image-only page carrying the date therefore holds the
    # encounter as SYSTEM work (obtain a channel), which is the correct fail-closed
    # answer even though it is not the maximally autonomous one.
    service_date_evidence = reconcile_service_date(
        source_evidence, dos.isoformat() if dos else "")
    integrity = extraction.get("note_integrity") or {}
    # The identity every evidence-span id is salted with, taken from the COMPILER's own
    # digest of the file rather than from the transcription's report of it (issue #6
    # F7-R5). The compiler has already refused a transcription whose claimed digest is
    # not the document's, so the two agree whenever both exist -- but the value that
    # travels onto the claim is the one that was computed from bytes, not asserted.
    document_version = str(source_evidence.document_sha256 or "").strip() or None
    page_count = integrity.get("page_count")
    return {
        "note_text": note_text,
        "date_of_service": dos.isoformat() if dos else None,
        "document_version": document_version,
        # Source-document identity carried through to the ClaimBundle. The
        # finding F6-R4-A1 named this function specifically: it obtained the
        # patient metadata and the result payload then discarded it, so the
        # retained claim path had no demographics to build a claim from. It is
        # now carried into the bundle's encounter context (as CORROBORATION —
        # see app/contracts/encounter_context.py) instead of being dropped.
        "extracted_text_sha256": str(
            integrity.get("extracted_text_sha256") or "").strip() or None,
        "page_count": int(page_count) if isinstance(page_count, int) else None,
        "patient_metadata": metadata,
        "source_evidence": source_evidence,
        "service_date_evidence": service_date_evidence,
        # The PAID second channel, for image-only pages the text layer cannot cover.
        # Constructed here (it needs the PDF) but INVOKED by the pipeline, and only for
        # the pages carrying a quotation behind a released line. It is BOUND to the
        # primary channel it must be independent of (issue #6 F7-R5): the reader then
        # establishes independence against the vendor that actually produced that
        # reading, and refuses -- loudly, before anything is paid for -- rather than
        # contributing a reading no reconciliation may credit.
        "source_reader": IndependentVisionReader(
            pdf_path, primary_channel=source_evidence.primary_channel),
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
def authority_binding(result, source) -> AuthorityBinding:
    """Which authoritative data and index the coder actually queried.

    Read from the CERTIFICATE first, deliberately: `source_identity.data` is the
    fingerprint the certificate was built over, so binding it here means the
    artifact and the certificate can never attest to different editions. Only
    when there is no certificate (a held encounter) is the live source asked
    directly, and a failure there stays empty rather than guessing — an empty
    authority binding is itself a release blocker in the contract.
    """
    identity = ((result.certificate or {}).get("source_identity") or {})
    fingerprint = identity.get("data") or {}
    if not fingerprint:
        try:
            fingerprint = source.data_fingerprint()
        except Exception as exc:
            logger.warning(f"  authoritative-data fingerprint unavailable "
                           f"({type(exc).__name__}: {exc}); the bundle will "
                           f"record no authority binding and cannot release")
            fingerprint = {}
    manifest = fingerprint.get("source_manifest") or {}
    return AuthorityBinding(
        data_fingerprint=str(fingerprint.get("fingerprint_sha256") or ""),
        source_manifest_fingerprint=str(manifest.get("manifest_sha256") or ""),
        source_manifest=manifest,
        index_checksum=str(fingerprint.get("codes_checksum") or ""),
        code_counts={k: int(v) for k, v in
                     (fingerprint.get("counts") or {}).items()},
        model_profiles=identity.get("models") or {},
    )


def build_bundle(result, *, pdf_path: Path, note: dict, context, source) -> dict:
    """The per-note artifact: one canonical `ClaimBundle`, serialized.

    Everything a downstream consumer needs travels here — ordered diagnoses,
    service lines with their units/modifiers/diagnosis pointers, the resolved
    (or explicitly unresolved) encounter context, every gate outcome, the
    authoritative-source identity, the certificate and the audit surface.
    `ClaimBundle.finalize()` stamps the claim/context fingerprints and writes
    the independently derived release blockers into the artifact, so the file
    itself says why a claim is not releasable instead of leaving a reader to
    infer it from an empty field.
    """
    bundle = bundle_from_coding_result(
        result,
        source_document=SourceDocument(
            filename=pdf_path.name,
            document_version=note["document_version"] or "",
            extracted_text_sha256=note["extracted_text_sha256"] or "",
            page_count=note["page_count"],
        ),
        context=context,
        authority=authority_binding(result, source),
        # The explainability surface, verbatim, so the JSON artifact is readable
        # without re-running anything.
        audit_trail=render(result),
        produced_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return bundle.to_payload()


def failure_payload(pdf_path: Path, exc: Exception) -> dict:
    """A note that could not be processed at all still gets an artifact.

    Writing nothing would leave a stale success from an earlier run in place and
    make a failed note invisible in OUTPUT_DIR — an empty success by omission.
    It is the SAME contract as a successful note (one shape per note, always),
    routed to SYSTEM_RETRY: a note that failed to process produced no coding
    judgement, so there is nothing for a human coder to review.
    """
    return failure_bundle(
        document_id=pdf_path.stem,
        filename=pdf_path.name,
        error=f"{type(exc).__name__}: {exc}",
        produced_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    ).to_payload()


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
    parser.add_argument("--encounter-context", type=str,
                        default=os.getenv("ENCOUNTER_CONTEXT_FILE", ""),
                        help="The encounter-context source resolving each encounter's "
                             "patient/subscriber/payer/rendering-provider/billing-"
                             "entity/facility/POS by stable identifier (env: "
                             "ENCOUNTER_CONTEXT_FILE). Either a path to a versioned "
                             "roster (schema encounter_context/2) or "
                             "'<adapter>:<locator>' for any registered adapter — see "
                             "app/contracts/encounter_context.py. Without it, no "
                             "encounter context is AUTHORITATIVELY resolved: the note's "
                             "own metadata travels with the claim as corroboration "
                             "only, and every bundle holds.")
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

    # The encounter-context provider is built (and its source READ) before any
    # note is processed: a malformed context file must stop the batch here,
    # loudly, rather than resolve UNRESOLVED on every note and look like a
    # deployment that simply has no roster. (Directive §2.)
    try:
        context_provider = build_provider(args.encounter_context or None)
        # PREFLIGHT, not a probe resolution: the source is read and validated
        # before any note is processed, so a malformed or unreachable context
        # source stops the batch HERE, loudly, instead of resolving UNRESOLVED
        # on every note and looking like a deployment that simply has no
        # roster. Asking for an encounter that does not exist (which is what
        # the previous empty-identifier `resolve()` call did) is a real
        # resolution request against whatever backend an adapter speaks to.
        preflight = context_provider.preflight()
    except Exception as exc:
        logger.error(f"--encounter-context could not be used: "
                     f"{type(exc).__name__}: {exc}")
        return 1
    if not args.encounter_context:
        logger.warning(
            "No --encounter-context supplied: no encounter's patient/subscriber/"
            "payer/rendering-provider/billing-entity/POS is AUTHORITATIVELY "
            "resolved, so every ClaimBundle holds with an UNRESOLVED context. The "
            "note's own extracted metadata still travels with the claim, as "
            "corroboration only.")
    else:
        logger.info(f"Encounter context source: {preflight}")
        duplicates = preflight.get("duplicate_identifiers") or {}
        if duplicates:
            # Not batch-fatal: only encounters resolving THROUGH a collided
            # identifier hold. Surfaced once here so the operator sees the
            # source defect without reading every note's holds.
            logger.warning(
                f"  encounter context source declares duplicate identifiers "
                f"{duplicates}; every encounter resolving through one of them "
                f"will hold")

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
                    f"  {pdf_path.name}: the transcription reports no parseable "
                    f"date of service; unless the encounter context source "
                    f"declares one, this encounter holds")
            logger.info(f"  DOS candidate: {note['date_of_service']} | "
                        f"document_version: {note['document_version']}")
            context = context_provider.resolve(
                encounter_id=pdf_path.stem,
                document_id=pdf_path.stem,
                date_of_service=note["date_of_service"],
                note_metadata=note["patient_metadata"],
                document_service_date=note["service_date_evidence"],
            )
            # ONE bound date of service, from here down. The candidate the
            # transcription proposed is corroboration; what every date-versioned
            # decision below is made against — coverage, affiliation,
            # authorization, code activity, NCCI/MUE, the graph, the certificate
            # and the claim's own service date — is this single value, with its
            # origin and its proof recorded in the artifact. (Issue #6 F7-R4.)
            binding = context.service_date
            dos = context.date_of_service or None
            logger.info(f"  DOS bound: {dos or 'NONE'} "
                        f"[{binding.source or 'unbound'}] | document reads "
                        f"{binding.documented_date or '-'} "
                        f"({binding.document_status or 'not compiled'})")
            logger.info(f"  Encounter context: {context.resolution.value} "
                        f"[{context.provider_id}"
                        + (f" {context.context_version}" if context.context_version
                           else "") + "]")
            for step in context.resolution_steps:
                logger.info(f"    {step.step}: {step.identifier or '-'} → "
                            f"{step.resolved_to or '-'} ({step.outcome})")
            for reason in (*context.unresolved, *context.conflicts):
                logger.info(f"    context hold: {reason}")
            result = code_encounter(
                pdf_path.stem,
                note["note_text"],
                dos,
                source=source,
                billing_context=billing_context,
                document_version=note["document_version"],
                source_evidence=note["source_evidence"],
                source_reader=note["source_reader"],
                service_date_binding=binding.model_dump(mode="json"),
            )
            payload = build_bundle(result, pdf_path=pdf_path, note=note,
                                   context=context, source=source)
            release = payload["release"]
            # `holds` is the CONSUMER-side re-derivation the contract stamps into
            # the artifact, not the producer's own flag: an empty holds list is
            # the only thing that means "billable without a human".
            logger.info(f"  VERDICT: {release['producer_verdict']} → "
                        f"{release['destination']} "
                        f"| {len(payload['diagnoses'])} diagnosis line(s) "
                        f"| {len(payload['service_lines'])} service line(s) "
                        f"| releasable={not release['holds']}")
            for hold in release["holds"]:
                logger.info(f"    hold: {hold}")
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
            if not payload.get("processing_error"):
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
        key = (payload.get("release") or {}).get("destination") or "UNKNOWN"
        destinations[key] = destinations.get(key, 0) + 1
    logger.info("Destinations: " + " | ".join(
        f"{name}: {count}" for name, count in sorted(destinations.items())))
    logger.info(f"Releasable (no ClaimBundle release blocker): "
                f"{sum(1 for p in payloads if not (p.get('release') or {}).get('holds'))}")
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
