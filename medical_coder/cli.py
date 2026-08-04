from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from .certificate import build_certificate
from .compiler import SourceCompiler
from .decision import DecisionEngine
from .documents import DocumentIngestor
from .audit import AuditStore
from .orchestrator import CodingWorkflow
from .refresh import RefreshRunner
from .serde import load_candidates, load_context, load_evidence, read_json
from .scope import AutonomyScope
from .snomed import prepare_snomed_us
from .terminology import TerminologySnapshot


def json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=json_default))


def compile_sources(args: argparse.Namespace) -> None:
    root = Path(args.repository_root).resolve()
    destination = SourceCompiler(root).compile(Path(args.pack), Path(args.output))
    emit({"snapshot_directory": str(destination), "snapshot_id": destination.name})


def search(args: argparse.Namespace) -> None:
    snapshot = TerminologySnapshot(Path(args.snapshot))
    evidence = load_evidence(Path(args.evidence))
    context = load_context(Path(args.context))
    candidates = snapshot.search(evidence, context.date_of_service, set(args.system), args.limit)
    emit([asdict(candidate) for candidate in candidates])


def evaluate(args: argparse.Namespace) -> None:
    snapshot = TerminologySnapshot(Path(args.snapshot))
    evidence = load_evidence(Path(args.evidence))
    context = load_context(Path(args.context))
    candidates = load_candidates(Path(args.candidates))
    coverage = {
        key: tuple(value)
        for key, value in (read_json(Path(args.coverage)) if args.coverage else {}).items()
    }
    units = {
        key: float(value)
        for key, value in (read_json(Path(args.units)) if args.units else {}).items()
    }
    scope_result = AutonomyScope(Path(args.scope)).evaluate(context) if args.scope else None
    required_capabilities = set(args.require_capability)
    if scope_result:
        required_capabilities.update(scope_result.required_capabilities)
    decisions = DecisionEngine(snapshot).evaluate(
        evidence,
        context,
        candidates,
        required_capabilities=required_capabilities,
        units=units,
        coverage_evidence=coverage,
        scope_eligible=scope_result.eligible if scope_result else True,
        scope_reason=scope_result.reason if scope_result else "no narrower autonomy scope configured",
    )
    certificate = build_certificate(args.encounter_id, snapshot.snapshot_id, evidence, context, decisions)
    emit(asdict(certificate))


def refresh_source(args: argparse.Namespace) -> None:
    root = Path(args.repository_root).resolve()
    result = RefreshRunner(root, Path(args.registry)).refresh(
        args.source,
        Path(args.pack).resolve(),
        Path(args.output).resolve(),
    )
    emit(result)


def refresh_sources(args: argparse.Namespace) -> None:
    root = Path(args.repository_root).resolve()
    result = RefreshRunner(root, Path(args.registry)).refresh_many(
        list(args.source), Path(args.pack).resolve(), Path(args.output).resolve()
    )
    emit(result)


def ingest_document(args: argparse.Namespace) -> None:
    document = DocumentIngestor(tuple(args.ocr_command) if args.ocr_command else None).ingest(
        Path(args.document), args.document_id
    )
    destination = document.write(Path(args.output))
    emit({"document_id": document.document_id, "source_sha256": document.source_sha256, "text_sha256": document.text_sha256, "output": str(destination)})


def code_note(args: argparse.Namespace) -> None:
    workflow = CodingWorkflow(
        Path(args.snapshot), Path(args.providers), Path(args.roles), Path(args.scope), Path(args.audit_database),
        ocr_command=tuple(args.ocr_command) if args.ocr_command else None,
    )
    result = workflow.code(args.encounter_id, Path(args.document), load_context(Path(args.context)))
    emit(asdict(result))


def verify_audit(args: argparse.Namespace) -> None:
    store = AuditStore(Path(args.audit_database))
    store.verify_chain(args.encounter_id)
    emit({"encounter_id": args.encounter_id, "valid": True, "result": store.result(args.encounter_id)})


def activate_snapshot(args: argparse.Namespace) -> None:
    snapshot = TerminologySnapshot(Path(args.snapshot))
    link = Path(args.link).resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(snapshot.directory)
    os.replace(temporary, link)
    emit({"snapshot_id": snapshot.snapshot_id, "active_link": str(link)})


def prepare_snomed(args: argparse.Namespace) -> None:
    metadata = prepare_snomed_us(Path(args.archive), Path(args.output))
    emit(metadata)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medical-coder")
    subcommands = parser.add_subparsers(dest="command", required=True)

    compile_parser = subcommands.add_parser("compile-sources")
    compile_parser.add_argument("--repository-root", default=".")
    compile_parser.add_argument("--pack", required=True)
    compile_parser.add_argument("--output", required=True)
    compile_parser.set_defaults(handler=compile_sources)

    search_parser = subcommands.add_parser("search")
    search_parser.add_argument("--snapshot", required=True)
    search_parser.add_argument("--evidence", required=True)
    search_parser.add_argument("--context", required=True)
    search_parser.add_argument("--system", action="append", required=True)
    search_parser.add_argument("--limit", type=int, default=20)
    search_parser.set_defaults(handler=search)

    evaluate_parser = subcommands.add_parser("evaluate")
    evaluate_parser.add_argument("--snapshot", required=True)
    evaluate_parser.add_argument("--encounter-id", required=True)
    evaluate_parser.add_argument("--evidence", required=True)
    evaluate_parser.add_argument("--context", required=True)
    evaluate_parser.add_argument("--candidates", required=True)
    evaluate_parser.add_argument("--units")
    evaluate_parser.add_argument("--coverage")
    evaluate_parser.add_argument("--require-capability", action="append", default=[])
    evaluate_parser.add_argument("--scope")
    evaluate_parser.set_defaults(handler=evaluate)

    refresh_parser = subcommands.add_parser("refresh-source")
    refresh_parser.add_argument("--repository-root", default=".")
    refresh_parser.add_argument("--registry", required=True)
    refresh_parser.add_argument("--source", required=True)
    refresh_parser.add_argument("--pack", required=True)
    refresh_parser.add_argument("--output", required=True)
    refresh_parser.set_defaults(handler=refresh_source)

    refresh_many_parser = subcommands.add_parser("refresh-sources")
    refresh_many_parser.add_argument("--repository-root", default=".")
    refresh_many_parser.add_argument("--registry", required=True)
    refresh_many_parser.add_argument("--source", action="append", required=True)
    refresh_many_parser.add_argument("--pack", required=True)
    refresh_many_parser.add_argument("--output", required=True)
    refresh_many_parser.set_defaults(handler=refresh_sources)

    ingest_parser = subcommands.add_parser("ingest-document")
    ingest_parser.add_argument("--document", required=True)
    ingest_parser.add_argument("--document-id")
    ingest_parser.add_argument("--output", required=True)
    ingest_parser.add_argument("--ocr-command", action="append")
    ingest_parser.set_defaults(handler=ingest_document)

    code_parser = subcommands.add_parser("code-note")
    code_parser.add_argument("--snapshot", required=True)
    code_parser.add_argument("--encounter-id", required=True)
    code_parser.add_argument("--document", required=True)
    code_parser.add_argument("--context", required=True)
    code_parser.add_argument("--providers", default="config/providers.json")
    code_parser.add_argument("--roles", default="config/system_roles.json")
    code_parser.add_argument("--scope", default="source-packs/scopes/medium-private-surgical-practice.json")
    code_parser.add_argument("--audit-database", default="var/audit.sqlite")
    code_parser.add_argument("--ocr-command", action="append")
    code_parser.set_defaults(handler=code_note)

    audit_parser = subcommands.add_parser("verify-audit")
    audit_parser.add_argument("--audit-database", default="var/audit.sqlite")
    audit_parser.add_argument("--encounter-id", required=True)
    audit_parser.set_defaults(handler=verify_audit)

    activate_parser = subcommands.add_parser("activate-snapshot")
    activate_parser.add_argument("--snapshot", required=True)
    activate_parser.add_argument("--link", required=True)
    activate_parser.set_defaults(handler=activate_snapshot)

    snomed_parser = subcommands.add_parser("prepare-snomed-us")
    snomed_parser.add_argument("--archive", required=True)
    snomed_parser.add_argument("--output", required=True)
    snomed_parser.set_defaults(handler=prepare_snomed)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)

