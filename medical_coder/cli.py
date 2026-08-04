from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from .certificate import build_certificate
from .compiler import SourceCompiler
from .decision import DecisionEngine
from .refresh import RefreshRunner
from .serde import load_candidates, load_context, load_evidence, read_json
from .scope import AutonomyScope
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)

