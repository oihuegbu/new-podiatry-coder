"""Transaction-safe persistence for independent coding runs.

The main result is the manifest for a note.  New run generations are written
under unique names before that manifest is atomically replaced, so a crash can
never make a previous result accidentally consume a partial or stale run set.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable
from uuid import uuid4


def _safe_document_id(document_id: str) -> str:
    value = str(document_id or "")
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError("document_id must be a non-empty basename")
    return value


def atomic_write_json(path: Path, payload: object) -> None:
    """Durably replace *path* with a complete JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def persist_runs(results_dir: Path, document_id: str,
                 runs: Iterable[dict]) -> list[str]:
    """Write one complete, uniquely named generation and return its manifest."""
    document_id = _safe_document_id(document_id)
    materialized = list(runs)
    if not materialized:
        raise ValueError("at least one consistency run is required")
    runs_dir = Path(results_dir) / "consistency_runs"
    generation = uuid4().hex
    references = [
        f"{document_id}__{generation}_run{index}.json"
        for index in range(1, len(materialized) + 1)
    ]
    written: list[Path] = []
    try:
        for reference, payload in zip(references, materialized):
            path = runs_dir / reference
            atomic_write_json(path, payload)
            written.append(path)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return references


def _validated_manifest(document_id: str, report: dict) -> list[str] | None:
    references = report.get("run_files")
    if references is None:
        return None
    if not isinstance(references, list) or not references:
        raise ValueError("consistency.run_files must be a non-empty list")
    expected_count = report.get("runs")
    if not isinstance(expected_count, int) or expected_count != len(references):
        raise ValueError("consistency run manifest count does not match report")
    pattern = re.compile(
        rf"^{re.escape(document_id)}__(?P<generation>[0-9a-f]{{32}})"
        rf"_run(?P<index>[1-9][0-9]*)\.json$")
    parsed = []
    for reference in references:
        match = pattern.fullmatch(reference) if isinstance(reference, str) else None
        if (not isinstance(reference, str)
                or Path(reference).name != reference
                or match is None):
            raise ValueError(f"invalid consistency run reference: {reference!r}")
        parsed.append((match.group("generation"), int(match.group("index"))))
    if len(set(references)) != len(references):
        raise ValueError("consistency run manifest contains duplicate references")
    if (len({generation for generation, _ in parsed}) != 1
            or [index for _, index in parsed]
            != list(range(1, len(references) + 1))):
        raise ValueError("consistency run manifest must name one ordered generation")
    return references


def load_runs(document_id: str, results_dir: Path,
              main: dict | None = None) -> list[dict]:
    """Load exactly the generation referenced by the main result.

    Results created before manifests were introduced retain a bounded legacy
    fallback.  Once a manifest exists, missing or malformed members fail
    loudly instead of silently mixing generations.
    """
    document_id = _safe_document_id(document_id)
    results_dir = Path(results_dir)
    if main is None:
        main_path = results_dir / f"{document_id}_results.json"
        if main_path.exists():
            candidate = json.loads(main_path.read_text(encoding="utf-8"))
            main = candidate if isinstance(candidate, dict) else None

    report = (main or {}).get("consistency")
    if isinstance(report, dict):
        references = _validated_manifest(document_id, report)
        if references is not None:
            runs = []
            for reference in references:
                path = results_dir / "consistency_runs" / reference
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError(f"consistency run is not an object: {reference}")
                runs.append(payload)
            return runs

    runs = []
    runs_dir = results_dir / "consistency_runs"
    for index in range(1, 10):
        path = runs_dir / f"{document_id}_run{index}.json"
        if not path.exists():
            break
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"legacy consistency run is not an object: {path.name}")
        runs.append(payload)
    return runs


def inherit_run_metadata(new_report: dict, previous_report: dict | None) -> dict:
    """Keep storage/execution metadata when tools recompute comparisons."""
    if not isinstance(previous_report, dict):
        return new_report
    for field in ("run_files", "execution_strategy"):
        if field in previous_report:
            new_report[field] = previous_report[field]
    return new_report


def discard_run_files(results_dir: Path, references: Iterable[str]) -> None:
    """Remove an uncommitted generation using already-validated basenames."""
    runs_dir = Path(results_dir) / "consistency_runs"
    for reference in references:
        if isinstance(reference, str) and Path(reference).name == reference:
            (runs_dir / reference).unlink(missing_ok=True)


def prune_obsolete_runs(results_dir: Path, document_id: str,
                        keep: Iterable[str]) -> None:
    """Remove older generations only after the new main manifest is durable."""
    document_id = _safe_document_id(document_id)
    keep_set = set(keep)
    runs_dir = Path(results_dir) / "consistency_runs"
    if not runs_dir.exists():
        return
    pattern = re.compile(
        rf"^{re.escape(document_id)}(?:__[0-9a-f]{{32}})?_run[1-9][0-9]*\.json$")
    for path in runs_dir.iterdir():
        if path.is_file() and pattern.fullmatch(path.name) and path.name not in keep_set:
            path.unlink()
