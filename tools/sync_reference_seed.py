#!/usr/bin/env python3
"""Atomically reconcile versioned reference inputs into the app-data volume.

Docker initializes a named volume from an image only once.  Without an
explicit reconciliation step, later releases run new application code against
the first image's terminology, source requirements, and code JSON forever.
This tool copies only paths declared by the versioned seed manifest. Runtime
databases, signed scopes, feedback, policy, and claim registries are outside
that manifest and are never touched.
"""

from __future__ import annotations

import argparse
import fcntl
import filecmp
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath


STATE_FILE = ".reference_seed_state.json"
LOCK_FILE = ".reference_seed.lock"


class ReferenceSeedError(RuntimeError):
    pass


def _relative(value: object) -> Path:
    text = str(value or "").strip()
    pure = PurePosixPath(text)
    if (not text or pure.is_absolute() or ".." in pure.parts
            or any(part in {"", "."} for part in pure.parts)):
        raise ReferenceSeedError(f"unsafe managed path: {text!r}")
    return Path(*pure.parts)


def _load_manifest(seed_root: Path, manifest_path: Path) \
        -> tuple[list[Path], list[Path], dict]:
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise ReferenceSeedError(f"seed manifest is unavailable: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise ReferenceSeedError("unsupported seed manifest schema")
    roots = [_relative(value) for value in manifest.get("managed_roots") or []]
    files = [_relative(value) for value in manifest.get("managed_files") or []]
    runtime_owned = {
        _relative(value) for value in manifest.get("runtime_owned_files") or []
    }
    if not roots and not files:
        raise ReferenceSeedError("seed manifest has no managed paths")

    expanded: set[Path] = set()
    for relative in roots:
        source = seed_root / relative
        if source.is_symlink() or not source.is_dir():
            raise ReferenceSeedError(f"managed root is not a directory: {relative}")
        for candidate in source.rglob("*"):
            if candidate.is_symlink():
                raise ReferenceSeedError(
                    f"managed source must not contain symlinks: {candidate}")
            if candidate.is_file():
                expanded.add(candidate.relative_to(seed_root))
    for relative in files:
        source = seed_root / relative
        if source.is_symlink() or not source.is_file():
            raise ReferenceSeedError(f"managed file is unavailable: {relative}")
        expanded.add(relative)
    unknown_runtime_paths = runtime_owned - expanded
    if unknown_runtime_paths:
        raise ReferenceSeedError(
            "runtime-owned files must be declared managed source files")
    managed = expanded - runtime_owned
    return (sorted(managed, key=lambda path: path.as_posix()),
            sorted(runtime_owned, key=lambda path: path.as_posix()), manifest)


def _fingerprint(seed_root: Path, paths: list[Path], manifest: dict) -> str:
    digest = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":")).encode())
    for relative in paths:
        digest.update(b"\0")
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        with (seed_root / relative).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_state(destination_root: Path) -> dict:
    try:
        value = json.loads((destination_root / STATE_FILE).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _atomic_copy(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        raise ReferenceSeedError(f"managed destination is a symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output, 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        shutil.copystat(source, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_state(destination_root: Path, state: dict) -> None:
    path = destination_root / STATE_FILE
    descriptor, temporary = tempfile.mkstemp(prefix=f".{STATE_FILE}.",
                                             dir=destination_root)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(state, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def synchronize(seed_root: Path, destination_root: Path,
                manifest_path: Path | None = None) -> dict:
    seed_root = seed_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    if destination_root.is_symlink():
        raise ReferenceSeedError("destination root must not be a symlink")
    destination_root = destination_root.resolve()
    lock_path = destination_root / LOCK_FILE
    if lock_path.is_symlink():
        raise ReferenceSeedError("seed lock must not be a symlink")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _synchronize_locked(seed_root, destination_root, manifest_path)


def _synchronize_locked(seed_root: Path, destination_root: Path,
                        manifest_path: Path | None) -> dict:
    manifest_path = (manifest_path or seed_root / "reference_seed_manifest.json")
    paths, runtime_owned, manifest = _load_manifest(seed_root, manifest_path)
    fingerprint = _fingerprint(seed_root, paths + runtime_owned, manifest)
    previous = _read_state(destination_root)
    names = [path.as_posix() for path in paths]
    runtime_names = {path.as_posix() for path in runtime_owned}
    previous_managed = set(previous.get("managed_files") or [])
    destinations_intact = all(
        (destination_root / relative).is_file()
        and not (destination_root / relative).is_symlink()
        for relative in paths + runtime_owned)
    if (previous.get("fingerprint") == fingerprint
            and previous.get("managed_files") == names
            and destinations_intact):
        return {"changed": False, "fingerprint": fingerprint,
                "files": len(paths) + len(runtime_owned)}

    current = set(names)
    for old_name in previous_managed:
        old_relative = _relative(old_name)
        if old_relative.as_posix() in current | runtime_names:
            continue
        old_path = destination_root / old_relative
        if old_path.is_symlink():
            raise ReferenceSeedError(
                f"retired managed destination is a symlink: {old_path}")
        if old_path.is_file():
            old_path.unlink()
    for relative in paths:
        source = seed_root / relative
        destination = destination_root / relative
        if destination.is_symlink():
            raise ReferenceSeedError(
                f"managed destination is a symlink: {destination}")
        if destination.exists() and not destination.is_file():
            raise ReferenceSeedError(
                f"managed destination is not a file: {destination}")
        # An unrelated release input changing must not rewrite every managed
        # file. ComplianceDataStore keys re-ingestion to file metadata; a
        # blind rewrite would clear additive NCCI/MUE history even when those
        # authoritative bytes had not changed.
        if destination.is_file() and filecmp.cmp(
                source, destination, shallow=False):
            continue
        _atomic_copy(source, destination)
    for relative in runtime_owned:
        destination = destination_root / relative
        if destination.is_symlink():
            raise ReferenceSeedError(
                f"runtime-owned destination is a symlink: {destination}")
        if destination.exists() and not destination.is_file():
            raise ReferenceSeedError(
                f"runtime-owned destination is not a file: {destination}")
        # These sources are autonomously refreshed in place or retained as
        # additive database history. Seed them only when a volume lacks the
        # file; a later image must never roll a fresher runtime source back.
        if not destination.exists():
            _atomic_copy(seed_root / relative, destination)
        elif relative.as_posix() in previous_managed:
            # A prior synchronizer version incorrectly managed this runtime
            # source. Signal its datastore fingerprint once so the datastore
            # can discard provenance for history that the old rewrite may
            # have erased and autonomously reacquire the governing snapshot.
            now = time.time_ns()
            os.utime(destination, ns=(now, now))
    _write_state(destination_root, {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "managed_files": names,
    })
    return {"changed": True, "fingerprint": fingerprint,
            "files": len(paths) + len(runtime_owned)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_root", type=Path)
    parser.add_argument("destination_root", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = synchronize(args.seed_root, args.destination_root, args.manifest)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
