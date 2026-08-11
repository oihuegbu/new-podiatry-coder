"""Capability manifest — make authoritative-source availability and IDENTITY loud.

The retrieval/decision layers degrade quietly when an optional source (a synonym
layer, the index-terms file, the SNOMED crosswalk) is absent, and a REQUIRED source
going missing (a code table, an edit-policy file) would silently weaken the claim.
This module inspects each source by its CONFIGURED path and reports what loaded, what
is missing, and whether the deployment is degraded — surfaced on the release
certificate and enforced by a fail-closed gate (a missing REQUIRED source BLOCKS).

Presence is not identity. A manifest that records only path/presence/status cannot tell
two materially different files with the same row counts apart, so a release certificate
built on it cannot prove WHICH authoritative bytes produced the claim. Every present
source therefore also carries a CONTENT DIGEST (SHA-256 of the exact bytes), its size,
and the release/edition/effective window recorded for that source — read from the
release-manifest registry in `app.release.source_manifest`, which is the codebase's
existing provenance authority, rather than from a parallel list that can drift.
(Codex F6-R5.)

Agnostic: source identities come from the filename stem and the shared registry, and
paths come from central config — no medical code, code family, or domain list is
written here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "capability-manifest-v2"

# A digest recorded as `sha256:<64 lowercase hex>` — the same shape the release manifest
# already uses, so both attestations are comparable byte-for-byte.
_SHA256_PREFIX = "sha256:"


def _registry() -> dict[str, str]:
    """{resolved absolute path -> authoritative source_id} from the release registry."""
    from app.release.source_manifest import authoritative_paths
    out: dict[str, str] = {}
    for source_id, path in authoritative_paths().items():
        try:
            out.setdefault(str(Path(path).resolve()), source_id)
        except OSError:
            continue
    return out


def _source_locks() -> dict[str, dict]:
    """{resolved output path -> reviewed lock} for build products pinned by a lock file.

    A build product (e.g. a snapshot regenerated at image-build time) declares the exact
    bytes its reviewed lock expects. Drift from that lock is an integrity error, not a
    silent difference.
    """
    from app.core import config
    locks: dict[str, dict] = {}
    lock_dir = config.DATA_DIR / "sources"
    if not lock_dir.is_dir():
        return locks
    for lock_path in sorted(lock_dir.glob("*.lock.json")):
        try:
            lock = json.loads(lock_path.read_text())
            output = lock.get("output")
            if not output:
                continue
            resolved = str((config.BASE_DIR / output).resolve())
            locks[resolved] = {"lock": lock_path.name,
                               "output_sha256": str(lock.get("output_sha256") or ""),
                               "effective_from": str(lock.get("effective_from") or ""),
                               "source": str(lock.get("source") or "")}
        except Exception:
            # A lock we cannot read is itself an integrity problem; surface it as a
            # sentinel so `build_manifest` reports it rather than skipping silently.
            locks[f"unreadable:{lock_path.name}"] = {"unreadable": True}
    return locks


def _probe(path: Any, required: bool, role: str, registry: dict[str, str],
           locks: dict[str, dict],
           source_id: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """One source record plus any integrity errors it raised."""
    p = Path(path)
    errors: list[str] = []
    try:
        stat = p.stat()
        present = p.is_file() and stat.st_size > 0
    except OSError:
        stat, present = None, False
    try:
        resolved = str(p.resolve())
    except OSError:
        resolved = str(p)
    # A REQUIRED source carries the identity the release registry declared for it; an
    # optional aid is identified by the registry path map, falling back to its stem.
    source_id = source_id or registry.get(resolved, p.stem)
    record: dict[str, Any] = {
        "source": p.stem,
        "source_id": source_id,
        "required": bool(required),
        "present": present,
        "status": "loaded" if present else ("MISSING" if required else "absent-optional"),
        "role": role,
        "path": str(p),
        "bytes": int(stat.st_size) if (present and stat is not None) else 0,
        "sha256": None,
        "release": None,
    }
    if not present:
        return record, errors
    # Content identity. A present-but-undigestible source is an INTEGRITY ERROR: the claim
    # would otherwise be certified against bytes nobody can name.
    try:
        from app.release.source_manifest import sha256_file
        record["sha256"] = sha256_file(p)
    except Exception as exc:
        errors.append(f"{record['source']}: content digest unavailable ({exc})")
        return record, errors
    try:
        from app.release.source_manifest import release_metadata
        record["release"] = release_metadata(source_id)
    except Exception as exc:
        errors.append(f"{record['source']}: release metadata unavailable ({exc})")
    lock = locks.get(resolved)
    if lock:
        record["lock"] = lock["lock"]
        record["lock_effective_from"] = lock.get("effective_from", "")
        expected = lock.get("output_sha256", "")
        if expected and record["sha256"] != _SHA256_PREFIX + expected.lower():
            errors.append(
                f"{record['source']}: content does not match its reviewed lock "
                f"{lock['lock']} (source drifted from the pinned release)")
    return record, errors


def manifest_digest(sources: list[dict[str, Any]]) -> str:
    """Content-addressed identity of the whole manifest — a single value that changes
    whenever ANY source's bytes, size, presence or release window changes."""
    body = json.dumps(sources, sort_keys=True, separators=(",", ":"), default=str)
    return _SHA256_PREFIX + hashlib.sha256(body.encode("utf-8")).hexdigest()


def fingerprint_digest(counts: dict[str, Any], manifest: dict[str, Any]) -> str:
    """The single canonical aggregate release fingerprint: the code counts bound to the
    manifest's own digest and to every source record.

    Defined ONCE here so the producer (`data_fingerprint`) and the release validator
    (`pipeline._fingerprint_schema_ok`) compute it the same way -- a validator that only
    pattern-matched the digest's SHAPE accepted an arbitrary all-zero / all-`f` / merely
    plausible hex string as the identity of the data. (Codex F6-R5.)"""
    return manifest_digest(
        [{"counts": counts, "manifest": manifest.get("manifest_sha256")}]
        + list(manifest.get("sources") or []))


def build_manifest() -> dict[str, Any]:
    """Probe every authoritative source the coder depends on. Required sources gate
    release (fail closed); optional sources are recorded so their absence is visible
    rather than silently degrading retrieval. Every present source is content-addressed
    and version-aware so the release certificate identifies the exact bytes used."""
    from app.core import config

    from app.release.source_manifest import (
        REQUIRED_SOURCE_SCHEMA_VERSION, required_release_sources)

    codes = config.CODES_DIR
    # REQUIRED sources are NOT enumerated here. They are the versioned release-source
    # declaration (identity + role + release-metadata expectation) in
    # `app.release.source_manifest`, resolved to that registry's own paths, so the
    # capability manifest and the release-fingerprint validator compare against the SAME
    # complete set and cannot drift apart. A declaration that disagrees with the registry
    # raises: `build_manifest` fails loudly and the source-manifest gate holds, rather
    # than a manifest quietly omitting a claim-affecting input. (Codex F6-R5.)
    required_spec = required_release_sources()
    checks: list[tuple[Any, bool, str, str | None]] = [
        (spec["path"], True, spec["role"], source_id)
        for source_id, spec in required_spec.items()
    ]
    # OPTIONAL recall/precision aids: their absence degrades quality, not correctness,
    # so they are recorded (visible) but never gate release. Identity comes from the
    # registry path map, falling back to the filename stem.
    optional: list[tuple[Any, str]] = [
        (config.LCD_FILE, "local coverage policy"),
        (codes / "icd10cm_index_terms.json", "official alphabetic index terms"),
        (codes / "icd10cm_instructional_notes.json", "tabular exclusion notes"),
        (codes / "cpt_synonyms.json", "synonym recall aid"),
        (codes / "hcpcs_synonyms.json", "synonym recall aid"),
        (codes / "icd10_synonyms.json", "synonym recall aid"),
        (codes / "snomed_icd10_map.json", "concept crosswalk"),
        (codes / "cpt_index_terms.json", "procedure descriptor index"),
        (codes / "learned_cpt_index.json", "learned resolution index"),
        (codes / "hcpcs_drug_table.json", "drug dosing table"),
    ]
    checks += [(path, False, role, None) for path, role in optional]

    registry = _registry()
    locks = _source_locks()
    integrity_errors = [f"source lock {name.split(':', 1)[1]} is unreadable"
                        for name in locks if name.startswith("unreadable:")]
    sources: list[dict[str, Any]] = []
    for p, req, role, sid in checks:
        record, errors = _probe(p, req, role, registry, locks, sid)
        sources.append(record)
        integrity_errors.extend(errors)
    missing_required = [s["source"] for s in sources if s["required"] and not s["present"]]
    degraded_optional = [s["source"] for s in sources
                         if not s["required"] and not s["present"]]
    # Deliberately NO build timestamp: the manifest is a pure function of the authoritative
    # content, so two runs over unchanged data produce the same manifest and therefore the
    # same release certificate. A per-call `generated_at` would silently make certificates
    # irreproducible (the certificate carries no other clock).
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        # Which required-source DEFINITION this manifest was built against, so a
        # certificate produced under a different required set is identifiable instead
        # of silently comparable.
        "required_sources_schema": REQUIRED_SOURCE_SCHEMA_VERSION,
        "sources": sources,
        "missing_required": missing_required,
        "degraded_optional": degraded_optional,
        "integrity_errors": integrity_errors,
        "status": "BLOCKED" if (missing_required or integrity_errors) else "OK",
    }
    manifest["manifest_sha256"] = manifest_digest(sources)
    return manifest
