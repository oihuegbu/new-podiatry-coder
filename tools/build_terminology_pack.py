#!/usr/bin/env python3
"""Derive a high-precision terminology pack from authoritative corpora.

Only explicit parenthetical definitions (``long form (ABC)`` or
``ABC (long form)``) are accepted, and the alias must match the long-form
initialism after the catalog's versioned ignore-word rules are applied.  The
tool never derives or stores a medical code mapping.  Generated expansions
remain retrieval-only and cannot serve as claim evidence or coding authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import (BASE_DIR, TERMINOLOGY_PACK_DIR,
                             TERMINOLOGY_SOURCE_CATALOG_FILE)


_ALIAS = r"[A-Z][A-Z0-9/-]{1,9}"
_WORD = r"[A-Za-z][A-Za-z'-]*"
_LONG = rf"{_WORD}(?:\s+{_WORD}){{1,11}}"
_LONG_ALIAS = re.compile(rf"(?P<long>{_LONG})\s*\((?P<alias>{_ALIAS})\)")
_ALIAS_LONG = re.compile(rf"(?P<alias>{_ALIAS})\s*\((?P<long>{_LONG})\)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _initialism(long_form: str, ignored: set[str]) -> str:
    words = re.findall(_WORD, long_form)
    return "".join(word[0].upper() for word in words
                   if word.casefold() not in ignored)


def _clean_long_form(value: str, ignored: set[str]) -> str:
    words = value.strip(" ,;:-").split()
    while words and words[0].casefold().strip("'\"") in ignored:
        words.pop(0)
    return " ".join(words)


def _pairs(text: str, *, ignored: set[str], min_words: int,
           max_words: int, min_alias: int, max_alias: int):
    seen = set()
    for pattern in (_LONG_ALIAS, _ALIAS_LONG):
        for match in pattern.finditer(text):
            alias = match.group("alias").strip().upper()
            long_form = _clean_long_form(match.group("long"), ignored)
            words = re.findall(_WORD, long_form)
            if not (min_words <= len(words) <= max_words
                    and min_alias <= len(alias) <= max_alias):
                continue
            normalized_alias = re.sub(r"[^A-Z0-9]", "", alias)
            if normalized_alias != _initialism(long_form, ignored):
                continue
            key = (alias, long_form.casefold())
            if key not in seen:
                seen.add(key)
                yield alias, long_form


def build_pack(catalog_path: Path = TERMINOLOGY_SOURCE_CATALOG_FILE) -> dict:
    catalog = json.loads(catalog_path.read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError("unsupported terminology source catalog schema")
    derivation = catalog.get("derivation") or {}
    if derivation.get("method") != "explicit_parenthetical_alias_only":
        raise ValueError("unsupported terminology derivation method")
    ignored = {str(value).casefold() for value in
               derivation.get("initialism_ignore_words") or []}
    limits = {
        "min_words": int(derivation["minimum_long_form_words"]),
        "max_words": int(derivation["maximum_long_form_words"]),
        "min_alias": int(derivation["minimum_alias_characters"]),
        "max_alias": int(derivation["maximum_alias_characters"]),
    }
    confidence = float(derivation["candidate_confidence"])
    if not 0.9 <= confidence <= 1.0:
        raise ValueError("derived terminology confidence must be high")

    source_rows, candidates = {}, {}
    inputs = []
    allowed_authorities = {"licensed_primary", "government_primary"}
    for source in catalog.get("sources") or []:
        source_id = str(source.get("id") or "")
        if not source_id or source_id in source_rows:
            raise ValueError("terminology catalog source ids must be unique")
        authority = str(source.get("authority") or "")
        if authority not in allowed_authorities:
            raise ValueError(
                f"{source_id}: source authority must be licensed/government primary")
        paths = [(BASE_DIR / str(value)).resolve()
                 for value in source.get("paths") or []]
        if not paths:
            raise ValueError(f"{source_id}: no source paths")
        hashes = []
        for path in paths:
            if not path.is_relative_to(BASE_DIR.resolve()):
                raise ValueError(f"{source_id}: source path escapes repository")
            if not path.is_file():
                raise FileNotFoundError(f"{source_id}: missing {path}")
            digest = _sha256(path)
            hashes.append(digest)
            inputs.append({"source_id": source_id,
                           "path": str(path.relative_to(BASE_DIR)),
                           "sha256": digest})
            text = path.read_text(encoding="utf-8", errors="replace")
            for alias, long_form in _pairs(text, ignored=ignored, **limits):
                candidates.setdefault(alias, {}).setdefault(
                    long_form, set()).add(source_id)
        source_rows[source_id] = {
            "name": str(source.get("publisher") or source_id),
            "version": "+".join(hashes),
            "review_status": "publisher-attested-explicit-definition",
            "scope": "Explicit parenthetical aliases extracted from pinned corpus bytes",
            "authority_role": "retrieval_only",
            "provenance_kind": authority,
        }

    entries = []
    for alias in sorted(candidates):
        rows = []
        for expansion in sorted(candidates[alias], key=str.casefold):
            source_ids = sorted(candidates[alias][expansion])
            rows.append({"expansion": expansion,
                         "confidence": confidence,
                         "source_id": source_ids[0],
                         "corroborating_source_ids": source_ids[1:],
                         "attestation": "explicit_parenthetical_definition"})
        entry_id = "zz_derived_" + hashlib.sha256(alias.encode()).hexdigest()[:16]
        entries.append({"id": entry_id,
                        "patterns": [rf"\b{re.escape(alias)}\b"],
                        "coding_impact": True,
                        "candidates": rows})
    fingerprint = "sha256:" + hashlib.sha256(json.dumps(
        inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema_version": 1,
            "version": str(catalog.get("version") or "") + "+derived",
            "input_fingerprint": fingerprint,
            "sources": source_rows,
            "entries": entries}


def materialize_pack(output: Path | None = None, *,
                     if_sources_changed: bool = True) -> dict:
    output = output or TERMINOLOGY_PACK_DIR / "derived_authoritative.json"
    pack = build_pack()
    if if_sources_changed and output.exists():
        try:
            # Compare the complete deterministic product, not its self-stated
            # input fingerprint. A tampered output can retain that field; it
            # must be repaired automatically on the next startup.
            if json.loads(output.read_text()) == pack:
                return {"changed": False, "path": str(output),
                        "aliases": len(pack["entries"]),
                        "input_fingerprint": pack["input_fingerprint"]}
        except (OSError, ValueError, TypeError):
            pass
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, output)
    return {"changed": True, "path": str(output),
            "aliases": len(pack["entries"]),
            "input_fingerprint": pack["input_fingerprint"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(
        TERMINOLOGY_PACK_DIR / "derived_authoritative.json"))
    parser.add_argument("--if-sources-changed", action="store_true")
    args = parser.parse_args()
    result = materialize_pack(Path(args.output),
                              if_sources_changed=args.if_sources_changed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
