#!/usr/bin/env python3
"""Canonicalize an external retrieval lexicon as an inert candidate pack.

Import never activates a pack.  The governed catalog and independent
mapping attestations control activation; this tool only validates structure,
binds the candidate to an authoritative code source, and writes deterministic
bytes suitable for review and later corroboration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import BASE_DIR


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _normalize_code(value) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _normalize_term(value) -> str:
    return " ".join(unicodedata.normalize(
        "NFKC", str(value or "")).split())


def _source_codes(path: Path) -> set[str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        for key in ("codes", "data", "items", "results", "entries"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("authoritative source has no code record list")
    codes = {_normalize_code(row.get("code")) for row in data
             if isinstance(row, dict) and row.get("code")}
    if not codes:
        raise ValueError("authoritative source has no code identities")
    return codes


def canonicalize(*, source: Path, code_source: Path, pack_id: str,
                 code_system: str) -> dict:
    raw = json.loads(source.read_text())
    terms = raw.get("terms") if isinstance(raw, dict) else None
    if not isinstance(terms, dict):
        raise ValueError("candidate source has no terms mapping")
    authoritative = _source_codes(code_source)
    output: dict[str, list[str]] = {}
    for raw_code, raw_terms in terms.items():
        code = _normalize_code(raw_code)
        if code not in authoritative:
            raise ValueError(f"candidate contains an unknown code identity: {raw_code}")
        if not isinstance(raw_terms, list):
            raise ValueError(f"candidate terms for {raw_code} are not a list")
        seen, values = set(), []
        for raw_term in raw_terms:
            if not isinstance(raw_term, str):
                raise ValueError(f"candidate term for {raw_code} is not text")
            term = _normalize_term(raw_term)
            key = term.casefold()
            if term and key not in seen:
                seen.add(key)
                values.append(term)
        if values:
            output[code] = values
    return {
        "schema_version": 1,
        "pack_id": pack_id,
        "code_system": code_system.strip().lower(),
        "authority_role": "retrieval_only",
        "provenance_kind": "generated",
        "source_candidate_sha256": _sha256(source),
        "code_source_sha256": _sha256(code_source),
        "generated": str(raw.get("generated") or ""),
        "source_provenance": str(raw.get("provenance") or ""),
        "count": len(output),
        "mapping_attestations": {},
        "terms": dict(sorted(output.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--code-source", required=True)
    parser.add_argument("--pack-id", required=True)
    parser.add_argument("--code-system", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    code_source = Path(args.code_source).resolve()
    output = Path(args.output).resolve()
    if output.is_relative_to(BASE_DIR.resolve()) is False:
        raise ValueError("candidate output must remain inside the repository")
    payload = canonicalize(
        source=source, code_source=code_source, pack_id=args.pack_id,
        code_system=args.code_system)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, output)
    print(json.dumps({
        "path": str(output.relative_to(BASE_DIR)),
        "count": payload["count"],
        "pack_sha256": _sha256(output),
        "code_source_sha256": payload["code_source_sha256"],
        "status": "candidate",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
