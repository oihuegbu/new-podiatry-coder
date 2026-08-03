#!/usr/bin/env python3
"""Independently corroborate every generated retrieval mapping.

The input is an inert candidate pack. Two or more explicitly authorized
provider domains independently compare each term with the exact authoritative
descriptor. Only unanimous approvals are emitted. The output remains inert
until its exact bytes are enabled in the versioned lexicon catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import BASE_DIR
from app.rag.retrieval_lexicon import mapping_key


_DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decisions"],
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["mapping_id", "decision", "reason"],
                "properties": {
                    "mapping_id": {"type": "string"},
                    "decision": {"type": "string",
                                 "enum": ["approved", "rejected", "abstain"]},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _norm_code(value) -> str:
    return "".join(char for char in str(value or "").upper()
                   if char.isalnum())


def _descriptor_map(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        for key in ("codes", "data", "items", "results", "entries"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("authoritative code source has no record list")
    output = {}
    for row in data:
        if not isinstance(row, dict) or not row.get("code"):
            continue
        descriptor = (row.get("long_description") or row.get("description")
                      or row.get("short_description") or "")
        if str(descriptor).strip():
            output[_norm_code(row["code"])] = str(descriptor).strip()
    if not output:
        raise ValueError("authoritative code source has no descriptors")
    return output


def _judge_batch(profile, mappings: list[dict]) -> dict[str, dict]:
    from app.core.llm_client import chat_completion
    from app.core.model_profiles import use_execution_profile
    system = (
        "You independently review retrieval-only terminology. For each mapping, "
        "approve only when the term is a clinically equivalent or commonly used "
        "name for the complete authoritative descriptor without adding, removing, "
        "or contradicting material anatomy, laterality, etiology, severity, "
        "temporality, encounter, technique, formulation, or quantity. Reject a "
        "broader, narrower, merely associated, or context-dependent mapping. "
        "Abstain when the supplied descriptor is insufficient. A decision affects "
        "retrieval only and must never infer a billable code.")
    user = json.dumps({"mappings": mappings}, ensure_ascii=False)
    with use_execution_profile(profile):
        text, _usage = chat_completion(
            system, user, model=profile.model, temperature=0.0,
            max_tokens=4096, json_mode=True, json_schema=_DECISION_SCHEMA)
    payload = json.loads(text)
    expected = {row["mapping_id"] for row in mappings}
    decisions = {}
    for row in payload.get("decisions") or []:
        mapping_id = str(row.get("mapping_id") or "")
        if mapping_id in expected and mapping_id not in decisions:
            decisions[mapping_id] = {
                "decision": row["decision"],
                "reason": str(row.get("reason") or ""),
            }
    return decisions


def corroborate(*, candidate_path: Path, code_source: Path,
                profiles: list, batch_size: int = 30,
                limit: int | None = None) -> dict:
    if batch_size < 1 or (limit is not None and limit < 0):
        raise ValueError("batch size and mapping limit must be non-negative")
    candidate = json.loads(candidate_path.read_text())
    terms = candidate.get("terms") if isinstance(candidate, dict) else None
    if (candidate.get("authority_role") != "retrieval_only"
            or candidate.get("provenance_kind") != "generated"
            or not str(candidate.get("pack_id") or "").strip()
            or not str(candidate.get("code_system") or "").strip()
            or not isinstance(terms, dict)):
        raise ValueError("input is not a generated retrieval-only candidate pack")
    if candidate.get("code_source_sha256") != _sha256(code_source):
        raise ValueError("candidate is not bound to the live authoritative source")
    invalid_profiles = [
        profile.profile_id for profile in profiles
        if not str(profile.profile_id).strip()
        or not str(profile.model).strip()
        or not str(profile.provider).strip()
        or str(profile.provider).strip().lower()
        != str(profile.independence_domain).strip().lower()
    ]
    if invalid_profiles:
        raise ValueError("corroboration profiles have invalid independence identity")
    domains = {str(profile.independence_domain).strip().lower()
               for profile in profiles}
    if len(domains) < 2:
        raise ValueError("corroboration requires at least two provider domains")
    descriptors = _descriptor_map(code_source)
    mappings = []
    mapping_ids = set()
    for raw_code, values in terms.items():
        code = _norm_code(raw_code)
        descriptor = descriptors.get(code)
        if not descriptor:
            raise ValueError("candidate references a code without a descriptor")
        if not isinstance(values, list):
            raise ValueError("candidate term mapping is not a list")
        for term in values:
            if not isinstance(term, str) or not term.strip():
                raise ValueError("candidate contains a malformed term")
            identifier = mapping_key(code, term)
            if identifier in mapping_ids:
                raise ValueError("candidate contains a duplicate mapping")
            mapping_ids.add(identifier)
            mappings.append({"mapping_id": identifier,
                             "code": code, "term": term.strip(),
                             "authoritative_descriptor": descriptor})
    total = len(mappings)
    evaluated = mappings[:limit] if limit is not None else mappings
    all_attestations: dict[str, list[dict]] = {
        row["mapping_id"]: [] for row in evaluated}
    for start in range(0, len(evaluated), batch_size):
        batch = evaluated[start:start + batch_size]
        for profile in profiles:
            decisions = _judge_batch(profile, batch)
            for row in batch:
                decision = decisions.get(row["mapping_id"], {
                    "decision": "abstain", "reason": "response omitted mapping"})
                all_attestations[row["mapping_id"]].append({
                    "profile_id": profile.profile_id,
                    "provider": profile.provider,
                    "model": profile.model,
                    "independence_domain": profile.independence_domain,
                    **decision,
                })
    accepted_terms: dict[str, list[str]] = {}
    accepted_attestations = {}
    rejected_fingerprints = []
    by_id = {row["mapping_id"]: row for row in evaluated}
    for mapping_id, attestations in all_attestations.items():
        approved_domains = {
            row["independence_domain"] for row in attestations
            if row["decision"] == "approved"}
        unanimous = (len(approved_domains) == len(domains)
                     and all(row["decision"] == "approved"
                             for row in attestations))
        if not unanimous:
            rejected_fingerprints.append(mapping_id)
            continue
        row = by_id[mapping_id]
        accepted_terms.setdefault(row["code"], []).append(row["term"])
        accepted_attestations[mapping_id] = attestations
    complete = len(evaluated) == total
    return {
        "schema_version": 1,
        "pack_id": str(candidate.get("pack_id") or "") + "+corroborated",
        "code_system": str(candidate.get("code_system") or ""),
        "authority_role": "retrieval_only",
        "provenance_kind": "generated_corroborated",
        "source_candidate_sha256": _sha256(candidate_path),
        "code_source_sha256": _sha256(code_source),
        "complete": complete,
        "candidate_mapping_count": total,
        "evaluated_mapping_count": len(evaluated),
        "accepted_mapping_count": sum(map(len, accepted_terms.values())),
        "rejected_mapping_count": len(rejected_fingerprints),
        "rejected_mapping_fingerprints": sorted(rejected_fingerprints),
        "corroboration_profiles": [profile.model_dump() for profile in profiles],
        "count": len(accepted_terms),
        "mapping_attestations": accepted_attestations,
        "terms": dict(sorted(accepted_terms.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--code-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    from app.core.model_profiles import configured_profiles
    profiles = configured_profiles(require_credentials=True)
    payload = corroborate(
        candidate_path=Path(args.candidate),
        code_source=Path(args.code_source), profiles=profiles,
        batch_size=args.batch_size, limit=args.limit)
    output = Path(args.output).resolve()
    if not output.is_relative_to(BASE_DIR.resolve()):
        raise ValueError("output must remain inside the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, output)
    print(json.dumps({
        "output": str(output.relative_to(BASE_DIR)),
        "complete": payload["complete"],
        "evaluated": payload["evaluated_mapping_count"],
        "accepted": payload["accepted_mapping_count"],
        "rejected": payload["rejected_mapping_count"],
        "pack_sha256": _sha256(output),
        "next_step": "add exact output binding to the versioned catalog",
    }, indent=2))
    return 0 if payload["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
