"""The release certificate — the defensibility artifact.

Binds the exact released claim to everything it depends on: the source note
(by hash), the date of service, each billed line with its verbatim evidence and
the authoritative record that defines it, the outcome of every gate, and the
verdict — then content-addresses the whole packet with a SHA-256. Re-running the
same inputs reproduces the same certificate hash; changing the note, a code, an
evidence span, a gate outcome, or the source edition invalidates it.

This is what makes an autonomous claim answerable after the fact: for any billed
line you can show the note text that supported it, the descriptor and edition it
came from, how it was chosen, and that it passed every control — and prove the
submitted claim is byte-for-byte the one that passed. (An HMAC signature with a
private key would add non-repudiation on top of this integrity hash.)
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import CodingResult


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def build_certificate(result: CodingResult, note_text: str,
                      source_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    lines = [{
        "system": ln.chosen.system,
        "code": ln.chosen.code,
        "descriptor": ln.chosen.descriptor,
        "method": ln.method.value,
        "evidence": [s.text for s in ln.fact.evidence],
        "authority": ln.chosen.authority,
    } for ln in result.billable_lines]

    payload: dict[str, Any] = {
        "encounter_id": result.encounter_id,
        "date_of_service": result.date_of_service,
        "note_sha256": _sha(note_text),
        "lines": lines,
        "gates": [{"name": g.name, "outcome": g.outcome.value} for g in result.gates],
        "verdict": result.verdict.value,
        "source_identity": source_identity or {},
    }
    payload["certificate_sha256"] = _sha(_canonical(payload))
    return payload
