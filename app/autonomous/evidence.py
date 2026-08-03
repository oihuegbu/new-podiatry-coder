"""Immutable evidence packet shared by every independent coding pass.

Model diversity is useful only when the models judge the same clinical facts.
The legacy pipeline independently re-extracted the PDF for every coding run,
which mixed extraction variance with coding variance.  ``EvidencePacket`` is
the process boundary: extraction, terminology normalization, fact compilation,
and authoritative retrieval happen once; all coders receive the exact same
canonical JSON bytes and fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class EvidencePacketError(ValueError):
    """The immutable packet is malformed or its fingerprint no longer matches."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str,
    )


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidencePacket:
    """A deeply immutable, process-safe coding input.

    Only the canonical JSON string is retained, rather than a nominally frozen
    object containing mutable nested dictionaries.  ``payload()`` returns a
    fresh copy and verifies the fingerprint every time it crosses a boundary.
    """

    schema_version: int
    document_id: str
    source_document_sha256: str
    evidence_fingerprint: str
    canonical_payload: str

    @classmethod
    def create(cls, *, document_id: str, source_document_sha256: str,
               payload: dict[str, Any]) -> "EvidencePacket":
        if (not document_id.strip() or document_id in {".", ".."}
                or Path(document_id).name != document_id):
            raise EvidencePacketError(
                "document_id must be a non-empty basename")
        if not source_document_sha256.startswith("sha256:"):
            raise EvidencePacketError(
                "source_document_sha256 must be a sha256 fingerprint")
        canonical = _canonical_json(payload)
        return cls(
            schema_version=1,
            document_id=document_id,
            source_document_sha256=source_document_sha256,
            evidence_fingerprint=_sha256(canonical),
            canonical_payload=canonical,
        )

    def payload(self) -> dict[str, Any]:
        if _sha256(self.canonical_payload) != self.evidence_fingerprint:
            raise EvidencePacketError("evidence packet fingerprint mismatch")
        try:
            value = json.loads(self.canonical_payload)
        except json.JSONDecodeError as exc:
            raise EvidencePacketError("evidence packet is not valid JSON") from exc
        if not isinstance(value, dict):
            raise EvidencePacketError("evidence packet payload must be an object")
        return value

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "source_document_sha256": self.source_document_sha256,
            "evidence_fingerprint": self.evidence_fingerprint,
            "artifact_name": f"{self.document_id}_evidence.json",
        }

    def persist(self, directory: str | Path) -> Path:
        """Atomically retain the full packet once for audit reconstruction."""
        from app.validation.run_store import atomic_write_json
        target = Path(directory) / self.manifest()["artifact_name"]
        atomic_write_json(target, {
            "manifest": self.manifest(),
            "payload": self.payload(),
        })
        return target

    @classmethod
    def load(cls, path: str | Path) -> "EvidencePacket":
        try:
            body = json.loads(Path(path).read_text())
            manifest = body["manifest"]
            packet = cls.create(
                document_id=str(manifest["document_id"]),
                source_document_sha256=str(
                    manifest["source_document_sha256"]),
                payload=body["payload"],
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise EvidencePacketError("evidence artifact is malformed") from exc
        if (packet.schema_version != manifest.get("schema_version")
                or packet.evidence_fingerprint != manifest.get(
                    "evidence_fingerprint")):
            raise EvidencePacketError("evidence artifact fingerprint mismatch")
        return packet
